"""릴리즈 V-JEPA 2 predictor vs 블록 AR predictor — attention 분포와 문맥 encoder 양방향 섞임 (2026-11-06, 릴리즈 → AR 사다리).

학습 없이 잰다. VOBench v1 permanence (가시 · early · mid · late 가림) 의 pos_a (물체가 끝까지 있는 판) 와 pos_b (빈 판).

(A) attention 시간 분포
    예측 블록 f (8..15) 를 내는 query 가 각 key 블록 t 에 준 attention 질량 (층 · 헤드 · query 평균, 행 합 1).
      릴리즈: query = 블록 f 자리의 mask token, key = 문맥 0..7 + 미래 mask token 8..15 (전부 양방향).
      AR    : query = 걸음 f-1 의 블록 토큰 (출력 = 블록 f 예측), key = 블록 0..f-1 (문맥 관측 0..7 + 되먹인 상태 8..f-1).
(B) 물체 자리 attention
    예측 블록 f 에서 물체가 보이는 칸 3×3 의 query 만. key 블록 t 의 물체 자리 3×3 (그 블록 시점의 실제 위치) 에 준 질량 / 균등 기대값.
(C) 문맥 encoder 양방향 섞임
    같은 encoder 가중치로 문맥 16 장을 (window = 한 번에 · causal = 블록 t 까지만 · block = 블록 하나씩) 인코딩.
    물체 대비 OC_t = 물체 3×3 의 |LN z(pos_a) − LN z(pos_b)| 평균 / 물체에서 먼 배경 칸의 같은 값.
    window 와 causal 의 차이 = 블록 t 뒤 프레임 (가림 사건 포함) 이 블록 t 토큰에 섞인 몫.

attention weight 는 SDPA 가 만들지 않으므로 torch.nn.functional.scaled_dot_product_attention 을 predictor 호출 동안만
감싸 q · k 로 다시 계산한다 (출력은 원래 SDPA 값 그대로). 첫 호출에서 재계산 p@v 와 실제 출력을 대조해 verify 에 남긴다.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, ROOT)

from analysis.intphys2.model import build_from_config, _ArBlockEncoder, _CausalCtxEncoder  # noqa: E402
from analysis.intphys2.surprise import _context_target_indices  # noqa: E402
from evals.world_model_analysis.data import WMADataset  # noqa: E402

RES = os.path.join(ROOT, "cvpr/results/eval/testbed")
CFGS = {
    "release": f"{RES}/vobench_v1_late_vith/_resolved.yaml",
    "ar": f"{RES}/vobench_v1_late_vast_ar_vw_state/_resolved.yaml",
    "release_causal": f"{RES}/vobench_v1_late_vith/_resolved.yaml",     # 사다리 R1: 같은 릴리즈, 문맥만 인과 인코딩
    # 2026-11-07 (사용자 "인접 블록만 보는 게 RoPE 때문 확실해? 학습 데이터 때문일 가능성은?"): 같은 구조 (RoPE 포함), 학습 데이터만 다른 AR 판
    "ar_v4bc4": f"{RES}/vobench_v1_late_vast_ar_vw_state/_resolved.yaml",
    "ar_v4bc4_neg": f"{RES}/vobench_v1_late_vast_ar_vw_state/_resolved.yaml",
}
OVERRIDE = {"release_causal": {"ctx_encoding": "causal"},
            "ar_v4bc4": {"predictor_checkpoint": os.path.join(ROOT, "z_training/runs/v4bc4_ar_vw_state_postft/latest.pt")},       # v4bc4 · 예측 L1 만
            "ar_v4bc4_neg": {"predictor_checkpoint": os.path.join(ROOT, "z_training/runs/v4bc4_ar_vw_state_negimp/latest.pt")}}   # v4bc4 · 음의 손실 λ1
VOB1 = "/workspace/dataset/benchmarks/VOBench_v1"
S, GW, NB, C = 256, 16, 16, 8          # 블록당 토큰, 격자 폭, 블록 수 (32 장 / tubelet 2), 문맥 블록
L, H = 12, 12


# ---------------------------------------------------------------- 물체 위치
def obj_track(file_name: str):
    """블록별 (가시 비율, 칸 행, 칸 열). 원본 512 px, 표본 = 원본 0, 3, …, 93, 블록 b = 표본 2b, 2b+1."""
    lab = json.load(open(os.path.join(VOB1, os.path.dirname(file_name), "label.json")))["labels"]["A"]
    by_f = {e["f"]: e for e in lab}
    vis, rc = [], []
    for b in range(NB):
        es = [by_f[3 * (2 * b + j)] for j in range(2)]
        vis.append(float(np.mean([1.0 - e["hidden"] for e in es])))
        u = np.mean([e["u"] for e in es]) / 512 * 256
        v = np.mean([e["v"] for e in es]) / 512 * 256
        rc.append((int(np.clip(v // 16, 0, GW - 1)), int(np.clip(u // 16, 0, GW - 1))))
    return np.array(vis), rc


def region(rc, r=1):
    """칸 (행, 열) 둘레 (2r+1)² 의 블록 안 토큰 인덱스."""
    i, j = rc
    return [a * GW + b for a in range(max(0, i - r), min(GW, i + r + 1)) for b in range(max(0, j - r), min(GW, j + r + 1))]


def far_bg(rcs, r=3):
    """모든 블록의 물체 위치에서 r 칸 넘게 떨어진 칸."""
    out = []
    for t in range(S):
        a, b = divmod(t, GW)
        if all(max(abs(a - i), abs(b - j)) > r for i, j in rcs):
            out.append(t)
    return out


# ---------------------------------------------------------------- attention 기록
class Rec:
    def __init__(self):
        self.on = False
        self.calls = 0
        self.ctx = None          # 지금 배치의 물체 정보: list of (obj_q[f] tokens, obj_k[t] tokens)
        self.groups = None
        self.mode = None         # release | ar
        self.mass = defaultdict(lambda: np.zeros((L, H, NB, NB)))     # [그룹] (층, 헤드, 예측블록 f, key 블록 t)
        self.nq = defaultdict(lambda: np.zeros(NB))                    # 예측 블록별 클립 수
        self.omass = defaultdict(lambda: np.zeros((L, NB, NB)))       # 물체 query → key 블록 t 물체 자리 질량 (헤드 평균)
        self.obmass = defaultdict(lambda: np.zeros((L, NB, NB)))      # 물체 query → key 블록 t 전체 질량
        self.on_ = defaultdict(lambda: np.zeros(NB))
        self.verify = {}


REC = Rec()
_ORIG_SDPA = F.scaled_dot_product_attention


def _sdpa_rec(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
    out = _ORIG_SDPA(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)
    if not REC.on:
        return out
    if attn_mask is not None or is_causal:
        raise RuntimeError("마스크 있는 SDPA 는 이 분석이 다루지 않는다")
    li = REC.calls % L
    step = REC.calls // L
    REC.calls += 1
    B, Hh, Nq, D = q.shape
    Nk = k.size(2)
    sc = scale if scale is not None else D ** -0.5
    kT = k.float().transpose(-1, -2)
    if REC.mode.startswith("release"):
        assert Nq == Nk == NB * S, (Nq, Nk)
        q_blocks = list(range(NB))            # query 블록 = 예측 블록
        q0 = 0
    else:
        assert Nq == S and Nk == (step + 1) * S, (Nq, Nk, step)
        q_blocks = [step + 1]                 # 걸음 step 의 출력 = 블록 step+1 예측
        q0 = None
    for qb_i, f in enumerate(q_blocks):
        if f >= NB:
            continue
        a = qb_i * S if REC.mode.startswith("release") else 0
        p = (torch.matmul(q[:, :, a:a + S].float(), kT) * sc).softmax(-1)       # (B, H, S, Nk)
        if REC.mode not in REC.verify and li == 0:
            o = torch.matmul(p, v.float())
            REC.verify[REC.mode] = {"max_abs_diff": float((o - out[:, :, a:a + S].float()).abs().max()),
                          "mean_abs_out": float(out.float().abs().mean())}
        nkb = Nk // S
        pb = p.view(B, Hh, S, nkb, S)
        m = pb.sum(-1).mean(2)                                                    # (B, H, nkb)
        for bi in range(B):
            g = REC.groups[bi]
            REC.mass[g][li, :, f, :nkb] += m[bi].double().cpu().numpy()
            if li == 0:
                REC.nq[g][f] += 1
            oq, ok = REC.ctx[bi]
            if oq[f] is None:
                continue
            pq = pb[bi][:, oq[f]].mean(0).mean(0)                                 # (nkb, S) 헤드 · 물체 query 평균
            for t in range(nkb):
                REC.omass[g][li, f, t] += float(pq[t, ok[t]].sum())
                REC.obmass[g][li, f, t] += float(pq[t].sum())
            if li == 0:
                REC.on_[g][f] += 1
    return out


@contextlib.contextmanager
def recording(mode, groups, ctx):
    REC.on, REC.calls, REC.mode, REC.groups, REC.ctx = True, 0, mode, groups, ctx
    F.scaled_dot_product_attention = _sdpa_rec
    try:
        yield
    finally:
        F.scaled_dot_product_attention = _ORIG_SDPA
        REC.on = False


# ---------------------------------------------------------------- 메인
def load_cfg(name):
    cfg = yaml.safe_load(open(CFGS[name]))
    cfg["data"]["root"] = os.path.join(ROOT, "data_csv/vobench_v1")
    cfg["model"].update(OVERRIDE.get(name, {}))
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "z_analysis/results/release_ladder_20261106/attn_release_vs_ar.json"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--models", default="release,ar")
    ap.add_argument("--skip-enc", action="store_true", help="(C) encoder 섞임 측정을 건너뛴다")
    args = ap.parse_args()
    dev = torch.device("cuda:0")
    torch.backends.cuda.enable_flash_sdp(True)

    rows = {r["video_id"]: r for r in csv.DictReader(open(os.path.join(ROOT, "data_csv/vobench_v1/index.csv")))}
    cfg0 = load_cfg("release")
    ds = WMADataset(cfg0)
    pick = [i for i, x in enumerate(ds.records)
            if rows[x.video_id]["violation_type"] == "permanence" and rows[x.video_id]["variant"] in ("pos_a", "pos_b")]
    by_block = defaultdict(dict)
    for i in pick:
        r = rows[ds.records[i].video_id]
        by_block[r["block_id"]][r["variant"]] = i
    blocks = sorted(b for b, d in by_block.items() if len(d) == 2)
    if args.limit:
        blocks = blocks[:args.limit]
    meta = {}
    for b in blocks:
        r = rows[ds.records[by_block[b]["pos_a"]].video_id]
        vis, rc = obj_track(r["file_name"])
        mv = "static" if r["motion"] == "flat_static" else "moving"
        meta[b] = dict(group=f"{r['occlusion']}/{mv}", vis=vis, rc=rc, motion=r["motion"], k=r["k"])
    print(f"블록 {len(blocks)}: {dict(sorted(defaultdict(int, {g: sum(1 for m in meta.values() if m['group'] == g) for g in {m['group'] for m in meta.values()}}).items()))}", flush=True)

    def obj_sets(m):
        oq = [region(m["rc"][f]) if (f >= C and m["vis"][f] >= 0.99) else None for f in range(NB)]
        ok = [region(m["rc"][t]) for t in range(NB)]
        return oq, ok

    ci, ti = _context_target_indices(ctx_frames=16, tgt_frames=16, tubelet_size=2, spatial_tokens=S,
                                     batch_size=args.bs, device=dev)
    enc_out = {}                                                     # (C) 결과
    for name in args.models.split(","):
        cfg = load_cfg(name)
        t0 = time.time()
        bundle = build_from_config(cfg["model"], dev)
        print(f"[{name}] 모델 {time.time()-t0:.0f}s, ctx enc {type(bundle.context_encoder).__name__}", flush=True)
        for s in range(0, len(blocks), args.bs):
            bb = blocks[s:s + args.bs]
            x = torch.stack([ds.clip(by_block[b]["pos_a"]) for b in bb]).to(dev, bundle.dtype)
            n = len(bb)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                z = bundle.context_encoder(x, masks=[ci[:n]])
                with recording(name, [f"{name}|{meta[b]['group']}" for b in bb], [obj_sets(meta[b]) for b in bb]):
                    bundle.predictor(z, ci[:n], ti[:n], mask_index=0)
            if s == 0:
                print(f"[{name}] verify {REC.verify.get(name)}", flush=True)
        # (C) encoder 섞임 — 릴리즈 차례에만 (같은 encoder 가중치로 window / causal / block)
        if name == "release" and not args.skip_enc:
            enc = bundle.context_encoder
            encs = {"window": None, "causal": _CausalCtxEncoder(enc, 2).eval(), "block": _ArBlockEncoder(enc, 2).eval(),
                    "ar_input": _ArBlockEncoder(bundle.target_encoder, 2).eval()}
            ocs = {k: defaultdict(list) for k in encs}
            for s in range(0, len(blocks), args.bs):
                bb = blocks[s:s + args.bs]
                n = len(bb)
                za, zb = {}, {}
                for var, dst in (("pos_a", za), ("pos_b", zb)):
                    x = torch.stack([ds.clip(by_block[b][var]) for b in bb]).to(dev, bundle.dtype)
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                        for k, e in encs.items():
                            if e is None:
                                zz = enc(x, masks=[ci[:n]])
                            else:
                                zz = e(x[:, :, :16].contiguous())
                            dst[k] = F.layer_norm(zz.float(), (zz.size(-1),))
                for k in encs:
                    d = (za[k] - zb[k]).abs().mean(-1).view(n, C, S).cpu().numpy()   # (n, 8, 256)
                    for bi, b in enumerate(bb):
                        m = meta[b]
                        bg = far_bg(m["rc"][:C])
                        oc = [float(d[bi, t, region(m["rc"][t])].mean() / max(d[bi, t, bg].mean(), 1e-6)) for t in range(C)]
                        ocs[k][m["group"]].append(dict(oc=oc, vis=m["vis"][:C].tolist(), k=m["k"], motion=m["motion"]))
            enc_out = {k: dict(v) for k, v in ocs.items()}
        del bundle
        torch.cuda.empty_cache()

    res = {"verify": REC.verify, "groups": sorted(REC.mass), "attn": {}, "enc": enc_out,
           "blocks": {b: dict(group=m["group"], vis=m["vis"].tolist(), rc=m["rc"], motion=m["motion"], k=m["k"]) for b, m in meta.items()}}
    for g in REC.mass:
        res["attn"][g] = {"mass": (REC.mass[g] / np.maximum(REC.nq[g], 1)[None, None, :, None]).tolist(),
                          "omass": (REC.omass[g] / np.maximum(REC.on_[g], 1)[None, :, None]).tolist(),
                          "obmass": (REC.obmass[g] / np.maximum(REC.on_[g], 1)[None, :, None]).tolist(),
                          "nq": REC.nq[g].tolist(), "nobj": REC.on_[g].tolist()}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"))
    print("저장", args.out)


if __name__ == "__main__":
    main()
