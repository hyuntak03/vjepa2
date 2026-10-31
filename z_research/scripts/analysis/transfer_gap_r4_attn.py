#!/usr/bin/env python3
"""transfer_gap R4 E37 — 다시 나타날 자리의 미래 query 가 문맥에서 물체가 보이던 자리를 보나 (2026-10-30, 실험 A [X1]).

물음: late (문맥 끝 가림) 에서 물체의 국소 꼴이 다시 나타날 자리로 옮겨지지 않는 것은, 그 자리의 미래 query 가
과거에 물체가 보이던 자리를 안 보기 때문인가? PFT 가 그 경로를 켰다면 (B13), 학습 생성기 밖 (v11) 에서도 켜지나?
B13 코드 (z_research/scripts/analysis/ar_attention_history.py, 릴리즈 분기) 의 attention 가로채기를 그대로 쓰고,
(1) predictor 여럿을 encoder 한 번 위에서 (2) 대조를 '같은 거리 고리' 로 바꾸고 (3) 판 (VOBench 사본 · v11 사본) 을 바꿨다.

정의 (clip i · 미래 걸음 r (튜블릿 C+r) · 층 l, head 평균):
  query  = 튜블릿 C+r 의 참 물체 자리 3×3 (상자 중심은 1–14 로 자름). 그 튜블릿 두 장 모두 보이고 화면 안일 때만 (= 다시 나타난 뒤).
  key b  = 문맥 튜블릿 b (0..7) 중 두 장 모두 보이고 화면 안인 것, 그리고 물체 자리가 query 자리에서 체비쇼프 3 칸 이상.
  BS[b]  = 9 query 토큰 평균 attention 을 key 블록 b 의 16×16 위에 놓고 3×3 상자 합 (상자 중심 1–14, 항상 9 토큰).
  obj    = BS[b, 물체 자리 b]
  ring   = BS[b, c] 의 평균, c ∈ {중심 1–14, | ‖c − q‖ − ‖o_b − q‖ | ≤ 0.5 (유클리드), ‖c − o_b‖∞ ≥ 3}  (같은 거리 대조, 주)
  mir    = BS[b, 2q − o_b] (화면 안 · 물체 상자와 안 겹칠 때만)                                       (거울 대조, 보조)
  ql     = BS[b, q] (query 와 같은 좌표, B13 의 대조 2)                                                (참고)
  균등   = 9 / 4096 (key 16 블록 × 256)
  저장: 위 넷의 (K, L, NB) head 평균 배열 + 유효 표시 + 물체 query 의 블록 질량 bm (K, L, NB) + head 별 obj · ring 합 (K, L, H).
판:
  vobench — data_csv/vobench_v1/index.csv 의 late (+ 참고 visible) · 이동 · 가능 · 물체 있는 영상. 프레임 = 256² 사본 (--frames-root),
            궤적 = results/r4_x1/stage/vobench_tracks.npz (label.json 에서 뽑은 512 px 좌표 · 가림, 이 셸에서 만듦).
  v11     — data_csv/intphysgen_v11_split/index_test.csv 의 vanish pos_a · 가림 이동 (평지 · 경사) k 1–4. 프레임 = 원본 288² 그대로 옮긴 NFS 사본,
            궤적 = 원본 metadata.csv 의 object_px_{x,y}_by_sample (288 px; 토큰 = px // 18), 가림 = 문맥 끝 k 장 + 미래 앞 k 장 (설계 · 실측 n_context_hidden == k).
모델 (encoder = 릴리즈, 문맥 = online encoder, predictor 만 바꿈; 수치 = rp.MODEL 과 같은 bf16 — B13 과 같은 조건):
  release · vobftg (z_training/runs/vobench_release_postft_g/latest.pt) · v11pft (z_training/runs/v11_postft/e10.pt) · ip1late_f0 (z_training/runs/intphys1_late_uni_pft_f0/e40.pt)
⚠️ attention 질량 ≠ 정보 사용. 인과는 E38 (transfer_gap_r4_ko.py).

재현:
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  sbatch -p batch_vll -w vll4 --gres=gpu:2 --cpus-per-gpu=4 --mem-per-gpu=40G --time=0-02:00:00 -o z_analysis/transfer_gap/logs/%x_%j.out -J X1R4_E37vob \
     --wrap "$P z_research/scripts/analysis/transfer_gap_r4_attn.py run --set vobench --gpus 2 --frames-root z_pixel_decoder/runs/_frames/VOBench_v1_256full"
  ... run --set v11 --frames-root z_pixel_decoder/runs/_frames/IntPhysGen_v11_sub
  $P z_research/scripts/analysis/transfer_gap_r4_attn.py summarize      # CPU → z_analysis/transfer_gap/results/r4_x1/E37_attn.json
산출: /data/.../z_analysis/transfer_gap/results/r4_x1/E37/<set>_r<rank>.npz , E37_attn.json
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
S, NB, C, K, G, H_, L_ = 256, 16, 8, 8, 16, 12, 12
OUTR = REPO / "z_analysis/transfer_gap/results/r4_x1"
OUT = OUTR / "E37"
STAGE = OUTR / "stage"
MODELS = {"release": "", "vobftg": "z_training/runs/vobench_release_postft_g/latest.pt",
          "v11pft": "z_training/runs/v11_postft/e10.pt", "ip1late_f0": "z_training/runs/intphys1_late_uni_pft_f0/e40.pt",
          "ip2e80": "z_training/runs/intphys2_postft/e80.pt", "ip2pair": "z_training/runs/intphys2_postft_pair/latest.pt"}   # ip2* = E37b (계획 밖 추가)
V11META = Path("/local_datasets/world/world_analysis/IntPhysGen_v11/metadata.csv")
BAND = list(range(5, 12))                                   # 사전 등록 층 띠 L5–L11 (B13)


# ───────────────────────── 항목 (clip) 과 궤적
def items_vobench():
    T = np.load(STAGE / "vobench_tracks.npz", allow_pickle=True)
    tix = {v: i for i, v in enumerate(T["vid"])}
    sp = {}
    for s in ("train", "test"):
        for r in csv.DictReader(open(REPO / f"data_csv/vobench_v1_split_v2/index_{s}.csv")):
            sp[r["video_id"]] = s
    out = []
    for r in csv.DictReader(open(REPO / "data_csv/vobench_v1/index.csv")):
        if r["occlusion"] not in ("late", "visible") or r["motion"] == "flat_static" or r["variant"] not in ("pos_a", "pos_b"):
            continue
        if r["obj_a" if r["variant"] == "pos_a" else "obj_b"] == "empty":
            continue
        j = tix[r["video_id"]]
        if not T["ok"][j]:
            continue
        P, hid = T["P"][j], T["hid"][j]
        tok = np.stack([np.clip(P[:, 1] // 32, 0, G - 1), np.clip(P[:, 0] // 32, 0, G - 1)], 1)       # (32, 2) (행, 열)
        ins = (P[:, 0] >= 0) & (P[:, 0] < 512) & (P[:, 1] >= 0) & (P[:, 1] < 512)
        vis = (hid < 0.5) & ins
        out.append(dict(vid=r["video_id"], block=f"{r['objset']}|{r['block_id']}", split=sp.get(r["video_id"], "?"),
                        cell=f"{r['occlusion']}|k{r['k']}|{r['motion']}|{r['violation']}|{r['objset']}", tok=tok, vis=vis,
                        row=r))
    return out


def items_v11():
    import pandas as pd
    want = {}
    for r in csv.DictReader(open(REPO / "data_csv/intphysgen_v11_split/index_test.csv")):
        if r["violation_type"] == "vanish" and r["variant"] == "pos_a" and r["condition"] in ("moving_occlusion", "moving_occlusion_flat"):
            want[r["video_id"]] = r
    meta = pd.read_csv(STAGE / "v11_meta_sub.csv") if (STAGE / "v11_meta_sub.csv").exists() else None
    if meta is None:
        meta = pd.read_csv(V11META, low_memory=False)
        meta = meta[meta["name"].isin(list(want))]
        meta.to_csv(STAGE / "v11_meta_sub.csv", index=False)
    meta = meta.set_index("name")
    out = []
    for vid, r in want.items():
        m = meta.loc[vid]
        px = np.array([float(v) for v in str(m["object_px_x_by_sample"]).split()]); py = np.array([float(v) for v in str(m["object_px_y_by_sample"]).split()])
        k = int(r["sym_k"])
        hid = np.zeros(32, bool); hid[16 - k:16 + k] = True
        ins = (px >= 0) & (px < 288) & (py >= 0) & (py < 288)
        tok = np.stack([np.clip(py // 18, 0, G - 1), np.clip(px // 18, 0, G - 1)], 1)
        out.append(dict(vid=vid, block=r["block_id"] + "|" + r["source_block"], split="test",
                        cell=f"{r['condition']}|k{k}", tok=tok, vis=(~hid) & ins, row=r))
    return out


def write_index(items, setname):
    """선택한 행만 담은 색인 (WMADataset 이 사본에 없는 clip 을 안 건드리게)."""
    d = OUT / f"index_{setname}"; d.mkdir(parents=True, exist_ok=True)
    rows = [it["row"] for it in items]
    with open(d / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return d


def block_geom(it):
    """튜블릿 16 개의 물체 토큰 (블록 평균 → 반올림, 중심 1–14) 과 보임 (두 장 모두)."""
    t = it["tok"].reshape(NB, 2, 2).astype(np.float64).mean(1)
    yx = np.clip(np.floor(t + 0.5).astype(int), 1, G - 2)
    vb = it["vis"].reshape(NB, 2).all(1)
    return yx, vb


# ───────────────────────── 실행
def _worker(rank, world, setname, frames_root, bs, models, limit):
    import torch
    import torch.nn.functional as F
    from importlib.util import module_from_spec, spec_from_file_location
    from analysis.intphys2.model import build_from_config, _clean_backbone_key, _load_state_dict
    from analysis.intphys2.surprise import _context_target_indices
    from evals.world_model_analysis.data import WMADataset
    torch.cuda.set_device(rank); torch.set_num_threads(4); dev = torch.device("cuda", rank)
    spec = spec_from_file_location("ir", REPO / "z_research/scripts/analysis/rollout2_identity_readout.py")
    ir = module_from_spec(spec); spec.loader.exec_module(ir); rp = ir.rp
    model = dict(rp.MODEL)
    bundle = build_from_config(model, dev)
    preds = {}
    for name in models:
        if not MODELS[name]:
            preds[name] = bundle.predictor; continue
        pm = copy.deepcopy(bundle.predictor)
        st = torch.load(REPO / MODELS[name], map_location="cpu", weights_only=False)
        kind = (st.get("arch") or {}).get("kind")
        assert kind in (None, "default", "oneshot"), (name, kind)
        psd = _clean_backbone_key(st["predictor"]); msd = pm.state_dict()
        assert not (set(msd) - set(psd)), (name, sorted(set(msd) - set(psd))[:3])
        _load_state_dict(pm, st["predictor"], tag=name)
        preds[name] = pm.to(dev).eval()
    items = items_vobench() if setname == "vobench" else items_v11()
    if limit: items = items[:limit]
    idir = OUT / f"index_{setname}"
    if setname == "vobench":
        cfgd = dict(root=str(idir), index_csv="index.csv", frames_root=str(REPO / frames_root), frames_pattern="{file_name}/{frame:06d}.png",
                    frames_start=0, frames_stride=3, n_frames=32, resolution=256, block_column="block_id", pair_column="pair_id",
                    variant_column="variant", plausible_column="plausible", type_column="condition")
    else:
        cfgd = dict(root=str(idir), index_csv="index.csv", frames_root=str(REPO / frames_root), frames_pattern="{file_name}/{frame:06d}.png",
                    frames_start=0, frames_stride=3, n_frames=32, resolution=256, block_column="block_id", pair_column="pair_id",
                    variant_column="variant", plausible_column="plausible", type_column="condition")
    ds = WMADataset(dict(data=cfgd, model=model, features={"cache_dir": "/tmp"}, surprise={}))
    di = {r.video_id: i for i, r in enumerate(ds.records)}
    mine = items[rank * len(items) // world:(rank + 1) * len(items) // world]

    _sdpa = F.scaled_dot_product_attention
    REC = {"on": False, "i": 0}

    def sdpa(q, k, v, *a, **kw):
        out = _sdpa(q, k, v, *a, **kw)
        if REC["on"] and q.shape[2] == NB * S and k.shape[2] == NB * S:
            REC["fn"](q, k, REC["i"]); REC["i"] += 1
        return out
    F.scaled_dot_product_attention = sdpa

    # 상자 중심 격자 (1..14) 의 좌표
    cy, cx = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    inner = (cy >= 1) & (cy <= G - 2) & (cx >= 1) & (cx <= G - 2)
    acc = {}

    def put(k_, v_): acc.setdefault(k_, []).append(v_)
    t0 = time.time()
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(4)
    batches = [mine[b0:b0 + bs] for b0 in range(0, len(mine), bs)]
    futs = [pool.submit(lambda bb: torch.stack([ds.clip(di[it["vid"]]) for it in bb]), bb) for bb in batches[:2]]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for bi, bb in enumerate(batches):
            x = futs[bi].result().to(dev, dtype=bundle.dtype)
            if bi + 2 < len(batches): futs.append(pool.submit(lambda bb: torch.stack([ds.clip(di[it["vid"]]) for it in bb]), batches[bi + 2]))
            B = x.size(0)
            geo = [block_geom(it) for it in bb]
            YX = np.stack([g[0] for g in geo]); VB = np.stack([g[1] for g in geo])                     # (B, 16, 2) · (B, 16)
            # query 토큰 (걸음 r 마다 9 토큰 인덱스)
            QI = np.zeros((B, K, 9), np.int64)
            for j in range(B):
                for r in range(K):
                    y, x_ = YX[j, C + r]
                    QI[j, r] = [(C + r) * S + (y + dy) * G + (x_ + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
            QIt = torch.tensor(QI, device=dev)
            # key 쪽 마스크: 걸음 r · 블록 b 마다 obj / ring / mir / ql 상자 중심 (16×16 bool)
            KEYOK = np.zeros((B, K, NB), bool)
            OBJ = np.zeros((B, K, NB, G, G), bool); RING = np.zeros_like(OBJ); MIR = np.zeros_like(OBJ); QL = np.zeros_like(OBJ)
            MOK = np.zeros((B, K, NB), bool); ROK = np.zeros((B, K, NB), bool)
            for j in range(B):
                for r in range(K):
                    q = YX[j, C + r]
                    for b in range(C):
                        o = YX[j, b]
                        if not VB[j, b] or np.abs(o - q).max() < 3: continue
                        KEYOK[j, r, b] = True
                        OBJ[j, r, b, o[0], o[1]] = True; QL[j, r, b, q[0], q[1]] = True
                        d0 = np.hypot(*(o - q)); dc = np.hypot(cy - q[0], cx - q[1])
                        rm = inner & (np.abs(dc - d0) <= 0.5) & (np.maximum(np.abs(cy - o[0]), np.abs(cx - o[1])) >= 3)
                        if rm.any(): RING[j, r, b] = rm; ROK[j, r, b] = True
                        m_ = 2 * q - o
                        if (m_ >= 1).all() and (m_ <= G - 2).all() and np.abs(m_ - o).max() >= 3:
                            MIR[j, r, b, m_[0], m_[1]] = True; MOK[j, r, b] = True
            T_ = lambda a: torch.tensor(a, device=dev, dtype=torch.float32)
            OBJt, RINGt, MIRt, QLt = T_(OBJ), T_(RING), T_(MIR), T_(QL)
            RINGn = RINGt.sum((-1, -2)).clamp(min=1)

            ci, ti = _context_target_indices(ctx_frames=2 * C, tgt_frames=2 * K, tubelet_size=2, spatial_tokens=S, batch_size=B, device=dev)
            z = bundle.context_encoder(x, masks=[ci]); z = z[-1] if isinstance(z, list) else z
            for name, pm in preds.items():
                res = {k_: np.full((B, K, L_, NB), np.nan, np.float32) for k_ in ("obj", "ring", "mir", "ql", "bm")}
                hobj = np.full((B, K, L_, H_), np.nan, np.float32); hring = np.full_like(hobj, np.nan)

                def fn(q, k, l):
                    for r in range(K):
                        qq = torch.gather(q, 2, QIt[:, r][:, None, :, None].expand(B, q.shape[1], 9, q.shape[-1]))      # (B, H, 9, d)
                        A = torch.softmax(qq.float() @ k.float().transpose(-1, -2) * q.shape[-1] ** -0.5, -1).mean(2)  # (B, H, 4096)
                        Ah = A.view(B, H_, NB, G, G)
                        bsum = F.avg_pool2d(Ah.reshape(B * H_ * NB, 1, G, G), 3, 1, 1, count_include_pad=True).view(B, H_, NB, G, G) * 9
                        bm = Ah.sum((-1, -2))                                                                      # (B, H, NB)
                        ob = (bsum * OBJt[:, r, None]).sum((-1, -2)); rg = (bsum * RINGt[:, r, None]).sum((-1, -2)) / RINGn[:, r, None]
                        mi = (bsum * MIRt[:, r, None]).sum((-1, -2)); qlv = (bsum * QLt[:, r, None]).sum((-1, -2))
                        res["obj"][:, r, l] = ob.mean(1).cpu().numpy(); res["ring"][:, r, l] = rg.mean(1).cpu().numpy()
                        res["mir"][:, r, l] = mi.mean(1).cpu().numpy(); res["ql"][:, r, l] = qlv.mean(1).cpu().numpy()
                        res["bm"][:, r, l] = bm.mean(1).cpu().numpy()
                        ko = torch.tensor(KEYOK[:, r], device=dev, dtype=torch.float32)[:, None, :]                  # (B, 1, NB)
                        hobj[:, r, l] = (ob * ko).sum(-1).cpu().numpy(); hring[:, r, l] = (rg * ko).sum(-1).cpu().numpy()
                REC.update(on=True, i=0, fn=fn); pm(z, ci, ti, mask_index=0); REC["on"] = False
                assert REC["i"] == L_, REC["i"]
                for k_, v_ in res.items(): put(f"{name}__{k_}", v_.astype(np.float16) if k_ != "bm" else v_.astype(np.float16))
                put(f"{name}__hobj", hobj); put(f"{name}__hring", hring)
            put("keyok", KEYOK); put("mok", MOK); put("rok", ROK); put("qvis", VB[:, C:]); put("yx", YX); put("vb", VB)
            put("vid", np.array([it["vid"] for it in bb])); put("block", np.array([it["block"] for it in bb]))
            put("split", np.array([it["split"] for it in bb])); put("cell", np.array([it["cell"] for it in bb]))
            if rank == 0 and bi % 5 == 0: print(f"[r0] {setname} {(bi + 1) * bs}/{len(mine)} ({time.time() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "" if set(models) <= {"release", "vobftg", "v11pft", "ip1late_f0"} else "_" + "-".join(m for m in models if m != "release")
    np.savez(OUT / f"{setname}{tag}_r{rank}.npz", **{k_: np.concatenate(v_) for k_, v_ in acc.items()})
    print(f"[r{rank}] done {len(mine)} clips {time.time() - t0:.0f}s", flush=True)


# ───────────────────────── 요약 (CPU)
def load(setname):
    """주 실행 (<set>_r*.npz) + 추가 모델 실행 (<set>_<tag>_r*.npz, E37b). 추가 실행은 clip 순서가 같아야 한다 (vid 로 검사)."""
    Z = {}
    for f in sorted(OUT.glob(f"{setname}_r*.npz")):
        z = np.load(f)
        for k in z.files: Z.setdefault(k, []).append(z[k])
    Z = {k: np.concatenate(v) for k, v in Z.items()}
    extra = sorted({f.name.rsplit("_r", 1)[0] for f in OUT.glob(f"{setname}_*_r*.npz")})
    for tg in extra:
        E = {}
        for f in sorted(OUT.glob(f"{tg}_r*.npz")):
            z = np.load(f)
            for k in z.files: E.setdefault(k, []).append(z[k])
        E = {k: np.concatenate(v) for k, v in E.items()}
        assert (E["vid"] == Z["vid"]).all(), tg
        for k, v in E.items():
            if "__" in k and not k.startswith("release__"): Z[k] = v
    return Z


def ratio_stats(Z, name, sel, layers, ctrl="ring", nboot=2000, seed=0):
    """비 R = Σ obj / Σ ctrl (clip · 걸음 · 층 · key 블록), 유효 = key 유효 & query 보임 & 대조 유효. block bootstrap."""
    obj = Z[f"{name}__obj"].astype(np.float64)[:, :, layers]; ct = Z[f"{name}__{ctrl}"].astype(np.float64)[:, :, layers]
    ok = Z["keyok"] & Z["qvis"][:, :, None] & (Z["rok"] if ctrl == "ring" else Z["mok"] if ctrl == "mir" else Z["keyok"])   # (N, K, NB)
    ok = ok[:, :, None, :] & np.ones((1, 1, len(layers), 1), bool)
    o = np.where(ok, obj, 0).sum((1, 2, 3)); c = np.where(ok, ct, 0).sum((1, 2, 3)); n = ok.sum((1, 2, 3))
    o, c, n = o[sel], c[sel], n[sel]; blocks = Z["block"][sel]
    ub, inv = np.unique(blocks, return_inverse=True)
    ob = np.bincount(inv, o, len(ub)); cb = np.bincount(inv, c, len(ub)); nb_ = np.bincount(inv, n, len(ub))
    R = ob.sum() / max(cb.sum(), 1e-12)
    rng = np.random.default_rng(seed); bs = []
    for _ in range(nboot):
        ii = rng.integers(0, len(ub), len(ub)); bs.append(ob[ii].sum() / max(cb[ii].sum(), 1e-12))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    U = 9 / 4096
    return dict(R=float(R), ci=[float(lo), float(hi)], obj_per_box=float(ob.sum() / max(nb_.sum(), 1)), ctrl_per_box=float(cb.sum() / max(nb_.sum(), 1)),
                obj_over_uniform=float(ob.sum() / max(nb_.sum(), 1) / U), n_clip=int(sel.sum()), n_block=int(len(ub)), n_keybox=int(nb_.sum()))


def delta_R(Z, a, b, sel, layers, ctrl="ring", nboot=2000, seed=1):
    """ΔR = R(a) − R(b), 같은 block 재표집 (짝 bootstrap)."""
    def per_block(name):
        obj = Z[f"{name}__obj"].astype(np.float64)[:, :, layers]; ct = Z[f"{name}__{ctrl}"].astype(np.float64)[:, :, layers]
        ok = Z["keyok"] & Z["qvis"][:, :, None] & (Z["rok"] if ctrl == "ring" else Z["mok"])
        ok = ok[:, :, None, :] & np.ones((1, 1, len(layers), 1), bool)
        return np.where(ok, obj, 0).sum((1, 2, 3))[sel], np.where(ok, ct, 0).sum((1, 2, 3))[sel]
    oa, ca = per_block(a); ob, cb = per_block(b)
    ub, inv = np.unique(Z["block"][sel], return_inverse=True)
    f = lambda v: np.bincount(inv, v, len(ub))
    oa, ca, ob, cb = f(oa), f(ca), f(ob), f(cb)
    d0 = oa.sum() / ca.sum() - ob.sum() / cb.sum()
    rng = np.random.default_rng(seed); bs = []
    for _ in range(nboot):
        ii = rng.integers(0, len(ub), len(ub)); bs.append(oa[ii].sum() / ca[ii].sum() - ob[ii].sum() / cb[ii].sum())
    return dict(dR=float(d0), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])


def verdict(st):
    if st["R"] >= 2.0 and st["ci"][0] >= 1.5: return "켜짐"
    if st["R"] <= 1.2: return "꺼짐"
    return "약함"


def summarize():
    out = {"definition": __doc__.split("정의")[1].split("판:")[0].strip(), "band": BAND, "uniform_per_box": 9 / 4096, "sets": {}}
    for setname in ("vobench", "v11"):
        if not list(OUT.glob(f"{setname}_r*.npz")): continue
        Z = load(setname)
        names = sorted({k.split("__")[0] for k in Z if "__obj" in k})
        subsets = {}
        if setname == "vobench":
            late = np.array([c.startswith("late") for c in Z["cell"]]); vis = np.array([c.startswith("visible") for c in Z["cell"]])
            subsets = {"late_test": late & (Z["split"] == "test"), "late_train": late & (Z["split"] == "train"), "visible_test": vis & (Z["split"] == "test")}
            for kk in ("k1", "k3"):
                subsets[f"late_test_{kk}"] = late & (Z["split"] == "test") & np.array([f"|{kk}|" in c for c in Z["cell"]])
        else:
            subsets = {"late_test": np.ones(len(Z["vid"]), bool)}
            for cnd in ("moving_occlusion_flat", "moving_occlusion"):
                subsets[f"late_{cnd}"] = np.array([c.split("|")[0] == cnd for c in Z["cell"]])
            for kk in ("k1", "k2", "k3", "k4"):
                subsets[f"late_test_{kk}"] = np.array([c.endswith("|" + kk) for c in Z["cell"]])
        res = {}
        for sub, sel in subsets.items():
            if not sel.any(): continue
            res[sub] = {}
            for name in names:
                d = {"band_ring": ratio_stats(Z, name, sel, BAND, "ring"), "band_mir": ratio_stats(Z, name, sel, BAND, "mir"),
                     "all_ring": ratio_stats(Z, name, sel, list(range(L_)), "ring"), "band_ql": ratio_stats(Z, name, sel, BAND, "ql")}
                d["per_layer_ring"] = [ratio_stats(Z, name, sel, [l], "ring", nboot=300)["R"] for l in range(L_)]
                d["per_layer_obj_over_uniform"] = [ratio_stats(Z, name, sel, [l], "ring", nboot=1)["obj_over_uniform"] for l in range(L_)]
                d["per_layer_mir"] = [ratio_stats(Z, name, sel, [l], "mir", nboot=300)["R"] for l in range(L_)]
                d["verdict_ring"] = verdict(d["band_ring"]); d["verdict_mir"] = verdict(d["band_mir"])
                d["verdict"] = d["verdict_ring"] if d["verdict_ring"] == d["verdict_mir"] else "판정 보류"
                if name != "release" and "release" in names:
                    d["dR_vs_release_ring"] = delta_R(Z, name, "release", sel, BAND, "ring")
                    d["dR_vs_release_mir"] = delta_R(Z, name, "release", sel, BAND, "mir")
                res[sub][name] = d
        out["sets"][setname] = res
        print(f"\n== {setname}")
        for sub, rr in res.items():
            print(f"  [{sub}]")
            for name, d in rr.items():
                b = d["band_ring"]; m = d["band_mir"]
                print(f"    {name:11s} R_ring {b['R']:.2f} [{b['ci'][0]:.2f},{b['ci'][1]:.2f}] · R_mir {m['R']:.2f} [{m['ci'][0]:.2f},{m['ci'][1]:.2f}] · obj/균등 {b['obj_over_uniform']:.2f} · n {b['n_clip']} clip {b['n_block']} blk → {d['verdict']}"
                      + "   층별 R " + " ".join(f"{v:.1f}" for v in d["per_layer_ring"]))
    json.dump(out, open(OUTR / "E37_attn.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["run", "summarize", "index"])
    ap.add_argument("--set", default="vobench", choices=["vobench", "v11"]); ap.add_argument("--frames-root", default="")
    ap.add_argument("--gpus", type=int, default=1); ap.add_argument("--bs", type=int, default=2); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--models", default="release,vobftg,v11pft,ip1late_f0")
    a = ap.parse_args()
    if a.cmd == "index":
        its = items_vobench() if a.set == "vobench" else items_v11()
        write_index(its, a.set); print(a.set, len(its))
    elif a.cmd == "run":
        import torch
        torch.multiprocessing.spawn(_worker, args=(a.gpus, a.set, a.frames_root, a.bs, a.models.split(","), a.limit), nprocs=a.gpus, join=True)
    else:
        summarize()
