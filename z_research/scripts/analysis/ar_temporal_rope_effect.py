"""블록 AR 의 시간 RoPE 가 먼 과거 · 과거 물체 자리 attention 을 얼마나 누르나 (2026-11-06, 사용자 "(2) 일단 측정해봐").

AR (kind=ar, KV rollout) 이 블록 9 를 만드는 걸음 (블록 8 입력 토큰이 query) 에서, 층마다 **실제 층 입력 (norm1 뒤)** 을 그대로 두고
q · k 를 다시 계산한다 (모델 forward 는 바꾸지 않는다).
  true  : 원래 RoPE (시간 · 행 · 열 회전) — 실제 attention 과 같아야 한다 (verify)
  notime: 시간 축 회전만 빼고 (q · k 모두 시간 위치 0) 행 · 열 회전은 그대로 → 시간 거리에 따른 위치 효과가 사라진다
두 분포의 차이 = 시간 위치 인코딩만의 효과. 재는 것: key 블록별 질량 (query 256 칸 평균) · 물체 query (블록 9 시점 물체 3×3) 의
블록 t 물체 자리 3×3 질량. VOBench v1 permanence 가시 · late 움직임 (attn_block9_object.py 와 같은 장면 · 물체 위치).
시간 RoPE 의 주파수 (head_dim 32 → 시간 10 차원 = 5 쌍): ω_i = 1 / 10000^(i/5) = 1, 0.158, 0.025, 0.004, 0.0006 rad / 블록.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from attn_release_vs_ar import C, L, NB, ROOT, S, load_cfg, obj_track, region  # noqa: E402

from analysis.intphys2.model import build_from_config  # noqa: E402
from analysis.intphys2.surprise import _context_target_indices  # noqa: E402
from evals.world_model_analysis.data import WMADataset  # noqa: E402
from src.models.utils.modules import ACRoPEAttention, rotate_queries_or_keys  # noqa: E402

F9 = 8
_ORIG_KV = ACRoPEAttention.forward_kv


class St:
    on = False
    li_of = {}          # 모듈 id → 층 번호
    X = None            # [층][걸음] = 층 입력 (B, S, C)
    pos = None          # [걸음] = 절대 토큰 인덱스 (B, S)
    out = None          # 결과 누적
    oq = ok = grp = None
    verify = []


def _rot(m, q, k, ids, notime):
    """ACRoPEAttention.forward_kv 의 회전 구간 (H = W = 16). notime 이면 시간 위치를 0 으로."""
    mm = ids.unsqueeze(1).repeat(1, m.num_heads, 1)
    d, h, w = m.separate_positions(mm, 16, 16)
    if notime:
        d = torch.zeros_like(d)
    h = h * (m.grid_size / 16)
    w = w * (m.grid_size / 16)
    s = 0
    outq, outk = [], []
    for dim, pos in ((m.d_dim, d), (m.h_dim, h), (m.w_dim, w)):
        outq.append(rotate_queries_or_keys(q[..., s:s + dim], pos=pos))
        outk.append(rotate_queries_or_keys(k[..., s:s + dim], pos=pos))
        s += dim
    if s < m.head_dim:
        outq.append(q[..., s:]); outk.append(k[..., s:])
    return torch.cat(outq, -1), torch.cat(outk, -1)


def _kv(self, x, mask, H, W, past=None):
    out = _ORIG_KV(self, x, mask, H, W, past)
    if not St.on:
        return out
    li = St.li_of[id(self)]
    step = 0 if past is None else past[0].size(2) // S
    St.X[li][step] = x.detach()
    if li == 0:
        St.pos[step] = mask.detach()
    if step != F9 - 1:
        return out
    X = torch.cat([St.X[li][t] for t in range(F9)], 1)                     # (B, 8S, C) 블록 1..8 의 층 입력
    ids = torch.cat([St.pos[t] for t in range(F9)], 1)
    B = X.size(0)
    qkv = self.qkv(X).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
    q, k = qkv[0].float(), qkv[1].float()
    for tag, nt in (("true", False), ("notime", True)):
        qr, kr = _rot(self, q, k, ids, nt)
        p = (torch.matmul(qr[:, :, -S:], kr.transpose(-1, -2)) * self.scale).softmax(-1)    # (B, H, S, 8S) query = 블록 8
        if tag == "true" and li == 0 and not St.verify:
            St.verify.append(float((torch.matmul(p, qkv[2].float()).transpose(1, 2).reshape(B, S, -1)
                                    - 0).abs().mean()))     # 값 크기 기록용 (아래 대조는 out 과)
            o = torch.matmul(p, qkv[2].float()).transpose(1, 2).reshape(B, S, -1)
            St.verify.append(float((self.proj(o.to(x.dtype)).float() - out[0].float()).abs().max()))
        pb = p.mean(1).view(B, S, F9, S)                                      # 헤드 평균 (B, Sq, 블록, Sk)
        blk = pb.sum(-1).mean(1)                                              # (B, 8)
        for b in range(B):
            g = St.grp[b]
            St.out[tag][g]["blk"][li] += blk[b].double().cpu().numpy()
            St.out[tag][g]["n"][li] += 1
            oq = St.oq[b]
            if oq is None:
                continue
            po = pb[b, oq].mean(0)                                            # (8, S)
            St.out[tag][g]["obj"][li] += np.array([float(po[t, St.ok[b][t]].sum()) for t in range(F9)])
            St.out[tag][g]["nobj"][li] += 1
    return out


@contextlib.contextmanager
def rec(pred, oq, ok, grp):
    St.on, St.oq, St.ok, St.grp = True, oq, ok, grp
    St.X = [dict() for _ in range(L)]
    St.pos = {}
    ACRoPEAttention.forward_kv = _kv
    try:
        yield
    finally:
        ACRoPEAttention.forward_kv = _ORIG_KV
        St.on = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ar", help="ar (natural_ar_vw_state) | ar_v4bc4 (v4bc4 λ1 기준판)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    cfg = load_cfg("ar")
    if a.model == "ar_v4bc4":
        cfg["model"]["predictor_checkpoint"] = os.path.join(ROOT, "z_training/runs/v4bc4_ar_vw_state_negimp/latest.pt")
    rows = {r["video_id"]: r for r in csv.DictReader(open(os.path.join(ROOT, "data_csv/vobench_v1/index.csv")))}
    ds = WMADataset(cfg)
    clips = [i for i, x in enumerate(ds.records) if rows[x.video_id]["violation_type"] == "permanence"
             and rows[x.video_id]["variant"] == "pos_a" and rows[x.video_id]["occlusion"] in ("visible", "late")
             and rows[x.video_id]["motion"] != "flat_static"]
    meta = []
    for i in clips:
        r = rows[ds.records[i].video_id]
        vis, rc = obj_track(r["file_name"])
        meta.append((r["occlusion"], rc))
    bundle = build_from_config(cfg["model"], dev)
    core = bundle.predictor.core
    St.li_of = {id(b.attn): li for li, b in enumerate(core.predictor_blocks)}
    St.out = {t: {g: {"blk": np.zeros((L, F9)), "n": np.zeros(L), "obj": np.zeros((L, F9)), "nobj": np.zeros(L)}
                  for g in ("visible", "late")} for t in ("true", "notime")}
    ci, ti = _context_target_indices(ctx_frames=16, tgt_frames=16, tubelet_size=2, spatial_tokens=S, batch_size=8, device=dev)
    for s in range(0, len(clips), 8):
        idx = list(range(s, min(s + 8, len(clips))))
        x = torch.stack([ds.clip(clips[j]) for j in idx]).to(dev, bundle.dtype)
        oq = [torch.tensor(region(meta[j][1][F9]), device=dev) for j in idx]
        ok = [[region(meta[j][1][t]) for t in range(F9)] for j in idx]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            z = bundle.context_encoder(x, masks=[ci[:len(idx)]])
            with rec(core, oq, ok, [meta[j][0] for j in idx]):
                bundle.predictor(z, ci[:len(idx)], ti[:len(idx)], mask_index=0)
    res = {"verify_max_abs_out_diff_layer1": St.verify[-1] if St.verify else None, "model": a.model, "n_clips": len(clips)}
    print(f"검증 (층 1, 재계산 출력 vs 실제): max|Δ| {res['verify_max_abs_out_diff_layer1']}")
    for g in ("visible", "late"):
        for tag in ("true", "notime"):
            d = St.out[tag][g]
            blk = d["blk"] / d["n"][:, None]
            obj = d["obj"] / np.maximum(d["nobj"], 1)[:, None]
            res[f"{g}/{tag}"] = {"blk": blk.tolist(), "obj": obj.tolist()}
        bt, bn = np.array(res[f"{g}/true"]["blk"]).mean(0), np.array(res[f"{g}/notime"]["blk"]).mean(0)
        ot, on = np.array(res[f"{g}/true"]["obj"]).mean(0), np.array(res[f"{g}/notime"]["obj"]).mean(0)
        print(f"\n[{g} · 움직임] 블록 1..8 질량 % (층 평균)")
        print("  원래 RoPE   : " + " ".join(f"{100*v:5.1f}" for v in bt) + f" | 과거 1–7 합 {100*bt[:7].sum():.1f}")
        print("  시간 회전 뺌: " + " ".join(f"{100*v:5.1f}" for v in bn) + f" | 과거 1–7 합 {100*bn[:7].sum():.1f}")
        print("  물체 자리 원래: " + " ".join(f"{100*v:5.2f}" for v in ot) + f" | 과거 1–7 합 {100*ot[:7].sum():.2f}")
        print("  물체 자리 뺌  : " + " ".join(f"{100*v:5.2f}" for v in on) + f" | 과거 1–7 합 {100*on[:7].sum():.2f}")
        lt = np.array(res[f"{g}/true"]["blk"])[:, :7].sum(1); ln = np.array(res[f"{g}/notime"]["blk"])[:, :7].sum(1)
        print("  층별 과거 1–7 합 (원래 → 뺌): " + " · ".join(f"{i+1}:{100*a_:.0f}→{100*b_:.0f}" for i, (a_, b_) in enumerate(zip(lt, ln))))
    out = a.out or os.path.join(ROOT, f"z_analysis/results/release_ladder_20261106/ar_temporal_rope_{a.model}.json")
    json.dump(res, open(out, "w"))
    print("저장", out)


if __name__ == "__main__":
    main()
