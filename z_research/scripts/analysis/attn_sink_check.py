"""attention sink 점검 — 블록 9 를 만드는 query 가 소수의 key 토큰에 몰리는가 (2026-11-06, 사용자 "Attention Sinking 관련 문제는 없어?").

query = 블록 9 를 만드는 자리 전체 256 칸 (V-JEPA 2: 블록 9 mask token · AR: 블록 8 입력 토큰). 층마다 (헤드 평균 · query 평균) key 토큰별 질량 m[k].
재는 것 (층별, 클립 평균)
  top1   : 질량 상위 1 % key 의 몫 (균등이면 1 %)
  max    : 가장 많이 받은 key 하나의 몫
  where  : 상위 key 의 (블록, 칸) — 클립 사이에 같은 칸이면 위치 sink (내용과 무관)
  vnorm  : 상위 key 의 value 크기 / 전체 key 의 value 크기 중앙값 (헤드 평균). < 1 이면 "받아도 출력에 적게 기여" = sink 꼴
  head_max: 헤드별로 본 top1 의 최대 (평균이 sink 를 가릴 수 있어서)
SDPA 를 predictor 호출 동안만 감싸 q·k 로 다시 계산 (출력은 원래 SDPA 그대로).
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from attn_release_vs_ar import L, NB, ROOT, S, load_cfg  # noqa: E402

from analysis.intphys2.model import build_from_config  # noqa: E402
from analysis.intphys2.surprise import _context_target_indices  # noqa: E402
from evals.world_model_analysis.data import WMADataset  # noqa: E402

F9 = 8
_ORIG = F.scaled_dot_product_attention


class St:
    on, calls, mode, rec, grp = False, 0, "", None, None


def _rec(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
    out = _ORIG(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)
    if not St.on:
        return out
    li, step = St.calls % L, St.calls // L
    St.calls += 1
    B, Hh, Nq, D = q.shape
    Nk = k.size(2)
    if St.mode == "ar":
        if step != F9 - 1:
            return out
        qs = slice(0, S)
    else:
        qs = slice(F9 * S, (F9 + 1) * S)
    sc = scale if scale is not None else D ** -0.5
    p = (torch.matmul(q[:, :, qs].float(), k.float().transpose(-1, -2)) * sc).softmax(-1)    # (B, H, 256, Nk)
    mh = p.mean(2)                                                                          # (B, H, Nk) 헤드별
    m = mh.mean(1)                                                                          # (B, Nk)
    vn = v.float().norm(dim=-1).mean(1)                                                     # (B, Nk)
    n1 = max(1, Nk // 100)
    for b in range(B):
        top = torch.topk(m[b], n1).indices
        St.rec[li]["top1"].append(float(m[b, top].sum()))
        St.rec[li]["max"].append(float(m[b].max()))
        St.rec[li]["argmax"].append(int(m[b].argmax()))
        St.rec[li]["vnorm"].append(float(vn[b, top].mean() / vn[b].median()))
        St.rec[li]["head_top1"].append(float(torch.topk(mh[b], n1, dim=-1).values.sum(-1).max()))
        St.rec[li]["Nk"] = Nk
        St.rec[li]["grp"].append(St.grp[b])
    return out


@contextlib.contextmanager
def rec(mode):
    St.on, St.calls, St.mode = True, 0, mode
    F.scaled_dot_product_attention = _rec
    try:
        yield
    finally:
        F.scaled_dot_product_attention = _ORIG
        St.on = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--out", default=os.path.join(ROOT, "z_analysis/results/release_ladder_20261106/attn_sink_check.json"))
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    rows = {r["video_id"]: r for r in csv.DictReader(open(os.path.join(ROOT, "data_csv/vobench_v1/index.csv")))}
    ds = WMADataset(load_cfg("release"))
    clips = [i for i, x in enumerate(ds.records) if rows[x.video_id]["violation_type"] == "permanence"
             and rows[x.video_id]["variant"] == "pos_a" and rows[x.video_id]["occlusion"] in ("visible", "late")]
    rng = np.random.default_rng(0)
    clips = sorted(rng.choice(clips, size=min(a.n, len(clips)), replace=False).tolist())
    ci, ti = _context_target_indices(ctx_frames=16, tgt_frames=16, tubelet_size=2, spatial_tokens=S, batch_size=8, device=dev)
    res = {}
    for name in ("release", "ar"):
        bundle = build_from_config(load_cfg(name)["model"], dev)
        St.rec = [defaultdict(list) for _ in range(L)]
        for s in range(0, len(clips), 8):
            idx = clips[s:s + 8]
            x = torch.stack([ds.clip(i) for i in idx]).to(dev, bundle.dtype)
            St.grp = [rows[ds.records[i].video_id]["occlusion"] for i in idx]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                z = bundle.context_encoder(x, masks=[ci[:len(idx)]])
                with rec("ar" if name == "ar" else "release"):
                    bundle.predictor(z, ci[:len(idx)], ti[:len(idx)], mask_index=0)
        out = []
        for li in range(L):
            r = St.rec[li]
            am = Counter(r["argmax"])
            pos, cnt = am.most_common(1)[0]
            out.append(dict(layer=li + 1, Nk=r["Nk"], top1=float(np.mean(r["top1"])), max=float(np.mean(r["max"])),
                            head_top1=float(np.mean(r["head_top1"])), vnorm=float(np.mean(r["vnorm"])),
                            top1_by={g: float(np.mean([t for t, gg in zip(r["top1"], r["grp"]) if gg == g])) for g in ("visible", "late")},
                            argmax_block=pos // S + 1, argmax_cell=[(pos % S) // 16, pos % 16], argmax_same=cnt / len(r["argmax"])))
        res[name] = out
        del bundle
        torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), indent=1)
    for name, out in res.items():
        print(f"\n### {name} (클립 {len(clips)}, key {out[0]['Nk']} 개, 상위 1 % = {max(1, out[0]['Nk'] // 100)} 개)")
        print("가시 / late 상위 1 % 몫: " + " · ".join(f"층{o['layer']} {100*o['top1_by']['visible']:.1f}/{100*o['top1_by']['late']:.1f}" for o in out))
        print("| 층 | 상위 1 % key 몫 | 최대 key 하나 | 헤드 중 최대 상위 1 % | 상위 key value 크기 / 중앙값 | 최대 key 자리 (블록, 행·열) | 클립 사이 같은 자리 |")
        print("|---:|---:|---:|---:|---:|---|---:|")
        for o in out:
            print(f"| {o['layer']} | {100*o['top1']:.1f} % | {100*o['max']:.1f} % | {100*o['head_top1']:.1f} % | {o['vnorm']:.2f} | "
                  f"블록 {o['argmax_block']}, ({o['argmax_cell'][0]},{o['argmax_cell'][1]}) | {100*o['argmax_same']:.0f} % |")


if __name__ == "__main__":
    main()
