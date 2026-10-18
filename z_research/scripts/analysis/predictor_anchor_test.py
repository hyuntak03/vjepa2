#!/usr/bin/env python3
"""미래 자리는 '역사의 운동' 에서 오나, '마지막 관측에 보이는 내용' 에서 오나 — anchor 이식 (2026-10-13).

물음 (사용자 10-13): visible 에서는 위치를 이어 가는데 late (문맥 끝 가림) 에서는 못 한다. 단순히 '속도를 위치로 적분 못 한다' 면 둘 다 못 해야 한다.
가설 H_anchor: predictor 의 운동은 독립된 상태 (where · velocity) 가 아니라 **마지막 관측에 보이는 물체 내용의 변환** 으로만 있다 (운동 · 외형 비분리).
가설 H_state : predictor 는 문맥 전체에서 자리 · 속도 상태를 들고, 마지막 관측은 그 상태의 한 표본일 뿐이다.

판: VOB_readout_primitive (등속 직선, 전부 보임, 10 fps 32 장) test clip. 문맥 8 블록 = 프레임 0–15 중 **마지막 블록 (프레임 14, 15) 만 픽셀에서 조작**:
  true     원본
  paste    프레임 13 의 물체 조각을 원래 자리 (c13 + v, c13 + 2v) 에 다시 붙임 — 붙이기 인공물 대조 (≈ true 여야)
  erase    물체를 지움 (물체가 멀리 있던 프레임의 같은 자리 배경으로) — 마지막 관측에서만 안 보임 (late k=1 과 같은 꼴, 가림막 없음)
  stay     프레임 13 자리에 그대로 (마지막 한 블록만 '멈춤')
  back     c13 − v, c13 − 2v (마지막 한 블록만 '되돌아감')
  double   c13 + 2v, c13 + 4v (마지막 한 블록만 '두 배')
앞 7 블록 (프레임 0–13) 은 모든 조건에서 같다 = 같은 역사.
측정: 각 모델의 자기 p 자 (identity_vobprim_<name>/p, 검증된 자) 로 미래 8 블록의 위치 · 없음 확률.
  s_k = (읽은 자리 − c_true(블록 C−1)) · v̂ / 18 px (운동 방향 칸), true 와의 차 Δs_k.
  H_state 면 stay · back · double 의 Δs 는 작고 (역사가 이김) erase 에서도 물체가 외삽 자리에 나온다.
  H_anchor 면 Δs 가 마지막 블록의 내용 자리 · 함축 속도를 따르고, erase 에서 물체가 사라진다.

  python predictor_anchor_test.py --name ar40 --ckpt z_training/ariel/ar_future_1e_5/latest.pt --gpus 4
  python predictor_anchor_test.py --name release --gpus 4
산출: z_research/VOBench_v1/bottleneck/anchor/<name>.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
import predictor_bottleneck as PB  # noqa: E402  (PRESENCE_* 환경을 VOB_readout 으로 맞춘다)

S, NB, C, K, D = 256, 16, 8, 8, 1280
CONDS = ("true", "paste", "erase", "erase2", "stay", "back", "double", "stay6", "back6")
OUT = REPO / "z_research/VOBench_v1/bottleneck/anchor"
SC = 256 / 288                                     # 288 px 좌표 → 256 px 입력


def select(max_n):
    Z = {}
    for f in sorted((PB.OUT / "ar40").glob("shard*.npz")):
        z = np.load(f, allow_pickle=True)
        for k in ("row", "part"): Z.setdefault(k, []).append(z[k])
    Z = {k: np.concatenate(v) for k, v in Z.items()}
    rows = PB.select(1600, 1200)[1]
    keep = []
    for i in Z["row"][Z["part"] == "test"]:
        r = rows[int(i)]
        px = PB.arr(r["px_x_by_sample"]); py = PB.arr(r["px_y_by_sample"]); vis = PB.arr(r["visible_by_sample"])
        v = np.array([px[13] - px[12], py[13] - py[12]])
        R = float(r["obj_apparent_px"]) / 2 + 6
        v11 = np.array([px[11] - px[10], py[11] - py[10]])
        cs = [np.array([px[13], py[13]]) + m * v for m in (-2, -1, 0, 1, 2, 4)] + [np.array([px[11], py[11]]) + m * v11 for m in (-2, -1, 0)]
        if vis[:16].min() < 1 or np.linalg.norm(v) * 2 < 9: continue                    # 한 블록 이동 ≥ 반 칸
        if not all(R + 2 <= c[0] <= 288 - R - 2 and R + 2 <= c[1] <= 288 - R - 2 for c in cs): continue
        keep.append(int(i))
    return keep[:max_n], rows


def manipulate(x, r, cond):
    """x (3, 32, 256, 256) 정규화 텐서. 기본은 마지막 문맥 블록 (프레임 14, 15) 만 바꾼다.
    erase2 = 프레임 12–15 지움 (late k=2 꼴). stay6 · back6 = 블록 6 (프레임 12, 13) 만 바꾸고 블록 7 은 원본 (영향 비교)."""
    if cond == "true": return x
    px = PB.arr(r["px_x_by_sample"]); py = PB.arr(r["px_y_by_sample"])
    P = np.stack([px, py], 1) * SC
    j = 6 if cond.endswith("6") else 7
    f0, f1 = 2 * j, 2 * j + 1; ref = f0 - 1; v = P[ref] - P[ref - 1]
    R = int(np.ceil((float(r["obj_apparent_px"]) / 2 + 6) * SC))
    x = x.clone()

    def box(c):
        cx, cy = int(round(c[0])), int(round(c[1])); return cy - R, cy + R + 1, cx - R, cx + R + 1

    fr = (12, 13, 14, 15) if cond == "erase2" else (f0, f1)
    far = max(range(32), key=lambda f: min(np.linalg.norm(P[f] - P[g]) for g in fr))
    src_obj = x[:, ref if cond != "erase2" else 11].clone()
    for f in fr:                                                                          # 지우기
        y0, y1, x0, x1 = box(P[f]); x[:, f, y0:y1, x0:x1] = x[:, far, y0:y1, x0:x1]
    if cond.startswith("erase"): return x
    mult = {"paste": (1, 2), "stay": (0, 0), "back": (-1, -2), "double": (2, 4)}[cond.rstrip("6")]
    sy0, sy1, sx0, sx1 = box(P[ref]); patch = src_obj[:, sy0:sy1, sx0:sx1]
    for f, m in zip((f0, f1), mult):
        y0, y1, x0, x1 = box(P[ref] + m * v); x[:, f, y0:y1, x0:x1] = patch
    return x


def _worker(rank, world, name, ckpt, bs, max_n):
    import torch
    import torch.nn.functional as F
    from importlib.util import module_from_spec, spec_from_file_location
    from analysis.intphys2.model import build_from_config
    from app.vjepa_frozen.ar import block_indices, encode_blocks
    from evals.world_model_analysis.data import WMADataset
    torch.cuda.set_device(rank); torch.set_num_threads(4); dev = torch.device("cuda", rank)
    spec = spec_from_file_location("ir", REPO / "z_research/scripts/analysis/rollout2_identity_readout.py")
    ir = module_from_spec(spec); spec.loader.exec_module(ir); rp = ir.rp
    model = dict(rp.MODEL) if name == "release" else dict(rp.MODEL, predictor_checkpoint=str(REPO / ckpt))
    bundle = build_from_config(model, dev); pred = bundle.predictor
    R = ir.IdReadout().to(dev).eval()
    R.load_state_dict(torch.load(REPO / f"z_research/VOBench_v1/readout/identity_vobprim_{name}/p/readout.pt", map_location=dev))
    ds = WMADataset(dict(data=rp.DATA, model=model, features={"cache_dir": "/tmp"}, surprise={}))
    keep, rows = select(max_n)
    mine = keep[rank * len(keep) // world:(rank + 1) * len(keep) // world]
    assert all(ds.records[i].video_id == rows[i]["video_id"] for i in mine[:5])
    LN = lambda t: F.layer_norm(t.float(), (t.size(-1),))
    out = {c: {"xy": [], "pn": []} for c in CONDS}; t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b0 in range(0, len(mine), bs):
            ii = mine[b0:b0 + bs]; base = [ds.clip(i) for i in ii]
            for c in CONDS:
                x = torch.stack([manipulate(base[j], rows[i], c) for j, i in enumerate(ii)]).to(dev, dtype=bundle.dtype); B = x.size(0)
                if name == "release":
                    from analysis.intphys2.surprise import _context_target_indices
                    ci, ti = _context_target_indices(ctx_frames=2 * C, tgt_frames=2 * K, tubelet_size=2, spatial_tokens=S, batch_size=B, device=dev)
                    z = bundle.context_encoder(x, masks=[ci]); z = z[-1] if isinstance(z, list) else z
                    p = pred(z, ci, ti, mask_index=0)
                else:
                    hb = encode_blocks(bundle.target_encoder, x[:, :, :2 * C], 2)
                    p = pred.rollout(hb[:, :C * S], block_indices(B, C, S, dev), K)
                xs, pns = [], []
                for k in range(K):
                    xy, lg, _ = R(p[:, k * S:(k + 1) * S].half())
                    xs.append(((xy.float() + 1) * 144).cpu().numpy()); pns.append(torch.softmax(lg.float(), -1)[:, PB.NONE].cpu().numpy())
                out[c]["xy"].append(np.stack(xs, 1)); out[c]["pn"].append(np.stack(pns, 1))
            if rank == 0: print(f"[r0] {name} {b0 + len(ii)}/{len(mine)} ({time.time() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{name}_r{rank}.npz", row=np.array(mine), **{f"{c}_{k}": np.concatenate(out[c][k]) for c in CONDS for k in ("xy", "pn")})


def summarize(name):
    Z = {}
    for f in sorted(OUT.glob(f"{name}_r*.npz")):
        z = np.load(f)
        for k in z.files: Z.setdefault(k, []).append(z[k])
    Z = {k: np.concatenate(v) for k, v in Z.items()}
    rows = PB.select(1600, 1200)[1]
    T = np.stack([PB.truth(rows, int(i))[0] for i in Z["row"]])                          # (N, 16, 2) 288 px 블록 중심
    v = T[:, C - 1] - T[:, C - 2]; step = np.linalg.norm(v, axis=1); vh = v / step[:, None]
    res = {"n": int(len(Z["row"])), "block_step_cells": float(np.median(step) / 18)}

    def along(c):                                                                         # (N, K) 칸, 블록 C−1 의 참 자리 기준
        return ((Z[f"{c}_xy"] - T[:, C - 1][:, None]) * vh[:, None]).sum(-1) / 18.0
    truth_s = ((T[:, C:] - T[:, C - 1][:, None]) * vh[:, None]).sum(-1) / 18.0
    res["truth_s"] = np.median(truth_s, 0).tolist()
    s_true = along("true")
    for c in CONDS:
        s = along(c)
        res[c] = {"s": np.median(s, 0).tolist(), "ds": np.median(s - s_true, 0).tolist(),
                  "pnone": Z[f"{c}_pn"].mean(0).tolist(), "absent": (Z[f"{c}_pn"] > 0.5).mean(0).tolist()}
    # 마지막 블록 내용이 함축하는 자리 이동 (블록 C−1 중심, 참 대비 칸): stay −0.5·step·…
    # 역사 속도를 쓰나: clip 마다 s_k 를 블록당 참 이동 (칸) 에 회귀한 기울기. 참 궤적이면 기울기 = k
    sc = step / 18.0
    for c in CONDS:
        s = along(c); res[c]["slope_on_history_speed"] = [float(np.polyfit(sc, s[:, k], 1)[0]) for k in range(K)]
    # 따름 지수 (0 = 역사 (true 조건) 를 따름, 1 = 마지막 블록 내용을 따름): 마지막 블록이 함축하는 궤적 = 그 블록 자리 + 그 속도로 등속
    st = res["block_step_cells"]; s_true_med = np.array(res["true"]["s"])
    last_path = {"stay": np.full(K, -0.75 * st), "back": -1.5 * st - st * np.arange(1, K + 1), "double": 1.5 * st + 2 * st * np.arange(1, K + 1)}
    res["follow_last_index"] = {c: ((np.array(res[c]["s"]) - s_true_med) / (last_path[c] - s_true_med)).tolist() for c in last_path}
    json.dump(res, open(OUT / f"{name}.json", "w"), indent=1)
    print(f"\n== {name} (test clip {res['n']}, 블록당 참 이동 {res['block_step_cells']:.2f} 칸)")
    print("  참 궤적 s  " + " ".join(f"{v:6.2f}" for v in res["truth_s"]))
    for c in CONDS:
        print(f"  {c:7s} s  " + " ".join(f"{v:6.2f}" for v in res[c]["s"]) + "  | Δs " + " ".join(f"{v:+5.2f}" for v in res[c]["ds"])
              + "  | 없음 " + " ".join(f"{v:.2f}" for v in res[c]["absent"]))
    print("  역사 속도 기울기 (참 궤적이면 k):")
    for c in CONDS: print(f"    {c:7s} " + " ".join(f"{v:5.2f}" for v in res[c]["slope_on_history_speed"]))
    print("  마지막 블록 따름 지수 (0 역사 · 1 마지막 블록):")
    for c, v in res["follow_last_index"].items(): print(f"    {c:7s} " + " ".join(f"{x:5.2f}" for x in v))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--ckpt", default="")
    ap.add_argument("--gpus", type=int, default=4); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--max-n", type=int, default=600)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if not a.summarize:
        import torch
        torch.multiprocessing.spawn(_worker, args=(a.gpus, a.name, a.ckpt, a.bs, a.max_n), nprocs=a.gpus, join=True)
    summarize(a.name)
