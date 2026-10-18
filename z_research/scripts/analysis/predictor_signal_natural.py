#!/usr/bin/env python3
"""자연 영상 (SSv2 val) 에서 학습 목표가 '변한 것' 에 신호를 얼마나 주나 (2026-10-12) — release · AR ep40 · AR-state ep40.

AR 둘은 SSv2 (+ K400) 로 학습했다 — 이 판은 그들의 학습 분포 안이다. release 는 VideoMix22M.
클립: SSv2 val, 원본 12 fps 그대로 (frame_step 1), 앞 32 장 → 16 블록, 문맥 C = 8 · 예측 K = 8. 256 리사이즈 + 가운데 자름.
걸음 k 마다 토큰 오차 e = mean_d |p − h| (AR 은 블록별 LN(h), release 는 창 LN(h)), 복사 오차 c = mean_d |h_{C−1} − h| (같은 공간).
'변한 토큰' = 그 걸음에서 c 가 클립 안 상위 10 % 인 토큰 (마지막 관측 뒤 가장 많이 바뀐 자리). 나머지 = '정적'.
보고: 복사 오차 중 변한 토큰 몫 · 모델이 복사를 이기는 몫 (c − e) 이 변한 / 정적 토큰에서 각각 얼마인가 · AR 은 roll 과 tf 둘 다.

  python predictor_signal_natural.py --name ar40 --ckpt z_training/ariel/ar_future_1e_5/latest.pt --gpus 4 --n 800
산출: z_research/VOBench_v1/bottleneck/natural_<name>.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
S, NB, C, K, D, NF = 256, 16, 8, 8, 1280, 32
OUT = REPO / "z_research/VOBench_v1/bottleneck"
VAL = REPO / "data_csv/ssv2/val_min48.csv"
TOPQ = 0.10


def items(n, seed=0):
    rows = [r for r in csv.reader(open(VAL), delimiter=" ")]
    rng = np.random.default_rng(seed); rng.shuffle(rows)
    return [r[0] for r in rows[:n]]


def _worker(rank, world, name, ckpt, n, bs, q):
    import decord
    import torch
    import torch.nn.functional as F
    from analysis.intphys2.dataset import build_video_transform
    from analysis.intphys2.model import build_from_config
    from app.vjepa_frozen.ar import block_indices, encode_blocks
    torch.cuda.set_device(rank); torch.set_num_threads(4); dev = torch.device("cuda", rank)
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("ir", REPO / "z_research/scripts/analysis/rollout2_identity_readout.py")
    ir = module_from_spec(spec); spec.loader.exec_module(ir)
    model = dict(ir.rp.MODEL) if name == "release" else dict(ir.rp.MODEL, predictor_checkpoint=str(REPO / ckpt))
    bundle = build_from_config(model, dev); pred = bundle.predictor
    is_ar = getattr(pred, "mask_mode", "release") in ("ar", "ar_state")
    tf_ = build_video_transform(256, 1.0)
    files = items(n); mine = files[rank * len(files) // world:(rank + 1) * len(files) // world]
    LN = lambda x: F.layer_norm(x.float(), (x.size(-1),))
    acc = {f"{c}_{m}{k}": [] for c in (("roll", "tf") if is_ar else ("roll",)) for m in ("e_chg", "e_sta", "c_chg", "c_sta") for k in range(1, K + 1)}
    acc.update({f"copyshare{k}": [] for k in range(1, K + 1)})

    def read(f):
        try:
            vr = decord.VideoReader(f, num_threads=1)
            if len(vr) < NF: return None
            return tf_(vr.get_batch(list(range(NF))).asnumpy())                      # (3, T, 256, 256)
        except Exception:
            return None

    t0 = time.time(); done = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b0 in range(0, len(mine), bs):
            xs = [v for v in (read(f) for f in mine[b0:b0 + bs]) if v is not None]
            if not xs: continue
            x = torch.stack(xs).to(dev, dtype=bundle.dtype); B = x.size(0)
            if is_ar:
                hb = encode_blocks(bundle.target_encoder, x, 2)
                roll = pred.rollout(hb[:, :C * S], block_indices(B, C, S, dev), K)
                tfo = pred.forward_seq(hb[:, :(NB - 1) * S], block_indices(B, NB - 1, S, dev))[:, (C - 1) * S:]
                preds = {"roll": roll, "tf": tfo}; tgt = hb[:, C * S:]; copy = hb[:, (C - 1) * S:C * S]
            else:
                from analysis.intphys2.surprise import _context_target_indices
                ci, ti = _context_target_indices(ctx_frames=2 * C, tgt_frames=2 * K, tubelet_size=2, spatial_tokens=S, batch_size=B, device=dev)
                z = bundle.context_encoder(x, masks=[ci]); z = z[-1] if isinstance(z, list) else z
                p = pred(z, ci, ti, mask_index=0)
                h = bundle.target_encoder(x); h = LN(h[-1] if isinstance(h, list) else h)
                preds = {"roll": p}; tgt = h[:, C * S:]; copy = h[:, (C - 1) * S:C * S]
            for k in range(1, K + 1):
                hk = tgt[:, (k - 1) * S:k * S].float()
                cerr = (copy.float() - hk).abs().mean(-1)                                  # (B, S)
                thr_ = torch.quantile(cerr, 1 - q, dim=1, keepdim=True); chg = cerr >= thr_
                acc[f"copyshare{k}"].append(((cerr * chg).sum(1) / cerr.sum(1)).cpu().numpy())
                for c, pk in preds.items():
                    e = (pk[:, (k - 1) * S:k * S].float() - hk).abs().mean(-1)
                    acc[f"{c}_e_chg{k}"].append(((e * chg).sum(1) / chg.sum(1)).cpu().numpy())
                    acc[f"{c}_e_sta{k}"].append(((e * ~chg).sum(1) / (~chg).sum(1)).cpu().numpy())
                    acc[f"{c}_c_chg{k}"].append(((cerr * chg).sum(1) / chg.sum(1)).cpu().numpy())
                    acc[f"{c}_c_sta{k}"].append(((cerr * ~chg).sum(1) / (~chg).sum(1)).cpu().numpy())
            done += B
            if rank == 0 and (b0 // bs) % 10 == 0:
                print(f"[r0] {name} {done}/{len(mine)} ({time.time() - t0:.0f}s)", flush=True)
    np.savez(OUT / f"_natural_{name}_r{rank}.npz", **{k: np.concatenate(v) for k, v in acc.items() if v})


def summarize(name):
    Z = {}
    for f in sorted(OUT.glob(f"_natural_{name}_r*.npz")):
        z = np.load(f)
        for k in z.files: Z.setdefault(k, []).append(z[k])
    Z = {k: np.concatenate(v) for k, v in Z.items()}
    res = {"n_clips": int(len(Z["copyshare1"])), "top_q": TOPQ, "copy_error_share_in_changed": [float(Z[f"copyshare{k}"].mean()) for k in range(1, K + 1)]}
    for c in ("roll", "tf"):
        if f"{c}_e_chg1" not in Z: continue
        rows = []
        for k in range(1, K + 1):
            ec, es, cc, cs = (Z[f"{c}_{m}{k}"].mean() for m in ("e_chg", "e_sta", "c_chg", "c_sta"))
            gain_chg = (cc - ec) * TOPQ; gain_sta = (cs - es) * (1 - TOPQ)                  # 토큰 수 가중 (클립 평균 손실에 대한 기여)
            rows.append(dict(k=k, err_changed=float(ec), copy_changed=float(cc), err_static=float(es), copy_static=float(cs),
                             gain_changed=float(gain_chg), gain_static=float(gain_sta),
                             frac_gain_from_changed=float(gain_chg / (gain_chg + gain_sta)) if (gain_chg + gain_sta) != 0 else None,
                             rel_gain_changed=float((cc - ec) / cc), rel_gain_static=float((cs - es) / cs)))
        res[c] = rows
    json.dump(res, open(OUT / f"natural_{name}.json", "w"), indent=1)
    print(f"\n== {name} SSv2 val {res['n_clips']} clip — 변한 토큰 = 복사 오차 상위 {int(TOPQ*100)} %")
    print("복사 오차 중 변한 토큰 몫  " + " ".join(f"{v:6.3f}" for v in res["copy_error_share_in_changed"]))
    for c in ("roll", "tf"):
        if c not in res: continue
        for m in ("err_changed", "copy_changed", "rel_gain_changed", "err_static", "copy_static", "rel_gain_static", "frac_gain_from_changed"):
            print(f"{c:4s} {m:24s} " + " ".join(f"{r[m]:6.3f}" if r[m] is not None else "     ·" for r in res[c]))
    for f in OUT.glob(f"_natural_{name}_r*.npz"): f.unlink()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--ckpt", default="")
    ap.add_argument("--gpus", type=int, default=4); ap.add_argument("--n", type=int, default=800); ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    import torch
    torch.multiprocessing.spawn(_worker, args=(a.gpus, a.name, a.ckpt, a.n, a.bs, TOPQ), nprocs=a.gpus, join=True)
    summarize(a.name)
