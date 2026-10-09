#!/usr/bin/env python3
"""H6b — IntPhys 1 Garrido 창에서 **예측 없이도** 판별이 되는가 (튜블릿 단위).

H6 결과: 이동+보임 칸은 사건이 문맥 끝에서 멀어도 92~95 %, 이동+가림 칸만 멀수록 무너진다.
가설: 보이는 사건은 불가능 영상의 **불연속 자체** (모양 바뀜·순간이동) 가 target 쪽 표현에 드러나서,
predictor 가 상태를 이어 가지 않아도 잡힌다.  → 예측을 쓰지 않는 기준선이 같은 칸에서 같이 높으면 지지.

창 (Garrido A.8 격자, 공식 채점과 같은 창 생성기 `_intphys1_windows`, frame_budget official, stride 2):
  combo ∈ {skip2_w16, skip2_w32, skip5_w16},  C ∈ {W/16·2, …, W/16·10}
미래 튜블릿 j 마다 (전 토큰 평균 L1, 표적 = LN(target_encoder(창 전체))[미래 j]):
  p      predictor(context_encoder(문맥))                 ← 표준 채점 (합치면 per_window.json 과 같아야 한다: 검증)
  copyh  LN(h)[문맥 마지막 튜블릿] 을 모든 미래에 복사      ← 예측 없음 (마지막 관측 그대로)
  copyz  LN(z)[문맥 마지막 튜블릿] 복사                    ← 예측 없음 (predictor 가 받는 입력 그대로)
  zero   0 (= mean|LN(h)|)                                 ← 표적의 크기만

저장: exp_results/h6/tubelet_l1_<model>.npz  — video, combo, start, C, j → 4 값.

  srun -w vll6 --gres=gpu:1 python auto_research/scripts/h6b_intphys1_baselines.py
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arlib  # noqa: E402
from evals.world_model_analysis.eval import _intphys1_windows  # noqa: E402

S_TOK, D = 256, 1280
OUT = arlib.ROOT / "auto_research/exp_results/h6"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    base = arlib.load_cfg("intphys1_sliding__intphys1_dev_vith")
    cfg = dict(base, model=dict(base["model"], window_size=64))
    ds = arlib.WMADataset(cfg)
    bundle = arlib.build_bundle(cfg, dev, window=64)
    W = base["surprise"]["intphys1"]
    combos = []
    for sk in (2, 5):
        for wz in (16, 32):
            cl = [m * wz // 16 for m in W["context_mult"]]
            wins = _intphys1_windows(ds.n_frames, sk, wz, cl, int(W["stride"]), 2, str(W["frame_budget"]))
            if wins:
                combos.append((f"skip{sk}_w{wz}", wins))
    print({c: len(w) for c, w in combos}, flush=True)
    idx = list(range(a.rank, len(ds), a.world))
    if a.limit:
        idx = idx[: a.limit]
    rows = []            # (vid_i, combo_i, start, C, j, p, copyh, copyz, zero)
    t0 = time.time()
    for n_done, i in enumerate(idx):
        clip = ds.clip(i)
        for ci_, (name, wins) in enumerate(combos):
            by_start = {}
            for C, cf, tf in wins:
                by_start.setdefault(cf[0], []).append((C, cf, tf))
            for st, lst in sorted(by_start.items()):
                frames = lst[0][1] + lst[0][2]
                wz = len(frames)
                x = clip[:, frames][None]
                ac = arlib.autocast_ctx(cfg)
                with torch.inference_mode(), ac():
                    hw = bundle.target_encoder(x.to(dev, bundle.dtype))
                    hw = F.layer_norm(hw, (hw.size(-1),)).float().reshape(1, wz // 2, S_TOK, D)
                    for C, cf, tf in lst:
                        Tc = C // 2
                        ci, ti = arlib.ci_ti(1, C, wz - C, dev)
                        z = bundle.context_encoder(x.to(dev, bundle.dtype), masks=[ci])
                        p = bundle.predictor(z, ci, ti, mask_index=0).float().reshape(1, -1, S_TOK, D)
                        hf, hT = hw[:, Tc:], hw[:, Tc - 1:Tc]
                        zT = F.layer_norm(z.float(), (D,)).reshape(1, Tc, S_TOK, D)[:, Tc - 1:Tc]
                        lp = (p - hf).abs().mean((-1, -2))[0]
                        lh = (hT - hf).abs().mean((-1, -2))[0]
                        lz = (zT - hf).abs().mean((-1, -2))[0]
                        l0 = hf.abs().mean((-1, -2))[0]
                        for j in range(hf.shape[1]):
                            rows.append((i, ci_, st, C, j, lp[j].item(), lh[j].item(), lz[j].item(), l0[j].item()))
        if n_done % 10 == 0:
            el = time.time() - t0
            print(f"[{a.rank}] {n_done+1}/{len(idx)}  {el/60:.1f} 분  남은 {el/(n_done+1)*(len(idx)-n_done-1)/60:.1f} 분", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    R = np.array(rows, dtype=np.float64)
    np.savez(OUT / f"tubelet_l1_vith_r{a.rank}.npz", rows=R, cols=np.array(
        ["video_i", "combo_i", "start", "C", "j", "p", "copyh", "copyz", "zero"]),
        combos=np.array([c for c, _ in combos]), video_ids=np.array([r.video_id for r in ds.records]))
    print("DONE", len(rows), flush=True)


if __name__ == "__main__":
    main()
