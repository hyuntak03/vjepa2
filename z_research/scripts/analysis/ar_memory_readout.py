"""AR 병목 좁히기 (2026-11-06, 사용자 "같이 생각하면서 병목을 좁혀 보자"): 물체를 봤던 문맥 (A) vs 빈 문맥 (B) 을 장면마다 짝지어,
그 차이가 (T3) encoder 블록 토큰 · (T2) AR 내부 상태 · (T1) AR 예측 에 얼마나 남아 있는지 잰다. 학습 없음.

장면 = 벤치의 블록 하나: pos_a (A 문맥 + 물체 다시 나오는 미래), pos_b (B 문맥 + 빈 미래). 둘은 같은 장면 · 같은 가림막.
T1 예측 반영도 (미래 블록 f 마다): β_f = <p_A − p_B, h_A − h_B> / ||h_A − h_B||²  (실제 미래 차이 중 예측이 옳은 방향으로 재현한 몫, 1 = 전부)
   위약: 같은 값을 다른 장면의 (h_A − h_B) 에 투영 (장면을 한 칸 밀어 짝 지음). 판정 = 실제 − 위약. 복사 기준선 (블록 8 토큰을 미래에) 도 같이.
T2 내부 기억: AR 의 굴리는 걸음마다 상태 (predictor_norm 출력, 384 차원) 를 토큰 평균해 A − B 차이 벡터를 만들고,
   다른 장면들의 평균 차이 방향으로 판정 (5 겹 교차검증) → A 문맥을 A 쪽으로 가르는 장면 비율 (우연 50 %).
T3 지각: 같은 판정을 encoder 블록 토큰 (문맥 블록 1..8, LN 뒤 토큰 평균) 에.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from scene_paired_tests import boards  # noqa: E402
from analysis.intphys2.model import build_from_config  # noqa: E402
from analysis.intphys2.surprise import _context_target_indices  # noqa: E402
from evals.world_model_analysis.data import WMADataset  # noqa: E402

S, C, K = 256, 8, 8
RES = os.path.join(ROOT, "cvpr/results/eval/testbed")
BENCH = {"real95": ("real_world_permanence_uniform", "real world (all 95)", None),
         "vob3_panel_moving": ("vobench_v3", "VOBench v3 late · panel", "moving")}
CKPT = {"ar": None, "ar_v4bc4_l1": "v4bc4_ar_vw_state_negimp"}


def probe_acc(D, folds=5, seed=0):
    """D (n, d) 장면별 A − B 차이. 다른 장면들의 평균 차이 방향 w 로 판정: w·D_i > 0 이면 맞음. 교차검증 정확도."""
    n = len(D)
    idx = np.random.default_rng(seed).permutation(n)
    ok = 0
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        w = D[tr].mean(0)
        ok += int((D[te] @ w > 0).sum())
    return ok / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="real95", choices=list(BENCH))
    ap.add_argument("--model", default="ar", choices=list(CKPT))
    a = ap.parse_args()
    ds_name, bn, mv = BENCH[a.bench]
    cfg = yaml.safe_load(open(f"{RES}/{ds_name}_vast_ar_vw_state/_resolved.yaml"))
    if CKPT[a.model]:
        cfg["model"]["predictor_checkpoint"] = os.path.join(ROOT, f"z_training/runs/{CKPT[a.model]}/latest.pt")
    blk = boards()[bn][1]
    if mv:
        import csv
        mset = {r["block_id"] for r in csv.DictReader(open(os.path.join(ROOT, f"data_csv/{ds_name}/index.csv"))) if r["motion"] != "flat_static"}
        blk = {b: v for b, v in blk.items() if (b in mset) == (mv == "moving")}
    ds = WMADataset(cfg)
    pos = {r.video_id: i for i, r in enumerate(ds.records)}
    scenes = sorted(blk)
    dev = torch.device("cuda:0")
    bundle = build_from_config(cfg["model"], dev)
    core = bundle.predictor.core
    ci, ti = _context_target_indices(ctx_frames=2 * C, tgt_frames=2 * K, tubelet_size=2, spatial_tokens=S, batch_size=2, device=dev)
    ln = lambda t: F.layer_norm(t.float(), (t.size(-1),))  # noqa: E731
    dP, dH, dCopy, dSt, dEnc = [], [], [], [], []
    for b in scenes:
        va, vb = blk[b]["A"][0], blk[b]["B"][0]            # pos_a (A 문맥 · 물체 다시 나옴), pos_b (B 문맥 · 빈 미래)
        x = torch.stack([ds.clip(pos[va]), ds.clip(pos[vb])]).to(dev, bundle.dtype)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            z = bundle.context_encoder(x, masks=[ci])                      # (2, C*S, 1280) 블록별
            zl = ln(z)
            p, st = core.rollout_kv(zl.to(z.dtype), ci, K, return_states=True)   # p (2, K*S, D) LN, st (2, K*S, 384)
            hf = bundle.target_encoder(x)
            h = ln(torch.gather(hf, 1, ti.unsqueeze(-1).expand(-1, -1, hf.size(-1))))
            copy = zl[:, -S:].repeat(1, K, 1)
        P, Hh, Cp = p.float().view(2, K, S, -1), h.float().view(2, K, S, -1), copy.float().view(2, K, S, -1)
        dP.append((P[0] - P[1]).cpu()); dH.append((Hh[0] - Hh[1]).cpu()); dCopy.append((Cp[0] - Cp[1]).cpu())
        Stt = st.float().view(2, K, S, -1).mean(2)                         # (2, K, 384)
        dSt.append((Stt[0] - Stt[1]).cpu().numpy())
        E = zl.float().view(2, C, S, -1).mean(2)                           # (2, C, 1280)
        dEnc.append((E[0] - E[1]).cpu().numpy())
    n = len(scenes)
    res = {"bench": a.bench, "model": a.model, "n_scenes": n, "T1_beta": [], "T1_placebo": [], "T1_copy": [], "T2_state_acc": [], "T3_enc_acc": []}
    for f in range(K):
        num = lambda X, Y: float(sum((x[f] * y[f]).sum() for x, y in zip(X, Y)))   # noqa: E731
        den = float(sum((y[f] ** 2).sum() for y in dH))
        res["T1_beta"].append(num(dP, dH) / den)
        res["T1_placebo"].append(num(dP, dH[1:] + dH[:1]) / float(sum((y[f] ** 2).sum() for y in dH[1:] + dH[:1])))
        res["T1_copy"].append(num(dCopy, dH) / den)
        res.setdefault("T1_mag", []).append(float(np.sqrt(sum(float((x[f] ** 2).sum()) for x in dP) / den)))   # 방향 무관 크기 비 ||p_A − p_B|| / ||h_A − h_B||
        res["T2_state_acc"].append(probe_acc(np.stack([d[f] for d in dSt])))
    for t in range(C):
        res["T3_enc_acc"].append(probe_acc(np.stack([d[t] for d in dEnc])))
    out = os.path.join(ROOT, f"z_analysis/results/release_ladder_20261106/ar_memory_readout_{a.bench}_{a.model}.json")
    json.dump(res, open(out, "w"), indent=1)
    print(f"[{a.bench} · {a.model}] 장면 {n}")
    print("T3 encoder 블록 1..8 판별 %: " + " ".join(f"{100*v:.0f}" for v in res["T3_enc_acc"]))
    print("T2 상태 (걸음 → 블록 9..16) 판별 %: " + " ".join(f"{100*v:.0f}" for v in res["T2_state_acc"]))
    print("T1 예측 반영도 β (블록 9..16): " + " ".join(f"{v:.2f}" for v in res["T1_beta"]))
    print("T1 위약              : " + " ".join(f"{v:.2f}" for v in res["T1_placebo"]))
    print("T1 실제 − 위약       : " + " ".join(f"{a_ - b_:.2f}" for a_, b_ in zip(res["T1_beta"], res["T1_placebo"])))
    print("T1 복사 기준선 β     : " + " ".join(f"{v:.2f}" for v in res["T1_copy"]))
    print("T1 크기 비 (방향 무관): " + " ".join(f"{v:.2f}" for v in res["T1_mag"]))


if __name__ == "__main__":
    main()
