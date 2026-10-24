#!/usr/bin/env python3
"""[P] 경쟁 가설 (b)(c) — 모델 간 비교: 운반 이득이 높은 모델일수록 같은 칸 평균이 덜 퍼졌나 (auto_research, 2026-10-21).

  /venv/vjepa/bin/python auto_research/scripts/pm_compare.py

입력: auto_research/results/pred_mean/*.json (pred_mean_solution.py 산출물). 출력: 표 + 상관 (Spearman) → results/pm_compare.json.
지표 (모두 TF 한 걸음, 3D 합성 검증 세트):
  gain_tf / gain_ro       운반 이득 중앙값 (같은 읽기)
  far_w0, far_H           먼 칸 같은 칸 가중합의 마지막 블록 몫 · 가중치 엔트로피 (정규화 |w|, 0 = 한 블록, 1 = 8 블록 고르게)
  far_R2                  그 가중합이 모델 출력을 설명하는 비율
  new_beta / true_beta    물체가 올 칸에서 "옮긴 복사" 회귀 계수 (모델 / 정답)
  new_R2_M0               물체가 올 칸에서 같은 칸 가중합만으로의 R²
  pres_new / trail_old    존재감 (1 = 정답처럼) / 잔상 (1 = 옛 자리 그대로)
  sobj_w0                 정지 물체 칸 가중합의 마지막 블록 몫"""
import glob, json, os
import numpy as np
D = os.path.join(os.path.dirname(__file__), "../results/pred_mean")
rows = []
for f in sorted(glob.glob(os.path.join(D, "*.json"))):
    s = json.load(open(f)); tf = s["tf"]
    if "mov/far" not in tf or "gain" not in s:
        continue
    wf = np.abs(np.array(tf["mov/far"]["pred"]["M0_w"])); pf = wf / wf.sum()
    r = dict(name=os.path.basename(f)[:-5],
             gain_tf=s["gain"]["tf/model"]["median"], gain_ro=s["gain"]["ro/model"]["median"],
             far_w0=tf["mov/far"]["pred"]["M0_w0_share"], far_H=float(-(pf * np.log(pf + 1e-12)).sum() / np.log(len(pf))),
             far_R2=tf["mov/far"]["pred"]["M0_R2"], far_sum=tf["mov/far"]["pred"]["M0_sum"],
             new_beta=tf["mov/new"]["pred"]["M1_w"][8], true_beta=tf["mov/new"]["true"]["M1_w"][8], new_R2_M0=tf["mov/new"]["pred"]["M0_R2"],
             old_beta=tf["mov/old"]["pred"]["M1_w"][8],
             pres_new=s["presence_tf"]["new"]["pred"], trail_old=s["presence_tf"]["old"]["pred"],
             sobj_w0=tf["sta/sobj"]["pred"]["M0_w0_share"] if "sta/sobj" in tf else np.nan,
             l1_far_mean8=tf["mov/far"]["l1_pred_to"]["mean_8"], l1_far_copy=tf["mov/far"]["l1_pred_to"]["copy"])
    rows.append(r)
keys = ["gain_tf", "gain_ro", "far_w0", "far_H", "far_R2", "far_sum", "new_beta", "true_beta", "new_R2_M0", "old_beta", "pres_new", "trail_old", "sobj_w0"]
print(f"{'model':58s} " + " ".join(f"{k:>9s}" for k in keys))
for r in rows:
    print(f"{r['name'][:58]:58s} " + " ".join(f"{r[k]:9.3f}" for k in keys))


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


out = {"rows": rows, "corr": {}}
if len(rows) >= 4:
    g = np.array([r["gain_tf"] for r in rows])
    print(f"\nSpearman 상관 (n = {len(rows)} 모델) — 운반 이득 (TF) 과")
    for k in keys[1:]:
        v = np.array([r[k] for r in rows])
        if np.isfinite(v).all():
            out["corr"][k] = spearman(g, v)
            print(f"  {k:10s} {out['corr'][k]:+.2f}")
json.dump(out, open(os.path.join(D, "../pm_compare.json"), "w"), indent=1)
