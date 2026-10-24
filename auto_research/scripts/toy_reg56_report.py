#!/usr/bin/env python3
"""[T] R5 · R6 장난감 표 (CPU): 두 세계 × 5 팔 → 조건화 갈림 · 종류별 계수 · 먼 칸 잡음 · TF 손실 (auto_research, 2026-10-22 KST).
  /venv/vjepa/bin/python auto_research/scripts/toy_reg56_report.py"""
import json, os
D = os.path.join(os.path.dirname(__file__), "../results/toy_reg56")
ARMS = [("c", "대조 L1"), ("r5a", "R5 λ 0.1"), ("r5b", "R5 λ 0.5"), ("r6a", "R6 λ 0.5"), ("r6b", "R6 λ 2")]
out = {}
for world, pre in (("단단함 / 무름 10 %", "rs10"), ("관성 / 제멋대로 10 %", "in10")):
    print(f"== {world}")
    for arm, nm in ARMS:
        p = os.path.join(D, f"{pre}_{arm}.json")
        if not os.path.exists(p):
            print(f"  {nm:10s} (없음)"); continue
        j = json.load(open(p)); b = j["branch"]; h = j["hist"][-1]
        if pre == "rs10":
            a_, z_ = b.get("beta_model_rigid"), b.get("beta_model_soft"); split = a_ - z_
        else:
            a_, z_ = b.get("cont_model_inert"), b.get("cont_model_erratic"); split = b.get("split_cont")
        far = b.get("far", {})
        r = dict(keep=a_, other=z_, split=split, far_p_minus_cm=far.get("p_minus_cm"), far_err=far.get("p_err_clean"), far_cm_err=far.get("cm_err_clean"),
                 tf1=h.get("m_tf"), pres_new=h.get("pres_tf"), bg_ratio=h.get("bg_ratio"))
        out[f"{pre}_{arm}"] = r
        print(f"  {nm:10s} 계수 (옮길 것 · 나머지) {a_:.3f} · {z_:.3f} → 갈림 {split:+.3f} | 먼 칸 |예측 − 문맥 평균| {far.get('p_minus_cm', float('nan')):.4f} · "
              f"오차 {far.get('p_err_clean', float('nan')):.4f} (문맥 평균 {far.get('cm_err_clean', float('nan')):.4f}) | TF 손실 (움직인 clip) {h.get('m_tf', float('nan')):.4f} · 새 칸 존재감 {h.get('pres_tf', float('nan')):.2f}")
json.dump(out, open(os.path.join(D, "report.json"), "w"), indent=1, ensure_ascii=False)
