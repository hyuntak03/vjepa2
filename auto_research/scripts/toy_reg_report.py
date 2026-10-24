#!/usr/bin/env python3
"""[E] 정규화 장난감 판정표 (auto_research, 2026-10-21).

  /venv/vjepa/bin/python auto_research/scripts/toy_reg_report.py

results/toy_reg/<팔>.json 을 모아 사전 등록 판정 (1) 갈림 ≥ 0.3 · (2) 무름 ≤ 정답 + 0.1 이고 먼 칸 오차 ≤ 기준 × 1.1 · (3) Δz 잡음 R² 을 낸다."""
import glob, json, os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/toy_reg")
R = {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob(os.path.join(D, "*.json"))) if not os.path.basename(f).startswith("_")}
base = R["none"]["branch"]
rows = []
order = ["none", "none_r50"] + sorted(k for k in R if k not in ("none", "none_r50"))
print(f"{'팔':12s} {'단단함':>7s} {'무름':>6s} {'갈림':>6s} {'정답무름':>7s} {'먼칸오차':>8s} {'(÷기준)':>7s} {'먼칸진폭':>8s} {'물체진폭':>8s} {'Δz잡음R²':>8s} {'Δz깨끗R²':>8s} {'TF L1':>7s} {'운반이득':>7s} {'존재감':>6s}  판정")
for k in order:
    if k not in R:
        continue
    b, h = R[k]["branch"], R[k]["hist"][-1]
    split = b["beta_model_rigid"] - b["beta_model_soft"]
    c1 = split >= 0.3
    c2 = (b["beta_model_soft"] <= b["beta_true_soft"] + 0.1) and (b["bg_err"] <= base["bg_err"] * 1.1)
    verdict = ("통과" if c1 and c2 else ("(1)만" if c1 else ("(2)만" if c2 else "둘 다 ✗"))) if not k.startswith("none") else "기준"
    if k.startswith("x_"):
        verdict += " (탐색)"
    rows.append(dict(arm=k, rigid=b["beta_model_rigid"], soft=b["beta_model_soft"], split=split, soft_true=b["beta_true_soft"], rigid_true=b["beta_true_rigid"],
                     n_rigid=b["n_rigid"], bg_err=b["bg_err"], bg_rel=b["bg_err"] / base["bg_err"], bg_amp=b["bg_amp"], obj_amp=b["obj_amp"],
                     dz_r2_noise=b["dz_R2_noise"], dz_r2_clean=b["dz_R2_clean"], dz_std=b["dz_std_mean"], tf_l1=h["m_tf"], g_tf=h["g_tf"], pres=h["pres_tf"],
                     c1=c1, c2=c2, verdict=verdict, dz_far_over_obj=b.get("dz_far_over_obj"), dz_far=b.get("dz_far"), dz_obj=b.get("dz_obj")))
    r = rows[-1]
    print(f"{k:12s} {r['rigid']:7.3f} {r['soft']:6.3f} {r['split']:6.3f} {r['soft_true']:7.3f} {r['bg_err']:8.4f} {r['bg_rel']:7.3f} {r['bg_amp']:8.4f} {r['obj_amp']:8.4f} "
          f"{r['dz_r2_noise']:8.3f} {r['dz_r2_clean']:8.3f} {r['tf_l1']:7.4f} {r['g_tf']:7.2f} {r['pres']:6.2f}  {verdict}"
          + (f"  | Δz 에너지 먼 칸 {r['dz_far']:.2f} · 물체 칸 {r['dz_obj']:.2f} (비 {r['dz_far_over_obj']:.3f})" if r["dz_far"] is not None else ""))
json.dump(rows, open(os.path.join(D, "_report.json"), "w"), indent=1, ensure_ascii=False)
