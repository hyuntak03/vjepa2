"""위치 판독 R² 는 "마지막에 본 위치" 가 거의 다 설명한다 — 직관판 (점 그래프 + 막대) (2026-10-26 저녁).
사용자: "position 그래프가 의미 없다는 걸 증명하는 figure 하나 따로" → "그래프를 직관적으로".
입력: z_analysis/results/position_readout_meaningless.json (position_readout_meaningless.py) · ar_layers val_meta (점 그래프용 물체 중심).
  (a) 점 = val 표본 (걸음 2–8, 블록 7 · b 물체 보임): 가로 = 마지막에 본 x (블록 7), 세로 = 실제 x (블록 b). 대각선에 몰리면 물체가 거의 안 움직인 것
  (b) 위치 R², 걸음 2–8 평균: encoder probe → encoder · AR probe → AR 예측 · 모델 없이 마지막에 본 위치만
  (c) 실제 움직임 중 따라 움직인 비율 (2026-10-26 사용자: "(c) 는 이해하기 힘들다" → R² 대신): 실제로 16 px 이상 움직인 표본에서
      Σ(예측 변위 · 실제 변위) / Σ|실제 변위|², 걸음 1–8. 100 % = 실제만큼, 0 % = 제자리 (마지막 위치 복사는 정의상 0 %)
  걸음 1 은 이동이 너무 작아 (중앙값 6.5 px) 뺐다.
  /venv/vjepa/bin/python z_research/scripts/figures/plot_position_readout_meaningless.py
출력: z_analysis/figures/position_readout_meaningless.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/workspace/world_model/vjepa2/z_analysis"
J = json.load(open(f"{R}/results/position_readout_meaningless.json"))
meta = json.load(open("/workspace/auto_research_cache/ar_layers/val_meta.json"))
COL = {"enc": "#2a78d6", "ar": "#eb6834", "base": "#888888"}
plt.rcParams["font.family"] = "DejaVu Sans"
fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1.1, 1, 1.4]})

# (a) 마지막에 본 x vs 실제 x
ax = axes[0]; xs, ys = [], []
for m in meta:
    if not m["blocks"][7]["ok"]:
        continue
    for b in range(9, 16):
        if m["blocks"][b]["ok"]:
            xs.append(m["blocks"][7]["ctr"][0]); ys.append(m["blocks"][b]["ctr"][0])
ax.scatter(xs, ys, s=8, color="#2a78d6", alpha=0.35, lw=0)
ax.plot([0, 256], [0, 256], color="black", lw=1, ls="--", label="no movement (y = x)")
ax.set_xlim(0, 256); ax.set_ylim(0, 256); ax.set_aspect("equal")
ax.set_xlabel("last-seen x position (px)\n(a)"); ax.set_ylabel("actual x position, steps 2–8 (px)")
ax.set_title("objects stay close to where they were last seen", fontsize=11); ax.legend(loc="upper left", fontsize=9, frameon=False)

# (b) 위치 R², 걸음 2–8 평균
ax = axes[1]; steps = [str(k) for k in range(2, 9)]
vals = [np.mean([J["position_r2"]["encoder_probe_on_encoder"][k] for k in steps]), np.mean([J["position_r2"]["ar_probe_on_ar_prediction"][k] for k in steps]),
        np.mean([J["position_r2"]["baseline_last_seen_only"][k] for k in steps])]
names = ["encoder\n(real video)", "AR prediction", "copy last-seen\nposition (uses no video)"]
bars = ax.bar(range(3), vals, color=[COL["enc"], COL["ar"], COL["base"]], width=0.6)
for b_, v in zip(bars, vals):
    ax.text(b_.get_x() + b_.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=9); ax.set_ylim(0, 1.05); ax.set_ylabel("position R² (mean over steps 2–8)")
ax.set_xlabel("(b)"); ax.set_title("position R²: copying the last-seen position does as well", fontsize=11)

# (c) 따라 움직인 비율, 걸음별
ax = axes[2]; ks = np.arange(1, 9)
for key, name, col, ls, mk, dx in (("encoder_probe_on_encoder", "encoder (real video)", COL["enc"], "--", "o", -0.06), ("ar_probe_on_ar_prediction", "AR prediction", COL["ar"], "-", "s", 0.06)):
    v = np.array([J["follow_ratio"][key][str(k)]["ratio"] for k in ks]) * 100
    ax.plot(ks + dx, v, ls=ls, marker=mk, color=col, lw=2, ms=6, label=name)
    for k, vv in zip(ks, v):
        ax.text(k + dx, vv + (3 if key.startswith("encoder") else -7), f"{vv:.0f}", ha="center", fontsize=8, color=col)   # 위 · 아래로 나눠 겹침 방지
ax.axhline(100, color="black", ls=":", lw=0.8); ax.axhline(0, color="#888888", lw=1.5, label="copy last-seen position (= 0 %)")
ax.set_xticks(ks); ax.set_xticklabels([f"{k}\n({0.5 * k:.1f} s)" for k in ks], fontsize=8); ax.set_ylim(-5, 115)
ax.set_ylabel("share of the actual movement reproduced (%)"); ax.set_xlabel("prediction step (time ahead)\n(c)")
ax.set_title("how far the predicted object moves vs the real one", fontsize=11); ax.legend(loc="lower left", fontsize=9, frameon=False)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Why position R² is not informative (attentive probe on 1280-d block tokens, val)", fontsize=12, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(f"{R}/figures/position_readout_meaningless.png", dpi=170, bbox_inches="tight")
print("saved", [round(v, 2) for v in vals])
