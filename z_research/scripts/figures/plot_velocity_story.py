"""속도 관련 분석 그림 묶음 → z_analysis/figures/velocity/ (2026-10-27).
사용자: "폴더 새로 만들어서 이와 관련된 분석들 figure 좀 그려줘".
그림 (이야기 순서):
  1_velocity_inside_ar        블록 AR 안의 속도: 은닉 층별 (입력 블록의 속도) · 출력 (다음 블록 속도, encoder 판독기 / 자기 판독기)
  2_training_data_persistence 학습 영상의 지속률 (데이터셋 · SSv2 부류 · 답을 아는 검사)
  3_ar_output_vs_data_natural 학습 영상에서 AR 출력이 담은 다음 움직임 방향 vs 데이터 (K710 · SSv2 부류)
  4_training_data_dose        학습 데이터에 따른 출력 속도 · 굴림 따라 움직인 비율 (AR 세 판 · release 두 판)
  5_output_velocity_form      출력 속도의 꼴: encoder 판독기 vs 자기 판독기
  6_low_variance              encoder 벡터에서 속도 · 색 · 모양 · 위치의 분산 몫 · 주성분 판독 · 차원별 분리도
  7_median_test_readout       섞인 벡터의 차원별 중앙값 — 판독기에 따라 답이 달라진다 (정정 기록)
입력: z_analysis/results/ 의 attnprobe384/tf_L*.json · tf_output_velocity · motion_predictability_flow · ssv2_class_motion · ar_output_motion_natural ·
      phys_only_ar_velocity · phys_mix_ar_velocity · release_velocity · release_vobench_pft_velocity · velocity_lowrank_median · velocity_dim_snr · median_test_matched_readout
  /venv/vjepa/bin/python z_research/scripts/figures/plot_velocity_story.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/workspace/world_model/vjepa2/z_analysis"; OUT = f"{R}/figures/velocity"; os.makedirs(OUT, exist_ok=True)
J = lambda f: json.load(open(f"{R}/results/{f}"))
BLUE, ORANGE, GREEN, GRAY, DARK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a8a", "#333333"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=170, bbox_inches="tight"); plt.close(fig); print("saved", name)


def label_bars(ax, bars, fmt="{:.2f}", dy=0.015):
    for b in bars:
        v = b.get_height(); v = 0.0 if abs(v) < 0.005 else v                          # "-0.00" 방지
        ax.text(b.get_x() + b.get_width() / 2, max(v, 0) + dy, fmt.format(v), ha="center", va="bottom", fontsize=8)


# 1 ---------------------------------------------------------------- 블록 AR 안의 속도
tf = [J(f"attnprobe384/tf_L{l}.json")["val_all"]["velocity_r2"] for l in range(14)]
tov = J("tf_output_velocity.json")["all_blocks_2_15"]; own = J("phys_only_ar_velocity.json")["own_probe"]["natural"]["velocity_r2"]
fig, ax = plt.subplots(figsize=(9.5, 4.2))
x = np.arange(14); ax.plot(x, tf, "-o", color=BLUE, lw=2, ms=4, label="hidden state at the input block (velocity of that block)")
ax.bar([15.2], [tov["tf_output"]["velocity_r2"]], 0.7, color=ORANGE, label="output = next-block prediction, encoder-trained probe")
ax.bar([16.2], [own], 0.7, color=GREEN, label="output = next-block prediction, probe trained on AR outputs")
ax.text(15.2, tov["tf_output"]["velocity_r2"] + 0.02, f"{tov['tf_output']['velocity_r2']:.2f}", ha="center", fontsize=9)
ax.text(16.2, own + 0.02, f"{own:.2f}", ha="center", fontsize=9)
ax.axhline(tov["encoder"]["velocity_r2"], color=GRAY, ls="--", lw=1.5, label=f"encoder tokens (upper bound, {tov['encoder']['velocity_r2']:.2f})")
ax.set_xticks(list(x) + [15.2, 16.2]); ax.set_xticklabels(["input"] + [str(i) for i in range(1, 13)] + ["norm", "out", "out"], fontsize=8)
ax.set_ylim(0, 1.05); ax.set_ylabel("within-block velocity R²"); ax.set_xlabel("predictor layer (teacher forcing)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8, frameon=False)
fig.suptitle("Velocity inside the block AR (3D synthetic val, attentive probes)", fontsize=12, y=1.0)
save(fig, "1_velocity_inside_ar")

# 2 ---------------------------------------------------------------- 학습 영상 지속률
mp = J("motion_predictability_flow.json"); sc = J("ssv2_class_motion.json")["persistence"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.2]})
w = 0.26; cols = [BLUE, ORANGE, GREEN]; names = ["0.25 s later", "0.5 s later (next block)", "0.5 s later, camera motion removed"]
for ax, items, xl in ((axes[0], [("Kinetics-710", mp["k710"]), ("SSv2", mp["ssv2"]), ("HowTo100M 1%", mp["howto"])], "training datasets\n(a)"),
                      (axes[1], [("objects moving\non their own", sc["passive"]), ("hand moves\nthe object", sc["actor"]), ("other\nSSv2 classes", sc["other"])], "SSv2 action groups\n(b)")):
    for i, (key, nm) in enumerate(zip((("raw_T2", "between_lag1"), ("raw_T2", "adv"), ("cam_T2", "adv")), names)):
        if "raw_T2" in items[0][1]:
            v = [it[1][key[0]][key[1]]["r"] for it in items]
        else:
            v = [it[1]["raw" if key[0] == "raw_T2" else "cam"]["lag_0.25s" if key[1] == "between_lag1" else "lag_0.5s"] for it in items]
        label_bars(ax, ax.bar(np.arange(3) + (i - 1) * w, v, w, color=cols[i], label=nm))
    ax.set_xticks(range(3)); ax.set_xticklabels([it[0] for it in items], fontsize=9); ax.set_xlabel(xl)
ax = axes[2]; P = mp["check_synthetic_patch"]; V = mp["check_vobench_visible"]
chk = [("constant-\nvelocity patch", np.mean([P[f"constant_{s}px"]["adv"]["r"] for s in (4, 8, 16)])), ("stopping\npatch", np.mean([P[f"stop_{s}px"]["adv"]["r"] for s in (4, 8, 16)])),
       ("VOBench\nvisible (5)", np.mean([V[m]["raft"]["adv"]["r"] for m in V])), ("3D synthetic\n(labels)", mp["phys3d_gt"]["T2"]["r"])]
label_bars(ax, ax.bar(range(4), [c[1] for c in chk], 0.6, color=[GRAY, GRAY, BLUE, DARK]))
ax.set_xticks(range(4)); ax.set_xticklabels([c[0] for c in chk], fontsize=9); ax.set_xlabel("known-answer checks, 0.5 s later\n(c)")
for ax in axes:
    ax.set_ylim(-0.03, 1.12); ax.axhline(1, color="black", ls=":", lw=0.8); ax.axhline(0, color="black", lw=0.6)
axes[0].set_ylabel("share of current motion that continues\n(median over moving cells)")
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.4, -0.06))
fig.suptitle("How much motion continues in the training videos (RAFT flow, 16×16 cells, 4 fps)", fontsize=12, y=1.0)
fig.tight_layout(rect=(0, 0.04, 1, 0.96)); save(fig, "2_training_data_persistence")

# 3 ---------------------------------------------------------------- 학습 영상에서 AR 출력 vs 데이터
an = J("ar_output_motion_natural.json"); so = J("ssv2_class_motion.json")["ar_output"]
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})
ax = axes[0]; xs = ["TF"] + [str(k) for k in range(1, 7)]
copy = [an["tf_data_copy_direction"]["dir_acc"]] + [an["rollout"][str(k)]["data_copy_direction"]["dir_acc"] for k in range(1, 7)]
cur = [an["tf"]["current_block_encoder"]["dir_acc"]] + [an["rollout"][str(k)]["current_block_encoder"]["dir_acc"] for k in range(1, 7)]
aro = [an["tf"]["ar_output"]["dir_acc"]] + [an["rollout"][str(k)]["ar_output"]["dir_acc"] for k in range(1, 7)]
for v, nm, c, ls in ((copy, "copy the real current flow (data)", GRAY, "--"), (cur, "read current block (encoder)", BLUE, "-"), (aro, "read AR output", ORANGE, "-")):
    ax.plot(range(7), np.array(v) * 100, ls, marker="o", color=c, lw=2, ms=5, label=nm)
ax.set_xticks(range(7)); ax.set_xticklabels(xs); ax.set_xlabel("teacher forcing / rollout step\n(a) Kinetics-710, 1,500 clips")
ax = axes[1]; grps = ["passive", "actor", "other"]; gl = ["objects moving\non their own", "hand moves\nthe object", "other"]; w = 0.26
for i, (key, nm, c) in enumerate((("copy_flow", "copy (data)", GRAY), ("current_block", "current block", BLUE), ("ar_output", "AR output", ORANGE))):
    label_bars(ax, ax.bar(np.arange(3) + (i - 1) * w, [so["teacher_forcing"][g][key]["acc"] * 100 for g in grps], w, color=c), fmt="{:.0f}", dy=0.3)
ax.set_xticks(range(3)); ax.set_xticklabels(gl, fontsize=9); ax.set_xlabel("teacher forcing, SSv2 action groups\n(b) 800 clips per group")
for ax in axes:
    ax.set_ylim(45, 70); ax.axhline(50, color="black", ls=":", lw=0.8)
axes[0].set_ylabel("next-block motion direction correct (%)")
axes[0].legend(loc="upper right", fontsize=8, frameon=False)
fig.suptitle("On training videos the AR output carries as much motion as the data allows (chance 50 %)", fontsize=12, y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.95)); save(fig, "3_ar_output_vs_data_natural")

# 4 ---------------------------------------------------------------- 학습 데이터에 따라
po, pm = J("phys_only_ar_velocity.json"), J("phys_mix_ar_velocity.json"); rv, rf = J("release_velocity.json"), J("release_vobench_pft_velocity.json")
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw={"width_ratios": [1, 1.4]})
ax = axes[0]
bars = [("encoder\n(upper)", po["encoder_probe"]["enc"]["velocity_r2"], GRAY), ("natural\nvideo\nAR", po["encoder_probe"]["natural"]["velocity_r2"], BLUE),
        ("natural\n+25 %\nsynthetic", pm["encoder_probe"]["mix_event"]["velocity_r2"], ORANGE), ("synthetic\nonly\n(single)", po["encoder_probe"]["phys3d_single"]["velocity_r2"], GREEN),
        ("synthetic\nonly\n(multi, e34)", po["encoder_probe"]["phys3d_multi"]["velocity_r2"], DARK)]
label_bars(ax, ax.bar(range(5), [b[1] for b in bars], 0.6, color=[b[2] for b in bars]))
ax.set_xticks(range(5)); ax.set_xticklabels([b[0] for b in bars], fontsize=8); ax.set_ylim(0, 1); ax.set_ylabel("output velocity R² (encoder-trained probe)")
ax.set_xlabel("teacher-forcing output of the next block\n(a)")
ax = axes[1]; ks = np.arange(1, 9)
lines = [("natural video AR", [po["rollout_follow_own_probe"]["natural"][str(k)]["follow"] for k in ks], BLUE, "-"),
         ("natural +25 % synthetic AR", [pm["rollout_follow_own_probe"]["mix_event"][str(k)]["follow"] for k in ks], ORANGE, "-"),
         ("synthetic-only AR (single)", [po["rollout_follow_own_probe"]["phys3d_single"][str(k)]["follow"] for k in ks], GREEN, "-"),
         ("release predictor", [rv["own_probe_on_release_pred"][str(k)]["follow"] for k in ks], DARK, "--"),
         ("release + VOBench fine-tune", [rf["own_probe_on_release_pred"][str(k)]["follow"] for k in ks], GRAY, "--")]
for nm, v, c, ls in lines:
    ax.plot(ks, v, ls, marker="o", color=c, lw=2, ms=5, label=nm)
ax.axhline(1, color="black", ls=":", lw=0.8); ax.axhline(0, color="black", lw=0.6)
ax.set_xticks(ks); ax.set_xticklabels([f"{k}" for k in ks]); ax.set_ylim(0, 1.1)
ax.set_ylabel("share of real movement reproduced"); ax.set_xlabel("prediction step (0.5 s each), each model read with its own probe\n(b)")
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False)
fig.suptitle("More persistent-motion training data → more velocity in the output and longer motion in rollout (3D synthetic val)", fontsize=12, y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.95)); save(fig, "4_training_data_dose")

# 5 ---------------------------------------------------------------- 출력 속도의 꼴
fig, ax = plt.subplots(figsize=(9, 4.2))
models = [("encoder tokens", po, "enc"), ("natural video AR", po, "natural"), ("natural +25 %\nsynthetic AR", pm, "mix_event"), ("synthetic-only AR", po, "phys3d_single")]
w = 0.36
label_bars(ax, ax.bar(np.arange(4) - w / 2, [m[1]["encoder_probe"][m[2]]["velocity_r2"] for m in models], w, color=BLUE, label="encoder-trained probe (encoder form)"))
label_bars(ax, ax.bar(np.arange(4) + w / 2, [m[1]["own_probe"][m[2]]["velocity_r2"] for m in models], w, color=ORANGE, label="probe trained on that model's outputs"))
ax.set_xticks(range(4)); ax.set_xticklabels([m[0] for m in models], fontsize=9); ax.set_ylim(0, 1.05); ax.set_ylabel("output velocity R²")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8, frameon=False)
fig.suptitle("Natural-video AR keeps velocity information, but not in the encoder's form", fontsize=12, y=1.0)
save(fig, "5_output_velocity_form")

# 6 ---------------------------------------------------------------- 저분산
lr, snr = J("velocity_lowrank_median.json"), J("velocity_dim_snr.json")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
attrs = [("velocity", ORANGE), ("color", BLUE), ("shape", GREEN), ("position", DARK)]
ax = axes[0]; label_bars(ax, ax.bar(range(4), [lr["attribute_explains_vector_variance"][a] * 100 for a, _ in attrs], 0.6, color=[c for _, c in attrs]), fmt="{:.1f}", dy=0.2)
ax.set_xticks(range(4)); ax.set_xticklabels([a for a, _ in attrs]); ax.set_ylabel("share of vector variance explained (%)"); ax.set_xlabel("(a)")
ax = axes[1]; KS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1280]
for a, c in attrs:
    v = np.array([lr["readout_by_top_k"][a][str(k)] for k in KS]); base = {"shape": 1 / 6, "color": 1 / 8}.get(a, 0.0)
    ax.plot(KS, (v - base) / (v[-1] - base), "-o", color=c, lw=2, ms=4, label=a)
ax.set_xscale("log", base=2); ax.set_xticks([1, 4, 16, 64, 256, 1024]); ax.set_xticklabels(["1", "4", "16", "64", "256", "1024"])
ax.axhline(0.9, color="black", ls=":", lw=0.8); ax.set_ylim(-0.1, 1.08)
ax.set_ylabel("readout from top-k PCs\n(fraction of full readout)"); ax.set_xlabel("number of principal components k\n(b)"); ax.legend(fontsize=8, frameon=False, loc="lower right")
ax = axes[2]; bins = np.linspace(0, 2.1, 43)
for a, c in (("velocity", ORANGE), ("color", BLUE)):
    d = np.array(snr[a]["per_dim_separation"]); ax.hist(d, bins=bins, histtype="step", lw=2, color=c,
                                                         label=f"{a}: median {np.median(d):.2f}, dims > 1: {int((d > 1).sum())}")
ax.set_xlabel("per-dimension separation |mean A − mean B| / within-group std\n(c) velocity: right vs left · color: red vs blue"); ax.set_ylabel("number of dimensions (of 1280)")
ax.legend(fontsize=8, frameon=False, loc="upper right")
fig.suptitle("Encoder object vectors: velocity is a small, spread-out signal — yet not less separable per dimension (3D synthetic, 18,665 vectors)", fontsize=12, y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.95)); save(fig, "6_low_variance")

# 7 ---------------------------------------------------------------- 중앙값 시험, 판독기 의존
mt = J("median_test_matched_readout.json"); ps = np.round(np.linspace(0, 1, 11), 1)
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
for ax, a, ttl in ((axes[0], "velocity", "velocity (right vs left)\n(a)"), (axes[1], "color", "color (red vs blue)\n(b)")):
    d = mt[a]
    ax.plot(ps, [d["B_mean_difference_direction"][str(p)]["mean_vector"] for p in ps], "--", color=GRAY, lw=2, label="mean vector (any readout)")
    for rn, nm, c in (("B_mean_difference_direction", "median vector, mean-difference direction", BLUE), ("A_two_group_pm1", "median vector, two-group ±1 readout", ORANGE),
                      ("C_original", "median vector, original readout (velocity: regression on all samples)", GREEN)):
        if a == "color" and rn == "C_original":
            continue                                                            # 색의 원래 판독기 = 두 무리 ±1 (같은 판독기라 생략)
        ax.plot(ps, [d[rn][str(p)]["median_vector"] for p in ps], "-o", color=c, lw=2, ms=4, label=nm)
    ax.plot([0, 0.5, 0.5, 1], [-1, -1, 1, 1], ":", color="black", lw=0.8, label="majority vote (reference)")
    ax.axhline(0, color="black", lw=0.5); ax.set_xlabel("share of group A in the mix (p)\n" + ttl); ax.set_ylim(-2.6, 1.6)   # 판별형 판독기 값이 −2.4 까지 내려간다
axes[0].set_ylabel("readout of the aggregated vector\n(−1 = pure B, +1 = pure A)")
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.07))
fig.suptitle("Per-dimension median of mixed encoder vectors: the answer depends on the readout (31 vectors, 200 draws)", fontsize=12, y=1.0)
fig.tight_layout(rect=(0, 0.05, 1, 0.95)); save(fig, "7_median_test_readout")
