"""encoder 정답으로 학습한 위치 · 속도 판독기를 굴린 예측에 적용 — 예측 안의 물체가 어디에 · 얼마나 빠르게 읽히나 (2026-10-26, 학습 없음, CPU).
사용자: "속도랑 위치는 encoder embedding (1280) 으로 (T 별로) 학습하고, 이 head 를 roll out 한 것에다가 적용시켜서 어떻게 읽히는지를 봐야".

특징: 블록마다 칸 256 개 평균 (1, 1280) — encoder 정답 · teacher forcing 출력 (블록 b 예측) · 굴린 예측 출력 (문맥 블록 0–7 = 16 장 → 걸음 k = 블록 7+k).
판독기 (ridge, λ 는 train 클립 80/20 로 고름) — train 8,000 편의 encoder 정답만으로:
  per_block : 블록 b (8–15) 마다 따로 학습, 같은 블록에 적용
  all_blocks: 블록 0–15 를 합쳐 하나
목표: 위치 = 물체 중심 (x, y) px · 속도 = 블록 안 이동 (dx, dy) px (블록 두 장 사이, 0.25 초). 물체가 보이는 블록만.
지표 (val 350 편):
  위치 오차 중앙값 (px) · R² · 따라감 = Σ (읽힌 − 마지막 관측)·(실제 − 마지막 관측) / Σ |실제 − 마지막 관측|² (|실제 − 마지막| ≥ 16 px 인 블록; 마지막 관측 = 블록 7 중심)
  속도 R² · 이득 = Σ 읽힌·실제 / Σ |실제|² (|실제| ≥ 2 px 인 블록)
  비교: "마지막 관측에 머묾" 예측 (위치 = 블록 7 중심) 의 오차 · R²

  /venv/vjepa/bin/python z_research/scripts/analysis/rollout_readout_pos_vel.py
출력: z_analysis/results/rollout_readout_pos_vel.json · z_analysis/figures/rollout_readout_pos_vel.png
"""
import json
import numpy as np

P = "/workspace/auto_research_cache/ar_layers_pooled"; R = "/workspace/world_model/vjepa2/z_analysis"
TR, VA = dict(np.load(f"{P}/train.npz")), dict(np.load(f"{P}/val.npz"))
LAMS = (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4)


def fit(X, Y, clip):
    mu, sd = X.mean(0), X.std(0) + 1e-6; Z = (X - mu) / sd; tr = (clip % 5) != 0
    def solve(Zs, Ys, lam):
        ym = Ys.mean(0); ev, V = np.linalg.eigh(Zs.T @ Zs); W = V @ ((V.T @ (Zs.T @ (Ys - ym))) / (ev + lam)[:, None]); return W, ym
    ev, V = np.linalg.eigh(Z[tr].T @ Z[tr]); ym = Y[tr].mean(0); ZtY = Z[tr].T @ (Y[tr] - ym)
    errs = [(((Z[~tr] @ (V @ ((V.T @ ZtY) / (ev + lam)[:, None])) + ym) - Y[~tr]) ** 2).sum() for lam in LAMS]
    W, ym = solve(Z, Y, LAMS[int(np.argmin(errs))])
    return lambda Xn: ((Xn - mu) / sd) @ W + ym


def train_set(blocks):
    X, Y, C = [], [], []
    for b in blocks:
        m = TR["lab"][:, b, 0] > 0
        X.append(TR["enc"][m, b].astype(np.float32)); Y.append(TR["lab"][m, b, 4:8].astype(np.float64)); C.append(np.nonzero(m)[0])
    return np.concatenate(X), np.concatenate(Y), np.concatenate(C)


heads = {"all_blocks": fit(*train_set(range(16)))}
for b in range(8, 16):
    heads[f"b{b}"] = fit(*train_set([b]))
r2 = lambda P_, Y: float(1 - ((P_ - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())
out = {}
lab = VA["lab"]
for hname in ("per_block", "all_blocks"):
    res = {}
    for k in range(1, 9):
        b = 7 + k
        m = (lab[:, b, 0] > 0) & (lab[:, 7, 0] > 0)
        h = heads[f"b{b}"] if hname == "per_block" else heads["all_blocks"]
        true_v, true_p, last_p = lab[m, b, 4:6], lab[m, b, 6:8], lab[m, 7, 6:8]
        src = {"encoder": VA["enc"][m, b], "tf_out": VA["tfout"][m, b], "rollout": VA["roout"][m, k - 1]}
        row = {"n": int(m.sum())}
        for s, X in src.items():
            Pr = h(X.astype(np.float32)); pv, pp = Pr[:, :2], Pr[:, 2:]
            d = true_p - last_p; mv = np.linalg.norm(d, axis=1) >= 16
            mvv = np.linalg.norm(true_v, axis=1) >= 2
            row[s] = {"pos_err_px": round(float(np.median(np.linalg.norm(pp - true_p, axis=1))), 1), "pos_r2": round(r2(pp, true_p), 3),
                      "pos_follow": round(float(((pp[mv] - last_p[mv]) * d[mv]).sum() / (d[mv] ** 2).sum()), 3) if mv.sum() > 5 else None,
                      "vel_r2": round(r2(pv, true_v), 3), "vel_gain": round(float((pv[mvv] * true_v[mvv]).sum() / (true_v[mvv] ** 2).sum()), 3) if mvv.sum() > 5 else None,
                      "vel_corr": round(float(np.corrcoef(pv[mvv].ravel(), true_v[mvv].ravel())[0, 1]), 3) if mvv.sum() > 5 else None}
        row["stay_last"] = {"pos_err_px": round(float(np.median(np.linalg.norm(last_p - true_p, axis=1))), 1), "pos_r2": round(r2(last_p, true_p), 3)}
        row["n_moved16"] = int((np.linalg.norm(true_p - last_p, axis=1) >= 16).sum())
        res[k] = row
    out[hname] = res
json.dump(out, open(f"{R}/results/rollout_readout_pos_vel.json", "w"), indent=1)

for hname in ("per_block", "all_blocks"):
    print("==", hname)
    for k in range(1, 9):
        r = out[hname][k]
        print(f" step {k} n{r['n']:4d} | pos err px enc {r['encoder']['pos_err_px']:5.1f} tf {r['tf_out']['pos_err_px']:5.1f} ro {r['rollout']['pos_err_px']:5.1f} stay {r['stay_last']['pos_err_px']:5.1f}"
              f" | follow enc {r['encoder']['pos_follow']} tf {r['tf_out']['pos_follow']} ro {r['rollout']['pos_follow']}"
              f" | vel gain enc {r['encoder']['vel_gain']} tf {r['tf_out']['vel_gain']} ro {r['rollout']['vel_gain']} | vel r2 enc {r['encoder']['vel_r2']} ro {r['rollout']['vel_r2']}")

# ---- 그림 (2026-10-26 사용자: "기준을 GT 로", read/true 비율 대신 픽셀 단위 그대로)
#  (a) 위치: 마지막 관측 (블록 7) 에서 실제로 움직인 거리 (정답) vs 읽힌 이동을 실제 이동 방향으로 투영한 거리 — 걸음별 평균 (실제 이동 ≥ 16 px 블록)
#  (b) 속도: 실제 속력 (블록 안 이동, px) vs 읽힌 속도를 실제 움직임 방향으로 투영한 값 — 걸음별 평균 (실제 ≥ 2 px 블록)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"
ks = list(range(1, 9)); curves = {"pos": {}, "vel": {}}
for k in ks:
    b = 7 + k; m = (lab[:, b, 0] > 0) & (lab[:, 7, 0] > 0); h = heads[f"b{b}"]
    tv, tp, lp = lab[m, b, 4:6], lab[m, b, 6:8], lab[m, 7, 6:8]
    d = tp - lp; dn = np.linalg.norm(d, axis=1); mp = dn >= 16; vn = np.linalg.norm(tv, axis=1); mvv = vn >= 2
    curves["pos"].setdefault("GT", []).append(float(dn[mp].mean())); curves["vel"].setdefault("GT", []).append(float(vn[mvv].mean()))
    for s_, X in (("encoder", VA["enc"][m, b]), ("teacher forcing", VA["tfout"][m, b]), ("rollout", VA["roout"][m, k - 1])):
        Pr = h(X.astype(np.float32)); pv, pp = Pr[:, :2], Pr[:, 2:]
        curves["pos"].setdefault(s_, []).append(float((((pp - lp)[mp] * d[mp]).sum(1) / dn[mp]).mean()))
        curves["vel"].setdefault(s_, []).append(float(((pv[mvv] * tv[mvv]).sum(1) / vn[mvv]).mean()))
out["figure_curves_px"] = curves
json.dump(out, open(f"{R}/results/rollout_readout_pos_vel.json", "w"), indent=1)
# 2026-10-26 사용자: "teacher forcing 빼고 pixel space GT 빼셈" → encoder 임베딩 · 굴린 예측 두 선만 (값은 json 에 넷 다 남는다)
STY = {"encoder": ("#1baf7a", "D", "-", "encoder embedding -> head"), "rollout": ("#eb6834", "s", "-", "rollout output -> head")}
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
for ax, (key, ylab) in zip(axes, (("pos", "displacement from last seen position (px)"), ("vel", "speed along true motion (px per block)"))):
    for s_, (col, mk, ls, nm) in STY.items():
        ax.plot(ks, curves[key][s_], marker=mk, color=col, ls=ls, lw=2.4 if s_ == "GT" else 2, ms=8 if s_ == "GT" else 5, label=nm)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(ks); ax.set_xlabel("rollout step"); ax.set_ylabel(ylab); ax.spines[["top", "right"]].set_visible(False)
for ax, lab_ in zip(axes, ("(a)", "(b)")):
    ax.text(0.5, -0.24, lab_, transform=ax.transAxes, ha="center", fontsize=11)
fig.tight_layout(rect=(0, 0.1, 1, 1))
h_, l_ = axes[0].get_legend_handles_labels(); fig.legend(h_, l_, loc="lower center", ncol=2, fontsize=9, frameon=False)   # 선과 안 겹치게 그림 아래
fig.savefig(f"{R}/figures/rollout_readout_pos_vel.png", dpi=170, bbox_inches="tight")
print("saved", {k: {s_: [round(v, 1) for v in vals] for s_, vals in c.items()} for k, c in curves.items()})
