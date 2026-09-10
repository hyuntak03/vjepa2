#!/usr/bin/env python3
"""선형 readout 의 토큰별 기여를 프레임 위에 그린다 — 읽힌 위치가 어디서 오는가.

`rollout_position.py` 의 readout 은 공간평균+선형이라 예측이 항등식으로 분해된다:

    x̂_t = (1/256) Σ_ij s_ij,t + b,      s_ij = ((f_ij − mu)/sd)·w

⚠️ **학습된 attention 이 아니다.** 이미 적합된 readout 을 사후 분해한 것이라
   "무엇을 보게 학습시켰나" 가 아니라 "무엇이 이 예측을 만들었나" 를 보여준다.
   그림에는 공간평균을 뺀 s − mean(s) 를 그린다 (평균은 예측의 상수 오프셋이라 위치 정보가 없다).

  fig_contrib_map      프레임 + 기여 히트맵 (행 = 속도, 열 = 튜블릿)
  fig_contrib_profile  열 주변화 프로파일 (물체 열 표시)

  python z_research/scripts/figures/plot_rollout_contrib.py --src h
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import matplotlib as mpl
import numpy as np
mpl.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.colors import LinearSegmentedColormap              # noqa: E402
from PIL import Image                                              # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import rollout_position as rp                                      # noqa: E402

FRAMES = Path("/local_datasets/world/world_analysis/RollOut_v1")
OUTROOT = rp.ROOT / "z_research/RollOutV1/figures"
FIGDIR = {"fig_contrib_map": "contribution", "fig_contrib_profile": "contribution"}
BLUE, ORANGE, GREEN, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a9a"
G, S, T = rp.G, rp.S, rp.T_FUT

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})
# 발산 컬러맵 — 0(=공간평균)이 투명, 음수 파랑 / 양수 주황
DIV = LinearSegmentedColormap.from_list("d", [
    (0.00, (0.16, 0.47, 0.84, 0.95)), (0.35, (0.16, 0.47, 0.84, 0.35)),
    (0.50, (1, 1, 1, 0.0)),
    (0.65, (0.92, 0.41, 0.20, 0.35)), (1.00, (0.92, 0.41, 0.20, 0.95))])


def frame(fn, raw_f, res=256):
    p = FRAMES / fn / f"{raw_f:06d}.png"
    return np.asarray(Image.open(p).convert("L").resize((res, res), Image.BILINEAR)) / 255.0


def save(fig, name, outroot):
    d = outroot / FIGDIR.get(name, "")
    d.mkdir(parents=True, exist_ok=True)
    for e in ("pdf", "png"):
        fig.savefig(d / f"{name}.{e}", dpi=200, bbox_inches="tight")
    print(f"  -> {d/name}.{{pdf,png}}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="h", choices=["h", "p"])
    ap.add_argument("--speeds", type=float, nargs="*", default=[-160, -60, 0, 60, 160])
    ap.add_argument("--anchor", type=float, default=0.0)
    ap.add_argument("--outdir", type=Path, default=OUTROOT)
    a = ap.parse_args()

    idx_rows = {r["video_id"]: r for r in csv.DictReader(rp.INDEX.open())}
    print("풀링 + ridge (rollout_position 재사용)")
    vids, F = rp.pool()
    rows = [idx_rows[v] for v in vids]
    Y, v_true, anchor_n, hw, fps = rp.labels_from_index(rows)
    cond = np.array([r["condition"] for r in rows])
    anc = np.array([float(r["flat_anchor_x_cm"]) for r in rows])

    rng = np.random.RandomState(0)
    tr, va = [], []
    for c in np.unique(cond):
        i = np.where(cond == c)[0]; rng.shuffle(i); h = len(i) // 2
        tr += list(i[:h]); va += list(i[h:])
    tr, va = np.array(sorted(tr)), np.array(sorted(va))
    X = F[a.src][tr].reshape(-1, F[a.src].shape[-1]).astype(np.float64)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    w = rp.ridge((X - mu) / sd, Y[tr].ravel(), 1.0)
    del F, X
    print(f"  ridge on {a.src} 적합 완료 (train {len(tr)})")

    # 예시 클립 — 속도별 하나, anchor 고정, held-out
    vaset = set(va.tolist()); picks = []
    for v in a.speeds:
        c = [i for i in np.where((v_true == v) & (np.abs(anc - a.anchor) < 1e-6))[0]
             if i in vaset]
        if c:
            picks.append((v, int(c[0])))
    if not picks:
        rp.die("예시 클립이 없다")

    # 해당 클립만 토큰 점수 계산 (전수 패스 불필요)
    A = np.load(rp.CACHE / ("predictor.npy" if a.src == "p" else "target.npy"), mmap_mode="r")
    ids = np.array([i for _, i in picks])
    f = np.asarray(A[ids] if a.src == "p" else A[ids, T * S:]).reshape(len(ids), T, S, -1)
    co, off = w[:-1] / sd, float((mu / sd) @ w[:-1])
    Sc = (f.astype(np.float32) @ co.astype(np.float32) - off).reshape(len(ids), T, G, G)
    Sc = Sc - Sc.mean((2, 3), keepdims=True)         # 공간평균 제거 = 위치를 만드는 성분만
    jstar = np.clip(np.round((Y + 1) / 2 * G - 0.5), 0, G - 1).astype(int)
    print(f"  기여맵 {Sc.shape}  |s−mean| 최대 {np.abs(Sc).max():.3f}")

    # ── 그림 1: 프레임 + 기여 히트맵 ────────────────────────────────────────
    vmax = float(np.percentile(np.abs(Sc), 99))
    fig, ax = plt.subplots(len(picks), T, figsize=(7.0, 7.0 / T * len(picks) * 1.04),
                           squeeze=False)
    for r, (v, ci) in enumerate(picks):
        fn = rows[ci]["file_name"]
        for t in range(T):
            x = ax[r][t]
            x.imshow(frame(fn, 48 + 6 * t), cmap="gray", vmin=0, vmax=1)
            x.imshow(np.kron(Sc[r, t], np.ones((16, 16))), cmap=DIV,
                     vmin=-vmax, vmax=vmax, interpolation="nearest")
            px = (Y[ci, t] + 1) / 2 * 256
            x.axvline(px, color=GREEN, lw=1.2)
            x.set_xticks([]); x.set_yticks([])
            for s_ in x.spines.values():
                s_.set_linewidth(0.4); s_.set_color(MUTED)
            if r == 0:
                x.set_title(f"$t_{{{t}}}$", fontsize=8, pad=2)
        ax[r][0].set_ylabel(f"v = {v:+.0f}\ncm/s", fontsize=8, labelpad=3)
    hs = [plt.Line2D([], [], color=GREEN, lw=1.5, label="true object $x$"),
          plt.Line2D([], [], color=ORANGE, lw=5, alpha=.8, label="contribution $> $ mean"),
          plt.Line2D([], [], color=BLUE, lw=5, alpha=.8, label="contribution $<$ mean")]
    fig.legend(handles=hs, loc="lower center", ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.035))
    fig.subplots_adjust(wspace=0.04, hspace=0.06)
    save(fig, "fig_contrib_map", a.outdir)

    # ── 그림 2: 열 주변화 프로파일 ──────────────────────────────────────────
    col = Sc.mean(2)                                  # (n, 8, 16)
    fig, ax = plt.subplots(1, len(picks), figsize=(7.0, 1.9), sharey=True)
    cmap = plt.get_cmap("viridis")
    for r, (v, ci) in enumerate(picks):
        for t in range(T):
            ax[r].plot(np.arange(G), col[r, t], lw=1.0, color=cmap(t / (T - 1)))
            ax[r].plot(jstar[ci, t], col[r, t, jstar[ci, t]], "o", ms=3.2,
                       color=cmap(t / (T - 1)), mec="k", mew=0.4)
        ax[r].axhline(0, color=MUTED, lw=0.5)
        ax[r].set_title(f"v = {v:+.0f}", fontsize=8)
        ax[r].set_xlabel("token column", fontsize=8)
        ax[r].tick_params(labelsize=7)
        for s_ in ("top", "right"):
            ax[r].spines[s_].set_visible(False)
    ax[0].set_ylabel("contribution\n$-$ spatial mean", fontsize=8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, T - 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("tubelet $t$", fontsize=8); cb.ax.tick_params(labelsize=7)
    save(fig, "fig_contrib_profile", a.outdir)

    # 수치도 같이 — 그림만 보고 판단하지 않게
    print("\n  물체 열(원 표시)이 프로파일에서 특별한가:")
    for r, (v, ci) in enumerate(picks):
        c = col[r]; j = jstar[ci]
        at = np.array([c[t, j[t]] for t in range(T)])
        rank = np.array([(np.abs(c[t]) > abs(c[t, j[t]])).sum() for t in range(T)])
        print(f"    v={v:+6.0f}  |기여|@물체열 평균 {np.abs(at).mean():.4f}  "
              f"(전체 평균 {np.abs(c).mean():.4f})   |순위| 중앙 {int(np.median(rank))}/16")


if __name__ == "__main__":
    main()
