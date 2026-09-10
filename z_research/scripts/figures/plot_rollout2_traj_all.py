#!/usr/bin/env python3
"""RollOut_v2 — 전 샘플 궤적 비교. 시나리오마다 primary 레벨 7 패널, 튜블릿 0..7 에 대한 위치의 클립 평균 ± SD.
  검정 = 정답 (plan)     주황 = predictor: p 의 pooled readout (p_A; ledge·wall 은 p_B = 판정 대상을 학습에서 뺀 자)
평가 클립 = train 30% 를 뺀 전부 (val 70% + holdout 레벨).
  python z_research/scripts/figures/plot_rollout2_traj_all.py --device cuda:0
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as Fn
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import rollout2_position as rp; import rollout2_token_decoder as td       # noqa: E402

OUT = rp.ROOT / "z_research/RollOutV2/figures/traj_all"; BLACK, ORANGE, GREEN = "#000000", "#eb6834", "#1baf7a"
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "pdf.fonttype": 42})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--readout", default="p", choices=["p", "h"], help="p: p 로 학습(ledge·wall 제외) / h: h 32프레임 전부로 학습 → p 에 test")
    a = ap.parse_args(); dev = torch.device(a.device)
    idx = list(csv.DictReader(rp.INDEX.open())); vids = [r["video_id"] for r in idx]
    meta = json.loads((rp.CACHE / "meta.json").read_text()); row_of = {v: i for i, v in enumerate(meta["video_ids"])}; rows = np.array([row_of[v] for v in vids])
    F = rp.pool(rows); res = 144.0
    px = np.stack([rp.arr(r["px_x_by_sample"]) for r in idx]); py = np.stack([rp.arr(r["px_y_by_sample"]) for r in idx])
    L16 = np.stack([px.reshape(-1, 16, 2).mean(2) / res - 1, py.reshape(-1, 16, 2).mean(2) / res - 1], -1).astype(np.float32); L = L16[:, 8:]
    inf = (np.stack([rp.arr(r["in_frame_by_sample"]) for r in idx]) > 0).reshape(-1, 16, 2).all(2) & (np.abs(L16) <= 1).all(2)
    sc = np.array([r["scenario"] for r in idx]); plaus = np.array([int(r["plausible"]) for r in idx]); prim = np.array([float(r["primary"]) for r in idx]); sec = np.array([float(r["secondary"]) for r in idx])
    hold = np.array([r["holdout"] in ("1", "True", "true") for r in idx]); blk = np.array([r["block_id"] for r in idx]); imp_of = {blk[i]: i for i in np.where(plaus == 0)[0]}
    rng = np.random.RandomState(0); tr = []
    for scn in rp.SCEN:
        ids = np.where((sc == scn) & (plaus == 1) & ~hold)[0]; cell = np.array([f"{prim[i]}|{sec[i]}" for i in ids])
        for c in np.unique(cell):
            i = ids[cell == c]; rng.shuffle(i); tr += list(i[:int(round(rp.TRAIN_FRAC * len(i)))])
    tr = np.array(sorted(tr)); tr_B = tr[~np.isin(sc[tr], ["ledge", "wall"])]; ev = np.array([i for i in np.where(plaus == 1)[0] if i not in set(tr)])
    fut = lambda s_, ids: F[s_][ids].reshape(-1, rp.D).astype(np.float64)
    W = {k: {c: rp.fit_lstsq(fut("p", t_), L[t_][:, :, ci].ravel()) for ci, c in enumerate("xy")} for k, t_ in (("p_A", tr), ("p_B", tr_B))}
    if a.readout == "h":                                   # h 의 16 튜블릿(문맥+미래) 풀링, train 클립 + 그 block 의 imp 클립, 화면 밖 제외
        tr_h = np.concatenate([tr, np.array([imp_of[blk[i]] for i in tr if blk[i] in imp_of])]); H = np.load(rp.CACHE / "target.npy", mmap_mode="r")
        o = np.argsort(rows[tr_h]); Xh = np.empty((len(tr_h), 16, rp.D), np.float32)
        for k in range(0, len(tr_h), 64):
            sel = tr_h[o[k:k + 64]]; Xh[o[k:k + 64]] = np.asarray(H[rows[sel]]).reshape(-1, 16, 256, rp.D).mean(2)
        m = inf[tr_h].ravel(); X = Xh.reshape(-1, rp.D).astype(np.float64)[m]
        W = {"h": {c: rp.fit_lstsq(X, L16[tr_h][:, :, ci].ravel()[m]) for ci, c in enumerate("xy")}}
    # 평가 클립 예측
    Xp = {c: np.empty((len(ev), 8), np.float32) for c in "xy"}
    for k, i in enumerate(ev):
        wname = "h" if a.readout == "h" else ("p_B" if sc[i] in ("ledge", "wall") else "p_A"); X = fut("p", [i])
        for c in "xy":
            Xp[c][k] = rp.apply_w(X, W[wname][c]).reshape(8)
    global OUT
    if a.readout == "h":
        OUT = OUT.with_name("traj_all_hfit")
    OUT.mkdir(parents=True, exist_ok=True); t = np.arange(8)
    for scn in rp.SCEN:
        m = sc[ev] == scn; lv = sorted(set(prim[ev][m])); coords = ["x", "y"] if rp.MOVING[scn] == "xy" else [rp.MOVING[scn]]
        fig, axes = plt.subplots(len(coords), len(lv), figsize=(1.75 * len(lv), 2.5 * len(coords)), sharex=True, sharey="row", squeeze=False)
        for j, v in enumerate(lv):
            k = m & (prim[ev] == v); n = int(k.sum())
            for r_, c in enumerate(coords):
                ax = axes[r_, j]; ci = "xy".index(c)
                for arr, col, lab, ls in ((L[ev][k][:, :, ci], BLACK, "truth", "-"), (Xp[c][k], ORANGE, "predictor (pooled readout fit on h, 32 frames)" if a.readout == "h" else "predictor (pooled readout fit on p)", "-")):
                    mu, sd = (arr.mean(0) + 1) * res, arr.std(0) * res          # 정규화 [-1,1] -> px
                    ax.plot(t, mu, ls, color=col, lw=1.6, label=lab if (j == 0 and r_ == 0) else None); ax.fill_between(t, mu - sd, mu + sd, color=col, alpha=0.12, lw=0)
                ax.set_ylim(0, 288); ax.invert_yaxis() if c == "y" else None
                for s_ in ("top", "right"):
                    ax.spines[s_].set_visible(False)
                ax.tick_params(labelsize=7); ax.grid(alpha=0.25, lw=0.4)
                if r_ == 0:
                    ax.set_title(f"{idx[ev[np.where(k)[0][0]]]['primary_name']} = {v:g}   (n={n}{', holdout' if hold[ev][k].all() else ''})", fontsize=7.5)
                if j == 0:
                    ax.set_ylabel(f"screen {c} (px)", fontsize=8)
                if r_ == len(coords) - 1:
                    ax.set_xlabel("future tubelet", fontsize=8)
        h_, l_ = axes[0, 0].get_legend_handles_labels(); fig.legend(h_, l_, frameon=False, fontsize=8.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.0))
        fig.suptitle(f"{scn}  —  mean ± SD over evaluation clips (train 30% excluded)   [readout fit on {a.readout}]", fontsize=9, y=0.90)
        fig.tight_layout(rect=(0, 0, 1, 0.86))
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"{scn}.{ext}", dpi=220)
        plt.close(fig); print(f"  [saved] {OUT.name}/{scn}.png")


if __name__ == "__main__":
    main()
