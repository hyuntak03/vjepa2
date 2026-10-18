"""VOBench AR readout 그림 + GIF (2026-10-12). 입력 = vobench_ar_readout.py extract 산출 (readout/vobench_<name>/shard*.npz).

  python z_research/scripts/figures/plot_vobench_ar_readout.py [--name ar40]
  → z_research/VOBench_v1/readout/vobench_<name>/figures/{fig_readout_steps.{png,pdf}, examples.gif}
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "z_research/scripts/analysis")); sys.path.insert(0, str(REPO / "z_research/scripts/figures"))
import vobench_ar_readout as V  # noqa: E402

OCCS = ("visible", "early", "mid", "late")
MOTS = {"flat_vel": "const. velocity", "flat_acc": "speeding", "ramp_down": "speeding", "flat_dec": "slowing", "ramp_up": "slowing"}


def collect(name):
    Z = V.load(name); vid = {v: i for i, v in enumerate(Z["video_id"])}
    thr = json.load(open(REPO / f"z_research/VOBench_v1/readout/identity_vobprim_{name}/summary.json"))["reps"]["p"]["thr_val_fpr5"]
    lab = {}; A = defaultdict(lambda: defaultdict(list))
    for r in V.rows_primitive():
        if r["variant"] not in ("pos_a", "pos_b") or r["motion"] == "flat_static": continue
        arm = "A" if r["variant"] == "pos_a" else "B"
        obj = r["obj_a"] if arm == "A" else r["obj_b"]
        i = vid[r["video_id"]]; pb = Z["pp_prob"][i]
        pres = 1 - pb[:, V.NONE] > thr
        if obj == "empty":
            A[(r["occlusion"], "empty")]["present"].append(pres); continue
        bdir = V.FR / Path(r["file_name"]).parent
        if bdir not in lab: lab[bdir] = json.load(open(bdir / "label.json"))["labels"]
        L = lab[bdir][arm]
        P = np.array([[L[3 * s]["u"], L[3 * s]["v"]] for s in range(32)], float)
        hid = np.array([L[3 * s]["hidden"] for s in range(32)], bool)
        vis = [s for s in range(16) if not hid[s]]; last = vis[-1]; first = max(vis[0], last - 2)
        vel = (P[last] - P[first]) / max(last - first, 1)
        sh, co = obj.split("/"); cls = V.SHAPES.index(sh) * len(V.COLORS) + V.COLORS.index(co)
        tru, ar, cv = [], [], []
        for j in range(8):
            t = (P[16 + 2 * j] + P[17 + 2 * j]) / 2; ax = t - P[last]; n = np.linalg.norm(ax); ax = ax / max(n, 1e-6)
            xy = V.to512(Z["pp_xy"][i, j]); c = P[last] + vel * (16.5 + 2 * j - last)
            tru.append(n / V.CELL); ar.append(((xy - P[last]) @ ax) / V.CELL if pres[j] else np.nan); cv.append(((c - P[last]) @ ax) / V.CELL)
        A[(r["occlusion"], "obj")]["present"].append(pres)
        A[(r["occlusion"], "obj")]["idgp"].append(np.where(pres, pb[:, :V.NONE].argmax(-1) == cls, np.nan))
        k = (r["occlusion"], MOTS[r["motion"]])
        A[k]["true"].append(tru); A[k]["ar"].append(ar); A[k]["cv"].append(cv)
    return A, Z, vid


def plot(A, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from plot_v11_surprise import BLUE, ORANGE, GREEN, MUTED, frame
    t = np.arange(8, 16)
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.9), sharex=True)
    for c, occ in enumerate(OCCS):
        ax = axes[0, c]; frame(ax, (-3, 104), (0, 25, 50, 75, 100))
        ax.plot(t, 100 * np.mean(A[(occ, "obj")]["present"], 0), "-o", color=BLUE, ms=2.5, lw=1.3, label="'present' (object context)")
        ax.plot(t, 100 * np.mean(A[(occ, "empty")]["present"], 0), ":", color=BLUE, lw=1.2, label="'present' (empty context)")
        ax.plot(t, 100 * np.nanmean(A[(occ, "obj")]["idgp"], 0), "-o", color=ORANGE, ms=2.5, lw=1.3, label="identity | present")
        ax.set_title(occ, fontsize=8.5)
        ax = axes[1, c]; frame(ax, (-0.5, 9.5), (0, 3, 6, 9))
        for mot, col in (("const. velocity", BLUE), ("speeding", ORANGE), ("slowing", GREEN)):
            d = A[(occ, mot)]
            ax.plot(t, np.mean(d["true"], 0), "-", color=col, lw=1.4, label=f"truth · {mot}")
            ax.plot(t, np.nanmean(d["ar"], 0), "--", color=col, lw=1.2, label=f"p read · {mot}")
        ax.set_xticks(t[::2]); ax.set_xticklabels([f"t{x}" for x in t[::2]], fontsize=7)
    axes[0, 0].set_ylabel("%", fontsize=8); axes[1, 0].set_ylabel("distance along motion\nfrom last view (cells)\np: 'present' steps only", fontsize=7)
    axes[0, 0].legend(fontsize=5.8, frameon=False, loc="lower left")
    axes[1, 0].legend(fontsize=5.3, frameon=False, loc="upper left", ncol=1)
    for i, x in enumerate("abcdefgh"):
        axes.flat[i].text(0.5, -0.30 if i >= 4 else -0.16, f"({x})", transform=axes.flat[i].transAxes, ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig_readout_steps.png", dpi=200); fig.savefig(out / "fig_readout_steps.pdf"); plt.close(fig)


def gif(Z, vid, out, name):
    global THR
    THR = json.load(open(REPO / f"z_research/VOBench_v1/readout/identity_vobprim_{name}/summary.json"))["reps"]["p"]["thr_val_fpr5"]
    from PIL import Image, ImageDraw
    rows = {r["video_id"]: r for r in V.rows_primitive()}
    want = [("early", "flat_vel"), ("early", "flat_acc"), ("early", "flat_dec"), ("late", "flat_vel")]
    picks = []
    for occ, mot in want:
        c = [r for r in rows.values() if r["occlusion"] == occ and r["motion"] == mot and r["variant"] == "pos_a" and r["obj_a"] != "empty"]
        c.sort(key=lambda r: (r["violation"] != "shape", r["k"] not in ("2", "0"), r["video_id"]))   # shape · k 2 우선, 없으면 아무거나 (test block 만일 때)
        if c: picks.append(c[0])
    PX = 256; frames = []
    for s in range(32):
        canvas = Image.new("RGB", (PX * len(picks), PX + 34), (255, 255, 255)); dr = ImageDraw.Draw(canvas)
        for c, r in enumerate(picks):
            fr = V.FR / r["file_name"] / f"{3 * s:06d}.png"
            im = Image.open(fr).convert("RGB").resize((PX, PX)); canvas.paste(im, (c * PX, 0))
            L = json.load(open(V.FR / Path(r["file_name"]).parent / "label.json"))["labels"]["A"][3 * s]
            d = ImageDraw.Draw(canvas); x0 = c * PX
            if s >= 16:
                j = (s - 16) // 2; i = vid[r["video_id"]]
                tu, tv = L["u"] / 2, L["v"] / 2
                d.ellipse([x0 + tu - 9, tv - 9, x0 + tu + 9, tv + 9], outline=(255, 255, 255), width=2)
                pb = Z["pp_prob"][i, j]; k = int(pb[:V.NONE].argmax())
                if 1 - pb[V.NONE] > THR:
                    au, av = V.to512(Z["pp_xy"][i, j]) / 2
                    d.line([x0 + au - 8, av, x0 + au + 8, av], fill=(0, 220, 255), width=3); d.line([x0 + au, av - 8, x0 + au, av + 8], fill=(0, 220, 255), width=3)
                    txt = f"p: {V.SHAPES[k // 8]}/{V.COLORS[k % 8]}  P(none) {pb[V.NONE]:.2f}"
                else:
                    txt = f"p: none  P(none) {pb[V.NONE]:.2f}"
            else:
                txt = "context"
            d.text((x0 + 4, PX + 2), f"{r['occlusion']} · {r['motion']} · true {r['obj_a']}", fill=(0, 0, 0))
            d.text((x0 + 4, PX + 17), f"t{s // 2}  {txt}", fill=(0, 0, 0))
        frames.append(canvas)
    frames[0].save(out / "examples.gif", save_all=True, append_images=frames[1:], duration=[250] * 16 + [500] * 16, loop=0)
    frames[24].save(out / "examples_t12.png")


def timeline_stats(Z, vid):
    """(가림 시점, static|moving) → h_id · h_none (16) · p_id · p_none (8) · hid (16) 영상별 배열. 물체 있는 가능 영상만, argmax 기준."""
    G = defaultdict(lambda: defaultdict(list)); lab = {}
    for r in V.rows_primitive():
        if r["variant"] not in ("pos_a", "pos_b"): continue
        arm = "A" if r["variant"] == "pos_a" else "B"; obj = r["obj_a"] if arm == "A" else r["obj_b"]
        if obj == "empty": continue
        bdir = V.FR / Path(r["file_name"]).parent
        if bdir not in lab: lab[bdir] = json.load(open(bdir / "label.json"))["labels"]
        hid = np.array([lab[bdir][arm][3 * s]["hidden"] for s in range(32)], bool).reshape(16, 2).all(1)
        sh, co = obj.split("/"); cls = V.SHAPES.index(sh) * len(V.COLORS) + V.COLORS.index(co)
        i = vid[r["video_id"]]; k = (r["occlusion"], "static" if r["motion"] == "flat_static" else "moving")
        h = Z["hh16_prob"][i].argmax(-1); p = Z["pp_prob"][i].argmax(-1)
        G[k]["h_id"].append(h == cls); G[k]["h_none"].append(h == V.NONE)
        G[k]["p_id"].append(p == cls); G[k]["p_none"].append(p == V.NONE); G[k]["hid"].append(hid)
    return G


def timeline(Z, vid, out):
    """x = t0–t15. 파랑 = h 자 → h (16 블록) · 주황 = p 자 → p (t8–t15). 실선 = 정체 정답률, 점선 = '없음' 으로 고른 비율 (argmax).
    물체 있는 가능 영상만. 회색 = 그 조건에서 물체가 실제로 가려진 튜블릿 (절반 넘는 영상)."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from plot_v11_surprise import BLUE, ORANGE, MUTED, frame
    G = timeline_stats(Z, vid)
    t = np.arange(16); tf = np.arange(8, 16)
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.9), sharex=True, sharey=True)
    for ri, mot in enumerate(("static", "moving")):
        for ci, occ in enumerate(OCCS):
            ax = axes[ri, ci]; frame(ax, (-3, 104), (0, 25, 50, 75, 100)); d = G[(occ, mot)]
            for tt in t[np.mean(d["hid"], 0) > 0.5]:
                ax.axvspan(tt - 0.5, tt + 0.5, color="0.88", lw=0, zorder=0)
            ax.axvline(7.5, color=MUTED, lw=0.8, ls="--")
            ax.plot(t, 100 * np.mean(d["h_id"], 0), "-", color=BLUE, lw=1.4, label="h: identity acc")
            ax.plot(t, 100 * np.mean(d["h_none"], 0), ":", color=BLUE, lw=1.4, label="h: 'none' chosen")
            ax.plot(tf, 100 * np.mean(d["p_id"], 0), "-", color=ORANGE, lw=1.4, label="p: identity acc")
            ax.plot(tf, 100 * np.mean(d["p_none"], 0), ":", color=ORANGE, lw=1.4, label="p: 'none' chosen")
            if ri == 0: ax.set_title(occ, fontsize=8.5)
            if ci == 0: ax.set_ylabel(f"{mot}\naccuracy (%)", fontsize=8)
            ax.set_xticks([0, 4, 8, 12, 15]); ax.set_xticklabels(["t0", "t4", "t8", "t12", "t15"], fontsize=7)
            ax.text(0.5, -0.30 if ri else -0.16, f"({'abcdefgh'[ri * 4 + ci]})", transform=ax.transAxes, ha="center", fontsize=8)
    axes[1, 0].legend(fontsize=5.5, frameon=False, loc="center left")
    fig.tight_layout(); fig.savefig(out / "fig_readout_timeline.png", dpi=200); fig.savefig(out / "fig_readout_timeline.pdf"); plt.close(fig)
    return {f"{o}/{m}": {k: (100 * np.mean(v, 0)).round(1).tolist() for k, v in d.items() if k != "hid"} | {"n": len(d["h_id"])} for (o, m), d in G.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--name", default="ar40")
    ap.add_argument("--test-only", action="store_true", help="split v2 test block 만 (post-FT 모델은 train block 을 봤다, 2026-10-14)")
    a = ap.parse_args()
    out = V.out_dir(a.name) / ("figures_test_v2" if a.test_only else "figures"); out.mkdir(parents=True, exist_ok=True)
    if a.test_only:
        import csv as _csv
        _test = {r["block_id"] for r in _csv.DictReader(open(REPO / "data_csv/vobench_v1_split_v2/index_test.csv"))}
        _all = V.rows_primitive
        V.rows_primitive = lambda: [r for r in _all() if r["block_id"] in _test]
    A, Z, vid = collect(a.name)
    plot(A, out); gif(Z, vid, out, a.name)
    json.dump(timeline(Z, vid, out), open(out / "timeline.json", "w"), indent=1)
    print("saved →", out)
