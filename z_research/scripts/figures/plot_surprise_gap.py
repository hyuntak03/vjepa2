"""쌍별 surprise gap = surprise(불가능) − surprise(가능) 분포 (2026-10-25, 사용자: "VOBench · IntPhys1 · IntPhys2 · IntPhys2_late_occ 의 surprise gap mean std 그래프").

모델 (기본): 기준 AR · AR + IntPhys2 이어 학습 · AR + IntPhys2 이어 학습 + encoder LoRA
벤치마크 (시트 칸과 같은 채점 정의):
  VOBench late      — matched 쌍 (같은 block · 같은 pair_id), per_block.json 의 per_video_surprise
  IntPhys1 (A.8)    — 모델마다 A.8 규칙으로 고른 (skip, 창) 칸, Filtered (C 최소) 를 시작점 평균 (garrido_rescore 와 같은 축약)
  IntPhys2 (공식)    — 시험 장면 (MainVuniTest), Overall 최고 창의 best C, surprise_avg, 쌍 = (scene, pair_id)
  IntPhys2 late occ — 위반 시점 고정 창, 시험 절반 252 쌍 (block_id)
gap > 0 이면 그 쌍을 맞힌 것. 그림: 벤치마크마다 바이올린 + 평균 ± 표준편차, 정확도 (gap > 0 비율).

  /venv/vjepa/bin/python z_research/scripts/figures/plot_surprise_gap.py --out z_training/figures/i2pft_lora_surprise_gap.png
"""
import argparse, csv, json, os, pathlib, sys
from collections import defaultdict

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
from garrido_rescore import collect, load_index, pair_key, score_table   # noqa: E402

MODELS = [  # (라벨, 학습 run, VOBench late 결과 폴더, IntPhys2 late 결과 폴더, 색)
    ("AR (no post-FT)", "natural_ar_vw_state",
     "cvpr/results/eval/testbed/vobench_v1_late_vast_ar_vw_state",
     "cvpr/results/eval/intphys1_late/intphys2_violation_uni_test_vast_ar_vw_state", "#9a9a9a"),
    ("AR + IntPhys2 post-FT", "intphys2_vuni_ar_vw_state_postft",
     "cvpr/results/eval/testbed/vobench_v1_late_vast_ar_vw_state_i2vupft",
     "cvpr/results/eval/intphys1_late/intphys2_violation_uni_test_vast_ar_vw_state_i2vupft", "#2a78d6"),
    ("AR + IntPhys2 post-FT + encoder LoRA", "intphys2_vuni_ar_vw_state_postft_lora",
     "z_research/VOBench_v1/exp_results/surprise_c16t32__vobench_v1_late_vast_ar_vw_state_i2vupft_lora",
     "cvpr/results/eval/intphys1_late/intphys2_violation_uni_test_vast_ar_vw_state_i2vupft_lora", "#eb6834"),
]


def _get(v):
    return float(v["all"] if isinstance(v, dict) else v)


def gaps_perblock(res_dir: str, index_csv: str, late_only: bool = False):
    """per_block.json 의 per_video_surprise → 쌍 (block_id, pair_id) 마다 s_imp − s_pos."""
    pv = json.load(open(REPO / res_dir / "per_block.json"))["per_video_surprise"]
    pairs = defaultdict(dict)
    for r in csv.DictReader(open(REPO / index_csv)):
        if late_only and r.get("occlusion") != "late":
            continue
        if r["video_id"] in pv:
            pairs[(r["block_id"], r.get("pair_id", ""))][int(r["plausible"])] = _get(pv[r["video_id"]])
    return np.array([p[0] - p[1] for p in pairs.values() if 0 in p and 1 in p])


def gaps_intphys1(run: str):
    """A.8 규칙으로 고른 칸 (combo) 의 Filtered 평균 surprise → 쌍 gap."""
    E = REPO / "z_training/runs" / run / "eval"
    best = collect(E, "intphys1_dev", "latest")["best"]["combo"]                 # 예: skip2_w16
    w = best.rsplit("_", 1)[-1]
    d = E / f"intphys1_sliding__intphys1_dev_latest_{w}"
    _, meta = load_index(d)
    pw = json.load(open(d / "per_window.json"))["windows"]
    vids, _, _, arr = score_table(pw, best)
    sc = {v: float(np.nanmean(np.nanmin(arr[i], axis=1))) for i, v in enumerate(vids)}
    pairs = defaultdict(dict)
    for v, s in sc.items():
        pairs[pair_key(meta[v])][int(meta[v]["plausible"])] = s
    return np.array([p[0] - p[1] for p in pairs.values() if 0 in p and 1 in p]), best


def gaps_intphys2(run: str):
    """공식 격자: Overall 최고 창 · 그 창의 best C · surprise_avg. 쌍 = (scene_index, pair_id)."""
    E = REPO / "z_training/runs" / run / "eval"
    best = None
    for d in sorted(E.glob("intphys2_main_latest_w*_MainVuniTest")):
        s = json.load(open(d / "summary.json"))
        o = (s.get("overall") or {}).get("pairwise_accuracy")
        if o is not None and (best is None or o > best[0]):
            best = (o, d, s["best_context_length"])
    if best is None:
        return None, None
    _, d, C = best
    pairs = defaultdict(dict)
    for r in csv.DictReader(open(d / "per_video.csv")):
        if int(r["context_length"]) != int(C):
            continue
        pairs[(r["scene_index"], r["pair_id"], r["condition"])][r["is_impossible"] == "False"] = float(r["surprise_avg"])
    return np.array([p[False] - p[True] for p in pairs.values() if True in p and False in p]), f"{d.name.split('_')[3]} C{C}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="z_training/figures/i2pft_lora_surprise_gap.png")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "DejaVu Sans"
    bench = ["VOBench late", "IntPhys1 (A.8)", "IntPhys2 (official, test scenes)", "IntPhys2 late occ (test half)"]
    data = {b: [] for b in bench}
    rows = []
    for lab, run, vob, i2l, col in MODELS:
        g = {}
        g["VOBench late"] = (gaps_perblock(vob, "data_csv/vobench_v1/index.csv", late_only=True), "")
        g["IntPhys1 (A.8)"] = gaps_intphys1(run)
        g2, note2 = gaps_intphys2(run)
        g["IntPhys2 (official, test scenes)"] = (g2, note2)
        g["IntPhys2 late occ (test half)"] = (gaps_perblock(i2l, "data_csv/intphys2_violation_uni/index_test.csv"), "")
        for b in bench:
            x, note = g[b]
            data[b].append((lab, col, x, note))
            if x is not None and len(x):
                rows.append((lab, b, len(x), x.mean(), x.std(), 100 * (x > 0).mean(), note))
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
    for ax, b in zip(axes, bench):
        xs = [(lab, col, x) for lab, col, x, _ in data[b] if x is not None and len(x)]
        pos = np.arange(len(xs))
        vp = ax.violinplot([x for _, _, x in xs], positions=pos, showextrema=False, widths=0.8)
        for body, (_, col, _) in zip(vp["bodies"], xs):
            body.set_facecolor(col); body.set_alpha(0.35); body.set_edgecolor(col)
        for p, (lab, col, x) in zip(pos, xs):
            m, s = x.mean(), x.std()
            ax.errorbar(p, m, yerr=s, fmt="o", color=col, capsize=5, lw=2, ms=6)
            ax.text(p, ax.get_ylim()[1] if False else m + s, f"{m:+.4f}\n±{s:.4f}\nacc {100 * (x > 0).mean():.1f}%",
                    ha="center", va="bottom", fontsize=7.5, color=col)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xticks(pos); ax.set_xticklabels(["AR", "+IntPhys2", "+IntPhys2\n+LoRA"][: len(xs)], fontsize=9)
        ax.set_title(b, fontsize=10)
        ax.set_ylabel("surprise(impossible) − surprise(possible)" if b == bench[0] else "")
        lo, hi = np.percentile(np.concatenate([x for _, _, x in xs]), [1, 99])
        pad = 0.35 * (hi - lo)
        ax.set_ylim(lo - 0.1 * (hi - lo), hi + pad)
    fig.tight_layout()
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print("saved", out)
    print(f"{'model':40s} {'benchmark':34s} {'n':>4s} {'mean':>9s} {'std':>8s} {'acc%':>6s}  note")
    for lab, b, n, m, s, acc, note in rows:
        print(f"{lab:40s} {b:34s} {n:4d} {m:+9.5f} {s:8.5f} {acc:6.1f}  {note}")


if __name__ == "__main__":
    main()
