#!/usr/bin/env python3
"""v11 가림 — 물체가 가림막 밖으로 나오는가: context encoder z vs predictor p (학습셋 v5 자). timing × k 마다 그림 한 장.

가림막 기하 (metadata): 가림막 중심 x = 가려진 구간 한가운데 샘플의 물체 x, 폭 = occ_apparent_px, 진행 방향 dir = sign(x[31] − x[0]),
출구 모서리 edge = 중심 + dir·폭/2.  s = (x − edge)·dir  (px, 288 화면).
  s ≤ −18 = 완전히 가려짐 (물체 반폭 18) / −18 < s < 18 = 걸쳐 나옴 / s ≥ 18 = 완전히 나옴 / s ≤ −폭 − 18 = 가림막에 아직 안 들어감.
  검증 (2026-09-14, late 1,792 clip): 마지막 가려진 샘플에서 진실 s = −18.0, reveal_frame 샘플에서 −8.9 (sd 0.5).
timing: late = 문맥 끝 + 미래 앞 k 샘플 가림 (미래 안에서 나온다) / early·mid = 문맥 안에서만 가림 (미래 전에 이미 나온다).

표현 (vanish 블록 pos_a, 셀당 56 clip):
  z = context encoder 에 32 frames (isolated_ctx_0_32) → 16 튜블릿 전부 = 실제 프레임을 본 기준
  p = predictor 출력 (v11_full_vith/predictor.npy) → 미래 8 튜블릿 (문맥만 보고 만든 미래)
튜블릿마다 상태:
  object-like = 자 attention 3×3 창 최대 질량 ≥ GATE (균등 0.035). 못 넘으면 자 출력은 토큰 평균 = 기본값 → 위치로 읽지 않는다 ("no object")
  out         = object-like ∧ s > 0 (읽은 중심이 출구 모서리를 넘음) / not out = object-like ∧ s ≤ 0
  ⚠️ object-like 는 attention 이 한곳에 몰렸다는 뜻이지 그곳이 물체라는 확인이 아니다 (v11 late 의 p 는 진실 3×3 질량이 0.03~0.16, attn_diag.md)

출력: figures/<train>/v11_vanish/emergence/
  by_condition/fig_<timing>_k<k>.png/.pdf   행 = flat / ramp, 열 = (a) 위치 (b) z 상태 (c) p 상태
  data/<motion>_<timing>_k<k>.npz            clip 단위 s·gate (그림은 여기서만 그린다)
  emergence.{json,md}                        튜블릿별 비율·평균, gate 민감도

  ROLLOUT2_TRAIN=v5 python z_research/scripts/figures/plot_v11_emergence.py            # 계산 + 그림
  ROLLOUT2_TRAIN=v5 python z_research/scripts/figures/plot_v11_emergence.py --plot-only
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))
import rollout2_test_readout as rt                                 # noqa: E402
from rollout2_attn_readout import AttnReadout                      # noqa: E402

META = Path("/local_datasets/world/world_analysis/IntPhysGen_v11/metadata.csv")
INDEX = ROOT / "data_csv/intphysgen_v11_full/index_probe.csv"
CACHE = Path("/local_datasets/world/world_analysis/cache")
SRC = {"p": (CACHE / "v11_full_vith", "predictor.npy", 8), "z": (CACHE / "v11_vanish_all_ctx32_vith", "isolated_ctx_0_32.npy", 16)}
OUT = rt.FIG_ROOT / "v11_vanish/emergence"
S, D, RES, HALF = 256, 1280, 144.0, 18.0
GATE, GATES = 0.15, (0.10, 0.15, 0.25)
MOTIONS, TIMINGS, KS = ("flat", "ramp"), ("late", "mid", "early"), ("1", "2", "3", "4")
COND = {("flat", "late"): "moving_occlusion_flat", ("flat", "mid"): "moving_occlusion_flat_mid", ("flat", "early"): "moving_occlusion_flat_early",
        ("ramp", "late"): "moving_occlusion", ("ramp", "mid"): "moving_occlusion_mid", ("ramp", "early"): "moving_occlusion_early"}
C_TRUTH, C_Z, C_P = "#222222", "#2a78d6", "#eb6834"
C_OUT, C_NOT, C_DIFF = "#1baf7a", "#6b6b6b", "#dcdcdc"


def max3(a):
    """(T, 256) attention → (T,) 3×3 창 최대 질량."""
    m = torch.from_numpy(a).reshape(-1, 1, 16, 16)
    return torch.nn.functional.avg_pool2d(m, 3, stride=1, padding=1, count_include_pad=True).mul(9).amax((1, 2, 3)).numpy()


def compute():
    idx = list(csv.DictReader(INDEX.open())); meta = {r["name"]: r for r in csv.DictReader(META.open())}
    models, arrays, rows = {}, {}, {}
    for rep, (c, f, _) in SRC.items():
        m = AttnReadout(); m.load_state_dict(torch.load(rt.RES_ROOT / f"attentive_pooling/{rep}/attn.pt", map_location="cpu")); m.eval(); models[rep] = m
        arrays[rep] = np.load(c / f, mmap_mode="r"); rows[rep] = {v: i for i, v in enumerate(json.loads((c / "meta.json").read_text())["video_ids"])}
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    for motion in MOTIONS:
        for timing in TIMINGS:
            for K in KS:
                pos = [r for r in idx if r["condition"] == COND[(motion, timing)] and r["sym_k"] == K and r["violation_type"] == "vanish" and r["variant"] == "pos_a"]
                acc = {k: [] for k in ("s_truth", "s_z", "g_z", "s_p", "g_p", "occ_w")}
                for r in pos:
                    mt = meta[r["video_id"]]
                    x = rt.arr(mt["object_px_x_by_sample"]); y = rt.arr(mt["object_px_y_by_sample"])
                    mid = (int(mt["hidden_start"]) // 3 + int(mt["hidden_end"]) // 3) // 2
                    dirn = np.sign(x[31] - x[0]); w = float(mt["occ_apparent_px"]); edge = x[mid] + dirn * w / 2
                    tub = np.stack([x, y], -1).reshape(16, 2, 2).mean(1)
                    acc["s_truth"].append((tub[:, 0] - edge) * dirn); acc["occ_w"].append(w)
                    for rep, (_, _, n_tub) in SRC.items():
                        tok = torch.from_numpy(np.asarray(arrays[rep][rows[rep][r["video_id"]]], np.float32)).reshape(n_tub, S, D)
                        with torch.no_grad():
                            pr, a = models[rep](tok)
                        acc[f"s_{rep}"].append(((pr.numpy() + 1) * RES)[:, 0] * dirn - edge * dirn); acc[f"g_{rep}"].append(max3(a.numpy()))
                np.savez(OUT / "data" / f"{motion}_{timing}_k{K}.npz", video_id=np.array([r["video_id"] for r in pos]), **{k: np.array(v) for k, v in acc.items()})
                print(f"  {motion} {timing} k={K}: n={len(pos)}", flush=True)


def load(motion, timing, K):
    d = np.load(OUT / "data" / f"{motion}_{timing}_k{K}.npz")
    return {k: d[k] for k in d.files}


def states(s, g, w, gate=GATE):
    """(n, T) s·gate, (n,) 가림막 폭 → before / at·behind / out / no object 비율 (T,) 넷.
    before = 물체 앞 모서리가 아직 입구에 안 닿음 (s ≤ −w − 18), at·behind = 가림막에 걸치거나 뒤 (−w − 18 < s ≤ 0), out = s > 0."""
    ol = g >= gate; before = s <= -w[:, None] - HALF
    return (ol & before).mean(0), (ol & ~before & (s <= 0)).mean(0), (ol & (s > 0)).mean(0), (~ol).mean(0)


def summarize():
    res = {}
    for motion in MOTIONS:
        for timing in TIMINGS:
            for K in KS:
                d = load(motion, timing, K); c = dict(n=int(len(d["s_truth"])), occ_w_median=float(np.median(d["occ_w"])),
                                                     truth_out=(d["s_truth"] > 0).mean(0).tolist(), truth_hidden=(d["s_truth"] <= -HALF).mean(0).tolist())
                for rep in ("z", "p"):
                    b_, a_, o, df = states(d[f"s_{rep}"], d[f"g_{rep}"], d["occ_w"])
                    c[f"{rep}_before"], c[f"{rep}_at"], c[f"{rep}_out"], c[f"{rep}_noobj"] = b_.tolist(), a_.tolist(), o.tolist(), df.tolist()
                    c[f"{rep}_s_median"] = np.median(d[f"s_{rep}"], 0).tolist()
                    c[f"{rep}_out_gate"] = {str(gg): states(d[f"s_{rep}"], d[f"g_{rep}"], d["occ_w"], gg)[2].tolist() for gg in GATES}
                res[f"{motion} {timing} k={K}"] = c
    json.dump(dict(gate=GATE, half_px=HALF, note="z: 16 tubelets (0-7 context, 8-15 future); p: 8 future tubelets", cells=res), open(OUT / "emergence.json", "w"), indent=1)
    f = lambda v: " | ".join(f"{x:.2f}" for x in v)
    md = ["# v11 가림 — 물체가 가림막 밖으로 나오는가 (z vs p)", "",
          f"`plot_v11_emergence.py`. 튜블릿 상태 (자 attention 3×3 최대 질량 ≥ {GATE} 일 때만 위치로 읽음): before = 가림막 입구 전, at/behind = 걸치거나 뒤 (출구 모서리 안쪽), out = 출구 모서리를 넘음; no object = 질량 < {GATE}.",
          "z 는 16 튜블릿 (0–7 문맥, 8–15 미래), p 는 미래 8 튜블릿. 진실 out = 진실 중심이 모서리를 넘음. 셀당 56 clip (vanish pos_a).",
          "⚠️ object-like (질량 ≥ gate) 는 attention 이 몰렸다는 뜻이지 그곳이 물체라는 확인이 아니다.", ""]
    head = "| | " + " | ".join(f"t{i}" for i in range(8, 16)) + " |"
    for key, lab in (("truth_out", "진실 out (미래 8 튜블릿)"), ("z_out", "z out"), ("p_out", "p out"), ("p_at", "p at/behind (object-like, 출구 모서리 안쪽)"), ("p_noobj", "p no object")):
        md += [f"## {lab}", "", head, "|---|" + "---|" * 8]
        md += [f"| {name} | " + f(c[key][-8:]) + " |" for name, c in res.items()]; md.append("")
    md += ["## gate 민감도 — p out (마지막 튜블릿)", "", "| | " + " | ".join(f"gate {g}" for g in GATES) + " |", "|---|" + "---|" * len(GATES)]
    md += [f"| {name} | " + " | ".join(f"{c['p_out_gate'][str(g)][-1]:.2f}" for g in GATES) + " |" for name, c in res.items()]
    (OUT / "emergence.md").write_text("\n".join(md) + "\n")


def plot():
    plt.rcParams.update({"font.family": "Nimbus Roman", "pdf.fonttype": 42, "font.size": 8})
    (OUT / "by_condition").mkdir(parents=True, exist_ok=True); T16 = np.arange(16); rng = np.random.RandomState(0)
    for timing in TIMINGS:
        for K in KS:
            fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.6), gridspec_kw=dict(width_ratios=[1.35, 1, 1])); lab = iter("abcdef")
            D_ = {m: load(m, timing, K) for m in MOTIONS}
            lo = min(min(D_[m]["s_truth"].min(), np.percentile(D_[m]["s_p"], 1), np.percentile(D_[m]["s_z"], 1)) for m in MOTIONS) - 10
            hi = max(max(D_[m]["s_truth"].max(), np.percentile(D_[m]["s_p"], 99), np.percentile(D_[m]["s_z"], 99)) for m in MOTIONS) + 10
            for i, motion in enumerate(MOTIONS):
                d = D_[motion]; w = float(np.median(d["occ_w"]))
                # (a) 위치: 출구 모서리 기준 거리. 회색 띠 = 가림막 (중앙 폭)
                ax = axes[i, 0]
                ax.axhspan(-w, 0, color="#bdbdbd", alpha=0.45, lw=0)
                ax.axhline(0, color="k", lw=0.6, ls=":"); ax.axvline(7.5, color="k", lw=0.5)
                tm = np.median(d["s_truth"], 0); ax.plot(T16, tm, color=C_TRUTH, lw=2.0)
                zm, zq1, zq3 = np.median(d["s_z"], 0), np.percentile(d["s_z"], 25, 0), np.percentile(d["s_z"], 75, 0)
                ax.plot(T16, zm, color=C_Z, lw=1.5); ax.fill_between(T16, zq1, zq3, color=C_Z, alpha=0.15, lw=0)
                ol = d["g_p"] >= GATE
                for t in range(8):
                    xs = 8 + t + rng.uniform(-0.28, 0.28, len(d["s_p"]))
                    ax.scatter(xs[~ol[:, t]], d["s_p"][~ol[:, t], t], s=4, color=C_DIFF, edgecolors="none", zorder=2)
                    ax.scatter(xs[ol[:, t]], d["s_p"][ol[:, t], t], s=4, color=C_P, edgecolors="none", zorder=3)
                ax.set_ylim(lo, hi); ax.set_xlim(-0.5, 15.5); ax.set_xticks([0, 4, 8, 12, 15])
                ax.set_ylabel(f"{motion}\nposition past exit edge (px)")
                ax.text(3.75, hi, "context", ha="center", va="top", fontsize=7); ax.text(11.75, hi, "future", ha="center", va="top", fontsize=7)
                # (b)(c) 상태 막대
                for j, rep in ((1, "z"), (2, "p")):
                    ax = axes[i, j]; b_, a_, o, df = states(d[f"s_{rep}"], d[f"g_{rep}"], d["occ_w"]); tt = T16 if rep == "z" else np.arange(8, 16)
                    ax.bar(tt, o, 0.8, color=C_OUT); ax.bar(tt, a_, 0.8, bottom=o, color=C_NOT)
                    ax.bar(tt, b_, 0.8, bottom=o + a_, color="white", edgecolor=C_NOT, hatch="////", lw=0.4); ax.bar(tt, df, 0.8, bottom=o + a_ + b_, color=C_DIFF)
                    ax.step(np.r_[T16 - 0.5, 15.5], np.r_[(d["s_truth"] > 0).mean(0), (d["s_truth"] > 0).mean(0)[-1]], where="post", color=C_TRUTH, lw=1.6)
                    ax.axvline(7.5, color="k", lw=0.5); ax.set_xlim(-0.5, 15.5); ax.set_ylim(0, 1.02); ax.set_xticks([0, 4, 8, 12, 15])
                    if i == 0:
                        ax.set_title("context encoder z (sees all frames)" if rep == "z" else "predictor p (context only)", fontsize=8)
                    if j == 1:
                        ax.set_ylabel("fraction of clips")
                    else:
                        ax.set_yticklabels([])
                for ax in axes[i]:
                    ax.spines[["top", "right"]].set_visible(False)
                    ax.set_xlabel(("tubelet" if i == 1 else "") + f"\n({next(lab)})")
            axes[0, 0].set_title(f"{timing} occlusion, k={K}", fontsize=8)
            handles = [Patch(color="#bdbdbd", alpha=0.45, label="occluder (median width)"), Line2D([], [], color="k", lw=0.6, ls=":", label="occluder exit edge"),
                       Line2D([], [], color=C_TRUTH, lw=2, label="truth (median) / fraction truly out"),
                       Line2D([], [], color=C_Z, lw=1.5, label="z readout (median, IQR)"),
                       Line2D([], [], marker="o", ls="", color=C_P, ms=3, label="p readout, one clip (object-like attention)"),
                       Line2D([], [], marker="o", ls="", color=C_DIFF, ms=3, label="p readout, one clip (diffuse = default output)"),
                       Patch(color=C_OUT, label="readout: out, past exit edge"), Patch(color=C_NOT, label="readout: at / behind occluder"),
                       Patch(facecolor="white", edgecolor=C_NOT, hatch="////", label="readout: before occluder"),
                       Patch(color=C_DIFF, label="no object-like readout (diffuse attention)")]
            fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.005))
            fig.tight_layout(rect=(0, 0.16, 1, 1))
            for ext in ("png", "pdf"):
                fig.savefig(OUT / "by_condition" / f"fig_{timing}_k{K}.{ext}", dpi=200)
            plt.close(fig)
    print(f"→ {OUT}/by_condition/fig_<late|mid|early>_k<1..4>.png/.pdf, emergence.{{json,md}}")


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        compute()
    summarize(); plot()
