#!/usr/bin/env python3
"""predictions.json -> confusion matrix. head 재학습이 필요 없다.

eval.py 가 probing 을 마치면 <output_dir>/predictions.json 에 val 행별 예측을 덤프한다
(evals/world_model_analysis/eval.py:1002). 구조는 자기완결적이다:

    val_video_ids : [str]                      val 행의 신원
    val_groups    : [str]                      행별 condition
    targets       : {target: {classes, gold}}  클래스 이름 + 정답 (행 순서 동일)
    heads         : [{fit, groups, target, eval, pred}]   행별 예측 클래스 인덱스

§5-4c 의 confusion 표는 이 파일이 없던 시절 head 를 5분씩 재학습해 뽑았다. 이제 불필요하다.

  # 뭐가 들어 있는지
  python z_research/scripts/plot_confusion.py --predictions <dir>/predictions.json --list

  # p self-probe 의 shape confusion, 조건별로 쪼개서
  python z_research/scripts/plot_confusion.py \
      --predictions z_research/IntPhysGenV10/exp_results/attn_probe__v10_vith/predictions.json \
      --fit pred__f17to32 --target shape --by-condition \
      --output z_research/IntPhysGenV10/figures/fig4_p_self_confusion.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

BLUE = "#2a78d6"
INK, INK2, MUTED = "#000000", "#3b3b3b", "#9a9a9a"
CONDS = ["static_visible", "moving_visible", "moving_occlusion", "static_occlusion"]
CLABEL = {"static_visible": "Static, no occluder", "moving_visible": "Moving, no occluder",
          "moving_occlusion": "Moving, occluded", "static_occlusion": "Static, occluded"}

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
CMAP = LinearSegmentedColormap.from_list("wblue", ["#ffffff", BLUE])


def pick(meta, fit, target, ev, groups):
    """조건에 맞는 head 하나. eval 을 안 주면 self (eval == fit)."""
    want_ev = ev or fit
    cand = [h for h in meta["heads"]
            if h["target"] == target and h["fit"] == fit and h["eval"] == want_ev
            and (h.get("groups") or None) == (groups or None)]
    if len(cand) != 1:
        raise SystemExit(
            f"head 를 특정하지 못했다 ({len(cand)}개). --list 로 확인할 것.\n"
            f"  fit={fit} target={target} eval={want_ev} groups={groups}")
    return cand[0]


def confusion(gold, pred, K):
    M = np.zeros((K, K), dtype=int)
    for g, p in zip(gold, pred):
        M[g, p] += 1
    return M


def draw(ax, M, classes, title, show_y=True, show_x=True):
    K = len(classes)
    row = M.sum(1, keepdims=True)
    R = np.divide(M, np.where(row == 0, 1, row)) * 100
    ax.imshow(R, cmap=CMAP, vmin=0, vmax=100, aspect="equal")
    for i in range(K):
        for j in range(K):
            if M[i, j] == 0:
                continue
            ax.text(j, i, f"{R[i, j]:.0f}", ha="center", va="center",
                    fontsize=5.8 if K > 6 else 7.0,
                    color="white" if R[i, j] > 55 else INK2)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels(classes if show_x else [""] * K, fontsize=6.4,
                       rotation=90, color=INK2)
    ax.set_yticklabels(classes if show_y else [""] * K, fontsize=6.4, color=INK2)
    ax.tick_params(length=0, pad=1.5)
    for s in ax.spines.values():
        s.set_color(MUTED); s.set_linewidth(0.5)
    acc = 100 * np.trace(M) / max(1, M.sum())
    ax.set_title(f"{title}\nacc {acc:.1f}%  (n={M.sum()})", fontsize=8.0, color=INK, pad=4)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--fit", default=None)
    ap.add_argument("--target", default=None)
    ap.add_argument("--eval", dest="ev", default=None, help="기본: self (eval == fit)")
    ap.add_argument("--groups", nargs="*", default=None)
    ap.add_argument("--by-condition", action="store_true")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    meta = json.loads(a.predictions.read_text())
    nval = len(meta["val_video_ids"])

    if a.list or not (a.fit and a.target):
        print(f"val {nval}행 | targets {list(meta['targets'])} | head {len(meta['heads'])}개\n")
        print(f"  {'fit':<22}{'target':<8}{'groups':<20}{'eval':<22}")
        for h in meta["heads"]:
            g = ",".join(h["groups"]) if h.get("groups") else "-"
            tag = "(self)" if h["eval"] == h["fit"] else ""
            print(f"  {h['fit']:<22}{h['target']:<8}{g:<20}{h['eval']:<22}{tag}")
        if not (a.fit and a.target):
            return

    h = pick(meta, a.fit, a.target, a.ev, a.groups)
    T = meta["targets"][a.target]
    classes, gold = T["classes"], np.asarray(T["gold"])
    pred = np.asarray(h["pred"])
    if len(pred) != nval or len(gold) != nval:
        raise SystemExit(f"길이 불일치: gold {len(gold)} / pred {len(pred)} / val {nval}")
    grp = np.asarray(meta["val_groups"])
    K = len(classes)

    tag = f"{h['fit']} -> {h['eval']}" if h["eval"] != h["fit"] else f"{h['fit']} (self)"
    print(f"\n{tag} | target={a.target} | groups={h.get('groups') or '전체'}")

    if a.by_condition:
        conds = [c for c in CONDS if (grp == c).any()] or sorted(set(grp))
        fig, axes = plt.subplots(1, len(conds),
                                 figsize=(1.72 * len(conds) + 0.55, 2.55))
        axes = np.atleast_1d(axes)
        for i, (ax, c) in enumerate(zip(axes, conds)):
            m = grp == c
            M = confusion(gold[m], pred[m], K)
            acc = draw(ax, M, classes, CLABEL.get(c, c), show_y=(i == 0))
            print(f"  {c:<18} acc {acc:6.2f}%  n={M.sum()}")
    else:
        fig, ax = plt.subplots(figsize=(2.9, 2.9))
        M = confusion(gold, pred, K)
        acc = draw(ax, M, classes, tag)
        print(f"  overall            acc {acc:6.2f}%  n={M.sum()}")
        # 대각선 밖에서 가장 큰 오분류
        off = [(M[i, j], classes[i], classes[j]) for i in range(K) for j in range(K) if i != j]
        for n, i, j in sorted(off, reverse=True)[:5]:
            if n:
                print(f"    {i:>12} -> {j:<12} {n:4d}  ({100*n/max(1,M[classes.index(i)].sum()):.0f}% of {i})")

    # summary.json 이 옆에 있으면 정확도를 대조한다
    s = a.predictions.parent / "summary.json"
    if s.exists():
        rep = json.loads(s.read_text()).get("probing", [])
        for r in rep:
            if (r["fit"] == h["fit"] and r["target"] == a.target
                    and (r.get("groups") or None) == (h.get("groups") or None)):
                want = r["evals"].get(h["eval"], {}).get("overall")
                if want is not None:
                    M = confusion(gold, pred, K)
                    got = np.trace(M) / max(1, M.sum())
                    ok = "OK" if abs(got - want) < 1e-6 else "불일치!"
                    print(f"  검증 {ok} — 재계산 {got*100:.4f}% vs summary.json {want*100:.4f}%")

    fig.tight_layout(pad=0.4)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        for ext in (".pdf", ".png"):
            fig.savefig(a.output.with_suffix(ext), dpi=400, bbox_inches="tight",
                        facecolor="white")
        print(f"[saved] {a.output.with_suffix('.pdf')}  +  .png")


if __name__ == "__main__":
    main()
