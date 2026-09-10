#!/usr/bin/env python3
"""knockout results.json -> markdown 표. README 의 `### 11. attention knockout` 절이 이걸로 만들어진다.

  python z_research/scripts/analysis/knockout_md.py            # 표를 stdout 으로
  python z_research/scripts/analysis/knockout_md.py --write    # README 의 marker 사이를 교체

수치는 전부 `breakdown` (merge.py 가 score_blocks 로 다시 채점한 block 평균) 에서 온다.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "z_research/IntPhysGenV11_occlusion_timing_ablation/exp_results/knockout__v11_full_vith/results.json"
README = ROOT / "z_research/IntPhysGenV11_occlusion_timing_ablation/README.md"
BEGIN, END = "<!-- knockout_md:begin -->", "<!-- knockout_md:end -->"

COLS = [("clean_null", "기준선"), ("mask->ctx@L0-3", "m←c L0-3"), ("mask->ctx@L4-10", "m←c L4-10"),
        ("mask->ctx@L8-11", "m←c L8-11"), ("mask->ctx@all", "m←c all"),
        ("mask->mask@all", "m←m all"), ("ctx->ctx@all", "c←c all")]
TIM = ["visible", "early", "mid", "late"]
VIOL = ["vanish", "shape", "color"]
MOT = ["static", "moving_flat", "moving"]


def pool(B, axis, keep):
    rows = [v for k, v in B[axis].items() if keep(k)]
    nb = sum(r["n_block"] for r in rows)
    return {n: sum(r["n_block"] * r["acc"][n] for r in rows) / nb for n in rows[0]["acc"]}, \
        sum(r["n_pair"] for r in rows)


def table(title, levels, get):
    out = [f"**{title}**", "", "| | " + " | ".join(c for _, c in COLS) + " | n_pair |",
           "|---|" + "---:|" * (len(COLS) + 1)]
    for lv, label in levels:
        acc, npair = get(lv)
        out.append(f"| {label} | " + " | ".join(f"{acc[n]:.1f}" for n, _ in COLS) + f" | {npair} |")
    return "\n".join(out) + "\n"


def build(R) -> str:
    B = R["breakdown"]
    L = R["meta"]["n_layers"]
    s = [f"기준선 `clean_null` **{R['clean_null']['surprise_acc']['acc']:.2f}%** "
         f"(clean {R['clean']['surprise_acc']['acc']:.2f}%, n_pair {R['clean_null']['surprise_acc']['n_pair']}, "
         f"명세 {len(R['specs'])}개, 창 폭 {R['meta']['window'][0]}, keep_self {R['meta']['keep_self']}). "
         f"열 이름 `m←c` = `mask->ctx` (mask query 가 ctx key 를 못 읽음), `m←m` = `mask->mask`, `c←c` = `ctx->ctx`.", ""]
    # 층 프로파일 (전체)
    s += ["**층 프로파일 — 전체 (창 중심 층, ±3)**", "",
          "| 간선 | " + " | ".join(str(i) for i in range(L)) + " | all |",
          "|---|" + "---:|" * (L + 1)]
    acc_all, _ = pool(B, "condition", lambda k: True)
    for e in ["mask->ctx", "mask->mask", "ctx->ctx"]:
        names = []
        for i in range(L):
            a, b = max(0, i - 3), min(L - 1, i + 3)
            names.append(f"{e}@L{a}-{b}")
        s.append(f"| `{e}` | " + " | ".join(f"{acc_all[n]:.1f}" for n in names) + f" | {acc_all[e + '@all']:.1f} |")
    s.append("")
    s.append(table("타이밍", [(t, t) for t in TIM], lambda t: pool(B, "timing", lambda k: k == t)))
    s.append(table("위반", [(v, v) for v in VIOL], lambda v: pool(B, "violation", lambda k: k == v)))
    s.append(table("운동", [(m, m) for m in MOT], lambda m: pool(B, "motion", lambda k: k == m)))
    s.append(table("k (가림 조건만 — visible 의 k 는 명목값)", [(k, f"k={k}") for k in "1234"],
                   lambda kk: pool(B, "k|timing", lambda k: k.split("|")[0] == kk and k.split("|")[1] != "visible")))
    s.append(table("타이밍 × 위반", [(f"{t}|{v}", f"{t} / {v}") for t in TIM for v in VIOL],
                   lambda key: pool(B, "timing|violation", lambda k: k == key)))
    s.append(table("타이밍 × 운동", [(f"{t}|{m}", f"{t} / {m}") for t in TIM for m in MOT],
                   lambda key: pool(B, "timing|motion", lambda k: k == key)))
    # drift
    d = {r["name"]: r["metrics"]["pred_drift"] for r in R["specs"] if "pred_drift" in r["metrics"]}
    if d:
        s += ["**예측 이동 `pred_drift` (전체, `|p−p_clean|/|p_clean|` · cosine)**", "",
              "| 간선 | " + " | ".join(str(i) for i in range(L)) + " | all |", "|---|" + "---:|" * (L + 1)]
        for e in ["mask->ctx", "mask->mask", "ctx->ctx"]:
            row = []
            for i in range(L):
                a, b = max(0, i - 3), min(L - 1, i + 3)
                row.append(f"{d[f'{e}@L{a}-{b}']['rel_l1']:.3f}")
            s.append(f"| `{e}` | " + " | ".join(row) + f" | {d[e + '@all']['rel_l1']:.3f} / cos {d[e + '@all']['cosine']:.3f} |")
        s.append("")
    return "\n".join(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=RES)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    md = build(json.loads(a.results.read_text()))
    if not a.write:
        print(md); return
    s = README.read_text()
    if BEGIN not in s or END not in s:
        raise SystemExit(f"{README} 에 marker 가 없다")
    s = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), BEGIN + "\n" + md + "\n" + END, s, flags=re.S)
    README.write_text(s); print(f"[written] {README}")


if __name__ == "__main__":
    main()
