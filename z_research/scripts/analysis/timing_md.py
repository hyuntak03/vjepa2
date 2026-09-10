#!/usr/bin/env python3
"""v11_full 전수 기록을 markdown 으로 뽑는다 (README 의 "전수 기록" 절).

채점 117 cell + probing 108 head 를 축마다 접어서 표로 만든다. 손으로 옮겨 적지 않는다 —
`summary.json` / `report.json` / `per_block.json` 에서 매번 다시 계산한다.

  python z_research/scripts/analysis/timing_md.py            # 표준출력
  python z_research/scripts/analysis/timing_md.py --write    # README 의 절을 갈아끼운다
"""
from __future__ import annotations
import argparse, collections, csv, json
from pathlib import Path
import numpy as np

E = Path("z_research/IntPhysGenV11_occlusion_timing_ablation/exp_results")
IDX = Path("data_csv/intphysgen_v11_full/index_probe.csv")
README = Path("z_research/IntPhysGenV11_occlusion_timing_ablation/README.md")
MARK = "## 전수 기록"
PT = {"contextF__f1to16": "z", "pred__f17to32": "p", "targetF__f17to32": "h"}
VIOL = ["vanish", "shape", "color"]
VL = {"vanish": "permanence", "shape": "shape", "color": "colour"}
MOT = ["static", "flat", "ramp"]
ML = {"static": "Static", "flat": "Moving (flat)", "ramp": "Moving (ramp)"}
TIM = ["vis", "early", "mid", "late"]
KS = ["1", "2", "3", "4"]
ARM = {"static": ["static_visible", "static_occlusion_early", "static_occlusion_mid",
                  "static_occlusion"],
       "flat": ["moving_visible_flat", "moving_occlusion_flat_early",
                "moving_occlusion_flat_mid", "moving_occlusion_flat"],
       "ramp": ["moving_visible", "moving_occlusion_early", "moving_occlusion_mid",
                "moving_occlusion"]}
COND = {g: (m, t) for m, gs in ARM.items() for t, g in zip(TIM, gs)}


def mot_of(c): return "static" if c.startswith("static") else ("flat" if "flat" in c else "ramp")


def tbl(head, rows):
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    R = json.loads((E / "report.json").read_text())
    ov = R["scoring"]["overall"][0]
    cells = [dict(c, _m=mot_of(c["condition"]), _t=(c.get("occ_timing") or "vis"))
             for c in R["scoring"]["cells"]]
    tot = sum(c["n"] for c in cells)
    wa = sum(c["acc"] * c["n"] for c in cells) / tot
    assert ov["verified"] and abs(wa - ov["overall"]) < 0.02

    def A(*ks):
        d = collections.defaultdict(lambda: [0.0, 0])
        for c in cells:
            k = tuple(c[x] for x in ks); d[k][0] += c["acc"] * c["n"]; d[k][1] += c["n"]
        return {k: (v[0] / v[1], v[1]) for k, v in d.items()}

    P = json.loads((E / "attn_probe_xfer__v11_full_vith" / "summary.json").read_text())["probing"]
    X, ch, tr = {}, {}, {}
    for r in P:
        ch[r["target"]] = 100 * r["chance"]
        for ev, cell in r["evals"].items():
            for g, c in cell["per_group"].items():
                X[(r["target"], PT[r["fit"]], PT.get(ev, ev), r["groups"][0], g)] = 100 * c["acc"]

    M = json.loads((E / "attn_probe_xfer__v11_full_vith" / "predictions.json").read_text())
    meta = {r["video_id"]: r for r in csv.DictReader(IDX.open())}
    vid = M["val_video_ids"]
    cond = np.array([meta[v]["condition"] for v in vid])
    kk = np.array([meta[v]["sym_k"] for v in vid])
    gold = {t: np.asarray(M["targets"][t]["gold"]) for t in M["targets"]}
    HP = {(h["target"], PT[h["fit"]], PT.get(h["eval"], h["eval"]), h["groups"][0]):
          np.asarray(h["pred"]) for h in M["heads"]}

    def acc(pr, g, m): return 100 * float((pr[m] == g[m]).mean()) if m.sum() else float("nan")

    S = json.loads((E / "surprise_c16t32__v11_full_vith" / "per_block.json").read_text())
    S = S["per_video_surprise"]
    blk = collections.defaultdict(dict)
    for v, s in S.items():
        blk[meta[v]["source_block"]][meta[v]["variant"]] = (s, meta[v])
    mg = collections.defaultdict(list)
    for d in blk.values():
        for pos, imp in (("pos_a", "imp_ab"), ("pos_b", "imp_ba")):
            if pos in d and imp in d:
                sp, mp = d[pos]; si, _ = d[imp]
                mg[(mp.get("occ_timing") or "vis", mp["violation_type"])].append((sp, si))

    L = [MARK, "",
         f"`timing_md.py` 가 산출물에서 매번 다시 계산한다. 손으로 고치지 말 것.",
         "",
         f"**overall {ov['overall']:.2f}%** · n={ov['n_pair']} matched pair · "
         f"채점 cell {len(cells)} · probing head {len(P)} · chance 50%", ""]

    # ── 채점 ────────────────────────────────────────────────────────────────
    L += ["### 1. 채점 — 타이밍 × 위반", "",
          tbl(["timing"] + [VL[v] for v in VIOL] + ["전체", "n"],
              [[t] + [f"{A('_t','violation_type')[(t,v)][0]:.1f}" for v in VIOL]
               + [f"**{A('_t')[(t,)][0]:.1f}**", f"{A('_t')[(t,)][1]}"] for t in TIM]), "",
          "### 2. 채점 — 타이밍 × 운동", "",
          tbl(["timing"] + [ML[m] for m in MOT],
              [[t] + [f"{A('_t','_m')[(t,m)][0]:.1f}" for m in MOT] for t in TIM]), "",
          "### 3. 채점 — 운동 × 위반 × 타이밍", "",
          tbl(["motion", "violation"] + TIM,
              [[ML[m], VL[v]] + [f"{A('_m','violation_type','_t')[(m,v,t)][0]:.1f}" for t in TIM]
               for m in MOT for v in VIOL]), "",
          "### 4. 채점 — k × 위반 × 타이밍", "",
          tbl(["violation", "timing"] + [f"k={k}" for k in KS] + ["k=0"],
              [[VL[v], t] + [f"{A('sym_k','violation_type','_t')[(k,v,t)][0]:.1f}" for k in KS]
               + [f"{A('violation_type','_t')[(v,'vis')][0]:.1f}"]
               for v in VIOL for t in ("early", "mid", "late")]), "",
          "### 5. 채점 — k × 운동 × 타이밍 (위반 접음)", "",
          tbl(["motion", "timing"] + [f"k={k}" for k in KS],
              [[ML[m], t] + [f"{A('sym_k','_m','_t')[(k,m,t)][0]:.1f}" for k in KS]
               for m in MOT for t in ("early", "mid", "late")]), ""]

    # ── probing ─────────────────────────────────────────────────────────────
    L += [f"### 6. probing self — 자기 조건 head (chance shape/colour {ch['shape']:.1f}, "
          f"env {ch['env']:.1f})", "",
          tbl(["condition", "shape z", "shape h", "shape p", "colour z", "colour h", "colour p",
               "env p"],
              [[g] + [f"{X[(t,p,p,g,g)]:.1f}" for t in ("shape", "color") for p in ("z", "h", "p")]
               + [f"{X[('env','p','p',g,g)]:.1f}"]
               for m in MOT for g in ARM[m]]), "",
          "### 7. probing 이식 — 비가림 head 를 같은 팔의 가림 조건에", "",
          tbl(["target", "motion", "point"] + TIM,
              [[tg, ML[m], p] + [f"{X[(t,p,p,ARM[m][0],g)]:.1f}" for g in ARM[m]]
               for t, tg in (("shape", "shape"), ("color", "colour"))
               for m in MOT for p in ("z", "h", "p")]), "",
          "### 8. probing 표현 이식 — GT 미래(h) ↔ 예측 미래(p), 같은 조건 안에서", "",
          tbl(["target", "motion", "direction"] + TIM,
              [[tg, ML[m], f"{aa} → {bb}"] + [f"{X[(t,aa,bb,g,g)]:.1f}" for g in ARM[m]]
               for t, tg in (("shape", "shape"), ("color", "colour"))
               for m in MOT for aa, bb in (("h", "h"), ("h", "p"), ("p", "p"), ("p", "h"))]), ""]

    # by_k
    def bk(tgt, kind, m, ec):
        gs = ARM[m]
        if kind == "self":
            pr = HP[(tgt, "p", "p", ec)]
        elif kind == "vis":
            pr = HP[(tgt, "p", "p", gs[0])]
        else:
            pr = HP[(tgt, "h", "p", ec)]
        return [acc(pr, gold[tgt], (cond == ec) & (kk == k)) for k in KS]

    L += ["### 9. probing × k (`predictions.json` 사후 분해, 셀당 172~215 clip)", ""]
    for kind, name in (("self", "self (자기 조건 head)"), ("vis", "이식 (비가림 head)"),
                       ("hp", "표현 이식 h → p")):
        L += [f"**{name}**  — `p` 만", "",
              tbl(["target", "motion", "timing"] + [f"k={k}" for k in KS],
                  [[tg, ML[m], t] + [f"{x:.1f}" for x in bk(tt, kind, m, ec)]
                   for tt, tg in (("shape", "shape"), ("color", "colour"))
                   for m in MOT for t, ec in zip(("early", "mid", "late"), ARM[m][1:])]), ""]

    # ── 기전 ────────────────────────────────────────────────────────────────
    rows = []
    for v in VIOL:
        for t in TIM:
            arr = np.array(mg[(t, v)]); rel = (arr[:, 1] - arr[:, 0]) / arr[:, 0] * 100
            rows.append([VL[v], t, f"{arr[:,0].mean():.4f}", f"{rel.mean():.3f}",
                         f"{abs(rel.mean())/rel.std():.3f}", f"{100*(rel>0).mean():.1f}",
                         f"{len(arr)}"])
    L += ["### 10. 기전 — margin 분해", "",
          "채점은 `|p − h(가능)|` vs `|p − h(불가능)|` 의 대소다. margin 은 그 차이.",
          "정확도는 `margin > 0` 인 비율이므로 **효과크기가 곧 정확도**다.", "",
          tbl(["violation", "timing", "base |p−h|", "margin (% of base)", "효과크기 |mean|/SD",
               "acc (= margin>0 비율)", "n"], rows), ""]

    md = "\n".join(L)
    if a.write:
        s = README.read_text()
        i = s.find(MARK)
        s = (s[:i] if i >= 0 else s.rstrip() + "\n\n") + md
        README.write_text(s)
        print(f"  [written] {README}  ({md.count(chr(10))} 줄)")
    else:
        print(md)


if __name__ == "__main__":
    main()
