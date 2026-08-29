#!/usr/bin/env python3
"""여러 산출물을 하나의 구조화된 JSON 으로 모은다 — 표·그림이 같은 출처를 읽게.

지금은 채점·probing·confusion 이 서로 다른 파일에 흩어져 있고, 분석 스크립트마다
같은 것을 따로 다시 계산한다. 이 스크립트가 한 번에 모아 **검증까지 하고** 낸다.

축은 인덱스에 **있는 것만** 잡는다 (`sym_k` 는 v10 index_probe.csv 에만 있고
v10_flat / occ_low 에는 없다). v11 처럼 k 가 내부 축이면 자동으로 잡힌다.

  python z_research/scripts/analysis/report.py \
      --run z_research/IntPhysGenV10/exp_results/surprise_c16t32__v10_vith:data_csv/intphysgen_v10/index_probe.csv \
      --run z_research/IntPhysGenV10/exp_results/surprise_c16t32__v10_flat_vith:data_csv/intphysgen_v10_flat/index_probe.csv \
      --probe z_research/IntPhysGenV10/exp_results/attn_probe__v10_vith \
      -o z_research/IntPhysGenV10/exp_results/report.json

구조
  meta        무엇을 읽었나, 검증 결과
  scoring
    overall     실행별 전체
    cells[]     (dataset, condition, violation, k) 한 칸씩
                acc / n / sensitivity / bias / pairs[]  <- pairs 가 방향별
    dose[]      k 별 집계 (violation 별과 전체)
  probing
    cells[]     (point, target, groups) x condition
    by_axis[]   --probe-index 를 주면 생긴다. head 마다 임의 축(k / surface / …)과
                condition x sym_k 교차까지 쪼갠 정확도
  confusion
    <fit>/<target>/<condition> : {classes, matrix}   행=정답 열=예측
"""
from __future__ import annotations
import argparse, collections, csv, json, sys
from pathlib import Path
import numpy as np

AXES = ["condition", "violation_type", "sym_k"]      # 있는 것만 쓴다
POINT = {"contextF__f1to16": "z", "targetF__f17to32": "h", "pred__f17to32": "p"}


def direction(pos, imp):
    v = imp["violation_type"]
    if v == "vanish":
        return ("object", "empty") if imp.get("role") == "imp_vanish" else ("empty", "object")
    if v == "shape":
        return (pos.get("shape_pre"), imp.get("shape_post"))
    if v == "color":
        return (pos.get("color_pre"), imp.get("color_post"))
    return (None, None)


def score_run(result_dir: Path, index: Path, name: str):
    S = json.loads((result_dir / "per_block.json").read_text())["per_video_surprise"]
    rows = list(csv.DictReader(index.open()))
    axes = [a for a in AXES if a in rows[0]]
    grp = collections.defaultdict(list)
    for r in rows:
        grp[(r["block_id"], r["pair_id"])].append(r)
    recs, allok = [], []
    for v in grp.values():
        pos = [x for x in v if x["plausible"] == "1"]
        imp = [x for x in v if x["plausible"] != "1"]
        if len(pos) != 1 or len(imp) != 1:
            continue
        p, i = pos[0], imp[0]
        sp, si = S[p["video_id"]], S[i["video_id"]]
        if isinstance(sp, dict):                       # sliding 결과
            k = sorted(sp)[0]; sp, si = sp[k], si[k]
        ok = 1.0 if si > sp else (0.5 if si == sp else 0.0)
        a, b = direction(p, i)
        recs.append(dict(key=tuple(p.get(x) for x in axes), a=a, b=b, ok=ok))
        allok.append(ok)

    got = float(np.mean(allok))
    want = json.loads((result_dir / "summary.json").read_text())["surprise"]
    want = want.get("overall", {}).get("block_pairwise")
    ver = None if want is None else abs(got - want) < 1e-9
    if ver is False:
        sys.exit(f"검증 실패 {name}: 재계산 {got:.6f} != summary {want:.6f}")
    return axes, recs, dict(dataset=name, n_pair=len(recs), overall=100 * got,
                            verified=ver, source=str(result_dir))


def cell_stats(rs):
    """한 칸의 방향 분해. 무순서 쌍마다 정/역이 다 있어야 sensitivity 가 정의된다."""
    d = collections.defaultdict(list)
    for r in rs:
        d[(r["a"], r["b"])].append(r["ok"])
    seen, pairs = set(), []
    for (a, b) in sorted(d, key=lambda t: (str(t[0]), str(t[1]))):
        if frozenset((a, b)) in seen or (b, a) not in d:
            continue
        seen.add(frozenset((a, b)))
        f, r_ = 100 * np.mean(d[(a, b)]), 100 * np.mean(d[(b, a)])
        pairs.append(dict(a=a, b=b, fwd=round(f, 2), rev=round(r_, 2),
                          n_fwd=len(d[(a, b)]), n_rev=len(d[(b, a)]),
                          sensitivity=round((f + r_) / 2 - 50, 2), bias=round(abs(f - r_) / 2, 2)))
    out = dict(n=len(rs), acc=round(100 * float(np.mean([r["ok"] for r in rs])), 2),
               n_pair_types=len(pairs), pairs=pairs)
    if pairs:
        out["sensitivity"] = round(float(np.mean([p["sensitivity"] for p in pairs])), 2)
        out["bias"] = round(float(np.mean([p["bias"] for p in pairs])), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[], metavar="RESULT_DIR:INDEX",
                    help="채점 결과와 인덱스 (여러 번 가능)")
    ap.add_argument("--probe", action="append", default=[], metavar="RESULT_DIR",
                    help="attn_probe 결과 디렉토리 (여러 번 가능)")
    ap.add_argument("--confusion", nargs="*", default=["pred__f17to32/shape", "pred__f17to32/color"],
                    metavar="FIT/TARGET")
    ap.add_argument("--probe-index", type=Path, default=None,
                    help="predictions.json 의 val_video_ids 와 조인할 index_probe.csv. "
                         "이걸 주면 **임의의 축(k, surface, direction …)으로 head 를 쪼갠다.** "
                         "probing 의 group_column 은 condition 이라 config 만으로는 k 를 못 본다.")
    ap.add_argument("--probe-axes", nargs="*", default=["sym_k", "condition", "surface"],
                    metavar="COL", help="쪼갤 축. 인덱스에 있는 것만 쓴다")
    ap.add_argument("-o", "--output", type=Path, required=True)
    a = ap.parse_args()

    rep = {"meta": {"runs": [], "probes": []}, "scoring": {"overall": [], "cells": [], "dose": []},
           "probing": {"cells": []}, "confusion": {}}

    # ---- 채점 ---------------------------------------------------------------
    for spec in a.run:
        rd, ix = spec.split(":", 1)
        rd, ix = Path(rd), Path(ix)
        name = rd.name.split("__")[-1]
        axes, recs, meta = score_run(rd, ix, name)
        rep["meta"]["runs"].append(dict(meta, index=str(ix), axes=axes))
        rep["scoring"]["overall"].append({k: meta[k] for k in ("dataset", "overall", "n_pair", "verified")})
        by = collections.defaultdict(list)
        for r in recs:
            by[r["key"]].append(r)
        for key, rs in sorted(by.items(), key=lambda t: tuple(map(str, t[0]))):
            rep["scoring"]["cells"].append(dict(dataset=name, **dict(zip(axes, key)), **cell_stats(rs)))
        # k 별 집계 (있을 때만)
        if "sym_k" in axes:
            ki, vi = axes.index("sym_k"), axes.index("violation_type") if "violation_type" in axes else None
            agg = collections.defaultdict(list)
            for r in recs:
                agg[(r["key"][ki], r["key"][vi] if vi is not None else "all")].append(r)
                agg[(r["key"][ki], "all")].append(r)
            for (k, v), rs in sorted(agg.items(), key=lambda t: (str(t[0][0]), str(t[0][1]))):
                rep["scoring"]["dose"].append(dict(dataset=name, sym_k=k, violation=v, **cell_stats(rs)))

    # ---- probing ------------------------------------------------------------
    for pd_ in a.probe:
        pd_ = Path(pd_)
        s = json.loads((pd_ / "summary.json").read_text())
        rep["meta"]["probes"].append(dict(source=str(pd_), n_head=len(s.get("probing", []))))
        for r in s.get("probing", []):
            fit = r["fit"]
            ev = r["evals"][fit]
            rep["probing"]["cells"].append(dict(
                point=POINT.get(fit, fit), fit=fit, target=r["target"],
                groups=r.get("groups"), n_train=r["n_train"],
                train_acc=round(r["train_acc"], 4), chance=round(100 * r["chance"], 2),
                overall=round(100 * ev["overall"], 2),
                per_group={g: round(100 * c["acc"], 2) for g, c in ev["per_group"].items()},
                converged=r["train_acc"] >= 0.95))
        # ---- 임의 축으로 head 쪼개기 (k 비교가 여기서 나온다) ------------------
        pf = pd_ / "predictions.json"
        if pf.exists() and a.probe_index:
            M = json.loads(pf.read_text())
            meta = {r["video_id"]: r for r in csv.DictReader(a.probe_index.open())}
            vid = M["val_video_ids"]
            axes = [c for c in a.probe_axes if c in next(iter(meta.values()))]
            col = {c: np.array([meta[v].get(c) for v in vid]) for c in axes}
            for h in M["heads"]:
                if h["eval"] != h["fit"]:            # self 만 (이식은 의미가 다르다)
                    continue
                gold = np.asarray(M["targets"][h["target"]]["gold"])
                pred = np.asarray(h["pred"])
                rec = dict(point=POINT.get(h["fit"], h["fit"]), fit=h["fit"],
                           target=h["target"], groups=h.get("groups"), by={})
                for c in axes:
                    rec["by"][c] = {}
                    for v in sorted(set(col[c])):
                        m = col[c] == v
                        if m.sum():
                            rec["by"][c][str(v)] = dict(
                                n=int(m.sum()),
                                acc=round(100 * float((pred[m] == gold[m]).mean()), 2))
                    # 두 축 교차 (조건 x k) — dose-response 를 조건별로 보려면 필요
                if "sym_k" in axes and "condition" in axes:
                    rec["by"]["condition_x_sym_k"] = {}
                    for cd in sorted(set(col["condition"])):
                        for k in sorted(set(col["sym_k"])):
                            m = (col["condition"] == cd) & (col["sym_k"] == k)
                            if m.sum():
                                rec["by"]["condition_x_sym_k"][f"{cd}|k{k}"] = dict(
                                    n=int(m.sum()),
                                    acc=round(100 * float((pred[m] == gold[m]).mean()), 2))
                rep["probing"].setdefault("by_axis", []).append(rec)

        # ---- confusion ------------------------------------------------------
        if not pf.exists():
            continue
        M = json.loads(pf.read_text())
        grp = np.asarray(M["val_groups"])
        for want in a.confusion:
            fit, tgt = want.split("/")
            if tgt not in M["targets"]:
                continue
            head = next((h for h in M["heads"] if h["fit"] == fit and h["target"] == tgt
                         and h["eval"] == fit and not h.get("groups")), None)
            if head is None:
                continue
            cls = M["targets"][tgt]["classes"]
            gold, pred = np.asarray(M["targets"][tgt]["gold"]), np.asarray(head["pred"])
            for cond in ["ALL"] + sorted(set(grp)):
                m = np.ones(len(gold), bool) if cond == "ALL" else (grp == cond)
                C = np.zeros((len(cls), len(cls)), int)
                for g, p in zip(gold[m], pred[m]):
                    C[g, p] += 1
                rep["confusion"][f"{fit}/{tgt}/{cond}"] = dict(
                    classes=cls, n=int(m.sum()),
                    acc=round(100 * float(np.trace(C)) / max(1, C.sum()), 2),
                    matrix=C.tolist())

    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    n = rep["scoring"]
    print(f"[saved] {a.output}")
    print(f"  scoring  run {len(n['overall'])} · cell {len(n['cells'])} · dose {len(n['dose'])}")
    print(f"  probing  head {len(rep['probing']['cells'])}"
          + (f" · by_axis {len(rep['probing'].get('by_axis', []))}" if rep["probing"].get("by_axis") else ""))
    print(f"  confusion {len(rep['confusion'])}")
    for o in n["overall"]:
        print(f"    {o['dataset']:<16}{o['overall']:6.2f}%  n={o['n_pair']}  검증 {'OK' if o['verified'] else '?'}")


if __name__ == "__main__":
    main()
