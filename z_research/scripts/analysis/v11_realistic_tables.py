#!/usr/bin/env python3
"""v11_realistic · v11_realistic_ledge 채점표 (2026-10-06). `surprise_c16t32` 실행의 per_block.json + index 에서 다시 계산한다.

  v11_realistic   조건 × 방향 (A = 물체→빈 `imp_vanish`, B = 빈→물체 `imp_appear`) · 조건 × k · 원본 v11 (vanish 만) 과 나란히
  ledge           조건 (속도 × 깊이) · 물체별 · surprise 차 (불가능 − 가능)

검증: 재계산 overall 이 summary.json 의 block_pairwise 와 다르면 죽는다. CI = block 단위 부트스트랩 95 % (2,000 회).

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/v11_realistic_tables.py            # → z_research/v11_realistic/exp_results/tables.{md,json}
"""
from __future__ import annotations
import collections, csv, json, re, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "z_research/v11_realistic/exp_results"
RUNS = dict(
    realistic=(OUT / "surprise_c16t32__v11_realistic_vith", REPO / "data_csv/intphysgen_v11_realistic/index.csv"),
    ledge=(OUT / "surprise_c16t32__v11_realistic_ledge_vith", REPO / "data_csv/intphysgen_v11_realistic_ledge/index.csv"),
    v11=(REPO / "z_research/IntPhysGenV11/exp_results/surprise_c16t32__v11_vith", REPO / "data_csv/intphysgen_v11/index.csv"),
)
META = dict(realistic="/local_datasets/world/world_analysis/IntPhysGen_v11_realistic/metadata.csv",
            ledge="/local_datasets/world/world_analysis/IntPhysGen_v11_realistic_ledge/metadata.csv")
COND = ["static_visible", "static_occlusion", "moving_visible_flat", "moving_occlusion_flat", "moving_visible", "moving_occlusion"]
RNG = np.random.default_rng(0)


def pairs(name):
    """matched pair 마다 dict(block, cond, k, dir, ok, margin, pos, imp)."""
    rdir, idx = RUNS[name]
    S = json.loads((rdir / "per_block.json").read_text())["per_video_surprise"]
    rows = list(csv.DictReader(idx.open()))
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["block_id"], r["pair_id"])].append(r)
    out = []
    for (b, _), v in g.items():
        p = [x for x in v if x["plausible"] == "1"]; i = [x for x in v if x["plausible"] != "1"]
        if len(p) != 1 or len(i) != 1:
            continue
        p, i = p[0], i[0]
        if p["video_id"] not in S:
            continue
        sp, si = S[p["video_id"]], S[i["video_id"]]
        k = int(i["sym_k"]) if "sym_k" in i else int(re.search(r"_k(\d)_", i["video_id"]).group(1))
        out.append(dict(block=b, cond=i["condition"], vio=i["violation_type"], k=k, role=i["role"],
                        dir={"imp_vanish": "A", "imp_appear": "B"}.get(i["role"], "-"),
                        ok=1.0 if si > sp else (0.5 if si == sp else 0.0), margin=si - sp, base=sp, pos=p, imp=i))
    got = float(np.mean([x["ok"] for x in out]))
    want = json.loads((rdir / "summary.json").read_text())["surprise"]["overall"]["block_pairwise"]
    if abs(got - want) > 1e-9:
        sys.exit(f"검증 실패 {name}: 재계산 {got:.6f} != summary {want:.6f}")
    return out


def stat(rs):
    """정확도 % + block 부트스트랩 CI."""
    if not rs:
        return dict(n=0, acc=float("nan"), lo=float("nan"), hi=float("nan"))
    by = collections.defaultdict(list)
    for r in rs:
        by[r["block"]].append(r["ok"])
    blocks = list(by.values())
    acc = 100 * np.mean([r["ok"] for r in rs])
    bs = []
    for _ in range(2000):
        pick = RNG.integers(0, len(blocks), len(blocks))
        v = np.concatenate([blocks[j] for j in pick]); bs.append(100 * v.mean())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(n=len(rs), acc=round(acc, 2), lo=round(lo, 1), hi=round(hi, 1))


def fmt(s, ci=True):
    if s["n"] == 0:
        return "—"
    return f"{s['acc']:.1f} [{s['lo']:.0f}, {s['hi']:.0f}]" if ci else f"{s['acc']:.1f}"


def main():
    md, js = [], {}
    R = pairs("realistic")
    V = [x for x in pairs("v11") if x["vio"] == "vanish"]
    js["realistic_overall"] = stat(R)
    js["v11_vanish_overall"] = stat(V)
    md += ["## v11_realistic — 조건 × 방향 (matched pair %, [block 부트스트랩 95 % CI])", "",
           f"overall **{fmt(stat(R))}** (n={len(R)}) · 원본 v11 vanish 만 {fmt(stat(V))} (n={len(V)})", "",
           "| condition | realistic 전체 | A 물체→빈 | B 빈→물체 | n | v11 vanish 전체 | v11 A | v11 B | v11 n |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    js["by_condition"] = {}
    for c in COND:
        r = [x for x in R if x["cond"] == c]; v = [x for x in V if x["cond"] == c]
        cell = dict(all=stat(r), A=stat([x for x in r if x["dir"] == "A"]), B=stat([x for x in r if x["dir"] == "B"]),
                    v11_all=stat(v), v11_A=stat([x for x in v if x["dir"] == "A"]), v11_B=stat([x for x in v if x["dir"] == "B"]))
        js["by_condition"][c] = cell
        md.append(f"| `{c}` | {fmt(cell['all'])} | {fmt(cell['A'], 0)} | {fmt(cell['B'], 0)} | {len(r)} | "
                  f"{fmt(cell['v11_all'])} | {fmt(cell['v11_A'], 0)} | {fmt(cell['v11_B'], 0)} | {len(v)} |")

    md += ["", "## v11_realistic — 가림 조건 × k (전체 / A / B)", "",
           "| condition | k | realistic 전체 | A | B | n | v11 vanish 전체 | v11 n |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    js["by_k"] = {}
    for c in [c for c in COND if "occlusion" in c]:
        for k in range(1, 5):
            r = [x for x in R if x["cond"] == c and x["k"] == k]; v = [x for x in V if x["cond"] == c and x["k"] == k]
            js["by_k"][f"{c}/k{k}"] = dict(all=stat(r), A=stat([x for x in r if x["dir"] == "A"]),
                                           B=stat([x for x in r if x["dir"] == "B"]), v11=stat(v))
            md.append(f"| `{c}` | {k} | {fmt(stat(r))} | {fmt(stat([x for x in r if x['dir'] == 'A']), 0)} | "
                      f"{fmt(stat([x for x in r if x['dir'] == 'B']), 0)} | {len(r)} | {fmt(stat(v))} | {len(v)} |")

    # 물체별 (realistic) — 메시가 실물이라 물체마다 다를 수 있다. 물체 = 그 쌍에 나오는 유일한 물체
    #   (A 쌍: pos_a · imp_ab 의 shape_pre / B 쌍: imp_ba 의 shape_post — pos_b 는 빈 장면)
    meta = {r["name"]: r for r in csv.DictReader(open(META["realistic"]))}

    def obj_of(x):
        names = {meta[x[w]["video_id"]][c] for w in ("pos", "imp") for c in ("shape_pre", "shape_post")} - {"none", ""}
        assert len(names) == 1, names
        return names.pop()
    md += ["", "## v11_realistic — 물체별 (가림 3 조건, 방향별) · 비가림은 전 물체 100", "",
           "| object | 비가림 | 가림 A 물체→빈 | 가림 B 빈→물체 | n (가림 A / B) |", "|---|---:|---:|---:|---:|"]
    js["by_object"] = {}
    for o in sorted({obj_of(x) for x in R}):
        own = [x for x in R if obj_of(x) == o]
        vis = [x for x in own if "visible" in x["cond"]]; occ = [x for x in own if "occlusion" in x["cond"]]
        oa, ob = [x for x in occ if x["dir"] == "A"], [x for x in occ if x["dir"] == "B"]
        js["by_object"][o] = dict(visible=stat(vis), occlusion_A=stat(oa), occlusion_B=stat(ob))
        md.append(f"| {o} | {fmt(stat(vis), 0)} | {fmt(stat(oa))} | {fmt(stat(ob), 0)} | {len(oa)} / {len(ob)} |")

    # ledge
    L = pairs("ledge")
    lm = {r["name"]: r for r in csv.DictReader(open(META["ledge"]))}
    js["ledge_overall"] = stat(L)
    md += ["", "## v11_realistic_ledge — 조건 (속도 × 깊이)", "",
           f"overall **{fmt(stat(L))}** (n={len(L)}) · surprise 차 (불가능 − 가능) 중앙값 {np.median([x['margin'] for x in L]):+.4f} · "
           f"상대 (차 / 가능) 중앙값 {100 * np.median([x['margin'] / x['base'] for x in L]):+.2f} %", "",
           "| condition | 정확도 | n | 차 중앙값 | 상대 차 중앙값 |", "|---|---:|---:|---:|---:|"]
    js["ledge_by_condition"] = {}
    for c in sorted({x["cond"] for x in L}):
        r = [x for x in L if x["cond"] == c]
        js["ledge_by_condition"][c] = dict(stat(r), margin_median=float(np.median([x["margin"] for x in r])))
        md.append(f"| `{c}` | {fmt(stat(r))} | {len(r)} | {np.median([x['margin'] for x in r]):+.4f} | "
                  f"{100 * np.median([x['margin'] / x['base'] for x in r]):+.2f} % |")
    md += ["", "| object | 정확도 | n |", "|---|---:|---:|"]
    js["ledge_by_object"] = {}
    for o in sorted({lm[x["pos"]["video_id"]]["shape_pre"] for x in L}):
        r = [x for x in L if lm[x["pos"]["video_id"]]["shape_pre"] == o]
        js["ledge_by_object"][o] = stat(r)
        md.append(f"| {o} | {fmt(stat(r), 0)} | {len(r)} |")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables.md").write_text("\n".join(md) + "\n")
    (OUT / "tables.json").write_text(json.dumps(js, indent=1, ensure_ascii=False))
    print("\n".join(md)); print(f"\n→ {OUT}/tables.md · tables.json (summary.json 과 overall 일치 검증 통과)")


if __name__ == "__main__":
    main()
