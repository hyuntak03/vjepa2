#!/usr/bin/env python3
"""shard 결과를 합치고, 채점을 축(k · timing · violation · motion)별로 다시 집계한다.

  PYTHONPATH=. python -m analysis.attention.knockout.merge -o <knockout outdir>

입력   <outdir>/shards/shardXXofNN.{json,npz}   (run.py --shard 가 남긴 것)
출력   <outdir>/results.json                     run.py 단일 실행과 같은 형태 + `breakdown`
       <outdir>/per_video.npz                    클립별 원값 (행 = clean, clean_null, 명세…)

⚠️ 채점은 `evals.world_model_analysis.eval.score_blocks` 를 그대로 다시 부른다.
   여기서 pairing 을 다시 구현하지 않는다. 축별 값은 그 per_block 결과를 block 의 축 값으로
   묶어 평균한 것이라 `by_block_type` 과 같은 정의(block 평균)다.
⚠️ 축 값은 인덱스 행에서 파생한다 — `k` 는 file_name 의 `_k{n}_`, `timing` 은 condition 접미사
   (`visible` / `_early` / `_mid` / 그 밖의 occlusion = `late`).
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from analysis.attention.knockout.run import delta                        # noqa: E402
from evals.world_model_analysis.data import WMADataset                   # noqa: E402
from evals.world_model_analysis.eval import score_blocks                 # noqa: E402

AXES = [("k",), ("timing",), ("violation",), ("motion",), ("condition",),
        ("timing", "motion"), ("timing", "violation"), ("k", "timing"),
        ("k", "motion"), ("k", "violation"),
        ("k", "timing", "motion"), ("k", "timing", "violation")]
_K = re.compile(r"_k(\d)_")


def axis_values(rec) -> dict:
    raw = rec.raw
    cond = raw.get("condition", "")
    m = _K.search(raw.get("file_name", ""))
    if "visible" in cond:
        timing = "visible"
    elif cond.endswith("_early"):
        timing = "early"
    elif cond.endswith("_mid"):
        timing = "mid"
    else:
        timing = "late"
    return dict(condition=cond, k=m.group(1) if m else "?", timing=timing,
                violation=raw.get("violation_type", ""), motion=raw.get("motion", ""))


def _agg(accs, npairs):
    return {"n_block": int(len(accs)), "n_pair": int(npairs.sum()),
            "acc": round(100 * float(accs.mean()), 3) if len(accs) else float("nan")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--outdir", required=True)
    a = ap.parse_args()
    out = Path(a.outdir)
    js = sorted((out / "shards").glob("shard*.json"))
    if not js:
        sys.exit(f"{out}/shards 에 shard*.json 이 없다")
    shards = [json.loads(p.read_text()) for p in js]
    npzs = [np.load(p.with_suffix(".npz"), allow_pickle=False) for p in js]

    # ---- 일관성 검사 ------------------------------------------------------
    n_tot = shards[0]["meta"]["shard"][1]
    got = sorted(s["meta"]["shard"][0] for s in shards)
    if got != list(range(n_tot)):
        sys.exit(f"shard 가 빠졌다: 있음 {got} / 기대 0..{n_tot-1}")
    names = [str(x) for x in npzs[0]["names"]]
    spec_sig = json.dumps([(r["name"], r["layers"]) for r in shards[0]["specs"]])
    cfg_sig = json.dumps(shards[0]["config"], sort_keys=True)
    for s, z, p in zip(shards, npzs, js):
        if [str(x) for x in z["names"]] != names:
            sys.exit(f"{p.name}: 명세 순서가 다르다")
        if json.dumps([(r["name"], r["layers"]) for r in s["specs"]]) != spec_sig:
            sys.exit(f"{p.name}: 명세가 다르다")
        if json.dumps(s["config"], sort_keys=True) != cfg_sig:
            sys.exit(f"{p.name}: config 가 다르다")
        v = s["verify"]["resume_vs_forward_max_abs_diff"]
        if v != 0.0:
            sys.exit(f"{p.name}: resume_vs_forward_max_abs_diff = {v} (0 이어야 한다)")
    fields = [k for k in npzs[0].files if k not in ("video_ids", "names")]

    # ---- 클립별 원값 이어붙이기 (ds.records 순서로) --------------------------
    cfg = shards[0]["config"]
    ds = WMADataset(cfg, limit=cfg.get("limit"))
    col: dict[str, tuple[int, int]] = {}
    for si, z in enumerate(npzs):
        for j, v in enumerate(z["video_ids"]):
            v = str(v)
            if v in col:
                sys.exit(f"video {v} 가 두 shard 에 있다")
            col[v] = (si, j)
    known = {r.video_id for r in ds.records}
    unknown = set(col) - known
    if unknown:
        sys.exit(f"인덱스에 없는 video {len(unknown)}개 (예 {sorted(unknown)[:3]})")
    ds.records = [r for r in ds.records if r.video_id in col]
    bad = [b for b, recs in ds.blocks().items() if len(recs) != 4]
    if bad:
        sys.exit(f"4개가 안 채워진 block {len(bad)}개 (예 {bad[:3]}) — shard 가 block 을 갈랐다")
    vids = [r.video_id for r in ds.records]
    mat = {f: np.stack([np.concatenate([npzs[si][f][i] for si in range(len(npzs))])
                        for i in range(len(names))]) for f in fields}
    # 위 concat 은 shard 순서라 vid 순서로 다시 놓는다
    order = np.array([sum(len(z["video_ids"]) for z in npzs[:si]) + j for si, j in
                      (col[v] for v in vids)])
    mat = {f: m[:, order] for f, m in mat.items()}

    # ---- 채점 (score_blocks 재사용) ---------------------------------------
    mnames = shards[0]["meta"]["metrics"]
    bt = {r.video_id: r.block_type for r in ds.records}
    blocks = ds.blocks()
    bids = sorted(blocks)
    bax = {b: axis_values(blocks[b][0]) for b in bids}
    acc_by: dict[str, np.ndarray] = {}
    npair_by: dict[str, np.ndarray] = {}
    results: dict[str, dict] = {}

    def by_type(vals):
        d = collections.defaultdict(list)
        for v, x in zip(vids, vals):
            d[bt[v]].append(float(x))
        return {k: round(float(np.mean(x)), 4) for k, x in sorted(d.items())}

    for i, name in enumerate(names):
        got = {}
        if "surprise" in mat:
            sup = {v: float(x) for v, x in zip(vids, mat["surprise"][i])}
            res, per_block = score_blocks(ds, sup, cfg)
            pb = {r["block_id"]: r for r in per_block}
            acc_by[name] = np.array([pb[b]["acc"] for b in bids])
            npair_by[name] = np.array([pb[b]["n_pairs"] for b in bids])
            o = res["overall"]
            if "surprise_acc" in mnames:
                got["surprise_acc"] = {
                    "acc": round(100 * o["block_pairwise"], 3), "n_block": o["n_block"],
                    "n_pair": o["n_pair"], "n_ties": o["n_ties"],
                    "perfect_ratio": round(o["perfect_ratio"], 4), "pairing": res["pairing"],
                    "by_block_type": {k: round(100 * v["block_pairwise"], 3)
                                      for k, v in res.get("by_block_type", {}).items()}}
            if "surprise_l1" in mnames:
                got["surprise_l1"] = {"mean": round(float(mat["surprise"][i].mean()), 5),
                                      "by_block_type": by_type(mat["surprise"][i])}
        if "pred_drift" in mnames and "drift_rel" in mat:
            got["pred_drift"] = {"rel_l1": round(float(mat["drift_rel"][i].mean()), 5),
                                 "cosine": round(float(mat["drift_cos"][i].mean()), 5),
                                 "rel_l1_by_block_type": by_type(mat["drift_rel"][i])}
        results[name] = got

    clean_res = results["clean"]
    null_res = results.get("clean_null")
    ref, ref_name = (null_res, "clean_null") if null_res else (clean_res, "clean")
    rows = []
    for r in shards[0]["specs"]:
        got = results[r["name"]]
        rows.append(dict(name=r["name"], q=r["q"], k=r["k"], rel=r["rel"], layers=r["layers"],
                         metrics=got, delta=delta(ref, got), delta_vs_clean=delta(clean_res, got)))

    # ---- 축별 집계 --------------------------------------------------------
    breakdown = {}
    for axes in AXES:
        key = "|".join(axes)
        groups = collections.defaultdict(list)
        for j, b in enumerate(bids):
            groups["|".join(bax[b][x] for x in axes)].append(j)
        tab = {}
        for lv, idx in sorted(groups.items()):
            idx = np.array(idx)
            tab[lv] = {"n_block": int(len(idx)), "n_pair": int(npair_by["clean"][idx].sum()),
                       "acc": {n: round(100 * float(acc_by[n][idx].mean()), 3) for n in names}}
        breakdown[key] = tab

    m0 = copy.deepcopy(shards[0]["meta"])
    m0.update(n_block=len(bids), n_clip=len(vids),
              blocks_by_cond=dict(sorted(collections.Counter(bt[v] for v in vids).items())),
              elapsed_s=round(sum(s["meta"]["elapsed_s"] for s in shards), 1),
              elapsed_wall_s=round(max(s["meta"]["elapsed_s"] for s in shards), 1),
              shard=None, n_shards=n_tot, merged_from=[p.name for p in js])
    # blocks_by_cond 는 clip 수를 4 로 나눈 block 수여야 한다
    m0["blocks_by_cond"] = {k: v // 4 for k, v in m0["blocks_by_cond"].items()}
    verify = dict(shards[0]["verify"])
    verify["per_shard"] = {p.name: s["verify"] for p, s in zip(js, shards)}
    verify["resume_vs_forward_max_abs_diff"] = max(
        s["verify"]["resume_vs_forward_max_abs_diff"] for s in shards)
    verify["null_mask_vs_forward_max_abs_diff"] = max(
        s["verify"]["null_mask_vs_forward_max_abs_diff"] for s in shards)
    outj = dict(meta=m0, verify=verify, clean=clean_res, clean_null=null_res, specs=rows,
                breakdown=breakdown, breakdown_axes=["|".join(x) for x in AXES], config=cfg)
    (out / "results.json").write_text(json.dumps(outj, ensure_ascii=False, indent=1))
    np.savez_compressed(out / "per_video.npz", video_ids=np.array(vids), names=np.array(names),
                        **mat)

    print(f"  shard {n_tot}개 병합 → block {len(bids)} / clip {len(vids)}  명세 {len(rows)}개")
    print(f"  검증  resume max|Δ| {verify['resume_vs_forward_max_abs_diff']:.3e} | "
          f"null-mask {verify['null_mask_vs_forward_max_abs_diff']:.3e}")
    if "surprise_acc" in mnames:
        base = ref["surprise_acc"]["acc"]
        print(f"\n  기준선 {ref_name} {base:.2f}%  (clean {clean_res['surprise_acc']['acc']:.2f}%, "
              f"n_pair {clean_res['surprise_acc']['n_pair']})  —  Δ 가 큰 순")
        for r in sorted(rows, key=lambda r: r["delta"]["surprise_acc"]["acc"]):
            print(f"  {r['name']:<24}{r['metrics']['surprise_acc']['acc']:>8.2f}"
                  f"{r['delta']['surprise_acc']['acc']:>+9.2f}")
    print(f"\n  [saved] {out}/results.json  +  per_video.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
