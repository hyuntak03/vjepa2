#!/usr/bin/env python
"""쪼개서 돌린 attn_probe job 들의 산출물을 하나로 합친다.

sbatch.sh 의 SPLIT/GSPLIT 은 job 마다 OUTDIR 을 따로 준다 (안 그러면 summary.json 을
서로 덮어써서 마지막 하나만 남는다). 그 하위 디렉토리들을 읽어 부모에 합친 결과를 쓴다.

    python z_research/scripts/analysis/merge_probe_runs.py <base_dir>

  <base>/<sfx>/summary.json      probing[] 을 이어 붙인다
  <base>/<sfx>/predictions.json  heads 를 합치고 targets 를 합집합한다
  -> <base>/summary.json, <base>/predictions.json

`_prep` 은 버리는 head 라 기본으로 뺀다 (--keep-prep 으로 포함).
split 은 (index, seed, block 단위, stratify) 가 같으므로 val_video_ids 가 job 간
동일해야 한다. 다르면 합치지 않고 죽는다 — 조용히 섞이면 안 된다.
"""
from __future__ import annotations
import argparse, json, os, sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--keep-prep", action="store_true")
    a = ap.parse_args()

    subs = sorted(d for d in os.listdir(a.base)
                  if os.path.isfile(os.path.join(a.base, d, "summary.json")))
    if not a.keep_prep:
        subs = [d for d in subs if d != "_prep"]
    if not subs:
        print(f"합칠 게 없다: {a.base}/*/summary.json", file=sys.stderr)
        return 1

    merged, probing, seen = None, [], set()
    pred = None
    for d in subs:
        p = os.path.join(a.base, d)
        s = json.load(open(os.path.join(p, "summary.json")))
        if merged is None:
            merged = {k: v for k, v in s.items() if k != "probing"}
        for r in s.get("probing", []):
            key = (r["fit"], json.dumps(r["groups"]), r["target"])
            if key in seen:
                print(f"  ⚠️ 중복 head {key} — {d} 것을 버린다", file=sys.stderr)
                continue
            seen.add(key)
            probing.append(dict(r, _from=d))
        print(f"  {d:28s} head {len(s.get('probing', []))}")

        pp = os.path.join(p, "predictions.json")
        if not os.path.exists(pp):
            continue
        q = json.load(open(pp))
        if pred is None:
            pred = q
            continue
        if q["val_video_ids"] != pred["val_video_ids"]:
            print(f"val_video_ids 가 {d} 에서 다르다 — split 이 갈렸다. 합치지 않는다.",
                  file=sys.stderr)
            return 2
        pred["targets"].update(q.get("targets", {}))
        pred["heads"].update(q.get("heads", {})) if isinstance(pred["heads"], dict) \
            else pred["heads"].extend(q.get("heads", []))

    merged["probing"] = probing
    merged["merged_from"] = subs
    json.dump(merged, open(os.path.join(a.base, "summary.json"), "w"), indent=2)
    out = [os.path.join(a.base, "summary.json")]
    if pred is not None:
        json.dump(pred, open(os.path.join(a.base, "predictions.json"), "w"))
        out.append(os.path.join(a.base, "predictions.json"))
    print(f"\nhead {len(probing)}개 -> " + " · ".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
