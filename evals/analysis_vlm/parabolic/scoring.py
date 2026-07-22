"""Two configurable scoring modes over per-scene, per-variant surprise scores.

Both consume `records = [{"scene_id": int, "surprise": {variant: float}, "meta": {...}}, ...]`.

argmin  : the candidate with the SMALLEST surprise should be `possible`
          (the "predictor output is closest to the correct future" ranking).
          accuracy = fraction of scenes where argmin_v surprise == possible.  chance = 1/|candidates|.

pairwise: for each (possible, impossible) pair, correct iff surprise(impossible) > surprise(possible)
          (the impossible future is more "surprising"). ties count 0.5.  chance = 0.5.
"""
from __future__ import annotations


def score_argmin(records, candidates, correct="possible") -> dict:
    candidates = list(candidates)
    n = ok = 0
    per_scene = []
    for r in records:
        s = {v: r["surprise"][v] for v in candidates if v in r["surprise"]}
        if len(s) < 2:
            continue
        pick = min(s, key=s.get)
        n += 1
        hit = int(pick == correct)
        ok += hit
        per_scene.append({"scene_id": r["scene_id"], "argmin": pick, "correct": hit})
    return {
        "mode": "argmin", "candidates": candidates, "correct": correct,
        "n": n, "accuracy": ok / n if n else float("nan"),
        "chance": 1.0 / len(candidates), "per_scene": per_scene,
    }


def score_pairwise(records, pairs) -> dict:
    out = {}
    for pos, imp in pairs:
        n = ok = tie = 0
        per_scene = []
        for r in records:
            s = r["surprise"]
            if pos not in s or imp not in s:
                continue
            n += 1
            if s[imp] > s[pos]:
                ok += 1; hit = 1
            elif s[imp] == s[pos]:
                tie += 1; hit = 0.5
            else:
                hit = 0
            per_scene.append({"scene_id": r["scene_id"], "s_pos": s[pos], "s_imp": s[imp], "hit": hit})
        out[f"{imp}_vs_{pos}"] = {
            "possible": pos, "impossible": imp, "n": n,
            "accuracy": (ok + 0.5 * tie) / n if n else float("nan"),
            "n_correct": ok, "n_tie": tie, "chance": 0.5, "per_scene": per_scene,
        }
    return out
