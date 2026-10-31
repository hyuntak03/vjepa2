#!/usr/bin/env python3
"""CLEVRER 영속성 v1 (testbed/cleverer_permanence_v1) → 표준 index.csv + 문맥 무결성 감사 (2026-10-31).

원본: 장면 396 × 네 영상 (present 가능 · vanish 불가능 · absent 가능 · appear 불가능), 영상마다 32 장 (문맥 16 + 미래 16, 원본 보폭 2, 256²).
쌍: A = present ↔ vanish (문맥: 물체 있음) · B = absent ↔ appear (문맥: 물체 없음). variant pos_a · imp_ab · pos_b · imp_ba.
condition = 운동 (static · behind · overtake · cross), 나머지 요인 열은 그대로 옮긴다 (occluder_kind · clip_anchor · 모양 · 색).
감사: 쌍마다 문맥 16 장이 화소까지 같은지 (decord) → context_integrity.json.
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; $P z_research/scripts/data/build_clevrer_perm_index.py
"""
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
SRC = "/data/hyuntak/project/2026/2027_cvpr/testbed/cleverer_permanence_v1"
OUT = f"{REPO}/data_csv/clevrer_perm_v1"
VAR = {"present": ("pos_a", "1", "A"), "vanish": ("imp_ab", "0", "A"), "absent": ("pos_b", "1", "B"), "appear": ("imp_ba", "0", "B")}


def audit(sc):
    import decord
    f = {c: decord.VideoReader(f"{SRC}/{sc[c]['video']}").get_batch(list(range(16))).asnumpy() for c in VAR}
    return sc["present"]["scene_id"], {"A": int(np.abs(f["present"].astype(int) - f["vanish"].astype(int)).max()),
                                       "B": int(np.abs(f["absent"].astype(int) - f["appear"].astype(int)).max())}


def main():
    rows = list(csv.DictReader(open(f"{SRC}/videos.csv")))
    by = {}
    for r in rows:
        by.setdefault(r["scene_id"], {})[r["condition"]] = r
    out = []
    for bi, (sid, sc) in enumerate(sorted(by.items())):
        assert set(sc) == set(VAR), sid
        for c, (var, plaus, pair) in VAR.items():
            r = sc[c]
            out.append(dict(video_id=f"{sid}_{c}", file=r["video"], file_name=os.path.dirname(r["video"]), block_id=str(bi),
                            source_block=sid, variant=var, plausible=plaus, pair_id=pair, condition=r["motion"], motion=r["motion"],
                            has_occlusion="1", violation_type="permanence", game_name="clevrer", role=var, sym_k="0",
                            objset="clevrer", violation="permanence", occlusion=r["clip_anchor"], occluder_kind=r["occluder_kind"],
                            target_shape=r["target_shape"], target_color=r["target_color"], k="0"))
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    with ThreadPoolExecutor(16) as ex:
        au = dict(ex.map(audit, [by[s] for s in sorted(by)]))
    bad = {s: v for s, v in au.items() if v["A"] or v["B"]}
    json.dump({"n_scenes": len(au), "n_pairs": 2 * len(au), "mismatch_pairs": sum((v["A"] > 0) + (v["B"] > 0) for v in au.values()),
               "mismatch": bad}, open(f"{OUT}/context_integrity.json", "w"), indent=1)
    print(f"{len(out)} 행 · 장면 {len(au)} · 문맥 불일치 쌍 {sum((v['A'] > 0) + (v['B'] > 0) for v in au.values())} / {2 * len(au)} -> {OUT}")


if __name__ == "__main__":
    main()
