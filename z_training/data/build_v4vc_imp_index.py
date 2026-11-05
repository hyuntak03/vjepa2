#!/usr/bin/env python3
"""BlenderTrack v4vc 학습 목록의 불가능 짝 정보 (2026-11-02, 사용자 지시 "impossible 도 넣고 1−L1 으로 negative loss").

학습 목록 all_mp4list.csv (11,250 편) 의 영상마다 묶음 · 단계 · hidden_splice 를 붙인다.
불가능 영상은 디스크에 쓰지 않는다 — 학습기 (Mp4ListDataset, spec.negimp) 가 두 영상을 읽어 문맥 · 미래를 이어 붙인다.
정의는 main compose_v4vc.py 시험 VoE 와 같다 (이음 = 프레임 32):
  vanish_k = A_k[0:32] + B[32:64] · appear_k = B[0:32] + A_k[32:64] · swap_k_j = A_k[0:32] + A_j[32:64]
hidden_splice (main render_timing): target 이 프레임 31 · 32 에서 완전히 안 보인다. 짝에 들어가는 단계 영상이 전부 통과해야 쓴다.

출력 <out>/families.json — {list, splice, families: [{plane, family, B, A: [4 개 목록 번호], hs: [4 개 bool]}], summary}

  /venv/vjepa/bin/python z_training/data/build_v4vc_imp_index.py \
      --root /workspace/dataset/training_data/v4_velocity_control --gen-dir <main blender_track> \
      --list data_csv/blender_track_v4vc/arms/all_mp4list.csv --out data_csv/blender_track_v4vc_negimp
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_v4vc_index import families  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    paths = [ln.strip() for ln in open(a.list) if ln.strip()]
    pos = {os.path.basename(p)[:-4]: i for i, p in enumerate(paths)}
    fams, used = [], set()
    for plane, cfgf, split in (("flat", "config.yaml", "train_fam"), ("ramp", "config_v4vcr.yaml", "rtrain_fam")):
        fl, K = families(a.gen_dir, f"{root}/{cfgf}", root, split, a.workers)
        for r in fl:
            sl = {s["level"]: s for s in r["slots"]}
            if not all(s["cid"] in pos for s in r["slots"]):
                continue                                   # 목록에 안 든 묶음
            if set(sl) != {"B", *range(K)}:
                raise ValueError(f"{plane} 묶음 {r['family']}: 단계 {sorted(map(str, sl))}")
            fams.append({"plane": plane, "family": int(r["family"]), "B": pos[sl["B"]["cid"]],
                         "A": [pos[sl[k]["cid"]] for k in range(K)], "hs": [bool(sl[k]["hidden_splice"]) for k in range(K)]})
            used.update(s["cid"] for s in r["slots"])
    if len(used) != len(paths):
        raise RuntimeError(f"묶음이 목록을 다 덮지 못한다: {len(used)} / {len(paths)}")
    hs = [h for f in fams for h in f["hs"]]
    n_v = sum(hs)
    n_s = sum(f["hs"][k] and f["hs"][j] for f in fams for k in range(4) for j in range(4) if j != k)
    elig_A = n_v                                            # 통과한 단계 영상 = vanish 가능 (swap 짝이 있으면 swap 도)
    elig_B = sum(any(f["hs"]) for f in fams)
    summ = {"families": len(fams), "flat": sum(f["plane"] == "flat" for f in fams), "ramp": sum(f["plane"] == "ramp" for f in fams),
            "videos": len(paths), "stage_videos": len(hs), "hidden_splice_pass": n_v, "hidden_splice_rate": round(n_v / len(hs), 4),
            "families_all_stages_pass": sum(all(f["hs"]) for f in fams),
            "combos": {"vanish": n_v, "appear": n_v, "swap": n_s},
            "eligible_context_videos": {"A": elig_A, "B": elig_B, "total": elig_A + elig_B, "frac": round((elig_A + elig_B) / len(paths), 4)},
            "per_level_pass": [sum(f["hs"][k] for f in fams) for k in range(4)]}
    os.makedirs(a.out, exist_ok=True)
    json.dump({"list": os.path.abspath(a.list), "splice": 32, "families": fams, "summary": summ}, open(f"{a.out}/families.json", "w"))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
