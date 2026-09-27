#!/usr/bin/env python3
"""RollOut v3 인덱스 — **64 프레임 stride 1**, 사건은 f32 에 고정.

rollout2 빌더와 프레임 구조가 달라 따로 둔다 (v2: 100 프레임 stride 3 → 32 샘플 / v3: 64 프레임 전부).
v3 는 **가림막이 아예 없다** (`has_occlusion=0`, `occluder=None`, 3,136 전부) → `visible = in_frame`,
제외(ignore) 칸은 없다. metadata.csv 가 위치·in_frame 을 64 샘플로 이미 들고 있어 거의 전사다.

창 격자: 문맥 `[32-C, 32)`, 예측 `[32, 32+P)`, `C·P ∈ {4,8,16,32}` → 16 조합이 64 프레임 안에 다 들어간다.
`wall`·`ledge` 만 불가능 짝이 있고 **f32 까지 픽셀 동일, f33 부터 갈라진다** → matched pairing 은 Δ=0 에서만 성립.

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/data/build_rollout3_index.py            # 검사만
  $P z_research/scripts/data/build_rollout3_index.py --write
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
SRC = Path("/data2/local_datasets/world/world_analysis/RollOut_v3")
OUT = ROOT / "data_csv/rollout_v3"
NF, SPLIT, RES_PX = 64, 32, 288

COLS = ["video_id", "file", "file_name", "block_id", "variant", "plausible", "pair_id",
        "condition", "scenario", "role", "violation_type", "primary", "secondary", "secondary_name",
        "holdout", "env", "surface", "shape_pre", "color_pre", "semantic_event_frame",
        "sample_frames", "px_x_by_sample", "px_y_by_sample",
        "in_frame_by_sample", "visible_by_sample", "ignore_by_sample",
        "x_cm_by_sample", "z_cm_by_sample", "frame_half_width_cm", "fps", "resolution"]


def die(m):
    print(f"[FAIL] {m}", file=sys.stderr); sys.exit(1)


def build(rows):
    out = []
    for r in rows:
        name, sc = r["name"], r["scenario"]
        inf = r["in_frame_by_sample"].split()
        out.append({
            "video_id": name,
            "file": "",
            # 프레임은 Images/<scenario>/roll/<name>/000000.png — pos·imp 둘 다 roll/ 아래에 있다
            "file_name": f"{sc}/roll/{name}",
            "block_id": r["block"],
            "variant": "pos_a" if r["role"] == "roll" else "imp_ab",
            "plausible": r["is_possible"],
            "pair_id": r["block"],                        # wall·ledge 는 block 으로 짝이 맺힌다
            "condition": sc,
            "scenario": sc,
            "role": r["role"],
            "violation_type": r["violation_type"],
            "primary": r["primary"], "secondary": r["secondary"], "secondary_name": r["secondary_name"],
            "holdout": r["holdout"], "env": r["env"], "surface": r["surface"],
            "shape_pre": r["shape_pre"], "color_pre": r["color_pre"],
            "semantic_event_frame": r["semantic_event_frame"],
            "sample_frames": r["sample_frames"],
            "px_x_by_sample": r["object_px_x_by_sample"],
            "px_y_by_sample": r["object_px_y_by_sample"],
            "in_frame_by_sample": r["in_frame_by_sample"],
            "visible_by_sample": " ".join(inf),           # 가림막이 없으므로 in_frame 과 같다
            "ignore_by_sample": " ".join("0" * len(inf)), # 제외 칸 없음
            "x_cm_by_sample": r["object_x_cm_by_sample"],
            "z_cm_by_sample": r["object_z_cm_by_sample"],
            "frame_half_width_cm": r["frame_half_width_cm"],
            "fps": r["fps"], "resolution": r["resolution"],
        })
    return out


def verify(out, n_frame_check):
    a = lambda s: np.array([float(v) for v in str(s).split()], np.float32)
    n = len(out)
    print(f"[1] clip {n}", "OK" if n == 3136 else die(f"3136 이어야 하는데 {n}"))

    bad = [o["video_id"] for o in out if len(o["sample_frames"].split()) != NF]
    print(f"[2] 샘플 {NF} 개", "OK" if not bad else die(f"{len(bad)} clip 이 다름"))

    print(f"[3] 해상도 {RES_PX}", "OK" if all(o["resolution"] == str(RES_PX) for o in out) else die("해상도 불일치"))

    ev = {o["semantic_event_frame"] for o in out}
    print(f"[4] 사건 프레임 = {ev}", "OK" if ev == {str(SPLIT)} else die(f"f{SPLIT} 하나여야 한다"))

    # 창 격자가 64 프레임 안에 들어가는가
    ok = all(SPLIT - c >= 0 and SPLIT + p <= NF for c in (4, 8, 16, 32) for p in (4, 8, 16, 32))
    print("[5] 16 창이 64 프레임 안", "OK" if ok else die("창이 넘친다"))

    # wall·ledge 의 pos/imp 가 f32 까지 같고 f33 부터 갈라지는가
    for sc in ("wall", "ledge"):
        P = {o["block_id"]: o for o in out if o["scenario"] == sc and o["role"] == "roll"}
        I = {o["block_id"]: o for o in out if o["scenario"] == sc and o["role"] == "impossible_roll"}
        if not I:
            die(f"{sc} 에 불가능 짝이 없다")
        miss = set(P) ^ set(I)
        if miss:
            die(f"{sc} 짝이 안 맞는 block {len(miss)} 개")
        d_pre, d_post = [], []
        for b in P:
            px, qx = a(P[b]["px_x_by_sample"]), a(I[b]["px_x_by_sample"])
            py, qy = a(P[b]["px_y_by_sample"]), a(I[b]["px_y_by_sample"])
            d = np.hypot(px - qx, py - qy)
            d_pre.append(d[:SPLIT + 1].max()); d_post.append(d[SPLIT + 1:].max())
        print(f"[6:{sc}] 짝 {len(P)} · f0~f{SPLIT} 최대차 {max(d_pre):.3f} px · f{SPLIT+1}~ 최대차 {max(d_post):.1f} px",
              "OK" if max(d_pre) < 1e-3 else die("문맥이 픽셀 단위로 같지 않다"))

    # 프레임 실물
    rng = np.random.RandomState(0)
    for o in [out[i] for i in rng.choice(n, min(n_frame_check, n), replace=False)]:
        d = SRC / "Images" / o["file_name"]
        if not (d / f"{0:06d}.png").exists() or not (d / f"{NF-1:06d}.png").exists():
            die(f"프레임 없음 {d}")
    print(f"[7] 프레임 실물 {min(n_frame_check, n)} clip 표본", "OK")

    # 예측 구간에서 물체가 화면 안인 비율 (창별로 몇 칸이 쓸 수 있는지)
    IF = np.stack([(a(o["in_frame_by_sample"]) > 0) for o in out])
    print("[8] 물체가 창 내내 화면 안인 clip 비율")
    print("      " + "".join(f"{f'C={c}':>9s}" for c in (4, 8, 16, 32)) +
          "   " + "".join(f"{f'P={p}':>9s}" for p in (4, 8, 16, 32)))
    print("      " + "".join(f"{100*IF[:, SPLIT-c:SPLIT].all(1).mean():8.0f}%" for c in (4, 8, 16, 32)) +
          "   " + "".join(f"{100*IF[:, SPLIT:SPLIT+p].all(1).mean():8.0f}%" for p in (4, 8, 16, 32)))
    print("      ⚠️ C=32 는 물체가 도중에 화면 밖에서 들어오는 clip 이 많다 — `in_frame_all_context` 로 걸러 읽는다")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--frame-check", type=int, default=120)
    a = ap.parse_args()

    rows = list(csv.DictReader((SRC / "metadata.csv").open()))
    out = build(rows)
    verify(out, a.frame_check)

    if a.write:
        OUT.mkdir(parents=True, exist_ok=True)
        for fn in ("index.csv", "index_probe.csv"):        # 하네스가 둘 다 찾는다 (v3 는 내용이 같다)
            with (OUT / fn).open("w", newline="") as f:
                w = csv.DictWriter(f, COLS); w.writeheader(); w.writerows(out)
        print(f"\n→ {OUT}/index.csv · index_probe.csv  ({len(out)} clip, {len(COLS)} 컬럼)")
    else:
        print("\n(--write 없음 — 검사만 했다)")


if __name__ == "__main__":
    main()
