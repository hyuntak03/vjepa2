#!/usr/bin/env python3
"""RollOut_v1 (등속 직선 운동) 의 metadata.csv -> index.csv / index_probe.csv.

RollOut_v1 은 **위반도 쌍도 없는** 세트다 (한 변이당 클립 하나, `is_possible` 전부 1).
그래서 `surprise_c16t32` 채점은 돌지 않는다 — 이 세트의 목적은
**미래 토큰에서 물체 위치를 튜블릿별로 읽어 그 기울기가 속도인지 보는 것**이고,
그건 토큰 캐시(z/p/h) 위에서 하는 회귀다 (`analysis/` 쪽 별도 스크립트).

여기서 하는 일은 두 가지뿐이다:
  1) 하네스가 요구하는 IntPhysGen 14컬럼 + probing 컬럼을 v11 과 같은 스키마로 채운다
  2) **위치 라벨을 만드는 데 필요한 컬럼**을 그대로 실어 준다 (아래)

위치 라벨 (프로브 입력이 아니라 채점자 쪽에만 쓴다):
    x_cm(raw f) = flat_anchor_x_cm + flat_v_cm_s * (f - flat_anchor_frame) / fps
    x_norm      = x_cm / frame_half_width_cm            # 정규화 이미지 좌표 [-1, 1]
  `flat_anchor_frame` = 45 = 문맥 마지막 프레임이므로 **anchor 가 곧 마지막 관측 위치**다.
  speed x anchor 가 설계상 직교라 "마지막 프레임 복사" 가설이 데이터 수준에서 차단된다.
  obj_depth_cm / 카메라 / fov 가 전부 상수라 cm -> 정규화 좌표가 **선형**이다 (검증 6번).

  python z_research/scripts/data/build_rollout_index.py            # 검증만
  python z_research/scripts/data/build_rollout_index.py --write    # 파일까지 쓴다
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import random
import sys

ROOT = "/data/hyuntak/project/2026/2027_cvpr/vjepa2"
SRC = "/data2/local_datasets/world/world_analysis/RollOut_v1"
FRAMES_ROOT = "/local_datasets/world/world_analysis/RollOut_v1"   # SRC 로 가는 심볼릭
OUT = f"{ROOT}/data_csv/rollout_v1"

# 프로토콜이 읽는 프레임: frames_start 0, stride 3, n_frames 32
PROTOCOL_FRAMES = list(range(0, 94, 3))

# v11 index_probe.csv 와 같은 앞 14 + probing 컬럼, 뒤에 위치 라벨용 컬럼
COLS = [
    "video_id", "file", "file_name", "block_id", "source_block", "variant",
    "plausible", "pair_id", "condition", "motion", "has_occlusion",
    "violation_type", "game_name", "role",
    "shape_pre", "color_pre", "shape_post", "color_post", "env", "surface",
    "obj_apparent_px", "obj_size_cm", "obj_depth_cm", "pair_group", "probe_type",
    # ── 위치 라벨용 (이 세트의 존재 이유) ──────────────────────────────────
    "speed", "anchor",
    "flat_v_cm_s", "flat_anchor_x_cm", "flat_anchor_frame", "flat_x0_cm",
    "frame_half_width_cm", "x_context_end_cm", "x_predict_first_cm",
    "x_predict_last_cm", "predict_travel_cm", "fps",
]


def die(msg):
    print(f"\nERROR: {msg}\n", file=sys.stderr)
    sys.exit(1)


def build(rows):
    out = []
    for i, r in enumerate(rows):
        vid = os.path.basename(r["file_name"])
        out.append({
            "video_id": vid,
            "file": "",                       # PNG 직독 (frames_root + frames_pattern)
            "file_name": r["file_name"],
            "block_id": i,                    # 클립 하나 = block 하나 (쌍이 없다)
            "source_block": r["block"],
            "variant": r["clip"],             # 전부 pos_roll
            "plausible": 1,
            "pair_id": "A",
            "condition": r["condition"],      # roll_v{spd}_a{anc} — 33종
            "motion": r["motion"],
            "has_occlusion": r["has_occlusion"],
            "violation_type": r["violation_type"],
            "game_name": r["game_name"],
            "role": r["role"],
            "shape_pre": r["shape_pre"], "color_pre": r["color_pre"],
            "shape_post": r["shape_post"], "color_post": r["color_post"],
            "env": r["env"], "surface": r["surface"],
            "obj_apparent_px": r["obj_apparent_px"], "obj_size_cm": r["obj_size_cm"],
            "obj_depth_cm": r["obj_depth_cm"], "pair_group": r["pair_group"],
            "probe_type": "obj",              # 전부 가능 + 물체 있음
            "speed": r["flat_v_cm_s"], "anchor": r["flat_anchor_x_cm"],
            "flat_v_cm_s": r["flat_v_cm_s"],
            "flat_anchor_x_cm": r["flat_anchor_x_cm"],
            "flat_anchor_frame": r["flat_anchor_frame"],
            "flat_x0_cm": r["flat_x0_cm"],
            "frame_half_width_cm": r["frame_half_width_cm"],
            "x_context_end_cm": r["x_context_end_cm"],
            "x_predict_first_cm": r["x_predict_first_cm"],
            "x_predict_last_cm": r["x_predict_last_cm"],
            "predict_travel_cm": r["predict_travel_cm"],
            "fps": r["fps"],
        })
    return out


def verify(rows, out, n_frame_check):
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        print(f"  {'OK  ' if cond else '실패'} {name}" + (f"   {detail}" if detail else ""))
        ok = ok and cond

    chk("행 수 7392", len(out) == 7392, f"{len(out)}")
    chk("video_id 고유", len(set(r["video_id"] for r in out)) == len(out))
    chk("block_id 고유", len(set(r["block_id"] for r in out)) == len(out))

    cross = collections.Counter((r["speed"], r["anchor"]) for r in out)
    chk("speed x anchor 33셀 x 224 균등",
        len(cross) == 33 and set(cross.values()) == {224},
        f"{len(cross)}셀 {sorted(set(cross.values()))}")

    # anchor 가 곧 문맥 마지막 위치여야 한다 (= 마지막 프레임 복사 가설 차단의 전제)
    same = sum(1 for r in out
               if abs(float(r["x_context_end_cm"]) - float(r["anchor"])) < 1e-6)
    chk("x_context_end == anchor (마지막 관측 위치)", same == len(out), f"{same}/{len(out)}")

    # 라벨 공식이 metadata 를 재현하는가 — 이게 틀리면 회귀 라벨이 전부 틀린다
    worst_f, worst_l = 0.0, 0.0
    for r in out:
        a, v = float(r["anchor"]), float(r["speed"])
        af, fps = float(r["flat_anchor_frame"]), float(r["fps"])
        pred = lambda f: a + v * (f - af) / fps
        worst_f = max(worst_f, abs(pred(48) - float(r["x_predict_first_cm"])))
        worst_l = max(worst_l, abs(pred(93) - float(r["x_predict_last_cm"])))
    chk("라벨 공식 == metadata (predict_first/last)",
        worst_f < 1e-6 and worst_l < 1e-6, f"max|Δ| {worst_f:.3g} / {worst_l:.3g}")

    # cm -> 정규화 좌표가 선형이려면 깊이·카메라·환산계수가 상수여야 한다
    for k in ("obj_depth_cm", "frame_half_width_cm", "fps", "flat_anchor_frame"):
        u = set(r[k] for r in out)
        chk(f"{k} 상수", len(u) == 1, f"{sorted(u)[:3]}")

    # 물체가 항상 화면 안에 있는가 (격자 밖이면 soft-argmax 가 못 읽는다)
    hw = float(out[0]["frame_half_width_cm"])
    xs = [float(r["x_predict_last_cm"]) / hw for r in out]
    chk("예측 끝 정규화 x 가 [-1,1] 안", max(abs(min(xs)), abs(max(xs))) < 1.0,
        f"[{min(xs):.3f}, {max(xs):.3f}]")

    # 클래스 — attn_probe.yaml 에 넘길 값과 맞아야 job 이 안 죽는다
    for k, n in (("shape_pre", 7), ("color_pre", 8), ("env", 4)):
        u = sorted(set(r[k] for r in out))
        chk(f"{k} {n}종", len(u) == n, ",".join(u))

    # 프레임 실물 — 프로토콜이 읽는 32장이 다 있는가
    rng = random.Random(0)
    miss = []
    for r in rng.sample(out, min(n_frame_check, len(out))):
        for f in PROTOCOL_FRAMES:
            p = f"{FRAMES_ROOT}/{r['file_name']}/{f:06d}.png"
            if not os.path.isfile(p):
                miss.append(p)
                break
    chk(f"프레임 실물 (표본 {n_frame_check}클립 x 32장)", not miss,
        miss[0] if miss else "")

    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="검증 통과 시 파일까지 쓴다")
    ap.add_argument("--frame-check", type=int, default=200, help="프레임 실물 확인 클립 수")
    a = ap.parse_args()

    meta = f"{SRC}/metadata.csv"
    if not os.path.isfile(meta):
        die(f"metadata 가 없다 -> {meta}")
    rows = list(csv.DictReader(open(meta, encoding="utf-8")))
    print(f"metadata {len(rows)} rows   <- {meta}\n검증:")

    out = build(rows)
    if not verify(rows, out, a.frame_check):
        die("검증 실패 — 파일을 쓰지 않는다")
    print("\n검증 통과")

    if not a.write:
        print("(--write 를 주면 파일까지 쓴다)")
        return
    os.makedirs(OUT, exist_ok=True)
    for name in ("index.csv", "index_probe.csv"):     # 내용 동일 (위반/쌍이 없어 분기가 없다)
        p = f"{OUT}/{name}"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader()
            w.writerows(out)
        print(f"  wrote {p}   ({len(out)} rows x {len(COLS)} cols)")


if __name__ == "__main__":
    main()
