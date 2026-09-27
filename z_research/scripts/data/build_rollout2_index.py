#!/usr/bin/env python3
"""RollOut_v2 (운동 법칙 7종) 의 metadata.csv + plan -> index.csv / index_probe.csv.

⚠️ **2026-09-10 15:19 metadata (재생성 데이터) 부터 불가능 변이도 자기 궤적을 싣는다** (plan pair 와 1 px 안) 그리고 in_frame 은
   "물체 전체가 화면 안" 기준으로 가능/불가능 모두 metadata 를 쓴다. 라벨(px)은 계속 plan 에서 만든다 (metadata 와 대조 검증).
⚠️ (기록) 2026-09-09 17:17 판 — 가능 클립은 plan 과 0.05 cm 안으로 일치했지만 **wall 불가능 클립(통과)은 가능(정지) 궤적을 복사**하고
   있었다 (픽셀 대조: 물체는 벽을 통과한다). 그때는 불가능 변이의 in_frame 을 투영으로 다시 쟀다. 아래는 그 전 기록.
⚠️ **(수정 전) metadata 의 `object_*_by_sample` 은 flat 두 시나리오에서만 맞았다.** arc / fall / ledge / wall / ramp_a 는
   assemble 이 block 별 인자(primary/secondary/fixed) 없이 track_cm 을 불러 **같은 배열**을 써 넣었다
   (2026-09-09 실측: primary 가 달라도 by_sample 이 동일, 몽타주에서 물체 위에 안 놓임).
   정본은 plan (`UnrealEngine/gen/plans/blocks_rollout2.json`) 의 `x_sampled` / `z_sampled` 다 —
   build 가 `Gc.x_at/z_at` 를 block 인자로 만든 값이고, flat_v 에서 metadata 와 일치한다.
   불가능 변이(ledge float / wall pass)는 `x_sampled_pair` / `z_sampled_pair`.

위치 라벨 (32 샘플 = 프로토콜 프레임 0,3,…,93):
    x_cm, z_cm            plan 의 world 좌표 (cm)
    px_x, px_y            geom.screen_xy 를 metadata 의 카메라 상수로 재현 (flat_v 에서 max|Δ| 0.07 px)
    x_img, y_img          px / 144 − 1  (정규화 화면 좌표 [-1, 1], 256 리사이즈에도 보존)
    in_frame              metadata 그대로 (예측 16장은 전부 1, 문맥은 off-screen 이 있을 수 있다)

  python z_research/scripts/data/build_rollout2_index.py            # 검증만
  python z_research/scripts/data/build_rollout2_index.py --write
  python z_research/scripts/data/build_rollout2_index.py --set training_v5 --write   # RollOut_v2_training (v5 사물 없음 + props 사물; plan 2개 합침)
  python z_research/scripts/data/build_rollout2_index.py --set v2_decel --write      # 감속 flat_d / ramp_d 만 (2026-09-19, 기존 v2 뒤에 붙이는 용도)
  python z_research/scripts/data/build_rollout2_index.py --set training_v8 --write   # 2026-09-24 합본 14,360 clip (라벨 = metadata)

⚠️ 2026-09-24 부터 `RollOut_v2_training` 폴더가 **합본 14,360 clip** 으로 바뀌었다. 옛 빈 장면 288 은 빠졌고
   metadata 도 새것이라 `--set training_v6` 는 **더 이상 원천에서 다시 만들 수 없다** (계획에 없는 블록으로 죽는다).
   그 인덱스는 `data_csv/rollout_v2_training_v6/` 와 특징 옆 사본만이 기록이다.

학습셋 v1~v4 는 2026-09-11 삭제 — `training_v5` 만 남았다.
"""
from __future__ import annotations
import argparse, collections, csv, json, math, os, random, re, sys
import numpy as np

ROOT = "/data/hyuntak/project/2026/2027_cvpr/vjepa2"
SETS = {
    "v2": dict(src="/data2/local_datasets/world/world_analysis/RollOut_v2",
               frames_root="/local_datasets/world/world_analysis/RollOut_v2",
               plan="/data/hyuntak/project/2026/2027_cvpr/UnrealEngine/gen/plans/blocks_rollout2.json",
               out=f"{ROOT}/data_csv/rollout_v2",
               n=5488, scen={s: 784 for s in ("arc", "fall", "flat_a", "flat_v", "ledge", "ramp_a", "wall")},
               imp={"ledge": 392, "wall": 392}, holdout=224, flat=("flat_v", "flat_a")),
    # 2026-09-19: 감속 두 시나리오 (flat_d 수평 감속, ramp_d 오르막 감속) 만 따로. 기존 v2 인덱스·캐시 순서를 건드리지 않고
    # 새 clip 만 추출해 뒤에 붙이기 위한 세트다 (merge_token_cache.py + concat_index.py). only = metadata 에서 이 시나리오만 쓴다.
    "v2_decel": dict(src="/data2/local_datasets/world/world_analysis/RollOut_v2",
                     frames_root="/local_datasets/world/world_analysis/RollOut_v2",
                     plan="/data/hyuntak/project/2026/2027_cvpr/UnrealEngine/gen/plans/blocks_rollout2.json",
                     out=f"{ROOT}/data_csv/rollout_v2_decel",
                     n=1568, scen={"flat_d": 784, "ramp_d": 784}, imp={}, holdout=224, flat=("flat_d",),
                     only=("flat_d", "ramp_d")),
    "training_v5": dict(src="/data2/local_datasets/world/world_analysis/RollOut_v2_training",
                        frames_root="/local_datasets/world/world_analysis/RollOut_v2_training",
                        plan=["/data/hyuntak/project/2026/2027_cvpr/UnrealEngine/gen/plans/blocks_rollout2_training_v5.json",
                              "/data/hyuntak/project/2026/2027_cvpr/UnrealEngine/gen/plans/blocks_rollout2_training_props.json"],
                        out=f"{ROOT}/data_csv/rollout_v2_training_v5",
                        n=None, scen=None, imp={}, holdout=0, flat=None),      # 09-11 19:34 합본: 사물 없음 4,480 + 사물 3,584. plan 두 개
    # 2026-09-22: v5 + **빈 장면 288 clip** (물체 없음, 장면 요소는 그대로: empty_flat/ledge/ramp/wall x 배경 4).
    #   presence (있음/없음) 를 같이 내는 자를 학습하기 위한 세트. 빈 clip 은 plan 의 x_sampled/z_sampled 가 0 이고
    #   metadata 의 in_frame_by_sample 이 전부 0 이라 visible_by_sample 도 전부 0 → 위치 손실에서 빠지고 presence 음성이 된다.
    "training_v6": dict(src="/data2/local_datasets/world/world_analysis/RollOut_v2_training",
                        frames_root="/local_datasets/world/world_analysis/RollOut_v2_training",
                        plan=["/data/hyuntak/project/2026/2027_cvpr/UnrealEngine/gen/plans/blocks_rollout2_training_v5.json",
                              "/data/hyuntak/project/2026/2027_cvpr/UnrealEngine/gen/plans/blocks_rollout2_training_props.json",
                              "/data/hyuntak/project/2026/2027_cvpr/UnrealEngine/gen/plans/blocks_rollout2_training_empty.json"],
                        out=f"{ROOT}/data_csv/rollout_v2_training_v6",
                        n=None, scen=None, imp={}, holdout=0, flat=None),
    # 2026-09-24: **RollOut_v2_training 합본 (v5 4,480 + props 3,584 + 증축 6,296 = 14,360 clip).**
    #   데이터 쪽 이름은 "v6 증축" 이지만 우리 인덱스 이름 `training_v6` 은 이미 (v5 + 옛 빈 장면 288) 에 쓰였으므로 **v8** 로 둔다 (사용자 명명 2026-09-24 — RollOutV3 의 지금 자).
    #   증축분: 구조물 (ledge/wedge/wall) x 운동 4 종 **물체가 있는** clip 4,032 · 화면 끝 1,344 · 빈 장면 920 (옛 288 대체).
    #   ⚠️ **라벨은 metadata 에서 읽는다** (`label_src`). 새 계획 `blocks_rollout2_training_v6.json` 의 x_sampled 는
    #      dirn=+1 로 쓰인 값이라 **거울상 2,688 블록은 부호가 반대**다 (실측 최대 |Δx| 2,693 cm, 2026-09-24).
    #      픽셀과 맞는 것은 metadata 다 (데이터 README §5). 옛 v5/props 계획은 metadata 와 0.1 cm 안에서 같다.
    #   ⚠️ **구조물 속 샘플 (`scenery_intersects_by_sample`) 은 제외**한다 — 양성도 음성도 아니다.
    #      안 빼면 양성 9,022 튜블릿이 "물체가 구조물 속" 이고 음성 4,095 가 "구조물 = 없음" 을 가르친다
    #      (이 데이터가 없애려던 바로 그 지름길). 데이터 README §3.
    #   가중치 `cell_weight_by_sample` (위치 분포 평탄화) · `balance_weight` (구조물·방해물 ↔ 물체유무 탈상관) 를 싣는다.
    "training_v8": dict(src="/data2/local_datasets/world/world_analysis/RollOut_v2_training",
                        frames_root="/local_datasets/world/world_analysis/RollOut_v2_training",
                        plan=["/data/hyuntak/project/2026/2027_cvpr/UnrealEngine/gen/plans/blocks_rollout2_training_v5.json",
                              "/data/hyuntak/project/2026/2027_cvpr/UnrealEngine/gen/plans/blocks_rollout2_training_props.json",
                              "/data/hyuntak/project/2026/2027_cvpr/UnrealEngine/gen/plans/blocks_rollout2_training_v6.json"],
                        out=f"{ROOT}/data_csv/rollout_v2_training_v8",
                        n=None, scen=None, imp={}, holdout=0, flat=None, label_src="metadata"),
}
SRC = FRAMES_ROOT = PLAN = OUT = None      # main() 이 --set 으로 채운다
LABEL_SRC = "plan"                         # main() 이 --set 으로 채운다 ("metadata" 면 라벨을 metadata 에서)
PROTOCOL_FRAMES = list(range(0, 94, 3))
OBJ_Y = 260.0                                   # geom.py OBJ_Y (카메라에서 물체 평면까지의 y)

COLS = [
    "video_id", "file", "file_name", "block_id", "source_block", "variant",
    "plausible", "pair_id", "condition", "motion", "has_occlusion",
    "violation_type", "game_name", "role",
    "shape_pre", "color_pre", "shape_post", "color_post", "env", "surface",
    "obj_apparent_px", "obj_size_cm", "obj_depth_cm", "pair_group", "probe_type",
    # ── 이 세트의 축 ──
    "scenario", "primary_name", "primary", "secondary_name", "secondary", "holdout",
    # ── 위치 라벨 (32 샘플, 공백 구분) ──
    "sample_frames", "x_cm_by_sample", "z_cm_by_sample", "px_x_by_sample", "px_y_by_sample",
    "in_frame_by_sample", "visible_by_sample", "ignore_by_sample", "occ_frac_by_sample", "frame_half_width_cm", "fps", "resolution",
    # ── training_v8 부터 (없는 세트에서는 빈 칸) ──
    "prop_intersects_by_sample", "scenery_intersects_by_sample", "cell_weight_by_sample", "balance_weight", "dirn",
]
# visible_by_sample (2026-09-11, training_v5 부터): in_frame 이고 **사물에 가려지지 않은** 샘플. 학습 라벨은 이것만 쓴다.
#   props 셋은 metadata 에 hidden 표시가 없다 (has_occlusion 0). prop_y == 260 (물체와 같은 깊이, 얇은 판) 이면 물체가 판을 관통하며
#   일부/전부 가려지므로, 물체 x 가 판의 화면 x 범위 ± 물체 반폭 안이면 가려진 것으로 본다. prop_y == 400 (뒤) 는 가림 없음.
OBJ_HALF_PX = 18.0


def die(m):
    print(f"\nERROR: {m}\n", file=sys.stderr); sys.exit(1)


def arr(s):
    return np.array([float(v) for v in re.split(r"[;,\s|]+", str(s).strip().strip("[]")) if v], float)


def screen(x, z, r):
    """geom.screen_xy 재현. world +x 는 화면 왼쪽 (부호 MINUS)."""
    fwd = OBJ_Y - float(r["cam_y"]); up = z - float(r["cam_z"])
    th = math.radians(float(r["cam_pitch"])); tanh = math.tan(math.radians(float(r["fov_deg"])) / 2)
    fw = fwd * math.cos(th) + up * math.sin(th); uu = -fwd * math.sin(th) + up * math.cos(th)
    half = float(r["resolution"]) / 2
    return half - (x - float(r["cam_x"])) / (fw * tanh) * half, half - uu / (fw * tanh) * half


# ⚠️ 2026-09-22 변경 — 가림을 **비율**로 잰다 (그 전에는 실루엣이 닿는 순간부터 전부 "안 보임" 이었다).
#   occ_frac = 물체 가로폭 중 판에 덮인 비율 (0 = 안 가림, 1 = 완전히 가림). `visible_by_sample` 은 이제
#   in_frame 이고 occ_frac < OCC_HIDDEN 일 때만 1 이다. 부분 가림 (OCC_PARTIAL ~ OCC_HIDDEN) 은
#   학습에서 **손실에서 빼는 구간**으로 쓰라고 비율을 그대로 싣는다 (문턱은 학습 쪽에서 정한다).
OCC_HIDDEN = 0.9


def occ_frac(r, px):
    """샘플마다 물체 가로폭 중 판에 덮인 비율 (0~1). 판이 없거나 깊이가 다르면 전부 0."""
    if not r.get("prop_w") or float(r.get("prop_w") or 0) == 0 or float(r.get("prop_y") or 0) != OBJ_Y:
        return [0.0] * len(px)
    px_x, _ = screen(float(r["prop_x"]), 0.0, r)                                   # 판의 화면 x (물체 평면과 같은 깊이)
    pw = float(r["prop_w"]) / 2 / float(r["frame_half_width_cm"]) * float(r["resolution"]) / 2
    out = []
    for p in px:
        lo, hi = max(p[0] - OBJ_HALF_PX, px_x - pw), min(p[0] + OBJ_HALF_PX, px_x + pw)
        out.append(max(0.0, min(1.0, (hi - lo) / (2 * OBJ_HALF_PX))))
    return out


def intersects(r):
    """생성기가 준 샘플별 '물체가 장애물과 겹친다' 표시 (2026-09-22 metadata 부터). 없으면 전부 0."""
    v = str(r.get("prop_intersects_by_sample") or "").split()
    return [x == "1" for x in v] if v else [False] * 32


def scenery(r):
    """물체가 **구조물 (단·쐐기·벽) 속**인 샘플 (training_v8 부터). 없으면 전부 0."""
    v = str(r.get("scenery_intersects_by_sample") or "").split()
    return [x == "1" for x in v] if v else [False] * 32


def visible(r, px, frac=None):
    """**presence 양성** = 화면 안이고 장애물과 안 겹치고 구조물 속도 아닌 샘플.
    겹치는 샘플 (`prop_intersects_by_sample` = 1) 은 양성도 음성도 아니고 `ignore_by_sample` 로 빠진다
    (물체가 부분적으로 보여 '있다/없다' 가 모호하다 — 2026-09-22 사용자 결정)."""
    inf = [v for v in r["in_frame_by_sample"].split()]
    ix, sx = intersects(r), scenery(r)
    return " ".join("1" if v == "1" and not x and not y else "0" for v, x, y in zip(inf, ix, sx))


def ignore(r):
    """손실에서 빼는 샘플 = (화면 안인데 장애물과 겹친다) 또는 (구조물 속).

    ⚠️ 구조물 속은 **화면 안팎과 무관하게** 뺀다 (2026-09-24). 데이터 README §3 — 마스크는 프레임 단위라
       "여기엔 물체가 없다" 를 가르치지 않고 그 프레임을 학습에서 뺀다. 화면 밖이면서 구조물 속인 샘플이
       12,761 개 있는데, 이걸 음성으로 두면 "구조물 = 없음" 을 다시 가르친다.
    """
    inf = [v for v in r["in_frame_by_sample"].split()]
    return " ".join("1" if (v == "1" and x) or y else "0" for v, x, y in zip(inf, intersects(r), scenery(r)))


def build(rows, plan):
    out = []
    j = lambda v: " ".join(f"{x:.2f}" for x in v)
    for r in rows:
        vid = os.path.basename(r["file_name"])
        bid = re.sub(r"_(pos|imp)_roll$", "", r["name"])
        b = plan.get(bid)
        if b is None:
            die(f"plan 에 없는 block {bid}")
        imp = r["is_possible"] in ("0", 0, "False")
        if LABEL_SRC == "metadata":            # training_v8: 거울상 블록의 계획 x 는 부호가 반대다 — metadata 를 쓴다
            mx, mz = r["object_x_cm_by_sample"], r["object_z_cm_by_sample"]
            xs = list(arr(mx)) if str(mx).strip() else [0.0] * 32      # 빈 장면: 좌표 없음 (라벨에 안 쓰인다)
            zs = list(arr(mz)) if str(mz).strip() else [0.0] * 32
        else:
            xs = b["x_sampled_pair"] if imp else b["x_sampled"]
            zs = b["z_sampled_pair"] if imp else b["z_sampled"]
        if len(xs) != 32:
            die(f"{bid}: x_sampled 길이 {len(xs)}")
        px = [screen(x, z, r) for x, z in zip(xs, zs)]
        out.append({
            "video_id": vid, "file": "", "file_name": r["file_name"],
            "block_id": bid,                      # ledge/wall 은 pos/imp 가 같은 block (문맥 동일)
            "source_block": r["block"], "variant": r["clip"],
            "plausible": 0 if imp else 1, "pair_id": "A",
            "condition": r["condition"], "motion": r["motion"], "has_occlusion": r["has_occlusion"],
            "violation_type": r["violation_type"], "game_name": r["game_name"], "role": r["role"],
            "shape_pre": r["shape_pre"], "color_pre": r["color_pre"],
            "shape_post": r["shape_post"], "color_post": r["color_post"],
            "env": r["env"], "surface": r["surface"],
            "obj_apparent_px": r["obj_apparent_px"], "obj_size_cm": r["obj_size_cm"],
            "obj_depth_cm": r["obj_depth_cm"], "pair_group": r["pair_group"],
            "probe_type": "imp" if imp else "obj",
            "scenario": r["scenario"], "primary_name": r["primary_name"], "primary": r["primary"],
            "secondary_name": r["secondary_name"], "secondary": r["secondary"], "holdout": r["holdout"],
            "sample_frames": " ".join(str(f) for f in PROTOCOL_FRAMES),
            "x_cm_by_sample": j(xs), "z_cm_by_sample": j(zs),
            "px_x_by_sample": j(p[0] for p in px), "px_y_by_sample": j(p[1] for p in px),
            # 09-10 15:19 metadata 부터 불가능 변이도 자기 궤적(plan pair 와 1 px 안) 기준 in_frame 이다 — 전부 metadata 를 쓴다.
            # 기준은 "물체 전체가 화면 안" (가장자리 ~20 px 안쪽부터 0). 그 전 판은 wall 불가능이 정지 궤적을 복사해 투영으로 다시 쟀었다
            "in_frame_by_sample": r["in_frame_by_sample"],
            "visible_by_sample": visible(r, px),
            "ignore_by_sample": ignore(r),
            "occ_frac_by_sample": j(occ_frac(r, px)),       # 기하로 계산한 가림 비율 (참고용; 판정은 생성기 표시로 한다)
            "frame_half_width_cm": r["frame_half_width_cm"], "fps": r["fps"], "resolution": r["resolution"],
            "prop_intersects_by_sample": r.get("prop_intersects_by_sample", ""),
            "scenery_intersects_by_sample": r.get("scenery_intersects_by_sample", ""),
            "cell_weight_by_sample": r.get("cell_weight_by_sample", ""),
            "balance_weight": r.get("balance_weight", ""),
            "dirn": b.get("dirn", ""),
        })
    return out


def verify(rows, out, n_frame_check, S):
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        print(f"  {'OK  ' if cond else '실패'} {name}" + (f"   {detail}" if detail else "")); ok = ok and cond

    chk(f"행 수 {S['n']}", len(out) == S["n"], f"{len(out)}")
    chk("video_id 고유", len(set(r["video_id"] for r in out)) == len(out))
    sc = collections.Counter(r["scenario"] for r in out)
    chk(f"시나리오 {S['scen']}", dict(sc) == S["scen"], str(dict(sc)))
    imp = collections.Counter(r["scenario"] for r in out if r["plausible"] == 0)
    chk(f"불가능 변이 {S['imp']}", dict(imp) == S["imp"], str(dict(imp)))
    # ledge/wall: pos/imp 가 block 을 공유하고 문맥(샘플 0..15) 라벨이 같다
    byb = collections.defaultdict(list)
    for r in out:
        byb[r["block_id"]].append(r)
    bad = 0
    for bid, rs in byb.items():
        if len(rs) == 2:
            a, b = (arr(x["x_cm_by_sample"]) for x in rs)
            if not np.allclose(a[:16], b[:16]):
                bad += 1
    chk("pos/imp 문맥 16샘플 라벨 동일", bad == 0, f"{bad} block 어긋남")
    # metadata 가 맞는 flat 두 시나리오에서 plan 라벨 == metadata by_sample, 투영 == metadata px
    mrow = {r["name"]: r for r in rows}
    wx = wp = 0.0
    for r in out:
        if r["scenario"] not in S["flat"]:
            continue
        m = mrow[r["video_id"]]
        if not str(m["object_x_cm_by_sample"]).strip():     # 빈 장면 clip (training_v6): 물체 좌표가 없다 — 대조 대상 아님
            continue
        wx = max(wx, np.abs(arr(r["x_cm_by_sample"]) - arr(m["object_x_cm_by_sample"])).max())
        wp = max(wp, np.abs(arr(r["px_x_by_sample"]) - arr(m["object_px_x_by_sample"])).max(),
                 np.abs(arr(r["px_y_by_sample"]) - arr(m["object_px_y_by_sample"])).max())
    if LABEL_SRC == "metadata":
        # 라벨이 metadata 이므로 위 대조는 자명하다. 대신 **계획과의 관계**를 dirn 별로 확인한다
        plan_ = PLAN_BLOCKS
        wpos = wneg = wflip = 0.0; npos = nneg = 0
        for r in out:
            m = mrow[r["video_id"]]
            if not str(m["object_x_cm_by_sample"]).strip():
                continue
            b = plan_[r["block_id"]]; px_ = arr(b["x_sampled"]); mx_ = arr(m["object_x_cm_by_sample"])
            if b.get("dirn", 1) == -1:
                nneg += 1; wneg = max(wneg, np.abs(px_ - mx_).max())
                wflip = max(wflip, np.abs((2 * float(m["cam_x"]) - px_) - mx_).max())
            else:
                npos += 1; wpos = max(wpos, np.abs(px_ - mx_).max())
        chk("dirn=+1 블록: 계획 x == metadata x", wpos < 0.2, f"{npos} clip, max|Δ| {wpos:.3f} cm")
        chk("dirn=−1 블록: 계획 x 를 카메라 축으로 뒤집으면 == metadata x", wflip < 0.2,
            f"{nneg} clip, 뒤집기 전 max|Δ| {wneg:.0f} cm → 뒤집은 뒤 {wflip:.3f} cm")
    else:
        chk(f"{S['flat']}: plan 라벨 == metadata by_sample", wx < 0.06, f"max|Δ| {wx:.3f} cm")
    chk(f"{S['flat']}: 투영식 == metadata px", wp < 0.2, f"max|Δ| {wp:.3f} px")
    # 비-flat 은 metadata by_sample 이 틀린 것을 확인 (primary 가 달라도 같은 배열)
    if "arc" in sc:
        a = [r for r in rows if r["scenario"] == "arc"]
        same = np.allclose(arr(a[0]["object_x_cm_by_sample"]), arr(a[-1]["object_x_cm_by_sample"]))
        chk("(기록) metadata arc by_sample 이 primary 와 무관하게 동일한가 (09-09 16:14 판의 버그; 이후 판은 고쳐짐)", True, f"동일={same}")
    if S["imp"]:
        # 불가능 변이: metadata by_sample 이 plan 의 pair 궤적을 따르는가 (09-10 15:19 판부터 그렇다; 그 전엔 wall 이 정지 궤적을 복사했다)
        wi = 0.0
        for r in out:
            if r["plausible"] == 0:
                mr = mrow[r["video_id"]]
                wi = max(wi, np.abs(arr(r["px_x_by_sample"]) - arr(mr["object_px_x_by_sample"])).max(), np.abs(arr(r["px_y_by_sample"]) - arr(mr["object_px_y_by_sample"])).max())
        chk("불가능 변이: plan pair 궤적 투영 == metadata px", wi < 1.0, f"max|Δ| {wi:.2f} px")
    # 예측 16 샘플은 전부 화면 안, px 도 [0, 288] 안
    # ── 라벨 감사 (2026-09-22): 양성·음성·제외가 서로 겹치지 않고 셋의 합이 전체인가 ──
    A = lambda k: np.stack([arr(r[k]) for r in out])
    vis_, ign_, inf_ = A("visible_by_sample") > 0, A("ignore_by_sample") > 0, A("in_frame_by_sample") > 0
    emp_ = np.array([r["scenario"] == "empty" for r in out])[:, None].repeat(vis_.shape[1], 1)
    chk("라벨: 양성 ∩ 제외 = 0", int((vis_ & ign_).sum()) == 0, f"{int((vis_ & ign_).sum())} 샘플")
    scn_ = np.stack([np.array([x == "1" for x in (r["scenery_intersects_by_sample"].split() or ["0"] * 32)])
                     for r in out])
    chk("라벨: 화면 밖 제외는 전부 구조물 속", int((ign_ & ~inf_ & ~scn_).sum()) == 0,
        f"{int((ign_ & ~inf_ & ~scn_).sum())} 샘플 (구조물 속 화면 밖 {int((ign_ & ~inf_ & scn_).sum())} 은 규칙대로 제외)")
    chk("라벨: 화면 안 ⊆ 양성 + 제외", int((inf_ & ~(vis_ | ign_)).sum()) == 0, f"{int((inf_ & ~(vis_ | ign_)).sum())} 샘플")
    chk("라벨: 양성 ⊆ 화면 안", int((vis_ & ~inf_).sum()) == 0, f"{int((vis_ & ~inf_).sum())} 샘플")
    chk("라벨: 구조물 속은 양성이 아니다", int((vis_ & scn_).sum()) == 0, f"{int((vis_ & scn_).sum())} 샘플")
    chk("라벨: 빈 장면은 전부 음성", int((emp_ & vis_).sum()) == 0, f"{int((emp_ & vis_).sum())} 샘플")
    chk("라벨: 빈 장면에 제외 없음", int((emp_ & ign_).sum()) == 0, f"{int((emp_ & ign_).sum())} 샘플")
    occ_ = A("occ_frac_by_sample")
    chk("(기록) 생성기 겹침 표시 vs 기하 계산", True,
        f"겹침 {int(ign_.sum())} 샘플, 그중 기하 occ>0 인 것 {int((ign_ & (occ_ > 0)).sum())}; "
        f"겹침 아닌데 기하 occ>0.5 인 것 {int((~ign_ & (occ_ > 0.5)).sum())}")
    n_pos, n_ign = int(vis_.sum()), int(ign_.sum())
    chk("(기록) 샘플 집계", True,
        f"양성 {n_pos} / 음성 {int((~vis_ & ~ign_).sum())} (빈 장면 {int((emp_ & ~ign_).sum())}, 화면 밖 {int((~inf_ & ~emp_).sum())}) / 제외 {n_ign}")

    empty = [r for r in out if r["scenario"] == "empty"]
    if empty:                                                # training_v6: 물체 없는 clip 은 presence 음성이어야 한다
        chk(f"빈 장면 {len(empty)} clip: visible_by_sample 전부 0",
            all(set(r["visible_by_sample"].split()) <= {"0"} for r in empty))
        chk(f"빈 장면 {len(empty)} clip: shape/color 가 none", all(r["shape_pre"] == "none" for r in empty))
    pos = [r for r in out if r["plausible"] == 1 and r["scenario"] != "empty"]
    inf = np.stack([arr(r["in_frame_by_sample"]) for r in pos])
    # 09-10 13:30 metadata 는 in_frame 을 더 엄격하게 잰다 (물체 일부가 화면 밖이면 0): arc 정점(샘플 14~20, y≈16px) 392 clip.
    # 위치 라벨은 안 바뀌었고 (index 대비 ≤0.11 px) 프레임도 그대로다 — 기록만 하고 in_frame 은 metadata 를 따른다
    part = collections.Counter(r["scenario"] for r, i in zip(pos, inf) if not i[16:].all())
    chk("(기록) 가능 변이: 예측 16 샘플 중 일부 off-screen 인 clip", True, str(dict(part)) if part else "없음")
    pxs = np.stack([arr(r["px_x_by_sample"]) for r in pos])[:, 16:]; pys = np.stack([arr(r["px_y_by_sample"]) for r in pos])[:, 16:]
    if inf[:, 16:].all():
        chk("가능 변이: 예측 px 가 화면 안 (물체 반폭 18px 여유)", pxs.min() > -18 and pxs.max() < 306 and pys.min() > -18 and pys.max() < 306,
            f"x [{pxs.min():.0f},{pxs.max():.0f}] y [{pys.min():.0f},{pys.max():.0f}]")
    else:   # 화면 밖 샘플을 일부러 둔 세트 (training_v5): in_frame 이 0 인 샘플만 화면 밖이어야 한다
        m = inf[:, 16:] > 0
        chk("가능 변이: in_frame=1 인 예측 샘플은 화면 안", pxs[m].min() > -18 and pxs[m].max() < 306 and pys[m].min() > -18 and pys[m].max() < 306,
            f"in_frame 샘플 x [{pxs[m].min():.0f},{pxs[m].max():.0f}] y [{pys[m].min():.0f},{pys[m].max():.0f}] / off-frame {100*(~m).mean():.1f}%")
    if "wall" in S["imp"]:
        impw = [r for r in out if r["plausible"] == 0 and r["scenario"] == "wall"]
        offs = sum(1 for r in impw if "0" in r["in_frame_by_sample"].split()[16:])
        chk("(기록) wall 불가능 변이는 미래에 화면 밖으로 나간다", True, f"{offs}/{len(impw)} clip 이 일부 슬롯 off-screen")
    hold = collections.Counter((r["scenario"], r["holdout"]) for r in out)
    chk(f"holdout 이 시나리오마다 {S['holdout']}", all(hold[(s, "1")] == S["holdout"] for s in sc))
    for k, n in (("shape_pre", 7), ("color_pre", 8), ("env", 4)):
        u = sorted(set(r[k] for r in out if r["plausible"] == 1 and r["scenario"] != "empty")); chk(f"{k} {n}종", len(u) == n, ",".join(u))
    for k in ("frame_half_width_cm", "fps", "resolution", "obj_depth_cm"):
        u = set(r[k] for r in out); chk(f"{k} 상수", len(u) == 1, f"{sorted(u)}")
    rng = random.Random(0); miss = []
    for r in rng.sample(out, min(n_frame_check, len(out))):
        for f in PROTOCOL_FRAMES:
            p = f"{FRAMES_ROOT}/{r['file_name']}/{f:06d}.png"
            if not os.path.isfile(p):
                miss.append(p); break
    chk(f"프레임 실물 (표본 {n_frame_check}클립 x 32장)", not miss, miss[0] if miss else "")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true"); ap.add_argument("--frame-check", type=int, default=200)
    ap.add_argument("--set", choices=sorted(SETS), default="v2")
    a = ap.parse_args()
    global SRC, FRAMES_ROOT, PLAN, OUT, LABEL_SRC, PLAN_BLOCKS
    S = SETS[a.set]; SRC, FRAMES_ROOT, PLAN, OUT = S["src"], S["frames_root"], S["plan"], S["out"]
    LABEL_SRC = S.get("label_src", "plan")
    if S["n"] is None:                                     # 구성을 metadata 에서 읽는 세트 (training_v3)
        _rows = list(csv.DictReader(open(f"{SRC}/metadata.csv", encoding="utf-8")))
        S = dict(S, n=len(_rows), scen=dict(collections.Counter(r["scenario"] for r in _rows)), flat=tuple(sorted(set(r["scenario"] for r in _rows))))
        print(f"[{a.set}] 구성 (metadata): n={S['n']} scen={S['scen']}")
    rows = list(csv.DictReader(open(f"{SRC}/metadata.csv", encoding="utf-8")))
    if S.get("only"):                                      # 부분 세트 (v2_decel): 지정 시나리오 행만, metadata 순서 그대로
        rows = [r for r in rows if r["scenario"] in S["only"]]
    plan = {}
    for pf in (PLAN if isinstance(PLAN, list) else [PLAN]):
        J = json.load(open(pf)); plan.update({b["id"]: b for b in (J["blocks"] if isinstance(J, dict) else J)})
    PLAN_BLOCKS = plan
    print(f"metadata {len(rows)} rows, plan {len(plan)} blocks, 라벨 원천 = {LABEL_SRC}\n검증:")
    out = build(rows, plan)
    if not verify(rows, out, a.frame_check, S):
        die("검증 실패 — 파일을 쓰지 않는다")
    print("\n검증 통과")
    if not a.write:
        print("(--write 를 주면 파일까지 쓴다)"); return
    os.makedirs(OUT, exist_ok=True)
    for name in ("index.csv", "index_probe.csv"):
        p = f"{OUT}/{name}"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(out)
        print(f"  wrote {p}   ({len(out)} rows x {len(COLS)} cols)")


if __name__ == "__main__":
    main()
