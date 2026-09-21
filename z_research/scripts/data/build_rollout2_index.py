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
}
SRC = FRAMES_ROOT = PLAN = OUT = None      # main() 이 --set 으로 채운다
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
    "in_frame_by_sample", "visible_by_sample", "frame_half_width_cm", "fps", "resolution",
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


def visible(r, px):
    """in_frame 이고 사물(같은 깊이의 판) 에 가려지지 않은 샘플 → '1', 아니면 '0'. 사물 컬럼이 없거나 전부 0 이면 in_frame 그대로."""
    inf = [v for v in r["in_frame_by_sample"].split()]
    if not r.get("prop_w") or float(r.get("prop_w") or 0) == 0 or float(r.get("prop_y") or 0) != OBJ_Y:
        return " ".join(inf)
    px_x, _ = screen(float(r["prop_x"]), 0.0, r)                                   # 판의 화면 x (물체 평면과 같은 깊이)
    half = float(r["prop_w"]) / 2 / float(r["frame_half_width_cm"]) * float(r["resolution"]) / 2 + OBJ_HALF_PX
    return " ".join("1" if v == "1" and abs(p[0] - px_x) > half else "0" for v, p in zip(inf, px))


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
            "frame_half_width_cm": r["frame_half_width_cm"], "fps": r["fps"], "resolution": r["resolution"],
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
        wx = max(wx, np.abs(arr(r["x_cm_by_sample"]) - arr(m["object_x_cm_by_sample"])).max())
        wp = max(wp, np.abs(arr(r["px_x_by_sample"]) - arr(m["object_px_x_by_sample"])).max(),
                 np.abs(arr(r["px_y_by_sample"]) - arr(m["object_px_y_by_sample"])).max())
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
    pos = [r for r in out if r["plausible"] == 1]
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
        u = sorted(set(r[k] for r in out if r["plausible"] == 1)); chk(f"{k} {n}종", len(u) == n, ",".join(u))
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
    global SRC, FRAMES_ROOT, PLAN, OUT
    S = SETS[a.set]; SRC, FRAMES_ROOT, PLAN, OUT = S["src"], S["frames_root"], S["plan"], S["out"]
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
    print(f"metadata {len(rows)} rows, plan {len(plan)} blocks\n검증:")
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
