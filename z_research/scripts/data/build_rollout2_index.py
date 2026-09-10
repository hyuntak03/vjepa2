#!/usr/bin/env python3
"""RollOut_v2 (운동 법칙 7종) 의 metadata.csv + plan -> index.csv / index_probe.csv.

⚠️ **2026-09-09 17:17 에 metadata 가 수정됐다** — 가능 클립은 전 시나리오에서 plan 과 0.05 cm 안으로 일치한다.
   그러나 **wall 의 불가능 클립(통과)은 여전히 가능(정지) 궤적을 복사**하고 있다 (픽셀 대조: 물체는 벽을 통과한다,
   scratch `wall_label_check.png`). 그래서 라벨은 계속 plan 을 쓴다. 아래는 수정 전 기록.
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
  python z_research/scripts/data/build_rollout2_index.py --set training --write   # RollOut_v2_training (readout 학습셋)

--set training (2026-09-10): 무중력 직선 line_x / line_z 896 clip, pos 만, holdout 없음. metadata by_sample 이 plan 과
0.05 cm 안에서 일치하지만 v2 와 같은 경로(plan)로 라벨을 만든다.
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
    "training": dict(src="/data2/local_datasets/world/world_analysis/RollOut_v2_training",
                     frames_root="/local_datasets/world/world_analysis/RollOut_v2_training",
                     plan="/data/hyuntak/project/2026/2027_cvpr/UnrealEngine/gen/plans/blocks_rollout2_training.json",
                     out=f"{ROOT}/data_csv/rollout_v2_training",
                     n=896, scen={"line_x": 448, "line_z": 448}, imp={}, holdout=0, flat=("line_x", "line_z")),
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
    "in_frame_by_sample", "frame_half_width_cm", "fps", "resolution",
]


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
            # metadata 의 in_frame 은 가능 변이 기준이다. 불가능 변이(wall pass 는 화면 밖으로 나간다)는 투영값으로 다시 잰다
            "in_frame_by_sample": (" ".join("1" if -18 < p[0] < 306 and -18 < p[1] < 306 else "0" for p in px)
                                   if imp else r["in_frame_by_sample"]),
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
    if "wall" in S["imp"]:
        # wall 불가능 클립은 어느 판이든 정지 궤적을 쓴다 — plan 의 pair 궤적과 갈리는지 기록만
        w = [r for r in out if r["scenario"] == "wall" and r["plausible"] == 0][0]; mw = mrow[w["video_id"]]
        chk("(기록) wall 불가능: metadata by_sample 이 plan pair 궤적과 갈리는가", True,
            f"max|Δ| {np.abs(arr(w['x_cm_by_sample']) - arr(mw['object_x_cm_by_sample'])).max():.0f} cm (plan 이 정본)")
    # 예측 16 샘플은 전부 화면 안, px 도 [0, 288] 안
    pos = [r for r in out if r["plausible"] == 1]
    inf = np.stack([arr(r["in_frame_by_sample"]) for r in pos])
    # 09-10 13:30 metadata 는 in_frame 을 더 엄격하게 잰다 (물체 일부가 화면 밖이면 0): arc 정점(샘플 14~20, y≈16px) 392 clip.
    # 위치 라벨은 안 바뀌었고 (index 대비 ≤0.11 px) 프레임도 그대로다 — 기록만 하고 in_frame 은 metadata 를 따른다
    part = collections.Counter(r["scenario"] for r, i in zip(pos, inf) if not i[16:].all())
    chk("(기록) 가능 변이: 예측 16 샘플 중 일부 off-screen 인 clip", True, str(dict(part)) if part else "없음")
    pxs = np.stack([arr(r["px_x_by_sample"]) for r in pos])[:, 16:]; pys = np.stack([arr(r["px_y_by_sample"]) for r in pos])[:, 16:]
    chk("가능 변이: 예측 px 가 화면 안 (물체 반폭 18px 여유)", pxs.min() > -18 and pxs.max() < 306 and pys.min() > -18 and pys.max() < 306,
        f"x [{pxs.min():.0f},{pxs.max():.0f}] y [{pys.min():.0f},{pys.max():.0f}]")
    if "wall" in S["imp"]:
        impw = [r for r in out if r["plausible"] == 0 and r["scenario"] == "wall"]
        offs = sum(1 for r in impw if "0" in r["in_frame_by_sample"].split()[16:])
        chk("(기록) wall 불가능 변이는 미래에 화면 밖으로 나간다 (in_frame 을 투영으로 다시 잼)", True, f"{offs}/{len(impw)} clip 이 일부 슬롯 off-screen")
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
    rows = list(csv.DictReader(open(f"{SRC}/metadata.csv", encoding="utf-8")))
    J = json.load(open(PLAN)); plan = {b["id"]: b for b in (J["blocks"] if isinstance(J, dict) else J)}
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
