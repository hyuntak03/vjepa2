#!/usr/bin/env python3
"""BlenderTrack v3 학습 목록 (팔마다) — train 장면에서 축 조건으로 고르고 (렌더 타이밍 통과만) 영상 A · B 목록을 만든다 (2026-11-01).

팔 = 이름 · 조건 (열=값, 여럿은 &) · 장면 수 N (0 = 조건에 맞는 전부) · 영상 (AB = 두 팔 · A = target 있는 팔만).
같은 N 이면 조건만 다르다 (데이터 양 고정). 고르기는 seed 0 고정 (학습 seed 와 별개).
출력 data_csv/blender_track_v3/arms/<이름>.npz — clip (영상 id) · vis (T,) target 보이는 비율 (B = nan) · root · layout mp4 · labels_root
  $P make_v3_meta.py --all      # ARMS 표 전부
"""
import argparse, csv, os
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
OUT = (os.environ.get("BT_ROOT") or "/data/hyuntak/project/2026/2027_cvpr/nfs_cache/blender_track") + "/v3"   # 11-02: env.sh 의 BT_ROOT (비우면 클러스터 자리)
ARMS = {   # 이름: (조건, 장면 수, 영상)
    "mix1600": ("", 1600, "AB"), "full": ("", 0, "AB"), "mix680": ("", 680, "AB"),
    "prim": ("objset=primitive", 1600, "AB"), "real": ("objset=realistic", 1600, "AB"),
    "front": ("camera=front", 1600, "AB"), "side": ("camera=side", 1600, "AB"), "bird": ("camera=bird", 1600, "AB"),
    "barrier": ("occ_type=barrier", 1600, "AB"), "object": ("occ_type=object", 1600, "AB"),
    "occstatic": ("occ_motion=static", 1600, "AB"), "occmoving": ("occ_motion=moving", 1600, "AB"),
    "tgtmoving": ("target_mode=moving", 1600, "AB"), "tgtstatic": ("target_mode=static", 1600, "AB"),
    "flat": ("place=flat", 1600, "AB"), "ramp": ("place=ramp", 1600, "AB"),
    "short": ("f_hide=short", 1600, "AB"), "long": ("f_hide=long", 1600, "AB"),
    "aonly": ("", 3200, "A"),                                                       # 빈 장면 뺌 (영상 수 3,200 으로 맞춤)
    "frontstatic": ("camera=front&occ_type=barrier&occ_motion=static", 0, "AB"),    # VOBench 와 가장 닮은 조각 (680)
}


def vis_of(path):
    z = np.load(path)
    if z["amodal"].shape[1] == 0 or str(z["roles"][0]) != "target":
        return np.full(z["amodal"].shape[0], np.nan, np.float16)
    a, v = z["amodal_n"][:, 0].astype(np.float32), z["visible_n"][:, 0].astype(np.float32)
    return np.where(a > 0, v / np.maximum(a, 1), np.nan).astype(np.float16)


def build(name):
    cond, N, which = ARMS[name]
    rows = [r for r in csv.DictReader(open(f"{REPO}/data_csv/blender_track_v3/train_arms.csv")) if r["variant"] == "pos_a" and r["render_timing_ok"] == "1"]
    for c in [c for c in cond.split("&") if c]:
        k, v = c.split("="); rows = [r for r in rows if r[k] == v]
    rng = np.random.default_rng(0)
    if N and N < len(rows):
        rows = [rows[i] for i in sorted(rng.choice(len(rows), N, replace=False))]
    clips = []
    for r in rows:
        a = os.path.basename(os.path.realpath(r["file"]))[:-4]                   # pos_a → clips/train40_<2b>.mp4
        b = f"{a.split('_')[0]}_{int(a.split('_')[1]) + 1:06d}"
        clips += [a] + ([b] if which == "AB" else [])
    vis = np.stack([vis_of(f"{OUT}/labels/{c}.npz") for c in clips])
    os.makedirs(f"{REPO}/data_csv/blender_track_v3/arms", exist_ok=True)
    np.savez(f"{REPO}/data_csv/blender_track_v3/arms/{name}.npz", clip=np.array(clips), vis=vis, root=np.array(f"{OUT}/clips"),
             layout=np.array("mp4"), labels_root=np.array(f"{OUT}/labels"))
    return len(rows), len(clips)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--all", action="store_true"); ap.add_argument("names", nargs="*")
    a = ap.parse_args()
    for n in (list(ARMS) if a.all else a.names):
        print(n, "장면 · 영상", build(n), ARMS[n])
