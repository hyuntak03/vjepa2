#!/usr/bin/env python3
"""BlenderTrack v3t (v3 + trapdoor) 학습 목록 (2026-11-01) — make_v3_meta.py 와 같은 꼴 (clip · vis · root · layout · labels_root).

팔 (v3 ablation 과 같은 설정으로 학습 — 2,000 step · 전역 batch 24):
  t_only       v3t (trapdoor 만) 렌더 타이밍 통과 장면 중 1,600 (모자라면 전부) × A · B          ↔ 비교: v3 섞음 1,600 (같은 장면 수)
  mix1600_t25  v3 섞음 1,600 과 같은 크기 — v3 1,200 장면 + v3t 400 장면 (trapdoor 25 %) × A · B   ↔ 비교: v3 섞음 1,600 (같은 크기, trapdoor 만 더함)
두 세트를 섞는 팔은 clips · labels 를 심볼릭 링크로 묶은 폴더 (blender_track/v3mix) 를 root 로 쓴다 (이름 train40_ · train60_ 은 겹치지 않는다).
출력 data_csv/blender_track_v3/arms/<이름>.npz (v3 ablation 큐가 읽는 자리)
  $P make_v3t_meta.py t_only mix1600_t25
"""
import csv, os, sys
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
BT = "/data/hyuntak/project/2026/2027_cvpr/nfs_cache/blender_track"
sys.path.insert(0, os.path.dirname(__file__))
from make_v3_meta import vis_of  # noqa: E402

SRC = {"v3": f"{REPO}/data_csv/blender_track_v3/train_arms.csv", "v3t": f"{REPO}/data_csv/blender_track_v3t/train_arms.csv"}
ARMS = {"t_only": {"v3t": 1600}, "mix1600_t25": {"v3": 1200, "v3t": 400}}


def pairs(src, n, seed):
    rows = [r for r in csv.DictReader(open(SRC[src])) if r["variant"] == "pos_a" and r["render_timing_ok"] == "1"]
    rng = np.random.default_rng(seed)
    if n < len(rows):
        rows = [rows[i] for i in sorted(rng.choice(len(rows), n, replace=False))]
    out = []
    for r in rows:
        a = os.path.basename(os.path.realpath(r["file"]))[:-4]
        b = f"{a.split('_')[0]}_{int(a.split('_')[1]) + 1:06d}"
        out += [(src, a), (src, b)]
    return out, len(rows)


def build(name):
    spec = ARMS[name]
    clips, nb = [], {}
    for src, n in spec.items():
        c, k = pairs(src, n, seed=0); clips += c; nb[src] = k
    if len(spec) == 1:
        src = next(iter(spec)); root = f"{BT}/{src}"
    else:                                                               # 섞는 팔: 링크 폴더
        root = f"{BT}/v3mix"
        for sub in ("clips", "labels"):
            os.makedirs(f"{root}/{sub}", exist_ok=True)
        for src, c in clips:
            for sub, ext in (("clips", "mp4"), ("labels", "npz")):
                lk = f"{root}/{sub}/{c}.{ext}"
                if not os.path.lexists(lk):
                    os.symlink(f"{BT}/{src}/{sub}/{c}.{ext}", lk)
    vis = np.stack([vis_of(f"{root}/labels/{c}.npz") for _, c in clips])
    np.savez(f"{REPO}/data_csv/blender_track_v3/arms/{name}.npz", clip=np.array([c for _, c in clips]), vis=vis, root=np.array(f"{root}/clips"),
             layout=np.array("mp4"), labels_root=np.array(f"{root}/labels"))
    return nb, len(clips)


if __name__ == "__main__":
    for n in sys.argv[1:] or list(ARMS):
        print(n, "장면 (출처별) · 영상", build(n))
