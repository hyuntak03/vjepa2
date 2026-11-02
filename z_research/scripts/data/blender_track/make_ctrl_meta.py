#!/usr/bin/env python3
"""v3 + v3t 대조 실험 학습 목록 (2026-11-01, 사용자 요청) — make_v3t_meta.py 와 같은 꼴.

팔 (v3 ablation 과 같은 설정 — 2,000 step · 전역 batch 24 · seed 0):
  timing_vob   VOBench late 와 재등장 타이밍이 같은 장면만: 가림 시작 샘플 13–15 · 다시 보이기 전 마지막 가림 샘플 16–18
               (VOBench late_k1–3 = 샘플 [16−k, 15+k] 가림). v3 1,200 + v3t 400 장면 × A · B   ↔ 비교: v3 섞음 1,600 + trapdoor 25 % (같은 구성, 타이밍만 제한)
  vob_like     VOBench late 와 가장 닮은 조각만 (11-01 사용자: "비슷한 것만 모아서 학습하면 VOBench 가 풀리나"):
               trapdoor · 정면 카메라 · 같은 재등장 타이밍 (위 조건) 인 v3t 장면 전부 (389) × A · B    ↔ 비교: trapdoor 만 (1,528 장면) · VOBench PFT
  oracle_test  "다시 나올 자리를 외울 수 있으면" 의 상한: v3 시험 1,719 + trapdoor 시험 446 장면의 가능 영상 (A · B) 을 그대로 학습.
               시험 쌍의 문맥 · 가능 미래를 다 본 상태 — 결과는 상한으로만 읽는다 (일반화 수치 아님).
출력 data_csv/blender_track_v3/arms/<이름>.npz. 두 세트를 섞으므로 root 는 링크 폴더 blender_track/v3mix (이름 train40_ · train60_ · test41_ · test61_ 은 겹치지 않는다)
  $P make_ctrl_meta.py vob_like timing_vob oracle_test
"""
import csv
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
BT = "/data/hyuntak/project/2026/2027_cvpr/nfs_cache/blender_track"
sys.path.insert(0, os.path.dirname(__file__))
from make_v3_meta import vis_of  # noqa: E402

TRAIN = {"v3": f"{REPO}/data_csv/blender_track_v3/train_arms.csv", "v3t": f"{REPO}/data_csv/blender_track_v3t/train_arms.csv"}
TEST = {"v3": f"{REPO}/data_csv/blender_track_v3/voe_test/index.csv", "v3t": f"{REPO}/data_csv/blender_track_v3t/voe_test/index.csv"}


def clip_name(f):
    return os.path.basename(os.path.realpath(f))[:-4]


def timing_vob(seed=0):
    out, nb = [], {}
    for src, n in (("v3", 1200), ("v3t", 400)):
        rows = [r for r in csv.DictReader(open(TRAIN[src])) if r["variant"] == "pos_a" and r["render_timing_ok"] == "1"
                and 13 <= int(r["event_s"]) <= 15 and 16 <= int(r["event_e"]) <= 18]
        rng = np.random.default_rng(seed)
        pool = len(rows)
        rows = [rows[i] for i in sorted(rng.choice(len(rows), min(n, len(rows)), replace=False))]
        for r in rows:
            a = clip_name(r["file"]); b = f"{a.split('_')[0]}_{int(a.split('_')[1]) + 1:06d}"
            out += [(src, a), (src, b)]
        nb[src] = (len(rows), pool)
    return out, nb


def vob_like():
    rows = [r for r in csv.DictReader(open(TRAIN["v3t"])) if r["variant"] == "pos_a" and r["render_timing_ok"] == "1"
            and r["camera"] == "front" and 13 <= int(r["event_s"]) <= 15 and 16 <= int(r["event_e"]) <= 18]
    out = []
    for r in rows:
        a = clip_name(r["file"]); b = f"{a.split('_')[0]}_{int(a.split('_')[1]) + 1:06d}"
        out += [("v3t", a), ("v3t", b)]
    return out, {"v3t": len(rows)}


def oracle_test():
    out, nb = [], {}
    for src in ("v3", "v3t"):
        rows = [r for r in csv.DictReader(open(TEST[src])) if r["variant"] in ("pos_a", "pos_b")]
        out += [(src, clip_name(r["file"])) for r in rows]
        nb[src] = sum(r["variant"] == "pos_a" for r in rows)
    return out, nb


def build(name):
    clips, nb = {"timing_vob": timing_vob, "vob_like": vob_like, "oracle_test": oracle_test}[name]()
    root = f"{BT}/v3mix"
    for sub in ("clips", "labels"):
        os.makedirs(f"{root}/{sub}", exist_ok=True)
    for src, c in clips:
        for sub, ext in (("clips", "mp4"), ("labels", "npz")):
            lk = f"{root}/{sub}/{c}.{ext}"
            if not os.path.lexists(lk):
                os.symlink(f"{BT}/{src}/{sub}/{c}.{ext}", lk)
    vis = np.stack([vis_of(f"{root}/labels/{c}.npz") for _, c in clips])
    np.savez(f"{REPO}/data_csv/blender_track_v3/arms/{name}.npz", clip=np.array([c for _, c in clips]), vis=vis,
             root=np.array(f"{root}/clips"), layout=np.array("mp4"), labels_root=np.array(f"{root}/labels"))
    return nb, len(clips)


if __name__ == "__main__":
    for n in sys.argv[1:] or ["timing_vob", "oracle_test"]:
        print(n, "장면 (출처별) · 영상", build(n))
