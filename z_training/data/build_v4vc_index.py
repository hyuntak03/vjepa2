#!/usr/bin/env python3
"""BlenderTrack v4_velocity_control 학습 목록 — main 의 compose_v4vc.py 규칙 그대로 다시 만든다 (2026-11-02, vast).

왜: main 의 "V-JEPA 2 └ V4 Post Training" (릴리즈 predictor + v4 평면 + 빗면 4,000 step) 과 같은 데이터로
    AR 기본판 (natural_ar_vw_state) 을 이어 학습하려고. 학습 목록 npz 는 gitignore 라 tar 에 없다.

규칙 (main z_research/scripts/data/blender_track/compose_v4vc.py main() 과 같다):
  평면 (config.yaml, 분할 train_fam · train_ind):
    fam_ok = 모든 단계가 렌더됐고 (빈 장면 B 제외) 렌더 타이밍을 통과한 묶음
    ind_A  = 대조 분할의 단계 영상 (렌더 · 타이밍 통과) · ind_B = 대조 분할의 빈 장면
    nA = min(K × |fam_ok|, |ind_A|) · nB = min(|fam_ok|, |ind_B|) · fam_keep = fam_ok[: nA // K]
    목록 = fam_keep 의 단계 영상 + fam_keep[:nB] 의 빈 장면
  빗면 (config_v4vcr.yaml, 분할 rtrain_fam, 대조 없음): fam_ok 의 영상 전부 (빈 장면 포함)
  all = 평면 + 빗면 (main 의 v4vc_all_s0 이 쓴 목록, 보고서: 평면 1,137 + 빗면 1,113 묶음 = 영상 11,250)

출력 (--out, 기본 data_csv/):
  blender_track_v4vc/arms/fam.npz · blender_track_v4vcr/arms/fam.npz · blender_track_v4vc/arms/all.npz  (main blender_track 로더 형식)
  blender_track_v4vc/arms/all_mp4list.csv  — 한 줄에 mp4 절대경로 하나 (vast 학습기 data.datasets[].type=mp4_list)
  blender_track_v4vc/arms/all_summary.json — 묶음 · 영상 수, 단계별 수

  /venv/vjepa/bin/python z_training/data/build_v4vc_index.py \
      --root /workspace/dataset/training_data/v4_velocity_control \
      --gen-dir <main 의 z_research/scripts/data/blender_track>  [--workers 32]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
C = None   # compose_v4vc 모듈 (worker 안에서 import)


def _init(gen_dir, cfg_path, root):
    global C
    sys.path.insert(0, gen_dir)
    import compose_v4vc as comp  # noqa: E402
    C = comp
    cfg = C.G.load_cfg(cfg_path, None)
    cfg["out"] = root                     # 설정의 out 은 클러스터 경로 → 풀어 둔 자리
    C.CFG["cfg"] = cfg


def _check(arg):
    return C.check_family(arg)


def families(gen_dir, cfg_path, root, split, workers):
    sys.path.insert(0, gen_dir)
    import compose_v4vc as comp  # noqa: E402
    cfg = comp.G.load_cfg(cfg_path, None)
    nc = int(cfg["splits"][split]["candidates"])
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(gen_dir, cfg_path, root)) as ex:
        fams = [r for r in ex.map(_check, [(split, f) for f in range(nc)], chunksize=8) if r is not None]
    return fams, int(cfg["velocity"]["levels"])


def ok_fam(r):
    return all(s["rendered"] and (s["level"] == "B" or s["timing_ok"]) for s in r["slots"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="풀어 둔 v4_velocity_control 폴더 (clips/ labels/ design/ …)")
    ap.add_argument("--gen-dir", required=True, help="main 의 blender_track 생성기 폴더 (compose_v4vc.py 가 있는 곳)")
    ap.add_argument("--out", default=f"{REPO}/data_csv")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    sys.path.insert(0, a.gen_dir)
    from make_v3_meta import vis_of  # noqa: E402

    # 평면: 묶음 분할 + 대조 분할 (대조 팔은 안 쓰지만 묶음 목록의 길이를 대조와 맞춘 규칙을 그대로 재현한다)
    fl, K = families(a.gen_dir, f"{root}/config.yaml", root, "train_fam", a.workers)
    ind, _ = families(a.gen_dir, f"{root}/config.yaml", root, "train_ind", a.workers)
    fam_ok = [r for r in fl if ok_fam(r)]
    ind_A = [s for r in ind for s in r["slots"] if s["level"] != "B" and s["rendered"] and s["timing_ok"]]
    ind_B = [s for r in ind for s in r["slots"] if s["level"] == "B" and s["rendered"]]
    nA = min(K * len(fam_ok), len(ind_A)); nB = min(len(fam_ok), len(ind_B))
    keep = fam_ok[: nA // K]
    flat = [s["cid"] for r in keep for s in r["slots"] if s["level"] != "B"] + \
           [s["cid"] for r in keep[:nB] for s in r["slots"] if s["level"] == "B"]
    # 빗면: 묶음 분할만
    rp, _ = families(a.gen_dir, f"{root}/config_v4vcr.yaml", root, "rtrain_fam", a.workers)
    rp_ok = [r for r in rp if ok_fam(r)]
    ramp = [s["cid"] for r in rp_ok for s in r["slots"]]

    lv = lambda fams: np.bincount([int(s["level"]) for r in fams for s in r["slots"] if s["level"] != "B"], minlength=K).tolist()
    summ = {"root": root, "K": K,
            "flat": {"families_ok": len(fam_ok), "families_kept": len(keep), "videos": len(flat), "nA": nA, "nB": nB,
                     "ind_A": len(ind_A), "ind_B": len(ind_B), "per_level": lv(keep)},
            "ramp": {"families_ok": len(rp_ok), "families_candidates_selected": len(rp), "videos": len(ramp), "per_level": lv(rp_ok)},
            "all_videos": len(flat) + len(ramp)}
    clips_dir, lab_dir = f"{root}/clips", f"{root}/labels"
    for c in flat + ramp:
        if not os.path.isfile(f"{clips_dir}/{c}.mp4"):
            raise FileNotFoundError(f"{clips_dir}/{c}.mp4")

    def save(path, clips):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        vis = np.stack([vis_of(f"{lab_dir}/{c}.npz") for c in clips])
        np.savez(path, clip=np.array(clips), vis=vis, root=np.array(clips_dir), layout=np.array("mp4"), labels_root=np.array(lab_dir))

    save(f"{a.out}/blender_track_v4vc/arms/fam.npz", flat)
    save(f"{a.out}/blender_track_v4vcr/arms/fam.npz", ramp)
    save(f"{a.out}/blender_track_v4vc/arms/all.npz", flat + ramp)
    with open(f"{a.out}/blender_track_v4vc/arms/all_mp4list.csv", "w") as f:
        f.writelines(f"{clips_dir}/{c}.mp4\n" for c in flat + ramp)
    json.dump(summ, open(f"{a.out}/blender_track_v4vc/arms/all_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
