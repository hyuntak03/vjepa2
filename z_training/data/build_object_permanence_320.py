#!/usr/bin/env python3
"""Object Permanence (WROP, Blender 150 과제 × 1 만) → 짧은 변 320 · 한 파일 · 학습/held-out 부분집합 (2026-10-15, 사용자 결정).

  python z_training/data/build_object_permanence_320.py --n-train 1450 --n-heldout 100 --workers 200          # 전체
  python z_training/data/build_object_permanence_320.py --n-train 2 --n-heldout 1 --tasks 3 --out /tmp/x     # 시험

원본: ${TRAIN_DATA_ROOT}/object_permanence/train/shard*/<task>_task/<sample>/{input_video,target_video}.mp4 (1280×720, 24 fps, 각 60 장)
출력: ${TRAIN_DATA_ROOT}/object_permanence_320/{train,heldout}/<task>/<sample>.mp4     (원본 불변)
      data_csv/object_permanence_320/{train,heldout}.csv   "<abs mp4> 0"  (build_fps4_table.py 입력 형식)
      data_csv/object_permanence_320/{train,heldout}_manifest.tsv   task · sample · 프레임 수 · 잘라 낸 패딩 ·
        input_frames · target_frames · boundary_frame (= input_frames, 이어 붙인 영상의 첫 target 프레임, 0-based, 24 fps 원본 프레임)

하는 일
  * input + target 을 **한 영상으로 이어 붙인다** (input 끝 = target 시작, 원본 렌더 프레임이 연속이다)
  * **패딩 프레임은 잘라 낸다** — metadata.video_split 의 input_pad_start_frames (input 앞) · target_pad_end_frames (target 뒤).
    이벤트 경계로 자른 샘플 (~23 %) 은 끝/앞을 같은 프레임 반복으로 채웠다. 반복 = 정지 화면 = 복사 정답이라 AR 에 나쁘다.
    그래서 그런 샘플은 120 장보다 짧다 (예: 100 장)
  * 짧은 변 320, 비율 유지 (1280×720 → 568×320), x264 veryfast crf 23, keyint 24 (resize_videos.sh 와 같은 설정), 24 fps 유지
  * 과제마다 표본을 seed 로 섞어 앞 n_train 개 = train, 그다음 n_heldout 개 = heldout (서로소, 같은 과제 안 held-out = 분포 안 일반화).
    과제 단위 held-out 은 나중에 train.csv 에서 과제를 빼는 것으로 한다 (다시 인코딩할 필요 없음)
  * 재실행 안전 — 이미 있는 출력은 건너뛴다. 실패는 <out>/_failed.txt
규모 (기본 1,450 / 100): train 217,500 (SSv2 train 218,620 과 비슷) · heldout 15,000.
"""
import argparse
import glob
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../z_research/scripts/harness"))
from paths import ROOT, expand  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=expand("${TRAIN_DATA_ROOT}/object_permanence/train"))
ap.add_argument("--out", default=expand("${TRAIN_DATA_ROOT}/object_permanence_320"))
ap.add_argument("--csv-dir", default=os.path.join(ROOT, "data_csv/object_permanence_320"))
ap.add_argument("--n-train", type=int, default=1450)
ap.add_argument("--n-heldout", type=int, default=100)
ap.add_argument("--tasks", type=int, default=0, help="앞 N 과제만 (시험용, 0 = 전부)")
ap.add_argument("--short", type=int, default=320)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--workers", type=int, default=200)
a = ap.parse_args()


def encode(job):
    split, task, sample, sdir = job
    out = os.path.join(a.out, split, task, sample + ".mp4")
    m = json.load(open(os.path.join(sdir, "metadata.json")))["video_split"]
    ps = int(m.get("input_pad_start_frames") or 0)
    pe = int(m.get("target_pad_end_frames") or 0)
    n_in, n_tg = 60 - ps, 60 - pe
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return split, task, sample, out, n_in + n_tg, ps, pe, None
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp.mp4"
    fc = (f"[0:v]trim=start_frame={ps},setpts=PTS-STARTPTS[a];"
          f"[1:v]trim=end_frame={n_tg},setpts=PTS-STARTPTS[b];"
          f"[a][b]concat=n=2:v=1:a=0,scale=w='if(gt(iw,ih),-2,{a.short})':h='if(gt(iw,ih),{a.short},-2)'[v]")
    cmd = ["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-threads", "1",
           "-i", os.path.join(sdir, "input_video.mp4"), "-i", os.path.join(sdir, "target_video.mp4"),
           "-filter_complex", fc, "-map", "[v]", "-r", "24", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-g", "24", "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", tmp]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(tmp):
        if os.path.exists(tmp):
            os.remove(tmp)
        return split, task, sample, None, 0, ps, pe, r.stderr.strip()[-300:]
    os.replace(tmp, out)
    return split, task, sample, out, n_in + n_tg, ps, pe, None


def main():
    tdirs = sorted(glob.glob(os.path.join(a.src, "shard*", "*_task")))
    if a.tasks:
        tdirs = tdirs[: a.tasks]
    jobs = []
    for td in tdirs:
        task = os.path.basename(td)[: -len("_task")]
        samples = sorted(os.listdir(td))
        random.Random(f"{a.seed}-{task}").shuffle(samples)
        if len(samples) < a.n_train + a.n_heldout:
            raise SystemExit(f"{task}: 표본 {len(samples)} < {a.n_train + a.n_heldout}")
        for i, s in enumerate(samples[: a.n_train + a.n_heldout]):
            jobs.append(("train" if i < a.n_train else "heldout", task, s, os.path.join(td, s)))
    print(f"과제 {len(tdirs)} · 작업 {len(jobs):,} (train {a.n_train}/과제, heldout {a.n_heldout}/과제) → {a.out}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    rows = {"train": [], "heldout": []}
    fails = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(encode, j) for j in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            split, task, sample, out, nf, ps, pe, err = f.result()
            if out is None:
                fails.append(f"{split}\t{task}\t{sample}\t{err}")
            else:
                rows[split].append((out, task, sample, nf, ps, pe))
            if k % 5000 == 0 or k == len(futs):
                print(f"  {k:,}/{len(futs):,} (실패 {len(fails)})", flush=True)
    with open(os.path.join(a.out, "_failed.txt"), "w") as g:
        g.write("\n".join(fails) + ("\n" if fails else ""))
    os.makedirs(a.csv_dir, exist_ok=True)
    for split, rs in rows.items():
        rs.sort()
        with open(os.path.join(a.csv_dir, f"{split}.csv"), "w") as g:
            g.writelines(f"{r[0]} 0\n" for r in rs)
        with open(os.path.join(a.csv_dir, f"{split}_manifest.tsv"), "w") as g:
            g.write("path\ttask\tsample\tn_frames\tpad_start_cut\tpad_end_cut\tinput_frames\ttarget_frames\tboundary_frame\n")
            g.writelines("\t".join(map(str, r + (60 - r[4], 60 - r[5], 60 - r[4]))) + "\n" for r in rs)
        n_pad = sum(1 for r in rs if r[4] or r[5])
        print(f"{split}: {len(rs):,} 편 (패딩 잘라 낸 것 {n_pad:,}) → {a.csv_dir}/{split}.csv")
    print(f"실패 {len(fails)} → {a.out}/_failed.txt")


if __name__ == "__main__":
    main()
