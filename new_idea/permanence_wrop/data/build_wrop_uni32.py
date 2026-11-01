#!/usr/bin/env python3
"""WROP (object_permanence_320) → 32 장 클립 (문맥 16 + 미래 16) — permanence_wrop 실험 A·B·C·D 용 (2026-10-31).

  python new_idea/permanence_wrop/data/build_wrop_uni32.py --workers 48            # 전부
  python new_idea/permanence_wrop/data/build_wrop_uni32.py --limit-per-task 2 --out /tmp/x --csv-dir /tmp/x/csv   # 시험

입력: data_csv/object_permanence_320/{train,heldout}_manifest.tsv
      (z_training/data/build_object_permanence_320.py 산출 — WROP input + target 을 이어 붙이고 패딩을 잘라 낸 24 fps · 568×320 mp4,
       boundary_frame = 첫 target 프레임)
규칙 (IntPhys2 violation-uni 채점 · intphys2_vuni_release_postft 학습과 같은 규칙):
  문맥 = [0, boundary) 에서 균일 16 장, 미래 = [boundary, 끝) 에서 균일 16 장 → 정확히 32 장 mp4 (568×320 그대로, 자르기는 로더 몫)
  81 % 는 60 + 60 장이라 문맥 · 미래 모두 3.9 장 간격 (≈ 6.1 fps; 경계 59→60 만 1 장 간격). 일부 과제는 input 이 25–57 장이라 문맥 간격이 더 촘촘하다.
고르는 표본:
  train   : 계열 op1 · op2 (가림) 은 과제당 1,450 전부, s1 · s2 · s3 (비영속) 는 과제당 앞 --n-nonop (기본 1,330) → 두 묶음 클립 수를 맞춘다.
            op3 (container) 는 이번 C 에서 빼 둔 계열이라 만들지 않는다 (--with-op3 로 만든다).
  heldout : 150 과제 × 100 전부 (같은 생성기 · 다른 표본, 학습과 서로소)
출력: /workspace/dataset/training_data/wrop_uni32/{train,heldout}/<task>/<sample>.mp4
      data_csv/wrop_uni32/{train_op12,train_nonop,train_op3,heldout}.csv  ("<mp4> 0", video_csv 형식)
      data_csv/wrop_uni32/{train,heldout}_manifest.tsv  (path · task · family · 원본 프레임 번호 32 개)
재실행 안전 — 이미 있는 출력은 건너뛴다. 실패는 <out>/_failed.txt
"""
import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../.."))
ap = argparse.ArgumentParser()
ap.add_argument("--manifest-dir", default=os.path.join(REPO, "data_csv/object_permanence_320"))
ap.add_argument("--family-tsv", default=os.path.join(HERE, "wrop_task_family.tsv"))
ap.add_argument("--out", default="/workspace/dataset/training_data/wrop_uni32")
ap.add_argument("--csv-dir", default=os.path.join(REPO, "data_csv/wrop_uni32"))
ap.add_argument("--n-nonop", type=int, default=1330)
ap.add_argument("--with-op3", action="store_true")
ap.add_argument("--limit-per-task", type=int, default=0)
ap.add_argument("--workers", type=int, default=48)
a = ap.parse_args()

FAM = {r["task"]: r["family"] for r in csv.DictReader(open(a.family_tsv), delimiter="\t")}


def uni_idx(b: int, n: int):
    ctx = np.round(np.linspace(0, b - 1, 16)).astype(int)
    fut = np.round(np.linspace(b, n - 1, 16)).astype(int)
    return np.concatenate([ctx, fut]).tolist()


def encode(job):
    split, src, task, sample, idx = job
    out = os.path.join(a.out, split, task, sample + ".mp4")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return job, out, None
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp.mp4"
    sel = "+".join(f"eq(n\\,{i})" for i in idx)
    cmd = ["nice", "-n", "5", "ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-threads", "1", "-i", src,
           "-vf", f"select='{sel}',setpts=N/(6*TB)", "-fps_mode", "passthrough",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-g", "32", "-pix_fmt", "yuv420p", "-an", tmp]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(tmp):
        if os.path.exists(tmp):
            os.remove(tmp)
        return job, None, r.stderr.strip()[-300:]
    os.replace(tmp, out)
    return job, out, None


def main():
    jobs = []
    for split in ("train", "heldout"):
        rows = list(csv.DictReader(open(os.path.join(a.manifest_dir, f"{split}_manifest.tsv")), delimiter="\t"))
        per_task = {}
        for r in rows:
            per_task.setdefault(r["task"], []).append(r)
        for task, rs in sorted(per_task.items()):
            fam = FAM[task]
            rs = sorted(rs, key=lambda r: r["sample"])
            if split == "train":
                if fam == "op3_container" and not a.with_op3:
                    continue
                if fam.startswith("s"):
                    rs = rs[: a.n_nonop]
            if a.limit_per_task:
                rs = rs[: a.limit_per_task]
            for r in rs:
                b, n = int(r["boundary_frame"]), int(r["n_frames"])
                if b < 16 or n - b < 16:
                    continue
                jobs.append((split, r["path"], task, r["sample"], uni_idx(b, n)))
    print(f"작업 {len(jobs):,} → {a.out}", flush=True)
    done = {"train": [], "heldout": []}
    fails = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(encode, j) for j in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            job, out, err = f.result()
            if out is None:
                fails.append(f"{job[0]}\t{job[2]}\t{job[3]}\t{err}")
            else:
                done[job[0]].append((out, job[2], FAM[job[2]], job[4]))
            if k % 10000 == 0 or k == len(futs):
                print(f"  {k:,}/{len(futs):,} (실패 {len(fails)})", flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "_failed.txt"), "w") as g:
        g.write("\n".join(fails) + ("\n" if fails else ""))
    os.makedirs(a.csv_dir, exist_ok=True)
    groups = {"train_op12": lambda f: f in ("op1_baillargeonian_occlusion", "op2_static_occlusion"),
              "train_nonop": lambda f: f.startswith("s"),
              "train_op3": lambda f: f == "op3_container"}
    for split, rs in done.items():
        rs.sort()
        with open(os.path.join(a.csv_dir, f"{split}_manifest.tsv"), "w") as g:
            g.write("path\ttask\tfamily\tsrc_frames\n")
            g.writelines(f"{p}\t{t}\t{fm}\t{','.join(map(str, ix))}\n" for p, t, fm, ix in rs)
        if split == "heldout":
            with open(os.path.join(a.csv_dir, "heldout.csv"), "w") as g:
                g.writelines(f"{p} 0\n" for p, *_ in rs)
            print(f"heldout: {len(rs):,}")
        else:
            for name, keep in groups.items():
                sel = [p for p, t, fm, ix in rs if keep(fm)]
                if sel:
                    with open(os.path.join(a.csv_dir, f"{name}.csv"), "w") as g:
                        g.writelines(f"{p} 0\n" for p in sel)
                print(f"{name}: {len(sel):,}")
    print(f"실패 {len(fails)} → {a.out}/_failed.txt")


if __name__ == "__main__":
    sys.exit(main())
