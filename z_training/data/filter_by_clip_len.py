"""video_csv 인덱스를 **실제 클립 길이**로 거른다 (2026-10-10). VideoDataset 의 규칙 그대로:
    fps 모드:  fstp = ceil(avg_fps) // fps,  필요 원본 프레임 = n_frames × fstp  (fstp == 0 이면 원 fps 그대로 = n_frames)
    python z_training/data/filter_by_clip_len.py --csv data_csv/k400_320/train_min96.csv --fps 4 --n-frames 32 -o data_csv/k400_320/train_fps4_f32.csv
런타임 재추첨 (filter_short_videos) 을 0 으로 만들려는 것이다 — 재추첨이 많으면 rank 하나가 배치를 못 채워 DDP 가 멈춘다 (datasets.md).
"""
import argparse, math, os, sys
from concurrent.futures import ProcessPoolExecutor

def probe(path):
    try:
        import decord
        vr = decord.VideoReader(path, num_threads=1)
        return path, len(vr), float(vr.get_avg_fps())
    except Exception:
        return path, -1, 0.0

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True); ap.add_argument("-o", required=True)
ap.add_argument("--fps", type=int, default=None); ap.add_argument("--frame-step", type=int, default=None)
ap.add_argument("--n-frames", type=int, required=True); ap.add_argument("--workers", type=int, default=24)
a = ap.parse_args()
rows = [l.rstrip("\n") for l in open(a.csv) if l.strip()]
paths = [r.rsplit(" ", 1)[0] for r in rows]          # 마지막 칸 = label. K400 클래스 폴더 이름에 공백이 있다 ("spinning poi")
keep, bad, short = [], 0, 0
with ProcessPoolExecutor(a.workers) as ex:
    for (p, n, fps), r in zip(ex.map(probe, paths, chunksize=64), rows):
        if n < 0:
            bad += 1; continue
        if a.frame_step:
            need = a.n_frames * a.frame_step
        else:
            fstp = math.ceil(fps) // a.fps if fps > 0 else 0
            need = a.n_frames * max(fstp, 1)
        if n >= need:
            keep.append(r)
        else:
            short += 1
os.makedirs(os.path.dirname(os.path.abspath(a.o)), exist_ok=True)
open(a.o, "w").write("\n".join(keep) + "\n")
print(f"{a.csv}: {len(rows)} -> keep {len(keep)} ({100*len(keep)/max(len(rows),1):.1f} %), short {short}, unreadable {bad} -> {a.o}")
