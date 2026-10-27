"""층별 attentive probe 용 — 블록 AR predictor 은닉 토큰 (256, 384) 을 층 14 개 × 자리 8–14 로 (2026-10-26 저녁, CPU).
사용자: "(층별 그림) attentive probing 으로 진행해줘. shape × color 로 할 수 있잖아".
원본: ar_layers 전체 토큰 캐시 tf_h (14, 16, 256, 384) · ro_h (14, 7, 256, 384). 층 = 입력 · 1–12 · 마지막 정규화.
표본: 자리 p = 8–14 (예측 걸음 k = p − 7) 중 블록 p 에 물체 보일 때. teacher forcing = 자리 p (입력 = 실제 encoder 블록 p), 굴린 예측 = 굴림 자리 p (입력 = 블록 p 를 예측한 상태).
라벨: 모양 · 색 · dx · dy · x · y (블록 p) · 클립 번호 · 걸음.
  /venv/vjepa/bin/python z_research/scripts/analysis/attnprobe384_extract.py --n_train 4000
출력: /workspace/auto_research_cache/attnprobe384/{split}_{tf,ro}.npy (14, N, 256, 384) fp16 · {split}_lab.npy (N, 8)
"""
import argparse, json, os
from multiprocessing import Pool
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--n_train", type=int, default=4000); a = ap.parse_args()
SRC = "/workspace/auto_research_cache/ar_layers"; OUT = "/workspace/auto_research_cache/attnprobe384"; os.makedirs(OUT, exist_ok=True)


def one(arg):
    split, m, rows = arg
    if not rows:
        return
    path = f"{SRC}/{split}/{m['i']:05d}.npz"; z = np.load(path); TF, RO = z["tf_h"], z["ro_h"]; z.close()
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    mt = np.load(f"{OUT}/{split}_tf.npy", mmap_mode="r+"); mr = np.load(f"{OUT}/{split}_ro.npy", mmap_mode="r+")
    for n, p in rows:
        mt[:, n] = TF[:, p]; mr[:, n] = RO[:, p - 8]
    mt.flush(); mr.flush(); del mt, mr


for split, n in (("val", 0), ("train", a.n_train)):
    meta = json.load(open(f"{SRC}/{split}_meta.json"))[: n or None]
    jobs, lab, N = [], [], 0
    for m in meta:
        rows = []
        for p in range(8, 15):
            bl = m["blocks"][p]
            if bl["ok"]:
                rows.append((N, p)); lab.append([m["shape"], m["color"], bl["disp"][0], bl["disp"][1], bl["ctr"][0], bl["ctr"][1], m["i"], p - 7]); N += 1
        jobs.append((split, m, rows))
    for nm in ("tf", "ro"):
        np.lib.format.open_memmap(f"{OUT}/{split}_{nm}.npy", mode="w+", dtype=np.float16, shape=(14, N, 256, 384)).flush()
    np.save(f"{OUT}/{split}_lab.npy", np.array(lab, np.float32))
    with Pool(6) as pool:
        list(pool.imap(one, jobs, chunksize=4))
    print(split, "표본", N, flush=True)
