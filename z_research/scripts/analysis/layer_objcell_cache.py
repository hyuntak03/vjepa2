"""블록 AR 층별 은닉 (384) 의 물체 칸 평균 — teacher forcing · 굴린 예측, 자리 8–14 (예측 걸음 1–7) (2026-10-26, CPU).
사용자: "물체 칸 concept vector 로 지금 결과들 다 바꿔줘" · "figure 전부 갱신". predictor_info_by_layer 의 모양 · 색을 물체 칸 기준으로 다시 하기 위한 캐시.
원본: ar_layers 전체 토큰 캐시 tf_h (14, 16, 256, 384) · ro_h (14, 7, 256, 384). 층 = 입력 · 1–12 · 마지막 정규화.
자리 p = 7 + k (k = 1–7): teacher forcing 자리 p (입력 = 실제 encoder 블록 p) · 굴림 자리 p (입력 = 블록 p 를 예측한 상태). 물체 칸 = meta 블록 p 의 cells (블록 p 에 물체 보일 때, 아니면 NaN).
  /venv/vjepa/bin/python z_research/scripts/analysis/layer_objcell_cache.py --n_train 4000
출력: /workspace/auto_research_cache/objcell_concept/layers_{split}.npz — tf (N, 7, 14, 384) · ro (N, 7, 14, 384) fp16 · i
"""
import argparse, json, os
from multiprocessing import Pool
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--n_train", type=int, default=4000); a = ap.parse_args()
SRC = "/workspace/auto_research_cache/ar_layers"; OUT = "/workspace/auto_research_cache/objcell_concept"


def one(arg):
    split, m = arg
    path = f"{SRC}/{split}/{m['i']:05d}.npz"; z = np.load(path); TF, RO = z["tf_h"], z["ro_h"]; z.close()
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    tf = np.full((7, 14, 384), np.nan, np.float32); ro = np.full((7, 14, 384), np.nan, np.float32)
    for j in range(7):
        bl = m["blocks"][8 + j]
        if bl["ok"]:
            c = bl["cells"] or [int(min(15, max(0, bl["ctr"][1] // 16)) * 16 + min(15, max(0, bl["ctr"][0] // 16)))]
            tf[j] = TF[:, 8 + j][:, c].astype(np.float32).mean(1); ro[j] = RO[:, j][:, c].astype(np.float32).mean(1)
    return tf.astype(np.float16), ro.astype(np.float16), m["i"]


for split, n in (("val", 0), ("train", a.n_train)):
    meta = json.load(open(f"{SRC}/{split}_meta.json"))[: n or None]
    with Pool(6) as pool:
        R = list(pool.imap(one, [(split, m) for m in meta], chunksize=4))
    np.savez(f"{OUT}/layers_{split}.npz", tf=np.stack([r[0] for r in R]), ro=np.stack([r[1] for r in R]), i=np.array([r[2] for r in R])); print(split, len(R), flush=True)
