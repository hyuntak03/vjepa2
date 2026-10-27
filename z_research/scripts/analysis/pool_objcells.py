"""ar_layers 캐시를 블록마다 칸 256 개 전체 평균 (N, D) → (1, D) 로 줄인다 (2026-10-26, 학습 없음, CPU).
사용자: "(N, D) → mean pool (1, D) 해서 각 block 별로 linear probe" · "물체 칸 평균은 일단 하지 말고 기본적인 거부터".
(파일 이름은 처음 물체 칸 평균판에서 왔다. 2026-10-26 사용자 지시로 전체 평균만 남겼다.)

입력 /workspace/auto_research_cache/ar_layers/{split}/*.npz (+ {split}_meta.json). 슬롯 = 라벨 블록:
  enc    (16, 1280)        slot j = encoder 블록 j
  tfh    (14, 16, 384)     slot j = teacher forcing 위치 j 의 층 0..13 (입력 블록 j 기준; 층 0 = 입력 임베딩, 1..12 = predictor 층, 13 = 마지막 norm)
  tfout  (16, 1280)        slot j = tf_out[j − 1] (블록 j 예측, slot 0 은 NaN)
  roh    (14, 8, 384)      slot k = 굴린 예측 위치 8+k 의 층 0..13 (입력 = 블록 8+k 상태, k = 0..6; slot 7 은 NaN)
  roout  (8, 1280)         slot k = 블록 8+k 굴린 예측
라벨 (클립 × 블록 16): ok (물체 보임) · 모양 · 색 · 시나리오 번호 · 블록 안 변위 dx dy · 중심 x y.

  /venv/vjepa/bin/python z_research/scripts/analysis/pool_objcells.py --workers 24
출력: /workspace/auto_research_cache/ar_layers_pooled/{split}.npz
"""
import argparse, json, os
from multiprocessing import Pool
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=6); a = ap.parse_args()
SRC = "/workspace/auto_research_cache/ar_layers"; OUT = "/workspace/auto_research_cache/ar_layers_pooled"; os.makedirs(OUT, exist_ok=True)
NB = 16


def one(arg):
    split, i = arg
    path = f"{SRC}/{split}/{i:05d}.npz"
    z = np.load(path)
    tfo = np.full((16, 1280), np.nan, np.float16); tfo[1:] = z["tf_out"][:15].astype(np.float32).mean(1)
    roh = np.full((14, 8, 384), np.nan, np.float16); roh[:, :7] = z["ro_h"].astype(np.float32).mean(2)
    r = {"enc": z["enc"].astype(np.float32).mean(1).astype(np.float16), "tfh": z["tf_h"].astype(np.float32).mean(2).astype(np.float16),
         "tfout": tfo, "roh": roh, "roout": z["ro_out"].astype(np.float32).mean(1).astype(np.float16)}
    z.close()
    # 2026-10-26: 읽은 파일을 페이지 캐시에서 바로 내린다 (컨테이너 메모리 상한 1.16 TB 에 캐시가 붙어 전체가 멈칫했다)
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return r


def pressure():
    try:
        return open("/proc/pressure/memory").readline().split()[1]
    except Exception:
        return "?"


if __name__ == "__main__":
    for split in ("val", "train"):
        meta = json.load(open(f"{SRC}/{split}_meta.json"))
        scen_names = sorted({m["scen"] for m in meta})
        res = []
        import time as _t
        t0 = _t.time()
        with Pool(a.workers) as pool:
            for n_, r in enumerate(pool.imap(one, [(split, m["i"]) for m in meta], chunksize=2)):
                res.append(r)
                if n_ % 250 == 0:
                    print(f"{split} {n_}/{len(meta)} ({_t.time() - t0:.0f}s) 메모리 대기 {pressure()}", flush=True)
        lab = np.zeros((len(meta), NB, 8), np.float32)
        for n, m in enumerate(meta):
            for j, B in enumerate(m["blocks"]):
                lab[n, j] = [float(B["ok"] and bool(B["cells"])), m["shape"], m["color"], scen_names.index(m["scen"]), *B["disp"], *B["ctr"]]
        np.savez(f"{OUT}/{split}.npz", **{k: np.stack([r[k] for r in res]) for k in res[0]}, lab=lab, scen_names=np.array(scen_names),
                 clip=np.array([m["clip"] for m in meta]), i=np.array([m["i"] for m in meta]))
        print(split, len(meta), flush=True)
