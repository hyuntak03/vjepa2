"""attentive probe 용 토큰 추출 — encoder 블록 토큰 (256, 1280) · AR 굴린 예측 출력 토큰 (256, 1280) + 라벨 + 구역별 평균 (2026-10-26, CPU).
사용자: "probe 를 attentive probe 로. color × shape 한 번에, velocity, position 까지 다 읽을 수 있도록". 개념 벡터 재검토용 구역 평균도 같이 저장.
원본: ar_layers 전체 토큰 캐시 (enc = 블록 단위 LN(target_encoder), ro_out = 블록 AR natural_ar_vw_state 굴린 예측 출력, 블록 8–15).
표본: 물체 보이는 블록 (보이는 비율 ≥ 0.5 · 화면 안 ≥ 0.9). train = 앞 N 편 (기본 4,000), val = 350 편 전부.
  enc  : 블록 0–15 · ro : 블록 8–15 (그 블록을 예측한 굴린 출력)
라벨 (표본마다): 모양 · 색 · dx · dy · x · y · 클립 번호 · 블록 · 물체 칸 수.
구역 평균 (enc 만): 물체 칸 · 1 칸 옆 · 2 칸 옆 · 3 칸 이상 · 화면 전체 (칸 거리 = 물체 칸까지 체스판 거리).
  /venv/vjepa/bin/python z_research/scripts/analysis/attnprobe_extract.py --n_train 4000
출력: /workspace/auto_research_cache/attnprobe/{split}_{enc,ro}_tok.npy (fp16) · {split}_{enc,ro}_lab.npy · {split}_enc_regions.npy
"""
import argparse, json, os
from multiprocessing import Pool
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--n_train", type=int, default=4000); a = ap.parse_args()
SRC = "/workspace/auto_research_cache/ar_layers"; OUT = "/workspace/auto_research_cache/attnprobe"; os.makedirs(OUT, exist_ok=True)
gy, gx = np.divmod(np.arange(256), 16)


def one(arg):
    split, m = arg
    path = f"{SRC}/{split}/{m['i']:05d}.npz"
    z = np.load(path); E = z["enc"]; RO = z["ro_out"]; z.close()
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    enc, ro = [], []
    for b, bl in enumerate(m["blocks"]):
        if not bl["ok"]:
            continue
        cells = bl["cells"] or [int(min(15, max(0, bl["ctr"][1] // 16)) * 16 + min(15, max(0, bl["ctr"][0] // 16)))]
        lab = np.array([m["shape"], m["color"], bl["disp"][0], bl["disp"][1], bl["ctr"][0], bl["ctr"][1], m["i"], b, len(cells)], np.float32)
        oy, ox = np.divmod(np.array(cells), 16)
        d = np.max(np.abs(gy[:, None] - oy[None]), 0) if False else np.min(np.maximum(np.abs(gy[:, None] - oy[None]), np.abs(gx[:, None] - ox[None])), 1)
        Ef = E[b].astype(np.float32)
        reg = np.stack([Ef[d == 0].mean(0), Ef[d == 1].mean(0) if (d == 1).any() else np.full(1280, np.nan, np.float32),
                        Ef[d == 2].mean(0) if (d == 2).any() else np.full(1280, np.nan, np.float32), Ef[d >= 3].mean(0), Ef.mean(0)]).astype(np.float16)
        enc.append((E[b], lab, reg))
        if b >= 8:
            ro.append((RO[b - 8], lab))
    return enc, ro


for split, n in (("val", 0), ("train", a.n_train)):
    meta = json.load(open(f"{SRC}/{split}_meta.json"))[: n or None]
    enc, ro = [], []
    with Pool(6) as pool:
        for e_, r_ in pool.imap(one, [(split, m) for m in meta], chunksize=4):
            enc += e_; ro += r_
    np.save(f"{OUT}/{split}_enc_tok.npy", np.stack([e[0] for e in enc])); np.save(f"{OUT}/{split}_enc_lab.npy", np.stack([e[1] for e in enc]))
    np.save(f"{OUT}/{split}_enc_regions.npy", np.stack([e[2] for e in enc]))
    np.save(f"{OUT}/{split}_ro_tok.npy", np.stack([r[0] for r in ro])); np.save(f"{OUT}/{split}_ro_lab.npy", np.stack([r[1] for r in ro]))
    print(split, "enc 표본", len(enc), "ro 표본", len(ro), flush=True)
