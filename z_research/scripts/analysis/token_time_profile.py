#!/usr/bin/env python3
"""시간칸(tubelet) 별로 **정답 쪽 신호와 반대 신호**를 전수 집계한다.

토큰마다  d = |p - h(imp)| - |p - h(pos)|.  d>0 이면 그 패치가 정답 쪽으로 민 것.
채점은 `mean_token d > 0` 이므로, 시간칸마다 그 합의 내역을 보면
**언제 벌고 언제 까먹는지**가 나온다.

시간칸 8개 = 미래 16프레임 / tubelet 2.  t_j 는 raw 48+6j, 51+6j 를 덮는다.
칸마다 공간 토큰 256개(16x16).

  per pair, per t:
    net      = mean_token d                     (합쳐서 8칸 평균이 곧 채점 margin)
    pos      = mean over d>0 인 토큰의 d        (버는 쪽 크기)
    neg      = mean over d<0 인 토큰의 -d       (까먹는 쪽 크기)
    frac_pos = d>0 인 토큰 비율                 (몇 칸이 정답 쪽인가)

⚠️ 캐시는 fp16 이라 재계산 오차가 |0.0005| 쯤 된다. **개별 pair 의 net 은 못 믿는다**
   (colour margin 0.0016 과 같은 자릿수). 여기서는 수천 pair 를 평균하므로 집계는 쓸 수 있다.
   correct/wrong 라벨은 fp32 채점(`per_block.json`)에서 가져온다.

  python z_research/scripts/analysis/token_time_profile.py --limit 400      # 빠른 확인
  python z_research/scripts/analysis/token_time_profile.py                  # 전수
"""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import numpy as np

CACHE = Path("/local_datasets/world/world_analysis/cache/v11_full_vith")
IDX = Path("data_csv/intphysgen_v11_full/index_probe.csv")
EXP = Path("z_research/IntPhysGenV11_occlusion_timing_ablation/exp_results")
OUT = EXP / "token_time_profile.npz"


def mot(c): return "static" if c.startswith("static") else ("flat" if "flat" in c else "ramp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="pair 수 상한 (0 = 전수)")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()

    meta = {r["video_id"]: r for r in csv.DictReader(IDX.open())}
    ix = {v: i for i, v in enumerate(json.loads((CACHE / "meta.json").read_text())["video_ids"])}
    S = json.loads((EXP / "surprise_c16t32__v11_full_vith" / "per_block.json").read_text())
    S = S["per_video_surprise"]
    blk = {}
    for v, r in meta.items():
        blk.setdefault(r["source_block"], {})[r["variant"]] = r
    pairs = []
    for b, d in sorted(blk.items()):
        for pos, imp in (("pos_a", "imp_ab"), ("pos_b", "imp_ba")):
            if pos in d and imp in d and d[pos]["video_id"] in S:
                vp, vi = d[pos]["video_id"], d[imp]["video_id"]
                pairs.append((vp, vi, d[pos], S[vi] > S[vp]))
    if a.limit:
        pairs = pairs[:: max(1, len(pairs) // a.limit)][:a.limit]
    print(f"  pair {len(pairs)}")

    P = np.load(CACHE / "predictor.npy", mmap_mode="r")
    H = np.load(CACHE / "target.npy", mmap_mode="r")
    C = H.shape[1] // 2
    keys, net, pos, neg, fpos = [], [], [], [], []
    t0 = time.time()
    for i, (vp, vi, r, ok) in enumerate(pairs):
        p = np.asarray(P[ix[vp]], np.float32)
        d = (np.abs(p - np.asarray(H[ix[vi]][C:], np.float32)).mean(1)
             - np.abs(p - np.asarray(H[ix[vp]][C:], np.float32)).mean(1)).reshape(8, 256)
        m = d > 0
        net.append(d.mean(1))
        pos.append(np.where(m.any(1), (d * m).sum(1) / np.maximum(m.sum(1), 1), 0.0))
        neg.append(np.where((~m).any(1), (-d * ~m).sum(1) / np.maximum((~m).sum(1), 1), 0.0))
        fpos.append(m.mean(1))
        keys.append((r["violation_type"], mot(r["condition"]),
                     r.get("occ_timing") or "vis", r["sym_k"], "correct" if ok else "wrong"))
        if i % 2000 == 1999:
            el = time.time() - t0
            print(f"    {i+1}/{len(pairs)}  {el:.0f}s  남은 {el/(i+1)*(len(pairs)-i-1):.0f}s",
                  flush=True)
    np.savez_compressed(a.out, keys=np.array(keys), net=np.array(net, np.float32),
                        pos=np.array(pos, np.float32), neg=np.array(neg, np.float32),
                        fpos=np.array(fpos, np.float32))
    print(f"  [saved] {a.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
