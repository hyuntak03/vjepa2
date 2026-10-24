#!/usr/bin/env python3
"""[P] IntPhys 2 "예측 없는 기준선" — 물리 공간 surprise 와 같은 자리에서 예측을 뺀 판 (학습 없음, CPU) (auto_research, 2026-10-22 KST).

  OMP_NUM_THREADS=8 /venv/vjepa/bin/python auto_research/scripts/ip2_nopred.py [--split Main]

입력: [E] 공유 캐시 pix.npy · meta.json + 내 ip2_block_flow.py 의 U.npy · V.npy (UV.done 필요).
영상 하나에 값 하나 (시간 · 화면 평균):
  frame_diff   = mean |I_{t+1} − I_t| (RGB 0–1, 6 fps 64 장)            — 외형 surprise 에서 흐름 옮기기를 뺀 것
  blk_still    = mean |V[b]| (칸 단위, 블록 대표 장 3 fps)               — 운동 예측 = 멈춤 (움직임의 양)
  blk_constvel = mean |V[b] + U[b]| (같은 칸, 등속 = 앞 변위를 그대로)   — 블록 단위 운동 surprise (내 6 fps 픽셀판의 거친 판)
쌍 정확도 (불가능 > 가능) 와 [비판] crit_ip2_xor.py 분해. 출력 results/ip2_phys/{split}_nopred.json · _{split}_tmp/nopred_*.csv."""
import argparse, csv, collections, json, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="Main")
ap.add_argument("--root", default="/workspace/auto_research_cache/intphys2")
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "../results/ip2_phys"))
a = ap.parse_args()
sys.path.insert(0, os.path.dirname(__file__))
import crit_ip2_xor as X                                                       # noqa: E402
D = os.path.join(a.root, a.split)
assert os.path.exists(os.path.join(D, "UV.done"))
meta = json.load(open(os.path.join(D, "meta.json")))
PIX = np.load(os.path.join(D, "pix.npy"), mmap_mode="r")
U = np.load(os.path.join(D, "U.npy"), mmap_mode="r").astype(np.float32); V = np.load(os.path.join(D, "V.npy"), mmap_mode="r").astype(np.float32)
N = len(meta)
val = {k: np.zeros(N) for k in ("frame_diff", "blk_still", "blk_constvel")}
for i in range(N):
    p = np.asarray(PIX[i]).astype(np.float32) / 255.0
    val["frame_diff"][i] = np.abs(p[1:] - p[:-1]).mean()
    v, u = V[i, 1:-1], U[i, 1:-1]                                               # 둘 다 정의된 블록 1..30
    val["blk_still"][i] = np.linalg.norm(v, axis=-1).mean()
    val["blk_constvel"][i] = np.linalg.norm(v + u, axis=-1).mean()
    if i % 200 == 0:
        print(i, flush=True)
tmp = os.path.join(a.out, f"_{a.split}_tmp"); os.makedirs(tmp, exist_ok=True)
paths, res = [], {}
for k, arr in val.items():
    pth = os.path.join(tmp, f"nopred_{k}.csv"); paths.append(pth)
    with open(pth, "w") as f_:
        w = csv.writer(f_); w.writerow(["scene_index", "pair_id", "is_impossible", "condition", "camera", "difficulty", "context_length", "surprise_avg"])
        for i, m in enumerate(meta):
            w.writerow([m["scene_index"], m["pair_id"], str(m["is_impossible"]), m["condition"], m["camera"], m["difficulty"], 0, arr[i]])
    by = collections.defaultdict(dict)
    for i, m in enumerate(meta):
        by[(m["scene_index"], m["pair_id"])][m["is_impossible"]] = (arr[i], m)
    ok = collections.defaultdict(list)
    for (s, pid), d in by.items():
        o = d[True][0] > d[False][0]; m = d[False][1]
        for g in ("all", m["condition"], f"cam={m['camera']}", f"pair{pid}", f"{m['condition']}|pair{pid}"):
            ok[g].append(o)
    res[k] = {g: [round(100 * np.mean(v), 1), len(v)] for g, v in sorted(ok.items())}
    print(k, " | ".join(f"{g} {v[0]} (n{v[1]})" for g, v in res[k].items()), flush=True)
json.dump(res, open(os.path.join(a.out, f"{a.split}_nopred.json"), "w"), indent=1, ensure_ascii=False)
sys.argv = [sys.argv[0]] + paths + ["--out", os.path.join(a.out, f"{a.split}_xor_nopred.json")]
X.main()
