#!/usr/bin/env python3
"""[P] IntPhys 2 물리 공간 surprise 기준선 (학습 없음) — 운동 (흐름 등속 외삽 오차) · 외형 (픽셀, 흐름으로 옮긴 현재 장 vs 다음 장) + 흐름 지속 계수 k (auto_research, 2026-10-22 KST).

  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=5 /venv/vjepa/bin/python -W ignore auto_research/scripts/ip2_phys_surprise.py --split Debug
  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=5 /venv/vjepa/bin/python -W ignore auto_research/scripts/ip2_phys_surprise.py --split Main

프레임: 60 fps 영상에서 6 fps (10 장마다, 채점 규약과 같은 속도), 512 → 256. RAFT-large 12 회.
 순방향 f_t (t → t+1) · 역방향 b_t (t → t−1) 를 이웃 장마다.
 운동 surprise (장 t, t = 1..T−2): mean_x |f_t(x) − (−b_t(x))| (픽셀 단위 변위, 같은 내용의 등속 외삽 vs 관측).
 외형 surprise (장 t+1): mean_x |I_{t+1}(x) − I_t(x + b_{t+1}(x))| (RGB 0–1, 쌍선형).
 영상 값 = 시간 평균 (avg) · 최대 (max).
흐름 지속 계수 (가능 영상만, [물리] flow_persistence.py 의 coef 그대로): 블록 = 6 fps 의 연속 2 장, 대표 장 = 둘째 장 (간격 1/3 초).
 U[b] = 블록 b 대표 장 → 블록 b−1 대표 장 역방향 흐름의 16 px 칸 평균 (칸 단위). 움직이는 칸 |U[t+1](c)| ≥ 0.5, Y = U[t+1](c), X = U[t](c + U[t+1](c)).
출력: auto_research/results/ip2_phys/{split}_per_video_{motion,appear}{,_max}.csv ([비판] crit_ip2_xor.py 와 같은 열) · {split}_persist.npz · {split}_rows.jsonl (이어 하기)."""
import argparse, csv, json, os, sys
import numpy as np, torch, torch.nn.functional as F
import decord

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="Debug", choices=["Debug", "Main"])
ap.add_argument("--root", default="/workspace/dataset/benchmarks/IntPhys2")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "../results/ip2_phys"))
a = ap.parse_args()
torch.set_num_threads(8)
dev = torch.device("cuda")
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights   # noqa: E402
raft = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(dev).eval()
os.makedirs(a.out, exist_ok=True)
R, G = 256, 16
yy, xx = torch.meshgrid(torch.arange(R, device=dev).float(), torch.arange(R, device=dev).float(), indexing="ij")


@torch.no_grad()
def rflow(i0, i1, bs=16):
    out = []
    for k in range(0, i0.shape[0], bs):
        out.append(raft(i0[k:k + bs], i1[k:k + bs], num_flow_updates=12)[-1])
    return torch.cat(out)                                                       # (n, 2, R, R) (dx, dy) px


def warp_img(img, flow):                                                        # img (n,3,R,R), flow (n,2,R,R): out(x) = img(x + flow(x))
    gx = (xx[None] + flow[:, 0]) / (R - 1) * 2 - 1; gy = (yy[None] + flow[:, 1]) / (R - 1) * 2 - 1
    return F.grid_sample(img, torch.stack([gx, gy], -1), align_corners=True, padding_mode="border")


meta = list(csv.DictReader(open(os.path.join(a.root, a.split, "metadata.csv"))))
scenes = sorted({m_["SceneIndex"] for m_ in meta}, key=int)
np.random.default_rng(0).shuffle(scenes)                                        # 장면 순서를 섞는다 (중간 보고가 전체를 대표하게)
rank = {s_: i for i, s_ in enumerate(scenes)}
meta = sorted(meta, key=lambda m_: rank[m_["SceneIndex"]])
if a.limit:
    meta = meta[: a.limit]
# [E] 공유 캐시 (1012, 64, 256, 256, 3) 가 다 만들어졌으면 디코딩 대신 쓴다 (Main 순서 = metadata.csv 순서)
PIX = None
pp = "/workspace/auto_research_cache/intphys2/pix.npy"
if a.split == "Main" and os.path.exists(pp) and os.path.exists(pp + ".done"):
    PIX = np.load(pp, mmap_mode="r"); ROWIDX = {m_["file_name"]: i for i, m_ in enumerate(csv.DictReader(open(os.path.join(a.root, a.split, "metadata.csv"))))}
    print("공유 픽셀 캐시 사용", flush=True)
rows_path = os.path.join(a.out, f"{a.split}_rows.jsonl")
done = set()
if os.path.exists(rows_path):
    done = {json.loads(l)["file_name"] for l in open(rows_path)}
fo = open(rows_path, "a")
PY, PX = [], []
for n, m in enumerate(meta):
    if m["file_name"] in done:
        continue
    if PIX is not None:
        fr = torch.from_numpy(np.asarray(PIX[ROWIDX[m["file_name"]]])).to(dev).permute(0, 3, 1, 2).float() / 255.0
    else:
        vr = decord.VideoReader(os.path.join(a.root, a.split, m["file_name"]), num_threads=2)
        idx = list(range(0, len(vr), 10))
        fr = torch.from_numpy(vr.get_batch(idx).asnumpy()).to(dev).permute(0, 3, 1, 2).float() / 255.0
        fr = F.interpolate(fr, size=(R, R), mode="bilinear", align_corners=False, antialias=True)        # (T,3,R,R) 0–1
    x = fr * 2 - 1
    T = x.shape[0]
    with torch.no_grad():
        f = rflow(x[:-1], x[1:])                                                 # f[t] : t → t+1  (t = 0..T−2)
        b = rflow(x[1:], x[:-1])                                                 # b[t] : t+1 → t  (역방향, 인덱스 t+1 자리)
        # 운동: 장 t (1..T−2) 에서 등속 예측 −b_t 와 관측 f_t
        mot = (f[1:] - (-b[:-1])).norm(dim=1).mean((1, 2))                       # (T−2,)  b[:-1][t−1] = b_t (t → t−1)
        # 외형: I_{t+1} vs I_t(x + b_{t+1}(x))
        app = (fr[1:] - warp_img(fr[:-1], b)).abs().mean((1, 2, 3))              # (T−1,)
        # 블록 흐름 (대표 장 = 홀수 장)
        rep = x[1::2]
        ub = F.avg_pool2d(rflow(rep[1:], rep[:-1]), 16) / 16                    # (nb−1, 2, G, G): 블록 b+1 → b
    rec = dict(file_name=m["file_name"], scene_index=m["SceneIndex"], type=m["type"], condition=m["condition"], difficulty=m["Difficulty"],
               camera=m["Camera"], n_frames=T, motion_avg=float(mot.mean()), motion_max=float(mot.max()), appear_avg=float(app.mean()), appear_max=float(app.max()))
    fo.write(json.dumps(rec) + "\n"); fo.flush()
    if m["type"].endswith("Possible"):
        U = ub.permute(0, 2, 3, 1).reshape(ub.shape[0], G * G, 2).cpu().numpy()      # U[k] = 블록 k+1 → k
        np.save(os.path.join(a.out, f"{a.split}_U_{m['file_name'].split('/')[-1][:16]}.npy"), U.astype(np.float16))
    if n % 50 == 0:
        print(f"{n}/{len(meta)}", flush=True)
fo.close()

# ---------------- per_video csv ([비판] 형식) + 정확도 표
rows = [json.loads(l) for l in open(rows_path)]
for key in ("motion_avg", "motion_max", "appear_avg", "appear_max"):
    with open(os.path.join(a.out, f"{a.split}_per_video_{key}.csv"), "w") as f_:
        w = csv.writer(f_); w.writerow(["row_index", "scene_index", "pair_id", "is_impossible", "condition", "difficulty", "camera", "env", "context_length", "surprise_avg", "surprise_max", "n_windows"])
        for i, r in enumerate(rows):
            w.writerow([i, r["scene_index"], r["type"].split("_")[0], str(r["type"].endswith("Impossible")), r["condition"], r["difficulty"], r["camera"], "", 0, r[key], r[key], 1])
print("per_video csv 저장:", a.out)
