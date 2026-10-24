#!/usr/bin/env python3
"""[물리] 존재 정답 품질: RAFT 앞-뒤 불일치 (가림) 가 VOBench 끝 가림 물체가 가려지는 순간을 잡나 (학습 없음).

  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 /venv/vjepa/bin/python -W ignore auto_research/scripts/exist_label_quality.py

VOBench 끝 가림 (late_k*) 블록마다 pos_a 영상, 장 0,3,…,93 (32 장, 256 으로). 이웃 장 31 쌍 RAFT 앞 · 뒤 흐름.
라벨: 장마다 물체 중심 (u, v) · 반지름 r_px · 가려진 비율 hidden. Δhidden = hidden_{i+1} − hidden_i.
잴 것: 장 i 에서 물체 원 (보이는 부분) 안 픽셀 중 "i+1 에서 안 보임" 으로 표시된 비율 (가림 비율 occ_frac).
 · 가려지는 쌍 (Δhidden ≥ 0.1) vs 보이는 채 유지 쌍 (hidden_i < 0.05 · Δhidden < 0.02) 의 occ_frac, 둘을 가르는 AUROC
 · occ_frac 과 Δhidden 의 상관 · 기울기 (1 이면 가린 만큼 정확히 표시)
 · 물체 밖 배경의 표시 비율 (헛표시)
정지 (flat_static) · 움직임 나눠서."""
import csv, json, os, sys, collections
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
torch.set_num_threads(8)
dev = torch.device("cuda")
raft = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(dev).eval()
VB = "/workspace/dataset/benchmarks/VOBench_v1"
IDX = list(range(0, 96, 3))[:32]
H = W = 256
yy, xx = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
base = torch.stack([xx, yy], 0).float()

def warp(t, flow):
    g = base[None] + flow
    return F.grid_sample(t, torch.stack([2 * g[:, 0] / (W - 1) - 1, 2 * g[:, 1] / (H - 1) - 1], -1), align_corners=True, padding_mode="border")

def fb(a, b):
    bw = warp(b, a); s = a + bw; g = base[None] + a
    ins = (g[:, 0] >= 0) & (g[:, 0] <= W - 1) & (g[:, 1] >= 0) & (g[:, 1] <= H - 1)
    return ((s ** 2).sum(1) > 0.01 * ((a ** 2).sum(1) + (bw ** 2).sum(1)) + 0.5) & ins

rows = [r for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv"))) if r["occlusion"] == "late" and r["variant"] == "pos_a"]
R = collections.defaultdict(lambda: {"occ_cov": [], "occ_vis": [], "dh": [], "of": [], "bg": []})
for n, r in enumerate(rows):
    d = os.path.join(VB, r["file_name"])
    lab = {q["f"]: q for q in json.load(open(os.path.join(os.path.dirname(d), "label.json")))["labels"]["A"]}
    if any(i not in lab for i in IDX):
        continue
    fr = [np.asarray(Image.open(f"{d}/{i:06d}.png").convert("RGB")) for i in IDX]
    W0 = fr[0].shape[1]
    x = torch.from_numpy(np.stack(fr)).to(dev).permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
    with torch.no_grad():
        xn = x / 127.5 - 1
        Fw = raft(xn[:-1], xn[1:], num_flow_updates=12)[-1]; Bw = raft(xn[1:], xn[:-1], num_flow_updates=12)[-1]
    occ = fb(Fw, Bw).cpu().numpy()
    mv = "static" if r["motion"] == "flat_static" else "moving"
    s = H / W0
    for i in range(len(IDX) - 1):
        q0, q1 = lab[IDX[i]], lab[IDX[i + 1]]
        u, v, rad = q0["u"] * s, q0["v"] * s, max(q0["r_px"] * s, 2.0)
        disk = ((xx.cpu().numpy() - u) ** 2 + (yy.cpu().numpy() - v) ** 2) <= rad ** 2
        far = ((xx.cpu().numpy() - u) ** 2 + (yy.cpu().numpy() - v) ** 2) > (3 * rad) ** 2
        if disk.sum() < 5:
            continue
        of = float(occ[i][disk].mean()); dh = q1["hidden"] - q0["hidden"]
        R[mv]["dh"].append(dh); R[mv]["of"].append(of); R[mv]["bg"].append(float(occ[i][far].mean()))
        if dh >= 0.1:
            R[mv]["occ_cov"].append(of)
        elif q0["hidden"] < 0.05 and dh < 0.02:
            R[mv]["occ_vis"].append(of)
    if (n + 1) % 50 == 0:
        print(n + 1, flush=True)

def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    allv = np.concatenate([pos, neg]); rk = allv.argsort().argsort() + 1
    return float((rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

out = {}
for mv, d in R.items():
    dh, of = np.array(d["dh"]), np.array(d["of"])
    m = dh > 0
    slope = float((of[m] * dh[m]).sum() / max((dh[m] ** 2).sum(), 1e-9)) if m.any() else float("nan")
    out[mv] = dict(n_pairs=int(len(dh)), n_covering=len(d["occ_cov"]), n_visible=len(d["occ_vis"]),
                   occ_frac_covering=round(float(np.mean(d["occ_cov"])) if d["occ_cov"] else float("nan"), 3),
                   occ_frac_visible=round(float(np.mean(d["occ_vis"])) if d["occ_vis"] else float("nan"), 3),
                   auroc=round(auroc(d["occ_cov"], d["occ_vis"]), 3), corr_dh=round(float(np.corrcoef(dh, of)[0, 1]), 3),
                   slope_on_dh=round(slope, 3), bg_false_rate=round(float(np.mean(d["bg"])), 4))
    print(mv, out[mv], flush=True)
json.dump(out, open(os.path.join(REPO, "auto_research/results/exist_label_quality.json"), "w"), indent=1)
