#!/usr/bin/env python3
"""학습 영상에서 다음 블록의 블록 안 움직임이 문맥으로 얼마나 정해지나 — RAFT 칸 흐름 (2026-10-26).

질문 (사용자 "해봐"): AR 출력에 블록 안 움직임이 안 남는 이유가
  갈래 A — 학습 영상에서 다음 블록 움직임이 문맥으로 잘 안 정해져 L1 최적이 "움직임 없음" 쪽인가
  갈래 B — 정해지는데 모델이 못 배운 것인가.
학습 없음 · 모델 없음. 자연 영상은 미리 뽑은 흐름 (z_training/data/extract_flow_fps4.py: RAFT large, fps4 연속 두 장, 16×16 칸 평균, 256² 화면 px).

표기 (fps4 연속 쌍 j 의 칸 흐름 f[j], (16, 16, 2) = (dx, dy)):
  블록 = 연속 2 장. 지금 블록 안 움직임 w1 = f[j], 다음 블록 안 움직임 w2 = f[j + 2] (예측 대상),
  앞 블록 안 움직임 w0 = f[j − 2], 앞 블록 → 지금 블록 사이 u0 = f[j − 1]. 문맥으로 쓸 수 있는 것은 w0 · u0 · w1 뿐 (f[j + 1] 은 미래 장을 쓴다).
  카메라 보정판 (cam): 쌍마다 256 칸 중앙값을 뺀다.
  움직이는 칸 = |w1| ≥ T px (0.25 s 당; T = 2 · 4).
  같은 칸: w2 를 같은 칸에서. 따라간 칸 (adv): w2 를 c + round(2 · w1 / 16) 칸에서 (물체가 두 쌍 뒤 있을 자리, 문맥만으로).
지표 (움직이는 칸):
  r   = median ρ, ρ = (w2 · w1) / |w1|² — 지금 움직임 방향으로 다음 움직임이 몇 배 이어지나 (1 = 그대로, 0 = 멈춤 또는 무작위). L1 최적 축소율의 근사
  gain = Σ (w2 · w1) / Σ |w1|² (L2 최적 배율) · cos = median cos(w2, w1)
  R²_copy = 1 − Σ|w2 − w1|² / Σ|w2 − w̄2|² (그대로 이어 붙이기) · R²_gain (배율 하나, 영상 절반으로 맞춤)
  R²_lin = 선형 예측 (w1 3×3 이웃 · w0 · u0 · 카메라 중앙값 → w2, 영상 절반으로 맞추고 나머지 절반에서 시험). 문맥 전체를 쓰는 예측기보다 약한 하한
측정 도구 검사 (phys3d val, 등속에 가까운 3D 합성 — 답을 아는 경우):
  GT: 라벨의 블록 안 변위 disp (0.25 s) 로 ρ = disp_{b+1} · disp_b / |disp_b|²
  (phys3d RAFT 는 장면 요소 움직임이 섞여 검사 기준에서 뺐다)
  합성 이동 패치: 실제 배경 위 40×40 조각, 계속 (기대 r = 1) / 멈춤 (기대 r = 0) — 같은 RAFT · 같은 칸 평균
  VOBench 보임 영상 (pos_a): 정답 라벨 r 과 RAFT 칸 r 비교, 운동 종류별
  한 쌍 뒤 지속성 (between_lag1): 지금 블록 → 다음 블록 사이 움직임 u1 = f[j + 1] 을 c + round(w1 / 16) 칸에서 (0.25 s 뒤)

  CUDA_VISIBLE_DEVICES=0 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/motion_predictability_flow.py --n_videos 4000
출력: z_analysis/results/motion_predictability_flow.json
"""
import argparse, glob, json, os, random
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
FLOW = "/workspace/dataset/training_data/flow_fps4"
SRCS = {"k710": "k710_320", "ssv2": "ssv2", "howto": "ht100m_1pct"}
ap = argparse.ArgumentParser()
ap.add_argument("--n_videos", type=int, default=4000)
ap.add_argument("--per_video", type=int, default=3)
ap.add_argument("--phys3d_n", type=int, default=150)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=os.path.join(REPO, "z_analysis/results/motion_predictability_flow.json"))
a = ap.parse_args()
TS = (2.0, 4.0)
G = 16


def load_src(src):
    vids = []
    for ip in sorted(glob.glob(f"{FLOW}/{src}/shard*_index.tsv")):
        mm = np.memmap(ip.replace("_index.tsv", "_flow.f16"), np.float16, "r")
        mm = mm[: (mm.size // 512) * 512].reshape(-1, G, G, 2)
        for l in open(ip):
            p, r0, n = l.rstrip("\n").split("\t"); r0, n = int(r0), int(n)
            if n >= 5 and r0 + n <= len(mm):
                vids.append((mm, r0, n))
    return vids


def sample(vids, rng):
    """→ F (N, 5, 16, 16, 2) = [w0, u0, w1, u1, w2], vid (N,)"""
    sel = rng.choice(len(vids), min(a.n_videos, len(vids)), replace=False)
    F, vid = [], []
    for k, vi in enumerate(sel):
        mm, r0, n = vids[vi]
        for j in rng.choice(np.arange(2, n - 2), size=min(a.per_video, n - 4), replace=False):
            F.append(np.asarray(mm[r0 + j - 2: r0 + j + 3], dtype=np.float32)); vid.append(k)
    return np.stack(F), np.array(vid)


def cam_removed(F):
    return F - np.median(F, axis=(2, 3), keepdims=True)


def metrics(F, vid, T):
    W0, U0, W1, W2 = F[:, 0], F[:, 1], F[:, 2], F[:, 4]
    n_, rr, cc = np.nonzero(np.linalg.norm(W1, axis=-1) >= T)
    if len(n_) < 50:
        return None
    a1 = W1[n_, rr, cc]
    dr = np.clip(rr + np.round(2 * a1[:, 1] / 16).astype(int), 0, G - 1); dc = np.clip(cc + np.round(2 * a1[:, 0] / 16).astype(int), 0, G - 1)
    half = (vid[n_] % 2 == 0)
    # 선형 예측 특징: w1 3×3 이웃 · w0 · u0 · 카메라 중앙값 3 개 · 상수
    P = np.pad(W1, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
    nb = np.concatenate([P[n_, rr + dy, cc + dx] for dy in range(3) for dx in range(3)], 1)
    camf = np.concatenate([np.median(X, axis=(1, 2))[n_] for X in (W1, U0, W0)], 1)
    X = np.concatenate([nb, W0[n_, rr, cc], U0[n_, rr, cc], camf, np.ones((len(n_), 1))], 1)
    out = {"n_cells": int(len(n_)), "speed_med_px": float(np.median(np.linalg.norm(a1, axis=1)))}
    U1 = F[:, 3]; br = np.clip(rr + np.round(a1[:, 1] / 16).astype(int), 0, G - 1); bc = np.clip(cc + np.round(a1[:, 0] / 16).astype(int), 0, G - 1)
    for name, b in (("same", W2[n_, rr, cc]), ("adv", W2[n_, dr, dc]), ("between_lag1", U1[n_, br, bc])):
        dot = (a1 * b).sum(1); nn = (a1 ** 2).sum(1)
        rho = dot / nn; cs = dot / (np.linalg.norm(a1, axis=1) * np.linalg.norm(b, axis=1) + 1e-6)
        sst = lambda m: ((b[m] - b[m].mean(0)) ** 2).sum()
        g_tr = (dot[half]).sum() / nn[half].sum()
        w, *_ = np.linalg.lstsq(X[half], b[half], rcond=None)
        te = ~half
        out[name] = {"r": float(np.median(rho)), "gain": float(dot.sum() / nn.sum()), "cos": float(np.median(cs)),
                     "next_speed_ratio_med": float(np.median(np.linalg.norm(b, axis=1) / np.linalg.norm(a1, axis=1))),
                     "R2_copy": float(1 - ((b - a1) ** 2).sum() / sst(np.ones(len(b), bool))),
                     "R2_gain": float(1 - ((b[te] - g_tr * a1[te]) ** 2).sum() / sst(te)),
                     "R2_lin": float(1 - ((b[te] - X[te] @ w) ** 2).sum() / sst(te))}
        sp = np.linalg.norm(a1, axis=1)
        out[name]["r_by_speed"] = {lb: (float(np.median(rho[(sp >= lo) & (sp < hi)])) if ((sp >= lo) & (sp < hi)).sum() >= 50 else None)
                                   for lo, hi, lb in ((2, 4, "2-4px"), (4, 8, "4-8px"), (8, 1e9, ">=8px"))}
    return out


def show(tag, m):
    if m is None:
        print(f"  {tag:28s} —"); return
    s, d = m["same"], m["adv"]
    print(f"  {tag:28s} n {m['n_cells']:7d} speed {m['speed_med_px']:4.1f}px | same r {s['r']:.2f} cos {s['cos']:.2f} R2copy {s['R2_copy']:.2f} "
          f"| adv r {d['r']:.2f} gain {d['gain']:.2f} cos {d['cos']:.2f} R2gain {d['R2_gain']:.2f} R2lin {d['R2_lin']:.2f} spd-ratio {d['next_speed_ratio_med']:.2f} "
          f"| lag1 r {m['between_lag1']['r']:.2f} cos {m['between_lag1']['cos']:.2f} | by-speed {({k: (None if v is None else round(v, 2)) for k, v in d['r_by_speed'].items()})}", flush=True)


rng = np.random.default_rng(a.seed)
R = {"config": vars(a)}
for name, src in SRCS.items():
    vids = load_src(src)
    if not vids:
        continue
    F, vid = sample(vids, rng)
    R[name] = {"n_videos_available": len(vids), "n_samples": int(len(F))}
    print(f"=== {name}: 영상 {len(vids)} 중 {len(np.unique(vid))} · 표본 {len(F)}", flush=True)
    for mode, FF in (("raw", F), ("cam", cam_removed(F))):
        for T in TS:
            m = metrics(FF, vid, T); R[name][f"{mode}_T{T:g}"] = m; show(f"{mode} T{T:g}", m)

# ---------------- 측정 도구 검사: phys3d val
if a.phys3d_n:
    import torch, torch.nn.functional as Fn, decord
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    meta = json.load(open("/workspace/auto_research_cache/ar_layers/val_meta.json"))
    # GT (라벨 변위)
    for T in TS:
        rho, dots, nns, cs = [], [], [], []
        for m in meta:
            B = m["blocks"]
            for b in range(15):
                if B[b]["ok"] and B[b + 1]["ok"]:
                    v1, v2 = np.array(B[b]["disp"]), np.array(B[b + 1]["disp"])
                    if np.linalg.norm(v1) >= T:
                        rho.append(v1 @ v2 / (v1 @ v1)); dots.append(v1 @ v2); nns.append(v1 @ v1)
                        cs.append(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
        R.setdefault("phys3d_gt", {})[f"T{T:g}"] = {"n": len(rho), "r": float(np.median(rho)), "gain": float(np.sum(dots) / np.sum(nns)), "cos": float(np.median(cs))}
        print(f"  phys3d GT T{T:g}: n {len(rho)} r {np.median(rho):.2f} gain {np.sum(dots) / np.sum(nns):.2f} cos {np.median(cs):.2f}", flush=True)
    # (2026-10-26) phys3d RAFT 는 검사 기준으로 못 쓴다: 장면 요소 (판 · 문 등) 가 짧게 크게 움직여 (한 클립 블록 7–9 에 49 칸 −18 ~ −35 px) 물체 지속성과 섞인다.
    # 대신 답을 아는 두 경우로 도구를 검사한다.
    dev = torch.device("cuda"); raft = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(dev).eval()

    def raft_cells(x):
        """x: (T, 3, 256, 256) float 0–255 → (T−1, 16, 16, 2) 칸 평균 흐름 (자연 영상 추출과 같은 설정)"""
        out = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for i in range(0, len(x) - 1, 8):
                seg = x[i:i + 9].to(dev).float() / 127.5 - 1
                f = raft(seg[:-1].contiguous(), seg[1:].contiguous(), num_flow_updates=12)[-1]
                out.append(Fn.avg_pool2d(f.float(), 16).permute(0, 2, 3, 1).cpu().numpy())
        return np.concatenate(out)

    # 검사 1: 합성 이동 패치 — K710 첫 장 위에 같은 영상의 40×40 조각을 블록마다 일정하게 움직인다.
    #   계속 (등속, 기대 r = 1) / 멈춤 (장 0–3 동안 움직이고 멈춤, j = 2 에서 w1 움직임 · w2 = 0, 기대 r = 0)
    trows = [l.rstrip("\n").split("\t") for l in open(os.path.join(REPO, "data_csv/k710_320/fps4_table.tsv"))]; random.Random(5).shuffle(trows)
    bgs = []
    for r in trows:
        if len(bgs) >= 40:
            break
        try:
            im = decord.VideoReader(r[0], num_threads=2)[0].asnumpy()
        except Exception:
            continue
        H, W = im.shape[:2]; s_ = min(H, W); y0, x0 = (H - s_) // 2, (W - s_) // 2
        t = torch.from_numpy(np.ascontiguousarray(im[y0:y0 + s_, x0:x0 + s_])).permute(2, 0, 1)[None].float()
        bgs.append(Fn.interpolate(t, size=(256, 256), mode="bilinear", align_corners=False)[0].clamp(0, 255).round())
    srng = np.random.default_rng(1)
    for spd in (4, 8, 16):
        for kind in ("constant", "stop"):
            FL, vid = [], []
            for k, bg in enumerate(bgs):
                ang = srng.uniform(0, 2 * np.pi); v = np.array([np.cos(ang), np.sin(ang)]) * spd
                patch = bg[:, 20:60, 20:60].flip(0).clone()                      # 색을 뒤집은 조각 (배경과 구별)
                p0 = np.array([128.0, 128.0]) - 4 * v - 20
                frames = []
                for t_ in range(8):
                    tt = t_ if (kind == "constant" or t_ <= 3) else 3
                    x_, y_ = np.clip(np.round(p0 + tt * v).astype(int), 0, 216)
                    fr_ = bg.clone(); fr_[:, y_:y_ + 40, x_:x_ + 40] = patch; frames.append(fr_)
                f = raft_cells(torch.stack(frames))                             # (7, 16, 16, 2)
                for j in ((2, 3, 4) if kind == "constant" else (2,)):
                    FL.append(f[j - 2: j + 3]); vid.append(k)
            m = metrics(np.stack(FL), np.array(vid), 2.0); R.setdefault("check_synthetic_patch", {})[f"{kind}_{spd}px"] = m
            show(f"check patch {kind} {spd}px", m)

    # 검사 2: VOBench 보임 영상 (정답 라벨 위치), 운동 종류별. 채점과 같은 간격 (30 fps 3 장마다)
    from PIL import Image
    import csv
    VOB = "/workspace/dataset/benchmarks/VOBench_v1"; VIDX = list(range(0, 96, 3))[:32]
    vrows = [r for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv")))
             if r["variant"] == "pos_a" and r["occlusion"] == "visible" and r["motion"] != "flat_static"]
    by = {}
    for r in vrows:
        d = os.path.join(VOB, r["file_name"])
        ims = [np.asarray(Image.open(f"{d}/{i:06d}.png").convert("RGB")) for i in VIDX]
        W0 = ims[0].shape[1]
        x = Fn.interpolate(torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).float(), size=(256, 256), mode="bilinear", align_corners=False).clamp(0, 255).round()
        f = raft_cells(x)
        lab = {q["f"]: q for q in json.load(open(os.path.join(os.path.dirname(d), "label.json")))["labels"]["A"]}
        P = np.array([[lab[i]["u"], lab[i]["v"]] for i in VIDX]) / W0 * 256
        e = by.setdefault(r["motion"], {"F": [], "vid": [], "gt": []}); k = len({*e["vid"]})
        for j in range(2, 29):
            e["F"].append(f[j - 2: j + 3]); e["vid"].append(k)
            w1, w2 = P[j + 1] - P[j], P[j + 3] - P[j + 2]
            if np.linalg.norm(w1) >= 2:
                e["gt"].append(w1 @ w2 / (w1 @ w1))
    for mo, e in sorted(by.items()):
        m = metrics(np.stack(e["F"]), np.array(e["vid"]), 2.0)
        R.setdefault("check_vobench_visible", {})[mo] = {"raft": m, "gt_r": float(np.median(e["gt"])) if e["gt"] else None, "gt_n": len(e["gt"])}
        show(f"check VOBench {mo} (GT r {np.median(e['gt']) if e['gt'] else float('nan'):.2f})", m)
    print(f"  RAFT 최대 메모리 {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)

json.dump(R, open(a.out, "w"), indent=1)
print("saved", a.out)
