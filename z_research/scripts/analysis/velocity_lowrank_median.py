"""encoder 벡터에서 속도는 저분산 · 저랭크 방향에 있나, 차원별 중앙값을 하면 속도가 사라지나 (2026-10-27, CPU, 모델 없음).
사용자: "N D shape 에서 1, D vector 보면 속도가 low rank 인지 PCA 해 보면 저분산 차원 이런 거 나올 것 같은데.
         그리고 차원별 중앙값을 해 버리면 진짜 속도가 날라가는지 그런 거 한번 해 봐야 될 듯, encoder embedding 으로".
왜: L1 최적 출력 = 차원별 조건부 중앙값. 속도가 (a) 분산이 작은 방향에 있거나 (b) 차원별 중앙값과 어긋나는 꼴로 들어 있으면,
    미래가 갈릴 때 중앙값이 속도를 지운다 → 블록 AR 출력에 encoder 꼴 속도가 없는 이유 (ar_output_velocity_linear.json) 를 encoder 쪽에서 확인.
벡터: 3D 합성 물체 하나 (ar_layers train 앞 N 편 + val 350 편), 블록 b = 2–15 중 b−1 · b 물체 보임 → 블록 b 의 물체 칸 평균 encoder 토큰 (1, 1280).
  라벨: 블록 안 변위 (vx, vy, px / 0.25 s) · 물체 중심 (x, y) · 모양 (6) · 색 (8).
① PCA (train 으로 맞춤): 분산 스펙트럼, 상위 k 주성분만으로 선형 판독 (최소제곱, train 맞춤 → val 시험) k = 1 … 1280.
   속성별로 전체 판독의 50 · 90 % 에 닿는 k · 각 속성 판독이 기대는 주성분의 분산 (판독 가중치 제곱으로 가중한 평균 분산 순위).
   + 속성이 벡터 분산을 얼마나 설명하나: 벡터 ~ 속성 (선형) 의 R² (속도 2 · 위치 2 · 색 원핫 · 모양 원핫).
② 중앙값 시험: 오른쪽 이동 (vx ≥ 6, |vy| < 4) · 왼쪽 이동 (vx ≤ −6, |vy| < 4) 묶음에서 n = 31 개를 오른쪽 비율 p 로 섞어
   차원별 중앙값 벡터 · 평균 벡터를 만들고, 개별 벡터로 맞춘 선형 판독기로 vx 를 읽는다 (p = 0, 0.1, …, 1, 반복 200).
   대조: 색 (빨강 vs 파랑, ±1 판독기) 을 같은 방식으로. 읽은 값을 순수 왼쪽 (파랑) = −1, 순수 오른쪽 (빨강) = +1 로 맞춘다.
   차원별 중앙값이 속성을 충실히 지키면 p > 0.5 에서 +1 근처 (계단), 지우면 0 쪽으로 눌린다.
  /venv/vjepa/bin/python z_research/scripts/analysis/velocity_lowrank_median.py --n_train 3000   → z_analysis/results/velocity_lowrank_median.json
"""
import argparse, json, os
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--n_train", type=int, default=3000); ap.add_argument("--threads", type=int, default=16); a = ap.parse_args()
os.environ["OMP_NUM_THREADS"] = str(a.threads)
SRC = "/workspace/auto_research_cache/ar_layers"; rng = np.random.default_rng(0)


def load(split, n):
    X, L = [], []
    for m in json.load(open(f"{SRC}/{split}_meta.json"))[:n]:
        path = f"{SRC}/{split}/{m['i']:05d}.npz"; z = np.load(path); enc = z["enc"]; z.close()
        fd = os.open(path, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
        for b in range(2, 16):
            bl = m["blocks"][b]
            if bl["ok"] and m["blocks"][b - 1]["ok"] and bl["cells"]:
                X.append(enc[b][bl["cells"]].astype(np.float32).mean(0)); L.append([*bl["disp"], *bl["ctr"], m["shape"], m["color"]])
    return np.stack(X), np.array(L, np.float64)


Xtr, Ltr = load("train", a.n_train); Xva, Lva = load("val", 350)
print("표본 train", len(Xtr), "val", len(Xva), "색 개수", np.bincount(Ltr[:, 5].astype(int)), flush=True)
mu = Xtr.mean(0); Ztr, Zva = Xtr - mu, Xva - mu
U, sv, Vt = np.linalg.svd(Ztr, full_matrices=False); var = sv ** 2 / len(Ztr); Ptr, Pva = Ztr @ Vt.T, Zva @ Vt.T     # 주성분 점수 (분산 큰 순)
onehot = lambda v, k: np.eye(k)[v.astype(int)]
ATTR = {"velocity": (Ltr[:, 0:2], Lva[:, 0:2], "r2"), "position": (Ltr[:, 2:4], Lva[:, 2:4], "r2"),
        "shape": (onehot(Ltr[:, 4], 6), onehot(Lva[:, 4], 6), "acc"), "color": (onehot(Ltr[:, 5], 8), onehot(Lva[:, 5], 8), "acc")}


def score(P_tr, P_va, Ytr, Yva, kind):
    A = np.c_[P_tr, np.ones(len(P_tr))]; w = np.linalg.lstsq(A, Ytr, rcond=None)[0]; Yh = np.c_[P_va, np.ones(len(P_va))] @ w
    if kind == "acc":
        return float((Yh.argmax(1) == Yva.argmax(1)).mean())
    return float(1 - ((Yh - Yva) ** 2).sum() / ((Yva - Yva.mean(0)) ** 2).sum())


KS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1280]
R = {"n_train": int(len(Xtr)), "n_val": int(len(Xva)), "explained_variance_ratio_top": [round(float(v), 4) for v in (var / var.sum())[:20]],
     "cum_variance_at_k": {str(k): round(float(var[:k].sum() / var.sum()), 4) for k in KS}, "readout_by_top_k": {}, "k_to_reach": {}, "attribute_explains_vector_variance": {},
     "readout_weight_variance_rank": {}}
for nm, (Ytr, Yva, kind) in ATTR.items():
    curve = {str(k): round(score(Ptr[:, :k], Pva[:, :k], Ytr, Yva, kind), 4) for k in KS}
    R["readout_by_top_k"][nm] = curve
    full = curve["1280"]; base = (1 / (6 if nm == "shape" else 8)) if kind == "acc" else 0.0
    def reach(fr):
        for k in KS:
            if curve[str(k)] - base >= fr * (full - base):
                return k
    R["k_to_reach"][nm] = {"50%": reach(0.5), "90%": reach(0.9), "full": full}
    # 판독 가중치가 어느 분산의 주성분에 실리나: 표준화 주성분 공간에서 최소제곱 가중치 제곱의 분포 → 가중 평균 순위 · 분산
    S = Ptr / np.sqrt(var); w = np.linalg.lstsq(np.c_[S, np.ones(len(S))], Ytr, rcond=None)[0][:-1]; wt = (w ** 2).sum(1); wt = wt / wt.sum()
    R["readout_weight_variance_rank"][nm] = {"weighted_mean_rank": round(float((wt * np.arange(1, len(wt) + 1)).sum()), 1),
                                             "weight_share_in_top_32": round(float(wt[:32].sum()), 3), "weight_share_in_top_128": round(float(wt[:128].sum()), 3)}
    # 속성 → 벡터 (반대 방향 회귀): 벡터 분산 중 이 속성으로 설명되는 몫
    A = np.c_[Ytr, np.ones(len(Ytr))]; Wb = np.linalg.lstsq(A, Ztr, rcond=None)[0]
    R["attribute_explains_vector_variance"][nm] = round(float(1 - ((Ztr - A @ Wb) ** 2).sum() / (Ztr ** 2).sum()), 4)
    print(nm, R["k_to_reach"][nm], R["readout_weight_variance_rank"][nm], "설명 분산", R["attribute_explains_vector_variance"][nm], flush=True)

# ② 중앙값 시험
Xall = np.concatenate([Xtr, Xva]); Lall = np.concatenate([Ltr, Lva])
wv = np.linalg.lstsq(np.c_[Xtr, np.ones(len(Xtr))], Ltr[:, 0], rcond=None)[0]; read_v = lambda v: v @ wv[:-1] + wv[-1]
rb = (Ltr[:, 5] == 0) | (Ltr[:, 5] == 5); yc = np.where(Ltr[rb, 5] == 0, 1.0, -1.0)
wc = np.linalg.lstsq(np.c_[Xtr[rb], np.ones(rb.sum())], yc, rcond=None)[0]; read_c = lambda v: v @ wc[:-1] + wc[-1]
groups = {"velocity": (Xall[(Lall[:, 0] >= 6) & (np.abs(Lall[:, 1]) < 4)], Xall[(Lall[:, 0] <= -6) & (np.abs(Lall[:, 1]) < 4)], read_v),
          "color": (Xall[Lall[:, 5] == 0], Xall[Lall[:, 5] == 5], read_c)}
R["median_test"] = {}
for nm, (Xp, Xn, rd) in groups.items():
    lo, hi = float(np.mean(rd(Xn))), float(np.mean(rd(Xp)))                     # 개별 벡터 판독 평균 (순수 왼쪽 / 오른쪽)
    norm = lambda v: 2 * (v - lo) / (hi - lo) - 1
    res = {"n_pos": int(len(Xp)), "n_neg": int(len(Xn))}
    for p in np.round(np.linspace(0, 1, 11), 1):
        med, mean = [], []
        for _ in range(200):
            k = int(round(p * 31)); G = np.concatenate([Xp[rng.choice(len(Xp), k, replace=False)], Xn[rng.choice(len(Xn), 31 - k, replace=False)]])
            med.append(norm(rd(np.median(G, 0)))); mean.append(norm(rd(G.mean(0))))
        res[str(p)] = {"median_vector": round(float(np.mean(med)), 3), "mean_vector": round(float(np.mean(mean)), 3)}
    R["median_test"][nm] = res
    print(nm, "중앙값 시험:", " ".join(f"p{p}: med {v['median_vector']:+.2f} mean {v['mean_vector']:+.2f}" for p, v in res.items() if p.startswith(("0", "1"))), flush=True)
json.dump(R, open("/workspace/world_model/vjepa2/z_analysis/results/velocity_lowrank_median.json", "w"), indent=1)
print("saved")
