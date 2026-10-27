"""선형 판독기 가중치 비교 — encoder 로 학습한 W_e 와 예측으로 학습한 W_p 의 클래스별 가중치 벡터 cos (2026-10-26, CPU).
사용자: "linear probe = Wx + b 니까 W 의 클래스별 방향끼리 cos sim 비교. encoder embedding 으로 학습한 W, predictor embedding 으로 학습한 Wp".
판독기: identity_linear_transfer.py 와 같다 — 선형층 하나 (1280 → K), 교차 엔트로피, AdamW lr 1e-2 · weight decay 1e-2, 묶음 256, 300 epoch,
  입력 = 블록마다 256 칸 평균 (1, 1280) 그대로 (표준화 없음), 쓰는 클립의 20 % 로 epoch 선택. 예측 블록 b = 7 + k 마다 따로.
비교 (W 의 행 = 클래스 k 를 읽는 1280 짜리 방향):
  행마다 클래스 평균 행을 뺀다 (softmax 는 모든 행에 같은 벡터를 더해도 답이 같아 그 성분은 의미 없음) → 같은 클래스끼리 cos (대각) · 다른 클래스끼리 cos (비대각, 기준).
  전체: W_e (train 전부) vs W_p (train 전부).
  기준선 (같은 표본 수): 학습 클립을 짝수 · 홀수 번째 둘로 나눠 반쪽마다 판독기 (씨앗도 다르게) →
    encoder 끼리 cos(W_e^A, W_e^B) · 예측끼리 cos(W_p^A, W_p^B) · 엇갈림 cos(W_e^A, W_p^B), cos(W_e^B, W_p^A) 평균.
  보정 cos = 엇갈림 / sqrt(encoder 끼리 × 예측끼리).
  /venv/vjepa/bin/python z_research/scripts/analysis/identity_head_weight_cos.py   → z_analysis/results/identity_head_weight_cos.json
"""
import copy, json
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(16)
P = "/workspace/auto_research_cache"
EPOCHS, BS, LR = 300, 256, 1e-2


def train_head(X, y, use, hold, K, seed):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    Xt, yt = torch.from_numpy(X[use & ~hold]), torch.from_numpy(y[use & ~hold]); Xh, yh = torch.from_numpy(X[use & hold]), torch.from_numpy(y[use & hold])
    head = torch.nn.Linear(X.shape[1], K); opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-2)
    best = (-1.0, None)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), BS):
            idx = perm[i:i + BS]; loss = F.cross_entropy(head(Xt[idx]), yt[idx]); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            a = (head(Xh).argmax(1) == yh).float().mean().item()
        if a > best[0]:
            best = (a, copy.deepcopy(head.state_dict()))
    return best[1]["weight"].numpy().astype(np.float64)          # (K, 1280)


def cosmat(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    A = A / np.linalg.norm(A, axis=1, keepdims=True); B = B / np.linalg.norm(B, axis=1, keepdims=True)
    C = A @ B.T; K = len(C)
    return C, float(np.diag(C).mean()), float(C[~np.eye(K, dtype=bool)].mean())


def load(split):
    L = np.load(f"{P}/ar_layers_pooled/{split}.npz"); enc, ro, lab, ii = L["enc"], L["roout"], L["lab"], L["i"]
    R = np.load(f"{P}/release_pooled/{split}.npz"); o = {int(v): n for n, v in enumerate(ii)}; sel = [o[int(v)] for v in R["i"]]
    return {"ar": (enc, ro, lab), "release": (R["gt"], R["pred"], lab[sel])}


TRD = load("train")
out = {}
for model in ("ar", "release"):
    GT, PR, LT = TRD[model]; out[model] = {}
    for fam, col, K in (("shape", 1, 6), ("color", 2, 8)):
        out[model][fam] = {}
        for k in range(1, 9):
            b = 7 + k; mt = LT[:, b, 0] > 0
            y, clip = LT[mt, b, col].astype(np.int64), np.nonzero(mt)[0]
            E, Pd = GT[mt, b].astype(np.float32), PR[mt, k - 1].astype(np.float32)
            allc = np.ones(len(y), bool); hold = (clip % 5) == 0
            hA, hB = (clip % 2) == 0, (clip % 2) == 1; holdh = ((clip // 2) % 5) == 0
            We, Wp = train_head(E, y, allc, hold, K, 0), train_head(Pd, y, allc, hold, K, 0)
            WeA, WeB = train_head(E, y, hA, holdh, K, 1), train_head(E, y, hB, holdh, K, 2)
            WpA, WpB = train_head(Pd, y, hA, holdh, K, 1), train_head(Pd, y, hB, holdh, K, 2)
            C, full_d, full_o = cosmat(We, Wp)
            _, ee, ee_o = cosmat(WeA, WeB); _, pp, pp_o = cosmat(WpA, WpB)
            _, x1, _ = cosmat(WeA, WpB); _, x2, _ = cosmat(WeB, WpA); xe = (x1 + x2) / 2
            out[model][fam][k] = {"full_enc_vs_pred_diag": round(full_d, 3), "full_enc_vs_pred_offdiag": round(full_o, 3),
                                  "full_per_class_diag": np.diag(C).round(3).tolist(),
                                  "half_enc_vs_enc": round(ee, 3), "half_pred_vs_pred": round(pp, 3), "half_enc_vs_pred": round(xe, 3),
                                  "corrected": round(xe / np.sqrt(max(ee, 1e-6) * max(pp, 1e-6)), 3)}
        print(model, fam, flush=True)
        for key in ("full_enc_vs_pred_diag", "full_enc_vs_pred_offdiag", "half_enc_vs_enc", "half_pred_vs_pred", "half_enc_vs_pred", "corrected"):
            print(f"   {key:26s}", [out[model][fam][k][key] for k in range(1, 9)])
json.dump(out, open("/workspace/world_model/vjepa2/z_analysis/results/identity_head_weight_cos.json", "w"), indent=1)
