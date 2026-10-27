"""같은 표본으로 encoder 1280 vs encoder → 입력 투영 384 (vs 굴린 예측 상태 384) 의 모양 × 색 판독 비교 (2026-10-26, CPU).
사용자: "같은 샘플로 encoder 1280, 384 각각 비교해봐".
표본: 블록 8–14 에서 물체가 보이는 표본 (세 표현 모두 같은 클립 · 같은 블록 · 같은 train 8,000 / val 350 분할).
표현: encoder 1280 (블록 단위 LN(target_encoder) 칸 평균) · encoder → 입력 투영 384 (블록 AR natural_ar_vw_state 의 predictor_embed) ·
      굴린 예측 상태 384 (굴림 자리 블록 b 의 마지막 정규화 출력). 모양 × 색은 클립마다 하나라 블록을 바꿔도 라벨이 같다.
판독기: 선형층 하나 (48 클래스, 교차 엔트로피), 입력은 자기 학습 표본 통계로 표준화, AdamW lr 1e-3 · weight decay 1e-2 · 묶음 256, train 클립 20 % 로 epoch 선택.
  블록 하나 = 블록마다 따로 학습 (약 3,700 표본, 최대 200 epoch) / 합침 = 블록 8–14 를 합쳐 하나 학습 (약 26,000 표본, 최대 100 epoch). 시험은 블록마다.
  /venv/vjepa/bin/python z_research/scripts/analysis/identity_1280_vs_384_same_samples.py   → z_analysis/results/identity_1280_vs_384_same_samples.json
"""
import copy, json
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(32)
P = "/workspace/auto_research_cache/ar_layers_pooled"
sd = torch.load("/workspace/world_model/vjepa2/z_training/runs/natural_ar_vw_state/latest.pt", map_location="cpu", weights_only=False)["predictor"]
pick = lambda suf: next(v for k, v in sd.items() if k.endswith(suf)).float().numpy()
W, bW = pick("predictor_embed.weight"), pick("predictor_embed.bias")
TR, VA = np.load(f"{P}/train.npz"), np.load(f"{P}/val.npz")
BLOCKS = list(range(8, 15))


def fit(X, y, clip, epochs):
    torch.manual_seed(0); X, y = torch.from_numpy(X), torch.from_numpy(y); h = torch.from_numpy(clip % 5 == 0)
    head = torch.nn.Linear(X.shape[1], 48); opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2); best = (-1, None)
    Xa, ya, Xh, yh = X[~h], y[~h], X[h], y[h]
    for ep in range(epochs):
        perm = torch.randperm(len(Xa))
        for i in range(0, len(Xa), 256):
            idx = perm[i:i + 256]; loss = F.cross_entropy(head(Xa[idx]), ya[idx]); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            a = (head(Xh).argmax(1) == yh).float().mean().item()
        if a > best[0]:
            best = (a, copy.deepcopy(head.state_dict()))
    head.load_state_dict(best[1]); return head


def gather(Z, get):
    X, y, c, bb = [], [], [], []
    for b in BLOCKS:
        m = Z["lab"][:, b, 0] > 0
        X.append(get(Z, b)[m].astype(np.float32)); y.append((Z["lab"][m, b, 1] * 8 + Z["lab"][m, b, 2]).astype(np.int64)); c.append(np.nonzero(m)[0]); bb.append(np.full(m.sum(), b))
    return np.concatenate(X), np.concatenate(y), np.concatenate(c), np.concatenate(bb)


REPS = {"encoder_1280": lambda Z, b: Z["enc"][:, b],
        "encoder_input_proj_384": lambda Z, b: Z["enc"][:, b].astype(np.float32) @ W.T + bW,
        "rollout_state_384": lambda Z, b: Z["roh"][:, 13, b - 8]}
res = {}
for name, get in REPS.items():
    Xt, yt, ct, bt = gather(TR, get); Xv, yv, cv, bv = gather(VA, get)
    mu, sdv = Xt.mean(0), Xt.std(0) + 1e-6; Xt, Xv = (Xt - mu) / sdv, (Xv - mu) / sdv
    head = fit(Xt, yt, ct, 100)
    with torch.no_grad():
        pv = head(torch.from_numpy(Xv)).argmax(1).numpy()
    pooled = [round(float((pv[bv == b] == yv[bv == b]).mean() * 100), 1) for b in BLOCKS]
    single = []
    for b in BLOCKS:
        m, mv_ = bt == b, bv == b
        mu1, sd1 = (Xt[m] * sdv + mu).mean(0), (Xt[m] * sdv + mu).std(0) + 1e-6
        h1 = fit(((Xt[m] * sdv + mu - mu1) / sd1).astype(np.float32), yt[m], ct[m], 200)
        with torch.no_grad():
            single.append(round(float((h1(torch.from_numpy(((Xv[mv_] * sdv + mu - mu1) / sd1).astype(np.float32))).argmax(1).numpy() == yv[mv_]).mean() * 100), 1))
    res[name] = {"single_block": single, "pooled_blocks": pooled, "n_train_single": int((bt == BLOCKS[0]).sum()), "n_train_pooled": int(len(yt))}
    print(f"{name:24s} 블록 하나 {single} 평균 {np.mean(single):.1f} | 합침 {pooled} 평균 {np.mean(pooled):.1f}", flush=True)
res["val_majority"] = [round(float(np.bincount((VA['lab'][VA['lab'][:, b, 0] > 0, b, 1] * 8 + VA['lab'][VA['lab'][:, b, 0] > 0, b, 2]).astype(int), minlength=48).max()
                                     / (VA['lab'][:, b, 0] > 0).sum() * 100), 1) for b in BLOCKS]
json.dump(res, open("/workspace/world_model/vjepa2/z_analysis/results/identity_1280_vs_384_same_samples.json", "w"), indent=1)
print("최빈 클래스", res["val_majority"])
