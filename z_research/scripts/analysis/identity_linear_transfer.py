"""encoder 임베딩으로 학습한 선형 판독기 (모양 · 색) 를 predictor 예측에 그대로 적용 — 이식되나 (2026-10-26, CPU).
사용자: "ridge 회귀 하지 말고 linear 로만 하자. head 를 encoder embedding 으로 학습하고 predictor 에 test. transfer 되는지 보자".
입력: 블록마다 256 칸 평균 (1, 1280), 표준화 없이 그대로. 물체 보이는 블록만.
판독기: 선형층 하나 (1280 → 6 또는 8, 치우침 포함), 교차 엔트로피, AdamW (lr 1e-2, weight decay 1e-2), 묶음 256, 최대 300 epoch.
  (입력은 거의 공통 평균 벡터 (크기 21.1) 에 붙어 있고 차원당 표준편차 0.12 라 lr 1e-3 · 100 epoch 은 덜 수렴 — 블록 8 모양 hold 75.5 → 이 설정 80.4)
  학습 클립의 20 % (클립 번호 % 5 == 0) 를 떼어 두고 그 정확도가 가장 높은 epoch 의 판독기를 쓴다 (encoder 판독기는 encoder 표본으로만 고른다).
  예측 블록 b = 7 + k (걸음 k = 1–8) 마다 따로 학습. train 8,000 편 → 시험 val 350 편.
적용: encoder → encoder (상한) · **encoder → 예측 (질문)** · 예측 → 예측 (예측으로 직접 학습, 정보가 있는가).
모델: AR = 굴린 예측 출력 vs 블록 단위 encoder (ar_layers_pooled) / 릴리즈 = 한 번에 8 블록 예측 vs 32 장 encoder (release_pooled).
진단 (판독기는 그대로, 예측 입력만 옮김): 평균만 맞춤 = 예측 − 예측 train 평균 + encoder train 평균 /
  평균 · 표준편차 맞춤 = (예측 − 예측 평균) / 예측 표준편차 × encoder 표준편차 + encoder 평균 (차원마다).
기준선: val 최빈 클래스 비율. encoder 판독기가 예측에서 한 클래스로 몰리는지 (가장 많이 낸 답의 비율) 도 같이.
  /venv/vjepa/bin/python z_research/scripts/analysis/identity_linear_transfer.py   → z_analysis/results/identity_linear_transfer.json
"""
import copy, json
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(16)
P = "/workspace/auto_research_cache"
EPOCHS, BS, LR = 300, 256, 1e-2


def train_head(X, y, clip, K, seed=0):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    X, y = torch.from_numpy(X), torch.from_numpy(y); hold = torch.from_numpy((clip % 5) == 0)
    Xt, yt, Xh, yh = X[~hold], y[~hold], X[hold], y[hold]
    head = torch.nn.Linear(X.shape[1], K); opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-2)
    best = (-1.0, None, -1)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), BS):
            idx = perm[i:i + BS]; loss = F.cross_entropy(head(Xt[idx]), yt[idx]); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            a = (head(Xh).argmax(1) == yh).float().mean().item()
        if a > best[0]:
            best = (a, copy.deepcopy(head.state_dict()), ep)
    head.load_state_dict(best[1])
    return head, best[2]


def evaluate(head, X, y):
    with torch.no_grad():
        p = head(torch.from_numpy(X)).argmax(1).numpy()
    return round(float((p == y).mean() * 100), 1), round(float(np.bincount(p).max() / len(p) * 100), 1)


def load(split):
    L = np.load(f"{P}/ar_layers_pooled/{split}.npz"); enc, ro, lab, ii = L["enc"], L["roout"], L["lab"], L["i"]
    R = np.load(f"{P}/release_pooled/{split}.npz"); o = {int(v): n for n, v in enumerate(ii)}; sel = [o[int(v)] for v in R["i"]]
    return {"ar": (enc, ro, lab), "release": (R["gt"], R["pred"], lab[sel])}


TRD, VAD = load("train"), load("val")
out = {}
for model in ("ar", "release"):
    (GT, PR, LT), (GV, PV, LV) = TRD[model], VAD[model]; out[model] = {}
    for fam, col, K in (("shape", 1, 6), ("color", 2, 8)):
        out[model][fam] = {}
        for k in range(1, 9):
            b = 7 + k; mt = LT[:, b, 0] > 0; mv = LV[:, b, 0] > 0
            yt, yv, clip = LT[mt, b, col].astype(np.int64), LV[mv, b, col].astype(np.int64), np.nonzero(mt)[0]
            Et, Pt = GT[mt, b].astype(np.float32), PR[mt, k - 1].astype(np.float32)
            Ev, Pv = GV[mv, b].astype(np.float32), PV[mv, k - 1].astype(np.float32)
            he, ep_e = train_head(Et, yt, clip, K); hp, ep_p = train_head(Pt, yt, clip, K)
            ee, _ = evaluate(he, Ev, yv); ep_acc, ep_mode = evaluate(he, Pv, yv); pp, _ = evaluate(hp, Pv, yv)
            mE, sE, mP, sP = Et.mean(0), Et.std(0) + 1e-6, Pt.mean(0), Pt.std(0) + 1e-6
            ms_acc, ms_mode = evaluate(he, Pv - mP + mE, yv); rs_acc, rs_mode = evaluate(he, (Pv - mP) / sP * sE + mE, yv)
            out[model][fam][k] = {"n_val": int(mv.sum()), "val_majority": round(float(np.bincount(yv, minlength=K).max() / len(yv) * 100), 1),
                                  "enc_head_on_enc": ee, "enc_head_on_pred": ep_acc, "enc_head_on_pred_top_answer_share": ep_mode,
                                  "enc_head_on_pred_meanshift": ms_acc, "meanshift_top_answer_share": ms_mode, "enc_head_on_pred_restd": rs_acc, "restd_top_answer_share": rs_mode,
                                  "pred_head_on_pred": pp, "epoch_enc_head": ep_e, "epoch_pred_head": ep_p,
                                  "feat_norm_enc": round(float(np.linalg.norm(Ev, axis=1).mean()), 3), "feat_norm_pred": round(float(np.linalg.norm(Pv, axis=1).mean()), 3)}
        print(model, fam, flush=True)
        for key in ("val_majority", "enc_head_on_enc", "enc_head_on_pred", "enc_head_on_pred_top_answer_share", "enc_head_on_pred_meanshift", "meanshift_top_answer_share", "enc_head_on_pred_restd", "restd_top_answer_share", "pred_head_on_pred"):
            print(f"   {key:34s}", [out[model][fam][k][key] for k in range(1, 9)])
json.dump(out, open("/workspace/world_model/vjepa2/z_analysis/results/identity_linear_transfer.json", "w"), indent=1)
