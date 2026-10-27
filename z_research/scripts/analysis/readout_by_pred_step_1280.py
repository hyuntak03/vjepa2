"""예측 걸음별 판독기 이식, 1280 공간 — encoder 임베딩 (1280) 으로 학습한 선형 판독기를 predictor 출력 (1280) 에 그대로 적용 (2026-10-26, CPU).
사용자: "그래프를 1280 에서 비교하는 게 맞을 듯. encoder embedding (1280), predictor output (1280), prediction step 을 x 축으로".
모델: 블록 AR natural_ar_vw_state latest. 블록마다 256 칸 평균 (1, 1280), ar_layers_pooled.
판독기 학습: 블록 p (8–15) 의 encoder 임베딩 (블록 단위 LN(target_encoder)), train 8,000 편 중 물체 보이는 표본, 라벨 = 블록 p. 블록마다 하나.
시험 (val 350 편), 예측 걸음 k = p − 7: 상한 = 같은 블록 encoder 임베딩 · 예측 = 굴린 예측 출력 (블록 p 예측, 1280).
판독기: 선형층 하나, 입력은 encoder 학습 표본 평균 · 표준편차로 표준화 (판독기의 일부, 예측에도 그대로).
  모양 × 색 48 클래스 = 교차 엔트로피 · 속도 (블록 안 변위 dx, dy, px) · 위치 (중심 x, y, px) = 제곱 오차 (라벨 표준화).
  AdamW lr 1e-3 · weight decay 1e-2 · 묶음 256 · 최대 300 epoch, 학습 클립 20 % 로 epoch 선택. 지표: 정확도 (%) · R² (두 축 합쳐 1 − SSE / SST).
  /venv/vjepa/bin/python z_research/scripts/analysis/readout_by_pred_step_1280.py   → z_analysis/results/readout_by_pred_step_1280.json
"""
import copy, json
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(16)
P = "/workspace/auto_research_cache/ar_layers_pooled"
EPOCHS, BS, LR = 300, 256, 1e-3


def train_head(X, Y, hold, kind, seed=0):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    X, Y = torch.from_numpy(X), torch.from_numpy(Y); h = torch.from_numpy(hold)
    head = torch.nn.Linear(X.shape[1], 48 if kind == "cls" else Y.shape[1]); opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-2)
    Xt, Yt, Xh, Yh = X[~h], Y[~h], X[h], Y[h]; loss_f = F.cross_entropy if kind == "cls" else F.mse_loss
    best = (-1e9, None, -1)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), BS):
            idx = perm[i:i + BS]; loss = loss_f(head(Xt[idx]), Yt[idx]); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            o = head(Xh); sc = (o.argmax(1) == Yh).float().mean().item() if kind == "cls" else -F.mse_loss(o, Yh).item()
        if sc > best[0]:
            best = (sc, copy.deepcopy(head.state_dict()), ep)
    head.load_state_dict(best[1])
    return head, best[2]


TR, VA = np.load(f"{P}/train.npz"), np.load(f"{P}/val.npz")
enc_t, lab_t = TR["enc"], TR["lab"]; enc_v, roo_v, lab_v = VA["enc"], VA["roout"], VA["lab"]
TASKS = {"identity": ("cls", None), "velocity": ("reg", (4, 6)), "position": ("reg", (6, 8))}
res = {t: {"upper_bound_encoder": [None] * 8, "predictor_output": [None] * 8, "epoch": [None] * 8} for t in TASKS}
res["identity_val_majority"] = [None] * 8; res["identity_pred_top_answer_share"] = [None] * 8; res["n_val"] = [None] * 8
for k in range(1, 9):
    p = 7 + k; mt, mv = lab_t[:, p, 0] > 0, lab_v[:, p, 0] > 0; clip = np.nonzero(mt)[0]; hold = (clip % 5) == 0
    Xt = enc_t[mt, p].astype(np.float32); mu, sdv = Xt.mean(0), Xt.std(0) + 1e-6; Xt = (Xt - mu) / sdv
    Ve = (enc_v[mv, p].astype(np.float32) - mu) / sdv; Vp = (roo_v[mv, k - 1].astype(np.float32) - mu) / sdv
    yid_t = lab_t[mt, p, 1].astype(np.int64) * 8 + lab_t[mt, p, 2].astype(np.int64); yid_v = lab_v[mv, p, 1].astype(np.int64) * 8 + lab_v[mv, p, 2].astype(np.int64)
    res["identity_val_majority"][k - 1] = round(float(np.bincount(yid_v, minlength=48).max() / len(yid_v) * 100), 1); res["n_val"][k - 1] = int(mv.sum())
    for t, (kind, cols) in TASKS.items():
        if kind == "cls":
            head, ep = train_head(Xt, yid_t, hold, "cls")
            with torch.no_grad():
                pe, pp = head(torch.from_numpy(Ve)).argmax(1).numpy(), head(torch.from_numpy(Vp)).argmax(1).numpy()
            sc = [float((pe == yid_v).mean() * 100), float((pp == yid_v).mean() * 100)]
            res["identity_pred_top_answer_share"][k - 1] = round(float(np.bincount(pp).max() / len(pp) * 100), 1)
        else:
            Yt = lab_t[mt, p, cols[0]:cols[1]].astype(np.float32); ym, ys = Yt.mean(0), Yt.std(0) + 1e-6; Yv = lab_v[mv, p, cols[0]:cols[1]].astype(np.float64)
            head, ep = train_head(Xt, ((Yt - ym) / ys).astype(np.float32), hold, "reg")
            sc = []
            with torch.no_grad():
                for V in (Ve, Vp):
                    Pr = head(torch.from_numpy(V)).numpy().astype(np.float64) * ys + ym
                    sc.append(float(1 - ((Pr - Yv) ** 2).sum() / ((Yv - Yv.mean(0)) ** 2).sum()))
        res[t]["upper_bound_encoder"][k - 1], res[t]["predictor_output"][k - 1], res[t]["epoch"][k - 1] = round(sc[0], 3), round(sc[1], 3), int(ep)
    print(f"걸음 {k} (블록 {p})", {t: (res[t]["upper_bound_encoder"][k - 1], res[t]["predictor_output"][k - 1], res[t]["epoch"][k - 1]) for t in TASKS},
          "한 답 비율", res["identity_pred_top_answer_share"][k - 1], flush=True)
json.dump(res, open("/workspace/world_model/vjepa2/z_analysis/results/readout_by_pred_step_1280.json", "w"), indent=1)
