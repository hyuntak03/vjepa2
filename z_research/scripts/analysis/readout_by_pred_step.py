"""예측 걸음별 판독기 이식 — encoder 임베딩의 입력 투영 (N, 384) 으로 학습한 판독기를 예측 블록에 적용 (2026-10-26, CPU).
사용자: "y 축에 점선 하나 (encoder embedding 에서 잰 상한), 나머지는 prediction block test. encoder embedding 1280 → 384 를 head 학습에, predictor output (N, 384) 을 test 로".
모델: 블록 AR natural_ar_vw_state latest. 입력 투영 W_in = predictor_embed (1280 → 384). 출력 = LN(W_out · 상태), 상태 = 마지막 정규화 층 출력 (384).
판독기 학습: 블록 p 의 encoder 토큰 평균 → W_in (N, 384), train 8,000 편 중 물체 보이는 표본, 라벨 = 블록 p. 블록 p = 8 … 15 마다 하나.
시험 (val 350 편), 예측 걸음 k = p − 7:
  상한          = 같은 블록의 encoder 임베딩 → W_in (val)
  predictor 출력 = 블록 p 를 예측한 상태 (384): 걸음 1 = teacher forcing 자리 7 의 마지막 정규화, 걸음 2–8 = 굴림 자리 7+k−1 의 마지막 정규화
  같은 좌표계    = 블록 p 의 굴린 예측 출력 (1280) → W_in
  참고 (encoder 경로의 상태) = teacher forcing 자리 p−1 의 마지막 정규화 (입력이 실제 encoder 블록 p−1 일 때 블록 p 를 예측한 상태)
판독기: 선형층 하나, 입력 표준화 (학습 표본 통계, 판독기의 일부). 모양 × 색 48 클래스 = 교차 엔트로피 · 속도 (블록 안 변위 dx, dy) · 위치 (중심 x, y) = 제곱 오차.
  AdamW lr 1e-3 · weight decay 1e-2 · 묶음 256 · 최대 300 epoch, 학습 클립 20 % 로 epoch 선택. 지표: 정확도 (%) · R² (두 축 합쳐).
  /venv/vjepa/bin/python z_research/scripts/analysis/readout_by_pred_step.py   → z_analysis/results/readout_by_pred_step.json
"""
import copy, json
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(16)
P = "/workspace/auto_research_cache/ar_layers_pooled"
CKPT = "/workspace/world_model/vjepa2/z_training/runs/natural_ar_vw_state/latest.pt"
EPOCHS, BS, LR = 300, 256, 1e-3
sd = torch.load(CKPT, map_location="cpu", weights_only=False)["predictor"]
pick = lambda suf: next(v for k, v in sd.items() if k.endswith(suf)).float().numpy()
W, bW = pick("predictor_embed.weight"), pick("predictor_embed.bias")
proj = lambda E: E.astype(np.float32) @ W.T + bW


def train_head(X, Y, hold, kind, seed=0):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    X, Y = torch.from_numpy(X), torch.from_numpy(Y); h = torch.from_numpy(hold)
    head = torch.nn.Linear(X.shape[1], 48 if kind == "cls" else Y.shape[1]); opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-2)
    Xt, Yt, Xh, Yh = X[~h], Y[~h], X[h], Y[h]; loss_f = F.cross_entropy if kind == "cls" else F.mse_loss
    best = (-1e9, None)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), BS):
            idx = perm[i:i + BS]; loss = loss_f(head(Xt[idx]), Yt[idx]); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            o = head(Xh); sc = (o.argmax(1) == Yh).float().mean().item() if kind == "cls" else -F.mse_loss(o, Yh).item()
        if sc > best[0]:
            best = (sc, copy.deepcopy(head.state_dict()))
    head.load_state_dict(best[1])
    return head


TR, VA = np.load(f"{P}/train.npz"), np.load(f"{P}/val.npz")
enc_t, lab_t = TR["enc"], TR["lab"]; enc_v, tfh_v, roh_v, roo_v, lab_v = VA["enc"], VA["tfh"], VA["roh"], VA["roout"], VA["lab"]
TASKS = {"identity": ("cls", None), "velocity": ("reg", (4, 6)), "position": ("reg", (6, 8))}
SRC = ("upper_bound_encoder", "predictor_output_state", "pred_output_via_input_proj", "encoder_path_state")
res = {t: {s: [None] * 8 for s in SRC} for t in TASKS}; res["identity_val_majority"] = [None] * 8
for k in range(1, 9):
    p = 7 + k; mt, mv = lab_t[:, p, 0] > 0, lab_v[:, p, 0] > 0; clip = np.nonzero(mt)[0]; hold = (clip % 5) == 0
    Xt = proj(enc_t[mt, p]); mu, sdv = Xt.mean(0), Xt.std(0) + 1e-6; Xt = ((Xt - mu) / sdv).astype(np.float32)
    st = tfh_v[mv, 13, 7] if k == 1 else roh_v[mv, 13, k - 2]
    V = {"upper_bound_encoder": proj(enc_v[mv, p]), "predictor_output_state": st.astype(np.float32),
         "pred_output_via_input_proj": proj(roo_v[mv, k - 1]), "encoder_path_state": tfh_v[mv, 13, p - 1].astype(np.float32)}
    V = {s: ((v - mu) / sdv).astype(np.float32) for s, v in V.items()}
    yid_t = lab_t[mt, p, 1].astype(np.int64) * 8 + lab_t[mt, p, 2].astype(np.int64); yid_v = lab_v[mv, p, 1].astype(np.int64) * 8 + lab_v[mv, p, 2].astype(np.int64)
    res["identity_val_majority"][k - 1] = round(float(np.bincount(yid_v, minlength=48).max() / len(yid_v) * 100), 1)
    for t, (kind, cols) in TASKS.items():
        if kind == "cls":
            head = train_head(Xt, yid_t, hold, "cls")
            score = lambda X_: float((head(torch.from_numpy(X_)).argmax(1).numpy() == yid_v).mean() * 100)
        else:
            Yt = lab_t[mt, p, cols[0]:cols[1]].astype(np.float32); ym, ys = Yt.mean(0), Yt.std(0) + 1e-6; Yv = lab_v[mv, p, cols[0]:cols[1]].astype(np.float64)
            head = train_head(Xt, ((Yt - ym) / ys).astype(np.float32), hold, "reg")
            def score(X_, head=head, ym=ym, ys=ys, Yv=Yv):
                Pr = head(torch.from_numpy(X_)).numpy().astype(np.float64) * ys + ym
                return float(1 - ((Pr - Yv) ** 2).sum() / ((Yv - Yv.mean(0)) ** 2).sum())
        with torch.no_grad():
            for s in SRC:
                res[t][s][k - 1] = round(score(V[s]), 3)
    print(f"걸음 {k} (블록 {p})", {t: [res[t][s][k - 1] for s in SRC] for t in TASKS}, flush=True)
json.dump(res, open("/workspace/world_model/vjepa2/z_analysis/results/readout_by_pred_step.json", "w"), indent=1)
