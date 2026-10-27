"""층별 · 예측 블록별 판독기 이식 — 판독기는 encoder 임베딩의 predictor 입력 투영 (1280 → 384) 하나로만 학습하고,
예측이 시작되는 블록부터 predictor 은닉 (층 0–13) 에 그대로 적용한다 (2026-10-26, CPU).
사용자: "판독기 학습은 입력 임베딩만 (N, 384) {encoder → projection}. 미래 예측하기 시작하는 부분부터 layer 별로 block 별로 평가".
모델: 블록 AR natural_ar_vw_state latest. 입력 투영 = predictor_embed (Linear 1280 → 384). 관측 입력 = 투영 + type[0], 굴림 입력 = 상태 + type[1].
표현 (블록마다 256 칸 평균; 투영은 선형이라 평균 뒤에 투영해도 같다):
  학습 = 블록 p 의 encoder 토큰 평균 → 입력 투영 (N, 384)  (표시 type 없음), train 8,000 편 중 물체 보이는 표본, 라벨 = 블록 p
  시험 (val 350 편), 블록 p = 8 … 14 (예측 걸음 1 … 7; 걸음 8 = 블록 15 는 다음 입력이 없어 은닉이 없다):
    굴린 예측 = 굴림 자리 p 의 층 L 은닉 (입력 = 블록 p 를 예측한 상태). 층 0 은 표시를 뺀 입력 (= 되먹인 상태)
    참고 (encoder 입력) = teacher forcing 자리 p 의 층 L 은닉 (입력 = 실제 encoder 블록 p). 층 0 은 표시를 뺀 입력 (= 투영 자체, 상한)
  층 L = 0 입력 · 1–12 predictor 층 · 13 마지막 정규화 (= 상태).
  같은 공간 비교 (2026-10-26 추가): 굴린 예측 출력 (블록 p 예측, 1280) 을 같은 입력 투영에 통과시킨 것 — 판독기 학습 공간과 같은 공간
판독기: 선형층 하나, 블록 · 과제마다 하나 (층마다 다시 학습하지 않는다). 입력은 학습 표본 평균 · 표준편차로 표준화 (판독기의 일부).
  모양 × 색 48 클래스 = 교차 엔트로피 · 속도 (블록 안 변위 dx, dy, px) · 위치 (중심 x, y, px) = 제곱 오차 (라벨 표준화).
  AdamW lr 1e-3 · weight decay 1e-2 · 묶음 256 · 최대 300 epoch, 학습 클립 20 % 로 epoch 선택.
지표: 모양 × 색 정확도 (%) · 속도 · 위치 R² (두 축 합쳐 1 − SSE / SST).
  /venv/vjepa/bin/python z_research/scripts/analysis/layer_step_readout_transfer.py   → z_analysis/results/layer_step_readout_transfer.json
"""
import copy, json, os, time
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(16)
P = "/workspace/auto_research_cache/ar_layers_pooled"
CKPT = "/workspace/world_model/vjepa2/z_training/runs/natural_ar_vw_state/latest.pt"
EPOCHS, BS, LR = 300, 256, 1e-3
LAYERS = ["input"] + [f"L{i}" for i in range(1, 13)] + ["norm"]

sd = torch.load(CKPT, map_location="cpu", weights_only=False)["predictor"]
pick = lambda suf: next(v for k, v in sd.items() if k.endswith(suf)).float().numpy()
W, bW, TE = pick("predictor_embed.weight"), pick("predictor_embed.bias"), pick("type_embed").reshape(2, 384)
proj = lambda E: E.astype(np.float32) @ W.T + bW


def train_head(X, Y, hold, kind, seed=0):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    X, Y = torch.from_numpy(X), torch.from_numpy(Y); h = torch.from_numpy(hold)
    out = 48 if kind == "cls" else Y.shape[1]
    head = torch.nn.Linear(X.shape[1], out); opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-2)
    Xt, Yt, Xh, Yh = X[~h], Y[~h], X[h], Y[h]
    loss_f = F.cross_entropy if kind == "cls" else F.mse_loss
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
enc_t, lab_t = TR["enc"], TR["lab"]; enc_v, tfh_v, roh_v, roo_v, lab_v = VA["enc"], VA["tfh"], VA["roh"], VA["roout"], VA["lab"]
chk = float(np.abs(proj(enc_v[:, 8]) + TE[0] - tfh_v[:, 0, 8].astype(np.float32)).max())          # 투영 + type[0] = 캐시의 층 0 인가
chk2 = float(np.abs(roh_v[:, 0, 1].astype(np.float32) - TE[1] - roh_v[:, 13, 0].astype(np.float32)).max())   # 굴림 층 0 − type[1] = 앞 자리 상태인가
print(f"점검: |투영 + type0 − 층0| 최대 {chk:.4f} · |굴림 층0 − type1 − 앞 자리 상태| 최대 {chk2:.4f}", flush=True)
TASKS = {"identity": ("cls", None), "velocity": ("reg", (4, 6)), "position": ("reg", (6, 8))}
res = {t: {"encoder_input_path": [[None] * 7 for _ in range(14)], "rollout": [[None] * 7 for _ in range(14)], "rollout_output_projected": [None] * 7} for t in TASKS}
meta = {"check_proj_plus_type0_vs_layer0": round(chk, 4), "check_rollout_layer0_minus_type1_vs_prev_state": round(chk2, 4), "blocks": {}}
t0 = time.time()
for k in range(1, 8):
    p = 7 + k; mt, mv = lab_t[:, p, 0] > 0, lab_v[:, p, 0] > 0; clip = np.nonzero(mt)[0]; hold = (clip % 5) == 0
    Xt = proj(enc_t[mt, p]); mu, sdv = Xt.mean(0), Xt.std(0) + 1e-6; Xt = ((Xt - mu) / sdv).astype(np.float32)
    yid_t = lab_t[mt, p, 1].astype(np.int64) * 8 + lab_t[mt, p, 2].astype(np.int64); yid_v = lab_v[mv, p, 1].astype(np.int64) * 8 + lab_v[mv, p, 2].astype(np.int64)
    meta["blocks"][k] = {"block": p, "n_train": int(mt.sum()), "n_val": int(mv.sum()), "identity_val_majority": round(float(np.bincount(yid_v, minlength=48).max() / len(yid_v) * 100), 1)}
    tests = {}
    for L in range(14):
        tf = proj(enc_v[mv, p]) if L == 0 else tfh_v[mv, L, p].astype(np.float32)
        ro = roh_v[mv, 0, k - 1].astype(np.float32) - TE[1] if L == 0 else roh_v[mv, L, k - 1].astype(np.float32)
        tests[L] = {"encoder_input_path": ((tf - mu) / sdv).astype(np.float32), "rollout": ((ro - mu) / sdv).astype(np.float32)}
    for t, (kind, cols) in TASKS.items():
        if kind == "cls":
            head, ep = train_head(Xt, yid_t, hold, "cls")
            score = lambda V: float((head(torch.from_numpy(V)).argmax(1).numpy() == yid_v).mean() * 100)
        else:
            Yt = lab_t[mt, p, cols[0]:cols[1]].astype(np.float32); ym, ys = Yt.mean(0), Yt.std(0) + 1e-6
            Yv = lab_v[mv, p, cols[0]:cols[1]].astype(np.float64)
            head, ep = train_head(Xt, ((Yt - ym) / ys).astype(np.float32), hold, "reg")
            def score(V, head=head, ym=ym, ys=ys, Yv=Yv):
                Pr = head(torch.from_numpy(V)).numpy().astype(np.float64) * ys + ym
                return float(1 - ((Pr - Yv) ** 2).sum() / ((Yv - Yv.mean(0)) ** 2).sum())
        meta["blocks"][k][f"epoch_{t}"] = int(ep)
        with torch.no_grad():
            res[t]["rollout_output_projected"][k - 1] = round(score(((proj(roo_v[mv, k - 1]) - mu) / sdv).astype(np.float32)), 3)
            for L in range(14):
                for path in ("encoder_input_path", "rollout"):
                    res[t][path][L][k - 1] = round(score(tests[L][path]), 3)
    print(f"블록 {p} (걸음 {k}) 끝 ({time.time() - t0:.0f}s)", {t: (res[t]["encoder_input_path"][0][k - 1], res[t]["rollout"][0][k - 1], res[t]["rollout"][13][k - 1]) for t in TASKS},
          "epochs", [meta["blocks"][k][f"epoch_{t}"] for t in TASKS], flush=True)
json.dump({"layers": LAYERS, "steps": list(range(1, 8)), "meta": meta, **res}, open("/workspace/world_model/vjepa2/z_analysis/results/layer_step_readout_transfer.json", "w"), indent=1)
for t in TASKS:
    print(t)
    print(f"   예측 출력 → 입력 투영 {res[t]['rollout_output_projected']}")
    for L in range(14):
        print(f"   {LAYERS[L]:5s} enc-input {res[t]['encoder_input_path'][L]}  rollout {res[t]['rollout'][L]}")
