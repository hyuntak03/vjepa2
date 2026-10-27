"""자연 영상 AR 출력의 속도 꼴 — 선형 판독기판 (2026-10-26, CPU). ar_output_velocity_form.py 의 후속.
attentive 판독기는 비선형 (토큰 LN · softmax) 이라 "작아졌을 뿐" 과 "다른 꼴" 을 가르기에 덜 깨끗하다. 선형은 크기를 그대로 보존한다:
  출력 = g × (encoder 의 속도 방향 성분) + 다른 것 이면, encoder 로 맞춘 선형 판독기를 출력에 걸었을 때 기울기 ≈ g, 상관은 높다 (작아졌을 뿐).
  상관이 낮은데 출력으로 맞춘 선형 판독기는 잘 읽으면 → 속도가 다른 방향 (다른 꼴) 에 있다. 두 판독기 가중치의 cos 로 직접 비교한다.
입력: 물체 칸 평균 토큰 (1280) — 대상 블록 b 의 정답 물체 칸 (라벨) 에서. encoder = enc[b], AR = tf_out[b−1] (natural_ar_vw_state teacher forcing, 블록 b 예측).
  학습 = ar_layers train 앞 N 편, 시험 = val 350 편. 대상 블록 b = 2–15 중 b−1 · b 둘 다 물체 보임 (tf_output_velocity.py 와 같은 표본 규칙).
판독기: 선형 (1280 → 2), 입력 표준화, AdamW (lr 1e-3, wd 1e-2) 전체 묶음 800 걸음, MSE.
  /venv/vjepa/bin/python z_research/scripts/analysis/ar_output_velocity_linear.py --n_train 1500   → z_analysis/results/ar_output_velocity_linear.json
"""
import argparse, json, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ap = argparse.ArgumentParser(); ap.add_argument("--n_train", type=int, default=1500); a = ap.parse_args()
torch.set_num_threads(32); torch.manual_seed(0)
SRC = "/workspace/auto_research_cache/ar_layers"


def load(split, n):
    meta = json.load(open(f"{SRC}/{split}_meta.json"))[:n]
    E, O, Y = [], [], []
    for m in meta:
        path = f"{SRC}/{split}/{m['i']:05d}.npz"; z = np.load(path); enc, tfo = z["enc"], z["tf_out"]; z.close()
        fd = os.open(path, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
        for b in range(2, 16):
            bl = m["blocks"][b]
            if bl["ok"] and m["blocks"][b - 1]["ok"] and bl["cells"]:
                c = bl["cells"]; E.append(enc[b][c].astype(np.float32).mean(0)); O.append(tfo[b - 1][c].astype(np.float32).mean(0)); Y.append(bl["disp"])
    return np.stack(E), np.stack(O), np.array(Y, np.float64)


Etr, Otr, Ytr = load("train", a.n_train); Eva, Ova, Yva = load("val", 350)
print("표본 train", len(Ytr), "val", len(Yva), flush=True)


def fit(X, Y):
    mu, sd = X.mean(0), X.std(0) + 1e-6; ym, ys = Y.mean(0), Y.std(0)
    Xt = torch.from_numpy((X - mu) / sd).float(); Yt = torch.from_numpy((Y - ym) / ys).float()
    lin = nn.Linear(X.shape[1], 2); opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=1e-2)
    for _ in range(800):
        loss = F.mse_loss(lin(Xt), Yt); opt.zero_grad(); loss.backward(); opt.step()
    W = lin.weight.detach().numpy() / sd[None, :] * ys[:, None]                     # 원래 입력 공간 가중치 (2, 1280)
    b = (lin.bias.detach().numpy() * ys + ym) - W @ mu
    return lambda Xq: Xq @ W.T + b, W


def stats(P, Y):
    out = {}
    for nm, ax in (("x", 0), ("y", 1)):
        p, y = P[:, ax], Y[:, ax]
        out[nm] = {"corr": round(float(np.corrcoef(p, y)[0, 1]), 3), "slope_readout_per_true": round(float(np.polyfit(y, p, 1)[0]), 3),
                   "R2": round(float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()), 3)}
    out["R2_xy"] = round(float(1 - ((P - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()), 3)
    return out


f_enc, W_enc = fit(Etr, Ytr); f_ar, W_ar = fit(Otr, Ytr)
cos = lambda u, v: round(float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v))), 3)
R = {"n_train": int(len(Ytr)), "n_val": int(len(Yva)),
     "encoder_probe_on_encoder": stats(f_enc(Eva), Yva),
     "encoder_probe_on_ar_output": stats(f_enc(Ova), Yva),
     "ar_probe_on_ar_output": stats(f_ar(Ova), Yva),
     "ar_probe_on_encoder": stats(f_ar(Eva), Yva),
     "weight_cos_encoder_vs_ar": {"x": cos(W_enc[0], W_ar[0]), "y": cos(W_enc[1], W_ar[1])}}
# 출력에서 encoder 속도 방향 성분의 크기: 같은 물체 칸에서 (출력 − 평균) 을 encoder 판독기로 읽은 값의 표준편차 / encoder 의 그것
R["encoder_velocity_component_std_ratio_ar_over_enc"] = [round(float(r), 3) for r in (f_enc(Ova).std(0) / f_enc(Eva).std(0))]
print(json.dumps(R, indent=1))
json.dump(R, open("/workspace/world_model/vjepa2/z_analysis/results/ar_output_velocity_linear.json", "w"), indent=1)
