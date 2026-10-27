"""attentive probe 하나로 모양×색 (48) · 속도 (dx, dy) · 위치 (x, y) 를 한 번에 — encoder 토큰 / AR 예측 출력 토큰 (2026-10-26, GPU 1 장).
사용자: "attentive probe 면 color × shape 한 번에, velocity, position head 이렇게 3 개 하던가 다 읽을 수 있도록".
입력: attnprobe_extract.py 출력 (블록 하나의 토큰 (256, 1280), 물체 보이는 블록). train 4,000 편 → val 350 편.
구조: 토큰 LN → + 학습 위치 임베딩 (256, 1280) → 교차 attention (16 head, 질의 3 개 = 과제마다 하나, k · v 공유) → LN → 과제별 선형층 (48 · 2 · 2).
  매개변수 약 6.9 M. 손실 = 교차 엔트로피 (모양×색) + 제곱 오차 (속도 · 위치, 라벨은 학습 표본 평균 · 표준편차로 표준화).
학습: AdamW lr 3e-4 · weight decay 0.05 · 묶음 128 · 최대 30 epoch · cosine, train 클립 10 % 로 epoch 선택 (세 과제 점수 평균: 정확도 + R² 둘).
  --train enc : encoder 토큰으로 학습 → encoder val · AR 예측 출력 val (걸음별) 에 적용
  --train ro  : AR 예측 출력 토큰으로 학습 → AR 예측 출력 val (걸음별)
지표: 모양×색 정확도 (%) · 모양만 · 색만 (48 클래스 답에서 분리) · 속도 R² · 위치 R² (두 축 합쳐 1 − SSE / SST).
  CUDA_VISIBLE_DEVICES=0 MEM_GB=16 /venv/vjepa/bin/python z_research/scripts/analysis/attnprobe_train.py --train enc
출력: z_analysis/results/attnprobe_{train}.json
"""
import argparse, copy, json, math, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser(); ap.add_argument("--train", choices=["enc", "ro"], default="enc"); ap.add_argument("--epochs", type=int, default=30); a = ap.parse_args()
D = "/workspace/auto_research_cache/attnprobe"
dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEM_GB", 16)) * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory, 0)
torch.manual_seed(0)


class AttnProbe(nn.Module):
    def __init__(s, d=1280, heads=16, outs=(48, 2, 2)):
        super().__init__()
        s.ln = nn.LayerNorm(d); s.pos = nn.Parameter(torch.randn(256, d) * 0.02); s.q = nn.Parameter(torch.randn(len(outs), d) * 0.02)
        s.attn = nn.MultiheadAttention(d, heads, batch_first=True); s.ln2 = nn.LayerNorm(d); s.heads = nn.ModuleList([nn.Linear(d, o) for o in outs])

    def forward(s, x):
        x = s.ln(x) + s.pos; o, w = s.attn(s.q.expand(len(x), -1, -1), x, x, need_weights=True); o = s.ln2(o)
        return [h(o[:, i]) for i, h in enumerate(s.heads)], w


def load(split, kind):
    return np.load(f"{D}/{split}_{kind}_tok.npy", mmap_mode="r"), np.load(f"{D}/{split}_{kind}_lab.npy")


Xt, Lt = load("train", a.train)
yid_t = (Lt[:, 0] * 8 + Lt[:, 1]).astype(np.int64); vel_t, pos_t = Lt[:, 2:4], Lt[:, 4:6]
vm, vs, pm, ps = vel_t.mean(0), vel_t.std(0) + 1e-6, pos_t.mean(0), pos_t.std(0) + 1e-6
hold = (Lt[:, 6].astype(int) % 10) == 0
tr_idx, ho_idx = np.nonzero(~hold)[0], np.nonzero(hold)[0]
Xt_t = torch.from_numpy(np.ascontiguousarray(Xt))                     # fp16, CPU
T = lambda arr: torch.from_numpy(np.ascontiguousarray(arr))
Y_id, Y_v, Y_p = T(yid_t), T(((vel_t - vm) / vs).astype(np.float32)), T(((pos_t - pm) / ps).astype(np.float32))
model = AttnProbe().to(dev); opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
steps = a.epochs * math.ceil(len(tr_idx) / 128); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps, pct_start=0.05)
print(f"학습 {a.train}: 표본 {len(tr_idx)} (+ 고르기 {len(ho_idx)}), 매개변수 {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M", flush=True)


def predict(X, bs=256):
    model.eval(); outs = [[], [], []]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(X), bs):
            o, _ = model(torch.from_numpy(np.ascontiguousarray(X[i:i + bs])).to(dev).float())
            for k in range(3):
                outs[k].append(o[k].float().cpu())
    model.train()
    return [torch.cat(o).numpy() for o in outs]


def scores(X, L):
    pid, pv, pp = predict(X); yid = (L[:, 0] * 8 + L[:, 1]).astype(int); a_ = pid.argmax(1)
    r2 = lambda P, Y: float(1 - ((P - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())
    return {"shape_x_color": round(float((a_ == yid).mean() * 100), 1), "shape": round(float((a_ // 8 == L[:, 0]).mean() * 100), 1), "color": round(float((a_ % 8 == L[:, 1]).mean() * 100), 1),
            "velocity_r2": round(r2(pv * vs + vm, L[:, 2:4]), 3), "position_r2": round(r2(pp * ps + pm, L[:, 4:6]), 3), "n": int(len(L))}


best = (-1e9, None, -1)
for ep in range(a.epochs):
    perm = tr_idx[np.random.default_rng(ep).permutation(len(tr_idx))]
    for i in range(0, len(perm), 128):
        b = torch.from_numpy(np.sort(perm[i:i + 128]))
        x = Xt_t[b].to(dev, non_blocking=True).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            (oid, ov, op), _ = model(x)
            loss = F.cross_entropy(oid.float(), Y_id[b].to(dev)) + F.mse_loss(ov.float(), Y_v[b].to(dev)) + F.mse_loss(op.float(), Y_p[b].to(dev))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    s = scores(Xt[ho_idx], Lt[ho_idx]); sc = (s["shape_x_color"] / 100 + s["velocity_r2"] + s["position_r2"]) / 3
    print(f"epoch {ep}: 손실 {loss.item():.3f} · 고르기 {s}", flush=True)
    if sc > best[0]:
        best = (sc, copy.deepcopy(model.state_dict()), ep)
model.load_state_dict(best[1])
res = {"train_on": a.train, "best_epoch": best[2], "n_train": int(len(tr_idx))}
if a.train == "enc":
    Xv, Lv = load("val", "enc"); res["test_encoder_val"] = scores(Xv, Lv)
Xr, Lr = load("val", "ro")
res["test_rollout_val_all"] = scores(Xr, Lr)
res["test_rollout_val_by_step"] = {int(b) - 7: scores(Xr[Lr[:, 7] == b], Lr[Lr[:, 7] == b]) for b in range(8, 16)}
if a.train == "enc":
    Xe, Le = load("val", "enc"); m = Le[:, 7] >= 8
    res["test_encoder_val_blocks8to15_by_step"] = {int(b) - 7: scores(Xe[Le[:, 7] == b], Le[Le[:, 7] == b]) for b in range(8, 16)}
json.dump(res, open(f"/workspace/world_model/vjepa2/z_analysis/results/attnprobe_{a.train}.json", "w"), indent=1)
torch.save(model.state_dict(), f"{D}/attnprobe_{a.train}.pt")
print(json.dumps({k: v for k, v in res.items() if "by_step" not in k}, indent=1))
for k in [k for k in res if "by_step" in k]:
    print(k); [print("  ", st, v) for st, v in res[k].items()]
