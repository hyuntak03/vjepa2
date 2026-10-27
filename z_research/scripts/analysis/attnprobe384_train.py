"""층별 attentive probe — 블록 AR predictor 은닉 토큰 (256, 384) 위에서 모양×색 (48) · 속도 · 위치를 한 번에, 은닉 자체로 학습 (2026-10-26 저녁, GPU).
사용자: "(층별 그림) attentive probing 으로 진행해줘. shape × color 로 할 수 있잖아".
입력: attnprobe384_extract.py ({split}_{tf,ro}.npy (14, N, 256, 384) · {split}_lab.npy). train 4,000 편 → val 350 편, 걸음 1–7 합쳐 학습, 걸음별 시험.
구조: attnprobe_train.py 와 같고 d = 384 (토큰 LN → + 학습 위치 임베딩 → 교차 attention 16 head, 질의 3 개 → LN → 과제별 선형층 48 · 2 · 2). 손실 = CE + MSE + MSE (라벨 표준화).
학습: AdamW lr 5e-4 · weight decay 0.05 · 묶음 128 · 최대 25 epoch · one-cycle, train 클립 10 % 로 epoch 선택 (정확도 + R² 둘 평균).
  --path enc --layers 0 : 상한 = encoder 임베딩 → predictor 입력 투영 (384) 토큰 = teacher forcing 입력층 − 관측 표시 type[0] (2026-10-26 사용자: "점선을 encoder embedding 으로 읽은 (384) 상한값으로")
  CUDA_VISIBLE_DEVICES=0 MEM_GB=10 /venv/vjepa/bin/python z_research/scripts/analysis/attnprobe384_train.py --path ro --layers 0,1,2,3,4,5,6
출력: z_analysis/results/attnprobe384/{path}_L{layer}.json
"""
import argparse, copy, json, math, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser(); ap.add_argument("--path", choices=["ro", "tf", "enc"], required=True); ap.add_argument("--layers", required=True); ap.add_argument("--epochs", type=int, default=25)
a = ap.parse_args()
D = "/workspace/auto_research_cache/attnprobe384"; OUT = "/workspace/world_model/vjepa2/z_analysis/results/attnprobe384"; os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEM_GB", 10)) * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory, 0)


class AttnProbe(nn.Module):
    def __init__(s, d=384, heads=16, outs=(48, 2, 2)):
        super().__init__()
        s.ln = nn.LayerNorm(d); s.pos = nn.Parameter(torch.randn(256, d) * 0.02); s.q = nn.Parameter(torch.randn(len(outs), d) * 0.02)
        s.attn = nn.MultiheadAttention(d, heads, batch_first=True); s.ln2 = nn.LayerNorm(d); s.heads = nn.ModuleList([nn.Linear(d, o) for o in outs])

    def forward(s, x):
        x = s.ln(x) + s.pos; o, _ = s.attn(s.q.expand(len(x), -1, -1), x, x, need_weights=False); o = s.ln2(o)
        return [h(o[:, i]) for i, h in enumerate(s.heads)]


Lt, Lv = np.load(f"{D}/train_lab.npy"), np.load(f"{D}/val_lab.npy")
vel, pos = Lt[:, 2:4], Lt[:, 4:6]; vm, vs, pm, ps = vel.mean(0), vel.std(0) + 1e-6, pos.mean(0), pos.std(0) + 1e-6
hold = (Lt[:, 6].astype(int) % 10) == 0; tr_idx, ho_idx = np.nonzero(~hold)[0], np.nonzero(hold)[0]
Y_id = torch.from_numpy((Lt[:, 0] * 8 + Lt[:, 1]).astype(np.int64)); Y_v = torch.from_numpy(((vel - vm) / vs).astype(np.float32)); Y_p = torch.from_numpy(((pos - pm) / ps).astype(np.float32))
r2 = lambda P, Y: float(1 - ((P - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())
src = "tf" if a.path == "enc" else a.path
MT, MV = np.load(f"{D}/train_{src}.npy", mmap_mode="r"), np.load(f"{D}/val_{src}.npy", mmap_mode="r")
T0 = None
if a.path == "enc":                                                   # 입력층 − type[0] = predictor_embed(encoder 토큰)
    _sd = torch.load("/workspace/world_model/vjepa2/z_training/runs/natural_ar_vw_state/latest.pt", map_location="cpu", weights_only=False)["predictor"]
    T0 = next(v for k, v in _sd.items() if k.endswith("type_embed")).float().reshape(2, 384)[0].numpy(); del _sd

for L in [int(x) for x in a.layers.split(",")]:
    torch.manual_seed(0)
    Xt = np.ascontiguousarray(MT[L]); Xv = np.ascontiguousarray(MV[L])
    if T0 is not None:
        Xt = (Xt.astype(np.float32) - T0).astype(np.float16); Xv = (Xv.astype(np.float32) - T0).astype(np.float16)
    Xt = torch.from_numpy(Xt)

    def predict(X):
        model.eval(); outs = [[], [], []]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(X), 512):
                o = model(torch.as_tensor(X[i:i + 512]).to(dev).float())
                for k in range(3):
                    outs[k].append(o[k].float().cpu())
        model.train(); return [torch.cat(o).numpy() for o in outs]

    def scores(X, Lb):
        pid, pv, pp = predict(X); a_ = pid.argmax(1); yid = (Lb[:, 0] * 8 + Lb[:, 1]).astype(int)
        return {"shape_x_color": round(float((a_ == yid).mean() * 100), 1), "shape": round(float((a_ // 8 == Lb[:, 0]).mean() * 100), 1), "color": round(float((a_ % 8 == Lb[:, 1]).mean() * 100), 1),
                "velocity_r2": round(r2(pv * vs + vm, Lb[:, 2:4]), 3), "position_r2": round(r2(pp * ps + pm, Lb[:, 4:6]), 3), "n": int(len(Lb))}

    model = AttnProbe().to(dev); opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-4, total_steps=a.epochs * math.ceil(len(tr_idx) / 128), pct_start=0.05)
    best = (-1e9, None, -1)
    for ep in range(a.epochs):
        perm = tr_idx[np.random.default_rng(ep).permutation(len(tr_idx))]
        for i in range(0, len(perm), 128):
            b = torch.from_numpy(np.sort(perm[i:i + 128]))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                oid, ov, op = model(Xt[b].to(dev).float())
                loss = F.cross_entropy(oid.float(), Y_id[b].to(dev)) + F.mse_loss(ov.float(), Y_v[b].to(dev)) + F.mse_loss(op.float(), Y_p[b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        s = scores(Xt[torch.from_numpy(ho_idx)].numpy(), Lt[ho_idx]); sc = (s["shape_x_color"] / 100 + s["velocity_r2"] + s["position_r2"]) / 3
        if sc > best[0]:
            best = (sc, copy.deepcopy(model.state_dict()), ep)
    model.load_state_dict(best[1])
    res = {"path": a.path, "layer": L, "best_epoch": best[2], "n_train": int(len(tr_idx)), "val_all": scores(Xv, Lv),
           "val_by_step": {k: scores(Xv[Lv[:, 7] == k], Lv[Lv[:, 7] == k]) for k in range(1, 8)}}
    json.dump(res, open(f"{OUT}/{a.path}_L{L}.json", "w"), indent=1)
    print(a.path, "층", L, "epoch", best[2], res["val_all"], flush=True)
    del Xt, Xv, model
