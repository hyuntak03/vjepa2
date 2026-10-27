"""굴림에서 걸음마다 이어 간 몫의 감소 비율 p — 학습 데이터가 다른 블록 AR 세 판 비교 (2026-10-27, CPU).
사용자: "해 보던가" (rollout_geometric_decay.json 의 후속: 자연 영상 AR 은 걸음 1–3 에서 걸음마다 약 절반 (p ≈ 0.5), 걸음 4 부터 0).
모델 (구조 · 손실 같음, 데이터만 다름): natural = natural_ar_vw_state · mix = natural_ar_vw_state_phys3d_event (자연 + 3D 합성 사건 25 %) ·
  synthetic = natural_ar_vw_state_phys3d_only (3D 합성 물체 하나만, 시험과 같은 분포).
입력: ar_layers 캐시의 encoder 블록 토큰 (enc, 블록 단위 LN) — encoder 를 다시 돌리지 않는다. 자연 AR 은 캐시된 출력 (ro_out · tf_out) 을 쓴다.
판독기: 모델마다 그 모델의 굴림 출력 (train 앞 N 편, 걸음 1–8 묶음, 물체 보이는 블록) 으로 attentive probe (속도 · 위치) 를 새로 학습
  (attnprobe_train.py 설정: AdamW 3e-4 · wd 0.05 · 묶음 128 · OneCycle, epoch 고정) — 세 모델 모두 같은 절차라 자연 AR 의 값도 다시 잰다.
지표 (val 350 편, 걸음 k = 1 … 8): 위치 증분 이득 · 블록 안 속도 이득 (rollout_geometric_decay.py 와 같은 정의), 굴림과 teacher forcing 각각.
  기하 맞춤: 굴림 위치 증분 이득의 걸음 1 → 2 → 3 비율 (p12 = g2 / g1, p23 = g3 / g2).
  /venv/vjepa/bin/python z_research/scripts/analysis/rollout_decay_models.py --n_train 1200   → z_analysis/results/rollout_decay_models.json
"""
import argparse, json, math, os, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ap = argparse.ArgumentParser(); ap.add_argument("--n_train", type=int, default=1200); ap.add_argument("--epochs", type=int, default=15); a = ap.parse_args()
torch.set_num_threads(48)
REPO = "/workspace/world_model/vjepa2"; sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "auto_research/scripts"))
_argv = list(sys.argv); sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT
sys.argv = _argv
SRC = "/workspace/auto_research_cache/ar_layers"; S, D, NB, C, K = 256, 1280, 16, 8, 8
MODELS = {"natural": None, "mix": "natural_ar_vw_state_phys3d_event", "synthetic": "natural_ar_vw_state_phys3d_only"}
ARS = {k: (FT.load_pred(v, "latest.pt").float().eval() if v else None) for k, v in MODELS.items()}
idx_all = torch.arange(NB * S).unsqueeze(0)


def outputs(name, z):
    """→ ro (8, 256, D), tf (8, 256, D) = 블록 8–15 예측 (굴림 / teacher forcing)."""
    if ARS[name] is None:
        return z["ro_out"].astype(np.float32), z["tf_out"][7:15].astype(np.float32)
    h = torch.from_numpy(z["enc"].astype(np.float32))
    with torch.no_grad():
        ro = ARS[name].rollout_kv(h[:C].reshape(1, C * S, D), idx_all[:, :C * S], K)[0].view(K, S, D)
        tf = ARS[name].forward_seq(h.reshape(1, -1, D), idx_all)[0].view(NB, S, D)[7:15]
    return ro.numpy(), tf.numpy()


def collect(split, n):
    meta = json.load(open(f"{SRC}/{split}_meta.json"))[:n]
    out = {k: {"ro": [], "tf": []} for k in MODELS}; lab = []
    for j, m in enumerate(meta):
        B = m["blocks"]
        if not B[7]["ok"]:
            continue
        path = f"{SRC}/{split}/{m['i']:05d}.npz"; z = np.load(path); zz = {k: z[k] for k in ("enc", "ro_out", "tf_out")}; z.close()
        fd = os.open(path, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
        sel = [k for k in range(1, 9) if B[7 + k]["ok"]]
        if not sel:
            continue
        for name in MODELS:
            ro, tf = outputs(name, zz)
            out[name]["ro"].append(ro[[k - 1 for k in sel]].astype(np.float16)); out[name]["tf"].append(tf[[k - 1 for k in sel]].astype(np.float16))
        for k in sel:
            bp = B[6 + k]
            lab.append([k, *B[7 + k]["disp"], *B[7 + k]["ctr"], *(bp["ctr"] if bp["ok"] else [np.nan, np.nan]), float(bp["ok"]), j])   # 0 걸음 · 1–2 속도 · 3–4 위치 · 5–6 직전 위치 · 7 직전 ok · 8 클립
        if (j + 1) % 200 == 0:
            print(f"  {split} {j + 1}/{len(meta)}", flush=True)
    return {k: {t: np.concatenate(v[t]) for t in v} for k, v in out.items()}, np.array(lab, np.float64)


class AttnProbe(nn.Module):
    def __init__(s, d=1280, heads=16, outs=(2, 2)):
        super().__init__()
        s.ln = nn.LayerNorm(d); s.pos = nn.Parameter(torch.randn(256, d) * 0.02); s.q = nn.Parameter(torch.randn(len(outs), d) * 0.02)
        s.attn = nn.MultiheadAttention(d, heads, batch_first=True); s.ln2 = nn.LayerNorm(d); s.heads = nn.ModuleList([nn.Linear(d, o) for o in outs])

    def forward(s, x):
        x = s.ln(x) + s.pos; o, _ = s.attn(s.q.expand(len(x), -1, -1), x, x, need_weights=False); o = s.ln2(o)
        return [h(o[:, i]) for i, h in enumerate(s.heads)]


def train_probe(X, L):
    yv, yp = L[:, 1:3], L[:, 3:5]; st = (yv.mean(0), yv.std(0) + 1e-6, yp.mean(0), yp.std(0) + 1e-6)
    Yv = torch.from_numpy(((yv - st[0]) / st[1]).astype(np.float32)); Yp = torch.from_numpy(((yp - st[2]) / st[3]).astype(np.float32))
    m = AttnProbe(); opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=a.epochs * math.ceil(len(X) / 128), pct_start=0.05)
    g = torch.Generator().manual_seed(0)
    for ep in range(a.epochs):
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]; ov, op = m(torch.from_numpy(X[b.numpy()].astype(np.float32)))
            loss = F.mse_loss(ov, Yv[b]) + F.mse_loss(op, Yp[b]); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    m.eval()

    def read(Xq):
        V, P = [], []
        with torch.no_grad():
            for i in range(0, len(Xq), 128):
                o = m(torch.from_numpy(Xq[i:i + 128].astype(np.float32))); V.append(o[0].numpy()); P.append(o[1].numpy())
        return np.concatenate(V) * st[1] + st[0], np.concatenate(P) * st[3] + st[2]
    return read


def gains(V, P, L, rollout):
    """걸음별 위치 증분 이득 · 속도 이득. 굴림: Δ읽은 = 읽은_k − 읽은_(k−1) (k = 1 은 블록 7 실제 위치); teacher forcing: Δ읽은 = 읽은_k − 실제_(k−1)."""
    res = {}; clip = L[:, 8]; step = L[:, 0]
    prev_read = np.full((len(L), 2), np.nan)
    for i in range(len(L)):
        k = int(step[i])
        if k == 1:
            prev_read[i] = L[i, 5:7]                                  # 블록 7 실제 위치
        elif rollout:
            j = np.nonzero((clip == clip[i]) & (step == k - 1))[0]
            prev_read[i] = P[j[0]] if len(j) else np.nan              # 직전 걸음 물체가 안 보였으면 표본 제외
        else:
            prev_read[i] = L[i, 5:7] if L[i, 7] > 0 else np.nan
    for k in range(1, 9):
        s = step == k; vt = L[s, 1:3]; vr = V[s]; mv = np.linalg.norm(vt, axis=1) >= 2
        dt = L[s, 3:5] - L[s, 5:7]; dr = P[s] - prev_read[s]; ok = (L[s, 7] > 0) & np.isfinite(dr).all(1) & np.isfinite(dt).all(1)
        ok &= np.linalg.norm(np.nan_to_num(dt), axis=1) >= 4
        res[str(k)] = {"position_increment_gain": round(float((dr[ok] * dt[ok]).sum() / (dt[ok] ** 2).sum()), 3) if ok.sum() else None, "n_p": int(ok.sum()),
                       "velocity_gain": round(float((vr[mv] * vt[mv]).sum() / (vt[mv] ** 2).sum()), 3) if mv.sum() else None, "n_v": int(mv.sum())}
    g = [res[str(k)]["position_increment_gain"] for k in (1, 2, 3)]
    if all(x is not None and x > 0 for x in g):
        res["p12"], res["p23"] = round(g[1] / g[0], 3), round(g[2] / g[1], 3)
    return res


Otr, Ltr = collect("train", a.n_train); Ova, Lva = collect("val", 350)
print("표본 train", len(Ltr), "val", len(Lva), flush=True)
R = {"n_train_samples": int(len(Ltr)), "n_val_samples": int(len(Lva)), "models": MODELS}
for name in MODELS:
    rd = train_probe(Otr[name]["ro"], Ltr)
    R[name] = {"rollout": gains(*rd(Ova[name]["ro"]), Lva, True), "teacher_forcing": gains(*rd(Ova[name]["tf"]), Lva, False)}
    for t in ("rollout", "teacher_forcing"):
        r = R[name][t]
        print(f"{name:9s} {t:15s} Δpos " + " ".join(f"{r[str(k)]['position_increment_gain']}" for k in range(1, 9)) + " | v " + " ".join(f"{r[str(k)]['velocity_gain']}" for k in range(1, 9)) +
              (f" | p12 {r.get('p12')} p23 {r.get('p23')}" if t == "rollout" else ""), flush=True)
json.dump(R, open(f"{REPO}/z_analysis/results/rollout_decay_models.json", "w"), indent=1)
print("saved")
