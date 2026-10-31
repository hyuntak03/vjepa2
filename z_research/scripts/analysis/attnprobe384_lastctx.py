"""층별 attentive probe — teacher forcing 의 마지막 문맥 자리 (블록 7, 입력 = 실제 블록 7) 은닉 (2026-10-27, CPU).
사용자: "(predictor_info_by_layer 그림) 이거 teacher forcing 하는 context 자리에서도 그려 줄래? 마지막 context 자리만 그려 줘도 될 듯".
은닉: 블록 AR natural_ar_vw_state, 캐시된 encoder 블록 토큰 (ar_layers enc) 의 블록 0–7 만 넣는다 (블록 인과라 자리 7 은 블록 ≤ 7 만 본다).
  층 14 개 = 입력 (관측 임베딩 + type[0], 첫 층 입력) · 1–12 층 출력 · 마지막 정규화 — attnprobe384_extract.py 와 같은 정의, 자리 7 의 토큰 (256, 384).
라벨 두 가지: 'own' = 블록 7 (입력 블록 자기 — 굴림 선과 같은 정의) · 'next' = 블록 8 (이 자리가 예측하는 다음 블록). 모양 × 색 · 블록 안 속도 · 위치.
판독기 · 학습: attnprobe384_train.py 와 같다 (d 384, 질의 3, 16 head, AdamW 5e-4 · wd 0.05 · 묶음 128 · 25 epoch · one-cycle, train 클립 10 % 로 epoch 선택, 라벨 표준화). CPU.
  CUDA_VISIBLE_DEVICES=7 MEM_GB=3 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/attnprobe384_lastctx.py --n_train 3000 --gpu
출력: z_analysis/results/attnprobe384/{prefix}_{own,next}_L{layer}.json
  train 8,000 클립 전부 (2026-10-27 재실행): --n_train 8000 --prefix tfctx7_full
"""
import argparse, copy, json, math, os, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ap = argparse.ArgumentParser(); ap.add_argument("--n_train", type=int, default=3000); ap.add_argument("--epochs", type=int, default=25); ap.add_argument("--threads", type=int, default=48); ap.add_argument("--gpu", action="store_true")
ap.add_argument("--prefix", default="tfctx7", help="2026-10-27: 출력 이름 앞머리 (train 8,000 클립 재실행은 tfctx7_full)"); a = ap.parse_args()
torch.set_num_threads(a.threads)
dev = torch.device("cuda" if a.gpu else "cpu")   # 2026-10-27: CPU 가 다른 작업과 겹쳐 느려 GPU 판 추가 (MEM_GB 상한)
if a.gpu:
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEM_GB", 3)) * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory, 0)
REPO = "/workspace/world_model/vjepa2"; sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "auto_research/scripts"))
_argv = list(sys.argv); sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT
sys.argv = _argv
SRC = "/workspace/auto_research_cache/ar_layers"; OUT = f"{REPO}/z_analysis/results/attnprobe384"; S, D, C = 256, 1280, 8
ar = FT.load_pred("natural_ar_vw_state", "latest.pt").float().eval().to(dev); idx8 = torch.arange(C * S, device=dev).unsqueeze(0)
CAP = {}
ar.predictor_blocks[0].register_forward_pre_hook(lambda m, inp: CAP.__setitem__(0, inp[0][0, (C - 1) * S:].detach()))
for i, blk in enumerate(ar.predictor_blocks):
    blk.register_forward_hook(lambda m, inp, out, i=i: CAP.__setitem__(i + 1, (out[0] if isinstance(out, tuple) else out)[0, (C - 1) * S:].detach()))
ar.predictor_norm.register_forward_hook(lambda m, inp, out: CAP.__setitem__(13, out[0, (C - 1) * S:].detach()))


def lab_of(m, b):
    bl = m["blocks"][b]
    return [m["shape"] * 8 + m["color"], *bl["disp"], *bl["ctr"]] if bl["ok"] else None


def extract(split, n):
    X, L = [], {"own": [], "next": []}; keep = {"own": [], "next": []}
    for j, m in enumerate(json.load(open(f"{SRC}/{split}_meta.json"))[:n]):
        lo, ln = lab_of(m, 7), lab_of(m, 8)
        if lo is None and ln is None:
            continue
        path = f"{SRC}/{split}/{m['i']:05d}.npz"; z = np.load(path); h = z["enc"][:C].astype(np.float32); z.close()
        fd = os.open(path, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
        with torch.no_grad():
            ar.forward_seq(torch.from_numpy(h).to(dev).reshape(1, -1, D), idx8)
        X.append(np.stack([CAP[l].float().cpu().numpy().astype(np.float16) for l in range(14)])); k = len(X) - 1
        for nm, lb in (("own", lo), ("next", ln)):
            if lb is not None:
                L[nm].append(lb); keep[nm].append(k)
        if (j + 1) % 500 == 0:
            print(f"  {split} {j + 1}/{n}", flush=True)
    return np.stack(X, 1), {k: np.array(v, np.float64) for k, v in L.items()}, {k: np.array(v) for k, v in keep.items()}


class AttnProbe(nn.Module):
    def __init__(s, d=384, heads=16, outs=(48, 2, 2)):
        super().__init__()
        s.ln = nn.LayerNorm(d); s.pos = nn.Parameter(torch.randn(256, d) * 0.02); s.q = nn.Parameter(torch.randn(len(outs), d) * 0.02)
        s.attn = nn.MultiheadAttention(d, heads, batch_first=True); s.ln2 = nn.LayerNorm(d); s.heads = nn.ModuleList([nn.Linear(d, o) for o in outs])

    def forward(s, x):
        x = s.ln(x) + s.pos; o, _ = s.attn(s.q.expand(len(x), -1, -1), x, x, need_weights=False); o = s.ln2(o)
        return [h(o[:, i]) for i, h in enumerate(s.heads)]


def run(Xtr, Ltr, Xva, Lva):
    st = (Ltr[:, 1:3].mean(0), Ltr[:, 1:3].std(0) + 1e-6, Ltr[:, 3:5].mean(0), Ltr[:, 3:5].std(0) + 1e-6)
    def tens(L):
        return (torch.from_numpy(L[:, 0].astype(np.int64)).to(dev), torch.from_numpy(((L[:, 1:3] - st[0]) / st[1]).astype(np.float32)).to(dev),
                torch.from_numpy(((L[:, 3:5] - st[2]) / st[3]).astype(np.float32)).to(dev))
    rng = np.random.default_rng(0); sel = rng.permutation(len(Xtr)); nh = len(Xtr) // 10; ho, tr = sel[:nh], sel[nh:]
    Yi, Yv, Yp = tens(Ltr)
    m = AttnProbe().to(dev); opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-4, total_steps=a.epochs * math.ceil(len(tr) / 128), pct_start=0.05)

    def evaluate(X, L):
        I, V, P = [], [], []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                o = m(torch.from_numpy(X[i:i + 256].astype(np.float32)).to(dev)); I.append(o[0].argmax(1).cpu().numpy()); V.append(o[1].cpu().numpy()); P.append(o[2].cpu().numpy())
        I, V, P = np.concatenate(I), np.concatenate(V) * st[1] + st[0], np.concatenate(P) * st[3] + st[2]
        r2 = lambda A, B: float(1 - ((A - B) ** 2).sum() / ((B - B.mean(0)) ** 2).sum())
        return {"shape_x_color": round(100 * float((I == L[:, 0]).mean()), 1), "velocity_r2": round(r2(V, L[:, 1:3]), 3), "position_r2": round(r2(P, L[:, 3:5]), 3), "n": int(len(L))}
    best, best_state = -1e9, None
    for ep in range(a.epochs):
        m.train(); perm = tr[np.random.default_rng(ep).permutation(len(tr))]
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]; oi, ov, op = m(torch.from_numpy(Xtr[b].astype(np.float32)).to(dev)); bt = torch.from_numpy(b).to(dev)
            loss = F.cross_entropy(oi, Yi[bt]) + F.mse_loss(ov, Yv[bt]) + F.mse_loss(op, Yp[bt]); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        m.eval(); e = evaluate(Xtr[ho], Ltr[ho]); sc = (e["shape_x_color"] / 100 + max(e["velocity_r2"], -1) + max(e["position_r2"], -1)) / 3
        if sc > best:
            best, best_state, best_ep = sc, copy.deepcopy(m.state_dict()), ep
    m.load_state_dict(best_state)
    return dict(evaluate(Xva, Lva), best_epoch=best_ep)


Xtr, Ltr, Ktr = extract("train", a.n_train); Xva, Lva, Kva = extract("val", 350)
print("표본 train", {k: len(v) for k, v in Ltr.items()}, "val", {k: len(v) for k, v in Lva.items()}, flush=True)
for lab in ("own", "next"):
    for l in range(14):
        r = run(Xtr[l][Ktr[lab]], Ltr[lab], Xva[l][Kva[lab]], Lva[lab])
        json.dump({"layer": l, "label_block": 7 if lab == "own" else 8, "val_all": r}, open(f"{OUT}/{a.prefix}_{lab}_L{l}.json", "w"), indent=1)
        print(lab, l, r, flush=True)
print("saved")
