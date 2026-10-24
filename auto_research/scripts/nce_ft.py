#!/usr/bin/env python3
"""[T] 가까운 음성 대조 이어 학습 (블록 AR 하나 · 곁가지 없음) — 설계 [L] KST 18:22, 사전 등록 진행판 [T] KST 18:33 (auto_research, 2026-10-22).
⚠️ 학습은 사용자 "올려" 뒤에만. 지금은 --check (옵티마이저 없이 앞방향 · 기울기 한 번) 만 돌린다.

  CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=8 /venv/vjepa/bin/python -W ignore auto_research/scripts/nce_ft.py --name nce_l05 --lam 0.5 --tau <τ>
  … --check                               # 학습 없음: 거리 · 손실 크기 · 기울기만 확인

데이터: 기존 이어 학습 팔과 같은 자연 영상 학습 캐시 (/workspace/auto_research_cache/natural_train, K710 · HT100M · SSv2 각 300 창 × 블록 8).
후보: 같은 창의 블록 8 개 전부를 ±1/255 로 다시 인코딩한 판 (nce_reencode.py, 판 2 개 중 무작위) — 문맥 · L1 정답은 깨끗한 캐시.
  예측 위치 t = 1..5 (블록 t+1 예측): 양성 = h'_{t+1}, 음성 = h'_t (멈춤) · h'_{t−1} (뒤로) · h'_{t+2} (건너뜀) · 같은 창 나머지 4 블록.
거리 d(p, c) = 칸 · 차원 평균 L1 (채점 surprise 와 같은 식). 손실 = L1(p, 깨끗한 h_{t+1}) (TF 전 위치) + λ · CE(−d/τ, 양성).
--check 출력: 후보 종류별 거리 (양성 · 멈춤 · 뒤로 · 건너뜀 · 나머지), 거리 차 크기 (τ 정하기), 손실 크기, 두 손실의 기울기 노름과 코사인."""
import argparse, glob, os, sys, time
import numpy as np, torch, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="nce_check")
ap.add_argument("--lam", type=float, default=0.5)
ap.add_argument("--tau", type=float, default=0.01)
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--draws", type=int, default=2)
ap.add_argument("--cache", default="/workspace/auto_research_cache/natural_train")
ap.add_argument("--reenc", default="/workspace/auto_research_cache/natural_train_reenc")
ap.add_argument("--out", default="/workspace/auto_research_cache/nce_ft")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--check", action="store_true")
ap.add_argument("--check_batches", type=int, default=20)
a = ap.parse_args()
sys.path.insert(0, os.path.dirname(__file__))
_argv = sys.argv; sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT                                                     # noqa: E402
sys.argv = _argv
torch.set_num_threads(8)
dev = torch.device("cuda"); S, D, NB = 256, 1280, 8
TPOS = [1, 2, 3, 4, 5]                                                      # 예측 위치 t → 블록 t+1


def loader():
    srcs = [os.path.basename(f)[:-6] for f in sorted(glob.glob(os.path.join(a.cache, "*_h.npy")))]
    H = [np.load(os.path.join(a.cache, f"{s}_h.npy"), mmap_mode="r") for s in srcs]
    R = [[np.load(os.path.join(a.reenc, f"{s}_hn{d}.npy"), mmap_mode="r") for d in range(a.draws)] for s in srcs]
    return srcs, H, R


def batch(rng, H, R):
    si = rng.integers(len(H), size=a.bs)
    wi = [int(rng.integers(H[s].shape[0])) for s in si]; di = rng.integers(a.draws, size=a.bs)
    x = torch.from_numpy(np.stack([np.asarray(H[s][w], np.float32) for s, w in zip(si, wi)])).to(dev)
    c = torch.from_numpy(np.stack([np.asarray(R[s][d][w], np.float32) for s, w, d in zip(si, wi, di)])).to(dev)
    return x, c


def forward(pred, x, c):
    B = x.size(0)
    idx = torch.arange(NB * S, device=dev).unsqueeze(0).expand(B, -1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        p = pred.forward_seq(x.reshape(B, NB * S, D), idx).float().view(B, NB, S, D)[:, :NB - 1]   # 위치 b → 블록 b+1
    l1 = (p - x[:, 1:]).abs().mean()
    pt = p[:, TPOS]                                                         # (B, 5, S, D)
    d = torch.stack([(pt - c[:, k:k + 1]).abs().mean((2, 3)) for k in range(NB)], -1)   # (B, 5, 8)
    tgt = torch.tensor([t + 1 for t in TPOS], device=dev).expand(B, -1)
    ce = F.cross_entropy((-d / a.tau).reshape(-1, NB), tgt.reshape(-1))
    return p, l1, d, ce, tgt


if __name__ == "__main__":
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    pred = FT.load_pred("natural_ar_vw_state", "latest.pt").to(dev)
    srcs, H, R = loader()
    if a.check:
        pred.train()
        rows = {k: [] for k in ("pos", "stop", "back", "skip", "rest", "spread", "gap_stop", "l1", "ce", "acc", "rank")}
        G = []
        for bi in range(a.check_batches):
            x, c = batch(rng, H, R)
            p, l1, d, ce, tgt = forward(pred, x, c)
            dn = d.detach()
            for j, t in enumerate(TPOS):
                rows["pos"].append(dn[:, j, t + 1]); rows["stop"].append(dn[:, j, t]); rows["back"].append(dn[:, j, t - 1]); rows["skip"].append(dn[:, j, t + 2])
                rest = [k for k in range(NB) if k not in (t - 1, t, t + 1, t + 2)]
                rows["rest"].append(dn[:, j, rest].mean(-1)); rows["gap_stop"].append(dn[:, j, t + 1] - dn[:, j, t])
            rows["spread"].append(dn.std(-1).flatten())
            rows["l1"].append(l1.detach()[None]); rows["ce"].append(ce.detach()[None])
            rows["acc"].append((dn.argmin(-1) == tgt).float().flatten())
            rows["rank"].append((dn < dn.gather(-1, tgt[..., None])).sum(-1).float().flatten())
            if bi < 4:                                                      # 기울기: 두 손실을 따로 (옵티마이저 없음)
                pred.zero_grad(set_to_none=True); l1.backward(retain_graph=True)
                g1 = torch.cat([q.grad.flatten() for q in pred.parameters() if q.grad is not None]).clone()
                pred.zero_grad(set_to_none=True); ce.backward()
                g2 = torch.cat([q.grad.flatten() for q in pred.parameters() if q.grad is not None]).clone()
                G.append((float(g1.norm()), float(g2.norm()), float(F.cosine_similarity(g1, g2, dim=0))))
                pred.zero_grad(set_to_none=True)
        R_ = {k: torch.cat(v).cpu().numpy() for k, v in rows.items()}
        print(f"[check] τ {a.tau} · 배치 {a.check_batches} × {a.bs} 창 × 위치 {len(TPOS)}", flush=True)
        print("  거리 (칸 평균 L1, 다시 인코딩한 후보): " + " · ".join(f"{k} {R_[k].mean():.4f}" for k in ("pos", "stop", "back", "skip", "rest")), flush=True)
        print(f"  양성 − 멈춤: 평균 {R_['gap_stop'].mean():+.4f} · 중앙값 {np.median(R_['gap_stop']):+.4f} · |차| 중앙값 {np.median(np.abs(R_['gap_stop'])):.4f} · 참이 멈춤보다 가까운 몫 {100 * (R_['gap_stop'] < 0).mean():.1f} %", flush=True)
        print(f"  후보 8 개 거리의 표준편차 (예측마다) 중앙값 {np.median(R_['spread']):.4f} · 사분위 {np.percentile(R_['spread'], 25):.4f}–{np.percentile(R_['spread'], 75):.4f}", flush=True)
        print(f"  지금 순위 정확도 {100 * R_['acc'].mean():.1f} % (우연 12.5) · 양성 순위 평균 {R_['rank'].mean():.2f} (0 = 1 등)", flush=True)
        print(f"  손실: L1 {R_['l1'].mean():.4f} · CE {R_['ce'].mean():.4f} (log 8 = {np.log(8):.4f})", flush=True)
        for i, (n1, n2, cs) in enumerate(G):
            print(f"  기울기 배치 {i}: ‖∇L1‖ {n1:.4f} · ‖∇CE‖ {n2:.4f} (λ 0.1 → {0.1 * n2:.4f} · λ 0.5 → {0.5 * n2:.4f}) · 코사인 {cs:+.3f}", flush=True)
        sys.exit(0)
    pred.train()
    opt = torch.optim.AdamW(pred.parameters(), lr=a.lr, weight_decay=0.0)
    t0 = time.time()
    for it in range(1, a.steps + 1):
        x, c = batch(rng, H, R)
        p, l1, d, ce, tgt = forward(pred, x, c)
        loss = l1 + a.lam * ce
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(pred.parameters(), 1.0); opt.step()
        if it % 100 == 0 or it == 1:
            dn = d.detach()
            acc = float((dn.argmin(-1) == tgt).float().mean()); stop = float((dn[:, range(len(TPOS)), [t + 1 for t in TPOS]] < dn[:, range(len(TPOS)), TPOS]).float().mean())
            print(f"[{a.name}] {it} L1 {float(l1):.4f} · CE {float(ce):.4f} · 순위 정확도 {100 * acc:.1f} % · 참 < 멈춤 {100 * stop:.1f} % ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(a.out, exist_ok=True)
    torch.save({"predictor": pred.state_dict(), "args": vars(a), "kind": "nce_ft"}, os.path.join(a.out, f"{a.name}.pt"))
    print("saved", flush=True)
