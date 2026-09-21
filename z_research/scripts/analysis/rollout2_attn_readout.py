#!/usr/bin/env python3
"""RollOut_v2 — 쿼리 하나짜리 attentive pooling 위치 readout. 학습은 rollout_v2_training 의 p 만, v2 는 test 만.

자: 슬롯의 256 토큰 (256, D) → 점수 s_i = (tok_i · q)/√D → softmax → pooled = Σ a_i tok_i (1, D) → Linear(D, 2) = (x, y).
파라미터 q (D) + 선형 (2D + 2) ≈ 3.8k. 풀링 readout (`rollout2_fit_readout.py`) 과 같은 라벨·같은 슬롯·같은 test 표.
저장: exp_results/<train>/attentive_pooling/p/{attn.pt, fit.json, test.json, preds.npz}

  python z_research/scripts/analysis/rollout2_attn_readout.py [--epochs 30] [--lr 1e-3]
  python z_research/scripts/analysis/rollout2_attn_readout.py --test-only      # 저장된 자로 test 만 (시나리오 추가 후)
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rollout2_test_readout as rt                         # noqa: E402

TR_CACHE = rt.TR_CACHE; TR_INDEX = rt.TR_INDEX                     # 학습셋 버전은 ROLLOUT2_TRAIN (rollout2_test_readout.py)
OUT = rt.RES_ROOT / "attentive_pooling/p"
S, T, D, RES = rt.S, rt.T, rt.D, rt.RES


class AttnReadout(nn.Module):
    def __init__(self, d=D):
        super().__init__(); self.q = nn.Parameter(torch.randn(d) / d ** 0.5); self.out = nn.Linear(d, 2)

    def forward(self, tok):                                   # (B, 256, D) → (B, 2), (B, 256)
        a = torch.softmax(tok @ self.q / D ** 0.5, -1); return self.out((a.unsqueeze(-1) * tok).sum(1)), a


def labels(index):
    """라벨 (n, 8, 2) 와 마스크 (n, 8). 마스크 = visible_by_sample (없으면 in_frame_by_sample): 화면 안이고 사물에 안 가려진 튜블릿.
    학습셋 v5 부터 화면 밖·가려진 샘플이 있어 **학습 손실도 test 지표도 이 마스크로 거른다**."""
    idx = list(csv.DictReader(index.open())); n = len(idx)
    px = np.stack([rt.arr(r["px_x_by_sample"]) for r in idx]); py = np.stack([rt.arr(r["px_y_by_sample"]) for r in idx])
    L = np.stack([px.reshape(n, 16, 2).mean(2) / RES - 1, py.reshape(n, 16, 2).mean(2) / RES - 1], -1)[:, T:]
    key = "visible_by_sample" if "visible_by_sample" in idx[0] else "in_frame_by_sample"
    inf = (np.stack([rt.arr(r[key]) for r in idx]) > 0).reshape(n, 16, 2).all(2)[:, T:]
    return idx, L.astype(np.float32), inf


class Loader:
    """학습 텐서 (n, 2048, 1280) fp16 = 42 GB. GPU 가 2 장 이상 보이면 (CUDA_VISIBLE_DEVICES) 장 수만큼 쪼개 GPU 에 올리고
    배치는 GPU→GPU 복사로 모은다 (epoch 당 1~2 s). 1 장이면 CPU pinned 에 두고 배치만 올린다 (epoch 당 20 s)."""
    def __init__(self, X, dev, bs):
        self.dev = torch.device(dev); self.bs = bs; n = len(X); nd = torch.cuda.device_count()
        if nd >= 2:
            self.mode = "gpu"; self.bounds = np.linspace(0, n, nd + 1).astype(int); self.shards = []
            for i in range(nd):
                self.shards.append(torch.from_numpy(np.ascontiguousarray(X[self.bounds[i]:self.bounds[i + 1]])).to(f"cuda:{i}"))
            print(f"    Loader: {n} clip 을 GPU {nd} 장에 샤딩 ({self.shards[0].numel()*2/2**30:.1f} GB/장)", flush=True)
        else:
            self.mode = "cpu"; self.X = torch.from_numpy(np.ascontiguousarray(X)).pin_memory()
            self.buf = torch.empty((bs,) + tuple(self.X.shape[1:]), dtype=self.X.dtype).pin_memory()

    def batch(self, b):
        if self.mode == "cpu":
            n = len(b); torch.index_select(self.X, 0, b, out=self.buf[:n]); return self.buf[:n].to(self.dev, non_blocking=True)
        b = np.asarray(b); out = torch.empty((len(b),) + tuple(self.shards[0].shape[1:]), dtype=self.shards[0].dtype, device=self.dev)
        for i, sh in enumerate(self.shards):
            sel = np.where((b >= self.bounds[i]) & (b < self.bounds[i + 1]))[0]
            if len(sel):
                out[torch.as_tensor(sel, device=self.dev)] = sh[torch.as_tensor(b[sel] - self.bounds[i], device=sh.device)].to(self.dev, non_blocking=True)
        return out

    def __len__(self):
        return int(self.bounds[-1]) if self.mode == "gpu" else len(self.X)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--holdout", type=float, default=0.0, help="학습셋 block 의 이 비율을 held-out 으로 뺀다 (scenario 층화, seed 0)")
    ap.add_argument("--test-only", action="store_true", help="저장된 attn.pt 를 불러 v2 test 만 한다 (자 재학습 없음, fit.json 유지). 2026-09-19 감속 시나리오 추가용"); a = ap.parse_args()
    global OUT
    if a.holdout > 0:
        OUT = rt.RES_ROOT / f"attentive_pooling/p_holdout{int(a.holdout*100)}"
    dev = "cuda"; torch.manual_seed(0)
    if a.test_only:
        model = AttnReadout().to(dev); model.load_state_dict(torch.load(OUT / "attn.pt", map_location=dev)); model.eval()
        print(f"test-only: {OUT / 'attn.pt'} 를 불러 v2 test 만 한다"); return _test_v2(model, dev)
    idx, L, inf = labels(TR_INDEX); n = len(idx); print(f"학습셋 {n} clip, visible 미래 튜블릿 {inf.sum()} / {inf.size}")
    tr_rows = np.arange(n); ho_rows = np.array([], int)
    if a.holdout > 0:
        rng = np.random.RandomState(0); blk = np.array([r["block_id"] for r in idx]); sc = np.array([r["scenario"] for r in idx]); ho_blk = set()
        for s_ in np.unique(sc):
            b = np.unique(blk[sc == s_]); ho_blk |= set(rng.choice(b, int(round(len(b) * a.holdout)), replace=False))
        ho_rows = np.array([i for i in range(n) if blk[i] in ho_blk]); tr_rows = np.array([i for i in range(n) if blk[i] not in ho_blk])
        print(f"held-out {a.holdout:.0%}: 학습 {len(tr_rows)} clip / held-out {len(ho_rows)} clip (block 단위, scenario 층화)")
    if json.loads((TR_CACHE / "meta.json").read_text())["video_ids"] != [r["video_id"] for r in idx]:
        sys.exit("학습셋 캐시 video_ids 불일치")
    ld = Loader(np.load(TR_CACHE / "predictor.npy", mmap_mode="r"), dev, a.bs)
    Y = torch.from_numpy(L).to(dev); M = torch.from_numpy(inf).to(dev)
    model = AttnReadout().to(dev); opt = torch.optim.Adam(model.parameters(), lr=a.lr); t0 = time.time()
    for ep in range(a.epochs):
        perm = torch.as_tensor(tr_rows)[torch.randperm(len(tr_rows))]; tot = 0.0; cnt = 0
        for k in range(0, len(tr_rows), a.bs):
            b = perm[k:k + a.bs]; tok = ld.batch(b).float().reshape(-1, S, D); bd = b.to(dev); y = Y[bd].reshape(-1, 2); m = M[bd].reshape(-1)
            pred, _ = model(tok); loss = (((pred - y) ** 2).mean(1) * m).sum() / m.sum().clamp(min=1); opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * m.sum().item(); cnt += m.sum().item()
        if ep % 5 == 0 or ep == a.epochs - 1:
            print(f"  epoch {ep:3d}  train MSE {tot/cnt:.5f}  ({time.time()-t0:.0f}s)", flush=True)
    model.eval(); OUT.mkdir(parents=True, exist_ok=True); torch.save(model.state_dict(), OUT / "attn.pt")
    with torch.no_grad():
        pr = torch.cat([model(ld.batch(torch.arange(k, min(k + a.bs, n))).float().reshape(-1, S, D))[0] for k in range(0, n, a.bs)]).reshape(n, T, 2).cpu().numpy()
    mm = inf.ravel()
    if len(ho_rows):                                        # held-out: 시나리오별 L2 (칸) 와 MAE
        hm = inf[ho_rows]; hp = pr[ho_rows]; hl = L[ho_rows]; hs = np.array([idx[i]["scenario"] for i in ho_rows]); ho = {}
        for s_ in np.unique(hs):
            k = hs == s_; m = hm[k]; e = np.linalg.norm(hp[k] - hl[k], axis=-1)[m] * RES
            ho[s_] = {"n": int(k.sum()), "l2_cells": float(e.mean() / 18.0), "mae_px_x": float(np.abs(hp[k] - hl[k])[:, :, 0][m].mean() * RES), "mae_px_y": float(np.abs(hp[k] - hl[k])[:, :, 1][m].mean() * RES)}
        e_all = np.linalg.norm(hp - hl, axis=-1)[hm] * RES; e_tr = np.linalg.norm(pr[tr_rows] - L[tr_rows], axis=-1)[inf[tr_rows]] * RES
        ho["_all"] = {"n": int(len(ho_rows)), "l2_cells": float(e_all.mean() / 18.0), "l2_cells_train": float(e_tr.mean() / 18.0)}
        print(f"held-out L2 {ho['_all']['l2_cells']:.2f} 칸 (학습 {ho['_all']['l2_cells_train']:.2f} 칸)"); [print(f"   {s_:<12} n={v['n']:4d}  L2 {v['l2_cells']:.2f} 칸  MAE {v['mae_px_x']:.1f}/{v['mae_px_y']:.1f} px") for s_, v in ho.items() if s_ != "_all"]
        (OUT / "holdout.json").write_text(json.dumps(ho, indent=1))
    fitrep = {"n_clip": int(len(tr_rows)), "holdout_clip": int(len(ho_rows)), "rows_visible": int(inf.sum()), "params": sum(p.numel() for p in model.parameters()), "epochs": a.epochs, "lr": a.lr,
              "r2_x_fit": rt.r2(pr[:, :, 0].ravel()[mm], L[:, :, 0].ravel()[mm]), "r2_y_fit": rt.r2(pr[:, :, 1].ravel()[mm], L[:, :, 1].ravel()[mm]),
              "mae_px_x": float(np.abs(pr - L)[:, :, 0].ravel()[mm].mean() * RES), "mae_px_y": float(np.abs(pr - L)[:, :, 1].ravel()[mm].mean() * RES)}
    print(f"학습셋 fit (visible 만): R² x {fitrep['r2_x_fit']:.3f} y {fitrep['r2_y_fit']:.3f}  MAE {fitrep['mae_px_x']:.1f} / {fitrep['mae_px_y']:.1f} px  (params {fitrep['params']})")
    (OUT / "fit.json").write_text(json.dumps(fitrep, indent=1)); del ld

    _test_v2(model, dev)


def _test_v2(model, dev):
    """저장/학습된 자 model 로 v2 전체를 test 한다 (preds.npz, test.json)."""
    # ── v2 test ──
    idx, L, inf = labels(rt.INDEX); n = len(idx)
    if json.loads((rt.CACHE / "meta.json").read_text())["video_ids"] != [r["video_id"] for r in idx]:
        sys.exit("v2 캐시 video_ids 불일치")
    Pv = np.load(rt.CACHE / "predictor.npy", mmap_mode="r"); pred = np.empty((n, T, 2), np.float32); att = np.empty((n, T, S), np.float16)
    with torch.no_grad():
        for k in range(0, n, 64):
            tok = torch.from_numpy(np.asarray(Pv[k:k + 64])).to(dev).float().reshape(-1, S, D); p, w = model(tok)
            pred[k:k + 64] = p.reshape(-1, T, 2).cpu().numpy(); att[k:k + 64] = w.reshape(-1, T, S).half().cpu().numpy()
    scen = np.array([r["scenario"] for r in idx]); plaus = np.array([int(r["plausible"]) for r in idx])
    np.savez(OUT / "preds.npz", video_id=np.array([r["video_id"] for r in idx]), pred=pred, truth=L, in_frame=inf, scenario=scen, plausible=plaus, attn=att)
    rep = {"per_scenario": {}}
    print(f"\n{'scenario':<8} {'n':>4}  {'R²x':>6} {'R²y':>6}  {'MAEx':>6} {'MAEy':>6}   {'βx':>5} {'βy':>5}   across-clip corr (slot mean) x / y")
    f = lambda v: "   -  " if v is None else f"{v:6.2f}"
    for s in rt.SCEN:
        k = (scen == s) & (plaus == 1); m = inf[k]; p = pred[k]; y = L[k]; mv = rt.MOVING[s]
        cx = np.mean([np.corrcoef(p[:, t, 0], y[:, t, 0])[0, 1] for t in range(T)]) if y[:, :, 0].std(0).mean() > 0.01 else None
        cy = np.mean([np.corrcoef(p[:, t, 1], y[:, t, 1])[0, 1] for t in range(T)]) if y[:, :, 1].std(0).mean() > 0.01 else None
        rec = {"n": int(k.sum()), "r2_x": rt.r2(p[:, :, 0][m], y[:, :, 0][m]) if "x" in mv else None, "r2_y": rt.r2(p[:, :, 1][m], y[:, :, 1][m]) if "y" in mv else None,
               "mae_px_x": float(np.abs(p - y)[:, :, 0][m].mean() * RES), "mae_px_y": float(np.abs(p - y)[:, :, 1][m].mean() * RES),
               "beta_x": rt.beta(rt.slope(p[:, :, 0]), rt.slope(y[:, :, 0])) if "x" in mv else None, "beta_y": rt.beta(rt.slope(p[:, :, 1]), rt.slope(y[:, :, 1])) if "y" in mv else None,
               "corr_x": None if cx is None else float(cx), "corr_y": None if cy is None else float(cy)}
        rep["per_scenario"][s] = rec
        print(f"{s:<8} {rec['n']:>4}  {f(rec['r2_x'])} {f(rec['r2_y'])}  {rec['mae_px_x']:6.1f} {rec['mae_px_y']:6.1f}   {f(rec['beta_x'])[1:]} {f(rec['beta_y'])[1:]}   {f(rec['corr_x'])} / {f(rec['corr_y'])}")
    (OUT / "test.json").write_text(json.dumps(rep, indent=1)); print(f"→ {OUT}")


if __name__ == "__main__":
    main()
