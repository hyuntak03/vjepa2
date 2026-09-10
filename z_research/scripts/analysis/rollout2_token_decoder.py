#!/usr/bin/env python3
"""RollOut_v2 — 토큰 단위 위치 decoder (최소형). h 에서 배워 p 에 적용한다.

    s_ij = w · token_ij + b        토큰마다 선형 점수 하나 (w 는 전 토큰 공유, 1,281 파라미터)
    a    = softmax(s) over 16x16
    (x̂, ŷ) = Σ a_ij (grid_x_j, grid_y_i)      격자 기대값. 전역 벡터를 쓸 길이 없다
loss = MSE(x̂, ŷ; 정답)  뿐. 보조 loss·confidence 없음.
학습: h 의 16 튜블릿 전부, 7 시나리오, pos+imp (h 는 렌더된 프레임을 봤으니 위치는 사실), 클립 30%, holdout 레벨 제외, 화면 밖 튜블릿 제외.
평가: h 미래(self) · h 문맥(시각 이동) · p(이식) · 안 본 레벨 · ledge/wall 두 미래 거리.
진단(사후): softmax peak 가 물체 반경(≈16 px) 안에 있는 비율.

  python z_research/scripts/analysis/rollout2_token_decoder.py --device cuda:0
"""
from __future__ import annotations
import argparse, csv, json, re, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
CACHE = Path("/local_datasets/world/world_analysis/cache/rollout_v2_vith")
INDEX = ROOT / "data_csv/rollout_v2/index_probe.csv"
OUT = ROOT / "z_research/RollOutV2/exp_results/token_decoder.json"
G, S, D, T8 = 16, 256, 1280, 8
SCEN = ["flat_v", "flat_a", "ramp_a", "arc", "fall", "ledge", "wall"]
MOVING = {"flat_v": "x", "flat_a": "x", "ramp_a": "x", "arc": "xy", "fall": "y", "ledge": "xy", "wall": "x"}
FPS, STRIDE = 16.0, 3


def arr(s):
    return np.array([float(v) for v in re.split(r"[;,\s|]+", str(s).strip().strip("[]")) if v], float)


class TokenDecoder(nn.Module):
    def __init__(self, d=D):
        super().__init__()
        self.lin = nn.Linear(d, 1)
        g = (torch.arange(G, dtype=torch.float32) + 0.5) / G * 2 - 1
        self.register_buffer("gx", g.repeat(G)); self.register_buffer("gy", g.repeat_interleave(G))   # 토큰 index = i*16 + j (i=행=y, j=열=x)

    def forward(self, tok):                           # (B, 256, D)
        a = self.lin(tok).squeeze(-1).softmax(-1)
        return torch.stack([(a * self.gx).sum(-1), (a * self.gy).sum(-1)], -1), a


def slope(x):
    t = np.arange(T8, dtype=np.float64); tc = t - t.mean(); return (x * tc).sum(1) / (tc ** 2).sum()


def quad(x):
    t = np.arange(T8, dtype=np.float64); A = np.vstack([np.ones_like(t), t, t * t]).T
    return np.linalg.lstsq(A, x.T, rcond=None)[0][2]


def r2_within(x, y):
    yc = y - y.mean(1, keepdims=True); xc = x - x.mean(1, keepdims=True); d = (yc ** 2).sum()
    return float(1 - ((yc - xc) ** 2).sum() / d) if d > 0 else None


def beta_r2(b, bl):
    if len(np.unique(np.round(bl, 6))) < 3:
        return None, None
    A = np.vstack([bl, np.ones_like(bl)]).T; beta, c0 = np.linalg.lstsq(A, b, rcond=None)[0]
    return float(beta), float(1 - ((b - A @ [beta, c0]) ** 2).sum() / ((b - b.mean()) ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=0.3); ap.add_argument("--bs", type=int, default=256); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("-o", "--out", type=Path, default=OUT)
    a = ap.parse_args(); torch.manual_seed(a.seed); dev = torch.device(a.device)
    idx = list(csv.DictReader(INDEX.open())); vids = [r["video_id"] for r in idx]
    meta = json.loads((CACHE / "meta.json").read_text()); row_of = {v: i for i, v in enumerate(meta["video_ids"])}
    rows = np.array([row_of[v] for v in vids]); res = float(idx[0]["resolution"]) / 2
    px = np.stack([arr(r["px_x_by_sample"]) for r in idx]); py = np.stack([arr(r["px_y_by_sample"]) for r in idx])
    L = np.stack([px.reshape(-1, 16, 2).mean(2) / res - 1, py.reshape(-1, 16, 2).mean(2) / res - 1], -1).astype(np.float32)   # (n, 16, 2)
    inf = (np.stack([arr(r["in_frame_by_sample"]) for r in idx]) > 0).reshape(-1, 16, 2).all(2) & (np.abs(L) <= 1).all(2)
    sc = np.array([r["scenario"] for r in idx]); plaus = np.array([int(r["plausible"]) for r in idx])
    prim = np.array([float(r["primary"]) for r in idx]); sec = np.array([float(r["secondary"]) for r in idx])
    hold = np.array([r["holdout"] in ("1", "True", "true") for r in idx]); blk = np.array([r["block_id"] for r in idx])
    zcm = np.stack([arr(r["z_cm_by_sample"]) for r in idx]); hw = float(idx[0]["frame_half_width_cm"])
    ay = np.polyfit(zcm.ravel(), py.ravel(), 1)[0]; cm_per = {"x": hw, "y": res / abs(ay)}; dt = STRIDE * 2 / FPS
    obj_r = float(idx[0]["obj_apparent_px"]) / 2 / res      # 물체 반경 (정규화 좌표)

    rng = np.random.RandomState(a.seed); tr, va = [], []
    cell = np.array([f"{s}|{p}|{q}|{pl}" for s, p, q, pl in zip(sc, prim, sec, plaus)])
    for c in np.unique(cell):
        i = np.where((cell == c) & ~hold)[0]; rng.shuffle(i); k = int(round(a.train_frac * len(i))); tr += list(i[:k]); va += list(i[k:])
    tr, va, ho = np.array(sorted(tr)), np.array(sorted(va)), np.where(hold)[0]
    print(f"split: train {len(tr)} clip → h 튜블릿 {int(inf[tr].sum())} (화면 안), val {len(va)}, holdout-level {len(ho)}")

    H = np.load(CACHE / "target.npy", mmap_mode="r"); P = np.load(CACHE / "predictor.npy", mmap_mode="r")
    t0 = time.time(); order = np.argsort(rows[tr]); Xtr = np.empty((len(tr), 16, S, D), np.float16)
    for k in range(0, len(tr), 32):
        sel = tr[order[k:k + 32]]; Xtr[order[k:k + 32]] = np.asarray(H[rows[sel]]).reshape(-1, 16, S, D)
    print(f"  h train 로딩 {time.time()-t0:.0f}s  {Xtr.nbytes/2**30:.1f} GB")
    keep = np.argwhere(inf[tr]); Ytr = L[tr]
    model = TokenDecoder().to(dev); opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    for ep in range(a.epochs):
        perm = np.random.permutation(len(keep)); tot = n = 0
        for b in range(0, len(perm) - a.bs + 1, a.bs):
            kk = keep[perm[b:b + a.bs]]
            x = torch.from_numpy(Xtr[kk[:, 0], kk[:, 1]]).to(dev, torch.float32); y = torch.from_numpy(Ytr[kk[:, 0], kk[:, 1]]).to(dev)
            pred, _ = model(x); loss = Fn.mse_loss(pred, y); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(kk); n += len(kk)
        print(f"    epoch {ep+1}  train RMSE {np.sqrt(tot/n)*res:.1f} px")
    del Xtr; model.eval()

    @torch.no_grad()
    def decode(src, ids):
        out = np.empty((len(ids), T8, 2), np.float32); att = np.empty((len(ids), T8, S), np.float16)
        order = np.argsort(rows[ids]); ids_s = ids[order]
        for k in range(0, len(ids), 16):
            sel = ids_s[k:k + 16]
            if src == "p":
                x = np.asarray(P[rows[sel]]).reshape(-1, T8, S, D)
            else:
                full = np.asarray(H[rows[sel]]).reshape(-1, 16, S, D); x = full[:, 8:] if src == "h" else full[:, :8]
            pr, at = model(torch.from_numpy(x).to(dev, torch.float32).reshape(-1, S, D))
            out[order[k:k + 16]] = pr.reshape(-1, T8, 2).cpu().numpy(); att[order[k:k + 16]] = at.reshape(-1, T8, S).half().cpu().numpy()
        return out, att

    gx = model.gx.cpu().numpy(); gy = model.gy.cpu().numpy()

    def metrics(pred, att, ids, lab, msk):
        rec = {"n": int(len(ids))}
        for ci, c in enumerate("xy"):
            x, y = pred[:, :, ci], lab[ids][:, :, ci]; b, bl = slope(x), slope(y); beta, r2v = beta_r2(b, bl)
            rec[c] = dict(r2_within=r2_within(x, y), mae_px=float(np.abs(x - y).mean() * res), offset_px=float((x - y).mean() * res), beta=beta, r2_speed=r2v,
                          vel_mae_cm_s=float(np.abs(b - bl).mean() * cm_per[c] / dt), acc_hat=float(quad(x).mean() * 2 * cm_per[c] / dt ** 2), acc_true=float(quad(y).mean() * 2 * cm_per[c] / dt ** 2))
        pk = att.astype(np.float32).argmax(-1); m = msk[ids]
        d = np.hypot(gx[pk] - lab[ids][:, :, 0], gy[pk] - lab[ids][:, :, 1])
        rec["peak_in_object_frac"] = float((d[m] <= obj_r + 1.0 / G).mean()) if m.any() else None      # peak 칸이 물체 반경(+반 칸) 안
        rec["peak_prob_mean"] = float(att.astype(np.float32).max(-1)[m].mean()) if m.any() else None
        return rec

    R = {"meta": dict(decoder="per-token linear score (1281 params) + softmax over 16x16 + grid expectation; MSE only; trained on h, 16 tubelets, 7 scenarios, pos+imp",
                      train_frac=a.train_frac, n_train=int(len(tr)), epochs=a.epochs, lr=a.lr, seed=a.seed, obj_radius_norm=obj_r), "by_scenario": {}, "two_futures": {}}
    Lf, Lc = L[:, 8:], L[:, :8]; Mf, Mc = inf[:, 8:], inf[:, :8]; PRED = {}
    print("\n[시나리오 × 평가셋 × 소스]  h = h 미래(self), h_ctx = h 문맥(시각 이동), p = 이식.  peak = softmax 최대 칸이 물체 안인 비율")
    for scn in SCEN:
        R["by_scenario"][scn] = {}
        for ev, pool in (("val", va), ("holdout", ho)):
            ids = pool[(sc[pool] == scn) & (plaus[pool] == 1)]; R["by_scenario"][scn][ev] = {}
            for src in ("h", "h_ctx", "p"):
                pred, att = decode(src, ids); PRED[(scn, ev, src)] = (ids, pred, att)
                rec = metrics(pred, att, ids, Lc if src == "h_ctx" else Lf, Mc if src == "h_ctx" else Mf); R["by_scenario"][scn][ev][src] = rec
                cols = ["x", "y"] if MOVING[scn] == "xy" else [MOVING[scn]]; f3 = lambda v: " n/a " if v is None else f"{v:.3f}"
                print(f"  {scn:<7} {ev:<7} {src:<5} n={rec['n']:>4} peak={f3(rec['peak_in_object_frac'])} pmax={f3(rec['peak_prob_mean'])}  " + "  ".join(
                    f"[{c}] R²={f3(rec[c]['r2_within'])} MAE={rec[c]['mae_px']:5.1f} off={rec[c]['offset_px']:+6.1f} β={f3(rec[c]['beta'])} vMAE={rec[c]['vel_mae_cm_s']:4.0f} acc={rec[c]['acc_hat']:5.0f}/{rec[c]['acc_true']:5.0f}" for c in cols))
    print("\n[두 미래 — ledge / wall, pos 클립 (val+holdout), 슬롯 0 제외, imp 궤적이 화면 안인 슬롯만]")
    imp_of = {blk[i]: i for i in np.where(plaus == 0)[0]}
    for scn in ("ledge", "wall"):
        ids = np.concatenate([va[(sc[va] == scn) & (plaus[va] == 1)], ho[(sc[ho] == scn) & (plaus[ho] == 1)]]); imp_ids = np.array([imp_of[blk[i]] for i in ids])
        use = Mf[imp_ids].copy(); use[:, 0] = False
        for k in range(len(ids)):
            if not use[k].any():
                use[k, 1] = True
        R["two_futures"][scn] = {}
        for src, eids in (("p", ids), ("h", ids), ("h_imp", imp_ids)):
            pred, att = decode("h" if src == "h_imp" else src, eids); PRED[(scn, "tf", src)] = (eids, pred, att)
            d_pos = np.array([np.hypot(pred[k, :, 0] - Lf[i, :, 0], pred[k, :, 1] - Lf[i, :, 1])[use[k]].mean() for k, i in enumerate(ids)])
            d_imp = np.array([np.hypot(pred[k, :, 0] - Lf[j, :, 0], pred[k, :, 1] - Lf[j, :, 1])[use[k]].mean() for k, j in enumerate(imp_ids)])
            rec = dict(n=int(len(ids)), frac_possible=float((d_pos < d_imp).mean()), dist_px=dict(possible=float(d_pos.mean() * res), impossible=float(d_imp.mean() * res)),
                       x_slope_cm_s=dict(read=float(slope(pred[:, :, 0]).mean() * -cm_per["x"] / dt), possible=float(slope(Lf[ids, :, 0]).mean() * -cm_per["x"] / dt), impossible=float(slope(Lf[imp_ids, :, 0]).mean() * -cm_per["x"] / dt)),
                       y_acc_cm_s2=dict(read=float(quad(pred[:, :, 1]).mean() * 2 * cm_per["y"] / dt ** 2), possible=float(quad(Lf[ids, :, 1]).mean() * 2 * cm_per["y"] / dt ** 2), impossible=float(quad(Lf[imp_ids, :, 1]).mean() * 2 * cm_per["y"] / dt ** 2)),
                       peak_prob_mean=float(att.astype(np.float32).max(-1)[:, 1:].mean()))
            R["two_futures"][scn][src] = rec
            print(f"  {scn:<6} {src:<6} possible 쪽 {100*rec['frac_possible']:5.1f}%  dist {rec['dist_px']['possible']:5.1f} vs {rec['dist_px']['impossible']:5.1f} px | x기울기 {rec['x_slope_cm_s']['read']:4.0f} (pos {rec['x_slope_cm_s']['possible']:.0f} / imp {rec['x_slope_cm_s']['impossible']:.0f}) | y가속 {rec['y_acc_cm_s2']['read']:4.0f} (pos {rec['y_acc_cm_s2']['possible']:.0f} / imp {rec['y_acc_cm_s2']['impossible']:.0f}) | pmax {rec['peak_prob_mean']:.3f}")
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(R, indent=1, ensure_ascii=False)); torch.save(model.state_dict(), a.out.with_suffix(".pt"))
    np.savez_compressed(a.out.with_name("token_decoder_preds.npz"), **{f"{s}|{e}|{r}__ids": v[0] for (s, e, r), v in PRED.items()}, **{f"{s}|{e}|{r}__pred": v[1] for (s, e, r), v in PRED.items()},
                        **{f"{s}|{e}|{r}__att": v[2] for (s, e, r), v in PRED.items() if e in ("tf", "holdout")})
    print(f"\n-> {a.out}  (+ .pt, token_decoder_preds.npz)")


if __name__ == "__main__":
    main()
