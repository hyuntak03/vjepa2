#!/usr/bin/env python3
"""RollOut_v2 — **과거 프레임만 본 문맥 encoder (z, 16 frames)** 위에 작은 자를 달아 **미래 8 슬롯의 (x, y)** 를 회귀한다 (2026-10-01, 사용자 제안).
질문: "떨어질 것" 이 encoder 표현에 (선형으로) 들어 있는가 — 들어 있는데 predictor 가 못 쓰는가.

입력 (셋을 같은 자 · 같은 라벨로):
  z : 캐시 rollout_v2_z16_vith/ctx_masked.npy  (n, 8 문맥 튜블릿 × 256, 1280)  = 문맥 16 frames 만 본 encoder, LN     ← 주
  p : 캐시 rollout_v2_vith/predictor.npy        (n, 8 미래 튜블릿 × 256, 1280)  = predictor 의 미래                      ← 대조 (predictor 가 하는 것)
  h : 캐시 rollout_v2_vith/target.npy 의 뒤 절반 (n, 8 미래 튜블릿 × 256, 1280) = 미래를 실제로 본 encoder               ← 상한
자 (둘 다 작다):
  A  (사용자 안) 튜블릿마다 N 축 attentive pooling (학습 쿼리 1 개 공유) → (8, D) → Linear(D, 2) 공유 → 튜블릿 t 의 벡터가 **미래 슬롯 t** 를 낸다.  파라미터 1280 + 2562 = 3,842
  B  미래 슬롯마다 쿼리 1 개 (8 개) 가 T×N 전체 토큰을 attend → (8, D) → Linear(D, 2).                                   파라미터 8·1280 + 2562 = 12,802
라벨: 미래 슬롯 (튜블릿 2 샘플 평균) 픽셀 /144 − 1, 마스크 visible_by_sample.  split: index 의 `holdout` (primary 레벨 2 · 6 번째 = 안 본 속도) 로 test, 나머지 train. 가능 (plausible=1) 클립만.
기준선 (라벨만으로): CV = 문맥 마지막 슬롯 위치 + (슬롯 7 − 슬롯 6) × (k+1) / COPY = 마지막 슬롯 고정 / TEMPLATE_GLOBAL = train 전체의 슬롯별 평균 / TEMPLATE_SCEN = 시나리오별 슬롯별 평균 (시나리오 라벨을 아는 oracle).
⚠️ 설계 한계 (실행 전에 확인): v2 ledge 의 낙하 시작은 **모든 클립에서 샘플 22 (미래 슬롯 3)** 로 고정이다 (속도가 달라도 선반 끝 도착 시각이 같게 만들어져 있다).
   그래서 "떨어진다" 자체는 **시나리오별 템플릿** 으로 맞출 수 있고, 이 실험이 재는 것은 "문맥 encoder 에서 **어느 시나리오 (선반 끝이 앞에 있는가)** 와 현재 위치 · 속도를 읽어 내는가" 다.
   낙하 **시각** 을 읽는지는 이 데이터로 못 잰다 (시각이 변하는 데이터가 필요). 결과는 TEMPLATE_SCEN 과 나란히 읽을 것.
  python rollout2_future_from_context.py --rep z|p|h [--head A|B|both] [--epochs 60] --out <dir>     (CPU 가능, 메모리 ≈ 25 GB/rep)
"""
import argparse, csv, json, os, re, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2"); CACHE = Path("/local_datasets/world/world_analysis/cache")
INDEX = ROOT / "data_csv/rollout_v2/index_probe.csv"; S, T, D, RES = 256, 8, 1280, 144.0
REP = {"z": (CACHE / "rollout_v2_z16_vith/ctx_masked.npy", slice(0, T * S)), "p": (CACHE / "rollout_v2_vith/predictor.npy", slice(0, T * S)), "h": (CACHE / "rollout_v2_vith/target.npy", slice(T * S, 2 * T * S))}
arr = lambda s: np.array([float(v) for v in re.split(r"[;,\s|]+", str(s).strip()) if v], np.float32)


class HeadA(nn.Module):
    def __init__(s): super().__init__(); s.q = nn.Parameter(torch.randn(D) / D ** 0.5); s.out = nn.Linear(D, 2)
    def forward(s, tok):                                        # (B, 8·256, D) → (B, 8, 2)
        x = tok.view(tok.size(0), T, S, D); a = torch.softmax((x @ s.q) / D ** 0.5, -1); return s.out((a.unsqueeze(-1) * x).sum(2)), a


class HeadB(nn.Module):
    def __init__(s): super().__init__(); s.q = nn.Parameter(torch.randn(T, D) / D ** 0.5); s.out = nn.Linear(D, 2)
    def forward(s, tok):                                        # (B, 8·256, D) → (B, 8, 2)
        a = torch.softmax(torch.einsum("bnd,td->btn", tok, s.q) / D ** 0.5, -1); return s.out(torch.einsum("btn,bnd->btd", a, tok)), a


def load_rows():
    idx = list(csv.DictReader(INDEX.open())); n = len(idx)
    px = np.stack([arr(r["px_x_by_sample"]) for r in idx]); py = np.stack([arr(r["px_y_by_sample"]) for r in idx])
    P = np.stack([px.reshape(n, 16, 2).mean(2), py.reshape(n, 16, 2).mean(2)], -1) / RES - 1                      # (n, 16 슬롯, 2) 정규화
    vis = (np.stack([arr(r["visible_by_sample"]) for r in idx]) > 0).reshape(n, 16, 2).all(2)
    meta = dict(scenario=np.array([r["scenario"] for r in idx]), block=np.array([r["block_id"] for r in idx]), holdout=np.array([r["holdout"] == "1" for r in idx]),
                plausible=np.array([r["plausible"] == "1" for r in idx]), vid=np.array([r["video_id"] for r in idx]), primary=np.array([r["primary"] for r in idx]))
    return P.astype(np.float32), vis, meta


def baselines(P, train_m, scen):
    ctx, fut = P[:, :T], P[:, T:]; last = ctx[:, -1]; v = ctx[:, -1] - ctx[:, -2]; k = np.arange(1, T + 1, dtype=np.float32)[None, :, None]
    cv = last[:, None] + v[:, None] * k; copy = np.repeat(last[:, None], T, 1)
    tg = np.repeat(fut[train_m].mean(0, keepdims=True), len(P), 0)
    ts = np.zeros_like(fut)
    for s in np.unique(scen): ts[scen == s] = fut[train_m & (scen == s)].mean(0) if (train_m & (scen == s)).any() else fut[train_m].mean(0)
    return dict(CV=cv, COPY=copy, TEMPLATE_GLOBAL=tg, TEMPLATE_SCEN=ts)


def mae_table(pred, fut, vis, scen, blocks, rng, B=300):
    """시나리오 × 슬롯 MAE px (x, y 각각) + 시나리오 전체 MAE (block bootstrap 95 % CI)."""
    out = {}
    e = np.abs(pred - fut) * RES                                                                           # (n, 8, 2) px
    for s in sorted(set(scen)):
        m = scen == s; em = e[m]; vm = vis[m]; bl = blocks[m]; ub = np.unique(bl); ix = {u: np.where(bl == u)[0] for u in ub}
        def agg(sel): w = vm[sel]; return [float((em[sel][..., c] * w).sum() / max(w.sum(), 1)) for c in (0, 1)]
        bs = np.array([agg(np.concatenate([ix[u] for u in rng.choice(ub, len(ub))])) for _ in range(B)])
        out[s] = dict(n=int(m.sum()), mae_x=agg(np.arange(m.sum()))[0], mae_y=agg(np.arange(m.sum()))[1], ci_x=[float(np.percentile(bs[:, 0], 2.5)), float(np.percentile(bs[:, 0], 97.5))], ci_y=[float(np.percentile(bs[:, 1], 2.5)), float(np.percentile(bs[:, 1], 97.5))],
                      slot_y=[float((em[:, t, 1] * vm[:, t]).sum() / max(vm[:, t].sum(), 1)) for t in range(T)], slot_x=[float((em[:, t, 0] * vm[:, t]).sum() / max(vm[:, t].sum(), 1)) for t in range(T)])
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rep", required=True, choices=list(REP)); ap.add_argument("--head", default="both"); ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--out", required=True); ap.add_argument("--threads", type=int, default=24)
    a = ap.parse_args(); torch.set_num_threads(a.threads); torch.manual_seed(0); rng = np.random.default_rng(0); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    P, vis, M = load_rows(); fut, vfut = P[:, T:], vis[:, T:]
    use = M["plausible"]; tr = use & ~M["holdout"]; te = use & M["holdout"]; te_imp = (~M["plausible"]) & M["holdout"]
    print(f"rep {a.rep}: train {tr.sum()} test {te.sum()} (impossible holdout {te_imp.sum()})", flush=True)
    path, sl = REP[a.rep]; X = np.load(path, mmap_mode="r"); assert X.shape[0] == len(P), (X.shape, len(P))
    rows = np.where(tr | te | te_imp)[0]; t0 = time.time()
    feat = torch.empty((len(rows), T * S, D), dtype=torch.float16)
    for i in range(0, len(rows), 32):
        r = rows[i:i + 32]; feat[i:i + len(r)] = torch.from_numpy(np.ascontiguousarray(X[r, sl]))
        if i % 1024 == 0: print(f"  load {i}/{len(rows)} {time.time()-t0:.0f}s", flush=True)
    pos = {r: i for i, r in enumerate(rows)}; tri = np.array([pos[r] for r in np.where(tr)[0]]); tei = np.array([pos[r] for r in np.where(te)[0]]); tii = np.array([pos[r] for r in np.where(te_imp)[0]])
    Y = torch.from_numpy(fut); W = torch.from_numpy(vfut.astype(np.float32)); res = {"rep": a.rep, "n_train": int(tr.sum()), "n_test": int(te.sum()), "epochs": a.epochs, "heads": {}}
    base = baselines(P, tr, M["scenario"])
    res["baselines_test"] = {k: mae_table(v[te], fut[te], vfut[te], M["scenario"][te], M["block"][te], rng) for k, v in base.items()}
    for name in (["A", "B"] if a.head == "both" else [a.head]):
        head = HeadA() if name == "A" else HeadB(); opt = torch.optim.Adam(head.parameters(), lr=a.lr); n_tr = len(tri)
        for ep in range(a.epochs):
            perm = tri[rng.permutation(n_tr)]; tot = 0.0
            for i in range(0, n_tr, a.bs):
                b = perm[i:i + a.bs]; rb = torch.from_numpy(np.where(tr)[0][np.searchsorted(np.where(tr)[0], rows[b])])
                x = feat[b].float(); pred, _ = head(x); w = W[rows[b]]; loss = ((pred - Y[rows[b]]).abs().sum(-1) * w).sum() / w.sum().clamp(min=1)
                opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(b)
            if ep % 10 == 0 or ep == a.epochs - 1: print(f"  head {name} ep {ep} train L1 {tot/n_tr*RES:.1f} px  {time.time()-t0:.0f}s", flush=True)
        head.eval(); preds = {}
        with torch.no_grad():
            for tag, ii in (("test", tei), ("test_impossible", tii)):
                if len(ii) == 0: continue
                pr = torch.cat([head(feat[ii[i:i + 128]].float())[0] for i in range(0, len(ii), 128)]).numpy(); preds[tag] = pr
        r_te = rows[tei]; tab = mae_table(preds["test"], fut[r_te], vfut[r_te], M["scenario"][r_te], M["block"][r_te], rng)
        # ledge 핵심: 슬롯 7 의 y 낙하량 (예측 − 문맥 마지막 y) vs 진실
        lm = M["scenario"][r_te] == "ledge"; drop_true = (fut[r_te][lm, -1, 1] - P[r_te][lm, T - 1, 1]) * RES; drop_pred = (preds["test"][lm, -1, 1] - P[r_te][lm, T - 1, 1]) * RES
        wm = M["scenario"][r_te] == "wall"
        extra = dict(ledge_drop_px_truth_mean=float(drop_true.mean()), ledge_drop_px_pred_mean=float(drop_pred.mean()), ledge_drop_px_pred_std=float(drop_pred.std()),
                     ledge_drop_corr=float(np.corrcoef(drop_true, drop_pred)[0, 1]) if drop_true.std() > 1e-6 else None,
                     flat_v_slot7_y_pred_minus_last_px=float(((preds["test"][M["scenario"][r_te] == "flat_v", -1, 1] - P[r_te][M["scenario"][r_te] == "flat_v", T - 1, 1]) * RES).mean()))
        if "test_impossible" in preds:
            r_ti = rows[tii]; li = M["scenario"][r_ti] == "ledge"; extra["ledge_impossible_pred_drop_px"] = float(((preds["test_impossible"][li, -1, 1] - P[r_ti][li, T - 1, 1]) * RES).mean())
        res["heads"][name] = dict(params=sum(p.numel() for p in head.parameters()), test=tab, extra=extra); torch.save(head.state_dict(), out / f"head_{name}_{a.rep}.pt")
        np.savez(out / f"preds_{name}_{a.rep}.npz", vid=M["vid"][r_te], pred=preds["test"], truth=fut[r_te], vis=vfut[r_te], scenario=M["scenario"][r_te], block=M["block"][r_te], ctx_last=P[r_te][:, T - 1])
        print(f"head {name}: " + " | ".join(f"{s} y {v['mae_y']:.1f} x {v['mae_x']:.1f}" for s, v in tab.items()), "\n   extra", json.dumps(extra), flush=True)
    json.dump(res, open(out / f"result_{a.rep}.json", "w"), indent=1); print("saved", out / f"result_{a.rep}.json")


if __name__ == "__main__":
    main()
