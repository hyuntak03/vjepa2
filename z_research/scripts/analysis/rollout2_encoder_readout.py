#!/usr/bin/env python3
"""RollOut_v2 — 인코더 대조. 인코더가 **32 frames 전부** 를 본 특징의 **미래 8 튜블릿** 으로 위치 readout 을 학습셋에서 정하고 v2 에 test.
  --encoder z       context encoder (online) 에 32 frames 를 마스크 없이 넣은 것 (`isolated_ctx:0_32`, LN)   ← 주 (2026-09-10 사용자 결정)
                    ⚠️ probing 관례의 z (`ctx_masked`, 문맥 16 frames) 와 다르다. 여기 z 는 32 frames 전부를 본 context encoder 다
  --encoder h       target encoder (EMA) 32 frames (`target`, LN)                                        ← 대조
p 자 (`rollout2_fit_readout.py`, `rollout2_attn_readout.py`) 와 같은 라벨·같은 슬롯·같은 두 자 형태:
  spatial   256 토큰 평균 → (1281, 2) 최소제곱
  attentive 쿼리 1개 softmax 가중합 → Linear(D, 2)
인코더는 실제 미래 frame 을 본 특징이라 "predictor 의 미래가 맞다" 를 말하는 게 아니라 **자의 학습 분포가 인코더 공간에서 충분한가** 를 재는 대조다.
저장: exp_results/<train>/{spatial_pooling, attentive_pooling}/{z|h}/{w.npy | attn.pt, fit.json, test.json, preds.npz}

  python z_research/scripts/analysis/rollout2_encoder_readout.py --encoder z|h [--epochs 300]
  python z_research/scripts/analysis/rollout2_encoder_readout.py --encoder z|h --test-only   # 저장된 자로 test 만
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rollout2_test_readout as rt                                   # noqa: E402
from rollout2_fit_readout import fit, apply                          # noqa: E402
from rollout2_attn_readout import AttnReadout, labels, Loader, TR_CACHE, TR_INDEX   # noqa: E402

S, T, D, RES = rt.S, rt.T, rt.D, rt.RES
CACHE_ROOT = Path("/local_datasets/world/world_analysis/cache")
ENC = {"z": dict(tr=rt.TR_CACHE_CTX32, te=CACHE_ROOT / "rollout_v2_ctx32_vith", file="isolated_ctx_0_32.npy", out="z", label="context encoder 32 frames"),
       "h": dict(tr=TR_CACHE, te=rt.CACHE, file="target.npy", out="h", label="target encoder 32 frames")}
FUT = slice(T * S, 2 * T * S)                                        # 4096 토큰 중 뒤 8 튜블릿 (frames 17–32)


def load_future(path, n, dev, bs=64):
    """(n, 4096, D) 캐시의 미래 8 튜블릿 → GPU fp16 (n, 2048, D). 학습셋은 4.7 GB, v2 는 스트리밍만 한다."""
    H = np.load(path, mmap_mode="r"); out = torch.empty((n, T * S, D), dtype=torch.float16, device=dev)
    for k in range(0, n, bs):
        out[k:k + bs] = torch.from_numpy(np.asarray(H[k:k + bs, FUT])).to(dev)
    return out


def table(pred, L, inf, scen, plaus, title):
    rep = {}
    print(f"\n[{title}]\n{'scenario':<8} {'n':>4}  {'R²x':>6} {'R²y':>6}  {'MAEx':>6} {'MAEy':>6}   {'βx':>5} {'βy':>5}   across-clip corr x / y")
    f = lambda v: "   -  " if v is None else f"{v:6.2f}"
    for s in rt.SCEN:
        k = (scen == s) & (plaus == 1); m = inf[k]; p = pred[k]; y = L[k]; mv = rt.MOVING[s]
        cx = float(np.mean([np.corrcoef(p[:, t, 0], y[:, t, 0])[0, 1] for t in range(T)])) if y[:, :, 0].std(0).mean() > 0.01 else None
        cy = float(np.mean([np.corrcoef(p[:, t, 1], y[:, t, 1])[0, 1] for t in range(T)])) if y[:, :, 1].std(0).mean() > 0.01 else None
        rec = {"n": int(k.sum()), "r2_x": rt.r2(p[:, :, 0][m], y[:, :, 0][m]) if "x" in mv else None, "r2_y": rt.r2(p[:, :, 1][m], y[:, :, 1][m]) if "y" in mv else None,
               "mae_px_x": float(np.abs(p - y)[:, :, 0][m].mean() * RES), "mae_px_y": float(np.abs(p - y)[:, :, 1][m].mean() * RES),
               "beta_x": rt.beta(rt.slope(p[:, :, 0]), rt.slope(y[:, :, 0])) if "x" in mv else None, "beta_y": rt.beta(rt.slope(p[:, :, 1]), rt.slope(y[:, :, 1])) if "y" in mv else None,
               "corr_x": cx, "corr_y": cy}
        rep[s] = rec
        print(f"{s:<8} {rec['n']:>4}  {f(rec['r2_x'])} {f(rec['r2_y'])}  {rec['mae_px_x']:6.1f} {rec['mae_px_y']:6.1f}   {f(rec['beta_x'])[1:]} {f(rec['beta_y'])[1:]}   {f(rec['corr_x'])} / {f(rec['corr_y'])}")
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--encoder", choices=sorted(ENC), default="z")
    ap.add_argument("--epochs", type=int, default=300); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--test-only", action="store_true", help="저장된 w.npy / attn.pt 로 v2 test 만 (재학습 없음). 2026-09-19 감속 시나리오 추가용"); a = ap.parse_args()
    E = ENC[a.encoder]; dev = "cuda"; torch.manual_seed(0)
    OUT_SP = rt.RES_ROOT / "spatial_pooling" / E["out"]; OUT_AT = rt.RES_ROOT / "attentive_pooling" / E["out"]
    if a.test_only:
        w = np.load(OUT_SP / "w.npy"); model = AttnReadout().to(dev); model.load_state_dict(torch.load(OUT_AT / "attn.pt", map_location=dev)); model.eval()
        print(f"test-only: {OUT_SP / 'w.npy'}, {OUT_AT / 'attn.pt'}"); return _test_v2(E, a, w, model, dev, OUT_SP, OUT_AT)
    idx, L, inf = labels(TR_INDEX); n = len(idx); print(f"학습셋 {n} clip, visible 미래 튜블릿 {inf.sum()} / {inf.size}")
    if json.loads((E["tr"] / "meta.json").read_text())["video_ids"] != [r["video_id"] for r in idx]:
        sys.exit("학습셋 캐시 video_ids 불일치")
    Hm = np.load(E["tr"] / E["file"], mmap_mode="r"); ld = Loader(Hm[:, FUT], dev, a.bs); Y = torch.from_numpy(L).to(dev); M = torch.from_numpy(inf).to(dev); mm = inf.ravel()
    # ── spatial: 풀링 → 최소제곱 (visible 행만) ──
    Xp = torch.cat([ld.batch(torch.arange(k, min(k + a.bs, n))).float().reshape(-1, T, S, D).mean(2).cpu() for k in range(0, n, a.bs)]).numpy().astype(np.float64)
    w = fit(Xp.reshape(-1, D)[mm], L.reshape(-1, 2)[mm]); OUT_SP.mkdir(parents=True, exist_ok=True); np.save(OUT_SP / "w.npy", w)
    pr = apply(Xp.reshape(-1, D), w).reshape(n, T, 2)
    fit_sp = {"n_clip": n, "rows_visible": int(inf.sum()), "source": E["label"] + ", future 8 tubelets, spatial mean", "r2_x_fit": rt.r2(pr[:, :, 0].ravel()[mm], L[:, :, 0].ravel()[mm]), "r2_y_fit": rt.r2(pr[:, :, 1].ravel()[mm], L[:, :, 1].ravel()[mm]),
              "mae_px_x": float(np.abs(pr - L)[:, :, 0].ravel()[mm].mean() * RES), "mae_px_y": float(np.abs(pr - L)[:, :, 1].ravel()[mm].mean() * RES)}
    print(f"{a.encoder} spatial fit: R² x {fit_sp['r2_x_fit']:.3f} y {fit_sp['r2_y_fit']:.3f}  MAE {fit_sp['mae_px_x']:.1f} / {fit_sp['mae_px_y']:.1f} px")
    (OUT_SP / "fit.json").write_text(json.dumps(fit_sp, indent=1))
    # ── attentive (visible 마스크 손실; 학습 텐서는 CPU, 배치만 GPU) ──
    model = AttnReadout().to(dev); opt = torch.optim.Adam(model.parameters(), lr=a.lr); t0 = time.time()
    for ep in range(a.epochs):
        perm = torch.randperm(n); tot = 0.0; cnt = 0
        for k in range(0, n, a.bs):
            b = perm[k:k + a.bs]; tok = ld.batch(b).float().reshape(-1, S, D); bd = b.to(dev); y = Y[bd].reshape(-1, 2); m = M[bd].reshape(-1)
            pred, _ = model(tok); loss = (((pred - y) ** 2).mean(1) * m).sum() / m.sum().clamp(min=1); opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * m.sum().item(); cnt += m.sum().item()
        if ep % 100 == 0 or ep == a.epochs - 1:
            print(f"  epoch {ep:3d}  train MSE {tot/cnt:.5f}  ({time.time()-t0:.0f}s)", flush=True)
    model.eval(); OUT_AT.mkdir(parents=True, exist_ok=True); torch.save(model.state_dict(), OUT_AT / "attn.pt")
    with torch.no_grad():
        pr = torch.cat([model(ld.batch(torch.arange(k, min(k + a.bs, n))).float().reshape(-1, S, D))[0] for k in range(0, n, a.bs)]).reshape(n, T, 2).cpu().numpy()
    fit_at = {"n_clip": n, "rows_visible": int(inf.sum()), "params": sum(p.numel() for p in model.parameters()), "epochs": a.epochs, "source": E["label"] + ", future 8 tubelets, 1-query attention",
              "r2_x_fit": rt.r2(pr[:, :, 0].ravel()[mm], L[:, :, 0].ravel()[mm]), "r2_y_fit": rt.r2(pr[:, :, 1].ravel()[mm], L[:, :, 1].ravel()[mm]), "mae_px_x": float(np.abs(pr - L)[:, :, 0].ravel()[mm].mean() * RES), "mae_px_y": float(np.abs(pr - L)[:, :, 1].ravel()[mm].mean() * RES)}
    print(f"{a.encoder} attentive fit: R² x {fit_at['r2_x_fit']:.3f} y {fit_at['r2_y_fit']:.3f}  MAE {fit_at['mae_px_x']:.1f} / {fit_at['mae_px_y']:.1f} px")
    (OUT_AT / "fit.json").write_text(json.dumps(fit_at, indent=1)); del ld, Y, Xp

    _test_v2(E, a, w, model, dev, OUT_SP, OUT_AT)


def _test_v2(E, a, w, model, dev, OUT_SP, OUT_AT):
    """spatial 자 w 와 attentive 자 model 로 v2 전체를 test 한다."""
    # ── v2 test (h 스트리밍) ──
    idx, L, inf = labels(rt.INDEX); n = len(idx)
    if json.loads((E["te"] / "meta.json").read_text())["video_ids"] != [r["video_id"] for r in idx]:
        sys.exit("v2 캐시 video_ids 불일치")
    Hv = np.load(E["te"] / E["file"], mmap_mode="r"); p_sp = np.empty((n, T, 2)); p_at = np.empty((n, T, 2), np.float32); t0 = time.time()
    with torch.no_grad():
        for k in range(0, n, 64):
            tok = torch.from_numpy(np.asarray(Hv[k:k + 64, FUT])).to(dev).float().reshape(-1, S, D)
            p_sp[k:k + 64] = apply(tok.mean(1).double().cpu().numpy(), w).reshape(-1, T, 2)
            p_at[k:k + 64] = model(tok)[0].reshape(-1, T, 2).cpu().numpy()
            if k % 1280 == 0:
                print(f"    v2 {a.encoder} {k}/{n} {time.time()-t0:.0f}s", flush=True)
    scen = np.array([r["scenario"] for r in idx]); plaus = np.array([int(r["plausible"]) for r in idx]); vid = np.array([r["video_id"] for r in idx])
    for out, pred, title in ((OUT_SP, p_sp, f"{a.encoder} spatial → v2"), (OUT_AT, p_at, f"{a.encoder} attentive → v2")):
        np.savez(out / "preds.npz", video_id=vid, pred=pred, truth=L, in_frame=inf, scenario=scen, plausible=plaus)
        (out / "test.json").write_text(json.dumps({"per_scenario": table(pred, L, inf, scen, plaus, title)}, indent=1))
    print(f"→ {OUT_SP}\n→ {OUT_AT}")


if __name__ == "__main__":
    main()
