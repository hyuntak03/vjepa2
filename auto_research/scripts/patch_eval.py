#!/usr/bin/env python3
"""[T] 방법 C 판정 (1) 안에서: K710 보류 조각 세트에서 단단함 vs 무름 조각의 운반 계수 — 데이터 (정답) 와 모델별 (auto_research, 2026-10-21).

  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 /venv/vjepa/bin/python -W ignore auto_research/scripts/patch_eval.py --ft ft_l1 ft_l1_lr1e4 ft_patch ft_patch_lr1e4

t = 4, 5, 6 (문맥 블록 0..t → 블록 t+1, TF 한 걸음). 새로 덮인 칸 = 블록 t+1 (둘째 장) 에서 조각이 0.8 이상 덮고 블록 t 에서 전혀 안 덮은 칸.
옮긴 직전 블록 토큰 = h_t(c − d), d = 두 블록 둘째 장 사이 조각 이동 (칸, 반올림).
회귀: y ≈ a·(같은 칸 5 블록 평균) + b·h_t(c − d), 종류별 (단단함 / 무름) 로 모아 풂. y = 정답 → 데이터 b, y = 모델 예측 → 모델 b."""
import argparse, json, os, sys
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--ft", nargs="*", default=[])
ap.add_argument("--ftdir", default="/workspace/auto_research_cache/ar_ft")
ap.add_argument("--data", default="/workspace/auto_research_cache/natural_patch/k710_holdout_patch")
args = ap.parse_args()
sys.path.insert(0, os.path.dirname(__file__))
sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT                                                     # noqa: E402
torch.set_num_threads(8)
dev = torch.device("cuda"); G, S = 16, 256
models = {"base": FT.NoiseWrap(FT.load_pred("natural_ar_vw_state", "latest.pt"), 32, 0.0).to(dev).eval()}
for n in args.ft:
    ck = torch.load(os.path.join(args.ftdir, f"{n}.pt"), map_location="cpu", weights_only=False)
    W = FT.NoiseWrap(FT.load_pred("natural_ar_vw_state", "latest.pt"), 32, 0.0); W.pred.load_state_dict(ck["predictor"]); models[n] = W.to(dev).eval()
H = np.load(args.data + "_h.npy", mmap_mode="r"); META = json.load(open(args.data + "_patch.json"))
c16 = np.arange(G) * 16.0


def cover(tl, s):
    x0, y0 = tl
    ox = np.clip(np.minimum(c16 + 16, x0 + s) - np.maximum(c16, x0), 0, 16) / 16
    oy = np.clip(np.minimum(c16 + 16, y0 + s) - np.maximum(c16, y0), 0, 16) / 16
    return (oy[:, None] * ox[None, :]).reshape(-1)


ACC = {}
for w, m in enumerate(META):
    h = torch.from_numpy(np.asarray(H[w], np.float32)).to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        P = {n: W.tf(h[:7].reshape(1, -1, h.size(-1))).float()[0].view(7, S, -1) for n, W in models.items()}
    kind = "soft" if m["soft"] else "rigid"
    for t in (4, 5, 6):
        cn, co = cover(m["tl"][2 * t + 3], m["size"]), cover(m["tl"][2 * t + 1], m["size"])
        new = np.where((cn >= 0.8) & (co == 0))[0]
        if new.size == 0:
            continue
        dx = int(round((m["tl"][2 * t + 3][0] - m["tl"][2 * t + 1][0]) / 16)); dy = int(round((m["tl"][2 * t + 3][1] - m["tl"][2 * t + 1][1]) / 16))
        ys, xs = np.divmod(new, G); sy, sx = np.clip(ys - dy, 0, G - 1), np.clip(xs - dx, 0, G - 1)
        c = torch.from_numpy(new).to(dev); sj = torch.from_numpy(sy * G + sx).to(dev)
        X = torch.stack([h[t - 4:t + 1].mean(0)[c], h[t][sj]], -1).reshape(-1, 2).double()
        for n, yv in [("data", h[t + 1][c])] + [(n, P[n][t][c]) for n in models]:
            A = ACC.setdefault((n, kind), [torch.zeros(2, 2, dtype=torch.float64, device=dev), torch.zeros(2, dtype=torch.float64, device=dev), 0])
            A[0] += X.T @ X; A[1] += X.T @ yv.reshape(-1).double(); A[2] += 1
res = {}
for n in ["data"] + list(models):
    r = {}
    for kind in ("rigid", "soft"):
        A = ACC.get((n, kind))
        if A:
            r[kind] = float(torch.linalg.solve(A[0], A[1])[1]); r[f"n_{kind}"] = A[2]
    r["split"] = r.get("rigid", np.nan) - r.get("soft", np.nan)
    res[n] = r
    print(f"  {n:16s} 운반 계수 단단함 {r.get('rigid', float('nan')):.3f} · 무름 {r.get('soft', float('nan')):.3f} · 갈림 {r['split']:+.3f}  (쌍 {r.get('n_rigid', 0)} · {r.get('n_soft', 0)})", flush=True)
json.dump(res, open(os.path.join(os.path.dirname(__file__), "../results/patch_eval.json"), "w"), indent=1)
