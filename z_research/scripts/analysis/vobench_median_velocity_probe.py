"""차원별 중앙값 latent 에서 속도가 읽히나 — L1 최적해 (중앙값) 가 속도를 지우는가 (2026-10-27).

입력: vobench_ar_token_transfer.py extract 의 토큰 memmap (h = 블록별 LN(target_encoder), 16 블록 × (256, 1280)), 가능 · 물체 있는 1,200 영상.
속도 자: 블록의 **물체 중심 토큰** 하나 (1280) → 그 블록 두 장 사이 물체 이동 (u, v) [칸 / 0.1 s, 1 칸 = 32 px @ 512].
  선형층 하나 (AdamW, 특징 표준화), block 단위 50/50 분할 (layer_velocity_probe 와 같음). 물체가 보이는 블록만.
중앙값 실험 (test 영상, 움직이는 운동만):
  (A) 물체를 따라간 중앙값 — 한 영상의 보이는 블록 k 개 (무작위) 에서 **각 블록의 물체 중심 토큰**을 모아 차원별 중앙값 → LN → 자.
      = "자리는 맞히고 속도만 불확실한" 미래들의 L1 최적해. 비교: 그 블록들 참 속도의 중앙값 · 평균 (L2 최적해) 판.
  (B) 자리 고정 중앙값 — 같은 k 블록에서 **한 칸 (고른 블록들 가운데 블록의 물체 중심 칸)** 의 토큰을 모아 중앙값 → LN → 자.
      = 자리도 불확실할 때. 비교: 같은 칸에서 물체다움 점수 (참 h 로 학습한 토큰 물체 자, ar_token_where 의 objness.pt).
판정: 읽힌 속도 vs 참 속도 중앙값의 기울기 (1 = 보존, 0 = 지워짐) · R².

  $PY z_research/scripts/analysis/vobench_median_velocity_probe.py --tok_dir <scratch>/tok_vast70
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
import vobench_ar_layer_velocity_probe as lv      # noqa: E402
import vobench_ar_token_transfer as tt             # noqa: E402

G, D, NB = 16, 1280, 16
dev = "cuda"
LN = lambda t: torch.nn.functional.layer_norm(t, (t.size(-1),))


def labels(vids):
    """블록마다 물체 중심 칸 (x, y) · 속도 (du, dv) [칸 / 0.1 s] · 보임."""
    import csv
    rows = {r["video_id"]: r for r in lv.rows_obj()}
    lab, XY, V, VIS, MOT = {}, [], [], [], []
    for v in vids:
        r = rows[v]; arm = "A" if r["variant"] == "pos_a" else "B"
        b = lv.FR / Path(r["file_name"]).parent
        if b not in lab:
            lab[b] = json.load(open(b / "label.json"))["labels"]
        L = lab[b][arm]; xy, vv, vis = [], [], []
        for k in range(NB):
            f0, f1 = L[3 * (2 * k)], L[3 * (2 * k + 1)]
            xy.append([(f0["u"] + f1["u"]) / 64, (f0["v"] + f1["v"]) / 64])
            vv.append([(f1["u"] - f0["u"]) / 32, (f1["v"] - f0["v"]) / 32])
            vis.append(max(f0["hidden"], f1["hidden"]) < 0.5)
        XY.append(xy); V.append(vv); VIS.append(vis); MOT.append(r["motion"])
    return np.array(XY, np.float32), np.array(V, np.float32), np.array(VIS), np.array(MOT)


def main(tok_dir, n_sets=60):
    vids, Hs, _ = tt.load_tokens(tok_dir)
    XY, V, VIS, MOT = labels(vids)
    meta = tt.block_labels(vids)[2]
    g = np.array([m["block"] for m in meta])
    tr = lv.block_split(g, MOT); te = ~tr
    cell = lambda xy: int(np.clip(np.floor(xy[1]), 0, G - 1)) * G + int(np.clip(np.floor(xy[0]), 0, G - 1))

    # ---- 물체 중심 토큰 모으기
    def gather(idx):
        H = tt.cat_rows(Hs, idx)                                                   # (m, 16, 256, D) fp16
        X, Y, own = [], [], []
        for j, i in enumerate(idx):
            for k in range(NB):
                if VIS[i, k]:
                    X.append(H[j, k, cell(XY[i, k])]); Y.append(V[i, k]); own.append((j, k))
        return H, np.stack(X).astype(np.float32), np.stack(Y), own
    itr, ite = np.where(tr)[0], np.where(te)[0]
    _, Xtr, Ytr, _ = gather(itr)
    Hte, Xte, Yte, _ = gather(ite)

    X = torch.tensor(Xtr, device=dev); Y = torch.tensor(Ytr, device=dev)
    mu, sd = X.mean(0), X.std(0) + 1e-6; ym, ys = Y.mean(0), Y.std(0) + 1e-6
    head = torch.nn.Linear(D, 2).to(dev); opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(3000):
        loss = torch.nn.functional.mse_loss(head((X - mu) / sd), (Y - ym) / ys); opt.zero_grad(); loss.backward(); opt.step()
    probe = lambda Z: (head((torch.as_tensor(Z, device=dev, dtype=torch.float32) - mu) / sd) * ys + ym).detach().cpu().numpy()
    r2 = lambda y, p: float(1 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())
    pte = probe(Xte)
    res = {"n_train_tokens": len(Xtr), "n_test_tokens": len(Xte), "probe_test_r2_u": r2(Yte[:, 0], pte[:, 0]),
           "probe_test_r2_v": r2(Yte[:, 1], pte[:, 1]), "probe_test_mae_cells": float(np.abs(Yte - pte).mean())}
    # 물체 토큰 하나를 LN 다시 한 것만으로 자가 흔들리지 않는지 (h 는 이미 LN — 거의 항등이어야 함)
    res["probe_test_r2_u_after_LN"] = r2(Yte[:, 0], probe(LN(torch.tensor(Xte)).numpy())[:, 0])

    # ---- 중앙값 실험
    rng = np.random.default_rng(0)
    out = {}
    for k in (3, 5, 8):
        rec = {key: [] for key in ("true_med", "true_mean", "A_med", "A_mean", "A_readmed", "B_med", "B_true_at_cell_frac")}
        for j, i in enumerate(ite):
            if MOT[i] == "flat_static":
                continue
            vb = np.where(VIS[i])[0]
            if len(vb) < k:
                continue
            for _ in range(n_sets // 10):
                sel = np.sort(rng.choice(vb, k, replace=False))
                toks = torch.tensor(np.stack([Hte[j, b, cell(XY[i, b])] for b in sel]).astype(np.float32))   # (k, D)
                vt = V[i, sel]
                rec["true_med"].append(np.median(vt, 0)); rec["true_mean"].append(vt.mean(0))
                rec["A_med"].append(probe(LN(toks.median(0).values)[None])[0])
                rec["A_mean"].append(probe(LN(toks.mean(0))[None])[0])
                rec["A_readmed"].append(np.median(probe(toks), 0))                  # 각각 읽고 중앙값 (선형 기준)
                c = cell(XY[i, sel[k // 2]])                                          # (B) 가운데 블록의 물체 칸에 고정
                tb = torch.tensor(np.stack([Hte[j, b, c] for b in sel]).astype(np.float32))
                rec["B_med"].append(probe(LN(tb.median(0).values)[None])[0])
                rec["B_true_at_cell_frac"].append(np.mean([cell(XY[i, b]) == c for b in sel]))
        R = {kk: np.array(v) for kk, v in rec.items()}
        def fit(pred, ref):                                                           # u 성분 (운동 대부분이 가로)
            x, y = ref[:, 0], pred[:, 0]
            sl = float((x * y).sum() / (x * x).sum())
            return {"slope_u": sl, "r2_u": r2(x, y), "mean_abs_read_u": float(np.abs(y).mean()), "mean_abs_ref_u": float(np.abs(x).mean())}
        out[f"k{k}"] = {"n_sets": len(R["true_med"]),
                        "A_median_vs_true_median": fit(R["A_med"], R["true_med"]),
                        "A_mean_vs_true_mean": fit(R["A_mean"], R["true_mean"]),
                        "A_read_each_then_median_vs_true_median": fit(R["A_readmed"], R["true_med"]),
                        "B_fixed_cell_median_vs_true_median": fit(R["B_med"], R["true_med"]),
                        "B_frac_blocks_object_in_that_cell": float(R["B_true_at_cell_frac"].mean())}
    res["median_sets"] = out
    od = REPO / "z_analysis/median_latent/velocity_probe"; od.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(od / "result.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tok_dir", required=True); a = ap.parse_args(); main(a.tok_dir)
