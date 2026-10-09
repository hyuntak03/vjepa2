#!/usr/bin/env python3
"""H6h — IntPhys 1 복사 기준선에서 z 에 씌운 LN 을 빼면 점수가 어떻게 되나 (2026-09-25, 사용자 요청).

H6 의 copy (copyz) = LN(z)[문맥 마지막 튜블릿] 을 모든 미래 튜블릿에 복사. 여기서 LN 은 affine 없는 layer_norm 이다.
z 는 context encoder 출력이라 이미 encoder 마지막 `self.norm` (학습된 γ, β 가 붙은 LayerNorm) 을 거쳤다
(src/models/vision_transformer.py:211). 즉 H6 의 LN 은 그 γ, β 를 걷어 내 표적 LN(h) 과 같은 척도로 맞추는 역할이다.

열 (미래 튜블릿 u 마다 전 토큰 mean L1; 창·배치·dtype 은 H6e 와 같다 → p_full · copy_ln 은 H6e 와 비트 단위로 같아야 한다):
  p_full        |predictor(z)_u     − LN(h)_u|     표준 채점 (재현 검증)
  copy_ln       |LN(z)_T            − LN(h)_u|     H6 copy (재현 검증: skip2_w32 85.00)
  copy_raw      |z_T                − LN(h)_u|     ← 요청: z 에 LN 을 안 씌운 복사, 표적은 표준 그대로
  copy_raw_rawh |z_T                − h_u|         보조: 양쪽 다 LN 없이 (각 encoder 의 γ, β 가 붙은 출력끼리)
  (h = target_encoder(창 전체), z_T = context_encoder(문맥)[마지막 튜블릿]; 모든 u 에 같은 z_T)
진단 열 (창마다, 튜블릿 u 무관 값은 반복): absz_raw = mean|z_T|, absz_ln = mean|LN(z_T)|, absh_raw = mean|h_u|, absh_ln = mean|LN(h_u)|

  추출 (GPU, rank 마다 1 장):  CUDA_VISIBLE_DEVICES=r python auto_research/scripts/h6h_intphys1_copy_noln.py --rank r --world 8
  분석 (CPU):                  python auto_research/scripts/h6h_intphys1_copy_noln.py --analyze
"""
from __future__ import annotations
import argparse, collections, csv, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

S_TOK, D = 256, 1280
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "auto_research/exp_results/h6"
AUX = ROOT / "auto_research/_stage/IntPhys1_dev_by_scene"
COLS = ["video_i", "combo_i", "start", "C", "j",
        "p_full", "copy_ln", "copy_raw", "copy_raw_rawh", "absz_raw", "absz_ln", "absh_raw", "absh_ln"]
METHODS = ["p_full", "copy_ln", "copy_raw", "copy_raw_rawh"]


def extract(a):
    import torch
    import torch.nn.functional as F
    import arlib  # noqa: E402
    from evals.world_model_analysis.eval import _intphys1_windows  # noqa: E402
    dev = torch.device("cuda:0")
    base = arlib.load_cfg("intphys1_sliding__intphys1_dev_vith")
    cfg = dict(base, model=dict(base["model"], window_size=64))
    ds = arlib.WMADataset(cfg)
    bundle = arlib.build_bundle(cfg, dev, window=64)
    W = base["surprise"]["intphys1"]
    combos = []
    for sk in (2, 5):
        for wz in (16, 32):
            cl = [m * wz // 16 for m in W["context_mult"]]
            wins = _intphys1_windows(ds.n_frames, sk, wz, cl, int(W["stride"]), 2, str(W["frame_budget"]))
            if wins:
                combos.append((f"skip{sk}_w{wz}", wins))
    idx = list(range(a.rank, len(ds), a.world))
    if a.limit:
        idx = idx[: a.limit]
    ac = arlib.autocast_ctx(cfg)
    rows = []; t0 = time.time()
    for n_done, i in enumerate(idx):
        clip = ds.clip(i)
        for ci_, (name, wins) in enumerate(combos):
            by_start = {}
            for C, cf, tf in wins:
                by_start.setdefault(cf[0], []).append((C, cf, tf))
            starts = sorted(by_start)
            X = torch.stack([clip[:, by_start[st][0][1] + by_start[st][0][2]] for st in starts]).to(dev, bundle.dtype)
            nS, _, wz = X.shape[:3]
            U = wz // 2
            with torch.inference_mode(), ac():
                hraw = bundle.target_encoder(X)
                hfull = F.layer_norm(hraw, (D,)).float().reshape(nS, U, S_TOK, D)
                hraw = hraw.float().reshape(nS, U, S_TOK, D)
                for si, st in enumerate(starts):
                    for C, cf, tf in by_start[st]:
                        Tc = C // 2
                        x = X[si:si + 1]
                        ci, ti = arlib.ci_ti(1, C, wz - C, dev)
                        z = bundle.context_encoder(x, masks=[ci])
                        p = bundle.predictor(z, ci, ti, mask_index=0).float().reshape(U - Tc, S_TOK, D)
                        zT_ln = F.layer_norm(z.float(), (D,)).reshape(Tc, S_TOK, D)[Tc - 1]
                        zT_raw = z.float().reshape(Tc, S_TOK, D)[Tc - 1]
                        az_raw, az_ln = zT_raw.abs().mean().item(), zT_ln.abs().mean().item()
                        for j in range(U - Tc):
                            u = Tc + j
                            hl, hr = hfull[si, u], hraw[si, u]
                            rows.append((i, ci_, st, C, j,
                                         (p[j] - hl).abs().mean().item(), (zT_ln - hl).abs().mean().item(),
                                         (zT_raw - hl).abs().mean().item(), (zT_raw - hr).abs().mean().item(),
                                         az_raw, az_ln, hr.abs().mean().item(), hl.abs().mean().item()))
        if n_done % 5 == 0:
            el = time.time() - t0
            print(f"[{a.rank}] {n_done+1}/{len(idx)}  {el/60:.1f} 분  남은 {el/(n_done+1)*(len(idx)-n_done-1)/60:.1f} 분", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"_smoke{a.limit}" if a.limit else ""
    np.savez(OUT / f"h6h_copy_noln_vith{tag}_r{a.rank}.npz", rows=np.array(rows, np.float64), cols=np.array(COLS),
             combos=np.array([c for c, _ in combos]), video_ids=np.array([r.video_id for r in ds.records]))
    print("DONE", len(rows), flush=True)


# ── 분석 ─────────────────────────────────────────────────────────────────────
B = 10000
rng = np.random.default_rng(0)


def load(glob):
    parts = sorted(OUT.glob(glob)); z = np.load(parts[0])
    combos, vids = [str(c) for c in z["combos"]], [str(v) for v in z["video_ids"]]
    R = np.concatenate([np.load(f)["rows"] for f in parts])
    return R, combos, vids


def garrido(R, combo_i, col, vids):
    """G(v) = mean_s min_C mean_j L1 (Filtered → AvgSurprise)."""
    W = collections.defaultdict(list)
    m = R[:, 1] == combo_i
    for r in R[m]:
        W[(int(r[0]), int(r[2]), int(r[3]))].append(r[col])
    vs = collections.defaultdict(lambda: collections.defaultdict(list))
    for (v, s, C), l in W.items():
        vs[v][s].append(np.mean(l))
    return {vids[v]: float(np.mean([min(cs) for cs in st.values()])) for v, st in vs.items()}


def analyze():
    R, combos, vids = load("h6h_copy_noln_vith_r*.npz")
    ci = {c: i for i, c in enumerate(COLS)}
    pairs = list(csv.DictReader((AUX / "pairs.csv").open()))
    scene = np.array([p["scene"] for p in pairs]); prin = np.array([p["principle"][:2] for p in pairs])

    def ok(G):
        return np.array([1.0 if G[p["imp"]] > G[p["pos"]] else (0.5 if G[p["imp"]] == G[p["pos"]] else 0.0) for p in pairs])

    def macro(v):
        est, bs = [], np.zeros(B)
        for pr in ("O1", "O2", "O3"):
            m = prin == pr; u, inv = np.unique(scene[m], return_inverse=True)
            s = np.bincount(inv, v[m], len(u)); c = np.bincount(inv, None, len(u))
            idx = rng.integers(0, len(u), (B, len(u)))
            bs += s[idx].sum(1) / c[idx].sum(1) / 3; est.append(v[m].mean())
        return [round(100 * float(np.mean(est)), 2), round(100 * np.percentile(bs, 2.5), 2), round(100 * np.percentile(bs, 97.5), 2)]

    out = {"_doc": "macro = [추정 %, 95% CI lo, hi] (principle 층화 scene bootstrap, B=10000, tie=0.5). diff_* = 같은 쌍 위 정답 차의 평균 (pt).",
           "n_rows": int(len(R)), "n_videos": int(len(np.unique(R[:, 0]))), "cells": {}, "scale": {}}
    # H6e 와 재현 대조 (p_full, copy_ln 은 H6e 의 p_full, copyz_full 과 같아야 한다)
    try:
        E, ecombos, evids = load("causal_l1_vith_r*.npz")
        ekey = {(evids[int(r[0])], ecombos[int(r[1])], int(r[2]), int(r[3]), int(r[4])): (r[7], r[8]) for r in E}
        dp, dc, miss = [], [], 0
        for r in R:
            k = (vids[int(r[0])], combos[int(r[1])], int(r[2]), int(r[3]), int(r[4]))
            if k not in ekey:
                miss += 1; continue
            dp.append(abs(r[ci["p_full"]] - ekey[k][0])); dc.append(abs(r[ci["copy_ln"]] - ekey[k][1]))
        out["repro_vs_h6e"] = {"n_matched": len(dp), "n_missing": miss,
                               "p_full_absdiff_max": float(np.max(dp)), "copy_ln_absdiff_max": float(np.max(dc))}
    except Exception as e:  # noqa: BLE001
        out["repro_vs_h6e"] = f"skip: {e}"
    for k, cb in enumerate(combos):
        O = {m: ok(garrido(R, k, ci[m], vids)) for m in METHODS}
        cell = {m: macro(O[m]) for m in METHODS}
        for a_, b_ in (("p_full", "copy_ln"), ("p_full", "copy_raw"), ("copy_raw", "copy_ln"), ("copy_raw_rawh", "copy_ln")):
            cell[f"diff_{a_}_minus_{b_}"] = macro(O[a_] - O[b_] + 0.5)
            cell[f"diff_{a_}_minus_{b_}"] = [round(x - 50.0, 2) for x in cell[f"diff_{a_}_minus_{b_}"]]
            cell[f"discordant_{a_}_vs_{b_}"] = [int(((O[a_] - O[b_]) > 0).sum()), int(((O[a_] - O[b_]) < 0).sum())]
        cell["ties"] = {m: int((O[m] == 0.5).sum()) for m in METHODS}
        out["cells"][cb] = cell
        m = R[:, 1] == k
        out["scale"][cb] = {c: round(float(R[m, ci[c]].mean()), 4) for c in ("absz_raw", "absz_ln", "absh_raw", "absh_ln")}
        out["scale"][cb].update({f"L1_{c}": round(float(R[m, ci[c]].mean()), 4) for c in METHODS})
    (OUT / "h6h_copy_noln.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    analyze() if a.analyze else extract(a)
