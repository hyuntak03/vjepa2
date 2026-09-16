#!/usr/bin/env python3
"""시간 정렬 행렬 — predictor 의 미래 슬롯 i 가 실제 어느 시점 j 의 장면과 가장 닮았나 (캐시만, GPU 불필요, 파라미터 0).

2026-09-15 H1 파일럿을 레포로 옮긴 것 (스크래치 h1_pilot.py). 결과: z_research/context_encoder_analysis/exp_results/time_alignment/
읽는 법·단서는 z_research/context_encoder_analysis/Archive/STATE_EVOLUTION_HYPOTHESES_2026-09-15.md §3.

D_ph[i][j] = mean_{s in S} mean_d |p[i,s,d] - LN(h)[j,s,d]|   (i = 0..7 predicted future tubelet, j = 0..15 true tubelet)
D_zh[i][j] = mean_{s in S} |LN(z)[8+i,s] - LN(h)[j,s]|         (control: must be diagonal, argmin_j = 8+i)
D_null     = same as D_ph but h from a random other clip B (same scenario / opposite direction), S = S_B.

Token layout (verified from the repo, see README/notes in the md output):
  idx = tubelet*256 + row*16 + col   (PatchEmbed3D flatten(2): T-major then H then W;
  src/models/utils/patch_embed.py:51, analysis/intphys2/surprise.py:185-197, and the reshape(8,256,-1) /
  `r*16+c` convention in z_research/scripts/analysis/v11_token_object_test.py:25,34-35).
Position → cell: px on the 288 render, cell = int(px // 18) (v11_token_object_test.py:25; 288/16 = 18 px per cell,
  identical to floor(px*256/288/16)). Tubelet position = mean of samples 2t, 2t+1 (t32.reshape(16,2,2).mean(1), line 37).

  python z_research/scripts/analysis/ctxenc_time_alignment.py [--per-scen 60] [--v11-n 224] [--skip-v11] [--out DIR]
"""
from __future__ import annotations
import argparse, csv, json, re, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT_DEFAULT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/context_encoder_analysis/exp_results/time_alignment")
ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
CACHE = Path("/local_datasets/world/world_analysis/cache")
V2_INDEX = ROOT / "data_csv/rollout_v2/index.csv"
V2_CACHE = CACHE / "rollout_v2_vith"; V2_CTX32 = CACHE / "rollout_v2_ctx32_vith"
V11_INDEX = ROOT / "data_csv/intphysgen_v11_full/index_probe.csv"
V11_META = Path("/local_datasets/world/world_analysis/IntPhysGen_v11/metadata.csv")
V11_CACHE = CACHE / "v11_full_vith"; V11_CTX32 = CACHE / "v11_vanish_all_ctx32_vith"
SCEN = ["flat_v", "flat_a", "ramp_a", "arc", "fall", "ledge", "wall"]
V11_CONDS = [("flat k=0", "moving_visible_flat", "0"), ("ramp k=0", "moving_visible", "0")]
S_TOK, T_FUT, T_ALL, D = 256, 8, 16, 1280
CELL_PX = 18.0           # 288 render / 16 cells
MIN_DISP_CELLS = 2.0     # total true future displacement (tubelet 8 -> 15)


def arr(s):
    return np.array([float(v) for v in re.split(r"[;,\s|]+", str(s).strip()) if v], float)


def ln(x, eps=1e-5):
    """affine-free LayerNorm over last dim in fp32 == torch.nn.functional.layer_norm(x, (D,))"""
    x = np.asarray(x, np.float32); mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def n3(x, y):
    """3x3 token window around (x, y) px on the 288 render — verbatim v11_token_object_test.py:25."""
    cx, cy = int(x // CELL_PX), int(y // CELL_PX)
    return [r * 16 + c for r in range(cy - 1, cy + 2) for c in range(cx - 1, cx + 2) if 0 <= r < 16 and 0 <= c < 16]


def token_set(tub, inframe):
    """union over in-frame tubelets t = 0..15 of the 3x3 window at the true position."""
    S = set()
    for t in range(T_ALL):
        if inframe[t]:
            S.update(n3(*tub[t]))
    return np.array(sorted(S), int)


def dist_matrix(a, b, S):
    """a (Ta, 256, D), b (Tb, 256, D) -> (Ta, Tb) mean over S and D of |a_i - b_j|."""
    a = a[:, S]; b = b[:, S]                                        # (Ta,|S|,D), (Tb,|S|,D)
    return np.abs(a[:, None] - b[None]).mean(axis=(2, 3))


def refine(row):
    """argmin with parabolic interpolation when interior."""
    j = int(row.argmin())
    if 0 < j < len(row) - 1:
        a, b, c = row[j - 1], row[j], row[j + 1]; den = a - 2 * b + c
        if den > 0:
            j = j + 0.5 * (a - c) / den
    return float(j)


def contrast(row):
    m = float(np.median(row)); return (m - float(row.min())) / m if m > 0 else 0.0


def analyse_clip(p, hl, zl, hl_null, S, S_null):
    D_ph = dist_matrix(p, hl, S); D_null = dist_matrix(p, hl_null, S_null)
    D_zh = dist_matrix(zl[T_FUT:], hl, S) if zl is not None else None
    D_ph_all = dist_matrix(p, hl, np.arange(S_TOK))
    D_ph_bg = dist_matrix(p, hl, np.setdiff1d(np.arange(S_TOK), S))     # background-only tokens (never near the trajectory)
    return D_ph, D_zh, D_null, D_ph_all, D_ph_bg


def summarise(clips, gate_pct=95.0):
    """clips: list of dict(D_ph, D_zh, D_null, D_ph_all, disp_per_tub, n_S). Returns per-scenario summary dict."""
    n = len(clips)
    if n == 0:
        return {"n": 0}
    c_null = np.array([[contrast(c["D_null"][i]) for i in range(T_FUT)] for c in clips])   # (n, 8)
    gate = float(np.percentile(c_null.ravel(), gate_pct))
    gate_per_slot = [float(np.percentile(c_null[:, i], gate_pct)) for i in range(T_FUT)]
    jstar = np.array([[refine(c["D_ph"][i]) for i in range(T_FUT)] for c in clips])        # (n, 8)
    argm = np.array([[int(c["D_ph"][i].argmin()) for i in range(T_FUT)] for c in clips])
    cval = np.array([[contrast(c["D_ph"][i]) for i in range(T_FUT)] for c in clips])
    match = cval > gate
    jall = np.array([[refine(c["D_ph_all"][i]) for i in range(T_FUT)] for c in clips])
    call = np.array([[contrast(c["D_ph_all"][i]) for i in range(T_FUT)] for c in clips])
    jbg = np.array([[refine(c["D_ph_bg"][i]) for i in range(T_FUT)] for c in clips])
    cbg = np.array([[contrast(c["D_ph_bg"][i]) for i in range(T_FUT)] for c in clips])
    match_slot = cval > np.array(gate_per_slot)[None]
    target = np.arange(T_FUT) + T_FUT
    mean_j = [float(jstar[match[:, i], i].mean()) if match[:, i].any() else None for i in range(T_FUT)]
    sd_j = [float(jstar[match[:, i], i].std()) if match[:, i].sum() > 1 else None for i in range(T_FUT)]
    frac = [float(match[:, i].mean()) for i in range(T_FUT)]
    ok = [i for i in range(T_FUT) if mean_j[i] is not None]
    slope = float(np.polyfit(target[ok], [mean_j[i] for i in ok], 1)[0]) if len(ok) >= 2 else None
    # clip-level slope over that clip's matched slots (>= 3), then mean / sd
    cs = []
    for k in range(n):
        m = match[k]
        if m.sum() >= 3:
            cs.append(np.polyfit(target[m], jstar[k, m], 1)[0])
    out = {
        "n": n, "gate": gate, "gate_per_slot": gate_per_slot,
        "n_S_mean": float(np.mean([c["n_S"] for c in clips])),
        "disp_cells_per_tubelet_mean": float(np.mean([c["disp_per_tub"] for c in clips])),
        "disp_cells_total_future_mean": float(np.mean([c["disp_total"] for c in clips])),
        "p": {
            "mean_jstar_matched": mean_j, "sd_jstar_matched": sd_j, "match_frac": frac,
            "slope_mean_jstar_vs_8pi": slope, "n_slots_used_for_slope": len(ok),
            "clip_slope_mean": float(np.mean(cs)) if cs else None, "clip_slope_sd": float(np.std(cs)) if cs else None, "clip_slope_n": len(cs),
            "mean_jstar_all_clips": [float(v) for v in jstar.mean(0)], "sd_jstar_all_clips": [float(v) for v in jstar.std(0)],
            "mean_offset_all_clips": [float(v) for v in (jstar - target).mean(0)],
            "frac_argmin_within1_all_clips": [float(v) for v in (np.abs(argm - target) <= 1).mean(0)],
            "frac_argmin_in_future_all_clips": [float(v) for v in (argm >= T_FUT).mean(0)],
            "match_frac_per_slot_gate": [float(v) for v in match_slot.mean(0)],
            "min_D_ph_row": [float(v) for v in np.mean([c["D_ph"] for c in clips], 0).min(1)],
            "contrast_mean": [float(v) for v in cval.mean(0)], "contrast_null_mean": [float(v) for v in c_null.mean(0)],
            "mean_D_ph_row": [float(v) for v in np.mean([c["D_ph"] for c in clips], 0).mean(1)],
        },
        "p_all_tokens": {
            "mean_jstar": [float(v) for v in jall.mean(0)], "contrast_mean": [float(v) for v in call.mean(0)],
            "frac_argmin_within1": [float(v) for v in (np.abs(np.round(jall) - target) <= 1).mean(0)],
        },
        "p_bg_tokens": {
            "mean_jstar": [float(v) for v in jbg.mean(0)], "contrast_mean": [float(v) for v in cbg.mean(0)],
            "frac_argmin_within1": [float(v) for v in (np.abs(np.round(jbg) - target) <= 1).mean(0)],
            "match_frac_vs_gate": [float(v) for v in (cbg > gate).mean(0)],
            "mean_D_ph_bg": np.mean([c["D_ph_bg"] for c in clips], 0).round(4).tolist(),
        },
        "mean_D_ph": np.mean([c["D_ph"] for c in clips], 0).round(4).tolist(),
        "mean_D_null": np.mean([c["D_null"] for c in clips], 0).round(4).tolist(),
    }
    if clips[0]["D_zh"] is not None:
        jz = np.array([[refine(c["D_zh"][i]) for i in range(T_FUT)] for c in clips]); az = np.array([[int(c["D_zh"][i].argmin()) for i in range(T_FUT)] for c in clips])
        cz = np.array([[contrast(c["D_zh"][i]) for i in range(T_FUT)] for c in clips])
        out["zh"] = {"mean_jstar": [float(v) for v in jz.mean(0)], "sd_jstar": [float(v) for v in jz.std(0)],
                     "slope_mean_jstar_vs_8pi": float(np.polyfit(target, jz.mean(0), 1)[0]),
                     "frac_argmin_exact": [float(v) for v in (az == target).mean(0)], "contrast_mean": [float(v) for v in cz.mean(0)],
                     "mean_D_zh": np.mean([c["D_zh"] for c in clips], 0).round(4).tolist()}
    return out


def run_v2(per_scen, rng):
    idx = [r for r in csv.DictReader(V2_INDEX.open()) if r["plausible"] == "1"]
    vids = json.loads((V2_CACHE / "meta.json").read_text())["video_ids"]; vids2 = json.loads((V2_CTX32 / "meta.json").read_text())["video_ids"]
    assert vids == vids2, "rollout_v2_vith and rollout_v2_ctx32_vith meta video_ids differ"
    row = {v: i for i, v in enumerate(vids)}
    P = np.load(V2_CACHE / "predictor.npy", mmap_mode="r"); H = np.load(V2_CACHE / "target.npy", mmap_mode="r"); Z = np.load(V2_CTX32 / "isolated_ctx_0_32.npy", mmap_mode="r")
    assert P.shape[1:] == (T_FUT * S_TOK, D) and H.shape[1:] == (T_ALL * S_TOK, D) and Z.shape[1:] == (T_ALL * S_TOK, D)
    geo = {}
    for r in idx:
        assert r["resolution"] == "288"
        t32 = np.stack([arr(r["px_x_by_sample"]), arr(r["px_y_by_sample"])], -1); assert t32.shape == (32, 2)
        tub = t32.reshape(16, 2, 2).mean(1); inf = (arr(r["in_frame_by_sample"]) > 0).reshape(16, 2).all(1)
        geo[r["video_id"]] = (tub, inf)
    res, counts = {}, {}
    for sc in SCEN:
        pool = [r["video_id"] for r in idx if r["scenario"] == sc]
        qual = [v for v in pool if np.linalg.norm(geo[v][0][15] - geo[v][0][8]) / CELL_PX >= MIN_DISP_CELLS and geo[v][1][T_FUT:].any()]
        sel = list(rng.choice(qual, min(per_scen, len(qual)), replace=False)) if qual else []
        counts[sc] = {"plausible": len(pool), "qualified_disp_ge_2cells": len(qual), "used": len(sel)}
        clips = []; t0 = time.time()
        for v in sel:
            tub, inf = geo[v]; S = token_set(tub, inf)
            others = [u for u in pool if u != v]; u = others[rng.integers(len(others))]; tub_u, inf_u = geo[u]; S_u = token_set(tub_u, inf_u)
            p = np.asarray(P[row[v]], np.float32).reshape(T_FUT, S_TOK, D)
            hl = ln(np.asarray(H[row[v]], np.float32)).reshape(T_ALL, S_TOK, D)
            zl = ln(np.asarray(Z[row[v]], np.float32)).reshape(T_ALL, S_TOK, D)
            hl_u = ln(np.asarray(H[row[u]], np.float32)).reshape(T_ALL, S_TOK, D)
            D_ph, D_zh, D_null, D_all, D_bg = analyse_clip(p, hl, zl, hl_u, S, S_u)
            steps = [np.linalg.norm(tub[t + 1] - tub[t]) / CELL_PX for t in range(T_FUT, T_ALL - 1) if inf[t] and inf[t + 1]]
            clips.append(dict(D_ph=D_ph, D_zh=D_zh, D_null=D_null, D_ph_all=D_all, D_ph_bg=D_bg, n_S=len(S), disp_per_tub=float(np.mean(steps)) if steps else 0.0,
                              disp_total=float(np.linalg.norm(tub[15] - tub[8]) / CELL_PX)))
        res[sc] = summarise(clips); res[sc]["counts"] = counts[sc]
        print(f"[v2] {sc:<7} used {len(sel):3d}/{len(qual):3d} qualified of {len(pool)} plausible  ({time.time()-t0:.0f}s)", flush=True)
        _print_scen(sc, res[sc])
    return res


def run_v11(n_per_cond, rng):
    idx = list(csv.DictReader(V11_INDEX.open())); meta = {r["name"]: r for r in csv.DictReader(V11_META.open())}
    row = {v: i for i, v in enumerate(json.loads((V11_CACHE / "meta.json").read_text())["video_ids"])}
    zrow = {v: i for i, v in enumerate(json.loads((V11_CTX32 / "meta.json").read_text())["video_ids"])}
    P = np.load(V11_CACHE / "predictor.npy", mmap_mode="r"); H = np.load(V11_CACHE / "target.npy", mmap_mode="r"); Z = np.load(V11_CTX32 / "isolated_ctx_0_32.npy", mmap_mode="r")
    res = {}
    for name, cond, K in V11_CONDS:
        pos = {r["block_id"]: r for r in idx if r["condition"] == cond and r["sym_k"] == K and r["violation_type"] == "vanish" and r["variant"] == "pos_a"}
        blocks = sorted(pos)[:n_per_cond]                                   # same selection rule as v11_token_object_test.py:30-31
        geo = {}
        for b in blocks:
            m = meta[pos[b]["video_id"]]; assert m["resolution"] == "288"
            t32 = np.stack([arr(m["object_px_x_by_sample"]), arr(m["object_px_y_by_sample"])], -1); tub = t32.reshape(16, 2, 2).mean(1)
            inf = np.array([(0 <= tub[t, 0] < 288) and (0 <= tub[t, 1] < 288) for t in range(T_ALL)])
            geo[b] = (tub, inf, m["travel_direction"])
        qual = [b for b in blocks if np.linalg.norm(geo[b][0][15] - geo[b][0][8]) / CELL_PX >= MIN_DISP_CELLS and geo[b][1][T_FUT:].any()]
        have_z = all(pos[b]["video_id"] in zrow for b in qual)
        clips = []; t0 = time.time(); n_null_fallback = 0
        for b in qual:
            v = pos[b]["video_id"]; tub, inf, d = geo[b]; S = token_set(tub, inf)
            opp = [c for c in qual if c != b and geo[c][2] != d]
            if not opp:
                opp = [c for c in qual if c != b]; n_null_fallback += 1
            u = opp[rng.integers(len(opp))]; tub_u, inf_u, _ = geo[u]; S_u = token_set(tub_u, inf_u)
            p = np.asarray(P[row[v]], np.float32).reshape(T_FUT, S_TOK, D)
            hl = ln(np.asarray(H[row[v]], np.float32)).reshape(T_ALL, S_TOK, D)
            zl = ln(np.asarray(Z[zrow[v]], np.float32)).reshape(T_ALL, S_TOK, D) if have_z else None
            hl_u = ln(np.asarray(H[row[pos[u]["video_id"]]], np.float32)).reshape(T_ALL, S_TOK, D)
            D_ph, D_zh, D_null, D_all, D_bg = analyse_clip(p, hl, zl, hl_u, S, S_u)
            steps = [np.linalg.norm(tub[t + 1] - tub[t]) / CELL_PX for t in range(T_FUT, T_ALL - 1) if inf[t] and inf[t + 1]]
            clips.append(dict(D_ph=D_ph, D_zh=D_zh, D_null=D_null, D_ph_all=D_all, D_ph_bg=D_bg, n_S=len(S), disp_per_tub=float(np.mean(steps)) if steps else 0.0,
                              disp_total=float(np.linalg.norm(tub[15] - tub[8]) / CELL_PX)))
        res[name] = summarise(clips); res[name]["counts"] = {"blocks_selected": len(blocks), "qualified_disp_ge_2cells": len(qual), "used": len(clips),
                                                             "control_z_available": have_z, "null_no_opposite_direction_fallback": n_null_fallback}
        print(f"[v11] {name:<9} used {len(clips):3d}/{len(blocks)} blocks, control z {'yes' if have_z else 'NO'} ({time.time()-t0:.0f}s)", flush=True)
        _print_scen(name, res[name])
    return res


def _fmt(vs, f="{:5.2f}"):
    return " ".join("  -  " if v is None else f.format(v) for v in vs)


def _print_scen(name, r):
    if r.get("n", 0) == 0:
        print("   (no clips)"); return
    print(f"   gate {r['gate']:.4f}  |S| {r['n_S_mean']:.0f}  disp/tubelet {r['disp_cells_per_tubelet_mean']:.2f} cells  total {r['disp_cells_total_future_mean']:.2f}")
    if "zh" in r:
        print("   zh  mean j*   ", _fmt(r["zh"]["mean_jstar"]), f"  slope {r['zh']['slope_mean_jstar_vs_8pi']:.3f}  exact {np.mean(r['zh']['frac_argmin_exact']):.2f}")
    print("   p   match frac", _fmt(r["p"]["match_frac"]))
    print("   p   mean j*(m)", _fmt(r["p"]["mean_jstar_matched"]), f"  slope {r['p']['slope_mean_jstar_vs_8pi'] if r['p']['slope_mean_jstar_vs_8pi'] is None else round(r['p']['slope_mean_jstar_vs_8pi'], 3)}")
    print("   p   mean j*(all)", _fmt(r["p"]["mean_jstar_all_clips"]), "  within±1", _fmt(r["p"]["frac_argmin_within1_all_clips"]))
    print("   p   contrast  ", _fmt(r["p"]["contrast_mean"], "{:5.3f}"), " null", _fmt(r["p"]["contrast_null_mean"], "{:5.3f}"))
    print("   bg  mean j*   ", _fmt(r["p_bg_tokens"]["mean_jstar"]), "  within±1", _fmt(r["p_bg_tokens"]["frac_argmin_within1"]), "  contrast", _fmt(r["p_bg_tokens"]["contrast_mean"], "{:5.3f}"))


def write_md(out, path):
    L = ["# H1 pilot — time-alignment matrix on cached p / h / z (cache-only, 2026-09-15)", "",
         "D_ph[i][j] = mean over S (3x3 object tokens, union over in-frame tubelets) of |p_i − LN(h_j)|. j*(i) = parabola-refined argmin_j; "
         "c(i) = (median_j − min_j)/median_j; match if c > gate (95th pct of the same contrast for a random other clip's h, same scenario). "
         "Control D_zh uses LN(z)[8+i] (isolated_ctx over all 32 frames) in place of p_i and must sit on the diagonal.", "",
         f"Token layout: {out['token_layout_verified_from']}", ""]
    for arm, res in (("RollOut_v2 (plausible, total future displacement ≥ 2 cells, ≤ 60/scenario, seed 0)", out["v2"]), ("IntPhysGen v11 k=0 vanish pos_a", out.get("v11", {}))):
        if not res or "skipped" in res:
            L += [f"## {arm}", "", f"skipped: {res.get('skipped') if res else '-'}", ""]; continue
        L += [f"## {arm}", "", "| scenario | n (qualified / pool) | disp cells/tubelet | gate | zh mean j* s0..s7 | zh slope | p match frac s0..s7 | p mean j* (matched) s0..s7 | p slope | p mean offset (all clips) |", "|---|---|---|---|---|---|---|---|---|---|"]
        for sc, r in res.items():
            if r.get("n", 0) == 0:
                L.append(f"| {sc} | 0 | | | | | | | | |"); continue
            c = r["counts"]; pool = c.get("plausible", c.get("blocks_selected"))
            zh = f"{_fmt(r['zh']['mean_jstar'])} | {r['zh']['slope_mean_jstar_vs_8pi']:.3f}" if "zh" in r else "n/a | n/a"
            sl = r["p"]["slope_mean_jstar_vs_8pi"]; sl = "-" if sl is None else f"{sl:.3f} ({r['p']['n_slots_used_for_slope']} slots)"
            L.append(f"| {sc} | {r['n']} ({c['qualified_disp_ge_2cells']} / {pool}) | {r['disp_cells_per_tubelet_mean']:.2f} | {r['gate']:.4f} | {zh} | {_fmt(r['p']['match_frac'])} | {_fmt(r['p']['mean_jstar_matched'])} | {sl} | {_fmt(r['p']['mean_offset_all_clips'])} |")
        L.append("")
        L += ["Mean D_ph rows (i = 0..7) × j = 0..15, per scenario (lower = closer):", ""]
        for sc, r in res.items():
            if r.get("n", 0) == 0:
                continue
            L += [f"### {sc}  (n={r['n']}, |S|≈{r['n_S_mean']:.0f})", "", "| i \\ j | " + " | ".join(str(j) for j in range(16)) + " |", "|---|" + "---|" * 16]
            for i, rowv in enumerate(r["mean_D_ph"]):
                mn = min(rowv); L.append(f"| p{i} (true {8+i}) | " + " | ".join((f"**{v:.3f}**" if v == mn else f"{v:.3f}") for v in rowv) + " |")
            if "zh" in r:
                for i, rowv in enumerate(r["zh"]["mean_D_zh"]):
                    mn = min(rowv); L.append(f"| z{8+i} | " + " | ".join((f"**{v:.3f}**" if v == mn else f"{v:.3f}") for v in rowv) + " |")
            L.append("")
    L += ["## Notes / caveats", ""] + [f"- {c}" for c in out["caveats"]]
    path.write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--per-scen", type=int, default=60); ap.add_argument("--v11-n", type=int, default=224); ap.add_argument("--skip-v11", action="store_true"); ap.add_argument("--out", default=str(OUT_DEFAULT)); a = ap.parse_args()
    OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    out = {"token_layout_verified_from": "idx = tubelet*256 + row*16 + col — src/models/utils/patch_embed.py:51 (Conv3d → flatten(2) → transpose: T-major, then H, then W); "
                                         "analysis/intphys2/surprise.py:179-198 (_context_target_indices: first ctx_tub*256 = context, rest = target, tubelets contiguous); "
                                         "evals/analysis_vlm/occlusion_identity/forward.py:3-7 docstring; mirrored the reshape(8,256,-1) + n3 (r*16+c, cell = px//18) convention of "
                                         "z_research/scripts/analysis/v11_token_object_test.py:25,34-37 and mass3 in v11_readout_attn_diag.py:26-28.",
           "caveats": []}
    t0 = time.time(); out["v2"] = run_v2(a.per_scen, rng); out["v2_seconds"] = time.time() - t0
    if a.skip_v11:
        out["v11"] = {"skipped": "--skip-v11"}
    else:
        try:
            t0 = time.time(); out["v11"] = run_v11(a.v11_n, np.random.default_rng(0)); out["v11_seconds"] = time.time() - t0
        except Exception as e:                                             # noqa: BLE001
            out["v11"] = {"skipped": f"{type(e).__name__}: {e}"}; print("v11 arm failed:", e)
    out["caveats"] += [
        "Cached target.npy / isolated_ctx_0_32.npy rows are ALREADY affine-free-LayerNormed at extraction (evals/analysis_vlm/occlusion_identity/forward.py:137-171 `_ln`); "
        "measured row mean ≈ 4e-5, row std ≈ 1.0000 on rollout_v2_vith. Re-applying LN here changes |LN(h)−h| by ~9e-5 (fp16 rounding), so the scorer's |p − LN(h)| is reproduced either way.",
        "p_bg_tokens = same matrix on the complement of S (tokens never within 3x3 of the trajectory): if it is as diagonal as S, the alignment is a per-tubelet time signature of the tokens, not the object. "
        "S is the union of 3x3 windows over ALL in-frame tubelets 0..15 (task spec), so it spans the whole trajectory; the object occupies only a sub-part of S at any one tubelet.",
        "Gate = 95th percentile of the null contrast pooled over clips × slots (per scenario). Null uses a different clip's h AND that clip's token set S_B.",
        "Parabolic refinement is applied only when argmin is interior and the parabola is convex; edge argmins (j=0 or 15) stay integer.",
        "wall (plausible) is stationary in the future → no clip passes the ≥ 2-cell displacement filter (reported in counts).",
        "Slopes are least squares of mean j*(i) over slots with ≥ 1 matched clip; when few clips match, that mean is noisy — read together with match_frac and the mean_D_ph tables.",
    ]
    (OUT / "time_alignment.json").write_text(json.dumps(out, indent=1)); write_md(out, OUT / "time_alignment.md"); print(f"→ {OUT}/time_alignment.{{json,md}}")
    print(f"→ {HERE}/h1_pilot.{{json,md}}")


if __name__ == "__main__":
    main()
