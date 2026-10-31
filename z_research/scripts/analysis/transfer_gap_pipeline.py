#!/usr/bin/env python3
"""transfer_gap Q3 · E31 — 눈에 안 보이는 입력 차이 (화소 반올림) 에 대한 PFT 의 반응은 어디서 오나. 2026-10-30, 실험 B [X2] R2.

왜: R1 에서 재양자화 (화소 차 ≤ 0.5/255) 만으로 쌍 판정이 PFT 쪽으로만 뒤집혔고 (IntPhys 1 skip2 14:1), VOBench 에서 PFT surprise 만
    +1.19 % 움직였다 (릴리즈 +0.10 %). 두 해석: H3-L (PFT 출력이 입력 화소 통계에 민감해졌다) vs H3-L0 (같은 절대 이동인데 PFT 오차가
    작아 상대 이동이 커 보인다 = 분모 착시). (a) 기존 산출물로 절대 · 상대 이동과 기준 오차 회귀를, (b) 문맥 · 목표 입력 교차 2×2 로 가른다.

(a) 정의 (CPU) — 영상 i, 모델 m, 판 orig / rq (재양자화).
  Δ_i = s_rq − s_orig (절대), r_i = Δ_i / s_orig (상대).  IntPhys 1: Garrido 창 점수 (시작점마다 C 최소 → 평균, 조합별),
  VOBench split v2 test: 고정 창 c16t32 표준 surprise (te 판 T0 vs Torig).  기준 오차 회귀: Δ_i = a + b·s_orig,i (모델별).
  분모 착시 판정: PFT 의 평균 Δ 가 릴리즈와 같고 (차 CI 가 0 포함) 상대 이동만 크면 H3-L0. 절대 Δ 자체가 PFT 에서 크면 H3-L 쪽.
(b) 정의 (GPU 1) — VOBench test 360 영상 (조건 층화). z ∈ {원본, T0} (문맥 encoder 입력 화소) × h ∈ {원본, T0} (목표 encoder 입력 화소).
  s(z_x, h_y) = mean | p(z_x) − LN(h_y)[미래] |.  문맥 쪽 몫 = s(z_T0, h_원) − s(z_원, h_원) · 목표 쪽 몫 = s(z_원, h_T0) − s(z_원, h_원)
  교호 = s(z_T0, h_T0) − s(z_T0, h_원) − s(z_원, h_T0) + s(z_원, h_원).  출력 민감도 = ‖p(z_T0) − p(z_원)‖₁ / ‖p(z_원) − h_원‖₁.
  판정: 문맥 쪽 몫이 PFT 에서 릴리즈의 3 배 이상 & 출력 민감도 2 배 이상 → H3-L. 목표 쪽 몫이 지배 → H3-L0.

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/transfer_gap_pipeline.py a                → results/q3_r2/E31a_shift.json
  $P z_research/scripts/analysis/transfer_gap_pipeline.py b --n 360        → results/q3_r2/E31b_cross.json   (GPU 1)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
sys.path.insert(0, str(REPO))
import numpy as np  # noqa: E402

OUTD = REPO / "z_analysis/transfer_gap/results/q3_r2"
TE = Path("/data2/local_datasets/world/world_analysis/cache/transfer_gap")
NF = REPO / "z_analysis/transfer_gap/results/q3/runs"
IP1 = REPO / "cvpr/results/eval/intphys1"
REL = str(REPO / "z_training/runs/release_vith/latest.pt")
PFT = str(REPO / "z_training/runs/vobench_release_postft_g/latest.pt")


def garrido_scores(pw):
    out = defaultdict(dict)
    for v, lst in pw.items():
        per = defaultdict(lambda: defaultdict(list))
        for combo, st, C, s in lst:
            per[combo][st].append(s)
        for combo, d in per.items():
            out[combo][v] = float(np.mean([min(x) for x in d.values()]))
    return out


def summarize(orig, rq, rng):
    vids = sorted(set(orig) & set(rq))
    o = np.array([orig[v] for v in vids]); r = np.array([rq[v] for v in vids])
    d = r - o
    b, a = np.polyfit(o, d, 1)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    return {"n": len(vids), "mean_s_orig": round(float(o.mean()), 5), "mean_abs_shift": round(float(d.mean()), 6),
            "mean_abs_shift_ci": [round(float(np.percentile(bs, q)), 6) for q in (2.5, 97.5)],
            "mean_rel_shift_pct": round(100 * float((d / o).mean()), 3), "frac_up": round(float((d > 0).mean()), 3),
            "mean_abs_|shift|": round(float(np.abs(d).mean()), 6), "regress_shift_on_s_orig": {"slope": round(float(b), 5), "intercept": round(float(a), 6),
            "corr": round(float(np.corrcoef(o, d)[0, 1]), 3)}}, dict(zip(vids, d))


def run_a():
    rng = np.random.default_rng(0)
    res = {"definition": __doc__.split("(a) 정의 (CPU) —")[1].split("(b) 정의")[0].strip(), "intphys1": {}, "vobench": {}}
    for w in (16, 32):
        for m, tag in (("release", "vith"), ("vob_pft", "vith_release_vobftg")):
            po = json.load(open(IP1 / f"intphys1_dev_{tag}_wgarrido_w{w}" / "per_window.json"))["windows"]
            pr = json.load(open(NF / f"NF_intphys1_requant_{tag}_w{w}" / "per_window.json"))["windows"]
            go, gr = garrido_scores(po), garrido_scores(pr)
            for combo in go:
                if combo in gr:
                    s, _ = summarize(go[combo], gr[combo], rng)
                    res["intphys1"].setdefault(combo, {})[m] = s
    for combo, d in res["intphys1"].items():
        if "release" in d and "vob_pft" in d:
            d["pft_over_release_abs_shift"] = round(d["vob_pft"]["mean_abs_shift"] / d["release"]["mean_abs_shift"], 2) if d["release"]["mean_abs_shift"] else None
    mo, mr = json.load(open(TE / "te_E12_Torig/meta.json")), json.load(open(TE / "te_E12_T0/meta.json"))
    lo, lr = np.load(TE / "te_E12_Torig/l1.npy").mean(2), np.load(TE / "te_E12_T0/l1.npy").mean(2)
    co, cr = np.load(TE / "te_E12_Torig/copy.npy").mean(1), np.load(TE / "te_E12_T0/copy.npy").mean(1)
    io = {v: i for i, v in enumerate(mo["video_id"])}
    ir = {v: i for i, v in enumerate(mr["video_id"])}
    vids = [v for v in mo["video_id"] if v in ir]
    shifts = {}
    for k, tag in enumerate(mo["pred_tags"]):
        assert mr["pred_tags"][k] == tag
        s, sh = summarize({v: lo[io[v], k] for v in vids}, {v: lr[ir[v], k] for v in vids}, rng)
        res["vobench"][tag] = s; shifts[tag] = sh
    s, _ = summarize({v: co[io[v]] for v in vids}, {v: cr[ir[v]] for v in vids}, rng)
    res["vobench"]["copy_z"] = s
    # 같은 영상의 절대 이동 차 (PFT − 릴리즈) 짝 bootstrap
    for name, D in (("vobench", shifts),):
        a = np.array([D["vobftg"][v] - D["release"][v] for v in vids])
        bs = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(2000)]
        res[name]["pft_minus_release_abs_shift"] = {"v": round(float(a.mean()), 6), "ci": [round(float(np.percentile(bs, q)), 6) for q in (2.5, 97.5)]}
    OUTD.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUTD / "E31a_shift.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({"ip1": {c: {m: (v[m]["mean_abs_shift"], v[m]["mean_rel_shift_pct"], v[m]["frac_up"]) for m in ("release", "vob_pft") if m in v} for c, v in res["intphys1"].items()},
                      "vob": {m: (v["mean_abs_shift"], v["mean_rel_shift_pct"], v["frac_up"], v["mean_s_orig"]) for m, v in res["vobench"].items() if isinstance(v, dict) and "mean_abs_shift" in v},
                      "vob_diff": res["vobench"]["pft_minus_release_abs_shift"]}, ensure_ascii=False, indent=0))


def run_b(n_pick: int, batch: int):
    import torch
    import torch.nn.functional as F
    import te_v11_score as te
    from transfer_gap_te import resolve
    dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
    cfg_o, _ = resolve("vobench_v1_split2_test", "vith", [], None, 0)
    cfg_r, _ = resolve("vobench_v1_split2_test", "vith", ["data.root=" + str(REPO / "data_csv/transfer_gap_vob"), "data.index_csv=index_T0.csv",
                                                           "data.frames_root=" + str(TE / "frames_T0")], None, 0)
    Wo, Wr = te.WMADataset(cfg_o), te.WMADataset(cfg_r)
    ir = {r.video_id: i for i, r in enumerate(Wr.records)}
    # 조건 층화: (objset, violation, occlusion) 마다 고르게, 가능 영상만
    cells = defaultdict(list)
    for i, r in enumerate(Wo.records):
        if str(r.raw.get("plausible")) == "1" and r.video_id in ir:
            cells[(r.raw.get("objset"), r.raw.get("violation"), r.raw.get("occlusion"))].append(i)
    per = max(1, n_pick // len(cells))
    pick = []
    for k in sorted(cells):
        L = cells[k]
        pick += [L[int(x)] for x in np.linspace(0, len(L) - 1, min(per, len(L)))]
    sc = te.Scorer(cfg_o, dev, [("release", REL), ("pft", PFT)])
    D, S, nc, nf = sc.D, sc.S, sc.nc, sc.nf
    rows = []
    red = lambda t: t.abs().mean((-1, -2, -3))  # noqa: E731
    for b0 in range(0, len(pick), batch):
        idx = pick[b0:b0 + batch]
        xo = torch.stack([Wo.clip(i) for i in idx]); xr = torch.stack([Wr.clip(ir[Wo.records[i].video_id]) for i in idx])
        with torch.inference_mode():
            zo, ho, ci, ti, _ = sc.encode(xo)
            zr, hr, _, _, _ = sc.encode(xr)
            B = zo.size(0)
            hfo = ho.float().reshape(B, nc + nf, S, D)[:, nc:]; hfr = hr.float().reshape(B, nc + nf, S, D)[:, nc:]
            dx = (xr.float() - xo.float()).abs().amax().item()
            for k, tag in enumerate(sc.tags):
                po = sc.predict(sc.mods[k], zo, ci, ti).float().reshape(B, nf, S, D)
                pr = sc.predict(sc.mods[k], zr, ci, ti).float().reshape(B, nf, S, D)
                s_oo, s_ro, s_or, s_rr = red(po - hfo), red(pr - hfo), red(po - hfr), red(pr - hfr)
                sens = red(pr - po) / red(po - hfo)
                for j in range(B):
                    rows.append({"vid": Wo.records[idx[j]].video_id, "model": tag, "s_oo": float(s_oo[j]), "s_ro": float(s_ro[j]),
                                 "s_or": float(s_or[j]), "s_rr": float(s_rr[j]), "sens": float(sens[j]), "max_dx": dx,
                                 "occ": Wo.records[idx[j]].raw.get("occlusion"), "objset": Wo.records[idx[j]].raw.get("objset")})
        print(f"{min(b0 + batch, len(pick))}/{len(pick)}", flush=True)
    rng = np.random.default_rng(0)
    res = {"definition": __doc__.split("(b) 정의 (GPU 1) —")[1].split("  P=")[0].strip(), "n_video": len(pick), "models": {}}
    by = defaultdict(list)
    for r in rows:
        by[r["model"]].append(r)
    for m, L in by.items():
        A = np.array([[r["s_oo"], r["s_ro"], r["s_or"], r["s_rr"], r["sens"]] for r in L])
        ctx = A[:, 1] - A[:, 0]; tgt = A[:, 2] - A[:, 0]; inter = A[:, 3] - A[:, 1] - A[:, 2] + A[:, 0]
        def ci(x):
            bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
            return [round(float(np.percentile(bs, q)), 7) for q in (2.5, 97.5)]
        res["models"][m] = {"mean_s_oo": round(float(A[:, 0].mean()), 5),
                            "ctx_part": round(float(ctx.mean()), 7), "ctx_part_ci": ci(ctx), "ctx_part_rel_pct": round(100 * float((ctx / A[:, 0]).mean()), 4),
                            "tgt_part": round(float(tgt.mean()), 7), "tgt_part_ci": ci(tgt), "tgt_part_rel_pct": round(100 * float((tgt / A[:, 0]).mean()), 4),
                            "interaction": round(float(inter.mean()), 7), "total_rr_minus_oo": round(float((A[:, 3] - A[:, 0]).mean()), 7),
                            "frac_ctx_up": round(float((ctx > 0).mean()), 3), "frac_tgt_up": round(float((tgt > 0).mean()), 3),
                            "output_sensitivity_median": round(float(np.median(A[:, 4])), 5), "output_sensitivity_mean": round(float(A[:, 4].mean()), 5)}
    r_, p_ = res["models"].get("release"), res["models"].get("pft")
    if r_ and p_:
        res["ratios_pft_over_release"] = {"ctx_part": round(p_["ctx_part"] / r_["ctx_part"], 2) if r_["ctx_part"] else None,
                                          "tgt_part": round(p_["tgt_part"] / r_["tgt_part"], 2) if r_["tgt_part"] else None,
                                          "output_sensitivity": round(p_["output_sensitivity_median"] / r_["output_sensitivity_median"], 2)}
    res["rows"] = rows
    json.dump(res, open(OUTD / "E31b_cross.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["a", "b"])
    ap.add_argument("--n", type=int, default=360)
    ap.add_argument("--batch", type=int, default=6)
    a = ap.parse_args()
    run_a() if a.part == "a" else run_b(a.n, a.batch)


if __name__ == "__main__":
    main()
