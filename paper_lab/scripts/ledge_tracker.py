#!/usr/bin/env python3
"""VOBench 중력 선반에 추적 기준선 (진행자 2, 2026-10-22, PLAN §13 B 첫 실험). 학습 없음.

판: 선반 묶음 144 (미래 둘: 참 = 받침 끝에서 떨어짐 · 떠 감). 하네스와 같은 '물체 자리 고른 미래' (두 미래 물체 3x3 칸 합집합에서 오차 최소).
렌더: 추적 기준선 (라벨 추적 · 관측 추적 × 등속 · 2 차, m2_obsonly 와 같은 붙이기) — 여기서 새로 만든다.
      복사 · 문맥 평균 · 오라클 · 기준 블록 AR — 하네스 산출 (vast70 의 gravity 오차 지도) 을 그대로 읽는다.
먼 칸 판: 참 미래의 물체 중심이 마지막으로 보인 문맥 자리에서 48 px (512 라벨) 넘게 떨어진 미래 블록만.
  CUDA_VISIBLE_DEVICES=<g> python paper_lab/scripts/ledge_tracker.py render
  python paper_lab/scripts/ledge_tracker.py score
"""
import argparse, csv, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p_ in (str(REPO), str(REPO / "z_research/scripts/analysis"), str(REPO / "paper_lab/scripts"), str(REPO / "paper_lab/eval")):
    if p_ not in sys.path:
        sys.path.insert(0, p_)
OUT = REPO / "paper_lab/_scratch/ledge"
G, C, K, NB, D = 16, 8, 8, 16, 1280
FAR = 48.0


def _ds():
    import yaml
    import design_headroom as H
    from evals.world_model_analysis.data import WMADataset
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("ir", REPO / "z_research/scripts/analysis/rollout2_identity_readout.py")
    ir = module_from_spec(spec); spec.loader.exec_module(ir)
    cfg = yaml.safe_load(open(H.SETS["gravity"]["cfg"]))
    ds = WMADataset(dict(data=dict(cfg["data"], index_csv="index.csv"), model=dict(ir.rp.MODEL), features={"cache_dir": "/tmp"}, surprise={}))
    rows = {r["video_id"]: r for r in csv.DictReader(open(H.SETS["gravity"]["index"]))}
    return H, ds, rows


def blocks(rows):
    blk = defaultdict(dict)
    for v, r in rows.items():
        if r.get("law") == "ledge":
            blk[r["source_block"]][r["variant"]] = v
    return sorted((sb, a) for sb, a in blk.items() if "pos" in a and "imp_float" in a)


def render():
    import torch
    import passline as PL
    import m2_obsonly as MO
    from app.vjepa_frozen.ar import encode_blocks
    torch.set_num_threads(4)
    H, ds, rows = _ds()
    dev = torch.device("cuda", 0)
    _, bundle = PL._load_model("z_training/vast/ar_state/latest.pt", dev)
    idx = {r.video_id: i for i, r in enumerate(ds.records)}
    rec = defaultdict(list); diag = []; t0 = time.time()
    for j, (sb, arms) in enumerate(blocks(rows)):
        order = ("pos", "imp_float")
        x = torch.stack([ds.clip(idx[arms[a]]) for a in order]).to(dev, dtype=bundle.dtype)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hb = encode_blocks(bundle.target_encoder, x, 2).float().view(2, NB, G, G, D)
        P, hid = H.track_gravity(rows[arms["pos"]])                                   # 문맥 0..15 는 두 미래가 같다
        Po, ho = MO.track_pixels(x[0])
        Po_full = np.vstack([Po, np.full((16, 2), np.nan)]); ho_full = np.concatenate([ho, np.ones(16)])
        vis = [s for s in range(16) if hid[s] < 0.5 and ho[s] < 0.5]
        if vis:
            diag.append(float(np.nanmedian([np.linalg.norm(P[s] - Po[s]) / 2.0 for s in vis])))
        r_lab = MO.render_moves(hb[0], P, hid, H)
        r_obs = MO.render_moves(hb[0], Po_full, ho_full, H)
        rend = {"lab_cv": r_lab["move_cv"], "lab_ca": r_lab["move_ca"], "obs_cv": r_obs["move_cv"], "obs_ca": r_obs["move_ca"]}
        for ai, a in enumerate(order):
            fut = hb[ai, C:]
            for m, R in rend.items():
                rec[f"E_{m}"].append((R.float() - fut).abs().mean(-1).half().cpu().numpy())
            rec["video_id"].append(arms[a])
        if j % 20 == 0:
            print(f"[render] {j + 1} blocks ({time.time() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "render.npz", video_id=np.array(rec.pop("video_id")), **{k: np.stack(v) for k, v in rec.items()})
    json.dump({"obs_track_median_err_px256": float(np.median(diag)) if diag else None, "n": len(diag)}, open(OUT / "track_diag.json", "w"))
    print(f"[render] done ({time.time() - t0:.0f}s)", flush=True)


def score():
    import design_headroom as H
    import passline as PL
    rows = {r["video_id"]: r for r in csv.DictReader(open(H.SETS["gravity"]["index"]))}
    z = np.load(OUT / "render.npz", allow_pickle=True)
    tr = ({v: i for i, v in enumerate(z["video_id"])}, {k: z[k] for k in z.files if k.startswith("E_")})
    Zg = PL._load("vast70", "gravity")
    hv = ({v: i for i, v in enumerate(Zg["video_id"])}, {k: Zg[k] for k in Zg if k.startswith("E_")})
    res = {"rows": {}, "track_diag": json.load(open(OUT / "track_diag.json"))}
    BL = blocks(rows)
    for far in (False, True):
        tag = "far48" if far else "all"
        out = {}
        for lab, src, rnd in (("copy", hv, "copy"), ("ctx-mean8", hv, "mean8"), ("oracle", hv, "oracle"), ("base block AR", hv, "model"),
                              ("label track + const-vel", tr, "lab_cv"), ("label track + quadratic", tr, "lab_ca"),
                              ("observed track + const-vel", tr, "obs_cv"), ("observed track + quadratic", tr, "obs_ca")):
            hits, efl, n = [], [], 0
            for sb, arms in BL:
                r = rows[arms["pos"]]
                labj = json.load(open(H.FR / "gravity" / Path(r["file_name"]).parent / "label.json"))["labels"]
                M = np.zeros((K, G, G), bool)
                for a in arms:
                    Pf, _ = PL.future_track(labj.get(H.GRAV_ARM.get(a, "")))
                    if Pf is not None:
                        M |= PL.cell_mask(Pf[16:].reshape(8, 2, 2).mean(1))
                if far:
                    Pp, hp = PL.future_track(labj.get("P"))
                    vis = [s for s in range(16) if hp[s] < 0.5]
                    if not vis:
                        continue
                    keep = np.linalg.norm(Pp[16:].reshape(8, 2, 2).mean(1) - Pp[vis[-1]], axis=1) > FAR
                    M[~keep] = False
                if not M.any():
                    continue
                idx, E = src
                e = {a: float(E["E_" + rnd][idx[v]].astype(np.float32)[M].mean()) for a, v in arms.items()}
                hits.append(1.0 if e["pos"] < e["imp_float"] else (0.5 if e["pos"] == e["imp_float"] else 0.0))
                efl.append(e["imp_float"]); n += 1
            hits = np.array(hits)
            bs = np.random.default_rng(0).choice(hits, size=(2000, len(hits))).mean(1)
            out[lab] = dict(acc=100 * float(hits.mean()), ci=[100 * float(np.percentile(bs, 2.5)), 100 * float(np.percentile(bs, 97.5))],
                            n=n, err_to_float=float(np.mean(efl)))
            print(f"  {tag:5s} {lab:30s} acc {out[lab]['acc']:5.1f} [{out[lab]['ci'][0]:5.1f}, {out[lab]['ci'][1]:5.1f}]  n {n}  err to float-future {out[lab]['err_to_float']:.4f}")
        res["rows"][tag] = out
    o, t = res["rows"]["all"]["oracle"]["acc"], res["rows"]["all"]["label track + quadratic"]["acc"]
    res["usable"] = bool(t < 0.5 * o)
    print(f"판 쓰기 규칙: 추적 (라벨 · 2 차) {t:.1f} vs 0.5 × 오라클 {0.5 * o:.1f} → {'추적기가 못 푸는 판' if res['usable'] else '추적기가 푼다'}")
    print("track diag:", res["track_diag"])
    json.dump(res, open(OUT / "score.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("phase", choices=["render", "score"])
    a = ap.parse_args()
    {"render": render, "score": score}[a.phase]()
