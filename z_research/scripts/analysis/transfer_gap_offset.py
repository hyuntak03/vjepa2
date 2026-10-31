#!/usr/bin/env python3
"""transfer_gap Q3 · E27 — PFT 의 '화면 공통 이동' 을 빼고 다시 채점한다 (재학습 없음). 2026-10-30, 실험 B [X2] R2.

왜: E16 에서 VOBench PFT 는 외부 벤치에서도 출력을 오차의 32–44 % 만큼 바꾸는데, 그중 약 40 % 가 튜블릿마다 256 칸이 같이 움직이는
    공통 이동이었다. 그 이동이 입력과 거의 무관하고 VOBench 목표 평균 쪽이라면 (H3-P) 빼면 외부 오차가 줄고 VOBench 효과는 남아야 한다.
정의 (고정 창 c16t32 · stride 3 (GRASP · InfLevel 은 5), matched, 표준 채점 = mean|p − LN(h)[미래]| 전체 토큰)
  c_{i,t} = (1/256) Σ_c (p_PFT − p_rel)_{i,t,c} ∈ R^1280          — 영상 i 의 화면 공통 이동 (튜블릿 t 마다)
  c̄_D     = 세트 D 의 가능 영상 평균.  c̄_VOBtrain = VOBench split v2 train 가능 영상 720 개 평균 (PFT 학습셋 쪽)
  g_{i,t} = (1/256) Σ_c (h − p_rel)_{i,t,c}                         — 릴리즈 출력의 평균 오차 (목표 평균 − 출력 평균)
  퍼짐    = ‖c_i − c̄_D‖ / ‖c̄_D‖  ((t, d) 를 펼친 벡터, 가능 영상 중앙값)
  세트 사이 cos(c̄_D, c̄_VOBtest) · 방향 cos(c̄_VOBtrain, ḡ_D)
  다시 채점 (같은 encoder · 같은 창):
    R0 = p_rel · V0 = p_PFT · V1 = p_PFT − c_i · V2 = p_PFT − c̄_VOBtrain · V3 = p_rel + c̄_VOBtrain · V4 = p_rel + c_i
  ε(V) = 가능 영상 평균 surprise(V) ÷ 평균 surprise(R0).  정확도 = matched 쌍 (동점 0.5).  VOBench 는 late 이동 사라짐 A · B 도.
  ⚠️ 고정 창이라 Garrido (sliding · 창 최고) 수치와 비교하지 않는다 (CLAUDE.md §1-3). GRASP · InfLevel 은 block 을 고르게 골라 약 1,200 clip.

단계
  --phase train : VOBench split v2 train 가능 영상 → c_train.npy (n,8,1280) · cbar.npy (8,1280)      (GPU 1, 약 3 분)
  --phase score --sets a,b : 세트마다 l1 (n,6,8) [R0,V0,V1,V2,V3,V4] · copy (n,8) · c · g (n,8,1280) fp16 · meta.json
  --phase report : → z_analysis/transfer_gap/results/q3_r2/E27_offset.json   (CPU)
배열: /data2/local_datasets/world/world_analysis/cache/transfer_gap/E27/

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/transfer_gap_offset.py --phase train
  $P z_research/scripts/analysis/transfer_gap_offset.py --phase score --sets vob_test,intphys1_dev,inflevel_continuity_s5
  $P z_research/scripts/analysis/transfer_gap_offset.py --phase score --sets v11sub,grasp_level2_s5
  $P z_research/scripts/analysis/transfer_gap_offset.py --phase report
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
sys.path.insert(0, str(REPO))
import numpy as np  # noqa: E402

ARR = Path("/data2/local_datasets/world/world_analysis/cache/transfer_gap/E27")
OUT = REPO / "z_analysis/transfer_gap/results/q3_r2/E27_offset.json"
REL = str(REPO / "z_training/runs/release_vith/latest.pt")
PFT = str(REPO / "z_training/runs/vobench_release_postft_g/latest.pt")
VOB = str(REPO / "data_csv/vobench_v1_split_v2")
SETS = {  # 이름 → (레지스트리 데이터셋, --set, limit, 분류)
    "vob_train": ("vobench_v1_split2_test", ["data.root=" + VOB, "data.index_csv=index_train.csv"], 0, "train"),
    "vob_test": ("vobench_v1_split2_test", [], 0, "in-dist"),
    "v11sub": ("v11_split_test", ["data.root=" + str(REPO / "data_csv/transfer_gap_v11sub"), "data.index_csv=index.csv"], 0, "near"),
    "intphys1_dev": ("intphys1_dev", [], 0, "external"),
    "grasp_level2_s5": ("grasp_level2", ["data.frames_stride=5"], 1200, "external"),
    "inflevel_continuity_s5": ("inflevel_continuity", ["data.frames_stride=5"], 1200, "external"),
}
_SC = None
SMOKE = 0          # --smoke N: 세트마다 약 N clip (배관 확인), 배열은 ARR/_smoke 아래
VNAMES = ["R0_rel", "V0_pft", "V1_pft-ci", "V2_pft-cbarVOBtrain", "V3_rel+cbarVOBtrain", "V4_rel+ci"]
EXTRA = ("objset", "violation", "occlusion", "k", "motion", "background", "condition", "variant", "violation_type", "sym_k")


def gpu_loop(name, cbar, only_pos=False, batch=8):
    import torch
    import torch.nn.functional as F
    import te_v11_score as te
    from transfer_gap_te import resolve
    ds_name, sets, limit, _ = SETS[name]
    if SMOKE:
        limit = SMOKE
    cfg, _ = resolve(ds_name, "vith", sets, None, limit)
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    W = te.WMADataset(cfg)
    rows = te.select_rows(W, limit)
    if only_pos:
        rows = [i for i in rows if str(W.records[i].raw.get("plausible")) == "1"]
    global _SC
    if _SC is None:                                    # ViT-H 로딩 (~2 분) 은 프로세스당 한 번 — 모델 · 창 규약은 세트 사이 같다
        _SC = te.Scorer(cfg, dev, [("release", REL), ("pft", PFT)])
    sc = _SC
    D, S, nc, nf = sc.D, sc.S, sc.nc, sc.nf
    n = len(rows)
    l1 = np.zeros((n, len(VNAMES), nf), np.float32)
    cp = np.zeros((n, nf), np.float32)
    C = np.zeros((n, nf, D), np.float16)
    G = np.zeros((n, nf, D), np.float16)
    cb = None if cbar is None else torch.as_tensor(cbar, device=dev, dtype=torch.float32)[None, :, None, :]
    loader = torch.utils.data.DataLoader(te.ClipDS(W, list(range(n)), rows), batch_size=batch, num_workers=6,
                                         collate_fn=te.collate, pin_memory=True)
    t0 = time.time()
    for it, (rr, clips) in enumerate(loader):
        with torch.inference_mode():
            z, h, ci, ti, _ = sc.encode(clips)
            B = z.size(0)
            hf = h.float().reshape(B, nc + nf, S, D)[:, nc:]
            with sc.autocast():
                zT = F.layer_norm(z.float(), (D,)).reshape(B, nc, S, D)[:, nc - 1:nc]
            zT = zT.float()
            pr = sc.predict(sc.mods[0], z, ci, ti).float().reshape(B, nf, S, D)
            pf = sc.predict(sc.mods[1], z, ci, ti).float().reshape(B, nf, S, D)
            c = (pf - pr).mean(2, keepdim=True)
            g = (hf - pr).mean(2)
            red = lambda t: t.abs().mean((-1, -2))  # noqa: E731
            vs = [pr, pf, pf - c]
            if cb is not None:
                vs += [pf - cb, pr + cb]
            else:
                vs += [pf * np.nan, pr * np.nan]
            vs.append(pr + c)
            out = torch.stack([red(v - hf) for v in vs], 1)
            ix = np.asarray(rr)
            l1[ix] = out.cpu().numpy()
            cp[ix] = red(zT - hf).cpu().numpy()
            C[ix] = c[:, :, 0].cpu().numpy().astype(np.float16)
            G[ix] = g.cpu().numpy().astype(np.float16)
        if it % 25 == 0:
            el = time.time() - t0
            print(f"[{name}] {min((it + 1) * batch, n)}/{n}  {el / 60:.1f} 분", flush=True)
    od = ARR / name
    od.mkdir(parents=True, exist_ok=True)
    np.save(od / "l1.npy", l1); np.save(od / "copy.npy", cp); np.save(od / "c.npy", C); np.save(od / "g.npy", G)
    meta = {"set": name, "dataset": ds_name, "sets": sets, "limit": limit, "n": n, "variants": VNAMES,
            "rel": REL, "pft": PFT, "cbar_used": cbar is not None}
    for k in te.META_COLS + EXTRA:
        meta[k] = [W.records[i].raw.get(k) for i in rows]
    meta["video_id"] = [W.records[i].video_id for i in rows]
    meta["block_id"] = [W.records[i].block_id for i in rows]
    meta["pair_id"] = [W.records[i].raw.get("pair_id") for i in rows]
    meta["plausible"] = [str(W.records[i].raw.get("plausible")) for i in rows]
    json.dump(meta, open(od / "meta.json", "w"), ensure_ascii=False)
    print(f"[{name}] done n={n} {time.time() - t0:.0f}s → {od}", flush=True)
    return C, meta


def phase_train():
    C, meta = gpu_loop("vob_train", None, only_pos=True)
    pos = np.array([p == "1" for p in meta["plausible"]])
    cbar = C[pos].astype(np.float64).mean(0)
    np.save(ARR / "cbar_vob_train.npy", cbar.astype(np.float32))
    print("cbar", cbar.shape, float(np.linalg.norm(cbar)), "n_pos", int(pos.sum()))


def matched(meta, s):
    grp = defaultdict(dict)
    for i, (b, p, pl) in enumerate(zip(meta["block_id"], meta["pair_id"], meta["plausible"])):
        grp[(b, p)]["pos" if pl == "1" else "imp"] = i
    out = []
    for (b, p), g in grp.items():
        if "pos" in g and "imp" in g:
            a, c = s[g["pos"]], s[g["imp"]]
            out.append((b, g["pos"], 1.0 if c > a else (0.5 if c == a else 0.0)))
    return out


def boot(vals, blocks, rng, nb=2000):
    vals, blocks = np.asarray(vals), np.asarray(blocks)
    if len(vals) == 0:
        return None
    u = np.unique(blocks)
    idx = {b: np.flatnonzero(blocks == b) for b in u}
    bs = [vals[np.concatenate([idx[u[k]] for k in rng.choice(len(u), len(u))])].mean() for _ in range(nb)]
    return [round(100 * float(np.percentile(bs, 2.5)), 2), round(100 * float(np.percentile(bs, 97.5)), 2)]


def phase_report():
    rng = np.random.default_rng(0)
    cbar = np.load(ARR / "cbar_vob_train.npy").astype(np.float64)
    res = {"definition": __doc__.split("정의")[1].split("단계")[0].strip(), "variants": VNAMES,
           "cbar_vob_train_norm": round(float(np.linalg.norm(cbar)), 4), "sets": {}}
    ctr = None
    cbars = {}
    for name in SETS:
        od = ARR / name
        if not (od / "meta.json").exists():
            continue
        meta = json.load(open(od / "meta.json"))
        l1 = np.load(od / "l1.npy"); C = np.load(od / "c.npy").astype(np.float64); G = np.load(od / "g.npy").astype(np.float64)
        pos = np.array([p == "1" for p in meta["plausible"]])
        cD = C[pos].mean(0); gD = G[pos].mean(0)
        cbars[name] = cD
        flat = lambda x: x.reshape(len(x), -1) if x.ndim == 3 else x.reshape(-1)  # noqa: E731
        spread = np.linalg.norm(flat(C[pos]) - flat(cD)[None], axis=1) / np.linalg.norm(flat(cD))
        cos = lambda a, b: float(np.dot(a.ravel(), b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b)))  # noqa: E731
        R = {"kind": SETS[name][3], "n_clip": int(len(pos)), "n_pos": int(pos.sum()),
             "c_norm_mean_set": round(float(np.linalg.norm(cD)), 4),
             "c_norm_per_clip_median": round(float(np.median(np.linalg.norm(flat(C[pos]), axis=1))), 4),
             "spread_median": round(float(np.median(spread)), 3), "spread_q10_q90": [round(float(np.percentile(spread, q)), 3) for q in (10, 90)],
             "cos_cbarD_cbarVOBtrain": round(cos(cD, cbar), 4), "cos_cbarVOBtrain_gD": round(cos(cbar, gD), 4),
             "cos_cbarD_gD": round(cos(cD, gD), 4), "g_norm": round(float(np.linalg.norm(gD)), 4),
             "cos_ci_gi_median": round(float(np.median([cos(C[i], G[i]) for i in np.flatnonzero(pos)])), 4)}
        if name == "vob_train":
            res["sets"][name] = R
            continue
        s = l1.mean(2)
        e0 = s[pos, 0].mean()
        R["eps"] = {v: round(float(s[pos, k].mean() / e0), 4) for k, v in enumerate(VNAMES)}
        R["acc"] = {}
        base = matched(meta, s[:, 0])
        for k, v in enumerate(VNAMES):
            m = matched(meta, s[:, k])
            R["acc"][v] = {"v": round(100 * float(np.mean([x[2] for x in m])), 2), "n": len(m)}
            if k:
                d = [x[2] - y[2] for x, y in zip(m, base)]
                R["acc"][v]["gain_vs_R0"] = round(100 * float(np.mean(d)), 2)
                R["acc"][v]["gain_ci"] = boot(d, [x[0] for x in m], rng)
        mc = matched(meta, np.load(od / "copy.npy").mean(1))
        R["acc"]["copy_z"] = {"v": round(100 * float(np.mean([x[2] for x in mc])), 2), "n": len(mc)}
        # V2 − V0 · V1 − V0 의 쌍 판정 차
        for a_, b_ in ((3, 1), (2, 1), (4, 0)):
            ma, mb = matched(meta, s[:, a_]), matched(meta, s[:, b_])
            d = [x[2] - y[2] for x, y in zip(ma, mb)]
            R["acc"][f"{VNAMES[a_]} − {VNAMES[b_]}"] = {"v": round(100 * float(np.mean(d)), 2), "ci": boot(d, [x[0] for x in ma], rng)}
        if name == "vob_test":   # late 이동 사라짐 A · B, late 전체
            cells = {}
            grp = defaultdict(dict)
            for i in range(len(pos)):
                grp[(meta["objset"][i], meta["block_id"][i])][meta["variant"][i]] = i
            for k, v in enumerate(VNAMES + ["copy_z"]):
                sv = s[:, k] if k < len(VNAMES) else np.load(od / "copy.npy").mean(1)
                for cell, f in (("late_move_perm_A", lambda r, d: r["occlusion"] == "late" and r["motion"] != "flat_static" and r["violation"] == "permanence" and d == "A"),
                                ("late_move_perm_B", lambda r, d: r["occlusion"] == "late" and r["motion"] != "flat_static" and r["violation"] == "permanence" and d == "B"),
                                ("late", lambda r, d: r["occlusion"] == "late"),
                                ("visible", lambda r, d: r["occlusion"] == "visible")):
                    hits, bl = [], []
                    for key, g in grp.items():
                        for d_, (p, q) in (("A", ("pos_a", "imp_ab")), ("B", ("pos_b", "imp_ba"))):
                            if p in g and q in g:
                                r = {c_: meta[c_][g[p]] for c_ in ("occlusion", "motion", "violation")}
                                if f(r, d_):
                                    hits.append(1.0 if sv[g[q]] > sv[g[p]] else 0.0); bl.append(str(key))
                    cells.setdefault(cell, {})[v] = (None if not hits else {"v": round(100 * float(np.mean(hits)), 1), "n": len(hits), "ci": boot(hits, bl, rng)})
            R["vob_cells"] = cells
        res["sets"][name] = R
        print(name, json.dumps({k: R[k] for k in ("spread_median", "cos_cbarD_cbarVOBtrain", "cos_cbarVOBtrain_gD", "eps")}, ensure_ascii=False), flush=True)
        print("   acc", {k: v["v"] for k, v in R["acc"].items()}, flush=True)
    names = list(cbars)
    res["cos_between_sets"] = {f"{a}|{b}": round(float(np.dot(cbars[a].ravel(), cbars[b].ravel()) / (np.linalg.norm(cbars[a]) * np.linalg.norm(cbars[b]))), 4)
                               for i, a in enumerate(names) for b in names[i + 1:]}
    # 판정 (사전 등록)
    ext = [n for n in ("intphys1_dev", "grasp_level2_s5", "inflevel_continuity_s5", "v11sub") if n in res["sets"]]
    v = {}
    try:
        v["spread_ok(≤0.5, 외부)"] = all(res["sets"][n]["spread_median"] <= 0.5 for n in ext)
        v["cos_ok(≥0.8 vs VOBtest)"] = all(res["cos_between_sets"].get(f"vob_test|{n}", res["cos_between_sets"].get(f"{n}|vob_test", 0)) >= 0.8 for n in ext)
        v["V2_ext_eps≤0.98 개수"] = sum(res["sets"][n]["eps"]["V2_pft-cbarVOBtrain"] <= 0.98 for n in ext)
        v["V2_ext_acc+2 개수"] = sum(res["sets"][n]["acc"]["V2_pft-cbarVOBtrain"]["v"] - res["sets"][n]["acc"]["V0_pft"]["v"] >= 2 for n in ext)
        v["V2−V0 ε 차 (외부)"] = {n: round(res["sets"][n]["eps"]["V2_pft-cbarVOBtrain"] - res["sets"][n]["eps"]["V0_pft"], 4) for n in ext}
        if "vob_test" in res["sets"]:
            v["VOB_lateA"] = {k_: (x["v"] if x else None) for k_, x in res["sets"]["vob_test"]["vob_cells"]["late_move_perm_A"].items()}
    except Exception as e:  # noqa: BLE001
        v["error"] = repr(e)
    res["verdict_inputs"] = v
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(v, ensure_ascii=False))
    print("→", OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["train", "score", "report"])
    ap.add_argument("--sets", default="")
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    global SMOKE, ARR
    if a.smoke:
        SMOKE, ARR = a.smoke, ARR / "_smoke"
    ARR.mkdir(parents=True, exist_ok=True)
    if a.phase == "train":
        phase_train()
    elif a.phase == "score":
        cbar = np.load(ARR / "cbar_vob_train.npy")
        for s in a.sets.split(","):
            gpu_loop(s, cbar)
    else:
        phase_report()


if __name__ == "__main__":
    main()
