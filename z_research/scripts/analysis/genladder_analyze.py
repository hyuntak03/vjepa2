#!/usr/bin/env python3
"""일반화 실패 원인 사다리 — 어디서 끊기나 (2026-10-31, z_analysis/release/05_generalization).

판: vob = VOBench_v2 test late permanence (분포 안 기준, 궤적 묶음 반대쪽) · rub = Rubric Office quads (실제 영상, uniform 창)
모델: release · a0 (L1 PFT, VOBench_v2) · a1 (+ binding)
물체 칸 R_t: 미래 블록 t 에서 물체가 차지한 16×16 칸 (vob: label.json 의 중심 · 반폭 → 외곽 상자, rub: SAM2 마스크 → 칸 면적 10 % 이상).
  두 장 (튜블릿) 의 합집합. 그 블록에 물체가 보이는 경우만 (vob: hidden < 0.5, rub: 마스크 있음) 센다.
차이 지도: Δp = p(물체 문맥) − p(빈 문맥),  Δh = h(물체 미래) − h(빈 미래)  (같은 장면, 둘 다 가능 영상)
  vob: pos_a (물체) − pos_b (빈 장면) · rub: possible_present − possible_absent
지표 (사전 정의):
  H1 기전 — cos_R  = cos(R 칸 평균 Δp, R 칸 평균 Δh)    (방향: 그 자리에 실제 물체 차이를 그리나)
          conc_R = ‖Δp‖ 의 R 평균 ÷ R 밖 평균              (물체 차이가 그 자리에 몰리나; Δh 의 같은 값이 천장)
  H2 자리 — W1 = 3×3 평균한 ‖Δp‖ 의 최대 칸이 R 에서 1 칸 안인 비율 (Δh 의 같은 값이 천장, R 1 칸 안 칸 비율이 우연)
  H3 채점 (rub 만, 불가능 영상의 h 가 있어서) — 짝 A (사라짐) 정답률:
          full     = 표준 (미래 8 블록 전체 토큰 L1)
          oracle   = 릴리즈 p(물체 문맥) 의 R (1 칸 넓힘) 토큰을 h(possible_present) 로 바꾼 뒤 표준 채점 — "맞는 자리에 맞는 내용" 상한
          region   = R (1 칸 넓힘) 토큰만으로 L1 채점
출력: z_analysis/release/05_generalization/results/ladder.json
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; $P z_research/scripts/analysis/genladder_analyze.py
"""
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
NFS = Path("/data/hyuntak/project/2026/2027_cvpr/nfs_cache")
OUT = REPO / "z_analysis/release/05_generalization/results/ladder.json"
G, NF = 16, 8
MODELS = ("release", "a0", "a1")


def dil_valid(m):
    v = np.zeros_like(m)
    rr, cc = np.nonzero(m)
    for r, c in zip(rr, cc):
        v[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2] = True
    return v


def box3(a):
    """(G, G) → 3×3 평균 (가장자리는 있는 칸만)."""
    s = np.zeros_like(a); n = np.zeros_like(a)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            sl_d = (slice(max(dr, 0), G + min(dr, 0)), slice(max(dc, 0), G + min(dc, 0)))
            sl_s = (slice(max(-dr, 0), G + min(-dr, 0)), slice(max(-dc, 0), G + min(-dc, 0)))
            s[sl_d] += a[sl_s]; n[sl_d] += 1
    return s / n


def map_metrics(dp, dh, R):
    """dp, dh: (256, D) 블록 하나의 차이 · R: (G, G) bool → dict."""
    np_ = np.linalg.norm(dp, axis=-1).reshape(G, G); nh = np.linalg.norm(dh, axis=-1).reshape(G, G)
    Rf = R.reshape(-1); near = dil_valid(R)
    out = {}
    a, b = dp[Rf].mean(0), dh[Rf].mean(0)
    out["cos_R"] = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    out["mag_R"] = float(np.linalg.norm(a) / (np.linalg.norm(b) + 1e-9))
    out["conc_p"] = float(np_[R].mean() / max(np_[~R].mean(), 1e-9)); out["conc_h"] = float(nh[R].mean() / max(nh[~R].mean(), 1e-9))
    for nm, mp in (("p", np_), ("h", nh)):
        r, c = divmod(int(box3(mp).argmax()), G)
        out[f"W1_{nm}"] = float(near[r, c])
    out["W1_chance"] = float(near.mean())
    return out


# ----------------------------------------------------------------------------- 실제 영상
def rubric():
    base = NFS / "genladder/rubric_uniform"
    meta = json.load(open(base / "meta.json"))["meta"]
    H = np.load(base / "h_fut.npy", mmap_mode="r"); P = {m: np.load(base / f"p_{m}.npy", mmap_mode="r") for m in MODELS}
    audit = json.load(open(REPO / "data_csv/rubric_quads_uniform/audit.json"))
    SRC = Path("/data/hyuntak/project/2026/2027_cvpr/rubric_office_quads")
    by = defaultdict(dict)
    for i, m in enumerate(meta):
        by[m["source_block"]][m["variant"]] = i
    rows = []
    for d, v in by.items():
        z = np.load(SRC / d / "masks.npz"); T, Hh, Ww = z["shape"]
        sam = np.unpackbits(z["sam2"], axis=-1)[..., :Ww].astype(bool)
        a = audit[d]; step = 2 if a["duplicated"] else 1
        fut = [f * step for f in a["frames_eff"][16:]]
        ia, ib, iv = v["pos_a"], v["pos_b"], v["imp_ab"]
        hA, hB, hV = (np.asarray(H[i], np.float32) for i in (ia, ib, iv))
        for t in range(NF):
            m2 = sam[fut[2 * t]] | sam[fut[2 * t + 1]]
            if not m2.any():
                continue
            # 640×480 → 16×16 칸 (채점기처럼 정사각으로 늘림): 칸 면적 10 % 이상
            cell = m2.reshape(G, Hh // G, G, Ww // G).mean((1, 3)) if (Hh % G == 0 and Ww % G == 0) else None
            R = cell >= 0.10
            if not R.any():
                R = cell == cell.max()
            r = dict(set="rub", scene=d, t=t)
            for mname in MODELS:
                pa, pb = np.asarray(P[mname][ia][t], np.float32), np.asarray(P[mname][ib][t], np.float32)
                for k, val in map_metrics(pa - pb, hA[t] - hB[t], R).items():
                    r[f"{mname}|{k}"] = val
            rows.append(r)
    # H3 채점 (짝 A, 영상 단위)
    h3 = []
    for d, v in by.items():
        z = np.load(SRC / d / "masks.npz"); T, Hh, Ww = z["shape"]
        sam = np.unpackbits(z["sam2"], axis=-1)[..., :Ww].astype(bool)
        a = audit[d]; step = 2 if a["duplicated"] else 1
        fut = [f * step for f in a["frames_eff"][16:]]
        Rall = np.zeros((NF, G * G), bool)
        for t in range(NF):
            m2 = sam[fut[2 * t]] | sam[fut[2 * t + 1]]
            if m2.any():
                cell = m2.reshape(G, Hh // G, G, Ww // G).mean((1, 3)); R = cell >= 0.10
                if not R.any():
                    R = cell == cell.max()
                Rall[t] = dil_valid(R).reshape(-1)
        ia, iv = v["pos_a"], v["imp_ab"]
        hA, hV = np.asarray(H[ia], np.float32), np.asarray(H[iv], np.float32)
        r = dict(scene=d, n_region_tok=int(Rall.sum()))
        for mname in MODELS:
            p = np.asarray(P[mname][ia], np.float32)
            r[f"{mname}|full"] = float(np.abs(p - hV).mean() > np.abs(p - hA).mean())
            if Rall.any():
                r[f"{mname}|region"] = float(np.abs(p - hV)[Rall].mean() > np.abs(p - hA)[Rall].mean())
            x = p.copy(); x[Rall] = hA[Rall]
            r[f"{mname}|oracle"] = float(np.abs(x - hV).mean() > np.abs(x - hA).mean())
        h3.append(r)
    return rows, h3


# ----------------------------------------------------------------------------- VOBench_v2 test late permanence
def vobench():
    base = NFS / "release_late_vobench_v2"
    meta = json.load(open(base / "meta.json"))["meta"]
    H = np.load(base / "h_fut.npy", mmap_mode="r")
    P = {"release": np.load(base / "p.npy", mmap_mode="r"), "a0": np.load(NFS / "bind_v2/a0_late/p.npy", mmap_mode="r"),
         "a1": np.load(NFS / "bind_v2/a1_late/p.npy", mmap_mode="r")}
    train_g = set(json.load(open(REPO / "data_csv/vobench_v2_split/split.json"))["train_groups"])
    grp = lambda m: "|".join(m[k] for k in ("objset", "background", "motion", "occlusion", "k"))
    paths = {r["block_id"]: r["path"] for r in csv.DictReader(open("/data2/local_datasets/world/Benchmark/VOBench_v2/blocks.csv"))}
    by = defaultdict(dict)
    for i, m in enumerate(meta):
        if m["violation"] == "permanence" and m["motion"] != "flat_static" and grp(m) not in train_g:
            by[m["source_block"]][m["variant"]] = i
    rows = []
    for sb, v in by.items():
        if "pos_a" not in v or "pos_b" not in v:
            continue
        lab = json.load(open(f"/data2/local_datasets/world/Benchmark/VOBench_v2/{paths[sb]}/label.json"))["labels"]["A"]
        ia, ib = v["pos_a"], v["pos_b"]
        hA, hB = np.asarray(H[ia], np.float32), np.asarray(H[ib], np.float32)
        for t in range(NF):
            fr = [48 + 6 * t, 51 + 6 * t]
            if np.mean([lab[f]["hidden"] for f in fr]) >= 0.5:
                continue
            R = np.zeros((G, G), bool)
            for f in fr:
                u, vv, eu, ev = lab[f]["u"] / 32, lab[f]["v"] / 32, lab[f]["ext_u"] / 32, lab[f]["ext_v"] / 32
                R[max(int(vv - ev), 0):min(int(vv + ev), G - 1) + 1, max(int(u - eu), 0):min(int(u + eu), G - 1) + 1] = True
            r = dict(set="vob", scene=sb, t=t, k=meta[ia]["k"])
            for mname in MODELS:
                pa, pb = np.asarray(P[mname][ia][t], np.float32), np.asarray(P[mname][ib][t], np.float32)
                for kk, val in map_metrics(pa - pb, hA[t] - hB[t], R).items():
                    r[f"{mname}|{kk}"] = val
            rows.append(r)
    return rows


def boot(vals, groups, n=2000):
    vals = np.asarray(vals, float); groups = np.asarray(groups); ub = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ub}
    rng = np.random.default_rng(0)
    bs = [np.mean(np.concatenate([vals[idx[g]] for g in rng.choice(ub, len(ub))])) for _ in range(n)]
    return [float(vals.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    res = {}
    rv = vobench(); rr, h3 = rubric()
    for name, rows in (("vob", rv), ("rub", rr)):
        sc = [r["scene"] for r in rows]
        d = {"n_blocks": len(rows), "n_scenes": len(set(sc))}
        for mname in MODELS:
            for k in ("cos_R", "mag_R", "conc_p", "W1_p"):
                d[f"{mname}|{k}"] = boot([r[f"{mname}|{k}"] for r in rows], sc)
        d["ceiling|conc_h"] = boot([r["release|conc_h"] for r in rows], sc); d["ceiling|W1_h"] = boot([r["release|W1_h"] for r in rows], sc)
        d["chance|W1"] = boot([r["release|W1_chance"] for r in rows], sc)
        res[name] = d
    sc = [r["scene"] for r in h3]
    res["rub_H3"] = {f"{m}|{k}": boot([r[f"{m}|{k}"] for r in h3 if f"{m}|{k}" in r], [r["scene"] for r in h3 if f"{m}|{k}" in r])
                     for m in MODELS for k in ("full", "region", "oracle")}
    res["rub_H3"]["n_scenes"] = len(h3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(definition=__doc__, results=res, rub_h3_rows=h3), open(OUT, "w"), indent=1, ensure_ascii=False)
    for name in ("vob", "rub"):
        d = res[name]
        print(f"\n[{name}] blocks {d['n_blocks']} scenes {d['n_scenes']} · 천장 conc_h {d['ceiling|conc_h'][0]:.2f} W1_h {100 * d['ceiling|W1_h'][0]:.0f} · 우연 W1 {100 * d['chance|W1'][0]:.0f}")
        for m in MODELS:
            print(f"  {m:8s} cos_R {d[m + '|cos_R'][0]:.2f} [{d[m + '|cos_R'][1]:.2f},{d[m + '|cos_R'][2]:.2f}] · mag_R {d[m + '|mag_R'][0]:.2f} · conc_p {d[m + '|conc_p'][0]:.2f} · W1_p {100 * d[m + '|W1_p'][0]:.0f} [{100 * d[m + '|W1_p'][1]:.0f},{100 * d[m + '|W1_p'][2]:.0f}]")
    print("\n[rub H3] 짝 A (사라짐) 정답률, 영상", res["rub_H3"]["n_scenes"])
    for m in MODELS:
        print(f"  {m:8s} " + " · ".join(f"{k} {100 * res['rub_H3'][m + '|' + k][0]:.0f} [{100 * res['rub_H3'][m + '|' + k][1]:.0f},{100 * res['rub_H3'][m + '|' + k][2]:.0f}]" for k in ("full", "region", "oracle")))


if __name__ == "__main__":
    main()
