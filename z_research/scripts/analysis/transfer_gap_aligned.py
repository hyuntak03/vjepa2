#!/usr/bin/env python3
"""transfer_gap Q3 · E28 — 외부 벤치의 '문맥 끝 판' (위반 직전에 문맥 끝을 맞춘 고정 창) 전이 행렬. 2026-10-30, 실험 B [X2] R2.

왜: R1 검토에서 IntPhys 2 문맥 끝 판 (라벨 −10 장) 의 VOBench PFT permanence 이득이 +9.2 인데 라벨 판은 −0.8 이었다.
    효과가 외부에서도 "문맥 끝 정렬 창" 에서만 생기나 (H3-W), 판마다 잡음인가를 같은 쌍 · 장면 bootstrap 으로 가른다.

정의 (CPU, 기존 per_block.json · copy_baseline.json 만 읽는다)
  쌍 = index.csv 의 (block_id, pair_id) 안 (가능 1, 불가능 1). 판정 = s(imp) > s(pos) (같으면 0.5).
  acc(model, panel, 칸) = 쌍 판정 평균 × 100.  칸 = 전체 · 성질 (IntPhys 2 condition / IntPhys 1 principle) · 운동 (IntPhys 1 group)
  이득 G = acc(PFT) − acc(기준) 같은 쌍. 기준: 릴리즈 계열 PFT → 릴리즈 (vith), vast AR + PFT → vast AR.
  복사 Δ = acc(model) − acc(copy_z) (copy_h 도). copy = 문맥 마지막 튜블릿 latent 를 미래 8 칸에 복사한 surprise.
  판 사이 이득 차 = G(판 X) − G(판 Y), 두 판에 모두 있는 쌍 (가능 · 불가능 영상 이름으로 맞춤) 만.
  판 사이 쌍 판정 일치율 = 같은 모델 · 같은 쌍에서 두 판의 판정이 같은 비율.
  CI = 장면 (IntPhys 2 = block_id 장면, IntPhys 1 = 사분 묶음 block_id) bootstrap 4,000, 2.5 · 97.5 분위.
  ⚠️ IntPhys 2 는 가능 · 불가능 문맥이 화소로 같지 않다 (p 가 쌍 안에서 다르다). IntPhys 1 late 창 PFT (ip1late) 는
     IntPhys 1 late uni 의 fold 0 로 학습했고 fold 목록 (data_csv/intphys1_late_uni_cv) 이 지금 없다 → IntPhys 1 판의 ip1late 수치는 누수가 섞여 쓰지 않는다.
     IntPhys 2 쌍 PFT (ip2pair) 는 IntPhys 2 Main 의 짝 split 으로 학습했다 → 겹치는 영상 비율을 같이 낸다.

입력: cvpr/results/eval/intphys{1,2}_late/<ds>_<model>/{per_block,copy_baseline}.json · data_csv/<ds>/index.csv
      처음 보는 물체 PFT: z_analysis/transfer_gap/results/q3_r2/runs/E28_<ds>_objho/per_block.json
출력: z_analysis/transfer_gap/results/q3_r2/E28_aligned.json

  /data/hyuntak/anaconda3/envs/vjepa2/bin/python z_research/scripts/analysis/transfer_gap_aligned.py
"""
from __future__ import annotations

import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
RES = REPO / "cvpr/results/eval"
RUNS = REPO / "z_analysis/transfer_gap/results/q3_r2/runs"
OUT = REPO / "z_analysis/transfer_gap/results/q3_r2/E28_aligned.json"
NB = 4000
PANELS = {  # 판 이름 → (task 폴더, 데이터셋, 성질 열, 운동 열)
    "ip2_uni": ("intphys2_late", "intphys2_late_uni", "condition", None),
    "ip2_uni_m10": ("intphys2_late", "intphys2_late_uni_m10", "condition", None),
    "ip2_final": ("intphys2_late", "intphys2_late_final", "condition", None),
    "ip1_uni": ("intphys1_late", "intphys1_late_uni", "principle", "group"),
    "ip1_hid": ("intphys1_late", "intphys1_late_hid", "principle", "group"),
    "ip1_rev": ("intphys1_late", "intphys1_late_rev", "principle", "group"),
    "ip1_rev_dn1": ("intphys1_late", "intphys1_late_rev_dn1", "principle", "group"),
    "ip1_auto2-6r12fit": ("intphys1_late", "intphys1_late_auto2-6r12fit", "principle", "group"),
}
MODELS = {  # 행 이름 → (레지스트리 이름 또는 None=objho, 기준 행)
    "release": ("vith", None),
    "vob_pft": ("vith_release_vobftg", "release"),
    "objho_pft": (None, "release"),
    "ip1late_pft": ("vith_pft_ip1late_f0", "release"),
    "ip2pair_pft": ("vith_pft_ip2pair", "release"),
    "vast_ar": ("vith_vast_ar_state_ep70", None),
    "vast_ar_pft": ("vith_vast_ar_state_ep70_vobftg", "vast_ar"),
}


def load_pairs(ds, pcol, gcol):
    by = defaultdict(dict)
    for r in csv.DictReader(open(REPO / "data_csv" / ds / "index.csv")):
        by[(r["block_id"], r["pair_id"])]["pos" if r["plausible"] == "1" else "imp"] = r
    pairs = []
    for (b, pid), v in by.items():
        if "pos" in v and "imp" in v:
            pairs.append({"key": (v["pos"]["video_id"], v["imp"]["video_id"]), "scene": b, "prop": v["pos"][pcol],
                          "move": v["pos"][gcol] if gcol else None, "file_pos": v["pos"].get("file", ""),
                          "file_imp": v["imp"].get("file", "")})
    return pairs


def surprise(task, ds, model):
    if model is None:
        p = RUNS / f"E28_{ds}_objho/per_block.json"
    else:
        p = RES / task / f"{ds}_{model}/per_block.json"
    if not p.exists():
        return None
    return json.load(open(p))["per_video_surprise"]


def copy_s(task, ds):
    p = RES / task / f"{ds}_vith/copy_baseline.json"
    if not p.exists():
        return None
    return json.load(open(p))["per_video"]


def judge(s, pairs):
    out = np.full(len(pairs), np.nan)
    for i, q in enumerate(pairs):
        a, b = s.get(q["key"][0]), s.get(q["key"][1])
        if a is None or b is None:
            continue
        out[i] = 1.0 if b > a else (0.5 if b == a else 0.0)
    return out


def boot_idx(scenes, rng):
    u = sorted(set(scenes))
    m = {s: np.flatnonzero(np.asarray(scenes) == s) for s in u}
    for _ in range(NB):
        pick = rng.choice(len(u), len(u))
        yield np.concatenate([m[u[k]] for k in pick])


def stat(x, scenes, rng, ref=None):
    """x (n,) 판정 0/1 → acc · CI. ref 가 있으면 (x − ref) 의 평균 이득 · CI. nan 은 뺀다 (둘 다 있는 쌍)."""
    ok = ~np.isnan(x) if ref is None else (~np.isnan(x) & ~np.isnan(ref))
    if ok.sum() == 0:
        return None
    xs = x[ok] if ref is None else x[ok] - ref[ok]
    sc = [s for s, o in zip(scenes, ok) if o]
    bs = [xs[ix].mean() for ix in boot_idx(sc, rng)]
    return {"n": int(ok.sum()), "v": round(100 * float(xs.mean()), 2),
            "ci": [round(100 * float(np.percentile(bs, 2.5)), 2), round(100 * float(np.percentile(bs, 97.5)), 2)]}


def ip2pair_overlap(pairs):
    """IntPhys 2 쌍 PFT 학습 split 에 들어간 Main 영상 (hash) 비율."""
    tr = set()
    for f in glob.glob(str(REPO / "data_csv/IntPhys2/pair_split/index_s*.csv")):
        for r in csv.DictReader(open(f)):
            tr.add(r["video_id"])
    files = [Path(q[k]).stem for q in pairs for k in ("file_pos", "file_imp")]
    return {"n_video": len(files), "in_ip2pair_train_any_split": sum(f in tr for f in files),
            "note": "pair_split 의 index_s*.csv 26 개 = ip2pair 학습 config 의 datasets 전부 (2026-10-30 확인) → 이 수가 학습에 쓰인 영상 수"}


def main():
    rng = np.random.default_rng(0)
    res = {"definition": __doc__.split("정의")[1].split("입력:")[0].strip(), "panels": {}, "between": {}}
    J = {}
    for pn, (task, ds, pcol, gcol) in PANELS.items():
        pairs = load_pairs(ds, pcol, gcol)
        scenes = [q["scene"] for q in pairs]
        P = {"dataset": ds, "n_pairs": len(pairs), "models": {}, "gains": {}, "vs_copy": {}}
        if pn.startswith("ip2"):
            P["ip2pair_train_overlap"] = ip2pair_overlap(pairs)
        cs = copy_s(task, ds)
        judged = {}
        if cs:
            for ck in ("copy_z", "copy_h"):
                judged[ck] = judge(cs[ck], pairs)
        for mn, (reg, _) in MODELS.items():
            s = surprise(task, ds, reg)
            if s is not None:
                judged[mn] = judge(s, pairs)
        J[pn] = (pairs, judged)
        cells = {"all": np.ones(len(pairs), bool)}
        for v in sorted(set(q["prop"] for q in pairs)):
            cells[f"prop={v}"] = np.array([q["prop"] == v for q in pairs])
        if gcol:
            for v in sorted(set(q["move"] for q in pairs)):
                cells[f"move={v}"] = np.array([q["move"] == v for q in pairs])
            for v in sorted(set(q["prop"] for q in pairs)):
                for w in sorted(set(q["move"] for q in pairs)):
                    m = np.array([q["prop"] == v and q["move"] == w for q in pairs])
                    if m.sum() >= 5:
                        cells[f"prop={v}|move={w}"] = m
        for mn, x in judged.items():
            P["models"][mn] = {c: stat(np.where(m, x, np.nan), scenes, rng) for c, m in cells.items()}
        for mn, (_, base) in MODELS.items():
            if base and mn in judged and base in judged:
                P["gains"][f"{mn}-{base}"] = {c: stat(np.where(m, judged[mn], np.nan), scenes, rng, judged[base])
                                               for c, m in cells.items()}
        for mn in judged:
            if mn.startswith("copy"):
                continue
            for ck in ("copy_z", "copy_h"):
                if ck in judged:
                    P["vs_copy"][f"{mn}-{ck}"] = {c: stat(np.where(m, judged[mn], np.nan), scenes, rng, judged[ck])
                                                  for c, m in cells.items() if c == "all" or c.startswith("prop=") and "|" not in c}
        P["missing_models"] = [m for m in MODELS if m not in judged]
        if pn.startswith("ip1"):
            P["note"] = "ip1late_pft 는 이 판의 영상 일부로 학습 (fold 목록 없음) → 해석 금지"
        res["panels"][pn] = P
        print(pn, len(pairs), {m: P["models"][m]["all"]["v"] for m in P["models"]}, flush=True)

    # 판 사이: 같은 벤치 안의 판 짝
    combos = [("ip2_uni", "ip2_uni_m10"), ("ip2_uni", "ip2_final"), ("ip2_uni_m10", "ip2_final"),
              ("ip1_uni", "ip1_hid"), ("ip1_uni", "ip1_rev"), ("ip1_rev", "ip1_rev_dn1"), ("ip1_uni", "ip1_auto2-6r12fit"),
              ("ip1_hid", "ip1_auto2-6r12fit")]
    for a, b in combos:
        pa, ja = J[a]
        pb, jb = J[b]
        ia = {q["key"]: i for i, q in enumerate(pa)}
        common = [(ia[q["key"]], j) for j, q in enumerate(pb) if q["key"] in ia]
        if not common:
            res["between"][f"{a}|{b}"] = {"n_common": 0}
            continue
        A = np.array([c[0] for c in common]); B = np.array([c[1] for c in common])
        scenes = [pa[i]["scene"] for i in A]
        props = [pa[i]["prop"] for i in A]
        E = {"n_common": len(common), "agree": {}, "gain_diff": {}}
        for mn in set(ja) & set(jb):
            x, y = ja[mn][A], jb[mn][B]
            ok = ~np.isnan(x) & ~np.isnan(y)
            if ok.sum():
                E["agree"][mn] = round(100 * float((x[ok] == y[ok]).mean()), 1)
        for mn, (_, base) in MODELS.items():
            if not base or not all(k in ja and k in jb for k in (mn, base)):
                continue
            ga = ja[mn][A] - ja[base][A]
            gb = jb[mn][B] - jb[base][B]
            d = {}
            for c in ["all"] + [f"prop={v}" for v in sorted(set(props))]:
                m = np.ones(len(A), bool) if c == "all" else np.array([p == c[5:] for p in props])
                d[c] = stat(np.where(m, gb - ga + 0.0, np.nan) + 0.5, scenes, rng)  # +0.5 이동 후 아래에서 되돌림
                if d[c]:
                    d[c] = {"n": d[c]["n"], "v": round(d[c]["v"] - 50, 2), "ci": [round(c_ - 50, 2) for c_ in d[c]["ci"]]}
            E["gain_diff"][f"{mn}-{base}: {b} − {a}"] = d
        res["between"][f"{a}|{b}"] = E
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print("→", OUT)


if __name__ == "__main__":
    main()
