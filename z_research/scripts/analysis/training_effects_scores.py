#!/usr/bin/env python
"""TrainingEffects — 새로 학습한 모델들의 v11 · IntPhys1 점수를 산출물에서 다시 계산해 표로 만든다 (GPU 불필요).

    python z_research/scripts/analysis/training_effects_scores.py            # -> z_research/TrainingEffects/scores/
    python z_research/scripts/analysis/training_effects_scores.py --boot 0   # CI 없이 빠르게

v11  : surprise_c16t32 × v11_split_test (10,752 matched pair, 학습 안 한 block 절반).
       per_video_surprise 를 index_test.csv 와 조인해 쌍 (block_id, pair_id) 마다 hit = 1[s_pos < s_imp] (+0.5 동점)
       — evals/world_model_analysis/eval.py::score_blocks 와 같은 규칙. 전체 값이 summary.json 과 일치하는지 검사한다.
       CI 는 block 단위 bootstrap (block 안 2 쌍은 context 를 공유한다).
IntPhys1: Garrido A.8 — garrido_rescore.collect() 로 창 16 / 32 실행을 합쳐 Filtered + property macro 최고 칸.
       CI 는 최고 칸에서 쌍 단위 bootstrap (property macro 를 매번 다시 낸다).
릴리즈 v11_split_test 는 재실행 없이 기존 v11 + v11_earlymid 실행의 per_video_surprise 에서 test 절반만 골라 쓴다
(z_training/runs/release_vith/eval/v11_split_test_from_existing_runs.json 과 같은 방법 — 그 값 75.83 과 일치해야 한다).
"""
from __future__ import annotations
import argparse, csv, json, pathlib, sys
from collections import defaultdict

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
import garrido_rescore as GR  # noqa: E402

OUT = REPO / "z_research/TrainingEffects/scores"
VX = REPO / "z_research/IntPhysGenV11/exp_results"
BX = REPO / "z_research/Benchmarks/exp_results"
ZR = REPO / "z_training/runs"
INDEX = REPO / "data_csv/intphysgen_v11_split/index_test.csv"

# (key, 표시 이름, group, v11 결과 폴더 목록, IntPhys1 모델 이름, 학습 설명)
MODELS = [
    # --- A: 릴리즈 ViT-H encoder (frozen) + predictor 만 다르다
    ("release", "release ViT-H", "A",
     [VX / "surprise_c16t32__v11_vith",
      REPO / "z_research/IntPhysGenV11_occlusion_timing_ablation/exp_results/surprise_c16t32__v11_earlymid_vith"],
     "vith", "Meta 릴리즈 (VideoMix22M, 학습 안 함)"),
    ("pft_v11", "PFT v11 e10", "A", [ZR / "v11_postft/eval/surprise_c16t32__v11_split_test_e10"],
     "vith_pft_v11_e10", "릴리즈 predictor post-FT · v11 train 절반 (가능 10,752 clip) · 10 ep"),
    ("pft_intphys1", "PFT IntPhys1 e40", "A", [ZR / "intphys1_postft/eval/surprise_c16t32__v11_split_test_e40"],
     "vith_pft_intphys1_e40", "post-FT · IntPhys1 train 장면 (~3,750 clip) · 40 ep"),
    ("pft_intphys2", "PFT IntPhys2 e80", "A", [ZR / "intphys2_postft/eval/surprise_c16t32__v11_split_test_e80"],
     "vith_pft_intphys2_e80", "post-FT · IntPhys2 Main train (254 clip) · 80 ep"),
    ("pft_predv1", "PFT Predictor_v1 e15", "A", [ZR / "predictor_v1_postft/eval/surprise_c16t32__v11_split_test_e15"],
     "vith_pft_predv1_e15", "post-FT · Predictor_v1_training (14,250 clip) · 15 ep"),
    ("ariel_bc43", "Ariel block-causal ep43", "A", [VX / "surprise_c16t32__v11_split_test_vith_ariel_bc_ep43"],
     "vith_ariel_bc_ep43", "scratch predictor · SSv2+K400 · prefix/block-causal attention · 43 ep"),
    # --- B: encoder + predictor 전부 새로 (jongseo 사전학습)
    ("vitb_k400_e240", "ViT-B K400 72K", "B", [VX / "surprise_c16t32__v11_split_test_vitb_k400_e240"],
     "vitb_k400_e240", "ViT-B · K400 240K clip · 72K step (90K 스케줄 중단, 최종)"),
    ("vitb_k400_e100", "ViT-B K400 30K*", "B", [VX / "surprise_c16t32__v11_split_test_vitb_k400_e100"],
     "vitb_k400_e100", "ViT-B · K400 · 30K step (90K 스케줄 중간, lr 미감소)"),
    ("vitb_synphys_30k", "ViT-B synphys 30K", "B", [VX / "surprise_c16t32__v11_split_test_vitb_synphys_30k"],
     "vitb_synphys_30k", "ViT-B · 합성 물리 50K clip · 30K step 완주"),
    ("vittiny_k400_5k", "Tiny K400 5K", "B", [VX / "surprise_c16t32__v11_split_test_vittiny_k400_5k"],
     "vittiny_k400_5k", "ViT-Tiny · K400 · 5K step 완주"),
    ("vittiny_synphys_5k", "Tiny synphys 5K", "B", [VX / "surprise_c16t32__v11_split_test_vittiny_synphys_5k"],
     "vittiny_synphys_5k", "ViT-Tiny · 합성 물리 · 5K step 완주"),
    ("vittiny_k400_30k", "Tiny K400 30K", "B", [VX / "surprise_c16t32__v11_split_test_vittiny_k400"],
     "vittiny_k400", "ViT-Tiny · K400 · 30K step 완주"),
    ("vittiny_synphys_30k_e225", "Tiny synphys 22.5K*", "B",
     [VX / "surprise_c16t32__v11_split_test_vittiny_synphys_30k_e225"],
     "vittiny_synphys_30k_e225", "ViT-Tiny · 합성 물리 · 30K run 의 22.5K (학습 중)"),
    ("vittiny_k400ssv2_30k_e225", "Tiny K400+SSv2 22.5K*", "B",
     [VX / "surprise_c16t32__v11_split_test_vittiny_k400ssv2_30k_e225"],
     "vittiny_k400ssv2_30k_e225", "ViT-Tiny · K400 0.77 + SSv2 0.23 · 30K run 의 22.5K (학습 중)"),
]

TIMING = {"visible": ["static_visible", "moving_visible_flat", "moving_visible"],
          "early": ["static_occlusion_early", "moving_occlusion_flat_early", "moving_occlusion_early"],
          "mid": ["static_occlusion_mid", "moving_occlusion_flat_mid", "moving_occlusion_mid"],
          "late": ["static_occlusion", "moving_occlusion_flat", "moving_occlusion"]}
MOTION = {"static": "static_", "flat": "moving_*_flat", "ramp": "moving_ramp"}
SIX = TIMING["visible"] + TIMING["late"]          # CLAUDE.md §5-1 의 v11 6 조건


def motion_of(cond: str) -> str:
    if cond.startswith("static"):
        return "static"
    return "flat" if "_flat" in cond else "ramp"


def timing_of(cond: str) -> str:
    for t, cs in TIMING.items():
        if cond in cs:
            return t
    raise KeyError(cond)


# ---------------------------------------------------------------- v11
def load_pairs():
    """-> list of pairs (block_id, pair_id, pos_vid, imp_vid, cond, viol, role_imp, sym_k)"""
    rows = list(csv.DictReader(open(INDEX)))
    g = defaultdict(dict)
    for r in rows:
        g[(r["block_id"], r["pair_id"])][int(r["plausible"])] = r
    pairs = []
    for (b, pid), d in sorted(g.items(), key=lambda x: (int(x[0][0]), x[0][1])):
        assert set(d) == {0, 1}, (b, pid)
        p, n = d[1], d[0]
        pairs.append(dict(block=b, pair=pid, pos=p["video_id"], imp=n["video_id"], cond=p["condition"],
                          viol=p["violation_type"], role=n["role"], k=int(p["sym_k"])))
    return pairs


def v11_hits(pairs, dirs):
    pv = {}
    for d in dirs:
        f = d / "per_block.json"
        if not f.exists():
            return None, None
        pv.update(json.load(open(f))["per_video_surprise"])
    missing = [p for p in pairs if p["pos"] not in pv or p["imp"] not in pv]
    if missing:
        return None, f"{len(missing)} pair 의 surprise 없음"
    h = np.array([1.0 if pv[p["pos"]] < pv[p["imp"]] else (0.5 if pv[p["pos"]] == pv[p["imp"]] else 0.0)
                  for p in pairs])
    return h, None


def boot_ci(hits, blocks, B, rng, ref=None):
    """block bootstrap. hits (n,), blocks (n,) int codes. ref 주면 차이 (hits − ref) 의 CI."""
    x = hits if ref is None else hits - ref
    ub, inv = np.unique(blocks, return_inverse=True)
    s = np.bincount(inv, weights=x); c = np.bincount(inv)
    if B <= 0:
        return float(x.mean() * 100), None, None
    idx = rng.integers(0, len(ub), size=(B, len(ub)))
    m = s[idx].sum(1) / c[idx].sum(1)
    return float(x.mean() * 100), float(np.percentile(m, 2.5) * 100), float(np.percentile(m, 97.5) * 100)


def v11_table(pairs, hits, B, rng, ref=None):
    blocks = np.array([int(p["block"]) for p in pairs])
    out = {}
    def put(name, mask):
        if mask.sum() == 0:
            return
        acc, lo, hi = boot_ci(hits[mask], blocks[mask], B, rng)
        e = {"acc": acc, "lo": lo, "hi": hi, "n_pair": int(mask.sum())}
        if ref is not None:
            d, dlo, dhi = boot_ci(hits[mask], blocks[mask], B, rng, ref[mask])
            e.update(delta=d, dlo=dlo, dhi=dhi)
        out[name] = e
    cond = np.array([p["cond"] for p in pairs]); viol = np.array([p["viol"] for p in pairs])
    role = np.array([p["role"] for p in pairs]); k = np.array([p["k"] for p in pairs])
    tim = np.array([timing_of(c) for c in cond]); mot = np.array([motion_of(c) for c in cond])
    put("overall", np.ones(len(pairs), bool))
    put("six_cond", np.isin(cond, SIX))
    for t in TIMING:
        put(f"timing={t}", tim == t)
    for m in ("static", "flat", "ramp"):
        put(f"motion={m}", mot == m)
    for v in ("vanish", "shape", "color"):
        put(f"viol={v}", viol == v)
    for c in sorted(set(cond)):
        put(f"cond={c}", cond == c)
        for v in ("vanish", "shape", "color"):
            put(f"cond={c}|viol={v}", (cond == c) & (viol == v))
    for t in TIMING:
        for m in ("static", "flat", "ramp"):
            for v in ("vanish", "shape", "color"):
                put(f"timing={t}|motion={m}|viol={v}", (tim == t) & (mot == m) & (viol == v))
            for r in ("imp_vanish", "imp_appear"):
                put(f"timing={t}|motion={m}|role={r}", (tim == t) & (mot == m) & (role == r))
    for r in ("imp_vanish", "imp_appear", "imp_A_to_B", "imp_B_to_A"):
        put(f"role={r}", role == r)
    for kk in range(5):
        put(f"occluded|k={kk}", (tim != "visible") & (k == kk))
    return out


# ---------------------------------------------------------------- IntPhys1
def ip1_entry(model, B, rng):
    r = GR.collect(BX, "intphys1_dev", model)
    if r is None:
        return None
    best = r["best"]
    cells = {}
    for combo, e in r["combos"].items():
        R = e["reductions"]
        acc, pick = GR._macro(R, ["filtered"])
        cells[combo] = {"macro": acc, "per_group": {g: v["acc"] for g, v in pick.items()},
                        "pooled": R["filtered"]["overall"], "n_pair": R["filtered"]["n_pair"],
                        "n_ties": R["filtered"]["n_ties"]}
    # 최고 칸 쌍 단위 bootstrap
    combo = best["combo"]
    lo = hi = None
    if B > 0:
        d = next(pathlib.Path(x) for x in r["dirs"] if combo.split("_")[-1] in pathlib.Path(x).name)
        cfg, meta = GR.load_index(d)
        pw = json.load(open(d / "per_window.json"))["windows"]
        vids, _, _, arr = GR.score_table(pw, combo)
        sc = {v: float(np.nanmean(np.nanmin(arr[i], axis=1))) for i, v in enumerate(vids)}
        pairs = defaultdict(list)
        for v, s in sc.items():
            m = meta[v]; pairs[GR.pair_key(m)].append((int(m["plausible"]), s, m.get("block_type", "all")))
        H, G = [], []
        for mem in pairs.values():
            pos = [x for x in mem if x[0] == 1]; imp = [x for x in mem if x[0] == 0]
            if len(pos) == 1 and len(imp) == 1:
                H.append(1.0 if pos[0][1] < imp[0][1] else (0.5 if pos[0][1] == imp[0][1] else 0.0))
                G.append(pos[0][2])
        H = np.array(H); G = np.array(G); groups = sorted(set(G))
        idx = rng.integers(0, len(H), size=(B, len(H)))
        macros = np.mean([np.array([(H[i][G[i] == g]).mean() for i in idx]) for g in groups], axis=0) * 100
        lo, hi = float(np.percentile(macros, 2.5)), float(np.percentile(macros, 97.5))
    return {"best_combo": combo, "best": best["accuracy"], "lo": lo, "hi": hi,
            "best_per_group": {g: v["acc"] for g, v in best["per_group"].items()},
            "alt_notebook": best["alt_notebook"], "cells": cells, "dirs": r["dirs"]}


def legacy_ip1(key):
    """z_training/eval.sh 로 창을 한 실행에 몰아 잰 옛 값 (model.window_size 32) — 대조용."""
    run = {"pft_v11": "v11_postft/eval/intphys1_sliding__intphys1_dev_e10",
           "pft_intphys1": "intphys1_postft/eval/intphys1_sliding__intphys1_dev_e40",
           "pft_intphys2": "intphys2_postft/eval/intphys1_sliding__intphys1_dev_e80",
           "pft_predv1": "predictor_v1_postft/eval/intphys1_sliding__intphys1_dev_e15"}.get(key)
    if run is None or not (ZR / run / "per_window.json").exists():
        return None
    r = GR.rescore(ZR / run)
    b = GR.select(r["combos"], r["dataset_kind"])
    return {"best_combo": b["combo"], "best": b["accuracy"],
            "skip2_w32": GR._macro(r["combos"]["skip2_w32"]["reductions"], ["filtered"])[0]
            if "skip2_w32" in r["combos"] else None}


def fmt(e, key="acc", d=1):
    if e is None:
        return "—"
    v = e[key] if isinstance(e, dict) else e
    return "—" if v is None else f"{v:.{d}f}"


def ci(e):
    if not e or e.get("lo") is None:
        return fmt(e)
    return f"{e['acc']:.1f} [{e['lo']:.1f}, {e['hi']:.1f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    pairs = load_pairs()
    assert len(pairs) == 10752, len(pairs)

    res, ref_hits = {}, None
    for key, name, grp, dirs, ip1m, desc in MODELS:
        e = {"name": name, "group": grp, "desc": desc, "v11_dirs": [str(d) for d in dirs], "ip1_model": ip1m}
        hits, err = v11_hits(pairs, dirs)
        if hits is not None:
            if key == "release":
                ref_hits = hits
            use_ref = ref_hits if (grp == "A" and key != "release") else None
            e["v11"] = v11_table(pairs, hits, a.boot, rng, use_ref)
            # 검사: 단일 실행이면 summary.json 전체값과 같아야 한다
            if len(dirs) == 1:
                s = json.load(open(dirs[0] / "summary.json"))["surprise"]["overall"]["block_pairwise"] * 100
                e["v11_check"] = {"summary_json": s, "recomputed": e["v11"]["overall"]["acc"],
                                  "ok": abs(s - e["v11"]["overall"]["acc"]) < 1e-6}
            else:
                old = json.load(open(ZR / "release_vith/eval/v11_split_test_from_existing_runs.json"))["overall"] * 100
                e["v11_check"] = {"v11_split_test_from_existing_runs": old, "recomputed": e["v11"]["overall"]["acc"],
                                  "ok": abs(old - e["v11"]["overall"]["acc"]) < 1e-6}
            e["v11_hits"] = hits.tolist()
        else:
            e["v11_status"] = err or "미실행"
        e["ip1"] = ip1_entry(ip1m, a.boot, rng)
        e["ip1_legacy"] = legacy_ip1(key)
        res[key] = e
        v = e.get("v11", {}).get("overall"); i = e["ip1"]
        print(f"{name:26s} v11 {fmt(v):>6s}  IP1 {fmt(i, 'best'):>6s} ({i['best_combo'] if i else '-'})"
              f"  check={e.get('v11_check', {}).get('ok')}")

    json.dump({"pairs": [{k: p[k] for k in ('block', 'pair', 'cond', 'viol', 'role', 'k')} for p in pairs],
               "models": res}, open(out / "scores.json", "w"))
    write_md(res, out)
    print("->", out / "SCORES.md")


def write_md(res, out):
    L = ["# 점수표 — v11_split_test · IntPhys1 (자동 생성, 손으로 고치지 않는다)", "",
         "`python z_research/scripts/analysis/training_effects_scores.py` 가 산출물에서 다시 계산해 쓴다.",
         "v11 = `surprise_c16t32` × `v11_split_test` (10,752 matched pair, chance 50, 95 % CI = block bootstrap 2,000). "
         "IntPhys1 = Garrido A.8 (창 16·32 합쳐 Filtered · property macro 최고 칸, CI = 쌍 bootstrap). "
         "`*` = 중간 체크포인트 (lr 미감소 또는 학습 중).", ""]
    L += ["## 1. 전체", "",
          "| 모델 | 그룹 | v11 overall [95% CI] | Δ vs 릴리즈 | v11 6조건 | IntPhys1 A.8 [95% CI] | 칸 | O1 · O2 · O3 | skip2_w32 | skip2_w16 |",
          "|---|---|---|---|---:|---|---|---|---:|---:|"]
    for key, e in res.items():
        v = e.get("v11", {}); o = v.get("overall"); i = e["ip1"]
        d = f"{o['delta']:+.1f} [{o['dlo']:+.1f}, {o['dhi']:+.1f}]" if o and o.get("delta") is not None and o.get("dlo") is not None else ("—" if key != "release" else "기준")
        pg = " · ".join(f"{i['best_per_group'][g]:.1f}" for g in sorted(i["best_per_group"])) if i else "—"
        c32 = f"{i['cells']['skip2_w32']['macro']:.1f}" if i and "skip2_w32" in i["cells"] else "—"
        c16 = f"{i['cells']['skip2_w16']['macro']:.1f}" if i and "skip2_w16" in i["cells"] else "—"
        ipv = (f"{i['best']:.2f} [{i['lo']:.1f}, {i['hi']:.1f}]" if i and i.get("lo") is not None else fmt(i, "best", 2)) if i else "—"
        L.append(f"| {e['name']} | {e['group']} | {ci(o) if o else e.get('v11_status', '—')} | {d} | {fmt(v.get('six_cond'))} | "
                 f"{ipv} | {i['best_combo'] if i else '—'} | {pg} | {c32} | {c16} |")
    L += ["", "그룹 A = 릴리즈 ViT-H encoder 고정 + predictor 만 다름 (Δ 는 같은 쌍 위의 차이, block bootstrap). "
          "그룹 B = encoder·predictor 전부 새로 (jongseo 사전학습).", ""]
    # 2. 가림 타이밍 × 운동
    L += ["## 2. v11 — 가림 타이밍 × 운동 (세 위반 합침)", "",
          "| 모델 | " + " | ".join(f"{t} {m}" for t in TIMING for m in ("static", "flat", "ramp")) + " |",
          "|---|" + "---:|" * 12]
    for key, e in res.items():
        v = e.get("v11")
        if not v:
            continue
        cells = []
        for t, cs in TIMING.items():
            for m, c in zip(("static", "flat", "ramp"), cs):
                cells.append(fmt(v.get(f"cond={c}")))
        L.append(f"| {e['name']} | " + " | ".join(cells) + " |")
    # 3. 위반 × 방향 (vanish)
    L += ["", "## 3. vanish 방향별 — 물체→빈 (`imp_vanish`) / 빈→물체 (`imp_appear`)", "",
          "CLAUDE.md §1-7: scalar surprise 의 외형 편향을 숨기지 않기 위해 방향을 항상 병기한다.", "",
          "| 모델 | " + " | ".join(f"{t} {m}" for t in ("visible", "late", "mid", "early") for m in ("static", "flat", "ramp")) + " |",
          "|---|" + "---:|" * 12]
    for key, e in res.items():
        v = e.get("v11")
        if not v:
            continue
        cells = []
        for t in ("visible", "late", "mid", "early"):
            for m in ("static", "flat", "ramp"):
                a1 = v.get(f"timing={t}|motion={m}|role=imp_vanish"); a2 = v.get(f"timing={t}|motion={m}|role=imp_appear")
                cells.append(f"{fmt(a1, d=0)} / {fmt(a2, d=0)}")
        L.append(f"| {e['name']} | " + " | ".join(cells) + " |")
    # 4. 위반 종류 × 타이밍
    L += ["", "## 4. 위반 종류 × 가림 타이밍 (세 운동 합침은 아래 json)", "",
          "| 모델 | " + " | ".join(f"{v} {t}" for v in ("vanish", "shape", "color") for t in ("visible", "early", "mid", "late")) + " |",
          "|---|" + "---:|" * 12]
    for key, e in res.items():
        v = e.get("v11")
        if not v:
            continue
        cells = []
        for vv in ("vanish", "shape", "color"):
            for t in ("visible", "early", "mid", "late"):
                hs = [v.get(f"timing={t}|motion={m}|viol={vv}") for m in ("static", "flat", "ramp")]
                hs = [h for h in hs if h]
                cells.append(f"{np.mean([h['acc'] for h in hs]):.1f}" if hs else "—")
        L.append(f"| {e['name']} | " + " | ".join(cells) + " |")
    # 5. IntPhys1 칸 전부
    L += ["", "## 5. IntPhys1 — A.8 칸별 (Filtered, property macro)", ""]
    combos = sorted({c for e in res.values() if e["ip1"] for c in e["ip1"]["cells"]})
    L += ["| 모델 | " + " | ".join(combos) + " | 옛 값 (창 한 실행) |", "|---|" + "---:|" * (len(combos) + 1)]
    for key, e in res.items():
        i = e["ip1"]
        if not i:
            continue
        lg = e.get("ip1_legacy")
        L.append(f"| {e['name']} | " + " | ".join(fmt(i["cells"].get(c), "macro") for c in combos)
                 + f" | {fmt(lg, 'best', 2) + ' (' + lg['best_combo'] + ')' if lg else '—'} |")
    L += ["", "## 검사", ""]
    for key, e in res.items():
        if "v11_check" in e:
            L.append(f"- {e['name']}: {e['v11_check']}")
    (out / "SCORES.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
