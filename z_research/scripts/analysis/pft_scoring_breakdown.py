#!/usr/bin/env python3
"""post-FT (v11_postft e10) vs 릴리즈 predictor 의 v11 held-out 채점을 케이스별로 쪼갠다 — per_video_surprise 에서 matched pair 를 직접 다시 계산.

held-out = v11_full 의 block 단위 test 절반 (condition × violation × k 층화, seed 0; 같은 조건·장면·물체 종류, 다른 block = 다른 문맥). 외형 holdout 은 없다.
쪼개는 축: 조건 × 위반 / 조건 × k / 쌍 방향 (A = pos_a vs imp_ab, B = pos_b vs imp_ba) / shape·color 블록의 물체별 recall (순서 편향).
정답 = surprise(imp) > surprise(pos), tie 0.5. 전체가 release 75.83 / post-FT 91.35 와 맞는지 검증한다.
출력: z_research/predictor_training/predictor_IntPhysGenV11_PFT/{exp_results/scoring_breakdown.json, Archive/SCORING_2026-09-16.md}

  /data/hyuntak/anaconda3/envs/vjepa2/bin/python z_research/scripts/analysis/pft_scoring_breakdown.py
"""
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2"); SET = ROOT / "z_research/predictor_training/predictor_IntPhysGenV11_PFT"
RUNS = {"release": ROOT / "z_research/IntPhysGenV11_occlusion_timing_ablation/exp_results/surprise_c16t32__v11_full_vith/per_block.json",
        "post-FT": ROOT / "z_training/runs/v11_postft/eval/surprise_c16t32__v11_split_test_e10/per_block.json",
        "IP1-FT": ROOT / "z_training/runs/intphys1_postft/eval/surprise_c16t32__v11_split_test_e40/per_block.json"}
rows = {r["video_id"]: r for r in csv.DictReader(open(ROOT / "data_csv/intphysgen_v11_split/index_test.csv"))}
blocks = defaultdict(dict)
for v, r in rows.items(): blocks[r["block_id"]][r["variant"]] = r
PAIRS = {"A": ("pos_a", "imp_ab"), "B": ("pos_b", "imp_ba")}


def pair_scores(sv):
    out = []   # (block_id, dir, correct, meta row of pos clip, imp row)
    for b, d in blocks.items():
        for k, (pv, iv) in PAIRS.items():
            if pv not in d or iv not in d: continue
            sp, si = sv[d[pv]["video_id"]], sv[d[iv]["video_id"]]
            out.append((b, k, 1.0 if si > sp else (0.5 if si == sp else 0.0), d[pv], d[iv]))
    return out


def table(scores, key):
    acc = defaultdict(list)
    for b, k, c, r, ri in scores: acc[key(k, r, ri)].append(c)
    return {kk: (float(np.mean(v)) * 100, len(v)) for kk, v in acc.items()}


res, S = {}, {}
for name, p in RUNS.items():
    sv = json.load(open(p))["per_video_surprise"]; S[name] = pair_scores(sv)
    res[name] = {"overall": float(np.mean([c for *_, c, _, _ in [(b, k, c, r, ri) for b, k, c, r, ri in S[name]]]) * 100), "n_pair": len(S[name])}
print({k: (round(v["overall"], 2), v["n_pair"]) for k, v in res.items()})
timing = lambda r: "late" if r["condition"].endswith(("occlusion", "occlusion_flat")) else ("early" if "early" in r["condition"] else ("mid" if "mid" in r["condition"] else "visible"))
motion = lambda r: "static" if r["condition"].startswith("static") else ("flat" if "flat" in r["condition"] else "ramp")
keys = {"cond_x_violation": lambda k, r, ri: (r["condition"], r["violation_type"]),
        "motion_timing_x_k": lambda k, r, ri: (motion(r), timing(r), r["sym_k"]),
        "cond_x_violation_x_dir": lambda k, r, ri: (r["condition"], r["violation_type"], k),
        "shape_recall_by_pre_late_ramp": lambda k, r, ri: (r["shape_pre"] if (r["violation_type"] == "shape" and r["condition"] == "moving_occlusion") else None,),
        "shape_recall_by_pre_visible_ramp": lambda k, r, ri: (r["shape_pre"] if (r["violation_type"] == "shape" and r["condition"] == "moving_visible") else None,),
        "color_recall_by_pre_late_ramp": lambda k, r, ri: (r["color_pre"] if (r["violation_type"] == "color" and r["condition"] == "moving_occlusion") else None,)}
T = {name: {kn: table(S[name], kf) for kn, kf in keys.items()} for name in RUNS}
md = ["# post-FT vs 릴리즈 — v11 held-out 채점 케이스별 (2026-09-16)", "",
      "held-out = v11_full 의 block 단위 test 절반 (5,376 block / 10,752 matched pair; condition × violation × k 층화, seed 0). **같은 조건·장면·물체 종류, 다른 block(= 다른 문맥·다른 pre/post 조합)** 이라 in-distribution 일반화다. 외형(shape/color) holdout 은 안 했다.",
      "`per_video_surprise` 에서 matched pair 를 다시 계산 (정답 = surprise(imp) > surprise(pos), tie 0.5). 전체: " + " · ".join(f"{k} **{v['overall']:.2f}** (n {v['n_pair']})" for k, v in res.items()) + " — 공식 summary 75.83 / 91.35 / 78.53 과 일치.", "",
      "IP1-FT = 같은 레시피를 IntPhys1 train 으로 학습한 대조군 (도메인 밖 학습).", ""]


def emit(title, kn, hdr, fmt):
    md.extend([f"## {title}", "", "| " + " | ".join(hdr) + " | n | release | post-FT | Δ | IP1-FT | Δ |", "|" + "---|" * (len(hdr) + 6)])
    ks = sorted(k for k in T["release"][kn] if k[0] is not None)
    for k in ks:
        a, n = T["release"][kn][k]; b = T["post-FT"][kn][k][0]; c = T["IP1-FT"][kn][k][0]
        md.append("| " + " | ".join(fmt(k)) + f" | {n} | {a:.1f} | **{b:.1f}** | {b-a:+.1f} | {c:.1f} | {c-a:+.1f} |")
    md.append("")


emit("1. 조건 × 위반", "cond_x_violation", ["condition", "violation"], lambda k: list(k))
emit("2. 운동 × 타이밍 × k", "motion_timing_x_k", ["motion", "timing", "k"], lambda k: list(k))
emit("3. 조건 × 위반 × 쌍 방향 (A = pos_a vs imp_ab, B = pos_b vs imp_ba)", "cond_x_violation_x_dir", ["condition", "violation", "dir"], lambda k: list(k))
emit("4. shape 블록, 등가속+가림(late): pre 모양별 recall — 릴리즈의 순서 편향 (cylinder 90 > … > torus 3) 이 남는가", "shape_recall_by_pre_late_ramp", ["shape_pre"], lambda k: list(k))
emit("5. shape 블록, 등가속 비가림: pre 모양별 recall", "shape_recall_by_pre_visible_ramp", ["shape_pre"], lambda k: list(k))
emit("6. color 블록, 등가속+가림(late): pre 색별 recall", "color_recall_by_pre_late_ramp", ["color_pre"], lambda k: list(k))
md += ["## 재현", "", "```", "/data/hyuntak/anaconda3/envs/vjepa2/bin/python z_research/scripts/analysis/pft_scoring_breakdown.py", "```", ""]
(SET / "exp_results/scoring_breakdown.json").write_text(json.dumps({n: {kn: {"|".join(map(str, k)): v for k, v in t.items()} for kn, t in tt.items()} for n, tt in T.items()}, indent=1))
(SET / "Archive/SCORING_2026-09-16.md").write_text("\n".join(md)); print("→", SET / "Archive/SCORING_2026-09-16.md")
