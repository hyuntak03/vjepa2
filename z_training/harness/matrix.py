#!/usr/bin/env python3
"""학습 데이터 × 채점 bench 전이 행렬 — 산출물에서 모은다 (2026-09-19). 채우는 쪽은 z_training/eval_matrix.sh.

열마다 그 bench 의 확립된 protocol 값을 읽는다:
  v11 test       surprise_c16t32 matched pair (10,752 쌍)  summary.json surprise.overall.block_pairwise
  IntPhys1 dev   intphys1_sliding, **skip2_w32/avg** (CLAUDE.md 확립 88.89 와 같은 칸) — 괄호 = 6 칸 중 최고 (descriptive)
  IntPhys2 test  analysis/intphys2, Main 장면 split test 126 장면 (252 쌍), 논문 Table 2 방식 열별 최고 (C 11 × avg/max) 의 Overall
  Predictor_v1   holdout 1,000 clip 예측 L1 (낮을수록 좋음; 가능 영상뿐이라 짝 정확도 없음)
출력: z_training/runs/_matrix/MATRIX.{md,json}
  python z_training/harness/matrix.py
"""
from __future__ import annotations
import json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "z_training/runs"; OUT = R / "_matrix"
ROWS = [("release", None, None, "릴리즈 (학습 안 함)"), ("v11_postft", "e10", "v11", "v11 (v11_split_train)"),
        ("predictor_v1_postft", "e15", "pv1", "Predictor_v1 (index_train)"), ("intphys1_postft", "e40", "ip1", "IntPhys1 (train 장면)"),
        ("intphys2_postft", "e80", "ip2", "IntPhys2 (Main split train)")]
COLS = [("v11", "v11 test"), ("pv1", "Predictor_v1 holdout L1 ↓"), ("ip1", "IntPhys1 dev"), ("ip2", "IntPhys2 test")]


def jl(p):
    try: return json.load(open(p))
    except Exception: return None


def cell(run, ep, col):
    if col == "v11":
        if run == "release":
            j = jl(R / "release_vith/eval/v11_split_test_from_existing_runs.json"); return (100 * j["overall"], None, "release_vith/eval/v11_split_test_from_existing_runs.json") if j else None
        p = R / run / f"eval/surprise_c16t32__v11_split_test_{ep}/summary.json"; j = jl(p)
        return (100 * j["surprise"]["overall"]["block_pairwise"], None, str(p.relative_to(ROOT))) if j else None
    if col == "ip1":
        p = (ROOT / "z_exp/world_model_analysis/results/intphys1_vith/summary.json") if run == "release" else R / run / f"eval/intphys1_sliding__intphys1_dev_{ep}/summary.json"
        j = jl(p)
        if not j: return None
        s = j["surprise"]; cells = {k: v["overall"]["block_pairwise"] for k, v in s.items() if isinstance(v, dict) and "overall" in v and "block_pairwise" in v["overall"]}
        b = max(cells, key=cells.get); return (100 * cells["skip2_w32/avg"], f"{100 * cells[b]:.2f} {b}", str(p.relative_to(ROOT)))
    if col == "ip2":
        if run == "release":
            p = ROOT / "z_exp/intphys2/vjepa2_vith_intphys2_main_repro/column_best_testsplit.json"
        elif run == "intphys2_postft":
            p = ROOT / "z_exp/intphys2/vjepa2_vith_intphys2_maintest_postft_e80/column_best.json"
        else:
            p = ROOT / f"z_exp/intphys2/vjepa2_vith_intphys2_maintest_{run}_{ep}/column_best.json"
        j = jl(p)
        return (j["columns"]["Overall"]["best_acc"], j["columns"]["Overall"]["best"], str(p.relative_to(ROOT))) if j else None
    if col == "pv1":
        j = jl(OUT / "predictor_v1_holdout_l1.json")
        key = "release" if run == "release" else f"{run} {ep}"
        if not j or key not in j["results"]: return None
        return (j["results"][key]["mean_l1"], None, "z_training/runs/_matrix/predictor_v1_holdout_l1.json")


def main():
    M = {}
    for run, ep, dom, lab in ROWS:
        M[run] = {c: cell(run, ep, c) for c, _ in COLS}
    base = M["release"]
    L = ["# 학습 데이터 × 채점 bench 전이 행렬 (ViT-H, encoder 동결, 릴리즈 predictor 에서 post-FT) — 2026-09-19\n",
         "칸 = 채점값 (Δ = 릴리즈 대비). **굵게 = 학습한 도메인 (대각)**. 정확도 열은 chance 50, L1 열은 낮을수록 좋음.\n",
         "| 학습 데이터 (ckpt) | " + " | ".join(n for _, n in COLS) + " |", "|---|" + "---:|" * len(COLS)]
    for run, ep, dom, lab in ROWS:
        row = []
        for c, _ in COLS:
            v = M[run][c]
            if v is None: row.append("—"); continue
            fmt = f"{v[0]:.4f}" if c == "pv1" else f"{v[0]:.2f}"
            if run != "release" and base[c]:
                d = v[0] - base[c][0]; fmt += f" ({d:+.4f})" if c == "pv1" else f" ({d:+.2f})"
            row.append(f"**{fmt}**" if c == dom else fmt)
        L.append(f"| {lab}{'' if not ep else f' `{ep}`'} | " + " | ".join(row) + " |")
    L += ["\n## 열 protocol\n",
          "- **v11 test**: `surprise_c16t32`, matched pair, `v11_split_test` 10,752 쌍 (학습 안 한 block 절반). 릴리즈 값은 기존 채점 per_block 에서 추출.",
          "- **Predictor_v1 holdout L1**: 가능 영상뿐 (짝 없음) → 정확도 대신 surprise_c16t32 계산의 평균 L1, holdout 1,000 clip (arm 층화, seed 0). **다른 단위의 열**.",
          "- **IntPhys1 dev**: `intphys1_sliding` (Garrido 공식) `skip2_w32/avg` — 확립 수치 88.89 와 같은 칸. 180 쌍.",
          "- **IntPhys2 test**: `analysis/intphys2` growing context, M 48, 6 fps, fp32 + bf16 autocast, 논문 Table 2 방식 열별 최고 (C 11 개 × avg/max) 의 Overall. "
          "Main 장면 split test 126 장면 252 쌍 — ⚠️ 논문 Table 2 (Main 전체) 와 비교 불가.",
          "\n## 칸별 출처 · 보조 값\n", "| 행 | 열 | 값 | 보조 (IntPhys1: 6 칸 중 최고 / IntPhys2: 최고 설정) | 산출물 |", "|---|---|---:|---|---|"]
    for run, ep, dom, lab in ROWS:
        for c, n in COLS:
            v = M[run][c]
            if v: L.append(f"| {run} {ep or ''} | {n} | {v[0]:.4f} | {v[1] or ''} | `{v[2]}` |")
    L += ["\n## 단서\n",
          "- 한 번씩만 학습했다 (seed 0) — 학습 분산 모름. IntPhys1 180 쌍 · IntPhys2 252 쌍은 95 % CI 가 약 ±7 / ±6 pt.",
          "- 체크포인트는 각 run 의 마지막 epoch. 학습 데이터 크기가 크게 다르다 (v11 10,752 / Predictor_v1 14,250 / IntPhys1 약 3,750 / IntPhys2 254 clip).",
          "- IntPhys2 는 짝의 문맥이 픽셀 단위로 같지 않다 (첫 프레임부터 1.7–3.7/255) — v11 · IntPhys1 과 채점의 성격이 다르다.",
          "\n## 재현\n\n```\nCUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 GPUS=7 bash z_training/eval_matrix.sh\n"
          "CUDA_VISIBLE_DEVICES=7 python z_research/scripts/analysis/predictor_holdout_l1.py --n 1000\npython z_training/harness/matrix.py\n```"]
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({k: {c: (None if v is None else {"value": v[0], "aux": v[1], "src": v[2]}) for c, v in d.items()} for k, d in M.items()}, open(OUT / "MATRIX.json", "w"), indent=1)
    (OUT / "MATRIX.md").write_text("\n".join(L)); print("\n".join(L))


if __name__ == "__main__":
    main()
