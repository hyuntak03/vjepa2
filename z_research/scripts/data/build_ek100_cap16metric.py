#!/usr/bin/env python3
"""EK100 cap16metric 부분집합 재생성 (2026-10-21, vast). main 의 GROUP4_CAP16METRIC 문서의 CSV 는 git 에 없어 (data_csv 는 gitignore) 정의로 다시 만든다.

정의 (실물 크기로 역산해 확인 — 원래 CSV 와 행 수가 정확히 같다):
  지표 클래스 = val 과 full train 에 둘 다 있는 action (verb_noun) 1,114 개
  train = 지표 action 마다 최대 16 구간 (16 개 이하면 전부)  → 11,470 행   (원래 11,470 행 / 470 편)
  val   = 지표 action 인 val 구간 전부                        →  9,296 행   (원래 9,296 행 / 138 편)
⚠️ 16 개 넘는 클래스에서 고른 구간 (무작위, seed 0) 은 원래 CSV 와 **같다는 보장이 없다** (원래 난수 순서를 모른다). 분포 (클래스별 개수) 는 같다.

  /venv/vjepa/bin/python z_research/scripts/data/build_ek100_cap16metric.py
  → data_csv/ek100/EPIC_100_{train,validation}_cap16metric_seed0.csv
"""
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[3]
ANN = Path("/workspace/dataset/benchmarks/repos/epic-kitchens-100-annotations")
OUT = REPO / "data_csv/ek100"; OUT.mkdir(parents=True, exist_ok=True)
tr = pd.read_csv(ANN / "EPIC_100_train.csv"); va = pd.read_csv(ANN / "EPIC_100_validation.csv")
act = lambda d: d.verb_class.astype(str) + "_" + d.noun_class.astype(str)
need = set(act(va)) & set(act(tr))
rng = np.random.default_rng(0)
a_tr = act(tr)
parts = [g.sample(n=min(16, len(g)), random_state=int(rng.integers(1 << 30))) for a, g in tr.groupby(a_tr) if a in need]
sub = pd.concat(parts).sort_index()
vsub = va[act(va).isin(need)]
sub.to_csv(OUT / "EPIC_100_train_cap16metric_seed0.csv", index=False)
vsub.to_csv(OUT / "EPIC_100_validation_cap16metric_seed0.csv", index=False)
print(f"train {len(sub)} 행 / {sub.video_id.nunique()} 편 · val {len(vsub)} 행 / {vsub.video_id.nunique()} 편 · 지표 action {len(need)}")
