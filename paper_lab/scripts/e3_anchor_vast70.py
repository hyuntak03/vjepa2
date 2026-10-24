#!/usr/bin/env python3
"""E3 (paper_lab 진행자 2, 2026-10-21) — 기준 블록 AR (vast AR-state ep70) 에 마지막 관측 바꿔치기.

기존 스크립트 z_research/scripts/analysis/predictor_anchor_test.py 를 고치지 않고 불러 쓰되, 산출 폴더만
paper_lab/_scratch/e3/anchor 로 바꾼다 (남의 폴더에 쓰지 않는다). GPU 1 장, 한 프로세스.
판독기 = z_research/VOBench_v1/readout/identity_vobprim_vast70/p (검증값: 57 지선다 87.3 %, 중심 오차 평균 1.78 칸 · encoder 판 0.41 칸).

  python paper_lab/scripts/e3_anchor_vast70.py [--bs 8] [--max-n 600]
"""
import argparse
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
spec = spec_from_file_location("pat", REPO / "z_research/scripts/analysis/predictor_anchor_test.py")
pat = module_from_spec(spec); spec.loader.exec_module(pat)
pat.OUT = REPO / "paper_lab/_scratch/e3/anchor"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--max-n", type=int, default=600)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if not a.summarize:
        pat._worker(0, 1, "vast70", "z_training/vast/ar_state/latest.pt", a.bs, a.max_n)
    pat.summarize("vast70")
