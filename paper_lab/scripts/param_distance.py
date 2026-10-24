"""파일럿 팔 사이 predictor 매개변수 거리 (진행자 1, 2026-10-22) — '개입이 학습을 얼마나 움직였나' 의 크기 검사.

    python paper_lab/scripts/param_distance.py                       # 기본 팔 · 쌍
    python paper_lab/scripts/param_distance.py A2m:AX A2g:AX A2m:A0  # 쌍을 인자로 (팔 이름 = pl_pilot_<팔>/e10.pt)

L2 거리 = sqrt(sum over predictor tensors of ||θ_a − θ_b||²), CPU, 체크포인트는 읽기만 한다.
출발점 = 기준 블록 AR (z_training/vast/ar_state/latest.pt), 팔 = paper_lab/_scratch/runs/pl_pilot_<arm>/e10.pt.
"""
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARMS = ("A0", "AX", "A1", "A1c", "A2c", "A4")
PAIRS = (("AX", "A0"), ("A2c", "AX"), ("A4", "AX"), ("A2c", "A0"), ("A4", "A0"), ("A1", "AX"), ("A1c", "AX"), ("A1c", "A1"))


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)["predictor"]


def dist(x, y):
    return sum(((x[k].float() - y[k].float()) ** 2).sum() for k in x if k in y and x[k].shape == y[k].shape).sqrt().item()


def main():
    pairs = [tuple(x.split(":")) for x in sys.argv[1:]] or list(PAIRS)
    arms = list(dict.fromkeys([a for p in pairs for a in p])) if sys.argv[1:] else list(ARMS)
    S = {a: load(os.path.join(ROOT, f"paper_lab/_scratch/runs/pl_pilot_{a}/e10.pt")) for a in arms
         if os.path.exists(os.path.join(ROOT, f"paper_lab/_scratch/runs/pl_pilot_{a}/e10.pt"))}
    start = load(os.path.join(ROOT, "z_training/vast/ar_state/latest.pt"))
    for a in S:
        print(f"{a:>4} − 출발점 {dist(S[a], start):.3f}")
    for a, b in pairs:
        if a in S and b in S:
            print(f"{a:>4} − {b:<4} {dist(S[a], S[b]):.3f}")


if __name__ == "__main__":
    main()
