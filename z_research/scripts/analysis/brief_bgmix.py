#!/usr/bin/env python3
"""첫 예측 (걸음 0) 의 배경이 어느 시점 표현과 닮았나 (2026-10-17, 100 샘플).

  python z_research/scripts/analysis/brief_bgmix.py --device cuda

brief_motion.py 의 샘플 · 로더 · 모델 정의를 그대로 쓴다. 물체 자리 3×3 을 뺀 배경 토큰에서, 걸음 0 예측과
(a) 실제 다음 블록 (b) 마지막 문맥 블록 (c) 마지막 2 · 4 블록 평균 (d) 문맥 8 블록 평균 의 L1 거리 (late / visible).
⚠️ 평균 벡터는 원래 "가운데" 라 흐린 예측과 가까워지기 쉽다 — 모델 사이 상대 비교로 읽을 것."""
import os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "brief_motion.py")).read()
src = src[: src.index("models = [Model(r) for r in a.runs]")]
__file__ = os.path.join(HERE, "brief_motion.py")
exec(src)
models = [Model(r) for r in a.runs]
out = {m.name: {k: [] for k in ("next", "last", "mean2", "mean4", "meanall")} for m in models}; occl = []
for i, r in enumerate(pick):
    x, posf, hid = load(r); hb = blocks(x)[0]; occl.append(r["occlusion"])
    R = set(region(posf[C])); bg = torch.tensor([t for t in range(S) if t not in R])
    H = hb.view(16, S, -1)
    refs = {"next": H[C], "last": H[C - 1], "mean2": H[C - 2:C].mean(0), "mean4": H[C - 4:C].mean(0), "meanall": H[:C].mean(0)}
    for m in models:
        _, _, p = m.go(x, hb.unsqueeze(0)); p0 = p[:S]
        for k, v in refs.items():
            out[m.name][k].append(float((p0[bg] - v[bg]).abs().mean()))
occl = np.array(occl)
print("첫 예측 배경과의 L1 거리 (late / visible)")
for m in models:
    print(" ", m.name)
    for k, v in out[m.name].items():
        v = np.array(v); print(f"    {k:8s} {v[occl == 'late'].mean():.3f} / {v[occl == 'visible'].mean():.3f}")
