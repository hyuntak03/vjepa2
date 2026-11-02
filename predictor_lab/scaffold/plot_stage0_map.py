#!/usr/bin/env python3
"""scaffold 2 판 0 단계 — 물체 지도를 눈으로 보는 그림 (2026-11-01).

행 = 영상 하나 · 열 = 시간 칸 16 개 (문맥 0–7 | 미래 8–15). 칸마다 예측 물체 확률 (16×16) 을 회색조로, 참 물체 중심을 빨간 점으로.
위 묶음: VOBench 속도 sweep (학습 밖) 같은 block 을 1/3 · 2/3 · 1 배 속도로. 아래 묶음: v3 test (학습 안) 앞쪽 영상 몇 개 (참 점유는 파란 윤곽).
입력 = stage0_map.py 가 남긴 predictor_lab/scaffold/results/stage0_<tag>_examples.npz
  $P predictor_lab/scaffold/plot_stage0_map.py [tag] [vob block 번호들 ...]
출력 predictor_lab/scaffold/results/stage0_<tag>_maps.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RES = Path(__file__).resolve().parent / "results"
tag = sys.argv[1] if len(sys.argv) > 1 else "map_d4"
import os
d = np.load(os.environ.get("EX", RES / f"stage0_{tag}_examples.npz"))   # EX= 로 다른 예시 파일 (중간 체크포인트 엿보기 등)
vob_blocks = [int(x) for x in sys.argv[2:]] or [0, 7, 13]
speeds = [("S1", "1/3x"), ("T1", "2/3x"), ("orig", "1x")]
v3_n = 4
rows = [("vob", b, nm, lab) for b in vob_blocks for nm, lab in speeds] + ([("v3", i, None, None) for i in range(min(v3_n, len(d["v3_P"])))] if "v3_P" in d.files else [])
fig, axes = plt.subplots(len(rows), 16, figsize=(16 * 0.62, len(rows) * 0.68), squeeze=False)
plt.rcParams["font.family"] = "DejaVu Sans"
for r, (src, i, nm, lab) in enumerate(rows):
    if src == "vob":
        P = d[f"vob_{nm}_P"][i].astype(np.float32); cen = d[f"vob_{nm}_cen"][i][:, None]; occ = None
        name = f"VOB #{i} {lab}"
    else:
        P = d["v3_P"][i].astype(np.float32); cen = d["v3_ctr"][i]; occ = d["v3_occ"][i].astype(np.float32)
        name = f"v3 {str(d['v3_cid'][i])[-6:]}"
    for t in range(16):
        ax = axes[r, t]
        ax.imshow(P[t].reshape(16, 16), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        if occ is not None:
            ax.contour(occ[t].reshape(16, 16), levels=[0.25], colors="#2a78d6", linewidths=0.6)
        c = cen[t]
        c = c[np.isfinite(c[:, 0])]
        ax.scatter(c[:, 0], c[:, 1], s=6, c="#eb6834", linewidths=0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-0.5, 15.5); ax.set_ylim(15.5, -0.5)
        for s in ax.spines.values():
            s.set_edgecolor("#1baf7a" if t >= 8 else "#999999"); s.set_linewidth(0.8 if t >= 8 else 0.4)
        if r == 0:
            ax.set_title(f"{'c' if t < 8 else 'f'}{t}", fontsize=6)
    axes[r, 0].set_ylabel(name, fontsize=6, rotation=0, ha="right", va="center")
fig.subplots_adjust(left=0.08, right=0.995, top=0.96, bottom=0.01, wspace=0.05, hspace=0.08)
out = Path(os.environ.get("OUT", RES / f"stage0_{tag}_maps.png"))
fig.savefig(out, dpi=170)
print(out)
