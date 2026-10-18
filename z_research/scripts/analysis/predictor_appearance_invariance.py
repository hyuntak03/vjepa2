#!/usr/bin/env python3
"""외형은 예측 진행을 바꾸나 — 모양 · 색별 진행 / 참 진행 vs 무작위 라벨 귀무 (2026-10-13, DESIGN_CHOICES_2026-10-13 §2-1).
입력 = predictor_bottleneck 산출물 z_research/VOBench_v1/bottleneck/<name>/shard*.npz.  python predictor_appearance_invariance.py"""
import sys, glob
import numpy as np
from pathlib import Path; R=Path(__file__).resolve().parents[3]; import os; os.chdir(R); sys.path.insert(0,str(R/"z_research/scripts/analysis")); sys.path.insert(0,str(R))
import predictor_bottleneck as PB
rows=PB.select(1600,1200)[1]
for name in ("release","ar40","ar_state40"):
    Z={}
    for f in sorted(glob.glob(f"z_research/VOBench_v1/bottleneck/{name}/shard*.npz")):
        z=np.load(f,allow_pickle=True)
        for k in ["row","part","truth"]+[f"{c}_xy{k}" for c in ("roll","tf") for k in range(1,9)]:
            if k in z.files: Z.setdefault(k,[]).append(z[k])
    Z={k:np.concatenate(v) for k,v in Z.items()}
    T=Z["truth"]; v=T[:,7]-T[:,6]; st=np.linalg.norm(v,axis=1); vh=v/st[:,None]
    sh=np.array([rows[int(i)]["shape_pre"] for i in Z["row"]]); co=np.array([rows[int(i)]["color_pre"] for i in Z["row"]])
    def gain(c):   # 걸음 k 의 진행 / 참 진행 (블록 C-1 기준)
        out=[]
        for k in range(1,9):
            s=((Z[f"{c}_xy{k}"]-T[:,7])*vh).sum(-1); ts=((T[:,7+k]-T[:,7])*vh).sum(-1); out.append(s/ts)
        return np.stack(out,1)
    for c in [c for c in ("tf","roll") if "%s_xy1"%c in Z]:
        G=gain(c); m=st>9
        print(f"== {name} {c}  (n {m.sum()}) 진행/참 진행, 걸음 k=2,4,6,8 — 모양별 중앙값")
        rng=np.random.default_rng(0); 
        for k in (1,3,5,7):
            per={s:np.median(G[m&(sh==s),k]) for s in sorted(set(sh))}
            # 무작위 라벨 귀무: 같은 크기 군의 중앙값 폭
            null=[]; lab=sh[m].copy()
            for _ in range(200):
                rng.shuffle(lab); g=G[m,k]; null.append(np.ptp([np.median(g[lab==s]) for s in sorted(set(sh))]))
            print(f"   k{k+1}: "+" ".join(f"{s[:4]} {v:.2f}" for s,v in per.items())+f" | 폭 {np.ptp(list(per.values())):.2f} (귀무 95% {np.quantile(null,.95):.2f})")
        percol={s:np.median(G[m&(co==s),5]) for s in sorted(set(co))}
        print(f"   색별 k6: "+" ".join(f"{s[:3]} {v:.2f}" for s,v in percol.items()))
