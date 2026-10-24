#!/usr/bin/env python3
"""[T] IntPhys 2 짝 알아맞히기 (학습 없음, CPU, 2026-10-22 KST): 불가능 영상의 문맥 (블록 0..5) 이 같은 짝 가능 영상에 다른 짝보다 가까운가 — 블록 토큰 · 시간 평균 토큰 · 픽셀.
  OMP_NUM_THREADS=8 /venv/vjepa/bin/python auto_research/scripts/ip2_pair_ident.py   → results/ip2_pair_ident.json"""
# 짝 알아맞히기: 불가능 영상의 문맥이 같은 짝 가능 영상에 더 가까운가 (다른 짝 가능 영상보다). 칸 전체, 문맥 블록 0..5.
import numpy as np, json, collections
C='/workspace/auto_research_cache/intphys2/Main/'
meta=json.load(open(C+'meta.json')); P=np.load(C+'pix.npy',mmap_mode='r'); H=np.load(C+'h.npy',mmap_mode='r')
sc=collections.defaultdict(dict)
for r in meta: sc[r['scene_index']][r['type']]=r['row']
NB=6; res=collections.defaultdict(list)
for s,v in sc.items():
    cond=meta[v['1_Possible']]['condition']; cam=meta[v['1_Possible']]['camera']
    T={t:np.asarray(H[v[t],:NB],np.float32) for t in v}; X={t:np.asarray(P[v[t],:2*NB],np.float32) for t in v}
    for k,o in (('1','2'),('2','1')):
        imp=f'{k}_Impossible'; same=f'{k}_Possible'; other=f'{o}_Possible'
        for name,F in (('tok_block',lambda a,b: np.abs(T[a]-T[b]).mean()),('tok_tmean',lambda a,b: np.abs(T[a].mean(0)-T[b].mean(0)).mean()),('pix',lambda a,b: np.abs(X[a]-X[b]).mean())):
            ok=F(imp,same)<F(imp,other)
            for key in ('all',cond,cam): res[(name,key)].append(ok)
for name in ('tok_block','tok_tmean','pix'):
    print(name, ' · '.join(f"{key} {np.mean(res[(name,key)])*100:.1f}" for key in ('all','continuity','immutability','permanence','solidity','Fixed','Moving')))
json.dump({f"{n}|{k}":float(np.mean(v)) for (n,k),v in res.items()},open('/workspace/world_model/vjepa2/auto_research/results/ip2_pair_ident.json','w'),indent=1)
