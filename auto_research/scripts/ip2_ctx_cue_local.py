#!/usr/bin/env python3
"""[T] IntPhys 2 짝 단서의 국소 대비 (학습 없음, CPU, 2026-10-22 KST): 짝 차이가 큰 칸 (픽셀 상위 10 %) 에서 픽셀 · 블록 토큰 · 시간 평균 토큰의 짝 사이 / 짝 안 차이.
  OMP_NUM_THREADS=8 /venv/vjepa/bin/python auto_research/scripts/ip2_ctx_cue_local.py   → results/ip2_ctx_cue_local.json"""
import numpy as np, json, collections
C='/workspace/auto_research_cache/intphys2/Main/'
meta=json.load(open(C+'meta.json')); P=np.load(C+'pix.npy',mmap_mode='r'); H=np.load(C+'h.npy',mmap_mode='r')
sc=collections.defaultdict(dict)
for r in meta: sc[r['scene_index']][r['type']]=r['row']
NB=6; out=[]
for s,v in sc.items():
    a,b,c,d=v['1_Possible'],v['1_Impossible'],v['2_Possible'],v['2_Impossible']
    pix=lambda i: np.asarray(P[i,:2*NB],np.float32).reshape(2*NB,16,16,16,16,3).mean((0,2,4,5)).reshape(-1) if False else np.asarray(P[i,:2*NB],np.float32)
    pa,pb,pc=pix(a),pix(b),pix(c)
    cell=lambda x: np.abs(x).mean((0,3)).reshape(16,16,16,16).mean((1,3)).reshape(-1)
    px_pair=cell(pa-pc); px_in=cell(pa-pb)
    tok=lambda i: np.asarray(H[i,:NB],np.float32)
    ta,tb,tc=tok(a),tok(b),tok(c)
    tp=np.abs(ta-tc).mean((0,2)); tw=np.abs(ta-tb).mean((0,2))
    mp=np.abs(ta.mean(0)-tc.mean(0)).mean(-1); mw=np.abs(ta.mean(0)-tb.mean(0)).mean(-1)
    top=np.argsort(px_pair)[::-1][:26]; rest=np.argsort(px_pair)[:128]
    out.append(dict(cond=meta[a]['condition'],cam=meta[a]['camera'],px_pair_top=float(px_pair[top].mean()),px_in_top=float(px_in[top].mean()),
                    tok_pair_top=float(tp[top].mean()),tok_in_top=float(tw[top].mean()),tok_pair_rest=float(tp[rest].mean()),tok_in_rest=float(tw[rest].mean()),mean_pair_top=float(mp[top].mean()),mean_in_top=float(mw[top].mean()),mean_pair_all=float(mp.mean()),mean_in_all=float(mw.mean()),
                    px_in_all=float(px_in.mean()),px_pair_all=float(px_pair.mean())))
def show(rs,name):
    g=lambda k: np.mean([r[k] for r in rs])
    print(f"{name:14s} n {len(rs):3d} | 픽셀 짝 안 전체 {g('px_in_all'):.2f} · 짝 사이 전체 {g('px_pair_all'):.2f} | 짝 차이 큰 칸 (상위 10 %): 픽셀 짝 사이 {g('px_pair_top'):.1f} / 짝 안 {g('px_in_top'):.1f} · 토큰 짝 사이 {g('tok_pair_top'):.3f} / 짝 안 {g('tok_in_top'):.3f} (비 {g('tok_pair_top')/g('tok_in_top'):.2f}) | 시간 평균 토큰: 상위 칸 짝 사이 {g('mean_pair_top'):.3f} / 짝 안 {g('mean_in_top'):.3f} (비 {g('mean_pair_top')/g('mean_in_top'):.2f}) · 전체 {g('mean_pair_all'):.3f} / {g('mean_in_all'):.3f}")
show(out,'all')
for k in ('continuity','immutability','permanence','solidity'): show([r for r in out if r['cond']==k],k)
for k in ('Fixed','Moving'): show([r for r in out if r['cam']==k],k)
json.dump(out,open('/workspace/world_model/vjepa2/auto_research/results/ip2_ctx_cue_local.json','w'))
