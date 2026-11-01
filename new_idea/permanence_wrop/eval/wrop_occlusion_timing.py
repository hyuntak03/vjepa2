#!/usr/bin/env python3
"""WROP 가림 계열 (op1 · op2) 학습 클립에서 물체가 경계 (문맥 16 | 미래 16) 기준 언제 가려지나 (2026-11-01).
대리 지표: 프레임별 채도 높은 화소 수 (가림막 · 바닥 · 벽은 회색, 물체는 유채색). 숨음 = 클립 최대의 30 % 미만.
검증: ball_behind_box_stack_0000 의 궤적 (trajectory.npz) 으로 계산한 완전 가림 구간과 1–2 장 안에서 일치.
  python new_idea/permanence_wrop/eval/wrop_occlusion_timing.py new_idea/permanence_wrop/results/wrop_occlusion_timing_op12.json
"""
# WROP 가림 계열 학습 클립의 "물체 보임" 대리 지표: 채도 높은 화소 수 (가림막 · 바닥 · 벽은 회색, 물체는 유채색)
import csv, sys, json, random
import numpy as np, decord
from concurrent.futures import ThreadPoolExecutor
man=list(csv.DictReader(open("/workspace/world_model/vjepa2/data_csv/wrop_uni32/train_manifest.tsv"),delimiter="\t"))
by={}
for r in man:
    if r["family"].startswith("op"): by.setdefault(r["task"],[]).append(r)
random.seed(0)
rows=[r for t in sorted(by) for r in random.sample(by[t],min(20,len(by[t])))]
def colored(path):
    vr=decord.VideoReader(path,num_threads=1,width=284,height=160)
    x=vr.get_batch(list(range(32))).asnumpy().astype(np.float32)/255.
    mx=x.max(-1); mn=x.min(-1); s=(mx-mn)/(mx+1e-6)
    return ((s>0.35)&(mx>0.15)).sum(axis=(1,2)).astype(float)
with ThreadPoolExecutor(16) as ex: C=list(ex.map(lambda r: colored(r["path"]), rows))
out=[{"task":r["task"],"family":r["family"],"path":r["path"],"c":c.tolist()} for r,c in zip(rows,C)]
json.dump(out,open(sys.argv[1],"w"))
print(len(out))
