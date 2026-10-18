"""VOBench_v1 gravity 한 run 의 두 지표를 objset · law · cell 별로 (2026-10-15, 평가 행렬 gravity 칸).
  고른 미래 = 진짜 : block 안 미래 중 surprise 최소가 진짜 (plausible=1) 인 block 비율 — 행렬 주 지표 (ledge chance 50 · arc 11.1)
  쌍 정확도        : 진짜 < 불가능 하나 인 쌍 비율 (pairing cross 와 같은 쌍, 쌍 단위 합산 — summary.json block_pairwise 는 block 평균이라 다르다)
  python z_research/scripts/analysis/vobench_gravity_pick.py <run 폴더>             # 216 block 전부. per_block.json 의 per_video_surprise 를 읽는다
  python z_research/scripts/analysis/vobench_gravity_pick.py <run 폴더> --test      # gravity test 108 block 만 (data_csv/vobench_v1_gravity_split/index_test.csv — PFT 가 gravity train 을 본 뒤의 행렬 칸)
"""
import json, sys, csv
from collections import defaultdict
run=sys.argv[1]
KEEP={r['block_id'] for r in csv.DictReader(open('data_csv/vobench_v1_gravity_split/index_test.csv'))} if '--test' in sys.argv else None
S=json.load(open(f'{run}/per_block.json'))['per_video_surprise']
rows=list(csv.DictReader(open('data_csv/vobench_v1_gravity/index.csv')))
B=defaultdict(list)
for r in rows:
    if r['video_id'] in S: B[r['block_id']].append(r)
def sc(v):
    x=S[v]; return x if isinstance(x,(int,float)) else x.get('all', list(x.values())[0])
agg=defaultdict(lambda:[0,0,0,0])  # pair_correct, pairs, argmin_correct, blocks
for b,rs in B.items():
    if KEEP is not None and b not in KEEP: continue
    pos=[r for r in rs if r['plausible']=='1']; imp=[r for r in rs if r['plausible']=='0']
    assert len(pos)==1, (b,len(pos))
    p=sc(pos[0]['video_id']); ims=[sc(r['video_id']) for r in imp]
    pc=sum(p<i for i in ims); am=int(p<min(ims))
    for k in ('ALL', rs[0]['objset'], rs[0]['objset']+'/'+rs[0]['law'], rs[0]['objset']+'/'+rs[0]['law']+'/'+rs[0]['cell']):
        a=agg[k]; a[0]+=pc; a[1]+=len(ims); a[2]+=am; a[3]+=1
for k in sorted(agg, key=lambda k:(k!='ALL', k.count('/'), k)):
    a=agg[k]; print(f'{k:45s} 쌍 {100*a[0]/a[1]:5.1f} ({a[0]}/{a[1]})   고른 미래=진짜 {100*a[2]/a[3]:5.1f} ({a[2]}/{a[3]} block)')
