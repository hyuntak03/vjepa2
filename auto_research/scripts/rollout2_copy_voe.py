#!/usr/bin/env python3
"""RollOut_v2 ledge · wall (pos/imp 쌍: 낙하 vs 부유, 정지 vs 통과) 에서 VoE 쌍 정확도 — predictor p vs **복사 (마지막 문맥 튜블릿 LN(z) × 8)** (2026-10-07).
질문 (사용자): 복사가 높게 나오면 metric 이 위치를 전혀 안 본다는 뜻 아닌가 → 위치 **만** 다른 쌍에서 복사가 몇 점인지 본다.
surprise = mean |pred − LN(h)[미래 8 튜블릿]|, 정답 = surprise(imp) > surprise(pos). 캐시 (vll5): rollout_v2_vith (target, predictor) · rollout_v2_z16_vith (ctx_masked = LN(z) 문맥 8 튜블릿).
"""
import csv, json, numpy as np, collections
C="/local_datasets/world/world_analysis/cache"; S, T, D = 256, 8, 1280
ids=json.load(open(f"{C}/rollout_v2_vith/meta.json"))["video_ids"]; pos={v:i for i,v in enumerate(ids)}
idz=json.load(open(f"{C}/rollout_v2_z16_vith/meta.json"))["video_ids"]; assert idz==ids
H=np.load(f"{C}/rollout_v2_vith/target.npy", mmap_mode="r"); P=np.load(f"{C}/rollout_v2_vith/predictor.npy", mmap_mode="r"); Z=np.load(f"{C}/rollout_v2_z16_vith/ctx_masked.npy", mmap_mode="r")
idx=list(csv.DictReader(open("/data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/rollout_v2/index_probe.csv")))
blk=collections.defaultdict(dict)
for r in idx:
    if r["scenario"] in ("ledge","wall"): blk[r["block_id"]][r["variant"]]=r
def sur(i, kind):
    h=H[i, T*S:].astype(np.float32)
    if kind=="p": pr=P[i].astype(np.float32)
    elif kind=="copy": pr=np.repeat(Z[i, (T-1)*S:].astype(np.float32), T, axis=0)       # 마지막 문맥 튜블릿 (256 토큰) 을 8 번
    elif kind=="copy_mean": pr=np.repeat(Z[i].astype(np.float32).reshape(T,S,D).mean(0), T, axis=0)
    return float(np.abs(pr-h).mean())
out={}
for sc in ("ledge","wall"):
    acc=collections.defaultdict(list); n=0
    for b,v in blk.items():
        if v.get("pos_roll",{}).get("scenario")!=sc or "imp_roll" not in v: continue
        ip, ii = pos[v["pos_roll"]["video_id"]], pos[v["imp_roll"]["video_id"]]; n+=1
        for kind in ("p","copy","copy_mean"):
            sp, si = sur(ip,kind), sur(ii,kind); acc[kind].append(1.0 if si>sp else (0.5 if si==sp else 0.0))
            acc[kind+"_gap"].append(si-sp)
    out[sc]={k:(round(100*np.mean(v),1) if not k.endswith("_gap") else round(float(np.mean(v)),4)) for k,v in acc.items()}; out[sc]["n_pairs"]=n
    print(sc, out[sc], flush=True)
json.dump(out, open("/data/hyuntak/project/2026/2027_cvpr/vjepa2/auto_research/exp_results/scene/rollout2_copy_voe.json","w"), indent=1)
