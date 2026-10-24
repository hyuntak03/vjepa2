#!/usr/bin/env python3
"""[E] X4 — IntPhys 2 토큰 기억 비교 surprise ([비판] 요청, 토론장 [비판] 라운드 1 §4–6) (auto_research, 2026-10-22). 학습 · 모델 없음.

  /venv/vjepa/bin/python auto_research/scripts/ip2_x4.py

입력: 공유 캐시 (ip2_cache.py) 의 블록 토큰 · 픽셀 + [비판] 의 장면별 드러남 t_r · 마지막 문맥 t_c (results/crit_ip2_gap.json, 6 fps 장 번호).
⚠️ 2026-10-22 [비판] 정정 요청: t_c 는 쓰지 않는다 (간격 중앙값 1 장 = 사실상 직전 비교). 대신 "본 적 있나" 기억 비교.
블록: b_r = t_r // 2 + 1 (드러남이 시작된 뒤 처음 온전한 블록) 부터 3 블록. 드러남 전 블록 = 0 .. b_r − 2.
영상 v 의 surprise (칸 · 차원 평균 L1, 3 블록 평균):
  본 적 있나 (seen)  min_{b ≤ b_r−2} |h_v[드러남 뒤] − h_v[b]|    직전 비교 (prev)  |h_v[b] − h_v[b − 1]|    첫 블록 비교 (first)  |h_v[드러남 뒤] − h_v[0]|
묶음판 ([물리] 요청): 갈린 칸 토큰을 평균한 한 벡터끼리 같은 비교 (poolR).
영역: 갈린 칸 = 같은 짝의 가능 · 불가능 픽셀이 b_r..b_r+2 에서 갈린 칸 (칸 평균 |Δ| 가 사건 전 바닥보다 8 단계 넘게; 두 영상에 같은 영역 = 대칭) · 화면 전체.
보고: [비판] crit_ip2_xor 의 정확도 · 짝 1/2 · 결과 항 몫 · 문맥 항 CI. 조건 · 카메라 고정 / 이동."""
import argparse, json, os, random, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="/workspace/auto_research_cache/intphys2/Main")
ap.add_argument("--gap", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/crit_ip2_gap.json"))
ap.add_argument("--nblk", type=int, default=3)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/ip2_x4.json"))
a = ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crit_ip2_xor as XOR                                                         # noqa: E402
meta = json.load(open(os.path.join(a.cache, "meta.json")))
H = np.load(os.path.join(a.cache, "h.npy"), mmap_mode="r"); PX = np.load(os.path.join(a.cache, "pix.npy"), mmap_mode="r")
G, NB = 16, H.shape[1]
gap = json.load(open(a.gap))
gs = gap["scenes"] if isinstance(gap, dict) and "scenes" in gap else gap
gmap = {str(g["scene"]): g for g in gs}
idx = {(r["scene_index"], r["type"]): r["row"] for r in meta}


def cellpix(x):
    return x.reshape(x.shape[0], G, 16, G, 16, 3).mean(axis=(2, 4, 5)).reshape(x.shape[0], -1)


rows = {k: [] for k in ("seen_R", "seen_all", "seen_poolR", "prev_R", "prev_all", "prev_poolR", "first_R", "first_all")}
used, skipped = 0, 0
for sc, g in gmap.items():
    if g.get("t_r") is None or g["t_r"] < 0:
        skipped += 1; continue
    b_r = g["t_r"] // 2 + 1
    if b_r + a.nblk > NB or b_r < 3:
        skipped += 1; continue
    blk = list(range(b_r, b_r + a.nblk)); pre = list(range(0, b_r - 1))          # 드러남 전 블록 (드러남이 시작된 블록 b_r − 1 제외)
    per = {}
    for pid in ("1", "2"):
        iP, iI = idx[(sc, f"{pid}_Possible")], idx[(sc, f"{pid}_Impossible")]
        pP, pI = np.asarray(PX[iP], np.float32), np.asarray(PX[iI], np.float32)
        cd = np.abs(cellpix(pI) - cellpix(pP)).reshape(NB, 2, -1).mean(1)
        floor = np.median(cd[1:b_r - 1], 0)
        Rm = ((cd[blk] - floor[None]) > 8).any(0)                               # (256,) 갈린 칸 (3 블록 합집합, 오라클)
        for v, iv in (("P", iP), ("I", iI)):
            h = np.asarray(H[iv], np.float32)
            post = h[blk]                                                         # (3, 256, D)
            dpre = np.stack([np.abs(post - h[b][None]).mean(-1) for b in pre])     # (n_pre, 3, 256)
            prv = np.abs(post - h[[b - 1 for b in blk]]).mean(-1)                 # (3, 256)
            r = dict(seen_all=float(dpre.mean(axis=(1, 2)).min()), prev_all=float(prv.mean()), first_all=float(dpre[0].mean()))
            if Rm.any():
                r.update(seen_R=float(dpre[:, :, Rm].mean(axis=(1, 2)).min()), prev_R=float(prv[:, Rm].mean()), first_R=float(dpre[0][:, Rm].mean()))
                pb = post[:, Rm].mean(1)                                          # (3, D) 묶음
                r["seen_poolR"] = float(min(np.abs(pb - h[b][Rm].mean(0)[None]).mean() for b in pre))
                r["prev_poolR"] = float(np.abs(pb - h[[b - 1 for b in blk]][:, Rm].mean(1)).mean())
            per[(pid, v)] = r
    r0 = meta[idx[(sc, "1_Possible")]]
    for k in rows:
        if any(per[(pid, v)].get(k) is None for pid in ("1", "2") for v in ("P", "I")):
            continue
        for pid in ("1", "2"):
            for v in ("P", "I"):
                rows[k].append(dict(scene_index=sc, pair_id=pid, is_impossible=str(v == "I"), condition=r0["condition"], camera=r0["camera"],
                                    difficulty=r0["difficulty"], context_length=0, surprise_avg=per[(pid, v)][k]))
    used += 1


def report(rs):
    by = {}
    for r in rs:
        t = r["pair_id"] + ("I" if r["is_impossible"] == "True" else "P")
        by[(r["scene_index"], t)] = (r["surprise_avg"], r["condition"], r["camera"], r["difficulty"])
    sc_ = XOR.scenes(by); rng = random.Random(0); out = {}
    groups = [("all", sc_)] + [(c, [x for x in sc_ if x["cond"] == c]) for c in ["permanence", "immutability", "continuity", "solidity"]] \
             + [(f"cam={k}", [x for x in sc_ if x["cam"] == k]) for k in ["Fixed", "Moving"]] \
             + [(f"{c}|cam={k}", [x for x in sc_ if x["cond"] == c and x["cam"] == k]) for c in ["permanence", "immutability"] for k in ["Fixed", "Moving"]]
    for gname, xs in groups:
        st = XOR.stats(xs, rng)
        if st is None:
            continue
        st["acc_pair1"] = 100 * float(np.mean([x["d1"] > 0 for x in xs])); st["acc_pair2"] = 100 * float(np.mean([x["d2"] > 0 for x in xs]))
        out[gname] = st
    return out


res = {"n_scenes_used": used, "n_skipped": skipped, "blocks": a.nblk, "report": {k: report(v) for k, v in rows.items()}}
json.dump(res, open(a.out, "w"), indent=1, default=float)
print(f"장면 {used} 개 사용 (t_r · t_c 없음 / 범위 밖 {skipped})")
for k, rep in res["report"].items():
    print(f"\n[{k}]")
    for gname, st in rep.items():
        print(f"  {gname:24s} n {st['n_scene']:3d} 정확도 {st['acc']:5.1f} [{st['acc_ci'][0]:4.1f},{st['acc_ci'][1]:4.1f}] 짝1 {st['acc_pair1']:5.1f} 짝2 {st['acc_pair2']:5.1f} "
              f"결과항몫 {st['outcome_share']:.2f} 문맥항 {st['mean_i']:+.4f} [{st['mean_i_ci'][0]:+.4f},{st['mean_i_ci'][1]:+.4f}]")
