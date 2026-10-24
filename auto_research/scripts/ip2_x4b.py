#!/usr/bin/env python3
"""[E] X4 후속 (auto_research, 2026-10-22). 학습 · 모델 없음. 공유 캐시 + [비판] 드러남 시각 + [P] 칸 흐름.

  /venv/vjepa/bin/python auto_research/scripts/ip2_x4b.py [--reg]

(1) 짝 정보 없이 자리 고르기 (영상 자기 정보만):
  칸별 "본 적 있나" 거리 s(c) = min_{드러남 전 블록 b'} 3 블록 평균 |h[드러남 뒤](c) − h[b'](c)| (칸마다 b' 를 따로 고름)
  A_k   s(c) 가 큰 k 칸의 평균 (k = 8 · 16 · 31 · 64)
  B_k   토큰 변화 Δ(c) = 3 블록 평균 |h[드러남 뒤](c) − h[b_r−2](c)| − 그 칸의 드러남 전 이웃 블록 변화 평균 → 큰 k 칸의 s 평균
  C_k   영상 자기 픽셀 변화 (드러남 직전 블록 대비 칸 평균 |Δ픽셀|) 가 큰 k 칸의 s 평균
  진단: 고른 칸 중 오라클 갈린 칸 몫 · 같은 짝 가능 / 불가능 영상이 고른 칸의 Jaccard · 고른 칸을 갈린 칸 안 / 밖으로 나눈 surprise
  기준: 오라클 갈린 칸의 s 평균 (칸별 최소) · 화면 전체 s 평균.
(2) --reg: 이동 카메라 주소 등록. [P] 의 칸 흐름 U[k] (블록 k+1 → k, 칸 단위, 가능 영상만) 를 짝의 두 영상에 똑같이 쓴다.
  드러남 뒤 블록의 칸 c 를 U 사슬로 드러남 전 블록 b' 의 자리로 옮겨 (가장 가까운 칸, 토큰 보간 없음) 비교. 화면 밖이면 그 b' 는 뺀다.
  판: flow (칸별 흐름 사슬) · med (블록마다 화면 중앙값 흐름만 = 카메라만). 영역 = 오라클 갈린 칸."""
import argparse, glob, json, os, random, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="/workspace/auto_research_cache/intphys2/Main")
ap.add_argument("--gap", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/crit_ip2_gap.json"))
ap.add_argument("--flow", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/ip2_phys"))
ap.add_argument("--reg", action="store_true")
ap.add_argument("--nblk", type=int, default=3)
ap.add_argument("--out", default=None)
a = ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crit_ip2_xor as XOR                                                         # noqa: E402
meta = json.load(open(os.path.join(a.cache, "meta.json")))
H = np.load(os.path.join(a.cache, "h.npy"), mmap_mode="r"); PX = np.load(os.path.join(a.cache, "pix.npy"), mmap_mode="r")
G, NB = 16, H.shape[1]
gs = json.load(open(a.gap))["scenes"]
idx = {(r["scene_index"], r["type"]): r["row"] for r in meta}
KS = (8, 16, 31, 64)
yy, xx = np.divmod(np.arange(G * G), G)


def cellpix(x):
    return x.reshape(x.shape[0], G, 16, G, 16, 3).mean(axis=(2, 4, 5)).reshape(x.shape[0], -1)


def chain_maps(U, blk, pre, mode):
    """U (NB−1, 256, 2) → maps[t][j] = (256,) 목적 칸 번호 (−1 = 화면 밖) : 드러남 뒤 블록 blk[t] 의 칸 → 드러남 전 블록 pre[j]."""
    out = []
    for b in blk:
        p = np.stack([xx, yy], 1).astype(np.float32)                              # (256, 2) (x, y)
        ok = np.ones(G * G, bool)
        cur = {}
        for k in range(b - 1, -1, -1):                                             # 블록 k+1 → k
            if mode == "med":
                p = p + np.median(U[k].astype(np.float32), 0)[None]
            else:
                ci = np.clip(np.round(p[:, 1]), 0, G - 1).astype(int) * G + np.clip(np.round(p[:, 0]), 0, G - 1).astype(int)
                p = p + U[k][ci].astype(np.float32)
            ok &= (p[:, 0] > -0.5) & (p[:, 0] < G - 0.5) & (p[:, 1] > -0.5) & (p[:, 1] < G - 0.5)
            cur[k] = np.where(ok, np.clip(np.round(p[:, 1]), 0, G - 1).astype(int) * G + np.clip(np.round(p[:, 0]), 0, G - 1).astype(int), -1)
        out.append([cur[bp] for bp in pre])
    return out


flows = {}
if a.reg:
    for f in glob.glob(os.path.join(a.flow, "Main_U_*.npy")):
        flows[os.path.basename(f)[len("Main_U_"):-4]] = f

rows = {}
diag = {"prec_A": [], "prec_B": [], "prec_C": [], "jac_A": [], "jac_B": [], "jac_C": [], "R_area": []}
for g in gs:
    sc = g["scene"]
    if g.get("t_r") is None or g["t_r"] < 0:
        continue
    b_r = g["t_r"] // 2 + 1
    if b_r + a.nblk > NB or b_r < 3:
        continue
    blk = list(range(b_r, b_r + a.nblk)); pre = list(range(0, b_r - 1))
    per = {}
    for pid in ("1", "2"):
        iP, iI = idx[(sc, f"{pid}_Possible")], idx[(sc, f"{pid}_Impossible")]
        pP, pI = np.asarray(PX[iP], np.float32), np.asarray(PX[iI], np.float32)
        cd = np.abs(cellpix(pI) - cellpix(pP)).reshape(NB, 2, -1).mean(1)
        Rm = ((cd[blk] - np.median(cd[1:b_r - 1], 0)[None]) > 8).any(0)          # 오라클 갈린 칸
        U = None
        if a.reg:
            key = meta[iP]["file"].split("/")[-1][:16]
            if key in flows:
                U = np.load(flows[key])
        sel = {}
        for v, iv in (("P", iP), ("I", iI)):
            h = np.asarray(H[iv], np.float32)
            post = h[blk]                                                         # (3, 256, D)
            Dm = np.stack([np.abs(post - h[bp][None]).mean(-1).mean(0) for bp in pre])   # (n_pre, 256) 3 블록 평균
            s = Dm.min(0)                                                         # 칸별 최소
            r = {"oracleR_cellmin": float(s[Rm].mean()) if Rm.any() else None, "all_cellmin": float(s.mean())}
            for k in KS:
                r[f"A{k}"] = float(np.sort(s)[-k:].mean())
            floor = np.abs(h[1:b_r - 1] - h[:b_r - 2]).mean(-1).mean(0)            # (256,)
            dtok = np.abs(post - h[b_r - 2][None]).mean(-1).mean(0) - floor
            pxv = np.asarray(PX[iv], np.float32)
            cp = cellpix(pxv).reshape(NB, 2, -1).mean(1)
            dpix = np.abs(cp[blk] - cp[b_r - 2][None]).mean(0)
            for name, score in (("B", dtok), ("C", dpix)):
                order = np.argsort(-score)
                for k in KS:
                    r[f"{name}{k}"] = float(s[order[:k]].mean())
                S = np.zeros(G * G, bool); S[order[:31]] = True
                sel[(v, name)] = S
                if Rm.any():
                    r[f"{name}31_inR"] = float(s[S & Rm].mean()) if (S & Rm).any() else None
                    r[f"{name}31_outR"] = float(s[S & ~Rm].mean()) if (S & ~Rm).any() else None
            SA = np.zeros(G * G, bool); SA[np.argsort(-s)[:31]] = True
            sel[(v, "A")] = SA
            if Rm.any():
                r["A31_inR"] = float(s[SA & Rm].mean()) if (SA & Rm).any() else None
                r["A31_outR"] = float(s[SA & ~Rm].mean()) if (SA & ~Rm).any() else None
            if U is not None:
                for mode in ("flow", "med"):
                    maps = chain_maps(U, blk, pre, mode)
                    best = np.full(G * G, np.inf)
                    for j, bp in enumerate(pre):
                        acc, cnt = np.zeros(G * G), np.zeros(G * G)
                        for t in range(len(blk)):
                            m = maps[t][j]; okc = m >= 0
                            acc[okc] += np.abs(post[t][okc] - h[bp][m[okc]]).mean(-1); cnt[okc] += 1
                        d = np.where(cnt == len(blk), acc / np.maximum(cnt, 1), np.inf)   # 세 블록 모두 화면 안일 때만
                        best = np.minimum(best, d)
                    good = np.isfinite(best)
                    r[f"reg_{mode}_R"] = float(best[Rm & good].mean()) if (Rm & good).any() else None
                    r[f"reg_{mode}_validR"] = float((Rm & good).sum() / max(Rm.sum(), 1))
                r["noreg_R_sameset"] = r["oracleR_cellmin"]
            per[(pid, v)] = r
        if Rm.any():
            diag["R_area"].append(float(Rm.mean()))
            for name in ("A", "B", "C"):
                for v in ("P", "I"):
                    diag[f"prec_{name}"].append(float((sel[(v, name)] & Rm).sum() / 31))
                inter = (sel[("P", name)] & sel[("I", name)]).sum(); uni = (sel[("P", name)] | sel[("I", name)]).sum()
                diag[f"jac_{name}"].append(float(inter / max(uni, 1)))
    r0 = meta[idx[(sc, "1_Possible")]]
    keys = set().union(*[set(x.keys()) for x in per.values()])
    for k in keys:
        if k.endswith("_validR"):                                                # 진단값 (surprise 아님)
            diag.setdefault(k, []).extend(per[(pid, v)][k] for pid in ("1", "2") for v in ("P", "I") if per[(pid, v)].get(k) is not None)
            continue
        if any(per[(pid, v)].get(k) is None for pid in ("1", "2") for v in ("P", "I")):
            continue
        for pid in ("1", "2"):
            for v in ("P", "I"):
                rows.setdefault(k, []).append(dict(scene_index=sc, pair_id=pid, is_impossible=str(v == "I"), condition=r0["condition"], camera=r0["camera"],
                                                   difficulty=r0["difficulty"], surprise_avg=per[(pid, v)][k]))


def report(rs):
    by = {}
    for r in rs:
        by[(r["scene_index"], r["pair_id"] + ("I" if r["is_impossible"] == "True" else "P"))] = (r["surprise_avg"], r["condition"], r["camera"], r["difficulty"])
    sc_ = XOR.scenes(by); rng = random.Random(0); out = {}
    groups = [("all", sc_)] + [(c, [x for x in sc_ if x["cond"] == c]) for c in ["permanence", "immutability", "continuity", "solidity"]] \
             + [(f"cam={k}", [x for x in sc_ if x["cam"] == k]) for k in ["Fixed", "Moving"]] \
             + [(f"no_solidity|cam={k}", [x for x in sc_ if x["cam"] == k and x["cond"] != "solidity"]) for k in ["Fixed", "Moving"]]
    for gname, xs in groups:
        st = XOR.stats(xs, rng)
        if st is None:
            continue
        st["acc_pair1"] = 100 * float(np.mean([x["d1"] > 0 for x in xs])); st["acc_pair2"] = 100 * float(np.mean([x["d2"] > 0 for x in xs]))
        out[gname] = st
    return out


if a.reg:                                                                        # 같은 장면 집합에서 등록 전 · 후 비교
    for mode in ("flow", "med"):
        kk = f"reg_{mode}_R"
        if kk in rows:
            scs = {r["scene_index"] for r in rows[kk]}
            rows[f"noreg_R_on_{mode}set"] = [r for r in rows["oracleR_cellmin"] if r["scene_index"] in scs]
res = {"report": {k: report(v) for k, v in sorted(rows.items())}, "diag": {k: (float(np.mean(v)) if v else None) for k, v in diag.items()},
       "n_flow_files": len(flows)}
out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), f"../results/ip2_x4b{'_reg' if a.reg else ''}.json")
json.dump(res, open(out, "w"), indent=1, default=float)
print("진단:", {k: (round(v, 3) if v is not None else None) for k, v in res["diag"].items()})
for k, rep in res["report"].items():
    st = rep.get("all")
    if st is None:
        continue
    f = lambda g: (f"{rep[g]['acc']:5.1f}" if g in rep else "  —  ")
    print(f"{k:18s} n {st['n_scene']:3d} 정확도 {st['acc']:5.1f} [{st['acc_ci'][0]:4.1f},{st['acc_ci'][1]:4.1f}] 짝1 {st['acc_pair1']:5.1f} 짝2 {st['acc_pair2']:5.1f} "
          f"결과항몫 {st['outcome_share']:.2f} 문맥항 {st['mean_i']:+.4f} [{st['mean_i_ci'][0]:+.4f},{st['mean_i_ci'][1]:+.4f}] | 고정 {f('cam=Fixed')} 이동 {f('cam=Moving')} | "
          f"영속 {f('permanence')} 불변 {f('immutability')} 연속 {f('continuity')} 고체성 {f('solidity')}")
