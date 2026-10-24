#!/usr/bin/env python3
"""[E] IntPhys 2 토큰 surprise 분해 (auto_research, 2026-10-22). 학습 없음. 입력 = ip2_cache.py 의 공유 캐시.

  CUDA_VISIBLE_DEVICES=2 /venv/vjepa/bin/python -W ignore auto_research/scripts/ip2_decomp.py [--split Main]

짝 (scene, pair_id ∈ {1, 2}) 마다 가능 영상 P · 불가능 영상 I. 블록 b = 장 2b, 2b+1 (6 fps).
사건 시각 (픽셀, 오라클): d(f) = 장 평균 |pix_I − pix_P|. 바닥 = 앞 8 장 중앙값, 퍼짐 = 앞 16 장 MAD.
  f0 = d(f) > 바닥 + max(1.5, 4·퍼짐) 이 처음인 장 (그 뒤 2 장 중 하나 더 넘어야 함). 사건 블록 b0 = f0 // 2.
갈린 칸 R_b (오라클): 블록 b 의 칸 평균 |Δpix| (두 장) 이 그 칸의 사건 전 바닥 (블록 1..b0−1 중앙값) 보다 8 단계 넘게 큰 칸.
① 짝 토큰 차 D(b) = 칸 · 차원 평균 |h_I − h_P|. 사건 전 바닥 D_pre = 블록 1..b0−1 평균 (칸별 바닥도). 사건 뒤 = 블록 b0..b0+K−1 (K = 8, 영상 끝까지).
   비 = D_post / D_pre. 초과분 e(b,c) = |Δh|(b,c) − 칸별 바닥(c) 의 양수 부분: 갈린 칸 몫 vs 갈린 칸 넓이.
② nuisance 대비: 초과분 평균 / 같은 칸의 영상 안 이웃 블록 변화 |h_v(b) − h_v(b−1)| (두 영상 평균) — 갈린 칸 · 화면 전체.
③ 예측: 영상마다 자기 문맥 블록 [b0−8, b0) 로 기준 AR rollout K 블록 (KV, 채점기와 같은 방식).
   선호 지수 = (Σ|p − h_I| − Σ|p − h_P|) / Σ|h_I − h_P| (갈린 칸 · 화면 전체, 사건 뒤 블록). > 0 = 가능 쪽에 가깝다. 두 영상의 p 평균.
④ 표준 보고: 영상별 surprise = rollout p 와 자기 정답의 L1 (사건 뒤 블록; 화면 전체 · 갈린 칸) · TF 한 걸음 (캐시 p_tf, 영상 전체 · 사건 뒤) →
   [비판] crit_ip2_xor 의 정확도 · 짝 1/2 · 결과 항 몫 · 문맥 항 CI. 조건 4 종 · 카메라로."""
import argparse, json, os, random, sys
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="Main")
ap.add_argument("--cache", default="/workspace/auto_research_cache/intphys2")
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--C", type=int, default=8)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/ip2_decomp.json"))
args = ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crit_ip2_xor as XOR                                                         # noqa: E402
_argv = list(sys.argv); sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT                                                            # noqa: E402
sys.argv = _argv
torch.set_num_threads(8)
dev = torch.device("cuda")
O = os.path.join(args.cache, args.split)
meta = json.load(open(os.path.join(O, "meta.json")))
H = np.load(os.path.join(O, "h.npy"), mmap_mode="r"); PX = np.load(os.path.join(O, "pix.npy"), mmap_mode="r"); PT = np.load(os.path.join(O, "p_tf.npy"), mmap_mode="r")
G, S, D, NB = 16, 256, 1280, H.shape[1]
W = FT.NoiseWrap(FT.load_pred("natural_ar_vw_state", "latest.pt"), 32, 0.0).to(dev).eval()
core = W.pred

idx = {}
for r in meta:
    idx[(r["scene_index"], r["type"])] = r["row"]


def cellpix(x):                                    # (T, 256, 256, 3) → (T, 256) 칸 평균
    return x.reshape(x.shape[0], G, 16, G, 16, 3).mean(axis=(2, 4, 5)).reshape(x.shape[0], -1)


@torch.no_grad()
def rollout(h, b0):                                # h (NB, S, D) → p (K, S, D) 블록 b0..b0+K−1
    c0 = max(0, b0 - args.C)
    ctx = torch.from_numpy(np.asarray(h[c0:b0], np.float32)).to(dev).reshape(1, -1, D)
    ci = torch.arange(ctx.shape[1], device=dev)[None]
    K = min(args.K, NB - b0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        p = core.rollout_kv(ctx, ci, K).float()
    return p.view(K, S, D).cpu().numpy()


REC = []
rows_sur = {k: [] for k in ("roll_all", "roll_R", "tf_all_video", "tf_post")}
scenes = sorted(set(r["scene_index"] for r in meta), key=int)
for si, sc in enumerate(scenes):
    for pid in ("1", "2"):
        iP, iI = idx[(sc, f"{pid}_Possible")], idx[(sc, f"{pid}_Impossible")]
        r0 = meta[iP]
        pP = np.asarray(PX[iP], np.float32); pI = np.asarray(PX[iI], np.float32)
        d = np.abs(pI - pP).mean(axis=(1, 2, 3))
        base, mad = np.median(d[:8]), np.median(np.abs(d[:16] - np.median(d[:16]))) + 1e-6
        thr = base + max(1.5, 4 * mad)
        over = d > thr
        f0 = None
        for f in range(2, len(d) - 2):
            if over[f] and (over[f + 1] or over[f + 2]):
                f0 = f; break
        rec = dict(scene=sc, pair=pid, condition=r0["condition"], camera=r0["camera"], difficulty=r0["difficulty"], f0=f0,
                   pix_floor=float(base), pix_peak=float(d.max()))
        for v, iv in (("P", iP), ("I", iI)):                                    # 영상 전체 TF surprise (사건 시각과 무관, 모든 짝)
            hv = np.asarray(H[iv], np.float32)
            tfv = np.abs(np.asarray(PT[iv], np.float32) - hv[1:]).mean()
            rows_sur["tf_all_video"].append(dict(scene_index=sc, pair_id=pid, is_impossible=str(v == "I"), condition=r0["condition"], camera=r0["camera"],
                                                 difficulty=r0["difficulty"], context_length=0, surprise_avg=float(tfv)))
        if f0 is None or f0 // 2 < 3 or f0 // 2 > NB - 2:
            rec["skip"] = "사건 시각 없음 / 너무 이르거나 늦음"; REC.append(rec); continue
        b0 = f0 // 2; K = min(args.K, NB - b0)
        cP = cellpix(pP).reshape(NB, 2, -1).mean(1); cI = cellpix(pI).reshape(NB, 2, -1).mean(1)
        cd = np.abs(cellpix(pI) - cellpix(pP)).reshape(NB, 2, -1).mean(1)        # (NB, 256) 칸별 픽셀 차
        cfloor = np.median(cd[1:b0], 0)
        Rm = (cd - cfloor[None]) > 8                                             # (NB, 256)
        hP = np.asarray(H[iP], np.float32); hI = np.asarray(H[iI], np.float32)
        dh = np.abs(hI - hP).mean(-1)                                            # (NB, 256)
        hfloor = dh[1:b0].mean(0)
        post = slice(b0, b0 + K)
        Dpre, Dpost = float(dh[1:b0].mean()), float(dh[post].mean())
        e = np.clip(dh[post] - hfloor[None], 0, None)
        Rp = Rm[post]
        nb_ = np.abs(hP[1:] - hP[:-1]).mean(-1) * 0.5 + np.abs(hI[1:] - hI[:-1]).mean(-1) * 0.5   # (NB−1, 256), 행 b−1 = 블록 b vs b−1
        nuis = nb_[b0 - 1:b0 - 1 + K]
        rec.update(b0=b0, K=K, D_pre=Dpre, D_post=Dpost, ratio=Dpost / Dpre,
                   R_area=float(Rp.mean()), R_share_excess=float(e[Rp].sum() / max(e.sum(), 1e-9)) if Rp.any() else None,
                   excess_all=float(e.mean()), excess_R=float(e[Rp].mean()) if Rp.any() else None,
                   nuis_all=float(nuis.mean()), nuis_R=float(nuis[Rp].mean()) if Rp.any() else None,
                   pix_post_R=float(cd[post][Rp].mean()) if Rp.any() else None)
        rec["excess_over_nuis_all"] = rec["excess_all"] / rec["nuis_all"]
        rec["excess_over_nuis_R"] = (rec["excess_R"] / rec["nuis_R"]) if Rp.any() else None
        # ③ rollout 선호
        pr = {}
        for v, hv in (("P", hP), ("I", hI)):
            p = rollout(hv, b0)                                                  # (K, S, D)
            eP = np.abs(p - hP[post]).mean(-1); eI = np.abs(p - hI[post]).mean(-1); eD = np.abs(hI[post] - hP[post]).mean(-1)
            pr[v] = dict(all=float((eI.sum() - eP.sum()) / eD.sum()),
                         R=float((eI[Rp].sum() - eP[Rp].sum()) / eD[Rp].sum()) if Rp.any() else None,
                         own_all=float((eP if v == "P" else eI).mean()), own_R=float((eP if v == "P" else eI)[Rp].mean()) if Rp.any() else None)
        rec["pref_all"] = 0.5 * (pr["P"]["all"] + pr["I"]["all"])
        rec["pref_R"] = 0.5 * (pr["P"]["R"] + pr["I"]["R"]) if Rp.any() else None
        # ④ 영상별 surprise
        tfP = np.abs(np.asarray(PT[iP], np.float32) - hP[1:]).mean(-1); tfI = np.abs(np.asarray(PT[iI], np.float32) - hI[1:]).mean(-1)   # (NB−1, 256)
        for v, own_all, own_R, tfv in (("P", pr["P"]["own_all"], pr["P"]["own_R"], tfP), ("I", pr["I"]["own_all"], pr["I"]["own_R"], tfI)):
            base_row = dict(scene_index=sc, pair_id=pid, is_impossible=str(v == "I"), condition=r0["condition"], camera=r0["camera"], difficulty=r0["difficulty"], context_length=0)
            rows_sur["roll_all"].append(dict(base_row, surprise_avg=own_all))
            if own_R is not None:
                rows_sur["roll_R"].append(dict(base_row, surprise_avg=own_R))
            rows_sur["tf_post"].append(dict(base_row, surprise_avg=float(tfv[b0 - 1:b0 - 1 + K].mean())))
        REC.append(rec)
    if si % 25 == 0:
        print(f"{si}/{len(scenes)} 장면", flush=True)


def xor_report(rows):
    by = {}
    for r in rows:
        t = ("1" if r["pair_id"] == "1" else "2") + ("I" if r["is_impossible"] == "True" else "P")
        by[(r["scene_index"], t)] = (r["surprise_avg"], r["condition"], r["camera"], r["difficulty"])
    sc = XOR.scenes(by)
    rng = random.Random(0)
    groups = [("all", sc)] + [(c, [x for x in sc if x["cond"] == c]) for c in ["permanence", "immutability", "continuity", "solidity"]] \
             + [(f"cam={k}", [x for x in sc if x["cam"] == k]) for k in ["Fixed", "Moving"]]
    out = {}
    for g, xs in groups:
        st = XOR.stats(xs, rng)
        if st is None:
            continue
        st["acc_pair1"] = 100 * float(np.mean([x["d1"] > 0 for x in xs])); st["acc_pair2"] = 100 * float(np.mean([x["d2"] > 0 for x in xs]))
        out[g] = st
    return out


def agg(recs, key):
    v = [r[key] for r in recs if r.get(key) is not None and np.isfinite(r[key])]
    if not v:
        return None
    v = np.array(v); rng = np.random.default_rng(0)
    bs = [rng.choice(v, len(v)).mean() for _ in range(2000)]
    return dict(n=len(v), mean=float(v.mean()), median=float(np.median(v)), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])


ok = [r for r in REC if "skip" not in r]
res = {"n_pairs": len(REC), "n_ok": len(ok), "skips": len(REC) - len(ok), "decomp": {}, "xor": {k: xor_report(v) for k, v in rows_sur.items()}}
KEYS = ("ratio", "R_area", "R_share_excess", "excess_over_nuis_all", "excess_over_nuis_R", "pref_all", "pref_R", "D_pre", "D_post", "pix_floor", "b0")
for gname, sel in [("all", lambda r: True)] + [(c, (lambda c: lambda r: r["condition"] == c)(c)) for c in ["permanence", "immutability", "continuity", "solidity"]] \
        + [(f"cam={k}", (lambda k: lambda r: r["camera"] == k)(k)) for k in ["Fixed", "Moving"]]:
    for pid in ("both", "1", "2"):
        rs = [r for r in ok if sel(r) and (pid == "both" or r["pair"] == pid)]
        res["decomp"][f"{gname}|pair{pid}"] = {k: agg(rs, k) for k in KEYS}
json.dump({"res": res, "records": REC}, open(args.out, "w"), indent=1, default=float)
print(f"\n짝 {len(REC)} 개 중 분석 {len(ok)} (사건 시각 못 찾음 · 범위 밖 {len(REC) - len(ok)})")
print(f"{'무리':22s} {'n':>4s} {'사건뒤/전':>9s} {'갈린칸넓이':>9s} {'초과분 갈린칸몫':>12s} {'초과/nuis 전체':>12s} {'초과/nuis 갈린칸':>13s} {'선호 전체 [CI]':>22s} {'선호 갈린칸 [CI]':>22s}")
for k, v in res["decomp"].items():
    if v["ratio"] is None:
        continue
    f = lambda x: f"{x['mean']:.3f}" if x else "—"
    fc = lambda x: f"{x['mean']:+.3f} [{x['ci'][0]:+.3f},{x['ci'][1]:+.3f}]" if x else "—"
    print(f"{k:22s} {v['ratio']['n']:4d} {f(v['ratio']):>9s} {f(v['R_area']):>9s} {f(v['R_share_excess']):>12s} {f(v['excess_over_nuis_all']):>12s} {f(v['excess_over_nuis_R']):>13s} {fc(v['pref_all']):>22s} {fc(v['pref_R']):>22s}")
for k, rep in res["xor"].items():
    print(f"\n[표준 보고 · surprise = {k}]")
    for g, st in rep.items():
        print(f"  {g:14s} n {st['n_scene']:3d} 정확도 {st['acc']:5.1f} [{st['acc_ci'][0]:4.1f},{st['acc_ci'][1]:4.1f}] 짝1 {st['acc_pair1']:5.1f} 짝2 {st['acc_pair2']:5.1f} "
              f"결과항몫 {st['outcome_share']:.2f} 문맥항 {st['mean_i']:+.4f} [{st['mean_i_ci'][0]:+.4f},{st['mean_i_ci'][1]:+.4f}]")
