#!/usr/bin/env python3
"""Rubric 에서 PFT 가 왜 안 오르나 — 진단 모음 (2026-11-01). 문서: z_analysis/release/08_rubric_transfer/README.md

모델 셋 (predictor 만 다르다, encoder 는 릴리즈 ViT-H 고정): release · vobftg (VOBench PFT) · v3_full_s0 (BlenderTrack v3 전체 PFT) + 복사 기준선.
입력은 이미 있는 한 번 통과 채점 원배열 (predictor_lab/results/v3_ablation/te/<벤치>_final/: l1 (n,K,8) · copy (n,8)) 과 데이터 메타뿐이다 (GPU 0 장).

정의 (quad = 같은 영상의 present · vanish · appear · absent 네 클립, S = 미래 블록 평균 surprise)
  m_A = S(vanish) − S(present)  (> 0 = 사라짐 정답) · m_B = S(appear) − S(absent)  (> 0 = 생겨남 정답)
  문맥 효과 c = m_A + m_B · 물체 미래 편향 b = (m_B − m_A)/2 · 둘 다 S(present) 로 나눠 %. m_A = c/2 − b → A 정답에는 c > 2b 가 필요
  H = 문맥 끝에서 이어서 완전히 가려진 샘플 수 · R = 다시 보이기 시작하는 첫 미래 샘플 (0 = 미래 첫 샘플). 샘플 = 모델 입력 32 장 중 하나
  E_j = [S_j(vanish) − S_j(absent)] / S_j(absent) — 두 미래가 모두 비었고 문맥만 다르다 → 물체 있던 문맥의 예측이 블록 j 에 넣은 '물체' 몫. PFT − 릴리즈 로 읽는다

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; $P z_research/scripts/analysis/rubric_transfer_diag.py [--only decomp,pixel,timing,matched,bytiming,expect]
  → 표 출력 + z_analysis/release/08_rubric_transfer/results/diag.json
"""
import argparse
import collections
import csv
import json
import os

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
TE = f"{REPO}/predictor_lab/results/v3_ablation/te"
CLV = "/data/hyuntak/project/2026/2027_cvpr/testbed/cleverer_permanence_v1"
RUB_FRAMES = "/data/hyuntak/project/2026/2027_cvpr/nfs_cache/rubric_quads_frames/uniform"
OUT = f"{REPO}/z_analysis/release/08_rubric_transfer/results/diag.json"
M = ["release", "vobftg", "v3_full_s0", "copy"]
NM = {"release": "릴리즈", "vobftg": "VOBench PFT", "v3_full_s0": "V3 PFT", "copy": "복사"}
BENCH = {"rubric": "rubric_final", "v3test": "v3test_final", "clevrer": "clevrer_final", "vob_late": "vob_late_final"}
RES = {}


def load(name):
    d = f"{TE}/{name}"
    meta = json.load(open(f"{d}/meta.json")); tags = meta["pred_tags"]
    l1 = np.load(f"{d}/l1.npy"); cp = np.load(f"{d}/copy.npy")
    S = {m: (cp if m == "copy" else l1[:, tags.index(m)]) for m in M}          # (n, 8 블록)
    g = collections.defaultdict(dict)
    for i, (b, v) in enumerate(zip(meta["block_id"], meta["variant"])):
        g[str(b)][v] = i
    g = {b: q for b, q in g.items() if all(k in q for k in ("pos_a", "imp_ab", "imp_ba", "pos_b"))}
    return meta, g, S


def margins(S, m, q, slot=None):
    s = S[m] if slot is None else S[m][:, slot]
    s = s.mean(-1) if s.ndim == 2 else s
    return s[q["imp_ab"]] - s[q["pos_a"]], s[q["imp_ba"]] - s[q["pos_b"]], s[q["pos_a"]]


def boot(x, n=10000, seed=0):
    x = np.asarray(x); bs = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def rubric_reveal():
    """Rubric 영상별 (H, R, 미래 블록별 물체 보임) — 실효 30 fps 프레임 번호 기준 (audit.json)."""
    a = json.load(open(f"{REPO}/data_csv/rubric_quads_uniform/audit.json")); out = {}
    for b, x in a.items():
        fe = np.array(x["frames_eff"]); fut = fe[16:]
        out[b.split("/")[-1]] = dict(H=int((fe[:16] > x["last_visible_eff"]).sum()), R=int(np.argmax(fut >= x["reveal_eff"])),
                                     vis=(fut >= x["reveal_eff"]).reshape(8, 2).any(1), dt=(fe[31] - fe[0]) / 31 / 30, src=b)
    return out


# ---------------------------------------------------------------- 1. 분해 (문맥 효과 · 편향)
def decomp():
    print("\n#### 1. 쌍 차이 분해 — 문맥 효과 c 와 물체 미래 편향 b (S 대비 %)")
    r = {}
    for key, name in BENCH.items():
        meta, g, S = load(name); r[key] = {}
        print(f"== {key} (quad {len(g)})")
        for m in M:
            mA, mB, S0 = map(np.array, zip(*[margins(S, m, q) for q in g.values()]))
            r[key][m] = dict(A=100 * np.mean(mA > 0), B=100 * np.mean(mB > 0), c=100 * np.mean((mA + mB) / S0), b=100 * np.mean((mB - mA) / 2 / S0))
            x = r[key][m]
            print(f"   {NM[m]:11s} A {x['A']:5.1f} B {x['B']:5.1f} | c/S {x['c']:+.3f}%  b/S {x['b']:+.3f}%  c/(2b) {x['c'] / (2 * x['b']):+.2f}")
    # Rubric: 판정 일치 · c 차이 CI · 블록 (다시 보임 전/후)
    meta, g, S = load(BENCH["rubric"]); rv = rubric_reveal()
    src = {b: meta["video_id"][q["pos_a"]].rsplit("_", 2)[0] for b, q in g.items()}
    J = {m: np.concatenate([[margins(S, m, q)[0] > 0 for q in g.values()], [margins(S, m, q)[1] > 0 for q in g.values()]]) for m in M}
    same = {f"{a}|{b}": int((J[a] == J[b]).sum()) for i, a in enumerate(M) for b in M[i + 1:]}
    three = int(((J["release"] == J["vobftg"]) & (J["vobftg"] == J["v3_full_s0"])).sum())
    print(f"\n== Rubric 판정 일치 (40 쌍): 세 모델 같음 {three} · " + " · ".join(f"{k} {v}" for k, v in same.items()))
    C = {m: np.array([sum(margins(S, m, q)[:2]) for q in g.values()]) for m in M}
    ci = {f"{m}-release": boot(C[m] - C["release"]) for m in ("vobftg", "v3_full_s0")}
    ci.update({f"{m}-copy": boot(C[m] - C["copy"]) for m in ("release", "vobftg", "v3_full_s0")})
    for k, (mu, lo, hi) in ci.items():
        print(f"   c 차이 {k:22s} {1e3 * mu:+.3f} [{1e3 * lo:+.3f}, {1e3 * hi:+.3f}] ×1e-3 (영상 bootstrap)")
    slot = {}
    for m in M:
        cv, cn, bv = [], [], []
        for b, q in g.items():
            vis = rv[src[b]]["vis"]; sm = S[m]
            mA = sm[q["imp_ab"]] - sm[q["pos_a"]]; mB = sm[q["imp_ba"]] - sm[q["pos_b"]]
            cv += list((mA + mB)[vis]); cn += list((mA + mB)[~vis]); bv += list(((mB - mA) / 2)[vis])
        slot[m] = dict(c_reveal=1e3 * np.mean(cv), c_pre=1e3 * np.mean(cn), b_reveal=1e3 * np.mean(bv))
        print(f"   블록별 {NM[m]:11s} c 다시 보인 블록 {slot[m]['c_reveal']:+.3f} · 그 전 블록 {slot[m]['c_pre']:+.3f} · b 다시 보인 블록 {slot[m]['b_reveal']:+.3f} (×1e-3)")
    # 가능 미래 예측 오차 S 변화 (릴리즈 대비)
    dS = {}
    for key, name in list(BENCH.items()) + [("ip1uni", "ip1uni_final"), ("ip2uni", "ip2uni_final")]:
        meta, _, S = load(name); pl = np.array(meta["plausible"]).astype(int) == 1
        base = S["release"][pl].mean()
        dS[key] = {m: 100 * (S[m][pl].mean() / base - 1) for m in ("vobftg", "v3_full_s0")}
    print("   가능 미래 S 변화 (릴리즈 대비 %): " + " · ".join(f"{k} VOB {v['vobftg']:+.1f} / V3 {v['v3_full_s0']:+.1f}" for k, v in dS.items()))
    RES["decomp"] = dict(bench=r, rubric_same=same, rubric_three_same=three, rubric_c_ci=ci, rubric_slot=slot, dS=dS)


# ---------------------------------------------------------------- 2. 화소 점검
def pixel():
    from PIL import Image
    print("\n#### 2. Rubric 화소 점검 — 두 미래 (물체 / 빈) 가 어디서 다른가 (256² 로 늘림, |Δ| > 25)")
    rv = rubric_reveal(); rows = {}
    L = lambda b, v, i: np.asarray(Image.open(f"{RUB_FRAMES}/{b}/{v}/{i:06d}.png").convert("RGB").resize((256, 256), Image.BILINEAR)).astype(np.int16)
    for name, x in rv.items():
        b = x["src"]; post = np.repeat(x["vis"], 2); pre_d, tok = [], []
        for j in range(16):
            d = np.abs(L(b, "possible_present", 16 + j) - L(b, "impossible_vanish", 16 + j)).max(2) > 25
            (tok if post[j] else pre_d).append(d.reshape(16, 16, 16, 16).any((1, 3)).mean() if post[j] else d.mean())
        rows[name] = dict(pre_reveal_pix=100 * float(np.mean(pre_d)) if pre_d else 0.0, post_reveal_tok=100 * float(np.mean(tok)) if tok else 0.0)
        print(f"   {name:20s} 다시 보이기 전 다른 화소 {rows[name]['pre_reveal_pix']:.2f}% · 다시 보인 뒤 다른 토큰 {rows[name]['post_reveal_tok']:5.2f}%")
    RES["pixel"] = rows


# ---------------------------------------------------------------- 3. 재등장 타이밍 분포
def timing_sets():
    out = {}
    rows = [r for r in csv.DictReader(open(f"{REPO}/data_csv/vobench_v1_split_v2g/index_train.csv")) if r["occlusion"] == "late" and r["variant"] == "pos_a"]
    H, R = [], []
    for r in rows:
        s = r["samples"].replace("|", ""); h = 0
        while s[15 - h] == "H":
            h += 1
        H.append(h); R.append(next(i for i in range(16, 32) if s[i] == ".") - 16)
    out["VOBench PFT 학습"] = dict(H=H, R=R, dt=0.10)
    vis = np.load(f"{REPO}/data_csv/blender_track_v3/arms/full.npz", allow_pickle=True)["vis"].astype(np.float32)[:, ::2]
    vis = vis[~np.isnan(vis).all(1)]                                   # pos_b (target 없음) 제외
    H, R = [], []
    for v in np.nan_to_num(vis, nan=0.0):
        h = 0
        while h < 16 and v[15 - h] < 0.05:
            h += 1
        r = np.nonzero(v[16:] >= 0.05)[0]; H.append(h); R.append(int(r[0]) if len(r) else 16)
    out["v3 학습 (전체)"] = dict(H=H, R=R, dt=0.10)
    rows = [r for r in csv.DictReader(open(f"{REPO}/data_csv/blender_track_v3/voe_test/index.csv")) if r["variant"] == "pos_a"]
    out["v3 test"] = dict(H=[16 - int(r["event_s"]) for r in rows], R=[int(r["event_e"]) + 1 - 16 for r in rows], dt=0.10)
    H, R = [], []
    for s in json.load(open(f"{CLV}/index.json"))["scenes"]:
        H.append(max(0, 16 - s["hidden_clip"][0])); R.append(s["reveal_clip"] - 16)   # hidden_clip = 완전히 가려진 샘플 · reveal_clip = 다시 보이는 샘플
    out["CLEVRER (test)"] = dict(H=H, R=R, dt=0.08)
    rv = rubric_reveal()
    out["Rubric"] = dict(H=[x["H"] for x in rv.values()], R=[x["R"] for x in rv.values()], dt=float(np.median([x["dt"] for x in rv.values()])))
    return out


def timing():
    print("\n#### 3. 재등장 타이밍 분포 (샘플 단위)")
    T = timing_sets(); res = {}
    for k, o in T.items():
        H, R = np.array(o["H"]), np.array(o["R"]); t = (H + R) * o["dt"]
        res[k] = dict(n=len(H), dt=o["dt"], H_med=float(np.median(H)), H_min=int(H.min()), H_max=int(H.max()), R_med=float(np.median(R)),
                      R_min=int(R.min()), R_max=int(R.max()), hid_s_med=float(np.median(t)), hid_s_min=float(t.min()), hid_s_max=float(t.max()),
                      H_hist=np.bincount(H, minlength=16)[:16].tolist(), R_hist=np.bincount(np.clip(R, 0, 15), minlength=16)[:16].tolist())
        x = res[k]
        print(f"   {k:18s} n {x['n']:5d} · 간격 {x['dt']:.2f}s · H 중앙 {x['H_med']:.0f} ({x['H_min']}–{x['H_max']}) · R 중앙 {x['R_med']:.0f} ({x['R_min']}–{x['R_max']})"
              f" · 가림 길이 중앙 {x['hid_s_med']:.2f}s ({x['hid_s_min']:.2f}–{x['hid_s_max']:.2f})")
    RES["timing"] = res


# ---------------------------------------------------------------- 4. 같은 타이밍 대조
def matched():
    print("\n#### 4. 같은 타이밍 대조 — Rubric 영상과 (H, R) 가 ±1 샘플 안인 v3 test block 의 A 정답률 = 기대값")
    idx = {r["block_id"]: r for r in csv.DictReader(open(f"{REPO}/data_csv/blender_track_v3/voe_test/index.csv")) if r["variant"] == "pos_a"}
    _, g, S = load(BENCH["v3test"])
    byHR = collections.defaultdict(list)
    for b, q in g.items():
        if b in idx:
            byHR[(16 - int(idx[b]["event_s"]), int(idx[b]["event_e"]) + 1 - 16)].append(q)
    meta, gr, Sr = load(BENCH["rubric"])
    rsrc = {meta["video_id"][q["pos_a"]].rsplit("_", 2)[0]: q for q in gr.values()}
    rv = rubric_reveal(); keep, exp = [], {m: [] for m in M}
    for name, x in rv.items():
        qs = [q for (h, r), L in byHR.items() if abs(h - x["H"]) <= 1 and abs(r - x["R"]) <= 1 for q in L]
        if len(qs) >= 20:
            keep.append(name)
            for m in M:
                exp[m].append(100 * np.mean([margins(S, m, q)[0] > 0 for q in qs]))
    res = {"n_video": len(keep), "videos": keep}
    for m in M:
        act = 100 * np.mean([margins(Sr, m, rsrc[n])[0] > 0 for n in keep])
        res[m] = dict(expected=float(np.mean(exp[m])), actual=float(act))
        print(f"   {NM[m]:11s} 기대 A {res[m]['expected']:5.1f} · Rubric 실제 A {act:5.1f}   (영상 {len(keep)})")
    RES["matched"] = res


# ---------------------------------------------------------------- 5. R 별 정답률 · 문맥 효과
def bytiming():
    print("\n#### 5. v3 test — 다시 보이는 미래 샘플 R 별 A 정답률 · c/S")
    idx = {r["block_id"]: r for r in csv.DictReader(open(f"{REPO}/data_csv/blender_track_v3/voe_test/index.csv")) if r["variant"] == "pos_a"}
    _, g, S = load(BENCH["v3test"]); by = collections.defaultdict(list)
    for b, q in g.items():
        if b in idx:
            r = int(idx[b]["event_e"]) + 1 - 16; by[r if r <= 6 else (7 if r <= 8 else 9)].append(q)
    res = {}
    for k in sorted(by):
        qs = by[k]; res[k] = {}
        for m in M:
            mA, mB, S0 = map(np.array, zip(*[margins(S, m, q) for q in qs]))
            res[k][m] = dict(n=len(qs), A=100 * np.mean(mA > 0), c=100 * np.mean((mA + mB) / S0))
        lab = {7: "7–8", 9: "9+"}.get(k, str(k))
        print(f"   R {lab:>3s} (n {len(qs):3d}) " + " | ".join(f"{NM[m]} A {res[k][m]['A']:5.1f} c {res[k][m]['c']:+5.2f}" for m in M))
    RES["bytiming_v3test"] = res


# ---------------------------------------------------------------- 6. 물체 기대 곡선
def expect():
    print("\n#### 6. 물체 기대 곡선 — PFT 가 더한 E_j (E_j(PFT) − E_j(릴리즈), %), 열 = 미래 블록 0..7")
    E = lambda S, m, q: (S[m][q["imp_ab"]] - S[m][q["pos_b"]]) / S[m][q["pos_b"]] * 100
    grp = lambda blk: "다시 보임 블록 0–1" if blk <= 1 else ("다시 보임 블록 2–3" if blk <= 3 else "다시 보임 블록 4+")
    idx = {r["block_id"]: r for r in csv.DictReader(open(f"{REPO}/data_csv/blender_track_v3/voe_test/index.csv")) if r["variant"] == "pos_a"}
    cj = {s["scene_id"]: s for s in json.load(open(f"{CLV}/index.json"))["scenes"]}
    rv = rubric_reveal(); res = {}
    sets = [("v3test", lambda meta, b, q: (int(idx[b]["event_e"]) + 1 - 16) // 2 if b in idx else None),
            ("clevrer", lambda meta, b, q: (cj[meta["video_id"][q["pos_a"]].split("_")[0]]["reveal_clip"] - 16) // 2),
            ("rubric", lambda meta, b, q: rv[meta["video_id"][q["pos_a"]].rsplit("_", 2)[0]]["R"] // 2)]
    for key, blk in sets:
        meta, g, S = load(BENCH[key]); by = collections.defaultdict(list)
        for b, q in g.items():
            k = blk(meta, b, q)
            if k is not None:
                by[grp(k)].append(q); by["전체"].append(q)
        res[key] = {}
        print(f"== {key}")
        for k in sorted(by):
            res[key][k] = {}
            for m in ("vobftg", "v3_full_s0"):
                d = np.array([E(S, m, q) - E(S, "release", q) for q in by[k]])
                bs = d[np.random.default_rng(0).integers(0, len(d), (5000, len(d)))].mean(1)
                lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
                res[key][k][m] = dict(n=len(d), mean=d.mean(0).tolist(), lo=lo.tolist(), hi=hi.tolist())
                print(f"   {k:16s} (n {len(d):4d}) {NM[m]:11s} " + " ".join(f"{v:+5.2f}" for v in d.mean(0)) + f"   평균 {d.mean():+.2f}")
    # 학습셋: 블록 j 까지 다시 보인 비율 = P(R ≤ 2j+1)
    T = timing_sets(); cdf = {}
    for k in ("VOBench PFT 학습", "v3 학습 (전체)"):
        R = np.array(T[k]["R"]); cdf[k] = [100 * float(np.mean(R <= 2 * j + 1)) for j in range(8)]
        print(f"   학습셋 {k}: 블록 j 까지 다시 보인 비율 " + " ".join(f"{v:5.1f}" for v in cdf[k]))
    res["train_cdf"] = cdf
    RES["expect"] = res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default="decomp,pixel,timing,matched,bytiming,expect"); a = ap.parse_args()
    for f in a.only.split(","):
        globals()[f]()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    old = json.load(open(OUT)) if os.path.exists(OUT) else {}
    old.update(RES)
    json.dump(old, open(OUT, "w"), ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
