#!/usr/bin/env python3
"""transfer_gap R4 검토 [R] — 사후 검사 셋 (BOARD "[R] R4 — 사전 등록" 에 판정 규칙을 먼저 적음). 2026-10-30.

CPU 만, 이 셸에서 돈다 (배열은 수십 MB). 새 학습 · GPU 없음. 다른 에이전트의 산출물은 읽기만 한다.

  n1  E38 막기의 판독 하락: 막은 p 로 판독기를 다시 맞추면 (train block) test block 정확도가 돌아오나.
      입력 results/r4_x1/E38/ko_r0.npz (feat__<model>__<cond> (240, 8, 1280) · feat__split · feat__task · feat__label · feat__block).
      절차는 transfer_gap_r4_ko.py summarize 의 fit 과 같다 (능형 → 원핫, λ ∈ {1e1..1e4} 를 train block 5 겹으로).
  n4  E37 경로 세기 (obj/U, 층 L5–L11) 와 사라짐 A 정답의 block 수준 짝: AUC + block bootstrap.
      입력 results/r4_x1/E37/v11*_r*.npz · results/r4_x1/te/v11sub3/{meta.json,l1.npy}.
  n6  X1 (vll4) · X2 (vll3) 두 통과의 IntPhys 1 · 2 쌍 판정 일치 (릴리즈 · 원 PFT).
  recount  E39 · X2 표 · 성질별 분해 · 쌍 전이 (사전 등록 전에 돌린 것 포함 — BOARD 에 기록함).

재현:
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/transfer_gap_review_r4.py n1 n4 n6 recount   # → z_analysis/transfer_gap/results/review_r4/*.json
"""
from __future__ import annotations

import collections
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
TG = REPO / "z_analysis/transfer_gap/results"
OUT = TG / "review_r4"
NB = 2000


def boot_mean(x, blocks, nb=NB, seed=0):
    ub, inv = np.unique(blocks, return_inverse=True)
    s = np.bincount(inv, x, len(ub)); n = np.bincount(inv, None, len(ub))
    ii = np.random.default_rng(seed).integers(0, len(ub), (nb, len(ub)))
    b = s[ii].sum(1) / n[ii].sum(1)
    return [float(100 * s.sum() / n.sum()), float(100 * np.percentile(b, 2.5)), float(100 * np.percentile(b, 97.5))]


# ───────────────────────── n1
def ridge_fit(X, Y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    return mu, sd, np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ Y)


def ridge_pred(m, X):
    return ((X - m[0]) / m[1]) @ m[2]


def n1():
    z = np.load(TG / "r4_x1/E38/ko_r0.npz", allow_pickle=True)
    task, split, lab, fb = z["feat__task"], z["feat__split"], z["feat__label"], z["feat__block"]
    res = {}
    for tk in sorted(set(task)):
        sel = task == tk
        classes = sorted(set(lab[sel])); ci = {c: i for i, c in enumerate(classes)}

        def rows(cn, which):
            F = z[f"feat__{cn}"]
            X, y, b = [], [], []
            for i in np.nonzero(sel & (split == which))[0]:
                for t in range(F.shape[1]):
                    if not np.isnan(float(F[i, t, 0])):
                        X.append(F[i, t].astype(np.float32)); y.append(ci[lab[i]]); b.append(fb[i])
            return np.array(X, np.float64), np.array(y), np.array(b)

        def fit(cn):
            X, y, b = rows(cn, "train")
            Y = np.eye(len(classes))[y]
            ub = np.unique(b); rng = np.random.default_rng(0); fold = {u: f for u, f in zip(ub, rng.permutation(len(ub)) % 5)}
            fo = np.array([fold[u] for u in b]); best = None
            for lam in (1e1, 1e2, 1e3, 1e4):
                ok = [(ridge_pred(ridge_fit(X[fo != f], Y[fo != f], lam), X[fo == f]).argmax(1) == y[fo == f]).mean() for f in range(5)]
                if best is None or np.mean(ok) > best[0]:
                    best = (np.mean(ok), lam)
            return ridge_fit(X, Y, best[1]), best[1]

        r = {"chance": 100 / len(classes), "n_train_block": int(len(set(fb[sel & (split == 'train')]))),
             "n_test_block": int(len(set(fb[sel & (split == 'test')])))}
        m_clean = fit("vobftg__clean"); m_rel = fit("release__clean")
        for cn, mm, nm in (("vobftg__clean", m_clean, "clean_on_clean"), ("vobftg__ko_obj", m_clean, "clean_on_ko"),
                           ("release__clean", m_rel, "release_self")):
            X, y, b = rows(cn, "test")
            r[nm] = boot_mean((ridge_pred(mm[0], X).argmax(1) == y).astype(float), b) + [mm[1]]
        for cn, nm in (("vobftg__ko_obj", "ko_refit"), ("vobftg__ko_ctl", "koctl_refit")):
            mm = fit(cn); X, y, b = rows(cn, "test")
            r[nm] = boot_mean((ridge_pred(mm[0], X).argmax(1) == y).astype(float), b) + [mm[1]]
        gap = r["clean_on_clean"][0] - r["release_self"][0]
        drop = r["clean_on_clean"][0] - r["clean_on_ko"][0]
        drop_refit = r["clean_on_clean"][0] - r["ko_refit"][0]
        r.update(gap=gap, drop=drop, drop_refit=drop_refit, refit_share=(drop_refit / drop if drop > 0 else None))
        res[tk] = r
    elig = [tk for tk, r in res.items() if r["gap"] > 0 and r["drop"] >= r["gap"] / 3]
    small = [tk for tk in elig if res[tk]["drop_refit"] < res[tk]["drop"] / 3]
    big = [tk for tk in elig if res[tk]["drop_refit"] >= 2 * res[tk]["drop"] / 3]
    v = ("막기 하락은 주로 형식 변화, 정체 정보는 남는다" if len(small) >= 2 else
         "정보 감소가 선다" if len(big) >= 2 else "부분")
    out = {"tasks": res, "eligible": elig, "refit_small": small, "refit_big": big, "verdict": v}
    json.dump(out, open(OUT / "N1_e38_refit.json", "w"), indent=1, ensure_ascii=False)
    for tk, r in res.items():
        print(f"{tk:16s} clean {r['clean_on_clean'][0]:5.1f} · clean판독기→막음 {r['clean_on_ko'][0]:5.1f} · 막음 재적합 {r['ko_refit'][0]:5.1f} "
              f"[{r['ko_refit'][1]:.0f},{r['ko_refit'][2]:.0f}] · 대조막음 재적합 {r['koctl_refit'][0]:5.1f} · 릴리즈 {r['release_self'][0]:5.1f} "
              f"· 원 하락 {r['drop']:.1f} · 재적합 하락 {r['drop_refit']:.1f}")
    print("판정:", v, "| 대상", elig, "| 재적합 하락 < 1/3:", small, "| ≥ 2/3:", big)


# ───────────────────────── n4
def n4():
    E = TG / "r4_x1/E37"
    Z = {}
    for f in sorted(glob.glob(str(E / "v11_r*.npz"))):
        zz = np.load(f)
        for k in zz.files: Z.setdefault(k, []).append(zz[k])
    Z = {k: np.concatenate(v) for k, v in Z.items()}
    for f in sorted(glob.glob(str(E / "v11_ip2*_r*.npz"))):
        zz = np.load(f); assert (zz["vid"] == Z["vid"]).all()
        for k in zz.files:
            if "__" in k and not k.startswith("release__"): Z[k] = zz[k]
    band = list(range(5, 12)); U = 9 / 4096
    ok = Z["keyok"] & Z["qvis"][:, :, None]
    ok3 = np.broadcast_to(ok[:, :, None, :], (ok.shape[0], ok.shape[1], len(band), ok.shape[2]))
    m = json.load(open(TG / "r4_x1/te/v11sub3/meta.json")); l1 = np.load(TG / "r4_x1/te/v11sub3/l1.npy").astype(np.float64).mean(-1)
    tags = m["pred_tags"]; vid = m["video_id"]
    by = collections.defaultdict(dict)
    for i in range(m["n"]):
        by[(m["block_id"][i], m["pair_id"][i])][m["variant"][i]] = i
    pos_of = {}
    for (b, pid), v in by.items():
        if "pos_a" in v and "imp_ab" in v:
            pos_of[vid[v["pos_a"]]] = (v["pos_a"], v["imp_ab"])
    def objU(name):
        obj = Z[f"{name}__obj"].astype(np.float64)[:, :, band]
        return np.where(ok3, obj, 0).sum((1, 2, 3)) / np.maximum(ok3.sum((1, 2, 3)), 1) / U

    def auc(x, y):
        a, b = x[y == 1], x[y == 0]
        if len(a) == 0 or len(b) == 0: return np.nan
        return float(((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean()))

    def strat_auc(x, y, st):
        num = den = 0.0
        for s in np.unique(st):
            k_ = st == s
            a_ = auc(x[k_], y[k_])
            if np.isnan(a_): continue
            num += a_ * k_.sum(); den += k_.sum()
        return num / den if den else np.nan

    def ci(fn, n, seed=0):
        rng = np.random.default_rng(seed); bs = []
        for _ in range(NB):
            ii = rng.integers(0, n, n); bs.append(fn(ii))
        bs = np.array(bs); bs = bs[~np.isnan(bs)]
        return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if len(bs) else [np.nan, np.nan]

    keep = np.array([v in pos_of for v in Z["vid"]])
    cells = np.array([c for c in Z["cell"]])[keep]
    ou_rel = objU("release")[keep]
    res = {}
    for name in ("ip2e80", "ip2pair", "vobftg", "ip1late_f0", "v11pft"):
        ou = objU(name)[keep]
        k = tags.index(name)
        y = np.array([float(l1[pos_of[v][1], k] > l1[pos_of[v][0], k]) for v in Z["vid"][keep]])
        x = ou
        r = dict(n=int(len(x)), n_correct=int(y.sum()), A=float(100 * y.mean()))
        if 0 < y.sum() < len(y):
            r.update(auc=auc(x, y), ci=ci(lambda ii: auc(x[ii], y[ii]), len(x)),
                     auc_strat=strat_auc(x, y, cells), ci_strat=ci(lambda ii: strat_auc(x[ii], y[ii], cells[ii]), len(x)),
                     auc_release_objU=auc(ou_rel, y), ci_release_objU=ci(lambda ii: auc(ou_rel[ii], y[ii]), len(x)),
                     auc_delta=auc(x - ou_rel, y), ci_delta=ci(lambda ii: auc((x - ou_rel)[ii], y[ii]), len(x)),
                     objU_correct=float(np.median(x[y == 1])), objU_wrong=float(np.median(x[y == 0])))
            print(f"{name:11s} n {len(x)} 맞힘 {int(y.sum())} (A {100 * y.mean():.1f}) · AUC {r['auc']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]"
                  f" · 층 AUC {r['auc_strat']:.3f} [{r['ci_strat'][0]:.3f}, {r['ci_strat'][1]:.3f}]"
                  f" · 릴리즈 obj/U 로 AUC {r['auc_release_objU']:.3f} [{r['ci_release_objU'][0]:.3f}, {r['ci_release_objU'][1]:.3f}]"
                  f" · Δobj/U AUC {r['auc_delta']:.3f} [{r['ci_delta'][0]:.3f}, {r['ci_delta'][1]:.3f}]"
                  f" · obj/U 중앙 맞힘 {r['objU_correct']:.2f} 틀림 {r['objU_wrong']:.2f}")
        else:
            print(f"{name:11s} n {len(x)} 맞힘 {int(y.sum())} — 한쪽 class 가 없어 AUC 없음")
        res[name] = r
    two = ("ip2e80", "ip2pair")
    lo = [res[n]["ci"][0] > 0.5 for n in two]
    inc = [res[n]["ci"][0] <= 0.5 <= res[n]["ci"][1] for n in two]
    res["verdict_N4"] = "block 수준에서도 같이 간다" if all(lo) else "모델 사이 일치뿐, block 수준 표지는 아니다" if all(inc) else "섞임"
    res["verdict_N4b"] = "층을 맞춰도 선다" if all(res[n]["ci_strat"][0] > 0.5 for n in two) else "층 구성 탓일 수 있다"
    geo = [res[n]["ci_release_objU"][0] > 0.5 and res[n]["ci_release_objU"][1] >= res[n]["ci"][0] for n in two]
    own = [res[n]["ci_release_objU"][0] <= 0.5 <= res[n]["ci_release_objU"][1] for n in two]
    res["verdict_N4c"] = ("연관은 block 기하가 낸다 — 경로 표지로 읽지 않는다" if all(geo) else
                          "모델이 바꾼 경로 몫이 정답과 같이 간다" if all(own) else "섞임")
    json.dump(res, open(OUT / "N4_path_block_auc.json", "w"), indent=1, ensure_ascii=False)
    print("판정 N4:", res["verdict_N4"], "| N4b:", res["verdict_N4b"], "| N4c:", res["verdict_N4c"])


# ───────────────────────── n6
def _judge_pairs(d, cell):
    m = json.load(open(d / "meta.json")); l1 = np.load(d / "l1.npy").astype(np.float64).mean(-1)
    idx = {r["video_id"]: r for r in csv.DictReader(open(m["index_csv"]))}
    rows = [idx[v] for v in m["video_id"]]
    by = collections.defaultdict(dict)
    for i, r in enumerate(rows):
        if cell(r): by[(r["block_id"], r["pair_id"])][r["plausible"]] = i
    keys = sorted(k for k, v in by.items() if "1" in v and "0" in v)
    J = {t: np.array([float(l1[by[k]["0"], j] > l1[by[k]["1"], j]) for k in keys]) for j, t in enumerate(m["pred_tags"])}
    return keys, J


def n6():
    res = {}
    for nm, a, b, cell in (("ip1", TG / "r4_x1/te/ip1", TG / "r4_x2/te/ip1uni_all", lambda r: r["group"] == "moving/occluded"),
                           ("ip2", TG / "r4_x1/te/ip2", TG / "r4_x2/te/ip2uni_all", lambda r: r["condition"] == "permanence"),
                           ("ip1_all", TG / "r4_x1/te/ip1", TG / "r4_x2/te/ip1uni_all", lambda r: True),
                           ("ip2_all", TG / "r4_x1/te/ip2", TG / "r4_x2/te/ip2uni_all", lambda r: True)):
        ka, Ja = _judge_pairs(a, cell); kb, Jb = _judge_pairs(b, cell)
        assert ka == kb
        res[nm] = {t: dict(n=len(ka), agree=float((Ja[t] == Jb[t]).mean()), acc_x1=float(100 * Ja[t].mean()), acc_x2=float(100 * Jb[t].mean()))
                   for t in ("release", "vobftg")}
        print(nm, res[nm])
    json.dump(res, open(OUT / "N6_cross_pass.json", "w"), indent=1, ensure_ascii=False)


# ───────────────────────── recount (사전 등록 전에 돌린 재계산 포함)
def recount():
    out = {}
    te = TG / "r4_x2/te"

    def load(pn):
        m = json.load(open(te / f"{pn}_all/meta.json")); l1 = np.load(te / f"{pn}_all/l1.npy").astype(np.float64).mean(-1)
        cp = np.load(te / f"{pn}_all/copy.npy").astype(np.float64).mean(-1)
        idx = {r["video_id"]: r for r in csv.DictReader(open(m["index_csv"]))}
        S = {t: l1[:, k] for k, t in enumerate(m["pred_tags"])}; S["copy"] = cp
        return [idx[v] for v in m["video_id"]], S
    cells = {"ip1uni": {"move_occ": lambda r: r["group"] == "moving/occluded",
                        **{p: (lambda r, p=p: r["group"] == "moving/occluded" and r["principle"] == p) for p in ("O1", "O2", "O3")}},
             "ip2uni": {"permanence": lambda r: r["condition"] == "permanence"}}
    for pn, cs in cells.items():
        rows, S = load(pn)
        for cn, fn in cs.items():
            by = collections.defaultdict(dict)
            for i, r in enumerate(rows):
                if fn(r): by[(r["block_id"], r["pair_id"])][r["plausible"]] = i
            pr = [(k[0], v["1"], v["0"]) for k, v in by.items() if "1" in v and "0" in v]
            sc = np.array([p[0] for p in pr]); pos = np.array([p[1] for p in pr]); imp = np.array([p[2] for p in pr])
            a = {t: np.where(S[t][imp] > S[t][pos], 1.0, np.where(S[t][imp] == S[t][pos], 0.5, 0.0)) for t in S}
            t0 = (a["t0s0"] + a["t0s1"]) / 2
            d = {"n": len(pr), "acc": {t: float(100 * a[t].mean()) for t in a}}
            for X in ("t1", "t2"):
                g = (a[X] == 1) & (t0 < 1); lo = (a[X] < 1) & (t0 == 1)
                d[X] = {"minus_release": boot_mean(a[X] - a["release"], sc, seed=3), "minus_t0s0": boot_mean(a[X] - a["t0s0"], sc, seed=3),
                        "gained_vs_T0": int(g.sum()), "gained_release_right": int((g & (a["release"] == 1)).sum()), "lost_vs_T0": int(lo.sum())}
            d["t0s0_vs_t0s1_discordant"] = int((a["t0s0"] != a["t0s1"]).sum())
            d["release_vs_t0s0_discordant"] = int((a["release"] != a["t0s0"]).sum())
            out[f"{pn}/{cn}"] = d
            print(pn, cn, json.dumps({k: v for k, v in d.items() if k != "acc"}, ensure_ascii=False), {t: round(v, 1) for t, v in d["acc"].items()})
    json.dump(out, open(OUT / "recount_x2.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for cmd in sys.argv[1:]:
        print(f"\n######## {cmd}")
        {"n1": n1, "n4": n4, "n6": n6, "recount": recount}[cmd]()
