#!/usr/bin/env python3
"""R3-(C): v11 copy-hard 부분집합 독립 재계산 (자체 pairing · 다른 bootstrap seed) + 표적 공간 반론의 정량 검사 (무작위 일치 시뮬레이션) + copy_blk 부분."""
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2"); CACHE = Path("/data2/local_datasets/world/world_analysis/cache/training_effects"); OUT = ROOT / "auto_research/_verify_scratch/R3/out"; OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20260929)
d = CACHE / "v11score_vith_curves"; meta = json.load(open(d / "meta.json")); tags = meta["pred_tags"]
l1 = np.load(d / "l1.npy"); cp = np.load(d / "copy.npy"); vid = meta["video_id"]
S = {t: l1[:, k].mean(1) for k, t in enumerate(tags) if t in ("release", "ariel_ep43", "v11_e10", "pv1_e15")}; S["copy"] = cp.mean(1)
for tag, folder in (("prefix_ep45", "vith_ariel_prefix_ep45"), ("full_ep40", "vith_ariel_full_ep40"), ("ar_ep38", "vith_ariel_ar_ep38"), ("ar_ep18", "vith_ariel_ar_ep18")):
    pv = json.load(open(ROOT / f"z_research/IntPhysGenV11/exp_results/surprise_c16t32__v11_split_test_{folder}/per_block.json"))["per_video_surprise"]
    S[tag] = np.array([pv.get(v, np.nan) for v in vid], np.float64)
# pairing (matched: 같은 block · 같은 pair_id, plausible 1 vs 0)
grp = defaultdict(dict)
for i, (b, p, pl) in enumerate(zip(meta["block_id"], meta["pair_id"], meta["plausible"])):
    grp[(str(b), str(p))]["pos" if str(pl) == "1" else "imp"] = i
pairs = [(k, g["pos"], g["imp"]) for k, g in grp.items() if "pos" in g and "imp" in g]
pi = np.array([p for _, p, _ in pairs]); ii = np.array([i for _, _, i in pairs]); blk = np.array([k[0] for k, _, _ in pairs])
rows = {r["video_id"]: r for r in csv.DictReader(open(ROOT / "data_csv/intphysgen_v11_split/index_test.csv"))}
tim = np.array([(rows[vid[p]]["occ_timing"] or "visible") for p in pi]); mot = np.array([rows[vid[p]]["motion"] for p in pi])
vio = np.array([("vanish→빈" if rows[vid[p]]["role"].startswith("pos_obj") else "vanish→물체") if rows[vid[p]]["violation_type"] == "vanish" else rows[vid[p]]["violation_type"] for p in pi])
O = {m: np.where(S[m][ii] - S[m][pi] > 0, 1.0, np.where(S[m][ii] == S[m][pi], 0.5, 0.0)) for m in S}
V = {m: np.isfinite(S[m][pi]) & np.isfinite(S[m][ii]) for m in S}


def boot(v, cl, B=2000):
    u, inv = np.unique(cl, return_inverse=True); s = np.bincount(inv, v, len(u)); c = np.bincount(inv, None, len(u)).astype(float)
    idx = rng.integers(0, len(u), (B, len(u))); bs = s[idx].sum(1) / c[idx].sum(1)
    return [round(100 * float(v.mean()), 2), round(100 * float(np.percentile(bs, 2.5)), 2), round(100 * float(np.percentile(bs, 97.5)), 2), int(len(v))]


hard = O["copy"] < 1; corr = O["copy"] == 1
res = dict(n_pair=len(pairs), n_hard=int(hard.sum()))
res["all"] = {m: boot(O[m][V[m]], blk[V[m]]) for m in O}
res["copy_hard"] = {m: boot(O[m][hard & V[m]], blk[hard & V[m]]) for m in O}
res["copy_correct"] = {m: boot(O[m][corr & V[m]], blk[corr & V[m]]) for m in O}
res["agreement_with_copy"] = {m: round(float(np.mean((O[m] == O["copy"])[V[m]])), 3) for m in O}
res["late_hard"] = {m: boot(O[m][hard & (tim == "late") & V[m]], blk[hard & (tim == "late") & V[m]]) for m in O}
res["late_vanish_to_empty_cells"] = {mo: {m: boot(O[m][(tim == "late") & (mot == mo) & (vio == "vanish→빈") & V[m]], blk[(tim == "late") & (mot == mo) & (vio == "vanish→빈") & V[m]]) for m in O} for mo in ("static", "moving_flat", "moving")}
# ---- 표적 공간 반론 시뮬레이션: 전체 정확도 acc 인 predictor 가 복사와 무작위로 일치하면 copy-hard 정확도는?
# 모형: 쌍마다 독립 Bernoulli(acc). copy-hard 는 복사가 틀리는 쌍 — 독립이면 E[acc|hard] = acc. 즉 '무작위 일치' 귀무에서 copy-hard 정확도 = 전체 정확도.
# 더 강한 검사: 복사와의 실제 일치율 a_m 을 유지한 채 (copy 와 predictor 의 2×2 표 고정) 어느 쌍이 hard 인지만 셔플 → copy-hard 정확도의 귀무 분포
sim = {}
for m in ("release", "prefix_ep45", "full_ep40", "ar_ep38", "ar_ep18", "v11_e10"):
    v = V[m]; om, oc = O[m][v], O["copy"][v]; b = blk[v]; obs = float(om[oc < 1].mean())
    null = []
    for _ in range(2000):
        perm = rng.permutation(len(oc)); null.append(float(om[oc[perm] < 1].mean()))
    # (이 셔플은 predictor 결과와 copy 결과의 짝을 끊어 '무작위 일치' 귀무를 만든다 → 귀무 하 copy-hard 정확도 ≈ 전체 정확도)
    sim[m] = dict(observed_copy_hard=round(100 * obs, 2), null_mean=round(100 * float(np.mean(null)), 2), null_95=[round(100 * float(np.percentile(null, 2.5)), 2), round(100 * float(np.percentile(null, 97.5)), 2)],
                  overall=round(100 * float(om.mean()), 2), agreement=round(float(np.mean(om == oc)), 3), n=int(len(om)))
res["random_agreement_null"] = sim
# ---- copy_blk 부분 (360 영상)
d2 = CACHE / "v11score_vith_ar_copyblk"; m2 = json.load(open(d2 / "meta.json")); done2 = np.load(d2 / "done.npy").astype(bool)
Sar = np.load(d2 / "l1.npy")[:, 0].mean(1); Scb = np.load(d2 / "copy_blk.npy").mean(1); idx2 = {v: i for i, v in enumerate(m2["video_id"])}
ok = np.array([done2[idx2[vid[p]]] and done2[idx2[vid[i]]] for p, i in zip(pi, ii)])
p2 = np.array([idx2[vid[p]] for p in pi[ok]]); i2 = np.array([idx2[vid[i]] for i in ii[ok]])
o_ar = np.where(Sar[i2] - Sar[p2] > 0, 1.0, np.where(Sar[i2] == Sar[p2], 0.5, 0.0)); o_cb = np.where(Scb[i2] - Scb[p2] > 0, 1.0, np.where(Scb[i2] == Scb[p2], 0.5, 0.0))
hb = o_cb < 1; hs = hard[ok]
sub = dict(n=int(ok.sum()), n_cb_hard=int(hb.sum()), n_std_hard=int(hs.sum()),
           all={"copy_blk": boot(o_cb, blk[ok]), "ar_ep18_arr": boot(o_ar, blk[ok]), "ar_ep18_harness": boot(O["ar_ep18"][ok], blk[ok]), "ar_ep38_harness": boot(O["ar_ep38"][ok], blk[ok]), "copy_std": boot(O["copy"][ok], blk[ok]), "release": boot(O["release"][ok], blk[ok])},
           copy_blk_hard={"ar_ep18_arr": boot(o_ar[hb], blk[ok][hb]), "ar_ep38_harness": boot(O["ar_ep38"][ok][hb], blk[ok][hb]), "release": boot(O["release"][ok][hb], blk[ok][hb]), "copy_std": boot(O["copy"][ok][hb], blk[ok][hb])},
           copy_std_hard={"ar_ep18_arr": boot(o_ar[hs], blk[ok][hs]), "ar_ep38_harness": boot(O["ar_ep38"][ok][hs], blk[ok][hs]), "release": boot(O["release"][ok][hs], blk[ok][hs]), "copy_blk": boot(o_cb[hs], blk[ok][hs])},
           both_hard={"ar_ep18_arr": boot(o_ar[hb & hs], blk[ok][hb & hs]) if (hb & hs).sum() else None, "release": boot(O["release"][ok][hb & hs], blk[ok][hb & hs]) if (hb & hs).sum() else None, "n": int((hb & hs).sum())},
           agreement_cb_std=round(float(np.mean(o_cb == O["copy"][ok])), 3), agreement_ar_arr_vs_harness=round(float(np.mean(o_ar == O["ar_ep18"][ok])), 3),
           composition_cb_hard={t: int((hb & (tim[ok] == t)).sum()) for t in ("visible", "early", "mid", "late")})
res["copy_blk_partial"] = sub
json.dump(res, open(OUT / "c_v11.json", "w"), indent=1)
for k in ("all", "copy_hard", "copy_correct", "late_hard"): print(k, {m: v[:3] for m, v in res[k].items()})
print("agreement", res["agreement_with_copy"]); print("late vanish→빈", {mo: {m: v[0] for m, v in d_.items()} for mo, d_ in res["late_vanish_to_empty_cells"].items()})
print("null", json.dumps(sim, ensure_ascii=False)); print("copy_blk partial", json.dumps(sub, ensure_ascii=False))
