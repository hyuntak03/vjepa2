#!/usr/bin/env python3
"""TrainingEffects — v11 있음 (presence) 읽기: 영속 (permanence) · 지속 (persistence), predictor 별 + 학습 곡선 (CPU).

입력: `te_v11_readout.py` 의 curves 추출 (`cache/training_effects/v11readout_curves/`, `load()` 로 읽는다). 수정하지 않는다.
출력: `z_research/TrainingEffects/v11_presence/` — PRESENCE.md (본문) · TABLES.md (전 칸 표) · presence.json · fig_*.png

정의 (전부 README `z_research/TrainingEffects/README.md` §1 과 `audit_v11_presence.py` 를 따른다):
  읽기 (reading) — 자 (ruler) 는 릴리즈 특징으로 학습·보정한 것 그대로, 문턱은 자마다 하나 (thr_val_fpr5):
      p@h = h 자로 p 를 읽음 (주 판독, 문턱 −0.961) · p@p = 릴리즈 p 자 (보조, 이식 판독, 0.8307)
      h@h · z@z = predictor 와 무관한 자 검사 (미래 튜블릿 8..15 = p 의 t0..t7). z 는 z_full (32 샘플을 다 본 encoder)
      복사 (copy) = 문맥만 본 z · h 의 마지막 문맥 튜블릿 (idx 7) 을 z 자 · h 자로 읽어 미래 8 튜블릿 전부의 값으로 둔 것
                    (`RollOutV3/audit/v11_presence/ctx_only_readings.npz`, 읽기 전용)
  '있다' = logit > 문턱.  빈 장면 바닥 (floor) = 같은 칸 (timing × motion × k) 의 obj=False clip (vanish pos_b) 의 '있다' 비율.
  AUROC = P(logit_obj > logit_empty) (+0.5 동점), 문턱 없음.  mass3 = 진실 칸 3×3 attention 질량 (균등 9/256 = 0.035).
  칸 (cell) = timing (visible k=0 / mid k≥2 / late k≥2) × motion (static / flat / ramp), 가능 변이만.
  영속 요약 = late · flat/ramp, t1..t4 평균 (present − floor), AUROC 는 t 마다 계산해 평균.
  지속 요약 = visible · flat/ramp, t3..t7 중 **읽을 수 있는 칸만** (진실이 보이고 · 화면 안 · 가장자리 36 px 안쪽; 튜블릿의
             두 샘플 모두) — (clip, t) 칸 비율. 빈 장면은 같은 궤적의 같은 칸 마스크 (진실 = 물체가 있었다면의 자리).
             ⚠️ v11 에서 이 칸은 flat t3·t4, ramp t3·t4·t5 뿐이다 (그 뒤는 물체가 가장자리 · 화면 밖).
  위치 = '있다' ∧ mass3 > 0.035 인 (clip, t) 만. L2 = |x̂ − 진실| (자 치우침 attn_bias_px 를 뺀다: p@h·h@h → h, p@p → p),
         부호 x 오차 = (x̂ − x_gt)·dir (+ = 앞섬), x 전진 = (x − x_seen)·dir (x_seen = 문맥에서 마지막으로 보인 샘플의 진실).
  CI — 두 단계 cluster bootstrap (2,000 회): 단위 = (궤적, k) (진실 32×2 를 1 px 로 반올림한 해시 × sym_k),
       단위를 복원추출한 뒤 단위 안의 obj clip 과 빈 장면 clip 을 각각 복원추출. **한 셀의 모든 predictor 가 같은 가중치를 쓴다**
       → predictor − release 의 짝지은 차이 CI. vanish A/B 점수는 block bootstrap (block = 문맥을 공유하는 4 clip).
  데이터 = `data_csv/intphysgen_v11_split/index_test.csv` 의 clip 만 (held-out block 절반). sanity 만 전 block.

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/te_analyze_presence.py            # done==1 인 행으로 전부 (추출이 끝났으면 전체)
  $P z_research/scripts/analysis/te_analyze_presence.py --nb 300   # 빠른 개발용
"""
from __future__ import annotations
import argparse, csv, hashlib, json, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))
from te_v11_readout import load  # noqa: E402  (memmap 로더; torch import 는 하지만 GPU 는 안 쓴다)

CACHE = Path("/data2/local_datasets/world/world_analysis/cache/training_effects/v11readout_curves")
OUT = ROOT / "z_research/TrainingEffects/v11_presence"
TEST_CSV = ROOT / "data_csv/intphysgen_v11_split/index_test.csv"
REF_VALUES = ROOT / "z_research/RollOutV3/figures/v11_presence/_superseded/presence_decoder/values.json"
REF_AUDIT = ROOT / "z_research/RollOutV3/audit/v11_presence/results.json"
CTX_ONLY = ROOT / "z_research/RollOutV3/audit/v11_presence/ctx_only_readings.npz"
BIAS_JSON = ROOT / "z_research/RollOutV3/exp_results/presence/attn_bias_px.json"
SCORES_JSON = ROOT / "z_research/TrainingEffects/scores/scores.json"

RES, TC, TP, KMIN, EDGE, W = 144.0, 8, 8, 2, 36.0, 288.0
MASS_U = 9 / 256
THR_TASK = {"p": 0.8307, "h": -0.961, "z": -0.662}
FINAL = ["release", "v11_e10", "ip1_e40", "ip2_e80", "pv1_e15", "ariel_ep43"]
FAMILIES = {
    "v11": [(0, "release"), (1, "v11_e1"), (2, "v11_e2"), (3, "v11_e3"), (5, "v11_e5"), (10, "v11_e10")],
    "pv1": [(0, "release"), (1, "pv1_e1"), (3, "pv1_e3"), (5, "pv1_e5"), (10, "pv1_e10"), (15, "pv1_e15")],
    "ip1": [(0, "release"), (5, "ip1_e5"), (10, "ip1_e10"), (20, "ip1_e20"), (40, "ip1_e40")],
    "ip2": [(0, "release"), (10, "ip2_e10"), (40, "ip2_e40"), (80, "ip2_e80")],
    "ariel": [(5, "ariel_ep5"), (19, "ariel_ep19"), (30, "ariel_ep30"), (43, "ariel_ep43")],
}
FAM_NAME = {"v11": "v11 post-FT", "pv1": "Predictor_v1 post-FT", "ip1": "IntPhys1 post-FT",
            "ip2": "IntPhys2 post-FT", "ariel": "Ariel scratch (block-causal)"}
SCORE_KEY = {"release": "release", "v11_e10": "pft_v11", "ip1_e40": "pft_intphys1", "ip2_e80": "pft_intphys2",
             "pv1_e15": "pft_predv1", "ariel_ep43": "ariel_bc43"}
MOT = {"static": "static", "moving_flat": "flat", "moving": "ramp"}
TIMINGS, MOTIONS = ("visible", "mid", "late"), ("static", "flat", "ramp")
PERM_T, PERS_T = [1, 2, 3, 4], [3, 4, 5, 6, 7]


# ============================================================================== 데이터
def load_all(nb_seed=0):
    M = load(CACHE)
    info = M["info"]
    thr = {r: float(info["thr_val_fpr5"][r]) for r in ("p", "h", "z")}
    for r, v in THR_TASK.items():
        assert abs(thr[r] - v) < 1e-3, f"문턱 불일치 {r}: {thr[r]} vs {v}"
    assert info["decoder_fp"] == "dcd24d8a8a47"
    bias = {r: np.array(v, np.float32) for r, v in json.loads(BIAS_JSON.read_text()).items()}
    tags = [p["tag"] for p in info["preds"]]
    m = M["meta"]
    n = len(m["video_id"])
    done = np.asarray(M["done"]).astype(bool)
    te = {r["video_id"] for r in csv.DictReader(TEST_CSV.open())}
    G = dict(n=n, vid=m["video_id"], done=done, test=np.array([v in te for v in m["video_id"]]),
             tim=m["timing"], mot=np.array([MOT[x] for x in m["motion"]]), k=m["sym_k"].astype(int), obj=m["obj"].astype(bool),
             vtype=m["violation_type"], variant=m["variant"], block=m["block_id"], cond=m["condition"],
             tdir=m["travel_dir"].astype(int))
    Ts = m["truth"] * RES + RES                                             # (n,32,2) px
    T = m["gt_tub_px"].astype(np.float64)                                   # (n,16,2) px (mass3 에 쓴 진실)
    G["Tf"] = T[:, TC:]
    hid = m["hidden"].reshape(n, 16, 2).any(-1); inf = m["in_frame"].reshape(n, 16, 2).all(-1)
    inside = ((T >= EDGE) & (T <= W - EDGE)).all(-1)
    G["visin"] = (~hid & inf & inside)[:, TC:]                              # 읽을 수 있는 칸 (미래 8 튜블릿)
    G["hidf"] = hid[:, TC:]
    hc = m["hidden"][:, :16]
    sseen = np.array([max([s for s in range(16) if not hc[i, s]], default=0) for i in range(n)])
    G["xseen"] = Ts[np.arange(n), sseen]                                    # 문맥에서 마지막으로 보인 샘플의 진실 (px)
    tr = np.round(Ts).astype(np.int32)
    G["traj"] = np.array([hashlib.sha1(t.tobytes()).hexdigest()[:10] for t in tr])
    G["unit"] = np.array([f"{a}|{b}" for a, b in zip(G["traj"], G["k"])])

    G["attn"] = {f"ph:{t}": M[f"attn_ph__{t}"] for t in FINAL if f"attn_ph__{t}" in M}   # (n,8,256) f16 memmap
    G["attn"]["hh"] = M["attn_h"]                                           # (n,16,256) — 미래는 [:, 8:]
    R = {}                                                                  # 읽기 이름 → (arr (n,8,C) float32, 자)
    for t in tags:
        R[f"ph:{t}"] = (np.asarray(M[f"rd_ph__{t}"], np.float32), "h")
        R[f"pp:{t}"] = (np.asarray(M[f"rd_p__{t}"], np.float32), "p")
    R["hh"] = (np.asarray(M["rd_h"][:, TC:], np.float32), "h")
    R["zz"] = (np.asarray(M["rd_z"][:, TC:], np.float32), "z")
    if CTX_ONLY.exists():
        c = np.load(CTX_ONLY, allow_pickle=True)
        assert str(c["decoder_fp"]) == "dcd24d8a8a47"
        pos = {v: i for i, v in enumerate(c["video_id"])}
        ii = np.array([pos[v] for v in G["vid"]])
        for rep in ("z", "h"):
            a = np.asarray(c[rep][ii][:, 7:8], np.float32)                  # (n,1,4) 문맥 튜블릿 7
            a = np.concatenate([a, np.full((n, 1, 1), np.nan, np.float32)], -1)
            R[f"copy_{rep}ctx"] = (np.repeat(a, TP, 1), rep)
    return G, R, thr, bias, tags, info


def cell_mask(G, tim, mo, obj, base):
    m = base & (G["tim"] == tim) & (G["mot"] == mo) & (G["obj"] == obj)
    if tim != "visible":
        m &= G["k"] >= KMIN
    return m


# ============================================================================== 통계 도구
def cluster_weights(uo, ue, nb, rng):
    """두 단계 cluster bootstrap 가중치. uo/ue = obj/빈 clip 의 단위 이름. → Wo (nb,n_o), We (nb,n_e) (복원추출 횟수)."""
    labels = sorted(set(uo) | set(ue))
    go = [np.where(uo == u)[0] for u in labels]; ge = [np.where(ue == u)[0] for u in labels]
    Wo = np.zeros((nb, len(uo)), np.float32); We = np.zeros((nb, len(ue)), np.float32)
    L = len(labels)
    for b in range(nb):
        for p in rng.integers(0, L, L):
            g = go[p]
            if len(g):
                Wo[b] += np.bincount(g[rng.integers(0, len(g), len(g))], minlength=len(uo))
            g = ge[p]
            if len(g):
                We[b] += np.bincount(g[rng.integers(0, len(g), len(g))], minlength=len(ue))
    return Wo, We


def ci(bs):
    bs = np.asarray(bs, float)
    with np.errstate(all="ignore"):
        return np.nanpercentile(bs, [2.5, 97.5], axis=0).T if bs.ndim > 1 else np.nanpercentile(bs, [2.5, 97.5])


def wmean(W, X):
    """W (nb,n) · X (n,...) → (nb,...) 가중 평균."""
    s = W.sum(1)
    with np.errstate(all="ignore"):
        return np.tensordot(W, X, axes=(1, 0)) / s.reshape((-1,) + (1,) * (X.ndim - 1))


def auroc(pos, neg):
    pos = np.asarray(pos, float); neg = np.sort(np.asarray(neg, float))
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    lo = np.searchsorted(neg, pos, "left"); hi = np.searchsorted(neg, pos, "right")
    return float((lo + 0.5 * (hi - lo)).sum() / (len(pos) * len(neg)))


def gmat(pos, neg):
    """G_ij = 1[pos_i > neg_j] + 0.5·1[=]  (n_pos, n_neg)."""
    d = pos[:, None] - neg[None, :]
    return (d > 0).astype(np.float32) + 0.5 * (d == 0).astype(np.float32)


def boot_auroc(Wp, We, pos, neg):
    """가중 AUROC (nb,). Wp (nb,n_pos), We (nb,n_neg)."""
    if len(pos) == 0 or len(neg) == 0:
        return np.full(Wp.shape[0], np.nan)
    G = gmat(pos, neg)
    num = np.einsum("bj,bj->b", Wp @ G, We)
    with np.errstate(all="ignore"):
        return num / (Wp.sum(1) * We.sum(1))


def r3(x):
    if isinstance(x, dict):
        return {k: r3(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [r3(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, str):
        return x
    x = float(x)
    return None if not np.isfinite(x) else round(x, 4)


# ============================================================================== 1. sanity — release p@p 전 block
def sanity(G, R, thr):
    ref = json.loads(REF_VALUES.read_text())
    aud = json.loads(REF_AUDIT.read_text())["cells"] if REF_AUDIT.exists() else {}
    base = G["done"]
    out = {}
    for tim in TIMINGS:
        for mo in MOTIONS:
            m = cell_mask(G, tim, mo, True, base)
            me = cell_mask(G, tim, mo, False, base)
            key = f"{mo}|{'last' if tim == 'late' else tim}"
            e = dict(n=int(m.sum()), n_ref=ref[key]["n"])
            say_p = R["pp:release"][0][..., 2] > thr["p"]
            e["pp_release"] = say_p[m].mean(0)
            e["ref_values"] = np.array(ref[key]["present_frac"])
            e["maxdiff_pp"] = float(np.max(np.abs(e["pp_release"] - e["ref_values"])))
            if key in aud:
                for nm, rk in (("hh", "h"), ("zz", "z")):
                    s = R[nm][0][..., 2] > thr[rk]
                    e[f"maxdiff_{nm}"] = float(np.max(np.abs(s[m].mean(0) - np.array(aud[key][rk], float))))
                    e[f"maxdiff_{nm}_null"] = float(np.max(np.abs(s[me].mean(0) - np.array(aud[key][rk + "_null"], float))))
                e["maxdiff_pp_null"] = float(np.max(np.abs(say_p[me].mean(0) - np.array(aud[key]["p_null"], float))))
            out[f"{tim}|{mo}"] = e
    return out


# ============================================================================== 2. 칸별 튜블릿 곡선
def cell_stats(G, R, thr, readings, base, nb, rng, ref="ph:release"):
    out, W = {}, {}
    for tim in TIMINGS:
        for mo in MOTIONS:
            m = cell_mask(G, tim, mo, True, base); me = cell_mask(G, tim, mo, False, base)
            io, ie = np.where(m)[0], np.where(me)[0]
            uo, ue = G["unit"][io], G["unit"][ie]
            Wo, We = cluster_weights(uo, ue, nb, rng)
            W[f"{tim}|{mo}"] = (io, ie, Wo, We)
            c = {"n_obj": len(io), "n_empty": len(ie), "n_units": len(set(uo)),
                 "frac_readable": G["visin"][io].mean(0), "frac_hidden": G["hidf"][io].mean(0)}
            boots = {}
            for nm in readings:
                arr, rk = R[nm]
                lg = arr[..., 2]; say = (lg > thr[rk]).astype(np.float32)
                so, se = say[io], say[ie]
                bo, be = wmean(Wo, so), wmean(We, se)
                boots[nm] = bo - be
                e = {"present": so.mean(0), "present_ci": ci(bo), "floor": se.mean(0), "floor_ci": ci(be),
                     "diff": so.mean(0) - se.mean(0), "diff_ci": ci(bo - be),
                     "auroc": np.array([auroc(lg[io, t], lg[ie, t]) for t in range(TP)])}
                if arr.shape[-1] >= 5 and np.isfinite(arr[io, :, 4]).any():
                    ms = arr[io, :, 4]
                    e["mass3_med"] = np.nanmedian(ms, 0)
                    e["at_gt"] = ((ms > MASS_U) & (so > 0)).mean(0)          # '있다' ∧ 진실 칸에 attention
                c[nm] = e
            for nm in readings:
                if nm.startswith(("ph:", "pp:")) and nm.split(":")[1] != "release":
                    rn = nm.split(":")[0] + ":release"
                    d = boots[nm] - boots[rn]
                    c[nm]["d_diff"] = c[nm]["diff"] - c[rn]["diff"]; c[nm]["d_diff_ci"] = ci(d)
            out[f"{tim}|{mo}"] = c
    return out, W


# ============================================================================== 3. 영속 · 지속 요약
def summary_one(G, arr, rk, thr, io, ie, Wo, We, tsel, masked):
    lg = arr[..., 2]; say = (lg > thr[rk]).astype(np.float32)
    ts = np.array(tsel)
    if masked:
        mo_, me_ = G["visin"][io][:, ts].astype(np.float32), G["visin"][ie][:, ts].astype(np.float32)
    else:
        mo_, me_ = np.ones((len(io), len(ts)), np.float32), np.ones((len(ie), len(ts)), np.float32)
    so, co = (say[io][:, ts] * mo_).sum(1), mo_.sum(1)
    se, ce = (say[ie][:, ts] * me_).sum(1), me_.sum(1)
    with np.errstate(all="ignore"):
        pres, flo = so.sum() / co.sum(), se.sum() / ce.sum()
        bpres, bflo = (Wo @ so) / (Wo @ co), (We @ se) / (We @ ce)
    aus, baus = [], []
    for j, t in enumerate(ts):
        po, ne = mo_[:, j] > 0, me_[:, j] > 0
        if po.sum() < 5 or ne.sum() < 5:
            continue
        aus.append(auroc(lg[io[po], t], lg[ie[ne], t]))
        baus.append(boot_auroc(Wo[:, po], We[:, ne], lg[io[po], t].astype(float), lg[ie[ne], t].astype(float)))
    au = float(np.mean(aus)) if aus else np.nan
    bau = np.nanmean(np.stack(baus), 0) if baus else np.full(Wo.shape[0], np.nan)
    # 배치 (placement) — 진실 칸 3×3 attention 질량 평균, 진실이 읽을 수 있는 (보이는) 칸만. 문턱 없음. 균등 = 0.035
    if arr.shape[-1] >= 5 and np.isfinite(arr[io][:, ts, 4]).all():
        vm = G["visin"][io][:, ts].astype(np.float32)
        m3s, m3c = (arr[io][:, ts, 4] * vm).sum(1), vm.sum(1)
        with np.errstate(all="ignore"):
            mass, bmass = m3s.sum() / m3c.sum(), (Wo @ m3s) / (Wo @ m3c)
    else:
        mass, bmass = np.nan, np.full(Wo.shape[0], np.nan)
    # 배치 귀무 (적대 검증 2026-09-25) — 물체가 한 번도 없던 빈 장면 (vanish pos_b) 에서 '물체가 있었다면의 진실 칸' 3×3 질량.
    # 문맥과 무관한 '가림막 출구에서 물체가 나온다' prior 를 잰다. 순 배치 = 배치 − 귀무
    if arr.shape[-1] >= 5 and len(ie) and np.isfinite(arr[ie][:, ts, 4]).all():
        ve = G["visin"][ie][:, ts].astype(np.float32)
        n3s, n3c = (arr[ie][:, ts, 4] * ve).sum(1), ve.sum(1)
        with np.errstate(all="ignore"):
            mnull, bnull = n3s.sum() / n3c.sum(), (We @ n3s) / (We @ n3c)
    else:
        mnull, bnull = np.nan, np.full(Wo.shape[0], np.nan)
    # 단위별 값 (단위 = (궤적, k))
    uv = {}
    for u in sorted(set(G["unit"][io])):
        a = G["unit"][io] == u; b = G["unit"][ie] == u
        with np.errstate(all="ignore"):
            uv[u] = (so[a].sum() / co[a].sum()) - (se[b].sum() / ce[b].sum() if b.any() else np.nan)
    return dict(present=pres, floor=flo, diff=pres - flo, auroc=au, mass=mass, mass_null=mnull, mass_net=mass - mnull,
                n_cells_obj=int(co.sum()), n_cells_emp=int(ce.sum())), \
        dict(present=bpres, floor=bflo, diff=bpres - bflo, auroc=bau, mass=bmass, mass_null=bnull, mass_net=bmass - bnull), uv


def summaries(G, R, thr, W, readings, nb=2000, rng=None):
    spec = {"permanence": ("late", PERM_T, False), "persistence": ("visible", PERS_T, True)}
    out = {}
    QS = ("present", "floor", "diff", "auroc", "mass", "mass_null", "mass_net")
    for sname, (tim, tsel, masked) in spec.items():
        for mo in ("flat", "ramp"):
            io, ie, Wo, We = W[f"{tim}|{mo}"]
            key = f"{sname}|{mo}"; res, boots, units = {}, {}, {}
            for nm in readings:
                arr, rk = R[nm]
                pt, bt, uv = summary_one(G, arr, rk, thr, io, ie, Wo, We, tsel, masked)
                boots[nm] = bt; units[nm] = uv
                res[nm] = {**pt, **{f"{q}_ci": ci(bt[q]) for q in QS}, "unit_diff": uv}
            for nm in readings:
                if nm.startswith(("ph:", "pp:")) and nm.split(":")[1] != "release":
                    rn = nm.split(":")[0] + ":release"
                    for q in ("present", "diff", "auroc", "mass", "mass_null", "mass_net"):
                        res[nm][f"d_{q}"] = res[nm][q] - res[rn][q]
                        res[nm][f"d_{q}_ci"] = ci(boots[nm][q] - boots[rn][q])
                    ud = {u: units[nm][u] - units[rn][u] for u in units[nm]}
                    res[nm]["unit_d_diff"] = ud
                    vals = np.array([v for v in ud.values() if np.isfinite(v)])
                    res[nm]["units_pos"] = int((vals > 0).sum()); res[nm]["units_neg"] = int((vals < 0).sum())
                    res[nm]["units_n"] = len(vals)
            # 궤적만 cluster (적대 검증 2026-09-25): (궤적,k) 단위는 궤적 해시 2 개에서 나온다 — k 단위는 독립이 아니다.
            # FINAL 의 p@h Δ 만 궤적 단위 (2 개) 두 단계 bootstrap 으로 다시 잰다 (넓은 쪽이 정직하다)
            if rng is not None:
                Wo2, We2 = cluster_weights(G["traj"][io], G["traj"][ie], nb, rng)
                bt2 = {}
                for tg in FINAL:
                    arr, rk = R[f"ph:{tg}"]
                    bt2[tg] = summary_one(G, arr, rk, thr, io, ie, Wo2, We2, tsel, masked)[1]
                for tg in FINAL[1:]:
                    for q in ("diff", "auroc", "mass_net"):
                        res[f"ph:{tg}"][f"d_{q}_ci_traj"] = ci(bt2[tg][q] - bt2["release"][q])
            out[key] = dict(timing=tim, tubelets=tsel, readable_only=masked, n_obj=len(io), n_empty=len(ie),
                            n_units=len(set(G["unit"][io])), readings=res)
    return out


# ============================================================================== 5. 위치
def position(G, R, thr, bias, W, readings):
    out = {}
    for tim in ("visible", "late"):
        for mo in ("flat", "ramp"):
            io, ie, Wo, We = W[f"{tim}|{mo}"]
            Tf, d = G["Tf"][io], G["tdir"][io]
            xs = G["xseen"][io]
            c = {"prog_gt_all": ((Tf[..., 0] - xs[:, None, 0]) * d[:, None]).mean(0)}   # 진실 x 전진, 칸의 모든 obj clip
            for nm in readings:
                arr, rk = R[nm]
                if arr.shape[-1] < 5:
                    continue
                xy = arr[io][..., :2].astype(np.float64) * RES + RES - bias[rk]
                say = arr[io][..., 2] > thr[rk]; ms = arr[io][..., 4]
                q = say & (ms > MASS_U)
                e = {k_: np.full(TP, np.nan) for k_ in ("l2", "err_x", "prog", "prog_gt", "l2_diffuse")}
                e["present"] = say.mean(0); e["at_gt"] = q.mean(0); e["n_at_gt"] = q.sum(0)
                e["diffuse_given_present"] = np.array([(~q[:, t] & say[:, t]).sum() / max(say[:, t].sum(), 1) for t in range(TP)])
                e["at_gt_ci"] = ci(wmean(Wo, q.astype(np.float32)))
                for t in range(TP):
                    s = q[:, t]
                    if s.sum() >= 5:
                        e["l2"][t] = np.linalg.norm(xy[s, t] - Tf[s, t], axis=-1).mean()
                        e["err_x"][t] = ((xy[s, t, 0] - Tf[s, t, 0]) * d[s]).mean()
                        e["prog"][t] = ((xy[s, t, 0] - xs[s, 0]) * d[s]).mean()
                        e["prog_gt"][t] = ((Tf[s, t, 0] - xs[s, 0]) * d[s]).mean()
                    sd = say[:, t] & ~q[:, t]
                    if sd.sum() >= 5:
                        e["l2_diffuse"][t] = np.linalg.norm(xy[sd, t] - Tf[sd, t], axis=-1).mean()
                if nm in G["attn"]:                                          # 마지막 관측 자리 3×3 attention 질량 (물체 clip 전부)
                    A = np.asarray(G["attn"][nm][io], np.float32)
                    A = A[:, TC:] if A.shape[1] == 2 * TP else A
                    cx = np.clip(np.floor(xs[:, 0] / 18.0), 0, 15); cy = np.clip(np.floor(xs[:, 1] / 18.0), 0, 15)
                    gg = np.arange(16)
                    msk = ((np.abs(gg[None, :, None] - cy[:, None, None]) <= 1) & (np.abs(gg[None, None, :] - cx[:, None, None]) <= 1)).reshape(len(io), 256)
                    mseen = np.einsum("ntk,nk->nt", A, msk.astype(np.float32))
                    e["mass_seen_med"] = np.median(mseen, 0); e["mass_gt_med"] = np.nanmedian(ms, 0)
                    e["seen_gt_cells_apart"] = np.median(np.abs(np.floor(Tf[..., 0] / 18.0) - cx[:, None]), 0)
                c[nm] = e
            out[f"{tim}|{mo}"] = c
    return out


# ============================================================================== vanish A/B (자 없는 교차검증)
def vanish_ab(G, R, thr, nb, rng):
    if not SCORES_JSON.exists():
        return None
    S = json.loads(SCORES_JSON.read_text())
    pairs = S["pairs"]
    cells = {"visible|flat": "moving_visible_flat", "visible|ramp": "moving_visible", "visible|static": "static_visible",
             "late|flat": "moving_occlusion_flat", "late|ramp": "moving_occlusion", "late|static": "static_occlusion"}
    # readout 쪽 pos_a (vanish obj) clip — 같은 block 의 p@h 영속 값 (t1..t4 '있다' 평균)
    vi = {(c, b): i for i, (c, b, v, vt) in enumerate(zip(G["cond"], G["block"], G["variant"], G["vtype"]))
          if v == "pos_a" and vt == "vanish"}
    out = {}
    for cname, cond in cells.items():
        tim = cname.split("|")[0]
        idx = [i for i, p in enumerate(pairs) if p["viol"] == "vanish" and p["cond"] == cond and (tim == "visible" or p["k"] >= KMIN)]
        blocks = sorted({pairs[i]["block"] for i in idx})
        bpos = {b: j for j, b in enumerate(blocks)}
        e = {"n_block": len(blocks)}
        hitsA, hitsB = {}, {}
        for tag, sk in SCORE_KEY.items():
            h = np.array(S["models"][sk]["v11_hits"], float)
            A = np.full(len(blocks), np.nan); B = np.full(len(blocks), np.nan)
            for i in idx:
                (A if pairs[i]["role"] == "imp_vanish" else B)[bpos[pairs[i]["block"]]] = h[i]
            hitsA[tag], hitsB[tag] = A, B
        Wb = np.stack([np.bincount(rng.integers(0, len(blocks), len(blocks)), minlength=len(blocks)) for _ in range(nb)]).astype(np.float32)
        for tag in SCORE_KEY:
            A, B = hitsA[tag], hitsB[tag]
            r = {"A_vanish": A.mean(), "A_ci": ci(Wb @ A / Wb.sum(1)), "B_appear": B.mean(), "B_ci": ci(Wb @ B / Wb.sum(1))}
            if tag != "release":
                dA = A - hitsA["release"]; r["dA"] = dA.mean(); r["dA_ci"] = ci(Wb @ dA / Wb.sum(1))
            # 같은 block 의 pos_a clip 에서 p@h 가 t1..t4 에 물체를 들고 있나 (평균 ≥ 0.5) → A 적중률
            if tim == "late" and f"ph:{tag}" in R:
                arr, rk = R[f"ph:{tag}"]
                keep = np.full(len(blocks), np.nan)
                for b, j in bpos.items():
                    ii = vi.get((cond, b))
                    if ii is not None and G["done"][ii]:
                        keep[j] = float((arr[ii, PERM_T, 2] > thr[rk]).mean() >= 0.5)
                ok = np.isfinite(keep)
                r["n_keep"] = int((keep[ok] == 1).sum()); r["n_drop"] = int((keep[ok] == 0).sum())
                r["A_given_keep"] = A[ok][keep[ok] == 1].mean() if r["n_keep"] else np.nan
                r["A_given_drop"] = A[ok][keep[ok] == 0].mean() if r["n_drop"] else np.nan
            e[tag] = r
        out[cname] = e
    return out


SCORE_CURVES = Path("/data2/local_datasets/world/world_analysis/cache/training_effects/v11score_vith_curves")
AB_CONDS = {"visible|flat": "moving_visible_flat", "visible|ramp": "moving_visible", "visible|static": "static_visible",
            "late|flat": "moving_occlusion_flat", "late|ramp": "moving_occlusion", "late|static": "static_occlusion"}


def curves_ab():
    """적대 검증 2026-09-25 — vanish A/B 를 curves 채점 (`te_v11_score.py` 산출, test block 만) 에서 복사 기준선 · 전 epoch 로.
    A = S(pos_a) < S(imp_ab) (물체→빈), B = S(pos_b) < S(imp_ba) (빈→물체), S = 미래 8 슬롯 L1 평균, 동점 0.5.
    k 부분집합: 'k>=2' (본문 기준) · 'k=1'. → {cell: {ksub: {tag|'copy': {A, B, n}}}}"""
    d = SCORE_CURVES
    if not (d / "meta.json").exists():
        return None
    m = json.loads((d / "meta.json").read_text())
    done = np.load(d / "done.npy").astype(bool)
    L = np.concatenate([np.load(d / "l1.npy"), np.load(d / "copy.npy")[:, None, :]], 1).astype(np.float64).mean(-1)   # (n, K+1)
    names = list(m["pred_tags"]) + ["copy"]
    row = {(c, b, v): i for i, (c, b, v, vt) in enumerate(zip(m["condition"], m["block_id"], m["variant"], m["violation_type"])) if vt == "vanish"}
    kk = {(c, b): int(k) for c, b, k in zip(m["condition"], m["block_id"], m["sym_k"])}
    out = {}
    for cname, cond in AB_CONDS.items():
        blocks = sorted({b for (c, b, v) in row if c == cond})
        e = {}
        for ks, ok in (("k>=2", lambda k: k >= KMIN), ("k=1", lambda k: k == 1)):
            if cname.startswith("visible") and ks == "k=1":
                continue
            bs = [b for b in blocks if (cname.startswith("visible") or ok(kk[(cond, b)]))
                  and all((cond, b, v) in row and done[row[(cond, b, v)]] for v in ("pos_a", "imp_ab", "pos_b", "imp_ba"))]
            if not bs:
                continue
            ia = np.array([row[(cond, b, "pos_a")] for b in bs]); iab = np.array([row[(cond, b, "imp_ab")] for b in bs])
            ib = np.array([row[(cond, b, "pos_b")] for b in bs]); iba = np.array([row[(cond, b, "imp_ba")] for b in bs])
            hA = (L[ia] < L[iab]) + 0.5 * (L[ia] == L[iab]); hB = (L[ib] < L[iba]) + 0.5 * (L[ib] == L[iba])
            e[ks] = {t: {"A": float(hA[:, j].mean()), "B": float(hB[:, j].mean()), "n": len(bs)} for j, t in enumerate(names)}
        out[cname] = e
    return out


# ============================================================================== 표 · 그림 · 문서
def f2(x, w=5, p=2):
    return f"{x:{w}.{p}f}" if x is not None and np.isfinite(x) else " " * (w - 1) + "·"


def fci(v, c, p=2, sign=False):
    if v is None or not np.isfinite(v):
        return "·"
    s = f"{v:+.{p}f}" if sign else f"{v:.{p}f}"
    if c is None or not np.all(np.isfinite(c)):
        return s
    lo, hi = (f"{c[0]:+.{p}f}", f"{c[1]:+.{p}f}") if sign else (f"{c[0]:.{p}f}", f"{c[1]:.{p}f}")
    return f"{s} [{lo}, {hi}]"


LABEL = {"release": "release", "v11_e10": "v11_e10", "ip1_e40": "ip1_e40", "ip2_e80": "ip2_e80",
         "pv1_e15": "pv1_e15", "ariel_ep43": "ariel_ep43"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", type=int, default=2000)
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--dump", default=None, help="계산 결과 (res) 를 pickle 로 저장 (문서만 다시 쓸 때)")
    ap.add_argument("--render-only", default=None, help="--dump 로 저장한 pickle 에서 문서 · 표 · 그림만 다시 쓴다")
    a = ap.parse_args()
    t0 = time.time()
    if a.render_only:
        import pickle
        res = pickle.load(open(a.render_only, "rb"))
        (OUT / "presence.json").write_text(json.dumps(r3(res), indent=1, ensure_ascii=False))
        write_tables(res, None)
        if not a.no_figs:
            figures(res)
        write_doc(res)
        print(f"→ {OUT} (render-only)")
        return
    rng = np.random.default_rng(0)
    G, R, thr, bias, tags, info = load_all()
    OUT.mkdir(parents=True, exist_ok=True)
    n_done = int(G["done"].sum()); n_test_done = int((G["done"] & G["test"]).sum())
    print(f"[load] {G['n']:,} clip, done {n_done:,} (test {n_test_done:,} / {int(G['test'].sum()):,}), predictor {len(tags)}  {time.time()-t0:.0f}s", flush=True)

    San = sanity(G, R, thr)
    print("[sanity] release p@p 전 block vs values.json 최대 차:",
          {k: round(v["maxdiff_pp"], 3) for k, v in San.items()}, flush=True)

    base = G["done"] & G["test"]
    rd_final = [f"{p}:{t}" for t in FINAL for p in ("ph", "pp")] + ["hh", "zz"] + [r for r in ("copy_zctx", "copy_hctx") if r in R]
    rd_all = [f"{p}:{t}" for t in tags for p in ("ph", "pp")] + ["hh", "zz"] + [r for r in ("copy_zctx", "copy_hctx") if r in R]
    Cells, Wts = cell_stats(G, R, thr, rd_final, base, a.nb, rng)
    print(f"[cells] {time.time()-t0:.0f}s", flush=True)
    Summ = summaries(G, R, thr, Wts, rd_all, a.nb, rng)
    print(f"[summaries] {time.time()-t0:.0f}s", flush=True)
    Pos = position(G, R, thr, bias, Wts, [f"{p}:{t}" for t in FINAL for p in ("ph", "pp")] + ["hh"])
    AB = vanish_ab(G, R, thr, a.nb, rng)
    CAB = curves_ab()
    print(f"[position · A/B] {time.time()-t0:.0f}s", flush=True)

    res = dict(source=str(CACHE), n_clip=G["n"], n_done=n_done, n_test=int(G["test"].sum()), n_test_done=n_test_done,
               thr=thr, bias_px={k: v.tolist() for k, v in bias.items()}, nb=a.nb, tags=tags,
               sanity=San, cells=Cells, summaries=Summ, position=Pos, vanish_ab=AB, vanish_ab_curves=CAB,
               definitions=__doc__)
    if a.dump:
        import pickle
        pickle.dump(res, open(a.dump, "wb"))
    (OUT / "presence.json").write_text(json.dumps(r3(res), indent=1, ensure_ascii=False))
    write_tables(res, G)
    if not a.no_figs:
        figures(res)
    write_doc(res)
    print(f"→ {OUT}  ({time.time()-t0:.0f}s)")


# ------------------------------------------------------------------------------ TABLES.md (전 칸)
def write_tables(res, G):
    L = []; P = L.append
    P("# v11 presence — 전 칸 표 (자동 생성, `te_analyze_presence.py`)\n")
    P(f"> test 절반 · done 행 {res['n_test_done']:,} / {res['n_test']:,} clip. 수치는 추출 배열에서 다시 계산했다. "
      "CI = (궤적, k) 두 단계 cluster bootstrap. 본문은 [`PRESENCE.md`](PRESENCE.md).\n")
    C = res["cells"]
    for key, c in C.items():
        P(f"\n## {key.replace('|', ' · ')} — obj {c['n_obj']} clip · 빈 장면 {c['n_empty']} · 단위 (궤적,k) {c['n_units']}\n")
        P(f"진실: 읽을 수 있는 칸 비율 {' '.join(f2(x) for x in c['frac_readable'])} · 가려진 칸 비율 {' '.join(f2(x) for x in c['frac_hidden'])}\n")
        P("| 읽기 | 양 | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 |")
        P("|---|---|" + "---:|" * 8)
        for nm in [r for r in c if isinstance(c[r], dict)]:
            e = c[nm]
            rows = [("present", e["present"]), ("  CI lo", e["present_ci"][:, 0]), ("  CI hi", e["present_ci"][:, 1]),
                    ("floor (빈)", e["floor"]), ("present−floor", e["diff"]), ("AUROC", e["auroc"])]
            if "mass3_med" in e:
                rows += [("mass3 중앙", e["mass3_med"]), ("있다∧진실칸", e["at_gt"])]
            if "d_diff" in e:
                rows += [("Δ(present−floor) vs rel", e["d_diff"]), ("  CI lo", e["d_diff_ci"][:, 0]), ("  CI hi", e["d_diff_ci"][:, 1])]
            for i, (q, v) in enumerate(rows):
                P(f"| {nm if i == 0 else ''} | {q} | " + " | ".join(f2(x, 1, 2).strip() for x in v) + " |")
    S = res["summaries"]
    P("\n## 영속 · 지속 요약 — 전 predictor (curves 22) × 읽기 (p@h · p@p)\n")
    P("값 [95% CI] · Δ vs release [CI]. 영속 = late k≥2 t1..t4 present − 바닥, 지속 = visible 읽을 수 있는 칸 t3..t7 present.\n")
    for sn, q in (("permanence", "diff"), ("persistence", "present")):
        for prefix in ("ph", "pp"):
            P(f"\n### {sn} · {prefix[0]}@{prefix[1]}\n")
            P("| tag | flat | Δ flat | AUROC flat | ramp | Δ ramp | AUROC ramp |")
            P("|---|---|---|---|---|---|---|")
            for tg in res["tags"]:
                row = [tg]
                for mo in ("flat", "ramp"):
                    e = S[f"{sn}|{mo}"]["readings"][f"{prefix}:{tg}"]
                    row += [fci(e[q], e[q + "_ci"]), fci(e.get(f"d_{q}"), e.get(f"d_{q}_ci"), sign=True) if f"d_{q}" in e else "기준",
                            fci(e["auroc"], e["auroc_ci"])]
                P("| " + " | ".join(row) + " |")
    (OUT / "TABLES.md").write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------------------ 그림
def figures(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#888888", "axes.labelcolor": "#333333", "xtick.color": "#555555", "ytick.color": "#555555"})
    BLUE, ORANGE, GREEN, BLACK, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#222222", "#9a9a9a"
    C, S = res["cells"], res["summaries"]
    t = np.arange(TP)
    letters = "abcdefghijklmnopqrstuvwxyz"

    def lab(ax, i, y=-0.30):
        ax.text(0.5, y, f"({letters[i] if i < 26 else letters[i // 26 - 1] + letters[i % 26]})",
                transform=ax.transAxes, ha="center", va="top", fontsize=9)

    # (1) 튜블릿 곡선 — 행 = predictor (release 제외), 열 = 칸
    cols = [("visible", "flat"), ("visible", "ramp"), ("late", "static"), ("late", "flat"), ("late", "ramp"), ("mid", "ramp")]
    rows = FINAL[1:]
    for prefix, rname in (("ph", "p@h"), ("pp", "p@p")):
        fig, axs = plt.subplots(len(rows), len(cols), figsize=(13, 2.25 * len(rows) + 0.6), sharey=True)
        i = 0
        for r, tag in enumerate(rows):
            for cc, (tim, mo) in enumerate(cols):
                ax = axs[r, cc]; c = C[f"{tim}|{mo}"]
                rel, e = c[f"{prefix}:release"], c[f"{prefix}:{tag}"]
                ax.plot(t, rel["present"], color=BLACK, lw=1.4, label="release present")
                ax.plot(t, rel["floor"], color=GRAY, lw=1.2, ls="--", label="release empty floor")
                ax.fill_between(t, e["present_ci"][:, 0], e["present_ci"][:, 1], color=BLUE, alpha=0.18, lw=0)
                ax.plot(t, e["present"], color=BLUE, lw=1.8, marker="o", ms=3, label=f"predictor present")
                ax.plot(t, e["floor"], color=ORANGE, lw=1.4, ls="--", label="predictor empty floor")
                ax.set_ylim(-0.03, 1.03); ax.set_xticks(t)
                if r == 0:
                    ax.set_title(f"{tim} · {mo}", fontsize=9)
                if cc == 0:
                    ax.set_ylabel(f"{tag}\nfraction 'present'")
                ax.grid(axis="y", color="#e6e6e6", lw=0.6)
                lab(ax, i); i += 1
        h, l = axs[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
        fig.supxlabel("future tubelet", fontsize=9)
        fig.tight_layout(rect=(0, 0.01, 1, 0.965)); fig.subplots_adjust(hspace=0.75)
        fig.savefig(OUT / f"fig_tubelet_{prefix}.png", dpi=150); plt.close(fig)

    # (2) Δ(present − floor) vs release 히트맵
    cmap = LinearSegmentedColormap.from_list("bgo", [ORANGE, "#d9d9d9", BLUE])
    cellkeys = [f"{tm}|{mo}" for tm in TIMINGS for mo in MOTIONS]
    fig, axs = plt.subplots(2, 1, figsize=(13, 5.2))
    for ii, (prefix, rname) in enumerate((("ph", "p@h"), ("pp", "p@p"))):
        M = np.array([np.concatenate([C[k][f"{prefix}:{tag}"]["d_diff"] for k in cellkeys]) for tag in FINAL[1:]])
        ax = axs[ii]
        im = ax.imshow(M, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        ax.set_yticks(range(len(FINAL) - 1)); ax.set_yticklabels(FINAL[1:])
        ax.set_xticks([8 * j + 3.5 for j in range(len(cellkeys))]); ax.set_xticklabels([k.replace("|", " · ") for k in cellkeys])
        from matplotlib.patches import Rectangle
        for j, k in enumerate(cellkeys):                       # 진실이 읽을 수 없는 칸 (가려짐 · 가장자리 · 화면 밖) 에 빗금
            fr = np.asarray(C[k]["frac_readable"], float)
            for tt in range(TP):
                if not fr[tt] >= 0.5:
                    ax.add_patch(Rectangle((8 * j + tt - 0.5, -0.5), 1, len(FINAL) - 1, fill=False, hatch="///",
                                           edgecolor="white", lw=0))
        for j in range(1, len(cellkeys)):
            ax.axvline(8 * j - 0.5, color="white", lw=2)
        ax.set_title(f"{rname}: delta (present - empty floor) vs release, tubelets t0..t7 per cell", fontsize=9)
        ax.spines[:].set_visible(False); ax.tick_params(length=0)
        lab(ax, ii)
        cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01); cb.outline.set_visible(False)
    fig.tight_layout(); fig.subplots_adjust(hspace=0.6)
    fig.savefig(OUT / "fig_delta_heatmap.png", dpi=150); plt.close(fig)

    # (3) 영속 · 지속 요약 (final)
    fig, axs = plt.subplots(1, 5, figsize=(15, 3.5))
    panels = [("permanence", "diff", "permanence: present - floor\n(late, t1..t4)"),
              ("permanence", "auroc", "permanence: AUROC object vs empty\n(late, t1..t4)"),
              ("permanence", "mass", "permanence: GT 3x3 attention\n(late, readable t1..t4)"),
              ("persistence", "present", "persistence: present\n(visible, readable t3..)"),
              ("persistence", "mass", "persistence: GT 3x3 attention\n(visible, readable t3..)")]
    x = np.arange(len(FINAL))
    for i, (sn, q, title) in enumerate(panels):
        ax = axs[i]
        for j, (mo, col, mk, off) in enumerate((("flat", BLUE, "o", -0.12), ("ramp", ORANGE, "s", 0.12))):
            rr = S[f"{sn}|{mo}"]["readings"]
            v = np.array([rr[f"ph:{tg}"][q] for tg in FINAL]); c_ = np.array([rr[f"ph:{tg}"][f"{q}_ci"] for tg in FINAL])
            ax.errorbar(x + off, v, yerr=[v - c_[:, 0], c_[:, 1] - v], fmt=mk, color=col, ms=5, lw=1.2, capsize=2, label=mo)
            if q == "mass":                                    # 빈 장면 귀무 (정정 2026-09-25): 속 빈 표식
                vn = np.array([rr[f"ph:{tg}"]["mass_null"] for tg in FINAL])
                ax.plot(x + off, vn, mk, mfc="none", color=col, ms=5, lw=0, label=f"{mo} empty-scene null")
            hh = rr["hh"][q]
            ax.axhline(hh, color=col, lw=0.9, ls=":", alpha=0.9)
        ax.set_xticks(x); ax.set_xticklabels(FINAL, rotation=40, ha="right")
        ax.set_title(title, fontsize=8.5); ax.grid(axis="y", color="#e6e6e6", lw=0.6)
        if q == "auroc":
            ax.axhline(0.5, color=GRAY, lw=0.8, ls="--")
        if q == "mass":
            ax.axhline(MASS_U, color=GRAY, lw=0.8, ls="--")
        lab(ax, i, y=-0.42)
    h, l = axs[2].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h += [Line2D([], [], color=GRAY, ls=":", lw=1)]; l += ["h@h (target, same ruler)"]
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.subplots_adjust(bottom=0.3)
    fig.savefig(OUT / "fig_summary_final.png", dpi=150); plt.close(fig)

    # (4) 학습 곡선
    fams = list(FAMILIES)
    fig, axs = plt.subplots(4, len(fams), figsize=(14, 10.5), sharey="row")
    i = 0
    for r, (sn, q, yl) in enumerate((("permanence", "diff", "permanence\npresent - floor (p@h)"),
                                     ("permanence", "auroc", "permanence\nAUROC (p@h)"),
                                     ("permanence", "mass", "permanence placement\nGT 3x3 attention (p@h)"),
                                     ("persistence", "mass", "persistence placement\nGT 3x3 attention (p@h)"))):
        for cc, fam in enumerate(fams):
            ax = axs[r, cc]
            for mo, col, mk in (("flat", BLUE, "o"), ("ramp", ORANGE, "s")):
                rr = S[f"{sn}|{mo}"]["readings"]
                ep = np.array([e for e, _ in FAMILIES[fam]]); v = np.array([rr[f"ph:{tg}"][q] for _, tg in FAMILIES[fam]])
                c_ = np.array([rr[f"ph:{tg}"][f"{q}_ci"] for _, tg in FAMILIES[fam]])
                ax.fill_between(ep, c_[:, 0], c_[:, 1], color=col, alpha=0.15, lw=0)
                ax.plot(ep, v, color=col, marker=mk, ms=4, lw=1.6, label=mo)
                if q == "mass":                                # 빈 장면 귀무 (정정 2026-09-25)
                    vn = np.array([rr[f"ph:{tg}"]["mass_null"] for _, tg in FAMILIES[fam]])
                    ax.plot(ep, vn, color=col, marker=mk, mfc="none", ms=4, lw=1.0, ls="--", label=f"{mo} empty-scene null")
                ax.axhline(rr["ph:release"][q], color=GRAY, lw=1.0, ls="--" if mo == "flat" else ":")
            if r == 0:
                ax.set_title(FAM_NAME[fam], fontsize=9)
            if cc == 0:
                ax.set_ylabel(yl)
            ax.set_xlabel("epoch"); ax.grid(axis="y", color="#e6e6e6", lw=0.6)
            lab(ax, i); i += 1
    h, l = axs[2, 0].get_legend_handles_labels()
    h += [Line2D([], [], color=GRAY, ls="--"), Line2D([], [], color=GRAY, ls=":")]; l += ["release flat", "release ramp"]
    for cc in range(len(fams)):
        for rr_ in (2, 3):
            axs[rr_, cc].axhline(MASS_U, color="#cccccc", lw=0.8)
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.subplots_adjust(hspace=0.7)
    fig.savefig(OUT / "fig_learning_curves.png", dpi=150); plt.close(fig)

    # (5) 위치 — x 전진 (행 = predictor, 열 = 칸), '있다' ∧ mass3 > 0.035 인 칸만
    Pz = res["position"]
    cols = [("visible", "flat"), ("visible", "ramp"), ("late", "flat"), ("late", "ramp")]
    fig, axs = plt.subplots(len(rows), len(cols), figsize=(11, 2.2 * len(rows) + 0.6), sharey=True)
    i = 0
    for r, tag in enumerate(rows):
        for cc, (tim, mo) in enumerate(cols):
            ax = axs[r, cc]; c = Pz[f"{tim}|{mo}"]
            rel, e = c["ph:release"], c[f"ph:{tag}"]
            ax.plot(t, c["prog_gt_all"], color=GRAY, lw=1.2, ls=":", label="ground truth")
            ax.plot(t, rel["prog"], color=BLACK, lw=1.3, marker="o", ms=3, label="release p@h")
            ax.plot(t, e["prog"], color=BLUE, lw=1.7, marker="o", ms=3, label="predictor p@h")
            ax.set_xticks(t); ax.grid(axis="y", color="#e6e6e6", lw=0.6)
            if r == 0:
                ax.set_title(f"{tim} · {mo}", fontsize=9)
            if cc == 0:
                ax.set_ylabel(f"{tag}\nx progress (px)")
            lab(ax, i); i += 1
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.supxlabel("future tubelet", fontsize=9)
    fig.tight_layout(rect=(0, 0.01, 1, 0.965)); fig.subplots_adjust(hspace=0.75)
    fig.savefig(OUT / "fig_position_progress.png", dpi=150); plt.close(fig)


# ------------------------------------------------------------------------------ PRESENCE.md
def write_doc(res):
    (OUT / "PRESENCE.md").write_text(render(res))


REFS = [("hh", "h@h (target 자체, 같은 h 자)"), ("zz", "z@z (z_full, z 자)"),
        ("copy_hctx", "복사 h_ctx@h (문맥 끝 튜블릿)"), ("copy_zctx", "복사 z_ctx@z (문맥 끝 튜블릿)")]


def _summary_table(S, sname, prefix, refs=True):
    L = ["| 운동 | 읽기 | present | 빈 바닥 | present − 바닥 [95% CI] | Δ vs release [CI] | (궤적,k) 단위 Δ>0 / Δ<0 | AUROC [CI] | ΔAUROC [CI] | 배치 (진실칸 attention) [CI] | 빈 장면 귀무 [CI] | 순 배치 (배치 − 귀무) [CI] | Δ순배치 [CI] |",
         "|---|---|---:|---:|---|---|:---:|---|---|---|---|---|---|"]
    for mo in ("flat", "ramp"):
        blk = S[f"{sname}|{mo}"]; rr = blk["readings"]
        items = [(f"{prefix}:{t}", f"{prefix[0]}@{prefix[1]} {t}") for t in FINAL]
        if refs:
            items += [(k, v) for k, v in REFS if k in rr]
        for nm, lab_ in items:
            e = rr[nm]
            dd = fci(e.get("d_diff"), e.get("d_diff_ci"), sign=True) if "d_diff" in e else "기준"
            un = f"{e['units_pos']} / {e['units_neg']} (of {e['units_n']})" if "units_n" in e else "—"
            da = fci(e.get("d_auroc"), e.get("d_auroc_ci"), sign=True) if "d_auroc" in e else "—"
            dm = fci(e.get("d_mass_net"), e.get("d_mass_net_ci"), sign=True) if "d_mass_net" in e else "—"
            L.append(f"| {mo} | {lab_} | {f2(e['present']).strip()} | {f2(e['floor']).strip()} | {fci(e['diff'], e['diff_ci'], sign=True)} | "
                     f"{dd} | {un} | {fci(e['auroc'], e['auroc_ci'])} | {da} | {fci(e.get('mass'), e.get('mass_ci'))} | "
                     f"{fci(e.get('mass_null'), e.get('mass_null_ci'))} | {fci(e.get('mass_net'), e.get('mass_net_ci'), sign=True)} | {dm} |")
    return L


def render(res):
    C, S, Pz, AB, San = res["cells"], res["summaries"], res["position"], res["vanish_ab"], res["sanity"]
    L = []; P = L.append
    full = res["n_test_done"] == res["n_test"]
    P("# v11 있음 (presence) — 영속 · 지속, predictor 별 · 학습 곡선")
    P("")
    L += correction_block(res)
    P("> 자동 생성 (`z_research/scripts/analysis/te_analyze_presence.py`) — 손으로 고치지 않는다. 2026-09-25.")
    P(f"> 데이터: `{Path(res['source']).name}` (`te_v11_readout.py` 추출), **`v11_split_test` held-out block 절반만** — "
      f"{res['n_test_done']:,} / {res['n_test']:,} clip{' (전부)' if full else ' ⚠️ 추출 미완료분 제외'}; 가능 변이 (물체 clip + 빈 장면 vanish pos_b).")
    P("> **모든 수치는 추출 배열 (`rd_*.npy`) 에서 이 스크립트가 다시 계산한 것**이다. 문턱은 자마다 하나, 릴리즈에서 보정한 값을 모든 predictor 에: "
      f"h 자 {res['thr']['h']:.3f} · p 자 {res['thr']['p']:.4f} · z 자 {res['thr']['z']:.3f}.")
    P("> 상위: [`../README.md`](../README.md) (정의 · 읽는 규칙) · [`../MODELS.md`](../MODELS.md) · 전 칸 표 [`TABLES.md`](TABLES.md) · 값 `presence.json`.")
    P("")
    P("## 한 줄")
    P("")
    L += headline(res)
    P("")
    P("## 표")
    P("")
    P("### 표 0. sanity — release p@p, **전 block**, `RollOutV3/figures/v11_presence/_superseded/presence_decoder/values.json` 과 대조")
    P("")
    P("| 칸 | n (여기 / 참조) | 최대 |Δ| p@p present | p@p 빈 바닥 | h@h | z@z |")
    P("|---|---:|---:|---:|---:|---:|")
    for k, e in San.items():
        P(f"| {k.replace('|', ' · ')} | {e['n']} / {e['n_ref']} | {e['maxdiff_pp']:.3f} | {e.get('maxdiff_pp_null', float('nan')):.3f} | "
          f"{e.get('maxdiff_hh', float('nan')):.4f} | {e.get('maxdiff_zz', float('nan')):.4f} |")
    mx = max(e["maxdiff_pp"] for e in San.values())
    P("")
    P(f"p@p 최대 차 {mx:.3f} (bf16 predictor 잡음 기준 ~0.01–0.02), encoder 쪽 (h@h · z@z) 은 사실상 비트 동일. "
      "빈 바닥 · h · z 는 `_audit/v11_presence/results.json` 과 대조. 이 아래는 전부 **test 절반**이다.")
    P("")
    P("### 표 A. 요약 — Δ vs release 가 **자 (p@h · p@p) · 문턱 없는 AUROC · 배치** 에서 같은 방향인가")
    P("")
    P("↑ / ↓ = Δ 의 95 % CI 가 0 을 넘지 않음, ≈ = CI 가 0 을 포함. 단위 = (궤적,k) 단위 중 p@h Δ 부호가 같은 수. "
      "배치 = 진실이 보이는 칸의 진실 3×3 attention 질량 평균 (p@h 자, 문턱 없음, 균등 0.035). 순 배치 = 배치 − 빈 장면 (물체가 없던 장면) 의 같은 자리 질량. "
      "판정 '견고' = p@h · p@p 가 같은 방향 · 단위 전부가 같은 부호 · **문턱 없는 AUROC Δ 와 순 배치 Δ 도 같은 방향** (정정 2026-09-25; 아니면 '문턱만'). "
      "'궤적만 cluster' 열 = (궤적,k) 대신 궤적 해시 (2 개) 로 두 단계 bootstrap 한 p@h Δ CI. "
      "⚠️ 지속 (visible) 은 단위가 **2 개** (두 궤적) 라 '견고' 는 '두 궤적 모두에서' 라는 뜻뿐이다.")
    P("")
    L += robust_table(res)
    P("")
    P("### 표 1. 영속 (permanence) — late 가림 k≥2 (문맥 끝에 물체가 가려짐), t1..t4, **p@h (주 판독)**")
    P("")
    b = S["permanence|flat"]
    P(f"물체 clip {b['n_obj']} · 빈 장면 {b['n_empty']} · 단위 (궤적,k) {b['n_units']} (flat; ramp 같음). "
      "present − 바닥 = t1..t4 평균. AUROC = t 마다 계산해 평균. Δ = 같은 clip 위 짝지은 차이 (같은 bootstrap 가중치). "
      "참조 줄 (h@h · z@z · 복사) 은 predictor 와 무관하다.")
    P("")
    L += _summary_table(S, "permanence", "ph")
    P("")
    P("### 표 2. 지속 (persistence) — visible (가림 없음), **읽을 수 있는 칸만** t3..t7, p@h")
    P("")
    fr = {mo: [t for t in PERS_T if C[f"visible|{mo}"]["frac_readable"][t] >= 0.5] for mo in ("flat", "ramp")}
    P(f"읽을 수 있는 칸 = 진실이 보이고 화면 안 · 가장자리 36 px 안쪽. v11 에서 실제로 남는 튜블릿: flat {fr['flat']} · ramp {fr['ramp']} "
      "(그 뒤는 물체가 가장자리에 붙거나 화면 밖). 즉 '지속' 은 **미래 절반 조금 넘게까지만** 잴 수 있다.")
    P("")
    L += _summary_table(S, "persistence", "ph")
    P("")
    P("### 표 3. p@p (보조 — 릴리즈 p 로 학습한 자를 다른 predictor 의 p 에 이식한 판독)")
    P("")
    P("**영속**")
    P("")
    L += _summary_table(S, "permanence", "pp", refs=False)
    P("")
    P("**지속**")
    P("")
    L += _summary_table(S, "persistence", "pp", refs=False)
    P("")
    P("### 표 4. 학습 곡선 — p@h, 값 [95% CI] (epoch 0 = release)")
    P("")
    P("| run | epoch | 영속 flat (present−바닥) | 영속 ramp | 영속 AUROC flat | 영속 AUROC ramp | 영속 배치 flat | 영속 배치 ramp | 영속 빈 장면 귀무 flat / ramp | 영속 순 배치 flat / ramp | 지속 flat (present) | 지속 ramp | 지속 배치 flat | 지속 배치 ramp |")
    P("|---|---:|---|---|---|---|---|---|---|---|---|---|---|---|")
    for fam, lst in FAMILIES.items():
        for ep, tg in lst:
            nm = f"ph:{tg}"
            g = lambda sn, mo, q: fci(S[f"{sn}|{mo}"]["readings"][nm][q], S[f"{sn}|{mo}"]["readings"][nm][q + "_ci"])
            P(f"| {fam} | {ep} | {g('permanence', 'flat', 'diff')} | {g('permanence', 'ramp', 'diff')} | "
              f"{g('permanence', 'flat', 'auroc')} | {g('permanence', 'ramp', 'auroc')} | {g('permanence', 'flat', 'mass')} | {g('permanence', 'ramp', 'mass')} | "
              f"{S['permanence|flat']['readings'][nm]['mass_null']:.2f} / {S['permanence|ramp']['readings'][nm]['mass_null']:.2f} | "
              f"{S['permanence|flat']['readings'][nm]['mass_net']:+.2f} / {S['permanence|ramp']['readings'][nm]['mass_net']:+.2f} | "
              f"{g('persistence', 'flat', 'present')} | {g('persistence', 'ramp', 'present')} | {g('persistence', 'flat', 'mass')} | {g('persistence', 'ramp', 'mass')} |")
    P("")
    P("ariel 은 scratch 라 epoch 0 이 release 가 아니다 (release 는 그림 4 의 회색 기준선). p@p 곡선은 `presence.json` → `summaries`.")
    P("")
    P("### 표 5. 튜블릿별 — p@h present / 빈 바닥 (t0 … t7)")
    P("")
    for key in ("late|flat", "late|ramp", "visible|flat", "visible|ramp", "late|static", "mid|flat", "mid|ramp"):
        c = C[key]
        P(f"**{key.replace('|', ' · ')}** (obj {c['n_obj']} · 빈 {c['n_empty']} · 단위 {c['n_units']}) — 진실 읽을 수 있는 칸 비율 "
          + " ".join(f"{x:.2f}" for x in c["frac_readable"]))
        P("")
        P("| 읽기 | " + " | ".join(f"t{t}" for t in range(TP)) + " | AUROC t0…t7 |")
        P("|---|" + "---:|" * TP + "---|")
        for nm in [f"ph:{t}" for t in FINAL] + ["hh"]:
            e = c[nm]
            P(f"| {nm.replace('ph:', 'p@h ') if nm != 'hh' else 'h@h (참조)'} | " +
              " | ".join(f"{a:.2f}/{b_:.2f}" for a, b_ in zip(e["present"], e["floor"])) + " | " +
              " ".join(f"{x:.2f}" if x is not None and np.isfinite(x) else "·" for x in e["auroc"]) + " |")
        P("")
    P("### 표 6. 위치 — '있다' ∧ mass3 > 0.035 (진실 칸 3×3 에 attention) 인 칸만, p@h")
    P("")
    P("각 칸: **있다∧진실칸 비율** / '있다' 비율 · L2 (px, 치우침 보정) · 부호 x 오차 (+ = 운동 방향으로 앞섬). 조건을 채운 clip 이 5 개 미만이면 `·`. "
      "1 칸 = 18 px. '있다' 인데 진실칸 attention 이 균등 이하인 칸 = 자 기본값 (어딘가에 '있다', 위치 주장 불가).")
    P("")
    for key in ("visible|flat", "visible|ramp", "late|flat", "late|ramp"):
        P(f"**{key.replace('|', ' · ')}**")
        P("")
        P("| 읽기 | 양 | " + " | ".join(f"t{t}" for t in range(TP)) + " |")
        P("|---|---|" + "---:|" * TP)
        for nm in [f"ph:{t}" for t in FINAL] + ["hh"]:
            e = Pz[key][nm]; lab_ = nm.replace("ph:", "p@h ") if nm != "hh" else "h@h (참조)"
            P(f"| {lab_} | 진실칸/있다 | " + " | ".join(f"{a:.2f}/{b_:.2f}" for a, b_ in zip(e["at_gt"], e["present"])) + " |")
            P(f"| | L2 px | " + " | ".join(f2(x, 1, 0).strip() for x in e["l2"]) + " |")
            P(f"| | x 오차 px | " + " | ".join((f"{x:+.0f}" if x is not None and np.isfinite(x) else "·") for x in e["err_x"]) + " |")
            if "mass_seen_med" in e:
                P(f"| | attention 3×3 중앙값 진실칸 / 마지막 관측칸 | " + " | ".join(f"{a:.2f}/{b_:.2f}" for a, b_ in zip(e["mass_gt_med"], e["mass_seen_med"])) + " |")
        P(f"| 진실 | x 전진 px (칸의 모든 물체 clip) | " + " | ".join(f2(x, 1, 0).strip() for x in Pz[key]["prog_gt_all"]) + " |")
        P(f"| 진실 | 진실칸 − 마지막 관측칸 거리 (칸, 중앙) | " + " | ".join(f"{x:.0f}" for x in Pz[key]["ph:release"]["seen_gt_cells_apart"]) + " |")
        P("")
    if AB:
        P("### 표 7. vanish 방향별 (자 없는 교차검증) — 표준 채점 `scores/scores.json`, 같은 test block, k≥2")
        P("")
        P("A = 문맥에 물체 → 불가능 미래가 빈 장면 (`imp_vanish`, 물체→빈) 을 잡았나 = S(pos_a) < S(imp_ab). "
          "B = 문맥이 빈 장면 → 물체가 나타남 (`imp_appear`, 빈→물체). CI = block bootstrap. "
          "'유지/놓침' = 같은 block 의 pos_a clip 에서 p@h 가 t1..t4 중 절반 이상 '있다' 인가 (readout 은 bf16, 채점은 fp32+autocast fp16 — 다른 실행).")
        P("")
        P("| 칸 | 모델 | A 물체→빈 [CI] | ΔA vs release [CI] | B 빈→물체 [CI] | p@h 유지 block 의 A (n) | p@h 놓침 block 의 A (n) |")
        P("|---|---|---|---|---|---|---|")
        for ck in ("late|flat", "late|ramp", "late|static", "visible|flat", "visible|ramp"):
            e = AB[ck]
            for tg in FINAL:
                r = e[tg]
                keep = (f"{r['A_given_keep']*100:.0f} % ({r['n_keep']})" if r.get("n_keep") else "·") if "n_keep" in r else "—"
                drop = (f"{r['A_given_drop']*100:.0f} % ({r['n_drop']})" if r.get("n_drop") else "·") if "n_drop" in r else "—"
                dA = fci(r["dA"] * 100, np.array(r["dA_ci"]) * 100, p=1, sign=True) if "dA" in r else "기준"
                P(f"| {ck.replace('|', ' · ')} (block {e['n_block']}) | {tg} | {fci(r['A_vanish']*100, np.array(r['A_ci'])*100, p=1)} | {dA} | "
                  f"{fci(r['B_appear']*100, np.array(r['B_ci'])*100, p=1)} | {keep} | {drop} |")
            CAB = res.get("vanish_ab_curves")
            if CAB and ck in CAB and "k>=2" in CAB[ck]:
                cc = CAB[ck]["k>=2"]["copy"]
                P(f"| | **복사 기준선** (curves 채점, block {cc['n']}) | {cc['A']*100:.1f} | — | {cc['B']*100:.1f} | — | — |")
        P("")
        CAB = res.get("vanish_ab_curves")
        if CAB:
            P("### 표 7b. vanish A/B — curves 채점 (`v11score_vith_curves`, test block), k≥2 vs k=1, 복사 기준선 포함")
            P("")
            P("A / B (%). curves 채점은 하네스 최종 채점 (표 7) 과 몇 block 에서 다르다 (예: ip2_e80 late flat A 46.4 vs 47.6). k=1 은 칸마다 28 block.")
            P("")
            P("| 칸 | k | n block | " + " | ".join(FINAL + ["copy"]) + " |")
            P("|---|---|---:|" + "---|" * (len(FINAL) + 1))
            for ck in ("late|flat", "late|ramp", "late|static"):
                for ks in ("k>=2", "k=1"):
                    d = CAB[ck].get(ks)
                    if d:
                        P(f"| {ck.replace('|', ' · ')} | {ks} | {d['copy']['n']} | " + " | ".join(f"{d[t]['A']*100:.0f} / {d[t]['B']*100:.0f}" for t in FINAL + ["copy"]) + " |")
            P("")
            P("Predictor_v1 epoch 별 (late flat k≥2): " + " · ".join(f"e{ep} A {CAB['late|flat']['k>=2'][tg]['A']*100:.1f} / B {CAB['late|flat']['k>=2'][tg]['B']*100:.1f}" for ep, tg in FAMILIES["pv1"]) + " %")
            P("")
    P("### 표 8. 궤적 2 개로 무엇이 남나 — (궤적, k) 단위별 Δ(present − 바닥) vs release 의 부호, p@h")
    P("")
    P("| 요약 | 운동 | 단위 수 | " + " | ".join(FINAL[1:]) + " |")
    P("|---|---|---:|" + "---|" * (len(FINAL) - 1))
    for sn in ("permanence", "persistence"):
        for mo in ("flat", "ramp"):
            rr = S[f"{sn}|{mo}"]["readings"]
            P(f"| {sn} | {mo} | {rr['ph:v11_e10']['units_n']} | " +
              " | ".join(f"{rr[f'ph:{t}']['units_pos']}+ / {rr[f'ph:{t}']['units_neg']}− (min {min(v for v in rr[f'ph:{t}']['unit_d_diff'].values()):+.2f}, max {max(v for v in rr[f'ph:{t}']['unit_d_diff'].values()):+.2f})"
                         for t in FINAL[1:]) + " |")
    P("")
    L += reading_text(res)
    P("")
    P("## 그림")
    P("")
    P("- `fig_summary_final.png` — (a) 영속 present − 바닥 (b) 영속 AUROC (c) 영속 배치 (d) 지속 present (e) 지속 배치. p@h, 점 = flat (파랑) · ramp (주황), 속 빈 점 = 빈 장면 귀무 (배치 패널), 막대 = (궤적,k) CI, 색 점선 = h@h (같은 자로 읽은 target), 회색 파선 = 0.5 (AUROC) · 균등 0.035 (배치)")
    P("- `fig_learning_curves.png` — 행 = 영속 present − 바닥 · 영속 AUROC · 영속 배치 · 지속 배치, 열 = run (x = epoch). 배치 행의 속 빈 점 · 파선 = 빈 장면 귀무. 회색 파선/점선 = release flat/ramp")
    P("- `fig_tubelet_ph.png` · `fig_tubelet_pp.png` — 행 = predictor, 열 = 칸. 검정 = release present, 회색 점선 = release 빈 바닥, 파랑 = predictor present (띠 = CI), 주황 점선 = predictor 빈 바닥")
    P("- `fig_delta_heatmap.png` — Δ(present − 바닥) vs release, 칸 × 튜블릿. 파랑 = 늘어남 · 주황 = 줄어듦. 빗금 = 진실이 읽을 수 없는 튜블릿 (가려짐 · 가장자리 · 화면 밖) — 해석하지 않는다")
    P("- `fig_position_progress.png` — '있다' ∧ mass3 > 0.035 인 clip 의 x 전진 (문맥에서 마지막으로 보인 자리 기준, 운동 방향 +). 회색 점선 = 진실")
    P("")
    L += caveats(res)
    P("")
    P("## 재현")
    P("")
    P("```bash")
    P("cd /data/hyuntak/project/2026/2027_cvpr/vjepa2")
    P("P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python")
    P("# 추출 (GPU, 이미 끝남 — 입력): te_v11_readout.py --preset curves → /data2/local_datasets/world/world_analysis/cache/training_effects/v11readout_curves/")
    P("# CUDA_VISIBLE_DEVICES=4,5,6,7 $P z_research/scripts/analysis/te_v11_readout.py --preset curves --gpus 4 --resume")
    P("$P z_research/scripts/analysis/te_analyze_presence.py            # CPU, 이 문서 · TABLES.md · presence.json · fig_*.png")
    P("```")
    P("")
    P("입력 (읽기 전용): 추출 폴더 · `data_csv/intphysgen_v11_split/index_test.csv` · `z_research/RollOutV3/exp_results/presence/attn_bias_px.json` · "
      "`z_research/RollOutV3/figures/v11_presence/_superseded/presence_decoder/values.json` (sanity) · `RollOutV3/audit/v11_presence/{results.json,ctx_only_readings.npz}` (sanity · 복사 기준선) · "
      "`z_research/TrainingEffects/scores/scores.json` (vanish A/B).")
    return "\n".join(L) + "\n"


def _arrow(e, q):
    c = e.get(f"d_{q}_ci")
    if c is None or not np.all(np.isfinite(c)):
        return "·"
    return "↑" if c[0] > 0 else ("↓" if c[1] < 0 else "≈")


def verdict(res, tg, sn, mo):
    S = res["summaries"]
    eh, ep = S[f"{sn}|{mo}"]["readings"][f"ph:{tg}"], S[f"{sn}|{mo}"]["readings"][f"pp:{tg}"]
    ah, ap = _arrow(eh, "diff"), _arrow(ep, "diff")
    n = eh["units_n"]; same = eh["units_pos"] == n or eh["units_neg"] == n
    if ah == ap and ah in "↑↓" and same:
        # 적대 검증 2026-09-25: 고정 문턱은 자 보정 이동을 섞는다 — 문턱 없는 AUROC 와 순 배치 (배치 − 빈 장면 귀무) 가 같은 방향일 때만 '견고'
        au, pl = _arrow(eh, "auroc"), _arrow(eh, "mass_net")
        if au == ah and pl == ah:
            return f"견고 {ah}"
        return f"문턱만 {ah} (AUROC {au} · 순배치 {pl})"
    if ah == ap and ah in "↑↓":
        return f"{ah} (단위 불일치)"
    if "↑" in (ah, ap) and "↓" in (ah, ap):
        return "자에 따라 반대"
    return "한쪽 자만"


def robust_table(res):
    S = res["summaries"]
    L = ["| predictor | 요약 | 운동 | p@h Δ(present−바닥) | p@h Δ CI (궤적만 cluster) | p@p Δ | AUROC Δ (p@h / p@p) | 배치 Δ (p@h) | 순 배치 Δ (배치 − 빈 장면 귀무) | 단위 (p@h) | 판정 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for tg in FINAL[1:]:
        for sn in ("permanence", "persistence"):
            for mo in ("flat", "ramp"):
                eh, ep = S[f"{sn}|{mo}"]["readings"][f"ph:{tg}"], S[f"{sn}|{mo}"]["readings"][f"pp:{tg}"]
                ah, ap = _arrow(eh, "diff"), _arrow(ep, "diff"); n = eh["units_n"]
                v = verdict(res, tg, sn, mo); v = f"**{v}**" if v.startswith("견고") else v
                ct = eh.get("d_diff_ci_traj")
                cts = f"[{ct[0]:+.2f}, {ct[1]:+.2f}]" if ct is not None and np.all(np.isfinite(ct)) else "·"
                L.append(f"| {tg} | {'영속' if sn == 'permanence' else '지속'} | {mo} | {ah} {eh['d_diff']:+.2f} | {cts} | {ap} {ep['d_diff']:+.2f} | "
                         f"{_arrow(eh, 'auroc')} {eh['d_auroc']:+.3f} / {_arrow(ep, 'auroc')} {ep['d_auroc']:+.3f} | "
                         f"{_arrow(eh, 'mass')} {eh['d_mass']:+.2f} | {_arrow(eh, 'mass_net')} {eh['d_mass_net']:+.2f} | {max(eh['units_pos'], eh['units_neg'])}/{n} | {v} |")
    return L


def _g(res, sn, mo, nm):
    return res["summaries"][f"{sn}|{mo}"]["readings"][nm]


def _dd(e, q="diff", p=2):
    """'+0.15 [+0.03, +0.29]' 형식의 Δ vs release."""
    return fci(e.get(f"d_{q}"), e.get(f"d_{q}_ci"), p=p, sign=True)


def _up(e, q="diff"):
    c = e.get(f"d_{q}_ci")
    return c is not None and np.isfinite(c[0]) and c[0] > 0


def _dn(e, q="diff"):
    c = e.get(f"d_{q}_ci")
    return c is not None and np.isfinite(c[1]) and c[1] < 0


def _flag(ok):
    return "" if ok else " ⚠️**(문장 검사 실패 — 이 문장을 수치에 맞게 고칠 것)**"


def _pos(res, key, nm, t, q):
    v = res["position"][key][nm][q][t]
    return v if v is not None else float("nan")


def _oa(res):
    """도메인 밖 셋의 영속 배치 범위 · 견고 목록 등 머리말과 정정 블록이 같이 쓰는 값."""
    g = lambda sn, mo, tg, pre="ph": _g(res, sn, mo, f"{pre}:{tg}")
    ood = [g("permanence", mo, t)["mass"] for t in ("ip1_e40", "ip2_e80", "ariel_ep43") for mo in ("flat", "ramp")]
    rob = [f"{t} {mo} {verdict(res, t, 'permanence', mo).split()[1]}" for t in FINAL[1:] for mo in ("flat", "ramp")
           if verdict(res, t, "permanence", mo).startswith("견고")]
    thr_only = [f"{t} {mo}" for t in FINAL[1:] for mo in ("flat", "ramp") if verdict(res, t, "permanence", mo).startswith("문턱만")]
    return dict(ood_lo=min(ood), ood_hi=max(ood), rob=rob, thr_only=thr_only)


def headline(res):
    g = lambda sn, mo, tg, pre="ph": _g(res, sn, mo, f"{pre}:{tg}")
    Pz, AB, CAB = res["position"], res["vanish_ab"], res.get("vanish_ab_curves")
    L = []
    ex = lambda key, tg, t: _pos(res, key, f"ph:{tg}", t, "err_x")
    vf, vr = g("permanence", "flat", "v11_e10"), g("permanence", "ramp", "v11_e10")
    pf, pr = g("permanence", "flat", "pv1_e15"), g("permanence", "ramp", "pv1_e15")
    rf, rr_ = g("permanence", "flat", "release"), g("permanence", "ramp", "release")
    hf = _g(res, "permanence", "flat", "hh")
    pv_null = [(ep, g("permanence", "flat", tg)["mass_null"], g("permanence", "flat", tg)["mass_net"]) for ep, tg in FAMILIES["pv1"] if ep > 0]
    pvB = [(ep, CAB["late|flat"]["k>=2"][tg]["B"] * 100) for ep, tg in FAMILIES["pv1"] if ep > 0] if CAB else []
    O = _oa(res)
    # 1. 영속 — 순 배치
    ok = vf["mass_net"] > 0.4 and vr["mass_net"] > 0.4 and pf["mass_null"] > 3 * vf["mass_null"] and pv_null[-1][1] > pv_null[0][1]
    L.append(f"- **영속 — 가려졌다 다시 나오는 물체 쪽으로 p 의 attention 을 확실히 모으는 것은 v11 궤적을 재생 학습한 v11_e10 하나다.** "
             f"v11 held-out block 절반, late 가림 k≥2, t1~t4 중 진실이 다시 보이는 칸의 진실 3×3 attention 질량 (문턱 없음, 균등 0.035; flat / ramp): "
             f"v11_e10 **{vf['mass']:.2f} / {vr['mass']:.2f}**, 물체가 한 번도 없던 빈 가림막 장면에서 같은 자리의 질량 (귀무) {vf['mass_null']:.2f} / {vr['mass_null']:.2f} "
             f"→ 순 배치 **{vf['mass_net']:+.2f} / {vr['mass_net']:+.2f}**; late flat t3 x 오차 {ex('late|flat', 'v11_e10', 3):+.0f} px (같은 자로 진짜 h 를 읽으면 {_pos(res, 'late|flat', 'hh', 3, 'err_x'):+.0f} px, 1 칸 18 px). "
             f"pv1_e15 도 질량은 {pf['mass']:.2f} / {pr['mass']:.2f} 로 오르지만 **빈 장면에서도 같은 자리에 {pf['mass_null']:.2f} / {pr['mass_null']:.2f}** 를 둔다 "
             f"(순 배치 {pf['mass_net']:+.2f} / {pr['mass_net']:+.2f}; x 오차 {ex('late|flat', 'pv1_e15', 3):+.0f} px = {abs(ex('late|flat', 'pv1_e15', 3)) / 18:.1f} 칸 뒤). "
             f"epoch 가 갈수록 이 귀무가 커지고 (flat " + " → ".join(f"e{ep} {nl:.2f}" for ep, nl, _ in pv_null) + ") "
             + (f"vanish B (빈→물체, late flat k≥2, curves 채점) 는 " + " → ".join(f"{b:.0f}" for _, b in pvB) + " % 로 떨어진다" if pvB else "")
             + f" — pv1 의 상당 부분은 '문맥과 무관하게 가림막 출구에서 물체가 나온다' 는 prior 이지 영속이 아니다. 둘 다 학습 도메인 안이다 (단서 2). "
             f"배치는 attention 이 모인 정도이지 정확도가 아니다 — v11_e10 의 {vf['mass']:.2f} 는 진짜 h 의 {hf['mass']:.2f} 보다 높다.{_flag(ok)}")
    # 2. 도메인 밖
    i1r = g("permanence", "ramp", "ip1_e40")
    ok = O["ood_hi"] < 0.2
    L.append(f"- **도메인 밖 셋 (IntPhys1 · IntPhys2 · Ariel) 은 영속 질량 {O['ood_lo']:.2f}–{O['ood_hi']:.2f} 로 release ({rf['mass']:.2f} / {rr_['mass']:.2f}) 수준에 머문다** "
             f"(순 배치 ip1_e40 {g('permanence', 'flat', 'ip1_e40')['mass_net']:+.2f} / {i1r['mass_net']:+.2f} · ip2_e80 {g('permanence', 'flat', 'ip2_e80')['mass_net']:+.2f} / {g('permanence', 'ramp', 'ip2_e80')['mass_net']:+.2f} · "
             f"ariel_ep43 {g('permanence', 'flat', 'ariel_ep43')['mass_net']:+.2f} / {g('permanence', 'ramp', 'ariel_ep43')['mass_net']:+.2f}). "
             f"ip1_e40 ramp 는 release 보다 조금 낮다 ({i1r['mass']:.3f} vs {rr_['mass']:.3f}).{_flag(ok)}")
    # 3. 고정 문턱
    i2f, i2r = g("permanence", "flat", "ip2_e80"), g("permanence", "ramp", "ip2_e80")
    i2fp, i2rp = g("permanence", "flat", "ip2_e80", "pp"), g("permanence", "ramp", "ip2_e80", "pp")
    i1f = g("permanence", "flat", "ip1_e40"); arf = g("permanence", "flat", "ariel_ep43")
    ok = i2f["d_diff"] < 0 < i2fp["d_diff"] and abs(i1f["d_auroc"]) < 0.05
    ct = vf.get("d_diff_ci_traj"); cp = pr.get("d_diff_ci_traj")
    L.append(f"- **고정 문턱 '있다' 비율 (present − 빈 바닥) 의 변화는 자를 바꾸면 부호가 뒤집히거나 문턱 없는 판독과 어긋난다.** "
             f"ip2_e80 영속 Δ 는 p@h {i2f['d_diff']:+.2f} / {i2r['d_diff']:+.2f} (flat / ramp) 인데 p@p {i2fp['d_diff']:+.2f} / {i2rp['d_diff']:+.2f}. "
             f"ip1_e40 flat 은 p@h {i1f['d_diff']:+.2f} 인데 AUROC Δ {i1f['d_auroc']:+.3f} · 순 배치 Δ {i1f['d_mass_net']:+.2f}; ariel_ep43 flat 은 p@h {arf['d_diff']:+.2f} 인데 AUROC Δ {arf['d_auroc']:+.3f} "
             f"(CI {arf['d_auroc_ci'][0]:+.2f}, {arf['d_auroc_ci'][1]:+.2f}) · 순 배치 Δ {arf['d_mass_net']:+.2f} (빈 바닥 {arf['floor']:.2f}). "
             f"두 자 · (궤적,k) 단위 · 문턱 없는 AUROC · 순 배치가 모두 같은 방향인 영속 변화 (표 A '견고') 는 {' · '.join(O['rob']) or '없음'} 뿐이고, "
             f"'문턱만' 인 것은 {' · '.join(O['thr_only']) or '없음'}. (궤적,k) 단위 6 개는 궤적 해시 **2 개**에서 나온다 — 궤적만 cluster 하면 "
             f"v11_e10 flat p@h Δ CI [{ct[0]:+.2f}, {ct[1]:+.2f}] · pv1_e15 ramp [{cp[0]:+.2f}, {cp[1]:+.2f}] 로 0 을 걸친다.{_flag(ok)}")
    # 4. 채점 교차검증 + 복사 기준선
    if AB and CAB:
        a = lambda ck, t: AB[ck][t]["A_vanish"] * 100
        cA = lambda ck, ks, t: CAB[ck][ks][t]["A"] * 100
        ok = abs(a("late|flat", "ariel_ep43") - cA("late|flat", "k>=2", "copy")) < 5 and cA("late|flat", "k=1", "pv1_e15") < cA("late|flat", "k=1", "copy")
        L.append(f"- **자 없는 교차검증 (vanish 채점, 물체→빈 A, late k≥2, flat / ramp)**: 복사 기준선 **{cA('late|flat', 'k>=2', 'copy'):.0f} / {cA('late|ramp', 'k>=2', 'copy'):.0f} %** · "
                 f"release {a('late|flat', 'release'):.0f} / {a('late|ramp', 'release'):.0f} · v11_e10 {a('late|flat', 'v11_e10'):.0f} / {a('late|ramp', 'v11_e10'):.0f} · "
                 f"pv1_e15 {a('late|flat', 'pv1_e15'):.0f} / {a('late|ramp', 'pv1_e15'):.0f} · ip2_e80 {a('late|flat', 'ip2_e80'):.0f} / {a('late|ramp', 'ip2_e80'):.0f} · "
                 f"ariel_ep43 {a('late|flat', 'ariel_ep43'):.0f} / {a('late|ramp', 'ariel_ep43'):.0f} · ip1_e40 {a('late|flat', 'ip1_e40'):.0f} / {a('late|ramp', 'ip1_e40'):.0f} %. "
                 f"**Ariel 은 복사 수준**이다. v11_e10 · pv1_e15 가 맨 위인 것은 **k≥2 에서만**이다 — k=1 flat (28 block, curves 채점) 에서는 pv1_e15 {cA('late|flat', 'k=1', 'pv1_e15'):.0f} % 가 "
                 f"복사 {cA('late|flat', 'k=1', 'copy'):.0f} · ip2_e80 {cA('late|flat', 'k=1', 'ip2_e80'):.0f} · ariel_ep43 {cA('late|flat', 'k=1', 'ariel_ep43'):.0f} 보다 낮다. "
                 f"late static A 는 pv1_e15 {a('late|static', 'pv1_e15'):.0f} · ariel_ep43 {a('late|static', 'ariel_ep43'):.0f} · release {a('late|static', 'release'):.0f} % 로 복사 {cA('late|static', 'k>=2', 'copy'):.0f} % 아래다.{_flag(ok)}")
    # 5. 지속
    au_min = min(min(g("persistence", mo, t, pre)["auroc"] for t in FINAL for pre in ("ph", "pp")) for mo in ("flat", "ramp"))
    cpa = min(_g(res, "persistence", mo, "copy_hctx")["auroc"] for mo in ("flat", "ramp")) if "copy_hctx" in res["summaries"]["persistence|flat"]["readings"] else float("nan")
    xs = {t: ex("visible|flat", t, 4) for t in FINAL}
    ok = au_min > 0.95 and xs["release"] < min(xs[t] for t in FINAL[1:])
    L.append(f"- **지속 — 보이는 물체 (visible) 는 모든 predictor · 두 자가 AUROC ≥ {au_min:.2f} 이지만 복사 기준선 (h_ctx@h) 도 {cpa:.2f} 이라 예측 능력의 증거가 아니다.** "
             f"갈리는 것은 자리다: visible flat t4 부호 x 오차 release {xs['release']:+.0f} · ip2_e80 {xs['ip2_e80']:+.0f} · ariel_ep43 {xs['ariel_ep43']:+.0f} · ip1_e40 {xs['ip1_e40']:+.0f} · "
             f"v11_e10 {xs['v11_e10']:+.0f} · pv1_e15 {xs['pv1_e15']:+.0f} px. 이 뒤처짐 순서는 **두 궤적 안에서만** 일관된다 (visible CI 단위 = 궤적 2 개).{_flag(ok)}")
    L.append("- **궤적이 운동 조건마다 2 개뿐이다** (좌→우 · 우→좌; (궤적,k) 단위 = late 6 · visible 2, 그러나 궤적 해시는 둘). 위 결론은 모양 · 색 · 배경을 가로질러서만 일반화되고, "
             "궤적 · 속도 · 가속을 가로지르는 일반화는 이 데이터로 말할 수 없다 (단서 1–2).")
    return L


def correction_block(res):
    """적대 검증 2026-09-25 반영 목록 (이전 문장 → 지금 문장). 수치는 이 실행에서 다시 계산."""
    g = lambda sn, mo, tg, pre="ph": _g(res, sn, mo, f"{pre}:{tg}")
    AB, CAB = res["vanish_ab"], res.get("vanish_ab_curves")
    O = _oa(res)
    vf, vr = g("permanence", "flat", "v11_e10"), g("permanence", "ramp", "v11_e10")
    pf, pr = g("permanence", "flat", "pv1_e15"), g("permanence", "ramp", "pv1_e15")
    pv = [(ep, g("permanence", "flat", tg)) for ep, tg in FAMILIES["pv1"] if ep > 0]
    ct, cp = vf["d_diff_ci_traj"], pr["d_diff_ci_traj"]
    cpa = _g(res, "persistence", "flat", "copy_hctx")["auroc"]
    B = [f"{CAB['late|flat']['k>=2'][tg]['B']*100:.0f}" for ep, tg in FAMILIES["pv1"] if ep > 0] if CAB else []
    cA = lambda ck, ks, t: CAB[ck][ks][t]["A"] * 100
    L = ["> ⚠️ **정정 (2026-09-25, 적대 검증 반영)** — 무엇이 바뀌었나 (이전 문장 → 지금 문장). 수치는 이 실행에서 다시 계산했다. 검증 원문: `../Archive/verify_raw/verify_presence.json`.",
         ">",
         f"> 1. **배치에 빈 장면 귀무를 뺐다.** 이전 '가려졌다 나오는 물체를 제자리에 두는 것은 도메인 안 두 predictor (v11_e10 · pv1_e15) 뿐' → 지금 "
         f"'확실히 모으는 것은 v11_e10 하나'. 순 배치 (배치 − 물체가 없던 빈 장면의 같은 자리 질량) flat / ramp: v11_e10 {vf['mass_net']:+.2f} / {vr['mass_net']:+.2f} (귀무 {vf['mass_null']:.2f} / {vr['mass_null']:.2f}), "
         f"pv1_e15 {pf['mass_net']:+.2f} / {pr['mass_net']:+.2f} (귀무 **{pf['mass_null']:.2f} / {pr['mass_null']:.2f}**, 배치 {pf['mass']:.2f} / {pr['mass']:.2f}). pv1 배치의 상당 부분은 문맥 무관 '가림막 출구 출현' prior.",
         f"> 2. **pv1 학습 곡선 해석을 뒤집었다.** 이전 '배치가 그대로인데 문턱 판독만 움직이는 것은 척도 · 빈 바닥 이동' → 지금 'pv1 은 epoch 따라 문맥 무관 출현 prior 를 키운다': "
         f"귀무 flat " + " → ".join(f"e{ep} {e['mass_null']:.2f}" for ep, e in pv) + ", 순 배치 " + " → ".join(f"{e['mass_net']:.2f}" for _, e in pv)
         + (f", vanish B (빈→물체, late flat k≥2, curves 채점) " + " → ".join(B) + " %." if B else "."),
         f"> 3. **도메인 밖 수치 정정.** 이전 '0.1 안팎, release 보다 조금 높을 뿐' → 지금 '{O['ood_lo']:.2f}–{O['ood_hi']:.2f}, release 수준; ip1_e40 ramp ({g('permanence', 'ramp', 'ip1_e40')['mass']:.3f}) 는 release ({g('permanence', 'ramp', 'release')['mass']:.3f}) 보다 조금 낮다'.",
         f"> 4. **'견고' 판정 규칙을 바꿨다.** 이전: 두 자 (p@h · p@p) 같은 방향 + (궤적,k) 단위 전부 같은 부호 → 지금: 여기에 **문턱 없는 AUROC Δ 와 순 배치 Δ 도 같은 방향 (CI 가 0 을 넘지 않음)**. "
         f"지금 견고: {' · '.join(O['rob']) or '없음'}. 이전 '견고' 4 개 (v11_e10 ramp ↑ · ip1_e40 flat ↓ · ip1_e40 ramp ↓ · ariel_ep43 flat ↓) 중 '문턱만' 으로 내려간 것: {' · '.join(O['thr_only']) or '없음'}.",
         f"> 5. **궤적만 cluster 한 CI 를 붙였다** (표 A 열). (궤적,k) 6 단위는 궤적 해시 2 개에서 나온다. v11_e10 flat p@h Δ [{ct[0]:+.2f}, {ct[1]:+.2f}] · pv1_e15 ramp [{cp[0]:+.2f}, {cp[1]:+.2f}] — 0 을 걸친다.",
         (f"> 6. **표 7 에 복사 기준선 줄과 k=1 표 (7b) 를 더했다.** 복사 A late flat / ramp k≥2 = {cA('late|flat', 'k>=2', 'copy'):.0f} / {cA('late|ramp', 'k>=2', 'copy'):.0f} % → ariel_ep43 ({AB['late|flat']['ariel_ep43']['A_vanish']*100:.0f} / {AB['late|ramp']['ariel_ep43']['A_vanish']*100:.0f}) 는 복사 수준. "
          f"이전 '배치가 모인 두 predictor 가 맨 위' → 지금 'k≥2 에서만': k=1 flat 에서 pv1_e15 {cA('late|flat', 'k=1', 'pv1_e15'):.0f} % < 복사 {cA('late|flat', 'k=1', 'copy'):.0f}. "
          f"late static A 는 pv1_e15 · ariel_ep43 · release 가 복사 ({cA('late|static', 'k>=2', 'copy'):.0f} %) 아래." if CAB else "> 6. (curves 채점 없음 — 복사 줄 미적용)"),
         f"> 7. **지속 문장.** 이전 '보이는 물체는 아무도 잃지 않는다' → 지금 'AUROC ≥ 0.97 이지만 복사 기준선도 {cpa:.2f} 이라 예측 능력의 증거가 아니다'.",
         "> 8. **위치 x 오차는 진실칸 질량 중앙값이 균등 (0.035) 을 넘는 칸에서만** 읽기 본문에 적는다 (규칙 d). 그 밖은 '기본값' 으로 표시 (release late ramp t3 · visible ramp t4 등).",
         "> 9. 배치는 'attention 이 모인 정도' 이지 정확도가 아니다 (v11_e10 배치 > h@h) — 머리말 문장에서 '제자리에' 를 뺐다.",
         ">",
         "> **미적용**: (a) vanish A/B CI 는 block bootstrap 그대로 — 궤적 2 개 구조를 반영한 CI 는 만들지 않았다 (단서 7b). "
         "(b) readout (bf16) 과 채점 (fp32 + autocast fp16) 이 다른 실행이라는 점 · curves 채점과 하네스 최종 채점의 작은 차이 (ip2 A 46.4 vs 47.6 등) 는 재추출 (GPU) 이 필요해 그대로 둔다. "
         "(c) 지속 (visible, 단위 2 개) 의 ↑/↓ 화살표는 남겼다 — 궤적만 CI 열과 같이 읽는다. "
         "(d) late flat t4 '있다 ∧ mass3>0.035' 를 빈 장면에도 적용한 비율 (검증: pv1 42 % vs 물체 66 %) 은 표로 만들지 않았다 — 귀무 질량 (1) 이 같은 것을 문턱 없이 잰다.",
         ""]
    return L


def reading_text(res):
    g = lambda sn, mo, tg, pre="ph": _g(res, sn, mo, f"{pre}:{tg}")
    Pz = res["position"]
    L = ["## 읽기", ""]
    P = L.append
    P("**1. 영속 — 세 가지 판독이 무엇을 말하나 (표 A · 1 · 5 · 6)**")
    P("")
    hh_f, hh_r = _g(res, "permanence", "flat", "hh"), _g(res, "permanence", "ramp", "hh")
    ch_f, ch_r = _g(res, "permanence", "flat", "copy_hctx"), _g(res, "permanence", "ramp", "copy_hctx")
    rf, rr = g("permanence", "flat", "release"), g("permanence", "ramp", "release")
    P(f"- 참조: 같은 h 자로 **진짜 미래 h** 를 읽으면 present − 바닥 flat {hh_f['diff']:+.2f} · ramp {hh_r['diff']:+.2f} (t1 은 k≥3 이면 아직 가려져 1 이 아니다). "
      f"문맥 끝 튜블릿 **복사** (h_ctx@h) 는 {ch_f['diff']:+.2f} · {ch_r['diff']:+.2f}. release p@h 는 {rf['diff']:+.2f} · {rr['diff']:+.2f} 지만 배치는 {rf['mass']:.2f} · {rr['mass']:.2f} "
      f"(균등 0.035) — release 의 '있다' 는 **퍼진 attention 의 기본값 읽기**다. 위치를 말하지 않는다.")
    for tg in FINAL[1:]:
        ef, er = g("permanence", "flat", tg), g("permanence", "ramp", tg)
        P(f"- **{tg}**: present − 바닥 flat {ef['diff']:+.2f} (Δ {_dd(ef)}, 바닥 {ef['floor']:.2f}) · ramp {er['diff']:+.2f} (Δ {_dd(er)}, 바닥 {er['floor']:.2f}); "
          f"AUROC Δ {_dd(ef, 'auroc')} / {_dd(er, 'auroc')}; 배치 {ef['mass']:.2f} / {er['mass']:.2f} − 빈 장면 귀무 {ef['mass_null']:.2f} / {er['mass_null']:.2f} "
          f"= **순 배치 {ef['mass_net']:+.2f} / {er['mass_net']:+.2f}** (Δ {_dd(ef, 'mass_net')} / {_dd(er, 'mass_net')}).")
    pf1 = Pz["late|flat"]["ph:pv1_e15"]
    pvf = g('permanence', 'flat', 'pv1_e15')
    P(f"- pv1_e15 flat: t1·t2 에 늦게 되살리고 (present {pf1['present'][1]:.2f} · {pf1['present'][2]:.2f}, 진짜 h {Pz['late|flat']['hh']['present'][1]:.2f} · {Pz['late|flat']['hh']['present'][2]:.2f}) "
      f"t3 부터 빈 장면 바닥이 오른다 (표 5). ⚠️ 정정 — 이전에는 '되살린 칸에서 진실 근처에 둔다 (배치 {pvf['mass']:.2f})' 로 읽었지만, 물체가 한 번도 없던 빈 가림막 장면에서도 같은 자리에 "
      f"{pvf['mass_null']:.2f} 를 둔다 (귀무). 순 배치는 {pvf['mass_net']:+.2f} 이고 나머지는 문맥과 무관한 '가림막 출구에서 물체가 나온다' prior 다.")
    P("- 고정 문턱 판독은 두 자 사이에서 ip2_e80 이 정반대로 나오고, ip1_e40 flat 은 AUROC 가 그대로다 — 척도 이동. 문턱 없는 판독 중 **순 배치 (배치 − 빈 장면 귀무)** 가 "
      "v11_e10 을 크게 올리고 pv1_e15 를 그 절반쯤 올린다. 도메인 밖 셋은 release 수준이다.")
    P("")
    P("**2. 위치 — '어딘가에 있다' 와 '있어야 할 자리에 있다' (표 6, 그림 position)**")
    P("")
    for key in ("late|flat", "late|ramp", "visible|flat", "visible|ramp"):
        t = 3 if key.startswith("late") else 4
        parts = []
        for tg in FINAL:
            e = Pz[key][f"ph:{tg}"]
            okm = e["mass_gt_med"][t] is not None and np.isfinite(e["mass_gt_med"][t]) and e["mass_gt_med"][t] > MASS_U
            parts.append(f"{tg} 질량 {e['mass_gt_med'][t]:.2f} · x 오차 " + ((f"{e['err_x'][t]:+.0f}" if e["err_x"][t] is not None and np.isfinite(e["err_x"][t]) else "·")
                                                                          if okm else "기본값 (중앙 질량 ≤ 균등, 위치 주장 안 함)"))
        P(f"- {key.replace('|', ' · ')} t{t} (진실 전진 {Pz[key]['prog_gt_all'][t]:.0f} px): " + "; ".join(parts))
    ms_max = max(float(np.max(Pz[k][f"ph:{t}"]["mass_seen_med"][2:])) for k in ("late|flat", "late|ramp", "visible|flat", "visible|ramp") for t in FINAL)
    P("- 질량 = 진실 3×3 attention 중앙값 (물체 clip 전부), x 오차 = '있다' ∧ 질량 > 0.035 인 clip 의 부호 x 오차 (+ = 앞섬). "
      f"마지막 관측 자리의 3×3 질량 중앙값은 네 칸 · 여섯 predictor 모두 t2 부터 ≤ {ms_max:.2f} 다 — **release 의 뒤처짐은 '마지막 자리에 머문다' 가 아니라 attention 이 퍼진 것**이다 (CLAUDE.md §5-5, 멈춤으로 읽지 않는다).{_flag(ms_max <= 0.12)} "
      "mass3 > 0.035 는 필요조건일 뿐이라 x 오차가 수십 px 인 칸도 통과한다.")
    P("")
    P("**3. 지속 — 보이는 물체 (표 2 · A)**")
    P("")
    for mo in ("flat", "ramp"):
        P(f"- {mo}: " + " · ".join(f"{tg} present {g('persistence', mo, tg)['present']:.2f} / p@p {g('persistence', mo, tg, 'pp')['present']:.2f} / AUROC {g('persistence', mo, tg)['auroc']:.2f} / 배치 {g('persistence', mo, tg)['mass']:.2f}" for tg in FINAL))
    cp = _g(res, "persistence", "flat", "copy_hctx")
    P(f"- present 는 **복사 기준선과 구별되지 않는다** (h_ctx@h 복사 {cp['present']:.2f}) — 보이는 물체의 '있다' 는 예측 없이도 나온다. 그래서 지속은 배치 · x 오차로 읽는다. "
      "배치는 다섯 predictor 모두 release 보다 높고 (도메인 밖 ip1_e40 · ariel_ep43 포함), 그 크기 순서는 v11_e10 ≈ pv1_e15 > ariel_ep43 > ip1_e40 > ip2_e80 (flat). "
      "단 visible 은 CI 단위가 **2 개** (두 궤적) 라 이 순서는 '두 궤적 모두에서' 이상을 말하지 않는다.")
    P("")
    P("**4. 학습 곡선 (표 4, 그림 learning_curves)**")
    P("")
    for fam, lst in FAMILIES.items():
        pm = [f"e{ep} {_g(res, 'permanence', 'flat', f'ph:{tg}')['mass']:.2f}/{_g(res, 'permanence', 'ramp', f'ph:{tg}')['mass']:.2f}" for ep, tg in lst]
        pf = [f"e{ep} {_g(res, 'permanence', 'flat', f'ph:{tg}')['diff']:+.2f}" for ep, tg in lst]
        ps = [f"e{ep} {_g(res, 'persistence', 'flat', f'ph:{tg}')['mass']:.2f}" for ep, tg in lst]
        P(f"- **{FAM_NAME[fam]}** — 영속 배치 flat/ramp: {' → '.join(pm)}; 영속 present−바닥 flat: {' → '.join(pf)}; 지속 배치 flat: {' → '.join(ps)}")
    v1 = _g(res, "permanence", "flat", "ph:v11_e1")["mass"]; p1 = _g(res, "permanence", "flat", "ph:pv1_e1")["mass"]
    pvl = [(ep, _g(res, "permanence", "flat", f"ph:{tg}")) for ep, tg in FAMILIES["pv1"] if ep > 0]
    CAB = res.get("vanish_ab_curves")
    pvB = " → ".join(f"{CAB['late|flat']['k>=2'][tg]['B']*100:.1f}" for ep, tg in FAMILIES["pv1"] if ep > 0) if CAB else "·"
    P(f"- **v11 · Predictor_v1 의 배치는 첫 epoch 에 거의 다 생긴다** (영속 flat 배치 e1 = {v1:.2f} / {p1:.2f}, release {rf['mass']:.2f}) 그 뒤 물체 clip 배치는 평평하다. "
      "⚠️ 정정 — 이전에는 'Predictor_v1 flat 의 present − 바닥 하락은 배치가 그대로이니 척도 · 빈 바닥 이동' 이라 읽었다. 문턱 없는 판독 둘이 이를 반박한다: "
      "빈 장면 귀무 (물체가 없던 장면의 진실칸 질량) 가 " + " → ".join(f"e{ep} {e['mass_null']:.2f}" for ep, e in pvl) +
      f" 로 오르고, vanish B (빈→물체, late flat k≥2, curves 채점) 가 {pvB} % 로 떨어진다. **Predictor_v1 은 epoch 따라 문맥과 무관한 '가림막 출구 출현' prior 를 키운다** — "
      "순 배치 " + " → ".join(f"{e['mass_net']:.2f}" for _, e in pvl) + " 로 줄어든다. 예측이 실제로 바뀐 것이지 자의 척도 문제가 아니다. "
      "v11 은 귀무가 " + " → ".join(f"e{ep} {_g(res, 'permanence', 'flat', f'ph:{tg}')['mass_null']:.2f}" for ep, tg in FAMILIES["v11"] if ep > 0) + " 로 낮게 머문다. "
      "IntPhys1 은 e5 에서 고정 문턱 영속이 바닥으로 가고 (AUROC 는 release 수준) 영속 배치는 끝까지 release 수준이다. IntPhys2 는 영속 배치가 거의 안 움직인다. "
      "Ariel (scratch) 은 지속 배치가 epoch 따라 오르고 (ep5 → ep43) 영속 배치는 0.1 안팎에 머문다.")
    P("")
    AB = res["vanish_ab"]
    if AB:
        P("**5. 자 없는 교차검증 — vanish 방향별 채점 (표 7)**")
        P("")
        for ck in ("late|flat", "late|ramp", "late|static"):
            e = AB[ck]
            P(f"- {ck.replace('|', ' · ')} (block {e['n_block']}): A 물체→빈 " + " · ".join(f"{tg} {e[tg]['A_vanish']*100:.0f}" for tg in FINAL)
              + " % / B 빈→물체 " + " · ".join(f"{e[tg]['B_appear']*100:.0f}" for tg in FINAL) + " %")
        kd = lambda tg: AB["late|flat"][tg]
        fmt = lambda v: f"{v*100:.0f}" if v is not None and np.isfinite(v) else "·"
        CAB = res.get("vanish_ab_curves")
        if CAB:
            P("- 복사 기준선 (curves 채점, 같은 block): A 물체→빈 " + " · ".join(f"{ck.split('|')[1]} {CAB[ck]['k>=2']['copy']['A']*100:.0f}" for ck in ("late|flat", "late|ramp", "late|static"))
              + " % — **ariel_ep43 의 late A 는 복사 수준**이고, late static 에서 pv1_e15 · ariel_ep43 · release 는 복사 아래다.")
            P("- ⚠️ 정정 — 맨 위 둘 (v11_e10 · pv1_e15) 순서는 **k≥2 에서만** 성립한다. k=1 (curves 채점, 표 7b) late flat A: "
              + " · ".join(f"{tg} {CAB['late|flat']['k=1'][tg]['A']*100:.0f}" for tg in FINAL + ["copy"]) + " %.")
        P("- k≥2 의 A 순서는 영속 **배치** 순서와 맨 위 둘 (v11_e10 · pv1_e15) 이 같다. 고정 문턱 영속과는 어긋난다 — ip2_e80 · ariel_ep43 은 p@h 영속이 release 보다 낮은데 A 는 높다. "
          "같은 block 의 pos_a 를 p@h 가 '유지 / 놓침' 으로 읽은 것으로 A 를 나누면 (late flat): "
          + " · ".join(f"{tg} {fmt(kd(tg)['A_given_keep'])} / {fmt(kd(tg)['A_given_drop'])} % (n {kd(tg)['n_keep']} / {kd(tg)['n_drop']})" for tg in FINAL)
          + " — release · v11_e10 · pv1_e15 에서는 '있다' 비트가 A 를 가르지 않는다. "
          f"late static (멈춘 물체) 은 pv1_e15 · ariel_ep43 이 A 를 release 보다 **떨어뜨린다** (Δ {AB['late|static']['pv1_e15']['dA']*100:+.0f} · {AB['late|static']['ariel_ep43']['dA']*100:+.0f} pt) — 움직이는 조건의 향상과 반대다.")
    return L


def caveats(res):
    S = res["summaries"]
    nu_l = S["permanence|flat"]["n_units"]; nu_v = S["persistence|flat"]["n_units"]
    L = ["## 단서", ""]
    P = L.append
    P(f"1. **궤적 2 개 (CI 단위 문제)**. 움직이는 조건마다 진실 궤적은 좌→우 · 우→좌 둘뿐이다 (test 절반에서도 같다; 1 px 해시로 확인). CI 단위 (궤적, k) 는 late 에서 {nu_l} 개, "
      f"visible 에서 **{nu_v} 개** — visible 의 CI 는 두 궤적 사이 차이를 보여 줄 뿐 모집단 CI 가 아니다. 단위 안의 수백 clip 은 모양 · 색 · 배경만 다른 외형 복사다. "
      "**살아남는 결론**: (궤적,k) 단위 전부에서 Δ 의 부호가 같은 것 (표 8) — '이 두 궤적 · 이 외형 범위에서 방향이 일관된다'. "
      "**못 하는 결론**: 다른 궤적 · 속도 · 가속으로 일반화, '처음 보는 운동에서도 영속이 좋아진다'.")
    P("2. **v11_e10 은 궤적 재생이다**. `v11_split_train` 은 같은 두 궤적의 가능 clip 이다 — held-out block 이어도 '처음 보는 궤적' 이 아니다 (MODELS.md). "
      "v11_e10 의 영속 · 위치 향상은 재생의 상한으로 읽는다. **pv1_e15 도 도메인 안이다**: Predictor_v1_training 에 무중력 등속 팔 (속도 다양), **가림 k=4 팔**, 경사 가속 팔이 있다 — "
      "v11 late k≥2 · flat · ramp 가 모두 그 학습 분포의 유형이다 (렌더 · 궤적은 다르다). 둘 다 도메인 밖 증거가 아니다. 도메인 밖은 IntPhys1 · IntPhys2 · Ariel 쪽이고, "
      "셋 다 영속 배치가 release 수준 (0.04–0.13) 에 머문다 — 고정 문턱 판독은 p@h 로 내려가지만 ip2_e80 은 p@p 로 반대다 (표 A). "
      "pv1_e15 는 빈 장면 귀무가 커서 (late flat 0.2 안팎) 배치의 상당 부분이 문맥 무관 prior 다 — 순 배치로 읽는다.")
    P("3. **자 이식**. 자 (p 자 · h 자) 는 릴리즈 특징으로 학습 · 보정했다. predictor 를 학습하면 p 의 logit 척도가 움직인다 — 고정 문턱의 present 는 '물체를 들고 있나' 와 '척도가 움직였나' 를 섞는다. "
      "그래서 빈 바닥과 AUROC 를 같은 칸에 붙였다. Ariel 은 빈 바닥이 높아 (late 0.6 안팎) 고정 문턱 present 만으로는 아무 말도 못 한다. p@p (릴리즈 p 로 학습한 자) 는 이식이 한 번 더 들어간 판독이라 보조로만 읽는다.")
    P("3b. **배치 (진실칸 attention 질량) 도 자에 기댄다** — h 자의 attention query 를 p 에 건 것이다. 문턱은 없지만 자 이식 문제는 남는다. "
      "그리고 '모였다' 는 '정확하다' 가 아니다: v11_e10 의 영속 배치 (0.6 안팎) 가 진짜 h 를 같은 자로 읽은 값 (0.4~0.5) 보다 높지만 x 오차는 h@h 쪽이 작다. 배치는 x 오차와 같이 읽는다.")
    P("4. **AUROC 의 대조군은 '처음부터 물체가 없는 장면'** (vanish pos_b) 이다 — 문맥에도 물체가 없다. 그래서 AUROC 는 '문맥에 물체가 있었다는 흔적이 미래 p 에 남나' 까지 잡고, "
      "물체를 **그 자리에** 두는지는 말하지 않는다. 위치는 표 6 으로 따로 본다.")
    P("5. **지속은 t3~t4/t5 까지만 잴 수 있다**. 그 뒤 v11 의 움직이는 물체는 가장자리 36 px 안이나 화면 밖이다 (h@h 도 그 칸에서 0). '미래 끝까지 남나' 는 v11 로 답할 수 없다.")
    P("6. **h 자 자체의 한계**. 가려진 물체도 h 자가 일부 '있다' 로 읽는다 (가림 편향) — late t0 · t1 의 값은 자 탓일 수 있다. 영속 요약의 t1 은 k≥3 이면 진실에서도 가려져 있다. h@h 줄이 그 천장이다.")
    P("7. **위치의 선택 편향**. 위치는 '있다' ∧ mass3 > 0.035 인 clip 만 센다 — 진실 근처에 attention 이 있는 clip 을 고른 것이라 L2 는 낙관적이다. 비교는 '진실칸/있다' 비율과 같이 읽는다.")
    P("7b. **vanish A/B 의 CI 는 block bootstrap** (v11 채점 규칙) 이라 궤적 2 개 구조를 반영하지 않는다 — 실제 불확실성은 CI 보다 크다 (단서 1).")
    P("8. **정밀도 · 실행이 다르다**. readout 은 bf16 (autocast 없음, `te_v11_readout.py`), vanish A/B 는 표준 채점 (fp32 + autocast fp16, `scores.json`). 같은 block 이지만 다른 실행이다. "
      "bf16 predictor 잡음은 위치 ~1 px · '있다' 판정 불일치 ≤ 1 % 수준 (추출 스크립트 검증).")
    P("9. **학습 한 번씩** (seed 0) — 학습 분산을 모른다. Ariel 은 attention 규칙 · 데이터 · scratch · epoch 네 가지가 동시에 달라 원인을 좁힐 수 없다. 학습 목표를 원인으로 걸지 않는다.")
    P("10. **복사 기준선**은 다른 세션의 감사 산출물 (`_audit/v11_presence/ctx_only_readings.npz`, 릴리즈 encoder, 문맥 16 샘플만) 을 읽기 전용으로 썼다 — predictor 와 무관해 모든 줄에 공통이다. "
      "z_ctx 복사는 z 자로 읽은 값이라 h 자 줄과 척도가 다르다.")
    if res["n_test_done"] != res["n_test"]:
        P(f"11. ⚠️ **추출 미완료** — test 절반 {res['n_test_done']:,} / {res['n_test']:,} clip 만 들어갔다. 칸마다 n 은 표에 있다.")
    return L


if __name__ == "__main__":
    main()
