#!/usr/bin/env python3
"""TrainingEffects — v11_split_test 점수 심층 분석: 복사 기준선 · 학습 곡선 · 슬롯별 · '예측한 변화량' (CPU 전용).

    PY=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
    $PY z_research/scripts/analysis/te_analyze_scores.py                 # -> z_research/TrainingEffects/scores/ANALYSIS.md + figures
    $PY z_research/scripts/analysis/te_analyze_scores.py --boot 200      # 빠른 개발용 (CI 거칠다)

입력 (읽기만 한다):
  /data2/.../training_effects/v11score_vith_curves/   te_v11_score.py 산출 (l1 · pz · pstep (n,K,8), copy · hcopy · hstep (n,8), done)
      정의는 te_v11_score.py docstring. done==1 행만 쓴다 (추출이 진행 중이면 부분 데이터 — md 머리에 n 을 밝힌다).
  /data2/.../training_effects/v11score_<그룹B모델>_own/   있으면 (그 encoder 자신의 copy 기준선)
  하네스 per_video_surprise (training_effects_scores.MODELS 의 폴더) — final 6 개 · 그룹 B 의 슬롯 평균 점수 (10,752 쌍 전부).
  data_csv/intphysgen_v11_split/index_test.csv — role (vanish 방향) · reveal_frame.

채점: 쌍 (block_id, pair_id) 마다 hit = 1[S_pos < S_imp] + 0.5 동점 (eval.py::score_blocks 와 같은 규칙).
  S = 슬롯 평균 l1 (= 표준 surprise) · 슬롯 j 만 · 드러남 단계 (hidden / reveal / after) 평균.
  copy: S_copy = mean_j copy[:, j] = mean_j |LN(z)[문맥 마지막 튜블릿] − h[8+j]| — 예측 없는 기준선. 한 쌍의 두 clip 은
  문맥이 같아 zT 가 같으므로 copy 는 "두 미래 중 어느 쪽이 마지막 관측과 가까운가" 를 묻는다.
CI: block bootstrap (block = 문맥을 공유하는 4 clip = 2 쌍), 칸마다 resample 하나를 모든 열이 공유 → 같은 쌍 위의 짝지은 차이.
  칸의 seed 는 (seed, crc32(칸 이름)) 이라 칸 순서·다른 칸과 무관하게 재현된다.
'예측한 변화량': 가능 clip 만 (불가능 clip 의 h 는 위반된 미래다). 크기 (latent L1) 이지 정답 여부가 아니다.

수치는 전부 배열에서 다시 계산한다. md 는 이 스크립트가 쓴다 (손으로 고치지 않는다).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
import training_effects_scores as TES  # noqa: E402  (읽기 전용 재사용: load_pairs · MODELS · timing_of · motion_of)

CACHE = Path("/data2/local_datasets/world/world_analysis/cache/training_effects")
CURVES = CACHE / "v11score_vith_curves"
OUT = REPO / "z_research/TrainingEffects/scores"
INDEX = REPO / "data_csv/intphysgen_v11_split/index_test.csv"
SCORES_JSON = OUT / "scores.json"
H6F = REPO / "auto_research/exp_results/h6/h6f_v11.json"

FINAL = ["release", "v11_e10", "pv1_e15", "ip1_e40", "ip2_e80", "ariel_ep43"]
TRAINED = FINAL[1:]
HKEY = {"release": "release", "v11_e10": "pft_v11", "ip1_e40": "pft_intphys1", "ip2_e80": "pft_intphys2",
        "pv1_e15": "pft_predv1", "ariel_ep43": "ariel_bc43"}
HARNESS_REF = {"release": 75.83, "v11_e10": 91.35, "ip1_e40": 78.53, "ip2_e80": 79.56, "pv1_e15": 81.63, "ariel_ep43": 79.39}
RUNS = {  # 학습 곡선: (표시 이름, [(epoch, tag)]) — post-FT 는 e0 = release, ariel 은 e0 없음
    "v11": ("v11 post-FT", [(0, "release"), (1, "v11_e1"), (2, "v11_e2"), (3, "v11_e3"), (5, "v11_e5"), (10, "v11_e10")]),
    "pv1": ("Predictor_v1 post-FT", [(0, "release"), (1, "pv1_e1"), (3, "pv1_e3"), (5, "pv1_e5"), (10, "pv1_e10"), (15, "pv1_e15")]),
    "ip1": ("IntPhys1 post-FT", [(0, "release"), (5, "ip1_e5"), (10, "ip1_e10"), (20, "ip1_e20"), (40, "ip1_e40")]),
    "ip2": ("IntPhys2 post-FT", [(0, "release"), (10, "ip2_e10"), (40, "ip2_e40"), (80, "ip2_e80")]),
    "ariel": ("Ariel scratch", [(5, "ariel_ep5"), (19, "ariel_ep19"), (30, "ariel_ep30"), (43, "ariel_ep43")]),
}
DOMAIN = {"v11_e10": "안 (v11 궤적 재생)", "pv1_e15": "안 (Predictor_v1 — 가림·경사 팔이 있어 README 규칙상 v11 도메인 안)",
          "ip1_e40": "밖 (IntPhys1 장면)", "ip2_e80": "밖 (IntPhys2, 254 clip)", "ariel_ep43": "밖 (자연 영상, scratch)"}
GROUPB = ["vitb_k400_e240", "vitb_k400_e100", "vitb_synphys_30k", "vittiny_k400_5k", "vittiny_synphys_5k",
          "vittiny_k400", "vittiny_synphys_30k_e225", "vittiny_k400ssv2_30k_e225"]
GB_HKEY = {"vittiny_k400": "vittiny_k400_30k"}
GB_MIDTRAIN = {"vittiny_synphys_30k_e225", "vittiny_k400ssv2_30k_e225", "vitb_k400_e100"}  # 학습 중간 checkpoint (MODELS.md)

TIMINGS = ("visible", "early", "mid", "late")
MOTIONS = ("static", "flat", "ramp")
VKINDS = ("van_o2e", "van_e2o", "shape", "color")      # vanish 는 방향으로 쪼갠다
VK_EN = {"van_o2e": "vanish obj→empty", "van_e2o": "vanish empty→obj", "shape": "shape", "color": "color"}
VK_KO = {"van_o2e": "vanish 물체→빈", "van_e2o": "vanish 빈→물체", "shape": "shape", "color": "color"}
ROLE_OF = {"van_o2e": "imp_vanish", "van_e2o": "imp_appear"}
PHASES = ("hidden", "reveal", "after")
N_SLOT = 8

C_BLUE, C_ORANGE, C_GREEN = "#2a78d6", "#eb6834", "#1baf7a"
C_REF, C_REF2, C_GRID = "#555555", "#9a9a9a", "#e3e3e3"


# ============================================================================ 데이터
def load_index():
    return {r["video_id"]: r for r in csv.DictReader(open(INDEX))}


def load_extraction(d: Path):
    """done 을 먼저 읽고 배열을 나중에 읽는다 (done 표시는 flush 뒤) → done 행은 완전하다."""
    meta = json.load(open(d / "meta.json"))
    done = np.load(d / "done.npy").astype(bool)
    A = {k: np.load(d / f"{k}.npy") for k in ("l1", "pz", "pstep", "copy", "hcopy", "hstep")}
    for k, v in A.items():
        bad = np.isnan(v[done]).any()
        if bad:
            raise SystemExit(f"{d}/{k}.npy: done 행에 NaN")
    return meta, done, A


def harness_S(dirs):
    pv = {}
    for dd in dirs:
        f = Path(dd) / "per_block.json"
        if not f.exists():
            return None
        pv.update(json.load(open(f))["per_video_surprise"])
    return pv


def hits_of(Spos, Simp):
    return (Spos < Simp).astype(np.float64) + 0.5 * (Spos == Simp)


class Pairs:
    def __init__(self, idx):
        P = TES.load_pairs()
        self.P = P
        self.n = len(P)
        self.block = np.array([int(p["block"]) for p in P])
        self.pos = [p["pos"] for p in P]
        self.imp = [p["imp"] for p in P]
        self.tim = np.array([TES.timing_of(p["cond"]) for p in P])
        self.mot = np.array([TES.motion_of(p["cond"]) for p in P])
        self.viol = np.array([p["viol"] for p in P])
        self.role = np.array([p["role"] for p in P])
        self.k = np.array([p["k"] for p in P])
        vk = np.where(self.viol == "vanish", np.where(self.role == "imp_vanish", "van_o2e", "van_e2o"), self.viol)
        self.vk = vk
        # 드러남 슬롯 (late 만): 샘플 프레임 = raw/3, 미래 슬롯 j 는 샘플 16+2j, 17+2j → 보이는 프레임이 처음 들어오는 슬롯
        rs = np.full(self.n, -1)
        for i, p in enumerate(P):
            r = idx[p["pos"]]
            if r["occ_timing"] == "late" and r["reveal_frame"]:
                rs[i] = (int(r["reveal_frame"]) // 3 - 16) // 2
        self.rs = rs

    def cells(self):
        """칸 이름 → mask. 이름 규칙: t=<timing>|m=<motion>|v=<vkind 또는 viol>|k=..."""
        T, M, V, VK, K = self.tim, self.mot, self.viol, self.vk, self.k
        one = np.ones(self.n, bool)
        C = {"all": one}
        for t in TIMINGS:
            C[f"t={t}"] = T == t
        for m in MOTIONS:
            C[f"m={m}"] = M == m
        for v in ("vanish", "shape", "color"):
            C[f"v={v}"] = V == v
        for vk in VKINDS[:2]:
            C[f"v={vk}"] = VK == vk
        C["v=noncolor"] = V != "color"          # 적대 검증 2026-09-25: 색을 뺀 전체
        for t in TIMINGS:
            for vk in VKINDS + ("vanish",):
                C[f"t={t}|v={vk}"] = (T == t) & ((VK == vk) if vk != "vanish" else (V == "vanish"))
            C[f"t={t}|m=moving"] = (T == t) & (M != "static")
            for vk in VKINDS[:2]:
                C[f"t={t}|m=moving|v={vk}"] = (T == t) & (M != "static") & (VK == vk)
            for m in MOTIONS:
                C[f"t={t}|m={m}"] = (T == t) & (M == m)
                for vk in VKINDS:
                    C[f"t={t}|m={m}|v={vk}"] = (T == t) & (M == m) & (VK == vk)
                C[f"t={t}|m={m}|v=vanish"] = (T == t) & (M == m) & (V == "vanish")
        for kk in range(1, 5):
            C[f"occ|k={kk}"] = (T != "visible") & (K == kk)
            C[f"t=late|k={kk}"] = (T == "late") & (K == kk)
        L2 = (T == "late") & (K >= 2)
        C["t=late|k>=2"] = L2
        for m in MOTIONS:
            C[f"t=late|k>=2|m={m}"] = L2 & (M == m)
        for vk in VKINDS:
            C[f"t=late|k>=2|v={vk}"] = L2 & (VK == vk)
        C["t=late|k>=2|m=moving|v=van_o2e"] = L2 & (M != "static") & (VK == "van_o2e")
        C["t=late|k>=2|m=moving|v=van_e2o"] = L2 & (M != "static") & (VK == "van_e2o")
        return C


FINE = [f"t={t}|m={m}|v={vk}" for t in TIMINGS for m in MOTIONS for vk in VKINDS]
TM = [f"t={t}|m={m}" for t in TIMINGS for m in MOTIONS]


# ============================================================================ bootstrap
class Boot:
    """block bootstrap. 칸마다 block resample 하나 (W: B × n_block 개수 행렬) 를 모든 열이 공유한다."""

    def __init__(self, blocks, B, seed=0):
        self.blocks, self.B, self.seed = blocks, B, seed

    def W(self, name, nb):
        rng = np.random.default_rng([self.seed, zlib.crc32(name.encode())])
        idx = rng.integers(0, nb, size=(self.B, nb))
        return np.bincount((idx + np.arange(self.B)[:, None] * nb).ravel(), minlength=self.B * nb).reshape(self.B, nb).astype(np.float64)

    def sums(self, name, mask, X):
        """→ (n_row, n_block, point sums (ncol,), point count, boot sums (B,ncol), boot counts (B,))"""
        sel = np.flatnonzero(mask)
        if len(sel) == 0:
            return None
        ub, inv = np.unique(self.blocks[sel], return_inverse=True)
        order = np.argsort(inv, kind="stable")
        starts = np.r_[0, np.flatnonzero(np.diff(inv[order])) + 1]
        Xs = X[sel][order]
        S = np.add.reduceat(Xs, starts, axis=0)
        c = np.bincount(inv).astype(np.float64)
        if self.B > 0:
            W = self.W(name, len(ub))
            return len(sel), len(ub), S.sum(0), c.sum(), W @ S, W @ c
        return len(sel), len(ub), S.sum(0), c.sum(), None, None


def ci(v):
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v is not None else (None, None)


class Table:
    """hits 행렬 (n_pair, ncol) + 열 이름. cell(name) → {'n','nb','acc'{col:(p,lo,hi)}, 'diff'{(a,b):(p,lo,hi)}}"""

    def __init__(self, X, cols, avail, boot, diffs):
        self.X, self.cols, self.avail, self.boot = X, list(cols), avail, boot
        self.ci_ = {c: i for i, c in enumerate(self.cols)}
        self.diffs = diffs          # [(a, b)] 열 이름 짝 — a − b
        self.cache = {}

    def cell(self, name, mask):
        if name in self.cache:
            return self.cache[name]
        r = self.boot.sums(name, mask & self.avail, np.nan_to_num(self.X))
        if r is None:
            self.cache[name] = None
            return None
        n, nb, S, c, WS, Wc = r
        pt = S / c * 100
        M = WS / Wc[:, None] * 100 if WS is not None else None
        acc = {}
        lo = np.percentile(M, 2.5, axis=0) if M is not None else [None] * len(self.cols)
        hi = np.percentile(M, 97.5, axis=0) if M is not None else [None] * len(self.cols)
        for i, col in enumerate(self.cols):
            acc[col] = (float(pt[i]), None if lo[i] is None else float(lo[i]), None if hi[i] is None else float(hi[i]))
        diff = {}
        for a, b in self.diffs:
            ia, ib = self.ci_[a], self.ci_[b]
            d = M[:, ia] - M[:, ib] if M is not None else None
            l, h = ci(d)
            diff[(a, b)] = (float(pt[ia] - pt[ib]), l, h)
        out = {"n": n, "nb": nb, "acc": acc, "diff": diff}
        self.cache[name] = out
        return out


# ============================================================================ 분석
def phase_scores(A, rs_row):
    """A (n, K, 8) → (n, K, 3) 단계 평균: hidden = j < rs, reveal = j == rs, after = j > rs (rs<0 → NaN)."""
    n, K, _ = A.shape
    out = np.full((n, K, 3), np.nan)
    for r in np.unique(rs_row):
        if r < 0:
            continue
        m = rs_row == r
        if r > 0:
            out[m, :, 0] = A[m][:, :, :r].mean(-1)
        out[m, :, 1] = A[m][:, :, r]
        if r < N_SLOT - 1:
            out[m, :, 2] = A[m][:, :, r + 1:].mean(-1)
    return out


def build_curves_table(pairs, meta, done, A, boot, idx):
    tags = list(meta["pred_tags"])
    vid = meta["video_id"]
    row = {v: i for i, v in enumerate(vid)}
    ipos = np.array([row[v] for v in pairs.pos])
    iimp = np.array([row[v] for v in pairs.imp])
    avail = done[ipos] & done[iimp]
    L = np.concatenate([A["l1"], A["copy"][:, None, :]], 1).astype(np.float64)       # (n, K+1, 8)
    names = tags + ["copy"]
    rs_row = np.full(len(vid), -1)
    for v, i in row.items():
        r = idx[v]
        if r["occ_timing"] == "late" and r["reveal_frame"]:
            rs_row[i] = (int(r["reveal_frame"]) // 3 - 16) // 2
    PH = phase_scores(L, rs_row)
    fams = {"S": L.mean(-1)}
    for j in range(N_SLOT):
        fams[f"j{j}"] = L[:, :, j]
    for q, ph in enumerate(PHASES):
        fams[ph] = PH[:, :, q]
    cols, Xs = [], []
    Hj = {}
    for f, S in fams.items():
        with np.errstate(invalid="ignore"):
            h = hits_of(S[ipos], S[iimp])
        h[np.isnan(S[ipos]) | np.isnan(S[iimp])] = np.nan
        Xs.append(h)
        cols += [f"{f}:{t}" for t in names]
        if f.startswith("j"):
            Hj[int(f[1:])] = h
    # 슬롯 수를 맞춘 단계 비교 (적대 검증 2026-09-25): 단계 안 슬롯별 hit 의 평균. S 평균은 슬롯이 많은 단계를 저절로 올린다
    rsp = pairs.rs
    for q, ph in enumerate(PHASES):
        sm = np.zeros_like(Hj[0]); cnt = np.zeros(len(rsp))
        for j in range(N_SLOT):
            inph = (rsp >= 0) & ((j < rsp) if q == 0 else ((j == rsp) if q == 1 else (j > rsp)))
            sm[inph] += Hj[j][inph]; cnt[inph] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            hph = sm / cnt[:, None]
        hph[cnt == 0] = np.nan
        Xs.append(hph)
        cols += [f"{ph}H:{t}" for t in names]
    X = np.concatenate(Xs, 1)
    diffs = []
    for f in list(fams) + [f"{ph}H" for ph in PHASES]:
        for t in names:
            if t != "release":
                diffs.append((f"{f}:{t}", f"{f}:release"))
            if t != "copy":
                diffs.append((f"{f}:{t}", f"{f}:copy"))
        for t in names:
            if t.startswith("ariel_ep") and t != "ariel_ep5":
                diffs.append((f"{f}:{t}", f"{f}:ariel_ep5"))
    for _, pts in RUNS.values():            # 이웃 checkpoint 사이 (적대 검증 2026-09-25: ip1 은 진동한다)
        for (_, ta), (_, tb) in zip(pts[:-1], pts[1:]):
            if ta in names and tb in names and (f"S:{tb}", f"S:{ta}") not in diffs:
                diffs.append((f"S:{tb}", f"S:{ta}"))
    return Table(X, cols, avail, boot, diffs), tags, (ipos, iimp, avail)


# ---------------------------------------------------------------- 검증
def validate(pairs, meta, done, A, har, ipos, iimp, avail):
    tags = meta["pred_tags"]
    vid = meta["video_id"]
    res = {}
    for t in FINAL:
        k = tags.index(t)
        S_ext = A["l1"][:, k].astype(np.float64).mean(1)
        pv = har.get(HKEY[t])
        if pv is None:
            res[t] = None
            continue
        S_har_rows = np.array([pv.get(v, np.nan) for v in vid])
        dm = done & ~np.isnan(S_har_rows)
        dS = np.abs(S_ext[dm] - S_har_rows[dm])
        h_ext = hits_of(S_ext[ipos], S_ext[iimp])
        Sh_pos = np.array([pv[v] for v in pairs.pos]); Sh_imp = np.array([pv[v] for v in pairs.imp])
        h_har = hits_of(Sh_pos, Sh_imp)
        a = avail
        flips = np.flatnonzero(a & (h_ext != h_har))
        margin = np.abs(Sh_pos - Sh_imp)
        res[t] = {"n_pair": int(a.sum()), "n_pair_total": pairs.n,
                  "acc_ext": float(h_ext[a].mean() * 100) if a.any() else None,
                  "acc_har_same_pairs": float(h_har[a].mean() * 100) if a.any() else None,
                  "acc_har_all": float(h_har.mean() * 100), "ref": HARNESS_REF[t],
                  "flips": int(len(flips)), "flip_max_margin": float(margin[flips].max()) if len(flips) else 0.0,
                  "median_margin": float(np.median(margin[a])) if a.any() else None,
                  "max_abs_dS": float(dS.max()) if len(dS) else None, "mean_abs_dS": float(dS.mean()) if len(dS) else None,
                  "ties_ext": int((S_ext[ipos][a] == S_ext[iimp][a]).sum()),
                  "flips_by_timing": {tt: int((pairs.tim[flips] == tt).sum()) for tt in TIMINGS},
                  "diff_same_pairs": float((h_ext[a].mean() - h_har[a].mean()) * 100) if a.any() else None}
    # 문맥 동일성: 한 쌍의 두 clip 은 문맥이 픽셀 단위로 같다 → pz (p 와 zT 모두 문맥만의 함수) 가 같아야 한다
    pzd = np.abs(A["pz"][ipos] - A["pz"][iimp])[avail]
    res["_context_identity"] = {"max_abs_dpz": float(pzd.max()) if pzd.size else None,
                                "p99_abs_dpz": float(np.percentile(pzd, 99)) if pzd.size else None,
                                "mean_pz": float(A["pz"][done].mean()) if done.any() else None}
    # 내부 정의 검사
    d0 = done
    res["_definitions"] = {
        "pstep0_eq_pz0": float(np.abs(A["pstep"][d0][:, :, 0] - A["pz"][d0][:, :, 0]).max()) if d0.any() else None,
        "hstep0_eq_hcopy0": float(np.abs(A["hstep"][d0][:, 0] - A["hcopy"][d0][:, 0]).max()) if d0.any() else None,
        "triangle_violations(l1 > pz + copy + 1e-4)": int((A["l1"][d0] > A["pz"][d0] + A["copy"][d0][:, None, :] + 1e-4).sum()) if d0.any() else None,
    }
    return res


# ---------------------------------------------------------------- 예측한 변화량 (가능 clip, clip 단위 · block bootstrap 비율)
def change_stats(meta, done, A, pairs_idx, boot, tags_sel):
    tags = meta["pred_tags"]
    plaus = np.array([str(x) == "1" for x in meta["plausible"]])
    tim = np.array([TES.timing_of(c) for c in meta["condition"]])
    mot = np.array([TES.motion_of(c) for c in meta["condition"]])
    blk = np.array([int(b) for b in meta["block_id"]])
    base = done & plaus
    out = {}
    cb = Boot(blk, boot.B, seed=boot.seed + 1)
    for t in TIMINGS:
        for m in MOTIONS:
            name = f"t={t}|m={m}"
            mk = base & (tim == t) & (mot == m)
            if not mk.any():
                out[name] = None
                continue
            e = {"n_clip": int(mk.sum()), "hcopy_j": A["hcopy"][mk].mean(0).tolist(), "copy_j": A["copy"][mk].mean(0).tolist(),
                 "hstep_j": A["hstep"][mk].mean(0).tolist(), "per": {}}
            # 열: hcopy · copy · hstep(1..7) · 태그마다 pz · l1 · pstep(1..7) · 1[l1<copy] (슬롯 평균)
            cols = [A["hcopy"].mean(1), A["copy"].mean(1), A["hstep"][:, 1:].mean(1)]
            for t_ in tags_sel:
                k = tags.index(t_)
                cols += [A["pz"][:, k].mean(1), A["l1"][:, k].mean(1), A["pstep"][:, k, 1:].mean(1),
                         (A["l1"][:, k] < A["copy"]).mean(1)]
            X = np.stack(cols, 1).astype(np.float64)
            r = cb.sums("chg:" + name, mk, X)
            n, nb, S, c, WS, Wc = r
            for q, t_ in enumerate(tags_sel):
                k = tags.index(t_)
                o = 3 + 4 * q
                R = S[o] / S[0]; Rp = S[o] / S[1]; lr = S[o + 1] / S[1]; st = S[o + 2] / S[2]; cl = S[o + 3] / c
                d = {"R": float(R), "Rcopy": float(Rp), "l1_over_copy": float(lr), "step_ratio": float(st),
                     "p_closer_than_copy": float(cl * 100),
                     "pz_j": A["pz"][mk][:, k].mean(0).tolist(), "pstep_j": A["pstep"][mk][:, k].mean(0).tolist(),
                     "l1_j": A["l1"][mk][:, k].mean(0).tolist()}
                if WS is not None:
                    d["R_ci"] = ci(WS[:, o] / WS[:, 0]); d["Rcopy_ci"] = ci(WS[:, o] / WS[:, 1])
                    d["l1_over_copy_ci"] = ci(WS[:, o + 1] / WS[:, 1]); d["step_ratio_ci"] = ci(WS[:, o + 2] / WS[:, 2])
                    if t_ != "release":
                        k0 = 3
                        d["dR_vs_release_ci"] = ci(WS[:, o] / WS[:, 0] - WS[:, k0] / WS[:, 0])
                        d["dR_vs_release"] = float(R - S[k0] / S[0])
                e["per"][t_] = d
            out[name] = e
    return out


# ---------------------------------------------------------------- 학습 곡선 요약
def curve_summary(T, C):
    res = {}
    for run, (_, pts) in RUNS.items():
        first = pts[0][1]
        for cname in FINE + ["all"] + [f"t={t}" for t in TIMINGS] + TM + ["t=late|m=moving|v=van_o2e", "t=late|m=moving|v=van_e2o"]:
            cell = T.cell(cname, C[cname])
            if cell is None:
                continue
            a = [cell["acc"][f"S:{tg}"][0] for _, tg in pts]
            eps = [e for e, _ in pts]
            last = pts[-1][1]
            key = (f"S:{last}", f"S:{first}") if first != "release" else (f"S:{last}", "S:release")
            if first == "release":
                d = cell["diff"][(f"S:{last}", "S:release")]
            else:
                d = cell["diff"].get((f"S:{last}", f"S:{first}"))
            G = a[-1] - a[0]
            sat = None
            if abs(G) >= 1.0:
                for e, v in zip(eps, a):
                    if (v - a[0]) / G >= 0.9:
                        sat = e
                        break
            steps = np.diff(a)
            sd = [cell["diff"].get((f"S:{tb}", f"S:{ta}")) for (_, ta), (_, tb) in zip(pts[:-1], pts[1:])]
            # max_abs_step 도 첫 걸음을 뺀 이웃 차이의 최대
            sgp = [d for d in sd[1:] if d is not None and d[1] is not None and d[1] > 0 and abs(d[0]) >= 1.0]   # 첫 걸음 (release→첫 checkpoint) 은 빼고, 1 pt 이상
            sgn = [d for d in sd[1:] if d is not None and d[2] is not None and d[2] < 0 and abs(d[0]) >= 1.0]
            res[(run, cname)] = {"step_d": sd, "osc": bool(sgp and sgn),
                                 "max_abs_step": float(max((abs(d[0]) for d in sd[1:] if d is not None), default=0.0)),"eps": eps, "acc": a, "gain": G, "d_last_first": d, "sat": sat,
                                 "mono_down": bool(len(steps) and (steps <= 0).all() and d[2] is not None and d[2] < 0),
                                 "mono_up": bool(len(steps) and (steps >= 0).all() and d[1] is not None and d[1] > 0),
                                 "peak_ep": eps[int(np.argmax(a))], "peak": float(max(a)), "n": cell["n"]}
    return res


# ============================================================================ 그림
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": "#888888",
                         "axes.linewidth": 0.6, "xtick.color": "#555555", "ytick.color": "#555555",
                         "axes.labelcolor": "#333333", "axes.titlesize": 8.5, "axes.titleweight": "normal",
                         "legend.frameon": False, "legend.fontsize": 7.5, "savefig.dpi": 170})
    return plt


def plabel(ax, i, dy=-26):
    lab = "(" + "abcdefghijklmnopqrstuvwxyz"[i] + ")" if i < 26 else f"({i})"
    ax.annotate(lab, xy=(0.5, 0), xycoords="axes fraction", xytext=(0, dy), textcoords="offset points",
                ha="center", va="top", fontsize=9)


def div_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("bgo", [C_ORANGE, "#f3c7b1", "#e6e6e6", "#b6cff0", C_BLUE])


def style(ax, grid_y=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid_y:
        ax.grid(axis="y", color=C_GRID, lw=0.6)
        ax.set_axisbelow(True)


def fig_copy_delta(T, C, gb_rows, path):
    """p − copy (pt) — 칸 = 운동 × 위반방향, 패널 = 가림 타이밍. 굵은 글씨 = CI 가 0 을 안 포함."""
    plt = setup_mpl()
    rows = [(t, t) for t in FINAL] + gb_rows
    cols = [(m, vk) for m in MOTIONS for vk in VKINDS]
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 0.42 * len(rows) + 2.2), sharey=True)
    cm = div_cmap()
    vmax = 40
    for pi, (ax, t) in enumerate(zip(axes, TIMINGS)):
        Z = np.full((len(rows), len(cols)), np.nan); S = np.zeros_like(Z, bool)
        for ri, (lab, spec) in enumerate(rows):
            for ci_, (m, vk) in enumerate(cols):
                name = f"t={t}|m={m}|v={vk}"
                if isinstance(spec, str):
                    cell = T.cell(name, C[name])
                    if cell is None:
                        continue
                    d = cell["diff"][(f"S:{spec}", "S:copy")]
                else:  # 그룹 B: (Table, own col, copy col)
                    Tb, a, b = spec
                    cell = Tb.cell(name, C[name])
                    if cell is None:
                        continue
                    d = cell["diff"][(a, b)]
                Z[ri, ci_] = d[0]
                S[ri, ci_] = d[1] is not None and (d[1] > 0 or d[2] < 0)
        ax.imshow(np.clip(Z, -vmax, vmax), cmap=cm, vmin=-vmax, vmax=vmax, aspect="auto")
        for ri in range(len(rows)):
            for ci_ in range(len(cols)):
                if np.isfinite(Z[ri, ci_]):
                    ax.text(ci_, ri, f"{Z[ri, ci_]:+.0f}", ha="center", va="center", fontsize=6.3,
                            color="#111111" if S[ri, ci_] else "#8a8a8a", fontweight="bold" if S[ri, ci_] else "normal")
                else:
                    ax.text(ci_, ri, "·", ha="center", va="center", fontsize=7, color="#aaaaaa")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([VK_EN[vk] for _, vk in cols], rotation=60, ha="right", fontsize=6.5)
        for j, m in enumerate(MOTIONS):
            ax.annotate(m, xy=(4 * j + 1.5, -0.62), xycoords="data", ha="center", va="bottom", fontsize=7.5, color="#333333",
                        annotation_clip=False)
        for x in (3.5, 7.5):
            ax.axvline(x, color="white", lw=2)
        if len(gb_rows):
            ax.axhline(len(FINAL) - 0.5, color="white", lw=2.5)
        ax.set_title(f"{t}", pad=20)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        plabel(ax, pi, dy=-62)
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.30, top=0.86, wspace=0.06)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(-vmax, vmax))
    cb = fig.colorbar(sm, cax=fig.add_axes([0.935, 0.34, 0.007, 0.48]))
    cb.set_label("acc(p) − acc(copy), pt", fontsize=7.5)
    cb.outline.set_visible(False)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


LC_CELLS = [("all", "all pairs"), ("t=late|m=moving|v=van_o2e", "late, moving, vanish obj→empty"),
            ("t=late|m=moving|v=van_e2o", "late, moving, vanish empty→obj"),
            ("t=visible|m=ramp|v=shape", "visible, ramp, shape"), ("t=visible|m=static|v=color", "visible, static, color"),
            ("t=early", "early occlusion (all)"), ("t=mid", "mid occlusion (all)"), ("t=late", "late occlusion (all)")]


def fig_curves(T, C, path):
    plt = setup_mpl()
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.6))
    # x 배치: run 마다 구간, 구간 안에서 epoch 순서 (등간격)
    xs, ticks, tlabs, spans = {}, [], [], []
    x0 = 0
    for run, (nm, pts) in RUNS.items():
        loc = []
        for i, (e, tg) in enumerate(pts):
            xs[(run, i)] = x0 + i
            ticks.append(x0 + i); tlabs.append(str(e)); loc.append(x0 + i)
        spans.append((run, nm, loc[0], loc[-1]))
        x0 += len(pts) + 1
    for ai, (ax, (cname, lab)) in enumerate(zip(axes.ravel(), LC_CELLS)):
        cell = T.cell(cname, C[cname])
        style(ax)
        if cell is None:
            ax.text(0.5, 0.5, "n = 0", transform=ax.transAxes, ha="center", color="#888888")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(lab); plabel(ax, ai, dy=-30); continue
        rel = cell["acc"]["S:release"][0]; cp = cell["acc"]["S:copy"][0]
        ax.axhline(rel, color=C_REF, lw=1.0, ls="-", zorder=1)
        ax.axhline(cp, color=C_REF2, lw=1.0, ls=":", zorder=1)
        ax.axhline(50, color="#cccccc", lw=0.8, ls="--", zorder=0)
        for run, (nm, pts) in RUNS.items():
            col = C_ORANGE if run == "ariel" else C_BLUE
            x = [xs[(run, i)] for i in range(len(pts))]
            y = [cell["acc"][f"S:{tg}"][0] for _, tg in pts]
            lo = [cell["acc"][f"S:{tg}"][1] for _, tg in pts]; hi = [cell["acc"][f"S:{tg}"][2] for _, tg in pts]
            if lo[0] is not None:
                ax.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0)
            ax.plot(x, y, color=col, lw=1.6, marker="o", ms=3.8, zorder=3)
        ax.set_xticks(ticks); ax.set_xticklabels(tlabs, fontsize=6, rotation=90)
        for run, nm, a, b in spans:
            ax.annotate({"v11": "v11", "pv1": "PV1", "ip1": "IP1", "ip2": "IP2", "ariel": "Ariel"}[run],
                        xy=((a + b) / 2, 0), xycoords=("data", "axes fraction"), xytext=(0, -20),
                        textcoords="offset points", ha="center", va="top", fontsize=7, color="#333333")
        ax.set_title(f"{lab}  (n={cell['n']})")
        ax.set_ylabel("matched-pair acc (%)" if ai % 4 == 0 else "")
        ax.set_xlim(-0.6, x0 - 1.4)
        plabel(ax, ai, dy=-36)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=C_BLUE, marker="o", lw=1.6, ms=4, label="post-FT (epoch; e0 = release)"),
          Line2D([], [], color=C_ORANGE, marker="o", lw=1.6, ms=4, label="Ariel scratch (epoch)"),
          Line2D([], [], color=C_REF, lw=1.0, label="release"),
          Line2D([], [], color=C_REF2, lw=1.0, ls=":", label="copy baseline"),
          Line2D([], [], color="#cccccc", lw=0.8, ls="--", label="chance")]
    fig.legend(handles=hs, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.12, hspace=0.62, wspace=0.18)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_per_slot(T, C, path):
    plt = setup_mpl()
    panels = ["copy"] + FINAL
    fig, axes = plt.subplots(2, 7, figsize=(16, 5.6))
    x = np.arange(N_SLOT)
    for pi, tg in enumerate(panels):
        ax = axes[0, pi]; style(ax)
        ax.axhline(50, color="#cccccc", lw=0.8, ls="--")
        for t, col in (("visible", C_BLUE), ("late", C_ORANGE)):
            cell = T.cell(f"t={t}", C[f"t={t}"])
            if cell is None:
                continue
            if tg not in ("copy", "release"):
                ax.plot(x, [cell["acc"][f"j{j}:release"][0] for j in x], color=C_REF2, lw=0.9,
                        ls="-" if t == "visible" else "--", zorder=1)
            y = [cell["acc"][f"j{j}:{tg}"][0] for j in x]
            ax.plot(x, y, color=col, lw=1.6, marker="o", ms=3, zorder=3)
        ax.set_ylim(35, 101); ax.set_xticks(x)
        ax.set_title(tg); ax.set_xlabel("future slot j")
        if pi == 0:
            ax.set_ylabel("per-slot acc (%)")
        plabel(ax, pi, dy=-28)
    # 아래 줄: Δ vs release (슬롯별), visible / late, CI 띠
    for pi, tg in enumerate(["copy"] + TRAINED):
        ax = axes[1, pi]; style(ax)
        ax.axhline(0, color="#999999", lw=0.8)
        for t, col in (("visible", C_BLUE), ("late", C_ORANGE)):
            cell = T.cell(f"t={t}", C[f"t={t}"])
            if cell is None:
                continue
            d = [cell["diff"][(f"j{j}:{tg}", f"j{j}:release")] for j in x]
            y = [v[0] for v in d]
            if d[0][1] is not None:
                ax.fill_between(x, [v[1] for v in d], [v[2] for v in d], color=col, alpha=0.18, lw=0)
            ax.plot(x, y, color=col, lw=1.6, marker="o", ms=3)
        ax.set_xticks(x); ax.set_title(f"{tg} − release"); ax.set_xlabel("future slot j")
        if pi == 0:
            ax.set_ylabel("Δ per-slot acc (pt)")
        plabel(ax, 7 + pi, dy=-28)
    axes[1, 6].axis("off")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=C_BLUE, lw=1.6, marker="o", ms=3, label="visible"),
          Line2D([], [], color=C_ORANGE, lw=1.6, marker="o", ms=3, label="late occlusion"),
          Line2D([], [], color=C_REF2, lw=0.9, label="release (visible)"),
          Line2D([], [], color=C_REF2, lw=0.9, ls="--", label="release (late)")]
    axes[1, 6].legend(handles=hs, loc="center")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.93, bottom=0.1, hspace=0.75, wspace=0.3)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_change(chg, path):
    plt = setup_mpl()
    fig, axes = plt.subplots(2, 6, figsize=(16, 5.6), sharey="row")
    x = np.arange(N_SLOT)
    for pi, tg in enumerate(FINAL):
        for row, key in enumerate(("R", "step")):
            ax = axes[row, pi]; style(ax)
            ax.axhline(1.0, color="#999999", lw=0.8)
            for m, col in zip(MOTIONS, (C_BLUE, C_ORANGE, C_GREEN)):
                for t, ls in (("visible", "-"), ("late", "--")):
                    e = chg.get(f"t={t}|m={m}")
                    if not e or tg not in e["per"]:
                        continue
                    d = e["per"][tg]
                    if key == "R":
                        y = np.array(d["pz_j"]) / np.array(e["hcopy_j"]); xx = x
                    else:
                        y = np.array(d["pstep_j"][1:]) / np.array(e["hstep_j"][1:]); xx = x[1:]
                    ax.plot(xx, y, color=col, ls=ls, lw=1.4, marker="o", ms=2.5)
            ax.set_xticks(x)
            ax.set_xlabel("future slot j")
            if row == 0:
                ax.set_title(tg)
            if pi == 0:
                ax.set_ylabel("R_j = pz_j / hcopy_j" if key == "R" else "pstep_j / hstep_j")
            plabel(ax, row * 6 + pi, dy=-28)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=c, lw=1.4, label=m) for m, c in zip(MOTIONS, (C_BLUE, C_ORANGE, C_GREEN))]
    hs += [Line2D([], [], color="#555555", lw=1.2, ls="-", label="visible"), Line2D([], [], color="#555555", lw=1.2, ls="--", label="late occlusion")]
    fig.legend(handles=hs, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.01))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.1, hspace=0.7, wspace=0.12)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_delta_heatmap(T, C, path):
    """Δ vs release, 모든 체크포인트 × 48 칸. 점 = CI 가 0 을 안 포함."""
    plt = setup_mpl()
    rows = []
    for run, (nm, pts) in RUNS.items():
        rows += [tg for _, tg in pts if tg != "release"]
    cols = [(m, vk) for m in MOTIONS for vk in VKINDS]
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 0.3 * len(rows) + 2.4), sharey=True)
    cm = div_cmap(); vmax = 40
    for pi, (ax, t) in enumerate(zip(axes, TIMINGS)):
        Z = np.full((len(rows), len(cols)), np.nan); S = np.zeros_like(Z, bool)
        for ri, tg in enumerate(rows):
            for ci_, (m, vk) in enumerate(cols):
                name = f"t={t}|m={m}|v={vk}"
                cell = T.cell(name, C[name])
                if cell is None:
                    continue
                d = cell["diff"][(f"S:{tg}", "S:release")]
                Z[ri, ci_] = d[0]; S[ri, ci_] = d[1] is not None and (d[1] > 0 or d[2] < 0)
        ax.imshow(np.clip(Z, -vmax, vmax), cmap=cm, vmin=-vmax, vmax=vmax, aspect="auto")
        yy, xx = np.nonzero(S)
        ax.scatter(xx, yy, s=5, color="#222222", lw=0)
        yy, xx = np.nonzero(~np.isfinite(Z))
        ax.scatter(xx, yy, s=6, marker="x", color="#aaaaaa", lw=0.6)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([VK_EN[vk] for _, vk in cols], rotation=60, ha="right", fontsize=6.5)
        for j, m in enumerate(MOTIONS):
            ax.annotate(m, xy=(4 * j + 1.5, -0.62), xycoords="data", ha="center", va="bottom", fontsize=7.5, annotation_clip=False)
        for x in (3.5, 7.5):
            ax.axvline(x, color="white", lw=2)
        b = 0
        for run, (nm, pts) in RUNS.items():
            b += len([1 for _, tg in pts if tg != "release"])
            if b < len(rows):
                ax.axhline(b - 0.5, color="white", lw=2)
        ax.set_title(t, pad=20)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=7)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        plabel(ax, pi, dy=-62)
    fig.subplots_adjust(left=0.07, right=0.92, bottom=0.26, top=0.88, wspace=0.06)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(-vmax, vmax))
    cb = fig.colorbar(sm, cax=fig.add_axes([0.935, 0.32, 0.007, 0.5]))
    cb.set_label("acc − acc(release), pt (dot: 95% CI excludes 0)", fontsize=7.5)
    cb.outline.set_visible(False)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ============================================================================ md
def f1(v, d=1):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


def fci(t, d=1, sign=False):
    if t is None:
        return "—"
    p, lo, hi = t
    s = "+" if sign else ""
    if lo is None:
        return f"{p:{s}.{d}f}"
    return f"{p:{s}.{d}f} [{lo:{s}.{d}f}, {hi:{s}.{d}f}]"


def sig(t):
    return t is not None and t[1] is not None and (t[1] > 0 or t[2] < 0)


def cname_ko(c):
    parts = dict(p.split("=") for p in c.split("|") if "=" in p)
    s = []
    if "t" in parts:
        s.append(parts["t"])
    if "m" in parts:
        s.append(parts["m"])
    if "v" in parts:
        s.append(VK_KO.get(parts["v"], parts["v"]))
    if "k" in parts:
        s.append(f"k={parts['k']}")
    return " · ".join(s) if s else c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", default=str(CURVES))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    idx = load_index()
    pairs = Pairs(idx)
    C = pairs.cells()
    boot = Boot(pairs.block, a.boot, a.seed)

    meta, done, A = load_extraction(Path(a.curves))
    assert meta["video_id"] == list(idx.keys()), "meta video_id 순서가 index_test.csv 와 다르다"
    T, tags, (ipos, iimp, avail) = build_curves_table(pairs, meta, done, A, boot, idx)
    n_done, n_rows = int(done.sum()), len(done)
    print(f"[curves] done {n_done}/{n_rows} clip, 온전한 쌍 {int(avail.sum())}/{pairs.n}")

    # ---- 하네스 (final 6 · 그룹 B) per_video_surprise
    har = {}
    for key, name, grp, dirs, ip1m, desc in TES.MODELS:
        pv = harness_S(dirs)
        if pv is not None and all(v in pv for v in pairs.pos + pairs.imp):
            har[key] = pv
    val = validate(pairs, meta, done, A, har, ipos, iimp, avail)
    for t in FINAL:
        v = val[t]
        if v:
            print(f"  [val] {t:11s} ext {f1(v['acc_ext'], 2)}  harness(same pairs) {f1(v['acc_har_same_pairs'], 2)}  "
                  f"harness(all) {v['acc_har_all']:.2f}  flips {v['flips']}  max|dS| {v['max_abs_dS']:.2e}")

    # 하네스 표 (10,752 쌍 전부): final 6 슬롯 평균
    Xh = np.stack([hits_of(np.array([har[HKEY[t]][v] for v in pairs.pos]), np.array([har[HKEY[t]][v] for v in pairs.imp]))
                   for t in FINAL], 1)
    TH = Table(Xh, [f"S:{t}" for t in FINAL], np.ones(pairs.n, bool), boot,
               [(f"S:{t}", "S:release") for t in TRAINED])

    # ---- 그룹 B (own 추출이 있으면 own p vs own copy; 없으면 하네스 점수만)
    gb = {}
    for m in GROUPB:
        d = CACHE / f"v11score_{m}_own"
        if not (d / "done.npy").exists():
            continue
        mb, db, Ab = load_extraction(d)
        if not db.any():
            continue
        rowb = {v: i for i, v in enumerate(mb["video_id"])}
        ip = np.array([rowb[v] for v in pairs.pos]); ii = np.array([rowb[v] for v in pairs.imp])
        av = db[ip] & db[ii]
        So = Ab["l1"][:, 0].astype(np.float64).mean(1); Sc = Ab["copy"].astype(np.float64).mean(1)
        X = np.stack([hits_of(So[ip], So[ii]), hits_of(Sc[ip], Sc[ii])], 1)
        Tb = Table(X, ["S:own", "S:copy"], av, Boot(pairs.block, a.boot, a.seed), [("S:own", "S:copy")])
        hk = GB_HKEY.get(m, m)
        chk = None
        if hk in har:
            Sh = np.array([har[hk].get(v, np.nan) for v in mb["video_id"]])
            dm = db & ~np.isnan(Sh)
            chk = float(np.abs(So[dm] - Sh[dm]).max()) if dm.any() else None
        gb[m] = {"T": Tb, "n_done": int(db.sum()), "n_pair": int(av.sum()), "max_abs_dS_vs_harness": chk,
                 "copy_mean": float(Ab["copy"][db].mean()), "own_mean": float(Ab["l1"][db].mean())}
        print(f"[groupB] {m}: done {int(db.sum())}, 쌍 {int(av.sum())}, max|dS| vs harness {chk}")
    gb_har = {}
    for m in GROUPB:
        hk = GB_HKEY.get(m, m)
        if hk in har:
            gb_har[m] = hits_of(np.array([har[hk][v] for v in pairs.pos]), np.array([har[hk][v] for v in pairs.imp]))

    # ---- 분석
    LS = curve_summary(T, C)
    chg = change_stats(meta, done, A, (ipos, iimp), boot, FINAL + [t for t in tags if t not in FINAL])
    h6f = json.load(open(H6F)) if H6F.exists() else None

    # ---- 그림
    gb_rows = [(f"{m} (own)", (gb[m]["T"], "S:own", "S:copy")) for m in GROUPB if m in gb]
    figs = {"copy": out / "fig_copy_delta.png", "curves": out / "fig_learning_curves.png",
            "slot": out / "fig_per_slot.png", "change": out / "fig_predicted_change.png",
            "heat": out / "fig_delta_vs_release.png"}
    fig_copy_delta(T, C, gb_rows, figs["copy"])
    fig_curves(T, C, figs["curves"])
    fig_per_slot(T, C, figs["slot"])
    fig_change(chg, figs["change"])
    fig_delta_heatmap(T, C, figs["heat"])

    # ---- json
    def cell_json(Tab, name):
        c = Tab.cell(name, C[name])
        if c is None:
            return None
        return {"n": c["n"], "n_block": c["nb"], "acc": {k: [round(x, 3) if x is not None else None for x in v] for k, v in c["acc"].items()},
                "diff": {f"{a_}-{b_}": [round(x, 3) if x is not None else None for x in v] for (a_, b_), v in c["diff"].items()}}
    J = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "curves_dir": str(a.curves), "n_done": n_done, "n_rows": n_rows,
         "n_pair_avail": int(avail.sum()), "boot": a.boot, "validation": val,
         "cells_curves": {n: cell_json(T, n) for n in C if T.cell(n, C[n]) is not None},
         "cells_harness_final": {n: cell_json(TH, n) for n in C},
         "curves_summary": {f"{r}|{c}": {k: (list(v) if isinstance(v, tuple) else v) for k, v in e.items()} for (r, c), e in LS.items()},
         "predicted_change": chg,
         "groupB": {m: {k: v for k, v in e.items() if k != "T"} | {"cells": {n: cell_json(e["T"], n) for n in ["all"] + [f"t={t}" for t in TIMINGS] + TM}}
                    for m, e in gb.items()}}
    json.dump(J, open(out / "analysis.json", "w"), ensure_ascii=False)

    write_md(out, a, pairs, C, T, TH, tags, val, LS, chg, gb, gb_har, h6f, n_done, n_rows, avail, figs, meta)
    print(f"-> {out / 'ANALYSIS.md'}  ({time.time() - t0:.0f}s)")


# ============================================================================ md 쓰기
def write_md(out, a, pairs, C, T, TH, tags, val, LS, chg, gb, gb_har, h6f, n_done, n_rows, avail, figs, meta):
    L = []
    P = L.append
    full = n_done == n_rows
    cov_t = defaultdict(int)
    for t in TIMINGS:
        for m in MOTIONS:
            cov_t[(t, m)] = int((C[f"t={t}|m={m}"] & avail).sum())
    covered = [f"{t}·{m}" for t in TIMINGS for m in MOTIONS if cov_t[(t, m)] == int(C[f"t={t}|m={m}"].sum())]
    partial = [f"{t}·{m} ({cov_t[(t, m)]}/{int(C[f't={t}|m={m}'].sum())})" for t in TIMINGS for m in MOTIONS
               if 0 < cov_t[(t, m)] < int(C[f"t={t}|m={m}"].sum())]
    missing = [f"{t}·{m}" for t in TIMINGS for m in MOTIONS if cov_t[(t, m)] == 0]

    def acc(Tab, cname, col):
        c = Tab.cell(cname, C[cname])
        return None if c is None else c["acc"][col]

    def dif(Tab, cname, a_, b_):
        c = Tab.cell(cname, C[cname])
        return None if c is None else c["diff"][(a_, b_)]

    P("# v11 점수 심층 분석 — 복사 기준선 · 학습 곡선 · 슬롯별 · 예측한 변화량 (자동 생성)")
    P("")
    P(f"> `te_analyze_scores.py` 가 배열에서 다시 계산해 쓴다 ({time.strftime('%Y-%m-%d %H:%M')}). 손으로 고치지 않는다. "
      f"상위: [`../README.md`](../README.md) · 모델: [`../MODELS.md`](../MODELS.md) · 점수표: [`SCORES.md`](SCORES.md).")
    P(f"> 데이터: `v11_split_test` (학습 안 한 block 절반) · 추출 `{Path(a.curves).name}` done **{n_done:,} / {n_rows:,} clip** → "
      f"온전한 쌍 **{int(avail.sum()):,} / {pairs.n:,}**" + ("" if full else " — ⚠️ **부분 데이터** (추출 진행 중이었다)") + ".")
    if not full:
        P(f"> 온전히 덮인 칸: {', '.join(covered) or '없음'} · 부분: {', '.join(partial) or '없음'} · **없음: {', '.join(missing) or '없음'}**. "
          f"없는 칸은 표에 '—'. 추출 폴더 순서가 조건별이라 부분 데이터는 조건이 치우친다 (무작위 표본이 아니다).")
    P(f"> CI = 95 % block bootstrap {a.boot} 회 (block = 두 미래를 공유하는 2 쌍 — 쌍 A 는 문맥 a, 쌍 B 는 문맥 b). 차이는 같은 쌍 위의 짝지은 차이. chance 50.")
    P("")

    # ------------------------------------------------ 한 줄
    P("## 한 줄")
    P("")
    head = []
    corr = {}                       # 정정 블록에 쓸 수치 (적대 검증 2026-09-25)

    def vb(d):
        return "CI>0" if (d[1] is not None and d[1] > 0) else ("CI<0" if (d[2] is not None and d[2] < 0) else "0 포함")
    o_rel = acc(T, "all", "S:release"); o_cp = acc(T, "all", "S:copy")
    if o_rel and o_cp:
        d = dif(T, "all", "S:release", "S:copy")
        corr["rc"] = d
        parts = {vk: dif(T, f"v={vk}", "S:release", "S:copy") for vk in ("color", "shape", "van_o2e", "van_e2o")}
        dnc = dif(T, "v=noncolor", "S:release", "S:copy"); corr["rc_nc"] = dnc; corr["rc_parts"] = parts
        dtm = {t: dif(T, f"t={t}", "S:release", "S:copy") for t in TIMINGS}; corr["rc_t"] = dtm
        d_ip1 = dif(T, "all", "S:ip1_e40", "S:copy"); d_ip1nc = dif(T, "v=noncolor", "S:ip1_e40", "S:copy")
        corr["ip1c"] = (d_ip1, d_ip1nc)
        head.append(f"**release − copy = {fci(d, 1, True)} pt 전체로는 높지만 ({vb(d)}; release {f1(o_rel[0])} vs copy {f1(o_cp[0])}, "
                    f"{T.cell('all', C['all'])['n']:,} 쌍, 추출 값) 이 우위는 색 위반에서만 나온다** — color {fci(parts['color'], 1, True)} ({vb(parts['color'])}) · "
                    f"shape {fci(parts['shape'], 1, True)} ({vb(parts['shape'])}) · vanish 물체→빈 {fci(parts['van_o2e'], 1, True)} ({vb(parts['van_o2e'])}) · "
                    f"vanish 빈→물체 {fci(parts['van_e2o'], 1, True)} ({vb(parts['van_e2o'])}). **색을 뺀 전체 {fci(dnc, 1, True)} ({vb(dnc)}).** "
                    f"타이밍별: " + " · ".join(f"{t} {fci(dtm[t], 1, True)}" for t in TIMINGS) + ". "
                    f"같은 구성 효과로 ip1_e40 − copy 도 전체 {fci(d_ip1, 1, True)} 인데 색을 빼면 {fci(d_ip1nc, 1, True)} ({vb(d_ip1nc)}).")
    # 복사를 못 이기던 칸에서 학습 뒤 이기는가
    fine_av = [c for c in FINE if T.cell(c, C[c]) is not None]
    ceil_ = [c for c in fine_av if acc(T, c, "S:copy")[0] >= 99 and acc(T, c, "S:release")[0] >= 99]
    rel_not = [c for c in fine_av if c not in ceil_ and not (dif(T, c, "S:release", "S:copy")[1] is not None and dif(T, c, "S:release", "S:copy")[1] > 0)]
    rel_below = [c for c in rel_not if dif(T, c, "S:release", "S:copy")[2] is not None and dif(T, c, "S:release", "S:copy")[2] < 0]
    if fine_av:
        s = []
        nb_ = []
        for t in TRAINED:
            now = [c for c in rel_not if (lambda d: d[1] is not None and d[1] > 0)(dif(T, c, f"S:{t}", "S:copy"))]
            s.append(f"{t} {len(now)}")
            # 새로 copy 를 밑도는 칸은 48 칸 전부에서 센다 (천장 칸 포함 — 적대 검증 2026-09-25)
            newb = [c for c in fine_av if c not in rel_below and (lambda d: d[2] is not None and d[2] < 0)(dif(T, c, f"S:{t}", "S:copy"))]
            nb_.append((t, newb))
        corr["newb"] = nb_
        head.append(f"세분 칸 {len(fine_av)} 개 (둘 다 ≥ 99 인 천장 {len(ceil_)} 칸 제외) 중 **release 가 copy 를 CI 로 못 이기는 칸 {len(rel_not)} 개** "
                    f"(그중 copy 보다 CI 로 낮은 칸 {len(rel_below)}) → 학습 뒤 그 칸에서 copy 를 CI 로 이기는 수: " + " · ".join(s)
                    + " (CI 하한이 정확히 +0.0 인 칸이 있어 bootstrap seed 에 따라 ±1). "
                    + "학습 뒤 **새로** copy 보다 CI 로 낮아진 칸 (48 칸 전부에서, 천장 칸 포함): "
                    + " · ".join(f"{t} {len(nb)}" + (f" ({', '.join(cname_ko(c) for c in nb)})" if nb else "") for t, nb in nb_) + ".")
    dn_of = {}
    for t in TRAINED:
        d = dif(TH, "all", f"S:{t}", "S:release")
        up = [c for c in FINE if sig(dif(TH, c, f"S:{t}", "S:release")) and dif(TH, c, f"S:{t}", "S:release")[0] > 0]
        dn = [c for c in FINE if sig(dif(TH, c, f"S:{t}", "S:release")) and dif(TH, c, f"S:{t}", "S:release")[0] < 0]
        dn_of[t] = (d, dn)
        head.append(f"`{t}` (도메인 {DOMAIN[t]}): 전체 Δ {fci(d, 1, True)} (하네스) · 48 칸 중 오름 {len(up)} / 내림 {len(dn)}.")
    sat = []
    osc = []
    for run, (nm, pts) in RUNS.items():
        e = LS.get((run, "all"))
        if e:
            sat.append(f"{run} {e['acc'][0]:.1f}→{e['acc'][-1]:.1f}" + (f" (90 % 도달 e{e['sat']})" if e["sat"] is not None else "")
                       + ((", checkpoint 사이 진동 (" + " → ".join(f"e{ep} {v:.1f}" for ep, v in zip(e['eps'], e['acc'])) + ")") if e["osc"] else
                          (f", e{e['peak_ep']} 최고 뒤 하락" if e["peak_ep"] != e["eps"][-1] and e["peak"] - e["acc"][-1] > 1.0 else "")))
            if run in ("ip1", "ip2", "ariel"):
                osc.append((run, e["max_abs_step"], [d for d in e["step_d"]]))
    if sat:
        ce = T.cell("all", C["all"])
        d5 = ce["diff"].get(("S:ariel_ep5", "S:release")) if ce else None
        corr["osc"] = osc; corr["ariel5"] = d5
        head.append("학습 곡선 (전체, 추출 값): " + " · ".join(sat) + ". "
                    + (f"ariel 의 시작점 ep5 는 release 가 아니다 (ep5 − release {fci(d5, 1, True)}). " if d5 else "")
                    + "도메인 밖 checkpoint 사이 최대 흔들림 (첫 checkpoint 이후 이웃 차이): " + " · ".join(f"{r} {m:.1f} pt" for r, m, _ in osc)
                    + " — 가장 큰 흔들림 (" + max(osc, key=lambda x: x[1])[0] + ") 은 도메인 밖 이득 (+3 pt 안팎) 과 같은 크기이고, 한 seed 라 block CI 는 이 학습 잡음을 담지 않는다 → ip1 / ip2 / ariel 끼리 순위를 매기지 않는다.")
    c2 = T.cell("t=late|k>=2", C["t=late|k>=2"])
    if c2:
        bigH, bigS = [], []
        for t in TRAINED:
            dH = [c2["diff"][(f"{ph}H:{t}", f"{ph}H:release")][0] for ph in PHASES]
            dS = [c2["diff"][(f"{ph}:{t}", f"{ph}:release")][0] for ph in PHASES]
            bigH.append((t, PHASES[int(np.argmax(dH))], dH)); bigS.append((t, PHASES[int(np.argmax(dS))], dS))
        corr["phase"] = (bigS, bigH)
        clear = [t for t, ph, dH in bigH if ph == "after" and dH[2] - sorted(dH)[-2] > 2.0
                 and (lambda d: d[1] is not None and d[1] > 0)(c2["diff"][(f"afterH:{t}", "afterH:release")])]
        corr["after_clear"] = clear
        head.append("드러남 단계 (late, k ≥ 2): 슬롯 수를 맞추면 (단계 안 슬롯별 hit 평균) hidden / reveal / after 의 Δ vs release 는 "
                    + " · ".join(f"`{t}` {dH[0]:+.1f} / {dH[1]:+.1f} / {dH[2]:+.1f}" for t, ph, dH in bigH)
                    + " — 'after 에 이득이 몰린다' (after 가 최대 · CI>0 · 다음 단계보다 2 pt 넘게) 는 "
                    + (", ".join(f"`{t}`" for t in clear) or "어느 predictor 에서도 안") + " 에서만 남는다.")
    gbf = {m: e for m, e in gb.items() if e["n_pair"] == pairs.n}
    if gbf:
        wins = [m for m, e in gbf.items() if (lambda d: d[1] is not None and d[1] > 0)(e["T"].cell("all", C["all"])["diff"][("S:own", "S:copy")])]
        loses = [m for m, e in gbf.items() if (lambda d: d[2] is not None and d[2] < 0)(e["T"].cell("all", C["all"])["diff"][("S:own", "S:copy")])]
        part = [f"{m} ({e['n_pair']:,} 쌍)" for m, e in gb.items() if m not in gbf]
        wnote = []
        for m in wins:
            dv = gbf[m]["T"].cell("t=visible", C["t=visible"])["diff"][("S:own", "S:copy")]
            wnote.append(f"{m}" + (" — 학습 중간 checkpoint" if m in GB_MIDTRAIN else "") + f", visible 에서는 own copy 대비 {fci(dv, 1, True)} ({vb(dv)})")
        corr["gbw"] = wnote
        head.append(f"그룹 B (추출이 온전한 {len(gbf)} 모델): 자기 encoder 의 copy 를 CI 로 이기는 own predictor {len(wins)}/{len(gbf)}"
                    + (f" ({'; '.join(wnote)})" if wins else "") + f", 밑도는 것 {len(loses)}" + (f" ({', '.join(loses)})" if loses else "")
                    + (f". 부분 추출이라 뺀 모델: {', '.join(part)}" if part else "") + ".")
    if "rc" in corr and dn_of:
        pr = corr["rc_parts"]
        ood = ("ip1_e40", "ip2_e80", "ariel_ep43")
        ip1dn = dn_of["ip1_e40"][1]
        ip1mv = [c for c in ip1dn if "|m=static|" not in c and c.endswith("v=van_o2e")]
        nbd = dict(corr.get("newb", []))
        osc_mx = max((m for _, m, _ in corr.get("osc", [])), default=0.0)
        head.insert(0, "**요약.** `v11_split_test` (학습 안 한 block 절반, 10,752 matched pair) 에서 릴리즈 ViT-H predictor 는 예측 없는 복사 기준선 "
                    f"(문맥 마지막 튜블릿 복사) 보다 전체로 {fci(corr['rc'], 1, True)} pt 높다. 다만 이 우위는 색 위반 ({fci(pr['color'], 1, True)}) 에서만 나오고, "
                    f"모양 ({fci(pr['shape'], 1, True)}) 과 물체→빈 사라짐 ({fci(pr['van_o2e'], 1, True)}) 에서는 복사보다 "
                    + ("CI 로 낮으며" if (pr['shape'][2] is not None and pr['shape'][2] < 0 and pr['van_o2e'][2] < 0) else "낮거나 같으며")
                    + f" 색을 빼면 {fci(corr['rc_nc'], 1, True)} 이다. predictor 만 다시 학습하면 v11 점수는 모두 오른다 (하네스, block CI): "
                    + " · ".join(f"`{t}` {dn_of[t][0][0]:+.1f}" for t in TRAINED)
                    + f". 그러나 도메인 밖 셋은 48 칸 중 " + " · ".join(f"{t} {len(dn_of[t][1])}" for t in ood) + " 칸에서 CI 로 내려가고 "
                    f"(IntPhys1 은 {len(ip1dn)} 칸 중 {len(ip1mv)} 칸이 움직이는 물체→빈 사라짐), IntPhys1 은 복사보다 새로 낮아진 칸이 {len(nbd.get('ip1_e40', []))} 개다. "
                    f"도메인 밖 predictor 의 +3 pt 안팎 이득은 한 seed 에서 checkpoint 사이 흔들림 (IntPhys1 이웃 checkpoint 사이 최대 {osc_mx:.1f} pt) 과 같은 크기라 서로 순위를 매길 수 없다. "
                    "`v11_e10` 의 이득은 궤적 재생이라 일반화가 아니다. 드러남 뒤 슬롯에 이득이 몰린다는 결론은 슬롯 수를 맞추면 "
                    + (", ".join(f"`{t}`" for t in corr.get("after_clear", [])) or "어느 predictor 에서도 안") + " 에서만 남는다.")
    head.append(f"§1–4 는 추출 값 (release {acc(T, 'all', 'S:release')[0]:.2f}), §5 와 위 '전체 Δ' 는 하네스 값 (release {acc(TH, 'all', 'S:release')[0]:.2f}) 이다 "
                "— 같은 predictor 도 소수점이 다르다 (§0, fp16 근접 동점).")
    for x in head:
        P(f"- {x}")
    P("")

    # ------------------------------------------------ 표
    P("## 표")
    P("")
    P("### 0. 검증 — 추출 배열이 하네스 점수를 재현하는가")
    P("")
    P("슬롯 평균 l1 (= 표준 surprise) 로 matched pair 정확도. '하네스 (같은 쌍)' 은 하네스 `per_video_surprise` 를 같은 쌍으로 제한한 값, "
      "'하네스 (전체)' 는 10,752 쌍 전부. flip = 두 경로에서 hit 가 다른 쌍 (fp16 batch 구성 잡음으로 근접 동점이 뒤집힌 것).")
    P("")
    P("| predictor | 추출 acc | 하네스 (같은 쌍) | Δ (추출 − 하네스) | 순 flip (쌍) | 하네스 (전체) | 기준값 | 쌍 | flip | flip 쌍의 최대 margin (하네스) | 쌍 margin 중앙값 | max\\|ΔS\\| | mean\\|ΔS\\| |")
    P("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for t in FINAL:
        v = val[t]
        if not v:
            P(f"| {t} | — | 하네스 없음 | | | | | | | | |"); continue
        P(f"| {t} | {f1(v['acc_ext'], 2)} | {f1(v['acc_har_same_pairs'], 2)} | {v['diff_same_pairs']:+.2f} | {v['diff_same_pairs'] * v['n_pair'] / 100:+.0f} | "
          f"{v['acc_har_all']:.2f} | {v['ref']:.2f} | {v['n_pair']:,} | "
          f"{v['flips']} | {v['flip_max_margin']:.1e} | {f1(v['median_margin'], 4) if v['median_margin'] is not None else '—'} | "
          f"{v['max_abs_dS']:.1e} | {v['mean_abs_dS']:.1e} |")
    ci_ = val["_context_identity"]; de = val["_definitions"]
    P("")
    vv = [(t, val[t]) for t in FINAL if val.get(t) and val[t]["acc_ext"] is not None]
    over = [(t, v) for t, v in vv if abs(v["diff_same_pairs"]) > 0.1]
    if vv:
        P(f"- **~0.1 pt 기준:** {len(vv) - len(over)}/{len(vv)} 통과. "
          + ("넘은 것: " + ", ".join(f"{t} {v['diff_same_pairs']:+.2f} pt (flip {v['flips']}, 순 {v['diff_same_pairs'] * v['n_pair'] / 100:+.0f} 쌍)" for t, v in over)
             + ". 차이는 전부 flip 에서 온다 — 뒤집힌 쌍은 모두 하네스 margin ≤ "
             + f"{max(v['flip_max_margin'] for _, v in vv):.1e} 이고 clip 별 \\|ΔS\\| 는 최대 {max(v['max_abs_dS'] for _, v in vv):.1e} 다 "
             "(te_v11_score.py 검증: 같은 clip 을 batch 만 바꿔도 슬롯 평균이 최대 ~5e-4 움직인다). 순 flip 의 부호는 predictor 마다 섞여 있다."
             if over else "모두 0.1 pt 안."))
    P(f"- 문맥 동일성: 한 쌍의 두 clip 은 문맥이 같으므로 pz (p 와 zT 모두 문맥만의 함수) 가 같아야 한다 — max\\|Δpz\\| = {ci_['max_abs_dpz']:.1e} "
      f"(99 분위 {ci_['p99_abs_dpz']:.1e}, 평균 pz {ci_['mean_pz']:.3f})."
      + (" 비트 단위로 같다 — 쌍의 p 가 같다는 CLAUDE.md §1-2 의 전제가 추출에서도 선다." if ci_['max_abs_dpz'] == 0 else
         " 0 이 아닌 것은 두 clip 이 다른 batch 에서 fp16 로 돌았기 때문이다."))
    P(f"- 정의 검사: max\\|pstep[:,:,0] − pz[:,:,0]\\| = {de['pstep0_eq_pz0']:.1e} · max\\|hstep[:,0] − hcopy[:,0]\\| = {de['hstep0_eq_hcopy0']:.1e} · "
      f"삼각부등식 (l1 ≤ pz + copy) 위반 {de['triangle_violations(l1 > pz + copy + 1e-4)']} 칸.")
    if h6f:
        e = h6f.get("all|ALL", {})
        P(f"- 외부 대조 (다른 표본): `auto_research` h6f 는 v11_full 의 block 1/4 표본 ({e.get('n')} 쌍, 두 절반 섞임) 에서 release {e.get('p_full')} / "
          f"copy {e.get('copyz_full')} 를 냈다 (`auto_research/exp_results/h6/h6f_v11.json`). 같은 정의 · 다른 쌍이라 방향만 대조한다.")
    P("")

    # ---- 1. copy
    P("### 1. 복사 기준선 — p 가 '예측 없이 마지막 관측을 복사' 보다 나은가")
    P("")
    P("S_copy = mean_j \\|LN(z)[문맥 마지막 튜블릿] − h[8+j]\\|. 한 쌍의 두 clip 은 문맥이 같아 zT 가 같다 → copy 는 "
      "\"두 미래 중 어느 쪽이 마지막 관측과 더 가까운가\" 로 채점한다. 그룹 A 여섯 predictor 는 encoder 가 같아 copy 를 공유한다.")
    P("")
    P(f"![p − copy]({figs['copy'].name})")
    P("")
    P("*그림 1. acc(p) − acc(copy) (pt), 칸 = 운동 × 위반 (vanish 는 방향별), 패널 = 가림 타이밍. 굵은 글씨 = 95 % CI 가 0 을 포함하지 않음, 회색 = 포함, '·' = 데이터 없음.*")
    P("")
    P("**1-a. copy 정확도 (타이밍 × 운동 × 위반)**")
    P("")
    P("| 타이밍 · 운동 | " + " | ".join(VK_KO[v] for v in VKINDS) + " | 합침 | 쌍 |")
    P("|---|" + "---:|" * (len(VKINDS) + 2))
    for t in TIMINGS:
        for m in MOTIONS:
            cells = [fci(acc(T, f"t={t}|m={m}|v={vk}", "S:copy")) for vk in VKINDS]
            c = T.cell(f"t={t}|m={m}", C[f"t={t}|m={m}"])
            P(f"| {t} · {m} | " + " | ".join(cells) + f" | {fci(acc(T, f't={t}|m={m}', 'S:copy'))} | {c['n'] if c else 0} |")
    P(f"| **전체** | " + " | ".join(fci(acc(T, f"v={vk}", "S:copy")) for vk in VKINDS[:2]) + " | "
      + " | ".join(fci(acc(T, f"v={v}", "S:copy")) for v in ("shape", "color")) + f" | {fci(acc(T, 'all', 'S:copy'))} | {T.cell('all', C['all'])['n'] if T.cell('all', C['all']) else 0} |")
    P("")
    P("**1-b. p − copy (pt, 같은 쌍) — 타이밍 × 위반 (세 운동 합침)**")
    P("")
    P("| 칸 | copy | " + " | ".join(FINAL) + " |")
    P("|---|---:|" + "---:|" * len(FINAL))
    for t in TIMINGS:
        for vk in VKINDS:
            cn = f"t={t}|v={vk}"
            P(f"| {t} · {VK_KO[vk]} | {f1(acc(T, cn, 'S:copy')[0]) if acc(T, cn, 'S:copy') else '—'} | "
              + " | ".join(("**" if sig(dif(T, cn, f'S:{tg}', 'S:copy')) else "") + fci(dif(T, cn, f"S:{tg}", "S:copy"), 1, True)
                           + ("**" if sig(dif(T, cn, f'S:{tg}', 'S:copy')) else "") for tg in FINAL) + " |")
    cn = "all"
    P(f"| **전체** | {f1(acc(T, cn, 'S:copy')[0]) if acc(T, cn, 'S:copy') else '—'} | "
      + " | ".join(fci(dif(T, cn, f"S:{tg}", "S:copy"), 1, True) for tg in FINAL) + " |")
    P("")
    P("굵게 = CI 가 0 을 포함하지 않음. 세분 (타이밍 × 운동 × 위반방향) 48 칸 전부는 `analysis.json` 의 `cells_curves` 와 그림 1.")
    P("")
    P("**1-c. release 가 copy 를 CI 로 못 이기는 세분 칸 — 학습 뒤에는?** (값 = p − copy, 굵게 = CI > 0)")
    P("")
    if rel_not:
        P("| 칸 | copy acc | " + " | ".join(FINAL) + " |")
        P("|---|---:|" + "---:|" * len(FINAL))
        for c in rel_not:
            row = []
            for tg in FINAL:
                d = dif(T, c, f"S:{tg}", "S:copy")
                b = d[1] is not None and d[1] > 0
                row.append(("**" if b else "") + fci(d, 1, True) + ("**" if b else ""))
            P(f"| {cname_ko(c)}{' ▼' if c in rel_below else ''} | {f1(acc(T, c, 'S:copy')[0])} | " + " | ".join(row) + " |")
        P("")
        P(f"▼ = release 가 copy 보다 CI 로 **낮은** 칸 ({len(rel_below)} 개). 천장 칸 (copy · release 모두 ≥ 99) {len(ceil_)} 개는 뺐다.")
    else:
        P("(해당 칸 없음)")
    P("")
    # 그룹 B
    P("**1-d. 그룹 B — 각 encoder 자신의 p vs 자신의 copy**")
    P("")
    if gb:
        P("| 모델 | 추출 done | 쌍 | own p acc | own copy acc | p − copy | visible | early | mid | late | max\\|ΔS\\| vs 하네스 |")
        P("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for m, e in gb.items():
            Tb = e["T"]
            c = Tb.cell("all", C["all"])
            if c is None:
                continue
            per_t = [fci(Tb.cell(f"t={t}", C[f"t={t}"])["diff"][("S:own", "S:copy")], 1, True) if Tb.cell(f"t={t}", C[f"t={t}"]) else "—" for t in TIMINGS]
            P(f"| {m}{' ⚠️부분' if e['n_pair'] < pairs.n else ''} | {e['n_done']:,} | {e['n_pair']:,} | {fci(c['acc']['S:own'])} | {fci(c['acc']['S:copy'])} | "
              f"{fci(c['diff'][('S:own', 'S:copy')], 1, True)} | " + " | ".join(per_t) + f" | {f1(e['max_abs_dS_vs_harness'], 6) if e['max_abs_dS_vs_harness'] is not None else '—'} |")
        P("")
        P("visible … late 열 = 그 타이밍의 own p − own copy. max\\|ΔS\\| = 추출 슬롯 평균 vs 하네스 per_video_surprise (clip 별, 그룹 B 채점 재현 검사). "
          "⚠️부분 = 이 실행 때 추출이 진행 중이었다 — 추출이 조건 순서로 쌓이므로 그 모델의 '전체' 는 덮인 조건만의 값이고 한 줄 요약에서 뺐다. "
          "그룹 B 모델끼리 · 그룹 A 와 copy 절대값은 비교하지 않는다 (encoder 가 다르다). 중간 체크포인트 (`*_e225`, `vitb_k400_e100`) 는 MODELS.md 참고.")
        missing_gb = [m for m in GROUPB if m not in gb]
        if missing_gb:
            P("")
            P(f"own 추출이 없던 그룹 B 모델: {', '.join(missing_gb)} (하네스 슬롯 평균 점수는 SCORES.md).")
    else:
        P("그룹 B 의 `v11score_<모델>_own` 추출이 이 실행 시점에 없었다 → own copy 기준선을 낼 수 없다. "
          "하네스 슬롯 평균 점수만 있는 모델 (copy 없음, 참고):")
        P("")
        P("| 모델 | own p acc (하네스, 10,752 쌍) |")
        P("|---|---:|")
        for m, h in gb_har.items():
            P(f"| {m} | {h.mean() * 100:.2f} |")
    P("")

    # ---- 2. 학습 곡선
    P("### 2. 학습 곡선 — epoch 별 정확도 (e0 = release, Ariel 은 e0 없음)")
    P("")
    P(f"![learning curves]({figs['curves'].name})")
    P("")
    P("*그림 2. 칸마다 checkpoint 별 matched-pair 정확도 (띠 = 95 % CI). 파랑 = post-FT (각 구간 첫 점 = release), 주황 = Ariel scratch, "
      "진회색 실선 = release, 회색 점선 = copy, 연회색 파선 = chance.*")
    P("")
    P("**2-a. 곡선 값** (행 = checkpoint, 열 = 칸)")
    P("")
    P("| run | epoch | " + " | ".join(lab for _, lab in LC_CELLS) + " |")
    P("|---|---:|" + "---:|" * len(LC_CELLS))
    for run, (nm, pts) in RUNS.items():
        for e, tg in pts:
            vals = []
            for cn, _ in LC_CELLS:
                v = acc(T, cn, f"S:{tg}")
                vals.append(f1(v[0]) if v else "—")
            P(f"| {nm} | {e} | " + " | ".join(vals) + " |")
    P(f"| copy | — | " + " | ".join(f1(acc(T, cn, 'S:copy')[0]) if acc(T, cn, 'S:copy') else "—" for cn, _ in LC_CELLS) + " |")
    P("")
    P("**2-b. 포화 epoch 와 끝점 차이** — 포화 = 처음으로 (끝 − 시작) 의 90 % 에 닿은 epoch (\\|끝 − 시작\\| < 1 pt 면 '—'). "
      "끝점 차이 = 마지막 − 첫 checkpoint (post-FT: − release · Ariel: − ep5).")
    P("")
    P("| run | " + " | ".join(lab for _, lab in LC_CELLS) + " |")
    P("|---|" + "---|" * len(LC_CELLS))
    for run, (nm, pts) in RUNS.items():
        row = []
        for cn, _ in LC_CELLS:
            e = LS.get((run, cn))
            if not e:
                row.append("—"); continue
            row.append(f"{fci(e['d_last_first'], 1, True)} · 포화 {e['sat'] if e['sat'] is not None else '—'} · 최고 e{e['peak_ep']}")
        P(f"| {nm} | " + " | ".join(row) + " |")
    P("")
    P("**2-c. 단조 악화 칸** — 48 세분 칸 중 모든 epoch 걸음이 ≤ 0 이고 끝점 차이 CI 가 0 아래인 칸")
    P("")
    any_ = False
    for run, (nm, pts) in RUNS.items():
        dn = [(c, LS[(run, c)]) for c in FINE if (run, c) in LS and LS[(run, c)]["mono_down"]]
        up = [(c, LS[(run, c)]) for c in FINE if (run, c) in LS and LS[(run, c)]["mono_up"]]
        nn = len([c for c in FINE if (run, c) in LS])
        P(f"- **{nm}** ({nn} 칸 중): 단조 악화 {len(dn)} — "
          + (", ".join(f"{cname_ko(c)} {fci(e['d_last_first'], 1, True)}" for c, e in dn) if dn else "없음")
          + f" · 단조 향상 {len(up)}")
        any_ = any_ or bool(dn)
    P("")

    # ---- 3. 슬롯별
    P("### 3. 슬롯별 — 미래의 어디서 좋아지나")
    P("")
    P(f"![per slot]({figs['slot'].name})")
    P("")
    P("*그림 3. 위: 슬롯 j 의 l1 만으로 채점한 정확도 (파랑 = visible, 주황 = late 가림; 학습한 predictor 칸의 회색 = release). "
      "아래: 슬롯별 Δ vs release (띠 = 95 % CI).*")
    P("")
    for t in ("visible", "late"):
        P(f"**3-{'a' if t == 'visible' else 'b'}. {t} — 슬롯별 정확도 (Δ vs release, 굵게 = CI 가 0 을 안 포함)**")
        P("")
        cell = T.cell(f"t={t}", C[f"t={t}"])
        if cell is None:
            P("(데이터 없음)"); P(""); continue
        P(f"n = {cell['n']:,} 쌍")
        P("")
        P("| predictor | " + " | ".join(f"j{j}" for j in range(N_SLOT)) + " | 슬롯 평균 S |")
        P("|---|" + "---:|" * (N_SLOT + 1))
        for tg in ["copy"] + FINAL:
            row = []
            for j in range(N_SLOT):
                v = cell["acc"][f"j{j}:{tg}"][0]
                if tg in ("copy", "release"):
                    row.append(f1(v))
                else:
                    d = cell["diff"][(f"j{j}:{tg}", f"j{j}:release")]
                    b = sig(d)
                    row.append(f"{v:.1f} ({'**' if b else ''}{d[0]:+.1f}{'**' if b else ''})")
            P(f"| {tg} | " + " | ".join(row) + f" | {f1(cell['acc'][f'S:{tg}'][0])} |")
        P("")
    P("**3-c. late 가림 (k ≥ 2) — 드러남 기준 단계.** 드러남 슬롯 rs = 처음으로 보이는 프레임이 들어오는 미래 슬롯 "
      "(k = 2, 3 → rs 1 · k = 4 → rs 2; k = 1 은 가려진 슬롯이 없어 뺀다). hidden = j < rs (미래 쪽이 전부 가려짐), reveal = j = rs, after = j > rs.")
    P("")
    cn = "t=late|k>=2"
    cell = T.cell(cn, C[cn])
    if cell:
        P(f"n = {cell['n']:,} 쌍")
        P("")
        P("| predictor | hidden | reveal | after | Δ hidden | Δ reveal | Δ after |")
        P("|---|---:|---:|---:|---:|---:|---:|")
        for tg in ["copy"] + FINAL:
            ac = [f1(cell["acc"][f"{ph}:{tg}"][0]) for ph in PHASES]
            if tg in ("copy", "release"):
                ds = ["—"] * 3
            else:
                ds = [("**" if sig(cell['diff'][(f"{ph}:{tg}", f"{ph}:release")]) else "") + fci(cell["diff"][(f"{ph}:{tg}", f"{ph}:release")], 1, True)
                      + ("**" if sig(cell['diff'][(f"{ph}:{tg}", f"{ph}:release")]) else "") for ph in PHASES]
            P(f"| {tg} | " + " | ".join(ac) + " | " + " | ".join(ds) + " |")
        P("")
        P("같은 단계를 운동 · 위반방향별로 (정확도, release → 학습):")
        P("")
        P("| 칸 | 쌍 | " + " | ".join(f"{ph}: release → " + " / ".join(TRAINED) for ph in PHASES[:1]) + " | reveal | after |")
        P("|---|---:|---|---|---|")
        for sub in [f"t=late|k>=2|m={m}" for m in MOTIONS] + [f"t=late|k>=2|v={vk}" for vk in VKINDS]:
            c2 = T.cell(sub, C[sub])
            if c2 is None:
                continue
            cells = []
            for ph in PHASES:
                cells.append(f"{c2['acc'][f'{ph}:release'][0]:.0f} → " + " / ".join(f"{c2['acc'][f'{ph}:{tg}'][0]:.0f}" for tg in TRAINED))
            P(f"| {cname_ko(sub)} | {c2['n']} | " + " | ".join(cells) + " |")
        P("")
    else:
        P("(late 데이터 없음)"); P("")

    # ---- 4. 예측한 변화량
    P("### 4. '예측한 변화량' — p 는 copy 에서 얼마나 움직이나 (크기, 정답 아님)")
    P("")
    P("가능 clip 만. 슬롯 j 마다 R_j = mean pz_j / mean hcopy_j (p 가 copy 에서 떨어진 거리 / 진짜 미래가 마지막 문맥 튜블릿에서 떨어진 거리), "
      "R'_j = mean pz_j / mean copy_j (**완벽한 p = h 이면 clip 마다 pz = copy 라 R' = 1**), l1/copy = mean \\|p − h\\| / mean \\|zT − h\\| "
      "(< 1 이면 평균적으로 p 가 copy 보다 진짜 미래에 가깝다), 걸음비 = mean pstep_j / mean hstep_j (j = 1…7). "
      "표 값은 슬롯 평균의 비 (Σ_j 분자 / Σ_j 분모), [ ] = block bootstrap 95 % CI.")
    P("")
    P(f"![predicted change]({figs['change'].name})")
    P("")
    P("*그림 4. 위: R_j = pz_j / hcopy_j. 아래: pstep_j / hstep_j (j ≥ 1). 색 = 운동, 실선 = visible, 파선 = late 가림. 1 = 진짜 변화량과 같은 크기.*")
    P("")
    P("**4-a. 진짜 변화량 (target 쪽) 과 copy 거리** — hcopy = \\|h[7] − h[8+j]\\|, copy = \\|zT − h[8+j]\\|, hstep = \\|h[8+j] − h[8+j−1]\\| (슬롯 평균)")
    P("")
    P("| 타이밍 · 운동 | clip | hcopy | copy | copy − hcopy (encoder 쪽 차이) | hstep (j≥1) |")
    P("|---|---:|---:|---:|---:|---:|")
    for t in TIMINGS:
        for m in MOTIONS:
            e = chg.get(f"t={t}|m={m}")
            if not e:
                P(f"| {t} · {m} | 0 | — | — | — | — |"); continue
            hc, cp, hs = np.mean(e["hcopy_j"]), np.mean(e["copy_j"]), np.mean(e["hstep_j"][1:])
            P(f"| {t} · {m} | {e['n_clip']} | {hc:.3f} | {cp:.3f} | {cp - hc:+.3f} | {hs:.3f} |")
    P("")
    for key, title, cik in (("R", "R = Σpz / Σhcopy", "R_ci"), ("Rcopy", "R' = Σpz / Σcopy (완벽한 p → 1)", "Rcopy_ci"),
                            ("l1_over_copy", "l1 / copy (< 1 = p 가 copy 보다 진짜에 가깝다)", "l1_over_copy_ci"),
                            ("step_ratio", "걸음비 Σpstep / Σhstep (j ≥ 1)", "step_ratio_ci"),
                            ("p_closer_than_copy", "p 가 copy 보다 가까운 (clip, 슬롯) 비율 %", None)):
        P(f"**4-{'bcdef'[['R', 'Rcopy', 'l1_over_copy', 'step_ratio', 'p_closer_than_copy'].index(key)]}. {title}**")
        P("")
        P("| 타이밍 · 운동 | " + " | ".join(FINAL) + " |")
        P("|---|" + "---:|" * len(FINAL))
        for t in TIMINGS:
            for m in MOTIONS:
                e = chg.get(f"t={t}|m={m}")
                if not e:
                    P(f"| {t} · {m} | " + " | ".join("—" for _ in FINAL) + " |"); continue
                row = []
                for tg in FINAL:
                    d = e["per"][tg]
                    if cik and cik in d and d[cik][0] is not None:
                        row.append(f"{d[key]:.2f} [{d[cik][0]:.2f}, {d[cik][1]:.2f}]" if key != "p_closer_than_copy" else f"{d[key]:.1f}")
                    else:
                        row.append(f"{d[key]:.2f}" if key != "p_closer_than_copy" else f"{d[key]:.1f}")
                P(f"| {t} · {m} | " + " | ".join(row) + " |")
        P("")
    # R 학습 곡선 (all 칸 평균은 없으니 대표 칸)
    P("**4-g. R 의 학습 곡선** (visible · late, 세 운동 각각; 값 = R)")
    P("")
    P("| run | epoch | " + " | ".join(f"{t}·{m}" for t in ("visible", "late") for m in MOTIONS) + " |")
    P("|---|---:|" + "---:|" * 6)
    for run, (nm, pts) in RUNS.items():
        for ep, tg in pts:
            row = []
            for t in ("visible", "late"):
                for m in MOTIONS:
                    e = chg.get(f"t={t}|m={m}")
                    row.append(f"{e['per'][tg]['R']:.2f}" if e and tg in e["per"] else "—")
            P(f"| {nm} | {ep} | " + " | ".join(row) + " |")
    P("")

    # ---- 5. 무엇이 좋아졌나
    P("### 5. 무엇이 좋아졌나 / 안 좋아졌나 — predictor 마다 (하네스 점수, 10,752 쌍 전부)")
    P("")
    P("final 다섯 predictor 의 Δ vs release 는 하네스 `per_video_surprise` (추출 배열과 같은 값, §0) 로 **전체 쌍**에서 낸다. "
      "세분 48 칸 (타이밍 × 운동 × 위반방향) 중 CI 가 0 을 포함하지 않는 칸만 센다.")
    P("")
    P(f"![delta vs release]({figs['heat'].name})")
    P("")
    P("*그림 5. 모든 checkpoint 의 Δ vs release (pt), 추출 배열 (§0 의 쌍 범위). 점 = 95 % CI 가 0 을 포함하지 않음, × = 데이터 없음.*")
    P("")
    P("| predictor | 학습 도메인 (v11 기준) | 전체 Δ | 오름 / 내림 (48 칸) | 가장 많이 오른 칸 | 내린 칸 |")
    P("|---|---|---|---|---|---|")
    for t in TRAINED:
        d = dif(TH, "all", f"S:{t}", "S:release")
        ups = sorted([(dif(TH, c, f"S:{t}", "S:release"), c) for c in FINE
                      if sig(dif(TH, c, f"S:{t}", "S:release")) and dif(TH, c, f"S:{t}", "S:release")[0] > 0], key=lambda x: -x[0][0])
        dns = sorted([(dif(TH, c, f"S:{t}", "S:release"), c) for c in FINE
                      if sig(dif(TH, c, f"S:{t}", "S:release")) and dif(TH, c, f"S:{t}", "S:release")[0] < 0], key=lambda x: x[0][0])
        P(f"| {t} | {DOMAIN[t]} | {fci(d, 1, True)} | {len(ups)} / {len(dns)} | "
          + "; ".join(f"{cname_ko(c)} {x[0]:+.0f}" for x, c in ups[:4]) + " | "
          + ("; ".join(f"{cname_ko(c)} {x[0]:+.0f}" for x, c in dns) or "없음") + " |")
    P("")
    P("**5-a. 타이밍 × 위반방향 (세 운동 합침) Δ vs release** (하네스, 전체 쌍; 굵게 = CI 가 0 을 안 포함)")
    P("")
    P("| 칸 | release | " + " | ".join(TRAINED) + " |")
    P("|---|---:|" + "---:|" * len(TRAINED))
    for t in TIMINGS:
        for vk in VKINDS:
            cn = f"t={t}|v={vk}"
            row = []
            for tg in TRAINED:
                d = dif(TH, cn, f"S:{tg}", "S:release"); b = sig(d)
                row.append(("**" if b else "") + fci(d, 1, True) + ("**" if b else ""))
            P(f"| {t} · {VK_KO[vk]} | {f1(acc(TH, cn, 'S:release')[0])} | " + " | ".join(row) + " |")
    P("")

    # ------------------------------------------------ 읽기 / 단서 (수치는 결과에서 채운다)
    write_reading(L, T, TH, C, LS, chg, gb, full, avail, pairs, val, n_done, n_rows)
    P("## 재현")
    P("")
    P("```bash")
    P("cd /data/hyuntak/project/2026/2027_cvpr/vjepa2")
    P("PY=/data/hyuntak/anaconda3/envs/vjepa2/bin/python")
    P("# 추출 (GPU, 다른 세션이 돌린다; 재개 가능 — 같은 명령)")
    P("CUDA_VISIBLE_DEVICES=0,1 $PY z_research/scripts/analysis/te_v11_score.py --model vith --preset curves --gpus 2")
    P("# (그룹 B own copy) CUDA_VISIBLE_DEVICES=... $PY z_research/scripts/analysis/te_v11_score.py --model <vitb_*|vittiny_*> --gpus N")
    P("# 이 분석 (CPU, ~수 분)")
    P(f"$PY z_research/scripts/analysis/te_analyze_scores.py --boot {a.boot}")
    P("```")
    P("")
    P(f"입력: `{a.curves}` (`meta.json` · `*.npy` · `done.npy`), 하네스 `per_block.json` (`training_effects_scores.py` 의 `MODELS` 폴더), "
      "`data_csv/intphysgen_v11_split/index_test.csv`. config: `configs/protocols/surprise_c16t32.yaml` + `configs/protocols/models.md` "
      "(추출 쪽 `_resolved.yaml`). 출력: `z_research/TrainingEffects/scores/{ANALYSIS.md, analysis.json, fig_*.png}`.")
    L[1:1] = [""] + correction_block(corr)
    (out / "ANALYSIS.md").write_text("\n".join(L) + "\n")


def correction_block(corr):
    """적대 검증 (2026-09-25, ../Archive/verify_raw/verify_scores.json) 반영 목록 — 이전 문장 → 지금 문장. 수치는 이번 실행 값."""
    Q = []
    Q.append("> ⚠️ **정정 (2026-09-25, 적대 검증 반영)** — 아래가 바뀌었다 (이전 문장 → 지금 문장, 수치는 이번 재실행). "
             "검증 원문: [`../Archive/verify_raw/verify_scores.json`](../Archive/verify_raw/verify_scores.json).")
    if "rc" in corr:
        pr = corr["rc_parts"]
        Q.append(f"> - **한 줄 1.** 이전 'release 는 예측 없는 복사 기준선을 전체로 CI 로 이긴다 (+2.5 [+1.6, +3.3])' → 지금 '전체 {fci(corr['rc'], 1, True)} 는 "
                 f"색 위반에서만 나온다: color {fci(pr['color'], 1, True)} · shape {fci(pr['shape'], 1, True)} · vanish 물체→빈 {fci(pr['van_o2e'], 1, True)} · "
                 f"빈→물체 {fci(pr['van_e2o'], 1, True)}, **색을 뺀 전체 {fci(corr['rc_nc'], 1, True)}**; early {fci(corr['rc_t']['early'], 1, True)} · "
                 f"mid {fci(corr['rc_t']['mid'], 1, True)}'. ip1_e40 − copy 도 전체 {fci(corr['ip1c'][0], 1, True)} 이지만 색을 빼면 {fci(corr['ip1c'][1], 1, True)}.")
    if "newb" in corr:
        Q.append("> - **새로 copy 를 밑도는 칸.** 이전 '천장 17 칸을 뺀 20 칸 안에서만 셈 (pv1_e15 1 · ip1_e40 1)' → 지금 '48 칸 전부에서 셈: "
                 + " · ".join(f"{t} {len(nb)}" for t, nb in corr["newb"]) + "'. 학습 뒤 copy 를 이기는 칸 수는 'bootstrap seed 에 따라 ±1' 로 적는다.")
    if "osc" in corr:
        ip1 = [x for x in corr["osc"] if x[0] == "ip1"]
        st = ""
        if ip1:
            st = " · ".join(fci(d, 1, True) for d in ip1[0][2] if d is not None)
        Q.append(f"> - **학습 곡선.** 이전 'ip1 e5 최고 뒤 하락' → 지금 'ip1 은 checkpoint 사이 진동한다 (이웃 checkpoint 차이 {st} pt)'. "
                 "도메인 밖 이득 (+3 pt 안팎) 이 한 seed 의 흔들림과 같은 크기라 ip1 / ip2 / ariel 순위를 매기지 않는다고 한 줄에 적었다. "
                 + (f"ariel 시작점 ep5 − release = {fci(corr['ariel5'], 1, True)} (release 가 아니다)." if corr.get("ariel5") else ""))
    if "phase" in corr:
        bS, bH = corr["phase"]
        Q.append("> - **드러남 단계.** 이전 'Δ 가 가장 큰 단계: 모두 after' (단계마다 슬롯 수가 달라 after 5–6 · reveal 1 · hidden 1–2 슬롯의 S 평균 — 같은 조건 비교가 아니다) → "
                 "지금 슬롯별 hit 를 단계 안에서 평균 (hidden / reveal / after Δ): "
                 + " · ".join(f"`{t}` {dH[0]:+.1f} / {dH[1]:+.1f} / {dH[2]:+.1f} (최대 {ph})" for t, ph, dH in bH)
                 + ". 옛 S 평균 값도 §읽기에 남겼다.")
    if corr.get("gbw"):
        Q.append("> - **그룹 B.** 이전 '유일하게 copy 를 이기는 own predictor' 에 단서 없음 → 지금 " + "; ".join(corr["gbw"]) + ".")
    Q.append("> - **예측한 변화량.** 이전 '(clip, 슬롯) 의 100.0 % 이상' → '100 % (예외 0)'. 이전 '화면 전체의 토큰 요동이 채운다' → "
             "'변화 크기는 물체 운동에서 오지 않는다 (static ≈ moving); hcopy 의 슬롯 모양이 조건끼리 같아 요동이 아니라 슬롯 번호에 따른 계통 구조다'. "
             "'L1 로 학습한 예측은 … 수축' 해석 문장은 목적함수를 원인으로 거는 문장이라 뺐다 (CLAUDE.md).")
    Q.append("> - **표기.** block 정의 '문맥을 공유하는 2 쌍' → '두 미래를 공유하는 2 쌍 (쌍 A 문맥 a · 쌍 B 문맥 b)'. `pv1_e15` 도메인 '가까움' → "
             "'안 (가림·경사 팔, README 규칙)'. 교차 참조 '단서 2' → '단서: v11 은 궤적 재생'. 한 줄에 하네스 값 / 추출 값 구분을 적었다. "
             "'가장 많이 오른 칸' 은 240 비교 중 귀무에서 ~12 개가 우연히 유의할 수 있다는 단서를 붙였다.")
    Q.append("> - **미적용.** (1) v11 에서 causal target h^c 로 copy 대조 — h^c 추출이 없고 GPU 추출이 필요하다 (IntPhys1 skip2_w32 에서는 release − copy 가 표준 표적 +3.9 → 인과 표적 +6.7 로 움직였다, "
             "[`../ip1_copy/IP1_COPY.md`](../ip1_copy/IP1_COPY.md)) → release − copy 의 크기는 target 정의에 따라 달라질 수 있다. (2) 여러 seed 재학습으로 학습 잡음을 CI 에 넣기 — 재학습 필요. "
             "(3) bootstrap seed 여러 개로 칸 수 재계산 — 하지 않고 '±1' 로만 적었다.")
    return Q


def write_reading(L, T, TH, C, LS, chg, gb, full, avail, pairs, val, n_done, n_rows):
    """읽기 · 단서 — 수치는 전부 결과에서 채운다. 방향이 데이터에 따라 달라지는 낱말은 조건으로 고른다."""
    P = L.append

    def acc(Tab, cn, col):
        c = Tab.cell(cn, C[cn]); return None if c is None else c["acc"][col]

    def dif(Tab, cn, a_, b_):
        c = Tab.cell(cn, C[cn]); return None if c is None else c["diff"][(a_, b_)]

    def pos(d):
        return d is not None and d[1] is not None and d[1] > 0

    def neg(d):
        return d is not None and d[2] is not None and d[2] < 0

    fine_av = [c for c in FINE if T.cell(c, C[c]) is not None]
    ceil_ = [c for c in fine_av if acc(T, c, "S:copy")[0] >= 99 and acc(T, c, "S:release")[0] >= 99]
    rel_not = [c for c in fine_av if c not in ceil_ and not pos(dif(T, c, "S:release", "S:copy"))]
    rel_below = [c for c in rel_not if neg(dif(T, c, "S:release", "S:copy"))]
    npair = int(avail.sum())

    P("## 읽기")
    P("")
    # ---- 검증
    vs = [(t, val[t]) for t in FINAL if val.get(t) and val[t]["acc_ext"] is not None]
    if vs:
        dmax = max(abs(v["diff_same_pairs"]) for _, v in vs)
        nfl = sum(v["flips"] for _, v in vs)
        fmx = max(v["flip_max_margin"] for _, v in vs)
        med = min(v["median_margin"] for _, v in vs)
        byt = {tt: sum(v["flips_by_timing"][tt] for _, v in vs) for tt in TIMINGS}
        n_ok = sum(abs(v["diff_same_pairs"]) <= 0.1 for _, v in vs)
        s = (f"**검증.** 추출 배열의 슬롯 평균 정확도는 같은 쌍 위의 하네스 값과 predictor 여섯 개 모두 {dmax:.2f} pt 안에서 같다 "
             f"({npair:,} 쌍; ~0.1 pt 기준은 {n_ok}/{len(vs)} 통과 — §0). 두 경로에서 hit 가 다른 쌍 (flip) 은 여섯 predictor 합 {nfl} 건 (쌍 × predictor 의 {nfl / (6 * npair) * 100:.2f} %) — "
             f"timing 별 " + " · ".join(f"{tt} {byt[tt]}" for tt in TIMINGS) + f". 뒤집힌 쌍의 하네스 margin \\|S_pos − S_imp\\| 은 최대 {fmx:.1e} 로 "
             f"쌍 margin 중앙값 (가장 작은 predictor {med:.1e}) 의 1/{med / max(fmx, 1e-12):.0f} 이다 — fp16 batch 구성 잡음 (두 경로의 clip 별 \\|ΔS\\| 최대 "
             f"{max(v['max_abs_dS'] for _, v in vs):.1e}) 이 근접 동점을 뒤집은 것이다. late 가림이 flip 의 {byt['late'] / max(nfl, 1) * 100:.0f} % "
             f"(쌍의 {int((pairs.tim[avail] == 'late').sum()) / max(npair, 1) * 100:.0f} %) 를 차지한다.")
        if full:
            dd = max(abs(v["acc_ext"] - v["ref"]) for _, v in vs)
            s += f" 전체 10,752 쌍에서 기준값 (release 75.83 · v11_e10 91.35 · ip1_e40 78.53 · ip2_e80 79.56 · pv1_e15 81.63 · ariel_ep43 79.39) 과의 차이는 최대 {dd:.2f} pt."
        P(f"- {s}")
    # ---- copy
    o = dif(T, "all", "S:release", "S:copy")
    if o:
        P(f"- **복사 기준선 — release.** release − copy 는 전체 {fci(o, 1, True)} pt. 세분 칸 {len(fine_av)} 개 (천장 {len(ceil_)} 칸 제외) 중 "
          f"release 가 copy 를 CI 로 이기는 칸은 {len([c for c in fine_av if c not in ceil_ and pos(dif(T, c, 'S:release', 'S:copy'))])} 개, "
          f"CI 로 밑도는 칸 {len(rel_below)} 개" + (": " + "; ".join(
              f"{cname_ko(c)} (release {acc(T, c, 'S:release')[0]:.0f} vs copy {acc(T, c, 'S:copy')[0]:.0f})" for c in rel_below) if rel_below else "")
          + ". copy 는 한 쌍에서 zT 가 같아 '마지막 관측에 더 가까운 미래' 를 고른다 — release 가 그보다 낮은 칸은 p 가 "
          "그 기본 선택보다 더 틀린 쪽 미래에 가깝게 놓였다는 뜻이다 (어느 쪽이 '물리적으로' 맞는지는 이 점수로 말하지 않는다).")
        lines = []
        for t in TRAINED:
            now = [c for c in rel_not if pos(dif(T, c, f"S:{t}", "S:copy"))]
            still = [c for c in rel_below if neg(dif(T, c, f"S:{t}", "S:copy"))]
            newly_below = [c for c in fine_av if c not in rel_below and neg(dif(T, c, f"S:{t}", "S:copy"))]   # 천장 칸 포함 (적대 검증 2026-09-25)
            lines.append(f"`{t}` {len(now)}/{len(rel_not)} 칸에서 copy 를 CI 로 이긴다"
                         + (f" ({', '.join(cname_ko(c) for c in now)})" if now and len(now) <= 6 else "")
                         + f", release 가 밑돌던 {len(rel_below)} 칸 중 {len(still)} 칸은 여전히 밑돈다"
                         + (f"; 새로 copy 를 밑도는 칸 {len(newly_below)} ({', '.join(cname_ko(c) for c in newly_below)})" if newly_below else ""))
        P("- **학습 뒤 copy 를 이기나** (release 가 copy 를 CI 로 못 이기던 칸 기준; '새로 밑도는 칸' 은 천장 칸 포함 48 칸 전부에서; 칸 수는 bootstrap seed 에 따라 ±1): " + " · ".join(lines) + ".")
    if gb:
        s = []
        for m, e in gb.items():
            c = e["T"].cell("all", C["all"])
            if c:
                s.append(f"{m} own − own copy {fci(c['diff'][('S:own', 'S:copy')], 1, True)} ({e['n_pair']:,} 쌍"
                         + (")" if e["n_pair"] == pairs.n else ", 부분)"))
        gbf = {m: e for m, e in gb.items() if e["n_pair"] == pairs.n}
        win = [m for m, e in gbf.items() if pos(e["T"].cell("all", C["all"])["diff"][("S:own", "S:copy")])]
        lose = [m for m, e in gbf.items() if neg(e["T"].cell("all", C["all"])["diff"][("S:own", "S:copy")])]
        P("- **그룹 B — 각자의 copy 대비.** " + " · ".join(s) + f". 온전한 {len(gbf)} 모델 중 own p 가 own copy 를 전체로 CI 로 이기는 모델 {len(win)}/{len(gbf)}"
          + (f" ({', '.join(win)})" if win else "") + f", CI 로 밑도는 모델 {len(lose)}" + (f" ({', '.join(lose)})" if lose else "")
          + ". encoder 가 다르면 copy 도 다르다 — 그룹 A 의 copy 와 절대값을 비교하지 않는다 (그룹 A: release − copy 는 위).")
    # ---- vanish 방향 (하네스, 전체 쌍): 한 방향이 오르고 다른 방향이 내리면 '맞힘' 이 아니라 선호의 이동이다
    rows = []
    for t in TIMINGS:
        for mm in ("static", "moving"):
            cn_o, cn_e = (f"t={t}|m={mm}|v=van_o2e", f"t={t}|m={mm}|v=van_e2o") if mm == "moving" else \
                         (f"t={t}|m=static|v=van_o2e", f"t={t}|m=static|v=van_e2o")
            for tg in TRAINED:
                do, de = dif(TH, cn_o, f"S:{tg}", "S:release"), dif(TH, cn_e, f"S:{tg}", "S:release")
                if do is None or de is None:
                    continue
                if (pos(do) and neg(de)) or (neg(do) and pos(de)):
                    toward = "'물체 있음' 쪽으로" if pos(do) else "'빈 장면' 쪽으로"
                    rows.append(f"`{tg}` {t}·{mm} ({toward}): 물체→빈 {acc(TH, cn_o, 'S:release')[0]:.0f}→{acc(TH, cn_o, f'S:{tg}')[0]:.0f} / "
                                f"빈→물체 {acc(TH, cn_e, 'S:release')[0]:.0f}→{acc(TH, cn_e, f'S:{tg}')[0]:.0f}")
    both = []
    for tg in TRAINED:
        o_ = acc(TH, "t=late|m=moving|v=van_o2e", f"S:{tg}"); e_ = acc(TH, "t=late|m=moving|v=van_e2o", f"S:{tg}")
        both.append(f"`{tg}` {o_[0]:.0f} / {e_[0]:.0f}")
    r_o = acc(TH, "t=late|m=moving|v=van_o2e", "S:release"); r_e = acc(TH, "t=late|m=moving|v=van_e2o", "S:release")
    P(f"- **vanish 방향 (CLAUDE.md §1-7; 하네스 전체 쌍).** late·moving 물체→빈 / 빈→물체: release {r_o[0]:.0f} / {r_e[0]:.0f} · " + " · ".join(both)
      + ". 한 방향은 CI 로 오르고 다른 방향은 CI 로 내린 칸 — '물체 있음' 과 '빈 장면' 미래 사이의 선호가 한쪽으로 옮겨 간 것이지 둘을 가려낸 것이 아니다 "
      "(물체→빈 이 오르고 빈→물체 가 내리면 p 가 '물체 있음' 미래 쪽으로 옮겨 갔다): "
      + ("; ".join(rows) if rows else "없음") + ". 두 방향이 모두 오른 칸만 '가려냄' 으로 읽는다.")
    # ---- 곡선
    s = []
    for run, (nm, pts) in RUNS.items():
        e = LS.get((run, "all"))
        if not e:
            continue
        t_ = f"{nm}: {e['acc'][0]:.1f} → {e['acc'][-1]:.1f} ({fci(e['d_last_first'], 1, True)})"
        if e["sat"] is not None:
            t_ += f", 90 % 도달 e{e['sat']}"
        if e["osc"]:
            t_ += (", **checkpoint 사이 진동** (이웃 차이 " + " · ".join(fci(d, 1, True) for d in e["step_d"] if d is not None)
                   + f"; 최대 {e['max_abs_step']:.1f} pt — 한 seed 라 block CI 는 이 학습 잡음을 담지 않는다)")
        elif e["peak_ep"] != e["eps"][-1] and e["peak"] - e["acc"][-1] > 1.0:
            t_ += f", **e{e['peak_ep']} 최고 {e['peak']:.1f} 뒤 하락**"
        s.append(t_)
    if s:
        P(f"- **학습 곡선 (추출에 덮인 쌍 전체, {npair:,} 쌍).** " + " · ".join(s) + ".")
    s = []
    for run, (nm, pts) in RUNS.items():
        dn = [c for c in FINE if (run, c) in LS and LS[(run, c)]["mono_down"]]
        pk = [c for c in FINE if (run, c) in LS and LS[(run, c)]["peak_ep"] != LS[(run, c)]["eps"][-1]
              and LS[(run, c)]["peak"] - LS[(run, c)]["acc"][-1] > 5 and LS[(run, c)]["n"] >= 50]
        s.append(f"{nm} 단조 악화 {len(dn)}" + (f" ({', '.join(cname_ko(c) for c in dn)})" if dn else "")
                 + f" · 중간 최고 뒤 5 pt 넘게 하락 {len(pk)}" + (f" ({', '.join(cname_ko(c) for c in pk[:6])}{' …' if len(pk) > 6 else ''})" if pk else ""))
    P("- **곡선의 모양 (세분 칸).** " + " · ".join(s) + ".")
    # ---- 슬롯
    cl = T.cell("t=late", C["t=late"]); cv = T.cell("t=visible", C["t=visible"])
    if cl:
        s = []
        for t in TRAINED:
            dj = [cl["diff"][(f"j{j}:{t}", f"j{j}:release")][0] for j in range(N_SLOT)]
            jm = int(np.argmax(dj))
            s.append(f"`{t}` 최대 j{jm} {dj[jm]:+.1f} (범위 {min(dj):+.1f}…{max(dj):+.1f})")
        rel = [cl["acc"][f"j{j}:release"][0] for j in range(N_SLOT)]
        P(f"- **슬롯별 (late 가림).** release 는 슬롯마다 {min(rel):.1f}…{max(rel):.1f}. Δ vs release: " + " · ".join(s) + ".")
    c2 = T.cell("t=late|k>=2", C["t=late|k>=2"])
    if c2:
        s = []
        for t in TRAINED:
            ds = [c2["diff"][(f"{ph}:{t}", f"{ph}:release")] for ph in PHASES]
            s.append(f"`{t}` " + " / ".join(f"{d[0]:+.1f}{'*' if (pos(d) or neg(d)) else ''}" for d in ds))
        sH, big = [], []
        for t in TRAINED:
            ds = [c2["diff"][(f"{ph}H:{t}", f"{ph}H:release")] for ph in PHASES]
            sH.append(f"`{t}` " + " / ".join(f"{d[0]:+.1f}{'*' if (pos(d) or neg(d)) else ''} {fci(d, 1, True)[len(f'{d[0]:+.1f}'):]}" for d in ds))
            if any(pos(d) for d in ds):
                big.append(f"`{t}` {PHASES[int(np.argmax([d[0] for d in ds]))]}")
        P(f"- **드러남 단계 (late, k ≥ 2, {c2['n']:,} 쌍; hidden / reveal / after 의 Δ vs release, * = CI 가 0 을 안 포함).** "
          "슬롯 수를 맞춘 값 (단계 안 슬롯별 hit 평균 — 정정 2026-09-25): " + " · ".join(sH)
          + (". Δ 가 가장 큰 단계: " + " · ".join(big) if big else "")
          + " (after 가 다음 단계보다 2 pt 넘게 크고 CI>0 인 것만 '몰린다' 로 읽는다 — 한 줄 참조)"
          + ". ⚠️ 옛 값 (단계 안 S 평균 — after 5–6 슬롯 · reveal 1 · hidden 1–2 슬롯을 평균해 슬롯이 많은 단계가 저절로 오른다, **같은 조건 비교가 아니다**): "
          + " · ".join(s) + f". release 자체는 (S 평균) hidden {c2['acc']['hidden:release'][0]:.1f} · reveal {c2['acc']['reveal:release'][0]:.1f} · "
          f"after {c2['acc']['after:release'][0]:.1f}, copy 는 {c2['acc']['hidden:copy'][0]:.1f} · {c2['acc']['reveal:copy'][0]:.1f} · {c2['acc']['after:copy'][0]:.1f}.")
    if cv:
        s = []
        for t in TRAINED:
            dj = [cv["diff"][(f"j{j}:{t}", f"j{j}:release")][0] for j in range(N_SLOT)]
            s.append(f"`{t}` {np.mean(dj):+.1f} (j0 {dj[0]:+.1f}, j7 {dj[-1]:+.1f})")
        P("- **슬롯별 (visible)** 평균 Δ: " + " · ".join(s) + ".")
    # ---- 변화량
    es = [e for e in chg.values() if e]
    if es:
        hc = np.mean([np.mean(e["hcopy_j"]) for e in es]); hs = np.mean([np.mean(e["hstep_j"][1:]) for e in es])
        cp = np.mean([np.mean(e["copy_j"]) for e in es])
        closer = min(e["per"][t]["p_closer_than_copy"] for e in es for t in FINAL)
        st = {t: np.mean([e["per"][t]["step_ratio"] for e in es]) for t in FINAL}
        Rr = {t: np.mean([e["per"][t]["R"] for e in es]) for t in FINAL}
        P(f"- **예측한 변화량.** 진짜 미래의 '변화량' 이 시간에 거의 무관하다: 칸 평균 hcopy (마지막 문맥 튜블릿에서) {hc:.3f} ≈ hstep (바로 앞 튜블릿에서) {hs:.3f}, "
          f"copy {cp:.3f}. 즉 전 토큰 평균 L1 로 잰 h 의 튜블릿 간 거리 (변화 크기) 는 물체 운동에서 오지 않는다 (static ≈ moving). "
          + "hcopy 의 슬롯 모양은 조건끼리 같다 (" + " · ".join(
              f"{mm} j1 {np.mean([chg[f't={tt}|m={mm}']['hcopy_j'][1] for tt in TIMINGS if chg.get(f't={tt}|m={mm}')]):.2f} / "
              f"j3 {np.mean([chg[f't={tt}|m={mm}']['hcopy_j'][3] for tt in TIMINGS if chg.get(f't={tt}|m={mm}')]):.2f} / "
              f"j5 {np.mean([chg[f't={tt}|m={mm}']['hcopy_j'][5] for tt in TIMINGS if chg.get(f't={tt}|m={mm}')]):.2f}"
              for mm in ("static", "ramp")) + ") — 무작위 요동이 아니라 슬롯 번호에 따른 계통 구조다. "
          f"그래서 R (Σpz/Σhcopy, 칸 평균) 은 predictor 마다 " + " · ".join(f"{t} {Rr[t]:.2f}" for t in FINAL)
          + (f" 로 좁게 모이고 크기만으로는 물체를 옮겼는지 말할 수 없다. p 는 모든 칸·predictor 에서 (clip, 슬롯) 의 "
             + ("100 % (예외 0)" if closer >= 100 - 1e-9 else f"최소 {closer:.1f} %") + " 에서 copy 보다 h 에 가깝지만 ")
          + "그것이 쌍 채점에서 copy 를 이긴다는 뜻은 아니다 (§1). "
          f"차이가 나는 것은 걸음비 (pstep/hstep, j ≥ 1): " + " · ".join(f"{t} {st[t]:.2f}" for t in FINAL)
          + f" — release 의 미래 슬롯끼리의 거리는 진짜 걸음의 {st['release']:.2f} 배이고, 이를 키운 predictor ("
          + ", ".join(f"`{t}` ×{st[t] / st['release']:.1f}" for t in TRAINED if st[t] / st["release"] > 1.2)
          + ") 와 키우지 않은 predictor ("
          + ", ".join(f"`{t}` ×{st[t] / st['release']:.1f}" for t in TRAINED if st[t] / st["release"] <= 1.2)
          + ") 가 갈린다. 이것도 크기이지 방향·정답이 아니다.")
    # ---- 도메인
    s, nd, ov = [], {}, {}
    for t in TRAINED:
        d = dif(TH, "all", f"S:{t}", "S:release")
        up = [c for c in FINE if pos(dif(TH, c, f"S:{t}", "S:release"))]
        dn = [c for c in FINE if neg(dif(TH, c, f"S:{t}", "S:release"))]
        nd[t], ov[t] = len(dn), d[0]
        pat = ""
        if dn:
            from collections import Counter
            mc = Counter(c.split("|")[1][2:] for c in dn); vc = Counter(VK_KO[c.split("|")[2][2:]] for c in dn)
            pat = " [내린 칸: 운동 " + " · ".join(f"{k} {v}" for k, v in mc.most_common()) + " / 위반 " + " · ".join(f"{k} {v}" for k, v in vc.most_common()) + "]"
        s.append(f"`{t}` ({DOMAIN[t]}) {fci(d, 1, True)}, 오름 {len(up)} / 내림 {len(dn)}{pat}")
    tail = []
    if ov["v11_e10"] == max(ov.values()):
        tail.append("전체 향상은 도메인 안 `v11_e10` 이 가장 크다")
    if nd["v11_e10"] == min(nd.values()):
        tail.append(f"내린 칸도 가장 적다 ({nd['v11_e10']})")
    ood = [t for t in ("ip1_e40", "ip2_e80", "ariel_ep43") if ov[t] > 0 and nd[t] > 0]
    if ood:
        tail.append("도메인 밖 " + ", ".join(f"`{t}`" for t in ood) + " 는 전체로 오르면서 내린 칸이 함께 있다 (표 5 의 '내린 칸' 열)")
    P("- **도메인 안 vs 밖 (하네스, 10,752 쌍).** " + " · ".join(s) + ". " + (" · ".join(tail) + ". " if tail else "")
      + "`v11_e10` 의 향상은 v11 궤적 재생이다 (단서 'v11 은 궤적 재생이다').")
    P("")

    # ---- 단서
    P("## 단서")
    P("")
    cav = []
    if not full:
        cav.append(f"**부분 데이터.** 추출 done {n_done:,}/{n_rows:,} clip, 온전한 쌍 {npair:,}/{pairs.n:,}. 추출 순서가 조건별이라 "
                   "빠진 칸은 무작위 결측이 아니다 — 곡선·슬롯·copy·변화량의 '전체' 값은 덮인 조건만의 평균이다. §5 (하네스) 만 전체 쌍이다.")
    cav.append("**v11 은 궤적 재생이다.** held-out block 이어도 운동 조건마다 궤적이 2 개 (좌→우 · 우→좌), static 은 자리 8 개뿐이고 학습 절반에도 같은 궤적이 있다. "
               "`v11_e10` 의 향상은 처음 보는 운동으로의 일반화가 아니다. block bootstrap CI 는 block 표집만 반영하고 궤적 수 (2) 에서 오는 불확실성은 담지 못한다 — "
               "궤적 단위로는 표본이 2 라 CI 를 낼 수 없다.")
    cav.append("**다중 비교.** 세분 48 칸 × predictor 5 = 240 개 차이를 95 % CI 로 읽는다. 귀무에서도 약 12 개가 우연히 '0 을 안 포함' 할 수 있다 — "
               "한 칸짜리 변화보다 칸 묶음 (타이밍 · 방향) 에서 같은 방향으로 반복되는 변화를 읽는다.")
    vs_ = [(t, val[t]) for t in FINAL if val.get(t) and val[t]["acc_ext"] is not None]
    if vs_:
        cav.append(f"**근접 동점.** 쌍 margin 이 작아 (§0) fp16 batch 구성만으로 hit 가 뒤집힌다 — 추출과 하네스 사이 flip 이 predictor 마다 "
                   f"{min(v['flips'] for _, v in vs_)}–{max(v['flips'] for _, v in vs_)} 쌍, 전체 정확도 차이 최대 {max(abs(v['diff_same_pairs']) for _, v in vs_):.2f} pt "
                   "(~0.1 pt 목표를 넘는 predictor 가 있다, §0). 칸 단위 1–2 pt 차이는 이 잡음과 같은 규모일 수 있다. §1–4 는 추출 배열, §5 는 하네스 값이라 "
                   "같은 칸도 두 절에서 소수점이 다를 수 있다.")
    cav.append("**copy 는 encoder 두 개를 섞는다.** copy = \\|LN(z)_7 − h_j\\| 는 문맥 encoder (online) 의 마지막 튜블릿과 target encoder (EMA) 의 미래를 잰다. "
               "쌍 안에서는 zT 가 같아 이 차이가 상쇄되지만 크기 비교 (§4 의 R') 에는 남는다 (§4-a 의 copy − hcopy).")
    cav.append("**h 는 32 장 전부를 본 target encoder 의 출력이다.** 가려진 미래 슬롯 (hidden) 의 h 도 뒤의 드러남 프레임 정보를 담을 수 있다 (양방향 attention). "
               "그래서 hidden 슬롯 정확도는 '가려진 내용을 예측했다' 가 아니라 '그 슬롯의 h 와 가까웠다' 이다.")
    cav.append("**'예측한 변화량' 은 크기다.** 전 토큰 평균 L1 이라 물체 (화면의 작은 부분) 가 크기를 정하지 않는다 (hcopy ≈ hstep, static ≈ moving; 슬롯 번호에 따른 계통 모양이 조건끼리 같다). "
               "R · 걸음비가 커진 것은 '물체를 옮겼다' 의 증거가 아니다 — 위치 주장은 readout (p@h, attention 질량 > 0.035) 쪽 분석이 한다.")
    cav.append("**한 번씩 학습했다 (seed 0).** 학습 분산을 모른다. `ariel` 은 attention 규칙 · 데이터 · scratch · epoch 네 가지가 동시에 다르고 "
               "e0 가 없다 (곡선의 시작은 ep5). `ip2` 는 254 clip · loss 0.605 → 0.585 로 거의 안 움직였는데도 점수가 오른 칸이 있다 — "
               "작은 predictor 변화가 near-tie 가 많은 칸을 크게 흔들 수 있다.")
    cav.append("**원인을 목적함수에 걸지 않는다.** 여기서 보이는 것은 '어느 칸이 바뀌었나' 이지 '왜' 가 아니다.")
    cav.append("**copy 대조의 target 은 표준 h 하나다** (32 장 전부를 본 target encoder). IntPhys1 skip2_w32 에서는 causal target h^c 로 바꾸면 release − copy 가 "
               "+3.9 → +6.7 로 움직였다 ([`../ip1_copy/IP1_COPY.md`](../ip1_copy/IP1_COPY.md)). v11 의 h^c 대조는 없다 (미적용 — GPU 추출 필요) → release − copy 의 크기는 target 정의에 달려 있을 수 있다.")
    cav.append("**'가장 많이 오른 칸' (표 5) 은 240 비교 중 상위다.** 귀무에서도 ~12 칸이 우연히 'CI 가 0 을 안 포함' 할 수 있고 오름 칸 수 (예: 17 vs 19) 로는 predictor 를 가를 수 없다. "
               "CI 하한이 정확히 +0.0 인 칸이 있어 칸 수는 bootstrap seed 에 따라 ±1 흔들린다.")
    if ceil_:
        cav.append(f"**천장 칸.** copy · release 모두 ≥ 99 인 칸 {len(ceil_)} 개는 copy 비교에서 뺐다 (이길 여지가 없다).")
    for i, x in enumerate(cav, 1):
        P(f"{i}. {x}")
    P("")


if __name__ == "__main__":
    main()
