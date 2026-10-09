#!/usr/bin/env python3
"""TrainingEffects — RollOut v3 운동: 전이 (transition) · 지속 (persistence) · 도달 거리 (reach), predictor 별 + 학습 곡선 (CPU).

입력: `te_v3_readout.py` 의 curves 추출 (`cache/training_effects/v3readout_curves/`). 끝났으면 `readings.npz`, 진행 중이면
      `_parts/` memmap + done 플래그 (창마다 **모든 tag · zh 가 done 인 clip 만** 쓴다 = 모든 predictor 가 같은 clip).
      수정하지 않는다. 메타 (진실 · in_frame · 궤적) 는 `data_csv/rollout_v3/index.csv` 에서 (추출과 같은 식, clip 순서 sha1 대조).
출력: `z_research/TrainingEffects/v3_motion/` — MOTION.md (본문) · TABLES.md (전 칸 표) · motion.json · fig_*.png

정의 (README `z_research/TrainingEffects/README.md` §1, `audit_v3_motion.py` 머리말을 따른다 — 읽기 전용 재사용)
  좌표   X(t) = GT 프레임 (32+2t, 33+2t) 평균 (px, 288 화면), x0 = GT 프레임 30·31 평균, v0 = x0 − GT 28·29 평균 (px/튜블릿)
  읽기   x̂ = (readings[..., :2] + 1)·144 − 치우침[자] (`attn_bias_px.json`),  '있다' = logit > 문턱[자] (자마다 하나, 릴리즈에서 정한 값):
         h 자 −0.961 (p@h, **주 판독**) · p 자 0.8307 (p@p, 보조 = 릴리즈 p 로 배운 자의 이식 판독) · z 자 −0.662 (자 검사)
  질량   mass3 = 진실 칸 3×3 attention 질량 (추출이 저장, 균등 9/256 = 0.035).  '모임' (localized) = '있다' ∧ mass3 > 0.035
  읽을 수 있는 칸 = 물체가 두 프레임 모두 화면 안 ∧ 가장자리에서 36 px 이상 (GT 기준 → 모든 predictor 에 같다)
  부호   s_a(t) = `plot_v3_l2_error.axis_dir` (import, audit 과 같은 함수): GT 운동 방향, 멈추면 마지막 방향, 안 움직인 y 는 위 = +
         부호 오차 e_a = (x̂_a − X_a)·s_a  (+ 앞섬 / − 뒤처짐).  기준선 (GT 만): 복사 = (x0 − X)·s,  등속 연장 = (x0 + v0(t+1) − X)·s
  이득   β = Σ (x̂_a − x0_a)·s_a / Σ (X_a − x0_a)·s_a  (궤적 안, 마스크 칸 중 |GT 이동| ≥ 1 칸 = 18 px 인 칸만).  복사 β = 0, GT β = 1,
         연장 β_ext = Σ v0·s·(t+1) / Σ GT 이동 (같은 칸)
  귀무   변위 귀무 = 같은 법칙 **다른 궤적**의 GT 이동과 읽은 이동의 차 (|d̂_i − d_o| 을 o ≠ i 에 평균); own − null < 0 = 궤적 특이적.
         위치 귀무 = 같은 법칙 다른 궤적의 GT 위치 (|x̂_i − X_o|) — 궤적마다 자리가 달라 약하다 (audit) → JSON 에만
  지속   recall(t) = 읽을 수 있는 칸 중 '있다' 비율 (궤적 안 복사본 평균 → 궤적 값). T50 = 자유 6 법칙 (층화) 곡선이 처음 0.5 아래로
         가는 튜블릿 (끝까지 안 가면 '≥ 창 길이' = 중도절단). 같은 것을 '모임' 비율로도 (문턱 + 질량, 위치까지 요구)
  도달   (flat_v, 궤적마다) t_cliff = '모임' 비율이 처음 0.5 아래인 튜블릿, reach = 그 직전 튜블릿의 GT 이동 (px) — 물체를 **진실
         자리 근처에** 두는 마지막 거리. plateau = 읽은 이동의 최댓값 ('있다' 칸, 질량 무관) = 자 기본값이 섞인 '멈춘 듯한' 값 (멈춤 아님)
  마스크 위치·운동 값은 **'모임' ∧ 읽을 수 있는 칸** 에서만 (CLAUDE.md §5-5). predictor 비교 (Δ vs release) 는 **두 predictor 모두
         '모임' 인 공통 칸** 에서 (칸 단위로 짝지음). 민감도: '있다' 만 (질량 무관) 공통 칸.
  단위 · CI  궤적 = (scenario, primary, secondary): 자유 6 법칙 × 14 = 84, ledge · wall 각 14. 외형 복사본 (28 / 14) 은 궤적 안에서 먼저
         평균 (칸 수 가중). 95 % CI = 법칙 안 궤적 14 개 bootstrap (4,000 회, seed 0, 법칙마다 고정 인덱스 → **모든 predictor 가 같은
         재추출** = 짝지은 Δ). 여러 법칙 묶음 = 법칙 평균의 평균 (층화). 법칙 대비 (law − flat_v) 는 두 법칙을 독립 재추출.
  범위   early t0–3 · late t4–7 · beyond t8–15 (P32 만; 자 셋 · v11 · Predictor_v1 post-FT (미래 ≤ 8 튜블릿) 의 학습 지평 밖.
         IntPhys1 post-FT (≤ 14) · IntPhys2 post-FT (≤ 18) 에는 (일부) 안 — 2026-09-25 정정)
  보정   (2026-09-25 정정) e_oc = e(범위) − raw(t0–1)·s(범위): t0–1 의 **화면 좌표** 치우침을 빼고 late 부호를 곱한다
  적합   (2026-09-25) 궤적 사이 OLS  읽은 이동 = 절편 + 기울기·GT 이동 (법칙 안 14 궤적, 같은 재추출) — own−null 대신
  방향 판정 (공통 칸, late):  Δe = e_k − e_R 의 CI 가 0 을 넘으면 '움직임'. 그때 기준 X ∈ {GT, 연장, 복사} 마다
         D_X = |e_k − X| − |e_R − X| (궤적 값) 의 CI 가 < 0 이면 'X 쪽으로'. 기준끼리 1 칸 안이면 둘 다 참일 수 있다 → 간격을 같이 싣는다.

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/te_analyze_v3.py                        # curves (끝났으면 readings.npz, 아니면 done 행)
  $P z_research/scripts/analysis/te_analyze_v3.py --nb 400 --no-fig      # 빠른 개발용
  $P z_research/scripts/analysis/te_analyze_v3.py --cache /data2/.../v3readout_final_sample48 --out /tmp/x   # 표본 (개발)
"""
from __future__ import annotations
import argparse, csv, hashlib, json, sys, time, warnings
from pathlib import Path

import numpy as np

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
sys.path.insert(0, str(ROOT / "z_research/scripts/figures"))
from plot_v3_l2_error import axis_dir  # noqa: E402  부호 규칙 정본 (audit_v3_motion 이 원본과 비트 대조한 함수)

CACHE = Path("/data2/local_datasets/world/world_analysis/cache/training_effects/v3readout_curves")
OUT = ROOT / "z_research/TrainingEffects/v3_motion"
INDEX = ROOT / "data_csv/rollout_v3/index.csv"
PRES = ROOT / "z_research/RollOutV3/exp_results/presence"
REF_WIN = ROOT / "z_research/RollOutV3/exp_results/windows/readings.npz"
REF_SIGNED = {"p": ROOT / "z_research/RollOutV3/figures/v3_signed_error/_superseded/presence_decoder/values.json",
              "h": ROOT / "z_research/RollOutV3/figures/v3_signed_error_hhead/_superseded/presence_decoder/values.json"}
EXPECT_FP = "dcd24d8a8a47"
RES, SPLIT, CELL, EDGE, W = 144.0, 32, 18.0, 36.0, 288.0
MASS_U = 9 / 256
THR = {"p": 0.8307, "h": -0.961, "z": -0.662}
LAWS = ["flat_v", "flat_a", "flat_d", "ramp_a", "ramp_d", "arc", "ledge", "wall"]
FREE = ["flat_v", "flat_a", "flat_d", "ramp_a", "ramp_d", "arc"]
XLAWS = ["flat_v", "flat_a", "flat_d", "ramp_a", "ramp_d", "arc"]          # x 로 움직이는 자유 법칙 (도달 거리 풀링)
WINDOWS = [(16, 16), (16, 32)]
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
READS = {"ph": ("p@h", "h"), "pp": ("p", "p")}                             # 판독 이름 → (키 앞머리, 자)
KEYAX = [("flat_v", 0), ("flat_a", 0), ("flat_d", 0), ("ramp_a", 0), ("ramp_d", 0), ("arc", 1), ("ledge", 1), ("wall", 0)]
EXTRA_AX = [(l, a) for l in LAWS for a in (0, 1) if (l, a) not in KEYAX]   # 나머지 축 (flat·wall y = 운동 없음 → 자의 y 치우침 대조)
AXN = "xy"
SUMMARY = {   # 이름 → (법칙·축 목록, 뜻) — 전부 C16/P16 late (t4–7) 가 기본
    "common_lag": ([("flat_v", 0)], "flat_v x 부호 오차 (공통 치우침; GT 0 = 연장 0)"),
    "accel_lag": ([("flat_a", 0), ("ramp_a", 0)], "가속 x 부호 오차 (− = 뒤처짐)"),
    "decel_lead": ([("flat_d", 0), ("ramp_d", 0)], "감속 x 부호 오차 (+ = 앞섬)"),
    "arc_fall": ([("arc", 1)], "포물선 y 부호 오차 (− = 덜 떨어짐)"),
    "ledge_fall": ([("ledge", 1)], "선반 낙하 y 부호 오차 (− = 덜 떨어짐; 사건)"),
    "wall_over": ([("wall", 0)], "벽 x 부호 오차 (+ = 정지점을 지나감; 사건)"),
}
NB, SEED = 4000, 0
CROSS_HEAD = {   # RollOutV3/exp_results/windows/CROSS_HEAD.md 에 인쇄된 release recall (%, in-frame 칸, 전 8 법칙) — sanity 대조용
    "C16_P16/p_release": [96, 97, 99, 97, 97, 94, 95, 90], "C16_P16/p@h_release": [100, 97, 99, 100, 88, 86, 100, 85],
    "C16_P32/p_release": [95, 96, 97, 96, 97, 94, 94, 86, 99, 91, 12, 35, 67, 7, 40, 23],
    "C16_P32/p@h_release": [100, 96, 96, 99, 89, 83, 100, 82, 97, 100, 60, 75, 82, 69, 73, 43]}
warnings.filterwarnings("ignore", category=RuntimeWarning)


def r2(x, nd=2):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


# ============================================================================== 데이터
def load(cache: Path):
    rows = list(csv.DictReader(INDEX.open()))
    byid = {r["video_id"]: r for r in rows}
    roll = [r["video_id"] for r in rows if r["role"] == "roll"]
    f = cache / "readings.npz"
    arr, valid = {}, {}
    if f.exists():
        Z = np.load(f, allow_pickle=True)
        assert str(Z["decoder_fp"]) == EXPECT_FP, f"자 지문 {Z['decoder_fp']}"
        ids = [str(v) for v in Z["video_id"]]
        for k in Z.files:
            if k.startswith("C") and "_P" in k:
                arr[k] = np.asarray(Z[k])
        rest = [k.split("_", 2)[2] for k in arr]                              # 'C16_P16_p@h_v11_e10' → 'p@h_v11_e10'
        tags = sorted({r[4:] for r in rest if r.startswith("p@h_")}, key=_tag_order)
        for c, p in WINDOWS:
            valid[(c, p)] = np.ones(len(ids), bool) if f"C{c}_P{p}_h" in arr else np.zeros(len(ids), bool)
        source = f"{f} (조립 완료)"
        complete = True
    else:
        parts = cache / "_parts"
        sig = json.loads((parts / "sig.json").read_text())
        assert sig["decoder_fp"] == EXPECT_FP
        ids = roll
        sha = hashlib.sha1("\n".join(ids).encode()).hexdigest()[:16]
        assert sig["n"] == len(ids) and sig["video_id_sha1"] == sha, "clip 순서가 추출과 다르다"
        tags = sorted(sig["tags"], key=_tag_order)
        complete = True
        for c, p in WINDOWS:
            fl = [parts / f"done_C{c}_P{p}__{g}.npy" for g in ["zh"] + tags]
            if not all(x.exists() for x in fl):
                valid[(c, p)] = np.zeros(len(ids), bool); complete = False; continue
            valid[(c, p)] = np.logical_and.reduce([np.load(x).astype(bool) for x in fl])
            complete &= bool(valid[(c, p)].all())
            for k in ["z", "h"] + [f"{pre}_{t}" for t in tags for pre in ("p", "p@h")]:
                arr[f"C{c}_P{p}_{k}"] = np.load(parts / f"C{c}_P{p}__{k}.npy", mmap_mode="r")
        source = f"{parts} (done 행)"
    sel = [byid[v] for v in ids]
    a = lambda s: np.array([float(v) for v in str(s).split()], np.float64)  # noqa: E731
    L = np.stack([np.stack([a(r["px_x_by_sample"]), a(r["px_y_by_sample"])], -1) for r in sel])   # (n,64,2) px
    inf = np.stack([a(r["in_frame_by_sample"]) for r in sel]) > 0
    tkey = np.array([f"{r['scenario']}|{r['primary']}|{r['secondary']}" for r in sel])
    ut = sorted(set(tkey), key=lambda s: (LAWS.index(s.split("|")[0]), float(s.split("|")[1]), float(s.split("|")[2])))
    D = dict(ids=ids, n=len(ids), L=L, inf=inf, sc=np.array([r["scenario"] for r in sel]), tkey=tkey, UT=ut,
             TID=np.array([ut.index(k) for k in tkey]), TLAW=np.array([u.split("|")[0] for u in ut]),
             arr=arr, valid=valid, tags=tags, source=source, complete=complete)
    D["LT"] = {law: np.where(D["TLAW"] == law)[0] for law in LAWS}
    return D


def _tag_order(t):
    fam = t.split("_")[0]
    order = ["release", "v11", "pv1", "ip1", "ip2", "ariel"]
    ep = int("".join(ch for ch in t.split("_")[-1] if ch.isdigit()) or 0) if "_" in t else 0
    return (order.index(fam) if fam in order else 99, ep)


def check_thresholds():
    S = json.loads((PRES / "summary.json").read_text())
    for r, v in THR.items():
        got = float(S["reps"][r]["attn"]["thr_val_fpr5"])
        assert abs(got - v) < 1e-3, f"문턱 {r}: {got} vs {v}"
    return {r: np.array(v, np.float64) for r, v in json.loads((PRES / "attn_bias_px.json").read_text()).items()}


# ============================================================================== 기하 · 읽기
def geom(D, c, p):
    n, tp, L = D["n"], p // 2, D["L"]
    X = L[:, SPLIT:SPLIT + p].reshape(n, tp, 2, 2).mean(2)
    x0 = L[:, SPLIT - 2:SPLIT].mean(1)
    v0 = x0 - L[:, SPLIT - 4:SPLIT - 2].mean(1)
    s, never = axis_dir(L, X)
    inf = D["inf"][:, SPLIT:SPLIT + p].reshape(n, tp, 2).all(2)
    edge = np.minimum(np.minimum(X[..., 0], W - X[..., 0]), np.minimum(X[..., 1], W - X[..., 1]))
    T = np.arange(tp)
    extp = x0[:, None] + v0[:, None] * (T + 1)[None, :, None]
    return dict(c=c, p=p, tp=tp, X=X, x0=x0, v0=v0, s=s, never=never, inf=inf, readable=inf & (edge >= EDGE),
                copy=(x0[:, None] - X) * s, copy_raw=(x0[:, None] - X), ext=(extp - X) * s, dgt=(X - x0[:, None]) * s, dext=(extp - x0[:, None]) * s,
                speed=np.linalg.norm(v0, axis=-1))


def reading(D, G, key, ruler, bias):
    v = np.asarray(D["arr"][f"C{G['c']}_P{G['p']}_{key}"], np.float64)
    xy = (v[..., :2] + 1.0) * RES - bias[ruler]
    pres = v[..., 2] > THR[ruler]
    mass = v[..., 4]
    return dict(xy=xy, pres=pres, mass=mass, loc=pres & (mass > MASS_U),
                e=(xy - G["X"]) * G["s"], raw=(xy - G["X"]), dr=(xy - G["x0"][:, None]) * G["s"], logit=v[..., 2])


# ============================================================================== 궤적 평균 · bootstrap
class Agg:
    """clip (n, T[, A]) → 궤적 (U, T[, A]). valid 가 아닌 clip 은 빠진다. 칸 수 가중 (외형 복사본을 먼저 평균)."""

    def __init__(self, D, valid):
        self.U, self.valid = len(D["UT"]), valid
        self.O = np.zeros((self.U, D["n"]))
        idx = np.where(valid)[0]
        self.O[D["TID"][idx], idx] = 1.0

    def sc(self, val, mask):
        m = (np.asarray(mask, bool) & self.valid[:, None]).astype(np.float64)
        v = np.nan_to_num(np.asarray(val, np.float64))
        n = m.shape[0]
        c = (self.O @ m.reshape(n, -1)).reshape((self.U,) + m.shape[1:])
        if v.ndim == 3:
            s = (self.O @ (v * m[..., None]).reshape(n, -1)).reshape((self.U,) + v.shape[1:])
            return s, c[..., None]
        return (self.O @ (v * m).reshape(n, -1)).reshape((self.U,) + v.shape[1:]), c

    def mean_t(self, val, mask):
        s, c = self.sc(val, mask)
        return np.where(c > 0, s / np.maximum(c, 1e-12), np.nan)

    def mean_r(self, val, mask, tr):
        s, c = self.sc(val, mask)
        s, c = s[:, tr].sum(1), c[:, tr].sum(1)
        return np.where(c > 0, s / np.maximum(c, 1e-12), np.nan)

    def ratio_r(self, num, den, mask, tr):
        """Σ num / Σ den (궤적 안, 마스크 칸, 범위 tr)."""
        sn, c = self.sc(num, mask); sd, _ = self.sc(den, mask)
        sn, sd, c = sn[:, tr].sum(1), sd[:, tr].sum(1), c[:, tr].sum(1)
        return np.where((c > 0) & (np.abs(sd) > 1e-9), sn / np.where(np.abs(sd) > 1e-9, sd, 1.0), np.nan)


class Boot:
    def __init__(self, D, nb, seed):
        rng = np.random.default_rng(seed)
        self.LT = D["LT"]; self.nb = nb
        self.I = {law: rng.integers(0, len(self.LT[law]), (nb, len(self.LT[law]))) for law in LAWS}

    def by(self, a, laws):
        return {l: np.asarray(a, np.float64)[self.LT[l]] for l in laws}

    def bs(self, a, laws):
        """(점추정, (nb,) 재추출 분포, n 유한 궤적) — 법칙 평균의 평균 (층화)."""
        bl = self.by(a, laws)
        ok = [l for l in laws if np.isfinite(bl[l]).any()]
        if not ok:
            return np.nan, np.full(self.nb, np.nan), 0
        pt = np.mean([np.nanmean(bl[l]) for l in ok])
        dist = np.mean([np.nanmean(bl[l][self.I[l]], axis=1) for l in ok], axis=0)
        return pt, dist, int(sum(np.isfinite(bl[l]).sum() for l in ok))

    def est(self, a, laws, nd=2):
        pt, dist, n = self.bs(a, laws)
        if n == 0:
            return [None, None, None, 0]
        lo, hi = np.nanpercentile(dist, [2.5, 97.5])
        return [r2(pt, nd), r2(lo, nd), r2(hi, nd), n]

    def est_diff(self, a, la, b, lb, nd=2):
        """두 법칙 (독립 궤적) 평균의 차 a(la) − b(lb)."""
        pa, da, na = self.bs(a, [la]); pb, db, nb_ = self.bs(b, [lb])
        if na == 0 or nb_ == 0:
            return [None, None, None, 0]
        lo, hi = np.nanpercentile(da - db, [2.5, 97.5])
        return [r2(pa - pb, nd), r2(lo, nd), r2(hi, nd), min(na, nb_)]


def sig(ci):
    """CI 가 0 을 넘지 않나: +1 / −1 / 0."""
    if ci is None or ci[0] is None:
        return 0
    return 1 if ci[1] > 0 else (-1 if ci[2] < 0 else 0)


def t50(curve, tp):
    b = np.where(curve < 0.5)[0]
    return int(b[0]) if len(b) else tp


def olsfit(x, y, I):
    """궤적 사이 OLS y = a + b·x. I = (nb, m) 법칙 안 재추출 인덱스. → dict(slope, intercept = [pt, lo, hi, n], r)."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4 or np.std(x[ok]) < 1e-9:
        return dict(slope=[None, None, None, 0], intercept=[None, None, None, 0], r=None)
    b, a = np.polyfit(x[ok], y[ok], 1)
    X, Y = x[I], y[I]; M = np.isfinite(X) & np.isfinite(Y)
    X, Y = np.where(M, X, np.nan), np.where(M, Y, np.nan)
    mx, my = np.nanmean(X, 1, keepdims=True), np.nanmean(Y, 1, keepdims=True)
    vx = np.nansum((X - mx) ** 2, 1)
    bs = np.where(vx > 1e-9, np.nansum((X - mx) * (Y - my), 1) / np.where(vx > 1e-9, vx, 1.0), np.nan)
    as_ = my[:, 0] - bs * mx[:, 0]
    n = int(ok.sum())
    return dict(slope=[r2(b), r2(np.nanpercentile(bs, 2.5)), r2(np.nanpercentile(bs, 97.5)), n],
                intercept=[r2(a, 1), r2(np.nanpercentile(as_, 2.5), 1), r2(np.nanpercentile(as_, 97.5), 1), n],
                r=r2(np.corrcoef(x[ok], y[ok])[0, 1]))


# ============================================================================== 창 하나
def window(D, c, p, bias, B, tags):
    G = geom(D, c, p)
    tp = G["tp"]; T = np.arange(tp)
    valid = D["valid"][(c, p)]
    A = Agg(D, valid)
    rd = G["readable"]
    RANGES = {"t0-3": [0, 1, 2, 3], "t4-7": [4, 5, 6, 7]}
    if tp >= 16:
        RANGES["t8-15"] = list(range(8, 16))
    out = dict(C=c, P=p, tp=tp, n_clip=int(valid.sum()), n_traj=int(np.isfinite(A.mean_t(np.ones((D["n"], tp)), np.ones((D["n"], tp), bool))[:, 0]).sum()),
               ranges=RANGES, readable_frac_by_law={l: [r2(v) for v in rd[(D["sc"] == l) & valid].mean(0)] for l in LAWS},
               persistence={}, transition={}, reach={}, contrast={}, summary={}, ruler_check={})
    Tcopy = {a: A.mean_t(G["copy"][..., a], np.ones_like(rd)) for a in (0, 1)}     # GT 만: 궤적 안에서 같다
    Text = {a: A.mean_t(G["ext"][..., a], np.ones_like(rd)) for a in (0, 1)}

    # ---------------------------------------------------------------- 자 검사 (predictor 무관): h@h · z@z
    for nm, key, ru in (("h@h", "h", "h"), ("z@z", "z", "z")):
        R = reading(D, G, key, ru, bias)
        rec = A.mean_t(R["pres"], rd); loc = A.mean_t(R["loc"], rd)
        cur = np.array([B.bs(rec[:, t], FREE)[0] for t in T])
        out["ruler_check"][nm] = dict(recall_free6=[B.est(rec[:, t], FREE) for t in T],
                                      loc_free6=[B.est(loc[:, t], FREE) for t in T], T50=t50(cur, tp),
                                      L2_free6=[B.est(A.mean_t(np.linalg.norm(R["xy"] - G["X"], axis=-1), R["loc"] & rd)[:, t], FREE) for t in T],
                                      flat_v_y_e=[B.est(A.mean_t(R["e"][..., 1], R["loc"] & rd)[:, t], ["flat_v"]) for t in T])   # 2026-09-25: 자 자기 토큰 y 치우침

    # ---------------------------------------------------------------- predictor 별 읽기 (한 번)
    RD = {rn: {t: reading(D, G, f"{pre}_{t}", ru, bias) for t in tags} for rn, (pre, ru) in READS.items()}

    for rn in READS:
        Rr = RD[rn]["release"]
        # ---------------- 지속
        rec = {t: A.mean_t(RD[rn][t]["pres"], rd) for t in tags}
        loc = {t: A.mean_t(RD[rn][t]["loc"], rd) for t in tags}
        mas = {t: A.mean_t(RD[rn][t]["mass"], rd) for t in tags}
        pers = {}
        for t in tags:
            d = {}
            for nm, M in (("recall", rec), ("loc", loc)):
                curve = [B.est(M[t][:, j], FREE) for j in T]
                dcurve = [B.est(M[t][:, j] - M["release"][:, j], FREE) for j in T]
                pt = np.array([B.bs(M[t][:, j], FREE)[0] for j in T])
                dist = np.stack([B.bs(M[t][:, j], FREE)[1] for j in T], 1)            # (nb, tp)
                distR = np.stack([B.bs(M["release"][:, j], FREE)[1] for j in T], 1)
                tb = np.array([t50(x, tp) for x in dist]); tbR = np.array([t50(x, tp) for x in distR])
                rng_ = {rk: B.est(np.nanmean(M[t][:, rv], 1), FREE) for rk, rv in RANGES.items()}
                drng = {rk: B.est(np.nanmean(M[t][:, rv], 1) - np.nanmean(M["release"][:, rv], 1), FREE) for rk, rv in RANGES.items()}
                bylaw = {l: {rk: B.est(np.nanmean(M[t][:, rv], 1), [l]) for rk, rv in RANGES.items()} for l in LAWS}
                per_traj_t50 = [t50(M[t][i], tp) for i in np.concatenate([B.LT[l] for l in FREE]) if np.isfinite(M[t][i]).all()]
                d[nm] = dict(curve=curve, d_curve=dcurve, T50=t50(pt, tp), T50_ci=[r2(np.percentile(tb, 2.5)), r2(np.percentile(tb, 97.5))],
                             T50_censored_frac=r2(np.mean(tb >= tp)),
                             dT50=[int(t50(pt, tp) - t50(np.array([B.bs(M['release'][:, j], FREE)[0] for j in T]), tp)),
                                   r2(np.percentile(tb - tbR, 2.5)), r2(np.percentile(tb - tbR, 97.5))],
                             ranges=rng_, d_ranges=drng, by_law=bylaw,
                             traj_T50_median=r2(np.median(per_traj_t50)) if per_traj_t50 else None,
                             traj_T50_censored=r2(np.mean(np.array(per_traj_t50) >= tp)) if per_traj_t50 else None)
            d["mass_curve"] = [B.est(mas[t][:, j], FREE, 3) for j in T]
            d["mass_ranges"] = {rk: B.est(np.nanmean(mas[t][:, rv], 1), FREE, 3) for rk, rv in RANGES.items()}
            d["d_mass_ranges"] = {rk: B.est(np.nanmean(mas[t][:, rv], 1) - np.nanmean(mas["release"][:, rv], 1), FREE, 3) for rk, rv in RANGES.items()}
            pers[t] = d
        out["persistence"][rn] = pers

        # ---------------- 전이
        trans = {}
        for t in tags:
            Rk = RD[rn][t]
            Mk = Rk["loc"] & rd; MR = Rr["loc"] & rd; Mc = Mk & MR
            Pc = Rk["pres"] & Rr["pres"] & rd
            dd = {}
            ONE = np.ones_like(rd)
            # 칸 합 (U, T, 2) 을 마스크마다 한 번만 — 법칙·축·범위는 여기서 잘라 쓴다
            S = {}
            for mn, M in (("k", Mk), ("c", Mc), ("pc", Pc)):
                S[mn] = {nm: A.sc(v, M) for nm, v in (("e", Rk["e"]), ("copy", G["copy"]), ("ext", G["ext"]), ("dr", Rk["dr"]))}
                if t != "release" and mn != "k":
                    S[mn]["eR"] = A.sc(Rr["e"], M); S[mn]["drR"] = A.sc(Rr["dr"], M)
            S["k"]["dgt"] = A.sc(G["dgt"], Mk)                                             # 2026-09-25: 기울기 적합용
            big = np.abs(G["dgt"]) >= CELL                                                   # (n, T, 2)
            Sb = {}
            for mn, M in (("k", Mk), ("c", Mc)):
                Mb = M[..., None] & big
                Sb[mn] = dict(dr=A.sc(Rk["dr"] * Mb, M), dgt=A.sc(G["dgt"] * Mb, M), dext=A.sc(G["dext"] * Mb, M))
                if t != "release" and mn == "c":
                    Sb[mn]["drR"] = A.sc(Rr["dr"] * Mb, M)
            Sxy = A.sc(Rk["xy"], Mk); SX = A.sc(G["X"], ONE); Sdg = A.sc(G["dgt"], ONE)

            def mt(sc_, a):                       # (U, T) 궤적 평균
                s_, c_ = sc_
                return np.where(c_[..., 0] > 0, s_[..., a] / np.maximum(c_[..., 0], 1e-12), np.nan)

            def mr(sc_, a, rv):                   # (U,) 범위 평균 (칸 수 가중)
                s_, c_ = sc_
                s1, c1 = s_[:, rv, a].sum(1), c_[:, rv, 0].sum(1)
                return np.where(c1 > 0, s1 / np.maximum(c1, 1e-12), np.nan)

            def rat(num, den, a, rv):
                s1, s2 = num[0][:, rv, a].sum(1), den[0][:, rv, a].sum(1)
                return np.where(np.abs(s2) > 1e-9, s1 / np.where(np.abs(s2) > 1e-9, s2, 1.0), np.nan)

            for law, a in KEYAX + EXTRA_AX:
                key = f"{law}|{AXN[a]}"
                inlaw = (D["sc"] == law) & valid
                ek_t, cp_t, ex_t = mt(S["k"]["e"], a), mt(S["k"]["copy"], a), mt(S["k"]["ext"], a)
                x = dict(e_t=[B.est(ek_t[:, j], [law]) for j in T],
                         copy_t=[B.est(cp_t[:, j], [law]) for j in T],
                         ext_t=[B.est(ex_t[:, j], [law]) for j in T],
                         n_cells_t=[int((Mk[:, j] & inlaw).sum()) for j in T],
                         frac_loc_t=[r2(Mk[inlaw & rd[:, j], j].mean()) if (inlaw & rd[:, j]).any() else None for j in T])
                if t != "release":
                    dct = mt(S["c"]["e"], a) - mt(S["c"]["eR"], a)
                    x["d_e_t"] = [B.est(dct[:, j], [law]) for j in T]
                rr = {}
                for rk, rv in RANGES.items():
                    y = dict(e=B.est(mr(S["k"]["e"], a, rv), [law]), copy=B.est(mr(S["k"]["copy"], a, rv), [law]),
                             ext=B.est(mr(S["k"]["ext"], a, rv), [law]),
                             beta=B.est(rat(Sb["k"]["dr"], Sb["k"]["dgt"], a, rv), [law]),
                             beta_ext=B.est(rat(Sb["k"]["dext"], Sb["k"]["dgt"], a, rv), [law]),
                             n_cells=int((Mk[:, rv] & inlaw[:, None]).sum()))
                    if t != "release":
                        ek, eR = mr(S["c"]["e"], a, rv), mr(S["c"]["eR"], a, rv)
                        cp, ex = mr(S["c"]["copy"], a, rv), mr(S["c"]["ext"], a, rv)
                        u = B.LT[law]
                        y["common"] = dict(e_k=B.est(ek, [law]), e_R=B.est(eR, [law]), copy=B.est(cp, [law]), ext=B.est(ex, [law]),
                                           d_e=B.est(ek - eR, [law]), D_gt=B.est(np.abs(ek) - np.abs(eR), [law]),
                                           D_ext=B.est(np.abs(ek - ex) - np.abs(eR - ex), [law]),
                                           D_copy=B.est(np.abs(ek - cp) - np.abs(eR - cp), [law]),
                                           sep_ext_gt=r2(np.nanmean(np.abs(ex[u]))), sep_copy_gt=r2(np.nanmean(np.abs(cp[u]))),
                                           sep_ext_copy=r2(np.nanmean(np.abs(ex - cp)[u])),
                                           n_cells=int((Mc[:, rv] & inlaw[:, None]).sum()),
                                           d_beta=B.est(rat(Sb["c"]["dr"], Sb["c"]["dgt"], a, rv) - rat(Sb["c"]["drR"], Sb["c"]["dgt"], a, rv), [law]))
                        ekp, eRp = mr(S["pc"]["e"], a, rv), mr(S["pc"]["eR"], a, rv)
                        y["common_present_only"] = dict(d_e=B.est(ekp - eRp, [law]), e_k=B.est(ekp, [law]), e_R=B.est(eRp, [law]),
                                                        n_cells=int((Pc[:, rv] & inlaw[:, None]).sum()))
                    # 변위 귀무 · 위치 귀무 (같은 법칙 다른 궤적) — 궤적 값 (범위 평균)
                    u = B.LT[law]; m_ = len(u)
                    drt, dgt = mr(S["k"]["dr"], a, rv)[u], mr(Sdg, a, rv)[u]
                    Xr, Xg = mr(Sxy, a, rv)[u], mr(SX, a, rv)[u]
                    off = ~np.eye(m_, dtype=bool)
                    nul = np.array([np.nanmean(np.abs(drt[i] - dgt[off[i]])) for i in range(m_)])
                    pnul = np.array([np.nanmean(np.abs(Xr[i] - Xg[off[i]])) for i in range(m_)])
                    full = np.full(len(D["UT"]), np.nan); full[u] = np.abs(drt - dgt) - nul
                    fullp = np.full(len(D["UT"]), np.nan); fullp[u] = np.abs(Xr - Xg) - pnul
                    y["own_minus_null_disp"] = B.est(full, [law]); y["own_minus_null_pos"] = B.est(fullp, [law])
                    # 2026-09-25 (적대 검증): 궤적 사이 기울기 + 절편 — 읽은 이동 = a + b·GT 이동 (자기 '모임' 칸). 곱셈꼴 이득이면 a≈0, b≈β;
                    # 더하기꼴 앞섬이면 b≈1, a>0. own−null 은 공통 오프셋이 GT 이동의 퍼짐보다 크면 추적을 못 잡는다 → 이것으로 대체
                    y["fit"] = olsfit(mr(S["k"]["dgt"], a, rv)[u], drt, B.I[law])
                    rr[rk] = y
                x["ranges"] = rr
                dd[key] = x
            trans[t] = dd
        out["transition"][rn] = trans

        # ---------------- 법칙 대비 (law − flat_v), late: 공통 치우침을 뺀다
        def cstat(arrs, law):
            """법칙 대비 c = mean(law) − mean(flat_v), 두 법칙 독립 재추출 (모든 arr 에 같은 인덱스 → 짝지은 비교)."""
            ul, uv = B.LT[law], B.LT["flat_v"]
            pt = {k: np.nanmean(v[ul]) - np.nanmean(v[uv]) for k, v in arrs.items()}
            bs = {k: np.nanmean(v[ul][B.I[law]], 1) - np.nanmean(v[uv][B.I["flat_v"]], 1) for k, v in arrs.items()}
            return pt, bs

        def ci_(pt, bs, nd=2):
            if not np.isfinite(pt):
                return [None, None, None, 0]
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            return [r2(pt, nd), r2(lo, nd), r2(hi, nd), 14]

        con = {}
        for t in tags:
            Rk = RD[rn][t]; Mk = Rk["loc"] & rd; Mc = Mk & (Rr["loc"] & rd)
            cc = {}
            for rk, rv in RANGES.items():
                ek = A.mean_r(Rk["e"][..., 0], Mk, rv); ex = A.mean_r(G["ext"][..., 0], Mk, rv)
                cp = A.mean_r(G["copy"][..., 0], Mk, rv)
                big0 = np.abs(G["dgt"][..., 0]) >= CELL
                bk = A.ratio_r(Rk["dr"][..., 0], G["dgt"][..., 0], Mk & big0, rv)
                bx = A.ratio_r(G["dext"][..., 0], G["dgt"][..., 0], Mk & big0, rv)
                if t != "release":
                    ekc, eRc = A.mean_r(Rk["e"][..., 0], Mc, rv), A.mean_r(Rr["e"][..., 0], Mc, rv)
                    exc = A.mean_r(G["ext"][..., 0], Mc, rv)
                    bkc = A.ratio_r(Rk["dr"][..., 0], G["dgt"][..., 0], Mc & big0, rv)
                    bRc = A.ratio_r(Rr["dr"][..., 0], G["dgt"][..., 0], Mc & big0, rv)
                    bxc = A.ratio_r(G["dext"][..., 0], G["dgt"][..., 0], Mc & big0, rv)
                row = {}
                for law in ["flat_a", "flat_d", "ramp_a", "ramp_d"]:
                    ul, uv, Il, Iv = B.LT[law], B.LT["flat_v"], B.I[law], B.I["flat_v"]

                    def rho(arr):          # 이득 비 ρ = β(law) / β(flat_v) — 곱셈꼴 속도 이득을 지운다. GT 1, 연장 = β_ext(law)
                        return (np.nanmean(arr[ul]) / np.nanmean(arr[uv]),
                                np.nanmean(arr[ul][Il], 1) / np.nanmean(arr[uv][Iv], 1))
                    row[law] = dict(k=B.est_diff(ek, law, ek, "flat_v"), ext_pred=B.est_diff(ex, law, ex, "flat_v"),
                                    copy_pred=B.est_diff(cp, law, cp, "flat_v"), rho=ci_(*rho(bk)), rho_ext=ci_(*rho(bx)))
                    if t != "release":
                        pt, bs = cstat(dict(k=ekc, R=eRc, x=exc), law)
                        (pk, sk), (pR, sR), (px_, sx) = rho(bkc), rho(bRc), rho(bxc)
                        row[law]["common"] = dict(
                            c_k=ci_(pt["k"], bs["k"]), c_R=ci_(pt["R"], bs["R"]), c_ext=ci_(pt["x"], bs["x"]),
                            d_c=ci_(pt["k"] - pt["R"], bs["k"] - bs["R"]),
                            D_gt=ci_(abs(pt["k"]) - abs(pt["R"]), np.abs(bs["k"]) - np.abs(bs["R"])),
                            D_ext=ci_(abs(pt["k"] - pt["x"]) - abs(pt["R"] - pt["x"]), np.abs(bs["k"] - bs["x"]) - np.abs(bs["R"] - bs["x"])),
                            rho_k=ci_(pk, sk), rho_R=ci_(pR, sR), rho_ext=ci_(px_, sx), d_rho=ci_(pk - pR, sk - sR),
                            D_gt_rho=ci_(abs(pk - 1) - abs(pR - 1), np.abs(sk - 1) - np.abs(sR - 1)),
                            D_ext_rho=ci_(abs(pk - px_) - abs(pR - px_), np.abs(sk - sx) - np.abs(sR - sx)))
                cc[rk] = row
            con[t] = cc
        out["contrast"][rn] = con

        # ---------------- 요약 (학습 곡선용)
        sm = {}
        e01 = [0, 1]
        for t in tags:
            Rk = RD[rn][t]; Mk = Rk["loc"] & rd; Mc = Mk & (Rr["loc"] & rd)
            s_ = {}
            for nm, (items, _) in SUMMARY.items():
                for rk in RANGES:
                    rv = RANGES[rk]
                    U_ = len(D["UT"])
                    V = {k_: np.full(U_, np.nan) for k_ in ("own", "copy", "ext", "oc", "coc", "kc", "Rc", "cc", "xc", "koc", "Roc", "ccoc")}
                    for law, a in items:
                        u = B.LT[law]
                        V["own"][u] = A.mean_r(Rk["e"][..., a], Mk, rv)[u]
                        V["copy"][u] = A.mean_r(G["copy"][..., a], Mk, rv)[u]
                        V["ext"][u] = A.mean_r(G["ext"][..., a], Mk, rv)[u]
                        # 치우침 보정 (oc): 같은 궤적의 t0–1 (아직 거의 안 움직인 칸) 값을 뺀다 — 자의 장면별 고정 y 치우침 (p@h) 을 지운다
                        # 2026-09-25 정정 (적대 검증): t0–1 치우침은 **화면 좌표** 로 빼고 그 뒤에 late 부호를 곱한다.
                        # 이전 식 e(rv) − e(t0–1) 은 s_y 가 t0–1 에 −1 (arc 꼭대기 절반 · ledge 전부) 이고 뒤에 +1 이라 치우침을 두 배로 더했다.
                        sk_ = A.mean_r(G["s"][..., a], Mk, rv)
                        V["oc"][u] = (A.mean_r(Rk["e"][..., a], Mk, rv) - A.mean_r(Rk["raw"][..., a], Mk, e01) * sk_)[u]
                        V["coc"][u] = (A.mean_r(G["copy"][..., a], Mk, rv) - A.mean_r(G["copy_raw"][..., a], Mk, e01) * sk_)[u]
                        if t != "release":
                            V["kc"][u] = A.mean_r(Rk["e"][..., a], Mc, rv)[u]; V["Rc"][u] = A.mean_r(Rr["e"][..., a], Mc, rv)[u]
                            V["cc"][u] = A.mean_r(G["copy"][..., a], Mc, rv)[u]; V["xc"][u] = A.mean_r(G["ext"][..., a], Mc, rv)[u]
                            sc_ = A.mean_r(G["s"][..., a], Mc, rv)
                            V["koc"][u] = V["kc"][u] - (A.mean_r(Rk["raw"][..., a], Mc, e01) * sc_)[u]
                            V["Roc"][u] = V["Rc"][u] - (A.mean_r(Rr["raw"][..., a], Mc, e01) * sc_)[u]
                            V["ccoc"][u] = V["cc"][u] - (A.mean_r(G["copy_raw"][..., a], Mc, e01) * sc_)[u]
                    laws = [l for l, _ in items]
                    d = dict(e=B.est(V["own"], laws), copy=B.est(V["copy"], laws), ext=B.est(V["ext"], laws),
                             e_oc=B.est(V["oc"], laws), copy_oc=B.est(V["coc"], laws), d_e=None, d_e_oc=None)
                    if t != "release":
                        k_, R_, c_, x_ = V["kc"], V["Rc"], V["cc"], V["xc"]
                        d.update(d_e=B.est(k_ - R_, laws), D_gt=B.est(np.abs(k_) - np.abs(R_), laws),
                                 D_ext=B.est(np.abs(k_ - x_) - np.abs(R_ - x_), laws), D_copy=B.est(np.abs(k_ - c_) - np.abs(R_ - c_), laws),
                                 d_e_oc=B.est(V["koc"] - V["Roc"], laws), D_gt_oc=B.est(np.abs(V["koc"]) - np.abs(V["Roc"]), laws),
                                 D_copy_oc=B.est(np.abs(V["koc"] - V["ccoc"]) - np.abs(V["Roc"] - V["ccoc"]), laws))
                    s_[f"{nm}|{rk}"] = d
            # 법칙 대비 요약 (flat 끼리, 같은 장면 형태)
            for rk in RANGES:
                for law in ("flat_a", "flat_d", "ramp_a", "ramp_d"):
                    q = con[t][rk][law]; cm = q.get("common") or {}
                    s_[f"{law}_minus_flat_v|{rk}"] = dict(e=q["k"], ext=q["ext_pred"], copy=q["copy_pred"], d_e=cm.get("d_c"),
                                                          D_gt=cm.get("D_gt"), D_ext=cm.get("D_ext"))
                    s_[f"{law}_rho|{rk}"] = dict(e=q["rho"], ext=q["rho_ext"], d_e=cm.get("d_rho"),
                                                 D_gt=cm.get("D_gt_rho"), D_ext=cm.get("D_ext_rho"))
            sm[t] = s_
        out["summary"][rn] = sm

        # ---------------- 도달 거리 (x 로 움직이는 자유 법칙; flat_v 가 주)
        rch = {}
        dgtx = A.mean_t(G["dgt"][..., 0], np.ones_like(rd))
        rdt = A.mean_t(rd.astype(float), np.ones_like(rd)) >= 0.5
        speed_t = A.mean_r(np.repeat(G["speed"][:, None], tp, 1), np.ones_like(rd), list(T))
        base = {}
        for t in tags:
            Rk = RD[rn][t]
            Lr = A.mean_t(Rk["loc"], rd)
            dr_un = A.mean_t(Rk["dr"][..., 0], Rk["pres"] & rd)
            dr_g = A.mean_t(Rk["dr"][..., 0], Rk["loc"] & rd)
            U = len(D["UT"])
            tcl, cen, reach, read_at, plat_un, plat_g = (np.full(U, np.nan) for _ in range(6))
            for i in range(U):
                if not np.isfinite(Lr[i]).any():
                    continue
                okt = np.where(rdt[i])[0]
                br = [j for j in okt if not (Lr[i, j] >= 0.5)]
                if br:
                    tc = br[0]; cen[i] = 0
                else:
                    tc = okt[-1] + 1; cen[i] = 1
                tcl[i] = tc
                reach[i] = dgtx[i, tc - 1] if tc > 0 else 0.0
                read_at[i] = dr_g[i, tc - 1] if tc > 0 else 0.0
                plat_un[i] = np.nanmax(dr_un[i]) if np.isfinite(dr_un[i]).any() else np.nan
                plat_g[i] = np.nanmax(dr_g[i]) if np.isfinite(dr_g[i]).any() else np.nan
            base[t] = dict(tcl=tcl, reach=reach, read_at=read_at, plat_un=plat_un, plat_g=plat_g, cen=cen)
            r = {}
            for grp, laws in (("flat_v", ["flat_v"]), ("xlaws", XLAWS)):
                g = dict(t_cliff=B.est(tcl, laws), reach_px=B.est(reach, laws, 1), read_disp_at_reach_px=B.est(read_at, laws, 1),
                         plateau_present_px=B.est(plat_un, laws, 1), plateau_loc_px=B.est(plat_g, laws, 1),
                         censored_frac=r2(np.nanmean(np.concatenate([cen[B.LT[l]] for l in laws]))))
                if t != "release":
                    g["d_reach_px"] = B.est(reach - base["release"]["reach"], laws, 1)
                    g["d_t_cliff"] = B.est(tcl - base["release"]["tcl"], laws)
                    g["d_plateau_present_px"] = B.est(plat_un - base["release"]["plat_un"], laws, 1)
                # 시간 한계 vs 거리 한계: 궤적 사이 상관 (중도절단 궤적 제외)
                u = np.concatenate([B.LT[l] for l in laws])
                ok = np.isfinite(tcl[u]) & (cen[u] == 0)
                if ok.sum() >= 6 and np.std(tcl[u][ok]) > 0:
                    a1, b1, c1 = tcl[u][ok], speed_t[u][ok], reach[u][ok]
                    rng = np.random.default_rng(SEED); bi = rng.integers(0, len(a1), (2000, len(a1)))
                    cs = [np.corrcoef(a1[j], b1[j])[0, 1] for j in bi if np.std(a1[j]) > 0 and np.std(b1[j]) > 0]
                    cr = [np.corrcoef(c1[j], b1[j])[0, 1] for j in bi if np.std(c1[j]) > 0 and np.std(b1[j]) > 0]
                    g["corr_tcliff_speed"] = [r2(np.corrcoef(a1, b1)[0, 1]), r2(np.percentile(cs, 2.5)), r2(np.percentile(cs, 97.5)), int(ok.sum())]
                    # 2026-09-25 (적대 검증): reach 는 속도에 기계적으로 비례해 |corr| 비교는 검정이 아니다 → log t_cliff 을 log 속도에
                    # 회귀한 기울기 (0 = 시간 한계, −1 = 거리 한계)
                    pos = (a1 > 0) & (b1 > 0)
                    if pos.sum() >= 6:
                        la, lb = np.log(a1[pos]), np.log(b1[pos])
                        bj = rng.integers(0, len(la), (2000, len(la)))
                        sl = [np.polyfit(lb[j], la[j], 1)[0] for j in bj if np.std(lb[j]) > 0]
                        g["loglog_tcliff_speed"] = [r2(np.polyfit(lb, la, 1)[0]), r2(np.percentile(sl, 2.5)), r2(np.percentile(sl, 97.5)), int(pos.sum())]
                    g["corr_reach_speed"] = [r2(np.corrcoef(c1, b1)[0, 1]) if np.std(c1) > 0 else None,
                                             r2(np.percentile(cr, 2.5)) if cr else None, r2(np.percentile(cr, 97.5)) if cr else None, int(ok.sum())]
                r[grp] = g
            r["flat_v_curve"] = dict(read_present=[B.est(dr_un[:, j], ["flat_v"], 1) for j in T],
                                     read_loc=[B.est(dr_g[:, j], ["flat_v"], 1) for j in T],
                                     gt=[B.est(dgtx[:, j], ["flat_v"], 1) for j in T],
                                     loc_frac=[B.est(Lr[:, j], ["flat_v"]) for j in T])
            rch[t] = r
        out["reach"][rn] = rch
    return out


# ============================================================================== sanity (release vs 정본)
def sanity(D, bias):
    ref = np.load(REF_WIN, allow_pickle=True)
    assert str(ref["decoder_fp"]) == EXPECT_FP
    pos = {v: i for i, v in enumerate(ref["video_id"])}
    j = np.array([pos[v] for v in D["ids"]])
    res = {"per_clip": {}, "recall_by_tubelet": {}, "signed_C16_P16": {}}
    for c, p in WINDOWS:
        val = D["valid"][(c, p)]
        if not val.any():
            continue
        tp = p // 2
        inf = D["inf"][:, SPLIT:SPLIT + p].reshape(D["n"], tp, 2).all(2)
        for ours, theirs, ru in (("p_release", "p", "p"), ("p@h_release", "p@h", "h"), ("z", "z", "z"), ("h", "h", "h")):
            A_ = np.asarray(D["arr"][f"C{c}_P{p}_{ours}"])[val]; B_ = ref[f"C{c}_P{p}_{theirs}"][j][val]
            dpx = np.linalg.norm((A_[..., :2] - B_[..., :2]) * RES, axis=-1)
            sa, sb = A_[..., 2] > THR[ru], B_[..., 2] > THR[ru]
            iv = inf[val]
            ra = [100 * sa[iv[:, t], t].mean() for t in range(tp)]
            rb = [100 * sb[iv[:, t], t].mean() for t in range(tp)]
            res["per_clip"][f"C{c}_P{p}/{ours}"] = dict(n_clip=int(val.sum()), pos_diff_median_px=r2(np.median(dpx)),
                                                         pos_diff_p99_px=r2(np.percentile(dpx, 99)), present_agree=r2(100 * (sa == sb).mean()),
                                                         recall_ours=r2(100 * sa[iv].mean()), recall_ref=r2(100 * sb[iv].mean()))
            res["recall_by_tubelet"][f"C{c}_P{p}/{ours}"] = dict(ours=[r2(x, 1) for x in ra], ref=[r2(x, 1) for x in rb],
                                                                  max_abs_diff_pt=r2(np.max(np.abs(np.array(ra) - np.array(rb))), 1))
    res["cross_head_printed"] = CROSS_HEAD
    # new_archive §2: C16/P32 p@h, flat_v 읽은 x 이동 (clip 평균, '있다' clip, 질량 무관) t8–t14 = 47–58 px
    c, p = 16, 32
    val = D["valid"][(c, p)]
    if val.any():
        G = geom(D, c, p)
        arr = np.asarray(D["arr"][f"C{c}_P{p}_p@h_release"], np.float64)
        xy = (arr[..., :2] + 1.0) * RES - bias["h"]; say = arr[..., 2] > THR["h"]
        dr = (xy[..., 0] - G["x0"][:, None, 0]) * G["s"][..., 0]
        k = (D["sc"] == "flat_v") & val
        res["flat_v_read_disp_C16_P32_ph"] = {f"t{t}": r2(dr[k & say[:, t], t].mean(), 1) for t in range(8, 15)}
    # 부호 오차 표 (values.json 정의: 시나리오 안 '있다' clip 평균, 질량 무관, 치우침 뺌) — C16/P16
    c, p = 16, 16
    val = D["valid"][(c, p)]
    if val.any():
        G = geom(D, c, p); tp = G["tp"]
        for ru, (ours, theirs) in (("p", ("p_release", "p")), ("h", ("p@h_release", "p@h"))):
            vals = json.loads(REF_SIGNED[ru].read_text())
            refv = {(r[2], r[3]): r[4] for r in vals if r[0] == c and r[1] == p}
            rows = {}
            for src, arr in (("ours", np.asarray(D["arr"][f"C{c}_P{p}_{ours}"])), ("ref_same_clips", ref[f"C{c}_P{p}_{theirs}"][j])):
                xy = (arr[..., :2] + 1.0) * RES - bias[ru]; say = arr[..., 2] > THR[ru]
                e = (xy - G["X"]) * G["s"]
                for law in LAWS:
                    k = (D["sc"] == law) & val
                    for a in (0, 1):
                        rows.setdefault(f"{law}|{AXN[a]}", {})[src] = [r2(e[k & say[:, t], t, a].mean(), 1) if (k & say[:, t]).any() else None for t in range(tp)]
            worst = {}
            for key, d in rows.items():
                law, ax = key.split("|")
                d["values_json"] = [None if (x[2]) else x[1] for x in refv[(law, ax)]]
                diffs = [abs(a - b) for a, b in zip(d["ours"], d["values_json"]) if a is not None and b is not None]
                diffs2 = [abs(a - b) for a, b in zip(d["ours"], d["ref_same_clips"]) if a is not None and b is not None]
                worst[key] = dict(max_abs_vs_values_json=r2(max(diffs), 1) if diffs else None,
                                  max_abs_vs_ref_same_clips=r2(max(diffs2), 1) if diffs2 else None)
            res["signed_C16_P16"][ru] = dict(rows=rows, worst=worst,
                                             max_vs_values_json=r2(max(v["max_abs_vs_values_json"] or 0 for v in worst.values()), 1),
                                             max_vs_ref_same_clips=r2(max(v["max_abs_vs_ref_same_clips"] or 0 for v in worst.values()), 1),
                                             n_clip=int(val.sum()))
    return res


# ============================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--nb", type=int, default=NB)
    ap.add_argument("--no-fig", action="store_true")
    ap.add_argument("--report-only", action="store_true", help="motion.json 에서 표 · 그림 · 본문만 다시 (계산 없음)")
    a = ap.parse_args()
    t0 = time.time()
    if a.report_only:
        out = Path(a.out)
        write_all(json.loads((out / "motion.json").read_text()), out, fig=not a.no_fig)
        print(f"→ {out} (report only, {time.time() - t0:.0f}s)")
        return
    bias = check_thresholds()
    D = load(Path(a.cache))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    B = Boot(D, a.nb, SEED)
    tags = D["tags"]
    print(f"[data] {D['source']} · tag {len(tags)} · 완료 {D['complete']} · "
          + " · ".join(f"C{c}/P{p} {int(D['valid'][(c, p)].sum())} clip" for c, p in WINDOWS), flush=True)
    R = dict(meta=dict(source=D["source"], complete=D["complete"], tags=tags, thr=THR, bias={k: v.tolist() for k, v in bias.items()},
                       mass_uniform=MASS_U, edge_px=EDGE, cell_px=CELL, nb=a.nb, seed=SEED, decoder_fp=EXPECT_FP,
                       n_clip={f"C{c}_P{p}": int(D["valid"][(c, p)].sum()) for c, p in WINDOWS}, n_traj_total=len(D["UT"]),
                       date=time.strftime("%Y-%m-%d %H:%M")),
             sanity=sanity(D, bias), windows={})
    print(f"[sanity] {time.time() - t0:.0f}s", flush=True)
    for c, p in WINDOWS:
        if not D["valid"][(c, p)].any():
            print(f"[skip] C{c}/P{p}: done clip 0", flush=True)
            continue
        R["windows"][f"C{c}_P{p}"] = window(D, c, p, bias, B, tags)
        print(f"[window] C{c}/P{p} {time.time() - t0:.0f}s", flush=True)
    (out / "motion.json").write_text(json.dumps(R, ensure_ascii=False))
    write_all(R, out, fig=not a.no_fig)
    print(f"→ {out} ({time.time() - t0:.0f}s)", flush=True)


# ============================================================================== 표 · 그림 · 본문
def f1(v, nd=1, pm=True):
    """[pt, lo, hi, n] → '+1.2 [−0.3, +2.5]'."""
    if v is None or v[0] is None:
        return "—"
    s = f"{{:{'+' if pm else ''}.{nd}f}}"
    return f"{s.format(v[0])} [{s.format(v[1])}, {s.format(v[2])}]"


def fp_(v, nd=2):
    return "—" if v is None or v[0] is None else f"{v[0]:.{nd}f}"


def mark(v):
    s = sig(v)
    return "↑" if s > 0 else ("↓" if s < 0 else "≈")


def verdict(cm):
    """공통 칸 값으로 방향 판정 (MOTION.md 규칙)."""
    if cm is None or cm["d_e"][0] is None:
        return "—"
    s = sig(cm["d_e"])
    if s == 0:
        return "≈ (Δ CI ∋ 0)"
    to = [nm for nm, k in (("GT", "D_gt"), ("연장", "D_ext"), ("복사", "D_copy")) if sig(cm[k]) < 0]
    return ("앞으로" if s > 0 else "뒤로") + (f" → {'·'.join(to)} 쪽" if to else " (세 기준 모두에서 멀어짐)")


def tagname(t):
    return t


def write_all(R, out, fig=True):
    W_ = R["windows"]
    tags = R["meta"]["tags"]
    L = []
    L += ["# v3_motion — 전 칸 표 (자동 생성, `te_analyze_v3.py`)", "",
          f"데이터 `{R['meta']['source']}` · 완료 {R['meta']['complete']} · clip {R['meta']['n_clip']} · 문턱 {R['meta']['thr']} · "
          f"치우침 {R['meta']['bias']} · NB {R['meta']['nb']} seed {R['meta']['seed']} · {R['meta']['date']}", "",
          "값 = 점추정 [95 % CI] (궤적 bootstrap). Δ = predictor − release (같은 궤적 · 공통 칸). ↑/↓ = CI 가 0 을 넘지 않음.", ""]
    # ---- sanity
    L += ["## T0. sanity — release 를 정본 (`RollOutV3/exp_results/windows/readings.npz`) 과", ""]
    L += ["| 읽기 | clip | 위치 차 중앙 (px) | p99 | '있다' 일치 % | recall 여기 / 정본 (in-frame, %) | 튜블릿 recall 최대 차 (pt) |", "|---|---|---|---|---|---|---|"]
    for k, v in R["sanity"]["per_clip"].items():
        rb = R["sanity"]["recall_by_tubelet"][k]
        L.append(f"| {k} | {v['n_clip']} | {v['pos_diff_median_px']} | {v['pos_diff_p99_px']} | {v['present_agree']} | {v['recall_ours']} / {v['recall_ref']} | {rb['max_abs_diff_pt']} |")
    L += ["", "튜블릿별 recall (in-frame 칸, %, 전 8 법칙 clip 평균 = CROSS_HEAD.md 정의):", ""]
    for k, rb in R["sanity"]["recall_by_tubelet"].items():
        L.append(f"- `{k}` 여기 {rb['ours']} · 정본 {rb['ref']}")
    for ru, d in R["sanity"].get("signed_C16_P16", {}).items():
        L += ["", f"부호 오차 C16/P16 ({'p 자' if ru == 'p' else 'h 자 (p@h)'}; values.json 정의 = 시나리오 안 '있다' clip 평균, 질량 무관): "
              f"최대 |차| vs values.json **{d['max_vs_values_json']} px** · vs 같은 clip 의 정본 readings {d['max_vs_ref_same_clips']} px (clip {d['n_clip']})", "",
              "| 법칙·축 | 여기 t0…t7 | values.json | 최대 |차| |", "|---|---|---|---|"]
        for key, row in d["rows"].items():
            L.append(f"| {key} | {row['ours']} | {row['values_json']} | {d['worst'][key]['max_abs_vs_values_json']} |")
    # ---- 자 검사
    for wk, w in W_.items():
        L += ["", f"## T1-{wk}. 자 검사 (predictor 무관) — 자유 6 법칙, 읽을 수 있는 칸", "",
              "| 자 | " + " | ".join(f"t{t}" for t in range(w["tp"])) + " | T50 |", "|---|" + "---|" * (w["tp"] + 1)]
        for nm, d in w["ruler_check"].items():
            L.append(f"| {nm} recall | " + " | ".join(fp_(v) for v in d["recall_free6"]) + f" | {d['T50']} |")
            L.append(f"| {nm} '모임' | " + " | ".join(fp_(v) for v in d["loc_free6"]) + " | |")
            L.append(f"| {nm} L2 px | " + " | ".join(fp_(v, 1) for v in d["L2_free6"]) + " | |")
    # ---- 지속
    for wk, w in W_.items():
        for rn in READS:
            L += ["", f"## T2-{wk}-{rn}. 지속 — {'p@h (주)' if rn == 'ph' else 'p@p (보조)'} · 자유 6 법칙 층화 · 읽을 수 있는 칸", "",
                  "| predictor | 판독 | " + " | ".join(w["ranges"]) + " | Δ " + " | Δ ".join(w["ranges"]) + " | T50 [CI] (중도절단) | ΔT50 [CI] | 궤적 T50 중앙 |",
                  "|---|---|" + "---|" * (2 * len(w["ranges"]) + 3)]
            for t in tags:
                for nm in ("recall", "loc"):
                    d = w["persistence"][rn][t][nm]
                    L.append(f"| {t} | {'있다' if nm == 'recall' else '모임'} | " + " | ".join(f1(d["ranges"][rk], 2, False) for rk in w["ranges"]) + " | "
                             + " | ".join(("—" if t == "release" else f1(d["d_ranges"][rk], 2)) for rk in w["ranges"])
                             + f" | {d['T50']} [{d['T50_ci'][0]}, {d['T50_ci'][1]}] ({d['T50_censored_frac']}) | "
                             + ("—" if t == "release" else f"{d['dT50'][0]:+d} [{d['dT50'][1]}, {d['dT50'][2]}]") + f" | {d['traj_T50_median']} |")
            L += ["", "튜블릿별 (있다 / 모임):", ""]
            for t in tags:
                d = w["persistence"][rn][t]
                L.append(f"- `{t}` 있다 " + " ".join(fp_(v) for v in d["recall"]["curve"]) + " · 모임 " + " ".join(fp_(v) for v in d["loc"]["curve"])
                         + " · 질량 " + " ".join(fp_(v, 3) for v in d["mass_curve"]))
    # ---- 전이
    for wk, w in W_.items():
        for rn in READS:
            for rk in w["ranges"]:
                L += ["", f"## T3-{wk}-{rn}-{rk}. 전이 — {'p@h' if rn == 'ph' else 'p@p'} · {rk} · '모임' 칸", "",
                      "e = 자기 '모임' 칸의 부호 오차 (px, + 앞섬) · 복사 / 연장 = 같은 칸의 GT 기준선 · β = 이득 (0 복사, 1 GT) · "
                      "Δ 이하 = release 와 공통 '모임' 칸 · D_X = |e−X| 의 변화 (− = X 에 가까워짐) · 간격 = |연장−GT| / |복사−GT| (공통 칸)", "",
                      "| 법칙·축 | predictor | e | 복사 | 연장 | β | β_연장 | Δe | D_GT | D_연장 | D_복사 | 판정 | 간격 연장/복사 | Δe ('있다' 만) | 변위 귀무 (own−null) | 칸 |",
                      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
                for key in [f"{l}|{AXN[a]}" for l, a in KEYAX + [("arc", 0), ("ledge", 0), ("flat_v", 1), ("wall", 1)]]:   # 나머지 축은 motion.json
                    for t in tags:
                        y = w["transition"][rn][t][key]["ranges"][rk]
                        cm = y.get("common"); po = y.get("common_present_only")
                        L.append(f"| {key} | {t} | {f1(y['e'])} | {fp_(y['copy'], 1)} | {fp_(y['ext'], 1)} | {f1(y['beta'], 2)} | {fp_(y['beta_ext'])} | "
                                 + (f"{f1(cm['d_e'])} | {f1(cm['D_gt'])} | {f1(cm['D_ext'])} | {f1(cm['D_copy'])} | {verdict(cm)} | {cm['sep_ext_gt']} / {cm['sep_copy_gt']} | {f1(po['d_e'])} | "
                                    if cm else "— | — | — | — | — | — | — | ")
                                 + f"{f1(y['own_minus_null_disp'])} | {y['n_cells']} |")
    # ---- 법칙 대비
    for wk, w in W_.items():
        for rn in READS:
            L += ["", f"## T4-{wk}-{rn}. 법칙 대비 (law − flat_v, x) · 이득 비 ρ = β(law)/β(flat_v) — 공통 치우침 · 곱셈꼴 속도 이득을 뺀다", "",
                  "대비: GT 예측 0 · 연장 예측 = 연장 기준선의 같은 대비 · 복사 예측 ≈ 0 (법칙마다 이동량이 비슷해서). ρ: GT 1 · 연장 = β_ext(law). "
                  "Δ · D 는 release 와 공통 '모임' 칸 (D < 0 = 그 기준에 가까워짐).", "",
                  "| predictor | 범위 | 법칙 | 대비 | 연장 예측 | Δ대비 | D_GT | D_연장 | ρ | ρ 연장 | Δρ | D_GT(ρ) | D_연장(ρ) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for t in tags:
                for rk in w["ranges"]:
                    for l in ["flat_a", "flat_d", "ramp_a", "ramp_d"]:
                        q = w["contrast"][rn][t][rk][l]; cm = q.get("common") or {}
                        L.append(f"| {t} | {rk} | {l} | {f1(q['k'])} | {fp_(q['ext_pred'], 1)} | {f1(cm.get('d_c'))} | {f1(cm.get('D_gt'))} | {f1(cm.get('D_ext'))} | "
                                 f"{f1(q['rho'], 2)} | {fp_(q['rho_ext'])} | {f1(cm.get('d_rho'), 2)} | {f1(cm.get('D_gt_rho'), 2)} | {f1(cm.get('D_ext_rho'), 2)} |")
    # ---- 도달
    for wk, w in W_.items():
        for rn in READS:
            L += ["", f"## T5-{wk}-{rn}. 도달 거리 — flat_v (14 궤적) · x 자유 6 법칙 (84)", "",
                  "| predictor | 묶음 | t_cliff | reach (px) | 그때 읽은 이동 (px) | plateau '있다' (px) | plateau '모임' (px) | 중도절단 | Δ reach | Δ t_cliff | corr(t_cliff, 속도) | corr(reach, 속도) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for t in tags:
                for grp in ("flat_v", "xlaws"):
                    g = w["reach"][rn][t][grp]
                    L.append(f"| {t} | {grp} | {f1(g['t_cliff'], 2, False)} | {f1(g['reach_px'], 1, False)} | {f1(g['read_disp_at_reach_px'], 1, False)} | "
                             f"{f1(g['plateau_present_px'], 1, False)} | {f1(g['plateau_loc_px'], 1, False)} | {g['censored_frac']} | {f1(g.get('d_reach_px'))} | "
                             f"{f1(g.get('d_t_cliff'), 2)} | {f1(g.get('corr_tcliff_speed'), 2)} | {f1(g.get('corr_reach_speed'), 2)} |")
    # ---- 요약 (학습 곡선)
    for wk, w in W_.items():
        L += ["", f"## T6-{wk}. 학습 곡선 요약 (p@h, 자기 '모임' 칸의 e; Δ = 공통 칸)", "",
              "| predictor | " + " | ".join(f"{nm} {rk}" for nm in SUMMARY for rk in w["ranges"]) + " |", "|---|" + "---|" * (len(SUMMARY) * len(w["ranges"]))]
        for t in tags:
            cells = []
            for nm in SUMMARY:
                for rk in w["ranges"]:
                    s = w["summary"]["ph"][t][f"{nm}|{rk}"]
                    cells.append(fp_(s["e"], 1) + ("" if t == "release" else f" (Δ {f1(s['d_e'])})"))
            L.append(f"| {t} | " + " | ".join(cells) + " |")
    (out / "TABLES.md").write_text("\n".join(L) + "\n")
    if fig:
        figures(R, out)
    report(R, out)


ITEMS = [   # (라벨, 요약 키, 값 키, Δ 키, {기준 이름: D 키})
    ("flat_v x 오차 (GT = 연장 = 0)", "common_lag|t4-7", "e", "d_e", {"GT": "D_gt"}),
    ("ρ flat_a/flat_v (GT 1)", "flat_a_rho|t4-7", "e", "d_e", {"GT": "D_gt", "연장": "D_ext"}),
    ("ρ flat_d/flat_v (GT 1)", "flat_d_rho|t4-7", "e", "d_e", {"GT": "D_gt", "연장": "D_ext"}),
    ("flat_a − flat_v (px, GT 0)", "flat_a_minus_flat_v|t4-7", "e", "d_e", {"GT": "D_gt", "연장": "D_ext"}),
    ("flat_d − flat_v (px, GT 0)", "flat_d_minus_flat_v|t4-7", "e", "d_e", {"GT": "D_gt", "연장": "D_ext"}),
    ("arc y 낙하, raw (px, GT 0)", "arc_fall|t4-7", "e", "d_e", {"GT": "D_gt", "연장": "D_ext", "복사": "D_copy"}),
    ("arc y 낙하, 보정 (px, GT 0)", "arc_fall|t4-7", "e_oc", "d_e_oc", {"GT": "D_gt_oc", "복사≈연장": "D_copy_oc"}),
    ("ledge y 낙하, 보정 (px, 사건)", "ledge_fall|t4-7", "e_oc", "d_e_oc", {"GT": "D_gt_oc", "복사≈연장": "D_copy_oc"}),
    ("wall x 넘어감 (px, 사건)", "wall_over|t4-7", "e", "d_e", {"GT": "D_gt", "연장": "D_ext", "복사": "D_copy"}),
]


RHO_ROWS = []


def narrative(R):
    """읽기 · 단서 · 도메인 메모. 수치는 전부 R 에서 꺼낸다 (손으로 옮기지 않는다)."""
    W16, W32 = R["windows"].get("C16_P16"), R["windows"].get("C16_P32")
    tags = R["meta"]["tags"]; fin = [t for t in FINAL if t in tags]; oth = [t for t in fin if t != "release"]

    def v(x, nd=1):
        return "—" if x is None or x[0] is None else f"{x[0]:.{nd}f}"

    def vs(x, nd=1):
        return "—" if x is None or x[0] is None else f"{x[0]:+.{nd}f}"

    def sm(w, rn, t, k):
        return w["summary"][rn][t][k]

    def P(w, rn, t, nm, rk, key="ranges"):
        return w["persistence"][rn][t][nm][key][rk]

    L = ["## 읽기", ""]
    if W16 and W32:
        m16 = {t: W16["persistence"]["ph"][t]["mass_ranges"]["t4-7"] for t in fin}
        m32 = {t: W32["persistence"]["ph"][t]["mass_ranges"]["t8-15"] for t in fin}
        L += ["**1. 지속 — 세 읽기 (p@h '모임' · p@p '모임' · 문턱 없는 진실 칸 질량, p@h 자) 가 같은 방향일 때만 믿는다.** 지평마다 따로 (P16 t4–7 = 학습 창 안, P32 t8–15 = v11 · Predictor_v1 · 자의 학습 창 밖; IntPhys1 · IntPhys2 post-FT 에는 (일부) 안). "
              "세 읽기는 모두 '모임' 또는 질량이다 — README 정의 ('있다' 비율) 는 표 1 마지막 다섯 열.", ""]
        for t in oth:
            segs = []
            for hz, w_, rk in (("P16 t4–7", W16, "t4-7"), ("P32 t8–15", W32, "t8-15")):
                dl = [P(w_, "ph", t, "loc", rk, "d_ranges"), P(w_, "pp", t, "loc", rk, "d_ranges"), w_["persistence"]["ph"][t]["d_mass_ranges"][rk]]
                sg_ = [sig(x) for x in dl]
                mk = w_["persistence"]["ph"][t]["mass_ranges"][rk][0]
                drop = [nm for nm, g_ in (("p@h", sg_[0]), ("p@p", sg_[1])) if g_ < 0]
                if sg_[0] == sg_[1] == sg_[2] != 0:
                    verdict_ = "**같은 방향 (" + ("늘어남" if sg_[0] > 0 else "줄어듦") + ")**"
                elif sg_[2] > 0 and drop and mk is not None and mk >= 2 * MASS_U:
                    verdict_ = (f"질량은 오르는데 {'·'.join(drop)} '모임' 이 준다 → 판정 못 함 (문턱 이동일 수 있으나 v3 에 빈 장면이 없어 "
                                f"바닥 · AUROC 로 확인할 수 없다; 질량 {mk:.3f})")
                elif mk is not None and mk < 2 * MASS_U:
                    verdict_ = f"질량이 균등 근처 ({mk:.3f}) 라 문턱 없는 읽기로는 판정 못 함 — '모임' 두 자만 본다"
                else:
                    verdict_ = "읽기에 따라 다르다"

                segs.append(f"{hz}: Δ '모임' p@h {vs(dl[0], 2)} · p@p {vs(dl[1], 2)} · 질량 {vs(dl[2], 3)} (release 질량 {v(w_['persistence']['ph']['release']['mass_ranges'][rk], 3)}) → {verdict_}")
            L.append(f"- `{t}` — " + "; ".join(segs) + ".")
        L.append("- P32 t8–15 의 release 질량은 균등 (0.035) 에 가깝다 — 거기서 질량이 오르는 predictor 는 창 뒤쪽에서도 attention 을 진실 칸에 모은다는 뜻이다.")
        rr_ = {t: (W32["persistence"]["ph"][t]["recall"]["d_ranges"]["t8-15"], W32["persistence"]["pp"][t]["recall"]["d_ranges"]["t8-15"]) for t in oth}
        L.append("- **README 정의 ('있다' 비율, 질량 무관) 로 다시 재면 (2026-09-25)** P32 t8–15 Δ p@h / p@p: " + " · ".join(f"{t} {vs(rr_[t][0], 2)} / {vs(rr_[t][1], 2)}" for t in oth)
                 + f". 두 정의 × 두 자 모두 같은 방향인 것은 ↑ {', '.join(PERS_ROB.get('up2', [])) or '없음'} · ↓ {', '.join(PERS_ROB.get('dn2', [])) or '없음'} 뿐이다 — "
                 "'모임' 과 질량이 오르는데 '있다' 가 주는 predictor (예: ariel_ep43) 는 '물체를 더 오래 둔다' 로 적지 않는다.")
        L.append("")
    if W16:
        L += ["**2. 전이 — 학습 뒤 predictor 는 릴리즈보다 앞서 읽히고 (기울기 ≈ 1 + 오프셋), 가속·감속을 넣는지는 이 자료로 가를 수 없다.**", ""]
        sigs_ = [("가속 → 뒤처짐 (flat_a − flat_v < 0)", "flat_a_minus_flat_v|t4-7", "e", -1), ("감속 → 앞섬 (flat_d − flat_v > 0)", "flat_d_minus_flat_v|t4-7", "e", 1),
                 ("포물선 → 덜 떨어짐 (arc y raw < 0)", "arc_fall|t4-7", "e", -1), ("낙하 → 덜 떨어짐 (ledge y 보정 < 0)", "ledge_fall|t4-7", "e_oc", -1),
                 ("벽 → 넘어감 (wall x > 0)", "wall_over|t4-7", "e", 1)]
        segs = []
        for nm_, k_, vk_, want in sigs_:
            a_, b_ = sm(W16, "ph", "release", k_)[vk_], sm(W16, "pp", "release", k_)[vk_]
            segs.append(f"{nm_}: p@h {f1(a_)} {'재현' if sig(a_) == want else '안 됨'} · p@p {f1(b_)} {'재현' if sig(b_) == want else '안 됨'}")
        L.append("- **release 의 알려진 서명 (new_archive §4, p 자로 읽은 것) 이 자마다 재현되나** (C16/P16 t4–7, CI 가 기대 방향으로 0 을 넘으면 '재현'): " + " ; ".join(segs)
                 + ". p 자로는 서명이 대부분 재현되고, h 자로는 일부만 — 서명 자체가 자에 걸려 있다 (audit `_audit/v3/AUDIT.md` (c1) 과 같은 방향). 아래 Δ 는 이 차이를 안고 읽는다.")
        b = {t: W16["transition"]["ph"][t]["flat_v|x"]["ranges"]["t4-7"]["beta"] for t in fin}
        bp = {t: W16["transition"]["pp"][t]["flat_v|x"]["ranges"]["t4-7"]["beta"] for t in fin}
        on = {t: (W16["transition"]["ph"][t]["flat_v|x"]["ranges"]["t4-7"]["own_minus_null_disp"],
                  W16["transition"]["pp"][t]["flat_v|x"]["ranges"]["t4-7"]["own_minus_null_disp"]) for t in fin}
        fitv = {t: (W16["transition"]["ph"][t]["flat_v|x"]["ranges"]["t4-7"].get("fit"), W16["transition"]["pp"][t]["flat_v|x"]["ranges"]["t4-7"].get("fit")) for t in fin}
        nonspec = [t for t in fin if all(f_ and f_["slope"][1] is not None and f_["slope"][1] <= 0 for f_ in fitv[t])]
        more = [t for t in oth if b[t][1] is not None and b[t][1] > 1 and bp[t][1] is not None and bp[t][1] > 1]
        L.append(f"- 등속 flat_v 에서 release 는 GT 보다 덜 간다 (β p@h {v(b['release'], 2)} · p@p {v(bp['release'], 2)}). 두 자 모두에서 β 의 CI 가 1 위인 것 ({len(more)}/{len(oth)}): {', '.join(more)} — "
                 + " · ".join(f"{t} {v(b[t], 2)} / {v(bp[t], 2)}" for t in oth)
                 + f". 이 공통 이동이 모든 법칙의 raw Δ 를 지배한다 (`fig_delta_heatmap.png`: 한 행이 법칙을 가로질러 같은 색). "
                 f"궤적 사이 기울기의 CI 가 두 자 모두 0 을 넘지 못하는 것 (궤적 속도를 안 따름): {', '.join(nonspec) or '없음'}. "
                 "β 는 각자의 '모임' 칸 값이라 predictor 마다 칸이 다르다. (이전 판의 'v11_e10 은 문맥 속도를 따르지 않는다' 는 own − null 지표의 한계 때문이었다 — **철회**.)")
        ea = sm(W16, "ph", "release", "flat_a_rho|t4-7")["ext"][0]; ed = sm(W16, "ph", "release", "flat_d_rho|t4-7")["ext"][0]
        L.append(f"- 가속·감속은 이득 비 ρ 로 읽는다 (GT 1, 등속 연장 flat_a {ea:.2f} · flat_d {ed:.2f}). CI 가 무엇을 품나 (flat_a p@h / p@p · flat_d p@h / p@p): "
                 + " · ".join(f"`{r[0]}` {r[1]} / {r[2]} · {r[3]} / {r[4]}" for r in RHO_ROWS)
                 + ".")
        ga = [r[0] for r in RHO_ROWS if r[1] == r[2] == "GT"]; gd = [r[0] for r in RHO_ROWS if r[3] == r[4] == "GT"]
        exa = [r[0] for r in RHO_ROWS if r[1] in ("연장", "연장 너머") and r[2] in ("연장", "연장 너머")]
        exd = [r[0] for r in RHO_ROWS if r[3] in ("연장", "연장 너머") and r[4] in ("연장", "연장 너머")]
        tog = {law: [t for t in oth if sig(sm(W16, "ph", t, f"{law}_rho|t4-7").get("D_gt")) < 0 and sig(sm(W16, "pp", t, f"{law}_rho|t4-7").get("D_gt")) < 0] for law in ("flat_a", "flat_d")}
        spec = {t: tuple(bool(f_ and f_["slope"][1] is not None and f_["slope"][1] > 0) for f_ in (W16["transition"][rn_][t]["flat_d|x"]["ranges"]["t4-7"].get("fit") for rn_ in ("ph", "pp"))) for t in fin}
        L.append(f"- 두 자 모두 CI 가 GT 만 품는 것 — flat_a: {', '.join(ga) or '없음'} · flat_d: {', '.join(gd) or '없음'} (flat_d 의 이들 중 두 자 모두 궤적 사이 기울기 CI 가 0 위인 것: "
                 f"{', '.join(t for t in gd if all(spec[t])) or '없음'}). 두 자 모두 연장 (또는 너머) — flat_a: {', '.join(exa) or '없음'} · flat_d: {', '.join(exd) or '없음'}. "
                 f"release 대비 ρ 가 두 자 모두에서 GT 쪽으로 간 것 (D_GT < 0) — flat_a: {', '.join(tog['flat_a']) or '없음'} · flat_d: {', '.join(tog['flat_d']) or '없음'}. "
                 + ("**두 법칙 모두에서 GT 로 확인된 predictor 는 없다** — 단 ρ 는 더하기꼴 오프셋 때문에 1 (GT) 쪽으로 끌리고 GT–연장 간격이 CI 반폭과 비슷해 이 분류는 기술값이다 (가를 수 없음). " if not (set(ga) & set(gd)) else f"두 법칙 모두 GT: {', '.join(sorted(set(ga) & set(gd)))}. ")
                 + (f"두 법칙 · 두 자에서 등속 연장 쪽으로 모이는 것: {', '.join(t for t in exa if t in exd and t != 'release')}. " if any(t in exd and t != 'release' for t in exa) else "")
                 + ("release 는 두 자가 서로 다르게 읽는다." if (RHO_ROWS[0][1] != RHO_ROWS[0][2] or RHO_ROWS[0][3] != RHO_ROWS[0][4]) else ""))
        ca = {t: (sm(W16, "ph", t, "flat_a_minus_flat_v|t4-7"), sm(W16, "pp", t, "flat_a_minus_flat_v|t4-7")) for t in fin}
        beyond = [t for t in oth if ca[t][1]["e"][2] is not None and ca[t][1]["e"][2] < ca["release"][1]["ext"][0]]
        L.append("- px 대비 (law − flat_v, p@p) 에서 CI 가 통째로 연장 예측보다 더 뒤처진 것: "
                 + (", ".join(f"{t} {vs(ca[t][1]['e'])}" for t in beyond) or "없음") + f" (연장 예측 {vs(ca['release'][1]['ext'])} px). "
                 "px 대비는 곱셈꼴 이득 (β > 1) × 법칙 사이 문맥 속도 차 (flat_a 는 flat_v 보다 느리게 시작) 를 섞는다. ρ 는 곱셈꼴 항만 지우고 더하기꼴 오프셋은 못 지운다 (1 쪽으로 끌림) — 둘 다 기술값이다.")
        af, afp = sm(W16, "ph", "release", "arc_fall|t4-7"), sm(W16, "pp", "release", "arc_fall|t4-7")
        robust = []
        for t in oth:
            r_ = [sig(sm(W16, rn, t, "arc_fall|t4-7")[k]) for rn in ("ph", "pp") for k in ("d_e", "d_e_oc")]
            if all(x == r_[0] != 0 for x in r_):
                robust.append(f"{t} ({'더 떨어짐' if r_[0] > 0 else '덜 떨어짐'})")
        L.append(f"- 중력 (arc y): release 의 낙하 오차가 자와 보정에 따라 다르다 — raw p@h {vs(af['e'])} · p@p {vs(afp['e'])}, 보정 (2026-09-25 부호 정정) p@h {vs(af['e_oc'])} · p@p {vs(afp['e_oc'])} px "
                 f"(복사 ≈ 연장 raw {vs(af['copy'])}). 그래서 release 가 '덜 떨어진다' 는 이 자료로 판정하지 않는다. 두 자 × 두 방식 (raw · 보정) 네 곳 모두 같은 방향으로 바뀐 것: "
                 f"{', '.join(robust) or '없음'}.")
        wh, wp = sm(W16, "ph", "release", "wall_over|t4-7")["e"], sm(W16, "pp", "release", "wall_over|t4-7")["e"]
        L.append(f"- 사건 (ledge · wall) 은 09-19 정의에서 뺀 common sense 문제라 참고로만 싣는다 (표 2 아래 두 칸). wall 에서 release 의 넘어감은 p@h {f1(wh)} · p@p {f1(wp)} px 다"
                 + (" — 여기서도 자가 먼저 갈린다." if sig(wh) != sig(wp) else "."))
        L.append("")
    if W32:
        rc = W32["reach"]["ph"]["release"]; x = rc["xlaws"]
        L += ["**3. 도달 거리 — 늘어난 것은 '어디까지 진실 근처에 두나' 이지 '멈추는 자리' 가 아니다.**", "",
              f"- release (p@h): 읽은 이동의 최댓값 {v(rc['flat_v']['plateau_present_px'])} px 는 정본의 '약 3 칸 (45–58 px) 에서 멈춘 듯한 값' 과 같다. 그런데 '모임' (질량 > 균등) 은 "
              f"GT 가 {v(rc['flat_v']['reach_px'])} px 갈 때까지 이어진다 — 그 튜블릿에서 읽은 이동은 {v(rc['flat_v']['read_disp_at_reach_px'])} px 다. 질량 게이트가 약해 (균등 바로 위) "
              "'모임' 이 '물체를 제자리에 둔다' 보다 넓다. plateau 는 자 기본값이 섞인 값이라 '멈춘다' 로 읽지 않는다 (CLAUDE.md §6).",
              f"- 시간 한계인가 거리 한계인가 (release, x 자유 6 법칙 {x['corr_tcliff_speed'][3] if x.get('corr_tcliff_speed') else '—'} 궤적): corr(t_cliff, 속도) {f1(x.get('corr_tcliff_speed'), 2)}, "
              f"corr(reach, 속도) {f1(x.get('corr_reach_speed'), 2)} — "
              + (f"이 |corr| 비교는 reach 가 속도에 기계적으로 비례해 검정이 아니다 (2026-09-25 정정). log t_cliff ~ log 속도 기울기 (0 = 시간 한계, −1 = 거리 한계): "
                 f"{f1(x.get('loglog_tcliff_speed'), 2)} → "
                 + (("시간 한계 쪽이다. " if x['loglog_tcliff_speed'][0] > -0.5 else "거리 한계 쪽이다. ") if x.get('loglog_tcliff_speed') and x['loglog_tcliff_speed'][0] is not None else "판정 못 함. "))
              + "`exp_results/windows/BEHAVIOR.md` 는 같은 창에서 'T50 이 속도 3 분위 모두 t10' 이었다. RollOutV2 (09-19) 의 '거리 한계' 와는 자료 (v2 무중력 · 가림 없음 · 다른 자) · 창이 달라 "
              "한쪽을 지우지 않는다.",
              "- 학습 뒤 (표 3): " + " · ".join(f"{t} reach {v(W32['reach']['ph'][t]['flat_v']['reach_px'], 0)} px (corr(reach, 속도) {v(W32['reach']['ph'][t]['xlaws'].get('corr_reach_speed'), 2)})" for t in oth)
              + ". " + "".join(f"{t} 는 궤적 {W32['reach']['ph'][t]['flat_v']['censored_frac']:.0%} 가 창 끝까지 끊기지 않아 reach 가 하한이다. "
                               for t in oth if (W32['reach']['ph'][t]['flat_v']['censored_frac'] or 0) > 0.5), ""]
    # 학습 곡선
    if W16:
        L += ["**4. 학습 곡선 (p@h)** — release → 체크포인트 순서. '첫 체크포인트' = 첫 체크포인트가 이미 마지막 값 쪽으로 절반 넘게 간 지표 수.", ""]

        def met(t):
            out_ = {"on-GT t4–7": P(W16, "ph", t, "loc", "t4-7")[0], "flat_v 오차": sm(W16, "ph", t, "common_lag|t4-7")["e"][0],
                    "ρ flat_a": sm(W16, "ph", t, "flat_a_rho|t4-7")["e"][0], "ρ flat_d": sm(W16, "ph", t, "flat_d_rho|t4-7")["e"][0]}
            if W32:
                out_["T50 P32"] = W32["persistence"]["ph"][t]["loc"]["T50"]
                out_["reach P32"] = W32["reach"]["ph"][t]["flat_v"]["reach_px"][0]
            return out_
        fmt_ = {"on-GT t4–7": "{:.2f}", "flat_v 오차": "{:+.0f}", "ρ flat_a": "{:.2f}", "ρ flat_d": "{:.2f}", "T50 P32": "{}", "reach P32": "{:.0f}"}
        base = met("release")
        for fam, lst in FAMILIES.items():
            tg = [t for _, t in lst if t in tags and t != "release"]
            if len(tg) < 2:
                continue
            seq = [met(t) for t in tg]
            parts, early = [], 0
            for k in base:
                vals = ([base[k]] if fam != "ariel" else []) + [q[k] for q in seq]
                if any(x_ is None for x_ in vals):
                    continue
                parts.append(f"{k} " + " → ".join(fmt_[k].format(x_) for x_ in vals))
                if fam != "ariel":
                    tot = seq[-1][k] - base[k]
                    early += abs(tot) > 1e-9 and (seq[0][k] - base[k]) / tot >= 0.5
            head = f"{FAM_NAME[fam]} ({'release → ' if fam != 'ariel' else ''}{' → '.join(tg)})"
            L.append(f"- {head}: " + " · ".join(parts) + ("" if fam == "ariel" else f" — 첫 체크포인트 {early}/{len(parts)}"))
        L.append("")

    # 단서
    L += ["## 단서", ""]
    if W16:
        fy = {t: W16["transition"]["ph"][t]["flat_v|y"]["e_t"] for t in fin}
        L.append("1. **자 두 개가 release 부터 다르게 읽는다.** flat_d − flat_v (p@h / p@p) "
                 f"{vs(sm(W16, 'ph', 'release', 'flat_d_minus_flat_v|t4-7')['e'])} / {vs(sm(W16, 'pp', 'release', 'flat_d_minus_flat_v|t4-7')['e'])} px, "
                 f"arc y raw {vs(sm(W16, 'ph', 'release', 'arc_fall|t4-7')['e'])} / {vs(sm(W16, 'pp', 'release', 'arc_fall|t4-7')['e'])}, "
                 f"wall x {vs(sm(W16, 'ph', 'release', 'wall_over|t4-7')['e'])} / {vs(sm(W16, 'pp', 'release', 'wall_over|t4-7')['e'])}"
                 + (f", flat_v plateau (P32) {v(W32['reach']['ph']['release']['flat_v']['plateau_present_px'])} / {v(W32['reach']['pp']['release']['flat_v']['plateau_present_px'])} px" if W32 else "")
                 + ". p@h 는 release p 를 flat 장면에서 **위로 떠 가게** 읽는다 (flat_v y, 위 = +: t0 "
                 f"{vs(fy['release'][0])} → t7 {vs(fy['release'][-1])} px). 학습한 predictor 의 떠감: "
                 + " · ".join(f"{t} {vs(fy[t][0])} → {vs(fy[t][-1])}" for t in oth)
                 + ". 그래서 p@h 의 Δ 일부는 '움직임이 바뀌었다' 가 아니라 'h 자가 p 를 더 (덜) 잘 읽게 됐다' 일 수 있다. p@p 는 release p 로 배운 자라 학습한 p 에는 이식 판독이다. "
                 "**두 자에서 같은 방향인 것만 '견고' 로 적었다.**")
        agree = tot_ = 0
        for w in (W16,):
            for t in tags:
                if t == "release":
                    continue
                for l_, a_ in KEYAX:
                    y = w["transition"]["ph"][t][f"{l_}|{AXN[a_]}"]["ranges"]["t4-7"]
                    g_, p_ = y.get("common", {}).get("d_e"), y.get("common_present_only", {}).get("d_e")
                    if sig(g_) and sig(p_):
                        tot_ += 1; agree += sig(g_) == sig(p_)
        L.append(f"2. **'모임' 게이트는 양쪽으로 샌다.** 진실 3×3 (±1.5 칸) 질량 > 균등이라 오차가 ~27 px 를 넘는 칸이 빠지면서 부호 오차가 0 쪽으로 잘리고, "
                 "동시에 균등 바로 위라 약하다 (release reach 때 읽은 이동은 GT 이동보다 작다, 읽기 3). 민감도: '있다' 만 (질량 무관) 공통 칸으로 다시 잰 Δ 와 부호가 같은 칸 "
                 f"{agree}/{tot_} (C16/P16 t4–7, 21 tag × 8 법칙·축 중 두 Δ 모두 CI 가 0 을 넘는 칸). 단 P16 에서는 v11 밖에서 '모임' 과 '있다' 마스크가 거의 같아 (질량 게이트가 거의 안 걸린다) 이 일치는 거의 자명하다.")
        ip = "ip1_e40" if "ip1_e40" in fin else None
        if ip and W32:
            L.append(f"3. **고정 문턱 '있다' 는 자의 보정 이동을 섞는다.** v3 에는 빈 장면이 없어 바닥 · AUROC 를 못 낸다. 예: {ip} 는 P16 t4–7 '모임' 비율이 "
                     f"{v(P(W16, 'ph', ip, 'loc', 't4-7'), 2)} (release {v(P(W16, 'ph', 'release', 'loc', 't4-7'), 2)}) 로 떨어지는데 같은 칸의 진실 칸 질량은 "
                     f"{v(W16['persistence']['ph'][ip]['mass_ranges']['t4-7'], 3)} (release {v(W16['persistence']['ph']['release']['mass_ranges']['t4-7'], 3)}) 로 오른다.")
    L += ["4. **지평 (2026-09-25 정정).** 자 셋은 미래 8 튜블릿까지만 학습했다. post-FT 중 v11 · Predictor_v1 (C16 / 32 프레임) 도 미래 8 튜블릿까지라 P32 t8–15 는 이들의 밖이다. "
          "그러나 IntPhys1 post-FT (skip2_w32, 문맥 4–20 → 미래 최대 14 튜블릿) 와 IntPhys2 post-FT (48 프레임, 문맥 12–42 → 최대 18 튜블릿) 에는 t8–15 가 (일부) **학습 지평 안** 이다 — 이전 판의 'post-FT 넷 모두 밖' 은 틀렸다 "
          "(release 는 64 프레임으로 사전학습). h 자 · z 자는 같은 t8–15 의 진짜 h · z 를 0.98 로 읽으므로 '자가 창 후반에서 깨진다' 는 배제되지만, p 고유의 외삽은 배제되지 않는다.",
          "5. **ρ 와 대비의 가정.** flat 세 법칙은 t4–7 의 평균 GT 이동이 비슷하게 설계됐다 — 속도를 무시하는 predictor 도 ρ ≈ 1 을 낸다. 그래서 ρ 는 궤적 사이 기울기 (표 2 아래) 와 같이 읽는다. 또 학습한 predictor 의 앞섬은 더하기꼴 (기울기 ≈ 1 + 오프셋) 이라 ρ 가 1 (GT) 쪽으로 끌린다 — ρ 는 기술값이다. "
          "t4–7 에서 등속 연장과 GT 의 간격은 1 칸 미만이라 (flat_a · flat_d 5–8 px) 절대 위치로는 못 가르고 비율로 읽었다. ramp 는 flat_v 와 장면이 달라 TABLES.md 에만 둔다.",
          "6. **보정 (t0–1 빼기) — 2026-09-25 정정.** 이전 판은 부호 오차끼리 뺐는데 s_y 가 t0–1 에 −1 (arc 꼭대기 절반 · ledge 전부) 이고 떨어지는 튜블릿에 +1 이라 자 치우침을 빼지 않고 두 배로 더했다. "
          "지금은 t0–1 의 치우침을 화면 좌표로 빼고 late 부호를 곱한다. 가정은 t0–1 의 오차가 순수한 자 치우침이라는 것. 초반 오차가 요동하는 predictor (IntPhys1 post-FT 의 arc y t0–1) 에서는 틀린다 — raw 와 같이 적었다.",
          "7. **표본.** 법칙마다 궤적 14 개 (외형 복사본은 독립 표본이 아니다), 학습 run 은 seed 하나씩 — 학습 분산은 모른다. 한 칸에 predictor 22 개 × 여러 요약을 동시에 보므로 CI 하나가 0 을 겨우 넘는 것은 다중 비교로 읽는다.",
          "8. **predictor 마다의 교란.** Ariel 은 attention 규칙 · 데이터 · scratch · epoch 네 가지가 동시에 다르다. v11 post-FT 는 운동 조건마다 궤적 2 개를 외운다. IntPhys2 post-FT 는 loss 0.605 → 0.585 로 거의 안 움직였다 (`MODELS.md`).",
          "9. 원인은 목적함수로 돌리지 않는다. 이 문서가 보는 것은 학습 데이터 · 시간 척도와 읽은 행동의 대응까지다.",
          "10. **ledge 의 P32 t12 이후** 는 GT 가 착지 뒤 튀어 y 이동의 부호가 바뀐다 (부호 규칙이 흔들린다; `fig_transition_curves_C16_P32.png` 의 뒤쪽 튐). 요약 (표 2b) 에서 뺐고 TABLES.md 의 그 칸은 해석하지 않는다. RollOutV3 README 는 ledge 낙하 구간 t9–t14 를 판정 불가로 적는다 — 여기서는 t12 이후만 뺐다 (미적용).",
          "11. **p@h 는 p 토큰에서 검증되지 않은 자다.** flat_v (세로 운동 없음) 에서 release p 를 19–40 px 위로 읽는다 (한 줄의 마지막 항목; RollOutV3 `RECALL_TUBELET_LIMIT` 도 p 토큰에서 p@h 위치 오차 37.5 px). "
          "검증 재계산에 따르면 '모임' 게이트는 이런 칸의 약 90 % 를 통과시킨다. 그래서 p@h 는 'predictor 무관한 자' 가 아니라 행동 지표이고, x 는 p@p 와 같은 방향인 것만 견고로 적었다.",
          "12. **T50 · reach 는 첫 교차 통계** 인데 곡선이 깜빡인다 (release p@p '모임' P32: t8 부터 0.96, 0.78, 0.05, 0.26, …). [10, 10] 같은 퇴화 CI 는 한 튜블릿 dip (t10) 에 대한 민감도를 숨긴다.",
          "13. **Ariel forward 는 수치 경로가 다르다** (`validation.json`: forward 당 SDPA q/k cast 108/204 vs 다른 predictor 0). 교란으로 남는다.",
          "14. **다중 비교** — 22 tag × 요약 ~10 × 두 자, 법칙당 궤적 14. 한 줄은 두 자 · 두 정의가 같은 방향인 것만 적었고, 단일 칸 판정 ('p@h 만 ↑') 은 표에만 둔다.", ""]
    best_pv1 = False
    if W32:
        dl_ = {t: W32["persistence"]["ph"][t]["loc"]["d_ranges"]["t8-15"][0] for t in tags if t != "release"}
        dr_ = {t: W32["reach"]["ph"][t]["flat_v"].get("d_reach_px", [None])[0] for t in tags if t != "release"}
        best_pv1 = max(dl_, key=dl_.get).startswith("pv1") and max(dr_, key=lambda k: dr_[k] if dr_[k] is not None else -1e9).startswith("pv1")
    L += ["## 도메인 메모", "",
          "- **시간 척도** — v3 는 stride 1 (30 fps): 1 튜블릿 = 2 프레임 ≈ 67 ms. v11 · Predictor_v1 post-FT 는 stride 3 (1 튜블릿 = 6 원 프레임 ≈ 200 ms), IntPhys1 post-FT 는 stride 2, "
          "Ariel 은 12 fps (1 튜블릿 ≈ 167 ms), IntPhys2 post-FT 는 frame_step 10. **어느 predictor 도 v3 의 시간 척도로 학습하지 않았다.** 같은 물리 속도면 v3 의 튜블릿당 이동은 v11 의 1/3 이다.",
          "- **v11 post-FT** — 운동 조건마다 한 속도 · 궤적 2 개. v3 flat_v 에서 읽은 이동은 궤적 속도를 따르고 (궤적 사이 기울기 ≈ 1) 고정 오프셋만큼 앞선다 (표 2 아래 적합). "
          "이전 판의 '문맥 속도를 따르지 않고 일정량을 옮긴다' 는 **철회** (own − null 지표의 한계).",
          "- **Predictor_v1 post-FT** — 무중력 등속 (다양한 속도) · 가림 k=4 · 10° 경사 (다양한 가속). v3 와 **같은 10° 쐐기 · 같은 법칙 족** 이라 (stride 만 다르다) 동역학상 근접 도메인이다 — 이득은 일반화 증거가 아니다"
          + (" — 지속 (P32 t8–15 '모임' Δ, p@h) 과 도달 거리 (Δ reach) 가 22 tag 중 가장 크게 늘었다" if best_pv1 else "")
          + ". 그래도 가속·감속 법칙의 ρ 분류 (기술값, 가를 수 없음) 로는 " + ("등속 연장 쪽이다" if "pv1_e15" in [r[0] for r in RHO_ROWS if r[1] in ("연장", "연장 너머") and r[3] in ("연장", "연장 너머")] else "GT 로 가지 않는다")
          + " — 가속을 본 것은 경사 팔뿐이고 stride 3 이다.",
          "- **IntPhys1 post-FT** — IntPhys1 은 이 문서의 도메인이 아니다 ('학습 도메인 안' 인 칸이 이 문서에는 없다).", ""]
    return L


def _refs(s, refs):
    out = []
    for nm, k in refs.items():
        g_ = sig(s.get(k))
        if g_:
            out.append(f"{nm} {'가까이' if g_ < 0 else '멀리'}")
    return " · ".join(out) if out else "≈"


def _two(a, b):
    sa, sb = sig(a), sig(b)
    if sa and sb:
        return f"**견고 {'↑' if sa > 0 else '↓'}**" if sa == sb else "자에 따라 반대"
    if sa or sb:
        return f"p@{'h' if sa else 'p'} 만 {'↑' if (sa or sb) > 0 else '↓'}"
    return "≈"


def _cls(v, gt, ext):
    """CI 안에 GT · 연장 중 무엇이 드나."""
    if v is None or v[0] is None:
        return "—"
    ig, ie = v[1] <= gt <= v[2], v[1] <= ext <= v[2]
    if ig and ie:
        return "둘 다 (못 가름)"
    if ig:
        return "GT"
    if ie:
        return "연장"
    lo_side = v[2] < min(gt, ext) or v[1] > max(gt, ext)
    return "연장 너머" if lo_side and abs(v[0] - ext) < abs(v[0] - gt) else ("GT 너머" if lo_side else "사이")


PERS_ROB = {}


def correction(R):
    """2026-09-25 적대 검증 정정 블록 — 이전 값은 이전 판 (MOTION.md 04:10) 의 인쇄값 (상수), 지금 값은 R 에서."""
    W16, W32 = R["windows"].get("C16_P16"), R["windows"].get("C16_P32")
    tags = R["meta"]["tags"]; fin = [t for t in FINAL if t in tags]; oth = [t for t in fin if t != "release"]

    def g(x, nd=1, pm=True):
        return "—" if x is None or x[0] is None else (f"{x[0]:+.{nd}f}" if pm else f"{x[0]:.{nd}f}")
    if not (W16 and W32):
        return []
    sm = lambda rn, t, k: W16["summary"][rn][t][k]
    two = lambda a, b: _two(a, b).replace("**", "")
    # 지속 (두 정의)
    up2, dn2, rest = [], [], []
    for t in oth:
        a = _two(W32["persistence"]["ph"][t]["loc"]["d_ranges"]["t8-15"], W32["persistence"]["pp"][t]["loc"]["d_ranges"]["t8-15"])
        b = _two(W32["persistence"]["ph"][t]["recall"]["d_ranges"]["t8-15"], W32["persistence"]["pp"][t]["recall"]["d_ranges"]["t8-15"])
        (up2 if a.startswith("**견고 ↑") and b.startswith("**견고 ↑") else dn2 if a.startswith("**견고 ↓") and b.startswith("**견고 ↓") else rest).append(t)
    ar = W32["persistence"]
    arl = "ariel_ep43" if "ariel_ep43" in tags else None
    a_txt = (f" ariel_ep43 은 '있다' t8–15 Δ p@h {g(ar['ph'][arl]['recall']['d_ranges']['t8-15'], 2)} · p@p {g(ar['pp'][arl]['recall']['d_ranges']['t8-15'], 2)}, "
             f"'있다' T50 p@h {ar['ph']['release']['recall']['T50']}→{ar['ph'][arl]['recall']['T50']} · p@p {ar['pp']['release']['recall']['T50']}→{ar['pp'][arl]['recall']['T50']} (정의에 따라 다름).") if arl else ""
    af = {rn: {t: sm(rn, t, "arc_fall|t4-7") for t in fin} for rn in ("ph", "pp")}
    lf = {rn: {t: sm(rn, t, "ledge_fall|t4-7") for t in fin} for rn in ("ph", "pp")}
    grob = [t for t in oth if len({sig(sm(rn, t, 'arc_fall|t4-7')[k]) for rn in ('ph', 'pp') for k in ('d_e', 'd_e_oc')}) == 1 and sig(af['ph'][t]['d_e_oc']) != 0]
    fv = {rn: W16["transition"][rn]["v11_e10"]["flat_v|x"]["ranges"]["t4-7"].get("fit") for rn in ("ph", "pp")} if "v11_e10" in tags else None

    def fs(f):
        return "—" if not f or f["slope"][0] is None else f"기울기 {f['slope'][0]:.2f} [{f['slope'][1]:.2f}, {f['slope'][2]:.2f}], 절편 {f['intercept'][0]:+.0f} px"
    ll = W32["reach"]["ph"]["release"]["xlaws"].get("loglog_tcliff_speed"); llp = W32["reach"]["pp"]["release"]["xlaws"].get("loglog_tcliff_speed")
    fy = W16["transition"]["ph"]["release"]["flat_v|y"]["e_t"]
    hy = [q[0] for q in W16["ruler_check"]["h@h"].get("flat_v_y_e", []) if q and q[0] is not None]
    L = ["> ⚠️ **정정 (2026-09-25, 적대 검증 반영)** — 이 판은 적대 검증의 지적을 스크립트에 넣고 다시 계산한 것이다. 검증 원문: `../Archive/verify_raw/verify_v3.json`. "
         "이전 판 (같은 날 04:10) 의 문장 → 지금 문장:",
         ">",
         f"> 1. **지속 '견고'** — 이전 \"두 자에서 같은 방향으로 늘어난 것: pv1_e15, ariel_ep43\" ('모임' 한 정의) → 지금 '모임' 과 README 정의 ('있다' 비율) 두 정의 × 두 자 모두 같은 방향만 견고: "
         f"↑ {', '.join(up2) or '없음'} · ↓ {', '.join(dn2) or '없음'} · 정의·자에 따라 다름 {', '.join(rest) or '없음'}.{a_txt}",
         f"> 2. **arc · ledge '보정' 부호 버그** — 이전 판은 부호 오차 e(t4–7) − e(t0–1) 를 썼는데 s_y 가 t0–1 에 −1 (arc 꼭대기 절반 · ledge 전부), 떨어질 때 +1 이라 자 치우침을 두 배로 더했다. "
         f"이전 release arc −9.5 / −22.7 · ledge −31.2 / −19.6 (p@h / p@p) → 지금 (t0–1 치우침을 화면 좌표로 빼고 late 부호) arc {g(af['ph']['release']['e_oc'])} / {g(af['pp']['release']['e_oc'])} · "
         f"ledge {g(lf['ph']['release']['e_oc'])} / {g(lf['pp']['release']['e_oc'])}. 네 곳 (두 자 × raw · 보정) 같은 방향: 이전 ip2_e80 (더 떨어짐) · pv1_e15 (덜 떨어짐) → 지금 "
         + ", ".join(f"{t} ({'더' if sig(af['ph'][t]['d_e_oc']) > 0 else '덜'} 떨어짐; 보정 Δ p@h {f1(af['ph'][t]['d_e_oc'])} · p@p {f1(af['pp'][t]['d_e_oc'])})" for t in grob)
         + (" — 검증 원문은 'pv1_e15 하나' 라고 적었으나 이 재계산에서는 ip1_e40 도 네 곳 같은 방향이다. 단 ip1_e40 의 보정 Δ 는 작고 IntPhys1 post-FT 는 t0–1 이 요동해 보정 가정이 약하다 (단서 6). " if "ip1_e40" in grob else ". ")
         + "보정 Δ 는 t0–1 에도 공통 '모임' 칸이 있는 궤적 (arc 7/14) 에서만 잰다. "
         "ledge 공통 칸 보정 Δ: 이전 pv1_e15 p@h +26.2 · ip1_e40 p@p −19.0 · ip2_e80 p@p +12.4 · ariel_ep43 p@h +3.9 → 지금 "
         + " · ".join(f"{t} {rn_} {g(lf[k_][t].get('d_e_oc'))}" for t, rn_, k_ in (("pv1_e15", "p@h", "ph"), ("ip1_e40", "p@p", "pp"), ("ip2_e80", "p@p", "pp"), ("ariel_ep43", "p@h", "ph")) if t in tags) + ".",
         (f"> 3. **'v11_e10 은 문맥 속도를 따르지 않는다' 철회** — own − null 변위 지표는 공통 오프셋이 GT 이동의 퍼짐보다 크면 추적을 못 잡는다. flat_v 14 궤적 사이 적합 (t4–7): "
          f"p@h {fs(fv['ph'])} · p@p {fs(fv['pp'])} → 궤적 속도를 따르고 고정 오프셋만큼 앞선다. 본문의 궤적 특이성 표를 기울기 · 절편 표로 바꿨다.") if fv else "> 3. —",
         "> 4. **ρ 판정** — 이전 \"ρ 는 공통 이동을 지운다 · 가속과 감속을 둘 다 GT 로 넣는 predictor 는 없다 · flat_d GT 만: v11 · ip2 · ariel\" → 지금 앞섬이 곱셈꼴이 아니라 더하기꼴 (기울기 ≈ 1 + 오프셋) 이라 "
         "ρ 는 1 (GT) 쪽으로 끌린다 → ρ 는 기술값, \"확인되지 않았다 / 가를 수 없다\" 로 적는다.",
         "> 5. **β · 앞섬** — 이전 \"학습한 predictor 는 모든 법칙에서 앞으로 간다\" → 지금 \"릴리즈보다 앞서 읽힌다 (공통 칸 Δ > 0)\". β 는 각자의 '모임' 칸 값 (칸이 다르다), 짝지은 것은 Δ 뿐.",
         "> 6. **학습 지평** — 이전 \"P32 t8–15 는 자 셋 · post-FT 넷의 학습 지평 밖\" → 지금 자 셋 · v11 · Predictor_v1 (미래 ≤ 8 튜블릿) 만 밖. IntPhys1 post-FT (≤ 14) · IntPhys2 post-FT (≤ 18) 에는 (일부) 안.",
         f"> 7. **시간 vs 거리 한계** — 이전 |corr(reach, 속도)| > |corr(t_cliff, 속도)| 비교 (reach 는 속도에 기계적으로 비례 → 검정 아님) → 지금 log t_cliff ~ log 속도 기울기 (0 = 시간, −1 = 거리) release p@h {g(ll, 2)} · p@p {g(llp, 2)}. 결론 (시간 쪽) 은 같다.",
         "> 8. **도메인** — 이전 \"v3 는 어느 predictor 의 학습 도메인도 아니다\" → 지금 Predictor_v1 post-FT 는 같은 10° 쐐기 · 등속 팔 · 같은 법칙 족 (stride 만 다름) 으로 동역학상 근접 도메인. pv1 의 지속 · 도달 이득은 일반화 증거가 아니다.",
         (f"> 9. **p@h 는 '주 판독 · predictor 무관한 자' 가 아니다** — p 토큰에서 검증되지 않았다. flat_v 에서 release p 를 t0 {g(fy[0])} → t7 {g(fy[-1])} px 위로 읽는다 (h 자를 h 에: |y 오차| 최대 {max(abs(x) for x in hy):.1f} px). "
          "p@h 는 행동 지표로만, x 는 p@p 와 같은 방향만 견고.") if hy else "> 9. —",
         "> 10. **'질량은 오르는데 모임이 준다 → 문턱 이동, 물체를 잃지 않았다'** → 판정 못 함 (v3 에 빈 장면이 없어 바닥 · AUROC 없음).",
         ">",
         "> **미적용**: (a) p@h 자를 p 토큰에서 검증 (라벨된 p 토큰 · 자 재학습 필요, GPU) · (b) ledge 판정 불가 구간 t9–t14 제외 (지금도 t12 이후만 제외; 한 줄은 ledge 를 쓰지 않는다) · "
         "(c) T50 · reach 를 깜빡임에 강한 통계로 바꾸기 (단서 12 에 적기만 함) · (d) Ariel SDPA 수치 경로 교란 (단서 13 에 적기만 함) · (e) 빈 장면 바닥 · AUROC (v3 에 빈 장면 없음) · (f) 다중 비교 보정.",
         ""]
    return L


def report(R, out):
    W16, W32 = R["windows"].get("C16_P16"), R["windows"].get("C16_P32")
    tags = R["meta"]["tags"]
    fin = [t for t in FINAL if t in tags]; oth = [t for t in fin if t != "release"]
    S = R["sanity"]
    L = []
    L += ["# v3 운동 — 전이 · 지속 · 도달 거리, predictor 별 · 학습 곡선", ""] + correction(R) + [
          f"> 자동 생성 (`z_research/scripts/analysis/te_analyze_v3.py`) — 손으로 고치지 않는다. {R['meta']['date']}.",
          f"> 데이터: `{R['meta']['source']}` (`te_v3_readout.py` curves 추출, 완료 {R['meta']['complete']}) — RollOut_v3 가능 clip "
          f"{R['meta']['n_clip']}, 궤적 {R['meta']['n_traj_total']} (자유 6 법칙 × 14 + ledge · wall 각 14).",
          "> **모든 수치는 추출 배열에서 이 스크립트가 다시 계산한 것**이다. 자는 릴리즈에서 학습·보정한 것 그대로, 문턱은 자마다 하나 "
          "(h 자 −0.961 · p 자 0.8307 · z 자 −0.662), 치우침 `attn_bias_px.json`. 95 % CI = 법칙 층화 궤적 bootstrap "
          f"({R['meta']['nb']} 회), 모든 predictor 가 같은 재추출 (짝지은 Δ).",
          "> 상위: [`../README.md`](../README.md) (정의 · 읽는 규칙) · [`../MODELS.md`](../MODELS.md) · 전 칸 표 [`TABLES.md`](TABLES.md) · 값 `motion.json` · 정의 전문은 스크립트 머리말.", ""]

    # ------------------------------------------------------------------ 계산해 둘 것
    def pr(w, rn, t, nm, rk):
        return w["persistence"][rn][t][nm]["ranges"][rk]

    def dpr(w, rn, t, nm, rk):
        return w["persistence"][rn][t][nm]["d_ranges"][rk]

    def sm(w, rn, t, key):
        return w["summary"][rn][t][key]

    def v1(x, nd=1):
        return "—" if x is None or x[0] is None else f"{x[0]:.{nd}f}"

    rob = {}
    if W32:
        for t in oth:
            rob[t] = _two(dpr(W32, "ph", t, "loc", "t8-15"), dpr(W32, "pp", t, "loc", "t8-15"))
    up = [t for t in oth if rob.get(t, "").startswith("**견고 ↑")]
    dn = [t for t in oth if rob.get(t, "").startswith("**견고 ↓")]
    # 2026-09-25: README 정의 ('있다' 비율) 로도 — 두 정의 × 두 자 모두 같은 방향만 견고
    rob_rec = {t: _two(dpr(W32, "ph", t, "recall", "t8-15"), dpr(W32, "pp", t, "recall", "t8-15")) for t in oth} if W32 else {}
    up2 = [t for t in up if rob_rec.get(t, "").startswith("**견고 ↑")]
    dn2 = [t for t in dn if rob_rec.get(t, "").startswith("**견고 ↓")]
    PERS_ROB.clear(); PERS_ROB.update(up=up, dn=dn, up2=up2, dn2=dn2, rob=rob, rob_rec=rob_rec)

    # ------------------------------------------------------------------ 한 줄  (2026-09-25 적대 검증 반영으로 다시 씀)
    def vs1(x, nd=2):
        return "—" if x is None or x[0] is None else f"{x[0]:+.{nd}f}"

    def fs(f):
        if not f or f["slope"][0] is None:
            return "—"
        return f"기울기 {f['slope'][0]:.2f} [{f['slope'][1]:.2f}, {f['slope'][2]:.2f}] · 절편 {f['intercept'][0]:+.0f} px"
    L += ["## 한 줄", ""]
    ncl = R["meta"]["n_clip"].get("C16_P32", R["meta"]["n_clip"].get("C16_P16"))
    if W32:
        t50 = {t: W32["persistence"]["ph"][t]["loc"]["T50"] for t in fin}
        t50p = {t: W32["persistence"]["pp"][t]["loc"]["T50"] for t in fin}
        t50r = {rn: {t: W32["persistence"][rn][t]["recall"]["T50"] for t in fin} for rn in ("ph", "pp")}
        seg = []
        for t in up2 + dn2:
            seg.append(f"`{t}` {'더 오래 유지' if t in up2 else '더 일찍 잃음'} — '모임' T50 p@h {t50['release']}→{t50[t]}{' (중도절단)' if t50[t] >= 16 else ''} · p@p {t50p['release']}→{t50p[t]}, "
                       f"t8–15 '모임' p@h {v1(pr(W32, 'ph', 'release', 'loc', 't8-15'), 2)}→{v1(pr(W32, 'ph', t, 'loc', 't8-15'), 2)} · p@p {v1(pr(W32, 'pp', 'release', 'loc', 't8-15'), 2)}→{v1(pr(W32, 'pp', t, 'loc', 't8-15'), 2)}, "
                       f"'있다' 만으로 t8–15 Δ p@h {vs1(dpr(W32, 'ph', t, 'recall', 't8-15'))} · p@p {vs1(dpr(W32, 'pp', t, 'recall', 't8-15'))}")
        rest = [t for t in oth if t not in up2 + dn2]
        ar = ""
        if "ariel_ep43" in rest:
            t = "ariel_ep43"
            ar = (f" 예: ariel_ep43 은 '모임' t8–15 Δ p@h {vs1(dpr(W32, 'ph', t, 'loc', 't8-15'))} · p@p {vs1(dpr(W32, 'pp', t, 'loc', 't8-15'))} (질량 {vs1(W32['persistence']['ph'][t]['d_mass_ranges']['t8-15'], 3)}) 인데 "
                  f"'있다' Δ p@h {vs1(dpr(W32, 'ph', t, 'recall', 't8-15'))} · p@p {vs1(dpr(W32, 'pp', t, 'recall', 't8-15'))}, '있다' T50 p@h {t50r['ph']['release']}→{t50r['ph'][t]} · p@p {t50r['pp']['release']}→{t50r['pp'][t]} — 정의에 따라 방향이 갈린다.")
        L.append(f"- **지속 — RollOut_v3 (가능 {ncl} clip · 궤적 {R['meta']['n_traj_total']}, 모든 predictor 같은 clip) 에서 릴리즈 대비 정의 ('모임' · README 의 '있다') 와 자 (p@h · p@p) 에 관계없이 같은 방향인 변화는 "
                 + " · ".join([f"{t} ↑" for t in up2] + [f"{t} ↓" for t in dn2] or ["없다"]) + " 뿐이다.** "
                 + "; ".join(seg) + f". 나머지 ({', '.join(rest) or '—'}) 는 정의나 자에 따라 다르다 (표 1)." + ar
                 + (" **pv1_e15 는 같은 10° 쐐기 · 등속 팔 · 연속 가속으로 학습했으니 (stride 만 다르다) 이것은 동역학상 근접 도메인의 이득이지 일반화가 아니다.**" if "pv1_e15" in up2 else ""))
    if W16:
        XL5 = ["flat_v", "flat_a", "flat_d", "ramp_a", "ramp_d"]
        cnt = tot = 0
        for t in oth:
            for law in XL5:
                for rn in ("ph", "pp"):
                    tot += 1
                    cnt += sig(W16["transition"][rn][t][f"{law}|x"]["ranges"]["t4-7"].get("common", {}).get("d_e")) > 0
        fit = {rn: {t: W16["transition"][rn][t]["flat_v|x"]["ranges"]["t4-7"].get("fit") for t in fin} for rn in ("ph", "pp")}
        b = {t: W16["transition"]["ph"][t]["flat_v|x"]["ranges"]["t4-7"]["beta"] for t in fin}
        bp = {t: W16["transition"]["pp"][t]["flat_v|x"]["ranges"]["t4-7"]["beta"] for t in fin}
        ncv = {t: W16["transition"]["ph"][t]["flat_v|x"]["ranges"]["t4-7"]["n_cells"] for t in fin}
        L.append(f"- **앞섬 — 학습한 {len(oth)} predictor 는 자유 x 법칙 전부에서 릴리즈보다 앞서 읽힌다** (공통 '모임' 칸의 짝지은 Δ 의 CI 가 0 위: {cnt}/{tot} = predictor × 5 법칙 × 2 자, C16/P16 t4–7). "
                 "이 앞섬은 곱셈꼴 이득이 아니라 **궤적을 따르는 기울기 ≈ 1 에 고정 오프셋이 더해진 모양**이다 — flat_v 14 궤적 사이 적합 (읽은 이동 = 절편 + 기울기 · GT 이동, 자기 '모임' 칸) p@h: "
                 + " · ".join(f"{t} {fs(fit['ph'][t])}" for t in fin) + "; p@p: " + " · ".join(f"{t} {fs(fit['pp'][t])}" for t in fin)
                 + ". 평균 앞섬 (flat_v x 부호 오차, p@h): " + " · ".join(f"{t} {vs1(sm(W16, 'ph', t, 'common_lag|t4-7')['e'], 0)} px" for t in fin)
                 + ". 그래서 'v11_e10 이 문맥 속도를 따르지 않는다' (이전 판) 는 성립하지 않는다. 참고로 이득 β (읽은 이동 / GT 이동) p@h / p@p: "
                 + " · ".join(f"{t} {v1(b[t], 2)} / {v1(bp[t], 2)}" for t in fin)
                 + f" — 각자의 '모임' 칸 값이라 칸이 다르다 (flat_v p@h: release {ncv['release']} 칸, v11_e10 {ncv.get('v11_e10', '—')} 칸). 짝지은 비교는 Δ 뿐이다.")
        rows_ = []
        for t in fin:
            ea, ed = sm(W16, "ph", "release", "flat_a_rho|t4-7")["ext"][0], sm(W16, "ph", "release", "flat_d_rho|t4-7")["ext"][0]
            a_h, a_p = sm(W16, "ph", t, "flat_a_rho|t4-7")["e"], sm(W16, "pp", t, "flat_a_rho|t4-7")["e"]
            d_h, d_p = sm(W16, "ph", t, "flat_d_rho|t4-7")["e"], sm(W16, "pp", t, "flat_d_rho|t4-7")["e"]
            rows_.append((t, _cls(a_h, 1, ea), _cls(a_p, 1, ea), _cls(d_h, 1, ed), _cls(d_p, 1, ed), a_h, a_p, d_h, d_p))
        gt_both = [r[0] for r in rows_ if r[0] != "release" and r[1] == r[2] == "GT" and r[3] == r[4] == "GT"]
        ext_both = [r[0] for r in rows_ if r[1] in ("연장", "연장 너머") and r[2] in ("연장", "연장 너머") and r[3] in ("연장", "연장 너머") and r[4] in ("연장", "연장 너머")]
        L.append("- **전이 — 가속과 감속을 두 자 모두에서 GT 로 넣는 predictor 는 확인되지 않았다 (이 자료로는 가를 수 없다).** 이득 비 ρ = β(law)/β(flat_v) 는 앞섬이 더하기꼴이라 "
                 f"공통 이동을 지우지 못하고 1 (= GT) 쪽으로 끌린다 — **기술값으로만** 싣는다 (t4–7, C16/P16; GT 1, 등속 연장 flat_a {ea:.2f} · flat_d {ed:.2f}; GT–연장 간격 {abs(1 - ea):.2f} · {abs(ed - 1):.2f} 은 CI 반폭과 비슷하다) p@h / p@p — flat_a: "
                 + " · ".join(f"{r[0]} {v1(r[5], 2)} / {v1(r[6], 2)}" for r in rows_)
                 + " ; flat_d: " + " · ".join(f"{r[0]} {v1(r[7], 2)} / {v1(r[8], 2)}" for r in rows_)
                 + f". CI 분류 (참고, 확정 아님) — 두 법칙 · 두 자 모두 GT 만: {', '.join(gt_both) or '없음'} · 모두 연장 (또는 너머): {', '.join(ext_both) or '없음'}.")
        RHO_ROWS.clear(); RHO_ROWS.extend(rows_)
        af = {t: sm(W16, "ph", t, "arc_fall|t4-7") for t in fin}; afp = {t: sm(W16, "pp", t, "arc_fall|t4-7") for t in fin}
        grob = [t for t in oth if len({sig(sm(W16, rn, t, 'arc_fall|t4-7')[k]) for rn in ('ph', 'pp') for k in ('d_e', 'd_e_oc')}) == 1 and sig(af[t]['d_e_oc']) != 0]
        L.append("- **중력 (arc y, 꼭대기에서 분할 → 문맥 끝 세로 속도 ≈ 0).** 보정 (t0–1 의 화면 좌표 치우침을 빼고 late 부호를 곱한 것; 이전 판은 부호 규칙 때문에 치우침을 두 배로 더했다) 낙하 오차 "
                 f"(− = 덜 떨어짐, GT 0, 복사 ≈ 연장 {v1(af['release']['copy_oc'])}) p@h / p@p: " + " · ".join(f"{t} {v1(af[t]['e_oc'])} / {v1(afp[t]['e_oc'])}" for t in fin)
                 + ". 두 자 × 두 방식 (raw · 보정) 네 곳 모두 같은 방향으로 바뀐 것: "
                 + (", ".join(f"{t} ({'더' if sig(af[t]['d_e_oc']) > 0 else '덜'} 떨어짐; 보정 Δ p@h {vs1(af[t]['d_e_oc'], 1)} · p@p {vs1(afp[t]['d_e_oc'], 1)})" for t in grob) or "없음")
                 + ". 사건 (ledge · wall) 은 능력 정의 밖이라 표 2 에만 둔다.")
    if W32:
        rc = {t: W32["reach"]["ph"][t]["flat_v"] for t in fin}; rcp = {t: W32["reach"]["pp"][t]["flat_v"] for t in fin}
        ll = {rn: W32["reach"][rn]["release"]["xlaws"].get("loglog_tcliff_speed") for rn in ("ph", "pp")}
        L.append(f"- **도달 거리 (flat_v, C16/P32)** — release 가 p@h 로 읽힌 이동의 최댓값 {v1(rc['release']['plateau_present_px'])} px (정본 §2 의 47–58 px 와 같은 크기), "
                 f"'모임' 이 끊기기 직전의 GT 이동 (reach) {v1(rc['release']['reach_px'])} px. reach p@h / p@p: "
                 + " · ".join(f"{t} {v1(rc[t]['reach_px'])} / {v1(rcp[t]['reach_px'])}{'*' if (rc[t]['censored_frac'] or 0) > 0.5 else ''}" for t in oth)
                 + " px (* = 절반 넘는 궤적이 창 끝까지 안 끊김 → 하한). 두 자 판정 (Δ reach): "
                 + " · ".join(f"{t} {_two(rc[t].get('d_reach_px'), rcp[t].get('d_reach_px')).replace('**', '')}" for t in oth)
                 + f". release 의 끊김은 시간 한계 쪽이다 (log t_cliff ~ log 속도 기울기, 0 = 시간 · −1 = 거리: p@h {f1(ll['ph'], 2)} · p@p {f1(ll['pp'], 2)}). T50 · reach 는 깜빡이는 곡선의 첫 교차라 한 튜블릿 dip 에 민감하다 (단서 12).")
    if W16:
        fy = W16["transition"]["ph"]["release"]["flat_v|y"]["e_t"]
        fyp = W16["transition"]["pp"]["release"]["flat_v|y"]["e_t"]
        hy = [q[0] for q in W16["ruler_check"]["h@h"].get("flat_v_y_e", []) if q and q[0] is not None]
        L.append(f"- **위치 판독의 주 자 p@h 는 p 토큰에서 검증되지 않았다.** 세로 운동이 없는 flat_v 에서 release p 를 t0 {vs1(fy[0], 1)} → t7 {vs1(fy[-1], 1)} px 위로 읽는다 "
                 f"(h 자를 진짜 h 에: |y 오차| 최대 {max(abs(x) for x in hy):.1f} px; p@p 를 release p 에: t0 {vs1(fyp[0], 1)} → t7 {vs1(fyp[-1], 1)} px). "
                 "그래서 위 결론은 **행동 수준의 비교**로만 읽고, p@h 의 y 는 행동 지표로만, x 는 p@p 와 같은 방향인 것만 견고로 적었다." if hy else "")
    L.append("- **도메인** — v3 (stride 1, 30 fps, 1 튜블릿 = 2 프레임) 의 시간 척도로 학습한 predictor 는 없다 (v11 · Predictor_v1 post-FT stride 3, IntPhys1 post-FT stride 2, Ariel 12 fps). "
             "그러나 **Predictor_v1 post-FT 는 같은 10° 쐐기 · 등속 팔 · 연속 가속 (같은 법칙 족) 으로 학습해 동역학상 근접 도메인이다** — 그 이득은 일반화 증거가 아니다. v11 post-FT 는 운동 조건마다 한 속도 · 궤적 2 개만 봤다.")
    L.append("")

    # ------------------------------------------------------------------ 표 0 sanity
    L += ["## 표", "", "### 표 0. sanity — release 를 정본과 대조", "",
          "| 읽기 | clip | 위치 차 중앙 / p99 (px) | '있다' 일치 % | recall 여기 / 정본 (%) | 튜블릿 recall 최대 차 (pt) | CROSS_HEAD.md 인쇄값과 최대 차 (pt) |",
          "|---|---|---|---|---|---|---|"]
    for k, v in S["per_clip"].items():
        rb = S["recall_by_tubelet"][k]
        ch = S["cross_head_printed"].get(k)
        chd = f"{max(abs(a - b) for a, b in zip(rb['ours'], ch)):.1f}" if ch else "—"
        L.append(f"| `{k}` | {v['n_clip']} | {v['pos_diff_median_px']} / {v['pos_diff_p99_px']} | {v['present_agree']} | {v['recall_ours']} / {v['recall_ref']} | {rb['max_abs_diff_pt']} | {chd} |")
    sg = S.get("signed_C16_P16", {})
    fvr = S.get("flat_v_read_disp_C16_P32_ph", {})
    L += ["", f"- 부호 오차 표 C16/P16 (`RollOutV3/figures/v3_signed_error{{,_hhead}}/_superseded/presence_decoder/values.json`, 시나리오 안 '있다' clip 평균): 최대 |차| p 자 **{sg.get('p', {}).get('max_vs_values_json')} px** · "
          f"h 자 **{sg.get('h', {}).get('max_vs_values_json')} px** (8 법칙 × x·y × 8 튜블릿).",
          f"- C16/P32 flat_v 읽은 x 이동 (p@h, '있다' clip 평균, t8–t14): {', '.join(f'{k} {v}' for k, v in fvr.items())} px — 정본 §2 의 '47–58 px' 와 같은 범위.",
          "- C16/P32 는 정본과 같은 배치 크기 (5) 라 비트 동일, C16/P16 은 배치 크기 5 vs 정본 8 이라 bf16 잡음 (중앙 0.1–0.9 px) 만큼 다르다 (`te_v3_readout.py` 머리말 ④ 와 같은 크기).", ""]

    # ------------------------------------------------------------------ 표 1 지속
    L += ["### 표 1. 지속 — 자유 6 법칙 (층화), 읽을 수 있는 칸, 값 [95 % CI] (Δ vs release)", "",
          "'있다' = 고정 문턱 (**README §1 의 지속 정의**), '모임' = '있다' ∧ 진실 3×3 질량 > 0.035 (위치 정확도가 섞인다), 질량 = 진실 3×3 attention 질량 평균 (문턱 없음). T50 = 층화 곡선이 처음 0.5 아래 (16 = 창 끝까지 안 떨어짐). "
          "'견고' = 두 정의 ('모임' · '있다') × 두 자 모두 같은 방향 (2026-09-25).", "",
          "| predictor | P16 p@h '모임' t4–7 | P16 p@p '모임' t4–7 | P16 질량 t4–7 (p@h) | P32 p@h '모임' T50 | P32 p@h '모임' t8–15 | P32 p@p '모임' T50 | P32 p@p '모임' t8–15 | P32 질량 t8–15 (p@h) | t8–15 '모임' 두 자 | P32 '있다' T50 p@h / p@p | P32 '있다' t8–15 p@h | P32 '있다' t8–15 p@p | t8–15 '있다' 두 자 | 두 정의 × 두 자 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def cell(v, dv, nd=2):
        return f"{v1(v, nd)}" + ("" if dv is None else f" ({f1(dv, nd)})")
    for t in fin:
        R16 = W16["persistence"] if W16 else None; R32 = W32["persistence"] if W32 else None
        dd = None if t == "release" else 1
        row = [t,
               cell(pr(W16, "ph", t, "loc", "t4-7"), dd and dpr(W16, "ph", t, "loc", "t4-7")),
               cell(pr(W16, "pp", t, "loc", "t4-7"), dd and dpr(W16, "pp", t, "loc", "t4-7")),
               cell(R16["ph"][t]["mass_ranges"]["t4-7"], dd and R16["ph"][t]["d_mass_ranges"]["t4-7"], 3)]
        if W32:
            for rn in ("ph", "pp"):
                q = R32[rn][t]["loc"]
                row.append(f"{q['T50']} [{q['T50_ci'][0]:.0f}, {q['T50_ci'][1]:.0f}]" + ("" if t == "release" else f" (Δ {q['dT50'][0]:+d} [{q['dT50'][1]:+.0f}, {q['dT50'][2]:+.0f}])"))
                row.append(cell(pr(W32, rn, t, "loc", "t8-15"), dd and dpr(W32, rn, t, "loc", "t8-15")))
            row.append(cell(R32["ph"][t]["mass_ranges"]["t8-15"], dd and R32["ph"][t]["d_mass_ranges"]["t8-15"], 3))
            row.append(rob.get(t, "—"))
            row.append(f"{R32['ph'][t]['recall']['T50']} / {R32['pp'][t]['recall']['T50']}")
            row.append(cell(pr(W32, 'ph', t, 'recall', 't8-15'), dd and dpr(W32, 'ph', t, 'recall', 't8-15')))
            row.append(cell(pr(W32, 'pp', t, 'recall', 't8-15'), dd and dpr(W32, 'pp', t, 'recall', 't8-15')))
            row.append(rob_rec.get(t, "—"))
            row.append("—" if t == "release" else ("**견고 ↑**" if t in up2 else ("**견고 ↓**" if t in dn2 else "정의·자에 따라 다름")))
        L.append("| " + " | ".join(str(x) for x in row) + " |")
    if W32:
        hh = W32["ruler_check"]["h@h"]; zz = W32["ruler_check"]["z@z"]
        L += ["", f"자 검사 (predictor 무관, C16/P32): h 자를 진짜 h 에 — '모임' t8–15 {np.mean([q[0] for q in hh['loc_free6'][8:]]):.2f}, T50 {hh['T50']}; "
              f"z 자를 z 에 — {np.mean([q[0] for q in zz['loc_free6'][8:]]):.2f}, T50 {zz['T50']}. 자는 t8–15 의 진짜 표현을 읽는다 — 거기서 p 가 떨어지는 것은 자가 창 후반이라 깨져서가 아니다 "
              "(단 자는 미래 8 튜블릿까지만 학습했다, 단서 4)."]
    L.append("")

    # ------------------------------------------------------------------ 표 2 전이
    if W16:
        L += ["### 표 2. 전이 — C16/P16 (v11 · Predictor_v1 post-FT 의 학습 창 길이), t4–7, 두 자", "",
              "값 = 자기 '모임' 칸 (궤적 bootstrap CI). Δ = release 와 공통 '모임' 칸의 짝지은 차 [CI]. 기준 = 공통 칸에서 |값 − 기준| 이 줄었나 (가까이) 늘었나 (멀리), CI 가 0 을 넘을 때만. "
              "두 자 = Δ 의 방향이 p@h · p@p 에서 같은가. ρ = 이득 비 (flat_v 로 나눈 β; 앞섬이 더하기꼴이라 공통 이동을 다 지우지 못한다 → **기술값**). "
              "'보정' = t0–1 의 **화면 좌표** 치우침을 빼고 late 부호를 곱한 것 (2026-09-25 부호 정정; 이전 판의 arc · ledge 보정 값은 인용 금지).", ""]
        for lab, key, vk, dk, refs in ITEMS:
            s0h, s0p = sm(W16, "ph", "release", key), sm(W16, "pp", "release", key)
            refv = []
            if s0h.get("ext") and s0h["ext"] and s0h["ext"][0] is not None and vk == "e":
                refv.append(f"연장 {s0h['ext'][0]:+.2f}" if "rho" in key else f"연장 {s0h['ext'][0]:+.1f}")
            if vk == "e" and s0h.get("copy") and s0h["copy"] and s0h["copy"][0] is not None and "rho" not in key:
                refv.append(f"복사 {s0h['copy'][0]:+.1f}")
            if vk == "e_oc":
                refv.append(f"복사≈연장 {s0h['copy_oc'][0]:+.1f}")
            nd = 2 if "rho" in key else 1
            L += [f"**{lab}** — 기준 (release 칸): {' · '.join(refv) or '—'}", "",
                  "| predictor | p@h 값 | p@h Δ | p@h 기준 | p@p 값 | p@p Δ | p@p 기준 | 두 자 |", "|---|---|---|---|---|---|---|---|"]
            for t in fin:
                sh, sp = sm(W16, "ph", t, key), sm(W16, "pp", t, key)
                if t == "release":
                    L.append(f"| release | {f1(sh[vk], nd)} | — | — | {f1(sp[vk], nd)} | — | — | — |")
                    continue
                L.append(f"| {t} | {f1(sh[vk], nd)} | {f1(sh.get(dk), nd)} | {_refs(sh, refs)} | {f1(sp[vk], nd)} | {f1(sp.get(dk), nd)} | {_refs(sp, refs)} | {_two(sh.get(dk), sp.get(dk))} |")
            L.append("")
        # 궤적 사이 기울기 + 절편 (2026-09-25; own−null 지표 대체)
        L += ["**궤적 사이 적합** — 읽은 이동 = 절편 + 기울기 · GT 이동 (법칙 안 14 궤적, t4–7, 자기 '모임' 칸, 궤적 bootstrap CI). "
              "곱셈꼴 이득이면 절편 ≈ 0 · 기울기 ≈ β, 더하기꼴 앞섬이면 기울기 ≈ 1 · 절편 > 0. 기울기 CI 가 0 위 = 궤적 속도를 따른다. "
              "(이전 판의 own − null 지표는 공통 오프셋이 GT 이동의 퍼짐보다 크면 추적을 못 잡아 뺐다 — motion.json 에만 남는다.)", "",
              "| predictor | flat_v p@h | flat_v p@p | flat_a p@h | flat_a p@p | flat_d p@h | flat_d p@p |", "|---|---|---|---|---|---|---|"]
        for t in fin:
            cs = []
            for key in ("flat_v|x", "flat_a|x", "flat_d|x"):
                for rn_ in ("ph", "pp"):
                    f_ = W16["transition"][rn_][t][key]["ranges"]["t4-7"].get("fit")
                    cs.append("—" if not f_ or f_["slope"][0] is None else f"{f1(f_['slope'], 2, False)} · {f_['intercept'][0]:+.0f}")
            L.append(f"| {t} | " + " | ".join(cs) + " |")
        L.append("")
    if W32:
        L += ["### 표 2b. 같은 요약, C16/P32 — t4–7 (학습 창 안) · t8–15 (밖), p@h", "",
              "| predictor | " + " | ".join(f"{lab.split(' (')[0]} {rk}" for lab, key, vk, dk, refs in ITEMS[:7] for rk in ("t4-7", "t8-15")) + " |",
              "|---|" + "---|" * 14]
        for t in fin:
            cs = []
            for lab, key, vk, dk, refs in ITEMS[:7]:
                for rk in ("t4-7", "t8-15"):
                    s_ = sm(W32, "ph", t, key.replace("t4-7", rk))
                    nd = 2 if "rho" in key else 1
                    cs.append(v1(s_[vk], nd) + ("" if t == "release" else f" {mark(s_.get(dk))}"))
            L.append(f"| {t} | " + " | ".join(cs) + " |")
        L += ["", "↑/↓/≈ = release 대비 Δ (공통 칸) 의 CI. t8–15 의 '모임' 칸은 release 도 절반 아래라 (표 1) 칸 수가 적다 — 칸 수는 TABLES.md T3.", ""]

    # ------------------------------------------------------------------ 표 3 도달
    if W32:
        L += ["### 표 3. 도달 거리 — flat_v 14 궤적, C16/P32", "",
              "reach = '모임' 비율이 처음 0.5 아래로 가기 직전 튜블릿의 GT 이동 (px; 창 끝까지 안 끊기면 그 값 = 하한). 그때 읽은 이동 = 같은 튜블릿의 읽은 이동 ('모임' 칸). "
              "plateau = 읽은 이동의 최댓값 ('있다' 칸, 질량 무관 — 자 기본값이 섞인다). corr = x 자유 6 법칙 84 궤적 (끊긴 궤적만) 의 궤적 사이 상관: "
              "시간 한계면 corr(reach, 속도) > 0 · corr(t_cliff, 속도) ≈ 0, 거리 한계면 corr(t_cliff, 속도) < 0 · corr(reach, 속도) ≈ 0.", "",
              "| predictor | 자 | t_cliff | reach (px) | 그때 읽은 이동 | plateau | 중도절단 | Δ reach | corr(t_cliff, 속도) | corr(reach, 속도) | log–log 기울기 (0 시간 · −1 거리) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for t in fin:
            for rn in ("ph", "pp"):
                g = W32["reach"][rn][t]["flat_v"]; x = W32["reach"][rn][t]["xlaws"]
                L.append(f"| {t} | {'p@h' if rn == 'ph' else 'p@p'} | {f1(g['t_cliff'], 1, False)} | {f1(g['reach_px'], 0, False)} | {v1(g['read_disp_at_reach_px'], 0)} | "
                         f"{f1(g['plateau_present_px'], 0, False)} | {g['censored_frac']} | {f1(g.get('d_reach_px'), 0)} | {f1(x.get('corr_tcliff_speed'), 2)} | {f1(x.get('corr_reach_speed'), 2)} | {f1(x.get('loglog_tcliff_speed'), 2)} |")
        L.append("")

    # ------------------------------------------------------------------ 표 4 학습 곡선
    L += ["### 표 4. 학습 곡선 — p@h (↑/↓/≈ = release 대비 Δ 의 CI; 지속은 궤적 차, 운동은 공통 칸)", "",
          "| run | 체크포인트 | P32 '모임' T50 | P32 '모임' t8–15 | P16 '모임' t4–7 | flat_v x 오차 | ρ flat_a | ρ flat_d | arc 낙하 (보정) | wall 넘어감 | flat_v reach (P32) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for fam, lst in FAMILIES.items():
        for ep, t in lst:
            if t not in tags or (t == "release" and fam != "v11"):
                continue
            cs = []
            if W32:
                q = W32["persistence"]["ph"][t]["loc"]
                cs.append(f"{q['T50']}" + ("" if t == "release" else f" ({q['dT50'][0]:+d})"))
                cs.append(v1(pr(W32, "ph", t, "loc", "t8-15"), 2) + ("" if t == "release" else f" {mark(dpr(W32, 'ph', t, 'loc', 't8-15'))}"))
            else:
                cs += ["—", "—"]
            cs.append(v1(pr(W16, "ph", t, "loc", "t4-7"), 2) + ("" if t == "release" else f" {mark(dpr(W16, 'ph', t, 'loc', 't4-7'))}"))
            for key, vk, dk, nd in (("common_lag|t4-7", "e", "d_e", 1), ("flat_a_rho|t4-7", "e", "d_e", 2), ("flat_d_rho|t4-7", "e", "d_e", 2),
                                    ("arc_fall|t4-7", "e_oc", "d_e_oc", 1), ("wall_over|t4-7", "e", "d_e", 1)):
                s_ = sm(W16, "ph", t, key)
                cs.append(v1(s_[vk], nd) + ("" if t == "release" else f" {mark(s_.get(dk))}"))
            if W32:
                g = W32["reach"]["ph"][t]["flat_v"]
                cs.append(v1(g["reach_px"], 0) + ("" if t == "release" else f" {mark(g.get('d_reach_px'))}"))
            L.append(f"| {FAM_NAME[fam] if t != 'release' else '(e0)'} | {t} | " + " | ".join(cs) + " |")
    L.append("")
    L += ["그림: `fig_persistence.png` (지속 곡선) · `fig_transition_C16_P16.png` · `fig_transition_C16_P32.png` (법칙별 late 오차와 복사 · 연장 기준) · "
          "`fig_contrast.png` (law − flat_v) · `fig_transition_curves_C16_P16.png` · `fig_transition_curves_C16_P32.png` (e(t)) · `fig_reach.png` · "
          "`fig_delta_heatmap.png` (22 tag × 법칙 Δ) · `fig_learning_curves.png`.", ""]

    # ------------------------------------------------------------------ 읽기 · 단서 · 도메인 · 재현
    L += narrative(R)
    L += ["## 재현", "", "```bash", "cd /data/hyuntak/project/2026/2027_cvpr/vjepa2",
          "P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python",
          "$P z_research/scripts/analysis/te_v3_readout.py --preset curves --gpus 4        # 추출 (GPU; → cache/training_effects/v3readout_curves/)",
          "$P z_research/scripts/analysis/te_analyze_v3.py                                  # 이 문서 · TABLES.md · motion.json · fig_*.png (CPU, ~6 분)",
          "```",
          "입력: `/data2/local_datasets/world/world_analysis/cache/training_effects/v3readout_curves/readings.npz` (자 지문 dcd24d8a8a47) · "
          "`z_research/RollOutV3/exp_results/presence/{summary.json, attn_bias_px.json}` · `data_csv/rollout_v3/index.csv`. "
          "sanity 대조: `RollOutV3/exp_results/windows/readings.npz` · `RollOutV3/figures/v3_signed_error{,_hhead}/_superseded/presence_decoder/values.json` · `CROSS_HEAD.md` (읽기 전용).", ""]
    (out / "MOTION.md").write_text("\n".join(L) + "\n")


BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
GRAY_D, GRAY_M, GRAY_L = "#444444", "#8c8c8c", "#c8c8c8"
KEYNAME = {"flat_v|x": "flat_v  x", "flat_a|x": "flat_a  x (accel)", "flat_d|x": "flat_d  x (decel)", "ramp_a|x": "ramp_a  x (accel)",
           "ramp_d|x": "ramp_d  x (decel)", "arc|y": "arc  y (gravity)", "ledge|y": "ledge  y (fall, event)", "wall|x": "wall  x (stop, event)",
           "arc|x": "arc  x"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": GRAY_M, "axes.labelcolor": "#222222", "xtick.color": "#333333", "ytick.color": "#333333"})
    return plt


def _plabel(ax, i, y=-0.30):
    ax.text(0.5, y, f"({chr(97 + i)})", transform=ax.transAxes, ha="center", va="top", fontsize=9)


def _band(ax, x, v, color, lw=1.8, ls="-", label=None, alpha=0.18, marker=None):
    x = np.asarray(x, float)
    m = np.array([np.nan if (q is None or q[0] is None) else q[0] for q in v], float)
    lo = np.array([np.nan if (q is None or q[1] is None) else q[1] for q in v], float)
    hi = np.array([np.nan if (q is None or q[2] is None) else q[2] for q in v], float)
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, lw=0)
    ax.plot(x, m, color=color, lw=lw, ls=ls, label=label, marker=marker, ms=3.5)


def figures(R, out):
    plt = _plt()
    from matplotlib.lines import Line2D
    from matplotlib.colors import LinearSegmentedColormap
    W_ = R["windows"]
    fin = [t for t in FINAL if t in R["meta"]["tags"]]
    oth = [t for t in fin if t != "release"]

    # ---------------------------------------------------------------- F1 지속 곡선 (C16/P32)
    if "C16_P32" in W_:
        w = W_["C16_P32"]; T = np.arange(w["tp"])
        rows = [("ph", "recall", "p@h present"), ("ph", "loc", "p@h present & on GT"), ("pp", "recall", "p@p present")]
        f, ax = plt.subplots(len(rows), len(oth), figsize=(2.9 * len(oth), 2.35 * len(rows) + 0.9), sharex=True, sharey=True, squeeze=False)
        k = 0
        for i, (rn, nm, lab) in enumerate(rows):
            for j, t in enumerate(oth):
                a = ax[i, j]
                ruler = "h@h" if rn == "ph" else None
                if ruler:
                    a.plot(T, [q[0] for q in w["ruler_check"]["h@h"]["loc_free6" if nm == "loc" else "recall_free6"]], color=GRAY_L, lw=1.3, ls="--")
                _band(a, T, w["persistence"][rn]["release"][nm]["curve"], "black", lw=1.6)
                _band(a, T, w["persistence"][rn][t][nm]["curve"], BLUE, lw=1.8)
                a.axhline(0.5, color=GRAY_M, lw=0.7, ls=":"); a.axvline(7.5, color=GRAY_M, lw=0.7, ls=":")
                a.set_ylim(-0.03, 1.05); a.set_xticks([0, 4, 8, 12, 15])
                if i == 0:
                    a.set_title(t, fontsize=9.5, fontweight="bold")
                if j == 0:
                    a.set_ylabel(f"{lab}\n(fraction, free 6 laws)")
                if i == len(rows) - 1:
                    a.set_xlabel("future tubelet (C16/P32)")
                _plabel(a, k, -0.36 if i == len(rows) - 1 else -0.12); k += 1
        hd = [Line2D([], [], color="black", lw=1.6, label="release"), Line2D([], [], color=BLUE, lw=1.8, label="trained predictor"),
              Line2D([], [], color=GRAY_L, lw=1.3, ls="--", label="h ruler on real h (ceiling)"),
              Line2D([], [], color=GRAY_M, lw=0.7, ls=":", label="t = 8: beyond training horizon of rulers and post-FT")]
        f.legend(handles=hd, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
        f.tight_layout(rect=(0, 0.06, 1, 1)); f.savefig(out / "fig_persistence.png", dpi=150); plt.close(f)

    # ---------------------------------------------------------------- F2 전이 수준 (late t4–7) — 창 둘
    for wk in [k_ for k_ in ("C16_P16", "C16_P32") if k_ in W_]:
        w = W_[wk]
        keys = [f"{l}|{AXN[a]}" for l, a in KEYAX]
        f, ax = plt.subplots(2, 4, figsize=(15, 6.4), squeeze=False)
        for i, key in enumerate(keys):
            a = ax.flat[i]
            ys = np.arange(len(fin))[::-1]
            for y, t in zip(ys, fin):
                q = w["transition"]["ph"][t][key]["ranges"]["t4-7"]
                col = "black" if t == "release" else BLUE
                if q["e"][0] is not None:
                    a.plot([q["e"][1], q["e"][2]], [y, y], color=col, lw=1.6)
                    a.plot(q["e"][0], y, "o", color=col, ms=6, mec="white", mew=1.0, zorder=3)
                if q["copy"][0] is not None:
                    a.plot(q["copy"][0], y, marker="o", ms=6, mfc="none", mec=GRAY_M, mew=1.2, ls="")
                    a.plot(q["ext"][0], y, marker="s", ms=6, mfc="none", mec=GRAY_D, mew=1.2, ls="")
            a.axvline(0, color=GRAY_M, lw=0.9)
            a.set_yticks(ys); a.set_yticklabels(fin if i % 4 == 0 else [""] * len(fin))
            a.set_title(KEYNAME.get(key, key), fontsize=9.5, fontweight="bold")
            a.set_xlabel("signed error, t4–7 (px; + ahead)")
            _plabel(a, i, -0.28)
        hd = [Line2D([], [], color="black", marker="o", lw=1.6, label="release (95% CI, trajectories)"),
              Line2D([], [], color=BLUE, marker="o", lw=1.6, label="trained predictor"),
              Line2D([], [], color=GRAY_M, marker="o", mfc="none", ls="", label="copy baseline (same cells)"),
              Line2D([], [], color=GRAY_D, marker="s", mfc="none", ls="", label="constant-velocity baseline (same cells)"),
              Line2D([], [], color=GRAY_M, lw=0.9, label="GT = 0")]
        f.legend(handles=hd, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.0))
        f.tight_layout(rect=(0, 0.07, 1, 1)); f.savefig(out / f"fig_transition_{wk}.png", dpi=150); plt.close(f)

    # ---------------------------------------------------------------- F3 법칙 대비 (law − flat_v)
    wks = [k_ for k_ in ("C16_P16", "C16_P32") if k_ in W_]
    f, ax = plt.subplots(len(wks), 4, figsize=(14, 3.0 * len(wks) + 0.8), squeeze=False)
    k = 0
    for i, wk in enumerate(wks):
        w = W_[wk]
        for j, law in enumerate(["flat_a", "flat_d", "ramp_a", "ramp_d"]):
            a = ax[i, j]; ys = np.arange(len(fin))[::-1]
            for y, t in zip(ys, fin):
                q = w["contrast"]["ph"][t]["t4-7"][law]
                col = "black" if t == "release" else BLUE
                if q["k"][0] is not None:
                    a.plot([q["k"][1], q["k"][2]], [y, y], color=col, lw=1.6)
                    a.plot(q["k"][0], y, "o", color=col, ms=6, mec="white", mew=1.0, zorder=3)
                    a.plot(q["ext_pred"][0], y, marker="s", ms=6, mfc="none", mec=GRAY_D, mew=1.2, ls="")
                    a.plot(q["copy_pred"][0], y, marker="o", ms=6, mfc="none", mec=GRAY_M, mew=1.2, ls="")
            a.axvline(0, color=GRAY_M, lw=0.9)
            a.set_yticks(ys); a.set_yticklabels(fin if j == 0 else [""] * len(fin))
            if i == 0:
                a.set_title(f"{law} − flat_v", fontsize=9.5, fontweight="bold")
            a.set_xlabel(f"x signed-error contrast, t4–7 (px) · {wk.replace('_', '/')}")
            _plabel(a, k, -0.30); k += 1
    hd = [Line2D([], [], color="black", marker="o", lw=1.6, label="release"), Line2D([], [], color=BLUE, marker="o", lw=1.6, label="trained predictor"),
          Line2D([], [], color=GRAY_D, marker="s", mfc="none", ls="", label="constant-velocity prediction"),
          Line2D([], [], color=GRAY_M, marker="o", mfc="none", ls="", label="copy prediction"), Line2D([], [], color=GRAY_M, lw=0.9, label="GT = 0")]
    f.legend(handles=hd, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    f.tight_layout(rect=(0, 0.07, 1, 1)); f.savefig(out / "fig_contrast.png", dpi=150); plt.close(f)

    # ---------------------------------------------------------------- F4 곡선 e(t) (C16/P32)
    for wk in [k_ for k_ in ("C16_P16", "C16_P32") if k_ in W_]:
        w = W_[wk]; T = np.arange(w["tp"])
        keys = [f"{l}|{AXN[a]}" for l, a in KEYAX]
        f, ax = plt.subplots(len(keys), len(oth), figsize=(2.8 * len(oth), 1.75 * len(keys) + 0.9), sharex=True, squeeze=False)
        for i, key in enumerate(keys):
            R0 = w["transition"]["ph"]["release"][key]
            vals = []
            for j, t in enumerate(oth):
                a = ax[i, j]; Q = w["transition"]["ph"][t][key]
                a.plot(T, [q[0] for q in R0["copy_t"]], color=GRAY_M, lw=1.0, ls="-.")
                a.plot(T, [q[0] for q in R0["ext_t"]], color=GRAY_D, lw=1.0, ls="--")
                _band(a, T, R0["e_t"], "black", lw=1.4)
                _band(a, T, Q["e_t"], BLUE, lw=1.6)
                a.axhline(0, color=GRAY_L, lw=0.7)
                if w["tp"] > 8:
                    a.axvline(7.5, color=GRAY_M, lw=0.7, ls=":")
                vals += [q[0] for q in R0["e_t"] + Q["e_t"] + R0["ext_t"] + R0["copy_t"] if q[0] is not None]
                if i == 0:
                    a.set_title(t, fontsize=9.5, fontweight="bold")
                if j == 0:
                    a.set_ylabel(KEYNAME.get(key, key).replace("  ", "\n"), fontsize=8)
                if i == len(keys) - 1:
                    a.set_xlabel("future tubelet"); _plabel(a, j, -0.45)
            if vals:
                lo_, hi_ = np.percentile(vals, [1, 99]); pad = 0.1 * (hi_ - lo_ + 1)
                for j in range(len(oth)):
                    ax[i, j].set_ylim(lo_ - pad, hi_ + pad)
        hd = [Line2D([], [], color="black", lw=1.4, label="release (p@h, on-GT cells)"), Line2D([], [], color=BLUE, lw=1.6, label="trained predictor"),
              Line2D([], [], color=GRAY_D, lw=1.0, ls="--", label="constant velocity"), Line2D([], [], color=GRAY_M, lw=1.0, ls="-.", label="copy"),
              Line2D([], [], color=GRAY_L, lw=0.7, label="GT = 0")]
        f.legend(handles=hd, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.0))
        f.tight_layout(rect=(0, 0.04, 1, 1)); f.savefig(out / f"fig_transition_curves_{wk}.png", dpi=140); plt.close(f)

    # ---------------------------------------------------------------- F5 도달 거리 (flat_v, C16/P32)
    if "C16_P32" in W_:
        w = W_["C16_P32"]; T = np.arange(w["tp"])
        f, ax = plt.subplots(1, len(oth) + 1, figsize=(3.0 * (len(oth) + 1), 3.3), squeeze=False)
        gt = w["reach"]["ph"]["release"]["flat_v_curve"]["gt"]
        for j, t in enumerate(oth):
            a = ax[0, j]
            a.plot(T, [q[0] for q in gt], color=GRAY_D, lw=1.2, ls="--")
            _band(a, T, w["reach"]["ph"]["release"]["flat_v_curve"]["read_loc"], "black", lw=1.4)
            _band(a, T, w["reach"]["ph"][t]["flat_v_curve"]["read_loc"], BLUE, lw=1.6)
            a.plot(T, [q[0] for q in w["reach"]["ph"][t]["flat_v_curve"]["read_present"]], color=BLUE, lw=1.0, ls=":")
            a.axvline(7.5, color=GRAY_M, lw=0.7, ls=":")
            a.set_title(t, fontsize=9.5, fontweight="bold"); a.set_xlabel("future tubelet")
            if j == 0:
                a.set_ylabel("flat_v: x displacement from\nlast seen position (px)")
            _plabel(a, j, -0.25)
        a = ax[0, -1]; ys = np.arange(len(fin))[::-1]
        for y, t in zip(ys, fin):
            g = w["reach"]["ph"][t]["flat_v"]; col = "black" if t == "release" else BLUE
            if g["reach_px"][0] is not None:
                a.plot([g["reach_px"][1], g["reach_px"][2]], [y, y], color=col, lw=1.6)
                a.plot(g["reach_px"][0], y, "o", color=col, ms=6, mec="white", zorder=3)
            if g["plateau_present_px"][0] is not None:
                a.plot(g["plateau_present_px"][0], y + 0.25, marker="^", ms=5.5, mfc="none", mec=col, ls="")
        a.set_yticks(ys); a.set_yticklabels(fin); a.set_xlabel("px from last seen position")
        a.set_title("reach (on GT) · plateau", fontsize=9.5, fontweight="bold"); _plabel(a, len(oth), -0.25)
        hd = [Line2D([], [], color=GRAY_D, lw=1.2, ls="--", label="GT displacement"), Line2D([], [], color="black", lw=1.4, label="release, on-GT cells"),
              Line2D([], [], color=BLUE, lw=1.6, label="trained, on-GT cells"), Line2D([], [], color=BLUE, lw=1.0, ls=":", label="trained, present cells (any mass)"),
              Line2D([], [], color=GRAY_D, marker="o", ls="", label="reach = GT displacement at last on-GT tubelet"),
              Line2D([], [], color=GRAY_D, marker="^", mfc="none", ls="", label="plateau = max read displacement")]
        f.legend(handles=hd, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
        f.tight_layout(rect=(0, 0.16, 1, 1)); f.savefig(out / "fig_reach.png", dpi=150); plt.close(f)

    # ---------------------------------------------------------------- F6 Δ 열지도 (Δ vs release, 공통 칸, t4–7)
    cmap = LinearSegmentedColormap.from_list("bgo", [BLUE, "#e9e9e9", ORANGE])
    tags = [t for t in R["meta"]["tags"] if t != "release"]
    cols = [f"{l}|{AXN[a]}" for l, a in KEYAX] + ["arc|x"]
    f, ax = plt.subplots(1, len(wks), figsize=(6.2 * len(wks), 0.34 * len(tags) + 2.4), squeeze=False)
    for i, wk in enumerate(wks):
        w = W_[wk]; a = ax[0, i]
        M = np.full((len(tags), len(cols) + 2), np.nan); S_ = np.zeros_like(M, bool)
        for r, t in enumerate(tags):
            for c, key in enumerate(cols):
                cm = w["transition"]["ph"][t][key]["ranges"]["t4-7"].get("common")
                if cm and cm["d_e"][0] is not None:
                    M[r, c] = cm["d_e"][0]; S_[r, c] = sig(cm["d_e"]) != 0
            for c, law in enumerate(["flat_a", "flat_d"]):
                q = w["contrast"]["ph"][t]["t4-7"][law]["common"]["d_c"]
                if q[0] is not None:
                    M[r, len(cols) + c] = q[0]; S_[r, len(cols) + c] = sig(q) != 0
        vmax = np.nanpercentile(np.abs(M), 95) if np.isfinite(M).any() else 1
        a.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        for r in range(M.shape[0]):
            for c in range(M.shape[1]):
                if np.isfinite(M[r, c]):
                    a.text(c, r, f"{M[r, c]:+.0f}" + ("" if S_[r, c] else "°"), ha="center", va="center", fontsize=6.8,
                           color="#111111" if S_[r, c] else GRAY_M)
        a.set_xticks(range(M.shape[1])); a.set_xticklabels([k_.replace("|", " ") for k_ in cols] + ["flat_a−flat_v", "flat_d−flat_v"], rotation=45, ha="right")
        a.set_yticks(range(len(tags))); a.set_yticklabels(tags if i == 0 else [""] * len(tags))
        a.axvline(len(cols) - 0.5, color="white", lw=3)
        for r in range(1, len(tags)):
            if tags[r].split("_")[0] != tags[r - 1].split("_")[0]:
                a.axhline(r - 0.5, color="white", lw=2)
        a.set_title(f"Δ signed error vs release, t4–7 (px) · {wk.replace('_', '/')}", fontsize=9)
        a.spines[:].set_visible(False)
        _plabel(a, i, -0.16)
    f.tight_layout(); f.savefig(out / "fig_delta_heatmap.png", dpi=150); plt.close(f)

    # ---------------------------------------------------------------- F7 학습 곡선
    metrics = []
    if "C16_P32" in W_:
        metrics += [("C16_P32", "pers", ("ph", "loc", "T50"), "T50, p@h on GT\n(C16/P32, tubelets)"),
                    ("C16_P32", "pers", ("ph", "loc", "t8-15"), "on-GT rate t8–15\n(C16/P32)")]
    metrics += [("C16_P16", "pers", ("ph", "loc", "t4-7"), "on-GT rate t4–7\n(C16/P16)"),
                ("C16_P16", "sum", ("common_lag|t4-7", "e"), "flat_v x error\nt4–7 (px)"),
                ("C16_P16", "sum", ("flat_a_rho|t4-7", "e"), "gain ratio flat_a\n/ flat_v, t4–7"),
                ("C16_P16", "sum", ("flat_d_rho|t4-7", "e"), "gain ratio flat_d\n/ flat_v, t4–7"),
                ("C16_P16", "sum", ("arc_fall|t4-7", "e_oc"), "arc y fall, offset-\ncorrected t4–7 (px)"),
                ("C16_P16", "sum", ("wall_over|t4-7", "e"), "wall x overshoot\nt4–7 (px)")]
    if "C16_P32" in W_:
        metrics += [("C16_P32", "reach", ("ph", "flat_v", "reach_px"), "flat_v reach on GT\n(C16/P32, px)")]
    fams = list(FAMILIES)
    f, ax = plt.subplots(len(metrics), len(fams), figsize=(2.9 * len(fams), 1.65 * len(metrics) + 0.9), sharey="row", squeeze=False)

    def getv(wk, kind, spec, t):
        w = W_[wk]
        if kind == "pers":
            rn, nm, what = spec; d = w["persistence"][rn][t][nm]
            return [d["T50"], d["T50_ci"][0], d["T50_ci"][1]] if what == "T50" else d["ranges"][what]
        if kind == "sum":
            return w["summary"]["ph"][t][spec[0]][spec[1]]
        rn, grp, what = spec
        return w["reach"][rn][t][grp][what]

    def refs(wk, kind, spec):
        if kind != "sum":
            return []
        s = W_[wk]["summary"]["ph"]["release"][spec[0]]
        if "_rho" in spec[0]:
            return [(1.0, GRAY_M, "-"), (s["ext"][0], GRAY_D, "--")]
        out_ = [(0.0, GRAY_M, "-")]
        if spec[1] == "e_oc":
            out_.append((s["copy_oc"][0], GRAY_M, "-."))
        else:
            if s.get("ext") and s["ext"][0] is not None:
                out_.append((s["ext"][0], GRAY_D, "--"))
            if s.get("copy") and s["copy"][0] is not None and abs(s["copy"][0]) < 30:
                out_.append((s["copy"][0], GRAY_M, "-."))
        return out_

    for i, (wk, kind, spec, lab) in enumerate(metrics):
        rel = getv(wk, kind, spec, "release")
        for j, fam in enumerate(fams):
            a = ax[i, j]; ep = [e for e, _ in FAMILIES[fam]]; tg = [t for _, t in FAMILIES[fam]]
            v = [getv(wk, kind, spec, t) for t in tg]
            if rel and rel[0] is not None:
                a.axhspan(rel[1], rel[2], color=GRAY_L, alpha=0.5, lw=0); a.axhline(rel[0], color="black", lw=1.2)
            for y0, c0, ls0 in refs(wk, kind, spec):
                if y0 is not None:
                    a.axhline(y0, color=c0, lw=0.9, ls=ls0)
            _band(a, ep, v, BLUE, lw=1.6, marker="o")
            if i == 0:
                a.set_title(FAM_NAME[fam], fontsize=9, fontweight="bold")
            if j == 0:
                a.set_ylabel(lab, fontsize=7.8)
            if i == len(metrics) - 1:
                a.set_xlabel("epoch"); _plabel(a, j, -0.55)
    hd = [Line2D([], [], color=BLUE, lw=1.6, marker="o", label="checkpoint (95% CI)"), Line2D([], [], color="black", lw=1.2, label="release"),
          Line2D([], [], color=GRAY_D, lw=0.9, ls="--", label="constant velocity"), Line2D([], [], color=GRAY_M, lw=0.9, ls="-.", label="copy"),
          Line2D([], [], color=GRAY_M, lw=0.9, label="GT = 0")]
    f.legend(handles=hd, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    f.tight_layout(rect=(0, 0.035, 1, 1)); f.savefig(out / "fig_learning_curves.png", dpi=140); plt.close(f)


if __name__ == "__main__":
    main()
