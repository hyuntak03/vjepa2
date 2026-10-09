#!/usr/bin/env python3
"""H1c — h1b 를 **궤적 단위 split** 으로 다시 한다 (적대 검증 2026-09-24 의 critical 1 대응).

왜: RollOut_v3 평지 세 법칙 1,176 clip 은 운동 궤적이 **42 개뿐**이다 (법칙마다 primary 7 속도 × secondary ±1,
   궤적마다 모양 7 × 환경 4 = 28 복사본). h1b 의 clip 단위 반반 split 과 clip 단위 λ-CV 는 같은 궤적을 train·test 에
   동시에 넣어 "처음 보는 외형, 이미 본 운동" 을 쟀다 (CLAUDE.md §1-5 block split 과 같은 누수).
   여기서는 group = law|primary|secondary 로 **split 도 λ-CV fold 도** 묶는다.

무엇 (h1b 와 같은 특징·표적, split 만 바꿈 + 기준선 추가):
  특징   문맥 마지막 3 튜블릿의 진실 물체 칸 3×3 평균 (3×D) — encoder 블록 l (짝수 + 31), encF (= z), tgt (문맥만 본 target),
         predictor 문맥 토큰 l. 모두 v3_h1 추출물 그대로.
  표적   v = |v_T| (원값, 읽기 확인),  a_T = x_T − 2x_{T−1} + x_{T−2},  r_k = x_{T+k} − x_T − k·v_T (k=4,8), 진행 방향 부호 정렬.
         a · r4 · r8 은 train 에서만 맞춘 poly2(x_T, |v_T|) OLS 로 잔차화 (h1b 와 같음).
  기준선 (같은 ridge · 같은 split):
         저차원 기준선은 모두 [RBF(σ) + 선형] 커널 ridge, σ ∈ {0.15, 0.3, 0.6, 1.2} (표준화 단위) 와 λ 를 inner CV 로 함께 고른다
         — "이 몇 개 수의 임의의 매끈한 함수" (일부러 유연하게: 기준선이 강할수록 비교가 보수적).
         B_state     (x_T, |v_T|) — 잔차화 poly2 가 못 뺀 위치·속도 몫 (법칙 × x_T 교락) 이 여기서 잡힌다.
         B_entry     (x_T, |v_T|, t_entry)  t_entry = 창 시작부터 물체가 처음 화면에 든 프레임 (C8·C16 은 모든 clip 0 → B_state 와 같음).
         B_xfirst    (x_T, |v_T|, x_first [, t_first]) — 창 안에서 처음 보인 위치 (와 시각). 관측 가능한 역사 1–2 개 수.
         B_kin3      참 (x_T, |v_T|, a_T) — 특징 창 3 튜블릿의 국소 운동 상태를 완벽히 담았을 때의 천장
                     (a_T 는 CSV 0.1 px 반올림 탓에 잡음이 있다 — r8 천장이 1 이 아닌 이유).
         (참 (x_T, x_{T−1}, x_{T−2}) 원값 커널은 셋이 거의 공선이라 ridge 가 a 방향을 못 살린다 — 천장으로 쓰지 않는다.)
         shuffle     encF 행을 clip 단위로 섞음.
  분류   법칙 3-way (one-hot ridge, λ 는 세 열 합 MSE) — 전 속도 / 속도 맞춤 구간 (|v_T| 5 분위 중 세 법칙 모두 ≥ 10 clip).
         속도 맞춤 구간의 사전 기준선 = train 의 구간별 최빈 법칙을 test 에 그대로 적용한 정확도.
  split  half : 법칙마다 14 궤적 중 7 을 train (층화), 20 회.  mean · sd · 5/50/95 백분위 · min/max.
         xfit : 법칙마다 궤적을 7 fold (fold 당 법칙별 2 궤적) — out-of-fold 예측을 모아 pooled R², 반복 3 회,
                95 % CI = 궤적 bootstrap (법칙 층화, 1000 회).
  ridge  dual (Gram) + 고윳값 분해로 λ 경로 (λ = 1e-2 … 1e6). 특징 표준화는 전체 clip 평균·표준편차 (라벨 무관),
         centering 은 train 안에서 커널로. λ-CV 는 train 궤적 5 fold (법칙 층화) — **궤적이 fold 를 넘지 않는다.**
  R²     1 − Σ(ŷ−y)² / Σ(y−ȳ)²  (test clip 전부, ȳ = test 평균).  한 궤적 = 28 clip 이 같은 y 를 가지므로 유효 n 은 궤적 수다.

  auto_research/scripts/srun6.sh 16 64G python auto_research/scripts/h1c_traj_split.py --C 16
  → exp_results/h1/h1c_traj_C{C}.json
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
IN_H1 = Path("/data2/local_datasets/world/world_analysis/cache/auto_research/v3_h1")
OUT = ROOT / "auto_research/exp_results/h1"
LAWS = ("flat_v", "flat_a", "flat_d")
LAMS = 10.0 ** np.arange(-2, 7)


def arr(s):
    return np.array([float(v) for v in s.split()], dtype=np.float64)


def tub(X, f0):
    return (X[:, f0] + X[:, f0 + 1]) / 2


def poly2(S):
    cols = [S] + [S[:, i:i + 1] * S[:, j:j + 1] for i in range(S.shape[1]) for j in range(i, S.shape[1])]
    return np.concatenate(cols, 1)


def stdz(F):
    F = np.asarray(F, np.float64)
    return (F - F.mean(0)) / (F.std(0) + 1e-6)


# ---------------------------------------------------------------- 커널 ridge (궤적 묶음 λ-CV)
def _center(K, tr, te):
    Ktt = K[np.ix_(tr, tr)]; m = Ktt.mean(0); mm = m.mean()
    Kc = Ktt - m[:, None] - m[None, :] + mm
    Kx = K[np.ix_(te, tr)]
    Kxc = Kx - Kx.mean(1, keepdims=True) - m[None, :] + mm
    return Kc, Kxc


def _path(K, tr, te, Y):
    """λ 경로 예측 (L, n_te, m). Y 는 train 평균으로 중심화해 넣고 다시 더한다."""
    Kc, Kxc = _center(K, tr, te)
    s, U = np.linalg.eigh(Kc); s = np.clip(s, 0, None)
    ym = Y[tr].mean(0); UtY = U.T @ (Y[tr] - ym)
    KU = Kxc @ U
    return np.stack([KU @ (UtY / (s + lam)[:, None]) + ym for lam in LAMS])


def strat_folds(tid, law_of_tid, tids, k, rng):
    """tids (궤적 id 목록) 를 법칙 층화로 k fold 에 배정 → {tid: fold}."""
    fo = {}
    for l in LAWS:
        tl = [t for t in tids if law_of_tid[t] == l]
        for r, t in enumerate(rng.permutation(tl)):
            fo[t] = r % k
    return fo


def fit_grouped(K, tr, te, Y, tid, law_of_tid, rng, joint=False, k_inner=5):
    """tr/te: 정수 인덱스. K 는 커널 하나 또는 목록 (목록이면 (커널, λ) 를 함께 고른다).
    λ 는 train 궤적 k_inner fold 로 (열마다, joint 면 열 합) — 궤적이 fold 를 넘지 않는다."""
    Ks = K if isinstance(K, list) else [K]
    ttr = np.unique(tid[tr]); fo = strat_folds(tid, law_of_tid, ttr, k_inner, rng)
    f_of = np.array([fo[t] for t in tid[tr]])
    err = np.zeros((len(Ks), len(LAMS), Y.shape[1]))
    for f in range(k_inner):
        itr, iva = tr[f_of != f], tr[f_of == f]
        if len(iva) == 0:
            continue
        for q, Kq in enumerate(Ks):
            P = _path(Kq, itr, iva, Y)
            err[q] += ((P - Y[iva][None]) ** 2).sum(1)
    E = err.reshape(-1, Y.shape[1])                      # (Ks·L, m)
    if joint:
        b = np.full(Y.shape[1], int(E.sum(1).argmin()))
    else:
        b = E.argmin(0)
    qk, lk = b // len(LAMS), b % len(LAMS)
    out = np.zeros((len(te), Y.shape[1]))
    for q in np.unique(qk):
        P = _path(Ks[q], tr, te, Y); cols = np.where(qk == q)[0]
        out[:, cols] = P[lk[cols], :, cols].T
    return out, LAMS[lk]


def r2(p, y):
    return float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def summ(v):
    v = np.asarray(v, float)
    return {"mean": round(float(v.mean()), 4), "sd": round(float(v.std()), 4), "p05": round(float(np.percentile(v, 5)), 4),
            "median": round(float(np.median(v)), 4), "p95": round(float(np.percentile(v, 95)), 4),
            "min": round(float(v.min()), 4), "max": round(float(v.max()), 4), "n_split": len(v)}


# ---------------------------------------------------------------- 데이터
def load_flat(split, C_for_entry=None):
    meta = json.load(open(IN_H1 / "meta.json")); ids = meta["video_ids"]
    rows = {r["video_id"]: r for r in csv.DictReader(open(ROOT / "data_csv/rollout_v3/index_probe.csv"))}
    R = [rows[v] for v in ids]
    scen = np.array([r["scenario"] for r in R]); plaus = np.array([r["plausible"] == "1" for r in R])
    X = np.stack([arr(r["px_x_by_sample"]) for r in R])
    ok = np.stack([arr(r["in_frame_by_sample"]) for r in R]) > 0
    xT, xT1, xT2 = tub(X, split - 2), tub(X, split - 4), tub(X, split - 6)
    v = xT - xT1; a = xT - 2 * xT1 + xT2
    keep = plaus & np.isin(scen, LAWS) & ok[:, split - 6:split].all(1)
    tg = {"a": a}
    for k in (4, 8):
        f0 = split - 2 + 2 * k; tg[f"r{k}"] = tub(X, f0) - xT - k * v; keep &= ok[:, f0] & ok[:, f0 + 1]
    idx = np.where(keep)[0]
    sgn = np.sign(v[idx]); sgn[sgn == 0] = 1
    traj = np.array([f"{R[i]['scenario']}|{R[i]['primary']}|{R[i]['secondary']}" for i in idx])
    ut = sorted(set(traj)); tid = np.array([ut.index(t) for t in traj])
    law_of_tid = {ut.index(t): t.split("|")[0] for t in ut}
    D = dict(idx=idx, law=scen[idx], tid=tid, law_of_tid=law_of_tid, n_traj=len(ut),
             xT=xT[idx], xT1=xT1[idx], xT2=xT2[idx], spd=np.abs(v[idx]), sgn=sgn,
             Y={k: tg[k][idx] * sgn for k in tg}, X=X[idx], ok=ok[idx])
    return D


def history_cues(D, split, C):
    """창 [split−C, split) 안에서 물체가 처음 화면에 든 프레임 (창 시작 기준) 과 그때의 x (진행 방향 부호)."""
    w0 = split - C
    okw = D["ok"][:, w0:split]
    tf = okw.argmax(1).astype(float)
    xf = D["X"][np.arange(len(tf)), w0 + tf.astype(int)] * D["sgn"]
    return tf, xf


def lowdim_kernels(V, sigmas=(0.15, 0.3, 0.6, 1.2)):
    """저차원 변수 V (n, d) → [RBF(σ) + 선형] 커널 목록 (변수는 표준화). inner CV 가 σ 와 λ 를 함께 고른다.
    "이 몇 개 수의 임의의 매끈한 함수" 로 표적을 얼마나 맞히나 — 기준선을 일부러 유연하게 (보수적 비교)."""
    Z = stdz(V); lin = Z @ Z.T
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    return ("kernels", [np.exp(-0.5 * d2 / s ** 2) * len(Z[0]) + lin for s in sigmas])


# ---------------------------------------------------------------- 한 특징 평가
def evaluate(F, D, n_half=20, n_xfit=3, n_boot=1000, seed0=100, do_xfit=True):
    """F: 특징 행렬 (선형 커널) 또는 ("kernels", [K, ...]) (저차원 기준선 — 커널과 λ 를 inner CV 로 함께 고른다)."""
    n = len(D["idx"]); tid, law, lot = D["tid"], D["law"], D["law_of_tid"]
    if isinstance(F, tuple) and F[0] == "kernels":
        K = F[1]
    else:
        K = stdz(F); K = K @ K.T
    st = poly2(np.stack([D["xT"], D["spd"]], 1))
    oh = np.stack([(law == l).astype(float) for l in LAWS], 1)
    qs = np.percentile(D["spd"], [20, 40, 60, 80]); vb = np.digitize(D["spd"], qs)
    good = np.zeros(n, bool)
    for b in range(5):
        if all(((law == l) & (vb == b)).sum() >= 10 for l in LAWS):
            good |= vb == b
    tnames = ["v", "a", "r4", "r8"]

    def targets(tr):
        """train 에서 맞춘 poly2 상태 회귀로 잔차화한 표적 행렬 (n, 4)."""
        cols = [D["spd"]]
        A = np.c_[np.ones(len(tr)), st[tr]]
        for k in ("a", "r4", "r8"):
            y = D["Y"][k]; w = np.linalg.lstsq(A, y[tr], rcond=None)[0]
            cols.append(y - np.c_[np.ones(n), st] @ w)
        return np.stack(cols, 1)

    def prior_acc(tr, te):
        """구간별 최빈 법칙 (전체 clip 기준 — 기술적 기준선. train 기준으로 하면 궤적 split 이 상보적이라 chance 아래로 간다)."""
        pr = {}
        for b in np.unique(vb[te]):
            c = [((law == l) & (vb == b)).sum() for l in LAWS]
            pr[b] = int(np.argmax(c))
        return float(np.mean([pr[vb[i]] == oh[i].argmax() for i in te]))

    out = {"half": {t: [] for t in tnames}, "half_acc_all": [], "half_acc_vm": [], "half_prior_vm": [], "half_lam": []}
    ut = np.arange(D["n_traj"])
    for s in range(n_half):
        rng = np.random.default_rng(seed0 + s)
        pick = []
        for l in LAWS:
            tl = [t for t in ut if lot[t] == l]; pick += list(rng.permutation(tl)[: len(tl) // 2])
        trm = np.isin(tid, pick); tr, te = np.where(trm)[0], np.where(~trm)[0]
        Y = targets(tr)
        P, lam = fit_grouped(K, tr, te, Y, tid, lot, rng)
        for j, t in enumerate(tnames):
            out["half"][t].append(r2(P[:, j], Y[te, j]))
        out["half_lam"].append(lam.tolist())
        Pc, _ = fit_grouped(K, tr, te, oh, tid, lot, rng, joint=True)
        hit = Pc.argmax(1) == oh[te].argmax(1)
        out["half_acc_all"].append(float(hit.mean()))
        g = good[te]; out["half_acc_vm"].append(float(hit[g].mean()))
        out["half_prior_vm"].append(prior_acc(tr, te[g]))
    res = {"half": {t: summ(v) for t, v in out["half"].items()},
           "half_acc_all": summ(out["half_acc_all"]), "half_acc_vm": summ(out["half_acc_vm"]),
           "half_prior_vm": summ(out["half_prior_vm"]),
           "lam_at_edge_frac": round(float(np.mean([(np.array(l) == LAMS[0]) | (np.array(l) == LAMS[-1]) for l in out["half_lam"]])), 3)}
    if not do_xfit:
        return res
    # ---- xfit: 7 fold (법칙마다 2 궤적) out-of-fold
    reps = []
    for rep in range(n_xfit):
        rng = np.random.default_rng(seed0 + 1000 + rep)
        fo = strat_folds(tid, lot, ut, 7, rng); f_of = np.array([fo[t] for t in tid])
        PY = np.zeros((n, 4)); TY = np.zeros((n, 4)); PC = np.zeros(n, int)
        for f in range(7):
            tr, te = np.where(f_of != f)[0], np.where(f_of == f)[0]
            Y = targets(tr)
            P, _ = fit_grouped(K, tr, te, Y, tid, lot, rng)
            PY[te] = P; TY[te] = Y[te]
            Pc, _ = fit_grouped(K, tr, te, oh, tid, lot, rng, joint=True); PC[te] = Pc.argmax(1)
        reps.append((PY, TY, PC))
    ytrue = oh.argmax(1)
    xf = {t: round(float(np.mean([r2(PY[:, j], TY[:, j]) for PY, TY, _ in reps])), 4) for j, t in enumerate(tnames)}
    xf["acc_all"] = round(float(np.mean([(PC == ytrue).mean() for _, _, PC in reps])), 4)
    xf["acc_vm"] = round(float(np.mean([(PC[good] == ytrue[good]).mean() for _, _, PC in reps])), 4)
    # 궤적 bootstrap (법칙 층화)
    rng = np.random.default_rng(7)
    members = {t: np.where(tid == t)[0] for t in ut}
    B = {t: [] for t in tnames + ["acc_all"]}
    for _ in range(n_boot):
        sel = []
        for l in LAWS:
            tl = [t for t in ut if lot[t] == l]; sel += list(rng.choice(tl, len(tl)))
        ii = np.concatenate([members[t] for t in sel])
        for j, t in enumerate(tnames):
            B[t].append(np.mean([r2(PY[ii, j], TY[ii, j]) for PY, TY, _ in reps]))
        B["acc_all"].append(np.mean([(PC[ii] == ytrue[ii]).mean() for _, _, PC in reps]))
    xf["ci95"] = {t: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] for t, v in B.items()}
    res["xfit"] = xf
    return res


def curve(F, D, Ks=(2, 3, 7, 14), reps=2, seed0=500):
    """학습 곡선: 법칙마다 궤적을 K fold 로 나눠 out-of-fold pooled R² (train 궤적/법칙 = 14·(K−1)/K ≈ 7, 9.3, 12, 13).
    잔차화·λ-CV 는 evaluate 와 같다. 반환 {K: {"train_per_law":…, "v"/"a"/"r4"/"r8": 반복 평균}}."""
    n = len(D["idx"]); tid, lot = D["tid"], D["law_of_tid"]
    if isinstance(F, tuple) and F[0] == "kernels":
        K = F[1]
    else:
        K = stdz(F); K = K @ K.T
    st = poly2(np.stack([D["xT"], D["spd"]], 1)); ut = np.arange(D["n_traj"])
    out = {}
    for k in Ks:
        rs = {t: [] for t in ("v", "a", "r4", "r8")}
        for rep in range(reps):
            rng = np.random.default_rng(seed0 + 10 * k + rep)
            fo = strat_folds(tid, lot, ut, k, rng); f_of = np.array([fo[t] for t in tid])
            PY = np.zeros((n, 4)); TY = np.zeros((n, 4))
            for f in range(k):
                tr, te = np.where(f_of != f)[0], np.where(f_of == f)[0]
                cols = [D["spd"]]; A = np.c_[np.ones(len(tr)), st[tr]]
                for kk in ("a", "r4", "r8"):
                    y = D["Y"][kk]; w = np.linalg.lstsq(A, y[tr], rcond=None)[0]; cols.append(y - np.c_[np.ones(n), st] @ w)
                Y = np.stack(cols, 1)
                P, _ = fit_grouped(K, tr, te, Y, tid, lot, rng)
                PY[te] = P; TY[te] = Y[te]
            for j, t in enumerate(("v", "a", "r4", "r8")):
                rs[t].append(r2(PY[:, j], TY[:, j]))
        out[str(k)] = {"train_per_law": round(14 * (k - 1) / k, 1), **{t: round(float(np.mean(v)), 4) for t, v in rs.items()},
                       "r8_reps": [round(float(x), 4) for x in rs["r8"]]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=int, required=True)
    ap.add_argument("--n-half", type=int, default=20)
    ap.add_argument("--n-xfit", type=int, default=3)
    ap.add_argument("--no-xfit", action="store_true")
    ap.add_argument("--curve", nargs="*", default=None, help="학습 곡선만 (특징 이름 목록) → json 의 curve 키")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    split = json.load(open(IN_H1 / "meta.json"))["split"]
    C = a.C
    D = load_flat(split)
    n = len(D["idx"]); idx = D["idx"]
    tf, xfirst = history_cues(D, split, C)
    print(f"C{C} clips {n} trajectories {D['n_traj']}  t_entry>0 frac {float((tf > 0).mean()):.3f}", flush=True)
    enc = np.load(IN_H1 / f"enc_C{C}.npy", mmap_mode="r")
    pctx = np.load(IN_H1 / f"pctx_C{C}.npy", mmap_mode="r")
    encF = np.load(IN_H1 / f"encF_C{C}.npy")[idx].reshape(n, -1)
    feats = {
        "B_state": lambda: lowdim_kernels(np.stack([D["xT"], D["spd"]], 1)),
        "B_entry": lambda: lowdim_kernels(np.stack([D["xT"], D["spd"], tf], 1) if tf.std() > 0 else np.stack([D["xT"], D["spd"]], 1)),
        "B_xfirst": lambda: lowdim_kernels(np.stack([D["xT"], D["spd"], xfirst] + ([tf] if tf.std() > 0 else []), 1)),
        "B_kin3": lambda: lowdim_kernels(np.stack([D["xT"], D["spd"], D["Y"]["a"]], 1)),
        "shuffle": lambda: encF[np.random.default_rng(1).permutation(n)],
        "encF": lambda: encF,
        "tgt": lambda: np.load(IN_H1 / f"tgt_C{C}.npy")[idx].reshape(n, -1),
    }
    for l in list(range(0, 32, 2)) + [31]:
        feats[f"enc{l}"] = (lambda l=l: np.asarray(enc[idx, :, l], np.float64).reshape(n, -1))
    for l in (0, 2, 4, 6, 8, 10, 11):
        feats[f"pctx{l}"] = (lambda l=l: np.asarray(pctx[idx, :, l], np.float64).reshape(n, -1))
    XFIT = set(feats) | {"B_entry"}
    out_p = OUT / f"h1c_traj_C{C}.json"
    res = json.load(open(out_p)) if (a.only and out_p.exists()) else {}
    res["_info"] = {"n_clip": n, "n_traj": D["n_traj"], "t_entry_pos_frac": round(float((tf > 0).mean()), 4),
                    "n_half": a.n_half, "n_xfit": a.n_xfit, "lams": LAMS.tolist()}
    if tf.std() == 0:
        feats.pop("B_entry")
    if a.curve is not None:
        res = json.load(open(OUT / f"h1c_traj_C{C}.json"))
        cv = res.get("curve", {})
        for name in a.curve:
            if name not in feats:
                continue
            cv[name] = curve(feats[name](), D)
            print(f"C{C} curve {name:12s} " + "  ".join(f"K{k}(tr {v['train_per_law']}) r8 {v['r8']:+.3f} a {v['a']:+.3f}" for k, v in cv[name].items())
                  + f"  {time.time() - t0:.0f}s", flush=True)
            res["curve"] = cv
            json.dump(res, open(OUT / f"h1c_traj_C{C}.json", "w"), indent=1)
        return                                   # C8·C16: 모든 clip 이 창 시작부터 화면 안 → B_state 와 같다
    for name, fx in feats.items():
        if a.only and name not in a.only:
            continue
        r = evaluate(fx(), D, n_half=a.n_half, n_xfit=a.n_xfit, do_xfit=(name in XFIT) and not a.no_xfit)
        res[name] = r
        h = r["half"]
        xs = f" | xfit r8 {r['xfit']['r8']:+.3f} {r['xfit']['ci95']['r8']} a {r['xfit']['a']:+.3f} acc {r['xfit']['acc_all']:.3f}" if "xfit" in r else ""
        print(f"C{C} {name:12s} half r8 {h['r8']['mean']:+.3f}±{h['r8']['sd']:.3f} [{h['r8']['p05']:+.2f},{h['r8']['p95']:+.2f}]"
              f" r4 {h['r4']['mean']:+.3f} a {h['a']['mean']:+.3f} v {h['v']['mean']:+.3f}"
              f" acc {r['half_acc_all']['mean']:.3f} vm {r['half_acc_vm']['mean']:.3f} (prior {r['half_prior_vm']['mean']:.3f}){xs}"
              f"  {time.time() - t0:.0f}s", flush=True)
        json.dump(res, open(out_p, "w"), indent=1)
    print("saved", out_p, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
