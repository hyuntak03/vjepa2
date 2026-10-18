#!/usr/bin/env python3
"""predictor_bottleneck.py 산출 분석 (2026-10-12). numpy 만.

A1 걸음 이득 (행동): 출력 블록 t 의 읽은 위치 X 와 '마지막 입력 블록' 위치 P_last 의 차를 운동 방향으로 투영 ÷ 참 한 걸음
     g = ((X − P_last)·d) / ((T_t − T_{t−1})·d)      1 = 한 블록 분 정확히 진행, 0 = 마지막 입력 그 자리
   입력 운동 m = ((P_last − P_2nd)·d) / 참 한 걸음   (입력 마지막 두 블록에 보이는 운동)
   P 는 입력 블록이 참이면 라벨, 자기 예측이면 그 블록을 같은 자로 읽은 값. roll 의 속도 (이웃 출력 차) 도 같이.
   '있다' (1 − P(없음) > 모델 자의 val 오탐 5 % 문턱) 인 블록만.
A2 믿음 (내부): 걸음 · 조건 · 특징 (st = 그리기 전 384 · r = 그린 1280, 각 g = 블록 평균 · l = 참 물체 3×3 평균) 마다
   ridge (train clip 으로 맞추고 test clip R²) → 표적: 속도 벡터 (px/블록), 위치 (px), 모양 · 색 (one-hot ridge → 정확도).
   천장 = 참 h 블록 · 문맥 마지막 블록. λ 는 train 안 5-fold.
A3 오차 몫: |출력 − 참| 의 물체 3×3 vs 그 밖, 복사 (문맥 마지막 블록) 대비.
A4 표류: 1 − max cos(출력 토큰, 문맥 토큰) — 출력 · 참 미래 · (복사 0).
A5 학습 신호: rollout 블록 j 의 기대 손실 가중치 E[1{K ≥ j} / K] (AR K ∈ {1,2,4,8} · AR-state K ∈ {4,6,8}).

  python predictor_bottleneck_analyze.py --names ar40 ar_state40 release
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "z_research/VOBench_v1/bottleneck"
META = Path("/data2/local_datasets/world/world_analysis/VOB_readout_primitive/metadata.csv")
INDEX = REPO / "data_csv/rollout_v2_vob_readout_primitive/index.csv"
C, K = 8, 8
SHAPES = ["sphere", "cube", "cylinder", "cone", "torus", "capsule", "pyramid"]


def load(name):
    Z = {}
    for f in sorted((OUT / name).glob("shard*.npz")):
        z = np.load(f, allow_pickle=True)
        for k in z.files:
            Z.setdefault(k, []).append(z[k])
    return {k: np.concatenate(v) for k, v in Z.items()}


def thr(name):
    return json.load(open(REPO / f"z_research/VOBench_v1/readout/identity_vobprim_{name}/summary.json"))["reps"]["p"]["thr_val_fpr5"]


# ------------------------------------------------------------------ A1
def step_gain(Z, name):
    th = thr(name); T = Z["truth"]; n = len(T)
    d = T[:, 15] - T[:, 0]; d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    step = np.einsum("nj,nj->n", T[:, 9] - T[:, 8], d)                       # 등속이라 한 걸음 = 상수
    ok = np.abs(step) > 1.0
    pres = lambda c, k: (1 - Z[f"{c}_pn{k}"]) > th
    X = lambda c, k: Z[f"{c}_xy{k}"]
    res = {}
    for c in ("roll", "tf", "injL", "injO"):
        if f"{c}_xy{K}" not in Z:
            continue
        rows = {}
        for k in range(1, K + 1):
            if f"{c}_xy{k}" not in Z:
                continue
            t = C + k - 1
            if c == "roll":
                last = T[:, t - 1] if k == 1 else X("roll", k - 1)
                prev = T[:, t - 2] if k <= 2 else X("roll", k - 2)
                okin = np.ones(n, bool) if k == 1 else pres("roll", k - 1)
                if k >= 3: okin &= pres("roll", k - 2)
            elif c == "tf":
                last, prev, okin = T[:, t - 1], T[:, t - 2], np.ones(n, bool)
            elif c == "injL":
                last = T[:, t - 1]; prev = X("roll", k - 2) if k >= 3 else T[:, t - 2]
                okin = pres("roll", k - 2) if k >= 3 else np.ones(n, bool)
            else:                                                             # injO: 마지막 입력 = tf 가 낸 블록 t−1
                last = X("tf", k - 1); prev = T[:, t - 2]; okin = pres("tf", k - 1)
            m = ok & okin & pres(c, k)
            g = np.einsum("nj,nj->n", X(c, k) - last, d) / step
            mi = np.einsum("nj,nj->n", last - prev, d) / step
            err_px = np.linalg.norm(X(c, k) - T[:, t], axis=1)
            lag = np.einsum("nj,nj->n", X(c, k) - T[:, t], d) / step               # 운동 방향 부호 오차 (걸음 단위, − = 뒤처짐)
            rows[k] = dict(n=int(m.sum()), gain=float(np.median(g[m])), gain_mean=float(np.mean(g[m])), lag=float(np.median(lag[m])),
                           input_motion=float(np.median(mi[m])), err_cells=float(np.median(err_px[m]) / 18.0),
                           present=float(pres(c, k)[ok].mean()))
        res[c] = rows
    # roll 의 속도 (이웃 출력 차) — horizon 분석과 같은 정의
    vel = {}
    for k in range(2, K + 1):
        m = ok & pres("roll", k) & pres("roll", k - 1)
        v = np.einsum("nj,nj->n", X("roll", k) - X("roll", k - 1), d) / step
        vel[k] = float(np.median(v[m]))
    res["roll_velocity"] = vel
    return res


# ------------------------------------------------------------------ A2
def ridge_fit(Xtr, Ytr, Xte, lams=(1e-1, 1, 10, 100, 1e3, 1e4)):
    """다중 표적 ridge. λ 는 train 안 5-fold (표적 전체 SSE 합) — 접마다 SVD 한 번으로 모든 λ 를 본다."""
    Xtr, Xte, Ytr = Xtr.astype(np.float32), Xte.astype(np.float32), Ytr.astype(np.float32)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A, Bt = (Xtr - mu) / sd, (Xte - mu) / sd
    idx = np.arange(len(A)); rng = np.random.default_rng(0); rng.shuffle(idx); folds = np.array_split(idx, 5)
    sse = np.zeros(len(lams))
    for f in folds:
        tr = np.setdiff1d(idx, f); ym = Ytr[tr].mean(0)
        U, s, Vt = np.linalg.svd(A[tr], full_matrices=False); UtY = U.T @ (Ytr[tr] - ym)
        for j, lam in enumerate(lams):
            W = Vt.T @ ((s / (s ** 2 + lam))[:, None] * UtY)
            sse[j] += ((A[f] @ W + ym - Ytr[f]) ** 2).sum()
    bl = lams[int(np.argmin(sse))]; ym = Ytr.mean(0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    W = Vt.T @ ((s / (s ** 2 + bl))[:, None] * (U.T @ (Ytr - ym)))
    return Bt @ W + ym, bl


def r2(pred, y):
    return float(1 - ((pred - y) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


def labels(Z):
    rows = list(csv.DictReader(INDEX.open())); meta = {r["name"]: r for r in csv.DictReader(META.open())}
    vid = [rows[i]["video_id"] for i in Z["row"]]
    shp = np.array([meta[v]["shape"] for v in vid]); col = np.array([meta[v]["color"] for v in vid])
    T = Z["truth"]; vel = (T[:, 15] - T[:, 0]) / 15.0
    onehot = lambda a: (a[:, None] == np.unique(a)[None]).astype(np.float32)
    return vel, shp, col, onehot(shp), onehot(col)


def probes(Z, name):
    vel, shp, col, Ys, Yc = labels(Z)
    tr, te = Z["part"] == "train", Z["part"] == "test"
    T = Z["truth"]
    out = {}

    vs = vel.std(0) + 1e-6                                     # 표적 척도를 맞춰 한 번에 푼다 (λ 선택이 한 표적에 쏠리지 않게)
    Yall = np.concatenate([vel / vs, Ys, Yc], 1); ns = Ys.shape[1]

    def one(F, key):
        P, _ = ridge_fit(F[tr], Yall[tr], F[te])
        pv, ps, pc = P[:, :2] * vs, P[:, 2:2 + ns], P[:, 2 + ns:]
        out[key] = dict(vel_r2=r2(pv, vel[te]), shape_acc=float((ps.argmax(1) == Ys[te].argmax(1)).mean()),
                        color_acc=float((pc.argmax(1) == Yc[te].argmax(1)).mean()))
    for k in range(1, K + 1):
        t = C + k - 1
        one(np.concatenate([Z[f"h_g{k}"], Z[f"h_l{k}"]], 1), f"truth_h/k{k}")
        for c in ("roll", "tf", "injL", "injO"):
            if f"{c}_st_g{k}" not in Z:
                continue
            one(np.concatenate([Z[f"{c}_st_g{k}"], Z[f"{c}_st_l{k}"]], 1), f"{c}_st/k{k}")
            one(np.concatenate([Z[f"{c}_r_g{k}"], Z[f"{c}_r_l{k}"]], 1), f"{c}_r/k{k}")
            if c == "roll":
                one(Z[f"{c}_st_g{k}"], f"{c}_st_globalonly/k{k}")
    if "ctx_g" in Z:
        one(np.concatenate([Z["ctx_g"], Z["ctx_l"]], 1), "context_end_h")
    out["_chance"] = dict(shape=float(1 / Ys.shape[1]), color=float(1 / Yc.shape[1]))
    return out


# ------------------------------------------------------------------ A3 · A4
def errors(Z):
    out = {}
    for c in ("roll", "tf"):
        if f"{c}_err_obj1" not in Z:
            continue
        rows = {}
        for k in range(1, K + 1):
            eo, eb = Z[f"{c}_err_obj{k}"], Z[f"{c}_err_bg{k}"]
            co, cb = Z[f"err_copy_obj{k}"], Z[f"err_copy_bg{k}"]
            rows[k] = dict(obj=float(eo.mean()), bg=float(eb.mean()), copy_obj=float(co.mean()), copy_bg=float(cb.mean()),
                           win_obj=float((eo < co).mean()), win_bg=float((eb < cb).mean()),
                           obj_share=float((eo * 9 / (eo * 9 + eb * 247)).mean()),
                           drift=float(Z[f"{c}_nov_all{k}"].mean()), drift_obj=float(Z[f"{c}_nov_obj{k}"].mean()),
                           true_novelty=float(Z[f"nov_true_all{k}"].mean()), true_novelty_obj=float(Z[f"nov_true_obj{k}"].mean()))
        out[c] = rows
    return out


def train_signal():
    def w(Ks):
        return {j: float(np.mean([(1.0 / k if k >= j else 0.0) for k in Ks])) for j in range(1, 9)}
    return {"ar (K∈{1,2,4,8})": w([1, 2, 4, 8]), "ar_state (K∈{4,6,8})": w([4, 6, 8])}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--names", nargs="+", default=["ar40", "ar_state40", "release"]); a = ap.parse_args()
    res = {"train_signal": train_signal()}
    for name in a.names:
        if not (OUT / name).exists():
            print("없음", name); continue
        Z = load(name)
        res[name] = dict(n=int(len(Z["row"])), n_test=int((Z["part"] == "test").sum()), step_gain=step_gain(Z, name), errors=errors(Z), probes=probes(Z, name))
        print(f"\n===== {name}  (clip {res[name]['n']}, test {res[name]['n_test']})")
        sg = res[name]["step_gain"]
        print("걸음 k            " + " ".join(f"{k:>6d}" for k in range(1, K + 1)))
        for c in ("roll", "tf", "injL", "injO"):
            if c in sg:
                print(f"gain {c:5s}       " + " ".join(f"{sg[c][k]['gain']:6.2f}" if k in sg[c] else "     ·" for k in range(1, K + 1)))
                print(f"  input motion   " + " ".join(f"{sg[c][k]['input_motion']:6.2f}" if k in sg[c] else "     ·" for k in range(1, K + 1)))
                print(f"  err (칸)        " + " ".join(f"{sg[c][k]['err_cells']:6.2f}" if k in sg[c] else "     ·" for k in range(1, K + 1)))
                print(f"  lag (걸음)       " + " ".join(f"{sg[c][k]['lag']:6.2f}" if k in sg[c] else "     ·" for k in range(1, K + 1)))
                print(f"  present        " + " ".join(f"{sg[c][k]['present']:6.2f}" if k in sg[c] else "     ·" for k in range(1, K + 1)))
        print("roll 속도 (이웃)  " + " ".join(f"{sg['roll_velocity'].get(k, np.nan):6.2f}" for k in range(1, K + 1)))
        pr = res[name]["probes"]
        for key in ("truth_h", "roll_st", "roll_st_globalonly", "roll_r", "tf_st", "tf_r", "injL_st", "injO_st"):
            if f"{key}/k{K}" in pr or f"{key}/k1" in pr:
                for m in ("vel_r2", "shape_acc", "color_acc"):
                    print(f"{key:20s} {m:9s} " + " ".join(f"{pr[f'{key}/k{k}'][m]:6.2f}" if f"{key}/k{k}" in pr else "     ·" for k in range(1, K + 1)))
        if "context_end_h" in pr: print("context_end_h", pr["context_end_h"], "chance", pr["_chance"])
        er = res[name]["errors"]
        for c in er:
            for m in ("obj", "copy_obj", "bg", "copy_bg", "win_obj", "win_bg", "obj_share", "drift", "true_novelty", "drift_obj", "true_novelty_obj"):
                print(f"err {c:4s} {m:16s} " + " ".join(f"{er[c][k][m]:6.3f}" for k in range(1, K + 1)))
    json.dump(res, open(OUT / "analysis.json", "w"), indent=1)
    print("\n학습 신호 (블록 j 기대 가중치):", json.dumps(res["train_signal"]))


if __name__ == "__main__":
    main()
