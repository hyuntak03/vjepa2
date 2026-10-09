#!/usr/bin/env python3
"""H1b — 가속은 z 에 있는가: 평지 세 법칙 (flat_v / flat_a / flat_d) 로 공정하게.

왜 평지 셋만: 세 법칙은 **장면이 같고 운동 윤곽 (가속 부호) 만 다르다** — 경사면처럼 가속을 알려 주는 장면 단서가 없다.
한 법칙 안에서는 가속이 거의 일정해 법칙 안 회귀는 잡음만 잰다 (h1_analyze_v3 의 within R² 음수; RollOutV3 §4 와 같은 함정).
그래서 두 검정:
  (1) 잔차 회귀: 표적 y ∈ {a_x, r4_x, r8_x} 를 먼저 참 상태 s = poly2(x_T, v_T) 로 회귀해 잔차 ỹ = y − ŝ(y) 를 만든다
      (위치·속도로 풀리는 몫을 뺀다). 층 특징 F 로 ỹ 를 ridge → held-out R²(ỹ).   F 가 가속을 담으면 > 0.
  (2) 속도 맞춤 법칙 분류: |v_T| 를 5 구간으로 나눠, 구간 안에서 법칙 3 개가 모두 ≥ 10 clip 인 곳만.
      ridge 분류 (one-hot) 정확도, 대조 = 같은 구간에서 poly2(x_T, v_T) 만 쓴 분류.   chance 1/3.
clip 단위 반반 split × 5 seed (법칙·속도 구간 층화).  특징: encoder 층 0..31, encF (= z), tgt (문맥만 본 target), pctx 0..11, state, shuffle.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h1_analyze_v3 import arr, tub, fit_predict, IN_DEFAULT, OUT, ROOT  # noqa: E402

LAWS = ("flat_v", "flat_a", "flat_d")


def poly2(S):
    cols = [S] + [S[:, i:i + 1] * S[:, j:j + 1] for i in range(S.shape[1]) for j in range(i, S.shape[1])]
    return np.concatenate(cols, 1)


def main():
    I = Path(IN_DEFAULT)
    meta = json.load(open(I / "meta.json")); ids, split = meta["video_ids"], meta["split"]
    rows = {r["video_id"]: r for r in csv.DictReader(open(ROOT / "data_csv/rollout_v3/index_probe.csv"))}
    R = [rows[v] for v in ids]
    scen = np.array([r["scenario"] for r in R]); plaus = np.array([r["plausible"] == "1" for r in R])
    X = np.stack([arr(r["px_x_by_sample"]) for r in R]); Y = np.stack([arr(r["px_y_by_sample"]) for r in R])
    ok = np.stack([arr(r["in_frame_by_sample"]) for r in R]) > 0
    xT, xT1, xT2 = tub(X, split - 2), tub(X, split - 4), tub(X, split - 6)
    v = xT - xT1; a = xT - 2 * xT1 + xT2
    keep = plaus & np.isin(scen, LAWS) & ok[:, split - 6:split].all(1)
    tg = {"a": a}
    for k in (4, 8):
        f0 = split - 2 + 2 * k; xk = tub(X, f0); tg[f"r{k}"] = xk - xT - k * v; keep &= ok[:, f0] & ok[:, f0 + 1]
    idx = np.where(keep)[0]; law = scen[idx]
    # 방향을 맞춘다: 진행 방향 + (왼쪽으로 가는 clip 은 부호를 뒤집는다)
    sgn = np.sign(v[idx]); sgn[sgn == 0] = 1
    st = poly2(np.stack([xT[idx], np.abs(v[idx])], 1))
    Ys = {k: tg[k][idx] * sgn for k in tg}
    spd = np.abs(v[idx]); qs = np.percentile(spd, [20, 40, 60, 80]); vb = np.digitize(spd, qs)
    print("clips", len(idx), {l: int((law == l).sum()) for l in LAWS})
    print("a_x 법칙별 평균 (px/튜블릿²):", {l: round(float(Ys['a'][law == l].mean()), 3) for l in LAWS},
          " 법칙 안 SD:", {l: round(float(Ys['a'][law == l].std()), 3) for l in LAWS})
    res = {}
    for C in meta["Cs"]:
        enc = np.load(I / f"enc_C{C}.npy", mmap_mode="r"); encF = np.load(I / f"encF_C{C}.npy")
        tgt = np.load(I / f"tgt_C{C}.npy"); pctx = np.load(I / f"pctx_C{C}.npy", mmap_mode="r")
        feats = {f"enc{l}": (lambda l=l: np.asarray(enc[idx, :, l], np.float64).reshape(len(idx), -1)) for l in range(0, 32, 2)}
        feats.update({"enc31": lambda: np.asarray(enc[idx, :, 31], np.float64).reshape(len(idx), -1),
                      "encF": lambda: encF[idx].reshape(len(idx), -1).astype(np.float64),
                      "tgt": lambda: tgt[idx].reshape(len(idx), -1).astype(np.float64),
                      "state": lambda: st,
                      "shuffle": lambda: encF[idx].reshape(len(idx), -1)[np.random.default_rng(1).permutation(len(idx))].astype(np.float64)})
        for l in range(0, 12, 2):
            feats[f"pctx{l}"] = (lambda l=l: np.asarray(pctx[idx, :, l], np.float64).reshape(len(idx), -1))
        feats["pctx11"] = lambda: np.asarray(pctx[idx, :, 11], np.float64).reshape(len(idx), -1)
        out = {}
        for name, fx in feats.items():
            F = fx(); r2s = {k: [] for k in Ys}; accs, accs_state = [], []
            for sd in range(5):
                rng = np.random.default_rng(sd)
                tr = np.zeros(len(idx), bool)
                for l in LAWS:
                    for b in range(5):
                        m = np.where((law == l) & (vb == b))[0]; tr[rng.permutation(m)[: len(m) // 2]] = True
                # (1) 잔차 회귀 — 상태 회귀는 train 에서만 맞춘다
                for k, y in Ys.items():
                    A_ = np.c_[np.ones(tr.sum()), st[tr]]; w = np.linalg.lstsq(A_, y[tr], rcond=None)[0]
                    res_ = y - np.c_[np.ones(len(y)), st] @ w
                    P, _ = fit_predict(F[tr], res_[tr, None], F[~tr], rng)
                    yt = res_[~tr]; r2s[k].append(1 - ((P[:, 0] - yt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum())
                # (2) 속도 맞춤 분류
                oh = np.stack([(law == l).astype(float) for l in LAWS], 1)
                good = np.zeros(len(idx), bool)
                for b in range(5):
                    if all(((law == l) & (vb == b)).sum() >= 10 for l in LAWS):
                        good |= vb == b
                te = ~tr & good
                P, _ = fit_predict(F[tr], oh[tr], F[te], rng); accs.append((P.argmax(1) == oh[te].argmax(1)).mean())
                P, _ = fit_predict(st[tr], oh[tr], st[te], rng); accs_state.append((P.argmax(1) == oh[te].argmax(1)).mean())
            out[name] = {**{f"R2_{k}_resid": round(float(np.mean(v_)), 4) for k, v_ in r2s.items()},
                         "law_acc_vmatched": round(float(np.mean(accs)), 4), "law_acc_state": round(float(np.mean(accs_state)), 4)}
            print(f"C{C} {name:8s} " + "  ".join(f"{k} {v_:+.3f}" for k, v_ in out[name].items()), flush=True)
        res[f"C{C}"] = out
    json.dump(res, open(OUT / "h1b_flat_v3.json", "w"), indent=1)


if __name__ == "__main__":
    main()
