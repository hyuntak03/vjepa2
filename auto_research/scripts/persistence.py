#!/usr/bin/env python3
"""[T] 판정용: (1) 문맥으로 예측한 이동이 맞을 확률 π̄ (속도 구간별, 정지 / 움직임) (2) 실제 이동 속도 구간별 운반 보상 (auto_research, 2026-10-21).

  /venv/vjepa/bin/python auto_research/scripts/persistence.py <rows 폴더> [<rows 폴더> ...]

입력: 흐름 오라클 측정이 남긴 칸별 손실 · 칸 흐름 (블록 b 대표 장 → b−1 역방향 흐름 U[b], 칸 단위).
π̄ (라그랑주 지속): 정답 블록 t+1 의 칸 c 에 온 내용의 블록 t 자리 s = c + U[t+1](c). 문맥만 보는 등속 예측은 s 의 내용을 s − U[t](s) 로 보낸다.
  맞음 = 그 예측 칸 == c. 구간 = 문맥 속도 |U[t](s)| (칸/블록). 정지 = 문맥 속도 < 0.25.
  허용 1 칸 판 (예측 칸과 c 의 체비셰프 거리 ≤ 1) 도 같이 — 물체 폭이 2 칸 넘으면 이쪽이 점유에 가깝다.
운반 보상 (실제 속도 |U[t+1](c)| 구간별, 칸 평균 손실의 복사 대비 %): 한 블록 옮기기 (참 흐름으로 h_t 를 옮김) vs 복사 · 궤적 평균 운반 vs 궤적 평균 제자리 · 같은 칸 평균."""
import json, os, sys, glob
import numpy as np

G = 16
BINS = [(0, 0.25), (0.25, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 99)]
out = {}
for d in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(d, "rows_*.npz"))):
        src = os.path.basename(f)[5:-4]
        z = np.load(f)
        key = f"{os.path.basename(d.rstrip('/'))}/{src}"
        r = {}
        # (2) 실제 속도 구간별 보상
        mag = z["mag"]
        print(f"\n=== {key}: 정답 블록 {mag.shape[0]}")
        print("  실제 속도 구간 (칸/블록) | 칸 비율 | 한 블록 옮기기 vs 복사 | 궤적평균 운반 vs 제자리 | 같은칸 평균 vs 복사 | 궤적평균 제자리 vs 같은칸 평균")
        rb = {}
        for lo, hi in BINS:
            m = (mag >= lo) & (mag < hi)
            if m.sum() < 200:
                continue
            c = z["copy"][m].mean()
            v = {k: float(z[k][m].mean()) for k in ("copy", "wc_oracle", "al_oracle", "al_stay", "mean8")}
            rb[f"{lo}-{hi}"] = dict(frac=float(m.mean()), **v)
            print(f"  [{lo:>4}, {hi:>4}) | {m.mean():.3f} | {100 * (v['copy'] - v['wc_oracle']) / c:+6.1f} % | {100 * (v['al_stay'] - v['al_oracle']) / c:+6.1f} % |"
                  f" {100 * (v['copy'] - v['mean8']) / c:+6.1f} % | {100 * (v['mean8'] - v['al_stay']) / c:+6.1f} %")
        r["reward_by_true_speed"] = rb
        # (1) π̄
        if "Uwin" in z:
            U = z["Uwin"].astype(np.float32)                              # (W, NB, 16, 16, 2)
            ctx = int(z["ctx"]) if "ctx" in z else 8
            W_, NB = U.shape[:2]
            cy, cx = np.divmod(np.arange(G * G), G)
            P0 = np.stack([cx + 0.5, cy + 0.5], 1)
            hit0, hit1, spd, tsp, perr = [], [], [], [], []
            for w in range(W_):
                for t in range(ctx - 1, NB - 1):
                    u1 = U[w, t + 1].reshape(-1, 2); u0 = U[w, t]
                    s = P0 + u1
                    si = np.floor(s).astype(int)
                    ok = (si >= 0).all(1) & (si < G).all(1)
                    si = np.clip(si, 0, G - 1)
                    v = u0[si[:, 1], si[:, 0]]                              # 문맥 속도 (역방향) at source
                    pred = np.floor(s - v).astype(int)
                    cc = np.stack([cx, cy], 1)
                    dch = np.abs(pred - cc).max(1)
                    hit0.append((dch == 0)[ok]); hit1.append((dch <= 1)[ok]); spd.append(np.linalg.norm(v, axis=1)[ok]); tsp.append(np.linalg.norm(u1, axis=1)[ok])
                    perr.append((16 * np.linalg.norm(u1 - v, axis=1))[ok])           # 등속 예측의 위치 오차 (px) = |실제 변위 − 문맥 변위|
            hit0, hit1, spd, tsp, perr = (np.concatenate(x) for x in (hit0, hit1, spd, tsp, perr))
            print("  π̄ (문맥 속도 구간별): 구간 | 칸 비율 | π̄ (같은 칸) | π̄ (허용 1 칸) | 실제 속도 중앙값 | 위치 오차 px (중앙 · RMS/√2 = 축당 σ_x)")
            pb = {}
            for lo, hi in BINS:
                m = (spd >= lo) & (spd < hi)
                if m.sum() < 200:
                    continue
                sx = float(np.sqrt((perr[m] ** 2).mean() / 2))
                pb[f"{lo}-{hi}"] = dict(frac=float(m.mean()), pi0=float(hit0[m].mean()), pi1=float(hit1[m].mean()), true_speed_med=float(np.median(tsp[m])),
                                        perr_med_px=float(np.median(perr[m])), sigma_x_px=sx)
                print(f"  [{lo:>4}, {hi:>4}) | {m.mean():.3f} | {hit0[m].mean():.3f} | {hit1[m].mean():.3f} | {np.median(tsp[m]):.2f} | {np.median(perr[m]):.1f} · {sx:.1f}")
            print("  π̄ (실제 속도 구간별 = 도착한 내용이 실제로 움직인 칸만; 가려졌다 드러나는 칸 제외 효과): 구간 | 칸 비율 | π̄ | π̄ ±1 | 위치 오차 중앙 px")
            pt = {}
            for lo, hi in BINS[2:]:
                m = (tsp >= lo) & (tsp < hi) & (spd >= 0.25)
                if m.sum() < 200:
                    continue
                pt[f"{lo}-{hi}"] = dict(frac=float(m.mean()), pi0=float(hit0[m].mean()), pi1=float(hit1[m].mean()), perr_med_px=float(np.median(perr[m])))
                print(f"  [{lo:>4}, {hi:>4}) | {m.mean():.3f} | {hit0[m].mean():.3f} | {hit1[m].mean():.3f} | {np.median(perr[m]):.1f}")
            r["pi_by_true_speed_ctxmoving"] = pt
            st, mv = spd < 0.25, spd >= 0.5
            r["pi_by_ctx_speed"] = pb
            r["pi_static"] = float(hit0[st].mean()); r["pi_moving"] = float(hit0[mv].mean()); r["pi1_moving"] = float(hit1[mv].mean())
            print(f"  정지 (문맥 속도 < 0.25) π̄ {hit0[st].mean():.3f} · 움직임 (≥ 0.5) π̄ {hit0[mv].mean():.3f} (허용 1 칸 {hit1[mv].mean():.3f})")
        out[key] = r
json.dump(out, open(os.path.join(os.path.dirname(__file__), "../results/persistence.json"), "w"), indent=1)


# (3) 움직이던 내용의 운명 (라그랑주, 칸 수준): 블록 t 의 칸 s (문맥 속도 v = −U[t](s) 가 1 칸 이상) 의 내용이 블록 t+1 에서 어디로 갔나.
#     도착 칸 = 블록 t+1 의 칸 c 가운데 역흐름 출발 칸이 s 인 것들의 평균 자리. 분류: 간다 (도착 칸 == s + v 를 반올림한 칸) · 남는다 (도착 칸 == s) · 그 밖 · 사라짐 (어느 c 도 s 에서 안 옴)
print("\n== 움직이던 내용의 운명 (문맥 속도 ≥ 1 칸/블록, 칸 수준): 간다 p · 남는다 q · 그 밖 · 사라짐")
fate = {}
for d in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(d, "rows_*.npz"))):
        z = np.load(f)
        if "Uwin" not in z:
            continue
        src = os.path.basename(f)[5:-4]
        U = z["Uwin"].astype(np.float32); ctx = int(z["ctx"]) if "ctx" in z else 8
        W_, NB = U.shape[:2]
        cy, cx = np.divmod(np.arange(G * G), G)
        P0 = np.stack([cx + 0.5, cy + 0.5], 1)
        cnt = {k: {"go": 0, "stay": 0, "other": 0, "lost": 0} for k in ("1-2", "2-4", "1-99")}
        for w in range(W_):
            for t in range(ctx - 1, NB - 1):
                u1 = U[w, t + 1].reshape(-1, 2); u0 = U[w, t].reshape(-1, 2)
                srcp = np.floor(P0 + u1).astype(int)
                ok = (srcp >= 0).all(1) & (srcp < G).all(1)
                sidx = np.where(ok, srcp[:, 1] * G + srcp[:, 0], -1)
                for s in np.where(np.linalg.norm(u0, axis=1) >= 1)[0]:
                    sp = np.linalg.norm(u0[s])
                    ks = ["1-99", "1-2" if sp < 2 else ("2-4" if sp < 4 else None)]
                    cs = np.where(sidx == s)[0]
                    if cs.size == 0:
                        out = "lost"
                    else:
                        dest = np.floor(P0[cs].mean(0)).astype(int)
                        sc = np.array([cx[s], cy[s]]); ext = np.floor(P0[s] - u0[s]).astype(int)
                        out = "go" if (dest == ext).all() else ("stay" if (dest == sc).all() else "other")
                    for k in ks:
                        if k:
                            cnt[k][out] += 1
        key = f"{os.path.basename(d.rstrip('/'))}/{src}"
        fate[key] = {}
        for k, c in cnt.items():
            n = sum(c.values())
            if n < 100:
                continue
            fr = {kk: v / n for kk, v in c.items()}
            fate[key][k] = dict(n=n, **fr)
            print(f"  {key:28s} 속도 {k:5s} (n {n:6d}): 간다 {fr['go']:.2f} · 남는다 {fr['stay']:.2f} · 그 밖 {fr['other']:.2f} · 사라짐 {fr['lost']:.2f}")
json.dump(fate, open(os.path.join(os.path.dirname(__file__), "../results/persistence_fate.json"), "w"), indent=1)
