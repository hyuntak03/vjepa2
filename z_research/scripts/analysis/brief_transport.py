#!/usr/bin/env python3
"""공간 운반 분해 (2026-10-18) — "시간 평균" 인가 "운반 실패" 인가.

  CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python z_research/scripts/analysis/brief_transport.py --device cuda

예측 블록의 물체 칸 i (물체 중심 3×3) 를 **직전 블록의 칸 i 주변 5×5 (25 칸)** 의 최소제곱 가중합으로 분해 (상수 없음, 칸마다 따로).
대조군: 진짜 다음 블록을 같은 방식으로 분해. 지도는 실제 이동 방향이 +행 · +열 이 되게 뒤집어 평균한다.
  직전 블록 = 진짜 블록 (걸음 0: 마지막 문맥 블록 C-1 → 예측 C / 걸음 3: 진짜 C+2 → 예측 C+3)
읽기:
  진짜는 이동 방향 칸에 몰리는데 예측은 같은 칸 (0,0) 에 몰림 → 운반을 안 함 (복사)
  진짜는 한 칸에 몰리는데 예측은 여러 칸에 퍼짐            → 운반은 하는데 흐림 (평균 · 모드 섞임)
  둘이 비슷                                              → 운반 문제가 아님
샘플: VOBench 움직이는 clip (flat_static 제외), visible, primitive + realistic, 최대 120. brief_motion.py 의 로더 · 모델."""
import os, sys, csv, json, collections
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "brief_motion.py")).read()
src = src[: src.index("models = [Model(r) for r in a.runs]")]
__file__ = os.path.join(HERE, "brief_motion.py")
exec(src)
models = [Model(r) for r in a.runs]
R5 = 2                                                                  # 5×5
rows = [r for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv")))
        if r["variant"] == "pos_a" and r["motion"] != "flat_static" and r["occlusion"] == "visible"]
samp = list(np.random.default_rng(3).permutation(rows))[:120]

def decomp(prev, y, i):
    """prev (S, D), y (D,), i 토큰 → 5×5 가중치 (없으면 None), R²."""
    r0, c0 = divmod(int(i), G)
    if r0 < R5 or c0 < R5 or r0 >= G - R5 or c0 >= G - R5:
        return None, None
    nb = [(r0 + dr) * G + (c0 + dc) for dr in range(-R5, R5 + 1) for dc in range(-R5, R5 + 1)]
    A = prev[nb].T                                                      # (D, 25)
    w = torch.linalg.lstsq(A, y.unsqueeze(1)).solution.flatten()
    res = y - A @ w
    return w.view(5, 5).cpu().numpy(), float(1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum())

acc = {k: collections.defaultdict(list) for k in ("step0", "step3")}
for n, r in enumerate(samp):
    x, posf, hid = load(r); hb = blocks(x); H = hb[0].view(16, S, -1)
    preds = {m.name: m.go(x, hb)[2].view(K, S, -1) for m in models}
    for key, k in (("step0", 0), ("step3", 3)):
        b = C + k
        if hid[b] >= 0.5 or hid[b - 1] >= 0.5:
            continue
        d = posf[b] - posf[b - 1]; sp = float(np.hypot(*d))
        fr, fc = d[0] < 0, d[1] < 0
        for i in region(posf[b]):
            ys = {"true": H[b][i]}; ys.update({m.name: preds[m.name][k][i] for m in models})
            for name, y in ys.items():
                w, r2 = decomp(H[b - 1], y, i)
                if w is None:
                    break
                if fr: w = w[::-1, :]
                if fc: w = w[:, ::-1]
                acc[key][name].append(w); acc[key]["r2_" + name].append(r2); acc[key]["sp_" + name].append(sp)
    if n % 30 == 0:
        print(f"{n}/{len(samp)}", flush=True)

names = ["true"] + [m.name for m in models]
def summary(W):
    W = np.array(W); M = W.mean(0); A = np.abs(W)
    conc = float((A.max(axis=(1, 2)) / A.sum(axis=(1, 2))).mean())
    return M, conc
print("\n가중치 지도 = 직전 블록 5×5 의 가운데 3×3 (행: 위→아래 = 이동 반대→이동 방향, 열: 같음). (0,0) = 같은 칸, (+1,+1) 쪽 = 이동 방향")
res = {}
for key in ("step0", "step3"):
    for sub, lo, hi in (("전체", 0, 99), ("느림 <0.5", 0, 0.5), ("빠름 ≥0.75", 0.75, 99)):
        print(f"\n=== {key} · {sub} ===")
        for nm in names:
            sp = np.array(acc[key]["sp_" + nm]); W = np.array(acc[key][nm]); sel = (sp >= lo) & (sp < hi)
            if sel.sum() < 5:
                continue
            M, conc = summary(W[sel]); r2 = float(np.mean(np.array(acc[key]["r2_" + nm])[sel]))
            c = M[1:4, 1:4]
            lab = "진짜 다음 블록" if nm == "true" else nm[-24:]
            print(f"  {lab:>24s} n{sel.sum():4d}  같은칸 {M[2,2]:+.2f}  이동방향 (+1,+1)·(+1,0)·(0,+1) {M[3,3]:+.2f} {M[3,2]:+.2f} {M[2,3]:+.2f}"
                  f"  반대 (−1,0)·(0,−1) {M[1,2]:+.2f} {M[2,1]:+.2f}  집중도 {conc:.2f}  R² {r2:.2f}")
            res[f"{key}|{sub}|{nm}"] = {"mean_map": M.tolist(), "conc": conc, "r2": r2, "n": int(sel.sum())}
json.dump(res, open(os.path.join(a.out, "brief_transport.json"), "w"), indent=1)
print("\n저장:", os.path.join(a.out, "brief_transport.json"))
