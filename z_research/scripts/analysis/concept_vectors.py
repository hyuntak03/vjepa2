"""개념 벡터 (모양 · 색 · 정체 · 방향 · 속력 · 위치) 를 encoder 와 블록 AR 층마다 따로 구한다 (2026-10-26, 학습 없음, CPU).
사용자: "concept vector (encoder), predictor 이렇게 각각 구해야지" · "속도, 물체 shape, color, 물체 위치 (가능해?)".

입력 pool_objcells.py 의 블록별 물체 칸 평균 (1, D). 표현 (view) × 묶음 (group):
  enc            encoder 출력 (1280)                     묶음: ctx 블록 0–7 · fut 블록 8–15 · s12 / s35 / s68 (블록 8–9 / 10–12 / 13–15, 굴린 예측과 같은 블록)
  tf_L{0..13}    teacher forcing 층 (384), 입력 블록 기준   묶음: ctx · fut
  tf_out         teacher forcing 출력 (1280), 예측 블록 기준 묶음: ctx (블록 1–7) · fut
  ro_L{0..13}    굴린 예측 층 (384), 입력 = 블록 8+k 상태    묶음: s12 (걸음 1–2) · s35 (3–5) · s68 (6–7)
  ro_out         굴린 예측 출력 (1280)                    묶음: s12 · s35 · s68 (걸음 6–8)
개념 (클래스): shape 6 · color 8 · identity 48 · direction 8 (블록 안 변위 ≥ 2 px 만) · speed 5 (encoder train 분위) · position 16 (4×4, 중심 64 px 구역)
개념 벡터: 시나리오 평균을 뺀 뒤 클래스 평균 m_k − (클래스 평균들의 평균). 클래스마다 같은 가중 (정체별 클립 수 72–380 불균형).
  반 A · 반 B (클립 단위 무작위) 로 따로 구해 재현도: 같은 클래스끼리 cos 평균 · 개념 부분공간 (상위 r) 겹침.
검증 (val 350 편): 개념 부분공간 좌표에서 최근접 클래스 평균 정확도 (우연 = 1/K). 분산 몫 η² = 클래스 간 / 전체 (train).

  /venv/vjepa/bin/python z_research/scripts/analysis/concept_vectors.py
출력: /workspace/auto_research_cache/concepts/concepts.npz (키 "{view}|{group}|{family}" → (K, D), "..|A" · "..|B" 반판) ·
      z_analysis/results/concept_vectors.json (요약)
"""
import json, os
import numpy as np

P = "/workspace/auto_research_cache/ar_layers_pooled"; OUTD = "/workspace/auto_research_cache/concepts"; os.makedirs(OUTD, exist_ok=True)
OUTJ = "/workspace/world_model/vjepa2/z_analysis/results/concept_vectors.json"   # 2026-10-26 사용자: 결과는 z_analysis 에
TR, VA = np.load(f"{P}/train.npz"), np.load(f"{P}/val.npz")
rng = np.random.default_rng(0)
half = rng.integers(0, 2, TR["lab"].shape[0])


def views(Z):
    """{(view, group): (X (N, S, D) 배열을 꺼내는 함수, 슬롯 → 라벨 블록, 슬롯 선택)}"""
    V = {}
    blocks16 = np.arange(16)
    for g, sl in (("ctx", range(0, 8)), ("fut", range(8, 16)), ("s12", range(8, 10)), ("s35", range(10, 13)), ("s68", range(13, 16))):
        V[("enc", g)] = (lambda Z: Z["enc"], blocks16, list(sl))
    for l in range(14):
        for g, sl in (("ctx", range(0, 8)), ("fut", range(8, 16))):
            V[(f"tf_L{l}", g)] = (lambda Z, l=l: Z["tfh"][:, l], blocks16, list(sl))
        for g, sl in (("s12", range(0, 2)), ("s35", range(2, 5)), ("s68", range(5, 7))):
            V[(f"ro_L{l}", g)] = (lambda Z, l=l: Z["roh"][:, l], np.arange(8) + 8, list(sl))
    for g, sl in (("ctx", range(1, 8)), ("fut", range(8, 16))):
        V[("tf_out", g)] = (lambda Z: Z["tfout"], blocks16, list(sl))
    for g, sl in (("s12", range(0, 2)), ("s35", range(2, 5)), ("s68", range(5, 8))):
        V[("ro_out", g)] = (lambda Z: Z["roout"], np.arange(8) + 8, list(sl))
    return V


lab_tr = TR["lab"]
spd_all = np.hypot(lab_tr[..., 4], lab_tr[..., 5])[lab_tr[..., 0] > 0]
SPD_EDGES = np.quantile(spd_all, [0.2, 0.4, 0.6, 0.8])


def classes(L, fam):
    """L (n, 8) 라벨 → 클래스 번호 (없으면 −1), K."""
    if fam == "shape":
        return L[:, 1].astype(int), 6
    if fam == "color":
        return L[:, 2].astype(int), 8
    if fam == "identity":
        return (L[:, 1] * 8 + L[:, 2]).astype(int), 48
    dx, dy = L[:, 4], L[:, 5]; s = np.hypot(dx, dy)
    if fam == "direction":
        return np.where(s >= 2.0, ((np.arctan2(dy, dx) + np.pi) / (2 * np.pi / 8)).astype(int) % 8, -1), 8
    if fam == "speed":
        return np.digitize(s, SPD_EDGES), 5
    if fam == "position":
        cx, cy = np.clip(L[:, 6], 0, 255.9), np.clip(L[:, 7], 0, 255.9)
        return (cy // 64).astype(int) * 4 + (cx // 64).astype(int), 16
    raise ValueError(fam)


FAMS = ("shape", "color", "identity", "direction", "speed", "position")


def gather(Z, getter, lab_blocks, slots):
    X = getter(Z)[:, slots].astype(np.float32)                                  # (N, s, D)
    L = Z["lab"][:, lab_blocks[slots]]                                          # (N, s, 8)
    clip = np.repeat(np.arange(X.shape[0])[:, None], len(slots), 1)
    X, L, clip = X.reshape(-1, X.shape[-1]), L.reshape(-1, 8), clip.reshape(-1)
    ok = (L[:, 0] > 0) & ~np.isnan(X).any(1)
    return X[ok], L[ok], clip[ok]


def cmeans(X, y, K):
    M = np.full((K, X.shape[1]), np.nan, np.float64); n = np.zeros(K, int)
    for k in range(K):
        m = y == k; n[k] = m.sum()
        if n[k]:
            M[k] = X[m].mean(0)
    return M, n


def concept(X, y, K):
    M, n = cmeans(X, y, K)
    good = n >= 5
    C = M - np.nanmean(M[good], 0)
    C[~good] = np.nan
    return C, n


def subspace(C, r):
    G = C[~np.isnan(C).any(1)]
    U = np.linalg.svd(G, full_matrices=False)[2][:r].T
    return U


store, summ = {}, {}
VTR, VVA = views(TR), views(VA)
for key, (getter, lb, sl) in VTR.items():
    Xt, Lt, ct = gather(TR, getter, lb, sl)
    Xv, Lv, _ = gather(VA, getter, lb, sl)
    if len(Xt) < 200:
        continue
    sc_t, sc_v = Lt[:, 3].astype(int), Lv[:, 3].astype(int)
    smu = {s: Xt[sc_t == s].mean(0) for s in np.unique(sc_t)}
    Xt = Xt - np.stack([smu[s] for s in sc_t]); Xv = Xv - np.stack([smu.get(s, Xt.mean(0)) for s in sc_v]) if len(Xv) else Xv
    tot = ((Xt - Xt.mean(0)) ** 2).sum()
    for fam in FAMS:
        yt, K = classes(Lt, fam); yv, _ = classes(Lv, fam)
        mt = yt >= 0
        C, n = concept(Xt[mt], yt[mt], K)
        hA = half[ct[mt]] == 0
        CA, _ = concept(Xt[mt][hA], yt[mt][hA], K); CB, _ = concept(Xt[mt][~hA], yt[mt][~hA], K)
        both = ~np.isnan(CA).any(1) & ~np.isnan(CB).any(1)
        cosAB = float(np.mean([CA[k] @ CB[k] / (np.linalg.norm(CA[k]) * np.linalg.norm(CB[k]) + 1e-9) for k in np.nonzero(both)[0]])) if both.any() else float("nan")
        r = int(min(K - 1, both.sum() - 1, 10))
        UA, UB = subspace(CA, r), subspace(CB, r)
        ovAB = float(np.linalg.norm(UA.T @ UB) ** 2 / r) if r > 0 else float("nan")
        # η² (클래스 간 분산 몫, train)
        Mk, nk = cmeans(Xt[mt], yt[mt], K); gm = Xt[mt].mean(0)
        eta2 = float(sum(nk[k] * ((Mk[k] - gm) ** 2).sum() for k in range(K) if nk[k]) / (((Xt[mt] - gm) ** 2).sum() + 1e-9))
        # val 최근접 평균 (개념 부분공간, r = K−1 최대)
        acc = float("nan")
        mv = yv >= 0
        if mv.sum() >= 20:
            U = subspace(C, int(min(K - 1, (~np.isnan(C).any(1)).sum() - 1)))
            good = ~np.isnan(C).any(1); gmean = np.nanmean(Mk[good], 0)
            cv = (Xv[mv] - gmean) @ U; cm = (Mk[good] - gmean) @ U
            pred = np.nonzero(good)[0][np.argmin(((cv[:, None] - cm[None]) ** 2).sum(-1), 1)]
            acc = float((pred == yv[mv]).mean() * 100)
        k_ = "|".join(key) + "|" + fam
        store[k_] = C.astype(np.float32); store[k_ + "|A"] = CA.astype(np.float32); store[k_ + "|B"] = CB.astype(np.float32)
        summ[k_] = {"n_train": int(mt.sum()), "n_val": int(mv.sum()), "K": K, "classes_used": int((n >= 5).sum()),
                    "rel_cos": round(cosAB, 3), "rel_subspace": round(ovAB, 3), "eta2": round(eta2, 4),
                    "val_acc": round(acc, 1), "chance": round(100 / K, 1), "mean_norm": round(float(np.nanmean(np.linalg.norm(C, axis=1))), 3)}
    print(key, {f: (summ["|".join(key) + "|" + f]["rel_cos"], summ["|".join(key) + "|" + f]["val_acc"]) for f in FAMS}, flush=True)
np.savez(f"{OUTD}/concepts.npz", **store)
json.dump(summ, open(OUTJ, "w"), indent=1, ensure_ascii=False)
print("saved", OUTJ, len(summ))
