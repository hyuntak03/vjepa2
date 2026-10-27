"""정체 (모양 · 색) 개념 벡터 — 전체 표본으로 만들어 저장 + 의미 있는가 점검 + encoder vs 예측 비교 (2026-10-26, 학습 없음, CPU).
사용자: "전체 샘플로 concept vector 만들어서 figure 그려줘. 자세한 config 와 왜 의미 있는지 md 로 기록".
정본 문서: z_analysis/docs/IDENTITY_CONCEPT_VECTORS_2026-10-26.md

입력 (블록마다 256 칸 평균 (1, 1280), 3D 합성 phys3d_single_v1 train 8,000 편):
  AR      : 정답 = 블록 단위 encoder 출력 (ar_layers_pooled enc, 블록 0–15) · 예측 = 굴린 예측 출력 (roout, 걸음 1–8 = 블록 8–15)
  릴리즈  : 정답 = 32 장 한 번에 encoder 출력 (release_pooled gt, 블록 0–15) · 예측 = 한 번에 낸 8 블록 (pred, 걸음 1–8)
개념 벡터 (사용자 정의): c_k = (클래스 k 표본 평균) − (전체 표본 평균), 그 블록에서 물체가 보이는 표본만. 모양 6 · 색 8.
저장: 전체 표본 개념 벡터 (공간 × 블록 (또는 걸음) × 클래스 × 1280) + 클래스별 표본 수.
의미 점검 (블록 8–15 / 걸음 1–8):
  재현도     = 클립 짝수 · 홀수 번째 반쪽에서 따로 구한 개념 벡터의 cos (클래스 평균), 라벨을 섞은 영 기준 (5 회 평균) 과 함께
  장면 일반성 = 서로 겹치지 않는 장면 묶음 A (장면 번호 짝수 7 종) · B (홀수 6 종) 에서 따로 구한 개념 벡터의 cos, 영 기준과 함께
encoder vs 예측 (걸음 1–8, 같은 블록):
  **직접 cos · 크기 비 (그림에 쓰는 값, 2026-10-26 사용자: "그냥 cos 를 직접 재면 안 돼? 8000 샘플이면 잡음 거의 없을 것")** = 전체 표본 개념 벡터끼리 클래스마다 cos · |c_pred| / |c_enc|
  같은 클립 cos (옛 방식, 부풀어 있음) 와 그 영 기준
  잡음 보정 cos   = 클래스마다 [⟨e^A, p^B⟩ + ⟨e^B, p^A⟩] / 2 ÷ sqrt(⟨e^A, e^B⟩ · ⟨p^A, p^B⟩)   (A · B = 클립 반쪽. 반쪽끼리 잡음이 독립이라 내적에서 잡음이 빠진다)
  잡음 보정 크기 비 = sqrt(⟨p^A, p^B⟩ / ⟨e^A, e^B⟩)   (전체 표본 크기 비 |c_pred| / |c_enc| 도 참고로)
  클래스 평균 ± 표준편차 (클래스 사이).
  /venv/vjepa/bin/python z_research/scripts/analysis/identity_concept_vectors.py
출력: z_analysis/results/identity_concept_vectors.npz (벡터) · z_analysis/results/identity_concept_vectors.json (점검 · 비교 수치)
"""
import json
import numpy as np
P = "/workspace/auto_research_cache"; R = "/workspace/world_model/vjepa2/z_analysis/results"
FAMS = (("shape", 1, 6), ("color", 2, 8))
SA = [0, 2, 4, 6, 8, 10, 12]
rng = np.random.default_rng(0)


def cv(X, y, K):
    return np.stack([X[y == c].mean(0) for c in range(K)]) - X.mean(0)


def cosm(A, B):
    return float(np.mean((A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))))


L = np.load(f"{P}/ar_layers_pooled/train.npz"); lab, ii = L["lab"], L["i"]; scen_names = [str(s) for s in L["scen_names"]]
Rl = np.load(f"{P}/release_pooled/train.npz"); o = {int(v): n for n, v in enumerate(ii)}; sel = np.array([o[int(v)] for v in Rl["i"]])
SPACES = {"ar": (L["enc"], L["roout"], lab), "release": (Rl["gt"], Rl["pred"], lab[sel])}
vec, res = {}, {"n_clips": {m: int(len(v[2])) for m, v in SPACES.items()}, "scene_group_A": [scen_names[i] for i in SA],
                "scene_group_B": [n for i, n in enumerate(scen_names) if i not in SA]}
for model, (GT, PR, LT) in SPACES.items():
    res[model] = {}
    for fam, col, K in FAMS:
        Ve, Vp, Ne, Np = np.full((16, K, 1280), np.nan, np.float32), np.full((8, K, 1280), np.nan, np.float32), np.zeros((16, K), int), np.zeros((8, K), int)
        for b in range(16):
            m = LT[:, b, 0] > 0; y = LT[m, b, col].astype(int)
            Ve[b] = cv(GT[m, b].astype(np.float64), y, K); Ne[b] = np.bincount(y, minlength=K)
            if b >= 8:
                Vp[b - 8] = cv(PR[m, b - 8].astype(np.float64), y, K); Np[b - 8] = Ne[b]
        vec[f"{model}_target_{fam}"], vec[f"{model}_pred_{fam}"] = Ve, Vp
        vec[f"{model}_target_{fam}_n"], vec[f"{model}_pred_{fam}_n"] = Ne, Np
        rows = {}
        for k in range(1, 9):
            b = 7 + k; m = LT[:, b, 0] > 0; y = LT[m, b, col].astype(int); clip = np.nonzero(m)[0]
            h = clip % 2 == 0; sa = np.isin(LT[m, b, 3].astype(int), SA)
            E, Pd = GT[m, b].astype(np.float64), PR[m, b - 8].astype(np.float64)
            perms = [rng.permutation(y) for _ in range(5)]
            row = {"n": int(m.sum()), "n_per_class": np.bincount(y, minlength=K).tolist()}
            for nm, X in (("target", E), ("pred", Pd)):
                row[f"reliability_{nm}"] = round(cosm(cv(X[h], y[h], K), cv(X[~h], y[~h], K)), 3)
                row[f"reliability_{nm}_null"] = round(float(np.mean([cosm(cv(X[h], yp[h], K), cv(X[~h], yp[~h], K)) for yp in perms])), 3)
                row[f"scene_general_{nm}"] = round(cosm(cv(X[sa], y[sa], K), cv(X[~sa], y[~sa], K)), 3)
                row[f"scene_general_{nm}_null"] = round(float(np.mean([cosm(cv(X[sa], yp[sa], K), cv(X[~sa], yp[~sa], K)) for yp in perms])), 3)
            ce, cp = cv(E, y, K), cv(Pd, y, K)
            dc = (ce * cp).sum(1) / (np.linalg.norm(ce, axis=1) * np.linalg.norm(cp, axis=1)); dr = np.linalg.norm(cp, axis=1) / np.linalg.norm(ce, axis=1)
            row.update({"direct_cos_per_class": np.round(dc, 3).tolist(), "direct_cos": round(float(dc.mean()), 3), "direct_cos_sd": round(float(dc.std()), 3),
                        "direct_norm_ratio_per_class": np.round(dr, 3).tolist(), "direct_norm_ratio": round(float(dr.mean()), 3), "direct_norm_ratio_sd": round(float(dr.std()), 3)})
            row["same_clip_cos"] = round(cosm(ce, cp), 3)
            row["same_clip_cos_null"] = round(float(np.mean([cosm(cv(E, yp, K), cv(Pd, yp, K)) for yp in perms])), 3)
            eA, eB, pA, pB = cv(E[h], y[h], K), cv(E[~h], y[~h], K), cv(Pd[h], y[h], K), cv(Pd[~h], y[~h], K)
            ee, pp = (eA * eB).sum(1), (pA * pB).sum(1); x = ((eA * pB).sum(1) + (eB * pA).sum(1)) / 2
            ok = (ee > 0) & (pp > 0)
            cc = np.where(ok, x / np.sqrt(np.clip(ee * pp, 1e-12, None)), np.nan); nr = np.where(ok, np.sqrt(np.clip(pp, 0, None) / np.clip(ee, 1e-12, None)), np.nan)
            row.update({"corrected_cos_per_class": np.round(cc, 3).tolist(), "corrected_cos": round(float(np.nanmean(cc)), 3), "corrected_cos_sd": round(float(np.nanstd(cc)), 3),
                        "corrected_norm_ratio_per_class": np.round(nr, 3).tolist(), "corrected_norm_ratio": round(float(np.nanmean(nr)), 3), "corrected_norm_ratio_sd": round(float(np.nanstd(nr)), 3),
                        "full_sample_norm_ratio": round(float(np.mean(np.linalg.norm(cp, axis=1) / np.linalg.norm(ce, axis=1))), 3), "n_class_undefined": int((~ok).sum())})
            rows[k] = row
        res[model][fam] = rows
        print(model, fam, flush=True)
        for key in ("reliability_target", "reliability_target_null", "reliability_pred", "scene_general_target", "scene_general_target_null", "scene_general_pred",
                    "scene_general_pred_null", "same_clip_cos", "same_clip_cos_null", "direct_cos", "direct_norm_ratio", "corrected_cos", "corrected_norm_ratio", "full_sample_norm_ratio", "n_class_undefined"):
            print(f"   {key:26s}", [rows[k][key] for k in range(1, 9)])
np.savez(f"{R}/identity_concept_vectors.npz", **vec)
json.dump(res, open(f"{R}/identity_concept_vectors.json", "w"), indent=1)
print("saved")
