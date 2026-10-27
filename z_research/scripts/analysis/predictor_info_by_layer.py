"""predictor 은닉 (384) 안에 속도 · 위치 · 모양 · 색 정보가 있나 — 층별 · 예측 걸음별, 판독기를 predictor 은닉 자체로 학습 (2026-10-26, CPU).
사용자: "velocity 정보 자체는 존재하나? 384 embedding 에서 layer 별로, predictor only" → "shape × color 는 따로 평가하도록, 그림 바꿔".
모델: 블록 AR natural_ar_vw_state latest, 블록마다 256 칸 평균 (1, 384), ar_layers_pooled.
  굴린 예측 = 굴림 자리 p (입력 = 블록 p 를 예측한 상태) 의 층 L 은닉 (roh), p = 7 + k, 걸음 k = 1–7
  참고 = teacher forcing 자리 p (입력 = 실제 encoder 블록 p) 의 층 L 은닉 (tfh)
  층 L = 0 입력 (표시 type 포함, 상수라 판독에 영향 없음) · 1–12 predictor 층 · 13 마지막 정규화. 라벨 = 블록 p (물체 보이는 표본).
판독기 (층 · 경로마다, train 8,000 편 → val 350 편, 입력은 자기 train 통계로 표준화):
  속도 (블록 안 변위 dx, dy) · 위치 (중심 x, y): 걸음마다 따로, 정규화 없는 최소제곱 (절편 포함) → val R² (두 축 합쳐)
  모양 (6) · 색 (8): **물체 칸 평균** (2026-10-26 저녁 갱신, layer_objcell_cache.py: train 4,000 · val 350 편) 으로, 따로. 걸음 1–7 (블록 8–14) 표본을 합쳐 판독기 하나 (모양 · 색은 클립마다 하나라 라벨이 같다; 표본 약 26,000),
    선형층 + 교차 엔트로피 (AdamW lr 1e-3, weight decay 1e-2, 묶음 256, 최대 100 epoch, train 클립 20 % 로 epoch 선택) → 걸음마다 val 정확도
  (2026-10-26 앞판은 모양 × 색 48 클래스를 걸음마다 따로 학습 — 둘 다 맞혀야 하고 클래스당 표본이 적어 낮게 나와 바꿨다; identity_readout_why_low.json)
  /venv/vjepa/bin/python z_research/scripts/analysis/predictor_info_by_layer.py   → z_analysis/results/predictor_info_by_layer.json
"""
import copy, json
import numpy as np
P = "/workspace/auto_research_cache/ar_layers_pooled"
TR, VA = np.load(f"{P}/train.npz"), np.load(f"{P}/val.npz")
TFH_T, ROH_T, LAB_T = TR["tfh"], TR["roh"], TR["lab"]; TFH_V, ROH_V, LAB_V = VA["tfh"], VA["roh"], VA["lab"]
# 모양 · 색은 물체 칸 평균 (층별 은닉) 으로: OC_*[path] (N, 7 걸음, 14 층, 384), 라벨은 meta
import json as _json
def _oc(split):
    Z = np.load(f"/workspace/auto_research_cache/objcell_concept/layers_{split}.npz"); meta = {m["i"]: m for m in _json.load(open(f"/workspace/auto_research_cache/ar_layers/{split}_meta.json"))}
    return {"rollout": Z["ro"], "teacher_forcing": Z["tf"]}, np.array([meta[int(i)]["shape"] for i in Z["i"]]), np.array([meta[int(i)]["color"] for i in Z["i"]]), Z["i"]
OC_T, OSH_T, OCO_T, OI_T = _oc("train"); OC_V, OSH_V, OCO_V, _ = _oc("val")
STEPS = list(range(1, 8)); PATHS = ("rollout", "teacher_forcing")


def hidden(path, H_ro, H_tf, L, k):
    return H_ro[:, L, k - 1] if path == "rollout" else H_tf[:, L, 7 + k]


def r2_ols(Xt, Yt, Xv, Yv):
    B = np.linalg.lstsq(np.c_[Xt, np.ones(len(Xt))], Yt, rcond=None)[0]; Pr = np.c_[Xv, np.ones(len(Xv))] @ B
    return float(1 - ((Pr - Yv) ** 2).sum() / ((Yv - Yv.mean(0)) ** 2).sum())


def cls_task(L):
    import torch, torch.nn.functional as F
    torch.set_num_threads(8)
    out = {t: {p: [None] * 7 for p in PATHS} for t in ("shape", "color")}
    for path in PATHS:
        Xt, ct, Xv, kv, yt, yv = [], [], [], [], {"shape": [], "color": []}, {"shape": [], "color": []}
        for k in STEPS:
            Ft, Fv = OC_T[path][:, k - 1, L].astype(np.float32), OC_V[path][:, k - 1, L].astype(np.float32)
            mt, mv = ~np.isnan(Ft).any(1), ~np.isnan(Fv).any(1)
            Xt.append(Ft[mt]); ct.append(OI_T[mt]); Xv.append(Fv[mv]); kv.append(np.full(mv.sum(), k))
            yt["shape"].append(OSH_T[mt].astype(np.int64)); yt["color"].append(OCO_T[mt].astype(np.int64)); yv["shape"].append(OSH_V[mv].astype(np.int64)); yv["color"].append(OCO_V[mv].astype(np.int64))
        Xt, ct, Xv, kv = np.concatenate(Xt), np.concatenate(ct), np.concatenate(Xv), np.concatenate(kv)
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-6; Xt, Xv = torch.from_numpy((Xt - mu) / sd), torch.from_numpy((Xv - mu) / sd)
        hold = torch.from_numpy(ct % 5 == 0)
        for t, K in (("shape", 6), ("color", 8)):
            Y = torch.from_numpy(np.concatenate(yt[t])); Yv = np.concatenate(yv[t])
            torch.manual_seed(0); head = torch.nn.Linear(384, K); opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2); best = (-1.0, None)
            Xa, Ya, Xh, Yh = Xt[~hold], Y[~hold], Xt[hold], Y[hold]
            for ep in range(100):
                perm = torch.randperm(len(Xa))
                for i in range(0, len(Xa), 256):
                    idx = perm[i:i + 256]; loss = F.cross_entropy(head(Xa[idx]), Ya[idx]); opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    a = (head(Xh).argmax(1) == Yh).float().mean().item()
                if a > best[0]:
                    best = (a, copy.deepcopy(head.state_dict()))
            head.load_state_dict(best[1])
            with torch.no_grad():
                pv = head(Xv).argmax(1).numpy()
            out[t][path] = [round(float((pv[kv == k] == Yv[kv == k]).mean() * 100), 1) for k in STEPS]
    print(f"층 {L} 끝", {t: out[t]["rollout"] for t in out}, flush=True)
    return L, out


if __name__ == "__main__":
    from multiprocessing import get_context
    res = {t: {p: [[None] * 7 for _ in range(14)] for p in PATHS} for t in ("velocity", "position", "shape", "color")}
    for k in STEPS:                                                   # 속도 · 위치: 걸음마다 최소제곱
        p = 7 + k; mt, mv = LAB_T[:, p, 0] > 0, LAB_V[:, p, 0] > 0
        for L in range(14):
            for path in PATHS:
                Xt_, Xv_ = hidden(path, ROH_T, TFH_T, L, k)[mt].astype(np.float64), hidden(path, ROH_V, TFH_V, L, k)[mv].astype(np.float64)
                mu, sd = Xt_.mean(0), Xt_.std(0) + 1e-6; Xt_, Xv_ = (Xt_ - mu) / sd, (Xv_ - mu) / sd
                for t, cols in (("velocity", (4, 6)), ("position", (6, 8))):
                    res[t][path][L][k - 1] = round(r2_ols(Xt_, LAB_T[mt, p, cols[0]:cols[1]].astype(np.float64), Xv_, LAB_V[mv, p, cols[0]:cols[1]].astype(np.float64)), 3)
    print("속도 · 위치 끝", flush=True)
    with get_context("fork").Pool(14) as pool:
        for L, out in pool.map(cls_task, range(14)):
            for t in out:
                for path in PATHS:
                    for j in range(7):
                        res[t][path][L][j] = out[t][path][j]
    res["shape_color_source"] = "object-cell mean (layer_objcell_cache.py)"
    for t, col, K in (("shape", 1, 6), ("color", 2, 8)):
        res[f"{t}_val_majority"] = [round(float(np.bincount(LAB_V[LAB_V[:, 7 + k, 0] > 0, 7 + k, col].astype(int), minlength=K).max() / (LAB_V[:, 7 + k, 0] > 0).sum() * 100), 1) for k in STEPS]
    res["layers"] = ["input"] + [str(i) for i in range(1, 13)] + ["norm"]; res["steps"] = STEPS
    json.dump(res, open("/workspace/world_model/vjepa2/z_analysis/results/predictor_info_by_layer.json", "w"), indent=1)
    for t in ("velocity", "position", "shape", "color"):
        print(t)
        for L in (0, 6, 13):
            print(f"   {res['layers'][L]:5s} rollout {res[t]['rollout'][L]}   TF {res[t]['teacher_forcing'][L]}")
    print("최빈 모양", res["shape_val_majority"], "최빈 색", res["color_val_majority"])
