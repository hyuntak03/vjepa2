"""정지 물체 vs 이동 물체 — 예측이 물체의 정체 · 진하기를 유지하나 (물체 칸 기준, 2026-10-26 저녁, CPU).
사용자: "정지한 물체만 해봐". 정지 물체는 진짜 자리 = 마지막에 본 자리라 위치 문제 없이 정체 · 진하기만 남는다.
입력: objcell_concept_cache.py (블록 8–15 물체 칸 평균: encoder 정답 tgt_true · 예측 pred_true; AR 굴린 출력 · 릴리즈), ar_layers meta (중심 · 블록 안 변위 · 보임).
나눔 (블록 b, 걸음 k = b − 7; 블록 7 · b 모두 물체 보일 때):
  정지 = |중심_b − 중심_7| < 4 px 이고 블록 b 안 변위 < 1 px · 이동 = |중심_b − 중심_7| ≥ 32 px (2 칸)
판독기 (선형, 모양 6 · 색 8 따로, 입력 표준화, AdamW lr 1e-3 · wd 1e-2 · 묶음 512 · 최대 60 epoch, 걸음 1–8 합쳐 학습):
  train 의 짝수 번째 클립으로 학습 → 홀수 번째 클립 + val 에서 걸음 · 정지 / 이동별 정확도.
  encoder 판독기 (정답으로 학습) → 정답 (상한) · 예측 / 예측 판독기 (예측으로 학습) → 예측.
표본별 진하기: μ = 정답 화면 평균 토큰의 전체 평균. 크기 비 |p − μ| / |t − μ| · 방향 cos(p − μ, t − μ) (p = 예측 물체 칸, t = 정답 물체 칸).
  /venv/vjepa/bin/python z_research/scripts/analysis/static_vs_moving_objcell.py   → z_analysis/results/static_vs_moving_objcell.json
"""
import copy, json
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(48)
D = "/workspace/auto_research_cache/objcell_concept"; M = "/workspace/auto_research_cache/ar_layers"


def meta_arrays(split, I):
    meta = {m["i"]: m for m in json.load(open(f"{M}/{split}_meta.json"))}
    sh = np.array([meta[int(i)]["shape"] for i in I]); co = np.array([meta[int(i)]["color"] for i in I])
    ok = np.array([[meta[int(i)]["blocks"][b]["ok"] for b in range(16)] for i in I])
    ctr = np.array([[meta[int(i)]["blocks"][b]["ctr"] for b in range(16)] for i in I], dtype=np.float64)
    disp = np.array([[np.hypot(*meta[int(i)]["blocks"][b]["disp"]) for b in range(16)] for i in I])
    return sh, co, ok, ctr, disp


def groups(ok, ctr, disp):
    """(N, 8) 정지 · 이동 표시, 걸음 k = 1–8."""
    d = np.linalg.norm(ctr[:, 8:16] - ctr[:, 7:8], axis=-1); both = ok[:, 8:16] & ok[:, 7:8]
    return both & (d < 4) & (disp[:, 8:16] < 1), both & (d >= 32)


def fit(X, y, K):
    torch.manual_seed(0); n = len(X); h = torch.from_numpy(np.arange(n) % 5 == 0); X, y = torch.from_numpy(X), torch.from_numpy(y)
    head = torch.nn.Linear(X.shape[1], K); opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2); best = (-1, None)
    Xa, ya, Xh, yh = X[~h], y[~h], X[h], y[h]
    for ep in range(60):
        perm = torch.randperm(len(Xa))
        for i in range(0, len(Xa), 512):
            idx = perm[i:i + 512]; loss = F.cross_entropy(head(Xa[idx]), ya[idx]); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            acc = (head(Xh).argmax(1) == yh).float().mean().item()
        if acc > best[0]:
            best = (acc, copy.deepcopy(head.state_dict()))
    head.load_state_dict(best[1]); return head


res = {"definition": {"static": "|ctr_b - ctr_7| < 4 px and in-block disp < 1 px", "moving": "|ctr_b - ctr_7| >= 32 px"}}
for model in ("ar", "release"):
    Z, V = np.load(f"{D}/{model}_train.npz"), np.load(f"{D}/{model}_val.npz")
    shT, coT, okT, ctrT, dT = meta_arrays("train", Z["i"]); shV, coV, okV, ctrV, dV = meta_arrays("val", V["i"])
    stT, mvT = groups(okT, ctrT, dT); stV, mvV = groups(okV, ctrV, dV)
    even = (np.arange(len(Z["i"])) % 2 == 0)
    # 평가 집합 = train 홀수 클립 + val
    T_ = {k: np.concatenate([Z[k][~even], V[k]]).astype(np.float32) for k in ("tgt_true", "pred_true")}
    E_ = {"shape": np.concatenate([shT[~even], shV]), "color": np.concatenate([coT[~even], coV]), "static": np.concatenate([stT[~even], stV]), "moving": np.concatenate([mvT[~even], mvV]),
          "ok": np.concatenate([okT[~even], okV])}
    mu = Z["tgt_whole"].astype(np.float32).reshape(-1, 1280); mu = np.nanmean(mu, 0)
    r = {"n_eval": {g: [int(E_[g][:, k].sum()) for k in range(8)] for g in ("static", "moving")}}
    for fam, yT, K in (("shape", shT, 6), ("color", coT, 8)):
        heads = {}
        for src in ("tgt_true", "pred_true"):
            X = Z[src][even].astype(np.float32); m = okT[even, 8:16]; Xs = X[m]; ys = np.repeat(yT[even][:, None], 8, 1)[m].astype(np.int64)
            mu_s, sd_s = Xs.mean(0), Xs.std(0) + 1e-6; heads[src] = (fit((Xs - mu_s) / sd_s, ys, K), mu_s, sd_s)
        out = {}
        for nm, (hsrc, tsrc) in {"enc_head_on_target": ("tgt_true", "tgt_true"), "enc_head_on_pred": ("tgt_true", "pred_true"), "pred_head_on_pred": ("pred_true", "pred_true")}.items():
            h, mu_s, sd_s = heads[hsrc]; out[nm] = {}
            for g in ("static", "moving"):
                accs = []
                for k in range(8):
                    m = E_[g][:, k]
                    if m.sum() < 10:
                        accs.append(None); continue
                    with torch.no_grad():
                        pr = h(torch.from_numpy(((T_[tsrc][m, k] - mu_s) / sd_s).astype(np.float32))).argmax(1).numpy()
                    accs.append(round(float((pr == E_[fam][m]).mean() * 100), 1))
                out[nm][g] = accs
        out["majority"] = {g: round(float(max(np.bincount(np.repeat(E_[fam][:, None], 8, 1)[E_[g]], minlength=K)) / E_[g].sum() * 100), 1) for g in ("static", "moving")}
        r[fam] = out
    # 표본별 진하기
    t = T_["tgt_true"] - mu; p = T_["pred_true"] - mu
    ratio = np.linalg.norm(p, axis=-1) / np.linalg.norm(t, axis=-1); cosv = (p * t).sum(-1) / (np.linalg.norm(p, axis=-1) * np.linalg.norm(t, axis=-1))
    r["strength"] = {g: {"norm_ratio": [round(float(np.nanmean(ratio[E_[g][:, k], k])), 3) for k in range(8)], "cos": [round(float(np.nanmean(cosv[E_[g][:, k], k])), 3) for k in range(8)]} for g in ("static", "moving")}
    res[model] = r
    print(model, "평가 표본 (정지 / 이동):", r["n_eval"]["static"], r["n_eval"]["moving"], flush=True)
    for fam in ("shape", "color"):
        for nm in ("enc_head_on_target", "enc_head_on_pred", "pred_head_on_pred"):
            print(f"   {fam:5s} {nm:20s} 정지 {r[fam][nm]['static']}  이동 {r[fam][nm]['moving']}", flush=True)
        print(f"   {fam:5s} 최빈 정지 {r[fam]['majority']['static']} 이동 {r[fam]['majority']['moving']}")
    for g in ("static", "moving"):
        print(f"   진하기 {g}: 크기 비 {r['strength'][g]['norm_ratio']}  cos {r['strength'][g]['cos']}", flush=True)
json.dump(res, open("/workspace/world_model/vjepa2/z_analysis/results/static_vs_moving_objcell.json", "w"), indent=1)
