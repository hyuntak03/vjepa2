"""VOBench 자체로 정체 (모양 · 색) 개념 벡터를 뽑을 수 있나 — 검증 (2026-10-26 저녁, CPU).
사용자: "VOBench 에서도 concept vector 뽑을 수 있어? 가능한지 검증해 봐".
입력: vobench_concept_cache.py (VOBench 도형 가능 영상 720 개, 물체 보이는 블록 8,280 표본, encoder 물체 칸 평균 · 화면 평균).
나눔은 모두 block_id (가능 · 불가능 4 팔 묶음) 단위 — 같은 영상의 블록끼리는 닮아서 표본 단위로 나누면 부풀어 오른다.
  (1) 클래스별 표본 · 영상 수
  (2) 반쪽 재현도: block_id 짝 / 홀로 나눈 두 개념 벡터의 cos. 영 기준 = 영상 단위로 라벨을 섞은 값
  (3) 일반성: 배경 (v11 / assets) · 운동 (정지 / 움직임) · 가림 조건 (보임 / 가림 셋) 으로 나눈 두 개념의 cos
  (4) 판독: 선형층 (모양 · 색 따로, 입력 표준화, AdamW lr 1e-3 · wd 1e-2 · 최대 60 epoch) 을 block_id 짝에서 학습 → 홀에서 정확도. 물체 칸 · 화면 평균
  (5) 3D 합성과 비교: 개념 cos (공통 클래스) · 3D 합성 물체 칸으로 학습한 판독기를 VOBench 물체 칸에 그대로 (도메인 이식)
  /venv/vjepa/bin/python z_research/scripts/analysis/vobench_concept_validate.py   → z_analysis/results/vobench_concept_validate.json
"""
import copy, json
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(48)
V = np.load("/workspace/auto_research_cache/objcell_concept/vobench_all.npz")
OBJ, WH = V["obj"].astype(np.float32), V["whole"].astype(np.float32); SH, CO, BID, BG, OCC, MOT = V["shape"], V["color"], V["block_id"].astype(int), V["background"], V["occlusion"], V["motion"]
vid = np.array([f"{b}_{a}" for b, a in zip(BID, V["arm"])])
SH6 = ["sphere", "cube", "cylinder", "cone", "pyramid", "torus"]; COL = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta"]
rng = np.random.default_rng(0)
cos = lambda A, B: float(np.mean((A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))))


def concept(X, y, classes):
    M = np.stack([X[y == c].mean(0) for c in classes]); return M - X[np.isin(y, classes)].mean(0)


def vid_shuffle(y):
    u = np.unique(vid); lab = {v: y[vid == v][0] for v in u}; perm = dict(zip(u, rng.permutation([lab[v] for v in u]))); return np.array([perm[v] for v in vid])


def fit_eval(Xt, yt, Xv, yv, K):
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6; Xt, Xv = torch.from_numpy((Xt - mu) / sd), torch.from_numpy((Xv - mu) / sd)
    torch.manual_seed(0); y = torch.from_numpy(yt); h = torch.from_numpy(np.arange(len(yt)) % 5 == 0)
    head = torch.nn.Linear(Xt.shape[1], K); opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2); best = (-1, None)
    for ep in range(60):
        perm = torch.nonzero(~h)[:, 0][torch.randperm(int((~h).sum()))]
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]; loss = F.cross_entropy(head(Xt[idx]), y[idx]); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            a = (head(Xt[h]).argmax(1) == y[h]).float().mean().item()
        if a > best[0]:
            best = (a, copy.deepcopy(head.state_dict()))
    head.load_state_dict(best[1])
    with torch.no_grad():
        return round(float((head(Xv).argmax(1).numpy() == yv).mean() * 100), 1)


res = {"n_samples": int(len(OBJ)), "n_videos": int(len(np.unique(vid)))}
even = BID % 2 == 0
for fam, y, ALL in (("shape", SH, sorted(set(SH))), ("color", CO, COL)):
    classes = [c for c in ALL if c in set(y)]; yi = np.array([classes.index(c) for c in y]); r = {"classes": classes,
         "n_samples": [int((y == c).sum()) for c in classes], "n_videos": [int(len(np.unique(vid[y == c]))) for c in classes]}
    ys = vid_shuffle(y)
    r["reliability"] = round(cos(concept(OBJ[even], y[even], classes), concept(OBJ[~even], y[~even], classes)), 3)
    r["reliability_null"] = round(cos(concept(OBJ[even], ys[even], classes), concept(OBJ[~even], ys[~even], classes)), 3)
    for nm, mA in (("background_v11_vs_assets", BG == "v11"), ("static_vs_moving", MOT == "flat_static"), ("visible_vs_occluded_conditions", OCC == "visible")):
        r[nm] = round(cos(concept(OBJ[mA], y[mA], classes), concept(OBJ[~mA], y[~mA], classes)), 3)
        r[nm + "_null"] = round(cos(concept(OBJ[mA], ys[mA], classes), concept(OBJ[~mA], ys[~mA], classes)), 3)
    r["readout_object_cells"] = fit_eval(OBJ[even], yi[even], OBJ[~even], yi[~even], len(classes))
    r["readout_whole_frame"] = fit_eval(WH[even], yi[even], WH[~even], yi[~even], len(classes))
    r["majority"] = round(float(np.bincount(yi[~even]).max() / (~even).sum() * 100), 1)
    res[fam] = r; print(fam, r, flush=True)
# 3D 합성과 비교
Z = np.load("/workspace/auto_research_cache/objcell_concept/ar_train.npz"); tm = {m["i"]: m for m in json.load(open("/workspace/auto_research_cache/ar_layers/train_meta.json"))}
P = Z["tgt_true"].astype(np.float32).reshape(-1, 1280); psh = np.repeat([SH6[tm[int(i)]["shape"]] for i in Z["i"]], 8); pco = np.repeat([COL[tm[int(i)]["color"]] for i in Z["i"]], 8)
ok = ~np.isnan(P).any(1); P, psh, pco = P[ok], psh[ok], pco[ok]
for fam, yv, yp, ALL in (("shape", SH, psh, SH6), ("color", CO, pco, COL)):
    common = [c for c in ALL if c in set(yv) and c in set(yp)]
    Cv, Cp = concept(OBJ, yv, common), concept(P, yp, common)
    res[fam]["cos_vs_phys3d_common_classes"] = {"classes": common, "cos_mean": round(cos(Cv, Cp), 3)}
    mp, mv = np.isin(yp, ALL), np.isin(yv, ALL)
    res[fam]["phys3d_head_on_vobench_object_cells"] = fit_eval(P[mp][::4], np.array([ALL.index(c) for c in yp[mp][::4]]), OBJ[mv], np.array([ALL.index(c) for c in yv[mv]]), len(ALL))
    print(fam, "3D 합성 비교", res[fam]["cos_vs_phys3d_common_classes"], "3D 합성 판독기 → VOBench", res[fam]["phys3d_head_on_vobench_object_cells"], flush=True)
json.dump(res, open("/workspace/world_model/vjepa2/z_analysis/results/vobench_concept_validate.json", "w"), indent=1)
