"""VOBench 가능 vs 불가능 — encoder 영상 벡터에서 선형 방향으로 갈리나 (블록 단위로 남긴 블록에서) (2026-10-27, CPU, 모델 없음).
사용자: "차원이 막 섞여 있는데 possible, impossible 로 나뉘는 게 아니라… 이거 pos, imp linear probing 해서 그 weight 를 가지고 그려야 되나?"
입력: vobench_tsne_encoders.py 캐시 (VOBench 보임 288 편 = 72 블록 × 4 팔, 32 장 한 번에, context encoder · target encoder (LN)).
특징 (영상 하나 = 1280): 전체 평균 · 뒤 절반 평균 (17–32 장) · 변화 = 뒤 절반 − 앞 절반 (앞 절반 = 2 × 전체 − 뒤 절반).
방향 (블록 단위 6 겹: 학습 블록으로만 정하고 남긴 블록에 투영):
  ① 쌍 평균 차: 학습 블록의 문맥 같은 쌍 (imp_ab − pos_a, imp_ba − pos_b) 평균 → 선형, 정규화 없음
  ② 로지스틱 회귀 (sklearn 기본 L2, 표준화) — 대조
지표 (남긴 블록): 쌍 정확도 = 문맥 같은 쌍에서 불가능 점수 > 가능 점수 비율 (채점의 matched pair 와 같은 꼴, 반반 50 %) · 영상 정확도 (남긴 블록 중앙값 문턱) ·
  대조: 라벨을 블록 안에서 섞은 영 분포 (20 번).
⚠️ VOBench 보임에서 불가능 = 영상 중간에 정체가 바뀜 (사라짐 · 모양 · 색) 이고 가능 = 안 바뀜. 갈린다면 'encoder 가 정체 변화를 담는다' 이지 물리 이해가 아니다.
  /venv/vjepa/bin/python z_research/scripts/analysis/vobench_pos_imp_probe.py → z_analysis/results/vobench_pos_imp_probe.json · z_analysis/figures/vobench_tsne/pos_imp_probe.png
"""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
REPO = "/workspace/world_model/vjepa2"; z = np.load("/workspace/auto_research_cache/vobench_tsne/emb.npz"); rows = json.loads(str(z["rows"]))
var = np.array([r["variant"] for r in rows]); blk = np.array([r["block_id"] for r in rows]); viol = np.array([r["violation"] for r in rows])
imp = np.isin(var, ["imp_ab", "imp_ba"]).astype(int); ub = np.unique(blk); rng = np.random.default_rng(0); fold = {b: i % 6 for i, b in enumerate(rng.permutation(ub))}
fid = np.array([fold[b] for b in blk])
pairs = []                                                                             # (pos 색인, imp 색인) 문맥 같은 쌍
for b in ub:
    i = {v: np.nonzero((blk == b) & (var == v))[0][0] for v in ("pos_a", "pos_b", "imp_ab", "imp_ba")}
    pairs += [(i["pos_a"], i["imp_ab"]), (i["pos_b"], i["imp_ba"])]
pairs = np.array(pairs)
FEATS = {}
for enc in ("ctx", "tgt"):
    A, Fu = z[f"{enc}_all"].astype(np.float64), z[f"{enc}_future"].astype(np.float64)
    FEATS[f"{enc}_all"], FEATS[f"{enc}_future"], FEATS[f"{enc}_change"] = A, Fu, Fu - (2 * A - Fu)


def cv_scores(X, y, method):
    s = np.zeros(len(X))
    for f in range(6):
        tr, te = fid != f, fid == f
        if method == "pair_mean_diff":
            pt = pairs[tr[pairs[:, 0]]]; w = (X[pt[:, 1]] - X[pt[:, 0]]).mean(0); s[te] = X[te] @ w
        else:
            sc = StandardScaler().fit(X[tr]); clf = LogisticRegression(max_iter=5000).fit(sc.transform(X[tr]), y[tr]); s[te] = clf.decision_function(sc.transform(X[te]))
    return s


def metrics(s, y):
    pa = float((s[pairs[:, 1]] > s[pairs[:, 0]]).mean())
    va = float(np.mean([((s[fid == f] > np.median(s[fid == f])).astype(int) == y[fid == f]).mean() for f in range(6)]))
    return pa, va


R = {}; S = {}
for fn, X in FEATS.items():
    for m in ("pair_mean_diff", "logistic"):
        s = cv_scores(X, imp, m); pa, va = metrics(s, imp)
        null = []
        for _ in range(20):                                                            # 블록 안에서 팔 라벨을 섞은 영 분포
            # 같은 문맥 쌍 구조는 그대로 두고, 쌍마다 어느 쪽이 '불가능' 인지만 동전 던지기로 뒤섞는다 (라벨도 그에 맞춤)
            sw = rng.random(len(pairs)) < 0.5; pr = pairs.copy(); pr[sw] = pr[sw][:, ::-1]
            yp = np.zeros_like(imp); yp[pr[:, 1]] = 1
            pairs_bak = pairs; pairs = pr; sn = cv_scores(X, yp, m); null.append(metrics(sn, yp)[0]); pairs = pairs_bak
        R[f"{fn}__{m}"] = {"pair_accuracy": round(pa, 3), "video_accuracy": round(va, 3), "null_pair_accuracy_mean": round(float(np.mean(null)), 3), "null_pair_accuracy_max": round(float(np.max(null)), 3),
                           "pair_accuracy_by_violation": {v: round(float((s[pairs[viol[pairs[:, 0]] == v][:, 1]] > s[pairs[viol[pairs[:, 0]] == v][:, 0]]).mean()), 3) for v in ("permanence", "shape", "color")}}
        S[f"{fn}__{m}"] = s
        print(fn, m, R[f"{fn}__{m}"], flush=True)
json.dump(R, open(f"{REPO}/z_analysis/results/vobench_pos_imp_probe.json", "w"), indent=1)

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))
for ri, enc in enumerate(("ctx", "tgt")):
    for ci, (ft, ftn) in enumerate((("all", "mean of all 32 frames"), ("future", "mean of frames 17–32"), ("change", "frames 17–32 minus frames 1–16"))):
        ax = axes[ri, ci]; key = f"{enc}_{ft}__pair_mean_diff"; s = S[key]; r = R[key]
        sd = s[pairs].std(); x0 = np.arange(len(pairs))
        order = np.argsort(s[pairs[:, 0]])                                             # 가능 점수 순으로 쌍을 늘어놓음
        for j, k in enumerate(order):
            p_, q_ = pairs[k]; ax.plot([j, j], [s[p_] / sd, s[q_] / sd], color="#cccccc", lw=0.6, zorder=1)
        ax.scatter(x0, s[pairs[order, 0]] / sd, s=9, color="#2a78d6", label="possible", zorder=2)
        ax.scatter(x0, s[pairs[order, 1]] / sd, s=9, color="#eb6834", marker="x", label="impossible", zorder=3)
        ax.set_xticks([]); ax.set_xlabel(f"same-context pairs (held-out blocks), sorted by possible score\n"
                                         f"pair acc {r['pair_accuracy'] * 100:.0f} % (shuffled ≤ {r['null_pair_accuracy_max'] * 100:.0f} %) · video acc {r['video_accuracy'] * 100:.0f} %")
        ax.set_title(("context encoder · " if enc == "ctx" else "target encoder (LN) · ") + ftn, fontsize=10)
        if ci == 0:
            ax.set_ylabel("projection on the held-out\nimpossible − possible direction (std units)")
axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
fig.suptitle("VOBench visible: do encoder video vectors separate possible from impossible along one direction? (6-fold, split by block)", fontsize=12, y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(f"{REPO}/z_analysis/figures/vobench_tsne/pos_imp_probe.png", dpi=160, bbox_inches="tight")
print("saved")
