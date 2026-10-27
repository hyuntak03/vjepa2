"""VOBench 가능 vs 불가능 영상의 encoder 임베딩 — 32 장 한 번에, context encoder · target encoder, t-SNE (2026-10-27, CPU).
사용자: "32 frame 다 넣고 한번 possible impossible 봐 봐. T-SNE 로 얼마나 차이 나는지 보자. context encoder 기준. target encoder 도 해 보면 좋고".
영상: VOBench_v1 보임 (occlusion = visible) 288 편 = 72 블록 × 4 팔 (pos_a · pos_b 가능 / imp_ab · imp_ba 불가능), 사물 2 종 (primitive · realistic) × 위반 3 종 (영속 · 모양 · 색).
  32 장 = 원본 0, 3, …, 93 장 (채점과 같은 간격, 장 0.1 s) → 256² bilinear (antialias 없음) → V-JEPA 2 ViT-H 에 한 번에 (토큰 16 × 256).
  context encoder = 체크포인트 `encoder` (online), 출력 그대로 · target encoder = `target_encoder` (EMA), affine 없는 LN (채점과 같다). CPU bf16.
임베딩 (영상 하나 = 벡터 하나, 1280): 전체 평균 (토큰 4,096 개) · 뒤 절반 평균 (블록 8–15, 위반이 일어나는 쪽).
t-SNE: scikit-learn (주성분 50 → perplexity 30, PCA 초기화, 시드 0).
얼마나 다른가 (숫자): 같은 블록 안에서 cos 거리 — 문맥이 같은 가능 · 불가능 쌍 (pos_a–imp_ab, pos_b–imp_ba) vs 가능 · 가능 (pos_a–pos_b) vs 다른 블록끼리 (기준).
  /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/vobench_tsne_encoders.py
출력: z_analysis/results/vobench_tsne_encoders.json · 그림 z_analysis/figures/vobench_tsne/tsne_*.png · 캐시 /workspace/auto_research_cache/vobench_tsne/emb.npz
"""
import argparse, csv, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument("--threads", type=int, default=96); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--plot_only", action="store_true"); a = ap.parse_args()
torch.set_num_threads(a.threads)
REPO = "/workspace/world_model/vjepa2"; sys.path.insert(0, REPO)
CA = "/workspace/auto_research_cache/vobench_tsne"; os.makedirs(CA, exist_ok=True)
FIG = f"{REPO}/z_analysis/figures/vobench_tsne"; os.makedirs(FIG, exist_ok=True)
VROOT = "/workspace/dataset/benchmarks/VOBench_v1"; IDX = list(range(0, 96, 3))[:32]; S, D = 256, 1280
rows = [r for r in csv.DictReader(open(f"{REPO}/data_csv/vobench_v1/index.csv")) if r["occlusion"] == "visible"]
rows.sort(key=lambda r: (r["block_id"], r["variant"]))
if a.limit:
    rows = rows[: a.limit]

if not a.plot_only:
    from analysis.intphys2.model import build_from_config
    CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
    MCFG = {"arch_name": "vit_huge", "checkpoint": CK, "img_size": 256, "patch_size": 16, "tubelet_size": 2, "use_rope": True, "uniform_power": False, "dual_encoder": True,
            "context_encoder_key": "encoder", "target_encoder_key": "target_encoder", "predictor": {"embed_dim": 384, "depth": 12, "num_heads": 12, "num_mask_tokens": 10},
            "dtype": "float32", "window_size": 32}
    rel = build_from_config(MCFG, torch.device("cpu")); ctx, tgt = rel.context_encoder.float().eval(), rel.target_encoder.float().eval()
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
    E = {k: [] for k in ("ctx_all", "ctx_future", "tgt_all", "tgt_future")}; t0 = time.time()
    for n, r in enumerate(rows):
        d = os.path.join(VROOT, r["file_name"])
        fr = np.stack([np.asarray(Image.open(os.path.join(d, f"{f:06d}.png")).convert("RGB")) for f in IDX])
        x = torch.from_numpy(fr).permute(3, 0, 1, 2).float() / 255.0
        x = F.interpolate(x.permute(1, 0, 2, 3), size=(256, 256), mode="bilinear", align_corners=False, antialias=False).permute(1, 0, 2, 3)
        xn = ((x - MEAN) / STD).unsqueeze(0)
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            hc = ctx(xn).float()[0].view(16, S, D); ht = F.layer_norm(tgt(xn).float(), (D,))[0].view(16, S, D)
        E["ctx_all"].append(hc.mean((0, 1)).numpy()); E["ctx_future"].append(hc[8:].mean((0, 1)).numpy())
        E["tgt_all"].append(ht.mean((0, 1)).numpy()); E["tgt_future"].append(ht[8:].mean((0, 1)).numpy())
        if n < 2 or (n + 1) % 24 == 0:
            print(f"  {n + 1}/{len(rows)} · {(time.time() - t0) / (n + 1):.1f} s/편", flush=True)
    np.savez(f"{CA}/emb.npz", **{k: np.stack(v) for k, v in E.items()}, rows=json.dumps(rows))

z = np.load(f"{CA}/emb.npz"); rows = json.loads(str(z["rows"]))
var = np.array([r["variant"] for r in rows]); blk = np.array([r["block_id"] for r in rows]); viol = np.array([r["violation"] for r in rows]); obj = np.array([r["objset"] for r in rows])
plaus = np.isin(var, ["pos_a", "pos_b"])


def tsne(X, perp=30.0, seed=0):
    """sklearn t-SNE (2026-10-27 사용자 "라이브러리 설치하면 되지" → scikit-learn 1.9.1 설치). 주성분 50 → perplexity 30, PCA 초기화, 시드 0."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    Z = PCA(n_components=50, random_state=seed).fit_transform(X)
    return TSNE(n_components=2, perplexity=perp, init="pca", random_state=seed).fit_transform(Z)


cosd = lambda u, v: 1 - (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
R = {"n_videos": int(len(rows))}
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
for key, ttl in (("ctx_all", "context encoder, mean of all 32 frames"), ("ctx_future", "context encoder, mean of frames 17–32"),
                 ("tgt_all", "target encoder (LN), mean of all 32 frames"), ("tgt_future", "target encoder (LN), mean of frames 17–32")):
    X = z[key].astype(np.float64)
    # 같은 블록 안 거리
    dd = {"possible_vs_impossible_same_context": [], "possible_vs_possible": [], "impossible_vs_impossible": []}
    for b in np.unique(blk):
        i = {v: np.nonzero((blk == b) & (var == v))[0][0] for v in ("pos_a", "pos_b", "imp_ab", "imp_ba")}
        dd["possible_vs_impossible_same_context"] += [cosd(X[i["pos_a"]], X[i["imp_ab"]]), cosd(X[i["pos_b"]], X[i["imp_ba"]])]
        dd["possible_vs_possible"].append(cosd(X[i["pos_a"]], X[i["pos_b"]])); dd["impossible_vs_impossible"].append(cosd(X[i["imp_ab"]], X[i["imp_ba"]]))
    rng = np.random.default_rng(0); pa, pb = rng.integers(0, len(X), 4000), rng.integers(0, len(X), 4000); okp = blk[pa] != blk[pb]
    dd["different_blocks"] = list(cosd(X[pa[okp]], X[pb[okp]]))
    R[key] = {"cos_distance_median": {k: round(float(np.median(v)), 5) for k, v in dd.items()}}
    # 가능 / 불가능만으로 갈리나: 최근접 이웃 (자기 블록 제외) 의 가능 여부 일치율
    Dm = ((X[:, None] - X[None]) ** 2).sum(-1); Dm[blk[:, None] == blk[None]] = np.inf
    R[key]["nearest_other_block_same_plausibility"] = round(float((plaus[Dm.argmin(1)] == plaus).mean()), 3)
    Y = tsne(X)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    ax = axes[0]
    for b in np.unique(blk):                                                          # 문맥이 같은 가능 – 불가능 쌍을 선으로
        i = {v: np.nonzero((blk == b) & (var == v))[0][0] for v in ("pos_a", "pos_b", "imp_ab", "imp_ba")}
        for p_, q_ in (("pos_a", "imp_ab"), ("pos_b", "imp_ba")):
            ax.plot(*Y[[i[p_], i[q_]]].T, color="#bbbbbb", lw=0.6, zorder=1)
    ax.scatter(*Y[plaus].T, s=16, color="#2a78d6", label="possible", zorder=2); ax.scatter(*Y[~plaus].T, s=16, color="#eb6834", marker="x", label="impossible", zorder=3)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("(a) possible vs impossible (gray line: same-context pair)"); ax.legend(frameon=False, fontsize=9, loc="best")
    ax = axes[1]; cols = {"permanence": "#2a78d6", "shape": "#eb6834", "color": "#1baf7a"}; mk = {"primitive": "o", "realistic": "^"}
    for v_ in cols:
        for o_ in mk:
            s_ = (viol == v_) & (obj == o_); ax.scatter(*Y[s_].T, s=16, color=cols[v_], marker=mk[o_], label=f"{v_} · {o_}")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("(b) same map, colored by violation type · object set"); ax.legend(frameon=False, fontsize=8, loc="best", ncol=2)
    fig.suptitle(f"VOBench visible, 288 videos — {ttl} — t-SNE", fontsize=12, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{FIG}/tsne_{key}.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    print(key, R[key], flush=True)
json.dump(R, open(f"{REPO}/z_analysis/results/vobench_tsne_encoders.json", "w"), indent=1)
print("saved")
