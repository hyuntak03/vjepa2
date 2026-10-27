"""굴린 예측에서 물체 정체 개념 (모양 + 색) 을 빼면 VOBench 영속 판별이 어떻게 되나 (2026-10-26, 학습 없음).
사용자: "(shape concept vector + color concept vector) (1, 1280) 해서 predictor 가 예측한 모든 frame 각각에 대해 (N, D) − (1, 1280) 해줘봐.
이랬을 때 permanence performance 어떻게 나오는지 궁금. 그리고 surprise gap figure 로".

데이터: VOBench 끝 가림 (캐시 /workspace/auto_research_cache/vobench_late, 블록 토큰 (16, 256, 1280) × 4 팔), 영속 (permanence) 중 도형 물체 36 block.
  짝 A = pos_a (물체 있음) vs imp_ab (문맥 끝 뒤 사라짐) — 사라짐 / 짝 B = pos_b (빈) vs imp_ba (나타남) — 나타남. 실사 물체 36 block 은 대응 개념이 없어 뺐다.
모델: 블록 AR natural_ar_vw_state, 문맥 8 블록 (16 장) → 굴린 예측 8 블록. surprise = mean |예측 − 정답 블록 토큰| (기존 채점과 같은 정의).
개념 벡터: 3D 합성 학습 쪽 굴린 예측 출력 (블록 전체 평균 (1, 1280)) 에서 걸음 k 마다 c_모양 · c_색 (클래스 평균 − 클래스 평균들의 평균).
  물체의 c = c_모양 + c_색 (캡슐은 3D 합성에 없어 c_색 만). 예측 블록 k 의 모든 칸 (256) 에서 α·c 를 뺀다.
대조: 같은 크기의 무작위 벡터 (등방, 3 회 평균) · 다른 정체 (모양 · 색을 무작위로 바꾼) 의 c.
출력: z_analysis/results/permanence_identity_subtract.json · z_analysis/figures/permanence_identity_subtract.png

  CUDA_VISIBLE_DEVICES=0 MEM_GB=3 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/permanence_identity_subtract.py
"""
import csv, json, os, sys
from collections import defaultdict
import numpy as np, torch

REPO = "/workspace/world_model/vjepa2"; sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "auto_research/scripts"))
dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEM_GB", 3)) * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory, 0)
_argv = list(sys.argv); sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT
sys.argv = _argv
S, C, K, D = 256, 8, 8, 1280
SHAPES = ["sphere", "cube", "cylinder", "cone", "pyramid", "torus"]
COLORS = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta"]
ALPHAS = (1.0, 2.0, 4.0)
rng = np.random.default_rng(0)

# ---- 개념 벡터 (3D 합성 학습 쪽 굴린 예측 출력, 걸음별)
TR = np.load("/workspace/auto_research_cache/ar_layers_pooled/train.npz")
lab, roout = TR["lab"], TR["roout"]
CON = {}
for k in range(1, 9):
    b = 7 + k; m = lab[:, b, 0] > 0; X = roout[m, k - 1].astype(np.float64)
    cs = np.stack([X[lab[m, b, 1] == i].mean(0) for i in range(6)]); cc = np.stack([X[lab[m, b, 2] == i].mean(0) for i in range(8)])
    CON[k] = (cs - cs.mean(0), cc - cc.mean(0))
del TR

# ---- VOBench 끝 가림 영속 · 도형
idx = defaultdict(dict)
for r in csv.DictReader(open(f"{REPO}/data_csv/vobench_v1/index.csv")):
    if r["occlusion"] == "late":
        idx[r["block_id"]][r["variant"]] = r
meta = json.load(open("/workspace/auto_research_cache/vobench_late/meta.json"))
H = np.load("/workspace/auto_research_cache/vobench_late/h.npy", mmap_mode="r")
ARMS = ["pos_a", "pos_b", "imp_ab", "imp_ba"]
ar = FT.load_pred("natural_ar_vw_state", "latest.pt").to(dev).eval()


def cvec(shape, color, k):
    cs, cc = CON[k]; v = cc[COLORS.index(color)].copy()
    if shape in SHAPES:
        v += cs[SHAPES.index(shape)]
    return v


recs = []
for mt in meta:
    r = idx[mt["block_id"]]["pos_a"]
    if r["violation"] != "permanence" or r["objset"] != "primitive":
        continue
    shape, color = r["obj_a"].split("/")
    hh = {v: torch.from_numpy(np.asarray(H[mt["i"], j], dtype=np.float32)).to(dev) for j, v in enumerate(ARMS)}
    for pid, (pos, imp, direc) in (("A", ("pos_a", "imp_ab", "vanish")), ("B", ("pos_b", "imp_ba", "appear"))):
        ctx = hh[pos][:C].reshape(1, C * S, D)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            p = ar.rollout_kv(ctx, torch.arange(C * S, device=dev).unsqueeze(0), K).float()[0].view(K, S, D)
        c_obj = torch.from_numpy(np.stack([cvec(shape, color, k) for k in range(1, 9)])).float().to(dev)          # (8, 1280)
        wshape = SHAPES[rng.integers(6)] if shape in SHAPES else shape
        wcolor = [c for c in COLORS if c != color][rng.integers(7)]
        c_wrong = torch.from_numpy(np.stack([cvec(wshape, wcolor, k) for k in range(1, 9)])).float().to(dev)
        sur = lambda q, v: float((q - hh[v][C:]).abs().mean())
        row = {"block": mt["block_id"], "dir": direc, "motion": r["motion"], "obj": r["obj_a"], "orig": sur(p, imp) - sur(p, pos)}
        for al in ALPHAS:
            q = p - al * c_obj[:, None, :]; row[f"minus_identity_a{al:g}"] = sur(q, imp) - sur(q, pos)
            q = p - al * c_wrong[:, None, :]; row[f"minus_wrong_a{al:g}"] = sur(q, imp) - sur(q, pos)
            g = []
            for _ in range(3):
                rv = torch.randn(K, D, device=dev); rv = rv / rv.norm(dim=1, keepdim=True) * c_obj.norm(dim=1, keepdim=True)
                q = p - al * rv[:, None, :]; g.append(sur(q, imp) - sur(q, pos))
            row[f"minus_random_a{al:g}"] = float(np.mean(g))
        row["c_norm_over_token_norm"] = float(c_obj.norm(dim=1).mean() / p.norm(dim=-1).mean())
        recs.append(row)
json.dump(recs, open(f"{REPO}/z_analysis/results/permanence_identity_subtract.json", "w"), indent=1)
conds = ["orig"] + [f"minus_{t}_a{al:g}" for al in ALPHAS for t in ("identity", "wrong", "random")]
for d in ("vanish", "appear"):
    R = [r for r in recs if r["dir"] == d]
    print(d, "n", len(R), {c: (round(float(np.mean([r[c] for r in R])), 4), round(100 * float(np.mean([r[c] > 0 for r in R])))) for c in conds})
print("개념 벡터 크기 / 예측 토큰 크기:", round(float(np.mean([r["c_norm_over_token_norm"] for r in recs])), 3))

# ---- 그림: (a) 사라짐 · (b) 나타남, 가로 = 운동 종류, 세로 = gap, 원본 / 정체 개념 뺌 / 무작위 뺌 (α = 1)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"
MOT = ["flat_static", "flat_vel", "flat_acc", "flat_dec", "ramp_down", "ramp_up"]; MLAB = ["static", "const. vel.", "accel.", "decel.", "ramp down", "ramp up"]
CND = [("orig", "original", "#2a78d6", -0.25), ("minus_identity_a1", "minus (shape + color) concept", "#eb6834", 0.0), ("minus_random_a1", "minus random (same norm)", "#1baf7a", 0.25)]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
for ax, d, title in zip(axes, ("vanish", "appear"), ("permanence: vanish", "permanence: appear")):
    for i, m in enumerate(MOT):
        R = [r for r in recs if r["dir"] == d and r["motion"] == m]
        for key, nm, col, off in CND:
            v = np.array([r[key] for r in R])
            ax.scatter(i + off + rng.uniform(-0.05, 0.05, len(v)), v, s=12, color=col, alpha=0.5, lw=0)
            ax.errorbar(i + off, v.mean(), yerr=v.std(), fmt="o", color=col, capsize=3, ms=5, lw=1.5, label=nm if i == 0 else None)
            ax.text(i + off, 1.0, f"{100 * (v > 0).mean():.0f}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7, color=col)
    ax.axhline(0, color="gray", lw=0.8, ls="--"); ax.set_xticks(range(len(MOT))); ax.set_xticklabels(MLAB, rotation=20, fontsize=9)
    ax.set_title(title, fontsize=11, pad=16); ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("surprise gap  (impossible − possible)")
for ax, lab_ in zip(axes, ("(a)", "(b)")):
    ax.text(0.5, -0.3, lab_, transform=ax.transAxes, ha="center", fontsize=11)
fig.tight_layout(rect=(0, 0.08, 1, 1))
h_, l_ = axes[0].get_legend_handles_labels(); fig.legend(h_, l_, loc="lower center", ncol=3, fontsize=9, frameon=False)
fig.savefig(f"{REPO}/z_analysis/figures/permanence_identity_subtract.png", dpi=170, bbox_inches="tight")
print("saved")
