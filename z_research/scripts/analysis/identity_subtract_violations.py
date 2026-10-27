"""[2026-10-26 저녁 갱신: 물체 칸 개념 벡터를 VOBench 물체 칸 (label.json 위치) 에서만 뺀다]
굴린 예측에서 문맥 물체의 정체 개념 (모양 + 색) 을 빼면 VOBench 위반 판별 (영속 · 모양 · 색) 이 어떻게 되나 (2026-10-26, 학습 없음).
사용자: "figure 를 subtraction (minus random 은 제외하고) violation type 별로 3 개". permanence_identity_subtract.py 를 위반 세 종류로 넓힌 판.

VOBench 끝 가림, 도형 물체 block (위반마다 36). 짝 A = pos_a vs imp_ab (문맥 물체 a), 짝 B = pos_b vs imp_ba (문맥 물체 b; 영속은 b = 빈).
빼는 벡터 = 문맥 물체의 c_모양 + c_색 (3D 합성 굴린 예측 출력에서 걸음별로 구한 개념 벡터, 블록 전체 평균 기준; 캡슐은 c_색 만, 빈 물체는 뺄 것 없음 → 원본).
  예측 블록 k 의 모든 칸 (256) 에서 α·c 를 뺀다. 그 밖 정의는 permanence_identity_subtract.py 와 같다.

  CUDA_VISIBLE_DEVICES=0 MEM_GB=3 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/identity_subtract_violations.py
출력: z_analysis/results/identity_subtract_violations.json · z_analysis/figures/identity_subtract_violations.png
"""
import csv, json, os, sys
from collections import defaultdict
from PIL import Image
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
ALPHAS = (1.0, 4.0)

# 2026-10-26 저녁 갱신: 화면 평균 개념 → 물체 칸 개념 (사용자: "물체 칸 concept vector 로 지금 결과들 다 바꿔줘"). 화면 평균 개념은 물체가 아니라 화면 전체에 퍼진 정체 신호였다 (concept_vector_regions.json)
OC = np.load("/workspace/auto_research_cache/objcell_concept/ar_train.npz"); _meta = {m["i"]: m for m in json.load(open("/workspace/auto_research_cache/ar_layers/train_meta.json"))}
_sh = np.array([_meta[int(i)]["shape"] for i in OC["i"]]); _co = np.array([_meta[int(i)]["color"] for i in OC["i"]]); _PT = OC["pred_true"].astype(np.float32)
CON = {}
for k in range(1, 9):
    X = _PT[:, k - 1]; m = ~np.isnan(X).any(1); X, s_, c_ = X[m], _sh[m], _co[m]
    CON[k] = (np.stack([X[s_ == i].mean(0) for i in range(6)]) - X.mean(0), np.stack([X[c_ == i].mean(0) for i in range(8)]) - X.mean(0))
del OC, _PT


def cvec(obj, k):
    if obj in ("", "empty"):
        return np.zeros(D)
    shape, color = obj.split("/"); cs, cc = CON[k]
    v = cc[COLORS.index(color)].copy() if color in COLORS else np.zeros(D)
    if shape in SHAPES:
        v += cs[SHAPES.index(shape)]
    return v



def cvec_parts(obj, k):
    """(c_모양, c_색) 따로. 빈 물체 · 모르는 모양은 0."""
    if obj in ("", "empty"):
        return np.zeros(D), np.zeros(D)
    shape, color = obj.split("/"); cs, cc = CON[k]
    return (cs[SHAPES.index(shape)] if shape in SHAPES else np.zeros(D)), (cc[COLORS.index(color)] if color in COLORS else np.zeros(D))


# 2026-10-26 사용자: "magnitude 를 키워 보겠다 이거지? 해 보던가" — 더하기 (크기 키우기). 되돌림 배수 = 1/r − 1, r = 정지 물체 물체 칸 개념 크기 비 (concept_norm_static_vs_moving.json, AR)
_NR = json.load(open(f"{REPO}/z_analysis/results/concept_norm_static_vs_moving.json"))["ar"]
RESTORE = {k: (1 / _NR["shape"]["static"][k - 1]["norm_ratio"] - 1, 1 / _NR["color"]["static"][k - 1]["norm_ratio"] - 1) for k in range(1, 9)}
BETAS = (1.0, 2.0, 4.0)

VROOT = "/workspace/dataset/benchmarks/VOBench_v1"
IDX = list(range(0, 96, 3))[:32]


def obj_cells(folder, key):
    """VOBench label.json 의 물체 key (A / B) 위치로 블록 8–15 마다 물체 칸 (블록 두 장 상자 합집합 안에 중심이 든 칸; 비면 중심 칸)."""
    L_ = json.load(open(os.path.join(VROOT, folder, "label.json")))["labels"].get(key, [])
    W0 = Image.open(os.path.join(VROOT, folder, "pos_a", "000000.png")).size[0]; lab = {q["f"]: q for q in L_}; sc = 256 / W0; cc = np.arange(16) * 16 + 8.0; out = []
    for b in range(8, 16):
        q = [lab.get(IDX[2 * b]), lab.get(IDX[2 * b + 1])]
        if any(t is None for t in q):
            out.append([]); continue
        x0 = min(t["u"] - t["ext_u"] for t in q) * sc; x1 = max(t["u"] + t["ext_u"] for t in q) * sc; y0 = min(t["v"] - t["ext_v"] for t in q) * sc; y1 = max(t["v"] + t["ext_v"] for t in q) * sc
        cols = np.nonzero((cc >= x0) & (cc <= x1))[0]; rows = np.nonzero((cc >= y0) & (cc <= y1))[0]
        cells = [int(r * 16 + c) for r in rows for c in cols]
        if not cells:
            u = np.mean([t["u"] for t in q]) * sc; v = np.mean([t["v"] for t in q]) * sc; cells = [int(min(15, v // 16) * 16 + min(15, u // 16))]
        out.append(cells)
    return out


idx = defaultdict(dict)
for r in csv.DictReader(open(f"{REPO}/data_csv/vobench_v1/index.csv")):
    if r["occlusion"] == "late":
        idx[r["block_id"]][r["variant"]] = r
meta = json.load(open("/workspace/auto_research_cache/vobench_late/meta.json"))
H = np.load("/workspace/auto_research_cache/vobench_late/h.npy", mmap_mode="r")
ARMS = ["pos_a", "pos_b", "imp_ab", "imp_ba"]
ar = FT.load_pred("natural_ar_vw_state", "latest.pt").to(dev).eval()
recs = []
for mt in meta:
    r = idx[mt["block_id"]]["pos_a"]
    if r["objset"] != "primitive":
        continue
    hh = {v: torch.from_numpy(np.asarray(H[mt["i"], j], dtype=np.float32)).to(dev) for j, v in enumerate(ARMS)}
    for pid, (pos, imp, obj) in (("A", ("pos_a", "imp_ab", r["obj_a"])), ("B", ("pos_b", "imp_ba", r["obj_b"]))):
        ctx = hh[pos][:C].reshape(1, C * S, D)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            p = ar.rollout_kv(ctx, torch.arange(C * S, device=dev).unsqueeze(0), K).float()[0].view(K, S, D)
        c = torch.from_numpy(np.stack([cvec(obj, k) for k in range(1, 9)])).float().to(dev)
        folder = os.path.dirname(r["file_name"]); cells = obj_cells(folder, pid)
        cells_sc = [sorted(set(a_) | set(b_)) for a_, b_ in zip(obj_cells(folder, "A"), obj_cells(folder, "B"))]   # 채점용 물체 칸 = 두 물체 (A · B) 칸 합 (2026-10-26 사용자: "물체 칸 해보던가")
        mask_sc = torch.zeros(K, S, device=dev)
        for j_, cl in enumerate(cells_sc):
            mask_sc[j_, cl] = 1.0
        mask = torch.zeros(K, S, 1, device=dev)
        for j_, cl in enumerate(cells):
            mask[j_, cl] = 1.0
        sur = lambda q, v: float((q - hh[v][C:]).abs().mean())
        sur_obj = lambda q, v: float(((q - hh[v][C:]).abs().mean(-1) * mask_sc).sum() / mask_sc.sum().clamp(min=1))   # 물체 칸만 평균
        d = ("vanish" if pid == "A" else "appear") if r["violation"] == "permanence" else "pair"
        row = {"block": mt["block_id"], "violation": r["violation"], "dir": d, "motion": r["motion"], "obj": obj, "orig": sur(p, imp) - sur(p, pos), "orig_obj": sur_obj(p, imp) - sur_obj(p, pos), "n_obj_cells": [len(cl) for cl in cells], "n_score_cells": [len(cl) for cl in cells_sc]}
        for al in ALPHAS:
            q = p - al * c[:, None, :] * mask; row[f"minus_identity_a{al:g}"] = sur(q, imp) - sur(q, pos)   # 물체 칸에서만
            row[f"minus_identity_a{al:g}_obj"] = sur_obj(q, imp) - sur_obj(q, pos)
        for be in BETAS:                                                    # 더하기: 문맥 물체 정체 개념을 물체 칸에
            q = p + be * c[:, None, :] * mask; row[f"plus_identity_b{be:g}"] = sur(q, imp) - sur(q, pos); row[f"plus_identity_b{be:g}_obj"] = sur_obj(q, imp) - sur_obj(q, pos)
        cr = torch.from_numpy(np.stack([RESTORE[k][0] * cvec_parts(obj, k)[0] + RESTORE[k][1] * cvec_parts(obj, k)[1] for k in range(1, 9)])).float().to(dev)
        q = p + cr[:, None, :] * mask; row["plus_restore"] = sur(q, imp) - sur(q, pos); row["plus_restore_obj"] = sur_obj(q, imp) - sur_obj(q, pos)
        recs.append(row)
json.dump(recs, open(f"{REPO}/z_analysis/results/identity_subtract_violations.json", "w"), indent=1)
VIO = ["permanence", "shape", "color"]
for v in VIO:
    R = [r for r in recs if r["violation"] == v]
    print(v, "n", len(R), {c: (round(float(np.mean([r[c] for r in R])), 4), round(100 * float(np.mean([r[c] > 0 for r in R])))) for c in ("orig", "minus_identity_a1", "minus_identity_a4")})

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"
MOT = ["flat_static", "flat_vel", "flat_acc", "flat_dec", "ramp_down", "ramp_up"]; MLAB = ["static", "const. vel.", "accel.", "decel.", "ramp down", "ramp up"]
CND = [("orig", "original", "#2a78d6", -0.15), ("minus_identity_a1", "minus object-cell (shape + color) concept, at object cells", "#eb6834", 0.15)]
rng = np.random.default_rng(0)
fig, axes = plt.subplots(1, 3, figsize=(18, 4.6), sharey=True)
for ax, v in zip(axes, VIO):
    for i, m in enumerate(MOT):
        R = [r for r in recs if r["violation"] == v and r["motion"] == m]
        for key, nm, col, off in CND:
            x = np.array([r[key] for r in R])
            ax.scatter(i + off + rng.uniform(-0.05, 0.05, len(x)), x, s=12, color=col, alpha=0.5, lw=0)
            ax.errorbar(i + off, x.mean(), yerr=x.std(), fmt="o", color=col, capsize=3, ms=5, lw=1.5, label=nm if i == 0 else None)
            ax.text(i + off, 1.0, f"{100 * (x > 0).mean():.0f}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7, color=col)
    ax.axhline(0, color="gray", lw=0.8, ls="--"); ax.set_xticks(range(len(MOT))); ax.set_xticklabels(MLAB, rotation=20, fontsize=9)
    ax.set_title(v, fontsize=11, pad=16); ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("surprise gap  (impossible − possible)")
for ax, lab_ in zip(axes, ("(a)", "(b)", "(c)")):
    ax.text(0.5, -0.3, lab_, transform=ax.transAxes, ha="center", fontsize=11)
fig.tight_layout(rect=(0, 0.08, 1, 1))
h_, l_ = axes[0].get_legend_handles_labels(); fig.legend(h_, l_, loc="lower center", ncol=2, fontsize=9, frameon=False)
if os.environ.get("PLOT") == "1":                                       # 2026-10-26 사용자가 그림을 지웠다 — 기본은 그리지 않는다
    fig.savefig(f"{REPO}/z_analysis/figures/identity_subtract_violations.png", dpi=170, bbox_inches="tight")
print("saved")
