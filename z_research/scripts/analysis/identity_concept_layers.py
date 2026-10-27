"""정체 (모양 · 색) 개념 벡터의 크기와 encoder 개념 벡터와의 cos — 블록 AR 층별 · 시점별 (2026-10-26, 학습 없음, CPU).
사용자: "layer 별로 각 시점별로 물체 identity concept vector 자체의 magnitude 를 좀 봐주라. 그리고 각 layer 별로 encoder concept 와의 cos sim 도 같이".

입력 pool_objcells.py 출력 (블록마다 칸 256 개 평균, 원래 좌표 — 표준화 안 함). 물체가 보이는 블록만.
개념 벡터 c_k = m_k − (클래스 평균들의 평균), shape 6 · color 8, (시점, 표현) 마다 따로.
시점: teacher forcing 위치 b = 0..15 (층 0..13 은 입력 블록 b 기준, 출력은 tf_out[b−1] = 블록 b 예측) · 굴린 예측 걸음 k = 1..8 (블록 7+k; 층은 걸음 1..7, 출력은 1..8).
크기: ‖c_k‖ 의 클래스 평균 (원래 크기) · 상대 크기 = ‖c_k‖ / RMS(‖x − x̄‖) (그 표현 · 시점의 표본 퍼짐 대비).
encoder 와의 cos (같은 블록의 encoder 개념 벡터 c_enc 와, 클래스마다 cos 후 평균):
  층 (384): 입력 쪽 = cos(c_L, W_embed c_enc) — encoder 개념을 predictor 입력 임베딩 (선형 1280→384) 에 통과. 평균 차라 bias · type 은 지워진다 → TF 층 0 은 1 이어야 한다 (검산)
            출력 쪽 = cos(W_proj c_L, c_enc) — 층 개념을 predictor 출력 투영 (선형 384→1280) 에 통과 (출력 LN 은 무시)
  출력 (1280): cos(c_out, c_enc) 직접
대조: 클래스 짝을 섞은 cos (c_L 의 k 와 c_enc 의 다른 k').
2026-10-26 추가 (사용자: "상대 크기를 predictor / encoder 로"): 같은 블록 encoder 대비 크기 비 셋
  ratio_rel = (‖c_L‖ / RMS_L) ÷ (‖c_enc‖ / RMS_enc)   — 자기 공간 퍼짐 대비 구분 세기의 비 (주 지표, 단위 없음)
  ratio_in  = ‖c_L‖ ÷ ‖W_embed c_enc‖                 — 384 공간, TF 층 0 은 1 (검산)
  ratio_out = ‖W_proj c_L‖ ÷ ‖c_enc‖                  — 1280 공간. 출력은 ‖c_out‖ ÷ ‖c_enc‖ (둘 다 칸마다 LN 공간)

  /venv/vjepa/bin/python z_research/scripts/analysis/identity_concept_layers.py
출력: z_analysis/results/identity_concept_layers.json · z_analysis/figures/identity_concept_layers_{shape,color}.png
"""
import json, os, sys
import numpy as np, torch

REPO = "/workspace/world_model/vjepa2"; sys.path.insert(0, REPO)
P = "/workspace/auto_research_cache/ar_layers_pooled"
OUTJ = f"{REPO}/z_analysis/results/identity_concept_layers.json"; FIG = f"{REPO}/z_analysis/figures"
TR = dict(np.load(f"{P}/train.npz"))
from analysis.intphys2.model import _clean_backbone_key  # noqa: E402
sd_ = _clean_backbone_key(torch.load(f"{REPO}/z_training/runs/natural_ar_vw_state/latest.pt", map_location="cpu", weights_only=False)["predictor"])
W_embed = sd_["predictor_embed.weight"].float().numpy()          # (384, 1280)
W_proj = sd_["predictor_proj.weight"].float().numpy()            # (1280, 384)
lab = TR["lab"]; ok = lab[..., 0] > 0
FAM = {"shape": (1, 6), "color": (2, 8)}


def concept(X, y, K):
    M = np.stack([X[y == k].mean(0) for k in range(K)]); return M - M.mean(0)


def stats(X, y, K):
    C = concept(X, y, K)
    rms = np.sqrt(((X - X.mean(0)) ** 2).sum(1).mean())
    return C, float(np.linalg.norm(C, axis=1).mean()), float(np.linalg.norm(C, axis=1).mean() / rms)


def ccos(A, B):
    """클래스마다 cos 평균 · 짝 섞은 대조."""
    na, nb = A / np.linalg.norm(A, axis=1, keepdims=True), B / np.linalg.norm(B, axis=1, keepdims=True)
    S = na @ nb.T
    return float(np.mean(np.diag(S))), float(np.mean(S[~np.eye(len(S), dtype=bool)]))


out = {}
for fam, (col, K) in FAM.items():
    R = {"enc": {}, "tf": {}, "ro": {}}
    Cenc, Renc = {}, {}
    for b in range(16):
        m = ok[:, b]; X = TR["enc"][m, b].astype(np.float32); y = lab[m, b, col].astype(int)
        C, mag, rel = stats(X, y, K); Cenc[b] = C; Renc[b] = (mag, rel)
        R["enc"][b] = {"n": int(m.sum()), "mag": round(mag, 4), "rel": round(rel, 4)}
    # teacher forcing: 층 0..13 (입력 블록 b) + 출력 (블록 b, tf_out[b−1])
    for b in range(16):
        m = ok[:, b]; y = lab[m, b, col].astype(int); ce = Cenc[b]
        row = {}
        for l in range(14):
            X = TR["tfh"][m, l, b].astype(np.float32); C, mag, rel = stats(X, y, K)
            ci, ci0 = ccos(C, ce @ W_embed.T); co, co0 = ccos(C @ W_proj.T, ce)
            row[f"L{l}"] = {"mag": round(mag, 4), "rel": round(rel, 4), "cos_in": round(ci, 3), "cos_in_shuf": round(ci0, 3), "cos_out": round(co, 3), "cos_out_shuf": round(co0, 3),
                            "ratio_rel": round(rel / Renc[b][1], 3), "ratio_in": round(mag / float(np.linalg.norm(ce @ W_embed.T, axis=1).mean()), 3),
                            "ratio_out": round(float(np.linalg.norm(C @ W_proj.T, axis=1).mean()) / Renc[b][0], 3)}
        if b >= 1:
            X = TR["tfout"][m, b].astype(np.float32); C, mag, rel = stats(X, y, K); c_, c0 = ccos(C, ce)
            row["out"] = {"mag": round(mag, 4), "rel": round(rel, 4), "cos": round(c_, 3), "cos_shuf": round(c0, 3), "ratio_rel": round(rel / Renc[b][1], 3), "ratio_raw": round(mag / Renc[b][0], 3)}
        R["tf"][b] = row
    # 굴린 예측: 걸음 k = 1..8 → 블록 7+k (roh slot k−1 = 위치 7+k, 걸음 1..7 / roout slot k−1)
    for k in range(1, 9):
        b = 7 + k; m = ok[:, b]; y = lab[m, b, col].astype(int); ce = Cenc[b]
        row = {}
        if k <= 7:
            for l in range(14):
                X = TR["roh"][m, l, k - 1].astype(np.float32); C, mag, rel = stats(X, y, K)
                ci, ci0 = ccos(C, ce @ W_embed.T); co, co0 = ccos(C @ W_proj.T, ce)
                row[f"L{l}"] = {"mag": round(mag, 4), "rel": round(rel, 4), "cos_in": round(ci, 3), "cos_in_shuf": round(ci0, 3), "cos_out": round(co, 3), "cos_out_shuf": round(co0, 3),
                            "ratio_rel": round(rel / Renc[b][1], 3), "ratio_in": round(mag / float(np.linalg.norm(ce @ W_embed.T, axis=1).mean()), 3),
                            "ratio_out": round(float(np.linalg.norm(C @ W_proj.T, axis=1).mean()) / Renc[b][0], 3)}
        X = TR["roout"][m, k - 1].astype(np.float32); C, mag, rel = stats(X, y, K); c_, c0 = ccos(C, ce)
        row["out"] = {"mag": round(mag, 4), "rel": round(rel, 4), "cos": round(c_, 3), "cos_shuf": round(c0, 3), "ratio_rel": round(rel / Renc[b][1], 3), "ratio_raw": round(mag / Renc[b][0], 3)}
        R["ro"][k] = row
    out[fam] = R
os.makedirs(os.path.dirname(OUTJ), exist_ok=True); os.makedirs(FIG, exist_ok=True)
json.dump(out, open(OUTJ, "w"), indent=1)

# ---- 그림: 층 × 시점 히트맵 (상대 크기 · 입력 쪽 cos · 출력 쪽 cos), teacher forcing 블록 0..15 | 굴린 예측 걸음 1..7
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"
for fam in FAM:
    R = out[fam]
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), gridspec_kw={"width_ratios": [16, 7]})
    for r_, (key, title, vmin, vmax, cmap) in enumerate((("ratio_rel", "magnitude ratio pred/enc: (|c|/RMS) / (|c_enc|/RMS_enc)", 0.6, 1.4, "coolwarm"),
                                                          ("cos_in", "cos(c_layer, W_embed c_enc)", -0.2, 1.0, "magma"),
                                                          ("cos_out", "cos(W_proj c_layer, c_enc)", -0.2, 1.0, "magma"))):
        tf = np.array([[R["tf"][b][f"L{l}"][key] for b in range(16)] for l in range(14)])
        ro = np.array([[R["ro"][k][f"L{l}"][key] for k in range(1, 8)] for l in range(14)])
        vmn = vmin if vmin is not None else min(tf.min(), ro.min()); vmx = vmax if vmax is not None else max(tf.max(), ro.max())
        for c_, (A, xl, xt) in enumerate(((tf, "teacher forcing: input block b", range(16)), (ro, "rollout step", range(1, 8)))):
            ax = axes[r_, c_]; im = ax.imshow(A, aspect="auto", origin="lower", vmin=vmn, vmax=vmx, cmap=cmap)
            ax.set_xticks(range(len(list(xt)))); ax.set_xticklabels(list(xt), fontsize=7); ax.set_yticks(range(14))
            ax.set_yticklabels(["emb"] + [str(i) for i in range(1, 13)] + ["norm"], fontsize=7)
            ax.set_xlabel(xl, fontsize=8); ax.set_ylabel("layer", fontsize=8); ax.set_title(f"{fam}: {title}", fontsize=9)
            if c_ == 0:
                ax.axvline(7.5, color="w", lw=1, ls="--")
            plt.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout(); fig.savefig(f"{FIG}/identity_concept_layers_{fam}.png", dpi=140); plt.close(fig)

for fam in FAM:
    R = out[fam]
    print(f"== {fam}  (rel magnitude | cos_in | cos_out)")
    for l in (0, 1, 3, 6, 9, 12, 13):
        tf = " ".join(f"{R['tf'][b][f'L{l}']['rel']:.3f}/{R['tf'][b][f'L{l}']['cos_in']:.2f}/{R['tf'][b][f'L{l}']['cos_out']:.2f}" for b in (1, 7, 8, 15))
        ro = " ".join(f"{R['ro'][k][f'L{l}']['rel']:.3f}/{R['ro'][k][f'L{l}']['cos_in']:.2f}/{R['ro'][k][f'L{l}']['cos_out']:.2f}" for k in (1, 4, 7))
        print(f"  L{l:2d} TF b1 b7 b8 b15: {tf}   RO s1 s4 s7: {ro}")
    print("  out TF b1 b7 b8 b15:", " ".join(f"{R['tf'][b]['out']['rel']:.3f}/{R['tf'][b]['out']['cos']:.2f}" for b in (1, 7, 8, 15)),
          "  RO s1 s4 s8:", " ".join(f"{R['ro'][k]['out']['rel']:.3f}/{R['ro'][k]['out']['cos']:.2f}" for k in (1, 4, 8)))
    print("  enc rel b1 b8 b15:", " ".join(f"{R['enc'][b]['rel']:.3f}" for b in (1, 8, 15)), " shuffled cos example:", R["tf"][8]["L6"]["cos_in_shuf"], R["ro"][4]["out"]["cos_shuf"])
print("saved", OUTJ)
