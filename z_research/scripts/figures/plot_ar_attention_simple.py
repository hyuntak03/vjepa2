#!/usr/bin/env python3
"""AR attention 예시 그림 — 쉬운 판 (2026-10-13, 사용자 "배치랑 글자가 이해가 안 간다").

visible 클립 하나와 late 클립 하나 (같은 배경 · 같은 등속 운동) 를 위아래로 놓고, 미래 첫 블록 (k=1) 하나만 본다.
  각 클립 2 줄: 윗줄 = 모델이 받는 입력 프레임 (블록 4–7) + 정답 미래 (모델은 못 봄)
               아랫줄 = '물체가 미래에 있을 자리' 토큰 (3×3) 이 각 입력 블록의 어디를 보나 (12 층 × 12 head 평균) + 블록별 합 막대
모델 = vast AR-state ep70. 라벨 영어.

  python z_research/scripts/figures/plot_ar_attention_simple.py
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
import ar_attention_history as AH  # noqa: E402
import design_headroom as Hm  # noqa: E402

FIG = REPO / "z_research/VOBench_v1/analysis/attention_map/vast_ar_state_ep70/figures/_superseded"   # rows 그림이 대체
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42})
CKPT = "z_training/vast/ar_state/latest.pt"
CLIPS = [("prim_v11_flat_vel_visible_perm_capsule_pos_a", "A. VISIBLE clip: object seen in every input block"),
         ("prim_v11_flat_vel_late_k1_perm_cube_pos_a", "B. LATE clip: object hidden behind the occluder in the last input block")]
C, S, G = 8, 256, 16
SHOW = (4, 5, 6, 7)


def compute():
    import torch
    import torch.nn.functional as F
    import yaml
    from importlib.util import module_from_spec, spec_from_file_location
    from analysis.intphys2.model import build_from_config
    from app.vjepa_frozen.ar import block_indices, encode_blocks
    from evals.world_model_analysis.data import MEAN, STD, WMADataset
    dev = torch.device("cuda", 0)
    spec = spec_from_file_location("ir", REPO / "z_research/scripts/analysis/rollout2_identity_readout.py")
    ir = module_from_spec(spec); spec.loader.exec_module(ir)
    model = dict(ir.rp.MODEL, predictor_checkpoint=str(REPO / CKPT))
    bundle = build_from_config(model, dev); pred = bundle.predictor
    cfg = yaml.safe_load(open(Hm.SETS["vobench"]["cfg"]))
    ds = WMADataset(dict(data=dict(cfg["data"], index_csv="index.csv"), model=model, features={"cache_dir": "/tmp"}, surprise={}))
    pos = {r.video_id: i for i, r in enumerate(ds.records)}
    ix = {r["video_id"]: r for r in csv.DictReader(open(Hm.SETS["vobench"]["index"]))}
    _sdpa = F.scaled_dot_product_attention; REC = {"on": False, "A": []}

    def sdpa(q, k, v, *a, **kw):
        if REC["on"] and q.shape[2] == S and k.shape[2] == C * S:                     # prefill 의 마지막 블록 query = 미래 블록 C 예측
            REC["A"].append(torch.softmax(q.float() @ k.float().transpose(-1, -2) * q.shape[-1] ** -0.5, -1)[0].mean(0).cpu().numpy())   # (S, C*S) head 평균
        return _sdpa(q, k, v, *a, **kw)
    F.scaled_dot_product_attention = sdpa
    out = []
    for vid, _ in CLIPS:
        x = ds.clip(pos[vid]); P_, hs = Hm.track_vobench(ix[vid]); yx, hid = AH.obj_tokens(P_, hs)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hb = encode_blocks(bundle.target_encoder, x[None].to(dev, dtype=bundle.dtype), 2)
            REC.update(on=True, A=[]); pred.rollout_kv(hb[:, :C * S], block_indices(1, C, S, dev), 1); REC["on"] = False
        A = np.stack(REC["A"]).mean(0)                                                 # (S, C*S) 층 평균
        q = AH.nbhd(yx[C])                                                             # 미래 블록 C 의 참 물체 자리 3×3
        att = A[q].mean(0).reshape(C, G, G)                                            # 그 query 의 attention, 입력 블록별 지도
        fr = (x * STD + MEAN).clamp(0, 1).permute(1, 2, 3, 0).numpy()
        out.append(dict(vid=vid, att=att, yx=yx, hid=hid, hid_s=hs > 0.5, frames=fr))
    return out


def main():
    D = compute()
    ncol = len(SHOW) + 2
    fig = plt.figure(figsize=(7.2, 6.6))
    gs = fig.add_gridspec(8, ncol, height_ratios=[0.16, 0.2, 1, 1, 0.12, 0.2, 1, 1], width_ratios=[1] * (len(SHOW) + 1) + [1.15],
                          hspace=0.1, wspace=0.08, left=0.11, right=0.97, top=0.88, bottom=0.03)
    for j, b in enumerate(list(SHOW) + [C]):                                          # 열 제목 (맨 위 한 번)
        ax = fig.add_subplot(gs[0, j]); ax.set_axis_off()
        ax.text(0.5, 0.0, f"input block {b}\n(frames {2 * b}-{2 * b + 1})" + ("\nLAST input" if b == C - 1 else "") if b < C else "future block 8\n(the answer;\nNOT given to model)",
                ha="center", va="bottom", fontsize=6.3, color="k" if b < C else "0.4", transform=ax.transAxes)
    for ci, (d, (_, title)) in enumerate(zip(D, CLIPS)):
        r0 = 1 + 4 * ci; att, yx, hid, hs, fr = d["att"], d["yx"], d["hid"], d["hid_s"], d["frames"]
        ax = fig.add_subplot(gs[r0, :]); ax.set_axis_off(); ax.text(0.0, 0.2, title, fontsize=7.5, weight="bold", transform=ax.transAxes)
        for j, b in enumerate(list(SHOW) + [C]):
            f_ = 2 * b + 1; y, x_ = yx[b]
            # 윗줄: 입력 프레임
            ax = fig.add_subplot(gs[r0 + 1, j]); ax.imshow(fr[f_], extent=(0, G, G, 0)); ax.set_xticks([]); ax.set_yticks([])
            if not hs[f_]: ax.add_patch(plt.Circle((x_ + 0.5, y + 0.5), 1.4, fill=False, ec="cyan", lw=1.2))
            else: ax.text(x_ + 0.5, y - 1.2, "object hidden\nbehind occluder", color="w", fontsize=5.3, ha="center", va="bottom")
            if b == C:
                ax.add_patch(plt.Rectangle((yx[C][1] - 1, yx[C][0] - 1), 3, 3, fill=False, ec="lime", lw=1.0, ls="--"))
                for sp in ax.spines.values(): sp.set_edgecolor("0.5"); sp.set_linestyle(":")
            if j == 0: ax.set_ylabel("input\nframes", fontsize=6.5, rotation=0, ha="right", va="center")
            if b == C: continue
            # 아랫줄: attention (패널마다 자기 최댓값으로 색 — 크기는 숫자로)
            ax = fig.add_subplot(gs[r0 + 2, j]); ax.imshow(0.45 * fr[f_] + 0.05, extent=(0, G, G, 0))
            v = np.clip(att[b] / att[b].max(), 0, 1); rgba = plt.get_cmap("autumn_r")(v); rgba[..., 3] = 0.9 * v ** 1.5
            ax.imshow(rgba, extent=(0, G, G, 0), interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
            if not hs[f_]: ax.add_patch(plt.Circle((x_ + 0.5, y + 0.5), 1.4, fill=False, ec="cyan", lw=1.2))
            ax.add_patch(plt.Rectangle((yx[C][1] - 1, yx[C][0] - 1), 3, 3, fill=False, ec="lime", lw=1.0, ls="--"))
            on_obj = att[b][AH.nbhd(yx[b]).reshape(G, G)].sum()
            ax.text(0.2, 15.6, f"this block: {att[b].sum():.0%}" + (f"\non object: {on_obj:.1%}" if not hs[f_] else ""), color="w", fontsize=5.5, va="bottom")
            if j == 0: ax.set_ylabel("attention of\nthe green box\n(per-panel\ncolor scale)", fontsize=6.2, rotation=0, ha="right", va="center")
        ax = fig.add_subplot(gs[r0 + 2, len(SHOW) + 1])
        ax.bar(range(C), att.sum((1, 2)) * 100, color=["0.7"] * (C - 1) + ["0.25"])
        ax.set_xticks(range(C)); ax.set_ylim(0, 80); ax.tick_params(labelsize=5.5); ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("input block", fontsize=6); ax.set_ylabel("% of attention", fontsize=6)
        if ci == 0: ax.set_title("share per input block", fontsize=6.3)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Line2D([], [], color="cyan", marker="o", mfc="none", ls="", ms=8, label="where the object IS in that frame"),
         Line2D([], [], color="lime", marker="s", mfc="none", ls="", ms=8, label="green box = where the object WILL BE in future block 8 (we look at these tokens' attention)"),
         Patch(color="#ffb000", label="bright = where the green box looks, predicting future block 8 (AR-state vast ep70; mean of 12 layers x 12 heads)")]
    fig.legend(handles=h, loc="upper left", ncol=1, frameon=False, fontsize=6.3, bbox_to_anchor=(0.09, 0.995))
    FIG.mkdir(parents=True, exist_ok=True); out = FIG / "fig_ar_attention_simple"
    fig.savefig(f"{out}.png", dpi=230); fig.savefig(f"{out}.pdf"); print("saved", out)
    print({d["vid"]: dict(block_mass=np.round(d["att"].sum((1, 2)), 3).tolist(),
                          on_obj=[round(float(d["att"][b][AH.nbhd(d["yx"][b]).reshape(G, G)].sum()), 4) for b in range(C)], hid=d["hid"][:9].astype(int).tolist()) for d in D})


if __name__ == "__main__":
    main()
