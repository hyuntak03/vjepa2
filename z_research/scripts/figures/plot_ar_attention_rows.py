#!/usr/bin/env python3
"""AR attention — 걸음마다 한 행 (2026-10-13, 사용자 배치 지시).

행 r = 미래 블록 t(9+r) 를 만들 때. 열 = 블록 t1..t16 (블록 = 2 프레임).
  t1–t8  : 입력 문맥 프레임 (블록의 둘째 프레임) — 하늘색 원 = 그 블록의 물체 자리
  t9..   : 앞 걸음에서 모델이 만든 블록 (픽셀이 없어 회색) — 그 행에서 입력으로 쓰인다
  만드는 칸 : 회색 + '?' + 초록 점선 = 물체가 실제로 있을 자리 (= 보는 query 토큰 3×3)
색 = 그 query 의 attention (12 층 × 12 head 평균), **행마다 한 색 범위** (행 안 칸끼리 비교 가능). 칸 위 숫자 = 그 블록에 간 %.
오른쪽 = 같은 값을 프레임 단위 행렬로 (행 = 만드는 블록 t9..t16, 열 = 보는 블록 t1..t15).
모델 vast AR-state ep70 (기본). 클립 visible · late 하나씩 → 그림 두 장. 라벨 영어.
--model vjepa2 (2026-10-14): 릴리즈 predictor. 미래 8 블록을 mask token 으로 **한 번에** 만들고 full attention 이라
  행 r = 블록 t(9+r) 의 mask token query 이고, 열은 16 블록 전부 (자기 블록 · 뒤 미래 mask 블록 포함) 에 값이 있다.
  입력 = 문맥 encoder (online) 프레임 0–15 (표준 surprise 와 같은 호출).

  python z_research/scripts/figures/plot_ar_attention_rows.py                      # → attention_map/vast_ar_state_ep70/figures/
  python z_research/scripts/figures/plot_ar_attention_rows.py --model vjepa2       # → attention_map/vjepa2/figures/
  python z_research/scripts/figures/plot_ar_attention_rows.py --model vjepa2 --layers   # + 층마다 한 장 · GIF · PDF → figures/layers/ (2026-10-14)
"""
import argparse
import csv
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

AMD = REPO / "z_research/VOBench_v1/analysis/attention_map"
CKPTS = {"vast_ar_state_ep70": "z_training/vast/ar_state/latest.pt", "vjepa2": None,
         "ariel_ar_ep40": "z_training/ariel/ar_future_1e_5/latest.pt", "ariel_ar_state_ep40": "z_training/ariel/ar_state_future_1e_5/latest.pt",
         "vobench_release_postft_ep30": "z_training/runs/vobench_release_postft/latest.pt",
         "vobench_vast_ar_state_postft_ep30": "z_training/runs/vobench_vast_ar_state_postft/latest.pt"}   # vast AR-state ep70 을 VOBench split_v2 train 으로 post-FT   # 릴리즈 predictor 를 VOBench split_v2 train 으로 post-FT (one-shot)
LABEL = {"vast_ar_state_ep70": "AR-state vast ep70", "vjepa2": "V-JEPA 2 release predictor", "ariel_ar_ep40": "Ariel AR ep40", "ariel_ar_state_ep40": "Ariel AR-state ep40",
         "vobench_release_postft_ep30": "release post-FT on VOBench (ep30)", "vobench_vast_ar_state_postft_ep30": "AR-state vast post-FT on VOBench (ep30)"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42})
CLIPS = [("prim_v11_flat_vel_visible_perm_capsule_pos_a", "visible", "VISIBLE clip (object seen in all input blocks t1-t8)"),
         ("prim_v11_flat_vel_late_k1_perm_cube_pos_a", "late", "LATE clip (object hidden behind the occluder at t8, the last input block)")]
C, K, S, G, NB = 8, 8, 256, 16, 16


def compute(mname, vids=None):
    """vids 를 주면 CLIPS 대신 그 clip 들 (plot_attention_layers.py --spotlight 가 test split clip 을 넘긴다)."""
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
    model = dict(ir.rp.MODEL) if CKPTS[mname] is None else dict(ir.rp.MODEL, predictor_checkpoint=str(REPO / CKPTS[mname]))
    bundle = build_from_config(model, dev); pred = bundle.predictor
    release = getattr(pred, "mask_mode", None) not in ("ar", "ar_state")              # 릴리즈 · 릴리즈 post-FT = one-shot
    cfg = yaml.safe_load(open(Hm.SETS["vobench"]["cfg"]))
    ds = WMADataset(dict(data=dict(cfg["data"], index_csv="index.csv"), model=model, features={"cache_dir": "/tmp"}, surprise={}))
    pos = {r.video_id: i for i, r in enumerate(ds.records)}
    ix = {r["video_id"]: r for r in csv.DictReader(open(Hm.SETS["vobench"]["index"]))}
    _sdpa = F.scaled_dot_product_attention; REC = {"on": False, "A": []}

    def sdpa(q, k, v, *a, **kw):
        if REC["on"] and release and q.shape[2] == NB * S and k.shape[2] == NB * S:   # 릴리즈: 층마다 하나, 미래 query 만
            REC["A"].append(torch.softmax(q[:, :, C * S:].float() @ k.float().transpose(-1, -2) * q.shape[-1] ** -0.5, -1)[0].mean(0).cpu().numpy())
        elif REC["on"] and not release and q.shape[2] == S and k.shape[2] >= C * S:  # AR: 층마다 걸음마다 하나 (rollout_kv)
            REC["A"].append(torch.softmax(q.float() @ k.float().transpose(-1, -2) * q.shape[-1] ** -0.5, -1)[0].mean(0).cpu().numpy())
        return _sdpa(q, k, v, *a, **kw)
    F.scaled_dot_product_attention = sdpa
    out = []
    for vid in (vids or [c[0] for c in CLIPS]):
        x = ds.clip(pos[vid]); P_, hs = Hm.track_vobench(ix[vid]); yx, hid = AH.obj_tokens(P_, hs)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            xb = x[None].to(dev, dtype=bundle.dtype)
            if release:
                from analysis.intphys2.surprise import _context_target_indices
                ci, ti = _context_target_indices(ctx_frames=2 * C, tgt_frames=2 * K, tubelet_size=2, spatial_tokens=S, batch_size=1, device=dev)
                z = bundle.context_encoder(xb, masks=[ci]); z = z[-1] if isinstance(z, list) else z
                REC.update(on=True, A=[]); pred(z, ci, ti, mask_index=0); REC["on"] = False
            else:
                hb = encode_blocks(bundle.target_encoder, xb, 2)
                REC.update(on=True, A=[]); pred.rollout_kv(hb[:, :C * S], block_indices(1, C, S, dev), K); REC["on"] = False
        assert len(REC["A"]) == (12 if release else K * 12)
        maps = np.zeros((K, NB, G, G))                                                  # [걸음 r, 보는 블록 b] 물체 query 의 attention 지도
        maps_l = np.zeros((12, K, NB, G, G))                                            # 층별 (plot_attention_layers.py 용)
        Arel = np.mean(REC["A"], 0) if release else None                                # 릴리즈: (K*S, NB*S) 층 평균
        for r in range(K):
            q = AH.nbhd(yx[C + r])
            if release:
                maps[r] = Arel[r * S:(r + 1) * S][q].mean(0).reshape(NB, G, G)
                for l in range(12): maps_l[l, r] = REC["A"][l][r * S:(r + 1) * S][q].mean(0).reshape(NB, G, G)
            else:
                for l in range(12): maps_l[l, r, :C + r] = REC["A"][r * 12 + l][q].mean(0).reshape(C + r, G, G)
                A = np.mean([REC["A"][r * 12 + l] for l in range(12)], 0)             # (S, (C+r)*S) 층 평균
                maps[r, :C + r] = A[q].mean(0).reshape(C + r, G, G)
        fr = (x * STD + MEAN).clamp(0, 1).permute(1, 2, 3, 0).numpy()
        out.append(dict(maps=maps, maps_l=maps_l, yx=yx, hid_s=hs > 0.5, frames=fr, release=release))
    return out


def draw(d, tag, title, mname, layer=None):
    """layer=None → 12 층 평균 (정본). layer=l → 그 층 하나 (head 평균), figures/layers/ 에 저장하고 fig 를 돌려준다."""
    maps, yx, hs, fr, rel = (d["maps"] if layer is None else d["maps_l"][layer]), d["yx"], d["hid_s"], d["frames"], d["release"]
    fig = plt.figure(figsize=(14.0, 7.6))
    gs = fig.add_gridspec(K, NB + 2, width_ratios=[1] * NB + [1.3, 7.6], wspace=0.04, hspace=0.06, left=0.06, right=0.99, top=0.86, bottom=0.07)
    for r in range(K):
        tgt = C + r; vmax = maps[r].max() if rel else maps[r, :tgt].max()
        for b in range(NB):
            ax = fig.add_subplot(gs[r, b]); ax.set_xticks([]); ax.set_yticks([])
            if r == 0: ax.set_title(f"t{b + 1}", fontsize=7, color="k" if b < C else "0.45")
            if b == 0: ax.set_ylabel(f"t{tgt + 1} mask query" if rel else f"making t{tgt + 1}", fontsize=7, rotation=0, ha="right", va="center")
            if b > tgt and not rel:                                                   # 아직 없는 블록 (AR)
                ax.set_axis_off(); continue
            if b == tgt:                                                              # 지금 만드는 블록
                ax.imshow(np.full((G, G, 3), 0.8), extent=(0, G, G, 0))
                ax.text(G / 2, G / 2 - 3, "?", ha="center", va="center", fontsize=11, color="0.35")
                ax.add_patch(plt.Rectangle((yx[b][1] - 1, yx[b][0] - 1), 3, 3, fill=False, ec="lime", lw=1.3, ls="--"))
                for sp in ax.spines.values(): sp.set_edgecolor("lime"); sp.set_linewidth(1.5)
                if not rel: continue                                                  # 릴리즈는 자기 블록 (mask) 도 본다 → 아래에서 덧칠
            if b < C:
                f_ = 2 * b + 1; ax.imshow(0.6 * fr[f_] + 0.1, extent=(0, G, G, 0))
                if not hs[f_]: ax.add_patch(plt.Circle((yx[b][1] + 0.5, yx[b][0] + 0.5), 1.5, fill=False, ec="cyan", lw=0.9))
            elif b != tgt:
                ax.imshow(np.full((G, G, 3), 0.62), extent=(0, G, G, 0))              # AR: 모델이 만든 블록 / 릴리즈: 미래 mask 블록 (픽셀 없음)
            v = np.clip(maps[r, b] / vmax, 0, 1); rgba = plt.get_cmap("autumn_r")(v); rgba[..., 3] = 0.95 * v
            ax.imshow(rgba, extent=(0, G, G, 0), interpolation="nearest")
            ax.text(0.4, 15.4, f"{maps[r, b].sum():.0%}", color="w", fontsize=5.6, va="bottom", weight="bold")
    # 행렬
    ax = fig.add_subplot(gs[:, NB + 1])
    M = np.full((K, NB), np.nan)
    for r in range(K): M[r, :NB if rel else C + r] = maps[r, :NB if rel else C + r].sum((1, 2)) * 100
    im = ax.imshow(M, cmap="Oranges", vmin=0, vmax=70, aspect="auto")
    for r in range(K):
        for b in range(NB if rel else C + r):
            if b == C + r and rel: continue
            ax.text(b, r, f"{M[r, b]:.0f}", ha="center", va="center", fontsize=6, color="w" if M[r, b] > 40 else "k")
        ax.add_patch(plt.Rectangle((C + r - 0.5, r - 0.5), 1, 1, fc="0.8", ec="lime", lw=1.0))
        ax.text(C + r, r, f"?{M[r, C + r]:.0f}" if rel else "?", ha="center", va="center", fontsize=6 if rel else 7, color="0.35")
    ax.axvline(C - 0.5, color="k", lw=1)
    ax.set_xticks(range(NB)); ax.set_xticklabels([f"t{b + 1}" for b in range(NB)], fontsize=6)
    ax.set_yticks(range(K)); ax.set_yticklabels([f"t{C + r + 1} mask query" if rel else f"making t{C + r + 1}" for r in range(K)], fontsize=6.5)
    ax.set_xlabel("attended block   (t1-t8 = input frames  |  t9-t16 = " + ("future mask-token blocks)" if rel else "blocks the model made itself)"))
    ax.set_title("same numbers as a matrix: % of attention per block", fontsize=7.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015, shrink=0.6); cb.set_label("% attention", fontsize=6)
    how = ("One-shot, bidirectional: all 8 future blocks are predicted in ONE forward pass from mask tokens. Rows are NOT time steps - each row is the mask-token query of one future block in the same pass,\n"
            "and it can attend to every block (context, its own block, earlier AND later future blocks). At layer 0 all mask tokens are identical (they differ only by position)."
           if rel else "Each row is one autoregressive step: the model makes the gray '?' block from the blocks to its left.")
    gray = "Gray = future mask-token block (no pixels)." if rel else "Gray = block the model generated (no pixels)."
    fig.suptitle(f"{title}\n{how}\nGreen dashed box = where the object really is in that block; "
                 f"we show where THESE tokens attend ({LABEL[mname]}, " + ("mean of 12 layers x 12 heads" if layer is None else f"LAYER L{layer} only, mean of 12 heads") + ").\n"
                 f"Cyan circle = object in that input frame. {gray} "
                 "Color scale shared within a row; number = % of attention on that whole block.", fontsize=7.5, x=0.06, ha="left", y=0.995)
    if layer is not None:                                                             # GIF 로 넘길 때 층이 바로 보이게
        fig.text(0.006, 0.955, f"L{layer}", fontsize=22, weight="bold", color="#2a78d6", ha="left", va="top")
        FIG = AMD / mname / "figures" / "layers"; FIG.mkdir(parents=True, exist_ok=True)
        out = FIG / f"fig_ar_attention_rows_{tag}_L{layer:02d}"; fig.savefig(f"{out}.png", dpi=130); return fig
    FIG = AMD / mname / "figures"; out = FIG / f"fig_ar_attention_rows_{tag}"
    FIG.mkdir(parents=True, exist_ok=True); fig.savefig(f"{out}.png", dpi=200); fig.savefig(f"{out}.pdf"); print("saved", out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="vast_ar_state_ep70", choices=list(CKPTS))
    ap.add_argument("--layers", action="store_true", help="층마다 한 장 (L0–L11) + 넘기는 GIF + 여러 쪽 PDF → figures/layers/")
    a = ap.parse_args()
    for d, (vid, tag, title) in zip(compute(a.model), CLIPS):
        if a.model.startswith("vobench_"):                                              # VOBench 로 학습한 판: 예시 clip 이 학습 split 에 있었나
            tr = {r["video_id"] for r in csv.DictReader(open(AH.SPLIT / "index_train.csv"))}
            title += "  [this clip is in the post-FT " + ("TRAIN" if vid in tr else "TEST") + " split]"
        draw(d, tag, title, a.model)
        if a.layers:
            from matplotlib.backends.backend_pdf import PdfPages
            from PIL import Image
            FIG = AMD / a.model / "figures" / "layers"; FIG.mkdir(parents=True, exist_ok=True)
            with PdfPages(FIG / f"fig_ar_attention_rows_{tag}_layers.pdf") as pdf:
                for l in range(12):
                    fig = draw(d, tag, title, a.model, layer=l); pdf.savefig(fig); plt.close(fig)
            ims = [Image.open(FIG / f"fig_ar_attention_rows_{tag}_L{l:02d}.png").convert("RGB") for l in range(12)]
            ims[0].save(FIG / f"fig_ar_attention_rows_{tag}_layers.gif", save_all=True, append_images=ims[1:], duration=1200, loop=0)
            print("saved", FIG / f"fig_ar_attention_rows_{tag}_layers.{{gif,pdf}}", "+ L00–L11 png")


if __name__ == "__main__":
    main()
