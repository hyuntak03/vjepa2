#!/usr/bin/env python3
"""토큰별 채점 신호를 원본 영상 위에 히트맵으로 얹어 GIF 로 만든다.

채점은  mean_token |p - h(imp)|  >  mean_token |p - h(pos)|  인지를 본다.
그 평균을 걷어내고 **토큰마다** 보면

    d(token) = |p - h(imp)|(token) - |p - h(pos)|(token)

가 그 토큰이 정답 쪽으로 민 정도다. **양수면 정답 신호, 음수면 반대로 민 것.**
정확도는 `mean_token d > 0` 이므로 이 그림이 그 합의 내역이다.

토큰 배치 (실측 확인)
  PatchEmbed3D 가 Conv3d 뒤 `flatten(2).transpose(1,2)` 라 토큰 순서는 **(T', H', W')**.
  256px / patch 16 -> 16x16 공간,  미래 16프레임 / tubelet 2 -> 8 시간칸.  8*16*16 = 2048.
  tubelet j 는 **샘플 프레임 16+2j, 16+2j+1** (= raw 48+6j, 51+6j) 을 덮는다.

  캐시:  predictor.npy (N,2048,1280) = 미래 8칸
        target.npy    (N,4096,1280) = 32프레임 16칸  ->  미래는 뒤 2048

  python z_research/scripts/figures/token_signal_gif.py \
    --violation color --motion static --timing early --k 2 \
    --out z_research/IntPhysGenV11_occlusion_timing_ablation/figures/04_token_signal
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

CACHE = Path("/local_datasets/world/world_analysis/cache/v11_full_vith")
FRAMES = Path("/local_datasets/world/world_analysis/IntPhysGen_v11")
IDX = Path("data_csv/intphysgen_v11_full/index_probe.csv")
BLUE, ORANGE, INK, INK2, MUTED = "#2a78d6", "#eb6834", "#000000", "#3b3b3b", "#9a9a9a"
# 발산 컬러맵 — 0 이 투명한 흰색. 양수(정답 쪽)=파랑, 음수(반대)=주황.
CM = LinearSegmentedColormap.from_list("sig", [ORANGE, "#ffffff", BLUE])
COND = {("static", "vis"): "static_visible", ("static", "early"): "static_occlusion_early",
        ("static", "mid"): "static_occlusion_mid", ("static", "late"): "static_occlusion",
        ("flat", "vis"): "moving_visible_flat", ("flat", "early"): "moving_occlusion_flat_early",
        ("flat", "mid"): "moving_occlusion_flat_mid", ("flat", "late"): "moving_occlusion_flat",
        ("ramp", "vis"): "moving_visible", ("ramp", "early"): "moving_occlusion_early",
        ("ramp", "mid"): "moving_occlusion_mid", ("ramp", "late"): "moving_occlusion"}
mpl.rcParams.update({"font.family": "serif",
                     "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"]})


def load_frames(file_name, idxs, size=256):
    """샘플 인덱스 -> raw 프레임 3*i.  모델과 같은 커널(bilinear, antialias 없음)로 줄인다."""
    out = []
    for i in idxs:
        p = FRAMES / file_name / f"{3 * i:06d}.png"
        im = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
        out.append(np.asarray(im, dtype=np.float32) / 255.0)
    return np.stack(out)


def token_signal(ip, ii, ipos):
    """d(token) = |p-h_imp| - |p-h_pos|,  (8,16,16).  양수 = 정답 쪽."""
    P = np.load(CACHE / "predictor.npy", mmap_mode="r")
    H = np.load(CACHE / "target.npy", mmap_mode="r")
    C = H.shape[1] // 2
    p = np.asarray(P[ip], np.float32)
    hi = np.asarray(H[ii][C:], np.float32)
    hp = np.asarray(H[ipos][C:], np.float32)
    d = np.abs(p - hi).mean(1) - np.abs(p - hp).mean(1)      # (2048,)
    return d.reshape(8, 16, 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--violation", default="color", choices=["vanish", "shape", "color"])
    ap.add_argument("--motion", default="static", choices=["static", "flat", "ramp"])
    ap.add_argument("--timing", default="early", choices=["vis", "early", "mid", "late"])
    ap.add_argument("--k", default="2")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--first", type=int, default=15, help="보여줄 첫 샘플 프레임 (1-idx)")
    ap.add_argument("--last", type=int, default=32)
    ap.add_argument("--fps", type=float, default=2.5)
    ap.add_argument("--rows", default="correct,wrong", help="쉼표로: correct / wrong")
    ap.add_argument("--thr", type=float, default=70.0,
                    help="|d| 가 이 분위 아래면 투명. 0 이면 전부 칠한다")
    a = ap.parse_args()

    meta = list(csv.DictReader(IDX.open()))
    vids = json.loads((CACHE / "meta.json").read_text())["video_ids"]
    row_of = {r["video_id"]: r for r in meta}
    ix = {v: i for i, v in enumerate(vids)}
    cond = COND[(a.motion, a.timing)]

    blk = {}
    for r in meta:
        if r["condition"] != cond or r["violation_type"] != a.violation:
            continue
        if a.timing != "vis" and r["sym_k"] != a.k:
            continue
        blk.setdefault(r["source_block"], {})[r["variant"]] = r

    S = json.loads((Path("z_research/IntPhysGenV11_occlusion_timing_ablation/exp_results")
                    / "surprise_c16t32__v11_full_vith" / "per_block.json").read_text())
    S = S["per_video_surprise"]
    cands = {"correct": [], "wrong": []}
    for b, d in sorted(blk.items()):
        for pos, imp in (("pos_a", "imp_ab"), ("pos_b", "imp_ba")):
            if pos not in d or imp not in d:
                continue
            vp, vi = d[pos]["video_id"], d[imp]["video_id"]
            if vp not in S or vi not in S:
                continue
            cands["correct" if S[vi] > S[vp] else "wrong"].append((vp, vi, d[pos], d[imp]))
    want = [w.strip() for w in a.rows.split(",")]
    picks = []
    for w in want:
        if not cands[w]:
            print(f"  ! {w} 예시가 없다 (건너뜀)"); continue
        picks.append((w, cands[w][0]))
    if not picks:
        raise SystemExit("보여줄 예시가 없다")
    print(f"  {cond} / {a.violation} / k={a.k} — correct {len(cands['correct'])} · "
          f"wrong {len(cands['wrong'])}")

    idxs = list(range(a.first - 1, a.last))          # 1-idx -> 0-idx
    fut0 = 16                                        # 미래 첫 샘플 인덱스
    n = len(picks)
    HEAD = 1.15                                   # 상단 머리말 높이(inch) — 겹침 방지
    H = 2.35 * n + HEAD
    fig, axes = plt.subplots(n, 3, figsize=(7.7, H), squeeze=False)
    top = 1 - HEAD / H
    fig.subplots_adjust(left=0.055, right=0.885, top=top, bottom=0.005, wspace=0.02,
                        hspace=0.14)
    bar = fig.add_axes([0.055, 1 - 0.46 * HEAD / H, 0.83, 0.020 * 2.9 / H])
    bar.set_xlim(0, len(idxs)); bar.set_ylim(0, 1); bar.set_yticks([])
    bar.set_xticks([]); bar.set_facecolor("#ededed")
    for sp in bar.spines.values():
        sp.set_color(MUTED); sp.set_linewidth(0.6)
    ctx_n = sum(1 for i in idxs if i < fut0)
    if ctx_n:
        bar.axvspan(0, ctx_n, color="#cfcfcf", zorder=1)
        bar.text(ctx_n / 2, 0.5, "context", ha="center", va="center", fontsize=6.2, color=INK2)
    bar.text((ctx_n + len(idxs)) / 2, 0.5, "future  (predictor's target)", ha="center",
             va="center", fontsize=6.2, color=INK2)
    cur = bar.axvline(0.5, color=INK, lw=1.6, zorder=5)
    ttl = fig.text(0.47, 1 - 0.16 * HEAD / H, "", ha="center", va="top", fontsize=9.0, color=INK)
    fig.text(0.47, 1 - 0.75 * HEAD / H,
             "blue = this patch pushes toward the correct answer    ·    "
             "orange = pushes the wrong way    ·    weak tokens are transparent",
             ha="center", va="top", fontsize=6.6, color=INK2)

    # ⚠️ 색 눈금은 **모든 행 · 모든 프레임 공통**이다. 프레임마다 다시 잡으면
    #    시간에 따른 세기 변화가 안 보이고 매 컷이 같아 보인다.
    pre = [token_signal(ix[vp], ix[vi], ix[vp]) for _, (vp, vi, _, _) in picks]
    allsig = np.concatenate([x.ravel() for x in pre])
    VMAX = float(np.percentile(np.abs(allsig), 99))
    THR = float(np.percentile(np.abs(allsig), a.thr)) if a.thr > 0 else 0.0

    ims, data = [], []
    for ri, ((tag, (vp, vi, rp, rimp)), sig) in enumerate(zip(picks, pre)):
        fp = load_frames(rp["file_name"], idxs)
        fi = load_frames(rimp["file_name"], idxs)
        v = VMAX
        lab = [f"possible\n{rp['color_pre'] if a.violation=='color' else rp['shape_pre']}",
               f"impossible\n-> {rimp['color_post'] if a.violation=='color' else rimp['shape_post']}",
               "impossible + token signal"]
        row = []
        for ci in range(3):
            ax = axes[ri][ci]; ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(MUTED); sp.set_linewidth(0.5)
            im = ax.imshow(fp[0] if ci == 0 else fi[0])
            # ⚠️ 균일 alpha 로 덮으면 0 근처 토큰(대부분)이 흰색으로 화면을 다 지운다.
            #    **|d| 에 비례한 alpha** 를 줘서 신호가 있는 자리에만 색이 얹히게 한다.
            # ⚠️ interpolation 은 **nearest** 다. bilinear 로 두면 16x16 토큰이 뭉개져
            #    Grad-CAM 같은 부드러운 덩어리로 보인다 — 실제 해상도를 속이는 그림이 된다.
            #    패치 하나 = 16x16 픽셀 블록 그대로 칠한다.
            ov = ax.imshow(np.zeros((16, 16, 4)), extent=(0, 256, 256, 0),
                           interpolation="nearest") if ci == 2 else None
            if ri == 0:
                ax.set_title(lab[ci], fontsize=7.8, color=INK, pad=4)
            if ci == 0:
                ax.set_ylabel(f"{tag}", fontsize=9.0, labelpad=3,
                              color=BLUE if tag == "correct" else ORANGE)
            row.append((im, ov))
        ims.append(row); data.append((fp, fi, sig, v))

    cax = fig.add_axes([0.905, 0.10, 0.016, 0.55])
    cax.imshow(np.linspace(1, 0, 256)[:, None], cmap=CM, aspect="auto")
    cax.set_xticks([]); cax.yaxis.tick_right(); cax.set_yticks([0, 127.5, 255])
    cax.set_yticklabels([f"+{VMAX:.3f}\ntoward correct", f"|d| < {THR:.4f}\n(transparent)",
                         f"−{VMAX:.3f}\ntoward wrong"], fontsize=5.8, color=INK2)
    cax.tick_params(length=0, pad=2)
    for sp in cax.spines.values():
        sp.set_color(MUTED); sp.set_linewidth(0.5)
    cax.set_ylabel("d = |p−h(imp)| − |p−h(pos)|   per patch token",
                   fontsize=6.2, color=INK2, labelpad=3)
    cax.yaxis.set_label_position("right")

    def draw(fi_):
        cur.set_xdata([fi_ + 0.5, fi_ + 0.5])
        si = idxs[fi_]
        ttl.set_text(f"{cond}  ·  {a.violation}  ·  k={a.k}   |   sampled frame {si+1}/32"
                     f"  (raw {3*si})")
        for (fp, fimp, sig, v), row in zip(data, ims):
            row[0][0].set_data(fp[fi_]); row[1][0].set_data(fimp[fi_])
            row[2][0].set_data(fimp[fi_])
            if si >= fut0:
                d = sig[(si - fut0) // 2]
                rgba = CM(np.clip(d / (2 * v) + 0.5, 0, 1))     # -v..v -> 0..1
                # 확실한 것만 진하게 — |d| 가 문턱 아래면 투명(영상이 그대로 보인다)
                rgba[..., 3] = np.clip((np.abs(d) - THR) / max(v - THR, 1e-12), 0, 1) * 0.92
                row[2][1].set_data(rgba)
            else:
                row[2][1].set_data(np.zeros((16, 16, 4)))
        return []

    a.out.mkdir(parents=True, exist_ok=True)
    name = f"{a.violation}_{a.motion}_{a.timing}_k{a.k}.gif"
    from matplotlib.animation import FuncAnimation
    an = FuncAnimation(fig, draw, frames=len(idxs), blit=False)
    an.save(a.out / name, writer=PillowWriter(fps=a.fps))
    plt.close(fig)
    print(f"  [saved] {a.out / name}   ({len(idxs)} 프레임, {a.fps} fps)")


if __name__ == "__main__":
    main()
