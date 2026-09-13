#!/usr/bin/env python3
"""v11 — 이동 (flat / ramp) × 가림 k=0..4, vanish 블록에서 predictor(p) 또는 context encoder(z) 가 놓는 물체 위치를 읽어 그린다.

자: RollOut_v2_training (ROLLOUT2_TRAIN, 기본 v4) 으로 정한 attentive 위치 readout. v11 은 fitting 에 안 썼다.
  --rep p   predictor 출력 (v11_full_vith/predictor.npy, 미래 8 튜블릿). imp_ab 와 pos_a 는 문맥이 같아 p 가 byte 단위로 같다 →
            읽힌 위치 = "물체가 계속 있을 때 predictor 가 놓는 자리".
  --rep z   context encoder 에 32 frames 를 다 넣은 것 (v11_vanish_all_ctx32_vith/isolated_ctx_0_32.npy, 16 튜블릿 전부). pos_a 클립 = 물체를 실제로 봄.
            자의 상한/대조. (캐시는 datasets.md `v11_vanish` 로 뽑는다.)
k=0 은 visible (`moving_visible{,_flat}`), k=1..4 는 late 가림 (`moving_occlusion{,_flat}`). 진실은 pos_a 의 metadata 픽셀 위치.

  figures/<train>/v11_vanish/<rep>/<motion>/k<k>/
      gif_<pos_a clip>.gif      가능 영상 32 샘플 (문맥 16 + 미래 16) 를 느리게 재생. 흰 = 진실 궤적 (읽기와 같은 tubelet 단위 = 두 샘플 평균), 주황(p) / 초록(z) = 읽기 궤적 (p 는 미래부터),
                                지나온 위치는 작은 점 + 선으로 남긴다. 아래 time bar (파랑 = 문맥, 밝은 회색 = 미래, 진회색 = 가려진 샘플, ▲ = 현재)
      overlay_<clip>.png        미래 8 프레임 (raw 48…90) 위에 읽기 / 진실 / ×(마지막 관측)   (p 는 imp_ab 프레임, z 는 pos_a 프레임)
      fig_gain.png, readout.npz
  figures/<train>/v11_vanish/<rep>/<motion>/fig_gain_all_k.png     k=0..4 의 x(t)·y(t) 변위를 한 패널에

  python z_research/scripts/figures/plot_v11_vanish_readout.py --rep p --motion flat|ramp|static --timing late|early|mid [--k 0 1 2 3 4] [--n 3]
  (출력 폴더: <motion> = late, <motion>_early, <motion>_mid. z 캐시는 v11_vanish_all_ctx32_vith = 전 조건 pos_a 2,688 clip)
"""
from __future__ import annotations
import argparse, csv, json, re, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))
from rollout2_attn_readout import AttnReadout                      # noqa: E402
import rollout2_test_readout as rt                                 # noqa: E402  (ROLLOUT2_TRAIN=v2|v4 → 자·그림 폴더)

FRAMES = Path("/local_datasets/world/world_analysis/IntPhysGen_v11"); META = FRAMES / "metadata.csv"
INDEX = ROOT / "data_csv/intphysgen_v11_full/index_probe.csv"
CACHE_ROOT = Path("/local_datasets/world/world_analysis/cache")
REP = {"p": dict(cache=CACHE_ROOT / "v11_full_vith", file="predictor.npy", attn=rt.RES_ROOT / "attentive_pooling/p/attn.pt", n_tub=8, clip="imp_ab",
                 label="predictor p"),
       "z": dict(cache=CACHE_ROOT / "v11_vanish_all_ctx32_vith", file="isolated_ctx_0_32.npy", attn=rt.RES_ROOT / "attentive_pooling/z/attn.pt", n_tub=16, clip="pos_a",
                 label="context encoder z (32 frames)")}
FIG = rt.FIG_ROOT / "v11_vanish"
S, D, RES, R = 256, 1280, 144.0, 288; T = 8
ORANGE, GREEN, WHITE, YEL, BLUE = (235, 104, 52), (27, 175, 122), (255, 255, 255), (255, 255, 0), (42, 120, 214)
FUT, HID = (215, 215, 215), (80, 80, 80)          # time bar: 문맥 파랑 / 미래 밝은 회색 / 가려짐 진회색 — 읽기 색(주황·초록)과 겹치지 않게
RCOL = {"p": ORANGE, "z": GREEN}                    # 읽기 마커: predictor 주황, encoder 초록 (fig_l2 와 같은 배색)
SAMPLES = list(range(0, 94, 3)); FR = [48 + 6 * t for t in range(T)]
VIOL = "vanish"
COND = {"flat": {"0": "moving_visible_flat", "occ": "moving_occlusion_flat"}, "ramp": {"0": "moving_visible", "occ": "moving_occlusion"},
        "static": {"0": "static_visible", "occ": "static_occlusion"}}
TIMING_SUF = {"late": "", "early": "_early", "mid": "_mid"}          # 가림 타이밍 (v11 조건 이름 접미사). k=0 은 타이밍과 무관하게 visible


def arr(s):
    return np.array([float(v) for v in re.split(r"[;,\s|]+", s.strip()) if v])


def mark(dd, x, y, kind, col=ORANGE):
    if kind == "truth":
        dd.ellipse([x - 9, y - 9, x + 9, y + 9], outline=WHITE, width=2)
    elif kind == "read":
        dd.ellipse([x - 10, y - 10, x + 10, y + 10], fill=col, outline=WHITE, width=2)
    else:                                                   # last seen
        dd.line([x - 5, y - 5, x + 5, y + 5], fill=WHITE, width=2); dd.line([x - 5, y + 5, x + 5, y - 5], fill=WHITE, width=2)


def hidden_samples(m):
    """metadata hidden_start/hidden_end (raw frame) 에 드는 샘플 index. late k 는 range(16−k, 16+k) 와 일치함을 확인했다 (2026-09-11)."""
    try:
        hs, he = float(m["hidden_start"]), float(m["hidden_end"])
    except (KeyError, ValueError):
        return set()
    return {t for t in range(32) if hs <= 3 * t <= he} if he >= hs else set()


def make_gif(path, file_name, truth32, read_tub, n_tub, hid, title, col):
    """가능 영상 32 샘플. truth32 (32,2) px, read_tub (n_tub,2) px — n_tub=8 이면 미래만 (샘플 16..31), 16 이면 전부. hid = 가려진 샘플 집합."""
    Z = 2; W, H = R * Z, R * Z; BAR_H, HEAD = 26, 22; frames = []
    for t, f in enumerate(SAMPLES):
        im = Image.open(FRAMES / file_name / f"{f:06d}.png").convert("RGB").resize((W, H), Image.NEAREST)
        canvas = Image.new("RGB", (W, H + HEAD + BAR_H), (20, 20, 20)); canvas.paste(im, (0, HEAD)); dd = ImageDraw.Draw(canvas)
        dd.text((6, 5), title + "   (both at tubelet = 2-frame granularity)", fill=WHITE)
        # 궤적: 지금까지의 진실 (흰 얇은 선 + 작은 점) 과 읽기 (색 선 + 작은 점), 현재 위치는 큰 마커
        # 진실도 읽기와 같은 tubelet 단위 (두 샘플 평균) 로 그린다 — 같은 granularity 로 비교 (2026-09-12)
        tt = truth32.reshape(16, 2, 2).mean(1)
        tp = [(tt[i // 2][0] * Z, tt[i // 2][1] * Z + HEAD) for i in range(t + 1)]
        if len(tp) > 1:
            dd.line(tp, fill=WHITE, width=2)
        for (x, y) in tp[:-1]:
            dd.ellipse([x - 3, y - 3, x + 3, y + 3], outline=WHITE, width=1)
        mark(dd, *tp[-1], "truth")
        tub = t // 2; j = tub - (16 - n_tub)
        rp = [(read_tub[i][0] * Z, read_tub[i][1] * Z + HEAD) for i in range(0, j + 1)] if j >= 0 else []
        if len(rp) > 1:
            dd.line(rp, fill=col, width=3)
        for (x, y) in rp[:-1]:
            dd.ellipse([x - 4, y - 4, x + 4, y + 4], fill=col)
        if rp:
            mark(dd, *rp[-1], "read", col)
        phase = "context (observed)" if t < 16 else "future"
        note = "" if rp else "   (predictor readout starts at the future)"
        dd.text((6, HEAD + 4), f"sample {t:2d}  raw frame {f:2d}   {phase}{note}", fill=YEL)
        # time bar
        y0 = HEAD + H + 6; x0, x1 = 8, W - 8; seg = (x1 - x0) / 32
        for i in range(32):
            c = HID if i in hid else (BLUE if i < 16 else FUT)
            dd.rectangle([x0 + i * seg, y0, x0 + (i + 1) * seg - 1, y0 + 8], fill=c)
        cx = x0 + (t + 0.5) * seg; dd.polygon([(cx, y0 + 9), (cx - 5, y0 + 17), (cx + 5, y0 + 17)], fill=WHITE)
        dd.text((x0, y0 + 9), "context", fill=BLUE); dd.text((x0 + 16 * seg + 2, y0 + 9), "future", fill=FUT)
        if hid:
            dd.text((x1 - 90, y0 + 9), "dark = hidden", fill=(160, 160, 160))
        frames.append(canvas)
    # 공통 팔레트로 양자화 — 프레임별 적응 팔레트는 작은 마커 색(초록)을 떨어뜨려 프레임마다 색이 바뀐다
    ref = Image.new("RGB", (W, (H + HEAD + BAR_H) * 4)); [ref.paste(frames[i], (0, (H + HEAD + BAR_H) * j)) for j, i in enumerate((0, 10, 20, 31))]
    for (x, y) in [(20, 20), (60, 20), (100, 20), (140, 20)]:                       # 마커 색을 팔레트에 강제로 포함
        ImageDraw.Draw(ref).rectangle([x, y, x + 30, y + 30], fill=col); ImageDraw.Draw(ref).rectangle([x, y + 40, x + 30, y + 70], fill=WHITE)
    pal = ref.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=[350] * 31 + [1200], loop=0)


def run(rep, model, src, row_of, idx, meta, motion, K, n_show, fig_root, timing="late"):
    E = REP[rep]; n_tub = E["n_tub"]
    cond = COND[motion]["0"] if K == "0" else COND[motion]["occ"] + TIMING_SUF[timing]; FIG_K = fig_root / f"k{K}"; FIG_K.mkdir(parents=True, exist_ok=True)
    cell = lambda v: [r for r in idx if r["condition"] == cond and r["sym_k"] == K and r["violation_type"] == VIOL and r["variant"] == v]
    pos = {r["block_id"]: r for r in cell("pos_a")}; imp = {r["block_id"]: r for r in cell("imp_ab")}
    blocks = sorted(pos); read_clip = pos if E["clip"] == "pos_a" else imp
    pred = np.empty((len(blocks), n_tub, 2)); same = 0
    with torch.no_grad():
        for i, b in enumerate(blocks):
            tok = torch.from_numpy(np.asarray(src[row_of[read_clip[b]["video_id"]]])).cuda().float().reshape(n_tub, S, D)
            pred[i] = (model(tok)[0].cpu().numpy() + 1) * RES
            if rep == "p" and i < 3:
                same += int(np.abs(np.asarray(src[row_of[imp[b]["video_id"]]], np.float32) - np.asarray(src[row_of[pos[b]["video_id"]]], np.float32)).max() == 0)
    truth32 = np.stack([np.stack([arr(meta[pos[b]["video_id"]]["object_px_x_by_sample"]), arr(meta[pos[b]["video_id"]]["object_px_y_by_sample"])], -1) for b in blocks])   # (n,32,2)
    truth = truth32.reshape(len(blocks), 16, 2, 2).mean(2)                            # 튜블릿 (n,16,2)
    last = truth[:, 7]; fut = truth[:, 8:]; pf = pred[:, -T:]                            # 미래 8 튜블릿의 읽기
    tag = lambda b: f"{pos[b]['video_id']}   {motion} k={K} ({'visible' if K == '0' else 'occluded, ' + timing})"
    for b, pr, pr_all, t32, tr, lc in list(zip(blocks, pf, pred, truth32, fut, last))[:n_show]:
        # gif: 가능 영상 (pos_a) 32 샘플
        make_gif(FIG_K / f"gif_{pos[b]['video_id']}.gif", pos[b]["file_name"], t32, pr_all, n_tub, hidden_samples(meta[pos[b]["video_id"]]),
                 f"{tag(b)}   white = truth   {'orange' if rep == 'p' else 'green'} = {E['label']} readout", RCOL[rep])
        # overlay png: 미래 8 프레임 (p 는 imp_ab 프레임 = 사라진 자리에 무엇을 그렸나, z 는 pos_a)
        fr_clip = read_clip[b]
        canvas = Image.new("RGB", (R * T, R + 24), "white"); d = ImageDraw.Draw(canvas)
        d.text((6, 5), f"{fr_clip['video_id']}   {motion} k={K}   {'orange' if rep == 'p' else 'green'} = {E['label']} readout   white ring = pos_a truth   x = last seen", fill=(0, 0, 0))
        for t, f in enumerate(FR):
            im = Image.open(FRAMES / fr_clip["file_name"] / f"{f:06d}.png").convert("RGB"); dd = ImageDraw.Draw(im)
            mark(dd, *tr[t], "truth"); mark(dd, *lc, "last"); mark(dd, *pr[t], "read", RCOL[rep]); dd.text((4, 4), f"t{t} f{f}", fill=YEL); canvas.paste(im, (R * t, 24))
        canvas.save(FIG_K / f"overlay_{fr_clip['video_id']}.png")
    sgn = np.sign(fut[:, 7, 0] - last[:, 0]); dt = (fut[:, :, 0] - last[:, None, 0]) * sgn[:, None]; dp = (pf[:, :, 0] - last[:, None, 0]) * sgn[:, None]
    dty = fut[:, :, 1] - last[:, None, 1]; dpy = pf[:, :, 1] - last[:, None, 1]
    g = dp[:, 7].mean() / dt[:, 7].mean() if abs(dt[:, 7].mean()) > 1 else float("nan")   # v11 은 속도 고정 → 비율 (읽은 변위 / 진실 변위, 슬롯 7); static 은 정의 안 됨
    l2 = np.linalg.norm(pf - fut, axis=-1).mean(); l2_last = np.linalg.norm(pf - last[:, None], axis=-1).mean()
    print(f"[{rep} {motion} k={K} {cond}] n={len(blocks)}" + (f" p(imp)==p(pos) {same}/3" if rep == "p" else "") +
          f" | x 슬롯7: 진실 {dt[:,7].mean():.0f} 읽기 {dp[:,7].mean():.0f} px  ratio {g:.2f} | y 슬롯7: 진실 {dty[:,7].mean():+.0f} 읽기 {dpy[:,7].mean():+.0f} | L2 진실 {l2:.1f} / 마지막관측 {l2_last:.1f} px")
    fig, ax = plt.subplots(figsize=(3.6, 2.6)); t = np.arange(T)
    ax.plot(t, dt.mean(0), color="#222", lw=2, label="pos_a truth"); ax.fill_between(t, dt.mean(0) - dt.std(0), dt.mean(0) + dt.std(0), color="#222", alpha=0.08)
    ax.plot(t, dp.mean(0), color="#eb6834", lw=1.8, label=f"{rep} readout"); ax.fill_between(t, dp.mean(0) - dp.std(0), dp.mean(0) + dp.std(0), color="#eb6834", alpha=0.12)
    ax.axhline(0, color="k", lw=0.5, ls=":"); ax.set_xlabel("future tubelet"); ax.set_ylabel("x from last-seen position (px)\nalong travel direction")
    ax.set_title(f"v11 {cond} k={K} (n={len(blocks)})\n{rep}: read/truth at slot 7 = {g:.2f}", fontsize=8); ax.legend(fontsize=7, frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG_K / "fig_gain.png", dpi=200); plt.close(fig)
    np.savez(FIG_K / "readout.npz", block=np.array(blocks), pred=pred, truth=truth, last=last)
    return dict(k=K, cond=cond, n=len(blocks), dt=dt, dp=dp, dty=dty, dpy=dpy, gain=g, l2=float(l2), l2_last=float(l2_last))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rep", choices=sorted(REP), default="p"); ap.add_argument("--motion", choices=sorted(COND), default="flat")
    ap.add_argument("--timing", choices=sorted(TIMING_SUF), default="late"); ap.add_argument("--k", nargs="+", default=["0", "1", "2", "3", "4"]); ap.add_argument("--n", type=int, default=3); a = ap.parse_args()
    E = REP[a.rep]; idx = list(csv.DictReader(INDEX.open())); meta = {r["name"]: r for r in csv.DictReader(META.open())}
    row_of = {v: i for i, v in enumerate(json.loads((E["cache"] / "meta.json").read_text())["video_ids"])}
    model = AttnReadout().cuda(); model.load_state_dict(torch.load(E["attn"], map_location="cuda")); model.eval()
    src = np.load(E["cache"] / E["file"], mmap_mode="r"); fig_root = FIG / a.rep / (a.motion if a.timing == "late" else f"{a.motion}_{a.timing}")
    res = [run(a.rep, model, src, row_of, idx, meta, a.motion, K, a.n, fig_root, a.timing) for K in a.k]
    json.dump([{k: v for k, v in r.items() if k in ("k", "cond", "n", "gain", "l2", "l2_last")} for r in res], open(fig_root / "summary.json", "w"), indent=1)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8)); t = np.arange(T); cmap = plt.get_cmap("viridis")
    for ax, key_t, key_p, lab in ((axes[0], "dt", "dp", "x along travel direction"), (axes[1], "dty", "dpy", "y (+ = down)")):
        ax.plot(t, res[0][key_t].mean(0), color="#222", lw=2.2, label="pos_a truth")
        for j, r in enumerate(res):
            ax.plot(t, r[key_p].mean(0), color=cmap(j / max(len(res) - 1, 1)), lw=1.6, label=f"{a.rep}, k={r['k']} ({'visible' if r['k'] == '0' else 'occl.'})  ratio {r['gain']:.2f}" if key_p == "dp" else f"k={r['k']}")
        ax.axhline(0, color="k", lw=0.5, ls=":"); ax.set_xlabel("future tubelet"); ax.set_ylabel(f"{lab}\nfrom last-seen position (px)"); ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=6.5, frameon=False); axes[0].set_title(f"v11 {a.motion} ({a.timing}): vanish, {E['label']} readout by k (n=56–224 each)", fontsize=8, loc="left")
    fig.tight_layout(); fig.savefig(fig_root / "fig_gain_all_k.png", dpi=200); print(f"→ {fig_root}/fig_gain_all_k.png")


if __name__ == "__main__":
    main()
