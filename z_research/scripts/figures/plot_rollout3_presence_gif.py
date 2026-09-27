#!/usr/bin/env python3
"""RollOutV3 — presence 자가 자기 학습셋에서 무엇을 읽는지 GIF 로. 위치와 **있음/없음** 을 같이 그린다.

한 clip 의 32 샘플 (raw 0, 3, …, 93) 을 차례로 보여 준다:
  흰색    진실 위치 (index 의 px 라벨, 튜블릿 = 2 샘플 평균). **보이는 튜블릿만** 채워 그리고, 안 보이면 테두리만
  색      자가 읽은 위치 (미래 8 튜블릿). presence 가 문턱을 넘으면 **채운 원**, 못 넘으면 **X** (= "여기 없다")
  아래    튜블릿별 presence 막대 — 위가 진실 (있음/없음), 아래가 자의 판정. 문턱은 summary.json 의 `thr_fpr5` (빈 장면 오탐 5 %)

기본 표본은 네 종류를 자동으로 고른다 (전부 held-out):
  normal    물체가 8 튜블릿 내내 보임          occluded  사물 판에 가려지는 구간이 있음
  empty     빈 장면 clip (물체 없음)            offscreen 미래에 화면 밖으로 나감

자는 **attn** 하나다 (attention pooling + Linear head 2 개). 기각한 후보는
`z_research/RollOutV3/Archive/READOUT_CHOICE_2026-09-23.md` 를 볼 것.

출력: z_research/RollOutV3/figures/presence_gif/<종류>_<rep>_<video_id>.gif

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/figures/plot_rollout3_presence_gif.py                      # p 로 4 개
  $P z_research/scripts/figures/plot_rollout3_presence_gif.py --rep h --ms 420
  $P z_research/scripts/figures/plot_rollout3_presence_gif.py --clips <video_id> ...
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
# 자를 배운 세트의 인덱스 (summary.json 의 train_set). 옛 v6 를 하드코딩했었다 — 자를 다시 배우면 행이 안 맞는다
TRAIN_SET = json.loads((ROOT / "z_research/RollOutV3/exp_results/presence/summary.json").read_text()).get("train_set", "training_v6")
INDEX = ROOT / f"data_csv/rollout_v2_{TRAIN_SET}/index_probe.csv"
FRAMES = Path("/local_datasets/world/world_analysis/RollOut_v2_training")
KIND = "attn"                       # 자는 attn 하나로 확정 (2026-09-23)
RES_DIR = ROOT / "z_research/RollOutV3/exp_results/presence"
FIG = ROOT / "z_research/RollOutV3/figures/presence_gif"
SAMPLES = list(range(0, 94, 3))
Z = 2; PW = 288 * Z; PAD, HEAD, FOOT = 24, 74, 160
BG, INK, MUTED = (18, 18, 22), (240, 240, 240), (150, 150, 160)
WHITE, ORANGE, GREEN, BLUE, FUT, RED = (255, 255, 255), (235, 104, 52), (27, 175, 122), (42, 120, 214), (200, 200, 205), (220, 70, 70)
COL = {"p": ORANGE, "z": BLUE, "h": GREEN}
NAME = {"p": "predictor p (sees only the first 16 frames)", "z": "context encoder z (32 frames)", "h": "target encoder h (32 frames)"}
FONT = fm.findfont("DejaVu Sans"); FONT_B = fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold"))


def font(sz, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, sz)


def arr(s):
    return np.array([float(v) for v in str(s).split()], np.float32)


def marker(dd, x, y, col, kind, r=11):
    """kind: 'dot' 채운 원 · 'ring' 테두리만 · 'x' 없다고 판정."""
    if kind == "dot":
        dd.ellipse([x - r, y - r, x + r, y + r], fill=col, outline=WHITE, width=2)
    elif kind == "ring":
        dd.ellipse([x - r, y - r, x + r, y + r], outline=col, width=3)
    else:
        d = r * 0.9
        dd.line([x - d, y - d, x + d, y + d], fill=col, width=4)
        dd.line([x - d, y + d, x + d, y - d], fill=col, width=4)


def encoder_full(r, reps, kind, dev):
    """그 clip 만 온라인으로 다시 뽑아 **문맥 8 + 미래 8 = 16 튜블릿 전부**에 자를 건다 (z, h 만. p 는 미래만 존재).
    학습된 자 가중치를 그대로 쓴다 (재학습 없음)."""
    sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))
    import torch
    import rollout2_presence_readout as pr                                  # noqa: E402
    from analysis.intphys2.model import build_from_config                   # noqa: E402
    from evals.world_model_analysis.data import WMADataset                  # noqa: E402
    import torch.nn.functional as Fn
    ds = WMADataset(dict(data=pr.DATA, model=pr.MODEL, features={"cache_dir": "/tmp"}, surprise={}))
    ids = [x.video_id for x in ds.records]
    bundle = build_from_config(pr.MODEL, dev)
    out = {}
    with torch.no_grad():
        clip = ds.clip(ids.index(r["video_id"])).unsqueeze(0).to(dev, dtype=bundle.dtype)
        for k in reps:
            if k == "p":
                continue
            mod = bundle.context_encoder if k == "z" else bundle.target_encoder
            f = mod(clip)
            f = f[-1] if isinstance(f, list) else f
            tok = Fn.layer_norm(f.float(), (f.size(-1),)).reshape(-1, pr.S, pr.D)      # (16, 256, 1280)
            m = pr.Readout(kind).to(dev)
            m.load_state_dict(torch.load(RES_DIR / k / f"readout_{kind}.pt", map_location=dev)); m.eval()
            xy, pres, _ = m(tok)
            out[k] = (xy.cpu().numpy(), pres.cpu().numpy())                             # (16,2), (16,)
    return out


def make_gif(r, R, kind, tag, out, ms, full=None):
    """R = [(rep, pred (8,2), presence (8,), thr), ...] — 표현 여러 개를 같은 clip 에 겹쳐 그린다.
    full = {rep: (xy (16,2), presence (16,))} 이면 그 표현은 **문맥 튜블릿에도** 읽기를 그린다."""
    px = np.stack([arr(r["px_x_by_sample"]), arr(r["px_y_by_sample"])], -1)
    truth = px.reshape(16, 2, 2).mean(1) * Z if px.size else np.zeros((16, 2))
    vis = (arr(r["visible_by_sample"]) > 0).reshape(16, 2).all(1)          # 양성 = 화면 안 & 장애물과 안 겹침
    ignr = (arr(r["ignore_by_sample"]) > 0).reshape(16, 2).any(1) & ~vis     # 제외 = 화면 안인데 겹침 (라벨 없음)
    PR = {k: (p + 1) * 144.0 * Z for k, p, _, _ in R}
    ON = {k: (q > t) for k, _, q, t in R}
    W = PAD * 2 + PW; H = HEAD + PW + FOOT; frames = []
    for t, f in enumerate(SAMPLES):
        cv = Image.new("RGB", (W, H), BG); dd = ImageDraw.Draw(cv)
        dd.text((PAD, 10), f"RollOut_v2_training {TRAIN_SET} · {r['scenario']}  ({tag})", fill=INK, font=font(19, True))
        dd.text((PAD, 38), f"{r['video_id']}", fill=MUTED, font=font(12))
        thr_s = "  ".join(f"{k} {t:+.2f}" for k, _, _, t in R)
        dd.text((PAD, 54), f"readout {kind}   presence threshold ({thr_s}) = 5% false alarm on empty scenes", fill=MUTED, font=font(12))
        im = Image.open(FRAMES / r["file_name"] / f"{f:06d}.png").convert("RGB").resize((PW, PW), Image.NEAREST)
        di = ImageDraw.Draw(im)
        tub = t // 2
        if px.size and vis[tub]:
            marker(di, *truth[tub], WHITE, "dot", 9)
        elif px.size:
            marker(di, *truth[tub], WHITE, "ring", 9)
            if ignr[tub]:
                di.text((12, 10), "this tubelet is EXCLUDED (object overlaps the obstacle)", fill=(240, 200, 60), font=font(14, True))
        j = tub - 8
        says = []
        for k, _, _, thr_k in R:
            if full and k in full:                                    # 문맥·미래 전부 (encoder)
                xy16, p16 = full[k]
                marker(di, *((xy16[tub] + 1) * 144.0 * Z), COL[k], "dot" if p16[tub] > thr_k else "x")
                if p16[tub] <= thr_k:
                    says.append(k)
            elif j >= 0:                                              # 미래만 (predictor)
                marker(di, *PR[k][j], COL[k], "dot" if ON[k][j] else "x")
                if not ON[k][j]:
                    says.append(k)
        if says:
            di.text((12, PW - 54), f"says NO OBJECT: {', '.join(says)}", fill=COL[says[0]], font=font(16, True))
        if j < 0:
            di.text((12, PW - 30), "context frames — predictor has no output yet", fill=INK, font=font(15, True))
        cv.paste(im, (PAD, HEAD))
        # ── 아래: 시간 막대 + 튜블릿별 presence ──
        y0 = HEAD + PW + 14; x0, x1 = PAD, W - PAD - 250; seg = (x1 - x0) / 32
        for i in range(32):
            dd.rectangle([x0 + i * seg, y0, x0 + (i + 1) * seg - 2, y0 + 10], fill=BLUE if i < 16 else FUT)
        cx = x0 + (t + 0.5) * seg; dd.polygon([(cx, y0 + 12), (cx - 7, y0 + 23), (cx + 7, y0 + 23)], fill=WHITE)
        yb = y0 + 28
        rowsy = [("truth", None)] + [(k, k) for k, _, _, _ in R]
        for ri, (lab, k) in enumerate(rowsy):
            ry = yb + ri * 20
            dd.text((x0, ry), lab, fill=MUTED, font=font(11))
            rng16 = full is not None and k is not None and k in full
            for i in range(16 if (rng16 or k is None) else 8):
                bx = x0 + 56 + i * 15 if (rng16 or k is None) else x0 + 56 + (i + 8) * 15
                if k is None:
                    dd.rectangle([bx, ry, bx + 11, ry + 14],
                                 fill=(WHITE if vis[i] else ((240, 200, 60) if ignr[i] else (60, 60, 66))))
                elif rng16:
                    thr_k = dict((kk, tt) for kk, _, _, tt in R)[k]
                    seen = i <= tub
                    dd.rectangle([bx, ry, bx + 11, ry + 14],
                                 fill=((COL[k] if full[k][1][i] > thr_k else RED) if seen else (45, 45, 50)))
                else:
                    seen = i <= max(tub - 8, -1)
                    dd.rectangle([bx, ry, bx + 11, ry + 14],
                                 fill=((COL[k] if ON[k][i] else RED) if seen else (45, 45, 50)))
        dd.text((x0, yb + len(rowsy) * 20 + 2), "16 tubelets (left 8 = context, right 8 = future) — white = object present, yellow = excluded (overlaps obstacle), red = readout says none", fill=MUTED, font=font(11))
        lx = W - PAD - 240
        marker(dd, lx + 8, y0 + 8, WHITE, "dot", 8); dd.text((lx + 24, y0), "true position (visible)", fill=INK, font=font(12))
        marker(dd, lx + 8, y0 + 30, WHITE, "ring", 8); dd.text((lx + 24, y0 + 22), "true position (hidden)", fill=MUTED, font=font(12))
        for ri, (k, _, _, _) in enumerate(R):
            marker(dd, lx + 8, y0 + 52 + ri * 22, COL[k], "dot", 8)
            dd.text((lx + 24, y0 + 44 + ri * 22), NAME[k].split(" (")[0], fill=INK, font=font(12))
        marker(dd, lx + 8, y0 + 52 + len(R) * 22, INK, "x", 8)
        dd.text((lx + 24, y0 + 44 + len(R) * 22), "that readout: no object", fill=MUTED, font=font(12))
        frames.append(cv)
    ref = Image.new("RGB", (W, H * 3)); [ref.paste(frames[i], (0, H * k)) for k, i in enumerate((4, 18, 31))]
    d = ImageDraw.Draw(ref)
    for k, c in enumerate(tuple(COL.values()) + (WHITE, FUT, RED, BG, INK, MUTED)):
        d.rectangle([20 + 70 * k, 20, 80 + 70 * k, 200], fill=c)
    pal = ref.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    q = [fr.quantize(palette=pal, dither=Image.Dither.NONE) for fr in frames]
    q[0].save(out, save_all=True, append_images=q[1:], duration=[ms] * 31 + [1800], loop=0)


def pick(rows, vis, IGN, te, n=1):
    """네 종류를 held-out 에서 고른다."""
    inf = np.stack([arr(r["in_frame_by_sample"]) for r in rows]).reshape(len(rows), 16, 2).all(2)[:, 8:]
    sc = np.array([r["scenario"] for r in rows])
    v = vis[:, :]
    out = {}
    cand = lambda m: [i for i in np.where(m & te)[0]][:n]
    out["normal"] = cand((sc == "line_x") & v.all(1))
    out["overlap"] = cand(IGN.any(1) & inf.all(1))              # 장애물과 겹치는 (= 라벨 제외) 구간이 있는 clip
    out["empty"] = cand(sc == "empty")
    out["offscreen"] = cand((sc == "line_x") & ~inf.all(1) & inf.any(1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="+", default=["p", "h"], choices=["p", "z", "h"], help="같은 clip 에 겹쳐 그릴 표현들")

    ap.add_argument("--clips", nargs="*")
    ap.add_argument("--n", type=int, default=1, help="종류마다 clip 수")
    ap.add_argument("--ms", type=int, default=400)
    ap.add_argument("--with-context", action="store_true", help="encoder (z/h) 를 문맥 튜블릿에도 그린다 (그 clip 만 온라인 재추출)")
    a = ap.parse_args()

    rows = list(csv.DictReader(INDEX.open()))
    Zs = {k: np.load(RES_DIR / k / f"preds_{KIND}.npz", allow_pickle=True) for k in a.reps}
    z = Zs[a.reps[0]]
    row_of = {v: i for i, v in enumerate(z["video_id"])}
    vis, ign, te = z["pos"], z["ign"], z["test"]
    THR = {}                                                   # 문턱 = val 에서 음성 5 % (학습 스크립트와 같은 방식)
    for k, zz in Zs.items():
        neg = (~zz["pos"]) & (~zz["ign"]); va = zz["val"]
        THR[k] = float(np.quantile(zz["presence"][va][neg[va]], 0.95))
    FIG.mkdir(parents=True, exist_ok=True)
    jobs = ([("clip", rows[row_of[c]]) for c in a.clips] if a.clips
            else [(tag, rows[i]) for tag, ii in pick(rows, vis, ign, te, a.n).items() for i in ii])
    for tag, r in jobs:
        i = row_of[r["video_id"]]
        R = [(k, Zs[k]["pred"][i], Zs[k]["presence"][i], THR[k]) for k in a.reps]
        full = None
        if a.with_context:
            import torch
            full = encoder_full(r, a.reps, KIND, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        out = FIG / f"{tag}_{'-'.join(a.reps)}_{KIND}{'_ctx' if a.with_context else ''}_{r['video_id']}.gif"
        make_gif(r, R, KIND, tag, out, a.ms, full)
        print(f"→ {out.name}  ({r['scenario']})  " +
              "  ".join(f"{k} {np.round(q, 1)}" for k, _, q, _ in R), flush=True)


if __name__ == "__main__":
    main()
