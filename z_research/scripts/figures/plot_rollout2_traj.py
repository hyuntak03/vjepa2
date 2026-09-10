#!/usr/bin/env python3
"""RollOut_v2 — h(32프레임 전부, 16 튜블릿) 로 학습한 pooled 위치 readout 을 predictor 미래 8 튜블릿에 test 로만 건다.

자: 튜블릿 256토큰 공간평균 → 좌표(x, y)마다 w 하나(1,281 파라미터), 16 튜블릿 공유, 최소제곱. 학습 = h 의 문맥+미래 16 튜블릿,
    7 시나리오, pos+imp, holdout 레벨 제외, 화면 밖 튜블릿 제외. p 는 학습에 없다.
그림 (holdout 레벨 클립, 시나리오당 2개):
  overlay_hfit/   미래 프레임 8장 위에 predictor 에서 읽은 위치(주황)
  traj_hfit/      마지막 미래 프레임 위에 궤적 두 줄 — 정답(흰) vs predictor(주황), 점 = 튜블릿 0..7
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import rollout2_position as rp                                            # noqa: E402

FRAMES = Path("/local_datasets/world/world_analysis/RollOut_v2"); OUTROOT = rp.ROOT / "z_research/RollOutV2/figures"
ORANGE, WHITE = (235, 104, 52), (255, 255, 255); R = 288


def main():
    idx = list(csv.DictReader(rp.INDEX.open())); vids = [r["video_id"] for r in idx]
    meta = json.loads((rp.CACHE / "meta.json").read_text()); row_of = {v: i for i, v in enumerate(meta["video_ids"])}; rows = np.array([row_of[v] for v in vids])
    res = R / 2
    px = np.stack([rp.arr(r["px_x_by_sample"]) for r in idx]); py = np.stack([rp.arr(r["px_y_by_sample"]) for r in idx])
    L = np.stack([px.reshape(-1, 16, 2).mean(2) / res - 1, py.reshape(-1, 16, 2).mean(2) / res - 1], -1)      # (n, 16, 2)
    inf = (np.stack([rp.arr(r["in_frame_by_sample"]) for r in idx]) > 0).reshape(-1, 16, 2).all(2) & (np.abs(L) <= 1).all(2)
    sc = np.array([r["scenario"] for r in idx]); prim = np.array([float(r["primary"]) for r in idx]); sec = np.array([float(r["secondary"]) for r in idx])
    hold = np.array([r["holdout"] in ("1", "True", "true") for r in idx]); plaus = np.array([int(r["plausible"]) for r in idx])
    # h 16 튜블릿 풀링 (target.npy 전체) — 학습용
    H = np.load(rp.CACHE / "target.npy", mmap_mode="r"); P = np.load(rp.CACHE / "predictor.npy", mmap_mode="r")
    tr = np.where(~hold)[0]; Xh = np.empty((len(tr), 16, rp.D), np.float32); o = np.argsort(rows[tr])
    for k in range(0, len(tr), 64):
        sel = tr[o[k:k + 64]]; Xh[o[k:k + 64]] = np.asarray(H[rows[sel]]).reshape(-1, 16, 256, rp.D).mean(2)
    m = inf[tr].ravel(); X = Xh.reshape(-1, rp.D).astype(np.float64)[m]
    W = {c: rp.fit_lstsq(X, L[tr][:, :, ci].ravel()[m]) for ci, c in enumerate("xy")}
    # 학습 분포 self 확인 (h 미래 8 튜블릿, holdout 클립)
    ho = np.where(hold & (plaus == 1))[0]; Xho = np.empty((len(ho), 16, rp.D), np.float32); o = np.argsort(rows[ho])
    for k in range(0, len(ho), 64):
        sel = ho[o[k:k + 64]]; Xho[o[k:k + 64]] = np.asarray(H[rows[sel]]).reshape(-1, 16, 256, rp.D).mean(2)
    Xp = np.empty((len(ho), 8, rp.D), np.float32)
    for k in range(0, len(ho), 64):
        sel = ho[o[k:k + 64]]; Xp[o[k:k + 64]] = np.asarray(P[rows[sel]]).reshape(-1, 8, 256, rp.D).mean(2)
    ph = {c: rp.apply_w(Xho[:, 8:].reshape(-1, rp.D).astype(np.float64), W[c]).reshape(len(ho), 8) for c in "xy"}
    pp = {c: rp.apply_w(Xp.reshape(-1, rp.D).astype(np.float64), W[c]).reshape(len(ho), 8) for c in "xy"}
    print(f"{'scenario':<8} {'h 미래 self: R²x  MAE':>22}   {'p (test): R²x  MAE   R²y':>24}")
    for scn in rp.SCEN:
        k = sc[ho] == scn; y = L[ho][k, 8:]
        r2h = rp.r2_within(ph["x"][k], y[:, :, 0]); r2p = rp.r2_within(pp["x"][k], y[:, :, 0]); r2py = rp.r2_within(pp["y"][k], y[:, :, 1])
        f = lambda v: " n/a " if v is None else f"{v:5.2f}"
        print(f"{scn:<8} {f(r2h)} {np.abs(ph['x'][k]-y[:,:,0]).mean()*res:5.1f}px            {f(r2p)} {np.abs(pp['x'][k]-y[:,:,0]).mean()*res:5.1f}px  {f(r2py)}")
    # 그림
    for d in ("overlay_hfit", "traj_hfit"):
        (OUTROOT / d).mkdir(parents=True, exist_ok=True)
    frames = [48 + 6 * t for t in range(8)]
    for scn in rp.SCEN:
        ids = ho[sc[ho] == scn]
        pick = [ids[(prim[ids] == lv) & (sec[ids] == sv)][0] for lv in sorted(set(prim[ids]))[:1] for sv in sorted(set(sec[ids]))][:2]
        for i in pick:
            k = int(np.where(ho == i)[0][0]); gt = L[i, 8:]; pr = np.stack([pp["x"][k], pp["y"][k]], -1)
            # overlay
            canvas = Image.new("RGB", (R * 8, R + 24), "white"); d = ImageDraw.Draw(canvas)
            d.text((6, 5), f"{scn}  {idx[i]['video_id']}  primary={prim[i]:g} secondary={sec[i]:g}   orange = position read from predictor output with the readout fit on target encoder (32 frames)", fill=(0, 0, 0))
            for t, f in enumerate(frames):
                im = Image.open(FRAMES / idx[i]["file_name"] / f"{f:06d}.png").convert("RGB"); dd = ImageDraw.Draw(im)
                x, y = (pr[t, 0] + 1) * res, (pr[t, 1] + 1) * res; dd.ellipse([x - 10, y - 10, x + 10, y + 10], fill=ORANGE, outline=WHITE, width=2)
                dd.text((4, 4), f"t{t} f{f}", fill=(255, 255, 0)); canvas.paste(im, (R * t, 24))
            canvas.save(OUTROOT / "overlay_hfit" / f"{scn}_{idx[i]['video_id']}.png")
            # trajectory: 마지막 미래 프레임 위에 두 궤적
            bg = Image.open(FRAMES / idx[i]["file_name"] / f"{frames[-1]:06d}.png").convert("RGB").resize((R * 2, R * 2), Image.NEAREST); dd = ImageDraw.Draw(bg)
            for traj, col, lab in ((gt, WHITE, "truth"), (pr, ORANGE, "predictor")):
                pts = [((p[0] + 1) * res * 2, (p[1] + 1) * res * 2) for p in traj]
                dd.line(pts, fill=col, width=4)
                for t, (x, y) in enumerate(pts):
                    dd.ellipse([x - 7, y - 7, x + 7, y + 7], fill=col, outline=(0, 0, 0)); dd.text((x + 9, y - 9), str(t), fill=col)
            dd.text((8, 8), f"{scn} {idx[i]['video_id']}   white = truth   orange = predictor (readout fit on h 32 frames, test only)   dots = future tubelet 0..7", fill=(255, 255, 0))
            bg.save(OUTROOT / "traj_hfit" / f"{scn}_{idx[i]['video_id']}.png")
        print(f"  [saved] {scn} x2 -> overlay_hfit/, traj_hfit/")


if __name__ == "__main__":
    main()
