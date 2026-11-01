"""S5–S6 — 가림 일정 탐색 → 합성 → 품질 검사 → 저장.

입력 (클립 폴더마다): <clean>.npy (M, H, W, 3) uint8, <object>.npz (amodal = 물체 mask, packbits)
출력: sample_<tag>.npz (occluded · amodal · occluder) + sample_<tag>.json (일정 · 통제 변수 · 덮임 곡선)

탐색: (방식, 완전 가림 시작 t_s, 끝 t_e, 가림막 속도 · 방향, 조각) 을 무작위로 뽑는다.
  가림막은 t_m = (t_s + t_e) / 2 에서 위치 c, 프레임 t 에서 c + v (t − t_m) dir 에 있다 (static 은 v = 0).
  [t_s, t_e] 동안 물체를 덮으려면 가림막 기준 좌표에서 물체 mask 들의 합집합 U 를 덮어야 한다
  → U 의 bbox 중심에 놓고, U 를 덮을 때까지 크기를 키운다.
  그다음 **실제 mask 로** 프레임별 덮임 c(t) 를 다시 재서 제약을 검사한다 (공식으로 판정하지 않는다).

제약 (common.SCHEDULE · SOURCES):
  첫 k_vis 장 c ≤ 0.05 / 완전 가림 (c ≥ 0.95) 구간이 문맥 안에서 시작, d_min 장 이상 /
  재등장 (보이는 비율 ≥ 0.5) 이 예측 구간 안 / 재등장 뒤 다시 완전 가림 없음 / 가림막 면적 ≤ 35 %
덮임은 흐린 경계를 넣은 뒤의 α ≥ 0.9 로 잰다 (경계로 비치는 물체 픽셀을 덮인 것으로 세지 않기 위해).

    python datagen/s5_synth.py --clips <dir>/ssv2 --bank <bank> --tag v0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SCHEDULE, SOURCES, pack, save_json, unpack  # noqa: E402

SPEEDS = {"ssv2": [6.0, 10.0, 16.0], "k400": [3.0, 5.0, 8.0, 12.0]}   # 가림막 속도 (px / 프레임)
D_MAX = {"ssv2": 8, "k400": 18}                                     # 완전 가림 설계 길이 상한 (프레임)


class Bank:
    def __init__(self, root: Path):
        self.root = root
        self.items = json.load(open(root / "index.json"))["patches"]

    def load(self, i: int) -> np.ndarray:
        p = cv2.imread(str(self.root / "patches" / f"{self.items[i]['ann_id']}.png"), cv2.IMREAD_UNCHANGED)
        return cv2.cvtColor(p, cv2.COLOR_BGRA2RGBA)


def scaled(patch: np.ndarray, s: float) -> np.ndarray:
    h, w = patch.shape[:2]
    nw, nh = max(2, int(round(w * s))), max(2, int(round(h * s)))
    return cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)


def paste(dst: np.ndarray, src: np.ndarray, cx: float, cy: float) -> None:
    """src 의 중심을 (cx, cy) 에 두고 dst 에 더한다 (화면 밖은 잘린다). dst/src 의 앞 두 축이 (H, W)."""
    H, W = dst.shape[:2]
    h, w = src.shape[:2]
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if xa < xb and ya < yb:
        dst[ya:yb, xa:xb] = src[ya - y0:yb - y0, xa - x0:xb - x0]


def shift(m: np.ndarray, dx: int, dy: int) -> np.ndarray:
    out = np.zeros_like(m)
    H, W = m.shape
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    if abs(dx) < W and abs(dy) < H:
        out[yd, xd] = m[ys, xs]
    return out


def bbox(m: np.ndarray):
    ys, xs = np.nonzero(m)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def occluder_track(M, t_m, center, v, direction):
    t = np.arange(M) - t_m
    return center[0] + v * t * direction[0], center[1] + v * t * direction[1]


def render_binary(pmask: np.ndarray, cxs, cys, H, W) -> np.ndarray:
    out = np.zeros((len(cxs), H, W), bool)
    for t, (cx, cy) in enumerate(zip(cxs, cys)):
        paste(out[t], pmask, cx, cy)
    return out


def check(cov: np.ndarray, occ_area: np.ndarray, t_s: int, M: int, T_ctx: int, k_vis: int, d_min: int):
    """덮임 곡선이 제약을 만족하면 일정 dict, 아니면 (None, 이유)."""
    S = SCHEDULE
    if (cov[:k_vis] > S["vis_max_cov"]).any():
        return None, "early_not_visible"
    full = cov >= S["full_cov"]
    if not full[t_s]:
        return None, "not_full_at_ts"
    s0, e0 = t_s, t_s
    while s0 > 0 and full[s0 - 1]:
        s0 -= 1
    while e0 < M - 1 and full[e0 + 1]:
        e0 += 1
    if s0 > T_ctx - 1:
        return None, "occ_starts_in_pred"
    if e0 - s0 + 1 < d_min:
        return None, "occ_too_short"
    vis = 1.0 - cov
    after = np.nonzero(vis[e0 + 1:] >= S["r_vis"])[0]
    if len(after) == 0:
        return None, "no_reappear"
    r = e0 + 1 + int(after[0])
    if r < T_ctx:
        return None, "reappear_in_ctx"
    if full[r:].any():
        return None, "reoccluded"
    if occ_area.max() > S["max_occ_area"]:
        return None, "occluder_too_big"
    return dict(occ_start=s0, occ_end=e0, d_occ=e0 - s0 + 1, reappear=r,
                boundary_cov=float(cov[T_ctx - 1]), boundary_full=bool(full[T_ctx - 1])), "ok"


def try_candidate(A, area, patch, mode, t_s, t_e, v, direction, M, T_ctx, k_vis, d_min, blur_sigma):
    H, W = A.shape[1:]
    t_m = 0.5 * (t_s + t_e)
    U = np.zeros((H, W), bool)
    for t in range(t_s, t_e + 1):
        d = v * (t - t_m)
        U |= shift(A[t], -int(round(d * direction[0])), -int(round(d * direction[1])))
    if not U.any():
        return None, "empty"
    ux0, uy0, ux1, uy1 = bbox(U)
    center = (0.5 * (ux0 + ux1), 0.5 * (uy0 + uy1))
    pm = patch[..., 3] >= 128
    ph, pw = pm.shape
    # 가림막 기준 좌표에서 U 를 덮는 최소 크기 (1.08 배씩 키운다)
    s = max((ux1 - ux0) / pw, (uy1 - uy0) / ph)
    for _ in range(40):
        pm_s = scaled(pm.astype(np.uint8) * 255, s) >= 128
        if pm_s.sum() > SCHEDULE["max_occ_area"] * H * W:
            return None, "occluder_too_big"
        canvas = np.zeros((H, W), bool)
        paste(canvas, pm_s, *center)
        if (U & canvas).sum() >= 0.995 * U.sum():
            break
        s *= 1.08
    else:
        return None, "cannot_cover"
    # 경계를 흐린 α 로 실제 프레임별 덮임을 잰다
    rgba = scaled(patch, s)
    pad = int(np.ceil(3 * blur_sigma)) + 1
    alpha_p = np.pad(rgba[..., 3].astype(np.float32) / 255.0, pad)
    if blur_sigma > 0:
        alpha_p = cv2.GaussianBlur(alpha_p, (0, 0), blur_sigma)
    rgb_p = np.pad(rgba[..., :3], ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    cxs, cys = occluder_track(M, t_m, center, v, direction)
    alpha = np.zeros((M, H, W), np.float32)
    for t in range(M):
        paste(alpha[t], alpha_p, cxs[t], cys[t])
    strict = alpha >= 0.9
    cov = (A & strict).sum((1, 2)) / np.maximum(area, 1)
    occ_area = (alpha >= 0.5).sum((1, 2)) / (H * W)
    sched, why = check(cov, occ_area, t_s, M, T_ctx, k_vis, d_min)
    if sched is None:
        return None, why
    return dict(sched=sched, cov=cov, alpha=alpha, rgb_p=rgb_p, cxs=cxs, cys=cys,
                occ_wh=[int(rgba.shape[1]), int(rgba.shape[0])], scale=float(s),
                occ_area_max=float(occ_area.max())), "ok"


def composite(clean, alpha, rgb_p, cxs, cys):
    M, H, W, _ = clean.shape
    out = clean.astype(np.float32)
    for t in range(M):
        layer = np.zeros((H, W, 3), np.float32)
        paste(layer, rgb_p.astype(np.float32), cxs[t], cys[t])
        a = alpha[t][..., None]
        out[t] = out[t] * (1 - a) + layer * a
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def object_motion(A):
    cents = []
    for m in A:
        ys, xs = np.nonzero(m)
        cents.append((xs.mean(), ys.mean()) if len(xs) else (np.nan, np.nan))
    g = np.array(cents)
    step = np.linalg.norm(np.diff(g, axis=0), axis=1)
    return g, step


def synth_clip(clip_dir: Path, bank: Bank, rng, tag: str, clean_name: str, object_name: str,
               n_try: int, blur_sigma: float):
    meta = json.load(open(clip_dir / "clip.json"))
    src = meta["source"]
    M, T_ctx = meta["M"], meta["T_ctx"]
    k_vis, d_min = SOURCES[src]["k_vis"], SOURCES[src]["d_min"]
    clean = np.load(clip_dir / clean_name)
    z = np.load(clip_dir / object_name)
    A = unpack(z["amodal_bits"], z["amodal_shape"])
    area = A.sum((1, 2))
    g, step = object_motion(A)

    pref = "static" if rng.random() < 0.5 else "moving"
    reasons: dict[str, int] = {}
    found = {"static": [], "moving": []}
    for _ in range(n_try):
        mode = pref if rng.random() < 0.7 else ("moving" if pref == "static" else "static")
        t_s = int(rng.integers(k_vis, T_ctx))
        lo = max(t_s + d_min - 1, T_ctx - 1)
        hi = min(t_s + D_MAX[src] - 1, M - 2)
        if lo > hi:
            continue
        t_e = int(rng.integers(lo, hi + 1))
        if mode == "static":
            v, direction = 0.0, (0.0, 0.0)
        else:
            v = float(rng.choice(SPEEDS[src]))
            th = rng.uniform(0, 2 * np.pi)
            direction = (float(np.cos(th)), float(np.sin(th)))
        pi = int(rng.integers(len(bank.items)))
        res, why = try_candidate(A, area, bank.load(pi), mode, t_s, t_e, v, direction,
                                 M, T_ctx, k_vis, d_min, blur_sigma)
        reasons[f"{mode}:{why}"] = reasons.get(f"{mode}:{why}", 0) + 1
        if res is not None:
            res.update(mode=mode, t_s_design=t_s, t_e_design=t_e, v=v, direction=direction, patch=pi)
            found[mode].append(res)
            if found[pref]:
                break
    pick = (found[pref] or found["static" if pref == "moving" else "moving"] or [None])[0]
    report = dict(clip_id=meta["clip_id"], tag=tag, preferred_mode=pref, tries=reasons)
    if pick is None:
        report["status"] = "no_feasible_schedule"
        save_json(clip_dir / f"sample_{tag}.json", report)
        return report

    occluded = composite(clean, pick["alpha"], pick["rgb_p"], pick["cxs"], pick["cys"])
    occ_mask = pick["alpha"] >= 0.5
    s = pick["sched"]
    occ_steps = step[max(s["occ_start"] - 1, 0):s["occ_end"] + 1]
    item = bank.items[pick["patch"]]
    report.update(
        status="ok", mode=pick["mode"], **s,
        T_ctx=T_ctx, M=M, fps=meta["fps"],
        occluder=dict(category=item["category"], ann_id=item["ann_id"], wh_px=pick["occ_wh"],
                      area_max=pick["occ_area_max"], speed_px_per_frame=pick["v"],
                      direction=pick["direction"], blur_sigma=blur_sigma),
        object=dict(area_frac_mean=float(area.mean() / A[0].size), speed_px_per_frame=float(np.nanmean(step)),
                    speed_during_occ=float(np.nanmean(occ_steps)) if len(occ_steps) else None,
                    path_px=float(np.nansum(step))),
        design=dict(t_s=pick["t_s_design"], t_e=pick["t_e_design"]),
        coverage=[round(float(c), 4) for c in pick["cov"]],
        occluder_track=[[round(float(x), 1), round(float(y), 1)] for x, y in zip(pick["cxs"], pick["cys"])],
    )
    np.savez_compressed(clip_dir / f"sample_{tag}.npz", occluded=occluded,
                        **{f"amodal_{k}": v for k, v in pack(A).items()},
                        **{f"occluder_{k}": v for k, v in pack(occ_mask).items()})
    save_json(clip_dir / f"sample_{tag}.json", report)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, required=True, help="S0 출력 폴더 (클립 폴더들의 부모)")
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--tag", default="v0")
    ap.add_argument("--clean_name", default="clean.npy")
    ap.add_argument("--object_name", default="object.npz")
    ap.add_argument("--n_try", type=int, default=400)
    ap.add_argument("--blur_sigma", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bank = Bank(args.bank)
    rng = np.random.default_rng(args.seed)
    n_ok = 0
    clips = sorted(p for p in args.clips.iterdir() if (p / args.object_name).exists())
    for d in clips:
        oj = d / args.object_name.replace(".npz", ".json")
        obj = json.load(open(oj)) if oj.exists() else {}
        if obj.get("status", "ok") != "ok":
            continue
        rep = synth_clip(d, bank, rng, args.tag, args.clean_name, args.object_name, args.n_try, args.blur_sigma)
        n_ok += rep["status"] == "ok"
        if rep["status"] == "ok":
            print(f"{d.name}: {rep['mode']:7s} occ {rep['occ_start']}-{rep['occ_end']} (d={rep['d_occ']}) "
                  f"reappear {rep['reappear']} (T_ctx={rep['T_ctx']}) occ_area {rep['occluder']['area_max']:.2f}")
        else:
            top = sorted(rep["tries"].items(), key=lambda kv: -kv[1])[:3]
            print(f"{d.name}: FAIL {top}")
    print(f"ok {n_ok} / {len(clips)}")


if __name__ == "__main__":
    main()
