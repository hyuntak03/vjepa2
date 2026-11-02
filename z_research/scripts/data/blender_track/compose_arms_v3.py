#!/usr/bin/env python3
"""BlenderTrack v3 네 팔 만들기 + 목록 (2026-11-01) — 장면 (block) 마다 렌더한 A (target 있음) · B (target 만 뺌) 를 문맥 끝에서 이어 붙인다.

  pos_a  = A                      가능 (가려졌다 다시 나옴)
  pos_b  = B                      가능 (빈 장면은 끝까지 빈다)
  imp_ab = A[0:32] + B[32:64]     불가능 — 가림막 뒤에서 사라짐 (target 이 완전히 가려진 프레임 32 에서 바꾼다)
  imp_ba = B[0:32] + A[32:64]     불가능 — 없던 물체가 가림막 뒤에서 나타남
짝 (문맥이 픽셀 단위로 같다): A = (pos_a, imp_ab), B = (pos_b, imp_ba). 학습에는 pos_a · pos_b 만.
출력 <out>/arms/<split>/<block id>/{pos_a,pos_b,imp_ab,imp_ba}.mp4 (pos 는 clips/ 로 상대 링크) + <arm>.npz (target 라벨: 프레임마다 있음 · amodal · visible)
     data_csv/blender_track_v3/<split>_arms.csv (팔마다 한 줄: 짝 · 가능 · 상황 축 · 요인 · 가림 샘플 · 이음 품질)
이음 품질: 프레임 32 에서 A 와 B 의 target 밖 화소 최대 차 (렌더 잡음 수준이면 이음이 보이지 않는다) · target 이 그 프레임에서 완전히 가려졌나.
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P compose_arms_v3.py --config configs/v3.yaml --split probe --workers 32
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_design_v3 as G  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
SPLICE = 32
CFG = {}


def _init(path, split):
    CFG["cfg"] = G.load_cfg(path, split)
    CFG["split"] = split


def write_mp4(path, frames, fps, crf, gop):
    H, W = frames.shape[1:3]
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
                          "-c:v", "libx264", "-crf", str(crf), "-g", str(gop), "-pix_fmt", "yuv420p", "-movflags", "+faststart", path + ".tmp.mp4"],
                         stdin=subprocess.PIPE)
    p.stdin.write(np.ascontiguousarray(frames).tobytes()); p.stdin.close(); p.wait()
    os.replace(path + ".tmp.mp4", path)


def target_labels(z, T):
    """A 라벨의 target (0 번) 만, 없으면 0."""
    if z is None or z["amodal"].shape[1] == 0 or str(z["roles"][0]) != "target":
        return {"present": np.zeros(T, bool), "amodal": np.zeros((T, 64, 8), np.uint8), "visible": np.zeros((T, 64, 8), np.uint8),
                "amodal_n": np.zeros(T, np.int16), "visible_n": np.zeros(T, np.int16), "center_px": np.full((T, 2), np.nan, np.float32),
                "bbox_px": np.full((T, 4), np.nan, np.float32)}
    return {"present": np.ones(T, bool), "amodal": z["amodal"][:, 0], "visible": z["visible"][:, 0], "amodal_n": z["amodal_n"][:, 0],
            "visible_n": z["visible_n"][:, 0], "center_px": z["center_px"][:, 0], "bbox_px": z["bbox_px"][:, 0]}


def splice(la, lb):
    return {k: np.concatenate([la[k][:SPLICE], lb[k][SPLICE:]]) for k in la}


def one(b):
    cfg, split = CFG["cfg"], CFG["split"]
    out, sd = cfg["out"], cfg["splits"][split]["seed"]
    ca, cb = f"{split}{sd}_{2 * b:06d}", f"{split}{sd}_{2 * b + 1:06d}"
    need = [f"{out}/clips/{c}.mp4" for c in (ca, cb)] + [f"{out}/labels/{c}.npz" for c in (ca, cb)]
    if not all(os.path.exists(f) for f in need):
        return {"block": b, "missing": True}
    import decord
    A = decord.VideoReader(f"{out}/clips/{ca}.mp4")[:].asnumpy(); Bv = decord.VideoReader(f"{out}/clips/{cb}.mp4")[:].asnumpy()
    za, zb = np.load(f"{out}/labels/{ca}.npz"), np.load(f"{out}/labels/{cb}.npz")
    sa = json.load(open(f"{out}/specs/{ca}.json"))
    T = A.shape[0]
    bid = f"{split}{sd}_b{b:06d}"
    d = f"{out}/arms/{split}/{bid}"
    os.makedirs(d, exist_ok=True)
    rc = cfg["render"]
    for arm, c in (("pos_a", ca), ("pos_b", cb)):
        lk = f"{d}/{arm}.mp4"
        if os.path.lexists(lk):
            os.remove(lk)
        os.symlink(os.path.relpath(f"{out}/clips/{c}.mp4", d), lk)
    write_mp4(f"{d}/imp_ab.mp4", np.concatenate([A[:SPLICE], Bv[SPLICE:]]), cfg["fps"], rc["crf"], rc["gop"])
    write_mp4(f"{d}/imp_ba.mp4", np.concatenate([Bv[:SPLICE], A[SPLICE:]]), cfg["fps"], rc["crf"], rc["gop"])
    LA, LB = target_labels(za, T), target_labels(None, T)
    for arm, lab in (("pos_a", LA), ("pos_b", LB), ("imp_ab", splice(LA, LB)), ("imp_ba", splice(LB, LA))):
        np.savez_compressed(f"{d}/{arm}.npz", **lab)
    # 이음 품질: 프레임 31 · 32 에서 target amodal 밖 화소 최대 차 (A vs B) · target 이 그 프레임들에서 완전히 가려졌나
    am = np.unpackbits(za["amodal"][:, 0], axis=-1).astype(bool) if za["amodal"].shape[1] else np.zeros((T, 64, 64), bool)
    mask = np.kron(am[SPLICE - 1] | am[SPLICE], np.ones((4, 4), bool))
    # 렌더 타이밍 (A): 렌더 라벨로 다시 잰 target 가려진 비율이 late 정의 (문맥 끝 완전히 가려짐 · 앞뒤로 보임 · 사건 하나) 를 지키나
    tok = False
    if za["amodal"].shape[1]:
        a_, v_ = za["amodal_n"][:, 0].astype(float), za["visible_n"][:, 0].astype(float)
        hid = np.where(a_ > 0, 1.0 - v_ / np.maximum(a_, 1), 0.0)
        bb = za["bbox_px"][:, 0]; S_ = 256
        w_ = np.maximum(bb[:, 2] - bb[:, 0], 1e-6); h_ = np.maximum(bb[:, 3] - bb[:, 1], 1e-6)
        inw = np.clip(np.minimum(bb[:, 2], S_) - np.maximum(bb[:, 0], 0), 0, None) * np.clip(np.minimum(bb[:, 3], S_) - np.maximum(bb[:, 1], 0), 0, None)
        off = 1.0 - inw / (w_ * h_)
        tok = bool(G.G2.timing_ok("late", hid[::2], off[::2], cfg["timing"])[0])
    return {"block": b, "bid": bid, "ca": ca, "cb": cb, "spec": sa, "dir": d, "render_timing_ok": tok,
            "seam_diff_out": float(np.abs(A[SPLICE].astype(int) - Bv[SPLICE].astype(int))[~mask].max()),
            "seam_mean_out": float(np.abs(A[SPLICE].astype(int) - Bv[SPLICE].astype(int))[~mask].mean()),
            "hidden_at_splice": bool(za["visible_n"][SPLICE - 1, 0] == 0 and za["visible_n"][SPLICE, 0] == 0) if za["amodal"].shape[1] else False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    cfg = G.load_cfg(a.config, a.split)
    nb = int(cfg["splits"][a.split]["n"]) // 2
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.config, a.split)) as ex:
        R = list(ex.map(one, range(nb), chunksize=4))
    rows, miss = [], 0
    for r in R:
        if r.get("missing"):
            miss += 1; continue
        sp = r["spec"]; c = sp["cell"]; F = sp["factors"]
        base = {"block_id": r["bid"], "objset": c["objset"], "place": c["place"], "camera": c["camera"], "occ_type": c["occ_type"], "occ_motion": c["occ_motion"],
                "target_mode": c["target_mode"], "occ_sub": c["occ_sub"], "target_kind": c.get("target_kind"), "ramp_mode": c.get("ramp_mode") or "",
                "count_a": c["count"], "event_s": sp["event_samples"][0], "event_e": sp["event_samples"][1],
                "seam_diff_out": round(r["seam_diff_out"], 1), "seam_mean_out": round(r["seam_mean_out"], 3), "hidden_at_splice": int(r["hidden_at_splice"]),
                "render_timing_ok": int(r["render_timing_ok"]), "voe_ok": int(r["render_timing_ok"] and r["hidden_at_splice"]),
                **{f"f_{k}": v for k, v in F.items()}}
        for arm, pl, pid in (("pos_a", 1, "A"), ("imp_ab", 0, "A"), ("pos_b", 1, "B"), ("imp_ba", 0, "B")):
            rows.append({"video_id": f"{r['bid']}_{arm}", "variant": arm, "plausible": pl, "pair_id": pid,
                         "file": f"{r['dir']}/{arm}.mp4", "labels": f"{r['dir']}/{arm}.npz",
                         "condition": f"{c['objset']}/{c['place']}/{c['camera']}/{c['occ_type']}_{c['occ_motion']}/{c['target_mode']}", **base})
    dcsv = f"{REPO}/data_csv/blender_track_{cfg['name']}"
    os.makedirs(dcsv, exist_ok=True)
    with open(f"{dcsv}/{a.split}_arms.csv", "w", newline="") as f:
        cols = list(dict.fromkeys(k for r in rows for k in r))          # 빗면에만 있는 요인 (f_ramp_mode) 등 열 합집합
        w = csv.DictWriter(f, fieldnames=cols, restval=""); w.writeheader(); w.writerows(rows)
    sd = np.array([r["seam_diff_out"] for r in R if not r.get("missing")]); hs = np.array([r["hidden_at_splice"] for r in R if not r.get("missing")])
    print(f"{a.split}: 장면 {nb} · 네 팔 만든 장면 {len(R) - miss} · 빠진 장면 {miss} → {dcsv}/{a.split}_arms.csv")
    rt = np.array([r["render_timing_ok"] for r in R if not r.get("missing")])
    print(f"  이음: target 밖 화소 최대 차 중앙 {np.median(sd):.1f} · 95 % {np.percentile(sd, 95):.1f} (0–255) · 이음 프레임에서 target 완전히 가려짐 {hs.mean():.3f}")
    print(f"  렌더 타이밍 (late 정의) 통과 {rt.mean():.3f} · 시험 짝으로 쓸 장면 (둘 다) {(rt & hs).mean():.3f}")


if __name__ == "__main__":
    main()
