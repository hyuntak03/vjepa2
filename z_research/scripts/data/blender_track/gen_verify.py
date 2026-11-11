#!/usr/bin/env python3
"""BlenderTrack v1 렌더 검증 — 스모크 (또는 아무 분할) 결과가 쓸 만한지 자동으로 본다. 어기면 종료 코드 1.

  · mp4 마다 decord 로 끝까지 읽힌다 · 프레임 수 = T · 크기 = size²
  · 라벨 npz 키 · 길이 = T · target amodal 칸이 화면 안 프레임에서 비지 않는다
  · 첫 clip 의 numpy ↔ Blender 투영 오차 ≤ 0.01 px (작업자 로그)
  · (10-31) 물리: 구가 바닥 · 빗면을 뚫지 않는다 · 물체가 자기 중심 둘레로 돈다 (physics_check)
  · clip 당 시간 → 본 렌더 예상 시간 (설정의 노드 · GPU · 작업자 수로)
  · 확인 시트 PNG: clip 12 개 × 프레임 8 장 + target amodal (주황) · visible (파랑) 윤곽
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; $P gen_verify.py --config configs/v1.yaml --split smoke
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_design as G  # noqa: E402


def physics_check(out, labs, sink_tol=0.02, ramp_tol=0.03):
    """(10-31) 렌더에 쓴 궤적 (gen_design.tracks: 메시 원점 + 회전) 으로 다시 낸 물체 중심이 바닥 · 빗면을 뚫지 않나.
    구: 중심 높이 ≥ r (바닥), 빗면 위 구간: 면까지 거리 ≥ r. 어긴 깊이를 지름 비율로 재고 tol 을 넘으면 bad.
    원점 + R (h ẑ) = 바닥면 중심 + h ẑ (자기 중심 둘레 회전) 도 확인한다 (첫 v1 렌더의 버그가 여기서 걸린다)."""
    r_ = {"n_clips": 0, "n_obj": 0, "sink_max_frac": 0.0, "ramp_pen_max_frac": 0.0, "pivot_err_max_m": 0.0, "bad": []}
    for f in labs:
        cid = os.path.basename(f)[:-4]
        spec = json.load(open(f"{out}/specs/{cid}.json"))
        trs, _, _ = G.tracks(spec)
        r_["n_clips"] += 1
        for k, (o, (loc, _, R)) in enumerate(zip(spec["objects"], trs)):
            h = 0.5 * o["s"] * G.DIMS[o["shape"]][2]
            c = loc + np.einsum("tij,j->ti", R, np.array([0.0, 0.0, h]))
            pos, _, _ = G.track(o, spec["T"], spec["fps"])
            pe = float(np.abs(c - (pos + np.array([0.0, 0.0, h]))).max())
            r_["pivot_err_max_m"] = max(r_["pivot_err_max_m"], pe)
            r_["n_obj"] += 1
            if o["shape"] == "sphere":
                sink = max(0.0, float((h - c[:, 2]).max())) / (2 * h)
                r_["sink_max_frac"] = max(r_["sink_max_frac"], sink)
                if sink > sink_tol:
                    r_["bad"].append(f"{cid} 물체 {k} 바닥 아래 {sink:.3f} 지름")
            m = o["motion"]
            if m["kind"] == "ramp":
                th, sx = np.radians(m["theta_deg"]), m["sx"]
                p0 = np.array([m["x_low"], m["y"], 0.0])
                u = (c - p0) @ np.array([sx * np.cos(th), 0.0, np.sin(th)])
                d = (c - p0) @ np.array([-sx * np.sin(th), 0.0, np.cos(th)])
                on = (u >= 0) & (u <= m["L"])
                if on.any():
                    pen = max(0.0, float((h - d[on]).max())) / (2 * h)
                    r_["ramp_pen_max_frac"] = max(r_["ramp_pen_max_frac"], pen)
                    if pen > ramp_tol:
                        r_["bad"].append(f"{cid} 물체 {k} 빗면 속 {pen:.3f} 지름")
            if pe > 1e-6:
                r_["bad"].append(f"{cid} 물체 {k} 회전 중심 어긋남 {pe:.3f} m")
    r_["n_bad"] = len(r_["bad"])
    return r_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="smoke")
    a = ap.parse_args()
    cfg = G.load_cfg(a.config, a.split)
    out, sp, rc = cfg["out"], cfg["splits"][a.split], cfg["render"]
    T, size = int(round(cfg["seconds"] * cfg["fps"])), int(cfg["size"])
    pre = f"{a.split}{sp['seed']}_"
    labs = sorted(glob.glob(f"{out}/labels/{pre}*.npz"))
    res = {"split": a.split, "expected": sp["n"], "found": len(labs), "errors": []}
    import decord
    for f in labs:
        cid = os.path.basename(f)[:-4]
        try:
            vr = decord.VideoReader(f"{out}/clips/{cid}.mp4")
            if len(vr) != T:
                res["errors"].append(f"{cid}: 프레임 {len(vr)} != {T}")
            fr = vr.get_batch([0, T // 2, T - 1]).asnumpy()
            if fr.shape[1:] != (size, size, 3):
                res["errors"].append(f"{cid}: 크기 {fr.shape}")
            z = np.load(f)
            if z["center_px"].shape[0] != T or z["amodal"].shape[0] != T:
                res["errors"].append(f"{cid}: 라벨 길이")
            if not (z["amodal_n"][:, 0] > 0).any():
                res["errors"].append(f"{cid}: target amodal 칸이 한 번도 없음")
        except Exception as e:  # noqa: BLE001
            res["errors"].append(f"{cid}: {type(e).__name__} {e}")
    projs, secs, blender_phys = [], [], []
    for lg in glob.glob(f"{out}/logs/{a.split}_w*.log"):
        for line in open(lg, errors="ignore"):
            m = re.search(r"\] \S+ ([0-9.]+)s ", line)
            if line.startswith("[w") and m:
                secs.append(float(m.group(1)))
                pe = re.search(r"proj_err ([0-9.]+)px", line)
                if pe:
                    projs.append(float(pe.group(1)))
                pv = re.search(r"pivot_err ([0-9.]+)m sink ([0-9.]+)", line)
                if pv:
                    blender_phys.append((float(pv.group(1)), float(pv.group(2))))
            if "Traceback" in line:
                res["errors"].append(f"{os.path.basename(lg)}: Traceback")
    res["proj_err_px_max"] = max(projs) if projs else None
    res["blender_pivot_err_max_m"] = max((x[0] for x in blender_phys), default=None)
    res["blender_sphere_sink_max_frac"] = max((x[1] for x in blender_phys), default=None)
    if blender_phys and (res["blender_pivot_err_max_m"] > 1e-3 or res["blender_sphere_sink_max_frac"] > 0.02):
        res["errors"].append(f"Blender 물리: 회전 중심 어긋남 {res['blender_pivot_err_max_m']:.4f} m · 구 바닥 아래 {res['blender_sphere_sink_max_frac']:.3f} 지름")
    if projs and max(projs) > 0.01:
        res["errors"].append(f"투영 오차 {max(projs):.3f} px")
    if secs:
        med = float(np.median(secs))
        n_total = sum(cfg["splits"][s]["n"] for s in ("train", "val", "ood"))
        workers = sum(rc.get("node_gpus") or [rc["gpus_per_node"]] * len(rc["nodes"])) * rc["workers_per_gpu"]
        res["sec_per_clip_worker_median"] = round(med, 1)
        res["full_render_hours_est"] = round(n_total * med / workers / 3600, 2)
    sizes = [os.path.getsize(f"{out}/clips/{os.path.basename(f)[:-4]}.mp4") for f in labs if os.path.exists(f"{out}/clips/{os.path.basename(f)[:-4]}.mp4")]
    res["mp4_mb_mean"] = round(float(np.mean(sizes)) / 1e6, 2) if sizes else None
    if len(labs) < sp["n"]:
        res["errors"].append(f"끝난 clip {len(labs)} / {sp['n']}")
    res["physics"] = physics_check(out, labs)
    for e in res["physics"]["bad"][:20]:
        res["errors"].append(f"물리: {e}")
    # 확인 시트
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fr_idx = np.linspace(0, T - 1, 8).astype(int)
        sel = labs[:12]
        fig, axes = plt.subplots(len(sel), len(fr_idx), figsize=(1.6 * len(fr_idx), 1.75 * len(sel)), squeeze=False)
        for r_, f in enumerate(sel):
            cid = os.path.basename(f)[:-4]
            z = np.load(f); sp_ = json.load(open(f"{out}/specs/{cid}.json"))
            frames = decord.VideoReader(f"{out}/clips/{cid}.mp4").get_batch(list(fr_idx)).asnumpy()
            am = np.unpackbits(z["amodal"][:, 0], axis=-1).astype(bool); vi = np.unpackbits(z["visible"][:, 0], axis=-1).astype(bool)
            for c_, t in enumerate(fr_idx):
                ax = axes[r_, c_]; ax.imshow(frames[c_]); ext = (0, size, size, 0)
                if am[t].any():
                    ax.contour(np.kron(am[t], np.ones((size // 64, size // 64))), levels=[0.5], colors=["#eb6834"], linewidths=0.8, extent=ext, origin="upper")
                if vi[t].any():
                    ax.contour(np.kron(vi[t], np.ones((size // 64, size // 64))), levels=[0.5], colors=["#2a78d6"], linewidths=0.8, extent=ext, origin="upper")
                ax.set_title(f"f{t} vis {int(z['visible_n'][t, 0])}/{int(z['amodal_n'][t, 0])}", fontsize=6, pad=1)
                ax.set_xlim(0, size); ax.set_ylim(size, 0); ax.set_xticks([]); ax.set_yticks([])
            axes[r_, 0].set_ylabel(f"{cid}\n{sp_['objects'][0]['motion']['kind']}", fontsize=6)
        plt.tight_layout(pad=0.2)
        res["sheet"] = f"{out}/{a.split}_sheet.png"
        plt.savefig(res["sheet"], dpi=120)
    except Exception as e:  # noqa: BLE001
        res["errors"].append(f"시트: {type(e).__name__} {e}")
    res["pass"] = not res["errors"]
    print(json.dumps(res, indent=1, ensure_ascii=False))
    json.dump(res, open(f"{out}/{a.split}_verify.json", "w"), indent=1, ensure_ascii=False)
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
