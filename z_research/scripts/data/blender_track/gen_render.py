#!/usr/bin/env python3
"""BlenderTrack v1 렌더 (conda env `blender`, bpy 5.0.1 Cycles) — 설정 파일 하나로 (2026-10-31).

v0 렌더러 (render.Scene) 를 그대로 쓰고 세 가지만 바꾼다:
  · 궤적 = gen_design.tracks (물리 운동 · 회전 행렬) · 소품 (경사로 · 벽) · 그림자 끔 (설정)
  · 프레임 → JPEG (메모리 디스크) → ffmpeg 로 clip 하나 = mp4 하나 (PNG 저장이 프레임 시간의 40 % 였다, 10-31 실측)
  · 출력 배치: <out>/clips/<id>.mp4 · <out>/labels/<id>.npz · <out>/specs/<id>.json (labels 가 마지막에 원자적으로 쓰여 완료 표시)
id = <분할 이름><seed>_<idx:06d> (예: train10_000123). 라벨 npz 는 v0 와 같은 키 (pos · center_px · amodal · visible …).
  B=/data/hyuntak/anaconda3/envs/blender/bin/python
  CUDA_VISIBLE_DEVICES=0 $B gen_render.py --config configs/v1.yaml --split smoke --worker 0 --nworkers 1
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_design as G  # noqa: E402
import render as R  # noqa: E402


def render_clip(S, spec, out, cfg, wtag, check=False):
    bpy = S.bpy
    cid = G.clip_id(spec)
    shm = f"/dev/shm/btv1_{wtag}_{cid}"
    os.makedirs(shm, exist_ok=True)
    S.setup(spec)
    K = sum(o["role"] in ("target", "distractor") for o in spec["objects"])
    T = spec["T"]
    acc = {k: [] for k in ("pos", "yaw", "center", "depth", "bbox", "am", "vi")}
    err = None
    phys = {"pivot": 0.0, "sink": 0.0}                       # (10-31) Blender 안에서 잰 회전 중심 어긋남 (m) · 구가 바닥 아래로 들어간 깊이 / 지름
    if check:
        from mathutils import Vector
        if hasattr(G, "center_path"):                                # v2: track(o, T, fps, g) → (pos, yaw, R), 중심 = center_path
            ctr_np = []
            for o in spec["objects"]:
                pos_, _, R_ = G.track(o, T, spec["fps"], spec["g"])
                ctr_np.append(G.center_path(o, pos_, R_))
        else:
            ctr_np = [G.track(o, T, spec["fps"])[0] + np.array([0.0, 0.0, 0.5 * o["s"] * G.DIMS[o["shape"]][2]]) for o in spec["objects"]]
    t_r = t_l = 0.0
    for f in range(T):
        S.set_frame(f)
        if check and f == 0:
            err = S.check_projection(spec)
        if check and f % 10 == 0:
            for k, (ob, o) in enumerate(zip(S.objs, spec["objects"])):
                mw = ob.matrix_world
                cB = mw @ Vector((0.0, 0.0, 0.5 * G.DIMS[o["shape"]][2]))
                phys["pivot"] = max(phys["pivot"], float(np.abs(np.array(cB) - ctr_np[k][f]).max()))
                if o["shape"] == "sphere":
                    zmin = min((mw @ v.co).z for v in ob.data.vertices)
                    phys["sink"] = max(phys["sink"], max(0.0, -zmin) / o["s"])
        t0 = time.time()
        lab = S.labels(f, K, spec)
        t1 = time.time()
        for k in acc:
            acc[k].append([x[k] for x in lab])
        S.scn.render.filepath = f"{shm}/{f:04d}.jpg"
        bpy.ops.render.render(write_still=True)
        t_l += t1 - t0; t_r += time.time() - t1
    S.teardown()
    t2 = time.time()
    rc = cfg["render"]
    uniq = f"{os.uname().nodename}_{os.getpid()}"                # 임시 파일 이름을 작업자마다 다르게 (도우미가 같은 clip 을 동시에 그려도 서로 안 덮는다)
    tmp = f"{out}/clips/{cid}.{uniq}.tmp.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(spec["fps"]), "-i", f"{shm}/%04d.jpg",
           "-c:v", "libx264", "-crf", str(rc["crf"]), "-g", str(rc["gop"]), "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg 실패 {cid}: {p.stderr[-500:]}")
    os.replace(tmp, f"{out}/clips/{cid}.mp4")
    shutil.rmtree(shm, ignore_errors=True)
    am, vi = np.array(acc["am"]), np.array(acc["vi"])
    if K == 0:                                                  # v3 빈 장면 (target · 방해 물체 없음): 모양을 (T, 0, …) 으로 맞춘다
        am = vi = np.zeros((T, 0, 64, 64), bool)
        for k in ("pos", "center", "bbox"):
            acc[k] = np.zeros((T, 0, {"pos": 3, "center": 2, "bbox": 4}[k]), np.float32)
        acc["yaw"] = acc["depth"] = np.zeros((T, 0), np.float32)
    np.savez_compressed(f"{out}/labels/{cid}.{uniq}.tmp.npz",
                        pos=np.array(acc["pos"], np.float32), yaw=np.array(acc["yaw"], np.float32),
                        center_px=np.array(acc["center"], np.float32), depth=np.array(acc["depth"], np.float32),
                        bbox_px=np.array(acc["bbox"], np.float32),
                        amodal=np.packbits(am, axis=-1), visible=np.packbits(vi, axis=-1),
                        amodal_n=am.sum((-1, -2)).astype(np.int16), visible_n=vi.sum((-1, -2)).astype(np.int16),
                        panel_ang=np.array(S.pan_ang, np.float32).T.reshape(T, -1),
                        panel_x=np.array(S.pan_x, np.float32).T.reshape(T, -1),
                        roles=np.array([o["role"] for o in spec["objects"][:K]]))
    json.dump(spec, open(f"{out}/specs/{cid}.json", "w"))
    os.replace(f"{out}/labels/{cid}.{uniq}.tmp.npz", f"{out}/labels/{cid}.npz")
    return t_r, t_l, time.time() - t2, (err, phys if check else None)


def claim(lk, stale_s=900):
    """잠금 파일을 만들면 이 작업자 몫 (NFS 에서도 O_EXCL 은 원자적). 이미 있으면 남의 것 — 단 15 분 넘게 안 끝났으면 죽은 작업자 것으로 보고 넘겨받는다.
    두 작업자가 같은 clip 을 그려도 임시 파일 이름이 달라 서로 안 덮고, 끝은 원자적 교체라 결과가 깨지지 않는다 (시간만 버린다)."""
    try:
        fd = os.open(lk, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.uname().nodename} {os.getpid()} {time.time():.0f}".encode()); os.close(fd)
        return True
    except FileExistsError:
        try:
            if time.time() - os.path.getmtime(lk) > stale_s:
                os.utime(lk)
                return True
        except OSError:
            pass
        return False


def main():
    global G
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--reverse", action="store_true", help="몫을 뒤에서부터 (남은 분할을 다른 노드가 거들 때)")
    ap.add_argument("--extra_list", default=None, help="11-01: 칸 채우기 번호 목록 (gen_topup_v2.py 의 extra_<split>.txt) — 이 번호들을 먼저 그린다")
    ap.add_argument("--claim", action="store_true", help="동적 분배: 분할 전체를 돌며 아직 아무도 안 잡은 clip 을 잠금 파일로 잡아 그린다 (노드 · job 수가 달라도, 늦게 떠도 된다)")
    a = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:])
    cfg = G.load_cfg(a.config, a.split)
    if cfg.get("design", "gen_design") != "gen_design":            # v2 (11-01): 설계 모듈을 설정에서 고른다 (gen_design_v2 = 축 교차 칸)
        import importlib
        G = importlib.import_module(cfg["design"])
    sp, rc = cfg["splits"][a.split], cfg["render"]
    out = cfg["out"]
    for d in ("clips", "labels", "specs"):
        os.makedirs(f"{out}/{d}", exist_ok=True)

    class A:
        samples = rc["samples"]
        threads = rc["threads"]
    S = R.Scene(A)
    S.track_fn = G.tracks
    S.scn.cycles.use_denoising = bool(rc["denoise"])
    st = S.scn.render.image_settings
    st.file_format, st.color_mode, st.quality = "JPEG", "RGB", int(rc["jpeg_quality"])
    todo = [i for i in range(sp["n"]) if i % a.nworkers == a.worker]
    if a.claim:                                                # 11-01: 전체를 돌되 작업자마다 다른 곳에서 시작 (잠금 다툼을 줄인다)
        st = (a.worker * sp["n"]) // max(a.nworkers, 1)
        todo = list(range(st, sp["n"])) + list(range(st))
        os.makedirs(f"{out}/locks", exist_ok=True)
    if a.extra_list and os.path.exists(a.extra_list):          # 칸 채우기 번호 (분할 크기 밖, 설계는 design/ 에 미리 있다)
        todo = [int(x) for x in open(a.extra_list).read().split()] + todo
    if a.reverse:                                              # 도우미: 같은 몫을 뒤에서부터 (본 작업자와 가운데서 만난다, 끝난 clip 은 건너뜀)
        todo = todo[::-1]
    t_all, n_done = time.time(), 0
    for n, i in enumerate(todo):
        cid = f"{a.split}{sp['seed']}_{i:06d}"
        if os.path.exists(f"{out}/labels/{cid}.npz") and os.path.exists(f"{out}/clips/{cid}.mp4"):
            continue
        if a.claim and not claim(f"{out}/locks/{cid}.lock"):
            continue
        dj, dfail = f"{out}/design/{cid}.json", f"{out}/design/{cid}.fail"          # 11-01: 미리 계산한 설계 (gen_design_cache.py) 가 있으면 읽는다
        try:
            if os.path.exists(dfail):
                raise RuntimeError(open(dfail).read().strip() + " (미리 계산)")
            if os.path.exists(dj):
                spec = json.load(open(dj))
            else:
                spec = (G.sample_clip(cfg, a.split, i, max_tries=1500, streams=int(cfg.get("checks", {}).get("design_streams", 10))) if cfg.get("design") in ("gen_design_v2", "gen_design_v3")
                        else G.sample_clip(cfg, sp["seed"], i))
        except RuntimeError as e:                                   # 설계 실패: 기록만 하고 넘어간다 (렌더 뒤 검사가 칸별 개수로 잡는다)
            with open(f"{out}/logs/design_fail_{a.split}.txt", "a") as f:
                f.write(f"{cid}\t{e}\n")
            print(f"[w{a.worker}] {cid} 설계 실패 — 건너뜀", flush=True)
            continue
        spec["render"] = {"samples": rc["samples"], "denoise": rc["denoise"], "jpeg_quality": rc["jpeg_quality"], "crf": rc["crf"],
                          "gop": rc["gop"], "engine": "CYCLES", "bpy": S.bpy.app.version_string}
        t0 = time.time()
        t_r, t_l, t_e, (err, phys) = render_clip(S, spec, out, cfg, f"w{a.worker}", check=(n_done < 3))
        n_done += 1
        print(f"[w{a.worker}] {cid} {time.time() - t0:.1f}s (render {t_r:.1f} labels {t_l:.1f} encode {t_e:.1f})"
              + (f" proj_err {err:.3f}px" if err is not None else "")
              + (f" pivot_err {phys['pivot']:.4f}m sink {phys['sink']:.3f}" if phys else "")
              + f" | {n + 1}/{len(todo)} elapsed {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
