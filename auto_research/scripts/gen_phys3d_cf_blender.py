"""[물리] 반사실 쌍 3D 합성 — 고체성 (장애물 있음 / 없음만 다름) · IntPhys 2 Fig. 1 의 2×2 구조 (auto_research, 2026-10-22).

z_training/data 의 생성기는 고치지 않는다 — import 해서 함수만 쓴다 (gen_phys3d_blender = B, gen_phys3d_single_blender 는
sphere 표본 점을 프로세스마다 같게 고친 sample_points 를 B 에 갈아 끼운다).

blender 안에서 돈다 (워커 하나가 장면 목록을 차례로):
  CUDA_VISIBLE_DEVICES=2 /workspace/tools/blender-4.2.3-linux-x64/blender -b -noaudio -P auto_research/scripts/gen_phys3d_cf_blender.py -- \
      --scenes 0,1,2 --split pilot --variants W_pos,N_pos,W_imp,N_imp --out /workspace/dataset/training_data/phys3d_cf_v1 [--engine gpu] [--keep-frames]

장면 하나 = 같은 방 · 같은 물체 (정체) · 같은 출발 · 같은 발사 · 같은 카메라 궤적 · 같은 가림막. 바뀌는 것은 장애물 (벽 또는 떠 있는 상자) 하나:
  변형        장애물 보임  장애물 충돌  궤적            가능?  IntPhys 2 짝
  W_pos       ○            ○            튕김 (bounce)   가능   짝 1 (장애물 문맥)
  W_imp       ○            ×            통과 (pass)     불가능 짝 1
  N_pos       × (없음)     ×            통과            가능   짝 2 (장애물 없는 문맥)
  N_imp       × (안 보임)  ○            튕김            불가능 짝 2
  → 같은 결과 영상 (튕김 · 통과) 이 한 짝에서는 가능, 다른 짝에서는 불가능 — 결과 내용만 보는 surprise 는 50 % 로 상쇄되고 문맥 (장애물) 으로만 가른다.
물리는 두 번만 돈다 (충돌 ○ = 튕김, × = 통과). 렌더는 변형마다. 가림막은 충돌이 없어 물리를 안 바꾼다.
가림막 (장면 절반, seed 로 고정): 사건 순간 (두 물리 판 궤적이 처음 갈리는 장) 둘레 [s, e) 동안 장애물 자리 앞을 덮는 판이 미끄러져 들어왔다 나간다 (IntPhys 2 처럼 사건이 가려짐). 네 변형 모두 같은 판.
카메라: 장면마다 무작위 궤적 (정지 30 % · 옆으로 이동 50 % · 앞뒤 20 %), 바라보는 점은 고정. 네 변형 모두 같은 궤적 ("카메라 고정이면 관성" 지름길 막기).
[T] 추가 (KST 04:10, 리더 지시 — 점 단위 후보): _poses.npz = 장마다 물체 · 장애물 · 가림막 · 카메라 world 행렬 + 카메라 K + 물체 꼭짓점 (local); --depth 면 _depth.npz (T, H, W) float16 m.
라벨: B.simulate_and_label 과 같은 키 · 단위 (pos px · vis_frac · hidden · bbox · pos3d …) + json: 짝 번호 · 변형 · 가능 · 장애물 상자 (x0, x1, z0, z1) ·
  장애물 보임 / 충돌 · 충돌 장 · 갈림 장 · 가림막 · 카메라 궤적 · 정체 번호 · 반사실 축 = "obstacle".
"""
import argparse, json, math, os, shutil, subprocess, sys, time

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

sys.path.insert(0, "/workspace/world_model/vjepa2/z_training/data")
import gen_phys3d_blender as B                                         # noqa: E402
import gen_phys3d_single_blender as S1                                 # noqa: E402,F401  (B.sample_points 갈아 끼움)
from gen_phys3d_blender import PRE, NC, SHAPES, COLORS, HIDDEN, SEEN, OCC_Y, box, make_object, launch, material, rand_rgb, occluder   # noqa: E402

N_FRAMES = 96                       # 12 fps 8 초 → 학습은 3 장마다 (4 fps) 32 장 = 블록 16
VARIANTS = {"W_pos": (True, True), "W_imp": (True, False), "N_pos": (False, False), "N_imp": (False, True)}   # (보임, 충돌)
POSSIBLE = {"W_pos": True, "N_pos": True, "W_imp": False, "N_imp": False}
PAIR = {"W_pos": 1, "W_imp": 1, "N_pos": 2, "N_imp": 2}
SLIDE = 6                           # 가림막이 들어오고 나가는 장 수
MAX_TRIES = 30


def cam_path(seed, cam):
    """같은 장면이면 같은 궤적. 키 = 렌더 앞 PRE 장부터 끝까지 직선 보간, 바라보는 점 (0, 0, 1.7) 고정."""
    rc = np.random.default_rng([seed, 9])
    mode = str(rc.choice(["static", "pan", "dolly"], p=[0.3, 0.5, 0.2]))
    p0 = np.array(cam.location[:])
    if mode == "pan":
        dx = float(rc.choice([-1, 1]) * rc.uniform(0.6, 1.6)); p1 = p0 + np.array([dx, 0, float(rc.uniform(-0.2, 0.2))])
    elif mode == "dolly":
        p1 = p0 + np.array([float(rc.uniform(-0.4, 0.4)), float(rc.uniform(0.6, 1.2)), float(rc.uniform(-0.2, 0.2))])
    else:
        p1 = p0.copy()
    return mode, p0, p1


def apply_cam(cam, p0, p1):
    tgt = Vector((0, 0, 1.7))
    for f, p in ((1, p0), (PRE, p0), (PRE + N_FRAMES - 1, p1)):
        cam.location = Vector(p.tolist())
        cam.rotation_euler = (tgt - cam.location).to_track_quat("-Z", "Y").to_euler()
        cam.keyframe_insert("location", frame=f); cam.keyframe_insert("rotation_euler", frame=f)
    for fc in cam.animation_data.action.fcurves:
        for k in fc.keyframe_points:
            k.interpolation = "LINEAR"


def build(seed, ident, variant, res, engine, frames_dir, screen=None):
    """장면 하나를 짓는다. 모든 난수는 seed 로만 정해져 변형 사이에 같다. variant 는 장애물 보임 · 충돌만 바꾼다."""
    vis, col = VARIANTS[variant]
    rng = np.random.default_rng([seed, 77])
    sc = B.reset()
    B.G = float(np.random.default_rng([seed, 5]).uniform(*B.G_RANGE))
    sc.gravity = (0, 0, -B.G)
    env = B.room(rng)
    env["g"] = B.G
    mode, p0, p1 = cam_path(seed, sc.camera)
    apply_cam(sc.camera, p0, p1)
    env["camera"] = dict(mode=mode, p0=p0.tolist(), p1=p1.tolist())
    side = float(rng.choice([-1, 1]))
    kind = "wall" if rng.random() < 0.6 else "box"                   # 바닥에 선 벽 · 바닥에 놓인 상자 (IntPhys 2: 굴러오는 공을 상자가 막음)
    bounce = float(rng.uniform(0.3, 0.8))
    if kind == "wall":
        ww, wh = 0.3, float(rng.uniform(1.0, 2.6))
    else:
        ww, wh = float(rng.uniform(0.5, 1.0)), float(rng.uniform(0.8, 1.3))
    wx, wz = float(side * rng.uniform(0.0, 1.2)), wh / 2
    m_obs = material("cf_obstacle", rand_rgb(rng), 0.7)
    obs = box("cf_obstacle", (wx, 0, wz), (ww, 1.6, wh), m_obs, bounce=bounce)
    obs_box = [wx - ww / 2, wx + ww / 2, wz - wh / 2, wz + wh / 2]
    x0 = float(-side * 2.6)
    o = make_object(ident, (x0, 0, 0.45), float(rng.uniform(0.4, 0.5)), rng, friction=float(rng.uniform(0.005, 0.03)))
    t_target = float(rng.uniform(20, 44)) / B.FPS                     # 장애물에 닿을 때까지 (초) — 충돌이 문맥 뒤쪽 · 예측 앞쪽에 오게
    dist = max(0.5, abs(wx - x0) - ww / 2 - float(max(o.dimensions)) / 2)
    vx = float(np.clip(1.1 * dist / t_target, 0.5, 3.5))
    throw, vz = False, 0.0
    launch(o, (side * vx, 0, vz))
    # 변형: 보임 · 충돌
    if not col:
        bpy.ops.object.select_all(action="DESELECT"); obs.select_set(True); bpy.context.view_layer.objects.active = obs
        bpy.ops.rigidbody.object_remove()
    if not vis:
        if not col:
            bpy.data.objects.remove(obs, do_unlink=True)             # N_pos: 장애물이 아예 없다
        else:
            obs.hide_render = True; obs.name = "glass_cf_obstacle"     # N_imp: 안 보이는 장애물 (보임 계산의 ray 도 건너뛴다 — 'glass' 이름)
    if screen is not None:                                          # 충돌 둘레 [s, e) 동안 장애물 앞을 덮는 판 (네 변형 같게)
        xc, w, z0, h, s, e = screen
        park = math.copysign(4.2 + w / 2, xc if xc != 0 else 1.0)
        anim = [(PRE + s - SLIDE, park), (PRE + s, xc), (PRE + e - 1, xc), (PRE + e - 1 + SLIDE, park)]
        occluder("cf_screen", xc, w, h, np.random.default_rng([seed, 31]), z0=z0, anim=anim)
    sc.frame_end = PRE + N_FRAMES - 1
    rw = sc.rigidbody_world
    rw.point_cache.frame_start = 1
    rw.point_cache.frame_end = sc.frame_end
    B.setup_render(sc, engine, res, os.path.join(frames_dir, "f"))
    info = dict(kind=kind, side=side, obstacle_box=obs_box, obstacle_bounce=bounce, throw=bool(throw), launch=[side * vx, 0.0, vz])
    return sc, env, o, info


def cam_pos_at(sc, f):
    sc.frame_set(f)
    return np.array(sc.camera.matrix_world.translation[:])


def proj_xz(c, p):
    """카메라 c 에서 점 p 를 가림막 평면 y = OCC_Y 로 투영 → (x, z)"""
    s = (OCC_Y - c[1]) / (p[1] - c[1])
    q = c + s * (p - c)
    return q[0], q[2]


def plan_screen(seed, sc, lab_on, lab_off, obs_box, radius, t_hit):
    """사건 장 t_hit (= 두 궤적이 처음 갈리는 장) 둘레 [s, e) 동안 장애물 + 두 궤적 (튕김 · 통과) 의 투영을 모두 덮는 고정 판. 장면 절반만."""
    rr = np.random.default_rng([seed, 41])
    if rr.random() < 0.5:
        return None
    s = int(max(SLIDE, t_hit - rr.integers(2, 6))); e = int(min(N_FRAMES - SLIDE - 1, t_hit + rr.integers(4, 10)))
    pts = []
    for t in range(s, e):
        c = cam_pos_at(sc, PRE + t)
        for lab in (lab_on, lab_off):
            p = lab["pos3d"][t, 0]
            for dx in (-radius, radius):
                for dz in (-radius, radius):
                    pts.append(proj_xz(c, p + np.array([dx, 0, dz])))
        for x in obs_box[:2]:
            for z in obs_box[2:]:
                pts.append(proj_xz(c, np.array([x, 0.0, z])))
    pts = np.array(pts)
    x0, x1, z0, z1 = pts[:, 0].min() - 0.15, pts[:, 0].max() + 0.15, max(0.0, pts[:, 1].min() - 0.15), pts[:, 1].max() + 0.15
    if x1 - x0 > 3.2 or z1 - z0 > 3.6:
        return None
    return [float((x0 + x1) / 2), float(x1 - x0), float(z0), float(z1 - z0), s, e]


def first_hit(lab, obs_box, radius):
    ev = B.events(lab)
    hits = S1.wall_hits(ev, lab, [obs_box], radius)
    return (min(hits) if hits else None), ev


POSES, DEPTH = True, False


def record_poses(sc, o, n_frames, lab):
    """[T] 점 (입자) 단위 후보용 라벨 (리더 KST 04:08): 장마다 world 행렬 (4×4) — 물체 · 장애물 · 가림막 · 카메라, 카메라 내부 파라미터,
    물체 메시 꼭짓점 (local). 표면 점 궤적 = 꼭짓점 (local) 을 장마다 물체 world 행렬로 옮긴 것. 카메라는 Blender 관례 (−Z 앞, +Y 위).
    검산: 물체 world 행렬의 이동 성분 vs 라벨 pos3d (같은 장) 최대 차 (m)."""
    T = n_frames
    names = {"obj": o.name, "obstacle": None, "screen": None}
    obs = bpy.data.objects.get("cf_obstacle") or bpy.data.objects.get("glass_cf_obstacle")
    scr = bpy.data.objects.get("cf_screen")
    W = {k: np.full((T, 4, 4), np.nan, np.float32) for k in ("obj_world", "obstacle_world", "screen_world", "cam_world")}
    for t in range(T):
        sc.frame_set(PRE + t)
        W["obj_world"][t] = np.array(o.matrix_world)
        W["cam_world"][t] = np.array(sc.camera.matrix_world)
        if obs is not None:
            W["obstacle_world"][t] = np.array(obs.matrix_world)
        if scr is not None:
            W["screen_world"][t] = np.array(scr.matrix_world)
    cd = sc.camera.data
    res = sc.render.resolution_x
    f_px = cd.lens / cd.sensor_width * res                       # sensor_fit AUTO · 정사각 화면 → 가로 기준
    K = np.array([[f_px, 0, res / 2 - cd.shift_x * res], [0, f_px, res / 2 + cd.shift_y * res], [0, 0, 1]], np.float32)
    eo = o.evaluated_get(bpy.context.evaluated_depsgraph_get())
    me = eo.to_mesh()
    verts = np.array([v.co[:] for v in me.vertices], np.float32)
    eo.to_mesh_clear()
    W.update(cam_K=K, cam_lens_mm=np.float32(cd.lens), cam_sensor_mm=np.float32(cd.sensor_width), cam_clip=np.array([cd.clip_start, cd.clip_end], np.float32),
             obj_verts_local=verts, obstacle_present=np.bool_(obs is not None), obstacle_rendered=np.bool_(obs is not None and not obs.hide_render),
             screen_present=np.bool_(scr is not None))
    chk = float(np.abs(W["obj_world"][:, :3, 3] - lab["pos3d"][:, 0]).max())
    return W, chk


def setup_depth(sc, ddir):
    """깊이 (카메라 축 거리, m) 를 장마다 EXR (half, ZIP) 로 — 렌더 PNG 는 그대로 (Composite 는 Render Layers 영상을 그대로 받는다)."""
    sc.view_layers[0].use_pass_z = True
    sc.use_nodes = True
    tree = sc.node_tree
    rl = next((n for n in tree.nodes if n.bl_idname == "CompositorNodeRLayers"), None) or tree.nodes.new("CompositorNodeRLayers")
    comp = next((n for n in tree.nodes if n.bl_idname == "CompositorNodeComposite"), None) or tree.nodes.new("CompositorNodeComposite")
    if not comp.inputs["Image"].is_linked:
        tree.links.new(rl.outputs["Image"], comp.inputs["Image"])
    for n in [n for n in tree.nodes if n.bl_idname == "CompositorNodeOutputFile"]:
        tree.nodes.remove(n)
    fo = tree.nodes.new("CompositorNodeOutputFile")
    fo.base_path = ddir
    fo.format.file_format = "OPEN_EXR"; fo.format.color_depth = "16"; fo.format.exr_codec = "ZIP"; fo.format.color_mode = "RGB"
    fo.file_slots[0].path = "d"
    tree.links.new(rl.outputs["Depth"], fo.inputs[0])


def collect_depth(ddir, out_npz, n_frames):
    """EXR 장들 → (T, H, W) float16 npz (배경은 1000 m 로 자름), EXR 폴더는 지운다."""
    D = []
    for t in range(n_frames):
        pth = os.path.join(ddir, f"d{PRE + t:04d}.exr")
        img = bpy.data.images.load(pth, check_existing=False)
        w, h = img.size
        buf = np.empty(w * h * img.channels, np.float32)
        img.pixels.foreach_get(buf)
        D.append(np.flipud(buf.reshape(h, w, img.channels)[..., 0]))          # Blender 영상은 아래 줄부터
        bpy.data.images.remove(img)
    np.savez_compressed(out_npz, depth=np.clip(np.stack(D), 0, 1000).astype(np.float16))
    shutil.rmtree(ddir, ignore_errors=True)


def copy_ctx(vdir, t_ev):
    """[T] 지적 (KST 03:57): 렌더 잡음 (OptiX 노이즈 제거 · 적응 표본) 으로 같은 장면도 픽셀이 1 계조씩 달라 고정 encoder 토큰이 0.14–0.22 달라진다.
    갈림 전 (t < t_ev) 에는 물리 · 보임이 같으므로 가능 변형의 장을 짝 불가능 변형으로 그대로 복사한다 (W_pos → W_imp · N_pos → N_imp) — 근사가 아니라 정확한 처리.
    렌더 장이 남아 있어야 한다 (--keep-frames). mp4 도 다시 만든다. json 에 ctx_copied_until · ctx_copied_from."""
    for imp, pos in (("W_imp", "W_pos"), ("N_imp", "N_pos")):
        di, dp = os.path.join(vdir, f"{imp}_frames"), os.path.join(vdir, f"{pos}_frames")
        if not (os.path.isdir(di) and os.path.isdir(dp)):
            continue
        for t in range(t_ev):
            shutil.copyfile(os.path.join(dp, f"f{PRE + t:04d}.png"), os.path.join(di, f"f{PRE + t:04d}.png"))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(B.FPS), "-start_number", str(PRE),
                        "-i", os.path.join(di, "f%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                        os.path.join(vdir, f"{imp}.mp4")], check=True)
        jp = os.path.join(vdir, f"{imp}.json")
        m = json.load(open(jp)); m.update(ctx_copied_until=int(t_ev), ctx_copied_from=pos)
        json.dump(m, open(jp, "w"), indent=1)


def scene(seed, k, out_dir, variants, res, engine, keep_frames):
    rid = np.random.default_rng([seed, 1])
    ident = int(rid.integers(0, len(SHAPES) * NC))
    tmp = os.path.join(out_dir, f"_tmp_{k:05d}")
    for attempt in range(MAX_TRIES):
        sd = seed + attempt * 1_000_003
        sc, env, o, info = build(sd, ident, "W_pos", res, engine, tmp)
        radius = float(max(o.dimensions)) / 2
        lab_on = B.simulate_and_label(sc, sc.camera, [o], N_FRAMES, res)
        t_hit, ev_on = first_hit(lab_on, info["obstacle_box"], radius)
        sc, _, o, _ = build(sd, ident, "N_pos", res, engine, tmp)
        lab_off = B.simulate_and_label(sc, sc.camera, [o], N_FRAMES, res)
        why = []
        d = np.linalg.norm(lab_on["pos"][:, 0] - lab_off["pos"][:, 0], axis=-1)
        div = np.where(d > 0.5)[0]                                   # 사건 시각 = 두 물리 판 궤적이 처음 갈리는 장 ([T] 지적: 스치는 첫 충돌을 wall_hits 가 놓쳐
        t_ev = int(div[0]) if len(div) else None                     #  t_hit 가 늦게 잡힌 장면은 t_hit 전에 이미 갈렸다 — 문맥 동일 · 가림막 시각이 깨짐)
        if t_hit is None:
            why.append("no hit")
        if t_ev is None or d.max() < 40:
            why.append(f"no divergence {d.max():.0f}")
        elif not (16 <= t_ev <= 64):
            why.append(f"diverge at {t_ev}")
        elif t_hit is not None and abs(t_hit - t_ev) > 3:
            why.append(f"hit {t_hit} vs diverge {t_ev}")             # 갈림이 장애물 충돌 때문이 아닌 장면은 버린다
        for nm, lab in (("on", lab_on), ("off", lab_off)):
            if (lab["inframe_frac"][:, 0] < 0.5).mean() > 0.1:
                why.append(f"out of frame {nm}")
        if why:
            print(f"[cf] scene {k} attempt {attempt}: {why}", flush=True)
            continue
        screen = plan_screen(sd, sc, lab_on, lab_off, info["obstacle_box"], radius, t_ev)
        res_v = {}
        for v in variants:
            vdir = os.path.join(out_dir, f"scene_{k:05d}")
            os.makedirs(vdir, exist_ok=True)
            stem = os.path.join(vdir, v)
            frames_dir = stem + "_frames"
            sc, env, o, info = build(sd, ident, v, res, engine, frames_dir, screen=screen)
            lab = B.simulate_and_label(sc, sc.camera, [o], N_FRAMES, res)
            pose_chk = None
            if POSES:
                Pz, pose_chk = record_poses(sc, o, N_FRAMES, lab)
                np.savez_compressed(stem + "_poses.npz", **Pz)
            ratio = lab["vis_frac"] / np.maximum(lab["inframe_frac"], 1e-6)
            present = np.ones((N_FRAMES, 1), bool)
            bb = lab["bbox"]
            ev = B.events(lab)
            impact = np.zeros((N_FRAMES, 1), np.float32)
            for e_ in ev:
                if e_["kind"] == "impact":
                    impact[e_["frame"], 0] = max(impact[e_["frame"], 0], e_["dv"])
            lab.update(present=present, hidden=present & (lab["inframe_frac"] > 0.25) & (ratio < HIDDEN),
                       seen=np.maximum.accumulate(lab["vis_frac"] > SEEN, axis=0), impact=impact,
                       identity=np.array([ident], np.int16), shape_id=np.array([ident // NC], np.int16), color_id=np.array([ident % NC], np.int16),
                       size=np.array([o.dimensions.length / math.sqrt(3)], np.float32),
                       area=((bb[0, :, 2] - bb[0, :, 0]) * (bb[0, :, 3] - bb[0, :, 1])).astype(np.float32))
            ref = lab_on if VARIANTS[v][1] else lab_off
            dev_px = float(np.abs(lab["pos"][:, 0] - ref["pos"][:, 0]).max())    # 변형 라벨 궤적이 물리 두 판 중 맞는 쪽과 같은가 (검산)
            np.savez_compressed(stem + "_states.npz", **lab)
            os.makedirs(frames_dir, exist_ok=True)
            sc.frame_start = PRE
            sc.frame_end = PRE + N_FRAMES - 1
            if os.environ.get("CF_NORENDER") != "1":
                if DEPTH:
                    setup_depth(sc, stem + "_depth")
                bpy.ops.render.render(animation=True)
                if DEPTH:
                    collect_depth(stem + "_depth", stem + "_depth.npz", N_FRAMES)
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(B.FPS), "-start_number", str(PRE),
                                "-i", os.path.join(frames_dir, "f%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                                stem + ".mp4"], check=True)
            if not keep_frames and os.path.isdir(frames_dir):
                shutil.rmtree(frames_dir)
            meta = dict(scene=k, seed=sd, base_seed=seed, attempt=attempt, variant=v, cf_axis="obstacle", possible=POSSIBLE[v], pair=PAIR[v],
                        obstacle_visible=VARIANTS[v][0], obstacle_collides=VARIANTS[v][1], obstacle=info, t_hit=int(t_hit),
                        t_event=t_ev, t_diverge=t_ev, screen=screen, env=env, fps=B.FPS, n_frames=N_FRAMES, res=res, engine=engine,
                        identity=ident, shape=SHAPES[ident // NC], color=COLORS[ident % NC][0], events=ev, label_traj_check_px=dev_px,
                        hidden_frac=float(lab["hidden"][:, 0].mean()), poses=bool(POSES), pose_check_m=pose_chk, depth=bool(DEPTH))
            with open(stem + ".json", "w") as f:
                json.dump(meta, f, indent=1)
            res_v[v] = dict(ok=True, traj_check_px=dev_px)
        if keep_frames and os.environ.get("CF_NORENDER") != "1":
            copy_ctx(os.path.join(out_dir, f"scene_{k:05d}"), t_ev)
        shutil.rmtree(tmp, ignore_errors=True)
        return dict(ok=True, scene=k, attempt=attempt, t_hit=int(t_hit), t_event=t_ev, screen=screen is not None, variants=res_v)
    shutil.rmtree(tmp, ignore_errors=True)
    return dict(ok=False, scene=k, why=why)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True, help="장면 번호 쉼표 목록")
    ap.add_argument("--split", default="pilot")
    ap.add_argument("--seed-base", type=int, default=7_000_000)
    ap.add_argument("--variants", default="W_pos,N_pos,W_imp,N_imp")
    ap.add_argument("--out", default="/workspace/dataset/training_data/phys3d_cf_v1")
    ap.add_argument("--engine", default="gpu")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument("--no-poses", action="store_true", help="[T] 장마다 world 행렬 · 카메라 라벨 (_poses.npz) 을 끈다")
    ap.add_argument("--depth", action="store_true", help="[T] 깊이 패스 (_depth.npz, float16)")
    a = ap.parse_args(argv)
    global POSES, DEPTH
    POSES, DEPTH = not a.no_poses, a.depth
    out_dir = os.path.join(a.out, a.split)
    os.makedirs(out_dir, exist_ok=True)
    variants = a.variants.split(",")
    for k in [int(x) for x in a.scenes.split(",") if x != ""]:
        if all(os.path.exists(os.path.join(out_dir, f"scene_{k:05d}", f"{v}.json")) for v in variants):
            continue                                                  # 이어 하기
        t0 = time.time()
        r = scene(a.seed_base + k * 101, k, out_dir, variants, a.res, a.engine, a.keep_frames)
        r["sec"] = round(time.time() - t0, 1)
        with open(os.path.join(out_dir, "results.jsonl"), "a") as f:
            f.write(json.dumps(r) + "\n")
        print("[cf]", json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
