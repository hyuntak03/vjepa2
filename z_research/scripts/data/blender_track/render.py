#!/usr/bin/env python3
"""BlenderTrack v0 — 렌더러 (conda env `blender`, bpy 5.0.1, Cycles OptiX). design.py 의 설계를 그대로 그린다 (2026-10-31).

한 프로세스 = GPU 하나 (CUDA_VISIBLE_DEVICES) 위의 작업자 하나. 장면 바탕 (바닥 · 하늘 · 해 · 카메라 · 메시 모음) 은
프로세스에서 한 번 짓고, clip 마다 물체 · 판만 만들고 지운다 (persistent data → 프레임마다 장면 재구축 없음).

clip 출력 <out>/clips/<cid>/
  rgb/0000.png …            256×256 RGB, 30 fps, 180 장
  labels.npz                 K = target + 방해 물체 (잡동사니 제외), L = 64 (라벨 1 칸 = 화면 4×4 px)
    pos (T,K,3) 바닥면 중심 m · yaw (T,K) · center_px (T,K,2) 물체 상자 중심의 투영 (u 오른쪽, v 아래쪽)
    bbox_px (T,K,4) 꼭짓점 투영 상자 (u0,v0,u1,v1, 자르지 않음) · depth (T,K)
    amodal (T,K,L,L/8) packbits — 가려져도 물체가 있는 칸 (물체만 따로 ray cast)
    visible (T,K,L,L/8) packbits — 카메라에서 처음 맞는 것이 그 물체인 칸
    amodal_n · visible_n (T,K) 칸 수 · panel_ang (T,P) · roles (K,)
  spec.json                  설계 그대로 + 렌더 설정
  (labels.npz 가 마지막에 원자적으로 쓰이므로 그것이 완료 표시다)

  B=/data/hyuntak/anaconda3/envs/blender/bin/python
  CUDA_VISIBLE_DEVICES=0 $B render.py --out <dir> --seed 0 --start 0 --end 8 --worker 0 --nworkers 1 --samples 16
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import design as D  # noqa: E402

L = 64


def _nodes(mat):
    try:
        mat.use_nodes = True
    except Exception:
        pass
    return mat.node_tree


def _ramp(nt, c1, c2):
    r = nt.nodes.new("ShaderNodeValToRGB")
    r.color_ramp.elements[0].color = (*c1, 1)
    r.color_ramp.elements[1].color = (*c2, 1)
    return r


class Scene:
    def __init__(self, a):
        import bpy
        self.bpy = bpy
        bpy.ops.wm.read_factory_settings(use_empty=True)
        scn = self.scn = bpy.context.scene
        scn.render.engine = "CYCLES"
        scn.cycles.samples = a.samples
        scn.cycles.use_denoising = True
        for k, v in (("denoiser", "OPENIMAGEDENOISE"), ("denoising_use_gpu", True), ("max_bounces", 4),
                     ("diffuse_bounces", 2), ("glossy_bounces", 2), ("transmission_bounces", 0),
                     ("transparent_max_bounces", 2), ("caustics_reflective", False), ("caustics_refractive", False)):
            try:
                setattr(scn.cycles, k, v)
            except Exception as e:
                print("cycles 설정 무시", k, e)
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for typ in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = typ
                prefs.get_devices()
                if any(d.type == typ for d in prefs.devices):
                    break
            except Exception:
                continue
        for d in prefs.devices:
            d.use = d.type == prefs.compute_device_type
        scn.cycles.device = "GPU"
        print("devices", [(d.name, d.type) for d in prefs.devices if d.use], flush=True)
        scn.render.use_persistent_data = True
        if a.threads:                                          # 작업자 여럿이 코어를 나눠 쓰게 (안 하면 프로세스마다 전 코어)
            scn.render.threads_mode, scn.render.threads = "FIXED", a.threads
        scn.render.resolution_x = scn.render.resolution_y = D.SIZE
        scn.render.resolution_percentage = 100
        scn.render.image_settings.file_format = "PNG"
        scn.render.image_settings.color_mode = "RGB"
        scn.render.image_settings.compression = 15
        try:
            scn.view_settings.view_transform = "Standard"     # AgX 는 중간톤을 들어 올린다 (v5 실측)
        except Exception:
            pass

        # 카메라
        cd = bpy.data.cameras.new("cam")
        cd.sensor_width, cd.sensor_fit, cd.clip_start, cd.clip_end = D.SENSOR, "AUTO", 0.1, 600
        self.cam = bpy.data.objects.new("cam", cd)
        scn.collection.objects.link(self.cam)
        scn.camera = self.cam
        # 해
        sun = bpy.data.lights.new("sun", "SUN")
        sun.angle = math.radians(2.0)
        self.sun = bpy.data.objects.new("sun", sun)
        scn.collection.objects.link(self.sun)
        # 하늘 (높이 램프)
        scn.world = bpy.data.worlds.new("w")
        wnt = _nodes(scn.world)
        bg = wnt.nodes["Background"]
        tc = wnt.nodes.new("ShaderNodeTexCoord")
        sep = wnt.nodes.new("ShaderNodeSeparateXYZ")
        mr = wnt.nodes.new("ShaderNodeMapRange")
        mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = -0.1, 0.9
        self.sky = _ramp(wnt, (0.8, 0.8, 0.8), (0.2, 0.3, 0.6))
        wnt.links.new(tc.outputs["Generated"], sep.inputs["Vector"])
        wnt.links.new(sep.outputs["Z"], mr.inputs["Value"])
        wnt.links.new(mr.outputs["Result"], self.sky.inputs["Fac"])
        wnt.links.new(self.sky.outputs["Color"], bg.inputs["Color"])
        self.sky_bg = bg
        # 바닥 (질감 4 종 재질을 미리 짓고 clip 마다 고른다)
        mesh = bpy.data.meshes.new("floor")
        S = 300.0
        mesh.from_pydata([(-S, -S, 0), (S, -S, 0), (S, S, 0), (-S, S, 0)], [], [(0, 1, 2, 3)])
        self.floor = bpy.data.objects.new("floor", mesh)
        scn.collection.objects.link(self.floor)
        self.floor_mats = {k: self._tex_mat(f"floor_{k}", k) for k in D.FLOORS}
        mesh.materials.append(self.floor_mats["checker"])
        # 뒷벽 (바닥 원점 상자)
        self.cube = self._cube_mesh("unitcube")
        self.wall = bpy.data.objects.new("wall", self.cube)
        scn.collection.objects.link(self.wall)
        self.wall_mats = {k: self._tex_mat(f"wall_{k}", k) for k in ("brick", "noise", "plain")}
        self.wall.material_slots[0].link = "OBJECT"
        # 물체 메시 모음 (DIMS 상자에 정확히 맞춤, 원점 = 바닥면 중심)
        self.lib = {name: self._shape_mesh(name) for name in D.SHAPES}
        from mathutils.bvhtree import BVHTree
        self.bvh = {n: BVHTree.FromPolygons([tuple(v.co) for v in m.vertices], [tuple(p.vertices) for p in m.polygons])
                    for n, m in self.lib.items()}
        self.verts = {n: np.array([tuple(v.co) for v in m.vertices]) for n, m in self.lib.items()}
        # 재질 풀 (clip 마다 값만 바꾼다)
        self.obj_mats = [self._obj_mat(f"obj{i}") for i in range(10)]
        self.pan_mats = [self._panel_mat(f"pan{i}") for i in range(6)]
        self.live = []

    # ------------------------------------------------------------ 재질 · 메시
    def _tex_mat(self, name, kind):
        bpy = self.bpy
        mat = bpy.data.materials.new(name)
        nt = _nodes(mat)
        b = nt.nodes["Principled BSDF"]
        b.inputs["Roughness"].default_value = 0.9
        tc = nt.nodes.new("ShaderNodeTexCoord")
        mp = nt.nodes.new("ShaderNodeMapping")
        nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
        mat["kind"] = kind
        if kind == "checker":
            tx = nt.nodes.new("ShaderNodeTexChecker")
            nt.links.new(mp.outputs["Vector"], tx.inputs["Vector"])
            nt.links.new(tx.outputs["Color"], b.inputs["Base Color"])
        elif kind == "brick":
            tx = nt.nodes.new("ShaderNodeTexBrick")
            nt.links.new(mp.outputs["Vector"], tx.inputs["Vector"])
            nt.links.new(tx.outputs["Color"], b.inputs["Base Color"])
        elif kind in ("noise", "wave"):
            tx = nt.nodes.new("ShaderNodeTexNoise" if kind == "noise" else "ShaderNodeTexWave")
            if kind == "noise":
                tx.inputs["Detail"].default_value = 6.0
            else:
                tx.inputs["Distortion"].default_value = 4.0
            r = _ramp(nt, (0.2, 0.2, 0.2), (0.5, 0.5, 0.5))
            nt.links.new(mp.outputs["Vector"], tx.inputs["Vector"])
            nt.links.new(tx.outputs["Fac"] if "Fac" in tx.outputs else tx.outputs[0], r.inputs["Fac"])
            nt.links.new(r.outputs["Color"], b.inputs["Base Color"])
        else:                                                  # plain
            tx = None
        return mat

    def _set_tex(self, mat, c1, c2, scale):
        nt = mat.node_tree
        k = mat["kind"]
        mp = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeMapping")
        mp.inputs["Scale"].default_value = (scale, scale, scale)
        b = nt.nodes["Principled BSDF"]
        if k == "checker":
            tx = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeTexChecker")
            tx.inputs["Color1"].default_value, tx.inputs["Color2"].default_value = (*c1, 1), (*c2, 1)
            tx.inputs["Scale"].default_value = 1.0
        elif k == "brick":
            tx = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeTexBrick")
            tx.inputs["Color1"].default_value, tx.inputs["Color2"].default_value = (*c1, 1), (*c2, 1)
            tx.inputs["Mortar"].default_value = tuple(0.6 * x for x in c1) + (1,)
            tx.inputs["Scale"].default_value = 1.0
        elif k in ("noise", "wave"):
            r = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeValToRGB")
            r.color_ramp.elements[0].color, r.color_ramp.elements[1].color = (*c1, 1), (*c2, 1)
        else:
            b.inputs["Base Color"].default_value = (*c1, 1)

    def _obj_mat(self, name):
        mat = self.bpy.data.materials.new(name)
        _nodes(mat)
        return mat

    def _panel_mat(self, name):
        mat = self.bpy.data.materials.new(name)
        nt = _nodes(mat)
        b = nt.nodes["Principled BSDF"]
        tc = nt.nodes.new("ShaderNodeTexCoord")
        mp = nt.nodes.new("ShaderNodeMapping")
        nz = nt.nodes.new("ShaderNodeTexNoise")
        wv = nt.nodes.new("ShaderNodeTexWave")
        wv.inputs["Distortion"].default_value = 6.0
        r = _ramp(nt, (0.3, 0.3, 0.3), (0.5, 0.5, 0.5))
        nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
        nt.links.new(mp.outputs["Vector"], nz.inputs["Vector"])
        nt.links.new(mp.outputs["Vector"], wv.inputs["Vector"])
        nt.links.new(r.outputs["Color"], b.inputs["Base Color"])
        return mat

    def _cube_mesh(self, name):
        m = self.bpy.data.meshes.new(name)
        v = [(x, y, z) for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (0.0, 1.0)]
        f = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        m.from_pydata(v, [], f)
        m.materials.append(self.bpy.data.materials.new(name + "_ph"))
        m.use_fake_user = True
        return m

    def _shape_mesh(self, name):
        bpy = self.bpy
        ops = bpy.ops.mesh
        if name == "sphere":
            ops.primitive_uv_sphere_add(radius=0.5, segments=48, ring_count=24)
        elif name == "cube":
            ops.primitive_cube_add(size=1.0)
        elif name == "cone":
            ops.primitive_cone_add(radius1=0.5, depth=1.0, vertices=48)
        elif name == "cylinder":
            ops.primitive_cylinder_add(radius=0.5, depth=1.0, vertices=48)
        elif name == "torus":
            ops.primitive_torus_add(major_radius=0.40, minor_radius=0.15, major_segments=48, minor_segments=18)
            bpy.context.active_object.rotation_euler = (math.radians(90), 0, 0)
            bpy.ops.object.transform_apply(rotation=True)
        elif name == "pyramid":
            ops.primitive_cone_add(radius1=0.62, depth=0.95, vertices=4)
        elif name == "capsule":
            ops.primitive_uv_sphere_add(radius=0.5, segments=40, ring_count=20)
            bpy.context.active_object.scale = (0.55, 0.55, 1.05)
            bpy.ops.object.transform_apply(scale=True)
        elif name == "diamond":
            ops.primitive_cone_add(radius1=0.55, depth=1.1, vertices=8)
            o1 = bpy.context.active_object
            ops.primitive_cone_add(radius1=0.55, depth=1.1, vertices=8)
            o2 = bpy.context.active_object
            o2.rotation_euler = (math.radians(180), 0, 0)
            bpy.ops.object.transform_apply(rotation=True)
            o2.location = (0, 0, -1.1)
            bpy.ops.object.transform_apply(location=True)
            o1.select_set(True); o2.select_set(True)
            bpy.context.view_layer.objects.active = o1
            bpy.ops.object.join()
        o = bpy.context.active_object
        m = o.data
        v = np.array([tuple(x.co) for x in m.vertices])
        lo, hi = v.min(0), v.max(0)
        tgt = np.array(D.DIMS[name], float)
        v = (v - (lo + hi) / 2) / (hi - lo) * tgt
        v[:, 2] += tgt[2] / 2
        m.vertices.foreach_set("co", v.astype(np.float32).ravel())
        m.update()
        smooth = name in ("sphere", "capsule", "torus", "cylinder", "cone")
        m.polygons.foreach_set("use_smooth", [smooth] * len(m.polygons))
        m.materials.append(bpy.data.materials.new(f"{name}_ph"))
        m.name = f"lib_{name}"
        m.use_fake_user = True
        bpy.data.objects.remove(o, do_unlink=True)
        return m

    # ------------------------------------------------------------ clip
    def setup(self, spec):
        bpy, scn = self.bpy, self.scn
        c, e = spec["cam"], spec["env"]
        self.cam.data.lens = c["lens"]
        self.cam.location = (0.0, 0.0, c["h"])
        self.cam.rotation_euler = (math.radians(90.0 + c["pitch"]), 0.0, 0.0)
        self.camN = D.Cam(c["h"], c["pitch"], c["lens"])
        self.sun.data.energy = e["sun_energy"]
        self.sun.rotation_euler = (math.radians(90 - e["sun_elev"]), 0.0, math.radians(e["sun_azim"]))
        self.sky.color_ramp.elements[0].color = (*e["sky_hor"], 1)
        self.sky.color_ramp.elements[1].color = (*e["sky_top"], 1)
        self.sky_bg.inputs["Strength"].default_value = e["sky_strength"]
        scn.view_settings.exposure = e["exposure"]
        fm = self.floor_mats[e["floor"]]
        self._set_tex(fm, e["floor_c1"], e["floor_c2"], e["floor_scale"])
        self.floor.data.materials[0] = fm
        self.wall.hide_render = not e["wall"]
        self.wall.location = (0.0, e["wall_y"], 0.0)
        self.wall.scale = (400.0, 0.3, e["wall_h"])
        wm = self.wall_mats[e["wall_tex"]]
        self._set_tex(wm, e["wall_c1"], e["wall_c2"], 1.0)
        self.wall.material_slots[0].material = wm

        self.objs = []
        for i, o in enumerate(spec["objects"]):
            ob = bpy.data.objects.new(f"o{i}_{o['role']}", self.lib[o["shape"]])
            scn.collection.objects.link(ob)
            ob.material_slots[0].link = "OBJECT"
            mat = self.obj_mats[i]
            b = mat.node_tree.nodes["Principled BSDF"]
            b.inputs["Base Color"].default_value = (*o["color"], 1)
            b.inputs["Roughness"].default_value = o["rough"]
            b.inputs["Metallic"].default_value = o["metal"]
            ob.material_slots[0].material = mat
            ob.scale = (o["s"],) * 3
            self.objs.append(ob)
        self.pans = []
        for j, p in enumerate(spec["panels"]):
            ob = bpy.data.objects.new(f"p{j}_{p['role']}", self.cube)
            scn.collection.objects.link(ob)
            ob.material_slots[0].link = "OBJECT"
            mat = self.pan_mats[j]
            nt = mat.node_tree
            r = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeValToRGB")
            col = p["color"]
            r.color_ramp.elements[0].color = (*[0.7 * x for x in col], 1)
            r.color_ramp.elements[1].color = (*col, 1)
            nz = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeTexNoise")
            wv = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeTexWave")
            for lk in list(nt.links):
                if lk.to_node == r:
                    nt.links.remove(lk)
            if p["tex"] == "noise":
                nt.links.new(nz.outputs["Fac"], r.inputs["Fac"])
            elif p["tex"] == "wave":
                nt.links.new(wv.outputs["Fac"] if "Fac" in wv.outputs else wv.outputs[1], r.inputs["Fac"])
            else:
                r.inputs["Fac"].default_value = 1.0
            mp = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeMapping")
            mp.inputs["Scale"].default_value = (p["tex_scale"],) * 3
            ob.material_slots[0].material = mat
            ob.location = (p["x"], p["y"], 0.0)
            ob.scale = (p["w"], p["th"], p["h"])
            self.pans.append(ob)
        self.tracks, self.pan_ang = D.all_tracks(spec)

    def teardown(self):
        for ob in self.objs + self.pans:
            self.bpy.data.objects.remove(ob, do_unlink=True)
        self.objs, self.pans = [], []

    def set_frame(self, f):
        for ob, (pos, yaw) in zip(self.objs, self.tracks):
            ob.location = tuple(pos[f])
            ob.rotation_euler = (0.0, 0.0, float(yaw[f]))
        for ob, ang in zip(self.pans, self.pan_ang):
            ob.rotation_euler = (float(-ang[f]), 0.0, 0.0)     # 설계 −π/2 (카메라 쪽 누움) = Blender x 축 +π/2
        self.bpy.context.view_layer.update()

    def labels(self, f, K, spec):
        """target · 방해 물체의 투영 상자와 64×64 amodal / visible 칸."""
        from mathutils import Vector
        dg = self.bpy.context.evaluated_depsgraph_get()
        cam = self.camN
        C = Vector(tuple(cam.C))
        out = []
        for k in range(K):
            ob, o = self.objs[k], spec["objects"][k]
            pos, yaw = self.tracks[k][0][f], self.tracks[k][1][f]
            cz, sz = math.cos(yaw), math.sin(yaw)
            Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            vw = (self.verts[o["shape"]] * o["s"]) @ Rz.T + pos
            u, v, d = cam.to_px(vw)
            ctr = pos + np.array([0, 0, 0.5 * o["s"] * D.DIMS[o["shape"]][2]])
            cu, cv, cd = cam.to_px(ctr[None])
            am = np.zeros((L, L), bool)
            vi = np.zeros((L, L), bool)
            if (d > 0.2).all():
                s = D.SIZE / L
                i0, i1 = int(max(0, math.floor(v.min() / s) - 1)), int(min(L - 1, math.ceil(v.max() / s) + 1))
                j0, j1 = int(max(0, math.floor(u.min() / s) - 1)), int(min(L - 1, math.ceil(u.max() / s) + 1))
                if i1 >= i0 and j1 >= j0:
                    ii, jj = np.mgrid[i0:i1 + 1, j0:j1 + 1]
                    dirs = cam.ray((jj.ravel() + 0.5) * s, (ii.ravel() + 0.5) * s)
                    Minv = np.linalg.inv(np.c_[np.r_[Rz * o["s"], [[0, 0, 0]]], np.r_[pos, 1]])
                    ol = Vector(tuple((Minv @ np.r_[cam.C, 1])[:3]))
                    dl = dirs @ Minv[:3, :3].T
                    bvh = self.bvh[o["shape"]]
                    name = ob.name
                    for n, (a, b) in enumerate(zip(ii.ravel(), jj.ravel())):
                        if bvh.ray_cast(ol, Vector(tuple(dl[n])))[0] is not None:
                            am[a, b] = True
                            hit, _, _, _, hob, _ = self.scn.ray_cast(dg, C, Vector(tuple(dirs[n] / np.linalg.norm(dirs[n]))))
                            if hit and hob is not None and hob.name == name:
                                vi[a, b] = True
            out.append(dict(pos=pos, yaw=yaw, center=(float(cu[0]), float(cv[0])), depth=float(cd[0]),
                            bbox=(float(u.min()), float(v.min()), float(u.max()), float(v.max())), am=am, vi=vi))
        return out

    def check_projection(self, spec, f=0):
        """numpy 카메라 = Blender 카메라 인지 (target 중심, 픽셀 오차)."""
        from bpy_extras.object_utils import world_to_camera_view
        from mathutils import Vector
        o = spec["objects"][0]
        pos = self.tracks[0][0][f] + np.array([0, 0, 0.5 * o["s"] * D.DIMS[o["shape"]][2]])
        co = world_to_camera_view(self.scn, self.cam, Vector(tuple(pos)))
        u, v, _ = self.camN.to_px(pos[None])
        return float(abs(co.x * D.SIZE - u[0]) + abs((1 - co.y) * D.SIZE - v[0]))


def render_clip(S, spec, cdir, check=False):
    bpy = S.bpy
    os.makedirs(f"{cdir}/rgb", exist_ok=True)
    S.setup(spec)
    K = sum(o["role"] in ("target", "distractor") for o in spec["objects"])
    T = spec["T"]
    acc = {k: [] for k in ("pos", "yaw", "center", "depth", "bbox", "am", "vi")}
    err = None
    t_r = t_l = 0.0
    for f in range(T):
        S.set_frame(f)
        if check and f == 0:
            err = S.check_projection(spec)
        t0 = time.time()
        lab = S.labels(f, K, spec)
        t1 = time.time()
        for k in acc:
            acc[k].append([x[k] for x in lab])
        S.scn.render.filepath = f"{cdir}/rgb/{f:04d}.png"
        bpy.ops.render.render(write_still=True)
        t_l += t1 - t0
        t_r += time.time() - t1
    S.teardown()
    am, vi = np.array(acc["am"]), np.array(acc["vi"])
    np.savez_compressed(f"{cdir}/labels_tmp.npz",
                        pos=np.array(acc["pos"], np.float32), yaw=np.array(acc["yaw"], np.float32),
                        center_px=np.array(acc["center"], np.float32), depth=np.array(acc["depth"], np.float32),
                        bbox_px=np.array(acc["bbox"], np.float32),
                        amodal=np.packbits(am, axis=-1), visible=np.packbits(vi, axis=-1),
                        amodal_n=am.sum((-1, -2)).astype(np.int16), visible_n=vi.sum((-1, -2)).astype(np.int16),
                        panel_ang=np.array(S.pan_ang, np.float32).T.reshape(T, -1),
                        roles=np.array([o["role"] for o in spec["objects"][:K]]))
    json.dump(spec, open(f"{cdir}/spec.json", "w"))
    os.replace(f"{cdir}/labels_tmp.npz", f"{cdir}/labels.npz")
    return t_r, t_l, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:])
    S = Scene(a)
    todo = [i for i in range(a.start, a.end) if i % a.nworkers == a.worker]
    t_all = time.time()
    for n, i in enumerate(todo):
        cid = f"s{a.seed}_{i:06d}"
        cdir = f"{a.out}/clips/{cid}"
        if os.path.exists(f"{cdir}/labels.npz"):
            continue
        spec = D.sample_clip(a.seed, i)
        spec["render"] = {"samples": a.samples, "engine": "CYCLES", "bpy": S.bpy.app.version_string}
        t0 = time.time()
        t_r, t_l, err = render_clip(S, spec, cdir, check=(n == 0))
        print(f"[w{a.worker}] {cid} {time.time() - t0:.1f}s (render {t_r:.1f} labels {t_l:.1f})"
              + (f" proj_err {err:.3f}px" if err is not None else "")
              + f" | {n + 1}/{len(todo)} elapsed {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
