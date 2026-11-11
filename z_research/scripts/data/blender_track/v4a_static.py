#!/usr/bin/env python3
"""BlenderTrack v4a (2026-11-05, 사용자 데이터 계획 A) — v4 묶음 전체에 평면 trapdoor 묶음의 정지 물체 단계를 더한다.
  ⚠️ 정정 (11-05 KST 00:05 사용자 "벽은 넣는 게 좋을 듯 · 서 있는 다른 물체도 괜찮은데 그냥 v4 가림막 상속"): 처음 판은 가림막을 판 · trapdoor 로
     좁혔다 (벽 · 서 있는 물체 묶음 뺌). 기본을 v4 가림막 전부 (판 · 벽 · trapdoor · 서 있는 물체) 로 바꿨다. 좁힌 판은 --keep-occ board,trapdoor
     --name blender_track_v4a_bt 로 남긴다. v4 에 지나가는 (움직이는) 가림막은 없다 — 모든 가림막이 제자리, trapdoor 만 제자리에서 일어섰다 눕는다.

왜
  real 영속 벤치의 절반 (PerceptionTest 50 / 95 장면) 은 '가만히 있는 물체를 덮개가 덮었다 걷는' 꼴인데 v4 에는 정지 물체가 없다.
  사용자 (11-05): "v4 가 v5 보다 좋긴 하더라" → v4 생성기를 기준으로 한다.
무엇 (v4 대비)
  1. 묶음 고르기: v4 학습 목록 (평면 + 빗면 2,250 묶음) 에서 --keep-occ 의 가림막 묶음 (기본 전부 = v4 그대로).
  2. 평면 trapdoor 묶음마다 단계 'S' (정지 물체) 하나를 더한다 — v5mc 의 k_static 과 같은 정의:
     target 을 가려지는 구간 가운데 (원점) 에 처음부터 끝까지 세워 둔다 (운동 flat_static). 판이 t_in 에 일어서 덮고 t_down 에 누워 드러낸다.
     카메라 · 환경 · 판 · 그 밖 물체 · target 생김새는 그 묶음의 속도 단계 0 설계 그대로. 검사는 v4 설계와 같다
     (late 타이밍 · 안쪽 extent 타이밍 · 판과 안 부딪힘 · 그 밖 물체와 안 부딪힘 · 서로 가리지 않음 · 물리 검사). 실패한 묶음은 S 없이 남는다.
  렌더: 평면 trapdoor 묶음은 원래 다섯 영상 (속도 4 + 빈 장면) 까지 이 머신에서 다시 그린다 — 같은 묶음 안 영상을 같은 렌더러로 맞춰
        문맥 · 미래를 이어 붙인 불가능 영상 (사라짐 · 나타남 · 바꿔치기) 의 이음에 렌더러 차이가 단서로 남지 않게 한다. 나머지 묶음은 v4 원본 영상.
번호: 정지 단계 영상 = train_fam70_<500000 + 묶음 번호> (v4 렌더 번호와 안 겹친다).

  design : $BT_PY v4a_static.py design --src <v4 root> --out <새 root>          (CPU, 수 초)
  render : bash run_v4a.sh <새 root>                                             (GPU)
  index  : $BT_PY v4a_static.py index  --src <v4 root> --out <새 root> --name blender_track_v4a
  v4b (긴 가림, 새로 설계한 폴더): design_root --out <root> --cfg <root>/config_v4b.yaml --split train_fam →
           렌더 (run_v4a.sh, CFG · SPLIT) → index_root --out <root> --cfg <root>/config_v4b.yaml --cfg-ramp <root>/config_v4br.yaml --name blender_track_v4b
           → data_csv/<name>/arms/all_mp4list.csv · data_csv/<name>_negimp/families.json · <name>/arms/all_summary.json
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_design_v2 as G2  # noqa: E402
import gen_design_v3 as G3  # noqa: E402
import gen_design_v4vc as G  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "../../../.."))
STATIC_OFF = 500000
KEEP_OCC_DEFAULT = "board,wall,trapdoor,object"   # v4 가림막 전부
SRC_LIST = f"{REPO}/data_csv/blender_track_v4vc/arms/all_mp4list.csv"
SRC_FAM = f"{REPO}/data_csv/blender_track_v4vc_negimp/families.json"


def _cid(p):
    return os.path.basename(p)[:-4]


def _root(p):
    return os.path.dirname(os.path.dirname(p))


def src_families():
    """v4 학습 목록의 묶음 (불가능 짝 정보 families.json 과 같은 순서) + 각 묶음의 설계 cell."""
    d = json.load(open(SRC_FAM))
    paths = [ln.strip() for ln in open(d["list"]) if ln.strip()]
    if os.path.abspath(d["list"]) != os.path.abspath(SRC_LIST):
        raise SystemExit(f"families.json 의 목록 {d['list']} != {SRC_LIST}")
    out = []
    for f in d["families"]:
        a0 = paths[f["A"][0]]
        sp = json.load(open(f"{_root(a0)}/design/{_cid(a0)}.json"))
        out.append({**f, "cell": sp["cell"], "A_path": [paths[i] for i in f["A"]], "B_path": paths[f["B"]]})
    return out, paths


def static_spec(s0, cfg, f):
    """속도 단계 0 설계 s0 → 정지 단계 설계 (실패하면 (None, 이유))."""
    T, fps, g = int(s0["T"]), float(s0["fps"]), float(s0["g"])
    O, W, V = cfg["objects"], cfg["timing"], cfg["velocity"]
    cam = G3.cam_obj(s0["cam"])
    tgt0 = s0["objects"][0]
    if tgt0.get("role") != "target":
        return None, "no_target"
    o = copy.deepcopy(tgt0)
    o["motion"] = {"kind": "flat_static", "p0": [0.0, 0.0], "yaw0": float(tgt0["motion"]["yaw0"])}
    pos, _, R = G2.track(o, T, fps, g)
    ctr = G2.center_path(o, pos, R)
    E = G2.extent_pts(o, ctr, R)
    u_, v_, d_ = cam.to_px(ctr[::G.STRIDE])
    px = cam.f * G3.s_eff(o) / np.maximum(d_, 1e-3)
    if (d_ < 0.3).any() or px.min() < O["min_px"]:
        return None, "px"
    panels = s0["panels"]
    others = s0["objects"][1:]
    if G2._hits_props(ctr, G2.foot_r(o), [], panels, T, fps, clear=0.04):
        return None, "hit_panel"
    des = G2.panel_boxes(panels, T, fps)
    world = []
    for d in others:
        pd, _, Rd = G2.track(d, T, fps, g); cd = G2.center_path(d, pd, Rd)
        if G2.collide(o, ctr, d, cd, 0.12):
            return None, "hit_extra"
        if G2.target_hidden(cam, G2.obj_boxes([d], T, fps, g, 1.05), E).max() > 0.02:
            return None, "extra_hides"
        world.append((d, cd, Rd))
    for i, (d, cd, Rd) in enumerate(world):                              # 정지 target 이 그 밖 물체를 가리지 않는다 (v4 설계의 extra 검사와 같은 기준)
        ud_, vd_, dd_ = cam.to_px(cd[::G.STRIDE])
        inn = (ud_ > 0) & (ud_ < 256) & (vd_ > 0) & (vd_ < 256) & (dd_ > 0.3)
        occ = des + G2.obj_boxes([o] + [w[0] for j, w in enumerate(world) if j != i], T, fps, g, 1.0)
        hid_d = G2.target_hidden(cam, occ, G2.extent_pts(d, cd, Rd))[::G.STRIDE]
        if hid_d[inn].max(initial=0.0) > O["extra_max_hidden"]:
            return None, "hides_extra"
    off = G2.off_frame(cam, E)[::G.STRIDE]
    ok, ev = G.timing_ok(G2.target_hidden(cam, des, E)[::G.STRIDE], off, W, float(V["max_off"]))
    if not ok:
        return None, "late"
    E_in = ctr[:, None] + 0.55 * (E - ctr[:, None])
    if not G.timing_ok(G2.target_hidden(cam, des, E_in)[::G.STRIDE], off, W, float(V["max_off"]))[0]:
        return None, "late_in"
    s = copy.deepcopy(s0)
    s.update(objects=[o] + others, level="S", event_samples=list(ev), target_px_median=float(np.median(px)), idx=STATIC_OFF + int(f),
             static_from="level 0 설계 (v4a_static.py, 2026-11-05)")
    s.pop("render", None)
    if G2.physics_check(s):
        return None, "phys"
    return s, None


def cmd_design(a):
    src, out = os.path.abspath(a.src), os.path.abspath(a.out)
    cfg = G.load_cfg(f"{src}/config.yaml", None)
    fams, _ = src_families()
    tr = [f for f in fams if f["plane"] == "flat" and f["cell"]["occ"] == "trapdoor"]
    os.makedirs(f"{out}/design", exist_ok=True); os.makedirs(f"{out}/logs", exist_ok=True)
    why, todo, rows = collections.Counter(), [], []
    for f in tr:
        cids = [_cid(p) for p in f["A_path"]] + [_cid(f["B_path"])]
        for c in cids:                                                  # 원래 다섯 영상 설계 그대로 (이 머신에서 다시 그린다)
            shutil.copyfile(f"{src}/design/{c}.json", f"{out}/design/{c}.json")
            todo.append(int(c.split("_")[-1]))
        s0 = json.load(open(f"{src}/design/{cids[0]}.json"))
        if s0.get("level") != 0:
            raise SystemExit(f"{cids[0]} 이 단계 0 이 아니다: {s0.get('level')}")
        s, e = static_spec(s0, cfg, s0["family"])
        if s is None:
            why[e] += 1; rows.append({"family": s0["family"], "static": None, "why": e}); continue
        c = G.clip_id(s)
        json.dump(s, open(f"{out}/design/{c}.json", "w"))
        todo.append(int(s["idx"]))
        rows.append({"family": s0["family"], "static": c, "event": s["event_samples"]})
    # 렌더 설정: v4 평면 설정 그대로, out 만 새 자리 · 분할 크기 0 (다시 그릴 번호는 extra 목록으로만)
    c2 = copy.deepcopy(cfg); c2["out"] = out
    for k in c2["splits"]:
        c2["splits"][k]["n"] = 0
    json.dump(c2, open(f"{out}/config.json", "w"), indent=1)
    shutil.copyfile(f"{src}/config.yaml", f"{out}/config_src_v4vc.yaml")
    open(f"{out}/extra_train_fam.txt", "w").write(" ".join(map(str, sorted(set(todo), key=lambda i: (i < STATIC_OFF, i)))))
    rep = {"src": src, "families_trapdoor_flat": len(tr), "static_ok": sum(r["static"] is not None for r in rows), "static_fail": dict(why),
           "render_clips": len(set(todo)), "rows": rows}
    json.dump(rep, open(f"{out}/v4a_design_report.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, indent=1))


def cmd_index(a):
    sys.path.insert(0, HERE)
    import compose_v4vc as CP  # noqa: E402
    from make_v3_meta import vis_of  # noqa: E402
    src, out = os.path.abspath(a.src), os.path.abspath(a.out)
    cfg = G.load_cfg(f"{src}/config.yaml", None)
    fams, _ = src_families()
    paths, fam_out, summ = [], [], collections.Counter()
    miss = collections.Counter()

    def timing(root, c):
        if not (os.path.isfile(f"{root}/clips/{c}.mp4") and os.path.isfile(f"{root}/labels/{c}.npz")):
            return None
        ok, ev, hs = CP.render_timing(cfg, f"{root}/labels/{c}.npz")
        return ok, hs

    for f in fams:
        occ = f["cell"]["occ"]
        if occ not in a.keep_occ.split(","):
            summ[f"drop_{f['plane']}_{occ}"] += 1; continue
        A, B, hs = list(f["A_path"]), f["B_path"], list(f["hs"])
        if f["plane"] == "flat" and occ == "trapdoor":                   # 이 머신에서 다시 그린 다섯 영상 + 정지 단계
            A = [f"{out}/clips/{_cid(p)}.mp4" for p in A]; B = f"{out}/clips/{_cid(B)}.mp4"
            if not all(os.path.isfile(p) for p in A + [B]):
                miss["rerender"] += 1; continue
            for k, p in enumerate(A):                                   # 라벨은 원본과 비트 같아야 한다 (같은 설계) — 타이밍 · 이음 다시 확인
                t = timing(out, _cid(p))
                if t is None or not t[0] or t[1] != hs[k]:
                    raise SystemExit(f"다시 그린 {p}: 타이밍 {t} 이 원본 (이음 {hs[k]}) 과 다르다")
            s0 = json.load(open(f"{out}/design/{_cid(A[0])}.json"))
            sc = f"train_fam{cfg['splits']['train_fam']['seed']}_{STATIC_OFF + int(s0['family']):06d}"
            if os.path.isfile(f"{out}/design/{sc}.json"):
                t = timing(out, sc)
                if t is None:
                    miss["static_unrendered"] += 1
                elif not t[0]:
                    miss["static_timing"] += 1
                else:
                    A.append(f"{out}/clips/{sc}.mp4"); hs.append(bool(t[1])); summ["static"] += 1
                    summ["static_hidden_splice"] += int(bool(t[1]))
        i0 = len(paths)
        paths += A + [B]
        fam_out.append({"plane": f["plane"], "family": int(f["family"]), "occ": occ, "A": list(range(i0, i0 + len(A))), "B": i0 + len(A), "hs": hs})
        summ[f"keep_{f['plane']}_{occ}"] += 1
    for p in paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
    N = a.name
    d1, d2 = f"{REPO}/data_csv/{N}/arms", f"{REPO}/data_csv/{N}_negimp"
    os.makedirs(d1, exist_ok=True); os.makedirs(d2, exist_ok=True)
    open(f"{d1}/all_mp4list.csv", "w").writelines(p + "\n" for p in paths)
    labs = [f"{_root(p)}/labels/{_cid(p)}.npz" for p in paths]
    np.savez(f"{d1}/all.npz", clip=np.array([_cid(p) for p in paths]), path=np.array(paths), vis=np.stack([vis_of(l) for l in labs]))
    hs_all = [h for f in fam_out for h in f["hs"]]
    s = {"name": N, "videos": len(paths), "families": len(fam_out), "stage_videos": len(hs_all), "hidden_splice_pass": int(sum(hs_all)),
         "counts": dict(summ), "missing": dict(miss), "src_list": SRC_LIST, "out": out}
    json.dump(s, open(f"{d1}/all_summary.json", "w"), indent=1)
    json.dump({"list": os.path.abspath(f"{d1}/all_mp4list.csv"), "splice": 32, "families": fam_out, "summary": s}, open(f"{d2}/families.json", "w"))
    print(json.dumps(s, indent=1))


# ----------------------------------------------------------------------------- v4b (2026-11-06, 긴 가림) — 새로 설계 · 렌더한 폴더 하나에서
def cmd_design_root(a):
    """<root>/design 의 평면 trapdoor 묶음 (단계 0 설계) 마다 정지 단계를 더한다. 번호 = <split><seed>_<500000 + 묶음>. 렌더 목록 <root>/extra_static_<split>.txt"""
    root = os.path.abspath(a.out)
    cfg = G.load_cfg(a.cfg, None)
    sp = cfg["splits"][a.split]; K = int(cfg["velocity"]["levels"])
    why, ids, n_tr = collections.Counter(), [], 0
    for f in range(int(sp["candidates"])):
        c0 = f"{a.split}{sp['seed']}_{f * (K + 1):06d}"
        if not os.path.isfile(f"{root}/design/{c0}.json"):
            continue
        s0 = json.load(open(f"{root}/design/{c0}.json"))
        if s0.get("level") != 0 or s0["cell"]["occ"] != "trapdoor" or "ramp" in str(s0["objects"][0]["motion"].get("kind", "")):
            continue
        n_tr += 1
        s, e = static_spec(s0, cfg, f)
        if s is None:
            why[e] += 1; continue
        json.dump(s, open(f"{root}/design/{G.clip_id(s)}.json", "w"))
        ids.append(int(s["idx"]))
    open(f"{root}/extra_static_{a.split}.txt", "w").write(" ".join(map(str, ids)))
    print(json.dumps({"split": a.split, "trapdoor_families": n_tr, "static_ok": len(ids), "static_fail": dict(why)}, ensure_ascii=False))


def cmd_index_root(a):
    """v4b 목록: 평면 (--cfg, --split) + 빗면 (--cfg-ramp, --split-ramp) 의 묶음 중 모든 속도 단계가 렌더 · 타이밍을 통과한 묶음 전부 + 평면 trapdoor 정지 단계.
    v4 의 대조 분할 길이 맞추기 규칙 (train_ind) 은 쓰지 않는다 (v4b 는 대조 분할을 만들지 않는다)."""
    sys.path.insert(0, f"{REPO}/z_training/data")
    import compose_v4vc as CP  # noqa: E402
    from build_v4vc_index import families, ok_fam  # noqa: E402
    from make_v3_meta import vis_of  # noqa: E402
    root = os.path.abspath(a.out)
    paths, fam_out, summ = [], [], collections.Counter()
    for plane, cfgp, split in (("flat", a.cfg, a.split), ("ramp", a.cfg_ramp, a.split_ramp)):
        if not cfgp:
            continue
        cfg = G.load_cfg(cfgp, None)
        fl, K = families(HERE, cfgp, root, split, a.workers)
        for r in fl:
            if not ok_fam(r):
                summ[f"{plane}_not_ok"] += 1; continue
            occ = r["cell"]["occ"]
            if occ not in a.keep_occ.split(","):
                continue
            sl = {s["level"]: s for s in r["slots"]}
            A = [f"{root}/clips/{sl[k]['cid']}.mp4" for k in range(K)]; hs = [bool(sl[k]["hidden_splice"]) for k in range(K)]
            if plane == "flat" and occ == "trapdoor":
                sc = f"{split}{cfg['splits'][split]['seed']}_{STATIC_OFF + int(r['family']):06d}"
                if os.path.isfile(f"{root}/design/{sc}.json"):
                    if not (os.path.isfile(f"{root}/clips/{sc}.mp4") and os.path.isfile(f"{root}/labels/{sc}.npz")):
                        summ["static_unrendered"] += 1
                    else:
                        ok, ev, h = CP.render_timing(cfg, f"{root}/labels/{sc}.npz")
                        if ok:
                            A.append(f"{root}/clips/{sc}.mp4"); hs.append(bool(h)); summ["static"] += 1
                        else:
                            summ["static_timing"] += 1
            B = f"{root}/clips/{sl['B']['cid']}.mp4"
            i0 = len(paths); paths += A + [B]
            fam_out.append({"plane": plane, "family": int(r["family"]), "occ": occ, "A": list(range(i0, i0 + len(A))), "B": i0 + len(A), "hs": hs})
            summ[f"keep_{plane}_{occ}"] += 1
    N = a.name
    d1, d2 = f"{REPO}/data_csv/{N}/arms", f"{REPO}/data_csv/{N}_negimp"
    os.makedirs(d1, exist_ok=True); os.makedirs(d2, exist_ok=True)
    open(f"{d1}/all_mp4list.csv", "w").writelines(p + "\n" for p in paths)
    np.savez(f"{d1}/all.npz", clip=np.array([_cid(p) for p in paths]), path=np.array(paths),
             vis=np.stack([vis_of(f"{_root(p)}/labels/{_cid(p)}.npz") for p in paths]))
    hs_all = [h for f in fam_out for h in f["hs"]]
    s = {"name": N, "videos": len(paths), "families": len(fam_out), "stage_videos": len(hs_all), "hidden_splice_pass": int(sum(hs_all)), "counts": dict(summ), "out": root}
    json.dump(s, open(f"{d1}/all_summary.json", "w"), indent=1)
    json.dump({"list": os.path.abspath(f"{d1}/all_mp4list.csv"), "splice": 32, "families": fam_out, "summary": s}, open(f"{d2}/families.json", "w"))
    print(json.dumps(s, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["design", "index", "design_root", "index_root"])
    ap.add_argument("--src", default="/workspace/dataset/training_data/v4_velocity_control", help="v4 평면 설계 · 영상 root (config.yaml · design/ · clips/ · labels/)")
    ap.add_argument("--out", required=True, help="새 root (다시 그린 평면 trapdoor 묶음 + 정지 단계)")
    ap.add_argument("--name", default="blender_track_v4a")
    ap.add_argument("--keep-occ", default=KEEP_OCC_DEFAULT, help="남길 가림막 (쉼표). 기본 v4 전부. 11-05 첫 판 = board,trapdoor (이름 blender_track_v4a_bt)")
    ap.add_argument("--cfg", default=None, help="design_root · index_root: 평면 설정 (<root>/config_v4b.yaml)")
    ap.add_argument("--split", default="train_fam")
    ap.add_argument("--cfg-ramp", default=None, help="index_root: 빗면 설정 (<root>/config_v4br.yaml), 없으면 평면만")
    ap.add_argument("--split-ramp", default="rtrain_fam")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    {"design": cmd_design, "index": cmd_index, "design_root": cmd_design_root, "index_root": cmd_index_root}[a.cmd](a)


if __name__ == "__main__":
    main()
