#!/usr/bin/env python3
"""BlenderTrack v4c (2026-11-06, 자동 탐색 데이터 후보 5) — 지나가는 (옆으로 미끄러지는) 가림막.

왜: v4 · v4b 의 가림막은 전부 제자리 (trapdoor 만 제자리에서 일어섰다 눕는다). VOBench v3 '지나가는 가림막' 칸과 real 의 손 · 덮개는 가림막이 움직인다.
    음의 손실 λ 를 올리면 VOB 지나가는 칸 (나타남) 이 먼저 무너진다 (자동 탐색 묶음 0 · 1) — 학습 분포에 없는 가림 꼴이다.
무엇: v4b 평면 '판 · 벽' 묶음의 장면 (카메라 · 환경 · target 생김새 · 그 밖 물체) 을 그대로 쓰고, 고정 판을 미끄러지는 판 (gen_design_v2.make_moving) 으로 바꾼다.
      장면 하나에서 두 묶음을 만든다 — 움직이는 target 묶음 {M, B_M} (v4b 속도 단계 하나) · 정지 target 묶음 {S, B_S} (가려지는 구간 가운데에 서 있음).
      판은 묶음마다 다시 설계한다: 가림 구간 [s, e] (샘플) 동안 target 을 덮도록 너비 · 출발 자리를 정하고, 같은 방향 · 반대 · 지나감 중 하나로 움직인다.
      가림 타이밍은 v4b 와 같은 late 정의 (gen_design_v4vc.timing_ok) — 시작 샘플 10–12, 길이 8–10 샘플 (real 가려진 샘플 중앙값 10 · 문맥 안 5).
번호: <split><seed>_<f·4 + j> (j 0 M · 1 B_M · 2 S · 3 B_S), split 'slide', seed 270. 렌더 · 목록은 v4a_static.py 와 같은 꼴 (index_slide).
  design: $BT_PY v4c_slide.py design --src <v4b root> --out <새 root> [--n 400]
  index : $BT_PY v4c_slide.py index --out <새 root> --name blender_track_v4c_slide
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_design_v2 as G2  # noqa: E402
import gen_design_v3 as G3  # noqa: E402
import gen_design_v4vc as G  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "../../../.."))
SPLIT, SEED = "slide", 270
SLIDE = {"slide_speed_frac": [0.06, 0.5], "slide_pass_below_frac": 0.02, "slide_same_p": 0.55,
         "slide_ratio_fast": [1.6, 3.0], "slide_ratio_slow": [0.15, 0.6], "slide_ratio_opp": [0.4, 2.0]}   # configs/v2.yaml 값


def _target_geom(o, T, fps, g):
    pos, _, R = G2.track(o, T, fps, g)
    ctr = G2.center_path(o, pos, R)
    return ctr, G2.extent_pts(o, ctr, R)


def design_one(s0, cfg, rng, static):
    """v4b 판 묶음의 한 단계 설계 s0 → (target 설계, 빈 장면 설계) 또는 (None, 이유)."""
    T, fps, g = int(s0["T"]), float(s0["fps"]), float(s0["g"])
    O, W, V = cfg["objects"], cfg["timing"], cfg["velocity"]
    cam = G3.cam_obj(s0["cam"])
    tgt = copy.deepcopy(s0["objects"][0])
    if static:
        tgt["motion"] = {"kind": "flat_static", "p0": [0.0, 0.0], "yaw0": float(tgt["motion"]["yaw0"])}
    others = s0["objects"][1:]
    ctr, E = _target_geom(tgt, T, fps, g)
    s_in = int(rng.integers(10, 13)); L = int(rng.integers(8, 11))
    fs, fe = 2 * s_in, min(T - 1, 2 * (s_in + L) - 1)
    c2 = copy.deepcopy(cfg); c2["occluders"] = {**cfg["occluders"], **SLIDE}
    r = G2.make_moving(rng, c2, cam, tgt, ctr, E, fs, fe, T, fps)
    if r is None:
        return None, "slide_none"
    pans = r[0]
    pans[0]["th"] = float(G2._u(rng, cfg["occluders"]["board_th"])) if "th" not in pans[0] else pans[0]["th"]
    des = G2.panel_boxes(pans, T, fps)
    if G2._hits_props(ctr, G2.foot_r(tgt), [], pans, T, fps, clear=0.04):
        return None, "hit_panel"
    off = G2.off_frame(cam, E)[::G.STRIDE]
    ok, ev = G.timing_ok(G2.target_hidden(cam, des, E)[::G.STRIDE], off, W, float(V["max_off"]))
    if not ok:
        return None, "late"
    E_in = ctr[:, None] + 0.55 * (E - ctr[:, None])
    if not G.timing_ok(G2.target_hidden(cam, des, E_in)[::G.STRIDE], off, W, float(V["max_off"]))[0]:
        return None, "late_in"
    for d in others:                                                     # 그 밖 물체: 판과 안 부딪히고, 판 · target 에 크게 안 가린다
        pd, _, Rd = G2.track(d, T, fps, g); cd = G2.center_path(d, pd, Rd)
        if G2._hits_props(cd, G2.foot_r(d), [], pans, T, fps):
            return None, "extra_hit_panel"
        if G2.collide(tgt, ctr, d, cd, 0.12):
            return None, "hit_extra"
        ud_, vd_, dd_ = cam.to_px(cd[::G.STRIDE])
        inn = (ud_ > 0) & (ud_ < 256) & (vd_ > 0) & (vd_ < 256) & (dd_ > 0.3)
        hid_d = G2.target_hidden(cam, des + G2.obj_boxes([tgt], T, fps, g, 1.0), G2.extent_pts(d, cd, Rd))[::G.STRIDE]
        if hid_d[inn].max(initial=0.0) > 0.5:
            return None, "hides_extra"
        if G2.target_hidden(cam, G2.obj_boxes([d], T, fps, g, 1.05), E).max() > 0.02:
            return None, "extra_hides"
    px = cam.f * G3.s_eff(tgt) / np.maximum(cam.to_px(ctr[::G.STRIDE])[2], 1e-3)
    s = copy.deepcopy(s0)
    s.update(objects=[tgt] + others, panels=pans, level="S" if static else "M", event_samples=list(ev), target_px_median=float(np.median(px)),
             split=SPLIT, seed=SEED, design="v4c_slide", slide_from=f"{s0['split']}{s0['seed']}_{int(s0['idx']):06d}",
             cell={**s0["cell"], "occ": "slide", "occ_sub": pans[0].get("rel")})
    s.pop("render", None)
    if G2.physics_check(s):
        return None, "phys"
    sB = copy.deepcopy(s); sB.update(objects=list(others), level="B", ghost_target=tgt, event_samples=None)
    return (s, sB), None


def cmd_design(a):
    src, out = os.path.abspath(a.src), os.path.abspath(a.out)
    cfg = G.load_cfg(f"{src}/config_v4b.yaml", None)
    os.makedirs(f"{out}/design", exist_ok=True); os.makedirs(f"{out}/logs", exist_ok=True)
    sp = cfg["splits"]["train_fam"]; K = int(cfg["velocity"]["levels"])
    fams = []
    for f in range(int(sp["candidates"])):
        c0 = f"train_fam{sp['seed']}_{f * (K + 1):06d}"
        if os.path.isfile(f"{src}/design/{c0}.json"):
            s0 = json.load(open(f"{src}/design/{c0}.json"))
            if s0["cell"]["occ"] in ("board", "wall") and s0.get("level") == 0:   # 고정 판 · 벽 장면을 재료로 (가림막만 미끄러지는 판으로 바꾼다)
                fams.append(f)
    rng0 = np.random.default_rng(SEED)
    rng0.shuffle(fams)
    why, n_ok, todo, rows = collections.Counter(), 0, [], []
    for i, f in enumerate(fams[: a.n]):
        k = int(rng0.integers(0, K))
        s_lv = json.load(open(f"{src}/design/train_fam{sp['seed']}_{f * (K + 1) + k:06d}.json"))
        for j, static in ((0, False), (2, True)):
            got = None
            for t in range(30):
                rng = np.random.default_rng([SEED, f, j, t])
                got, e = design_one(s_lv, cfg, rng, static)
                if got:
                    break
                why[e] += 1
            if not got:
                rows.append({"src_family": f, "static": static, "ok": False}); continue
            for jj, s in zip((j, j + 1), got):
                s["idx"] = int(i * 4 + jj); s["family"] = int(i)
                json.dump(s, open(f"{out}/design/{G.clip_id(s)}.json", "w")); todo.append(s["idx"])
            n_ok += 1; rows.append({"src_family": f, "static": static, "ok": True, "event": got[0]["event_samples"], "rel": got[0]["panels"][0].get("rel")})
    c2 = copy.deepcopy(cfg); c2["out"] = out; c2["name"] = "v4c"
    c2["splits"] = {SPLIT: {"seed": SEED, "families": len(fams[: a.n]), "candidates": len(fams[: a.n]), "n": 0}}
    json.dump(c2, open(f"{out}/config_v4c.json", "w"), indent=1)
    import yaml
    yaml.safe_dump(c2, open(f"{out}/config_v4c.yaml", "w"), allow_unicode=True, sort_keys=False)
    open(f"{out}/extra_{SPLIT}.txt", "w").write(" ".join(map(str, todo)))
    rep = {"src": src, "source_families": len(fams[: a.n]), "families_ok": n_ok, "clips": len(todo), "fail": dict(why),
           "rel": dict(collections.Counter(r.get("rel") for r in rows if r.get("ok"))),
           "hidden_len": [int(r["event"][1] - r["event"][0] + 1) for r in rows if r.get("ok")], "rows": rows}
    json.dump(rep, open(f"{out}/v4c_design_report.json", "w"), indent=1)
    hl = np.array(rep["hidden_len"]) if rep["hidden_len"] else np.zeros(1)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("rows", "hidden_len")}, ensure_ascii=False), "| 가려진 샘플 중앙값", float(np.median(hl)))


def cmd_index(a):
    sys.path.insert(0, f"{REPO}/z_training/data")
    import compose_v4vc as CP  # noqa: E402
    from make_v3_meta import vis_of  # noqa: E402
    out = os.path.abspath(a.out)
    cfg = G.load_cfg(f"{out}/config_v4c.yaml", None)
    rep = json.load(open(f"{out}/v4c_design_report.json"))
    paths, fams, summ = [], [], collections.Counter()
    for i in range(rep["source_families"]):
        for j in (0, 2):
            cA, cB = f"{SPLIT}{SEED}_{i * 4 + j:06d}", f"{SPLIT}{SEED}_{i * 4 + j + 1:06d}"
            if not os.path.isfile(f"{out}/design/{cA}.json"):
                continue
            if not all(os.path.isfile(f"{out}/{d}/{c}.{x}") for c in (cA, cB) for d, x in (("clips", "mp4"), ("labels", "npz"))):
                summ["unrendered"] += 1; continue
            ok, ev, hs = CP.render_timing(cfg, f"{out}/labels/{cA}.npz")
            if not ok:
                summ["timing_fail"] += 1; continue
            i0 = len(paths); paths += [f"{out}/clips/{cA}.mp4", f"{out}/clips/{cB}.mp4"]
            fams.append({"plane": "flat", "family": int(i), "occ": "slide", "static": j == 2, "A": [i0], "B": i0 + 1, "hs": [bool(hs)]})
            summ["static" if j == 2 else "moving"] += 1
    N = a.name
    d1, d2 = f"{REPO}/data_csv/{N}/arms", f"{REPO}/data_csv/{N}_negimp"
    os.makedirs(d1, exist_ok=True); os.makedirs(d2, exist_ok=True)
    open(f"{d1}/all_mp4list.csv", "w").writelines(p + "\n" for p in paths)
    s = {"name": N, "videos": len(paths), "families": len(fams), "hidden_splice_pass": int(sum(f["hs"][0] for f in fams)), "counts": dict(summ), "out": out}
    json.dump(s, open(f"{d1}/all_summary.json", "w"), indent=1)
    json.dump({"list": os.path.abspath(f"{d1}/all_mp4list.csv"), "splice": 32, "families": fams, "summary": s}, open(f"{d2}/families.json", "w"))
    print(json.dumps(s, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["design", "index"])
    ap.add_argument("--src", default="/workspace/dataset/training_data/blender_track/v4b_long_hide")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10000, help="쓸 v4b 판 · 벽 묶음 (장면) 수 (기본 전부)")
    ap.add_argument("--name", default="blender_track_v4c_slide")
    a = ap.parse_args()
    {"design": cmd_design, "index": cmd_index}[a.cmd](a)


if __name__ == "__main__":
    main()
