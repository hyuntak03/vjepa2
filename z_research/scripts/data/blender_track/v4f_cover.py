#!/usr/bin/env python3
"""BlenderTrack v4f (2026-11-07) — 덮어 미는 상자 (가려진 동안 물체가 옮겨지는 장면).

왜 [측정, z_analysis/results/slot_diag_20261107 · realbench 사건별]: 음의 손실 + 가속 데이터 판 (real95 74.7) 이 실제 영상 벤치에서 틀리는 쌍은
    '가려진 동안 물체가 옮겨지는' 장면 (PerceptionTest moved, 사라짐 쌍 46 % 틀림) 과 '덮개 · 그릇' 장면 (RubricOffice containment 40 %) 에 몰린다.
    같은 출처 · 같은 가리개에서 그대로 두면 (nothing) 8 % 만 틀린다. 학습 데이터 (v4 계열) 에는 가림막이 물체를 옮기는 장면이 없다
    (데이터 설계 원칙 1.5 부 9 줄 '평가 사건을 덮기 — 그릇 덮기 · 손으로 옮김' ❌).
무엇: v4b 평면 '판 · 벽' 장면 (카메라 · 환경 · target 생김새 · 그 밖 물체) 을 그대로 쓰고, 가림막을 '앞이 트인 상자' (hood) 로 바꾼다.
    상자는 바닥 위를 미끄러져 와 물체를 덮고, 멈췄다가 되돌아간다. 상자의 뒷벽에 닿은 물체는 상자가 멈출 때까지 밀려 간다 (원인이 보이는 이동).
    한 장면 = 한 묶음 {P, N, B}: 상자 움직임은 셋이 같고 target 자리만 다르다.
      P (밀림)  : target 이 상자 길 위 앞쪽에 있어 뒷벽에 밀려 끝자리 x_f 로 옮겨진다 (= PerceptionTest moved)
      N (그대로): target 이 상자가 멈춘 자리 안 앞쪽에 있어 덮이기만 하고 제자리 x_n 에서 다시 보인다 (= nothing)
      B (빈 장면): target 없음
    문맥 끝 (샘플 15–16) 에 두 target 모두 상자 안에 있어 문맥 마지막 장들이 같다 → 짝 (P 문맥 + N 미래 = 엉뚱한 자리에서 나옴, P·N 문맥 + B 미래 = 사라짐 …) 이음새가 없다.
    상자는 렌더에서 속이 찬 상자다 (앞이 트인 모양은 설계 · 운동 계산에만 쓴다).
렌더: 이 모듈이 gen_render 의 설계 모듈이다 (설정 design: v4f_cover). import 하면 gen_design.panel_x · panel_ang 과 gen_design_v2.track 에
    hood 판 · pushed 운동을 붙인다 (다른 모드 · 운동 계산은 그대로).
번호: cover<seed>_<3i + j> (j 0 P · 1 N · 2 B).
  design: $BT_PY v4f_cover.py design --out <root> [--n 8]
  index : $BT_PY v4f_cover.py index --out <root> --name blender_track_v4f_cover
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_design as G1  # noqa: E402
import gen_design_v2 as G2  # noqa: E402
import gen_design_v3 as G3  # noqa: E402
import gen_design_v4vc as G  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "../../../.."))
SPLIT, SEED = "cover", 370
SRC = "/workspace/dataset/training_data/blender_track/v4b_long_hide"
LATE = {"start": [6, 15], "end": [16, 27]}         # v4b 는 시작 9–15. 밀리는 target 은 상자 길이만큼 먼저 덮인다

# ----------------------------------------------------------------------------- hood 판 · pushed 운동 (렌더 · 설계 공통)


def hood_x(p, T, fps):
    """상자 가운데 x (T,): t0 까지 xa → v_in 으로 d_in 만큼 (sg 방향) → dwell 멈춤 → v_out 으로 되돌아감."""
    t = np.arange(T) / fps
    t1 = p["t0"] + p["d_in"] / p["v_in"]
    t2 = t1 + p["dwell"]
    x = p["xa"] + p["sg"] * p["v_in"] * np.clip(t - p["t0"], 0.0, t1 - p["t0"])
    back = t > t2
    x[back] = p["xa"] + p["sg"] * p["d_in"] - p["sg"] * p["v_out"] * (t[back] - t2)
    return x


_panel_x0, _panel_ang0, _track0 = G1.panel_x, G1.panel_ang, G2.track


def panel_x(p, T, fps):
    return hood_x(p, T, fps) if p.get("mode") == "hood" else _panel_x0(p, T, fps)


def panel_ang(p, T, fps):
    return np.zeros(T) if p.get("mode") == "hood" else _panel_ang0(p, T, fps)


def track(o, T, fps, g):
    m = o["motion"]
    if m["kind"] != "pushed":
        return _track0(o, T, fps, g)
    h = m["hood"]
    bf = hood_x(h, T, fps) - h["sg"] * h["w"] / 2                     # 뒷벽 (안쪽 면) x
    over = h["sg"] * (bf - m["p0"][0]) + m["hx"]                       # 뒷벽이 물체 뒤쪽 끝을 넘은 깊이
    push = np.maximum.accumulate(np.maximum(over, 0.0))                # 밀린 거리 (되돌아갈 때 끌려오지 않는다)
    pos = np.c_[m["p0"][0] + h["sg"] * push, np.full(T, m["p0"][1]), np.zeros(T)]
    R = np.stack([G1._rz(m.get("yaw0", 0.0))] * T)
    return pos, np.full(T, m.get("yaw0", 0.0)), R


G1.panel_x, G1.panel_ang, G2.track = panel_x, panel_ang, track      # G1 · G2 안의 호출도 이 판을 쓴다

# gen_render 가 쓰는 이름
load_cfg, tracks, clip_id, center_path, DIMS = G.load_cfg, G2.tracks, G.clip_id, G2.center_path, G2.DIMS


def sample_clip(*a, **k):
    raise RuntimeError("v4f_cover: 설계는 design 명령으로 미리 만든다 (design/<id>.json)")


# ----------------------------------------------------------------------------- 설계
def _geom(o, T, fps, g):
    pos, _, R = track(o, T, fps, g)
    ctr = G2.center_path(o, pos, R)
    return ctr, G2.extent_pts(o, ctr, R)


def design_one(s0, cfg, rng):
    """v4b 판 · 벽 장면 s0 → (P, N, B 설계) 또는 (None, 이유)."""
    T, fps, g = int(s0["T"]), float(s0["fps"]), float(s0["g"])
    dt = G.STRIDE / fps
    W, V = {"late": LATE}, cfg["velocity"]
    cam = G3.cam_obj(s0["cam"])
    tgt = copy.deepcopy(s0["objects"][0])
    yaw0 = float(tgt["motion"].get("yaw0", 0.0)) if tgt["shape"] != "sphere" else 0.0
    hx, hy, hz = G2.obj_half(tgt)
    D = 2 * max(hx, hy)                                                # 물체 폭 (돌려도 상자에 들어가게 큰 쪽)
    sg = float(rng.choice([-1.0, 1.0]))
    y0 = float(rng.uniform(-0.05, 0.05))
    gap, mfront = G2._u(rng, (0.6, 1.2)) * D, G2._u(rng, (0.3, 0.9)) * D
    w = D + gap + D + mfront + 0.02                                    # 뒷벽 · 밀린 물체 · 틈 · 그대로 물체 · 앞쪽 여유
    th = D + G2._u(rng, (0.25, 0.7)) * D
    h = 2 * hz * G2._u(rng, (1.25, 1.8))
    xs = float(rng.uniform(-0.1, 0.1))                                 # 상자가 멈추는 가운데
    bf_s = xs - sg * w / 2
    xf = bf_s + sg * D / 2                                             # 밀린 물체의 끝자리 (뒷벽에 닿음)
    xn = xf + sg * (D + gap)                                           # 그대로 둘 물체 자리 (상자 안 앞쪽)
    delta = G2._u(rng, (0.8, 2.0)) * D                                 # 밀린 거리
    xp0 = xf - sg * delta
    gap0 = G2._u(rng, (0.3, 1.2)) * D                                  # 처음에 앞벽이 밀릴 물체 뒤쪽 끝에서 떨어진 거리
    xa = xp0 - sg * (D / 2 + gap0 + w / 2)
    d_in = sg * (xs - xa)
    S0 = G2._u(rng, (2.0, 6.0))                                        # 상자가 출발하는 샘플 (그 전에는 xa 에 서 있다, 문맥 안)
    S_stop = G2._u(rng, (13.0, 15.5))                                  # 멈추는 샘플
    hood = {"role": "event", "kind": "hood", "mode": "hood", "x": float(xs), "y": y0, "w": float(w), "th": float(th), "h": float(h),
            **G3.panel_look(rng, "board"), "sg": sg, "xa": float(xa), "t0": float(S0 * dt), "d_in": float(d_in), "v_in": float(d_in / ((S_stop - S0) * dt)),
            "dwell": float(G2._u(rng, (2.0, 6.0)) * dt), "v_out": 0.0}
    hood["v_out"] = float(hood["v_in"] * G2._u(rng, (0.7, 1.3)))
    P = copy.deepcopy(tgt)
    P["motion"] = {"kind": "pushed", "p0": [float(xp0), y0], "yaw0": yaw0, "hx": float(D / 2), "hood": {k: hood[k] for k in ("sg", "xa", "t0", "d_in", "v_in", "dwell", "v_out", "w")}}
    N = copy.deepcopy(tgt)
    N["motion"] = {"kind": "flat_static", "p0": [float(xn), y0], "yaw0": yaw0}
    des = G2.panel_boxes([hood], T, fps)
    evs, pxs = [], []
    for o in (P, N):
        ctr, E = _geom(o, T, fps, g)
        u_, v_, d_ = cam.to_px(ctr[::G.STRIDE])
        if (d_ < 0.3).any():
            return None, "behind_cam"
        off = G2.off_frame(cam, E)[::G.STRIDE]
        ok, ev = G.timing_ok(G2.target_hidden(cam, des, E)[::G.STRIDE], off, W, float(V["max_off"]))
        if not ok:
            return None, f"late_{'P' if o is P else 'N'}"
        E_in = ctr[:, None] + 0.55 * (E - ctr[:, None])
        if not G.timing_ok(G2.target_hidden(cam, des, E_in)[::G.STRIDE], off, W, float(V["max_off"]))[0]:
            return None, "late_in"
        evs.append(list(ev)); pxs.append(float(np.median(cam.f * G3.s_eff(o) / np.maximum(d_, 1e-3))))
    cP, _ = _geom(P, T, fps, g)
    cN, _ = _geom(N, T, fps, g)
    if abs(cP[-1, 0] - cN[-1, 0]) < 0.9 * D:
        return None, "too_close"
    xh = hood_x(hood, T, fps)
    others = []
    for d in s0["objects"][1:]:                                         # 그 밖 물체: 상자가 쓸고 가는 자리와 안 겹치고, 서로 크게 안 가린다
        pd, _, Rd = track(d, T, fps, g); cd = G2.center_path(d, pd, Rd)
        rd = G2.foot_r(d)
        if ((np.abs(cd[:, 0] - xh) < w / 2 + rd + 0.06) & (np.abs(cd[:, 1] - y0) < th / 2 + rd + 0.06)).any():
            return None, "extra_hood"
        if G2.collide(P, cP, d, cd, 0.12) or G2.collide(N, cN, d, cd, 0.12):
            return None, "hit_extra"
        ud_, vd_, dd_ = cam.to_px(cd[::G.STRIDE])
        inn = (ud_ > 0) & (ud_ < 256) & (vd_ > 0) & (vd_ < 256) & (dd_ > 0.3)
        hid_d = G2.target_hidden(cam, des, G2.extent_pts(d, cd, Rd))[::G.STRIDE]
        if hid_d[inn].max(initial=0.0) > 0.5:
            return None, "hides_extra"
        for o in (P, N):
            if G2.target_hidden(cam, G2.obj_boxes([d], T, fps, g, 1.05), _geom(o, T, fps, g)[1]).max() > 0.02:
                return None, "extra_hides"
        others.append(d)
    base = copy.deepcopy(s0)
    base.pop("render", None)
    base.update(panels=[hood], split=SPLIT, seed=SEED, design="v4f_cover", cover_from=f"{s0['split']}{s0['seed']}_{int(s0['idx']):06d}",
                cell={**s0["cell"], "occ": "hood", "occ_sub": "push"})
    out = []
    for lv, o, ev, px in (("P", P, evs[0], pxs[0]), ("N", N, evs[1], pxs[1])):
        s = copy.deepcopy(base)
        s.update(objects=[o] + others, level=lv, event_samples=ev, target_px_median=px)
        chk = copy.deepcopy(s); chk["panels"] = []                     # 상자 안 겹침은 설계된 담김이라 판은 빼고 검사
        bad = [b for b in G2.physics_check(chk) if "obj0" not in b or "바닥" in b]
        if bad:
            return None, "phys"
        out.append(s)
    sB = copy.deepcopy(base); sB.update(objects=list(others), level="B", ghost_target=P, event_samples=None)
    out.append(sB)
    return out, None


_CFG = {}


def _design_job(job):
    """병렬 설계 하나: (묶음 번호 i, 원본 장면 f, 반복 r) → (설계 목록 또는 None, 실패 이유 수)."""
    i, f, r = job
    if "cfg" not in _CFG:
        _CFG["cfg"] = G.load_cfg(f"{SRC}/config_v4b.yaml", None)
    cfg = _CFG["cfg"]
    sp = cfg["splits"]["train_fam"]; K = int(cfg["velocity"]["levels"])
    s0 = json.load(open(f"{SRC}/design/train_fam{sp['seed']}_{f * (K + 1):06d}.json"))
    why = collections.Counter()
    for t in range(60):
        got, e = design_one(s0, cfg, np.random.default_rng([SEED, f, r, t]) if r else np.random.default_rng([SEED, f, t]))
        if got:
            return got, why
        why[e] += 1
    return None, why


def cmd_design(a):
    out = os.path.abspath(a.out)
    cfg = G.load_cfg(f"{SRC}/config_v4b.yaml", None)
    os.makedirs(f"{out}/design", exist_ok=True); os.makedirs(f"{out}/logs", exist_ok=True)
    sp = cfg["splits"]["train_fam"]; K = int(cfg["velocity"]["levels"])
    fams = []
    for f in range(int(sp["candidates"])):
        c0 = f"train_fam{sp['seed']}_{f * (K + 1):06d}"
        if os.path.isfile(f"{SRC}/design/{c0}.json"):
            s0 = json.load(open(f"{SRC}/design/{c0}.json"))
            if s0["cell"]["occ"] in ("board", "wall") and s0.get("level") == 0:
                fams.append(f)
    rng0 = np.random.default_rng(SEED)
    rng0.shuffle(fams)
    jobs = [(i, f, r) for i, (f, r) in enumerate((f, r) for f in fams[: a.n] for r in range(a.reps))]
    why, rows, todo = collections.Counter(), [], []
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=a.procs) as ex:
        for (i, f, r), (got, w8) in zip(jobs, ex.map(_design_job, jobs, chunksize=4)):
            why.update(w8)
            if not got:
                rows.append({"src_family": f, "rep": r, "ok": False}); continue
            for j, s in enumerate(got):
                s["idx"] = int(i * 3 + j); s["family"] = int(i)
                json.dump(s, open(f"{out}/design/{G.clip_id(s)}.json", "w")); todo.append(s["idx"])
            rows.append({"src_family": f, "rep": r, "ok": True, "event_P": got[0]["event_samples"], "event_N": got[1]["event_samples"], "sg": got[0]["panels"][0]["sg"]})
    c2 = copy.deepcopy(cfg); c2["out"] = out; c2["name"] = "v4f"; c2["design"] = "v4f_cover"
    c2["timing"] = {"late": LATE}
    c2["splits"] = {SPLIT: {"seed": SEED, "families": len(jobs), "candidates": len(jobs), "n": 0}}
    json.dump(c2, open(f"{out}/config_v4f.json", "w"), indent=1)
    import yaml
    yaml.safe_dump(c2, open(f"{out}/config_v4f.yaml", "w"), allow_unicode=True, sort_keys=False)
    open(f"{out}/extra_{SPLIT}.txt", "w").write(" ".join(map(str, todo)))
    ok = [r for r in rows if r["ok"]]
    hl = lambda k: [int(r[k][1] - r[k][0] + 1) for r in ok]   # noqa: E731
    rep = {"src": SRC, "source_families": len(jobs), "source_scenes": len(fams[: a.n]), "reps": a.reps, "families_ok": len(ok), "clips": len(todo), "fail": dict(why),
           "hidden_len_P_median": float(np.median(hl("event_P"))) if ok else None, "hidden_len_N_median": float(np.median(hl("event_N"))) if ok else None,
           "start_P_median": float(np.median([r["event_P"][0] for r in ok])) if ok else None, "rows": rows}
    json.dump(rep, open(f"{out}/v4f_design_report.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, ensure_ascii=False))


def cmd_index(a):
    sys.path.insert(0, f"{REPO}/z_training/data")
    import compose_v4vc as CP  # noqa: E402
    out = os.path.abspath(a.out)
    cfg = G.load_cfg(f"{out}/config_v4f.yaml", None)
    rep = json.load(open(f"{out}/v4f_design_report.json"))
    paths, fams, summ = [], [], collections.Counter()
    for i in range(rep["source_families"]):
        cs = [f"{SPLIT}{SEED}_{i * 3 + j:06d}" for j in range(3)]
        if not os.path.isfile(f"{out}/design/{cs[0]}.json"):
            continue
        if not all(os.path.isfile(f"{out}/{d}/{c}.{x}") for c in cs for d, x in (("clips", "mp4"), ("labels", "npz"))):
            summ["unrendered"] += 1; continue
        hs, ok_all = [], True
        for c in cs[:2]:
            ok, ev, h = CP.render_timing(cfg, f"{out}/labels/{c}.npz")
            ok_all &= bool(ok); hs.append(bool(h))
        if not ok_all:
            summ["timing_fail"] += 1; continue
        i0 = len(paths); paths += [f"{out}/clips/{c}.mp4" for c in cs]
        fams.append({"plane": "flat", "family": int(i), "occ": "hood", "A": [i0, i0 + 1], "B": i0 + 2, "hs": hs})
        summ["ok"] += 1
    N = a.name
    d1, d2 = f"{REPO}/data_csv/{N}/arms", f"{REPO}/data_csv/{N}_negimp"
    os.makedirs(d1, exist_ok=True); os.makedirs(d2, exist_ok=True)
    open(f"{d1}/all_mp4list.csv", "w").writelines(p + "\n" for p in paths)
    s = {"name": N, "videos": len(paths), "families": len(fams), "hidden_splice_pass": int(sum(sum(f["hs"]) for f in fams)), "counts": dict(summ), "out": out}
    json.dump(s, open(f"{d1}/all_summary.json", "w"), indent=1)
    json.dump({"list": os.path.abspath(f"{d1}/all_mp4list.csv"), "splice": 32, "families": fams, "summary": s}, open(f"{d2}/families.json", "w"))
    print(json.dumps(s, indent=1, ensure_ascii=False))


def cmd_preview(a):
    """묶음마다 [P | N | B] 를 나란히 (모델이 읽는 간격 2 의 32 샘플). 샘플 16 부터 (미래) 테두리 빨강. GIF + 정지 장 격자 PNG."""
    import decord
    from PIL import Image, ImageDraw
    out = os.path.abspath(a.out)
    os.makedirs(f"{out}/preview", exist_ok=True)
    rep = json.load(open(f"{out}/v4f_design_report.json"))
    fams = [i for i in range(rep["source_families"]) if os.path.isfile(f"{out}/clips/{SPLIT}{SEED}_{i * 3:06d}.mp4")][: a.n]
    names, sc = ("pushed (P)", "stays (N)", "empty (B)"), float(a.scale)
    sheets = []
    for i in fams:
        vids = [decord.VideoReader(f"{out}/clips/{SPLIT}{SEED}_{i * 3 + j:06d}.mp4").get_batch(list(range(0, 64, 2))).asnumpy() for j in range(3)]
        frames = []
        for t in range(32):
            row = []
            for j in range(3):
                im = Image.fromarray(vids[j][t]).resize((int(256 * sc), int(256 * sc)))
                d = ImageDraw.Draw(im)
                d.rectangle([0, 0, im.width - 1, im.height - 1], outline=(220, 40, 40) if t >= 16 else (60, 60, 60), width=3)
                d.text((5, 4), f"{names[j]}  s{t}", fill=(255, 255, 255))
                row.append(im)
            canvas = Image.new("RGB", (sum(r.width for r in row) + 8, row[0].height), (20, 20, 20))
            x = 0
            for r in row:
                canvas.paste(r, (x, 0)); x += r.width + 4
            frames.append(canvas)
        frames[0].save(f"{out}/preview/family{i:03d}.gif", save_all=True, append_images=frames[1:], duration=200, loop=0)
        keep = [0, 6, 10, 14, 16, 20, 24, 30]
        sheets.append([frames[t] for t in keep])
    if sheets:
        w, h = sheets[0][0].width, sheets[0][0].height
        big = Image.new("RGB", (w * 2 + 8, (h + 4) * 4 * len(sheets)), (255, 255, 255))
        for fi, row in enumerate(sheets):
            for k, im in enumerate(row):
                big.paste(im, ((k % 2) * (w + 8), (fi * 4 + k // 2) * (h + 4)))
        big.save(f"{out}/preview/sheet.png")
    print(json.dumps({"families": fams, "dir": f"{out}/preview"}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["design", "index", "preview"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10000, help="쓸 v4b 판 · 벽 장면 수 (기본 전부)")
    ap.add_argument("--name", default="blender_track_v4f_cover")
    ap.add_argument("--scale", type=float, default=0.75)
    ap.add_argument("--reps", type=int, default=1, help="원본 장면마다 묶음 수 (배치 · 시각을 다시 뽑는다)")
    ap.add_argument("--procs", type=int, default=32)
    a = ap.parse_args()
    {"design": cmd_design, "index": cmd_index, "preview": cmd_preview}[a.cmd](a)


if __name__ == "__main__":
    main()
