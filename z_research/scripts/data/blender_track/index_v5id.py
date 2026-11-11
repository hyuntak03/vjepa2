#!/usr/bin/env python3
"""BlenderTrack v5id 학습 목록 + 음의 손실 묶음 (2026-11-09). GPU 불필요.

재료: <root> 의 평면 (config_v5id, train_fam) · 빗면 (config_v5idr, rtrain_fam) · 미끄러지는 가림막 (config_v5c, slide) 렌더.
묶음 = 같은 장면 · 같은 가림막. A = target 이 있는 영상들, B = 빈 장면.
  평면 · 빗면: A = 속도 단계 K 편 (원래 정체) + 정체 변형 (단계 k* 와 같은 궤적) / B
  미끄러지는 가림막: 장면마다 두 묶음 (움직이는 target · 정지 target), A = 원래 + 정체 변형 / B
통과 규칙: 원래 정체 단계 · 빈 장면이 모두 렌더되고 원래 단계가 렌더 타이밍 (late) 을 통과해야 묶음을 쓴다 (v4b 와 같음).
  변형은 렌더 · 타이밍을 통과한 것만 A 에 넣는다.
families.json 의 묶음마다 (ImpTable 이 읽는 키 A · B · hs 는 v4b 와 같다):
  lvl   A 마다 속도 단계 (변형 = k*)
  ident A 마다 'base' 또는 변형 이름 (C · S · CS · 빗면 primitive C1 · C2 · C3)
  idc   A 마다 [색 바뀜, 모양 바뀜] (원래 정체 대비)
→ 정체 위반 짝 = 같은 lvl 의 두 A (문맥 A_i[0:32] + 미래 A_j[32:64], 둘 다 hs). 사라짐 · 나타남 짝은 v4b 와 같다.
  $BT_PY index_v5id.py --root <root> --name blender_track_v5id [--parts flat,ramp,slide]
출력: data_csv/<name>/arms/all_mp4list.csv · all.npz · all_summary.json · data_csv/<name>_negimp/families.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, "../../../.."))
import compose_v4vc as CP  # noqa: E402
import gen_design_v5id as G5  # noqa: E402
from make_v3_meta import vis_of  # noqa: E402

ROOT = None
CFGS = {}
SPLITS = {"flat": "train_fam", "ramp": "rtrain_fam"}


def _init(root, splits=None):
    global ROOT
    ROOT = root
    if splits:
        SPLITS.update(splits)
    for k, p in (("flat", "config_v5id.yaml"), ("ramp", "config_v5idr.yaml"), ("slide", "config_v5c.json")):
        if os.path.isfile(f"{root}/{p}"):
            CFGS[k] = G5.load_cfg(f"{root}/{p}", None)


def _slot(cfg, split, idx):
    """(cid, 설계, 렌더됨, 타이밍 통과, 이음에서 가려짐)."""
    cid = f"{split}{cfg['splits'][split]['seed']}_{idx:06d}"
    dj = f"{ROOT}/design/{cid}.json"
    if not os.path.isfile(dj):
        return cid, None, False, None, None
    sp = json.load(open(dj))
    have = os.path.isfile(f"{ROOT}/clips/{cid}.mp4") and os.path.isfile(f"{ROOT}/labels/{cid}.npz")
    if not have:
        return cid, sp, False, None, None
    if sp.get("level") in ("B",):
        return cid, sp, True, True, None
    ok, _, hs = CP.render_timing(cfg, f"{ROOT}/labels/{cid}.npz")
    return cid, sp, True, bool(ok), bool(hs)


def _group(cfg, split, base_idx, var_idx, b_idx, lvl_base, kstar):
    """원래 단계 번호들 · 변형 번호들 · 빈 장면 번호 → 묶음 dict 또는 (None, 이유)."""
    rows = [_slot(cfg, split, i) for i in base_idx]
    if any(r[1] is None for r in rows):
        return None, "unselected"
    if not all(r[2] for r in rows):
        return None, "base_unrendered"
    if not all(r[3] for r in rows):
        return None, "base_timing"
    rb = _slot(cfg, split, b_idx)
    if not rb[2]:
        return None, "B_unrendered"
    A, hs, lvl, ident, idc = [], [], [], [], []
    for (cid, sp, _, _, h), lv in zip(rows, lvl_base):
        A.append(cid); hs.append(bool(h)); lvl.append(int(lv)); ident.append("base"); idc.append([0, 0])
    nv_drop = 0
    for i in var_idx:
        cid, sp, have, ok, h = _slot(cfg, split, i)
        if sp is None or not have or not ok:
            nv_drop += 1; continue
        idn = sp.get("identity", {})
        A.append(cid); hs.append(bool(h)); lvl.append(int(kstar)); ident.append(idn.get("var", "?"))
        idc.append([int(bool(idn.get("color"))), int(bool(idn.get("shape")))])
    s0 = rows[0][1]
    return {"A": A, "B": rb[0], "hs": hs, "lvl": lvl, "ident": ident, "idc": idc, "var_dropped": nv_drop,
            "cell": s0.get("cell", {}), "kind0": s0.get("identity", {}).get("kind0"), "shape0": s0.get("identity", {}).get("shape0")}, None


def _fam(arg):
    plane, f = arg
    if plane in ("flat", "ramp"):
        cfg = CFGS[plane]; split = SPLITS[plane]
        K, S = int(cfg["velocity"]["levels"]), G5.n_slots(cfg)
        c0 = _slot(cfg, split, f * S)
        if c0[1] is None:
            return []
        kstar = int(c0[1]["identity"]["base_level"])
        g, e = _group(cfg, split, [f * S + k for k in range(K)], [f * S + K + j for j in range(3)], f * S + S - 1, list(range(K)), kstar)
        return [(plane, f, g, e)]
    cfg = CFGS["slide"]
    out = []
    for nm, b0 in (("slide_move", 0), ("slide_static", 5)):
        g, e = _group(cfg, "slide", [f * 10 + b0], [f * 10 + b0 + 1 + j for j in range(3)], f * 10 + b0 + 4, [0], 0)
        if e != "unselected":
            out.append((nm, f, g, e))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/dataset/training_data/blender_track/v5id")
    ap.add_argument("--name", default="blender_track_v5id")
    ap.add_argument("--parts", default="flat,ramp,slide")
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--flat-split", default="train_fam"); ap.add_argument("--ramp-split", default="rtrain_fam")
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    spl = {"flat": a.flat_split, "ramp": a.ramp_split}
    _init(root, spl)
    jobs = []
    for plane in a.parts.split(","):
        if plane in ("flat", "ramp"):
            cfg = CFGS[plane]; split = SPLITS[plane]
            jobs += [(plane, f) for f in range(int(cfg["splits"][split]["candidates"]))]
        elif plane == "slide" and "slide" in CFGS:
            jobs += [("slide", f) for f in range(int(CFGS["slide"]["splits"]["slide"]["candidates"]))]
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(root, spl)) as ex:
        res = [x for r in ex.map(_fam, jobs, chunksize=16) for x in r]
    paths, fam_out, summ = [], [], collections.Counter()
    var_ct = collections.Counter()
    for plane, f, g, e in res:
        if g is None:
            summ[f"{plane}_{e}"] += 1; continue
        i0 = len(paths)
        paths += [f"{root}/clips/{c}.mp4" for c in g["A"]] + [f"{root}/clips/{g['B']}.mp4"]
        fam_out.append({"plane": plane, "family": int(f), "occ": g["cell"].get("occ"), "A": list(range(i0, i0 + len(g["A"]))), "B": i0 + len(g["A"]),
                        "hs": g["hs"], "lvl": g["lvl"], "ident": g["ident"], "idc": g["idc"], "kind0": g["kind0"], "shape0": g["shape0"]})
        summ[f"keep_{plane}"] += 1; summ["var_dropped"] += g["var_dropped"]
        for nm, h in zip(g["ident"], g["hs"]):
            var_ct[(plane, nm, "hs" if h else "not_hs")] += 1
    N = a.name
    d1, d2 = f"{REPO}/data_csv/{N}/arms", f"{REPO}/data_csv/{N}_negimp"
    os.makedirs(d1, exist_ok=True); os.makedirs(d2, exist_ok=True)
    open(f"{d1}/all_mp4list.csv", "w").writelines(p + "\n" for p in paths)
    vis = [vis_of(f"{root}/labels/{os.path.basename(p)[:-4]}.npz") for p in paths]
    np.savez(f"{d1}/all.npz", clip=np.array([os.path.basename(p)[:-4] for p in paths]), path=np.array(paths), vis=np.stack(vis) if vis else np.zeros((0, 64)))
    n_id_pairs = sum(sum(1 for i, li in enumerate(fm["lvl"]) for j, lj in enumerate(fm["lvl"]) if i != j and li == lj and fm["hs"][i] and fm["hs"][j]
                         and fm["ident"][i] != fm["ident"][j]) for fm in fam_out)
    s = {"name": N, "videos": len(paths), "families": len(fam_out), "identity_ordered_pairs": int(n_id_pairs), "counts": dict(summ),
         "variants": {"|".join(k): v for k, v in sorted(var_ct.items())}, "out": root}
    json.dump(s, open(f"{d1}/all_summary.json", "w"), indent=1, ensure_ascii=False)
    json.dump({"list": os.path.abspath(f"{d1}/all_mp4list.csv"), "splice": 32, "families": fam_out, "summary": s}, open(f"{d2}/families.json", "w"))
    print(json.dumps({k: v for k, v in s.items() if k != "variants"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
