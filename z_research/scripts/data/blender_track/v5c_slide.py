#!/usr/bin/env python3
"""BlenderTrack v5c (2026-11-09) — v4c (미끄러지는 가림막) 를 v5id 평면 묶음 위에서, target 정체 변형까지.

v4c 와 같은 방식 (v4c_slide.design_one): v5id 평면 '판 · 벽' 묶음의 장면 (카메라 · 환경 · 그 밖 물체) 을 쓰고 고정 판을 미끄러지는 판으로 바꾼다.
재료 단계 = 그 묶음의 정체 변형이 붙은 속도 단계 k* (변형과 궤적이 같다). 장면 하나에서 두 묶음:
  움직이는 target  M (원래 정체) · M_C · M_S · M_CS (같은 궤적, 정체만 다름) · B_M (빈 장면)
  정지 target      S · S_C · S_S · S_CS · B_S (가려지는 구간 가운데에 서 있음)
판은 원래 정체로 설계하고 (v4c 와 같은 검사), 변형 셋도 같은 판으로 v5id 의 변형 검사 (gen_design_v5id._check_variant) 를 통과해야 한다.
안 되면 판을 다시 뽑는다 (최대 30 번, v4c 와 같음). 번호: slide<seed>_<i·10 + j>, j 0 M · 1–3 M 변형 · 4 B_M · 5 S · 6–8 S 변형 · 9 B_S.
  design: $BT_PY v5c_slide.py design --src <v5id root> --n 4200
  렌더  : CFG=<root>/config_v5c.json SPLIT=slide bash run_v4a.sh <root> <root>/extra_slide.txt
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
import gen_design_v5id as G5  # noqa: E402
import v4c_slide as V4C  # noqa: E402

SPLIT, SEED = "slide", 470
VARS = ("C", "S", "CS")


def design_scene(s_lv, variants, cfg, f, static):
    """(묶음 설계 목록 [원래, 변형 3, 빈 장면], None) 또는 (None, 이유)."""
    why = collections.Counter()
    for t in range(30):
        rng = np.random.default_rng([SEED, f, int(static), t])
        got, e = V4C.design_one(s_lv, cfg, rng, static)
        if not got:
            why[e] += 1; continue
        s, sB = got
        out = []
        for name, v0 in variants:
            v = copy.deepcopy(v0)
            if static:
                v["motion"] = copy.deepcopy(s["objects"][0]["motion"])
            r = G5._check_variant(cfg, s, v)
            if isinstance(r, str):
                why[f"id_{r}"] += 1; out = None; break
            sv = copy.deepcopy(s); sv["objects"][0] = v
            sv.update(level=("S" if static else "M") + "V" + name, event_samples=r[0], target_px_median=r[1])
            out.append(sv)
        if out is None:
            continue
        return [s] + out + [sB], None
    return None, dict(why)


def cmd_design(a):
    src = os.path.abspath(a.src)
    cfg = G5.load_cfg(f"{src}/config_v5id.yaml", None)
    os.makedirs(f"{src}/design", exist_ok=True)
    sp = cfg["splits"]["train_fam"]; S = G5.n_slots(cfg); K = int(cfg["velocity"]["levels"])
    fams = []
    for f in range(int(sp["candidates"])):
        c0 = f"train_fam{sp['seed']}_{f * S:06d}"
        if os.path.isfile(f"{src}/design/{c0}.json"):
            s0 = json.load(open(f"{src}/design/{c0}.json"))
            if s0["cell"]["occ"] in ("board", "wall"):
                fams.append(f)
    rng0 = np.random.default_rng(SEED)
    rng0.shuffle(fams)
    why, n_ok, todo, rows = collections.Counter(), 0, [], []
    for i, f in enumerate(fams[: a.n]):
        s0 = json.load(open(f"{src}/design/train_fam{sp['seed']}_{f * S:06d}.json"))
        k = int(s0["identity"]["base_level"])
        s_lv = json.load(open(f"{src}/design/train_fam{sp['seed']}_{f * S + k:06d}.json"))
        variants = []
        for j in range(3):
            sv = json.load(open(f"{src}/design/train_fam{sp['seed']}_{f * S + K + j:06d}.json"))
            variants.append((sv["identity"]["var"], sv["objects"][0]))
        for base_j, static in ((0, False), (5, True)):
            got, e = design_scene(s_lv, variants, cfg, f, static)
            if not got:
                for k_, v_ in (e or {}).items():
                    why[k_] += v_
                rows.append({"src_family": f, "static": static, "ok": False}); continue
            for jj, s in enumerate(got):
                s.update(idx=int(i * 10 + base_j + jj), family=int(i), split=SPLIT, seed=SEED, design="gen_design_v5id",
                         slide_from=f"train_fam{sp['seed']}_{f * S + k:06d}")
                if jj == 0:
                    s["identity"] = {**s_lv["identity"], "var": "base"}
                elif jj <= 3:
                    s["identity"] = {**json.load(open(f"{src}/design/train_fam{sp['seed']}_{f * S + K + jj - 1:06d}.json"))["identity"]}
                else:
                    s.pop("identity", None)
                s.pop("render", None)
                json.dump(s, open(f"{src}/design/{SPLIT}{SEED}_{s['idx']:06d}.json", "w")); todo.append(s["idx"])
            n_ok += 1
            rows.append({"src_family": f, "static": static, "ok": True, "event": got[0]["event_samples"], "rel": got[0]["panels"][0].get("rel")})
        if i % 200 == 0:
            print(f"{i}/{min(a.n, len(fams))} 장면 · 성공 묶음 {n_ok} · 실패 {dict(why)}", flush=True)
    c2 = copy.deepcopy(cfg); c2["name"] = "v5c"; c2["design"] = "gen_design_v5id"
    c2["splits"] = {SPLIT: {"seed": SEED, "families": len(fams[: a.n]), "candidates": len(fams[: a.n]), "n": 0}}
    json.dump(c2, open(f"{src}/config_v5c.json", "w"), indent=1)
    open(f"{src}/extra_{SPLIT}.txt", "w").write(" ".join(map(str, todo)))
    rep = {"src": src, "source_families": len(fams[: a.n]), "families_ok": n_ok, "clips": len(todo), "fail": dict(why),
           "rel": dict(collections.Counter(r.get("rel") for r in rows if r.get("ok"))), "rows": rows}
    json.dump(rep, open(f"{src}/v5c_design_report.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["design"])
    ap.add_argument("--src", default="/workspace/dataset/training_data/blender_track/v5id")
    ap.add_argument("--n", type=int, default=4200, help="재료 장면 수 (장면마다 움직임 · 정지 두 묶음, 묶음마다 5 편)")
    a = ap.parse_args()
    {"design": cmd_design}[a.cmd](a)


if __name__ == "__main__":
    main()
