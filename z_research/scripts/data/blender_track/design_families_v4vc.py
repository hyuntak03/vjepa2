#!/usr/bin/env python3
"""BlenderTrack v4vc 묶음 설계 (CPU, 렌더 전) — 후보 묶음을 설계하고, 상황마다 앞에서부터 고른 뒤 렌더 번호마다 spec 을 쓴다 (2026-11-02).

  설계   : 후보 f = 0..candidates−1 → <out>/design_fam/<split>_f<f:06d>.json (단계 K 개 + 빈 장면 spec 목록) 또는 .fail
           여러 노드 · 프로세스로 나눈다 (--part k --nparts m, f % m == k). 이미 있는 후보는 건너뛴다.
  고르기 (--select): 상황 (물체 셋 × 카메라 × 가리는 것, 16) 마다 성공한 후보를 상황 안 번호 순으로 families / 16 개.
           고른 묶음의 렌더 번호 (gen_design_v4vc.render_slots) 마다 <out>/design/<split><seed>_<idx>.json, 나머지 번호는 .fail ("unselected").
           보고 <out>/select_<split>.json (상황마다 후보 · 성공 · 고른 수, 요인 수준별 개수, 실패 이유 합)
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P design_families_v4vc.py --config configs/v4vc.yaml --split train_fam --procs 90
  $P design_families_v4vc.py --config configs/v4vc.yaml --split train_fam --select
"""
import argparse
import collections
import copy
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib  # noqa: E402
G = importlib.import_module(os.environ.get("BT_V4_DESIGN", "gen_design_v4vc"))  # noqa: E402  (2026-11-06: BT_V4_DESIGN=gen_design_v4e → 평면 가속 판. 기본은 예전과 같다)

CFG = {}


def _init(path, split):
    CFG["cfg"] = G.load_cfg(path, split); CFG["split"] = split


def _dump(obj, path):
    tmp = f"{path}.{os.getpid()}.tmp"
    json.dump(obj, open(tmp, "w"), default=lambda x: x.item() if hasattr(x, "item") else str(x))
    os.replace(tmp, path)


def one(f):
    cfg, split = CFG["cfg"], CFG["split"]
    p = f"{cfg['out']}/design_fam/{split}_f{f:06d}"
    if os.path.exists(p + ".json") or os.path.exists(p + ".fail"):
        return 0
    try:
        specs = G.design_family(cfg, split, f, max_tries=400, streams=int(cfg["checks"].get("design_streams", 3)))
    except RuntimeError as e:
        open(p + ".fail", "w").write(str(e))
        return 2
    _dump(specs, p + ".json")
    return 1


def select(cfg, split):
    out, sp = cfg["out"], cfg["splits"][split]
    S = G.situations(cfg); NS = len(S)
    NC, NF = int(sp["candidates"]), int(sp["families"])
    per = NF // NS
    os.makedirs(f"{out}/design", exist_ok=True)
    rep = {"split": split, "per_situation": per, "situations": {}}
    chosen = []
    why_all = collections.Counter()
    for si, sit in enumerate(S):
        cands = [j * NS + si for j in range((NC + NS - 1) // NS) if j * NS + si < NC]
        ok = [f for f in cands if os.path.exists(f"{out}/design_fam/{split}_f{f:06d}.json")]
        for f in cands:
            fp = f"{out}/design_fam/{split}_f{f:06d}.fail"
            if os.path.exists(fp):
                m = re.search(r"이유 (\{.*\})", open(fp).read())
                if m:
                    for k, v in eval(m.group(1)).items():
                        why_all[re.sub(r"\d+$", "", k)] += v
        pick = ok[:per]
        chosen += pick
        lv = collections.defaultdict(collections.Counter)
        for f in pick:
            for k, v in G.family_of(cfg, split, f)[3].items():
                lv[k][str(v)] += 1
        rep["situations"][G.sit_key(sit)] = {"candidates": len(cands), "ok": len(ok), "picked": len(pick), "levels": {k: dict(v) for k, v in lv.items()}}
    chosen = set(chosen)
    n_json = n_fail = 0
    for f in range(NC):
        slots = dict(G.render_slots(cfg, split, f)) if f in chosen else {}
        specs = json.load(open(f"{out}/design_fam/{split}_f{f:06d}.json")) if f in chosen else None
        for idx in G.all_slots(cfg, split, f):
            cid = f"{split}{sp['seed']}_{idx:06d}"
            if idx in slots:
                lev = slots[idx]
                s_ = copy.deepcopy(specs[-1] if lev == "B" else specs[lev])
                s_["idx"] = int(idx)
                _dump(s_, f"{out}/design/{cid}.json"); n_json += 1
                if os.path.exists(f"{out}/design/{cid}.fail"):
                    os.remove(f"{out}/design/{cid}.fail")
            else:
                open(f"{out}/design/{cid}.fail", "w").write("unselected" if f not in chosen else "unused slot (ind: 단계 하나만)")
                n_fail += 1
    rep.update(chosen=len(chosen), render_json=n_json, render_fail=n_fail, fail_reasons=dict(why_all))
    json.dump(rep, open(f"{out}/select_{split}.json", "w"), indent=1, ensure_ascii=False)
    print(f"{split}: 고른 묶음 {len(chosen)} / 목표 {NF} · 렌더 번호 {n_json} (건너뜀 {n_fail}) · 실패 이유 합 {dict(why_all)}")
    for k, v in rep["situations"].items():
        print(f"   {k:32s} 후보 {v['candidates']:4d} 성공 {v['ok']:4d} 고름 {v['picked']:4d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--split", required=True)
    ap.add_argument("--procs", type=int, default=32); ap.add_argument("--part", type=int, default=0); ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--select", action="store_true")
    a = ap.parse_args()
    cfg = G.load_cfg(a.config, a.split)
    if a.select:
        return select(cfg, a.split)
    os.makedirs(f"{cfg['out']}/design_fam", exist_ok=True)
    fams = [f for f in range(int(cfg["splits"][a.split]["candidates"])) if f % a.nparts == a.part]
    t0, c = time.time(), [0, 0, 0]
    with ProcessPoolExecutor(a.procs, initializer=_init, initargs=(a.config, a.split)) as ex:
        for k, r in enumerate(ex.map(one, fams, chunksize=1)):
            c[r] += 1
            if k % 200 == 0:
                print(f"{a.split} part {a.part}/{a.nparts}: {k}/{len(fams)} 건너뜀 {c[0]} 성공 {c[1]} 실패 {c[2]} | {time.time() - t0:.0f}s", flush=True)
    print(f"끝 {a.split} part {a.part}: {len(fams)} 묶음 · 성공 {c[1]} · 실패 {c[2]} · 건너뜀 {c[0]} | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
