#!/usr/bin/env python3
"""BlenderTrack v5mc 묶음 설계 (CPU, 렌더 전) — 후보 묶음 설계 → 상황마다 앞에서부터 고르기 → 렌더 번호 (빈틈 없이) · 렌더 설정 (2026-11-02).

  설계   : 후보 f → <out>/design_fam/<split>_f<f:06d>.json (단계 키 → spec) 또는 .fail. 여러 노드로 나눠 돈다 (--part k --nparts m).
  고르기 (--select): 상황 (26) 마다 성공한 후보를 상황 안 번호 순으로 families / 26 개. 고른 묶음의 spec 을 순서대로 번호 0, 1, 2 … 에
           <out>/design/<split><seed>_<idx:06d>.json 으로 쓴다 (v4 와 달리 빈 번호 · .fail 파일이 없다 — NFS 의 작은 파일 수를 줄인다).
           목록 <out>/manifest_<split>.json [{idx, cid, family, level, ftype, …}] · 렌더 설정 <out>/config_render.json (분할마다 n = 쓴 번호 수).
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P design_families_v5mc.py --config configs/v5mc.yaml --split s_speed --procs 80 --part 0 --nparts 3
  $P design_families_v5mc.py --config configs/v5mc.yaml --select-all
"""
import argparse
import collections
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_design_v5mc as G  # noqa: E402

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
        specs = G.design_family(cfg, split, f, max_tries=int(cfg["checks"].get("max_tries", 400)), streams=int(cfg["checks"].get("design_streams", 3)))
    except RuntimeError as e:
        open(p + ".fail", "w").write(str(e))
        return 2
    _dump(specs, p + ".json")
    return 1


def select(cfg_path):
    import yaml
    cfgy = yaml.safe_load(open(cfg_path))
    out = cfgy["out"]
    os.makedirs(f"{out}/design", exist_ok=True)
    S = G.situations(cfgy); NS = len(S)
    rep = {}
    for split, sp in cfgy["splits"].items():
        NC, NF = int(sp["candidates"]), int(sp["families"])
        per = NF // NS
        chosen, why_all, per_sit = [], collections.Counter(), {}
        for si, sit in enumerate(S):
            cands = [j * NS + si for j in range((NC + NS - 1) // NS) if j * NS + si < NC]
            ok = [f for f in cands if os.path.exists(f"{out}/design_fam/{split}_f{f:06d}.json")]
            for f in cands:
                fp = f"{out}/design_fam/{split}_f{f:06d}.fail"
                if os.path.exists(fp):
                    m = re.search(r"이유 (\{.*\})", open(fp).read())
                    if m:
                        for k, v in eval(m.group(1)).items():
                            why_all[re.sub(r"_k.*$", "_k", k)] += v
            pick = ok[:per]
            chosen += pick
            per_sit[G.sit_key(sit)] = {"candidates": len(cands), "ok": len(ok), "picked": len(pick)}
        man, idx = [], 0
        for f in sorted(chosen):
            specs = json.load(open(f"{out}/design_fam/{split}_f{f:06d}.json"))
            keys = [k for k in specs if not k.startswith("B")] + [k for k in specs if k.startswith("B")]
            for k in keys:
                s_ = specs[k]
                s_["idx"] = idx
                cid = f"{split}{sp['seed']}_{idx:06d}"
                _dump(s_, f"{out}/design/{cid}.json")
                c = s_["cell"]
                man.append({"idx": idx, "cid": cid, "family": f, "level": k, "ftype": sp["ftype"], "place": c["place"], "objset": c["objset"],
                            "camera": c["camera"], "occ": c["occ"], "ramp_mode": c.get("ramp_mode"), "b_for": s_.get("b_for"),
                            "level_info": s_.get("level_info"), "event": s_.get("event_samples"), "v_hide": s_.get("v_hide_mps"), "kind": s_.get("motion_kind")})
                idx += 1
        json.dump(man, open(f"{out}/manifest_{split}.json", "w"), default=str)
        sp["n"] = idx
        rep[split] = {"chosen": len(chosen), "target": NF, "clips": idx, "situations": per_sit, "fail_reasons": dict(why_all)}
        print(f"{split}: 고른 묶음 {len(chosen)} / 목표 {NF} · 렌더 영상 {idx} · 실패 이유 합 {dict(why_all.most_common(6))}", flush=True)
        for k, v in per_sit.items():
            if v["picked"] < per:
                print(f"   모자람 {k}: 후보 {v['candidates']} 성공 {v['ok']} 고름 {v['picked']} / {per}", flush=True)
    json.dump(cfgy, open(f"{out}/config_render.json", "w"), indent=1)
    json.dump(rep, open(f"{out}/select_report.json", "w"), indent=1, ensure_ascii=False)
    tot_train = sum(r["clips"] for s, r in rep.items() if s.startswith("s_"))
    print(f"학습 영상 합 {tot_train} · 시험 {sum(r['clips'] for s, r in rep.items() if s.startswith('t_'))} · 확인 {sum(r['clips'] for s, r in rep.items() if s.startswith('p_'))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--split")
    ap.add_argument("--procs", type=int, default=32); ap.add_argument("--part", type=int, default=0); ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--select-all", action="store_true")
    a = ap.parse_args()
    if a.select_all:
        return select(a.config)
    cfg = G.load_cfg(a.config, a.split)
    os.makedirs(f"{cfg['out']}/design_fam", exist_ok=True)
    fams = [f for f in range(int(cfg["splits"][a.split]["candidates"])) if f % a.nparts == a.part]
    t0, c = time.time(), [0, 0, 0]
    with ProcessPoolExecutor(a.procs, initializer=_init, initargs=(a.config, a.split)) as ex:
        for k, r in enumerate(ex.map(one, fams, chunksize=1)):
            c[r] += 1
            if k % 500 == 0:
                print(f"{a.split} part {a.part}/{a.nparts}: {k}/{len(fams)} 성공 {c[1]} 실패 {c[2]} 건너뜀 {c[0]} | {time.time() - t0:.0f}s", flush=True)
    print(f"끝 {a.split} part {a.part}: {len(fams)} · 성공 {c[1]} · 실패 {c[2]} · 건너뜀 {c[0]} | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
