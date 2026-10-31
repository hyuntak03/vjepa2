#!/usr/bin/env python3
"""predictor_lab 채점표 — Google 시트 (2027_CVPR_world_model, Overall 탭) 와 같은 행 이름 · 열 이름 · 열 순서 (2026-10-31).

  python predictor_lab/table.py <run> [<run> …]          # 기준 4 행 + run (z_training/runs/<run>/eval_lab/ = eval.sh 결과)
  python predictor_lab/table.py <run> --tsv               # 시트 B–R 열에 그대로 붙일 TSV (탭 구분, P 열은 빈 칸)
  python predictor_lab/table.py --check                   # 기준 4 행을 기존 결과 폴더에서 다시 계산해 시트 값과 대조
  python predictor_lab/table.py <run> --no-refs --json out.json

시트 열 (B–R)
  B 이름 · C encoder · D predictor · E predictor params · F Predictor type · G Step · H Seen clip · I Data scale · J Epoch
  K VOBench P · L VOBench R · M VOBench Gravity P · N VOBench Gravity R · O rubric office · P (빈 칸) · Q IntPhys 1 · R IntPhys1 late occ
점수 (% , 산출물 per_block.json / per_window.json 에서 다시 계산)
  VOBench P / R           VOBench_v1 split v2 test late 쌍 정답률 — primitive 108 · realistic 108 (matched, 동점 0.5)
  VOBench Gravity P / R   gravity test block 중 진짜 미래의 surprise 가 가장 작은 비율 — primitive 54 · realistic 54 (ledge 2 지선다 · arc 9 지선다)
  rubric office           실제 영상 40 쌍 정답률 (사라짐 20 + 생겨남 20, 균일 창)
  IntPhys 1               Garrido A.8 격자 (창 16 · 32 × skip 2 · 5 · 10) Filtered 최고 칸 — descriptive
  IntPhys1 late occ       균일 창 판에서 가림 쌍 90 개 정답률
run 의 C–J 는 run 폴더 (config.yaml · latest.pt · train.log) 에서 읽는다. 이름은 config 의 lab.sheet_name (없으면 run 이름).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))

META_COLS = ["encoder", "predictor", "predictor params", "Predictor type", "Step", "Seen clip", "Data scale", "Epoch"]
SCORE_COLS = ["VOBench P", "VOBench R", "VOBench Gravity P", "VOBench Gravity R", "rubric office", "IntPhys 1", "IntPhys1 late occ"]


def _env():
    """paths.env 를 읽는다 (export K=V, $LAB_REPO 치환)."""
    env = {}
    for line in open(HERE / "paths.env"):
        line = line.split("#", 1)[0].strip()
        if line.startswith("export ") and "=" in line:
            k, v = line[7:].split("=", 1)
            env[k.strip()] = v.strip().replace("$LAB_REPO", env.get("LAB_REPO", ""))
    return env


ENV = _env()


# ----------------------------------------------------------------------------- 점수
def _index(res: Path):
    import yaml
    d = yaml.safe_load(open(res / "_resolved.yaml"))["data"]
    return {r["video_id"]: r for r in csv.DictReader(open(os.path.join(d["root"], d.get("index_csv", "index.csv"))))}


def pair_acc(res: Path, keep=lambda r: True, group=lambda r: "all", ids=None):
    """per_block.json 의 영상별 surprise → 그룹별 (정답률 %, 쌍 수). 쌍 = 같은 (objset, block_id, pair_id) 의 가능 1 + 불가능 1."""
    pv = json.load(open(res / "per_block.json"))["per_video_surprise"]
    meta = _index(res) if ids is None else ids
    by = defaultdict(dict)
    for vid, r in meta.items():
        if vid in pv and keep(r):
            by[(r.get("objset", ""), r["block_id"], r["pair_id"])]["pos" if str(r["plausible"]) == "1" else "imp"] = (vid, r)
    out = defaultdict(list)
    for v in by.values():
        if "pos" in v and "imp" in v:
            sp, si = pv[v["pos"][0]], pv[v["imp"][0]]
            out[group(v["pos"][1])].append(1.0 if si > sp else 0.5 if si == sp else 0.0)
    return {g: (100.0 * sum(a) / len(a), len(a)) for g, a in out.items()}


def vobench_late(res: Path):
    late = {r["video_id"]: r for r in csv.DictReader(open(ENV["VOB_INDEX"])) if r["occlusion"] == "late"}
    g = pair_acc(res, ids=late, group=lambda r: r["objset"])
    return g.get("primitive"), g.get("realistic")


def gravity_pick(res: Path):
    """gravity test block 마다 진짜 미래 (가능) 의 surprise < 모든 불가능 미래 → 정답. objset 별 (%, block 수)."""
    pv = json.load(open(res / "per_block.json"))["per_video_surprise"]
    by = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(ENV["GRAV_INDEX_DIR"], "index_test.csv"))):
        by[r["block_id"]].append(r)
    acc = defaultdict(list)
    for rs in by.values():
        pos = [r for r in rs if r["plausible"] == "1"]
        imp = [r for r in rs if r["plausible"] == "0"]
        if len(pos) != 1 or not imp or any(r["video_id"] not in pv for r in rs):
            continue
        acc[pos[0]["objset"]].append(float(pv[pos[0]["video_id"]] < min(pv[r["video_id"]] for r in imp)))
    f = lambda k: (100.0 * sum(acc[k]) / len(acc[k]), len(acc[k])) if acc.get(k) else None
    return f("primitive"), f("realistic")


def intphys1(dirs):
    import garrido_rescore as GR
    combos, kind = {}, None
    for d in dirs:
        r = GR.rescore(Path(d)); kind = r["dataset_kind"]; combos.update(r["combos"])
    b = GR.select(combos, kind)
    return b["accuracy"], b["n_pair"], b["combo"]


def intphys1_late_occ(res: Path):
    occ = {r["video_id"] for r in csv.DictReader(open(ENV["IP1L_OCC_LIST"]))}
    return pair_acc(res, keep=lambda r: r["video_id"] in occ)["all"]


def te_rubric(te_dir: Path, tag: str):
    """한 번 통과 채점기 (transfer_gap_te) 배열에서 Rubric 쌍 정답률 (기준 행 대조용)."""
    import numpy as np
    m = json.load(open(te_dir / "meta.json"))
    s = np.load(te_dir / "l1.npy").mean(-1)[:, m["pred_tags"].index(tag)]
    by = defaultdict(dict)
    for i in range(len(m["video_id"])):
        by[(m["block_id"][i], m["pair_id"][i])]["pos" if str(m["plausible"][i]) == "1" else "imp"] = i
    a = [1.0 if s[v["imp"]] > s[v["pos"]] else 0.5 if s[v["imp"]] == s[v["pos"]] else 0.0 for v in by.values()]
    return 100.0 * sum(a) / len(a), len(a)


def scores(spec: dict):
    """spec: {vobench, gravity, intphys1: [w16, w32], intphys1_late, rubric | {te, tag}} (레포 기준 경로) → (열 값, 메모)."""
    out, note = {}, {}

    def _try(key, fn):
        try:
            fn()
        except FileNotFoundError as e:
            note[key] = f"없음 {e.filename}"

    def vob():
        p, r = vobench_late(REPO / spec["vobench"])
        out["VOBench P"], out["VOBench R"] = (p and p[0]), (r and r[0]); note["vobench_n"] = [p and p[1], r and r[1]]

    def grav():
        p, r = gravity_pick(REPO / spec["gravity"])
        out["VOBench Gravity P"], out["VOBench Gravity R"] = (p and p[0]), (r and r[0]); note["gravity_n"] = [p and p[1], r and r[1]]

    def rub():
        rb = spec["rubric"]
        a, n = te_rubric(REPO / rb["te"], rb["tag"]) if isinstance(rb, dict) else pair_acc(REPO / rb)["all"]
        out["rubric office"] = a; note["rubric_n"] = n

    def ip1():
        a, n, combo = intphys1([REPO / d for d in spec["intphys1"]]); out["IntPhys 1"] = a; note["intphys1"] = f"{combo} n={n}"

    def ip1l():
        a, n = intphys1_late_occ(REPO / spec["intphys1_late"]); out["IntPhys1 late occ"] = a; note["intphys1_late_n"] = n

    for key, fn in (("vobench", vob), ("gravity", grav), ("rubric", rub), ("intphys1", ip1), ("intphys1_late", ip1l)):
        if spec.get(key):
            _try(key, fn)
    return out, note


# ----------------------------------------------------------------------------- run 폴더 → 시트 행
def _run_dirs(run: str):
    """<run> 이름 · run 폴더 · eval 폴더 무엇을 줘도 (run 폴더, eval 폴더)."""
    p = (Path(run) if os.path.isdir(run) else REPO / "z_training/runs" / run).resolve()
    if p.name.startswith("eval_lab"):
        return p.parent, p
    return p, p / "eval_lab"


def run_spec(e: Path):
    rel = lambda q: os.path.relpath(q, REPO)
    spec = {k: rel(e / sub) for k, sub in (("vobench", "vobench_late"), ("gravity", "gravity_test"),
                                           ("intphys1_late", "intphys1_late_uni"), ("rubric", "rubric_uniform"))
            if (e / sub / "per_block.json").exists()}
    w = [rel(e / f"intphys1_dev_garrido_{x}") for x in ("w16", "w32") if (e / f"intphys1_dev_garrido_{x}" / "per_window.json").exists()]
    if w:
        spec["intphys1"] = w
    return spec


def _k(x):
    return f"{x / 1000:.1f}K" if x >= 1000 else str(int(x))


def _m(x):
    return f"{x / 1e6:.3g}M" if x >= 1e6 else (f"{x / 1e6:.3f}".rstrip("0").rstrip(".") + "M")


def run_meta(rundir: Path):
    """시트 C–J 를 run 폴더에서: encoder (고정 X · LoRA) · predictor O · params · type · Step · Seen clip (= step × 전역 배치) · Data scale (학습 clip 수) · Epoch."""
    import yaml
    meta = {c: "-" for c in META_COLS}
    cfgp = rundir / "config.yaml"
    if not cfgp.exists():
        return rundir.name, meta
    cfg = yaml.safe_load(open(cfgp))
    lab = cfg.get("lab") or {}
    meta["encoder"] = "LoRA" if cfg.get("lora") else "X"
    meta["predictor"] = "O"
    meta["Predictor type"] = lab.get("predictor_type", "future mask latent prediction")
    step = epoch = ipe = None
    ws = 1
    ck = rundir / "latest.pt"
    if ck.exists():
        import torch
        d = torch.load(ck, map_location="cpu", weights_only=False)
        n = sum(v.numel() for v in (d.get("predictor") or {}).values() if hasattr(v, "numel"))
        meta["predictor params"] = f"{n / 1e6:.0f}M" if n else "-"
        epoch, step = d.get("epoch"), d.get("step")
    log = rundir / "train.log"
    if log.exists():
        for line in open(log, errors="ignore"):
            if "rank batch" in line and "x ws" in line and "ipe" in line:   # "rank batch 16 x ws 2 = 32 | ipe 125 (loader 125)"
                try:
                    ws = int(line.split("x ws")[1].split("=")[0]); ipe = int(line.split("| ipe")[1].split()[0])
                except Exception:
                    pass
    D, O = cfg.get("data", {}), cfg.get("optimization", {})
    ep = epoch if epoch is not None else O.get("epochs")
    if step is None and ipe and ep:
        step = ipe * int(ep)
    gb = int(D.get("batch_size", 0)) * ws
    if step:
        meta["Step"] = _k(step)
        meta["Seen clip"] = _m(step * gb) if gb else "-"
    for s in D.get("datasets", []):
        if s.get("type") == "blender_track":
            import numpy as np
            n_clip = int(len(np.load(s["meta"])["clip"]))
            n_clip = min(n_clip, int(s["limit"])) if s.get("limit") else n_clip
            meta["Data scale"] = _k(n_clip) if n_clip < 1e5 else _m(n_clip)
    if ep is not None:
        meta["Epoch"] = str(ep)
    return lab.get("sheet_name") or rundir.name, meta


# ----------------------------------------------------------------------------- 출력
def fmt(v):
    return "-" if v is None else f"{v:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--check", action="store_true", help="기준 행을 기존 결과 폴더에서 다시 계산해 시트 값과 대조")
    ap.add_argument("--no-refs", action="store_true", help="기준 4 행을 빼고 run 만")
    ap.add_argument("--tsv", action="store_true", help="시트 B–R 에 붙일 TSV (P 열 빈 칸 포함)")
    ap.add_argument("--json")
    a = ap.parse_args()
    rows, report = [], {}
    if not a.no_refs or a.check:
        for r in json.load(open(HERE / "refs.json"))["rows"]:
            vals, note = scores(r["dirs"])
            rows.append((r["name"], dict(zip(META_COLS, r["meta"])), vals, r.get("sheet")))
            report[r["name"]] = {"scores": vals, "note": note, "sheet": r.get("sheet")}
    for run in a.runs:
        rundir, e = _run_dirs(run)
        name, meta = run_meta(rundir)
        vals, note = scores(run_spec(e))
        rows.append((name, meta, vals, None))
        report[name] = {"run": str(rundir), "eval": str(e), "meta": meta, "scores": vals, "note": note}
    if a.tsv:
        for name, meta, vals, _ in rows:
            print("\t".join([name] + [meta.get(c, "-") for c in META_COLS] + [fmt(vals.get(c)) for c in SCORE_COLS[:5]]
                            + [""] + [fmt(vals.get(c)) for c in SCORE_COLS[5:]]))
    else:
        print("| " + " | ".join(["이름"] + META_COLS + SCORE_COLS) + " |")
        print("|---|" + "---|" * len(META_COLS) + "---:|" * len(SCORE_COLS))
        for name, meta, vals, _ in rows:
            print(f"| {name} | " + " | ".join(meta.get(c, "-") for c in META_COLS) + " | "
                  + " | ".join(fmt(vals.get(c)) for c in SCORE_COLS) + " |")
    if a.check:
        print("\n검증: 다시 계산한 값 vs 시트 (|차| ≤ 0.05 면 OK)")
        bad = 0
        for name, _, vals, sheet in rows:
            for c in (SCORE_COLS if sheet else []):
                if sheet.get(c) is None:
                    continue
                v = vals.get(c)
                ok = v is not None and abs(v - sheet[c]) <= 0.05
                bad += not ok
                print(f"  {'OK  ' if ok else 'DIFF'} {name:26s} {c:18s} 계산 {fmt(v):>6s}  시트 {sheet[c]:.1f}")
        print(f"\n불일치 {bad} 칸")
    if a.json:
        json.dump(report, open(a.json, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
