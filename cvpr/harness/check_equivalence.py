#!/usr/bin/env python3
"""새 resolver (cvpr/harness/resolve.py) 가 옛 resolver (z_research/scripts/harness/resolve.py) 와 **같은 config** 를 내는지 검사.

    python cvpr/harness/check_equivalence.py            # 기본 행렬
    python cvpr/harness/check_equivalence.py -v         # 차이를 전부 출력

행마다: 옛 `resolve.py <proto> <ds> <model> --set ...` vs 새 `resolve.py <cvpr proto> <ds> <model> --results legacy --window ... --set ...`.
`--results legacy` 라 tag · output_dir 까지 같아야 한다. 차이가 0 이면 옛 결과 (summary.json · 토큰 캐시) 와 그대로 이어진다.
GPU · 모델 로딩 없음 (몇 초).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

CVPR = Path(__file__).resolve().parents[1]
REPO = CVPR.parent
PY = sys.executable
OLD = REPO / "z_research/scripts/harness/resolve.py"
NEW = CVPR / "harness/resolve.py"

# (설명, 옛 프로토콜, 데이터셋, 모델, 옛 --set 목록, 새 프로토콜, 새 --window, 새 --set 목록)
ROWS = [
    ("testbed v11 vith",            "surprise_c16t32", "v11", "vith", [], "eval/testbed", None, []),
    ("testbed v11 vitl",            "surprise_c16t32", "v11", "vitl", [], "eval/testbed", None, []),
    ("testbed v11_realistic vith",  "surprise_c16t32", "v11_realistic", "vith", [], "eval/testbed", None, []),
    ("testbed rollout_v2 vith",     "surprise_c16t32", "rollout_v2", "vith", [], "eval/testbed", None, []),
    ("testbed gravity cross (레지스트리 overrides)", "surprise_c16t32", "gravity_realistic", "vith", ["scoring.pairing=cross"], "eval/testbed", None, []),
    ("testbed v11 ariel prefix",    "surprise_c16t32", "v11", "vith_ariel_prefix_ep45", [], "eval/testbed", None, []),
    ("testbed v11 pft",             "surprise_c16t32", "v11", "vith_pft_v11_e10", [], "eval/testbed", None, []),
    ("testbed v11 vjepa21g",        "surprise_c16t32", "v11", "vjepa21g", [], "eval/testbed", None, []),
    ("testbed v11 videomae2g c8t16", "surprise_c16t32", "v11", "videomae2g",
        ["data.n_frames=16", "data.frames_stride=6", "surprise.context_length=8", "model.window_size=16"],
        "eval/testbed", "c8t16_s6", []),
    ("testbed v11 custom c8t32 via set", "surprise_c16t32", "v11", "vith", ["surprise.context_length=8"],
        "eval/testbed", None, ["window.context=8"]),
    ("intphys1 full grid",          "intphys1_sliding", "intphys1_dev", "vith", [], "eval/intphys1", None, []),
    ("intphys1 w32 (run_all 칸)",    "intphys1_sliding", "intphys1_dev", "vith",
        ["surprise.intphys1.frame_skips=[2,5,10]", "surprise.intphys1.window_sizes=[32]", "model.window_size=32"],
        "eval/intphys1", "garrido_w32", []),
    ("intphys1 w16 vitl",           "intphys1_sliding", "intphys1_dev", "vitl",
        ["surprise.intphys1.frame_skips=[2,5,10]", "surprise.intphys1.window_sizes=[16]", "model.window_size=16"],
        "eval/intphys1", "garrido_w16", []),
    ("grasp w32 vjepa21g",          "intphys1_sliding", "grasp_level2", "vjepa21g",
        ["surprise.intphys1.frame_skips=[5,10]", "surprise.intphys1.window_sizes=[32]", "model.window_size=32", "surprise.intphys1.max_batch=8"],
        "eval/intphys1", "garrido_w32", ["surprise.intphys1.max_batch=8"]),
    ("inflevel_gravity w16 videomae2g", "intphys1_sliding", "inflevel_gravity", "videomae2g",
        ["surprise.intphys1.frame_skips=[5,10,20]", "surprise.intphys1.window_sizes=[16]", "model.window_size=16"],
        "eval/intphys1", "garrido_w16", []),
    ("testbed v11 dinof (chain_1006 SET → 레지스트리)", "surprise_c16t32", "v11", "dinof_highres",
        ["surprise.batch_size=8", "surprise.decode_workers=8"], "eval/testbed", None, []),
    ("testbed v11 dinof copy",      "surprise_c16t32", "v11", "dinof_highres",
        ["surprise.batch_size=8", "surprise.decode_workers=8", "model.dinof_copy=true"], "eval/testbed", None, ["model.dinof_copy=true"]),
    ("intphys1 dinof w8 (chain_1002)", "intphys1_sliding", "intphys1_dev", "dinof_highres",
        ["surprise.intphys1.frame_skips=[2,5,10]", "surprise.intphys1.window_sizes=[8]", "surprise.intphys1.context_mult=[8]",
         "model.window_size=8", "surprise.intphys1.max_batch=16", "surprise.intphys1.video_batch=2",
         "surprise.batch_size=8", "surprise.decode_workers=8"],   # 뒤 둘은 레지스트리 (dinof 자원 노브) 가 넣는 값 — sliding 에선 미사용
        "eval/intphys1", "dinof_w8", []),
    ("probing v11",                 "attn_probe", "v11", "vith", [], "analysis/probing/attn_probe", None, []),
    ("probing v11_full xfer",       "attn_probe_xfer", "v11_full", "vith", [], "analysis/probing/attn_probe_xfer", None, []),
    ("probing v11 imp",             "attn_probe_imp", "v11", "vith", [], "analysis/probing/attn_probe_imp", None, []),
    ("probing v11 set optim",       "attn_probe", "v11", "vith", ["probing.optim=attn_50", "probing.targets.shape=null"],
        "analysis/probing/attn_probe", None, ["probing.optim=attn_50", "probing.targets.shape=null"]),
]

IGNORE = set()   # 비교에서 뺄 최상위 키 (없음 — legacy 모드는 tag/output_dir 까지 같아야 한다)

def _ip2_sets(w, mb):
    C = [int(w * f) for f in (0.25, 0.375, 0.5, 0.625, 0.75, 0.875)]
    return [f"surprise.window_size={w}", f"surprise.context_length_sweep={C}", f"surprise.context_length={w // 2}",
            f"surprise.max_window_batch={mb}", "surprise.video_batch=1"]

# 통짜 yaml 하네스 (옛 analysis/intphys2 · EK100): 옛 yaml + 옛 --set  vs  새 resolve. 이름 (tag/folder) 은 설계상 다르다 → 무시.
#   model.window_size: 옛 intphys2 eval 은 실행 때 setdefault 로 채운다 → 무시
MONO_ROWS = [
    ("intphys2 vith w48 (run_grid)", "analysis/intphys2/configs/bench_intphys2_main_vith.yaml", _ip2_sets(48, 8),
        "eval/intphys2", "intphys2_main", "vith", "ip2_w48", [f"surprise.max_window_batch=8"], {"tag", "folder", "model.window_size"}),
    ("intphys2 vith w16", "analysis/intphys2/configs/bench_intphys2_main_vith.yaml", _ip2_sets(16, 16),
        "eval/intphys2", "intphys2_main", "vith", "ip2_w16", ["surprise.max_window_batch=16"], {"tag", "folder", "model.window_size"}),
    ("intphys2 vjepa21g w32", "analysis/intphys2/configs/bench_intphys2_main_vjepa21g.yaml", _ip2_sets(32, 8),
        "eval/intphys2", "intphys2_main", "vjepa21g", "ip2_w32", ["surprise.max_window_batch=8"], {"tag", "folder", "model.window_size"}),
    ("intphys2 videomae2g w16 fps6", "analysis/intphys2/configs/bench_intphys2_main_videomae2g.yaml", _ip2_sets(16, 16),
        "eval/intphys2", "intphys2_main", "videomae2g", "ip2_vmae_fps6", ["surprise.max_window_batch=16"], {"tag", "folder", "model.window_size"}),
    ("ek100 vith (ek100_vith.yaml)", "z_research/anticipation/EK100/configs/ek100_vith.yaml", [],
        "eval/ek100", "ek100", "vith", None, [], {"tag", "folder"}),
    ("ek100 vith heads=grid8", "z_research/anticipation/EK100/configs/ek100_vith.yaml", ["__heads__=grid8"],
        "eval/ek100", "ek100", "vith", None, ["heads=grid8"], {"tag", "folder"}),
]
EK_LRS = {"sweep": ([5e-3, 3e-3, 1e-3, 3e-4, 1e-4], [1e-4, 1e-3, 1e-2, 1e-1]), "grid8": ([3e-4, 1e-4], [1e-4, 1e-3, 1e-2, 1e-1]), "hi2": ([1e-3, 3e-3], [1e-2])}


def legacy_set(cfg, kv):
    """옛 _apply_sets / EK100 set_path 의미. __heads__ 는 EK100 resolve --heads 흉내."""
    k, v = kv.split("=", 1)
    if k == "__heads__":
        lrs, wds = EK_LRS[v]
        cfg["experiment"]["optimization"]["multihead_kwargs"] = [
            dict(lr=lr, start_lr=lr, final_lr=0.0, weight_decay=wd, final_weight_decay=wd, warmup=0.0) for wd in wds for lr in lrs]
        return
    if k.split(".")[0] in ("data", "optimization", "classifier", "evaluation") and "experiment" in cfg:
        k = "experiment." + k
    val = yaml.safe_load(v)
    parts = k.split("."); cur = cfg
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    if val is None:
        cur.pop(parts[-1], None)
    else:
        cur[parts[-1]] = val


def run(cmd, env=None):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return r.returncode, r.stdout, r.stderr


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--only", default=None, help="설명에 이 문자열이 든 행만")
    a = ap.parse_args()
    env = {k: v for k, v in os.environ.items() if k not in ("TAG", "OUTDIR", "RESULTS_ROOT")}
    n_ok = n_bad = n_skip = 0
    with tempfile.TemporaryDirectory() as td:
        for i, (desc, op, ds, m, osets, np_, win, nsets) in enumerate(ROWS):
            if a.only and a.only not in desc:
                continue
            fa, fb = f"{td}/old_{i}.yaml", f"{td}/new_{i}.yaml"
            ca = [PY, str(OLD), op, ds, m, "-o", fa, "--quiet"] + sum((["--set", s] for s in osets), [])
            cb = [PY, str(NEW), np_, ds, m, "-o", fb, "--quiet", "--results", "legacy"] + \
                 (["--window", win] if win else []) + sum((["--set", s] for s in nsets), [])
            ra, oa, ea = run(ca, env)
            rb, ob, eb = run(cb, env)
            if ra != 0 or rb != 0:
                tag = "SKIP(old 실패)" if ra != 0 and rb != 0 else ("BAD(old 만 실패)" if ra != 0 else "BAD(new 실패)")
                if tag.startswith("SKIP"):
                    n_skip += 1
                else:
                    n_bad += 1
                print(f"[{tag:>16}] {desc}")
                if a.verbose or not tag.startswith("SKIP"):
                    print("    old:", (ea.strip().splitlines() or [""])[-1][:200])
                    print("    new:", (eb.strip().splitlines() or [""])[-1][:200])
                continue
            A, B = flat(yaml.safe_load(open(fa))), flat(yaml.safe_load(open(fb)))
            keys = sorted((set(A) | set(B)) - IGNORE)
            diff = [(k, A.get(k, "<없음>"), B.get(k, "<없음>")) for k in keys if A.get(k) != B.get(k)]
            suffix_only = diff and all(k in ("tag", "output_dir") and isinstance(va, str) and isinstance(vb, str)
                                       and vb.startswith(va) and vb[len(va):].startswith("_w") for k, va, vb in diff)
            if suffix_only:
                n_ok += 1
                print(f"[{'OK(창 접미사)':>16}] {desc}  ({len(keys) - len(diff)} 키 동일, tag/output_dir 에 {diff[0][2][len(diff[0][1]):]})")
            elif diff:
                n_bad += 1
                print(f"[{'DIFF':>16}] {desc}  ({len(diff)} 키)")
                for k, va, vb in diff[: (None if a.verbose else 8)]:
                    print(f"    {k}: old={va!r}  new={vb!r}")
            else:
                n_ok += 1
                print(f"[{'OK':>16}] {desc}  ({len(keys)} 키 동일)")
        for desc, oyaml, osets, np_, ds, m, win, nsets, ign in MONO_ROWS:
            if a.only and a.only not in desc:
                continue
            A0 = yaml.safe_load(open(REPO / oyaml))
            for kv in osets:
                legacy_set(A0, kv)
            fb = f"{td}/mono_{desc[:8]}.yaml"
            cb = [PY, str(NEW), np_, ds, m, "-o", fb, "--quiet", "--results", "legacy"] + \
                 (["--window", win] if win else []) + sum((["--set", s] for s in nsets), [])
            rb, ob, eb = run(cb, env)
            if rb != 0:
                n_bad += 1; print(f"[{'BAD(new 실패)':>16}] {desc}\n    new: {(eb.strip().splitlines() or [''])[-1][:200]}"); continue
            A, B = flat(A0), flat(yaml.safe_load(open(fb)))
            keys = sorted(k for k in (set(A) | set(B)) if k not in ign)
            diff = [(k, A.get(k, "<없음>"), B.get(k, "<없음>")) for k in keys if A.get(k) != B.get(k)]
            if diff:
                n_bad += 1
                print(f"[{'DIFF':>16}] {desc}  ({len(diff)} 키)")
                for k, va, vb in diff[: (None if a.verbose else 8)]:
                    print(f"    {k}: old={va!r}  new={vb!r}")
            else:
                n_ok += 1
                print(f"[{'OK':>16}] {desc}  ({len(keys)} 키 동일, 무시 {sorted(ign)})")
    print(f"\n동일 {n_ok} · 차이/실패 {n_bad} · 건너뜀 {n_skip}")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
