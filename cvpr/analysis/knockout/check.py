#!/usr/bin/env python3
"""knockout 경로의 config 동일성 검사 — GPU · 모델 로딩 없음 (옛 resolver 호출 + torch import 로 20 초 안팎).

run.sh 는 cvpr resolver 로 <out>/_resolved.yaml 을 만들고 그 **경로**를 run.py --protocol 로 넘긴다. run.py 는
analysis/attention/predictor_attn/extract.py:resolve_config() 으로 옛 resolver (z_research/scripts/harness/resolve.py) 를
다시 부르는데, 옛 resolver 는 yaml 경로를 받으면 그 위에 옛 md 레지스트리 (configs/protocols/*.md) 를 한 번 더 깔고
TAG/OUTDIR env 로 이름을 다시 짓는다. 그래서 세 config 를 비교한다:

  A  cvpr   resolve.py cvpr/analysis/knockout/protocol.yaml <ds> <model> [--results legacy] [--window/--set/--limit ...]
  B  legacy resolve_config(<A 의 yaml 경로>, ds, model, [])  with env TAG/OUTDIR = A 의 meta      <- run.py 가 실제로 보는 것
  C  legacy resolve_config("surprise_c16t32", ds, model, ["data.type_column=condition", ...])     <- 옛 run_sharded.sh 가 보던 것

  A == B  이면 run.py 가 cvpr 의 config 그대로 돈다 (전 키 — tag/output_dir 까지).
  A == C  이면 옛 실행과 같은 의미다 (tag/output_dir 는 설계상 다르므로 뺀다; --results legacy 면 그것까지 같아야 한다).
          C 는 --window/--limit/--suffix 가 없을 때만 돈다 (옛 명령에 그 대응이 없다). --set 은 C 에도 그대로 얹는다.

    python cvpr/analysis/knockout/check.py                        # 기본 행렬: v11_full/vith · v11/vith · v11/vitl
    python cvpr/analysis/knockout/check.py v11_full vith --results legacy
    python cvpr/analysis/knockout/check.py v11 vith --set scoring.pairing=cross --window c8t32 --limit 2
    python cvpr/analysis/knockout/check.py -v                     # 차이를 전부 출력
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CVPR = HERE.parents[1]
REPO = CVPR.parent
sys.path.insert(0, str(REPO))
NEW = CVPR / "harness/resolve.py"
PROTO = HERE / "protocol.yaml"
OLD_PROTO, OLD_SETS = "surprise_c16t32", ["data.type_column=condition"]     # 옛 run_sharded.sh 가 run.py 에 준 것
ROWS = [("v11_full", "vith"), ("v11", "vith"), ("v11", "vitl")]
NAME_KEYS = {"tag", "output_dir"}


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def diff(A, B, ignore=()):
    fa, fb = flat(A), flat(B)
    keys = sorted((set(fa) | set(fb)) - set(ignore))
    return [(k, fa.get(k, "<없음>"), fb.get(k, "<없음>")) for k in keys if fa.get(k) != fb.get(k)], len(keys)


def report(label, d, n, verbose):
    if d:
        print(f"[{'DIFF':>8}] {label}  ({len(d)} / {n} 키)")
        for k, va, vb in d[: (None if verbose else 8)]:
            print(f"      {k}: {va!r}  vs  {vb!r}")
        return 1
    print(f"[{'OK':>8}] {label}  ({n} 키 동일)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", nargs="?"); ap.add_argument("model", nargs="?")
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="K=V")
    ap.add_argument("--window"); ap.add_argument("--limit", type=int); ap.add_argument("--suffix")
    ap.add_argument("--results", default="cvpr", help="cvpr | legacy | <경로>")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    rows = [(a.dataset, a.model or "vith")] if a.dataset else ROWS
    plain = not (a.window or a.limit or a.suffix)

    from analysis.attention.predictor_attn.extract import resolve_config   # run.py 가 쓰는 바로 그 함수 (torch import, 모델 없음)

    def legacy(proto, ds, m, sets, tag=None, outdir=None):
        env_bak = {k: os.environ.pop(k, None) for k in ("TAG", "OUTDIR")}
        if tag:
            os.environ["TAG"], os.environ["OUTDIR"] = tag, outdir
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                return resolve_config(proto, ds, m, sets)
        finally:
            for k, v in env_bak.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v

    bad = 0
    with tempfile.TemporaryDirectory() as td:
        for ds, m in rows:
            label = f"{ds}/{m}" + (f" --window {a.window}" if a.window else "") + (f" --limit {a.limit}" if a.limit else "") \
                + (f" --suffix {a.suffix}" if a.suffix else "") + (f" --results {a.results}" if a.results != "cvpr" else "") \
                + "".join(f" --set {s}" for s in a.sets)
            fa, fm = f"{td}/{ds}_{m}.yaml", f"{td}/{ds}_{m}.json"
            cmd = [sys.executable, str(NEW), str(PROTO), ds, m, "-o", fa, "--meta", fm, "--quiet", "--results", a.results]
            cmd += sum((["--set", s] for s in a.sets), [])
            if a.window:
                cmd += ["--window", a.window]
            if a.limit:
                cmd += ["--limit", str(a.limit)]
            if a.suffix:
                cmd += ["--suffix", a.suffix]
            env = {k: v for k, v in os.environ.items() if k not in ("TAG", "OUTDIR", "RESULTS_ROOT")}
            r = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if r.returncode:
                print(f"[{'BAD':>8}] {label}  cvpr resolve 실패: {(r.stderr.strip().splitlines() or [''])[-1][:200]}"); bad += 1; continue
            A, meta = yaml.safe_load(open(fa)), json.load(open(fm))
            assert A["tag"] == meta["tag"] and A["output_dir"] == meta["output_dir"], "meta 와 yaml 의 이름이 다르다"

            # B: run.py 가 보는 것 — 옛 resolver 에 A 의 yaml 경로 + TAG/OUTDIR
            B = legacy(fa, ds, m, [], meta["tag"], meta["output_dir"])
            bad += report(f"A(cvpr) == B(run.py 가 다시 푼 _resolved.yaml)   {label}", *diff(A, B), a.verbose)

            # C: 옛 run_sharded.sh 와 같은 의미인가
            if plain:
                C = legacy(OLD_PROTO, ds, m, OLD_SETS + a.sets)
                # 옛 run.py 의 outdir 기본값은 cfg.output_dir 의 **부모**/knockout__<ds>_<model> 이었다 (run.py: Path(cfg["output_dir"]).parent / ...).
                # 옛 config 의 output_dir 자체 (surprise_c16t32__...) 는 아무도 안 쓰므로 그 파생값으로 바꿔 비교한다.
                C["output_dir"] = str(Path(C["output_dir"]).parent / f"knockout__{ds}_{m}")
                ign = () if a.results == "legacy" else NAME_KEYS
                d, n = diff(A, C, ign)
                bad += report(f"A(cvpr) == C(옛 surprise_c16t32 + --set type_column){'' if ign else ' [이름까지]'}   {label}", d, n, a.verbose)
                if ign:
                    print(f"           (이름은 설계상 다르다: A {A['output_dir']}  /  C {C['output_dir']})")
            else:
                print(f"[{'SKIP':>8}] A == C  — --window/--limit/--suffix 는 옛 명령에 대응이 없다")
    print(f"\n{'전부 동일' if not bad else f'차이/실패 {bad}'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
