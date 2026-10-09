#!/usr/bin/env python3
"""eval/intphys1/copy_baseline.py — IntPhys 1 (Garrido A.8 sliding) 복사 기준선을 **기존 스크립트**
z_research/scripts/analysis/te_intphys1_score.py 로 돌린다 (추출 GPU → 채점 CPU → copy_summary.json).

    python cvpr/eval/intphys1/copy_baseline.py --dataset intphys1_dev --model vith --gpus 8 [--limit N | --subset valid6] [--dryrun] [--validate]
    (보통은 copy_baseline.sh 가 부른다. 정의 · 노브 · 결과 위치는 README.md)

하는 일
  1. cvpr/harness/resolve.py (새 resolver, 프로토콜 eval/intphys1 = 창 garrido_a8 전체) 로 (데이터셋, 모델) 을 검사하고 이름을 받는다
       → 결과 폴더 = $CVPR_RESULTS/eval/intphys1/copy/<데이터셋>_<모델>[_smokeN|_valid6]     (--out / OUTDIR 로 덮는다)
     창 접미사는 없다 — te 는 격자 세 칸 (skip2_w16 · skip2_w32 · skip5_w16) 을 한 번에 뽑는다 (모델은 window 32 로 한 번 짓는다; 창별로 지은
     공식 실행과 비트 동일 — te docstring "수치 경로").
  2. te_intphys1_score.resolve_cfg() (옛 resolver) 의 config 와 1 의 config 의 data · model · surprise · scoring 블록을 대조한다.
  3. te_intphys1_score.main() 을 두 번 부른다: `--preds release` 로 추출 (release = z_training/runs/release_vith/latest.pt
     = model.pth predictor 추출본, te --selftest 가 state · p 비트 동일을 확인했다 — 문서 값) → `--score` (CPU, garrido_rescore 식 그대로).
     te 의 CACHE_ROOT (:163) · DOC_DIR (:164) 는 /data2 · z_research/TrainingEffects 가 리터럴이라 **부모 프로세스의 모듈 변수만** 결과 폴더로
     덮는다 (파일 수정 없음; spawn 된 rank `worker` (:504) 는 job["out"] 을 받는다).
  4. <out>/doc/<tag>_scores.json 에서 copy_summary.json (release · copy · Δ, 표준 · 인과 표적 × 칸 × O1/O2/O3 × 운동×가림) 을 만든다.

⚠️ intphys1_dev × vith 만 된다. te_intphys1_score.py:169 `S_TOK, D = 256, 1280` (ViT-H) 와 :248-249 의 `"intphys1_sliding", "intphys1_dev", "vith"`
   리터럴 때문이며, D 는 spawn 된 rank 가 모듈에서 다시 읽으므로 부모에서 덮어도 소용없다. 그 밖은 README 의 "엔진 수정 제안".
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent            # cvpr/eval/intphys1
CVPR = HERE.parents[1]
REPO = CVPR.parent
TE_DIR = REPO / "z_research/scripts/analysis"
sys.path.insert(0, str(CVPR / "harness"))
sys.path.insert(0, str(TE_DIR))
from resolve import load_env  # noqa: E402  (cvpr/harness/resolve.py 의 env.sh 로더)
import te_intphys1_score as te  # noqa: E402  (numpy · garrido_rescore 만 import; torch 는 함수 안에서)

SUPPORTED = ("intphys1_dev", "vith")              # te_intphys1_score.py:169 (D=1280) · :248-249 (리터럴)
PRED_TAG = "release"
CMP_KEYS = ("data", "model", "surprise", "scoring")


def cvpr_resolve(dataset: str, model: str, limit: int) -> tuple[dict, dict]:
    """새 resolver 를 subprocess 로 (병합 config, meta). 레지스트리 · available · 이름 규칙은 거기서 (복사 없음)."""
    with tempfile.TemporaryDirectory(prefix="cvpr_copy_") as td:
        o, m = Path(td) / "c.yaml", Path(td) / "m.json"
        args = [sys.executable, str(CVPR / "harness/resolve.py"), str(HERE / "protocol.yaml"), dataset, model,
                "-o", str(o), "--meta", str(m), "--quiet", "--results", "cvpr"]
        if limit:
            args += ["--limit", str(limit)]
        r = subprocess.run(args, text=True, capture_output=True)
        if r.returncode:
            sys.exit(f"ERROR: cvpr/harness/resolve.py 실패 ({dataset}, {model}):\n{r.stderr}")
        with open(o, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        with open(m, encoding="utf-8") as f:
            return cfg, json.load(f)


def cfg_diff(a: dict, b: dict) -> list[str]:
    out = []

    def walk(x, y, p):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                walk(x.get(k), y.get(k), f"{p}.{k}")
        elif x != y:
            out.append(f"{p}: cvpr={x!r} legacy={y!r}")

    for k in CMP_KEYS:
        walk(a.get(k), b.get(k), k)
    return out


def summarize(out: Path, dataset: str, model: str) -> dict:
    """te --score 산출 (<out>/doc/<tag>_scores*.json 중 최신) → copy_summary.json: release · copy · Δ 를 표적 × 칸 × 그룹으로."""
    files = sorted(glob.glob(str(out / "doc" / f"{out.name}_scores*.json")), key=os.path.getmtime)
    if not files:
        sys.exit(f"ERROR: {out}/doc 에 *_scores.json 이 없다 (te --score 가 안 끝났다)")
    with open(files[-1], encoding="utf-8") as f:
        sc = json.load(f)
    cells = sc["cells"]
    if PRED_TAG not in cells or "copy" not in cells:
        sys.exit(f"ERROR: scores json 에 {PRED_TAG!r} 또는 'copy' 칸이 없다: {list(cells)}")
    res = {"dataset": dataset, "model": model, "predictor": f"{PRED_TAG} (= model.pth predictor 추출본, z_training/runs/release_vith/latest.pt)",
           "n_videos": sc["n_videos"], "chance": 50.0, "rule": sc["rule"],
           "definition": {
               "model": "창마다 S = mean_j mean|p_j − h_{Tc+j}| (표준 표적 h = LN(target_encoder(창 W 장 전부)); 인과 표적 h^c_u = 튜블릿 u 까지만 본 것)",
               "copy": "창마다 S = mean_j mean|LN(z)[Tc−1] − h_{Tc+j}| — 예측 없음: 문맥 마지막 튜블릿을 미래 전부에 복사 (같은 창 · 같은 표적)",
               "cell": "Filtered = 시작점마다 C 최소 → AvgSurprise = 시작점 평균 → 쌍 1[G(pos) < G(imp)] (strict, 동점 오답) → property macro (O1/O2/O3)",
               "delta": "release − copy (pt, 같은 쌍). CI 는 여기서 내지 않는다 (scene bootstrap 은 te_analyze_ip1.py / IP1_COPY.md)",
               "best": "격자 최고 칸은 descriptive (같은 격자에서 고른 값, held-out 아님 — CLAUDE.md §1-3)"},
           "cells": {}, "best": {},
           "source": {"script": "z_research/scripts/analysis/te_intphys1_score.py", "scores_json": os.path.relpath(files[-1], out),
                      "arrays": sc.get("arrays")}}
    for tgt in ("std", "causal"):
        cm, cc = cells[PRED_TAG].get(tgt), cells["copy"].get(tgt)
        if cm is None or cc is None:
            continue
        res["best"][tgt] = {"model": cm["best"], "copy": cc["best"]}
        res["cells"][tgt] = {}
        for combo in cm["combos"]:
            a, b = cm["combos"][combo], cc["combos"][combo]
            row = {"model": round(a["macro"], 2), "copy": round(b["macro"], 2), "delta": round(a["macro"] - b["macro"], 2),
                   "n_pair": a["n_pair"], "n_ties": {"model": a["n_ties"], "copy": b["n_ties"]},
                   "alt_notebook": {"model": round(a["alt_notebook"], 2), "copy": round(b["alt_notebook"], 2)},
                   "per_group": {g: {"model": round(a["per_group"][g], 2), "copy": round(b["per_group"][g], 2),
                                     "delta": round(a["per_group"][g] - b["per_group"][g], 2)} for g in a["per_group"]},
                   "per_C_macro": {C: {"model": round(a["per_C_macro"][C], 2), "copy": round(b["per_C_macro"][C], 2),
                                       "delta": round(a["per_C_macro"][C] - b["per_C_macro"][C], 2)} for C in a["per_C_macro"]}}
            if a.get("mv") and b.get("mv"):
                row["mv"] = {g: {"n": a["mv"][g]["n"], "model": round(a["mv"][g]["acc"], 2), "copy": round(b["mv"][g]["acc"], 2),
                                 "delta": round(a["mv"][g]["acc"] - b["mv"][g]["acc"], 2)} for g in a["mv"]}
            res["cells"][tgt][combo] = row
    with open(out / "copy_summary.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"\n[copy] {dataset} / {model}  videos {res['n_videos']}  (칸 = Filtered property macro %, Δ = {PRED_TAG} − copy)")
    for tgt, combos in res["cells"].items():
        for combo, r in combos.items():
            print(f"  {tgt:6s} {combo:10s} n_pair {r['n_pair']:4d}  model {r['model']:6.2f}  copy {r['copy']:6.2f}  Δ {r['delta']:+6.2f}"
                  + (f"   mv Δ " + " ".join(f"{g}={v['delta']:+.1f}" for g, v in r["mv"].items()) if "mv" in r else ""))
    print(f"copy_summary -> {out / 'copy_summary.json'}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="intphys1_dev")
    ap.add_argument("--model", default="vith")
    ap.add_argument("--gpus", type=int, default=1, help="mp.spawn rank 수 (보이는 GPU 앞 N 장)")
    ap.add_argument("--limit", type=int, default=0, help="영상 앞 N 개 (te --limit; 쌍이 온전하지 않을 수 있다) + 폴더에 _smokeN")
    ap.add_argument("--subset", default=None, help="valid6 = O1/O2/O3 앞 2 block (24 영상 · 12 쌍 · 운동×가림 4 칸) + 폴더에 _valid6")
    ap.add_argument("--threads", type=int, default=6, help="rank 당 torch CPU 스레드 (te 기본 6)")
    ap.add_argument("--max-starts", type=int, default=0, help="encoder 배치의 시작점 상한 (0 = 영상 하나의 시작점 전부, te 기본)")
    ap.add_argument("--out", default=os.environ.get("OUTDIR") or None, help="결과 폴더 (기본 $CVPR_RESULTS/eval/intphys1/copy/<ds>_<model>)")
    ap.add_argument("--dryrun", action="store_true", help="두 resolver 대조 · 창 격자 · 이름 · 명령만 (GPU · 모델 없음)")
    ap.add_argument("--validate", action="store_true", help="채점 때 te --validate 도 (CPU; 공식 per_window.json · H6e 대조)")
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args()
    load_env()
    for k in ("TAG", "OUTDIR"):                 # 옛 resolver 가 읽는 env (te_intphys1_score.resolve_cfg 는 env 를 그대로 넘긴다)
        os.environ.pop(k, None)
    if (a.dataset, a.model) != SUPPORTED:
        sys.exit(f"ERROR: IntPhys 1 복사 기준선은 지금 {SUPPORTED[0]} × {SUPPORTED[1]} 만 된다 (받은 값 {a.dataset} × {a.model}).\n"
                 "  te_intphys1_score.py:169 (D=1280 리터럴) · :248-249 (프로토콜 · 데이터셋 · 모델 리터럴) — README '엔진 수정 제안'")
    if a.limit and a.subset:
        sys.exit("ERROR: --limit 과 --subset 은 같이 못 쓴다")

    cfg_new, meta = cvpr_resolve(a.dataset, a.model, a.limit)
    base = os.path.basename(meta["output_dir"]) + (f"_{a.subset}" if a.subset else "")      # <ds>_<model>[_smokeN][_valid6]
    out = Path(a.out) if a.out else Path(os.environ["CVPR_RESULTS"]) / "eval/intphys1/copy" / base
    te.CACHE_ROOT = out.parent                  # te 의 out_dir_of(a) = CACHE_ROOT / out_tag  → 결과 폴더
    te.DOC_DIR = out / "doc"                    # *_scores.json · *_validation.json 이 z_research/TrainingEffects 로 가지 않게

    cfg_old = te.resolve_cfg()                  # 옛 resolver (intphys1_sliding, intphys1_dev, vith) + window_size 32
    diff = [] if a.no_check else cfg_diff(cfg_new, cfg_old)
    pred = te.pred_path(PRED_TAG)
    try:
        lay = te.Layout(cfg_old)                # 창 격자 (evals.world_model_analysis.eval._intphys1_windows 그대로; 모델 없음)
        grid = f"R={lay.R} 행, combo={lay.names}, 영상당 창 {sum(len(cb['starts']) * len(cb['Cs']) for cb in lay.combos)}"
    except Exception as e:                      # noqa: BLE001
        grid = f"(Layout 실패: {e})"
    ip = cfg_old["surprise"]["intphys1"]
    print("==================================================")
    print(f"  copy     : eval/intphys1  dataset={a.dataset}  model={a.model}  (te_intphys1_score.py --preds {PRED_TAG})")
    print(f"  grid     : skips={ip['frame_skips']} windows={ip['window_sizes']} C_mult={ip['context_mult']} stride={ip['stride']}  → {grid}")
    print(f"  model    : {cfg_old['model'].get('arch_name')} dtype={cfg_old['model'].get('dtype')} autocast={cfg_old['model'].get('autocast')} "
          f"window_size={cfg_old['model'].get('window_size')}  predictor={pred.relative_to(REPO)} {'있음' if pred.exists() else '없음!'}")
    print(f"  resolver : cvpr vs legacy {'일치' if not diff else '불일치'} ({', '.join(CMP_KEYS)})" + ("  [--no-check]" if a.no_check else ""))
    print(f"  output   : {out}")
    print("==================================================")
    if diff:
        print("\n".join("    " + x for x in diff))
        sys.exit("ERROR: 두 resolver 의 config 가 다르다 — cvpr/harness/check_equivalence.py 를 볼 것")
    if not pred.exists():
        sys.exit(f"ERROR: predictor 없음 {pred}")

    common = ["--preds", PRED_TAG, "--out_tag", out.name, "--threads", str(a.threads)]
    if a.limit:
        common += ["--limit", str(a.limit)]
    if a.subset:
        common += ["--subset", a.subset]
    argv_extract = ["te_intphys1_score.py", *common, "--gpus", str(a.gpus), "--max_starts", str(a.max_starts)]
    argv_score = ["te_intphys1_score.py", *common, "--score"] + (["--validate"] if a.validate else [])
    if a.dryrun:
        print("--- DRYRUN: 실행하지 않는다. 실행할 명령 (같은 프로세스에서 te_intphys1_score.main() 두 번) ---")
        print("  " + " ".join(argv_extract) + "        # 추출 (GPU)")
        print("  " + " ".join(argv_score) + "        # 채점 (CPU)")
        print(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}   CACHE_ROOT→{te.CACHE_ROOT}   DOC_DIR→{te.DOC_DIR}")
        return

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "_meta.json", "w", encoding="utf-8") as f:
        json.dump({**meta, "copy": True, "output_dir": str(out), "driver": str(Path(__file__).relative_to(REPO)),
                   "engine_script": "z_research/scripts/analysis/te_intphys1_score.py", "predictor": PRED_TAG,
                   "argv": [argv_extract, argv_score], "subset": a.subset}, f, indent=1, ensure_ascii=False)
    sys.argv = argv_extract
    te.main()                                   # 재개: shards/ 에 있는 영상은 건너뛴다 (manifest 가 다르면 죽는다)
    # 추출이 다 됐는지 (te 는 남은 영상 수를 찍기만 하고 0 이 아니어도 exit 0 이다)
    from evals.world_model_analysis.data import WMADataset  # noqa: E402  (index 만 읽는다)
    n_expect = len(te.select_videos(WMADataset(cfg_old).records, a.subset, a.limit))
    n_have = len(glob.glob(str(out / "shards" / "v*.npz")))
    if n_have < n_expect:
        sys.exit(f"ERROR: shard {n_have}/{n_expect} — 같은 명령으로 재개할 것")
    sys.argv = argv_score
    te.main()
    summarize(out, a.dataset, a.model)


if __name__ == "__main__":
    main()
