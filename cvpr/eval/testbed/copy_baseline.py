#!/usr/bin/env python3
"""eval/testbed/copy_baseline.py — 복사 기준선 (fixed c16t32) 을 **기존 스크립트** z_research/scripts/analysis/te_v11_score.py 로 돌린다.

    python cvpr/eval/testbed/copy_baseline.py --dataset v11_split_test --model vith --gpus 8 [--limit N] [--dryrun] [--validate]
    (보통은 copy_baseline.sh 가 부른다. 정의 · 노브 · 결과 위치는 README.md)

하는 일
  1. cvpr/harness/resolve.py (새 resolver) 로 (데이터셋, 모델) 을 검사하고 이름을 받는다
       → 결과 폴더 = $CVPR_RESULTS/eval/testbed/copy/<데이터셋>_<모델>[_smokeN]      (--out / OUTDIR 로 덮는다)
  2. te_v11_score.resolve_cfg() (옛 resolver — 스크립트 내부 규칙 그대로) 가 낸 config 와 1 의 config 의 data · model · surprise ·
     scoring 블록이 같은지 대조한다. 다르면 죽는다 (두 resolver 가 갈린 것 — cvpr/harness/check_equivalence.py 로 확인할 것).
  3. te_v11_score.main() 을 같은 프로세스에서 부른다. predictor 는 `own` 하나 (= 병합 config 대로 지은 predictor; vith 면 model.pth 의
     릴리즈 predictor, vith_pft_* 면 그 post-FT predictor) + 복사 기준선. 산출 = summary.json (own · copy 의 matched-pair 정확도),
     l1.npy · copy.npy · hcopy.npy · meta.json · _resolved.yaml (전부 te 가 쓴다).
  4. 배열에서 copy_summary.json 을 만든다: model · copy · copy_target 의 정확도와 Δ = model − copy 를
     overall · condition · violation_type · condition|violation_type · condition|violation_type|pair_id (A/B 방향) 로.

⚠️ te_v11_score.py 는 데이터셋이 모듈 상수다 (te_v11_score.py:156 `PROTOCOL, DATASET = "surprise_c16t32", "v11_split_test"`).
   v11_split_test 이외를 주면 이 래퍼가 **부모 프로세스의 모듈 상수만** 덮어쓴다 (파일 수정 없음). mp.spawn 된 rank (`_worker`, :477-483)
   는 병합 config 를 인자로 받고 그 상수를 읽지 않는다. 다만 v11_split_test 이외의 데이터셋은 실제로 돌려 보지 않았다 (미검증).
   깔끔한 해법 (`--dataset` 인자) 은 README 의 "엔진 수정 제안".
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent            # cvpr/eval/testbed
CVPR = HERE.parents[1]
REPO = CVPR.parent
TE_DIR = REPO / "z_research/scripts/analysis"
sys.path.insert(0, str(CVPR / "harness"))
sys.path.insert(0, str(TE_DIR))
from resolve import load_env  # noqa: E402  (cvpr/harness/resolve.py 의 env.sh 로더 — 복사하지 않는다)
import te_v11_score as te  # noqa: E402  (torch 를 import 하지만 모델은 짓지 않는다. spawn 된 rank 도 이 줄을 다시 탄다)

CMP_KEYS = ("data", "model", "surprise", "scoring")
TE_DEFAULT_DATASET = te.DATASET                     # "v11_split_test"
GROUPS = (("condition",), ("violation_type",), ("condition", "violation_type"), ("condition", "violation_type", "pair_id"))


def cvpr_resolve(dataset: str, model: str, limit: int) -> tuple[dict, dict]:
    """새 resolver 를 subprocess 로 불러 (병합 config, meta) 를 받는다. 레지스트리 · available · 이름 규칙은 전부 거기서 (복사 없음)."""
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
    """CMP_KEYS 블록에서 다른 잎 경로 (cvpr vs legacy)."""
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


def acc_row(meta: dict, idx: np.ndarray, scores: dict) -> dict:
    """부분집합 idx 의 matched-pair 정확도 (te.matched_acc — eval.py::score_blocks 와 같은 규칙, 동점 0.5) 를 방법마다."""
    sub = {c: [meta[c][i] for i in idx] for c in ("block_id", "pair_id", "plausible")}
    row = {"n_pair": None, "n_ties": {}}
    for name, s in scores.items():
        acc, n, ties = te.matched_acc(sub, s[idx])
        row[name] = round(100 * acc, 2) if n else None
        row["n_pair"] = n
        row["n_ties"][name] = ties
    row["delta"] = round(row["model"] - row["copy"], 2) if row["n_pair"] else None
    return row


def summarize(out: Path, dataset: str, model: str) -> dict:
    """te 산출 배열 → copy_summary.json (model · copy · copy_target, Δ = model − copy; 전체 + 조건별)."""
    with open(out / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    l1, cp, hc = (np.load(out / f"{k}.npy") for k in ("l1", "copy", "hcopy"))
    k = meta["pred_tags"].index("own")
    scores = {"model": l1[:, k].mean(1), "copy": cp.mean(1), "copy_target": hc.mean(1)}
    n = len(meta["video_id"])
    res = {"dataset": dataset, "model": model, "n_clip": n, "chance": 50.0,
           "definition": {
               "model": "S = mean_j mean|p[j] − h[8+j]| (own predictor = 병합 config 대로 지은 predictor; 표준 surprise)",
               "copy": "S = mean_j mean|LN(z)[7] − h[8+j]| — 예측 없음: 문맥 encoder 마지막 튜블릿을 미래 8 슬롯 전부에 복사",
               "copy_target": "S = mean_j mean|h[7] − h[8+j]| — target encoder 쪽 마지막 문맥 튜블릿 복사 (h[7] 은 32 장 전부를 본 양방향 표현 — 참고용)",
               "acc": "matched pair (block_id, pair_id): hit = 1[S(pos) < S(imp)] + 0.5 동점 (te_v11_score.matched_acc = eval.py::score_blocks)",
               "delta": "model − copy (pt, 같은 쌍 위의 차). CI 는 여기서 내지 않는다 (block bootstrap 은 te_analyze_scores.py)"},
           "mean_S": {k_: float(v.mean()) for k_, v in scores.items()},
           "overall": acc_row(meta, np.arange(n), scores), "by": {},
           "source": {"script": "z_research/scripts/analysis/te_v11_score.py", "arrays": ["l1.npy", "copy.npy", "hcopy.npy"],
                      "summary": "summary.json", "pred_tag": "own"}}
    for cols in GROUPS:
        if any(c not in meta or all(v is None for v in meta[c]) for c in cols):
            continue                                                   # index 에 그 열이 없는 데이터셋
        groups = defaultdict(list)
        for i in range(n):
            groups["|".join(str(meta[c][i]) for c in cols)].append(i)
        res["by"]["|".join(cols)] = {g: acc_row(meta, np.array(ix), scores) for g, ix in sorted(groups.items())}
    with open(out / "copy_summary.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    o = res["overall"]
    print(f"\n[copy] {dataset} / {model}  n_pair {o['n_pair']}  model {o['model']}  copy {o['copy']}  copy_target {o['copy_target']}  "
          f"Δ(model−copy) {o['delta']:+.2f}")
    for key in ("condition", "violation_type", "condition|violation_type"):
        if key in res["by"]:
            print(f"  by {key}:")
            for g, r in res["by"][key].items():
                print(f"    {g:<40s} n {r['n_pair']:>6d}  model {r['model']:6.2f}  copy {r['copy']:6.2f}  Δ {r['delta']:+6.2f}")
    print(f"copy_summary -> {out / 'copy_summary.json'}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=TE_DEFAULT_DATASET, help="cvpr/registry/datasets.yaml 과 configs/protocols/datasets.md 양쪽에 있는 이름")
    ap.add_argument("--model", default="vith", help="양쪽 레지스트리에 있는 V-JEPA 계열 이름 (dual_encoder · LN target 규격)")
    ap.add_argument("--gpus", type=int, default=1, help="mp.spawn rank 수 (보이는 GPU 앞 N 장)")
    ap.add_argument("--limit", type=int, default=0, help="block 단위로 약 N clip (te --limit) + 폴더에 _smokeN")
    ap.add_argument("--batch", type=int, default=8, help="rank 당 clip (te 기본 8)")
    ap.add_argument("--workers", type=int, default=6, help="rank 당 PNG 디코드 프로세스 (te 기본 6)")
    ap.add_argument("--threads", type=int, default=4, help="rank 당 torch CPU 스레드 (te 기본 4)")
    ap.add_argument("--out", default=os.environ.get("OUTDIR") or None, help="결과 폴더 (기본 $CVPR_RESULTS/eval/testbed/copy/<ds>_<model>)")
    ap.add_argument("--dryrun", action="store_true", help="두 resolver 대조 · 이름 · 명령만 (GPU · 모델 없음)")
    ap.add_argument("--validate", action="store_true", help="끝난 뒤 te --validate (GPU 1 장, 48 clip; 참조 per_block 이 v11 것이라 v11_split_test 만)")
    ap.add_argument("--no-check", action="store_true", help="두 resolver 대조를 건너뛴다")
    a = ap.parse_args()
    load_env()
    for k in ("TAG", "OUTDIR"):                 # 옛 resolver 가 읽는 env — te 도 지우지만 (te_v11_score.py:216) 여기서도 지운다
        os.environ.pop(k, None)

    cfg_new, meta = cvpr_resolve(a.dataset, a.model, a.limit)
    base = os.path.basename(meta["output_dir"])                          # <ds>_<model>[_smokeN]  (새 resolver 의 이름 규칙)
    out = Path(a.out) if a.out else Path(os.environ["CVPR_RESULTS"]) / "eval/testbed/copy" / base

    if a.dataset != TE_DEFAULT_DATASET:
        print(f"[copy] ⚠️ te_v11_score.DATASET {TE_DEFAULT_DATASET!r} → {a.dataset!r} 로 덮는다 (부모 프로세스만, 파일 수정 없음; 미검증 경로)")
        te.DATASET = a.dataset
    te.DOC_DIR = out / "doc"                                              # --validate 산출물이 z_research/TrainingEffects 로 가지 않게

    cfg_old, _ = te.resolve_cfg(a.model)                                  # 옛 resolver + surprise_c16t32 규격 검사 (te 자신의 규칙)
    diff = [] if a.no_check else cfg_diff(cfg_new, cfg_old)
    d, m, s = cfg_old["data"], cfg_old["model"], cfg_old["surprise"]
    print("==================================================")
    print(f"  copy     : eval/testbed  dataset={a.dataset}  model={a.model}  (te_v11_score.py --preset own)")
    print(f"  frames   : n={d['n_frames']} start={d.get('frames_start', 1)} stride={d.get('frames_stride', 1)}  ctx={s['context_length']}  "
          f"res={d.get('resolution')}  pairing={cfg_old.get('scoring', {}).get('pairing')}")
    print(f"  model    : {m.get('arch_name')} dtype={m.get('dtype')} autocast={m.get('autocast')} window_size={m.get('window_size')}"
          + (f" predictor_ckpt={os.path.basename(str(m['predictor_checkpoint']))}" if m.get("predictor_checkpoint") else ""))
    print(f"  resolver : cvpr vs legacy {'일치' if not diff else '불일치'} ({', '.join(CMP_KEYS)})" + ("  [--no-check]" if a.no_check else ""))
    print(f"  output   : {out}")
    print("==================================================")
    if diff:
        print("\n".join("    " + x for x in diff))
        sys.exit("ERROR: 두 resolver 의 config 가 다르다 — cvpr/harness/check_equivalence.py 를 볼 것")

    argv = ["te_v11_score.py", "--model", a.model, "--preset", "own", "--out", str(out), "--gpus", str(a.gpus),
            "--batch", str(a.batch), "--workers", str(a.workers), "--threads", str(a.threads)]
    if a.limit:
        argv += ["--limit", str(a.limit)]
    argv_val = ["te_v11_score.py", "--model", a.model, "--preds", "own", "--validate",
                "--batch", str(a.batch), "--workers", str(a.workers), "--threads", str(a.threads)]
    if a.dryrun:
        print("--- DRYRUN: 실행하지 않는다. 실행할 명령 (같은 프로세스에서 te_v11_score.main()) ---")
        print("  " + " ".join(argv))
        if a.validate:
            print("  " + " ".join(argv_val))
        print(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}")
        return

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "_meta.json", "w", encoding="utf-8") as f:
        json.dump({**meta, "copy": True, "output_dir": str(out), "driver": str(Path(__file__).relative_to(REPO)),
                   "engine_script": "z_research/scripts/analysis/te_v11_score.py", "argv": argv,
                   "dataset_override": a.dataset != TE_DEFAULT_DATASET}, f, indent=1, ensure_ascii=False)
    sys.argv = argv
    te.main()                                                             # 재개: 같은 폴더에 같은 meta 면 done==0 행만 돈다
    if a.validate:
        sys.argv = argv_val
        te.main()
    summarize(out, a.dataset, a.model)


if __name__ == "__main__":
    main()
