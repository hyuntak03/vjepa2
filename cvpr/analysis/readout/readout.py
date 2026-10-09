#!/usr/bin/env python3
"""cvpr/analysis/readout/readout.py — 학습된 자 (readout) 분석의 config 기반 단일 진입점 (run.sh 가 부른다).

    python cvpr/analysis/readout/readout.py --list
    python cvpr/analysis/readout/readout.py <step> [--dataset D] [--model M] [--dry] [--help-script]

하는 일: config.yaml 의 스텝 하나를 읽어 (스크립트, 인자, 환경변수) 를 **정확한 명령 하나**로 만든다.
  --dry          명령 · 환경변수 · 입력 실물 (OK/MISSING) 만 보이고 끝 (GPU 0 장, 아무것도 쓰지 않는다)
  --help-script  그 스크립트의 --help (argparse 가 있는 스텝만; 없으면 docstring 을 보인다)
  (기본)         실행. stdout 은 화면과 $CVPR_RESULTS/analysis/readout/<step>__<dataset>_<model>/stdout.log 에, 명령은 _cmd.json 에
kind: launch 스텝은 cvpr/harness/launch.sh 로 넘긴다 (--dry 는 DRYRUN=1 로 — 병합 config 를 launch.sh 가 보인다).

과학은 없다 — 치환과 실물 검사뿐. 치환 규칙은 config.yaml 머리말. 레지스트리 · ${CVPR_*} 치환은 cvpr/harness/resolve.py 의
함수 (load_env · expand · load_registry · resolve_entry) 를 그대로 import 한다 (복제하지 않는다).

환경변수 (run.sh 가 넘긴다): GPUS GPU_IDS DRYRUN HELP FORCE OUTDIR + config.yaml knobs: 의 이름들 (TRAIN · DECODER · NAME …)
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CVPR = HERE.parents[1]
sys.path.insert(0, str(CVPR / "harness"))
import resolve as R  # noqa: E402  (단일 resolver 의 레지스트리 · 치환 함수)

CFG_PATH = HERE / "config.yaml"
TOK = re.compile(r"(?<!\$)\{([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)(\?)?\}")   # `${CVPR_X}` 는 resolve.expand 몫
MODEL_RULES = ("vith", "vith_predictor", "any")


def die(msg: str):
    print(f"\nERROR: {msg}\n", file=sys.stderr)
    sys.exit(1)


def load_cfg() -> dict:
    with open(CFG_PATH, encoding="utf-8") as f:
        c = yaml.safe_load(f) or {}
    for k in ("knobs", "steps"):
        if k not in c:
            die(f"{CFG_PATH} 에 {k}: 가 없다")
    c.setdefault("datasets_inline", {})
    return c


# ── 치환 ─────────────────────────────────────────────────────────────────────
class Ctx:
    def __init__(self, cfg, step_name, step, dataset, model, gpu_ids):
        self.cfg, self.step, self.knobs = cfg, step, cfg["knobs"]
        self.simple = dict(step=step_name, dataset=dataset, model=model,
                           gpus=str(len(gpu_ids)), gpu_ids=" ".join(gpu_ids), cuda=",".join(gpu_ids))
        self.ds: dict = {}
        self.md: dict = {}

    def knob(self, name: str) -> str:
        return os.environ.get(name, str(self.knobs[name].get("default", "")))

    def sub(self, s, depth: int = 0) -> str:
        """`{...}` 자리표시자 → 값, 그 다음 `${CVPR_*}` (resolve.expand)."""
        if s is None:
            return ""
        s = str(s)
        if depth > 12:
            die(f"치환 순환: {s}")

        def rep(m):
            key, opt = m.group(1), m.group(2)
            if "." in key:
                head, sub = key.split(".", 1)
                tbl = {"ds": self.ds, "model": self.md}.get(head)
                if tbl is None:
                    die(f"모르는 자리표시자 {{{key}}} (ds.* · model.* 만)")
                if sub not in tbl:
                    if opt:
                        return ""
                    die(f"{{{key}}} — '{head}' 항목에 '{sub}' 가 없다 (있는 키: {', '.join(sorted(tbl))})")
                return str(tbl[sub])
            if key in self.simple:
                return self.simple[key]
            if key in self.knobs:
                return self.sub(self.knob(key), depth + 1)
            if opt:
                return ""
            die(f"모르는 자리표시자 {{{key}}} (knobs: {', '.join(self.knobs)})")

        return R.expand(TOK.sub(rep, s))


# ── 레지스트리 ───────────────────────────────────────────────────────────────
def load_dataset(cfg, name):
    DS = R.load_registry("datasets")
    if name in DS:
        e, src = R.expand(R.resolve_entry(DS, name, "데이터셋")), "cvpr/registry/datasets.yaml"
    elif name in cfg["datasets_inline"]:
        e, src = R.expand(dict(cfg["datasets_inline"][name])), "config.yaml datasets_inline"
    else:
        die(f"데이터셋 '{name}' 이 레지스트리에도 config.yaml datasets_inline 에도 없다\n"
            f"  레지스트리: {', '.join(sorted(k for k in DS if not k.startswith('_')))}\n"
            f"  inline    : {', '.join(sorted(cfg['datasets_inline']))}")
    if e.get("available") is False:
        die(f"데이터셋 '{name}' 은 available: false 다. {e.get('note', '')}")
    return e, src


def load_model(name):
    MD = R.load_registry("models")
    e = R.expand(R.resolve_entry(MD, name, "모델"))
    if e.get("available") is False:
        die(f"모델 '{name}' 은 available: false 다. {e.get('note', '')}")
    return e, MD[name].get("base")


def check_model_rule(rule, name, base):
    if rule not in MODEL_RULES:
        die(f"model_rule 은 {MODEL_RULES} 중 하나 -> {rule}")
    if rule == "vith" and name != "vith":
        die(f"이 스텝은 MODEL=vith 만 된다 (스크립트가 ViT-H 토큰 캐시 · 1280 차원을 전제) -> {name}")
    if rule == "vith_predictor" and not (name == "vith" or base == "vith"):
        die(f"이 스텝은 vith 또는 base: vith 인 모델만 된다 (encoder 는 릴리즈 고정, predictor_checkpoint 만 PRED_CKPT 로) -> {name}")


# ── GPU ──────────────────────────────────────────────────────────────────────
def gpu_ids(step) -> list[str]:
    want = int(step.get("gpu", 0))
    if os.environ.get("GPU_IDS"):
        return os.environ["GPU_IDS"].split()
    if os.environ.get("GPUS"):
        return [str(i) for i in range(int(os.environ["GPUS"]))]
    if want > 0 and os.environ.get("CUDA_VISIBLE_DEVICES"):
        return [x for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x]
    return [str(i) for i in range(want)]


# ── 명령 만들기 ──────────────────────────────────────────────────────────────
def build(cfg, name, dataset_arg, model_arg):
    steps = cfg["steps"]
    if name not in steps:
        die(f"스텝 '{name}' 이 없다. 목록: {', '.join(steps)}")
    step = dict(steps[name])
    kind = step.get("kind", "script")
    if kind not in ("script", "bash", "launch"):
        die(f"[{name}] kind 는 script | bash | launch -> {kind}")
    ids = gpu_ids(step)
    ctx = Ctx(cfg, name, step, "", "", ids)

    # 데이터셋 · 모델 (이름 자체에 노브가 들어갈 수 있다: rollout_v2_{PSET})
    ds_default = ctx.sub(step.get("dataset", ""))
    dataset = dataset_arg or ds_default
    if not dataset:
        die(f"[{name}] dataset 이 없다 (config 의 dataset: 또는 run.sh 인자)")
    if dataset_arg and step.get("dataset_lock") and dataset_arg != ds_default:
        die(f"[{name}] 은 데이터셋 경로를 스크립트가 하드코딩한다 (dataset_lock) — DATASET={dataset_arg} 를 받을 수 없다. "
            f"기본 {ds_default}. 바꾸려면 README '엔진 수정 제안' 을 볼 것")
    allowed = step.get("datasets")
    if allowed and dataset not in allowed:
        die(f"[{name}] 데이터셋 {dataset} 은 이 스텝이 검증한 목록 밖이다: {allowed}")
    model = model_arg or step.get("model", "vith")
    ctx.simple.update(dataset=dataset, model=model)
    ctx.ds, ds_src = load_dataset(cfg, dataset)
    ctx.md, base = load_model(model)
    check_model_rule(step.get("model_rule", "vith"), model, base)

    repo = os.environ["CVPR_REPO"]
    env = {}
    for k, v in (step.get("env") or {}).items():
        val = ctx.sub(v)
        if val != "":
            env[k] = val
    want_gpu = int(step.get("gpu", 0))
    if kind == "launch":
        env.update(GPUS=str(len(ids)), GPU_IDS=" ".join(ids))
        if step.get("tag"):
            env["TAG"] = ctx.sub(step["tag"])
        if step.get("set_file"):
            env["SET_FILE"] = ctx.sub(step["set_file"])
        if step.get("suffix"):                          # 결과 폴더만 가른다 — 캐시 이름은 TAG 가 이긴다 (resolve.py 이름 규칙)
            env["SUFFIX"] = ctx.sub(step["suffix"])
        proto = ctx.sub(step["protocol"])
        cmd = ["bash", f"{os.environ['CVPR_ROOT']}/harness/launch.sh", proto, dataset, model]
        script = proto
    else:
        if want_gpu > 0 and step.get("cuda", True) and ids:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(ids)
        script = f"{repo}/{ctx.sub(step['script'])}"
        args = [a for a in (ctx.sub(x) for x in (step.get("args") or [])) if a != ""]
        args += shlex.split(ctx.sub("{ARGS}"))
        interp = [os.environ["CVPR_PY"], "-u"] if kind == "script" else ["bash"]
        cmd = interp + [script] + args

    inputs = [(p, Path(p).exists()) for p in (ctx.sub(x) for x in (step.get("inputs") or []))]
    outputs = [ctx.sub(x) for x in (step.get("outputs") or [])]
    return dict(name=name, step=step, kind=kind, dataset=dataset, ds_src=ds_src, model=model, gpu_ids=ids,
                env=env, cmd=cmd, script=script, inputs=inputs, outputs=outputs, cwd=repo, ctx=ctx)


def shell_line(b) -> str:
    envs = " ".join(f"{k}={shlex.quote(v)}" for k, v in b["env"].items())
    return f"cd {shlex.quote(b['cwd'])} && {envs}{' ' if envs else ''}{' '.join(shlex.quote(c) for c in b['cmd'])}"


def show(b, dry: bool):
    s = b["step"]
    print("==================================================")
    print(f"### readout/{b['name']}   dataset={b['dataset']} ({b['ds_src']})   model={b['model']}   "
          f"gpu={len(b['gpu_ids'])}{' [' + ','.join(b['gpu_ids']) + ']' if b['gpu_ids'] else ' (CPU)'}   kind={b['kind']}")
    print(f"  doc     : {s.get('doc', '')}")
    if s.get("note"):
        print(f"  note    : {s['note']}")
    print(f"  script  : {b['script']}")
    for k, v in b["env"].items():
        print(f"  env     : {k}={v}")
    for p, ok in b["inputs"]:
        print(f"  input   : [{'OK     ' if ok else 'MISSING'}] {p}")
    for p in b["outputs"]:
        print(f"  output  : {p}")
    print(f"  cmd     : {shell_line(b)}")
    if dry and b["kind"] == "launch":
        print("  (launch.sh 의 DRYRUN — 아래는 병합된 config)")
    print("==================================================")


def log_dir(b) -> Path:
    if os.environ.get("OUTDIR"):
        return Path(os.environ["OUTDIR"])
    return Path(os.environ["CVPR_RESULTS"]) / "analysis" / "readout" / f"{b['name']}__{b['dataset']}_{b['model']}"


def run(b) -> int:
    env = {**os.environ, **b["env"]}
    if b["kind"] == "launch":                       # launch.sh 가 자기 결과 폴더 · 로그를 만든다
        return subprocess.call(b["cmd"], env=env, cwd=b["cwd"])
    d = log_dir(b)
    d.mkdir(parents=True, exist_ok=True)
    (d / "_cmd.json").write_text(json.dumps(dict(
        step=b["name"], dataset=b["dataset"], model=b["model"], cmd=b["cmd"], env=b["env"], cwd=b["cwd"],
        shell=shell_line(b), inputs=[dict(path=p, exists=ok) for p, ok in b["inputs"]], outputs=b["outputs"],
        host=socket.gethostname(), started=time.strftime("%F %T")), indent=1, ensure_ascii=False))
    log = d / "stdout.log"
    print(f"  log     : {log}")
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n===== {time.strftime('%F %T')} {socket.gethostname()}  {shell_line(b)}\n")
        p = subprocess.Popen(b["cmd"], env=env, cwd=b["cwd"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            sys.stdout.write(line); sys.stdout.flush(); f.write(line)
        rc = p.wait()
        f.write(f"===== exit {rc}  {time.strftime('%F %T')}\n")
    print(f"  exit    : {rc}")
    return rc


def help_script(b) -> int:
    """argparse 가 있으면 --help (GPU 0 장), 없으면 docstring 머리말 — 인자 없는 스크립트는 --help 로 본체가 돈다."""
    if b["kind"] == "launch":
        print("  (launch 스텝 — --help 대신 DRYRUN=1 로 병합 config 를 본다)"); return 0
    if b["step"].get("help"):
        env = {**os.environ, **{k: v for k, v in b["env"].items() if k != "CUDA_VISIBLE_DEVICES"}}
        return subprocess.call(b["cmd"][:-len(b["cmd"]) + (2 if b["kind"] == "script" else 1)] + [b["script"], "--help"], env=env, cwd=b["cwd"])
    src = Path(b["script"]).read_text(encoding="utf-8")
    doc = ast.get_docstring(ast.parse(src)) if b["kind"] == "script" else "\n".join(l for l in src.splitlines()[:40] if l.startswith("#"))
    print(f"  (argparse 없음 → --help 가 본체를 돌린다. docstring 만 보인다)\n{doc}")
    return 0


def list_steps(cfg):
    print("knobs (환경변수 = 노브 이름):")
    for k, v in cfg["knobs"].items():
        cur = os.environ.get(k)
        print(f"  {k:<14} {str(v.get('default', '')):<40} {('← env ' + cur) if cur is not None else ''}  {v.get('doc', '')}")
    print("\nsteps:")
    print(f"  {'step':<22} {'kind':<7} {'gpu':>3}  {'dataset':<24} {'model_rule':<14} script")
    for n, s in cfg["steps"].items():
        print(f"  {n:<22} {s.get('kind', 'script'):<7} {int(s.get('gpu', 0)):>3}  {str(s.get('dataset', '')):<24} "
              f"{s.get('model_rule', 'vith'):<14} {s.get('script', s.get('protocol', ''))}")
        print(f"  {'':<22} {s.get('doc', '')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", nargs="?")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--help-script", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    R.load_env()
    cfg = load_cfg()
    if a.list or not a.step:
        return list_steps(cfg)
    b = build(cfg, a.step, a.dataset, a.model)
    show(b, a.dry)
    if a.help_script:
        sys.exit(help_script(b))
    if a.dry:
        if b["kind"] == "launch":
            sys.exit(subprocess.call(b["cmd"], env={**os.environ, **b["env"], "DRYRUN": "1"}, cwd=b["cwd"]))
        return
    missing = [p for p, ok in b["inputs"] if not ok]
    if missing and not os.environ.get("FORCE"):
        die(f"[{b['name']}] 입력 실물이 없다 (FORCE=1 로 넘긴다):\n  " + "\n  ".join(missing))
    sys.exit(run(b))


if __name__ == "__main__":
    main()
