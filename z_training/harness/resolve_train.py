#!/usr/bin/env python3
"""학습 config 병합기: configs/training/<이름>.yaml (+extends) + 데이터/모델 레지스트리 -> 실행 config 하나.

  python z_training/harness/resolve_train.py <이름|경로> -o <run>/config.yaml --folder <run> [--set a.b=1 ...]

하는 일 (configs/protocols 의 resolve.py 와 같은 철학 — 실험마다 yaml 을 새로 뜨지 않는다)
  1. `extends:` 를 따라가 부모 위에 자식을 깊은 병합 (자식이 이김, dict 재귀, list 교체, null 은 키 삭제)
  2. `data.datasets` 의 이름을 configs/training/datasets.md 섹션으로 푼다
     (문자열 "intphys1_train" 또는 {name: intphys1_train, limit: 16, weight: 1.0} — dict 의 키가 레지스트리 값을 덮는다)
  3. `model.base` 를 configs/protocols/models.md 로 풀어 checkpoint / arch_name 을 채운다
  4. `val.dataset` 을 configs/protocols/datasets.md 로 풀어 val.data 를 채운다
  5. `--set a.b.c=값` 점 경로 덮어쓰기 (값은 YAML, null = 삭제). 리스트는 정수 인덱스로 (data.datasets.0.limit=8)
  6. 모델을 로드하기 전에 실물을 검사한다: 체크포인트, index, 프레임 파일 실재(첫 행·모든 start),
     resolution == img_size, context_frames tubelet 정렬, n_frames 정렬
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import yaml

ROOT = "/data/hyuntak/project/2026/2027_cvpr/vjepa2"
TRAIN_CFG = f"{ROOT}/configs/training"
sys.path.insert(0, f"{ROOT}/z_research/scripts/harness")
from resolve import parse_registry  # noqa: E402  (protocols 레지스트리와 같은 파서)

META_KEYS = ("raw_frames", "available", "note")


def die(msg):
    print(f"\nERROR: {msg}\n", file=sys.stderr)
    sys.exit(1)


def merge(dst, src):
    for k, v in src.items():
        if v is None:
            dst.pop(k, None)
        elif isinstance(v, dict) and isinstance(dst.get(k), dict):
            dst[k] = merge(dict(dst[k]), v)
        else:
            dst[k] = v
    return dst


def load_cfg(name_or_path, seen=()):
    path = name_or_path if name_or_path.endswith((".yaml", ".yml")) else f"{TRAIN_CFG}/{name_or_path}.yaml"
    if not os.path.isfile(path):
        avail = sorted(f[:-5] for f in os.listdir(TRAIN_CFG) if f.endswith(".yaml") and not f.startswith("_"))
        die(f"config 없음 -> {path}\n  사용 가능: {', '.join(avail)}")
    c = yaml.safe_load(open(path, encoding="utf-8")) or {}
    base = c.pop("extends", None)
    if not base:
        return c, path
    if base in seen:
        die(f"extends 순환: {' -> '.join(seen + (base,))}")
    parent, _ = load_cfg(base, seen + (base,))
    return merge(parent, c), path


def set_path(cfg, key, val):
    parts = key.split(".")
    cur = cfg
    for k in parts[:-1]:
        if isinstance(cur, list):
            i = int(k)
            if isinstance(cur[i], str):            # 레지스트리 이름 -> {name: ...} 로 열어 준다
                cur[i] = {"name": cur[i]}
            cur = cur[i]
        else:
            if isinstance(cur.get(k), str):
                cur[k] = {"name": cur[k]}
            elif not isinstance(cur.get(k), (dict, list)):
                cur[k] = {}
            cur = cur[k]
    leaf = parts[-1]
    if isinstance(cur, list):
        i = int(leaf)
        if val is None:
            cur.pop(i)
        else:
            cur[i] = val
    elif val is None:
        cur.pop(leaf, None)
    else:
        cur[leaf] = val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--folder", required=True, help="run 폴더 (cfg.folder)")
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--limit", type=int, default=None, help="모든 frames_index 데이터셋에 limit 주입 (스모크)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    cfg, cfg_path = load_cfg(a.config)
    cfg["folder"] = os.path.abspath(a.folder)
    if cfg.get("app") != "vjepa_frozen":
        die(f"app 이 vjepa_frozen 이 아니다: {cfg.get('app')!r}")

    # ── --set (레지스트리 확장 **전**에 적용해 data.datasets=[...] 같은 치환을 허용) ──
    for kv in a.sets:
        if "=" not in kv:
            die(f"--set 은 KEY=VALUE 여야 한다 -> {kv}")
        key, val = kv.split("=", 1)
        try:
            val = yaml.safe_load(val)
        except yaml.YAMLError:
            pass
        set_path(cfg, key, val)

    # ── data.datasets: 이름 -> 레지스트리 ─────────────────────────────────────────
    DS = parse_registry(f"{TRAIN_CFG}/datasets.md", "type")
    out = []
    for item in cfg["data"].get("datasets") or []:
        if isinstance(item, str):
            item = {"name": item}
        if not isinstance(item, dict):
            die(f"data.datasets 항목이 이상하다: {item!r}")
        name = item.pop("name", None)
        if name:
            if name not in DS:
                die(f"학습 데이터셋 '{name}' 이 configs/training/datasets.md 에 없다\n  사용 가능: {', '.join(sorted(DS))}")
            spec = dict(DS[name])
            if spec.get("available") is False:
                die(f"데이터셋 '{name}' 은 available: false 다. {spec.get('note', '')}")
            spec.update(item)                      # 사용자 키가 이긴다
            spec["name"] = name
        else:
            spec = item
        if a.limit and spec.get("type") == "frames_index":
            spec["limit"] = int(a.limit)
        out.append(spec)
    if not out:
        die("data.datasets 가 비어 있다 — 학습 데이터는 아직 정하지 않았다. 돌릴 때 준다:\n"
            "  SET=\"data.datasets=[<이름>]\"  또는 config 의 data.datasets 를 채운다\n"
            f"  후보 (configs/training/datasets.md): {', '.join(sorted(DS))}\n"
            "  ⚠️ 채점 세트(v11 등)와 문맥을 공유하는 데이터로 학습하면 그 세트 점수는 학습셋 점수다")
    cfg["data"]["datasets"] = out

    # ── model.base -> models.md ───────────────────────────────────────────────────
    m = cfg["model"]
    if "checkpoint" not in m:
        MD = parse_registry(f"{ROOT}/configs/protocols/models.md", "checkpoint")
        base = m.get("base", "vith")
        if base not in MD:
            die(f"model.base='{base}' 가 configs/protocols/models.md 에 없다. 사용 가능: {', '.join(sorted(MD))}")
        for k, v in MD[base].items():
            m.setdefault(k, v)                    # checkpoint, arch_name
    if not os.path.isfile(m["checkpoint"]):
        die(f"model.checkpoint 가 없다 -> {m['checkpoint']}")

    # ── val.dataset -> protocols/datasets.md ─────────────────────────────────────
    V = cfg.get("val")
    if V and V.get("dataset") and not V.get("data"):
        PD = parse_registry(f"{ROOT}/configs/protocols/datasets.md", "root")
        if V["dataset"] not in PD:
            die(f"val.dataset='{V['dataset']}' 이 configs/protocols/datasets.md 에 없다")
        p = dict(PD[V["dataset"]])
        if p.get("available") is False:
            die(f"val.dataset='{V['dataset']}' 은 available: false 다")
        keep = ("root", "index_csv", "frames_root", "frames_pattern", "frames_start", "frames_stride",
                "block_column", "pair_column", "plausible_column", "type_column", "variant_column", "raw_frames")
        V["data"] = {k: p[k] for k in keep if k in p}
        if "frames_root" not in V["data"]:
            die(f"val.dataset='{V['dataset']}' 은 frames_root 가 없다 (PNG 직독만 지원)")
    if V is not None and not V.get("data"):
        cfg["val"] = None

    # ── 실물 검사 ──────────────────────────────────────────────────────────────────
    d = cfg["data"]
    n, res = int(d["n_frames"]), int(d["resolution"])
    tub = int(m.get("tubelet_size", 2))
    if res != int(m.get("img_size", 256)):
        die(f"data.resolution({res}) != model.img_size({m.get('img_size')})")
    if n % tub:
        die(f"data.n_frames({n}) 가 tubelet({tub}) 배수가 아니다")
    for mk in cfg.get("mask") or []:
        if mk.get("type", "temporal_prefix") == "temporal_prefix":
            cf = mk.get("context_frames", [n // 2])
            for c in (cf if isinstance(cf, list) else [cf]):
                if int(c) % tub or not (0 < int(c) < n):
                    die(f"mask.context_frames={c}: tubelet 정렬이고 0 < C < {n} 여야 한다")
    if not cfg.get("mask"):
        die("mask 리스트가 비어 있다")

    def check_frames(spec, label):
        for k in ("root", "frames_root"):
            if not os.path.isdir(spec[k]):
                die(f"{label}.{k} 가 없다 -> {spec[k]}")
        ipath = os.path.join(spec["root"], spec.get("index_csv", "index.csv"))
        if not os.path.isfile(ipath):
            die(f"{label}: index 가 없다 -> {ipath}")
        with open(ipath, newline="", encoding="utf-8") as f:
            row = next(csv.DictReader(f), None)
        if row is None:
            die(f"{label}: index 가 비어 있다 -> {ipath}")
        starts = spec.get("frames_start_choices") or [spec.get("frames_start", 0)]
        stride = int(spec.get("frames_stride", 1))
        for st in starts:
            last = int(st) + (n - 1) * stride
            p = os.path.join(spec["frames_root"], spec.get("frames_pattern", "{file_name}/{frame:06d}.png").format(frame=last, **row))
            if not os.path.isfile(p):
                die(f"{label}: 프레임 예산 초과/패턴 오류 — {p} 가 없다 (start {st}, stride {stride}, n_frames {n})")
            rf = spec.get("raw_frames")
            if rf and last >= int(rf) + int(spec.get("frames_start", 0) == 1):
                die(f"{label}: 마지막 프레임 {last} 가 raw_frames {rf} 를 넘는다")

    for i, spec in enumerate(d["datasets"]):
        if spec.get("type") == "frames_index":
            check_frames(spec, f"data.datasets[{i}]({spec.get('name', '?')})")
            for k in META_KEYS:
                spec.pop(k, None)
        elif spec.get("type") == "video_csv":
            if not os.path.isfile(spec["csv"]):
                die(f"data.datasets[{i}].csv 가 없다 -> {spec['csv']}")
        else:
            die(f"data.datasets[{i}].type={spec.get('type')!r}; frames_index | video_csv")
    if cfg.get("val"):
        check_frames(cfg["val"]["data"], "val.data")
        cfg["val"]["data"].pop("raw_frames", None)

    os.makedirs(os.path.dirname(os.path.abspath(a.output)) or ".", exist_ok=True)
    yaml.safe_dump(cfg, open(a.output, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)

    if not a.quiet:
        O = cfg["optimization"]
        print(f"  config   : {cfg_path}")
        print(f"  folder   : {cfg['folder']}")
        print(f"  model    : base={m.get('base', '-')} {m.get('arch_name')} | load_predictor={bool(m.get('load_predictor'))} "
              f"| ctx<-{m.get('context_encoder_key')} tgt<-{m.get('target_encoder_key')} | predictor {m.get('predictor')}")
        print(f"  ckpt     : {m['checkpoint']}")
        for s in d["datasets"]:
            print(f"  data     : {s.get('name', s.get('type'))}  type={s['type']}  weight={s.get('weight', '-')}"
                  + (f"  limit={s['limit']}" if s.get("limit") else "")
                  + (f"  starts={s.get('frames_start_choices')}" if s.get("frames_start_choices") else ""))
        print(f"  frames   : n={n} res={res} | batch/rank {d['batch_size']} | workers {d.get('num_workers')} | aug {d.get('aug')}")
        print(f"  masks    : {cfg['mask']}")
        print(f"  optim    : epochs {O['epochs']} ipe {O.get('ipe') or 'auto'} lr {O['lr']} warmup {O.get('warmup')} wd {O.get('weight_decay')}")
        v = cfg.get("val")
        print(f"  val      : {v['dataset']} n_blocks_per_type={v.get('n_blocks_per_type')} C={v.get('context_length')} every {v.get('every_epochs')} ep"
              if v else "  val      : (없음)")


if __name__ == "__main__":
    main()
