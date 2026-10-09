#!/usr/bin/env python3
"""cvpr 단일 resolver — 프로토콜 yaml + 레지스트리 (datasets · models · windows) + 덮어쓰기 -> 실행 config 하나.

    python cvpr/harness/resolve.py <프로토콜.yaml|eval/testbed> <데이터셋> [모델] -o out.yaml [--meta out.json]
        [--window 이름] [--set K=V ...] [--limit N] [--smoke] [--recache] [--tag T] [--outdir D] [--results legacy]
    python cvpr/harness/resolve.py --list

세 하네스 (world_model_analysis · analysis/intphys2 · EK100) 가 따로 갖고 있던 병합·--set·extends·실물 검사를
**이 파일 하나**로 모았다. 엔진은 프로토콜 yaml 의 `engine:` 이 정한다 (wma | intphys2 | ek100).

병합 규칙 (엔진 공통)
    1. 프로토콜 yaml 을 읽는다 (`extends:` 를 따라가 깊은 병합, 자식이 이기고 null = 키 삭제)
    2. 창 preset 을 편다: `window: <이름>` (cvpr/registry/windows.yaml) -> 엔진별 키 (data.n_frames · surprise.context_length · ...)
       `--window` / `--set window.<키>=값` 은 **이 단계 전에** preset 에 덮는다
    3. 레지스트리를 깐다 — data = {**datasets[이름], **프로토콜.data}, model = {**models[이름], **프로토콜.model}
       즉 **프로토콜이 이긴다**. 단, 모델 고유 키 (MODEL_OWNED: family · checkpoint · arch_name · img_size · patch_size · tubelet_size ·
       encoder key 둘 · predictor · window_size · uniform_power · use_rope · dual_encoder) 는 레지스트리에 있으면 **모델이 이긴다**
       (프로토콜의 model 블록은 V-JEPA 2 ViT-H 기준값이라 다른 모델에 씌우면 조용히 틀린 모델이 된다).
       `predictor.<키>` 점 키는 predictor 블록 위에, `surprise.<키>` 는 surprise 블록 위에 덮는다. md `img_size` 가 data.resolution 을 정한다.
       (옛 z_research/scripts/harness/resolve.py 와 **같은 규칙** — check_equivalence.py 가 그것을 검사한다)
    4. tag / output_dir 를 짓는다 (아래)
    5. `--set a.b=c` 점 경로 덮어쓰기 (값은 YAML, null = 삭제, 중간 dict 자동 생성)
    6. `fit_groups_sweep: auto|auto_conditions` 를 index.csv 의 group 값으로 편다 (probing)
    7. 모델을 로드하기 전에 실물 검사 (경로 · 인덱스 · resolution==img_size · 프레임 예산 · features 블록 ...)

이름 규칙
    tag        = <cache_tag>_<모델>[_w<창>][_smoke<N>]      토큰 캐시 이름. 프로토콜과 무관 — 같은 (데이터셋, 모델, 창) 이면 같은 캐시
    output_dir = ${CVPR_RESULTS}/<task>/[<프로토콜 stem>__]<데이터셋>_<모델>[_w<창>][_smoke<N>]
                 --results legacy  -> <datasets.legacy_results_root>/<legacy_name>__<데이터셋>_<모델>   (옛 폴더 규약 그대로)
    `_w<창>` 은 프로토콜 기본 창이 아닐 때만 붙는다 (캐시 서명이 context 길이를 안 보므로 창이 다르면 캐시를 갈라야 한다).

레지스트리 값의 `${CVPR_*}` 는 cvpr/env.sh 의 변수로 치환한다 (env 에 없으면 env.sh 를 source 해서 읽는다).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CVPR = HERE.parent
REPO = CVPR.parent
REG = CVPR / "registry"

META = ("raw_frames", "cache_tag", "legacy_results_root", "results_root", "available", "note", "engine",
        "sliding_frame_skips", "base", "overrides")
MODEL_OWNED = ("family", "checkpoint", "arch_name", "img_size", "patch_size", "tubelet_size",
               "context_encoder_key", "target_encoder_key", "predictor", "window_size",
               "uniform_power", "use_rope", "dual_encoder")
ENGINES = ("wma", "intphys2", "ek100")
VAR = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def die(msg: str):
    print(f"\nERROR: {msg}\n", file=sys.stderr)
    sys.exit(1)


# ── env.sh ───────────────────────────────────────────────────────────────────
def load_env():
    """CVPR_* 가 env 에 없으면 cvpr/env.sh 를 source 해서 가져온다 (python 만 단독으로 불러도 되게)."""
    if os.environ.get("CVPR_REPO"):
        return
    out = subprocess.check_output(["bash", "-c", f"source '{CVPR}/env.sh' >/dev/null 2>&1 && env -0"])
    for item in out.split(b"\0"):
        if b"=" in item:
            k, v = item.decode(errors="replace").split("=", 1)
            if k.startswith("CVPR_"):
                os.environ.setdefault(k, v)


def expand(v):
    """`${CVPR_X}` 치환. 문자열·dict·list 재귀."""
    if isinstance(v, str):
        def rep(m):
            val = os.environ.get(m.group(1))
            if val is None:
                die(f"레지스트리의 {m.group(0)} 가 env 에 없다 — cvpr/env.sh 를 보라")
            return val
        return VAR.sub(rep, v)
    if isinstance(v, dict):
        return {k: expand(x) for k, x in v.items()}
    if isinstance(v, list):
        return [expand(x) for x in v]
    return v


# ── 공용 도구 ────────────────────────────────────────────────────────────────
def merge(dst: dict, src: dict) -> dict:
    """깊은 병합. src 가 이긴다. dict 는 재귀, list/스칼라는 교체, **null 은 그 키를 지운다.**"""
    for k, v in src.items():
        if v is None:
            dst.pop(k, None)
        elif isinstance(v, dict) and isinstance(dst.get(k), dict):
            dst[k] = merge(dict(dst[k]), v)
        else:
            dst[k] = v
    return dst


def set_path(cfg: dict, key: str, raw):
    """`a.b.c=값` 덮어쓰기. 값은 YAML (이미 파싱돼 있으면 그대로). null = 삭제. 점 없는 최상위 키도 된다."""
    val = raw
    if isinstance(raw, str):
        try:
            val = yaml.safe_load(raw)
        except yaml.YAMLError:
            val = raw
    parts = key.split(".")
    cur = cfg
    for k in parts[:-1]:
        if not isinstance(cur.get(k), dict):
            cur[k] = {}
        cur = cur[k]
    if val is None:
        cur.pop(parts[-1], None)
    else:
        cur[parts[-1]] = val


def load_yaml(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_registry(name: str) -> dict:
    p = REG / f"{name}.yaml"
    if not p.is_file():
        die(f"레지스트리 없음 -> {p}")
    return load_yaml(p)


def resolve_entry(table: dict, name: str, what: str, seen=()) -> dict:
    """`base:`/`extends:` 상속을 풀어 한 dict 로."""
    if name not in table:
        die(f"{what} '{name}' 이 레지스트리에 없다\n  사용 가능: {', '.join(sorted(k for k in table if not k.startswith('_')))}")
    e = dict(table[name])
    parent = e.pop("base", None) or e.pop("extends", None)
    if parent:
        if parent in seen:
            die(f"{what} 상속 순환: {' -> '.join(seen + (name, parent))}")
        base = resolve_entry(table, parent, what, seen + (name,))
        e = merge(dict(base), e)
    return e


def find_protocol(spec: str) -> Path:
    """`eval/testbed` -> cvpr/eval/testbed/protocol.yaml, `eval/probing/attn_probe` -> .../attn_probe.yaml, 경로면 그대로."""
    p = Path(spec)
    if p.suffix in (".yaml", ".yml"):
        return p if p.is_absolute() else (Path.cwd() / p)
    cands = [CVPR / spec / "protocol.yaml", CVPR / f"{spec}.yaml"]
    for c in cands:
        if c.is_file():
            return c
    die(f"프로토콜 없음 -> {spec}\n  찾은 곳: {', '.join(str(c) for c in cands)}\n  목록: resolve.py --list")


def load_protocol(path: Path, seen=()) -> dict:
    c = load_yaml(path)
    base = c.pop("extends", None)
    if not base:
        return c
    bp = Path(base)
    bp = bp if bp.is_absolute() else (path.parent / bp)
    if bp.suffix not in (".yaml", ".yml"):
        bp = find_protocol(base)
    bp = bp.resolve()
    if str(bp) in seen:
        die(f"extends 순환: {' -> '.join(seen + (str(bp),))}")
    if not bp.is_file():
        die(f"extends 대상 없음 -> {bp}")
    return merge(load_protocol(bp, seen + (str(path.resolve()),)), c)


def task_rel(path: Path) -> str:
    """cvpr/eval/testbed/protocol.yaml -> 'eval/testbed'. cvpr 밖이면 'adhoc'."""
    try:
        return str(path.resolve().parent.relative_to(CVPR))
    except ValueError:
        return "adhoc"


# ── 창 preset ────────────────────────────────────────────────────────────────
def window_spec(name: str, overrides: dict) -> tuple[dict, str]:
    W = load_registry("windows")
    w = resolve_entry(W, name, "창") if name else {}
    if overrides:
        w = merge(dict(w), overrides)
    if not w:
        return {}, ""
    if "kind" not in w:
        die(f"창 '{name}' 에 kind 가 없다 (fixed | sliding | intphys2 | ek100)")
    # 임의 값으로 바꾼 창의 이름
    label = name or ""
    if overrides:
        k = w["kind"]
        if k == "fixed":
            label = f"c{w['context']}t{w['n_frames']}" + (f"s{w['stride']}" if "stride" in w else "")
        elif k == "sliding":
            label = "skip" + "-".join(map(str, w["frame_skips"])) + "_w" + "-".join(map(str, w["window_sizes"])) \
                if isinstance(w.get("frame_skips"), list) else f"{name}_custom"
        elif k == "intphys2":
            label = f"ip2w{w['window_size']}fps{w.get('target_fps')}"
        else:
            label = f"{name}_custom"
    return w, label


def apply_window(cfg: dict, w: dict, ds: dict, engine: str):
    """preset 을 엔진별 키로 편다. 프로토콜 cfg 를 직접 고친다 (레지스트리 병합 전)."""
    if not w:
        return
    k = w["kind"]
    if engine == "wma" and k == "fixed":
        d = cfg.setdefault("data", {})
        d["n_frames"] = int(w["n_frames"])
        if "stride" in w:
            d["frames_stride"] = int(w["stride"])
        if "frames_start" in w:
            d["frames_start"] = int(w["frames_start"])
        cfg.setdefault("model", {})["window_size"] = int(w.get("model_window", w["n_frames"]))
        if "surprise" in cfg:
            s = cfg["surprise"]
            s.pop("mode", None); s.pop("intphys1", None)          # 단창 고정
            s["context_length"] = int(w["context"])
        if "features" in cfg:
            cfg["features"]["context_length"] = int(w["context"])
        if "surprise" not in cfg and "features" not in cfg:
            die("fixed 창인데 프로토콜에 surprise: 도 features: 도 없다 — context 를 놓을 자리가 없다")
    elif engine == "wma" and k == "sliding":
        d = cfg.setdefault("data", {})
        d["n_frames"] = "RAW"
        d["frames_stride"] = int(w.get("data_stride", 1))
        skips = w.get("frame_skips", "dataset")
        if skips == "dataset":
            skips = ds.get("sliding_frame_skips")
            if not skips:
                die("창 frame_skips: dataset 인데 데이터셋 레지스트리에 sliding_frame_skips 가 없다")
        sizes = list(w["window_sizes"])
        s = cfg.setdefault("surprise", {})
        s["mode"] = "intphys1"
        ip = {"frame_skips": list(skips), "window_sizes": sizes, "context_mult": list(w["context_mult"]),
              "stride": int(w.get("stride", 2)), "context_reduce": w.get("context_reduce", "min"),
              "frame_budget": w.get("frame_budget", "official"), "aggregate": list(w.get("aggregate", ["avg"])),
              "dump_windows": bool(w.get("dump_windows", True)), "video_batch": int(w.get("video_batch", 6)),
              "max_batch": int(w.get("max_batch", 48))}
        s["intphys1"] = {**(s.get("intphys1") or {}), **ip}
        s.pop("context_length", None)
        cfg.setdefault("model", {})["window_size"] = int(w.get("model_window", max(sizes)))
    elif engine == "intphys2" and k == "intphys2":
        ws = int(w["window_size"])
        s = cfg.setdefault("surprise", {})
        s["window_size"] = ws
        s["context_length_sweep"] = [int(ws * f) for f in w["context_fracs"]]
        s["context_length"] = int(ws * float(w.get("context_frac", 0.5)))
        if "stride" in w:
            s["stride"] = int(w["stride"])
        cfg.setdefault("model", {})["window_size"] = ws
        d = cfg.setdefault("data", {})
        if "target_fps" in w:
            d["target_fps"] = float(w["target_fps"])
        if "frame_step" in w:
            d["frame_step"] = int(w["frame_step"])
    elif engine == "ek100" and k == "ek100":
        d = cfg.setdefault("experiment", {}).setdefault("data", {})
        for key in ("frames_per_clip", "frames_per_second", "train_anticipation_time_sec", "train_anticipation_point",
                    "anticipation_time_sec", "val_anticipation_point"):
            if key in w:
                d[key] = w[key]
        if "num_output_frames" in w:
            cfg.setdefault("model_kwargs", {}).setdefault("wrapper_kwargs", {})["num_output_frames"] = int(w["num_output_frames"])
    else:
        die(f"창 kind '{k}' 는 엔진 '{engine}' 에 못 쓴다")


# ── 엔진별 병합 ───────────────────────────────────────────────────────────────
def merge_wma(cfg, ds, md):
    """옛 resolve.py 와 같은 규칙 (주석은 그쪽 docstring)."""
    cfg["data"] = {**ds, **(cfg.get("data") or {})}
    md_surprise = {k.split(".", 1)[1]: md.pop(k) for k in list(md) if k.startswith("surprise.")}
    cfg["model"] = {**md, **(cfg.get("model") or {})}
    for k in MODEL_OWNED:
        if k in md:
            cfg["model"][k] = md[k]
    pred_over = {k.split(".", 1)[1]: cfg["model"].pop(k) for k in list(cfg["model"]) if k.startswith("predictor.")}
    if pred_over:
        cfg["model"]["predictor"] = {**(cfg["model"].get("predictor") or {}), **pred_over}
    if "img_size" in md:
        cfg["data"]["resolution"] = int(md["img_size"])
    if md_surprise:
        cfg["surprise"] = {**(cfg.get("surprise") or {}), **md_surprise}


def merge_intphys2(cfg, ds, md):
    """analysis/intphys2 통짜 yaml 과 같은 모양을 만든다. data/model 블록 규칙은 wma 와 같다."""
    cfg["data"] = {**ds, **(cfg.get("data") or {})}
    md_surprise = {k.split(".", 1)[1]: md.pop(k) for k in list(md) if k.startswith("surprise.")}
    cfg["model"] = {**md, **(cfg.get("model") or {})}
    for k in MODEL_OWNED:
        if k in md:
            cfg["model"][k] = md[k]
    pred_over = {k.split(".", 1)[1]: cfg["model"].pop(k) for k in list(cfg["model"]) if k.startswith("predictor.")}
    if pred_over:
        cfg["model"]["predictor"] = {**(cfg["model"].get("predictor") or {}), **pred_over}
    if "img_size" in md:
        cfg["data"]["img_size"] = int(md["img_size"])
    if md_surprise:
        cfg["surprise"] = {**(cfg.get("surprise") or {}), **md_surprise}


def merge_ek100(cfg, ds, md):
    """EK100 통짜 yaml (experiment.* / model_kwargs.*) 에 레지스트리 값을 꽂는다."""
    d = cfg.setdefault("experiment", {}).setdefault("data", {})
    for k in ("base_path", "dataset_train", "dataset_val", "file_format"):
        if k in ds:
            d[k] = ds[k]
    mk = cfg.setdefault("model_kwargs", {})
    if "checkpoint" in md:
        mk["checkpoint"] = md["checkpoint"]
    enc = mk.setdefault("pretrain_kwargs", {}).setdefault("encoder", {})
    if "arch_name" in md:
        enc["model_name"] = md["arch_name"]
    for k in ("patch_size", "tubelet_size", "use_rope", "uniform_power"):
        if k in md:
            enc[k] = md[k]
    if "predictor_checkpoint" in md:
        mk["predictor_checkpoint"] = md["predictor_checkpoint"]     # 엔진이 읽는지는 엔진 쪽 README 참고


# ── 실물 검사 ─────────────────────────────────────────────────────────────────
def check_wma(cfg, raw_frames):
    d, m = cfg["data"], cfg["model"]
    for label, p in (("model.checkpoint", m.get("checkpoint")),
                     ("model.predictor_checkpoint", m.get("predictor_checkpoint")),
                     ("data.root", d.get("root")), ("data.frames_root", d.get("frames_root"))):
        if p and not os.path.exists(p):
            die(f"{label} 이 없다 -> {p}")
    idx = os.path.join(d["root"], d.get("index_csv", "index.csv"))
    if not os.path.isfile(idx):
        die(f"index 가 없다 -> {idx}\n  z_research/scripts/data/build_*_index.py 를 먼저 돌려야 한다")
    if d.get("resolution") != m.get("img_size"):
        die(f"data.resolution({d.get('resolution')}) != model.img_size({m.get('img_size')}). "
            "작으면 CUDA assert, 크면 에러 없이 엉뚱한 구간을 잘라 쓴다")
    if raw_frames and isinstance(d.get("n_frames"), int):
        need = d.get("frames_start", 1) + (d["n_frames"] - 1) * d.get("frames_stride", 1)
        if need > raw_frames:
            die(f"프레임 예산 초과: 마지막 프레임 인덱스 {need} > raw_frames {raw_frames} "
                f"(n_frames={d['n_frames']}, stride={d.get('frames_stride', 1)})")
    if (cfg.get("probing") or {}).get("enabled") and "features" not in cfg:
        die("probing.enabled 인데 features: 블록이 없다 (surprise 로 폴백해 cache_dir KeyError 로 죽는다)")
    if cfg.get("surprise") and cfg["surprise"].get("mode", "single") not in ("single", "intphys1"):
        die(f"surprise.mode 는 single | intphys1 이다 -> {cfg['surprise'].get('mode')}")
    pairing = (cfg.get("scoring") or {}).get("pairing", "cross")
    if pairing not in ("matched", "cross"):
        die(f"scoring.pairing 은 matched | cross 다 -> {pairing}")


def check_intphys2(cfg):
    d, m, s = cfg["data"], cfg["model"], cfg.get("surprise") or {}
    for label, p in (("model.checkpoint", m.get("checkpoint")), ("model.predictor_checkpoint", m.get("predictor_checkpoint")),
                     ("data.root", d.get("root"))):
        if p and not os.path.exists(p):
            die(f"{label} 이 없다 -> {p}")
    if d.get("split") and not os.path.isdir(os.path.join(d["root"], d["split"])):
        die(f"data.root/split 이 없다 -> {os.path.join(d['root'], d['split'])}")
    if m.get("window_size") != s.get("window_size"):
        die(f"model.window_size({m.get('window_size')}) != surprise.window_size({s.get('window_size')})")
    if "img_size" in m and d.get("img_size") != m["img_size"]:
        die(f"data.img_size({d.get('img_size')}) != model.img_size({m['img_size']})")
    if s.get("context_length_sweep") and s.get("context_length") not in s["context_length_sweep"]:
        die(f"surprise.context_length {s.get('context_length')} 가 sweep {s['context_length_sweep']} 안에 없다")


def check_ek100(cfg, val_only=False):
    exp, data = cfg["experiment"], cfg["experiment"]["data"]
    mk = cfg["model_kwargs"]
    enc, prd, wrap = mk["pretrain_kwargs"]["encoder"], mk["pretrain_kwargs"]["predictor"], mk["wrapper_kwargs"]
    for key in ("dataset_train", "dataset_val"):
        if not os.path.exists(data[key]):
            die(f"annotation 없음: {data[key]}")
    if not os.path.exists(mk["checkpoint"]):
        die(f"checkpoint 없음: {mk['checkpoint']}")
    if not os.path.isdir(data["base_path"]):
        die(f"영상 폴더 없음 (노드 로컬): {data['base_path']}")
    res, width = int(data["resolution"]), int(data.get("width") or data["resolution"])
    patch, tub = int(enc["patch_size"]), int(enc["tubelet_size"])
    if res % patch or width % patch:
        die(f"해상도 {res}x{width} 가 patch {patch} 로 안 나눠진다")
    fpc, fps = int(data["frames_per_clip"]), int(data["frames_per_second"])
    if fpc % tub:
        die(f"frames_per_clip {fpc} 이 tubelet {tub} 로 안 나눠진다")
    max_at = max(list(data["train_anticipation_time_sec"]) + list(data["anticipation_time_sec"]))
    need = fpc // tub + int(max_at * fps / tub) + max(int(wrap["num_output_frames"]), tub) // tub
    have = int(prd["num_frames"]) // tub
    if need > have:
        die(f"프레임 예산 초과: {need} tubelet > predictor {have}")
    if data.get("spatial_mode") not in ("short_side", "center_crop", "letterbox"):
        die(f"spatial_mode: {data.get('spatial_mode')!r} (short_side | center_crop | letterbox)")


# ── probing sweep 확장 (wma) ─────────────────────────────────────────────────
def expand_fit_groups(cfg):
    P = cfg.get("probing") or {}
    if not (P.get("enabled") and P.get("fit_groups_sweep") in ("auto", "auto_conditions")):
        return
    dd = cfg["data"]
    ipath = os.path.join(dd["root"], dd.get("index_csv", "index.csv"))
    gcol = dd.get("group_column") or dd.get("variant_column") or "variant"
    tcol, want = dd.get("type_column"), set(P.get("block_types") or [])
    if not os.path.isfile(ipath):
        die(f"index 가 없다 -> {ipath}")
    seen = []
    with open(ipath, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if want and tcol and r.get(tcol) not in want:
                continue
            if r[gcol] not in seen:
                seen.append(r[gcol])
    cond = [[g] for g in sorted(seen)]
    P["fit_groups_sweep"] = cond if P["fit_groups_sweep"] == "auto_conditions" else [None] + cond


# ── --list ───────────────────────────────────────────────────────────────────
def list_all():
    DS, MD, W = load_registry("datasets"), load_registry("models"), load_registry("windows")
    print("프로토콜 (cvpr/<task>/<name>/*.yaml):")
    for p in sorted(glob.glob(str(CVPR / "*/*/*.yaml"))):
        rel = Path(p).relative_to(CVPR)
        try:
            eng = load_protocol(Path(p)).get("engine")
        except Exception:
            eng = None
        if not eng:
            continue                                   # engine 없는 yaml (readout/config.yaml 등) 은 프로토콜이 아니다
        print(f"  {str(rel):<40} engine={eng}")
    print("데이터셋:")
    for k, v in DS.items():
        bad = v.get("available") is False
        print(f"  {k:<24} {v.get('engine','wma'):<9}" + (f" [사용 불가] {str(v.get('note',''))[:70]}" if bad else ""))
    print("모델:")
    for k, v in MD.items():
        bad = v.get("available") is False
        fam = v.get("family", "vjepa2")
        print(f"  {k:<28} {fam:<9}" + (f" [사용 불가] {str(v.get('note',''))[:60]}" if bad else ""))
    print("창 (window):")
    for k, v in W.items():
        print(f"  {k:<20} {json.dumps({kk: vv for kk, vv in v.items() if kk != 'kind'}, ensure_ascii=False)[:90]}")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("protocol", nargs="?")
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("model", nargs="?", default="vith")
    ap.add_argument("-o", "--output")
    ap.add_argument("--meta", help="엔진·output_dir·tag 를 json 으로 (launch.sh 가 읽는다)")
    ap.add_argument("--window", default=None, help="창 preset 이름 (프로토콜의 window: 를 덮는다)")
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--limit", type=int, default=None, help="block/영상 N 개만 + tag/output_dir 에 _smoke{N}")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--recache", action="store_true")
    ap.add_argument("--tag"); ap.add_argument("--outdir")
    ap.add_argument("--suffix", default="", help="tag · output_dir 에 붙일 접미사 (예: _copy). 창 접미사 뒤, _smoke 앞")
    ap.add_argument("--results", default=os.environ.get("RESULTS_ROOT", "cvpr"), help="cvpr (기본) | legacy | <경로>")
    ap.add_argument("--val-only", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    load_env()
    if a.list:
        return list_all()
    if not (a.protocol and a.dataset and a.output):
        ap.error("<프로토콜> <데이터셋> [모델] -o <out.yaml> 이 필요하다 (목록: --list)")

    # 1. 프로토콜
    pf = find_protocol(a.protocol)
    cfg = expand(load_protocol(pf))
    engine = cfg.pop("engine", None)
    if engine not in ENGINES:
        die(f"프로토콜 {pf} 의 engine 이 {ENGINES} 중 하나여야 한다 -> {engine}")
    legacy_name = cfg.pop("legacy_name", None)
    default_window = cfg.pop("window", None)
    task = task_rel(pf)
    stem = pf.stem

    # 2. 레지스트리
    DS, MD = load_registry("datasets"), load_registry("models")
    ds_full = expand(resolve_entry(DS, a.dataset, "데이터셋"))
    md_full = expand(resolve_entry(MD, a.model, "모델"))
    for what, e in (("데이터셋", ds_full), ("모델", md_full)):
        if e.get("available") is False:
            die(f"{what} 은 available: false 다.\n  {e.get('note', '')}")
    ds_engine = ds_full.get("engine", "wma")
    if ds_engine != engine:
        die(f"데이터셋 '{a.dataset}' 은 engine {ds_engine} 용인데 프로토콜 engine 은 {engine} 이다")
    raw_frames = ds_full.get("raw_frames")
    cache_tag = ds_full.get("cache_tag", a.dataset)
    legacy_root = ds_full.get("legacy_results_root", f"{os.environ['CVPR_REPO']}/z_exp/world_model_analysis/results")
    ds = {k: v for k, v in ds_full.items() if k not in META}
    md = {k: v for k, v in md_full.items() if k not in META}

    # 3. 창 — window.* set 은 preset 에 먼저 덮는다
    win_sets = {}
    other_sets = []
    for kv in a.sets:
        if "=" not in kv:
            die(f"--set 은 KEY=VALUE 여야 한다 -> {kv}")
        k, v = kv.split("=", 1)
        if k == "window":
            a.window = v
        elif k.startswith("window."):
            set_path(win_sets, k[len("window."):], v)
        else:
            other_sets.append((k, v))
    wname = a.window or default_window
    w, wlabel = window_spec(wname, win_sets)
    apply_window(cfg, w, ds_full, engine)
    win_suffix = f"_w{wlabel}" if (wlabel and (wname != default_window or win_sets)) else ""
    if a.suffix:
        win_suffix += a.suffix if a.suffix.startswith("_") else "_" + a.suffix

    # heads=... 는 ek100 의 preset 선택이라 병합 전에 cfg 로 올린다 (SET="heads=sweep")
    for k, v in list(other_sets):
        if k == "heads":
            cfg["heads"] = v; other_sets.remove((k, v))
    # 4. 병합
    {"wma": merge_wma, "intphys2": merge_intphys2, "ek100": merge_ek100}[engine](cfg, ds, md)
    if engine == "wma" and cfg["data"].get("n_frames") == "RAW":
        if raw_frames is None:
            die(f"n_frames: RAW 인데 '{a.dataset}' 에 raw_frames 가 없다")
        cfg["data"]["n_frames"] = int(raw_frames)

    # 5. 이름
    name = f"{a.dataset}_{a.model}{win_suffix}"
    tag = a.tag or os.environ.get("TAG") or f"{cache_tag}_{a.model}{win_suffix}"
    if a.outdir or os.environ.get("OUTDIR"):
        output_dir = a.outdir or os.environ["OUTDIR"]
    elif a.results == "legacy":
        output_dir = f"{legacy_root}/{legacy_name or stem}__{a.dataset}_{a.model}{win_suffix}"
    else:
        root = os.environ["CVPR_RESULTS"] if a.results == "cvpr" else a.results
        output_dir = f"{root}/{task}/{'' if stem == 'protocol' else stem + '__'}{name}"
    if a.limit:
        tag += f"_smoke{a.limit}"
        output_dir += f"_smoke{a.limit}"

    if engine == "wma":
        cfg["tag"], cfg["output_dir"] = tag, output_dir
        if a.limit:
            cfg["limit"] = int(a.limit)
        if a.smoke:
            cfg["smoke"] = True
        if a.recache:
            cfg["recache"] = True
    elif engine == "intphys2":
        cfg["folder"], cfg["tag"] = os.path.dirname(output_dir), os.path.basename(output_dir)
        if a.limit or a.smoke:
            cfg.setdefault("evaluation", {})["limit_videos"] = int(a.limit or 8)
    else:  # ek100: 엔진이 <folder>/<eval_name>/<tag> 에 쓴다. 옛 이름은 <EK100/exp_results>/action_anticipation_frozen/ek100_vith
        ev = cfg.get("eval_name", "action_anticipation_frozen")
        if a.outdir or os.environ.get("OUTDIR"):
            cfg["folder"], cfg["tag"] = os.path.dirname(os.path.dirname(output_dir)), os.path.basename(output_dir)
        elif a.results == "legacy":
            cfg["folder"], cfg["tag"] = legacy_root, f"{a.dataset}_{a.model}{win_suffix}"
        else:
            cfg["folder"], cfg["tag"] = os.path.dirname(output_dir), os.path.basename(output_dir)
        if a.limit:
            cfg["tag"] += f"_smoke{a.limit}"
        elif a.smoke and not cfg["tag"].endswith("_smoke"):
            cfg["tag"] += "_smoke"
        tag = cfg["tag"]
        output_dir = os.path.join(cfg["folder"], ev, cfg["tag"])
        # probe head preset (옛 EK100/resolve.py --heads)
        heads, presets = cfg.pop("heads", "config"), cfg.pop("heads_presets", {}) or {}
        if heads != "config":
            if heads not in presets:
                die(f"heads={heads} 는 heads_presets {list(presets)} 에 없다")
            lrs, wds = presets[heads]["lrs"], presets[heads]["wds"]
            cfg["experiment"]["optimization"]["multihead_kwargs"] = [
                dict(lr=lr, start_lr=lr, final_lr=0.0, weight_decay=wd, final_weight_decay=wd, warmup=0.0) for wd in wds for lr in lrs]
        if a.smoke or a.limit:
            data, opt = cfg["experiment"]["data"], cfg["experiment"]["optimization"]
            data["limit_videos"] = int(a.limit or 4)
            if a.smoke:
                data["num_workers"] = 2
                opt.update(num_epochs=1, batch_size=2)
                cfg["resume_checkpoint"] = False
        if a.val_only:
            cfg["val_only"] = True

    # 6. 데이터셋이 요구하는 채점 옵션 (레지스트리 overrides) -> 그 위에 사용자 --set
    for k, v in (ds_full.get("overrides") or {}).items():
        set_path(cfg, k, v)
    for k, v in other_sets:
        set_path(cfg, k, v)
    # 7. probing sweep · 검사
    if engine == "wma":
        expand_fit_groups(cfg)
        check_wma(cfg, raw_frames)
    elif engine == "intphys2":
        check_intphys2(cfg)
    else:
        check_ek100(cfg, a.val_only)

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    if a.meta:
        Path(a.meta).write_text(json.dumps({"engine": engine, "output_dir": output_dir, "tag": tag,
                                            "protocol": str(pf), "task": task, "dataset": a.dataset, "model": a.model,
                                            "window": wname, "window_label": wlabel, "legacy_name": legacy_name},
                                           ensure_ascii=False, indent=1))
    if not a.quiet:
        print(f"  protocol : {task}/{stem}   engine={engine}   ({pf})")
        print(f"  dataset  : {a.dataset}   model: {a.model}" + (f"   family={md_full.get('family')}" if md_full.get('family') else ""))
        print(f"  window   : {wname or '-'}{' (+set)' if win_sets else ''}   {json.dumps({k: v for k, v in w.items() if k != 'kind'}, ensure_ascii=False)[:100] if w else ''}")
        if engine == "wma":
            d, m, s = cfg["data"], cfg["model"], cfg.get("surprise") or {}
            n, st, sr = d["n_frames"], d.get("frames_start", 1), d.get("frames_stride", 1)
            print(f"  frames   : n={n} start={st} stride={sr} -> raw {st}..{st + (n - 1) * sr} (raw_frames={raw_frames})  res={d.get('resolution')}")
            print(f"  model    : {m.get('arch_name')} dtype={m.get('dtype')} autocast={m.get('autocast', '-')} window_size={m.get('window_size')}"
                  + (f" predictor_ckpt={os.path.basename(str(m.get('predictor_checkpoint')))}" if m.get("predictor_checkpoint") else ""))
            if s:
                print(f"  surprise : mode={s.get('mode', 'single')} ctx={s.get('context_length', '-')} pairing={(cfg.get('scoring') or {}).get('pairing', 'cross')}"
                      + (f" grid={s['intphys1']['frame_skips']}x{s['intphys1']['window_sizes']}" if s.get("intphys1") else ""))
            if (cfg.get("probing") or {}).get("enabled"):
                P = cfg["probing"]
                print(f"  probing  : targets={list(P['targets'])} block_types={P.get('block_types')} optim={P.get('optim')} sweep={len(P.get('fit_groups_sweep') or [])}")
        elif engine == "intphys2":
            d, s = cfg["data"], cfg["surprise"]
            print(f"  data     : {d['root']}/{d.get('split')} fps={d.get('target_fps')} img={d.get('img_size')}")
            print(f"  surprise : window={s['window_size']} C={s.get('context_length')} sweep={s.get('context_length_sweep')} stride={s.get('stride')} LN={s.get('target_layer_norm')}")
        else:
            d = cfg["experiment"]["data"]
            print(f"  ek100    : {d['frames_per_clip']}f@{d['frames_per_second']}fps ap={d.get('anticipation_point_mode')} time={d.get('time_source')} val_at={d.get('anticipation_time_sec')}")
        print(f"  tag      : {tag}")
        print(f"  output   : {output_dir}")


if __name__ == "__main__":
    main()
