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
# 2026-10-16: 노드별 하네스가 자기 config 를 옆에 두고 `extends:` 로 공용 부모를 가져올 수 있게 탐색 경로를 연다.
#   TRAIN_CFG_DIR=<dir>  (예: z_vll7_training/configs) 를 먼저 보고, 없으면 configs/training 으로 떨어진다.
TRAIN_CFG_DIRS = [d for d in (os.environ.get("TRAIN_CFG_DIR"), TRAIN_CFG) if d]
sys.path.insert(0, f"{ROOT}/z_research/scripts/harness")
from resolve import parse_registry  # noqa: E402  (protocols 레지스트리와 같은 파서)

META_KEYS = ("available", "note")          # raw_frames 는 window_grid 의 frame_budget 이 쓰므로 남긴다


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
    if name_or_path.endswith((".yaml", ".yml")):
        path = name_or_path
    else:
        cands = [f"{d}/{name_or_path}.yaml" for d in TRAIN_CFG_DIRS]
        path = next((c for c in cands if os.path.isfile(c)), cands[0])
    if not os.path.isfile(path):
        avail = sorted({f[:-5] for d in TRAIN_CFG_DIRS if os.path.isdir(d)
                        for f in os.listdir(d) if f.endswith(".yaml") and not f.startswith("_")})
        die(f"config 없음 -> {path}\n  찾은 곳: {', '.join(TRAIN_CFG_DIRS)}\n  사용 가능: {', '.join(avail)}")
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
        if a.limit and spec.get("type") in ("frames_index", "frames_index_track", "blender_track"):
            spec["limit"] = int(a.limit)
        out.append(spec)
    vw = cfg["data"].get("var_window")
    if vw:   # 가변 창 로더 (app/vjepa_frozen/var_window.py) — data.datasets 대신 var_window.datasets 의 표를 쓴다
        if not vw.get("datasets"):
            die("data.var_window.datasets 가 비어 있다")
        for t in vw["datasets"]:
            tb = os.path.expandvars(str(t.get("table", "")))
            if not os.path.isfile(tb):
                die(f"data.var_window.datasets 의 표가 없다: {t.get('table')!r} "
                    "(z_training/data/build_fps4_table.py 로 만든다)")
            t["table"] = tb
        bx = vw.get("boundary")
        if bx:   # 2026-10-20: boundary (사건 경계 창 + 상태 라벨). mix=sum 이면 합성 보조 손실 로더가 쓴다
            mf = os.path.expandvars(str(bx.get("manifest", "")))
            if not os.path.isfile(mf):
                die(f"data.var_window.boundary.manifest 가 없다: {bx.get('manifest')!r} "
                    "(z_training/data/build_phys2d_mix_manifest.py 로 만든다)")
            bx["manifest"] = mf
            if str(bx.get("mix", "")) == "sum" and tuple(bx.get("crop_ratio", (1.0, 1.0))) != (1.0, 1.0) and bx.get("labels"):
                die("boundary.labels 가 있으면 crop_ratio 는 [1, 1] 이어야 한다 (좌표 라벨 — 로더도 거부한다)")
        if not cfg.get("optimization", {}).get("ipe"):
            die("data.var_window 는 optimization.ipe 가 필요하다 (epoch 길이를 데이터 크기로 정할 수 없다)")
    elif not out:
        die("data.datasets 가 비어 있다 — 학습 데이터는 아직 정하지 않았다. 돌릴 때 준다:\n"
            "  SET=\"data.datasets=[<이름>]\"  또는 config 의 data.datasets 를 채운다\n"
            f"  후보 (configs/training/datasets.md): {', '.join(sorted(DS))}\n"
            "  ⚠️ 채점 세트(v11 등)와 문맥을 공유하는 데이터로 학습하면 그 세트 점수는 학습셋 점수다")
    cfg["data"]["datasets"] = out

    # ── model.base -> models.md ───────────────────────────────────────────────────
    m = cfg["model"]
    if m.get("init_predictor"):                          # post-FT 출발점 (우리 run 의 predictor-only ckpt), 2026-10-09
        m["init_predictor"] = expand(m["init_predictor"])
        if not os.path.isfile(m["init_predictor"]):
            die(f"model.init_predictor 가 없다 -> {m['init_predictor']}")
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
    # 2026-10-16: val.data 를 **일부만** 줘도 되게 한다 (예: frames_stride 만). 레지스트리 값이 기본값으로 깔리고
    #   config 가 준 키가 이긴다. 전에는 val.data 가 하나라도 있으면 레지스트리 채움을 통째로 건너뛰어 root 가 없었다.
    if V and V.get("dataset"):
        PD = parse_registry(f"{ROOT}/configs/protocols/datasets.md", "root")
        if V["dataset"] not in PD:
            die(f"val.dataset='{V['dataset']}' 이 configs/protocols/datasets.md 에 없다")
        p = dict(PD[V["dataset"]])
        if p.get("available") is False:
            die(f"val.dataset='{V['dataset']}' 은 available: false 다")
        keep = ("root", "index_csv", "frames_root", "frames_pattern", "frames_start", "frames_stride",
                "block_column", "pair_column", "plausible_column", "type_column", "variant_column", "raw_frames")
        V["data"] = {**{k: p[k] for k in keep if k in p}, **(V.get("data") or {})}   # config 가 이긴다
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
        elif mk.get("type") == "prefix_window":
            T = n // tub
            cb = [int(x) for x in mk.get("context_blocks", [T // 2])]
            kb = [int(x) for x in mk.get("predict_blocks", [T // 2])]
            if any(not (0 < c < T) for c in cb) or any(k <= 0 for k in kb):
                die(f"mask.prefix_window: 0 < context_blocks < {T}, predict_blocks >= 1 여야 한다 ({cb}, {kb})")
            if not any(c + k <= T for c in cb for k in kb):
                die(f"mask.prefix_window: C+K <= {T} 인 조합이 없다 ({cb} x {kb})")
    pk = str((m.get("predictor") or {}).get("kind", "oneshot"))
    # 2026-09-27: Ariel 0927 이식 — prefix_full (문맥 양방향 · 미래 전부 양방향) · ar (팔 C, app/vjepa_frozen/ar.py) 추가
    #   (app/vjepa_frozen/utils.py 의 build_predictor whitelist 와 같은 집합)
    KINDS = ("oneshot", "prefix", "prefix_full", "causal", "ar", "ar_state", "ctx_ar", "ctx_state", "mret")  # utils.build_predictor 와 같은 집합
    if pk not in KINDS:
        die(f"model.predictor.kind 는 {' | '.join(KINDS)}: {pk!r}")
    if pk in ("ar", "ar_state", "ctx_ar", "ctx_state"):
        # 2026-09-27: kind=ar 의 전제를 DRYRUN 에서 잡는다. train.py 도 같은 것을 검사하지만
        #   거기서는 ViT-H 로딩(~2 분) **뒤**에 죽는다.
        meta = cfg.get("meta") or {}
        if not bool(meta.get("sync_masks", False)):
            die("kind=ar 는 meta.sync_masks: true 가 필요하다 — (C, K) 를 학습 루프가 step 당 한 번 뽑고 "
                "창을 C+K 블록으로 잘라야 ar_losses 가 돈다 (train.py 가 모델 로드 뒤에 같은 이유로 죽는다)")
        if d.get("window_grid"):
            die("kind=ar 는 data.window_grid 와 같이 못 쓴다 (window_grid 는 sync_masks 경로를 끈다)")
        bad = [mk.get("type", "temporal_prefix") for mk in (cfg.get("mask") or []) if mk.get("type") != "prefix_window"]
        if bad:
            die(f"kind=ar 는 mask.type=prefix_window 만 된다 (C·K 블록 수가 필요): {bad}")
        if not bool((cfg.get("loss") or {}).get("target_layer_norm", True)):
            die("kind=ar 는 loss.target_layer_norm: true 가 전제다 (입력·타깃·출력이 모두 LN 공간)")
        ck, tk = m.get("context_encoder_key", "encoder"), m.get("target_encoder_key", "target_encoder")
        if pk in ("ar", "ar_state") and ck != tk:
            die(f"kind={pk} 는 context_encoder_key == target_encoder_key 여야 한다 ({ck!r} != {tk!r}). "
                "AR 은 입력·타깃 모두 LN(target_encoder(블록)) 이라 문맥 encoder 는 안 쓰인다 — "
                "키가 다르면 안 쓰는 ViT-H 하나가 메모리만 먹는다")
        if pk in ("ctx_ar", "ctx_state") and ck == tk:
            die(f"kind={pk} 는 문맥을 **context encoder (online)** 로, 타깃을 target_encoder (EMA) 로 읽는다 — "
                f"두 키가 같다 ({ck!r}). surprise_c16t32 와 같은 역할 분리를 쓸 것")
    ick = m.get("predictor_init_checkpoint")
    if ick and not os.path.isfile(ick):
        die(f"model.predictor_init_checkpoint 가 없다 -> {ick}")
    if ick and not m.get("load_predictor"):
        die("model.predictor_init_checkpoint 는 load_predictor: true 일 때만 쓰인다")
    sc = str(((d.get("aug") or {}).get("square_crop")) or "none")
    if sc not in ("none", "center"):
        die(f"data.aug.square_crop 은 none | center: {sc!r}")
    if not cfg.get("mask"):
        die("mask 리스트가 비어 있다")

    def _xp(spec, *keys):
        """경로 값의 ${VAR} 를 환경변수로 푼다 (2026-10-16). main 은 vast 의 expand() 를 뺐는데 config 에는
        ${DATA_CSV} 가 남아 있어 natural_ar_state 등이 "csv 가 없다 -> ${DATA_CSV}/..." 로 죽었다.
        ${} 가 없는 절대경로는 그대로 통과한다 (무해)."""
        for k in keys:
            if isinstance(spec.get(k), str):
                spec[k] = os.path.expandvars(spec[k])

    def check_frames(spec, label, n=n):
        # n 기본값 = data.n_frames. val 은 **val.n_frames** 로 따로 줄 수 있다 (가변 창은 data.n_frames 가 최대 창이라 val 과 다르다)
        _xp(spec, "root", "frames_root")
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

    grid = d.get("window_grid")
    if grid:
        sys.path.insert(0, ROOT)
        from app.vjepa_frozen.data import WindowGridSampler   # 검증 로직을 한 곳에 둔다 (torch import 만, 모델 없음)
        for i, spec in enumerate(d["datasets"]):
            if spec.get("type") != "frames_index":
                die(f"data.window_grid 는 frames_index 데이터셋에서만 된다 (datasets[{i}] type={spec.get('type')})")
            if spec.get("raw_frames") and int(grid["raw_span"]) > int(spec["raw_frames"]):
                die(f"window_grid.raw_span({grid['raw_span']}) > datasets[{i}].raw_frames({spec['raw_frames']})")
            grid.setdefault("raw_frames", int(spec.get("raw_frames", 100)))
            st = int(spec.get("frames_start", 0))
            row = next(csv.DictReader(open(os.path.join(spec["root"], spec.get("index_csv", "index.csv")), newline="", encoding="utf-8")))
            last = os.path.join(spec["frames_root"], spec.get("frames_pattern", "{file_name}/{frame:06d}.png").format(frame=st + int(grid["raw_span"]) - 1, **row))
            if not os.path.isfile(last):
                die(f"datasets[{i}]: raw_span 마지막 프레임이 없다 -> {last}")
        try:
            gs = WindowGridSampler(grid, tub)
        except Exception as e:
            die(f"data.window_grid: {e}")
        if max(c["n_frames"] for c in gs.cells) > n:
            die(f"window_grid n_frames 최대 {max(c['n_frames'] for c in gs.cells)} > data.n_frames {n}")
        grid["_resolved"] = gs.describe()
    for i, spec in enumerate(d["datasets"]):
        if spec.get("type") in ("frames_index", "frames_index_track"):     # _track (2026-10-31): + VOBench 궤적 라벨 (label.json)
            check_frames(spec, f"data.datasets[{i}]({spec.get('name', '?')})")
            for k in META_KEYS:
                spec.pop(k, None)
        elif spec.get("type") == "blender_track":
            # 2026-10-31 (z_analysis/release/06_blender_track): 메타 npz (z_training/data/build_blender_track_index.py) + clip 폴더
            _xp(spec, "meta")
            if not os.path.isfile(spec["meta"]):
                die(f"data.datasets[{i}].meta 가 없다 -> {spec['meta']} (build_blender_track_index.py 를 먼저)")
            if (d.get("aug") or {}).get("random_resized_crop"):
                die("blender_track 은 data.aug.random_resized_crop 과 같이 못 쓴다 (라벨이 256² 원본 좌표)")
            for k in META_KEYS:
                spec.pop(k, None)
        elif spec.get("type") == "video_csv":
            _xp(spec, "csv")
            if not os.path.isfile(spec["csv"]):
                die(f"data.datasets[{i}].csv 가 없다 -> {spec['csv']}")
        elif spec.get("type") == "video_window":
            # 창이 행마다 박혀 있는 인덱스 (phys2d_v1 처럼 "사건이 예측 구간에만" 인 창 세트).
            # 모델 로드 전에 인덱스 실물 + 행의 n_frames 가 data.n_frames 와 같은지 본다.
            _xp(spec, "csv")
            if not os.path.isfile(spec["csv"]):
                die(f"data.datasets[{i}].csv 가 없다 -> {spec['csv']}")
            import csv as _csv
            with open(spec["csv"], newline="", encoding="utf-8") as _f:
                _rd = _csv.DictReader(_f)
                if not _rd.fieldnames or not {"path", "start", "stride", "n_frames"} <= set(_rd.fieldnames):
                    die(f"data.datasets[{i}].csv 에 path,start,stride,n_frames 컬럼이 있어야 한다 -> {spec['csv']}")
                _r0 = next(_rd, None)
            if _r0 is None:
                die(f"data.datasets[{i}].csv 가 비었다 -> {spec['csv']}")
            if int(_r0["n_frames"]) != n:
                die(f"data.datasets[{i}].csv 의 n_frames={_r0['n_frames']} != data.n_frames={n} "
                    f"— 인덱스를 그 길이로 다시 만들 것 (z_training/data/build_phys2d_window_index.py --n-frames {n})")
            if not os.path.isfile(_r0["path"]):
                die(f"data.datasets[{i}].csv 첫 행의 영상이 없다 -> {_r0['path']}")
        elif spec.get("type") == "phys2d_sup":
            # 2026-10-19: phys2d_sup 영상 + 상태 라벨 (state_head). 창 인덱스 (path,start,stride,n_frames) 또는 clip 목록 ("<mp4> 0")
            _xp(spec, "csv")
            if not os.path.isfile(spec["csv"]):
                die(f"data.datasets[{i}].csv 가 없다 -> {spec['csv']}")
            with open(spec["csv"]) as _f:
                _h = _f.readline(); _l = _f.readline()
            _p0 = (_l.split(",")[0] if _h.startswith("path,") else _h.split()[0]).strip()
            if not os.path.isfile(_p0) or not os.path.isfile(_p0[:-4] + "_states.npz"):
                die(f"data.datasets[{i}] 첫 영상 · 라벨이 없다 -> {_p0} (+ _states.npz)")
            if _h.startswith("path,") and int(_l.split(",")[3]) != n:
                die(f"data.datasets[{i}].csv 의 n_frames={_l.split(',')[3]} != data.n_frames={n}")
            if (d.get("aug") or {}).get("random_resized_crop"):
                die("phys2d_sup 는 data.aug.random_resized_crop 과 같이 못 쓴다 (라벨 좌표와 어긋난다)")
        elif spec.get("type") == "wall":
            spec["root"] = expand(spec["root"])
            for sub in ("obses_u8.npy", "states.pth", "actions.pth"):
                if not os.path.exists(os.path.join(spec["root"], sub)):
                    die(f"data.datasets[{i}] (wall) 에 {sub} 가 없다 -> {spec['root']}")
        elif spec.get("type") == "pusht":
            spec["root"] = expand(spec["root"])
            for sub in ("train/obses", "train/states.pth", "train/rel_actions.pth", "val/obses"):
                if not os.path.exists(os.path.join(spec["root"], sub)):
                    die(f"data.datasets[{i}] (pusht) 에 {sub} 가 없다 -> {spec['root']}")
        else:
            die(f"data.datasets[{i}].type={spec.get('type')!r}; frames_index | frames_index_track | blender_track | video_csv | video_window | phys2d_sup | pusht | wall")
    if cfg.get("val"):
        # ⚠️ 2026-10-18: encoder·predictor 는 **data.n_frames 로** 지어진다 (app/vjepa_frozen/utils.py
        #    build_encoders / build_predictor 의 num_frames). val 창이 그보다 길면 토큰 인덱스가 격자를 넘어
        #    학습을 다 끝낸 뒤 val 에서 **device-side assert** 로 죽는다 (실제로 겪었다: n_frames 16 + val 32).
        _vn = int(cfg["val"].get("n_frames", n))
        if _vn > n:
            die(f"val.n_frames={_vn} > data.n_frames={n} — 모델이 data.n_frames 로 지어져 val 에서 "
                f"인덱스가 격자를 넘는다. val.n_frames 와 val.context_length 를 줄일 것")
        check_frames(cfg["val"]["data"], "val.data", n=_vn)
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
        print(f"  masks    : {cfg['mask']}" if not d.get("window_grid") else f"  grid     : {d['window_grid'].get('_resolved')}  (mask.context_frames 는 무시된다)")
        print(f"  optim    : epochs {O['epochs']} ipe {O.get('ipe') or 'auto'} lr {O['lr']} warmup {O.get('warmup')} wd {O.get('weight_decay')}")
        v = cfg.get("val")
        print(f"  val      : {v['dataset']} n_blocks_per_type={v.get('n_blocks_per_type')} C={v.get('context_length')} every {v.get('every_epochs')} ep"
              if v else "  val      : (없음)")


if __name__ == "__main__":
    main()
