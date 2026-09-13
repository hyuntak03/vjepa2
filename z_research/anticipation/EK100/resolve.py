#!/usr/bin/env python3
"""EK100 anticipation config 를 검사하고 `_resolved.yaml` 로 떨어뜨린다 — 모델 로딩 전에 죽으라고.

ViT-H 로딩은 프로세스당 ~2분이다. 경로 오타·해상도 불일치·프레임 예산 초과를 그 전에 잡는다
(`z_research/scripts/harness/resolve.py` 와 같은 역할·같은 관례).

    python z_research/anticipation/EK100/resolve.py ek100_vith --dry-run
    python z_research/anticipation/EK100/resolve.py ek100_vith --set data.num_workers=2 -o <dir>

`--set` 은 점 경로다. `experiment.` 접두어는 생략할 수 있다 (`data.*` / `optimization.*` /
`classifier.*` 는 자동으로 `experiment.` 아래로 붙는다). 값은 YAML 로 읽고 `null` 은 그 키를 지운다.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
MODULE_NONSQUARE = "vit_encoder_predictor_concat_ar_nonsquare"


def die(msg):
    print(f"[resolve] ERROR: {msg}", file=sys.stderr)
    sys.exit(2)


def set_path(cfg, dotted, raw):
    if dotted.split(".")[0] in ("data", "optimization", "classifier"):
        dotted = "experiment." + dotted
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            die(f"--set {dotted}: '{k}' 가 dict 가 아니다")
        node = node[k]
    val = yaml.safe_load(raw)
    if val is None:
        node.pop(keys[-1], None)
    else:
        node[keys[-1]] = val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="configs/<이름>.yaml 의 <이름>, 또는 yaml 경로")
    ap.add_argument("--set", action="append", default=[], metavar="a.b=1")
    ap.add_argument("--tag")
    ap.add_argument("--lr", type=float, help="probe head 를 하나로 두고 lr 을 이 값으로")
    ap.add_argument("--wd", type=float, help="probe head 를 하나로 두고 weight decay 를 이 값으로")
    ap.add_argument("-o", "--outdir", help="_resolved.yaml 을 쓸 디렉토리 (기본: folder/<eval>/<tag>)")
    ap.add_argument("--dry-run", action="store_true", help="검사만 하고 파일을 쓰지 않는다")
    a = ap.parse_args()

    f = Path(a.config)
    if not f.exists():
        f = HERE / "configs" / f"{a.config}.yaml"
    if not f.exists():
        avail = sorted(p.stem for p in (HERE / "configs").glob("*.yaml"))
        die(f"config 없음: {a.config} (있는 것: {', '.join(avail)})")
    cfg = yaml.safe_load(f.read_text())

    for kv in a.set:
        if "=" not in kv:
            die(f"--set 은 a.b=c 형식이다: {kv!r}")
        k, v = kv.split("=", 1)
        set_path(cfg, k.strip(), v.strip())
    if a.tag:
        cfg["tag"] = a.tag
    if a.lr is not None or a.wd is not None:
        base = cfg["experiment"]["optimization"]["multihead_kwargs"][0]
        lr = a.lr if a.lr is not None else base["lr"]
        wd = a.wd if a.wd is not None else base["weight_decay"]
        cfg["experiment"]["optimization"]["multihead_kwargs"] = [
            dict(weight_decay=wd, final_weight_decay=wd, lr=lr, start_lr=lr, final_lr=0.0, warmup=0.0)
        ]

    exp, data, opt = cfg["experiment"], cfg["experiment"]["data"], cfg["experiment"]["optimization"]
    mk = cfg["model_kwargs"]

    # ---------------------------------------------------------------- 실물 검사
    for key in ("dataset_train", "dataset_val"):
        if not Path(data[key]).exists():
            die(f"annotation 없음: {key} = {data[key]}")
    if not Path(mk["checkpoint"]).exists():
        die(f"checkpoint 없음: {mk['checkpoint']}")

    base = Path(data["base_path"])
    if not base.is_dir():
        die(f"base_path 없음: {base}")
    pat = "*/videos/*.MP4" if int(data.get("file_format", 1)) == 0 else "*/*.MP4"
    found = glob.glob(str(base / pat))
    if not found:
        die(f"base_path 아래에 비디오가 없다: {base / pat}  (file_format 를 확인할 것)")

    # 인덱스에 있는 비디오가 실제로 몇 개 있는지 (filter_annotations 가 없는 것은 조용히 버린다)
    import csv as _csv

    def vids(p):
        with open(p) as fh:
            return sorted({r["video_id"] for r in _csv.DictReader(fh)})

    stems = {Path(p).stem for p in found}
    tr, va = vids(data["dataset_train"]), vids(data["dataset_val"])
    miss_tr = [v for v in tr if v not in stems]
    miss_va = [v for v in va if v not in stems]

    # ---------------------------------------------------------------- 설정 정합성
    res = int(data["resolution"])
    width = data.get("width") or res
    patch = int(mk["pretrain_kwargs"]["encoder"]["patch_size"])
    tub = int(mk["pretrain_kwargs"]["encoder"]["tubelet_size"])
    module = mk["module_name"]
    nonsquare_module = module.endswith(MODULE_NONSQUARE)
    mask_index = mk.get("wrapper_kwargs", {}).get("mask_index", 1)

    if res % patch or width % patch:
        die(f"resolution/width 가 patch_size {patch} 로 안 나눠진다: {res}x{width}")
    if (width != res or mask_index != 1) and not nonsquare_module:
        die(
            f"width({width}) != resolution({res}) 또는 mask_index({mask_index}) != 1 인데 "
            f"module_name 이 정사각 판이다.\n"
            f"           → module_name 을 ...{MODULE_NONSQUARE} 로 바꿀 것 "
            f"(정사각 판의 predictor RoPE 는 grid_size**2 를 가정한다)"
        )
    if data["spatial_mode"] not in ("short_side", "cover", "center_crop", "letterbox"):
        die(f"spatial_mode 는 short_side|center_crop|letterbox: {data['spatial_mode']!r}")
    if data["spatial_mode"] == "center_crop" and width != res:
        die("spatial_mode: center_crop 은 정사각 전용이다 (비정사각은 cover 또는 letterbox)")
    if data.get("anticipation_point_mode", "released") not in ("released", "paper"):
        die(f"anticipation_point_mode 는 released|paper: {data['anticipation_point_mode']!r}")
    if data.get("time_source", "frame") not in ("frame", "timestamp"):
        die(f"time_source 는 frame|timestamp: {data['time_source']!r}")

    fpc = int(data["frames_per_clip"])
    if fpc % tub:
        die(f"frames_per_clip({fpc}) 이 tubelet_size({tub}) 로 안 나눠진다")

    # 프레임 예산: predictor 의 위치 공간 안에 (문맥 + anticipation skip + 예측) 이 들어가야 한다
    gh, gw = res // patch, width // patch
    n_ctx_tub = fpc // tub
    max_at = max(max(data["train_anticipation_time_sec"]), max(data["anticipation_time_sec"]))
    fps = int(data["frames_per_second"])
    skip_tub = int(max_at * fps / tub)
    n_out_tub = max(int(mk["wrapper_kwargs"]["num_output_frames"]), tub) // tub
    need_tub = n_ctx_tub + skip_tub + n_out_tub
    have_tub = int(mk["pretrain_kwargs"]["predictor"]["num_frames"]) // tub
    if need_tub > have_tub:
        die(
            f"프레임 예산 초과: 문맥 {n_ctx_tub} + skip {skip_tub} + 예측 {n_out_tub} = {need_tub} tubelet "
            f"> predictor num_frames/{tub} = {have_tub}"
        )

    n_tokens_ctx = n_ctx_tub * gh * gw
    n_tokens_probe = n_tokens_ctx + n_out_tub * gh * gw

    # ---------------------------------------------------------------- 출력
    gpus = int(os.environ.get("GPUS", 1))
    outdir = Path(a.outdir) if a.outdir else Path(cfg["folder"]) / cfg["eval_name"] / cfg["tag"]

    print("=" * 78)
    print(f"config          {f}")
    print(f"tag             {cfg['tag']}")
    print(f"모델            {mk['pretrain_kwargs']['encoder']['model_name']}  "
          f"({Path(mk['checkpoint']).parts[-4] if len(Path(mk['checkpoint']).parts) > 4 else mk['checkpoint']})")
    print(f"module          {module.split('.')[-1]}    mask_index={mask_index}"
          + ("   ⚠️ mask_tokens[1..9] 는 릴리즈 ckpt 에서 전부 0 이다" if mask_index != 0 else ""))
    print(f"입력            {fpc} frames @ {fps} fps  ->  {res}x{width}  (spatial_mode={data['spatial_mode']}, "
          f"train={data.get('train_spatial_mode')})")
    print(f"토큰            grid {gh}x{gw} x {n_ctx_tub} tubelet = {n_tokens_ctx} (문맥)  "
          f"-> probe 입력 {n_tokens_probe}")
    print(f"프레임 예산     {need_tub} / {have_tub} tubelet  (문맥 {n_ctx_tub} + skip {skip_tub} + 예측 {n_out_tub})")
    print(f"프로토콜        anticipation_point_mode={data.get('anticipation_point_mode', 'released')}  "
          f"time_source={data.get('time_source', 'frame')}")
    print(f"                train at={data['train_anticipation_time_sec']} ap={data['train_anticipation_point']}  |  "
          f"val at={data['anticipation_time_sec']} ap={data['val_anticipation_point']}")
    mh = opt["multihead_kwargs"]
    if len(mh) == 1:
        print(f"probe head      1 개   lr={mh[0]['lr']}  wd={mh[0]['weight_decay']}  "
              f"(블록 4 = self-attn 3 + cross-attn 1, query 3 -> verb/noun/action)")
    else:
        print(f"probe head      {len(mh)} 개 (lr x wd sweep — 지표는 head 중 max = val 로 고른 값)")
    ev = exp.get("evaluation") or {}
    print(f"지표            mean class recall@{ev.get('topk', 5)} (verb / noun / action)   "
          f"exact_val_pass={ev.get('exact_val_pass', False)}"
          + ("" if ev.get("exact_val_pass") else "   ⚠️ 공식 val 루프 — clip 중복·누락"))
    print(f"batch           rank 당 {opt['batch_size']} x GPUS={gpus} = global {opt['batch_size'] * gpus}"
          f"   (논문 global 128)")
    print(f"epoch           {opt['num_epochs']}   num_workers {data['num_workers']}")
    print(f"비디오          base_path 실측 {len(found)}개 | train 인덱스 {len(tr)} (없는 것 {len(miss_tr)}) "
          f"| val 인덱스 {len(va)} (없는 것 {len(miss_va)})")
    if data.get("limit_videos"):
        print(f"                ⚠️ limit_videos={data['limit_videos']} — 배관 점검 전용")
    if miss_tr[:3] or miss_va[:3]:
        print(f"                없는 예: {(miss_tr + miss_va)[:3]}")
    print(f"출력            {outdir}")
    print("=" * 78)

    if a.dry_run:
        print("[resolve] --dry-run: 파일을 쓰지 않고 끝낸다")
        return
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "_resolved.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print(f"[resolve] wrote {out}")
    print(str(out))  # run.sh 가 마지막 줄을 읽는다


if __name__ == "__main__":
    main()
