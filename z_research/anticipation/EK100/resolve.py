#!/usr/bin/env python3
"""EK100 config 를 덮어쓰기·검사하고 `<출력>/_resolved.yaml` 로 떨어뜨린다. run.sh 가 부른다 — 모델 로딩(~2분) 전에 죽으라고.

    python z_research/anticipation/EK100/resolve.py --dry-run
    python z_research/anticipation/EK100/resolve.py --set data.num_workers=2 --tag x --smoke

--set a.b=c   점 경로. data. / optimization. / classifier. / evaluation. 는 experiment. 를 생략해도 된다. 값은 YAML, null 은 키 삭제.
--smoke       비디오 4개 / 1 epoch / batch 2 / worker 2, tag 뒤에 _smoke, 이어하기 끔 (배관 점검)
--heads sweep 논문의 probe head 20개 (lr 5종 x wd 4종). 지표는 head 중 max = val 로 고른 값이 된다
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PAPER_LRS = [5e-3, 3e-3, 1e-3, 3e-4, 1e-4]
PAPER_WDS = [1e-4, 1e-3, 1e-2, 1e-1]
EXP_KEYS = ("data", "optimization", "classifier", "evaluation")


def die(msg):
    sys.exit(f"[resolve] ERROR: {msg}")


def set_path(cfg, dotted, raw):
    if dotted.split(".")[0] in EXP_KEYS:
        dotted = "experiment." + dotted
    *parents, last = dotted.split(".")
    node = cfg
    for k in parents:
        if not isinstance(node.get(k), dict):
            die(f"--set {dotted}: '{k}' 가 없거나 dict 가 아니다")
        node = node[k]
    val = yaml.safe_load(raw)
    if val is None:
        node.pop(last, None)
    else:
        node[last] = val


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
    ap.add_argument("config", nargs="?", default="ek100_vith", help="configs/<이름>.yaml 또는 yaml 경로")
    ap.add_argument("--set", action="append", default=[], metavar="a.b=1")
    ap.add_argument("--tag")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--heads", choices=["config", "sweep", "grid8"], default="config",
                    help="sweep = 논문 20 개 (lr 5 x wd 4) / grid8 = lr {3e-4, 1e-4} x wd 4 종 (RTX 4090 24 GB 에 들어가는 크기, 2026-09-15)")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--val-only", action="store_true", help="val 만 돈다: train 비디오가 base_path 에 없어도 된다")
    a = ap.parse_args()

    f = Path(a.config) if Path(a.config).suffix == ".yaml" else HERE / "configs" / f"{a.config}.yaml"
    if not f.exists():
        die(f"config 없음: {f}")
    cfg = yaml.safe_load(f.read_text())
    exp, data, opt = cfg["experiment"], cfg["experiment"]["data"], cfg["experiment"]["optimization"]

    if a.heads in ("sweep", "grid8"):
        lrs = PAPER_LRS if a.heads == "sweep" else [3e-4, 1e-4]     # lr 1e-3 은 head 1 개에서 80 iter 안에 붕괴 (README §7)
        opt["multihead_kwargs"] = [dict(lr=lr, start_lr=lr, final_lr=0.0, weight_decay=wd, final_weight_decay=wd, warmup=0.0)
                                   for wd in PAPER_WDS for lr in lrs]
    if a.smoke:
        data.update(limit_videos=4, num_workers=2)
        opt.update(num_epochs=1, batch_size=2)
        cfg["resume_checkpoint"] = False   # 배관 점검은 늘 처음부터
    for kv in a.set:
        if "=" not in kv:
            die(f"--set 은 a.b=c 형식이다: {kv!r}")
        k, v = kv.split("=", 1)
        set_path(cfg, k.strip(), v.strip())
    if a.tag:
        cfg["tag"] = a.tag
    if a.smoke and not cfg["tag"].endswith("_smoke"):
        cfg["tag"] += "_smoke"

    mk = cfg["model_kwargs"]
    enc, prd, wrap = mk["pretrain_kwargs"]["encoder"], mk["pretrain_kwargs"]["predictor"], mk["wrapper_kwargs"]

    # ---------------------------------------------------------------- 실물
    for key in ("dataset_train", "dataset_val"):
        if not Path(data[key]).exists():
            die(f"annotation 없음: {data[key]}")
    if not Path(mk["checkpoint"]).exists():
        die(f"checkpoint 없음: {mk['checkpoint']}")
    base = Path(data["base_path"])
    sub = "videos" if int(data.get("file_format", 1)) == 0 else ""

    def missing(csv_path):
        vids = sorted({r["video_id"] for r in csv.DictReader(open(csv_path))})
        return vids, [v for v in vids if not (base / v.split("_")[0] / sub / f"{v}.MP4").exists()]

    tr, miss_tr = missing(data["dataset_train"])
    va, miss_va = missing(data["dataset_val"])
    if miss_va or (miss_tr and not a.val_only):
        die(f"{base} 에 없는 비디오: train {len(miss_tr)} / val {len(miss_va)} (예: {(miss_tr + miss_va)[:3]}). "
            f"노드를 확인할 것 — 비디오는 노드 로컬 /data2 에 있다")

    # ---------------------------------------------------------------- 정합성
    res, width = int(data["resolution"]), int(data.get("width") or data["resolution"])
    patch, tub = int(enc["patch_size"]), int(enc["tubelet_size"])
    mask_index = int(wrap.get("mask_index", 1))
    nonsquare_module = mk["module_name"].endswith("_nonsquare")
    if res % patch or width % patch:
        die(f"해상도 {res}x{width} 가 patch {patch} 로 안 나눠진다")
    if (width != res or mask_index != 1) and not nonsquare_module:
        die("비정사각 입력이나 mask_index != 1 은 module_name ..._nonsquare 에서만 된다")
    if data["spatial_mode"] not in ("short_side", "center_crop", "letterbox"):
        die(f"spatial_mode: {data['spatial_mode']!r} (short_side | center_crop | letterbox)")
    if data["spatial_mode"] == "center_crop" and width != res:
        die("center_crop 은 정사각 전용이다")
    if data.get("anticipation_point_mode") not in ("released", "paper"):
        die(f"anticipation_point_mode: {data.get('anticipation_point_mode')!r} (paper | released)")
    if data.get("time_source") not in ("frame", "timestamp", "frame_fixfps"):
        die(f"time_source: {data.get('time_source')!r} (timestamp | frame | frame_fixfps)")
    fpc, fps = int(data["frames_per_clip"]), int(data["frames_per_second"])
    if fpc % tub:
        die(f"frames_per_clip {fpc} 이 tubelet {tub} 로 안 나눠진다")
    max_at = max(data["train_anticipation_time_sec"] + data["anticipation_time_sec"])
    need = fpc // tub + int(max_at * fps / tub) + max(int(wrap["num_output_frames"]), tub) // tub
    have = int(prd["num_frames"]) // tub
    if need > have:
        die(f"프레임 예산 초과: {need} tubelet > predictor {have}")

    # ---------------------------------------------------------------- 이어하기 가드
    outdir = Path(cfg["folder"]) / cfg["eval_name"] / cfg["tag"]
    old_cfg = outdir / "_resolved.yaml"
    resume_note = "새로 시작"
    if cfg.get("resume_checkpoint") and (outdir / "latest.pt").exists():
        resume_note = "latest.pt 에서 이어간다"
        if old_cfg.exists():
            old = flat(yaml.safe_load(old_cfg.read_text()))
            new = flat(cfg)
            diff = sorted(k for k in set(old) | set(new) if k not in ("val_only",) and old.get(k) != new.get(k))
            if diff:
                die(f"{outdir} 에 다른 설정으로 학습한 latest.pt 가 있다. 다른 TAG 를 쓸 것.\n"
                    + "\n".join(f"           {k}: {old.get(k)!r} -> {new.get(k)!r}" for k in diff))

    # ---------------------------------------------------------------- 요약
    mh = opt["multihead_kwargs"]
    heads = (f"1 개 (lr {mh[0]['lr']}, wd {mh[0]['weight_decay']})" if len(mh) == 1
             else f"{len(mh)} 개 sweep — 지표는 head 중 max")
    print("=" * 78)
    print(f"tag         {cfg['tag']}   ({resume_note})")
    print(f"모델        {enc['model_name']}  mask_index={mask_index}  no_predictor={wrap.get('no_predictor', False)}")
    print(f"입력        {fpc} frames @ {fps} fps, {res}x{width}, spatial={data['spatial_mode']}")
    print(f"프로토콜    ap={data['anticipation_point_mode']}  time={data['time_source']}  "
          f"train at {data['train_anticipation_time_sec']} ap {data['train_anticipation_point']}  "
          f"val at {data['anticipation_time_sec']} ap {data['val_anticipation_point']}")
    print(f"probe       blocks {exp['classifier']['num_probe_blocks']}, heads {heads}")
    print(f"batch       {opt['batch_size']} x {a.gpus} GPU = global {opt['batch_size'] * a.gpus}  (논문 128)   "
          f"epoch {opt['num_epochs']}")
    print(f"비디오      {base}  (train {len(tr) - len(miss_tr)}/{len(tr)} · val {len(va)}/{len(va)} 있음"
          + (", val 만 도는 실행이라 train 은 없어도 됨)" if miss_tr else ")")
          + (f"  ⚠️ limit_videos={data['limit_videos']}" if data.get("limit_videos") else ""))
    print(f"출력        {outdir}")
    print("=" * 78)
    if a.dry_run:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    old_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print(old_cfg)  # run.sh 가 마지막 줄을 읽는다


if __name__ == "__main__":
    main()
