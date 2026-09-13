#!/usr/bin/env python3
"""EK100 anticipation config 를 찍어낸다 — 20개 head 보일러플레이트를 손으로 쓰지 않기 위해.

`configs/*.yaml` 은 **이 스크립트의 산출물**이다. 값을 바꿀 때는 yaml 이 아니라 여기를 고치고
다시 돌린다 (yaml 을 직접 고쳐도 동작은 하지만 다음 실행에서 덮인다).

    python z_research/anticipation/EK100/make_configs.py            # configs/ 에 4개 생성
    python z_research/anticipation/EK100/make_configs.py --check    # 덮지 않고 차이만 본다

head 20개 = classifier lr 5종 x weight decay 4종 (논문 Table 19). eval.py 는 20개를 동시에
학습하고 **매 지표를 20개 중 max 로 보고**한다 (train_one_epoch/validate 의 `max([...])`).
즉 20개는 앙상블이 아니라 probe 하이퍼파라미터 sweep 이고, 보고 값은 그 sweep 의 best 다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

# ---------------------------------------------------------------- 고정 경로
ANNOT = "/data/hyuntak/project/2026/2027_cvpr/epic-kitchens-100-annotations"
# ⚠️ NFS 다. 비디오 700개 / 1.2T. 실제 학습 전에 노드 로컬로 옮길 것 (README §4).
EK100_ROOT = "/data/dataset/EPIC-KITCHENS"
CKPT = {
    "vith": REPO
    / "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots"
    / "b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth",
    "vitl": REPO
    / "checkpoint/models--facebook--vjepa2-vitl-fpc64-256/snapshots"
    / "b3c1679b7c34d3255ef3547f27c7b226aefab26f/original/model.pth",
}
ARCH = {"vith": "vit_huge", "vitl": "vit_large"}
RESULTS = REPO / "z_research/anticipation/EK100/exp_results"

# ---------------------------------------------------------------- 논문 Table 19
LRS = [5e-3, 3e-3, 1e-3, 3e-4, 1e-4]
WDS = [1e-4, 1e-3, 1e-2, 1e-1]

# 2026-09-13 사용자 결정: probe head 는 하나, lr/wd 고정 (위 sweep 은 heads="sweep" 로만).
# 값은 논문 grid 의 가운데를 골랐다 — 근거가 있는 값이 아니다. run.sh 의 LR=/WD= 로 바꾼다.
HEAD_LR = 1e-3
HEAD_WD = 1e-2

MODULE_SQUARE = "evals.action_anticipation_frozen.modelcustom.vit_encoder_predictor_concat_ar"
MODULE_NONSQUARE = "evals.action_anticipation_frozen.modelcustom.vit_encoder_predictor_concat_ar_nonsquare"


def multihead_yaml(indent="      "):
    out = []
    for wd in WDS:
        for lr in LRS:
            out += [
                f"{indent}- weight_decay: {wd}",
                f"{indent}  final_weight_decay: {wd}",
                f"{indent}  lr: {lr}",
                f"{indent}  start_lr: {lr}",
                f"{indent}  final_lr: 0.0",
                f"{indent}  warmup: 0.",
            ]
        out += [""]
    return "\n".join(out).rstrip() + "\n"


def single_head_yaml(lr, wd, indent="      "):
    return (
        f"{indent}- weight_decay: {wd}\n"
        f"{indent}  final_weight_decay: {wd}\n"
        f"{indent}  lr: {lr}\n"
        f"{indent}  start_lr: {lr}\n"
        f"{indent}  final_lr: 0.0\n"
        f"{indent}  warmup: 0.\n"
    )


def make(
    name,
    model="vith",
    resolution=256,
    width=None,          # None = 정사각 (짧은변을 256 에 맞추고 긴변만 자른다)
    spatial_mode="center_crop",   # 2026-09-13 사용자 결정: 당분간 공식 center_crop (short_side 는 SET 으로)
    train_spatial_mode="rrc",
    batch_size=16,
    num_epochs=20,
    num_workers=6,
    mask_index=1,
    anticipation_point_mode="paper",
    time_source="frame",
    limit_videos=None,
    heads="single",
    lr=None,
    wd=None,
    note="",
):
    nonsquare = width is not None and width != resolution
    module = MODULE_NONSQUARE if (nonsquare or mask_index != 1) else MODULE_SQUARE
    if heads == "sweep":
        mh = multihead_yaml()
    else:
        mh = single_head_yaml(HEAD_LR if lr is None else lr, HEAD_WD if wd is None else wd)
    wrapper = [
        "    no_predictor: false",
        "    num_output_frames: 2",
        "    num_steps: 1",
    ]
    if module == MODULE_NONSQUARE:
        wrapper += [f"    mask_index: {mask_index}"]

    txt = f"""# {name} — {note}
# 생성물이다. 고칠 때는 z_research/anticipation/EK100/make_configs.py 를 고치고 다시 돌린다.
# 실행:  GPUS=8 bash z_research/anticipation/EK100/run.sh {name}
#
# nodes/tasks_per_node 는 공식 submitit 경로(evals/main_distributed.py)용 필드다.
# 우리는 한 노드에서 evals/main.py 로 돌리므로 실제 world_size 는 GPUS 가 정한다.
nodes: 1
tasks_per_node: 8
cpus_per_task: 12
tag: {name}
eval_name: action_anticipation_frozen
folder: {RESULTS}
resume_checkpoint: true
experiment:
  classifier:
    num_probe_blocks: 4
    num_heads: 16
  data:
    dataset: EK100
    # $base_path/$participant_id/videos/$video_id.MP4  -> file_format 0 (실측 확인)
    file_format: 0
    base_path: {EK100_ROOT}
    dataset_train: {ANNOT}/EPIC_100_train.csv
    dataset_val: {ANNOT}/EPIC_100_validation.csv
    frames_per_clip: 32
    frames_per_second: 8
    resolution: {resolution}
    # -- spatial (추가 인자). center_crop = 공식, cover = 짧은변 맞추고 긴변만 자름
    spatial_mode: {spatial_mode}
    width: {width if width is not None else "null"}
    train_spatial_mode: {train_spatial_mode}
    # -- 프로토콜 변형 (README 차이 2·3). released/frame = 공식 구현 그대로
    anticipation_point_mode: {anticipation_point_mode}
    time_source: {time_source}
    # -- anticipation 시점
    train_anticipation_time_sec:
    - 0.25
    - 1.75
    train_anticipation_point:
    - 0.0
    - 0.25
    anticipation_time_sec:        # val. eval.py 가 val_anticipation_time_sec 로 읽는다
    - 1.0
    - 1.0
    val_anticipation_point:
    - 0.0
    - 0.0
    # -- 증강 (train 만)
    auto_augment: true
    motion_shift: false
    reprob: 0.25
    random_resize_scale:
    - 0.08
    - 1.0
    # -- 로더
    num_workers: {num_workers}
    pin_memory: true
    limit_videos: {limit_videos if limit_videos is not None else "null"}
  evaluation:
    # 보고 지표 = mean class recall@5 (verb / noun / action). 논문·EK100 공식 지표
    topk: 5
    # val 을 rank 별로 정확히 한 번 끝까지 돌고 all_reduce 한 번 (evals/action_anticipation_frozen/exact_val.py).
    # false 면 공식 validate 루프 — ipe 로 잘라 다시 감으므로 clip 이 중복·누락된다
    exact_val_pass: true
  optimization:
    num_epochs: {num_epochs}
    batch_size: {batch_size}          # rank 당. global = batch_size x GPUS
    use_bfloat16: true
    use_focal_loss: true              # alpha 0.25 / gamma 2.0 (losses.py 고정)
    multihead_kwargs:
{mh}model_kwargs:
  checkpoint: {CKPT[model]}
  module_name: {module}
  wrapper_kwargs:
{chr(10).join(wrapper)}
  pretrain_kwargs:
    encoder:
      model_name: {ARCH[model]}
      checkpoint_key: target_encoder
      tubelet_size: 2
      patch_size: 16
      uniform_power: true
      use_rope: true
    predictor:
      model_name: vit_predictor
      checkpoint_key: predictor
      num_frames: 64
      depth: 12
      num_heads: 12
      predictor_embed_dim: 384
      num_mask_tokens: 10
      uniform_power: true
      use_mask_tokens: true
      use_sdpa: true
      use_silu: false
      wide_silu: false
      use_rope: true
"""
    return txt


CONFIGS = {
    "ek100_vith": dict(
        model="vith", batch_size=16,
        note=("본 설정. ViT-H 256x256. spatial_mode=center_crop (공식: 짧은변 292 -> 가운데 256). "
              "anticipation_point_mode=paper 이므로 문맥이 action segment 시작 1초 전에서 끝난다"),
    ),
    "ek100_smoke": dict(
        model="vith", batch_size=2,
        num_epochs=1, num_workers=2, limit_videos=4,
        note="배관 점검. 비디오 4개 / 1 epoch / head 1개. 수치를 읽는 용도가 아니다",
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="쓰지 않고 차이만 보고한다")
    a = ap.parse_args()
    outdir = HERE / "configs"
    outdir.mkdir(parents=True, exist_ok=True)
    for name, kw in CONFIGS.items():
        txt = make(name, **kw)
        f = outdir / f"{name}.yaml"
        if a.check:
            old = f.read_text() if f.exists() else ""
            print(f"{'SAME ' if old == txt else 'DIFF '} {f}")
        else:
            f.write_text(txt)
            print(f"wrote {f}  ({len(txt.splitlines())} lines)")


if __name__ == "__main__":
    main()
