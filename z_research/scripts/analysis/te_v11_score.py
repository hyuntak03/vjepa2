#!/usr/bin/env python3
"""TrainingEffects — v11_split_test 를 **predictor 여러 개로 한 번에** 채점한다 (튜블릿 슬롯별 L1 + 복사 기준선).

encoder 는 rank 마다 **한 번만** 짓고 (clip 마다 context/target encoder 한 번), predictor 만 바꿔 끼운다.
수치 경로는 `surprise_c16t32` 그대로다 (`configs/protocols/surprise_c16t32.yaml` 을 resolve.py 로 병합해 쓴다):
fp32 가중치 + autocast fp16, raw 100 장 stride 3 → 32 장, 문맥 16 장 = 튜블릿 8 개, 미래 8 튜블릿,
target = LN(target_encoder(32 장 전부)) (affine 없는 토큰 LN), L1, mask_index 0, PNG 직독,
antialias=False bilinear 288→256 (`evals/world_model_analysis/data.py::WMADataset.clip`).

## 정의 (clip i, predictor k, 미래 슬롯 j = 0..7; 평균은 슬롯 j 의 S 토큰 × D 채널)

    z      = context_encoder(문맥 16 장)                       (online `encoder`, 최종 norm 포함)  (B, 8·S, D)
    h      = LN(target_encoder(32 장))                           (affine 없는 LN)                    (B, 16·S, D)
    p_k    = predictor_k(z, 문맥 idx, 미래 idx, mask_index=0)                                        (B, 8·S, D)
    zT     = LN(z)[문맥 마지막 튜블릿 7]                         (h6f_v11_baselines.py 와 같은 식)
    hT     = h[튜블릿 7]                                         (target 쪽 문맥 마지막 튜블릿)

    l1   [i,k,j] = mean | p_k[j] − h[8+j] |       ← 슬롯 평균 = 표준 surprise (per_video_surprise)
    copy [i,j]   = mean | zT − h[8+j] |            ← 예측 없는 복사 기준선 (문맥 encoder 쪽 마지막 관측)
    hcopy[i,j]   = mean | hT − h[8+j] |            ← 진짜 변화량 (target 쪽 마지막 문맥 튜블릿 대비)
    hstep[i,j]   = mean | h[8+j] − h[8+j−1] |      ← 진짜 한 걸음 (j=0 은 h[7] 기준 = hcopy[i,0])
    pz   [i,k,j] = mean | p_k[j] − zT |            ← p 가 복사에서 얼마나 떨어졌나
    pstep[i,k,j] = mean | p_k[j] − p_k[j−1] |      ← 예측한 한 걸음 (p_k[−1] := zT, 즉 pstep[:,:,0] = pz[:,:,0])

D·S 는 bundle 에서 온다 (ViT-H 1280·256 / ViT-B 768 / ViT-Tiny 192). 하드코딩 없음.

## kind=ar (자기회귀, Ariel 팔 C) — 2026-09-27 추가

AR predictor 는 mask token 규약 forward 가 막혀 있고 (RuntimeError) 문맥 · 타깃이 **블록 (튜블릿 2 장) 마다 따로** 인코딩한
LN(target_encoder) 공간이다. 정의는 `analysis/predictors/ar_scoring.py` 그대로 (표준 하네스 `eval.py` 의 AR 분기와 같은 함수):

    hb     = LN(target_encoder(블록 b 의 2 장))  블록마다 따로               (B, 16·S, D)   ← AR predictor 가 한 clip 에 1 개라도 있을 때만 계산
    p_ar   = rollout(hb[블록 0..7], 8 블록)       문맥 8 블록 → 미래 8 블록 자기회귀 (fp32 가중치 + autocast fp16, 표준과 같다)
    l1   [i,k,j] = mean | p_ar[j] − hb[8+j] |      ← 같은 배열 (슬롯 = 미래 블록). 슬롯 평균 = 하네스 per_video_surprise (AR 판)
    pz   [i,k,j] = mean | p_ar[j] − hb[7] |        ← AR 의 '복사' 는 LN(z)[7] 이 아니라 hb[7] (AR 문맥 = 타깃 공간)
    pstep[i,k,j] = mean | p_ar[j] − p_ar[j−1] |    (p_ar[−1] := hb[7])
    copy_blk[i,j] = mean | hb[7] − hb[8+j] |       ← **AR 전용 복사 기준선** (n, 8). AR predictor 가 목록에 있을 때만 생긴다

⚠️ 공유 열 copy · hcopy · hstep 은 **표준 타깃** (LN(target_encoder(32 장))[미래]) 의 것이라 AR 의 l1 과 **같은 공간이 아니다** —
   AR 행의 l1 과 비교하지 말 것. AR 의 복사 대조는 copy_blk. 다른 kind 와는 **쌍 정확도만** 비교한다 (크기 · margin 비교 금지).
   어느 predictor 가 어느 공간인지는 meta.json `pred_space` (`window` | `ar_block`) 와 summary.json 에 적힌다.
⚠️ AR 이 목록에 없으면 배열 · 수치 · 재개 규칙이 전부 이전과 같다 (hb 를 계산하지 않고 copy_blk 도 없다).

## predictor 바꿔 끼우기

`load_pred()` 는 `analysis/intphys2/model.py::build_from_config` 의 predictor 분기를 그대로 따른다:
`_build_predictor(num_frames=bundle.num_frames (=window 32), encoder_embed_dim=bundle.embed_dim,
predictor 제원 = 병합 config 의 model.predictor, use_rope/uniform_power = model 값,
kind = 체크포인트 arch.kind (없으면 default), kind 전용 키 = arch 의 prefix_impl/n_registers)`
→ `load_state_dict(strict=True)` → bundle.dtype (fp32). **kind 는 체크포인트에서만 온다** (Ariel = prefix).

predictor 목록 (`--preset` 또는 `--preds`):
  final  = release, v11_e10, ip1_e40, ip2_e80, pv1_e15, ariel_prefix_ep45
  curves = release, v11_e{1,2,3,5,10}, pv1_e{1,3,5,10,15}, ip1_e{5,10,20,40}, ip2_e{10,40,80}
  own    = bundle 이 병합 config 대로 지은 predictor 그 자체 (그룹 B 기본값)
  --preds 는 콤마 목록: 카탈로그 태그 (release, v11_eN, pv1_eN, ip1_eN, ip2_eN, ariel_prefix_ep45, ariel_full_ep40,
          ariel_ar_ep18) | own | tag=/abs/path.pt
  카탈로그 경로: release = z_training/runs/release_vith/latest.pt (model.pth predictor 추출본),
  v11/pv1/ip1/ip2 = z_training/runs/{v11,predictor_v1,intphys1,intphys2}_postft/e{N}.pt,
  ariel_prefix_ep45 = z_training/ariel/block_causal_future_only_1e_5/latest.pt   (arch.kind prefix, 팔 B, epoch 45/45)
  ariel_full_ep40   = z_training/ariel/full_attn_future_pred_1e_5/latest.pt      (arch.kind oneshot, 팔 A, epoch 40/40)
  ariel_ar_ep18     = z_training/ariel/ar_future_1e_5/latest.pt                  (arch.kind ar, 팔 C, epoch 18/40 학습 중간 — 위 AR 절)
  ⚠️ 2026-09-27 — 옛 태그 `ariel_ep{5,19,30,41,43}` (`block_causal_future_only_1e_5_ep_{N}`) 는 **파일이 지워져** 카탈로그에서 뺐다.
     ep43 을 ep45 로 바꿔 부르지 않는다 (다른 파일이다). 옛 산출물 (meta.json · 배열 · summary.json 의 `ariel_ep*` 태그) 은 그대로
     읽힌다 — 검증 참조 (`ref_files`) 도 옛 태그를 계속 안다. preset 변경: final 의 `ariel_ep43` → `ariel_prefix_ep45`,
     curves 의 `ariel_ep{5,19,30,43}` 은 **대체 곡선이 없어 뺐다** (새 run 은 epoch 별 체크포인트가 없다).
     옛 폴더 (`v11score_vith_{final,curves}`) 에 새 preset 을 치면 meta 불일치로 죽는다 (덮어쓰지 않는다) → `--out`/`--name` 을 바꿀 것.
그룹 B (`--model vitb_* / vittiny_*`, models.md 등록) 는 기본이 `own` 하나 (+ copy). 경로 predictor 를 주면
체크포인트의 base_checkpoint 가 모델과 다를 때 죽는다 (`--allow-foreign-base` 로만 푼다).

## CLI

    PY=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
    # 검증 (GPU 1 장, 48 clip; 결과 → z_research/TrainingEffects/v11score/validate_<model>.json)
    CUDA_VISIBLE_DEVICES=0 $PY z_research/scripts/analysis/te_v11_score.py --model vith --validate
    CUDA_VISIBLE_DEVICES=0 $PY z_research/scripts/analysis/te_v11_score.py --model vittiny_k400 --validate
    # 본 실행 (visible GPU 앞 N 장을 cuda:0..N-1 로 mp.spawn)
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $PY z_research/scripts/analysis/te_v11_score.py --model vith --preset final --gpus 8
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $PY z_research/scripts/analysis/te_v11_score.py --model vith --preset curves --gpus 8
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $PY z_research/scripts/analysis/te_v11_score.py --model vitb_k400_e240 --gpus 8
  주요 옵션: --batch 8 (rank 당) · --workers 6 (rank 당 PNG 디코드 프로세스) · --threads 4 · --limit N (block 단위로
  약 N clip, 출력 폴더에 _limitN) · --out DIR · --name (폴더 이름의 preset 자리).

## 출력 — /data2/local_datasets/world/world_analysis/cache/training_effects/v11score_<model>_<preset>/

    l1.npy    (n, K, 8) float32      pz.npy (n, K, 8) · pstep.npy (n, K, 8)
    copy.npy  (n, 8)    float32      hcopy.npy (n, 8) · hstep.npy (n, 8)
    copy_blk.npy (n, 8) float32      AR predictor 가 목록에 있을 때만 (2026-09-27, 위 AR 절)
    done.npy  (n,)      uint8        1 = 그 행이 다 쓰였다 (배열 flush 뒤에 표시 → 죽어도 반쪽 행이 done 이 되지 않는다)
    meta.json                        video_id 순서 (= index_test.csv 순서) · block/pair/plausible/condition/… 열 · predictor 태그·경로·
                                     arch.kind·epoch·크기·mtime · 병합 config 전체 · D · S · 정의
    _resolved.yaml                   병합 config (resolve.py 산출물 그대로)
    progress.jsonl                   rank 마다 시작/끝·clip 당 초 (append 전용)
    summary.json                     다 끝나면 한 번: predictor 마다 matched-pair 정확도 (슬롯 평균 = 표준 surprise) + copy
  **재개**: 같은 명령을 다시 치면 meta.json 이 같은지 (video_id · predictor 태그·경로 · 병합 config) 검사하고 done==0 행만
  다시 돈다 (world size 가 달라도 된다 — 남은 행을 rank 로 다시 나눈다). 다르면 죽는다 (덮어쓰지 않는다).
  clip 순서는 결정적: 남은 행을 index 순서대로 rank 에 round-robin.

## 검증 (2026-09-25, vll5 GPU 0 = 다른 eval 과 공유한 RTX 4090, batch 8)

  결과 파일: z_research/TrainingEffects/v11score/validate_{vith,vittiny_k400,vitb_k400_e240}.json (수치는 거기서 다시 읽을 것)
  --validate 는 참조 per_block.json 전부에 4 clip 이 다 있는 block 중 고르게 12 block = 48 clip 을 쓴다.
  (a) p 비트 대조 (첫 batch 8 clip):
      release 추출본 가중치 vs model.pth predictor              max|Δw| = 0.0
      release (바꿔 끼운 경로) vs build_from_config 기본 predictor  max|Δp| = 0.0
      v11_e10    vs build_from_config({predictor_checkpoint}) (CPU 에서 짓고 GPU 로)  max|Δp| = 0.0  (VisionTransformerPredictor)
      ariel_ep43 vs 〃  max|Δp| = 0.0  (arch.kind=prefix → VisionTransformerPredictorPrefix, n_registers 0)
  (b)–(d) 슬롯 평균 l1 vs 하네스 per_video_surprise, 48 clip (기준 ≤ 1e-3):
      release    vs v11_vith + v11_earlymid                       max|Δ| 2.81e-4 · mean|Δ| 9.7e-5   PASS
      v11_e10    vs v11_postft/eval/..._v11_split_test_e10         max|Δ| 3.53e-4 · mean|Δ| 1.2e-4   PASS
      ariel_ep43 vs ..._v11_split_test_vith_ariel_bc_ep43          max|Δ| 2.95e-4 · mean|Δ| 9.4e-5   PASS
      vittiny_k400 own vs ..._v11_split_test_vittiny_k400          max|Δ| 1.04e-5                    PASS
      vitb_k400_e240 own vs ..._v11_split_test_vitb_k400_e240      max|Δ| 1.42e-4                    PASS
      본 실행 경로 (mp.spawn + memmap, 64 clip, curves 22 개) 도 참조가 있는 7 개 (release, v11_e10, pv1_e15, ip1_e40,
      ip2_e80, ariel_ep30 (6조건 실행), ariel_ep43) 전부 max|Δ| ≤ 4.9e-4 — 행 배치가 맞다.
  남는 차이는 fp16 batch 구성 잡음이다: 같은 16 clip 을 batch 16 vs 8 / 4 / 1 로 돌리면 슬롯 평균이 최대 2.5e-4 / 4.9e-4 /
  4.1e-4 움직인다 (슬롯 하나는 ~1.3e-3). 같은 batch 에서는 하네스 식 (gather → LN → _distance) 과 3e-8, flat 인덱스로
  독립 재계산한 l1·copy·hcopy·hstep·pz·pstep 과 ≤ 4.5e-8 로 일치한다.
  copy 슬롯 평균 (48 clip): ViT-H 0.8224 (p release 0.5845) · ViT-B K400 e240 0.5448 (own 0.5429) · Tiny K400 0.1648 (own 0.2842).
  재개: done 11 행을 지우고 다시 돌리면 그 11 행만 다시 계산, 나머지 행은 비트 동일. 다른 preset 으로 같은 폴더에 쓰면 죽는다.
  속도 (공유 GPU, steady, rank 당): ViT-H final (6) 0.199 s/clip · curves (22) 0.363 s/clip · ViT-B own 0.067 · Tiny own 0.024
  (Tiny 는 PNG 디코드 한계 ~0.035). 모델 적재 58–72 s. 최대 메모리 ViT-H final 6.6 GB · curves 8.0 GB · ViT-B 1.6 · Tiny 0.6.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from analysis import predictors as PV  # noqa: E402
from analysis.intphys2.model import _build_predictor, _clean_backbone_key, build_from_config  # noqa: E402
from analysis.intphys2.surprise import _context_target_indices  # noqa: E402
from analysis.predictors import ar_scoring as ARS  # noqa: E402  (kind=ar 채점 — 2026-09-27)
from evals.world_model_analysis.data import WMADataset  # noqa: E402

PROTOCOL, DATASET = "surprise_c16t32", "v11_split_test"
RESOLVE = REPO / "z_research/scripts/harness/resolve.py"
OUT_ROOT = Path("/data2/local_datasets/world/world_analysis/cache/training_effects")
DOC_DIR = REPO / "z_research/TrainingEffects/v11score"
ZR = REPO / "z_training/runs"
AR = REPO / "z_training/ariel"
VX = REPO / "z_research/IntPhysGenV11/exp_results"
TA = REPO / "z_research/IntPhysGenV11_occlusion_timing_ablation/exp_results"
N_SLOT = 8
ARR_K = ("l1", "pz", "pstep")                 # (n, K, 8)
ARR_1 = ("copy", "hcopy", "hstep")            # (n, 8)
ARR_AR = ("copy_blk",)                        # (n, 8) — kind=ar predictor 가 있을 때만 (2026-09-27)
META_COLS = ("video_id", "block_id", "pair_id", "variant", "plausible", "condition", "motion", "violation_type",
             "sym_k", "occ_timing", "has_occlusion", "direction", "shape_pre", "color_pre", "shape_post", "color_post")
RUNS = {"v11": "v11_postft", "pv1": "predictor_v1_postft", "ip1": "intphys1_postft", "ip2": "intphys2_postft"}
EPOCHS = {"v11": range(1, 11), "pv1": range(1, 16), "ip1": range(5, 41, 5), "ip2": range(5, 81, 5)}


# Ariel 세 팔 (2026-09-27 수령, 최종 · 진행 중 체크포인트). 태그 → (폴더, models.md 레지스트리 이름).
# ⚠️ 옛 `ariel_ep{5,19,30,41,43}` (block_causal_future_only_1e_5_ep_{N}) 은 2026-09-27 에 파일이 지워져 카탈로그에서 뺐다.
#    ep43 을 ep45 로 바꿔 부르지 않는다. 옛 산출물의 `ariel_ep*` 태그는 meta 로 그대로 읽히고, ref_files 도 옛 태그를 안다.
ARIEL = {"ariel_prefix_ep45": ("block_causal_future_only_1e_5", "vith_ariel_prefix_ep45"),   # kind prefix (팔 B)
         "ariel_full_ep40": ("full_attn_future_pred_1e_5", "vith_ariel_full_ep40"),          # kind oneshot (팔 A)
         "ariel_ar_ep18": ("ar_future_1e_5", "vith_ariel_ar_ep18")}                           # kind ar (팔 C, 학습 중간)
ARIEL_REMOVED = re.compile(r"ariel_ep\d+")


def catalog() -> dict:
    c = {"release": ZR / "release_vith/latest.pt"}
    for fam, run in RUNS.items():
        for n in EPOCHS[fam]:
            c[f"{fam}_e{n}"] = ZR / run / f"e{n}.pt"
    for tag, (d, _) in ARIEL.items():
        c[tag] = AR / d / "latest.pt"
    return c


PRESETS = {
    # 2026-09-27: final 의 ariel_ep43 → ariel_prefix_ep45 (같은 run 의 마지막 epoch, 다른 파일).
    "final": ["release", "v11_e10", "ip1_e40", "ip2_e80", "pv1_e15", "ariel_prefix_ep45"],
    # 2026-09-27: ariel_ep{5,19,30,43} 은 파일이 지워졌고 새 run 에는 epoch 별 체크포인트가 없어 곡선에서 뺐다 (대체 없음).
    "curves": ["release", "v11_e1", "v11_e2", "v11_e3", "v11_e5", "v11_e10",
               "pv1_e1", "pv1_e3", "pv1_e5", "pv1_e10", "pv1_e15",
               "ip1_e5", "ip1_e10", "ip1_e20", "ip1_e40", "ip2_e10", "ip2_e40", "ip2_e80"],
    "own": ["own"],
}
VAL_DEFAULT = {"vith": ["release", "v11_e10", "ariel_prefix_ep45"]}   # 2026-09-27: ariel_ep43 (지워짐) → ariel_prefix_ep45


def die(msg: str):
    print(f"\nERROR: {msg}\n", file=sys.stderr, flush=True)
    sys.exit(1)


# ----------------------------------------------------------------------------- config
def resolve_cfg(model: str) -> tuple[dict, str]:
    """resolve.py surprise_c16t32 v11_split_test <model> → (config dict, yaml 원문). /dev/shm 의 고유 임시 폴더 사용."""
    tmpd = tempfile.mkdtemp(prefix="te_v11score_resolve_", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)
    try:
        out = os.path.join(tmpd, "resolved.yaml")
        env = {k: v for k, v in os.environ.items() if k not in ("TAG", "OUTDIR")}
        r = subprocess.run([sys.executable, str(RESOLVE), PROTOCOL, DATASET, model, "-o", out, "--quiet"],
                           env=env, capture_output=True, text=True)
        if r.returncode != 0:
            die(f"resolve.py 실패 ({model}):\n{r.stderr}")
        text = open(out, encoding="utf-8").read()
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)
    cfg = yaml.safe_load(text)
    m, s, d = cfg["model"], cfg["surprise"], cfg["data"]
    want = [(m.get("dtype"), "float32"), (str(m.get("autocast")).lower(), "float16"), (s.get("context_length"), 16),
            (d.get("n_frames"), 32), (s.get("distance"), "l1"), (float(s.get("loss_exp")), 1.0),
            (bool(s.get("target_layer_norm")), True), (int(s.get("mask_index")), 0), (bool(m.get("dual_encoder")), True),
            (m.get("context_encoder_key"), "encoder"), (m.get("target_encoder_key"), "target_encoder"),
            (int(d.get("frames_stride")), 3), (int(d.get("resolution")), int(m.get("img_size")))]
    bad = [(g, w) for g, w in want if g != w]
    if bad:
        die(f"병합 config 가 surprise_c16t32 규격과 다르다: (got, want) = {bad}")
    return cfg, text


def parse_preds(a) -> tuple[list, str]:
    """→ ([(tag, path|None)], preset 이름)."""
    cat = catalog()
    if a.preds:
        items, name = [x.strip() for x in a.preds.split(",") if x.strip()], (a.name or "custom")
    else:
        pre = a.preset or ("final" if a.model == "vith" else "own")
        if pre not in PRESETS:
            die(f"--preset {pre!r} 없음; {sorted(PRESETS)}")
        items, name = PRESETS[pre], (a.name or pre)
    out, seen = [], set()
    for it in items:
        if "=" in it:
            tag, p = it.split("=", 1)
            tag, p = tag.strip(), Path(p.strip())
        elif it == "own":
            tag, p = "own", None
        elif it in cat:
            tag, p = it, cat[it]
        elif ARIEL_REMOVED.fullmatch(it):
            die(f"predictor {it!r}: 옛 Ariel epoch 체크포인트 (block_causal_future_only_1e_5_ep_N) 는 2026-09-27 에 지워졌다. "
                f"지금 판: {sorted(ARIEL)} (ep43 ≠ ep45 — 바꿔 부르지 않는다)")
        else:
            die(f"predictor {it!r} 를 모른다 (카탈로그 태그 / own / tag=path). 카탈로그: {sorted(cat)}")
        if tag in seen:
            die(f"predictor 태그 중복: {tag}")
        if p is not None and not p.is_file():
            die(f"predictor 체크포인트 없음: {tag} -> {p}")
        seen.add(tag)
        out.append((tag, None if p is None else str(p)))
    return out, name


def pred_info(tag: str, path: str | None, cfg: dict) -> dict:
    """meta 에 남길 predictor 정보. 경로 predictor 는 mmap 으로 열어 arch/epoch 만 읽는다."""
    if path is None:
        m = cfg["model"]
        src = m.get("predictor_checkpoint") or m["checkpoint"]
        kind = str((m.get("predictor") or {}).get("kind") or "default(or ckpt)")
        if m.get("predictor_checkpoint"):                    # 2026-09-27: kind 는 ckpt arch 가 이긴다 (ar 이면 배열이 달라진다)
            kind = str((_torch_load(m["predictor_checkpoint"]).get("arch") or {}).get("kind") or "").strip() or kind
        return {"tag": tag, "path": None, "source": src, "kind": kind, "space": _space(kind),
                "note": "bundle 이 병합 config 대로 지은 predictor (build_from_config 그대로)"}
    st = _torch_load(path)
    arch = dict(st.get("arch") or {})
    kind = str(arch.get("kind") or "").strip() or "default"
    fs = os.stat(path)
    return {"tag": tag, "path": path, "kind": kind, "space": _space(kind),
            "kind_kwargs": {k: arch[k] for k in PV.KIND_ONLY_KEYS if k in arch},
            "arch": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in arch.items()},
            "epoch": st.get("epoch"), "step": st.get("step"),
            "loss": float(st["loss"]) if isinstance(st.get("loss"), (int, float)) else None,
            "base_checkpoint": st.get("base_checkpoint"), "format": st.get("format"),
            "size": fs.st_size, "mtime": fs.st_mtime}


def _space(kind: str) -> str:
    """l1 이 어느 타깃 공간인가 (2026-09-27): kind=ar → 블록별 LN(target_encoder), 그 밖 → 창 전체 (표준)."""
    return "ar_block" if kind == "ar" else "window"


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:                                        # 옛 (비-zip) 포맷이면 mmap 불가
        return torch.load(path, map_location="cpu", weights_only=False)


def _same_base(a: str | None, b: str) -> bool:
    """base_checkpoint 비교 — 마운트가 달라도 (Ariel 은 /nas2/...) HF snapshot 꼬리 3 칸이 같으면 같은 파일."""
    return a is not None and Path(a).parts[-3:] == Path(b).parts[-3:]


def encoder_dim(cfg: dict) -> int:
    """체크포인트의 target encoder patch_embed 에서 D 를 읽는다 (mmap, 모델 짓지 않음)."""
    st = _torch_load(cfg["model"]["checkpoint"])
    sd = st[cfg["model"]["target_encoder_key"]]
    for k, v in sd.items():
        if k.endswith("patch_embed.proj.weight"):
            return int(v.shape[0])
    die("target encoder 에 patch_embed.proj.weight 가 없다")


# ----------------------------------------------------------------------------- 모델
def load_pred(bundle, cfg: dict, path: str):
    """경로 predictor 를 bundle 에 맞게 짓는다 — build_from_config 의 predictor 분기와 같은 인자."""
    st = _torch_load(path)
    arch = st.get("arch") or {}
    kind = str(arch.get("kind") or "").strip() or "default"
    kk = {k: arch[k] for k in PV.KIND_ONLY_KEYS if k in arch}
    m = cfg["model"]
    pc = m.get("predictor") or {}
    if "encoder_embed_dim" in arch and int(arch["encoder_embed_dim"]) != int(bundle.embed_dim):
        raise RuntimeError(f"{path}: arch.encoder_embed_dim {arch['encoder_embed_dim']} != encoder D {bundle.embed_dim}")
    pred = _build_predictor(
        img_size=bundle.img_size, patch_size=bundle.patch_size, tubelet_size=bundle.tubelet_size,
        num_frames=bundle.num_frames, encoder_embed_dim=bundle.embed_dim,
        predictor_embed_dim=int(pc.get("embed_dim", 384)), predictor_depth=int(pc.get("depth", 12)),
        predictor_num_heads=int(pc.get("num_heads", 12)), num_mask_tokens=int(pc.get("num_mask_tokens", 10)),
        use_rope=bool(m.get("use_rope", True)), uniform_power=bool(m.get("uniform_power", False)),
        kind=kind, kind_kwargs=kk)
    pred.load_state_dict(_clean_backbone_key(st["predictor"]), strict=True)
    del st
    pred = pred.to(device=bundle.device, dtype=bundle.dtype).eval()
    for q in pred.parameters():
        q.requires_grad_(False)
    return pred, kind


class Scorer:
    """encoder 한 번 + predictor K 개. run(clips) → 슬롯별 통계 (numpy)."""

    def __init__(self, cfg: dict, dev: torch.device, preds: list, keep_own: bool = False):
        self.cfg = cfg
        self.bundle = b = build_from_config(cfg["model"], dev)
        self.D, self.S, ts = int(b.embed_dim), int(b.num_spatial_tokens), int(b.tubelet_size)
        ctx, N = int(cfg["surprise"]["context_length"]), int(cfg["data"]["n_frames"])
        self.ctx, self.tgt = ctx, N - ctx
        self.nc, self.nf = ctx // ts, (N - ctx) // ts
        assert self.nf == N_SLOT, self.nf
        self.mask_index = int(cfg["surprise"].get("mask_index", 0))
        ac = str(cfg["model"].get("autocast", "none")).lower()
        self.ac_dtype = {"float16": torch.float16, "fp16": torch.float16, "bfloat16": torch.bfloat16,
                         "bf16": torch.bfloat16}.get(ac)
        self.tags, self.mods, self.kinds = [], [], []
        for tag, path in preds:
            if path is None:
                mod, kind = b.predictor, "own"
            else:
                mod, kind = load_pred(b, cfg, path)
            self.tags.append(tag); self.mods.append(mod); self.kinds.append(kind)
        # 2026-09-27: kind=ar 는 mask token forward 가 막혀 있어 블록별 경로 (ar_scoring) 로 채점한다. 없으면 hb 를 안 만든다
        self.is_ar = [PV.is_ar(m) for m in self.mods]
        self.any_ar = any(self.is_ar)
        if self.any_ar:
            ARS.check_ar_ready(bool(cfg["surprise"].get("target_layer_norm", True)))
        if not keep_own and all(p is not None for _, p in preds):
            b.predictor = None                                 # 안 쓰는 기본 predictor 는 내려놓는다
            torch.cuda.empty_cache()

    def autocast(self):
        return torch.autocast("cuda", dtype=self.ac_dtype) if self.ac_dtype else contextlib.nullcontext()

    def idx(self, B: int):
        return _context_target_indices(ctx_frames=self.ctx, tgt_frames=self.tgt, tubelet_size=self.bundle.tubelet_size,
                                       spatial_tokens=self.S, batch_size=B, device=self.bundle.device)

    @torch.inference_mode()
    def encode(self, clips: torch.Tensor):
        b = self.bundle
        x = clips.to(b.device, b.dtype, non_blocking=True)
        B = x.size(0)
        ci, ti = self.idx(B)
        with self.autocast():
            z = b.context_encoder(x, masks=[ci])
            h = F.layer_norm(b.target_encoder(x), (self.D,))          # 토큰 LN — gather 뒤 LN (하네스) 과 같은 값
            # kind=ar 용 블록별 타깃 (2026-09-27): 블록 b 토큰은 블록 b 의 2 장만 본다, 이미 LN. AR 이 없으면 계산 안 함
            hb = ARS.encode_blocks(b.target_encoder, x, b.tubelet_size) if self.any_ar else None
        return z, h, ci, ti, hb

    @torch.inference_mode()
    def predict(self, mod, z, ci, ti):
        with self.autocast():
            return mod(z, ci, ti, mask_index=self.mask_index)

    @torch.inference_mode()
    def predict_ar(self, mod, hb):
        """kind=ar: 블록별 LN(target) 문맥 nc 블록 → nf 블록 rollout (ar_scoring 그대로, 표준과 같은 autocast)."""
        with self.autocast():
            p, _, _ = ARS.rollout_from_blocks(mod, hb, self.nc, self.S)
        ARS.assert_finite(p, "te_v11_score")
        return p

    @torch.inference_mode()
    def run(self, clips: torch.Tensor, keep_p: bool = False) -> dict:
        D, S, nc, nf = self.D, self.S, self.nc, self.nf
        z, h, ci, ti, hb = self.encode(clips)
        B = z.size(0)
        hh = h.float().reshape(B, nc + nf, S, D)
        hf, hT = hh[:, nc:], hh[:, nc - 1:nc]
        with self.autocast():
            zT = F.layer_norm(z.float(), (D,)).reshape(B, nc, S, D)[:, nc - 1:nc]
        zT = zT.float()
        red = lambda t: t.abs().mean((-1, -2))                     # noqa: E731  (B, nf)
        out = {"copy": red(zT - hf), "hcopy": red(hT - hf),
               "hstep": red(hf - torch.cat([hT, hf[:, :-1]], 1))}
        if hb is not None:                                         # AR 전용 복사 기준선: 블록 공간의 문맥 마지막 블록
            hbb = hb.float().reshape(B, nc + nf, S, D)
            hbf, hbT = hbb[:, nc:], hbb[:, nc - 1:nc]
            out["copy_blk"] = red(hbT - hbf)
        l1, pz, pstep, ps = [], [], [], {}
        for tag, mod, ar in zip(self.tags, self.mods, self.is_ar):
            if ar:                                                 # 타깃 · 복사 = 블록 공간 (hbf, hbT)
                p = self.predict_ar(mod, hb).float().reshape(B, nf, S, D)
                tgt, ref = hbf, hbT
            else:
                p = self.predict(mod, z, ci, ti).float().reshape(B, nf, S, D)
                tgt, ref = hf, zT
            l1.append(red(p - tgt)); pz.append(red(p - ref)); pstep.append(red(p - torch.cat([ref, p[:, :-1]], 1)))
            if keep_p:
                ps[tag] = p
            del p
        out.update(l1=torch.stack(l1, 1), pz=torch.stack(pz, 1), pstep=torch.stack(pstep, 1))
        res = {k: v.cpu().numpy().astype(np.float32) for k, v in out.items()}
        if keep_p:
            res["_p"], res["_z"], res["_ci"], res["_ti"], res["_hb"] = ps, z, ci, ti, hb
        return res


class ClipDS(torch.utils.data.Dataset):
    def __init__(self, ds, rows, ds_index):
        self.ds, self.rows, self.ix = ds, rows, ds_index

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        r = self.rows[k]
        return r, self.ds.clip(self.ix[r])


def collate(b):
    return [x[0] for x in b], torch.stack([x[1] for x in b])


def select_rows(ds, limit: int) -> list:
    """데이터셋 인덱스 (index_test.csv 순서). limit 이면 block 을 고르게 골라 약 limit clip (쌍이 온전하다)."""
    if not limit:
        return list(range(len(ds.records)))
    by = defaultdict(list)
    for i, r in enumerate(ds.records):
        by[r.block_id].append(i)
    blocks = list(by)                                          # 처음 나온 순서
    nb = max(1, min(len(blocks), round(limit / 4)))
    pick = [blocks[int(round(x))] for x in np.linspace(0, len(blocks) - 1, nb)]
    keep = set(i for b in pick for i in by[b])
    return [i for i in range(len(ds.records)) if i in keep]


# ----------------------------------------------------------------------------- 본 실행
def _worker(rank: int, world: int, a, cfg: dict, preds: list, todo: list, ds_index: list, out: str):
    nvis = torch.cuda.device_count()
    dev = torch.device(f"cuda:{rank % nvis}")
    torch.cuda.set_device(dev)
    torch.set_num_threads(a.threads)
    t0 = time.time()
    ds = WMADataset(cfg)
    sc = Scorer(cfg, dev, preds)
    t_load = time.time() - t0
    mine = todo[rank::world]
    # ⚠️ mp.spawn 자식 안에서는 multiprocessing 기본값이 spawn 이라 DataLoader worker 마다 이 모듈을 다시 import 한다
    #    (실측: 첫 batch 96 s). worker 는 PNG 디코드 (CPU) 만 하므로 fork 로 띄운다.
    loader = torch.utils.data.DataLoader(ClipDS(ds, mine, ds_index), batch_size=a.batch, num_workers=a.workers,
                                         collate_fn=collate, pin_memory=True,
                                         multiprocessing_context="fork" if a.workers > 0 else None)
    outp = Path(out)
    names = arr_names(sc.any_ar)
    miss = [k for k in names if not (outp / f"{k}.npy").exists()]
    if miss:                                                   # meta 의 kind 판정 (ckpt arch) 과 실제 모듈이 다르면 여기서 죽는다
        raise RuntimeError(f"{outp}: {miss}.npy 가 없다 — predictor kind (is_ar={sc.is_ar}) 가 meta 와 다르다")
    mm = {k: np.lib.format.open_memmap(outp / f"{k}.npy", mode="r+") for k in names}
    done = np.lib.format.open_memmap(outp / "done.npy", mode="r+")
    _log(outp, {"event": "start", "rank": rank, "world": world, "n": len(mine), "load_s": round(t_load, 1),
                "device": str(dev), "gpu": torch.cuda.get_device_name(dev)})
    t1, n_done, t_first = time.time(), 0, None
    for it, (rows, clips) in enumerate(loader):
        r = sc.run(clips)
        rows = np.asarray(rows, dtype=np.int64)
        for k in names:
            mm[k][rows] = r[k]
        for k in names:
            mm[k].flush()
        done[rows] = 1
        done.flush()
        n_done += len(rows)
        if it == 0:
            t_first = time.time()
        if rank == 0 and (it % a.log_every == 0 or n_done == len(mine)):
            el = time.time() - t1
            print(f"[rank0] {n_done}/{len(mine)}  {el / 60:.1f} 분  남은 {el / max(n_done, 1) * (len(mine) - n_done) / 60:.1f} 분",
                  flush=True)
    el = time.time() - t1
    steady = (time.time() - t_first) / max(n_done - a.batch, 1) if t_first and n_done > a.batch else None
    _log(outp, {"event": "end", "rank": rank, "n": n_done, "loop_s": round(el, 1),
                "s_per_clip": round(el / max(n_done, 1), 4), "s_per_clip_steady": None if steady is None else round(steady, 4),
                "peak_mem_gb": round(torch.cuda.max_memory_allocated(dev) / 2**30, 2)})
    print(f"[rank{rank}] 끝 {n_done} clip, {el:.0f}s ({el / max(n_done, 1):.3f} s/clip), "
          f"peak {torch.cuda.max_memory_allocated(dev) / 2**30:.2f} GB", flush=True)


def arr_names(has_ar: bool) -> tuple:
    """쓰는 배열 이름. AR predictor 가 없으면 2026-09-27 이전과 같다 (재개 · 옛 폴더 호환)."""
    return ARR_K + ARR_1 + (ARR_AR if has_ar else ())


def _log(outp: Path, rec: dict):
    rec = dict(rec, time=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(outp / "progress.jsonl", "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def matched_acc(meta: dict, s: np.ndarray) -> tuple[float, int, int]:
    """(block_id, pair_id) 마다 가능 1 · 불가능 1. hit = 1[s_pos < s_imp] + 0.5 동점 — eval.py::score_blocks 와 같은 규칙."""
    grp = defaultdict(dict)
    for i, (b, p, pl) in enumerate(zip(meta["block_id"], meta["pair_id"], meta["plausible"])):
        grp[(b, p)]["pos" if str(pl) == "1" else "imp"] = i
    hits, ties = [], 0
    for g in grp.values():
        if "pos" not in g or "imp" not in g:
            continue
        a, b = s[g["pos"]], s[g["imp"]]
        hits.append(1.0 if a < b else (0.5 if a == b else 0.0))
        ties += int(a == b)
    return float(np.mean(hits)) if hits else float("nan"), len(hits), ties


def write_summary(outp: Path, meta: dict):
    l1 = np.load(outp / "l1.npy")
    cp = np.load(outp / "copy.npy")
    res = {"n_clip": int(l1.shape[0]), "definition": "S = l1.mean(slot) (= 표준 surprise); acc = matched pair, 동점 0.5",
           "preds": {}}
    space = meta.get("pred_space") or {}                      # 2026-09-27 이전 meta 에는 없다 (= 전부 window)
    for k, tag in enumerate(meta["pred_tags"]):
        acc, n, ties = matched_acc(meta, l1[:, k].mean(1))
        res["preds"][tag] = {"acc": acc, "n_pair": n, "n_ties": ties, "mean_S": float(l1[:, k].mean()),
                             "mean_l1_by_slot": [float(v) for v in l1[:, k].mean(0)]}
        if space.get(tag) == "ar_block":
            res["preds"][tag].update(space="ar_block", copy_ref="copy_blk",
                                     note="kind=ar: l1 은 블록별 LN(target) 공간 — copy/hcopy 와 비교 금지, 다른 kind 와는 acc 만")
    acc, n, ties = matched_acc(meta, cp.mean(1))
    res["copy"] = {"acc": acc, "n_pair": n, "n_ties": ties, "mean_S": float(cp.mean()),
                   "mean_by_slot": [float(v) for v in cp.mean(0)]}
    if (outp / "copy_blk.npy").exists():                      # AR 전용 복사 기준선
        cb = np.load(outp / "copy_blk.npy")
        acc, n, ties = matched_acc(meta, cb.mean(1))
        res["copy_blk"] = {"acc": acc, "n_pair": n, "n_ties": ties, "mean_S": float(cb.mean()),
                           "mean_by_slot": [float(v) for v in cb.mean(0)],
                           "definition": "mean|hb[7] - hb[8+j]|, hb = LN(target_encoder(블록 2 장 따로)) — kind=ar 행의 복사 대조"}
    f = outp / "summary.json"
    if f.exists():
        f = outp / f"summary_{time.strftime('%Y%m%d-%H%M%S')}.json"
    json.dump(res, open(f, "w"), indent=1, ensure_ascii=False)
    print(f"summary -> {f}")
    for tag, v in res["preds"].items():
        print(f"  {tag:>12s}  acc {100 * v['acc']:.2f}  (n_pair {v['n_pair']}, ties {v['n_ties']})  mean S {v['mean_S']:.5f}"
              + ("  [ar_block 공간]" if v.get("space") == "ar_block" else ""))
    print(f"  {'copy':>12s}  acc {100 * res['copy']['acc']:.2f}  mean S {res['copy']['mean_S']:.5f}")
    if "copy_blk" in res:
        print(f"  {'copy_blk':>12s}  acc {100 * res['copy_blk']['acc']:.2f}  mean S {res['copy_blk']['mean_S']:.5f}  [ar_block 공간]")


def main_run(a):
    cfg, text = resolve_cfg(a.model)
    preds, pname = parse_preds(a)
    infos = [pred_info(t, p, cfg) for t, p in preds]
    for inf in infos:
        if inf["path"] and not _same_base(inf.get("base_checkpoint"), cfg["model"]["checkpoint"]) and not a.allow_foreign_base:
            die(f"{inf['tag']}: base_checkpoint {inf.get('base_checkpoint')} 가 --model {a.model} 의 checkpoint "
                f"{cfg['model']['checkpoint']} 와 다르다 (그룹 B 에 릴리즈 predictor 를 걸 수 없다). --allow-foreign-base 로만 푼다")
    has_ar = any(inf["space"] == "ar_block" for inf in infos)          # 2026-09-27: kind=ar 가 있으면 copy_blk 배열을 더 만든다
    ds = WMADataset(cfg)
    ds_index = select_rows(ds, a.limit)
    n, K = len(ds_index), len(preds)
    out = Path(a.out) if a.out else OUT_ROOT / (f"v11score_{a.model}_{pname}" + (f"_limit{a.limit}" if a.limit else ""))
    out.mkdir(parents=True, exist_ok=True)
    meta = {"script": str(Path(__file__).relative_to(REPO)), "model": a.model, "preset": pname,
            "protocol": PROTOCOL, "dataset": DATASET, "limit": a.limit, "n": n, "ds_index": ds_index,
            "index_csv": str(Path(cfg["data"]["root"]) / cfg["data"]["index_csv"]),
            "pred_tags": [t for t, _ in preds], "preds": infos,
            "D": encoder_dim(cfg), "S_TOK": (int(cfg["model"]["img_size"]) // int(cfg["model"]["patch_size"])) ** 2,
            "n_ctx_tub": 8, "n_fut_tub": N_SLOT,
            "arrays": {"l1": "(n,K,8) mean|p_k[j]-h[8+j]|", "pz": "(n,K,8) mean|p_k[j]-LN(z)[7]|",
                       "pstep": "(n,K,8) mean|p_k[j]-p_k[j-1]|, p_k[-1]=LN(z)[7]",
                       "copy": "(n,8) mean|LN(z)[7]-h[8+j]|", "hcopy": "(n,8) mean|h[7]-h[8+j]|",
                       "hstep": "(n,8) mean|h[8+j]-h[8+j-1]|", "done": "(n,) uint8"},
            "h": "h = LN(target_encoder(32 frames)) affine-free; z = context_encoder(16 frames); fp32 weights + autocast fp16",
            "pred_space": {inf["tag"]: inf["space"] for inf in infos},
            "resolved_config": cfg}
    if has_ar:                                                          # AR 행의 배열 뜻 (2026-09-27)
        meta["arrays"].update(
            copy_blk="(n,8) mean|hb[7]-hb[8+j]|, hb = LN(target_encoder(블록 2 장 따로)) — kind=ar 행의 복사 대조",
            ar_rows="pred_space == ar_block 인 k: l1 = mean|p_ar[j]-hb[8+j]|, pz = mean|p_ar[j]-hb[7]|, "
                    "pstep 의 p[-1] = hb[7]. copy/hcopy/hstep 과 같은 공간이 아니다 — 쌍 정확도만 다른 kind 와 비교")
    for c in META_COLS:
        meta[c] = [ds.records[i].raw.get(c) for i in ds_index]
    meta["video_id"] = [ds.records[i].video_id for i in ds_index]
    mp_ = out / "meta.json"
    KEY = ("model", "dataset", "protocol", "n", "ds_index", "video_id", "pred_tags")
    if mp_.exists():
        old = json.load(open(mp_))
        diff = [k for k in KEY if old.get(k) != meta.get(k)]
        diff += [f"preds[{i}].path" for i, (x, y) in enumerate(zip(old["preds"], infos)) if x.get("path") != y.get("path")]
        oc, nc = old.get("resolved_config", {}), cfg
        diff += [f"resolved_config.{s}" for s in ("model", "data", "surprise") if oc.get(s) != nc.get(s)]
        diff += [f"preds[{i}].size/mtime" for i, (x, y) in enumerate(zip(old["preds"], infos))
                 if (x.get("size"), x.get("mtime")) != (y.get("size"), y.get("mtime"))]
        if diff:
            die(f"{out} 에 다른 실행의 산출물이 있다 (다른 항목: {diff}). 덮어쓰지 않는다 — --out 을 바꿀 것")
        for k in arr_names(has_ar) + ("done",):
            if not (out / f"{k}.npy").exists():
                die(f"{out}/{k}.npy 가 없다 — 불완전한 폴더. 손으로 확인할 것")
        print(f"[resume] {out}: meta 일치")
    else:
        if any((out / f"{k}.npy").exists() for k in arr_names(True) + ("done",)):
            die(f"{out} 에 meta.json 없이 배열이 있다. 덮어쓰지 않는다")
        for k in ARR_K:
            m = np.lib.format.open_memmap(out / f"{k}.npy", mode="w+", dtype=np.float32, shape=(n, K, N_SLOT)); m[:] = np.nan; m.flush(); del m
        for k in arr_names(has_ar)[len(ARR_K):]:
            m = np.lib.format.open_memmap(out / f"{k}.npy", mode="w+", dtype=np.float32, shape=(n, N_SLOT)); m[:] = np.nan; m.flush(); del m
        m = np.lib.format.open_memmap(out / "done.npy", mode="w+", dtype=np.uint8, shape=(n,)); m[:] = 0; m.flush(); del m
        json.dump(meta, open(mp_, "w"), ensure_ascii=False)
        open(out / "_resolved.yaml", "w", encoding="utf-8").write(text)
    done = np.load(out / "done.npy")
    todo = [int(r) for r in np.flatnonzero(done == 0)]
    print(f"out={out}\n  model={a.model}  preds({K})={[t for t, _ in preds]}\n  n={n}  남은 {len(todo)}  gpus={a.gpus}  "
          f"batch={a.batch}  kinds={[i['kind'] for i in infos]}", flush=True)
    if todo:
        nvis = torch.cuda.device_count()
        if a.gpus > nvis and not a.allow_shared_device:
            die(f"--gpus {a.gpus} > 보이는 GPU {nvis}. CUDA_VISIBLE_DEVICES 를 확인할 것")
        world = min(a.gpus, len(todo))
        mp.spawn(_worker, args=(world, a, cfg, preds, todo, ds_index, str(out)), nprocs=world, join=True)
    done = np.load(out / "done.npy")
    if not done.all():
        die(f"끝났는데 done 이 아닌 행 {int((done == 0).sum())} 개 — 같은 명령으로 재개할 것")
    for k in arr_names(has_ar):
        v = np.load(out / f"{k}.npy", mmap_mode="r")
        if np.isnan(v).any():
            die(f"{k}.npy 에 NaN {int(np.isnan(v).sum())} 개")
    print("ALL DONE", flush=True)
    if not (out / "summary.json").exists():
        write_summary(out, json.load(open(mp_)))


# ----------------------------------------------------------------------------- 검증
def ref_files(tag: str, model: str) -> list:
    if tag == "release" or (tag == "own" and model == "vith"):
        fs = [VX / "surprise_c16t32__v11_vith/per_block.json", TA / "surprise_c16t32__v11_earlymid_vith/per_block.json"]
    elif re.fullmatch(r"(v11|pv1|ip1|ip2)_e\d+", tag):
        fam, n = tag.split("_e")
        fs = [ZR / RUNS[fam] / f"eval/surprise_c16t32__v11_split_test_e{n}/per_block.json"]
    elif tag in ARIEL:                                         # 2026-09-27 세 팔: run.sh surprise_c16t32 v11_split_test <레지스트리>
        fs = [VX / f"surprise_c16t32__v11_split_test_{ARIEL[tag][1]}/per_block.json"]
    elif re.fullmatch(r"ariel_ep\d+", tag):                   # 옛 태그 (파일은 2026-09-27 에 지워짐) — 옛 산출물 검증용으로 남긴다
        n = tag[len("ariel_ep"):]
        st = VX / f"surprise_c16t32__v11_split_test_vith_ariel_bc_ep{n}/per_block.json"
        fs = [st] if st.exists() else [VX / f"surprise_c16t32__v11_vith_ariel_bc_ep{n}/per_block.json"]
    elif tag == "own":
        fs = [VX / f"surprise_c16t32__v11_split_test_{model}/per_block.json"]
    else:
        fs = []
    return [f for f in fs if f.exists()]


def load_refs(files: list) -> dict:
    d = {}
    for f in files:
        pv = json.load(open(f))["per_video_surprise"]
        for k, v in pv.items():
            d[k] = float(v["all"] if isinstance(v, dict) else v)
    return d


def main_validate(a):
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    torch.set_num_threads(a.threads)
    cfg, _ = resolve_cfg(a.model)
    if a.preds:
        preds, _ = parse_preds(a)
    else:
        a.preds = ",".join(VAL_DEFAULT.get(a.model, ["own"]))
        preds, _ = parse_preds(a)
    has_own = any(p is None for _, p in preds)
    if not has_own:
        preds = preds + [("own", None)]                       # (a) 의 기준: build_from_config 기본 predictor
    ds = WMADataset(cfg)
    refs = {t: ref_files(t, a.model) for t, _ in preds}
    refd = {t: load_refs(fs) for t, fs in refs.items() if fs}
    # 모든 참조에 다 있는 block 만 (block 4 clip 온전히), 고르게 n_val/4 개
    by = defaultdict(list)
    for i, r in enumerate(ds.records):
        by[r.block_id].append(i)
    cand = [b for b, ix in by.items() if all(ds.records[i].video_id in d for d in refd.values() for i in ix)]
    nb = max(1, min(len(cand), a.n_val // 4))
    pick = [cand[int(round(x))] for x in np.linspace(0, len(cand) - 1, nb)]
    rows = [i for b in pick for i in by[b]]
    print(f"[validate] model={a.model} preds={[t for t, _ in preds]}  후보 block {len(cand)} → {len(rows)} clip", flush=True)
    t0 = time.time()
    sc = Scorer(cfg, dev, preds, keep_own=True)
    t_load = time.time() - t0
    rep = {"model": a.model, "n_clip": len(rows), "video_ids": [ds.records[i].video_id for i in rows],
           "ref_files": {t: [str(f.relative_to(REPO)) for f in fs] for t, fs in refs.items()},
           "kinds": dict(zip(sc.tags, sc.kinds)), "D": sc.D, "S_TOK": sc.S, "load_s": round(t_load, 1), "checks": {}}

    # (a) 가중치: release 추출본 == model.pth predictor
    if "release" in sc.tags and sc.bundle.predictor is not None:
        own_sd = sc.bundle.predictor.state_dict()
        rel_sd = sc.mods[sc.tags.index("release")].state_dict()
        rep["checks"]["a_release_weights_max_abs"] = max(float((own_sd[k].float() - rel_sd[k].float()).abs().max()) for k in own_sd)
    # (a') build_from_config({predictor_checkpoint}) 로 지은 predictor (CPU 에서 짓고 GPU 로) — 경로 predictor 마다
    rebuilt = {}
    for tag, path in preds:
        if path is None or tag == "release" or a.no_rebuild:
            continue
        t1 = time.time()
        mcfg = dict(cfg["model"], predictor_checkpoint=path)
        bcpu = build_from_config(mcfg, torch.device("cpu"))
        rebuilt[tag] = bcpu.predictor.to(dev).eval()
        del bcpu
        print(f"[validate] build_from_config({tag}) {time.time() - t1:.0f}s kind(ckpt)={sc.kinds[sc.tags.index(tag)]} "
              f"class={type(rebuilt[tag]).__name__}", flush=True)

    loader = torch.utils.data.DataLoader(ClipDS(ds, list(range(len(rows))), rows), batch_size=a.batch,
                                         num_workers=a.workers, collate_fn=collate,
                                         multiprocessing_context="fork" if a.workers > 0 else None)
    acc = {k: [] for k in arr_names(sc.any_ar)}
    pdiff = defaultdict(float)
    torch.cuda.reset_peak_memory_stats(dev)
    ts, n_seen = [], 0
    for it, (rr, clips) in enumerate(loader):
        torch.cuda.synchronize(); tb = time.time()
        r = sc.run(clips, keep_p=(it == 0))
        torch.cuda.synchronize(); ts.append((time.time() - tb, len(rr)))
        if it == 0:                                            # p 비교는 첫 batch 에서
            z, ci, ti, hb = r["_z"], r["_ci"], r["_ti"], r["_hb"]
            if "release" in r["_p"] and "own" in r["_p"]:
                pdiff["release_vs_build_from_config_default"] = float((r["_p"]["release"] - r["_p"]["own"]).abs().max())
            for tag, mod in rebuilt.items():
                if PV.is_ar(mod):                              # 2026-09-27: kind=ar 는 블록별 rollout 으로 대조
                    pr = sc.predict_ar(mod, hb).float().reshape_as(r["_p"][tag])
                else:
                    pr = sc.predict(mod, z, ci, ti).float().reshape_as(r["_p"][tag])
                pdiff[f"{tag}_vs_build_from_config_ckpt"] = float((r["_p"][tag] - pr).abs().max())
            del r["_p"], r["_z"], r["_ci"], r["_ti"], r["_hb"]
        for k in arr_names(sc.any_ar):
            acc[k].append(r[k])
        n_seen += len(rr)
    arr = {k: np.concatenate(v) for k, v in acc.items()}
    rep["checks"]["a_max_abs_dp"] = dict(pdiff)
    rep["peak_mem_gb"] = round(torch.cuda.max_memory_allocated(dev) / 2**30, 2)
    steady = ts[1:] if len(ts) > 1 else ts
    rep["s_per_clip_steady"] = round(sum(t for t, _ in steady) / sum(n for _, n in steady), 4)
    rep["batch"], rep["n_preds_timed"] = a.batch, len(sc.tags)
    vids = rep["video_ids"]
    rep["slot_mean"] = {}
    for k, tag in enumerate(sc.tags):
        s = arr["l1"][:, k].mean(1)
        row = {"mean": float(s.mean())}
        if tag in refd:
            ref = np.array([refd[tag][v] for v in vids])
            d = np.abs(s - ref)
            row.update(ref_mean=float(ref.mean()), max_abs_diff=float(d.max()), mean_abs_diff=float(d.mean()),
                       max_rel_diff=float((d / np.abs(ref)).max()),
                       pass_1e3=bool(d.max() < 1e-3))
        rep["slot_mean"][tag] = row
    rep["copy_slot_mean"] = {"mean": float(arr["copy"].mean()), "min": float(arr["copy"].mean(1).min()),
                             "max": float(arr["copy"].mean(1).max())}
    rep["hcopy_slot_mean"] = float(arr["hcopy"].mean())
    rep["hstep_by_slot"] = [float(v) for v in arr["hstep"].mean(0)]
    rep["l1_by_slot"] = {tag: [float(v) for v in arr["l1"][:, k].mean(0)] for k, tag in enumerate(sc.tags)}
    rep["copy_by_slot"] = [float(v) for v in arr["copy"].mean(0)]
    if "copy_blk" in arr:                                      # 2026-09-27 kind=ar 전용
        rep["copy_blk_by_slot"] = [float(v) for v in arr["copy_blk"].mean(0)]
        rep["pred_space"] = {t: ("ar_block" if ar else "window") for t, ar in zip(sc.tags, sc.is_ar)}
    # 자기 일관성: pstep[:, :, 0] == pz[:, :, 0], hstep[:, 0] == hcopy[:, 0]
    rep["checks"]["pstep0_eq_pz0"] = float(np.abs(arr["pstep"][:, :, 0] - arr["pz"][:, :, 0]).max())
    rep["checks"]["hstep0_eq_hcopy0"] = float(np.abs(arr["hstep"][:, 0] - arr["hcopy"][:, 0]).max())
    ok_a = all(v == 0.0 for v in pdiff.values()) and rep["checks"].get("a_release_weights_max_abs", 0.0) == 0.0
    ok_b = all(v.get("pass_1e3", True) for v in rep["slot_mean"].values())
    rep["pass"] = {"a_bitwise_p": ok_a, "bcd_slot_mean_1e-3": ok_b}
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    f = DOC_DIR / f"validate_{a.model}.json"
    if f.exists():
        f = DOC_DIR / f"validate_{a.model}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    json.dump(rep, open(f, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: rep[k] for k in ("checks", "slot_mean", "copy_slot_mean", "pass", "peak_mem_gb",
                                          "s_per_clip_steady", "load_s")}, indent=1, ensure_ascii=False))
    print(f"-> {f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="vith", help="configs/protocols/models.md 의 이름 (기본 vith)")
    ap.add_argument("--preset", choices=sorted(PRESETS), help="final | curves | own (기본: vith→final, 그 밖→own)")
    ap.add_argument("--preds", help="콤마 목록: 카탈로그 태그 | own | tag=/path.pt  (--preset 대신)")
    ap.add_argument("--name", help="출력 폴더 이름의 preset 자리 (--preds 일 때 기본 custom)")
    ap.add_argument("--out", help="출력 폴더 (기본 training_effects/v11score_<model>_<preset>)")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8, help="rank 당 clip 수")
    ap.add_argument("--workers", type=int, default=6, help="rank 당 PNG 디코드 프로세스")
    ap.add_argument("--threads", type=int, default=4, help="rank 당 torch CPU 스레드")
    ap.add_argument("--limit", type=int, default=0, help="block 단위로 약 N clip (스모크)")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--allow-foreign-base", action="store_true")
    ap.add_argument("--allow-shared-device", action="store_true", help="--gpus 가 보이는 GPU 보다 많으면 rank %% nvis 로 겹쳐 쓴다 (시험용)")
    ap.add_argument("--validate", action="store_true", help="검증 (GPU 1 장): (a) p 비트 대조 (b-d) 하네스 per_video_surprise 대조")
    ap.add_argument("--n-val", type=int, default=48)
    ap.add_argument("--no-rebuild", action="store_true", help="검증에서 build_from_config({predictor_checkpoint}) 재구성을 건너뜀")
    a = ap.parse_args()
    if a.validate:
        main_validate(a)
    else:
        main_run(a)


if __name__ == "__main__":
    main()
