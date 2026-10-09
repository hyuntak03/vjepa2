#!/usr/bin/env python3
"""te_intphys1_score — IntPhys 1 dev Garrido 격자에서 **predictor 여러 개를 한 번에** 채점한다:
창 × 미래 튜블릿 단위 surprise (표준 표적 · 인과 표적) + 복사 기준선.  TrainingEffects 그룹 A.

auto_research H6e (`auto_research/scripts/h6e_intphys1_causal_target.py`) 를 predictor 목록으로 일반화했다.
그룹 A predictor 는 전부 **frozen 릴리즈 ViT-H encoder** (registry `vith`, online `encoder` → 문맥,
EMA `target_encoder` → 표적) 를 공유하므로 encoder 는 clip 마다 **한 번만** 돌리고 predictor 만 바꿔 끼운다.

## 정의 (기호는 PROTOCOLS.md §1 과 같다)

창 격자 (Garrido A.8, `configs/protocols/intphys1_sliding.yaml` 그대로, 창 생성은 채점기의 `_intphys1_windows`):
  프레임 skip s 로 솎은 뒤 (frame_budget official = (100−1)//s 장) 시작점 t 를 stride 2 로 민 W = C + M 장 창.
  combo ∈ {skip2_w16 (시작 17), skip2_w32 (시작 9), skip5_w16 (시작 2)} — skip5_w32 · skip10_* 는 프레임 부족.
  C ∈ {2,4,6,8,10} × W/16.  튜블릿 2 장 → 창 튜블릿 U = W/2, 문맥 튜블릿 Tc = C/2, 미래 j = 0 … U−Tc−1 (u = Tc + j).
  → 영상당 창 140 개, (창, j) 행 R = 925 개 (skip2_w16 425 · skip2_w32 450 · skip5_w16 50).

  z      = context_encoder(창, mask = 앞 C 장 토큰)                 (문맥 토큰만; 미래는 transformer 전에 버린다)
  p^k    = predictor_k(z, ctx_idx, tgt_idx, mask_index=0)           미래 U−Tc 튜블릿
  h      = LN(target_encoder(창 W 장 전부))                        표준 표적 (affine 없는 토큰 LN). 창 안 미래까지 양방향으로 본다
  h^c_u  = LN(target_encoder(창 앞 2(u+1) 장))[튜블릿 u]             인과 표적 (H6e 정의; 튜블릿 u 까지만 본다)
  copy   = LN(z)[튜블릿 Tc−1]                                      복사 기준선 — 문맥 마지막 튜블릿을 모든 미래에 (예측 없음)
  L1(x, y)_j = mean_{토큰 256, 채널 1280} |x_j − y_j|              (float32)

  열 (행 = (video, combo, start, C, j)):
    p_std[k]    = L1(p^k_j, h_u)      p_causal[k] = L1(p^k_j, h^c_u)          ← predictor 마다
    copy_std    = L1(copy,  h_u)      copy_causal = L1(copy,  h^c_u)          ← predictor 와 무관, 한 번만
  창 surprise S(v, combo, t, C) = mean_j L1_j.  표준 표적의 S 는 공식 하네스 per_window.json 의 값과 같은 양이다.

  kind=ar (자기회귀, Ariel 팔 C — 2026-09-27 추가). mask token forward 가 막혀 있어 `analysis/predictors/ar_scoring.py`
  (표준 하네스 eval.py 의 AR 분기와 같은 함수) 로 돈다:
    hb_u   = LN(target_encoder(창의 블록 u 의 2 장))  블록마다 따로 — 창 시작점마다 한 번 (AR 이 목록에 있을 때만)
    p_ar   = rollout(hb[블록 0..Tc−1], U−Tc 블록)    C 마다 (문맥 Tc = C/2 블록)
    p_std[k]    = L1(p_ar_j, hb_u)       ← **같은 열에 넣지만 표적이 블록별 공간**이다 (하네스 per_window 의 AR 값과 같은 양)
    p_causal[k] = NaN                   ← AR 에는 인과 표적 h^c 열을 정의하지 않는다 (공간이 다르다). --score 는 AR 의 causal 칸을 건너뛴다
    copy_blk    = L1(hb_{Tc−1}, hb_u)    ← AR 전용 복사 기준선 (행마다, predictor 와 무관). AR 이 목록에 있을 때만 생긴다
  ⚠️ copy_std / copy_causal (LN(z) 복사) 는 AR 의 p_std 와 **같은 공간이 아니다** — AR 과 비교하지 말 것 (AR 은 copy_blk 와).
     다른 kind 와는 **쌍 정확도만** 비교한다. manifest 의 `space` (`window` | `ar_block`) · `ar_tags` 가 어느 쪽인지 적는다.
  ⚠️ 학습 분포는 문맥 {2,4,8,16} 블록 · rollout ≤ 8 블록이다. 격자의 Tc = 1,3,5,6,10 과 skip2_w32 의 K = 14 는 분포 밖이라
     min-over-C 가 큰 C 쪽으로 기운다 — C 별 (per_C) 을 같이 볼 것.
  AR 이 목록에 없으면 shard · 배열 · 수치가 전부 이전과 같다.

Garrido 채점 (`--score`, CPU; 식은 `garrido_rescore.py` 의 함수를 **그대로 불러** 쓴다):
  G(v) = mean_t min_C S(v, t, C)           (Filtered → AvgSurprise; IntPhys 규칙, PROTOCOLS.md §3)
  쌍 (block_id, pair_id) 정답 = 1[G(pos) < G(imp)]   (공식 strict <, 동점은 오답 — n_ties 로 보고)
  칸 점수 = property (O1/O2/O3) macro.  최고 칸 = macro 최대 combo (A.8 descriptive).
  같은 규칙으로 C 각각 (per_C) · 노트북 규칙 (alt_notebook) · 운동 × 가림 4 칸 (auto_research/_stage 의 라벨) 도 낸다.
  방법 = predictor 태그들 + `copy`, 표적 = std / causal.

## 수치 경로 (결정과 근거)

- fp32 가중치 + autocast fp16, mask_index 0, target 에만 LN — 공식 Garrido 재현 경로 (CLAUDE.md §1-7).
- **모델은 window_size 32 로 한 번 짓는다.** 공식 하네스는 창마다 (w16/w32) 짓지만 RoPE 위치는 토큰 번호로만
  계산되고 window_size 는 release predictor 의 mask-token 복제 수만 바꾼다 → 창 크기대로 지은 것과 비트 동일
  (`auto_research/scripts/check_window_build.py`). predictor 채점 기준값 (ip1_e40 등 z_training eval) 도 window 32
  하나로 두 창을 돌렸다. prefix predictor 는 grid_depth 를 attention 에 쓰지 않는다 (ACRoPEAttention).
- 배치 = **영상 하나의 시작점 전부** (skip2_w16 17 · skip2_w32 9 · skip5_w16 2; `--max_starts` 로 나눌 수 있다).
  공식 하네스는 영상 6 개 × 시작점을 48 개씩 묶어서 (rank 배정이 run 마다 다르다) fp16 GEMM 알고리즘 선택이
  달라진다 → 창 값이 중앙값 ~1e-4 로 흔들린다 (공식끼리도 `video_batch>1` 이면 ~5e-4, yaml 주석). 비트 재현은 목표가 아니다.
- skip2_w32 의 인과 표적 u < 8 은 같은 시작점 skip2_w16 의 것과 **입력 프레임이 같아서** 재사용한다 (계산 ~11 % 절약).
- predictor 로드: 체크포인트 `arch.kind` (없으면 default) + `analysis.predictors.KIND_ONLY_KEYS` 로
  `analysis.intphys2.model._build_predictor` 를 부르고 strict load — `build_from_config(predictor_checkpoint=…)` 과 같은 경로
  (`--selftest` 가 확인). Ariel block-causal 은 kind='prefix' (prefix_impl 기본 split) 로 지어진다.
- 메인 프로세스가 predictor state_dict 만 `/dev/shm/te_ip1score_<pid>_<rand>/` 에 풀어 두고 rank 들이 거기서 읽는다
  (post-FT 체크포인트는 opt 상태까지 256 MB 라 rank × predictor 수만큼 NFS 를 읽지 않으려고). 끝나면 지운다.

## CLI

  # 추출 (GPU). 모델: preset final (6) / curves (18) 또는 --preds tag=path,tag,...  (tag 만 주면 경로 규칙으로 찾는다)
  #   Ariel 태그 (2026-09-27): ariel_prefix_ep45 (block_causal_future_only_1e_5, prefix) · ariel_full_ep40 (full_attn_future_pred_1e_5,
  #   oneshot) · ariel_ar_ep18 (ar_future_1e_5, ar — 위 AR 절). 옛 ariel_ep{5,19,30,41,43} 은 파일이 지워져 경로 규칙에서 뺐다
  #   (ep43 ≠ ep45, 바꿔 부르지 않는다). preset: final 의 ariel_ep43 → ariel_prefix_ep45, curves 의 ariel_ep{5,19,30,43} 은 대체 없이 뺐다.
  #   옛 폴더 (ip1score_final · ip1score_curves) 의 --score/--validate 는 manifest 로 읽어 그대로 된다. 같은 폴더에 새 preset 으로
  #   추출하면 manifest 불일치로 죽는다 → --out_tag 를 새로 줄 것.
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python z_research/scripts/analysis/te_intphys1_score.py --preset final --gpus 8
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python z_research/scripts/analysis/te_intphys1_score.py --preset curves --gpus 8
  # 채점 · 검증 (CPU)
  python z_research/scripts/analysis/te_intphys1_score.py --preset final --score
  python z_research/scripts/analysis/te_intphys1_score.py --preset final --validate
  # 로더 동치 검사 (GPU 1 장)
  CUDA_VISIBLE_DEVICES=1 python z_research/scripts/analysis/te_intphys1_score.py --selftest
  옵션: --subset valid6 (O1/O2/O3 앞 2 block = 24 영상) · --limit N · --out_tag · --max_starts · --threads

  **이어 돌리기**: 영상마다 shard 를 원자적으로 쓴다 (tmp → rename). 죽으면 같은 명령을 다시 치면 남은 영상만 돈다
  (GPU 수를 바꿔도 된다). predictor 목록이나 창 배치가 manifest 와 다르면 죽는다 → 다른 --out_tag 를 쓸 것.

## 출력

  /data2/local_datasets/world/world_analysis/cache/training_effects/<out_tag>/     (out_tag 기본 ip1score_<preset>)
    manifest.json                 predictor (tag, 원경로, 크기, mtime, kind, 지문), 창 배치 요약, config
    shards/v{i:04d}.npz           영상 i: p_std/p_causal (K, 925) · copy_std/copy_causal (925,) · sec
    arrays_v{V}.npz               병합: p_std/p_causal (K, V, 925) f32 · copy_* (V, 925) · rows (925, 4)=(combo_i, start, C, j)
                                  · combos · video_ids · tags   (--score / --validate 가 없으면 만든다)
  z_research/TrainingEffects/ip1score/<out_tag>_scores.json       --score (있으면 시각 접미사로 새 이름)
  z_research/TrainingEffects/ip1score/<out_tag>_validation.json   --validate
  z_research/TrainingEffects/ip1score/loader_selftest.json         --selftest

## 검증 (2026-09-25, vll5 RTX 4090 GPU 1 — 다른 eval 과 공유 중)

  부분집합 valid6 = O1/O2/O3 각 앞 2 block (O1_01 O1_02 O2_01 O2_02 O3_01 O3_02, 24 영상 · 12 쌍 · 운동×가림 4 칸 전부),
  preset final. 산출: TrainingEffects/ip1score/{loader_selftest_20260925_025512, ip1score_final_valid6_validation_20260925_025551,
  ip1score_final_valid6_scores}.json.

  (0) 로더 (--selftest, 같은 z 에 p): release latest.pt vs model.pth predictor (build_from_config, predictor_checkpoint 없음)
      state · p max|Δ| = 0.0 / ip1_e40 · ariel_ep43 의 load_pred vs build_from_config(predictor_checkpoint) = 0.0 (클래스 ·
      mask_mode='prefix' · prefix_impl='split' 일치). ariel 을 default 로 지으면 p 상대차 0.400 → kind 는 반드시 ckpt 에서.
  (a) 릴리즈 p 창 surprise vs 공식 per_window (Benchmarks vith_w16 / w32), |Δ| 중앙값 / 최대:
      skip2_w16 5.1e-5 / 1.03e-3 · skip2_w32 6.2e-5 / 8.4e-4 · skip5_w16 8.7e-5 / 1.14e-3 (창 2040/1080/240, 쌍 뒤집힘 0/12).
      기준 (h6g 공식 vs H6e): 중앙값 5.6e-5~9.8e-5, 최대 1.5e-3~3.4e-3 → 합격선 중앙값 ≤ 2e-4, 최대 ≤ 5e-3. PASS.
  (c) ip1_e40 (z_training eval e40): 5.5e-5 / 1.04e-3 · 7.0e-5 / 6.8e-4 · 1.2e-4 / 8.6e-4, 뒤집힘 0.
      ariel_ep43 (Benchmarks ariel_bc_ep43_w16/32): 3.6e-5 / 5.3e-4 · 4.7e-5 / 3.3e-4 · 8.4e-5 / 5.0e-4, 뒤집힘 0.
      나머지 final 셋 (v11_e10 · pv1_e15 · ip2_e80) 도 중앙값 ≤ 1.2e-4 · 최대 ≤ 1.65e-3 PASS. 뒤집힘은 ip2_e80 skip5_w16 의 2/12 뿐 —
      기준값 자체의 |G(pos) − G(imp)| 가 2e-5 · 2e-4 인 동점 근처 쌍 (공식 skip5_w16 180 쌍 중 34 쌍이 3e-4 안).
  (b) 복사 · 인과 열 vs H6e (auto_research/exp_results/h6/causal_l1_vith_r*.npz, 릴리즈, 튜블릿 단위):
      · **정의 동일 (비트)**: 배치만 맞추면 그대로 나온다 — max_starts=1 (H6b 배치) vs H6b p · copyz max|Δ| = 0.0 (925/925 행 ×
        2 영상), h6e_compat (H6e 배치) vs H6e 네 열 (p/copy × 표준/인과) max|Δ| ≤ 1.8e-7 (H6e 는 튜블릿 L1 을 조각별 .mean() 으로 줄여
        마지막 fp32 합산 순서만 다르다 = 몇 ulp).
      · 기본 배치 (시작점 묶음) vs H6e, 24 영상: p 중앙값 6.0e-5~8.6e-5, copy 중앙값 1.95e-4~2.6e-4, 부호 평균 |·| ≤ 3.2e-5.
        잡음 바닥 = 같은 릴리즈의 H6b vs H6e (target 배치만 다름): p 1.0e-4~1.4e-4, copy 1.4e-4~2.3e-4 → 비율 0.48~1.56. PASS
        (합격선: 중앙값 ≤ 2×바닥, |부호 평균| ≤ 1e-4).  최대 6.7e-3 은 O2_02_2 (정지) skip2_w32 C=4 j=1 한 튜블릿 —
        입력이 같은 시작점 16 · 32 에서 공식 하네스 자신도 창 값이 0.541403 / 0.541710 으로 3e-4 흔들리는 fp16 민감 칸.
      ⚠️ 정정 — 처음 판은 창 평균에서 정한 합격선 (중앙값 2e-4) 을 튜블릿 값에 그대로 걸어 copy 열을 FAIL 로 냈다
         (ip1score_final_valid6_validation.json). 튜블릿 값의 잡음이 창 평균보다 큰 것이 당연해서, 같은 릴리즈 두 실행
         (H6b vs H6e) 의 차이를 바닥으로 다시 잡았다. 판정 근거는 위 비트 검사다.
  (d) 채점 함수: 공식 per_window.json 8 개 (release w16/w32, v11_e10, ip1_e40, ip2_e80, pv1_e15, ariel_ep43 w16/w32) 를
      garrido_rescore.rescore 와 score_windows 에 넣은 결과 dict 가 완전히 같다 (88.89 skip2_w32 / 83.33 skip2_w16 재현).
  (e) 결정론: 같은 영상을 다른 실행 · 다른 predictor 목록 (final vs curves) 으로 뽑아도 공통 열 max|Δ| = 0.0. 이어 돌리기 확인
      (같은 명령 재실행 → "이미 끝남 4 → 남은 0").

  시간 · 메모리 (영상당, encoder 한 번 + predictor K 개): final K=6 18.0~20.9 s (피크 할당 7.6 GiB · 예약 9.3 GiB),
  curves K=22 50 s (8.95 / 10.45 GiB), rank 당 로드 ~65 s. 대략 7.6 s + 1.9 s × K.  전체 360 영상 추정:
  final 8 GPU ≈ 15 분 (+로드) / 4 GPU ≈ 30 분, curves 8 GPU ≈ 38 분 / 4 GPU ≈ 76 분. **curves 가 final 의 6 개를 전부 포함**하므로
  둘 다 필요하면 curves 한 번이면 된다 (--score 가 태그마다 낸다).
  ⚠️ 정정 2026-09-27 — preset 이 바뀌어 curves 는 이제 final 을 다 담지 않는다 (ariel_prefix_ep45 가 final 에만 있다). 위 시간은 09-25
     옛 preset (curves 22 = ariel_ep{5,19,30,43} 포함) 에서 잰 값이다. kind=ar 는 encoder 를 블록별로 한 번 더 돌린다 (재지 않음).
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import contextlib
import csv
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import garrido_rescore as GR  # noqa: E402  (채점 식은 이것을 그대로 쓴다)

PY = sys.executable
CACHE_ROOT = Path("/data2/local_datasets/world/world_analysis/cache/training_effects")
DOC_DIR = ROOT / "z_research/TrainingEffects/ip1score"
BENCH = ROOT / "z_research/Benchmarks/exp_results"
TRUNS = ROOT / "z_training/runs"
H6 = ROOT / "auto_research/exp_results/h6"
AUX_PAIRS = ROOT / "auto_research/_stage/IntPhys1_dev_by_scene/pairs.csv"
S_TOK, D = 256, 1280
WINDOW_BUILD = 32          # 모델 window_size (격자 최대 창). 위 docstring 참고

# ------------------------------------------------------------------ predictor 목록
PRESETS = {
    # 2026-09-27: final 의 ariel_ep43 (파일 지워짐) → ariel_prefix_ep45 (같은 run 의 마지막 epoch, 다른 파일)
    "final": ["release", "v11_e10", "ip1_e40", "ip2_e80", "pv1_e15", "ariel_prefix_ep45"],
    # 2026-09-27: ariel_ep{5,19,30,43} 은 파일이 지워졌고 새 run 에 epoch 별 체크포인트가 없어 곡선에서 뺐다 (대체 없음)
    "curves": ["release", "v11_e1", "v11_e2", "v11_e3", "v11_e5", "v11_e10",
               "pv1_e1", "pv1_e3", "pv1_e5", "pv1_e10", "pv1_e15",
               "ip1_e5", "ip1_e10", "ip1_e20", "ip1_e40",
               "ip2_e10", "ip2_e40", "ip2_e80"],
}
_PAT = [(re.compile(r"^v11_e(\d+)$"), "z_training/runs/v11_postft/e{}.pt"),
        (re.compile(r"^pv1_e(\d+)$"), "z_training/runs/predictor_v1_postft/e{}.pt"),
        (re.compile(r"^ip1_e(\d+)$"), "z_training/runs/intphys1_postft/e{}.pt"),
        (re.compile(r"^ip2_e(\d+)$"), "z_training/runs/intphys2_postft/e{}.pt")]
# Ariel 세 팔 (2026-09-27 수령). 태그 → (z_training/ariel/ 아래 폴더, models.md 레지스트리 이름 = Benchmarks 산출물 이름).
# ⚠️ 옛 ariel_ep{5,19,30,41,43} (block_causal_future_only_1e_5_ep_{N}) 는 2026-09-27 에 파일이 지워져 경로 규칙에서 뺐다.
#    ep43 을 ep45 로 바꿔 부르지 않는다. 옛 산출물 (manifest · shard 의 ariel_ep* 태그) 은 그대로 읽히고 ref_dirs 도 옛 태그를 안다.
ARIEL = {"ariel_prefix_ep45": ("block_causal_future_only_1e_5", "vith_ariel_prefix_ep45"),   # kind prefix (팔 B)
         "ariel_full_ep40": ("full_attn_future_pred_1e_5", "vith_ariel_full_ep40"),          # kind oneshot (팔 A)
         "ariel_ar_ep18": ("ar_future_1e_5", "vith_ariel_ar_ep18")}                           # kind ar (팔 C, 학습 중간)


def pred_path(tag: str) -> Path:
    if tag == "release":
        return ROOT / "z_training/runs/release_vith/latest.pt"
    if tag in ARIEL:
        return ROOT / "z_training/ariel" / ARIEL[tag][0] / "latest.pt"
    if re.fullmatch(r"ariel_ep\d+", tag):
        raise SystemExit(f"{tag}: 옛 Ariel epoch 체크포인트 (block_causal_future_only_1e_5_ep_N) 는 2026-09-27 에 지워졌다. "
                         f"지금 판: {sorted(ARIEL)} (ep43 ≠ ep45 — 바꿔 부르지 않는다)")
    for rx, fmt in _PAT:
        m = rx.match(tag)
        if m:
            return ROOT / fmt.format(m.group(1))
    raise SystemExit(f"predictor 태그 {tag!r} 의 경로 규칙이 없다 — tag=path 로 줄 것")


def parse_preds(a) -> list:
    if a.preds:
        out = []
        for item in a.preds.split(","):
            tag, _, p = item.strip().partition("=")
            out.append((tag, Path(p) if p else pred_path(tag)))
    else:
        out = [(t, pred_path(t)) for t in PRESETS[a.preset]]
    tags = [t for t, _ in out]
    if len(set(tags)) != len(tags):
        raise SystemExit(f"태그 중복: {tags}")
    for t, p in out:
        if not p.exists():
            raise SystemExit(f"{t}: {p} 없음")
    return out


# 공식 채점 산출물 (per_window.json) — 검증 (a)(c) 의 기준값
def ref_dirs(tag: str) -> list:
    if tag == "release":
        c = [BENCH / "intphys1_sliding__intphys1_dev_vith_w16", BENCH / "intphys1_sliding__intphys1_dev_vith_w32"]
    elif tag in ARIEL:                                     # 2026-09-27 세 팔: Benchmarks/run_all.sh 의 창별 실행
        c = [BENCH / f"intphys1_sliding__intphys1_dev_{ARIEL[tag][1]}_w{w}" for w in (16, 32)]
    elif (m := re.match(r"^ariel_ep(\d+)$", tag)):        # 옛 태그 (파일은 2026-09-27 에 지워짐) — 옛 산출물 검증용
        c = [BENCH / f"intphys1_sliding__intphys1_dev_vith_ariel_bc_ep{m.group(1)}_w{w}" for w in (16, 32)]
    else:
        run = {"v11": "v11_postft", "pv1": "predictor_v1_postft", "ip1": "intphys1_postft", "ip2": "intphys2_postft"}
        m = re.match(r"^(v11|pv1|ip1|ip2)_e(\d+)$", tag)
        c = [TRUNS / run[m.group(1)] / "eval" / f"intphys1_sliding__intphys1_dev_e{m.group(2)}"] if m else []
    return [d for d in c if (d / "per_window.json").exists()]


# ------------------------------------------------------------------ config · 창 배치
def resolve_cfg() -> dict:
    """표준 하네스 (resolve.py: 프로토콜 intphys1_sliding + datasets.md + models.md[vith]) 에서 config 를 받는다."""
    import tempfile
    import yaml
    with tempfile.TemporaryDirectory(prefix="te_ip1score_") as td:
        out = Path(td) / "r.yaml"
        subprocess.run([PY, str(ROOT / "z_research/scripts/harness/resolve.py"), "intphys1_sliding", "intphys1_dev",
                        "vith", "-o", str(out), "--quiet"], check=True, cwd=str(ROOT))
        cfg = yaml.safe_load(open(out))
    cfg.pop("probing", None)
    cfg["model"] = dict(cfg["model"], window_size=WINDOW_BUILD)
    for k in ("predictor_checkpoint",):
        assert k not in cfg["model"], cfg["model"]
    assert cfg["model"]["dtype"] == "float32" and cfg["model"]["autocast"] == "float16", cfg["model"]
    return cfg


class Layout:
    """창 격자와 행 배치. 순서 = combo (공식 순서) → 시작점 → C 오름차순 → j."""

    def __init__(self, cfg: dict):
        from evals.world_model_analysis.eval import _intphys1_windows
        W = cfg["surprise"]["intphys1"]
        n_frames = int(cfg["data"]["n_frames"])
        self.combos = []
        for sk in W["frame_skips"]:
            for wz in W["window_sizes"]:
                cl = [m * wz // 16 for m in W["context_mult"]]
                wins = _intphys1_windows(n_frames, int(sk), int(wz), cl, int(W["stride"]), 2, str(W["frame_budget"]))
                if not wins:
                    continue
                by = collections.OrderedDict()
                for C, cfr, tfr in wins:
                    by.setdefault(cfr[0], {})[C] = cfr + tfr
                starts = sorted(by)
                frames = [by[s][min(by[s])] for s in starts]
                for s in starts:                                      # 시작점 안에서는 C 와 무관하게 창 프레임이 같다
                    assert all(v == by[s][min(by[s])] for v in by[s].values())
                self.combos.append(dict(name=f"skip{sk}_w{wz}", skip=int(sk), W=int(wz), starts=starts,
                                        Cs=sorted({C for C, _, _ in wins}), frames=frames))
        rows, base = [], 0
        for ci, cb in enumerate(self.combos):
            U = cb["W"] // 2
            cb["base"] = base
            cb["off"] = {}
            per_start = 0
            for C in cb["Cs"]:
                cb["off"][C] = per_start
                per_start += U - C // 2
            cb["per_start"] = per_start
            for si, st in enumerate(cb["starts"]):
                for C in cb["Cs"]:
                    for j in range(U - C // 2):
                        rows.append((ci, st, C, j))
            base += per_start * len(cb["starts"])
        self.rows = np.array(rows, np.int32)                           # (R, 4)
        self.R = len(rows)
        assert base == self.R
        # 창 id: 행 → (combo_i, start, C) 의 연속 구간
        key = self.rows[:, :3]
        brk = np.r_[0, np.nonzero((key[1:] != key[:-1]).any(1))[0] + 1]
        self.win_start = brk                                           # 각 창의 첫 행
        self.win_len = np.diff(np.r_[brk, self.R])
        self.win_key = [(self.combos[int(r[0])]["name"], int(r[1]), int(r[2])) for r in key[brk]]
        self.names = [cb["name"] for cb in self.combos]
        # 인과 표적 공유: (skip, start) 가 같고 창이 더 긴 combo 는 앞 combo 의 u 를 재사용
        for cb in self.combos:
            cb["share_from"] = None
            for pb in self.combos:
                if pb is cb:
                    break
                if pb["skip"] == cb["skip"] and pb["W"] < cb["W"] and set(cb["starts"]) <= set(pb["starts"]):
                    cb["share_from"] = pb["name"]

    def idx(self, cb: dict, si: np.ndarray, C: int) -> np.ndarray:
        """(len(si), M) 행 번호."""
        M = cb["W"] // 2 - C // 2
        return cb["base"] + si[:, None] * cb["per_start"] + cb["off"][C] + np.arange(M)[None, :]

    def fingerprint(self) -> str:
        return hashlib.sha1(self.rows.tobytes()).hexdigest()[:16]


def select_videos(ds_records, subset: str | None, limit: int) -> list:
    idx = list(range(len(ds_records)))
    if subset == "valid6":           # O1/O2/O3 각 앞 2 block (4 칸 전부 포함: O1 이동가림·이동보임, O2 이동가림·정지보임, O3 정지가림 x2)
        firsts = collections.defaultdict(list)
        for r in ds_records:
            bt, b = r.raw["block_type"], r.raw["block_id"]
            if b not in firsts[bt]:
                firsts[bt].append(b)
        keep = {b for bt in firsts for b in firsts[bt][:2]}
        idx = [i for i in idx if ds_records[i].raw["block_id"] in keep]
    elif subset:
        raise SystemExit(f"--subset {subset!r}: valid6 만 있다")
    return idx[:limit] if limit else idx


def out_dir_of(a) -> Path:
    tag = a.out_tag or f"ip1score_{a.preset if not a.preds else 'custom'}"
    if not a.out_tag and a.subset:
        tag += f"_{a.subset}"
    if not a.out_tag and a.limit:
        tag += f"_lim{a.limit}"
    return CACHE_ROOT / tag


# ------------------------------------------------------------------ predictor 로드
def pred_meta(tag, path) -> dict:
    import torch
    st = torch.load(path, map_location="cpu", weights_only=False)
    arch = st.get("arch") or {}
    sd = st["predictor"]
    w = sd.get("predictor_proj.weight", sd.get("module.predictor_proj.weight"))
    return dict(tag=tag, src=str(path), size=os.path.getsize(path), mtime=os.path.getmtime(path),
                kind=arch.get("kind") or "default", format=st.get("format"), epoch=st.get("epoch"),
                n_tensors=len(sd), fp=float(w.double().abs().sum()) if w is not None else None), st


def stage_predictors(preds, shm_dir: Path) -> tuple:
    import torch
    shm_dir.mkdir(parents=True, exist_ok=False)
    staged, metas = [], []
    for tag, path in preds:
        m, st = pred_meta(tag, path)
        f = shm_dir / f"{tag}.pt"
        torch.save({"predictor": st["predictor"], "arch": st.get("arch") or {}, "src": str(path)}, f)
        staged.append((tag, str(f)))
        metas.append(m)
        print(f"  staged {tag:12s} kind={m['kind']:8s} {path}", flush=True)
    return staged, metas


def load_pred(bundle, ckpt):
    """체크포인트의 arch.kind 로 짓고 strict load (build_from_config 의 predictor_checkpoint 경로와 같은 규칙)."""
    import torch
    from analysis.intphys2.model import _build_predictor, _clean_backbone_key
    from analysis import predictors as PV
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    arch = st.get("arch") or {}
    kind = str(arch.get("kind") or "").strip() or "default"
    kk = {k: arch[k] for k in PV.KIND_ONLY_KEYS if k in arch}
    pred = _build_predictor(img_size=bundle.img_size, patch_size=bundle.patch_size, tubelet_size=bundle.tubelet_size,
                            num_frames=bundle.num_frames, encoder_embed_dim=bundle.embed_dim,
                            kind=kind, kind_kwargs=kk)
    pred.load_state_dict(_clean_backbone_key(st["predictor"]), strict=True)
    pred.attn_regime = getattr(pred, "attn_regime", "full_self_attention")
    return pred.to(device=bundle.device, dtype=bundle.dtype).eval().requires_grad_(False), kind


def _autocast(cfg):
    import torch
    dt = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(str(cfg["model"].get("autocast", "none")))
    return (lambda: torch.autocast("cuda", dtype=dt)) if dt else contextlib.nullcontext


# ------------------------------------------------------------------ GPU: 한 clip
def process_clip(clip, lay: Layout, bundle, preds, ac, max_starts: int, h6e_compat: bool = False):
    """-> dict(p_std (K,R), p_causal (K,R), copy_std (R,), copy_causal (R,)) numpy float32.

    배치: target encoder · context encoder · predictor 모두 시작점 묶음 (max_starts 개씩, 0 = 전부).
    max_starts=1 → 전부 배치 1 (= H6b 의 배치). h6e_compat → target 은 시작점 묶음, 문맥·predictor 는 배치 1,
    인과 표적 공유 없음 (= H6e 의 배치). 둘은 **검증용** — 정의가 같은지 비트 단위로 확인할 때만 쓴다.
    kind=ar predictor (2026-09-27): p_std = 블록별 표적 대비, p_causal = NaN, + copy_blk (R,) — 머리말 AR 절.
    """
    import torch
    import torch.nn.functional as F
    from analysis import predictors as PV
    from analysis.intphys2.surprise import _context_target_indices
    from analysis.predictors import ar_scoring as ARS
    dev = bundle.device
    K, R = len(preds), lay.R
    ar = [PV.is_ar(m) for m in preds]
    any_ar = any(ar)
    o_ps = torch.full((K, R), float("nan"), device=dev)
    o_pc = torch.full((K, R), float("nan"), device=dev)
    o_cs = torch.full((R,), float("nan"), device=dev)
    o_cc = torch.full((R,), float("nan"), device=dev)
    o_cb = torch.full((R,), float("nan"), device=dev) if any_ar else None
    share = {}                                                    # (skip, start) -> (U, S, D) 인과 표적
    ln = lambda t: F.layer_norm(t.float(), (D,))                  # noqa: E731
    with torch.inference_mode():
        for cb in lay.combos:
            nS_all, W, U = len(cb["starts"]), cb["W"], cb["W"] // 2
            step = max_starts or nS_all
            for s0 in range(0, nS_all, step):
                si = np.arange(s0, min(nS_all, s0 + step))
                starts = [cb["starts"][i] for i in si]
                n = len(si)
                X = torch.stack([clip[:, cb["frames"][i]] for i in si]).to(dev, bundle.dtype)
                with ac():
                    hfull = ln(bundle.target_encoder(X)).view(n, U, S_TOK, D)
                hbr = hbv = None
                if any_ar:                                            # 블록별 LN(target) — 시작점마다 한 번 (eval.py AR 분기와 같다)
                    if W % 2 or any(C % 2 for C in cb["Cs"]):
                        raise ValueError(f"kind=ar: 창 {W} · 문맥 {cb['Cs']} 가 tubelet 2 의 배수가 아니다")
                    with ac():
                        hbr = ARS.encode_blocks(bundle.target_encoder, X, 2)          # (n, U*S, D), 이미 LN
                    hbv = hbr.float().view(n, U, S_TOK, D)
                hc = torch.empty(n, U, S_TOK, D, device=dev)
                for u in range(min(cb["Cs"]) // 2, U):
                    got = [share.get((cb["skip"], st)) for st in starts] if cb["share_from"] and not h6e_compat else [None]
                    if all(g is not None and g.shape[0] > u for g in got):
                        hc[:, u] = torch.stack([g[u] for g in got])
                        continue
                    with ac():
                        o = bundle.target_encoder(X[:, :, : 2 * (u + 1)])
                    hc[:, u] = ln(o).view(n, u + 1, S_TOK, D)[:, u]
                if any(c["share_from"] == cb["name"] for c in lay.combos):
                    for k_, st in enumerate(starts):
                        share[(cb["skip"], st)] = hc[k_]
                sub = [np.arange(n)] if not h6e_compat else [np.array([q]) for q in range(n)]
                for C in cb["Cs"]:
                    Tc = C // 2
                    for q in sub:                                  # h6e_compat 만 시작점 하나씩
                        m = len(q)
                        ci, ti = _context_target_indices(ctx_frames=C, tgt_frames=W - C, tubelet_size=2,
                                                         spatial_tokens=S_TOK, batch_size=m, device=dev)
                        with ac():
                            z = bundle.context_encoder(X[q] if m < n else X, masks=[ci])
                        zT = ln(z).view(m, Tc, S_TOK, D)[:, Tc - 1][:, None]
                        hf, hcc = hfull[q, Tc:], hc[q, Tc:]
                        rid = torch.as_tensor(lay.idx(cb, si[q], C), device=dev).reshape(-1)
                        o_cs[rid] = (zT - hf).abs().mean((2, 3)).reshape(-1)
                        o_cc[rid] = (zT - hcc).abs().mean((2, 3)).reshape(-1)
                        if any_ar:                                     # AR 전용 복사: 블록 공간의 문맥 마지막 블록
                            o_cb[rid] = (hbv[q, Tc - 1][:, None] - hbv[q, Tc:]).abs().mean((2, 3)).reshape(-1)
                        for k, pred in enumerate(preds):
                            if ar[k]:                                  # kind=ar: rollout, 표적 = 블록별. p_causal 은 NaN 으로 둔다
                                with ac():
                                    p, _, _ = ARS.rollout_from_blocks(pred, hbr[q], Tc, S_TOK)
                                ARS.assert_finite(p, "te_intphys1_score")
                                p = p.float().view(m, U - Tc, S_TOK, D)
                                o_ps[k, rid] = (p - hbv[q, Tc:]).abs().mean((2, 3)).reshape(-1)
                                del p
                                continue
                            with ac():
                                p = pred(z, ci, ti, mask_index=0)
                            p = p.float().view(m, U - Tc, S_TOK, D)
                            o_ps[k, rid] = (p - hf).abs().mean((2, 3)).reshape(-1)
                            o_pc[k, rid] = (p - hcc).abs().mean((2, 3)).reshape(-1)
                            del p
                        del z
                del X, hfull, hbr, hbv
    cols = dict(p_std=o_ps, p_causal=o_pc, copy_std=o_cs, copy_causal=o_cc)
    if any_ar:
        cols["copy_blk"] = o_cb
    out = {k: v.cpu().numpy() for k, v in cols.items()}
    for k, v in out.items():
        if k == "p_causal" and any_ar:                         # AR 행은 정의상 NaN — 나머지 행만 검사
            v = v[[i for i in range(K) if not ar[i]]]
        assert np.isfinite(v).all(), f"{k}: 채워지지 않은 행 {int((~np.isfinite(v)).sum())}"
    return out


def _save_atomic(path: Path, **arrs):
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        np.savez(f, **arrs)
    os.replace(tmp, path)


def worker(rank: int, job: dict):
    import torch
    from analysis.intphys2.model import build_from_config
    from evals.world_model_analysis.data import WMADataset
    torch.set_num_threads(job["threads"])
    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)
    cfg = job["cfg"]
    lay = Layout(cfg)
    ds = WMADataset(cfg)
    my = job["pending"][rank:: job["world"]]
    if not my:
        print(f"[r{rank}] 할 영상 없음", flush=True)
        return
    t0 = time.time()
    bundle = build_from_config(cfg["model"], dev)
    bundle.predictor = None                                      # model.pth predictor 는 안 쓴다 (release 는 latest.pt 로)
    torch.cuda.empty_cache()
    preds, tags = [], []
    for tag, f in job["staged"]:
        m, kind = load_pred(bundle, f)
        preds.append(m); tags.append(tag)
    t_load = time.time() - t0
    print(f"[r{rank}] encoder+predictor {len(preds)} 로드 {t_load:.0f}s, 영상 {len(my)}", flush=True)
    ac = _autocast(cfg)
    shard_dir = Path(job["out"]) / "shards"
    pool = cf.ThreadPoolExecutor(1)
    fut = pool.submit(ds.clip, my[0])
    peak_load = torch.cuda.max_memory_allocated(dev) / 2**30
    torch.cuda.reset_peak_memory_stats(dev)
    secs, t_all = [], time.time()
    for n, vi in enumerate(my):
        clip = fut.result()
        if n + 1 < len(my):
            fut = pool.submit(ds.clip, my[n + 1])
        ts = time.time()
        res = process_clip(clip, lay, bundle, preds, ac, job["max_starts"])
        sec = time.time() - ts
        secs.append(sec)
        _save_atomic(shard_dir / f"v{vi:04d}.npz", **res, video_id=np.array(ds.records[vi].video_id),
                     video_index=np.int32(vi), tags=np.array(tags), sec=np.float32(sec))
        el = time.time() - t_all
        print(f"[r{rank}] {n + 1}/{len(my)} {ds.records[vi].video_id} {sec:.1f}s  경과 {el / 60:.1f}분  "
              f"남은 {el / (n + 1) * (len(my) - n - 1) / 60:.1f}분  peak {torch.cuda.max_memory_allocated(dev) / 2**30:.2f}GiB",
              flush=True)
    pool.shutdown()
    rep = dict(rank=rank, n=len(my), load_s=t_load, sec=secs, peak_load_alloc_gib=peak_load,
               peak_alloc_gib=torch.cuda.max_memory_allocated(dev) / 2**30,
               peak_reserved_gib=torch.cuda.max_memory_reserved(dev) / 2**30, n_preds=len(preds), gpu=torch.cuda.get_device_name(dev))
    with open(Path(job["out"]) / f"timing_r{rank}_{int(time.time())}.json", "w") as f:
        json.dump(rep, f, indent=1)


def extract(a):
    import torch.multiprocessing as mp
    preds = parse_preds(a)
    out = out_dir_of(a)
    cfg = resolve_cfg()
    lay = Layout(cfg)
    from evals.world_model_analysis.data import WMADataset
    ds = WMADataset(cfg)
    sel = select_videos(ds.records, a.subset, a.limit)
    (out / "shards").mkdir(parents=True, exist_ok=True)
    shm = Path(f"/dev/shm/te_ip1score_{os.getpid()}_{uuid.uuid4().hex[:6]}")
    try:
        print(f"predictor {len(preds)} 개 → {shm}", flush=True)
        staged, metas = stage_predictors(preds, shm)
        man = dict(tags=[t for t, _ in preds], preds=metas, layout=dict(combos=lay.names, R=lay.R, fp=lay.fingerprint(),
                   rows_per_combo={cb["name"]: cb["per_start"] * len(cb["starts"]) for cb in lay.combos}),
                   model=cfg["model"], surprise=cfg["surprise"], data=cfg["data"], window_build=WINDOW_BUILD,
                   script=str(Path(__file__).relative_to(ROOT)),
                   # 2026-09-27: p_std 의 표적 공간 (kind=ar → 블록별). ar_tags 가 있으면 shard 에 copy_blk 가 붙고 p_causal 은 NaN
                   space={m["tag"]: ("ar_block" if m["kind"] == "ar" else "window") for m in metas},
                   ar_tags=[m["tag"] for m in metas if m["kind"] == "ar"])
        mf = out / "manifest.json"
        if mf.exists():
            old = json.load(open(mf))
            same = (old["tags"] == man["tags"] and old["layout"]["fp"] == man["layout"]["fp"]
                    and [(p["src"], p["size"], p["fp"]) for p in old["preds"]] == [(p["src"], p["size"], p["fp"]) for p in metas])
            if not same:
                raise SystemExit(f"{mf} 의 predictor/창 배치가 이번 실행과 다르다 — 다른 --out_tag 를 쓸 것")
        else:
            json.dump(man, open(mf, "w"), indent=1)
        done = {int(p.stem[1:]) for p in (out / "shards").glob("v*.npz")}
        pending = [i for i in sel if i not in done]
        print(f"영상 {len(sel)} (이미 끝남 {len(sel) - len(pending)}) → 남은 {len(pending)}, GPU {a.gpus}, 출력 {out}", flush=True)
        if pending:
            job = dict(cfg=cfg, staged=staged, pending=pending, world=a.gpus, out=str(out), threads=a.threads,
                       max_starts=a.max_starts)
            mp.spawn(worker, args=(job,), nprocs=a.gpus, join=True)     # GPU 1 장이어도 같은 경로 (검증 = 본 실행)
    finally:
        if shm.exists():
            shutil.rmtree(shm)
    left = [i for i in sel if not (out / "shards" / f"v{i:04d}.npz").exists()]
    print(f"끝. 남은 영상 {len(left)}", flush=True)


# ------------------------------------------------------------------ CPU: 병합 · 채점 · 검증
def load_arrays(out: Path) -> dict:
    man = json.load(open(out / "manifest.json"))
    shards = sorted((out / "shards").glob("v*.npz"))
    if not shards:
        raise SystemExit(f"{out}/shards 가 비었다")
    mg = out / f"arrays_v{len(shards)}.npz"
    if mg.exists():
        z = np.load(mg)
        return {k: z[k] for k in z.files} | {"manifest": man}
    P, PC, CS, CC, CB, V, VI, SEC = [], [], [], [], [], [], [], []
    has_cb = bool(man.get("ar_tags"))                          # 2026-09-27: AR 이 있을 때만 copy_blk
    for f in shards:
        z = np.load(f)
        assert list(z["tags"]) == man["tags"], f"{f}: 태그 {list(z['tags'])} != manifest {man['tags']}"
        P.append(z["p_std"]); PC.append(z["p_causal"]); CS.append(z["copy_std"]); CC.append(z["copy_causal"])
        if has_cb:
            CB.append(z["copy_blk"])
        V.append(str(z["video_id"])); VI.append(int(z["video_index"])); SEC.append(float(z["sec"]))
    cfg = dict(data=man["data"], surprise=man["surprise"], model=man["model"])
    lay = Layout(cfg)
    assert lay.fingerprint() == man["layout"]["fp"]
    arr = dict(p_std=np.stack(P, 1), p_causal=np.stack(PC, 1), copy_std=np.stack(CS), copy_causal=np.stack(CC),
               video_ids=np.array(V), video_index=np.array(VI, np.int32), sec=np.array(SEC, np.float32),
               tags=np.array(man["tags"]), rows=lay.rows, combos=np.array(lay.names))
    if has_cb:
        arr["copy_blk"] = np.stack(CB)
    _save_atomic(mg, **arr)
    print(f"병합 → {mg}", flush=True)
    return arr | {"manifest": man}


def window_dict(arr: dict, lay: Layout, vals: np.ndarray) -> dict:
    """vals (V, R) → per_window.json 형식 {vid: [(combo, start, C, S), ...]}, S = mean_j."""
    s = np.add.reduceat(vals.astype(np.float64), lay.win_start, axis=1) / lay.win_len[None, :]
    return {str(v): [(c, st, C, float(x)) for (c, st, C), x in zip(lay.win_key, s[i])] for i, v in enumerate(arr["video_ids"])}


def index_meta(cfg_data: dict) -> dict:
    rows = list(csv.DictReader(open(os.path.join(cfg_data["root"], cfg_data.get("index_csv", "index.csv")))))
    meta = {r["video_id"]: dict(r) for r in rows}
    if AUX_PAIRS.exists():                                                  # 운동 × 가림 (H6g 와 같은 라벨)
        mo, vi = {"정지": "static", "이동": "moving"}, {"눈앞": "visible", "가려짐": "occluded"}
        for p in csv.DictReader(open(AUX_PAIRS)):
            g = f'{mo.get(p["label_motion"])}/{vi.get(p["label_vis"])}'
            for v in (p["pos"], p["imp"]):
                if v in meta:
                    meta[v]["mv"] = g
    return meta


def score_windows(pw: dict, meta: dict, group_by: str = "block_type") -> dict:
    """garrido_rescore.rescore 와 같은 계산을 메모리의 per_window 로 (score_table · pairwise 를 그대로 부른다)."""
    combos = sorted({w[0] for rec in pw.values() for w in rec})
    report = {}
    for combo in combos:
        vids, starts, Cs, arr = GR.score_table(pw, combo)
        entry = {"n_starts": len(starts), "context_lengths": Cs, "reductions": {}}
        cand = {"filtered": np.nanmin(arr, axis=2)}
        for j, C in enumerate(Cs):
            cand[f"C{C}"] = arr[:, :, j]
        for red, m in cand.items():
            sc = {v: float(np.nanmean(m[i])) for i, v in enumerate(vids)}
            g, bad = GR.pairwise(sc, meta, group_by)
            tot = sum(x[0] for x in g.values()); cor = sum(x[1] for x in g.values())
            entry["reductions"][red] = {
                "overall": cor / tot * 100 if tot else float("nan"),
                "n_pair": tot, "n_ties": sum(x[2] for x in g.values()), "dropped_groups": bad,
                "per_group": {k: {"n": v[0], "acc": v[1] / v[0] * 100} for k, v in sorted(g.items())}}
        report[combo] = entry
    return report


def pair_outcomes(pw: dict, meta: dict, combo: str) -> tuple:
    """Filtered 영상 점수로 쌍마다 1 (pos<imp) / 0 (pos>imp) / -1 (동점). 쌍 순서 = (block_id, pair_id) 정렬."""
    vids, _, _, arr = GR.score_table(pw, combo)
    sc = {v: float(np.nanmean(np.nanmin(arr[i], axis=1))) for i, v in enumerate(vids)}
    by = collections.defaultdict(list)
    for v, s in sc.items():
        r = meta[v]
        by[GR.pair_key(r)].append((int(r["plausible"]), s, v))
    keys, oc = [], []
    for k in sorted(by, key=lambda x: (int(x[0]), int(x[1]))):
        pos = [m for m in by[k] if m[0] == 1]; imp = [m for m in by[k] if m[0] == 0]
        if len(pos) != 1 or len(imp) != 1:
            continue
        keys.append((pos[0][2], imp[0][2]))
        oc.append(1 if pos[0][1] < imp[0][1] else (-1 if pos[0][1] == imp[0][1] else 0))
    return keys, oc


def _new_name(p: Path) -> Path:
    if not p.exists():
        return p
    return p.with_name(p.stem + datetime.datetime.now().strftime("_%Y%m%d_%H%M%S") + p.suffix)


def score(a):
    out = out_dir_of(a)
    arr = load_arrays(out)
    man = arr["manifest"]
    lay = Layout(dict(data=man["data"], surprise=man["surprise"], model=man["model"]))
    meta = index_meta(man["data"])
    methods = [(t, k) for k, t in enumerate(man["tags"])] + [("copy", None)]
    ar_tags = set(man.get("ar_tags") or [])                    # 2026-09-27: kind=ar — std 열 = 블록별 표적, causal 없음
    if "copy_blk" in arr:
        methods.append(("copy_blk", "blk"))
    res = {"out_tag": out.name, "arrays": str(out), "n_videos": int(len(arr["video_ids"])),
           "preds": {p["tag"]: p["src"] for p in man["preds"]},
           "target_space": man.get("space") or {t: "window" for t in man["tags"]},
           "ar_note": ("kind=ar 태그의 std 칸은 블록별 LN(target) 표적 (다른 kind 의 std 와 다른 공간) — 쌍 정확도만 비교, "
                       "복사 대조는 copy_blk. AR 과 copy_blk 에는 causal 칸이 없다") if ar_tags else None,
           "rule": "Garrido A.8 IntPhys (garrido_rescore.py 함수): S=mean_j L1_j; Filtered=min_C per start; "
                   "AvgSurprise=mean over starts; 쌍 정답 = S(pos)<S(imp) strict (동점 오답, n_ties); property macro "
                   "(O1/O2/O3); best = macro 최대 combo. mv = 운동×가림 (auto_research/_stage 라벨, 쌍 수 가중 아님 → 칸별 acc).",
           "outcome_code": "1 = pos<imp (정답), 0 = pos>imp, -1 = 동점",
           "cells": {}, "outcomes": {}, "pairs": {}}
    for tgt in ("std", "causal"):
        for name, k in methods:
            if tgt == "causal" and (name in ar_tags or k == "blk"):
                continue                                         # AR 에는 인과 표적 열이 없다 (NaN)
            vals = arr["copy_blk"] if k == "blk" else (arr[f"copy_{tgt}"] if k is None else arr[f"p_{tgt}"][k])
            pw = window_dict(arr, lay, vals)
            rep = score_windows(pw, meta)
            best = GR.select(rep, "intphys")
            cell = {"best": {"combo": best["combo"], "macro": best["accuracy"], "alt_notebook": best["alt_notebook"]},
                    "combos": {}}
            for cb, e in rep.items():
                R_ = e["reductions"]
                sel = e["selected"]
                mv = {}
                if any("mv" in m for m in meta.values()):
                    tv, _, _, ta = GR.score_table(pw, cb)
                    g, _ = GR.pairwise({v: float(np.nanmean(np.nanmin(ta[i], axis=1))) for i, v in enumerate(tv)}, meta, "mv")
                    mv = {kk: {"n": v[0], "acc": v[1] / v[0] * 100} for kk, v in sorted(g.items())}
                cell["combos"][cb] = {"macro": sel["accuracy"], "alt_notebook": sel["alt_notebook"],
                                      "per_group": {g_: v["acc"] for g_, v in R_["filtered"]["per_group"].items()},
                                      "n_pair": R_["filtered"]["n_pair"], "n_ties": R_["filtered"]["n_ties"],
                                      "dropped": R_["filtered"]["dropped_groups"],
                                      "per_C_macro": {r: float(np.mean([x["acc"] for x in R_[r]["per_group"].values()]))
                                                      for r in R_ if r != "filtered"},
                                      "mv": mv}
                keys, oc = pair_outcomes(pw, meta, cb)
                res["outcomes"][f"{name}|{tgt}|{cb}"] = oc
                res["pairs"].setdefault(cb, [[p, i, meta[p]["block_type"], meta[p].get("mv", "")] for p, i in keys])
            res["cells"].setdefault(name, {})[tgt] = cell
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    f = _new_name(DOC_DIR / f"{out.name}_scores.json")
    json.dump(res, open(f, "w"), indent=1, ensure_ascii=False)
    shutil.copyfile(f, _new_name(out / f.name))
    combos = lay.names
    print(f"\n영상 {res['n_videos']} · 칸 = Filtered property macro (%)  [p − copy 는 같은 쌍 위 차]")
    print(f"{'method':12s} {'tgt':6s} " + " ".join(f"{c:>10s}" for c in combos) + "   best")
    for name, _ in methods:
        for tgt in ("std", "causal"):
            c = res["cells"].get(name, {}).get(tgt)
            if c is None:                                        # AR · copy_blk 의 causal
                continue
            dd = " ".join(f"{c['combos'][cb]['macro']:10.2f}" for cb in combos)
            print(f"{name:12s} {tgt:6s} {dd}   {c['best']['combo']} {c['best']['macro']:.2f}"
                  + ("   [ar_block 표적]" if name in ar_tags or name == "copy_blk" else ""))
    print(f"저장: {f}")


def _h6_rows(pattern: str, want: set) -> dict:
    """auto_research H6 npz → {(vid, combo, start, C, j): 값 배열}  (H6e: p_causal, copyz_causal, p_full, copyz_full /
    H6b: p, copyh, copyz, zero — 표준 표적)."""
    parts = sorted(H6.glob(pattern))
    if not parts:
        return {}
    z0 = np.load(parts[0])
    hv, hc = [str(x) for x in z0["video_ids"]], [str(x) for x in z0["combos"]]
    out = {}
    for f in parts:
        for r in np.load(f)["rows"]:
            v = hv[int(r[0])]
            if v in want:
                out[(v, hc[int(r[1])], int(r[2]), int(r[3]), int(r[4]))] = r[5:]
    return out


def _diffstats(d):
    s = np.asarray(d, np.float64)
    d = np.abs(s)
    return {"n": int(d.size), "median": float(np.median(d)), "p99": float(np.percentile(d, 99)), "max": float(d.max()),
            "mean_signed": float(s.mean())} if d.size else {"n": 0}


def validate(a):
    """(a)(c) 공식 per_window.json 대조, (b) H6e 대조, (d) 채점 함수 = garrido_rescore."""
    out = out_dir_of(a)
    arr = load_arrays(out)
    man = arr["manifest"]
    lay = Layout(dict(data=man["data"], surprise=man["surprise"], model=man["model"]))
    meta = index_meta(man["data"])
    vids = [str(v) for v in arr["video_ids"]]
    vset = set(vids)
    TOL_MED, TOL_MAX = 2e-4, 5e-3        # 공식 하네스 vs H6e 재추출 (h6g): 중앙값 5.6e-5~9.8e-5, 최대 1.5e-3~3.4e-3
    res = {"out_tag": out.name, "n_videos": len(vids), "tolerance": {"median": TOL_MED, "max": TOL_MAX,
           "basis": "h6g official_vs_h6e: median 5.6e-5/6.3e-5/9.8e-5, max 3.4e-3/1.5e-3/2.3e-3 (skip2_w16/skip2_w32/skip5_w16)"},
           "per_window": {}, "h6e": {}, "rescore_equal": {}, "pairs": {}}
    ok = True
    # (a)(c) 공식 per_window
    for k, tag in enumerate(man["tags"]):
        dirs = ref_dirs(tag)
        if not dirs:
            continue
        ours = window_dict(arr, lay, arr["p_std"][k])
        ref = {}
        for d in dirs:
            for v, rows in json.load(open(d / "per_window.json"))["windows"].items():
                if v in vset:
                    ref.setdefault(v, []).extend(rows)
        ent = {"ref_dirs": [str(d.relative_to(ROOT)) for d in dirs]}
        for cb in lay.names:
            R_ = {(v, int(s), int(C)): float(x) for v, rows in ref.items() for c, s, C, x in rows if c == cb}
            O_ = {(v, s, C): x for v, rows in ours.items() for c, s, C, x in rows if c == cb}
            common = sorted(set(R_) & set(O_))
            if not common:
                continue
            st = _diffstats([O_[q] - R_[q] for q in common])
            st["missing_in_ref"] = len(set(O_) - set(R_))
            # 쌍 단위 (이 부분집합 안의 완전한 쌍)
            ko, oo = pair_outcomes(ours, meta, cb)
            kr, orr = pair_outcomes({v: [r for r in rows if r[0] == cb] for v, rows in ref.items()}, meta, cb)
            mo, mr = dict(zip(ko, oo)), dict(zip(kr, orr))
            flips = [list(q) for q in mo if q in mr and mo[q] != mr[q]]
            st.update(n_pairs=len(set(mo) & set(mr)), pair_flips=flips,
                      pass_=bool(st["median"] <= TOL_MED and st["max"] <= TOL_MAX))
            ok &= st["pass_"]
            ent[cb] = st
        res["per_window"][tag] = ent
        # (d) 채점 함수 동치: 같은 per_window.json 을 garrido_rescore.rescore 와 우리 score_windows 에
        for d in dirs:
            g = GR.rescore(d)["combos"]
            full = json.load(open(d / "per_window.json"))["windows"]
            mine = score_windows({v: [tuple(r) for r in rows] for v, rows in full.items()}, meta)
            eq = json.dumps(g, sort_keys=True) == json.dumps(mine, sort_keys=True)
            res["rescore_equal"][str(d.relative_to(ROOT))] = eq
            ok &= eq
    # (b) H6e (릴리즈 model.pth predictor, 튜블릿 단위). 기준 = 잡음 바닥: 같은 릴리즈 모델의 H6b 와 H6e 가
    #     서로 얼마나 다른가 (target encoder 배치만 다르다; 표준 표적 열만 있다 → 인과 열은 같은 방법의 표준 열 바닥을 쓴다).
    #     합격 = 중앙값 ≤ 2 × 바닥 중앙값 이고 부호 평균 |mean| ≤ 1e-4 (치우침 없음). p99 · max 는 비율과 함께 보고만 한다.
    #     정의 자체의 동일성은 --selftest 의 비트 검사 (배치를 H6b/H6e 와 같게 하면 max|Δ| = 0) 가 맡는다.
    if "release" in man["tags"]:
        E, B = _h6_rows("causal_l1_vith_r*.npz", vset), _h6_rows("tubelet_l1_vith_r*.npz", vset)
        kr = man["tags"].index("release")
        pos = {(vid, lay.names[int(r[0])], int(r[1]), int(r[2]), int(r[3])): (vi_, ri)
               for vi_, vid in enumerate(vids) for ri, r in enumerate(lay.rows)}
        cols = (("p_causal", "p_causal", kr, 0), ("copyz_causal", "copy_causal", None, 1),
                ("p_full", "p_std", kr, 2), ("copyz_full", "copy_std", None, 3))
        floor_col = {"p_causal": 0, "p_full": 0, "copyz_causal": 2, "copyz_full": 2}     # H6b 열: p=0, copyz=2
        floor_h6e = {"p_causal": 2, "p_full": 2, "copyz_causal": 3, "copyz_full": 3}
        for cb in lay.names:
            keys = [k for k in E if k[1] == cb and k[:1][0] in vset and k in pos]
            if not keys:
                continue
            for hname, oname, kk, c in cols:
                ours_v = np.array([(arr[oname][kk] if kk is not None else arr[oname])[pos[k]] for k in keys], np.float64)
                st = _diffstats(ours_v - np.array([E[k][c] for k in keys]))
                bk = [k for k in keys if k in B]
                fl = _diffstats(np.array([B[k][floor_col[hname]] - E[k][floor_h6e[hname]] for k in bk]))
                st["floor_H6b_vs_H6e"] = fl
                st["ratio_median"] = st["median"] / fl["median"]
                st["ratio_p99"] = st["p99"] / fl["p99"]
                st["pass_"] = bool(st["median"] <= 2 * fl["median"] and abs(st["mean_signed"]) <= 1e-4)
                ok &= st["pass_"]
                res["h6e"][f"{cb}|{hname}"] = st
        res["h6e_rule"] = "median ≤ 2×floor median and |mean_signed| ≤ 1e-4; floor = |H6b − H6e| on the same rows"
        sts = sorted(DOC_DIR.glob("loader_selftest*.json"))
        if sts:
            bx = json.load(open(sts[-1])).get("bitexact", {})
            res["selftest_bitexact"] = {"file": str(sts[-1].relative_to(ROOT)),
                                        "max_abs_by_check": {k: v["max_abs"] for k, v in bx.items()}}
    res["pass"] = bool(ok)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    f = _new_name(DOC_DIR / f"{out.name}_validation.json")
    json.dump(res, open(f, "w"), indent=1, ensure_ascii=False)
    for tag, ent in res["per_window"].items():
        for cb, st in ent.items():
            if cb == "ref_dirs":
                continue
            print(f"(a/c) {tag:11s} {cb:10s} n={st['n']:5d} median {st['median']:.2e} p99 {st['p99']:.2e} max {st['max']:.2e}"
                  f"  pairs {st['n_pairs']} flips {len(st['pair_flips'])}  {'PASS' if st['pass_'] else 'FAIL'}")
    for kk, st in res["h6e"].items():
        fl = st["floor_H6b_vs_H6e"]
        print(f"(b) H6e {kk:24s} n={st['n']:6d} median {st['median']:.2e} (floor {fl['median']:.2e}, x{st['ratio_median']:.2f})"
              f" p99 {st['p99']:.2e} (x{st['ratio_p99']:.2f}) max {st['max']:.2e} (floor {fl['max']:.2e}) bias {st['mean_signed']:+.1e}"
              f"  {'PASS' if st['pass_'] else 'FAIL'}")
    for k, v in (res.get("selftest_bitexact") or {}).get("max_abs_by_check", {}).items():
        print(f"(b') selftest {k:42s} max|Δ| {v:.3e}")
    for d, eq in res["rescore_equal"].items():
        print(f"(d) score_windows == garrido_rescore  {d}: {eq}")
    print(f"전체 {'PASS' if ok else 'FAIL'} → {f}")


def selftest(a):
    """로더 동치: (1) release latest.pt == model.pth predictor (build_from_config, predictor_checkpoint 없음),
    (2) 각 태그의 load_pred == build_from_config(predictor_checkpoint=…) 의 predictor (CPU 로 지어 GPU 로 옮김).
    같은 z 를 넣은 p 의 max|Δ|, state_dict max|Δ|, 클래스·mask_mode·prefix_impl. 덤: prefix ckpt 를 default 로 지으면 얼마나 다른가."""
    import torch
    from analysis import predictors as PV
    from analysis.intphys2.model import build_from_config
    from analysis.intphys2.surprise import _context_target_indices
    from analysis.predictors import ar_scoring as ARS
    from evals.world_model_analysis.data import WMADataset
    cfg = resolve_cfg()
    lay = Layout(cfg)
    ds = WMADataset(cfg)
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    ac = _autocast(cfg)
    bundle = build_from_config(cfg["model"], dev)
    clip = ds.clip(0)
    cb = next(c for c in lay.combos if c["name"] == "skip2_w32")
    X = torch.stack([clip[:, cb["frames"][i]] for i in (0, 4)]).to(dev, bundle.dtype)
    tests = {}
    with torch.inference_mode():
        Z = {}
        for C in (4, 12):
            ci, ti = _context_target_indices(ctx_frames=C, tgt_frames=32 - C, tubelet_size=2, spatial_tokens=S_TOK,
                                             batch_size=2, device=dev)
            with ac():
                Z[C] = (bundle.context_encoder(X, masks=[ci]), ci, ti)

        HB = []                                                   # kind=ar 용 블록별 표적 (처음 필요할 때 한 번)

        def run(pred):
            outs = []
            if PV.is_ar(pred):                                    # 2026-09-27: kind=ar 는 같은 창 · 같은 C 의 rollout 으로 대조
                if not HB:
                    with ac():
                        HB.append(ARS.encode_blocks(bundle.target_encoder, X, 2))
                for C in Z:
                    with ac():
                        outs.append(ARS.rollout_from_blocks(pred, HB[0], C // 2, S_TOK)[0].float())
                return outs
            for C, (z, ci, ti) in Z.items():
                with ac():
                    outs.append(pred(z, ci, ti, mask_index=0).float())
            return outs

        def cmp(m1, m2):
            s1, s2 = m1.state_dict(), m2.state_dict()
            sd = max(float((s1[k].float() - s2[k].float()).abs().max()) for k in s1) if set(s1) == set(s2) else None
            o1, o2 = run(m1), run(m2)
            pd = max(float((x - y).abs().max()) for x, y in zip(o1, o2))
            rel = max(float((x - y).abs().mean() / y.abs().mean()) for x, y in zip(o1, o2))
            attrs = lambda m: {k: getattr(m, k, None) for k in ("mask_mode", "prefix_impl", "n_registers")}  # noqa: E731
            return {"class": [type(m1).__name__, type(m2).__name__], "attrs": [attrs(m1), attrs(m2)],
                    "keys_equal": set(s1) == set(s2), "state_max_abs": sd, "p_max_abs": pd, "p_rel_mean": rel}

        preds = parse_preds(a) if (a.preds or a.preset) else []
        # 2026-09-27: 기본 목록의 ariel_ep43 (파일 지워짐) → ariel_prefix_ep45
        want = [t for t, _ in preds] if preds else ["release", "ip1_e40", "ariel_prefix_ep45"]
        paths = dict(preds) if preds else {t: pred_path(t) for t in want}
        mine_rel, _ = load_pred(bundle, paths.get("release", pred_path("release")))
        tests["release_latest.pt_vs_model.pth"] = cmp(mine_rel, bundle.predictor)
        for t in want:
            if t == "release":
                continue
            m, kind = load_pred(bundle, paths[t])
            ref_b = build_from_config(dict(cfg["model"], predictor_checkpoint=str(paths[t])), torch.device("cpu"))
            ref = ref_b.predictor.to(dev).eval()
            del ref_b
            r = cmp(m, ref)
            r["kind"] = kind
            if kind not in ("default", "ar"):                     # ar 를 release 규약으로 돌리는 대조는 뜻이 없다
                from analysis.intphys2.model import _build_predictor, _clean_backbone_key
                st = torch.load(paths[t], map_location="cpu", weights_only=False)
                wrong = _build_predictor(img_size=256, patch_size=16, tubelet_size=2, num_frames=bundle.num_frames,
                                         encoder_embed_dim=bundle.embed_dim, kind="default")
                wrong.load_state_dict(_clean_backbone_key(st["predictor"]), strict=True)
                wrong = wrong.to(dev).eval()
                r["if_built_as_default"] = {k: v for k, v in cmp(wrong, ref).items() if k in ("p_max_abs", "p_rel_mean")}
                del wrong
            tests[f"{t}_load_pred_vs_build_from_config"] = r
            del m, ref
            torch.cuda.empty_cache()
    ok = all(v["keys_equal"] and v["state_max_abs"] == 0 and v["p_max_abs"] == 0 and v["class"][0] == v["class"][1]
             and v["attrs"][0] == v["attrs"][1] for v in tests.values())
    # 정의 검사 (비트 단위): 배치만 H6b / H6e 와 같게 하면 그 산출물이 그대로 나와야 한다.
    #   max_starts=1          = H6b 배치 (전부 배치 1)            → p · copyz (표준 표적) 비교
    #   h6e_compat            = H6e 배치 (target 시작점 묶음, 문맥·predictor 배치 1, 인과 공유 없음) → 네 열 전부
    bitexact = {}
    if H6.exists():
        bundle.predictor = None
        torch.cuda.empty_cache()
        E, B = _h6_rows("causal_l1_vith_r*.npz", set(a.bit_videos)), _h6_rows("tubelet_l1_vith_r*.npz", set(a.bit_videos))
        pos = {(lay.names[int(r[0])], int(r[1]), int(r[2]), int(r[3])): i for i, r in enumerate(lay.rows)}
        vids = [r.video_id for r in ds.records]
        for vid in a.bit_videos:
            clip = ds.clip(vids.index(vid))
            o1 = process_clip(clip, lay, bundle, [mine_rel], ac, 1)
            oE = process_clip(clip, lay, bundle, [mine_rel], ac, 0, h6e_compat=True)
            oN = process_clip(clip, lay, bundle, [mine_rel], ac, 0)
            for tag, ref, cols in (("batch1_vs_H6b", B, (("p_std", 0), ("copy_std", 2))),
                                   ("h6e_compat_vs_H6e", E, (("p_causal", 0), ("copy_causal", 1), ("p_std", 2), ("copy_std", 3))),
                                   ("default_vs_H6e", E, (("p_causal", 0), ("copy_causal", 1), ("p_std", 2), ("copy_std", 3)))):
                o = {"batch1_vs_H6b": o1, "h6e_compat_vs_H6e": oE, "default_vs_H6e": oN}[tag]
                keys = [k for k in ref if k[0] == vid]
                ri = np.array([pos[k[1:]] for k in keys]); rv = np.array([ref[k] for k in keys])
                for name, c in cols:
                    x = o[name][0] if name.startswith("p_") else o[name]
                    d = np.abs(x[ri] - rv[:, c])
                    bitexact[f"{vid}|{tag}|{name}"] = {"n": int(len(d)), "max_abs": float(d.max()), "median_abs": float(np.median(d)),
                                                       "n_exact": int((d == 0).sum())}
        # H6b 는 L1 을 우리와 같은 .mean((-1,-2)) 로 줄여 비트 동일해야 한다. H6e 는 튜블릿마다 (256,1280) 조각의 .mean() 이라
        # 마지막 fp32 합산 순서만 달라 몇 ulp (값 ~0.5 에서 ulp 6e-8) 차이가 난다 → 1e-6 이하면 같은 정의.
        ok &= all(v["max_abs"] == 0 for k, v in bitexact.items() if "batch1" in k)
        ok &= all(v["max_abs"] <= 1e-6 for k, v in bitexact.items() if "h6e_compat" in k)
    res = {"pass": ok, "windows": "video 0, skip2_w32 starts {0, 16}, C {4, 12}, autocast fp16", "tests": tests,
           "bitexact_doc": "batch1 = max_starts 1 (H6b 배치) → max|Δ| = 0 요구. h6e_compat = H6e 배치 → ≤ 1e-6 요구 "
                           "(H6e 의 튜블릿 L1 은 조각별 .mean() 이라 마지막 fp32 합산 순서만 다르다). "
                           "default 는 추출 배치 — 참고 (배치 구성에 따른 fp16 잡음)",
           "bitexact": bitexact}
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    f = _new_name(DOC_DIR / "loader_selftest.json")
    json.dump(res, open(f, "w"), indent=1, ensure_ascii=False)
    for k, v in tests.items():
        print(f"{k:45s} class {v['class'][0]:>34s} state {v['state_max_abs']} p max|Δ| {v['p_max_abs']:.3e}  attrs {v['attrs'][0]}"
              + (f"  (default 로 지으면 p rel {v['if_built_as_default']['p_rel_mean']:.3f})" if "if_built_as_default" in v else ""))
    for k, v in bitexact.items():
        print(f"  {k:45s} n={v['n']:4d} max|Δ| {v['max_abs']:.3e} median {v['median_abs']:.2e} exact {v['n_exact']}/{v['n']}")
    print(f"selftest {'PASS' if ok else 'FAIL'} → {f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None)
    ap.add_argument("--preds", default=None, help="tag=path,tag,... (tag 만 주면 경로 규칙)")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--subset", default=None, help="valid6 = O1/O2/O3 앞 2 block (24 영상)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out_tag", default=None)
    ap.add_argument("--max_starts", type=int, default=0, help="encoder 배치의 시작점 상한 (0 = 영상 하나의 시작점 전부)")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--bit_videos", type=lambda s: s.split(","), default=["O1_01_1", "O2_02_2"],
                    help="--selftest 의 H6b/H6e 비트 검사 영상")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest(a)
    if not a.preset and not a.preds:
        ap.error("--preset 또는 --preds")
    if a.score or a.validate:
        if a.validate:
            validate(a)
        if a.score:
            score(a)
        return
    extract(a)


if __name__ == "__main__":
    main()
