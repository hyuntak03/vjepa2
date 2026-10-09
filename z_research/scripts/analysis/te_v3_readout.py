#!/usr/bin/env python3
"""RollOut v3 — **predictor 여럿**을 같은 frozen encoder 위에서 돌려 창별로 p 의 위치·존재를 읽는다.

`rollout3_window_readout.py` (정본, 16 창 × p·z·h) 를 **predictor 목록**으로 일반화한 것이다 (TrainingEffects, 2026-09-25).
규약은 그 스크립트와 **같다** — 정의 (MODEL · DATA · labels · prefetch · 자 클래스) 를 거기서 import 하고 복제하지 않는다.

  · bf16 (`MODEL.dtype`), 창마다 모델을 다시 짓는다 (`window_size = C+P` → RoPE grid_depth · predictor num_patches)
  · 사건 f32 고정, 문맥 `[32−C, 32)`, 예측 `[32, 32+P)`. 기본 창은 C16/P16 · C16/P32 (`--c8` 로 C8/P8 · C8/P16 추가)
  · 자 = `RollOutV3/exp_results/presence/{p,h,z}/readout_attn.pt` (attn, 지문 **dcd24d8a8a47**, 다르면 죽는다)
  · 가능 clip 만 (`role == roll`, 2,744 개), index.csv 순서 그대로

무엇을 재나 (튜블릿마다, 미래 t = 0 … P/2−1)
-------------------------------------------
encoder 는 clip·창마다 **한 번만** 돈다. predictor 만 갈아 끼운다.

    z_ctx      = context_encoder(clip, masks=[문맥 토큰])                    (B, C/2·256, 1280)   predictor 입력
    p_tag      = fp16( pred_tag(z_ctx, ctx_idx, tgt_idx, mask_index=0) )    (B·P/2, 256, 1280)   LN 공간 (재정규화 안 함)
    z          = fp16( LN(context_encoder(clip))[:, 미래] )                  창 전체를 본다 = 자의 천장 (predictor 무관, 한 번)
    h          = fp16( LN(target_encoder(clip))[:, 미래] )                   〃
    자 r ∈ {p, h, z}:   a = softmax(tok·q_r/√D) (256,),  pooled = a·tok,  xy = W_r·pooled,  logit = w_r·pooled

저장 값 (열 5 개, `COLS`):  x, y (정규화 px/144 − 1, **치우침 안 뺀 원값** — 정본과 같다), logit, top1 = max a, mass3
    mass3 = Σ_{|r−r*|≤1, |c−c*|≤1} a[r·16 + c],   (c*, r*) = (⌊x_GT/18⌋, ⌊y_GT/18⌋)
            x_GT, y_GT = 그 튜블릿 두 프레임의 GT 평균 (288 px 좌표, `in_frame` 무관). 균등 attention 이면 9/256 = 0.035
            (`rollout2_two_futures_attn.py::mass3` 와 같은 정의. CLAUDE.md §5-5 "위치 주장은 3×3 질량이 균등을 넘을 때만")

키 (readings.npz 의 `C{c}_P{p}_<키>`, 모양 (n, P/2, 5) float32)
    p_<tag>      p 를 p 자로           (정본 readings 의 `p`)
    p@h_<tag>    p 를 h 자로 (이식)     (정본 `p@h`) — p 는 LN(target) 을 맞추도록 학습됐으니 원리적인 자
    p@z_<tag>    p 를 z 자로 (대조)     (정본 `p@z`)
    z, h         encoder 자기 자로      (정본 `z`, `h`) — predictor 와 무관하게 한 번
  판정 문턱은 **자를 배운 표현의 것** (`presence/summary.json` thr_val_fpr5): p 자 +0.831 · h 자 −0.961 · z 자 −0.662.
  좌표 치우침 `attn_bias_px.json` 은 쓰는 쪽이 뺀다 (meta.json 에 복사해 둔다).

predictor 교체 (`load_sd` + `build_pred`)
    st = torch.load(ckpt);  kind = st['arch'].get('kind') or 'default';  kk = arch 의 KIND_ONLY_KEYS (prefix 는 n_registers 0)
    pred = analysis.intphys2.model._build_predictor(num_frames = C+P, kind, kk, 나머지는 MODEL.predictor)
    pred.load_state_dict(_clean_backbone_key(st['predictor']), strict=True) → GPU bf16. 창 총합마다 tag 전부를 GPU 에 올려 둔다
  = `build_from_config({..., predictor_checkpoint: ckpt})` 의 predictor 경로와 같은 인자 (kind 는 **ckpt 가 선언**).
  `--check-loader` 가 그 동치를 실측한다 (아래 검증 ①).
  ⚠️ **prefix (Ariel) 는 순수 bf16 에서 그대로는 안 돈다** — `ACRoPEAttention` 이 RoPE 위치를 float (`1.0 * frame_ids`) 로 만들어
     q·k 가 fp32 로 승격되고 v 는 bf16 이라 PrefixSpec 분기의 SDPA 가 dtype 오류로 죽는다 (2026-09-25 실측). 기본 (`--pred-autocast none`)
     은 `_SDPACast` 로 **SDPA 직전에 q·k 만 v 의 dtype (bf16) 으로 내린다** — 나머지는 정본과 같은 autocast 없는 bf16.
     `te_v11_readout.py` 와 같은 처리라 두 readout 의 Ariel 수치가 맞물린다. dtype 이 같으면 no-op 이므로 default predictor ·
     encoder 는 비트 단위로 그대로다 (캐스트 횟수: ariel C16/P16 108 = 12 층 × SDPA 9 번, C16/P32 204, default 0 — validation.json).
     `--pred-autocast prefix|all` = autocast(bf16) 로 감싼다 (학습 방식, LayerNorm·RoPE 각 fp32; all 은 release 도 정본에서 벗어난다).

프리셋 (`--preset`, 또는 `--preds tag=path,tag=path`)
    final  = release, v11_e10, ip1_e40, ip2_e80, pv1_e15, ariel_prefix_ep45
    curves = release, v11_e{1,2,3,5,10}, pv1_e{1,3,5,10,15}, ip1_e{5,10,20,40}, ip2_e{10,40,80}
  `release` = `z_training/runs/release_vith/latest.pt` (model.pth predictor 추출본).
  `ariel_prefix_ep45` = `z_training/ariel/block_causal_future_only_1e_5/latest.pt` (arch.kind prefix, Ariel 팔 B, epoch 45/45).
  Ariel 팔 A (full attention, kind oneshot) 는 `--preds ariel_full_ep40=z_training/ariel/full_attn_future_pred_1e_5/latest.pt` 로 준다.
  ⚠️ 2026-09-27 — 옛 `ariel_ep{5,19,30,41,43}` (`block_causal_future_only_1e_5_ep_{N}`) 은 **파일이 지워져** preset 에서 뺐다
     (ep43 ≠ ep45, 바꿔 부르지 않는다). final 의 ariel_ep43 → ariel_prefix_ep45, curves 의 ariel_ep{5,19,30,43} 은 대체 없이 뺐다.
     옛 산출물 (`v3readout_curves/readings.npz` 등) 은 그대로 읽힌다. 옛 폴더에 새 목록을 치면 **readings.npz 에서 옛 tag 가
     빠지므로** 조립 전에 죽는다 (덮어쓰지 않는다) → --name/--out 을 바꿀 것.
  ⚠️ kind=ar (Ariel 팔 C, `ar_future_1e_5`) 는 **거부한다** (NotImplementedError): AR 의 'p' 는 블록별 LN(target) 공간의 rollout 이라
     이 스크립트의 p (release 규약 predictor 출력) 와 뜻이 다르고 자 공간에서 정의하지 않았다. bf16 · autocast 없이는 SDPA 도 죽는다.

실행 · 재개
-----------
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/te_v3_readout.py --preset final --gpus 8                    # 본 실행 (2,744 clip × 2 창)
  $P z_research/scripts/analysis/te_v3_readout.py --preset curves --gpus 8 [--c8]
  $P z_research/scripts/analysis/te_v3_readout.py --preset final --sample 48 --gpus 1 --validate   # 검증 (정본 대조)
  $P z_research/scripts/analysis/te_v3_readout.py --preset final --sample 48 --check-loader         # 로더 동치 (GPU 1 장, 검사 clip 2 개)

  · 멀티 GPU = `mp.spawn` × `--gpus N` (보이는 장치의 cuda:0..N−1, rank r 은 **미완료 clip 목록**의 r::N). clip 순서 = index.csv 순서
  · 출력 = `/data2/local_datasets/world/world_analysis/cache/training_effects/v3readout_<preset|--name>[_sample N|_limit N]/`
      _parts/C{c}_P{p}__<키>.npy   (n, P/2, 5) memmap, clip 번호 = 출력 순서 (NaN = 아직 안 씀)
      _parts/done_C{c}_P{p}__<군>.npy (n,) uint8 — 군 = predictor tag 하나 또는 `zh`. **값을 쓴 뒤에** 표시한다
      _parts/sig.json               clip 목록 sha1 · 자 지문 · autocast · tag→ckpt. 다르면 죽는다 (tag 추가는 허용 = 새 tag 만 뽑는다)
      _parts/bsz_C{c}_P{p}.npy      (n,) clip 마다 p 를 계산한 실제 배치 크기 (readings.npz 에도 `bsz_C*_P*` 로 싣는다)
      _parts/chk_*.npy · timing_r*.json   내장 검사용 토큰 (검사 clip 2 개) · rank 별 시간·최대 메모리
      readings.npz · meta.json      전부 끝나면 조립 (정본과 같은 메타 배열 + secondary · traj = scenario|primary|secondary · cols)
      validation.json               내장 검사 + (`--validate`) 정본 대조
    죽은 run 은 **같은 명령을 다시 치면** (GPU 수가 달라도) 미완료 clip 만 이어서 한다.
    ⚠️ 재개하면 배치 구성이 달라져 값이 bf16 잡음 (~1 px) 만큼 흔들린다
  · 배치 크기 (`bs_for`): 기본은 **모든 창에 같은 값** (`--token-budget` // 가장 긴 창) — 창 사이 비교 (③) 에서 배치 차이를 없앤다.
    `--base-bs` = 정본처럼 창 총합마다 (C16/P16 8 · C16/P32 5) → 정본 readings 를 비트 수준으로 재현한다.
    ⚠️ bf16 읽은 값은 **배치 크기**에 따라 흔들린다 (커널 선택). 같은 배치 크기면 옆 clip 이 달라도 같은 값이 나온다 (아래 ④)
  · ⚠️ `z_research/RollOutV3/exp_results/windows` 에는 절대 쓰지 않는다 (정본은 읽기만)

내장 검사 (매 실행, validation.json)
------------------------------------
  ② release 동치: 창 총합마다 rank 0 이 검사 clip 2 개로 `build_from_config` (model.pth predictor) 의 p 와
     `release` tag (latest.pt) 의 p 를 비교 → max|Δ|, bitwise
  ③ 창 독립성: 같은 C 에서 P 만 다른 두 창의 겹치는 튜블릿 (t0 … P1/2−1) 을 tag 마다 비교
     (a) 읽은 값 (전 clip): 위치 차 중앙·p99 (px), logit 차, 판정 일치율
     (b) 토큰 (검사 clip 2 개 = 출력 번호 0, n//2), rel = mean|a−b| / mean|b|:
         enc_zc      z_ctx(P1) vs z_ctx(P2)                  encoder 가 창 길이를 타는가
         same_input  p_P1(z_ctx P1) vs p_P2(z_ctx P1)        **입력이 같고 창 길이만 다른** predictor 차
         floor       같은 창·같은 z_ctx 를 배치 2 vs 1 로     predictor 의 bf16 잡음 바닥
     prefix (block-causal) 통과 규칙: same_input ≤ max(2·floor, 2⁻⁸) 이고, **두 창에서 같은 배치 크기로 계산된 clip** (`bsz_C*_P*`,
     10 개 이상일 때) 에서 (a) 위치 차 중앙 ≤ 1 px · 판정 일치 ≥ 99 % 도. 배치 크기가 다른 clip (재개 · 마지막 배치 · `--base-bs`)
     은 배치 잡음이 섞여 참고로만 싣는다. default (full attention) 는 미래 슬롯이 서로 보므로 달라지는 게 정상 — 대조로만 싣는다

검증 (2026-09-25, vll5 GPU 3 한 장 — 다른 job 과 공유, 수치는 validation.json · check_loader.json 에서 옮겼다. 기본 설정 = SDPA 캐스트)
----------------------------------------------------------------------------------------------------------------------
  ① 로더 동치 (`--check-loader`, 검사 clip 2 개, C16/P16 · C16/P32): final 6 tag 전부 `build_from_config(predictor_checkpoint)`
     와 **max|Δ| = 0** (ariel_ep43 prefix 포함), release 는 model.pth predictor 와도 **0**. curves 22 ckpt 는 전부 같은 키 160 개,
     ariel 넷은 kind prefix · n_registers 0 (CPU 로 확인)
  ② release 동치 (내장): T32 · T48 둘 다 max|Δ| 0, bitwise True
  ③ 창 독립 (`--sample 48`, bs 5 공통), C16: P16 vs P32, t0–t7:
       ariel_ep43 (prefix)  enc z_ctx 0 (비트 동일) · same_input 6.52e-3 (바닥 7.12e-3) · 읽은 값 중앙 0.16 px · p99 0.57 · 일치 100 %  → PASS
       release (default)    same_input 6.71e-2 (바닥 9.33e-3, 7 배) · 읽은 값 중앙 2.93 px · p99 10.1 · 일치 98.4 %  (대조: full attention)
       post-FT 넷 (default)  읽은 값 중앙 3.7~7.9 px · 일치 92.4~95.8 %
     C8: P8 vs P16 (`--windows 8x8 8x16 --base-bs`, 16 clip): ariel same_input 6.40e-3 (바닥 7.18e-3) → PASS · release 1.03e-1
  ④ 정본 대조 (`--validate`, 48 clip = 시나리오마다 6, 기대: 중앙 ~0.8 px · recall ±1 pt → 규칙 중앙 ≤ 1.5 px · |Δrecall| ≤ 1 pt)
       C16/P16 (bs 5 vs 정본 8)  중앙 p 0.24 · p@h 0.09 · p@z 0.09 · z 0.94 · h 0.79 px; recall 차 0 · −0.26 · 0 · 0 · 0 pt → PASS
       C16/P32 (bs 5 = 정본 5)   전부 중앙 0.00 px (p99 1.6~4.2); recall 차 0 · −0.26 · 0 · 0 · 0 pt → PASS
       `--base-bs` (16 clip): C16/P16 전부 중앙 0.00 · p99 0.00 (비트 재현), C16/P32 중앙 0.00 (마지막 배치 1 clip 만 다름)
       C8 `--base-bs` (bs 16 · 10, 16 clip): 전부 중앙 0.00. C8/P16 의 p recall −1.56 pt (2/128 칸) 만 규칙 밖 — 두 칸 모두 마지막
       부분 배치 (6 clip) 의 문턱 근처 (logit 0.71~0.73 vs 정본 0.876, 문턱 0.831). 앞 10 clip (꽉 찬 배치) 은 비트 동일.
       같은 C8 을 bs 2 로 돌리면 p 중앙 2.3~2.6 px → **배치 크기 잡음이다** (같은 배치 크기면 옆 clip 이 달라도 같은 값)
  ⑤ 재개: 2 rank (한 장에 둘, bs 2) 로 돌리다 C16/P16 중간에 kill (done 32, 값만 쓰고 표시 안 된 2 개는 다시 계산) → 1 rank · bs 5 로
     이어서 끝냄. 정본 대조 전부 PASS (중앙 0.00~0.98 px, |Δrecall| ≤ 0.78 pt), ariel 창 독립 PASS (같은 배치 15 clip 중앙 0.17 px · 100 %;
     배치 크기가 섞인 전체는 1.14 px · 98.4 % — 그래서 규칙을 같은 배치 clip 으로 잰다)
  시간 (공유 GPU, rank 하나, bs 5, final 6 tag, 48 clip 평균 · 첫 배치 포함): C16/P16 0.28~0.50 s/clip · C16/P32 0.73~0.77 s/clip
     (실행마다 옆 job 부하로 흔들린다), 빌드 창 총합마다 41~70 s, 최대 메모리 3.92 GB (curves 는 predictor 17 개 × ~45 MB 더).
     predictor 하나 추가 = 창당 0.019~0.023 s/clip (predictor + 자 셋, bs 5 마이크로 벤치)
  mass3 방향 점검 (48 clip): h 자를 h 에 건 질량 t0–t6 0.36~0.54 (균등 0.035 의 10~15 배) — x·y 를 바꿔 끼웠다면 균등 근처였을 것.
     p@h release C16/P32 는 0.61 (t0) → 0.035 (t15) 로 균등까지 내려간다 (정본 "h 자 top1 이 t8 뒤 무너진다" 와 같은 방향)
  디버그: `TE_ONE_GPU=1` 이면 rank 전부가 cuda:0 을 쓴다 (한 장에서 멀티 rank 경로 점검용, ⑤)

재현
----
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  CUDA_VISIBLE_DEVICES=3 $P z_research/scripts/analysis/te_v3_readout.py --preset final --sample 48 --check-loader      # ①
  CUDA_VISIBLE_DEVICES=3 $P z_research/scripts/analysis/te_v3_readout.py --preset final --sample 48 --gpus 1 --validate # ②③④
  CUDA_VISIBLE_DEVICES=3 $P z_research/scripts/analysis/te_v3_readout.py --preset final --base-bs --sample 16 --gpus 1 --validate --name final_basebs
  CUDA_VISIBLE_DEVICES=3 $P z_research/scripts/analysis/te_v3_readout.py --preset final --windows 8x8 8x16 --base-bs --sample 16 --gpus 1 --validate --name final_c8basebs
  CUDA_VISIBLE_DEVICES=3 TE_ONE_GPU=1 $P z_research/scripts/analysis/te_v3_readout.py --preset final --sample 48 --gpus 2 --bs 2 \
      --out /data2/local_datasets/world/world_analysis/cache/training_effects/v3readout_final_sample48_resumetest   # ⑤ (중간에 kill 후 --gpus 1 로 다시)
  산출물: cache/training_effects/v3readout_final_sample48/{validation.json, check_loader.json, meta.json, readings.npz}
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))
import rollout3_window_readout as W                                        # noqa: E402  정의의 정본
from rollout3_paths import PRES as READOUT_DIR, WIN as REF_WIN, fingerprint  # noqa: E402

CACHE = Path("/data2/local_datasets/world/world_analysis/cache/training_effects")
EXPECT_FP = "dcd24d8a8a47"
COLS = ("x", "y", "logit", "top1", "mass3")
SPLIT, S, D, RES, CELL = W.SPLIT, W.S, W.D, W.RES, W.CELL
TR = ROOT / "z_training"
RELEASE = TR / "runs/release_vith/latest.pt"
BF16_EPS = 2.0 ** -8


def _ck(fam, e):
    return {"v11": TR / f"runs/v11_postft/e{e}.pt", "pv1": TR / f"runs/predictor_v1_postft/e{e}.pt",
            "ip1": TR / f"runs/intphys1_postft/e{e}.pt", "ip2": TR / f"runs/intphys2_postft/e{e}.pt"}[fam]


# Ariel (2026-09-27 수령). ⚠️ 옛 ariel_ep{5,19,30,41,43} (block_causal_future_only_1e_5_ep_{N}) 는 2026-09-27 에 파일이
# 지워져 뺐다 — ep43 을 ep45 로 바꿔 부르지 않는다. kind=ar 팔 (ar_future_1e_5) 은 readout 에서 거부한다 (refuse_ar).
ARIEL_PREFIX_EP45 = ("ariel_prefix_ep45", TR / "ariel/block_causal_future_only_1e_5/latest.pt")   # kind prefix (팔 B)


def _tags(spec, extra=()):
    out = [("release", RELEASE)]
    for fam, eps in spec:
        for e in eps:
            out.append((f"{fam}_e{e}", _ck(fam, e)))
    return out + list(extra)


PRESETS = {
    # 2026-09-27: final 의 ariel_ep43 (지워짐) → ariel_prefix_ep45 (같은 run 의 마지막 epoch, 다른 파일)
    "final": _tags([("v11", [10]), ("ip1", [40]), ("ip2", [80]), ("pv1", [15])], extra=[ARIEL_PREFIX_EP45]),
    # 2026-09-27: ariel_ep{5,19,30,43} 은 대체 곡선이 없어 뺐다
    "curves": _tags([("v11", [1, 2, 3, 5, 10]), ("pv1", [1, 3, 5, 10, 15]), ("ip1", [5, 10, 20, 40]),
                     ("ip2", [10, 40, 80])]),
}


def refuse_ar(tag, kind):
    """kind=ar (자기회귀) predictor 거부 (2026-09-27). 이 readout 의 p 는 release 규약 predictor 출력이다."""
    if kind == "ar":
        raise NotImplementedError(
            f"{tag}: kind=ar (자기회귀) predictor 의 readout 은 정의하지 않았다 — AR 의 'p' 는 블록별 LN(target_encoder) "
            f"공간의 rollout 이라 자 (readout) 공간의 p 와 뜻이 다르다 (bf16 · autocast 없는 이 스크립트에서는 SDPA 도 죽는다). "
            f"채점은 표준 하네스 (run.sh surprise_c16t32 · intphys1_sliding, 모델 vith_ariel_ar_ep18) 로 한다")


# ----------------------------------------------------------------------------- 공통
def keys_for(tags, zh=True):
    ks = (["z", "h"] if zh else [])
    for t in tags:
        ks += [f"p_{t}", f"p@h_{t}", f"p@z_{t}"]
    return ks


def load_sd(path):
    """predictor-only ckpt → (kind, kind_kwargs, state_dict). mmap 으로 열어 opt 같은 나머지는 안 읽는다."""
    try:
        st = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:                                                     # 옛 직렬화면 mmap 불가
        st = torch.load(path, map_location="cpu", weights_only=False)
    from analysis import predictors as PV
    from analysis.intphys2.model import _clean_backbone_key
    arch = st.get("arch") or {}
    kind = str(arch.get("kind") or "default")
    refuse_ar(str(path), kind)                                            # 2026-09-27: 모델을 짓기 전에 거부
    kk = {k: arch[k] for k in PV.KIND_ONLY_KEYS if k in arch}
    return kind, kk, _clean_backbone_key(st["predictor"])


def build_pred(bundle, entry):
    """`build_from_config` 의 predictor 경로와 같은 인자로 짓고 strict 로 싣는다 (kind 는 ckpt 가 선언)."""
    from analysis.intphys2.model import _build_predictor
    kind, kk, sd = entry
    pc = W.MODEL["predictor"]
    pred = _build_predictor(
        img_size=W.MODEL["img_size"], patch_size=W.MODEL["patch_size"], tubelet_size=W.MODEL["tubelet_size"],
        num_frames=bundle.num_frames, encoder_embed_dim=bundle.embed_dim,
        predictor_embed_dim=pc["embed_dim"], predictor_depth=pc["depth"], predictor_num_heads=pc["num_heads"],
        num_mask_tokens=pc["num_mask_tokens"], use_rope=W.MODEL["use_rope"], uniform_power=W.MODEL["uniform_power"],
        kind=kind, kind_kwargs=kk)
    pred.load_state_dict(sd, strict=True)
    return pred.to(bundle.device, bundle.dtype).eval().requires_grad_(False)


class _SDPACast:
    """`src/models/utils/modules.py` 의 `F` 대신 끼우는 얇은 대리자 — prefix 를 **autocast 없는 bf16** 으로 돌리기 위한 우회.
    (`te_v11_readout.py::_SDPACast` 와 같은 동작 — TrainingEffects 두 readout 이 prefix 를 같은 수치로 돌리게 맞췄다.)
    ACRoPEAttention 은 RoPE 위치가 float (`1.0 * frame_ids`) 라 q·k 가 fp32 로 올라가고 v 는 bf16 → PrefixSpec 분기의 SDPA 가
    dtype 오류로 죽는다 (2026-09-25 실측). **q·k 를 v 의 dtype 으로 내리기만** 한다 (autocast 가 SDPA 에 하는 캐스트와 같다).
    dtype 이 같으면 no-op → encoder · default predictor 는 비트 단위로 그대로 (release 동치 ② 가 그 확인). 공유 코드는 안 고친다."""
    n_cast = 0

    def __getattr__(self, name):
        return getattr(F, name)

    @staticmethod
    def scaled_dot_product_attention(q, k, v, *args, **kw):
        if q.dtype != v.dtype or k.dtype != v.dtype:
            _SDPACast.n_cast += 1
            q, k = q.to(v.dtype), k.to(v.dtype)
        return F.scaled_dot_product_attention(q, k, v, *args, **kw)


def install_sdpa_cast():
    import src.models.utils.modules as MU
    if not isinstance(MU.F, _SDPACast):
        MU.F = _SDPACast()


def needs_autocast(kind, mode):
    """`--pred-autocast`: none (기본) = 전부 autocast 없는 bf16 (prefix 는 _SDPACast 로 q·k 만 bf16 으로) ·
    prefix = prefix 류만 autocast(bf16) (학습과 같은 방식, LayerNorm·RoPE 각이 fp32) · all = 전부."""
    return mode == "all" or (mode == "prefix" and kind not in ("default", "oneshot"))


def pfwd(pred, ac, zc, ci, ti):
    if ac:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = pred(zc, ci, ti, mask_index=0)
    else:
        o = pred(zc, ci, ti, mask_index=0)
    return o[-1] if isinstance(o, list) else o


def bs_for(a, total):
    """배치 크기. 기본 = **모든 창에 같은 값** (토큰 예산 // 가장 긴 창) — 창 사이 비교에서 배치 차이를 없앤다.
    `--base-bs` = 정본 (`rollout3_window_readout.py`) 처럼 창 총합마다 (토큰 예산 // 그 창) — 정본 readings 를 비트 수준으로 재현한다.
    ⚠️ bf16 읽은 값은 **배치 크기**에 따라 흔들린다 (커널 선택; 같은 배치 크기면 옆 clip 이 달라도 같다 — 2026-09-25 실측)."""
    if a.bs:
        return a.bs
    ntok = (total if a.base_bs else max(c + p for c, p in a.win)) // 2 * S
    return max(1, min(a.max_bs, a.token_budget // ntok))


def gt_future_px(L, rid, p):
    """(n, P/2, 2) 미래 튜블릿의 GT (두 프레임 평균, 288 px 절대 좌표)."""
    X = (L[rid] + 1.0) * RES                                               # 정규화 → px
    return X[:, SPLIT:SPLIT + p].reshape(len(rid), p // 2, 2, 2).mean(2)


def mass3(amap, gt):
    """amap (M, 256), gt (M, 2) px → GT 칸 3×3 attention 질량 (M,)."""
    cx, cy = torch.floor(gt[:, 0] / CELL), torch.floor(gt[:, 1] / CELL)
    ar = torch.arange(16, device=amap.device, dtype=gt.dtype)
    rr = (ar[None, :, None] - cy[:, None, None]).abs() <= 1                # (M, 16, 1) 행 = y
    cc = (ar[None, None, :] - cx[:, None, None]).abs() <= 1                # (M, 1, 16) 열 = x
    return (amap.reshape(-1, 16, 16) * (rr & cc)).sum((1, 2))


def read(head, tok, gt):
    """tok (M, 256, D) fp32, gt (M, 2) → (M, 5) = x, y, logit, top1, mass3."""
    xy, pres, amap = head(tok)
    return torch.cat([xy, pres[:, None], amap.max(-1).values[:, None], mass3(amap, gt)[:, None]], 1)


def rel(a, b):
    a, b = a.astype(np.float32), b.astype(np.float32)
    return float(np.abs(a - b).mean() / max(np.abs(b).mean(), 1e-12)), float(np.abs(a - b).max())


class _Sub:
    """출력 번호 i → index.csv 행 번호로 clip 을 읽는다 (W.prefetch 가 ds.clip(i) 를 부른다)."""
    def __init__(self, ds, rid):
        self.ds, self.rid = ds, rid

    def clip(self, i):
        return self.ds.clip(int(self.rid[i]))


def part(parts, c, p, k):
    return parts / f"C{c}_P{p}__{k}.npy"


def donep(parts, c, p, g):
    return parts / f"done_C{c}_P{p}__{g}.npy"


def bszp(parts, c, p):
    """clip 마다 p 를 계산한 **실제 배치 크기** (0 = 아직). 재개·마지막 배치로 섞이는 배치 잡음을 가려내는 데 쓴다 (③)."""
    return parts / f"bsz_C{c}_P{p}.npy"


# ----------------------------------------------------------------------------- 워커
def _worker(rank, world, a, plan):
    dev_i = 0 if os.environ.get("TE_ONE_GPU") == "1" else rank           # 디버그: 여러 rank 를 한 장에
    torch.cuda.set_device(dev_i)
    dev = torch.device("cuda", dev_i)
    torch.set_num_threads(a.threads)
    from analysis.intphys2.model import build_from_config
    from analysis.intphys2.surprise import _context_target_indices
    from evals.world_model_analysis.data import WMADataset
    import decord  # noqa: F401  (WMADataset 이 쓴다)
    install_sdpa_cast()

    parts = Path(plan["parts"])
    rid = np.asarray(plan["rid"])
    n = len(rid)
    tags, zh = plan["tags"], plan["zh"]
    groups = (["zh"] if zh else []) + tags
    chk_ids = plan["chk_ids"]
    R = W.readout_cls()
    heads = {}
    for r in ("p", "h", "z"):
        h = R("attn").to(dev).eval()
        h.load_state_dict(torch.load(READOUT_DIR / r / "readout_attn.pt", map_location="cpu"))
        heads[r] = h.requires_grad_(False)
    _, L, _ = W.labels()
    ln = lambda x: F.layer_norm(x, (x.size(-1),))
    pool = ThreadPoolExecutor(max_workers=a.workers)
    sds = {t: load_sd(ck) for t, ck in plan["ckpts"].items()}
    kinds = {t: sds[t][0] for t in tags}
    ac = {t: needs_autocast(kinds[t], a.pred_autocast) for t in tags}
    chk_tags = [t for t in tags if kinds[t] != "default"] + [t for t in tags if kinds[t] == "default"][:1]
    timing = {"rank": rank, "device": str(dev), "windows": {}, "builds": {}}

    by_total = {}
    for c, p in a.win:
        by_total.setdefault(c + p, []).append((c, p))

    def pending(c, p):
        dn = {g: np.load(donep(parts, c, p, g), mmap_mode="r") for g in groups}
        return [i for i in range(n) if any(dn[g][i] == 0 for g in groups)]

    with torch.inference_mode():
        for total, wins in sorted(by_total.items()):
            work = {(c, p): pending(c, p) for c, p in wins}
            mine_any = any(len(v[rank::world]) for v in work.values())
            chk_any = rank == 0 and any(len(v) for v in work.values())
            if not (mine_any or chk_any):
                continue
            t_b = time.time()
            bs = bs_for(a, total)
            bundle = build_from_config({**W.MODEL, "window_size": total}, dev)
            preds = {t: build_pred(bundle, sds[t]) for t in tags}
            timing["builds"][str(total)] = round(time.time() - t_b, 1)
            if rank == 0:
                print(f"  [r0] 창 총합 {total}f 빌드 {time.time()-t_b:.0f}s · predictor {len(preds)} 개 · bs {bs}", flush=True)

            def run_p(clips, c, B):
                ci, ti = _context_target_indices(ctx_frames=c, tgt_frames=total - c, tubelet_size=2,
                                                 spatial_tokens=S, batch_size=B, device=dev)
                zc = bundle.context_encoder(clips, masks=[ci])
                zc = zc[-1] if isinstance(zc, list) else zc
                return zc, ci, ti

            # ② release 동치 (총합마다, rank 0): model.pth predictor vs release latest.pt
            if rank == 0 and chk_any:
                c0, p0 = wins[0]
                ds0 = _Sub(WMADataset(dict(data={**W.DATA, "frames_start": SPLIT - c0, "n_frames": total},
                                           model={**W.MODEL, "window_size": total}, features={"cache_dir": "/tmp"},
                                           surprise={})), rid)
                clips = torch.stack([ds0.clip(i) for i in chk_ids]).to(dev, dtype=bundle.dtype)
                zc, ci, ti = run_p(clips, c0, len(chk_ids))
                ref = bundle.predictor(zc, ci, ti, mask_index=0)
                rp = preds["release"] if "release" in preds else build_pred(bundle, load_sd(RELEASE))
                got = pfwd(rp, a.pred_autocast == "all", zc, ci, ti)
                (parts / f"chk_release_equiv_T{total}.json").write_text(json.dumps(dict(
                    window=[c0, p0], clips=chk_ids, max_abs=float((ref.float() - got.float()).abs().max()),
                    bitwise_equal=bool(torch.equal(ref, got)))))

            for c, p in wins:
                todo = work[(c, p)]
                if not todo:
                    continue
                tc, tp = c // 2, p // 2
                ds = _Sub(WMADataset(dict(data={**W.DATA, "frames_start": SPLIT - c, "n_frames": total},
                                          model={**W.MODEL, "window_size": total}, features={"cache_dir": "/tmp"},
                                          surprise={})), rid)
                GT = torch.from_numpy(gt_future_px(L, rid, p).astype(np.float32))       # (n, tp, 2)
                # ③(b) 토큰 검사 (rank 0, 검사 clip 2 개). 저장: z_ctx · p (배치 2) · p (같은 z_ctx 를 한 개씩 = predictor 잡음 바닥)
                #      · 같은 C 의 **앞 창에서 저장한 z_ctx** 를 이 창의 predictor 에 넣은 p (= 입력 동일, 창 길이만 다름)
                if rank == 0:
                    Bc = len(chk_ids)
                    clips = torch.stack([ds.clip(i) for i in chk_ids]).to(dev, dtype=bundle.dtype)
                    zc, ci, ti = run_p(clips, c, Bc)
                    np.save(parts / f"chk_zc_C{c}_P{p}.npy", zc.float().cpu().numpy())
                    c1, t1 = ci[:1], ti[:1]
                    ov = min(tp, 8)
                    prev = sorted(pp for cc, pp in a.win if cc == c and pp < p
                                  and (parts / f"chk_zc_C{c}_P{pp}.npy").exists())
                    casts = {}
                    for t in tags:                                   # SDPA 캐스트 횟수 (default 는 0 이어야 한다)
                        n0 = _SDPACast.n_cast
                        pfwd(preds[t], ac[t], zc, ci, ti)
                        casts[t] = _SDPACast.n_cast - n0
                    (parts / f"chk_sdpa_casts_C{c}_P{p}.json").write_text(json.dumps(casts))
                    for t in chk_tags:
                        o = pfwd(preds[t], ac[t], zc, ci, ti).reshape(Bc, tp, S, D)[:, :ov]
                        o1 = torch.cat([pfwd(preds[t], ac[t], zc[j:j + 1], c1, t1).reshape(1, tp, S, D)[:, :ov]
                                        for j in range(Bc)])
                        np.save(parts / f"chk_tok_C{c}_P{p}__{t}.npy", o.to(torch.float16).cpu().numpy())
                        np.save(parts / f"chk_tok1_C{c}_P{p}__{t}.npy", o1.to(torch.float16).cpu().numpy())
                        for pp in prev:
                            zo = torch.from_numpy(np.load(parts / f"chk_zc_C{c}_P{pp}.npy")).to(dev, bundle.dtype)
                            ox = pfwd(preds[t], ac[t], zo, ci, ti).reshape(Bc, tp, S, D)[:, :pp // 2]
                            np.save(parts / f"chk_tokx_C{c}_P{p}_zfromP{pp}__{t}.npy", ox.to(torch.float16).cpu().numpy())
                mine = todo[rank::world]
                if not mine:
                    continue
                keys = keys_for(tags, zh)
                mm = {k: np.lib.format.open_memmap(part(parts, c, p, k), mode="r+") for k in keys}
                dn = {g: np.lib.format.open_memmap(donep(parts, c, p, g), mode="r+") for g in groups}
                bz = np.lib.format.open_memmap(bszp(parts, c, p), mode="r+")
                torch.cuda.synchronize(dev)
                t0, done_c = time.time(), 0
                for ids, clips in W.prefetch(ds, mine, bs, pool):
                    B = clips.size(0)
                    ids = np.asarray(ids)
                    clips = clips.to(dev, dtype=bundle.dtype, non_blocking=True)
                    gt = GT[torch.from_numpy(ids)].reshape(B * tp, 2).to(dev)
                    need = {g: ids[dn[g][ids] == 0] for g in groups}
                    rows = {g: dn[g][ids] == 0 for g in groups}
                    ptags = [t for t in tags if len(need[t])]
                    if ptags:
                        zc, ci, ti = run_p(clips, c, B)
                        for t in ptags:
                            o = pfwd(preds[t], ac[t], zc, ci, ti)
                            tok = o.to(torch.float16).float().reshape(B * tp, S, D)   # 정본: fp16 경유
                            for hr, k in (("p", f"p_{t}"), ("h", f"p@h_{t}"), ("z", f"p@z_{t}")):
                                v = read(heads[hr], tok, gt).reshape(B, tp, 5).cpu().numpy()
                                mm[k][ids[rows[t]]] = v[rows[t]]
                        bz[ids[np.any([rows[t] for t in ptags], 0)]] = B
                        del zc
                    if zh and len(need["zh"]):
                        for rep, mod in (("z", "context_encoder"), ("h", "target_encoder")):
                            fo = getattr(bundle, mod)(clips)
                            fo = fo[-1] if isinstance(fo, list) else fo
                            tok = ln(fo.float()).reshape(B, -1, S, D)[:, tc:].reshape(B * tp, S, D).half().float()
                            v = read(heads[rep], tok, gt).reshape(B, tp, 5).cpu().numpy()
                            mm[rep][ids[rows["zh"]]] = v[rows["zh"]]
                            del fo, tok
                    for k in mm:                                             # 값 먼저, 완료 표시는 그 뒤
                        mm[k].flush()
                    bz.flush()
                    for g in groups:
                        if len(need[g]):
                            dn[g][need[g]] = 1
                            dn[g].flush()
                    done_c += B
                    if rank == 0 and (done_c // B) % a.log_every == 0:
                        el = time.time() - t0
                        print(f"    [C{c}_P{p}] {done_c}/{len(mine)} clip (rank 0)  {el:.0f}s  "
                              f"{el/done_c:.3f} s/clip", flush=True)
                torch.cuda.synchronize(dev)
                el = time.time() - t0
                timing["windows"][f"C{c}_P{p}"] = dict(n_clip=done_c, sec=round(el, 2),
                                                       sec_per_clip=round(el / max(done_c, 1), 4), bs=bs)
                if rank == 0:
                    print(f"    [C{c}_P{p}] 끝 {done_c} clip  {el:.1f}s  {el/max(done_c,1):.3f} s/clip (rank 0)", flush=True)
            del bundle, preds
            torch.cuda.empty_cache()
    timing["peak_mem_gb"] = round(torch.cuda.max_memory_allocated(dev) / 2**30, 2)
    (parts / f"timing_r{rank}.json").write_text(json.dumps(timing, indent=1))


# ----------------------------------------------------------------------------- 로더 동치 ①
def check_loader(a, tags, rid, chk_ids):
    """tag 마다 `load_pred` 경로 vs `build_from_config({..., predictor_checkpoint})` 의 p (같은 z_ctx 입력).
    release 는 추가로 model.pth predictor (predictor_checkpoint 없음) 와 비교한다. GPU 1 장, 검사 clip 2 개."""
    import analysis.intphys2.model as M
    from analysis.intphys2.surprise import _context_target_indices
    from evals.world_model_analysis.data import WMADataset
    dev = torch.device("cuda", 0)
    install_sdpa_cast()
    _orig = M._load_checkpoint
    _memo = {}

    def _load_memo(path):                      # model.pth (10 GB) 를 tag 마다 다시 읽지 않는다 — 내용은 같다
        if path not in _memo:
            try:
                _memo[path] = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            except Exception:
                _memo[path] = _orig(path)
        return _memo[path]
    M._load_checkpoint = _load_memo
    res = {}
    with torch.inference_mode():
        for c, p in a.win:
            total = c + p
            t0 = time.time()
            bundle = M.build_from_config({**W.MODEL, "window_size": total}, dev)
            ds = _Sub(WMADataset(dict(data={**W.DATA, "frames_start": SPLIT - c, "n_frames": total},
                                      model={**W.MODEL, "window_size": total}, features={"cache_dir": "/tmp"},
                                      surprise={})), rid)
            clips = torch.stack([ds.clip(i) for i in chk_ids]).to(dev, dtype=bundle.dtype)
            ci, ti = _context_target_indices(ctx_frames=c, tgt_frames=p, tubelet_size=2, spatial_tokens=S,
                                             batch_size=len(chk_ids), device=dev)
            zc = bundle.context_encoder(clips, masks=[ci])
            ref_rel = pfwd(bundle.predictor, a.pred_autocast == "all", zc, ci, ti)
            print(f"  [check-loader] C{c}/P{p} 빌드 {time.time()-t0:.0f}s", flush=True)
            for t, ck in tags:
                ent = load_sd(ck)
                acx = needs_autocast(ent[0], a.pred_autocast)
                mine = pfwd(build_pred(bundle, ent), acx, zc, ci, ti)
                t1 = time.time()
                rb = M.build_from_config({**W.MODEL, "window_size": total, "predictor_checkpoint": str(ck)},
                                         torch.device("cpu"))
                rp = rb.predictor.to(dev)
                ref = pfwd(rp, acx, zc, ci, ti)
                row = dict(kind=ent[0], kind_kwargs=ent[1], ckpt=str(ck), autocast_bf16=acx,
                           built_class=type(rp).__name__,
                           vs_build_from_config_max_abs=float((ref.float() - mine.float()).abs().max()),
                           vs_build_from_config_bitwise=bool(torch.equal(ref, mine)))
                if t == "release":
                    row["vs_model_pth_predictor_max_abs"] = float((ref_rel.float() - mine.float()).abs().max())
                    row["vs_model_pth_predictor_bitwise"] = bool(torch.equal(ref_rel, mine))
                else:
                    row["rel_diff_vs_release"] = rel(mine.float().cpu().numpy(), ref_rel.float().cpu().numpy())[0]
                res[f"C{c}_P{p}/{t}"] = row
                print(f"    {t:12s} kind={ent[0]:8s} vs build_from_config max|Δ|={row['vs_build_from_config_max_abs']:.3g}"
                      + (f"  vs model.pth predictor max|Δ|={row['vs_model_pth_predictor_max_abs']:.3g}" if t == "release" else "")
                      + f"  ({time.time()-t1:.0f}s)", flush=True)
                del rb, rp
            del bundle
            torch.cuda.empty_cache()
    M._load_checkpoint = _orig
    return res


# ----------------------------------------------------------------------------- 검사 · 조립
def window_checks(parts, out, tags, kinds, win, thr_p):
    """③ 같은 C, 다른 P 의 겹치는 튜블릿 (P1 < P2, 겹침 = P1/2 튜블릿).
    (a) 읽은 값 (전 clip, 두 창 모두 같은 배치 구성): 위치 차 · logit 차 · 판정 일치
    (b) 토큰 (검사 clip 2 개), 상대 차 rel = mean|a−b| / mean|b|:
        enc_zc      z_ctx(P1) vs z_ctx(P2)                     — encoder 가 창 길이를 타는가
        end2end     p_P1(z_ctx P1) vs p_P2(z_ctx P2)
        same_input  p_P1(z_ctx P1) vs p_P2(z_ctx P1)           — **predictor 만의 창 길이 효과** (입력 동일)
        floor       같은 창·같은 z_ctx 를 배치 2 vs 1 로 (predictor bf16 잡음 바닥, 두 창 중 큰 값)
    prefix (block-causal) 통과 규칙: same_input ≤ max(2·floor, 2⁻⁸) 이고 (a) 위치 차 중앙 ≤ 1 px · 판정 일치 ≥ 99 %.
    default (full attention) 는 미래가 미래를 보므로 달라지는 것이 정상 — 대조로만 싣는다."""
    res = {}
    for c in sorted({c for c, _ in win}):
        ps = sorted(p for cc, p in win if cc == c)
        for i, p1 in enumerate(ps):
            for p2 in ps[i + 1:]:
                ov = p1 // 2
                b1, b2 = np.load(bszp(parts, c, p1)), np.load(bszp(parts, c, p2))
                same = (b1 == b2) & (b1 > 0)                 # 두 창에서 같은 배치 크기로 계산된 clip
                z1f, z2f = parts / f"chk_zc_C{c}_P{p1}.npy", parts / f"chk_zc_C{c}_P{p2}.npy"
                enc = rel(np.load(z1f), np.load(z2f)) if z1f.exists() and z2f.exists() else None
                for t in tags:
                    A, B = out[f"C{c}_P{p1}_p_{t}"][:, :ov], out[f"C{c}_P{p2}_p_{t}"][:, :ov]
                    d = np.linalg.norm((A[..., :2] - B[..., :2]) * RES, axis=-1)
                    agree = float(((A[..., 2] > thr_p) == (B[..., 2] > thr_p)).mean())
                    ds, ags = (d[same], float(((A[same][..., 2] > thr_p) == (B[same][..., 2] > thr_p)).mean())) \
                        if same.any() else (None, None)
                    r = dict(kind=kinds[t], overlap_tubelets=ov, n_clip_same_batch_size=int(same.sum()),
                             readings_same_bs=None if ds is None else dict(
                                 pos_diff_median=float(np.median(ds)), pos_diff_p99=float(np.percentile(ds, 99)),
                                 present_agree=ags),
                             readings_pos_diff_px=dict(median=float(np.median(d)), p99=float(np.percentile(d, 99)),
                                                       max=float(d.max())),
                             readings_logit_absdiff_median=float(np.median(np.abs(A[..., 2] - B[..., 2]))),
                             readings_present_agree=agree)
                    f1, f2 = parts / f"chk_tok_C{c}_P{p1}__{t}.npy", parts / f"chk_tok_C{c}_P{p2}__{t}.npy"
                    fx = parts / f"chk_tokx_C{c}_P{p2}_zfromP{p1}__{t}.npy"
                    if f1.exists() and f2.exists() and fx.exists():
                        a1, a2, ax = np.load(f1)[:, :ov], np.load(f2)[:, :ov], np.load(fx)[:, :ov]
                        fl = max(rel(np.load(parts / f"chk_tok1_C{c}_P{pp}__{t}.npy")[:, :ov],
                                     np.load(parts / f"chk_tok_C{c}_P{pp}__{t}.npy")[:, :ov])[0] for pp in (p1, p2))
                        si, si_max = rel(ax, a1)
                        r.update(token_rel_enc_zc=enc[0] if enc else None, token_rel_end2end=rel(a2, a1)[0],
                                 token_rel_same_input=si, token_maxabs_same_input=si_max, token_rel_floor=fl)
                    if kinds[t] not in ("default", "oneshot"):
                        ok_tok = ("token_rel_same_input" in r
                                  and r["token_rel_same_input"] <= max(2 * r["token_rel_floor"], BF16_EPS))
                        if same.sum() >= 10:
                            r["pass_rule"] = ("token same_input <= max(2*floor, 2^-8) and, on clips computed with the same "
                                              "batch size in both windows, readings median pos diff <= 1 px and present agree >= 0.99")
                            r["passed"] = bool(ok_tok and np.median(ds) <= 1.0 and ags >= 0.99)
                        else:     # 같은 배치 크기로 계산된 clip 이 거의 없다 (재개·--base-bs) → 읽은 값엔 배치 잡음 → 토큰만
                            r["pass_rule"] = "token same_input <= max(2*floor, 2^-8) (<10 clips share batch size: readings informational)"
                            r["passed"] = bool(ok_tok)
                    res[f"C{c}: P{p1} vs P{p2} / {t}"] = r
    return res


def validate_ref(out, ids, win, thr, fp):
    """정본 `RollOutV3/exp_results/windows/readings.npz` 와 같은 video_id 로 대조 (release tag)."""
    ref = np.load(REF_WIN / "readings.npz", allow_pickle=True)
    if str(ref["decoder_fp"]) != fp:
        return {"skipped": f"정본 지문 {ref['decoder_fp']} != {fp}"}
    pos = {v: i for i, v in enumerate(ref["video_id"])}
    j = np.array([pos[v] for v in ids])
    pairs = [("p_release", "p", "p"), ("p@h_release", "p@h", "h"), ("p@z_release", "p@z", "z"),
             ("z", "z", "z"), ("h", "h", "h")]
    res = {}
    for c, p in win:
        if f"C{c}_P{p}_p" not in ref.files:
            continue
        for ours, theirs, ruler in pairs:
            k = f"C{c}_P{p}_{ours}"
            if k not in out:
                continue
            A, B = out[k], ref[f"C{c}_P{p}_{theirs}"][j]
            d = np.linalg.norm((A[..., :2] - B[..., :2]) * RES, axis=-1)
            ra, rb = 100 * (A[..., 2] > thr[ruler]).mean(), 100 * (B[..., 2] > thr[ruler]).mean()
            res[f"C{c}_P{p}/{ours} vs ref {theirs}"] = dict(
                n_clip=len(j), pos_diff_px=dict(median=float(np.median(d)), p90=float(np.percentile(d, 90)),
                                                p99=float(np.percentile(d, 99)), max=float(d.max())),
                logit_absdiff_median=float(np.median(np.abs(A[..., 2] - B[..., 2]))),
                top1_absdiff_median=float(np.median(np.abs(A[..., 3] - B[..., 3]))),
                recall_ours=float(ra), recall_ref=float(rb), recall_diff_pt=float(ra - rb),
                present_agree=float(((A[..., 2] > thr[ruler]) == (B[..., 2] > thr[ruler])).mean()),
                pass_rule="median pos diff <= 1.5 px and |recall diff| <= 1.0 pt",
                passed=bool(np.median(d) <= 1.5 and abs(ra - rb) <= 1.0))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None)
    ap.add_argument("--preds", nargs="*", default=None, metavar="tag=path",
                    help="predictor 목록 (쉼표 또는 공백 구분). --preset 대신")
    ap.add_argument("--name", default=None, help="출력 폴더 꼬리표 (기본 = preset 이름, --preds 면 custom)")
    ap.add_argument("--out", default=None, help="출력 폴더를 직접 (기본 CACHE/v3readout_<name>)")
    ap.add_argument("--windows", nargs="*", default=["16x16", "16x32"], metavar="CxP")
    ap.add_argument("--c8", action="store_true", help="C8/P8 · C8/P16 추가")
    ap.add_argument("--no-zh", action="store_true", help="z·h 를 안 읽는다 (predictor 무관 값)")
    ap.add_argument("--pred-autocast", choices=["none", "prefix", "all"], default="none",
                    help="none (기본) = 전부 autocast 없는 bf16 (prefix 는 SDPA 앞에서 q·k 만 bf16 으로, te_v11_readout 과 같다) · "
                         "prefix = prefix 류만 autocast(bf16) · all = 전부 (release 도 정본 규약에서 벗어난다)")
    ap.add_argument("--gpus", type=int, default=torch.cuda.device_count() or 1)
    ap.add_argument("--sample", type=int, default=0, help="시나리오마다 고르게 N clip (검증)")
    ap.add_argument("--limit", type=int, default=0, help="앞 N clip (배관 점검)")
    ap.add_argument("--token-budget", type=int, default=32768)
    ap.add_argument("--max-bs", type=int, default=16)
    ap.add_argument("--bs", type=int, default=0, help="배치 크기 직접 (기본 = 토큰 예산 // 가장 긴 창)")
    ap.add_argument("--base-bs", action="store_true",
                    help="정본처럼 창 총합마다 배치 크기 (토큰 예산 // 그 창) — 정본 readings 를 비트 수준 재현")
    ap.add_argument("--workers", type=int, default=6, help="PNG 읽기 스레드 (GPU 당)")
    ap.add_argument("--threads", type=int, default=4, help="torch CPU 스레드 (rank 당)")
    ap.add_argument("--log-every", type=int, default=20, help="rank 0 진행 출력 간격 (배치)")
    ap.add_argument("--validate", action="store_true", help="정본 readings.npz 와 대조 (release tag 필요)")
    ap.add_argument("--check-loader", action="store_true", help="로더 동치 검사만 하고 끝 (GPU 1 장)")
    ap.add_argument("--expect-fp", default=EXPECT_FP, help="자 지문. 'any' 면 검사 안 함")
    a = ap.parse_args()

    # ---- predictor 목록
    if a.preds:
        tags = []
        for s in ",".join(a.preds).split(","):
            if s.strip():
                t, pth = s.split("=", 1)
                tags.append((t.strip(), Path(pth.strip())))
        name = a.name or "custom"
    else:
        tags = list(PRESETS[a.preset or "final"])
        name = a.name or (a.preset or "final")
    assert len({t for t, _ in tags}) == len(tags), f"tag 중복: {[t for t, _ in tags]}"
    for t, pth in tags:
        assert Path(pth).exists(), f"{t}: ckpt 없음 {pth}"
        assert "@" not in t and "__" not in t, f"tag 에 '@'·'__' 금지: {t}"
        load_sd(pth)                                                      # 2026-09-27: kind=ar 는 폴더를 만들기 전에 거부 (mmap)
    win = [tuple(int(v) for v in w.lower().split("x")) for w in a.windows]
    if a.c8:
        win += [(8, 8), (8, 16)]
    win = sorted(set(win), key=lambda w: (w[0], w[1]))
    for c, p in win:
        assert c % 2 == 0 and p % 2 == 0 and SPLIT - c >= 0 and SPLIT + p <= W.NF, (c, p)
    a.win = win

    fp = fingerprint(READOUT_DIR)
    if a.expect_fp != "any" and fp != a.expect_fp:
        raise SystemExit(f"자 지문 {fp} != 기대 {a.expect_fp} ({READOUT_DIR}). R3_DECODER 를 확인할 것")

    # ---- clip 목록 (가능 clip, index 순서)
    rows, L, inf = W.labels()
    roll = [i for i, r in enumerate(rows) if r["role"] == "roll"]
    if a.sample:
        sc = np.array([rows[i]["scenario"] for i in roll])
        k = math.ceil(a.sample / len(set(sc)))
        pick = []
        for s in sorted(set(sc)):
            w_ = np.where(sc == s)[0]
            pick += list(w_[np.unique(np.linspace(0, len(w_) - 1, k).round().astype(int))])
        roll = [roll[i] for i in sorted(pick)]
    elif a.limit:
        roll = roll[:a.limit]
    rid = np.array(roll)
    n = len(rid)
    ids = [rows[i]["video_id"] for i in rid]
    chk_ids = sorted({0, n // 2})
    sfx = f"_sample{a.sample}" if a.sample else (f"_limit{a.limit}" if a.limit else "")
    out_dir = Path(a.out) if a.out else CACHE / f"v3readout_{name}{sfx}"
    assert "RollOutV3" not in str(out_dir.resolve()), "정본 폴더에는 쓰지 않는다"
    print(f"[data] 가능 clip {n} · 창 {win} · predictor {len(tags)} 개 {[t for t, _ in tags]} → {out_dir}", flush=True)

    if a.check_loader:
        out_dir.mkdir(parents=True, exist_ok=True)
        res = check_loader(a, tags, rid, chk_ids)
        f = out_dir / "check_loader.json"
        f.write_text(json.dumps(dict(clips=[ids[i] for i in chk_ids], windows=win, results=res), indent=1))
        ok = all(r["vs_build_from_config_max_abs"] == 0.0 for r in res.values()) and \
            all(r.get("vs_model_pth_predictor_max_abs", 0.0) == 0.0 for r in res.values())
        print(f"[check-loader] {'PASS' if ok else 'FAIL'} (max|Δ| 전부 0) → {f}", flush=True)
        return

    # ---- 부품 (재개 가능)
    parts = out_dir / "_parts"
    parts.mkdir(parents=True, exist_ok=True)
    sig_f = parts / "sig.json"
    vid_sha = hashlib.sha1("\n".join(ids).encode()).hexdigest()[:16]
    sig = dict(n=n, video_id_sha1=vid_sha, decoder_fp=fp, tags={t: str(p_) for t, p_ in tags}, zh_any=not a.no_zh,
               pred_autocast=a.pred_autocast)
    if sig_f.exists():
        old = json.loads(sig_f.read_text())
        if (old["n"], old["video_id_sha1"], old["decoder_fp"], old.get("pred_autocast")) != (n, vid_sha, fp, a.pred_autocast):
            raise SystemExit(f"{parts} 는 다른 clip 목록·자·autocast 로 만든 것이다 ({old['n']} clip, {old['decoder_fp']}, "
                             f"{old.get('pred_autocast')}). --out 을 바꿀 것")
        for t, p_ in sig["tags"].items():
            if t in old["tags"] and old["tags"][t] != p_:
                raise SystemExit(f"tag {t} 가 전에 다른 ckpt ({old['tags'][t]}) 였다. tag 이름을 바꿀 것")
        # 2026-09-27: 조립 (readings.npz) 은 지금 목록의 tag 만 싣는다 → 이미 조립된 폴더에서 tag 가 빠지면 옛 값을 잃는다
        #   (preset 에서 지운 ariel_ep* 가 그 경우). 덮어쓰지 않고 죽는다.
        dropped = sorted(set(old["tags"]) - set(sig["tags"]))
        if dropped and (out_dir / "readings.npz").exists():
            raise SystemExit(f"{out_dir}/readings.npz 에 지금 목록에 없는 tag {dropped} 가 있다 — 다시 조립하면 사라진다. "
                             f"덮어쓰지 않는다: --name/--out 을 바꿀 것 (ckpt 가 남아 있는 tag 라면 --preds 로 같이 줘도 된다)")
        sig["tags"] = {**old["tags"], **sig["tags"]}
        print(f"[resume] {parts} 에서 이어 간다", flush=True)
    sig_f.write_text(json.dumps(sig, indent=1))
    keys = keys_for([t for t, _ in tags], not a.no_zh)
    groups = ([] if a.no_zh else ["zh"]) + [t for t, _ in tags]
    for c, p in win:
        for k in keys:
            f = part(parts, c, p, k)
            if not f.exists():
                m = np.lib.format.open_memmap(f, mode="w+", dtype=np.float32, shape=(n, p // 2, len(COLS)))
                m[:] = np.nan; m.flush(); del m
        for g in groups:
            f = donep(parts, c, p, g)
            if not f.exists():
                m = np.lib.format.open_memmap(f, mode="w+", dtype=np.uint8, shape=(n,))
                m[:] = 0; m.flush(); del m
        f = bszp(parts, c, p)
        if not f.exists():
            m = np.lib.format.open_memmap(f, mode="w+", dtype=np.int16, shape=(n,))
            m[:] = 0; m.flush(); del m
    todo = {f"C{c}_P{p}": {g: int((np.load(donep(parts, c, p, g)) == 0).sum()) for g in groups} for c, p in win}
    print(f"[todo] 미완료 clip {todo}", flush=True)

    plan = dict(parts=str(parts), rid=rid.tolist(), tags=[t for t, _ in tags],
                ckpts={t: str(p_) for t, p_ in tags}, zh=not a.no_zh, chk_ids=chk_ids)
    t0 = time.time()
    if any(v for d_ in todo.values() for v in d_.values()):
        for f in parts.glob("timing_r*.json"):
            f.unlink()
        mp.spawn(_worker, args=(a.gpus, a, plan), nprocs=a.gpus, join=True)
    wall = time.time() - t0
    print(f"[extract] 끝 {wall/60:.1f} 분", flush=True)

    # ---- 조립
    left = {f"C{c}_P{p}/{g}": int((np.load(donep(parts, c, p, g)) == 0).sum()) for c, p in win for g in groups}
    if any(left.values()):
        raise SystemExit(f"미완료가 남았다 {left} — 같은 명령으로 다시 돌릴 것")
    out = {f"C{c}_P{p}_{k}": np.asarray(np.load(part(parts, c, p, k))) for c, p in win for k in keys}
    bad = [k for k, v in out.items() if np.isnan(v).any()]
    assert not bad, f"NaN 이 남은 키 {bad[:5]}"
    Sm = json.loads((READOUT_DIR / "summary.json").read_text())
    thr = {r: float(Sm["reps"][r]["attn"]["thr_val_fpr5"]) for r in ("p", "h", "z")}
    bias = json.loads((READOUT_DIR / "attn_bias_px.json").read_text())
    kinds = {t: load_sd(p_)[0] for t, p_ in tags}
    sel = [rows[i] for i in rid]
    keep = dict(video_id=np.array(ids), scenario=np.array([r["scenario"] for r in sel]),
                role=np.array([r["role"] for r in sel]), block_id=np.array([r["block_id"] for r in sel]),
                holdout=np.array([int(r["holdout"]) for r in sel]),
                primary=np.array([float(r["primary"]) for r in sel]),
                secondary=np.array([float(r["secondary"]) for r in sel]),
                traj=np.array([f"{r['scenario']}|{r['primary']}|{r['secondary']}" for r in sel]),
                truth=L[rid], in_frame=inf[rid], decoder_fp=np.array(fp), decoder_dir=np.array(str(READOUT_DIR)),
                cols=np.array(COLS), **out,
                **{f"bsz_C{c}_P{p}": np.load(bszp(parts, c, p)) for c, p in win})   # clip 마다 실제 배치 크기
    np.savez_compressed(out_dir / "readings.npz", **keep)

    timing = {f.stem: json.loads(f.read_text()) for f in sorted(parts.glob("timing_r*.json"))}
    rel_eq = {f.stem: json.loads(f.read_text()) for f in sorted(parts.glob("chk_release_equiv_T*.json"))}
    casts = {f.stem: json.loads(f.read_text()) for f in sorted(parts.glob("chk_sdpa_casts_*.json"))}
    wc = window_checks(parts, out, [t for t, _ in tags], kinds, win, thr["p"])
    val = dict(release_equiv=rel_eq, sdpa_qk_casts_per_forward=casts, window_independence=wc)
    if a.validate:
        val["vs_reference"] = validate_ref(out, ids, win, thr, fp)
    (out_dir / "validation.json").write_text(json.dumps(val, indent=1))
    meta = dict(n_clip=n, windows=win, split=SPLIT, cols=COLS, cell_px=CELL, norm=RES,
                keys={f"C{c}_P{p}": keys for c, p in win},
                tags=[t for t, _ in tags], ckpts={t: str(p_) for t, p_ in tags}, kinds=kinds,
                preset=a.preset if not a.preds else None, decoder_fp=fp, readout=str(READOUT_DIR),
                thr_val_fpr5=thr, attn_bias_px=bias, zh=not a.no_zh, pred_autocast=a.pred_autocast,
                batch_size={f"C{c}_P{p}": bs_for(a, c + p) for c, p in win},
                autocast_bf16={t: needs_autocast(kinds[t], a.pred_autocast) for t in kinds}, sample=a.sample, limit=a.limit,
                check_clips=[ids[i] for i in chk_ids], extract_wall_min=round(wall / 60, 2), timing=timing,
                video_id=ids)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))

    # ---- 요약 출력
    for k, v in casts.items():
        print(f"[SDPA q·k 캐스트/forward] {k}: {v}", flush=True)
    for k, v in rel_eq.items():
        print(f"[② release 동치] {k}: max|Δ| {v['max_abs']:.3g}  bitwise {v['bitwise_equal']}", flush=True)
    for k, v in wc.items():
        tok = (f"  token rel: same-input {v['token_rel_same_input']:.2e} · floor {v['token_rel_floor']:.2e} · "
               f"end2end {v['token_rel_end2end']:.2e} · enc z_ctx {v['token_rel_enc_zc']:.2e}") \
            if "token_rel_same_input" in v else ""
        verdict = ("PASS" if v["passed"] else "FAIL") if "passed" in v else "(대조)"
        sb = v.get("readings_same_bs")
        sbs = (f" [같은 배치 {v['n_clip_same_batch_size']} clip: 중앙 {sb['pos_diff_median']:.2f} px · "
               f"일치 {100*sb['present_agree']:.1f}%]") if sb else " [같은 배치 clip 0]"
        print(f"[③ 창 독립] {k:32s} kind={v['kind']:7s} 위치차 중앙 {v['readings_pos_diff_px']['median']:.2f} px "
              f"p99 {v['readings_pos_diff_px']['p99']:.2f} · 판정 일치 {100*v['readings_present_agree']:.1f}%{sbs}{tok}  {verdict}",
              flush=True)
    for k, v in (val.get("vs_reference") or {}).items():
        if isinstance(v, dict):
            print(f"[정본 대조] {k:34s} 위치차 중앙 {v['pos_diff_px']['median']:.2f} px p99 {v['pos_diff_px']['p99']:.2f} · "
                  f"recall {v['recall_ours']:.1f} vs {v['recall_ref']:.1f} ({v['recall_diff_pt']:+.2f} pt) "
                  f"{'PASS' if v['passed'] else 'FAIL'}", flush=True)
    for r_, tm in timing.items():
        print(f"[timing] {r_}: {tm['windows']} builds {tm['builds']} peak {tm['peak_mem_gb']} GB", flush=True)
    print(f"→ {out_dir}/readings.npz", flush=True)


if __name__ == "__main__":
    main()
