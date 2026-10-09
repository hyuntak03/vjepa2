#!/usr/bin/env python3
"""TrainingEffects — IntPhysGen v11 **여러 predictor** 의 있음/위치 읽기 + 튜블릿 평균 풀링 특징을 한 번에 뽑는다.

`v11_readout_online.py` (자 읽기) 와 `v11_pooled_features.py` (풀링 특징) 를 **predictor 여러 개**로 일반화한 것.
encoder 둘 (frozen 릴리즈 ViT-H, 레지스트리 `vith`) 은 clip 마다 **한 번만** 돌리고, predictor 만 프로세스 안에서 갈아 끼운다.

대상 clip — 두 원본 스크립트의 합집합, 순서 고정 (late 집합 다음 mid 집합, 각자 `index_probe.csv` 순서):
  sel=late : `v11_readout_online.select("late")` = visible 3 조건 (k=0) + late 가림 3 조건 (k=1..4)   10,752 clip
  sel=mid  : `v11_readout_online.select("mid")`  = mid 가림 3 조건 (k=1..4)                          5,376 clip
  둘 다 **가능 변이만** (CLAUDE.md §1-4), `probe_type` ∈ obj (물체 있음) / empty (vanish pos_b 빈 장면 = 자의 '없다' 음성).
  합계 16,128 clip (obj 14,112 · empty 2,016). `timing` = visible / late / mid (occ_timing 이 비면 visible).
  ⚠️ `v11_pooled_features.py` 의 9,408 clip (sel=late ∧ obj) 은 이 집합의 부분집합이다. 풀링 특징은 **16,128 전부**에 저장한다
     — 원본과 같은 집합이 필요하면 `meta.npz` 의 `sel == "late"` & `obj` 로 거른다 (순서도 원본과 같다).

규약 (probing/readout 관례): 모델 전체 bf16, **autocast 없음**. raw 100 프레임 stride 3 → 32 샘플 (raw 0,3,…,93),
  문맥 16 샘플 (8 튜블릿), 미래 16 샘플 (8 튜블릿), `mask_index 0`, 창 = 모델 `window_size` 32.
  z_full = LN(context_encoder(clip 32 샘플 전부))          — 자 검사용 (원본 readings 의 `z`, 미래 프레임도 본다)
  z_ctx  = context_encoder(clip, masks=[문맥 토큰])         — 미래 토큰을 transformer 전에 떨군 것 = predictor 입력 (`ctx_masked`)
  h      = LN(target_encoder(clip 32 샘플 전부))
  p_tag  = predictor_tag(z_ctx, ctx_idx, tgt_idx, mask_index=0)  — 이미 LN(target) 공간 추정치라 **다시 LN 하지 않는다** (forward.py `_ln`)
  LN = affine 없는 F.layer_norm (D=1280). 정밀도 경로는 원본을 그대로 따른다:
    읽기  z·h: LN(float32) → fp16 → float32 → 자 / p: bf16 → fp16 → float32 → 자   (v11_readout_online)
    풀링  z_ctx·h: LN(bf16) → float32 → 256 토큰 평균 → fp16 / p: bf16 → float32 → 평균 → fp16 (v11_pooled_features)
  ⚠️ prefix kind (ariel) 는 upstream `ACRoPEAttention` 이 RoPE 에서 q·k 를 float32 로 올리고 v 는 bf16 이라 autocast 없이는
     SDPA 가 죽는다. 이 스크립트는 `src/models/utils/modules.py` 의 `F` 에 얇은 대리자 (`_SDPACast`) 를 끼워 **dtype 이 다를 때만
     q·k 를 v dtype 으로 내린다** (autocast 가 SDPA 에 하는 캐스트와 같다; 같으면 no-op). src/ 는 고치지 않는다. 영향은 아래 검증 (c).

자 (ruler) = `RollOutV3/exp_results/presence/{p,h,z}/readout_attn.pt` (학습셋 training_v8, 지문 **dcd24d8a8a47** — 다르면 죽는다)
  attn 자:  a = softmax(tok·q/√D) (256,) ;  pooled = a·tok ;  xy = W_xy·pooled (정규화 좌표, px = (v+1)·144, 288 px 화면) ;
            logit = w_pres·pooled ;  top1 = max(a)
  문턱 (summary.json thr_val_fpr5): p 0.8307 · h −0.961 · z −0.662. 좌표 치우침 (attn_bias_px.json) 은 **빼지 않고** 원값을 저장한다.
  mass3 = 진실 칸 3×3 attention 질량 (균등이면 9/256 = 0.035; CLAUDE.md §5-5 — 위치 주장은 mass3 > 0.035 일 때만):
            (gx, gy) = 튜블릿 t 의 진실 = 두 샘플 (2t, 2t+1) 의 metadata object_px 평균 (288 px 좌표)
            cx = ⌊gx/18⌋, cy = ⌊gy/18⌋ ;  mass3 = Σ_{|r−cy|≤1, |c−cx|≤1, 0≤r,c<16} a[16r + c]
            (`v11_readout_attn_diag.py::mass3` 와 같은 정의. 격자 밖 칸은 버린다 — 화면 밖 진실이면 부분합 또는 0. `in_frame` 로 거를 것)
  읽기 조합: z@z (z_full, 16 튜블릿) · h@h (16) · 각 tag 마다 p@p (p 자, 8) · p@h (h 자, 8 — p 는 h 를 맞추도록 학습돼 원리적인 쪽)
  ⚠️ 자는 물체가 없어도 값을 낸다 (attention 이 퍼지면 토큰 평균 읽기 = 기본값). 기본값 = W_xy·(토큰 평균) 은 풀링 특징으로 다시 낼 수 있다.

predictor 목록 (`--preset` 또는 `--preds`). 전부 형식 `vjepa_frozen/predictor-only/v1`, 키 `predictor` (160 텐서, 릴리즈와 같은 이름) + `arch`.
  **kind 는 체크포인트 `arch.kind` 에서만** 읽는다 (없으면 default = 릴리즈 full self-attention; `ariel_prefix_*` 는 prefix = block-causal,
  `ariel_full_*` 는 oneshot = 릴리즈와 같은 full attention).
  kind 전용 키 (`prefix_impl`, `n_registers`) 도 arch 에서. 즉 `build_from_config(predictor_checkpoint=…)` 와 같은 경로.
  release          z_training/runs/release_vith/latest.pt  (model.pth predictor 추출본 — 실행마다 model.pth predictor 와 파라미터·출력 대조)
  v11_e{N}         z_training/runs/v11_postft/e{N}.pt              N = 1..10
  pv1_e{N}         z_training/runs/predictor_v1_postft/e{N}.pt     N = 1..15
  ip1_e{N}         z_training/runs/intphys1_postft/e{N}.pt         N = 5,10,…,40
  ip2_e{N}         z_training/runs/intphys2_postft/e{N}.pt         N = 5,10,…,80
  ariel_prefix_ep45  z_training/ariel/block_causal_future_only_1e_5/latest.pt   (kind prefix, Ariel 팔 B, epoch 45/45)
  ariel_full_ep40    z_training/ariel/full_attn_future_pred_1e_5/latest.pt      (kind oneshot, Ariel 팔 A, epoch 40/40)
  preset final  = release v11_e10 ip1_e40 ip2_e80 pv1_e15 ariel_prefix_ep45
  preset curves = release v11_e{1,2,3,5,10} pv1_e{1,3,5,10,15} ip1_e{5,10,20,40} ip2_e{10,40,80}
  ⚠️ 2026-09-27 — 옛 `ariel_ep{5,19,30,41,43}` (`block_causal_future_only_1e_5_ep_{N}`) 는 **파일이 지워져** 목록에서 뺐다
     (ep43 ≠ ep45, 바꿔 부르지 않는다). preset: final 의 ariel_ep43 → ariel_prefix_ep45, curves 의 ariel_ep{5,19,30,43} 은
     대체 없이 뺐다 (새 run 은 epoch 별 체크포인트가 없다). 옛 산출물 (`v11readout_curves` 등) 은 `load()` 로 그대로 읽힌다 —
     같은 폴더에 새 preset 을 치면 서명 불일치로 죽는다 (덮어쓰지 않는다) → --name/--out 을 바꿀 것.
  ⚠️ kind=ar (Ariel 팔 C, `ar_future_1e_5`) 는 **거부한다** (NotImplementedError): AR 의 'p' 는 블록별 LN(target) 공간의 rollout 이라
     이 스크립트의 p (release 규약 predictor 출력, 창 전체 타깃 공간) 와 뜻이 다르고, 자 (readout) 공간에서 아직 정의하지 않았다.
     AR 의 채점은 표준 하네스 (`run.sh surprise_c16t32 v11_split_test vith_ariel_ar_ep18`) 나 te_v11_score.py 로 한다.

출력 — `/data2/local_datasets/world/world_analysis/cache/training_effects/v11readout_<name>/` (name = preset, `--limit N` 이면 `_smoke{N}`):
  index.csv                      로더가 읽는 행 (순서 = 배열의 clip 축)
  meta.npz                       video_id block_id variant condition motion violation_type probe_type sym_k occ_timing shape_pre color_pre env
                                 timing (visible/late/mid) sel (late/mid) obj truth (n,32,2 정규화) in_frame (n,32) hidden (n,32)
                                 travel_dir gt_tub_px (n,16,2; 288 px, mass3 에 쓴 진실)
  meta.json                      predictor tag·ckpt·kind·kind_kwargs·epoch·가중치 sha1, 자 지문, 규약, 배열 목록, 서명 (재개 검사)
  rd_z.npy  (n,16,5) f32         z@z   — 열 [x, y, logit, top1, mass3]  (x·y 는 자 정규화 좌표, 치우침 미보정)
  rd_h.npy  (n,16,5) f32         h@h
  rd_p__<tag>.npy  (n,8,5) f32   p@p   (미래 튜블릿 0..7 = 전체 튜블릿 8..15)
  rd_ph__<tag>.npy (n,8,5) f32   p@h
  pool_z.npy (n,8,D) f16         z_ctx 튜블릿 평균 (문맥 8 튜블릿)  = v11_pooled_features 의 z
  pool_h.npy (n,16,D) f16        h 튜블릿 평균 (32 샘플 전부)       = 〃 h
  pool_p__<tag>.npy (n,8,D) f16  p 튜블릿 평균 (미래 8 튜블릿)      = 〃 p
  attn_{z,h}.npy (n,16,256) f16 · attn_p__<tag>.npy · attn_ph__<tag>.npy (n,8,256) f16   자 attention 지도 (--no-attn 이면 없음).
                                 임의 위치 (마지막 관측 · 가림막) 의 3×3 질량을 나중에 다시 낼 수 있다. 토큰 순서 = 16·행(y) + 열(x)
  done.npy (n,) u8               clip 완료 표시. **재개**: 같은 명령에 `--resume` — done=0 인 clip 만 다시 (서명이 다르면 죽는다)
  timing_rank{r}_{stamp}.json    단계별 시간 (encoder / predictor tag 별 / 읽기 대기), 최대 VRAM
  checks_rank0_{stamp}.json      release ↔ model.pth predictor 대조, (--validate 면) build_from_config 대조
  읽을 때: `from te_v11_readout import load; M = load(out_dir)` → memmap dict + meta.

GPU: `--gpus N` → mp.spawn, rank r 는 cuda:r (보이는 장치 기준 — CUDA_VISIBLE_DEVICES 로 고른다), clip i 는 rank i mod N.
  배열은 clip 축 memmap 이라 rank 가 겹치지 않게 제자리에 쓴다. `--flush-every` 배치마다 flush 뒤 done 표시.
  배치 크기 = min(--max-bs 16, --token-budget 65536 // 4096) = 16 — 원본 readings 와 같다.
  ⚠️ bf16 은 배치 구성이 달라지면 ~1 px 흔들린다 (rollout3_window_readout.py 머리말). 재개하면 배치 구성이 바뀐다.

검증 (`--validate`, 2026-09-25, GPU 1 장 (vll5 cuda 2, 다른 eval 과 공유), bs 16, `--limit 48` → 54 clip = 18 층 × 3):
  (a) release 읽기 ↔ `RollOutV3/exp_results/v11{,_mid}/readings.npz` (같은 video_id; 원본도 bs 16). 기준: 위치 차 중앙 ≤ 1 px · logit 중앙 ≤ 0.1 · '있다' 판정 일치 ≥ 97 %
        late 36 clip  p@p  위치 차 중앙 0.24 px · p95 0.57 · 최대 0.82 px, |Δlogit| 중앙 0.019 (최대 0.19), 판정 일치 100 %     통과
                      z@z · h@h  최대 4.3e-5 / 6.9e-5 px, |Δlogit| 최대 1.0e-5 — encoder 경로는 사실상 비트 동일         통과
        mid 18 clip   p@p  중앙 0.29 px · p95 2.1 · 최대 5.9 px, |Δlogit| 중앙 0.029 (최대 0.64), 판정 일치 99.3 % (144 칸 중 1)  통과
                      z@z · h@h  최대 3.5e-5 px                                                                          통과
        p 만 흔들리는 것은 bf16 predictor 커널 잡음이다 (rollout3_window_readout.py 머리말의 ~1 px 와 같은 크기).
  (b) 풀링 ↔ `cache/v11_pooled_vith` (sel=late ∧ obj 18 clip; video_id 로 짝짓고 block_id·condition·sym_k·shape·color 일치 확인)
        z  최대|Δ| 4.8e-7, 상대 L2 2.0e-9 (99.9995 % 비트 동일) · h 1.2e-7, 3.5e-10 ·
        p  최대|Δ| 5.9e-3, 상대 L2 9.0e-4, 최소 cos 0.999999 (원본 bs 32 ↔ 여기 bs 16 의 bf16 잡음). 기준 상대 L2 ≤ 1e-2 · cos ≥ 0.999   통과
  (c) predictor 6 개 전부 유한값. release latest.pt ↔ model.pth predictor: 파라미터 160/160 동일, 같은 배치 출력 max|Δp| = 0.
      `build_from_config(predictor_checkpoint=…)` 로 지은 predictor 와 같은 배치 출력 대조: v11_e10 · ariel_ep43 둘 다 max|Δp| = 0.
      ariel_ep43 = kind prefix (VisionTransformerPredictorPrefix, mask_mode prefix, prefix_impl split, n_registers 0) — 로그에 찍힌다.
      SDPA q·k 캐스트 (아래 ⚠️) 는 ariel 에서 배치당 108 회 (12 층 × 9 호출), default 5 개는 0 회.
      캐스트 우회의 수치 영향 (8 clip, 스크래치 검사): ariel p 의 bf16+우회 vs fp32 predictor 상대 L2 0.0061 (autocast-bf16 vs fp32 0.0067),
      release 의 순수 bf16 vs fp32 는 0.0098 — 보통 bf16 잡음 안이다. p@p 위치 차 중앙 0.20 px (release 0.40 px).
  시간 (cuda 2 공유, 첫 배치 제외 38 clip, bs 16): clip 당 0.275 s = encoder 3 회 0.220 + predictor 6 개 0.046
      (default 0.0066~0.0075, prefix 0.0104) + PNG 대기 0.009. 최대 VRAM 5.4 GiB.
      16,128 clip 추정 (로드 ~2 분 별도, 공유 GPU 면 더 느리다): final 8 GPU ≈ 9 분 · 4 GPU ≈ 19 분 /
      curves (clip 당 ≈ 0.40 s) 8 GPU ≈ 13 분 · 4 GPU ≈ 27 분. 디스크 final 3.8 GiB · curves 10.8 GiB (--no-attn 2.8 / 7.8).

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  CUDA_VISIBLE_DEVICES=2 $P z_research/scripts/analysis/te_v11_readout.py --preset final --gpus 1 --limit 48 --validate   # 검증 (~54 clip)
  $P z_research/scripts/analysis/te_v11_readout.py --preset final --gpus 8          # 본 실행 → cache/training_effects/v11readout_final/
  $P z_research/scripts/analysis/te_v11_readout.py --preset curves --gpus 8         # 학습 곡선 23 predictor
  $P z_research/scripts/analysis/te_v11_readout.py --preset final --gpus 8 --resume # 죽은 실행 이어서
  $P z_research/scripts/analysis/te_v11_readout.py --preds release,v11_e10,mine=/abs/path/e3.pt --name custom --gpus 8
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, re, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))
from rollout3_paths import PRES as READOUT_DIR, EXP, fingerprint          # noqa: E402
from rollout3_window_readout import MODEL, readout_cls, prefetch            # noqa: E402  (모델·자 정의를 복제하지 않는다)
import v11_readout_online as vro                                            # noqa: E402  (clip 선택 · 가림 샘플 규약 공유)

CACHE_ROOT = Path(os.environ.get("TE_CACHE", "/data2/local_datasets/world/world_analysis/cache/training_effects"))
DOC_DIR = ROOT / "z_research/TrainingEffects/v11readout"
POOLED_REF = Path("/local_datasets/world/world_analysis/cache/v11_pooled_vith")
FP_WANT = "dcd24d8a8a47"
NS, CS, S, D, RES, CELL = 32, 16, 256, 1280, 144.0, 18.0
TC, TP, TA = CS // 2, (NS - CS) // 2, NS // 2
KEEP = ["video_id", "block_id", "variant", "condition", "motion", "violation_type", "probe_type", "sym_k",
        "occ_timing", "shape_pre", "color_pre", "env"]

RUNS, ARIEL = ROOT / "z_training/runs", ROOT / "z_training/ariel"
FAMILY = {"v11": "v11_postft", "pv1": "predictor_v1_postft", "ip1": "intphys1_postft", "ip2": "intphys2_postft"}
# Ariel (2026-09-27 수령) 태그 → (폴더, 기대 arch.kind). kind=ar 팔 (ar_future_1e_5) 은 readout 에서 거부하므로 목록에 없다.
# ⚠️ 옛 ariel_ep{5,19,30,41,43} (block_causal_future_only_1e_5_ep_{N}) 는 2026-09-27 에 파일이 지워졌다 — 옛 산출물만 읽힌다.
ARIEL_TAGS = {"ariel_prefix_ep45": ("block_causal_future_only_1e_5", "prefix"),
              "ariel_full_ep40": ("full_attn_future_pred_1e_5", "oneshot")}
PRESETS = {
    # 2026-09-27: final 의 ariel_ep43 (지워짐) → ariel_prefix_ep45 (같은 run 의 마지막 epoch, 다른 파일)
    "final": ["release", "v11_e10", "ip1_e40", "ip2_e80", "pv1_e15", "ariel_prefix_ep45"],
    # 2026-09-27: ariel_ep{5,19,30,43} 은 대체 곡선이 없어 뺐다
    "curves": ["release", "v11_e1", "v11_e2", "v11_e3", "v11_e5", "v11_e10",
               "pv1_e1", "pv1_e3", "pv1_e5", "pv1_e10", "pv1_e15",
               "ip1_e5", "ip1_e10", "ip1_e20", "ip1_e40", "ip2_e10", "ip2_e40", "ip2_e80"],
}


def expected_kind(tag: str):
    """validate (c) 의 kind 검사: Ariel 태그가 선언해야 할 arch.kind (옛 ariel_ep* = prefix). 그 밖은 None (검사 안 함)."""
    if tag in ARIEL_TAGS:
        return ARIEL_TAGS[tag][1]
    return "prefix" if re.fullmatch(r"ariel_ep\d+", tag) else None


def refuse_ar(tag: str, kind: str):
    """kind=ar (자기회귀) predictor 거부 (2026-09-27). 이 readout 의 p 는 release 규약 predictor 출력이다."""
    if kind == "ar":
        raise NotImplementedError(
            f"{tag}: kind=ar (자기회귀) predictor 의 readout 은 정의하지 않았다 — AR 의 'p' 는 블록별 LN(target_encoder) "
            f"공간의 rollout 이라 자 (readout) 공간의 p 와 뜻이 다르다 (bf16 · autocast 없는 이 스크립트에서는 SDPA 도 죽는다). "
            f"채점은 run.sh surprise_c16t32 v11_split_test vith_ariel_ar_ep18 또는 te_v11_score.py 로 한다")


# ----------------------------------------------------------------------------- predictor 목록
def ckpt_of(tag: str) -> Path:
    if tag == "release":
        return RUNS / "release_vith/latest.pt"
    m = re.fullmatch(r"(v11|pv1|ip1|ip2)_e(\d+)", tag)
    if m:
        return RUNS / FAMILY[m[1]] / f"e{int(m[2])}.pt"
    if tag in ARIEL_TAGS:
        return ARIEL / ARIEL_TAGS[tag][0] / "latest.pt"
    if re.fullmatch(r"ariel_ep\d+", tag):
        raise SystemExit(f"{tag}: 옛 Ariel epoch 체크포인트 (block_causal_future_only_1e_5_ep_N) 는 2026-09-27 에 지워졌다. "
                         f"지금 판: {sorted(ARIEL_TAGS)} (ep43 ≠ ep45 — 바꿔 부르지 않는다)")
    if tag == "ariel_ar_ep18":
        refuse_ar(tag, "ar")
    raise SystemExit(f"모르는 tag {tag!r} — tag=경로 로 줄 것")


def parse_preds(a) -> list:
    items = PRESETS[a.preset] if a.preset else [s for s in re.split(r"[,\s]+", a.preds or "") if s]
    out = []
    for it in items:
        tag, _, path = it.partition("=")
        if not re.fullmatch(r"[A-Za-z0-9_]+", tag):
            raise SystemExit(f"tag 는 [A-Za-z0-9_] 만: {tag!r}")
        p = Path(path) if path else ckpt_of(tag)
        if not p.is_file():
            raise SystemExit(f"체크포인트가 없다: {tag} → {p}")
        out.append((tag, str(p)))
    if len({t for t, _ in out}) != len(out):
        raise SystemExit(f"tag 중복: {[t for t, _ in out]}")
    return out


def load_pred(bundle, sd: dict, arch: dict):
    """`build_from_config(predictor_checkpoint=…)` 의 predictor 경로와 같은 순서로 짓는다 (kind·kind 전용 키 = arch)."""
    from analysis.intphys2.model import _build_predictor, _clean_backbone_key
    from analysis import predictors as PV
    kind = str(arch.get("kind") or "").strip() or "default"
    refuse_ar("predictor", kind)                                           # 2026-09-27: 모델을 짓기 전에 거부
    kk = {k: arch[k] for k in PV.KIND_ONLY_KEYS if k in arch}
    pc = MODEL["predictor"]
    pred = _build_predictor(img_size=MODEL["img_size"], patch_size=MODEL["patch_size"], tubelet_size=MODEL["tubelet_size"],
                            num_frames=bundle.num_frames, encoder_embed_dim=bundle.embed_dim,
                            predictor_embed_dim=pc["embed_dim"], predictor_depth=pc["depth"],
                            predictor_num_heads=pc["num_heads"], num_mask_tokens=pc["num_mask_tokens"],
                            use_rope=MODEL["use_rope"], uniform_power=MODEL["uniform_power"], kind=kind, kind_kwargs=kk)
    pred.load_state_dict(_clean_backbone_key(sd), strict=True)
    pred.attn_regime = getattr(pred, "attn_regime", "full_self_attention")
    pred = pred.to(device=bundle.device, dtype=bundle.dtype).eval().requires_grad_(False)
    return pred, kind, kk


def describe(pred) -> str:
    return (f"{type(pred).__name__} mask_mode={getattr(pred, 'mask_mode', 'full(release)')} "
            f"prefix_impl={getattr(pred, 'prefix_impl', '-')} n_registers={getattr(pred, 'n_registers', 0)}")


class _SDPACast:
    """`src/models/utils/modules.py` 의 `F` 대신 끼우는 얇은 대리자 — **prefix kind 를 bf16 · autocast 없이 돌리기 위한 우회**.

    ACRoPEAttention (prefix predictor 의 attention) 은 RoPE 위치 텐서가 float32 라 q·k 가 float32 로 올라가고 v 는 bf16 에
    남는다 → PrefixSpec 분기의 `F.scaled_dot_product_attention` 이 dtype 불일치로 죽는다 (autocast 아래에서만 도는 코드).
    여기서는 **q·k 를 v 의 dtype 으로 내리기만** 한다 (autocast 가 SDPA 에 하는 것과 같은 캐스트). dtype 이 같으면 no-op 이라
    encoder · default predictor 는 비트 단위로 그대로다 (release ↔ model.pth max|Δp| = 0 이 그 확인). 캐스트 횟수를 센다.
    공유 코드 (src/) 는 고치지 않는다.
    """
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


def sd_sha(sd: dict) -> str:
    h = hashlib.sha1()
    for k in sorted(sd):
        h.update(k.encode()); h.update(sd[k].detach().cpu().float().contiguous().numpy().tobytes())
    return h.hexdigest()[:12]


# ----------------------------------------------------------------------------- clip 집합
def select_all():
    rows, parts = [], []
    for sel in ("late", "mid"):
        r, L, inf, hid, obj, tdir = vro.select(sel)
        for x in r:
            x["sel"] = sel
            x["timing"] = x["occ_timing"] or "visible"
        rows += r; parts.append((L, inf, hid, obj, tdir))
    L, inf, hid, obj, tdir = (np.concatenate([p[j] for p in parts]) for j in range(5))
    return rows, L, inf, hid, obj, tdir


def pick_limit(rows, limit):
    """층 (sel, condition, probe_type) 마다 ⌈limit/층수⌉ 개를 층 안에서 고르게 (k 가 고루 들어가게)."""
    strata = {}
    for i, r in enumerate(rows):
        strata.setdefault((r["sel"], r["condition"], r["probe_type"]), []).append(i)
    per = max(1, math.ceil(limit / len(strata)))
    pick = []
    for key in sorted(strata):
        ids = strata[key]
        pick += [ids[j] for j in sorted(set(np.linspace(0, len(ids) - 1, per).round().astype(int).tolist()))]
    return sorted(pick)


def array_specs(tags, n, attn=True):
    """이름 → (shape, dtype)."""
    sp = {"rd_z": ((n, TA, 5), np.float32), "rd_h": ((n, TA, 5), np.float32),
          "pool_z": ((n, TC, D), np.float16), "pool_h": ((n, TA, D), np.float16)}
    if attn:
        sp.update(attn_z=((n, TA, S), np.float16), attn_h=((n, TA, S), np.float16))
    for t in tags:
        sp[f"rd_p__{t}"] = ((n, TP, 5), np.float32); sp[f"rd_ph__{t}"] = ((n, TP, 5), np.float32)
        sp[f"pool_p__{t}"] = ((n, TP, D), np.float16)
        if attn:
            sp[f"attn_p__{t}"] = ((n, TP, S), np.float16); sp[f"attn_ph__{t}"] = ((n, TP, S), np.float16)
    return sp


def load(out_dir, mode="r"):
    """출력 폴더 → {배열 이름: memmap, 'meta': meta.npz dict, 'info': meta.json}."""
    out = Path(out_dir); info = json.loads((out / "meta.json").read_text())
    M = {k: np.load(out / f"{k}.npy", mmap_mode=mode) for k in info["arrays"]}
    M["done"] = np.load(out / "done.npy", mmap_mode=mode)
    M["meta"] = dict(np.load(out / "meta.npz", allow_pickle=False)); M["info"] = info
    return M


# ----------------------------------------------------------------------------- worker
def mass3(a, gt):
    """a (B,T,256) attention, gt (B,T,2) 288 px → (B,T) 진실 칸 3×3 질량 (격자 밖 칸 제외; 진실 NaN 이면 NaN)."""
    cx, cy = torch.floor(gt[..., 0] / CELL), torch.floor(gt[..., 1] / CELL)
    g = torch.arange(16, device=a.device, dtype=gt.dtype)
    m = ((g.view(1, 1, 16, 1) - cy[..., None, None]).abs() <= 1) & ((g.view(1, 1, 1, 16) - cx[..., None, None]).abs() <= 1)
    out = (a.view(*a.shape[:2], 16, 16) * m).sum((-1, -2))
    return torch.where(torch.isfinite(gt).all(-1), out, torch.full_like(out, float("nan")))


def _worker(rank, world, args, n, spec):
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    torch.set_num_threads(max(1, min(8, (os.cpu_count() or 8) // max(1, world))))
    from analysis.intphys2.model import build_from_config
    from analysis.intphys2.surprise import _context_target_indices
    from evals.world_model_analysis.data import WMADataset
    import decord  # noqa: F401  (WMADataset 이 쓴다)

    out = Path(spec["out"]); tags = [t for t, _ in spec["preds"]]
    say = (lambda *s: print(f"[r{rank}]", *s, flush=True))
    install_sdpa_cast()                                                    # prefix kind 용 (dtype 같으면 no-op)
    bundle = build_from_config({**MODEL, "window_size": NS}, dev)          # encoder 둘 + 릴리즈 predictor (model.pth)
    R = readout_cls(); heads = {}
    for rep in ("p", "z", "h"):
        hd = R("attn").to(dev).eval()
        hd.load_state_dict(torch.load(READOUT_DIR / rep / "readout_attn.pt", map_location="cpu"))
        heads[rep] = hd.requires_grad_(False)

    preds, checks = {}, {"predictors": {}}
    for tag, _ in spec["preds"]:
        st = torch.load(Path(spec["stage"]) / f"{tag}.pt", map_location="cpu", weights_only=False)
        preds[tag], kind, kk = load_pred(bundle, st["predictor"], st["arch"])
        if kind != st["kind"]:
            raise SystemExit(f"{tag}: kind 불일치 {kind} vs {st['kind']}")
        checks["predictors"][tag] = dict(kind=kind, kind_kwargs=kk, module=describe(preds[tag]))
        if rank == 0:
            say(f"predictor {tag:<11} kind={kind:<8} {kk or ''} {describe(preds[tag])}")
        del st
    if "release" in preds:                                      # latest.pt 추출본 = model.pth predictor 인가 (파라미터)
        a_, b_ = preds["release"].state_dict(), bundle.predictor.state_dict()
        neq = [k for k in b_ if not torch.equal(a_[k], b_[k])]
        checks["release_vs_modelpth_params"] = dict(n=len(b_), n_unequal=len(neq), unequal=neq[:5])
        if neq or set(a_) != set(b_):
            raise SystemExit(f"release latest.pt ≠ model.pth predictor: {neq[:5]}")
    xcheck_ref = {}
    if rank == 0 and args.xcheck:                               # build_from_config(predictor_checkpoint=…) 로 지은 것과 출력 대조
        for tag in args.xcheck:
            ck = dict(spec["preds"])[tag]
            b2 = build_from_config({**MODEL, "window_size": NS, "predictor_checkpoint": ck}, torch.device("cpu"))
            xcheck_ref[tag] = b2.predictor.to(dev); del b2
            say(f"xcheck {tag}: build_from_config predictor = {describe(xcheck_ref[tag])}")

    ds = WMADataset(dict(data={**vro.DATA, "root": str(out)}, model={**MODEL, "window_size": NS},
                         features={"cache_dir": "/tmp"}, surprise={}))
    assert len(ds) == n, f"로더 clip 수 {len(ds)} ≠ {n}"
    mm = {k: np.load(out / f"{k}.npy", mmap_mode="r+") for k in spec["arrays"]}
    done = np.load(out / "done.npy", mmap_mode="r+")
    gt_all = torch.from_numpy(np.load(out / "meta.npz")["gt_tub_px"]).float()        # (n,16,2)
    bs = max(1, min(args.max_bs, args.token_budget // (TA * S)))
    idx = [i for i in range(rank, n, world) if not done[i]]
    say(f"{len(idx):,} clip 남음 (전체 {n:,} 중 이 rank 몫) · bs={bs}") if rank == 0 else None
    pool = ThreadPoolExecutor(max_workers=args.workers)
    ln = lambda x: F.layer_norm(x, (x.size(-1),))
    sync = lambda: torch.cuda.synchronize(dev)
    T = {"wait": 0.0, "enc": 0.0, **{f"pred:{t}": 0.0 for t in tags}}; n_timed, pending, nb = 0, [], 0
    t_start = time.time(); t_last = time.time()

    def put(key, arr, ids):
        if key in mm:
            mm[key][ids] = arr

    def read(head, tok, B, nt, gt, name, ids):
        xy, pres, a = head(tok)                                            # tok (B·nt,256,D) f32
        a = a.view(B, nt, S)
        v = torch.cat([xy.view(B, nt, 2), pres.view(B, nt, 1), a.max(-1).values[..., None], mass3(a, gt)[..., None]], -1)
        put(f"rd_{name}", v.cpu().numpy(), ids)
        if f"attn_{name}" in mm:
            put(f"attn_{name}", a.half().cpu().numpy(), ids)
        return v

    first_checks = True
    with torch.inference_mode():
        for ids, clips in prefetch(ds, idx, bs, pool):
            sync(); t0 = time.time(); w = t0 - t_last
            B = clips.size(0); ids = np.asarray(ids)
            clips = clips.to(dev, dtype=bundle.dtype, non_blocking=True)
            gt = gt_all[torch.from_numpy(ids)].to(dev)
            ctx_idx, tgt_idx = _context_target_indices(ctx_frames=CS, tgt_frames=NS - CS, tubelet_size=bundle.tubelet_size,
                                                       spatial_tokens=S, batch_size=B, device=dev)
            zc = bundle.context_encoder(clips, masks=[ctx_idx]); zc = zc[-1] if isinstance(zc, list) else zc
            ht = bundle.target_encoder(clips); ht = ht[-1] if isinstance(ht, list) else ht
            zf = bundle.context_encoder(clips); zf = zf[-1] if isinstance(zf, list) else zf
            # 풀링 (v11_pooled_features: LN 은 bf16 에서, 평균은 float32)
            put("pool_z", ln(zc).float().view(B, TC, S, D).mean(2).half().cpu().numpy(), ids)
            put("pool_h", ln(ht).float().view(B, TA, S, D).mean(2).half().cpu().numpy(), ids)
            # 읽기 (v11_readout_online: LN 은 float32 에서, fp16 으로 한 번 반올림)
            read(heads["z"], ln(zf.float()).half().float().view(B * TA, S, D), B, TA, gt, "z", ids)
            read(heads["h"], ln(ht.float()).half().float().view(B * TA, S, D), B, TA, gt, "h", ids)
            del zf
            sync(); t1 = time.time(); T_enc = t1 - t0; T_p = {}
            gtf = gt[:, TC:]
            for tag in tags:
                n0 = _SDPACast.n_cast
                p = preds[tag](zc, ctx_idx, tgt_idx, mask_index=0); p = p[-1] if isinstance(p, list) else p
                if first_checks:
                    checks["predictors"][tag]["sdpa_qk_casts_first_batch"] = _SDPACast.n_cast - n0
                    if rank == 0:
                        say(f"{tag}: SDPA q·k→v dtype 캐스트 {_SDPACast.n_cast - n0} 회 (첫 배치)")
                if first_checks and rank == 0:
                    if tag == "release":
                        r0 = bundle.predictor(zc, ctx_idx, tgt_idx, mask_index=0); r0 = r0[-1] if isinstance(r0, list) else r0
                        checks["release_vs_modelpth_output"] = dict(max_abs=float((p.float() - r0.float()).abs().max()),
                                                                    n_clip=B)
                        say(f"release latest.pt vs model.pth predictor: max|Δp| = {checks['release_vs_modelpth_output']['max_abs']:.3g}")
                    if tag in xcheck_ref:
                        r1 = xcheck_ref[tag](zc, ctx_idx, tgt_idx, mask_index=0); r1 = r1[-1] if isinstance(r1, list) else r1
                        d = (p.float() - r1.float()).abs()
                        checks.setdefault("xcheck_build_from_config", {})[tag] = dict(
                            max_abs=float(d.max()), rel_l2=float(d.norm() / r1.float().norm()), n_clip=B,
                            ours=describe(preds[tag]), ref=describe(xcheck_ref[tag]))
                        say(f"xcheck {tag}: ours vs build_from_config max|Δp| = {float(d.max()):.3g}")
                put(f"pool_p__{tag}", p.float().view(B, TP, S, D).mean(2).half().cpu().numpy(), ids)
                tok = p.half().float().view(B * TP, S, D)
                read(heads["p"], tok, B, TP, gtf, f"p__{tag}", ids)
                read(heads["h"], tok, B, TP, gtf, f"ph__{tag}", ids)
                del p, tok
                sync(); t2 = time.time(); T_p[tag] = t2 - t1; t1 = t2
            if first_checks:
                first_checks = False; xcheck_ref.clear()
                if rank == 0:
                    (out / f"checks_rank0_{spec['stamp']}.json").write_text(json.dumps(checks, indent=1))
            else:                                                   # 첫 배치 (warm-up) 는 시간에서 뺀다
                n_timed += B; T["wait"] += w; T["enc"] += T_enc
                for t_ in tags:
                    T[f"pred:{t_}"] += T_p[t_]
            pending += ids.tolist(); nb += 1
            if nb % args.flush_every == 0:
                for v in mm.values():
                    v.flush()
                done[pending] = 1; done.flush(); pending = []
            if rank == 0 and nb % 10 == 0:
                el = time.time() - t_start; frac = nb * bs / max(1, len(idx))
                say(f"bs={bs} {nb*bs*world:,}/{len(idx)*world:,} clip  {el:.0f}s  남은 {el*(1-frac)/max(frac,1e-9)/60:.1f}분  "
                    f"최대 VRAM {torch.cuda.max_memory_allocated(dev)/2**30:.1f} GiB")
            t_last = time.time()
    for v in mm.values():
        v.flush()
    if pending:
        done[pending] = 1; done.flush()
    tim = dict(rank=rank, n_clip_timed=n_timed, bs=bs, preds=tags,
               per_clip_s={k: v / max(1, n_timed) for k, v in T.items()},
               per_clip_total_s=sum(T.values()) / max(1, n_timed),
               max_vram_gib=torch.cuda.max_memory_allocated(dev) / 2**30, wall_s=time.time() - t_start)
    (out / f"timing_rank{rank}_{spec['stamp']}.json").write_text(json.dumps(tim, indent=1))
    say(f"끝 · clip 당 {tim['per_clip_total_s']:.3f}s (encoder {tim['per_clip_s']['enc']:.3f}s, "
        f"predictor 합 {sum(v for k, v in tim['per_clip_s'].items() if k.startswith('pred:')):.3f}s, "
        f"읽기 대기 {tim['per_clip_s']['wait']:.3f}s) · 최대 VRAM {tim['max_vram_gib']:.1f} GiB")


# ----------------------------------------------------------------------------- 검증
def validate(out: Path, stamp: str):
    """(a) release 읽기 ↔ RollOutV3/exp_results/v11{,_mid}/readings.npz  (b) 풀링 ↔ cache/v11_pooled_vith  (c) predictor 전부."""
    M = load(out); meta, info = M["meta"], M["info"]; tags = [p["tag"] for p in info["preds"]]
    assert M["done"].all(), "완료 안 된 clip 이 있다"
    S_ = json.loads((READOUT_DIR / "summary.json").read_text())["reps"]
    thr = {r: S_[r]["attn"]["thr_val_fpr5"] for r in ("p", "z", "h")}
    res = {"out": str(out), "n_clip": int(len(meta["video_id"])), "thr": thr}

    def cmp_read(new, old, th):
        d = new[..., :4].astype(np.float64) - old.astype(np.float64)
        px = np.hypot(d[..., 0], d[..., 1]) * RES
        agree = ((new[..., 2] > th) == (old[..., 2] > th)).mean()
        r = dict(n=int(px.size), px_med=float(np.median(px)), px_p95=float(np.percentile(px, 95)), px_max=float(px.max()),
                 logit_med=float(np.median(np.abs(d[..., 2]))), logit_max=float(np.abs(d[..., 2]).max()),
                 top1_max=float(np.abs(d[..., 3]).max()), decision_agree=float(agree),
                 exact_frac=float((d == 0).all(-1).mean()))
        r["pass"] = bool(r["px_med"] <= 1.0 and r["logit_med"] <= 0.1 and r["decision_agree"] >= 0.97)
        return r

    # (a)
    a = {}
    vid = meta["video_id"]; sel = meta["sel"]
    for s, d in (("late", "v11"), ("mid", "v11_mid")):
        R = np.load(EXP / d / "readings.npz", allow_pickle=True)
        assert str(R["decoder_fp"]) == FP_WANT, f"{d}/readings.npz 자 지문 {R['decoder_fp']}"
        pos = {v: j for j, v in enumerate(R["video_id"])}
        ii = np.where(sel == s)[0]; jj = np.array([pos[v] for v in vid[ii]])
        assert (R["video_id"][jj] == vid[ii]).all()
        sub = {}
        if "release" in tags:
            sub["p"] = cmp_read(np.asarray(M["rd_p__release"][ii]), R["p"][jj], thr["p"])
        sub["z"] = cmp_read(np.asarray(M["rd_z"][ii]), R["z"][jj], thr["z"])
        sub["h"] = cmp_read(np.asarray(M["rd_h"][ii]), R["h"][jj], thr["h"])
        sub["n_clip"] = int(len(ii)); a[s] = sub
    res["a_readings_vs_release"] = a
    # (b)
    b = {}
    ref = np.load(POOLED_REF / "meta.npz")
    pos = {v: j for j, v in enumerate(ref["video_id"])}
    ii = np.array([i for i, v in enumerate(vid) if v in pos]); jj = np.array([pos[v] for v in vid[ii]])
    for key, mine in (("z", "pool_z"), ("h", "pool_h"), ("p", "pool_p__release")):
        if mine not in M:
            continue
        X = np.load(POOLED_REF / f"{key}.npy", mmap_mode="r")
        new = np.asarray(M[mine][ii]).astype(np.float64); old = np.asarray(X[jj]).astype(np.float64)
        dd = new - old
        cos = (new * old).sum(-1) / (np.linalg.norm(new, axis=-1) * np.linalg.norm(old, axis=-1))
        r = dict(n_clip=int(len(ii)), max_abs=float(np.abs(dd).max()), ref_abs_mean=float(np.abs(old).mean()),
                 rel_l2=float(np.linalg.norm(dd) / np.linalg.norm(old)), min_cos=float(cos.min()),
                 exact_frac=float((dd == 0).mean()))
        r["pass"] = bool(r["rel_l2"] <= 1e-2 and r["min_cos"] >= 0.999)
        b[key] = r
    # 같은 video_id 인지 (meta 정렬) — 라벨 열도 같아야 한다
    b["meta_aligned"] = bool(all((ref[k][jj] == meta[k][ii]).all() for k in ("block_id", "condition", "sym_k", "shape_pre", "color_pre")))
    res["b_pooled_vs_cache"] = b
    # (c)
    c = {}
    for t in tags:
        v = np.asarray(M[f"rd_p__{t}"]); pp = np.asarray(M[f"pool_p__{t}"])
        c[t] = dict(kind=next(p["kind"] for p in info["preds"] if p["tag"] == t),
                    finite=bool(np.isfinite(v[..., :4]).all() and np.isfinite(pp.astype(np.float32)).all()),
                    say_frac_obj=float((v[meta["obj"], :, 2] > thr["p"]).mean()),
                    say_frac_empty=float((v[~meta["obj"], :, 2] > thr["p"]).mean()),
                    mass3_obj_mean=float(np.nanmean(v[meta["obj"], :, 4])),
                    diff_vs_release=(float(np.abs(pp.astype(np.float32) - np.asarray(M["pool_p__release"]).astype(np.float32)).mean())
                                     if "release" in tags else None))
    res["c_predictors"] = c
    for f in ("checks_rank0", "timing_rank0"):                  # 이번 실행 것, 없으면 (완료된 폴더 재검증) 가장 최근 것
        g = out / f"{f}_{stamp}.json"
        cand = [g] if g.exists() else sorted(out.glob(f"{f}_*.json"))
        if cand:
            res[f] = json.loads(cand[-1].read_text()); res[f + "_file"] = cand[-1].name
    ck = res.get("checks_rank0", {})
    passes = dict(
        a=all(r["pass"] for s in a.values() for k, r in s.items() if isinstance(r, dict)),
        b=all(r["pass"] for k, r in b.items() if isinstance(r, dict)) and b["meta_aligned"],
        c=all(x["finite"] for x in c.values())
          and all(x["kind"] == expected_kind(t) for t, x in c.items() if expected_kind(t)),   # 2026-09-27: full 팔은 oneshot
        release_eq=ck.get("release_vs_modelpth_output", {}).get("max_abs", 1.0) == 0.0
                   and ck.get("release_vs_modelpth_params", {}).get("n_unequal", 1) == 0,
        xcheck=all(v["max_abs"] == 0.0 or v["rel_l2"] <= 1e-4
                   for v in ck.get("xcheck_build_from_config", {}).values()),
    )
    res["pass"] = passes; res["all_pass"] = bool(all(passes.values()))
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    f = DOC_DIR / f"validation_{out.name}.json"
    if f.exists():
        f = DOC_DIR / f"validation_{out.name}_{stamp}.json"
    f.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("a_readings_vs_release", "b_pooled_vs_cache", "pass", "all_pass")}, indent=1))
    print(f"→ {f}")
    return res


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preset", choices=sorted(PRESETS))
    g.add_argument("--preds", help="tag 또는 tag=경로, 쉼표 구분. 예: release,v11_e10,x=/abs/e3.pt")
    ap.add_argument("--name", default=None, help="출력 폴더 꼬리 (기본: preset 이름, --preds 면 필수)")
    ap.add_argument("--out", default=None, help="출력 폴더를 직접 (기본 cache/training_effects/v11readout_<name>)")
    ap.add_argument("--gpus", type=int, default=torch.cuda.device_count() or 1)
    ap.add_argument("--limit", type=int, default=0, help="층 (sel·조건·probe_type) 18 개에서 고르게 ⌈limit/18⌉ 개씩 (검증·배관)")
    ap.add_argument("--resume", action="store_true", help="기존 출력 폴더를 이어 쓴다 (서명이 같아야 한다)")
    ap.add_argument("--no-attn", action="store_true", help="attention 지도 (attn_*.npy) 를 저장하지 않는다")
    ap.add_argument("--token-budget", type=int, default=65536, help="배치 토큰 상한 (4,096 토큰/clip → bs 16)")
    ap.add_argument("--max-bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8, help="PNG 읽기 스레드 (GPU 당)")
    ap.add_argument("--flush-every", type=int, default=20, help="몇 배치마다 flush + done 표시")
    ap.add_argument("--validate", action="store_true", help="끝나고 (a)(b)(c) 대조 → z_research/TrainingEffects/v11readout/")
    ap.add_argument("--xcheck", nargs="*", default=None,
                    help="build_from_config(predictor_checkpoint) 와 출력 대조할 tag (--validate 기본: v11_e10 · ariel_prefix_ep45 · "
                         "ariel_full_ep40 중 있는 것)")
    a = ap.parse_args()

    fp = fingerprint(READOUT_DIR)
    if fp != FP_WANT:
        raise SystemExit(f"자 지문 {fp} ≠ {FP_WANT} ({READOUT_DIR}) — 릴리즈 readings 와 다른 자다")
    preds = parse_preds(a); tags = [t for t, _ in preds]
    for tag, ck in preds:                                                  # 2026-09-27: kind=ar 는 폴더를 만들기 전에 거부
        try:
            st_ = torch.load(ck, map_location="cpu", weights_only=False, mmap=True)
        except Exception:
            st_ = torch.load(ck, map_location="cpu", weights_only=False)
        refuse_ar(tag, str((st_.get("arch") or {}).get("kind") or "").strip() or "default")
        del st_
    name = a.name or a.preset
    if not name:
        raise SystemExit("--preds 를 쓰면 --name 을 줄 것")
    if a.xcheck is None:
        a.xcheck = [t for t in ("v11_e10", "ariel_prefix_ep45", "ariel_full_ep40") if t in tags] if a.validate else []
    a.xcheck = [t for t in a.xcheck if t in tags]
    out = Path(a.out) if a.out else CACHE_ROOT / (f"v11readout_{name}" + (f"_smoke{a.limit}" if a.limit else ""))
    stamp = time.strftime("%Y%m%d-%H%M%S")

    rows, L, inf, hid, obj, tdir = select_all()
    if a.limit:
        pick = pick_limit(rows, a.limit)
        rows = [rows[i] for i in pick]; L, inf, hid, obj, tdir = L[pick], inf[pick], hid[pick], obj[pick], tdir[pick]
    n = len(rows)
    vids = [r["video_id"] for r in rows]
    sig = hashlib.sha1(("\n".join(vids) + "|" + ",".join(f"{t}={p}" for t, p in preds) + f"|{fp}|attn={not a.no_attn}").encode()).hexdigest()[:16]
    arrays = array_specs(tags, n, attn=not a.no_attn)
    print(f"[data] {n:,} clip (물체 {int(obj.sum()):,} · 빈 장면 {int((~obj).sum()):,}) · "
          f"timing {dict(zip(*np.unique([r['timing'] for r in rows], return_counts=True)))}"
          + (f"  ⚠️ --limit {a.limit}" if a.limit else ""), flush=True)
    print(f"[pred] {len(preds)} 개: {tags}", flush=True)
    print(f"[out]  {out}  ({sum(int(np.prod(s)) * np.dtype(d).itemsize for s, d in arrays.values())/2**30:.2f} GiB)", flush=True)

    if out.exists() and any(out.iterdir()):
        if not a.resume:
            raise SystemExit(f"{out} 가 이미 있다 — 덮어쓰지 않는다. 이어 쓰려면 --resume, 새로 쓰려면 --out/--name 을 바꿀 것")
        info = json.loads((out / "meta.json").read_text())
        if info["signature"] != sig:
            raise SystemExit(f"서명 불일치 ({info['signature']} vs {sig}) — clip·predictor·자·attn 설정이 다르다")
        done = np.load(out / "done.npy")
        print(f"[resume] 완료 {int(done.sum()):,}/{n:,}", flush=True)
    else:
        out.mkdir(parents=True, exist_ok=True)
        with (out / "index.csv").open("w", newline="") as f:              # 로더는 index.csv 만 읽는다
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        px = (L.astype(np.float64) + 1) * RES
        gt_tub = px.reshape(n, TA, 2, 2).mean(2).astype(np.float32)
        np.savez(out / "meta.npz", **{k: np.array([r[k] for r in rows]) for k in KEEP + ["timing", "sel"]},
                 obj=obj, truth=L, in_frame=inf, hidden=hid, travel_dir=tdir, gt_tub_px=gt_tub)
        for k, (shape, dt) in arrays.items():
            m = np.lib.format.open_memmap(out / f"{k}.npy", mode="w+", dtype=dt, shape=shape)
            if dt == np.float32:
                m[:] = np.nan
            m.flush(); del m
        np.lib.format.open_memmap(out / "done.npy", mode="w+", dtype=np.uint8, shape=(n,)).flush()

    # predictor 가중치를 /dev/shm 에 한 번만 풀어 둔다 (rank 마다 NFS 에서 opt 상태까지 읽지 않게)
    stage = Path(f"/dev/shm/te_v11readout_{os.getpid()}_{stamp}"); stage.mkdir(parents=True)
    pinfo = []
    try:
        for tag, ck in preds:
            st = torch.load(ck, map_location="cpu", weights_only=False)
            arch = dict(st.get("arch") or {}); kind = str(arch.get("kind") or "").strip() or "default"
            torch.save(dict(predictor=st["predictor"], arch=arch, kind=kind), stage / f"{tag}.pt")
            pinfo.append(dict(tag=tag, ckpt=ck, kind=kind, arch=arch, format=st.get("format"), epoch=st.get("epoch"),
                              step=st.get("step"), sha1=sd_sha(st["predictor"])))
            print(f"  {tag:<11} kind={kind:<8} epoch={st.get('epoch')} sha1={pinfo[-1]['sha1']}  {ck}", flush=True)
            del st
        info = dict(signature=sig, n_clip=n, preds=pinfo, arrays=list(arrays), attn=not a.no_attn,
                    decoder_fp=fp, decoder_dir=str(READOUT_DIR), n_samples=NS, context_samples=CS, stride=3,
                    mask_index=0, dtype="bfloat16 (no autocast)", cell_px=CELL, norm=RES,
                    rd_cols=["x", "y", "logit", "top1", "mass3_gt"], limit=a.limit,
                    thr_val_fpr5={r: json.loads((READOUT_DIR / "summary.json").read_text())["reps"][r]["attn"]["thr_val_fpr5"]
                                  for r in ("p", "z", "h")},
                    bias_px=json.loads((READOUT_DIR / "attn_bias_px.json").read_text()),
                    script="z_research/scripts/analysis/te_v11_readout.py", status="running", stamp=stamp)
        if (out / "meta.json").exists():                                  # 재개: 처음 실행 기록은 남긴다
            old = json.loads((out / "meta.json").read_text()); info["runs"] = old.get("runs", []) + [stamp]
            info["first_stamp"] = old.get("first_stamp", old.get("stamp"))
        else:
            info["runs"] = [stamp]; info["first_stamp"] = stamp
        (out / "meta.json").write_text(json.dumps(info, indent=1))
        spec = dict(out=str(out), preds=preds, stage=str(stage), arrays=list(arrays), stamp=stamp)
        t0 = time.time()
        if np.load(out / "done.npy").all():
            print("[extract] 이미 전부 완료 — 모델을 띄우지 않는다", flush=True)
        else:
            mp.spawn(_worker, args=(a.gpus, a, n, spec), nprocs=a.gpus, join=True)
            print(f"[extract] 끝 {(time.time()-t0)/60:.1f}분", flush=True)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    done = np.load(out / "done.npy")
    info["status"] = "complete" if done.all() else f"partial {int(done.sum())}/{n}"
    (out / "meta.json").write_text(json.dumps(info, indent=1))
    print(f"→ {out}  ({info['status']})", flush=True)
    if a.validate:
        validate(out, stamp)


if __name__ == "__main__":
    main()
