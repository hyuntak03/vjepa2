# Ariel predictor 체크포인트 · 자기회귀 (AR) 이식 — 시작점 (2026-09-27)

> `z_training/ariel/README.md` (09-23 ~ 09-24 기록) 는 폴더 정리 때 사라졌다 (`z_training/ariel/` 은 gitignore 라 복구 불가).
> 그 문서의 결론 (ep5 ~ ep43 epoch sweep, prefix 구조 검사) 은 `configs/protocols/models.md` 의 옛 `vith_ariel_bc_ep*` 항목과
> `z_research/TrainingEffects/` 에 남아 있다. 이 문서가 **지금 판**의 정본이다. 레포 규칙은 `CLAUDE.md`.

## 1. 받은 것 — 세 팔 (`z_training/ariel/<run>/latest.pt`)

셋 다 **encoder 는 릴리즈 ViT-H 그대로 (frozen)**, predictor 는 **scratch**. 데이터 · 창 · (C, K) 분포 · lr 이 같고 **predictor 규칙만** 다르다.
공통: SSv2 `train_min48` + K400 320p `train_min96` @ 12 fps, 48 프레임 (24 블록), `prefix_window` 문맥 {2,4,8,16} × 예측 {1,2,4,8} 블록,
lr 3e-4 (warmup 1, cosine → 1e-6), batch 28/rank (A6000 48 GB). 형식 `vjepa_frozen/predictor-only/v1` (`predictor` 160 키 + `arch` + `config`).

| 폴더 | 레지스트리 (`models.md`) | 팔 | `arch.kind` | epoch · step | 학습 loss | 문맥 입력 |
|---|---|---|---|---|---|---|
| `full_attn_future_pred_1e_5` | `vith_ariel_full_ep40` | **A** — full attention + mask token (릴리즈 위상) | `oneshot` | 40/40 · 35,840 | 0.5761 | online `encoder` (창 한 번) |
| `block_causal_future_only_1e_5` | `vith_ariel_prefix_ep45` | **B** — prefix 마스크 + mask token | `prefix` | 45/45 · 36,064 | 0.5836 | online `encoder` (창 한 번) |
| `ar_future_1e_5` | `vith_ariel_ar_ep18` | **C** — **자기회귀**, mask token 없음 (V-JEPA 2-AC 레시피, action 없음) | `ar` | ⚠️ **18/40** · 16,128 (학습 중) | 1.047 (tf + rollout — 다른 팔과 비교 불가) | **블록별** `target_encoder` |

- A 는 09-24 `z_training/ariel/README.md` §4-c 가 "attention 규칙 효과를 가르려면 필요하다" 고 적은 **대조 팔**이다. 이제 A vs B 로 규칙 하나만 다른 비교가 된다 (같은 데이터 · 같은 예산).
- ⚠️ C 는 A · B 와 **규칙 하나만 다르지 않다** — 인코딩 단위 (창 한 번 vs 블록마다) · 문맥 encoder (online vs EMA) · 입력 (mask token vs 되먹임) 이 같이 바뀐다 (Ariel `natural_ar.yaml` 머리말도 같은 경고).
- 옛 `block_causal_future_only_1e_5_ep_{5,19,30,41,43}` 은 지워졌다. B 의 ep45 는 그 run 의 마지막 epoch 다 (ep43 과 같은 파일이 아니다).

## 1-b. 2026-09-29 추가 (auto_research 세션) — 19:00 사용자가 가중치를 다시 옮김: `ar_future_1e_5/latest.pt` = **AR ep40 최종**, `context_full_ar_1e_5/latest.pt` = **ctx_ar ep16 (학습 중)**. 고정 사본 `ariel_eval/ckpt/ar_ep40.pt` · `ctx_ar_ep16.pt`. 아래 문단은 17:36 판 (ctx_ar 자리에 AR ep40 이 있던 때) 의 기록이다.

- `z_training/ariel/context_full_ar_1e_5/latest.pt` (09-29 17:36 복사) = **`nat_ar_scratch` 의 epoch 40 최종** (arch.kind `ar`, `type_embed` 키 없음, config.folder `nat_ar_scratch`, step 35,840 · loss 1.04417 = `z_ariel/vjepa2_train_code_20260929_ctx_ar/z_training/runs/nat_ar_scratch/metrics.jsonl` epoch 40 행). 즉 팔 C 의 완료본이다 (ep38 사본과 별개).
- 진짜 팔 C' **ctx_ar** (`nat_ctx_ar_scratch`, kind `ctx_ar`: 문맥 = 창 단위 z (online encoder, prefix 와 같음) + 미래만 블록별 LN(h) 되먹임 + type embedding) 은 09-29 기준 **epoch 15/40 학습 중** — 체크포인트 아직 없음. 코드 스냅샷 `z_ariel/vjepa2_train_code_20260929_ctx_ar/` (README_SNAPSHOT.md).
- 이식 (09-29): `src/models/rollout_predictor.py` · `app/vjepa_frozen/ar.py` 를 0929 판으로 교체 (0927 판과 동일했음), `app/vjepa_frozen/utils.py` · `z_training/harness/resolve_train.py` kind 목록에 `ctx_ar`, `analysis/predictors/__init__.py` 에 `ctx_ar` factory + `is_ctx_ar()`, `z_training/tests/ctx_ar_cpu_test.py` (통과) · `ar_cpu_test.py` (통과). 표준 채점기 (`ar_scoring.py`, `evals/…/eval.py`) 의 ctx_ar 분기는 **아직 없다** (문맥 encoder 가 다르다 — `run_val_ctx_ar` 를 참고해 붙여야 한다). 장면 판독 (`auto_research/scripts/e_scene_ar.py`) 은 `SCENE_AR_EXTRA=CX=<ckpt>` 로 ctx_ar 을 받는다.

## 2. 무엇을 이식했나 (`z_ariel/vjepa2_train_code_20260927` → 우리 레포)

| 파일 | 내용 | 방식 |
|---|---|---|
| `src/models/rollout_predictor.py` | `mask_mode="ar"` (`forward_seq` 블록 인과 · 출력 LN, `rollout` 되먹임, `forward(x, ar=…)` 학습용), `prefix_full` | 0927 그대로 (우리 판 = 0923 이었다) |
| `src/models/utils/modules.py` | `PrefixSpec(future_causal=…)` | 0927 그대로 |
| `app/vjepa_frozen/ar.py` | `encode_blocks` · `ar_losses` (tf + rollout) · `run_val_ar` | 0927 그대로 (신규) |
| `app/vjepa_frozen/train.py` | `is_ar` 분기 · `train_step_ar` · val dispatch · CPU affinity 복원 | **3-way 병합** (0923 기준) — 우리 `sync_masks` · `compile_encoders` 기본 false 유지. **AR 은 `meta.sync_masks: true` 필수** (없으면 죽는다) |
| `app/vjepa_frozen/utils.py` | kind 목록에 `prefix_full` · `ar` | 3-way 병합 — 우리 `mret` · `predictor_init_checkpoint` 유지 |
| `analysis/predictors/__init__.py` | 채점 레지스트리에 `prefix_full` · `ar`, `is_ar()` | 추가 |
| **`analysis/predictors/ar_scoring.py`** | **AR 채점 경로** (아래 §3) | 신규 — 인코딩 함수는 학습 코드 것을 그대로 import |
| `evals/world_model_analysis/eval.py` | `run_surprise` (v11 등 고정 창) · `run_surprise_intphys1` (Garrido 창) 에 AR 분기, probing 은 AR 이면 NotImplementedError | 분기 추가 (비-AR 경로 무변경 — §4 회귀 검사) |
| `evals/world_model_analysis/attn_probe.py` · `app/vjepa_frozen/val.py` | AR 이면 명확한 에러 | 가드 |
| `z_training/tests/ar_{cpu,gpu}_test.py` | Ariel 정합성 검사 (CPU: 인과 · split=dense · rollout=TF 첫 스텝 · checkpointing · grad / GPU: VRAM · step 시간) | 그대로 |
| `z_training/harness/val_ckpt.py` | 체크포인트만으로 in-loop val 재현 | 신규 |
| `configs/training/natural_{prefix,ar}.yaml` · `resolve_train.py` · `datasets.md` · `build_video_index.py` · `video_dataset.py` | 학습 config · AR 가드 · SSv2 인덱스 · 로더 수정 | (진행 중 — §6) |

**옮기지 않은 것** (Ariel 서버 전용): `paths.env/paths.py/env.sh` · `/nas2` 경로 · `sbatch.sh` 의 partition/QOS · `tmux_train.sh` · `model_loaders.py` env 기본값 ·
`analysis/intphys2/model.py` 의 0927 판 (우리 판이 체크포인트 `arch.kind` 를 읽어 더 엄격하다) · `models.md` 의 `${CKPT_ROOT}` 개편.

## 3. AR 채점 정의 (`analysis/predictors/ar_scoring.py`)

```
h_b   = LN(target_encoder(블록 b 의 2 프레임))            블록마다 따로 (블록 b 는 블록 b 만 본다 — 시간 누수 없음)
p_ar  = predictor.rollout(h[블록 0..C-1], K)              문맥 C 블록 → K 블록 자기회귀 (자기 예측을 되먹임)
S_ar  = mean_{블록 C..C+K-1, 토큰, D} |p_ar − h|           ← 주지표 (Ariel run_val_ar 과 같은 식)
보조  S_tf = mean |forward_seq(h[0..nb-2])[C-1..] − h[C..]|   (surprise.ar_tf: true 로 켠다; 쌍 안에서 p 가 다르다 — 보조로만)
```
- 표준 채점기에서: `surprise_c16t32` → 문맥 8 블록 · rollout 8 블록 (학습 분포 안). `intphys1_sliding` → 창마다 C/2 블록 문맥에서 (W−C)/2 블록 rollout,
  min-over-C · 시작점 평균 · 쌍 비교는 그대로. ⚠️ skip2_w32 의 C=4 프레임 칸은 rollout **14 블록** (학습 최대 8) — 분포 밖. `dump_windows` 로 C 별로 본다.
- ⚠️ **타깃 공간이 다르다.** 표준 = LN(target_encoder(창 전체))[미래] (양방향 시간 attention), AR = 블록별. **surprise 크기 · margin · 복사 기준선은 다른 kind 와 비교 불가, 쌍 정확도만 비교한다.**
- matched pair 는 문맥 픽셀이 같아 p_ar 이 쌍 안에서 같다 (블록별 인코딩이라 미래가 새지 않는다).
- 정밀도: 표준 채점 관례 (fp32 가중치 + fp16 autocast) 로 돈다. AR 은 bf16 autocast 로 학습했다 — fp16 넘침은 `assert_finite` 가 잡는다. 학습과 같은 정밀도로 보려면 `SET="model.autocast=bfloat16"`.
- probing · readout (`attn_probe`, `te_v11_readout`, `te_v3_readout`) 은 AR 의 'p' 를 아직 정의하지 않아 에러를 낸다.

## 4. 검증 (2026-09-27, vll5 RTX 4090)

| 검사 | 결과 |
|---|---|
| Ariel CPU 정합성 (`z_training/tests/ar_cpu_test.py`) | 전부 통과 — 인과 Δ(앞 블록)=0, split = dense Δ=0, rollout[0] = TF[C-1] Δ=0, rollout 은 미래 GT 와 무관 Δ=0, forward(ar) = 개별 호출 Δ=0, 출력 LN, checkpointing 경로 Δ=0, grad 30/40 (mask token 10 개 제외), release 규약 차단 |
| prefix 구조 회귀 (`check_prefix_predictor.py --ckpt …/block_causal_future_only_1e_5/latest.pt`) | 통과 — 새 `modules.py` 로도 인과 불변 6.2e-6, split = dense 0.0, 독립 복원과 상대 3.2e-7 |
| 채점기 비-AR 회귀 (release `vith`, v11_split_test 64 clip) | 기존 하네스 per_video_surprise 와 **max\|Δ\| = 0.0** (비트 동일) |
| AR 표준 채점 스모크 (v11 16 block · IntPhys1 w32 8 영상) | 돈다, 유한값, 쌍 정확도 100 % / 87.5 % |
| **in-loop val 재현** (`val_ckpt.py`, Ariel 학습 config 그대로, IntPhys1 dev 60 쌍 · 문맥 32 → 16 프레임) | 아래 표 |

| 체크포인트 | 우리 acc (eager / compile) | Ariel 기록 acc | surprise_pos 우리 / Ariel | margin (Ariel) |
|---|---|---|---|---|
| full ep40 | **46.67 / 46.67** | **46.67** (O1/O2/O3 45/45/50 — compile 판은 칸까지 같다) | 0.5730 / 0.5773 | 9.3e-4 |
| prefix ep45 | 48.33 / 50.00 | 43.33 | 0.5784 / 0.5826 | 4.8e-4 |
| AR ep18 (rollout / tf) | 75.0 / 76.7 · tf 68.3 / 73.3 | ep18 기록 없음 (ep17 73.3 · tf 73.3) | 0.4996 / (ep17 0.5059) | 8.3e-3 (ep17) |

- 가중치 로드는 strict 로 전부 맞고 평가 함수는 학습 루프 것 그대로다. full 은 정확도 · 칸별 정확도가 같다.
- ⚠️ **surprise 절대값이 우리 쪽이 일정하게 ~0.004 (0.7 %) 낮다** — compile 을 켜도 그대로라 코드가 아니라 **입력 프레임** 차이로 본다
  (Ariel val 프레임 `…/benchmarks/IntPhys1` vs 우리 `IntPhys1_dev_frame_png`, 같은 패턴 · 같은 stride 2). **미결 — Ariel 쪽 프레임 출처 확인 필요.**
  prefix 의 5 pt (3 쌍) 차이는 margin 이 5e-4 로 그 오프셋보다 작은 동점 근처 쌍이 뒤집힌 것으로 읽는다.

## 5. 표준 채점 결과

### 5-1. IntPhys1 dev — Garrido A.8 (`intphys1_sliding`, 2026-09-27)

180 쌍, Filtered (시작점마다 C 최소) → AvgSurprise → property macro. 창 (w16 · w32) 마다 실행을 나눴다.
100 프레임이라 격자에서 실제로 서는 칸은 셋이다 (skip10 · skip5_w32 는 프레임 부족). 값은 `garrido_rescore.collect` 로
`per_window.json` 에서 다시 계산했다 (`z_training/ariel_eval/intphys1_four_models.json`).

| 모델 | skip2_w32 | skip2_w16 | skip5_w16 | **best** | best 칸 O1 / O2 / O3 |
|---|---:|---:|---:|---|---|
| `vith` (릴리즈) | **88.89** | 83.33 | 60.56 | skip2_w32 **88.89** | 85.0 / 96.7 / 85.0 |
| `vith_ariel_full_ep40` (팔 A) | **71.67** | 65.56 | 54.44 | skip2_w32 **71.67** | 75.0 / 80.0 / 60.0 |
| `vith_ariel_prefix_ep45` (팔 B) | **70.00** | 63.89 | 53.33 | skip2_w32 **70.00** | 76.7 / 71.7 / 61.7 |
| `vith_ariel_ar_ep18` (팔 C) ⚠️ | 87.78 | **90.56** | 77.78 | skip2_w16 **90.56** | 86.7 / 96.7 / 88.3 |

- **A vs B (attention 규칙 하나만 다름): 71.67 vs 70.00 — 차이 없음** (3 쌍). 세 칸 모두 같은 순서 · 1.1–1.7pt 차.
- **Ariel 셋 vs 릴리즈는 비교 대상이 아니다** — scratch · SSv2+K400 · 40–45 epoch 대 VideoMix22M 사전학습 predictor.
- ⚠️ **AR 의 +19pt (vs A · B) 는 predictor 몫으로 읽을 수 없다.** 문맥 encoder (EMA 하나) · 인코딩 단위 (블록별) ·
  **타깃 공간** (블록별 LN(target_encoder)) 이 같이 바뀌었다 (§3). 블록별 타깃이 위반을 더 잘 드러내는 공간일 수 있다.
  분리하려면 (a) 블록별 복사 기준선 `copy_blk` (b) 입력은 블록별 · 타깃만 표준 (창 전체 인코딩) 인 변형 — **미실행**.
- ⚠️ AR 은 학습 중간 (18/40 epoch). 학습 범위 (문맥 {2,4,8,16} 블록, rollout ≤ 8) 밖 칸이 Filtered min 에 섞인다
  (w32 의 C = 4·8·12 프레임 → rollout 14·12·10 블록).
- best 는 같은 격자에서 고른 값이라 held-out 추정이 아니다 (CLAUDE.md §1-3). 180 쌍의 이항 SE ≈ 3pt —
  릴리즈 88.89 vs AR 90.56 은 잡음 안이다.
- v11_split_test 채점은 사용자가 범위를 IntPhys1 로 줄여 보류.

### 5-1b. IntPhys1 — AR ep36 (2026-09-28, `latest.pt` 가 epoch 36 판으로 덮임 → 레지스트리 `vith_ariel_ar_ep36`)

| 모델 | skip2_w32 | skip2_w16 | skip5_w16 | best | O1 / O2 / O3 |
|---|---:|---:|---:|---|---|
| AR ep18 | 87.78 | **90.56** | 77.78 | skip2_w16 90.56 | 86.7 / 96.7 / 88.3 |
| **AR ep36** | 90.00 | **92.78** | 77.78 | skip2_w16 **92.78** | 90.0 / 98.3 / 90.0 |

- ep18 → ep36 로 +2.2pt (4 쌍). 180 쌍 SE ≈ 3pt 라 잡음 안이다. 같은 격자 · 같은 규칙 (§5-1).
- 블록 공간 복사 기준선 83.33 (§5-2) 대비 +9.4pt.

### 5-1c. AR ep38 — 세 벤치 (2026-09-28, 고정 사본 `z_training/ariel_eval/ckpt/ar_ep38.pt`, `run_ar_all.sh EP=38`)

| 판 | IntPhys1 | IntPhys2 | IntPhysGenV11 |
|---|---:|---:|---:|
| AR ep18 | 90.56 | 51.19 | 83.2 |
| AR ep36 | 92.78 | — | 83.8 |
| **AR ep38** | **92.78** (칸 · O1/O2/O3 까지 ep36 과 같음) | **50.79** (w16/C12; 18 칸 48.6–50.8) | **83.7** [83.0, 84.3] |

- ep36 → ep38: IntPhys1 surprise 원값은 창마다 평균 5e-4 · 최대 7e-3 다르지만 180 쌍 승패는 그대로. v11 −0.1pt. 학습 끝물이라 거의 움직이지 않는다.
- IntPhys 2 는 세 판 모두 우연 수준 (w16 최고).

### 5-1d. ctx_ar ep18 — IntPhys1 · v11_split_test (2026-09-29 22:02, vll6 8 GPU, auto_research 세션; 채점기 ctx_ar 분기 `ar_scoring.py:ctx_ar_from_blocks` 신설)

고정 사본 `ariel_eval/ckpt/ctx_ar_ep18.pt` (21:37 수령, step 16128, loss 1.0635). 레지스트리 `vith_ariel_ctx_ar_ep18`. 스크립트 `ariel_eval/run_ctx_ar_ip1_v11.sh`.

| 모델 | skip2_w32 | skip2_w16 | skip5_w16 | best | O1 / O2 / O3 (best 칸) | v11_split_test |
|---|---:|---:|---:|---|---|---:|
| ViT-H 릴리즈 | 88.89 | 83.33 | 60.56 | 88.89 | 85.0 / 96.7 / 85.0 | 75.7 |
| full ep40 | 71.67 | 65.56 | 55.00 | 71.67 | 75.0 / 80.0 / 60.0 | 78.1 |
| prefix ep45 | 70.00 | 63.89 | 54.17 | 70.00 | 76.7 / 71.7 / 61.7 | 79.5 |
| AR ep18 / ep38 | 87.78 / 90.00 | 90.56 / 92.78 | 77.78 / 78.33 | 90.56 / 92.78 | — / 90.0 / 98.3 / 90.0 | 83.2 / 83.7 |
| **ctx_ar ep18** | **60.00** | **64.44** | **53.33** | **64.44** | 66.7 / 58.3 / 68.3 | **61.0** |

- ctx_ar ep18 은 같은 epoch 의 AR ep18 (90.56 / 83.2) 보다 **훨씬 낮고**, 릴리즈 · 복사 (85 / 73) 아래, full/prefix 와 같은 60 대다. v11 은 12 조건 전부 낮다 (static_visible 77.8 — AR 100, prefix 81; moving 팔 52–57).
- 읽기 (1 차): 문맥을 창 단위 z 로 바꾸면 AR 의 "정지 내용 = 복사" 이점이 사라지고 (장면 판독 §2-7 과 같다), 되먹임 rollout 은 그대로 불안정 → surprise 기준으로는 학습 중 판이 복사보다도 못 한 미래를 낸다. `auto_research` presence 검정에서 ctx_ar 이 빈 문맥에서도 "물체 있음" 미래를 고른 편향 (empty 쌍 21 %) 과 맞물린다.
- ⚠️ 학습 중 (18/40). 40 epoch 판으로 다시 잰다. in-loop val (Ariel, 60 쌍, 문맥 32 → rollout 16 프레임) 은 ep15 에서 .72 였다 — 프로토콜이 달라 직접 비교 불가.
- ⚠️ skip2_w32 의 C=4 프레임 칸은 rollout 14 블록 (학습 최대 8) — AR 과 같은 분포 밖 칸.

### 5-2. IntPhys1 — AR 공간 복사 기준선 `copy_blk` (2026-09-27, auto_research 요청)

`copy_blk = L1(hb_{Tc−1}, hb_u)` — 문맥 마지막 블록의 블록별 LN(target_encoder) 을 미래 모든 블록에 복사, **AR 과 같은 블록별 표적**에 대고 잰다.
같은 격자 · Filtered · property macro. `te_intphys1_score.py --preds ariel_ar_ep18 --out_tag ip1score_ar_copyblk`
(`z_research/TrainingEffects/ip1score/ip1score_ar_copyblk_scores.json`).

| 방법 (표적) | skip2_w16 | skip2_w32 | skip5_w16 | best |
|---|---:|---:|---:|---|
| AR ep18 (블록별) | **90.56** | 87.78 | 77.22 | skip2_w16 90.56 |
| **copy_blk** (블록별) | **83.33** | 80.56 | 72.78 | skip2_w16 83.33 |
| 릴리즈 (창 전체) | 83.33 | **88.89** | 60.56 | skip2_w32 88.89 |
| copy LN(z) (창 전체) | 73.89 | 85.00 | 63.33 | skip2_w32 85.00 |

- **AR − copy_blk = +7.2 / +7.2 / +4.4pt** (칸별). 릴리즈 − copy (창 전체 공간) 는 +9.4 / +3.9 / −2.8pt.
  best 끼리는 AR +7.2 (90.56 vs 83.33), 릴리즈 +3.9 (88.89 vs 85.00).
- **블록별 공간은 복사만으로 83.33 이다** — AR 의 90.56 중 상당 부분이 공간 몫이다. "AR 이 물리를 더 안다" 로 읽지 않는다.
  복사 위 여유 (+7.2) 는 릴리즈 (+3.9) 보다 크지만, 180 쌍 (SE ≈ 3pt) 에서 두 여유의 차 3pt 는 잡음 안이다.
- 이 스크립트의 AR skip5_w16 77.22 는 하네스 값 77.78 과 1 쌍 다르다 (배치 구성 차이, 스크립트 머리말). 나머지 두 칸은 같다.

### 5-2b. v11_split_test — AR ep18 (`surprise_c16t32`, 10,752 쌍, 2026-09-27 실행 · 09-28 집계)

산출물 `z_research/IntPhysGenV11/exp_results/surprise_c16t32__v11_split_test_vith_ariel_ar_ep18/` (GPU 1 장, 백업 `…_run0927_1gpu/`).
칸별은 `training_effects_scores.py` 의 함수 그대로 (`z_training/ariel_eval/v11_ariel_cells.json`), 전체값은 summary.json 과 일치 (83.1938).

| 모델 | overall [95% CI, block bootstrap] | 6조건 | vanish | shape | color |
|---|---|---:|---:|---:|---:|
| 릴리즈 ViT-H | 75.8 [75.1, 76.6] | 73.3 | 87.8 | 73.0 | 70.7 |
| **AR ep18** | **83.2 [82.6, 83.8]** | 80.6 | 99.0 | 79.5 | 76.4 |
| (옛 prefix ep43) | 79.4 [78.7, 80.2] | 73.1 | 91.0 | 78.5 | 72.5 |

가림 타이밍 × 운동 (세 위반 합침):

| 모델 | vis static | vis flat | vis ramp | early static | early flat | early ramp | mid static | mid flat | mid ramp | late static | late flat | late ramp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 릴리즈 | 92.3 | **91.1** | 79.8 | 78.7 | **82.3** | 75.2 | 76.2 | **83.7** | 74.3 | 57.4 | 59.4 | 59.6 |
| AR ep18 | **99.9** | 85.0 | **83.4** | **99.7** | 81.7 | **78.5** | **99.3** | 78.9 | **76.8** | **83.4** | **69.1** | **62.7** |

- **오른 자리는 정지 장면과 vanish 다** — 정지 네 칸 99–100 (late 83.4 vs 57.4), vanish late 95.8 vs 56.7,
  vanish 방향 물체→빈 98.4 / 빈→물체 99.6 (릴리즈 78.0 / 97.6 — 비대칭이 거의 사라짐).
- **등속 (flat) 이동은 릴리즈보다 낮다** (visible 85.0 vs 91.1, mid 78.9 vs 83.7). 등가속 (ramp) 은 +2–4pt.
- ⚠️ 정지 장면은 **복사가 정답인 칸**이고, AR rollout 은 ~3 스텝 뒤 복사 비슷한 토큰으로 흐른다 (auto_research
  `AR_PREDICTOR_2026-09-27.md`). 블록별 표적 공간에서의 복사 기준선 (`copy_blk`) 은 **사용자 지시로 v11 에서는 돌리지 않았다**
  (09-28) — 그래서 이 +7.4pt 중 predictor 몫과 표적 공간 몫을 가를 수 없다. IntPhys1 에서는 공간 몫이 컸다 (§5-2).
- 표적 공간이 달라 릴리즈와는 쌍 정확도만 비교한다 (§3).

**AR ep36 (2026-09-28, GPU 8 장, `surprise_c16t32__v11_split_test_vith_ariel_ar_ep36/`)** — overall **83.8 [83.1, 84.4]** (ep18 83.2, +0.6pt).
full · prefix 도 같은 날 채점: full-attn **78.1 [77.3, 78.8]** · block-causal **79.5 [78.8, 80.3]**. 전부 summary.json 과 재계산 일치.

| 모델 | vis 정지 / 등속 / 등가속 | late 정지 / 등속 / 등가속 | vanish · shape · color |
|---|---|---|---|
| full-attn ep40 | 81.4 / 92.7 / 83.1 | 51.9 / 59.8 / 62.4 | 89.8 · 76.6 · 71.8 |
| block-causal ep45 | 81.4 / **93.6** / **89.2** | 51.2 / 62.7 / 61.8 | 91.1 · 78.7 · 72.6 |
| AR ep18 | 99.9 / 85.0 / 83.4 | 83.4 / 69.1 / 62.7 | 99.0 · 79.5 · 76.4 |
| AR ep36 | **100.0** / 87.2 / 84.9 | 82.6 / 67.2 / 61.6 | 97.9 · 80.3 · 77.8 |

- ep18 → ep36 은 가림 없는 이동 (등속 +2.2, 등가속 +1.5) 에서 조금 오르고 나머지는 거의 그대로다. 패턴 (정지 · vanish 강, 이동 약) 은 같다.

### 5-3. IntPhys 2 Main — AR ep18 (2026-09-27, `analysis/intphys2` 하네스에 AR 분기 추가)

506 쌍, 창 16/32/48 × C = 창 × {¼ … ⅞}, growing prefix, 열마다 최고 (논문 Table 2 규칙) —
`z_research/scripts/analysis/intphys2_grid_best.py` (`z_training/ariel_eval/intphys2_ar_vs_vith.json`).

| 모델 | Easy | Medium | Hard | **Overall** | Overall 설정 18 칸 범위 |
|---|---:|---:|---:|---:|---|
| AR ep18 | 57.69 | 52.50 | 54.76 | **51.19** (w16/C8) | 47.83 – 51.19 |
| 릴리즈 ViT-H | 57.69 | 57.50 | 55.36 | **54.35** (w48/C24) | 50.20 – 54.35 |
| 논문 V-JEPA 2-h | 54.00 | 58.50 | 59.38 | 57.51 | — |

- **AR 은 18 칸 전부 우연 수준**이다 (최고 51.19). IntPhys1 에서 릴리즈와 같던 것이 IntPhys 2 에서는 이어지지 않는다.
- 릴리즈도 우연에서 멀지 않다 — 506 쌍 SE ≈ 2.2pt 에 18 칸 최대를 고르면 귀무에서도 ~53 이 나온다.
- ⚠️ AR 은 `video_batch 8` · 창 배치 64/48/16 (w48 은 32 에서 OOM → 16 으로 재실행), 릴리즈는 `video_batch 1`.
  배치 모양 차이로 쌍 정확도가 ±0.5pt 흔들릴 수 있다 (`run_grid.sh` 주석의 실측). 결론 (우연 수준) 은 바뀌지 않는다.
- ⚠️ w32 · w48 은 rollout 이 학습 범위 (≤ 8 블록) 를 넘는다 (최대 15 · 23 블록). 범위 안인 w16 만 봐도 48.42 – 51.19.
- **세 팔 (2026-09-28, full · prefix 도 같은 격자 · video_batch 8):** full-attn **53.75** (w32/C8) · block-causal **53.56** (w48/C36) ·
  AR ep18 **51.19** · 릴리즈 54.35 — 전부 우연 수준 근처. `z_training/ariel_eval/intphys2_three_arms.json`.

## 6. 학습 (`configs/training/natural_ar.yaml`)

*(학습 쪽 이식 — config · 가드 · SSv2 인덱스 — 이 끝나면 채운다)*

## 재현

```bash
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2; PY=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
$PY z_training/tests/ar_cpu_test.py                                                          # AR 정합성 (CPU)
CUDA_VISIBLE_DEVICES=0 $PY z_research/scripts/analysis/check_prefix_predictor.py --ckpt z_training/ariel/block_causal_future_only_1e_5/latest.pt --device cuda:0
R=z_ariel/vjepa2_train_code_20260927/z_training/runs
CUDA_VISIBLE_DEVICES=0 $PY z_training/harness/val_ckpt.py z_training/ariel/full_attn_future_pred_1e_5/latest.pt --ref $R/nat_full_scratch/metrics.jsonl [--compile]
CUDA_VISIBLE_DEVICES=0 $PY z_training/harness/val_ckpt.py z_training/ariel/ar_future_1e_5/latest.pt [--compile]
GPU_IDS="0" bash z_research/scripts/run.sh surprise_c16t32 v11_split_test vith_ariel_ar_ep18          # 표준 채점 (AR 경로 자동)
GPU_IDS="0" OUTDIR=z_research/Benchmarks/exp_results/intphys1_sliding__intphys1_dev_vith_ariel_ar_ep18_w32 \
  SET="surprise.intphys1.window_sizes=[32] model.window_size=32" bash z_research/scripts/run.sh intphys1_sliding intphys1_dev vith_ariel_ar_ep18
bash z_training/ariel_eval/run_ariel_eval.sh                                                   # 세 팔 전부 (GPU 4 장)
```
