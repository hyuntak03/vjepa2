# 08 — 다시 학습해도 같다: post-FT 넷 · Ariel 세 팔 · 사전학습 8

> **결론** — frozen ViT-H encoder 위에서 predictor 만 다시 학습한 모델 (post-FT 넷, Ariel 의 미래 전용 · 되먹임 세 팔, 팬 증강 post-FT) 도, encoder 부터 다시 학습한 jongseo 8 개도 **점수가 학습 도메인 안에서만 오른다.** 문맥에 없던 운동 변화 (가속 · 감속 · 중력 · 사건) 를 미래에 넣게 됐다고 확인되는 predictor 는 하나도 없다. 이것이 09 장 "Ariel 을 넘어야 한다" 의 출발선이다.
> **근거** — (1) 전이 행렬은 대각만 오른다: v11 post-FT 는 v11 75.83 → 91.35 인데 IntPhys 1 88.89 → 77.22, IntPhys 1 post-FT 는 93.89 인데 v11 78.53 (`z_training/runs/_matrix/MATRIX.md`).
> (2) Ariel 세 팔: EK100 reach R50 은 릴리즈 2.12 → full · prefix 2.17 · AR 1.75 (천장 2.2), K400 팬 가림 띠 permanence 는 .46–.47 (릴리즈 .53), AR 은 자기 되먹임 첫 걸음부터 무너진다 (팬 슬롯 1 −.042) (`auto_research/Archive/AR_PREDICTOR_2026-09-27.md` §2 · `VERIFY_AR_2026-09-27.md`).
> (3) 가속 · 감속을 두 자 모두에서 진실 쪽으로 넣는 predictor 없음 (`z_research/TrainingEffects/v3_motion/MOTION.md`), AR ep40 가속 반영 계수 arc −0.42 · ledge −0.46 (등속 외삽보다도 뒤) (`auto_research/Archive/VERIFY_AR40RD_2026-09-30.md` §1).
> **미결** — 학습은 전부 seed 한 번, Ariel 은 네 조건이 동시에 바뀌고, 주 자 `p@h` 는 p 토큰에서 검증되지 않았다. 그래서 "어떤 학습으로도 안 된다" 는 이 자료로 말할 수 없다. 말할 수 있는 것은 "**지금까지 해 본 학습은** 도메인 밖에서 안 올랐다" 까지다.
> 이 장이 받치는 논문 문장: **P3 (v)** "다시 학습해도 같다 — 미래 전용 · 되먹임 목표 (Ariel 세 팔) 와 post-FT 넷 모두 도메인 안에서만 오르고, 문맥에 없던 운동 변화를 넣게 된 predictor 는 없다."  ·  원재료: G10 · G10b · G10c · G03 · G03b · G02a · G02b · G11 · G13 추출문 → 원 문서는 각 절 끝 `출처`.

> ⚠️ **이 장에서 쓰지 않는 문장** (정본 §7): "post-FT 가 intuitive physics 를 배웠다" (도메인 안 궤적 재생이다, §4) · "action-free 학습으로 해결된다" (Ariel 이 반례다 — 합격선을 넘기 전까지) · "AR 이 물리를 더 안다" (내부 귀속, §5) · "목적함수 때문이다" (discussion 한 문단으로만, §7) · "상태를 이어 간다 / 잃는다" (state 프레임 — 09-30 금지).

---

## 0. 이 장의 용어

| 용어 | 뜻 | 정의 문서 |
|---|---|---|
| **도메인 안 / 밖** | 학습 데이터와 같은 생성기 · 같은 궤적 분포 = 안. v11 post-FT 의 v11 test, IntPhys 1 post-FT 의 IntPhys 1 dev, Ariel 의 SSv2 · K400 팬 은 **안**. EK100 · RollOut_v3 · IntPhys 은 Ariel 에게 **밖** | `z_research/TrainingEffects/README.md` §5 · `auto_research/paper/PAPER_STORY_2026-10-08.md` §2-4 방법 절 2 |
| **복사 기준선** | 문맥 마지막 튜블릿 LN(z) 를 미래 모든 자리에 그대로 둔 "예측 없는" 점수. AR 은 같은 블록별 공간의 `copy_blk` | 03 장 · `auto_research/Archive/PREDICTOR_SHARE_2026-09-29.md` |
| **copy-hard** | 복사가 틀리는 쌍만 모은 부분집합 | `auto_research/Archive/COPY_HARD_SUBSET_2026-09-29.md` |
| **R50 (reach)** | 출처 찾기 판독의 적중이 첫 구간의 절반으로 떨어지는 마지막 관측에서의 거리 (칸 = 토큰 패치 한 칸; RollOut 위치 자 문서의 "1 칸 = 18 px = 물체 반폭" 과 다른 단위) | `auto_research/Archive/I_SCENE_CAUSAL_BOTTLENECK_2026-09-25.md` §4-1 |
| **permanence 비 hid / vis** | K400 인공 팬 + 마지막 k 프레임 가림 띠: 가려졌던 내용의 제자리 복원율 ÷ 보이던 내용의 복원율 (k=4, 0–1 칸) | `I_SCENE` §4-4 · `VERIFY_ISCENE2_2026-09-26.md` §1 |
| **닫힘 (closure)** | 자기 예측을 되먹인 rollout (ar_pA) − 한 번에 예측 (os) 의 출처 적중 차. ≥ 0 이면 되먹여도 안 무너진다 | `I_SCENE` §4-5 |
| **가속 반영 계수 (α_acc)** | 위치 자로 읽은 미래가 등속 외삽 (0) 과 진실 (1) 사이 어디인가. ⚠️ **인용 금지인 "α 분해 (헤지)" 와 이름만 같은 다른 양** — 그 값은 이 장에 없다 | `auto_research/Archive/AR40_READOUT_2026-09-30.md` §0 |
| **p@h / p@p** | h (target) 로 학습한 위치 · 있음 자를 p 에 건 것 / 릴리즈 p 로 학습한 자를 이식한 것. 둘이 같은 방향일 때만 '견고' | `TrainingEffects/v3_motion/MOTION.md` |

---

## 1. 학습한 predictor 목록

같은 encoder (릴리즈 ViT-H, `encoder` → 문맥 · `target_encoder` → 타깃, 둘 다 동결) 위에서 predictor 만 다른 것이 **그룹 A**, encoder 부터 새로 학습한 것이 **그룹 B**. 체크포인트 상태는 2026-10-10 에 파일 목록 · `torch.load` 의 `epoch` 키로 확인했다 (읽기 전용).

### 1-1. 그룹 A — post-FT 넷 (릴리즈 predictor 에서 이어 학습)

| 레지스트리 | 학습 데이터 (clip) | 목표 · 마스크 · attention | 창 | epoch · lr · global batch | 최종 loss | 지금 체크포인트 | 정본 문서 |
|---|---|---|---|---|---|---|---|
| `vith` (기준) | Meta 릴리즈 (VideoMix22M) | multi-block 마스크, full attention | 64f | — | — | 릴리즈 `model.pth` | `configs/protocols/models.md` |
| `vith_pft_v11_e10` | `v11_split_train` 10,752 (v11 block 절반, 가능만, 117 셀 층화) | mean\|p − LN(target)\| (채점과 같은 식), `temporal_prefix` C16 → P16, full attention | 32f stride 3 | 10 · 5e-5 · 64 | 0.582 → 0.480 (`RESULTS` 는 0.475) | `z_training/runs/v11_postft/e1–e10.pt` 있음 | `z_training/RESULTS_2026-09-19.md` |
| `vith_pft_intphys1_e40` | `intphys1_train` 약 3,750 (IntPhys 2019 train, 가능만) | 같은 손실, `window_grid` = Garrido 채점 격자 | 32f stride 2 | 40 · 5e-5 · 32 | 0.564 → 0.508 (`RESULTS` 0.503) | `runs/intphys1_postft/e5–e40.pt` | 〃 |
| `vith_pft_intphys2_e80` | `intphys2_main_split_train` 254 (127 장면, 가능만) | 같은 손실, `temporal_prefix` C ∈ {12…42} | 48f frame_step 10 | 80 (240 step) · 5e-5 · 80 | 0.605 → 0.585 | `runs/intphys2_postft/e5–e80.pt` | `z_training/runs/_matrix/MATRIX.md` |
| `vith_pft_predv1_e15` | `predictor_v1_training` 14,250 (무중력 등속 · 지면 등속 + 가림 k=4 · 10° 쐐기 등가속 a 5–60 cm/s²; held-out 모양 · 색 · 배경 · 속도) | 같은 손실, `temporal_prefix` C16 | 32f stride 3 | 15 · 5e-5 · 64 | 0.586 → 0.474 (`RESULTS` 0.473) | `runs/predictor_v1_postft/e1–e15.pt` | `MATRIX.md` · `configs/training/datasets.md` |

⚠️ 최종 loss 가 `MODELS.md` 와 `RESULTS_2026-09-19.md` 에서 0.005 씩 다르다 (0.480 / 0.475 등). 어느 쪽이 마지막 step 값이고 어느 쪽이 epoch 평균인지 문서에 없다 — 결론에는 영향 없음.
⚠️ `v11_scratch` (같은 데이터 scratch 대조군) 는 config 만 있고 `z_training/runs/` 에 run 폴더가 없다 (`z_training/README.md` 는 대조군으로 적었다).
⚠️ `intphys1_postft` 는 train 장면 학습 · dev 채점 — 장면은 다르나 **같은 생성기**다 (도메인 안). `v11_postft` 의 held-out block 도 **같은 궤적**이다 (§4).

### 1-2. 그룹 A — Ariel (scratch, 실영상 미래 예측, action 없음)

공통: predictor scratch, SSv2 `train_min48` + K400 320p `train_min96` @ 12 fps, 48 프레임 (24 블록), `prefix_window` 문맥 {2,4,8,16} × 예측 {1,2,4,8} 블록, lr 3e-4 cosine (warmup 1 → 1e-6), batch 28/rank. **관측만 (action 없음).**

| 팔 · 레지스트리 | `arch.kind` | 학습 목표 · 입력 | epoch · step | loss | 지금 체크포인트 (10-10 확인) |
|---|---|---|---|---|---|
| 옛 `vith_ariel_bc_ep5 / 19 / 30 / 41 / 43` (+ `_ep5_full`) | `prefix` | 문맥 양방향 · 미래 block-causal, mask token, 미래 전용 | ep 5–43 / 45 | ep30 이후 0.5821 평탄 | ⚠️ **2026-09-27 폴더 삭제** — 레지스트리 항목은 옛 결과 해독용, 실행하면 죽는다. 수치는 옛 파일의 것으로 남기고 ep45 로 옮겨 적지 않는다 |
| 팔 A `vith_ariel_full_ep40` | `oneshot` | full attention + mask token (릴리즈와 같은 위상), 문맥 = online encoder 창 한 번 | 40 / 40 · 35,840 | 0.5761 | `z_training/ariel/full_attn_future_pred_1e_5/latest.pt` = ep40 ✓ |
| 팔 B `vith_ariel_prefix_ep45` | `prefix` | prefix 마스크 + mask token, 미래 전용 (옛 bc 와 같은 run 의 마지막 epoch) | 45 / 45 · 36,064 | 0.5836 | `…/block_causal_future_only_1e_5/latest.pt` = ep45 ✓ |
| 팔 C `vith_ariel_ar_ep18` → ep36 · ep38 · ep40 | `ar` | **자기회귀 되먹임** (V-JEPA 2-AC 레시피의 action 없는 판), mask token 없음, 입력 = 타깃 = **블록별** LN(`target_encoder`), 손실 teacher-forcing + rollout | ep40 / 40 · 35,840 (최종) | 1.044 (tf + rollout — 다른 팔과 비교 불가) | ⚠️ `…/ar_future_1e_5/latest.pt` 는 ep18 → ep36 (09-28 17:22) → ep38 (20:19) → **지금 ep40** (10-10 `torch.load` 확인). 레지스트리 이름 `_ar_ep18` · `_ar_ep36` 으로 돌리면 **ep40 이 돈다**. 고정 사본 `z_training/ariel_eval/ckpt/ar_ep38.pt` · `ar_ep40.pt`. ep18 · ep36 파일은 없다 |
| 팔 C′ `vith_ariel_ctx_ar_ep18` | `ctx_ar` | 문맥 = 창 단위 z (online, prefix 와 같음) + 미래만 블록별 되먹임 + type embedding | ep18 / 40 · 16,128 (09-29 학습 중) | 1.0635 | `…/context_full_ar_1e_5/latest.pt` = ep18 ✓ · 고정 사본 `ctx_ar_ep16.pt` · `ctx_ar_ep18.pt`. **40 epoch 판 결과 문서 없음** |

⚠️ A vs B 는 **attention 규칙 하나만** 다른 대조다. C 는 인코딩 단위 (블록별) · 문맥 encoder (online → EMA) · 입력 (mask token → 되먹임) 이 함께 바뀐다 (`z_training/ARIEL_CHECKPOINTS.md` §1).
⚠️ Ariel 은 릴리즈와 **네 가지가 동시에** 다르다 — attention 규칙 · 데이터 (SSv2 + K400 vs VideoMix22M) · scratch · epoch. 릴리즈 대비 차이를 한 원인에 붙이지 않는다 (`z_research/TrainingEffects/MODELS.md`).
⚠️ 09-29 기록: `context_full_ar_1e_5/latest.pt` 17:36 판은 실은 AR ep40 이었고 19:00 사용자가 다시 옮겼다 (`ARIEL_CHECKPOINTS.md` 1-b). 그 시각의 결과를 ctx_ar 로 읽지 않는다.

### 1-3. 그룹 A — 팬 증강 post-FT (reach 를 늘리려는 학습, auto_research 단계 M · N)

| 단계 · 팔 | 시작점 | 데이터 | 목표 | epoch | run 폴더 (이름으로 대응, 추정) | 정본 |
|---|---|---|---|---|---|---|
| M · T0 | 릴리즈 predictor | K400 40k (`k400_train_40k`), 고정 창 | temporal-prefix 미래 전용 | 3 | `z_training/runs/reach_k400_static/` | `I_SCENE` §5-4 |
| M · T1 | 〃 | K400 40k + **인공 팬 증강** (큰 변위 커리큘럼, DC1(a)) | 〃 | 3 | `runs/reach_k400_pan/` | 〃 |
| N · N0 / N1 | Ariel bc ep43 | 고정 창 / 팬 증강 | 〃 | 3 | `runs/reachN_prefix_static/` · `reachN_prefix_pan/` | `I_SCENE` §5-1 |
| O (mret) | 〃 | 팬 증강 + 조회 보조 모듈 | 〃 | 3 | `runs/reachO_mret_pan/` | ⚠️ **무효** — `mret_beta` 0.494–0.498 (초기 0.5), bias 미학습 → 판정 보류 (`I_SCENE` §5-2) |

### 1-4. 그룹 B — jongseo 사전학습 8 (encoder + predictor 전부 새로)

레시피: V-JEPA 2 Table 10 축약 (16f @ 4 fps, 256, multi-block 8×0.15 + 2×0.7, global batch 3072, cosine lr 6.25e-4 → 1e-6, EMA 0.999 → 1.0), predictor 384d × 12L × 12h (`z_research/TrainingEffects/MODELS.md`). 원본 `/data/jongseo/project/world-model/vjepa2-pretrain/exp/`.

| 레지스트리 | encoder · 데이터 | 진행 | loss |
|---|---|---|---|
| `vitb_k400_e240` | ViT-B · K400 | 72K (14 일 벽에서 멈춤) | 0.574 |
| `vitb_k400_e100` \* | ViT-B · K400 | 30K (90K 스케줄 중간) | — |
| `vitb_synphys_30k` | ViT-B · 합성 물리 50K clip (2D pymunk 30K + 3D pybullet 20K; IntPhys · IntPhysGen 안 씀) | 30K 완주 | 0.505 |
| `vittiny_k400_5k` | Tiny · K400 | 5K | 0.563 |
| `vittiny_synphys_5k` | Tiny · 합성 물리 | 5K | 0.285 |
| `vittiny_k400` | Tiny · K400 | 30K | 0.493 |
| `vittiny_synphys_30k_e225` \* | Tiny · 합성 물리 | 22.5K / 30K (학습 중) | ~0.37 |
| `vittiny_k400ssv2_30k_e225` \* | Tiny · K400 0.77 + SSv2 0.23 | 22.5K / 30K (학습 중) | ~0.57 |

\* = 중간 체크포인트. ⚠️ loss 절대값은 데이터끼리 비교 금지, 학습 입력 16f @ 4 fps 가 채점 시간 간격과 다르다, 그룹 B 는 encoder 까지 달라 **predictor 몫을 분리할 수 없다**.

출처: `z_research/TrainingEffects/MODELS.md` · `z_training/ARIEL_CHECKPOINTS.md` §1 · `configs/protocols/models.md` (`vith_ariel_*` 항목) · `configs/training/datasets.md` · `z_training/RESULTS_2026-09-19.md` §1 · `auto_research/Archive/I_SCENE_CAUSAL_BOTTLENECK_2026-09-25.md` §5 · `z_training/runs/*/` (파일 목록, 10-10)

---

## 2. 전이 행렬 — 대각만 오른다

### 2-1. post-FT 넷 × 네 판 (정본 행렬)

ViT-H, encoder 동결, 각 run 의 마지막 epoch. Δ = 릴리즈 대비. **굵게 = 학습 도메인 (대각).**

| 학습 (ckpt) | v11 test (`v11_split_test` 10,752 쌍) | Predictor_v1 holdout L1 ↓ (1,000 clip, 다른 단위) | IntPhys 1 dev (`skip2_w32/avg`, 180 쌍) | IntPhys 2 test (Main split test 252 쌍, 열별 최고) |
|---|---:|---:|---:|---:|
| 릴리즈 | 75.83 | 0.5854 | 88.89 | 57.94 |
| v11 `e10` | **91.35 (+15.52)** | 0.5182 | 77.22 (−11.67) | 60.71 (+2.78) |
| Predictor_v1 `e15` | 81.63 (+5.80) | **0.4876 (−0.0978)** | 80.56 (−8.33) | 57.94 (0) |
| IntPhys 1 `e40` | 78.53 (+2.71) | 0.5528 | **93.89 (+5.00)** | 60.71 (+2.78) |
| IntPhys 2 `e80` | 79.56 (+3.73) | 0.5839 | 85.56 (−3.33) | **56.75 (−1.19)** |
| *복사 (예측 없음)* | *73.2 [72.4, 74.1]* | — | *85.00* (표준 표적) | *없음* (252 쌍 판) |

출처: 행렬 `z_training/runs/_matrix/MATRIX.md` (칸별 `summary.json` 경로 표 포함, 원 문서와 대조함); 복사 v11 `z_research/TrainingEffects/scores/ANALYSIS.md` (추출 배열), 복사 IntPhys 1 `z_research/TrainingEffects/ip1_copy/IP1_COPY.md` 표 2a. 등급: 행렬은 자동 생성 정본 (등급 문서에 없음), 값은 정본 §재현 에서 10-08 재확인 ✓.

- **어떤 학습셋도 자기 도메인이 아니면 IntPhys 1 을 릴리즈 위로 올리지 못한다.** v11 · Predictor_v1 학습은 IntPhys 1 을 **복사 85.0 아래로** 떨어뜨린다 (77.22 · 80.56).
- v11 은 모든 학습에서 조금씩 오른다 (+2.7 ~ +5.8) — 그러나 그 이득은 copy-correct 쌍 위에서만이다: copy-hard 2,877 쌍에서 ip2 51.0 · ip1 59.5 · pv1 61.5 (릴리즈 48.4) (`COPY_HARD_SUBSET` §2-1).
- IntPhys 2 열의 ±3 pt 는 252 쌍 CI ±6 안 — 이 열은 아무것도 가르지 않는다.

⚠️ IntPhys 2 test 열 (Main split test 252 쌍, 릴리즈 57.94) 은 Main 506 쌍 (릴리즈 54.35) · 논문 Table 2 와 **다른 표본** 이다. 같은 표에서 섞지 않는다 (`MATRIX.md` 단서).
⚠️ IntPhys 2 는 짝 문맥이 픽셀 단위로 같지 않다 (첫 프레임부터 1.7–3.7/255) — v11 · IntPhys 1 과 채점 성격이 다르다.
⚠️ `RESULTS_2026-09-19.md` 는 "Predictor_v1 은 v11 전수 채점 안 함 (192 쌍)" 이라 적었고, 같은 날 나중 실행으로 `MATRIX.md` 가 81.63 을 채웠다. `intphys2_postft` 는 RESULTS 종합에 없다.

### 2-2. 복사 대비 Δ (그룹 A)

| predictor | v11 | p − 복사 (v11) | IntPhys 1 (A.8 최고 칸) | p − 복사 (IntPhys 1 skip2_w32, 표준 표적) |
|---|---:|---|---:|---|
| 릴리즈 | 75.8 | +2.5 [+1.6, +3.4] ⚠️ 색 빼면 **−1.7** [−2.6, −0.7] | 88.89 | +3.9 [−0.6, +8.9] (CI 0 포함) |
| post-FT v11 | 91.4 | +17.9 | 77.22 ↓ | −6.7 (CI 0 포함) |
| post-FT IntPhys 1 | 78.5 | +5.5 ⚠️ 색 빼면 −0.4 [−1.4, +0.6] | 93.89 | **+8.9 [+4.4, +13.9]** (도메인 안) |
| post-FT IntPhys 2 | 79.6 | +6.3 | 85.56 | +0.6 (CI 0 포함) |
| post-FT Predictor_v1 | 81.6 | +8.1 | 80.56 ↓ | −4.4 (CI 0 포함) |
| Ariel bc ep43 (옛) | 79.4 | +6.1 [+5.2, +6.9] | 69.44 ↓ | **−16.1 [−23.3, −8.9]** (인과 표적 −7.8 [−15.0, −0.6]) |
| 복사 | 73.2 | — | 85.00 | — |

출처: `z_research/TrainingEffects/README.md` §1-1 · `scores/SCORES.md` §1 · `ip1_copy/IP1_COPY.md` 표 2a · 2b. 등급: 확정 (TrainingEffects 적대 검증 반영본, `Archive/VERIFICATION_2026-09-25.md`).

⚠️ **IntPhys 1 값은 배열과 하네스가 칸 macro 에서 1 pt 안팎 다르다** — 하네스 (`SCORES.md`) v11_e10 77.22 · Ariel 69.44, 배열 (`IP1_COPY.md`) 78.33 · 68.89 (`IP1_COPY.md` 정정 8). 위 표의 점수는 하네스 값, Δ 는 배열 값 (같은 쌍 bootstrap) 이다.
⚠️ IntPhys 1 Δ 의 CI 끝값이 문서마다 다르다: `IP1_COPY.md` [−0.6, +8.9] vs `PREDICTOR_SHARE` [−0.6, +8.3] — bootstrap 난수 차이, 같은 값.
⚠️ v11 p − 복사 CI 끝값도 문서마다 0.1 다르다: 릴리즈 `TrainingEffects/README.md` [+1.6, +3.4] vs `scores/ANALYSIS.md` [+1.6, +3.3]; Ariel bc ep43 `PREDICTOR_SHARE` [+5.2, +6.9] vs `ANALYSIS.md` [+5.3, +6.9] — 위 표는 앞쪽 값. 결론에는 영향 없음.
⚠️ 도메인 밖 셋 (ip2 · pv1 · Ariel) 의 v11 +3 pt 안팎 차이는 한 seed 의 체크포인트 흔들림 (ip1 최대 2.4 pt) 과 같은 크기라 **순위를 매기지 않는다** (`ANALYSIS.md` 결론).
⚠️ Ariel 의 IntPhys 1 skip2_w32 칸은 예측 14 slot 을 요구하는데 Ariel 학습 지평은 ≤ 8 slot — 다섯째 교란. "복사보다 못하다" 를 attention 규칙의 성질로 읽지 않는다 (`IP1_COPY.md` 정정 1).

### 2-3. Ariel 세 팔 × 세 판

| predictor | IntPhys 1 (A.8 최고, 칸) | IntPhys 1 같은 공간 복사 | v11_split_test | IntPhys 2 Main 506 쌍 (열별 최고, 창/C) |
|---|---|---|---|---|
| 릴리즈 | 88.89 (skip2_w32) | 85.00 (창 공간) | 75.8 [75.1, 76.6] | 54.35 (w48/C24) |
| 팔 A full ep40 | 71.67 (skip2_w32) | 85.00 | 78.1 [77.3, 78.8] | 53.75 (w32/C8) |
| 팔 B prefix ep45 | 70.00 (skip2_w32) | 85.00 | 79.5 [78.8, 80.3] | 53.56 (w48/C36) |
| 팔 C AR ep18 | **90.56** (skip2_w16) | `copy_blk` **83.33** (블록 공간) | 83.2 [82.6, 83.8] | 51.19 (w16/C8) |
| 팔 C AR ep36 / ep38 | 92.78 / 92.78 | 83.33 | 83.8 / 83.7 [83.0, 84.3] | — / 50.79 |
| 팔 C′ ctx_ar ep18 | 64.44 | — | **61.0** | — |
| 옛 bc ep43 | 69.44 | 85.00 | 79.4 | (bc_ep30: w16 53.16 · w32 53.75 · w48 52.96) |
| 복사 (표준) | — | — | 73.2 | 55.53 (w48/C42); 같은 C24 에서 릴리즈 − 복사 +1.8 [−1.8, +5.5] |
| 논문 V-JEPA 2-h | — | — | — | 57.51 |

출처: `z_training/ARIEL_CHECKPOINTS.md` §5-1 · §5-1d · §5-2 · §5-2b · §5-3 (원 문서와 대조함) · `z_research/Benchmarks/exp_results/intphys2/bench_intphys2_main_ariel_bc_ep30_w{16,32,48}/column_best.md` · `auto_research/Archive/INTPHYS2_PERMANENCE_AUDIT_2026-09-26.md` §3. 등급: 점수는 summary.json 재계산 일치 (문서 표기), 해석은 1 차.

- **A vs B (attention 규칙만 다름): IntPhys 1 71.67 vs 70.00, 세 칸 모두 같은 순서 · 1.1–1.7 pt 차 — 차이 없음.** 미래 → 미래를 막느냐 여느냐는 이 판에서 아무것도 바꾸지 않는다.
- 미래 전용 두 팔 (A · B) 은 IntPhys 1 을 **복사 85 아래 (70–72)** 로 떨어뜨리고 v11 은 +2 ~ +4 올린다. v11 가림 끝 칸은 릴리즈처럼 바닥이다 (late 정지 51.9 / 51.2, `ARIEL_CHECKPOINTS.md` §5-2b).
- IntPhys 2 는 세 팔 · 릴리즈 모두 우연 근처. AR 은 범위 안인 w16 만 봐도 48.42 – 51.19.
- AR 의 IntPhys 1 · v11 상승은 §5 에서 따로 본다.

⚠️ ctx_ar 은 ep18 (학습 중) 판이다. 40 epoch 판으로 다시 잰다고 적혀 있으나 그 결과 문서는 없다.
⚠️ AR 은 표적 공간이 블록별이라 surprise 크기 · margin · 복사 기준선을 다른 kind 와 **비교할 수 없다 — 쌍 정확도만** (`ARIEL_CHECKPOINTS.md` §5 단서). v11 의 `copy_blk` 는 **사용자 지시로 돌리지 않았다** (09-28) → AR v11 +7.4 pt 의 predictor 몫 / 공간 몫 분리 불가.
⚠️ Ariel 판 surprise 절대값이 우리 쪽에서 ~0.004 (0.7 %) 낮다 — 입력 프레임 출처 차이로 추정, 미결.

### 2-4. 도메인 밖 운동 판 (RollOut_v2 · v3) — 요약 (자세히는 §3 · §4)

| predictor | RollOut_v2 (위치 자, β_peak = attention 최댓값 칸 이득, 진실 1) | RollOut_v3 가속 · 감속 ρ (GT 1) | RollOut_v3 flat_v reach (p@h / p@p, px) |
|---|---|---|---|
| 릴리즈 | flat_v 1.01 · flat_a 0.60 · ramp_a 0.50 · arc 0.96 · ledge 0.55 | 두 자 불일치 | 82 / 82 |
| post-FT v11 | **flat_v 2.10 · flat_a 1.83 · ramp_a −0.24 · arc 0.07 · ledge 1.23** (v11 속도 강요, 경사 · 낙하 무시) | 궤적 속도 따라감 (기울기 0.86) + 고정 앞섬 +37 px | 43 / 47 (견고 ↓) |
| post-FT Predictor_v1 | 안 잼 | 등속 외삽 쪽 | 127 / 127 (하한, 견고 ↑ — 근접 도메인) |
| post-FT IntPhys 1 | 안 잼 | 등속 외삽 쪽 | 64 / 115 |
| post-FT IntPhys 2 | 안 잼 | 섞임 | 36 / 82 |
| Ariel bc ep43 | 안 잼 | 섞임 | 104 / 65 |

출처: `z_research/predictor_training/predictor_IntPhysGenV11_PFT/Archive/RESULTS_2026-09-16.md` §2-1 · `z_research/TrainingEffects/v3_motion/MOTION.md` 표 3 · `README.md` §2.

⚠️ RollOut_v2 의 post-FT 는 v11 하나만 쟀다. 릴리즈 p 로 학습한 위치 자의 선형 층은 post-FT p 에 **이식되지 않는다** (y 편향 50–80 px) → MAE · 선형 β 대신 최댓값 칸 β_peak 로 읽는다 (attention 질량은 유효).

### 2-5. gap-recovery (기존 수치 재배열)

recovery = (Acc_FT − Acc_release) / (Acc_in-domain FT − Acc_release).

| 평가셋 (release → in-domain FT) | FT on | Acc | recovery |
|---|---|---:|---:|
| v11 test (75.83 → 91.35) | IntPhys 1 | 78.53 | **17 %** |
| v11 test | IntPhys 2 | 79.56 | 24 % |
| IntPhys 1 dev (88.89 → 93.89) | v11 | 77.22 | **−233 %** |
| IntPhys 1 dev | IntPhys 2 | 85.56 | −67 % |

출처: 루트 `V-JEPA 2 predictor 분석 ~하면 되지 않을까 에 답하는 실험 제안.md` 실험 3 (2026-09-24, @이종서; 원 문서와 대조함). 새 실험 아님. ⚠️ IntPhys 2 는 in-domain FT (56.75) < 릴리즈 (57.94) 라 recovery 정의 불가.

### 2-6. 그룹 B — 사전학습부터 다시 해도 v11 에서 자기 복사를 못 넘는다

| 모델 | v11 (own p) | own 복사 | own p − own 복사 [95 % CI] | IntPhys 1 (A.8 최고) | 칸 |
|---|---:|---:|---|---:|---|
| ViT-B K400 72K | 53.6 | 61.1 | −7.6 [−8.2, −7.1] | 52.78 | skip5_w16 |
| ViT-B K400 30K\* | 54.7 | 76.9 | −22.3 [−23.1, −21.4] | 49.44 | skip2_w32 |
| ViT-B 합성 물리 30K | 69.3 | 69.0 | +0.3 [−0.5, +1.1] | **96.11** [92.9, 98.6] | skip2_w32 |
| Tiny K400 5K | 67.2 | 66.8 | +0.4 [−0.0, +0.8] | 55.56 | skip2_w16 |
| Tiny 합성 물리 5K | 67.1 | 67.4 | −0.3 [−0.8, +0.1] | 53.89 | skip2_w16 |
| Tiny K400 30K | 68.7 | 69.6 | −0.9 [−1.4, −0.4] | 62.22 | skip2_w16 |
| Tiny 합성 물리 22.5K\* | 67.4 | 70.2 | −2.9 [−3.5, −2.3] | 72.22 | skip2_w32 |
| Tiny K400+SSv2 22.5K\* | 72.4 | 71.0 | +1.4 [+0.8, +2.0] (visible 에서는 −2.2) | 63.33 | skip2_w32 |

출처: `z_research/TrainingEffects/README.md` §1-2 · `scores/SCORES.md` §1 (원 문서와 대조함). 등급: 확정 (정본).

- 데이터가 IntPhys 1 을 좌우한다 (합성 물리 ViT-B 96.1 vs K400 ViT-B ~50). 그러나 v11 에서는 **여덟 중 누구도 자기 encoder 복사를 의미 있게 넘지 못한다** (−22.3 ~ +1.4; CI 로 넘는 하나는 학습 중 체크포인트이고 visible 에선 −2.2).
- 이동 칸 vanish 는 그룹 B 대부분이 '물체→빈' 0 / '빈→물체' 100 (방향 편향, `SCORES.md` §3).

⚠️ **그룹 B 의 IntPhys 1 에는 복사 기준선이 없다** — 96.1 이 predictor 몫인지 모른다 (`TrainingEffects/README.md` §5 미결). 복사 절대값은 encoder 마다 달라 행끼리 비교하지 않는다.

출처 (§2 전체): `z_training/runs/_matrix/MATRIX.md` · `z_training/runs/_matrix/predictor_v1_holdout_l1.md` · `z_training/RESULTS_2026-09-19.md` · `z_research/TrainingEffects/README.md` §1 · `z_research/TrainingEffects/scores/SCORES.md` · `z_research/TrainingEffects/scores/ANALYSIS.md` · `z_research/TrainingEffects/ip1_copy/IP1_COPY.md` · `z_training/ARIEL_CHECKPOINTS.md` §5 · `auto_research/Archive/COPY_HARD_SUBSET_2026-09-29.md` · `auto_research/Archive/PREDICTOR_SHARE_2026-09-29.md` · `z_research/Benchmarks/exp_results/intphys2/bench_intphys2_main_ariel_bc_ep30_w{16,32,48}/column_best.md` · `V-JEPA 2 predictor 분석 ~하면 되지 않을까 에 답하는 실험 제안.md`

---

## 3. 능력별 — 네 능력 (+ 도달 · 닫힘) 에서 무엇이 바뀌었나

조작적 정의는 03 장 (1 읽기 · 2 지속 · 3a 추세 · 3b 사건 · 4 영속). 이 절은 **같은 판 · 같은 자 · 같은 문턱** 으로 predictor 만 바꾼 비교만 모았다. 자는 전부 **릴리즈에서 학습 · 보정한 것을 이식** 했다 (단서 §7).

### 3-1. 한 장 요약

| 능력 | 판 · 판독 | 학습이 바꾼 것 | 도메인 밖으로 갔나 | 등급 | 출처 |
|---|---|---|---|---|---|
| 1 읽기 (정체) | v11 late, 풀링 logistic probe | **아무것도** — 모든 predictor 천장 (shape 96.1–97.8 · color 94.0–97.5 %, 릴리즈 96.8 / 97.1) | — (처음부터 천장) | 확정 | `TrainingEffects/v11_identity/IDENTITY.md` (i) |
| 배치 (p 를 h 쪽으로 옮김) | cos(p_fut, h_fut), late | 같은 생성기 둘만 크게 (릴리즈 0.931 → v11 0.996 · pv1 0.991) | 아니오 — ip1 0.974 · ip2 0.934 · Ariel 0.959 | 확정 | `IDENTITY.md` (ii) |
| 4 영속 (v11) | late k≥2 진실 칸 attention 순 배치 | v11_e10 만 깨끗 (+0.60 / +0.53) | 아니오 — 도메인 밖 셋 0.04–0.13 = 릴리즈 수준 | 확정 (궤적 재생 단서) | `TrainingEffects/v11_presence/PRESENCE.md` |
| 4 영속 (v11, 자 없음) | vanish '물체→빈' late k≥2 | v11 100 / 100 · pv1 96 / 92 (교환) | 아니오 — **Ariel 27 / 5 = 복사 26 / 5** | 확정 | `PRESENCE.md` 표 7 (하네스 채점; 복사 줄은 curves 채점) |
| 4 영속 (K400 팬 가림 띠) | permanence 비 hid / vis, k=4 | 어떤 학습도 > .53 안 됨 (.42–.50) | 아니오 (Ariel 에게는 도메인 안인데도) | 확정 | `VERIFY_ISCENE2` §1 · `VERIFY_AR` |
| 2 지속 (v11 보임) | 진실 칸 attention · x 오차 | 오른다 (릴리즈 0.17 → v11 0.75 · pv1 0.72 · Ariel 0.64) | 복사 AUROC 1.00 이라 **예측 증거 아님**; 갈리는 건 x 오차 | 확정 | `PRESENCE.md` |
| 2 지속 (RollOut_v3) | '모임' t8–15, 두 정의 × 두 자 | 견고한 변화는 pv1 ↑ · v11 ↓ 둘뿐 | pv1 은 **근접 도메인** (같은 10° 쐐기 · 등속 팔) | 확정 | `MOTION.md` 표 1 |
| 2 지속 (RollOut_v3, 파라미터 0) | 외형 템플릿 경로 기울기 t3–7 (진실 0.46) | 릴리즈 0.07 → pv1 0.52 · Ariel 0.48 · v11 0.58 | **예 (예외 하나)** — 가시 물체를 등속에 가깝게 더 멀리 (Ariel). 단 약 5 칸에서 정체 | 검증 반영 정본 | `TRAINED_PREDICTORS_SIGNATURES` §3-1 |
| 3a 추세 (가속 · 감속) | v3 ρ (GT 1; 등속 연장 0.89 / 1.13) | 두 법칙 · 두 자 모두 GT 쪽인 predictor **없음** | — | 기술값 (확인되지 않음) | `MOTION.md` 표 2 |
| 3a 추세 (중력 arc) | v3 arc y 보정 낙하 오차 | pv1 은 오히려 **덜** 떨어짐 (Δ −17.6 / −14.5); ip1 조금 더 (+2.6 / +7.0, 작음) | — | 확정 (부호 버그 수정판) | `MOTION.md` |
| 3a 추세 (AR) | 가속 반영 계수 α_acc, arc · ledge t4–7 | AR ep40 −0.42 / −0.46 (릴리즈 +0.08 / −0.25, h +1.43 / +1.28) | — (등속 외삽보다 뒤) | 1 차 + 적대 검증 (확정 범위 좁음) | `AR40_READOUT` · `VERIFY_AR40RD` |
| 3b 사건 (ledge · wall) | v3 참고 칸 · RollOut_v2 위치 자 | Ariel ledge 견고 ↓ (GT 에서 멀리) · wall 넘어감 ↑; v11 post-FT wall 통과 · 낙하 무시 | — | 1 차 (MOTION 은 '참고 칸') | `MOTION.md` · `RESULTS_2026-09-16.md` |
| 도달 reach (실영상) | R50 칸 — 팬 · SSv2 · **EK100** | 팬 · SSv2 는 오른다 (Ariel 도메인 안) | **EK100 은 아무도 2.2 를 못 넘음** | 확정 (EK100) | `I_SCENE` §4-2 · §5 · `AR_PREDICTOR` §2-1 |
| 닫힘 | 자기 되먹임 − one-shot (팬 슬롯 3–7) | 미래 예측 학습 predictor 일수록 **더 나빠짐** (Ariel −0.130 · N1 −0.189, 릴리즈 −0.014) | — | 확정 | `VERIFY_ISCENE2` §2 |
| (벤치마크) IntPhys 2 Main | VoE 쌍 정확도 | 세 팔 · 릴리즈 · post-FT 모두 우연 근처 | — | 확정 | `ARIEL_CHECKPOINTS.md` §5-3 |

### 3-2. 영속 (4) — 칸별

| 판 · 판독 | 복사 | 릴리즈 | v11_e10 | pv1_e15 | ip1_e40 | ip2_e80 | Ariel bc ep43 | Ariel 세 팔 | 출처 |
|---|---|---|---|---|---|---|---|---|---|
| v11 late k≥2 진실 칸 attention (flat / ramp) | — | 0.06 / 0.04 | 0.66 / 0.57 | 0.50 / 0.35 | 0.08 / 0.04 | 0.08 / 0.07 | 0.13 / 0.10 | 안 잼 (AR 은 readout 미정의) | `PRESENCE.md` 표 1 |
| 같은 자리 · 빈 장면 (귀무) | — | — | 0.06 / 0.04 | **0.21 / 0.09** (epoch 마다 ↑) | — | — | — | — | 〃 |
| 순 배치 (= 위 − 귀무) | — | — | **+0.60 / +0.53** | +0.29 / +0.26 | +0.03 / +0.00 | +0.04 / +0.04 | +0.09 / +0.07 | — | 〃 (참조 h@h +0.45 / +0.39) |
| vanish '물체→빈' late k≥2 (flat / ramp, %, 자 없음) | 26 / 5 | 0 / 0 | 100 / 100 | 96 / 92 ⚠️ | 1 / 0 | 48 / 26 | **27 / 5** | — | `PRESENCE.md` 표 7 (하네스 채점; 복사 줄은 curves 채점 — 표 7b curves 판은 ip1 0 / 0 · ip2 46 / 27 · Ariel 29 / 4) |
| vanish '물체→빈' late, 112 쌍씩 (static / flat / ramp, %) | 67.9 / 39.3 / 9.8 | 65.2 / 2.7 / 0.0 | 86.6 / 100 / 100 | 52.7 / 88.4 / 77.7 | — | — | — | prefix 49.1 / 37.5 / 14.3 · full 50.9 / 15.2 / 8.9 · AR ep38 99.1 / 85.7 / 72.3 | `COPY_HARD_SUBSET` §2-3 |
| k=1 late flat A (28 block, %) | **79** | 11 | 100 | 64 | 14 | 75 | 68 | — | `PRESENCE.md` 정정 6 |
| K400 팬 가림 띠 hid / vis 비 (k=4, 0–1 칸) | — | .53 [.50, .56] | — | .50 | — | — | .47 | full .47 · prefix .46 · AR18 .47 [.44, .50] (k=2 .59 최저) · AR38 .46 · ctx_ar ep16 .466 · N1 .42 · T1 .43 | `VERIFY_ISCENE2` §1 · `VERIFY_AR` · `VERIFY_R3` A5 |

- **영속 배치를 확실히 모은 것은 v11 궤적을 재생 학습한 v11_e10 하나** 다. 궤적만으로 묶은 CI 는 flat [−0.01, +0.32] 로 0 을 걸친다 (`PRESENCE.md` 정정 5).
- pv1 의 상당 부분 (flat ~40 % · ramp ~26 %) 은 "문맥과 무관하게 가림막 출구에서 물체가 나온다" prior 다 — 빈 장면 귀무가 epoch 마다 0.12 → 0.21 로 커지고, 반대 방향 '빈→물체' 가 100 → 72.6 % 로 떨어진다 (교환) (`PRESENCE.md` 정정 2 · pv1 epoch 표).
- Ariel 의 vanish '물체→빈' 은 **복사 수준** (27 / 5 vs 26 / 5) — 합격선 표의 Ariel 칸 근거.
- AR 의 late 정답 (72–99) 은 **presence-without-placement**: 재등장 칸 margin −.040 / −.029 (음수 — 릴리즈 · prefix 도 음수), 원거리 (링 ≥ 3) +.011 ~ +.018 의 전역 오프셋이 이긴다. AR margin 지도 ≈ 블록 복사 지도 (r .59). 위치 자로 읽으면 late 가림 뒤 AR 물체 토큰 0–8 % = 빈 장면 오탐 2.5 % 와 구분 안 됨 (`VERIFY_PRESENCE` §1–§5 · `VERIFY_AR40RD` §4).
- K400 팬 가림 띠에서는 미래 예측 학습 (Ariel · N1 · T1) 이 오히려 비를 낮춘다 (.42–.47). 같은 칸 비가림 대비로는 약 1/3 만 남고 predictor 는 원래 내용보다 가림막을 그린다 (cos 회색 .53 > 내용 .42) (`VERIFY_ISCENE2` §1).

⚠️ k=1 에서는 복사 (79) 가 pv1 (64) · Ariel (68) 보다 높다 — "배치가 모인 predictor 가 맨 위" 는 k≥2 에서만이다.
⚠️ 고정 문턱 판독 (present − 바닥) 은 자를 바꾸면 부호가 뒤집힌다 (ip2 p@h −0.24 / −0.30 vs p@p +0.13 / +0.42) — 단독 인용 금지 (`PRESENCE.md` 결론).
⚠️ vanish A/B 의 CI 는 block bootstrap 이라 궤적 2 개 구조를 반영하지 못한다 (`PRESENCE.md` 미적용).

### 3-3. 지속 (2) · 도달 (reach) — 칸별

| 판 · 판독 | 릴리즈 | v11_e10 | pv1_e15 | ip1_e40 | ip2_e80 | Ariel bc ep43 | 출처 |
|---|---|---|---|---|---|---|---|
| v11 보임 진실 칸 attention (flat / ramp) | 0.17 / 0.05 | 0.75 / 0.66 | 0.72 / 0.62 | 0.55 / 0.15 | 0.26 / 0.12 | 0.64 / 0.37 | `PRESENCE.md` |
| v11 보임 flat t4 x 오차 (진실 전진 86 px; 궤적 2 개) | −72 px | −17 | −16 | −28 | −64 | −36 | 〃 |
| v3 '모임' t8–15 (P32, p@h) | 0.46 | **0.04** (↓ 견고) | **0.84** (↑ 견고) | 0.31 | 0.32 | 0.52 (정의에 따라 반대) | `MOTION.md` 표 1 |
| v3 C16/P16 t4–7 flat_v x 부호 오차 (p@h; 복사 −53.9) | −8.4 px | **+37.3** | +6.7 | +10.4 | −2.3 | +9.4 | `MOTION.md` 표 2 |
| v3 궤적 사이 기울기 · 절편 (flat_v 14 궤적, p@h) | 0.91 · −4 px | 0.86 [0.57, 1.13] · +44 | 1.04 · +5 | 1.00 · +10 | 1.10 · −8 | 0.92 · +14 | `MOTION.md` |
| v3 flat_v reach (p@h / p@p, px) | 82 / 82 | 43 / 47 (↓ 견고) | 127 / 127 (하한, ↑ 견고) | 64 / 115 | 36 / 82 | 104 / 65 | `MOTION.md` 표 3 |
| v3 파라미터 0 외형 경로 기울기 t3–7 / t8–11 / t12–15 (진실 .46 / .45 / .44) | 0.07 / 0.03 / 0.03 | 0.58 / −0.13 / −0.05 | 0.52 / 0.24 / −0.00 | — | — | 0.48 / 0.29 / −0.08 | `TRAINED_…` §3-1 |
| 정체 자리 (칸, 진실 t15 7.2) | ≈ 2.1 | ≈ 5.5 | ≈ 5.1 | — | — | ≈ 5.4 | 〃 |

| 실영상 reach R50 (칸) | 팬 (K400 인공) | SSv2 | EK100 | v3 | 출처 |
|---|---:|---:|---:|---:|---|
| 릴리즈 | 4.83 | 2.69 | 2.12 | 4.65 | `AR_PREDICTOR` §2-1-c |
| pv1 | 5.0 | — | — | — | `I_SCENE` §4-1 |
| Ariel bc ep43 | 6.54 | 4.0 | 2.17 | 7.7 | `I_SCENE` §4-1 · §4-2 · §5-3 |
| 팔 A full ep40 | 6.99 | 4.17 | 2.17 | 7.34 | `AR_PREDICTOR` §2-1-c |
| 팔 B prefix ep45 | 6.57 | 4.06 | 2.17 | 7.76 | 〃 |
| 팔 C AR roll ep18 / ep38 | **3.24 / 3.91** | **2.18 / 2.22** | **1.75 / 1.77** | **4.24** | 〃 · §2-4 |
| (AR tf — 진짜 블록 되먹임, 상한 탐침) | 9.19 | ∞ | 4.00 | ∞ | 〃 |
| M · T0 (릴리즈 post-FT K400 고정 창) | 6.34 | 3.38 | 2.17 | — | `I_SCENE` §5-4 |
| M · T1 (〃 팬 증강) | 10.23 (v = 12 팔 값) | 3.55 | 2.20 | — | 〃 |
| N · N0 / N1 (Ariel ep43 post-FT 고정 / 팬 증강) | 6.3 / **8.7** | 3.8 / 3.8 | 2.1 / 2.1 | — | `I_SCENE` §5-1 |

- **EK100 R50 은 어떤 학습으로도 2.2 를 넘지 않는다** (2.11 – 2.20). EK100 미래 변위의 67–75 % 가 카메라 운동이고 encoder 는 그것을 R² .69 / .63 으로 읽는다 (`ENCODER_MOTION_READOUT` · `VERIFY_R3` B) → 막히는 곳은 encoder 가 아니다 (07 장).
- 팬 · SSv2 의 증가는 Ariel 에게 **도메인 안** (K400 · SSv2 로 학습) 이다. 팬 증강 post-FT 는 **팬에서만** 든다 — T1 − T0 팬 +3.89 [3.78, 4.01] · SSv2 +0.17 · EK100 +0.02 (`VERIFY_ISCENE2` §5-4); N1 은 팬 8.7 인데 SSv2 · EK100 불변.
- RollOut_v3 (합성, Ariel 에게 도메인 밖) 에서도 Ariel 은 릴리즈보다 멀리 옮긴다 (v3 R50 7.7 vs 4.65; 파라미터 0 경로 기울기 0.48 vs 0.07) — `TrainingEffects/README.md` §3 의 **"예외 하나"** 다. 그러나 (i) 가시 물체 · 등속에 가까운 외삽뿐이고, (ii) 약 5 칸에서 정체하며, (iii) 같은 Ariel 의 IntPhys 1 은 69.4, (iv) arc 에서는 어느 predictor 도 이득 0 (t10 pv1 +0.01 · Ariel −0.02) 이다 (`TRAINED_…` §3).
- 거리 단위: Ariel 은 reach 를 약 1.5 배 늘리지만 한계가 남는다 (거리만 R² 적합 — 릴리즈 3.2 칸 · pv1 3.9 · Ariel 5.6; 슬롯만 R² 0) (`SC1_REACH_NOT_HORIZON` §3-1, 1 차).

⚠️ "v11 post-FT 의 reach ↓ (43 / 47)" 와 "앞섬 +37 px" 는 같은 현상의 두 면이다 — 고정 앞섬 2–3 칸 뒤 정체 (SC1: v11_postft 는 어느 단위로도 안 모임, R² 0.15 = 다른 실패 양식).
⚠️ v3 지속의 주 자 `p@h` 는 p 토큰에서 따로 검증되지 않았다 — 평지에서 릴리즈 p 를 t0 +19 → t7 +40 px 위로 읽는다 (`MOTION.md` 단서 1). 파라미터 0 판독 (`TRAINED_…`) 과 방향이 같은 칸만 주장한다.
⚠️ P32 t8–15 는 v11 · pv1 의 학습 지평 (미래 ≤ 8 튜블릿) 밖이다 (ip1 ≤ 14 · ip2 ≤ 18 은 일부 안) (`MOTION.md` 정정 6).
⚠️ RollOut_v2 09-19 의 "거리 한계" 와 `MOTION.md` 의 "릴리즈 끊김은 시간 한계 쪽" (log t_cliff ~ log 속도 기울기 −0.12) 은 자료 · 창이 달라 둘 다 남긴다 (04 장).

### 3-4. 전이 3a · 3b — 칸별

| 판 | GT · 기준 | 릴리즈 | v11_e10 | pv1_e15 | ip1_e40 | ip2_e80 | Ariel ep43 | AR ep40 | 출처 |
|---|---|---|---|---|---|---|---|---|---|
| v3 flat_a ρ (p@h / p@p) | GT 1 · 등속 연장 0.89 | 0.94 / 0.80 | 1.03 / 0.96 | 0.85 / 0.84 | 0.83 / 0.78 | 0.93 / 0.85 | 0.93 / 0.90 | — | `MOTION.md` |
| v3 flat_d ρ (p@h / p@p) | GT 1 · 연장 1.13 | 0.89 / 1.14 | 0.93 / 0.96 | 1.12 / 1.14 | 1.09 / 1.18 | 0.92 / 1.03 | 0.99 / 1.02 | — | 〃 |
| v3 arc y 보정 낙하 오차 (p@h / p@p; − = 덜 떨어짐, 복사 ≈ 연장 −15.2) | GT 0 | −1.2 / −7.7 | −2.4 / −12.4 | **−18.8 / −21.1** | 3.5 / 0.9 | −0.7 / −15.1 | −2.2 / −3.9 | — | 〃 |
| v3 arc · ledge 가속 반영 계수 α_acc (t4–7, 각 predictor 자기 자) | 0 = 등속, 1 = 진실; h +1.43 / +1.28 | +0.08 / −0.25 | — | — | — | — | — | **−0.42 / −0.46** | `AR40_READOUT` §1 · `VERIFY_AR40RD` §1 |
| v3 ledge y 보정 (참고 칸; 복사 ≈ 연장 −13.4) | GT 0 | −10.7 / −9.7 | — | — | −8.9 / −8.4 (↑ 견고, 작음) | — | **−14.6 / −13.1** (↓ 견고, GT 에서 멀리) | — | `MOTION.md` 표 2 |
| v3 wall x 넘어감 (참고 칸; 연장 +46.5, 복사 −6.0) | GT 0 | −0.2 / +16.1 | — | +21.3 / +22.9 | — | — | **+13.0 / +46.7** (GT 에서 멀리) | — | 〃 |
| RollOut_v2 β_peak (슬롯 0→3; ramp_a · arc · ledge) | 진실 이득 1 | 0.50 · 0.96 · 0.55 | **−0.24 · 0.07 · 1.23** | — | — | — | — | — | `RESULTS_2026-09-16.md` §2-1 |

- **가속과 감속을 두 자 모두에서 GT 로 넣는 predictor 는 없다**; 두 법칙 · 두 자 모두 연장 (또는 너머) 쪽인 것이 ip1 · pv1. GT–연장 간격 (0.11 · 0.13) 이 CI 반폭과 비슷해 ρ 는 **기술값** 이고, "넣지 않는다" 가 아니라 "**넣는다는 증거가 없다**" 로 쓴다 (`MOTION.md` 정정 4 · `VERIFICATION` (6)).
- 가속 · 감속 · 경사가 있는 Predictor_v1 로 학습한 pv1 은 arc 에서 오히려 **더 덜 떨어진다** (보정 Δ −17.6 [−18.8, −16.4] / −14.5).
- AR ep40 은 분포 안 (t0–7) arc · ledge 에서 **등속 외삽보다도 뒤** 다. 확정으로 쓸 수 있는 것은 "같은 자 안에서 AR ≤ 릴리즈, h 천장 ≈ 1" 뿐이다 (`VERIFY_AR40RD` 판정 표).
- v11 post-FT 는 RollOut_v2 에서 경사 (ramp_a) · 포물선 (arc) 의 attention 최댓값이 앞으로 나가지 않고 wall 을 통과한다 (MAE x 69.6 px vs 릴리즈 31.5, 선형 자 — §2-4 단서).

⚠️ **이전 판 (09-25 04:10) 의 arc · ledge 보정 값은 부호 버그 (자 치우침을 두 배로 더함) 로 인용 금지** — 위 값은 04:32 정정판 (`MOTION.md` 정정 2).
⚠️ α_acc 절대값은 자 의존이다 — 교차 자 p@h 로 읽으면 arc t4–7 AR +0.60 · 릴리즈 +1.10 (부호 반전). flat_a 는 정의에 민감해 철회, ramp_a −0.99 · 감속 α_acc > 1 은 rollout 9–16 블록 (학습 최대 8) 분포 밖이라 뺀다 (`VERIFY_AR40RD`).
⚠️ prefix · full 팔의 α_acc · gravity_realistic 꼭대기 · ledge 실물 메시 판은 **★ 미측정** (정본 §2-4 합격선 표).
⚠️ ledge · wall 은 10-08 결정으로 3b 범위 안이지만 `MOTION.md` 에서는 여전히 '참고 칸' 이고 P32 t12 이후는 해석하지 않는다. 떠 가는 미래는 복사와 닮아 VoE 쌍 하나로는 못 가른다 (06 장).

### 3-5. 닫힘 — 자기 예측을 되먹이면

| 판독 | 릴리즈 | Ariel ep43 | N1 | pv1 | AR (roll − tf) | 출처 |
|---|---|---|---|---|---|---|
| K400 팬 Δ(ar_pA − os) 슬롯 3–7 합산 | −0.014 [−0.017, −0.010] | **−0.130** [−0.137, −0.124] | **−0.189** | −0.043 | — | `VERIFY_ISCENE2` §2 |
| 〃 정확 칸 | −0.005 | −0.087 | −0.138 | −0.008 | — | 〃 |
| AR 팬 슬롯 1 / 슬롯 6–7 | — | — | — | — | **−.042** [−.046, −.037] / −.352 · −.319 | `VERIFY_AR` §1 |
| AR40 · ctx_ar ep16 슬롯 1 (팬 / SSv2) | — | — | — | — | AR40 −.022 / −.043 · CX16 −.064 / −.091 | `VERIFY_R3` A4 |
| v3 파라미터 0 ar_pA j3–7 기울기 (진실 0.440) | −0.012 [−0.043, +0.018] | **+0.226** [0.171, 0.277] | — | +0.139 | — | `TRAINED_…` §3 |

- 진짜 관측을 붙이면 계속 맞히고, 자기 예측을 붙이면 one-shot 보다 낫지 않다. **미래 예측을 학습한 predictor 일수록 자기 예측을 붙였을 때 크게 나빠진다** — rollout 손실로 학습한 AR 도 붕괴가 첫 걸음부터다 (확정).
- 아래 줄 (v3) 은 반대 방향처럼 보이지만 비교가 다르다 — 위 줄은 **같은 predictor 안에서** 되먹임 − one-shot, 아래 줄은 **predictor 사이** 되먹임 경로 기울기다. Ariel 은 되먹여도 릴리즈보다는 멀리 가지만 자기 one-shot 보다는 나빠진다. 두 결과는 모순이 아니다.

⚠️ "자기 출력이 입력 분포 밖이라서" 는 **해석** (미검정) 이다 (`VERIFY_ISCENE2` §2). 어댑터 A = 채널 affine, 팬은 카메라 운동 하나, 400 clip.

출처 (§3 전체): `z_research/TrainingEffects/README.md` §2 · `v11_identity/IDENTITY.md` · `v11_presence/PRESENCE.md` · `v3_motion/MOTION.md` · `Archive/VERIFICATION_2026-09-25.md` · `auto_research/Archive/TRAINED_PREDICTORS_SIGNATURES_2026-09-25.md` · `I_SCENE_CAUSAL_BOTTLENECK_2026-09-25.md` · `VERIFY_ISCENE2_2026-09-26.md` · `AR_PREDICTOR_2026-09-27.md` · `VERIFY_AR_2026-09-27.md` · `VERIFY_PRESENCE_2026-09-29.md` · `VERIFY_R3_2026-09-29.md` · `AR40_READOUT_2026-09-30.md` · `VERIFY_AR40RD_2026-09-30.md` · `COPY_HARD_SUBSET_2026-09-29.md` · `ENCODER_MOTION_READOUT_2026-09-29.md` · `SC1_REACH_NOT_HORIZON_2026-09-25.md` · `z_research/predictor_training/predictor_IntPhysGenV11_PFT/Archive/RESULTS_2026-09-16.md`

---

## 4. v11 post-FT 의 실체 — 도메인 전체 적응 + 궤적 prior

v11 post-FT 는 이 장에서 **유일하게 크게 오른** predictor 다 (+15.5). 그 이득이 무엇인지가 "post-FT 가 intuitive physics 를 배웠다" 를 쓰지 않는 이유다.

### 4-1. v11 안 (도메인 안) — 가림만이 아니라 전부 오른다

| 칸 (v11_split_test) | 릴리즈 | v11_e10 | 출처 |
|---|---:|---:|---|
| overall | 75.83 | 91.35 | `SCORING_2026-09-16.md` |
| late static / flat / ramp | 57.4 / 59.4 / 59.6 | 87.1 / 84.0 / 87.4 | `SCORES.md` §2 |
| late, k = 1–4 | k 무관 바닥 (53–61; static k2 52.7) | k 무관 천장 (flat 82.1–85.7 · ramp 84.4–89.7 · static 84.4–89.3) | `SCORING` §2 |
| visible static / flat / ramp | 92.3 / 91.1 / 79.8 | 96.9 / 95.0 / 92.7 | `SCORES.md` §2 |
| early static / flat / ramp | 78.7 / 82.3 / 75.2 | 97.5 / 92.1 / 89.0 | 〃 |
| mid static / flat / ramp | 76.2 / 83.7 / 74.3 | 95.6 / 90.4 / 88.5 | 〃 |
| ramp visible shape (등가속 · 가림 없음) | 69.9 | 94.6 | `SCORING` |
| vanish '물체→빈' moving_occlusion / flat | 0.0 / 2.7 | 100 / 100 | `SCORING` §3 |
| 모양 순서 편향 (등가속 + 가림 late recall, torus) | 4.5 (cylinder > … > torus) | 81.8–90.9 로 평평 (torus +77.3) | `SCORING` §4 |
| 드러남 단계 hidden / reveal / after (late k≥2, 2,016 쌍, Δ vs 릴리즈) | — | +5.9 / +12.3 / +20.3 (v11 만 '다시 나온 뒤' 에 몰림) | `ANALYSIS.md` |
| 위치 자 질량, flat late k=4 슬롯 0–7 | 0.03–0.15 (처음부터 물체다운 토큰 없음) | 0.00 0.19 0.58 0.69 0.84 0.97 0.92 0.80 | `RESULTS_2026-09-16.md` §1 |
| 학습 곡선 (v11 overall) | 75.7 | e1 89.4 → e2 (90 % 도달) → e10 91.2 | `ANALYSIS.md` |

- late 가림이 가장 크게 오르지만 **비가림 · early · mid 도 함께 오른다** — 이득의 상당 부분은 가림 특이가 아니라 v11 도메인 전체 적응이다 (`SCORING` 결론). 외형 (모양 · 색) holdout 은 하지 않았다.
- 남은 실패의 대부분은 색이다 (late magenta 62.5 · purple 69.6 · cyan 75.0).

### 4-2. v11 밖 — v11 궤적 prior 를 강요한다

| 판 | 릴리즈 | v11_e10 | 읽기 | 출처 |
|---|---|---|---|---|
| RollOut_v2 flat_v β_peak | 1.01 | **2.10** | 진실보다 2 배 빨리 (v11 물체 속도 18 px/튜블릿 vs RollOut flat_v 최대 11) | `RESULTS_2026-09-16.md` §2-1 |
| RollOut_v2 ramp_a β_peak / R² | 0.50 / 0.56 | **−0.24 / −4.45** | 경사를 무시 | 〃 |
| RollOut_v2 arc β_peak | 0.96 | **0.07** | 포물선에서 attention 최댓값이 앞으로 안 나감 | 〃 |
| RollOut_v2 wall MAE x | 31.5 px | 69.6 px | 벽 통과 (선형 자 값 — 단서) | 〃 §2 |
| RollOut_v2 질량 슬롯 4 이후 | — | 대부분 ≤ 0.27, ramp_a · wall ≈ 0 | 도메인 밖에선 물체도 오래 못 둔다 | 〃 |
| RollOut_v3 앞섬 · 기울기 | −8.4 px · 0.91 | +37.3 px · 0.86 [0.57, 1.13] + 절편 +44 px | 궤적 속도는 따라가고 고정 오프셋을 더한다 | `MOTION.md` |
| RollOut_v3 '모임' t8–15 · reach | 0.46 · 82 px | 0.04 · 43 px | 고정 앞섬 2–3 칸 뒤 정체 | `MOTION.md` · `SC1` |
| IntPhys 1 skip2_w32 | 88.89 | 77.22 (−11.67) — 복사 85.0 아래 | | `MATRIX.md` |
| IntPhys 1 skip2_w16, e1 → e10 | — | −5.6 [−10.0, −1.7] (더 학습할수록 더 떨어짐) | | `IP1_COPY.md` 정정 2 |

- `RESULTS_2026-09-16.md` 결론: "v11 로 학습한 predictor 가 배운 것은 일반 규칙이 아니라 **v11 궤적 prior**." 같은 문서는 "encoder 를 두고 predictor 만 학습해도 가림 뒤 재출현을 만들 수 있다 (재료는 있다)" 도 적는다 — 07 장 "정보는 encoder 에 있다" 와 같은 방향의 존재 증명이다.
- **대조 — IntPhys 1 post-FT 의 v11:** 정지 칸은 오른다 (static early/mid color +31 ~ +38, static late +8.5 ~ +15.2) 그러나 이동 칸은 안 오르고 early/mid vanish '물체→빈' 이 39.3 / 35.7 (−49.1 / −45.5) 로 무너진다. "정지 물체가 가림막 뒤에 그대로 있다" 는 IntPhys 1 로도 배워지고, **움직이는 물체를 가림막 너머로 옮기는 것은 v11 에서만** 배워진다 (`SCORING` §3 결론).

### 4-3. 09-17 철회

> **철회 (2026-09-17, 사용자 지적)** — `RESULTS_2026-09-16.md` §3-1 "학습 도메인 안에서는 진짜 상태 진화처럼 보인다" (state 프레임 — 현재 금지). v11 은 속도 1 종이고 물체가 늘 화면 중앙을 지나 **held-out block 도 궤적이 학습셋과 같다** → late 91 % · 재출현 일치 · 순서 편향 소실이 전부 **학습 분포 (고정 궤적) 재생** 으로 설명된다. post-FT p 가 문맥에서 무엇을 읽는지는 잰 적이 없다. 첫 증거 조건 = 다양한 속도로 학습해 RollOut β → 1 (`predictor_IntPhysGenV11_PFT/README.md` · `Archive/RESULTS_2026-09-16.md`).

그 뒤의 정정 둘 (§8 표에도 있음):
- "v11 post-FT 가 문맥 속도를 무시한다, 진실 2 배" → **틀림** — 궤적 사이 기울기 0.86 + 고정 앞섬 +37 ~ +44 px (`MOTION.md` 정정 3).
- "v11_postft stride 3 → 3 배 과잉 외삽" → **철회** (b ≤ 1, 고정 앞섬 2–3 칸 후 정체) (`TRAINED_…` 정정 5).

⚠️ 그래서 합격선의 안 보이는 것 칸은 "v11 late vanish '물체→빈' > 50, **v11 로 학습하지 않고**" 이다 (정본 §2-4). v11 post-FT 의 87 / 100 / 100 은 기준이 아니다.
⚠️ v11 vanish clip 이 train 절반에 들어 있는지 index_train 대조는 하지 않았다 (`RESULTS_2026-09-16.md` 단서) → in-distribution 으로 읽는다. 정체 분석의 풀링 9,408 clip 중 절반 (4,704) 이 학습셋이다 (`IDENTITY.md` 단서).
⚠️ "가속을 넣는가" 에 대한 pv1 (속도 · 가속이 다양한 데이터) 의 답: v3 에서 기울기 1.04 · 앞섬 +6.7 px 로 릴리즈보다 진실에 가깝지만 (근접 도메인), arc 에서는 덜 떨어지고 IntPhys 1 은 80.56 ↓.

출처 (§4): `z_research/predictor_training/predictor_IntPhysGenV11_PFT/README.md` · `Archive/RESULTS_2026-09-16.md` · `Archive/SCORING_2026-09-16.md` · `exp_results/{pft,release}/*/RESULTS.md` · `z_research/TrainingEffects/scores/SCORES.md` §2 · `scores/ANALYSIS.md` · `v3_motion/MOTION.md` · `ip1_copy/IP1_COPY.md` · `auto_research/Archive/TRAINED_PREDICTORS_SIGNATURES_2026-09-25.md` · `auto_research/Archive/SC1_REACH_NOT_HORIZON_2026-09-25.md`

---

## 5. AR 의 IntPhys 1 90.56 — 벤치마크가 이 능력을 안 잰다는 가장 날카로운 쌍

같은 predictor (팔 C AR) 가 **IntPhys 1 최고점** 과 **실영상 reach 최저 · 영속 절반 · 닫힘 붕괴 · 가속 반영 음수** 를 동시에 낸다.

| 판 | 릴리즈 | AR ep18 | AR ep36 / 38 / 40 | 비교 기준 | 등급 | 출처 |
|---|---|---|---|---|---|---|
| IntPhys 1 (A.8 최고 칸) | 88.89 (skip2_w32) | **90.56** (skip2_w16) | 92.78 / 92.78 / — | | summary 일치 | `ARIEL_CHECKPOINTS.md` §5-1 · §5-1d |
| 같은 공간 복사 | 85.00 (창 공간) | `copy_blk` **83.33** (블록 공간) | 83.33 | 예측 없이 | 〃 | `ARIEL_CHECKPOINTS.md` §5-2 |
| Δ = p − 복사 | +3.9 [−0.6, +8.3] (CI 0 포함) | **+7.2 [+2.2, +12.8]** | +9.4 (CI 없음) | **두 Δ 의 차 3 pt 는 180 쌍 SE ≈ 3 pt 안** | 1 차 (표 자체는 검증 전) | `PREDICTOR_SHARE` |
| IntPhys 1 copy-hard (skip2_w32, n 27) | 44.4 [25.9, 63.4] | 74.1 | ep38 70.4 | copy_blk-hard (n 35) 에서는 반대 (릴리즈 77.1 vs AR 57.1 / 60.0 — ep18 / ep38) | 시사 | `COPY_HARD_SUBSET` §3 |
| IntPhys 2 Main | 54.35 | 51.19 | — / 50.79 / — | 우연 | 확정 (우연) | `ARIEL_CHECKPOINTS.md` §5-3 |
| v11_split_test | 75.8 | 83.2 | 83.8 / 83.7 / — | 표준 복사 73.2; v11 `copy_blk` **미실행** | — | `ARIEL_CHECKPOINTS.md` §5-2b |
| v11 어디서 올랐나 | 정지 late 57.4 · 등속 visible 91.1 · mid 83.7 | 정지 네 칸 99–100 (late 83.4) · 등속 visible **85.0** · mid 78.9 · 등가속 +2–4 | | 복사가 정답인 칸 (정지 · vanish) 에서 오르고 이동 칸은 릴리즈보다 낮다 | | 〃 |
| v11 copy-hard (2,877 쌍) · 무작위 겹침 귀무 대비 | 48.4 · −27 | — | ep38 73.8 · **−10** | "복사가 못 푸는 것을 푼다" 가 아니라 "오류가 복사 오류와 덜 묶여 있다" | 확정 (표현 정정) | `VERIFY_R3` C3 |
| 실영상 reach R50 팬 / SSv2 / EK100 / v3 | 4.83 / 2.69 / 2.12 / 4.65 | **3.24 / 2.18 / 1.75 / 4.24** (네 판 모두 최단) | ep38 3.91 / 2.22 / 1.77; AR40 ≡ ep38 | | 확정 | `AR_PREDICTOR` §2-1 · `VERIFY_AR` · `VERIFY_R3` A1 |
| K400 팬 permanence k=4 (k=2) | .53 (.76) | .47 (.59 — 최저) | .46 · AR40 .444 | | 확정 | `VERIFY_AR` · `VERIFY_R3` |
| 닫힘 roll − tf 슬롯 1 | — | −.042 | AR40 −.022 (팬) | 첫 자기 되먹임 걸음부터 | 확정 | `VERIFY_AR` · `VERIFY_R3` A4 |
| 가속 반영 α_acc arc / ledge (t4–7) | +0.08 / −0.25 | — | AR40 −0.42 / −0.46 | 0 = 등속 외삽 | 1 차 + 검증 | `VERIFY_AR40RD` |
| v11 late 가림 뒤 물체 토큰 (위치 자, moving k1–4) | '있음' 이지만 자리 77–100 px 틀림 (출구 prior) | — | AR40 0–8 % (빈 장면 오탐 2.5 % 와 구분 안 됨) | presence-without-placement | 확정 | `VERIFY_AR40RD` §4 |
| EK100 action anticipation (1/3 train · 20 ep, val 8,761) | 21.80 | — | AR40 22.03 | 재현 간 차 0.06 → 잡음 안 | 1 차 | `AR_PREDICTOR` §2-9 |

- **IntPhys 1 최고 + reach 최저** — 이 벤치마크 점수는 이 능력 (마지막 관측에서 멀리 · 안 보이는 것 · 없는 것) 을 요구하지 않는다 (`AR_PREDICTOR` 결론). 02 장 leakage 의 가장 날카로운 사례로 쓸 수 있다.
- AR 의 IntPhys 1 상승 중 상당 부분은 **블록별 표적 공간 몫** 이다 (같은 공간 복사만으로 83.33). 원 문서도 "AR 이 물리를 더 안다" 로 읽지 않는다고 명시한다 (원 문서 표현 — 현재 금지, 내부 귀속을 부정하는 문장).
- 문맥을 창 단위 z 로 되돌린 ctx_ar 은 AR 의 벤치마크 이점이 사라진다 (IntPhys 1 64.44 · v11 61.0 — 복사 아래, empty 쌍 정답 .21 = "물체 있음" 편향) 그런데 되먹임 reach · 붕괴 · permanence 는 **회복되지 않는다** (`VERIFY_R3` A2–A6; "문맥 인코딩은 원인 아님" 은 ep16 이라 **시사** 로 강등).

⚠️ AR 은 A · B 와 세 요소 (블록별 인코딩 · EMA 문맥 · 되먹임 입력) 가 함께 달라 어느 것 때문인지 못 가른다. 분리 실험 ((b) 입력 블록별 · 타깃 표준) 미실행.
⚠️ skip2_w32 의 C=4 칸은 rollout 14 블록 (학습 최대 8) — 분포 밖이 Filtered min 에 섞인다. AR ep18 은 학습 중간 (18 / 40) 판이고, IntPhys 1 은 ep36 · ep38 에서 92.78 로 조금 더 오른다 (+2.2, 잡음 안).
⚠️ 09-30 의 "EK100 AR ep40 val 21.74" 는 `predictor_checkpoint` 키 누락으로 **릴리즈가 돈 것** 이다 — 그 run 과 `ek100_rel_third10_grid8_ar40` · `ek100_vith_official256_released_grid8_ar40` 은 어디에도 인용 금지 (`AR_PREDICTOR` §2-9, 10-01 정정). 위 22.03 은 재실행 값 (CI 없음, 1/3 val 은 전체 기준선 35.34 와 비교 금지).

출처 (§5): `z_training/ARIEL_CHECKPOINTS.md` · `auto_research/Archive/AR_PREDICTOR_2026-09-27.md` · `VERIFY_AR_2026-09-27.md` · `VERIFY_PRESENCE_2026-09-29.md` · `VERIFY_R3_2026-09-29.md` · `PREDICTOR_SHARE_2026-09-29.md` · `COPY_HARD_SUBSET_2026-09-29.md` · `AR40_READOUT_2026-09-30.md` · `VERIFY_AR40RD_2026-09-30.md`

---

## 6. "학습이 덜 됐을 뿐" 반론 — 무엇을 답할 수 있고 무엇은 못 하나

| 반론 | 자료 | 결과 | 판정 · 등급 | 출처 |
|---|---|---|---|---|
| Ariel 을 더 학습하면 reach 가 는다 | bc ep 5 / 19 / 30 / 41 / 43 R50 — 팬 4.26 / 5.72 / 6.26 / 6.50 / 6.54 · SSv2 3.48 / 3.93 / 4.08 / 3.95 / 4.02 · **EK100 2.17 / 2.21 / 2.18 / 2.17 / 2.17** | EK100 기울기 −0.003 [−0.015, +0.005] / ln epoch | **EK100 불변 확정** · SSv2 ep19 이후 평탄 (R∞ 4.02 [3.70, 4.32]) **시사** · 팬 **미결** | `VERIFY_ISCENE2` §5-3 |
| 〃 (팬 포화) | lr cosine peak 3e-4 → ep30 / 41 / 43 은 peak 의 26 / 2.4 / 0.8 % · 누적 lr .93 / .998 / .999 | 팬 R50 이 누적 lr 에 거의 선형 (R∞ 6.77 [6.72, 6.83], τ 15.9 epoch) | **스케줄 끝과 구분 안 됨** — "더 학습해도 안 는다" 는 팬에선 못 쓴다 | 〃 |
| 〃 (v3 정체 자리) | Ariel s1_C16 plateau ep5 3.52 → 4.25 → 4.69 → ep43 5.08 칸 | 단조 증가 | 1 차, CI 없음 — 반론에 **유리한** 자료 | `SC1` §3-6 |
| post-FT 는 epoch 가 모자랐다 | v11 overall e1 89.4 (향상의 90 % 가 e2 까지) → e10 91.2 · pv1 e1 82.4 (최고) → e15 81.4 · ip1 진동 (e5 80.3 · e10 77.9 · e20 80.0 · e40 78.8) · ip2 → e80 79.5 (향상의 90 % 가 e40 까지) | 첫 epoch 에 대부분 | 확정 (도메인 안 점수) | `ANALYSIS.md` |
| 〃 (능력 판독) | 영속 배치 v11 e1 0.51 (e10 0.66) · pv1 e1 0.53 (e15 0.50); v3 지표 대부분 첫 체크포인트에 절반 넘게 (v11 · pv1 · ip1 5/6, ip2 2/6); pv1 reach e1 126 → e15 127 | 첫 epoch 에 생기고 그 뒤 평평 | 확정 | `PRESENCE.md` · `MOTION.md` 표 4 |
| 더 학습하면 도메인 밖도 오른다 | IntPhys 1 skip2_w16 v11 e1 → e10 **−5.6** [−10.0, −1.7] · pv1 e1 → e15 **−9.4** [−14.4, −5.0] | **반대 — 도메인 밖은 더 떨어진다** | 확정 | `IP1_COPY.md` 정정 2 |
| 〃 (Ariel IntPhys 1) | ep5 47.2 → ep19 62.2 → ep30 67.2 → ep43 68.9 (skip2_w32 배열); skip2_w16 ep5 → ep43 +21.1 [+15.6, +27.2] | **오른다** — 그러나 끝까지 복사 85.0 아래 (−16.1) | 반론에 유리한 유일한 오름 | `IP1_COPY.md` 학습 곡선 |
| AR 을 끝까지 학습하면 | ep18 → ep36 / 38 / 40: v11 83.2 → 83.8 / 83.7 · IntPhys 1 90.56 → 92.78 · IntPhys 2 51.19 → 50.79 · reach AR40 ≡ ep38 (Δ ≤ .01) · 팬 R50 3.24 → 3.91 | reach 최단 · 붕괴는 그대로 | 확정 (AR40 ≡ 38) | `ARIEL_CHECKPOINTS.md` · `VERIFY_R3` A1 |
| 데이터가 틀렸다 — 큰 변위를 보여 주면 | 팬 증강 post-FT (M · T1 / N · N1, K400 40k, 3 epoch) | 팬에서만 오름 (T1 − T0 +3.89 · SSv2 +0.17 · EK100 +0.02; N1 팬 8.7, SSv2 · EK100 불변) | 확정 (§5-4) · 1 차 (§5-1) — "큰 변위 커리큘럼은 도메인 안에서만" | `I_SCENE` §5 · `VERIFY_ISCENE2` §5-4 |
| 데이터가 틀렸다 — 속도 · 가속이 다양하면 | Predictor_v1 (등속 · 가림 · 경사 가속, 속도 다양, 14,250) | v3 근접 도메인에서만 이득 (지속 ↑, reach 127); arc 는 덜 떨어짐 · IntPhys 1 80.56 ↓ · 영속은 출구 prior | 확정 | `MOTION.md` · `PRESENCE.md` |
| 사전학습부터 다시 하면 | 그룹 B 8 (K400 · 합성 물리 · SSv2) | v11 own p − own copy −22.3 ~ +1.4 | 확정 | `TrainingEffects/README.md` §1-2 |

- **"첫 epoch 에 대부분" + "더 학습하면 도메인 밖은 더 떨어진다"** 가 post-FT 쪽의 답이고, **"EK100 불변"** 이 Ariel 쪽의 답이다.
- 반론에 유리한 자료도 있다 — Ariel 의 v3 정체 자리 · IntPhys 1 은 epoch 따라 오른다. 둘 다 복사 · 진실에 한참 못 미치는 자리에서의 오름이고, 팬 R50 포화 여부는 lr 스케줄과 교락돼 판정 못 한다.

⚠️ 답하지 못하는 것: 다른 seed · 더 긴 스케줄 · 훨씬 큰 데이터 · 다른 목표. 그룹 B 는 encoder 까지 바꿔 predictor 몫을 분리할 수 없다. 그래서 이 장의 결론은 "지금까지의 학습" 에 한정된다 (09 장 합격선이 이 빈칸을 겨눈다).

출처 (§6): `auto_research/Archive/VERIFY_ISCENE2_2026-09-26.md` §5-3 · §5-4 · `I_SCENE_CAUSAL_BOTTLENECK_2026-09-25.md` §5 · `SC1_REACH_NOT_HORIZON_2026-09-25.md` §3-6 · `z_research/TrainingEffects/scores/ANALYSIS.md` · `ip1_copy/IP1_COPY.md` · `v11_presence/PRESENCE.md` · `v3_motion/MOTION.md` 표 4 · `z_training/ARIEL_CHECKPOINTS.md` · `VERIFY_R3_2026-09-29.md`

---

## 7. 공통 한계 (결론과 같은 무게)

| 한계 | 무엇을 못 하게 하나 | 출처 |
|---|---|---|
| **학습은 전부 seed 0 한 번** | 학습 분산 모름. IntPhys 1 180 쌍 CI ±7 · IntPhys 2 252 쌍 ±6; ip1 학습 곡선이 ±2 pt 진동 → 도메인 밖 셋 (ip2 · pv1 · Ariel) 의 v11 순위 불가 | `MATRIX.md` · `ANALYSIS.md` · `VERIFICATION` (15) |
| **Ariel 은 릴리즈와 네 조건이 동시에 다르다** (attention 규칙 · 데이터 · scratch · epoch) | 릴리즈 대비 차이를 한 요인에 못 붙인다. 한 요인 대조는 A vs B (attention) 뿐이고 차이 없음. C 는 세 요소가 함께 바뀐다 | `MODELS.md` · `ARIEL_CHECKPOINTS.md` §1 |
| **v11 은 운동 조건마다 궤적 2 개** (좌→우 · 우→좌, 속도 1 종) | v11 post-FT 의 이득은 궤적 재생의 상한. "처음 보는 운동에서도 좋아진다" 는 못 하는 결론 | `PRESENCE.md` 단서 · `RESULTS_2026-09-16.md` 철회 |
| **목적함수를 원인으로 걸지 않는다** | "L1 로 학습한 예측은 … 수축" 해석 문장은 삭제됐다 (`ANALYSIS.md` 정정 6). 원인 논의는 discussion 한 문단 | `ANALYSIS.md` · CLAUDE.md §0 |
| **주 자 `p@h` 는 p 토큰에서 미검증** | decoder-first 규칙 위반. 평지에서 릴리즈 p 를 19–40 px 위로 읽음 → p@p · 파라미터 0 판독과 같은 방향만 주장 | `VERIFICATION` 주요 지적 · `MOTION.md` 정정 9 |
| **자는 릴리즈에서 학습 · 보정한 것을 이식** | 학습 후 p 의 logit 척도 이동이 고정 문턱 판독에 섞인다 (ip2 영속 부호가 자에 따라 반전). 배치 · AUROC · 순 배치를 같이 본다 | `PRESENCE.md` 정정 4 |
| **복사 기준선의 encoder 교차** | 표준 복사는 online z 의 LN 을 EMA target 공간에 대고 잰다 (깨끗한 copy 미실행). AR 공간 `copy_blk` 는 v11 에서 미실행 (사용자 보류 09-28) | `PREDICTOR_SHARE` 단서 · `ARIEL_CHECKPOINTS.md` |
| **학습 지평 밖 칸** | v3 P32 t8–15 는 v11 · pv1 지평 밖; Ariel 의 IntPhys 1 skip2_w32 는 14 slot (학습 8); AR rollout 9–16 블록 · IntPhys 2 w32 · w48 은 분포 밖 | `MOTION.md` 정정 6 · `IP1_COPY.md` 정정 1 · `VERIFY_AR40RD` |
| **근접 도메인** | pv1 은 v3 와 같은 10° 쐐기 · 등속 팔 → v3 이득은 일반화 아님; `intphys1_postft` 는 dev 와 같은 생성기 | `MOTION.md` 정정 8 · `RESULTS_2026-09-19.md` |
| **다중 비교** | ANALYSIS 240 차이 중 ~12 는 우연 유의; 22 tag × 요약 ~10 × 두 자 | `ANALYSIS.md` · `MOTION.md` |
| **IntPhys 2 표본 둘** | Main split test 252 쌍 (post-FT) · Main 506 쌍 (Ariel · 릴리즈) — 섞지 않음 | `MATRIX.md` · `ARIEL_CHECKPOINTS.md` |
| **그룹 B IntPhys 1 복사 없음** | 합성 물리 ViT-B 96.1 의 predictor 몫 모름 | `TrainingEffects/README.md` §5 |
| **판정 보류 · 무효 실험** | 단계 O (mret) bias 미학습 → 무효; Ariel M13 역사 삭제 팔 (`build_prefix_mask` 버그) → 무효, 인용 안 함; ctx_ar 40 epoch 판 결과 없음 | `I_SCENE` §5-2 · `TRAINED_…` 정정 12 |
| **레지스트리 이름 ≠ 파일** | `vith_ariel_ar_ep18` · `_ar_ep36` 은 지금 ep40 파일을 가리킨다 (10-10 확인); `vith_ariel_bc_*` 는 실행 불가 | §1-2 |

출처 (§7): 위 표의 각 문서 · `z_research/TrainingEffects/Archive/VERIFICATION_2026-09-25.md` · `configs/protocols/models.md`

---

## 8. 철회 · 정정 (이 장 주제) → 통합 표는 10 장

| # | 옛 문장 | 지금 | 왜 | 어디 | 날짜 |
|---|---|---|---|---|---|
| 1 | "학습 도메인 안에서는 진짜 상태 진화처럼 보인다" (v11 post-FT) (state 프레임 — 현재 금지) | **철회** — 학습 궤적 재생 | v11 속도 1 종 · held-out 도 같은 궤적 | `predictor_IntPhysGenV11_PFT/Archive/RESULTS_2026-09-16.md` §3-1 | 09-17 |
| 2 | "v11 post-FT 는 문맥 속도를 따르지 않는다 · 진실 2 배" | **틀림** — 기울기 0.86 + 고정 앞섬 +44 px | own − null 지표 한계 | `TrainingEffects/v3_motion/MOTION.md` 정정 3 · `VERIFICATION` (4) | 09-25 |
| 3 | "v11_postft stride 3 → 3 배 과잉 외삽" | **철회** — b ≤ 1, 고정 앞섬 2–3 칸 | 회귀 재적합 | `TRAINED_PREDICTORS_SIGNATURES` 정정 5 | 09-25 |
| 4 | "Ariel 은 자연 영상으로만 학습 → 도메인 암기로 설명 안 됨" | **철회** | 릴리즈 사전학습 VideoMix22M 에 SSv2 · K400 포함 | `TRAINED_…` 정정 11 | 09-25 |
| 5 | "pv1 도 역사 교체에 둔감 → carried state 구조 처방" (state 프레임) | **철회**; Ariel 역사 삭제 팔 **무효** | hist_swap 장면 절단 교란 · `build_prefix_mask` 버그 | `TRAINED_…` 정정 12 | 09-25 |
| 6 | "pv1 · v11_postft 는 물체 유지가 크게 낫다" | **철회** | 물체다움이 위치 무관, 문턱 이식 불가 | `TRAINED_…` 정정 6 | 09-25 |
| 7 | "제자리에 두는 것은 v11_e10 · pv1_e15" | v11_e10 **하나**; pv1 = 출구 prior | 빈 장면 귀무를 빼면 | `PRESENCE.md` 정정 1 · 2 | 09-25 |
| 8 | "가속 · 감속을 넣는 predictor 없음" | "넣는다는 **증거가 없다**" (약화) | ρ 가 더하기꼴 앞섬에 끌림, 간격 ≈ CI | `MOTION.md` 정정 4 · `VERIFICATION` (6) | 09-25 |
| 9 | arc · ledge 보정 값 (릴리즈 arc −9.5 / −22.7 등) | **인용 금지** — 부호 버그 | 자 치우침을 두 배로 더함 | `MOTION.md` 정정 2 | 09-25 |
| 10 | "Ariel 은 IntPhys 1 세 칸 모두 copy 보다 CI 로 낮다" | 표준 표적에서만 (인과 표적은 skip2_w32 한 칸) | 표적 정의 · 학습 지평 교란 | `IP1_COPY.md` 정정 1 | 09-25 |
| 11 | "IntPhys 1 학습 곡선은 첫 checkpoint 에 끝난다" | **철회** — skip2_w16 에서 epoch 마다 더 떨어짐 | 한 칸에서 고른 결론 | `IP1_COPY.md` 정정 2 | 09-25 |
| 12 | v11_postft IntPhys 1 80.83 | 77.22 | 80.83 은 철회된 Max 지표 (`skip2_w32/max`) | `TRAINED_…` 정정 8 · `RESULTS_2026-09-19.md` | 09-25 |
| 13 | "배치는 shape 만 · color 는 어디서도 안 오른다" · block bootstrap "짝 비교는 괜찮다" | 두 속성 모두 · (궤적, k) 묶음 bootstrap | 적대 검증 | `IDENTITY.md` 정정 | 09-25 |
| 14 | "Ariel 을 더 학습해도 reach 는 안 는다" | EK100 확정 · SSv2 시사 · 팬 미결 | lr cosine 교락 (peak 를 1e-5 로 잘못 읽음) | `VERIFY_ISCENE2` §5-3 | 09-26 |
| 15 | "팬 증강 post-FT 로 Ariel 수준까지" | 팬 (간격의 88 %) · EK100 (동률) 만, SSv2 52 % | 짝 Δ 재계산 | `VERIFY_ISCENE2` §5-4 | 09-26 |
| 16 | AR "슬롯 4 부터 뭉개짐 · 가림막을 더 그린다" · "3 걸음 안에" | 슬롯 5 (릴리즈도 같음) · 가림막은 **덜** (k=4 −.045 vs −.104) · 붕괴는 슬롯 1 부터 | 독립 재계산 | `VERIFY_AR` | 09-27 |
| 17 | AR 은 "복사가 못 푸는 것을 푼다" | "오류가 복사 오류와 덜 묶여 있다" | 무작위 겹침 귀무 대비 −10 | `VERIFY_R3` C3 · `COPY_HARD_SUBSET` 머리 | 09-29 |
| 18 | ctx_ar "움직이는 내용은 AR 과 같거나 +.03" · "문맥 인코딩은 원인 아님" | 팬 −.035 ~ −.074 (틀림) · **시사** 로 강등 | ep16 · 구간별 양방향 차 | `VERIFY_R3` A2 · A6 | 09-29 |
| 19 | AR40 flat_a α_acc 음수 · ramp_a −0.99 · 감속 > 1 · "AR 자가 더 좋으니 자 탓 아님" | flat_a 철회 · 나머지 분포 밖 · 자 탓 가능 (AR 자 위치가 더 나쁨 1.16 vs 0.85 칸) | 정의 민감 · rollout 9–16 블록 | `VERIFY_AR40RD` | 09-30 |
| 20 | "EK100 AR ep40 val 21.74" | **인용 금지** — 릴리즈가 돈 것 | `predictor_checkpoint` 키 누락 | `AR_PREDICTOR` §2-9 | 10-01 |
| 21 | "encoder 는 충분하고 릴리즈 predictor 가 병목" (학습 predictor 비교에서) | **철회** (09-25) — 충분성 대조가 진실 위치를 썼다; 10-08 에 다른 근거로 다시 섰다 (07 장) | | `TRAINED_…` 정정 1 | 09-25 |
| 22 | `RESULTS_2026-09-19.md` §3 "이 논문은 frozen 위에서 닫는다" | 옛 프레임 (09-06 폐기) — 지금 목표 자체가 predictor 학습 | 스토리 교체 | CLAUDE.md §12-6 | 09-06 |
| 23 | "L1 로 학습한 예측은 … 수축" (해석 문장) | **삭제** | 목적함수를 원인으로 거는 문장 | `ANALYSIS.md` 정정 6 | 09-25 |
| 24 | "Predictor_v1 vanish '물체→빈' 상승 = 영속 개선" | 교환 ('빈→물체' 100 → 73) | 방향 병기 | `TrainingEffects/README.md` §5 | 09-25 |

---

## 출처

**학습 · 체크포인트**
- `z_training/README.md`
- `z_training/RESULTS_2026-09-19.md`
- `z_training/ARIEL_CHECKPOINTS.md`
- `z_training/runs/_matrix/MATRIX.md`
- `z_training/runs/_matrix/predictor_v1_holdout_l1.md`
- `z_training/runs/*/` (파일 목록 · `config.yaml`, 2026-10-10 확인) · `z_training/ariel/*/latest.pt` · `z_training/ariel_eval/ckpt/` (`torch.load` epoch 확인, 읽기 전용)
- `configs/protocols/models.md` (`vith_pft_*` · `vith_ariel_*` 항목)
- `configs/training/README.md` · `configs/training/datasets.md`

**TrainingEffects (그룹 A · B 비교)**
- `z_research/TrainingEffects/README.md`
- `z_research/TrainingEffects/MODELS.md`
- `z_research/TrainingEffects/Archive/VERIFICATION_2026-09-25.md`
- `z_research/TrainingEffects/scores/SCORES.md`
- `z_research/TrainingEffects/scores/ANALYSIS.md`
- `z_research/TrainingEffects/ip1_copy/IP1_COPY.md`
- `z_research/TrainingEffects/v11_identity/IDENTITY.md`
- `z_research/TrainingEffects/v11_presence/PRESENCE.md`
- `z_research/TrainingEffects/v3_motion/MOTION.md`

**v11 post-FT 위치 · 채점 분해**
- `z_research/predictor_training/predictor_IntPhysGenV11_PFT/README.md`
- `z_research/predictor_training/predictor_IntPhysGenV11_PFT/Archive/RESULTS_2026-09-16.md`
- `z_research/predictor_training/predictor_IntPhysGenV11_PFT/Archive/SCORING_2026-09-16.md`
- `z_research/predictor_training/predictor_IntPhysGenV11_PFT/exp_results/{pft,release}/{v11,v11_color,v11_shape,rollout_v2}/RESULTS.md`

**auto_research (Ariel 세 팔 · 장면 판독 · 복사 대비)**
- `auto_research/paper/PAPER_STORY_2026-10-08.md` §2-3 (v) · §2-4 · §4
- `auto_research/Archive/AR_PREDICTOR_2026-09-27.md`
- `auto_research/Archive/VERIFY_AR_2026-09-27.md`
- `auto_research/Archive/VERIFY_PRESENCE_2026-09-29.md`
- `auto_research/Archive/VERIFY_R3_2026-09-29.md`
- `auto_research/Archive/AR40_READOUT_2026-09-30.md`
- `auto_research/Archive/VERIFY_AR40RD_2026-09-30.md`
- `auto_research/Archive/COPY_HARD_SUBSET_2026-09-29.md`
- `auto_research/Archive/PREDICTOR_SHARE_2026-09-29.md`
- `auto_research/Archive/ENCODER_MOTION_READOUT_2026-09-29.md`
- `auto_research/Archive/TRAINED_PREDICTORS_SIGNATURES_2026-09-25.md`
- `auto_research/Archive/I_SCENE_CAUSAL_BOTTLENECK_2026-09-25.md` §4 · §5
- `auto_research/Archive/VERIFY_ISCENE2_2026-09-26.md`
- `auto_research/Archive/SC1_REACH_NOT_HORIZON_2026-09-25.md`
- `auto_research/Archive/INTPHYS2_PERMANENCE_AUDIT_2026-09-26.md`

**벤치마크 · 기타**
- `z_research/Benchmarks/exp_results/intphys2/bench_intphys2_main_ariel_bc_ep30_w16/column_best.md` (· `_w32` · `_w48`)
- `V-JEPA 2 predictor 분석 ~하면 되지 않을까 에 답하는 실험 제안.md` (레포 루트, 실험 3 gap-recovery)
