# 03 — testbed 와 측정: 무엇으로 어떻게 쟀나

> **결론** — P3 의 모든 수치는 **문맥 픽셀이 같은 쌍 (matched pairing)** 위의 latent L1 surprise, **복사 기준선 대비 Δ**, 그리고 **검증을 통과한 자 (readout) 또는 파라미터 0 검사** 로 쟀다. intuitive physics 는 행동 네 칸 (1 읽기 · 2 지속 · 3a 추세 · 3b 사건 · 4 영속) 으로 판정한다.
> **근거** — (1) v11 본체 21,504 clip / 10,752 matched 쌍, 문맥 무결성 172,032 프레임 쌍 mismatch 0 (`z_research/IntPhysGenV11/Archive/surprising_score/RESULTS_2026-08-30.md` §1), `max|p_pos − p_imp| = 0.0` (CLAUDE.md §1-2 · `configs/protocols/README.md`). (2) 복사 기준선 = 마지막 문맥 LN(z) 를 미래 전 슬롯에 둔 채점 — IntPhys 1 85.00 vs predictor 88.89 (`auto_research/paper/PAPER_STORY_2026-10-08.md` §2-2). (3) 지금 자 `identity_r8` — v11 판 빈 장면 '있음' 0 % (옛 자 56–99 %), 위치 자 v5 held-out 0.87 칸 (`z_research/RollOutV3/Archive/IDENTITY_R8_2026-09-26.md` §1, `z_research/RollOutV2/figures/v5/summary/POSITION_READOUT_2026-09-12.md` §2-3).
> **미결** — 주 평가를 intuitive physics 판 (IntPhys 1/2 · GRASP · InfLevel + copy-hard · 위치 쌍) 으로 옮기는가 (사용자 결정 대기, 정본 §3). GRASP · InfLevel 의 복사 기준선은 아직 없다.
> 이 장이 받치는 논문 문장: P3 "통제 testbed 로 보니 마지막 관측에 보이는 것만 짧게 외삽하고, 안 보이는 것 · 없는 것은 못 한다" 의 **측정 방법** · P2 의 leakage 정의 · 목표 절의 판정 규칙  ·  원재료: G06 · G07 · G08 · G08b · G09 · G10c · G11 · G12 · G13 추출문 → 원 문서는 각 절 끝 `출처`.

---

## 0. 기호 · 단위 · 등급

| 기호 | 뜻 |
|---|---|
| `z` | 문맥 encoder (online `encoder` 키) 출력, 문맥 16 장만 봄 |
| `h` | target encoder (EMA `target_encoder` 키) 출력, 32 장 전부 봄. 채점 표적은 `LN(h)[future]` (affine-free) |
| `p` | predictor 의 미래 슬롯 출력 (mask token `[0]`) |
| 튜블릿 / 슬롯 | 샘플 2 장 = 원본 6 장. 미래 8 튜블릿 = 슬롯 0–7 |
| 1 칸 | 18 px = 물체 반폭 (RollOutV2 · V3 관례) |
| 자 (readout) | 학습된 decoder. 위치 자 · presence 자 · identity 자 |
| 복사 | 마지막 문맥 튜블릿 LN(z) 를 미래 전 슬롯에 복사한 "예측" |
| A / B 방향 | vanish 에서 A = 물체→빈 미래, B = 빈→물체 미래 |

등급은 원 문서를 따른다: **확정** (적대 검증 또는 10-08 산출물 재확인 ✓) · **1 차** · **시사** · **철회** · **★ 미실행**. 원 문서에 등급이 없으면 "등급 문서에 없음".

---

## 1. testbed 세대 표

### 1-1. VoE testbed (가능 · 불가능 쌍) — blender → IntPhysGen → v11 계열 → 실물 메시

규모는 `data_csv/<세트>/index.csv` 행 수로 다시 셌다 (2026-10-10, 읽기 전용). 쌍 수는 원 문서 값.

| 이름 | 날짜 | 무엇을 조작하나 | 규모 (clip / block / 쌍) | 지금 상태 | 정본 문서 |
|---|---|---|---|---|---|
| blender occlusion_v2 | 08-13 (08-15 재렌더) | 가림 × presence / shape / color, 40 프레임 @ 8 fps, 물체 24 px | 1,700 / 425 / — | 아카이브 (재렌더 전 수치 superseded) | `z_world_model_analysis/RESULTS_2026-08-16.md` |
| blender probe_set | 08-13 (08-15 재렌더) | 가림 없음, 정지 / 이동 × shape · color · floor 8 종 | 1,024 / — / probing 만 | 아카이브 | 같음 |
| blender occlusion_v3 | 08-19 | 탑다운, 물체 48 px, `divergence_frame` 19–29 | 1,760 / 440 / 880 (문맥일치) | 아카이브 | `z_world_model_analysis/FRAME_ABLATION_2026-08-19.md` |
| blender occlusion_v4 | 08-20 | `n_hidden_frames` 3–14 (렌더 시 무작위, 다른 축과 얽힘) | 문서에 없음 | 아카이브 — 결론 "가림 길이가 레버" 는 v10 · v11 에서 뒤집힘 | `z_world_model_analysis/OCCLUSION_LENGTH_2026-08-20.md` |
| IntPhysGen v1 | 08-20 | UE, 원근 카메라, context 16 / future 16 stride 3 | 1,024 / 256 / — | 삭제 (CLAUDE.md §2-6: v1 데이터셋 없음) | `z_world_model_analysis/MECHANISM_2026-08-20.md` |
| IntPhysGen v2–v5 · v9 | 08 월 | 문서에 없음 (v9 는 `UnrealEngine/IntPhysGen_v9_ramp_probe` 표본 폴더만) | 문서에 없음 | 삭제 / 미상 | — |
| IntPhysGen v6 · v6_augmented · v7 | 08 월 | 4 조건 (static/moving × visible/occlusion) × vanish/shape/color | 768 / 192 · 2,048 / 512 · 2,048 / 512 | 레지스트리에서 뺌, 수치는 `z_exp/.../summary.json` | CLAUDE.md §1-3 · §4-3 |
| IntPhysGen v8 | 08 월 | 4 조건, 가림 4+4 | 2,048 / 512 / 1,024 | 아카이브 (`available: false`) | `z_research/IntPhysGenV8/` (md 없음) · `configs/protocols/datasets.md ## v8` |
| v8_halfsize | 08 월 | 물체 28.1 px (크기 통제) | 2,048 / 512 / — | 아카이브 | `datasets.md ## v8_halfsize` |
| 2D transit (`2d_v8_transit`) | 08 월 | 2D 렌더판 v8 | 2,048 / 512 / — | **프레임 · 생성 코드 08-28 삭제**, 캐시 41 GB 만 | `datasets.md ## 2d_v8_transit` |
| IntPhysGen v10 | 08-28 | 4 조건, moving = 경사면 가속 50.06 cm/s², `sym_k: 4`, 물체 150 cm (31.8 px) | 8,192 / 2,048 / 4,096 | 프레임 삭제 · 결과만 | `z_research/IntPhysGenV10/Archive/V10_2026-08-28.md` |
| v10_flat · v10_occ_low | 08-28 | flat = 평지 등속 (가속만 다름) · occ_low = 가림 4 → 1 프레임 | 4,096 / 1,024 / 2,048 · 1,024 / 256 / 512 | 프레임 삭제 | 같음 |
| **IntPhysGen v11 (본체)** | 08-30 | **조건 6 × 위반 3 × sym_k 0–4** (아래 1-2) | **21,504 / 5,376 / 10,752** | **본 실험 (현행)** | `z_research/IntPhysGenV11/README.md` §3 |
| v11_earlymid | 09-01 렌더 | `*_early` (raw 12–21) · `*_mid` (raw 21–30) 문맥 가운데 가림 6 조건 | 21,504 / 5,376 / 10,752 | 현행 | `z_research/IntPhysGenV11_occlusion_timing_ablation/README.md` |
| **v11_full** | 09-01 ~ 09-03 | v11 + v11_earlymid = 12 조건 (visible / early / mid / late) | **43,008 / 10,752 / 21,504** | 현행 (캐시 840 GiB, vll5 에 없음) | 같음 |
| v11_timing | 09-02 | 9 조건 판 | 32,256 / 8,064 / 16,128 | **대체됨 → v11_full** (캐시 뽑지 말 것) | `datasets.md ## v11_timing` |
| v11_occtiming (pilot) | 09-01 이전 | early/mid 가 **2k** 장 가림 (후속은 k) | 12,096 / 3,024 / 6,048 | `available: false`, 68.11 % **인용 금지** | `datasets.md ## v11_occtiming` · 68.11 은 `IntPhysGenV11_occlusion_timing_ablation/README.md` |
| v11_split_train / test | 09-10 | v11_full block 절반 (117 셀 층화, seed 0) | train 10,752 clip (가능만) / test 10,752 쌍 | 현행 (predictor 학습 held-out) | `z_training/README.md`, `datasets.md ## v11_split_test` |
| v13_black | 09-03 | 가림막 `black` (v11 은 `wood`) — 가림막 색 가설 대조 | 4,608 / 1,152 / 2,304 (1/9 규모) | 프레임 삭제 · 결과만 | `datasets.md ## v13_black` |
| v11_vanish_all | 09-11 | v11 pos_a 의 vanish clip (z 32 장 캐시용) | 2,688 / — / — | 현행 (위치 readout 전용) | `datasets.md ## v11_vanish_all` |
| **v11_realistic** | 10-06 | 실물 메시 (15–20 종) 로 v11 vanish 재렌더, 6 조건 × k 0–4 × 배경 4, 16 fps | **1,440 / 360 / 720** | 현행 | `z_research/v11_realistic/README.md` §1 |
| **v11_realistic_ledge** | 10-06 | `pos_fall` (포물선 낙하) vs `imp_float` (같은 높이 직진), 속도 3 × 깊이 2 | **240 / 120 / 120** | 현행 (10-08 번복으로 논문 포함, 3b 사건) | 같음 §2 |
| **gravity_realistic** | 10-07 (20:36 재렌더) | 과거 하나 + 미래 다섯 (`arc` · `rise` · `float` · `line` · `stop`), 꼭대기 위치 3 조건 | **720 / 144 / 576 (cross)** | 현행 (옛 미래 넷 판 대체됨) | 같음 §4, `exp_results/gravity_retrieval.md` |

⚠️ **v11 규모 두 판** — CLAUDE.md §2-2 는 "v11 은 2026-09-01 에 43,008 clip 으로 커졌다" 고 쓰지만, `data_csv/intphysgen_v11/index.csv` 는 **21,504 행** 이고 v11 채점 (73.37 %, 10,752 쌍) 은 이 판이다. 43,008 clip 은 `v11_full` (12 조건 · 21,504 쌍) 이다 (`data_csv/intphysgen_v11_full/index_probe.csv` 43,008 행). 두 판의 수치를 섞지 않는다 (v11 73.37 ↔ v11_full 76.00).
⚠️ **v8 가림 대칭 문서 모순** — `datasets.md ## v8` 은 "가림 4+4 대칭" 이라 쓰고 `## v10` 은 "v8 은 4/5 로 비대칭이었다" 고 쓴다. 어느 쪽이 맞는지 이 아카이브는 확인하지 않았다.
⚠️ **gravity_realistic 의 g** — README §4 · 생성 로그 `UnrealEngine/gravreal_219837.log` 는 현행 판 g 120 cm/s² (옛 미래 넷 판 g 170). `datasets.md ## gravity_realistic` 안에는 "g 120" · "g = 101" · "실제 g 170" 이 섞여 있다 (옛 판 문구가 남은 것으로 보인다).

### 1-2. v11 설계 축 (본 testbed)

| 축 | 값 | 비고 | 출처 |
|---|---|---|---|
| `condition` | `static / moving_flat / moving(ramp)` × `visible / occlusion` = **6** | k 를 접지 않음 (접으면 probing head 216, 지금 63) | `IntPhysGenV11/README.md` §3 |
| `sym_k` | 0 (가림 없음) / 1 / 2 / 3 / 4 — **문맥 끝 k 장 + 미래 앞 k 장** 대칭 | `n_context_hidden == k` 실측 확인. 가림막 폭 17.5 % (k=1) → 36.9 % (k=4) — 조작의 구현 (§7) | `RESULTS_2026-08-30.md` §1 |
| `violation_type` | vanish (불가능 2,688) / shape (4,032) / color (4,032) | shape 7 종 21 쌍 전부, color 8 종 28 쌍 전부 | 같음 |
| 속도 | ramp v₀ 116 cm/s (가속 49.1 cm/s²) · flat 260 cm/s · static 0 | k 와 무관하게 고정 (가림 길이는 가림막 폭으로) | 같음 |
| 물체 · 배경 | 190 cm (화면 12.4 %, 224 px 기준 27.7 px), silhouette 7 종 등화 (`shape_scale_fix`), env 4 종 | 축 정렬 bound 로 맞추면 cube 면적 2.6 배 — 그래서 실루엣 등화 | 같음 · `TALK_2026-08-30.md` §3 |
| 프레임 | 34 장 저장 · 프로토콜 32 장 사용 (raw 100, stride 3) | 원본 PNG 직독 | `datasets.md ## v11` |
| 셀 크기 | 조건 × 위반 vanish 448 / shape · color 672 쌍; k 칸 168; 방향당 16 (shape k=0) / 4 (k>0) | **방향당 n=4 칸은 읽지 않는다** | `RESULTS` 표 A · E |

**v11_full 타이밍 축** (`IntPhysGenV11_occlusion_timing_ablation/README.md`): `visible` (가림막 없음) / `early` (문맥 raw 12–21, 문맥 k 장 숨김) / `mid` (raw 21–30) / `late` (= v11 본체, 문맥 끝 k + 미래 앞 k = **2k**). 총 가림 장수 짝 `early k=2 ↔ late k=1`. ⚠️ late 는 "문맥 끝 가림" 과 "미래 앞 가림" 을 함께 바꾼다 — 문맥 끝만 가리는 팔이 없어 둘을 못 가른다.

**v11_realistic** (`v11_realistic/README.md`): 288×288, 16 fps, 물체 겉보기 35.6 px. 칸당 30 쌍 · 방향별 15 — 선 모양만 읽는다. ledge = 속도 140 · 180 · 220 cm/s × 깊이 z 300 · 400, 조건당 20 쌍, 가림 없음. gravity = 조건 `arc_pre` (꼭대기 f66, 문맥은 오르는 중) · `arc_apex` (f48, 경계 = 꼭대기) · `arc_post` (f30, 내려가는 중), 조건당 48 block = 공 6 × 배경 4 × 방향 2.

### 1-3. 운동 · 위치 세트 (RollOut)

| 이름 | 날짜 | 무엇을 조작하나 | 규모 | 지금 상태 | 정본 문서 |
|---|---|---|---|---|---|
| RollOut_v1 | 09-06 전후 | 등속 직선, 위치 readout 방법 첫 판 | 7,392 clip | **프레임 · 캐시 · 결과 삭제 (09-10~11)** | `z_research/scripts/_superseded/README.md` (폴더 `RollOutV1/` 없음) |
| **RollOut_v2** | 09-10 재생성 (09-09 판은 wall 불가능이 정지 궤적 복사라 폐기) | 7 시나리오 `flat_v flat_a ramp_a arc fall ledge wall`, ledge · wall 은 pos 1 + imp 1 | 5,488 clip (가능 4,704) | 현행 (vll5) | `z_research/RollOutV2/README.md` |
| rollout_v2_decel | 09-19 | `flat_d` · `ramp_d` 감속 (가능만) | 1,568 → 합쳐 9 시나리오 **7,056 clip** | 현행 | 같음 §4b |
| **RollOut_v3** | 09-22 | 8 법칙 × 392, 64 프레임, 사건 **f32** 고정, 16 창 (C · P ∈ {4,8,16,32}), 불가능 392 (`wall_pass` · `gravity_float`, 문맥 최대차 0.000 px) | 3,136 clip (가능 2,744) | 현행 | `z_research/RollOutV3/README.md` · `_superseded/README_2026-09-25.md` §0 |

⚠️ RollOut_v2 의 VoE 채점 (`surprise_c16t32`) 은 ledge · wall 에서만 의미가 있다 (block 당 pos 1 + imp 1). ledge 쌍은 프레임 35 에서 갈려 **슬롯 0 의 h 차이가 갈림 시점 누수** 다 (정본 §2-3 iii 조건 2).

### 1-4. 자 (readout) 학습셋

| 이름 | 규모 | 무엇 | 지금 상태 | 출처 |
|---|---|---|---|---|
| `rollout_v2_training` v1–v4 | — | 첫 위치 자 학습셋들 | **09-11 전부 삭제** | `POSITION_READOUT_2026-09-12.md` §6 · §8 |
| v3 (kink · 고속) | — | 위치 자 학습셋 변형 | **삭제 · 재시도 금지** (학습 분포 안에서도 자가 흐려짐) | CLAUDE.md §6 |
| **`rollout_v2_training_v5`** | 8,064 clip (사물 없음 4,480 + 사물 1 개 3,584), 무중력 등속, 위치 544/544 셀 | RollOutV2 위치 자 (현행) | 현행 | `datasets.md ## rollout_v2_training_v5` |
| `rollout_v2_training_v6` | 8,352 = v5 + 빈 장면 288 | 첫 presence 자 (v6 자) 음성 | 기록 (원천 재생성 불가, 특징 캐시 09-26 삭제) | `datasets.md ## rollout_v2_training_v6` |
| `rollout_v2_training_v8` (= `training_v8`) | 14,360 (v5 4,480 + props 3,584 + 증축 6,296) | presence 자 · 옛 identity 자 | 기록 (특징 캐시 09-26 삭제) | `datasets.md ## rollout_v2_training_v8` |
| **`training_r8`** (= 렌더 이름 `RollOut_v2_training_v8`) | **8,640 clip** (v11 판 arm 물체 뒤 + 빈 짝 1,920) | **identity_r8** (현행 자) | 현행 — 감사 3 차 통과 | `RollOutV3/audit/training_sets/RollOut_v2_training_v8.md` |

⚠️ **이름 함정** — 레지스트리 `rollout_v2_training_v8` (14,360) 과 렌더 폴더 `RollOut_v2_training_v8` (= `training_r8`, 8,640) 은 **다른 세트** 다. `identity_r8/summary.json` 의 `train_set` 이 `training_v8` 로 잘못 적혀 있던 것을 09-26 에 고쳤다 (`RollOutV3/exp_results/README.md`).

### 1-5. predictor 학습셋 (08 장에서 쓰는 것 — 목록만)

| 이름 | 규모 | 용도 · 주의 | 출처 |
|---|---|---|---|
| `v11_split_train` | 10,752 clip (가능만) | `v11_postft`. ⚠️ v11 · v11_full 전체 채점은 문맥을 공유 → 학습 후 점수로 읽지 않음 | `configs/training/datasets.md` |
| `intphys1_train` | 3,750 scene (가능만) | `intphys1_postft`, dev 비오염 | 같음 |
| `predictor_v1_training` | 14,250 (렌더 20,000) | line · occ (트랩도어 k=4) · ramp (a 5–60 cm/s²) 세 팔, held-out 모양 torus · cone 등 | 같음, `UnrealEngine/gen/PREDICTOR_V1_TRAINING_DESIGN.md` |
| `intphys2_main_split_train` | 254 clip / 127 장면 | `intphys2_postft`. ⚠️ 논문 A.3 이 Main 학습 금지 → 유효 test 는 HeldOut | 같음 |
| `ssv2_min48` + K400 `train_min96` | 97,416 + K400 | Ariel 세 팔 (관측 전용 실영상) | `z_training/ARIEL_CHECKPOINTS.md` |

### 1-6. 외부 벤치마크 (규모만 — 상세는 02 장)

| 벤치마크 | 규모 | 쌍짓기 | 출처 |
|---|---|---|---|
| IntPhys 1 dev | 360 영상 = 90 4 중항 → **180 matched 쌍**, O1 영속 · O2 모양 · O3 연속 | matched 2 쌍/블록 | `z_research/Benchmarks/PROTOCOLS.md` |
| IntPhys 2 Main | 253 scene × 4 = 1,012 영상 → **506 쌍** | `type` 번호 = 쌍. **2×2 cross 금지** | `analysis/intphys2/README.md` |
| GRASP level2 | 16 시나리오 × P/IP × 128 = 4,096 영상, 2,048 쌍 | 같은 시나리오 · 같은 인덱스 (Garrido A.4 해석) | `datasets.md ## grasp_level2` |
| InfLevel-lab | continuity 1,116 · gravity 1,182 · solidity 450 쌍 | 공식 csv 한 줄 = 한 쌍, priming 절단. gravity · solidity 는 원리적으로 못 푼다 (Garrido App. E) | `PROTOCOLS.md` §5-3 |

### 출처 (§1)
`z_research/IntPhysGenV11/README.md` · `z_research/IntPhysGenV11/Archive/surprising_score/RESULTS_2026-08-30.md` · `z_research/IntPhysGenV11/Archive/TALK_2026-08-30.md` · `z_research/IntPhysGenV11_occlusion_timing_ablation/README.md` · `z_research/IntPhysGenV10/Archive/V10_2026-08-28.md` · `z_world_model_analysis/{RESULTS_2026-08-16,FRAME_ABLATION_2026-08-19,OCCLUSION_LENGTH_2026-08-20,MECHANISM_2026-08-20}.md` · `z_research/v11_realistic/README.md` · `z_research/RollOutV2/README.md` · `z_research/RollOutV3/README.md` · `z_research/RollOutV3/_superseded/README_2026-09-25.md` · `z_research/RollOutV3/audit/training_sets/RollOut_v2_training_v8.md` · `z_research/RollOutV3/exp_results/README.md` · `z_research/scripts/_superseded/README.md` · `configs/protocols/datasets.md` · `configs/training/datasets.md` · `z_research/Benchmarks/PROTOCOLS.md` · `analysis/intphys2/README.md`

---

## 2. 표준 채점

### 2-1. 식

```
surprise(clip) = mean_{미래 토큰} | predictor(z_ctx)  −  LN( target_encoder(clip 32 장) )[future] |       (latent L1, loss_exp 1)
정답 ⟺ surprise(impossible) > surprise(possible)          chance 50 %, 동점 = 오답 (Garrido 규칙)
```

- `z_ctx` = 문맥 encoder 가 **문맥 16 장** 을 masked forward 로 본 출력. mask token 은 `[0]` 하나만 (릴리즈 ckpt 에서 `mask_tokens[1..9]` 는 norm 0 — `analysis/intphys2/ANALYSIS_NOTES.md` T6).
- 학습 손실과 같은 식이다 (`app/vjepa/train.py:432,447`; `z_training` 의 `loss = mean|p − h|`). 채점은 학습 목표의 값을 그대로 읽는다.
- LN 은 affine-free → `‖token‖₂ = sqrt(1280)` 상수 (`evals/world_model_analysis/eval.py:542, :748`). 그래서 "`s ∝ ‖LN(h)‖`" 류 설명은 항등적으로 거짓 (`HYPOTHESIS_SCALAR_ORDER_2026-08-30.md` §3-3).

### 2-2. matched pairing — 왜, 그리고 함정

| 쌍 종류 | 정의 | 판정 |
|---|---|---|
| **문맥일치 (matched) 2 쌍** | 4 중항 (pos_a · pos_b · imp_ab · imp_ba) 중 문맥 픽셀이 같은 두 쌍 | **표준.** 입력이 같아 `p` 가 비트 단위 동일 (`max\|p_pos − p_imp\| = 0.0`, 08-16 실측 `0.00e+00`) → 채점이 "p 가 h_pos 와 h_imp 중 어디에 가까운가" 라는 순수 기하 문제 |
| 미래일치 2 쌍 | 미래가 같은 두 쌍 | 금지 — `p` 부터 달라 누수를 흡수한다 (`FRAME_ABLATION_2026-08-19.md` §2) |
| cross 4 쌍 전수 | 가능 × 불가능 4 조합 | 금지 (아래 함정) |

- 08-13 occlusion_v2 판은 "future-matched 가 공정" (presence 0.9427) 이라 권고했으나 08-16 부터 문맥일치가 유일한 통제 비교 (presence 50.00 = 100 / 0) 로 뒤집혔다 (0.9427 은 `z_world_model_analysis/occlusion_v2_2026-08-13_PREVIOUS-DATA.md`, 뒤집힘은 `z_world_model_analysis/RESULTS_2026-08-16.md` §1.4). **철회된 쪽은 미래일치 권고.**
- ⚠️ **`scoring.pairing` 기본값은 `"cross"`** — 명시하지 않으면 공식과 다른 전수 4 쌍 채점이 조용히 돈다. `surprise_c16t32.yaml` 은 `pairing: matched` 를 ★ 로 박아 둔다.
- 예외 하나: **`gravity_realistic` 은 `SET="scoring.pairing=cross"`** — block 하나가 문맥 하나라 cross (가능 1 × 불가능 4) 가 곧 문맥 일치 쌍 전부다. v11 처럼 문맥이 둘인 4 중항에 cross 를 쓰면 안 된다 (`datasets.md ## gravity_realistic`).
- ⚠️ 용어 충돌 — CLAUDE.md §1-2 의 "미래일치 쌍의 누수" 와 정본 §2-2 의 **leakage (= 예측 없이 풀리는 정도)** 는 다른 뜻이다. 논문에서는 leakage 를 한 번 정의하고 쓴다 (§3).

### 2-3. 고정 스펙 — `surprise_c16t32` (= CLAUDE.md §1-7)

| 항목 | 값 | 왜 / 함정 |
|---|---|---|
| 프레임 | raw 100 장 stride 3 → 32 장 (`0,3,…,93`) | 원본 PNG 직독, 리사이즈 `antialias=False` bilinear (공식 커널) |
| 문맥 / 미래 | 문맥 = 첫 16 장 (raw 0–45), 미래 = 마지막 16 장 (raw 48–93) | target encoder 는 **32 장 전부** |
| 모델 | ViT-H `facebook/vjepa2-vith-fpc64-256`, `img_size 256`, `window_size 32`, RoPE, `uniform_power: false` | 기준 모델 ViT-H 는 외부 벤치마크에서 먼저 정함 (IntPhys 1 88.89 vs ViT-L 64.44) |
| 역할 분리 | `dual_encoder: true`, `context_encoder_key: encoder` (online), `target_encoder_key: target_encoder` (EMA) | 학습 때의 역할 분리 재현 |
| 정밀도 | `dtype: float32` + `autocast: float16` | **probing 은 `bfloat16`** — 섞지 말 것 |
| 채점 | `target_layer_norm: true` · `distance: l1` · `loss_exp: 1.0` · `mask_index: 0` | `mask_index: 0` critical |
| 쌍 | `scoring.pairing: matched`, `breakdown: [block_type]` | |
| 진입점 | `bash z_research/scripts/run.sh surprise_c16t32 <데이터셋> vith` (= `cvpr/eval/testbed/run.sh`) | `evals.main` 직접 호출 금지 (DDP 포트 · `WMA_EXPECT_WS` 가드) |

⚠️ IntPhys 2 프로토콜 대조 (`analysis/intphys2/PROTOCOL_CHECK_2026-09-14.md`) 는 IntPhys 2 하네스에 `dtype: float32` + `autocast: bfloat16` 을 권하는데 CLAUDE.md 는 "미반영" 이라 적는다 — IntPhys 2 수치는 이 정밀도 차이를 안고 있다.

### 2-4. 방향 병기와 sensitivity / bias 분해

| 지표 | 식 | 쓰는 이유 | 출처 |
|---|---|---|---|
| A / B 방향 정확도 | vanish: A = 물체→빈 미래가 정답 쌍, B = 빈→물체 | scalar surprise 의 외형 편향을 숨기지 않는다. **v11 ramp 가림 (`moving_occlusion`) vanish 50 % = A 0.00 / B 100.00** (전 k, 방향당 56); flat 가림 52.0 은 A 12.50 → 0.00 (k=1→4) / B 100.00 | `RESULTS` 표 D |
| sensitivity | `(acc(A→B) + acc(B→A)) / 2 − 50` | 탐지력 (대칭 성분) | `RESULTS` §1 |
| bias | `\|acc(A→B) − acc(B→A)\| / 2` | 방향 편향 (반대칭 성분). `acc = 50 + sens + b` | 같음 · `HYPOTHESIS_SCALAR_ORDER` §3-1 |

⚠️ CI 폭 0 인 셀은 버그가 아니다 (100 % / 0 % 상쇄 구조). ⚠️ recall–precision 음의 상관은 구조적이다 — 정보는 두 폭의 대비에 있다 (`VOTE_GRID_2026-08-31.md` §4).

### 2-5. 정합성 검사 (모든 세트에 돈 것)

| 검사 | 기대값 | 실측 | 출처 |
|---|---|---|---|
| byte-identical context 감사 (`--strict-context`) | 모든 matched 쌍의 16 문맥 프레임 픽셀 동일 | v6 ~ v8 · 2D · v10 (65,536 쌍) · v11 (172,032) · v11_earlymid (172,032) · v13_black (36,864) · v11_realistic (720) · ledge (120) · gravity (144 × 4) 전부 **mismatch 0** | `data_csv/<세트>/context_integrity.json`, `datasets.md` 각 절 |
| `p` 비트 동일 | `max\|p_pos − p_imp\| = 0.0` | 0.0 (v8 alpha 스크립트도 매번 검증) | `RESULTS_2026-08-16.md` §1.4 · `PREDICTOR_HEDGING_2026-08-26.md` |
| `\|z(p)\|` 단독 채점 | **정확히 0.00 % (전 쌍 tie)** | 0.00 % — 파이프라인 정합성 증거 | CLAUDE.md §7-4 |
| tie 수 | 0 | v11 n_ties 0 · ledge tie 0 | `RESULTS` §2 · 정본 §재현 |
| 재계산 대조 | `per_block.json` 재합산 == `summary.json` overall | v11 73.3654 == summary · v11_full cell 117 ↔ per_block 불일치 0 · gravity 쌍 정확도 85.2 일치 | `RESULTS` §2 · timing `figures/README.md` · `gravity_retrieval.md` |
| 결정론 | 같은 config 재실행 전 자리 일치 | token_subset `all` `max\|Δ\| = 1.2e-07` | CLAUDE.md §7-2 |

### 출처 (§2)
`configs/protocols/surprise_c16t32.yaml` · `configs/protocols/README.md` · `z_world_model_analysis/RESULTS_2026-08-16.md` · `z_world_model_analysis/FRAME_ABLATION_2026-08-19.md` · `z_research/IntPhysGenV11/Archive/surprising_score/{RESULTS_2026-08-30,HYPOTHESIS_SCALAR_ORDER_2026-08-30,VOTE_GRID_2026-08-31}.md` · `analysis/intphys2/ANALYSIS_NOTES.md` · `analysis/intphys2/PROTOCOL_CHECK_2026-09-14.md` · `configs/protocols/datasets.md` · `z_research/v11_realistic/exp_results/gravity_retrieval.md`

---

## 3. 복사 기준선 · leakage · copy-hard · 위치만 다른 쌍

### 3-1. 정의

| 용어 | 정의 | 출처 |
|---|---|---|
| **복사 기준선 (copy)** | 문맥 **마지막 튜블릿의 LN(z)** 를 미래 전 슬롯에 그대로 둔 "예측" 으로 같은 채점을 한 정확도. predictor 를 안 쓴다 | `auto_research/Archive/COPY_HARD_SUBSET_2026-09-29.md` §1 (H6 관례) |
| **leakage** | 미래를 예측하지 않고 관측만으로 (마지막 관측과 실제 미래의 변화를 비교해서) 정답이 나오는 정도 = **복사 기준선 정확도** | 정본 `PAPER_STORY_2026-10-08.md` §2-2 (사용자 10-08) |
| **predictor 몫** | predictor − 복사 (Δ, cluster bootstrap CI) | 같음 |
| **copy-hard (쌍 단위)** | 복사가 틀리거나 동점인 쌍 (o_copy < 1). 칸 단위: 복사 정확도 ≤ 50 인 칸, 천장 칸 = ≥ 95 | `COPY_HARD_SUBSET` §1 |
| **위치만 다른 쌍** | 외형은 같고 미래 위치 · 궤적만 다른 가능 / 불가능 쌍 (RollOut_v2 ledge 낙하 vs 부유, wall 정지 vs 통과, 392 쌍씩) | 정본 §2-2 표 · `auto_research/Archive/COPY_VS_POSITION_2026-10-07.md` |

변형: IntPhys 1 은 copy(LN) 85.00 · copy_raw 77.78 (인과 표적 · LN 없음) — 보고는 LN 판 (`z_research/Benchmarks/Archive/INTPHYS1_EVENT_POSITION_2026-09-25.md` §2). AR 팔은 표적 공간이 블록별이라 `copy_blk` (블록별 LN(target_encoder) 복사) 와 비교한다. DINO-Foresight 는 "같은 공간 복사" (`model.dinof_copy=true`).

### 3-2. 왜 모든 점수에 복사 대비 Δ 를 내나

1. **VoE 점수는 예측 없이 풀릴 수 있다.** IntPhys 1 은 복사만으로 85.00 — predictor 88.89 와 Δ +3.9 [−0.6, +8.9], CI 가 0 을 포함 (확정, 정본 §2-2).
2. **복사가 틀리는 쌍에서 릴리즈 predictor 는 우연이다** — v11 (`v11_split_test`) 48.4 [46.6, 50.2] (n 2,877) · IntPhys 1 44.4 (n 27) · IntPhys 2 21.2 (n 240). 복사가 맞는 쌍에서는 85.7 / 96.7 / 84.2 (1 차, `COPY_HARD_SUBSET` §0).
3. **학습이 올린 점수도 복사가 이미 맞는 쌍 위에서 난다** — Ariel prefix · full · post-FT ip2 의 copy-hard 49.6–51.0 (같음). 그래서 목표 절의 합격선은 "copy-hard · 위치 쌍에서 오른다" 로 쓴다 (정본 §2-4 방법 절 3).
4. 하네스 질문 5 (정본 §6): 복사 기준선을 같이 내고, metric 이 위치를 못 보는 교란이 남으면 **그 결과로 못 하는 주장을 미리 적는다**.

⚠️ **적대 검증 3 차 정정** (`auto_research/Archive/VERIFY_R3_2026-09-29.md` C3, `COPY_HARD_SUBSET` 머리) — 복사와 오류가 무작위로 겹치는 귀무 모델에서 copy-hard 정확도 = 전체 정확도다. 모든 predictor 가 그 아래 (release −27, prefix −28, AR −10, v11_e10 −9). 그래서 AR 의 copy-hard 73.8 은 "복사가 못 푸는 것을 푼다" 가 아니라 **"오류가 복사의 오류와 덜 묶여 있다"** 로 읽는다.
⚠️ copy-hard 쌍의 구성이 위반마다 다르다 — v11 2,877 쌍 = color 1,571 · shape 1,023 · vanish→빈 207 · vanish→물체 76; late 1,272 (같음 §2-1). copy-hard 평균 하나로 "보이는 / 안 보이는 / 없는 것" 을 섞지 않는다.

### 3-3. predictor vs 복사 — 현재 표 (프로토콜이 달라 행끼리 비교하지 않는다)

| 판 · 프로토콜 · 모델 | predictor | 복사 | Δ / 판정 | 등급 | 출처 |
|---|---:|---:|---|---|---|
| IntPhys 1 dev · Garrido `skip2_w32` Filtered · ViT-H (180 쌍) | 88.89 | 85.00 [79.4, 90.0] | +3.9 [−0.6, +8.9] n.s. | 확정 ✓ | 정본 §2-2 · `TrainingEffects/ip1_copy/IP1_COPY.md` |
| 〃 사건이 예측 구간 안인 창만 (V1) | 91.67 | 87.22 | +4.4 [+0.6, +8.3] | 등급 문서에 없음 | `INTPHYS1_EVENT_POSITION_2026-09-25.md` §2 |
| IntPhys 2 Main · IntPhys 2 하네스 · ViT-H (506 쌍) | 54.35 (C 24) | 55.53 (C 42) | 같은 C 에서 +1.8 [−1.8, +5.5] | 확정 ✓ | `auto_research/Archive/INTPHYS2_PERMANENCE_AUDIT_2026-09-26.md` |
| IntPhys 1 dev · DINO-Foresight highres (`skip2_w16`) | 84.4 | 88.9–92.2 | 복사가 더 높다 | 1 차 | `auto_research/Archive/DINOF_INTPHYS1_2026-10-02.md` |
| v11 · `surprise_c16t32` · DINO-Foresight | 77.10 | 76.44 | 가림 세 조건 chance, vanish A/B 0/100 | 1 차 ✓ | `auto_research/Archive/DINOF_V11_2026-10-07.md` |
| v11 `v11_split_test` · `surprise_c16t32` · ViT-H 릴리즈 (10,752 쌍) | 75.7 | 73.2 | copy-hard 48.4 | 1 차 | `COPY_HARD_SUBSET` §2-1 |
| v11 문맥 끝 가림 (late) 칸 · ViT-H | 50–70 | 39–63 | 예측이 필요한 칸에서 우연 근처 | 확정 | 정본 §2-2 (`RETHINK_PREDICTOR_ROLE_2026-09-29.md` §5-b) |
| RollOut_v2 위치만 다른 쌍 (ledge / wall, 392 쌍씩) · ViT-H | 0.0 / 17.6 | 7.9 / 2.3 | predictor 는 불가능 쪽을 고른다 | 1 차 ✓ json | `COPY_VS_POSITION_2026-10-07.md` · `auto_research/exp_results/scene/rollout2_copy_voe.json` |
| EK100 action anticipation · released 규약 | 35.34 | (규약 누수: 문맥의 88.6 % 가 정답 action 프레임을 담음) | paper 규약 19.07 | 등급 문서에 없음 | `z_research/anticipation/EK100/README.md` §2-A |

⚠️ 위치 쌍의 VoE metric (토큰 평균 L1) 은 위치를 거의 못 본다 (`COPY_VS_POSITION` §2) — 0 % 는 "metric 한계 + predictor 의 틀린 미래" 가 섞인 값이다. **위치 주장은 자 판독 (§5) 과 같이** 쓴다.
⚠️ ledge · 부유 미래는 복사와 닮았다 — 복사 LN(z) 도 59–99 % 가 부유 쪽 (`auto_research/Archive/H23_PREDICTOR_LAYERS_TRAJECTORY_2026-09-24.md` §3-7, 정본 §2-3 iii 조건 1). VoE 쌍 하나로 "불가능한 연속을 예측한다" 와 "그냥 머문다" 를 못 가른다 → **다중 미래 설계** (§3-4).
⚠️ "IntPhys 2 가 낮은 것은 leakage 가 작아서" 는 상관 서술이다 — 장면 복잡도 등이 섞일 수 있다. 인과로 쓰려면 같은 장면에서 leakage 만 바꾼 쌍 (v11 의 가림 위치 축) 을 같이 보인다 (정본 §2-2).

### 3-4. 파라미터 0 의 다중 미래 판정 (gravity_realistic 형식)

| 항목 | 내용 |
|---|---|
| 판 | 과거 하나 + 미래 다섯 (`arc` 가능 · `rise` · `float` · `line` · `stop`) |
| 고른 미래 | argmin surprise, **chance 20 %**. 쌍 정확도 (가능 vs 각 불가능) 와 따로 보고 |
| 기준선 | 문맥 끝 수직 속도로 만든 위치 기준선 (등속 → arc / 높이 유지 → float / 멈춤 → stop) |
| 분해 | 튜블릿별 (전체 토큰 / 물체 토큰) · 공 자리 / 그림자 / 나머지 칸 기여 |
| 왜 | 불가능 미래 하나가 복사와 같은 방향이면 (ledge 부유) VoE 가 "머묾" 과 "불가능 예측" 을 못 가른다. 미래 여럿 중 '멈춤' 같은 가장 덜 움직이는 미래를 거의 안 고르면 (stop 1/144) "덜 움직이는 미래가 L1 로 싸다" 만으로는 설명되지 않는다 |
| 단서 | 튜블릿 하나의 argmin 은 흔들린다 — 줄의 흐름으로만. 꼭대기 문맥 arc 62.5 % 는 공 자리가 아니라 나머지 칸 (+1.51) 이 공 자리 (−0.77) 를 뒤집은 것 (공 자리는 98 % block 이 float 쪽, 10-08 정정) |

출처: `z_research/v11_realistic/README.md` §4 · `exp_results/gravity_retrieval.md` §1–7. 다음 할 것 4 번 (ledge · wall 다중 미래 세트) 이 이 형식을 쓴다 (정본 §5).

### 출처 (§3)
`auto_research/paper/PAPER_STORY_2026-10-08.md` §2-2 · §2-3 · §2-4 · §6 · `auto_research/Archive/COPY_HARD_SUBSET_2026-09-29.md` · `auto_research/Archive/VERIFY_R3_2026-09-29.md` · `auto_research/Archive/COPY_VS_POSITION_2026-10-07.md` · `auto_research/Archive/INTPHYS2_PERMANENCE_AUDIT_2026-09-26.md` · `auto_research/Archive/DINOF_INTPHYS1_2026-10-02.md` · `auto_research/Archive/DINOF_V11_2026-10-07.md` · `auto_research/paper/RETHINK_PREDICTOR_ROLE_2026-09-29.md` · `z_research/TrainingEffects/ip1_copy/IP1_COPY.md` · `z_research/Benchmarks/Archive/INTPHYS1_EVENT_POSITION_2026-09-25.md` · `z_research/anticipation/EK100/README.md` · `z_research/v11_realistic/README.md` · `z_research/v11_realistic/exp_results/gravity_retrieval.md`

---

## 4. 프로토콜

### 4-1. 프로토콜 표

| 이름 | 무엇을 재나 | 핵심 설정 | 확립된 기준값 | 진입점 · config | 정본 문서 |
|---|---|---|---|---|---|
| **`surprise_c16t32`** | 고정 단창 latent L1 surprise (testbed 표준) | §2-3 | v11 **73.37 %** (10,752 쌍) · v8 79.10 % · v11_full 76.00 % (21,504) | `run.sh surprise_c16t32 <D>` · `configs/protocols/surprise_c16t32.yaml` | `IntPhysGenV11/README.md` §4 |
| **`intphys1_sliding`** | Garrido et al. sliding 창 (IntPhys 1 · GRASP · InfLevel) | `S_t = ‖p(f(V[t:t+C])) − LN(g(V[t:t+C+M]))‖₁`, stride 2 전수 창, A.8 격자 skip [2,5,10] × 창 [16,32] × C [2,4,6,8,10]×(C+M)/16 (InfLevel skip [5,10,20]), **IntPhys 1 은 Filtered (시작점마다 min over C)**, GRASP · InfLevel 은 property 별 최고 C, property macro, **AvgSurprise 하나** (Max 철회 09-21), 동점 오답 | IntPhys 1 dev ViT-H **88.89** (`skip2_w32`); 같은 격자 `skip2_w16` 83.33 · `skip5_w16` 61.11 · 노트북 규칙 90.56 (참고) | `run_all.sh` · `configs/protocols/intphys1_sliding.yaml` · 기준값 검사 `check_oracle.py` | `z_research/Benchmarks/PROTOCOLS.md` |
| **`attn_probe`** | z / p / h 세 지점 attentive probing (정보가 있는가) | attentive (16 head, `num_probe_blocks: 1`), `attn_30` = 35 epoch · bs 16 · lr 1e-4 · wd 0.05, **bfloat16**, `index_probe.csv` 의 `obj` 만, block split 0.5 seed 0 stratify condition | v11 54 항목 (fit 3 × group 6 × target 3), 등가속 + 가림 `p` shape 98.46 / color 99.49 / env 100 | `run.sh attn_probe <D>` · SLURM 분할 | `RESULTS_2026-08-30.md` 제 2 부 |
| `attn_probe_imp` | 불가능 변이에서 target encoder 가 바뀐 정체성을 읽는가 (라벨 의미가 다른 프로토콜) | runs 에 target 둘만 (z · p 는 문맥일치 쌍에서 같은 텐서) | — (v10 에서만 실행; 지금 돌릴 데이터셋 없음) | `configs/protocols/attn_probe_imp.yaml` | `V10_2026-08-28.md` · `cvpr/README.md` |
| `attn_probe_xfer` | 표현을 가로지르는 이식 (h → p, p → h) — 예측 미래가 진짜 미래와 같은 좌표계인가 | `extends: attn_probe`, eval 목록만 다름 (head 수 같음 108 = 12 조건 × 3 지점 × 3 target). z 는 문맥 구간이라 이식 대상에서 뺌 | v11_full: 비가림 head → late 가림 `p` colour 15.1 / 20.6 / 16.4; `h→p` 비가림에서 94.9 / 79.0 / 57.3 | `run.sh attn_probe_xfer v11_full vith` | `IntPhysGenV11_occlusion_timing_ablation/README.md` |
| IntPhys 2 하네스 (③, 별도) | Bordes et al. 프로토콜 | 6 fps 고정, `frames_per_clip = C + M`, stride 2, **`max_context_mode`** (맨 앞 창 C 2,4,…), 창 16/32/48, C = 창 × {¼,⅜,½,⅝,¾,⅞}, 쌍 = `type` 번호, AvgSurprise, **열마다 최고 실행** (descriptive) | Main ViT-H w48 **54.35** (C 24; 논문 57.51) · w16 53.16 · w32 52.77 | `python -m analysis.intphys2.eval --config …` · `analysis/intphys2/configs/` | `analysis/intphys2/README.md` |
| EK100 action anticipation (⑤) | frozen encoder + predictor → attentive probe (focal, 20 ep, global batch 128), val 9,296 전수, mean class recall@5 | **released 규약** (문맥이 action **끝** 1 s 전에 끝) vs **paper 규약** (action **시작** 1 s 전에 끝) | released 56.36 / 56.16 / **35.34** (논문 59.2 / 54.6 / 36.5) · paper **19.07** · released ckpt 를 paper 규약으로 평가 14.14 | `z_research/anticipation/EK100/run.sh` | `z_research/anticipation/EK100/README.md` |
| AR 채점 (`ar_scoring`) | 자기회귀 predictor (Ariel 팔 C) | 문맥 · 타깃 = **블록별** LN(target_encoder), 미래는 rollout — 표준 타깃과 다른 공간 | **쌍 정확도만** 비교 (손실 · surprise 절대값 비교 금지) | `analysis/predictors/ar_scoring.py` | CLAUDE.md §3-1 · `z_training/ARIEL_CHECKPOINTS.md` |

⚠️ `configs/protocols/README.md` 하단 "현재 프로토콜 4 개" 표는 `attn_probe_xfer` 추가 뒤 갱신되지 않았다 (상단은 5 종) — 문서 내부 불일치.
⚠️ A.8 격자 최고값은 같은 격자에서 고른 값이라 **held-out 추정치가 아니라 descriptive** 다. 창 (C+M) 마다 실행을 나눈다 — 공식은 창마다 모델을 그 프레임 수로 짓는다 (혼합 격자 산출물은 `_invalid_20260921_mixedgrid/` 로 무효).
⚠️ 우리 IntPhys 2 하네스를 IntPhys 1 에 걸면 67.22 / 66.11 (논문 Table 2 의 IntPhys 1 열 87.22 와 20 pt) — **원인 미규명**. IntPhys 1 은 Garrido 하네스로만 보고한다 (`PROTOCOLS.md` §7). Table 2 의 IntPhys 1 열 87.22 는 IntPhys 2 프로토콜 값이라 88.89 와 자가 다르다.
⚠️ EK100 기준은 **released 규약** (사용자 09-20) — 논문과 비교 가능한 유일한 축. released × frame 규약에서 val 문맥의 88.6 % 가 정답 action 프레임 (평균 14.00 / 32 장) 을 담는다. "EK100 에서 못 한다" 는 쓰지 않는다 (19.07 은 우연 0.45 의 42 배).

### 4-2. 프로토콜이 다른 숫자 — 비교 금지 (CLAUDE.md §1-3 + 이 아카이브가 모은 것)

| 비교 | 값 | 흔들림 | 출처 |
|---|---|---:|---|
| IntPhys 1: Garrido sliding vs 우리 fixed C16 | 88.89 vs 66.67 | **22.2 pt** | CLAUDE.md §1-3 |
| IntPhys 1: A.8 최고 칸 vs Table S3 칸 | 88.89 (`skip2_w32`) vs 83.33 (`skip2_w16`) | 5.6 pt | `PROTOCOLS.md` §2 |
| IntPhys 1: Garrido vs IntPhys 2 프로토콜 | 88.89 vs 67.22 / 66.11 (우리 구현) · 87.22 (논문 Table 2) | ~21 pt | `analysis/intphys2/configs/_superseded/README.md` |
| IntPhys 1: 08-13 자체 재구성 (C32/M8, 4 쌍 전수) | 0.6750 (360 pair) | — 헤드라인 아님 | `z_world_model_analysis/intphys1_dev_2026-08-13.md` |
| IntPhys 1: 최고 단창 (start 24, C 8) vs sliding | 87.78 · 88.3 · 88.33 (같은 창, 반올림 차 추정) vs 88.89 | 1 pt | G07 (c)17 |
| v6aug: fixed / sliding avg / sliding max | 75.49 / 76.76 / 79.49 | 4.0 pt | CLAUDE.md §1-3 |
| v6aug static + occluded shape: fixed vs sliding max | 46.88 vs **90.62** | **43.7 pt** | 같음 |
| v11 vs v11_full | 73.37 (6 조건) vs 76.00 (12 조건) | 조건 구성이 다름 | §1-1 |
| v11 전체 vs `v11_split_test` | 73.37 vs 75.83 (릴리즈, 12 조건 test 절반) | 표본이 다름 | `z_training/README.md` |
| knockout 층 8 단일 Δ | −18.06 (`knockout_window__v11_vith`, 288 쌍 표본) vs −14.4 (`knockout_w1`, v11_full) | 표본 다름 | `IntPhysGenV11/figures/knockout/README.md` (−18.06) · `IntPhysGenV11/Archive/PAPER_STORY_2026-09-06.md` §재현 (−14.4) |
| knockout 기준선 vs 공식 채점 | 75.86 (clean_null) vs 76.00 | 0.14 pt (SDPA 커널) | timing README §전수 11 |
| probing `attn_probe_xfer` vs `attn_probe` 같은 self 칸 | 108 칸 중 1 개 6.1 pt 차 (미수렴 head) | 섞지 말 것 | timing README 단서 |
| 다른 모델 계열 surprise 절대값 | V-JEPA 2 latent 1280 · 2.1-g 5632 · VideoMAEv2 픽셀 1176 · DINO-F PCA 1152 | 절대값 비교 금지 (정확도만) | `Benchmarks/README.md` · `configs/protocols/models.md` |

### 4-3. 진입점 (요약 — 실행 규칙은 CLAUDE.md §2)

| 무엇 | 옛 진입점 (그대로 돈다) | 새 진입점 (10-09) |
|---|---|---|
| testbed 채점 · probing | `z_research/scripts/run.sh <프로토콜> <데이터셋> [모델]` | `cvpr/eval/testbed/run.sh` · `cvpr/analysis/probing/run.sh` |
| 외부 벤치마크 | `z_research/Benchmarks/run_all.sh` | `cvpr/eval/intphys1/run.sh` (`WINDOWS=garrido_skip2_w32` = 88.89 칸) |
| IntPhys 2 | `analysis/intphys2/run_grid.sh` | `cvpr/eval/intphys2/run.sh` (창 `ip2_w48` 기본) |
| EK100 | `z_research/anticipation/EK100/run.sh` | `cvpr/eval/ek100/run.sh` |
| 복사 기준선 | `TrainingEffects` · `auto_research/scripts` | `COPY=1` 노브 (`copy_baseline.{sh,py}`, testbed · intphys1) |

`cvpr/harness/check_equivalence.py` — 옛 resolver 와 같은 config 를 내는지 28 행: **동일 27 · 차이 0 · 건너뜀 1** (`attn_probe_imp v11`, 인덱스 없음) (`cvpr/README.md`). 10-09 smoke 는 rc=0 만 확인, 수치 대조는 안 했다.

### 출처 (§4)
`configs/protocols/{README.md,surprise_c16t32.yaml,intphys1_sliding.yaml,attn_probe.yaml,attn_probe_imp.yaml,attn_probe_xfer.yaml,models.md}` · `z_research/Benchmarks/{README,PROTOCOLS}.md` · `analysis/intphys2/README.md` · `analysis/intphys2/configs/_superseded/README.md` · `z_research/anticipation/EK100/README.md` · `z_research/IntPhysGenV11_occlusion_timing_ablation/README.md` · `z_research/IntPhysGenV11/figures/knockout/README.md` · `z_world_model_analysis/intphys1_dev_2026-08-13.md` · `z_training/README.md` · `z_training/ARIEL_CHECKPOINTS.md` · `cvpr/README.md`

---

## 5. probing 규칙 · 자 (readout)

### 5-1. probing 규칙

| 규칙 | 내용 | 왜 | 출처 |
|---|---|---|---|
| **가능 변이만** | `index_probe.csv` 의 `probe_type == obj` (가능 + 물체 있음) 만 학습 · 평가 | 불가능 변이를 넣으면 세 표현이 전부 오염 — `contextF` · `pred` 는 문맥일치 쌍에서 비트 동일, `targetF` 는 라벨 `before` · 렌더 `after` 라 **encoder 가 정확할수록 0 점** | CLAUDE.md §1-4 |
| 예외 | `attn_probe_imp` 만 불가능 변이 — "target encoder 가 바뀐 정체성을 읽는가" 라는 다른 질문, 라벨 의미가 다름 | | 같음 |
| **block 단위 split** | `split: {mode: ratio, group_by: block_id, train_frac: 0.5, seed: 0, stratify_by: <축>}` | block 안 4 개가 2×2 로 문맥 · 미래를 공유 | CLAUDE.md §1-5 |
| **이식 줄만 대조** | self 칸은 천장 (head 49.3 M 과대용량, contextF · targetF `train_acc 1.000`), 대조는 이식 줄 (비가림 → 가림 등) 에서만 | v11 `z`·`h` 51 / 54 칸 천장 | CLAUDE.md §7-3 · `SYNTHESIS_2026-08-30.md` §3-4 |
| 팔을 가로지르는 이식 실패 | 증거가 아니라 **기대값** (attention query 가 위치 특이적) | 정지 → 이동 70.1 vs 이동 → 정지 17.3 도 같은 이유 | `RESULTS_2026-08-30.md` §12 |
| self 는 `per_group` 에서 | `summary.json` 의 `overall` 은 조건 평균이라 이식 실패가 섞임 (v11 `p` overall 65.01) — 인용 금지 | 09-02 에 혼동한 적 있음 | 정본 §재현 · timing `figures/README.md` · 65.01 은 `RESULTS_2026-08-30.md` 표 G · `OVERVIEW_2026-09-22.md` §5-4 |
| `group_column: condition` 이면 | `pos_a/pos_b` 필터 불가 → 파생 컬럼 + `type_column` + `probing.block_types` | | CLAUDE.md §3-2 |
| prep 과소추출 | `probing.runs` 를 줄이면 캐시에 뽑히는 표현도 준다 → runs 는 두고 `num_epochs` 만 1 | | CLAUDE.md §7-1 |
| DDP probing 흔들림 | 결정론이 필요하면 `GPUS=1`; `batch_size` 는 global | | CLAUDE.md §7-2 |
| 미수렴 head | v11 `p` static_occlusion color 0.931 · shape 0.954 · moving_occlusion shape 0.978 | static_occlusion `p` self 85.13 이 낮은 것은 표현 탓인지 최적화 탓인지 **미해결** | `RESULTS` §7 · §11-(5) |

⚠️ 용어 충돌 — v11 timing 의 `shape consistency / colour consistency` 는 **채점 축**이고 probing target `shape / color` 와 다르다 (같은 조건 `p` probe 98.2 vs 채점 54.5).
⚠️ shape 은 encoder 도 흔들린다 (`z` 3 팔 평균 64.8, 등속 팔 42.4) — "encoder 는 조건과 무관" 은 color · env 로만.

### 5-2. 자 (readout) 세대 표

| 세대 | 날짜 | 구조 · 파라미터 | 학습셋 | 검증 수치 (결과와 같은 무게) | 지금 상태 · 물러난 이유 | 출처 |
|---|---|---|---|---|---|---|
| RollOut_v1 위치 readout | 09-06 | 공간평균 특징 + h 로 배운 pooled / token 자 이식 | RollOut_v1 | — | **폐기 (09-10)** — 프레임 · 캐시 삭제, "학습셋에서 자를 정한다" 로 교체 | `z_research/scripts/_superseded/README.md` · `z_research/v11_roll_out/README.md` |
| v4 위치 자 | 09-11 이전 | attentive | v4 학습셋 | encoder 자 사물 셀 1.2–2.2 칸 | 대체 → v5 (사물 셀 0.4–0.6 칸); p 판정은 같았다 | `POSITION_READOUT_2026-09-12.md` §6 |
| v11 k=0 p 만 자 | 09-12 | attentive | v11 k=0 `p` 슬롯 0–2 (336 clip) | held-out 3–5 px 인데 물체 칸 질량 0.10–0.16 | **기각** — 자리 prior 를 외움 (early 슬롯 0–2 에서 47–63 px) | 같음 §2-3 E |
| **v5 위치 자 (RollOutV2, 현행)** | 09-12 | query 1 → softmax over 256 토큰 → Linear(1280→2), **3,842**, 튜블릿마다 독립, p · z · h 각각 | `training_v5` 8,064 | 학습셋 R² p 0.959/0.956 · z 0.989/0.969 · h 0.988/0.979; **held-out 0.87 칸 (학습 0.83)**; **encoder 자 v2 7 시나리오 전부 ≤ 0.61 칸**, gain x 0.99–1.09; v11 은 가림 여부 · 타이밍 무관 추적; **자 없는 검사와 슬롯 단위 일치**; 회귀 계수 같음 (자 질량 −0.43 / 자 없는 적중 −0.47 per 100 px) | **현행** (RollOutV2 · v11 위치). 감속도 읽는다 (h · z gain 1.00–1.15) | `POSITION_READOUT_2026-09-12.md` §2 · `PREDICTOR_DISTANCE_LIMIT_2026-09-19.md` |
| v6 자 (첫 presence) | 09-23 | `attn` = attention pooling + xy · presence head, **5,123** (heat · mass · heatmap · linear 기각) | `training_v6` 8,352 | v6 ROC p .988 / z .996 / h .997; v3 에서 z '없다' 0.4–4.4 % (mass 2.9–14.1 %); 좌표 편향 v3 x b −13.6 · y b −28.7 px | 대체 → training_v8 로 재학습 (09-24). 옛 세트는 구조물이 물체 없는 clip 에만 → "구조물 = 없음" 학습 (v3 ledge · ramp · wall recall 81–85 % vs 평면 97–98 %) | `RollOutV3/Archive/READOUT_CHOICE_2026-09-23.md` · `exp_results/_superseded/windows_v6_20260923/` |
| presence 자 `dcd24d8a8a47` | 09-24 | `attn`, 5,123 | `training_v8` 14,360 | test p 중심 1.06 칸 · ROC 0.986 · 빈 장면 오탐 1.4 %; z 0.66 칸 · 0.997 | **물러남 (09-25)** — v11 가림막 (처음 보는 재질 `MI_OccDark`, 물체 앞) 을 물체로 읽음: 판 빈 장면 '있음' p **7–45 %**, h 에서도 뜸. presence 하나로는 "만들었는데 안 움직임" vs "못 만듦" 을 못 가름 | `exp_results/presence/RESULTS.md` · `audit/training_v8/AUDIT.md` §E |
| identity 자 `aeda55874c1c` | 09-25 저녁 | 위치 + 57-way (모양 7 × 색 8 + 없음) | `training_v8` | — | **물러남 (09-26)** — v11 판 빈 장면 '있음' **56–99 %**, '있음' 자리가 판 위 44–100 % | `IDENTITY_R8_2026-09-26.md` §1 |
| **identity_r8 `01c3afaac37f` (현행)** | 09-26 | query 1 + attention pooling → 위치 + 57-way, '있음' = argmax ≠ 없음 (**튜닝 문턱 없음**) | **`training_r8` 8,640** (감사 3 차 통과) | test 조합 p 84.0 / z 90.5 / h 90.9 %; 없음 AUROC 0.9997–1.0000; '있음' 오탐 p 0.27 / z 0.05 / h 0.03 %; 위치 p 13.0 / z 8.9 / h 9.8 px; **v11 판 빈 장면 p · h '있음' 0 %**; v11 다시 보인 칸 h 97–100 % | **현행** (RollOutV3) | `IDENTITY_R8_2026-09-26.md` §3 · `RollOutV3/README.md` |
| 자 A / 자 B (FUTURE_FROM_CONTEXT) | 10-01 ~ 02 | A: 튜블릿당 query 1 (3,842) · B: 슬롯별 query 8, 토큰 전체 attend (**12,802**) | RollOut_v2 가능 4,480 (holdout = 안 본 속도 2 개) | z 자 ledge y 3.3–4.3 px; h 자는 불가능 ledge (떠 있음) 에서 낙하 13–16 px 만 (가능 57–59) → 장면이면 무조건 떨어뜨리는 자는 아님 | 1 차. ⚠️ **지도 라벨 곡선을 붙이므로 "p 가 낙하를 담는다" · "encoder 가 미래를 표현한다" 의 근거가 못 된다** | `auto_research/Archive/FUTURE_FROM_CONTEXT_2026-10-02.md` §1 · §3 |
| 방향 tiny probe (실영상) | 09-20 | query 1 + linear **3,842** (v5 자와 같은 구조) · attentive 18.0 M | ssv2 · ntu 정방향 clip | 정방향 test ssv2 91.2 / ntu 100.0; **뒤집힘 비율** 85.2 / 96.2 (라벨 불필요) | 현행 (1 읽기 칸). "방향" 과 "끝 자리" 를 못 가른다 | `context_encoder_analysis/Archive/ENCODER_TEMPORAL_DYNAMICS_2026-09-20.md` |

### 5-3. 자 검증 규칙 (사용자 09-12: "decoder 문제 없다를 보여야 predictor 가 만드는 세상에 대해서 주장할 수 있지")

| 검증 | 무엇을 보이나 | 예 |
|---|---|---|
| **held-out 정밀도** | 자가 학습 분포 밖에서도 읽는다 | v5 0.87 칸 · identity_r8 test (단 **외형 held-out, 궤적 held-out 아님** — test 물체 clip 의 46 % 가 같은 궤적을 train 에) |
| **encoder 대조** | 같은 자가 z · h 에서는 맞게 읽는다 → p 에서 틀리면 p 의 문제 | v5 encoder 자 ≤ 0.61 칸 · identity_r8 v11 다시 보인 칸 h 97–100 % |
| **파라미터 없는 교차검증** | 자 없이도 같은 결론 | 아래 표 |
| 빈 장면 오탐 | 대상이 없을 때 '있음' 을 안 낸다 | identity_r8 v11 판 빈 장면 0 % (옛 자 56–99 %) |
| 학습셋 감사 | 라벨과 장면 속성 사이 누수 | `training_r8` 3 차: 판 위치 x MI 0.0033 bits (귀무 0.0009) · 전부 0.041 (귀무 0.014) · 빈 clip 1,920 전부 같은 장면 짝 |

**파라미터 0 검사 목록**

| 검사 | 식 · 규칙 | 쓰인 곳 |
|---|---|---|
| 자 없는 r | `r = \|p − h_imp\| / \|p − h_pos\|` (진실 물체 3×3 토큰, 채점기와 같은 L1) | RollOutV2 v11 슬롯별 · ledge |
| 자 없는 적중 | LN(h) 진실 물체 토큰 템플릿 × p 슬롯 256 토큰 cos argmax, 1 칸 안이면 적중 (우연 ≈ 0.035) | `PREDICTOR_DISTANCE_LIMIT` §4 |
| 자 없는 vanish block 비교 | 문맥일치 쌍에서 p 가 물체 미래 쪽인 block 비율 | RollOutV3 `RESULTS_2026-09-25.md` §5 |
| 다중 미래 argmin | §3-4 | gravity_realistic |
| 역재생 뒤집힘 비율 | 같은 clip 의 정방향 예측 ≠ 역재생 예측 비율 | ENCODER_TEMPORAL_DYNAMICS |
| 시간 정렬 행렬 (gated) | `Δc = c_S − c_BG` 로 배경 몫을 뺀 물체 몫 (배경 토큰만으로도 argmin 이 슬롯 5 까지 대각선이라 gate 필수) | `ctxenc_time_alignment_gated/RESULTS.md` |

### 5-4. 자 읽는 규칙 (기본값 함정)

1. **자는 물체가 없어도 위치를 낸다** — attention 이 퍼지면 토큰 평균 읽기 (기본값). v11 기본값 ≈ 마지막 관측 위치 ≈ 화면 중심, wall 기본값 ≈ 정지점. 그래서 **위치 주장은 그 슬롯의 3×3 attention 질량이 균등 (0.035) 을 넘을 때만** 한다 (`RollOutV2/README.md` 읽는 규칙 1).
2. 8 슬롯 평균 지표 (슬롯 7 진행 비율 등) 는 기본값을 섞는다 — 슬롯별 표 (`attn_diag.md`, `two_futures_attn.md`) 와 같이 읽는다.
3. "X 에 둔다" 와 "물체가 없다" 는 다른 도구다 (위치 자 vs presence · identity 자 / 자 없는 r).
4. identity_r8: 위치 주장은 '있음' 칸에서만; **t7 은 표에서 뺀다** (p 오탐이 예측 창 마지막 자리에서 최고); 반쯤 가린 칸 · 출구 칸은 자 정의 밖.
5. 자 attention 지도 (top1) 로는 판정하지 않는다 (v6 "물체 있음" 0.262 vs "화면 밖" 0.267).
6. 속도 그림은 위치 차분이라 자 잡음이 두 배 — 후반 지그재그를 값으로 읽지 않는다.

이 규칙으로 **철회된 문장** (10 장에 통합): "p 가 물체를 마지막 자리에 둔다" (기본값; 마지막 관측 칸 질량 0.01–0.03) · "wall 에서 벽 속 한 칸 반 파고들어 멈춘다" (슬롯 1–2 통과 + 3–7 기본값의 평균) · "p 가 슬롯 4 이후 멈춘다" (물체를 놓친 뒤의 기본값) · 자 gain 0.4–0.5 = "시간이 절반쯤 흐른다" · presence 자의 "t8 절벽 · 깜빡임".

### 5-5. 자로 판정하지 않는 것

| 무엇 | 왜 | 대신 | 출처 |
|---|---|---|---|
| wall · ledge 가능 / 불가능 (`attn` 자) | 분리량 P=8 에서 28.7 px 인데 자 오차 15–20 px, 좌표 편향 Δx +25 px 가 결론을 뒤집음 | 자 없는 잠재 공간 검사 `\|p−h_pos\|` vs `\|p−h_imp\|` · 다중 미래 | `READOUT_CHOICE_2026-09-23.md` ② |
| "p 가 낙하를 담는다" | 자 B 는 장면 종류를 읽고 지도 라벨 곡선을 붙인다; 등속 학습셋 자 (v5) 는 선반 높이 유지 → **자 둘이 다른 답** | 두 결과를 조건 표로 같이 (정본 §2-3 iii 조건 3) | `FUTURE_FROM_CONTEXT` §1 |
| 가림 '없음' 판정 (자 학습셋에 가림 음성 넣기) | 사용자 결정 09-25: 안 넣는다 | 자 없는 vanish 짝 비교 | `RollOutV3/_superseded/README_2026-09-25.md` §4-2 |
| static 물체 가림 영속 (presence 자) | trapdoor 빈 장면 오탐 0.23–0.92 | 판정 불가로 둠 | `audit/v11_presence/AUDIT.md` |
| 정체 유지 = 영속 | p 미래 정체 95–97 % 는 문맥 복사 (z_all 98–100) 이하, [z_all, p] 로 안 늚 | "p 는 문맥이 가진 정체를 옮겨 싣는다" 까지만 | `audit/v11_identity/AUDIT.md` §3 |

### 출처 (§5)
CLAUDE.md §1-4 · §1-5 · §3 · §7-3 · `z_research/IntPhysGenV11/Archive/surprising_score/RESULTS_2026-08-30.md` · `z_research/IntPhysGenV11/Archive/SYNTHESIS_2026-08-30.md` · `z_research/IntPhysGenV11_occlusion_timing_ablation/{README,figures/README}.md` · `z_research/RollOutV2/README.md` · `z_research/RollOutV2/figures/v5/summary/POSITION_READOUT_2026-09-12.md` · `z_research/RollOutV2/Archive/PREDICTOR_DISTANCE_LIMIT_2026-09-19.md` · `z_research/RollOutV3/{README.md,Archive/IDENTITY_R8_2026-09-26.md,Archive/READOUT_CHOICE_2026-09-23.md,Archive/RESULTS_2026-09-25.md,exp_results/README.md,exp_results/presence/RESULTS.md}` · `z_research/RollOutV3/audit/{training_v8,v11_presence,v11_identity}/AUDIT.md` · `z_research/RollOutV3/audit/training_sets/RollOut_v2_training_v8.md` · `z_research/RollOutV3/exp_results/_superseded/windows_v6_20260923/{CROSS_HEAD,WINDOW_SUMMARY}.md` · `z_research/RollOutV3/_superseded/README_2026-09-25.md` · `z_research/v11_roll_out/README.md` · `z_research/scripts/_superseded/README.md` · `auto_research/Archive/FUTURE_FROM_CONTEXT_2026-10-02.md` · `z_research/context_encoder_analysis/Archive/ENCODER_TEMPORAL_DYNAMICS_2026-09-20.md` · `z_research/context_encoder_analysis/exp_results/ctxenc_time_alignment_gated/RESULTS.md`

---

## 6. 네 능력 = intuitive physics 의 조작적 정의

### 6-1. 정의 (09-19 사용자 결정, 정의는 그대로 · 10-08 에 읽는 틀만 바뀜)

| # | 능력 | 정의 (원 문서 그대로) | "한다" 로 판정될 관측 |
|---|---|---|---|
| **1** | **읽기** | 관측 (문맥) 에서 상태 — 무엇이 있고, 어디 있고, 어떻게 변하고 있는지 — 를 꺼낼 수 있다 | encoder 표현 (z) 에서 상태 변수가 readout 된다 |
| **2** | **지속** | 새 관측 없이 예측이 길어져도 (시간이 가고, 마지막 관측에서 멀어져도) 상태가 미래 표현에 남는다 | 미래 슬롯의 진실 자리에 물체다운 토큰이 있다 — 거리 · 슬롯과 무관하게 |
| **3a** | **전이 — 추세** | 문맥 내용에 따라 미래를 바꿀 수 있다 — 같은 현재라도 문맥이 드러낸 운동 추세 (방향 · 속도 · 가속도, 꼭대기에서 되돌아옴) 가 다르면 다른 미래 | 문맥 조건끼리 진실 미래가 다를 때 예측 미래도 그만큼 다르다 (clip 간 gain ≈ 1) |
| **3b** | **전이 — 사건** (10-08 추가) | 장면 요소가 촉발하는 변화 — 받침 끝 낙하 · 벽 충돌 | 사건 뒤 미래가 사건을 따른다 (다중 미래에서 가능 미래 선택 · 위치 자) |
| **4** | **영속** | 관측이 끊긴 구간 (가림 · 화면 밖) 을 넘어 이어 가고, 다시 보일 때 맞는 자리에 둔다 | 공백 뒤 미래 슬롯에 물체가 있고 자리가 맞다 |

- 능력 표에 넣지 않는 것: **R 재귀** (미래 토큰끼리 넘기는가) = 기제 진단 · **정합** (예측 미래가 관측과 같은 표현 형식 · 자리로 나오는가; v11 비가림 → 가림 이식 `p` 17.3) = 측정 조건.
- 원칙 (§0): 행동으로 정의 (구조 아님) · 의존 순서 읽기 → 지속 → {전이, 영속} · 칸마다 encoder 천장 (z 또는 h) · 자 검증 · 파라미터 없는 교차검증. **"모델이 중력을 안다 / 물리 법칙을 적용한다" 는 쓰지 않는다.**
- ⚠️ 원 정의문은 "상태" 어휘를 쓴다 (원 문서 표현 — 09-30 이후 memory · state 프레임 금지). 판정은 오른쪽 열의 관측으로만 한다.

### 6-2. 논문 문장과의 대응 (정본 §3, 10-08)

| 논문 (P3) | 마지막 관측 기준 | 능력 칸 | 주 testbed | 주 측정 |
|---|---|---|---|---|
| **보이는 것** 은 짧게 외삽한다 | 마지막 관측에 보이는 물체의 계속 | 2 지속 · reach | RollOut_v2 · v3, gravity 앞 튜블릿, 실영상 reach | 위치 자 v5 · 자 없는 적중 × 거리 · identity_r8 |
| **안 보이는 것** 은 못 한다 | 문맥 끝에 가려진 물체 | 4 영속 | v11 (late) · v11_full timing · v11_realistic | A/B 방향 채점 · 위치 자 슬롯 · identity_r8 '있음' · 자 없는 vanish |
| **없는 것 — 추세** 는 못 한다 | 가속 · 감속 · 꼭대기에서 되돌아옴 | 3a | v11 등가속 (ramp) · RollOut flat_a / flat_d / ramp · gravity apex | 위치 자 gain · 등속 외삽 대비 · 다중 미래 |
| **없는 것 — 사건** 은 못 한다 | 받침 끝 낙하 · 벽 충돌 | 3b | RollOut_v2 ledge / wall · v11_realistic_ledge · (★ 다중 미래 세트) | 위치 자 슬롯별 질량 · 위치 쌍 · ledge VoE |
| **encoder 에는 정보가 있다** | 문맥에서 꺼낼 수 있나 | 1 읽기 | 전부 + 실영상 | z / h probe · ridge · 방향 probe (역재생) |

⚠️ **3b 를 쓰는 조건 넷** (정본 §2-3 iii, 같은 무게): ① 떠 가는 미래는 복사와 닮았다 → 불가능 미래가 복사와 다른 방향인 다중 미래 설계로 ② 갈림 시점 누수 (RollOut_v2 ledge 프레임 35 갈림, 슬롯 0) → 갈림을 예측 구간 안쪽에, 슬롯 0 따로 ③ 자 둘의 다른 답 (§5-5) ④ L1 교란 (같은 높이 직진은 문맥 끝 속도 그대로라 L1 이 작을 수 있다).

### 6-3. 칸별 측정 규칙 (`STATE_EVOLUTION_CAPABILITIES_2026-09-19.md` §3)

| 칸 | 측정 | 지켜야 할 것 |
|---|---|---|
| 1 읽기 | z / h probe (ridge · attentive), block split | h 천장 · 시나리오 단서 통제 (정지 대조 쌍). 가속은 지금 데이터로 못 잰다 — flat_a 는 v0 = 0 이라 a ∝ v, ramp_a 의 a 는 경사각 장면 단서, 교차 쌍은 둘 다 정지인 대조 쌍도 갈림 (AUC 0.20) → (v0 × a) 세트 필요 |
| 2 지속 | 자 없는 적중 (h 템플릿 cos) · 자 attention 질량을 **거리 × 슬롯** 으로 | 자 held-out · encoder 자 동반. 변수는 슬롯보다 거리 |
| 3a 전이 | 조건끼리 진실 미래 차이 vs 예측 차이 (gain · corr) | **지속이 성립하는 슬롯만**. 문맥 끝 속도 맞춘 쌍 |
| 3b 전이 | 다중 미래 · 위치 자 슬롯별 질량 · 위치 쌍 | 조건 넷 (§6-2) |
| 4 영속 | v11 가림 타이밍 × k 의 적중 · 자리 | 마지막으로 보인 자리로부터의 거리가 같은 비가림 슬롯과 비교 (영속 = 지속의 특수한 경우인지 — 09-19 §5-(1) 미실행) |

### 6-4. 성적표 — V-JEPA 2 ViT-H 릴리즈 predictor

| 칸 | 판정 | 근거 수치 (조건 · 도구) | 등급 | 출처 |
|---|:-:|---|---|---|
| 1 읽기 | ✅ | z → 문맥 끝 속도 ridge R² **0.996 / 0.989** (RollOut_v2, block 반반); 등가속 + 가림 `p` probe shape 98.46 / color 99.49 / env 100; 실영상 좌 / 우 방향 뒤집힘 ssv2 98.1 / ntu 96.2 %. 가속 ⬜ (못 잼), 경계에서 가려진 물체 ◐ (경계 토큰 이식 static 20.2 / flat 16.0, 문맥 전체 C 98–99) | 확정 (R²) | `PREDICTOR_DISTANCE_LIMIT_2026-09-19.md` §5-3 · CLAUDE.md §5-2 · `ENCODER_TEMPORAL_DYNAMICS` · `ctxenc_boundary_identity_v2/RESULTS.md` |
| 2 지속 | ❌ | 물체다운 토큰은 마지막 관측 자리에서 **약 2–3 칸 (40–60 px)** 안: 자 없는 적중 슬롯 7 — 0–20 px 0.87 / 40–60 px 0.57 / 80–120 px 0.24; 회귀 거리 100 px 당 −0.47 · 슬롯 −0.28; 등속 arc 수평 p 53 px 에서 멈춤 (진실 92) | 등급 문서에 없음 (정본 §재현 "확인하지 않은 것") | `PREDICTOR_DISTANCE_LIMIT` §3–4 |
| 3a 추세 | ◐ | 방향 · 속도는 잇는다 (flat_v 0.71 칸, gain 0.85); **가속도는 반영 안 함** — flat_a 마지막 슬롯 p **28.1** ≈ 등속 외삽 **32.4** (진실 **54.0**); 감속 flat_d gain 0.26 / corr 0.49; v11 등가속은 가림 없이도 shape 69.0 (정지 94.0 · 등속 89.9); gravity 꼭대기 문맥 공 자리 98 % 가 높이 유지 쪽 | 1 차 ~ 확정 (v11 채점 ✓) | `PREDICTOR_DISTANCE_LIMIT_2026-09-19.md` §3 (flat_a 28.1 / 32.4 / 54.0) · `POSITION_READOUT_2026-09-12.md` §3 (flat_v) · `RollOutV2/README.md` §4b (flat_d) · CLAUDE.md §5-1 · `v11_realistic/README.md` §4 |
| 3b 사건 | ❌ | ledge: 위치 자 슬롯 3–4 부유 궤적 질량 0.43 / 0.39 (낙하 0.17 / 0.19), 392 쌍 100 %; ledge 실물 메시 VoE **0 / 120**; wall: 2 슬롯 통과 (질량 0.61 / 0.48) 뒤 ≤ 0.18 (기본값) | 1 차 ~ 확정 (0/120 ✓) | `POSITION_READOUT_2026-09-12.md` §4-3 · `v11_realistic/README.md` §2 |
| 4 영속 | ❌ | v11 물체다운 토큰 슬롯 late **0** < early/mid **3** < 가림막 없음 **4–5**; 채점 vanish A 0 / B 100 (ramp 가림; flat 가림은 A 0–12.5), k=1–4 평평; 실물 메시 A 0–10 / B 100 | 확정 (채점 ✓) · 1 차 (위치) | CLAUDE.md §5-1 · §5-5 · `v11_realistic` §1 |
| (R 재귀) | — | `mask→mask` 전 층 차단 −1.4 pt (예측 자체는 rel_l1 0.42 움직임) | — | `IntPhysGenV11_occlusion_timing_ablation/README.md` §knockout |
| (정합) | ❌ | 비가림 → 가림 이식 color `z` 97.4 / `h` 99.2 / **`p` 17.3** (세 팔 평균) | 확정 ✓ | `RESULTS_2026-08-30.md` §8 |

**자가 바뀌면 판정이 한 칸 움직인다 — 조건 차이 표**

| 칸 | 09-19 성적표 (v5 위치 자 · 자 없는 적중, RollOut_v2 · v11) | 09-26 RollOutV3 판정 (identity_r8, v3 · v11) | 차이의 출처 |
|---|---|---|---|
| 2 지속 | ❌ (2–3 칸) | ◐ (v3 t2–t15 '있음' 칸의 82–94 % 가 복사보다 진실에 가깝다; 정체는 거리 40–60 px 부터 흐려짐, 3–5 칸 (51–97 px) 넘어 못 따라감) | 재는 양이 다르다 — 물체다운 토큰의 자리 (v5) vs '있음' + 조합 정답 (r8). 둘 다 "짧게 외삽" 과 맞는 방향 |
| 4 영속 | ❌ (late 0 슬롯) | 대부분 ❌ (문맥 끝 가림 t3–t6 p '있음' k=1→4 정지 28→17 · flat 62→37 · ramp 60→25 %; 빈 장면 0 %; 문맥 중간 가림 75–100 %) | r8 은 k 따라 계속 줄고 채점은 k=1–4 평평 — 재는 양이 다르다. **미결 긴장**: 자 없는 vanish 비교는 이동 k=2–4 98–100 % 빈 미래 쪽인데 r8 은 k=2 초반 ramp 58–69 % '있음' → "짧은 가림을 이어 붙인다" 는 조건부 |

### 6-5. 정의의 변천

| 날짜 | 틀 | 바뀐 것 | 출처 |
|---|---|---|---|
| 09-15 | "predictor 는 상태를 이어가지 않는다" — 축 R 재귀 · A 지평 · B 가림 · C 동역학 전환 (구조 + 행동 혼합) | (state 프레임 — 현재 금지) | `CLAIM_NO_STATE_EVOLUTION_2026-09-15.md` (대체됨) |
| **09-19** | **행동으로 정의한 네 능력** (읽기 · 지속 · 전이 · 영속), A → 지속, B → 영속, C · 가속 → 전이. ledge / wall 은 "common sense, 범위 밖" | R 은 기제, 정합은 측정 조건으로 뺌 | `STATE_EVOLUTION_CAPABILITIES_2026-09-19.md` |
| **10-08** | 같은 네 칸을 **intuitive physics 를 따르는 plausible future prediction 의 조작적 정의** 로 읽음. 3 전이 = 3a 추세 + 3b 사건 | **ledge / wall / 꼭대기 번복 → 범위 안** (사용자 "이것도 포함해서 논문 쓸거야") | 같은 문서 상단 blockquote · 정본 §3 |

### 출처 (§6)
`z_research/context_encoder_analysis/Archive/STATE_EVOLUTION_CAPABILITIES_2026-09-19.md` · `z_research/context_encoder_analysis/Archive/CLAIM_NO_STATE_EVOLUTION_2026-09-15.md` · `z_research/context_encoder_analysis/Archive/ENCODER_TEMPORAL_DYNAMICS_2026-09-20.md` · `z_research/context_encoder_analysis/exp_results/{ctxenc_boundary_kinematics,ctxenc_boundary_identity_v2}/RESULTS.md` · `z_research/RollOutV2/README.md` · `z_research/RollOutV2/Archive/PREDICTOR_DISTANCE_LIMIT_2026-09-19.md` · `z_research/RollOutV2/figures/v5/summary/POSITION_READOUT_2026-09-12.md` · `z_research/RollOutV3/README.md` · `z_research/RollOutV3/Archive/IDENTITY_R8_2026-09-26.md` · `z_research/IntPhysGenV11/Archive/surprising_score/RESULTS_2026-08-30.md` · `z_research/IntPhysGenV11_occlusion_timing_ablation/README.md` · `z_research/v11_realistic/README.md` · `auto_research/paper/PAPER_STORY_2026-10-08.md` §2-3 · §3

---

## 7. 설계 결정 (교란 아님) · 데이터 설계 교훈

### 7-1. 의도된 조작 — 다시 교란으로 제기하지 않는다 (CLAUDE.md §8-5)

| 축 | 비교하는 것 | 함께 바뀌는 것 (조작의 구현) | 쓸 수 있는 문장 | 쓸 수 없는 문장 (안 쟀다) |
|---|---|---|---|---|
| **가림 유무** (`visible` vs `occlusion`) | 물체가 가려지는가 | 가림막이 장면에 있는가 | "물체가 가려지면 무너진다. 가려지는 **시간**은 무관하다 (k=1–4 평평)" | "가림막이 화면에 있는 것 자체는 영향이 없다" |
| **지면** (`moving_flat` vs `moving`) | 등속 vs 등가속 | 쐐기 (경사면) 가 장면에 있는가 | "등가속이 등속보다 나쁘다" | "가속만이 원인이고 장면 구성은 무관하다" |
| **k** (v11) | 가려지는 샘플 수 | 가림막 폭 (k=1 17.5 % → k=4 36.9 %) — 속도를 고정했기 때문 | 같음 | — |
| **타이밍** (v11_full late) | 가림이 문맥 끝 경계에 닿는가 | 미래 앞 가림 · 총 가림 장수 2 배 | "경계에 닿을 때만 무너진다" (총량 맞춰도 +20.2 / +17.3) | "늦게 가릴수록 나쁘다" · "미래를 가려서 / 문맥 끝을 가려서" (가르는 팔 없음) |

- "가림막은 있는데 물체를 안 가리는" 조건 신설은 기각 — "가림 유무" 는 곧 "물체가 가려지는가" 이고 가림막 존재는 그 구현이다 (CLAUDE.md §6).
- 가림막 색면 가설은 v13_black 으로 **기각** (colour visible → early −14.1 → −13.5 같음; 대신 shape −6.2 → 0.0 — 나뭇결이 모양을 깎았다).
- 가림 하에서 ramp 54.46 / flat 56.85 는 **바닥 효과** — "가림 하에서 가속 무관" 금지 (`RESULTS` §3-(2) 정정).
- corner case 판정 (09-03 외부 피드백): 문맥 끝 가림은 corner case 비판을 받는다 — 정본 §4-3 이 실영상 reach (SSv2 46 · EK100 51 %) · OcclusionStats 로 답한다 (02 · 05 장).

### 7-2. 데이터 설계 교훈 (CLAUDE.md §11 + 세대마다 배운 것)

| 교훈 | 어디서 배웠나 | 출처 |
|---|---|---|
| **위반이 "관계" 로만 존재하면 안 된다** — splice (정상 A 앞 + 정상 B 뒤) 는 두 반쪽이 각각 자연스러워 구분 안 됨 | blender 세대 | CLAUDE.md §11 |
| **block 안에서 예측 구간 내용량을 맞춘다** — A/B 비대칭은 누수 이용 | 내용량 누수: corr(내용차, surprise차) IntPhys 0.486 vs IntPhysGen v1 0.922; vanish 대칭 설계 → A 100 / B 0 → 원리적 50 % | `WHY_INTPHYS_WORKS_2026-08-20.md` §2–4 |
| **가림 길이는 가림막 크기가 아니라 물체 속도 / 타이밍으로** | blender v4 는 n_hidden · divergence · slab_side 가 얽혀 분리 불가 → v10 `v10_occ_low` 는 `occ_height_cm` 250–450 이 블록마다 달라 높이와 길이를 못 가름 | `OCCLUSION_LENGTH_2026-08-20.md` · `datasets.md ## v10_occ_low` |
| **`condition` 에 축을 접지 않는다** | v11 은 k 를 `sym_k` 로 따로 (접으면 head 216); v11_full 은 타이밍이 condition 에 접혀 12 조건 → probing 을 쪼개 제출 | `IntPhysGenV11/README.md` §3 · timing README |
| **silhouette 을 클래스 전부 동일하게** | v10 은 모양이 폭과 교락 | CLAUDE.md §11 |
| **surface 와 motion 을 공선으로 두지 않는다** | v10 은 surface · motion 완전 공선 → `v10_flat` 팔로 교락을 끊음 (평균 속도 · 시작점 · 도착 프레임 같고 가속만 다름) | `V10_2026-08-28.md` |
| **"물체 하나 = 화면 움직임의 전부" 상태를 깬다** · distractor 속도 강제 (`\|v\| > 100 cm/s`) | 정지 물체는 시간 중앙값 배경에 흡수 | CLAUDE.md §11 |
| **자 학습셋은 test 의 장면 요소 (가림막 · 구조물) 를 포함하고, 빈 장면을 같은 장면 짝으로** | presence 자가 v11 가림막을 물체로 읽음 (재질 · 깊이 다름); training_v8 은 구조물이 물체 없는 clip 에만 → "구조물 = 없음"; r8 1 · 2 차 감사 실패 (판 상태 · 판 위치가 라벨을 예측) | `audit/training_v8/AUDIT.md` §E · `audit/training_sets/RollOut_v2_training_v8.md` |
| **자 학습셋은 위치를 고루 덮는다** (v11 k=0 만으로는 자리 prior) · kink · 고속 섞지 않는다 (v3) | RollOutV2 | `POSITION_READOUT` §2-3 E · CLAUDE.md §6 |
| **갈림 (사건) 은 예측 구간 안쪽에** · 슬롯 0 따로 보고 | RollOut_v2 ledge 프레임 35 갈림 → 슬롯 0 h 차이가 target encoder 가 뒤를 본 몫 | 정본 §2-3 iii 조건 2 |
| **불가능 미래가 복사와 다른 방향으로 움직이게** (다중 미래) | ledge 부유 = 복사와 닮음 | 정본 §2-3 iii 조건 1 · §5-4 |
| **가속을 재려면 (v0 × a) 교차 세트** | flat_a v0 = 0, ramp_a 의 a = 경사각 장면 단서, 교차 쌍은 시나리오 지름길 (AUC 0.20) | `ctxenc_boundary_kinematics/RESULTS.md` · `CLAIM` §2-8 |
| **학습셋에 테스트 궤적 분포를 넓게** — 속도 1 종이면 held-out block 도 같은 궤적 | v11 post-FT "도메인 안 진짜 상태 진화" **철회** (09-17, 궤적 prior) | `predictor_IntPhysGenV11_PFT/Archive/RESULTS_2026-09-16.md` §3-1 |

### 출처 (§7)
CLAUDE.md §6 · §8-5 · §11 · `z_research/IntPhysGenV11/README.md` · `z_research/IntPhysGenV11/Archive/surprising_score/RESULTS_2026-08-30.md` · `z_research/IntPhysGenV11_occlusion_timing_ablation/README.md` · `z_research/IntPhysGenV10/Archive/V10_2026-08-28.md` · `z_world_model_analysis/{WHY_INTPHYS_WORKS,OCCLUSION_LENGTH}_2026-08-20.md` · `configs/protocols/datasets.md` · `z_research/RollOutV3/audit/training_v8/AUDIT.md` · `z_research/RollOutV3/audit/training_sets/RollOut_v2_training_v8.md` · `z_research/RollOutV2/figures/v5/summary/POSITION_READOUT_2026-09-12.md` · `z_research/context_encoder_analysis/exp_results/ctxenc_boundary_kinematics/RESULTS.md` · `z_research/predictor_training/predictor_IntPhysGenV11_PFT/Archive/RESULTS_2026-09-16.md` · `z_research/IntPhysGenV11/Archive/STATUS_AND_CRITIQUE_2026-09-03.md`

---

## 8. 함정 · 검증 습관 (결과를 읽는 데 영향 있는 것만)

| 함정 | 증상 | 대응 | 출처 |
|---|---|---|---|
| **토큰 캐시 서명이 좁다** | `TokenCache.matches()` 는 `video_ids` · 토큰 수 · dtype 만 본다 — 체크포인트 · 해상도 · autocast · 재렌더 · 문맥 길이는 서명에 없어 **조용히 재사용** | 데이터를 다시 만들면 캐시를 치운다 (`cache/_previous_YYYYMMDD/<tag>`); cvpr 는 창이 기본과 다르면 tag 에 접미사 | CLAUDE.md §7-1 · `cvpr/README.md` |
| 캐시 `video_ids` 완전 일치 요구 | 상위 · 부분집합 재사용 불가 — 조건을 하나 더하면 전부 다시 뽑음 (`v11_full` 은 이어 붙이기 840 GiB) | 인덱스를 만들 때 묶을지 나눌지 정한다 | CLAUDE.md §7-1 · `datasets.md ## v11_full` |
| **`data.resolution != model.img_size`** | 작으면 CUDA assert, **크면 에러 없이 엉뚱한 구간을 잘라 씀** | `resolve.py` 가 모델 로드 전에 검사 | CLAUDE.md §7-1 |
| **`scoring.pairing` 기본 cross** | 공식과 다른 4 쌍 채점이 조용히 돔 | 프로토콜 yaml 에 `matched` 명시 | §2-2 |
| **`model.predictor.kind` 미지정** | state_dict 키가 릴리즈와 같아 **조용히 full attention** (출력 상대차 0.5) | 로더가 ckpt 의 `arch.kind` 를 기본값으로, 다르면 죽음 | CLAUDE.md §3-1 · `models.md` |
| `mask_index` ≠ 0 | 학습 안 된 mask token (norm 0) 사용 | `mask_index: 0` | `ANALYSIS_NOTES.md` T6 |
| **DDP split-brain** | `init_distributed` 가 bind 실패를 삼키고 `world_size=1` 로 폴백, 두 job 이 섞임 | `run.sh` 의 포트 탐색 + `WMA_EXPECT_WS`; `evals.main` 직접 호출 금지 | CLAUDE.md §7-1 |
| exit 0 on failure | rank 가 죽어도 0 반환, `afterok` 가 풀림 | `evals/main.py` 가 exitcode 확인 | CLAUDE.md §7-1 |
| `features:` 블록 누락 | `cache_dir` KeyError 가 **surprise 채점을 다 끝낸 뒤** 터짐 | 프로토콜에 `features:` | CLAUDE.md §7-1 |
| **bf16 vs fp32** | surprise 는 fp32 + fp16 autocast, probing 은 bf16. bf16 probing 캐시로 채점을 재계산하면 근접 tie 가 뒤집혀 static_occlusion 63.37 vs 참조 58.16 | Δ 는 믿되 정식 수치는 fp32 재추출 후 | `ctxenc_subspace_v2/RESULTS.md` §1 |
| bf16 predictor 출력의 batch 의존 | batch 1 vs 4 상대 L1 0.049 | hook 재현은 같은 batch 구성으로 (상대 L1 0.0000, 800 clip) | `PREDICTOR_DISTANCE_LIMIT` 도구 검증 |
| FP16 batching tie | 근접 tie 를 뒤집을 수 있음 | 재현성이 필요하면 `video_batch: 1` | CLAUDE.md §7-2 |
| **캐시 `ctx_masked` = LN(z)** | predictor 입력으로 쓰면 캐시 p 와 상대 L1 0.60 | predictor 를 다시 돌리려면 문맥 encoder 를 다시 돌림 | CLAUDE.md §6 |
| `summary.json` 의 `overall` vs `per_group` | probing `overall` 은 조건 평균 (이식 실패 섞임) | self 는 `evals[fit]["per_group"][fit_group]` | 정본 §재현 |
| `best_avg` 집계 | Filtered + 쌍 수 합산 — IntPhys 는 macro 와 우연히 같지만 GRASP · InfLevel 은 다름 | property macro 로 다시 계산 | `PROTOCOLS.md` 단서 |
| knockout Δ = 0 | 정확도는 계단 함수 (분해능 1/(2·n_block)) — "아무 일도 없었다" 가 아님 | `pred_drift` 같이; 절단 후 전부 chance 아래라 Δ 는 조건 간 비교 불가 | `IntPhysGenV11/figures/knockout/README.md` |
| 셀 크기 | (조건, k) 로 쪼개면 비대각 방향당 n=4; v11_realistic k 칸 15 쌍; 그 칸들의 함수 (Luce 확률 등) 도 같은 잡음 | 행 평균 · 선 모양만 | CLAUDE.md §7-3 · `figures/surprise/README.md` |
| 자 기본값 | 대상이 없어도 위치를 냄 | §5-4 | §5-4 |
| AR 표적 공간 | 블록별 LN(target_encoder) — 표준 타깃과 다른 공간 | 쌍 정확도만 비교 | CLAUDE.md §3-1 |
| 노드 의존 | 데이터가 노드 로컬 (`/local_datasets`, vll5); NFS 쓰기 57 MB/s · 캐시는 `/data2` | `-w vll5`; 캐시 로컬 | CLAUDE.md §1-6 · §4-1 |

**검증 습관** (CLAUDE.md §7-4 · §12): 수치는 문서가 아니라 `summary.json` / `report.json` / `per_block.json` 에서 다시 계산하고 그렇게 했다고 적는다 (`plot_v11_*.py` 는 실행마다 자동 대조, cell 재합산 ↔ overall 0.02 %p 넘으면 죽음). 학습된 자를 쓰면 held-out · encoder 대조 · 파라미터 없는 교차검증을 같이 낸다. 모순되는 결과는 지우지 않고 조건 차이를 표로 둔다 (§6-4 처럼). 모델 로딩이 필요 없는 검증은 `DRYRUN=1` 로 먼저.

### 출처 (§8)
CLAUDE.md §1 · §3 · §4 · §6 · §7 · `cvpr/README.md` · `configs/protocols/{datasets,models}.md` · `analysis/intphys2/ANALYSIS_NOTES.md` · `z_research/context_encoder_analysis/exp_results/ctxenc_subspace_v2/RESULTS.md` · `z_research/RollOutV2/Archive/PREDICTOR_DISTANCE_LIMIT_2026-09-19.md` · `z_research/Benchmarks/PROTOCOLS.md` · `z_research/IntPhysGenV11/figures/{knockout,surprise}/README.md`

---

## 출처

레포 루트 기준. 이 장은 새 계산 없이 아래에서 옮겼다 (예외: §1 의 clip 수는 `data_csv/*/index.csv` 행 수를 2026-10-10 에 다시 셌다).

- 스토리 · 정의: `auto_research/paper/PAPER_STORY_2026-10-08.md` · `z_research/context_encoder_analysis/Archive/STATE_EVOLUTION_CAPABILITIES_2026-09-19.md` · `z_research/context_encoder_analysis/Archive/CLAIM_NO_STATE_EVOLUTION_2026-09-15.md` · `auto_research/paper/RETHINK_PREDICTOR_ROLE_2026-09-29.md`
- v11 계열: `z_research/IntPhysGenV11/README.md` · `z_research/IntPhysGenV11/Archive/surprising_score/{RESULTS_2026-08-30,HYPOTHESIS_SCALAR_ORDER_2026-08-30,VOTE_GRID_2026-08-31}.md` · `z_research/IntPhysGenV11/Archive/{SYNTHESIS_2026-08-30,TALK_2026-08-30,STATUS_AND_CRITIQUE_2026-09-03}.md` · `z_research/IntPhysGenV11/figures/{knockout,surprise}/README.md` · `z_research/IntPhysGenV11_occlusion_timing_ablation/{README,figures/README}.md` · `z_research/IntPhysGenV10/Archive/V10_2026-08-28.md`
- 옛 세대: `z_world_model_analysis/{intphys1_dev_2026-08-13,RESULTS_2026-08-16,FRAME_ABLATION_2026-08-19,MECHANISM_2026-08-20,OCCLUSION_LENGTH_2026-08-20,WHY_INTPHYS_WORKS_2026-08-20,PREDICTOR_HEDGING_2026-08-26}.md`
- 실물 메시: `z_research/v11_realistic/README.md` · `z_research/v11_realistic/exp_results/{tables,gravity_retrieval}.md`
- RollOut · 자: `z_research/RollOutV2/README.md` · `z_research/RollOutV2/Archive/PREDICTOR_DISTANCE_LIMIT_2026-09-19.md` · `z_research/RollOutV2/figures/v5/summary/POSITION_READOUT_2026-09-12.md` · `z_research/v11_roll_out/README.md` · `z_research/RollOutV3/README.md` · `z_research/RollOutV3/Archive/{IDENTITY_R8_2026-09-26,READOUT_CHOICE_2026-09-23,RESULTS_2026-09-25}.md` · `z_research/RollOutV3/_superseded/README_2026-09-25.md` · `z_research/RollOutV3/exp_results/{README.md,presence/RESULTS.md}` · `z_research/RollOutV3/exp_results/_superseded/windows_v6_20260923/{CROSS_HEAD,WINDOW_SUMMARY}.md` · `z_research/RollOutV3/audit/{training_v8,v11_presence,v11_identity}/AUDIT.md` · `z_research/RollOutV3/audit/training_sets/RollOut_v2_training_v8.md` · `z_research/scripts/_superseded/README.md`
- encoder 측정: `z_research/context_encoder_analysis/Archive/ENCODER_TEMPORAL_DYNAMICS_2026-09-20.md` · `z_research/context_encoder_analysis/exp_results/{ctxenc_boundary_kinematics,ctxenc_boundary_identity_v2,ctxenc_subspace_v2,ctxenc_time_alignment_gated}/RESULTS.md` · `auto_research/Archive/FUTURE_FROM_CONTEXT_2026-10-02.md`
- 복사 기준선 · leakage: `auto_research/Archive/{COPY_HARD_SUBSET_2026-09-29,VERIFY_R3_2026-09-29,COPY_VS_POSITION_2026-10-07,INTPHYS2_PERMANENCE_AUDIT_2026-09-26,DINOF_INTPHYS1_2026-10-02,DINOF_V11_2026-10-07,H23_PREDICTOR_LAYERS_TRAJECTORY_2026-09-24}.md` · `z_research/TrainingEffects/ip1_copy/IP1_COPY.md` · `z_research/Benchmarks/Archive/INTPHYS1_EVENT_POSITION_2026-09-25.md`
- 프로토콜 · 하네스: `configs/protocols/{README.md,surprise_c16t32.yaml,intphys1_sliding.yaml,attn_probe.yaml,attn_probe_imp.yaml,attn_probe_xfer.yaml,datasets.md,models.md}` · `configs/training/datasets.md` · `z_research/Benchmarks/{README,PROTOCOLS}.md` · `analysis/intphys2/{README,ANALYSIS_NOTES,PROTOCOL_CHECK_2026-09-14}.md` · `analysis/intphys2/configs/_superseded/README.md` · `z_research/anticipation/EK100/README.md` · `cvpr/README.md` · `z_training/{README,ARIEL_CHECKPOINTS}.md` · `z_research/predictor_training/predictor_IntPhysGenV11_PFT/Archive/RESULTS_2026-09-16.md`
