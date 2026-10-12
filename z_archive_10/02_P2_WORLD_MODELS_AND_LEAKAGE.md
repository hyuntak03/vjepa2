# 02 — P2: world model 의 발전 · 외부 벤치마크 · leakage (복사 기준선)

> **결론** — intuitive physics "창발" 의 근거로 쓰인 IntPhys 1 은 **예측 없이 마지막 문맥 latent 를 복사하기만 해도 85.00 이 나오는 (leakage 가 큰) 벤치**다. leakage 가 작은 곳 (IntPhys 2 · v11 문맥 끝 가림 · 위치만 다른 쌍) 에서 V-JEPA 2 와 DINO-Foresight 는 복사 수준이거나 우연 근처다. `PREDICTOR_SHARE` 의 요약 문장은 "어느 벤치마크에서도 predictor 몫 (predictor − 복사) 이 +2 ~ +7 pt 를 넘지 않는다 (EK100 포함)" 이다 — ⚠️ 같은 문서 표에 IntPhys 1 `skip2_w16` 표준 +9.4 · AR ep38 +9.4 가 있어 요약과 표가 어긋나고, 그 표 밖 판 (v11 late 정지 +21.6 — 복사가 chance 아래라서, DINO-F M = 1 +27) 은 더 크다 (§3-2).
> **근거** — (1) IntPhys 1 dev · ViT-H · `skip2_w32`: 88.89 vs 복사 85.00, Δ +3.9 [−0.6, +8.9] (180 쌍, CI 0 포함); DINO-Foresight 84.4 < 같은 공간 복사 88.9–92.2 (`z_research/TrainingEffects/ip1_copy/IP1_COPY.md`, `auto_research/Archive/H6_INTPHYS1_WHAT_THE_SCORE_MEASURES_2026-09-24.md`, `auto_research/Archive/DINOF_INTPHYS1_2026-10-02.md`).
> (2) IntPhys 2 Main 506 쌍: 54.35 (C24) vs 복사 55.53 (C42), 같은 C24 에서 +1.8 [−1.8, +5.5] — 구분 안 됨; 위치만 다른 쌍 (ledge / wall, 392 쌍씩) predictor 0.0 / 17.6 vs 복사 7.9 / 2.3 (`auto_research/Archive/INTPHYS2_PERMANENCE_AUDIT_2026-09-26.md`, `VERIFY_ISCENE2_2026-09-26.md` §8, `COPY_VS_POSITION_2026-10-07.md`).
> (3) predictor 몫 표: v11_split_test +2.5 [+1.6, +3.3] (색 빼면 −1.7), EK100 1 / 2 / 3 s +2.54 / +2.10 / +2.06, AR 공간 IntPhys 1 +7.2 [+2.2, +12.8]; 저자 Table 20 의 predictor 몫 +0.6 (`auto_research/Archive/PREDICTOR_SHARE_2026-09-29.md`).
> **미결** — GRASP 미실행 · InfLevel 복사 기준선 없음 (P2 그림의 빈칸) · **픽셀 생성형 world model 은 안 쟀다** · P2 문헌 (생성형 물리 벤치마크) 실물 확인.
> 이 장이 받치는 논문 문장: **P2** "world model 은 planning · prediction · anticipation 에서 빠르게 발전하지만 intuitive physics 를 따르는 미래는 예측하지 못한다" (근거 축: **IntPhys 1 은 leakage 가 있는 벤치다**, 사용자 10-08)  ·  원재료: G11 · G01 (H6) · G03 · G03b · G02b (SC3 · SC5) · G02a (VERIFY_ISCENE2) · G07 (08-20) · G10c (ARIEL_CHECKPOINTS) · G04a · G13 (OVERVIEW) 추출문 → 원 문서는 각 절 끝 `출처`.

---

## 0. 용어 — leakage 와 복사 기준선

**leakage** = 미래를 예측하지 않고 관측만으로 (마지막 관측과 실제 미래의 변화를 비교해서) 정답이 나오는 정도. **복사 기준선** 으로 잰다 (정본 `auto_research/paper/PAPER_STORY_2026-10-08.md` §2-2).
⚠️ CLAUDE.md §1-2 의 "미래일치 쌍의 누수" 와 **다른 뜻**이다. 논문에서는 용어를 한 번 정의하고 쓴다 (정본 §2-2).

| 이름 | 무엇을 미래 자리에 두나 | 표적 공간 | 쓰는 곳 | 출처 |
|---|---|---|---|---|
| **copy** (주 기준선) | LN(z)[문맥 마지막 튜블릿] (online 문맥 encoder) 을 모든 미래 튜블릿에 | 표준 = LN(target_encoder(창 전체))[미래] | IntPhys 1 · IntPhys 2 (`IP2_COPY=1`) · v11 | `H6_*` §2, `INTPHYS2_PERMANENCE_AUDIT` |
| copy_raw | 위에서 LN 을 뺀 것 | 표준 | IntPhys 1 (H6h, 09-25) — 척도 불일치라 "copy 가 약하다" 근거로 안 쓴다 | `H6_*` §3-7 |
| copy (인과 표적) | copy 와 같음 | 인과 = 튜블릿 u 까지 프레임만 본 target encoder | IntPhys 1 민감도 | `H6_*` §2, `IP1_COPY.md` 표 2b |
| `copy_blk` | 블록별 LN(target_encoder(2 장)) 의 마지막 블록 | AR 공간 (블록별) | Ariel AR 팔 IntPhys 1 | `z_training/ARIEL_CHECKPOINTS.md` §5-2 |
| DINO-F copy | 마지막 문맥 프레임 DINOv2 특징 × M | PCA-1152, LN 끔 | DINO-Foresight IntPhys 1 · v11 | `DINOF_INTPHYS1` · `DINOF_V11` |
| 문맥 평균 복사 | 문맥 8 튜블릿 평균 | 표준 | RollOut_v2 위치 쌍 (보조) | `COPY_VS_POSITION` §2 |
| `copy_last` · `mean` | 고정 probe 의 predictor 입력을 LN(encoder 마지막 토큰) / 전 토큰 평균으로 치환 | (probe 입력) | EK100 anticipation | `SC3_EK100_PREDICTOR_SUBSTITUTION_2026-09-25.md` |

⚠️ **Δ = predictor − 복사 는 가법 분해가 아니다** — "predictor 의 몫은 X pt" 를 한 숫자로 쓰지 않는다 (H6 정정). 표적 공간이 다르면 Δ 끼리만 비교한다 (공간을 가로지르는 비교는 쌍 정확도만) (`PREDICTOR_SHARE` 단서).
⚠️ 복사는 "예측 없음" 이지 "정보 없음" 이 아니다 (H6 단서 (b)). 표준 copy 는 online z 의 LN 을 EMA target 공간에 대고 잰다 — 깨끗한 copy (`copy_h_ctxonly`, target encoder 를 문맥 프레임만으로) 는 **미실행** (`H6_*` §5, `PREDICTOR_SHARE` 단서).

---

## 1. world model 이 잘하는 것 (P2 의 앞 반 — 짧게)

| 무엇 | 값 | 조건 | 등급 | 출처 |
|---|---|---|---|---|
| V-JEPA 2 EK100 action anticipation (논문 Table 5, Frozen, R@5 verb / noun / action) | ViT-L 57.8 / 53.8 / 32.7 · **ViT-H 59.2 / 54.6 / 36.5** · ViT-g384 63.6 / 57.1 / 39.7 | 논문 수치 | 문헌 ✅ | `z_research/anticipation/EK100/README.md` (논문 Table 5 전사) |
| 우리 EK100 재현 ViT-H | **56.36 / 56.16 / 35.34** | `released` 규약 (사용자 09-20 기준), val 9,296 clip 전수 | 등급 문서에 없음 (정본 재현) | 같은 문서 §A-1 |
| 공개 ViT-L probe 재평가 | 31.01 (논문 32.7) | released · frame_fixfps · 공식 기하 세트 | 등급 문서에 없음 | 같은 문서 §A-1 |
| V-JEPA 2.1 EK100 | action 40.8 (Ego4D STA 7.71) | 논문 수치, dense loss · predictor 변경 없음 | 문헌 ✅ | `auto_research/Archive/LITERATURE_STATE_WORLD_MODELS_2026-09-26.md` |
| Garrido et al. 2502.11831 "Intuitive physics understanding emerges from SSL" | IntPhys 98 / GRASP 66 / InfLevel-lab 62 | **그들 모델 V-JEPA-h + RoPE** (공개 릴리즈 아님) | 문헌 ✅ | `z_research/Benchmarks/PROTOCOLS.md` · G04a |
| 우리 IntPhys 1 dev 재현 · V-JEPA 2 ViT-H | **88.89** (160/180) | Garrido A.8 최고 `skip2_w32` · Filtered · property macro | 확정 (`check_oracle.py`) | `z_research/Benchmarks/README.md` §1 · `PROTOCOLS.md` §2–3 |
| IntPhys 2 논문 Table 2 의 V-JEPA 2-h IntPhys 1 열 | 87.22 (157/180) — 우리 88.89 와 1.7 pt | **IntPhys 2 프로토콜** (Garrido 와 자가 다름) | 문헌 | `analysis/intphys2/README.md` §4 |
| DINO-world (2507.19468) | IntPhys 91.3 / GRASP 76.0 / InfLevel 63.7; 같은 표 V-JEPA ViT-H 89.4 / 73.0 / 59.9 — **V-JEPA 1** (V-JEPA 2 아님, `DINOF_INTPHYS1_2026-10-02.md` §4) | 저자 스스로 "sanity check rather than a benchmark" | 문헌 ✅ | `auto_research/Archive/LITERATURE_STATE_WORLD_MODELS_2026-09-26.md` · `LITERATURE_POSITIONING_2026-09-25.md` |
| 생성 · 탐색형 | NWM "After 8 seconds, predictions degrade" · Genie 3 "a few minutes of continuous interaction" · V-JEPA 2-AC (새 predictor, "error accumulation") | 문헌 | 문헌 ✅ (Genie 3 는 blog) | G04a |

→ P2 의 "빠르게 발전한다" 는 **문헌으로** 선다. 우리 재현 (EK100 35.34 · IntPhys 1 88.89) 은 비교 기준이 서는지 확인하는 용도다. 아래 §3 · §6 은 그 두 점수 모두 **predictor 의 예측을 거의 요구하지 않는다** 는 것을 보인다.
⚠️ EK100 재현은 논문보다 verb −2.8 · action −1.2 (noun 은 +1.6) — head 20 sweep 이 24 GB 에 안 들어가 head 8 로 돌렸고 논문은 20 head 중 best 라 붕괴 head 가 가려진다 (`EK100/README.md` 단서 목록).

**출처**: `z_research/anticipation/EK100/README.md` · `z_research/Benchmarks/{README,PROTOCOLS}.md` · `analysis/intphys2/README.md` · `auto_research/Archive/LITERATURE_*` (G04a).

---

## 2. 외부 벤치마크 표 (프로토콜별로 나눈다)

지표는 **쌍 비교 AvgSurprise 정확도 하나** (chance 50). MaxSurprise 는 2026-09-21 철회 (단일 영상 + AUROC 용) (`Benchmarks/PROTOCOLS.md` §1-b). A.8 최고값은 같은 데이터에서 고른 **descriptive** 값이다 (held-out 아님, 논문도 그렇게 한다).

### 2-1. IntPhys 1 dev × V-JEPA 2 ViT-H — Garrido A.8 칸별 (180 쌍)

| 칸 (skip × 창) | avg | 비고 |
|---|---:|---|
| `skip2_w16` | 83.33 | Table S3 가 **그들 모델**에 고른 칸 — 09-21 이전 보고값 (폐기) |
| **`skip2_w32`** | **88.89** | A.8 탐색 최고 → 보고값 |
| `skip5_w16` | 61.11 | |
| `skip5_w32` · `skip10_*` | — | 프레임 부족으로 성립 불가 |

| 축약 (`skip2_w32`) | O1 | O2 | O3 | macro |
|---|---:|---:|---:|---:|
| **Filtered** (논문 텍스트 규칙, 시작점마다 min over C) | 85.00 | 96.67 | 85.00 | **88.89** |
| C4 | 88.33 | 96.67 | 86.67 | 90.56 |
| C8 | 86.67 | 96.67 | 83.33 | 88.89 |
| C12 | 85.00 | 96.67 | 85.00 | 88.89 |
| C16 | 78.33 | 83.33 | 83.33 | 81.67 |
| C20 | 73.33 | 70.00 | 80.00 | 74.44 |
| 노트북 규칙 (`figures.ipynb`, property 별 최고 — 참고) | 88.33 | 96.67 | 86.67 | 90.56 |

⚠️ 논문 텍스트와 공식 노트북이 어긋난다 — **텍스트를 따른다** (88.89), 노트북 값은 `alt_notebook` 으로 병기 (`PROTOCOLS.md` §2).
⚠️ 창 16 · 32 를 한 실행에 섞은 산출물은 무효 — 공식은 창마다 모델을 그 프레임 수로 짓는다 (`exp_results/_invalid_20260921_mixedgrid/README.md`).
**출처**: `z_research/Benchmarks/PROTOCOLS.md` §2–3 · `README.md` §1 · §6.

### 2-2. IntPhys 1 dev — 모델별 (같은 Garrido 프로토콜)

| 모델 | 최고 칸 | 쌍 정확도 | 등급 · 단서 | 출처 |
|---|---|---:|---|---|
| **V-JEPA 2 ViT-H** (기준 모델) | `skip2_w32` | **88.89** | 확정 | `Benchmarks/README.md` §1 |
| V-JEPA 2 ViT-L | 동일 프로토콜 | 64.44 | CLAUDE.md §5-4 (이 그룹 Benchmarks 문서 본문에는 없음, DINOF 문서가 64.4 로 같은 값 인용) | CLAUDE.md §5-4 · `DINOF_INTPHYS1` |
| V-JEPA 2.1 ViT-g | `skip2_w32` | 51.11 | ⚠️ 블록 90 중 80 이 정확히 0.5 (무작위면 45), 블록 안 σ / 사이 σ 0.04 (ViT-H 0.23) — **원인 미규명**; 공식 해상도 384 를 256 으로 잼 | `Benchmarks/README.md` §4 단서 |
| VideoMAEv2-g | `skip5_w16` | 54.44 | 픽셀 예측 공간 (1176) — surprise 절대값 비교 불가 | 같은 곳 |
| DINO-Foresight highres (448) | `skip2_w16` | 84.4 (O1 73.3 / O2 91.7 / O3 88.3) | 1 차. C=4 고정 (C 스윕 불가, Filtered 항등) · 288→448 업샘플 | `DINOF_INTPHYS1_2026-10-02.md` |
| DINO-Foresight lowres (224) | — | 75.6 | 해상도 · 위치 임베딩 보간 영향 9 pt | 같은 문서 |
| DINO-Foresight 공개값 대조 | property 별 최고 칸 | 86.1 (정방형) · 86.1 (stretch) · 86.7 (합침) vs DINO-world 표 87.8 | 1–2 pt 안 재현 (차이는 O1 3–5 pt) | 같은 문서 §4 |
| Ariel full ep40 / prefix ep45 | — | 71.67 / 70.00 | scratch · SSv2+K400 (릴리즈와 비교 대상 아님) | `z_training/ARIEL_CHECKPOINTS.md` §5-1 |
| Ariel AR ep18 / ep36·38 | `skip2_w16` | 90.56 / 92.78 | ⚠️ **AR 공간 (블록별 표적)** — 쌍 정확도만 비교, 복사 `copy_blk` 83.33 (§3) | 같은 문서 §5-1 · §5-2 |
| Ariel ctx_ar ep18 | — | 64.44 | 학습 중 판 (18/40) | 같은 문서 §5-1d |
| post-FT `ip1_e40` (IntPhys 1 로 학습) | `skip2_w32` | 93.89 | **학습 도메인 안** — 일반화 증거 아님 | `IP1_COPY.md` 표 2 · `z_training/RESULTS_2026-09-19.md` |

⚠️ 세 모델 (ViT-H · 2.1-g · VideoMAEv2-g) 의 surprise **절대값은 비교 불가** (latent 1280 / 5632 / 픽셀 1176) — 비교 가능한 것은 쌍 판별 정확도뿐 (`Benchmarks/README.md` §4).
⚠️ 180 쌍 SE ≈ 3 pt — 릴리즈 88.89 vs AR 90.56 은 잡음 안 (`ARIEL_CHECKPOINTS.md` 단서).

### 2-3. IntPhys 1 dev — 운동 × 가림 4 칸 (V-JEPA 2 ViT-H, 88.89 칸)

| | 보임 | 가림 |
|---|---:|---:|
| 정지 (30 쌍씩) | 100 | 100 |
| 이동 (60 쌍씩) | 91.7 | **75.0** |

세 칸이 천장이고 **이동 + 가림 한 칸만 떨어진다** — 실제 벤치마크에서 P3 ("움직이는 물체가 마지막 관측에서 가려지면") 의 다리 (CLAUDE.md §5-4). 사건이 예측 구간 안인 창만 쓰면 이동+가림 75.0 → 80.0 (`INTPHYS1_EVENT_POSITION_2026-09-25.md` §2).
⚠️ 이 칸에서도 표준 표적 p − 복사 +5.0 [−5.0, +15.0] (n.s.); 인과 표적에서만 +20.0 [+6.7, +35.0]; 재등장 튜블릿 단위 +16.4 [+12.5, +20.5] (`H6_*` §3-2 · §3-4).
**출처**: `z_research/OcclusionStats/Archive/INTPHYS1_OCCLUSION_2026-09-26.md` §2-1 · CLAUDE.md §5-4 · `H6_*` §3-2.

### 2-4. InfLevel-lab — 3 모델 × 3 property (Garrido 프로토콜, property 별 최고 C · macro)

| property | n_pair | V-JEPA 2 ViT-H | V-JEPA 2.1 ViT-g | VideoMAEv2-g |
|---|---:|---:|---:|---:|
| **macro** (`skip10_w16` / 2.1-g `skip5_w32`) | 2,748 | 58.27 | **60.66** | 54.18 |
| ┗ **continuity** (유일하게 믿을 수 있는 property) | 1,116 | **72.85** | **77.60** | 61.83 |
| ┗ solidity ¹ | 450 | 50.00 | 53.78 | 50.89 |
| ┗ gravity ¹ | 1,182 | 51.95 | 50.59 | 49.83 |

¹ **원리적으로 못 푸는 property** — 컵 상태가 공식 로더가 **잘라내는** priming 구간에만 나온다 (Garrido App. E). 50 근처가 정상이고 모델 비교에 쓰지 않는다.
⚠️ **문서 간 상태 차이** — `z_research/Benchmarks/README.md` §4 표는 InfLevel 을 "미완" (continuity ViT-H 72.85 만) 으로 적고 있고, `z_research/OVERVIEW_2026-09-22.md` §5-8 · §9-2 가 `garrido_rescore.py` 재실행으로 **3 × 3 전부 나온다** 고 정정했다 (README 표는 스크립트가 박는 표라 갱신이 안 된 것). 위 표는 OVERVIEW 값 ✓ (OVERVIEW 가 재실행 표시).
⚠️ 09-21 이전 InfLevel 수치는 전부 폐기 (자작 쌍 · cross 2×2 · priming 미절단, `_invalid_20260921_inflevel/`).
⚠️ **InfLevel 복사 기준선은 없다** — P2 그림의 빈칸 (§7-2).
**출처**: `z_research/OVERVIEW_2026-09-22.md` §5-8 · §9-2 · `z_research/Benchmarks/README.md` §4 · §6 · `PROTOCOLS.md` §5-3.

### 2-5. GRASP level2 — ★ 미실행

`intphys1_sliding__grasp_level2_vith_w16/` 에 `_resolved.yaml` 하나뿐 (OVERVIEW §9-2). 데이터는 16 시나리오 × P/IP × 128 = 4,096 영상 · 2,048 쌍 (`configs/protocols/datasets.md`).
⚠️ GRASP 은 **단일 영상 평가용** 벤치이고 쌍짓기는 Garrido A.4 해석이다 — "untrained network 가 spurious feature 로 높은 정확도" 단서를 같이 적는다 (`configs/protocols/datasets.md` `grasp_level2`).
대조값 (문헌): Garrido 그들 모델 GRASP 66 · DINO-world 76.0 · V-JEPA (v1) ViT-H 73.0 (DINO-world 표).

### 2-6. IntPhys 2 Main — **다른 프로토콜** (Bordes et al., 범위 밖 하네스 `analysis/intphys2`)

506 쌍 (`type` 번호 = 쌍, 2×2 cross 금지), 6 fps, 창 16 / 32 / 48, C = 창 × {¼ … ⅞}, growing prefix (`max_context_mode`), 열마다 최고 실행 (논문 Table 2 캡션) (`analysis/intphys2/README.md`).

**모델 × 창 — overall (C 별 최고, descriptive)**

| 모델 | w16 | w32 | w48 | 출처 |
|---|---:|---:|---:|---|
| **V-JEPA 2 ViT-H 릴리즈** | 53.16 (C10) | 52.77 (C24) | **54.35 (C24)** | `Benchmarks/exp_results/intphys2/bench_intphys2_main_vith{_w16,_w32,}/column_best.md` |
| Ariel `bc_ep30` | 53.16 (C10) | 53.75 (C8) | 52.96 (C42) | `…/bench_intphys2_main_ariel_bc_ep30_w{16,32,48}/column_best.md` |
| Ariel full ep40 / prefix ep45 (18 칸 중 각 팔 최고) | — | full **53.75 (C8)** | prefix **53.56 (C36)** | `z_training/ARIEL_CHECKPOINTS.md` §5-3 (full 최고는 w32/C8, block-causal 최고는 w48/C36) |
| Ariel AR ep18 / ep38 | ep18 **51.19 (C8)** · ep38 50.79 (C12) | — | — | 같은 문서 §5-3 · §5-1c (둘 다 최고 칸이 w16) |
| **복사 (copy, LN)** | — | 54.2 (C28) | **55.53 (C42)** | `INTPHYS2_PERMANENCE_AUDIT` §3 · CLAUDE.md §5-6 |
| 논문 V-JEPA 2-h / V-JEPA-h + RoPE / VideoMAEv2-g | — | — | **57.51** / 53.75 / 53.75 (Table 2) | `analysis/intphys2/README.md` §4 |

⚠️ `bc_ep30` 이 어느 팔인지는 `column_best.md` 에 적혀 있지 않다. 폴더 이름 `block_causal_future_only_1e_5` (prefix 팔) 의 옛 epoch 로 **보인다** (옛 레지스트리 `vith_ariel_bc_ep*`, `ARIEL_CHECKPOINTS.md` §1) — 확인 안 됨.
⚠️ V-JEPA 2.1-g · VideoMAEv2-g 의 IntPhys 2 산출물 폴더 (`bench_intphys2_main_{vjepa21g,videomae2g}*`) 는 있지만 **결과를 적은 md 문서가 없다** — 이 아카이브는 수치를 옮기지 않는다.
⚠️ AR 은 w32 · w48 에서 rollout 15 · 23 블록 (학습 최대 8) — 분포 밖 (`ARIEL_CHECKPOINTS.md` 단서).

**ViT-H w48 열별 — 우리 vs 논문** (`bench_intphys2_main_vith/column_best.md`)

| 열 | 짝 | 열별 최고 (C) | overall 최고 설정 C24 로 읽은 값 | 논문 V-JEPA 2-h |
|---|---:|---:|---:|---:|
| Easy | 52 | 57.69 (C30) | 51.92 | 54.00 |
| Medium | 200 | 54.50 (C24) | 54.50 | 58.50 |
| Hard | 168 | 54.76 (C42) | 53.57 | 59.38 |
| Unknown | 86 | 61.63 (C36) | 56.98 | — |
| **Overall** | 506 | **54.35 (C24)** | **54.35** | **57.51** |
| permanence · Fixed / Moving | 52 / 68 | 59.62 / 51.47 | 59.62 / 45.59 | 59.62 / 57.35 |
| immutability · Fixed / Moving | 52 / 68 | 61.54 / 50.00 | 57.69 / 45.59 | 55.77 / 58.72 |
| continuity · Fixed / Moving | 52 / 68 | 55.77 / **69.12** | 51.92 / 69.12 | 57.69 / 75.00 |
| solidity · Fixed / Moving | 52 / 94 | 50.00 / 55.32 | 50.00 / 55.32 | 46.15 / 58.51 |

C 별 overall (w48): C4 48.81 · C6 49.21 · C8 49.60 · C10 48.81 · C12 51.19 · C14 50.59 · C18 51.78 · **C24 54.35** · C30 54.15 · C36 53.75 · C42 53.75.
→ continuity · Moving 하나만 우연을 분명히 넘는다. 우리 54.35 는 논문 57.51 보다 3.2 낮다 (논문은 열별 최고 run, 12 회 중 최고 — `INTPHYS2_PERMANENCE_AUDIT` 단서).
⚠️ IntPhys 2 하네스 config 에는 `PROTOCOL_CHECK_2026-09-14` 의 정밀도 수정 (`dtype: float32` + `autocast: bfloat16`) 이 **미반영** (CLAUDE.md §2-7) — 감사 문서는 이를 언급하지 않는다.

### 2-7. 프로토콜이 다른 IntPhys 1 수치 — **위 표와 섞지 않는다** ("미규명" 포함)

| 값 | 프로토콜 · 설정 | 상태 | 출처 |
|---|---|---|---|
| 67.22 (w16) / 66.11 (w32) | IntPhys 1 영상에 **IntPhys 2 프로토콜** (우리 구현) | 논문 87.22 와 20 pt — **원인 미규명**. IntPhys 1 은 Garrido 로만 보고 | `analysis/intphys2/configs/_superseded/README.md` · `PROTOCOLS.md` §7 |
| 68.89 (C=10; O1 61.67 / O2 68.33 / O3 76.67) | IntPhys 2 식 하네스, fps 15 · M 48 · single-EMA (07-12 감사 뒤) | 기록 (라이브 노트) | `analysis/intphys2/ANALYSIS_NOTES.md` §9 · §13 |
| 67.22 | 위 + `dual_encoder` | 07-12 "적용하지 말 것" 권고 — **현행 프로토콜 (`dual_encoder: true`) 과 반대**, 번복 근거 문서 없음 | 같은 문서 T9 |
| 46.4 → 58.89 (ViT-L) · 60.0 → 68.89 (ViT-H) | pre-fix → post-fix (`mask_index=0` · masked context · matched pair · fps 15) | 기록 — `mask_index: 0` critical 의 발견 기록 | 같은 문서 T6 |
| 66.67 | 우리 fixed C16 / stride 3 | 기록 (CLAUDE.md §1-3 "22.2 pt 흔들림") | CLAUDE.md §5-4 |
| 87.78 · 88.3 · 88.33 | 최고 단창 `skip2_w32` start=24 C=8 (08-20) | 문서 간 반올림 차이 미설명; sliding 88.89 와 구분 | `z_world_model_analysis/WHY_INTPHYS_WORKS_2026-08-20.md` · `VERIFICATION_2026-08-20.md` · `MECHANISM_2026-08-20.md` |
| 87.22 | IntPhys 2 논문 Table 2 (그들 IntPhys 2 프로토콜) | 문헌 | `analysis/intphys2/README.md` §4 |

**출처 (§2 전체)**: `z_research/Benchmarks/{README,PROTOCOLS}.md` · `z_research/Benchmarks/exp_results/intphys2/*/column_best.md` · `z_research/OVERVIEW_2026-09-22.md` §5-8 · §9-2 · `analysis/intphys2/{README,ANALYSIS_NOTES,PROTOCOL_CHECK_2026-09-14}.md` · `analysis/intphys2/configs/_superseded/README.md` · `z_training/ARIEL_CHECKPOINTS.md` · `auto_research/Archive/DINOF_INTPHYS1_2026-10-02.md` · `configs/protocols/datasets.md` · CLAUDE.md §5-4.

---

## 3. leakage 표 — 복사 기준선 (정본 §2-2 (b) 확장)

### 3-1. 주 표 — 벤치마크 · 칸 · 모델마다 predictor vs 복사

| # | 벤치마크 · 칸 | 모델 | predictor | 복사 (= leakage) | Δ [95 % CI] | 등급 | 출처 |
|---|---|---|---:|---:|---|---|---|
| 1 | **IntPhys 1 dev · `skip2_w32` · 표준 표적 (공식 전 창, V0)** | V-JEPA 2 ViT-H | **88.89** | **85.00** [79.4, 90.0] | **+3.9 [−0.6, +8.9]** (12 승 / 5 패) | **확정** (정본 ✓ 10-08 재확인) | `z_research/TrainingEffects/ip1_copy/IP1_COPY.md` · `H6_*` §3-1 |
| 2 | IntPhys 1 · `skip2_w32` · **사건이 예측 구간 안인 창만 (V1)** | ViT-H | 91.67 | 87.22 (copy_raw 88.89) | +4.4 [+0.6, +8.3] | 등급 문서에 없음 (분석용 변형 — 벤치마크 점수 아님) | `z_research/Benchmarks/Archive/INTPHYS1_EVENT_POSITION_2026-09-25.md` §2 |
| 3 | IntPhys 1 · `skip2_w16` · 표준 | ViT-H | 83.33 | 73.89 | +9.4 [+3.3, +15.6] | 1 차 | `IP1_COPY.md` 표 2 · `PREDICTOR_SHARE` |
| 4 | IntPhys 1 · `skip2_w32` · **인과 표적** | ViT-H | 84.44 | 77.78 | +6.7 [+0.6, +13.3] | 1 차 | `IP1_COPY.md` 표 2b · `PREDICTOR_SHARE` |
| 5 | IntPhys 1 · `skip5_w16` · 표준 | ViT-H | 59.44 | 63.33 | −3.9 [−9.4, +1.7] | 1 차 | `IP1_COPY.md` 표 2 |
| 6 | IntPhys 1 · `skip2_w32` · copy_raw (LN 제거) | ViT-H | 88.89 | 77.78 [71.1, 83.9] | +11.1 [+5.6, +17.2] | 기록 — **"copy 가 약하다" 근거로 쓰지 않는다** (척도 불일치) | `H6_*` §3-7 (H6h) |
| 7 | IntPhys 1 · `skip2_w16` · **AR 공간** | Ariel AR ep18 | 90.56 | `copy_blk` 83.33 | **+7.2 [+2.2, +12.8]** (19 / 6) | 1 차 | `PREDICTOR_SHARE` · `ARIEL_CHECKPOINTS.md` §5-2 |
| 7b | 〃 `skip2_w32` / AR ep38 `skip2_w16` | AR ep18 / ep38 | 87.78 / 92.78 | 80.56 / 83.33 | +7.2 [+1.7, +12.8] / +9.4 (CI 없음) | 1 차 | 같은 곳 |
| 8 | IntPhys 1 · `skip2_w32` · 표준 | Ariel prefix ep43 | 68.89 | 85.00 | **−16.1 [−23.3, −8.9]** | 1 차 | `IP1_COPY.md` 표 2 |
| 9 | **IntPhys 1 · DINO-F highres** | DINO-Foresight | 84.4 (`skip2_w16`) | 88.9 (w16) · 92.2 (skip5 w16) · w8 84.4 / 87.2 | 복사가 4–8 pt 위 | **1 차** | `auto_research/Archive/DINOF_INTPHYS1_2026-10-02.md` |
| 9b | 〃 M = 1 (w5) | DINO-F | 81.1 | 54.4 / 52.2 / 56.1 (chance) | predictor 가 27 pt 압도 | 1 차 | 같은 문서 |
| 10 | **IntPhys 2 Main · w48 · 같은 C24** | ViT-H | 54.35 | 52.6 | **+1.8 [−1.8, +5.5]**; permanence all +2.5 [−4.2, +10.0] | **확정** (`VERIFY_ISCENE2` §8) | `INTPHYS2_PERMANENCE_AUDIT_2026-09-26.md` · `VERIFY_ISCENE2_2026-09-26.md` · `COPY_HARD_SUBSET` §2 |
| 10b | IntPhys 2 · w48 · 각자 최고 C (24 / 42) | ViT-H | 54.3 | **55.5** | −1.2 [−5.1, +3.0] | 확정 (구분 안 됨) | 같은 곳 |
| 10c | IntPhys 2 · w32 · 각자 최고 C | ViT-H | 52.8 | 54.2 | −1.4 [−5.3, +2.6]; 같은 C +0.4 | 확정 | `PREDICTOR_SHARE` · `VERIFY_ISCENE2` |
| 11 | v11_split_test · `surprise_c16t32` | ViT-H | 75.70 | 73.24 | +2.5 [+1.6, +3.3] — 색 +9.3 / 모양 −1.6 / vanish 물체→빈 −6.8, **색 빼면 −1.7 [−2.6, −0.7]** | 1 차 | `PREDICTOR_SHARE` · `z_research/TrainingEffects/scores/ANALYSIS.md` 한 줄 1 · §1 (⚠️ `PREDICTOR_SHARE` 는 출처를 `SCORES.md` 로 적었으나 색 분해는 `ANALYSIS.md` 에 있다) |
| 11b | 〃 | Ariel prefix ep43 | 79.33 | 73.24 | +6.1 [+5.2, +6.9] | 1 차 | 같은 곳 |
| 12 | v11 전체 (문맥 공유 — 참고) | ViT-H | 73.37 | 73.2 | ≈ +0.2 (CI 없음) | 참고 | `PREDICTOR_SHARE` |
| 13 | v11 block 1/4 표본 (5,376 쌍) | ViT-H | 76.4 | 72.4 | +4.0 [+2.7, +5.2]; **이동 8 칸 합침 +1.0 [−0.5, +2.5]** | 정본 (H6 정정판) | `H6_*` §3-6 |
| 14 | **v11 · 문맥 끝 가림 (late) 칸** (3 운동 × 위반) | ViT-H | 50–70 (ramp vanish 50 = A 0 / B 100) | 39–63 | 예측이 필요한 칸에서 우연 근처 | 등급 문서에 없음 (RETHINK 는 대체된 문서, §5-b 는 정본이 유효로 남김) | `auto_research/paper/RETHINK_PREDICTOR_ROLE_2026-09-29.md` §5-b |
| 14b | v11 late · vanish 물체→빈 (112 쌍씩, static / flat / ramp) | ViT-H | 65.2 / **2.7 / 0.0** | 67.9 / 39.3 / 9.8 | 이동 두 칸은 복사보다 아래 | 1 차 | `COPY_HARD_SUBSET_2026-09-29.md` §2-3 |
| 14c | v11 late 정지 | ViT-H | 57.6 | 35.9 (복사 − chance −14.1 [−18.1, −10.0]) | +21.6 — 복사가 chance 아래라서 ("문맥이 가림막에서 끝나면 복사는 역정보") | 정본 (H6 정정판) | `H6_*` §3-6 |
| 15 | **v11 · DINO-F** (10,752 쌍) | DINO-Foresight | 77.10 | 76.44 | +0.7; 가림 세 조건 shape · color 51.6–64.9, vanish A/B 0/100 (ViT-H 와 같은 실패) | 1 차 (✓ 10-08 재확인) | `auto_research/Archive/DINOF_V11_2026-10-07.md` |
| 16 | **RollOut_v2 위치만 다른 쌍** — ledge 낙하 vs 부유 (392) | ViT-H | **0.0** | 7.9 (문맥 평균 1.5) | predictor 는 392 쌍 전부 부유 (불가능) 쪽 | 1 차 (✓ json) | `auto_research/Archive/COPY_VS_POSITION_2026-10-07.md` §2 |
| 16b | 〃 wall 정지 vs 통과 (392) | ViT-H | **17.6** | 2.3 (문맥 평균 66.3) | predictor 도 복사도 chance 아래 | 1 차 | 같은 곳 |
| 17 | EK100 anticipation 1 s · released · action R@5 | ViT-H | 35.34 | `copy_last` 32.80 (`mean` 34.01) | +2.54 (CI 없음) | 1 차 (SC5 §3 은 `VERIFY_ISCENE2` §7 확정) | `SC3_*` · `PREDICTOR_SHARE` — §6 |

→ 쓸 문장 (정본 §2-2): **"intuitive physics 창발의 근거인 IntPhys 1 은 예측 없이 85 % 가 풀리는 (leakage 가 큰) 벤치이고, leakage 가 작은 곳 (IntPhys 2 · 문맥 끝 가림 · 위치 위반) 에서 world model 은 우연 근처다."**
⚠️ 행 1 의 Δ 는 **칸 · 표적에 의존한다** — `skip2_w16` +9.4 (유의), 인과 표적 +6.7 (유의), `skip5_w16` −3.9, V1 (깨끗한 창) +4.4 (CI 하한 +0.6). "양날, 프로토콜 민감도로만" 쓴다 (G04a (d)). 행 1 만 골라 "Δ 가 0" 으로 쓰지 않는다.
⚠️ 행 16 의 0 % 는 **"metric 한계 (토큰 평균 L1 은 위치를 거의 못 본다) + predictor 의 틀린 미래"** 가 섞인 값 — 위치 주장은 위치 자 판독 (RollOutV2 §5-5, 06 장) 과 같이 쓴다 (정본 §2-2 ⚠️).
⚠️ "leakage 가 작아서 무너진다" 는 **상관 서술**이다 — IntPhys 2 의 낮은 점수에는 장면 복잡도 등이 섞일 수 있다. 인과로 쓰려면 같은 장면에서 leakage 만 바꾼 쌍 (v11 의 가림 위치 축: visible · early · mid · late) 을 같이 보인다 (정본 §2-2 ⚠️).
⚠️ best-C · A.8 최고 칸은 descriptive. IntPhys 2 는 release 와 copy 의 최고 C 가 달라 (24 vs 42) Δ 부호가 뒤집힌다 → **같은 C 판을 주 값으로** (`PREDICTOR_SHARE` 단서).
⚠️ 이 표는 "predictor 가 쓸모없다" 가 아니라 **"이 벤치마크들이 predictor 의 예측력을 요구하지 않는다"** 의 근거다 (H6 정정 8; `DINOF_INTPHYS1` — M = 1 에서는 predictor 가 27 pt 압도).

### 3-2. predictor 몫 한 줄 요약 (`PREDICTOR_SHARE_2026-09-29.md`, 1 차 · 표 자체는 적대 검증 전)

| 판 | Δ = predictor − 복사 | CI 가 0 을 제외하나 |
|---|---|---|
| IntPhys 1 표준 `skip2_w32` | +3.9 | ❌ |
| IntPhys 1 표준 `skip2_w16` / 인과 `skip2_w32` / AR 공간 | +9.4 / +6.7 / +7.2 | ✅ / ✅ / ✅ |
| IntPhys 2 같은 C24 / best-C | +1.8 / −1.2 | ❌ / ❌ |
| v11_split_test 릴리즈 | +2.5 (색 하나에서 나옴) | ✅ |
| EK100 released 1 / 2 / 3 s · paper 1 / 2 s | +2.54 / +2.10 / +2.06 · +2.61 / +2.75 (같은 head 고정 paper 1 s +4.67) | CI 없음 (clip 예측 미저장) |
| V-JEPA 2 논문 Table 20 (저자, EK100 g384) | encoder 만 39.1 → + predictor 39.7 = **+0.6** | 정의가 다름 ("predictor 제거 대비") — 같이 인용하되 섞지 않는다 |

→ (원 문서 요약 문장) **어느 벤치마크에서도 predictor 몫이 +2 ~ +7 pt 를 넘지 않는다.** ⚠️ 같은 표의 `skip2_w16` 표준 +9.4 · AR ep38 +9.4 는 이 범위를 넘는다 — 원 문서 안에서 요약과 표가 어긋난다 (고치지 않고 둘 다 적는다). 새 predictor 의 downstream 평가는 이 Δ 로 보고한다 (`PREDICTOR_SHARE` 읽기 · `RETHINK_PREDICTOR_ROLE_2026-09-29.md` §5).
⚠️ AR 몫 (+7.2) 과 릴리즈 몫 (+3.9) 의 차 3 pt 는 180 쌍 잡음 안 — 그 차로 AR 을 낫다고 쓰지 않는다. AR 공간 복사가 없는 판 (IntPhys 2 · v11) 은 사용자 지시로 보류 (`ARIEL_CHECKPOINTS.md` §5-2b).

### 3-3. IntPhys 1 — 사건 위치로 창 나누기 (88.89 는 부풀려진 값인가)

`skip2_w32` 의 180 × 45 창을 사건 시점 `d` (pos/imp PNG 첫 차이 raw 프레임) 로 분류 (`INTPHYS1_EVENT_POSITION_2026-09-25.md` §0 · §2, scene bootstrap B = 10,000).

| 칸 | V0 공식 전 창 | **V1 사건이 예측 구간 안** | V3 사건이 context 안 |
|---|---:|---:|---:|
| `skip2_w32` | 88.89 [84.4, 92.8] | **91.67 [87.8, 95.0]** | 58.83 [50.2, 67.3] (n 162) |
| `skip2_w16` | 83.33 [77.8, 88.9] | 90.56 [86.1, 94.4] | 73.96 [67.6, 79.9] (n 174) |
| `skip5_w16` | 61.11 | 61.76 (n 178) | 55.28 (n 132) |

| 창 종류 (`skip2_w32`) | 창 비율 | 창 정답 |
|---|---:|---:|
| future (사건이 예측 구간 안) | 62.5 % | 86.9 |
| context (사건이 문맥 안) | 27.7 % | **60.3** |
| after / before | 6.8 % / 3.1 % | 50.0 (전부 동점) |

- Filtered (시작점마다 min over C) 가 고른 창: imp 쪽 context 43.8 % · future 46.4 %, pos 쪽 context 37.5 % · future 52.6 % — **시작점의 약 40 % 에서 위반이 이미 문맥 안인 창을 고른다.**
- 결론 (문서 문장): **88.89 는 모델이 위반을 미리 봐서 부풀려진 값이 아니라 그런 창이 깎아 먹은 값이다.** 깨끗한 창 (V1) 에서도 복사가 함께 올라 p − copy 는 +4.4 (CI 하한 +0.6) — "IntPhys 1 점수의 상당 부분은 복사로도 나온다" 는 V1 에서도 그대로.
⚠️ V1 은 분석용 변형 (`d` 는 matched pair 에서만 안다). 표적은 양방향 target encoder 라 future 창 안에서도 사건 전 튜블릿에 사건 뒤 정보가 번진다 (H6 §3-4). §5 가림 통계는 대체됨 → `OcclusionStats` (§4).

### 3-4. copy-hard 부분집합 — 복사가 틀리는 쌍에서 predictor 는?

사용자 지시 (09-29): "late 가림은 복사로 안 나온다 — 이게 오히려 좋은 거지". 쌍을 복사가 틀리는 쌍 (copy-hard) / 맞는 쌍으로 나눠 predictor 쌍 정확도 (`COPY_HARD_SUBSET_2026-09-29.md` §2–4, 원배열 재계산, cluster bootstrap B 2,000).

| 판 | 복사 전체 | copy-hard n | 릴리즈 copy-hard / copy-correct | prefix · full | AR ep38 | 도메인 안 post-FT |
|---|---:|---:|---|---|---|---|
| v11_split_test | 73.2 | 2,877 | **48.4 [46.6, 50.2]** / 85.7 | 51.0 · 49.6 | 73.8 [72.2, 75.4] | v11_e10 82.5 |
| IntPhys 1 `skip2_w32` | 85.0 | 27 | **44.4 [25.9, 63.4]** / 96.7 | — | 70.4 | — |
| IntPhys 1 `skip5_w16` | — | — | 16.7 | — | 68.2 | — |
| IntPhys 2 w48 C24 | 52.6 | 240 | **21.2 [16.0, 26.6]** / 84.2 | prefix 19.2 | 40.4 | — |

- v11 copy-hard 구성: color 1,571 · shape 1,023 · vanish→빈 207 · vanish→물체 76; late 1,272 · mid 626 · early 559 · visible 420. copy-hard 안 late 에서 릴리즈 38.3 · prefix 37.8 · AR 65.8.
- 결론 (문서 문장): 릴리즈 점수는 **"복사 + 나머지에서 동전 던지기"** 로 분해된다; 미래 전용 재학습 (prefix · full · ip2) 이 올린 +3 ~ +4 는 copy-correct 위에서만 난다. 새 predictor 의 1 차 downstream 지표는 **copy-hard 정확도** 여야 한다.
⚠️ **정정 (`VERIFY_R3_2026-09-29.md` C3, 3 차 적대 검증)** — 무작위 겹침 귀무에서 copy-hard 정확도의 기대값은 50 이 아니라 **그 모델의 전체 정확도**다. 모든 predictor 가 그 아래 (릴리즈 −27 · prefix −28 · full −28 · **AR −10** · v11_e10 −9). 그래서 AR 73.8 은 "복사가 못 푸는 것을 푼다" 가 **아니라** "AR 의 오류가 복사 오류와 덜 묶여 있다" 로 쓴다.
⚠️ AR 표적 공간이 블록별이라 표준 복사와 덜 상관될 뿐일 수 있다 (IntPhys 1 copy_blk-hard 에서 AR 57–63 vs 릴리즈 77 로 대칭). IntPhys 1 copy-hard n 27–66 (CI ±15–20), 세분 n 2–19 은 읽지 않는다. 쌍 단위 "우연 = 50" 은 복사와 독립일 때만의 기대 — 양의 상관이면 50 아래가 자연스럽다.

### 3-5. 같은 수치의 문서 간 차이 (지우지 않고 조건을 적는다)

| 값 | 문서 A | 문서 B | 차이의 원인 (문서가 밝힌 것) |
|---|---|---|---|
| IntPhys 1 `skip2_w32` Δ 의 CI | `IP1_COPY.md` · 정본 §2-2: +3.9 [−0.6, **+8.9**] | `H6_*` §3-1: [−0.6, **+8.4**] · `PREDICTOR_SHARE`: [−0.6, **+8.3**] | 다른 실행 (H6g, tie 0.5) · 다른 bootstrap 난수 — 끝값만 다름 (`IP1_COPY.md` 대조표) |
| IntPhys 1 `skip2_w16` 표준 · 인과 `skip2_w32` Δ 의 CI | `IP1_COPY.md`: +9.4 [**+3.3**, +15.6] · +6.7 [+0.6, **+13.3**] | `PREDICTOR_SHARE`: +9.4 [**+3.9**, +15.6] · +6.7 [+0.6, **+12.8**] | bootstrap 난수 · 실행 (끝값만 다름; 이 장 표는 `IP1_COPY` 값) |
| IntPhys 1 `skip2_w16` 표준 | `IP1_COPY.md` / `PREDICTOR_SHARE`: p 83.33 · 복사 73.89 · **+9.4** | `H6_*` (h6g): p 83.89 · 복사 71.67 · **+12.2** | 같은 이유 (tie 처리 · 실행). 공식 하네스 값은 83.33 |
| IntPhys 1 인과 표적 `skip2_w32` 복사 | 77.78 (+6.7) | H6g 77.22 (+7.2) | 같은 이유 |
| 복사 85.00 의 CI [79.4, 90.0] | 정본 §재현 ✓ (`ip1_copy.json` · `h6g_intphys1.json` 두 파생 json 일치) | `H6_*` 본문에는 복사 − chance +35.0 [+30.0, +40.0] 만 있음 | 원 배열 (`/data2/.../ip1score_final/arrays_v360.npz`) 은 vll3 에 없다 (정본 §재현) |
| IntPhys 2 best-C Δ 의 CI | `INTPHYS2_PERMANENCE_AUDIT`: −1.2 [**−5.5**, +3.0] | `PREDICTOR_SHARE` · `VERIFY_ISCENE2`: [**−5.1**, +3.0] | bootstrap 난수 — 같은 값 |
| IntPhys 2 "복사가 −5 pt 앞선다" | `INTPHYS2_PERMANENCE_AUDIT` 원문 | 2 차 정정: best-C 산물 → **구분 안 됨** | `VERIFY_ISCENE2` §8 |
| IntPhys 1 Ariel 의 p − 복사 | prefix ep43 −16.1 (`IP1_COPY` · `TRAINED_PREDICTORS_SIGNATURES` §3-6) | AR ep18 +7.2 (`PREDICTOR_SHARE`) | 모순 아님 — 다른 팔 · 다른 표적 공간 (G03 (c)) |

### 3-6. 방향 편향 — 점수 하나가 숨기는 것 (CLAUDE.md §1-7 "A/B 방향 병기")

| 판 | 나타남 (빈→물체) | 사라짐 (물체→빈) | 출처 |
|---|---|---|---|
| IntPhys 1 `skip2_w32` · 모든 방법 (predictor 여섯 + copy, O1 · O3 쌍) | 92–100 % | **36–64 %** | `IP1_COPY.md` 요약 (표 2c) |
| IntPhys 1 사라짐 25 쌍 (O1 + O3) | — | p **56.0 = 복사 56.0** (p − chance +6.0 [−14, +26]) | `H6_*` §3-2 |
| v11 late vanish (H6 표본) | B: p 92.3 / 복사 76.2 | A: p **22.0** / 복사 41.7 | `H6_*` §3-6 |
| v11 vanish 가림 ramp (전수) | 100 | **0** | CLAUDE.md §5-1 |
| v11 DINO-F vanish 가림 이동 두 조건 | 100 | **0** (복사도 0/100) | `DINOF_V11` |

→ "사라지지 않음" 방향은 IntPhys 1 에서도 predictor 와 복사가 **같이 우연 근처**다. leakage 가 큰 벤치에서도 P1 의 첫 예시 (사라지지 않음) 는 풀리지 않는다.
⚠️ IntPhys 1 사라짐 25 쌍은 CI 폭 40 pt — 방향 결론은 v11 (전수) 와 같이 쓴다.

**출처 (§3 전체)**: `z_research/TrainingEffects/ip1_copy/IP1_COPY.md` · `auto_research/Archive/{H6_INTPHYS1_WHAT_THE_SCORE_MEASURES_2026-09-24, PREDICTOR_SHARE_2026-09-29, COPY_HARD_SUBSET_2026-09-29, VERIFY_R3_2026-09-29, INTPHYS2_PERMANENCE_AUDIT_2026-09-26, VERIFY_ISCENE2_2026-09-26, DINOF_INTPHYS1_2026-10-02, DINOF_V11_2026-10-07, COPY_VS_POSITION_2026-10-07, SC3_EK100_PREDICTOR_SUBSTITUTION_2026-09-25}.md` · `auto_research/paper/RETHINK_PREDICTOR_ROLE_2026-09-29.md` §5 · §5-b · `z_research/Benchmarks/Archive/INTPHYS1_EVENT_POSITION_2026-09-25.md` · `z_research/TrainingEffects/scores/ANALYSIS.md` · `z_training/ARIEL_CHECKPOINTS.md` · 정본 §2-2 · §재현.

---

## 4. 왜 IntPhys 1 은 새나 — 기제 후보 (측정된 것만)

| # | 기제 | 수치 | 등급 · 날짜 | 출처 |
|---|---|---|---|---|
| 1 | **위반이 외형 사건뿐이다** (사라짐 · 모양 · 연속성). 궤적 · 속도 위반은 dev 에 **0 건** | "teleport" 로 보이던 것은 10 프레임 소멸 뒤 **옳은 자리** 재출현 (`O1_02_3`, magic 46 → 56) — 오분류 정정 | 기록 (08-20, 정정 포함) · CLAUDE.md §6 기각표 ("궤적·속도 위반 추가") | `z_world_model_analysis/VERIFICATION_2026-08-20.md` §2.2 |
| 2 | **모든 위반이 magic 프레임에 즉시 흔적을 남긴다** | magic 이전 평균 화소차 180/180 쌍 정확히 0.0000, `delay = 0` 180/180; 가림막 뒤 사건도 전역 조명 · 그림자가 즉시 바뀜 (`O2_05_1` frame 33 0.3378) | 기록 (08-20) | 같은 문서 §2.3 |
| 3 | **토큰 평균 L1 은 "무엇이 / 있는지" 를 보고 "어디" 는 거의 못 본다** | 외형 위반은 복사로 풀리고 (위치 오차가 matched 쌍에서 상쇄) 위치만 다른 쌍은 predictor 0.0 / 17.6 · 복사 7.9 / 2.3 (§3-1 행 16) | 정본 (10-07) | `auto_research/Archive/COPY_VS_POSITION_2026-10-07.md` §1–2 |
| 4 | **이동 물체는 0.2 초 (모델에게 1–3 장) 만 가려진다** — 위반이 사실상 보이는 채로 일어난다 | 이동+가림 60 쌍 사건 가림 2–6 raw 프레임 (중앙 3 = 0.2 초), skip-2 로 1 장 (28) · 2 장 (14) · 3 장 (3); 정지+가림만 4.9 초 | 정본 (09-26, 같은 날 두 번 정정 후) | `z_research/OcclusionStats/Archive/INTPHYS1_OCCLUSION_2026-09-26.md` §2-1 |
| 5 | 표적 번짐 — 표준 표적은 양방향 target encoder (창 전체) 라 사건 뒤 정보가 사건 전 튜블릿에 번진다 | 인과 표적으로 바꾸면 복사 85.0 → 77.8, p 88.9 → 84.4; 사건 전 튜블릿 (pos/imp 픽셀 · p 동일) 에서 인과 표적은 정확히 50.0 | 정본 (H6 정정판) | `H6_*` §3-4 · `IP1_COPY.md` 요약 10 |
| 6 | Filtered 가 시작점의 약 40 % 에서 위반이 이미 문맥 안인 창을 고른다 | 그 창 정답 60.3 % — leakage 를 **부풀리지 않고 깎는다** (§3-3) | 등급 문서에 없음 (09-25) | `INTPHYS1_EVENT_POSITION_2026-09-25.md` §0 |
| 7 | 내용량 누수 (예측 없이 "예측 구간에 움직이는 픽셀이 얼마나 있나") | corr(내용차, surprise차) **IntPhys 0.486 vs IntPhysGen v1 0.922**; IntPhys A 그룹 (누수가 정답 방향) 60 쌍 100.00 · B 49 쌍 71.43 · Z 71 쌍 88.73 → 87.78 중 33 pt 가 공짜 | 기록 (08-20, **v1 데이터 · 단창 87.78**) — 현행 복사 기준선의 전신 (다른 측정) | `z_world_model_analysis/WHY_INTPHYS_WORKS_2026-08-20.md` §2–§6 |
| 8 | IntPhys 불가능 클립은 표현 공간에서 정상 분포를 벗어난다 (우리 v1 은 아님) | 가능 PCA-128 부분공간 밖 에너지 A(imp) − A(pos): IntPhys +0.0100 (88.2 % > 0) vs v1 +0.0036 (67.5 %) | 기록 (08-20) | `VERIFICATION_2026-08-20.md` §3.2 |
| 9 | 4 중항은 단일 프레임 감지기는 막지만 "문맥 대비 변화 감지기 (copy)" 는 막지 못한다 | (설계 논리 — 수치 없음) | 문헌 문서의 **해석** (원 논문 주장 아님) | `auto_research/Archive/LITERATURE_POSITIONING_2026-09-25.md` (G04a) |

**OcclusionStats — IntPhys 1 dev 의 가림 두 축** (180 쌍, mask 정답, `INTPHYS1_OCCLUSION_2026-09-26.md` §2-1)

| | 이동 + 가림 (60) | 정지 + 가림 (30) | 이동 + 보임 (60) | 정지 + 보임 (30) |
|---|---|---|---|---|
| 사건 종류 (쌍) | 가림 안 33 · 끝 직후 12 · **사건 전 미관측 15** | 가림 안 25 · 미관측 5 | 보이는 중 49 · 미관측 11 | 25 · 5 |
| 사건 가림 길이 | **2–6 프레임, 중앙 3 = 0.2 초** | **27–89, 중앙 73 = 4.9 초** | 없음 | 없음 |
| skip-2 장수 | 1 (28) · 2 (14) · 3 (3) | 13–44, 중앙 36 | — | — |
| 가려진 동안 이동 | **중앙 4.6 물체 폭** (3.4–8.7) | 0.14 폭 | — | — |
| `skip2_w32` 창 중 문맥 끝이 가림인 비율 | 12.9 % | **89.6 %** | 0.2 % | 0 % |
| V-JEPA 2 정답 (`skip2_w32`) | **75.0** | 100 | 91.7 | 100 |

→ 결론 (문서 문장): IntPhys 1 의 가림은 양극단 — 움직이는 물체는 0.2 초, 5 초 가림은 정지 물체뿐, **움직이는 물체가 1 초 넘게 가려지는 사건은 없다**. 정지+가림은 문맥 끝이 가림인 창이 90 % 인데도 100 % ("그 자리에 있다" 만 알면 된다). **IntPhys 1 의 88.89 는 움직이는 물체를 오래 가려도 이어 가는가를 재지 않는다.**
⚠️ 같은 날 두 번 정정 (v1 가짜 1 프레임 가림 5 scene → v2 가짜 1–1.6 초 가림 4 개 → 현행). 3 장 칸 n = 3, 쌍 비독립, CI 없음 — **가림 길이 효과로 읽지 않는다**. ❌ "IntPhys 2 는 움직이는 물체를 오래 가린다" (안 쟀다).
⚠️ 기제 7 (내용량 누수) 은 v1 데이터 · 단창 87.78 기준의 08-20 문서다. 그 문서의 "모델은 규칙 하나만 쓴다", "물리를 몰라도 내용량만 세면 된다" 는 (원 문서 표현 — 현재 금지). 같은 문서의 "IntPhys 물체가 2 배 크다" 는 **철회** (개수와 크기를 섞어 잰 오류, CLAUDE.md §6).
⚠️ dev set 에서 유형과 블록이 완전히 얽혀 있다 (shape 전부 O2) — "shape 가 쉽다" 와 "O2 (물체 많은 장면) 가 쉽다" 를 못 가른다 (`VERIFICATION_2026-08-20.md` 교란).

**출처**: `z_world_model_analysis/{VERIFICATION,WHY_INTPHYS_WORKS}_2026-08-20.md` · `z_research/OcclusionStats/{README.md, Archive/INTPHYS1_OCCLUSION_2026-09-26.md}` · `auto_research/Archive/{COPY_VS_POSITION_2026-10-07, H6_*}.md` · `z_research/Benchmarks/Archive/INTPHYS1_EVENT_POSITION_2026-09-25.md` · CLAUDE.md §6.

---

## 5. IntPhys 2 — leakage 가 작은 벤치에서 무너진다

### 5-1. permanence 감사 (`INTPHYS2_PERMANENCE_AUDIT_2026-09-26.md` §3, 1 차 + 2 차 정정 반영)

| | n | release w48 (C24) | Ariel43 w32 (C8) | **copy w48 (C42)** | release − copy (best-C) |
|---|---:|---|---|---|---|
| overall | 506 | 54.3 [50.4, 58.1] | 53.8 | **55.5 [51.8, 59.1]** | −1.2 [−5.5, +3.0] |
| permanence · Fixed | 52 | 59.6 [44.2, 75.0] | 71.2 [59.6, 80.8] | 63.5 [51.9, 75.0] | −3.8 [−17.3, +9.6] |
| permanence · Moving | 68 | 45.6 [35.3, 54.4] | 39.7 [29.4, 51.5] | 51.5 | — |
| permanence · all | 120 | 51.7 | 53.3 | 56.7 | −5.0 [−13.3, +3.3] |
| continuity · all | 120 | 61.7 [54.2, 69.2] | 62.5 | 64.2 | −2.5 |

- 같은 C = 24: release − 복사 **+1.8 [−1.8, +5.5]**, permanence all +2.5 [−4.2, +10.0], Fixed +3.8 [−7.7, +15.4] (`VERIFY_ISCENE2` §8). 전 C 평균은 copy 53.4 vs release 51.5 (w48).
- 결론 (문서 문장): release 의 IntPhys 2 permanence 는 Fixed · Moving · 전체 모두 CI 가 50 포함 — 논문의 "permanence 가 가장 쉽다" 는 우리 모델 · 채점에서 불성립; 우연을 넘는 조건은 continuity 뿐. 복사 permanence Fixed 63.5 는 CI 하한 > 50 → **"가장 쉬운 조건" 의 신호는 예측 없이 나온다**; permanence 점수는 사건 순간 예측이 아니라 재등장 뒤 장면 vs 마지막 관측의 전역 불일치. **IntPhys 1 (복사 85.0 vs 88.9) 결론이 IntPhys 2 로 확장.**
⚠️ **2 차 정정** — 원문 "복사가 같거나 조금 낫다 / −5 pt 앞선다" 는 best-C 산물 → "구분 안 됨". n 52–94 쌍 (CI ±15). Ariel 의 Fixed 71.2 는 Moving 39.7 (우연 아래) 과 짝이고, 창 선택에 민감하며 (w16 59.6), max 집계에서 46.2 로 사라진다. 복사는 LN 판.
⚠️ H2 ("문맥 경계 corner case") 는 이 하네스로 검정 불가 (metadata 에 사건 프레임 없음) → §5-2 의 onset 추정과 v11 이 그 자리.

### 5-2. 위반 시점 (onset) — 고정 창 예측 구간 밖이 32 % (`INTPHYS2_ONSET_2026-09-30.md`, 정본)

| 위반 | onset 중앙값 (영상 10.6 s) | onset < 4 s (고정 창 어디서도 예측 대상 아님) |
|---|---:|---:|
| continuity | 6.91 s | 24 % |
| immutability | 6.83 s | 22 % |
| permanence | 6.73 s | 19 % |
| **solidity** | **3.93 s** | **55 %** |
| 전체 | 6.05 s | 32 % |

- 검출 457 / 506 (90 %); 눈 검증 무작위 8 중 6 정확 (±0.3 s). **쌍은 위반 전에도 픽셀 단위로 같지 않다** (국소 셀 pre-level 1–65) → IntPhys 2 에서는 matched-context 채점의 "순수 기하 환원" (CLAUDE.md §1-2) 이 애초에 안 된다.
⚠️ ±0.5 s · 약 10–15 % 오류율 추정. solidity 저점과의 관계는 "시사". 폐기한 방법 셋 (전역 평균차 절대 문턱 등) 은 다시 쓰지 말 것.

### 5-3. encoder linear probe 73.7 vs VoE 54.3 — **지도 자라 다른 능력**

| 조건 | probe 쌍 정확도 | probe 영상 정확도 | VoE (릴리즈, w48 C24) |
|---|---:|---:|---:|
| 전체 (506 쌍) | **73.7 [69.9, 77.5]** | 62.5 (라벨 섞기 48.5–53.6) | **54.3** |
| continuity | 73.3 | 61.3 | 61.7 |
| immutability | **84.2** | 71.3 | 50.8 |
| permanence | **78.3** | 68.8 | 51.7 |
| solidity | 61.6 | 51.0 | 53.4 |

ViT-H 문맥 encoder 토큰의 T×N 평균 1280-d, 표준화 + L2 로지스틱, 장면 단위 외부 5-fold. 풀링 변형 넷 ±4 pt 안 (`INTPHYS2_ENCODER_LINEAR_PROBE_2026-09-30.md`, 기록).
→ 결론 (문서 문장): encoder 평균 벡터 하나에도 위반이 선형으로 남는다 — 그러나 절반만 (영상 정확도 62.5). "encoder 는 정보를 놓치지 않는다" 쪽의 증거이고 (07 장), 그 위에 예측이 얹히지 않는다는 것이 논지의 빈자리.
⚠️ **지도 학습된 자** (impossible 라벨을 보고 배움) — VoE 와 **같은 능력을 재지 않는다** (위반 후 장면 통계를 배울 수 있다). 프레임 배치도 다르다 (probe 9.3 s 한 창 vs VoE 6 fps 8 s 슬라이딩 + growing prefix). VoE 와의 차이를 "predictor 가 못 한다" 로 옮기려면 라벨 없는 검사가 필요하다. "encoder 가 물리를 이해한다" 는 (내부 귀속 표현 — 현재 금지, 문서도 금지로 명시).

### 5-4. 다시 학습한 predictor 도 구분 안 됨

ViT-H 릴리즈 w16 / w32 / w48 = 53.16 / 52.77 / 54.35 vs Ariel `bc_ep30` 53.16 / 53.75 / 52.96 (§2-6); Ariel full 53.75 · prefix 53.56 · AR ep18 51.19 · AR ep38 50.79 — **전부 우연 근처** (`ARIEL_CHECKPOINTS.md` §5-3). copy-hard 240 쌍에서 릴리즈 21.2 · prefix 19.2 · AR 40.4 (§3-4). → 08 장 ("다시 학습해도 같다") 의 IntPhys 2 판.
⚠️ Ariel 은 SSv2 + K400 학습 (IntPhys 2 도메인 밖), AR `video_batch 8` vs 릴리즈 1 → ±0.5 pt 흔들림.

### 5-5. IntPhys 2 를 읽을 때의 단서

- 논문 (Bordes et al. 2025) 은 NeurIPS 2025 D&B **Reject** (OpenReview 리뷰: "Is it due to memory? Or representation and perception?") — G04a (venue 는 문헌 문서 사이에서 "채택 미확인" vs "Reject", 종합 = Reject).
- 논문 Table 2 의 IntPhys 1 열 87.22 는 IntPhys 2 프로토콜 값 — Garrido 88.89 와 섞지 않는다 (§2-7).
- 우리 구현은 공식 config 가 실행 불가 (`TypeError`, `filename` 컬럼) 여서 코드를 따라 다시 짰다; C 목록은 논문 D.3 텍스트 (4–14) 가 아니라 config 비율 (`PROTOCOL_CHECK_2026-09-14.md`, 대체됨 → `analysis/intphys2/README.md`).
- 정본 §2-2 (c): IntPhys 2 는 평가한 모든 모델이 사람 (96.4) 에 크게 못 미친다고 보고한다 (V-JEPA 2 계열 57.51 — 논문 수치).

**출처**: `auto_research/Archive/{INTPHYS2_PERMANENCE_AUDIT_2026-09-26, VERIFY_ISCENE2_2026-09-26, INTPHYS2_ONSET_2026-09-30, INTPHYS2_ENCODER_LINEAR_PROBE_2026-09-30, COPY_HARD_SUBSET_2026-09-29}.md` · `z_training/ARIEL_CHECKPOINTS.md` §5-3 · `analysis/intphys2/{README,PROTOCOL_CHECK_2026-09-14}.md` · `z_research/Benchmarks/exp_results/intphys2/*/column_best.md` · 정본 §2-2 (c).

---

## 6. 실영상 · downstream — EK100 action anticipation

### 6-1. 재현과 평가 규약 (`z_research/anticipation/EK100/README.md` §A-1, 정본)

| TAG | 학습 규약 | 평가 규약 | verb | noun | action |
|---|---|---|---:|---:|---:|
| `ek100_vith_official256_released_grid8` (20 ep, head 8) | released | released | **56.36** | **56.16** | **35.34** |
| `ek100_vith_rel_grid8_valpaper` (같은 ckpt, val 만) | released | paper | 26.90 | 35.28 | **14.14** |
| `ek100_vith_official256_paper_grid8` (20 ep, head 8) | paper | paper | 33.25 | 35.24 | **19.07** |
| `ek100_vitl_rel_resized_mask0` (공개 ViT-L probe) | released | released | 48.58 | 47.61 | 28.51 |
| `ek100_vitl_paper_resized_mask0` (같은 공개 probe) | released | paper | 24.25 | 31.48 | 11.92 |
| 논문 Table 5 ViT-H | — | — | 59.2 | 54.6 | 36.5 |

`released` = 문맥이 **action 끝** 1 s 전에 끝난다 (릴리즈 코드) / `paper` = 문맥이 **action 시작** 1 s 전에 끝난다 (논문 서술). **사용자 결정 (09-20): 기준은 `released` 규약 56.36 / 56.16 / 35.34** (논문과 비교 가능한 유일한 축), `paper` 19.07 은 "정직한 anticipation" 으로 병기.

### 6-2. 규약 누수 — 문맥에 정답 action 프레임이 든 비율 (§2-A)

| 규약 × 시간 기준 | val 문맥에 정답 action 프레임 포함 | 비고 |
|---|---:|---|
| **released × frame** | **88.6 %** (평균 14.00 / 32 장) | 릴리즈 코드 기본 |
| released × timestamp | 78.8 % | |
| paper × frame | 6.9 % | |
| paper × timestamp | 0.1 % | |

→ 결론 (문서 문장): **같은 모델 · 같은 probe 인데 문맥에 동작이 보이면 35.34, 안 보이면 19.07** — 릴리즈 규약 수치의 절반 가까이가 "이미 시작된 동작을 보고 있다" 에서 온다 (우연 0.45 의 42 배 — "못 한다" 가 아니라 두 배 차이). 학습 규약을 벗어나면 이식되지 않는다 (35.34 → 14.14). 이것이 EK100 판 leakage 다 (복사 기준선이 아니라 평가 규약으로 잰 것).
⚠️ val 은 `timestamp`, ckpt 는 `frame_fixfps` 로 학습 — 규약만의 효과를 떼려면 `frame_fixfps` arm 이 더 필요. "8 비디오 0.9 % 무해" 는 틀린 서술 (59.94 fps 도 최대 2 s 밀림, 90 fps 비디오 있음) → `timestamp` 기본. encoder 만 arm (Table 20 대조) 미실행.

### 6-3. predictor 출력 치환 · 지평 스윕 (SC3 · SC5, 1 차 · SC5 §3 은 `VERIFY_ISCENE2` §7 확정)

고정 probe 의 predictor 미래 토큰만 바꾼다 (새 학습 없음, val 9,296 clip, 8 head 중 열별 최고 — 낙관 규약).

| 규약 · 지평 | 원래 predictor | `copy_last` | `mean` (공간 정보 없음) | Δ (원래 − copy_last) | 같은 head 고정 Δ |
|---|---:|---:|---:|---:|---:|
| released 1 s | 35.34 | 32.80 | 34.01 | **+2.54** | +2.54 |
| released 2 s | 30.20 | 28.10 | — | **+2.10** | +2.47 |
| released 3 s | 24.91 | 22.85 | — | **+2.06** | +2.12 |
| paper 1 s | 19.07 | 16.46 | 17.59 | +2.61 | **+4.67** |
| paper 2 s | 15.22 | 12.47 | — | +2.75 | +2.86 |

- 두 규약 모두 action 의 86–96 % 가 predictor 미래 없이 남는다; verb · noun 하락은 1.5 점 안 (SC3).
- 결론 (SC5 문서 문장): Δ 는 지평이 늘어도 줄지 않는다 (약 2–3 점 일정) → **"EK100 은 지평 1–3 s 어디서도 predictor 에 거의 기대지 않는다 (encoder + probe 가 대부분)"**. "EK100 은 reach 밖을 못 본다" 는 쓰지 않는다 (예측 (a) 기각).
- 저자 Table 20 (g384): encoder 만 39.1 / predictor 만 20.2 / 둘 다 39.7 — 본문 "mostly requires strong semantic understanding, as opposed to forecasting capabilities" (`PREDICTOR_SHARE` 부록, 로컬 PDF p.44 직접 확인).
⚠️ 치환 하락은 predictor 기여의 **상한** (probe 가 원래 출력에 맞춰 학습 → 치환은 분포 밖). probe 학습 지평 0.25–1.75 s 라 2–3 s 는 분포 밖 — 절대값이 아니라 Δ 만 읽는다. clip CI 없음 (예측 파일 미저장).

### 6-4. encoder 는 ego-motion 을 담는다 · 어떤 predictor 도 reach 를 못 늘린다

| 표적 (ridge, clip 5-fold CV R², z) | K400 인공 팬 | SSv2 | EK100 |
|---|---|---|---|
| (a) 전역 flow x / y (ego-motion 대리) | .99 / — | .15 / .26 | **.69 / .63** (10-fold .709 / .651, 영상 단위 split .685 / .614) |
| flow 자기 일관성 clip-CV (표적 신뢰도 대리) | GT .75 | .15 / .09 | .53 / .36 |
| (b') 국소 잔차 (물체 · 시차) | — | .22 / .01 | .04 / .09 |
| 미래 변위 중 카메라 몫 슬롯 0 / 3 / 7 | — | .17 / .29 / .34 | **.42 / .65 / .74** (전역 67–75 %) |

- 결론 (문서 문장): EK100 encoder 는 ego-motion 을 flow 표적의 자기 일관성보다 높게 읽는다 → "1 인칭 카메라 운동을 encoder 가 못 담아서 EK100 reach 가 2.2 에 멈춘다" **기각**. 어떤 predictor (릴리즈 · full · prefix · AR · post-FT) 도 EK100 reach 2.2 를 못 넘는다 (릴리즈 R50 2.1, `I_SCENE_CAUSAL_BOTTLENECK_2026-09-25.md` §4-2).
⚠️ **정정 (`VERIFY_R3` B2 · B3)**: "표적 신뢰도의 천장까지 읽는다" → "자기 일관성보다 높게"; "74 %" → "67–75 %". 국소 잔차 (물체 운동) 는 거의 못 읽는다 — encoder 한계인지 표적 한계인지 미결. "encoder 가 미래를 표현한다" 로 쓰지 않는다 (문서도 "담는다 (문맥 끝 운동)" 까지만).

### 6-5. probe 를 AR predictor 로 바꿔도 같다

EK100 (1/3 train · 20 ep · val 8,761): **AR ep40 action 22.03 / verb 48.21 / noun 36.94 vs 릴리즈 21.80 / 48.26 / 36.04** (릴리즈 재현 간 차 action 0.06) — 잡음 안 (`AR_PREDICTOR_2026-09-27.md` §2-9, 10-01 재실행).
⚠️ **정정 (10-01)**: 09-30 "AR ep40 val 21.74" 는 `predictor_checkpoint` 키 누락으로 **릴리즈가 돈 것** — 그 run 들 (`ek100_rel_third10_grid8_ar40`, `ek100_vith_official256_released_grid8_ar40`) 인용 금지. 1/3 val 8,761 은 희귀 클래스가 빠진 val 이라 전체 기준선 35.34 (9,296) 와 비교 금지.

→ 이 절이 P2 에 주는 문장: **anticipation 에서 world model 의 발전으로 보고되는 점수도 predictor 의 미래 예측을 거의 요구하지 않는다** (복사 치환 −2 ~ −3, 저자 Table 20 +0.6). P2 의 "prediction · anticipation 에서 빠르게 발전" 은 그 자체로 intuitive physics 를 따르는 미래 예측의 증거가 아니다.

**출처**: `z_research/anticipation/EK100/README.md` · `auto_research/Archive/{SC3_EK100_PREDICTOR_SUBSTITUTION_2026-09-25, SC5_EK100_HORIZON_SWEEP_2026-09-26, VERIFY_ISCENE2_2026-09-26, ENCODER_MOTION_READOUT_2026-09-29, VERIFY_R3_2026-09-29, AR_PREDICTOR_2026-09-27, PREDICTOR_SHARE_2026-09-29, I_SCENE_CAUSAL_BOTTLENECK_2026-09-25}.md`.

---

## 7. Figure 1 후보 · P2 그림 후보

### 7-1. Figure 1 hook 후보 (정본 §2-2 — 사람에게는 뻔한데 모델은 틀리는 장면, P1 의 두 예시와 짝)

| 후보 | P1 예시 | 모델이 내는 것 | 근거 (장) | 단서 |
|---|---|---|---|---|
| **가림막 뒤로 굴러가는 공** (문맥 끝 가림) | 사라지지 않는다 | p 의 미래에 처음부터 물체다운 토큰이 없다; '물체→빈' 미래를 덜 놀랍게 본다 (v11 0 %, 실물 메시 0–10 %) | CLAUDE.md §5-1 · §5-5, `z_research/v11_realistic/README.md` §1 (05 장) | 경계 가림은 corner case 비판 (§9 4-3) |
| **선반 끝을 지나는 공** | 중력 | 떨어지는 미래보다 같은 높이로 떠 가는 미래가 덜 놀랍다 (실물 메시 120 / 120 쌍); 위치 자는 선반 높이 유지 | `v11_realistic` §2 · RollOutV2 §5-5 (06 장) | L1 이 덜 움직이는 미래를 싸게 볼 수 있다 (§9 4-4) |
| 위로 던진 공 (문맥이 오르는 중 · 꼭대기에서 끝남) | 중력 | 오르는 속도는 잇되 꼭대기에서 돌아 내려오지 않는다; 꼭대기 문맥에서 공 자리는 98 % block 이 높이 유지 쪽 | `v11_realistic` §4 (10-08 정정판, 06 장) | 튜블릿별 판정은 흔들린다 — 줄의 흐름으로만 |

추천 (정본): **앞 두 개를 한 그림의 두 패널** (P1 의 "사라지지 않음 · 중력" 과 1:1). 모델의 미래는 latent 라 그대로 못 그린다 → **후보 미래 중 무엇에 가까운가** (파라미터 0, `fig_gravity_choice` 형식) 를 주 표현으로, 검증된 위치 자 궤적을 보조로. 정본 §5 #1 (담당 Claude, CPU · 기존 산출물) — **★ 아직 안 그렸다**.
⚠️ ledge · wall 을 쓰는 조건 넷 (정본 §2-3 iii): 떠 가는 미래는 복사와 닮았다 (H23 §3-7) → 불가능 미래가 복사와 다른 방향인 다중 미래 설계로 / 갈림 시점 누수 / 자 둘의 다른 답 / L1 교란 (06 장).

### 7-2. P2 그림 후보 — leakage (복사 정확도, x) vs 모델 정확도 (y)

벤치마크 · 칸마다 한 점. 대각선 위에 붙으면 "모델이 leakage 이상을 못 한다" (정본 §2-2). 지금 있는 점과 빈칸 (같은 표적 공간 안의 복사만 쓴다):

| 점 | x = 복사 | y = predictor | 모델 | 출처 (§) |
|---|---:|---:|---|---|
| IntPhys 1 `skip2_w32` 표준 | 85.00 | 88.89 | V-JEPA 2 ViT-H | §3-1 #1 |
| IntPhys 1 `skip2_w16` 표준 | 73.89 | 83.33 | ViT-H | #3 |
| IntPhys 1 `skip5_w16` 표준 | 63.33 | 59.44 | ViT-H | #5 |
| IntPhys 1 V1 (깨끗한 창) | 87.22 | 91.67 | ViT-H | #2 (분석용 변형 — 별도 기호) |
| IntPhys 1 AR 공간 `skip2_w16` | 83.33 | 90.56 | Ariel AR ep18 | #7 |
| IntPhys 1 `skip2_w16` (DINO-F 최고 칸) | 88.9 | 84.4 | DINO-Foresight | #9 |
| IntPhys 2 w48 같은 C24 | 52.6 | 54.35 | ViT-H | #10 |
| v11_split_test | 73.24 | 75.70 | ViT-H | #11 |
| v11 전체 | 76.44 | 77.10 | DINO-Foresight | #15 |
| v11 late vanish 물체→빈 static / flat / ramp | 67.9 / 39.3 / 9.8 | 65.2 / 2.7 / 0.0 | ViT-H | #14b |
| RollOut_v2 ledge / wall | 7.9 / 2.3 | 0.0 / 17.6 | ViT-H | #16 |
| **GRASP level2** | ★ 없음 | ★ 미실행 | — | §2-5 |
| **InfLevel continuity / macro** | ★ 없음 | 72.85 / 58.27 (ViT-H) | — | §2-4 |
| V-JEPA 2.1-g · VideoMAEv2-g (IntPhys 1) | ★ 없음 | 51.11 / 54.44 | — | §2-2 |

⚠️ EK100 (action R@5, 35.34 vs 32.80) 은 지표가 달라 같은 축에 두지 않는다 — 따로 패널로.
⚠️ 빈칸을 채우는 일은 정본 §5 #2 ("leakage 그림 — GRASP · InfLevel 에 복사 기준선", GPU 소량, 담당 Claude). 2.1-g · VideoMAEv2-g 는 복사 기준선이 없고 4 중항 안 점수 미움직임 (2.1-g) 문제가 있다.
⚠️ 점마다 칸 선택이 다르다 (A.8 최고 · best-C) — 그림 캡션에 "descriptive, 같은 칸 안의 복사" 를 적는다.

**출처**: 정본 §2-2 · §5 · §2-3 iii · `z_research/v11_realistic/README.md` · §3 의 각 출처.

---

## 8. P2 문헌 · 선점 위험 (G04a, 문헌 문서 셋의 표를 옮김)

확인 표기는 원문 그대로: ✅ 원문 확인 · ◐ 초록 / 요약만 (인용 전 재확인) · ❓ 미확인. 문서 속 "우리 수치" 는 문헌 문서가 다른 문서에서 옮겨 적은 것이라 문헌 문서 안에서 재검증되지 않았다 (G04a 머리).

### 8-1. 인용 후보 (P2a world model 발전 · P2b intuitive physics 벤치마크 · P2c VoE leakage · 복사 기준선 선행)

| 문단 | 논문 | 확인 | 한 줄 요지 | 우리 문장과의 관계 |
|---|---|---|---|---|
| P2a | V-JEPA 2 (2506.09985), Assran et al. 2025 | ✅ | EK100 Table 20 encoder 39.1 / pred 20.2 / both 39.7, "mostly requires strong semantic understanding, as opposed to forecasting"; §4 AR 오차 누적; §5 "inherently limited by the state information encoded" | 지지 (P2 · EK100 복사) · must-cite |
| P2a | V-JEPA 2.1 (2603.14482) | ✅ | dense loss, EK100 40.8; predictor/rollout 변경 없음 | 선행 — 리뷰어 첫 질문 "2.1 에서도?" (IntPhys 1 은 51.11, §2-2) |
| P2a | DINO-world (2507.19468) | ✅ | frozen DINOv2 + future predictor 도 "inaccurate as the interval approaches 1 second"; VoE 를 "sanity check rather than a benchmark" | 지지 (VoE 격하 인용구) + 반례 (새 predictor 가 자명한 해결 아님) |
| P2a | Navigation World Models (2412.03572) | ✅ | "After 8 seconds, predictions degrade" | 선행 (복사 기준선 없음) |
| P2a | Genie 3 (blog) | ✅ (비심사) | "a few minutes of continuous interaction" | 배경 |
| P2a | Cosmos (NVIDIA 2025–26) | ◐ (Cosmos 3) | IntPhys 2 에서 Cosmos-Predict-4B chance 근처 | 지지 (생성형도 chance) |
| P2a | Objects in Generated Videos Are Slower Than They Appear (2512.02016) | venue 미확인 | 생성 영상 유효 중력 0.38–2.27 m/s² | 지지 (3a 가속의 생성형 대응) |
| P2b | Garrido et al. 2502.11831 | ✅ | IntPhys 98 / GRASP 66 / InfLevel 62; 대조군 = 무작위 초기화 · VideoMAEv2 · MLLM; **copy-last 없음**; "never trained using a causal prediction task" | **선행 (반박 대상 측정)** — "반박 아닌 분해" 톤 |
| P2b | IntPhys (1803.07616), Riochet et al. | ✅ | 원 벤치마크; "Static" 은 장면 조건, copy-last 모델 기준선 없음 | 선행 (복사 대조 부재) |
| P2b | IntPhys 2 (2506.09849), Bordes et al. 2025 | ✅ (NeurIPS D&B Reject) | 전 모델 ≤ 57.5; permanence "easiest"; 재등장 시 surprise 상승 없음 | 지지 + 긴장 (§5-1 이 해소) |
| P2b / P2c | MVP (2506.09987), Krojer et al. TMLR 2025 | ✅ | minimal video pair 로 shortcut 차단, SOTA 40.2 vs 인간 92.9 | 지지 (matched pairing 의 인용 근거) |
| P2b | InfLevel, Weihs et al. 2022 TMLR | ◐ | README "tight control opens the door for designing heuristics" | 지지 (heuristic leakage 저자 명시; static 기준선 미확인) |
| P2b | Physics-IQ (2501.09038) | ✅ | best 29.5 %; visual realism 과 물리 무상관; copy-last 없음 | 선행 (정적 프레임 점수 빈칸) |
| P2b | LikePhys (2510.11512) | ✅ | 기존 방법은 물리와 외형을 못 가른다 | 지지 (surprise 외형 교락) |
| P2c | Physion (2106.08261), Bear et al. 2021 | ✅ | observed + simulated ≈ observed (p = 0.53) | **선행 / 선점 (no-dynamics 대조)** — 결론은 반대 ("지각이 병목") |
| P2c | Nayebi et al. NeurIPS 2023 (2305.11772) | 등급 문서에 없음 (상세 인용됨) | No Dynamics (마지막 문맥 latent 복사) 19.03 / 24.79 % vs CTRNN 46.34–48.83 % | **선행 (우리 복사 기준선과 같은 설계 — 표준 기준선으로 인용 가능)** |
| P2c | Joseph, Garrido et al. 2602.07050 | ✅ | App. C.1.4 layer 0 위 predictor 도 VoE 점수 | 지지 (저자 측 증거) |
| P2c | Punzo et al. 2606.09646 | ✅ | encoder 만 probe; IntPhys 2 linear 50.98 / attentive VideoMAE 73.9; "generic motion sensitivity … dataset biases" | 지지 (§5-3 과 대조) |
| P2c | Human-JEPA (2608.21160), Wei, Sun, Zhao 2026 | ✅ | copy-last latent 0.873 vs 0.798, 마진 0.057 → 0.087 | **선점 (복사 귀무), 반드시 인용** |
| P2c | GEOPHYS (2606.20707) | ✅ 초록 · split ❓ | frozen image embedding 기하 5 개로 IntPhys 2 93.3 % | 선점 위험 + 지지 — **split ❓ 라 인용 금지** (G04a (b)) |
| P2c | R2M-Bench (2608.27328) | ✅ | revisit 유사도는 motion 과 교락 (ρ 0.207), gap-matched 상대화 | 지지 (copy 기준선의 생성형 쌍둥이) |
| P2c | Luc et al. 2017 (1703.07684) · DINO-Foresight (2412.11673) | ✅ | "The first baseline copies the last input frame" · Copy-Last | 선행 (copy-last 계보 — 복사 기준선이 표준이던 곳은 semantic / feature forecasting) |
| P2c | WMReward (2601.10553), CVPR 2026 | 등급 문서에 없음 | V-JEPA 2 surprise 를 생성 보상으로 | 지지 (use case — 복사 결과가 이 관행에 닿음) |

- 문헌 문서의 실사 (G04a P-29 §3-1): 벤치마크별 복사 / 정적 기준선 — IntPhys 1 **없음 ✅** · IntPhys 2 **없음 ✅** · GRASP **없음 ✅** · InfLevel 미확인 ◐ · Physics-IQ 없음 ✅ · EK100 (RULSTM · AVT · TAP-JEPA · V-JEPA 2) 없음.

### 8-2. 선점 위험 (같은 주장을 먼저 한 논문)

| 위험 | 논문 | 선점한 것 | 남는 delta (문헌 문서 기준) |
|---|---|---|---|
| 최대 | StEvo-Bench (ECCV 2026) · WRBench (2606.20545) | 관측이 끊겨도 바뀌어야 하는 미래 개념 · 'tracking shot' (**생성형**) | latent predictor · 국소화 · 점수 감사. 개념 최초 주장 금지 |
| 높음 | Human-JEPA (2608.21160) | JEPA predictor 에 copy-last 귀무 (자기 predictor, cosine) | 릴리즈 predictor · 벤치마크 점수 · 기제. **반드시 인용** |
| 높음 | Garrido et al. 2601.05230 (FAIR) | frozen V-JEPA 2 + 실영상 action-free 새 predictor | 복사 귀무 · encoder 정보 국소화 · 무엇을 채울지 먼저 잼 |
| 높음 | HERA (2608.05523) | frozen V-JEPA 2 predictor + 기억 adapter, 동기 문장 동일 (IntPhys 2 52.57 → 54.35) | 학습 없는 측정 · 복사 대조 · 위치 readout |
| 높음 | GEOPHYS (2606.20707) | predictor 없이 IntPhys 2 풀림 | 판별기 vs 예측기 — "그럼 predictor 는 왜?" 반문에 답해야 함 |
| 중 | Physion (2021) · Nayebi (2023) | no-dynamics 대조 · frozen encoder + dynamics | 결론 반대 (Physion "지각 병목") / 다른 encoder · 과제 |
| 중 | Joseph (ICML 2026) · Bouwens (ICML 2026 WS) · Punzo | encoder 물리 probing | predictor 미분석 |
| 중 | DINO-world · V-JEPA 2-AC · JEPA-WMs · UWM-JEPA · DINO-Foresight | frozen encoder + 새 predictor 표준, VoE "sanity check" | 새 predictor 자체는 새롭지 않다 |
| 낮음 | Mechanistic View (2601.17067) · Kang ICML 2025 · Vafa 2024/2025 | 'stateless' 선언 · case-based retrieval · 점수 ≠ 상태 | 층위 · 대상 다름 |

⚠️ 문헌 문서 셋 (09-25 · 09-26 · 09-29) 의 논지 틀 (reach · permanence · closure / state evolution · belief / 주 평가 EK100 · Ego4D) 은 10-08 정본으로 **대체됨** — 문헌 표는 유효, 프레임 문장은 현재 금지 (memory / state) (G04a (d)).
⚠️ **문서 간 차이** — Physics-IQ 는 문헌 문서에 ✅ 인데 정본 §2-2 (c) 는 "생성형 world model 쪽 대응 문헌 (Physics-IQ 류) 은 **실물 확인 후** 인용" 이라 적었다. 이 아카이브는 정본을 따른다 (논문에 넣기 전 원문 재확인).
⚠️ ❓ 인 것: StEvo 모델 목록 · "< 10 %" 수치, WRBench Wan 0.66 → 0.62 (abstract 에 없음), InfLevel static 기준선, NWM VAE 동결 — 전부 미확인 (G04a).

**출처**: `auto_research/Archive/{LITERATURE_POSITIONING_2026-09-25, LITERATURE_STATE_WORLD_MODELS_2026-09-26, LITERATURE_PREDICTOR_ROLE_2026-09-29}.md` · 정본 §2-2 (c).

---

## 9. 리뷰어 공격과 남은 단서 (정본 §4-1 · §4-2 · §4-3 + 이 장의 단서)

| # | 공격 | 답 | 남는 단서 (결론과 같은 무게) |
|---|---|---|---|
| 4-1 | "Garrido 는 창발했다고 했다 (IntPhys 1 88.89)" | **IntPhys 1 은 leakage 가 있는 벤치다** — 복사만으로 85.0 (Δ CI 0 포함), DINO-F 는 복사가 더 높다, 위반이 외형 사건뿐이고 이동 물체 가림은 0.2 초. leakage 가 작은 IntPhys 2 에서는 복사 수준 | 반박 대상은 "창발" 자체가 아니라 **그 근거로 쓰인 점수가 예측을 요구하지 않는다** 는 것. leakage 를 복사로 잰다는 정의를 본문에 먼저 (정본 §4-1) |
| 4-2 | "V-JEPA 2 하나 아닌가 — world model 일반이라 할 수 있나" | 잰 것: V-JEPA 2 ViT-H/L 릴리즈 · Ariel 세 팔 · post-FT 넷 · 사전학습부터 다시 한 8 개 · DINO-Foresight (다른 encoder + frame predictor, v11 가림에서 같은 실패) | **픽셀 생성형 world model 은 안 쟀다** → P2 의 "world model" 은 latent 예측형으로 좁히거나 하나를 더 잰다 (정본 §5 #6, 모델 선택은 사용자) |
| 4-3 | "문맥 끝 가림은 corner case" (09-03 외부 피드백) | 실영상에서 움직이는 내용의 약 절반이 reach 밖 (SSv2 46 · EK100 51 %, `I_SCENE` §4-3 C), 실영상 가림 띠에서도 절반 복원. 벤치마크는 이 상황을 거의 안 낸다 (IntPhys 1 이동 물체 가림 0.2 초) | "context 에서 안 보이면" 은 **마지막 관측** 으로 좁혀야 측정과 맞는다. "가림 띠에서 절반 복원" 은 `VERIFY_ISCENE2` §4-4 가 같은 clip 비가림 대비로 다시 재서 **약 1/3 만 남는다 ("절반" 은 과소 추정)** 로 정정했다 (hid .63 → .21, vis .45 → .43). IntPhys 2 · GRASP · InfLevel 의 가림 통계는 아직 없다 (정본 §5 #7, 추적기 필요) |
| — | "인과 표적 · `skip2_w16` 에서는 Δ 가 유의하다 — 복사가 이긴다는 것은 골라 쓴 것" | 맞다 — Δ 는 칸 · 표적에 의존한다 (+3.9 ~ +12.2, `skip5_w16` 은 −3.9). 주장은 "Δ = 0" 이 아니라 **"Δ 가 작다 — 표준 표적 주 칸 +3.9, 가장 큰 칸도 +9.4 ~ +12.2 (판에 따라)"** (§3-2; `PREDICTOR_SHARE` 의 "+2 ~ +7 을 넘지 않는다" 요약은 그 표의 +9.4 와 어긋난다) | 표에는 모든 칸을 같이 낸다. 깨끗한 창 (V1) 에서도 +4.4, CI 하한 +0.6 |
| — | "복사 기준선이 공정한가 (online z 를 EMA target 공간에 댄다 · LN 판)" | LN 을 빼면 77.78 로 p 가 +11.1 앞선다 — 그러나 표적과 같은 척도인 LN 판이 예측 없는 기준선으로 알맞다 (H6 §3-7) | 깨끗한 copy (`copy_h_ctxonly`) 미실행 — 더 센 기준선일 수 있다 |
| — | "best-C · A.8 최고 칸은 같은 데이터에서 고른 값" | 논문 (Garrido · Bordes) 도 그렇게 보고한다; IntPhys 2 는 같은 C 판을 주 값으로 | held-out 추정치가 아니다 — descriptive 라고 캡션에 쓴다 |
| — | "IntPhys 2 가 낮은 건 장면 복잡도 · 짧은 문맥 때문" | "leakage 가 작아서 무너진다" 는 상관 서술로만 쓴다; 같은 장면에서 leakage 만 바꾼 v11 가림 위치 축 (visible · early · mid · late) 을 같이 보인다 (05 장) | IntPhys 2 onset 의 32 % 는 고정 창 예측 구간 밖 · 쌍 문맥 비동일 (§5-2) — 프로토콜 자체의 한계도 섞인다 |
| — | "GEOPHYS 는 predictor 없이 IntPhys 2 를 93 % 푼다 — 그럼 predictor 는 왜?" | 판별기와 예측기는 다른 질문이다; 우리 주장은 "VoE 점수가 예측을 요구하지 않는다" 이고 이는 GEOPHYS 와 같은 방향 | GEOPHYS split ❓ — 인용 전 확인 (§8-1) |
| — | "encoder linear probe 가 73.7 이면 IntPhys 2 도 풀리는 것 아닌가" | 지도 자는 impossible 라벨을 보고 배운다 — VoE 와 같은 능력이 아니다 (§5-3) | 라벨 없는 검사로 옮기기 전에는 "predictor 가 못 한다" 의 근거로 쓰지 않는다 |
| — | "위치 위반 0 % 는 metric 탓" | 반은 맞다 — 토큰 평균 L1 은 위치를 거의 못 본다 (`COPY_VS_POSITION`) | 위치 주장은 검증된 위치 자 (RollOutV2 §5-5) 와 같이 쓴다 (06 장) |
| — | "EK100 도 world model 발전의 근거 아닌가" | 복사 치환 −2 ~ −3 (지평 1–3 s 일정), 저자 Table 20 +0.6, 규약 누수 88.6 % (§6) | EK100 은 clip CI 가 없고 head 선택이 낙관 규약이다 |

**금지 표현 점검** (정본 §7): "world model 은 물리를 이해하지 못한다" (조건 없이) · "직관물리는 창발하지 않았다" (반박 대상은 그 측정 — "IntPhys 1 은 leakage 가 크다, 예측 없이 85 %") · "IntPhys 1 은 틀린 벤치다" (잰 것은 leakage 크기) · "모델이 중력을 안다 · 모른다" · "predictor 가 쓸모없다" (→ "벤치마크가 예측력을 요구하지 않는다") — 이 장은 이 문장들을 쓰지 않는다.

**출처**: 정본 `auto_research/paper/PAPER_STORY_2026-10-08.md` §4 · §5 · §7 · `auto_research/Archive/I_SCENE_CAUSAL_BOTTLENECK_2026-09-25.md` §4-3 · 이 장 §2–§8 의 출처.

---

## 출처

정본 · 논리
- `auto_research/paper/PAPER_STORY_2026-10-08.md` — §1 · §2 (문단 뼈대) · §2-2 (P2 (a)(b)(c) · Figure 1 후보) · §2-3 iii · §4 (4-1 · 4-2 · 4-3) · §5 · §7 · §재현 (10-08 재확인표)
- `auto_research/paper/RETHINK_PREDICTOR_ROLE_2026-09-29.md` — §5 · §5-b (대체됨 → 정본, §5 · §5-b 는 그대로 유효)
- `CLAUDE.md` — §1-2 · §1-3 · §1-7 · §2-7 · §5-1 · §5-4 · §5-6 · §6

외부 벤치마크
- `z_research/Benchmarks/README.md` — §0 · §1 · §4 · §6
- `z_research/Benchmarks/PROTOCOLS.md` — §1-b · §2 · §3 · §5-3 · §7
- `z_research/Benchmarks/Archive/INTPHYS1_EVENT_POSITION_2026-09-25.md` — §0 · §2 (§5 대체됨)
- `z_research/Benchmarks/exp_results/_invalid_20260921_mixedgrid/README.md`
- `z_research/Benchmarks/exp_results/intphys2/bench_intphys2_main_{vith,vith_w16,vith_w32,ariel_bc_ep30_w16,ariel_bc_ep30_w32,ariel_bc_ep30_w48}/column_best.md`
- `z_research/OVERVIEW_2026-09-22.md` — §5-8 · §9-2 (§6 스토리는 대체됨)
- `analysis/intphys2/README.md` · `analysis/intphys2/ANALYSIS_NOTES.md` · `analysis/intphys2/PROTOCOL_CHECK_2026-09-14.md` (대체됨) · `analysis/intphys2/configs/_superseded/README.md`
- `configs/protocols/datasets.md` — `grasp_level2`

leakage · 복사 기준선
- `z_research/TrainingEffects/ip1_copy/IP1_COPY.md` — 요약 · 표 2 · 2b · 2c · H6g 대조표
- `auto_research/Archive/H6_INTPHYS1_WHAT_THE_SCORE_MEASURES_2026-09-24.md` — §3-1 ~ §3-7 · §5 (정정판)
- `auto_research/Archive/PREDICTOR_SHARE_2026-09-29.md` — 표 · 단서 · 부록 (Table 20)
- `auto_research/Archive/COPY_HARD_SUBSET_2026-09-29.md` — §2 ~ §4 · `VERIFY_R3_2026-09-29.md` C3 (정정)
- `auto_research/Archive/INTPHYS2_PERMANENCE_AUDIT_2026-09-26.md` — §3 · `VERIFY_ISCENE2_2026-09-26.md` — §7 · §8
- `auto_research/Archive/DINOF_INTPHYS1_2026-10-02.md` · `DINOF_V11_2026-10-07.md` · `COPY_VS_POSITION_2026-10-07.md`
- `z_research/TrainingEffects/scores/ANALYSIS.md` (v11 +2.5 구성 — 색 +9.3 [+7.7, +10.9] · 모양 −1.6 · 물체→빈 −6.8 · 색 빼면 −1.7) · `z_research/TrainingEffects/Archive/VERIFICATION_2026-09-25.md` (같은 값)
- `z_training/ARIEL_CHECKPOINTS.md` — §1 · §5-1 · §5-2 · §5-3
- `z_training/RESULTS_2026-09-19.md` (post-FT IntPhys 1 93.89 · 도메인 안)
- `auto_research/Archive/TRAINED_PREDICTORS_SIGNATURES_2026-09-25.md` §3-6 (Ariel prefix ep43 IntPhys 1 −16.1 의 다른 출처)

왜 새나 · IntPhys 2
- `z_world_model_analysis/VERIFICATION_2026-08-20.md` — §2.2 · §2.3 · §3.2 (α 관련 수치 인용 금지)
- `z_world_model_analysis/WHY_INTPHYS_WORKS_2026-08-20.md` — §2 ~ §6 (§5 철회 포함)
- `z_world_model_analysis/MECHANISM_2026-08-20.md` (단창 88.33 만)
- `z_research/OcclusionStats/README.md` · `z_research/OcclusionStats/Archive/INTPHYS1_OCCLUSION_2026-09-26.md` — §2-1 · §2-2
- `auto_research/Archive/INTPHYS2_ONSET_2026-09-30.md` · `INTPHYS2_ENCODER_LINEAR_PROBE_2026-09-30.md`

EK100
- `z_research/anticipation/EK100/README.md` — §2-A · §2-B · §A-1 · 단서 목록 · §8 정정표
- `auto_research/Archive/SC3_EK100_PREDICTOR_SUBSTITUTION_2026-09-25.md` · `SC5_EK100_HORIZON_SWEEP_2026-09-26.md`
- `auto_research/Archive/ENCODER_MOTION_READOUT_2026-09-29.md` · `VERIFY_R3_2026-09-29.md` B
- `auto_research/Archive/AR_PREDICTOR_2026-09-27.md` — §2-9 (10-01 정정)
- `auto_research/Archive/I_SCENE_CAUSAL_BOTTLENECK_2026-09-25.md` — §4-2 · §4-3

Figure 1 · 문헌
- `z_research/v11_realistic/README.md` — §1 · §2 · §4
- `auto_research/Archive/LITERATURE_POSITIONING_2026-09-25.md` · `LITERATURE_STATE_WORLD_MODELS_2026-09-26.md` · `LITERATURE_PREDICTOR_ROLE_2026-09-29.md`

원재료 추출문: `scratchpad/archive10/extracts/{G11_benchmarks_realistic, G01_auto_H, G03_auto_late, G03b_auto_late_rest, G02b_auto_P_SC, G02a_auto_scene_M, G07_v11timing_v10_wma, G10c_ztraining_pft, G04a_literature, G13_ac_overview_misc}.md`
