# 결과 종합 — VOBench_v1 · IntPhys 1 · IntPhys 2 · AR 분석 · post-FT (2026-10-10 ~ 10-12)

> 이 문서는 `z_research/VOBench_v1/` · `cvpr/results/` · `z_training/runs/vobench_ar_state_postft/` 의 결과만 모은다 (`z_research/previous/` 는 인용하지 않는다).
> 수치는 각 폴더의 `tables.json` · `summary.json` · `compare.json` · `readout.json` · `horizon.json` 에서 옮겼다. 원 문서 링크는 절마다 붙인다.
> 표준 채점 = `surprise_c16t32` (stride 3 · 문맥 16 장 → 예측 16 장 · latent L1 · matched pair, chance 50). **복사 기준선** = 마지막 문맥 latent 를 미래 전부에 붙여 같은 규칙으로 채점한 값 (예측 없음).
> 갱신 기록: 10-12 첫 판 → 같은 날 §3 상세 (튜블릿 probe · 시간 일반화 · gravity_horizon) · §2-4 수치 잡음 · §4-3 문맥 길이 · §5 운동별 · **§6 예측 길이 (c16t24)** · **§7 gravity 전체** 추가 → 10-13 §3 target encoder 중간층 probe · **ar_state 를 vast ep70 로 교체** (§2-1 · §2-2 · §6 에 행 추가).

---

## 0. 한 쪽 요약

1. **V-JEPA 2 (release) 는 VOBench 에서 복사 기준선을 넘지 못한다** (primitive 71.7 vs 73.8 · realistic 81.9 vs 81.9). 바꿈이 보이는 조건은 예측 없이 풀리고, 복사로 안 풀리는 **late (문맥 끝 가림) 에서 57.9 / 65.7** — 움직이는 물체의 late permanence 는 사라짐 0 % / 나타남 100 %.
2. **정보는 encoder 에 있다** — 영상 전체를 T×N 평균한 encoder 표현의 linear probe 가 VOBench 위반을 99.7 % (late 칸도 100) 로 가른다. IntPhys 1 98.9 · **IntPhys 2 는 74** (encoder 수준에서도 절반만).
3. **Ariel AR (관측만으로 학습한 자기회귀 predictor) 은 late 의 "존재" 를 푼다** (late permanence 97–100, 복사 대비 late 전체 +20 pt, IntPhys 1 복사 대비 +8–9 pt) — 그러나 **정체 (모양 · 색) 와 자리는 못 이어 간다**, IntPhys 2 는 chance (51–53). 상태 되먹임판 (ar_state) 은 세 벤치 모두 ar 와 구분되지 않는다.
4. **AR 미래를 자 (readout) 로 읽으면**: 처음 2–3 걸음은 문맥 끝 속도를 그대로 잇지만 가속도는 반영하지 않고, **속도가 '마지막 관측 뒤 몇 번째 예측 블록인가' 에 따라 0 으로 감쇠한다** (속도 · 문맥 길이와 무관하게 4–5 번째 예측 블록에서 절반 근처). 정체도 뒤 걸음에서 약해진다. late 에서는 "무언가 있다" 는 남지만 무엇인지는 거의 모르고, 마지막 관측 자리에 머문다.
5. **VOBench 가능 영상으로 post-FT 하면 같은 분포 held-out block 에서 오른다** (83.5 → 90.3, late 이동 모양+색 60.8 → 75.8, p < 0.002) — 단 같은 물체 · 장면 분포 안이고, IntPhys 1 in-loop 는 83 → 78. 색이 가장 덜 오른다.
6. **예측을 8 프레임으로 줄여도 AR 의 late 이동 모양 · 색은 거의 안 오른다** (ar 59 → 65 · ar_state 61 → 63) — 반면 release 는 62.5 → 75.0 으로 AR 을 넘는다. AR 의 late 정체 실패는 "길게 예측해서 운동이 감쇠" 가 아니라 **다시 나타난 물체의 정체를 처음부터 못 이어 가는 것**이다.

---

## 1. 무엇을 썼나

| 항목 | 내용 |
|---|---|
| VOBench_v1 | UE 5.4 TestBed, 720 block × 4 arm = 2,880 clip (matched 1,440 쌍). objset (primitive · realistic) × 위반 (permanence · shape · color) × 운동 6 × 배경 2 × 가림 (visible · early · mid · late × k 1–3). **late 만 A↔B 바꿈을 가린다**. 문맥 무결성 1,440 쌍 × 16 장 mismatch 0 |
| 모델 | V-JEPA 2 ViT-H 릴리즈 · DINO-Foresight · VideoMAEv2-g · Ariel AR ep40 (`ar_future_1e_5`) · Ariel AR-state ep40 (`ar_state_future_1e_5`, 상태 384 되먹임) · AR-state + VOBench post-FT |
| AR 채점 | 블록별 LN(target_encoder) 문맥 → rollout, 타깃도 블록별 (`analysis/predictors/ar_scoring.py`). 릴리즈와 타깃 공간이 달라 **쌍 정확도만** 비교, 복사도 같은 공간 (`copy_blk` · `ar_baseline=copy_last`) |
| 하네스 정비 (10-11) | main 채점기에 `ar_state` 통합 (`kv_rollout` · `ar_baseline` 노브), 기존 AR 비트 단위 재현 128/128 · KV vs 전체 재계산 판정 뒤집힘 0 |

---

## 2. VoE 점수

### 2-1. VOBench (`*/surprise/tables.json`)

| 모델 | primitive overall | realistic overall | primitive late | realistic late | late permanence (prim / real) |
|---|---:|---:|---:|---:|---|
| **V-JEPA 2 release** | **71.7** | **81.9** | **57.9** | **65.7** | 55.6 / 61.1 (A 19–22 · B 92–100) |
| 복사 (release 공간) | 73.8 | 81.9 | 56.5 | 61.1 | 61.1 / 61.1 |
| DINO-Foresight | 84.9 | 84.7 | — | — | 복사 +2–3, late 는 release 와 같은 실패 |
| VideoMAEv2-g | 51.5 | 52.5 | — | — | 전부 chance |
| **Ariel AR ep40** | 81.9 | 86.4 | 75.0 | 78.2 | **98.6 / 100** |
| **Ariel AR-state ep40** | 81.5 | 85.4 | 74.5 | 77.8 | 97.2 / 95.8 |
| **AR-state vast ep70** (10-13, 이후 ar_state 기준) | 81.5 | 85.3 | 73.6 | 77.3 | — |
| 복사 (AR 공간) | 77.6 | 81.4 | 55.1 | 55.6 | 55.6 / 54.2 |

- release: late 는 k = 1–3 에 평평 (permanence primitive 54 / 54 / 58 · realistic 62 / 62 / 58). 보이는 칸 (visible · early · mid) 은 복사로도 풀리고, early 의 몇 칸은 모델이 복사보다 낮다 (primitive shape −12.5 · realistic permanence −6.9).
- late 에서 복사 대비: release +1.4 / +4.6 (CI 0 포함) · **AR +19.9 / +22.7** · AR-state +19.4 / +22.2 — 거의 전부 permanence (+42–46). shape · color 는 +6–13.
- AR · AR-state 쌍 단위 차이: primitive 11 vs 8 (p 0.65) · realistic 10 vs 3 (p 0.09) — 구분되지 않는다.
- 정지 vs 이동: AR 은 **정지 물체는 모양 · 색 92–100** (가림 시점 무관), **이동 물체는 보일 때도 72–85 · late 56–63** — 존재는 운동을 따라가지만 정체는 못 따라간다.

### 2-2. IntPhys 1 dev (Garrido A.8 · Filtered macro, 180 쌍; `cvpr/results/eval/intphys1/`)

| | 최고 칸 | 정확도 | 복사 | Δ (복사 대비) |
|---|---|---:|---:|---|
| release | skip2_w32 | 88.9 | 85.0 (release 공간) | +3.9 [−0.6, +8.9] — **구분 안 됨** |
| AR ep40 | skip2_w16 | 92.8 | 83.3 (AR 공간) | **+9.4 [+5.0, +13.9]** |
| AR-state ep40 | skip2_w16 | 91.1 | 83.3 | **+7.8 [+3.9, +12.2]** |
| AR-state vast ep70 (10-13) | skip2_w16 | 90.0 | 83.3 | +6.7 |

AR-state vs AR 쌍 단위 1 vs 4 (McNemar p 0.38). AR 의 이득은 O1 (사라짐) · O3 (연속성) 에서. release 는 skip2_w16 에서만 복사를 넘는다 (+9.4 [+3.3, +15.6]), moving / occluded 칸 75 (복사 70).

### 2-3. IntPhys 2 Main (506 쌍; `cvpr/results/eval/intphys2/_breakdown_ar_state_2026-10-12/`)

| | 최고 칸 | 전체 | camera Fixed / Moving (공통 C = 24) |
|---|---|---:|---|
| release | w48 C24 | 54.3 [51, 58] | 54.8 / 54.0 |
| 복사 (release 공간) | w48 C42 | 55.5 | 53.8 / 51.7 |
| AR ep40 | w16 C12 | 52.6 | 56.2 / 46.6 |
| AR-state ep40 | w16 C10 | 51.2 | 56.0 / 47.0 |

- 모두 chance 근처. release 가 높은 칸 (continuity / Moving 69) 은 **복사도 70.6** — 예측 없이 풀리는 장면. AR 이 나은 칸은 solidity 하나 (57.5 vs 53.4, CI 겹침).
- AR 은 카메라를 탄다 (Fixed 56 vs Moving 47) 그러나 Fixed 도 CI 하한 50. occluder 유무 · difficulty 는 일정한 경향 없음, game 16 개는 칸당 6–22 쌍이라 읽지 않는다.

### 2-4. 쌍 판정의 수치 잡음 (10-11)

- 쌍 상대차 r = (S_imp − S_pos) / S_pos 의 계산 잡음 (같은 입력 · 다른 배치) 은 95 분위 4–6e-4 · 최대 1.2–1.7e-3. 판정이 뒤집힌 쌍은 모두 |r| < 4e-4.
- 모델별 |r| 중앙값: release 0.46 % · DINO-F 2.1 % · AR 2.4 % · VideoMAEv2 0.12 % — 대부분의 쌍은 잡음보다 1–2 자리 크다. 쌍 하나의 통계적 문턱은 렌더 노이즈 시드를 바꾼 영상 (null) 이 있어야 정할 수 있다 (없음).
- binomial 부호 검정 문턱 (쌍 수 → 유의 정답 수): 6 → 6 · 12 → 10 · 24 → 17 · 72 → 44 · 144 → 83. 4 쌍 칸은 절대 유의하지 않다.

---

### 2-5. ar_state 교체 — vast ep70 (10-13)

`z_training/vast/ar_state/latest.pt` (vast 학습 70 epoch · step 20,580, K710 + SSv2 + HowTo100M fps 4 가변 창 · TF + rollout 16). arch 는 `kind: ar` + `ar_feedback: state` — 로더가 ar_state 로 읽는다 (vast 코드와 수치 일치, 상대 8e-7). 레지스트리 `vith_vast_ar_state_ep70`.
- VOBench 1,440 쌍: primitive 81.5 / realistic 85.3 (ariel ep40 판 81.5 / 85.4) — 쌍 단위 ariel 만 맞춤 23 vs vast 만 22 (p 1.0), **구분되지 않는다**.
- IntPhys 1: 최고 칸 skip2_w16 90.0 (ariel 판 91.1 · ar ep40 92.8 · AR 공간 복사 83.3).
- IntPhys 2 · VOBench post-FT 는 vast 판으로 아직 안 돌렸다.

## 3. late 실패의 성격 (release)

| 검사 | 결과 | 출처 |
|---|---|---|
| **asymmetry** — 창을 밀어 가림이 문맥 끝에서 끝나게 (가능 미래는 첫 샘플부터 물체가 보임, 창 16:12) | late 바뀌지 않음 (56.5 → 56.9 · 66.7 → 65.7), 이동 permanence 0 / 100 그대로 → 실패는 "가림 너머 예측" 이 아니라 **마지막 관측에 물체가 안 보여서** | `vjepa2/surprise_asymmetry/` |
| **encoder VoE probe** (영상 32 장 → context · target encoder T×N 평균 → linear) | 쌍 정확도 VOBench 99.7 / 99.9 (24 칸 모두 96–100, late 포함) · IntPhys 1 98.9 / 96.7 · **IntPhys 2 74.1 / 72.5** (solidity 63 / 57, Moving 71 / 69) | `vjepa2/encoder_voe_probe/` |
| gravity (ledge · arc) | ledge 낙하 0 / 0 · arc 는 진짜 운동을 몇 샘플 잇다 **공중에서 멈춤** — 상세 **§7** | `vjepa2/gravity/` |
| gravity_horizon — 예측 16 · 8 · 4 · 2 샘플 | 같은 3–4 번째 샘플이 예측 16 · 8 에서는 진짜 운동 100 % 인데 예측 4 에서 창의 끝이 되면 떨어진다 — **긴 예측이 아니라 '창의 끝' 에서 정지 쪽으로** (전체 토큰). 공 칸만 보면 창 끝 효과 없음 | `vjepa2/gravity_horizon/` |

**튜블릿 정체 probe** (`vjepa2/probing/`, 결합 (모양/색) + none, attentive depth 1 · query 1, objset 따로, 4 run 범위)

| 정답 정체 % (none 이라 답한 %) | primitive | realistic |
|---|---|---|
| ① h 보임 (천장) | 94–97 (0) | 89–91 (0) |
| ② h 가림 튜블릿 | 9–15 (24–39) | 13–21 (9–21) |
| ③ p 보임 — 첫 미래 t8 | **83–85 (0)** | **86–88 (0)** |
| ③ p 보임 — 마지막 t15 | 13–15 (58–66) | 38–56 (29–35) |
| ④ p late (재등장) | 0–1 (47–94) | 0–5 (40–96) |

- 가까운 미래에서는 h head 가 p 에 통한다 (t8 83–88). p 로 학습한 head (p_only) 도 late p 는 0–8 % (정지 realistic 19 %) — late 는 정보 자체가 없는 쪽, 보일 때의 감쇠와는 다른 실패. ~~"멀어지며 떨어지는 것은 정보 소실"~~ 은 p_only 결과로 **철회** (보일 때의 감쇠는 정렬 · 형식 문제).
- ⚠️ 가림막 판이 head 에 물체처럼 읽힌다 — ② · ④ 의 none 비율은 시드에 따라 흔들린다. 안정적인 것은 "정답 정체 ≈ 0" 까지.

**target encoder 중간층 (16 / 32) vs 마지막 층** (10-13, `vjepa2/probing/results/L16_e35_s{0,1}/` · 그림 `vjepa2/probing/figures/layers/fig_probe_layers`)

| h 정체 probe, 가림 튜블릿 (시드 0 / 1) | 정체 | none | 다른 물체 |
|---|---|---|---|
| 마지막 층 · primitive / realistic | 9.3 / 14.5 · 13.5 / 12.5 | 37 / 27 · 21 / 9 | 54 / 58 · 65 / 79 |
| **16 층** · primitive / realistic | **0.7 / 0.5 · 0.0 / 0.7** | **71 / 74 · 100 / 85** | 28 / 25 · 0 / 14 |

- 중간층은 가려진 튜블릿을 '물체 없음' 으로 읽는다 (그 시점에 보이는 것) — 가려진 물체의 정체가 조금이라도 들어오는 것은 뒤쪽 층 (시간 attention 이 다른 프레임을 섞은 뒤) 에서이고, 그것도 약하다 (9–15 %, chance 2 %). 보일 때는 16 층이 같거나 약간 높다 (99 vs 97 · 91 vs 90).
- 단서: 마지막 층의 '다른 물체' 에는 가림막 판 오독이 섞인다. head 는 층마다 따로 학습.

**시간 일반화** (`vjepa2/probing/temporal_generalization/`, 한 시점 학습 → 미래 t8–t15, 3 시드, 모양 정답 %)

| | 같은 시점 | 앞으로 4 칸 이상 | 뒤로 4 칸 이상 |
|---|---:|---:|---:|
| p → p, primitive | 94.5 | **45.7** | 78.0 |
| p → p, realistic | 98.2 | **55.0** | 74.8 |
| h → h | 대칭 (물체가 움직여도 평평) | | |

- p 의 표현 형식은 시점마다 달라지고, 앞 시점 head 가 먼 미래에 덜 통한다 (뒤로는 더 통한다). late p 에서 t15 head 는 모양을 다수 기준선의 약 2 배 (41 · 38 vs 20) 로 읽는다 — 1 차 결과.
- 단서: 시점 하나 학습 샘플 약 270, realistic 색은 79 % 가 `native` 라 다수 기준선에 묻힌다. mean-pooled DiM 개념 벡터는 희석 · 표본 부족으로 무효.

---

## 4. AR 분석 — 무엇을 이어 가고 무엇을 못 하나

### 4-1. 토큰 Δ (파라미터 없음; `ariel_ar_ep40/surprise/TRACKING_2026-10-12.md` · GIF `figures/token_surprise_vs_vjepa2/`)

- late · 이동 permanence: 물체 자리 Δ > 0 인 쌍 **0 %**, 배경 Δ > 0 **100 %** — 이기는 곳이 물체 자리가 아니라 배경 (**presence without placement**). 정지 물체는 물체 자리에서 이긴다 (Δ 크게 양수).
- shape · color 쌍은 배경 차가 거의 없어 정답 자리에 맞게 놓아야만 이긴다 — 그래서 존재는 풀고 정체는 못 푼다.
- late · 이동: 마지막 관측 자리에 빈 문맥 대비 초과 오차 **+19 → +55** (걸음마다 커짐) — 물체 같은 것을 옮기지 않고 남긴다. release 는 이 잔상이 없다.
- 물체 정보의 위치 (encoder): 짝 차이 |h_a − h_b| 는 물체 토큰에서 가장 크지만, 물체에서 8 토큰 이상 떨어진 고리만으로도 모양이 선형으로 82 % (primitive) · 95 % (realistic) 읽힌다 — 전역 사본이 있으나 L1 에서는 작다.
- ⚠️ 정정: Δ 의 공간 분포 (몫 · profile · 질량중심) 는 정답 자리에 구조적으로 고정되어 **위치 증거가 아니다** (같은 날 시도한 번짐 · 앞섬 분석 철회, 다시 시도하지 말 것).

### 4-1b. 물체 자리 토큰만으로 채점 (보조, 10-13; `vjepa2/surprise_object_region/`)

실제 물체 상자 (×1.1) 와 겹치는 토큰 (평균 7 %) 만으로 쌍 채점 (release · AR ep40, token_surprise 다시 셈).
- 보일 때 모양 · 색은 오른다 (release primitive 63–75 → 90–96) — 전체 토큰에서는 배경이 신호를 묽힌다.
- **late 는 두 모델 모두 chance 이하** (이동 모양+색 48–53).
- **AR 의 late permanence 우위는 물체 자리에서 사라진다** (98.6 → 56.9, A 방향 100 → 16.7 = release 와 같음) — AR 이 이긴 곳은 배경 (presence without placement 의 독립 확인). 예외: AR late 정지 물체는 물체 자리에서도 95.8 / 79.2.
- 단서: 보조 분석 (주 결론은 전체 토큰), 상자 크기 민감도 미확인.

### 4-2. 위치 + 정체 자 (`readout/README.md`)

자 = VOB_readout_primitive (VOBench 장면 · 가림 없음 · 직선 등속 150 / 300 cm/s, 8,820 clip) 에서 AR p 로 학습한 attentive head (위치 + 49 클래스). held-out: p 정체 95.2 · 중심 1.00 칸 / h (천장) 98.6 · 0.41 칸. VOBench h 에서도 위치 0.2–0.6 칸 · 정체 93–100 · 빈 문맥 '있다' 0 %. h 자를 AR p 에 걸면 t10 부터 '없음' 으로 무너진다 (p 와 h 분포가 걸음마다 벌어짐) — 교차 읽기로 못 쓴다.

| VOBench primitive (AR ep40) | 결과 |
|---|---|
| 보일 때 · 이동 | '있다' 중 정체 98 → 70 (early, t15) · 95 → 49 (mid) · 위치는 걸음마다 뒤처짐 (t15 등속 −2.3 칸 · 빨라짐 −3.8 칸, 등속 외삽보다 더 뒤) |
| late · 이동 | '있다' 81 → 55 % (빈 문맥 대조 57 → 17) — **무언가 있다는 남음**, '있다' 중 정체 13–15 % (chance 2) — **무엇인지 거의 모름**, 위치는 마지막 관측 자리 0–1 칸에 머묾 (attention 퍼짐 단서) |
| late · 정지 | 정체 30 → 53 % 로 부분 회복 |
| 시간축 그림 | `readout/vobench_ar40/figures/fig_readout_timeline` — h (파랑, t0–t15) 는 가림 칸에서만 정체↓ · 없음↑, AR p (주황, t8–t15) 는 뒤 걸음에서 정체↓ · 없음↑ (mid 는 재등장 뒤 관측이 짧아 early 보다 일찍 무너짐) |

release (p_only probe) 와 같은 꼴 비교: late 정체 0–5 % vs AR 7–12 %, 보일 때 70–76 → 47–59 vs 98 → 65–70 — 단 head 가 달라 (VOBench 자체 표본 vs VOB_readout 자) 직접 비교는 같은 자로 다시 해야 한다 (미실행).

### 4-3. 운동 정보 — 속도는 예측 블록 순서에 따라 감쇠 (자 기반)

| VOBench 보일 때 | t8→9 | t10→11 | t12→13 | t14→15 |
|---|---:|---:|---:|---:|
| 등속: 정답 / AR 속도 (칸/튜블릿) | 0.77 / 0.76 | 0.77 / 0.63 | 0.77 / 0.43 | 0.77 / −0.15 |
| 빨라짐: 정답 / AR | 0.87 / 0.86 | 1.02 / 0.91 | 1.18 / 0.55 | 1.34 / −0.03 |
| 느려짐: 정답 / AR | 0.72 / **0.86** | 0.56 / 0.60 | 0.40 / −0.13 | 0.24 / −0.59 |

정답 속도 = `label.json` 물체 화면 좌표 (512 px) 를 튜블릿 (2 장 = 0.2 초) 평균해 운동 방향으로 투영한 이웃 차 ÷ 32 px (토큰 1 칸). 0.77 칸/튜블릿 ≈ 원본 프레임당 4 px.

- 처음 2–3 걸음 = **문맥 끝 속도의 복사** (가속 · 감속 미반영), 그 뒤 0 으로 감쇠. 운동 (속도) 이 정적 정보 (정체) 보다 먼저 흐려진다 (t10–11 vs t12 이후).
- **속도별** (자 학습셋 test, 150 / 237 / 300 cm/s): 절반이 되는 시점이 4–6 번째 예측 블록으로 비슷하고 그때 이동 거리는 1.2 / 2.4 / 3.0–3.6 칸 — 거리가 아니라 **예측 블록 순서**에 묶인 감쇠. (release 는 09-19 분석에서 거리 쪽이었다 — 같은 방법으로 아직 안 봤다.)
- **문맥 길이별** (C = 4 · 6 · 8 · 10 · 12 블록, `readout/horizon_ar40/`): 예측 블록 순서 k 로 줄 세우면 C 와 무관하게 같은 곡선 (300 cm/s: k1 0.94–1.01 · k3 0.83–0.90 · k4 0.70–0.74 · k5 0.55–0.62). 영상의 절대 시점으로 줄 세우면 C 만큼 밀린다 — 감쇠는 **마지막 관측 뒤 몇 번째 예측 블록인가** 에 묶이고, 문맥을 3 배로 늘려도 · 학습 범위 밖 K = 12 로 늘려도 바뀌지 않는다.
- 단서: 자는 AR p 로 학습해 AR 의 일관된 오차를 가린다 (실제 감쇠는 더 클 수 있다). 뒤 걸음의 음수 속도는 자 기본값일 수 있다. 가림 없음 · 직선 영역.

### 4-4. gravity (AR; `ariel_ar_ep40/gravity/`, 상세 §7-4)

AR 은 release 처럼 공중에서 멈추지 않는다 (멈춤 후보 0 / 72). 그러나 고르는 미래는 **등속 · 높이 유지 규칙**의 후보 — 오르는 중 arc 83 / 75 (등속으로 이어도 arc 가 가장 가깝다), 꼭대기 float · rise, 내려가는 중 rise · float, **진짜 포물선 0 / 24** (내려가는 중). ledge 9.7 / 0.0 (release 0 / 0) — 낙하 미래를 만들지 않는다.

---

## 5. VOBench post-FT (AR-state; `ariel_ar_state_ep40/postft_vobench/`)

학습 = VOBench block split train 절반의 가능 영상 720 clip, 문맥 8 → 예측 8 블록, lr 5e-5, 30 epoch (30 분, 24 GB × 4). 채점 = 나머지 360 block (720 쌍). 학습 코드 = ariel 브랜치 학습 경로를 main 에 3-way 병합.

| held-out test block | release | AR ep40 | AR-state | **AR-state + FT** | 복사 (AR) |
|---|---:|---:|---:|---:|---:|
| primitive | 70.3 | 82.5 | 81.9 | **88.9** | 77.2 |
| realistic | 82.2 | 86.4 | 85.0 | **91.7** | 81.9 |
| late 이동 모양+색 (두 objset) | — | — | 60.8 | **75.8** (p 0.0014) | — |

쌍 단위 (전 → 후): 전체 83.5 → 90.3 (10 vs 59, p < 1e-4) · late 76.4 → 86.6 (6 vs 28, p 0.0002) · 보일 때 이동 모양+색 75.7 → 85.4 (4 vs 31).

**운동별 (두 objset 합침, 쌍 수 12–56)**

| 위반 · 가림 | 정지 | 등속 | 빨라짐 | 느려짐 |
|---|---|---|---|---|
| 모양 · 보임 (FT 전 → 후) | 100 → 100 | 85.7 → **96.4** | 83.9 → **94.6** | 78.6 → 83.9 |
| 모양 · late | 91.7 → 100 | 50.0 → **83.3** | 62.5 → **87.5** | 54.2 → 66.7 |
| 색 · 보임 | 100 → 100 | 78.6 → 89.3 | 67.9 → 83.9 | 66.1 → 71.4 |
| 색 · late | 100 → 100 | 75.0 → 83.3 | 62.5 → 75.0 | 62.5 → 66.7 |

- 모양은 late +17–22 pt, **색은 +6–8 pt** — 색은 보일 때도 복사보다 낮다 (primitive visible 83 vs 92). **느려지는 물체가 가장 덜 오른다** (모양 late 66.7 · 색 보임 71.4).
- 그림 `postft_vobench/figures/fig_postft_compare` (막대 = 쌍 정확도 [block 부트스트랩 95 %], 점 = release, 선 = AR 공간 복사).
- 단서: train · test 가 물체 · 장면 · 운동 설계를 공유 (block 만 다름). in-loop IntPhys 1 (60 쌍) 83.3 → 78.3. 점수 상승이 정체 회복인지 자리 회복인지는 post-FT p 용 자로 아직 안 읽었다.

---

## 6. 예측 길이를 줄이면 — 문맥 16 · 예측 8 프레임 (c16t24, 10-12)

**묻는 것**: AR 이 4–5 번째 예측 블록부터 운동을 잃는다면 (§4-3), 예측을 4 블록 (8 프레임) 으로 줄이면 late 의 정체 판정이 오르는가. 같은 360 test block (720 쌍), 창 `c16t24` (raw 0–69). 원시 `exp_results/*_wc16t24` · post-FT `z_training/runs/vobench_ar_state_postft/eval/surprise_c16t24__*` · 표 `postft_vobench/c16t24/compare.{md,json}`.

| 쌍 정확도 % (예측 16 → 예측 8) | 쌍 | release | AR ep40 | AR-state | AR-state + FT |
|---|---:|---|---|---|---|
| 전체 | 720 | 76.2 → **82.6** | 84.4 → 87.4 | 83.5 → 86.1 | 90.3 → 91.1 |
| 보임 (visible · early · mid) | 504 | 82.5 → 88.7 | 87.9 → 91.5 | 86.5 → 89.9 | 91.9 → 93.8 |
| late 전체 | 216 | 61.6 → 68.5 | 76.4 → 77.8 | 76.4 → 77.3 | 86.6 → 84.7 |
| late · permanence | 72 | 59.7 → 58.3 | 98.6 → 97.2 | 95.8 → 97.2 | 100 → 100 |
| late · 모양 | 72 | 55.6 → **68.1** | 61.1 → 63.9 | 62.5 → 65.3 | 81.9 → 79.2 |
| late · 색 | 72 | 69.4 → 79.2 | 69.4 → 72.2 | 70.8 → 69.4 | 77.8 → 75.0 |
| **late · 모양+색 · 이동** | 120 | 62.5 → **75.0** | 59.2 → 65.0 | 60.8 → 62.5 | 75.8 → 72.5 |
| 보임 · 모양+색 · 이동 | 280 | 77.9 → 85.4 | 78.2 → 84.6 | 75.7 → 81.8 | 85.4 → 88.9 |

- **예측을 짧게 하면 모두 조금 오르지만, release 가 가장 많이 오른다** (전체 +6.4 · late 이동 모양+색 +12.5 → AR 을 넘음).
- **AR 의 late 이동 모양 · 색은 짧게 해도 거의 안 오른다** (+2–6 pt). AR 은 4 블록 안에서는 속도를 85 % 이상 유지하므로 (§4-3), **late 정체 실패의 원인은 긴 예측의 운동 감쇠가 아니다** — 다시 나타난 물체의 정체를 처음부터 못 이어 간다 (§4-2 의 late 정체 13–15 % 와 같은 그림).
- AR 의 우위는 여전히 permanence (존재) 뿐 (97 vs release 58). post-FT 는 이미 높아 짧게 해도 거의 같다 (late 86.6 → 84.7).
- **vast ar_state ep70 (10-13) 도 같다** — test block 예측 16 → 8: 전체 83.3 → 85.1 · 보임 87.1 → 90.7 · **late 74.5 → 72.2** (permanence 93.1 → 90.3 · 모양 59.7 → 59.7 · 색 70.8 → 66.7 · 모양+색 이동 60.0 → 60.0). late 는 오히려 조금 내려간다.
- ✅ 쓸 수 있는 문장: "예측을 8 프레임으로 줄여도 AR 은 late 에 다시 나타난 움직이는 물체의 모양 · 색을 맞히지 못한다 (59–61 → 63–65) — 실패는 예측 길이가 아니다."
- ⚠️ 단서: late 는 예측 구간 첫 2–6 프레임이 아직 가려져 있어 (k = 1–3) 예측 8 프레임 창에서는 재등장 프레임이 2–6 장뿐이다 — release 의 상승에도 이 구성 차이가 섞인다. 칸 72–120 쌍 → CI ±8–10 pt. 두 창은 문맥 프레임이 같고 미래 길이만 다르다.

---

## 7. gravity — 꼭대기에서 되돌아옴 · 낙하 (3a 추세 · 3b 사건) (10-11)

원 문서: `vjepa2/gravity/` (release) · `vjepa2/gravity_horizon/` (예측 길이) · `ariel_ar_ep40/gravity/` (AR) · VOBench README §3-12–3-14.

### 7-1. 데이터 · 채점

- `/data2/local_datasets/world/Benchmark/VOBench_v1/gravity` — 216 block / 936 arm, g 375 cm/s² (10-11 재렌더판). **block = 과거 하나 + 미래 여럿**, 모든 arm 이 f0–47 을 공유 (감사 720 arm × 48 장 mismatch 0).
- **ledge** 144 block: `pos` (선반 끝에서 낙하) vs `imp_float` (같은 높이로 직진). 셀 = 높이 z300 · z400 × 속도 200 · 260 · 320 cm/s.
- **arc** 72 block, 미래 9: `pos` (포물선) · `rise` (계속 올라감) · `float` · `line` · `stop` (f45 자리 멈춤) · `stop2/4/8/12` (진짜 운동을 f50/52/56/60 까지 따르다 공중에서 멈춤).
  문맥 끝 상태 셀 = **오르는 중** (`apex66`) · **꼭대기** (`apex48`) · **내려가는 중** (`apex30`). objset × 셀마다 12 block.
- 채점 = 표준 `surprise_c16t32` + `scoring.pairing=cross`. arc 는 **고른 미래 = argmin_k S_k** ("p 가 가장 가까운 미래", chance 11 %).

### 7-2. release

**ledge**: 144 쌍 전부 낙하보다 같은 높이 직진이 덜 놀랍다 — 모델 **0.0 / 0.0** (primitive / realistic), 상대차 중앙 −1.4 % / −2.4 %, 높을수록 · 느릴수록 더 틀린다. ⚠️ **복사도 0 %** — 떠 가는 미래는 마지막 문맥과 닮아, 이 쌍만으로는 "낙하를 만들지 않는다" 와 "덜 변하는 미래가 L1 로 싸다" 를 못 가른다.

**arc — 고른 미래 %**

| | arc (진짜) | rise | float · line | stop · stop2 · stop4 | stop8 | stop12 |
|---|---:|---:|---:|---:|---:|---:|
| primitive 모델 (전체 토큰) | **0** | 0 | 0 | 0 | 28 | **72** |
| realistic 모델 | **0** | 3 | 0 | 0 | 25 | **72** |
| primitive 복사 | 0 | 17 | 36 | 39 | 3 | 6 |
| realistic 복사 | 0 | 25 | 67 | 6 | 0 | 3 |

공 칸만 세도 같다 (stop8 · stop12 = 25–28 · 72–75). 셀 6 칸 모두 stop8 + stop12 = 92–100 %, 진짜 포물선 72 block 중 0.

**튜블릿 묶음** (p 가 가장 가까운 후보 %, primitive / realistic; 앞 = t0–t2 · 뒤 = t3–t7)

| 문맥 | 앞: arc | 앞: rise | 앞: float · line | 앞: 멈춤 | 뒤: arc | 뒤: 멈춤 |
|---|---:|---:|---:|---:|---:|---:|
| 오르는 중 | 28 / 42 | 8 / 3 | 8 / 8 | 56 / 47 | 2 / 3 | 75 / 83 |
| **꼭대기** | **3 / 3** | **31 / 69** | 28 / 19 | 39 / 8 | 3 / 0 | 68 / 73 |
| 내려가는 중 | 36 / 53 | 0 / 0 | 22 / 11 | 42 / 36 | 3 / 0 | 77 / 85 |

- **p 는 진짜 포물선을 끝까지 잇지 않는다** — 진짜 운동을 문맥 끝 뒤 11–15 프레임 따르다 **공중에서 멈춘** 미래가 가장 가깝다 (문맥 상태 무관). 뒤 튜블릿은 멈춤 68–85 %.
- **꼭대기 문맥은 앞 튜블릿부터 갈린다** — 되돌아 내려오는 진짜 대신 계속 올라가는 (rise 31 / 69) · 수평 미래 쪽.
- 위치 규칙 (등속 · 높이 유지 · 그 자리 멈춤) 어느 것과도 다르다 — 오르는 중은 등속만 이어도 진짜 포물선이 가장 가까운데 p 는 0 / 24. 복사는 rise · float · stop 을 고른다 (stop8 · 12 는 3–6 %) — 복사로는 이 고르기가 안 나온다.
- 쌍 정확도로 읽으면 vs float 17 % · vs rise 28–39 % · vs stop8 · 12 0 %.

### 7-3. 예측 길이 (gravity_horizon, release)

문맥 16 샘플 고정, 예측 M = 16 · 8 · 4 · 2. 미래 튜블릿마다 p 가 **정지 (f45 자리)** 보다 **진짜 운동** 에 가까운 block % (전체 토큰).

| 진짜 운동에 가까운 % (primitive / realistic) | 3–4 번째 | 7–8 번째 | 마지막 튜블릿 |
|---|---|---|---|
| 예측 16 | 100 / 100 | 0–67 / 0–75 | **0 / 0** |
| 예측 8 | 100 / 100 | 0–17 / 0–25 | **0–17 / 0–25** |
| 예측 4 | 8–42 / 17–75 | — | **8–42 / 17–75** |
| 예측 2 | — | — | 42–100 / 83–100 |

- 같은 3–4 번째 샘플이 예측 16 · 8 에서는 100 % 인데 예측 4 에서 창의 끝이 되면 떨어진다 — 망가지는 곳은 절대 순번이 아니라 **창의 끝 (마지막 1–2 튜블릿)**. "긴 예측이라서" 가 아니다.
- **공 칸만 보면 창 끝 효과가 없다** — 공 자체는 문맥 끝 뒤 6–8 번째 (오르는 중 10–12 번째) 에서 진짜 운동을 떠난다. 창 끝 효과는 공이 아닌 토큰 (배경 · 그림자) 에서 온다 (기제 미검증).
- AR 의 감쇠 (§4-3) 와 비교: release 는 '창의 끝' (전체 토큰) · AR 은 '예측 블록 순서' — 측정이 달라 (고른 미래 vs 자) 같은 축으로 아직 안 놓았다.

### 7-4. AR ep40 vs release (같은 데이터 · 표준 창)

| 문맥 | release | **Ariel AR** | AR 공간 복사 | 위치 규칙: 등속 · 높이 유지 |
|---|---|---|---|---|
| 오르는 중 | 멈춤 100 / 100 | **arc 83 / 75** · float 17 / 0 · line 0 / 17 | arc 42 / 17 · rise 17 / 42 · float 33 / 17 | arc · rise |
| 꼭대기 | 멈춤 100 / 92 | **float 50 / 58 · rise 33 / 33** · arc 17 / 8 | rise 42 / 33 · float 42 / 67 | rise · float |
| 내려가는 중 | 멈춤 100 / 100 | **rise 58 / 58 · float 42 / 42** · arc **0 / 0** | rise 92 / 75 · float 8 / 25 | float · float |

- **AR 은 멈추지 않는다** (멈춤 후보 0 / 72) — release 의 "몇 샘플 가다 공중에서 멈춤" 이 AR 에는 없다.
- 그러나 **고르는 미래는 등속 · 높이 유지 규칙의 후보** — 오르는 중 arc (등속으로 이어도 arc 가 가장 가깝다), 꼭대기 · 내려가는 중 rise · float. **진짜 포물선은 내려가는 중 0 / 24 · 꼭대기 3 / 24.**
- 복사 (copy_blk) 도 꼭대기 · 내려가는 중에서 rise · float — AR 의 고르기는 마지막 문맥과 닮은 미래 쪽 (오르는 중만 복사보다 arc 를 더 고른다).
- ledge: AR 9.7 [3, 17] / 0.0 · copy_blk 12.5 / 1.4 (Δ CI 0 포함) — release 처럼 낙하보다 직진.

### 7-5. 쓸 수 있는 문장 · 못 쓰는 문장 · 단서

- ✅ (release) "꼭대기에서 되돌아옴 (3a) · 낙하 (3b) 를 따르는 미래를 만들지 않는다 — 가장 가까운 미래는 진짜 운동을 몇 샘플 이은 뒤 공중에서 멈춘 미래다 (문맥 상태 무관)"
- ✅ (release) "예측 길이를 16 → 8 → 4 로 줄여도 창의 끝에서 진짜 운동을 떠나 정지 쪽으로 간다 — 긴 예측이 만든 현상이 아니다 (전체 토큰)"
- ✅ (AR) "관측만으로 학습한 AR predictor 는 멈추지는 않지만, 그리는 미래는 문맥 끝 운동을 등속 · 높이 유지로 이은 것이다 — 되돌아옴 · 낙하를 따르는 미래는 만들지 않는다 (내려가는 중 진짜 포물선 0 / 24)"
- ❌ "중력을 모른다" (내부 귀속) · "ledge 에서 불가능한 연속을 예측한다" (복사도 0 %) · "p 는 15 프레임 뒤 멈춘다" (stop12 가 후보 중 가장 늦은 멈춤 — 상한일 뿐) · "예측이 짧으면 물리를 따른다" · "AR 은 오르는 중 포물선을 예측한다" (등속 규칙과 같은 후보) · "AR 이 release 보다 물리를 더 안다" (차이는 멈춤 유무)
- ⚠️ 칸마다 12 block · "가장 가까운 미래" 는 후보 9 개 안의 표면 측도 (위치 자로 확인 안 함) · 멈춤 지점을 재려면 stop16 · 20 · 24 같은 더 늦은 멈춤 후보가 필요 · 정지 대조군은 f45 자리 하나 (느려짐은 이분법으로 못 봄) · 창 끝 효과가 VOBench 본판 · v11 에도 있는지 미확인
- 옛 판 (10-10, g 981, 꼭대기 f36 · f42, sliding 6 창) 은 대체됨 — 같은 방향 (ledge 0 / 48 · arc stop8 · 12 71–76 % · 진짜 8–17 %), 섞어 읽지 말 것 (`vjepa2/gravity/_previous_20261010_g981/`).

## 8. 진행 중 · 미결

| 항목 | 상태 |
|---|---|
| AR 문맥 길이별 rollout (C = 4 · 6 · 8 · 10 · 12) | **완료** — 예측 블록 순서에 묶인 감쇠 (§4-3, `readout/horizon_ar40/`) |
| 예측 길이 c16t24 | **완료** (§6) |
| release 와 post-FT 모델을 같은 자 · 같은 속도 분석으로 | 미실행 (모델마다 자 학습 약 15 분) |
| post-FT 의 분포 밖 정식 채점 (IntPhys 1 · 2) | 미실행 |
| 색이 약한 이유 (위반이 만드는 latent 차이 크기 · 외형 흐려짐) | 미실행 |
| 파라미터 없는 위치 교차검증 (AR p ↔ 참 h 튜블릿 시간 맞추기) | 미실행 |
| EK100 action anticipation — AR ep40 전체 데이터 (vll1, job 220853) | 진행 중, 20 epoch 약 2.3 일 (10-11 시작, 첫 job 은 RAM OOM 으로 재제출) |
| IntPhys 2 의 AR 공간 복사 기준선 | 미실행 (엔진에 `ar_baseline` 미연결) |

## 9. 정정 · 철회 (같은 기간)

- 토큰 Δ 의 공간 분포로 "AR 이 물체를 어디 그리나" 를 읽은 것 (번짐 profile · 앞섬 질량중심) — **방법 결함, 철회** (Δ 는 h_pos ≠ h_imp 인 정답 자리에만 생긴다).
- "late 에서 AR 미래에 물체다운 것이 없다" — **정정**: 정체 head 의 '없음' 으로 보면 물체 특이적 '있다' 신호가 있고 정체만 거의 없다.
- "보일 때 정체는 유지된다" — **좁힘**: t12 까지. 뒤 걸음에서는 정체도 떨어진다.
- "멀어지며 떨어지는 것은 정보 소실" (튜블릿 probe) — **철회** (p_only 결과로 보일 때의 감쇠는 정렬 · 형식 문제).
- "AR 의 감쇠는 거리 한계" — 속도별 · 문맥 길이별 결과로 **예측 블록 순서** 쪽 (release 의 거리 결론과 같은 방법 비교는 미실행).
- 튜블릿 probe 의 realistic 첫 실행 (손실 2.6 고정, 전부 none) — 일회성 실패, `_invalid_run1_realistic_collapsed/` 로 보관.
- mean-pooled DiM 개념 벡터 — 희석 (물체 2–4 / 256 토큰) · 클래스당 약 5 개로 무효.

## 10. 파일 지도

| 무엇 | 어디 |
|---|---|
| VOBench 본문 (모든 절) | `z_research/VOBench_v1/README.md` |
| 모델별 채점표 · 그림 | `z_research/VOBench_v1/<vjepa2 · dino_foresight · ariel_ar_ep40 · ariel_ar_state_ep40 · videomae_v2>/surprise/` |
| release probing · encoder probe · asymmetry · gravity | `z_research/VOBench_v1/vjepa2/{probing, encoder_voe_probe, surprise_asymmetry, gravity, gravity_horizon}/` · AR gravity `z_research/VOBench_v1/ariel_ar_ep40/gravity/` |
| AR 토큰 Δ · 자 · 시간축 · GIF · 문맥 길이 | `z_research/VOBench_v1/ariel_ar_ep40/surprise/TRACKING_2026-10-12.md` · `z_research/VOBench_v1/readout/{identity_vobprim_ar40, vobench_ar40, horizon_ar40}/` |
| post-FT · 예측 길이 | `z_research/VOBench_v1/ariel_ar_state_ep40/postft_vobench/{README.md, compare.md, c16t24/, figures/}` · run `z_training/runs/vobench_ar_state_postft/` |
| IntPhys 1 · 2 결과 | `cvpr/results/eval/intphys{1,2}/` (IntPhys 2 분해 `_breakdown_ar_state_2026-10-12/`) |
| 주요 스크립트 | `z_research/scripts/analysis/{vobench_tables, vobench_track_delta, vobench_ar_readout, ar_horizon_readout, encoder_voe_probe, intphys2_breakdown, vobench_split_compare}.py` · `z_research/scripts/figures/{plot_vobench, plot_vobench_ar_readout, plot_vobench_postft}.py` · 학습 `z_training/data/build_vobench_split_index.py` · `configs/training/vobench_ar_state_postft.yaml` |

## 재현

각 절의 원 문서 `## 재현` 을 따른다. 이 문서의 표는 `compare.json` · `readout.json` · `horizon.json` · `tables.json` · `results.json` 에서 옮겼다 (§6 표는 `vobench_split_compare.py --window c16t24` 와 같은 쌍 · 같은 규칙으로 계산).
