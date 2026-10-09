# 스토리 ↔ 근거 지도 (2026-09-26)

> ⚠️ **대체됨 (2026-10-08).** 논문 스토리 정본은 [`PAPER_STORY_2026-10-08.md`](PAPER_STORY_2026-10-08.md) 다. 옛 스토리 (DIRECTION §2 · ABSTRACT v0/v1) 의 근거 지도다. 새 근거 지도는 정본 §2 (문단별) 와 §재현. 이 문서의 행별 수치 · 등급 · 빈틈 목록은 유효하다.
> 2026-09-29 논지 재정리 (RETHINK_PREDICTOR_ROLE): "예측의 자리가 비어 있다". 근거 표 추가 — predictor 의 몫 (PREDICTOR_SHARE: 복사 대비 +2 ~ +7 pt), encoder 충분성 표 (RETHINK §1-b), AR predictor 네 판 (AR_PREDICTOR §2). 초록 v1.
> 2026-09-26 저녁 추가 (1 차): 질의 위치 평행이동 (I-scene §4-6) · 연속 판독 (§4-7: 실영상은 거리 + 시간) · release 팬 증강 M (§5-4). 적대 검증 2 차 완료 (`Archive/VERIFY_ISCENE2_2026-09-26.md`): permanence · closure · M · SC5 확정, qshift · 연속 판독 · 팬 epoch 포화는 시사로 강등, IntPhys 2 "복사 ≥" 는 "구분 안 됨" 으로 정정.
> 2026-09-26 오후 추가 (1 차): permanence 실영상 장면 판 (I-scene §4-4) · closure 실영상 장면 판 (§4-5) · Ariel epoch 포화 (§5-3) · SC5 EK100 1/2/3 s Δ ≈ +2 (SC5 §3-1). 빈틈 2-1 의 "실영상 permanence · closure" 는 채워졌고, 2-3 의 "학습 부족" 반론은 포화로 1 차 해소.
> 2026-09-26 추가: IntPhys 2 permanence 감사 — 복사 55.5 vs release 54.3 (overall), permanence Fixed 63.5 vs 59.6; release permanence 전 조건 CI 가 50 포함 → 평가 감사 표에 IntPhys 2 행 추가 (`Archive/INTPHYS2_PERMANENCE_AUDIT_2026-09-26.md`, 1 차). 문헌 긴장 (IntPhys 2 permanence 가 쉬운 조건) 은 "조회만으로 잡히는 재등장 불일치" 로 해소.

> 방향 정본 [`DIRECTION_2026-09-25.md`](DIRECTION_2026-09-25.md) §2 와 [`ABSTRACT_v0_2026-09-25.md`](ABSTRACT_v0_2026-09-25.md) 의 문장마다 **검증된 수치 · 출처 · 등급 · 들어갈 그림 · 남은 검증** 을 적는다.
> 등급: **확정** (적대 검증 통과) · **시사** (검증 뒤 범위 한정) · **1 차** (적대 검증 전) · **★** (미실행). 표적은 따로 적지 않으면 표준 양방향 h, 판독은 파라미터 없는 템플릿 argmax (Chebyshev ≤ 1).
> 수치는 편집 시점에 JSON · 문서에서 옮겼다. 인용할 때는 JSON 키를 다시 연다.

## 0. 분석 절의 뼈대 — 그림 셋 + 표 하나

| 그림 | 무엇을 보이나 | 패널 | 주 데이터 |
|---|---|---|---|
| **Fig.1 reach** | 관측이 끊기면 내용은 마지막 관측에서 일정 거리까지만 옮겨진다 | (a) v3 합성 GT: 절벽 거리 vs 슬롯 (stride 1/2/4 × 분할 16/32) (b) K400 인공 팬 A(d) (c) SSv2 · EK100 A(d) (d) 실영상 내용 중 reach 밖 비율 | `exp_results/p2/sc1_horizon_units.json` · `scene/scene_reach{,_nat_ssv2,_nat_ek100}.json` · `scene/scene_deep*.json` |
| **Fig.2 permanence** | 마지막 관측에 없으면 그리지 않는다 | (a) v11 vanish 방향별 (b) RollOutV2/V3 슬롯별 물체다운 토큰 (late 0 vs 가림막 없음 4–5) (c) M2 경계 패치: 있음은 경계 토큰을 따라감 (d) ★ 장면 단위 인공 가림 띠 (묶음 Q) | `z_research/IntPhysGenV11/README.md` · `RollOutV2/.../POSITION_READOUT_2026-09-12.md` · `Archive/M2_BOUNDARY_PATCH` · ★ `scene/scene_permanence.json` |
| **Fig.3 개입** | 병목은 먼 출처 조회 강도; 규칙 · encoder 아님 | (a) 규칙 교체 ΔA ≈ 0 (b) 출처 bias β = 1/2/4 단조 · 반대 방향 대조 (c) p_src / p_ring / p_copy 거리 감쇠 (d) 매개 (거리 층화 44–67 %) | `scene/scene_reach.json` (arms.*.delta_vs_base) · `scene/scene_reach.json` attention · `verify/verify_iscene.json` |
| **Table 평가 감사** | 쓰이는 벤치마크는 이 영역을 안 잰다 | VoE 복사 기준선 · 인과 표적 번짐 · EK100 predictor 치환 | `Archive/H6_*` · `Archive/SC3_*` |

## 1. 문장별 근거 (DIRECTION §2 · ABSTRACT 순)

### 1-1. "JEPA 는 latent world model 후보. 근거: 직관물리 창발 · anticipation SOTA"
- 문헌 (Garrido et al. IntPhys 1 88.89 %; V-JEPA 2 EK100 Table 5). 우리 재현: IntPhys 1 skip2_w32 Filtered 88.89 (CLAUDE.md §5-4), EK100 released 규약 35.34 (`z_research/anticipation/EK100/README.md` §2-A-1).
- 등급 확정 (재현). 그림: 동기 문단 · Table.

### 1-2. "predictor 는 state 를 진화시키지 않고 마지막 관측에서 가져와 채운다"
| 하위 문장 | 수치 | 출처 | 등급 | 그림 | 남은 검증 |
|---|---|---|---|---|---|
| 내용은 마지막 관측에서 일정 거리까지만 옮겨진다 (reach) | v3 21 절벽: 슬롯 상수 모형 R² 0, 거리만 R² 0.71 / 0.84 / 0.68 (release / pv1 / Ariel); Ariel stride 1 에서 13–17 슬롯, stride 4 에서 2–3 슬롯인데 거리 6–7 칸 동일; 분할 16 ↔ 32 거리 4.27 / 4.31 (release), 6.50 / 6.14 (Ariel) | `SC1_REACH_NOT_HORIZON` §3-1–3-4, `p2/sc1_horizon_units.json` (`cv`, `fit_K_D`) | 1 차 | Fig.1a | 정확 칸 판독에서 release 슬롯 계수 > 거리 계수 (VERIFY_ISCENE C1) → 연속 위치 판독 (묶음 P cos 지도) 로 재검정 |
| release 는 순수 거리가 아니라 "약 2 칸 + 1.6 걸음" | plateau stride 1/2/4: 2.3 · 3.2 · 4.9 칸; 적합 K 1.6 [1.1, 2.0] · D 2.4 | SC1 §3-3 · 정정 블록 | 1 차 | Fig.1a 주석 | — |
| 거리이지 경과 시간이 아니다 | 거리 고정 시 t 삼분위 효과 없음 (3–4 칸: .31/.35/.18), 시간 고정 시 거리 효과 (t 16–24: .23/.06/−.08); 부분 R² dist .056 vs t .003 | SC1 §3-7, `p2/sc1_dist_vs_time.json` | 1 차 | Fig.1a 보조 | 궤적 CI 는 회귀에만 |
| 장면 전체 (물체 하나 아님) 에서도 같다 | K400 팬 1,600 clip: release R50 4.8, Ariel 6.5, pv1 5.0 칸; 칸당 −0.077 / 슬롯당 −0.035 (release), Ariel 슬롯당 −0.008 | `I_SCENE` §4-1(1), `scene/scene_reach.json` (`arms.R:base.R50`, `slot_vs_dist`) | 1 차 (VERIFY C1 확정) | Fig.1b | — |
| 실영상에서도 | SSv2 R50 release 2.7 / Ariel 4.0; EK100 2.1 / 2.1 (Ariel = release); 칸당 −0.069 (SSv2) · −0.129 (EK100) | `I_SCENE` §4-2(1), `scene/scene_reach_nat_{ssv2,ek100}.json` | 1 차 | Fig.1c | flow 유사 정답 합의 토큰 27–37 % 만 (질감 편향) |
| flow 유사 정답 (독립 판독) 도 같은 곡선 | SSv2 합의 쌍 2,031: 변위 0–1 → 4–6 칸 release .73 → .13, Ariel .85 → .37 (짝 +0.24 [0.18, 0.30]); v3 에서 flow vs GT 곡선 일치 | `SC2V2_FLOW_REACH_SSV2` §3, `ssv2/flowtrack.json` (`flow_valid`) | 1 차 | Fig.1c 보조 | — |
| 실영상 움직이는 내용의 약 절반이 reach 밖 | SSv2 46 % (변위 가중 62 %), EK100 51 % (62 %), 팬 22 % | `I_SCENE` §4-3 C, `scene/scene_deep_nat_*.json` (`C_coverage`) | 1 차 | Fig.1d | — |
| reach 밖 출력 = 절반 제자리 복사 + 절반 흐림 | 팬 release 6–9 칸: 출처 .07 / 복사 .31 / 그 밖 .62; cos(p, 출처) .51 vs 복사 .55 (encoder 천장 .65) | `I_SCENE` §4-3 B, `scene_deep.json` (`B_output_beyond`) | 1 차 | Fig.1 캡션 | — |
| 가려진 내용은 옮기지 않는다 (permanence) | v11 vanish `물체→빈` 0.0 (`빈→물체` 100.0); k=1…4 평평; RollOutV2/V3: late 0 슬롯 vs early/mid 3 vs 가림막 없음 4–5; 자 없는 검사 late k=4 슬롯 0 부터 없음 (frac 0.54 → 0.09) | CLAUDE.md §5-1 · §5-5, `POSITION_READOUT_2026-09-12.md` §C, `IntPhysGenV11/README.md` | **확정 (합성 물체)** | Fig.2a–b | **장면 단위 · 실영상 없음 → 묶음 Q (인공 가림 띠) ★** |
| 있음은 경계 1–2 튜블릿 토큰을 따라간다 | M2: 움직이는 물체 연속 contrast 회복 1.10 [1.08, 1.12] / 파괴 1.19 [1.17, 1.20]; 정지 물체는 역사에도 기댐 (파괴 0.53–0.71); M6 hist_none 물체다움 −0.16 / −0.28 / −0.20, 진행 비율 0.547 vs 0.542 | `M2_BOUNDARY_PATCH`, `M6_HISTORY_SHIFT` 정정 2, `m6/m6_v3.json` | 확정 (범위 한정) | Fig.2c | 경계 인코딩 길이 L 스윕 (VERIFY_R2 §4) 미제출 |
| 자기 예측은 관측 대신 못 쓴다 (closure) | ar_pA j3–7 기울기 −0.012 [−0.042, +0.018] (진실 0.440); 진짜 관측 ar_z +0.361 [0.319, 0.402] (탐침); Ariel 사슬 0.226, pv1 0.139 (판독에 따라 0.260); 한쪽만 참이면 약 78 % 회복 | `P3_CLOSURE_TEST` §표, `P3B` 정정 2, `p3/p3_v3_p3b.json` (`r2`) | 확정 (행동) · 78 % 는 시사 | 본문 (그림 없음, 표 한 줄) | 누적 vs 토큰 질은 판정 보류 |

### 1-3. "정보는 있는데 배치가 없다"
| 하위 문장 | 수치 | 출처 | 등급 | 그림 |
|---|---|---|---|---|
| p 에 정체성 정보가 있다 | v11 등가속+가림 p probe shape 98.46 / color 99.49 / env 100 (채점 민감도 shape 4.46) | CLAUDE.md §5-2, `IntPhysGenV11/Archive/surprising_score/RESULTS_2026-08-30.md` | 확정 | Fig.3 캡션 · 본문 |
| encoder 는 병목이 아니다 (내용은 있고, 가리키면 그려진다) | 출처 bias +4 → 팬 release 4–6 칸 +0.22, Ariel 6–9 칸 +0.43, pv1 +0.23; SSv2 Ariel R50 4.0 → 6.3; 근거리 손상 없음 (+0.01 ~ +0.04) | `I_SCENE` §4-1(2) · §4-2(3), `scene/scene_reach.json` (`arms.R:bias4.delta_vs_base`) | **확정 (VERIFY C3)** | Fig.3b |
| encoder 등변성 | 창 3 칸 이동: 정렬 cos 0.706 vs 비정렬 0.431 (z) | `I_SCENE` §4-1(5) | 1 차 · 기준값 없음 (보조) | 부록 |

### 1-4. "병목은 먼 출처 조회 강도 (어디서 + 얼마나)"
| 하위 문장 | 수치 | 출처 | 등급 | 그림 | 남은 검증 |
|---|---|---|---|---|---|
| 규칙 (양방향) 은 병목 아님 | release 에 prefix 규칙: 팬 +0.02 / +0.01, SSv2 · EK100 ≈ 0; isolated −0.06; Ariel 에 full −0.21 (분포 밖) | `I_SCENE` §4-1(2), VERIFY C2 | **시사 (범위 한정)** — v3 에서는 prefix 규칙이 release 를 −0.06 ~ −0.10 | Fig.3a | — |
| 출처 bias 는 세기별 단조 · 방향 특이 | β 1/2/4: +0.03 / +0.07 / +0.22 (4–6 칸); 반대 방향 −0.05 (근거리 −0.058 손상 = 대조군) | VERIFY C3 | 확정 | Fig.3b | — |
| 정답 없는 개입은 자연 영상에서 무효 | anticopy · temp ≤ +0.03 (팬 · SSv2 · EK100); v3 물체에서는 +0.06–0.09 유효; Ariel temp2 팬 6–9 칸 +0.05–0.08 | VERIFY C3 정정 4 | 시사 | Fig.3 캡션 | — |
| attention: 방향은 (일부만) 알고 강도가 거리로 준다 | 팬 release p_src/p_ring 5–9 배 (균등 귀무 70 → 4.5 배), p_src 1–2 → 9–13 칸 15 배 감쇠, 6 칸 넘으면 p_copy > p_src; SSv2 2 배 · EK100 1.5 배 (토큰 중앙값 < 1) | `I_SCENE` §4-1(3) · §4-2(2), VERIFY C4 | **확정 + 추가** | Fig.3c | — |
| 방향을 아는 토큰은 첫 슬롯 근처 · 국소 일관 운동 | SSv2 아는 비율 .33 (슬롯 0 .61 → 7 .24); 일관성 4 분위 적중 .40 → .14; 중간 층 가장 선택적 | `I_SCENE` §4-3 D, `scene_deep_nat_*.json` (`D_direction_knowers`) | 1 차 | Fig.3c 보조 | 토큰 종류 (손 · 물체 · 배경) 라벨 없음 |
| Ariel 우위의 약 절반이 조회 강도로 설명 | 거리 층화 매개 팬 44–56 %, SSv2 41–67 % (pooled 95 / 99 % 는 거리 교락); 이득의 75–95 % 가 "출처 ↑" 칸 | VERIFY C5, `verify/verify_iscene.json`, `I_SCENE` §4-3 F | **시사** (95 % 철회) | Fig.3d | 나머지 절반 (내용 변환) 미분리 |
| 학습 없는 등속 외삽 출처는 실영상에서 무효 | bias4_cv: SSv2 −0.01 / −0.05, EK100 ≈ 0 (추정 출처 = 진짜 45 %); v3 에서는 +0.05–0.10 | `I_SCENE` §4-2(3) | 1 차 | Fig.3 캡션 | — |

### 1-5. "미래 예측으로 다시 학습해도 같다 (구조 문제)"
| 하위 문장 | 수치 | 출처 | 등급 | 그림 | 남은 검증 |
|---|---|---|---|---|---|
| Ariel (scratch, future-only, SSv2+K400) 도 reach 한계 | 팬 6.5 / SSv2 4.0 / EK100 2.1 칸; 실영상 방향 아는 토큰 .48 / .32; 자기 예측 사슬 0.226 | 위 표들 | 1 차 | Fig.1–3 A 행 | **rollout 학습 (action 없는 AC 형) ★ — 구조 문제 확정/반증** |
| 학습하면 reach 는 늘지만 (epoch 단조) 한계는 남는다 | Ariel s1_C16 plateau ep5/19/30/43 = 3.5 / 4.3 / 4.7 / 5.1 칸 (release 2.3); t10 hit−null release −0.139, pv1 +0.227, Ariel +0.255 (천장 +0.324) | SC1 §3-6 `p2/g_reach_epochs.json`; `TRAINED` 정정 2, `cmp/cmp_predictors.json` | 1 차 · TRAINED 는 정정 2 | Fig.1 보조 | epoch 별 CI 없음 |
| 데이터 커리큘럼 (큰 변위) 은 도메인 안에서만 | 팬 증강 post-FT (Ariel 규칙): 팬 R50 6.5 → 8.7, SSv2 3.8 / 3.8, EK100 2.1 / 2.1 (팬 vs 고정 창) | `I_SCENE` §5-1, `scene/scene_reach_*_trainedN.json` | 1 차 | Fig.1 보조 또는 부록 | release 판 (M 재평가) 대기 |

### 1-6. "벤치마크는 이 영역을 재지 않는다"
| 하위 문장 | 수치 | 출처 | 등급 | 그림 |
|---|---|---|---|---|
| VoE: 마지막 latent 복사가 릴리즈와 구분 안 됨 | IntPhys 1 skip2_w32: 복사 85.00 vs p 88.89, +3.9 [−0.6, +8.4] (p 만 맞힌 쌍 12 vs 복사만 5); 공식 칸 skip2_w16 은 +12.2 유의 → "칸에 따라 다름" 병기 | `H6` 정정 1–3 | 확정 (프로토콜 민감) | Table |
| 양방향 표적 번짐이 사건 전 튜블릿을 판별 | 이동+보임 90.4 % → 인과 표적 50.0 %; 복사도 표준 − 인과 +7.8 [3.3, 13.3] | `H6` 정정 2 | 확정 | Table |
| EK100: predictor 출력을 복사 · 평균으로 바꿔도 거의 그대로 | released 35.34 → 32.80 / 34.01; paper 19.07 → 16.46 / 17.59; 재현 검사 35.34 | `SC3_EK100_PREDICTOR_SUBSTITUTION` | 1 차 (probe 고정 = 상한) | Table |
| VoE 순위 ≠ 외삽 순위 | IntPhys 1 에서만 (Ariel 69.44 vs 릴리즈 88.89); v11_split_test 는 릴리즈 최하 75.8 | `TRAINED` §4-4, `z_research/TrainingEffects/` | 시사 | Table 각주 |

### 1-7. "state evolution 을 직접 재는 프로토콜" (기여 i)
- reach 곡선 (인공 팬 정답 · flow 유사 정답 + encoder 합의 · v3 GT), permanence (v11 · RollOut · ★ 띠), closure (P3 팔). 판독 보정: flow vs GT 0.87–0.95, v3 에서 flow 곡선 = GT 곡선 (`SC2V2` §2). 등급 있음 (permanence 장면 판 ★).

### 1-8. "state 를 굴리는 world model · 주 평가 향상" (기여 ii) — 전부 ★
- state 모듈 (렌더러 유지 + slot state + adapter): 설계만 (`DESIGN_CHOICES` §3 단계 S 미기재 · 대화 기록). EK100 장거리 · IntPhys 2 · Ego4D · EgoExo4D: 미실행 (Ego4D · EgoExo4D 데이터 · 하네스 없음).

## 2. 빈틈 목록

### 2-1. 실영상 근거가 없는 문장
| 문장 | 지금 근거 | 닫는 분석 (학습 없음) | 비용 |
|---|---|---|---|
| 가려진 내용은 옮기지 않는다 (permanence) | 합성 물체 (v11 · RollOut) 만 | **묶음 Q**: K400 팬 + 마지막 k 프레임 화면 고정 띠, hid vs vis 토큰 복원 (release · Ariel · pv1 · 학습 run) — 큐에 있음 | GPU 약 30 분 |
| 정보는 있는데 배치가 없다 | v11 probe (합성) | 실영상 판: SSv2 · EK100 에서 p 에 선형 probe (방향 · 정체성 = flow 방향 / 좌우 라벨) — `e_ssv2_extract` 캐시 재사용 | CPU 1 h |
| 자기 예측은 관측 대신 못 쓴다 | v3 만 | 인공 팬에서 P3 팔 (ar_z vs ar_pA) — 정답 정확 | GPU 20 분 |

### 2-2. 1 차뿐인 핵심 문장 (적대 검증 필요)
- SC1 거리 법칙 (판독 의존: 정확 칸에서 release 슬롯 몫) → 묶음 P (cos 지도) 로 연속 위치 판독 (soft-argmax · top-k) 재검정. GPU 15 분 + CPU.
- SC2v2 flow 곡선 · 합의 필터의 편향 (질감) → 합의 없이 flow 만 · 합의 비율 층화. CPU.
- SC3 (probe 고정 상한) → 치환 입력으로 probe 재학습이 진짜 기여 (학습 필요 → 사용자 결정) 또는 "상한" 으로만 쓴다.
- I-scene §4-3 B · C · D · F, §5-1 N, SC1 §3-7 → 적대 검증 워크플로 1 회. CPU.
- Ariel epoch 단조 (CI 없음) → 궤적 bootstrap 추가. CPU 10 분.

### 2-3. 결정 실험 (스토리의 "구조 문제" 문장)
- rollout 학습 predictor (action 없는 AC: frame-causal + teacher forcing + rollout 손실) 를 같은 판에. **학습 필요 → 사용자가 나중에.** 분석만으로 대신할 수 있는 것: Ariel 의 자기 예측 사슬 (0.226) 과 attention 관측이 이미 "미래 전용 학습으로도 조회형" 을 시사 — 문장을 "재학습해도 (Ariel) 같다" 로 한정해 쓴다.

### 2-4. 다시 쓰면 안 되는 문장 (철회 · 금지)
- "encoder 는 충분하고 predictor 가 병목" · "predictor 가 X 를 못 만든다" · "가림이 표현을 망가뜨린다" · "목적함수 때문이다" · α · 보정 · 헤지 계수 (CLAUDE.md §12).
- "학습 지평 (8 슬롯) 에서 멈춘다" (SC1 기각) · "reach 는 fps · 걸음 수와 무관한 고정 거리" (release 는 걸음 몫 섞임) · "매개 95 / 99 %" (거리 층화 44–67 %).
- "마지막 토큰 하나가 정한다 · 역사 무관" · "역사 토큰이 뒤로 끈다" · "속도를 한 걸음만 쓴다" (VERIFY_R2).
- "predictor 가 물체를 마지막 자리에 둔 채 머문다" (자 기본값) · "wall 에서 벽 속 한 칸 반 파고들어 멈춘다" (CLAUDE.md §6).
- "SSv2 시간 정렬 기울기 0.01 → 진전 없음" · "토큰 크기비 진실의 15 %" (E-SSv2 철회).
- "synthetic 에서 predictor 가 문제다" (DIRECTION §5).

## 재현
- 수치 원본: `auto_research/exp_results/{p2,scene,ssv2,verify,cmp,p3,m6}/*.json`. 각 문서 `## 재현` 절에 추출 · 분석 명령이 있다.
- 이 문서는 수치를 새로 계산하지 않았다. 편집 시점 (2026-09-26 아침) 의 문서 · JSON 값을 옮긴 것이다.
