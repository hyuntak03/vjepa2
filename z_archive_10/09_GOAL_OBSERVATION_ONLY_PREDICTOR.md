# 09 — 목표: action 없이 관측만으로 intuitive physics 를 따르는 미래를 예측하는 predictor

> **결론** — 목표 predictor 는 아직 없다 (★, 학습은 사용자 몫). 이 장이 고정하는 것은 셋이다: 무엇을 넘으면 "Ariel 을 넘었다" 고 하나 (합격선 8 칸), Ariel 이 어디서 왜 실패했나 → 설계 후보 (사건 3b 는 후보 없음), 이미 시도된 처방은 전부 도메인 안에서만 듣거나 학습이 안 됐다.
> **근거** — Ariel 세 팔 (frozen ViT-H + action 없는 실영상 미래 예측) 은 EK100 reach 2.2 (릴리즈 2.1) · permanence .46–.47 (릴리즈 .53) · v11 late vanish 물체→빈 flat/ramp 37 / 13 · 가속 α AR −0.42 / −0.46 (`auto_research/Archive/AR_PREDICTOR_2026-09-27.md` · `z_research/TrainingEffects/scores/SCORES.md` · `auto_research/Archive/AR40_READOUT_2026-09-30.md`).
> 큰 변위 커리큘럼은 팬 평가에서만 reach 를 늘린다 (팬 R50 4.83 → 10.23, SSv2 2.70 → 3.55, EK100 2.11 → 2.20; `auto_research/Archive/I_SCENE_CAUSAL_BOTTLENECK_2026-09-25.md` §5-4) · 움직임 정렬 조회 (mret) 는 bias 가 학습되지 않아 판정 보류 (§5-2).
> 진짜 출처를 알려 주면 먼 내용이 나온다 (bias +4 → 팬 4–6 칸 +0.22, 충분성 확정) — 정보는 encoder 에 있고 조회가 막혀 있다 (`I_SCENE` §4-1, `VERIFY_ISCENE`).
> **미결** — 합격선 ★ 셋 (Ariel 의 gravity 꼭대기 · ledge 다중 미래 · prefix/full 의 α) 이 아직 측정되지 않았고, 사건 (3b: 낙하 · 충돌) 을 겨누는 설계 후보가 없다.
> 이 장이 받치는 논문 문장: 목표 "action 없이 관측만으로 intuitive physics 를 따르는 미래를 예측하는 predictor — Ariel 을 넘는다"  ·  원재료: G05 · G10 · G10b · G10c · G03 · G03b · G02a · G02b · G04a · G13 추출문 → 원 문서는 각 절 끝 `출처`.

---

## 1. 목표 문장과 방법 절의 첫 줄

### 1-1. 목표 (사용자 원문, 그대로)

| 날짜 | 원문 | 출처 |
|---|---|---|
| 10-08 | "Action conditioned 없이, 순수 observation 만으로 intuitive physics를 배우고 ⇒ 이에 따른 plausible future prediction 을 잘하는 predictor를 만들고 싶음" | `auto_research/paper/PAPER_STORY_2026-10-08.md` §0 |
| 10-08 | Ariel 반례에 대해: "이건 어케든 맞춰야지." → 합격선 | 같은 문서 §0 후속 결정 |
| 10-10 | "아직 AC가 우리가 도달할 이상은 아니야 그걸 명심해." → 풀이: "아직 최종이 아니라는거라서 AC 스토리도 일단 작성" → AC 는 작은 subsection 초안 (`09b_AC_STORY.md`), 이 장은 그 앞 단계 | 사용자 (이 아카이브 작업 중) |

한 문장 (정본 §1): "우리는 action 없이 관측만으로 intuitive physics 를 따르는 미래를 예측하는 predictor 를 만든다." 실패의 자리 (P3) 는 **마지막 관측에** 보이는 것만 외삽하고, 안 보이는 것 (가려진 물체) · 없는 것 (가속의 누적 · 낙하 · 되돌아옴) 은 예측하지 못한다는 것이다 (`PAPER_STORY_2026-10-08` §1).

### 1-2. 넘어야 할 반례 — Ariel 세 팔

Ariel 세 팔은 이미 "frozen encoder + 새 predictor + action 없는 실영상 미래 예측" 이고 실패했다 (정본 §2-4). 그래서 새 predictor 는 "관측만으로 배운다" 만으로는 기여가 되지 않는다.

| 팔 | `arch.kind` | 학습 목표 · 입력 | 지금 판 | 출처 |
|---|---|---|---|---|
| A full | `oneshot` | full attention + mask token, 문맥 = online encoder 창 한 번 | ep40 (loss 0.5761) | `z_training/ARIEL_CHECKPOINTS.md` §1 |
| B prefix (block-causal) | `prefix` | prefix 마스크 + mask token, 미래 전용 | ep45 (loss 0.5836) | 같은 곳 |
| C AR | `ar` | 자기회귀 되먹임 (V-JEPA 2-AC 레시피에서 action 뺀 것), 문맥 · 타깃 = 블록별 LN(target_encoder), 손실 tf + rollout 8 블록 | ep18 → ep40 최종 | 같은 곳 · `AR_PREDICTOR_2026-09-27.md` |
| 공통 | — | 릴리즈 ViT-H encoder frozen, predictor scratch, SSv2 + K400 @ 12 fps, 48 프레임, 문맥 {2,4,8,16} × 예측 {1,2,4,8} 블록, lr 3e-4 cosine, **action 없음** | — | 같은 곳 |

⚠️ 옛 prefix ep5 / 19 / 30 / 41 / 43 체크포인트는 2026-09-27 에 지워졌다. ep45 는 ep43 과 다른 파일이다 (`ARIEL_CHECKPOINTS.md` 1-b). 아래 합격선 칸 가운데 일부 (vanish · closure · IntPhys 1) 는 ep43 판 값이다. 새 predictor 와 같은 판에서 다시 재려면 ep45 · ep40 로 잰다.
⚠️ AR 팔은 A · B 와 다른 점이 셋이다: 인코딩 단위, 문맥 encoder (online vs EMA), 입력 (mask token vs 되먹임). 타깃 공간도 달라 **쌍 정확도만** 비교할 수 있다 (`ARIEL_CHECKPOINTS.md` 단서).

### 1-3. 방법 절의 첫 줄 — 형식

> **"Ariel (같은 encoder · 같은 action 없는 실영상 미래 예측) 과 무엇이 다른가: §3 표의 어느 실패를 어느 후보로 겨누었고, 왜 그것이 Ariel 의 학습에는 없던 압력인가."**

정본 §2-4 방법 절 규칙 1 이다. 아직 쓸 문장은 없다 (★). 후보는 §3, 이미 실패한 것은 §4 에 있다.

---

## 2. 합격선 — 새 predictor 가 같은 판에서 Ariel 을 넘어야 하는 칸

정본 §2-4 표를 그대로 옮기고, 원 문서 · 등급 · 칸마다 Ariel 값을 낸 팔을 덧붙였다. ✓ = 정본이 2026-10-08 에 산출물에서 다시 확인한 값, ★ = Ariel 값 미측정.

| P3 축 | 지표 (판) | 릴리즈 | Ariel (가장 나은 팔) | 합격선 | 등급 | 원 문서 |
|---|---|---:|---:|---|---|---|
| 보이는 것 | 실영상 reach R50 (칸) — SSv2 · **EK100** | 2.7 · 2.1 | 4.1 · 2.2 (full 4.17 / prefix 4.06 · 둘 다 2.17) | **EK100 > 2.2** (도메인 밖, 어떤 학습도 못 넘은 천장) | 1 차 (reach 곡선은 VERIFY_ISCENE 확정, EK100 불변은 VERIFY_ISCENE2 확정) | `auto_research/Archive/I_SCENE_CAUSAL_BOTTLENECK_2026-09-25.md` §4-2 · `AR_PREDICTOR_2026-09-27.md` §2-1-c |
| 안 보이는 것 | v11 문맥 끝 가림 vanish '물체→빈' (late, static · flat · ramp, %) | 65 · 3 · 0 | 49 · 37 · 13 (block-causal **ep43**; prefix ep45 는 49.1 · 37.5 · 14.3) | 셋 다 > 50, **v11 로 학습하지 않고** | 확정 (하네스 재계산 `ok: True`) | `z_research/TrainingEffects/scores/SCORES.md` §3 · `COPY_HARD_SUBSET_2026-09-29.md` §2-1 |
| 안 보이는 것 | 실영상 permanence (가림 띠, hid/vis 복원 비, k=4) | .53 [.50, .56] | .46–.47 (full .47 · prefix .46 · AR .47 [.44, .50]) | > .53 | 확정 (VERIFY_ISCENE2 §1 · VERIFY_AR) | `I_SCENE` §4-4 · `AR_PREDICTOR` §2-2 |
| 없는 것 (3a) | 가속 반영 α (RollOut_v3 arc · ledge, t4–7; 0 = 등속 외삽, 1 = 진실) | +0.08 · −0.25 | AR −0.42 · −0.46 (prefix · full ★) | 둘 다 분명히 > 0 (encoder 천장 h +1.43 · +1.28) | 1 차 → arc · ledge t0–7 의 AR α < 0 은 VERIFY_AR40RD 확정 | `AR40_READOUT_2026-09-30.md` §1 · `VERIFY_AR40RD_2026-09-30.md` |
| 없는 것 (3a) | gravity 꼭대기 문맥 — 공 자리가 arc 쪽인 block | 2 % | ★ | > 50 % | 등급 문서에 없음 (10-08 정정판) | `z_research/v11_realistic/README.md` §4 |
| 없는 것 (3b) | ledge 실물 메시 VoE · 다중 미래에서 낙하 선택 | 0 / 120 ✓ | ★ | 복사 이상 + 낙하 선택 > 우연 | 등급 문서에 없음 (tie 0, 정본 ✓) | `z_research/v11_realistic/README.md` §2 |
| 닫힘 | 자기 예측 되먹임 − one-shot (장면 적중, 슬롯 3–7) | −0.03 ~ +0.01 | −0.12 ~ −0.22 (Ariel ep43 −0.12 ~ −0.16 · N1 −0.15 ~ −0.22 를 합친 범위) | ≥ 0 | 확정 (VERIFY_ISCENE2 §2) | `I_SCENE` §4-5 |
| 떨어지지 않음 | IntPhys 1 (leakage 큼 — 올라도 증거 아님) · 복사 대비 Δ | 88.89 (복사 85.0) ✓ | 69.44 (block-causal ep43, skip2_w32) | 복사 85.0 아래로 안 떨어짐 | 확정 (SCORES 재계산) | `SCORES.md` §1 · `z_research/TrainingEffects/ip1_copy/IP1_COPY.md` |

⚠️ **"가장 나은 팔" 은 칸마다 다른 팔이다** — reach 는 full · prefix, vanish · IntPhys 1 은 block-causal ep43, permanence 는 세 팔, α 는 AR 만, closure 는 Ariel ep43 + N1 (Ariel post-FT) 이다. 새 predictor 를 잴 때는 세 팔 (ep45 · ep40 · AR ep40) 을 같은 판에 다시 건다 (정본 §5-3).
⚠️ **AR 은 vanish 칸에서 합격선을 이미 넘는다** — AR ep38 late 물체→빈 99.1 · 85.7 · 72.3 (`COPY_HARD_SUBSET` §2-1). 그런데 같은 칸 위치 자 판독에서 AR 의 미래에는 물체 토큰이 '있음' 0–10 % 이고, 이것은 빈 장면 오탐 2.5 % 와 구분되지 않는다 (`VERIFY_AR40RD` §4). VoE 정답은 먼 칸의 전역 오프셋에서 온다 (presence-without-placement, `VERIFY_PRESENCE_2026-09-29.md`). **이 칸은 VoE 점수만으로 합격 판정을 할 수 없다.** 위치 · 있음 판독을 같이 낸다.
⚠️ α 의 절대값은 어느 자로 읽느냐에 달렸다. 교차 자 (p@h) 로 읽으면 arc t4–7 이 AR +0.60 / 릴리즈 +1.10 으로 부호가 뒤집힌다 (`VERIFY_AR40RD` §2). 합격선 "> 0" 은 같은 자 · 교차 자 둘 다에서 본다. 확정으로 쓸 수 있는 것은 "같은 자 안에서 AR ≤ 릴리즈" 와 "h 천장 ≈ 1" 뿐이다. 이 α 는 가속 반영 투영 계수다. 인용 금지인 α 분해 (헤지) 와는 다른 양이다.
⚠️ EK100 "어떤 학습도 못 넘은 천장" 은 open-loop 예측 이야기다. 진짜 블록을 되먹이는 AR tf 는 4.00 이지만 예측이 아니다 (`AR_PREDICTOR` §2-1-c). Ariel epoch 을 늘려도 EK100 은 불변이다 (2.17–2.21, 확정). 팬 판의 포화는 lr 스케줄 끝과 구분되지 않는다 (미결, `I_SCENE` §5-3 2 차 검증).
⚠️ "v11 로 학습하지 않고" — v11 post-FT 의 87 · 100 · 100 은 도메인 안이다 (v11 궤적 재생 — 궤적으로 묶으면 영속 flat Δ 의 CI 가 0 에 걸침, `z_research/TrainingEffects/README.md` §5 단서 1). 근접 도메인 Predictor_v1 (등속 · 가림 · 경사) 의 52 · 88 · 78 도 '빈→물체' 100 → 73 과 맞바꾼 값이다 (같은 README). 그러므로 근접 도메인 학습도 이 칸의 기준이 되기 어렵다 (이 장의 해석).
⚠️ IntPhys 1 칸은 Ariel 이 이미 떨어진 칸이다 (69.44 < 복사 85.0, p − 복사 −16.1 [−23.3, −8.9]; `TrainingEffects/README.md` §1-1). 이 칸을 넘는 것은 "떨어지지 않음" 일 뿐 intuitive physics 의 증거가 아니다 (정본 규칙 3).

### 2-1. 정본 표에 없지만 정본 규칙 3 ("판정은 예측이 필요한 칸에서") 이 요구하는 보조 칸

| 칸 | 릴리즈 | Ariel | 복사 | 등급 | 원 문서 |
|---|---:|---:|---:|---|---|
| v11 copy-hard 부분집합 (복사가 틀리는 쌍, n 2,877) | 48.4 [46.6, 50.2] | prefix ep45 51.0 · full ep40 49.6 · AR ep38 73.8 [72.2, 75.4] | (정의상 틀림) | 1 차 + VERIFY_R3 정정 | `COPY_HARD_SUBSET_2026-09-29.md` §2-1 |
| IntPhys 2 Main copy-hard (n 240, w48 C24) | 21.2 [16.0, 26.6] | prefix 19.2 · AR ep38 40.4 | — | 1 차 | 같은 곳 |
| RollOut_v2 위치만 다른 쌍 (ledge / wall, 392 쌍씩) | 0.0 / 17.6 | ★ | 7.9 / 2.3 | 정본 ✓ (json) | `auto_research/Archive/COPY_VS_POSITION_2026-10-07.md` |

⚠️ copy-hard 를 읽는 법 — 복사와 오류가 무작위로 겹친다는 귀무 아래에서는 copy-hard 정확도의 기대값이 전체 정확도와 같다. 그 귀무와 비교하면 모든 predictor 가 그 아래에 있다 (release −27 · prefix −28 · full −28 · AR −10 · v11_e10 −9). AR 73.8 은 "복사가 못 푸는 것을 푼다" 가 **아니라** "AR 의 오류가 복사 오류와 덜 묶여 있다" 로 읽는다 (`VERIFY_R3_2026-09-29.md` C3).
⚠️ 위치 쌍 채점 (토큰 평균 L1) 은 위치를 원리적으로 잘 보지 못한다 (`COPY_VS_POSITION`). 위치 주장은 위치 자 · 파라미터 없는 판독과 같이 낸다 (CLAUDE.md §5-5 읽는 규칙).

### 출처
`auto_research/paper/PAPER_STORY_2026-10-08.md` §2-4 · `z_training/ARIEL_CHECKPOINTS.md` · `z_research/TrainingEffects/scores/SCORES.md` · `z_research/TrainingEffects/README.md` · `auto_research/Archive/I_SCENE_CAUSAL_BOTTLENECK_2026-09-25.md` · `auto_research/Archive/AR_PREDICTOR_2026-09-27.md` · `auto_research/Archive/AR40_READOUT_2026-09-30.md` · `auto_research/Archive/VERIFY_AR40RD_2026-09-30.md` · `auto_research/Archive/VERIFY_PRESENCE_2026-09-29.md` · `auto_research/Archive/COPY_HARD_SUBSET_2026-09-29.md` · `auto_research/Archive/VERIFY_R3_2026-09-29.md` · `auto_research/Archive/COPY_VS_POSITION_2026-10-07.md` · `z_research/v11_realistic/README.md`

---

## 3. Ariel 이 왜 실패했나 → 설계 후보

### 3-1. 정본 대응표 (정본 §2-4 그대로, 측정값 · 등급을 붙였다) — 후보는 전부 **미검증**

| Ariel 의 실패 (측정) | 측정값 · 등급 · 원 문서 | 설계 후보 | 겨누는 칸 |
|---|---|---|---|
| 손실의 최저점이 복사다 — 복사가 틀리는 쌍에서 prefix · full 은 우연 (49–51) | v11 copy-hard prefix 51.0 · full 49.6, 릴리즈 48.4 (1 차, `COPY_HARD_SUBSET`) | copy-hard 가중 · 먼 지평 (§3-b 1, §3-d B) | 보이는 것 · 안 보이는 것 |
| 먼 출처 조회가 약하다 — reach 가 도메인 안에서만 는다 | 팬 R50 Ariel 6.5 vs 릴리즈 4.8 / EK100 2.1 = 2.1 (`I_SCENE` §4-1 · §4-2) | where 경로 (변위 → 조회 위치, §3-c ②) | 보이는 것 |
| 마지막 관측에 없는 내용은 이어 가지 않는다 — permanence 그대로 | hid/vis 릴리즈 .53 · Ariel .46–.47; 같은 칸 비가림 대비 약 1/3 (확정, VERIFY_ISCENE2) | 마지막 관측 dropout (§3-d D) · 역사 축적 (§3-c ④) | 안 보이는 것 |
| 가속을 넣지 않는다 (α < 0) | AR arc −0.42 · ledge −0.46 (VERIFY_AR40RD 확정) | 시간 척도 합성 (§3-d C) · 합성 규칙 (§3-d H) | 없는 것 3a |
| 자기 출력을 되먹이면 첫 걸음부터 무너진다 (AR) | roll − tf 슬롯 1 팬 −.042 [−.046, −.037] → 슬롯 6–7 −.35 (확정, `VERIFY_AR`) | 상태/렌더 분리 (§3-c ①) | 닫힘 |
| **사건 (낙하 · 충돌) 을 만들지 않는다** | ledge 0 / 120, 위치 쌍 ledge 0.0 · wall 17.6 (릴리즈; Ariel ★) | **후보 없음 — 빈칸.** 관측만으로 사건을 배우려면 사건이 있는 창의 비중 (데이터) 이나 그 창을 고르는 압력 (§3-d E 가림 사건 채굴의 사건판) 이 필요하다 (정본 원문) | 없는 것 3b |

⚠️ **3b 는 비어 있다.** 아래 후보 표의 어느 것도 낙하 · 벽 충돌을 겨누지 않는다. DC 표는 09-25, §3-b–d 는 09-29–30 에 쓰였다. 둘 다 ledge · wall 을 정의 밖에 두던 시기 (09-19 결정, 10-08 번복) 다. 방법 절이 3b 칸을 주장하려면 새 후보부터 있어야 한다.
⚠️ 대응표는 "분석이 아는 것 → 후보" 다. 원인 검정이 아니다 (`RETHINK_PREDICTOR_ROLE_2026-09-29.md` §4 "원칙 넷은 분석에서 **나온** 것이지 검증된 것이 아니다").

### 3-2. 후보 상세 — DESIGN_CHOICES §1 DC 표 (09-25)

| # | 후보 | 무엇을 고치나 | 근거 (원 문서) | 등급 | 시도 상태 |
|---|---|---|---|---|---|
| **DC1** | **움직임 정렬 조회** — 미래 질의가 복사 key (자기 자리) 가 아니라 **출처 자리** 의 문맥 내용을 강하게 가져오게 한다. 실영상에서는 출처 추정 (어디서) 과 조회 강도 (얼마나) 가 **둘 다** 모자란다. 학습 없는 등속 외삽은 출처를 못 맞힌다 (45 %) → 출처 추정은 학습된 모듈이어야 한다 | 병목 자체 | 진짜 출처 bias +4: 릴리즈 팬 4–6 칸 +0.22, Ariel +0.22 / +0.43 (6–9 칸). Ariel − 릴리즈 차의 약 절반 (거리 층화 44–67 %) 이 출처 attention 강도로 설명된다. 실영상 방향 선택성 1.5–2 배 (팬 5–9 배) (`I_SCENE` §4-1 · §4-2) | 충분성 확정 · 매개 시사 (VERIFY_ISCENE) | (a) 큰 변위 커리큘럼 → 도메인 안만 (§4) · (b) 출처 정렬 attention bias (`kind: mret`) → **무효 (학습 안 됨)** · (c) 운반 (latent warp) 모듈 → 미구현 |
| **DC2** | **미래 전용 · block-causal 목표** | 학습 목표 | Ariel: reach 1.5 배, 걸음 의존 소멸 (슬롯당 −0.008 vs 릴리즈 −0.035), 먼 출처 attention 3–6 배. 릴리즈 사전학습 마스크는 시간 전체 tube 라 future-only 질의를 배운 적이 없다 | 확정 (존재 증명) · 요인 미분리 | Ariel 코드 그대로 (`src/models/rollout_predictor.py`) — **새 predictor 의 기본값이지 차이가 아니다** (Ariel 이 이미 했다) |
| **DC3** | **자기 예측을 앵커로 삼는 닫힘 학습** (latent scheduled sampling) | open-loop 사슬 | 진짜 관측 한 튜블릿이 앵커면 매 걸음 전진한다 (ar_z j3–7 기울기 +0.361, 진실 0.440). 자기 예측은 앵커가 못 된다 (ar_pA −0.012 [−0.042, +0.018]) (`P3_CLOSURE_TEST_2026-09-25.md`) | 확정 (행동) | 미구현. ⚠️ AR 팔이 rollout 손실 8 걸음으로 학습했는데도 첫 걸음부터 무너졌다 → "되먹임 손실만으로는 안 닫힌다" (§3-b 전제) |
| DC4 | **있음 (presence) 을 시간에 따라 잃지 않게** | 지속 | 물체다움이 문맥 끝부터의 시간에 따라 준다 (튜블릿당 −4 ~ −6 pt; H4 · H23 · M6) | 확정 (행동) · 기전 미상 | DC1–3 이 해결하는지 먼저 본다 |
| DC5 | **평가는 reach 곡선으로** (라벨 없이, 변위 축) | 평가 | VoE 복사 85.0 vs 88.9 n.s.; EK100 1 s 에서 predictor 미래를 복사로 바꿔도 −2.5 — 둘 다 reach 안쪽만 잰다 | 확정 · 1 차 | `e_scene_reach{,_nat}.py` 로 이미 쓰는 중 (§2 합격선 reach 칸) |

⚠️ DESIGN_CHOICES 의 머리 한 문장 ("이 조회를 설계 목표로 삼은 predictor 가 실영상 open-loop 예측에서 낫다") 과 "DC1 이 논문의 기여" 는 스토리 부분이다. 10-08 에 대체됐다. 유효한 것은 DC 표와 §3-b–d 후보뿐이다 (DESIGN_CHOICES 머리 blockquote).
⚠️ DESIGN_CHOICES §3 단계 S "state 모듈" 과 §3-c "상태 s_t" 는 지금 금지된 state 프레임의 표현이다 (원 문서 표현 — 현재 금지). RETHINK §0-c: "상태/렌더 분리" 는 **구조 선택지로만** 두고 논지로 세우지 않는다 (09-30 사용자).

### 3-3. 후보 상세 — §3-b (09-29, "motion supervision 을 줘야 하나?" 에 대한 답: 외부 라벨 없이)

전제 (원 문서): 정보는 encoder 에 있다 (속도 · 카메라 운동 R² .69) · 정답 위치를 줘도 절반만 오른다 (qshift, 2 차 검증에서 시사로 강등) · 손실의 최저점이 복사다 · 되먹임 손실만으로는 안 닫힌다 · AR 의 late 정답은 전역 오프셋이다.

| 순 | 팔 | 무엇 | 판정 | 어느 쪽이든 서는 문장 | 상태 |
|---|---|---|---|---|---|
| 1 | **copy-hard 가중** | 창마다 복사 기준선 오차 (encoder 만으로 계산) 로 손실 가중 — 큰 이동 · 가림 · 재등장 창을 높게 | 실영상 reach R50, copy-hard Δ | 늘면 "표적 선택이 원인", 안 늘면 "구조 문제" 확정 | ★ (학습 코드에 가중치 한 줄) |
| 2 | **where 자기지도** | 변위 · 조회 위치 경로 (mret 류). 표적 = frozen encoder 토큰 대응 (미래 토큰 ↔ 마지막 관측 cos 최대, 합의 토큰만; GT 와 .87–.95 일치) | reach (특히 4–6 칸), qshift 상한과의 간격 | 상한에 닿으면 "어디" 가 병목, 못 닿으면 "옮기는 계산" 이 병목 | ★ |
| 3 | **상태/렌더 분리** | 렌더하지 않는 굴림 표현을 따로 두고 되먹임은 그것으로만 | closure (roll − tf → 0), permanence > .5, EK100 2–3 s Δ | 닫히면 DC 확정, 안 닫히면 "되먹임 공간" 가설 기각 | ★ (새 구조) |

분석 쪽 선행 준비로 원 문서가 적은 것: (1) copy-hard 가중이 실제로 어떤 창을 고르는지 (K400 · SSv2 학습 창 표본), (2) 팔 2 의 표적을 학습 창에 미리 계산해 두는 스크립트. ⚠️ 둘 다 **실행 기록을 찾지 못했다** (추출문 · README 에 산출물 없음).

### 3-4. 후보 상세 — §3-c (09-30, "어떤 모델을 만들어야 하나")

| 부품 | 무엇 | 분석 근거 (원 문서 요약) | 대조 팔 |
|---|---|---|---|
| ① 굴림 표현 (렌더 안 됨) | predictor 블록의 투영 전 은닉 (또는 별도 K×D). 되먹임은 이것으로만 | 렌더 출력을 되먹이면 첫 걸음부터 붕괴 (AR · ctx_ar) | ctx_ar 그대로 vs 되먹임을 은닉으로 (코드 몇 줄) |
| ② where 경로 | 미래 칸마다 조회 위치 u(s) (변위 · 속도장) 를 예측하고, 내용은 마지막 관측 토큰에서 그 자리로 가져온다 | 자리만 틀림 · 출처 힌트 하나로 +.2~.3 · EK100 미래 변위의 67–75 % 가 카메라 운동 → 전역 항 + 국소 항 | where on/off |
| ③ 렌더 | 굴림 표현 → encoder 공간 (LN h) 한 걸음 | 한 걸음 자체는 잘 한다 (tf reach ∞) | — |
| ④ 역사 | 문맥은 창 단위 z (prefix 팔) + 굴림 표현에 축적 | AR (블록별 문맥) 은 k=2 permanence 최저 .59 | ①+④ vs ① |
| ⑤ 학습 표적 선택 | 복사 오차로 가중 + 큰 걸음 (4 블록 앞) 혼합 | 걸음당 이동 ≤ 1.5 칸이면 복사로 풀린다 | 가중 on/off |
| ⑥ 되먹임 노출 | rollout 손실 + 굴림 표현 되먹임 | AR 은 rollout 손실 8 걸음에도 붕괴 → ① 없이 ⑥ 만으로는 안 된다 | — |

사다리 (싼 순서, 원 문서): ① → ⑤ → ② (mret 류 + encoder 대응 보조 손실) → 큰 걸음 혼합. **안 하는 것** (원 문서): slot attention · 물체 중심 구조 (장면 단위 유지, 사용자 09-25), flow 라벨, encoder 학습, 목적함수 교체.

### 3-5. 후보 상세 — §3-d (09-30, 규칙을 미는 손실 후보: 전부 라벨 없음 · 실영상 · 예측 목표 안)

| | 후보 | 강제하는 것 | 닿는 실패 | 위험 · 단점 (원 문서) | 원 문서의 판정 칸 |
|---|---|---|---|---|---|
| A | 굴림 = 읽기 일관성 `\|s_roll(t+k) − sg(s_read(t+k))\|` | 관측 없이 굴린 것 = 관측했을 때 | closure | 정지 해로도 만족 (렌더 손실 필수), RSSM · BYOL 과 같음 | 안정화 보조로만 |
| B | 먼 지평 + copy-hard 가중 렌더 | 복사로 안 풀리는 표적을 맞힌다 | reach · permanence | 조회로 풀 수 있는 창이 남으면 효과 작음 | 기본 조합 |
| C | **시간 척도 합성** `F_{12fps}∘F_{12fps} ≈ F_{6fps}` | 전이가 걸음 크기에 맞는 속도 규칙을 담는다 | 전이 (3a) · reach | 프레임률 증강 구현 필요 | reach (SSv2 4–6 칸) · 가속/등속/감속 분리 |
| D | **마지막 관측 dropout** (문맥 끝 m 블록을 지우고도 같은 미래) | 마지막 관측만으로 못 풀게 해서 역사에 기대게 | permanence | 너무 세면 불안정, m 스케줄 필요 | permanence hid/vis · v11 late placement |
| E | 가림 사건 채굴 (encoder 대응이 끊겼다 이어지는 창 가중) | 재등장 시점 표현을 맞히려면 가려진 동안 들고 가야 | permanence | 채굴 규칙이 판독과 같은 도구에 기댄다 | — (3b 판을 만들 자리로 정본이 지목) |
| F | 역사 이용 대조 (전체 문맥 vs 앞부분 뺀 문맥의 오차 차) | 문맥 앞부분 정보를 남기게 | 지속 · 영속 | 두 forward, 이득 0 인 창에서 무의미 | — |
| G | 대조 동역학 (InfoNCE) | A 와 같되 붕괴를 대조로 막음 | closure | 음성 설계, 배치 크기 의존 | — |
| H | **합성 규칙** `F_k = F∘…∘F` | 전이 연산자의 합성 일관성 | 전이 (3a) | k 걸음 one-shot 이 따로 필요 (prefix 팔이 이미 있다) | closure 간격 |

원 문서의 읽기: 규칙을 미는 힘은 C · D · H 에 있다 (A · G 는 일관성만, B · E 는 표적 선택). 셋 다 물리를 넣지 않고 **"걸음 크기가 달라도 · 마지막 관측이 없어도 · 걸음을 나눠도 같은 미래"** 를 요구한다. 추천 조합 = 되먹임 + B 를 기본으로, 그 위에 D 먼저 (구현 최소, permanence 직격), 다음 C, H 는 prefix ↔ AR 을 잇는 대조.
⚠️ H 와 가장 가까운 선행은 Semigroup-JEPA (2609.10464) 다. 이 논문은 rollout 손실을 encoder 까지 넣고 오차를 closure + operator mismatch 로 나눈다. 우리 쪽에서는 **반례** (표현 기하가 병목) 로 분류돼 있다 (`LITERATURE_POSITIONING_2026-09-25.md` §6). §5-3 을 볼 것.

### 출처
`auto_research/paper/PAPER_STORY_2026-10-08.md` §2-4 · `auto_research/paper/DESIGN_CHOICES_2026-09-25.md` §1 · §3 · §3-b · §3-c · §3-d · `auto_research/paper/RETHINK_PREDICTOR_ROLE_2026-09-29.md` §0-c · §4 · `auto_research/Archive/I_SCENE_CAUSAL_BOTTLENECK_2026-09-25.md` · `auto_research/Archive/VERIFY_ISCENE_2026-09-25.md` · `auto_research/Archive/VERIFY_ISCENE2_2026-09-26.md` · `auto_research/Archive/P3_CLOSURE_TEST_2026-09-25.md` · `auto_research/Archive/VERIFY_AR_2026-09-27.md` · `auto_research/Archive/COPY_HARD_SUBSET_2026-09-29.md` · `auto_research/Archive/LITERATURE_POSITIONING_2026-09-25.md`

---

## 4. 이미 시도된 처방 후보와 결과

### 4-1. 처방별 결과

| 처방 (어디서 나왔나) | 무엇을 했나 | 결과 | 등급 | 학습? | 원 문서 |
|---|---|---|---|---|---|
| **미래 전용 목표** (DC2) | Ariel prefix · full (scratch, SSv2 + K400) | 팬 · v3 reach 1.5 배, SSv2 4.06–4.17. **EK100 2.17 = 릴리즈**, permanence .46–.47, IntPhys 1 70.00 / 71.67 (< 복사 85.0), v11 late 가림 칸은 릴리즈처럼 바닥 (late 정지 51.9 / 51.2) | 확정 | 사용자 (Ariel) | `ARIEL_CHECKPOINTS.md` §5 · `AR_PREDICTOR` §2-1 |
| **되먹임 · rollout 손실** (V-JEPA 2-AC 레시피, action 없음) | Ariel AR | reach 가 **네 판 모두 가장 짧다** (팬 3.24 · SSv2 2.18 · EK100 1.75 · v3 4.24). 슬롯 1 부터 붕괴, permanence .47, α < 0 (arc · ledge). IntPhys 1 90.56 이지만 블록 공간 복사가 83.33 | 확정 (VERIFY_AR · VERIFY_AR40RD) | 사용자 (Ariel) | `AR_PREDICTOR` · `ARIEL_CHECKPOINTS.md` §5-2 |
| 창 단위 문맥 + 미래만 되먹임 (ctx_ar) | Ariel C' ep16–18 | 팬 3.94 · SSv2 2.77 · perm .44 · IntPhys 1 64.4 · v11 61.0, 빈 문맥 쌍 21 % 에서 "물체 있음" 쪽으로 편향 | 시사 (ep16, 40 epoch 판 재확인 필요) | 사용자 | `AR_PREDICTOR` §2-7 · `VERIFY_R3` |
| **큰 변위 커리큘럼** (ORAL (iii) = DC1(a)) — 단계 M | 릴리즈 post-FT, K400 40k, 3 epoch, T1 팬 증강 vs T0 고정 창 | 팬 R50 4.83 → T0 6.34 → **T1 10.23**; SSv2 2.70 → 3.38 → 3.55; EK100 2.11 → 2.17 → 2.20. 짝 Δ 는 전부 CI 가 0 을 뺀다 (T1 − T0 팬 +3.9, SSv2 +0.17, EK100 +0.02) | 확정 (2 차 검증, 표현 정정) | Claude 가 돌림 (09-25, 사용자 승인) | `I_SCENE` §5-4 · `configs/training/reach_k400_postft.yaml` |
| 큰 변위 커리큘럼 — 단계 N | Ariel ep43 post-FT (prefix 규칙), N1 팬 증강 vs N0 고정 창 | 팬 R50 6.5 / 6.3 / **8.7**; SSv2 4.0 / 3.8 / 3.8; EK100 2.1 / 2.1 / 2.1; IntPhys 1 in-loop 0.65 → 0.67 (두 팔 같음) → **팬 평가에서만 든다** | 1 차 | Claude (09-25) | `I_SCENE` §5-1 · `configs/training/reach_k400_prefix.yaml` |
| **움직임 정렬 조회 bias** (DC1(b), `kind: mret`) — 단계 O | 마지막 문맥 토큰마다 속도 head v_k, 미래 질의 → key logit 에 β·exp(−‖s − (k + v_kΔt)‖²/2σ²). v · β 만 학습 (17k 파라미터) | **무효** — `mret_beta` 0.494–0.498 (초기값 0.5), 속도 head bias ±0.002 → O1 ≈ N1 (팬 8.80 vs 8.68, SSv2 3.75 vs 3.83, EK100 2.14 vs 2.14). 최적화 실패라 판정 보류. 재시도 안: head 전용 lr · flow 보조 감독 · β 고정 (사용자 학습) | 무효 (판정 보류) | Claude (09-25) → 재시도는 사용자 | `I_SCENE` §5-2 · `DESIGN_CHOICES` §3 단계 O · `src/models/mret_predictor.py` |
| **진짜 출처 조회 보조** (학습 없음, 개입) | 진짜 출처 key logit +β (β 1/2/4) | 릴리즈 팬 4–6 칸 +0.03 / +0.07 / **+0.22**, R50 4.8 → 6.2; Ariel +4 +0.22 / +0.43; 반대 방향 +4 는 −0.05; 실영상 bias4 (flow 출처 — 미래를 본 상한) SSv2 R50 2.7 → 3.3, Ariel 4.0 → 6.3 | 충분성 확정 (VERIFY_ISCENE) | 아님 | `I_SCENE` §4-1 · §4-2 |
| 질의 평행이동 (qshift, 학습 없음) | 미래 질의 RoPE 를 출처 − 자기 칸만큼 이동 | EK100 릴리즈 4–6 / 6–9 칸 .03 / .02 → .47 / .45. 단 팬 근거리 −0.12 ~ −0.25 손상, 먼 거리는 base 근거리의 53–72 %. "위치를 맞춰 주면 한계가 사라진다" **철회**. 인과 추정판 (qshift_cv) 은 무효 | 시사 (2 차 검증 강등) | 아님 | `I_SCENE` §4-6 · `VERIFY_ISCENE2` §3 |
| **등속 외삽** (ORAL (i) 운반형의 학습 없는 판) | (가) SC2v2: 등속 오라클 자체를 predictor 대신 판독 · (나) I_SCENE: 등속 외삽 출처를 attention bias 로 (bias4_cv) | (가) SSv2 2–4 칸에서 릴리즈가 오라클보다 **낮다** (Ariel 은 3–6 칸에서 오라클보다 높다) · (나) 실영상 −0.05 ~ +0.01 **무효** (추정 출처 일치 45 %), v3 합성에서는 유효 (+0.05 / +0.06) | (가) 1 차 (적대 검증 전) · (나) 확정 | 아님 | `SC2V2_FLOW_REACH_SSV2_2026-09-25.md` §3-1 · `I_SCENE` §4-2 |
| **닫힘 학습** (ORAL (ii) = DC3) | — | 학습 run 없음. 근거는 행동 측정뿐 (P3 앵커 · I_SCENE §4-5 closure) | ★ | — | `ORAL_DIRECTION_2026-09-25.md` §4 · `DESIGN_CHOICES` 단계 P |
| **도메인 데이터 post-FT 넷** | 릴리즈 predictor 를 v11 · IntPhys 1 · IntPhys 2 · Predictor_v1 로 post-FT | 학습 도메인 안에서만 오른다: v11 post-FT v11 75.83 → 91.35 인데 IntPhys 1 88.89 → 77.22, IntPhys 1 post-FT 93.89 인데 v11 78.53. 문맥에 없던 운동 변화 (가속 · 감속 · 중력) 를 미래에 넣게 된 predictor 는 **없다** | 확정 | 사용자 · Claude (09-10 ~ 09-17) | `z_training/RESULTS_2026-09-19.md` · `TrainingEffects/README.md` §0 |
| Ariel epoch 늘리기 | ep 5 → 43 판독 | 팬 R50 4.26 → 6.54, SSv2 3.49 → 4.02 (ep19 이후 평탄, 시사), **EK100 2.17 → 2.17 (불변 확정)**; 팬 포화는 lr cosine 끝과 구분 안 됨 | EK100 확정 · 팬 미결 · ⚠️ 재현 불가 (ep5–43 삭제) | 사용자 (Ariel) | `I_SCENE` §5-3 |

⚠️ 등속 외삽 두 줄은 모순이 아니다. 조건이 다르다.

| | SC2v2 등속 오라클 | I_SCENE bias4_cv |
|---|---|---|
| 무엇 | 학습 없는 등속 외삽을 **그 자체로 예측** 으로 두고 flow 유사 정답과 비교 | 등속 외삽 출처를 predictor attention 에 **bias 로** 넣음 |
| 판 | SSv2 677 clip, flow · encoder 합의 쌍 2,031 | SSv2 677 · EK100 800 · v3 588, 합의 토큰 |
| 결과 | 2–4 칸에서 릴리즈보다 낫다 | 실영상 무효, v3 유효 |
| 등급 | 1 차 (적대 검증 전) | 확정 |
| 읽기 | 운반형 predictor 의 가능성 (ORAL §4 (i)) | 학습 없는 출처 추정으로는 DC1 을 못 채운다 → 출처 추정은 학습된 모듈 (DC1 본문) |

### 4-2. 학습 없이 확인된 것 vs ★

| 확인됨 (학습 없음, 개입 · 판독) | ★ (학습이 있어야 답이 남) |
|---|---|
| 진짜 출처를 알려 주면 먼 내용이 나온다 — encoder 는 병목이 아니다 (충분성 확정) | 출처 추정을 **학습한** 모듈이 그 이득을 인과적으로 (미래를 보지 않고) 가져오는가 (DC1(b) 재시도 · where 자기지도) |
| 진짜 관측 한 걸음은 넷 (릴리즈 · Ariel · N1 · pv1) 이 같다 · 자기 예측은 앵커가 못 된다 | 닫힘 학습 · 굴림/렌더 분리가 closure 를 ≥ 0 으로 만드는가 |
| 마지막 관측에서 가려진 내용은 같은 칸 비가림 대비 약 1/3 만 남는다 (네 predictor 공통) | 마지막 관측 dropout (D) 이 permanence > .53 을 만드는가 |
| encoder 천장 h 는 가속을 반영한다 (α ≈ +1.2 ~ +1.5) | 시간 척도 합성 (C) · 합성 규칙 (H) 이 α > 0 을 만드는가 |
| 등속 외삽은 실영상 출처를 못 맞힌다 (45 %) | copy-hard 가중 (B) 이 copy-hard Δ 를 올리는가 |
| — | **사건 (3b)** — 후보도 없음 |

### 출처
`auto_research/Archive/I_SCENE_CAUSAL_BOTTLENECK_2026-09-25.md` §4-1 · §4-2 · §4-5 · §4-6 · §5-1 ~ §5-4 · `auto_research/Archive/VERIFY_ISCENE2_2026-09-26.md` · `auto_research/Archive/SC2V2_FLOW_REACH_SSV2_2026-09-25.md` · `auto_research/paper/ORAL_DIRECTION_2026-09-25.md` §4 · `auto_research/paper/DESIGN_CHOICES_2026-09-25.md` §3 · `auto_research/Archive/AR_PREDICTOR_2026-09-27.md` · `auto_research/Archive/VERIFY_R3_2026-09-29.md` · `z_training/ARIEL_CHECKPOINTS.md` · `z_training/RESULTS_2026-09-19.md` · `z_research/TrainingEffects/README.md`

---

## 5. 방법 절이 지켜야 하는 것

### 5-1. 정본 §2-4 의 네 줄

| # | 규칙 (정본 원문 요지) | 왜 (근거) |
|---|---|---|
| 1 | **첫 줄은 "Ariel 과 무엇이 다른가"** — §3 표의 어느 후보를 왜 골랐는지 | Ariel 이 이미 "관측만 · frozen encoder · 새 predictor" 다 |
| 2 | **궤적 외우기와 구분** — 학습 도메인 밖 (held-out 장면 · 운동 · 데이터셋) 에서 잰다. 합성 물리로 학습하면 testbed 판은 도메인 안이 되므로 합격선은 실영상 · 다른 testbed 에서 본다 | v11 post-FT 는 운동 조건마다 궤적 2 개를 재생한 것이었다 (`TrainingEffects/README.md` §5 단서 1, `z_research/IntPhysGenV11/Archive/PAPER_STORY_2026-09-22.md` §4-3) |
| 3 | **판정은 예측이 필요한 칸에서** — 모든 점수에 복사 대비 Δ, copy-hard, 위치만 다른 쌍. VoE 전체 점수는 "떨어지지 않는다" 용도만 | IntPhys 1 은 복사만으로 85.0 · copy-hard 에서 릴리즈 · prefix · full 은 우연 (`COPY_HARD_SUBSET`) |
| 4 | **intuitive physics 의 판정 = 행동** — 보이는 것 · 안 보이는 것 · 없는 것 3a / 3b 대응표 | 내부 귀속 ("중력을 안다") 금지. 조작적 정의 = 네 능력 (`STATE_EVOLUTION_CAPABILITIES_2026-09-19.md`, 3 전이를 3a / 3b 로 나눈 것은 10-08) |

### 5-2. 하네스 질문 (정본 §6) 가운데 학습에 걸리는 것

| 질문 | 학습 실험에서의 형태 |
|---|---|
| 2 결과가 어느 쪽이어도 서나 | §3-b 표처럼 팔마다 "늘면 / 안 늘면" 문장을 미리 적는다 |
| 4 보이는 것 / 안 보이는 것 / 없는 것 | 합격선 8 칸을 축별로 따로 보고한다. 평균 하나로 합치지 않는다 |
| 5 복사 기준선 | 모든 칸에 복사 대비 Δ. AR 처럼 표적 공간이 다르면 그 공간의 복사 (`copy_blk`) 를 쓴다 (AR IntPhys 1 90.56 vs copy_blk 83.33) |
| 6 학습이면 | action 없이 관측만인가 · 도메인 밖에서 재나 · Ariel · post-FT 와 무엇이 다른가 — 셋 다 방법 절 첫 문단에 쓴다 |
| 7 표본 크기 | v11 조건 × 위반 672 쌍 / k 168 / 방향당 16; v11_realistic k 칸 15 쌍; ledge 120 쌍; IntPhys 1 180 쌍 (SE ≈ 3 pt) |
| 8 학습된 자 | 새 predictor 의 미래를 자로 읽으면 held-out 정밀도 · encoder 대조 · 파라미터 없는 교차검증을 같이 낸다. 자가 바뀌면 α 의 부호까지 바뀐다 (§2 ⚠️) |
| 9 합성에서만 서나 | 합격선 1 순위는 실영상 칸 (EK100 reach · 실영상 permanence). 문맥 끝 가림은 corner case 가 아니다 — 움직이는 내용의 SSv2 46 % · EK100 51 % 가 릴리즈 reach 밖에 있다 (`I_SCENE` §4-3 C, 1 차) |

그 밖에 원 문서들이 남긴 운영 규칙:
- 긴 run 은 smoke 단계에서 "무엇이 로드됐는가" 를 assert 한다. EK100 AR ep40 run 에서 `predictor_checkpoint` 키가 넘어가지 않아 실은 릴리즈가 돈 일이 있다 (`AR_PREDICTOR` §2-9, 10-01 정정).
- predictor `kind` 는 state_dict 키가 같아서 config 에 안 주면 조용히 full attention 으로 돈다. 로더가 `arch.kind` 를 확인하게 한다 (CLAUDE.md §3-1).
- 학습 데이터는 가능 변이만 쓴다. 불가능 변이를 넣으면 위반을 정상으로 가르친다 (`z_training/README.md`).

### 5-3. 비교 기준선 넷 + 반드시 넘을 팔

문헌 조사의 검증 규칙은 이렇다: "기준선 넷 (복사 · 등속 · 릴리즈 · 2-AC 식 무액션 predictor), **점수가 아니라 진단 시그니처 소거로 판정**" (`LITERATURE_POSITIONING_2026-09-25.md` §4). 같은 문서는 Registers (ICLR 2024 Oral) 를 방법 규범으로 든다 — 분석 → 기제 → 최소 개입 → 다중 모델 순서이고, 새 predictor 는 진단 시그니처가 사라지는 것을 보여야 한다.

| 기준선 | 우리 쪽 구현 | 이미 있는 값 (예) | 원 문서 |
|---|---|---|---|
| 복사 (마지막 문맥 latent) | `S_copy = mean_t \|LN(z)_{문맥 마지막} − h_t\|`; AR 공간은 `copy_blk` | IntPhys 1 85.0 · v11 73.2 · IntPhys 2 55.5 (C 42) | `TrainingEffects/README.md` · `ARIEL_CHECKPOINTS.md` §5-2 |
| 등속 외삽 | flow 등속 오라클 (SSv2) · v3 GT 등속 (α = 0 의 기준) | SSv2 2–4 칸에서 릴리즈보다 높음 (1 차) | `SC2V2` §3-1 · `AR40_READOUT` |
| 릴리즈 predictor | V-JEPA 2 ViT-H 릴리즈 | 합격선 표 "릴리즈" 열 | §2 |
| action 없는 2-AC 류 | **Ariel AR 팔이 바로 이것이다** (V-JEPA 2-AC 레시피, action 없음) | reach 최단 · α < 0 · closure 붕괴 | `AR_PREDICTOR` |
| (반드시 넘을 팔) Ariel prefix ep45 · full ep40 | 미래 전용 · block-causal | 합격선 표 "Ariel" 열 | `ARIEL_CHECKPOINTS.md` |

### 5-4. 선점 위험과 우리 차이 (문헌 등급 — 인용 전 원문 확인 필요한 것 표시)

| 선행 | 무엇 | 위험 | 우리 차이 (원 문서) | 원 문서 |
|---|---|---|---|---|
| Garrido et al. **2601.05230** (FAIR 2026-01, "Learning Latent Action WMs In The Wild") | frozen V-JEPA 2-L + 실영상 새 predictor | **선점 (구성)** | 복사 귀무 · reach · permanence 를 재지 않았다. 한계 문장 "This representation space was not designed with prediction in mind" → 우리 delta = 복사 귀무 · reach/permanence · encoder 정보 국소화 | `LITERATURE_PREDICTOR_ROLE_2026-09-29.md` §0 |
| HERA (2608.05523) | frozen V-JEPA 2 predictor + 3.00M memory adapter, IntPhys 2 Main 52.57 → 54.35 | **선점 위험 (동기 문장 동일, latent 쪽 최근접)** | 가려진 물체 readout · 복사 기준선이 없다. ⚠️ 그 기준선 52.57 은 V-JEPA 2 논문 57.5 와 프로토콜이 다르다 (원문 확인 필요). ⚠️ "memory" 프레임은 우리 쪽에서 금지 | `LITERATURE_STATE_WORLD_MODELS_2026-09-26.md` · `LITERATURE_POSITIONING` §6-5 |
| Loci-Looped (2310.10372) | percept gate + 가림 중 99.2 % 상상 (보일 때 91.1 %) | 선행 (관측 → 상상 전환 처방) | 물체 중심 구조라 우리 "안 하는 것" (slot · 물체 중심) 과 갈린다. ⚠️ venue (ICANN 2024) 미확인 | `LITERATURE_POSITIONING` R3 |
| VGGT-World · VLWM · Flow-JEPA (2026) | frozen latent 의 exposure bias 처방 (flow-forcing · 지평 직접 예측 · 궤적 flow matching) | closure 처방 선점 | frozen-encoder 설정에서 "teacher-forced 정확도가 rollout 으로 옮겨지지 않는다" 를 잰 문장은 없다 | `LITERATURE_PREDICTOR_ROLE` §5-2 |
| G-SWM · MCnet · DrNet · SDC-Net | what / where 분리 | 선행 (§3-c ② 의 원칙) | 장면 단위 · frozen latent | `LITERATURE_PREDICTOR_ROLE` |
| DINO-WM · DINO-world · PLDM · UWM-JEPA · Causal-JEPA · JEPA-WMs | frozen encoder + 새 predictor 가 표준 | "new predictor 는 그 자체로 새롭지 않다" | **진단 시그니처를 없앨 때만 기여** (P-25 결론) | `LITERATURE_POSITIONING` §0 · §6-1 |
| Semigroup-JEPA (2609.10464) · GRWM (2510.26782) · JEPA-WMs | 병목은 표현 기하 | **반례** — 새 predictor 가 실패하면 뒤집힐 결론 | §3-d H 와 가장 가깝다. encoder 를 얼린 채 합성 규칙을 거는 것이 차이다 (이 장의 해석) | `LITERATURE_POSITIONING` §6 |
| Nayebi et al. 2305.11772 | frozen encoder + CTRNN 46–49 % vs No Dynamics 19–25 % | 선행 (복사 기준선 설계 동일 · encoder 충분 + dynamics 필요) | — | `LITERATURE_POSITIONING` |
| StEvo-Bench (2603.13215) · WRBench (2606.20545) | 생성형에서 관측이 끊긴 뒤의 진화 실패 | 개념 선점 ("최초" 문장 금지) | JEPA latent predictor · 통제 물리 · 국소화 | `LITERATURE_POSITIONING` §0 |

### 5-5. 목표 절에서 특히 쓰지 않을 문장 (정본 §7 중)

| 쓰지 말 것 | 대신 |
|---|---|
| "action-free 학습으로 해결된다" | Ariel · post-FT 반례가 있다. 방법은 차이를 대야 한다 |
| "post-FT 가 intuitive physics 를 배웠다" | 도메인 안 궤적 prior |
| "목적함수 때문이다" | discussion 한 문단까지. 원인으로 걸지 않는다 (`RETHINK` §2 도 "해석, 원인 검정 아님") |
| "기억 (memory) 문제 · state 를 잃는다" | 정보는 남아 있다. 기억이 필요 없는 곳 (가림 없음) 에서도 가속을 안 넣는다 |
| "AC 는 post 여야 효과가 있다" (사실로) | 가설. §7 |
| "predictor 가 X 를 못 만든다" | p probe 98–100. → "맞는 미래로 놓지 않는다" |

### 출처
`auto_research/paper/PAPER_STORY_2026-10-08.md` §2-4 · §6 · §7 · `auto_research/Archive/LITERATURE_PREDICTOR_ROLE_2026-09-29.md` · `auto_research/Archive/LITERATURE_STATE_WORLD_MODELS_2026-09-26.md` · `auto_research/Archive/LITERATURE_POSITIONING_2026-09-25.md` · `z_research/context_encoder_analysis/Archive/STATE_EVOLUTION_CAPABILITIES_2026-09-19.md` · `z_training/README.md` · `auto_research/Archive/AR_PREDICTOR_2026-09-27.md` §2-9 · `auto_research/Archive/SC2V2_FLOW_REACH_SSV2_2026-09-25.md`

---

## 6. 분업

| 무엇 | 누가 | 근거 |
|---|---|---|
| 새 predictor 학습 (어느 후보로 · 언제 · 자원) | **사용자** | `DIRECTION_2026-09-25.md` §0 (09-26 사용자: "학습 … 은 사용자가 한다"), 정본 §2-4 · §5-8 |
| 합격선 · 판정 기준 · 설계 근거 (이 장) · 채점 · 적대 검증 | Claude | 같은 곳 ("설계 · 예측 · 판정 기준은 DESIGN_CHOICES §3 에 적어 두기만") |
| 합격선 ★ 채우기 — Ariel 세 팔 (prefix ep45 · full ep40 · AR ep40) 을 v11_realistic · ledge · gravity_realistic 에 | Claude (GPU 소량) | 정본 §5-3 |
| ledge · wall 다중 미래 세트 설계 (3b 판) | Claude (설계) · 렌더 | 정본 §5-4 |
| §3-b 선행 준비 둘 (copy-hard 가중이 고르는 창 · where 표적 사전 계산) | Claude | `DESIGN_CHOICES` §3-b (실행 기록 없음) |

⚠️ 09-25 의 단계 M · N · O 는 사용자 승인 아래 Claude 가 돌린 학습이었다 ("자원이 되면 노드에 데이터 풀어서 학습", `DESIGN_CHOICES` §3). 09-26 사용자 결정 이후 학습 run 은 사용자 몫이다 (`I_SCENE` §5 단서). auto_research 의 자율 운영 규칙도 같다 — 계획 · 공식은 문서에 적고 바로 돌리되, 묻는 것은 새 학습 · 커밋뿐이다 (MEMORY `feedback-autonomous-oral-push`).

### 출처
`auto_research/paper/DIRECTION_2026-09-25.md` §0 · `auto_research/paper/PAPER_STORY_2026-10-08.md` §5 · `auto_research/paper/DESIGN_CHOICES_2026-09-25.md` §3 · `auto_research/Archive/I_SCENE_CAUSAL_BOTTLENECK_2026-09-25.md` §5

---

## 7. AC — 이 predictor 위의 post-training (스토리 초안 → `09b_AC_STORY.md`)

> 사용자 (2026-10-10): "아직 AC가 우리가 도달할 이상은 아니야" → 같은 날 풀이: "이게 아직 최종이 아니라는거라서 AC 스토리도 일단 작성을 해줘야".
> AC 는 논문의 **작은 subsection** 이다 (10-08). 스토리 · 근거 · 실험 설계 · 열린 결정은 **`09b_AC_STORY.md`** 에 초안으로 있고, 아직 최종이 아니다.

| 항목 | 내용 | 원 문서 |
|---|---|---|
| 주장 (★ 가설, 근거 0) | 관측만으로 intuitive physics 를 배운 predictor (이 장의 목표) 가 먼저여야 AC 학습이 효과가 있다 — AC 는 scratch 가 아니라 post-training | 정본 §0 · §2-5 |
| 이 장과의 관계 | 09b 의 팔 (b) "관측 학습 → AC post" 는 이 장의 predictor 가 있어야 돈다. 그래서 순서는 09 → 09b | 정본 §2-4 · §2-5 |
| 실험 하나 (10-08 설계) | 같은 AC 데이터 · step 예산에서 (a) AC scratch vs (b) 관측 학습 predictor 위 post vs (c) 릴리즈 위 post, AC 데이터 10 / 30 / 100 % | 정본 §2-5 · `z_AC_training/README.md` 머리 |
| 열린 결정 | AC 데이터 (DROID 로컬 없음) · encoder (ViT-H AC) | `z_AC_training/README.md` §2 · §6 |

## 출처

이 장이 인용한 원 문서 (레포 루트 기준). 새 계산은 없다. 수치는 아래 문서와 추출문에서 옮겼다. 합격선 표의 vanish 열 (65 · 3 · 0 / 49 · 37 · 13) 과 v11_realistic 의 2 % · 0 / 120 은 원 문서에서 직접 다시 읽었다.

- 정본 · 방향: `auto_research/paper/PAPER_STORY_2026-10-08.md` (§0 · §1 · §2-4 · §2-5 · §5 · §6 · §7) · `auto_research/paper/DESIGN_CHOICES_2026-09-25.md` (§1 · §3 · §3-b · §3-c · §3-d) · `auto_research/paper/ORAL_DIRECTION_2026-09-25.md` (§4 · §7) · `auto_research/paper/DIRECTION_2026-09-25.md` (§0) · `auto_research/paper/RETHINK_PREDICTOR_ROLE_2026-09-29.md` (§0-c · §2 · §4)
- 장면 인과 · 처방 시험: `auto_research/Archive/I_SCENE_CAUSAL_BOTTLENECK_2026-09-25.md` · `auto_research/Archive/VERIFY_ISCENE_2026-09-25.md` · `auto_research/Archive/VERIFY_ISCENE2_2026-09-26.md` · `auto_research/Archive/SC1_REACH_NOT_HORIZON_2026-09-25.md` · `auto_research/Archive/SC2V2_FLOW_REACH_SSV2_2026-09-25.md` · `auto_research/Archive/P3_CLOSURE_TEST_2026-09-25.md`
- Ariel · 되먹임: `auto_research/Archive/AR_PREDICTOR_2026-09-27.md` · `auto_research/Archive/VERIFY_AR_2026-09-27.md` · `auto_research/Archive/VERIFY_PRESENCE_2026-09-29.md` · `auto_research/Archive/VERIFY_R3_2026-09-29.md` · `auto_research/Archive/AR40_READOUT_2026-09-30.md` · `auto_research/Archive/VERIFY_AR40RD_2026-09-30.md` · `z_training/ARIEL_CHECKPOINTS.md`
- 복사 기준선 · 판정 칸: `auto_research/Archive/COPY_HARD_SUBSET_2026-09-29.md` · `auto_research/Archive/COPY_VS_POSITION_2026-10-07.md` · `z_research/TrainingEffects/ip1_copy/IP1_COPY.md`
- 재학습 · 점수표: `z_research/TrainingEffects/README.md` · `z_research/TrainingEffects/scores/SCORES.md` · `z_training/README.md` · `z_training/RESULTS_2026-09-19.md`
- testbed 칸: `z_research/v11_realistic/README.md` (§2 · §4) · `z_research/context_encoder_analysis/Archive/STATE_EVOLUTION_CAPABILITIES_2026-09-19.md`
- 문헌: `auto_research/Archive/LITERATURE_PREDICTOR_ROLE_2026-09-29.md` · `auto_research/Archive/LITERATURE_STATE_WORLD_MODELS_2026-09-26.md` · `auto_research/Archive/LITERATURE_POSITIONING_2026-09-25.md`
- AC (→ `09b_AC_STORY.md`): `z_AC_training/README.md` · `z_AC_training/Archive/PLANNING_AND_ROBOCASA_2026-09-20.md`
- 학습 config · 코드 (참조만): `configs/training/reach_k400_postft.yaml` · `configs/training/reach_k400_prefix.yaml` · `configs/training/reach_k400_mret.yaml` · `src/models/rollout_predictor.py` · `src/models/mret_predictor.py`
