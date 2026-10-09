# 연구 방향 정본 (2026-09-25 밤, 사용자 결정) — 실험이 이 문서를 벗어나면 되돌린다

> ⚠️ **대체됨 (2026-10-08).** 논문 스토리 정본은 [`PAPER_STORY_2026-10-08.md`](PAPER_STORY_2026-10-08.md) 다. 스토리 (§0-b 논지 · §1 목표 · §2 스토리 · §3 여섯 질문) 는 새 정본으로 바뀌었다. **그대로 유효**: §0 분업 (분석 = Claude, 학습 = 사용자), §0-b 의 memory · state 금지와 'world knowledge' 경고, §4 근거 현황 표의 수치 · 등급, §4-2 문헌, §5 드리프트 방지 (단 '인간 비유는 첫 문장과 discussion 에만' 은 P1 이 intro 첫 문단이 되며 바뀜). 주 평가 (EK100 · IntPhys 2 · Ego4D · EgoExo4D) 는 정본 §3 결정 대기 2.

> 이 문서는 **사용자가 정한 방향**을 적은 것이다. README · DESIGN_CHOICES · ABSTRACT 는 이 문서에 종속된다.
> 새 실험을 제안하거나 결과를 해석할 때 §3 의 질문 여섯 개에 답이 되어야 한다. 답이 안 되면 그 실험은 하지 않는다.

## 0. 분업 (2026-09-26 사용자)

- 논문 = **분석 → 학습**. Claude 가 맡은 것은 **분석 절을 완벽하게** 만드는 것: 측정 · 개입 · 적대 검증 · 그림 · 근거 지도 · 리뷰 대응. 결과가 어느 쪽이든 문장이 서는 실험만.
- 학습 (state 모듈 · rollout predictor · EK100/IntPhys 2/Ego4D/EgoExo4D) 은 **사용자가 한다.** 설계 · 예측 · 판정 기준은 `DESIGN_CHOICES` §3 에 적어 두기만 한다.
- 분석의 완성 기준: STORY_EVIDENCE 의 모든 행이 "확정" 또는 "시사 (단서 명시)" 이고 ★ 는 학습 항목뿐; 세 그림이 검증된 수치로 채워짐; 리뷰 패널의 분석 공격에 답이 있음.

## 0-b. 논지의 이름 (2026-09-29 사용자 — 방향 변경이 아니라 강조점 정리)

- **역할 분리가 논지다.** encoder 는 관찰을 **표현**하는 기관, predictor 는 **예측**하는 기관. 세계에 대한 지식은 예측에 쓰이므로 **predictor 에 있어야 한다.**
  → **frozen encoder (ViT-H) + 새 predictor**, 새 predictor 는 **실영상만으로 video prediction 목표로** 학습. 합성은 측정 도구.
- 기존 결과는 이 프레임에서 세 근거로 그대로 쓰인다:
  1. *encoder 를 얼려도 된다* — 정보는 encoder 에 있다 (z 속도 R² 0.996 · probe 98–100 · 미래 encoder 출처 판독). 병목은 encoder 가 아니다 (I-scene, Sobal 대비).
  2. *지금 predictor 는 world model 이 아니다* — 마지막 관측에서 가져와 채운다 (reach 2–5 칸 · permanence 1/3 · closure 실패).
  3. *video prediction 목표를 그냥 걸면 되는가 → 아니다* — Ariel 세 팔 (full · prefix · AR) 이 정확히 "frozen encoder + 새 predictor + 실영상 미래 예측" 인데 reach 는 도메인 안에서만 늘고 EK100 은 그대로, permanence 그대로, AR 은 자기 되먹임에서 무너진다 → **목표만으로는 안 되고 설계 선택이 필요하다** (방법 절의 자리).
- 바뀌는 것: 주인공의 이름 ("state 모듈" → "world knowledge 를 가진 predictor"). 안 바뀌는 것: 목표 · 분업 · 분석 축 (reach · permanence · closure · 네 능력) · 여섯 질문.
- ⚠️ (09-30) **"memory · state" 프레임 금지** (사용자). 실패는 기억이 아니라 **규칙 미적용** 이다 — 정보는 encoder 에 있고, 기억이 필요 없는 상황에서도 가속을 안 넣는다. RETHINK §0-c.
- ⚠️ "world knowledge" 는 그대로 쓰면 리뷰어가 잡는다. 본문 기준은 측정 가능한 정의 (네 능력 · reach · permanence · closure) 로 쓰고 "물리 엔진을 가졌는가" 는 묻지 않는다 (2026-09-19 결정 유지).

## 1. 목표 (한 줄)

**video prediction 사전학습으로, 관측이 불완전해도 관측한 state 를 진화시켜 근미래를 anticipate 하는 — 더 인간 같은 — latent world model 을 만든다.**

- 주 평가: **EK100 (action anticipation) · IntPhys 2 · Ego4D · EgoExo4D** 의 anticipation · state evolution 성능. action sequence 예측을 포함한다.
- 합성 데이터 (RollOut_v3 · IntPhysGen v11 · 인공 팬) 는 **분석용 testbed 이자 state evolving 판정용**이다. 목표가 아니다. 합성에서만 성립하는 결론은 쓰지 않는다.
- action conditioning (planning) 은 **그다음**이다. 인간 (유아) 은 action 없이도 근미래를 anticipate 한다 → state 가 관측 없이 스스로 진화하는 predictor 를 먼저 만들고, 그 위에 action 을 얹는다.

## 2. 스토리 (사용자 초안 → 다듬은 것, `ABSTRACT_v0_2026-09-25.md`)

1. JEPA 계열은 latent world model 의 유력 후보다. 근거로 인용되는 것: 직관물리 (VoE) 가 창발한다, encoder+predictor 로 anticipation SOTA.
2. **그런데 predictor 는 state 를 진화시키지 않는다.** 마지막 관측에서 가져와 채울 뿐이다 (reach · permanence · closure 세 지표). 정보는 있는데 배치가 없다. 쓰이는 벤치마크는 이 영역을 재지 않는다.
3. 그래서 (i) state evolution 을 직접 재는 평가 프로토콜을 만들고, (ii) **관측 없이 스스로 진화하는 state 를 가진 latent world model** 을 만든다. 사전학습 predictor 는 렌더러로 살린다 (이미 잘하는 것: 정보 보유 · 가리키면 그림).
4. action conditioning 은 그 위에.

**"inpainter 니까 당연하다" 에 대한 답** (허수아비를 치지 않기 위해 반드시 같이 쓴다):
- 그 predictor 를 근거로 "물리를 안다 · 미래를 예측한다" 는 주장이 쓰이고 있고, 그 평가들이 state 를 요구하지 않는다는 것 (복사 기준선 · predictor 치환) 을 우리가 처음 보였다.
- 미래 예측으로 **다시 학습해도** (Ariel, scratch) 같은 한계가 남는다 → 목표만의 문제가 아니라 구조의 문제다. (rollout 학습 팔이 확정/반증)
- 정보는 있다 (probe 98 %) · 가리키면 그린다 (개입) → 없는 것은 "들고 가는 state" 다.

## 3. 실험을 걸기 전 여섯 질문

1. 이 실험이 §2 의 어느 문장을 세우거나 무너뜨리나.
2. 결과가 어느 쪽이어도 문장이 서나 (한쪽만이면 확증편향).
3. 합성에서만 성립할 위험이 있나. 실영상 (K400 · SSv2 · EK100) 재현 계획이 붙어 있나.
4. "inpainter 라 당연하다" 반론을 넘는가.
5. 주 평가 (EK100 · IntPhys 2 · Ego4D · EgoExo4D) 에 어떻게 닿나.
6. 새 구조 · 새 학습이 필요하면, 분석이 먼저 닫혔나 (사용자: "너무 어렵게 가지 말고 분석 먼저 철저히").

## 4. 지금까지 세운 것 · 남은 것

| 문장 | 상태 |
|---|---|
| 마지막 관측에서 가져와 채운다: reach 2–5 칸, 실영상 움직이는 내용 절반이 밖. **실영상에서는 거리와 경과 시간이 둘 다** 줄인다 (연속 판독 §4-7; ⚠️ τ 의존 — τ 0.02 는 슬롯 우세, 0.1 은 둘 다 0). 질의를 출처 자리에 세우면 먼 적중이 오른다 (§4-6; ⚠️ 2 차: base 근거리의 53–72 %, 팬 근거리 −0.2 비용, "한계가 사라진다" 철회) | 1 차 · 적대 검증 2 차 반영 (VERIFY_ISCENE2) |
| 가림 뒤 미운반 (permanence) | 합성 물체 확정; **장면 · 실영상 프레임 (K400 팬 + 가림 띠) 1 차**: 가려졌던 내용은 보이는 내용의 절반만 복원, 가림막을 그림, 네 predictor 동일 (I-scene §4-4) |
| 자기 예측은 관측 대신 못 씀 (closure) | 확정 (행동, v3) + **실영상 장면 1 차** (I-scene §4-5): 자기 예측 되먹임은 one-shot 보다 낫지 않고, 미래 예측 학습 predictor 는 오히려 −0.12 ~ −0.22 |
| 정보는 있고 가리키면 그린다 | 확정 · 1 차 |
| 벤치마크가 이 영역을 안 잰다 (VoE 복사 IntPhys 1 · **IntPhys 2** (w16/32/48 모두), EK100 치환 **1 · 2 · 3 s 모두 Δ ≈ +2**) | 확정 · 1 차 (IntPhys 2 · SC5 는 1 차) |
| 미래 예측 학습 (Ariel) 도 같다 | 1 차 |
| **rollout 학습 (AC 구조, action 없음) 도 같은가** — 구조 문제임을 확정하는 결정 실험 | **🆕 2026-09-27 체크포인트 도착** — 사용자가 Ariel 세 팔을 받았다 (`z_training/ARIEL_CHECKPOINTS.md`): full-attention 대조 `full_attn_future_pred_1e_5` (ep40), prefix `block_causal_future_only_1e_5` (ep45, 옛 ep43 대체), **자기회귀 `ar_future_1e_5` (kind `ar`, ep18/40 학습 중, mask token 없이 블록별 되먹임)**. **1 차 답 (09-27 저녁, `Archive/AR_PREDICTOR_2026-09-27.md`): 같거나 더 나쁘다.** AR rollout R50 팬 3.2 · SSv2 2.2 · EK100 1.8 · v3 4.2 (릴리즈 4.8 · 2.7 · 2.1 · 4.7, full/prefix 7.0 · 4.1 · 2.2 · 7.5); 진짜 블록을 되먹이면 9.2 · ∞. 자기 예측을 되먹이면 3 걸음 안에 뭉개진다 → closure 는 되먹임 학습으로도 안 닫힌다. 같은 AR 이 IntPhys 1 90.56 (최고) — 벤치마크 감사의 가장 날카로운 쌍. permanence 도 절반 (hid/vis 비 .47 = full/prefix, 릴리즈 .53 보다 낮음; k=2 최저). 적대 검증 완료 (`Archive/VERIFY_AR_2026-09-27.md`: reach 최단 확정 — 템플릿 · 걸음 크기 산물 아님, 0 칸 정지 내용은 AR 최고, 결손은 움직이는 내용에만). AR 복사 기준선 도착 (블록 공간 복사 83.33, AR 90.56 = +7 vs 릴리즈 +4, 잡음 안). ep38 재판독 완료 — 결론 유지 (팬 3.9 · SSv2 2.2 < 릴리즈; perm .46). **확정.** ⚠️ 단, copy-hard 채점 (COPY_HARD_SUBSET) 에서는 AR 만 v11 late 가림을 맞힌다 (72–99 vs 릴리즈 0–65) — presence vs placement 모순으로 기록, presence 검정 완료 (1 차): presence-without-placement — 재등장 칸에서는 사라진 미래 쪽, 나머지 칸이 정답을 만든다. 모순 해소. ctx_ar ep16 (문맥 창 단위 z + 미래 되먹임): 되먹임 reach · 붕괴 · permanence 가 회복되지 않고 정지 우위만 사라짐 (검증 3 차; "문맥 인코딩은 원인 아님" 은 시사, 40 epoch 재확인 필요). ⚠️ 옛 ep5–43 은 삭제돼 §5-3 epoch 결과는 재실행 불가. **사용자 학습 항목.** 분석 쪽 대답 (묶음 R): Ariel 은 epoch 에 따라 reach 가 **포화** 한다 (팬 R50 4.3 → 6.5, SSv2 3.5 → 4.0 뒤 평평, EK100 2.2 불변) → "학습 부족" 반론 해소는 **EK100 한정 확정** (⚠️ 2 차: cosine lr 이 ep30 이후 peak 의 ≤ 26 % 라 팬 포화는 스케줄 끝과 구분 안 됨; SSv2 시사) (I-scene §5-3) |
| 데이터 커리큘럼 (팬 증강) 은 도메인 안 한정 | 1 차 (N · M 모두: 팬 6.3 → 10.2, SSv2 +0.2, EK100 +0.0). 미래 전용 post-FT 자체는 release reach 를 Ariel 수준까지 올림 — **팬 · EK100 한정** (SSv2 는 간격의 52 %; §5-4 2 차) |
| state 모듈 (렌더러 유지 + state adapter) | 설계만. 분석이 닫힌 뒤 |
| EK100 장거리 · IntPhys 2 · Ego4D · EgoExo4D | 미실행. Ego4D · EgoExo4D 는 데이터 · 하네스 없음 |

### 4-1. 리뷰 2 차 (`REVIEW_v2_2026-09-26.md`) 가 짚은 가장 약한 고리와 대응

- **약한 고리:** "미래 예측으로 학습해도 멈춘다 → 구조 문제" 가 Ariel run 하나에 기대고, epoch 에 따라 reach 가 아직 오르는 중 (3.5 → 5.1 칸). 계속 오르면 "학습 부족" 으로 되돌아간다.
- **대응 (학습 없음, 큐):** 묶음 R = Ariel ep5 · 19 · 30 · 41 × 팬 · SSv2 · EK100 reach (포화 검정). 묶음 P = 연속 위치 판독 + 질의 위치 평행이동 개입 (정답 없는 판 포함). 묶음 Q = 실영상 장면 permanence. SC5 = EK100 horizon sweep.
- **분석 → 방법 다리 (다음):** 출처 bias 상한을 v11 가림 셀 · IntPhys 2 · EK100 2–3 s 에 적용 — "state 가 있으면 여기까지 오른다" 의 상한.
- 근거 지도: `STORY_EVIDENCE_2026-09-26.md`. 그림 초안: `auto_research/figures/story/`.

### 4-2. 문헌 (`Archive/LITERATURE_STATE_WORLD_MODELS_2026-09-26.md`) 이 바꾸는 것

- **아무도 사전학습 V-JEPA 2 predictor 의 open-loop state 운반을 재지 않았다.** 굴리는 latent WM (V-JEPA 2-AC · DINO-world · LeWorldModel) 은 전부 새 predictor 를 학습하고, 릴리즈 predictor 를 zero-shot 으로 쓰는 쪽 (Garrido · IntPhys 2 · EK100 · JFAA · TAP-JEPA · HERA · WMReward) 은 전부 ~1 s 의 one-shot mask-token 질의다. → 우리 측정은 비어 있던 자리다 (허수아비 반론에 대한 문헌 쪽 답).
- **가장 가까운 선행 = HERA** (동결 V-JEPA 2 predictor + 3 M 메모리 adapter, IntPhys 2 Main 52.6 → 54.4). surprise 만 보고, 가려진 물체 판독 · 복사 기준선 · open-loop rollout 이 없다. 우리 state 모듈은 이것과 정면 비교해야 한다.
- **긴장:** IntPhys 2 는 permanence 를 가장 쉬운 조건으로 보고한다 (고정 카메라 > 60 %). 우리 "가려진 내용 미운반" 과 맞추려면 셋 중 하나를 재야 한다 — (H1) 재등장 불일치는 조회만으로 잡힌다 (복사 기준선이 같은 점수), (H2) 우리 실패는 경계 corner case, (H3) 60 % 는 약하다. → **감사 결과 (`Archive/INTPHYS2_PERMANENCE_AUDIT_2026-09-26.md`, 1 차): H1 성립.** IntPhys 2 Main 506 쌍에서 마지막 문맥 latent **복사** 가 release predictor 와 같거나 낫다 (overall 55.5 vs 54.3; permanence 고정 카메라 63.5 vs 59.6; 차이 CI 모두 0 포함). release 의 permanence 는 전 조건 CI 가 50 을 포함 (H3 도 성립). 신호는 사건 봉우리가 아니라 전역 평균에서 나온다. 즉 IntPhys 2 permanence 는 "가려진 내용을 들고 가는가" 를 재지 않는다 — 모순이 아니라 평가 감사 표의 한 줄이 늘었다 (IntPhys 1 85.0 vs 88.9 → IntPhys 2 로 확장). H2 는 metadata 에 사건 프레임이 없어 검정 불가.
- VoE 에 복사 기준선을 둔 논문은 없다. Garrido 의 대조군은 random init · VideoMAEv2 · MLLM 뿐. DINO-world 는 IntPhys/GRASP 를 "sanity check" 로 부른다.
- 인용 금지: 2607.16274 (철회됨). PLM 은 IntPhys 2 모델 목록에 없다.
- Ego4D LTA 는 LLM 이 언어 공간에서 푼다; Ego4D STA 상위는 V-JEPA 2.1 encoder 만 씀; EgoExo4D 에는 공식 anticipation 트랙이 없다 → 주 평가 목록에서 EgoExo4D 는 파생 과제로만 가능 (사용자 결정 필요).

## 5. 하지 말 것 (드리프트 방지)

- "synthetic 에서 predictor 가 문제다" 를 결론으로 쓰지 않는다.
- inpainter 의 실패 자체를 주인공으로 두지 않는다. 주인공은 새 latent world model 이고, inpainter 는 동기 · 대조군 · 렌더러다.
- 인간 · 유아 비유는 첫 문장과 discussion 에만. 본문 기준은 reach · permanence · closure.
- 분석이 닫히기 전에 새 구조를 여러 개 벌리지 않는다 (mret · slot 은 대조군 · 설계안으로만).
- 목적함수를 원인으로 걸지 않는다. α 계열 수치 인용 금지 (CLAUDE.md).

## 6. 실험 사다리 (현재 큐 → 다음)

1. (큐) O 대조군 · M 재평가 · P (연속 위치 판독) · **Q permanence 실영상 장면 판**.
2. rollout 학습 predictor (action 없는 AC 형, K400 40k) → 같은 판 (reach · permanence · closure). §2-2 확정.
3. 분석 문서 정리 + 적대 검증 → 그림 셋 (reach · permanence · 개입).
4. state 모듈 (사전학습 predictor 렌더러 + slot state + adapter) 학습 → 같은 판 + **EK100 장거리 anticipation · IntPhys 2**.
5. Ego4D · EgoExo4D 데이터 · 하네스 → action sequence anticipation.
6. action conditioning (AC fine-tune) → planning.
