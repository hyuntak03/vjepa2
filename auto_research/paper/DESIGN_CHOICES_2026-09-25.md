# 새 latent video predictor 의 설계 원칙 — pretrained predictor 의 실패에서 얻은 design choice (2026-09-25 저녁)

> ⚠️ **대체됨 (2026-10-08).** 논문 스토리 정본은 [`PAPER_STORY_2026-10-08.md`](PAPER_STORY_2026-10-08.md) 다. 스토리 부분 (§0 한 문장, '이 문서가 논문 뼈대의 정본이다') 은 대체됐다. §1 DC 표와 §3-b · §3-c · §3-d 의 설계 후보는 **방법 절 (정본 §2-4 목표) 의 후보로 유효** — 전부 미검증 설계안이고, 'Ariel · post-FT 와 무엇이 다른가' 에 답해야 한다.
> 방향 정본: [`DIRECTION_2026-09-25.md`](DIRECTION_2026-09-25.md). 이 문서는 그에 종속된다.

> **프레임 (사용자, 2026-09-25 저녁): 주인공은 새 video predictor 다.** pretrained predictor 분석 (합성 testbed · 실영상 · 개입) 은 그 predictor 의 **design choice 를 뽑는 근거** 다. "synthetic 에서 predictor 가 문제다" 는 결론이 아니다.
> 이야기: 분석 → design choice → 그렇게 만든 predictor 가 실영상에서 낫다 → (확장) action-conditioned fine-tuning 에서도 낫다.
> 이 문서가 논문 뼈대의 정본이다. [`ORAL_DIRECTION_2026-09-25.md`](ORAL_DIRECTION_2026-09-25.md) 의 A′ 는 이 문서 §2 로 흡수한다.
> 등급: 근거 열의 등급을 따른다. 1 차 = 적대 검증 전.

## 0. 한 문장

**latent video predictor 는 마지막 관측에서 먼 내용을 attention 으로 가져오지 못해 일정 거리 뒤에 장면을 멈춘다. 이 조회를 설계 목표로 삼은 predictor 가 실영상 open-loop 예측에서 낫다.**

## 1. Design choice 표

| # | 설계 원칙 | 무엇을 고치나 | 근거 (분석) | 등급 | 검정 (구현 · 상태) |
|---|---|---|---|---|---|
| **DC1** | **움직임 정렬 조회 (motion-aligned retrieval).** 미래 질의가 자기 자리 (복사 key) 가 아니라 **출처 자리** 의 문맥 내용을 강하게 가져오게 한다. 실영상에서는 **출처 추정 (어디서) 과 조회 강도 (얼마나) 둘 다** 모자라고, 학습 없는 등속 외삽으로는 출처를 못 맞힌다 (45 %) → 출처 추정은 학습된 모듈 | 병목 자체 | I-scene (§4-1): 먼 출처로 가는 attention 강도만 올리면 reach 가 세기에 비례해 는다 (release 4–6 칸 +0.22, Ariel 6–9 칸 +0.43). Ariel − release 차의 **약 절반 (거리 층화 44–67 %)** 이 이 강도로 설명 (시사; pooled 95–99 % 는 거리 교락). 규칙 · encoder 는 아님 (규칙은 팬 · SSv2 · EK100 한정; v3 에서는 prefix 규칙이 release 를 −0.06 ~ −0.10 깎음). 자연 영상 (I-scene §4-2): 진짜 출처 bias 는 SSv2 · EK100 · v3 모두 유효, 등속 외삽 bias 는 실영상에서 무효, 방향 선택성 1.5–2 배 (팬 5–9 배) | 1 차 → 적대 검증 1 차 반영 (VERIFY_ISCENE): 충분성 확정 · 매개 시사 | (a) **데이터**: 큰 변위 커리큘럼 — **도메인 안에서만 유효** (단계 N: 팬 평가 R50 6.5 → 8.7, 자연 영상 변화 없음). 데이터만으로는 자연 운동의 출처를 못 배운다. (b) **구조**: 출처 정렬 attention bias — 문맥에서 추정한 변위장으로 미래 질의의 key 를 옮겨 놓는 항 (학습 가능). (c) **구조**: 운반 (latent warp) 모듈 — z_L 을 추정 변위로 옮긴 뒤 predictor 가 다듬는다 |
| **DC2** | **미래 전용 · block-causal 목표**로 학습 | 학습 목표 | Ariel (같은 encoder): reach 1.5 배, 걸음 의존 소멸 (슬롯당 −0.008 vs release −0.035), 먼 출처 attention 3–6 배. 릴리즈 사전학습 마스크는 시간 전체 tube 라 future-only 질의를 배운 적 없음 | 확정 (존재 증명) · 요인 미분리 | Ariel 코드 그대로 (`src/models/rollout_predictor.py`). 새 predictor 의 기본값 |
| **DC3** | **자기 예측을 앵커로 삼는 닫힘 학습** (latent scheduled sampling) | open-loop 사슬 | P3 · P3b: 진짜 관측 한 튜블릿이 앵커면 매 걸음 전진 (0.361, 진실 0.440), 자기 예측은 앵커가 못 됨 (release −0.012, Ariel 도 진실의 절반 0.226). 학습된 predictor 는 토큰 하나는 소화 (tf_last ≈ 0) 하지만 사슬은 누적 | 확정 (행동) | 학습 루프에 되먹임 팔 추가 (predictor 출력을 문맥 끝에 붙여 다음 창을 예측). 미구현 |
| DC4 | **있음 (presence) 을 시간에 따라 잃지 않게** | 지속 | H4 · H23 · M6: 물체다움이 문맥 끝부터의 시간에 따라 준다 (튜블릿당 −4~−6 pt); 역사 토큰은 자리엔 무관, 있음에 기여 | 확정 (행동) · 기전 미상 | DC1–3 이 해결하는지 먼저 본다. 남으면 별도 |
| DC5 | **평가는 reach 곡선으로** (라벨 없이, 변위 축) | 평가 | VoE: 복사 85.0 vs 88.9 n.s. EK100 1 s: predictor 미래를 복사로 바꿔도 −2.5. 둘 다 reach 안쪽만 잰다 | 확정 · 1 차 | `e_scene_reach{,_nat}.py`: 인공 팬 (정확한 정답) · flow 유사 정답 (SSv2 · EK100) · v3 GT |

**DC 사이 순서.** DC2 는 공짜 (기존 코드). DC1 이 논문의 기여다. DC3 은 open-loop 사슬을 위한 것이고, planning (AC) 으로 갈 때 필요하다.

## 2. 근거 요약 (분석 절에 들어갈 것, 각 문서에서 옮김)

1. **읽기.** encoder 입력 z 에는 가시 운동이 선형으로 있다 (SSv2 방향 0.89–0.97; v3 속력 R² 0.95). encoder 는 병목이 아니다 — 출처 내용을 끌어오기만 하면 먼 자리에 맞는 내용이 나온다 (I-scene bias 팔).
2. **reach.** 관측이 끊기면 predictor 는 마지막 관측에서 일정 거리 쪽으로 멈춘다 (고정 걸음 수 R² 0, 거리 R² 0.68–0.84; 장면 전체도 같음: 칸당 −0.08). 합성 · 실영상 (SSv2 flow) · 인공 팬 모두.
3. **병목.** 먼 출처로 가는 attention 강도 (I-scene §4-1). 방향은 알지만 강도가 거리로 15 배 줄고 복사 key 에 진다. 규칙 · encoder · 일반 날카롭히기 아님.
4. **앵커.** 진짜 관측 한 걸음은 넷이 같다. 자기 예측은 앵커가 못 된다.
5. **학습이 늘린다.** Ariel epoch 에 따라 reach 단조 증가 (3.5 → 5.1 칸). 학습 데이터 변위 폭이 원인인지는 묶음 M 이 가른다.
6. **평가.** VoE · 1 s anticipation 은 reach 를 못 본다.

## 3. 새 predictor 실험 계획 (frozen ViT-H encoder 위, 사용자 승인: "자원이 되면 노드에 데이터 풀어서 학습")

| 단계 | 무엇 | 데이터 · 비용 | 판정 |
|---|---|---|---|
| M (재제출, N 뒤; 첫 제출은 병합 부작용 (collator pickle) 으로 실패 → 고침) | DC1(a): release post-FT × {인공 팬 증강, 고정 창} 3 epoch | K400 40k, vll6 8 GPU 약 2 × 1 h (N 실측 1.1 s/step, 625 step/epoch) | 팬 팔의 reach · 먼 출처 attention 이 고정 창 팔보다 크면 데이터 커리큘럼이 유효 |
| N (**끝남**: 팬 증강은 팬 평가에서만 R50 6.5 → 8.7, SSv2 · EK100 은 변화 없음 → DC1(a) 는 도메인 안 한정, I-scene §5-1) | DC2 + DC1(a): Ariel ep43 post-FT (prefix 규칙, `configs/training/reach_k400_prefix.yaml`) × {팬 증강, 고정 창} 3 epoch | K400 40k (SSv2 train 은 vll6 에 없음), `scripts/chain_0925n.sh` | 팬 팔의 자연 영상 reach (SSv2 · EK100) 가 고정 창 팔과 Ariel ep43 을 넘나. 하네스 병합 (kind · prefix_window · sync_masks · predictor_init_checkpoint) 은 2026-09-25 검증됨 (기존 config 병합 결과 동일) |
| O (**끝남 · 무효**: mret_beta 0.494–0.498 (초기 0.5), 속도 head 미학습 → O1 ≈ N1 (팬 R50 8.80 vs 8.68, SSv2 3.75 vs 3.83, EK100 2.14 vs 2.14). 최적화 실패라 DC1(b) 판정 보류; 재시도는 head 전용 lr · flow 보조 감독 · β 고정 — **사용자가 학습**, I-scene §5-2) | DC1(b): **움직임 정렬 조회 bias predictor** (`kind: mret`, `src/models/mret_predictor.py`). 마지막 문맥 블록 encoder 토큰마다 학습된 head 가 속도 v_k (칸/블록) 를 내고, 미래 질의 (b, s) → key (L, k) logit 에 β_l·exp(−‖s − (k + v_k Δt)‖²/2σ²) 를 더한다. v · β 만 JEPA 손실로 학습 (17k 파라미터). Ariel ep43 에서 시작, 데이터 · 증강 · 마스크는 N1 과 같음 (`configs/training/reach_k400_mret.yaml`, `scripts/chain_0925o.sh`) | K400 40k 팬, 3 epoch | ~~N1 보다 reach 가 길고 먼 출처 attention 이 크면 DC1(b) 성립~~ → 판정 보류 (학습 안 됨). 검사: β=0 이면 prefix 판과 출력 동일 (상대 1.6e-7) |
| P | DC3: 닫힘 학습 팔 | 학습 루프 수정 | 자기 예측 사슬 기울기가 진실에 가까워지나 |
| S (★ 사용자 학습) | **state 모듈** (DIRECTION §2-3): 문맥 → 학습 질의 K 개 (slot) 가 encoder 토큰에 cross-attention → S₀; slot 열 위 block-causal 전이 (step embedding, 학습 병렬 · 추론 순차) → S_b; 사전학습 predictor (동결 또는 낮은 lr) 블록 사이 gated cross-attention adapter 로 미래 질의가 S_b 를 읽음; 문맥 토큰 경로는 유지 (정지 배경 = 복사). 손실 = 블록별 JEPA L1 (state 를 굴리므로 multi-step 이 곧 rollout 손실). 새 파라미터 = slot encoder + 전이 + adapter | K400 (+SSv2) | reach · permanence · closure 에서 Ariel · mret 을 넘고, VoE · 1 s anticipation 은 안 떨어지며, EK100 2–4 s · IntPhys 2 가림 · v11 가림 셀이 오르나. 대조: slot 없이 adapter 만 |
| Q | **AC fine-tuning**: 가장 좋은 predictor 를 DROID 로 action-conditioned 학습 (`z_AC_training/`) 하고 planning 검정 | 데이터 (DROID) 준비 필요 | 기존 V-JEPA 2-AC 대비 planning 성공률 · open-loop 예측 오차 |

## 3-b. 2026-09-29 분석에서 나온 다음 학습 팔 (사용자 질문 "motion supervision 을 줘야 하나?" 에 대한 답 — 외부 라벨 없이, 순서대로)

전제 (09-29 분석): 정보는 encoder 에 있다 (속도 · 카메라 운동 R² .69) · 정답 위치를 줘도 절반만 오른다 (qshift) · 손실의 최저점이 복사다 (copy-hard 부분집합: 릴리즈 · prefix · full 은 복사가 틀리는 쌍에서 우연) · 되먹임 손실만으로는 안 닫힌다 (AR · ctx_ar 첫 걸음부터 붕괴) · AR 의 late 정답은 전역 오프셋 (presence-without-placement).
→ **외부 motion 라벨은 안 준다.** 영상 자체에서 나오는 압력 둘 + 구조 하나.

| 순 | 팔 | 무엇 | 라벨 | 판정 (우리 프로토콜) | 어느 쪽이든 서는 문장 |
|---|---|---|---|---|---|
| 1 | **copy-hard 가중** | 창마다 복사 기준선 오차 (encoder 만으로 계산) 로 손실 가중 — 복사가 맞는 창은 낮게, 큰 이동 · 가림 · 재등장 창은 높게 | 없음 | 실영상 reach R50 (SSv2 · EK100), copy-hard Δ | 늘면 "표적 선택이 원인", 안 늘면 "구조 문제" 확정 |
| 2 | **where 자기지도** | predictor 에 변위 · 조회 위치 경로 (mret 류) 를 두고, 표적 = frozen encoder 토큰 대응 (미래 토큰 ↔ 마지막 관측 cos 최대, 합의 토큰만; GT 와 .87–.95 일치) | 없음 (encoder 에서 도출) | reach (특히 4–6 칸), qshift 상한과의 간격 | qshift 상한에 닿으면 "어디" 가 병목, 못 닿으면 "옮기는 계산" 이 병목 |
| 3 | **상태/렌더 분리** | 굴러가는 상태 (렌더 안 됨) 를 따로 두고 predictor 는 상태를 읽어 encoder 공간으로 그림; 되먹임은 상태로만 | 없음 | closure (roll − tf → 0), permanence (hid/vis 비 > .5), EK100 2–3 s Δ | 닫히면 DC 확정, 안 닫히면 "되먹임 공간" 가설 기각 |

- 분석 쪽 선행 준비 (Claude): (1) copy-hard 가중이 실제로 어떤 창을 고르는지 (이동 크기 · 가림 분포, K400/SSv2 학습 창 표본) — GPU 소량; (2) 팔 2 의 표적 (encoder 대응) 을 학습 창에 미리 계산해 두는 스크립트.
- 단서: 셋 다 학습 항목 (사용자). 팔 1 은 지금 학습 코드 (`z_training`, prefix_window) 에 가중치 한 줄이면 된다. 팔 3 은 새 predictor 구조.

## 3-c. 분석에서 나온 모델 (2026-09-30, 사용자 질문 "어떤 모델을 만들어야 하나")

**한 줄.** frozen ViT-H 위에, **관측 공간이 아니라 상태 공간에서 한 걸음씩 굴리고**, 내용은 마지막 관측에서 **"어디서" 를 계산해 가져오며**, 출력은 encoder 공간으로 **그리기만** 하는 predictor. 목표는 latent prediction 그대로 (외부 라벨 없음), 데이터는 실영상, 손실 가중은 복사가 못 푸는 창에.

| 부품 | 무엇 | 분석 근거 | 대조 팔 (검정) |
|---|---|---|---|
| ① 상태 s_t (렌더 안 됨) | predictor 블록의 투영 전 은닉 (또는 별도 K×D 슬롯). 되먹임은 **s 로만** | closure: 렌더 출력을 되먹이면 첫 걸음부터 붕괴 (AR · ctx_ar). RSSM 류가 굴러가는 이유 | ctx_ar 그대로 vs `rollout_ctx` 되먹임을 s 로 (코드 몇 줄) |
| ② where 경로 | 미래 칸 s 마다 조회 위치 u(s) (변위 · 속도장) 를 상태에서 예측하고, 내용은 마지막 관측 토큰에서 그 자리로 가져온다 (mret 류: attention logit 에 위치 편향) | reach: 정보는 encoder 에 있고 자리만 틀림; 출처 힌트 하나로 +.2~.3; qshift 상한. EK100 미래 변위의 67–75 % 가 전역 카메라 운동 → 전역 항 + 국소 항 | where 경로 on/off; 표적은 encoder 토큰 대응 (자기지도) 으로 보조 손실 |
| ③ 렌더 | s_t → encoder 공간 (LN h) 한 걸음. one-step 유지 | 한 걸음 자체는 잘 한다 (tf reach ∞) | — |
| ④ 역사 | 문맥은 창 단위 z (prefix 팔) + 상태에 축적 | permanence: 블록별 문맥 (AR) 은 k=2 최저; 창 단위여도 상태가 없으면 절반 | ①+④ vs ① 만 |
| ⑤ 학습 표적 선택 | 창마다 복사 오차로 가중 (copy-hard ↑) + 큰 걸음 (4 블록 앞) 혼합 | 복사가 손실의 최저점 (copy-hard 에서 릴리즈 · prefix · full 우연); 걸음당 이동 ≤ 1.5 칸이면 복사로 풀림 | 가중 on/off |
| ⑥ 되먹임 노출 | rollout 손실 (있음) + 상태 되먹임 | AR 은 rollout 손실 8 걸음에도 붕괴 → ① 없이는 ⑥ 만으로 안 됨 | — |

**사다리 (싼 순서).** (1) ctx_ar 되먹임을 s 로 (①) — 1 팔, 코드 최소. (2) copy-hard 가중 (⑤) — 데이터 한 줄. (3) where 경로 (②, mret 류 + encoder 대응 보조 손실). (4) 큰 걸음 혼합. 판정은 전부 우리 프로토콜 (실영상 reach · permanence · closure, copy-hard Δ, 그다음 EK100 2–3 s · IntPhys 2 가림).
**안 하는 것.** slot attention · 물체 중심 구조 (장면 단위 유지, 사용자 09-25), flow 라벨, encoder 학습, 목적함수 교체. 가장 가까운 선행 = Garrido 2026 (frozen V-JEPA 2 + 새 predictor, 복사 귀무 · reach 없음) · RSSM (상태 공간 전이) · mret (where).

## 3-d. "state 가 규칙을 배우는 손실" 후보 (2026-09-30, 사용자 요청: 후보를 더) — 전부 라벨 없음 · 실영상 · 예측 목표 안

| | 후보 | 무엇을 강제하나 | 분석의 어느 실패에 닿나 | 장점 | 위험 · 단점 | 새로움 |
|---|---|---|---|---|---|---|
| A | **굴림 = 읽기 일관성** (상태 JEPA) `\|s_roll(t+k) − sg(s_read(t+k))\|` | 관측 없이 굴린 상태가 관측했을 때의 상태와 같다 | closure | 직접적, 계산 싸다 | 정지 상태로도 만족 (렌더 손실 필수); RSSM·BYOL 과 같음 | 낮음 |
| B | **먼 지평 + copy-hard 가중 렌더** (일관성 항 없음) | 복사로 안 풀리는 표적을 맞힌다 | reach · permanence | 가장 순수한 "video prediction" | 여전히 조회로 풀 수 있는 창이 남으면 효과 작음 | 낮음 (데이터 선택) |
| C | **시간 척도 합성** `F_{12fps}∘F_{12fps} ≈ F_{6fps}` (같은 clip 을 두 프레임률로, 상태 전이가 합성돼야) | 전이 연산자가 "걸음 크기" 에 맞는 속도 규칙을 담는다 | 전이 (속도 지속) · reach | 라벨 없음, 자연스러움, 정지 상태로는 못 속임 (두 척도의 렌더가 달라야 하니) | 프레임률 증강 구현, fps 혼재 데이터 (K400) 는 이미 있음 | 중 |
| D | **마지막 관측 dropout** (문맥 끝 m 블록을 지우거나 마스킹하고도 같은 미래를 맞히게) | 마지막 관측만으로 못 풀게 해서 상태 (역사) 에 의존하게 | permanence · 역사 경로 | 구현 한 줄 (증강), "temporal dropout" 은 encoder 강건성 문헌에 있음 | 너무 세면 학습 불안정; m 스케줄 필요 | 중 (predictor 학습에 쓰는 건 새로움) |
| E | **가림 사건 채굴** (encoder 대응이 끊겼다 다시 이어지는 창을 찾아 가중) | 재등장 시점의 표현을 맞히려면 가려진 동안 들고 있어야 | permanence | 실영상에 가림이 많다 (손 · 물체) | 채굴 규칙이 encoder 대응에 의존 (판독과 같은 도구) | 중 |
| F | **역사 이용 대조** (전체 문맥 vs 앞부분을 뺀 문맥의 예측 오차 차이를 손실에 넣어, 역사를 쓰면 이득이 나게) | 문맥 앞부분의 정보를 상태에 남기게 | 지속 · 영속 | "predictive information" 계열 | 두 forward, 이득이 0 인 창에서 의미 없음 | 중 |
| G | **대조 동역학** (굴린 상태 vs 읽은 상태 InfoNCE, 음성 = 다른 clip · 다른 시점) | A 와 같되 붕괴를 대조로 막음 | closure | 렌더 없이도 붕괴 안 함 (CPC) | 음성 설계, 배치 크기 의존 | 낮음 |
| H | **합성 규칙** `F_k = F∘…∘F` (k 걸음 한 번 = 1 걸음 k 번) | 전이 연산자의 합성 일관성 | 전이 | 물리 없이 "규칙" 다움이 가장 명시적 | k 걸음 한 번 predictor 가 따로 필요 (one-shot 팔이 이미 있음!) | 중 |

**읽기.** "규칙" 을 밀어 주는 힘은 C · D · H 에 있다 (A · G 는 일관성만, B · E 는 표적 선택). 셋 다 물리를 안 넣고 예측 목표 안에서 **"걸음 크기가 달라도 · 마지막 관측이 없어도 · 걸음을 나눠도 같은 미래"** 를 요구한다 — 즉 규칙 = 척도 · 결측 · 합성에 대한 불변성.
**추천 조합.** 기본 = 상태 되먹임 + B (먼 지평 · copy-hard). 규칙 항 = **D (마지막 관측 dropout)** 먼저 (구현 최소, permanence 직격), 다음 **C (시간 척도 합성)** (전이 규칙), H 는 one-shot 팔 (prefix) 과 AR 을 잇는 대조로. A 는 안정화 보조로만.
**각 항의 판정.** D → permanence hid/vis 비 · v11 late placement; C → reach (특히 SSv2 4–6 칸) · CONTEXT_TO_FUTURE 의 가속/등속/감속 분리; H → closure 간격; 전부 우리 프로토콜.

## 4. 쓰지 않을 문장

- "synthetic 에서 pretrained predictor 가 문제다" — 합성은 측정 도구다.
- "predictor 가 X 를 못 만든다" · "목적함수 때문이다" (금지 표현).
- "reach 는 fps · 걸음 수와 무관한 고정 거리" (release 는 걸음 몫이 섞임).
