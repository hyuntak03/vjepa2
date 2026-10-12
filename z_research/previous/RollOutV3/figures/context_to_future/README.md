# context_to_future — 문맥이 바뀌면 p 의 미래도 바뀌나

> 해석 · 수치 · 단서의 정본: [`../../Archive/CONTEXT_TO_FUTURE_2026-09-26.md`](../../Archive/CONTEXT_TO_FUTURE_2026-09-26.md). 자는 `identity_r8` (`_decoder.json`).
> **논문 판 (2026-09-27).** `1_` … `4_` 는 Nimbus Roman · 7.0 in · 벡터 PDF + PNG 300 dpi 다. 모든 선에 범례가 있다. 그림 안에는 서술 문장을 넣지 않고, 캡션은 아래에 둔다.

## 공통 표기

| 표기 | 뜻 |
|---|---|
| 점선 | ground truth (진짜 궤적) |
| 실선 | predictor p 의 미래를 자 (`identity_r8`) 로 읽은 값. '있음' 칸만 |
| 회색 파선 | **context encoder z** — context encoder 를 창 전체 (문맥 + 미래 프레임) 에 통과시켜 그 미래 칸을 같은 자로 읽은 값 (p 가 도달해야 할 목표). v11 패널은 **target encoder h** |
| 가로축 | 문맥이 끝나고 몇 튜블릿 뒤 (1 튜블릿 = 샘플 2 개) |
| 세로축 (px) | 문맥에서 마지막으로 본 자리에서, 문맥 끝 운동 방향으로 간 거리 |
| 세로축 (%) | p 가 간 거리 ÷ 진짜 간 거리 (같은 clip 들의 합). 100 % = 진짜만큼 감 |

- **slow / medium / fast** = 문맥 끝의 물체 **속도** 3 분위다 (px/튜블릿). 기준값과 궤적 수는 범례에 있다.
  - 주 창 (문맥 16 프레임) 에서 궤적마다 한 번 정한다. 그래서 창 16 개에서 같은 궤적이 늘 같은 무리에 든다.
- ledge (낙하) · wall (정지) 은 문맥에 없는 사건이라 모든 그림에서 뺐다.

## 어떻게 그렸나

**창 (window).**
- RollOut v3 clip 은 64 샘플이다. 문맥은 **늘 샘플 32 에서 끝난다.** 창마다 문맥을 C 샘플 (32 − C … 31) 앞으로 늘리고, 미래를 P 샘플 (32 … 32 + P − 1) 뒤로 늘린다.
  - C, P ∈ {4, 8, 16, 32} 로 16 개 창이다.
  - 1 튜블릿 = 샘플 2 개이므로, P = 32 는 미래 16 튜블릿이다.
- **창마다 predictor 를 따로 돌렸다** (`rollout3_window_readout.py`: 창 길이 C + P 로 모델을 짓고, context_length = C 로 같은 clip 을 다시 읽음).
- 그래서 모든 창에서 **마지막으로 본 자리 (샘플 30 · 31) 가 같다.** 창이 다르면 predictor 가 본 과거 길이 (C) 와 예측한 길이 (P) 만 다르다.
- `1_` · `2_` · `3_` 의 v3 패널은 **주 창 C = 16, P = 32 하나**다. `4_motion_windows` 가 같은 그림을 16 개 창 전부로 그린 것이다.
- 실제 프레임 선 (z) 은 문맥 encoder 를 **그 창의 프레임 전부 (C + P)** 에 건 것이다.

**`1_motion` (a) 한 곡선.**
1. ledge · wall 을 뺀 84 궤적 (6 법칙 × 14) 의 가능 clip 을 쓴다. 궤적마다 겉모습만 다른 clip 이 14–28 개다.
2. 궤적마다 **문맥 끝 속도**를 구한다. 문맥 샘플 16 개의 **진짜** 위치에 2 차식을 맞춘 기울기이고, 단위는 px/튜블릿이다. p 는 쓰지 않는다.
3. 84 궤적을 속도로 줄 세워 28 · 28 · 28 로 나눈다: slow < 7.1 ≤ medium ≤ 9.0 < fast.
4. clip 마다, 튜블릿마다 "마지막으로 본 자리에서 문맥 끝 운동 방향으로 간 거리" 를 잰다.
   - 점선: 진짜 위치로 잰다.
   - 실선: p 에서 자가 읽은 위치로 잰다. p 가 '없음' 이라 한 칸은 빠진다.
5. **두 단계로 평균낸다.**
   - 먼저 궤적 안에서 겉모습 복사본 (clip) 들을 평균낸다.
   - 그다음 무리 안 28 궤적의 평균을 평균낸다.
   - 즉 "속도 샘플의 평균" 이 아니라 **그 속도 무리에 속한 궤적들의 평균 거리 곡선**이다.

**`1_motion` (b).**
- (a) 와 같은 방식이다. 같은 바닥 (flat_a · flat_v · flat_d) 에서 문맥 속도가 6.5–9.5 px/튜블릿인 궤적만 썼다 (10 · 8 · 6 개).

**`1_motion` (c) · `2_look`.**
- 무리 (화면 위치 1/3 · 배경 · 모양 · 색) 마다 비율 = "그 무리 clip 들의 p 거리 합 ÷ 같은 clip 들의 진짜 거리 합" 이다.
- p 가 '있음' 인 칸만 쓴다.
- 겉모습은 한 궤적 안에서 바뀌므로 궤적 평균을 거치지 않는다.

**`3_identity`.**
- (a): 57-way argmax 가 문맥 물체의 모양 / 색 / 둘 다와 맞는 clip 비율 (전체 clip 중).
- (b–d): p 가 '있음' 인 칸 중 비율이다.

## 캡션 (논문 초안, 영어)

**Figure `1_motion`.** Does the predictor's future change with the context's motion? RollOut v3, context 16 frames, prediction 32 frames; 84 trajectories without events, each shown with 14–28 appearances. Dashed: ground truth; solid: position decoded from the predictor output (tubelets decoded as present only). (a) Trajectories split into terciles of context-end speed. (b) Accelerating, constant-speed and decelerating trajectories on the same floor with matched context speed (6.5–9.5 px per tubelet). (c) Ratio of predicted to true displacement by the object's screen position at context end.

**Figure `2_look`.** The same trajectories rendered with different appearances. Ratio of predicted to true displacement, grouped by (a) background, (b) object shape and (c) object colour. Physically the lines should coincide.

**Figure `3_identity`.** Is the object in the predictor's future the same object? (a) RollOut v3 without occlusion: shape, colour and shape-and-colour accuracy of the decoded predictor output, over all clips; grey dashed: the same decoder on the context encoder z run over the whole window (context and future frames). (b–d) IntPhysGen v11, accuracy among tubelets decoded as present, for objects never occluded, occluded mid-context (k = 1–4, reappearing inside the context) and occluded at the end of the context (k = 1–4). Dotted: chance (1 of 56 combinations).

**Figure `5_acc57_v3`.** 57-class identity accuracy (56 shape × colour combinations + none; the object is visible throughout, so 'none' counts as wrong) of the object decoded from the predictor output, per prediction tubelet, RollOut v3 (all 2,744 possible clips). (a) Context 16 frames, prediction 32 frames; band: 95% bootstrap over 112 trajectories; grey dashed: the same decoder on the context encoder z run over the whole window (context and future frames). (b) Prediction 32 frames with context 4, 8, 16 or 32 frames. (c) Share of clips for which 'none' is chosen. Dotted: chance (1/57).

**Figure `6_shape_colour_v3`.** Shape (a) and colour (b) of the 57-way argmax combination decoded from the predictor output, RollOut v3, context 16 frames, prediction 32 frames; choosing 'none' counts as wrong. Band: 95% bootstrap over 112 trajectories; grey dashed: the same decoder on the context encoder z run over the whole window (context and future frames); dotted: chance.

**Figure `4_motion_windows`.** Panel (a) of `1_motion` for all 16 windows (context C × prediction P, in frames). Speed groups are fixed per trajectory from the C = 16 window.

## 그림별로 보이는 것

| 그림 | 보이는 것 |
|---|---|
| `1_motion` (a) | 빠른 물체를 더 멀리 둔다 (순서 맞음). 멀어질수록 간격이 좁다 |
| `1_motion` (b) | 문맥 속도가 같으면 가속 · 등속 · 감속의 실선이 거의 겹친다. 가속을 거의 반영하지 않는다 |
| `1_motion` (c) | 시작 위치는 상관없다 |
| `2_look` | 배경만 선이 벌어진다. 모양 · 색은 겹친다 |
| `3_identity` | v3 는 8 튜블릿 뒤부터 모양 · 색이 같이 흐려진다. v11 은 문맥 끝에 가려지면 처음부터 낮다 |
| `4_motion_windows` | 창과 무관하게 같은 모양이다 |
| `5_acc57_v3` 57-class 정확도가 1 튜블릿 뒤 80 % 에서 16 튜블릿 뒤 18 % 로 떨어진다. 실제 프레임은 76–86 % 로 평평하다. 문맥 길이 (4–32 프레임) 는 곡선을 거의 바꾸지 않는다. (c) '없음' 을 고르는 비율은 9 튜블릿까지 1 % 미만이고, 10 튜블릿 뒤부터 0–16 % 로 튄다 — 정확도 저하의 대부분은 '없음' 이 아니라 **다른 조합**을 고른 것이다 |
| `6_shape_colour_v3` | 57-way argmax 로 고른 조합의 모양 · 색 정확도. 모양 87 → 38 %, 색 92 → 37 % (1 → 16 튜블릿), 실제 프레임은 86–93 % 로 평평하다. 색은 7 튜블릿까지 실제 프레임과 같다가 그 뒤 떨어지고, 모양은 처음부터 실제 프레임보다 조금 낮다 |

## 자세한 판 (`detail_*`, 탐색용 DejaVu)

같은 질문을 계수로 잰다.
- 속도 β_v · 가속 β_a · 시작 위치 β_x (궤적 bootstrap 구간).
- 겉모습이 설명하는 분산 몫 η² (귀무 = 궤적 안 라벨 섞기).
- 정체 틀림의 종류.
- 실제 미래 프레임을 읽은 z · h 선 (점선) 도 범례에 있다.

## 물러난 판

- `_superseded/first_draft_2026-09-26/` — 계수만으로 그린 첫 판 (너무 어려웠다).
- `_superseded/simple_draft_2026-09-26/` — 쉬운 판의 탐색용 첫 버전 (범례 · 속도 기준값이 빠져 있었다).

## 재현

```bash
R3_DECODER=identity_r8 R3_OUT=identity_r8 /data/hyuntak/anaconda3/envs/vjepa2/bin/python z_research/scripts/figures/plot_context_to_future.py
```
