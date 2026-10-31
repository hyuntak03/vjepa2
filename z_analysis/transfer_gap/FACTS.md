# 전제 사실 고정 — 이식 실패 · PFT 비전이 (R1, 2026-10-30, 검토 [R])

> **질문** — 사용자 전제 넷 (encoder 는 가려진 물체를 읽는다 · predictor 미래에도 정체가 읽힌다 · encoder 판독기가 predictor 에서는 안 읽힌다 · PFT 는 다른 벤치마크로 안 옮겨진다) 이 어느 모델 · 판독기 · 조건에서 맞나.
> **답** — 넷 다 "어떤 판독기로 쟀나" 에 따라 갈린다. 가려진 물체의 모양은 **그 가려진 표현으로 학습한 판독기**에서만 읽히고 (encoder 62–66 %, 출력 72–87 %), 다른 조건에서 학습한 판독기를 걸면 encoder 안에서도 9–21 %, predictor 에서는 0–16 % 다. **이식 실패는 late 만의 일이 아니다** — 보일 때 먼 미래에서도, 같은 시점에서도 일어나고, PFT 뒤에는 크게 줄어든다 (보일 때 24 → 76 %, late 0 → 54 %). PFT 비전이 수치는 산출물에서 다시 셌고 맞다.
> **다음** — Q1 은 "보일 때 먼 미래" 와 "late 안에서 학습한 판독기" 에서, 물체를 **predictor 가 둔 자리**와 **참 자리**를 나눠 잰다. Q2 는 PFT 전후를 자연 대조로 쓰고, vast 의 이미 해 본 개입 둘 (정답 토큰 판독기 감독 · 정체 개념 더하기) 을 먼저 반영한다. Q3 은 채점 안 된 "처음 보는 물체" PFT 체크포인트부터.

모든 수치는 [측정] 이고, 이 노드 (vll5) 의 산출물에서 다시 셌다 (재계산 스크립트와 결과 JSON 은 부록). "문서값" 이라고 적은 것만 원 문서에서 옮겼고 산출물을 다시 열지 못한 것이다.

## 용어

| 용어 | 뜻 |
|---|---|
| 문맥 encoder 출력 z | V-JEPA 2 ViT-H 의 온라인 encoder 에 **문맥 16 장만** 넣은 출력. 미래 프레임을 보지 않는다. shape (8 튜블릿, 256, 1280) |
| 목표 encoder 출력 h | EMA 목표 encoder 출력에 학습 계수 없는 층 정규화. 릴리즈 판은 **32 장을 한 번에** 넣는다 (미래 16 장 포함 — 문맥 끝에 가린 튜블릿도 뒤의 재등장 프레임과 attention 으로 섞인다). AR 판은 블록 (2 장) 마다 따로 넣는다 |
| 예측 p | predictor 가 낸 미래 튜블릿 · 블록 토큰, h 와 같은 1280 공간. 릴리즈는 미래 8 튜블릿을 한 번에, AR 은 한 블록씩 굴린다 |
| 은닉 상태 | predictor 마지막 정규화 뒤 384 차원 값 (출력 투영 전) |
| 튜블릿 · 블록 · 걸음 t | 연속 2 장. 미래는 t8–t15 (걸음 1–8) |
| 판독기 (probe) | 표현에서 정체 · 위치 · 속도를 읽도록 따로 학습한 작은 모듈. **"X 판독기 → Y"** = X 로 학습해 Y 에 그대로 건 것 (이식). "Y 안 판독기" = Y 자체로 학습 · 시험 |
| 결합 정체 | 모양 × 색 + '없음' 의 다지선다 (primitive 49 지선다, 우연 약 2 %; realistic 38 지선다) |
| 다수 클래스 함정 | realistic 물체의 **80 % 가 원래 색 (칠하지 않음)** 이다 (index 에서 셈). realistic 의 색 · 결합 정확도는 이 80 % 에 대어 읽어야 한다 |
| late · 보임 조건 | late = 문맥 끝 k 장 + 미래 앞 k 장이 가려짐. 보임 조건 = visible · early · mid (문맥 끝에서 물체가 보임) |
| PFT | 릴리즈 · vast AR predictor 를 VOBench 가능 영상 (split v2 train, 장면 설정 묶음째 분리) 으로 이어 학습. 새 판 = + gravity train (828 clip, 30 epoch), 옛 판 = VOBench 만 (720 clip) |
| 모델 넷 | 릴리즈 · 릴리즈+PFT · vast AR (상태 되먹임 블록 AR, 자연 영상 70 epoch) · vast AR+PFT. 비교용으로 Ariel AR (40 epoch) 둘 |

---

## 1. 왜 · 어떻게

**왜** — 사용자 전제가 기존 기록과 일부 어긋난다는 브리프 경고가 있었다. 가설 · 실험 에이전트가 같은 숫자를 다르게 읽지 않도록, 전제마다 "어느 모델 · 어느 판독기 · 어느 조건의 수치인가" 를 먼저 고정한다.

**어떻게** — 새 학습 · GPU 없이, 이미 있는 표본 단위 예측 · 특징 · 채점 파일에서 다시 셌다. 판독기 종류를 넷으로 나눴다: (a) 목표 encoder 보임으로 학습 → 다른 곳에 이식, (b) 예측 p 로 학습 (다른 조건에서) → late 에 이식, (c) late 안에서 학습 · 시험 (블록 단위 5 겹), (d) 다른 학습셋 (VOBench 장면, 가림 없음 · 등속) 에서 모델마다 학습한 자.

---

## 2. 전제 (i) — late 에서 encoder 는 가려진 물체의 모양 · 색 · 속도를 읽나

| encoder 출력 | 판독기 학습 → 시험 | 지표 (우연) | 값 | 근거 |
|---|---|---|---|---|
| 문맥 encoder z (문맥 16 장만) | 완전히 가려진 문맥 튜블릿 → 같은 종류의 다른 영상 (5 분할) | 모양 6 (17 %) | **late 62.2 (평균 토큰) / 65.7 (토큰 전부)** · realistic 63.8 / 66.0 · early 74–91 | [측정] |
| 〃 | 〃 | 색 8 (12.5 %), primitive 만 | **late 27.0 / 24.9** · early 39–46 | [측정] |
| 〃 (천장) | 같은 영상의 가림 전 보이는 튜블릿 | 모양 · 색 | late 영상 모양 78.7 (primitive) / 87.0 (realistic) · 색 42.0 | [측정] |
| 목표 encoder h, 마지막 층 (32 장) | **보임 튜블릿 → 가림 튜블릿** (early · mid · late 합) | 결합 (≈2 %) | **9.3–20.9** (4 run) · 보임 천장 89.0–97.4 | [측정] |
| 〃, late 의 가린 튜블릿 t8 만 | 〃 | 결합 | 12.9–25.7 (8 run); 재등장한 t15 는 87–96 | [측정] |
| 목표 encoder h, **16 번째 층** | 보임 → 가림 | 결합 | **0.0–0.7** ('없음' 71–100 %) | [측정] |
| 목표 encoder, 블록 (2 장) 따로 | VOB_readout 자 → VOBench late (primitive, split v2 test) | 결합 49 | 가린 t8 28 %, 재등장 t15 97–98 % | [측정] |

- **모양은 읽힌다 — 단 가려진 표현으로 학습했을 때만.** 문맥 encoder 는 미래를 안 보므로 이 정보는 가리기 전 프레임에서 attention 으로 실려 온 것이다 [측정 + 이론].
- **보임으로 학습한 판독기는 가려진 encoder 표현에서도 거의 안 읽힌다** (9–21 %, 16 층은 0–1 %). 즉 "이식 실패" 는 **encoder 안에서 이미** 보임 → 가림 사이에 있다 [측정]. 이것이 Q1 의 첫 대조군이다.
- 마지막 층 h 의 9–21 % 는 32 장 입력이라 **재등장 프레임에서 섞여 온 몫**을 배제하지 못한다 [이론]. 가림막 판이 물체처럼 읽혀 오답의 절반 이상이 "다른 물체" 다 (원 문서 단서).
- **색은 약하다** — late 27 % (우연 12.5), 보이는 천장도 42 % 뿐이라 이 작은 판독기 (420 영상) 의 한계와 섞여 있다 [측정]. realistic 색은 재지 않았다.
- **속도: VOBench late 의 encoder 쪽은 미측정이다.** 있는 것: v11 late k1–4 에서 문맥 encoder 위 문맥 안 자가 마지막 본 속도를 오차 0.5–0.9 px/슬롯 (참 속력의 3–6 %) 로 읽는다 (문서값). vast AR 평균 풀링 속도 probe 의 "참 미래 블록 h" 는 late 에서도 참 기울기의 0.72–1.03 배를 따른다 — 그러나 이것은 재등장한 미래 블록을 직접 본 값이지 가려진 물체 값이 아니다 [측정].

---

## 3. 전제 (ii) — late 에서 예측 p 에 모양 · 색이 읽히나

브리프의 "릴리즈 late p 는 p 로 학습해도 0–8 %" 와 "vast AR 블록 평균 63–76 %" 는 **판독기 · 라벨 · 표현 · 모델이 모두 다른 두 숫자**다.

| 모델 | 판독기 (학습 데이터) | 표현 | 라벨 (우연) | late · 이동 물체 | 근거 |
|---|---|---|---|---|---|
| 릴리즈 | 목표 encoder h 보임 튜블릿 | 출력 p, attentive | 결합 + 없음 (≈2 %) | **0.0–3.1** (정지 포함 0.0–4.6) | [측정] |
| 릴리즈 | **p, 보임 조건 (visible · early · mid) 의 미래** | 출력 p, attentive | 결합 + 없음 | primitive **1.1–2.6** · realistic 10.1–12.9 | [측정] |
| 〃 (realistic 분해) | 〃 | 〃 | 모양만 (17 %) | realistic **13.8–17.0** (우연 수준), 원색 물체 13–18 · 칠한 물체 0–4 | [측정] |
| 〃 (primitive 분해) | 〃 | 〃 | 모양만 (17 %) | 모양 12.5–16.7 (우연 수준) | [측정] |
| 릴리즈 | p, 보임 조건 · **t15 한 시점** | 출력 p | 모양 (17 %) · 색 (12.5 %) | 모양 **40.6 / 37.5** · 색 primitive 19.2 | [측정] |
| 릴리즈 | VOB_readout 자 (다른 학습셋, 그 모델의 p) | 출력 p | 결합 49 | **1.5** ('없음' 76 %) | [측정] |
| 릴리즈 | **late 안 ridge** (5 겹, 강제 선택, '없음' 없음) | 출력 p 블록 평균 / 참 자리 3×3 | 모양 6 (17 %) | **72 / 57** (primitive) · **87 / 71** (realistic) | [측정] |
| 릴리즈 | 〃 | 은닉 384 블록 평균 / 참 자리 | 모양 6 | 63 / 50 · 76 / 62 (← 브리프의 "63–76") | [측정] |
| Ariel AR 40 · AR-state 40 | late 안 ridge | 출력 블록 평균 / 참 자리 | 모양 6 | 71–78 / 41–57 | [측정] |
| vast AR ep70 | VOB_readout 자 (그 모델의 p) | 출력 p | 결합 49 | **10.0** ('없음' 46 %) · 정지 52.5 | [측정] |
| vast AR ep70 | late 안 ridge | — | — | **미측정** | — |
| 릴리즈 + PFT (옛 판) | VOB_readout 자 | 출력 p | 결합 49 | **81.2** (t9–t15 87–93) | [측정] |
| vast AR + PFT (옛 판) | 〃 | 〃 | 〃 | **79.5** | [측정] |

- **어긋남의 정체** — "63–76 %" 는 vast AR 이 아니라 **릴리즈 · Ariel AR 계열의 은닉 상태**를 **late 안에서 학습한 6 지선다 ridge** 로 잰 값이다. 출력 p 로 재면 더 높다 (릴리즈 72 / 87). "0–8 %" 는 **다른 조건에서 학습한 49 지선다 + 없음 판독기**를 late 에 이식한 값이다 [측정].
- 그러므로 "late 는 정보 부재" 기록 (VOBench README §3-10) 은 **이 판독기로는 안 읽힌다** 까지만 말한다. late 안 판독기는 모양을 읽는다 — **정보는 있고, 다른 조건의 p 와도 형식이 다르다** [측정]. 두 단서: (1) 강제 선택 ridge 는 '없음' 을 고를 수 없다. (2) late 안 판독기가 읽는 정체는 참 자리보다 블록 평균에서 높다 — 물체 자리가 아니라 화면에 퍼진 신호일 수 있다 (vast 의 phys3d 분석도 화면 평균 정체 개념이 물체 칸 개념과 cos 0.31 / 0.47 로 다르다고 했다, 문서값) [측정 + 추측].
- **색은 late p 에서 사실상 미측정이다** — primitive 은닉 상태 ridge 하나만 있다 (릴리즈 late 28 %, 우연 12.5) [측정]. realistic 색 수치는 다수 클래스 함정에 걸린다 (아래 정정 6).

---

## 4. 전제 (iii) — h → p 이식 실패는 정확히 어디서인가

**이식 실패는 late 만의 일이 아니다.** 보일 때 먼 미래 · 같은 시점 · encoder 안에서도 있고, PFT 뒤에는 크게 준다.

| 모델 | 조건 | h 판독기 → p | p 판독기 → p (같은 모델) | h → h | 근거 |
|---|---|---|---|---|---|
| 릴리즈, 튜블릿 판독기 (결합) | 보임 조건 미래 | t8 82.7–90.0 → **t15 13.1–56.5** | t15 45.5–57.1 (primitive) | 89–97 | [측정] |
| 〃 | visible 만 | t15 26.5–51.6 | t15 primitive 47–59 · realistic 74.2 (⚠️ 정정 6) | — | [측정] |
| 〃 | late | 0.0–4.6 | 0.9–14.6 (모양만 보면 우연 수준) | 재등장 87–96 | [측정] |
| 릴리즈, 시점 하나 판독기 (모양 6) | **같은 시점** | **51.1 / 70.4** | 94.5 / 98.2 | 98.1 / 99.3 | [측정] |
| 〃 (색 8, primitive) | 같은 시점 | 38.8 | 79.9 | 86.8 | [측정] |
| 〃 (역방향 p → h) | 같은 시점 | p 판독기 → h 81.1 / 84.3 | — | — | [측정] |
| 릴리즈, VOB_readout 자 (결합 49, test) | 보임 · 이동 | **24.4** (t8 89 → t11 4 → t15 0) | 76.6 | 99.3 | [측정] |
| 〃 | late · 이동 | 0.2 | 1.5 | 88.5 | [측정] |
| vast AR | 보임 · 이동 | **6.9** (t8 51 → t9 3) | 77.6 | 97.6 | [측정] |
| 〃 | late · 이동 | 0.0 | 10.0 | 88.2 | [측정] |
| Ariel AR 40 | 보임 · 이동 | 21.3 (t8 79 → t11 6) | 83.4 | 97.6 | [측정] |
| **릴리즈 + PFT (옛)** | 보임 · 이동 | **75.8** | 98.0 | 99.3 | [측정] |
| 〃 | late · 이동 | **53.5** | 81.2 | 88.5 | [측정] |
| **vast AR + PFT (옛)** | 보임 · 이동 | **75.1** | 95.9 | 97.6 | [측정] |
| 〃 | late · 이동 | **51.2** | 79.5 | 88.2 | [측정] |
| vast 노드 (phys3d, 보이는 표본, 물체 칸, 선형) | AR · 걸음 1 → 8 | 모양 73 → 35 · 색 85 → 31 | 81 → 68 · 92 → 54 | 88–96 · 90–98 | 문서값 |
| 〃 | 릴리즈 · 걸음 1 → 8 | 67 → 37 · 91 → 34 | 91 → 80 · 95 → 71 | 〃 | 문서값 |

- **이식은 한쪽으로 깨진다** — p 판독기는 h 를 읽는데 (81 / 84) h 판독기는 같은 시점 p 를 반쯤만 읽는다 (51 / 70) [측정]. p 에는 h 정보의 일부가 h 와 다른 꼴로 실린다.
- **멀어질수록 깨진다** — 릴리즈는 t11, vast AR 은 t9 부터 h 판독기가 거의 0 이 되는데, 같은 모델 p 판독기는 t11–t15 에도 53–82 % 다 [측정]. vast AR 이 릴리즈보다 더 빨리 깨진다.
- **PFT 는 이식을 크게 고친다** — 보일 때 24 → 76 % (릴리즈) · 7 → 75 % (vast AR), late 0 → 51–54 % [측정]. 이것이 Q2 의 자연 대조다 (단 PFT 는 이식 말고도 많은 것을 바꿨다).
- vast 노드에서 이미 한 Q1 판 (phys3d, 보이는 표본만): 물체 칸 정체 개념 벡터의 **방향 cos 0.86 → 0.45 (AR 모양) · 0.92 → 0.70 (AR 색) · 0.77 → 0.41 / 0.88 → 0.65 (릴리즈)**, **크기 비 0.6–0.8 → 0.18–0.34**; 판독기 가중치 cos (encoder vs 예측) 0.30–0.44 (같은 공간 반쪽끼리 0.54–0.74); 속도는 판독기 가중치 cos 0.06 / 0.07 — "작아진 게 아니라 다른 방향" (문서값). **방향도 틀어지고 크기도 준다** 가 그쪽의 답이었다. 산출물 · 데이터는 vast 노드에만 있다.

---

## 5. 전제 (iv) — PFT 비전이

**평가 행렬 (split v2 test · gravity test · 외부 벤치 공식 규칙, 전부 다시 셈)**

| 모델 | VOBench late p / r (216 쌍) | IntPhys 1 (180 쌍) | IntPhys 2 Main (506) | GRASP level2 | InfLevel macro (continuity) |
|---|---|---:|---:|---:|---|
| 릴리즈 | 58.3 / 65.7 | **88.89** | 54.3 | **70.41** | 58.3 (72.8) |
| 릴리즈 + PFT (새) | **88.0 / 91.7** | **83.33** (−5.6) | 53.6 | **68.75** | 58.3 (72.3) |
| vast AR | 75.9 / 75.9 | 90.0 | 51.8 | 미채점 (산출물 없음) | 58.4 (75.2) |
| vast AR + PFT (새) | **89.8 / 89.8** | 90.0 (±0) | 50.4 | 68.55 | 58.0 (73.9) |

- 다시 센 것: VOBench late (per_block · test block), IntPhys 1 · GRASP · InfLevel (창별 per_window → 공식 규칙), IntPhys 2 (요약 파일, 릴리즈+PFT 271 / 506 = 53.56) — 원 문서와 전부 같다 [측정].
- ⚠️ 릴리즈 IntPhys 2 54.3 · InfLevel 58.3 은 원 문서값 (옛 Benchmarks 산출물, 이번에 다시 열지 않음).
- IntPhys 1 은 **복사 기준선 85.0** 아래로 내려갔다 (릴리즈+PFT 83.33) [측정 + 문서값].

**옛 PFT · 다른 PFT (IntPhys 1 은 다시 셈, 나머지 문서값)**

| PFT 데이터 | 학습 도메인 (분포 안) | IntPhys 1 | 그 밖 |
|---|---|---:|---|
| v11 가능 절반 | v11 held-out 75.83 → 91.35 | 88.89 → 77.22 | — |
| IntPhys 1 train | (IntPhys 1 dev = 같은 생성기) 93.89 | 93.89 | v11 78.53 · **GRASP 70.41 → 55.71** (다시 셈) |
| IntPhys 2 쌍 split | 시험 쌍 59.3 → 59.29 | 85.56 | +encoder LoRA: IntPhys 1 64.44 |
| 2D 물리 (미래 예측 팔) | 2D held-out L1 0.575 → 0.474 | 격자 최고 91.11 (표준 칸 87.78) | 중력 등속 직선 79 → 29 |
| AR + IntPhys 2 위반 라벨 창 (vast) | 그 창 채점 +11.5 | 90.6 | 공식 프로토콜 +0.4 → "학습한 창 배치 안에서만" |
| VOBench (20 epoch) → phys3d (vast) | — | 84.44 | 다른 합성 세트에서 물체를 더 멀리 옮김 (따라 움직인 비율 걸음 3–8 0.20–0.53 → 0.62–1.01), encoder 꼴 속도는 그대로 (문서값) |

- **정정할 만한 단서 하나**: "PFT 가 다른 곳으로 전혀 안 옮겨진다" 는 너무 세다. 외부 벤치 (IntPhys 1 · 2 · GRASP · InfLevel) 로는 안 옮겨지지만, **비슷한 합성 세트 (phys3d) 로는 "계속 움직임" 이 일부 옮겨진다** (vast 문서값). 그리고 같은 VOBench 안에서는 **h → p 이식이 크게 오른다** (§4).
- **이미 있는 Q3 자원**: "처음 보는 물체" PFT 체크포인트 (모양 torus · cone, 메시 2 개, 색 cyan · purple 을 학습에서 통째로 뺌, 30 epoch, 10-21 학습) 가 이 노드에 있고 **채점은 한 번도 안 됐다** [측정: 학습 로그 · eval 폴더 없음].

---

## 6. 전제 정정 — 전제 → 실제 (출처)

1. "encoder 는 가려져도 모양 · 색 · 속도가 다 읽힌다" → **모양만 분명하다** (문맥 encoder, 가려진 표현으로 학습한 판독기 62–66 %). 색은 약하고 (27 %, 우연 12.5, 보이는 천장도 42 %), **VOBench late 속도는 encoder 쪽 미측정**. 보임으로 학습한 판독기를 가려진 encoder 표현에 걸면 9–21 % (16 층 0–1 %) (§2).
2. "predictor 가 만든 (late) 미래에서도 모양 · 색이 읽힌다" → **late 안에서 학습한 6 지선다 판독기로만 모양이 읽힌다** (릴리즈 출력 72 / 87 %); 다른 조건 · 학습셋에서 배운 판독기는 우연 수준 (0–16 %). 색은 거의 미측정 (§3).
3. "encoder 로 학습한 판독기가 predictor 에서 안 읽힌다 (late 의 문제)" → **late 만의 문제가 아니다** — 보일 때 먼 미래 (릴리즈 t8 89 → t11 4 %, vast AR t8 51 → t9 3 %), 같은 시점 (모양 51 / 70 vs p 판독기 95 / 98), encoder 안 보임 → 가림 (9–21 %), p 보임 조건 → p late 에서도 같다 (§4).
4. 브리프 "vast AR late 정체 블록 평균 63–76 %" → **릴리즈 · Ariel AR 40 · AR-state 40 의 은닉 상태 (384)** 값이다; vast AR ep70 의 late 안 판독기는 미측정 (§3).
5. 브리프 "릴리즈 late p 는 p 로 학습한 head 로도 0–8 %" → 그 head 는 **보임 조건 p 로 학습한 이식**이다 (late 안 판독기가 아니다); 값은 primitive 1–3 %, realistic 10–15 % 지만 모양만 보면 두 objset 모두 13–17 % (우연 17) (§3).
6. VOBench README "p 로 학습한 head 는 visible t15 realistic 74 %" · 시점 판독기 "realistic 색 late 75 %" → **다수 클래스 효과**다: realistic 물체의 80 % 가 원색이고, 칠한 물체 정확도는 h 보임 천장에서도 22–48 %, p 보임에서 23–28 % 다; 색 우연은 11 % 가 아니라 다수 클래스 80 % 로 읽어야 한다 (§3).
7. "PFT 해도 다른 벤치마크로 전이가 안 된다" → **외부 벤치는 맞다** (릴리즈+PFT IntPhys 1 88.89 → 83.33, GRASP 70.41 → 68.75, IntPhys 2 54.3 → 53.6, InfLevel 58.3 → 58.3). 단 vast AR+PFT 의 IntPhys 1 은 90.0 그대로 (내려가지 않음), 비슷한 합성 세트로는 운동 이어가기가 일부 옮겨지고, 같은 분포 안에서는 h → p 이식이 24 → 76 % 로 오른다 (§4 · §5).
8. 브리프 "vast 33 커밋, 10-01 ~ 10-27" → 33 커밋은 **09-29 ~ 10-27** 이다 (09-29 커밋 2 개 포함) (지도 문서).

---

## 7. Q1–Q3 가 의미 있는 모델 · 조건

| 질문 | 의미 있는 곳 (정보가 양쪽에 있고 이식만 깨지는 곳) | 의미 없는 · 조심할 곳 |
|---|---|---|
| **Q1** 방향 · 크기 · 꼴 | (a) **보임 조건 · 이동 · 먼 미래 (t10–t15)**, 릴리즈 · vast AR — t10–t15 에서 p 판독기 53–88 % vs h 판독기 0–34 % · (b) **같은 시점** 판독기 (h → p 51 / 70 vs p → p 95 / 98) · (c) **late · 이동, late 안 판독기** (릴리즈 출력 72 / 87 %) vs h 판독기 0 — 재등장한 h (t9 이후, 84–98 %) 가 짝 · (d) encoder 안 보임 → 가림 (같은 질문의 대조군) · (e) PFT 전후 (이식 24 → 76 %) 에서 무엇이 바뀌었나 | **물체 자리 혼동을 먼저 떼야 한다** — p 의 물체는 마지막 관측 근처에 머문다 (anchor). 참 자리에서 재면 "자리 오차" 가 "꼴 차이" 로 보인다 → p 가 물체를 둔 칸 · 마지막 본 칸 · 참 칸을 나눠 잰다. 화면 평균 개념은 물체가 아니라 화면에 퍼진 정체 신호다 (vast) — 물체 칸 기준을 주로. realistic 색은 쓰지 않는다 (다수 클래스) |
| **Q2** 이식을 고치면 영속이 풀리나 | 릴리즈 · vast AR 의 **late · 이동, A 방향 (물체 → 빈) 0 %** 칸 — 표준 채점 (전체 토큰) 을 주 지표로. PFT 는 "이식 ↑ + late ↑" 가 같이 일어난 자연 대조 (단 분포 안) | **이미 vast 에서 해 본 것**: (1) 정답 토큰 판독기를 고정해 predictor 손실로 → 판독기 기준 이식은 성공 (굴린 예측 위치 118 → 25 px, 정체 0 → 86 %) 인데 VOBench late 66–69 · 모양 · 색 55–57, 물체를 옛 자리에 둔 채 (운반 이득 0.09). (2) 문맥 정체 개념을 물체 칸에 더하기 → 물체 칸 채점은 오르나 (이동 모양 50 → 72) **전체 토큰 채점 ±1–2 pt, 이동 사라짐 0 % 그대로** (문서값). 그러므로 "읽히게 만들기" 만으로 영속이 풀린다는 가설은 이미 반례가 둘 있다 |
| **Q3** PFT 는 진행 방향 · 분포를 외웠나 | 릴리즈+PFT · vast AR+PFT (새 · 옛), 옛 PFT 셋 (v11 · IntPhys 1 · IntPhys 2 쌍). **처음 보는 물체 PFT 체크포인트 (미채점)** — 물체 정체를 외웠나 · 궤적을 외웠나를 가르는 첫 판. split v2 test 는 운동 · 속도 칸이 train 과 같다 (다른 물체 · 배경) → "궤적 외우기" 는 split v2 test 로 못 가른다 | 외부 벤치는 leakage 가 다르다: IntPhys 1 은 복사 85.0 (PFT 83.33 은 복사 아래), IntPhys 2 는 encoder 선형 판독도 74 (predictor 몫이 작다), InfLevel gravity · solidity 는 원리상 못 푸는 property → continuity 만. GRASP vast AR 기준선은 산출물이 없다 |

---

## 8. 모르는 것

- VOBench late 에서 encoder (문맥 · 목표) 가 가려진 물체의 **속도**를 읽는가 — 미측정.
- vast AR ep70 의 late 안 정체 판독 (ridge 블록 평균 · 참 자리) — 미측정 (릴리즈 · Ariel 판만 있음).
- late p 의 **색** — 출력에서는 미측정.
- PFT 판독기 수치 (§4) 는 **옛 PFT** (gravity 없음) 판이다. 새 PFT 판은 자로 안 읽었다.
- vast 노드의 phys3d 개념 벡터 결과 (방향 · 크기) 는 산출물 · 데이터가 이 노드에 없어 다시 셀 수 없다.

---

## 부록: 재현

다시 센 스크립트 (CPU, 1 분 안쪽): `z_research/scripts/analysis/transfer_gap_r1_facts.py` → `z_analysis/transfer_gap/results/R1_facts_recount.json` (키 = probe · occluded · late_id · tempgen · readout · velocity · late_acc · bench).

| 절 | 산출물 (다시 연 것) | 원 문서 |
|---|---|---|
| §2 문맥 encoder 가림 | `z_analysis/occluded_identity_probe/result.json` (스크립트 `vobench_occluded_identity_probe.py`) | — |
| §2–§4 튜블릿 판독기 | `z_research/VOBench_v1/vjepa2/probing/results/{e35_s0_user,e35_s1,e10_s0,e10_s1,p_only_*,L16_*}/{primitive,realistic}.json` (표본 단위 `samples`) | `z_research/VOBench_v1/README.md` §3-10 · `vjepa2/probing/README.md` |
| §3 · §4 시점 판독기 | `vjepa2/probing/temporal_generalization/summary.json` | 같은 폴더 `RESULTS.md` |
| §3 late 안 ridge | `z_research/VOBench_v1/bottleneck/late_identity/late_identity.json` (특징 `late_identity/{release,ar40,ar_state40}/shard*.npz` — 1,200 영상 × 걸음 8 × {은닉 384, 출력 1280, 참 h} × {블록 평균, 참 자리 3×3}, 120 MB · **Q1 에 CPU 로 바로 쓸 수 있다**) | `auto_research/Archive/PREDICTOR_BOTTLENECK_2026-10-13.md` §2-6 |
| §3 · §4 VOB_readout 자 | `z_research/VOBench_v1/readout/vobench_{rel,relft,vast70,vast70ft,ar40}/shard*.npz` (예측 확률만) — split v2 test 만 골라 셈 | `readout/README.md` · `vjepa2_pft/README.md` |
| §2 속도 | `z_analysis/ar_layer_velocity_probe/vast70/probe.json` (특징 `shard*.npz` 149 MB: 층 · p · h · 복사, 평균 풀링) | 같은 폴더 `README.md` |
| §5 VOBench late | `exp_results/surprise_c16t32__vobench_v1_{vith,vith_vast_ar_state_ep70}/per_block.json` · `z_training/runs/vobench_{release,vast_ar_state}_postft_g/eval/surprise_c16t32__vobench_v1_split2_test_latest/per_block.json` · `data_csv/vobench_v1_split_v2/index_test.csv` | `z_research/VOBench_v1/Archive/EVAL_MATRIX_2026-10-15.md` |
| §5 외부 벤치 | `cvpr/results/eval/intphys1/{intphys1_dev,grasp_level2,inflevel_*}_<모델>_wgarrido_w{16,32}/per_window.json` → `garrido_rescore.rescore/select` · IntPhys 2 `cvpr/results/eval/intphys2/intphys2_main_vith_release_vobftg/summary.txt` | 같음 · `z_training/RESULTS_2026-09-19.md` · `z_archive_10/13_SINCE_2026-10-10.md` §7 |
| §5 처음 보는 물체 PFT | `z_training/runs/vobench_release_postft_objholdout/` (e5–e30 · latest, eval 없음) · split `data_csv/vobench_v1_objholdout/split.json` · 빌더 `z_training/data/build_vobench_objholdout_index.py` | — |
| §4 · §7 vast 노드 문서값 | `git show origin/vast:z_analysis/docs/IDENTITY_CONCEPT_VECTORS_2026-10-26.md` · `git show origin/vast:z_analysis/README.md` · `z_training/docs/RESULTS_2026-10-24.md` · `TRAINING_STATUS_2026-10-24.md` (vast) | 지도 문서 `VAST_BRANCH_MAP.md` |
| §6 정정 6 | `data_csv/vobench_v1/index.csv` (realistic 가능 · 물체 있는 600 영상의 색 분포: 원색 80.0 %) | — |
