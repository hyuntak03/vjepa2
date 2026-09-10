# v11 가림 타이밍 — `k` 장을 문맥 어디에 두는가

> 본 실험 세트는 `../IntPhysGenV11/` 이다. 이건 그 위의 팔 하나.
> 레포 규칙은 루트 `CLAUDE.md`, 논문 뼈대는 `../IntPhysGenV11/Archive/PAPER_STORY_2026-08-31.md`.

---

## 한 줄

**`late` 에서만 무너지고, 무너지는 것은 정보가 아니다.**

| 단계 | 무엇이 바뀌나 | 채점 | `p` 정보 | `p` 자리 |
|---|---|---|---|---|
| no occluder → early | 가림막이 생긴다 | −2 ~ −14 | 그대로 100 | 그대로 98~100 |
| early → mid | 같은 장수, 자리만 뒤로 | ±2 | 그대로 | 그대로 |
| **mid → late** | **가림이 예측 경계에 닿는다** | **−4 ~ −38** | **거의 그대로 86~99** | **15~21 로 붕괴** |

- 총 가림 장수를 맞춰도 격차가 **+20.2 / +17.3pt** 남는다 → **양의 문제가 아니다**
- `early`/`mid` 는 정보도 자리도 그대로인데 채점은 떨어진다 → **신호가 묽어진 것**
- `p` 와 `h` 는 **비가림에서도 이미** 같은 좌표계가 아니다 (`h→p` 57~95)

⚠️ 계단은 **위반마다 다르다.** 영속성만 깨끗한 계단이고, 색은 `early` 에서 이미
떨어진 뒤 평평하다.

## 용어 — 채점 축과 probing target 을 구별할 것

⚠️ 이 문서와 그림의 **`shape consistency` / `colour consistency` 는 채점 축**이다 —
위반 clip 을 정상 clip 과 갈라내는가. probing 의 **`shape` / `color` target**(표현에서
그 속성을 읽어낼 수 있는가)과 **다른 것**이고 수치를 섞으면 안 된다.
같은 조건(등가속+가림)에서 `p` probe 는 shape 을 **98.2%** 로 읽는데
shape consistency 채점은 **54.5%** 다 (둘 다 v11_full, §전수 기록 3·6).

## 설계

문맥 = raw 0–45, 미래 = 48–93 (stride 3). `k` 장을 문맥 **어디에** 둘지만 바꾼다.

| timing | 가리는 raw 프레임 (k=4) | 문맥 숨김 | 미래 숨김 | 총 |
|---|---|---:|---:|---:|
| `early` | 12–21 | k | 0 | **k** |
| `mid` | 21–30 | k | 0 | **k** |
| `late` | 36–57 | k | **k** | **2k** |

- **`late` 는 별도 데이터셋이 아니라 v11 본체다.** 원래 설계가 경계(48) 대칭이라
  문맥 끝 k장 + 미래 앞 k장을 가린다
- `early`/`mid` 는 2026-09-01 에 v11 에 **추가 렌더**된 6개 조건 (`*_early`, `*_mid`)
- 기존 6조건은 프레임이 그대로라(mtime Aug 30) **다시 채점하지 않았다**

---

## 채점 결과 (`surprise_c16t32`, v11_full 21,504 pair, overall **76.00%**, chance 50%)

`v11_full` **한 run** 이다 — 43,008 clip / **21,504 matched pair** / 12조건,
`exp_results/report.json` 의 `verified = true`. 전수는 §전수 기록.

### 4단계 × 위반

| timing | vanish | shape | color | 전체 |
|---|---:|---:|---:|---:|
| `visible` (가림막 없음) | 100.0 | 84.3 | 81.5 | **87.2** |
| `early` | 97.8 | 78.1 | 67.4 | **79.0** |
| `mid` | 95.8 | 76.0 | 68.7 | **78.2** |
| `late` (= v11) | 57.5 | 55.4 | 65.1 | **59.5** |

### 4단계 × 운동 유형

| motion | visible | early | mid | late |
|---|---:|---:|---:|---:|
| `static` | 91.0 | 79.0 | 77.4 | **58.9** |
| `moving_flat` | 91.0 | 83.0 | 84.2 | **60.0** |
| `moving` (ramp) | 79.7 | 75.1 | 73.1 | **59.7** |

**세 운동이 `late` 에서 58.9 / 60.0 / 59.7 로 모인다.** 비가림에서 11pt 벌어져 있던 것이
경계에 닿는 순간 사라진다 — **바닥이 공통이다.**

### 타이밍 × k

| k | early | mid | late | 총 가림 (e/m · late) |
|---|---:|---:|---:|---|
| 1 | 81.7 | 80.5 | 59.6 | 1 · 2 |
| 2 | 79.1 | 80.5 | 58.6 | 2 · 4 |
| 3 | 78.9 | 76.6 | 59.1 | 3 · 6 |
| 4 | 76.4 | 75.4 | 60.9 | 4 · 8 |

**위반별로 k 반응이 다르다** (`fig_timing_k`, 운동 접음):

| 위반 | early k1→k4 | mid k1→k4 | late k1→k4 |
|---|---|---|---|
| Object permanence | 99.1 → 96.4 | 97.3 → 92.0 | 61.3 → 54.5 |
| Shape consistency | 80.6 → 76.2 | 77.6 → 74.4 | 55.0 → 55.2 |
| Colour consistency | 71.2 → 63.3 | 72.2 → 65.3 | 63.1 → **70.8** |

- **모양만 `late` 에서 완전히 평평하다** (55.0 → 55.2). 바닥에 붙어 더 안 내려간다
- **색은 `late` 에서 오히려 오른다** (63.1 → 70.8). 다른 둘과 반대 방향이다
- ⚠️ 이 표는 셀당 504쌍이다. 운동까지 쪼개면 168쌍이라 선 모양만 읽을 것

**영속성의 k 반응은 등가속에서만 나타난다** (전수는 §전수 기록 5):

| 운동 | early k1→k4 | mid k1→k4 |
|---|---|---|
| static | 100.0 → 100.0 | 99.1 → 100.0 |
| moving (flat) | 100.0 → 100.0 | 100.0 → 99.1 |
| **moving (ramp)** | **97.3 → 89.3** | **92.9 → 76.8** |

정지·등속은 문맥을 4장 가려도 100 에서 안 움직인다. 등가속에서만 양이 듣는다.

### 물체 편향 — 이 셋에는 그림이 없다

`late` 에서만 하삼각이 채워지고 `early`/`mid` 는 대각이 살아 있다는 관찰은
**v11 본체 그림**에 있다 (`../IntPhysGenV11/figures/surprise/04_object_order/`).
읽는 법은 `../IntPhysGenV11/Archive/surprising_score/READING_THE_MATRIX_2026-08-31.md`.
이 셋에서는 다시 그리지 않았다 — 같은 결론을 §probing 의 이식이 더 직접적으로 말한다.

### 총 가림 장수를 맞춘 비교 — **총량 교란 제거**

| 총 가림 | early/mid | late | 차 |
|---|---:|---:|---:|
| 2장 | `k=2` **79.8** | `k=1` **59.6** | **+20.2** |
| 4장 | `k=4` **75.9** | `k=2` **58.6** | **+17.3** |

### 단계별 하락폭 (`fig_timing_step` 의 선 꺾임)

| 위반 | visible→early | early→mid | mid→late | 총 |
|---|---:|---:|---:|---:|
| vanish | −2.2 | −1.9 | **−38.3** | −42.5 |
| shape | −6.2 | −2.1 | **−20.7** | −29.0 |
| colour | **−14.1** | +1.3 | −3.6 | −16.5 |

- ⚠️ **"계단" 은 영속성의 이야기다.** 모양은 두 계단을 다 타고, 색은 첫 계단만 탄다
- `early → mid` 는 세 위반 모두 2pt 안쪽이다 — **문맥 안 위치는 무해하다**
- 운동별로 보면 `late` 에서 영속성·모양이 세 운동 모두 50~57 로 모이는데,
  **색만 흩어진 채로 남는다** (55 / 68 / 71). 색은 다른 축을 탄다

---

## 말할 수 있는 것

✅ **문맥 안에서 어디를 가리든 거의 같다** — `early` 79.0 vs `mid` 78.2 (0.8pt).
   위반별로도 2pt 안쪽이다.

✅ **양도 거의 무관하다** — `early` k=1 81.7 → k=4 76.4 (5.3pt).
   `late` 는 k 에 평평하다 (58.6~60.9).

✅ **결정적인 것은 가림이 예측 경계에 닿는가다** — 총 장수를 맞춰도 **+20.2 / +17.3pt**.
   양의 문제가 아니라 **자리의 문제**다.

✅ **문맥만 가리는 것은 비가림에서 8pt 손해에 그친다** (87.2 → 79.0).
   경계에 닿으면 **27.7pt** 떨어진다 (87.2 → 59.5).

✅ **세 운동이 `late` 에서 한 점으로 모인다** (58.9 / 60.0 / 59.7).
   비가림에서 11pt 벌어져 있던 차이가 사라진다.

✅ **색은 다른 축을 탄다** — 가림막이 생기는 순간(visible → early) 14.1pt 떨어지고
   경계에 닿아도 3.6pt 밖에 더 안 떨어진다. 계단은 **영속성·모양의 이야기**다.

## 말할 수 없는 것

❌ **"미래를 가려서" 인지 "문맥 끝을 가려서" 인지 못 가린다.**
   `late` 는 둘을 동시에 한다. 가르려면 **문맥 끝 k장만 가리고 미래는 안 가리는 팔**이 필요하다.
   지금 셋에 없다.

❌ **"늦게 가릴수록 나쁘다" 는 못 쓴다** — `early`(12–21)와 `mid`(21–30)는 사실상 같고,
   `late` 만 떨어지는데 그건 시점이 아니라 경계에 닿기 때문일 수 있다.

❌ 이 결과로 **predictor 의 내부**를 말할 수 없다. 채점 정확도만 잰 것이다.

---

## 해석 — "보이는 프레임이 적어서" 는 아니다

솔깃한 읽기는 **"물체가 보이는 프레임이 적으면 모양·색 정보가 덜 담긴다"** 이다.
**양의 축이 그걸 지지하지 않는다:**

- `k` 를 1 → 4 로 **네 배** 늘려도 `early` 는 81.7 → 76.4 (5.3pt) 밖에 안 떨어진다
- **총 가림 장수를 맞추면** 격차가 그대로 남는다 (+20.2 / +17.3pt).
  같은 장수를 가려도 어디를 가리느냐로 20pt 가 갈린다

→ **몇 장 보이느냐가 아니라 어느 장이 보이느냐다.**
구체적으로 **예측 경계에 붙은 프레임**이 결정한다. 최근성(recency) 효과로 읽힌다.

### ⚠️ 가림막 색 가설은 기각됐다 (2026-09-03)

아래 "가림막이 색면으로 들어온다" 는 가설은 **`v13_black` 으로 검정해 기각됐다.**
가림막을 검정으로 바꿔도 색의 `visible → early` 하락이 −14.1 → **−13.5** 로 같다.
대신 **모양**이 −6.2 → **0.0** 으로 좋아졌다 — 나뭇결 텍스처가 모양 판별을 방해하고 있었다.
전수는 `../IntPhysGenV13_Black_occluder/README.md`.

### 색만은 다르게 읽어야 한다

색은 `visible → early` 에서 **−14.1** 로 가장 크게 떨어지는데, 이건 양도 위치도 아니다.
그 단계에서 바뀌는 것은 **가림막이 장면에 존재하게 된다**는 사실이다.

⚠️ **경쟁 가설**: 가림막은 그 자체로 큰 색면이다. 색 채점이 떨어진 것이
"물체 색 정보가 줄어서" 가 아니라 **가림막 색이 장면 색 통계를 지배해서**일 수 있다.
`index_probe.csv` 에 `occ_color` 가 있으므로 **검정 가능하다** — 가림막 색과 물체 색의
관계에 따라 정확도가 갈리는지 보면 된다. **아직 안 했다.**

## probing — 정보인가 자리인가 (`attn_probe_xfer`, 12조건 108 head)

채점만으로는 "정보가 줄었나 / 자리가 틀렸나" 를 못 가른다. probing 이 그걸 가른다.

### 1. 정보는 어디서도 안 없어진다

`p` self probe (자기 조건 head, held-out, chance 12.5):

| | vis | early | mid | late |
|---|---:|---:|---:|---:|
| shape | 100 | 100 | 100 | 95.2 ~ 98.9 |
| colour | 100 | 99.5 ~ 100 | 98.9 ~ 100 | 86.3 ~ 99.2 |

`env` 는 36칸 전부 100.0. **채점이 떨어지는 것은 정보 손실이 아니다.**

### 2. `late` 에서만 자리가 바뀐다

비가림에서 배운 readout 을 같은 팔의 가림 조건에 건다 (`p`, colour):

| | early | mid | late |
|---|---:|---:|---:|
| static | 98.6 | 98.2 | **15.1** |
| flat | 99.9 | 99.4 | **20.6** |
| ramp | 100.0 | 98.9 | **16.4** |

`z` · `h` 는 `late` 에서도 93~100 을 유지한다. **`p` 만, `late` 에서만** 무너지고
그 붕괴는 `k=0 → 1` 한 칸에서 일어난다.

→ **`early`/`mid` 는 정보도 자리도 그대로인데 채점은 떨어진다.**
   그 둘로는 설명이 안 된다 (§"기전" 으로).

### 3. `p` 와 `h` 는 애초에 같은 좌표계가 아니다

GT 미래에서 배운 readout 을 예측 미래에 걸면 (`h→p`, colour):

| | vis | early | mid | late |
|---|---:|---:|---:|---:|
| static | **94.9** | 84.9 | 77.9 | **12.8** |
| flat | **79.0** | 75.3 | 76.4 | **28.2** |
| ramp | **57.3** | 59.6 | 53.1 | **17.8** |

**가림막이 없는데도 이미 낮다.** `h→h`·`p→p` 는 100 인데 그렇다.
`|p − h|` 의 대부분이 이 좌표 불일치이고, 그래서 margin 이 base 의 0.3% 밖에 안 된다.
(그림은 안 만든다 — `h→h`/`p→p` 는 천장이고 `late` 붕괴는 §2 와 겹친다. 전수는 §전수 기록 8.)

## 기전 — 신호가 묽어진다

채점은 `|p − h(가능)|` vs `|p − h(불가능)|` 의 대소다.
그 차이(margin)는 **base 의 0.08 ~ 3.7% 밖에 안 된다.**

| | base `|p−h|` | margin (% of base) | 효과크기 |
|---|---:|---:|---:|
| no occluder | 0.573 ~ 0.578 | 0.27 ~ 3.73 | 0.76 ~ 1.30 |
| early / mid | 0.578 ~ 0.583 | 0.11 ~ 2.39 | 0.47 ~ 1.17 |
| **late** | **0.584 ~ 0.589** | **0.08 ~ 0.77** | **0.22 ~ 0.44** |

가림이 붙으면 base 는 늘고 margin 은 줄어 **효과크기**가 준다.
정확도는 `margin > 0` 인 비율이므로 **효과크기가 곧 정확도**다 —
12셀에서 **Spearman ρ = 0.986**.

**두 고장이 다르다.**
- `early`/`mid` — 정보도 자리도 그대로. **신호가 묽어진 것뿐**이다
- `late` — 거기에 **자리 변화가 더해진다** (`p` 이식 100 → 17)

## attention knockout — 채점을 나르는 경로는 `mask→ctx` 하나다 (2026-09-03)

predictor 안의 attention 간선을 끊고(frozen, 재학습 없음) 채점 정확도가 얼마나 떨어지는지
쟀다. 12조건 전수 21,504 pair. 명세 39개 = 간선 3종 × (층 창 12개 + 전 층 1개).
창은 **±3 (폭 7)**, 중심 층 0~11 (양 끝은 잘린다: 중심 0 = L0-3, 중심 11 = L8-11).
표기는 `<query>-><key>` 다 — `mask->ctx` 가 "mask 토큰이 문맥 토큰을 못 읽는다" 이다.
도구는 `analysis/attention/knockout/`, 결과는 `exp_results/knockout__v11_full_vith/results.json`,
그림은 [`figures/06_knockout/`](figures/06_knockout/).

| 전 층 끊음 | 정확도 | Δ vs 기준선 75.9 |
|---|---:|---:|
| `mask->ctx` (mask 가 문맥을 못 읽음) | **38.3** | **−37.5** |
| `mask->mask` (미래 토큰끼리) | 74.4 | −1.4 |
| `ctx->ctx` (문맥 토큰끼리) | 76.4 | +0.5 |

### 말할 수 있는 것

- **채점을 나르는 경로는 `mask->ctx` 뿐이다.** `ctx->ctx` 는 어느 층·어느 축에서도 ±1 안이고,
  `mask->mask` 는 shape 에서만 −5 (73.4 → 68.2) 다
- **층 프로파일은 후반에 몰려 있다.** 창 중심 0~4 (L0-3 … L1-7) 는 −10pp 근처로 평평하고,
  중심 5 (L2-8) 부터 떨어져 중심 8 (L5-11) 이후 38 로 포화한다 = 전 층 값과 같다.
  즉 **층 5~11 에서 문맥을 못 읽으면 앞 층에서 읽은 것으로 메워지지 않는다**
- **k 와 무관하다.** k = 1~4 의 곡선이 겹친다 (전 층 41.4~42.5). 채점의 k 무관성과 같다
- **타이밍은 곡선 모양을 안 바꾼다.** visible / early / mid / late 전부 같은 자리(중심 5)에서
  꺾인다. late 는 기준선이 55~60 이라 **떨어질 폭 자체가 작을 뿐**이다 — 절대값으로 읽을 것
- **앞 층 창(L0-3)의 효과는 운동에서만 난다.** static 76.4 → 73.7 인데 moving_flat 79.4 → 63.1,
  moving 71.7 → 62.3. 앞 층의 문맥 읽기가 필요한 것은 **움직임**이다
- **colour 는 앞 층을 끊으면 오히려 오른다** (70.4 → 74.4, late 65.3 → 75.6). 해석은 보류한다

### 말할 수 없는 것 — 읽기 전에

- **38% 는 "정보가 없다" 가 아니다.** chance 아래다 (visible/vanish 는 13.2). 문맥을 못 읽은
  predictor 는 무작위가 아니라 **문맥과 무관한 고정 예측**을 내고, 그것이 불가능 미래(특히
  물체가 사라진 장면)에 더 가깝다. 즉 이 값은 고정 예측의 **외형 편향**이지 능력 척도가 아니다.
  같은 이유로 `mask->ctx` 곡선의 절대값보다 **꺾이는 자리**를 읽을 것
- **개입이지 관찰이 아니다.** 간선을 끊은 predictor 는 학습 때 본 적 없는 입력 위에서 돈다.
  "그 경로가 없으면 이만큼 떨어진다" 까지이고 "그 경로가 정보를 이만큼 나른다" 는 아니다
- **`ctx->ctx` 가 0 인 것은 "불필요" 와 "인코더와 중복" 을 못 가른다.** 문맥 토큰은 이미 ViT-H
  인코더를 통과한 뒤라 predictor 안에서 다시 섞을 것이 없을 수 있다
- **Δ = 0 이 아무 일도 없었다는 뜻이 아니다.** `ctx->ctx@all` 도 예측은 `rel_l1` 만큼 움직인다
  (전수 기록 11 의 `pred_drift` 표). 정확도는 계단 함수다
- **폭 7 창은 12층의 58% 다.** 단일 층 기여가 아니라 "그 7층 어디서도 못 읽을 때" 다.
  ±1 창 결과(v11 본체, `IntPhysGenV11/exp_results/knockout_window__v11_vith`)와 폭이 다르니 나란히 놓지 말 것
- 기준선은 `clean_null`(전부 True 인 마스크로 돈 판) 75.86 이고 공식 채점 76.00 과 0.14pp 다르다.
  `attn_mask` 를 주면 SDPA 커널이 바뀌어 예측이 최대 1.5e-2 움직이기 때문이다. **같은 실행 안 비교만** 한다
- `keep_self=True` — 자기 자신으로의 attention 은 항상 남긴다 (softmax NaN 방지). 결과 json 에 기록됨

---

## v11 본체 주장의 갱신

v11 README 는 "`k=0` 과 `k>0` 사이에서만 끊기고 `k=1…4` 는 평평하다 —
**가려진 시간은 무관하다**" 였다. 이 팔이 그것을 좁힌다:

> **가려진 양도, 문맥 안 위치도 거의 무관하다.
> 무너지는 것은 가림이 예측 경계에 닿을 때뿐이다.**

v11 의 `k` 평평함은 **모든 k 가 경계에 닿아 이미 바닥이었기 때문**이다.
경계를 비켜 가리면(`early`/`mid`) k 반응이 살아나고(81.7 → 76.4) 전체가 79 로 올라간다.

---

## 데이터

```
v11_earlymid   21,504 clip / 5,376 block / 10,752 matched pair
  condition   {static_occlusion, moving_occlusion_flat, moving_occlusion} x {_early, _mid}
  sym_k       1 / 2 / 3 / 4
  violation   vanish / shape / color
late 는 v11 본체의 가림 3조건 (10,752 pair) 을 그대로 쓴다
```

문맥 무결성 전수 감사 **172,032쌍, mismatch 0**
(`data_csv/intphysgen_v11_earlymid/context_integrity.json`).
레지스트리 `configs/protocols/datasets.md` 의 `## v11_earlymid`.
⚠️ **타이밍이 `condition` 에 접혀 있다.** v11 과 합치면 12 조건이라
`fit_groups_sweep: auto` 가 12 그룹을 만든다 — probing 은 쪼개서 제출할 것.

### 폐기된 pilot (기록)

2026-09-01 이전에 `IntPhysGenV11_occlusion_timing_ablation` 이라는 **별도 렌더**로 한 번 쟀다
(6,048 pair, overall 68.11%). **인용하지 말 것** — 그 pilot 은 `early`/`mid` 가 문맥을
**`2k`** 장 가려서 지금 셋(`k`)과 조건이 다르다. 원본 렌더는 v11 병합 시 사라졌고
결과 파일도 남아 있지 않다. `datasets.md` 의 `## v11_occtiming` 은 `available: false` 다.

---

## 그림

전부 `z_research/scripts/figures/plot_v11_timing.py` 하나가 만든다.
**그림 하나 = 주장 하나**이고 폴더 번호가 논증 순서다.
자세한 것은 [`figures/README.md`](figures/README.md).

```
figures/
├── 01_scoring/        무엇이 무너지는가          fig_timing_step  (+ by_k/)
├── 02_probing/
│   ├── 01_self/       정보가 있는가              fig_self_*       (+ by_k/)
│   └── 02_vis_occ/    비가림 -> 가림 이식        fig_vis_occ_*    (+ by_k/)
├── 03_mechanism/      왜 떨어지는가              fig_margin · fig_margin_vs_acc
└── 06_knockout/       어느 경로가 나르는가       fig_ko_layers · fig_ko_drift  (+ by_violation/ by_motion/ by_timing/ by_k/)
```

`06_knockout/` 만 `plot_v11_knockout.py` 가 만든다 (입력이 `results.json` 하나다).

`by_k/` 안의 그림은 전부 같은 판이다 — `x = k(0 = 가림막 없음)` · `선 = 타이밍` ·
`패널 = 운동` · 나머지 고정. 세 폴더를 나란히 놓고 같은 방식으로 읽는다.

⚠️ **probing 수치는 전부 `attn_probe_xfer__v11_full_vith` 한 run 에서 온다.**
그 run 이 상위집합이다 (`z` self + `h` self·`h→p` + `p` self·`p→h`).
`attn_probe` 와 섞으면 같은 self 칸이 108 중 1개에서 6.1pt 벌어진다 (미수렴 head).

## 재현

```bash
# 채점 — 새로 생긴 6조건만
GPUS=4 BATCH_SIZE=16 DECODE_WORKERS=10 \
  bash z_research/scripts/run.sh surprise_c16t32 v11_earlymid vith

# report (채점 + probing 을 하나로)
python z_research/scripts/analysis/report.py \
  --run z_research/IntPhysGenV11_occlusion_timing_ablation/exp_results/surprise_c16t32__v11_full_vith:data_csv/intphysgen_v11_full/index_probe.csv \
  --probe z_research/IntPhysGenV11_occlusion_timing_ablation/exp_results/attn_probe_xfer__v11_full_vith \
  --probe-index data_csv/intphysgen_v11_full/index_probe.csv \
  --out z_research/IntPhysGenV11_occlusion_timing_ablation/exp_results/report.json

# 전수 기록 절 재생성
python z_research/scripts/analysis/timing_md.py --write

python z_research/scripts/figures/plot_v11_occtiming.py \
  --report z_research/IntPhysGenV11_occlusion_timing_ablation/exp_results/report.json \
  --outdir z_research/IntPhysGenV11_occlusion_timing_ablation/figures/surprise
```

```bash
# attention knockout — 8 GPU 로 block 을 8 조각 내 돌리고 merge (전수 43,008 clip, 37분)
bash analysis/attention/knockout/run_sharded.sh            # D=v11_full M=vith N=8 W=7 기본
python z_research/scripts/figures/plot_v11_knockout.py \
  --results z_research/IntPhysGenV11_occlusion_timing_ablation/exp_results/knockout__v11_full_vith/results.json \
  --outdir  z_research/IntPhysGenV11_occlusion_timing_ablation/figures
python z_research/scripts/analysis/knockout_md.py --write   # 전수 기록 11 재생성
```

플롯 스크립트는 실행할 때마다 **run 별로** `cells` 를 재합산해 `overall` 과 대조한다
(0.02%p 넘게 어긋나면 죽는다). 위 표의 타이밍 집계는 가림 조건 cell 만 골라
`condition` 의 `_early`/`_mid` 접미사를 떼고 `occ_timing`(없으면 `late`)으로 묶은 것이다.

## 전수 기록

`timing_md.py` 가 산출물에서 매번 다시 계산한다. 손으로 고치지 말 것.

**overall 76.00%** · n=21504 matched pair · 채점 cell 117 · probing head 108 · chance 50%

### 1. 채점 — 타이밍 × 위반

| timing | permanence | shape | colour | 전체 | n |
|---|---:|---:|---:|---:|---:|
| vis | 100.0 | 84.3 | 81.5 | **87.2** | 5376 |
| early | 97.8 | 78.1 | 67.4 | **79.0** | 5376 |
| mid | 95.8 | 76.0 | 68.7 | **78.2** | 5376 |
| late | 57.5 | 55.4 | 65.1 | **59.5** | 5376 |

### 2. 채점 — 타이밍 × 운동

| timing | Static | Moving (flat) | Moving (ramp) |
|---|---:|---:|---:|
| vis | 91.0 | 91.0 | 79.7 |
| early | 79.0 | 83.0 | 75.1 |
| mid | 77.4 | 84.2 | 73.1 |
| late | 58.9 | 60.0 | 59.7 |

### 3. 채점 — 운동 × 위반 × 타이밍

| motion | violation | vis | early | mid | late |
|---|---:|---:|---:|---:|---:|
| Static | permanence | 100.0 | 100.0 | 99.8 | 70.5 |
| Static | shape | 94.0 | 86.0 | 83.2 | 54.8 |
| Static | colour | 81.8 | 57.9 | 56.7 | 55.4 |
| Moving (flat) | permanence | 100.0 | 99.3 | 99.8 | 52.0 |
| Moving (flat) | shape | 89.9 | 79.6 | 79.0 | 56.8 |
| Moving (flat) | colour | 86.0 | 75.4 | 79.0 | 68.5 |
| Moving (ramp) | permanence | 100.0 | 94.0 | 88.0 | 50.0 |
| Moving (ramp) | shape | 69.0 | 68.8 | 65.9 | 54.5 |
| Moving (ramp) | colour | 76.8 | 68.9 | 70.4 | 71.4 |

### 4. 채점 — k × 위반 × 타이밍

| violation | timing | k=1 | k=2 | k=3 | k=4 | k=0 |
|---|---:|---:|---:|---:|---:|---:|
| permanence | early | 99.1 | 98.8 | 96.7 | 96.4 | 100.0 |
| permanence | mid | 97.3 | 98.5 | 95.5 | 92.0 | 100.0 |
| permanence | late | 61.3 | 57.4 | 56.8 | 54.5 | 100.0 |
| shape | early | 80.6 | 77.8 | 78.0 | 76.2 | 84.3 |
| shape | mid | 77.6 | 77.2 | 75.0 | 74.4 | 84.3 |
| shape | late | 55.0 | 55.2 | 56.1 | 55.2 | 84.3 |
| colour | early | 71.2 | 67.3 | 67.9 | 63.3 | 81.5 |
| colour | mid | 72.2 | 71.8 | 65.5 | 65.3 | 81.5 |
| colour | late | 63.1 | 62.9 | 63.5 | 70.8 | 81.5 |

### 5. 채점 — k × 운동 × 타이밍 (위반 접음)

| motion | timing | k=1 | k=2 | k=3 | k=4 |
|---|---:|---:|---:|---:|---:|
| Static | early | 81.7 | 78.3 | 78.8 | 77.0 |
| Static | mid | 81.0 | 79.0 | 73.9 | 75.7 |
| Static | late | 57.4 | 55.8 | 62.7 | 59.8 |
| Moving (flat) | early | 86.4 | 82.4 | 82.8 | 80.4 |
| Moving (flat) | mid | 87.1 | 85.0 | 82.8 | 81.9 |
| Moving (flat) | late | 61.6 | 59.8 | 56.2 | 62.3 |
| Moving (ramp) | early | 77.0 | 76.6 | 75.0 | 71.9 |
| Moving (ramp) | mid | 73.4 | 77.5 | 73.0 | 68.5 |
| Moving (ramp) | late | 59.8 | 60.3 | 58.3 | 60.5 |

### 6. probing self — 자기 조건 head (chance shape/colour 12.5, env 25.0)

| condition | shape z | shape h | shape p | colour z | colour h | colour p | env p |
|---|---:|---:|---:|---:|---:|---:|---:|
| static_visible | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| static_occlusion_early | 100.0 | 100.0 | 100.0 | 99.9 | 100.0 | 99.5 | 100.0 |
| static_occlusion_mid | 100.0 | 99.9 | 100.0 | 99.1 | 99.2 | 98.9 | 100.0 |
| static_occlusion | 99.9 | 99.7 | 95.2 | 100.0 | 99.7 | 86.3 | 100.0 |
| moving_visible_flat | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| moving_occlusion_flat_early | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| moving_occlusion_flat_mid | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| moving_occlusion_flat | 100.0 | 100.0 | 98.9 | 100.0 | 100.0 | 97.0 | 100.0 |
| moving_visible | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| moving_occlusion_early | 99.9 | 100.0 | 100.0 | 99.9 | 100.0 | 100.0 | 100.0 |
| moving_occlusion_mid | 99.9 | 100.0 | 100.0 | 100.0 | 100.0 | 99.7 | 100.0 |
| moving_occlusion | 100.0 | 100.0 | 98.2 | 100.0 | 100.0 | 99.2 | 100.0 |

### 7. probing 이식 — 비가림 head 를 같은 팔의 가림 조건에

| target | motion | point | vis | early | mid | late |
|---|---:|---:|---:|---:|---:|---:|
| shape | Static | z | 100.0 | 96.8 | 96.4 | 90.3 |
| shape | Static | h | 100.0 | 97.6 | 98.6 | 94.4 |
| shape | Static | p | 100.0 | 88.2 | 89.5 | 18.3 |
| shape | Moving (flat) | z | 100.0 | 75.0 | 79.5 | 51.8 |
| shape | Moving (flat) | h | 100.0 | 97.2 | 97.2 | 61.9 |
| shape | Moving (flat) | p | 100.0 | 56.3 | 61.9 | 28.5 |
| shape | Moving (ramp) | z | 100.0 | 82.7 | 73.3 | 66.3 |
| shape | Moving (ramp) | h | 100.0 | 100.0 | 99.1 | 66.2 |
| shape | Moving (ramp) | p | 100.0 | 95.7 | 88.2 | 31.5 |
| colour | Static | z | 100.0 | 100.0 | 99.5 | 93.4 |
| colour | Static | h | 100.0 | 99.9 | 99.6 | 97.5 |
| colour | Static | p | 100.0 | 98.6 | 98.2 | 15.1 |
| colour | Moving (flat) | z | 100.0 | 100.0 | 99.5 | 100.0 |
| colour | Moving (flat) | h | 100.0 | 100.0 | 100.0 | 100.0 |
| colour | Moving (flat) | p | 100.0 | 99.9 | 99.4 | 20.6 |
| colour | Moving (ramp) | z | 100.0 | 99.6 | 99.2 | 98.8 |
| colour | Moving (ramp) | h | 100.0 | 100.0 | 100.0 | 100.0 |
| colour | Moving (ramp) | p | 100.0 | 100.0 | 98.9 | 16.4 |

### 8. probing 표현 이식 — GT 미래(h) ↔ 예측 미래(p), 같은 조건 안에서

| target | motion | direction | vis | early | mid | late |
|---|---:|---:|---:|---:|---:|---:|
| shape | Static | h → h | 100.0 | 100.0 | 99.9 | 99.7 |
| shape | Static | h → p | 94.8 | 76.2 | 76.1 | 18.2 |
| shape | Static | p → p | 100.0 | 100.0 | 100.0 | 95.2 |
| shape | Static | p → h | 99.9 | 99.2 | 99.5 | 59.6 |
| shape | Moving (flat) | h → h | 100.0 | 100.0 | 100.0 | 100.0 |
| shape | Moving (flat) | h → p | 78.8 | 77.6 | 80.4 | 29.6 |
| shape | Moving (flat) | p → p | 100.0 | 100.0 | 100.0 | 98.9 |
| shape | Moving (flat) | p → h | 56.4 | 55.8 | 56.4 | 59.5 |
| shape | Moving (ramp) | h → h | 100.0 | 100.0 | 100.0 | 100.0 |
| shape | Moving (ramp) | h → p | 28.6 | 28.1 | 29.0 | 21.9 |
| shape | Moving (ramp) | p → p | 100.0 | 100.0 | 100.0 | 98.2 |
| shape | Moving (ramp) | p → h | 63.5 | 65.6 | 62.4 | 59.2 |
| colour | Static | h → h | 100.0 | 100.0 | 99.2 | 99.7 |
| colour | Static | h → p | 94.9 | 84.9 | 77.9 | 12.8 |
| colour | Static | p → p | 100.0 | 99.5 | 98.9 | 86.3 |
| colour | Static | p → h | 99.1 | 97.0 | 95.9 | 60.1 |
| colour | Moving (flat) | h → h | 100.0 | 100.0 | 100.0 | 100.0 |
| colour | Moving (flat) | h → p | 79.0 | 75.3 | 76.4 | 28.2 |
| colour | Moving (flat) | p → p | 100.0 | 100.0 | 100.0 | 97.0 |
| colour | Moving (flat) | p → h | 94.0 | 98.0 | 93.4 | 77.6 |
| colour | Moving (ramp) | h → h | 100.0 | 100.0 | 100.0 | 100.0 |
| colour | Moving (ramp) | h → p | 57.3 | 59.6 | 53.1 | 17.8 |
| colour | Moving (ramp) | p → p | 100.0 | 100.0 | 99.7 | 99.2 |
| colour | Moving (ramp) | p → h | 90.7 | 94.2 | 97.5 | 91.5 |

### 9. probing × k (`predictions.json` 사후 분해, 셀당 172~215 clip)

**self (자기 조건 head)**  — `p` 만

| target | motion | timing | k=1 | k=2 | k=3 | k=4 |
|---|---:|---:|---:|---:|---:|---:|
| shape | Static | early | 100.0 | 100.0 | 100.0 | 100.0 |
| shape | Static | mid | 100.0 | 100.0 | 100.0 | 100.0 |
| shape | Static | late | 96.2 | 94.1 | 96.2 | 94.3 |
| shape | Moving (flat) | early | 100.0 | 100.0 | 100.0 | 100.0 |
| shape | Moving (flat) | mid | 100.0 | 100.0 | 100.0 | 100.0 |
| shape | Moving (flat) | late | 99.1 | 99.0 | 98.8 | 98.5 |
| shape | Moving (ramp) | early | 100.0 | 100.0 | 100.0 | 100.0 |
| shape | Moving (ramp) | mid | 100.0 | 100.0 | 100.0 | 100.0 |
| shape | Moving (ramp) | late | 98.1 | 99.0 | 98.3 | 97.4 |
| colour | Static | early | 98.6 | 99.5 | 100.0 | 100.0 |
| colour | Static | mid | 98.6 | 98.5 | 98.3 | 100.0 |
| colour | Static | late | 89.0 | 90.9 | 85.2 | 80.5 |
| colour | Moving (flat) | early | 100.0 | 100.0 | 100.0 | 100.0 |
| colour | Moving (flat) | mid | 100.0 | 100.0 | 100.0 | 100.0 |
| colour | Moving (flat) | late | 95.7 | 99.0 | 98.8 | 94.6 |
| colour | Moving (ramp) | early | 100.0 | 100.0 | 100.0 | 100.0 |
| colour | Moving (ramp) | mid | 100.0 | 99.5 | 100.0 | 99.5 |
| colour | Moving (ramp) | late | 100.0 | 100.0 | 98.9 | 97.9 |

**이식 (비가림 head)**  — `p` 만

| target | motion | timing | k=1 | k=2 | k=3 | k=4 |
|---|---:|---:|---:|---:|---:|---:|
| shape | Static | early | 95.7 | 88.7 | 87.9 | 80.5 |
| shape | Static | mid | 91.6 | 87.9 | 91.7 | 86.8 |
| shape | Static | late | 17.7 | 17.7 | 20.9 | 17.1 |
| shape | Moving (flat) | early | 68.9 | 64.0 | 50.5 | 41.9 |
| shape | Moving (flat) | mid | 70.3 | 67.2 | 58.8 | 51.4 |
| shape | Moving (flat) | late | 28.9 | 28.9 | 31.4 | 25.1 |
| shape | Moving (ramp) | early | 99.5 | 97.3 | 96.2 | 90.0 |
| shape | Moving (ramp) | mid | 94.7 | 95.7 | 87.4 | 75.7 |
| shape | Moving (ramp) | late | 40.7 | 42.4 | 23.9 | 17.5 |
| colour | Static | early | 99.0 | 99.5 | 98.4 | 97.6 |
| colour | Static | mid | 98.6 | 99.0 | 98.3 | 96.8 |
| colour | Static | late | 10.5 | 15.6 | 19.8 | 15.2 |
| colour | Moving (flat) | early | 100.0 | 100.0 | 100.0 | 99.5 |
| colour | Moving (flat) | mid | 99.5 | 98.4 | 100.0 | 99.5 |
| colour | Moving (flat) | late | 32.2 | 19.9 | 13.4 | 15.3 |
| colour | Moving (ramp) | early | 100.0 | 100.0 | 100.0 | 100.0 |
| colour | Moving (ramp) | mid | 99.0 | 100.0 | 98.9 | 97.6 |
| colour | Moving (ramp) | late | 25.8 | 13.1 | 14.4 | 11.3 |

**표현 이식 h → p**  — `p` 만

| target | motion | timing | k=1 | k=2 | k=3 | k=4 |
|---|---:|---:|---:|---:|---:|---:|
| shape | Static | early | 77.5 | 78.0 | 81.3 | 69.0 |
| shape | Static | mid | 80.5 | 75.3 | 78.5 | 69.8 |
| shape | Static | late | 21.5 | 15.6 | 19.8 | 15.7 |
| shape | Moving (flat) | early | 80.9 | 82.8 | 78.0 | 69.5 |
| shape | Moving (flat) | mid | 82.3 | 83.3 | 83.5 | 73.3 |
| shape | Moving (flat) | late | 29.4 | 28.9 | 32.0 | 28.6 |
| shape | Moving (ramp) | early | 28.7 | 30.1 | 28.6 | 25.2 |
| shape | Moving (ramp) | mid | 30.6 | 31.2 | 29.1 | 25.2 |
| shape | Moving (ramp) | late | 34.0 | 29.3 | 12.8 | 9.8 |
| colour | Static | early | 85.2 | 85.5 | 87.9 | 81.4 |
| colour | Static | mid | 80.5 | 83.3 | 72.9 | 74.1 |
| colour | Static | late | 12.9 | 15.1 | 12.6 | 11.0 |
| colour | Moving (flat) | early | 72.2 | 75.3 | 78.6 | 75.7 |
| colour | Moving (flat) | mid | 75.1 | 76.9 | 79.7 | 74.3 |
| colour | Moving (flat) | late | 31.8 | 32.3 | 19.8 | 27.6 |
| colour | Moving (ramp) | early | 62.2 | 61.3 | 59.9 | 55.2 |
| colour | Moving (ramp) | mid | 49.3 | 53.2 | 52.2 | 57.6 |
| colour | Moving (ramp) | late | 24.9 | 17.7 | 17.8 | 10.3 |

### 10. 기전 — margin 분해

채점은 `|p − h(가능)|` vs `|p − h(불가능)|` 의 대소다. margin 은 그 차이.
정확도는 `margin > 0` 인 비율이므로 **효과크기가 곧 정확도**다.

| violation | timing | base |p−h| | margin (% of base) | 효과크기 |mean|/SD | acc (= margin>0 비율) | n |
|---|---:|---:|---:|---:|---:|---:|
| permanence | vis | 0.5729 | 3.726 | 1.299 | 100.0 | 1344 |
| permanence | early | 0.5778 | 2.389 | 1.174 | 97.8 | 1344 |
| permanence | mid | 0.5778 | 2.260 | 1.149 | 95.8 | 1344 |
| permanence | late | 0.5842 | 0.767 | 0.440 | 57.5 | 1344 |
| shape | vis | 0.5783 | 0.537 | 0.965 | 84.3 | 2016 |
| shape | early | 0.5823 | 0.347 | 0.763 | 78.1 | 2016 |
| shape | mid | 0.5825 | 0.324 | 0.734 | 76.0 | 2016 |
| shape | late | 0.5892 | 0.078 | 0.224 | 55.4 | 2016 |
| colour | vis | 0.5782 | 0.274 | 0.761 | 81.5 | 2016 |
| colour | early | 0.5824 | 0.108 | 0.469 | 67.4 | 2016 |
| colour | mid | 0.5825 | 0.111 | 0.477 | 68.7 | 2016 |
| colour | late | 0.5892 | 0.083 | 0.405 | 65.1 | 2016 |

### 11. attention knockout (`knockout__v11_full_vith/results.json`, `knockout_md.py` 가 생성)

<!-- knockout_md:begin -->
기준선 `clean_null` **75.86%** (clean 75.85%, n_pair 21504, 명세 39개, 창 폭 7, keep_self True). 열 이름 `m←c` = `mask->ctx` (mask query 가 ctx key 를 못 읽음), `m←m` = `mask->mask`, `c←c` = `ctx->ctx`.

**층 프로파일 — 전체 (창 중심 층, ±3)**

| 간선 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mask->ctx` | 66.4 | 65.3 | 65.3 | 66.3 | 63.9 | 55.3 | 47.8 | 44.9 | 38.5 | 38.3 | 38.7 | 37.6 | 38.3 |
| `mask->mask` | 76.0 | 75.4 | 73.4 | 74.5 | 73.7 | 75.8 | 75.5 | 75.4 | 75.5 | 77.1 | 75.9 | 76.5 | 74.4 |
| `ctx->ctx` | 76.2 | 76.5 | 76.3 | 76.3 | 76.2 | 75.9 | 75.9 | 75.8 | 75.7 | 75.9 | 76.0 | 75.9 | 76.4 |

**타이밍**

| | 기준선 | m←c L0-3 | m←c L4-10 | m←c L8-11 | m←c all | m←m all | c←c all | n_pair |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| visible | 87.2 | 71.4 | 37.9 | 30.3 | 27.9 | 85.3 | 87.6 | 5376 |
| early | 78.8 | 65.8 | 50.2 | 42.4 | 45.8 | 75.4 | 79.3 | 5376 |
| mid | 77.8 | 64.9 | 50.7 | 44.7 | 45.0 | 75.5 | 78.3 | 5376 |
| late | 59.7 | 63.5 | 40.7 | 32.9 | 34.7 | 61.5 | 60.3 | 5376 |

**위반**

| | 기준선 | m←c L0-3 | m←c L4-10 | m←c L8-11 | m←c all | m←m all | c←c all | n_pair |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vanish | 87.7 | 68.8 | 50.3 | 45.1 | 30.8 | 88.0 | 88.0 | 5376 |
| shape | 73.4 | 56.7 | 42.7 | 33.6 | 36.3 | 68.2 | 72.5 | 8064 |
| color | 70.4 | 74.4 | 43.4 | 36.6 | 45.4 | 71.6 | 72.5 | 8064 |

**운동**

| | 기준선 | m←c L0-3 | m←c L4-10 | m←c L8-11 | m←c all | m←m all | c←c all | n_pair |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| static | 76.4 | 73.7 | 44.0 | 37.0 | 42.9 | 73.0 | 78.2 | 7168 |
| moving_flat | 79.4 | 63.1 | 44.8 | 39.0 | 36.1 | 79.7 | 78.9 | 7168 |
| moving | 71.7 | 62.3 | 45.8 | 36.8 | 36.0 | 70.5 | 71.9 | 7168 |

**k (가림 조건만 — visible 의 k 는 명목값)**

| | 기준선 | m←c L0-3 | m←c L4-10 | m←c L8-11 | m←c all | m←m all | c←c all | n_pair |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k=1 | 73.8 | 66.8 | 47.3 | 40.0 | 41.7 | 73.4 | 74.6 | 4032 |
| k=2 | 72.7 | 64.9 | 47.0 | 39.7 | 41.7 | 71.2 | 73.2 | 4032 |
| k=3 | 71.4 | 63.6 | 46.5 | 39.5 | 41.4 | 70.2 | 72.0 | 4032 |
| k=4 | 70.5 | 63.6 | 47.9 | 40.8 | 42.5 | 68.4 | 70.7 | 4032 |

**타이밍 × 위반**

| | 기준선 | m←c L0-3 | m←c L4-10 | m←c L8-11 | m←c all | m←m all | c←c all | n_pair |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| visible / vanish | 99.9 | 77.2 | 41.2 | 32.0 | 13.2 | 99.8 | 99.9 | 1344 |
| visible / shape | 84.4 | 61.0 | 37.8 | 28.6 | 30.1 | 78.5 | 83.9 | 2016 |
| visible / color | 81.5 | 77.8 | 35.7 | 31.0 | 35.4 | 82.6 | 83.0 | 2016 |
| early / vanish | 97.6 | 69.4 | 59.2 | 56.0 | 40.7 | 94.6 | 97.9 | 1344 |
| early / shape | 77.8 | 57.1 | 49.0 | 38.7 | 45.1 | 69.8 | 76.6 | 2016 |
| early / color | 67.2 | 72.0 | 45.6 | 37.0 | 49.8 | 68.2 | 69.5 | 2016 |
| mid / vanish | 95.7 | 68.2 | 60.0 | 58.6 | 39.7 | 92.6 | 96.2 | 1344 |
| mid / shape | 76.0 | 55.3 | 48.8 | 39.0 | 42.5 | 69.4 | 74.5 | 2016 |
| mid / color | 67.6 | 72.2 | 46.3 | 41.0 | 50.9 | 70.3 | 70.2 | 2016 |
| late / vanish | 57.7 | 60.4 | 40.9 | 33.6 | 29.4 | 65.0 | 57.8 | 1344 |
| late / shape | 55.4 | 53.5 | 35.2 | 27.9 | 27.3 | 55.2 | 55.0 | 2016 |
| late / color | 65.3 | 75.6 | 45.9 | 37.5 | 45.6 | 65.4 | 67.2 | 2016 |

**타이밍 × 운동**

| | 기준선 | m←c L0-3 | m←c L4-10 | m←c L8-11 | m←c all | m←m all | c←c all | n_pair |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| visible / static | 90.9 | 77.8 | 35.4 | 31.6 | 28.9 | 86.8 | 92.1 | 1792 |
| visible / moving_flat | 91.1 | 69.3 | 39.0 | 31.4 | 28.9 | 90.2 | 91.0 | 1792 |
| visible / moving | 79.5 | 67.0 | 39.3 | 28.0 | 25.8 | 79.1 | 79.7 | 1792 |
| early / static | 79.0 | 75.7 | 60.5 | 52.0 | 63.2 | 72.4 | 81.1 | 1792 |
| early / moving_flat | 82.6 | 61.4 | 44.8 | 39.2 | 37.8 | 81.5 | 81.6 | 1792 |
| early / moving | 74.7 | 60.2 | 45.4 | 36.0 | 36.3 | 72.2 | 75.1 | 1792 |
| mid / static | 76.9 | 72.0 | 58.8 | 53.9 | 59.9 | 73.3 | 79.0 | 1792 |
| mid / moving_flat | 83.7 | 61.1 | 47.6 | 42.9 | 38.2 | 82.5 | 82.9 | 1792 |
| mid / moving | 72.8 | 61.5 | 45.6 | 37.3 | 36.8 | 70.8 | 72.9 | 1792 |
| late / static | 58.8 | 69.4 | 21.1 | 10.5 | 19.5 | 59.7 | 60.7 | 1792 |
| late / moving_flat | 60.3 | 60.5 | 47.9 | 42.4 | 39.5 | 64.7 | 60.2 | 1792 |
| late / moving | 59.9 | 60.7 | 53.0 | 45.9 | 45.1 | 60.0 | 59.9 | 1792 |

**예측 이동 `pred_drift` (전체, `|p−p_clean|/|p_clean|` · cosine)**

| 간선 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mask->ctx` | 0.581 | 0.678 | 0.681 | 0.706 | 0.550 | 1.644 | 1.772 | 1.941 | 1.912 | 1.910 | 1.902 | 1.888 | 1.897 / cos 0.143 |
| `mask->mask` | 0.226 | 0.315 | 0.353 | 0.376 | 0.356 | 0.349 | 0.362 | 0.365 | 0.294 | 0.223 | 0.200 | 0.188 | 0.419 / cos 0.931 |
| `ctx->ctx` | 0.091 | 0.098 | 0.105 | 0.114 | 0.102 | 0.103 | 0.095 | 0.080 | 0.067 | 0.053 | 0.043 | 0.027 | 0.129 / cos 0.993 |

<!-- knockout_md:end -->
