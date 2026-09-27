# 자(readout) 는 `attn` 하나다 — 고른 이유와 **다시 시도하지 말 것** (2026-09-23)

> **결론: attention pooling + head 2 개 (`attn`).** 다른 후보는 코드·결과에서 전부 뺐다.
> 이 문서는 하루에 네 번 후보를 옮겨 다닌 기록이다. **같은 비교를 다시 하지 말 것.**

## 자

```python
a      = softmax(tok @ q / √D)     # (256,)   어디를 볼지
pooled = a @ tok                   # (1280,)
xy     = Linear(pooled)            # (2,)     좌표 head
pres   = Linear(pooled)            # ()       있음/없음 head
```
파라미터 5,123. 학습: `MSE(xy)` (양성만) + `BCE(pres)` (양성+음성, 제외 칸 제외), λ=1.0,
Adam lr 1e-3, batch 32 clip, 300 epoch.

**학습은 `RollOut_v2_training` v6 에서만** 한다 (8,352 clip, block split, 문턱은 val 음성 5 % 오탐).
**v3 · v11 은 frozen 으로 걸기만** 한다 — 측정 대상으로 자를 학습하지 않는다.

## 왜 `attn` 인가 — **v3 에서 존재 판정이 전이된다**

v3 는 물체가 f16~f63 내내 화면 안이라 `z` 는 **항상 "있다"** 여야 한다. 라벨이 필요 없는 검증이다.

| v3 에서 `z` 가 "없다" 고 하는 비율 | 전체 | `ledge` | `ramp_a` | `arc` | `wall` |
|---|---:|---:|---:|---:|---:|
| **`attn`** | **0.4 ~ 4.4 %** | **1 %** | 5 % | 3 % | 2 % |
| `mass` (버림) | 2.9 ~ 14.1 % | **21 %** | 17 % | 16 % | 14 % |

존재 판정이 이 실험의 축이라 이게 결정적이었다. v6 에서도 `attn` 이 제일 좋다 (ROC p .988 / z .996 / h .997).

## 기각한 후보 — 다시 시도하지 말 것

| 후보 | 무엇 | 기각 사유 |
|---|---|---|
| `linear` | 256×1280 flatten → Linear | 표현마다 lr 창이 좁다. `p` 0.86 칸인데 `z`·`h` 는 1e-5 에서 2.43(미학습) / 1e-4 에서 5.49·4.93(과적합) |
| `heat` | 토큰 점수 soft-argmax + `logsumexp` presence | `mass` 의 하위 집합 (보정 head 없음). 좌표 0.79~0.86 칸 |
| `mass` | `heat` + ±1 칸 보정 | **좌표는 최고** (v3 `z` 5~9 px vs attn 13~21) 인데 **존재가 장면에 밀린다** (위 표). presence 가 `logsumexp(s)` 의 **절대 크기**라 장면이 s 를 통째로 밀면 문턱이 어긋난다 |
| `mass` + 스케일 γ | presence 에 자유도 추가 | presence 를 `s` 에서 떼면 좌표 손실만 남아 **지도가 죽는다** (top1 0.54→0.20, 좌표 0.58→1.04 칸). ROC 는 0.973→0.979 로 0.006 올랐을 뿐 |
| `heatmap` (CenterNet 식) | patch 마다 sigmoid + 가우시안 target + focal | v6 에서 셋 중 최악 (좌표 1.62/1.65 칸, ROC .938/.942, top1 **0.11**). `sigmoid → q/Σq` 는 softmax 와 달리 **차이를 증폭하지 않아 지도가 퍼진다.** sigmoid heatmap 에는 soft-argmax 가 아니라 **argmax + offset** 이 맞다 (CenterNet 원본). 섞어 쓴 게 잘못 |
| query 2 개 (`attn2`) | 위치용·존재용 분리 | 안 쟀다. softmax 는 몇 개든 정규화돼 "있음/없음" 을 만들지 않는다 |
| λ (presence 손실 가중치) 튜닝 | — | **안 한다.** 자는 도구이지 최적화 대상이 아니다. 학습셋에 맞출수록 측정셋 전이가 나빠지고, test 숫자로 고르면 test 선택이 된다 |

## `attn` 의 한계 — 읽을 때 반드시 붙일 것

**① 좌표를 학습된 회귀로 낸다.** `Linear(pooled)` 라 "이 벡터면 화면 어디" 를 학습 분포에서 배운다.

```
읽은값 = a × 진실 + b       attn/z  v6 test   x: a 0.99  b  −2.6 px    y: a 1.00  b  −7.8
                                   v3        x: a 1.03  b −13.6       y: a 1.12  b −28.7
```
v3 위치 오차 13~21 px — 물체 폭 36 px 의 절반이라 **쓸 수는 있다.**
(격자에서 읽는 `mass` 는 v3 에서 a 0.99 / b 0.5 로 이 실패 모드가 없다. 좌표만 보면 `mass` 가 옳다.)

**② `wall`·`ledge` 의 가능/불가능 판정은 이 자로 하지 않는다.**
분리량이 P=8 에서 28.7 px 인데 자 오차가 15~20 px 이라 **잡음 안**이다. 실제로 좌표 편향(Δx +25 px)이
결론을 뒤집었다 — 원본은 "p 가 정지를 예측", 편향을 빼면 "p 가 통과를 예측" 이 됐고 `mass` 와 반대였다.
이 질문은 **자를 안 쓰는 잠재 공간 검사** (`|p−h_pos|` vs `|p−h_imp|`) 로 답한다.

**③ attention 지도(top1)는 판정에 못 쓴다.** v3 에서 0.03~0.27 로 평평하고, v6 에서도
"물체 있음"(0.262)과 "물체가 화면 밖"(0.267)을 구분하지 못했다. **판정은 presence 값으로만.**

**④ v3 에서는 문턱을 재조정할 수 없다.** 물체가 내내 화면 안이고 빈 장면도 없어 **음성이 0 개**다.
v6 문턱을 그대로 쓰고, 위 라벨 없는 검증으로 타당성만 확인한다.

## 재현

```bash
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
$P z_research/scripts/analysis/rollout2_presence_readout.py --reps p z h --skip-extract --keep-shm \
   --feat-dir /data2/local_datasets/world/world_analysis/cache/rollout2_training_v6_feats
$P z_research/scripts/analysis/rollout3_window_readout.py --reps p z --gpus 8
```
