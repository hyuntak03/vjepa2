> ⚠️ 2026-09-25 감사 반영 정본 → [`../../Archive/RESULTS_2026-09-25.md`](../../Archive/RESULTS_2026-09-25.md) §6 (2026-09-26 정리: 옛 new_archive/README 를 옮긴 문서. 자와 무관해 여전히 정본). 감사는 [`../../audit/v11_identity/AUDIT.md`](../../audit/v11_identity/AUDIT.md) (아래 `../_audit/` 는 옛 경로). 이 문서의 §5 항목 3 ("가림 조건에서 사상이 한 번 더 바뀐다") 은 **철회** — encoder 쌍도 같이 무너진다 (`../_audit/v11_identity/AUDIT.md` §2). 본문은 기록으로 둔다.

# v11 — 평균 풀링 선형 probe 로 shape · color 읽기 (2)~(6) (2026-09-24)

> 수치는 전부 `results.json` (λ 선택 판) · `results_lam0.1.json` (λ 민감도) 에서 옮겼다 (`v11_pooled_probe.py --tables`).
> 독립 재계산 (닫힌 해 ridge, 레포 probe 코드를 쓰지 않음) 을 §8 에 같이 적는다. 이 문서는 분석 에이전트가 끝내지 못해 메인 세션이 마무리했다.

## 0. 설정

| | |
|---|---|
| 데이터 | IntPhysGen v11 가능 변이 · 물체 있는 clip 9,408 개 (block 5,376). visible (k=0) + 문맥 끝 가림 late (k=1~4) |
| 특징 | 튜블릿마다 256 토큰 평균 (`v11_pooled_features.py`). **z** = context encoder, 문맥 16 샘플만 (predictor 입력과 같다), LN · **p** = predictor 미래 8 튜블릿 · **h** = target encoder 32 샘플 전부, LN (대조군) |
| 풀링 | z_all = 문맥 8 튜블릿 평균 · z_hid = **완전히 가려진** 문맥 튜블릿 평균 (k=2,3 → t7, k=4 → t6~7) · p_fut = 미래 8 튜블릿 평균 · h_fut = h 미래 8 튜블릿 평균 |
| probe | 표준화 (source train 통계) + multinomial logistic, L-BFGS float64, λ ∈ {1e-4 … 1e-1} 을 block 단위 hold-out 으로 선택 |
| split | 레포 `WMADataset._split` — block 단위, train 0.5, seed 0, condition 층화. train 4,692 / test 4,716 clip, **겹치는 block 0** |
| CI | test block bootstrap 1,000 회, 95 % |
| 우연 | shape 1/7 = 14.3 %, color 1/8 = 12.5 %. 라벨 섞기 대조: (2) 12.0 / 10.0 %, (5) 11.6~13.8 % |

## 1. (2) context encoder 전체 문맥 → 읽힌다

| test | shape | color |
|---|---|---|
| visible | 100.0 [100.0, 100.0] | 99.5 [99.2, 99.8] |
| late (k=1~4) | 98.8 [98.3, 99.3] | 98.8 [98.4, 99.3] |
| late / static · flat · ramp | 97.8 · 99.1 · 99.5 | 97.0 · 99.8 · 99.9 |

## 2. (3) **가려진 프레임만** 풀링해도 읽힌다

late k ≥ 2 clip, 완전히 가려진 문맥 튜블릿만 (물체가 그 프레임에 안 보인다):

| | shape | color |
|---|---|---|
| (3) z_hid (가려진 튜블릿) | **95.7** [94.6, 96.6] | **93.9** [92.5, 95.0] |
| (3c2) 같은 clip 의 가려지지 않은 문맥 튜블릿 | 98.3 [97.7, 98.9] | 98.2 [97.6, 98.8] |
| 차이 (짝 비교) | −2.7 [−3.6, −1.7] | −4.3 [−5.5, −3.2] |
| (3c1) visible clip, 같은 자리 (t7) | 100.0 | 98.5 |
| z_hid by motion static · flat · ramp | 91.3 · 98.1 · 97.5 | 84.6 · 99.0 · 98.2 |

→ **context encoder 는 물체가 가려진 프레임의 토큰에도 정체 (shape · color) 를 싣고 있다.** 보이는 프레임보다 3~4 pt 낮을 뿐이다 (static 이 가장 낮다).
⚠️ encoder 는 16 프레임 전체에 attention 을 건다 — 가려진 프레임 토큰의 정보는 앞 프레임에서 옮겨 온 것일 수 있다. "그 프레임 혼자서 안다" 가 아니다.

## 3. (4) 전체 문맥 머리 → 가려진 프레임 : **이식이 안 된다** (encoder 안에서도)

| late k ≥ 2 test | shape | color |
|---|---|---|
| (4) z_all 머리 → z_hid | 59.6 [56.8, 62.1] | 35.5 [32.9, 38.2] |
| (4b) z_hid 머리 → z_all | 60.9 [58.2, 63.5] | 52.8 [50.0, 55.5] |
| 기준: z_all 머리 → z_all (같은 clip) | 99.0 | 98.7 |
| (4) by motion static · flat · ramp | 35.1 · 76.2 · 67.5 | 18.3 · 49.1 · 39.2 |

→ 정보는 두 곳 모두에 있는데 (§1·§2 에서 95~99 %) **놓이는 자리가 다르다** — 같은 encoder 안에서도. 이식 실패가 곧 정보 부재가 아니라는 기준선이다.

## 4. (5) predictor 미래 (p_fut) → 조건마다 머리를 따로 배우면 읽힌다

| 머리 → test | shape | color |
|---|---|---|
| visible 머리 → visible | 99.9 [99.8, 100.0] | 99.1 [98.7, 99.5] |
| **late 머리 → late** | **97.3** [96.5, 98.0] | **97.2** [96.5, 97.9] |
| visible 머리 → late (교차) | 40.9 [38.7, 43.2] | 33.1 [30.9, 35.4] |
| late 머리 → visible (교차) | 72.8 [70.7, 75.0] | 67.8 [65.6, 70.0] |
| visible 머리 → late, 라벨 없는 재표준화 | 59.5 | 60.8 |

(5b) **미래 튜블릿 하나씩** — p late 의 shape 94.2~97.0 %, color 94.3~96.8 %, t0~t7 에서 **떨어지지 않는다** (h late 96.8~99.1 / 94.0~96.9). k 별로도 같다.

→ **문맥 끝에 물체가 가려져도 p 의 미래 토큰에는 정체가 끝까지 실려 있다.** 그런데 §6 (v11_presence) 의 위치 자는 같은 칸에서 "물체 없음" 이다
(flat · ramp last 에서 '있다' 0~35 %). **정체는 있는데 '여기 물체가 있다' 는 형태로는 없다** — 사용자 가설 "잠재적으로만 존재" 와 맞는 조합이다.

## 5. (6) encoder 머리 → predictor : raw 로는 안 되고, **라벨 없는 직교 정렬 하나로 거의 다 돌아온다**

| 머리 · 변형 | shape visible | shape late | color visible | color late |
|---|---|---|---|---|
| 상한: z 머리 → z_all · h 머리 → h_fut · p 자기 머리 | 100 · 99.9 · 99.9 | 98.8 · 99.7 · 97.3 | 99.5 · 99.8 · 99.1 | 98.8 · 99.4 · 97.2 |
| **z 머리 → p, raw** | 41.9 | 34.5 | 32.7 | 18.2 |
| z 머리 → p, 평균 이동 (라벨 없음) | 70.0 | 58.0 | 63.5 | 53.8 |
| z 머리 → p, **Procrustes (visible 짝으로 맞춤)** | **99.8** | 54.5 | **99.2** | 59.2 |
| z 머리 → p, Procrustes (**late 짝으로** 맞춤) | 82.6 | **96.4** | 80.6 | **96.9** |
| z 머리 → p, Procrustes (전체 짝) | 99.8 | 95.4 | 99.1 | 95.6 |
| z 머리 → p, Procrustes, 짝 섞기 (대조) | 12.4 | 14.9 | 14.2 | 13.3 |
| **h 머리 → p, raw** (대조군) | 44.0 | 39.5 | 47.6 | 30.5 |
| h 머리 → p, Procrustes (visible 짝) | 99.5 | 47.4 | 98.8 | 35.0 |
| h 머리 → p, Procrustes (late 짝) | 77.8 | 92.5 | 59.0 | 93.1 |

Procrustes 는 test 에 없는 **train clip 의 (p, z) 짝**으로 직교 사상 하나를 맞춘 것 (라벨은 안 쓴다). 상대 잔차: visible 짝 0.15 (test visible) → 0.33 (test late).

1. **h 대조군도 z 와 같은 모양이다** (raw 44 / 39.5 vs 42 / 34.5) → 이식 실패는 "z 공간 ≠ h 공간" 때문이 아니다. p 가 encoder 들과 다른 자리에 놓는다
2. **직교 사상 하나로 visible 은 99.8 % 까지 돌아온다** → p 의 정체 정보는 encoder 의 것과 같은 구조를 **회전·이동한 형태**로 갖고 있다 (짝을 섞으면 우연 수준)
3. **visible 에서 맞춘 사상은 late 에 안 통하고 (54.5 / 59.2), late 짝으로 맞추면 통한다 (96.4 / 96.9)** → 문맥 끝에 가려진 clip 의 p 는 visible clip 의 p 와 **또 다른 자리**에 놓인다.
   조건마다 놓는 자리가 바뀐다 — CLAUDE.md §5-2 의 attentive probe 결과 ("비가림→가림 이식은 p 에서만 무너진다") 를 평균 풀링 선형 probe 로 다시 본 것

## 6. 이 결과로 **못 하는** 주장

- ⚠️ **이식 정확도의 크기는 probe 에 따라 크게 흔들린다** — 같은 split 에서 닫힌 해 ridge 로 다시 내면 z 머리 → p raw 가 23.6 / 18.6 (logistic 41.9 / 34.5),
  (4) 가 24.7 / 18.9 (logistic 59.6 / 35.5), visible→late 가 18.9 / 30.6 (logistic 40.9 / 33.1). λ=0.1 로도 5~13 pt 움직인다 (`results_lam0.1.json`).
  **주장할 수 있는 것은 "같은 조건 안 (93~100 %) 보다 훨씬 낮다" 와 "Procrustes 를 같은 조건 짝으로 맞추면 95~99 % 로 돌아온다" 까지다.** 이식 수치 자체를 비교하지 말 것
- "p 가 정체를 **잃지 않는다**" 는 평균 풀링 선형 probe 기준이다 — 풀링 벡터에는 장면 (배경·조건) 이 분산의 75~81 % 를 차지한다 (`../v11_geometry/GEOMETRY.md`). 정체는 작은 부분공간이다
- Procrustes 는 라벨은 안 쓰지만 **clip 짝**을 쓴다 — "정렬하면 된다" 는 짝을 아는 경우의 이야기다
- z 는 **문맥** 프레임, p 는 **미래** 프레임이다 — 같은 물체지만 다른 시점이다. h_fut (미래 프레임, target encoder) 이 그 차이를 통제하는 대조군이다
- "표현이 회전했다" 는 **직교 사상 하나로 이식이 돌아온다** 는 뜻으로만 쓴다 — 회전 각도나 부분공간 겹침은 `../v11_geometry/` (주각) 을 볼 것

## 7. 그림

- `fig_probe.png` — 주요 정확도 막대 (shape · color, 우연 선)
- `fig_probe_tubelet.png` — 미래 튜블릿별 정확도 (p · h, visible · late)

## 8. 독립 재계산 (메인 세션, 2026-09-24)

닫힌 해 ridge (one-vs-rest, λ=10, 같은 표준화, 같은 split — 레포 `WMADataset._split`), float64. 레포 probe 코드는 쓰지 않았다.

| | shape (ridge / logistic) | color (ridge / logistic) |
|---|---|---|
| (2) z → z late | 99.4 / 98.8 | 98.3 / 98.8 |
| (3) z_hid 같은 조건 | 96.8 / 95.7 | 93.4 / 93.9 |
| (5) p late → late | 96.0 / 97.3 | 95.5 / 97.2 |
| (5) p visible → late | 18.9 / 40.9 | 30.6 / 33.1 |
| (4) z 머리 → z_hid | 24.7 / 59.6 | 18.9 / 35.5 |
| (6a) z 머리 → p raw, visible | 23.6 / 41.9 | 18.5 / 32.7 |
| (6c) Procrustes (visible 짝) → visible | 99.8 / 99.8 | 99.0 / 99.2 |
| (6c) Procrustes (visible 짝) → late | 19.2 / 54.5 | 35.8 / 59.2 |
| (6c) Procrustes (late 짝) → late | 94.6 / 96.4 | 94.4 / 96.9 |
| (6h) h 머리 → p raw, visible | 19.8 / 44.0 | 20.9 / 47.6 |

같은 조건 안 · 같은 조건 짝 Procrustes 는 두 방법이 1~3 pt 안에서 맞고, **이식 (raw · 다른 조건) 은 방향만 같고 크기가 다르다** — §6 첫 줄의 근거.

## 재현

```bash
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
$P z_research/scripts/analysis/v11_pooled_features.py --gpus 8                 # 특징 (~9 분) → /local_datasets/.../cache/v11_pooled_vith/
$P z_research/scripts/analysis/v11_pooled_probe.py --device cuda:0             # results.json + 그림 (~22 분)
$P z_research/scripts/analysis/v11_pooled_probe.py --device cuda:0 --lams 0.1 --tag lam0.1 --skip-tubelet   # λ 민감도
$P z_research/scripts/analysis/v11_pooled_probe.py --tables                    # 이 문서의 표
```
