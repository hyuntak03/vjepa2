# GEOPHYS 각도 일관성 — V-JEPA 2 encoder 궤적, VOBench late (2026-10-29)

- **질문**: GEOPHYS (Internò et al. 2026) 의 "꺾임각 표준편차" 를 V-JEPA 2 encoder 궤적에 걸면, 예측 없이 late (A|B 바꿈이 가려진 칸) 를 푸는가.
- **답**: 못 푼다. 43.5 % 로 chance 아래다 (realistic 41.2 [35, 48] 은 CI 가 50 을 넘지 않는다). 같은 칸의 V-JEPA 2 surprise 는 57.9 / 65.7, 복사 기준선은 56.5 / 61.1 이다.
- **다음**: 바꿈이 보이는 칸 (visible · early · mid) 에도 걸어 leakage 측정값으로 쓸 수 있는지 본다. 이 신호가 왜 chance 아래로 가는지 (가능 영상이 더 불규칙) 도 확인한다.

## 왜 했나
GEOPHYS 는 학습 없이 "궤적이 덜 매끄러운 쪽이 위반" 이라는 규칙 하나로 판정한다. predictor 를 쓰지 않으므로 복사 기준선과 같은 범주다. 즉 관측만으로 풀리는 정도를 재는 자다. late 는 복사로 안 풀리는 칸이다. 그래서 이 칸에서 매끄러움 규칙이 통하는지가 질문이었다.

## 무엇을 어떻게 했나
- **표현**: 샘플 32 장 (raw 0, 3, …, 93) 을 한 번에 target encoder 에 넣었다. 문맥 · 미래로 나누지 않았다. 모델은 V-JEPA 2 ViT-H 릴리즈의 마지막 층이다. 출력은 튜블릿 16 개 × 공간 토큰 256 개 × 1280 차원이다.
- **궤적**: 튜블릿마다 공간 평균을 내서 16 × 1280 궤적을 만든다.
  - 변위는 이웃한 두 튜블릿의 차이로 15 개다.
  - 꺾임각은 이웃한 두 변위 사이의 각으로 14 개다.
- **점수**: 영상 하나의 점수는 꺾임각의 표준편차다. matched 쌍에서 불가능 영상의 점수가 크면 정답이다. 학습 파라미터와 threshold 는 없다.
- **두 판**: 원 특징 (논문 규칙, 정규화 없음) 과 토큰별 LN 뒤 평균 (우리 채점 공간). 결과는 거의 같다.
- **보조 신호**: 꺾임각 평균 (φ_curv) 과 속도 표준편차 (φ_speed).
- **표본**: late 432 쌍 = objset 2 × 위반 3 × 72 쌍.

## 결과 [측정]
칸 = 정확도 % (A / B 방향). permanence 에서 A 는 물체→빈, B 는 빈→물체다. 원 특징 판이다.

| | primitive | realistic |
|---|---:|---:|
| **꺾임각 std (GEOPHYS)** | **45.8** [40, 52] | **41.2** [35, 48] |
| 꺾임각 평균 | 50.0 | 36.1 |
| 속도 std | 32.4 | 33.8 |
| 참고: V-JEPA 2 surprise | 57.9 | 65.7 |
| 참고: 복사 기준선 | 56.5 | 61.1 |

꺾임각 std 를 위반별로 나누면 다음과 같다.

| | permanence | shape | color |
|---|---:|---:|---:|
| primitive | 51.4 (58/44) | 44.4 (53/36) | 41.7 (44/39) |
| realistic | 47.2 (75/19) | 36.1 (33/39) | 40.3 (39/42) |

- 궤적의 평균 꺾임각은 1.957 rad (112°) 다. 논문의 DINOv3 값은 1.78–1.84 rad 다.

### 블록 단위 인코딩 (2026-10-29 추가)
튜블릿 (2 장) 마다 encoder 에 따로 넣었다. 블록 토큰은 자기 2 장만 본다. AR 학습 · 채점과 같은 입력이다. 나머지 (공간 평균 · 꺾임각 · 쌍 판정) 는 같다. 원 특징 판이다.

| | 한 번에 (32 장) | 블록 단위 |
|---|---:|---:|
| 꺾임각 std — 전체 | 43.5 | **51.2** |
| primitive | 45.8 [40, 52] | 52.3 [46, 59] |
| realistic | 41.2 [35, 48] | 50.0 [44, 56] |
| 꺾임각 평균 — 전체 | 43.1 | 46.5 |
| 속도 std — 전체 | 33.1 | 50.7 |
| 궤적 평균 꺾임각 | 1.957 rad | 1.876 rad |

블록 단위 꺾임각 std 를 위반별로 나누면 다음과 같다.

| | permanence | shape | color |
|---|---:|---:|---:|
| primitive | 56.9 (39/75) | 51.4 (56/47) | 48.6 (50/47) |
| realistic | 55.6 (33/78) | 48.6 (50/47) | 45.8 (47/44) |

- **블록 단위에서는 세 신호 모두 chance 다** (46–52). 한 번에 넣을 때 보이던 "chance 아래" (속도 std 33) 가 사라진다 [측정].
- 그러므로 chance 아래는 신호의 성질이 아니라 **한 번에 넣는 인코딩의 성질**이다.
  - 한 번에 넣으면 문맥 튜블릿의 토큰도 미래 프레임을 본다. 그래서 가능 영상과 불가능 영상의 궤적이 바꿈 이전부터 갈라진다.
  - 블록 단위에서는 문맥 블록 8 개가 쌍 안에서 같다. 차이는 미래 쪽에서만 생긴다.
  - 이 설명은 아직 확인하지 않았다 [추측]. 쌍 안에서 문맥 구간 궤적이 같은지 직접 비교하면 확인된다.
- 블록 단위에서도 permanence 의 A / B 는 갈린다 (33–39 / 75–78). 빈→물체 (B) 는 물체가 나타나는 쪽이 더 꺾인다. 물체→빈 (A) 은 반대다.

### 가림 시점 네 칸 전부 (2026-10-29 추가)
VOBench 전체 1,440 쌍에 걸었다. 바꿈이 보이는 칸에서는 이 신호가 통하는지 보기 위해서다. 원 특징 판이고, 두 objset 과 세 위반을 합친 값이다.

| 신호 · 인코딩 | visible (144) | early (432) | mid (432) | late (432) |
|---|---:|---:|---:|---:|
| 속도 std · 블록 | **79.9** | **67.4** | **61.1** | 50.7 |
| 꺾임각 std · 블록 | 59.0 | 50.7 | 53.7 | 51.2 |
| 꺾임각 평균 · 블록 | 38.9 | 38.2 | 45.6 | 46.5 |
| 속도 std · 한 번에 | **15.3** | 34.3 | 34.7 | 33.1 |
| 꺾임각 std · 한 번에 | 25.0 | 29.6 | 31.2 | 43.5 |
| 꺾임각 평균 · 한 번에 | 34.7 | 26.9 | 32.9 | 43.1 |

- **블록 단위 (프레임마다 따로 = 논문의 이미지 encoder 와 같은 조건) 는 바꿈이 보일수록 맞힌다** [측정]. 속도 std 는 visible 79.9 → late 50.7 이다.
- 이 순서는 복사 기준선과 같다. 복사도 visible · early · mid 는 높고 late 는 chance 근처다. **이 신호는 예측 없이 풀리는 칸만 푼다.**
- **한 번에 넣으면 방향이 뒤집힌다.** visible 에서 위반 영상의 궤적이 오히려 더 매끄럽다 (속도 std 15.3). 이유는 모른다.
  - 후보: 영상 encoder 가 attention 으로 바꿈을 시간축 전체에 퍼뜨려 한 지점의 튐을 지운다 [추측].
  - 뒤집은 방향 (85 %) 은 데이터를 보고 고른 것이라 점수로 쓰지 않는다.

### surprise · 복사 기준선과 나란히 (2026-10-29 추가)
같은 1,440 쌍이다. surprise 와 복사는 기존 산출물 (surprise 채점의 영상별 surprise, 복사 실행 배열) 에서 다시 계산했다. 두 objset 을 합친 값이다.
README §3-1 · §3-4 의 objset 별 값의 평균과 일치한다.

| | visible | early | mid | late | 전체 |
|---|---:|---:|---:|---:|---:|
| V-JEPA 2 surprise (예측) | 91.0 | 82.9 | 81.0 | **61.8** | 76.8 |
| 복사 기준선 (예측 없음) | 90.3 | 88.4 | 82.2 | **58.8** | 77.8 |
| GEOPHYS 속도 std · 블록 (예측 없음 · 학습 없음) | 79.9 | 67.4 | 61.1 | **50.7** | 61.7 |

permanence 에서는 GEOPHYS 도 visible · early 에서 100 · 99.3 이다 (surprise 97.9 · 95.1). shape · color 는 복사보다 15–35 pt 낮다.

## 해석
- **매끄러움 규칙은 late 를 풀지 못한다** [측정]. 모든 위반 · 두 objset 에서 chance 근처이거나 그 아래다.
- **chance 아래라는 점 자체는 신호다.** 속도 std 는 33 % 로, 가능 영상이 오히려 더 불규칙하다는 뜻이다. 이유는 아직 모른다.
  - 후보 하나: 물체가 계속 움직이는 가능 미래가 "사라짐 · 바뀜" 보다 궤적을 더 흔든다 [추측].
  - 규칙의 방향을 뒤집으면 67 % 가 된다. 하지만 그 방향은 test 세트를 보고 고른 것이다. 그래서 점수로 쓰지 않는다.
- **못 하는 주장**: 공간 평균 궤적이라 위치와 정체 정보를 버린다. 이 결과는 "GEOPHYS 신호가 late 를 못 본다" 까지만 말한다. "encoder 에 정보가 없다" 로 읽으면 안 된다. 같은 encoder 의 probe 는 정체를 읽는다.
- 논문은 DINOv3 이고 우리는 V-JEPA 2 encoder 다. 논문 Table 1 에는 V-JEPA 2 판 (LikePhys 78.3, IntPhys2 59.5) 이 있지만, 그 판이 32 장을 한 번에 넣은 것인지는 적혀 있지 않다.

## 부록: 재현
```bash
P=/data/hyuntak/anaconda3/envs/vjepa2_bw/bin/python        # vll7 (Blackwell). vll5 는 envs/vjepa2 + 스크립트의 FRAMES_ROOT 를 바꿀 것
$P z_research/scripts/analysis/vobench_geophys_angle.py extract 4   # 864 clip, GPU 4 장 약 30 초 (+ 모델 로드)
$P z_research/scripts/analysis/vobench_geophys_angle.py score       # → results.json
$P z_research/scripts/analysis/vobench_geophys_angle.py extract 4 block && $P z_research/scripts/analysis/vobench_geophys_angle.py score block   # → results_block.json
GEOPHYS_OCC=all $P z_research/scripts/analysis/vobench_geophys_angle.py extract 4 [window|block] && GEOPHYS_OCC=all $P … score [window|block]   # → results{,_block}_all.json (2,880 clip, 약 70 초)
```
config: `z_research/VOBench_v1/exp_results/surprise_c16t32__vobench_v1_vith/_resolved.yaml` (`frames_root` 만 `/local_datasets/world/testbed/VOBench_v1` 로 바꿈), float32 + autocast float16. 궤적 캐시는 `_traj/{raw,ln}.npy` 다 (864 × 16 × 1280).
