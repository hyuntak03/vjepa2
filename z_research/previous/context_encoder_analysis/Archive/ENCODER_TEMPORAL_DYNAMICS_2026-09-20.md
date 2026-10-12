# 실험 6 — 실제 영상에서 문맥 encoder 가 운동 방향을 담는가 (정방향 학습 → 역재생 시험, 2026-09-20)

> 세트 시작점 `../README.md`. 레포 규칙 루트 `CLAUDE.md`. 능력 정의 `STATE_EVOLUTION_CAPABILITIES_2026-09-19.md` 의 **1 읽기** 칸을 실제 영상으로 채운다.
> 산출물 `../exp_results/encoder_temporal_dynamics/{ssv2,ntu}/{results.json, RESULTS.md, preds.npz}` · `summary.json`.

## 0. 한 줄

**문맥 encoder 는 실제 영상에서 좌·우 운동 방향을 담는다.** 정방향 clip 으로만 배운 probe 가 같은 clip 을 역재생하면 답을 뒤집는다
(ssv2 98.1% · ntu 96.2% 뒤집힘). 3,842 파라미터짜리 probe 로도 같다 — 큰 probe 의 과적합이 아니다.

## 1. 동기

합성 세트 (RollOut_v2) 에서 z 는 문맥 끝 속도를 ridge R² 0.99 로 담았다 (`../../RollOutV2/Archive/PREDICTOR_DISTANCE_LIMIT_2026-09-19.md` §5-3).
그것이 **우리가 만든 장면** 의 성질인지, 실제 영상에서도 서는지가 미결이었다. 능력 표의 **1 읽기** 를 real-world 로 옮기는 칸이다.

**왜 "역재생" 인가.** probe 정확도만으로는 무엇을 읽었는지 모른다 — 장면·물체·손 자세 같은 **정지 단서** 로도 left / right 데이터셋은 갈릴 수 있다
(예: "왼쪽으로 미는 영상은 물체가 오른쪽에서 시작한다"). 같은 clip 을 역재생하면 **정지 통계는 그대로이고 시간 순서만 뒤집힌다.**
- 방향을 담았다면 → 예측이 뒤집힌다 (뒤집힘 비율 ~1).
- 정지 단서로 맞혔다면 → 같은 답 (뒤집힘 비율 ~0).
뒤집힘 비율은 **라벨을 쓰지 않는다** (같은 clip 두 방향의 예측 비교). CLAUDE.md §12 "파라미터 없는 교차검증" 의 자리다.

## 2. 세팅 (정확히 무엇을 했나)

| 항목 | 값 |
|---|---|
| 데이터 | `ssv2` `/local_datasets/vlm_direction/ssv2_VP/ssv2_VP_default/{left,right}` 361 + 361 · `ntu` `/local_datasets/vlm_direction/ntu_direction_benchmark/{left,right}` 208 + 208 (그 세트의 나머지 6 방향 폴더는 쓰지 않았다 — 사용자 지시: 좌/우만) |
| clip | 원본 **전체 길이에서 균등 32 장** (`np.linspace`, 반올림) → 256×256 bilinear (`antialias=False`, 레포 관례) → ImageNet 정규화. 종횡비를 무시하고 전체 화면을 눌러 넣는다 (`--spatial full`; 가로 운동이 crop 으로 잘리지 않게) |
| 역재생 | **샘플한 32 장의 순서만 뒤집는다** (다시 샘플하지 않는다). encoder 에 두 번 통과시킨다 |
| encoder | ViT-H, 문맥 encoder (체크포인트의 `encoder` 키 = online), `window_size 32`, **마스크 없음**, bfloat16, frozen. 출력 (16 tubelet × 256) = **4,096 토큰 × 1,280** |
| probe | **attentive** = `src/models/attentive_pooler.AttentiveClassifier` (depth 1, 16 head, 18.0 M) · **tiny** = query 1 개 + linear (**3,842**, RollOutV2 v5 자와 같은 구조) |
| 학습 | **정방향 clip 의 train 쪽만.** AdamW lr 1e-4 · wd 0.01 · 30 epoch · cosine · batch 16 · bf16. 보고는 마지막 epoch |
| 시험 | 같은 probe 에 **held-out clip 의 정방향과 역재생** |
| split | ssv2 = clip 단위 층화 7:3 (seed 0) · **ntu = 연기자 (P###) 단위** 3 명 중 1 명 (같은 사람·장면이 양쪽에 걸치지 않게, CLAUDE.md §1-5 의 취지) |
| 캐시 | **디스크에 토큰 캐시를 쓰지 않는다.** clip 마다 정방향·역재생을 한 번씩 통과시켜 메모리에만 들고 probe 를 학습한다 (샘플이 결정론적이라 매 epoch 다시 돌리는 것과 같은 값) |

⚠️ **좌우 뒤집기 augmentation 을 쓰지 않았다** — 그것이 곧 라벨을 뒤집기 때문이다. 시간 jitter 도 없다.

## 3. 결과

역재생 표의 진실 라벨은 **뒤집은 것**이다 (left 영상을 거꾸로 틀면 오른쪽으로 간다).

| 데이터셋 | probe | 파라미터 | train | 정방향 test | 역재생 test (뒤집힌 라벨) | **뒤집힘 비율** |
|---|---|---:|---:|---:|---:|---:|
| ssv2 (n_test 216) | attentive | 18,041,602 | 100.0 | **99.5** | **97.7** | **98.1** |
| ssv2 | tiny | 3,842 | 95.7 | 91.2 | 93.1 | 85.2 |
| ntu (n_test 156) | attentive | 18,041,602 | 100.0 | **100.0** | **96.2** | **96.2** |
| ntu | tiny | 3,842 | 99.6 | 100.0 | 96.2 | 96.2 |

### 2×2 혼동행렬 4 개 (attentive, 진실 \ 예측)

| ssv2 정방향 (99.5%) | left | right |  | ssv2 역재생 (97.7%) | left | right |
|---|---:|---:|---|---|---:|---:|
| left | 108 | 0 |  | left | 105 | 3 |
| right | 1 | 107 |  | right | 2 | 106 |

| ntu 정방향 (100.0%) | left | right |  | ntu 역재생 (96.2%) | left | right |
|---|---:|---:|---|---|---:|---:|
| left | 79 | 0 |  | left | 77 | 0 |
| right | 0 | 77 |  | right | 6 | 73 |

- **네 칸 모두 대각이 두껍다.** 역재생에서도 뒤집은 라벨 기준으로 대각이다 = 예측이 통째로 뒤집혔다.
- ⚠️ **정정 (2026-09-20, 같은 날) — 역재생 정확도는 독립된 수가 아니다.** 정의상
  `역재생 정확도 = P(정방향 맞음 ∧ 뒤집힘) + P(정방향 틀림 ∧ 안 뒤집힘)` 이다. 두 번째 항은 **틀린 답을 그대로 유지한 clip** 이 정답으로 세어지는 몫이다.
  ssv2 tiny 가 그 경우다: 84.7 + **8.3** = 93.1 (정방향에서 틀린 18 clip 이 안 뒤집혀 더해졌다). 그래서 정방향 정확도가 낮은 줄에서는 역재생 정확도가 부풀려진다.
  **믿고 읽을 수는 정방향 정확도와 뒤집힘 비율 둘뿐이다** (`preds.npz` 에서 재계산해 확인).
- 예측이 한쪽으로 쏠리지 않는다: left 로 예측한 비율이 정방향 50.5 / 50.6%, 역재생 47.7~53.2%.
- **tiny (3,842 파라미터) 가 같은 결론을 낸다.** 18 M probe 는 train 을 100% 외우므로 (손실 0) 그것만으로는 아무 말도 못 한다. 용량이 1/4700 인 probe 에서도 정방향 91~100, 뒤집힘 85~96 이다.

### 그림 (`../figures/encoder_temporal_dynamics/`, `z_research/scripts/figures/plot_ctxenc_direction.py`)

![혼동행렬](../figures/encoder_temporal_dynamics/fig_direction_confusion.png)

**`fig_direction_confusion.{png,pdf}`** — (a) ssv2 정방향 · (b) ssv2 역재생 · (c) ntu 정방향 · (d) ntu 역재생 (attentive probe).
역재생 패널의 진실 라벨은 뒤집은 것이다. 네 칸 모두 대각이 두껍다.

![요약](../figures/encoder_temporal_dynamics/fig_direction_summary.png)

**`fig_direction_summary.{png,pdf}`** — 데이터셋 × probe 별 정방향 정확도 (파랑) · 역재생 정확도 (초록) · 뒤집힘 비율 (주황).
방향을 읽는 표현이면 주황이 100 근처, 정지 단서로 맞히는 것이면 주황이 0 근처다. 점선은 우연 (50%).
그림의 수치는 실행 때마다 `results.json` 에서 다시 읽어 그린다 (stdout 대조).

## 4. 읽는 규칙과 단서 (결론과 같은 비중)

1. **이 실험은 "읽기" 칸만 채운다.** encoder 표현에 방향이 있다는 것이고, predictor 가 그 방향으로 미래를 만드는가 (전이) 는 다른 실험이다.
   합성 세트에서 전이는 ◐ 였다 (방향·속도는 이어 가고 가속도는 아님).
2. **"방향" 과 "끝 자리" 를 아직 못 가른다 — 이 실험으로 못 하는 주장.** 마지막 시점에 물체·손이 어디 있는지**만** 읽어도 (한 장짜리 정지 단서)
   역재생하면 그 자리가 반대가 되어 예측이 뒤집힌다. 뒤집힘 비율은 두 설명을 가르지 못한다.
   그래서 **"encoder 가 시간 구조 (temporal dynamics) 를 읽는다" 는 이 결과만으로 쓰지 않는다.** 쓸 수 있는 문장은
   "좌/우 라벨을 가르는 신호가 encoder 표현에 있고, 그 신호는 시간 순서를 뒤집으면 부호가 바뀐다" 까지다.
   가르는 검사는 세 가지였다 — `static_last` / `static_first` (그 한 장을 32 번 반복: 한 장으로 풀리면 자리 단서) · `shuffle` (순서 무작위).
   스크립트에 `--variants` 로 구현돼 있다 (`ctxenc_direction_probe.py`).
   ⚠️ **2026-09-20 사용자 결정: 돌리지 않는다** ("걍 취소해"). job 215132 는 ssv2 shuffle 추출 중에 취소했다. 다시 하려면 위 인자 하나면 된다.
3. **역재생 영상은 자연스럽지 않다** (물리·생체 운동이 거꾸로). encoder 가 본 적 없는 분포일 수 있다. 그래도 뒤집힘 비율이 0.96 이상이라는 사실은 분포 밖에서도 방향 신호가 살아 있다는 뜻이다.
4. **ntu 는 두 probe 수치가 완전히 같다** (정방향 100.0, 역재생 96.2, 같은 혼동행렬). 과제가 쉬워 두 probe 가 같은 해에 수렴한 것으로 보이지만, 카메라 시점·사람 위치 같은 세트 고유 단서가 있을 수 있다. ntu 수치는 ssv2 와 같이 읽는다.
5. **ssv2 의 라벨은 폴더 이름**이고 우리가 만든 것이 아니다. 라벨 정의 (무엇이 left 인가: 물체의 이동 방향인가 손의 방향인가) 는 그 세트 문서에 따른다 — 확인 안 함.
6. 종횡비를 눌러 넣었다 (`full`). center crop (`--spatial crop`) 은 안 돌렸다. 두 전처리 모두 정방향·역재생에 똑같이 걸리므로 **뒤집힘 비율 판정에는 영향이 없다**.

## 5. 다음

1. ~~순서 섞기·한 장 대조 (§4-2)~~ — **2026-09-20 사용자 결정으로 보류.** 하려면 `--variants shuffle static_first static_last` (GPU 1 장 40 분).
2. ntu 의 나머지 6 방향 (위·아래·대각) 으로 넓히기 — 2×2 가 아니라 8 지선다.
3. 같은 자를 **predictor 출력 (p)** 에 걸기 — "읽기" 에서 "전이" 로 넘어가는 칸. 미래 표현이 문맥의 방향을 이어받는지.
4. EK100 에 같은 검사 (능력 표 §3 의 다른 도메인 probe).

## 재현

```bash
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python

# GPU 1 장, 약 11 분 (ViT-H 로딩 + clip 1,138 개 × 정방향·역재생 특징 추출 + probe 2 종 × 30 epoch)
$P z_research/scripts/analysis/ctxenc_direction_probe.py                 # 두 데이터셋
bash z_research/scripts/analysis/ctxenc_direction_sbatch.sh              # SLURM (vll3, 영상이 그 노드의 /local_datasets 에 있다)

# 배관 점검 (모델 없이, CPU 몇 초)
$P z_research/scripts/analysis/ctxenc_direction_probe.py --no-encoder --smoke 4 --epochs 2
```

실행 기록: job 215127 (vll3, GPU 1 장, 11 분 16 초, 2026-09-20). 수치는 전부 `exp_results/encoder_temporal_dynamics/summary.json` 에서 옮겼다.
