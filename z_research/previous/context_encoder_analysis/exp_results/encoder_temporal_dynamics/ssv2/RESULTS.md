# ssv2 — 문맥 encoder 의 좌/우 운동 방향 (정방향 학습 → 역재생 시험)

clip 722 (left 361 / right 361), train 506 / test 216, split = clip 단위, 균등 32 장 → 256x256 (full), ViT-H 문맥 encoder (frozen), probe 는 아래 표.

역재생 표의 진실 라벨은 **뒤집은 것** 이다 (left 영상을 거꾸로 틀면 오른쪽으로 간다). 뒤집힘 비율 = 같은 clip 의 정방향 예측과 역재생 예측이 다른 비율 (라벨이 필요 없는 검사).

| probe | 파라미터 | train | 정방향 test | 역재생 test (뒤집힌 라벨) | 뒤집힘 비율 |
|---|---:|---:|---:|---:|---:|
| attentive | 18,041,602 | 100.0 | 99.5 | 97.7 | 98.1 |
| tiny | 3,842 | 95.7 | 91.2 | 93.1 | 85.2 |

## attentive

**ssv2 정방향 test** — n 216, 정확도 99.5%

| 진실 \ 예측 | left | right |
|---|---:|---:|
| left | 108 | 0 |
| right | 1 | 107 |


**ssv2 역재생 test (진실 = 뒤집은 라벨)** — n 216, 정확도 97.7%

| 진실 \ 예측 | left | right |
|---|---:|---:|
| left | 105 | 3 |
| right | 2 | 106 |


## tiny

**ssv2 정방향 test** — n 216, 정확도 91.2%

| 진실 \ 예측 | left | right |
|---|---:|---:|
| left | 99 | 9 |
| right | 10 | 98 |


**ssv2 역재생 test (진실 = 뒤집은 라벨)** — n 216, 정확도 93.1%

| 진실 \ 예측 | left | right |
|---|---:|---:|
| left | 98 | 10 |
| right | 5 | 103 |


## 재현

```bash
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
$P z_research/scripts/analysis/ctxenc_direction_probe.py --sets ssv2
```
