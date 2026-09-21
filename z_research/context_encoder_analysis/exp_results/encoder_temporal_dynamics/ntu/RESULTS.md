# ntu — 문맥 encoder 의 좌/우 운동 방향 (정방향 학습 → 역재생 시험)

clip 416 (left 208 / right 208), train 260 / test 156, split = performer 단위, 균등 32 장 → 256x256 (full), ViT-H 문맥 encoder (frozen), probe 는 아래 표.

역재생 표의 진실 라벨은 **뒤집은 것** 이다 (left 영상을 거꾸로 틀면 오른쪽으로 간다). 뒤집힘 비율 = 같은 clip 의 정방향 예측과 역재생 예측이 다른 비율 (라벨이 필요 없는 검사).

| probe | 파라미터 | train | 정방향 test | 역재생 test (뒤집힌 라벨) | 뒤집힘 비율 |
|---|---:|---:|---:|---:|---:|
| attentive | 18,041,602 | 100.0 | 100.0 | 96.2 | 96.2 |
| tiny | 3,842 | 99.6 | 100.0 | 96.2 | 96.2 |

## attentive

**ntu 정방향 test** — n 156, 정확도 100.0%

| 진실 \ 예측 | left | right |
|---|---:|---:|
| left | 79 | 0 |
| right | 0 | 77 |


**ntu 역재생 test (진실 = 뒤집은 라벨)** — n 156, 정확도 96.2%

| 진실 \ 예측 | left | right |
|---|---:|---:|
| left | 77 | 0 |
| right | 6 | 73 |


## tiny

**ntu 정방향 test** — n 156, 정확도 100.0%

| 진실 \ 예측 | left | right |
|---|---:|---:|
| left | 79 | 0 |
| right | 0 | 77 |


**ntu 역재생 test (진실 = 뒤집은 라벨)** — n 156, 정확도 96.2%

| 진실 \ 예측 | left | right |
|---|---:|---:|
| left | 77 | 0 |
| right | 6 | 73 |


## 재현

```bash
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
$P z_research/scripts/analysis/ctxenc_direction_probe.py --sets ntu
```
