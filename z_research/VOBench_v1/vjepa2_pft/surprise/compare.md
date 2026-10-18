# VOBench split test block — post-FT 전후 · 창 c16t32 (720 쌍, 360 block)

모델 = 전체 채점에서 test block 만 고름 · post-FT = split_test 로 채점. 복사 = AR 공간 copy_blk (AR 계열과 같은 공간). 칸 = 쌍 정확도 %.

## primitive · overall

| 칸 | 쌍 | release | ar_ep40 | ar_state | release+FT | ar_state+FT | copy (AR space) | copy (release space) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 360 | 70.3 | 81.9 | 81.4 | 89.2 | 89.4 | 76.7 | 72.5 |

## primitive · 위반 × 가림

| 칸 | 쌍 | release | ar_ep40 | ar_state | release+FT | ar_state+FT | copy (AR space) | copy (release space) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| permanence/visible | 12 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 91.7 | 100.0 |
| permanence/early | 36 | 97.2 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| permanence/mid | 36 | 97.2 | 100.0 | 100.0 | 100.0 | 100.0 | 97.2 | 100.0 |
| permanence/late | 36 | 52.8 | 100.0 | 97.2 | 100.0 | 100.0 | 58.3 | 55.6 |
| shape/visible | 12 | 66.7 | 66.7 | 66.7 | 91.7 | 66.7 | 100.0 | 75.0 |
| shape/early | 36 | 69.4 | 80.6 | 77.8 | 86.1 | 88.9 | 75.0 | 72.2 |
| shape/mid | 36 | 61.1 | 77.8 | 77.8 | 88.9 | 94.4 | 83.3 | 75.0 |
| shape/late | 36 | 58.3 | 61.1 | 63.9 | 80.6 | 83.3 | 52.8 | 58.3 |
| color/visible | 12 | 66.7 | 75.0 | 75.0 | 91.7 | 75.0 | 91.7 | 91.7 |
| color/early | 36 | 75.0 | 72.2 | 75.0 | 88.9 | 75.0 | 69.4 | 69.4 |
| color/mid | 36 | 50.0 | 72.2 | 72.2 | 72.2 | 86.1 | 77.8 | 52.8 |
| color/late | 36 | 63.9 | 75.0 | 69.4 | 80.6 | 86.1 | 58.3 | 52.8 |

## primitive · late × 운동 (위반 셋 합침)

| 칸 | 쌍 | release | ar_ep40 | ar_state | release+FT | ar_state+FT | copy (AR space) | copy (release space) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flat_static | 18 | 50.0 | 100.0 | 100.0 | 94.4 | 100.0 | 61.1 | 33.3 |
| flat_vel | 18 | 61.1 | 72.2 | 77.8 | 88.9 | 100.0 | 55.6 | 66.7 |
| flat_acc | 18 | 61.1 | 77.8 | 77.8 | 88.9 | 94.4 | 61.1 | 72.2 |
| flat_dec | 18 | 50.0 | 72.2 | 66.7 | 72.2 | 83.3 | 55.6 | 61.1 |
| ramp_down | 18 | 72.2 | 77.8 | 72.2 | 94.4 | 77.8 | 55.6 | 55.6 |
| ramp_up | 18 | 55.6 | 72.2 | 66.7 | 83.3 | 83.3 | 50.0 | 44.4 |

## primitive · late permanence 방향

| 칸 | 쌍 | release | ar_ep40 | ar_state | release+FT | ar_state+FT | copy (AR space) | copy (release space) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 18 | 16.7 | 100.0 | 100.0 | 100.0 | 100.0 | 44.4 | 66.7 |
| B | 18 | 88.9 | 100.0 | 94.4 | 100.0 | 100.0 | 72.2 | 44.4 |

## realistic · overall

| 칸 | 쌍 | release | ar_ep40 | ar_state | release+FT | ar_state+FT | copy (AR space) | copy (release space) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 360 | 81.1 | 85.8 | 85.0 | 94.4 | 92.5 | 81.4 | 82.2 |

## realistic · 위반 × 가림

| 칸 | 쌍 | release | ar_ep40 | ar_state | release+FT | ar_state+FT | copy (AR space) | copy (release space) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| permanence/visible | 12 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| permanence/early | 36 | 94.4 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| permanence/mid | 36 | 88.9 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 97.2 |
| permanence/late | 36 | 55.6 | 100.0 | 97.2 | 97.2 | 100.0 | 52.8 | 58.3 |
| shape/visible | 12 | 100.0 | 91.7 | 91.7 | 100.0 | 100.0 | 100.0 | 100.0 |
| shape/early | 36 | 91.7 | 94.4 | 91.7 | 94.4 | 100.0 | 97.2 | 94.4 |
| shape/mid | 36 | 83.3 | 83.3 | 83.3 | 88.9 | 88.9 | 91.7 | 80.6 |
| shape/late | 36 | 72.2 | 63.9 | 63.9 | 94.4 | 88.9 | 55.6 | 69.4 |
| color/visible | 12 | 100.0 | 83.3 | 83.3 | 100.0 | 91.7 | 83.3 | 83.3 |
| color/early | 36 | 72.2 | 83.3 | 83.3 | 91.7 | 80.6 | 91.7 | 86.1 |
| color/mid | 36 | 83.3 | 72.2 | 69.4 | 94.4 | 88.9 | 72.2 | 83.3 |
| color/late | 36 | 69.4 | 69.4 | 69.4 | 83.3 | 80.6 | 58.3 | 58.3 |

## realistic · late × 운동 (위반 셋 합침)

| 칸 | 쌍 | release | ar_ep40 | ar_state | release+FT | ar_state+FT | copy (AR space) | copy (release space) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flat_static | 18 | 77.8 | 88.9 | 94.4 | 88.9 | 100.0 | 61.1 | 55.6 |
| flat_vel | 18 | 72.2 | 72.2 | 72.2 | 100.0 | 94.4 | 55.6 | 66.7 |
| flat_acc | 18 | 66.7 | 94.4 | 88.9 | 94.4 | 94.4 | 61.1 | 61.1 |
| flat_dec | 18 | 66.7 | 72.2 | 72.2 | 100.0 | 77.8 | 55.6 | 83.3 |
| ramp_down | 18 | 61.1 | 72.2 | 66.7 | 83.3 | 88.9 | 50.0 | 55.6 |
| ramp_up | 18 | 50.0 | 66.7 | 66.7 | 83.3 | 83.3 | 50.0 | 50.0 |

## realistic · late permanence 방향

| 칸 | 쌍 | release | ar_ep40 | ar_state | release+FT | ar_state+FT | copy (AR space) | copy (release space) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 18 | 11.1 | 100.0 | 100.0 | 94.4 | 100.0 | 22.2 | 55.6 |
| B | 18 | 100.0 | 100.0 | 94.4 | 100.0 | 100.0 | 83.3 | 61.1 |

