# VOBench split test block — post-FT 전후 · 창 c16t32 (720 쌍, 360 block)

모델 = 전체 채점에서 test block 만 고름 · post-FT = split_test 로 채점. 복사 = AR 공간 copy_blk (AR 계열과 같은 공간). 칸 = 쌍 정확도 %.

## primitive · overall

| 칸 | 쌍 | release | ar_ep40 | ar_state | ar_state+FT | copy (AR space) |
|---|---:|---:|---:|---:|---:|---:|
| all | 360 | 70.3 | 82.5 | 81.9 | 88.9 | 77.2 |

## primitive · 위반 × 가림

| 칸 | 쌍 | release | ar_ep40 | ar_state | ar_state+FT | copy (AR space) |
|---|---:|---:|---:|---:|---:|---:|
| permanence/visible | 12 | 91.7 | 100.0 | 100.0 | 100.0 | 91.7 |
| permanence/early | 36 | 97.2 | 100.0 | 100.0 | 100.0 | 100.0 |
| permanence/mid | 36 | 97.2 | 100.0 | 100.0 | 100.0 | 100.0 |
| permanence/late | 36 | 58.3 | 97.2 | 97.2 | 100.0 | 55.6 |
| shape/visible | 12 | 75.0 | 83.3 | 83.3 | 83.3 | 100.0 |
| shape/early | 36 | 63.9 | 83.3 | 80.6 | 86.1 | 80.6 |
| shape/mid | 36 | 66.7 | 80.6 | 83.3 | 94.4 | 77.8 |
| shape/late | 36 | 50.0 | 61.1 | 63.9 | 80.6 | 55.6 |
| color/visible | 12 | 66.7 | 83.3 | 75.0 | 83.3 | 91.7 |
| color/early | 36 | 69.4 | 77.8 | 75.0 | 80.6 | 77.8 |
| color/mid | 36 | 58.3 | 69.4 | 66.7 | 83.3 | 77.8 |
| color/late | 36 | 63.9 | 66.7 | 66.7 | 75.0 | 52.8 |

## primitive · late × 운동 (위반 셋 합침)

| 칸 | 쌍 | release | ar_ep40 | ar_state | ar_state+FT | copy (AR space) |
|---|---:|---:|---:|---:|---:|---:|
| flat_static | 18 | 50.0 | 100.0 | 100.0 | 100.0 | 61.1 |
| flat_vel | 18 | 55.6 | 66.7 | 72.2 | 88.9 | 44.4 |
| flat_acc | 18 | 61.1 | 77.8 | 77.8 | 94.4 | 55.6 |
| flat_dec | 18 | 55.6 | 72.2 | 66.7 | 77.8 | 50.0 |
| ramp_down | 18 | 66.7 | 61.1 | 61.1 | 72.2 | 55.6 |
| ramp_up | 18 | 55.6 | 72.2 | 77.8 | 77.8 | 61.1 |

## primitive · late permanence 방향

| 칸 | 쌍 | release | ar_ep40 | ar_state | ar_state+FT | copy (AR space) |
|---|---:|---:|---:|---:|---:|---:|
| A | 18 | 27.8 | 100.0 | 100.0 | 100.0 | 55.6 |
| B | 18 | 88.9 | 94.4 | 94.4 | 100.0 | 55.6 |

## realistic · overall

| 칸 | 쌍 | release | ar_ep40 | ar_state | ar_state+FT | copy (AR space) |
|---|---:|---:|---:|---:|---:|---:|
| all | 360 | 82.2 | 86.4 | 85.0 | 91.7 | 81.9 |

## realistic · 위반 × 가림

| 칸 | 쌍 | release | ar_ep40 | ar_state | ar_state+FT | copy (AR space) |
|---|---:|---:|---:|---:|---:|---:|
| permanence/visible | 12 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| permanence/early | 36 | 94.4 | 100.0 | 100.0 | 100.0 | 100.0 |
| permanence/mid | 36 | 88.9 | 100.0 | 100.0 | 100.0 | 100.0 |
| permanence/late | 36 | 61.1 | 100.0 | 94.4 | 100.0 | 55.6 |
| shape/visible | 12 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| shape/early | 36 | 91.7 | 94.4 | 91.7 | 97.2 | 100.0 |
| shape/mid | 36 | 88.9 | 80.6 | 80.6 | 91.7 | 88.9 |
| shape/late | 36 | 61.1 | 61.1 | 61.1 | 83.3 | 55.6 |
| color/visible | 12 | 100.0 | 83.3 | 83.3 | 83.3 | 83.3 |
| color/early | 36 | 80.6 | 83.3 | 77.8 | 80.6 | 86.1 |
| color/mid | 36 | 80.6 | 77.8 | 75.0 | 88.9 | 75.0 |
| color/late | 36 | 75.0 | 72.2 | 75.0 | 80.6 | 63.9 |

## realistic · late × 운동 (위반 셋 합침)

| 칸 | 쌍 | release | ar_ep40 | ar_state | ar_state+FT | copy (AR space) |
|---|---:|---:|---:|---:|---:|---:|
| flat_static | 18 | 88.9 | 94.4 | 94.4 | 100.0 | 66.7 |
| flat_vel | 18 | 66.7 | 77.8 | 77.8 | 88.9 | 55.6 |
| flat_acc | 18 | 66.7 | 88.9 | 88.9 | 100.0 | 72.2 |
| flat_dec | 18 | 61.1 | 72.2 | 77.8 | 77.8 | 55.6 |
| ramp_down | 18 | 55.6 | 66.7 | 55.6 | 83.3 | 50.0 |
| ramp_up | 18 | 55.6 | 66.7 | 66.7 | 77.8 | 50.0 |

## realistic · late permanence 방향

| 칸 | 쌍 | release | ar_ep40 | ar_state | ar_state+FT | copy (AR space) |
|---|---:|---:|---:|---:|---:|---:|
| A | 18 | 22.2 | 100.0 | 100.0 | 100.0 | 27.8 |
| B | 18 | 100.0 | 100.0 | 88.9 | 100.0 | 83.3 |

