# VOBench split test block — post-FT 전후 · 창 c16t24 (720 쌍, 360 block)

모델 = 전체 채점에서 test block 만 고름 · post-FT = split_test 로 채점. 복사 = AR 공간 copy_blk (AR 계열과 같은 공간). 칸 = 쌍 정확도 %.

## primitive · overall

| 칸 | 쌍 | release | ar_ep40 | ar_state | ar_state+FT | copy (AR space) |
|---|---:|---:|---:|---:|---:|---:|
| all | 360 | 76.4 | 84.7 | 83.3 | 88.9 | 78.6 |

## primitive · 위반 × 가림

| 칸 | 쌍 | release | ar_ep40 | ar_state | ar_state+FT | copy (AR space) |
|---|---:|---:|---:|---:|---:|---:|
| permanence/visible | 12 | 100.0 | 100.0 | 100.0 | 100.0 | 91.7 |
| permanence/early | 36 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| permanence/mid | 36 | 97.2 | 100.0 | 100.0 | 100.0 | 100.0 |
| permanence/late | 36 | 61.1 | 97.2 | 97.2 | 100.0 | 55.6 |
| shape/visible | 12 | 75.0 | 100.0 | 91.7 | 100.0 | 100.0 |
| shape/early | 36 | 80.6 | 80.6 | 80.6 | 80.6 | 83.3 |
| shape/mid | 36 | 63.9 | 83.3 | 80.6 | 94.4 | 83.3 |
| shape/late | 36 | 55.6 | 66.7 | 66.7 | 75.0 | 61.1 |
| color/visible | 12 | 83.3 | 91.7 | 75.0 | 91.7 | 100.0 |
| color/early | 36 | 77.8 | 83.3 | 80.6 | 86.1 | 83.3 |
| color/mid | 36 | 75.0 | 72.2 | 75.0 | 80.6 | 72.2 |
| color/late | 36 | 66.7 | 66.7 | 63.9 | 75.0 | 50.0 |

## primitive · late × 운동 (위반 셋 합침)

| 칸 | 쌍 | release | ar_ep40 | ar_state | ar_state+FT | copy (AR space) |
|---|---:|---:|---:|---:|---:|---:|
| flat_static | 18 | 66.7 | 88.9 | 94.4 | 100.0 | 61.1 |
| flat_vel | 18 | 55.6 | 72.2 | 72.2 | 83.3 | 44.4 |
| flat_acc | 18 | 66.7 | 83.3 | 83.3 | 77.8 | 66.7 |
| flat_dec | 18 | 55.6 | 66.7 | 72.2 | 77.8 | 55.6 |
| ramp_down | 18 | 55.6 | 66.7 | 61.1 | 72.2 | 50.0 |
| ramp_up | 18 | 66.7 | 83.3 | 72.2 | 88.9 | 55.6 |

## primitive · late permanence 방향

| 칸 | 쌍 | release | ar_ep40 | ar_state | ar_state+FT | copy (AR space) |
|---|---:|---:|---:|---:|---:|---:|
| A | 18 | 22.2 | 100.0 | 100.0 | 100.0 | 55.6 |
| B | 18 | 100.0 | 94.4 | 94.4 | 100.0 | 55.6 |

## realistic · overall

| 칸 | 쌍 | release | ar_ep40 | ar_state | ar_state+FT | copy (AR space) |
|---|---:|---:|---:|---:|---:|---:|
| all | 360 | 88.9 | 90.0 | 88.9 | 93.3 | 81.9 |

## realistic · 위반 × 가림

| 칸 | 쌍 | release | ar_ep40 | ar_state | ar_state+FT | copy (AR space) |
|---|---:|---:|---:|---:|---:|---:|
| permanence/visible | 12 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| permanence/early | 36 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| permanence/mid | 36 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| permanence/late | 36 | 55.6 | 97.2 | 97.2 | 100.0 | 55.6 |
| shape/visible | 12 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| shape/early | 36 | 94.4 | 100.0 | 97.2 | 100.0 | 100.0 |
| shape/mid | 36 | 91.7 | 86.1 | 77.8 | 94.4 | 86.1 |
| shape/late | 36 | 80.6 | 61.1 | 63.9 | 83.3 | 55.6 |
| color/visible | 12 | 100.0 | 91.7 | 91.7 | 83.3 | 91.7 |
| color/early | 36 | 91.7 | 94.4 | 94.4 | 94.4 | 94.4 |
| color/mid | 36 | 83.3 | 86.1 | 86.1 | 91.7 | 75.0 |
| color/late | 36 | 91.7 | 77.8 | 75.0 | 75.0 | 55.6 |

## realistic · late × 운동 (위반 셋 합침)

| 칸 | 쌍 | release | ar_ep40 | ar_state | ar_state+FT | copy (AR space) |
|---|---:|---:|---:|---:|---:|---:|
| flat_static | 18 | 83.3 | 88.9 | 94.4 | 100.0 | 66.7 |
| flat_vel | 18 | 77.8 | 66.7 | 66.7 | 77.8 | 50.0 |
| flat_acc | 18 | 77.8 | 88.9 | 88.9 | 94.4 | 66.7 |
| flat_dec | 18 | 72.2 | 88.9 | 83.3 | 88.9 | 50.0 |
| ramp_down | 18 | 72.2 | 66.7 | 66.7 | 77.8 | 50.0 |
| ramp_up | 18 | 72.2 | 72.2 | 72.2 | 77.8 | 50.0 |

## realistic · late permanence 방향

| 칸 | 쌍 | release | ar_ep40 | ar_state | ar_state+FT | copy (AR space) |
|---|---:|---:|---:|---:|---:|---:|
| A | 18 | 11.1 | 100.0 | 100.0 | 100.0 | 27.8 |
| B | 18 | 100.0 | 94.4 | 94.4 | 100.0 | 83.3 |

