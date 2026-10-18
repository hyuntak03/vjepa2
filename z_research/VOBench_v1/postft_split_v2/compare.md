# VOBench split test block — post-FT 전후 · 창 c16t32 (720 쌍, 360 block)

모델 = 전체 채점에서 test block 만 고름 · post-FT = split_test 로 채점. 복사 = AR 공간 copy_blk (AR 계열과 같은 공간). 칸 = 쌍 정확도 %.

## primitive · overall

| 칸 | 쌍 | release | ar_ep40 | ar_state | copy (AR space) |
|---|---:|---:|---:|---:|---:|
| all | 360 | 70.3 | 81.9 | 81.4 | 76.7 |

## primitive · 위반 × 가림

| 칸 | 쌍 | release | ar_ep40 | ar_state | copy (AR space) |
|---|---:|---:|---:|---:|---:|
| permanence/visible | 12 | 100.0 | 100.0 | 100.0 | 91.7 |
| permanence/early | 36 | 97.2 | 100.0 | 100.0 | 100.0 |
| permanence/mid | 36 | 97.2 | 100.0 | 100.0 | 97.2 |
| permanence/late | 36 | 52.8 | 100.0 | 97.2 | 58.3 |
| shape/visible | 12 | 66.7 | 66.7 | 66.7 | 100.0 |
| shape/early | 36 | 69.4 | 80.6 | 77.8 | 75.0 |
| shape/mid | 36 | 61.1 | 77.8 | 77.8 | 83.3 |
| shape/late | 36 | 58.3 | 61.1 | 63.9 | 52.8 |
| color/visible | 12 | 66.7 | 75.0 | 75.0 | 91.7 |
| color/early | 36 | 75.0 | 72.2 | 75.0 | 69.4 |
| color/mid | 36 | 50.0 | 72.2 | 72.2 | 77.8 |
| color/late | 36 | 63.9 | 75.0 | 69.4 | 58.3 |

## primitive · late × 운동 (위반 셋 합침)

| 칸 | 쌍 | release | ar_ep40 | ar_state | copy (AR space) |
|---|---:|---:|---:|---:|---:|
| flat_static | 18 | 50.0 | 100.0 | 100.0 | 61.1 |
| flat_vel | 18 | 61.1 | 72.2 | 77.8 | 55.6 |
| flat_acc | 18 | 61.1 | 77.8 | 77.8 | 61.1 |
| flat_dec | 18 | 50.0 | 72.2 | 66.7 | 55.6 |
| ramp_down | 18 | 72.2 | 77.8 | 72.2 | 55.6 |
| ramp_up | 18 | 55.6 | 72.2 | 66.7 | 50.0 |

## primitive · late permanence 방향

| 칸 | 쌍 | release | ar_ep40 | ar_state | copy (AR space) |
|---|---:|---:|---:|---:|---:|
| A | 18 | 16.7 | 100.0 | 100.0 | 44.4 |
| B | 18 | 88.9 | 100.0 | 94.4 | 72.2 |

## realistic · overall

| 칸 | 쌍 | release | ar_ep40 | ar_state | copy (AR space) |
|---|---:|---:|---:|---:|---:|
| all | 360 | 81.1 | 85.8 | 85.0 | 81.4 |

## realistic · 위반 × 가림

| 칸 | 쌍 | release | ar_ep40 | ar_state | copy (AR space) |
|---|---:|---:|---:|---:|---:|
| permanence/visible | 12 | 100.0 | 100.0 | 100.0 | 100.0 |
| permanence/early | 36 | 94.4 | 100.0 | 100.0 | 100.0 |
| permanence/mid | 36 | 88.9 | 100.0 | 100.0 | 100.0 |
| permanence/late | 36 | 55.6 | 100.0 | 97.2 | 52.8 |
| shape/visible | 12 | 100.0 | 91.7 | 91.7 | 100.0 |
| shape/early | 36 | 91.7 | 94.4 | 91.7 | 97.2 |
| shape/mid | 36 | 83.3 | 83.3 | 83.3 | 91.7 |
| shape/late | 36 | 72.2 | 63.9 | 63.9 | 55.6 |
| color/visible | 12 | 100.0 | 83.3 | 83.3 | 83.3 |
| color/early | 36 | 72.2 | 83.3 | 83.3 | 91.7 |
| color/mid | 36 | 83.3 | 72.2 | 69.4 | 72.2 |
| color/late | 36 | 69.4 | 69.4 | 69.4 | 58.3 |

## realistic · late × 운동 (위반 셋 합침)

| 칸 | 쌍 | release | ar_ep40 | ar_state | copy (AR space) |
|---|---:|---:|---:|---:|---:|
| flat_static | 18 | 77.8 | 88.9 | 94.4 | 61.1 |
| flat_vel | 18 | 72.2 | 72.2 | 72.2 | 55.6 |
| flat_acc | 18 | 66.7 | 94.4 | 88.9 | 61.1 |
| flat_dec | 18 | 66.7 | 72.2 | 72.2 | 55.6 |
| ramp_down | 18 | 61.1 | 72.2 | 66.7 | 50.0 |
| ramp_up | 18 | 50.0 | 66.7 | 66.7 | 50.0 |

## realistic · late permanence 방향

| 칸 | 쌍 | release | ar_ep40 | ar_state | copy (AR space) |
|---|---:|---:|---:|---:|---:|
| A | 18 | 11.1 | 100.0 | 100.0 | 22.2 |
| B | 18 | 100.0 | 100.0 | 94.4 | 83.3 |

