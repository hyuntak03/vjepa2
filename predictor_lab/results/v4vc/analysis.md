# v4_velocity_control 분석 (자동 생성)

## ① 같은 장면 · 다른 속도 시험 (평면) — 묶음 305 개, 정답률 % [95 % CI]

| 모델 | 사라짐 | 나타남 | 속도 바꿔치기 (전체) | 바꿔치기 · 옆으로 나옴 | 바꿔치기 · trapdoor | 바꿔치기 차 1 단계 | 차 3 단계 | 제 속도 고르기 (우연 25) |
|---|---|---|---|---|---|---|---|---|
| 복사 | 46.5 [42.0, 51.1] | 95.3 [93.1, 97.3] | 51.0 [48.9, 53.1] | 51.2 [48.6, 53.7] | 50.5 [46.5, 54.3] | 52.0 [49.4, 54.6] | 50.8 [48.0, 53.6] | 31.5 [29.0, 34.1] |
| 원래 (release) | 2.8 [1.7, 4.0] | 100.0 [100.0, 100.0] | 67.9 [65.7, 70.2] | 69.8 [67.2, 72.5] | 63.5 [59.1, 67.9] | 67.5 [65.0, 70.1] | 69.5 [66.6, 72.5] | 48.4 [45.2, 51.4] |
| VOBench 학습 | 64.3 [60.0, 68.8] | 100.0 [100.0, 100.0] | 72.5 [70.5, 74.2] | 73.9 [71.9, 75.9] | 69.0 [65.0, 73.1] | 69.0 [67.0, 71.1] | 80.3 [77.7, 83.1] | 51.4 [48.6, 54.3] |
| v3 전부 | 74.1 [69.4, 78.3] | 100.0 [100.0, 100.0] | 72.3 [70.4, 74.3] | 76.2 [74.2, 78.4] | 62.9 [59.1, 66.9] | 69.0 [66.8, 71.1] | 81.5 [78.5, 84.3] | 51.8 [48.9, 54.8] |
| trapdoor 만 | 26.8 [22.6, 31.1] | 100.0 [100.0, 100.0] | 69.2 [67.0, 71.3] | 71.7 [69.2, 74.1] | 63.3 [59.4, 66.9] | 67.0 [64.8, 69.4] | 74.4 [71.5, 77.4] | 49.5 [46.4, 52.5] |
| v3 섞음 | 72.1 [67.8, 76.6] | 100.0 [100.0, 100.0] | 70.7 [68.9, 72.7] | 73.9 [71.8, 76.0] | 63.3 [59.3, 67.3] | 67.6 [65.5, 69.6] | 79.2 [76.4, 82.0] | 50.2 [47.5, 53.0] |
| v4 평면 2k | 99.8 [99.6, 100.0] | 100.0 [100.0, 100.0] | 94.8 [93.9, 95.7] | 95.8 [94.8, 96.7] | 92.3 [90.2, 94.3] | 90.4 [88.8, 91.8] | 99.7 [99.2, 100.0] | 85.7 [83.2, 88.0] |
| v4 평면 6k | 99.9 [99.8, 100.0] | 100.0 [100.0, 100.0] | 96.4 [95.6, 97.2] | 97.3 [96.5, 98.1] | 94.3 [92.7, 95.9] | 93.1 [91.6, 94.4] | 100.0 [100.0, 100.0] | 89.7 [87.5, 91.9] |
| v4 평면+빗면 4k | 99.9 [99.8, 100.0] | 100.0 [100.0, 100.0] | 93.4 [92.3, 94.5] | 94.5 [93.3, 95.6] | 90.9 [88.4, 93.3] | 88.5 [86.7, 90.3] | 99.3 [98.7, 99.8] | 83.0 [80.3, 85.5] |

바꿔치기 McNemar: v4vc_fam_s0 vs release: 1063 대 81 p=4.6e-219 · v4vc_fam6k_s0 vs release: 1106 대 63 p=4.6e-247 · v4vc_fam6k_s0 vs v4vc_fam_s0: 80 대 19 p=4.4e-10 · v4vc_fam6k_s0 vs vobftg: 923 대 46 p=6.1e-213 · v4vc_all_s0 vs v4vc_fam_s0: 48 대 96 p=7.8e-05 · v4vc_all_s0 vs release: 1032 대 98 p=3.4e-197 · v4vc_all_s0 vs v3_full_s0: 874 대 99 p=1.2e-155

## ① 같은 장면 · 다른 속도 시험 (빗면) — 묶음 218 개, 정답률 % [95 % CI]

| 모델 | 사라짐 | 나타남 | 속도 바꿔치기 (전체) | 바꿔치기 · 옆으로 나옴 | 바꿔치기 · trapdoor | 바꿔치기 차 1 단계 | 차 3 단계 | 제 속도 고르기 (우연 25) |
|---|---|---|---|---|---|---|---|---|
| 복사 | 25.6 [21.2, 29.7] | 99.2 [98.2, 100.0] | 54.2 [51.3, 57.0] | 53.7 [49.8, 57.9] | 55.0 [51.4, 58.5] | 55.7 [52.5, 58.8] | 50.0 [46.1, 54.1] | 36.2 [32.9, 39.6] |
| 원래 (release) | 0.3 [0.0, 0.8] | 100.0 [100.0, 100.0] | 66.2 [63.6, 68.7] | 69.0 [65.7, 72.2] | 61.6 [57.8, 65.3] | 67.7 [64.6, 70.6] | 64.0 [60.8, 67.4] | 44.5 [40.9, 48.1] |
| VOBench 학습 | 36.7 [31.2, 42.0] | 100.0 [100.0, 100.0] | 62.3 [60.0, 64.7] | 64.9 [62.0, 67.8] | 57.8 [54.6, 61.1] | 64.6 [62.2, 67.2] | 59.6 [56.7, 62.6] | 38.9 [35.7, 42.0] |
| v3 전부 | 46.7 [41.1, 52.3] | 100.0 [100.0, 100.0] | 65.1 [62.5, 67.5] | 69.3 [65.8, 72.6] | 57.9 [54.9, 61.1] | 64.8 [62.2, 67.5] | 67.2 [64.0, 70.6] | 43.9 [40.6, 47.5] |
| trapdoor 만 | 39.2 [34.4, 44.0] | 100.0 [100.0, 100.0] | 65.2 [62.9, 67.7] | 68.7 [65.4, 72.0] | 59.2 [56.1, 62.3] | 65.4 [62.7, 68.0] | 66.1 [62.4, 69.5] | 43.6 [40.3, 46.9] |
| v4 평면 2k | 58.3 [52.8, 63.5] | 100.0 [100.0, 100.0] | 59.5 [57.6, 61.6] | 63.2 [60.4, 65.8] | 53.2 [50.6, 55.8] | 59.8 [57.8, 61.9] | 63.1 [59.9, 66.3] | 34.5 [31.9, 37.4] |
| v4 평면 6k | 51.1 [46.0, 56.5] | 100.0 [100.0, 100.0] | 57.5 [55.7, 59.4] | 60.2 [57.9, 62.7] | 52.9 [50.4, 55.3] | 57.6 [55.7, 59.6] | 59.9 [56.9, 63.1] | 31.4 [29.1, 33.9] |
| v4 평면+빗면 4k | 96.7 [94.5, 98.3] | 100.0 [100.0, 100.0] | 86.7 [84.7, 88.6] | 91.7 [90.0, 93.5] | 78.0 [74.1, 81.9] | 80.4 [78.0, 82.9] | 96.1 [94.3, 97.7] | 70.4 [66.7, 74.1] |

바꿔치기 McNemar: v4vc_fam_s0 vs release: 549 대 725 p=9.1e-07 · v4vc_fam6k_s0 vs release: 546 대 774 p=3.8e-10 · v4vc_fam6k_s0 vs v4vc_fam_s0: 26 대 78 p=3.3e-07 · v4vc_fam6k_s0 vs vobftg: 168 대 292 p=8e-09 · v4vc_all_s0 vs v4vc_fam_s0: 898 대 187 p=7.2e-112 · v4vc_all_s0 vs release: 723 대 188 p=1.3e-74 · v4vc_all_s0 vs v3_full_s0: 741 대 177 p=1.1e-82

## ② 바깥 벤치 정답률 %

| 모델 | vob_late/A (n 36) | vob_late/A_moving (n 30) | vob_late/A_static (n 6) | vob_late/B (n 36) | vob_late/all (n 72) | vob_late/A_flat_vel (n 6) | vob_late/A_flat_acc (n 6) | vob_late/A_flat_dec (n 6) | vob_late/A_ramp_down (n 6) | vob_late/A_ramp_up (n 6) | vomo/A (n 96) | vomo/B (n 96) | v3test/A (n 1719) | v3ttest/A (n 446) | clevrer/all (n 612) | ip1uni/all (n 30) | ip1uni/moving (n 20) | ip2uni/all (n 120) | rubric/all (n 40) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 복사 | 38.9 | 36.7 | 50.0 | 63.9 | 51.4 | 16.7 | 0.0 | 0.0 | 66.7 | 100.0 | 76.0 | 28.1 | 31.1 | 43.9 | 65.5 | 63.3 | 55.0 | 50.0 | 55.0 |
| 원래 (release) | 13.9 | 0.0 | 83.3 | 94.4 | 54.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 18.8 | 99.0 | 18.4 | 36.3 | 51.3 | 53.3 | 55.0 | 59.2 | 47.5 |
| VOBench 학습 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 77.1 | 90.6 | 48.9 | 52.5 | 77.0 | 63.3 | 55.0 | 60.0 | 50.0 |
| v3 전부 | 16.7 | 0.0 | 100.0 | 97.2 | 56.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 35.4 | 100.0 | 77.1 | 63.9 | 81.5 | 60.0 | 50.0 | 58.3 | 50.0 |
| trapdoor 만 | 27.8 | 13.3 | 100.0 | 100.0 | 63.9 | 33.3 | 0.0 | 33.3 | 0.0 | 0.0 | 56.2 | 94.8 | 46.8 | 86.8 | 64.2 | 56.7 | 45.0 | 61.7 | 47.5 |
| v3 섞음 | 16.7 | 0.0 | 100.0 | 97.2 | 56.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 35.4 | 100.0 | 76.0 | 62.8 | 82.5 | 60.0 | 50.0 | 58.3 | 50.0 |
| v4 평면 2k | 55.6 | 46.7 | 100.0 | 100.0 | 77.8 | 100.0 | 50.0 | 66.7 | 16.7 | 0.0 | 64.6 | 99.0 | 57.6 | 62.3 | 81.7 | 60.0 | 50.0 | 62.5 | 47.5 |
| v4 평면 6k | 52.8 | 43.3 | 100.0 | 100.0 | 76.4 | 100.0 | 33.3 | 83.3 | 0.0 | 0.0 | 71.9 | 96.9 | 58.0 | 63.2 | 82.8 | 56.7 | 45.0 | 60.8 | 45.0 |
| v4 평면+빗면 4k | 69.4 | 63.3 | 100.0 | 100.0 | 84.7 | 100.0 | 66.7 | 100.0 | 33.3 | 16.7 | 81.2 | 97.9 | 59.1 | 72.2 | 82.7 | 60.0 | 50.0 | 65.0 | 45.0 |

- vob_late/A: v4vc_fam_s0 vs release: 15 대 0 p=6.1e-05 · v4vc_fam6k_s0 vs release: 14 대 0 p=0.00012 · v4vc_fam6k_s0 vs v4vc_fam_s0: 1 대 2 p=1 · v4vc_fam6k_s0 vs vobftg: 0 대 17 p=1.5e-05 · v4vc_all_s0 vs v4vc_fam_s0: 5 대 0 p=0.062 · v4vc_all_s0 vs release: 20 대 0 p=1.9e-06 · v4vc_all_s0 vs v3_full_s0: 19 대 0 p=3.8e-06
- vob_late/A_moving: v4vc_fam_s0 vs release: 14 대 0 p=0.00012 · v4vc_fam6k_s0 vs release: 13 대 0 p=0.00024 · v4vc_fam6k_s0 vs v4vc_fam_s0: 1 대 2 p=1 · v4vc_fam6k_s0 vs vobftg: 0 대 17 p=1.5e-05 · v4vc_all_s0 vs v4vc_fam_s0: 5 대 0 p=0.062 · v4vc_all_s0 vs release: 19 대 0 p=3.8e-06 · v4vc_all_s0 vs v3_full_s0: 19 대 0 p=3.8e-06
- vob_late/A_static: v4vc_fam_s0 vs release: 1 대 0 p=1 · v4vc_fam6k_s0 vs release: 1 대 0 p=1 · v4vc_fam6k_s0 vs v4vc_fam_s0: 0 대 0 p=1 · v4vc_fam6k_s0 vs vobftg: 0 대 0 p=1 · v4vc_all_s0 vs v4vc_fam_s0: 0 대 0 p=1 · v4vc_all_s0 vs release: 1 대 0 p=1 · v4vc_all_s0 vs v3_full_s0: 0 대 0 p=1
- vob_late/B: v4vc_fam_s0 vs release: 2 대 0 p=0.5 · v4vc_fam6k_s0 vs release: 2 대 0 p=0.5 · v4vc_fam6k_s0 vs v4vc_fam_s0: 0 대 0 p=1 · v4vc_fam6k_s0 vs vobftg: 0 대 0 p=1 · v4vc_all_s0 vs v4vc_fam_s0: 0 대 0 p=1 · v4vc_all_s0 vs release: 2 대 0 p=0.5 · v4vc_all_s0 vs v3_full_s0: 1 대 0 p=1
- vob_late/all: v4vc_fam_s0 vs release: 17 대 0 p=1.5e-05 · v4vc_fam6k_s0 vs release: 16 대 0 p=3.1e-05 · v4vc_fam6k_s0 vs v4vc_fam_s0: 1 대 2 p=1 · v4vc_fam6k_s0 vs vobftg: 0 대 17 p=1.5e-05 · v4vc_all_s0 vs v4vc_fam_s0: 5 대 0 p=0.062 · v4vc_all_s0 vs release: 22 대 0 p=4.8e-07 · v4vc_all_s0 vs v3_full_s0: 20 대 0 p=1.9e-06
- vob_late/A_flat_vel: v4vc_fam_s0 vs release: 6 대 0 p=0.031 · v4vc_fam6k_s0 vs release: 6 대 0 p=0.031 · v4vc_fam6k_s0 vs v4vc_fam_s0: 0 대 0 p=1 · v4vc_fam6k_s0 vs vobftg: 0 대 0 p=1 · v4vc_all_s0 vs v4vc_fam_s0: 0 대 0 p=1 · v4vc_all_s0 vs release: 6 대 0 p=0.031 · v4vc_all_s0 vs v3_full_s0: 6 대 0 p=0.031
- vob_late/A_flat_acc: v4vc_fam_s0 vs release: 3 대 0 p=0.25 · v4vc_fam6k_s0 vs release: 2 대 0 p=0.5 · v4vc_fam6k_s0 vs v4vc_fam_s0: 0 대 1 p=1 · v4vc_fam6k_s0 vs vobftg: 0 대 4 p=0.12 · v4vc_all_s0 vs v4vc_fam_s0: 1 대 0 p=1 · v4vc_all_s0 vs release: 4 대 0 p=0.12 · v4vc_all_s0 vs v3_full_s0: 4 대 0 p=0.12
- vob_late/A_flat_dec: v4vc_fam_s0 vs release: 4 대 0 p=0.12 · v4vc_fam6k_s0 vs release: 5 대 0 p=0.062 · v4vc_fam6k_s0 vs v4vc_fam_s0: 1 대 0 p=1 · v4vc_fam6k_s0 vs vobftg: 0 대 1 p=1 · v4vc_all_s0 vs v4vc_fam_s0: 2 대 0 p=0.5 · v4vc_all_s0 vs release: 6 대 0 p=0.031 · v4vc_all_s0 vs v3_full_s0: 6 대 0 p=0.031
- vob_late/A_ramp_down: v4vc_fam_s0 vs release: 1 대 0 p=1 · v4vc_fam6k_s0 vs release: 0 대 0 p=1 · v4vc_fam6k_s0 vs v4vc_fam_s0: 0 대 1 p=1 · v4vc_fam6k_s0 vs vobftg: 0 대 6 p=0.031 · v4vc_all_s0 vs v4vc_fam_s0: 1 대 0 p=1 · v4vc_all_s0 vs release: 2 대 0 p=0.5 · v4vc_all_s0 vs v3_full_s0: 2 대 0 p=0.5
- vob_late/A_ramp_up: v4vc_fam_s0 vs release: 0 대 0 p=1 · v4vc_fam6k_s0 vs release: 0 대 0 p=1 · v4vc_fam6k_s0 vs v4vc_fam_s0: 0 대 0 p=1 · v4vc_fam6k_s0 vs vobftg: 0 대 6 p=0.031 · v4vc_all_s0 vs v4vc_fam_s0: 1 대 0 p=1 · v4vc_all_s0 vs release: 1 대 0 p=1 · v4vc_all_s0 vs v3_full_s0: 1 대 0 p=1
- vomo/A: v4vc_fam_s0 vs release: 44 대 0 p=1.1e-13 · v4vc_fam6k_s0 vs release: 52 대 1 p=1.2e-14 · v4vc_fam6k_s0 vs v4vc_fam_s0: 13 대 6 p=0.17 · v4vc_fam6k_s0 vs vobftg: 14 대 19 p=0.49 · v4vc_all_s0 vs v4vc_fam_s0: 16 대 0 p=3.1e-05 · v4vc_all_s0 vs release: 60 대 0 p=1.7e-18 · v4vc_all_s0 vs v3_full_s0: 46 대 2 p=8.4e-12
- vomo/B: v4vc_fam_s0 vs release: 0 대 0 p=1 · v4vc_fam6k_s0 vs release: 0 대 2 p=0.5 · v4vc_fam6k_s0 vs v4vc_fam_s0: 0 대 2 p=0.5 · v4vc_fam6k_s0 vs vobftg: 6 대 0 p=0.031 · v4vc_all_s0 vs v4vc_fam_s0: 0 대 1 p=1 · v4vc_all_s0 vs release: 0 대 1 p=1 · v4vc_all_s0 vs v3_full_s0: 0 대 2 p=0.5
- v3test/A: v4vc_fam_s0 vs release: 708 대 35 p=6e-164 · v4vc_fam6k_s0 vs release: 717 대 37 p=1.9e-164 · v4vc_fam6k_s0 vs v4vc_fam_s0: 49 대 42 p=0.53 · v4vc_fam6k_s0 vs vobftg: 251 대 94 p=1.2e-17 · v4vc_all_s0 vs v4vc_fam_s0: 77 대 51 p=0.027 · v4vc_all_s0 vs release: 738 대 39 p=2.6e-168 · v4vc_all_s0 vs v3_full_s0: 76 대 386 p=5.2e-51
- v3ttest/A: v4vc_fam_s0 vs release: 150 대 34 p=1.4e-18 · v4vc_fam6k_s0 vs release: 154 대 34 p=1.9e-19 · v4vc_fam6k_s0 vs v4vc_fam_s0: 10 대 6 p=0.45 · v4vc_fam6k_s0 vs vobftg: 67 대 19 p=1.9e-07 · v4vc_all_s0 vs v4vc_fam_s0: 57 대 13 p=1e-07 · v4vc_all_s0 vs release: 187 대 27 p=1.2e-30 · v4vc_all_s0 vs v3_full_s0: 86 대 49 p=0.0018
- clevrer/all: v4vc_fam_s0 vs release: 186 대 0 p=2e-56 · v4vc_fam6k_s0 vs release: 193 대 0 p=1.6e-58 · v4vc_fam6k_s0 vs v4vc_fam_s0: 15 대 8 p=0.21 · v4vc_fam6k_s0 vs vobftg: 69 대 33 p=0.00047 · v4vc_all_s0 vs v4vc_fam_s0: 16 대 10 p=0.33 · v4vc_all_s0 vs release: 193 대 1 p=1.6e-56 · v4vc_all_s0 vs v3_full_s0: 39 대 32 p=0.48
- ip1uni/all: v4vc_fam_s0 vs release: 4 대 2 p=0.69 · v4vc_fam6k_s0 vs release: 3 대 2 p=1 · v4vc_fam6k_s0 vs v4vc_fam_s0: 0 대 1 p=1 · v4vc_fam6k_s0 vs vobftg: 0 대 2 p=0.5 · v4vc_all_s0 vs v4vc_fam_s0: 0 대 0 p=1 · v4vc_all_s0 vs release: 4 대 2 p=0.69 · v4vc_all_s0 vs v3_full_s0: 0 대 0 p=1
- ip1uni/moving: v4vc_fam_s0 vs release: 1 대 2 p=1 · v4vc_fam6k_s0 vs release: 0 대 2 p=0.5 · v4vc_fam6k_s0 vs v4vc_fam_s0: 0 대 1 p=1 · v4vc_fam6k_s0 vs vobftg: 0 대 2 p=0.5 · v4vc_all_s0 vs v4vc_fam_s0: 0 대 0 p=1 · v4vc_all_s0 vs release: 1 대 2 p=1 · v4vc_all_s0 vs v3_full_s0: 0 대 0 p=1
- ip2uni/all: v4vc_fam_s0 vs release: 9 대 5 p=0.42 · v4vc_fam6k_s0 vs release: 9 대 7 p=0.8 · v4vc_fam6k_s0 vs v4vc_fam_s0: 2 대 4 p=0.69 · v4vc_fam6k_s0 vs vobftg: 8 대 7 p=1 · v4vc_all_s0 vs v4vc_fam_s0: 5 대 2 p=0.45 · v4vc_all_s0 vs release: 14 대 7 p=0.19 · v4vc_all_s0 vs v3_full_s0: 9 대 1 p=0.021
- rubric/all: v4vc_fam_s0 vs release: 2 대 2 p=1 · v4vc_fam6k_s0 vs release: 1 대 2 p=1 · v4vc_fam6k_s0 vs v4vc_fam_s0: 0 대 1 p=1 · v4vc_fam6k_s0 vs vobftg: 0 대 2 p=0.5 · v4vc_all_s0 vs v4vc_fam_s0: 0 대 1 p=1 · v4vc_all_s0 vs release: 1 대 2 p=1 · v4vc_all_s0 vs v3_full_s0: 1 대 3 p=0.62