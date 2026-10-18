# VOBench_v1 튜블릿 정체 probe — h (가림 없음) 로 학습한 head 를 h 가림 · p 에 (ViT-H 릴리즈)

head = AttentiveClassifier (depth 1, query 1, head 16) · {'epochs': 35, 'batch_size': 16, 'lr': 0.0001, 'weight_decay': 0.05} · 라벨 = (모양/색) 결합 + none · block 50/50 (seed 0) · objset 따로.
정의 · 재현은 `z_research/scripts/analysis/vobench_tubelet_probe.py` 머리말. 칸 = n · 결합 정확도 · none 이라고 답한 % · 다른 물체로 답한 % · 모양만 · 색만 맞힘.

## primitive (클래스 49, 학습 튜블릿 4964)

| 칸 | n | 결합 정확도 | none % | 다른 물체 % | 모양 | 색 |
|---|---:|---:|---:|---:|---:|---:|
| ① h_vis · object | 4054 | 99.1 | 0.0 | 0.9 | 99.4 | 99.7 |
| ① h_vis · empty | 918 | 100.0 | 100.0 | 0.0 | nan | nan |
| ② h_occ · object | 408 | 0.5 | 74.3 | 25.2 | 2.2 | 5.4 |
| ② h_occ · empty | 100 | 100.0 | 100.0 | 0.0 | nan | nan |
| ③ p_vis · object | 1528 | 4.5 | 13.4 | 82.1 | 22.0 | 15.1 |
| ③ p_vis · empty | 344 | 52.3 | 52.3 | 47.7 | nan | nan |
| ④ p_late · object, 정답 가림 | 70 | 1.4 | 22.9 | 75.7 | 5.7 | 8.6 |
| ④ p_late · object, 정답 재등장 | 684 | 1.2 | 15.4 | 83.5 | 12.6 | 10.4 |
| ④ p_late · empty | 192 | 29.7 | 29.7 | 70.3 | nan | nan |
| ⑤ h_late_all · object | 1508 | 89.7 | 7.5 | 2.9 | 90.1 | 90.7 |
| ⑤ p_late_all · object | 754 | 1.2 | 16.0 | 82.8 | 11.9 | 10.2 |

미래 튜블릿 t 별 결합 정확도 (none %):

| 칸 | t8 | t9 | t10 | t11 | t12 | t13 | t14 | t15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ③ p_vis · object | 9 (1) | 8 (2) | 6 (15) | 3 (20) | 3 (18) | 3 (17) | 2 (19) | 2 (16) |
| ③ p_vis · empty | 77 (77) | 40 (40) | 81 (81) | 77 (77) | 37 (37) | 40 (40) | 47 (47) | 21 (21) |
| ④ p_late · object, 정답 가림 | 1 (23) | — | — | — | — | — | — | — |
| ④ p_late · object, 정답 재등장 | — | 0 (17) | 1 (32) | 2 (25) | 1 (10) | 0 (6) | 1 (17) | 3 (1) |
| ④ p_late · empty | 25 (25) | 25 (25) | 71 (71) | 42 (42) | 8 (8) | 12 (12) | 46 (46) | 8 (8) |

## realistic (클래스 38, 학습 튜블릿 4958)

| 칸 | n | 결합 정확도 | none % | 다른 물체 % | 모양 | 색 |
|---|---:|---:|---:|---:|---:|---:|
| ① h_vis · object | 4184 | 90.2 | 0.1 | 9.7 | 96.2 | 92.7 |
| ① h_vis · empty | 794 | 100.0 | 100.0 | 0.0 | nan | nan |
| ② h_occ · object | 416 | 0.7 | 85.3 | 13.9 | 3.8 | 2.6 |
| ② h_occ · empty | 86 | 100.0 | 100.0 | 0.0 | nan | nan |
| ③ p_vis · object | 1600 | 10.8 | 22.9 | 66.2 | 19.2 | 27.6 |
| ③ p_vis · empty | 352 | 83.8 | 83.8 | 16.2 | nan | nan |
| ④ p_late · object, 정답 가림 | 70 | 0.0 | 97.1 | 2.9 | 0.0 | 2.9 |
| ④ p_late · object, 정답 재등장 | 685 | 3.6 | 71.1 | 25.3 | 6.4 | 15.3 |
| ④ p_late · empty | 112 | 93.8 | 93.8 | 6.2 | nan | nan |
| ⑤ h_late_all · object | 1510 | 83.2 | 7.7 | 9.1 | 86.9 | 85.8 |
| ⑤ p_late_all · object | 755 | 3.3 | 73.5 | 23.2 | 5.8 | 14.2 |

미래 튜블릿 t 별 결합 정확도 (none %):

| 칸 | t8 | t9 | t10 | t11 | t12 | t13 | t14 | t15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ③ p_vis · object | 12 (0) | 16 (0) | 18 (19) | 1 (56) | 10 (25) | 18 (14) | 3 (48) | 8 (20) |
| ③ p_vis · empty | 100 (100) | 73 (73) | 100 (100) | 100 (100) | 86 (86) | 57 (57) | 100 (100) | 55 (55) |
| ④ p_late · object, 정답 가림 | 0 (97) | — | — | — | — | — | — | — |
| ④ p_late · object, 정답 재등장 | — | 7 (60) | 0 (100) | 0 (100) | 5 (57) | 8 (41) | 0 (100) | 7 (36) |
| ④ p_late · empty | 100 (100) | 86 (86) | 100 (100) | 100 (100) | 93 (93) | 86 (86) | 100 (100) | 86 (86) |

## 재현

```bash
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
$P z_research/scripts/analysis/vobench_tubelet_probe.py extract --gpus 4
$P z_research/scripts/analysis/vobench_tubelet_probe.py probe
```
