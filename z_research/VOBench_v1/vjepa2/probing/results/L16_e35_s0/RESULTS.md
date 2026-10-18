# VOBench_v1 튜블릿 정체 probe — h (가림 없음) 로 학습한 head 를 h 가림 · p 에 (ViT-H 릴리즈)

head = AttentiveClassifier (depth 1, query 1, head 16) · {'epochs': 35, 'batch_size': 16, 'lr': 0.0001, 'weight_decay': 0.05} · 라벨 = (모양/색) 결합 + none · block 50/50 (seed 0) · objset 따로.
정의 · 재현은 `z_research/scripts/analysis/vobench_tubelet_probe.py` 머리말. 칸 = n · 결합 정확도 · none 이라고 답한 % · 다른 물체로 답한 % · 모양만 · 색만 맞힘.

## primitive (클래스 49, 학습 튜블릿 4964)

| 칸 | n | 결합 정확도 | none % | 다른 물체 % | 모양 | 색 |
|---|---:|---:|---:|---:|---:|---:|
| ① h_vis · object | 4054 | 99.0 | 0.1 | 1.0 | 99.4 | 99.5 |
| ① h_vis · empty | 918 | 100.0 | 100.0 | 0.0 | nan | nan |
| ② h_occ · object | 408 | 0.7 | 71.3 | 27.9 | 2.2 | 7.1 |
| ② h_occ · empty | 100 | 100.0 | 100.0 | 0.0 | nan | nan |
| ③ p_vis · object | 1528 | 4.6 | 3.7 | 91.7 | 21.0 | 18.3 |
| ③ p_vis · empty | 344 | 47.7 | 47.7 | 52.3 | nan | nan |
| ④ p_late · object, 정답 가림 | 70 | 0.0 | 58.6 | 41.4 | 2.9 | 4.3 |
| ④ p_late · object, 정답 재등장 | 684 | 0.7 | 5.7 | 93.6 | 13.5 | 12.3 |
| ④ p_late · empty | 192 | 37.0 | 37.0 | 63.0 | nan | nan |
| ⑤ h_late_all · object | 1508 | 89.5 | 7.4 | 3.1 | 90.2 | 90.3 |
| ⑤ p_late_all · object | 754 | 0.7 | 10.6 | 88.7 | 12.5 | 11.5 |

미래 튜블릿 t 별 결합 정확도 (none %):

| 칸 | t8 | t9 | t10 | t11 | t12 | t13 | t14 | t15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ③ p_vis · object | 12 (0) | 9 (0) | 3 (0) | 3 (2) | 2 (9) | 2 (0) | 3 (13) | 3 (6) |
| ③ p_vis · empty | 77 (77) | 33 (33) | 40 (40) | 58 (58) | 53 (53) | 30 (30) | 58 (58) | 33 (33) |
| ④ p_late · object, 정답 가림 | 0 (59) | — | — | — | — | — | — | — |
| ④ p_late · object, 정답 재등장 | — | 0 (0) | 0 (2) | 1 (10) | 2 (5) | 2 (0) | 0 (20) | 0 (2) |
| ④ p_late · empty | 71 (71) | 12 (12) | 25 (25) | 58 (58) | 38 (38) | 8 (8) | 58 (58) | 25 (25) |

## realistic (클래스 38, 학습 튜블릿 4958)

| 칸 | n | 결합 정확도 | none % | 다른 물체 % | 모양 | 색 |
|---|---:|---:|---:|---:|---:|---:|
| ① h_vis · object | 4184 | 92.8 | 0.1 | 7.2 | 97.9 | 94.5 |
| ① h_vis · empty | 794 | 100.0 | 100.0 | 0.0 | nan | nan |
| ② h_occ · object | 416 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| ② h_occ · empty | 86 | 100.0 | 100.0 | 0.0 | nan | nan |
| ③ p_vis · object | 1600 | 17.2 | 8.1 | 74.7 | 21.3 | 69.9 |
| ③ p_vis · empty | 352 | 25.3 | 25.3 | 74.7 | nan | nan |
| ④ p_late · object, 정답 가림 | 70 | 7.1 | 45.7 | 47.1 | 10.0 | 40.0 |
| ④ p_late · object, 정답 재등장 | 685 | 9.6 | 26.9 | 63.5 | 13.3 | 53.1 |
| ④ p_late · empty | 112 | 42.0 | 42.0 | 58.0 | nan | nan |
| ⑤ h_late_all · object | 1510 | 85.2 | 9.4 | 5.4 | 88.6 | 86.8 |
| ⑤ p_late_all · object | 755 | 9.4 | 28.6 | 62.0 | 13.0 | 51.9 |

미래 튜블릿 t 별 결합 정확도 (none %):

| 칸 | t8 | t9 | t10 | t11 | t12 | t13 | t14 | t15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ③ p_vis · object | 27 (1) | 24 (0) | 14 (6) | 12 (28) | 18 (0) | 14 (0) | 13 (26) | 15 (4) |
| ③ p_vis · empty | 34 (34) | 32 (32) | 20 (20) | 41 (41) | 9 (9) | 5 (5) | 41 (41) | 20 (20) |
| ④ p_late · object, 정답 가림 | 7 (46) | — | — | — | — | — | — | — |
| ④ p_late · object, 정답 재등장 | — | 12 (27) | 9 (29) | 4 (67) | 12 (7) | 15 (3) | 8 (44) | 9 (11) |
| ④ p_late · empty | 50 (50) | 43 (43) | 43 (43) | 57 (57) | 50 (50) | 21 (21) | 43 (43) | 29 (29) |

## 재현

```bash
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
$P z_research/scripts/analysis/vobench_tubelet_probe.py extract --gpus 4
$P z_research/scripts/analysis/vobench_tubelet_probe.py probe
```
