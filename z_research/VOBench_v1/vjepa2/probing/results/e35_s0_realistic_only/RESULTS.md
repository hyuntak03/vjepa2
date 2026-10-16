# VOBench_v1 튜블릿 정체 probe — h (가림 없음) 로 학습한 head 를 h 가림 · p 에 (ViT-H 릴리즈)

head = AttentiveClassifier (depth 1, query 1, head 16) · {'epochs': 35, 'batch_size': 16, 'lr': 0.0001, 'weight_decay': 0.05} · 라벨 = (모양/색) 결합 + none · block 50/50 (seed 0) · objset 따로.
정의 · 재현은 `z_research/scripts/analysis/vobench_tubelet_probe.py` 머리말. 칸 = n · 결합 정확도 · none 이라고 답한 % · 다른 물체로 답한 % · 모양만 · 색만 맞힘.

## realistic (클래스 38, 학습 튜블릿 4958)

| 칸 | n | 결합 정확도 | none % | 다른 물체 % | 모양 | 색 |
|---|---:|---:|---:|---:|---:|---:|
| ① h_vis · object | 4184 | 89.6 | 0.0 | 10.4 | 97.1 | 91.8 |
| ① h_vis · empty | 794 | 100.0 | 100.0 | 0.0 | nan | nan |
| ② h_occ · object | 416 | 13.5 | 21.2 | 65.4 | 31.2 | 16.8 |
| ② h_occ · empty | 86 | 100.0 | 100.0 | 0.0 | nan | nan |
| ③ p_vis · object | 1600 | 62.6 | 17.8 | 19.6 | 69.6 | 71.6 |
| ③ p_vis · empty | 352 | 100.0 | 100.0 | 0.0 | nan | nan |
| ④ p_late · object, 정답 가림 | 70 | 0.0 | 95.7 | 4.3 | 0.0 | 0.0 |
| ④ p_late · object, 정답 재등장 | 685 | 0.7 | 90.5 | 8.8 | 1.5 | 1.8 |
| ④ p_late · empty | 112 | 100.0 | 100.0 | 0.0 | nan | nan |

미래 튜블릿 t 별 결합 정확도 (none %):

| 칸 | t8 | t9 | t10 | t11 | t12 | t13 | t14 | t15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ③ p_vis · object | 88 (0) | 86 (0) | 70 (6) | 66 (15) | 56 (24) | 50 (22) | 46 (41) | 38 (35) |
| ③ p_vis · empty | 100 (100) | 100 (100) | 100 (100) | 100 (100) | 100 (100) | 100 (100) | 100 (100) | 100 (100) |
| ④ p_late · object, 정답 가림 | 0 (96) | — | — | — | — | — | — | — |
| ④ p_late · object, 정답 재등장 | — | 1 (79) | 1 (92) | 0 (97) | 1 (93) | 1 (80) | 0 (96) | 1 (92) |
| ④ p_late · empty | 100 (100) | 100 (100) | 100 (100) | 100 (100) | 100 (100) | 100 (100) | 100 (100) | 100 (100) |

## 재현

```bash
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
$P z_research/scripts/analysis/vobench_tubelet_probe.py extract --gpus 4
$P z_research/scripts/analysis/vobench_tubelet_probe.py probe
```
