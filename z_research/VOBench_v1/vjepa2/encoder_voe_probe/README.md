# encoder 표현에 위반이 선형으로 남아 있나 — VOBench · IntPhys 1 · IntPhys 2 (2026-10-12)

V-JEPA 2 ViT-H 의 context encoder (online `encoder`) · target encoder (EMA) 에 **영상 32 장 전체**를 넣고 토큰 (T×N, 1280) 을 **시공간 평균 (1, 1280)** 한 뒤, possible vs impossible 이진 linear probe (표준화 + L2 로지스틱, group 단위 5-fold · 내부 4-fold λ).
group = VOBench / IntPhys 1 block · IntPhys 2 장면. 주지표 = **쌍 정확도** (같은 쌍에서 P(imp) 가 impossible 쪽이 큰가), chance 50. 라벨 섞기 = 쌍 단위로 라벨을 뒤집어 같은 CV (5 회).

## 결과

| 벤치 | 쌍 | context encoder | target encoder | 라벨 섞기 | (참고) VoE 쌍 정확도 |
|---|---:|---:|---:|---:|---|
| VOBench v1 | 1,440 | **99.7** [99.4, 99.9] | **99.9** [99.7, 100] | 49.4 · 49.9 | release 71.7 / 81.9 (prim / real), late 57.9 / 65.7 |
| IntPhys 1 dev | 180 | **98.9** [97.2, 100] | **96.7** [93.9, 98.9] | 50.8 · 48.6 | release 88.9 · AR 92.8 |
| IntPhys 2 Main | 506 | **74.1** [70.6, 77.9] | **72.5** [68.4, 76.5] | 52.3 · 49.1 | release 54.3 · AR 52.6 |

영상 단위 정확도: VOBench 97.4 / 97.7 · IntPhys 1 73.1 / 69.2 · IntPhys 2 60.6 / 61.8 (context / target). 대괄호 = group 부트스트랩 95 %.

- **VOBench · IntPhys 1 은 천장이다.** VOBench 는 objset · 위반 · 가림 시점 24 칸이 전부 96–100 — **late 도 100** (VoE 는 late 에서 57.9 / 65.7). 정보는 encoder 에 있고 VoE 가 late 에서 무너지는 곳은 그 뒤 (predictor · 채점) 이다 (정본 P3 "encoder 에는 정보가 있다" 와 같은 방향).
- **VOBench late 만 (432 쌍)**: context 100.0 · target 99.8 (permanence 100 / 100 · shape 100 / 99.3 · colour 100 / 100), 라벨 섞기 49–52. 재실행의 전체 쌍 정확도 99.7 / 99.8 은 위 표 (99.7 / 99.9) 와 같다 (fold 난수 차).
- **IntPhys 2 는 encoder 에서도 절반만 남는다** (74). condition 별 context / target: continuity 74.2 / 74.2 · immutability 84.2 / 79.2 · permanence 77.5 / 83.3 · **solidity 63.0 / 56.8**; camera Fixed 78.4 / 77.9 · Moving 71.1 / 68.8.
  VoE (52–54) 보다는 20 pt 높지만 VOBench · IntPhys 1 처럼 천장이 아니다 — IntPhys 2 에서 predictor 가 못 오르는 것을 predictor 탓으로만 돌릴 수 없다.
- IntPhys 2 수치는 9 월 감사 (`auto_research/Archive/INTPHYS2_ENCODER_LINEAR_PROBE_2026-09-30.md`, context 73.7) 와 같은 특징 · 같은 probe 로 재현된다 (fold 난수만 다름).

## 단서 (결론과 같은 비중)

- **지도 학습된 자다.** probe 는 impossible 라벨을 보고 "위반 뒤 장면 통계" (사라진 화면 · 바뀐 색 · 모양) 나 **영상 안에서 정체가 바뀌는 사건 자체**를 배울 수 있다. 예측 없이 영상 전체를 본 판정이라 VoE 와 같은 능력을 재지 않는다. "encoder 가 물리를 안다" 로 쓰지 않는다.
- **VOBench 의 impossible 은 두 렌더의 splice 다** (프레임 48 에서 다른 렌더로 바뀜, 렌더 노이즈 차 평균 0.72 levels · 픽셀 72 %). possible 은 렌더 하나라, probe 가 위반이 아니라 이 이음매를 읽었을 가능성을 이 실험은 배제하지 못한다. 배제하려면 "같은 사건을 다른 노이즈 시드로 이어 붙인 possible" 대조가 필요하다.
- 입력 프레임이 벤치마다 다르다: VOBench · IntPhys 1 은 원본 stride 3 (raw 0–93), IntPhys 2 는 60 fps stride 18 × 32 장으로 위반 시점 (`intphys2_onset_main_v2.json`) 을 포함하도록 시작점을 민 것. VoE 채점 창과도 다르다.
- 쌍 정확도는 장면 공통 성분을 지운다 — 영상 단위 정확도 (IntPhys 1 73 vs 쌍 99) 와 크게 다를 수 있다. 표의 두 줄을 같이 읽는다.
- target encoder 는 LN 전 출력 (probe 가 표준화한다). VoE 의 타깃 공간 (LN(h)) 과 다르다.

## 파일

| 파일 | 무엇 |
|---|---|
| `results.json` | 벤치 × encoder: acc · pair_acc · pair_ci · λ · breakdown (VOBench objset / violation / occlusion / 셋 조합, IntPhys 1 block_type, IntPhys 2 condition · camera) · perm_pair_acc |
| `figures/fig_encoder_probe_bench` | 벤치 3 × encoder 2, 쌍 정확도 [95 % CI], 회색 = 라벨 섞기 평균 |
| `figures/fig_encoder_probe_vobench` | VOBench objset × 위반, x = 가림 시점 |
| `figures/fig_encoder_probe_vobench_late` | VOBench late 쌍만 (전체 · 위반별, primitive + realistic), [block 부트스트랩 95 %], 회색 = 라벨 섞기. probe 는 VOBench 전체로 학습한 것 (2026-10-14) |
| `results_vobench_pairs.json` | 위 그림의 VOBench 재실행 — 쌍별 적중 · 부분집합 (all · late · late/위반) 정확도 · CI · 라벨 섞기 |

특징 캐시: `/local_datasets/world/world_analysis/cache/encoder_voe_probe/{vobench_v1,intphys1_dev}/` (vll5, 튜블릿별 공간 평균 (16, 1280) fp16) · IntPhys 2 는 `auto_research/exp_results/intphys2_probe/shard*.npz`.

## 재현

```bash
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
$P z_research/scripts/analysis/encoder_voe_probe.py extract intphys1_dev --gpus 4    # vll5, 30 s
$P z_research/scripts/analysis/encoder_voe_probe.py extract vobench_v1 --gpus 4      # vll5, 3 분
$P z_research/scripts/analysis/encoder_voe_probe.py probe --n-perm 5                 # CPU, 약 20 분
$P z_research/scripts/analysis/encoder_voe_probe.py plot
$P z_research/scripts/analysis/encoder_voe_probe.py late --n-perm 5             # VOBench 만 다시 → late 그림 (CPU, 부하에 따라 15–50 분)
```
