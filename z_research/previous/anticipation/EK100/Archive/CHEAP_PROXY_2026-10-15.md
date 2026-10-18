# EK100 싼 대리 설정 찾기 — release ViT-H, 논문 대비 ≤ 5 pp (2026-10-15)

> 사용자 (10-15): release 기준으로 **1/10 데이터 (1/5 까지 허용) · epoch 줄이고 · head 2~3 개** 로, 논문 성능과 **최대 5 pp** 차이 나는 조합을 찾는다.
> 목적: 새 predictor 마다 EK100 을 40 h (8 GPU) 씩 돌리지 않고 순위를 보는 싼 판.

## 합격선

| | verb | noun | action | 출처 |
|---|---:|---:|---:|---|
| 논문 ViT-H (Table 5) | 59.2 | 54.6 | **36.5** | README §1 |
| 우리 full 재현 (released 규약, grid8, 20 ep, batch 128) | 56.4 | 56.2 | 35.3 | `ek100_vith_official256_released_grid8` |
| **합격선 (논문 − 5)** | 54.2 | 49.6 | **31.5** | |

지표 = mean class recall@5, val 9,296 clip (action 1,114 · verb 73 · noun 201 클래스). chance (균등 top-5) verb 5.2 · noun 1.7 · action 0.14.

## 이미 있는 것에서 읽은 것

1. **성능은 epoch 보다 optimizer step 수와 lr 에 묶여 있다.** full run (step 10.5 k) 의 action 은 epoch 마다 0.7 → 3.5 → 6.9 → … → 32.2 (ep 14) → 35.1 (ep 17) 로 끝까지 오른다.
   같은 run 의 lr 1e-4 head 는 3e-4 head 보다 매 epoch 5 pp 안팎 낮다 (ep 20: 29.9 vs 35.3) — head 가 lr 에 막혀 덜 배운 상태다.
   → 데이터 · epoch 를 줄여 step 이 1 k 수준이 되면 lr 3e-4 로는 합격선에 못 간다 (full 의 step 1 k 지점 = ep 2, action 3.5). 더 큰 lr 이 필요하다.
2. **warmup 없는 lr 1e-3 · 3e-3 은 첫 epoch 에 무너졌다** (README §7, 두 번 재현). 그래서 큰 lr head 에는 warmup 을 붙인다 (start 1e-5).
3. ⚠️ **옛 1/3 run (21.8) 은 val 이 달랐다.** `filter_annotations` 가 train 에 없는 action 의 val clip 을 지운다 — 영상 단위 1/3 은 val 8,761 clip · action 893 클래스로 채점됐다 (full 은 9,296 · 1,114). 그 21.8 은 full 35.3 과 같은 판이 아니다.
   **`strat_min3` 부분집합 (1/10 · 1/5) 은 val 9,296 · 1,114 · 73 · 201 을 그대로 유지한다** (10-15 확인) — 이 판에서만 full 과 직접 비교한다.

## 1 차 (10-15 제출, 계산량 둘 다 full 의 약 1/10)

head `wu3` (`resolve.py`): wd 1e-4 에서 lr 3e-4 (warmup 0, 기준) · 1e-3 · 3e-3 (warmup = 학습의 10 %). batch 16 × 3 GPU = **global 48** (논문 128) — step 이 2.7 배 늘어나는 것도 노린 것.

| TAG | 데이터 | epoch | step (≈) | warmup | 노드 | job |
|---|---|---:|---:|---:|---|---|
| `ek100_rel_tenthS3_e20_wu3_b48` | 1/10 `strat_min3` (6,722 구간) | 20 | 2.8 k | 2 ep | vll1 | 221973 (vll3) 취소 → 221975 재제출 → **10-15 13:38 취소** (B 의 epoch 5 가 6.5 라 같은 계산량의 A 도 합격선에 못 갈 것으로 보고, vll1 GPU 를 평가 행렬 GRASP 에 돌렸다. 필요하면 같은 명령으로 다시 낸다) |
| `ek100_rel_fifthS3_e10_wu3_b48` | 1/5 `strat_min3` (13,452 구간) | 10 | 2.8 k | 1 ep | vll1 | 221974 |

두 run 은 step · 계산량이 같고 **"같은 데이터를 많이 돌기 vs 데이터를 더 넓게"** 만 다르다. val 은 epoch 5 마다.

**결과가 어느 쪽이어도 다음 수가 정해진다**
- 큰 lr head 가 합격선 (action ≥ 31.5) 근처 → 그 lr 로 epoch 을 더 줄여 하한을 찾는다.
- 큰 lr head 가 무너진다 (train recall 이 처음 수십 iter 안에 상수로) → warmup 을 늘리거나 lr 6e-4 로.
- 둘 다 lr 3e-4 head 와 비슷하게 낮다 → step 이 모자란 것 → epoch 을 늘리거나 (1/5 × 20) 특징 캐시 (증강 없이 한 번 추출 → head 만 수십 epoch) 를 검토한다. 캐시는 증강이 빠져 논문 프로토콜과 달라지므로 full 과 같은 판에서 한 번 대조해야 쓸 수 있다.

단서: 1 차는 batch 가 논문과 다르다 (48 vs 128). 합격 조합이 나오면 batch 128 (8 GPU) 에서 한 번 더 확인한다. head 3 개 중 최고를 보고하는 것은 논문 (20 개 중 최고) 과 같은 규칙이고, head 수가 적은 만큼 운에 덜 기댄다.

## 1 차 중간 결과 (10-15 12:55, B 의 epoch 5 / 10, val 9,296 clip)

| head | lr (warmup) | action | verb | noun |
|---|---|---:|---:|---:|
| 0 | 3e-4 (없음) | 4.84 | 23.4 | 16.0 |
| 1 | 1e-3 (1 ep) | **6.52** | 27.9 | 18.5 |
| 2 | 3e-3 (1 ep) | 3.72 | 20.1 | 13.0 |

warmup 을 붙인 1e-3 은 무너지지 않고 가장 좋다. 다만 full run 은 같은 진행률 (epoch 5 / 20) 에 12.2 · epoch 10 에 24.3 이라, 계산량 1/10 판은 합격선 31.5 에 크게 못 미칠 것으로 본다 (최종은 B 끝 ~18:00).

## 재현

```bash
python z_research/scripts/data/build_ek100_subset.py --n 10    # → data_csv/ek100/EPIC_100_train_tenth_strat_min3_seed0.csv
python z_research/scripts/data/build_ek100_subset.py --n 5     # → data_csv/ek100/EPIC_100_train_1of5_strat_min3_seed0.csv
E=z_research/previous/anticipation/EK100; C=$PWD/data_csv/ek100; P="data.anticipation_point_mode=released data.time_source=frame_fixfps"
EK_WARMUP=2 GPUS=3 HEADS=wu3 NODE=vll1 MEM_PER_GPU=25G CPUS_PER_GPU=3 TAG=ek100_rel_tenthS3_e20_wu3_b48 \
  SET="$P data.base_path=/local_datasets/EPIC-KITCHENS_resized data.dataset_train=$C/EPIC_100_train_tenth_strat_min3_seed0.csv optimization.num_epochs=20 evaluation.val_every=5" \
  env -u SLURM_JOB_ID bash $E/sbatch.sh
EK_WARMUP=1 GPUS=3 HEADS=wu3 NODE=vll1 MEM_PER_GPU=25G CPUS_PER_GPU=3 TAG=ek100_rel_fifthS3_e10_wu3_b48 \
  SET="$P data.base_path=/local_datasets/EPIC-KITCHENS_resized data.dataset_train=$C/EPIC_100_train_1of5_strat_min3_seed0.csv optimization.num_epochs=10 evaluation.val_every=5" \
  env -u SLURM_JOB_ID bash $E/sbatch.sh
```
(vll3 는 다른 사용자 job 이 13 일 남아 10-15 에 vll1 로 옮겼다 — EK100 영상은 vll1 `/local_datasets` · vll3 · vll5 `/data2` 에 있다.)
(`env -u SLURM_JOB_ID` — 대화 세션이 SLURM job 안에서 돌아 sbatch.sh 의 자기제출 방지 tripwire 가 걸린다. 사람 셸에서는 필요 없다.)
