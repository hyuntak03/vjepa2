# 평가 행렬 — 모델 4 × 벤치마크 6 (2026-10-15 사용자 지정)

> 사용자 지정 (2026-10-15): **모델 = release · release + PFT · vast AR · vast AR + PFT**, **벤치마크 = IntPhys 1 · IntPhys 2 · VOBench · VOBench gravity · GRASP · InfLevel**. 이 24 칸을 채운다.
> **PFT (10-15 사용자 재정의)** = VOBench split **v2** train + **gravity train** 의 가능 clip (720 + 108 = 828, `vobench_split2g_train`) post-FT 30 epoch.
> 모델 이름 `vith_release_vobftg` · `vith_vast_ar_state_ep70_vobftg` (run `z_training/runs/vobench_{release,vast_ar_state}_postft_g`). gravity 를 뺀 옛 PFT (`*_vobft`) 는 아래 "기록" 으로 내렸다.
> 그래서 VOBench 는 **split v2 test** 블록만, gravity 는 **gravity test 108 block** (`data_csv/vobench_v1_gravity_split/index_test.csv`) 만 읽고, 네 모델 모두 같은 test 로 비교한다.
> 칸 = 각 벤치마크의 공식 · 우리 프로토콜 지표. 갱신할 때 날짜를 칸에 붙인다.

| 모델 (레지스트리) | IntPhys 1 dev (A.8 · Filtered macro, 180 쌍) | IntPhys 2 Main (D.3 격자, 506 쌍) | VOBench **late** p / r (split v2 test, 문맥 끝 가림 216 쌍 = p 108 · r 108) | VOBench gravity **고른 미래 = 진짜** p / r (gravity test, 54 block 씩: ledge 2 지선다 chance 50 · arc 9 지선다 chance 11.1) | GRASP level2 | InfLevel (property 별 최고 C) |
|---|---|---|---|---|---|---|
| **release** (`vith`) | **88.89** (skip2_w32) | **54.3** (w48 C24) | **58.3 / 65.7** (late 62.0 · test 전체 75.7) | **0.0 / 0.0** (0 / 54 · 0 / 54) · 참고 쌍 25.0 / 13.9 | **70.4** (skip10_w32; w16 최고 69.4) — 공식 규칙 (시나리오 16 개마다 최고 C → macro). ⚠️ 10-17 정정: 10-16 에 쓴 56.25 는 summary.json 의 best_avg (C 고르지 않은 쌍 micro) 였다 | continuity **72.8** · gravity 51.9 · solidity 50.0 (macro 58.3, skip10_w16) — ⚠️ 10-15 정정: 옛 칸 52.7 / 59.2 는 공식 규칙 (`garrido_rescore.collect_inflevel`) 과 다르다 |
| **release + PFT** (`vith_release_vobftg`) | **83.33** (skip2_w16, 10-15; O1 73.3 · O2 85.0 · O3 91.7) | **53.6** (w48 C36, 10-15; w16 53.4 · w32 53.2) | **88.0 / 91.7** (late 89.8 · test 전체 91.9, 10-15) | **100.0 / 92.6** (54 / 54 · 50 / 54, 10-15; ledge 36 / 36 · 36 / 36, arc 18 / 18 · 14 / 18) · 참고 쌍 100 / 97.2 · train block 포함 216 → 98.1 | **68.8** (skip5_w16; w32 최고 64.5) — 공식 규칙. ⚠️ 정정: 옛 59.42 는 micro | continuity **72.3** · gravity 51.3 · solidity 51.3 (macro 58.3, skip10_w16, 10-15) |
| **vast AR** (`vith_vast_ar_state_ep70`) | **90.0** (skip2_w16) | **51.8** (w48 C42, 10-15; w16 51.0 · w32 51.6 — 창별 최고 중 최고) | **75.9 / 75.9** (late 75.9 · test 전체 82.8) | **18.5 / 11.1** (10 / 54 · 6 / 54, 10-15; ledge 3 / 36 · 1 / 36, arc 7 / 18 · 5 / 18) · 참고 쌍 71.1 / 65.6 | — (사용자 지시로 취소, 10-15) | continuity **75.2** · gravity 50.1 · solidity 50.0 (macro 58.4, skip10_w16, 10-15) |
| **vast AR + PFT** (`vith_vast_ar_state_ep70_vobftg`) | **90.0** (skip2_w16, 10-15; O1 81.7 · O2 96.7 · O3 91.7) | **50.4** (w48 C42, 10-15; w16 49.2 · w32 50.2) | **89.8 / 89.8** (late 89.8 · test 전체 90.7, 10-15) | **57.4 / 59.3** (31 / 54 · 32 / 54, 10-15; ledge 15 / 36 · 14 / 36, arc 16 / 18 · 18 / 18) · 참고 쌍 87.2 / 87.8 · 216 → 58.8 | 68.6 (skip10_w32, **w32 만** — w16 취소, 10-16) — 공식 규칙. ⚠️ 정정: 옛 56.84 는 micro | continuity **73.9** · gravity 50.7 · solidity 49.3 (macro 58.0, skip10_w16, 10-15) |

⏳ = 학습 · 실행 중 또는 큐 (10-15 02:33 학습 시작, vll5 GPU 4 장)

**칸 기록 규칙 (사용자 10-15)**: VOBench 는 **late 만** p (primitive) / r (realistic) 로 적는다 (괄호는 참고 — late 합계 · test 전체).
gravity 는 **고른 미래 = 진짜** 를 p / r 로 적는다. 다시 세는 법: VOBench 는 각 run 의 `per_block.json` 에서 `data_csv/vobench_v1_split_v2/index_test.csv` 의 block 중
`occlusion == late` 만 골라 `acc × n_pairs` 를 objset 별로 합산 (matched, 동점 0.5) — 10-15 산출물에서 다시 셌고 기존 late 합계 (62.0 · 89.4 · 75.9 · 90.3) 와 일치.
gravity 는 `python z_research/scripts/analysis/vobench_gravity_pick.py <run 폴더> --test` (`per_block.json` 의 `per_video_surprise` 에서, 216 block 을 다 채점하고 test 108 만 센다).
VOBench test 전체 (참고): release p 70.3 · r 81.1 / vast AR 81.1 · 84.4. release 전체 1,440 쌍 71.7 / 81.9, vast AR 83.4.

**gravity split** (`z_training/data/build_vobench_gravity_split_index.py`): (objset, law, 배경, 물체) 묶음 단위로 반반 — arc 의 apex66 · 48 · 30 은 같은 물체 · 배경이면
같은 포물선을 시간만 밀어 자른 것이라 한 묶음 (그래야 test 미래가 train 에 안 나온다). 셀 (objset × law × cell) 18 개 모두 train 6 · test 6 block. train 은 가능 'pos' 108 clip.
단서: 같은 셀의 궤적 (같은 z · v 또는 apex) 은 다른 물체 · 배경으로 train 에 나온다 — VOBench split v2 가 같은 운동을 다른 물체로 양쪽에 두는 것과 같은 정책이다. PFT 의 gravity 칸은 **분포 안 held-out** 이다 (분포 밖이 아니다).

⚠️ **gravity 칸 정정 (10-15)** — 처음 적은 8.9 는 `summary.json` 의 `block_pairwise` (block 마다 쌍 정확도를 낸 뒤 216 block 평균) 였다.
gravity 는 block 마다 **진짜 미래가 하나**뿐이므로 행렬 지표를 "가장 덜 놀라운 미래가 진짜인가" (argmin) 로 바꾼다. 쌍 정확도 (진짜 vs 불가능 하나씩) 는
arc 에서 진짜가 일부 불가능 후보 (멈춤 등) 를 이기는 몫까지 세서 정답 하나 문제를 부분 점수로 만든다 — 참고로만 둔다.
release 216 block 전부: 고른 미래 0 / 216, 쌍 정확도 ledge 0 / 144 · arc primitive 31.9 · realistic 21.5 (apex30 42.7 / 31.2 → apex66 20.8 / 14.6).

### 기록 — 옛 PFT (VOBench split v2 train 만, gravity 없음, 10-14 학습 · `*_vobft`)

| 모델 | IntPhys 1 | IntPhys 2 | VOBench late p / r | gravity test p / r (고른 미래) | 비고 |
|---|---|---|---|---|---|
| release + PFT (옛, `vith_release_vobft`) | 81.67 (skip2_w32, e30) | 10-15 실행 중 중단 (PFT 재정의) | 87.0 / 91.7 (late 89.4 · test 91.8) | 1.9 / 3.7 (1 / 54 · 2 / 54) · 쌍 57.8 / 51.7 | 216 block 전부 6 / 216 · 쌍 56.8 |
| vast AR + PFT (옛, `vith_vast_ar_state_ep70_vobft`) | 90.0 (skip2_w32) | 10-15 실행 중 중단 · 이전 부분 w48 C24 49.8 | 90.7 / 89.8 (late 90.3 · test 91.0) | 9.3 / 13.0 (5 / 54 · 7 / 54) · 쌍 65.6 / 67.2 | |

## 읽기 (지금까지)

- **gravity 가능 clip 을 PFT 에 넣으면 release 는 gravity test 를 거의 다 푼다** — 고른 미래 0 → **96.3** (ledge 72 / 72, arc 32 / 36). train block 포함 216 에서 98.1 이라 test 와 차이가 작다.
  단 test 는 같은 셀 궤적을 다른 물체 · 배경으로 본 **분포 안 held-out** 이다 — "낙하를 배웠다" 가 아니라 "이 18 셀의 궤적을 물체 · 배경을 바꿔도 낸다" 까지가 이 칸이 세우는 문장이다.
- **vast AR + PFT 는 arc 는 풀고 (34 / 36) ledge 는 chance 아래 (29 / 72 = 40 %)** — 같은 데이터 · 과제로 post-FT 했는데 release 와 갈린다. 출발점 (AR rollout, 8 블록) 이 낙하 미래를 덜 받아들이는지 · 학습 과제 차이 (문맥 8 블록 → rollout) 인지는 아직 못 가른다.
- 10-15 03:38 첫 시도는 두 평가 줄이 같은 DDP 포트를 잡아 멈췄다 (vast AR 쪽은 split-brain 가드가 `world_size=1` 로 잡아 죽였다). 04:04 포트를 줄마다 따로 주고 다시 돌린 결과다 (`stdout.log` 15870 행 이후, world_size=2).

- (옛 PFT 기준, 기록) PFT 는 VOBench (같은 분포 test) 를 크게 올렸다 — release 75.7 → 91.8, vast AR 82.8 → 91.0, late 62 → 89 · 76 → 90. IntPhys 1 은 release 88.9 → 81.7 (−7.2), vast AR 90.0 → 90.0.
- **gravity 는 쌍 정확도와 고른 미래가 갈린다** — release → 옛 release+PFT 에서 쌍 정확도는 21.4 → 56.8 로 오르지만 진짜 미래를 고르는 block 은 0 → 6 / 216. ledge 는 둘 다 0.
- **vast AR 은 PFT 없이도 gravity 에서 release 보다 낫다** (test 고른 미래 14.8 vs 0; arc 33 % vs 0, ledge 4 / 72 vs 0). 다만 arc chance 11.1 · ledge chance 50 이라 ledge 는 여전히 chance 훨씬 아래.

## 남은 칸과 실행 계획 (10-15 13:40 — GRASP 만 남음)

GRASP 는 mp4 디코딩 + 창 격자라 창 하나가 GPU 약 9 시간 (vll4 1 GPU 0.13 영상/s · vll1 3 GPU 6 시간 · vll5 2 GPU 5.5 시간). 창 (w16 · w32) 마다 job 을 나눠 세 노드에 흩었다.

| 창 | 어디 | 상태 |
|---|---|---|
| release w16 | vll1 | 끝 (12:30) |
| release w32 | vll4 job 222243 | 진행 (vll1 에서 9 시간 남아 취소 · FORCE 로 다시) |
| vast AR w16 · w32 | vll4 job 222244 · 222245 | 진행 |
| release + PFT w16 | vll5 줄 L1 | 진행 → 끝나면 w32 는 아래 job 이 하고 있어 건너뛴다 |
| release + PFT w32 | vll4 job 222246 | 진행 |
| vast AR + PFT w16 | vll5 줄 L2 | 진행 |
| vast AR + PFT w32 | vll1 job 222250 | 진행 |

같은 창을 두 곳에서 돌리지 않게 `cvpr/harness/launch.sh` 에 가드를 넣었다 — summary 없이 `stdout.log` 가 30 분 안에 갱신된 폴더는 건너뛴다 (`RUNNING_MIN`, `FORCE=1` 로 무시).
진입점: vll5 = `eval_matrix_chain.sh` (L1 · L2), 다른 노드 = `eval_matrix_node.sh` (`WINDOWS=` 로 창 하나, `EVAL_DDP_PORT=` 를 job 마다 따로).

단서: InfLevel gravity · solidity 는 공식 로더가 판정 구간을 잘라 원리적으로 풀 수 없다 (Benchmarks README) — continuity 가 믿을 수 있는 칸이다. AR 계열은 타깃 공간이 달라 쌍 정확도만 비교한다.

## 출처

- IntPhys 1: `cvpr/results/eval/intphys1/intphys1_dev_vith_vast_ar_state_ep70*` · `z_training/runs/vobench_release_postft/eval/intphys1_sliding__intphys1_dev_e30` · release 는 Benchmarks 기준값 88.89 (`garrido_rescore` 규칙으로 다시 셈)
- IntPhys 2: release `z_research/previous/Benchmarks/exp_results/intphys2/bench_intphys2_main_vith` · 나머지 `cvpr/results/eval/intphys2/`
- VOBench: 전체 run 은 `z_research/VOBench_v1/exp_results/`, PFT 는 `z_training/runs/vobench_*_postft/eval/surprise_c16t32__vobench_v1_split2_test_latest` — split v2 test 블록 (`data_csv/vobench_v1_split_v2/index_test.csv`) 으로 네 모델을 같은 쌍에서 셈
- gravity: `z_research/VOBench_v1/exp_results/surprise_c16t32__vobench_v1_gravity_<모델>` (216 block 전부 채점, test 108 block 만 셈)
- InfLevel: `z_research/previous/Benchmarks/exp_results/intphys1_sliding__inflevel_*_vith_w{16,32}` (`garrido_rescore.collect`)
