# 복사가 못 푸는 부분집합 (copy-hard) 에서의 predictor 정확도 — 2026-09-29

> 사용자 (09-29): "late 가림은 복사로 안 나온다 — 이게 오히려 좋은 거지." 그 칸을 벤치마크마다 격리해서, **복사가 틀리는 쌍에서 predictor 는 몇 점인가** 를 쌍 단위로 다시 셌다.
> 산출물 `auto_research/exp_results/verify/copy_hard_subset.json`, 스크립트 `auto_research/scripts/copy_hard_subset.py` (CPU, GPU 없음). 1 차 · 적대 검증 전.
> 근거 문서: [`PREDICTOR_SHARE_2026-09-29.md`](PREDICTOR_SHARE_2026-09-29.md) (전체 Δ), `z_research/TrainingEffects/scores/ANALYSIS.md` (v11 칸별 copy).

## 0. 결론 먼저

- **복사가 틀리는 쌍에서 릴리즈 predictor 는 우연 수준이다.** v11 48.4 [46.6, 50.2] (n 2,877) · IntPhys 1 skip2_w32 44.4 [25.9, 63.4] (n 27) · IntPhys 2 C=24 21.2 [16.0, 26.6] (n 240, 우연 아래). 복사가 맞는 쌍에서는 85.7 / 96.7 / 84.2. → 릴리즈의 점수는 **복사 + 나머지에서 동전 던지기** 로 분해된다 (복사와의 쌍별 일치율 v11 .77 · IntPhys 1 .91 · IntPhys 2 .82).
- **미래 전용 · prefix (Ariel full / prefix, post-FT ip2) 도 같다** — v11 copy-hard 49.6 ~ 51.0, IntPhys 2 19 ~ 20. 학습이 늘린 전체 점수 (+3 ~ +4) 는 전부 복사가 이미 맞는 쌍 위에서 난다 (copy-correct 89.9 ~ 90.0).
- **예외 둘.** (i) 도메인 안 post-FT `v11_e10` 82.5 · `pv1_e15` 61.5 (v11), `ip1_e40` 66.7 ~ 76.6 (IntPhys 1) — 학습 분포 안에서만. (ii) **AR (ep18 · ep38) 은 셋 다 copy-hard 에서 높다** — v11 73.8, IntPhys 1 70 ~ 81, IntPhys 2 40 ~ 42. 단 AR 은 표적 공간이 블록별이라 표준 복사와 덜 상관될 뿐일 수 있다 (§4 단서) — IntPhys 2 에서는 copy-correct 가 57 로 같이 떨어져 **복사와 독립인 잡음** 에 가깝고 (일치율 .58), v11 · IntPhys 1 에서는 copy-correct 도 유지돼 (87 · 93) 실제 이득이다.
- **v11 의 late 가림 (사용자가 짚은 칸)**: 복사 9.8 ~ 67.9, 릴리즈 0.0 ~ 65.2, prefix 14 ~ 49, **AR ep38 72 ~ 99**, v11_e10 87 ~ 100. 이 칸이 새 predictor 의 1 차 보고 칸이다.
- 복사 천장 칸 (copy ≥ 95, v11 20 칸 · 2,240 쌍) 에서는 모든 predictor 가 98 ~ 100 — **거기서는 predictor 차이가 사라진다.**

> ⚠️ **적대 검증 3 차 정정 ([`VERIFY_R3_2026-09-29.md`](VERIFY_R3_2026-09-29.md) C3)** — 수치는 소수점까지 재현 (release 48.38, prefix 51.03, AR ep38 73.83; late·vanish→빈 99.11 / 85.71 / 72.32 vs 65.18 / 2.68 / 0.00). 그러나 **읽기 문장을 바꾼다**: 복사와 오류가 무작위로 겹치는 귀무 모델에서 copy-hard 정확도 = 전체 정확도다. 모든 predictor 가 그 아래에 있다 (release −27, prefix −28, full −28, **AR −10**, v11_e10 −9). 즉 AR 의 copy-hard 73.8 은 "복사가 못 푸는 것을 푼다" 가 아니라 **"AR 의 오류가 복사의 오류와 훨씬 덜 묶여 있다"** 다. 숫자는 두고 문장을 이렇게 쓴다. copy_blk 부분 표본 (n 13–45) 은 시사 이상 아님; cb-hard 45 쌍은 전부 late.

## 1. 정의

- 쌍 결과 o_m = 1[S_m(imp) > S_m(pos)], 동점 0.5 (IntPhys 1 은 garrido 규칙 strict: 동점 = 0).
- **copy-hard (쌍 단위)** = {복사가 틀리거나 동점인 쌍} (o_copy < 1). **copy-correct** = 그 여집합. 칸 단위: copy-hard 칸 = 복사 정확도 ≤ 50, 천장 칸 = ≥ 95.
- 복사 = 문맥 마지막 튜블릿 LN(z) 를 미래 전 슬롯에 복사 (H6 관례; IntPhys 2 는 `IP2_COPY=1`). AR 공간 복사 = `copy_blk` (블록별 LN(target_encoder) 복사) — IntPhys 1 만 완전, v11 은 360 영상 부분.
- CI = cluster bootstrap B 2,000 (v11 block · IntPhys 1 scene · IntPhys 2 scene_index), 같은 부분집합의 방법끼리 같은 resample.
- **재계산 / 전사**: 아래 수치는 전부 원배열 · per_video · per_window 에서 다시 계산했다 (v11 배열 = `training_effects/v11score_vith_curves`, Ariel 0927 팔과 AR 은 하네스 `per_block.json` 의 per_video_surprise; IntPhys 1 = `ip1score_final` · `ip1score_ar_copyblk` 배열 + Ariel 0927 팔은 하네스 `per_window.json` → G; IntPhys 2 = `per_video.csv`). 전사한 값 없음. 검산: v11 release 75.7 / copy 73.2, IntPhys 1 skip2_w32 release 88.9 / copy 85.0, AR ep18 skip2_w16 90.6 / copy_blk 83.3, IntPhys 2 C24 release 54.4 / copy 52.6 — 기존 문서와 일치.

## 2. IntPhysGen v11 (`v11_split_test`, 10,752 쌍)

### 2-1. 쌍 단위

| | 전체 (10,752) | **copy-hard (2,877)** | copy-correct (7,875) | 복사와 일치율 |
|---|---|---|---|---|
| copy | 73.2 | 0 | 100 | 1 |
| **release** | 75.7 [75.0, 76.4] | **48.4 [46.6, 50.2]** | 85.7 [84.9, 86.4] | .766 |
| Ariel prefix ep45 | 79.5 | 51.0 [49.2, 52.9] | 89.9 | .790 |
| Ariel full ep40 | 78.1 | 49.6 [47.8, 51.4] | 88.5 | .783 |
| Ariel prefix ep43 (배열) | 79.3 | 50.3 [48.5, 52.1] | 89.9 | .791 |
| post-FT ip2_e80 | 79.5 | 51.0 [49.2, 52.7] | 90.0 | .790 |
| post-FT ip1_e40 | 78.8 | 59.5 [57.8, 61.4] | 85.8 | .736 |
| post-FT pv1_e15 | 81.4 | 61.5 [59.8, 63.1] | 88.7 | .753 |
| post-FT v11_e10 (도메인 안) | 91.2 | 82.5 [81.2, 83.9] | 94.3 | .738 |
| **AR ep38** | 83.7 | **73.8 [72.2, 75.4]** | 87.3 | .709 |
| AR ep18 | 83.2 | 74.0 [72.5, 75.5] | 86.5 | .703 |

- 릴리즈 · prefix · full · ip2 post-FT 는 복사가 틀리는 2,877 쌍에서 **48 ~ 51 = 우연**. 전체 점수 차 (75.7 → 79.5) 는 copy-correct 위 (85.7 → 89.9) 에서만 난다.
- copy-hard 2,877 쌍의 구성: color 1,571 · shape 1,023 · vanish→빈 207 · vanish→물체 76; 타이밍 late 1,272 · mid 626 · early 559 · visible 420. 즉 **복사가 못 푸는 건 대부분 색 · 모양 위반이고, 사라짐은 late 에만 남는다.**

### 2-2. copy-hard 쌍을 위반 · 운동 · 타이밍으로 나눠서

| copy-hard 안 | n | release | prefix ep45 | AR ep38 | v11_e10 |
|---|---|---|---|---|---|
| vanish→빈 | 207 | **9.7** [5.8, 14.0] | 18.8 | **79.7** [73.9, 85.0] | 95.2 |
| vanish→물체 | 76 | 64.5 | 81.6 | 94.7 | 100 |
| shape | 1,023 | 43.0 | 46.6 | 66.5 | 84.2 |
| color | 1,571 | 56.2 | 56.6 | 76.8 | 79.0 |
| static | 1,136 | 50.3 | 48.1 | **89.5** | 87.8 |
| moving_flat | 775 | 50.8 | 54.7 | 65.9 | 78.5 |
| moving (ramp) | 966 | 44.2 | 51.5 | 61.7 | 79.6 |
| visible | 420 | 69.0 | 63.1 | 86.7 | 87.9 |
| early | 559 | 52.8 | 61.5 | 77.8 | 81.2 |
| mid | 626 | 51.1 | 60.4 | 78.0 | 80.7 |
| **late** | 1,272 | **38.3** [35.8, 40.9] | **37.8** | **65.8** | 82.3 |

- late 는 릴리즈 · prefix 가 **우연 아래** (38) — 복사가 틀리는 late 쌍은 predictor 도 복사 쪽으로 틀린다. AR 만 65.8.
- AR 의 이득은 **static (89.5) 과 vanish→빈 (79.7)** 에 몰린다 — 이동 물체 (moving 61.7) 에서는 작다. I-scene/AR 판독 ("정지 내용은 AR 최고, 움직이는 내용은 최저") 과 같은 방향.

### 2-3. 칸 단위 — copy-hard 칸 5 개 (복사 ≤ 50) 과 천장 칸 20 개 (복사 ≥ 95)

copy-hard 칸: `late|static|vanish→물체` · `late|static|shape` · `late|static|color` · `late|moving_flat|vanish→빈` · `late|moving|vanish→빈` (합 1,008 쌍). 천장 칸: visible/early/mid 의 vanish 전부 + visible 의 shape 일부 등 20 칸 (2,240 쌍).

| | copy-hard 칸 합 (1,008) | 천장 칸 합 (2,240) |
|---|---|---|
| copy | 32.2 | 99.9 |
| release | 43.6 [40.9, 46.4] | 98.3 |
| prefix ep45 / full ep40 | 45.8 / 43.1 | 100 / 100 |
| ip2_e80 / ip1_e40 / pv1_e15 | 49.4 / 55.8 / 62.7 | 100 / 91.4 / 98.8 |
| **AR ep38** | **79.6** [77.3, 81.7] | 100 |
| v11_e10 | 89.9 | 99.8 |

**late · vanish→빈 (사용자가 짚은 칸, 각 112 쌍):**

| 칸 | copy | release | pv1_e15 | prefix ep45 | full ep40 | **AR ep38** | v11_e10 |
|---|---|---|---|---|---|---|---|
| late · static | 67.9 | 65.2 | 52.7 | 49.1 | 50.9 | **99.1** | 86.6 |
| late · moving_flat | 39.3 | **2.7** | 88.4 | 37.5 | 15.2 | **85.7** | 100 |
| late · moving (ramp) | 9.8 | **0.0** | 77.7 | 14.3 | 8.9 | **72.3** | 100 |

- 릴리즈는 움직이는 물체가 가려진 채 사라지는 쌍을 **0 ~ 3 %** 로 맞힌다 — 복사 (10 ~ 39) 보다도 낮다 (§5-1 의 "vanish 50 = 100 과 0 의 평균" 의 그 0). Ariel 미래 전용 팔도 9 ~ 38. **rollout 학습 AR 만 72 ~ 86** (static 99).
- ⚠️ AR 의 late·static 99.1 은 "정지 물체가 가려진 채 사라짐" — AR 은 정지 내용을 가장 잘 놓는 predictor 이고 (AR_PREDICTOR §2-4 검증), 블록별 인코딩의 마지막 블록에는 물체가 없으므로 이 칸이 AR 공간 복사로도 맞을 수 있다 (§4). 이동 칸 72 ~ 86 은 그 설명으로 안 된다.

### 2-4. AR 공간 복사 (`copy_blk`) — 부분 (360 영상 · 104 쌍)

| 104 쌍 | copy_blk | AR ep18 | copy (표준) | release |
|---|---|---|---|---|
| 전체 | 56.7 [49.5, 64.0] | 79.8 | 77.9 | 63.5 |
| copy_blk-hard (45) | 0 | **82.2** [71.1, 91.1] | 71.1 | 40.0 |
| copy_std-hard (23) | 43.5 | 73.9 | 0 | — |

- 표준 복사와 블록 복사의 쌍별 일치율 .596 — **두 공간의 복사는 다른 쌍을 맞힌다.** AR 은 자기 공간 복사가 틀리는 45 쌍에서도 82 → AR 의 copy-hard 이득이 "자기 공간 복사" 로 환원되지는 않는다 (n 45, 부분 추출이라 시사).
- v11 전체 copy_blk 는 사용자 지시로 미실행 (`ARIEL_CHECKPOINTS.md` §5-2b). 필요하면 `te_v11_score.py --preds ariel_ar_ep38` 에 copy_blk 이 같이 나온다 (GPU).

## 3. IntPhys 1 dev (180 쌍, Garrido 격자 · Filtered · strict)

| skip2_w32 | 전체 | **copy-hard (27)** | copy-correct (153) | copy_blk-hard (35, AR 공간) | 일치율 (copy / copy_blk) |
|---|---|---|---|---|---|
| copy / copy_blk | 85.0 / 80.6 | 0 / 55.6 | 100 / 85.0 | 65.7 / 0 | 1 / .79 |
| **release** | 88.9 | **44.4 [25.9, 63.4]** | 96.7 | 77.1 | **.906** / .78 |
| ip1_e40 (도메인 안) | 93.9 | 66.7 | 98.7 | 80.0 | .89 / .82 |
| prefix ep45 / full ep40 | 70.0 / 71.7 | 33.3 / 37.0 | 76.5 / 77.8 | 40.0 / 37.1 | .75 / .74 |
| Ariel ep43 | 68.9 | 25.9 | 76.5 | 40.0 | .76 / .73 |
| **AR ep38 / ep18** | 90.0 / 87.8 | **70.4 / 74.1** | 93.5 / 90.2 | 60.0 / 57.1 | .84 / **.86** |

| skip2_w16 | 전체 | copy-hard (47) | copy-correct (133) | copy_blk-hard (30) |
|---|---|---|---|---|
| copy / copy_blk | 73.9 / 83.3 | 0 / 63.8 | 100 / 90.2 | 43.3 / 0 |
| release | 83.3 | 51.1 [36.7, 64.5] | 94.7 | 53.3 |
| AR ep38 / ep18 | 92.8 / 90.6 | 80.8 / 80.8 | 97.0 / 94.0 | 60.0 / 63.3 |
| prefix ep45 | 63.9 | 42.5 | 71.4 | 56.7 |

| skip5_w16 (긴 간격) | 전체 | copy-hard (66) | copy-correct (114) | copy_blk-hard (49) |
|---|---|---|---|---|
| copy / copy_blk | 63.3 / 72.8 | 0 / 62.1 | 100 / 79.0 | 49.0 / 0 |
| release | 59.4 | **16.7 [8.1, 25.8]** | 84.2 | 34.7 |
| AR ep38 / ep18 | 77.8 / 77.2 | **68.2 / 69.7** | 83.3 / 81.6 | 44.9 / 46.9 |
| prefix ep45 | 53.3 | 16.7 | 74.6 | 32.6 |

- 릴리즈는 복사와 **쌍별 91 % 일치** (skip2_w32). 복사가 틀리는 27 쌍에서 44 (우연), 긴 간격에서는 17 (우연 아래). copy-hard 는 대부분 **moving/occluded** (skip2_w32: O1 9 · O3 5 · O2 4 / 18 of 27) 와 **disappear 방향** (release 18 %).
- AR 은 표준 copy-hard 에서 70 ~ 81 이지만 **자기 공간 (copy_blk) 이 틀리는 쌍에서는 57 ~ 63** — 릴리즈가 copy_blk-hard 에서 77 인 것과 대칭. 즉 IntPhys 1 에서 "AR 이 copy-hard 를 푼다" 의 절반은 **표적 공간이 달라 다른 쌍을 틀리는 것** 이다. 180 쌍이라 CI 가 넓다 (±15 ~ 20 pt).
- 칸 단위 copy-hard 칸은 skip2_w32 에 없고 (모든 principle × 운동/가림 칸이 복사 > 50), skip5_w16 에서 O2 · O3 moving 4 칸.

## 4. IntPhys 2 Main (506 쌍, w48 · 같은 C = 24)

| w48 · C24 | 전체 | **copy-hard (240)** | copy-correct (266) | 일치율 |
|---|---|---|---|---|
| copy | 52.6 | 0 | 100 | 1 |
| **release** | 54.4 [50.4, 57.9] | **21.2 [16.0, 26.6]** | 84.2 | **.816** |
| prefix ep45 / full ep40 | 52.6 / 52.8 | 19.2 / 20.0 | 82.7 / 82.3 | .82 / .81 |
| AR ep38 / ep18 | 49.2 / 49.6 | 40.4 / 41.7 | 57.1 / 56.8 | **.58** / .58 |

- 릴리즈 · prefix · full 은 복사와 82 % 일치하고 복사가 틀리는 240 쌍에서 **20 % (우연 아래)** — 복사가 틀리면 같이 틀린다. permanence 만 보면 copy-hard 61 쌍에서 release 19.7, prefix 11.5, full 9.8.
- AR 은 일치율 .58 로 복사와 거의 독립이고 copy-hard 40 / copy-correct 57 — 전체 49 (우연) 를 두 부분집합에 고루 뿌린 것. **IntPhys 2 에서는 어느 predictor 도 복사 위에 없다** (PREDICTOR_SHARE 와 같음).
- w32 · C24 도 같다 (release copy-hard 22.4, copy-correct 80.4). copy-hard 칸 (복사 ≤ 50): immutability · permanence · solidity 의 Moving.

## 5. 읽기

복사가 못 푸는 쌍은 벤치마크마다 있고 (v11 27 % · IntPhys 1 15 % · IntPhys 2 47 %), **거기서 지금 predictor 들은 우연이거나 우연 아래다.** 릴리즈뿐 아니라 미래 전용으로 다시 학습한 prefix · full 도 같다 — 학습이 올린 전체 점수는 복사가 이미 맞는 쌍 위에서만 났다. 이 부분집합이 "predictor 가 실제로 예측해야 하는 칸" 이고, 새 predictor 의 1 차 downstream 지표는 전체가 아니라 **copy-hard 부분집합 정확도** 여야 한다 (전체 점수는 복사가 천장인 칸에 희석된다: v11 천장 20 칸에서 모든 predictor 98 ~ 100). 되먹임 학습 AR 은 v11 · IntPhys 1 의 copy-hard 에서 유일하게 높지만 (74 · 70 ~ 81), (i) 표적 공간이 달라 다른 쌍을 틀리는 몫이 있고 (IntPhys 1 copy_blk-hard 57 ~ 63), (ii) IntPhys 2 에서는 복사와 독립인 우연이며, (iii) 이득이 정지 내용과 사라짐에 몰린다 — reach 판독 (움직이는 내용 최단) 과 모순되지 않는다.

## 6. 단서

- **쌍 단위 vs 칸 단위.** copy-hard 쌍은 복사의 오답 하나하나 (잡음 포함) 이고, 칸은 평균 ≤ 50 인 조건이다. 쌍 단위 부분집합에서 "우연 = 50" 은 predictor 가 복사와 독립일 때의 기대값이고, 복사와 양의 상관이면 50 아래로 내려간다 (IntPhys 2 릴리즈 21).
- **표적 공간.** AR 은 블록별 LN(target_encoder) 공간에서 채점되고 표준 복사는 창 전체 공간이다. 표준 copy-hard 에서의 AR 값은 "다른 공간의 복사가 틀리는 쌍" 에서의 값이라 릴리즈와 대칭이 아니다. IntPhys 1 은 copy_blk 로 대칭 비교를 했고 (§3), v11 은 부분 (§2-4), IntPhys 2 는 없다.
- **best-C.** IntPhys 2 는 같은 C = 24 로 맞췄다 (복사 best C 42 · 릴리즈 24 로 고르면 부호가 바뀐다 — VERIFY_ISCENE2 §8). IntPhys 1 은 격자 칸별로 보고했다.
- **n.** IntPhys 1 copy-hard 는 27 ~ 66 쌍 (CI ±15 ~ 20), 방향 · principle 세분은 n 2 ~ 19 라 읽지 않는다. v11 late 칸은 112 쌍.
- v11 배열 값과 하네스 값은 fp16 동점 근처 쌍이 ±0.24 pt 다르다 (`TrainingEffects/README.md`); Ariel ep43 배열 79.3 vs 하네스 79.4 로 확인.
- 복사는 LN 을 씌운 판 (H6 관례). raw 복사는 IntPhys 1 에서 7 pt 낮았다.

## 재현

```bash
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
/data/hyuntak/anaconda3/envs/vjepa2/bin/python auto_research/scripts/copy_hard_subset.py     # CPU ~3 분 → exp_results/verify/copy_hard_subset.json
```
입력: `/data2/local_datasets/world/world_analysis/cache/training_effects/{v11score_vith_curves,v11score_vith_ar_copyblk,ip1score_final,ip1score_ar_copyblk}`,
`z_research/IntPhysGenV11/exp_results/surprise_c16t32__v11_split_test_vith_ariel_{prefix_ep45,full_ep40,ar_ep38,ar_ep18,bc_ep43}/per_block.json`,
`z_research/Benchmarks/exp_results/intphys1_sliding__intphys1_dev_vith_ariel_{prefix_ep45,full_ep40,ar_ep38}_w{16,32}/per_window.json`,
`z_research/Benchmarks/exp_results/intphys2/bench_intphys2_main_{vith,vith_copy_w48,vith_copy_w32,vith_w32,ariel_*_w48}/per_video.csv`,
`data_csv/intphysgen_v11_split/index_test.csv`, `auto_research/_stage/IntPhys1_dev_by_scene/{pairs,videos}.csv` (te_analyze_ip1 의 Scorer · Boot 규칙 재사용).
