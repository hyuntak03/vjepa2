# V-JEPA 2 release predictor + VOBench post-FT (2026-10-14)

릴리즈 ViT-H predictor (oneshot, mask token) 를 **VOBench 가능 영상만** 으로 post-FT 하면 VOBench 의 late (문맥 끝 가림) 를 배우나.

| 폴더 | 무엇 |
|---|---|
| `SETTINGS.md` | 학습 · 분할 · 채점 · 자 설정 정리 (epoch · step · batch · lr · 데이터 수) |
| `surprise/` | test block 채점 비교 (`compare.{md,json}`) · 그림 `figures/fig_postft_compare_release` (release vs release+FT, 막대 = 쌍 정확도 [block 부트스트랩 95 %], 가로선 = release 공간 복사) · 운동별 `figures/fig_{late,visible,seen}_motion_pft` (+ `_ar70` = AR-state ep70 vast 빗금 막대 추가, VOBench 전체 채점에서 같은 test block 만) |
| `readout/` | 위치 + 정체 자 (VOB_readout 학습셋) 와 VOBench 읽기 — release (`*_rel`) · release+FT (`*_relft`). `../readout/` 로의 링크 (스크립트가 그 자리에 쓴다) |
| `run` | 학습 run `z_training/runs/vobench_release_postft/` (체크포인트 · 로그 · `eval/`) 링크 |

## 설정

- config `configs/training/vobench_release_postft.yaml` (extends `frozen_predictor_postft`): 릴리즈 predictor 에서 시작, 문맥 16 장 → 미래 16 장 (`temporal_prefix`, 표준 채점과 같은 꼴), lr 5e-5, 30 epoch, batch 8 × 4 GPU (22 step/epoch), 무작위 자르기 없음. 30 분 (vll5 24 GB × 4).
- 데이터 = **split v2** (`z_training/data/build_vobench_split_index.py`): 가능 arm 설정 (objset · 운동 · 배경 · 가림 · k · 물체) 을 공유하는 block 을 묶어 묶음째 배정 — **train 과 test 가 공유하는 설정 0** (v1 은 test 가능 영상 52 개가 train 과 같은 설정 · 궤적이었다). train = 360 block 의 가능 영상 720, test = 나머지 360 block × 4 변이 (720 쌍).

## 결과 — 같은 분포 held-out (split v2 test block, 720 쌍)

| | release | **release + FT** | AR ep40 | AR-state | AR-state + FT | 복사 (release 공간) |
|---|---:|---:|---:|---:|---:|---:|
| 전체 | 75.7 | **91.8** | 83.9 | 83.2 | 91.0 | — |
| primitive · realistic | 70.3 · 81.1 | **89.2 · 94.4** | 81.9 · 85.8 | 81.4 · 85.0 | 89.4 · 92.5 | 72.5 · 82.2 |
| 보임 (visible · early · mid) | 81.5 | **92.9** | 86.3 | 85.9 | 91.5 | |
| late 전체 | 62.0 | **89.4** | 78.2 | 76.9 | 89.8 | |
| late · permanence (A 방향) | 54.2 (13.9) | **98.6 (97.2)** | 100 | 97.2 | 100 (100) | |
| late · 모양 · 색 | 65.3 · 66.7 | **87.5 · 81.9** | 62.5 · 72.2 | 63.9 · 69.4 | 86.1 · 83.3 | |
| **late · 모양+색 · 이동** | 66.7 | **83.3** | 62.5 | 60.8 | 81.7 | |

AR-state + FT = ariel AR-state ep40 에서 같은 데이터 (split v2 train 720 clip) · 30 epoch (`vobench_ar_state_postft_v2`, 2026-10-14).
쌍 단위 (release → +FT): 전체 14 vs 130 (McNemar p ≈ 1e-24) · late 4 vs 63 · late 이동 모양+색 4 vs 24 (p 0.0002).
분포 밖 감시 (in-loop IntPhys 1 dev, 60 쌍): 66.7 → 70.0 (떨어지지 않음). 정식 IntPhys 1 · 2 채점은 미실행.

- **release 도 이 장면 분포를 보면 late 를 배운다** — 사라짐 A 방향 14 → 97 %, late 모양 · 색 66 → 82–88 %. AR 계열 (post-FT 없음) 은 존재만 풀고 모양 · 색은 못 풀었는데 (62 %), release+FT 는 둘 다 푼다.

## 상한선 — 가능 영상 전부로 post-FT (`vobench_release_postft_all`, 2026-10-14)

같은 설정 (30 epoch) 에 학습 영상만 VOBench 가능 영상 전부 (1,440 clip, 1,350 step). test 쌍의 가능 영상도 학습에서 봤다 (불가능 영상은 안 봄).

| 같은 split v2 test 720 쌍 | release | held-out FT (720 clip) | **상한선 FT (1,440 clip)** |
|---|---:|---:|---:|
| 전체 | 75.7 | 91.8 | **93.6** |
| 보임 (visible · early · mid) | 81.5 | 92.9 | 93.8 |
| late 전체 | 62.0 | 89.4 | **93.1** |
| late 사라짐 (A) | 54.2 (13.9) | 98.6 (97.2) | 100 (100) |
| late 모양 · 색 | 65.3 · 66.7 | 87.5 · 81.9 | 91.7 · 87.5 |
| late 이동 모양+색 | 66.7 | 83.3 | 88.3 |

- 쌍 단위 held-out → 상한선: 전체 6 vs 19 (McNemar p 0.015) · late 3 vs 11 (p 0.057). **차이는 +2–4 pt 이고, held-out 판이 이미 상한선 가까이 있다.**
- 상한선의 VOBench 전체 1,440 쌍은 93.5 (train 절반 block 93.5 · test 절반 93.6). train 절반의 late 는 86.6 (모양 79.2 · 색 80.6) 으로, 학습에서 본 가능 영상이 있는 쌍에서도 late 모양 · 색이 80 대에 머문다.
- IntPhys 1 in-loop (60 쌍) 66.7 → 66.7.

## 자 (decoder) 로 본 late 행동 — release vs release+FT (test block, primitive 가능 영상, 2026-10-14)

그림 (a) release · (b) release + FT: `readout_figures/fig_timeline_rel_vs_relft_<가림>_<moving|static>` (`plot_vobench_readout_compare.py --occ late`).
자는 VOB_readout 학습셋에서 predictor 마다 따로 학습했다 (held-out p 정체 87.4 / 87.7 · 중심 1.26 / 1.02 칸, h 93.8 · 0.75 칸). 그림 `readout/vobench_{rel,relft}/figures_test_v2/fig_readout_timeline`.

| late · 이동 (t8 → t15) | release | release + FT |
|---|---|---|
| p 정체 정답률 | 3 · 5 · 0 · 0 · 1 · 1 · 0 · 1 % | 20 · **89 · 91 · 93 · 89 · 87 · 93 · 87 %** |
| p '없음' 고름 | 48 → 73–96 % | 40 (t8 은 물체가 실제로 67 % 가려진 튜블릿, h 정체도 23 %) → **0–8 %** |
| 위치 뒤처짐 (등속, 칸) | 물체가 '없음' 이라 읽지 않는다 | −1.3 ~ −2.1 (진실 이동 1.3 → 6.7 칸을 따라간다) |
| 위치 뒤처짐 (가속 / 감속, t15) | 〃 | −3.0 / −1.2 (가속을 덜 반영) |

- **release+FT 의 p 는 late 에서 가림막 뒤로 나온 물체를 바른 정체로 다시 낸다.** release 는 같은 칸에서 '없음' 을 낸다.
- 보일 때 (visible · early) 후반 정체 감소도 사라진다: early 이동 t15 39 → 97 %.
- late static 은 오르지만 (t11–t12 100 %) 후반 t13–t15 에 53–77 % 로 다시 떨어진다 (h 도 67–80 %).
- ⚠️ 같은 분포 (test block) 다 — 장면 설계 · 가림막 · 운동을 학습에서 봤다. 일반화는 IntPhys 1 · 2 로.

## FT 전후 자 시간축 — release · vast AR (2026-10-14)

그림 `readout_figures/fig_timeline_ft_pairs_<가림>_<all|moving|static>` (`plot_vobench_readout_ft_pairs.py`): (a) release vs +FT · (b) vast AR-state ep70 vs +FT, 각 모델의 p 를 그 모델의 p 자로. split v2 test · primitive.
자 검증 (held-out p 정체 · 중심): release 87.4 · 1.26 칸 / release+FT 87.7 · 1.02 / vast 87.3 · 1.78 / vast+FT 90.9 · 1.61.

| late 전체 (90 clip), p 정체 t8 → t15 | FT 전 | FT 후 |
|---|---|---|
| release | 2 · 4 · 0 · 0 · 1 · 1 · 0 · 1 % | 17 · 82 · 89 · 93 · 91 · 80 · 89 · 79 % |
| vast AR-state ep70 | 10 · 12 · 13 · 18 · 21 · 20 · 21 · 21 % | 32 · 80 · 82 · 87 · 89 · 90 · 89 · 86 % |

- 두 출발점 모두 FT 뒤 late 에서 가림막 뒤의 물체를 바른 정체로 다시 낸다 ('없음' 1–11 %). FT 전 vast 는 '없음' 38–50 % · 정체 10–21 % 로 release 보다 조금 나을 뿐.

## IntPhys 2 Main 전체 (506 쌍) — release+FT (2026-10-14)

release 54.35 → release+FT **55.14** (둘 다 22 설정 중 최고 = C24/avg). C 별 avg 도 ±2 안 — 변화 없음.

## 단서

- **같은 분포다.** test block 은 장면 설정까지 train 과 겹치지 않지만 물체 48 · 배경 · 운동 6 · 가림 설계를 공유한다 — "이 분포를 보면 배울 수 있다" 까지만. 일반화는 IntPhys 1 · 2 정식 채점으로.
- 점수가 오른 이유 (예측 구간에서 정체 · 자리를 이어 가게 됐나, 다른 단서를 쓰나) 는 채점만으로 모른다 → `readout/` (자 시간축, test block 만) 로 본다 → 위 '자로 본 late 행동' 절.
- 복사 (release 공간) 는 v2 test block 에서 다시 셈 (`surprise/compare.md`). 칸 72–120 쌍 → CI ±8–10 pt.

## 재현

```bash
python z_training/data/build_vobench_split_index.py                         # split v2
GPUS=4 bash z_training/train.sh vobench_release_postft
GPUS=4 DECODE_WORKERS=8 BATCH_SIZE=16 bash z_training/eval.sh vobench_release_postft vobench_v1_split2_test
python z_research/scripts/analysis/vobench_split_compare.py --split v2      # → surprise/compare.{md,json}
python z_research/scripts/figures/plot_vobench_postft.py --split v2 --base release
for o in late visible seen; do python z_research/scripts/figures/plot_vobench_pft_late_motion.py --occ $o [--ar70]; done
NAME=relft CKPT=z_training/runs/vobench_release_postft/latest.pt FEAT_ROOT=/data2/local_datasets/world/world_analysis/cache bash z_research/scripts/analysis/vob_readout_chain.sh
python z_research/scripts/analysis/vobench_ar_readout.py extract --gpus 4 --name relft --ckpt z_training/runs/vobench_release_postft/latest.pt
python z_research/scripts/figures/plot_vobench_ar_readout.py --name relft --test-only
```
