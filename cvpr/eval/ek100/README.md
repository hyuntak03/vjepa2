# eval/ek100 — EK100 action anticipation (V-JEPA 2 §6, Table 19)

**목적.** frozen encoder + predictor 위에 attentive probe 를 학습해 1 s 뒤 action 을 맞힌다 (mean class recall@5, verb/noun/action). 엔진 `evals/action_anticipation_frozen` (evals.main, 업스트림). 논문과 같은 값 32 f @ 8 fps · 256² · probe 4 blocks/16 heads · focal loss · 20 epoch · global batch 128.
논문·릴리즈 코드와 우리 기본값이 다른 곳 (ap 규약 paper · timestamp · head 1 개 · mask_index 0) 은 `z_research/anticipation/EK100/README.md` 가 정본이고 `protocol.yaml` 머리 주석에 요약돼 있다. 옛 `configs/ek100_vith.yaml` 과 config 가 같다 (check_equivalence 2 행, tag/folder 제외).

## 명령
```bash
GPUS=8 bash cvpr/eval/ek100/run.sh ek100 vith                  # head 1 개 (lr 3e-4, wd 1e-2)
GPUS=8 HEADS=sweep bash cvpr/eval/ek100/run.sh ek100 vith      # 논문 20 head (lr 5 × wd 4) — DRYRUN 으로 20 개 확인
GPUS=8 bash cvpr/eval/ek100/run.sh ek100 vith_ariel_prefix_ep45   # 다른 predictor (모델 레지스트리)
GPUS=1 SMOKE=1 bash cvpr/eval/ek100/run.sh                     # 배관 점검: 영상 4 · 1 epoch · batch 2 · resume 끔 → tag ek100_vith_smoke
VAL_ONLY=1 … / DRYRUN=1 bash cvpr/eval/ek100/run.sh
```

## 노브
| env / SET 키 | 기본 | 의미 |
|---|---|---|
| `$1` `$2` / `DATASETS` `MODELS` | `ek100` / `vith` | |
| `HEADS` | `config` (head 1) | `sweep` 20 · `grid8` 8 (24 GB 에 맞춘 격자) · `hi2` 2. `protocol.yaml` 의 `heads_presets` |
| `WINDOW` | `ek100_32f8fps` | `ek100_16f8fps` 면 16 프레임 (tag `ek100_vith_wek100_16f8fps`, DRYRUN 확인; 엔진에서 도는지는 미검증) |
| `SMOKE=1` | — | `limit_videos 4` · `num_workers 2` · `num_epochs 1` · `batch_size 2` · `resume_checkpoint false` · tag `_smoke` |
| `LIMIT=N` | — | `data.limit_videos=N` + tag `_smokeN` |
| `VAL_ONLY=1` | — | `val_only: true` + `--val_only` |
| `TAG` | `<ds>_<model>` | ⚠️ `resume_checkpoint: true` — **설정을 바꾸면 TAG 도 바꾼다** (같은 폴더에 다른 설정으로 이어 돌리지 말 것) |
| `SET="experiment.optimization.num_epochs=5"` | — | 옛 EK100/run.sh 의 `experiment.` 생략 (`data.*`) 은 **안 된다** — 전체 경로로 |
| `GPUS` | `1` | rank 당 batch 16 → 8 GPU 면 global 128 = 논문 |

## 결과 위치
`$CVPR_RESULTS/eval/ek100/action_anticipation_frozen/<ds>_<model>[_w<창>][_smoke|_smokeN]/` — 엔진이 `<folder>/<eval_name>/<tag>` 에 쓴다 — `val_metrics.jsonl` (epoch 마다 한 줄, `exact_val_pass`) · `log_r<rank>.csv` · `latest.pt` (`evals/action_anticipation_frozen/eval.py:185-187`; vll3 smoke 폴더 `ek100_vith_smoke/` 실물로 확인). `smoke_all.sh` 는 `val_metrics.jsonl` 을 본다.
`RESULTS_ROOT=legacy` → `z_research/anticipation/EK100/exp_results/action_anticipation_frozen/ek100_vith` — 그 폴더는 **없다** (옛 run 의 TAG 는 `ek100_vith_official256_paper_grid8` 같은 이름이었다). 옛 결과를 이으려면 `TAG=` 로.

## 옛 명령 대응
| 옛 | 새 |
|---|---|
| `GPUS=8 bash z_research/anticipation/EK100/run.sh` | `GPUS=8 bash cvpr/eval/ek100/run.sh ek100 vith` |
| `HEADS=sweep` / `SMOKE=1` / `VAL_ONLY=1` / `DRYRUN=1` | 그대로 |
| `CONFIG=ek100_<x>` (configs/ 의 다른 yaml) | 모델은 레지스트리 이름으로 (`ek100 <model>`), 나머지 차이는 `SET=` 또는 새 protocol yaml (`extends:`) |
| `python z_research/anticipation/EK100/resolve.py ek100_vith --heads grid8` | `HEADS=grid8 DRYRUN=1 bash cvpr/eval/ek100/run.sh` |

## 단서 · 미완
- 영상은 노드 로컬 `$CVPR_DATA2/local_datasets/EPIC-KITCHENS_resized` (train 495 + val 138 재인코딩본) — 다른 노드면 resolve 가 죽는다.
- `predictor_checkpoint` 를 `model_kwargs.predictor_checkpoint` 로 넣기는 하지만 **이 엔진이 그 키를 읽는지는 미검증** (resolve.py 주석도 그렇게 적었다). 다른 predictor 로 돌리기 전에 `evals/action_anticipation_frozen/` 에서 확인할 것.
- 2026-10-09 vll3 smoke: `SMOKE=1` rc=0 (451 s; exact val epoch 1, 580/580 clip, recall@5 action 1.44 / verb 16.13 / noun 5.08 — 영상 4 개 1 epoch 의 배관 수치, 의미 없음; `cvpr/logs/smoke_vll3.log`). 18:05 요약의 `FAIL ek100 rc=0 산출물=없음` 은 그때 `smoke_all.sh` 가 다른 파일명을 기대해서였다 (지금은 `val_metrics.jsonl`).
