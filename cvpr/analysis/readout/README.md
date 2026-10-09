# analysis/readout — 학습된 자 (readout, "자") 분석의 단일 진입점 (2026-10-09)

## 목적

RollOutV2 (위치 자) · RollOutV3 (정체 자) 의 분석은 `z_research/scripts/analysis/` 의 **독립 스크립트 20여 개**가
각자 경로를 하드코딩하고 환경변수 (`ROLLOUT2_TRAIN` · `R3_DECODER` · `PRESENCE_SET` …) 를 따로 읽는 상태였다.
이 폴더는 그것들을 **고치지 않고** (엔진 코드 수정 없음) 한 config 로 부른다:

- `config.yaml` — 스텝 33 개 = (스크립트, 인자, 환경변수, 입·출력 실물). 자리표시자 `${CVPR_*}` · `{노브}` · `{ds.키}` · `{model.키}`
- `readout.py` — 치환 · 레지스트리 조회 (`cvpr/harness/resolve.py` 의 함수를 import, 복제 없음) · 실물 검사 · 명령 하나 만들기 · 실행 + 로그
- `run.sh` — `STEP=<이름> [DATASET] [MODEL]` 얇은 껍데기
- `sets/*.set` — 토큰 캐시 추출용 `launch.sh` SET_FILE 셋 (옛 재현의 `$PH` · `$CX` · z16)

과학은 여기 없다. 수치·해석의 정본은 [`z_research/RollOutV2/figures/v5/summary/POSITION_READOUT_2026-09-12.md`](../../../z_research/RollOutV2/figures/v5/summary/POSITION_READOUT_2026-09-12.md)
와 [`z_research/RollOutV3/Archive/IDENTITY_R8_2026-09-26.md`](../../../z_research/RollOutV3/Archive/IDENTITY_R8_2026-09-26.md) 다.

## 명령

```bash
bash cvpr/analysis/readout/run.sh --list                                   # 스텝 33 개 · 노브 17 개
STEP=attn_position DRYRUN=1 bash cvpr/analysis/readout/run.sh              # 명령 · 환경변수 · 입력 OK/MISSING 만 (GPU 0 장)
STEP=attn_position GPUS=2 bash cvpr/analysis/readout/run.sh                # 실행 (GPU 는 스크립트 자신이 쓴다)
STEP="fit_position test_position" bash cvpr/analysis/readout/run.sh        # 차례로, 실패하면 멈춤
STEP=cache_ctx32 GPUS=8 bash cvpr/analysis/readout/run.sh v11_vanish_all   # [DATASET] [MODEL] (launch 스텝 · lock 아닌 스텝만)
STEP=window_readout HELP=1 bash cvpr/analysis/readout/run.sh               # 그 스크립트의 --help (argparse 없는 스텝은 docstring)
GPUS=8 STEP=v11_online bash cvpr/harness/submit.sh cvpr/analysis/readout/run.sh   # sbatch (env 는 --export=ALL 로 그대로) — 미검증
```

### 스텝 (묶음별)

| 묶음 | 스텝 | GPU | 무엇 |
|---|---|---:|---|
| A 캐시 (launch.sh) | `cache_ph` · `cache_ctx32` · `cache_z16` | 8 | attn_probe 프로토콜로 p+h / z(32 frames) / z16 토큰 캐시. probe head 는 버린다 |
| B 위치 자 (RollOutV2) | `fit_position` · `test_position` · `ceiling` | 0 | spatial OLS 자 (대조) |
| | `attn_position` ★ · `attn_position_holdout` · `attn_position_test` | 2/2/1 | attentive 위치 자 학습 · held-out 검증 A · test 만 |
| | `encoder_ruler_z` · `encoder_ruler_h` | 2 | 같은 자를 z / h 에 (encoder 대조) |
| | `two_futures_attn` · `distance_decay` · `token_object_test` | 0 | ledge/wall 슬롯별 근거 · 거리 한계 · 자 없는 검사 C |
| | `attn_diag` · `probe_pos_imp` | 1 | v11 attention 진단 · pos/imp probe |
| C 정체 자 (RollOutV3) | `presence_extract` · `presence_train` | 8 | 학습셋 특징 추출 (FEAT_DIR) · 옛 presence 자 |
| | `identity_dry` · `identity_smoke` · `identity_train` ★ · `identity_bias` | 0/1/6/0 | 정체 자 라벨 개수 · 배관 · 학습 (NAME) · 치우침 |
| | `window_readout` ★ · `window_summary` · `cross_head` | 8/0/0 | v3 16 창 읽기 · 전이 표 · 이식 표 |
| | `v11_online` ★ · `panel_false_alarm` | 8/0 | v11 읽기 (TIMING) · 빈 장면 판 오탐 |
| | `presence_compare` · `doc_numbers` · `paths` · `redraw_figs` | 0 | 오류표 · 문서 표 재계산 · 지문 · 그림 |
| D 체인 (옛 sh) | `identity_chain` · `rerun_v3` | 8 | `decoder_train_redraw.sh` · `rollout3_rerun.sh` 를 그대로 |

## 노브

`run.sh` 환경변수. 스크립트의 **원래 환경변수 (`R3_DECODER` · `ROLLOUT2_TRAIN` …) 를 직접 주지 말 것** — 스텝 env 가 노브 값으로 덮어쓴다.

| env | 기본값 | 의미 |
|---|---|---|
| `STEP` | — | 스텝 이름 (공백으로 여러 개). 없으면 `run.sh` 첫 인자 |
| `DATASET` `MODEL` (= `$1` `$2`) | 스텝의 `dataset:` / `vith` | `dataset_lock: true` 스텝은 다른 DATASET 을 거부 (스크립트가 경로를 하드코딩). MODEL 은 `model_rule` (vith / vith_predictor) 로 검사 |
| `GPUS` / `GPU_IDS="4 5"` | 스텝의 `gpu:` | script 스텝은 `CUDA_VISIBLE_DEVICES`, `--gpus {gpus}` 로; launch 스텝은 launch.sh 로. `cuda: false` 스텝 (`identity_train` = PLAN, 체인) 은 안 건드린다 |
| `DRYRUN=1` | — | 명령 · env · 입력 실물만. launch 스텝은 launch.sh DRYRUN (병합 config) 까지 |
| `HELP=1` | — | 스크립트 `--help` (`help: true` 18 개) / docstring (script 10 + bash 2 = 12 개) / launch 스텝 3 개는 "DRYRUN=1 로 보라" 안내 |
| `FORCE=1` | — | 입력 MISSING 이어도 실행 |
| `OUTDIR` | `$CVPR_RESULTS/analysis/readout/<step>__<ds>_<model>` | `_cmd.json` · `stdout.log` 자리 (launch 스텝은 launch.sh 의 OUTDIR 의미) |
| `ARGS` | "" | 스크립트 인자 뒤에 그대로 (`ARGS='--test-only'`, `ARGS='--windows 16x32'`, `ARGS='--bias'`, `ARGS=12`) |
| `TRAIN` | `v5` | `ROLLOUT2_TRAIN` — 위치 자 학습셋 판. 표가 v5 뿐 (`rollout2_test_readout.py:23`) |
| `EPOCHS` | `300` | attentive 위치 자 epoch |
| `DECODER` | `identity_r8` | `R3_DECODER` — 자 폴더 `<R3_EXP>/<DECODER>` (지문 `01c3afaac37f`). 스크립트 기본 `presence` 와 다르다 |
| `R3_OUT` | `identity_r8` | readings 꼬리표 `windows_<R3_OUT>/` · `v11_<R3_OUT>/` |
| `R3_EXP` | `${CVPR_REPO}/z_research/RollOutV3/exp_results` | 자 · readings 뿌리. ⚠️ `doc_numbers` · `redraw_figs` 는 안 읽는다 (아래 제안) |
| `PSET` | `training_r8` | `PRESENCE_SET` — 정체 자 학습셋 `data_csv/rollout_v2_<PSET>`. C 묶음 스텝의 dataset 이 이것으로 정해진다 |
| `FRAMES` | `{ds.frames_root}` | `PRESENCE_FRAMES` — 학습셋 프레임 (training_r8 → `${CVPR_DATA2}/…/RollOut_v2_training_v8`) |
| `FEAT_DIR` | `${CVPR_DATA2}/…/cache/rollout2_{PSET}_feats` | 학습셋 p·z·h 특징 (123 GB) |
| `NAME` | `identity_r8` | 새 정체 자 폴더. ⚠️ 기본값 = 지금 자 → `identity_train` 을 그대로 돌리면 덮어쓴다 |
| `PLAN` | `p=0,1 z=2,3 h=4,5` | `identity_train` 의 표현 → **물리** GPU |
| `PRESENCE_REPS` | `p,z,h` | `presence_extract` 표현 (Ariel `ar` 는 `p,h`) |
| `TIMING` | `late` | `v11_online` 가림 타이밍 late / mid / early |
| `V3_FRAMES` | `${CVPR_DATA2}/…/RollOut_v3` | `R3_SRC` |
| `R3_TMP` `V11_TMP` | "" (스크립트 기본 `/dev/shm/…`) | memmap 임시 폴더 |
| `R3_STEPS` | `bias extract tables figs gifs` | `rerun_v3` 단계 |
| launch 스텝 전용 | `SET` `SET_FILE` `LIMIT` `RESULTS_ROOT` … | launch.sh 노브가 그대로 통한다 (`SET_FILE` · `TAG` · `SUFFIX` · `GPUS` 는 스텝이 준다) |

`config.yaml` 의 `datasets_inline:` — 레지스트리에 없는 `rollout_v3` · `rollout_v2_training_r8` 을 이 폴더 안에서만 정의한다 (제안 아래).

## 결과 위치

- **스크립트 자신의 자리** (옛 그대로): `z_research/RollOutV2/exp_results/<TRAIN>/{spatial,attentive}_pooling/{p,z,h}/`, `z_research/RollOutV2/figures/<TRAIN>/…`,
  `<R3_EXP>/<NAME|DECODER>/`, `<R3_EXP>/windows_<R3_OUT>/readings.npz`, `<R3_EXP>/v11[_mid]_<R3_OUT>/readings.npz`, `<FEAT_DIR>/{p,z,h}.npy`
- **명령 · 로그 사본**: `$CVPR_RESULTS/analysis/readout/<step>__<dataset>_<model>/{_cmd.json, stdout.log}` (OUTDIR 로 바꾼다)
- **캐시 스텝**: 토큰 캐시 `$CVPR_CACHE/<tag>/` (tag = `<cache_tag>_<model>` · `…_ctx32_<model>` · `…_z16_<model>` — 옛 이름 그대로라 B 묶음 스크립트가 그대로 읽는다),
  실행 기록 `$CVPR_RESULTS/analysis/probing/attn_probe__<ds>_<model>[_ctx32|_z16]/_resolved.yaml` (`RESULTS_ROOT=legacy` 면 옛 폴더)

## 옛 명령 대응표 (old → new)

정본 `## 재현` 절 (POSITION_READOUT_2026-09-12 · IDENTITY_R8_2026-09-26 · RollOutV2/README §6 · RollOutV3/README) 의 줄마다.

| 옛 명령 | 새 명령 |
|---|---|
| `GPUS=8 BATCH_SIZE=8 SET="$PH" bash z_research/scripts/run.sh attn_probe rollout_v2_training_v5 vith` | `STEP=cache_ph GPUS=8 bash cvpr/analysis/readout/run.sh` |
| `… TAG=rollout_v2_training_v5_ctx32_vith SET="$CX" … attn_probe rollout_v2_training_v5 vith` | `STEP=cache_ctx32 GPUS=8 bash …/run.sh` |
| `… SET="$PH" … attn_probe rollout_v2 vith` / `TAG=rollout_v2_ctx32_vith SET="$CX" … rollout_v2` | `STEP=cache_ph … run.sh rollout_v2` / `STEP=cache_ctx32 … run.sh rollout_v2` |
| `… TAG=v11_vanish_all_ctx32_vith SET="$CX" … attn_probe v11_vanish_all vith` | `STEP=cache_ctx32 GPUS=8 bash …/run.sh v11_vanish_all` |
| `… TAG=rollout_v2_z16_vith SET="$S probing.runs=[…[1,16]…]" … rollout_v2` | `STEP=cache_z16 GPUS=8 bash …/run.sh` |
| `BATCH_SIZE=8` (옛 run.sh 는 `surprise.batch_size` 를 덮었다 — attn_probe 는 `features.batch_size` 를 읽으므로 효과 미검증) | 필요하면 `SET="features.batch_size=8"` (yaml 주석: 4 가 최적) |
| `python …/rollout2_fit_readout.py` / `…/rollout2_test_readout.py` | `STEP=fit_position` / `STEP=test_position` |
| `CUDA_VISIBLE_DEVICES=0,1 $PY …/rollout2_attn_readout.py --epochs 300` | `STEP=attn_position GPU_IDS="0 1"` |
| `… rollout2_attn_readout.py --epochs 300 --holdout 0.5` | `STEP=attn_position_holdout GPU_IDS="0 1"` |
| `ROLLOUT2_TRAIN=v5 python …/rollout2_attn_readout.py --test-only` | `STEP=attn_position_test` |
| `CUDA_VISIBLE_DEVICES=2,3 $PY …/rollout2_encoder_readout.py --encoder z --epochs 300` (h 도) | `STEP=encoder_ruler_z GPU_IDS="2 3"` / `STEP=encoder_ruler_h GPU_IDS="4 5"` |
| `… rollout2_encoder_readout.py --encoder z --test-only` | `STEP=encoder_ruler_z ARGS=--test-only` |
| `$PY …/rollout2_ceiling.py` | `STEP=ceiling` |
| `$PY …/rollout2_two_futures_attn.py` | `STEP=two_futures_attn` |
| `python …/rollout2_distance_decay.py` | `STEP=distance_decay` |
| `CUDA_VISIBLE_DEVICES=0 $PY …/v11_readout_attn_diag.py` | `STEP=attn_diag` |
| `$PY …/v11_token_object_test.py [n]` | `STEP=token_object_test [ARGS=n]` |
| `CUDA_VISIBLE_DEVICES=0 $PY …/rollout2_probe_pos_imp.py` | `STEP=probe_pos_imp` |
| `PRESENCE_SET=training_r8 $P …/rollout2_presence_readout.py --set training_r8 --feat-dir … --extract-only --extract-bs 32 --gpus 8` | `STEP=presence_extract GPUS=8` |
| `PRED_CKPT=…/ariel/…/latest.pt PRESENCE_REPS=p,h … --extract-only` (ariel_readout_chain.sh) | `STEP=presence_extract MODEL=vith_ariel_ar_ep18 PRESENCE_REPS=p,h FEAT_DIR=…` |
| `$P …/rollout2_presence_readout.py` (전체) / `bash …/rollout2_presence_sbatch.sh` | `STEP=presence_train GPUS=8` / `GPUS=8 STEP=presence_train bash cvpr/harness/submit.sh cvpr/analysis/readout/run.sh` (미검증) |
| `$P …/rollout2_identity_readout.py --dry` / `--limit 64 --epochs 3 --reps h` / `--bias` | `STEP=identity_dry` / `STEP=identity_smoke` / `STEP=identity_bias` |
| `PRESENCE_SET=… IDENTITY_FEAT_DIR=… $P …/rollout2_identity_readout.py --launch --name identity_r8` | `STEP=identity_train NAME=identity_r8` (`IDENTITY_OUT=<R3_EXP>/<NAME>`, `--name` 은 안 쓴다 — 아래 제안 6) |
| `R3_OUT=… $P …/rollout3_window_readout.py --reps p z h --cross h z --gpus 8` / `--limit 16` / `--reps p h` | `STEP=window_readout GPUS=8` / `ARGS='--limit 16'` / `ARGS` 로 `--reps` 를 바꾼다 (기본 인자 뒤에 붙어 argparse 가 마지막 값을 쓴다) |
| `$P …/rollout3_window_summary.py` / `…/rollout3_cross_head.py` | `STEP=window_summary` / `STEP=cross_head` |
| `$P …/v11_readout_online.py --gpus 8` / `--timing mid` | `STEP=v11_online GPUS=8` / `TIMING=mid` |
| `R3_DECODER=identity R3_OUT=identity $P …/v11_panel_false_alarm.py` | `STEP=panel_false_alarm DECODER=identity R3_OUT=identity` |
| `$P …/presence_readout_compare.py --md` / `--bias` | `STEP=presence_compare` / `ARGS=--bias` |
| `$P …/rollout3_doc_numbers.py --decoder identity_r8` | `STEP=doc_numbers` |
| `$P …/rollout3_paths.py` | `STEP=paths` |
| `$P z_research/scripts/figures/new_archive_redraw.py --decoder identity_r8 --steps figs` | `STEP=redraw_figs` |
| `SET=training_r8 NAME=identity_r8 FRAMES=… bash …/decoder_train_redraw.sh` | `STEP=identity_chain` (PSET · NAME · FRAMES 노브) |
| `R3_OUT=training_v8 bash …/rollout3_rerun.sh` | `STEP=rerun_v3 R3_OUT=training_v8 DECODER=presence` |

## 검증한 것 (2026-10-09, vll5, GPU 0 장)

- `bash -n run.sh` · `py_compile readout.py` · `config.yaml` yaml 파싱 (knobs 17 · steps 33 · inline 2)
- `DRYRUN=1` **33 스텝 전부 rc=0**. launch 스텝 셋은 `launch.sh` DRYRUN 까지 통과 — 병합 config 에서 `probing.runs` 가 `.set` 의 것으로 바뀌고
  `targets=['shape']` · `sweep=1 ([null])` · `num_epochs: 1`, tag = `rollout_v2_training_v5_vith` / `…_ctx32_vith` / `rollout_v2_z16_vith` (옛 캐시 이름과 같다),
  결과 폴더 `attn_probe__<ds>_<model>[_ctx32|_z16]` 로 갈린다
- `HELP=1` 33 스텝 전부 rc=0 (argparse `--help` 18 · docstring 12 · launch 안내 3 — 2026-10-09 검증 때 다시 세어 20/13 을 정정). 모델 로딩 없음
- 거부 검사: `dataset_lock` (fit_position 에 v11) · `model_rule` (vitl) · 모르는 스텝 · launch 허용 목록 밖 — 전부 ERROR 로 멈춘다
- 치환 검사: `MODEL=vith_ariel_prefix_ep45` → `PRED_CKPT=…/block_causal_future_only_1e_5/latest.pt` 가 env 에 들어가고 vith 면 빠진다;
  `GPU_IDS="4 5"` → `CUDA_VISIBLE_DEVICES=4,5 --gpus 2`; `DECODER=presence R3_OUT=test` · `TIMING=mid` · `PSET=training_v6` (FEAT_DIR 안의 `{PSET}` 재귀) · `ARGS`
- **실제 실행 (CPU, 출력만)**: `STEP=paths` → 자 `identity_r8` 지문 `01c3afaac37f`, `windows_identity_r8/readings.npz` 지문 일치;
  `STEP=identity_dry` → 8,640 clip (빈 장면 1,920) · 조합 28,901 · 없음 19,052 · split 4320/1293/3027. `_cmd.json` · `stdout.log` 생성 확인 (OUTDIR 스크래치)
- **GPU 스텝은 하나도 안 돌렸다** (전부 미검증). 캐시 스텝이 새 launch.sh 에서 "학습셋이 비어 있다" exit 1 로 끝나는지도 미검증 (옛 하네스의 기록)

## 단서 / 미완

- **이 노드 (vll5) 에 지금 없는 입력** (DRYRUN MISSING): `$CVPR_CACHE/rollout_v2_training_v5_vith/{predictor,target}.npy,meta.json` ·
  `rollout_v2_training_v5_ctx32_vith/isolated_ctx_0_32.npy` · `v11_full_vith/{predictor,target}.npy` · `RollOutV2/exp_results/v5/spatial_pooling/p/fit/w_p.npy`
  → `fit_position` · `attn_position*` (학습) · `encoder_ruler_*` (학습) · `attn_diag` · `token_object_test` · `test_position` · `ceiling` 은 캐시 (`cache_ph` · `cache_ctx32`) 와 `fit_position` 부터.
  있는 것: `rollout_v2_{vith,ctx32_vith,z16_vith}` · `rollout2_training_r8_feats` · `identity_r8` 자 · `windows/v11_identity_r8` readings · RollOut_v3 · RollOut_v2_training_v8 프레임
- 안 담은 스크립트: `z_research/scripts/figures/plot_*` (그림 20여 개), `ariel_readout_chain.sh` (세 팔 전부 하드코딩 — `presence_extract MODEL=vith_ariel_*` + `identity_train IDENTITY_OUT` 으로 손으로 조립해야 한다),
  `te_v11_readout.py` · `te_v3_readout.py` (TrainingEffects), `rollout2_predictor_locality.py`, `v11_position.py`, `pft_ruler_direct.py`, `audit_*.py`, `data/build_rollout2_index.py`, `harness/merge_token_cache.py` (감속 clip 붙이기)
- 감속 clip (`rollout_v2_decel`) 캐시는 옛 재현이 별도 프로토콜 사본 + `merge_token_cache.py` 로 했다 — 여기 `cache_*` 는 그 절차를 담지 않았다
- `identity_train` 의 `--gpus` 가 docstring 에만 있고 argparse 에 없다 (`rollout2_identity_readout.py:31` vs `--help`) — 스텝은 `--gpus` 를 안 준다
- `ARGS` 는 기본 인자 **뒤에** 붙는다. 같은 옵션을 다시 주면 argparse 는 마지막 값을 쓴다 (`--reps`, `--epochs`). 위치 인자 (`token_object_test`) 도 ARGS 로
- `TRAIN` 은 사실상 v5 고정 (`rollout2_test_readout.py:23` 의 표). `distance_decay` 는 `v5` 폴더를 코드에 박아 TRAIN 을 안 읽는다 (`:21`)
- launch 스텝의 `datasets:` 허용 목록은 옛 재현에서 쓴 조합만이다 (`rollout_v2_training_v5` · `rollout_v2` · `v11_vanish_all`)
- `submit.sh` 경유 (sbatch) 는 안 돌려 봤다 (미검증). `identity_chain` · `rerun_v3` 는 GPU 수를 `nvidia-smi -L` 로 세므로 `GPUS` 가 안 먹는다

## 엔진 수정 제안 (고치지 않았다 — 바깥에서 덮어쓸 수 없는 하드코딩, file:line)

| # | 파일:줄 | 무엇 | 제안 |
|---|---|---|---|
| 1 | `z_research/scripts/analysis/rollout2_test_readout.py:16-18, 24-29` | `ROOT` · `CACHE` (`rollout_v2_vith`) · `INDEX` · `TR_CACHE` · `RES_ROOT` · `FIG_ROOT` 절대경로. 이것을 import 하는 `rollout2_fit_readout.py:21-22` · `rollout2_attn_readout.py:19-20` · `rollout2_encoder_readout.py:26-28` · `rollout2_two_futures_attn.py:14` · `rollout2_ceiling.py:20` · `v11_readout_attn_diag.py:18-20` · `v11_token_object_test.py:12-16` 가 전부 따라간다 | `ROOT = Path(os.environ.get("CVPR_REPO", …))`, 캐시 뿌리 `CVPR_CACHE`, 인덱스 `CVPR_DATA_CSV`. 한 파일만 고치면 B 묶음 전부가 풀린다 |
| 2 | `rollout2_test_readout.py:23` | `TR_NAME = {"v5": …}[TRAIN]` — 학습셋 판을 코드 표로 | `ROLLOUT2_TRAIN_NAME` 환경변수 또는 `rollout_v2_training_{TRAIN}` 규칙 |
| 3 | `rollout2_distance_decay.py:18, 20-21` | 캐시 · 인덱스 · `exp_results/v5/locality` 고정 (TRAIN 무시) | 1 과 같이 `rt.*` 로 |
| 4 | `rollout2_probe_pos_imp.py:23-28` | `ROOT` · 두 캐시 · `OUT` · `FIG` 절대경로 | 1 과 같이 |
| 5 | `rollout2_presence_readout.py:50, 57, 66, 74, 78-79, 91` | `ROOT`; `OUT = …/exp_results/presence` 고정 (`presence_train` 이 옛 자를 덮어쓴다); `FEAT_DEFAULT_DISK` /data2; **encoder 체크포인트 릴리즈 ViT-H 고정** (PRED_CKPT 는 predictor 만); `DATA.root` | `PRESENCE_OUT` 환경변수, 체크포인트는 `cvpr/registry/models.yaml` 의 `vith` 를 읽거나 `ENC_CKPT` 환경변수 |
| 6 | `rollout2_identity_readout.py:49, 51, 423` | `ROOT`; presence 모듈 경로; `--name` 이 `IDENTITY_OUT` 을 무시하고 `ROOT/…/exp_results/<name>` 으로 (R3_EXP 밖) | `--name` 을 `IDENTITY_OUT` 의 부모 아래로. 이 폴더는 `--name` 을 안 쓰고 `IDENTITY_OUT=<R3_EXP>/<NAME>` 으로 우회했다 |
| 7 | `rollout3_paths.py:24, 28` | `ROOT`; `FIGROOT` (옛 그림 뿌리) 고정. `EXP` 는 `R3_EXP` 로 된다 | `R3_FIGROOT` 처럼 `R3_FIGROOT_OLD` 하나 더 |
| 8 | `rollout3_window_readout.py:42, 45, 57-58, 67` | `ROOT`; `INDEX = data_csv/rollout_v3` 고정; encoder 체크포인트 고정; `DATA.root` | `R3_INDEX` 환경변수 + 5 와 같은 체크포인트 처리 |
| 9 | `v11_readout_online.py:39, 46` | `ROOT`; `INDEX = data_csv/intphysgen_v11_full` 고정 (`V11_FRAMES` 만 됨) | `V11_INDEX` 환경변수 |
| 10 | `rollout3_doc_numbers.py:22-25` | `ROOT` · `EXP` · `AUD` · `FIG` 고정 — **`R3_EXP` 를 안 읽는다** (다른 predictor 판의 표를 못 찍는다) | `rollout3_paths.EXP` 를 import |
| 11 | `v11_panel_false_alarm.py:21, 25` | `ROOT`; `FR` = v11 프레임 고정 (`V11_FRAMES` 무시) | `V11_FRAMES` 읽기 |
| 12 | `presence_readout_compare.py:21` · `rollout3_window_summary.py:23` · `rollout3_cross_head.py:25` | `ROOT` 절대경로 (그 밖은 `rollout3_paths` 경유) | `rollout3_paths.ROOT` 를 쓰게 |
| 13 | `z_research/scripts/figures/new_archive_redraw.py:39, 41-42` | `ROOT` · `EXP` (R3_EXP 무시) · `V3_FRAMES` /data2 고정 | `rollout3_paths` 경유 + `R3_SRC` |
| 14 | `decoder_train_redraw.sh:17-20` · `rollout3_rerun.sh:15-16` · `ariel_readout_chain.sh:15-23` | `cd /data/…` · `P=/data/…/python` · `/data2/…` 특징 자리 · Ariel 팔 셋 전부 | `source cvpr/env.sh` 로 `$CVPR_REPO` · `$CVPR_PY` · `$CVPR_DATA2`; `ariel_readout_chain.sh` 는 `presence_extract` + `identity_train` 스텝 조합으로 대체 가능 |
| 15 | `rollout2_presence_sbatch.sh:4, 10, 21-22` | `-w vll5` · 로그 경로 · `PROJ` · `PY` 하드코딩 + 자기제출 패턴 | `cvpr/harness/submit.sh cvpr/analysis/readout/run.sh` (`STEP=presence_train`) 로 대체 |
| 16 | `cvpr/registry/datasets.yaml` (내 폴더 밖) | `rollout_v3` · `rollout_v2_training_r8` 이 없다 → 이 폴더의 `config.yaml datasets_inline` 으로 임시 | 레지스트리에 두 항목 추가 (내용은 `datasets_inline` 그대로) 후 inline 삭제 |
