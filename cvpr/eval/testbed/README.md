# eval/testbed — 자체 testbed surprise 채점 (fixed c16t32) + 복사 기준선

이 폴더 = `run.sh` (표준 채점, `protocol.yaml`) + `copy_baseline.sh` / `copy_baseline.py` (복사 기준선). 수치 · 경로는 전부 이 세션에서 산출물로 확인한 것이고,
돌려 보지 않은 것은 **미검증** 이라고 적었다 (2026-10-09; 이 코드 스페이스에서 GPU 실행은 아직 한 번도 없다).

## 1. 목적

- **`run.sh`** — 합성 testbed (IntPhysGen v11 · v11_realistic · RollOut_v2 …) 를 **fixed context 16 / target 32** 로 채점한다.
  `surprise = mean |predictor(context) − LN(target_encoder(clip))[future]|` (latent L1), matched pair, `surprise(imp) > surprise(pos)` 면 정답, chance 50 %.
  엔진 `evals/world_model_analysis/eval.py` 를 `cvpr/harness/launch.sh` 가 띄운다 (CLAUDE.md §1-7 의 고정 프로토콜 스펙 = `protocol.yaml`).
- **`copy_baseline.sh`** — 같은 쌍을 **예측 없이** 채점하는 기준선. VoE 점수는 예측 없이도 풀린다 (leakage) 는 것을 재기 위해 **모델 점수에는 항상 복사 대비 Δ 를 같이 낸다** (CLAUDE.md §0 · §5-6 · §12-5).
  dinof_* 는 로더 플래그 (`model.dinof_copy=true`, `analysis/model_loaders.py:342`) 로 `run.sh` 가 직접 돌리고, **V-JEPA 계열은 기존 분석 스크립트 `z_research/scripts/analysis/te_v11_score.py` 를 `copy_baseline.py` 가 import 해 돌린다** (엔진 수정 없음).

### 1-1. 복사 기준선 — 정의 (testbed, `te_v11_score.py` docstring 의 식 그대로)

clip 하나 (raw 100 장 stride 3 → 32 장 = 튜블릿 16 개; 문맥 = 앞 16 장 = 튜블릿 0..7, 미래 = 튜블릿 8..15), S = 공간 토큰 256, D = 1280 (ViT-H):

```
z   = encoder(문맥 16 장)                              online `encoder` (model.context_encoder_key), 최종 norm 포함          (8·S, D)
h   = LN(target_encoder(32 장 전부))                   EMA `target_encoder`, affine 없는 토큰 LN (surprise.target_layer_norm) (16·S, D)
p   = predictor(z, ctx_idx, tgt_idx, mask_index=0)                                                                       (8·S, D)
zT  = LN(z)[튜블릿 7]                                  문맥 encoder 쪽 **마지막 문맥 튜블릿** (raw 42 · 45 장) 에 같은 affine 없는 LN

S_model(clip) = mean_{j=0..7} mean_{S·D} | p[j]  − h[8+j] |     = 표준 surprise (하네스 per_video_surprise 와 같은 값; te 검증 max|Δ| ≤ 3.5e-4)
S_copy (clip) = mean_{j=0..7} mean_{S·D} | zT    − h[8+j] |     = 마지막 관측 latent 를 미래 8 슬롯 전부에 복사 — predictor 를 쓰지 않는다
S_copy_target = mean_{j=0..7} mean_{S·D} | h[7]  − h[8+j] |     = target 쪽 마지막 문맥 튜블릿 복사 (참고용; h[7] 은 32 장을 다 본 양방향 표현)
```

- **pairing = matched** (`scoring.pairing: matched`, `protocol.yaml`): block 의 (block_id, pair_id) 쌍 = 문맥이 **픽셀 단위로 같은** 가능 1 · 불가능 1.
  문맥이 같으니 z · zT · p 가 두 영상에서 비트 동일하고, 채점은 "마지막 관측 (zT) 또는 예측 (p) 이 두 미래 중 어디에 가까운가" 로 환원된다.
- **copy accuracy** = 쌍마다 `hit = 1[S_copy(pos) < S_copy(imp)] (+0.5 동점)` 의 평균 (`te_v11_score.matched_acc` = `eval.py::score_blocks` 규칙). chance 50 %.
  **model accuracy** 도 같은 쌍 · 같은 규칙 (S_model). **Δ = model − copy** (pt, 같은 쌍 위의 차). Δ ≈ 0 이면 그 칸의 점수는 predictor 가 아니라 "마지막 관측과의 거리" 로 이미 풀린다.
- **읽는 법 (leakage)** — 복사가 높은 벤치마크는 예측 없이 풀리는 벤치다. IntPhys 1 dev: 복사 **85.0** vs 릴리즈 ViT-H **88.89** (Δ +3.9 [−0.6, +8.9], CI 0 포함; `../intphys1/README.md`).
  v11_split_test (10,752 쌍): 복사 **73.24** vs 릴리즈 **75.70** (Δ +2.46; `/data2/local_datasets/world/world_analysis/cache/training_effects/v11score_vith_curves/summary.json` 에서 이 세션에 다시 읽음;
  CI 판 +2.5 [+1.6, +3.4] 은 `z_research/TrainingEffects/README.md` 표 1-1). **릴리즈의 +2.5 는 전부 색 위반 (+9.3) 에서 나오고 모양 (−1.6) · vanish 물체→빈 (−6.8) 은 복사보다 낮다** (같은 문서).
  DINO-Foresight: v11 77.10 vs 복사 76.44 · IntPhys 1 84.4 < 복사 88.9–92.2 (CLAUDE.md §5-6). 위치만 다른 쌍 (RollOut_v2 ledge / wall) 은 둘 다 chance 아래 (p 0.0 / 17.6, 복사 7.9 / 2.3 — `auto_research/Archive/COPY_VS_POSITION_2026-10-07.md`):
  토큰 평균 L1 은 "무엇이 / 있는지" 는 보고 "어디 있나" 는 거의 못 본다. 그래서 **복사가 높다 = 모델이 나쁘다** 가 아니라 **그 칸은 예측을 재지 않는다** 로 읽는다.
- **A/B 방향** — v11 의 pair_id A / B 가 두 방향이다 (vanish 50 % 는 100 과 0 의 평균, CLAUDE.md §1-7). `copy_summary.json` 의 `by.condition|violation_type|pair_id` 가 그것이다.
  (index 의 `direction` 열은 운동 종류 (constant_velocity …) 이지 A/B 가 아니다.)

## 2. 명령

```bash
GPUS=8 bash cvpr/eval/testbed/run.sh v11 vith                              # 표준 채점 (73.37 %, 10,752 쌍 — 옛 산출물 값)
COPY=1 GPUS=8 bash cvpr/eval/testbed/run.sh v11_split_test vith             # 복사 기준선 (V-JEPA 계열 → copy_baseline.sh; dinof_* → 로더 플래그)
GPUS=8 bash cvpr/eval/testbed/copy_baseline.sh v11_split_test vith          # 복사 기준선 직접 (model `own` + copy 를 한 번에, Δ 까지)
DRYRUN=1 bash cvpr/eval/testbed/copy_baseline.sh v11_split_test vith        # 두 resolver 대조 · 이름 · 실행할 명령만 (GPU 0 장, ~20 s)
LIMIT=16 GPUS=1 bash cvpr/eval/testbed/copy_baseline.sh v11_split_test vith # 스모크 (폴더에 _smoke16)
```
SLURM: `GPUS=8 bash cvpr/harness/submit.sh cvpr/eval/testbed/copy_baseline.sh v11_split_test vith` (env 는 `--export=ALL` 로 그대로 넘어간다; 미검증).

## 3. 노브 표

### `run.sh` (전부 `cvpr/harness/launch.sh` 로 넘어간다)

| env / SET 키 | 기본값 | 의미 |
|---|---|---|
| `$1` / `DATASETS`, `$2` / `MODELS` | `v11`, `vith` | 데이터셋 · 모델 (공백 구분 행렬). 목록 `python cvpr/harness/resolve.py --list` |
| `GPUS`, `GPU_IDS` | `1`, `0..GPUS-1` | GPU 수 / 직접 지정 (`GPU_IDS` 가 이긴다; 바깥 `CUDA_VISIBLE_DEVICES` 는 evals.main 이 덮는다) |
| `WINDOW` | `c16t32` | 창 preset (`cvpr/registry/windows.yaml`). 기본이 아니면 tag · 폴더에 `_w<이름>` |
| `SET="a.b=1 c.d=null"` | — | 병합 config 점 경로 덮어쓰기 (`null` = 삭제). 예 `scoring.pairing=cross` (gravity_realistic 은 레지스트리 overrides 가 알아서) |
| `LIMIT=N` / `SMOKE=1` / `RECACHE=1` | — | block N 개 (+`_smokeN`) / shape 디버그 / 토큰 캐시 무시 |
| `TAG`, `OUTDIR`, `SUFFIX` | 자동 | 캐시 이름 · 결과 폴더 · 접미사 |
| `RESULTS_ROOT` | `cvpr` | `legacy` 면 옛 폴더 `z_research/<셋>/exp_results/surprise_c16t32__<ds>_<model>` |
| `DRYRUN=1` | — | 병합 · 실물 검사만 |
| `COPY=1` | — | 복사 기준선. `dinof_*` → `SET model.dinof_copy=true` + `_copy` 접미사 (launch.sh) / 그 밖 → `copy_baseline.sh "$d" "$m"` |

### `copy_baseline.sh` → `copy_baseline.py` → `te_v11_score.py`

| env | 기본값 | 의미 (→ te 인자) |
|---|---|---|
| `$1` / `DATASETS`, `$2` / `MODELS` | `v11_split_test`, `vith` | **양쪽 레지스트리** (`cvpr/registry/*.yaml` 과 `configs/protocols/*.md`) 에 있는 이름. 모델은 V-JEPA 계열 (dual_encoder · LN target) 만 — te 가 규격을 검사한다 (`te_v11_score.py:226-233`; vjepa21g · videomae2g · dinof 는 거부) |
| `GPUS`, `GPU_IDS` | `1` | → `CUDA_VISIBLE_DEVICES` + `--gpus N` (te 는 mp.spawn 으로 보이는 GPU 앞 N 장을 `cuda:0..N-1` 로 쓴다 — evals.main 과 반대로 CUDA_VISIBLE_DEVICES 가 먹는다) |
| `LIMIT=N` | — | `--limit N` (block 단위로 약 N clip, 쌍 온전) + 폴더 `_smokeN` |
| `DRYRUN=1` | — | 두 resolver 대조 · 이름 · 명령만. 모델 · GPU 없음 |
| `VALIDATE=1` | — | 끝난 뒤 `te --validate` (GPU 1 장, 48 clip: own 과 하네스 per_block.json 대조). 참조가 v11 것이라 **v11_split_test 만** |
| `OUTDIR` | `$CVPR_RESULTS/eval/testbed/copy/<ds>_<model>[_smokeN]` | 결과 폴더 (데이터셋 · 모델이 하나일 때만) |
| `BATCH`, `WORKERS`, `THREADS` | te 기본 8 · 6 · 4 | rank 당 clip · PNG 디코드 프로세스 · torch 스레드 |
| `WINDOW`, `SET` | — | **적용되지 않는다.** `WINDOW≠c16t32` 면 죽고, `SET` 은 경고만 (te 는 `surprise_c16t32` 병합 config 그대로) |
| `TAG` | — | 지운다 (옛 resolver 가 읽는 env; te 도 지운다 `te_v11_score.py:213`) |

`copy_baseline.py` 자체 인자: `--dataset --model --gpus --limit --batch --workers --threads --out --dryrun --validate --no-check` (`--help`).

## 4. 결과 위치

| 무엇 | 어디 |
|---|---|
| 표준 채점 | `$CVPR_RESULTS/eval/testbed/<ds>_<model>[_w<창>][_copy][_smokeN]/{summary.json, per_block.json, _resolved.yaml, _meta.json, stdout.log}` |
| 복사 기준선 (V-JEPA) | `$CVPR_RESULTS/eval/testbed/copy/<ds>_<model>[_smokeN]/` — `summary.json` (te: `preds.own.acc` · `copy.acc`, 슬롯 평균), **`copy_summary.json`** (model · copy · copy_target · Δ, 전체 + condition · violation_type · pair_id), `l1.npy (n,1,8)` · `copy.npy (n,8)` · `hcopy.npy` · `hstep.npy` · `pz.npy` · `pstep.npy` · `done.npy` · `meta.json` (video_id 순서 · 라벨 열 · 병합 config) · `_resolved.yaml` (옛 resolver 산출) · `_meta.json` (cvpr resolver meta + 실행 인자) · `progress.jsonl` · `doc/` (VALIDATE 산출) |
| 복사 기준선 (dinof) | `$CVPR_RESULTS/eval/testbed/<ds>_dinof_*_copy/` (표준 채점과 같은 형식) |

`$CVPR_RESULTS` 기본값 = `cvpr/results` (`cvpr/env.sh`). 재개: 같은 명령을 다시 치면 te 가 `meta.json` 을 대조하고 `done==0` 행만 돈다 (다른 preset · 모델이면 죽는다 — 덮어쓰지 않는다).

## 5. 옛 명령 대응표

| 옛 | 새 |
|---|---|
| `GPUS=8 bash z_research/scripts/run.sh surprise_c16t32 v11 vith` | `GPUS=8 bash cvpr/eval/testbed/run.sh v11 vith` (`RESULTS_ROOT=legacy` 면 옛 폴더 그대로) |
| `SET="scoring.pairing=cross" ... run.sh surprise_c16t32 gravity_realistic` | `bash cvpr/eval/testbed/run.sh gravity_realistic` (레지스트리 `overrides`) |
| `COPY=1 ... run.sh ... dinof_highres` (`SET model.dinof_copy=true`) | `COPY=1 bash cvpr/eval/testbed/run.sh v11 dinof_highres` (그대로) |
| `CUDA_VISIBLE_DEVICES=0..7 python z_research/scripts/analysis/te_v11_score.py --model vith --preset own --gpus 8` → `/data2/.../training_effects/v11score_vith_own/` | `GPUS=8 bash cvpr/eval/testbed/copy_baseline.sh v11_split_test vith` → `$CVPR_RESULTS/eval/testbed/copy/v11_split_test_vith/` (+ `copy_summary.json`) |
| `... te_v11_score.py --model vith --limit 48` | `LIMIT=48 bash cvpr/eval/testbed/copy_baseline.sh v11_split_test vith` |
| `... te_v11_score.py --model vith --validate` → `z_research/TrainingEffects/v11score/validate_vith.json` | `VALIDATE=1 ...` → `<out>/doc/validate_vith.json` |
| `python z_research/scripts/analysis/te_analyze_scores.py` (CI · 학습 곡선 · 슬롯별) | 대응 없음 — 옛 폴더 이름을 읽으므로 그대로 쓴다 (미이식) |

옛 복사 기준선 산출물 (preset final / curves, predictor 여러 개 + copy): `/data2/local_datasets/world/world_analysis/cache/training_effects/v11score_vith_{curves,...}/` — 거기 `copy.npy` 가 여기 `copy.npy` 와 같은 정의다.

## 6. 단서 / 미완

- **이 코드 스페이스에서 GPU 로 돌린 적이 없다.** 검증 = `bash -n` · `py_compile` · `--help` · `DRYRUN=1` (두 resolver 의 data · model · surprise · scoring 블록 일치: v11_split_test · v11_realistic × vith 확인; `cvpr/harness/check_equivalence.py` 27 행 동일 · 0 차이 · 1 건너뜀, 이 세션 재실행). 끝까지 도는지 · 수치가 옛 폴더와 같은지는 **미검증** (첫 실행은 `LIMIT=16 GPUS=1` 로).
- **데이터셋은 v11_split_test 가 정식이다.** `te_v11_score.py:156` 이 데이터셋을 상수로 갖고 있어 다른 데이터셋 (v11 · v11_realistic · v11_realistic_ledge · rollout_v2 …) 은 래퍼가 **부모 프로세스의 모듈 상수를 덮어쓴다** (파일 수정 없음; spawn 된 rank 는 병합 config 를 받는다 `:477-483`). DRYRUN 은 통과하지만 실제 실행은 **미검증**. `gravity_realistic` 은 pairing cross 라 옛 resolver 와 scoring 이 달라 죽는다 (의도 — te 의 `matched_acc` 는 cross 를 모른다).
- **`own` = 병합 config 대로 지은 predictor.** vith 면 model.pth 의 릴리즈 predictor (te 검증 (a): `release` 추출본과 가중치 · p 모두 max|Δ| 0.0), `vith_pft_*` · `vith_ariel_*` 면 그 predictor_checkpoint. kind=ar (`vith_ariel_ar_*`) 는 te 가 블록 공간으로 분기하고 `copy_blk.npy` 를 더 만든다 — `copy_summary.json` 은 그 분기를 모른다 (**AR 모델에서는 쓰지 말 것**, summary.json 의 `copy_blk` 를 볼 것).
- **encoder 교차.** copy 는 LN(online z) 이고 h 는 EMA target encoder 다 — 두 encoder 의 척도 차가 S_copy 에 섞인다 (`z_research/TrainingEffects/ip1_copy/IP1_COPY.md` 단서 9). `copy_target` (h[7]) 는 target 쪽이지만 32 장을 다 본 양방향 표현이라 "깨끗한 복사" (target encoder 를 문맥 16 장에만 돌린 것) 는 아니다 — 그것은 아직 아무 데도 없다 (같은 문서 §10a).
- **CI 는 내지 않는다.** `copy_summary.json` 은 점 추정 (pt) 뿐. block bootstrap 은 `z_research/scripts/analysis/te_analyze_scores.py` (옛 폴더 이름 기준, 미이식).
- fp16 배치 구성 잡음: 슬롯 평균 ±2.5e-4 → 동점 근처 쌍이 뒤집혀 하네스 값과 ±0.24 pt 다를 수 있다 (te docstring 09-25).
- `--validate` 는 참조 per_block (`z_research/IntPhysGenV11/exp_results/surprise_c16t32__v11_vith`, `..._v11_earlymid_vith`) 을 읽으므로 v11_split_test × vith (또는 `surprise_c16t32__v11_split_test_<model>` 이 있는 모델) 만.
- 토큰 캐시는 쓰지 않는다 (te 는 encoder 를 매번 돌린다, ViT-H 8 GPU ≈ 0.2 s/clip/rank — te docstring 값).
- 관련: IntPhys 2 엔진에는 이미 복사 스위치가 있다 (`analysis/intphys2/surprise.py:229 _copy_baseline`, env `IP2_COPY=1` `:368 · :573`) — `cvpr/eval/intphys2` 쪽.

## 7. 엔진 수정 제안 (하지 않았다 — 제안만)

1. **`z_research/scripts/analysis/te_v11_score.py` 에 `--dataset` (기본 `v11_split_test`)** — `:156` 의 `DATASET` 상수를 `resolve_cfg(model, dataset)` (`:211`, `:217`) 인자로, `main_run` `:589` · `main_validate` `:704` 에서 넘기고 `meta["dataset"]` `:603` 에 적는다. `ref_files` `:672` 는 `surprise_c16t32__<dataset>_<model>/per_block.json` 을 먼저 찾게. `OUT_ROOT` `:158` · `DOC_DIR` `:159` 는 `--out` / `--doc-dir` 로. 그러면 래퍼의 모듈 상수 덮어쓰기가 필요 없다.
2. **(장기) wma 엔진 복사 스위치** — `evals/world_model_analysis/eval.py:571` (single) · `:799` (intphys1 sliding) 의 `bundle.predictor(z, ci, ti, mask_index=…)` 자리에서 `analysis/intphys2/surprise.py:229 _copy_baseline` (문맥 마지막 튜블릿 토큰 → affine 없는 LN → 미래 튜블릿 수만큼 복제) 과 같은 함수를 `model.copy_baseline: true` (또는 `WMA_COPY=1`) 일 때 쓴다.
   그러면 `COPY=1` 이 모든 모델 · 창 · 데이터셋에서 `launch.sh` 를 그대로 타고 `_copy` 이름 · `summary.json` 의 `by_block_type` · 토큰 캐시 재사용이 공짜다. 넣으면 te 의 `copy.npy` 와 48 clip 비트 대조 (≤ 1e-6) 를 붙일 것.
