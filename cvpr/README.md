# cvpr/ — 평가·분석의 새 진입점 (2026-10-09)

> 이 한 장이 시작점이다. 세부는 각 작업 폴더의 `README.md` (`eval/{testbed,intphys1,intphys2,ek100}` · `analysis/{probing,knockout,readout}`).
> 옛 진입점 (`z_research/scripts/run.sh` 등) 은 **그대로 돈다.** 레포 운영 규칙은 루트 `CLAUDE.md`.
> 아래 수치·명령은 전부 이 세션에서 `DRYRUN=1` · `bash -n` · `python` 으로 확인했다. 확인 못 한 것은 "미검증" 이라고 적었다.

## 1. 30초 요약

- 하나의 resolver (`harness/resolve.py`) 가 **프로토콜 yaml + 창(window) preset + 레지스트리 (datasets · models · windows) + `SET=`** 를 합쳐 config 하나를 만들고,
  하나의 런처 (`harness/launch.sh`) 가 엔진 셋 (`wma` = `evals/world_model_analysis` · `intphys2` = `analysis/intphys2` · `ek100` = `evals/action_anticipation_frozen`) 중 맞는 명령으로 띄운다.
- 절대경로·노드·인터프리터는 **`env.sh` 한 파일** 에만 있다 (`CVPR_*`). 레지스트리 yaml 은 `${CVPR_*}` 로만 경로를 적는다.
- 옛 resolver 와 **같은 config** 를 낸다 — `python cvpr/harness/check_equivalence.py` 가 28 행 (옛 `--set` 조합 vs 새 창 preset) 을 대조한다.
  이 세션 실행 결과: **동일 27 · 차이 0 · 건너뜀 1** (건너뜀 = `attn_probe_imp v11` — 옛 쪽도 `index_probe_imp.csv` 가 없어 실패). 27 중 7 행은 tag/output_dir 에 `_w<창>` 접미사만 다르다 (`OK(창 접미사)`, §6).
- 엔진 코드 (`evals/ analysis/ src/ app/`) 는 손대지 않았다. 옛 결과 폴더·토큰 캐시는 `RESULTS_ROOT=legacy` 로 그대로 이어진다 (§8).

## 2. 구조

```
cvpr/
├── env.sh                     CVPR_* 변수 (경로·노드·인터프리터). 모든 run.sh 가 맨 먼저 source
├── registry/
│   ├── datasets.yaml          데이터셋 (engine · raw_frames · cache_tag · legacy_results_root · overrides · available)
│   ├── models.yaml            모델 (base: vith · predictor.<키> · surprise.<키> · family · available)
│   └── windows.yaml           창 preset (fixed · sliding · intphys2 · ek100)
├── harness/
│   ├── resolve.py             단일 resolver. `--list` 가 목록. 모델 로딩 전에 실물 검사
│   ├── launch.sh              공용 런처: resolve → _resolved.yaml/_meta.json → DDP 포트 + *_EXPECT_WS → 엔진 명령
│   ├── submit.sh              sbatch 템플릿 하나 (run.sh 와 파일이 달라 자기제출 폭주가 없다)
│   ├── check_equivalence.py   옛 resolver == 새 resolver 검사 (GPU 없음, 몇 초)
│   ├── smoke_all.sh           다섯 엔진 경로 배관 점검 (1 GPU, LIMIT/SMOKE)
│   └── import_registry.py     configs/protocols/*.md → registry/*.yaml 변환기 (다시 돌리면 yaml 을 덮는다)
├── eval/
│   ├── testbed/{protocol.yaml,run.sh}     자체 testbed surprise (fixed c16t32, matched pair)       engine wma
│   │     + copy_baseline.{sh,py}          복사 기준선 (V-JEPA 계열, te_v11_score.py 를 import)
│   ├── intphys1/{protocol.yaml,run.sh}    Garrido A.8 sliding — IntPhys1 · GRASP · InfLevel         engine wma
│   │     + copy_baseline.{sh,py}          복사 기준선 (intphys1_dev × vith, te_intphys1_score.py)
│   ├── intphys2/{protocol.yaml,run.sh}    IntPhys 2 논문 D.3 격자                                   engine intphys2
│   └── ek100/{protocol.yaml,run.sh}       EK100 action anticipation                                 engine ek100
├── analysis/probing/{attn_probe,attn_probe_xfer,attn_probe_imp}.yaml + run.sh   z/p/h attentive probing   engine wma
├── analysis/knockout/{protocol.yaml,run.sh,check.py}   predictor attention knockout (resolve.py → run.py --shard × N → merge.py; launch.sh 분기 밖)
├── analysis/readout/{config.yaml,readout.py,run.sh,sets/}   학습된 자 (RollOutV2 · V3) 분석 스텝 33 개 (옛 스크립트를 config 로 부른다)
├── results/                   $CVPR_RESULTS — 새 결과 루트 (§8)
└── logs/                      $CVPR_LOGS — sbatch 로그 · smoke 로그
```

`run.sh` 는 얇다: `source env.sh` → `$1`/`$2` 또는 env 에서 DATASETS/MODELS → 루프 → `harness/launch.sh <protocol.yaml> <ds> <model>` (`WINDOW`/`SET`/`SUFFIX` 는 env 로).

## 3. 실행

```bash
GPUS=8 bash cvpr/eval/testbed/run.sh v11 vith                                   # 자체 testbed (73.37 % 칸)
GPUS=8 WINDOWS=garrido_skip2_w32 bash cvpr/eval/intphys1/run.sh intphys1_dev vith   # IntPhys 1 (88.89 칸). WINDOWS 생략 = 창 16·32 둘 다
GPUS=8 bash cvpr/eval/intphys2/run.sh intphys2_main vith                        # IntPhys 2 (창 16·32·48 순서로 돈다)
GPUS=8 bash cvpr/eval/ek100/run.sh ek100 vith                                   # EK100 (head 1 개; HEADS=sweep 이 논문 20 개)
GPUS=8 bash cvpr/analysis/probing/run.sh v11 vith                               # attn_probe (PROTO=attn_probe_xfer|attn_probe_imp)

GPUS=8 bash cvpr/harness/submit.sh cvpr/eval/testbed/run.sh v11 vith            # 위 어느 run.sh 든 sbatch 로 (env 그대로 넘어간다)
GPUS=1 bash cvpr/harness/smoke_all.sh                                           # 다섯 경로 배관 점검 (ONLY="testbed intphys1" 로 일부만)
DRYRUN=1 bash cvpr/eval/testbed/run.sh v11 vith                                 # 병합 config 만 (GPU 0 장, 몇 초)
python cvpr/harness/resolve.py --list                                           # 프로토콜 · 데이터셋 · 모델 · 창 목록
```

`launch.sh` 를 직접 부를 때 프로토콜 철자는 `eval/testbed` · `analysis/probing/attn_probe` · yaml 경로 셋이다 (`cvpr/eval/testbed` 는 **안 된다** — `resolve.find_protocol` 이 `cvpr/<spec>/protocol.yaml` 을 찾는다; launch.sh 머리 주석은 2026-10-09 에 고쳤다).
여러 데이터셋·모델은 공백 구분: `bash cvpr/eval/testbed/run.sh "v11 v11_realistic" "vith vitl"`.

## 4. 노브 (환경변수) — 스크립트에서 읽어 적었다

| 변수 | 기본 | 읽는 곳 | 의미 |
|---|---|---|---|
| `GPUS` | `1` (launch) / `8` (submit) | launch · submit | GPU 수 → `--devices cuda:0..N-1` + `WMA_EXPECT_WS`/`ANT_EXPECT_WS` (split-brain 가드) |
| `GPU_IDS` | `0..GPUS-1` | launch | 쓸 GPU 를 직접 (`"1 2 3"`). 주면 GPUS 는 이 개수로 바뀐다. 바깥 `CUDA_VISIBLE_DEVICES` 는 wma/ek100 에서 안 먹는다 (`evals/main.py:51` 이 덮는다); intphys2 는 GPU_IDS 로 만들어 준다 |
| `WINDOW` | 프로토콜의 `window:` | launch | 창 preset 이름 하나 (§6). 기본 창이 아니면 이름에 `_w<창>` |
| `WINDOWS` | run.sh 마다 (§6) | intphys1 · intphys2 run.sh | 창 preset **목록** — 창마다 실행을 나눈다 |
| `SET` | — | launch | `"a.b=1 c.d=null"` 점 경로 덮어쓰기 (공백 구분, 값은 YAML, `null` = 삭제). `window.<키>=` / `window=<이름>` 은 preset 에 먼저 덮인다. `heads=` 는 ek100 preset |
| `SET_FILE` | — | launch | 한 줄에 `KEY=VALUE` 하나 (`#` 주석 허용). 값에 공백·콤마가 필요할 때 |
| `LIMIT` | — | launch | wma: `limit: N` + 이름에 `_smokeN` / intphys2: `evaluation.limit_videos=N` + `_smokeN` / ek100: `data.limit_videos=N` + tag `_smokeN` |
| `SMOKE` | — | launch | wma: `smoke: true` (shape 디버그, **접미사 없음**) / intphys2: `limit_videos=8` 인데 **접미사가 없어 본 폴더에 쓴다** — LIMIT 을 쓸 것 / ek100: 영상 4 · 1 epoch · batch 2 · resume 끔 · tag `_smoke` |
| `RECACHE` | — | launch (wma) | `recache: true` — 토큰 캐시 무시하고 재추출 |
| `TAG` / `OUTDIR` | 자동 (§8) | launch | 이름을 직접. OUTDIR 은 `_smokeN` 이 뒤에 더 붙는다 |
| `SUFFIX` | — | launch | tag · output_dir 접미사 (`_copy`). 창 접미사 뒤 · `_smoke` 앞. 앞의 `_` 는 없어도 붙여 준다 |
| `RESULTS_ROOT` | `cvpr` | launch | `cvpr` (`$CVPR_RESULTS/<task>/…`) · `legacy` (옛 폴더 규약) · `<경로>` |
| `DRYRUN` | — | launch | 병합·실물 검사만 하고 config 를 출력. 모델 로딩 없음 |
| `FORCE` | — | launch | 결과 폴더에 끝난 산출물 (`summary.json` / ek100 `val_metrics.jsonl`) 이 있으면 기본은 **건너뛴다** (옛 run_all.sh 와 같음). `1` 이면 덮어쓴다 |
| `VAL_ONLY` | — | launch (ek100) | `val_only: true` + `--val_only` |
| `COPY` | `0` | testbed · intphys1 run.sh | `1` = 복사 기준선. dinof_* 는 `model.dinof_copy=true` + 이름 `_copy` (launch.sh 그대로); 그 밖 (V-JEPA 계열) 은 같은 폴더의 `copy_baseline.sh "$d" "$m"` 으로 (intphys1 은 첫 창에서 한 번만). 각 폴더 README |
| `HEADS` | `config` (head 1) | ek100 run.sh | `sweep` (20 = lr 5 × wd 4) · `grid8` (8) · `hi2` (2) — DRYRUN 으로 개수 확인함 |
| `DATASETS` / `MODELS` | run.sh 마다 (§3) | 모든 run.sh | `$1` / `$2` 가 없을 때의 목록 |
| `BENCHES` | 5 벤치 전부 | intphys1 run.sh | `$1` 이 없을 때의 벤치 목록 |
| `MB21` / `MBVM` | `8` / `8` | intphys1 run.sh | `surprise.intphys1.max_batch` — vjepa21g 전부 / videomae2g 창 32. VRAM 상한, 수치 무관 |
| `PROTO` | `attn_probe` | probing run.sh | `attn_probe` · `attn_probe_xfer` · `attn_probe_imp` |
| `MODEL` | `vith` | launch | `$3` 이 없을 때의 모델 |
| `EVAL_DDP_PORT` | 자동 (`20000 + (SLURM_JOB_ID|$$) % 10000` 부터 200 개 탐색) | launch → `evals/main.py:96` | 지정하면 탐색 생략 |
| `EVAL_DDP_TIMEOUT_S` | `7200` (env.sh) | `evals/main.py:97` · `analysis/intphys2/eval.py:114` | NCCL timeout. torch 기본 600 s 는 rank 불균형에서 죽는다 |
| `WMA_BAR` | `auto` | `evals/world_model_analysis/eval.py:102` | 진행바 `on`/`off`/`auto` (auto = stderr 가 TTY 일 때) |
| `OMP_NUM_THREADS` | `min(nproc/GPUS, 12)` | launch | rank 당 CPU 스레드 (전체 코어를 잡으면 decode 7 배 느려진다) |
| `JOBNAME` | run.sh 의 폴더 이름 (`testbed` …) | submit | sbatch `--job-name` |
| `NODE` | `$CVPR_NODE` = `vll5` | submit | `--nodelist`. `any` 면 고정 안 함 |
| `PARTITION` | `$CVPR_PARTITION` = `batch_vll` | submit | |
| `CPUS_PER_GPU` / `MEM_PER_GPU` / `TIME` | `12` / `45G` / `0-08:00:00` | submit | `--cpus-per-gpu` / `--mem-per-gpu` / `--time` |
| `DEP` | — | submit | `--dependency=afterok:<id>` |

`env.sh` 의 `CVPR_*` (`CVPR_PY` `CVPR_LOCAL` `CVPR_DATA2` `CVPR_NFS` `CVPR_CKPT` `CVPR_DATA_CSV` `CVPR_CACHE` `CVPR_RESULTS` `CVPR_LOGS` `CVPR_NODE` …) 도 전부 `${VAR:-기본값}` 이라 바깥 export 가 이긴다.
옛 `BATCH_SIZE` / `DECODE_WORKERS` 는 없다 → `SET="surprise.batch_size=8 surprise.decode_workers=8"`.

## 5. 레지스트리 — 추가하는 법

- **규칙**: 경로는 반드시 `${CVPR_*}` (env.sh 변수). resolve 가 치환하고, env 에 없는 변수면 죽는다. 새 절대경로를 yaml 에 쓰지 않는다.
- `available: false` + `note:` → `--list` 에 `[사용 불가]`, resolve 는 note 를 내고 죽는다. 변환기가 실물 없는 항목에 자동으로 붙였다 (vll5 기준).
- ⚠️ `import_registry.py` 를 다시 돌리면 **yaml 을 통째로 덮는다**. 손으로 고친 항목은 그 파일의 `MANUAL_*` 에도 넣을 것 (yaml 머리 주석의 `_manual` 블록은 실제로는 없다).

**데이터셋** (`registry/datasets.yaml`) — `META` 키 (`engine raw_frames cache_tag legacy_results_root available note sliding_frame_skips overrides`) 를 뺀 나머지가 `data:` 블록으로 들어간다. 프로토콜의 `data:` 가 이긴다.
```yaml
my_set:
  engine: wma                      # wma | intphys2 | ek100 — 프로토콜의 engine 과 같아야 한다
  raw_frames: 100                  # 프레임 예산 검사 · sliding 의 n_frames: RAW
  cache_tag: my_set                # tag = <cache_tag>_<모델>[_w<창>]
  root: ${CVPR_DATA_CSV}/my_set    # index.csv 가 있는 곳
  index_csv: index.csv
  frames_root: ${CVPR_LOCAL}/world/world_analysis/MySet
  frames_pattern: '{file_name}/{frame:06d}.png'
  frames_start: 0
  frames_stride: 3
  block_column: block_id
  pair_column: pair_id
  variant_column: variant
  plausible_column: plausible
  type_column: condition
  legacy_results_root: ${CVPR_REPO}/z_research/MySet/exp_results   # RESULTS_ROOT=legacy 일 때
  sliding_frame_skips: [2, 5, 10]  # (sliding 창의 frame_skips: dataset 이 읽는다)
  overrides: {scoring.pairing: cross}   # 이 데이터셋이 요구하는 채점 옵션. 병합 뒤 · 사용자 SET 앞에 덮인다 (gravity_realistic 이 예)
```
**모델** (`registry/models.yaml`) — `base: vith` 로 vith 의 키를 깔고 다른 키만 적는다. 모델 고유 키 (`family checkpoint arch_name img_size patch_size tubelet_size context_encoder_key target_encoder_key predictor window_size uniform_power use_rope dual_encoder`) 는 **레지스트리가 프로토콜을 이긴다**. `img_size` 가 `data.resolution` (intphys2 는 `data.img_size`) 을 정한다.
```yaml
vith_my_predictor:
  base: vith
  predictor_checkpoint: ${CVPR_REPO}/z_training/runs/my_run/latest.pt
  predictor.kind: prefix           # predictor.<키> → model.predictor 블록 위에 덮인다 (kind · embed_dim · depth …)
my_family_model:
  family: dinof                    # analysis/model_loaders.py 의 family
  arch_name: dinof_vitb14
  checkpoint: ${CVPR_NFS}/…/x.ckpt
  img_size: 448
  patch_size: 14
  tubelet_size: 1
  surprise.target_layer_norm: false   # surprise.<키> → surprise 블록 위에 덮인다 (프로토콜보다 이긴다)
  surprise.batch_size: 8
```
**창** (`registry/windows.yaml`) — `kind` 필수, `extends:` 로 상속.
```yaml
c12t24: {kind: fixed, n_frames: 24, context: 12}            # stride/frames_start 생략 = 데이터셋 값
garrido_skip5_w32: {extends: garrido_a8, frame_skips: [5], window_sizes: [32]}
ip2_w24: {extends: ip2_w48, window_size: 24}
```

## 6. 창 (window)

| kind | preset | 채우는 키 |
|---|---|---|
| fixed (wma 단창) | `c16t32` (표준) · `c12t32` · `c8t32` · `c4t32` · `c16t24` · `c24t32` · `c8t16_s6` (VideoMAEv2) | `data.n_frames` · `surprise.context_length` / `features.context_length` · `model.window_size` (· `data.frames_stride`) |
| sliding (wma, Garrido A.8) | `garrido_a8` (전체 격자, intphys1 프로토콜 기본) · `garrido_w16` · `garrido_w32` · `garrido_skip2_w32` (88.89 칸) · `garrido_skip2_w16` (83.33 칸) · `dinof_w5` · `dinof_w8` · `dinof_w16` | `surprise.mode=intphys1` · `surprise.intphys1.{frame_skips(데이터셋별) window_sizes context_mult stride …}` · `data.n_frames=RAW` · `data.frames_stride=1` · `model.window_size=max(window_sizes)` |
| intphys2 | `ip2_w48` (기본) · `ip2_w32` · `ip2_w16` · `ip2_vmae_fps6/3/2` (창 16 고정, fps) | `surprise.window_size` · `context_length_sweep` (= 창 × {1/4 … 7/8}) · `context_length` (= 창/2) · `stride` · `data.target_fps` · `frame_step` · `model.window_size` |
| ek100 | `ek100_32f8fps` (기본, 논문) · `ek100_16f8fps` | `experiment.data.{frames_per_clip frames_per_second …}` · `wrapper_kwargs.num_output_frames` |

- **접미사 규칙**: 프로토콜 기본 창이면 이름에 아무것도 안 붙고, `WINDOW=` 로 바꾸거나 `SET="window.<키>="` 로 손대면 tag 와 output_dir 에 `_w<이름>` 이 붙는다.
  손댄 창은 이름이 자동이다: fixed `c{context}t{n_frames}[s{stride}]` (예 `SET="window.context=12"` → `_wc12t32`), sliding `skip2-5_w16-32` 꼴, intphys2 `ip2w{창}fps{fps}`.
  run.sh 의 `WINDOWS=` 는 창마다 `WINDOW=` 로 launch 를 부르므로 창마다 폴더가 갈린다 (`intphys1_dev_vith_wgarrido_w32`).
- **왜**: 토큰 캐시 서명 (`evals/analysis_vlm/occlusion_identity/cache.py:49 matches()`) 은 `video_ids · embed_dim · dtype · base_counts` 만 본다 — **문맥 길이 · 체크포인트 · 해상도는 서명에 없다.** 같은 `(데이터셋, 모델)` 이름으로 문맥 8 을 돌리면 문맥 16 캐시를 조용히 재사용한다. 그래서 창이 기본과 다르면 tag 를 갈라 캐시를 나눈다.
- 창(C+M) 마다 실행을 나누는 이유는 `eval/intphys1/README.md` (공식이 창마다 모델을 그 프레임 수로 짓는다).

## 7. 모델 고르기 (`python cvpr/harness/resolve.py --list` 의 모델 줄, 레지스트리 값으로 적었다)

| 묶음 | 이름 | 제약 (레지스트리 · 로더에서 읽음) |
|---|---|---|
| 릴리즈 | `vith` (기준 모델, CLAUDE.md §1-1) · `vitl` | — |
| 학습한 predictor (`base: vith` + `predictor_checkpoint`) | post-FT `vith_pft_v11_e10` · `vith_pft_intphys1_e40` · `vith_pft_intphys2_e80` · `vith_pft_predv1_e15` / Ariel `vith_ariel_prefix_ep45` (kind prefix) · `vith_ariel_full_ep40` (oneshot) · `vith_ariel_ar_ep18` · `_ar_ep36` · `_ar_ep38` (ar) · `vith_ariel_ctx_ar_ep18` (ctx_ar) | encoder 는 vith 그대로, predictor 만 바꾼다. ⚠️ `ar_ep18` 과 `ar_ep36` 은 **같은 파일** (`ariel/ar_future_1e_5/latest.pt`) 을 가리킨다 — 어느 epoch 인지 미검증. `vith_ariel_bc_ep*` 6 개는 `available: false` (옛 `_resolved.yaml` 해독용) |
| 다른 계열 (`family`) | `vjepa21g` (V-JEPA 2.1 ViT-g, 256) | `surprise.target_layer_norm: false`. 타깃 5632 차원이라 sliding `MB21` · intphys2 배치 12/8/6 |
| | `videomae2g` (224, patch 14) | `target_layer_norm: false`. 로더가 **224 고정** 을 assert. 창은 **16 고정** — intphys1 run.sh 는 창 32 에 `MBVM`, intphys2 run.sh 는 `ip2_vmae_fps*` 를 돈다, testbed 는 `WINDOW=c8t16_s6` (기본 c16t32 가 엔진에서 도는지는 미검증) |
| | `dinof_highres` (448) · `dinof_lowres` (224), tubelet 1 | `target_layer_norm: false` · `surprise.batch_size 8` · `decode_workers 8`. 문맥 **4 프레임 고정** (`model_loaders.py:232`): testbed 는 문맥 16 중 마지막 4 장으로 unroll, IntPhys1 은 `dinof_w5/w8/w16` 창을 run.sh 가 자동으로 돈다. 복사 기준선 `COPY=1` |
| jongseo 사전학습 (`$CVPR_OTHER_CKPT`, 읽기 전용) | `vittiny_k400` · `vittiny_k400_5k` · `vittiny_synphys_5k` · `vittiny_synphys_30k_e225` · `vittiny_k400ssv2_30k_e225` · `vitb_k400_e240` · `vitb_k400_e100` · `vitb_synphys_30k` | `uniform_power: true` · predictor 384/12/12 · `num_mask_tokens: 2` (레지스트리 값). 수치 해석은 `z_research/TrainingEffects/README.md` |

데이터셋 중 `available: false`: `v8` `v8_halfsize` `v10` `v10_flat` `v10_occ_low` `v13_black` `v11_occtiming` `2d_v8_transit` (vll5 에 실물 없음). engine 별: `intphys2_main` · `intphys2_main_nfs` (intphys2), `ek100` (ek100), 나머지 wma.

## 8. 결과 위치

```
$CVPR_RESULTS = cvpr/results/
  eval/testbed/<ds>_<model>[_w<창>][<suffix>][_smokeN]/          summary.json · per_block.json · _resolved.yaml · _meta.json · stdout.log
  eval/intphys1/<bench>_<model>_w<창>…/                          + per_window.json
  eval/intphys2/<ds>_<model>[_w<창>]/                            (엔진이 folder/tag 로 쓴다; summary.json — smoke_all.sh 의 기대 산출물)
  eval/ek100/action_anticipation_frozen/<ds>_<model>[_smoke]/   (엔진이 <folder>/<eval_name>/<tag> 로 쓴다; val_metrics.jsonl · log_r<rank>.csv · latest.pt — vll3 smoke 실물, eval.py:185/187)
  eval/testbed/copy/<ds>_<model>[_smokeN]/ · eval/intphys1/copy/<ds>_<model>[_smokeN|_valid6]/   복사 기준선 (copy_summary.json …, 각 README)
  analysis/knockout/<ds>_<model>[_w<창>][<suffix>][_smokeN]/   results.json · per_video.npz · shards/ · logs/
  analysis/readout/<step>__<ds>_<model>/                      _cmd.json · stdout.log (산출물 자체는 옛 자리 — analysis/readout/README.md)
  analysis/probing/<proto>__<ds>_<model>[_w<창>]/                summary.json · predictions.json
$CVPR_CACHE/<tag>/        토큰 캐시 (probing 만. surprise 프로토콜은 캐시를 안 만든다)
$CVPR_LOGS/<jobname>_<jobid>.out|.err   submit.sh 로그
```
프로토콜 yaml 이름이 `protocol.yaml` 이면 stem 을 생략하고, probing 처럼 이름이 있으면 `<stem>__` 이 앞에 붙는다.

**legacy 모드** (`RESULTS_ROOT=legacy`): `<datasets.legacy_results_root>/<legacy_name>__<ds>_<model>[_w<창>]` — 옛 폴더에 그대로 쓴다. 확인한 대응:
`eval/testbed v11 vith` → `z_research/IntPhysGenV11/exp_results/surprise_c16t32__v11_vith` (실물 있음) ·
`intphys1 WINDOWS=garrido_w32` → `z_research/Benchmarks/exp_results/intphys1_sliding__intphys1_dev_vith_w32` + TAG `intphys1_dev_vith_w32` (run.sh 가 옛 이름으로 바꿔 준다, 실물 있음) ·
`probing v11 vith` → `z_research/IntPhysGenV11/exp_results/attn_probe__v11_vith` (실물 있음) ·
`ek100` → `z_research/anticipation/EK100/exp_results/action_anticipation_frozen/ek100_vith` (그런 폴더는 없다 — 옛 run 은 `ek100_vith_official256_paper_grid8` 같은 TAG 였다) ·
`intphys2 WINDOWS=ip2_w16` → `…/intphys2/bench_intphys2_main_vith_w16` (2026-10-09 검증 때 `_w216` 이던 것을 run.sh 한 줄로 고쳤다 — DRYRUN 확인; `ip2_vmae_fps*` 는 옛 대응 이름이 없어 `_wvmae_fps6` 처럼 간다).
`cvpr/results/` · `cvpr/logs/` 는 `.gitignore` 에 있다 (`.gitignore:98-99`, `git check-ignore` 로 확인) — `_resolved.yaml` 도 추적되지 않는다.

## 9. 옛 명령 → 새 명령

| 옛 | 새 | 메모 |
|---|---|---|
| `GPUS=8 bash z_research/scripts/run.sh surprise_c16t32 v11 vith` | `GPUS=8 bash cvpr/eval/testbed/run.sh v11 vith` | config 동일 (check_equivalence) |
| `bash z_research/scripts/run.sh intphys1_sliding intphys1_dev vith` (A.8 격자 한 프로세스) | `bash cvpr/harness/launch.sh eval/intphys1 intphys1_dev vith` (같은 한 프로세스) / 창별 = `WINDOWS="garrido_w16 garrido_w32" bash cvpr/eval/intphys1/run.sh intphys1_dev vith` | run.sh 는 창마다 나눈다 (PROTOCOLS.md 규칙) |
| `bash z_research/scripts/run.sh attn_probe v11 vith` / `attn_probe_xfer v11_full` | `bash cvpr/analysis/probing/run.sh v11 vith` / `PROTO=attn_probe_xfer … v11_full` | |
| `bash z_research/scripts/run.sh --list` | `python cvpr/harness/resolve.py --list` | |
| `SET="…" TAG= OUTDIR= LIMIT= SMOKE=1 RECACHE=1 GPU_IDS=` | 그대로 | `BATCH_SIZE`/`DECODE_WORKERS` 만 `SET="surprise.batch_size= surprise.decode_workers="` 로 |
| `BENCHES= MODELS= WINDOWS="16 32" EXTRA_SET= TAG_SUFFIX= MB21= bash z_research/Benchmarks/run_all.sh` | `BENCHES= MODELS= WINDOWS="garrido_w16 garrido_w32" SET= SUFFIX= MB21= MBVM= bash cvpr/eval/intphys1/run.sh` (+ `RESULTS_ROOT=legacy` 로 옛 폴더·이름) | 옛 "summary.json 있으면 건너뜀" 과 `wait_gpu_free` 는 없다 — 다시 돌리면 덮는다. 최고 칸 고르기는 여전히 `z_research/scripts/analysis/garrido_rescore.py` |
| `MODELS= WINDOWS="16 32 48" VIDEO_BATCH=1 bash analysis/intphys2/run_grid.sh` | `WINDOWS="ip2_w16 ip2_w32 ip2_w48" bash cvpr/eval/intphys2/run.sh intphys2_main <model>` | `VIDEO_BATCH` → `SET="surprise.video_batch=N"` (기본 1 은 protocol.yaml). max_window_batch 표는 run.sh 가 같다 |
| `GPUS=8 [HEADS=sweep] [SMOKE=1] [VAL_ONLY=1] [CONFIG=ek100_vith] bash z_research/anticipation/EK100/run.sh` | `GPUS=8 [HEADS=…] [SMOKE=1] [VAL_ONLY=1] bash cvpr/eval/ek100/run.sh ek100 <model>` | `CONFIG=` 는 모델 레지스트리 이름으로 (`ek100 vith_ariel_prefix_ep45`). 옛 `data.*` 의 `experiment.` 생략은 안 된다 — 전체 경로로 |
| `GPUS=8 bash z_training/eval.sh <run> v11` | 등록된 run 은 `bash cvpr/eval/testbed/run.sh v11 vith_pft_v11_e10`. 임의 ckpt 는 `SET="model.predictor_checkpoint=<run>/<ckpt>" TAG=<ds>_<run>_<stem> OUTDIR=<run>/eval/surprise_c16t32__<ds>_<stem> bash cvpr/eval/testbed/run.sh v11 vith` | DRYRUN 확인. 옛 eval.sh 의 summary 요약 출력은 없다 |
| `bash analysis/attention/knockout/run_sharded.sh` (D= M= N= BPC= W= EDGES= SPECS=) | `N=8 bash cvpr/analysis/knockout/run.sh v11_full vith` (같은 노브 이름; `RESULTS_ROOT=legacy` 면 옛 폴더) | `analysis/knockout/README.md` 대응표 · `check.py` 가 run.py 가 보는 config == cvpr config 를 검사 |
| `python z_research/scripts/analysis/rollout2_*.py` · `rollout3_*.py` · `v11_readout_*.py` (자 분석 20여 개) | `STEP=<이름> bash cvpr/analysis/readout/run.sh` (`--list` 가 스텝 33 개) | `analysis/readout/README.md` 대응표 (35 줄) |
| `sbatch --export=ALL,WMA_RUN=1,P=…,D=…,M=…,GPUS=8 z_research/scripts/sbatch.sh` | `GPUS=8 [NODE= TIME= DEP=] bash cvpr/harness/submit.sh cvpr/<task>/run.sh <ds> <model>` | `WMA_RUN` · `SET_B64` 불필요 (`--export=ALL` 로 env 통째). 옛 `SPLIT`/`GSPLIT` (probing 을 target 별 job 으로) 은 없다 |
| `watch bash z_research/scripts/monitor.sh` | **안 옮김** — `squeue -u $USER` + `$CVPR_LOGS/<job>_<id>.out` | |

## 10. 아직 안 옮긴 것 · 알려진 제약

- **안 옮김**: predictor 학습 (`z_training/{train,sbatch,eval_matrix}.sh`) · 그림 (`z_research/scripts/figures/`) · `auto_research/scripts/` 의 일회성 chain 스크립트 · `z_research/scripts/monitor.sh` · `z_research/Benchmarks/monitor.sh` · A.8 최고 칸 선택 (`garrido_rescore.py`) · probing 의 target 별 SLURM 분할 (`merge_probe_runs.py` 흐름) · 복사 기준선의 CI (`te_analyze_scores.py` · `te_analyze_ip1.py`, 옛 폴더 경로 고정) · IntPhys 1 복사 기준선의 vith 이외 모델 · GRASP · InfLevel (`te_intphys1_score.py` 리터럴 — `eval/intphys1/README.md` 엔진 수정 제안). (knockout · readout · V-JEPA 복사 기준선은 2026-10-09 에 옮겼다 — §2.)
- **미검증 (이 세션)**: `submit.sh` 의 실제 sbatch 제출 (`bash -n` 만). 엔진이 끝까지 도는지는 **vll3 smoke** (`cvpr/logs/smoke_vll3*.log`, 2026-10-09 18:07 최종): testbed `v11_vith_smoke2` rc=0 (144 s, summary.json) · intphys1 `…_wgarrido_skip2_w32_smoke2` rc=0 (summary.json · per_window.json) · intphys2 `…_wip2_w16_smoke4` rc=0 (98 s, summary.json) · probing — 첫 시도 `LIMIT=4` + `auto_conditions` 는 rc=1 (한 조건으로 쏠려 5 조건 학습셋 0, `eval.py:268`) → `smoke_all.sh` 의 probing 줄을 `LIMIT=12` + `probing.fit_groups_sweep=[null]` 로 바꿔 재실행 rc=0 (65 s, `attn_probe__v11_vith_smoke12/summary.json`, `smoke_vll3_probing.log`) · ek100 `SMOKE=1` rc=0 (451 s, exact val 580 clip, `ek100_vith_smoke/val_metrics.jsonl` 실물; 18:05 요약의 "FAIL 산출물=없음" 은 그때 기대 파일명이 달라서였고 지금 `smoke_all.sh` 는 `val_metrics.jsonl` 을 본다). 다섯 경로 모두 한 번씩 끝까지 돌았다 — 수치 대조는 안 했다 (smoke 는 배관 점검).
- `launch.sh` 머리 주석의 `cvpr/eval/testbed` 철자와 `intphys2` legacy 이름 `_w216` 은 2026-10-09 검증 때 고쳤다 (§3 · §8). `SMOKE=1` 은 intphys2 에서 접미사 없이 본 폴더에 쓴다 (§4). `resolve.py --list` 의 프로토콜 줄에 `analysis/readout/config.yaml engine=?` 가 섞여 나온다 (glob `*/*/*.yaml` 이 readout 의 스텝 config 를 긁는다 — 프로토콜이 아니다, 무시).
- `attn_probe_imp` 는 `index_probe_imp.csv` 가 `v10` (available: false) 에만 있어 지금 돌릴 데이터셋이 없다.
- `import_registry.py` 재실행은 손 편집을 덮는다 (§5). 옛 `configs/protocols/*.md` 는 옛 run.sh 가 아직 읽으므로 지우지 않는다 — 두 레지스트리가 **갈라질 수 있다**: 새 데이터셋은 양쪽에 적거나, md 에 적고 `import_registry.py` 를 돌린다.
- 데이터는 노드 로컬이다: IntPhysGen·IntPhys1 프레임 `$CVPR_LOCAL`, IntPhys2·EK100·GRASP `$CVPR_DATA2` — 다른 노드면 resolve 의 실물 검사에서 죽는다 (`NODE=` 는 vll5 기본).
