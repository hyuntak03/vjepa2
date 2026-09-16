# CLAUDE.md — V-JEPA 2 직관물리 분석 레포 하네스

세션 시작 시 자동으로 읽힌다. **새 세션이 이전 세션과 같은 방식으로 일하도록** 하는 게 목적이다.
수치·경로·키 이름은 전부 실물에서 검증했다. 확실하지 않은 것은 "미검증" 이라고 적었다.

> # 시작점
> **`z_research/IntPhysGenV11/README.md` 를 먼저 읽는다.** 현재 논문·데이터·수치·다음 할 일이 거기 있다.
> 이 파일은 그 아래 깔린 **레포 운영 규칙**이다.

---

## 0. 30초 요약

V-JEPA 2 world model 이 **직관물리 위반**을 얼마나 잡아내는지, 그리고 **왜 특정 조건에서
실패하는지**를 규명한다. 채점은 하나뿐이다 —
**surprise = `mean |predictor(context) − LN(target_encoder(clip))[future]|`** (latent L1).
`surprise(impossible) > surprise(possible)` 이면 정답. chance 50%.

### 현재 논문 (정본: `z_research/IntPhysGenV11/Archive/PAPER_STORY_2026-08-31.md`)

> **Latent predictor 는 관측이 끊겨도 정보를 잃지 않는다.
> 잃는 것은 그 정보를 놓는 자리이고, 그 자리는 frozen 상태에서 되돌릴 수 있다.**

- **motivation** — latent prediction objective 만으로 intuitive physics 가 emerge 한다고
  보고됐고 [Garrido et al.] IntPhys 1 에서 88.89% 다. **그런데 IntPhys 2 [Bordes et al.]
  에서는 무너진다.** 두 벤치마크의 무엇이 다른지는 알려져 있지 않다 → 조작 가능한 testbed
- **정보는 남아 있다** — 등가속+가림에서 `p` probe 가 shape 98.46 / color 99.49 / env 100 인데
  같은 조건 채점 민감도는 4.46 이다
- **무너지는 건 배치다** — 비가림에서 배운 readout 이 가림에서 `p` 에만 안 통한다
  (color: `z` 97.4 / `h` 99.2 / **`p` 17.3**)
- **거리로 재면 고정된 순서가 나온다** — `cylinder 90 > … > torus 3`,
  42방향 중 40개 일치, Hodge R² 0.82–0.91 (기준선 0.286). 그 순서는 **가림이 만들지 않는다**
  (등가속 비가림에서 이미 R² 0.816)
- **개입은 frozen 위에서** — 정렬 사상(Procrustes) · 채점 층 개입. **재학습 없이 논문이 닫힌다**

⚠️ **2026-09-03 스토리 재검토 중.** 외부 피드백 "가림이 사건 순간에 걸리는 건 corner case" 에 동의.
무너지는 조건은 물체 × 사건 순간 × **고정 문맥 경계** 셋이 겹칠 때뿐이고 sliding 으로 상당 부분
회복된다 (§1-3). corner case 가 아닌 결과는 **점수가 무엇을 재는가** 쪽이다 (margin/base 0.08–3.7%,
비가림에서 이미 있는 물체 순서, precision 편향). 판정·후보·확장 축:
`z_research/IntPhysGenV11/Archive/STATUS_AND_CRITIQUE_2026-09-03.md`. **가림은 주인공이 아니라 testbed 로.**

⚠️ **목적함수를 원인으로 걸지 않는다.** discussion 한 문단으로만 쓴다.
재학습 없이 검증이 안 되는 원인을 축으로 두면 스스로 못 갚는 빚이 된다.

⚠️ **α 분해는 인용 금지다** (사용자 지시가 있을 때까지). 옛 문서 본문에는 남아 있다.

---

## 1. 절대 어기지 말 것 (관례와 그 이유)

### 1-1. 기준 모델은 **ViT-H**
IntPhys1 동일 프로토콜에서 ViT-H 88.89% vs ViT-L 64.44% (+24.4pt). **외부 벤치마크에서
먼저 정한 것**이다. 자체 데이터에서는 ViT-L 이 이기는 축이 있다 (v8 shape violation
ViT-L 83.98% vs ViT-H 73.83%; 2D transit ViT-L 75.39% vs ViT-H 71.48%; Jongseo physv3
ViT-L 92% → ViT-H 80%). **그래도 ViT-H 를 고수한다** — 자체 데이터로 사후 모델 선택을
하지 않는다는 원칙. "ViT-H 가 항상 낫다" 는 거짓이니 그렇게 쓰지 말 것.

### 1-2. 주지표는 **matched pairing** (`scoring.pairing: matched`)
block(4중항)의 가능×불가능 조합은 4개지만 **문맥일치 2쌍만 context 가 픽셀 단위로 같다.**
그래야 predictor 입력이 같아 `p` 가 **비트 단위로 동일**(실측 `max|p_pos − p_imp| = 0.0`)하고,
채점이 "p 가 h_pos 와 h_imp 중 어디에 가까운가" 라는 순수 기하 문제로 환원된다.
미래일치 쌍은 p 부터 달라 비교가 성립하지 않고 실제로 누수를 흡수한다.

> ⚠️ **`scoring.pairing` 기본값은 `"cross"` 다.** 안 쓰면 공식과 다른 전수 4쌍 채점이 조용히 돌아간다.

### 1-3. 프로토콜을 밝히지 않은 숫자는 비교 금지
| 비교 | 값 | 흔들림 |
|---|---|---:|
| IntPhys1: Garrido sliding vs 우리 fixed C16 | 88.89% vs 66.67% | **22.2pt** |
| v6aug: fixed / sliding avg / sliding max | 75.49 / 76.76 / 79.49 | 4.0pt |
| v6aug **static+occluded shape**: fixed vs sliding max | 46.88% vs **90.62%** | **43.7pt** |

sliding best 는 같은 그리드에서 고른 값이라 **held-out 추정치가 아니라 descriptive** 다.

### 1-4. probing 은 **가능(possible) 변이만**
불가능 변이를 넣으면 세 표현이 전부 오염된다:
- `contextF`(ctx_masked)는 미래 토큰을 transformer 이전에 떨궈 `imp_ab` 가 `pos_a` 와 비트 단위 동일
- `pred` 도 입력이 context 뿐이라 동일
- `targetF` 는 라벨이 `before` 인데 렌더는 `after` 라 **인코더가 정확할수록 0점**

예외는 `attn_probe_imp` 뿐이다 — 그건 "불가능 변이에서 target encoder 가 바뀐 정체성을
읽는가" 를 재는 **다른 프로토콜**이고 라벨 의미가 다르다.

### 1-5. split 은 **반드시 block 단위**
block 안 4개는 2×2 로 context/future 를 공유한다.
`split: {mode: ratio, group_by: block_id, train_frac: 0.5, seed: 0, stratify_by: <실험축>}`

### 1-6. 토큰 캐시는 **반드시 로컬 디스크**
`/data` 는 **NFS, 쓰기 57MB/s**. 캐시가 세트당 30~40GB(v11 은 420GiB)라 여기 두면 시간을 그냥 버린다.
`cache_dir: /local_datasets/world/world_analysis/cache` → `/data2`(xfs) 심볼릭 링크.

### 1-7. 고정 프로토콜 스펙 (`surprise_c16t32` / `attn_probe` 공통)
- `dual_encoder: true`, `context_encoder_key: encoder`(online), `target_encoder_key: target_encoder`(EMA)
- **surprise**: `dtype: float32` + `autocast: float16` (공식 Garrido 재현) /
  **probing**: `dtype: bfloat16` — 관례가 다르니 섞지 말 것
- raw 100프레임 stride 3 → 32장(`0,3,…,93`), context = 첫 16장(raw 0–45),
  future = 마지막 16장(raw 48–93), target encoder 는 **32장 전부**
- `target_layer_norm: true`(affine-free, target 에만), `distance: l1`, `loss_exp: 1.0`, `mask_index: 0`
  - `mask_index: 0` 은 critical — 릴리즈 체크포인트에서 학습된 mask token 은 `[0]` 하나뿐
- 가림은 **문맥 끝 k 장 + 미래 앞 k 장** 대칭. v11 은 `sym_k` 를 데이터셋 축으로 갖는다
  (0~4). **`n_context_hidden == k` 를 실측 확인했다**
- 프레임은 mp4 가 아니라 **원본 PNG 직독**. 리사이즈는 `antialias=False` bilinear — 공식과 같은 커널
- **A/B 방향별 정확도를 항상 병기.** scalar surprise 의 appearance bias 를 숨기지 않기 위함
  (v11 vanish 50% 는 100 과 0 의 평균이다)

---

## 2. 실행 방법

### 2-1. 표준 진입점 — `z_research/scripts/run.sh`

```bash
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
bash z_research/scripts/run.sh --list                                # 뭐가 있는지
GPUS=8 bash z_research/scripts/run.sh <프로토콜> <데이터셋> [모델]    # 모델 기본 vith
DRYRUN=1 bash z_research/scripts/run.sh attn_probe v11                # 병합 결과만, GPU 안 씀
```

**세 조각이 실행 직전에 합쳐진다** (`z_research/scripts/harness/resolve.py`):

| 조각 | 담는 것 | 파일 |
|---|---|---|
| 프로토콜 | 프레임 배치, 채점 규칙, probe 정의, dtype 관례 | `configs/protocols/<이름>.yaml` |
| 데이터셋 | 경로, 인덱스, 컬럼 이름 | `configs/protocols/datasets.md` |
| 모델 | 체크포인트, arch_name | `configs/protocols/models.md` |

**데이터를 바꿀 때 고치는 곳은 `datasets.md` 섹션 하나뿐이다.** yaml 은 건드리지 않는다.

병합 규칙 — **프로토콜이 이긴다**:
```
data  = { **datasets.md[<데이터셋>], **프로토콜.data  }
model = { **models.md[<모델>],       **프로토콜.model }
```
`n_frames: RAW` 는 그 데이터셋의 `raw_frames` 로 치환된다.

자동으로 정해지는 것 (`TAG=` / `OUTDIR=` 으로 덮을 수 있다):
```
tag        = <데이터셋.cache_tag>_<모델>                    예: v11_vith
output_dir = <데이터셋.results_root>/<프로토콜>__<데이터셋>_<모델>
```
`tag` 는 **토큰 캐시의 이름**이다. 프로토콜이 달라도 (데이터셋, 모델)이 같으면 같은 캐시를
쓰라고 일부러 프로토콜을 뺐다. 병합된 config 는 `<output_dir>/_resolved.yaml` 로 남는다.

`resolve.py` 가 **모델을 로드하기 전에** 경로·인덱스 실물, `resolution == img_size`,
프레임 예산을 검사하고 안 맞으면 죽는다. ViT-H 로딩은 프로세스당 ~2분이라 그 전에 죽는 게 훨씬 싸다.

### 2-2. 프로토콜 4종

| 이름 | 무엇을 재나 | 확립된 수치 |
|---|---|---|
| `surprise_c16t32` | fixed context16 / target32 latent-L1 | **73.37%** (v11, 10752 pair) · 79.10% (v8) |
| `intphys1_sliding` | IntPhys1 Garrido 공식 sliding | **88.89%** (intphys1_dev, 180 pair) |
| `attn_probe` | z / p / h 세 지점 attentive probing | v11 54 항목 (fit 3 × group 6 × target 3) |
| `attn_probe_imp` | 불가능 변이에서 target encoder 가 바뀐 정체성을 읽는가 | — |

등록된 데이터셋: `intphys1_dev`, `v8`, `v8_halfsize`, **`v11`**, `v11_earlymid`, `v11_timing`, `v11_full`, `v11_split_test`, `v13_black`,
`v10`, `v10_flat`, `v10_occ_low`, `jongseo_physv3`, **`rollout_v2`**, **`rollout_v2_training_v5`**, **`v11_vanish_all`** (위치 readout, §5-5)
(+ `available: false` 인 `2d_v8_transit`, `v11_occtiming`). 모델: `vith`, `vitl`. 지운 데이터셋 (RollOut_v1, 학습셋 v1~v4) 은 레지스트리에서도 뺐다 — 기록은 각 세트 문서.

⚠️ **v11 은 2026-09-01 에 43,008 clip 으로 커졌다** (`*_early`/`*_mid` 6조건 추가).
기존 6조건은 프레임이 그대로라 재채점하지 않는다. 새 팔만 담은 것이 `v11_earlymid`,
probing 용으로 `visible + early + mid` 를 묶은 것이 `v11_timing` 이다.
가림 타이밍 결과는 `z_research/IntPhysGenV11_occlusion_timing_ablation/`.

`results_root`: `intphys1_dev` → `z_research/IntPhys/exp_results`,
`v8`·`v8_halfsize` → `IntPhysGenV8`, `v10`류 → `IntPhysGenV10`, **`v11` → `IntPhysGenV11`**,
나머지는 기본값 `z_exp/world_model_analysis/results`.

### 2-3. 환경변수

| 변수 | 기본 | 동작 |
|---|---|---|
| `GPUS` | `1` | `--devices cuda:0..N-1`, `WMA_EXPECT_WS` |
| `TAG`, `OUTDIR` | 자동 | `tag` / `output_dir` 직접 지정 |
| `SET="a.b=1 c.d=null"` | — | 병합 config 를 점 경로로 덮어씀. **`null` = 키 삭제** |
| `LIMIT=N` | — | `limit: N` 주입 + `tag`/`output_dir` 에 `_smoke{N}` 접미사 |
| `SMOKE=1` | — | shape 디버그 |
| `RECACHE=1` | — | 토큰 캐시 무시하고 재추출 |
| `BATCH_SIZE`, `DECODE_WORKERS` | — | `surprise.*` 덮어씀 |
| `EVAL_DDP_PORT` | 자동 탐색 | 지정 시 탐색 생략 |
| `EVAL_DDP_TIMEOUT_S` | `7200` | NCCL timeout. torch 기본 600s 로는 rank 부하 불균형에서 죽는다 |
| `WMA_BAR` | `auto` | `on`/`off` |
| `DRYRUN=1` | — | 병합·검사만 하고 끝 |

### 2-4. 스모크

```bash
DRYRUN=1 bash z_research/scripts/run.sh surprise_c16t32 v11 vith      # 몇 초, GPU 0장
GPUS=1 SMOKE=1 LIMIT=2 BATCH_SIZE=8 bash z_research/scripts/run.sh surprise_c16t32 v11
```
⚠️ 스모크마다 프로세스가 각각 ViT-H 를 NFS 에서 새로 로드해 ~2분씩 쓴다.
`DRYRUN=1` 이 경로·컬럼·프레임 예산을 모델 없이 다 잡아 주므로 그걸 먼저 쓸 것.

### 2-5. SLURM — **`WMA_RUN=1` 을 반드시 붙인다**

`sbatch.sh` **한 파일이 두 역할**을 한다. 사람이 `bash` 로 실행하면 스스로를 제출하고,
SLURM 이 실행하면 본체를 돈다. **우리가 직접 붙이는 `WMA_RUN=1`** 로 갈린다.

```bash
sbatch --job-name=prb --export=ALL,WMA_RUN=1,P=attn_probe,D=v11,M=vith,GPUS=8 \
       z_research/scripts/sbatch.sh
watch -n 1 bash z_research/scripts/monitor.sh
```

- `-w vll5` 고정 — 데이터가 노드 로컬 `/local_datasets` 에 있다. `--gres` 와 `GPUS` 를 맞출 것
- ⚠️ **`WMA_RUN=1` 없이 직접 sbatch 하면 본체가 안 돌고 제출만 한다**
- ⚠️ **`--export` 는 콤마가 구분자다.** `SET` 에 YAML 리스트가 있으면 잘린다 → base64(`SET_B64`)로 싣는다
- v11 probing 63 head 는 8시간 벽을 넘는다 → `_prep` 이 캐시를 뽑고
  `(target × group half)` 6개가 `afterok` 로 붙는다.
  합치기: `python z_research/scripts/analysis/merge_probe_runs.py --base <output_dir>`

### 2-7. predictor 학습 — `z_training/` (2026-09-10 추가)

frozen encoder 위에서 predictor 만 학습한다 (scratch / 릴리즈 predictor post-FT).
**시작점은 `z_training/README.md`**, config 스키마는 `configs/training/README.md`.

```bash
GPUS=2 bash z_training/train.sh --smoke-ddp                                          # 런처만 점검 (모델·데이터 없음)
LIMIT=16 SET="data.datasets=[<이름>]" DEBUG=1 bash z_training/train.sh debug        # 배관 점검 (GPU 1장)
GPUS=8 SET="data.datasets=[<이름>]" bash z_training/train.sh frozen_predictor_scratch   # 새 predictor
GPUS=8 SET="data.datasets=[<이름>]" bash z_training/train.sh frozen_predictor_postft    # 릴리즈 predictor 이어 학습
GPUS=8 SET="data.datasets=[<이름>]" bash z_training/sbatch.sh frozen_predictor_scratch  # SLURM (TRAIN_RUN=1 로 갈린다)
GPUS=8 bash z_training/eval.sh <run> v11                      # 채점 = run.sh 에 model.predictor_checkpoint
```

- **결정 (2026-09-10): 릴리즈 predictor post-FT, 데이터 = v11 block split.** `v11_postft` config.
  train 절반(가능만) `v11_split_train` / test 절반(가능+불가능 10,752 쌍) `v11_split_test`
  (`z_training/data/build_v11_split_index.py`). 출발선은 기존 채점 per_block.json 에서 골라낸 **75.83%** (재실행 불필요,
  `z_training/runs/release_vith/eval/v11_split_test_from_existing_runs.json`).
  **v11/v11_full 전체 채점은 문맥을 공유하므로 학습 후 점수로 읽지 말 것**
- 손실·토큰 배치·역할 분리(context<-`encoder`, target<-`target_encoder`)가 `surprise_c16t32` 와 같다
- 체크포인트는 predictor 만 (`<run>/latest.pt`). 채점기는 `model.predictor_checkpoint` 로 그것만 바꿔 읽는다
- 기존 코드는 안 고쳤다. 런처는 `z_training/harness/launch.py` (app/main.py 무수정). 기존 파일 수정은
  `analysis/intphys2/model.py` 의 `predictor_checkpoint` 분기(새 키가 있을 때만)와 `resolve.py` 실물 검사 한 줄뿐

### 2-6. 구버전 진입점

`configs/world_model_analysis/` 는 2026-08-28 에 43개를 지웠다
(`occlusion_v2.yaml`, `probe_set.yaml`, `intphys1_single_vith.yaml` 만 남음).
**신규 실험은 전부 `z_research/scripts/run.sh` 를 쓴다.**
config 를 되살려야 하면 `z_exp/.../summary.json` 안에 그때 쓴 config 가 통째로 들어 있다.

⚠️ **`evals.main` 을 직접 부르지 말 것.** `run.sh` 만 하는 두 가지가 있다 — DDP 포트 자동
탐색과 `WMA_EXPECT_WS` export. 이게 없으면 §7-1 의 split-brain 가드가 통째로 비활성된다.

여전히 깨진 것: `z_world_model_analysis/` 의 `extract.py`·`spatial.py`·`metrics.py`·
`layers.py`·`mech.py` — `intphysgen_v1_vith.yaml` 을 찾는데 **v1 데이터셋 자체가 이미 없다.**

---

## 3. config 스키마 (world_model_analysis)

`eval_name` → `evals.<eval_name>.eval` 의 `main(args_eval, resume_preempt)` 를 부른다.

### 3-1. 블록별 유효 키

**최상위**: `eval_name`(필수), `tag`, `output_dir`(필수), `data`, `model`, `features`,
`surprise`, `scoring`, `probing`, `limit`, `recache`, `smoke`

**`data:`** `root`(필수), `n_frames`(40), `resolution`(None), `index_csv`("index.csv"),
`frames_root`, `frames_pattern`, `frames_start`(1), `frames_stride`(1), `block_column`,
`group_column`(→`variant_column`→"variant"), `plausible_column`("plausible"), `type_column`,
`pair_column`("pair_id"), `split`

**`model:`** `checkpoint`(필수), `arch_name`("vit_large"), `img_size`(256), `patch_size`(16),
`tubelet_size`(2), `window_size`(48), `use_rope`(true), `uniform_power`(false),
`dual_encoder`(false), `context_encoder_key`/`target_encoder_key`,
`predictor:{embed_dim 384, depth 12, num_heads 12, num_mask_tokens 10}`, `dtype`, `autocast`

**`features:`** (없으면 `surprise` 로 폴백) `cache_dir`(**필수, 없으면 KeyError**),
`context_length`(32), `mask_index`(0), `batch_size`(4), `cache_dtype`("float16")

**`surprise:`** `mode`("single"|"intphys1"), `distance`("l1"), `loss_exp`(1.0),
`target_layer_norm`(true), `mask_index`(0) / `mode: single`: `context_length`, `batch_size`,
`decode_workers` / `mode: intphys1` → `surprise.intphys1:` 서브블록

**`scoring:`** `pairing`("cross"), `breakdown`([]), `single_video`(false),
`token_subset:{modes, mode, object_radius}`

**`probing:`** `enabled`(false), `probe`, `optim`, `probes`(필수), `optims`(필수), `targets`,
`runs`(필수), `fit_groups_sweep`([None]), `block_types`, `split`, `fit_variants`, `eval_variants`

`runs[].train` / `.eval[]` 의 4축:

| `model` | `input` | encoder | source 이름 | offset |
|---|---|---|---|---|
| `context` | `full` | `ctx_masked` | `contextF__f{a}to{b}` | 0 |
| `context` | `window` | `isolated_ctx` | `contextW__f{a}to{b}` | f0 |
| `target` | `full` | `target` | `targetF__f{a}to{b}` | 0 |
| `target` | `window` | `isolated_target` | `targetW__f{a}to{b}` | f0 |
| `predictor` | (자동) | `predictor` | `pred__f{C+1}to{N}` | ctx_len |

- `frames` 는 **사람이 세는 1-idx 양끝 포함**. tubelet 정렬 필수
- `eval: [self, …]` — `self` 는 train source 자기 자신 (= 정보 존재 여부 = 이식의 상한)
- head 는 `(fit, groups)` 마다 **딱 한 번** 학습되고 eval 목록 전체에 frozen 적용
- `fit_groups_sweep: auto` — 데이터의 `group_column` 값을 읽어 sweep 자동 생성.
  `block_types` 에 해당하는 행이 없는 group 은 자동으로 빠진다

**probe 스펙**: `{type: attentive, num_heads: 16, num_probe_blocks: N}` 또는
`{type: linear, pooling: mean|max|meanmax, pre_norm: bool}`
**optim 프리셋**: `{num_epochs, batch_size, lr, weight_decay}` **4개만 소비된다.**
스케줄러 없음, 항상 AdamW 상수 lr.

### 3-2. 죽은 키 — 쓰지 말 것 (조용히 무시된다)

| 키 | 실체 |
|---|---|
| `report:` (`confusion`, `save_features`) | `occlusion_identity/eval.py` standalone 만 읽는다 |
| `scoring.metrics` | 아무 데서도 안 읽음 |
| `features.num_workers` | wma eval.py 는 안 읽음 |
| `data.eval_groups`, `probing.eval_groups` | 진짜 키는 `probing.eval_variants` |
| `data.strata` | `Rec.strata` 에 실리기만 하고 리포트 미출력 |
| `data.expect_per_block` | 안 읽음 |

⚠️ `fit_variants`/`eval_variants` 는 **`group_column` 값으로 거른다.**
`group_column: condition` 이면 `pos_a/pos_b` 필터를 못 쓴다 → index 에 파생 컬럼을 만들고
`type_column` + `probing.block_types` 로 거른다.

### 3-3. 출력물

```
<output_dir>/summary.json      # tag, world_size, n_videos, config 전체, surprise{...}, probing[...]
<output_dir>/per_block.json    # per_video_surprise{}, per_block[]  (surprise 실행 시)
<output_dir>/predictions.json  # probe 예측 (재학습 없이 confusion 을 다시 그릴 수 있다)
<output_dir>/_resolved.yaml    # 병합된 config
<features.cache_dir>/<tag>/    # meta.json + target.npy / ctx_masked.npy / predictor.npy …
```

`summary.json.probing[]` 항목:
```jsonc
{"fit": "pred__f17to32", "groups": ["moving_occlusion"], "target": "shape",
 "n_train": 389, "train_acc": 1.0, "chance": 0.143,
 "evals": {"<source>": {"overall": 0.98, "per_group": {"<group>": {"n":95,"acc":1.0,"bacc":1.0}}}}}
```

### 3-4. 실험 케이스마다 yaml 을 만들지 않는다

**새 yaml 의 기준은 하나다 — 재는 방식 자체가 다른가?** (다른 index, 다른 표현 지점,
다른 라벨 의미) 그 밖은 전부 아래 셋으로 처리한다:

| 장치 | 쓸 때 |
|---|---|
| `fit_groups_sweep: auto` | 데이터셋마다 조건 구성이 다를 때 |
| `SET="a.b=1 c.d=null"` | 일회성 변형 (epoch, target 하나만, sweep 지정). `null` = 키 삭제 |
| `extends:` | 구조가 진짜 다른 실험. 자식이 이기고 dict 재귀 병합, list 교체, `null` 삭제 |

실제로 `attn_probe_flat`(sweep 만 다름)·`attn_probe_e50_staticocc`(optim/runs/targets 만
다름)를 이렇게 없앴다. **프로토콜은 4개로 유지된다.** 자세한 건 `configs/protocols/README.md`.

---

## 4. 데이터셋과 디스크

### 4-1. 디스크 — **여유가 거의 없다. 쓰기 전에 반드시 `df -h`**

| 마운트 | 타입 | 용량 | 성격 |
|---|---|---|---|
| `/` | ext4 NVMe | 915G | **`/local_datasets` 가 여기 있다. 사실상 꽉 참** |
| `/data` | **NFS** | 91T | 레포·UnrealEngine 원본. **쓰기 57MB/s** |
| `/data2` | xfs 로컬 | 7.0T (여유 ~900G) | 토큰 캐시 전용 |

`/local_datasets/world/world_analysis/cache → /data2/local_datasets/world/world_analysis/cache`
⚠️ v11 토큰 캐시(`cache/v11_vith`)는 **421G** 다. 지우기 전에 무엇이 그걸 쓰는지 확인할 것.

### 4-2. index 스키마

**IntPhysGen 공통 14컬럼**: `video_id, file, file_name, block_id, source_block, variant,
plausible, pair_id, condition, motion, has_occlusion, violation_type, game_name, role`
- `file` 은 **비어 있다** — 프레임은 `frames_root` + `frames_pattern`
- `variant` ∈ pos_a / pos_b / imp_ab / imp_ba, `plausible` 정확히 반반
- 동반 `context_integrity.json`
- **v11 은 여기에 `sym_k` 가 추가된다** (0~4). `condition` 은 6개이고 k 를 접지 않았다

**`index_probe.csv`** (probing 전용) = 위 + 라벨 컬럼 + **`probe_type`**
`probe_type` ∈ `obj`(가능+물체 있음 = **유일한 probing 대상**) / `empty` / `imp`

**Jongseo physv3** 만 `file` 이 실제 mp4 절대경로다 — 유일하게 PNG 가 아니라 영상 직독.

### 4-3. 원본 프레임 위치

**등록된 것은 `configs/protocols/datasets.md` 가 정본이다.** 그 밖의 실물 위치:

- **IntPhysGen v11**: `/local_datasets/world/world_analysis/IntPhysGen_v11`
  설계 문서 `/data/.../UnrealEngine/gen/V11_DESIGN.md`
- IntPhysGen v10: `/local_datasets/world/world_analysis/IntPhysGen_v10` (42G)
- IntPhysGen v1~v9: `/data/hyuntak/project/2026/2027_cvpr/UnrealEngine/IntPhysGen_v{N}/` (NFS)
- IntPhys1 dev PNG: `/local_datasets/world/world_analysis/IntPhys1_dev_frame_png/` (3.3G)
- ⚠️ **2D transit 프레임(195M)은 2026-08-28 에 사라졌고 생성 코드도 함께 없어졌다.**
  토큰 캐시(41G)는 남아 있어 **캐시 기반 사후 분석만 가능**하다
- v6 / v6_augmented / v7 은 레지스트리에서 뺐다. 수치는 `z_exp/.../summary.json` 에서 재검증 가능

---

## 5. 확립된 결과

### 5-1. v11 채점 — `surprise_c16t32`, matched pair, chance 50%

**overall 73.37%** (n=10752, 45 cells, `report.json.verified = true`)

| condition | vanish | shape | color |
|---|---:|---:|---:|
| `static_visible` | 100.0 | 94.0 | 81.8 |
| `static_occlusion` | 70.5 | 54.8 | 55.4 |
| `moving_visible_flat` | 100.0 | 89.9 | 86.0 |
| `moving_occlusion_flat` | 52.0 | 56.8 | 68.5 |
| `moving_visible` (ramp) | 100.0 | **69.0** | 76.8 |
| `moving_occlusion` (ramp) | 50.0 | 54.5 | 71.4 |

- **가림이 전부를 무너뜨린다** — `visible` 행은 전부 높고 `occlusion` 행은 전부 바닥
- **등가속은 가림 없이도 이미 낮다** (shape 94.0 / 89.9 → **69.0**). 물리를 안다면 반대여야 한다
- **vanish 50% 는 100 과 0 의 평균이다** — `빈→물체` 100.0 / `물체→빈` 0.0
- **`k=0` 과 `k>0` 사이에서만 끊기고 `k=1…4` 는 평평하다.** 가려진 시간은 무관하다

### 5-2. v11 probing (`attn_probe`, 54 항목)

등가속+가림에서 `p` probe: shape **98.46** / color **99.49** / env **100**.
같은 조건 채점 민감도 shape **4.46**.
- `env` 는 self·이식·k 분해 **72칸 전부 100.0** — 표현이 통째로 망가진 게 아니다
- 비가림→가림 이식 (color): `z` 97.4 / `h` 99.2 / **`p` 17.3**
- ❌ **shape 은 인코더도 흔들린다** (`z` 3팔 평균 64.8, 등속 팔 42.4).
  "인코더는 조건과 무관" 은 **color·env 로만** 주장할 수 있다

전수는 `z_research/IntPhysGenV11/Archive/surprising_score/RESULTS_2026-08-30.md`.

### 5-3. v11 방향 비대칭 — 고정된 순서

등가속+가림 shape 지킴 순:
```
cylinder 90 > cube 86 > pyramid 59 > capsule 58 > cone 48 > sphere 36 > torus 3
```
- 21쌍 × 2방향 = **42방향 중 40개**가 이 순서 하나와 일치
- Hodge 적합 **R² 0.82–0.91** (무작위 기준선 6/21 = 0.286)
- **순서는 가림이 만들지 않는다** — 등가속·비가림에서 이미 R² **0.816**.
  정지·등속 비가림은 방향 편향 2.7 / 4.2 로 **잡음바닥 7.1 아래**
- ⚠️ **"특정 물체로 몰린다" 는 성립하지 않는다** — 최다 목적지 열의 비대각 질량이 21~40%
  (균등 16.7%). 정확히는 **순서 위쪽으로 흘러내린다**이지 "cylinder 를 만든다" 가 아니다
- **precision 이 편향임을 확정한다** (등가속+가림): recall 폭 **86** 인데 precision 폭 **24**.
  `cylinder` R 89.6 / **P 51.5**(chance 50) · `torus` R 3.1 / P 75.0.
  **능력 차이가 아니라 과생산/과소생산이다.** 정지·비가림 대조군은 recall 폭 6 / precision 폭 12
- 가장 날카로운 대비: **probe 는 `torus` 를 100% 로 읽는데 채점은 3% 만 지켜낸다**

### 5-4. 이전 세트 (아카이브)

| 데이터셋 | 모델 | 프로토콜 | overall | n_pair |
|---|---|---|---:|---:|
| **IntPhys1 dev** | ViT-H | Garrido `skip2_w32/avg` | **88.89%** | 180 |
| IntPhys1 dev | ViT-L | 동일 | 64.44% | 180 |
| IntPhys1 dev | ViT-H | 우리 fixed C16/stride3 | 66.67% | 180 |
| IntPhysGen v8 | ViT-H | fixed | 79.10% | 1024 |
| IntPhysGen v8 | ViT-L | fixed | 73.44% | 1024 |
| 2D transit | ViT-H / ViT-L | fixed | 71.48 / 75.39% | 1024 |
| Jongseo physv3 | ViT-L → ViT-H | C16/P16 | 92.0% → 80.0% | 50 |

**IntPhys1 을 운동 × 가림으로 쪼개면** (sliding): 정지 100 / 100, 이동(포물선) 91.67 / **75.0**.
**세 셀이 천장이고 한 셀만 떨어진다** — 논문 beat 1 의 다리다.

v8 의 정보손실/정렬손실 분해, 2D 대조는 `z_research/IntPhysGenV8/`.

⚠️ **α 분해 관련 수치는 인용 금지다** (지시 전까지). `z_world_model_analysis/PREDICTOR_HEDGING_*`,
`MECHANISM_*` 및 옛 문서의 α·보정 값 전부 해당.

### 5-5. 위치 readout — RollOut_v2 학습셋 v5 자 (2026-09-12) · `z_research/RollOutV2/`

**정본**: `RollOutV2/figures/v5/summary/POSITION_READOUT_2026-09-12.md` (수치·정정·재현 전부). 시작점 `RollOutV2/README.md`.
별도 학습셋 (무중력 등속 8,064 clip, `visible_by_sample` 마스크) 에서 attentive 위치 자 (3,842 파라미터) 를 p / z / h 각각에 정하고
`RollOut_v2` (7 시나리오, 가능 4,704 clip) 와 v11 가림 셀 (2,688 clip) 에 test 로만 건다. 1 칸 = 18 px = 물체 반폭.

| 무엇 | 수치 |
|---|---|
| 자 검증 | held-out 0.87 칸 (학습 0.83); encoder 자 v2 전 시나리오 ≤ 0.61 칸, v11 가림 타이밍 무관 추적; 자 없는 검사 (r = \|p−h_imp\|/\|p−h_pos\|, 파라미터 0) 와 슬롯 단위 일치 |
| p, v2 | flat_v **0.71 칸** gain 0.85 / flat_a 1.07 · 0.57 / ramp·arc·fall 0.9~1.4 칸 · gain 0.4~0.5 / ledge 2.06 / wall 1.78 |
| ledge | p 는 선반 높이 유지 (슬롯 3~4 부유 궤적 attention 0.43 / 0.39, 기본값은 낙하 쪽), 392 쌍 100 %; pos/imp probe P(imp) 0.91~0.96 |
| wall | p 는 **2 슬롯 통과 뒤 물체를 잃는다** (통과 질량 0.61 / 0.48 → ≤ 0.18). "벽 속 +29 px 에서 멈춤" 은 **정정** (기본값). probe P(imp) 0.70~0.90 |
| v11 | 물체다운 토큰이 남는 길이: late **0** 슬롯 < early/mid **3** < 가림막 없음 **4~5**. ramp late 만 k 에 따라 악화 (ratio 0.21→0.04). static 도 late 만 1 칸 넘게 이탈 |
| 채점 vs 위치 | vanish 채점 early 97.8 / mid 95.8 / late 57.5 인데 위치는 셋 다 3 슬롯 안 소실 — **별개 축** |

⚠️ **읽는 규칙** — 자는 물체가 없어도 위치를 낸다 (attention 이 퍼지면 토큰 평균 읽기 = 기본값). v11 은 기본값 ≈ 마지막 관측 위치, wall 은 ≈ 정지점.
**위치 주장은 그 슬롯의 3×3 attention 질량이 균등 (0.035) 을 넘을 때만** 하고, 8 슬롯 평균 지표는 슬롯별 표 (`attn_diag.md`, `two_futures_attn.md`) 와 같이 읽는다.
"predictor 가 물체를 마지막 자리에 둔다" 는 **철회** (기본값이었다). 맞는 문장: "가림이 경계에 걸리면 p 의 미래에는 처음부터 물체다운 토큰이 없다."

---

## 6. 이미 기각된 가설 — **다시 시도하지 말 것**

| 가설 | 기각 근거 |
|---|---|
| context 길이가 문제 | C=4/8/12/16 전부 50.9~54.3% |
| 희석 — 물체 토큰만 보면 됨 | `token_subset: object` 25~39배 좁혀도 **더 나빠짐** |
| 채점 프로토콜(sliding)이 원인 | IntPhys 는 고정 단창 하나로도 88.33% |
| 배경 복잡도 / 카메라 시점 | env 8종, 탑다운 vs 원근 전부 무차이 |
| predictor 내부가 정보를 잃음 | 층별 probe 68→86% 단조 증가, proj 후 88.7% |
| **"p 에 정보가 없다"** | v11 등가속+가림에서 shape 98.46 / color 99.49 / env 100 — **철회** |
| **"물체를 엉뚱한 자리에 그린다"** | Mh 가 물체 위치에 안 몰림(상위1% = 1.9%) — **철회** |
| **"IntPhys 물체가 2배 크다"** | 개수와 크기를 섞어 잰 오류 — **철회** |
| **"우리 신호가 작아서"** | v8 RGB MAE 1.484–2.079 vs Jongseo 1.639 로 **오히려 더 큼** |
| 궤적·속도 위반 추가 | **IntPhys1 dev 에 그런 사례가 0건** |
| **"방향 비대칭은 p 가 그 모양을 못 담아서"** | 설계 쌍 probe confusion 0%. **probe 가 잘 읽는 쪽이 채점은 0%** |
| **"세기를 키우면 된다"** | 균등 증폭 천장 51~65%, 잘 맞던 조건까지 부숨(98.4→59.4). **부족한 건 크기가 아니라 배치다** |
| **block 밖 7-way retrieval** | 같은 block 다른 모양 0.5438 / 다른 block 같은 모양 0.7602 → 신호/잡음 **0.715**. `h` 대조군조차 40.3%(chance 14.3). `analysis/retrieval_confusion.py` 최상단 |
| **Luce 로 7지선다 확률 변환** | `s=(1-p)/p` 가 **최저 상대 하나에 지배**된다. **등가속·비가림**에서 cube 짝평균 83.3 → 7지선다 8.7 (cylinder 에게만 6%). `k` 별은 비대각 n=4 라 더 나쁘다. `_superseded/` |
| **"가림막이 색면이라 색 채점을 죽인다"** | `v13_black`(검정 가림막)에서 색의 `visible→early` 하락이 −14.1 → **−13.5** 로 같다. **기각.** 대신 모양이 −6.2 → **0.0** — 나뭇결이 모양을 깎고 있었다. `z_research/IntPhysGenV13_Black_occluder/` |
| **"가림막은 있는데 물체를 안 가리는" 조건 신설** | "가림 유무" 는 곧 "물체가 가려지는가" 이고 가림막 존재는 그 구현이다. §8-5 |
| **"predictor 가 (v11 가림에서) 물체를 마지막 관측 자리에 둔 채 머문다"** | 자의 기본값 (토큰 평균 읽기 ≈ 화면 중심 ≈ 마지막 관측). 마지막 관측 칸 attention 0.01~0.03. **철회** (§5-5) |
| **"wall 에서 벽 속 한 칸 반 파고들어 멈춘다"** | 슬롯 1~2 통과 + 슬롯 3~7 기본값의 평균. 멈춤은 한 번도 없다. **정정** |
| v11 k=0 p 만으로 위치 자 학습 | 자리 prior 를 외움 (오차 3 px, 물체 attention 0.1; early 슬롯 0~2 에서 47~63 px). 위치를 고루 덮는 학습셋 필요 |
| 위치 자 학습셋에 kink·고속 (v3) | 자가 학습 분포 안에서도 흐려짐. 삭제, 재시도 금지 |
| 슬롯 7 진행 비율 하나로 v11 요약 | 물체가 사라진 뒤의 기본값을 섞어 조건 차이 (early/mid 슬롯 0~2 는 정확) 를 지운다. 슬롯별 질량과 함께만 |

---

## 7. 함정과 검증 습관

### 7-1. 인프라
- **sbatch 자기제출 폭주** — `SLURM_BATCH_SCRIPT` 가 batch job 에서 비어 있어 모든 job 이
  제출 분기를 다시 탔다. 세대마다 ×7 로 **9,924 개**. **`WMA_RUN=1`** 로 갈리고,
  `SLURM_JOB_ID` + `OUTDIR` 이 함께 있으면 거부하는 tripwire 가 있다.
  **SLURM 이 무엇을 넣어 주는지에 기대지 말 것. 표시는 우리가 붙인다**
- **`--export` 는 콤마가 구분자다** — YAML 리스트가 잘려 `'[capsule'` 로 도착했다. base64 로 싣는다
- **exit 0 on failure** — `mp.Process(...).start()` 후 join 을 안 하면 rank 가 죽어도 0 을
  반환하고 `afterok` 가 풀린다. `evals/main.py` 가 exitcode 를 확인한다
- **prep 과소추출** — `probing.runs` 를 줄이면 캐시에 뽑히는 표현도 줄어든다.
  **runs 는 두고 `num_epochs` 만 1 로** 한다
- **DDP split-brain** — `init_distributed` 가 bind 실패를 삼키고 조용히 `world_size=1` 로
  폴백해 두 job 이 섞인다. `run.sh` 가 포트 자동 탐색 + `WMA_EXPECT_WS` 가드.
  **`evals.main` 을 직접 부르면 가드가 통째로 비활성**
- **캐시는 `video_ids` 완전 일치를 요구한다** (`occlusion_identity/cache.py:57`).
  **상위집합·부분집합 재사용이 안 된다** — 조건을 하나 더한 데이터셋은 전부 다시 뽑는다.
  묶어서 뽑을지(이식이 가능해짐) 나눠서 뽑을지를 **인덱스를 만들 때** 정해야 한다
- **토큰 캐시가 조용히 재사용된다** — `TokenCache.matches()` 는 `video_ids`/토큰수/dtype 만 본다.
  **체크포인트·해상도·autocast·재렌더는 서명에 없다.** 데이터를 다시 만들면 캐시도 반드시 치울 것
  (보관 관례 `cache/_previous_YYYYMMDD/<tag>`)
- **`features:` 블록 누락** → `cfg["surprise"]` 로 폴백 → `cache_dir` KeyError.
  **surprise 채점을 다 끝낸 뒤에 죽는다** (가장 아까운 실패)
- **`data.resolution != model.img_size`** → 작으면 CUDA assert, 크면 **에러 없이 엉뚱한 구간을 잘라 쓴다**
- **CPU 스레드 과다구독** — rank 마다 전체 코어를 잡으면 decode 가 39ms → 268ms
- **probing `batch_size` 는 global batch** (DDP 가 rank 당 1/ws 로 나눔). surprise 의 것은 rank 당

### 7-2. 수치
- FP16 batching 은 근접 tie 를 뒤집을 수 있다. 재현성 원하면 `video_batch: 1`
- **DDP probing 은 실행마다 흔들린다.** 결정론이 필요하면 `GPUS=1`
- 결정론 확인됨: 같은 config 재실행 전 자리 일치, token_subset `all` 은 `max|Δ| = 1.2e-07`

### 7-3. 통계 해석
- **CI 폭 0 인 셀은 버그가 아니다** — 100%/0% 상쇄 구조
- **`(조건, k)` 로 쪼개면 비대각이 방향당 n=4 다.** 개별 칸을 읽지 말 것.
  그 칸들의 함수(행 평균, Luce 확률)도 같은 잡음을 물려받는다
- **probe head 가 과대용량**(49.3M) → contextF/targetF 는 `train_acc=1.000` 으로 포화.
  **천장에 안 닿은 건 이식 줄뿐이므로 대조는 이식 줄에 한해 읽을 것**
- **팔을 가로지르는 이식 실패는 증거가 아니다** — 기대값이다 (attention query 가 위치 특이적)
- **교란: 가림막 존재 자체** — occluded 조건에만 가림막이 장면에 있다. §8-5 를 볼 것
- **학습된 readout(자) 은 대상이 없어도 값을 낸다** — attentive 자는 attention 이 퍼지면 토큰 평균 읽기 (기본값) 를 내고, 그 기본값이 데이터의 의미 있는 자리 (v11 마지막 관측, wall 정지점) 와 겹치면 하루를 잘못 읽는다 (2026-09-12). 자로 읽은 값에는 **attention 질량·기본값 거리·파라미터 없는 교차검증** 을 붙인다 (§5-5)

### 7-4. 필수 검증 루틴
- **byte-identical context 감사** — 모든 matched pair 의 16 context 프레임이 픽셀 단위로
  같은지 전수 검사. v6~v8·2D·v10·v11 전부 mismatch 0. 산출물 `data_csv/<dataset>/context_integrity.json`
- **`|z(p)|` 단독 채점이 정확히 0.00%(전 쌍 tie)** 로 나오는 것이 파이프라인 정합성 증거
- 수치는 문서가 아니라 **`summary.json`/`report.json` 에서 다시 계산해 검증**하고,
  그렇게 했다고 문서에 명시할 것. `plot_v11_*.py` 는 실행마다 자동으로 대조한다

---

## 8. 문서·그림 작성 관례

### 8-1. 디렉토리 역할

| 위치 | 역할 | 추적 |
|---|---|---|
| `configs/protocols/` | 프로토콜 yaml + `datasets.md`/`models.md` 레지스트리 | ✅ |
| `z_research/scripts/` | 최상위엔 **직접 치는 것만**. 나머지는 `harness/`·`data/`·`figures/`·`analysis/` | ✅ |
| **`z_research/IntPhysGenV11/`** | **본 실험 세트.** `README.md` 가 시작점 | 부분 |
| `z_research/anticipation/EK100/` | V-JEPA 2 EK100 action anticipation 재현 (논문 §6). `README.md` 가 시작점 — 릴리즈 코드와 논문이 다른 곳과 우리 기본값 | 부분 |
| **`z_research/RollOutV2/`** | **위치 readout 세트** (p 가 물체를 어디에 두나). `README.md` → `figures/v5/summary/POSITION_READOUT_2026-09-12.md` | 부분 |
| `z_research/IntPhysGen{V8,V10}/`, `IntPhys/` | 아카이브 | 부분 |
| `z_research/<셋>/Archive/*.md` | 분석 문서. **파일명에 날짜** `TOPIC_YYYY-MM-DD.md` | ✅ |
| `z_research/<셋>/exp_results/` | 원시 산출물 (`summary.json`, `_resolved.yaml`) | ❌ |
| `z_research/<셋>/figures/` | 그림 | ❌ |
| `z_world_model_analysis/` | 레포 전역 분석 | ✅ |
| `z_exp/world_model_analysis/results/` | `results_root` 없는 데이터셋 + 과거 실행 42개 | ❌ |

> ⚠️ 2026-08-28 에 `z_research` 를 정리하면서 `IntPhys-Like_data/` 아래 리포트·figure 가
> 전부 사라졌다. **git 추적이 0개였기 때문에 복구가 불가능했다.**
> 그래서 새 스크립트·config·문서는 추적되는 자리에 두고 커밋한다.

### 8-2. 새 파일을 만들 때

⚠️ **레포 루트나 폴더 최상위에 파일을 흘리지 않는다.**

| 종류 | 어디 |
|---|---|
| 스크립트 | `z_research/scripts/{harness,data,figures,analysis}/` 중 하나. 맞는 폴더가 없으면 만들고 `scripts/README.md` 에 한 줄 추가 |
| 임시 출력 | `/tmp` 나 스크래치패드. 레포에 쓰지 않는다 |
| 분석 문서 | `z_research/<셋>/Archive/TOPIC_YYYY-MM-DD.md` |
| 프로토콜 | **§3-4 를 먼저 볼 것.** 대부분 새 yaml 이 필요 없다 |

### 8-3. 문서 규칙
- 하단에 반드시 **`## 재현`** 절: 정확한 명령줄 + config 경로
- **철회/정정을 지우지 않고 문서 안에 남긴다.** `⚠️ 정정 —`, `**철회.**`,
  `## 폐기한 가설들 (기록)`. "다시 시도하지 말 것" 까지 적는다
- 대체된 문서는 지우지 않고 **맨 위에 blockquote 로 "대체됨 + 정본 경로"** 를 붙인다
- 한 문서 안에서 언어를 통일 (심층 분석·종합 노트는 한국어)
- **단서(caveat)를 결론과 같은 비중으로 쓴다**

### 8-4. 그림
- **한글 폰트가 없다. 플롯 라벨은 반드시 영어.** 주석은 한글 OK
- 탐색용 `DejaVu Sans`. **논문용 `Nimbus Roman`** + `pdf.fonttype 42` + 벡터 PDF, double-column 7.0in
- 패널 라벨 `(a)`–`(d)` 를 **각 패널 아래 가운데**
- **서술 문장은 그림에 넣지 않는다 — 캡션으로 뺀다**
- 색은 검증된 팔레트 slot 1–3 만: `#2a78d6` / `#eb6834` / `#1baf7a`.
  aqua 는 light surface 에서 3:1 미만이라 쓰면 **모든 마크에 직접 라벨** 필수
- **그림은 폴더가 정해져 있다.** `plot_v11_surprise.py` 상단 `FIGDIR` 표가 배정한다.
  표에 없는 이름은 최상위에 떨어진다 (새 그림이 눈에 띄라고 일부러)
- 대체된 그림은 지우지 않고 `_superseded/` 로 내리고 **README 에 왜 물러났는지 적는다**

### 8-5. 설계 결정 — 다시 교란으로 제기하지 말 것

두 대비는 **의도된 조작**이다. 여러 번 "교란" 으로 제기했다가 매번 같은 답을 받았다.

| 축 | 무엇을 비교하는가 | 함께 바뀌는 것 |
|---|---|---|
| **가림 유무** (`visible` vs `occlusion`) | **물체가 가려지는가** | 가림막이 장면에 있는가 |
| **지면** (`moving_flat` vs `moving`) | **등속 vs 등가속도** | 쐐기가 장면에 있는가 |

물체를 가리려면 가림막이 있어야 하고, 중력으로 가속시키려면 경사면이 있어야 한다.
조작의 구현이지 교란이 아니다. v11 은 **가림막 폭이 k 와 함께 변하는 것**도 같은 이유다.

- ✅ "물체가 가려지면 무너진다. 가려지는 **시간**은 무관하다" (k=1~4 평평)
- ✅ "등가속이 등속보다 나쁘다"
- ❌ "가림막이 화면에 있는 것 자체는 영향이 없다" — 안 쟀다
- ❌ "가속만이 원인이고 장면 구성은 무관하다" — 안 쟀다

---

## 9. 환경

```
python : /data/hyuntak/anaconda3/envs/vjepa2/bin/python   (3.12.13)
torch 2.12.0+cu126 · GPU 8장 · numpy 2.4.6 · matplotlib 3.11.0 · decord 0.6.0 · timm 1.0.27
없는 것: scikit-learn, seaborn, PyAV, node/npm, 한글 폰트
```
SLURM 스크립트는 `source /data/hyuntak/anaconda3/bin/activate vjepa2`.

### `.gitignore` 주의
`z_scripts`, `*csv`, `*.json`, `*.png`, `*.pt`, `z_exp/`, `data_gen/`, `checkpoint/` 가
**전부 무시된다.** → 새로 만든 index·figure 는 **untracked 가 정상**이고 `git checkout` 으로
되돌릴 수 없다. 덮어쓰기 전에 확인할 것.

⚠️ **파일 편집에 `sed -i` 를 쓰지 말 것.** 2026-08-28 에 `datasets.md` 에서 한 줄을 바꾸려던
`sed` 가 섹션 4개를 통째로 지웠다. 여러 줄짜리 md/yaml 은 Python 으로 고친다.

---

## 10. 다음에 할 것 (beat 기준)

**정본은 `z_research/IntPhysGenV11/Archive/PAPER_STORY_2026-08-31.md` 의 실험 순서다** — 단
2026-09-03 부터 **`STATUS_AND_CRITIQUE_2026-09-03.md` §7 이 앞선다**: IntPhys 2 실측 → 경계 위치 축
(sliding 재현) → 외형 편향의 비가림 정량화 → 선행 연구 → 그 다음 정렬 개입.

| 순 | 실험 | beat | 비용 |
|---|---|---|---|
| **1** | **부분공간 겹침 / Procrustes** (`p`·`z`·`h`, 조건별) | **5** | 캐시만, SVD 두 번 |
| 2 | ~~`p` 문맥 잔상 검정 (위치 디코딩)~~ **✅ 2026-09-12 완료** — `RollOutV2/` §5-5. 가림이 경계에 걸리면 p 미래에 물체 토큰 없음, ledge 부유 / wall 2 슬롯 통과 | 3 | 캐시만 |
| 3 | IntPhys 2 의 V-JEPA 2 실측치 확인 | 1 | 문헌 |
| 4 | 정렬 사상 `W: p → h` (Procrustes, 가능 변이로만) | 6·7 | 소량 |
| 5 | 채점 층 개입 (`distance: pooled_l2` 구현 · 순서 보정) | 6·7 | 구현 |
| 6 | 개입 후 순서 재측정 (방향 편향 · Hodge R²) | 7 | 캐시만 |
| 7 | ViT-L · v11 변종 · IntPhys 1·2 재분해 | 8 | 추출 포함 |

**1 번이 논문의 새 중심이고 제일 싸다.** 결과가 어느 쪽이든 beat 가 선다 —
겹침이 작으면 "배치가 바뀐다" 확정, 크면 개입이 채점 층으로 좁혀진다.

그 밖: `attn_100` 으로 미수렴 head 재학습 / linear probe 대조 / 역방향 이식(p→h) /
축별 풀링(현행 토큰평균은 시간축·공간축을 둘 다 버려 위치·궤적 위반을 원리적으로 못 잡는다).

---

## 11. 데이터를 새로 설계할 때

- **위반이 "관계" 로만 존재하면 안 된다.** splice(정상 A 의 앞 + 정상 B 의 뒤)는 두 반쪽이
  각각 자연스러워 두 후보가 구분되지 않는다
- **"물체 하나 = 화면 움직임의 전부" 상태를 깰 것.** distractor 추가 + 가림물을 크게 움직이게
- **distractor 속도를 강제할 것** (`|v| > 100 cm/s`). 정지 물체는 시간 중앙값 배경에 흡수된다
- **가림 길이는 가림막 크기가 아니라 물체 속도/타이밍으로 조절** (v4 는 세 축이 얽혀 분리 불가였다)
- **block 안에서 예측 구간 내용량을 맞출 것** (A/B 비대칭은 누수 이용이라 금지)
- **`condition` 에 축을 접어 넣지 말 것** — `fit_groups_sweep: auto` 가 group 수만큼 head 를 만든다
- **silhouette 을 클래스 전부 동일하게** 맞출 것 (v10 은 모양이 폭과 교락됐다)

---

## 12. 작업 스타일

- **지시받은 실험을 비판적으로 평가한다.** 돌리기 전에 **"이게 논문 방향성에 맞는가"** 를 먼저 묻는다.
  `PAPER_STORY_2026-08-31.md` 의 하네스 6개 질문을 쓴다:

  1. **어느 beat 를 세우는가.** 못 대면 그렇게 말한다
  2. **결과가 어느 쪽이어도 beat 가 서는가.** 한쪽 결과로만 서면 확증편향이다
  3. **이미 답이 있는가** (§6, `SYNTHESIS` §9)
  4. **교란이 남는가.** 남으면 그 결과로 **못 하는** 주장을 미리 적는다
  5. **재학습이 필요한가.** 필요하면 그 자체로 재검토 대상이다 — 이 논문은 frozen 위에서 닫힌다
  6. **표본이 결론을 낼 크기인가** (v11: 조건×위반 672쌍 / k 까지 168쌍 / 방향당 16쌍)
  7. **corner case 인가, 어느 use case 에 닿는가** (2026-09-03 추가). 그 조건이 실제 사용·벤치마크에서
     얼마나 흔한지, 결과가 planning / 벤치마크 갭 / 평가 방법론 중 어디에 닿는지 먼저 말한다.
     **피드백을 줄 때 이 둘을 항상 같이 준다.** 못 대면 그 실험은 스토리를 못 세운다

  **동의만 하고 돌리는 것이 가장 큰 실패다.** 하루에 GPU 수 시간을 그렇게 태운 적이 있다.
  반대로 사용자가 이유를 대면 그건 결정이다 — 한 번 말하고 진행한다.

- **금지 표현** (`PAPER_STORY` 에 전체 표):
  "predictor 가 X 를 못 만든다"(probe 가 100% 로 읽는다) / "가림이 표현을 망가뜨린다"(env 72칸 100) /
  "표현이 회전했다"(beat 5 전에는 못 쓴다) / "z 와 h 가 같다"(천장) /
  "목적함수 때문이다"(discussion 한 문단) / **α·보정·헤지 계수**(인용 금지)

- **기존 하네스를 먼저 찾는다.** 새 진입점·새 규칙을 발명하기 전에 `run.sh`, `SET=`,
  `extends:`, `fit_groups_sweep: auto` 로 되는지 확인할 것
- **모델 로딩이 필요 없는 검증은 따로 떼서 몇 초에 한다** (`DRYRUN=1`)
- **수치는 문서가 아니라 산출물에서 재확인**하고 그렇게 했다고 밝힌다
- **학습된 자(readout) 로 predictor 의 세계를 말하기 전에 자 검증부터 보인다** — held-out 정밀도 · encoder 대조 · 파라미터 없는 교차검증 (2026-09-12 사용자 지시: "decoder 문제 없다를 보여야 predictor 가 만드는 세상에 대해서 주장할 수 있지"). 한 슬롯 요약 숫자 하나로 결론 내지 않는다
- **모순되는 결과를 발견하면 한쪽을 지우지 말고 조건 차이를 표로 명시한다**
- **단서를 결론과 같은 비중으로 쓴다**
- **`sed -i` 를 쓰지 않는다.** 여러 줄짜리 md/yaml 은 Python 으로 고친다
- **새로 만드는 스크립트·config·문서는 git 추적되는 자리에 둔다**
- 사용자는 한국어로 소통한다. 코드 식별자·경로·플롯 라벨은 원문 유지
