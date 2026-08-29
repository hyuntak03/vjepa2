# CLAUDE.md — V-JEPA 2 직관물리 분석 레포 하네스

이 파일은 세션 시작 시 자동으로 읽힌다. **새 세션이 이전 세션과 같은 방식으로 일하도록** 하는 게 목적이다.
수치·경로·키 이름은 전부 실물에서 검증했다. 확실하지 않은 것은 "미검증"이라고 적었다.

---

## 0. 이 레포가 하는 일 (30초 요약)

V-JEPA 2 world model이 **직관물리 위반(violation of expectation)** 을 얼마나 잡아내는지, 그리고 **왜 특정 조건에서 실패하는지** 를 규명한다.

채점 방식은 하나뿐이다 — **surprise = `mean |predictor(context) − LN(target_encoder(clip))[future]|`** (latent L1).
`surprise(impossible) > surprise(possible)` 이면 정답. chance 50%.

현재 확립된 핵심 서사:

> **predictor 는 특정 미래를 예측하지 않는다. 가림이 있든 없든 두 후보 미래의 거의 정중앙에 앉아 있다.**
> 문맥일치 쌍에서 `α = <p−h_imp, h_pos−h_imp>/‖h_pos−h_imp‖²` 를 재면 **전 조건에서 α ≈ 0.5**
> (정답 ⟺ α > 0.5). 100% 를 맞히는 static+visible 조차 마진이 **+0.031** 뿐이고 color 는 **+0.002** 다.
> **채점 성능 전체가 중앙에서 천분의 몇 벗어난 "보정" 위에 얹혀 있다.**
>
> **가림이 하는 일은 정보를 지우는 게 아니라 그 보정을 0 으로 만드는 것이다** (+0.034 → −0.003).
> 보정이 0 이 되면 기준선 `α(mu)`(= p 자리에 미래 표현의 전역 평균을 넣은 값)가 답을 정한다.
> vanish 만 두 후보가 "물체有/빈" 으로 비대칭이라 `α(mu) = 0.489`(빈 쪽) → ctxA **0%** / ctxB **100%**.
> shape·color 는 `α(mu) ≈ 0.500` 이라 동전던지기(56~72%).
> 정체성 정보 자체는 `z`(context encoder)에 네 조건 전부 온전하다(shape 100%).

**2026-08-28 추가 증거**: shape 전이 8개에서 `α(mu)` 하나가 방향 비대칭을 **r = 0.948** 로 설명한다
(0.5 기준으로 예외 없이 갈린다). 반면 probe confusion 과는 **무상관(0.00)** 이고 recall 과는
오히려 −0.549 다 — `pyramid→cylinder` 는 채점 0% 인데 pyramid recall 96%, `cylinder→pyramid` 는
채점 100% 인데 cylinder recall 46%. **§5-4c.**

근거: `z_world_model_analysis/PREDICTOR_HEDGING_2026-08-26.md` (기전),
`z_exp/world_model_analysis/results/attn_probe*/summary.json` (probing 수치 — 리포트 md 는 2026-08-28 정리 때 사라졌고 원시 산출물에서 재생성해야 한다)

---

## 1. 절대 어기지 말 것 (관례와 그 이유)

### 1-1. 기준 모델은 **ViT-H**
IntPhys1 동일 프로토콜에서 ViT-H 88.89% vs ViT-L 64.44% (+24.4pt). **외부 벤치마크에서 먼저 정한 것**이다.
자체 데이터에서는 ViT-L이 이기는 축이 있다 (v8 shape violation: ViT-L 83.98% vs ViT-H 73.83%; 2D transit overall ViT-L 75.39% vs ViT-H 71.48%; Jongseo physv3는 ViT-L 92% → ViT-H 80%).
**그래도 ViT-H를 고수한다** — 자체 데이터로 사후 모델 선택을 하지 않는다는 원칙. "ViT-H가 항상 낫다"는 거짓이니 그렇게 쓰지 말 것.

### 1-2. 주지표는 **matched pairing** (`scoring.pairing: matched`)
block(4중항)의 가능×불가능 조합은 4개지만 **문맥일치 2쌍만 context가 픽셀 단위로 같다.**
그래야 predictor 입력이 같아 `p`가 **비트 단위로 동일**(실측 `max|p_pos − p_imp| = 0.0`)하고, 채점이 "p가 h_pos와 h_imp 중 어디에 가까운가"라는 순수 기하 문제로 환원된다.
미래일치 쌍은 p부터 달라 비교가 성립하지 않고 실제로 누수를 흡수한다 (FRAME_ABLATION: 4쌍 지표 상승분이 전부 미래일치 쌍에서 나왔다).

> ⚠️ **`scoring.pairing` 기본값은 `"cross"` 다.** 안 쓰면 공식과 다른 전수 4쌍 채점이 조용히 돌아간다.

### 1-3. 프로토콜을 밝히지 않은 숫자는 비교 금지
| 비교 | 값 | 흔들림 |
|---|---|---:|
| IntPhys1: Garrido sliding vs 우리 fixed C16 | 88.89% vs 66.67% | **22.2pt** |
| v6aug: fixed / sliding avg / sliding max | 75.49 / 76.76 / 79.49 | 4.0pt |
| v6aug **static+occluded shape**: fixed vs sliding max | 46.88% vs **90.62%** | **43.7pt** |

sliding best는 같은 그리드에서 고른 값이라 **held-out 추정치가 아니라 descriptive** 다. 그렇게 명시할 것.

### 1-4. probing은 **가능(possible) 변이만**
불가능 변이를 넣으면 세 표현이 전부 오염된다:
- `contextF`(ctx_masked)는 미래 토큰을 transformer 이전에 떨궈 `imp_ab` 표현이 `pos_a`와 비트 단위 동일 → 같은 텐서 중복
- `pred`도 입력이 context뿐이라 동일
- `targetF`는 라벨이 `before`인데 렌더는 `after`라 **인코더가 정확할수록 0점**

### 1-5. split은 **반드시 block 단위**
block 안 4개는 2×2로 context/future를 공유한다. 안 묶으면 train/val에 사실상 같은 입력·같은 라벨이 들어간다.
`split: {mode: ratio, group_by: block_id, train_frac: 0.5, seed: 0, stratify_by: <실험축>}`

### 1-6. 토큰 캐시는 **반드시 로컬 디스크**
`/data`는 **NFS, 쓰기 57MB/s**. 캐시가 세트당 30~40GB라 여기 두면 10분을 그냥 버린다.
`cache_dir: /local_datasets/world/world_analysis/cache` → `/data2`(xfs) 심볼릭 링크.

### 1-7. 고정 프로토콜 스펙 (`surprise_c16t32` / `attn_probe` 공통)
- `dual_encoder: true`, `context_encoder_key: encoder`(online), `target_encoder_key: target_encoder`(EMA) — 학습 때 역할 분리 재현
- **surprise 실행**: `dtype: float32` + `autocast: float16` (공식 Garrido 재현) / **probing 실행**: `dtype: bfloat16` — 관례가 다르니 섞지 말 것
- raw 100프레임 stride 3 → 32장(`0,3,…,93`), context = 첫 16장(raw 0–45), future = 마지막 16장(raw 48–93), target encoder는 **32장 전부**
- `target_layer_norm: true`(affine-free, target에만), `distance: l1`, `loss_exp: 1.0`, `mask_index: 0`
  - `mask_index: 0`은 critical — 릴리즈 체크포인트에서 학습된 mask token은 `[0]` 하나뿐, 1~9는 0 초기화 그대로
- 가림 조건은 **4+4 대칭**: fully hidden raw 36–57 → context 끝 4장 + future 앞 4장
  - ⚠️ **v8 은 실제로 4/5 비대칭이었다.** 패널을 선언된 plateau + 고정 마진으로 잡았기 때문.
    v10 이 `sym_k: 4` 로 프레임에서 직접 풀어 정확히 4+4 를 맞췄다 (`dataset.json.protocol.note`)
- 프레임은 mp4가 아니라 **원본 PNG 직독** (코덱 손실·yuv420 크로마 서브샘플링 차단). 리사이즈는 `antialias=False` bilinear — 공식과 같은 커널
- **A/B 방향별 정확도를 항상 병기** ("Pair direction diagnostic"). scalar surprise의 appearance bias를 숨기지 않기 위함

---

## 2. 실행 방법

### 2-1. 표준 진입점 — `z_research/scripts/run.sh` (이게 정답이다)

```bash
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
bash z_research/scripts/run.sh --list                                # 뭐가 있는지
GPUS=4 bash z_research/scripts/run.sh <프로토콜> <데이터셋> [모델]   # 모델 기본 vith
DRYRUN=1 bash z_research/scripts/run.sh attn_probe v8                # 병합 결과만, GPU 안 씀
```

**세 조각이 실행 직전에 합쳐진다** (`z_research/scripts/harness/resolve.py`). 이게 이 레포의 핵심 구조다:

| 조각 | 담는 것 | 파일 |
|---|---|---|
| 프로토콜 | 프레임 배치, 채점 규칙, probe 정의, dtype 관례 | `configs/protocols/<이름>.yaml` |
| 데이터셋 | 경로, 인덱스, 컬럼 이름 | `configs/protocols/datasets.md` |
| 모델 | 체크포인트, arch_name | `configs/protocols/models.md` |

**데이터를 바꿀 때 고치는 곳은 `datasets.md` 섹션 하나뿐이다. yaml 은 건드리지 않는다.**
새 데이터셋 = `## 이름` 섹션 하나 추가.

병합 규칙 — **프로토콜이 이긴다**:
```
data  = { **datasets.md[<데이터셋>], **프로토콜.data  }
model = { **models.md[<모델>],       **프로토콜.model }
```
프로토콜이 소유한 키(`n_frames`, `resolution`, probing 의 `index_csv`/`type_column`)는
레지스트리를 덮고, 레지스트리에만 있는 키(`root`, `frames_pattern`, `block_column`)는 채워진다.
`n_frames: RAW` 는 그 데이터셋의 `raw_frames` 로 치환된다 (영상 전체를 쓰는 sliding 용).

자동으로 정해지는 것 (`TAG=` / `OUTDIR=` 으로 덮을 수 있다):
```
tag        = <데이터셋.cache_tag>_<모델>                                예: v10_vith
output_dir = <데이터셋.results_root>/<프로토콜>__<데이터셋>_<모델>
```
`results_root` 도 `datasets.md` 에서 온다 — `intphys1_dev` → `z_research/IntPhys/exp_results`,
`v8`·`v8_halfsize` → `z_research/IntPhysGenV8/exp_results`, `v10` → `z_research/IntPhysGenV10/exp_results`,
나머지는 기본값 `z_exp/world_model_analysis/results`.
`tag` 는 **토큰 캐시의 이름**이다. 프로토콜이 달라도 (데이터셋, 모델)이 같으면 같은 캐시를
쓰라고 일부러 프로토콜을 뺐다. 병합된 config 는 `<output_dir>/_resolved.yaml` 로도 남는다.

`resolve.py` 가 **모델을 로드하기 전에** 경로·인덱스 실물, `resolution == img_size`,
프레임 예산을 검사하고 안 맞으면 죽는다. ViT-H 로딩은 프로세스당 ~2분이라 그 전에 죽는 게 훨씬 싸다.

### 2-2. 현재 프로토콜 3종

| 이름 | 무엇을 재나 | 확립된 수치 |
|---|---|---|
| `intphys1_sliding` | IntPhys1 Garrido 공식 sliding | **88.89%** (intphys1_dev, vith, 180 pair) |
| `surprise_c16t32` | fixed context16 / target32 latent-L1 | **79.10%** (v8, vith, 1024 pair) |
| `attn_probe` | z / p / h 세 지점 attentive probing | z·h 100%, p 90.2%, h→p 48.0% (v8, shape) |

등록된 데이터셋: `intphys1_dev`, `v8`, `v8_halfsize`, `v10`, `jongseo_physv3`
(+ `available: false` 인 `2d_v8_transit`). 모델: `vith`, `vitl`.
**42조합 전수 확인 완료** — 통과 30 / 막힘 12 (막힌 건 전부 `index_probe.csv` 가 없는
데이터셋에 `attn_probe` 를 건 경우이고, `build_probe_index.py` 를 돌리라고 알려준다).

### 2-3. 환경변수

| 변수 | 기본 | 동작 |
|---|---|---|
| `GPUS` | `1` | `--devices cuda:0..N-1`, `WMA_EXPECT_WS` |
| `TAG`, `OUTDIR` | 자동 | `tag` / `output_dir` 직접 지정 |
| `LIMIT=N` | — | `limit: N` 주입 + **`tag`/`output_dir`에 `_smoke{N}` 접미사** (본 결과·캐시 안 덮음) |
| `SMOKE=1` | — | shape 디버그 |
| `RECACHE=1` | — | 토큰 캐시 무시하고 재추출 |
| `BATCH_SIZE`, `DECODE_WORKERS` | — | `surprise.*` 덮어씀 |
| `EVAL_DDP_PORT` | 자동 탐색 | 지정 시 탐색 생략 |
| `EVAL_DDP_TIMEOUT_S` | `7200` | NCCL collective timeout. torch 기본 600s로는 rank 부하 불균형에서 죽는다 |
| `WMA_BAR` | `auto` | `on`/`off`. `auto`는 stderr가 TTY일 때만 |
| `DRYRUN=1` | — | 병합·검사만 하고 끝 |

### 2-4. 스모크 (배관 점검)

```bash
DRYRUN=1 bash z_research/scripts/run.sh surprise_c16t32 v8 vith      # 몇 초, GPU 0장
GPUS=1 SMOKE=1 LIMIT=2 BATCH_SIZE=8 bash z_research/scripts/run.sh surprise_c16t32 v8
```
⚠️ **스모크마다 프로세스가 각각 ViT-H를 NFS에서 새로 로드해 ~2분씩 소모된다.**
`DRYRUN=1` 이 경로·컬럼·프레임 예산을 모델 없이 다 잡아 주므로 그걸 먼저 쓸 것.

### 2-5. SLURM

```bash
sbatch --job-name=prb --gres=gpu:4 --export=ALL,P=attn_probe,D=v8,M=vith,GPUS=4 \
       z_research/scripts/sbatch.sh
watch -n 1 bash z_scripts/world_model_analysis/monitor.sh
```
`-w vll5` 고정 — 데이터가 노드 로컬 `/local_datasets`에 있다. `--gres`와 `GPUS`를 반드시 맞출 것.

### 2-6. 구버전 진입점 — **2026-08-28 대부분 정리됨**

`configs/world_model_analysis/` 의 config 45개 중 **43개를 지웠다** (git 추적 2개만 남김:
`occlusion_v2.yaml`, `probe_set.yaml`). 그래서 아래가 전부 깨진 상태다:

| 깨진 것 | 없어진 config |
|---|---|
| `z_scripts/.../run_intphysgen_v8_fixed.sh` | `intphysgen_v8_vith` → **`run.sh surprise_c16t32 v8 vith`** 로 대체됨 |
| `run_intphysgen_v8_halfsize_fixed.sh` | `intphysgen_v8_halfsize_vith` → **`surprise_c16t32 v8_halfsize vith`** |
| `run_jongseo_physv3_swapshape_vith.sh` | `jongseo_physv3_swapshape_vith` → **`surprise_c16t32 jongseo_physv3 vith`** |
| `run_v8_3d_and_2d_vitl.sh` | `intphysgen_v8_vitl` → **`surprise_c16t32 v8 vitl`** |
| `run_intphysgen_v7_{fixed,sliding}.sh` | v7 — 데이터셋도 레지스트리에서 뺐다 (§4-3) |
| `run_2d_intphysgen_v8_transit_fixed.sh` | 2D — 프레임 자체가 없다 (§4-3) |
| `z_scripts/.../run.sh <이름>` | `occlusion_v2` / `probe_set` 만 아직 된다 |

**신규 실험은 전부 `z_research/scripts/run.sh` 를 쓴다.**
config 를 되살려야 하면 `z_exp/.../summary.json` 42개 안에 **그때 쓴 config 가 통째로
들어 있다**(`summary.json.config`) — 거기서 뽑아 쓰면 된다.

⚠️ 삭제 후 `z_world_model_analysis/` 의 분석 스크립트 8개가 지운 config 를 참조하는 것을
발견해 이렇게 정리했다:
- `intphys1_single_vith.yaml` **복원** — `ip_spatial.py` / `ip_mech.py` 가 쓰고 IntPhys1 데이터는 살아 있다
- `probe_signal_check.py` — `resolve.py` 경로로 갈아탐 (`--protocol/--dataset/--model`, 기본 `attn_probe v8 vith`)
- `extract.py` · `spatial.py` · `metrics.py` · `layers.py` · `mech.py` — **여전히 깨져 있다.**
  `intphysgen_v1_vith.yaml` 을 찾는데 **v1 데이터셋 자체가 이미 없어서**(`data_csv/intphysgen_v1`,
  `/local_datasets/.../intphysgen_v1` 둘 다 부재) config 삭제 이전부터 죽은 스크립트였다.
  §5-5 의 v1 수치(토큰평균+표준화 L2 92.4%)는 이 스크립트들로 다시 못 뽑는다.

아직 살아 있는 별도 진입점:
- `python -m evals.world_model_analysis.attn_probe --fname … --phase {extract,merge,probe,summary}`
  ⚠️ 이게 `<output_dir>/features/` 에 shard 를 쏟는다. 2026-08-28 에 그렇게 쌓인 **33G 를 지웠다**
  (현행 `eval.py` 는 `features.cache_dir` 만 쓰므로 중복이었다). 이 경로를 다시 쓸 거면 용량을 볼 것.
- `evals/analysis_vlm/{occlusion_identity,occlusion_surprise,parabolic}/eval.py` — 자체 argparse

⚠️ **`evals.main`을 직접 부르지 말 것.** `run.sh` 만 하는 두 가지가 있다 — DDP 포트 자동 탐색과
`WMA_EXPECT_WS` export. 이게 없으면 §7-1의 split-brain 가드가 통째로 비활성된다.
`evals/main.py`가 rank마다 `CUDA_VISIBLE_DEVICES`를 하나로 고정하므로 eval 코드는 항상 `cuda:0`을 쓴다.

---

## 3. config 스키마 (world_model_analysis)

`eval_name` → `evals.<eval_name>.eval` 의 `main(args_eval, resume_preempt)` 을 부른다 (`evals/scaffold.py`).

### 3-1. 블록별 유효 키

**최상위**: `eval_name`(필수), `tag`, `output_dir`(필수), `data`, `model`, `features`, `surprise`, `scoring`, `probing`, `limit`, `recache`, `smoke`

**`data:`** — `evals/world_model_analysis/data.py`
`root`(필수), `n_frames`(40), `resolution`(None), `index_csv`("index.csv"), `frames_root`(None), `frames_pattern`, `frames_start`(1), `frames_stride`(1), `block_column`, `group_column`(→`variant_column`→"variant"), `plausible_column`("plausible"), `type_column`, `pair_column`("pair_id"), `split`

**`model:`** — `analysis/intphys2/model.py::build_from_config`
`checkpoint`(필수), `arch_name`("vit_large"), `img_size`(256), `patch_size`(16), `tubelet_size`(2), `window_size`(48), `use_rope`(true), `uniform_power`(false), `dual_encoder`(false), `context_encoder_key`/`target_encoder_key`("target_encoder"), `predictor:{embed_dim 384, depth 12, num_heads 12, num_mask_tokens 10}`, `dtype`("bfloat16"), `autocast`(호출부가 읽음)

**`features:`** (없으면 `surprise`로 폴백)
`cache_dir`(**필수, 없으면 KeyError**), `context_length`(32), `mask_index`(0), `batch_size`(4), `cache_dtype`("float16")

**`surprise:`**
공통: `mode`("single"|"intphys1"), `distance`("l1"), `loss_exp`(1.0), `target_layer_norm`(true), `mask_index`(0)
`mode: single`: `context_length`(32), `batch_size`(4), `decode_workers`(0)
`mode: intphys1` → `surprise.intphys1:` 서브블록: `frame_skips`([2,5,10]), `window_sizes`([16,32]), `context_mult`([2,4,6,8,10]), `stride`(2), `context_reduce`("min"), `aggregate`(["avg","max"]), `max_batch`(16), `frame_budget`("full"), `video_batch`(1), `dump_windows`(false)

**`scoring:`**
`pairing`("cross"), `breakdown`([]), `single_video`(false), `token_subset:{modes, mode, object_radius}`

**`probing:`**
`enabled`(false), `probe`("attentive"), `optim`("attn_default"), `probes`(필수), `optims`(필수), `targets`, `runs`(필수), `fit_groups_sweep`([None]), `block_types`, `split`, `fit_variants`, `eval_variants`

`runs[].train` / `.eval[]` 의 4축 (`evals/world_model_analysis/schema.py`):

| `model` | `input` | encoder | source 이름 | offset |
|---|---|---|---|---|
| `context` | `full` | `ctx_masked` | `contextF__f{a}to{b}` | 0 |
| `context` | `window` | `isolated_ctx` | `contextW__f{a}to{b}` | f0 |
| `target` | `full` | `target` | `targetF__f{a}to{b}` | 0 |
| `target` | `window` | `isolated_target` | `targetW__f{a}to{b}` | f0 |
| `predictor` | (자동) | `predictor` | `pred__f{C+1}to{N}` | ctx_len |

- `frames`는 **사람이 세는 1-idx 양끝 포함**. tubelet 정렬 필수
- `eval: [self, …]` — `self`는 train source 자기 자신 (= 정보 존재 여부 = 이식의 상한)
- head는 `(fit, groups)`마다 **딱 한 번** 학습되고 eval 목록 전체에 frozen 적용

**probe 스펙** (`evals/analysis/probes.py`): `{type: attentive, num_heads: 16, num_probe_blocks: N}` 또는 `{type: linear, pooling: mean|max|meanmax, pre_norm: bool}`
**optim 프리셋**: `{num_epochs, batch_size, lr, weight_decay}` **4개만 소비된다.** 스케줄러 없음, 항상 AdamW 상수 lr.

### 3-2. 죽은 키 — 쓰지 말 것 (조용히 무시된다)

| 키 | 실체 |
|---|---|
| `report:` (`confusion`, `save_features`) | **`occlusion_identity/eval.py` standalone만 읽는다.** wma eval.py는 안 읽음 |
| `scoring.metrics` | 아무 데서도 안 읽음 |
| `features.num_workers` | wma eval.py는 안 읽음 (attn_probe.py만 `decode_workers` 사용) |
| `data.eval_groups`, `probing.eval_groups` | 코드가 안 읽음. 진짜 키는 `probing.eval_variants` |
| `data.strata` | `Rec.strata`에 실리기만 하고 리포트 미출력 |
| `data.expect_per_block` | 안 읽음 |

⚠️ `fit_variants`/`eval_variants`는 **`group_column` 값으로 거른다.** `group_column: condition`으로 두면 `pos_a/pos_b` 필터를 못 쓴다.
→ 그때는 index에 파생 컬럼을 만들고 `type_column` + `probing.block_types`로 거른다 (`build_probe_index.py`의 `probe_type` 참고).

### 3-3. 출력물

```
<output_dir>/summary.json      # tag, world_size, n_videos, config 전체, surprise{...}, probing[...]
<output_dir>/per_block.json    # per_video_surprise{}, per_block[]  (surprise 실행 시)
<output_dir>/per_window.json   # intphys1.dump_windows: true 일 때만
<features.cache_dir>/<tag>/    # meta.json + target.npy / ctx_masked.npy / predictor.npy …
```

`summary.json.surprise` 모양이 3가지다:
- `mode: single` + token_subset 없음 → `{"overall": {...}, "chance", "pairing", "block_distribution", "by_block_type"}`
- `mode: single` + token_subset → 키가 subset 이름(`all`/`object`/…)
- `mode: intphys1` → 키가 `"{combo}/{agg}"` (예: `skip2_w32/avg`)

`summary.json.probing[]` 항목:
```jsonc
{"fit": "contextF__f1to16", "groups": null, "target": "shape",
 "n_train": 389, "train_acc": 1.0, "chance": 0.125,
 "evals": {"<source>": {"overall": 0.94, "per_group": {"<group>": {"n":95,"acc":1.0,"bacc":1.0}}}}}
```

---

## 4. 데이터셋과 디스크

### 4-1. 디스크 — **여유가 거의 없다. 쓰기 전에 반드시 `df -h` 확인**

| 마운트 | 타입 | 용량 | 여유 | 성격 |
|---|---|---|---|---|
| `/` | ext4 NVMe | 915G | **~0.1G (100%)** | **`/local_datasets`가 여기 있다. 사실상 꽉 참** |
| `/data` | **NFS** | 91T | ~14T | 레포·UnrealEngine 원본. **쓰기 57MB/s** |
| `/data2` | xfs 로컬 | 7.0T | **~200G (98%)** | 토큰 캐시 전용 |

`/local_datasets/world/world_analysis/cache → /data2/local_datasets/world/world_analysis/cache` (심볼릭 링크)

### 4-2. index 스키마

**IntPhysGen 공통 14컬럼** (`build_intphysgen_v6_index.py` 산출):
`video_id, file, file_name, block_id, source_block, variant, plausible, pair_id, condition, motion, has_occlusion, violation_type, game_name, role`
- `file`은 **비어 있다** — 프레임은 `frames_root` + `frames_pattern: "{file_name}/{frame:06d}.png"`
- `condition` ∈ moving_occlusion / moving_visible / static_occlusion / static_visible (균등 4분할, 512씩)
- `violation_type` ∈ vanish(1024) / color(512) / shape(512)
- `variant` ∈ pos_a / pos_b / imp_ab / imp_ba, `plausible` 정확히 반반
- 동반 `context_integrity.json`

**`index_probe.csv`** (probing 전용, `build_probe_index.py v10|v8|2d`) = 위 + 라벨 컬럼 + **`probe_type`**
`probe_type` ∈ `obj`(가능+물체 있음 = **유일한 probing 대상**) / `empty` / `imp`
- v8 : obj 768 / empty 256 / imp 1024
- v10: obj **3072** / empty 1024 / imp 4096 — shape·color **클래스당 정확히 384개**
  (v8 은 40~58, 28~68 로 불균형했다. §5-4c 의 교란이 v10 설계에서 해소됐다)
- **v10 만 `surface`(flat/ramp)를 갖는다.** index.csv 엔 없고 index_probe.csv 에만 있다

**2D transit index**는 `shape_pre/color_pre/shape_post/color_post/direction/occlusion_design`을 index.csv 단계부터 이미 갖고 있다. **v8 3D는 없어서 metadata.csv에서 조인해야 한다.**

**Jongseo physv3**만 `file`이 실제 mp4 절대경로(`/data/jongseo/…`)다 — 유일하게 PNG가 아니라 영상 직독.

### 4-3. 원본 프레임 위치

**등록된 것은 `configs/protocols/datasets.md` 가 정본이다.** 아래는 그 밖의 실물 위치.

- IntPhysGen v1~v9: `/data/hyuntak/project/2026/2027_cvpr/UnrealEngine/IntPhysGen_v{N}/` (NFS)
- **IntPhysGen v10**: `/local_datasets/world/world_analysis/IntPhysGen_v10` → `/data2/...` (42G, 로컬 xfs)
  - 8192 clip / 2048 block. `metadata.csv` **94컬럼**, `dataset.json` 에 설계 의도가 적혀 있다
  - v8 대비: `surface`(flat/ramp) 축 신설 · 가림 4+4 **정확** 대칭(v8 은 4/5 였다) ·
    배경 완전 교차 · 물체 300→150cm. 자세한 건 `configs/protocols/datasets.md`
  - `--strict-context` 로 **65,536 문맥 프레임 쌍 전수 검사, mismatch 0**
  - v8 `metadata.csv`는 **78컬럼** — index.csv로 축약되기 전 원본. `hidden_start/hidden_end/reveal_frame`, `obj_apparent_px`, `speed_cm_s`, `lag` 등이 여기 있다
- IntPhys1 dev PNG: `/local_datasets/world/world_analysis/IntPhys1_dev_frame_png/` (3.3G)
- blender occlusion v2/v3/v4, probe_set: `/local_datasets/world/world_analysis/`
- **⚠️ 2D transit 프레임(195M)은 2026-08-28 `z_research` 정리 때 사라졌고 생성 코드도 함께 없어졌다.**
  레포 전역에 `occlusion_design`/`shape_post` 를 쓰는 `.py` 가 하나도 안 남았다.
  토큰 캐시(`cache/2d_v8_transit_vith` → `attn_probe_2d_v8_transit_vith`, 41G)는 남아 있어
  **캐시 기반 사후 분석은 계속 가능**하지만 재추출·재렌더·프레임 오버레이는 불가능하다.
  `datasets.md` 에 `available: false` 로 등록돼 있다.
- **v6 / v6_augmented / v7 은 2026-08-28 레지스트리에서 뺐다** (v8 이전 버전 정리).
  §5의 수치는 `z_exp/world_model_analysis/results/intphysgen_v{6,7}*/summary.json` 에서
  여전히 재검증할 수 있다. 다시 돌리려면 `datasets.md` 에 섹션을 되살리면 된다
  (git 이력: `configs/protocols/datasets.md`).

---

## 5. 확립된 결과 (새 세션이 알고 시작해야 할 숫자)

### 5-1. surprise 채점 (matched pair, chance 50%) — **artifacts에서 직접 검증함**

| 데이터셋 | 모델 | 프로토콜 | overall | n_pair |
|---|---|---|---:|---:|
| **IntPhys1 dev** | **ViT-H** | Garrido `skip2_w32/avg` | **88.89%** | 180 |
| IntPhys1 dev | ViT-L | 동일 | 64.44% | 180 |
| IntPhys1 dev | ViT-H | 우리 fixed C16/stride3 | 66.67% | 180 |
| IntPhysGen v6 augmented | ViT-H | fixed | 75.49% | 1024 |
| IntPhysGen v7 | ViT-H | fixed | 78.22% | 1024 |
| **IntPhysGen v8 (정식)** | **ViT-H** | fixed | **79.10%** | 1024 |
| IntPhysGen v8 | ViT-L | fixed | 73.44% | 1024 |
| v8 halfsize (28.1px) | ViT-H | fixed | 74.71% | 1024 |
| 2D transit | ViT-H | fixed | 71.48% | 1024 |
| 2D transit | ViT-L | fixed | 75.39% | 1024 |
| Jongseo physv3 swapshape | ViT-L→ViT-H | C16/P16 | 92.0% → 80.0% | 50 |

### 5-2. motion × occlusion (ViT-H) — **occlusion이 전 버전 공통 최대 병목**

| 데이터셋 | Mov+Occ | Mov+Vis | Sta+Occ | Sta+Vis |
|---|---:|---:|---:|---:|
| IntPhys1 (sliding) | **75.00** | 91.67 | **100** | **100** |
| v6 augmented | 56.25 | 94.14 | 55.86 | 95.70 |
| v7 | 54.69 | 98.05 | 62.11 | 98.05 |
| **v8** | **53.52** | 98.05 | **66.80** | 98.05 |
| v8 halfsize | 53.52 | 92.19 | 59.77 | 93.36 |
| 2D transit | 54.69 | 89.06 | 57.81 | 84.38 |

**static+occluded shape 근-chance가 버전을 관통한다**: v6 46.88 → v7 54.69 → v8 53.12 → halfsize 50.00.
⚠️ 단, 같은 v6 데이터에 sliding `skip2_w32/max`를 쓰면 **90.62%** 로 살아난다. 이 두 서술은 반드시 함께 인용할 것.
**moving+occluded는 sliding으로도 회복 안 된다** → 진짜 하드 케이스는 "motion under occlusion".

### 5-3. 방향 비대칭 (v8 3D)

| Family | w/o Occ | w/ Occ | 하락 |
|---|---:|---:|---:|
| Shape | 96.88 | 50.78 | −46.1p |
| Color | 95.31 | 67.97 | −27.3p |
| Vanish 존재→없음 (A) | 100 | **32.81** | −67.2p |
| Vanish 없음→존재 (B) | 100 | 89.06 | −10.9p |

Moving+Occluded vanish: **A 0.00% (0/64), B 100.00% (64/64).** 모델이 "빈 미래"를 구조적으로 선호한다.
이 0%/100% 짝 구조는 blender 시절부터 동일 (`pos_obj<imp_vanish` = 0.00%가 7개 설정 전부).

### 5-4. Attentive probing (chance 12.5%, 8-way) — **현재 가장 중요한 결과**

세 지점 모두 `(2048, 1280)`: `2048 = (16/tubelet 2) × (256/16)²`

| | z self | h self | p self | h→p 이식 |
|---|---|---|---|---|
| **v8 3D 전체** | 100 / 94.46 | 100 / 96.57 | **90.24 / 70.71** | **48.02 / 49.87** |
| v8 3D, 가림 조건만 | 100 / 93.1 | 100 / 96.8 | 82.0 / 54.5 | **14.8 / 14.8 (chance)** |
| **2D transit** | 100 / 100 | 100 / 100 | **100 / 100** | 가림시 31.9·45.3 / 10.6·9.5 |

(shape / color 순)

**분해**: 정보손실 = (h self) − (p self), 정렬손실 = (p self) − (h→p)

| 데이터셋 | target | 가림 | 정보손실 | 정렬손실 |
|---|---|---|---:|---:|
| 3D | shape | 없음 → 있음 | +1.6p → **+18.0p** | +17.4p → **+67.2p** |
| 3D | color | 없음 → 있음 | +9.5p → **+42.3p** | +2.1p → **+39.7p** |
| **2D** | shape | 없음 → 있음 | **+0.0p → +0.0p** | +1.1p → **+61.4p** |
| **2D** | color | 없음 → 있음 | **+0.0p → +0.0p** | +0.0p → **+89.9p** |

**읽는 법** (⚠️ 이 분해는 *증상*이다. 기전은 §5-4b 의 α 를 볼 것):
- **정렬손실 = occlusion이 유발.** 3D/2D 공통, 가림에서만 폭발
- **정보손실 = 3D 렌더링 복잡도가 얹은 별개 항.** 2D는 전부 0.0p
- shape은 정렬손실이 지배(4배), color는 둘이 비등하고 `static_occlusion`은 정보손실이 더 크다
- **"predictor에 정보가 빈약하다"고 쓰지 말 것.** 정확한 서술: **"predictor는 가림에서 일부 정보를 잃고(3D 한정) 동시에 h의 표현 공간을 이탈한다(3D·2D 공통)"**

### 5-4b. **왜 무너지는가 — α 분해 (기전)**

`z_world_model_analysis/PREDICTOR_HEDGING_2026-08-26.md`. 재현: `python z_world_model_analysis/alpha_hedging.py` (캐시만, GPU 불필요)

문맥일치 쌍은 `p`가 비트 단위로 동일하므로 `α = <p−h_imp, D>/‖D‖²` (`D = h_pos−h_imp`) 하나로
"predictor가 어느 미래를 예측했나"를 잰다. **정답 ⟺ α > 0.5.** D에 직교하는 잔차는 정확히 상쇄된다.
(⚠️ `MECHANISM_2026-08-20.md`는 D를 반대로 잡아 α<0.5가 정답 — 부호 규약이 다르다.)

`α(mu)` = p 자리에 미래 표현의 **전역 평균**을 넣은 기준선. **보정 = α(p) − α(mu)** 가 순수 과제 신호.

| 실험 | sta+VIS | mov+VIS | mov+OCC | sta+OCC |
|---|---|---|---|---|
| vanish ctxA α(p) / 보정 / 정답 | .525 / **+.031** / 100% | .522 / **+.034** / 100% | .486 / **−.003** / **0%** | .501 / +.006 / 67% |
| vanish ctxB | .539 / +.031 / 100% | .559 / +.050 / 100% | .523 / +.011 / **100%** | .505 / −.002 / 88% |
| shape ctxA | .517 / +.016 / 97% | .510 / +.010 / 100% | .502 / +.002 / 56% | .501 / +.001 / 62% |
| color ctxA | .504 / +.002 / 97% | .503 / +.001 / 88% | .501 / +.000 / 69% | .501 / −.001 / 72% |

1. **predictor는 가림이 없어도 커밋하지 않는다.** α ≈ 0.5가 전 조건. 100%를 맞히는 sta+VIS조차 마진 +0.031, color는 +0.002. **성능 전체가 종잇장 마진 위에 있다**
2. **가림 = 보정을 0으로 만드는 것.** 정보를 지우는 게 아니다 (`z` self-probe는 네 조건 전부 shape 100%)
3. **보정이 0이면 α(mu)가 답을 정한다** — vanish만 후보가 "물체有/빈"으로 비대칭이라 α(mu)=0.489(빈 쪽) → ctxA **0%**, ctxB **100%**. shape/color는 α(mu)≈0.500이라 동전던지기(56~72%). **0%/100% 비대칭은 "빈 미래 선호"가 아니라 편향된 기준선 + 사라진 보정**
4. **보정의 출처 = "context 끝에서 물체가 보이는가".** 지평 유지 여부는 motion에 달렸다 — 정지는 t0~t7 평평(0.53), 이동은 0.57→0.48 감쇠. **"단기 외삽"은 이동 조건에만 맞는 서술이다**. 가림은 `hidden_start=36`이라 공급원 자체가 없어 t0부터 0
5. **물체 영속성 실패**: `reveal_frame=58`(future tubelet 2) 이후에도 mov+OCC의 α는 0.48~0.49 그대로. **predictor는 가림막이 지나간 뒤에도 물체를 다시 만들지 않는다**
6. `|p−mu| ≈ 15` vs `|h−mu| ≈ 31` — **p는 실제 미래보다 평균에 두 배 가깝다.** `p`는 LN을 안 받아 `|p|/|h| ≈ 0.60`

**아직 없는 대조군**: h-head를 가림 조건만으로 학습(`fit_groups_sweep`) / linear+mean-pool probe 이식 / 선형 사상 `W: h→p`

### 5-4c. p self-probe confusion + 방향 비대칭의 정체 (2026-08-28)

재현 (단일 GPU, `ddp=False`, 결정론적 — DDP 는 실행마다 흔들린다):
```bash
DRYRUN=1 bash z_research/scripts/run.sh attn_probe v8 vith   # 병합 config 확인
python z_world_model_analysis/p_self_confusion.py            # 약 5분, GPU 1장
```
⚠️ **아래 수치를 뽑은 원시 예측(`fig_p_self_confusion_raw.json`)과 그림은
2026-08-28 `z_research` 정리 때 사라졌다.** 위 명령으로 다시 뽑아야 하고,
결과는 `z_research/IntPhysGenV8/figures/` 로 간다. 그 다음부터는 `_raw.json` 이
예측을 담고 있어 **재학습 없이** 다시 그릴 수 있다.
`eval.py` 도 이제 `<output_dir>/predictions.json` 을 뱉으므로 앞으로는 이 재학습 자체가 불필요하다.

**(1) 오류가 무작위가 아니다** — 인접한 이웃 한둘로만 샌다.

| | 전체 | 가림없음 | 가림 |
|---|---:|---:|---:|
| shape (train_acc 0.949) | 90.8% | 96.3% | **85.2%** |
| color (train_acc 0.830) | 70.4% | 82.1% | **58.7%** |

가림 조건 주요 오분류: `cylinder→cube` 50%, `cone→pyramid` 25%, `narrowcap→capsule` 10%
(기하학적으로 닮은 쌍) / `cyan→blue` 68%, `magenta→purple` 33%, `yellow→orange` 47%
(색상환 인접). `sphere`·`torus`·`capsule`·`blue` 는 가림에서도 100%.
신호가 없다면 8클래스에 고르게 흩어져야 한다 — **"신호 없음" 가설을 한 번 더 반박한다.**

**(2) surprise 방향 비대칭은 probe confusion 과 무관하다.** shape 8개 전이 전부
`conf A→B = 0%` — **설계 쌍(`capsule↔cone`, `cube↔torus`, `cylinder↔pyramid`,
`narrowcap↔sphere`)은 probe 가 한 번도 안 헷갈린다.** probe 가 헷갈리는 건 쌍을 가로지르는 조합이다.

| 설명 후보 | 방향 비대칭과의 상관 (n=8) |
|---|---:|
| probe 가 A 를 B 로 헷갈리는 정도 | **0.00** |
| probe 가 A 를 읽는 정확도 | −0.549 (오히려 반대) |
| **α(mu) — 데이터 기하의 기준선** | **+0.948** |
| α(p) | +0.906 |
| 보정 = α(p) − α(mu) | +0.440 |

```
전이                 surprise    α(mu)          전이                surprise   α(mu)
pyramid → cylinder      0.0%     0.498          capsule → cone        81.2%    0.502
torus   → cube          6.2%     0.495          narrowcap → sphere    87.5%    0.503
sphere  → narrowcap    12.5%     0.497          cube → torus          93.8%    0.505
cone    → capsule      25.0%     0.499          cylinder → pyramid   100.0%    0.503
```
**0.5 를 기준으로 예외 없이 갈린다.** α(mu) < 0.5 인 넷이 하위 넷(0~25%),
> 0.5 인 넷이 상위 넷(81~100%).

가장 극적인 대비: `pyramid→cylinder` 는 채점 **0%** 인데 probe 의 pyramid recall 은 **96%**,
`cylinder→pyramid` 는 채점 **100%** 인데 cylinder recall 은 **46%** 다.
**probe 가 잘 읽는 쪽이 채점은 0%, 잘 못 읽는 쪽이 채점은 100%.**

→ **방향 비대칭 = "p 에 정체성이 없어서"가 아니다.** 가림에서 보정이 ~0(+0.000~0.004)이라
판정이 전적으로 "두 후보 미래 중 어느 쪽이 전역 평균에 가까운가"로 결정된다.
§5-4b 의 vanish 0%/100% 와 **같은 기전**이고, shape 은 α(mu) 편차가 작아(0.495~0.505)
0~100% 스펙트럼으로 퍼질 뿐이다.

⚠️ probe confusion 과 surprise 는 **서로 다른 질문**이다. probe 는 `pos` 영상만 보고
"p 에 A 가 있나"를 묻고(imp 표현을 본 적이 없다), 채점은 "p 가 h(A) 와 h(B) 중 어디에
가까운가"를 묻는다. 두 결과가 어긋나는 건 모순이 아니다.

**(3) probing 클래스 불균형 (미해결 교란)** — `stratify_by: condition` 이라 조건은
192씩 균등하지만 **클래스는 아니다.**

```
shape  capsule 55  cone 58  cube 42  cylinder 42  narrowcap 47  pyramid 56  sphere 40  torus 49
color  blue 44  cyan 55  green 68  magenta 55  orange 36  purple 51  red 52  yellow 28   <- 2.4배
```
`yellow` 는 train 28 개로 최소인데 가림 recall 도 21% 로 최소다. color self 가 shape 보다
낮은 게 색 표현이 약해서인지 표본이 적어서인지 **아직 안 갈렸다.**
가르려면 `build_probe_index.py` 에서 `condition|color_pre` 결합 컬럼을 만들어 거기에 stratify 한다.

---

### 5-5. 지표를 바꾸면 살아난다 (IntPhysGen v1)

| 거리 정의 | shape | color | 전체 |
|---|---:|---:|---:|
| 토큰 단위 L1 (현재 코드) | 52.9 | 65.3 | 56.1 |
| 토큰평균 + L1 | 88.2 | 59.4 | 82.2 |
| 토큰평균 + 중심화·표준화 L2 | **97.1** | **80.6** | **92.4** |
| probe로 읽기 | **100.0** | **98.3** | — |

⚠️ **`distance: pooled_l2` 계열은 아직 eval.py에 없다.** 위 표는 전부 사후 스크립트 계산이다.

### 5-6. α 분해 — IntPhys와 우리 데이터를 가르는 단일 지표
`α = <p−h_pos, Δ>/‖Δ‖²`, `Δ = h_imp − h_pos`. L2/코사인/사영 채점은 정확히 `α<0.5`일 때 맞는다.
> ⚠️ **§5-4b와 부호 규약이 반대다** (거기선 `D = h_pos−h_imp`, 정답 ⟺ α>0.5). 두 절의 수치를 직접 비교하지 말 것.

| | α−0.5 | (α−0.5)/σ | 정확도 |
|---|---:|---:|---:|
| IntPhys 전체 (119쌍) | **−0.0324** | −0.91 | 89.1% |
| IntPhysGen shape (60) | **−0.0002** | −0.05 | 41.7% |

**153배 차이.** 원인은 "불가능 미래가 자연 분포에서 벗어나는가": PCA 부분공간 밖 에너지로 재면 IntPhys corr **+0.710**, 우리 **−0.589(부호 반대)**.

---

## 6. 이미 기각된 가설 — **다시 시도하지 말 것**

| 가설 | 기각 근거 |
|---|---|
| context 길이가 문제 | C=4/8/12/16 전부 50.9~54.3% |
| 희석 — 물체 토큰만 보면 됨 | `token_subset: object` 25~39배 좁혀도 **더 나빠짐** (누수/신호 1.80→3.97) |
| 채점 프로토콜(sliding)이 원인 | IntPhys는 고정 단창 하나로도 88.33% |
| 배경 복잡도 / 카메라 시점 | env 8종, 탑다운 vs 원근 전부 무차이 |
| predictor 내부가 정보를 잃음 | 층별 probe 68→86% 단조 증가, proj 후 88.7% |
| **"p에 정보가 없다"** | probe 88.7%, 채점규칙으로 쓰면 100% — **철회** |
| **"물체를 엉뚱한 자리에 그린다"** | Mh가 물체 위치에 안 몰림(상위1% = 1.9%) — **철회** |
| **"IntPhys 물체가 2배 크다"** | 개수와 크기를 섞어 잰 오류 — **철회** |
| **"우리 신호가 작아서"** | v8 RGB MAE 1.484–2.079 vs Jongseo 1.639로 **오히려 더 큼** — v6 결론이 v8에서 뒤집힘 |
| 궤적·속도 위반 추가 | **IntPhys1 dev에 그런 사례가 0건** (`n_violations=2`는 순간이동이 아니라 일시소멸→원위치 재출현) |
| `vanish`를 현행 대칭 구조로 유지 | A/B 상쇄로 **원리적으로 50%** (|dc|/평균내용 = 200%) |
| **"방향 비대칭은 p 가 그 모양을 못 담아서"** | 설계 쌍 8개 전부 probe confusion 0%. 상관 0.00, recall 과는 −0.549 로 **반대**. 진짜 원인은 α(mu) (r=0.948) — **§5-4c** |
| **"p self 가 낮은 건 신호가 없어서"** | 오분류가 기하·색상환 이웃으로만 샌다(고르게 안 흩어짐). 단일 GPU 결정론 실행에서 shape 90.8% — DDP 3회 중 90.2/90.5 와 일치, 80.7 이 이상치. **미수렴 문제** |

---

## 7. 함정과 검증 습관

### 7-1. 인프라
- **DDP split-brain**: `init_distributed`가 bind 실패를 `except Exception`으로 삼키고 조용히 `world_size=1` 폴백. 뒤에 뜬 job의 rank0만 폴백하고 나머지는 먼저 뜬 job의 store에 붙어 **두 job이 섞인다.** → `run.sh`가 포트 자동 탐색 + `WMA_EXPECT_WS` 가드. **`evals.main`을 직접 부르면 가드가 통째로 비활성**
- **토큰 캐시가 조용히 재사용된다**: `TokenCache.matches()`는 `video_ids`/토큰수/dtype만 본다. **체크포인트·해상도·autocast·재렌더는 서명에 없다.** 데이터를 다시 만들면 캐시도 반드시 치울 것. 보관 관례: `cache/_previous_YYYYMMDD/<tag>`, `results/_previous_<name>_<date>/`
  - 참고: `attn_probe.py`의 `_signature()`는 제대로 된 서명(체크포인트·dtype·autocast·stride 전부 포함)을 쓴다 — 두 캐시 시스템의 엄밀성이 다르다
- **`features:` 블록 누락** → `cfg["surprise"]`로 폴백 → `cache_dir` KeyError. **surprise 채점을 다 끝낸 뒤에 죽는다** (가장 아까운 실패)
- **`data.resolution != model.img_size`** → 작으면 CUDA assert, 크면 **에러 없이 엉뚱한 구간을 잘라 쓴다.** `WMADataset`이 `ValueError`로 막아 준다
- **CPU 스레드 과다구독** — rank마다 전체 코어를 잡으면 decode가 39ms → 268ms(7배). `torch.set_num_threads(cpu//ws)`로 방어됨
- **probing `batch_size`는 global batch** (DDP가 rank당 1/ws로 나눔). surprise의 `batch_size`는 rank당

### 7-2. 수치
- FP16 batching은 근접 tie를 뒤집을 수 있다. bf16 + `video_batch>1`이면 reduction 순서가 바뀐다 → 재현성 원하면 `video_batch: 1`
- `RESULTS_2026-08-16`의 occlusion/probe_set 수치는 **bfloat16 시절**이다. 현행 config(fp32+autocast fp16)로 다시 돌리면 움직인다
- 결정론은 확인됨: 같은 config 재실행 전 자리 일치, token_subset `all`은 기존과 `max|Δ| = 1.2e-07`

### 7-3. 통계 해석
- **CI 폭 0인 셀은 버그가 아니다** — presence의 [50.00, 50.00]은 100%/0% 상쇄 구조
- **transition 히트맵 셀은 n=8** — 12.5%p = 1쌍. 과해석 금지
- **probe head가 과대용량**(49.3M, n_train 256~389) → contextF/targetF는 `train_acc=1.000`으로 포화. **천장에 안 닿은 건 이식 줄뿐이므로 대조는 이식 줄에 한해 읽을 것**
- **fold 없음/seed 1개** 실행에서 중간 크기 값(예: `moving_visible` shape 정렬손실 +31.6p)은 재확인 필요
- `folds: 1 + holdout: 0.25`와 `folds: 4`는 **train 크기가 같다**(144). fold를 쓰면 test 커버리지만 4배(48→192)가 된다 — 표본이 적을 때 공짜로 얻는 정밀도
- **교란: 가림막 존재 자체** — occluded 조건에만 가림막이 장면에 추가로 있다. "가려짐"과 "가림막이 화면에 있음"이 3D·2D 둘 다 분리 안 된다

### 7-4. 필수 검증 루틴
- **byte-identical context 감사**: 모든 matched pair의 16 context 프레임이 픽셀 단위로 같은지 전수 검사. v6~v8·2D 전부 16,384 pair 중 mismatch 0. 산출물 `data_csv/<dataset>/context_integrity.json`
  - **실패 사례**: v6 원본 `static+occluded`는 splice=36이라 96쌍 전부가 마지막 4 context 프레임에서 갈렸다 (6144쌍 중 384 mismatch)
- **`|z(p)|` 단독 채점이 정확히 0.00%(전 쌍 tie)** 로 나오는 것이 파이프라인 정합성 증거
- 수치는 문서가 아니라 **`summary.json`/`per_block.json`에서 다시 계산해 검증**하고, 그렇게 했다고 문서에 명시할 것

---

## 8. 문서·그림 작성 관례

### 8-1. 디렉토리 역할

| 위치 | 역할 |
|---|---|
| `configs/protocols/` | **프로토콜 yaml + `datasets.md` / `models.md` 레지스트리.** git 추적됨 |
| `z_research/scripts/` | 최상위엔 **직접 치는 것만** (`run.sh`·`sbatch.sh`·`monitor.sh`). 나머지는 `harness/`·`data/`·`figures/`·`analysis/`. **git 추적됨** — §8-4 |
| `z_research/IntPhys/`, `z_research/IntPhysGenV8/`, `z_research/IntPhysGenV10/` | 데이터셋별 결과 아카이브 (2026-08-28 신설) |
| `z_findings/*.md` | run 스크립트가 자동 생성하는 결과 카드 + 수기 종합 노트 |
| `z_world_model_analysis/` | 심층 분석. **파일명에 날짜**: `TOPIC_YYYY-MM-DD.md`. 분석 `.py`를 나란히 둔다 |
| `z_research/<셋>/exp_results/<프로토콜>__<데이터셋>_<모델>/` | 원시 산출물 (`summary.json`, `_resolved.yaml`). 레지스트리 `results_root` 가 정한다 |
| `z_exp/world_model_analysis/results/` | `results_root` 를 안 준 데이터셋의 기본 자리 + 과거 실행 42개 |
| `configs/world_model_analysis/` | 2026-08-28 에 43개 삭제, `occlusion_v2`·`probe_set` 만 남음. 신규는 `configs/protocols/` |

> ⚠️ 2026-08-28 에 `z_research` 를 통째로 정리하면서 `IntPhys-Like_data/` 아래 리포트·figure가
> 전부 사라졌다. **git 추적이 0개였기 때문에 복구가 불가능했다.** 그래서 새 스크립트·config 는
> 추적되는 자리(`configs/`, `z_research/scripts/`)에 두고 커밋한다.
> 원시 산출물(`z_exp`)·분석 스크립트(`z_world_model_analysis`)는 무사해서 리포트·그림은 재생성 가능하다.

### 8-4. 새 파일을 만들 때 (2026-08-29 정리에서 굳어진 것)

⚠️ **레포 루트나 폴더 최상위에 파일을 흘리지 않는다.** 하루 만에 `z_research/scripts/` 에
py 20개가 평평하게 쌓였고 루트에 `.diag.txt` 류 임시 출력 13개가 남았다. 둘 다 정리했다.

**스크립트** → `z_research/scripts/` 의 역할 폴더 중 하나. 최상위엔 **사람이 직접 치는 것만**.

| 폴더 | 무엇 |
|---|---|
| (최상위) | `run.sh` · `sbatch.sh` · `monitor.sh` |
| `harness/` | `run.sh` 가 부르는 것 (`resolve.py`) |
| `data/` | 인덱스·데이터 준비 |
| `figures/` | 논문 그림. **전부 산출물에서 재계산해 `summary.json` 과 대조 검증** |
| `analysis/` | 토큰 캐시 기반 분석. **GPU 불필요** |

맞는 폴더가 없으면 **폴더를 새로 만들고 `z_research/scripts/README.md` 에 한 줄 추가**한다.

**임시 출력** → 레포에 쓰지 말고 `/tmp` 나 스크래치패드로. 문서화할 값이면 먼저
`z_research/<셋>/Archive/` 의 md 로 옮기고 원본을 지운다.

**분석 문서** → `z_research/<셋>/Archive/TOPIC_YYYY-MM-DD.md`.
`z_world_model_analysis/` 는 레포 전역 분석용으로 남긴다.

**⚠️ 실험 케이스마다 프로토콜 yaml 을 만들지 않는다.** 새 yaml 의 기준은 하나다 —
**재는 방식 자체가 다른가?** (다른 index, 다른 표현 지점, 다른 라벨 의미)
그 밖은 전부 아래 셋으로 처리한다 (`configs/protocols/README.md`):

| 장치 | 쓸 때 |
|---|---|
| `fit_groups_sweep: auto` | 데이터셋마다 조건 구성이 다를 때. group 을 읽어 sweep 자동 생성 |
| `SET="a.b=1 c.d=null"` | 일회성 변형 (epoch, target 하나만, sweep 지정). `null` = 키 삭제 |
| `extends:` | 구조가 진짜 다른 실험 |

실제로 `attn_probe_flat.yaml`(sweep 만 다름) · `attn_probe_e50_staticocc.yaml`(optim/runs/targets 만
다름)을 이렇게 없앴다. 프로토콜은 4개로 유지된다.

### 8-5. 설계 결정 — 다시 교란으로 제기하지 말 것 (2026-08-30)

두 대비는 **의도된 조작**이다. 여러 번 "교란" 으로 제기했다가 매번 같은 답을 받았다.

| 축 | 무엇을 비교하는가 | 함께 바뀌는 것 |
|---|---|---|
| **가림 유무** (`visible` vs `occlusion`) | **물체가 가려지는가** | 가림막이 장면에 있는가 |
| **지면** (`moving_flat` vs `moving`) | **등속 vs 등가속도** | 쐐기가 장면에 있는가 |

두 경우 모두 "장면 구성이 함께 바뀐다" 는 **분리 대상이 아니다.** 물체를 가리려면
가림막이 있어야 하고, 중력으로 가속시키려면 경사면이 있어야 한다. 조작의 구현이지 교란이 아니다.

**따라서 쓸 수 있는 주장과 못 쓰는 주장:**

- ✅ "물체가 가려지면 무너진다. 가려지는 **시간**은 무관하다" (k=1~4 평평)
- ✅ "등가속도가 등속보다 나쁘다"
- ❌ "가림막이 화면에 있는 것 자체는 영향이 없다" — 안 쟀다
- ❌ "가속만이 원인이고 장면 구성은 무관하다" — 안 쟀다

k 축(1~4)은 **가림막이 있는 상태에서** 시간만 바꾸므로 이 제약이 없다. **"시간은 무관하다"
는 주장이 논문의 핵심이고, 그건 k 축 안에서 깨끗하게 성립한다.**

### 8-2. 문서 규칙
- 하단에 반드시 **`## 재현`** 절: 정확한 명령줄 + config 경로
- **철회/정정을 지우지 않고 문서 안에 남긴다.** `⚠️ 정정 —`, `**철회.**`, `## 폐기한 가설들 (기록)` 표. "다시 시도하지 말 것"까지 적는다
- 낡은 문서는 파일명에 `_PREVIOUS-DATA`를 붙이고 맨 위에 blockquote 경고(재렌더 일시, 산출물 보관 경로, 그때 쓴 캐시 경로)
- 한 문서 안에서 언어를 통일 (심층 분석·종합 노트는 한국어, 결과 카드는 영어가 많다)
- 리포트 생성기가 있으면 하단에 스크립트 경로를 밝힌다

### 8-3. 그림
- **한글 폰트가 없다.** `fc-list :lang=ko`에 비트맵 PCF 3개뿐이라 matplotlib에서 못 쓴다. **플롯 라벨은 반드시 영어.** 주석은 한글 OK
- 탐색용: `DejaVu Sans`. **논문용: `Nimbus Roman`**(Times 계열) + `pdf.fonttype 42` + 벡터 PDF, double-column 7.0in
- 논문 figure 관례: 패널 라벨 `(a)`–`(d)`를 **각 패널 아래 가운데**, 각 패널이 독립 subfigure처럼 x축을 갖는다, 서술 문장은 캡션으로 빼고 그림에는 안 넣는다
- 색은 검증된 팔레트 slot 1–3만: `#2a78d6`(blue) / `#eb6834`(orange) / `#1baf7a`(aqua). **`node`가 없어서 팔레트 validator를 못 돌린다** — 이 3개는 all-pairs 검증 완료로 문서화된 조합이라 안전. aqua는 light surface에서 3:1 미만이라 쓰면 **모든 마크에 직접 라벨** 필수
- 기존 플롯 스크립트: `plot_violation_accuracy.py`, `plot_v8_color_directional_preference.py`, `plot_attn_probe_paper.py`(논문판)

---

## 9. 환경

```
python : /data/hyuntak/anaconda3/envs/vjepa2/bin/python   (3.12.13)
torch 2.12.0+cu126 · GPU 4장 · numpy 2.4.6 · matplotlib 3.11.0 · decord 0.6.0 · timm 1.0.27
없는 것: scikit-learn, seaborn, PyAV, node/npm, 한글 폰트
```
SLURM 스크립트는 `source /data/hyuntak/anaconda3/bin/activate vjepa2` (환경 34개, `CONDA_ENV`로 선택).

### `.gitignore` 주의
`z_scripts`, `*csv`, `*.json`, `*.png`, `*.pt`, `z_exp/`, `data_gen/`, `checkpoint/` 가 **전부 무시된다.**
→ 새로 만든 index·figure는 **untracked가 정상**이고 `git checkout`으로 되돌릴 수 없다. 덮어쓰기 전에 확인할 것.

**추적되는 자리 / 안 되는 자리** — 2026-08-28 에 `z_research` 를 정리하다 추적 0개였던
리포트·figure를 통째로 잃었다. 그 뒤로:

| 자리 | 추적 | 용도 |
|---|---|---|
| `configs/protocols/` | ✅ | 프로토콜 yaml + 레지스트리 md — **여기 쓸 것** |
| `z_research/scripts/` | ✅ | run/resolve/sbatch — **여기 쓸 것** |
| `configs/world_model_analysis/` | ⚠️ 2개만 | 나머지 43개는 2026-08-28 삭제 (복구는 `summary.json.config`) |
| `z_scripts/` | ❌ | `.gitignore` 38번 줄 |
| `z_exp/`, `z_research/*/figures/` | ❌ | 산출물·그림 |

⚠️ **파일 편집에 `sed -i` 를 쓰지 말 것.** 2026-08-28 에 `datasets.md` 에서 `cache_tag`
한 줄을 바꾸려던 `sed` 가 섹션 4개를 통째로 지웠다. 여러 줄짜리 문서는 Python 으로 고칠 것.

---

## 10. 열려 있는 문제 (다음에 할 것)

**우선순위 높음**
1. **12-variation factorial 설계** (`z_findings/attn_probe_and_next_steps.md`) — 가림막 유무 × 물체 갯수(1/2/3) × 움직임 유무. 물체 갯수가 독립 주효과인지, occlusion×motion과의 상호작용인지 분리. IntPhys1에서 n_obj 1/2/3이 86~89%로 평평하고 block별 방향이 제각각이라 필요
2. **IntPhys1에 pooled + 차원별 표준화 L1 적용** — v5의 65.62→87.50% 향상이 v5 특유 아티팩트인지 일반적 채점 개선인지 가르는 가장 중요한 미해결 실험. `distance: pooled_l2`를 eval.py에 구현해야 함
3. **static+occluded shape 근-chance의 직접 검증** — local/post-reveal token surprise vs 현행 global future-token average. 아직 안 함
4. ~~**"가림막은 있는데 물체를 안 가리는" 조건 신설**~~ — **2026-08-30 철회.** "가림 유무" 는 곧 "물체가 가려지는가" 이고 가림막 존재는 그 구현이다. §8-5
5. **가림 길이 통제 세트(`IntPhysGen_v4`)**: lag 3/8/16/24 프레임. **L1이 IntPhys 수준(85~95%)까지 오르는가**
6. **α(mu) 를 설계 지표로 승격** — §5-4c 가 r=0.948 을 보였으니, v10 을 설계할 때
   **블록마다 두 후보 미래의 α(mu) 를 0.5 에 맞추는 것**이 곧 "기하 누수 제거"다.
   렌더 전에 캐시 없이도 못 재지만, 파일럿 배치로 먼저 재고 설계를 고칠 수 있다
7. **probe 클래스 stratify 수정** — `condition|color_pre` 결합 컬럼으로 층화.
   color self 가 낮은 게 표현 탓인지 `yellow` train 28개 탓인지 지금은 못 가른다 (§5-4c)
8. **p self 수렴** — 30ep 로는 `train_acc` 0.83~0.95 에 머물러 DDP 실행마다 최대 20pt 흔들린다.
   `attn_100` 프리셋이 `attn_probe.yaml` 에 이미 있다. 또는 `GPUS=1` 로 결정론 확보

**그 밖**
- color 모순: predictor 정보는 제일 적은데 raw L1은 제일 잘 맞힌다
- 정렬 학습: `W`를 가능 변이로만 학습해 `|W·p − h|`로 채점 / probe 기반 채점
- 역방향 이식(p→h), linear probe 비교, `input: window` 대조군
- ~~2D가 3D보다 낮은 이유~~ — **프레임과 생성 코드가 사라져 재렌더 불가.** 남은 41G 토큰 캐시로
  할 수 있는 캐시 기반 분석(α, token evidence)만 가능 (§4-3)
- Jongseo 벤치마크의 permanence-noemerge/solidity/gravity가 전 generator에서 chance 이하 → V-JEPA2 구조적 약점 가능성
- 토큰평균은 시간축(8 tubelet)·공간축(256 위치)을 둘 다 버린다 → **위치·궤적 위반은 원리적으로 못 잡는다.** 축별 풀링 미측정

---

## 11. 데이터를 새로 설계할 때 지킬 것

- **위반이 "관계"로만 존재하면 안 된다.** splice(정상 A의 앞 + 정상 B의 뒤)는 두 반쪽이 각각 자연스러워 `π_imp ≈ π_pos`가 되고 α가 0.5에 붙는다
- **"물체 하나 = 화면 움직임의 전부" 상태를 깰 것.** `|dc|/전체 내용`을 200% → 10% 아래로. distractor 추가 + **가림물을 크게 움직이게**(가장 쌈)
- **distractor 속도를 강제할 것** (`|v| > 100 cm/s`). v3는 ±가 상쇄돼 상당수가 사실상 정지였고, 정지 물체는 시간 중앙값 배경에 흡수된다
- **가림 길이는 가림막 크기가 아니라 물체 속도/타이밍으로 조절**. v4는 세 축이 얽혀 분리 불가였다
- **block 안에서 예측 구간 내용량을 맞출 것** (A/B 비대칭은 누수 이용이라 금지)

---

## 12. 작업 스타일 (이전 세션에서 굳어진 것)

- **지시받은 실험을 비판적으로 평가한다.** 시키는 대로 돌리기 전에 **"이게 논문 방향성에 맞는가"**
  를 먼저 묻는다. 사용자가 요청한 실험이라도 다음이면 **돌리기 전에** 말한다:
  - 이미 답이 나온 것 (예: "세기를 키우면 되나?" 는 v10 에서 기각됐다 — §8-5 아래 GEOMETRY 문서)
  - 결과가 어느 쪽이든 논문의 beat 를 못 세우는 것
  - 교란이 남아 그 결과로는 주장을 못 하는 것
  - 더 싼 방법으로 같은 답이 나오는 것 (attentive 2시간 vs ridge 몇 초)
  - 통계적으로 결론이 안 나오는 표본 크기

  **동의만 하고 돌리는 것이 가장 큰 실패다.** 하루에 GPU 수 시간을 그렇게 태운 적이 있다.
  반대로 사용자가 이유를 대면 그건 결정이다 — 한 번 말하고 진행한다 (§8-5 가 그 예).

- **현재 논문 방향은 `z_research/IntPhysGenV10/Archive/PAPER_STORY_*.md` 에 있다.**
  실험을 제안하거나 평가할 때 **어느 beat 를 세우는지 명시**한다. 어느 beat 에도 안 붙으면
  그렇게 말한다. beat 별 상태표(✅/⚠️)를 보고 **비어 있는 것부터** 제안한다.

- **기존 하네스를 먼저 찾는다.** `configs/world_model_analysis/`에 유사 실험 yaml이, `z_scripts/world_model_analysis/`에 `run.sh`가 이미 있다. 새 진입점·새 스크립트 규칙을 만들기 전에 반드시 확인할 것 (실제로 한 번 `PROCS` 같은 걸 새로 발명했다가 되돌렸다)
- **모델 로딩이 필요 없는 검증은 따로 떼서 몇 초에 한다.** 스모크는 프로세스마다 ViT-H를 NFS에서 새로 읽어 ~2분씩 쓴다
- **수치는 문서가 아니라 산출물(`summary.json`)에서 재확인**하고 그렇게 했다고 밝힌다
- **모순되는 결과를 발견하면 한쪽을 지우지 말고 조건 차이를 표로 명시한다** (§6, §5-2의 sliding 사례)
- **단서(caveat)를 결론과 같은 비중으로 쓴다** — n이 작아서 재확인이 필요한 값, 분리 안 된 교란은 반드시 문서에 남긴다
- **파일 편집에 `sed -i` 를 쓰지 않는다.** 여러 줄짜리 md/yaml 은 Python 으로 고친다 (§9)
- **새로 만드는 스크립트·config 는 git 추적되는 자리에 둔다** (`configs/`, `z_research/scripts/`).
  추적 0개였던 `z_research/IntPhys-Like_data/` 를 통째로 잃은 적이 있다
- 사용자는 한국어로 소통한다. 코드 식별자·경로·플롯 라벨은 원문 유지
