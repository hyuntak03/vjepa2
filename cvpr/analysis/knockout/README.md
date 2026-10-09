# analysis/knockout — predictor attention knockout (cvpr 하네스)

> 엔진 (무엇을 끊고 무엇을 재나 · 명세 문법 · 검증 플래그 · 단서) 은 `analysis/attention/knockout/README.md` 가 정본이다.
> 관찰 짝 (`predictor_attn`) 과의 관계는 `analysis/attention/README.md`. 레포 규칙은 루트 `CLAUDE.md`, cvpr 계약은 `cvpr/env.sh` · `cvpr/harness/`.
> 이 폴더는 **그 엔진을 cvpr 코드 스페이스에서 같은 의미로 띄우는 얇은 껍데기**다 — 엔진 코드는 한 줄도 안 고쳤다.

## 목적

predictor 안의 attention 간선 (질의집합 → 키집합 × 층) 을 끊고 **확립된 채점** (matched pair surprise 정확도, `score_blocks` 재사용) 이
얼마나 떨어지는지로 information flow 를 잰다. 옛 진입점 `analysis/attention/knockout/run_sharded.sh` (GPU N 장 `--shard` + `merge.py`) 와
**같은 명령·같은 config** 를 cvpr 레지스트리 (`cvpr/registry/*.yaml`) · 창 preset · `SET=` · 이름 규칙으로 돌린다.

흐름 (`run.sh`):

```
cvpr/harness/resolve.py protocol.yaml <ds> <model> [WINDOW/SET/LIMIT/...]  ->  <out>/_resolved.yaml + _meta.json
TAG=<tag> OUTDIR=<out>  python -m analysis.attention.knockout.run --protocol <out>/_resolved.yaml --shard i/N --device cuda:g   x N (병렬)
python -m analysis.attention.knockout.merge -o <out>                                                   ->  <out>/results.json + per_video.npz
```

`run.py` 는 `analysis/attention/predictor_attn/extract.py:resolve_config()` 으로 **옛** resolver (`z_research/scripts/harness/resolve.py`) 를
다시 부른다. 옛 resolver 는 `<protocol>` 자리에 yaml **경로**를 받으면 (그 파일 89–90 행) 그 yaml 위에 옛 md 레지스트리를 한 번 더 깔고
`TAG` / `OUTDIR` env 로 이름을 짓는다. 그래서 run.sh 는 cvpr 가 푼 yaml 의 경로를 `--protocol` 로 넘기고 `TAG`/`OUTDIR` 을 export 한다.
옛 resolver 가 돌려주는 config 가 cvpr 의 것과 **전 키 동일** 한지는 `check.py` 가 검사한다 (아래 "검증").

## 명령

```bash
N=8 bash cvpr/analysis/knockout/run.sh v11_full vith                   # 옛 run_sharded.sh 기본 (BPC=0 전수 · LAYERS=window · W=3 · 간선 3 + @all 3)
BPC=1 N=2 bash cvpr/analysis/knockout/run.sh v11 vith                  # 스모크 (조건당 block 1)
DRYRUN=1 bash cvpr/analysis/knockout/run.sh v11_full vith              # 병합 config + shard · merge 명령만 찍는다 (GPU 0 장, 폴더 안 만든다)
python cvpr/analysis/knockout/check.py [--results legacy]              # run.py 가 보는 config == cvpr config == 옛 run_sharded.sh config (GPU 불필요)
GPUS=8 bash cvpr/harness/submit.sh cvpr/analysis/knockout/run.sh v11_full vith   # SLURM. env (W, EDGES, SPECS ...) 는 --export=ALL 로 통째로 넘어간다 (B64 불필요)
```

## 노브

인자: `run.sh [DATASETS] [MODELS]` (공백 구분 행렬). 없으면 env `DATASETS`/`MODELS`, 그것도 없으면 옛 이름 `D`/`M`, 기본 `v11_full` / `vith`.

| env | 기본값 | 의미 (→ 어디로 가나) |
|---|---|---|
| `N` | `GPUS`, 없으면 **8** | shard 수 = GPU 수. `cuda:0..N-1` (옛 run_sharded.sh 와 같다). submit.sh 아래서는 `GPUS` 를 따른다 |
| `GPU_IDS` | — | 쓸 GPU 를 직접 (`"4 5 6 7"`). 주면 **N = 그 개수** (N 보다 우선). shard i 가 `cuda:<i 번째 id>` 에 간다 |
| `BPC` | 0 | `--blocks-per-cond` 조건당 block 수 (0 = 전부). 표본은 block 단위 (matched pair 가 안 깨지게) |
| `BS` | 4 | `--batch-size` (shard 당) |
| `LAYERS` | `window` | `--layers` each \| window \| all \| prefix \| suffix (콤마로 여러 개) |
| `W` | 3 | `--window` **층 창 폭** (홀수, 콤마 여러 개 `1,3,7`). ⚠️ cvpr 의 `WINDOW` (프레임 창 preset) 과 다른 것이다 |
| `EDGES` | `mask..ctx mask..mask ctx..ctx` | 끊을 간선. 각 간선마다 `--edge <e> --spec <e>@all`. **`EDGES=` (빈 값) 은 간선 스윕 없음** (SPECS 만 돌릴 때; 옛 스크립트는 빈 값이면 기본값이었다) |
| `SPECS` | — | 명세를 통째로 (`--spec`, 공백 구분). 예 `mask..hidden@all mask..hidden_ctrl@all` |
| `COND_RE` | — | `--cond-re` 조건 (block_type) 정규식 필터 (예 `occlusion`) |
| `HCACHE` | — | `--h-from-cache DIR` — target 인코더 대신 `DIR/target.npy`. 캐시 자리는 `$CVPR_CACHE/<tag>`. ⚠️ v11 계열 캐시는 probing 관례 (bf16) 로 뽑혀 쓰면 안 된다 (엔진 README §5) |
| `METRICS` | `surprise_acc,surprise_l1,pred_drift` | `--metrics` |
| `WORKERS` / `OMP_NUM_THREADS` | 8 / 8 | `--workers` 디코드 스레드 / shard 프로세스의 OMP (옛 스크립트 고정값 8) |
| `EXTRA` | — | run.py 의 나머지 인자를 그대로 (`"--seed 1 --no-null-baseline --no-prefix-cache --hidden-csv ..."`) |
| `WINDOW` | 프로토콜 기본 `c16t32` | 프레임 창 preset (`cvpr/registry/windows.yaml`). 바꾸면 tag/폴더에 `_w<이름>` |
| `SET` / `SET_FILE` | — | 점 경로 덮어쓰기 (`resolve.py --set`). **_resolved.yaml 에 구워진다** — run.py `--set` 으로는 안 넘긴다 (같은 값이 두 번 적용될 뿐이라 생략) |
| `LIMIT` | — | `limit: N` + tag/폴더 `_smokeN` (run.py 가 `cfg.limit` 을 읽는다) |
| `SUFFIX` | — | tag · 폴더 접미사 (예 `_hidden`) |
| `TAG` / `OUTDIR` | 자동 | 이름 직접 지정 — resolve.py 와 run.py 안의 옛 resolver 양쪽에 같은 값이 간다 |
| `RESULTS_ROOT` | `cvpr` | `cvpr` \| `legacy` \| `<경로>` — 아래 "결과 위치" |
| `DRYRUN` | — | resolve 만 하고 config · shard 명령 · merge 명령을 찍는다. 결과 폴더를 만들지 않는다 |

`SMOKE` · `RECACHE` 는 knockout 에 의미가 없어 받지 않는다 (토큰 캐시를 안 쓴다 — `HCACHE` 만 예외).

## 결과 위치

| `RESULTS_ROOT` | 폴더 |
|---|---|
| `cvpr` (기본) | `$CVPR_RESULTS/analysis/knockout/<ds>_<model>[_w<창>][<suffix>][_smokeN]/` |
| `legacy` | `<datasets.legacy_results_root>/knockout__<ds>_<model>[...]/` — v11_full → `z_research/IntPhysGenV11_occlusion_timing_ablation/exp_results/knockout__v11_full_vith` (옛 run_sharded.sh 기본과 같다), v11 → `z_research/IntPhysGenV11/exp_results/knockout__v11_vith` (옛 run.py 단일 실행 기본과 같다) |

폴더 안: `_resolved.yaml` · `_meta.json` (cvpr 관례) · `logs/shard<i>.log` · `logs/merge.log` · `shards/shardXXofNN.{json,npz}` ·
`results.json` + `per_video.npz` (merge.py). `results.json.verify.resume_vs_forward_max_abs_diff` 가 `0.0` 이어야 유효하다 (엔진 README §3).
그림은 옛것 그대로: `python -m analysis.attention.knockout.plot --results <폴더>/results.json --outdir ...`,
축별 그림 `z_research/scripts/figures/plot_v11_knockout.py` (breakdown 에서).

## 옛 명령 대응표

| old | new |
|---|---|
| `bash analysis/attention/knockout/run_sharded.sh` (D=v11_full M=vith N=8) | `N=8 RESULTS_ROOT=legacy bash cvpr/analysis/knockout/run.sh v11_full vith` (폴더까지 같음) / `RESULTS_ROOT` 없이면 `$CVPR_RESULTS/analysis/knockout/v11_full_vith` |
| `D=v11 M=vitl BPC=24 N=4 bash .../run_sharded.sh` | `BPC=24 N=4 bash cvpr/analysis/knockout/run.sh v11 vitl` |
| `OUT=/tmp/x BPC=1 N=2 bash .../run_sharded.sh` | `OUTDIR=/tmp/x BPC=1 N=2 bash cvpr/analysis/knockout/run.sh v11_full vith` |
| `W=1 OUT=.../knockout_w1__v11_full_vith bash .../run_sharded.sh` | `W=1 SUFFIX=_w1 bash cvpr/analysis/knockout/run.sh v11_full vith` (폴더 `..._w1`; 옛 폴더 이름 그대로 쓰려면 `OUTDIR=`) |
| `SPECS="mask..hidden@all mask..hidden_ctrl@all" COND_RE=occlusion OUT=.../knockout_hidden__... bash .../run_sharded.sh` | `EDGES= SPECS="mask..hidden@all mask..hidden_ctrl@all" COND_RE=occlusion SUFFIX=_hidden bash cvpr/analysis/knockout/run.sh v11_full vith` (`EDGES=` 를 비워야 간선 스윕이 안 붙는다 — 옛 스크립트는 비울 수 없어 기본 간선 3 이 같이 돌았다. 옛 `knockout_hidden__` 폴더가 어느 쪽으로 돌았는지는 **미검증** — 그 `results.json.specs` 를 볼 것) |
| `sbatch --export=ALL,N=8,W=3,LAYERS_B64=...,EDGES_B64=... analysis/attention/knockout/sbatch_knockout.sh` | `GPUS=8 W=3 LAYERS=each,window EDGES="..." bash cvpr/harness/submit.sh cvpr/analysis/knockout/run.sh v11_full vith` (콤마·공백 값 그대로 — submit.sh 는 env 를 통째로 넘긴다) |
| `python -m analysis.attention.knockout.run --dataset v11 --model vith --set data.type_column=condition ... --shard i/N -o OUT` (직접) | run.sh 가 shard 마다 치는 명령이 이것이다 (`DRYRUN=1` 로 찍어 본다). 다른 점은 `--protocol` 이 이름 대신 `<out>/_resolved.yaml` 경로이고 `--set` 은 yaml 에 구워져 있다는 것 |
| `python -m analysis.attention.knockout.merge -o OUT` | run.sh 끝에서 자동. 따로 치려면 그대로 (`-o <결과 폴더>`) |

## 검증 (2026-10-09, vll5, GPU 0 장)

| 무엇 | 어떻게 | 결과 |
|---|---|---|
| 문법 | `bash -n run.sh` · `ast.parse(check.py)` · `yaml.safe_load(protocol.yaml)` | OK |
| 병합 config · shard 명령 | `DRYRUN=1 bash run.sh v11_full vith` / `DRYRUN=1 GPU_IDS="4 5" BPC=1 LIMIT=2 EDGES= SPECS=... COND_RE=occlusion RESULTS_ROOT=legacy WINDOW=c8t32 SET=... EXTRA=... bash run.sh v11 vitl` | config 가 testbed (surprise_c16t32) 와 같은 값 + `type_column: condition`; shard 8 개 / 2 개 (cuda:4, cuda:5) · merge 명령이 찍힘; 폴더 안 생김 |
| **run.py 가 보는 config == cvpr config** | `check.py` — `extract.resolve_config(<cvpr _resolved.yaml 경로>, ds, model, [])` (run.py 가 쓰는 바로 그 함수, TAG/OUTDIR = meta) vs cvpr yaml, dict 평탄 비교 | v11_full/vith · v11/vith · v11/vitl × {cvpr, legacy} **43 키 전부 동일** (tag · output_dir 포함) |
| **cvpr config == 옛 run_sharded.sh config** | `check.py` — `resolve_config("surprise_c16t32", ds, model, ["data.type_column=condition"])` vs cvpr yaml | 41 키 동일 (이름 2 키는 설계상 다름). `--results legacy` 면 옛 run.py 의 outdir 파생값 (`parent/knockout__<ds>_<model>`) 과 이름까지 동일 |
| 실제 shard 실행 · merge 완주 · 수치 | — | **미검증** (이 작업은 GPU 실행 금지). 엔진 코드와 인자가 옛것과 같으므로 의미는 같아야 하나, 돌려서 `results.json.verify` 를 봐야 한다 |

## 단서 / 미완

- **run.py 가 옛 md 레지스트리를 요구한다.** `--protocol` 에 yaml 경로를 줘도 옛 resolver 는 `<ds>` / `<model>` 을 `configs/protocols/{datasets,models}.md` 에서 찾는다
  (없으면 죽는다). cvpr 레지스트리에만 있는 데이터셋·모델 (`cvpr/registry/*.yaml` 에만 추가한 것) 은 resolve 는 되지만 **run.py 에서 죽는다**.
  → 아래 "엔진 수정 제안". 지금 등록된 v11 · v11_full · vith · vitl 은 양쪽에 다 있다.
- 옛 resolver 는 그 위에 md 를 다시 깔 때 **MODEL_OWNED 키 (checkpoint · arch_name · img_size · window_size · predictor ...) 는 md 가 이긴다**.
  `SET="model.window_size=..."` 처럼 그런 키를 바꾸면 md 에 그 키가 있는 모델에선 되돌아갈 수 있다 (vith/vitl md 는 arch_name · checkpoint 만 있어 해당 없음).
  바꾼 SET 으로 돌리기 전에 `python check.py <ds> <model> --set ...` 으로 A == B 를 확인할 것.
- `results.json.meta.protocol` 이 옛 실행에선 `surprise_c16t32` 였고 지금은 `<out>/_resolved.yaml` 경로다. `plot.py` · `plot_v11_knockout.py` 는 이 값을 읽지 않는다 (grep 으로 확인). merge.py 의 shard 간 config 일치 검사는 통과한다 (같은 yaml).
- `run.py` 가 shard 마다 옛 resolver 의 요약 (protocol/dataset/model/frames ...) 을 stdout 에 찍는다 — `logs/shard<i>.log` 첫 줄들이 그것이다. 거기 `protocol : _resolved` 로 보이는 게 정상.
- **run.sh 가 env → `resolve.py` 인자 대응 (WINDOW/SET/SET_FILE/LIMIT/SUFFIX/TAG/OUTDIR/RESULTS_ROOT) 을 `cvpr/harness/launch.sh` 와 같은 줄로 들고 있다.**
  엔진이 launch.sh 의 세 분기 (evals.main / torchrun) 중 어디에도 없어 launch.sh 를 못 쓴다. 병합·--set·이름 규칙 자체는 resolve.py 의 것이고 여기엔 없다.
  launch.sh 가 그 대응을 바꾸면 여기도 같이 바꿔야 한다 (DDP 포트 탐색은 필요 없다 — shard 는 독립 프로세스).
- `EDGES` 빈 값 허용은 옛 run_sharded.sh 와 다른 점이다 (옛것은 `${EDGES:-기본}` 이라 비울 수 없었다).
- 옛 `run_sharded.sh` 의 `OUT` 기본은 D 와 무관하게 `IntPhysGenV11_occlusion_timing_ablation/exp_results/knockout__<D>_<M>` 였다.
  `RESULTS_ROOT=legacy` 는 데이터셋의 `legacy_results_root` 를 따르므로 v11_full 에서만 그 폴더와 같고, v11 은 `IntPhysGenV11/exp_results/` 로 간다
  (옛 run.py 단일 실행의 기본 자리 — 기존 `knockout__v11_vith` 가 거기 있다). 옛 폴더를 꼭 맞추려면 `OUTDIR=`.
- 옛 엔진 README 의 "W=7 기본" 은 스크립트 (`W=${W:-3}`) 와 다르다. 여기 기본은 스크립트 값 3 을 따랐다.
- 미완: cvpr 결과 폴더 (`$CVPR_RESULTS/analysis/knockout/`) 로 실제 완주한 실행이 아직 없다. 첫 실행은 `BPC=1 N=2 LIMIT=` 없이 `BPC=1` 스모크 → `results.json.verify.resume_vs_forward_max_abs_diff == 0` 확인 → 전수.

## 엔진 수정 제안

(이 작업에서는 하지 않았다 — 엔진 폴더는 읽기 전용)

1. `analysis/attention/predictor_attn/extract.py:resolve_config()` — `protocol` 이 `.yaml` 경로이고 그 안에 `tag` · `output_dir` 이 이미 있으면 **다시 resolve 하지 않고 그대로 읽게** 하는 분기.
   그러면 옛 md 레지스트리 의존 (위 단서 1) 과 MODEL_OWNED 되돌림 (단서 2) 이 사라지고 `TAG`/`OUTDIR` export 도 필요 없어진다.
   대안: `RESOLVE` 를 env (`KO_RESOLVER`) 로 바꿔 `cvpr/harness/resolve.py` 를 가리키게 — 단 cvpr resolver 는 인자가 다르므로 (`--results`, `--meta`) 호출부도 바뀐다.
2. `knockout/run.py` — `meta.protocol` 에 경로 대신 `Path(a.protocol).stem` 또는 cvpr `_meta.json` 의 `protocol` 을 적으면 옛 산출물과 메타 모양이 같아진다 (읽는 쪽은 없으므로 급하지 않다).
