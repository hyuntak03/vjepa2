# analysis/probing — z / p / h 세 지점 attentive probing

**목적.** context encoder (z, frames 1–16) · predictor 출력 (p, 17–32) · target encoder (h, 17–32) 에서 정체성 (shape · color · env) 을 읽어 "정보가 있는가 / 같은 좌표계인가" 를 잰다. 세 지점이 같은 shape (2048 tokens) 라 직접 비교된다.
프로토콜 셋: `attn_probe` (self 만) · `attn_probe_xfer` (h↔p 양방향 이식, `extends: attn_probe.yaml`) · `attn_probe_imp` (불가능 변이에서 target encoder 가 바뀐 정체성을 읽는가 — 라벨 의미가 다른 **별도 프로토콜**). 확립된 수치 CLAUDE.md §5-2. 옛 `attn_probe*` 와 config 가 같다 (check_equivalence 3 행 OK, imp 는 양쪽 다 index 없어 SKIP).

## 명령
```bash
GPUS=8 bash cvpr/analysis/probing/run.sh v11 vith                                  # attn_probe, fit_groups_sweep auto_conditions (v11 6 조건)
PROTO=attn_probe_xfer GPUS=8 bash cvpr/analysis/probing/run.sh v11_full vith       # 이식 (sweep 12)
SET="probing.optim=attn_50 probing.targets.shape=null" GPUS=8 bash cvpr/analysis/probing/run.sh v11 vith   # 일회성 변형 — yaml 을 새로 만들지 않는다
WINDOW=c8t32 GPUS=8 bash cvpr/analysis/probing/run.sh v11 vith                     # 문맥 8 → 캐시·결과가 _wc8t32 로 갈린다
LIMIT=4 SET="probing.optims.attn_30.num_epochs=1 probing.targets.color=null probing.targets.env=null probing.fit_groups_sweep=[null]" GPUS=1 bash cvpr/analysis/probing/run.sh v11 vith   # 배관 점검 (pooled head 하나, sweep=1 — DRYRUN 확인)
```

## 노브
| env / SET 키 | 기본 | 의미 |
|---|---|---|
| `PROTO` | `attn_probe` | `attn_probe` · `attn_probe_xfer` · `attn_probe_imp` (`cvpr/analysis/probing/<PROTO>.yaml`) |
| `$1` `$2` / `DATASETS` `MODELS` | `v11` / `vith` | `index_probe.csv` 가 있어야 한다 (아래) |
| `WINDOW` / `SET="window.context=8"` | `c16t32` | `data.n_frames` · `features.context_length` · `model.window_size`. 창이 다르면 **tag 가 갈려 캐시를 다시 뽑는다** (캐시 서명이 문맥 길이를 안 본다 — `cvpr/README.md` §6) |
| `SET="probing.optim=attn_50"` · `probing.targets.<t>=null` · `probing.fit_groups_sweep=auto` | `attn_30` · shape/color/env · `auto_conditions` | optim 프리셋 · target 빼기 · pooled head 되살리기 |
| `SET="probing.optims.attn_30.num_epochs=1"` | `35` | 배관 점검용. **runs 는 줄이지 말 것** — 캐시에 뽑히는 표현이 줄어든다 |
| `LIMIT=N` · `RECACHE=1` | — | block N 개 + `_smokeN` (캐시도 `_smokeN` 이름) · 캐시 무시 |
| `GPUS` | `1` | ⚠️ DDP probing 은 실행마다 흔들린다 (p self 최대 20 pt). 결정론이 필요하면 `GPUS=1`. probing `batch_size` 는 **global** |
| `RESULTS_ROOT=legacy` | `cvpr` | `z_research/IntPhysGenV11/exp_results/attn_probe__v11_vith` (실물 있음) |

## 결과 위치
`$CVPR_RESULTS/analysis/probing/<PROTO>__<ds>_<model>[_w<창>][_smokeN]/` — `summary.json` (`probing[]`: fit · groups · target · evals.<source>.overall/per_group) · `predictions.json` · `_resolved.yaml`.
토큰 캐시 `$CVPR_CACHE/<tag>/` (= `/local_datasets/world/world_analysis/cache/<cache_tag>_<model>[_w<창>]`, `features.cache_dir`). v11 한 벌 ≈ 420 GiB, **노드 로컬**. 옛 캐시 이름 `v11_vith` 와 같아 그대로 재사용된다 (`video_ids` 완전 일치 조건).

## 옛 명령 대응
| 옛 | 새 |
|---|---|
| `GPUS=8 bash z_research/scripts/run.sh attn_probe v11 vith` | `GPUS=8 bash cvpr/analysis/probing/run.sh v11 vith` |
| `… run.sh attn_probe_xfer v11_full` / `attn_probe_imp v10` | `PROTO=attn_probe_xfer … v11_full` / `PROTO=attn_probe_imp … v10` (v10 은 available: false) |
| `sbatch --export=ALL,WMA_RUN=1,P=attn_probe,D=v11,GPUS=8 z_research/scripts/sbatch.sh` | `GPUS=8 bash cvpr/harness/submit.sh cvpr/analysis/probing/run.sh v11 vith` |
| `SPLIT="shape color env" … sbatch.sh` + `merge_probe_runs.py` | **없다** — target 별 job 분할은 `SET="probing.targets.color=null probing.targets.env=null" OUTDIR=…` 로 손으로 나누고 옛 `merge_probe_runs.py --base` 로 합친다 (미검증) |

## 단서 · 미완
- `index_probe.csv` 가 있는 등록 데이터셋 (data_csv 실물): `v11` `v11_full` `v11_earlymid` `v11_timing` `v11_vanish_all` `rollout_v2` `rollout_v2_decel` `rollout_v2_training_v5/v6/v8` (+ available: false 인 v8 · v10 · v10_flat · v13_black · v11_occtiming · 2d_v8_transit). 없으면 resolve 가 "index 가 없다" 로 죽는다.
- `attn_probe_imp` 는 `index_probe_imp.csv` 가 `v10` 에만 있다 (available: false) → 지금 돌릴 데이터셋이 없다 (`v11` 은 DRYRUN 에서 index 없음으로 죽는 것을 확인).
- v11 전수 (63 head) 는 8 시간 벽을 넘긴다 — 옛 `_prep` + `afterok` 분할 흐름은 새 공간에 없다. `TIME=` 을 늘리거나 target 별로 손으로 나눈다.
- 2026-10-09 vll3 smoke (`smoke_all.sh` 의 probing 줄, `LIMIT=4` + `auto_conditions`): **rc=1** — `LIMIT=4` 가 앞 4 block (전부 `moving_occlusion_flat`) 만 남기는데 `fit_groups_sweep: auto_conditions` 는 전체 index 의 6 조건으로 펴져 5 조건의 학습셋이 0 이라 `eval.py:268` 이 죽었다 (`cvpr/logs/smoke_vll3.log`). 엔진 문제가 아니라 smoke 설정 문제 — 위 명령처럼 `probing.fit_groups_sweep=[null]` 을 더하면 sweep=1 이 된다 (DRYRUN 확인, GPU 실행은 미검증). `cvpr/harness/smoke_all.sh` 의 probing 줄은 그 뒤 `LIMIT=12` + `probing.fit_groups_sweep=[null]` 로 바뀌었고 vll3 재실행 **rc=0** (65 s, `attn_probe__v11_vith_smoke12/summary.json` · `predictions.json`, `cvpr/logs/smoke_vll3_probing.log`) — 2026-10-09 검증 때 확인.
