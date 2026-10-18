# z_training — 얼린 V-JEPA 2 encoder 위에서 predictor 학습

> **처음이면 이 세 줄만:**
> 1. 무엇을 왜 학습했고 결과가 어땠나 → **[`VARIANTS.md`](VARIANTS.md)**
> 2. 지금 무엇이 문제이고 어디로 가나 → **[`docs/PROBLEMS_AND_DIRECTION_2026-10-18.md`](docs/PROBLEMS_AND_DIRECTION_2026-10-18.md)**
> 3. 학습 · 채점 명령 → 아래 §2

---

## 1. 지금 어디까지 (2026-10-18)

| | |
|---|---|
| 가장 좋은 모델 | `natural_ar_vw_state_op` — IntPhys1 92.8 · GRASP 71.3 (기준 AR `natural_ar_vw_state` 90.0) |
| 가장 큰 문제 | 예측이 문맥의 **평균**으로 무너진다 → 움직임 위축, 움직이는 물체 영속 없음 |
| 방금 끝난 것 | residual gating — AR 보다 나빴다 ([`docs/RESULTS_GATING_2026-10-18.md`](docs/RESULTS_GATING_2026-10-18.md)) |
| 지금 도는 것 | `natural_ar_vw_state_vidctx` — 영상 문맥 · 영상 정답 · TF 문맥만 |
| 성적표 | [시트](https://docs.google.com/spreadsheets/d/1b2CyYWqIFUNbPUB12Tx_BxtagImkLwQB1F1dNQ4iIiU) (Overall · Intuitive Physics) |

## 2. 명령

```bash
cd /workspace/world_model/vjepa2

# 학습
GPUS=8 bash z_training/tmux_train.sh <config>                         # configs/training/<config>.yaml
GPUS=8 SET="model.predictor.gating_residual=false" NAME=<run> bash z_training/tmux_train.sh <config>   # 키 하나 바꾼 ablation
DRYRUN=1 bash z_training/train.sh <config>                            # 병합 · 검사만 (몇 초)

# 감시
watch -c -n 2 bash z_training/monitor_ar.sh <run>

# 채점 (run config 의 predictor 키를 자동으로 넘긴다)
bash z_training/eval_bench.sh <run> latest.pt                         # IntPhys1 A.8 + IntPhys 2
NOWAIT=1 RUNS=<run> BENCHES="inflevel_continuity inflevel_solidity inflevel_gravity grasp_level2" bash z_training/chain_bench.sh
RESULTS_ROOT=legacy GPUS=8 bash cvpr/eval/testbed/run.sh vobench_v1_late <레지스트리 모델>   # cvpr/registry/models.yaml

# 표 · 검사
/venv/vjepa/bin/python z_training/tools/variants_table.py             # 모든 run 의 설정 · 성적 (산출물에서 다시 계산)
CUDA_VISIBLE_DEVICES= /venv/vjepa/bin/python z_training/tests/gating_ctx_block_cpu_test.py
```

⚠️ **돌고 있는 `.sh` 를 고치지 말 것** — bash 는 읽어 가며 실행해서 뒷부분이 깨진다 (10-18 에 두 번 겪었다).

## 3. 문서 지도

| 문서 | 한 줄 |
|---|---|
| [`VARIANTS.md`](VARIANTS.md) | **모든 학습 variant** — 철학 · 구조 그림 · run 표 · 카드 · 채점 정의 · 조립 키 |
| [`docs/PROBLEMS_AND_DIRECTION_2026-10-18.md`](docs/PROBLEMS_AND_DIRECTION_2026-10-18.md) | 능력별 (읽기 · 전이 · 지속 · 영속) 무엇이 안 되나, 문제 6 개, 방향 5 개 |
| [`docs/RESULTS_GATING_2026-10-18.md`](docs/RESULTS_GATING_2026-10-18.md) | residual gating 구조 · 학습 · 결과 · 진단 |
| [`docs/RESULTS_2026-10-16.md`](docs/RESULTS_2026-10-16.md) | 릴리즈 GRASP · op 데이터 · ctxt_video (영상 문맥) |
| [`docs/RESULTS_2026-10-15.md`](docs/RESULTS_2026-10-15.md) | AR 후속 — 힌트 · 차이 L1 · VOBench 학습 · GRASP · InfLevel · EK100 |
| [`docs/RESULTS_2026-10-12.md`](docs/RESULTS_2026-10-12.md) | kind=ar 가변 창 · 힌트 · 상태 되먹임 (AR 처음으로 IntPhys1 90) |
| [`docs/RESULTS_2026-10-01.md`](docs/RESULTS_2026-10-01.md) | Vast 이전 · tube 기준선 82.2 · rollout 만 학습은 우연 |
| [`docs/RESULTS_2026-09-19.md`](docs/RESULTS_2026-09-19.md) | 릴리즈 predictor post-FT 세 번 — 도메인 안에서만 오른다 |
| [`docs/HARNESS_2026-09.md`](docs/HARNESS_2026-09.md) | 옛 README — 런처 · 함정 · v11 분할 설명 (명령은 아직 유효) |
| [`../z_training_config/`](../z_training_config/) | 설정 결정 기록 (왜 그 config 인가) |
| [`../configs/training/`](../configs/training/) | config yaml (각 파일 맨 위 주석에 설계 이유) |

## 4. 폴더

| 위치 | 무엇 |
|---|---|
| `tmux_train.sh` · `train.sh` · `harness/` | 학습 진입점 · 병합기 (`resolve_train.py`) · 런처 (`launch.py`) |
| `eval_bench.sh` · `eval_bench_summary.py` · `chain_bench.sh` · `chain_*.sh` | 채점 · 요약 · 학습 뒤 채점 묶음 |
| `monitor_ar.sh` · `monitor.sh` | 감시 |
| `tools/variants_table.py` | variant 표 |
| `tests/` | CPU 구조 검사 |
| `data/` | 인덱스 · fps4 표 만들기 |
| `runs/<run>/` | `config.yaml` · `latest.pt` · `e{N}.pt` · `train.log` · `metrics.jsonl` · `eval/` (gitignore) |
| 학습 코드 | `app/vjepa_frozen/` (`train.py` · `ar.py` · `var_window.py` · `utils.py`) · `src/models/rollout_predictor.py` |
