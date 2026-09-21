# IntPhys 2 프로토콜 (Bordes et al.) — 시작점

> 전면 재작성 2026-09-21. **Garrido 프로토콜과 다른 자다.** 섞지 말 것
> (`z_research/Benchmarks/PROTOCOLS.md` §7 에 차이표).

## 0. 30초 요약

| 축 | 값 |
|---|---|
| 데이터 | IntPhys 2 **Main** — 253 scene × 4 영상 = 1012 영상 → **506 쌍** |
| 지표 | **쌍 비교 AvgSurprise** 하나 (D.1 *"We use average surprise unless specified otherwise"*) |
| 창 | `frames_per_clip` = C + M, **6 fps 고정**, stride 2, `num_frames_to_pred: -1` (창 끝까지 예측) |
| 특징 | **`max_context_mode`** — 맨 앞 창에서 C 를 2,4,…,C−2 로 키워 곡선 앞부분을 메운다 |
| 격자 | V-JEPA 계열 창 **16/32/48** · VideoMAEv2 창 16 고정 + fps 변경 (D.3) |
| 보고 | 열(Easy/Medium/Hard/Overall)마다 **최고 실행** (Table 2 캡션) |

## 1. 실행 — 진입점은 둘

```bash
# 격자 전체 (권장)
GPUS=8 bash analysis/intphys2/run_grid.sh
MODELS="vith" WINDOWS="16 32" bash analysis/intphys2/run_grid.sh

# 한 번만
PYTHONPATH=$PWD torchrun --nproc-per-node=8 -m analysis.intphys2.eval \
  --config analysis/intphys2/configs/bench_intphys2_main_vith.yaml \
  --set surprise.window_size=32 --set 'surprise.context_length_sweep=[8,12,16,20,24,28]'
```

**`--set` 이 있으므로 실험마다 yaml 을 뜨지 않는다** (`configs/README.md` 규칙과 같다).
점 경로 · YAML 파싱 · `null` 은 키 삭제 — `z_research/scripts/run.sh` 의 `SET=` 과 같은 규약.

## 2. config — 모델당 하나

| 파일 | 창 | C |
|---|---:|---|
| `configs/bench_intphys2_main_vith.yaml` | 48 | [12,18,24,30,36,42] |
| `configs/bench_intphys2_main_vjepa21g.yaml` | 48 | 〃 |
| `configs/bench_intphys2_main_videomae2g.yaml` | 16 | [4,6,8,10,12,14] |
| `configs/intphys2_TEMPLATE.yaml` | — | 새 실험 시작점 |

**C = 창 × {¼, ⅜, ½, ⅝, ¾, ⅞}.** 공식 `vjepa_2.yaml`(M=48)과 `videomaev2.yaml`(M=16)이
같은 비율이라 이것이 정본이다. 창을 바꾸면 `run_grid.sh` 가 C 를 그 비율로 다시 만든다.

⚠️ **논문 D.3 텍스트는 "4,6,8,10,12,14 for VideoMAEv2, V-JEPA and V-JEPA 2" 라고 적는데**
공식 `vjepa_2.yaml` 은 [12,…,42] 다. **코드를 따랐다** (텍스트는 M=16 의 값을 인용한 것으로 본다).

`configs/_superseded/` 는 일회성 실험 config, `configs/vjepa2_vith_intphys2_maintest_*` 는
**z_training 의 post-FT 채점 기록**이라 남겨 둔다.

## 3. 채점 — 쌍은 `type` 컬럼이 정한다

`metadata.csv` 의 `type` ∈ {`1_Possible`, `1_Impossible`, `2_Impossible`, `2_Possible`} (각 253개).
**번호가 곧 쌍**이다 (`dataset.py:109  pair_id = type.split("_")[0]`).

⚠️ **2×2 cross 로 채점하면 안 된다.** 논문 Figure 1:
> *"**The presence of an obstacle or occluder determines the outcome**: a possible outcome in the
> first pair becomes impossible in the second, and vice versa."*

쌍 1 과 쌍 2 는 **장애물 유무가 다르다.** 가로질러 비교하면 위반 때문인지 장면 구성 때문인지
분리되지 않는다 (Garrido 가 IntPhys1 에서 matched pairing 만 쓰는 것과 같은 이유).

난이도는 `Difficulty` 컬럼이고 **`Unknown` 이 172 영상(86쌍) 있다 — 공식 metadata 가 그렇다.**
그래서 Easy/Medium/Hard 합(52+200+168=420)이 전체 506 과 다르다. 버그가 아니다.

## 4. 대조값 (논문 Table 2)

| 모델 | Easy | Medium | Hard | **Overall** | Held Out | IntPhys 1 |
|---|---:|---:|---:|---:|---:|---:|
| **V-JEPA 2-h** | 54.00 | 58.50 | 59.38 | **57.51** | 56.40 | 87.22 |
| V-JEPA-h + RoPE | 52.00 | 53.00 | 57.42 | 53.75 | 54.65 | 98.30 |
| VideoMAEv2-g | 46.00 | 58.50 | 52.73 | 53.75 | 53.49 | 59.40 |

> Table 2 캡션: *"Most of the models were run a dozen of time with a different set of
> hyper-parameters. For a given model, we only report **its best run for a given column**."*

⚠️ **IntPhys 1 열(87.22)은 IntPhys 2 프로토콜로 잰 값이다.** Garrido 프로토콜 값이 아니다.
우리 IntPhys 1 값은 `z_research/Benchmarks/` 쪽(88.89)이고 자가 다르다.

## 5. 공식 코드에서 확인한 것

| 항목 | 위치 |
|---|---|
| 창 구성 · `max_context_mode` | `IntPhys2/prediction_evals/evals/intphys2/eval.py:254, 268-295` |
| 모델별 config | `.../configs/{vjepa_2,videomaev2,vjepa_rope}.yaml` |
| **정확도 계산 코드는 예측 모델용이 없다** | 레포의 `ComputeScoreIntPhys2.ipynb` 는 **MLLM 단일영상용**(yes/no 파싱) |

우리 구현(`surprise.py:plan_growing_prefix`)이 `max_context_mode` 와 줄 단위로 대응한다.

## 6. 산출물

```
z_research/Benchmarks/exp_results/intphys2/<tag>/
  summary.json      sweep.per_context_length[C].{overall, breakdown}
  per_video.csv     scene_index, pair_id, is_impossible, condition, difficulty, camera,
                    context_length, surprise_avg, n_windows      <- 열별 재채점은 이걸로
  per_window.jsonl  창별 원값
```

열별 최고 재채점: `python z_research/scripts/analysis/intphys2_column_best.py --run <tag 경로>`

## 재현

```bash
GPUS=8 bash analysis/intphys2/run_grid.sh
python z_research/scripts/analysis/intphys2_column_best.py --run z_research/Benchmarks/exp_results/intphys2/<tag>
```
