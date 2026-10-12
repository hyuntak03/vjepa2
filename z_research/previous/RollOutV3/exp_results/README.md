# RollOutV3 산출물 (자 가중치 · 읽은 값)

> 시작점은 [`../README.md`](../README.md) 다. 경로의 정본은 `z_research/scripts/analysis/rollout3_paths.py` 다.
> - 자는 `R3_DECODER=<폴더>` 로 고른다.
> - 읽은 값은 `R3_OUT=<꼬리표>` 로 고른다 (→ `windows_<꼬리표>` · `v11_<꼬리표>` · `v11_mid_<꼬리표>`).
>
> ⚠️ **`readings.npz` 에는 자가 구워져 있다.** 읽은 값에는 자 지문 (`decoder_fp`) 이 실려 있다. 그림 스크립트는 지문이 자와 다르면 멈춘다.
>
> ⚠️ **다른 세션이 이 폴더들을 쓴다.** `presence` · `windows` · `v11` 은 TrainingEffects (`te_*.py`) 와 auto_research 가 읽는다. 옮기지 말 것.

## 자 (decoder)

| 폴더 | 자 | 지문 | 학습셋 | v11 판 빈 장면 '있음' (p) | 상태 |
|---|---|---|---|---:|---|
| **`identity_r8/`** | 위치 + 57-way (56 조합 + 없음) | `01c3afaac37f` | `training_r8` (8,640 clip) | **0 %** | **지금 자** |
| `identity/` | 같은 구조 | `aeda55874c1c` | `training_v8` (14,360 clip) | 56–99 % | 대체됨 |
| `presence/` | 위치 + 있음 logit (문턱 val 5 % 오탐) | `dcd24d8a8a47` | `training_v8` | 7–45 % | 대체됨. `rollout3_paths` 의 기본값 (다른 세션 호환) |

각 폴더에는 다음이 있다.
- `{p,z,h}/readout.pt` (identity) 또는 `readout_attn.pt` (presence)
- `summary.json`: 학습셋 test 성능과 라벨 규칙
- `attn_bias_px.json`: 위치 치우침. 그림에서 뺀다

⚠️ `identity_r8/summary.json` 의 `train_set` 은 원래 `training_v8` 로 잘못 적혀 있었다 (스크립트에 박혀 있었다). 2026-09-26 에 `training_r8` 로 고치고 `train_set_note` 를 달았다. 스크립트도 고쳤다.

## 읽은 값 (readings.npz)

| 자 | v3 16 창 | v11 가림 없음 + 문맥 끝 가림 | v11 문맥 중간 가림 |
|---|---|---|---|
| **identity_r8** | `windows_identity_r8/` | `v11_identity_r8/` | `v11_mid_identity_r8/` |
| identity | `windows_identity/` | `v11_identity/` | `v11_mid_identity/` |
| presence | `windows/` (+ `BEHAVIOR.md` · `CROSS_HEAD.md` · `WINDOW_SUMMARY.md`) | `v11/` (+ `V11_READOUT.md`) | `v11_mid/` |

- `v11_*` 의 열: `truth` · `in_frame` · `hidden` · `obj` · `sym_k` · `occ_timing` · `travel_dir` · `p` / `h` (x, y, score, top1).
- identity 자는 `p_prob` / `h_prob` (57-way 확률) 도 싣는다.
- `redraw_identity_r8/`: identity_r8 전체 사슬 (`decoder_train_redraw.sh`) 의 로그다.

## `_superseded/`

| 폴더 | 무엇 |
|---|---|
| `presence_v6_20260923/` · `windows_v6_20260923/` | 학습셋 v6 로 배운 첫 presence 자와 그 읽은 값 (2026-09-23) |
| `smoke/<폴더>/` | 각 폴더의 배관 점검 (`_smoke`) |

## 재현

```bash
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
# 자 학습 + 읽기 전체 (GPU 8 장, 약 45 분) — ../README.md §재현
SET=training_r8 NAME=identity_r8 FRAMES=/data2/local_datasets/world/world_analysis/RollOut_v2_training_v8 \
    bash z_research/scripts/analysis/decoder_train_redraw.sh
# 지금 경로 · 자 지문 · readings 지문
R3_DECODER=identity_r8 R3_OUT=identity_r8 $P z_research/scripts/analysis/rollout3_paths.py
```
