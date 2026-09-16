# context_encoder_analysis — 관측 없는 상태 진화를 위한 재료는 encoder 에 있는가 (시작점)

> **마지막 갱신 2026-09-16 · 바뀐 것 3 줄**
> 1. 주장 (`CLAIM_… v0.2`): predictor 는 마지막 관측 상태를 토큰별로 짧게 외삽한다 (등속 3–4 튜블릿, 가속·수직 ≤1–2; 재귀 없음; 경계에서 가려진 물체는 놓지 않음). "4 슬롯 뒤 없어진다" 는 진실 자리에서만 — 옛 자리에 남는지 미결.
> 2. **새 결과**: 전역 선형 정렬 p→h 를 걸면 채점이 오른다 (late +3~6, early +7; 대조군 0) — 정렬 개입은 "기각" 이 아니라 **부분 효과** (post-FT +25 의 1/5). 경계 토큰 안에서 위치·속도·시각 서명·장면은 거의 직교. 가속은 교차 쌍으로도 못 가름 (시나리오 지름길) → **(v0×a) 세트 필요**.
> 3. **post-FT predictor 의 미래** (`../predictor_training/predictor_IntPhysGenV11_PFT/`): v11 안에서는 8 슬롯 내내 물체를 놓고 가림 뒤 재출현도 맞추지만, RollOut_v2 에서는 v11 속도를 2 배로 강요 — 배운 것은 궤적 prior. 남은 것: v11 정체성·가림막 부분공간 (스크립트만 있음), attentive E2, 옛 자리 검사, GPU 결정 하나 (시간 범위 knockout). 이전 세션에서 끊긴 백그라운드 작업은 없다.

> **먼저 볼 것: [`Archive/EXPERIMENTS_2026-09-16.md`](Archive/EXPERIMENTS_2026-09-16.md) — 실험별 동기 · 세팅 · 결과 · 단서를 한 문서에 정리한 것.**
> 그 다음: 주장 종합 [`Archive/CLAIM_NO_STATE_EVOLUTION_2026-09-15.md`](Archive/CLAIM_NO_STATE_EVOLUTION_2026-09-15.md) (v0.2), 가설·큐 [`Archive/STATE_EVOLUTION_HYPOTHESES_2026-09-15.md`](Archive/STATE_EVOLUTION_HYPOTHESES_2026-09-15.md) (v1 + 상단 정정).
> 레포 규칙은 루트 `CLAUDE.md`, 논문 뼈대는 `../IntPhysGenV11/Archive/PAPER_STORY_2026-09-06.md`.

## 폴더

```
Archive/EXPERIMENTS_2026-09-16.md     실험 전체 정리 (시작점)
Archive/CLAIM_…  / STATE_EVOLUTION_…  주장 종합 / 가설·큐
exp_results/
  time_alignment/ · ctxenc_time_alignment_gated/      실험 1 시간 정렬 행렬 (+ 물체 특이 gate)
  ctxenc_boundary_kinematics/ · ctxenc_accel_crosspair/  실험 2 경계 토큰 속도·가속 (+ 교차 쌍)
  ctxenc_boundary_identity_occlusion/ · _v2/           실험 3 가려진 프레임의 모양·색
  ctxenc_subspace_overlap/ · ctxenc_subspace_v2/       실험 4 p–h 부분공간 · 정렬 후 채점
  ctxenc_state_subspaces_kinematics/                   실험 5 상태 부분공간 겹침 (운동 편)
```
스크립트는 `../scripts/analysis/ctxenc_*.py` (docstring 마지막 줄에 실행 예시).

## 읽는 규칙
- 수치는 `exp_results/` 산출물에서 다시 읽는다. 문서 표마다 출처가 있다.
- 시간 정렬 행렬은 배경 토큰에도 시각 서명이 있다는 단서와 같이 읽는다 (실험 1).
- 목적함수·마스크 토폴로지는 원인 축으로 두지 않는다 (서술·인용까지).
