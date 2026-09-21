# context_encoder_analysis — 관측 없는 상태 진화를 위한 재료는 encoder 에 있는가 (시작점)

> **마지막 갱신 2026-09-20 · 바뀐 것 3 줄**
> 1. **실제 영상에서 문맥 encoder 는 좌·우 운동 방향을 담는다** (실험 6, 새것): 정방향으로만 배운 probe 가 역재생 clip 에서 답을 뒤집는다 — ssv2 98.1% / ntu 96.2% 뒤집힘, 3,842 파라미터 probe 로도 같다. 능력 표의 **1 읽기** 칸을 real-world 로 채웠다. 정본 [`Archive/ENCODER_TEMPORAL_DYNAMICS_2026-09-20.md`](Archive/ENCODER_TEMPORAL_DYNAMICS_2026-09-20.md). 남은 단서: "방향" 과 "마지막 자리" 를 아직 못 가른다 (순서 섞기 대조 미실행).
> 2. state evolution 의 정의가 **네 능력** (읽기·지속·전이·영속) 으로 정해졌다 (2026-09-19 사용자 결정) — 정본 [`Archive/STATE_EVOLUTION_CAPABILITIES_2026-09-19.md`](Archive/STATE_EVOLUTION_CAPABILITIES_2026-09-19.md). 현재 V-JEPA 2: 읽기 ✅ / 지속 ❌ / 전이 ◐ / 영속 ❌.
> 3. 아래 1~5 는 합성 세트 (v11 · RollOut_v2) 에서의 이전 결과다. 주장 종합은 `CLAIM_…` v0.2, 실험별 세팅은 `EXPERIMENTS_2026-09-16.md`.

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
  encoder_temporal_dynamics/                           실험 6 실제 영상의 좌/우 운동 방향 — 정방향 학습 → 역재생 시험 (ssv2_VP · ntu_direction, 2026-09-20)
```
스크립트는 `../scripts/analysis/ctxenc_*.py` (docstring 마지막 줄에 실행 예시).

## 읽는 규칙
- 수치는 `exp_results/` 산출물에서 다시 읽는다. 문서 표마다 출처가 있다.
- 시간 정렬 행렬은 배경 토큰에도 시각 서명이 있다는 단서와 같이 읽는다 (실험 1).
- 목적함수·마스크 토폴로지는 원인 축으로 두지 않는다 (서술·인용까지).
