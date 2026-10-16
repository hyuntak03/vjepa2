# z_training_config — 학습 설정 결정 기록

학습 run 의 **설정을 왜 그렇게 정했는지** 를 날짜별로 남긴다. 결과 (채점 수치) 의 정본은 `z_training/RESULTS_*.md`, config 파일은 `configs/training/`.

| 문서 | 내용 |
|---|---|
| [`NATURAL_AR_MS_2026-10-14.md`](NATURAL_AR_MS_2026-10-14.md) | kind=ar **multistart** 설계 (여러 시작점 rollout, 데이터셋 우선 샘플링, V-JEPA 2 공식 비율, 창 B 안) — 여기까지 온 진단 · 실패한 시도 · 숫자 · 남은 일 |
| [`../z_training/RESULTS_2026-10-16.md`](../z_training/RESULTS_2026-10-16.md) §2 · §3 · §4 | object_permanence 경계 정렬 샘플링 (`boundary:`), ctxt_video (문맥 = 영상, 문맥 최소 2 블록, 릴리즈 데이터 비율), 영상 타깃판 — 왜 그렇게 정했나 · 결과 · 정정 |
| [`CTXT_VIDEO_TGTVIDEO_HOLE_2026-10-16.md`](CTXT_VIDEO_TGTVIDEO_HOLE_2026-10-16.md) | 문맥 = 영상 · 타깃 = 영상 (tgtvideo) · encoding 전 문맥 구멍 (tgtvideo_hole) — 샘플링 점검 (계획 그대로 셈), 구멍 설계, 단서, 미리 적은 판정 |

규칙: 파일명에 날짜 (`TOPIC_YYYY-MM-DD.md`), 철회 · 정정은 지우지 않고 남긴다, 맨 아래 `## 재현`.
