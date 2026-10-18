# z_vll7_training/tests — 모델 로드 없는 검증

| 테스트 | 무엇을 재나 | 2026-10-16 결과 |
|---|---|---|
| `ctx_state_cpu_test.py` | ctx_state forward 모양 · **KV rollout == 전체 재계산** (값·grad) · tf_weight 0 경로 · mask token 규약 거부 · `ctx_state_losses` | 출력 차 **0.0**, grad 차 5.5e-30 (float64) |
| `var_window_cpu_test.py` | 가변 창 로더를 **실제 mp4 로** — rank 동기 계획 / rank 별 영상 선택 / 모양 · 패딩 없음 / 마스크 범위 | 6 step 읽기 실패 0 |

⚠️ `ctx_state_cpu_test.py` 는 `predictor_embed_dim / num_heads >= 12` 로 돌려야 한다 —
레포 RoPE 의 `emb_sin.squeeze(-1).repeat(1,1,1,2)` 가 `d_dim // 2 == 1` 에서 차원이 어긋난다
(릴리즈 구성 384/12 = 32 → d_dim 10 에서는 안 걸린다).
