# _superseded — 실행 불가, 기록만

- `rollout_position.py`, `rollout_token_contrib.py`, `plot_rollout_position.py`, `plot_rollout_contrib.py`, `build_rollout_index.py` — **RollOut_v1** (등속 직선, 7,392 clip). 프레임·캐시·결과 전부 삭제 (2026-09-10~11). 레지스트리에서도 뺐다.
- `rollout2_position.py`, `rollout2_token_decoder.py`, `plot_rollout2_overlay.py`, `plot_rollout2_traj*.py`, `plot_rollout2_confusion.py` — 2026-09-10 이전 RollOut_v2 pipeline (h 로 배운 pooled/token 자를 p 에 이식, `exp_results/position_regression`·`token_decoder`). 결과는 09-10 에 삭제됐고, 학습셋에서 자를 정하는 현행 pipeline (`analysis/rollout2_*_readout.py`) 으로 대체됐다.

현행 진입점: `z_research/RollOutV2/README.md`.
