# 처음 보는 물체 — 실제 영상 | 처음 보는 물체 PFT 예측 (2026-11-02)

모델 = 처음 보는 물체 PFT (`z_training/runs/vobench_release_postft_objholdout`): 학습에서 모양 torus · cone, 실물 농구공 · 스쿠터, 색 cyan · purple 을 통째로 뺐다 (학습 목록 721 영상에 0 회).
쌍 = 그 split 시험의 late 사라짐 쌍 중 처음 보는 속성이 있는 31 개. GIF = [실제 영상 (가능 미래) | 예측 (문맥 구간은 실제 영상, 미래 구간은 예측을 픽셀 decoder 로 그린 것)].
아래 줄의 correct / wrong = 표준 채점의 쌍 판정.

| 폴더 | 무엇이 처음인가 | GIF |
|---|---|---:|
| `unseen_shape/` | 모양만 (torus · cone, 색은 학습에 있음) + 실물 (농구공 · 스쿠터) | 22 |
| `unseen_color/` | 색만 (cyan · purple, 모양은 학습에 있음) | 7 |
| `unseen_shape_and_color/` | 모양과 색 둘 다 | 2 |
| `_all_models/` | 이전 판: 37 쌍 전부 × [실제 둘 · decoder 상한 · 원래 · 처음 보는 물체 PFT · v3 전부] GIF + 패치 L1 지도 GIF + `summary.json` | — |

단서: 궤적 틀 (운동 · 가림 길이 · 배경) 은 학습과 공유한다. 픽셀 decoder 는 VOBench 전체 (처음 보는 물체 포함) 로 학습됐다.
재현: `VIZ_SET=objholdout python predictor_lab/v3_ablation/page/gif_v3full_vob.py` (GPU 1) → `_all_models/`, 그 뒤 두 칸 GIF 는 `_all_models` 의 decoder GIF 에서 칸 0 · 4 를 잘라 만든다.
