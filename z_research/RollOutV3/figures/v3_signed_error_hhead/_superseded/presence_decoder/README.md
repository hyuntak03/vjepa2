# presence_decoder — 옛 자로 그린 판 (2026-09-25 물림)

이 폴더의 그림 · `values*.json` 은 **옛 자** (`exp_results/presence/`, 지문 `dcd24d8a8a47`) 로 읽은 값이다.
위치 xy + presence logit 하나, '있다' 문턱 = val 음성 5 % 오탐 (`thr_val_fpr5`).

## 왜 물렸나
사용자 결정 (2026-09-25) 으로 자를 **위치 + 정체 (모양 7 × 색 8 = 56 조합 + 없음, 57 분류)** 로 바꿨다
(`exp_results/identity/`, `z_research/scripts/analysis/rollout2_identity_readout.py`).
- presence 하나로는 "p 가 물체를 만들었는데 안 움직인다" 와 "못 만든다" 를 가를 수 없다. 정체까지 맞으면 (우연 1/56) 그 물체를 만든 것이다
- 새 자의 '없음' 라벨은 빈 장면과 **통째로** 화면 밖인 칸뿐이다. 가장자리에 걸린 칸은 학습에서 뺐다 (옛 자는 이 칸을 '없음' 으로 배웠다)
- '있다' 판정은 새 자에서 argmax (문턱 0 = `rollout3_paths.present_thr`)

같은 test 칸 비교 (학습셋, `exp_results/identity/summary.json` 과 옛 `preds_attn.npz`): 있음/없음 AUROC 는 둘 다 천장
(p 0.9963 → 0.9989), 위치 오차 평균 p 19.0 → 16.6 px · z 11.8 → 10.6 · h 12.2 → 13.7, 정체 조합 p 88.3 · z 92.2 · h 93.4 %.

## 재현
옛 자는 환경변수 없이 (기본 `R3_DECODER=presence`) 그림 스크립트를 돌리면 이 판이 다시 나온다.
2026-09-25 에 스크립트를 고친 뒤 옛 자로 다시 돌려 `values*.json` 이 바이트 단위로 같음을 확인했다.
