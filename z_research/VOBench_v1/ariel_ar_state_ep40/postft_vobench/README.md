# ar_state ep40 → VOBench post-FT (2026-10-12)

run `z_training/runs/vobench_ar_state_postft` (config `configs/training/vobench_ar_state_postft.yaml`): ar_state ep40 에서 시작, VOBench block split 의 train 절반 **가능 영상 720 clip** (360 block, primitive · realistic) 로
문맥 8 블록 → 예측 8 블록 (TF + rollout 8), lr 5e-5, 30 epoch (450 step, vll5 24 GB × 4, 30 분). 학습 코드 = ariel 브랜치 학습 경로를 main 에 3-way 병합한 것.

## 같은 분포 · held-out block (`vobench_v1_split_test`, 360 block · 720 쌍) — `compare.{md,json}`

| | release | ar_ep40 | ar_state | **ar_state + FT** | 복사 (AR 공간) |
|---|---:|---:|---:|---:|---:|
| primitive overall | 70.3 | 82.5 | 81.9 | **88.9** | 77.2 |
| realistic overall | 82.2 | 86.4 | 85.0 | **91.7** | 81.9 |
| primitive late shape · colour | 50.0 · 63.9 | 61.1 · 66.7 | 63.9 · 66.7 | **80.6 · 75.0** | 55.6 · 52.8 |
| realistic late shape · colour | 61.1 · 75.0 | 61.1 · 72.2 | 61.1 · 75.0 | **83.3 · 80.6** | 55.6 · 63.9 |

쌍 단위 (ar_state → +FT, 두 objset): 전체 83.5 → 90.3 (바뀐 쌍 10 vs 59, McNemar p < 1e-4) · late 76.4 → 86.6 (6 vs 28, p 0.0002) ·
**late · shape+colour · 이동 60.8 → 75.8** (6 vs 24, p 0.0014) · 보일 때 shape+colour · 이동 75.7 → 85.4 (4 vs 31, p < 1e-4).

## 분포 밖 (in-loop IntPhys 1 dev val, 60 쌍 — 감시용)

epoch 0 (출발) 83.3 → epoch 1 78.3 → epoch 5–30 76.7–80.0 (끝 78.3). margin 은 +0.008 → +0.015. 정식 채점 (180 쌍, A.8 격자) 은 미실행.

## 단서

- **같은 분포다.** train · test 가 block 은 다르지만 같은 물체 48 + realistic 메시 · 같은 장면 · 같은 운동 6 · 같은 가림 설계를 공유한다. 점수 상승은 "이 장면 분포를 보면 배울 수 있다" 까지만 말한다 — release post-FT (v11) 도 도메인 안에서만 올랐다 (z_training/RESULTS_2026-09-19.md).
- 칸 (위반 × 가림) 은 36 쌍, late × 운동은 18 쌍 — 칸 하나의 CI 는 ±15 pt 안팎. 결론은 쌍 단위 McNemar 줄로 읽는다.
- 점수가 오른 이유 (정체를 이어 가게 됐나 · 자리를 맞추게 됐나) 는 이 채점으로 못 가른다 — post-FT p 에 맞춘 자 (readout) 로 다시 읽어야 한다 (미실행).
- AR 타깃 공간 (블록별 LN(h)) 이라 release 열과는 쌍 정확도만 비교한다.

## 재현

```bash
python z_training/data/build_vobench_split_index.py
GPUS=4 bash z_training/train.sh vobench_ar_state_postft
GPUS=4 DECODE_WORKERS=8 BATCH_SIZE=16 bash z_training/eval.sh vobench_ar_state_postft vobench_v1_split_test
python z_research/scripts/analysis/vobench_split_compare.py
```
