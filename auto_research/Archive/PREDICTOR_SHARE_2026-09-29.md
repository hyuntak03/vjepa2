# predictor 의 몫 — downstream 점수에서 "마지막 관측 복사" 를 뺀 나머지 (2026-09-29)

> 논지 정본 [`../paper/RETHINK_PREDICTOR_ROLE_2026-09-29.md`](../paper/RETHINK_PREDICTOR_ROLE_2026-09-29.md) §5 "모든 downstream 점수에 복사 대비 Δ 를 병기".
> 복사 = 예측 없이 **같은 표적 공간** 에서 문맥 마지막 latent 를 미래 자리에 그대로 둔 채점 (같은 쌍 · 같은 encoder). Δ = predictor − 복사.
> 수치 산출물 `../exp_results/verify/predictor_share.json`. 재계산한 것과 전사한 것을 행마다 표시했다. 1 차 · 이 표 자체는 적대 검증 전 (원자료의 개별 검증은 각 출처 문서).

## 표

| 벤치마크 · 프로토콜 | predictor | 복사 (같은 공간) | **Δ = predictor − 복사** [95 % CI] | n | 출처 / 재계산 |
|---|---|---|---|---|---|
| IntPhys 1 dev, Garrido skip2_w32, 표준 표적 (릴리즈 최고 칸) | 릴리즈 **88.89** | 85.00 | **+3.9** [−0.6, +8.3] (12 승 / 5 패) | 180 쌍 | `ip1score_final_scores.json` outcomes → 쌍 bootstrap **재계산** |
| 〃 skip2_w16 표준 | 릴리즈 83.33 | 73.89 | +9.4 [+3.9, +15.6] (24 / 7) | 180 | 〃 재계산 |
| 〃 skip2_w32 **인과 표적** | 릴리즈 84.44 | 77.78 | +6.7 [+0.6, +12.8] (23 / 11) | 180 | 〃 재계산 |
| IntPhys 1 dev, skip2_w16, **AR 공간** (블록별 LN(target)) | AR ep18 **90.56** | copy_blk 83.33 | **+7.2** [+2.2, +12.8] (19 / 6) | 180 | `ip1score_ar_copyblk_scores.json` → 재계산 |
| 〃 skip2_w32, AR 공간 | AR ep18 87.78 | copy_blk 80.56 | +7.2 [+1.7, +12.8] (20 / 7) | 180 | 〃 재계산 |
| 〃 skip2_w16, AR 공간 | AR ep38 **92.78** | copy_blk 83.33 | +9.4 (CI 없음 — ep38 쌍별 결과 미저장) | 180 | `z_training/ariel_eval/intphys1_four_models.json` 전사 |
| IntPhys 2 Main, w48, **각자 최고 C** (release C24 · copy C42) | 릴리즈 54.3 | 55.5 | −1.2 [−5.1, +3.0]; permanence all −5.0 [−13.3, +3.3] | 506 쌍 | 감사 JSON + VERIFY_ISCENE2 §8 재계산값 전사 |
| IntPhys 2 Main, w48, **같은 C = 24** | 릴리즈 54.3 | (C24 copy) | **+1.8** [−1.8, +5.5]; permanence all +2.5 [−4.2, +10.0]; Fixed +3.8 [−7.7, +15.4] | 506 | VERIFY_ISCENE2 재계산값 전사 |
| IntPhys 2 Main, w32, 각자 최고 C | 릴리즈 52.8 | 54.2 | −1.4 [−5.3, +2.6] | 506 | 〃 |
| IntPhys 2 Main, AR 공간 | AR ep38 50.79 (w16/C12) | **없음** | — | 506 | copy_blk 를 IntPhys 2 에 안 걸었다 (GPU 필요) |
| IntPhysGen v11_split_test, `surprise_c16t32`, 표준 | 릴리즈 75.70 | 73.24 | **+2.5** [+1.6, +3.3] | 10,752 쌍 | `TrainingEffects/scores/analysis.json` (block bootstrap 2,000) 전사 |
| 〃 | Ariel prefix ep43 79.33 | 73.24 | +6.1 [+5.2, +6.9] | 10,752 | 〃 전사 |
| 〃 AR 공간 | AR ep38 83.68 [83.0, 84.3] | **없음** | — | 10,752 | v11 copy_blk 는 사용자 지시로 미실행 (`ARIEL_CHECKPOINTS.md` §5-2b) |
| IntPhysGen v11 전체 (문맥 공유, 참고) | 릴리즈 73.37 | 73.2 | ≈ +0.2 (CI 없음) | 10,752 | `TrainingEffects/scores/SCORES.md` 전사 |
| EK100 anticipation 1 s, released 규약, action mean-class R@5 | 35.34 | copy_last 32.80 (mean 34.01) | **+2.54** (같은 head +2.54) | 9,296 clip | `val_metrics.jsonl` **재계산** |
| 〃 2 s | 30.20 | 28.10 | +2.10 (같은 head +2.47) | 9,296 | 재계산 |
| 〃 3 s | 24.91 | 22.85 | +2.06 (같은 head +2.12) | 9,296 | 재계산 |
| EK100 1 s, paper 규약 | 19.07 | 16.46 (mean 17.59) | +2.61 (같은 head +4.67) | 9,296 | copy 재계산 · 원래값은 학습 끝 val 전사 |
| 〃 2 s | 15.22 | 12.47 | +2.75 (같은 head +2.86) | 9,296 | 재계산 |

## 읽기 (한 문단)

predictor 를 빼고 마지막 관측을 그대로 두어도 점수의 대부분이 남는다. IntPhys 1 은 복사만으로 85 (표준) · 83 (AR 공간) 이고 predictor 의 몫은 릴리즈 +3.9 (CI 가 0 포함) · AR +7.2 (CI 가 0 제외) 다. 두 몫의 차 3 pt 는 180 쌍 잡음 안이라 "AR 이 더 안다" 로는 못 쓴다. IntPhys 2 는 C 선택에 따라 부호가 바뀌고 어느 판도 CI 가 0 을 넘지 않는다 — **구분 안 됨**. v11 은 +2.5 로 CI 는 0 을 넘지만 크기가 작고, 릴리즈의 그 몫은 색 위반에 몰려 있다 (SCORES.md: 색 +9.3, 모양 −1.6). EK100 은 지평 1–3 s 어디서도 +2 ~ +3 으로 일정하다. 요약하면 **어느 벤치마크에서도 predictor 의 몫이 +2 ~ +7 pt 를 넘지 않고, 그 중 CI 로 0 을 제외하는 것은 IntPhys 1 skip2_w16 · 인과 표적 · AR 공간, v11, (CI 없는) EK100 뿐이다.** 이것이 "downstream 이 predictor 의 예측력을 거의 안 쓴다" 의 수치 근거이고, 새 predictor 의 downstream 평가는 이 Δ 로 보고해야 한다.

## 단서 (행별)

- **best-C 선택 (IntPhys 1 · 2).** Garrido 격자에서 run 마다 최고 칸을 고른다. IntPhys 2 는 release 와 copy 의 최고 C 가 달라 (24 vs 42) Δ 부호가 뒤집힌다 → 같은 C 판을 주 값으로 쓴다. IntPhys 1 은 릴리즈 · 복사 모두 skip2_w32 가 최고라 문제가 덜하나, AR 은 skip2_w16 이 최고 (칸이 다르다).
- **표적 공간이 다르다.** 표준 = LN(target_encoder(창 전체))[미래] (양방향 시간 attention → 번짐, H6 §2), 인과 = 문맥만 본 표적, AR 공간 = 블록별 LN(target_encoder(2 장)). 복사 기준선은 **각 공간 안에서만** 짝이 된다. 공간을 가로지르는 비교 (AR 90.56 vs 릴리즈 88.89) 는 쌍 정확도만 가능하고 Δ 끼리 비교는 안 된다.
- **CI 없는 행.** AR ep38 IntPhys 1 (쌍별 결과 미저장), v11 전체 (문맥 공유라 CI 정의 자체가 불명확), EK100 전부 (clip 별 예측 미저장). EK100 은 val 9,296 clip 전수라 표본 오차는 작지만 head 선택 규약이 낙관적이다 (열별 독립 최고; 같은 head 고정값을 병기).
- **head 선택 (EK100).** 8 head 중 최고를 열마다 독립으로 고른다. paper 1 s 는 원래 predictor 의 최고 head 로 고정하면 Δ 가 +4.67 로 커진다 (copy 의 최고 head 가 다른 head).
- **복사의 encoder 교차.** 표준 copy 는 online z 의 LN 을 EMA target 공간에 대고 잰다 (H6 정정 14). 방향은 안 쟀다. 깨끗한 copy (target encoder 를 문맥 프레임만으로) 는 미실행.
- **AR 공간 복사 없는 행 (IntPhys 2 · v11).** AR 의 점수를 "predictor 몫" 으로 읽으려면 copy_blk 가 필요하다. IntPhys 2 copy_blk 는 `analysis/intphys2` 하네스에 블록별 복사 팔을 추가해 GPU 로 돌려야 하고, v11 은 `te_v11_score.py --preds copy_blk` 류가 필요하다 (사용자 지시로 보류).
- **v11 +2.5 의 구성.** SCORES.md: 색 +9.3 / 모양 −1.6 [−2.9, −0.2] / vanish '물체→빈' −6.8 — 색을 빼면 −1.7 [−2.6, −0.7]. 즉 v11 의 predictor 몫은 색 위반 감지 하나에서 나온다.
- 이 표는 "predictor 가 쓸모없다" 가 아니라 "이 벤치마크들이 predictor 의 예측력을 요구하지 않는다" 의 근거다 (H6 정정 8: 쌍 비교 surprise 는 상태 진화 여부를 못 가른다).

## 재현

```bash
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2; PY=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
# IntPhys 1 paired Δ (쌍 bootstrap 10,000, tie = 오답): outcomes 배열에서
#   z_research/TrainingEffects/ip1score/ip1score_final_scores.json      (release|std|skip2_w32 vs copy|std|skip2_w32 등)
#   z_research/TrainingEffects/ip1score/ip1score_ar_copyblk_scores.json (ariel_ar_ep18|std|skip2_w16 vs copy_blk|std|skip2_w16)
# IntPhys 2: auto_research/exp_results/verify/intphys2_permanence_audit.json (best-C), auto_research/_verify_scratch/ISCENE2/out/ip2.json["diff"] (같은 C 24)
# v11: z_research/TrainingEffects/scores/analysis.json cells_curves.all.{acc,diff}; AR: z_training/ariel_eval/v11_ariel_cells.json
# EK100: z_research/anticipation/EK100/exp_results/action_anticipation_frozen/ek100_vith_{rel,pap}_grid8_val_*/val_metrics.jsonl 마지막 줄; 같은 head Δ: auto_research/_verify_scratch/ISCENE2/out/sc5.json
# 종합 JSON: auto_research/exp_results/verify/predictor_share.json (이 문서와 같은 스크립트가 썼다; 2026-09-29)
```

## 부록 — 저자 측 수치: V-JEPA 2 논문 Table 20 (EK100 anticipation probe 입력 ablation, 로컬 PDF p.44 에서 직접 확인 2026-09-29)

| probe 입력 | verb | noun | action R@5 |
|---|---|---|---|
| encoder 출력만 | 61.3 | 57.0 | **39.1** |
| predictor 출력만 | 48.7 | 34.7 | 20.2 |
| 둘 다 (논문 규약) | 63.6 | 57.1 | **39.7** |

- 저자 자신의 ablation 에서 **predictor 의 몫은 action +0.6 pt** 다 (encoder 만 39.1 → 둘 다 39.7). 본문 (p.43) 도 "mostly requires strong semantic understanding, as opposed to forecasting capabilities" 라 적었다. 우리 SC3 (predictor 미래를 복사로 바꿔도 −1.3 ~ −2.6) · SC5 (Δ +2) 와 같은 크기.
- 옮겨 적은 값 (재계산 아님). 이 표는 "복사 대비" 가 아니라 "predictor 제거 대비" 라 정의가 다르다 — 같이 인용하되 섞지 않는다. 출처 정리 [`LITERATURE_PREDICTOR_ROLE_2026-09-29.md`](LITERATURE_PREDICTOR_ROLE_2026-09-29.md) §1-3.
