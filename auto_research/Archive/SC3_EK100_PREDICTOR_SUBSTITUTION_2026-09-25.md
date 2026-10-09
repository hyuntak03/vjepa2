# SC3 — EK100 1 s anticipation 은 predictor 의 "미래" 를 거의 쓰지 않는다. 학습된 probe 를 고정한 채 predictor 출력만 치환 (2026-09-25)

> 상태: **1 차 결과, 적대 검증 전.** oral 방향 sanity check 의 셋째다 ([`../paper/ORAL_DIRECTION_2026-09-25.md`](../paper/ORAL_DIRECTION_2026-09-25.md) §4).
> 수치 출처: `z_research/anticipation/EK100/exp_results/action_anticipation_frozen/<TAG>/val_metrics.jsonl` (val 9,296 clip 전수, mean class recall@5, head 8 개 중 최고).

## 0. 한 줄

학습이 끝난 anticipation probe 에서 predictor 의 1 s 뒤 토큰만 바꿔 넣었다. 바꿔 넣은 것은 (a) 문맥 마지막 튜블릿 encoder 토큰의 복사와 (b) 문맥 전체 평균 토큰이다.
- action R@5 는 2.5 점 안에서 떨어진다.
  - released 규약 35.34 → 32.80 / 34.01.
  - paper 규약 19.07 → 16.46 / 17.59.
- verb · noun 은 1.5 점 안에서 떨어진다.
- 공간 정보를 통째로 지운 평균 토큰도 복사와 비슷하거나 덜 떨어진다.
- 그래서 **이 벤치마크 점수의 대부분은 predictor 가 만든 미래 상태 없이 나온다.** probe 를 고정한 채 입력 분포를 바꿨으므로, 이 하락 폭은 predictor 기여의 **상한** 이다.

## 1. 왜

- EK100 anticipation 은 V-JEPA 2 가 "prediction" 능력의 근거로 드는 실세계 과제다 (논문 §6).
- 우리 척도로 1 s 는 사전학습 4 fps 기준 약 2 튜블릿이다. SC1 · SC2v2 의 도달 거리 **안쪽** 에 해당한다.
- 논문 Table 20 (ViT-g384) 에서 encoder 만 쓴 probe 가 이미 39.1 이고, predictor 를 더하면 39.7 (+0.6) 이다. 다만 그 비교는 probe 를 따로 학습한 것이다.
- 여기서는 **같은 probe** 에서 predictor 출력만 바꾼다.

## 2. 방법

- **체크포인트.** 기존 학습 probe 둘을 그대로 쓴다 (새 학습 없음).
  - `ek100_vith_official256_released_grid8` (released 규약: 문맥 = action 끝 1 s 전).
  - `ek100_vith_official256_paper_grid8` (paper 규약: 문맥 = action 시작 1 s 전, 정직한 anticipation).
- **치환.** `AnticipativeWrapper` 에 `pred_replace` 인자를 더했다 (기본 None = 원래 동작).
  - 위치: `evals/action_anticipation_frozen/modelcustom/vit_encoder_predictor_concat_ar_nonsquare.py`.
  - `copy_last`: predictor 출력 자리에 LN(encoder 마지막 N_pred 토큰) 을 넣는다. 예측 없이 마지막 관측을 복사하는 것이다.
  - `mean`: LN(encoder 전 토큰 평균) 을 N_pred 자리에 반복해 넣는다. 공간 정보가 없다.
  - predictor 출력은 원래 LN(target) 척도로 학습되므로 두 치환 모두 LN 을 걸었다.
- **재현 검사.** 같은 방식 (VAL_ONLY, 치환 없음) 으로 released 를 다시 돌렸다. 56.36 / 56.16 / 35.34 로 학습 끝 값과 **소수 둘째 자리까지 같다.**

## 3. 결과 (verb / noun / action, mean class R@5)

| 규약 | predictor 출력 (원래) | `copy_last` | `mean` |
|---|---|---|---|
| released | 56.36 / 56.16 / 35.34 | 55.80 / 55.69 / **32.80** (−0.56 / −0.47 / −2.54) | 55.56 / 54.67 / **34.01** (−0.80 / −1.49 / −1.33) |
| paper | 33.25 / 35.24 / 19.07 | 32.48 / 34.71 / **16.46** (−0.77 / −0.53 / −2.61) | 32.07 / 34.62 / **17.59** (−1.18 / −0.62 / −1.48) |

- 두 규약 모두 action 의 86–96 % 가 predictor 미래 없이 남는다.
- paper 규약 (정직한 anticipation) 에서도 하락 폭이 released 와 비슷하다.

## 4. 단서

1. **상한이다.** probe 가 원래 predictor 출력에 맞춰 학습됐으므로, 치환은 분포 밖 입력이다. 하락의 일부는 분포 이동이다.
   - 진짜 기여를 재려면 치환 입력으로 probe 를 다시 학습해야 한다 (probe 학습, 20 epoch). 하지 않았다.
2. **"predictor 가 쓸모없다" 가 아니다.** 1 s anticipation 이라는 과제가 predictor 의 상태 진화를 거의 요구하지 않는다는 뜻이다.
   - 논문 Table 20 의 +0.6 과 방향이 같다.
3. **anticipation 시간을 늘리지 않았다** (SC5 후보). probe 는 0.25–1.75 s 로 학습됐다. 2 s 이상은 분포 밖이라 같은 방식으로는 깨끗하지 않다.
4. action 은 verb × noun 조합이라 작은 흔들림이 커진다. clip 단위 bootstrap CI 는 내지 않았다. 예측 파일을 저장하지 않았기 때문이다.

## 5. 이것이 oral 이야기에서 하는 일

**흔한 평가가 도달 거리를 재지 않는다** 는 절의 실세계 증거다.
- VoE (IntPhys 1): 복사 기준선 85.00 vs 릴리즈 88.89, n.s. ([`VERIFY_0925_2026-09-25.md`](VERIFY_0925_2026-09-25.md)).
- EK100 1 s anticipation: predictor 미래를 복사로 바꿔도 action −2.5 점 안이다.
- 둘 다 predictor 가 물체를 멀리 운반하는지 묻지 않는다. 그래서 "world model 의 예측 능력" 근거로 쓰기에 약하다.

## 재현

```bash
R=z_research/anticipation/EK100/exp_results/action_anticipation_frozen
mkdir -p $R/<TAG> && ln -sfn $PWD/$R/ek100_vith_official256_{released|paper}_grid8/latest.pt $R/<TAG>/latest.pt
REL="data.anticipation_point_mode=released data.time_source=frame_fixfps"; PAP="data.anticipation_point_mode=paper data.time_source=timestamp"
VAL_ONLY=1 HEADS=grid8 TAG=<TAG> SET="$REL model_kwargs.wrapper_kwargs.pred_replace=copy_last" NODE=vll3 MEM_PER_GPU=30G GPUS=8 \
  bash z_research/anticipation/EK100/sbatch.sh     # SLURM job 안 셸이면 SLURM_* 를 먼저 unset (자기제출 방지 가드)
# TAG: ek100_vith_{rel,pap}_grid8_val_{copylast,mean} · ek100_vith_rel_grid8_val_repro (치환 없음)
```
