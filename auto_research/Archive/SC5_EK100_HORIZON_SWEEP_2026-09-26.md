# SC5 — EK100 anticipation 지평 스윕: predictor 몫은 지평이 reach 를 넘으면 사라지나 (2026-09-26)

> 상태: **설계 · 명령을 먼저 적음. 결과 §3 은 vll3 val job 이 끝나는 대로 채운다.** 1 차 · 적대 검증 전.
> 앞 결과: [SC3](SC3_EK100_PREDICTOR_SUBSTITUTION_2026-09-25.md) — 1 s 에서 predictor 미래를 복사로 바꿔도 action −2.5 (released 35.34 → 32.80; paper 19.07 → 16.46).
> 방향 정본 [`../paper/DIRECTION_2026-09-25.md`](../paper/DIRECTION_2026-09-25.md) §2-2 "쓰이는 벤치마크는 state evolution 영역을 재지 않는다" 의 EK100 판.

## 1. 왜

- 1 s anticipation 은 사전학습 4 fps 기준 약 2 튜블릿, 즉 predictor 의 reach (실영상 2–3 칸) **안쪽** 이다. 그래서 predictor 몫이 작아도 (SC3) "reach 밖에서는 어떤가" 를 말할 수 없다.
- 지평을 2 s · 3 s 로 늘리면 (4–6 튜블릿) reach 경계 안팎을 걸친다. 예측:
  - (a) predictor 몫 Δ = R@5(원래) − R@5(복사) 가 지평에 따라 **줄면** → predictor 는 reach 안에서만 무언가를 더하고, 벤치마크가 긴 지평을 요구하지 않는 한 그 한계가 안 보인다.
  - (b) Δ 가 유지 · 증가하면 → anticipation 에서 predictor 가 쓰는 정보는 위치 운반이 아닌 것 (의미 · 문맥 요약) 이고, reach 와 무관하다. 이 경우 "벤치마크가 못 본다" 문장을 anticipation 에는 쓰지 않는다.
- 어느 쪽이든 문장이 선다 (DIRECTION §3-2).

## 2. 설계

- probe: 학습된 `ek100_vith_official256_released_grid8` (released 규약, 20 epoch, head 8) 와 `..._paper_grid8` 을 **그대로** 쓴다 (새 학습 없음).
- 지평: `data.anticipation_time_sec=[at, at]`, at ∈ {1.0 (SC3 값 재사용), 2.0, 3.0}. wrapper 는 `anticipation_steps = at · fps / tubelet` 으로 mask 위치를 옮긴다 (RoPE 절대 위치).
- 팔: 원래 predictor vs `model_kwargs.wrapper_kwargs.pred_replace=copy_last` (encoder 마지막 튜블릿의 LN 복사). Δ(at) = 원래 − 복사.
- val 9,296 clip 전수, mean class R@5, head 8 개 중 최고 (기존 규약).
- **단서 (미리 적음).** probe 는 anticipation 0.25–1.75 s 로 학습됐다. 2 s 이상은 probe 학습 분포 밖이라 절대값은 떨어질 것이고, **Δ 의 상대 변화만** 읽는다. 문맥 clip 도 action 에서 더 멀어지므로 (released 규약에서는 action 프레임 포함 비율이 줄어든다) 원래 · 복사 둘 다 같이 떨어지는 몫은 Δ 에서 상쇄된다.

## 3. 결과

**채움 (2026-09-26 오후; job 216631–216636 완료).** 아래 표 = val 9,296 clip 전수, mean class R@5, head 8 개 중 최고 (열마다 독립 선택 — 낙관 규약; 같은 head 로 고정하면 Δaction +2.1 ~ +2.9, paper 1 s +4.7. 2 차 검증 [`VERIFY_ISCENE2_2026-09-26.md`](VERIFY_ISCENE2_2026-09-26.md) §7, Δ 전부 재현. per-clip 예측이 없어 CI 없음). Δ = 원래 predictor − copy_last (predictor 고유 몫의 상한).

~~아직 없음 (2026-09-26 새벽).~~ 6 개 val job (216631–216636, vll3, 8 GPU 요청) 은 60 분 동안 자원을 못 받았다 — vll3 는 다른 사용자 4+4 GPU, 이후 우리 4 GPU job (216651) 이 차지. job 은 큐에 남겨 두었다; 끝나면 `<TAG>/val_metrics.jsonl` 의 마지막 줄을 읽어 아래 표를 채운다.

| 지평 | 규약 | 원래 (verb / noun / action) | copy_last | Δ action |
|---|---|---|---|---|
| 1 s | released | 56.36 / 56.16 / 35.34 | 55.80 / 55.69 / 32.80 | +2.54 (SC3) |
| 1 s | paper | 33.25 / 35.24 / 19.07 | 32.48 / 34.71 / 16.46 | +2.61 (SC3) |
| 2 s | released | 53.77 / 49.82 / **30.20** | 52.91 / 49.77 / **28.10** | Δaction **+2.10** (verb +0.86, noun +0.05) |
| 3 s | released | 43.61 / 43.46 / **24.91** | 44.75 / 42.68 / **22.85** | Δaction **+2.06** (verb −1.14, noun +0.78) |
| 2 s | paper | 28.29 / 32.27 / **15.22** | 26.69 / 30.82 / **12.47** | Δaction **+2.75** (verb +1.60, noun +1.45) |

수집: `python - <<EOF` 형태로 `val_metrics.jsonl` 마지막 줄의 `verb/noun/action.recall` 을 읽는다 (SC3 문서와 같은 방식).

### 3-1. 판정 (1 차)

- **Δ 는 지평이 늘어도 줄지 않는다**: released Δaction 1 s +2.54 → 2 s +2.10 → 3 s +2.06; paper 1 s +2.61 → 2 s +2.75. 절대 점수는 지평에 따라 크게 떨어지지만 (released action 35.3 → 30.2 → 24.9), predictor 고유 몫은 약 2–3 점으로 일정하다.
- 설계 §2 의 두 갈래 중 **"Δ 유지" 쪽**이다: EK100 anticipation 에서 predictor 가 기여하는 것은 reach 밖 운반이 아니라 지평과 무관한 작은 성분 (약 2 점) 이다. 따라서 "EK100 은 reach 밖을 못 본다" 가 아니라 **"EK100 은 지평 1–3 s 어디서도 predictor 에 거의 기대지 않는다 (encoder + probe 가 대부분)"** 로 쓴다. 평가 감사 표의 결론 (predictor 미래를 복사로 바꿔도 −2~−3) 은 세 지평에서 재현된다.
- 단서: probe 는 0.25–1.75 s 로 학습됐다 (2–3 s 는 분포 밖 → 절대값은 해석하지 않고 Δ 만). copy_last 는 상한 (probe 가 원래 출력에 맞춰졌으므로 분포 이동 손실 포함). clip 단위 CI 없음 (예측 파일 미저장).

## 재현

```bash
R=z_research/anticipation/EK100/exp_results/action_anticipation_frozen
mkdir -p $R/<TAG> && ln -sfn $PWD/$R/ek100_vith_official256_{released|paper}_grid8/latest.pt $R/<TAG>/latest.pt
REL="data.anticipation_point_mode=released data.time_source=frame_fixfps"; PAP="data.anticipation_point_mode=paper data.time_source=timestamp"
for at in 2.0 3.0; do
  VAL_ONLY=1 HEADS=grid8 TAG=ek100_vith_rel_grid8_val_at${at%.*} SET="$REL data.anticipation_time_sec=[$at,$at]" NODE=vll3 MEM_PER_GPU=30G GPUS=8 bash z_research/anticipation/EK100/sbatch.sh
  VAL_ONLY=1 HEADS=grid8 TAG=ek100_vith_rel_grid8_val_at${at%.*}_copylast SET="$REL data.anticipation_time_sec=[$at,$at] model_kwargs.wrapper_kwargs.pred_replace=copy_last" NODE=vll3 MEM_PER_GPU=30G GPUS=8 bash z_research/anticipation/EK100/sbatch.sh
done
# paper 규약은 at=2.0 만 (TAG ek100_vith_pap_grid8_val_at2{,_copylast}). SLURM job 안 셸이면 SLURM_* 를 먼저 unset.
# 제출된 job: 216631–216636 (2026-09-26). 결과: <TAG>/val_metrics.jsonl
```
