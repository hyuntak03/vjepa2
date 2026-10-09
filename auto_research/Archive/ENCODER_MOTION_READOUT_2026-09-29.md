# frozen encoder 의 문맥 끝 토큰이 운동을 담는가 — 팬 · SSv2 · EK100 (2026-09-29)

> RETHINK §1-b 의 미확정 칸 ("EK100 에서 encoder 가 1 인칭 카메라 운동을 충분히 담는가") 을 채운다. 1 차 · 적대 검증 전.
> 스크립트 `scripts/e_ek_egomotion.py` (추출, GPU) · `scripts/e_ek_egomotion_analyze.py` (ridge, CPU). 산출 `exp_results/scene/ego_motion_{pan,ssv2,ek100}.json`. 원자료 `_stage/results_ar/ego_{pan,ssv2,ek100}/`.

## 1. 정의

- 입력: online encoder (predictor 가 먹는 z) 와 target encoder (LN h) 의 **문맥 마지막 튜블릿 256 토큰** (프레임 14–15).
- 표적 (Farneback flow, px/프레임): (a) 전역 = 프레임 12→15 flow 장의 중앙값 (ego-motion 대리) — 토큰 평균에서 읽는다; (b) 국소 = 칸별 flow — 토큰 하나에서; (b') 국소 잔차 = 칸별 flow − 전역 (물체 · 시차 운동); (c) 미래 변위 = 미래 슬롯 i 의 칸 s 내용이 마지막 관측에서 어디 있었나 (역추적, 슬롯 0–7) — 토큰 하나에서.
- ridge, α 안쪽 3-fold, **clip 5-fold 교차검증 R²**, 무작위 대조 = 표적 셔플. 팬 = 양성 대조 (v 알려짐, y = 0).
- **표적 신뢰도 상한 (대리)**: flow 는 잡음이 커서 R² 의 천장이 1 이 아니다. 문맥 끝 flow 로 슬롯 0 의 1 걸음 변위를 선형으로 맞힌 R² 를 "flow 자기 일관성" 으로 병기한다 (같은 잡음을 가진 두 측정의 일치).

## 2. 결과 (z = online encoder; h 는 같은 값 ± .02)

| 표적 | 팬 (양성) | SSv2 | EK100 | flow 자기 일관성 (SSv2 / EK100) |
|---|---|---|---|---|
| (a) 전역 ego-motion x / y | .99 / — | .15 / .26 | **.69 / .63** | 전역: .22 / .18 · **.70 / .50** |
| (b) 국소 flow x / y | .93 / — | .29 / .05 | .42 / .33 | 토큰: .20 / .00 · .31 / .24 |
| (b') 국소 잔차 (물체 · 시차) | — (0) | .22 / .01 | .04 / .09 | — |
| (c) 미래 변위, 슬롯 0 → 7 (x) | .93 전 슬롯 | .19 → .18 | .29 → .15 | — |
| 셔플 | .50 (y 상수 산물) | ≈ 0 | ≈ 0 | |
| 미래 변위 중 전역 (카메라) 몫, 슬롯 0 / 3 / 7 | — | .17 / .29 / .34 | **.42 / .65 / .74** | |

- **EK100: encoder 는 ego-motion 을 표적 신뢰도의 천장까지 읽는다** — 전역 R² .69 / .63 vs flow 자기 일관성 .70 / .50, 국소 .42 / .33 vs .31 / .24. "1 인칭 카메라 운동을 encoder 가 못 담아서 EK100 reach 가 2.2 에 멈춘다" 는 **기각**. RETHINK §1-b 의 미확정 칸 → 확정 (encoder 충분).
- **SSv2** 도 같은 결론이지만 절대값이 낮은 이유는 표적이다: Farneback flow 의 자기 일관성이 .20 / .00 (손 · 물체 영상에서 flow 가 불안정) 이라 R² .29 / .05 는 천장 근처다. 팬 (GT) 에서 .93–.99 로 판독 자체는 정상.
- **EK100 의 미래 변위는 대부분 카메라 운동이다** (슬롯 7 에서 74 %). predictor 가 EK100 에서 이어 가야 할 상태의 주성분은 **전역 ego-motion** 이고, encoder 는 그것을 담고 있다. 그런데 어떤 predictor 도 (릴리즈 · full · prefix · AR · post-FT) EK100 reach 2.2 를 넘지 못한다 → 병목은 encoder 가 아니라 **predictor 가 전역 운동을 미래 자리로 옮겨 쓰는 것**. 설계 원칙 "무엇/어디 분리" 의 EK100 판 근거.
- 토큰 하나에서 미래 변위를 읽는 R² 가 슬롯이 갈수록 준다 (.29 → .15) — 먼 미래의 출처는 한 토큰의 선형 함수가 아니다 (전역 + 국소 합성이 필요). predictor 가 해야 할 계산이 그것이다.

> ⚠️ **적대 검증 3 차 정정 ([`VERIFY_R3_2026-09-29.md`](VERIFY_R3_2026-09-29.md) B)** — EK100 전역 R² .69 / .63 은 **확정** (10-fold .709 / .651, 영상 단위 split .685 / .614, 64-토큰 부표본 평균 .682 / .613, 셔플 ≈ 0, clip 누출 검사 +.04). 그러나 §2 의 "flow 자기 일관성 천장" 은 train = test 최소제곱 값이었다 — clip-CV 로 다시 재면 EK100 전역 **.53 / .36** (토큰 .23 / .18), SSv2 .15 / .09, 팬 (GT) .75. encoder R² 가 그것을 **넘는다** → 이 값은 천장이 아니라 하한 성격의 참조다. "천장까지 읽는다" 를 "flow 표적의 자기 일관성보다 높게 읽는다" 로 고친다 (encoder 충분성 결론은 더 강해진다). 74 % 는 중앙값 .74 · 평균 .75 · 화면 안 출처만 .67 → **67–75 %** 로 쓴다.

## 3. 단서

- 표적이 flow 라 절대 R² 는 flow 잡음에 묶인다. "천장 근처" 는 자기 일관성 대리에 기대는 말이다 (두 flow 측정이 같은 계통 오차를 공유하면 상한이 과소일 수 있다).
- 토큰 평균으로 전역을 읽는 것은 자리 정보를 버린 판독이다. predictor 입력 (토큰별) 에 그 정보가 **쓰기 좋은 형태**인지는 별개 (RETHINK §1 "못 배제한 것").
- 국소 잔차 (물체 운동) 는 EK100 에서 거의 못 읽는다 (.04 / .09) — 1 인칭에서 물체 운동은 카메라 운동에 묻힌다 (표적 잡음 포함). 이건 encoder 의 한계일 수도 표적의 한계일 수도 있다 — 미결.
- 1 차. 적대 검증 (표적 정의 · fold · α 선택 · 셔플) 필요.

## 재현

```bash
# 추출 (vll1, K400 val 로컬 · SSv2 tar · EK100 스테이징)
sbatch --job-name=ego --export=ALL,CMD="bash auto_research/scripts/chain_0929ego.sh" auto_research/scripts/run2_vll1.sbatch
sbatch --job-name=ek_vll1 --export=ALL,CMD="bash auto_research/scripts/chain_0929ek_vll1.sh" auto_research/scripts/run2_vll1.sbatch
# 분석 (로그인 노드, OMP 8 스레드, 셋 합쳐 ~10 분)
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
for s in pan ssv2 ek100; do OUT_TAG=_$s OMP_NUM_THREADS=8 $P auto_research/scripts/e_ek_egomotion_analyze.py auto_research/_stage/results_ar/ego_$s; done
```
flow 자기 일관성 · 전역 몫은 이 문서 §2 의 인라인 계산 (fut[0] vs flow 최소제곱; 슬롯별 clip 중앙값 설명 비율) — 스크립트화 안 함 (수치는 세션 로그).
