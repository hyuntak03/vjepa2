# 복사 기준선이 높다는 것의 뜻 — VoE 점수는 물체의 **위치**를 보는가 (2026-10-07)

**사용자 질문**: moving_visible (ramp) 에서 복사 (= 정지한 세계) 가 98 % 면, metric 이 물체 위치를 전혀 반영하지 않는 것 아닌가.
**답: 맞다.** 토큰 평균 L1 surprise 는 "무엇이 / 있는지가 바뀌었나" 를 보고 "어디 있나" 는 거의 못 본다. 두 근거.

## 1. 외형 위반 (v11 · IntPhys 1) 에서는 복사 ≈ predictor

matched 쌍은 문맥이 같고 미래만 다르다. 가능 · 불가능 미래 **둘 다** 물체가 같은 궤적으로 움직이므로 위치 오차는 양쪽 surprise 에 똑같이 들어가 **상쇄**되고, 남는 차이는 위반 (사라짐 · 모양 · 색) 뿐이다. 그 차이는 마지막 프레임 특징과의 거리 (= 복사) 로도 보인다.
→ DINO-F v11 77.1 vs 복사 76.4 · IntPhys 1 84.4 vs 복사 88.9 · ViT-H IntPhys 1 88.9 vs 복사 85.0 (`DINOF_V11`, `DINOF_INTPHYS1`, `H6`).

## 2. 위치**만** 다른 쌍 (RollOut_v2 ledge · wall, V-JEPA 2 ViT-H 공간) 에서는 둘 다 chance 아래

가능 = 낙하 / 불가능 = 떠서 직진 (ledge), 가능 = 벽에서 정지 / 불가능 = 통과 (wall). 문맥은 같고 미래의 물체 **위치**만 다르다. surprise = mean |pred − LN(h)[미래 8 튜블릿]|, 캐시 (`rollout_v2_vith`, `rollout_v2_z16_vith`) 에서 계산, 392 쌍씩.

| 쌍 | predictor p | 복사 (LN(z) 마지막 튜블릿 × 8, 하네스 `_copy_baseline` 과 같은 정의) | 복사 (문맥 8 튜블릿 평균) |
|---|---:|---:|---:|
| ledge 낙하 vs 부유 | **0.0 %** | 7.9 % | 1.5 % |
| wall 정지 vs 통과 | **17.6 %** | 2.3 % | 66.3 % |

- ledge 0 %: predictor 의 미래는 392 쌍 전부에서 **떠 있는 (불가능) 미래에 더 가깝다.** RollOutV2 §5-5 의 "p 는 선반 높이 유지, 392 쌍 100 %, pos/imp probe P(imp) 0.91–0.96" 과 같은 사실의 VoE 판. 복사도 7.9 % — 정지한 세계도 부유 쪽.
- wall: p 17.6 % (통과 쪽 — "2 슬롯 통과" 와 일치). 복사 2.3 % 는 직관 (정지 = 마지막 프레임) 과 반대인데, 토큰 평균 L1 에서 물체 (반폭 18 px, 256 토큰 중 몇 개) 의 자리 차이는 배경 · 공간 (z vs h) 차이에 묻힌다는 뜻이다. 문맥 평균 복사가 66 % 인 것도 위치가 아니라 다른 요인 (문맥 평균이 정지 장면의 전역 통계에 가까움) 으로 읽어야 한다.
- 즉 **위치만 다른 쌍에서 이 metric 은 정답률이 chance 근처거나 아래**다. v11 · IntPhys 1 의 70–90 % 는 위치가 아니라 외형 사건에서 온다.

## 좋은 신호인가

- **논문에는 좋은 신호**다. "벤치마크가 predictor 의 예측을 재지 않는다" 의 가장 직접적인 그림: 외형 위반은 복사로 풀리고, 위치 위반은 predictor 도 복사도 못 푼다 (predictor 는 틀린 쪽을 고른다).
- **벤치마크 · metric 으로는 나쁜 신호**다. 토큰 평균 거리는 공간 · 시간 축을 둘 다 버려 궤적 위반을 원리적으로 못 잡는다 (CLAUDE.md §10 마지막 줄, H6). 위치를 재려면 (i) 위치만 다른 쌍 (v11 · IntPhys 1 에는 없음 — IntPhys 1 dev 에 궤적 · 속도 위반 0 건) 과 (ii) 토큰 평균이 아닌 거리 (물체 자리의 국소 거리, 또는 `FUTURE_FROM_CONTEXT` 처럼 자로 읽은 위치의 오차) 가 둘 다 필요하다. ⚠️ `token_subset: object` (물체 토큰만) 는 v11 외형 위반에서 **더 나빠졌다** (§6 기각) — 위치 쌍에서는 안 재 봤다.

## 재현
```bash
srun --jobid=<vll5 job> --overlap -N1 -n1 bash -c 'source /data/hyuntak/anaconda3/bin/activate vjepa2; cd /data/hyuntak/project/2026/2027_cvpr/vjepa2; python auto_research/scripts/rollout2_copy_voe.py'
# 산출 auto_research/exp_results/scene/rollout2_copy_voe.json
```
