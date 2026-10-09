# SC2v2 — 실영상 (SSv2) 에서도 predictor 적중은 **변위** 로 떨어진다. flow 유사 정답 (v3 보정) · predictor 넷 (2026-09-25)

> 상태: **1 차 결과, 적대 검증 전.** oral 방향 sanity check 의 둘째다 ([`../paper/ORAL_DIRECTION_2026-09-25.md`](../paper/ORAL_DIRECTION_2026-09-25.md)).
> 수치 출처: `exp_results/ssv2/flowtrack.json` (SSv2 넷) · `flowtrack_v3_rel_ariel.json` (v3 보정) · 텍스트판 `flowtrack.txt`.
> 표적: encoder 판독은 **인과 템플릿** Tc = hc_7[q] (문맥 16 장만 본 target encoder) 를 **양방향 h_{8+i}** 에 건다. predictor 판독은 같은 Tc 를 p_i 에 건다. 표준 템플릿 T = h_7[q] 판 (`pT`) 도 같이 싣는다.

## 0. 한 줄

**SSv2 에서도 predictor 가 물체를 제자리에 두는 확률은 마지막 관측에서의 변위가 커질수록 떨어진다.**
- 대상은 encoder 판독과 flow 가 같은 칸을 가리키는 쌍뿐이다.
- 절반이 되는 변위는 release 약 3.4 칸, Ariel 약 4.6 칸이다.
- 네 predictor 모두 칸당 −0.07 ~ −0.08, 슬롯당 −0.04 ~ −0.06 이다.
- 자연 영상으로 미래 예측을 학습한 Ariel 은 모든 변위에서 release 보다 높다 (같은 쌍 +0.09 ~ +0.24). 그래도 곡선은 같은 모양으로 떨어진다.
- 합성으로 학습한 pv1 · v11_postft 는 SSv2 에서 release 와 거의 같다 (4–6 칸에서만 +0.10 / +0.12).

## 1. 왜 · 무엇을 바꿨나

- **SC2 1 판은 무효였다** (`e_ssv2_track.py`). 유사 정답 = encoder 템플릿 argmax 궤적이었다.
  - 궤적이 슬롯 사이에서 평균 5.5 칸씩 뛰었다. 이동의 55 % 가 4 칸 이상이다.
  - 질의 선택 에너지가 clip 사이에서 평평했다 (10–90 분위 0.99–1.04).
  - 그래서 그 수치는 쓰지 않는다. 산출물 `ssv2_track*.json` 은 남겨 두되 인용하지 않는다.
- **2 판** (`e_flowtrack.py`): 유사 정답을 encoder 와 독립인 **픽셀 optical flow** (OpenCV Farneback) 추적으로 바꿨다.
  - 질의 q 는 인과로 고른다. 문맥 끝 flow (프레임 14→15) 가 가장 큰 토큰 주변 3×3 토큰 창에서, 움직이는 화소의 크기 가중 중심을 구한다. 그 중심을 한 프레임 민 자리가 출발점이다.
  - 추적: 한 프레임씩 16×16 창의 flow 중앙값으로 이동한다.

## 2. 판독 보정 (판독이 맞는지 먼저)

| 검사 | 결과 | 어디서 |
|---|---|---|
| flow 추적 vs GT (v3, CPU 59 clip) — 첫 판 (토큰 중심 출발) | 1 칸 안 17–31 % → **폐기** | `_verify_scratch/flow_calib_cpu.py` |
| 〃 둘째 판 (12→15 합산 중심) | 41–51 % (2–3 프레임 뒤처짐) → **폐기** | 〃 |
| 〃 최종 (14→15 중심 + 한 프레임 밀기) | **슬롯 0–7 모두 92–95 %** (GT 변위 최대 7 칸) | 〃 |
| flow vs GT (v3, GPU 588 clip, 32 장 균등 배치) | 1 칸 안 **0.87–0.92** | `flowtrack_v3_rel_ariel.json` `flow_vs_gt` |
| 같은 지표를 flow 로 잰 값 vs GT 로 잰 값 (v3, release) | 변위별 P(hit_p) 0.98 / 0.97 / 0.93 / 0.88 / 0.47 / 0.05 (flow) vs 0.98 / 0.99 / 0.95 / 0.89 / 0.47 / 0.05 (GT) | 〃 `flow_valid` · `gt_valid` |
| SSv2 에서 encoder 판독 vs flow 합의 | 움직임 clip 슬롯 쌍의 **42 %** (v3 88 %). encoder 진행률 중앙값은 슬롯 0–4 에서 1.00 | `flowtrack.json` |

- **합성에서는 flow 유사 정답이 GT 와 같은 결론을 낸다.**
- **실영상에서는 encoder 천장이 낮다** (슬롯 0 에 0.62, 슬롯 7 에 0.26). 그래서 주 지표를 **두 독립 유사 정답 (flow · encoder) 이 합의한 쌍** 위의 P(hit_p) 로 둔다.
  - 이렇게 하면 늦은 슬롯에서 천장 자체가 떨어지는 교란이 빠진다.
  - 합의 여부는 predictor 와 무관하다. 그래서 predictor 사이 짝 비교가 된다.

## 3. 결과 (SSv2 좌/우 677 clip, 문맥 끝 속도 ≥ 2 px/프레임 598 clip, 합의 쌍 2,031)

### 3-1. 변위별 P(hit_p) (합의 쌍, 슬롯 풀링)

| 변위 (칸, Cheb) | n | release | Ariel | pv1 | v11_postft | 등속 오라클 |
|---|---:|---:|---:|---:|---:|---:|
| 0–1 | 248 | 0.73 | **0.85** | 0.71 | 0.75 | 0.81 |
| 1–2 | 519 | 0.67 | **0.77** | 0.67 | 0.69 | 0.68 |
| 2–3 | 348 | 0.53 | **0.62** | 0.51 | 0.51 | 0.64 |
| 3–4 | 262 | 0.34 | **0.52** | 0.35 | 0.33 | 0.47 |
| 4–6 | 335 | 0.13 | **0.37** | 0.24 | 0.26 | 0.27 |
| 6–9 | 254 | 0.06 | **0.17** | 0.11 | 0.11 | 0.16 |
| 9+ | 65 | 0.03 | 0.09 | 0.01 | 0.14 | 0.00 |

- **짝 차이 (같은 쌍, clip bootstrap).**
  - Ariel − release: +0.12 [0.04, 0.18] / +0.09 / +0.10 / +0.18 [0.10, 0.26] / **+0.24 [0.18, 0.30]** / +0.11 / +0.06 [−0.03, 0.16].
  - pv1 · v11_postft − release: 4–6 칸에서만 +0.10 / +0.12 이고, 나머지는 CI 가 0 을 포함한다.
- **회귀 (합의 쌍) P(hit_p) ~ 1 + slot + cell.**
  - release: 슬롯 −0.061 [−0.071, −0.051] · 칸 −0.077 [−0.087, −0.067].
  - Ariel: 슬롯 −0.042 · 칸 −0.079.
  - pv1: 슬롯 −0.044 · 칸 −0.073.
  - v11_postft: 슬롯 −0.045 · 칸 −0.071.
- **등속 오라클** = 문맥 끝 flow 속도로 출발점을 그대로 밀어 둔 자리다 (학습 없음). 2–4 칸에서는 release 가 이 오라클보다 낮고 (1–2 칸은 같다), Ariel 은 3–6 칸에서 오라클보다 높다.

### 3-2. 합성 (v3) 과 나란히

같은 배치 (32 장, 문맥 8 / 미래 8, stride 2) 와 같은 판독에서 쟀다. 합의 쌍 4,093 (88 %) 이다.

| 변위 (칸) | 0–1 | 1–2 | 2–3 | 3–4 | 4–6 | 6–9 | 9+ |
|---|---:|---:|---:|---:|---:|---:|---:|
| release v3 | 0.98 | 0.97 | 0.93 | 0.88 | **0.47** | 0.05 | 0.00 |
| Ariel v3 | 1.00 | 0.99 | 0.96 | 0.89 | **0.80** | **0.69** | 0.17 |
| Ariel − release (v3) | +0.02 | +0.01 | +0.03 | +0.01 | **+0.34** | **+0.64** | +0.17 |

- 합성에서는 절벽이 가파르다. release 는 4–6 칸, Ariel 은 6–9 칸 뒤다 (SC1 의 거리 법칙과 같은 자리).
- 실영상에서는 두 predictor 모두 더 일찍, 더 완만하게 떨어진다.

## 4. 단서

1. **실영상에서는 슬롯 효과도 있다** (슬롯당 −0.04 ~ −0.06). 합성의 Ariel 처럼 "거리만" 은 아니다.
   - 늦은 슬롯에서 작게 움직인 물체도 놓친다. release 는 변위 0–1 칸에서 이른 슬롯 0.81 vs 늦은 슬롯 0.42 다.
   - 손 가림 · 변형 · 카메라 운동 같은 실영상 변화가 원인 후보다. 가르지 않았다.
2. **Ariel 은 SSv2 train 으로 학습했다.** 이 clip 들이 그 train 에 들었는지 확인하지 못했다. release 의 사전학습 데이터 (VideoMix22M) 에도 SSv2 가 들어 있다.
   - 그래서 Ariel 우위에는 도메인 안 이점이 섞여 있을 수 있다.
3. **유사 정답 합의는 42 % 뿐이다.** 합의 쌍은 추적이 쉬운 (질감이 뚜렷한) 물체 쪽으로 치우친다.
4. **질의는 clip 당 하나** (가장 움직이는 것) 다. 손일 수도 물체일 수도 있다. 구분하지 않았다.
5. **변위 구간의 표본이 predictor 와 무관하게 같다.** 짝 비교는 공정하다. 다만 9+ 칸은 n = 65 로 작다.
6. **flow 추적도 오류가 있다** (v3 에서 5–13 %). 실영상에서는 더 클 것이다. 합의 조건이 이 오류를 줄인다.

## 재현

```bash
bash auto_research/scripts/chain_0925i.sh     # e_flowtrack.py: v3 보정 (release · Ariel) + SSv2 (predictor 넷), vll6 8 GPU 약 30 분
C=/data2/local_datasets/world/world_analysis/cache/auto_research
python auto_research/scripts/e_flowtrack_analyze.py release=$C/flowtrack_ssv2_release ariel=$C/flowtrack_ssv2_ariel pv1=$C/flowtrack_ssv2_pv1 v11ft=$C/flowtrack_ssv2_v11ft
OUT_TAG=_v3_rel_ariel python auto_research/scripts/e_flowtrack_analyze.py release=$C/flowtrack_v3_release ariel=$C/flowtrack_v3_ariel
python auto_research/_verify_scratch/flow_calib_cpu.py      # CPU flow 보정 (모델 없음)
```
