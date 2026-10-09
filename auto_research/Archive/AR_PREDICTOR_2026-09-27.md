# 자기회귀 predictor (Ariel 팔 C, kind=ar) 를 같은 판독에 — 2026-09-27

> 방향 정본 [`../paper/DIRECTION_2026-09-25.md`](../paper/DIRECTION_2026-09-25.md) §4 "rollout 학습도 같은가" 행의 결정 실험.
> 체크포인트 · 이식 정본 `z_training/ARIEL_CHECKPOINTS.md` (다른 세션). 이 문서는 **판독 결과** 만 다룬다. 모든 수치 1 차 · 적대 검증 전.

## 0. 왜 이게 결정 실험인가

I-scene 의 세 결과 (reach 2–5 칸 · permanence 1/3 · closure 자기 예측 되먹임 불가) 는 전부 **한 번에 예측하는 (mask token) predictor** 에서 쟀다.
릴리즈 · Ariel prefix · post-FT 넷 다 그렇다. 반론: "되먹여서 학습하면 (rollout objective) 저절로 해결되는 것 아닌가."
Ariel 팔 C 는 정확히 그 predictor 다 — 입력 = 타깃 = LN(target_encoder(블록)), mask token 없음, 학습 손실에 8 블록 rollout 이 든다 (V-JEPA 2-AC 레시피, action 없음).
같은 데이터 · 같은 예산의 full-attention (팔 A) · prefix (팔 B) 와 **구조만** 다르다 (단, 인코딩 단위 · 문맥 encoder · 입력 형식이 같이 바뀐다 — `ARIEL_CHECKPOINTS.md` §1 경고).

| 결과 | 읽기 |
|---|---|
| AR 의 reach · permanence 가 A · B 와 같다 | 되먹임 학습만으로는 안 된다 → "state 를 굴리는 모듈" 이 목표 정의가 아니라 **구조** 여야 한다는 근거 강화 |
| AR 이 확실히 낫다 | **DC (design choice) 가 확정된다**: rollout 목표 + 되먹임 = 답. 논문의 방법 절이 "AR predictor + 우리 프로토콜" 로 바로 선다 |

어느 쪽이든 beat 가 선다 (§3 여섯 질문 2 번 통과).

## 1. 사용자 보고 (2026-09-27, IntPhys 1 dev, Garrido 격자 · 쌍 정확도, 다른 세션 산출)

| 모델 | skip2_w32 | skip2_w16 | skip5_w16 | best | O1 / O2 / O3 (best 칸) |
|---|---|---|---|---|---|
| ViT-H 릴리즈 | 88.89 | 83.33 | 60.56 | 88.89 | 85.0 / 96.7 / 85.0 |
| full-attn (ep40) | 71.67 | 65.56 | 54.44 | 71.67 | 75.0 / 80.0 / 60.0 |
| block-causal prefix (ep45) | 70.00 | 63.89 | 53.33 | 70.00 | 76.7 / 71.7 / 61.7 |
| **AR (ep18)** ⚠️ | 87.78 | **90.56** | 77.78 | **90.56** | 86.7 / 96.7 / 88.3 |

- 같은 데이터 · 예산의 A · B (70 %) 와 AR (90 %) 사이 **20 pt**. AR 은 학습 중 (18/40 epoch) 인데도 릴리즈를 넘는다.
- ⚠️ **읽기 전 단서**: AR 의 타깃 공간은 블록별 LN(target_encoder(2 장)) 이라 표준 (창 전체 인코딩) 과 다르다. 쌍 정확도만 비교 가능.
  **AR 의 복사 기준선 (h_blk[C−1] 반복) 이 아직 없다** — 표준 판에서는 복사가 85.0 이었다. 블록별 인코딩은 시간 문맥이 없어 복사가 더 잘 맞을 수도, 덜 맞을 수도 있다. 이게 나오기 전에는 "AR 이 물리를 더 안다" 로 쓰지 않는다. (다른 세션 항목)
- skip5_w16 (긴 간격) 에서 AR 77.8 vs 릴리즈 60.6 — 프레임 간격이 넓을수록 격차가 커진다. reach 와 맞물리는지 §2 에서 본다.
- ✅ **AR 공간 복사 기준선 도착 (다른 세션, `z_training/ARIEL_CHECKPOINTS.md` §5-2, 2026-09-28)**: `copy_blk` (문맥 마지막 블록의 블록별 LN(target_encoder) 을 미래 전 블록에 복사) = **83.33** (skip2_w16). AR ep18 90.56 − 83.33 = **+7.2**; 릴리즈는 자기 공간 복사 85.00 대비 +3.9. 두 여유의 차 3 pt 는 180 쌍 SE ≈ 3 pt 안이다. → **블록별 공간은 복사만으로 83 이고, AR 의 90 중 상당 부분이 공간 몫이다.** "AR 이 물리를 더 안다" 로 쓰지 않는다. 벤치마크 감사 표 (IntPhys 1: 복사 85 / 83 vs predictor 89 / 91) 에 AR 행이 추가된다.
- ✅ **AR ep36 · ep38 (09-28)**: IntPhys 1 92.78 (ep18 → +2.2, 4 쌍, 잡음 안) · IntPhys 2 50.79 (우연) · v11 83.7. 학습 끝물이라 거의 안 움직인다. 고정 사본 `z_training/ariel_eval/ckpt/ar_ep38.pt`. 판독 재실행 (§2-4) 은 이 사본으로.

## 2. 판독 결과 (이 문서의 본체) — 실행 중

스크립트 `scripts/e_scene_ar.py` (`--task reach|perm`), 분석 `scripts/e_scene_ar_analyze.py` · 기존 `e_scene_permanence_analyze.py`.
판독 정의는 I-scene 과 같고 (Chebyshev ≤ 1 · 합의 토큰 · clip bootstrap · R50), **AR 만 템플릿이 h_blk[7]** (AR 이 실제로 입력으로 먹은 마지막 관측) 이다. 대조로 hc_L 템플릿 판 (`AR:roll_x`) 도 낸다.
팔: `R:base` 릴리즈 · `F:base` full ep40 · `B:base` prefix ep45 · `AR:roll` (자기 예측 되먹임 8 블록) · `AR:tf` (진짜 블록 되먹임 한 스텝 = ar_z 의 AR 판).
`AR:roll − AR:tf` 가 closure 의 AR 판이다 (되먹임 학습을 했으니 0 에 가까워야 한다).

| 판 | 데이터 노드 | job | 상태 |
|---|---|---|---|
| 팬 800 reach · SSv2 677 reach · 팬 900 perm | vll2 (K400 /data2, SSv2 tar 스테이징) | 217111 | 큐 |
| EK100 800 reach · v3 reach | vll3/5/6 | 217112 | 큐 (GPU 대기, 최대 ~20 h) |

### 2-1. reach — 팬 (K400 val 인공 팬 800 clip, 1 차) ✅ 2026-09-27 19:17

`exp_results/scene/scene_ar_ar_pan.json`. A(d) = 합의 토큰 적중률 (팔마다 자기 템플릿의 encoder 천장으로 합의).

| 팔 | 0–1 | 1–2 | 2–3 | 3–4 | 4–6 | 6–9 | 9–13 | **R50** | 슬롯 0 → 7 | 회귀 슬롯당 / 칸당 |
|---|---|---|---|---|---|---|---|---|---|---|
| R 릴리즈 | .69 | .71 | .65 | .54 | .32 | .07 | .06 | 4.83 | .77 → .28 | −.035 / −.077 |
| F full ep40 | .74 | .78 | .75 | .72 | .63 | .30 | .07 | **6.99** | .80 → .51 | −.006 / −.071 |
| B prefix ep45 | .75 | .79 | .74 | .72 | .60 | .25 | .05 | 6.57 | .81 → .48 | −.008 / −.077 |
| **AR rollout (자기 되먹임)** | .78 | .63 | .48 | .36 | .21 | .08 | .03 | **3.24** | .74 → .26 | −.029 / −.084 |
| AR teacher-forced (진짜 블록 되먹임) | .78 | .70 | .67 | .62 | .60 | .48 | .29 | 9.19 | .74 → .58 | +.006 / −.046 |
| AR rollout, hc_L 템플릿 (대조) | .51 | .42 | .35 | .23 | .16 | .09 | .06 | 3.31 | — | — |

Δ (같은 토큰, clip bootstrap 95 %):
- **AR rollout − AR tf**: 0–1 칸 0, 1–2 −.07, 2–3 −.18, 3–4 −.27, 4–6 −.39, 6–9 −.41. 한 걸음 (진짜 블록을 주면) 은 잘 옮기는데 **자기 예측을 되먹이면 걸음마다 무너진다.**
- **AR rollout − B prefix** (교집합 토큰): 1–2 −.13, 2–3 −.24, 3–4 −.38, 4–6 −.43. **되먹임 학습 predictor 가 한 번에 예측하는 predictor 보다 reach 가 짧다.**
- F − R / B − R: 3–4 +.18 / +.17, 4–6 +.31 / +.28 — 같은 데이터 · 예산에서 full 과 prefix 는 **같다** (attention 규칙 무관, I-scene §4-1 rule swap 의 학습 판 확인).

왜 무너지나 (같은 산출물에서):
- rollout 출력의 cos(출처) 와 cos(복사) 가 슬롯이 갈수록 붙는다 (슬롯 0 .725 / .710 → 슬롯 7 .616 / **.642**, 슬롯 4 부터 복사가 출처를 넘는다). tf 는 끝까지 출처 > 복사 (.672 / .610). argmax 가 자기 칸으로 무너지는 건 아니다 (8 %) — **뭉개진다 (특정 자리 없이 평균 같은 토큰).** I-scene §4-3 B "beyond-reach 출력 절반 blur" 와 같은 모양이 AR 에서는 3 걸음 만에 온다.
- 즉 closure (자기 출력을 입력으로 못 먹는다) 가 **되먹임을 손실에 넣고 학습해도 남는다** (ep18 기준).

- ⚠️ **적대 검증 ([`VERIFY_AR_2026-09-27.md`](VERIFY_AR_2026-09-27.md))**: reach 최단 **확정** — 교집합 합의 토큰 R50 3.43 vs B 6.63, 정확 칸 2.21 vs 6.25, 느린 팬 (v=2, 블록당 0.25 칸) 에서도 1–3 칸 −.15 / −.24 → 템플릿 · 걸음 크기 산물 아님. AR 은 **0 칸 (정지 내용) 에서는 최고** (정확 칸 .43 vs B .35) 이고 결손은 전부 움직이는 내용 (≥ 1 칸) 에 있다. "뭉개짐" 은 **시사**: 출처 < 복사 역전은 슬롯 **5** (4 가 아님) 이고 릴리즈도 같은 슬롯에서 뒤집힌다 — AR 의 뭉개짐이 릴리즈보다 심한 건 아니다. 자기 되먹임 붕괴는 **슬롯 1 부터** (−.042 [−.046, −.037]).
읽기 (1 차): DIRECTION §4 "rollout 학습도 같은가" 의 답은 팬에서 **"같거나 더 나쁘다"** 다. rollout 목표 하나로는 상태가 굴러가지 않는다.
IntPhys 1 90.56 과 나란히 두면 — 벤치마크가 이 능력을 안 잰다는 §1 의 감사 결과가 AR 에서 가장 날카롭게 나온다 (reach 최저 · IntPhys 최고). ⚠️ AR 복사 기준선 (다른 세션에 요청) 이 와야 닫힌다.



### 2-1-b. reach — SSv2 (677 clip, flow 출처, 1 차) ✅ 19:30

`exp_results/scene/scene_ar_ar_ssv2.json`. 팬과 같은 그림이 실영상에서 그대로 나온다.

| 팔 | 0–1 | 1–2 | 2–3 | 3–4 | 4–6 | 6–9 | **R50** | 슬롯 0 → 7 |
|---|---|---|---|---|---|---|---|---|
| R 릴리즈 | .54 | .46 | .29 | .17 | .06 | .02 | 2.69 | .66 → .22 |
| F full ep40 | .63 | .56 | .45 | .37 | .25 | .15 | **4.17** | .74 → .38 |
| B prefix ep45 | .63 | .56 | .45 | .36 | .24 | .14 | 4.06 | .75 → .38 |
| **AR rollout** | .61 | .42 | .25 | .17 | .07 | .01 | **2.18** | .74 → .38 |
| AR teacher-forced | .67 | .60 | .58 | .55 | .52 | .48 | ∞ (절반 아래로 안 감) | .74 → .60 |

- AR rollout − B prefix (교집합): 1–2 −.15, 2–3 −.22, 3–4 −.22, 4–6 −.19. **실영상에서도 되먹임 학습 predictor 의 reach 가 가장 짧다** (릴리즈보다도 짧다).
- AR rollout − tf: 1–2 −.19, 2–3 −.33, 4–6 −.45. 한 걸음은 거리 무관하게 맞히는데 (tf 6–9 칸 .48) 되먹이면 무너진다.
- F ≈ B 도 그대로 (Δ vs R 둘 다 +.16 / +.19 / +.19).
- ⚠️ 적대 검증: 교집합 R50 2.21 vs B 4.13, 정확 칸 1.14 vs 2.20 — **확정**. 슬롯 1 부터 −.054.

### 2-1-c. reach — EK100 (800 clip, flow 출처) · RollOut_v3 (588 clip, GT 출처) — 1 차, 2026-09-27 실행 · 09-29 집계

`exp_results/scene/scene_ar_ar_{ek100,v3}.json`. 네 판 전부 같은 순서: **AR rollout < 릴리즈 < full ≈ prefix ≪ AR teacher-forced.**

| R50 | 팬 | SSv2 | EK100 | v3 (물체) |
|---|---|---|---|---|
| R 릴리즈 | 4.83 | 2.69 | 2.12 | 4.65 |
| F full ep40 | 6.99 | 4.17 | 2.17 | 7.34 |
| B prefix ep45 | 6.57 | 4.06 | 2.17 | 7.76 |
| **AR rollout** | **3.24** | **2.18** | **1.75** | **4.24** |
| AR teacher-forced | 9.19 | ∞ | 4.00 | ∞ |

- **EK100**: AR rollout − B (교집합) 1–2 칸 −.13, 2–3 −.09; 4 칸 이상은 넷 다 바닥 (≤ .07) 이라 차이 없음. F/B − R 은 EK100 에서 +.03 ~ +.05 로 거의 사라진다 (I-scene §5-3 · §5-4 의 "1 인칭에서는 학습이 아무것도 안 준다" 재확인). AR tf 도 EK100 에서는 4.0 에 그친다 — 한 걸음 전이조차 1 인칭 카메라 운동에서는 짧다.
- **v3 (정지 배경 + 물체 하나)**: AR rollout − B 는 **0–1 칸 (정지 배경) +.26** 인데 물체 (≥ 1 칸) 에서 −.15 ~ −.28. 팬 검증과 같은 그림 — **AR 은 정지 내용은 가장 잘 놓고 움직이는 내용은 가장 못 옮긴다.** tf 는 6–9 칸에서도 .94 (물체 한 걸음 전이는 거의 완벽).
- AR roll − tf 는 v3 에서 −.69 (4–6 칸) · −.90 (6–9 칸) 까지 벌어진다. 자기 되먹임 붕괴가 가장 극적으로 보이는 판이다.

### 2-2. permanence — 팬 + 화면 고정 가림 띠 (900 clip, 1 차 + 적대 검증) ✅ 19:50

`exp_results/scene/scene_permanence_ar.json` (`e_scene_permanence_analyze.py`). hid = 프레임 15 에서 띠 안 (마지막 관측에 없음), vis = 같은 clip 의 띠 밖. 판독 템플릿 = 가려지기 전 튜블릿 (표준 팔 hc[L−m], AR 은 h_blk[7−m]).

| k=4, 변위 | R 릴리즈 hid / vis | F full hid / vis | B prefix hid / vis | **AR rollout hid / vis** | AR rollout cos(원래 내용) / cos(가림막) |
|---|---|---|---|---|---|
| 0–1 칸 | .21 / .42 | .21 / .45 | .21 / .46 | **.26 / .55** | .57 / .64 |
| 1–2 | .21 / .38 | .19 / .43 | .20 / .42 | .18 / .39 | .57 / .61 |
| 2–3 | .17 / .25 | .14 / .32 | .17 / .31 | .13 / .27 | .58 / .60 |
| hid/vis 비 (0–2 칸, k=2 / 4 / 8, 검증 재계산 CI) | .76 / **.53 [.50, .56]** / .50 | .71 / .47 / .49 | .68 / .46 / .49 | **.59 / .47 [.44, .50] / .45** | — |

- **AR 도 절반이다.** 되먹임 학습이 permanence 를 만들지 않는다. ⚠️ 적대 검증 ([`VERIFY_AR_2026-09-27.md`](VERIFY_AR_2026-09-27.md)) 정정: k=4 비는 F/B 와 같지만 **릴리즈 (.53) 보다 낮고** (CI 분리), k=2 는 AR 이 **가장 낮다** (.59 vs R .76 / B .68). 블록별 인코딩이라 encoder 안의 역사 경로 (가려지기 전 프레임을 같은 창에서 보는 길) 가 없기 때문으로 읽는다 (해석).
- "가림막을 그린다" 는 AR 에서 **약하다** — hid 토큰 cos(내용) − cos(가림막) k=4 −.045 (R −.10, B −.135), k=2 n.s. AR 은 가림막을 다른 팔보다 덜 그린다.
- k 단조 (k2 > k4 ≈ k8) 는 넷이 같다.
- ⚠️ `AR:tf` 행은 permanence 측정이 **아니다** — 진짜 미래 블록 (가림막 없는 프레임) 을 되먹이므로 슬롯 1 부터는 재관측이다 (hid 적중 roll .19 평평 vs tf .18 → .50). 슬롯 0 은 rollout 과 같아 rollout 행이 포함한다.

### 2-3. closure — AR:roll vs AR:tf 가 곧 closure 다 (§2-1 의 Δ 행)

되먹임 학습 predictor 에서 "자기 예측 되먹임 (roll) − 진짜 관측 되먹임 (tf)" 는 팬 2–3 칸 −.18 · 4–6 칸 −.39, SSv2 −.33 · −.45. I-scene §4-5 의 `ar_pA − ar_z` (릴리즈 −0.1 ~ −0.2, Ariel −0.2 ~ −0.3) 와 같은 크기다. **rollout 손실로 학습해도 자기 출력은 관측을 대신하지 못한다** (ep18). 슬롯별: roll .74 → .26 (팬) / .74 → .38 (SSv2), tf .74 → .58 / .60. ⚠️ 적대 검증: 차이는 **첫 자기 되먹임 걸음 (슬롯 1) 부터** 유의하고 단조 (팬 −.04 → −.35). 슬롯 0 은 roll = tf (argmax 99.1 % 일치, bf16 잡음). `AR:roll_x` 는 AR 에 불리한 대조일 뿐 비교 팔이 아니다.



### 2-4. AR ep38 (고정 사본) 재판독 — 결론 유지 ✅ 2026-09-29

`exp_results/scene/scene_ar_ar38_{pan,ssv2}.json`, `scene_permanence_ar38.json`. meta.ckpt_epoch = {F 40, B 45, AR 38}.

| | 팬 R50 | SSv2 R50 | 팬 Δ(roll − B, 교집합) 3–4 / 4–6 | SSv2 Δ 2–3 / 3–4 | perm hid/vis (k=4, 0–1 칸) |
|---|---|---|---|---|---|
| AR ep18 | 3.24 | 2.18 | −.38 / −.43 | −.22 / −.22 | .26 / .55 (비 .47) |
| **AR ep38** | **3.91** | **2.22** | −.27 / −.34 | −.20 / −.21 | .26 / .56 (비 .46) |
| (릴리즈 / prefix) | 4.83 / 6.57 | 2.70 / 4.05 | — | — | .21/.42 (.50) / .21/.46 (.46) |

- 팬에서 +0.7 칸 늘었지만 **여전히 릴리즈 아래**, SSv2 는 변화 없음. teacher-forced 는 9.9 / ∞ 그대로. 자기 되먹임 붕괴 (roll − tf) 도 그대로 (팬 4–6 칸 −.36, SSv2 −.47).
- EK100 (스테이징 800 clip, 09-29 저녁): AR ep38 rollout R50 **1.77** (ep18 1.75), tf 4.37, 릴리즈 2.11, full/prefix 2.17 — 그대로. `exp_results/scene/scene_ar_ar38_ek100.json`.
- permanence 도 그대로 (.46). → **§2-1 ~ 2-3 의 결론은 학습 완료 판에서도 성립.** "학습 중이라서" 반론 해소. AR 절을 **확정 (ep38)** 으로 올린다 (EK100 ep38 은 실행 중, v3 는 ep18 값만).

### 2-5. `z_training/ariel/context_full_ar_1e_5/latest.pt` 의 정체 (2026-09-29 확인) — **ctx_ar 이 아니라 AR ep40 (nat_ar_scratch 최종)**

- 근거: arch.kind = `ar`, `type_embed` 키 없음 (ctx_ar 은 type embedding 2 개를 가진다), config.folder = `nat_ar_scratch`, epoch 40 · step 35,840 · loss 1.04417 = `z_ariel/vjepa2_train_code_20260929_ctx_ar/z_training/runs/nat_ar_scratch/metrics.jsonl` 의 epoch 40 행과 일치.
- 진짜 ctx_ar run (`nat_ctx_ar_scratch`, kind `ctx_ar`: 문맥은 창 단위 z (prefix 팔과 같은 online encoder) + 미래만 블록별 LN(h) 되먹임, type embedding 으로 구분) 은 **epoch 15/40 학습 중** (in-loop IntPhys1 60 쌍 0.72). 체크포인트가 오면 §2-6 으로 건다. 코드 스냅샷 `z_ariel/vjepa2_train_code_20260929_ctx_ar/README_SNAPSHOT.md`.
- 그래서 `chain_0929ar2.sh` 의 `AR2` 팔 = **AR ep40** 이다 (ep38 대조로 쓴다). 문서 · 그림에서 "context_full_ar" 라는 이름은 쓰지 않는다.
- ctx_ar 이 왜 중요한가: AR 의 결손 후보 셋 (블록별 문맥 인코딩 · EMA 문맥 · 되먹임 입력) 중 **문맥 인코딩만 prefix 팔과 같게** 되돌린 팔이라, AR 이 정지 내용에 강하고 움직이는 내용에 약한 것이 블록별 문맥 (역사 경로 없음) 탓인지 되먹임 탓인지를 가른다.

### 2-6. 모순 기록 — 채점 (v11 late 가림) 에서는 AR 이 혼자 높다 (2026-09-29, [`COPY_HARD_SUBSET_2026-09-29.md`](COPY_HARD_SUBSET_2026-09-29.md))

| v11 late · vanish 물체→빈 (112 쌍씩) | static | flat | ramp |
|---|---|---|---|
| 복사 | 67.9 | 39.3 | 9.8 |
| 릴리즈 | 65.2 | 2.7 | 0.0 |
| prefix ep45 | 49 | 38 | 14 |
| **AR ep38** | **99.1** | **85.7** | **72.3** |
| v11_e10 (도메인 안) | 87 | 100 | 100 |

copy-hard 쌍 전체에서도 AR 73.8 vs 릴리즈 · prefix · full 48–51. 그런데 §2-2 의 장면 판독에서 AR 의 permanence 는 다른 팔과 같은 절반 (.46) 이고 reach 는 최단이다.
- 두 측정은 정의가 다르다: 판독 = 가려졌던 내용을 **어디에** 놓았나 (Chebyshev ≤ 1 자리 적중), 채점 = 전 토큰 L1 로 두 미래 중 어느 쪽에 가까운가 (**무엇이 있나** 에 가까움).
- 잇는 가설: **AR 은 "돌아온다" 는 알지만 (presence) "어디로" 는 모른다 (placement).** AR 이 정지 내용에서 최고 (§2-1 검증) 인 것과 맞물린다. 검정: v11 late 클립에 AR 출력의 presence 판독 (RollOutV3 identity_r8 류 · 또는 물체 템플릿 cos) 을 걸어 자리 무관 presence 가 다른 팔보다 높은지 — GPU 큐 (v11 프레임은 vll5 /local_datasets).
- ✅ **검정 결과 (2026-09-29 밤, 1 차)** — `scripts/e_v11_presence.py` (v11 split_test late · vanish 336 block = 672 쌍, 학습된 자 없음): 토큰별 margin M = |p − h_imp| − |p − h_pos| (+ = 가능 쪽), 두 미래가 갈리는 자리 Dh = |h_pos − h_imp| (물체가 다시 나타나는 칸). placement share = M⁺ 질량 중 Dh 상위 5 % 칸의 몫 (균등 .05). `exp_results/scene/v11_presence_v1.json`.

| obj 쌍 (물체 있는 문맥, 가능 = 다시 나타남) | 정답률 | 복사 | placement share | margin @ 재등장 칸 (Dh 상위) | margin @ 나머지 |
|---|---|---|---|---|---|
| 릴리즈 · moving / flat / static | .00 / .05 / .66 | .05 / .36 / .69 | .076 / .083 / .105 | −.020 / −.009 / +.009 | −.006 / −.003 / .000 |
| prefix ep45 | .11 / .38 / .46 | 〃 | .084 / .088 / .105 | −.010 / −.001 / +.006 | −.003 / −.001 / −.001 |
| **AR ep40** | **.67 / .86 / 1.00** | .02 / .03 / .68 | **.068 / .064** / .113 | **−.040 / −.029** / +.029 | **+.011 / +.016** / +.009 |
| ctx_ar ep16 | 1.00 / 1.00 / .85 | 〃 | .056 / .055 / .115 | +.018 / +.014 / +.016 | +.035 / +.033 / +.002 |
| empty 쌍 (빈 문맥, 가능 = 계속 빈) 정답률 | 릴리즈 .92 · prefix .95 · **AR .99** · **ctx_ar .21** | 복사 .78 / .90 | | | |

- **모순 해소 (1 차): AR 의 late 가림 정답은 presence-without-placement 다.** 움직이는 물체에서 AR 이 맞히는 쌍의 margin 은 **재등장 칸에서 음수** (−.04 / −.03: 그 자리에서는 오히려 "사라진" 미래에 가깝다) 이고 **나머지 칸에서 양수** (+.011 / +.016) 다. placement share 는 균등 (.05) 근처 (.064–.068) 로 다른 팔보다도 낮다. 즉 "무언가 돌아온다" 는 전 장면에 퍼진 신호로 맞히고, "어디로" 는 틀린다 — 장면 판독의 permanence 절반 · reach 최단과 **양립**한다. 정지 물체 (static) 만 자리도 맞다 (share .113, 재등장 칸 +.029) — 같은 자리라 복사와 같다.
- 릴리즈 · prefix 는 재등장 칸에서도 나머지에서도 음수 → 사라진 미래 쪽. 복사와 같은 오류.
- ⚠️ ctx_ar ep16 은 obj 쌍 100 % 인데 **empty 쌍 21 %** — 문맥에 물체가 없어도 "물체 있는" 미래를 고른다 (나머지 칸 +.035). 이건 지식이 아니라 편향이다 (ep16). A/B 양방향을 같이 보지 않으면 놓친다 (CLAUDE.md §1-7).
- ⚠️ **적대 검증 ([`VERIFY_PRESENCE_2026-09-29.md`](VERIFY_PRESENCE_2026-09-29.md))** — 표 수치 소수점 3 자리까지 재현. 정정 · 보강:
  - 재등장 칸 음수는 q 1/2/5/10 % · 슬롯별/합침 · **릴리즈 공간 Dh** 로도 유지 (표적 공간 산물 아님). 다만 **AR 만의 것이 아니다** — 릴리즈 · prefix 도 재등장 칸 (ring 0–1) 에서 음수. **AR 만의 것은 먼 칸 (ring ≥ 3) 의 양수** 다. ring 분석: AR moving ring 0 −.057 · 1 −.038 · 2 −.004 · 3–6 +.011~+.018 → 근접 오차 봉우리가 없다 = 자리 오류가 아니라 **자리 부재**.
  - "전 장면에 퍼진" → **"먼 칸 (≥ 3 칸) 에 퍼진, 가운데 띠 (행 6–9) 제외"** 로 정정. 공간적으로 균등하지 않다 (상위 10 칸이 양의 질량 12 %, 균등 3.9 %, Gini .45).
  - **예측 vs 문맥 편향 둘 다 있다 (시사)**: 가려진 슬롯 (물체가 아직 못 나올 때) 에도 작은 양의 margin (+.003~+.008, 슬롯 0 만으로 .75 정답) — 문맥에 물체가 있었다는 편향. 단 k=4 에서는 가려진 슬롯 ≈ 0 이고 재등장 뒤 단조 증가 (슬롯 7 +.011) — 주로 재등장 반응.
  - **AR 의 margin 지도는 복사 지도와 닮았다** (r = .59 obj / .67 empty; 릴리즈 .26 / .30) → "AR = 복사 지도 + 전역 오프셋". COPY_HARD 의 "오류가 복사와 덜 묶여 있다" 와 같이 읽는다.
  - sym_k 에 따라 AR 정답률 .94 / .92 / .80 / .71, 먼 칸 margin .020 → .005 로 준다; 재등장 칸은 모든 k 에서 음수.
  - "presence" 는 물체 정체가 아니라 **"물체 있는 미래" 쪽의 전역 편향** 이라는 뜻으로만 쓴다.
- 단서: Dh 상위 5 % 를 "재등장 칸" 으로 쓴 것은 encoder 가 정한 자리다 (GT 위치 아님; 상위 13 칸 중 24 % 만 최상위 칸 5×5 안, 행 0–2 에 몰림). 1 % 핵심 · 릴리즈 공간으로도 결론 불변. AR · ctx_ar 의 표적 공간은 블록별.
- 단서: AR 표적 공간 (블록별) 이 표준 복사와 덜 상관돼 copy-hard 가 AR 에 유리할 수 있다. IntPhys 1 copy_blk-hard 에서는 AR 57–63 vs 릴리즈 77 로 대칭이 나온다. v11 copy_blk 전체 (사용자 보류) 가 있어야 닫힌다.

### 2-7. AR ep40 최종 · ctx_ar ep16 (문맥 = 창 단위 z, 미래만 되먹임) — 1 차, 2026-09-29 밤

`exp_results/scene/scene_ar_ar2_{pan,ssv2}.json`, `scene_permanence_ar2.json` (고정 사본 `ariel_eval/ckpt/ar_ep40.pt` · `ctx_ar_ep16.pt`). ⚠️ ctx_ar 은 16/40 epoch (학습 중).

| R50 / 적중 | 팬 roll | 팬 tf | 팬 0–1 칸 (정지) | SSv2 roll | SSv2 tf | perm hid/vis (k=4, 0–1) |
|---|---|---|---|---|---|---|
| 릴리즈 | 4.83 | — | .69 | 2.70 | — | .21 / .42 (.50) |
| prefix ep45 | 6.57 | — | .75 | 4.05 | — | .21 / .46 (.46) |
| AR ep38 | 3.91 | 9.93 | .80 | 2.22 | ∞ | .26 / .56 (.46) |
| **AR ep40** | 3.97 | 10.0 | .80 | 2.24 | ∞ | .26 / .56 (.46) |
| **ctx_ar ep16** | 3.94 | 9.64 | **.70** | 2.77 | ∞ | .22 / .50 (.44) |

- **AR ep40 ≡ ep38** (모든 구간 Δ ≤ .01). AR 절 확정.
- **ctx_ar (문맥을 prefix 팔과 같은 창 단위 z 로 되돌린 팔)**: 자기 되먹임 reach 는 AR 과 같다 (팬 3.94 · SSv2 2.77; roll − tf 붕괴도 같은 크기 −.34 / −.41). **AR 의 정지 내용 우위는 사라진다** (0–1 칸 .80 → .70, Δ −.10 [−.10, −.10]) — 그 우위는 블록별 문맥 (입력 = 표적 공간, 복사가 곧 정답) 이 만든 것이었다. 움직이는 내용은 AR 과 같거나 SSv2 에서 +.03. permanence 도 절반 (.44).
- 읽기 (1 차, ep16): AR 의 결손 후보 셋 중 **문맥 인코딩은 원인이 아니다** — 문맥을 되돌려도 되먹임 reach · 붕괴 · permanence 가 그대로다. 남는 후보 = **되먹임 입력 구조 자체** (자기 출력을 다음 입력으로 쓰는 것). 40 epoch 판으로 재확인해야 한다.
- ⚠️ **적대 검증 3 차 ([`VERIFY_R3_2026-09-29.md`](VERIFY_R3_2026-09-29.md))**: AR ep40 ≡ ep38 확정. ctx_ar 의 reach 는 R50 은 같지만 구간별로 **양방향** — 팬 1–6 칸 −.035 ~ −.074 (CI 0 제외, 전 속도), SSv2 2–9 칸 +.02 ~ +.04 → "움직이는 내용은 AR 과 같거나 +.03" 은 팬에서 틀림 (정정). 정지 우위 상실은 확정이고 더 강함 (교집합 0–1 칸 .761 vs AR .844, **prefix .810 보다도 낮음**; 템플릿 산물 아님). 첫 걸음 붕괴는 AR 의 2–3 배 (−.064 / −.091). permanence 절반 확정 (.466 [.434, .497]). "문맥 인코딩은 원인이 아니다" 는 **시사** 로 낮춘다 — ep16 이라 정지 하락도 학습 부족으로 설명 가능. 안전한 문장: "창 단위 z 문맥으로 되돌려도 되먹임 reach · 붕괴 · permanence 는 회복되지 않는다."
- **벤치마크 (ctx_ar ep18, 09-29 22:02)**: IntPhys 1 best 64.4 (skip2_w16; w32 60.0, skip5 53.3) · v11_split_test **61.0** — 같은 epoch 의 AR ep18 (90.6 / 83.2) 보다 훨씬 낮고 릴리즈 · 복사 아래, full/prefix 수준 (`z_training/ARIEL_CHECKPOINTS.md` §5-1d). AR 의 정지 이점 (= 복사) 이 사라지고 되먹임은 그대로 불안정한 결과와 일관.
- 단서: ep16 이라 절대값은 오를 수 있다 (AR 은 ep18 → 38 에서 팬 +0.7). 붕괴 모양 (첫 걸음부터, 단조) 이 같다는 것이 핵심 관찰이고 이건 epoch 에 덜 민감했다 (AR ep18 vs 38).

### 2-8. AR ep40 의 p 를 새 정체 자로 읽다 — RollOut_v3 (가속 · 중력) · v11 late (2026-09-30) → [`AR40_READOUT_2026-09-30.md`](AR40_READOUT_2026-09-30.md)

- 자 `identity_ar40` (AR p 로 학습, test 조합 91.1 > 릴리즈 자 84.0). **가속 반영 α** (검증 뒤): arc −0.42 · ledge −0.46 (분포 안 t0–7, 정의 변형 전부 음수; flat_a n.s., ramp/감속은 분포 밖이라 제외) — 등속 외삽보다 뒤처짐, 릴리즈와 같거나 나쁨 (같은 자 안). **중력 · 가속을 미래에 넣지 않는다.** h 천장 α ≈ 1. α 절대값은 자 의존 (AR 자 위치 오차 1.16 vs 0.85 칸).
- 정체 유지 더 짧음 (t8–15 '있음' 41–90 % vs 릴리즈 83–98 %). **late 가림 뒤 움직이는 물체 '있음' 0–8 % = 빈 장면 오탐 (2.5 %) 과 구분 안 됨** (릴리즈 40–90 %, 자리 77–100 px 틀림) → v11 late 채점 정답은 물체 토큰이 아니다 (§2-6 과 정합). 1 차 · 적대 검증 진행 중.

### 2-9. EK100 action anticipation probe 를 AR ep40 으로 학습 (2026-09-30, 진행 중)

> ⚠️ **2026-10-01 정정 — 아래 "AR ep40 val 21.74" 는 AR 이 아니다.** `SET=model_kwargs.predictor_checkpoint=…` 가 `model_kwargs` **최상위**에 실렸는데 `eval.py` 는 `model_kwargs.pretrain_kwargs` 만 `init_module` 에 넘긴다 → 키가 조용히 무시되고 **릴리즈 predictor 로 돌았다.** 증거: (i) stdout 에 `predictor <- … (kind ar)` 로그가 없고 릴리즈 경로에서만 찍는 `mask_token norms` 가 있다, (ii) 같은 부분집합 릴리즈 run 과 epoch 마다 일치 (action 4.29/13.01/19.21/21.74 vs 4.26/13.00/19.23/**21.80**, verb 48.42 vs 48.26, noun 36.47 vs 36.04 — 시드 차이 수준). 그래서 이 두 run 은 **릴리즈 재현 2 회** (1/3 데이터 · 20 epoch: action 21.8 · verb 48.3 · noun 36.2 ± 0.2) 로 읽는다. 같은 버그로 09-30 의 `ek100_rel_third10_grid8_ar40` · `ek100_vith_official256_released_grid8_ar40` (전체 데이터) 도 AR 이 아니다 — **어디에도 인용하지 말 것.**
> 수정: `eval.py` 가 최상위 키를 `pretrain_kwargs` 로 넘기고, 체인의 smoke 가 `predictor <- … kind ar` 로그를 확인한 뒤에만 본 run 을 돈다. 재실행 `ek100_rel_third20_grid8_ar40_v2` (vll1, 2026-10-01; smoke 는 job 218056 에서 `predictor <- ar_ep40.pt (kind ar, epoch 40)` 확인 — smoke 만 2 h 15 m 걸려 14 h 상자에 본 run 12 h 가 안 들어가 취소하고, `SKIP_SMOKE=1` · 16 h 로 job 218060 재제출 05:55). 교훈: **12 시간짜리 run 은 "무엇이 로드됐는지" 로그 한 줄을 smoke 에서 assert 한 뒤 돌린다** (SKIP_SMOKE 로 건너뛴 것이 화근).

- 하네스 `evals/action_anticipation_frozen/modelcustom/vit_encoder_predictor_concat_ar_nonsquare.py` 에 `model_kwargs.predictor_checkpoint` (ckpt `arch.kind` 로 predictor 를 짓고, kind=ar 이면 문맥을 블록별 LN(target_encoder) 로 인코딩해 anticipation 걸음 + 예측 블록만큼 rollout, 걸음 수가 같은 샘플끼리 묶음). probe 의 encoder 입력 (창 전체 인코딩 4,096 토큰) 은 원래대로, predictor 토큰 256 개만 AR 출력 (블록별 LN 공간).
- 설정 (사용자 결정, 속도): train **1/3** (영상 단위 seed 0, 22,264 clip; `data_csv/ek100/EPIC_100_train_third_seed0.csv`), 20 epoch, batch 128, released 규약, grid8 head, val 5 epoch 마다. vll1 (EK100 66 GB 를 NFS tar 로 스테이징). 같은 부분집합에서 **릴리즈도 이어 돌려** 비교선을 만든다. 체인 `auto_research/scripts/chain_0930_ek100_third.sh`, TAG `ek100_rel_third20_grid8_{ar40,release}`.
- ⚠️ 전체 데이터 기준선 (35.34, 10.5k step) 과는 비교하지 않는다 (3.5k step). ⚠️ val n_clips 가 8,761 (기준선 9,296) — 같은 val csv · 같은 영상인데 줄었다; **원인 확인 (10-01)**: `epickitchens.filter_annotations` 가 "val 에서 train 에 없는 action (verb, noun) 쌍을 제거" 한다 (`epickitchens.py:261`). 1/3 train 에는 없는 action 쌍이 있어 val 535 clip (5.8 %) 이 빠진다 → 희귀 클래스가 빠진 **조금 쉬운 val** 이다. AR · 릴리즈 둘 다 같은 8,761 이라 내부 비교는 성립하고, 전체 데이터 기준선 (9,296) 과는 이 이유로도 비교 불가.
- epoch 1 학습 top-1 은 기준선과 같은 step 에서 ±0.5 pt (encoder 토큰 · 클래스 prior 가 만드는 값). step 11 s (GPU 100 %, VRAM 20 GB; 계산 병목).
- **AR ep40 val (1/3 · 20 epoch · 3,480 step, 2026-09-30 15:37 완료)**: epoch 5 → 10 → 15 → 20 = action **4.29 → 13.01 → 19.21 → 21.74** / verb 20.2 → 37.3 → 43.9 → **48.4** / noun 13.7 → 26.1 → 34.6 → **36.5** (val 8,761 clip). 아직 오르는 중 (ep15 → 20 +2.5) — 1/3 데이터 · 3.5k step 은 미수렴 구간. 릴리즈 같은 조건 run 이 15:38 시작 (완료 ≈ 10/1 04:00). **그 전에는 판정하지 않는다** (전체 데이터 기준선 35.34 와 비교 금지).

- **✅ AR ep40 재실행 결과 (`ek100_rel_third20_grid8_ar40_v2`, job 218060, 2026-10-01 05:55 → 21:15, 15.4 h — AR rollout 이 릴리즈보다 느리다)**. 로그에 `predictor <- ar_ep40.pt (kind ar, epoch 40)` 확인. 같은 1/3 부분집합 · 같은 val 8,761 clip:

| epoch | AR ep40 action / verb / noun | release action / verb / noun | (릴리즈 재현 2 회차, 참고) |
|---|---|---|---|
| 5 | 4.45 / 21.56 / 13.75 | 4.26 / 20.28 / 13.67 | 4.29 / 20.24 / 13.66 |
| 10 | 13.16 / 37.65 / 25.90 | 13.00 / 37.63 / 26.16 | 13.01 / 37.33 / 26.10 |
| 15 | 19.32 / 43.05 / 34.90 | 19.23 / 42.99 / 34.37 | 19.21 / 43.89 / 34.59 |
| **20** | **22.03 / 48.21 / 36.94** | **21.80 / 48.26 / 36.04** | 21.74 / 48.42 / 36.47 |

  **AR = 릴리즈.** 최종 action +0.23, verb −0.05, noun +0.90 인데 릴리즈 두 재현 사이의 차이가 action 0.06 · verb 0.16 · noun 0.43 이라 noun 도 잡음 2 배 안이다. 학습 top-1 은 AR 이 조금 낮다 (마지막 step action 63.9 vs 릴리즈 65.1 / 65.7 — val 과 반대 방향, 역시 잡음). per-clip 예측은 저장되지 않아 (`val_metrics.jsonl` 은 집계만) 부트스트랩 CI 는 못 낸다. 전체 데이터 기준선 (35.34 / copy_last 32.80, 10.5k step) 과는 step 수 · val 집합이 달라 비교하지 않는다.
  읽는 법: EK100 probe 는 encoder 토큰 (창 전체) + predictor 출력 2 블록을 attentive head 가 읽는다. predictor 를 rollout 학습한 AR 로 바꿔도 점수가 안 움직인다는 것은 (i) 이 probe 에서 predictor 몫 자체가 작고 (전체 데이터 기준 +2.5), (ii) 그 작은 몫도 "무엇을 예측하느냐" 보다 "문맥을 한 번 더 요약한 토큰" 에서 온다는 §2-x 복사 기준선과 같은 방향이다. 단서: 1/3 데이터 · 3.5k step 미수렴 구간 (ep15 → 20 +2.7) 이라 수렴 뒤 차이는 못 봤고, head 는 grid8 하나다.

## 3. 단서 (미리)

- AR ep18 은 학습 중이다. 40 epoch 판으로 다시 건다.
- AR 은 12 fps SSv2 + K400 320p 로 학습했고 판독 입력은 다른 팔과 같다 (팬 인공 · SSv2 균등 32 장 · EK100 ≈ 15 fps). 분포 차이는 넷에 공통.
- 팔 A · B 대비 AR 은 "규칙 하나만" 다르지 않다 (블록별 인코딩 · EMA 문맥 · 되먹임 입력). AR 이 이기면 **어느 요소 때문인지는 이 실험이 못 가른다** — 후속 (블록별 인코딩 + mask token 팔) 이 필요하다.
- 판독 템플릿이 팔마다 달라 (hc_L vs h_blk[7]) 합의 집합도 다르다. 교집합 토큰의 Δ 를 같이 낸다.

## 재현

```bash
sbatch --job-name=ar_scene --export=ALL,CMD="bash auto_research/scripts/chain_0927ar.sh" -o auto_research/_stage/results_ar/slurm_%j.out auto_research/scripts/run4_vll2.sbatch
sbatch --job-name=ar_ek    --export=ALL,CMD="bash auto_research/scripts/chain_0927ar_ek.sh" -o auto_research/_stage/results_ar/slurm_ek_%j.out auto_research/scripts/run4_ek.sbatch
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; R=auto_research/_stage/results_ar
OUT_TAG=_ar_pan  $P auto_research/scripts/e_scene_ar_analyze.py $R/scene_ar_pan
OUT_TAG=_ar_ssv2 $P auto_research/scripts/e_scene_ar_analyze.py $R/scene_ar_ssv2
OUT_TAG=_ar      $P auto_research/scripts/e_scene_permanence_analyze.py $R/scene_ar_perm
```
