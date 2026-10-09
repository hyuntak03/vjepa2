# IntPhys 2 Main "permanence" 감사 — predictor vs 복사 기준선 vs Ariel (2026-09-26)

> 상태: **1 차 · 적대 검증 전. copy w48 · **w32 반영 (2026-09-26 오후)**.** 문헌 문서 (`LITERATURE_STATE_WORLD_MODELS_2026-09-26.md` §2-0) 의 긴장 "IntPhys 2 는 permanence 가 가장 쉬운 조건 (고정 카메라 >60 %) 인데 우리는 predictor 가 가려진 내용을 못 들고 간다고 한다" 를 푼다.
> 수치 출처: `auto_research/exp_results/verify/intphys2_permanence_audit.json` (`auto_research/scripts/intphys2_permanence_audit.py`), `intphys2_condition_ci.json`. 원자료 `z_research/Benchmarks/exp_results/intphys2/<tag>/per_video.csv`.
> ⚠️ IntPhys 2 Main 은 논문 A.3 이 학습 금지한 세트다 — 여기서는 채점만 한다. 하네스는 `analysis/intphys2` (Bordes 프로토콜: 6 fps, 창 M, 문맥 C 스윕, growing prefix, AvgSurprise, type-matched 쌍 506). 우리 표는 **run 의 best C 하나** 로 읽고, 논문식 "열별 최고 C" 는 따로 적는다 (선택 편향 있음).

## 1. 가설과 판정 규칙

| | 가설 | 판정 |
|---|---|---|
| H1 | 재등장 불일치는 **가져오기만으로** 잡힌다: 예측 없는 복사 기준선 (문맥 마지막 튜블릿 LN 복사, H6 의 copyz) 이 permanence 쌍에서 predictor 만큼 맞힌다 | pred − copy 의 쌍 차이 CI 가 0 을 포함하면 H1 |
| H2 | 우리 실패는 **문맥 경계 corner case** — 가림 사건이 문맥/미래 경계에 걸릴 때만 | 하네스가 사건 프레임을 주지 않아 직접 검정 불가 (§4). 대리: avg 집계 vs **max 집계** (사건 시점 봉우리에 민감) |
| H3 | 보고 수치가 애초에 우연 근처 | scene bootstrap 95 % CI 가 50 을 포함하면 H3 |

## 2. 실행

```bash
# 기존 run (predictor): z_research/Benchmarks/exp_results/intphys2/bench_intphys2_main_{vith,vith_w32,vith_w16,ariel_bc_ep43_w32,ariel_bc_ep43_w16,ariel_bc_ep30_w48}
# 복사 기준선 (2026-09-26): analysis/intphys2/surprise.py 에 IP2_COPY=1 분기 (predictor 자리에 LN(z_ctx 마지막 튜블릿) 복사), 두 호출 지점 모두
sbatch analysis/intphys2/ip2_copy.sbatch        # vll3 4 GPU: smoke 20 영상 → w48 (C 12..42) → w32 (C 8..28), tag bench_intphys2_main_vith_copy_w{48,32}
python auto_research/scripts/intphys2_permanence_audit.py
```
smoke 검사: 같은 영상 · 같은 C 에서 copy surprise 0.851 vs predictor 0.611 (H6 의 IntPhys 1 척도 copy 0.813 / p 0.593 과 같은 방향). 복사 분기가 실제로 켜졌다.

## 3. 결과 — 조건 × 카메라 쌍 정확도 (%, scene bootstrap 95 % CI, run 의 best C)

### 3-1. predictor (release ViT-H) 와 Ariel

| | n 쌍 | release w48 (C24) | release w32 (C24) | release w16 (C10) | Ariel43 w32 (C8) | Ariel43 w16 (C10) | Ariel30 w48 (C42) |
|---|---:|---|---|---|---|---|---|
| overall | 506 | 54.3 [50.4, 58.1] | 52.8 [49.0, 56.9] | 53.2 [49.6, 56.7] | 53.8 [50.0, 57.5] | 52.4 [48.4, 56.1] | 53.0 [49.2, 56.7] |
| **permanence · Fixed** | 52 | 59.6 [44.2, 75.0] | 57.7 [44.2, 71.2] | 57.7 [46.1, 71.2] | **71.2 [59.6, 80.8]** | 59.6 [46.2, 73.1] | 57.7 [48.1, 67.3] |
| permanence · Moving | 68 | 45.6 [35.3, 54.4] | 51.5 [41.2, 60.3] | 52.9 [44.1, 61.8] | 39.7 [29.4, 51.5] | 48.5 [39.7, 55.9] | 47.1 [39.7, 54.4] |
| permanence · all | 120 | 51.7 [43.3, 60.0] | 54.2 [45.8, 61.7] | 55.0 [47.5, 62.5] | 53.3 [44.2, 61.7] | 53.3 [45.8, 60.8] | 51.7 [45.8, 57.5] |
| continuity · all | 120 | 61.7 [54.2, 69.2] | 58.3 [49.2, 67.5] | 56.7 [48.3, 65.0] | 62.5 [55.8, 68.3] | 58.3 [49.2, 67.5] | 62.5 [55.0, 70.8] |
| immutability · all | 120 | 50.8 [42.5, 60.0] | 50.0 [40.8, 59.2] | 51.7 [42.5, 60.0] | 48.3 [40.8, 55.8] | 46.7 [37.5, 55.0] | 45.0 [35.8, 53.4] |
| solidity · all | 146 | 53.4 [47.9, 58.9] | 49.3 [44.5, 54.1] | 50.0 [44.5, 54.8] | 51.4 [45.9, 56.8] | 51.4 [45.2, 57.5] | 52.7 [46.6, 58.2] |

- **H3 (우연 근처):** release 의 permanence 는 Fixed · Moving · 전체 모두 CI 가 50 을 포함한다 (Fixed 59.6 [44.2, 75.0], n = 52 쌍). "가장 쉬운 조건" 은 논문 표의 열별 최고 선택 + 모델 차이이고, ViT-H 릴리즈에서는 permanence 가 우연과 구분되지 않는다. 전체에서 우연을 넘는 조건은 **continuity** 뿐 (61.7 [54.2, 69.2]).
- Ariel ep43 w32 만 permanence · Fixed 71.2 [59.6, 80.8] 로 CI 가 50 을 넘는다. 그러나 같은 모델의 Moving 은 39.7 [29.4, 51.5] 로 **우연 아래** 이고 전체는 53.3 이다. 즉 카메라 고정 장면에서만 (배경이 정지 → 복사가 잘 맞는 조건) 오른다.
- 열별 최고 C 로 고르면 release permanence Fixed 는 w32 C12 61.5 까지 올라간다 (선택 편향; 6 개 C 중 최고).

### 3-2. 집계 방식 (H2 의 대리): avg vs max

| permanence | avg 집계 | max 집계 |
|---|---|---|
| release w48 Fixed / Moving | 59.6 / 45.6 | 51.9 / 44.1 |
| Ariel43 w32 Fixed / Moving | **71.2** / 39.7 | 46.2 / 50.0 |

- max 집계 (사건 시점의 봉우리에 민감) 로 바꾸면 Ariel 의 Fixed 71.2 가 46.2 로 사라진다. permanence 의 신호는 **사건 순간의 놀람이 아니라 창 전체에 퍼진 평균 차이** 다. 재등장 뒤 장면이 "다르다" 는 전역 불일치를 평균이 잡는 것과 맞는다 (H1 쪽).

### 3-3. 복사 기준선 (IP2_COPY=1)

w48 (C 12..42, best C 42) 완료. w32 는 실행 중 (216651) — 끝나면 같은 스크립트로 붙인다.

| | n 쌍 | **copy w48 (C42)** | release w48 (C24) | release − copy (같은 쌍) | Ariel30 w48 − copy |
|---|---:|---|---|---|---|
| overall | 506 | **55.5 [51.8, 59.1]** | 54.3 [50.4, 58.1] | −1.2 [−5.5, +3.0] | −2.6 [−6.5, +1.0] |
| permanence · Fixed | 52 | **63.5 [51.9, 75.0]** | 59.6 [44.2, 75.0] | −3.8 [−17.3, +9.6] | −5.8 [−15.4, +3.8] |
| permanence · Moving | 68 | 51.5 [42.6, 60.3] | 45.6 [35.3, 54.4] | — | — |
| permanence · all | 120 | 56.7 [49.2, 63.3] | 51.7 [43.3, 60.0] | −5.0 [−13.3, +3.3] | −5.0 [−11.7, +0.8] |
| continuity · all | 120 | 64.2 [55.8, 72.5] | 61.7 [54.2, 69.2] | −2.5 [−10.0, +4.2] | −1.7 [−9.2, +5.8] |
| immutability · all | 120 | 47.5 [39.2, 56.7] | 50.8 [42.5, 60.0] | +3.3 [−7.5, +14.2] | −2.5 [−11.7, +6.7] |
| solidity · all | 146 | 54.1 [48.6, 59.6] | 53.4 [47.9, 58.9] | −0.7 [−6.8, +4.8] | −1.4 [−6.8, +3.4] |

- **예측 없는 복사와 predictor 를 구분할 수 없다** (⚠️ 2 차 정정 — 원문 "같거나 조금 낫다"). 전체 55.5 vs 54.3, permanence Fixed 63.5 vs 59.6 은 **각자 최고 C (복사 42 · release 24)** 에서의 값이고, **같은 C = 24 로 맞추면 release − 복사 = +1.8 [−1.8, +5.5] (전체), permanence 전 조건 +2.5 [−4.2, +10.0]** 로 부호가 뒤집힌다 ([`VERIFY_ISCENE2_2026-09-26.md`](VERIFY_ISCENE2_2026-09-26.md) §8). −5 pt 는 best-C 선택의 산물이다. 모든 조건에서 pred − copy 의 CI 가 0 을 포함한다 (release · Ariel30 모두).
- copy 의 permanence · Fixed 63.5 [51.9, 75.0] 는 CI 하한이 50 을 넘는다 — **"가장 쉬운 조건" 의 신호는 예측 없이 나온다.** 재등장 뒤 장면과 마지막 관측의 전역 불일치 (평균 L1) 만으로 잡히는 것이다.

## 4. 어느 가설이 남나

- **H3 는 이미 선다.** release ViT-H 의 IntPhys 2 permanence 는 CI 로 우연과 구분되지 않는다. "permanence 가 쉽다" 는 우리 모델 · 우리 채점에서는 성립하지 않는다.
- **H1 확정 (1 차):** 복사 기준선이 같은 쌍에서 predictor 와 같다 (차이 CI 가 모든 조건에서 0 포함; ⚠️ 2 차 정정: "복사가 −5 pt 앞" 은 best-C 산물, 같은 C 에서는 +2.5 [−4.2, +10.0] — **구분 안 됨**). max 집계로 바꾸면 신호가 사라지는 것 (§3-2) 과 합쳐, IntPhys 2 permanence 점수는 사건 순간의 예측이 아니라 **재등장 뒤 장면 vs 마지막 관측의 전역 불일치** 를 평균으로 잡은 것이다.
- **H2 는 이 하네스로 검정 불가.** `metadata.csv` 는 `SceneIndex, name, file_name, game_name, condition, env, type, occluder, Difficulty, Camera` 뿐이고 사건 프레임이 없다. 창 별 surprise 궤적 (`per_window.jsonl`) 은 있으나 사건 시점 정답이 없어 "경계에 걸린 창" 을 정의할 수 없다. 우리 v11 (가림 시점을 설계한 세트) 이 H2 의 자리다.

## 5. 이야기에 미치는 것

- 문헌 긴장은 **모순이 아니다.** IntPhys 2 permanence 점수는 (i) ViT-H predictor 에서 우연 근처이고, (ii) **예측 없는 복사 기준선이 같거나 더 높으며** (Fixed 63.5), (iii) 오르는 경우도 사건 순간이 아니라 전역 평균에서 온다. 즉 이 벤치마크의 permanence 는 "가려진 내용을 들고 가는가" 를 재지 않는다 — IntPhys 1 (복사 85.0 vs 88.9) 과 같은 결론이 IntPhys 2 로 확장된다 (평가 감사 표에 추가).
- 따라서 초록의 "occluded content is not carried" 는 IntPhys 2 수치와 충돌하지 않는다. 다만 우리 쪽 근거도 아직 합성 물체 (v11 · RollOut) 라 장면 · 실영상 판 (묶음 Q) 이 필요하다.

## 6. 단서

- n 이 작다 (조건 × 카메라 52–94 쌍). CI 폭 ±15 pt.
- 논문 Table 2 는 열별 최고 run 이라 우리 단일 C 값보다 높게 나온다 (release overall 우리 54.3 vs 논문 57.51; 논문은 hyper-parameter 12 회 중 최고).
- Ariel 은 SSv2 + K400 학습이라 IntPhys 2 렌더 도메인 밖이다.
- 복사 기준선은 LN 을 씌운 판 (H6 관례). LN 없는 copy_raw 는 IntPhys 1 에서 7 pt 낮았다 — 필요하면 같은 변형을 여기서도 돌린다.
- max 집계는 이상치에 약하다 (한 창의 잡음이 결정). avg/max 차이는 방향 근거로만.

## 재현

§2 의 명령. 조건 CI 재계산은 CPU 몇 분 (`intphys2_permanence_audit.py`).

## 부록 A. copy w32 · Ariel w32 · w16 반영 (2026-09-26 오후, `intphys2_permanence_audit.py` 재실행)

- copy w32 (best C 28): overall 54.2 [50.6, 57.9]; permanence Fixed 55.8 / all 53.3; continuity all 60.0.
- release − copy (w32): overall −1.4 [−5.7, +3.2]; permanence Fixed +1.9 [−9.6, +13.5], all +0.8 [−7.5, +9.2]. w48 과 같이 모든 permanence 차이의 CI 가 0 을 포함한다. immutability Fixed 만 +11.5 [+1.9, +23.1] (w32) 로 CI 가 0 을 넘지만 all 로는 +2.5 [−6.7, +11.7].
- Ariel43 − copy (w32): permanence Fixed +15.4 [+1.9, +30.8] 이지만 Moving 39.7 (우연 아래) 이라 all 로는 +0.0 [−8.3, +9.2]. Ariel w16 permanence Fixed 59.6 [46.2, 73.1] 로 w32 의 71.2 가 창 선택에 민감함을 보인다.
- 판정 유지: H1 (조회만으로 잡히는 재등장 불일치) · H3 (우연 근처). 창 (w16/w32/w48) 을 바꿔도 같다.
