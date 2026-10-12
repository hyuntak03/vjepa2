> **2026-09-26 정리 — 유효.** 자와 무관한 감사 (평균 풀링 probe · 기하) 라 지금도 정본이다. 결론은 [`../../Archive/RESULTS_2026-09-25.md`](../../Archive/RESULTS_2026-09-25.md) §6–7. 문서 안의 `new_archive/…` · `_audit/…` 는 옛 경로다 (→ `figures/…` · `audit/…`).

# 감사 — v11 정체 probe (`v11_probe`) · 풀링 기하 (`v11_geometry`) (2026-09-25)

> 수치는 모두 이 폴더의 `audit_probe.json` · `audit_twin.json` · `audit_geom.json` 에서 옮겼다 (`audit_v11_identity.py` 산출).
> probe 두 개로 다시 냈다: **logistic** = 레포 `v11_pooled_probe.train_head` 그대로 (results.json 을 한 자리까지 재현함) · **ridge** = 닫힌 해, 레포 probe 코드와 독립.
> NCM (표준화 공간 최근접 클래스 평균) 은 참고로만 돌렸다 — 조건 안에서도 shape 38–52 / color 13–26 % 라 (장면 분산에 묻힘) 판정에 쓰지 않는다.
> CI = test block (또는 성분) bootstrap 1,000 회 95 %. 표기 `logistic / ridge`, 칸은 `shape | color`.

## 0. 결론

1. **v11 의 궤적은 12 개뿐이고 (정지 8 자리 · 등속 2 방향 · 경사 2 방향) 모든 궤적이 train 과 test 양쪽에 있다.** 정체 라벨과는 완전히 교차한다 (궤적마다 7 shape · 8 color, Cramér V 0.022 / 0.024, 궤적→라벨 조회 11.5 / 10.1 % < 우연). 궤적 누수는 정체 probe 에 정보를 주지 않는다.
2. 대신 **외형 쌍둥이**가 있다 — test clip 의 83 % 는 같은 (운동·궤적·shape·color·env) clip 이 train 에 있다 (k 만 다름). 쌍둥이 없는 test clip 에서 가려진 튜블릿 · 마지막 튜블릿 읽기가 color 에서 **11–12 pt** 떨어진다. 장면 (궤적×env) 을 통째로 빼면 color 는 70–90 % 까지 떨어진다.
3. 두 probe 가 질적으로 같은 것: (2) encoder 문맥, (3) 가려진 문맥 튜블릿, (5) p 미래 (late, 튜블릿별 평평), (6) 같은 조건 짝 Procrustes 로 회복, 모든 짝 사상 하나로 두 조건 회복. **이식 크기는 probe 에 따라 최대 ~48 pt 다르다** (z 머리 → h_fut raw shape visible: logistic 83.7 / ridge 35.5).
4. **복사 기준선** — p_fut 의 정체 가독성은 문맥 전체 (z_all) 보다 낮고, z_all 에 p_fut 를 붙여도 늘지 않는다. 마지막 가려진 튜블릿 (z_t7) 보다는 color 에서 2.6–3.0 pt 높다. **"p 가 정체를 지킨다" 는 문맥 복사 수준 이상의 증거가 아니다.**
5. **PROBE.md §5-3 ("visible 짝 사상이 late 에 안 통한다 → late 의 p 는 또 다른 자리") 은 철회한다** — encoder 끼리의 같은 사상도 late 에서 같이 무너진다 (두 probe 모두).
6. 기하: 장면 몫 0.77–0.81 · 상수 이동 · shape 클래스 평균 부분공간 (p 0.785–0.809 vs encoder 0.944–0.962, 차 CI 가 0 을 넘지 않음) 은 독립 코드로 재현됐다. **color 쪽 부분공간 차는 약하다** (block 순열 귀무를 넘지 못하는 쌍이 하나 있다).

## 1. 궤적 반복과 쌍둥이

| | 값 |
|---|---|
| 서로 다른 궤적 (truth, 조건 무시 / 조건 포함) | 12 / 24 (같은 궤적 안 최대 편차 2.4e-5) |
| train·test 양쪽에 있는 궤적 | 24 / 24 · test clip 의 100 % |
| 궤적 → 라벨 조회 test 정확도 (shape / color) | 11.5 / 10.1 % (우연 14.3 / 12.5) · 궤적×env 11.96 / 10.7 |
| 같은 외형조합이 train 에 있는 test clip | 82.8 % (3,905 / 4,716) |

같은 머리로 test 를 쌍둥이 유무로 나눔 (block split 그대로, `audit_twin.json`, logistic / ridge):

| | 쌍둥이 있음 | 쌍둥이 없음 | 차 (없음 − 있음) |
|---|---|---|---|
| (2) z_all, late — shape | 99.2 / 99.7 | 96.8 / 98.5 | −2.4 / −1.2 |
| (3) z_hid, late k≥2 — shape | 96.5 / 97.5 | 91.5 / 93.2 | −5.0 / −4.3 |
| (3) z_hid, late k≥2 — color | 96.0 / 95.0 | **83.7 / 82.7** | **−12.4 / −12.4** (logistic [−16.7, −8.2]) |
| (5) p_fut, late k≥2 — shape | 97.5 / 96.7 | 97.3 / 94.2 | −0.2 / −2.4 |
| (5) p_fut, late k≥2 — color | 98.3 / 96.7 | 92.9 / 90.5 | −5.5 / −6.2 |
| 복사 z_t7, late k≥2 — color | 96.4 / 95.0 | **84.7 / 83.0** | −11.7 / −12.1 |

장면 hold-out (block ∪ 외형조합 연결성분 split — block 이 pos_a/pos_b 를 묶어 성분이 장면 통째가 됐다. test clip 의 63 % 가 처음 보는 궤적×env, 성분 bootstrap):
z_all late 95.8 / 92.5 | 88.5 / 90.4 · z_hid 87.8 / 88.7 | 72.0 / 70.6 · p_fut late 93.0 / 90.1 | 71.3 / 76.7 · z_t7 90.4 / 91.0 | 76.7 / 77.6.

→ **조건 안 95–99 % 는 "본 장면, 대부분 쌍둥이가 train 에 있는 clip" 의 수치다.** 가려진 튜블릿의 color 는 쌍둥이 없이 83 %, 장면 밖에서 71 % 다 (우연 12.5). 정보가 있다는 결론은 서지만 수준은 이 조건을 같이 적어야 한다.

## 2. 두 probe 로 재도출 (block split, `audit_probe.json → 2_probe_main`)

| 주장 | shape (logistic / ridge) | color (logistic / ridge) | 판정 |
|---|---|---|---|
| (2) z_all → late | 98.8 [98.3,99.3] / 99.5 [99.2,99.8] | 98.8 [98.4,99.3] / 98.3 [97.8,98.8] | holds (라벨 섞기 12.0/14.1 · 10.0/10.5) |
| (3) z_hid, late k≥2 | 95.7 [94.6,96.6] / 96.8 [95.9,97.6] | 93.9 [92.5,95.0] / 92.9 [91.6,94.2] | holds |
| (3) − 같은 clip 의 보이는 문맥 튜블릿 | −2.7 [−3.6,−1.7] / −2.4 [−3.2,−1.5] | −4.3 [−5.5,−3.2] / −4.8 [−6.1,−3.6] | holds |
| (5) p_fut late 머리 → late | 97.3 [96.5,98.0] / 95.9 [95.1,96.7] | 97.2 [96.5,97.9] / 95.5 [94.5,96.4] | holds (섞기 14.0/13.6 · 10.6/11.0) |
| (5b) 튜블릿별 p, late (t0–t7 범위) | 94.2–97.0 / 91.5–95.1 | 94.3–96.8 / 92.9–95.0 | holds (평평) |
| (6) z 머리 → p raw (vis / late) | 41.9 / 34.5 · **18.0 / 16.6** | 32.7 / 18.2 · **18.5 / 20.8** | holds_weaker (크기 불일치) |
| (6) h 머리 → p raw (vis / late) | 44.0 / 39.5 · 19.8 / 14.8 | 47.6 / 30.5 · 20.9 / 18.1 | holds_weaker |
| encoder 끼리 raw: z 머리 → h_fut | 83.7 / 66.9 · 35.5 / 39.8 | 57.4 / 48.4 · 60.2 / 44.5 | — |
| encoder 끼리 raw: h 머리 → z_all | 81.6 / 78.1 · 86.2 / 75.8 | 53.6 / 45.7 · **32.0 / 18.6** | — |
| Procrustes p→z, 같은 조건 짝 (vis→vis / late→late) | 99.8 / 96.4 · 99.7 / 94.9 | 99.2 / 96.9 · 99.0 / 94.4 | holds (짝 섞기 12.4/14.5 · 14.2/13.3) |
| Procrustes p→z, 전체 짝 하나 (vis / late) | 99.8 / 95.4 · 99.6 / 93.7 | 99.1 / 95.6 · 98.7 / 93.5 | holds |
| visible 짝 사상 → late: p→z | 54.5 · 18.4 | 59.2 · 35.8 | |
| 같은 것, encoder 쌍 h_fut→z / z→h | 65.0, 42.5 · 26.0, 29.1 | 44.6, 46.7 · 42.4, 43.1 | **§5-3 fails** |

- "raw 에서 p 만 encoder 와 어긋난다" 는 16 비교 중 15 개가 같은 방향이다 (encoder 끼리 > encoder → p). 예외는 ridge color h 머리 → z_all late (18.6 vs → p 18.1) 이다. **방향은 서고 크기는 probe 에 달렸다.**
- visible 짝 사상의 late 잔차 (라벨 없음): p→z 0.411 · p→h 0.619 vs h_fut→z 0.707 · z→h 0.623. p 의 사상이 encoder 쌍보다 late 로 더 잘 옮겨진다.

## 3. 복사 기준선 — late k≥2 (마지막 문맥 튜블릿에서 물체가 완전히 가려짐), 같은 late 머리 규칙

| 표현 (late 머리 → late k≥2) | shape | color |
|---|---|---|
| p_fut (미래 8 튜블릿 평균) | 97.5 / 96.2 | 97.4 / 95.6 |
| 복사 z_t7 (마지막 가려진 튜블릿) | 95.3 / 96.1 | 94.4 / 93.0 |
| 복사 z_t67 | 97.9 / 98.2 | 95.8 / 95.2 |
| 복사 z_all (문맥 전체) | 99.0 / 99.7 | 98.2 / 98.3 |
| p_fut − z_t7 (짝) | +2.1 [+0.9,+3.4] / +0.2 [−1.1,+1.4] | +3.0 [+1.9,+4.1] / +2.6 [+1.4,+3.8] |
| p_fut − z_all (짝) | −1.6 [−2.5,−0.8] / −3.5 [−4.4,−2.6] | −0.8 [−1.4,−0.1] / −2.7 [−3.7,−1.7] |
| [z_all, p_fut] − z_all (결합 머리) | −0.2 [−0.6,+0.1] / −0.1 [−0.3,+0.2] | +0.5 [+0.1,+0.9] / +0.2 [−0.2,+0.8] |
| 같은 풀링 크기 (튜블릿 하나): p[:,t] vs z_t7 | 94.2–97.0 vs 95.3 / 91.5–95.1 vs 96.1 | 94.3–96.8 vs 94.4 / 92.9–95.0 vs 93.0 |

쌍둥이 없는 clip: p_fut vs z_t7 = shape 97.3 vs 91.5 / 94.2 vs 93.9, color 92.9 vs 84.7 / 90.5 vs 83.0. 장면 hold-out: p_fut − z_all = color −11.8 [−17.0,−5.7] / −9.4 [−13.0,−5.8].

→ p 는 **마지막 가려진 튜블릿 하나보다는** color 정체를 더 잘 싣지만, **문맥 전체보다 낮고 문맥 전체에 더해 주는 것이 없다.** p 가 문맥 16 샘플 전체의 함수라는 점을 생각하면 "p 는 문맥이 가진 정체를 (일부 잃으며) 옮겨 싣는다" 까지다.

## 4. 기하 (`audit_geom.json`, 독립 numpy float64)

| 주장 | 재계산 | 판정 |
|---|---|---|
| 장면 칸 (env×cond×k) 분산 몫 | z 0.810 · p 0.784 · h_fut 0.772 · h_ctx 0.791; shape 0.031–0.039, color 0.005–0.006 | holds |
| 상수 이동 | 짝 제곱거리 중 중심 이동 몫 p–h_fut 0.819 / 0.837, z–p 0.927 / 0.949, h_ctx–h_fut 0.540 / 0.645; 중심 방향 한 축 분류 12 쌍×조건 전부 1.000 | holds (z·h 는 토큰 LN, p 는 아님 — 노름·퍼짐은 학습된 기하로 읽지 않는다) |
| shape 클래스 평균 부분공간 (resid, 교차 적합, split-half 정규화) | encoder 쌍 0.944–0.962, p 쌍 0.785–0.809 (vis·late). 차 (encoder 최소 − p 최대) vis 0.154 [0.147,0.154], late 0.135 [0.130,0.136]. block 순열 p95 0.39–0.63 < xfit 0.76–0.93 | holds |
| color 같은 것 | encoder 0.888–0.928, p 0.769–0.834. 차 vis 0.073 [0.009,0.079], late 0.054 [0.030,0.061]. **h_fut–p vis xfit 0.620 < block 순열 p95 0.639** | holds_weaker |
| 같은 표현 visible ↔ late | shape z 0.937 · p 0.944 · h_fut 0.953 · h_ctx 0.951; color 0.945 · 0.940 · 0.965 · 0.951 | holds |
| clip 단위노름 뒤에도 같은가 | shape p 쌍 0.784–0.805, encoder 0.941–0.960 | holds (노름만의 효과가 아니다; 토큰 LN 대조는 아님) |

bootstrap 은 block 을 복원추출한 가중 클래스 평균이다. 점추정이 CI 위쪽 끝에 붙는 칸이 있다 (복원추출이 잡음을 더해 captured 를 낮춘다) — CI 는 폭만 읽는다.

## 5. 쓸 수 있는 문장

- "v11 의 궤적은 12 개이고 정체 라벨과 완전히 교차해, block split 의 궤적 공유는 shape · color probe 에 정보를 주지 않는다."
- "context encoder 의 문맥 풀링에서 정체가 선형으로 읽힌다 (late 98.8 / 99.5 %, shape, logistic / ridge)."
- "물체가 완전히 가려진 문맥 튜블릿만 풀링해도 정체가 읽힌다 (shape 95.7 / 96.8, color 93.9 / 92.9 %). 같은 외형 clip 이 train 에 없으면 color 는 83 % 로, 처음 보는 장면에서는 71 % 로 낮아진다."
- "late 가림 clip 에서 p 미래 풀링의 정체는 late 전용 머리로 95–97 % 읽히고 미래 튜블릿마다 평평하다. 이 수준은 문맥 전체 복사 (98–100 %) 보다 낮고, 문맥에 p 를 더해도 늘지 않는다."
- "encoder 에서 배운 머리는 p 에 그대로 걸면 조건 안 수준보다 훨씬 낮다. clip 짝으로 맞춘 직교 사상 하나 (+ 평균) 를 거치면 95–99 % 로 돌아온다."
- "visible 짝으로 맞춘 사상이 late 로 옮겨지지 않는 것은 encoder 끼리도 같다."
- "풀링 벡터 분산의 77–81 % 는 장면이고, 표현끼리의 차이는 대부분 상수 이동이다. 장면 평균을 뺀 shape 클래스 평균 부분공간은 encoder 셋끼리 0.94–0.96, p 와는 0.78–0.81 을 공유하고, 이 차이는 가림이 없는 조건에서 이미 있다."

## 6. 단서

- 풀링 벡터 · 선형 readout 이다. 토큰 수준 결론이 아니다. 모델은 ViT-H 하나, frozen.
- late 에는 가림막이 장면에 있다 (§8-5). visible↔late 이식은 가려짐과 가림막 존재를 가르지 않는다.
- 조건 안 수치는 쌍둥이 공유 split 의 값이다 (§1). 수준을 인용할 때는 쌍둥이 없음 · 장면 hold-out 값도 같이 적는다. 장면 hold-out 은 성분 350 개 bootstrap 이라 CI 가 넓다 (±5–10 pt).
- 이식 크기 (raw · 교차 조건) 는 probe 에 따라 최대 ~48 pt 다르다. 이식은 "조건 안보다 훨씬 낮다" 까지만 쓴다.
- 복사 기준선은 같은 late 머리 규칙이지만, 풀링 크기 (p 8 튜블릿 vs z_t7 1 튜블릿) 가 다르다. 튜블릿 하나끼리 비교하면 p 는 z_t7 과 같은 급이거나 낮다.
- Procrustes 는 라벨을 안 쓰지만 clip 짝을 쓴다.
- geometry 의 bootstrap 은 cell 평균 (resid) 을 고정한 채 block 만 다시 뽑았다.

## 7. xfer — 스크래치 감사를 레포로 옮김 (2026-09-25 2차, `audit_xfer.json` → `5_xfer`)

스크래치 `v11_probe_geom_audit.json` (레포 밖) 에만 있던 세 가지를 `--part xfer` 로 다시 냈다. 운동 합침 값은 스크래치와 소수 둘째 자리까지 같다 (CI 는 bootstrap 1,000 회라 ±0.1 pt 다르다).

| 항목 | 값 (logistic / ridge, %) |
|---|---|
| (4c) z_all 머리 → z_hid (late k≥2) | shape 59.6 / 23.1 · color 35.5 / 19.0 (같은 clip 의 z_all 99.0 / 99.5 · 98.7 / 98.2) |
| (4c) z_all 머리 → visible 단일 t7 | shape 85.5 / 47.1 · color 56.9 / 24.1 |
| (A3) h_fut 로 직교 Procrustes (전체 train 짝) 뒤 test 상대 잔차, vis / late | p_fut 0.259 / 0.333 · z_all 0.231 / 0.258 · z_t7 0.615 / 0.566 · h_ctx 0.172 / 0.164 |
| (A4) G5 resid visible→late, 운동 합침 (z / p / h_fut / h_ctx) | shape 89.6 / 81.6 / 94.6 / 87.6 · color 91.9 / 80.4 / 93.8 / 93.1 |
| (A4) 운동별 p − h_fut | shape static −15.4 · flat −11.3 · ramp −8.1, color −25.9 · −6.1 · −3.1 (모두 CI < 0) |
| (A4) 운동별 p − z_all | shape −8.8 · −8.3 · −0.1 (n.s.), color −21.2 · +1.1 (n.s.) · **+5.1** [+2.7, +7.8] |

→ "p 가 모든 표현보다 이식이 나쁘다" 는 운동 합침에서만 선다. 모든 운동에서 서는 것은 p < h_fut 다. (4c) 는 풀링 불일치만으로도 visible 에서 85.5 / 56.9 로 떨어져, 가려진 튜블릿 이식 실패의 기준선으로 쓰지 않는다.

## 재현

```bash
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12   # 공용 노드: 안 주면 BLAS 과다구독으로 수십 배 느려진다
$P z_research/scripts/analysis/audit_v11_identity.py --part probe --device cuda:5   # §1 궤적 · 장면 hold-out, §2, §3 → audit_probe.json (~75 분)
$P z_research/scripts/analysis/audit_v11_identity.py --part twin  --device cuda:1   # §1 쌍둥이 유무 → audit_twin.json (~6 분)
$P z_research/scripts/analysis/audit_v11_identity.py --part geom                    # §4 → audit_geom.json (CPU ~3 분)
$P z_research/scripts/analysis/audit_v11_identity.py --part xfer --device cuda:6    # §7 → audit_xfer.json (~7 분), 로그 xfer.log
```

- 입력: `/local_datasets/world/world_analysis/cache/v11_pooled_vith/{z,p,h}.npy, meta.npz` (`v11_pooled_features.py`)
- split: 레포 `WMADataset._split` (block, 0.5, seed 0, condition 층화), hold-out 은 train 안 0.8 seed 1 — `v11_pooled_probe.py` 와 같다
- 로그: 이 폴더의 `probe.log` · `twin.log` · `geom.log`
