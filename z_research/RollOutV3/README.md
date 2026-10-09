# RollOutV3 — p 는 어떤 미래를 만드나 (위치 + 정체 자)

> **시작점은 이 파일이다** (2026-09-26 정리).
> - 앞 판 README: [`_superseded/README_2026-09-25.md`](_superseded/README_2026-09-25.md)
> - 옛 `new_archive/` 는 `figures/` 로, `new_archive/_audit/` 는 `audit/` 로 합쳤다 (호환 링크는 없다. 다른 세션 스크립트 경로는 새 자리로 옮겼다).
>
> 대상은 frozen V-JEPA 2 ViT-H 릴리즈다. 표현은 셋이다: 문맥 encoder `z`, predictor `p`, target encoder `h`.
> 능력 정의는 1 읽기 · 2 지속 · 3 전이 · 4 영속을 따른다 ([정본](../context_encoder_analysis/Archive/STATE_EVOLUTION_CAPABILITIES_2026-09-19.md)).
> **물리 장면은 측정 도구다.** 가림 (IntPhysGen v11) 은 testbed 다.

## 결론

**p 의 미래 = 문맥 끝 상태를 짧은 거리만큼 굴려 놓은 것이다.**

1. **문맥 끝에 보인 물체.** 복사하지 않고 본 방향으로 옮긴다. 뒤로 갈수록 뒤처진다. 정체는 t8 무렵부터 흐려진다.
2. **문맥 끝에 가려진 물체.** 대부분 미래에 없다.
   - 짧게 (k=1–2) 가려진 이동 물체는 절반 이상이 출구 근처에 나와 1–2 튜블릿 옮겨진다. 그 뒤로는 뒤처지고 흐려진다.
   - 정지 물체는 거의 없다 (남은 '있음' 17–28 % 는 그 물체를 이어 간 것으로 보기 어렵다).
   - **무엇이 남는지는 물체 모양이 가장 크게 가른다.** sphere · torus 는 가려져도 74–93 % 남고 (정지도 절반), 나머지 다섯 모양은 20–41 % (정지 1–8 %) 다. 자 탓이 아니다 (학습셋 · 실제 프레임에서 모든 모양 99 % 이상) — [IDENTITY_R8 §5-4](Archive/IDENTITY_R8_2026-09-26.md).
3. **어느 경우든 마지막으로 본 자리에서 약 3–5 칸 (51–97 px) 을 넘어서는 따라가지 못한다.**
4. **문맥 속도에 따라 미래를 바꾸지만, 가속은 거의 담지 않는다** ([`Archive/CONTEXT_TO_FUTURE_2026-09-26.md`](Archive/CONTEXT_TO_FUTURE_2026-09-26.md), 그림 [`figures/context_to_future/`](figures/context_to_future/)).
   - 빨리 가던 물체를 더 멀리 둔다. 멀어질수록 그 차이를 줄여 담는다 (12 튜블릿 뒤 빠름 − 느림: 진짜 43 px, p 29 px).
   - 문맥 속도가 같으면 가속 · 등속 · 감속 물체의 미래가 거의 겹친다 (진짜 112 / 94 / 76 px, p 70 / 72 / 71 px).
   - 물리와 무관한 **배경**이 물체 자리를 흔든다. 같은 운동인데 배경에 따라 진짜 거리의 61–84 % 로 벌어진다. 실제 프레임을 같은 자로 읽으면 차이가 없다.
   - 모양 · 색 · 시작 위치는 거의 영향이 없다.

네 능력으로 보면 이렇다.

| 능력 | 판정 |
|---|---|
| 읽기 | ✅ |
| 지속 | ◐ |
| 전이 | ◐ — 속도는 문맥대로 바꾼다 (멀수록 줄여서). 가속은 거의 반영 안 함 |
| 영속 | 대부분 ❌ |

## 근거 3 줄

1. **v3 (가림 없음, C16/P32).**
   - t2–t15 에서 '있음' 칸의 82–94 % 가 복사 (문맥 마지막 자리) 보다 진실에 가깝다. 예외는 물체가 벽에서 멈추는 wall 이다.
   - p 조합 정답은 80 → 52 (t8) → 17 % (t15) 로 떨어진다. z 는 77–85 % 로 평평하다.
2. **v11 문맥 끝 가림 (다시 보인 튜블릿 t3–t6).** p '있음' 은 아래와 같다.

   | | k=0 | k=1 | k=2 | k=3 | k=4 |
   |---|---:|---:|---:|---:|---:|
   | 정지 | 100 | 28 | 18 | 16 | 17 |
   | flat | 81 | 62 | 49 | 40 | 37 |
   | ramp | 100 | 60 | 48 | 34 | 25 |

   - 같은 조건의 빈 장면은 p · h 모두 0 % 다.
   - 문맥 중간 가림은 75–100 % 다.
3. **v11 ramp k=2.**
   - t1 에서 '있음' 69 % 이고, 그 '있음' 칸의 55 % 가 조합까지 맞는다 (우연 1.8 %). 진실에서 24 px 떨어져 있다.
   - t3 이후 마지막 본 자리에서 67–70 px (약 3.8 칸) 에 머문다.

## 미결 1 줄

**자 없는 비교와 긴장이 남는다.** vanish block 비교에서는 이동 k=2–4 가 98–100 % 빈 미래 쪽이었다. 그런데 새 자는 k=2 초반 (t1–t2) 에 ramp 58–69 % · flat 34–59 % 가 '있음' 이다. 튜블릿별 자 없는 비교로 확인하기 전까지 "짧은 가림을 이어 붙인다" 는 조건부다.

---

## 측정 도구 — 자 (decoder)

| | 지금 |
|---|---|
| 자 | **`identity_r8`**, 지문 `01c3afaac37f`. 정본 [`exp_results/identity_r8/`](exp_results/identity_r8/) |
| 구조 | learnable query 1 개 + attention pooling → 위치 (x, y) + **57-way** (모양 7 × 색 8 = 56 조합 + 없음) |
| '있음' 규칙 | `log max_c P(c) − log P(없음) > 0` (= argmax ≠ 없음). 튜닝한 문턱이 없다 |
| 학습셋 | `training_r8` = 새 렌더 `RollOut_v2_training_v8`, 8,640 clip. v11 판 arm (물체 뒤) + 같은 장면의 빈 짝 1,920. 감사 3 차 통과 |
| 검증 | 학습셋 test ([`figures/train_readout/`](figures/train_readout/)): 조합 p 84.0 / z 90.5 / h 90.9 % · '있음' 오탐 p 0.27 / z 0.05 / h 0.03 % · 위치 오차 p 13.0 / z 8.9 / h 9.8 px<br>**v11 판 빈 장면 '있음' 0 %** (옛 identity 자 56–99 %) · v11 에서 다시 보인 칸의 h '있음' 97–100 % |

- 자를 바꾼 이유와 검증 전체: **[`Archive/IDENTITY_R8_2026-09-26.md`](Archive/IDENTITY_R8_2026-09-26.md)** (이 README 수치의 정본). 처음 판에서 고친 수치는 그 문서 §8 에 있다.
- 이 README 와 그 문서의 모든 표는 `rollout3_doc_numbers.py` 한 번으로 다시 찍힌다 (`## 재현`).
- 자 구조를 고른 이유 (attention pooling vs heat/mass): [`Archive/READOUT_CHOICE_2026-09-23.md`](Archive/READOUT_CHOICE_2026-09-23.md).
- 옛 자 두 개:
  - `presence` (`dcd24d8a8a47`, 위치 + 있음 logit)
  - `identity` (`aeda55874c1c`, 같은 57-way · 옛 학습셋)
  - 둘 다 v11 가림막을 물체로 읽었다. 그 시기 결과는 [`Archive/RESULTS_2026-09-25.md`](Archive/RESULTS_2026-09-25.md) 에 있다.

## 결과 지도

| 질문 | 그림 | 자 | 정본 |
|---|---|---|---|
| **자 자신: 학습셋 test 에서 무엇을 얼마나 틀리나** (먼저 본다) | [`figures/train_readout/`](figures/train_readout/) | identity_r8 | IDENTITY_R8 §3-1 |
| v3 축마다 앞섬 / 뒤처짐 (p, z · h 자로 본 p) | [`figures/v3_signed_error/`](figures/v3_signed_error/), [`figures/v3_signed_error_hhead/`](figures/v3_signed_error_hhead/) | identity_r8 | IDENTITY_R8 §4 |
| v3 마지막 관측에서 옮긴 양 | [`figures/v3_trajectory/`](figures/v3_trajectory/) | identity_r8 | IDENTITY_R8 §4 |
| v11 p 미래에 물체가 있나 (k 별) | [`figures/v11_presence/`](figures/v11_presence/) | identity_r8 | IDENTITY_R8 §5-1 |
| v11 있을 때 앞섬 / 뒤처짐 | [`figures/v11_signed_error/`](figures/v11_signed_error/) | identity_r8 | IDENTITY_R8 §5-3 |
| v11 있을 때 어디에 (프레임 겹침, k=0–4) | [`figures/v11_readout_overlay/`](figures/v11_readout_overlay/) | identity_r8 | IDENTITY_R8 §5-2 |
| **문맥 (속도 · 가속 · 시작 위치 · 외형) 이 바뀌면 p 의 미래가 바뀌나 · p 의 미래 물체가 같은 정체인가** (v3 16 창 · v11) | [`figures/context_to_future/`](figures/context_to_future/) | identity_r8 | [CONTEXT_TO_FUTURE](Archive/CONTEXT_TO_FUTURE_2026-09-26.md) |
| v11 정체는 p 에서 선형으로 읽히나 | [`figures/v11_probe/`](figures/v11_probe/) | **자 없음** (평균 풀링 선형 probe) | RESULTS_2026-09-25 §6 |
| v11 풀링 기하 (p 의 정체 배치는 encoder 와 다른가) | [`figures/v11_geometry/`](figures/v11_geometry/) | **자 없음** | RESULTS_2026-09-25 §7 |
| v3 영상 위 진실 · p · h (삽화) | [`figures/v3_gif/`](figures/v3_gif/) | presence (옛 자, 주장 없음) | RESULTS_2026-09-25 §4 |

### 자와 무관해서 여전히 유효한 결과 (RESULTS_2026-09-25 §6–7)

- **정체는 p 에서 읽힌다. 수준은 문맥 복사다.**
  - 문맥 끝 가림 clip 에서 p 미래 풀링의 shape · color 는 95–97 % 로 읽힌다.
  - 문맥 전체 복사 (z_all 98–100 %) 보다 낮다. 문맥에 p 를 더해도 늘지 않는다.
  - → "p 는 문맥이 가진 정체를 옮겨 싣는다" 까지만 말한다. 영속의 증거가 아니다.
- **p 는 encoder 와 다르게 놓는다. 그 차이는 가림 없이 이미 있다.**
  - 장면 평균을 뺀 shape 클래스 평균 부분공간을 encoder 끼리는 0.94–0.96, p 와는 0.78–0.81 공유한다.
  - 가림이 p 의 배치를 encoder 보다 더 옮기지 않는다 (visible ↔ late 겹침은 네 표현 모두 0.94–0.97).
  - 토큰 LN 대조가 없다.
- **clip 짝 직교 사상 하나로 encoder 머리가 p 에서 95–99 % 로 돌아온다.** raw 이식은 probe 에 따라 15–48 % 다.

## 단서 (결론과 같은 비중)

1. **자 없는 비교와의 긴장** (위 미결). "짧은 가림을 이어 붙인다" 는 조건부다.
2. **자는 보이는 물체만 읽는다.** 정지 물체의 '없음' 은 "판을 세운 채 두어 가렸다" 일 수도 있다. 가르는 검사 (p 가 실제 미래 · 판이 선 마지막 문맥 · 빈 장면 중 어디에 가까운가) 는 돌리지 않았다.
3. **학습 판은 물체 뒤, v11 판은 물체 앞이다.** 반쯤 가려진 칸 · 출구 칸은 자의 정의 밖이다.
4. **후반 튜블릿 y 의 위쪽 끌림**에는 자가 물체를 놓칠 때 내는 기본값이 섞여 있다. 위치 주장은 '있음' 칸에서만 한다.
5. **정체 정확도가 옛 자보다 낮다.** 조합당 학습 양성이 약 1/3 이다. v11 가림 없음 정지에서 p 조합 56–81 % (옛 92–97 %) 다.
6. **표본과 방향.** v11 이동 팔은 궤적이 둘 (l2r · r2l) 이고, 거리 크기가 방향에 따라 두 배까지 다르다 (ramp k=2 t1 마지막 본 자리까지 37 vs 77 px). v3 의 유효 표본은 궤적이다 (법칙마다 14 개). 그림은 clip 평균이다.
   - **마지막 튜블릿 t7 은 표에서 뺐다.** 이동 물체는 화면을 나가는 중이다 (h '있음' 11–31 %). 그런데 p '있음' 은 flat 78–95 %, ramp 40–100 % 다. 정지 물체도 t7 에서 '있음' 이 41–58 % 로 오른다. 학습셋에서도 p 오탐이 t7 에서 가장 높아, 예측 창 마지막 자리의 성질일 수 있다.
   - **p 는 학습셋 안에서도 뒤 튜블릿에서 나빠진다** (위치 t3 11 → t7 18 px, 조합 89 → 74 %). v3 · v11 의 뒤 튜블릿 저하에는 자의 성질이 섞여 있다.
7. **late 에는 가림막이 장면에 있다** (CLAUDE.md §8-5). 모델은 하나 (ViT-H, frozen) 이고, 합성 데이터 두 세트다.

## 폴더

```
README.md                 ← 이 파일 (시작점)
Archive/
  IDENTITY_R8_2026-09-26.md     지금 자 (identity_r8) 의 근거 · v3 · v11 결과 — 정본
  CONTEXT_TO_FUTURE_2026-09-26.md  문맥 운동 · 외형이 p 의 미래를 바꾸나 + 정체 유지 (v3 16 창 · v11)
  RESULTS_2026-09-25.md         옛 자 시기 전체 분석. §6 probe · §7 기하 · 자 없는 비교는 여전히 정본
  READOUT_CHOICE_2026-09-23.md  자 구조 선택 (attention pooling)
  _superseded/                  대체된 문서 (옛 new_archive README 두 판 · 요약 · 절벽 문서)
figures/                  그림. 폴더마다 _decoder.json = 어느 자로 그렸나. 옛 자 그림은 각 폴더 _superseded/<자>_<지문>/
  README.md                     폴더 · 스크립트 · 도장 표
  _superseded/presence_era_2026-09-24/   presence 자 시기의 옛 파이프라인 그림 (windows · examples · readout · direction · v11)
audit/                    감사 (학습셋 · 판 오탐 · v3 운동 · v11 probe/기하) — README.md
exp_results/              자 가중치 · 읽은 값 (readings.npz) — README.md (어느 폴더가 어느 자인지)
_superseded/              옛 README · RERUN · monitor.sh
```

## 재현

```bash
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python

# 0) 문서의 모든 표를 산출물에서 다시 찍기 (CPU 몇 초)
$P z_research/scripts/analysis/rollout3_doc_numbers.py --decoder identity_r8

# 1) 학습셋 감사 (CPU) → audit/training_sets/
$P z_research/scripts/analysis/audit_training_set.py --root /data2/local_datasets/world/world_analysis/RollOut_v2_training_v8

# 2) index → 특징 (/dev/shm) → 자 학습 → v3 16 창 · v11 다시 읽기 → 그림 → 판 오탐 검사 (GPU 8 장, 약 45 분)
python z_research/scripts/data/build_rollout2_index.py --set training_r8 --write
SET=training_r8 NAME=identity_r8 FRAMES=/data2/local_datasets/world/world_analysis/RollOut_v2_training_v8 \
    bash z_research/scripts/analysis/decoder_train_redraw.sh

# 3) 그림만 (CPU 몇 분). readings 의 자 지문이 자와 다르면 멈춘다
$P z_research/scripts/figures/new_archive_redraw.py --decoder identity_r8 --steps figs   # 8 폴더 (figures/README.md 표)

# 4) 자 없는 결과 (probe · 기하) — Archive/RESULTS_2026-09-25.md §재현
```

- 새 자로 바꾸는 법: 2) 를 `NAME=<새 이름>` 으로 돌린다. 그러면 `figures/*` 의 옛 그림이 도장을 보고 `_superseded/` 로 내려간다.
- 경로와 자 지문의 정본은 `z_research/scripts/analysis/rollout3_paths.py` 다.
  - 기본 자는 `presence` 로 남겨 두었다. TrainingEffects · auto_research 가 그 기본값을 쓴다.
  - 이 폴더의 그림은 `R3_DECODER=identity_r8` 로 그린다.
