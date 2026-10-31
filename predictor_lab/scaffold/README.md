# 자리 계산기 (scaffold) — 학습 때만 쓰는 "어디" 비계 (2026-11-01)

> **질문** — late 가림에서 predictor 는 미래 물체 자리를 문맥 속도로 계산하지 않는다 (VOBench PFT 속도 따라감 0.17). 학습 때만 자리를 계산해 주는 비계를 붙였다 떼면, 그 계산이 predictor 안으로 옮겨질까?
> **지금** — 설계 확정 · 계산기 모듈 구현 · 0 단계 sanity (계산기 혼자 자리를 맞히나) 실행 중. 1–3 단계 (predictor 와 같이 학습 → 떼기 → 제거) 는 아직 안 함.
> **다음** — 0 단계가 통과하면 (속도 따라감 ≈ 1, 미래 자리 오차 < 머무름 기준선) 1 단계 구현안을 먼저 보여 주고 학습한다 (학습은 사용자 결정).

---

## 용어

| 용어 | 뜻 |
|---|---|
| 블록 | 연속 2 장을 encoder 에 넣어 얻은 시점 하나. 창 32 장 = 블록 16 개 (문맥 8 · 미래 8) |
| z_t | 블록 t 의 문맥 encoder 출력, shape (256, D). 토큰 256 개 = 16 × 16 칸 |
| 자리 x_t | 블록 t 의 물체 위치 (열, 행), 토큰 단위. 1 칸 = 16 px (256² 화면) |
| 보이는 정도 m_t | 블록 t 의 물체다움 점수 중 최고값. 보이면 높고 가려지면 낮다 |
| 자리 표시 Δ | "여기에 물체를 그려라" 신호. 미래 블록마다 계산한 자리 중심 Gaussian 지도 × "물체 있음" 벡터 |
| 비계 (scaffold) | 학습 때만 쓰고 떼어 내는 부품. 최종 predictor 는 릴리즈와 같은 구조 (파라미터 추가 없음) |
| 속도 따라감 비 | 같은 장면을 느리게 보여 줬을 때, 예측 끝 자리가 움직인 거리 ÷ 참 끝 자리가 움직인 거리. 1 = 속도를 따라 계산, 0 = 끝 자리 고정 |

## 1. 왜

- [측정] VOBench PFT 는 뒤쪽 미래의 물체 자리를 학습셋 끝 자리에 둔다. 속도를 1/3 로 줄여도 예측 끝 열이 13 → 12 (참 13 → 7), 따라감 0.17 (`z_analysis/transfer_gap/results/r4_x2/speed_sweep.json`).
- [측정] 자리만 맞혀 주면 많이 풀린다. 참 자리에 물체 토큰을 넣으면 사라짐 정답 8 → 63 %. 문맥 운동으로 추정한 자리 (2–3 칸 오차) 에 넣으면 25–30 %.
- [측정] encoder 에는 속도가 있다 (문맥 끝 속도 R² 0.996). 계산할 재료는 있다.
- [이론] 예전 보조 head (상태 head · binding 항 · 점유 head) 는 predictor 와 동시에 학습한 "읽기" 였다. 위치를 담기만 해도 손실이 줄어 "쓰는 길" 이 안 생길 수 있다. 비계는 먼저 "주어진 자리에 그리기" 를 시키고 (쓰는 길을 만들고), 그 입력을 predictor 가 스스로 만들게 옮긴다 (특권 정보 증류 · 교사 강제 → 점진적 끊기와 같은 꼴).

## 2. 자리 계산기 구조 (사용자 설계)

예시: 공이 블록마다 오른쪽으로 1 열, 블록 1–5 보임 (2–6 열), 블록 6–8 가림 (late). 미래 블록 9–12 → 10–13 열.

| 단계 | 무엇 | 학습? |
|---|---|---|
| ① 위치 읽기 | 토큰마다 점수 a_t(r) = w · LN(z_t(r)) + b. softmax 무게중심 = 자리 x_t, 최고 점수 = 보이는 정도 m_t | **w 학습 (0 단계, 라벨)** |
| ② 속도 추정 | m_t > τ 인 블록 (또는 m 무게) 에 x = x₀ + u t (+ ½ a t²) 최소제곱 | 없음 (계산) |
| ③ 굴리기 | 그 식을 미래 블록으로 연장. 가려진 동안에도 움직였다고 본다. 외운 표가 없으니 처음 보는 속도에도 같은 계산 | 없음 |
| ④ 자리 표시 | Δ_t = γ · g_t ⊗ e. g = 계산한 자리 중심 Gaussian (σ 1–2 토큰), e = "물체 있음" 벡터 (d_p = 384). 모양 · 색 정보 없음 (where 만, what 은 predictor 가 과거에서 불러옴). γ 는 0 에서 시작 → 첫 step 은 릴리즈와 같다 | **e · γ 학습 (1 단계)** |

출력: 미래 자리 x_fut (F, 2) — hint 의 정답 · 자리 표시 Δ (F, 256, d_p) — 1–2 단계에서 미래 mask token 에 더함.

깨질 수 있는 지점 (사용자 정리):
1. 반쯤 가려진 블록 — 보이는 쪽으로 무게중심이 쏠린다 (VOBench late 는 가려지기 직전 블록이 거의 늘 반쯤 가려짐). 대응: m 이 충분한 블록만 · m 무게.
2. 배경이 무게중심을 끌어당김 — softmax 가 퍼짐. 대응: 온도 · 상위 k 토큰.
3. 가속 (빗면) — 직선이 안 맞음. 대응: 2 차식 (역시 계산만).

## 3. 학습 단계 (계획)

| 단계 | 켜진 것 | 손실 | 목표 |
|---|---|---|---|
| 0 | 계산기만 (predictor 없음) | w: 보이는 target 토큰 분류 (CE + BCE) | 계산기가 혼자 자리를 맞히나 (sanity) |
| 1 안내 | 계산기 + predictor (릴리즈 초기화) + Δ | 미래 L1 + **Δ 등변성** (Δ 를 s 칸 옮기면 물체 영역 출력도 s 칸) | "계산된 자리에 그리기" 해에 먼저 도달 |
| 2 떼기 | Δ 끄는 확률 0 → 1 (샘플 단위로 섞음) | 미래 L1 + **hint** | Δ 없이도 같은 출력 → 자리 계산이 predictor 안으로 |
| 3 제거 | predictor 만 (릴리즈와 같은 구조, strict load 통과) | — | 평가 |

설계 메모 [이론]:
- hint 의 목표는 좌표가 아니라 **Δ 가 주입되던 그 층 · 그 신호** 로 둔다. 좌표만 읽게 하면 "읽히기만 하는" 해로 빠질 수 있다.
- Δ 끄기는 샘플 단위로 섞는다 (한 배치에 켜진 · 꺼진 샘플이 같이 → 두 경로가 같은 출력으로 묶임).
- 데이터는 외울 시간표가 없는 BlenderTrack v1 (옆으로 움직이는 판 없음) 과 같이 쓴다 — Δ 를 뗀 뒤 shortcut 으로 돌아가지 않게.

## 4. 판정 (학습 전에 정함)

같은 데이터 · 같은 step · 같은 seed 의 팔 셋.

| 팔 | 내용 |
|---|---|
| A | L1 만 (`btrack_v1ns_l1`, 학습 중) |
| H | L1 + hint (Δ 없음) |
| **S** | 비계 0–3 단계 |

- ① 속도 sweep 따라감 비 (VOBench, 학습에 안 쓴 생성기): S 가 H 보다 확실히 높고 0.7 이상
- ② Summary 시나리오 열 (cleverer overtake · IntPhys1 이동 물체 · moving_occluder) 이 A · H 보다 오름
- ③ 다시 나타날 자리에 정체가 몰려 있음 (참 자리 봉우리)

## 5. 0 단계 sanity (실행 중)

- **무엇:** 계산기 혼자. w 는 BlenderTrack v1 train 400 clip 에서 학습, 평가는 v1 val 200 late 창 · VOBench 속도 sweep 24 block (w 학습에 안 쓴 생성기).
- **보는 것:** ① 읽기 오차 (보이는 블록) · m 이 보임 / 가림을 가르는 정도 (AUC) ② 미래 자리 오차 vs 머무름 기준선 (마지막 보인 자리에 멈춤) ③ 속도 따라감 비 (PFT 0.17 과 비교). 변형 4 개 = (보이는 블록만 · m 무게) × (직선 · 2 차식).
- **통과 기준 (제안):** VOBench 따라감 ≥ 0.7 · 미래 자리 오차 < 머무름 기준선.
- 결과: `results/stage0.json` (나오면 이 절에 적는다).

---

## 부록: 재현 · 파일

```bash
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
sbatch -p batch_vll --exclude=vll7 --gres=gpu:1 -c 8 --mem=48G -J scaffold_s0 -o predictor_lab/scaffold/logs/%j_stage0.out \
  --wrap "$P predictor_lab/scaffold/stage0_sanity.py run"     # 뽑기 (z → nfs_cache/scaffold_stage0, 약 3.6 GB) → w 학습 → 평가
$P predictor_lab/scaffold/stage0_sanity.py eval               # 뽑아 둔 z 로 w 학습 · 평가만 다시 (CPU 가능)
```

| 파일 | 무엇 |
|---|---|
| `app/vjepa_frozen/scaffold.py` | 계산기: `ObjectScore` (①) · `readout` · `fit_track` (②) · `rollout` (③) · `PlaceMarker` (④) · `calc` (①–③) |
| `predictor_lab/scaffold/stage0_sanity.py` | 0 단계 sanity |
| `predictor_lab/scaffold/results/` | `stage0.json` · `object_score_w.pt` (w) |
| `z_analysis/transfer_gap/results/r4_x2/speed_sweep.json` | 비교 기준: 기존 모델의 속도 sweep (PFT 0.17) |
