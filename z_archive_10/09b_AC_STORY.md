# 09b — AC: 관측으로 배운 predictor 위의 action-conditioned post-training (스토리 초안 · 최종 아님)

> **결론** — AC 는 논문의 **작은 subsection** 이다 (사용자 10-08). 주장은 하나다: "관측만으로 intuitive physics 를 배운 predictor 가 먼저여야 action-conditioned 학습이 효과가 있다 — AC 는 scratch 가 아니라 post-training." **아직 최종이 아니다** (사용자 10-10) — 이 장은 지금 쓸 수 있는 스토리 초안 · 근거 · 실험 설계 · 열린 결정을 모은다.
> **근거** — (1) 지금의 V-JEPA 2-AC 는 frozen encoder 위에 **predictor 를 새로 (scratch)** 학습한다 (`load_predictor: false`, DROID < 62 h, 94,500 step) (`z_AC_training/README.md` §1). (2) 그 predictor 의 평가는 horizon 1 planning 이고 "미래가 물리적으로 그럴듯한가" 를 묻지 않는다 (`z_AC_training/Archive/PLANNING_AND_ROBOCASA_2026-09-20.md` §2 · README §3). (3) 이 주장을 받치는 우리 측정은 **아직 0 이다** — 학습 실행 0, 로컬에 DROID · ViT-g · AC 체크포인트 없음 (README 결론 2).
> **미결** — AC 데이터 (DROID 를 받나, 무엇으로 대신하나) · encoder (ViT-H 로 바꾼 AC 인가) · 앞 단계인 관측 학습 predictor (09 장 목표) 가 아직 없다는 것.
> 이 장이 받치는 논문 문장: 목표 둘째 줄 "(1) 이 선행되어야 action conditioned 학습도 효과가 있다. {scratch 부터 action 학습하는게 아니라, action-conditioned 은 post 가 되어야 한다.}"  ·  원재료: G13 (z_AC_training) · G05 (DIRECTION · DESIGN_CHOICES) · G04a (문헌) 추출문 + 원 문서 직접 확인 → 원 문서는 각 절 끝 `출처`.

---

## 0. 위치와 상태

### 0-1. 사용자 원문과 결정 (그대로)

| 날짜 | 원문 | 뜻 | 출처 |
|---|---|---|---|
| 10-08 (목표 원문) | "Action conditioned 없이, 순수 observation 만으로 intuitive physics를 배우고 ⇒ 이에 따른 plausible future prediction 을 잘하는 predictor를 만들고 싶음 / (1)이 선행되어야 action conditioned 학습도 효과가 있다. {scratch 부터 action 학습하는게 아니라, action-conditioned은 post 가 되어야 한다.}" | AC 는 관측 학습 predictor 의 **다음 단계 (post-training)** | `auto_research/paper/PAPER_STORY_2026-10-08.md` §0 |
| 10-08 (후속 결정) | "이건 이제 자그많게 subsection 느끼믕로 실험 하나 넣을거라 괜찬ㅇ하." | AC = **작은 subsection + 실험 하나** | 정본 §0 · §2-5 |
| 10-10 | "아직 AC가 우리가 도달할 이상은 아니야 그걸 명심해." → (같은 날 풀이) "AC 보류는 뭔 말이냐면 이게 아직 최종이 아니라는거라서 AC 스토리도 일단 작성을 해줘야" | AC 스토리는 **쓴다**. 다만 **최종 판이 아니다** — 주 목표 (관측 학습 predictor) 보다 앞에 두지 않고, 설계 · 결정은 바뀔 수 있다 | 이 아카이브 작업 중 사용자 메시지 (레포 문서에는 아직 없음) |

### 0-2. 논문 안의 자리

```
P1 사람: 관측으로 intuitive physics 를 먼저 갖춘다 → 그것으로 새 환경 · task 에 빨리 적응
P2 / P3 world model: intuitive physics 를 따르는 미래를 예측 못 한다 (predictor 실패)
목표   관측만으로 intuitive physics 를 배우는 predictor            ← 09 장 (주인공)
AC     그 predictor 위에 action 을 얹는다 (post-training)           ← 이 장 (작은 subsection)
```

- AC subsection 은 P1 의 마지막 문장 ("이 능력 덕분에 새로운 환경이나 task 에 빨리 적응") 을 **world model 쪽에서 잇는 문단**이다 — 관측으로 배운 물리가 action 학습 (새 task) 을 빠르게 만드는가.
- 등급: **★ 가설 · 근거 0** (정본 §2 표 AC 행, §2-5). 결과가 나오기 전까지 "AC 는 post 여야 효과가 있다" 는 사실로 쓰지 않는다 (정본 §7).

---

## 1. 스토리 초안 — subsection 문단 (지금 쓸 수 있는 판)

> 아래는 **초안 문장**이다. 괄호 안은 그 문장을 받치는 근거 · 상태. ★ 는 실험 전이라 결과 자리만 비워 둔 것.

**AC-1. 왜 action 인가.** world model 의 쓰임은 결국 행동 — planning 이다. V-JEPA 2 는 같은 encoder 위에 action-conditioned predictor (V-JEPA 2-AC) 를 학습해 로봇 planning 을 보였다. (문헌: V-JEPA 2 §3, Assran et al. 2025 ✅ — `auto_research/Archive/LITERATURE_POSITIONING_2026-09-25.md` 표)

**AC-2. 지금의 AC 는 predictor 를 처음부터 배운다.** V-JEPA 2-AC 는 frozen encoder 위에 **새 predictor** 를 DROID 로봇 영상 (< 62 h) 으로 학습하고, 관측으로 사전학습된 predictor 는 쓰지 않는다 (`load_predictor: false`). 즉 장면 동역학 (물체가 어떻게 움직이고 떨어지고 가려지는가) 과 action 의 효과를 **로봇 데이터 하나로 동시에** 배워야 한다. (코드 확인 — `z_AC_training/README.md` §1)

**AC-3. 우리 주장.** 사람이 관측으로 intuitive physics 를 먼저 갖추고 그 위에서 행동을 배우듯 (P1), world model 도 **관측만으로 intuitive physics 를 따르는 미래를 예측하도록 먼저 배운 predictor** 위에 action 을 얹어야 한다 — AC 는 scratch 가 아니라 post-training 이다. (★ 가설. P1 의 "적응" 인용은 아직 없다 — 01 장 §3)

**AC-4. 실험 하나.** 같은 AC 데이터 · 같은 step 예산에서 (a) AC scratch (V-JEPA 2-AC 레시피) 와 (b) 우리 관측 학습 predictor 위 AC post-training 을 AC 데이터 10 / 30 / 100 % 로 비교한다. (c) 릴리즈 predictor 위 post-training 을 더하면 "intuitive physics 를 배운 predictor 가 먼저" 와 "아무 사전학습이나 먼저" 를 가른다. (설계 — 정본 §2-5 · §3 아래)

**AC-5. 결과.** ★ — "(b) 가 적은 데이터에서 (a) 보다 action 조건부 예측 오차가 낮고, AC 뒤에도 intuitive physics 판 (09 장 합격선 일부) 이 남는다" 면 subsection 문장이 선다. (b) ≈ (a) 면 그대로 보고하고 본 논지 (P1–P3 · 목표) 는 영향받지 않는다. (정본 §2-5 "어느 쪽이어도")

⚠️ 초안에 넣지 않는 문장 (§5): "AC 는 post 여야 효과가 있다" (사실로) · "V-JEPA 2-AC 는 action 을 안 읽는다" · "AC 가 물리를 배웠다".

---

## 2. 이 문단이 기대는 사실 (측정 · 코드 분석 · 문헌)

### 2-1. V-JEPA 2-AC 레시피 (코드에서 확인, 학습 실행 0)

| 항목 | 값 | 출처 |
|---|---|---|
| encoder | ViT-g, context · target 둘 다 `target_encoder` 키, **사실상 frozen** (optimizer 에는 있지만 grad 없음) | `z_AC_training/README.md` §1 |
| predictor | `vit_ac_predictor` depth 24 · dim 1024 · heads 16 (300M), **새로 초기화** (`load_predictor: false`) | 같은 곳 |
| 입력 · 출력 | 프레임마다 `[a_k, s_k, z_k 256 토큰]`, block-causal, 각 프레임 자리 → 다음 프레임 ẑ_{k+1} | 같은 곳 · `PLANNING_AND_ROBOCASA` §1 |
| action · state | 7D (xyz · Euler · gripper), action = 상태 차 | 같은 곳 |
| 손실 | teacher forcing L1 (`jloss`) + 2-step rollout L1 (`sloss`) | README §1 |
| 데이터 · 규모 | DROID < 62 h (4 초 미만 버림), global 256 × 94,500 step ≈ 24.2M clip, 4 노드 × 8 GPU | README §1 · §2 |
| 코드 ≠ 논문 | clip 8 장 (논문 16) · augmentation 사실상 고정 center crop (2000 회 실측 결과 1 종) · rollout 손실에 ẑ_2 가 두 번 | README §4 |
| **action 없이 이미 시험된 것** | 이 레시피의 구조 (frame-causal + TF · rollout 손실) 를 action 없이 실영상에 학습한 것이 **Ariel AR 팔** 이다 — reach 가 네 판 모두 가장 짧고 (팬 3.24 · SSv2 2.18 · EK100 1.75 · v3 4.24), 자기 되먹임 첫 걸음부터 무너진다. 구조만으로는 관측 학습이 해결되지 않는다는 뜻 (08 장 · 09 장 §4) | `auto_research/Archive/AR_PREDICTOR_2026-09-27.md` · `VERIFY_AR_2026-09-27.md` |

### 2-2. AC 의 평가 — 무엇을 재고 무엇을 안 재나

| 사실 | 값 · 내용 | 출처 |
|---|---|---|
| 평가 = closed-loop planning | 현재 사진 1 장 + 센서 자세 → CEM 후보 800 개 × 10 회 → 상상한 **다음 장면 1 개 (horizon 1)** 가 목표 사진과 가장 가까운 action 실행, step 마다 새로 | `PLANNING_AND_ROBOCASA` §2 · 논문 부록 B.2 |
| 한 action 고르는 계산 | predictor forward 10 회 (batch 800) ≈ 16 초 | `PLANNING_AND_ROBOCASA` §2 |
| 긴 과제 | rollout 대신 **하위목표 이미지 3 장** 으로 끊는다 | README §3 |
| AC 평가가 묻는 것 | "어떤 action 이 latent 거리를 줄이는가" — **"미래가 물리적으로 그럴듯한가" 를 묻지 않는다** | README §3 |
| AC 는 우리 가림 판의 측정 대상이 못 된다 (09-20 사용자 지적 · 동의) | 배포 시 autoregressive 아님 · AC 의 세계는 agent 가 일으킨 변화 (준정적 장면) · 우리 장면에 action 0 을 넣으면 "아무 일도 안 일어남" → 실패해도 증거 아님 | README §5-1 |

→ **이 subsection 이 새로 재야 하는 것**: AC 의 planning 성공만이 아니라, **action 조건부 미래가 intuitive physics 를 따르는가** (09 장 합격선 판을 AC 뒤에도 거는 것). 이것이 우리 판정 기준이 기존 AC 평가와 다른 점이다.

### 2-3. 인용 불가로 판정된 것

| 문장 | 왜 못 쓰나 | 출처 |
|---|---|---|
| "V-JEPA 2-AC 는 action 을 거의 안 읽는다" (Jongseo RoboCasa E4 · E9, L1 0.2849–0.2864) | 공식 체크포인트 손실 버그 (jepa-wms `hubconf.py:138`) · proprio 입력 불일치 (world 좌표 state) · zero-shot 도메인 | `PLANNING_AND_ROBOCASA` §3–4 |
| "DROID 는 camera-frame state 로 학습" · "RoboCasa 는 DINO-WM 학습셋" | 둘 다 틀림 (`camera_frame` 기본 False = base 좌표; jepa-wms 도 DROID 만 학습) | `PLANNING_AND_ROBOCASA` §4 |

### 2-4. 문헌 (인용 후보 — 확인 상태 그대로)

| 문헌 | 요지 | 이 문단과의 관계 | 확인 | 출처 |
|---|---|---|---|---|
| V-JEPA 2-AC (2506.09985 §3), Assran et al. 2025 | frozen encoder + 새 AC predictor, TF (T=15) + rollout (T=2); "error accumulation" · long horizon 한계 | **대조 팔 (a) 의 레시피** · must-cite | ✅ | `auto_research/Archive/LITERATURE_POSITIONING_2026-09-25.md` |
| DINO-WM (2411.04983) · DINO-world (2507.19468) · PLDM (2502.14819) | frozen encoder + 새 predictor 가 표준 | 선행 — "frozen encoder + 새 predictor" 자체는 새롭지 않다 | ✅ (표기) | 같은 곳 (G04a) |
| JEPA-WMs (2512.24497) | 계획에서 DINOv2/v3 encoder > V-JEPA encoder | **반례 후보** — 병목이 encoder 쪽일 수 있다 | venue 미확인, 원문 미대조 (WebFetch 요약) | 같은 곳 · `PLANNING_AND_ROBOCASA` 머리 |
| Hierarchical Planning with Latent WMs (2604.03208) | 오차 누적 동기, 성공률 0 → 70 % (초록) | use case 지지 | 초록 기준 | 같은 곳 |
| Garrido et al. 2601.05230 (FAIR) | frozen V-JEPA 2 + 실영상 action-free 새 predictor | 목표 쪽 선점 (09 장 §5) — AC 쪽은 아님 | ✅ | `auto_research/Archive/LITERATURE_PREDICTOR_ROLE_2026-09-29.md` |

⚠️ "관측으로 사전학습한 predictor 를 AC 의 시작점으로 쓰면 낫다" 를 직접 잰 선행은 문헌 문서에 **없다** (있다고도 없다고도 확정 못 함 — WebSearch 한도로 조사가 완전하지 않다, `LITERATURE_POSITIONING` 머리).

---

## 3. 실험 설계 (정본 §2-5 · `z_AC_training/README.md` 머리, 10-08 판)

### 3-1. 팔

| 팔 | 시작 predictor | 학습 | 무엇을 가르나 |
|---|---|---|---|
| **(a) AC scratch** | 새로 초기화 (`load_predictor: false`) | V-JEPA 2-AC 레시피 | 대조 (지금의 표준) |
| **(b) 관측 학습 → AC post** | **우리 관측 학습 predictor** (09 장 목표 — 아직 없음) | 같은 AC 데이터 · 같은 step 예산 | 주장 팔 |
| (c) 릴리즈 → AC post (여유가 있으면) | V-JEPA 2 릴리즈 predictor | 같음 | "intuitive physics 가 먼저" vs "아무 사전학습이나 먼저" |
| 데이터 양 축 | — | AC 데이터 **10 / 30 / 100 %** | post-training 이득은 적은 데이터에서 먼저 보인다 → 어느 결과든 읽힌다 |

### 3-2. 판정

| 지표 | 무엇 | 출처 |
|---|---|---|
| held-out 행동 조건부 rollout 오차 | k 걸음 latent L1 (TF · rollout 둘 다) | 정본 §2-5 |
| intuitive physics 판 유지 | 09 장 합격선 일부 (가림 · 가속 · 낙하 판) 를 AC 뒤에도 걸어 무너지지 않는가 — action 0 판은 "아무 일도 안 일어남" 이라 쓸 수 없으니 action 없는 장면 판에서 | 정본 §2-5 · README §5-1 |
| planning 성공률 | 환경이 있으면 (RoboCasa) — 1-step action sweep 이 "planning 이 성립하나" 를 가장 직접 잰다 | `PLANNING_AND_ROBOCASA` 결론 3 |

### 3-3. 정해야 할 것 (사용자) — 10-08 판에서 열린 채

| 결정 | 선택지 · 사실 | 출처 |
|---|---|---|
| AC 데이터 | DROID raw MP4-only **5.6 TB** (최소 필요분, raw 는 에피소드 약 20 % 빠짐) · 카메라 1 대만이면 약 2 TB (추정, 미검증) · 논문 62 h 부분집합 수백 GB (추정) · 로컬 없음 · `/data2` 여유 거의 없음 → NFS `/data` 뿐 (쓰기 57 MB/s) | `z_AC_training/README.md` §2 |
| encoder | 우리 predictor 는 ViT-H 위 → AC 도 **ViT-H** (릴리즈 AC 는 ViT-g — 그대로 비교 불가) | 정본 §2-5 · README §6 B |
| 규모 | 원 config 32 GPU 기준; vll 8 GPU 면 global batch 64 → lr · accumulation 조정 (코드에 accumulation 없음); ViT-g + 300M 메모리 미측정 | README §5 |
| 옛 선택지 A–D 와의 관계 | A 재현 (ViT-g + DROID) · **B ViT-H AC + DROID** (10-08 설계와 맞음) · C AC 구조만 action 0 (README §5-1 로 해석 불가) · D 릴리즈 AC 분석만 (ViT-g) — 10-08 설계는 B 를 전제한다 | README §6 |
| 순서 의존 | 팔 (b) 는 09 장의 관측 학습 predictor 가 있어야 돈다 → 그 전에는 (a) · (c) 와 데이터 · 하네스만 준비 가능 | 정본 §2-4 · §2-5 |

### 3-4. 돌리기 전 함정 (코드에서 확인)

- `app/vjepa_droid/train.py` import 시 `CUDA_VISIBLE_DEVICES = SLURM_LOCALID` — 단일 task spawn 이면 모든 rank 가 GPU 0. 런처를 따로 둔다 (`z_training/harness/launch.py` 방식) (README §5).
- `load_pretrained` 는 `strict=False` — 키가 안 맞아도 조용히 넘어간다. 팔 (b) · (c) 에서 predictor 가 실제로 로드됐는지 `missing/unexpected` 를 본다 (README §5).
- 데이터 로더가 실패한 영상을 무작위 다른 영상으로 조용히 대체 — 경로가 전부 틀려도 무한 루프 (README §5).
- 체크포인트가 encoder 까지 통째 저장 (ViT-g 면 수십 GB) → predictor 만 저장하도록 (README §5).
- 학습 때는 진짜 상태 s 를, planning 때는 action 을 적분한 상태를 넣는다 — 평가 차이 주의 (README §3).

---

## 4. 결과별 읽기 (하네스 질문 2 — 어느 쪽이어도 문장이 서나)

| 결과 | 쓸 문장 | 못 쓰는 문장 |
|---|---|---|
| (b) < (a) 오차, 특히 10 % 에서 · 물리 판 유지 | "관측으로 intuitive physics 를 배운 predictor 위의 AC post-training 이 scratch 보다 적은 action 데이터로 낫다" | "AC 는 반드시 post 여야 한다" (데이터 · encoder · 한 번의 seed 로는 일반화 못 함) |
| (b) ≈ (c) < (a) | "사전학습 predictor 가 먼저면 낫다 — intuitive physics 학습 고유의 몫은 이 판에서 안 갈린다" | "intuitive physics 덕분" |
| (b) ≈ (a) | 그대로 보고 — "이 AC 데이터 · 판에서는 시작점이 결과를 바꾸지 않았다". 본 논지 (P1–P3 · 목표) 는 영향 없음 | "AC 에는 사전학습이 필요 없다" (AC 데이터가 물리를 요구하지 않을 수 있다 — §5) |
| (b) > (a) (더 나쁨) | 보고 + 원인 후보 (관측 학습 predictor 의 표현 · 시간 척도가 로봇 4 fps 와 다름 등) | — |

---

## 5. 리뷰어가 칠 곳 · 단서 (결론과 같은 무게)

| 공격 | 지금 답 | 남는 단서 |
|---|---|---|
| "아무 사전학습이나 먼저면 낫다 — intuitive physics 와 무관" | 팔 (c) 릴리즈 post 로 가른다 | (c) 를 못 돌리면 이 공격에 답이 없다 |
| "AC 데이터 (DROID) 는 준정적 · agent 가 일으킨 변화뿐 — 물리가 필요 없다" | 판정에 intuitive physics 판 유지를 넣는다 · planning 이 자유 동역학을 거의 요구하지 않는다는 건 README §5-1 이 이미 적었다 | 그렇다면 (b) ≈ (a) 가 나와도 이상하지 않다 → 데이터 선택이 결과를 정한다. 물리가 필요한 AC 과제 (떨어뜨림 · 굴림 · 가림 뒤 집기) 가 있는 데이터인지 먼저 본다 |
| "ViT-H AC 는 릴리즈 AC (ViT-g) 와 다르다" | 팔 (a) 를 같은 ViT-H 로 다시 학습하므로 비교는 공정 | 릴리즈 AC 수치와 직접 비교 불가 |
| "planning 성공을 안 쟀다" | 환경이 있으면 RoboCasa 1-step sweep | 로봇 · 시뮬레이터 없음 — 오차 판정만이면 subsection 으로도 약하다 |
| "encoder 가 병목일 수 있다 (JEPA-WMs: DINOv2 > V-JEPA)" | 07 장 — 우리 판에서는 정보가 encoder 에 있다 | planning 판에서 encoder 병목 가능성은 우리가 안 쟀다 |
| "seed 한 번" | — | 08 장 공통 한계와 같음 |
| "앞 단계 predictor 가 없는데 AC 를 말하나" | 그래서 subsection · 초안이다 | 09 장 목표가 먼저다 |

---

## 6. 금지 표현 (이 장 주제)

| 쓰지 말 것 | 왜 · 대신 | 출처 |
|---|---|---|
| "AC 는 post 여야 효과가 있다" (사실로) | 실험 전까지 가설 → "관측 학습 predictor 위의 AC post-training 이 scratch 보다 낫다는 가설을 실험 하나로 검정한다" | 정본 §7 |
| "V-JEPA 2-AC 는 action 을 안 읽는다" | RoboCasa 실험 인용 불가 (§2-3) | `PLANNING_AND_ROBOCASA` §4 |
| "AC 가 물리를 배웠다 / 모른다" | 내부 귀속 + AC 평가는 물리 그럴듯함을 안 잰다 | 정본 §7 · README §3 |
| "저자들도 pretrained predictor 를 버렸다 = 암묵적 시인" | 09-06 판 beat 7 의 해석 — 10-08 부터 scratch 레시피는 **대조 팔** 일 뿐 | `z_AC_training/README.md` §1 |
| "action-free 학습으로 해결된다" | Ariel 반례 (08 장) | 정본 §7 |

---

## 7. 이 장의 변천

| 날짜 | AC 의 자리 | 출처 |
|---|---|---|
| 09-06 | beat 7 근거 — "AC 는 pretrained predictor 를 버리고 새로 학습했다" (원 문서 표현 — 현재 쓰지 않음) | `z_research/IntPhysGenV11/Archive/PAPER_STORY_2026-09-06.md` |
| 09-19 · 09-20 | `z_AC_training/` 개설 (코드 분석만) · planning 프로토콜 · RoboCasa 검토 · "AC 는 가림 측정 대상 못 됨" | `z_AC_training/README.md` · `PLANNING_AND_ROBOCASA` |
| 09-25 | DESIGN_CHOICES 학습 단계 Q "AC fine-tuning: 가장 좋은 predictor 를 DROID 로 학습하고 planning 검정" · DIRECTION "그다음" | `auto_research/paper/DESIGN_CHOICES_2026-09-25.md` §3 · `DIRECTION_2026-09-25.md` §1 |
| 10-08 | 목표 둘째 줄 + "작은 subsection, 실험 하나" — scratch vs post × 10/30/100 % | 정본 §0 · §2-5 |
| 10-10 | **스토리는 쓴다, 아직 최종 아님** (이 장) | 사용자 메시지 |

---

## 출처

- `auto_research/paper/PAPER_STORY_2026-10-08.md` §0 · §2 · §2-4 · §2-5 · §4-8 · §5 · §7
- `z_AC_training/README.md` (머리 · §1 · §2 · §3 · §4 · §5 · §5-1 · §6)
- `z_AC_training/Archive/PLANNING_AND_ROBOCASA_2026-09-20.md` (결론 · §1 · §2 · §3 · §4)
- `auto_research/paper/DESIGN_CHOICES_2026-09-25.md` §3 (학습 단계 Q) · `auto_research/paper/DIRECTION_2026-09-25.md` §1
- `auto_research/Archive/LITERATURE_POSITIONING_2026-09-25.md` · `auto_research/Archive/LITERATURE_PREDICTOR_ROLE_2026-09-29.md`
- `z_research/IntPhysGenV11/Archive/PAPER_STORY_2026-09-06.md` (beat 7, 대체됨)
- `auto_research/Archive/AR_PREDICTOR_2026-09-27.md` · `auto_research/Archive/VERIFY_AR_2026-09-27.md` (Ariel AR = action 없는 AC 구조)
- 관련 장: 01 (P1 "적응" 인용 빈칸) · 07 (encoder 정보) · 08 (Ariel 반례) · 09 (목표 · 합격선) · 11 (결정 대기)
