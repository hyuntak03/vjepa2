# 11 — 미결 · 문서 간 모순 · 리뷰어 공격 · 다음 할 것

> **결론** — 10-08 정본의 P2 · P3 문장은 서 있다. 다만 그 근거 셋 (지평 "2–3 칸 = 거리" · "마지막 관측에 안 보이는 것은 못 한다" · 3a 의 "encoder 에 정보가 있다") 에는 리뷰가 남긴 단서가 있고, 정본 §4 는 아직 그 단서에 답하지 않았다. 사용자 결정 대기는 주 평가 하나와 운영 하나 (auto_research 커밋) 다. AC 는 스토리 초안 · 아직 최종 아님 (사용자 2026-10-10) — 데이터 · encoder 결정이 열려 있다 (`09b_AC_STORY.md`).
> **근거** — (1) 거리 회귀 (거리 100 px 당 −0.47 · 슬롯 −0.28) 는 귀무를 보정하지 않았고 clip 단위다. 그래서 리뷰는 "재적합 전에는 논문 문장으로 쓰지 말라" 고 했다 (`auto_research/Archive/REVIEW_CVPR_ORAL_2026-09-25.md` 통계 §3-8).
> (2) IntPhys 1 재등장 튜블릿에서 p − 복사 +16.4 [+12.5, +20.5] 가 나왔고, v11 가려진 미래 프레임은 pos/imp 가 픽셀 단위로 같은 쌍이 0/788 이다 (같은 문서 S8 · R2 §6-4).
> (3) encoder 의 가속 판독은 C8 ≈ 0 이고, C32 0.76 은 역사 기준선 0.90–0.93 보다 낮다 (같은 문서 X9 · 통계 §3-2).
> **미결** — 이 단서를 닫는 실험 (재적합 · `copy_h_ctxonly` · 렌더 누수를 뺀 채점 · 확증 재현) 은 전부 ★ 미실행이다. 실행했다는 기록이 추출문 어디에도 없다.
> 이 장이 받치는 논문 문장: 전 문단 (P2 · P3 · 목표) 의 단서 절과 정본 §4 "리뷰어가 칠 곳"  ·  원재료: G04b (리뷰 문서 셋) + 모든 G 추출문의 (c) 묶음 요약 → 원 문서는 각 절 끝 `출처`.

**판정 상태 어휘** (이 장 전체에 쓴다)
- **미결** — 어느 쪽이 맞는지 문서에 없다.
- **부분** — 답은 있으나 단서가 남는다.
- **병기** — 조건이 달라 모순은 아니다. 섞어 읽지 않는다.
- **미규명** — 원인을 모른다고 문서가 적었다.
- **소멸** — 그 공격이 겨눈 문장이 10-08 스토리에서 빠졌다.

등급 표기는 원 문서를 따른다: 확정 · 1 차 · 시사 · 가설 · ★ 미실행.

---

## 1. 결정 대기 (사용자)

### 1-1. 열린 결정

| # | 결정 | 지금 문서의 기본값 · 상태 | 이 결정에 걸리는 것 | 출처 |
|---|---|---|---|---|
| D1 | **주 평가를 intuitive physics 판으로 옮기나** (EK100 · Ego4D 를 어디에 둘지) | 기본값: IntPhys 1/2 · GRASP · InfLevel + testbed copy-hard · 위치 쌍을 1 차로 둔다. EK100 등 world model downstream 은 "떨어지지 않는다 / 오른다" 를 보이는 2 차로 둔다. 정본 문서는 기본값을 전제로 썼다 | 합격선 표의 EK100 reach 칸 (정본 §2-4). 기본값 쪽 근거: EK100 은 predictor 출력을 복사로 바꿔도 Δaction 이 약 2–3 점이다 (1 s +2.54 / +2.61 · 3 s +2.06, SC3 · SC5). REVIEW_v2 R3-1 은 "시간 스윕에서 치환 하락이 **커지지 않으면** EK100 은 주 평가에서 빼야 한다" 고 조건을 걸었고, SC5 에서 커지지 않았다 (1 s +2.54 → 3 s +2.06) | `auto_research/paper/PAPER_STORY_2026-10-08.md` §3 · CLAUDE.md §0 |
| D2 | **auto_research 커밋** | git 추적 밖이고 커밋 결정을 기다린다. 정본 스토리 · 결과 로그 · 철회표 · 분석 문서 · 스크립트가 모두 이 폴더에 있다 | 08-28 사고 (`IntPhys-Like_data/` 의 리포트 · figure 가 추적 0 이라 복구 못 함) 와 같은 위험 | CLAUDE.md §8-1 |
| D3 | 커밋 안 된 3 파일 (`analysis/intphys2/run_grid.sh` · `z_research/Benchmarks/run_all.sh` · `build_intphys1_pairs.py`) | 09-22 기준 기록이다. 지금 상태는 이 장에서 확인하지 않았다 | ② 외부 벤치마크 진입점 | `z_research/HANDOFF_2026-09-22.md` §2 |
| D4 | **P2 일반성용 모델 선택** — latent 예측형 하나 더 (V-JEPA 2.1 · DINO-world 류) 또는 생성형 하나 | 모델은 사용자가 고른다. 실행은 Claude | 정본 §4-2 "픽셀 생성형은 안 쟀다". 리뷰 전원이 "두 번째 모델 계열 없이는 oral 불가" 라 했다 (REVIEW_v1 AC/JEPA · CVPR_ORAL O-13) | 정본 §5-6 |
| D5 | **관측 학습 predictor 의 설계 후보 선택 · 학습 실행** | 학습은 사용자, 판정 기준 · 설계 근거는 Claude 가 맡는다 | 합격선 표 · "Ariel 이 왜 실패했나 → 설계 후보" 표 (정본 §2-4). **3b 사건을 겨누는 후보는 빈칸이다** | 정본 §2-4 · `auto_research/paper/DESIGN_CHOICES_2026-09-25.md` §3-b–d |
| D6 | α 분해 인용 해제 | 사용자 지시가 있을 때까지 인용 금지다. 값은 이 아카이브에 옮기지 않는다 | 옛 문서 본문 (`z_world_model_analysis/PREDICTOR_HEDGING_*` · `MECHANISM_*` · `z_research/IntPhysGenV10/Archive/GEOMETRY_2026-08-29.md`) | CLAUDE.md §0 · §5-4 |
| D7 | **AC subsection 실험** | **스토리 초안 · 아직 최종 아님 (사용자 2026-10-10).** 정본 §2-5 가 "정할 것" 으로 둔 AC 데이터 (DROID 는 로컬에 없다) · encoder (ViT-H vs 릴리즈 ViT-g) 와 `z_AC_training` §6 의 선택지 A/B/C/D (10-08 설계는 B 를 전제) 가 열려 있다. 팔 (b) 는 09 장 predictor 가 있어야 돈다 | `09b_AC_STORY.md` §3-3 | OUTLINE §1 · 정본 §2-5 · `z_AC_training/README.md` §6 |

⚠️ D1 의 기본값은 문서 안에서 이미 기정사실처럼 쓰인다 (합격선 표, 하네스 질문). 사용자가 반대로 정하면 정본 §2-4 표의 "보이는 것" 행 (EK100 reach) 지위부터 바뀐다.

### 1-2. 결정은 났는데 실행 · 반영 기록이 문서에 없는 것

| 결정 | 언제 · 누가 | 지금 상태 (추출문 기준) | 걸리는 미결 | 출처 |
|---|---|---|---|---|
| (v0 × a) 교차 세트를 만든다 ("가속은 지금 데이터로 못 잰다" 에 대한 답) | 09-16 무렵 사용자 | 렌더 · 측정 기록 없음 | §2-2 e (3a 에서 encoder 정보) | `z_research/context_encoder_analysis/Archive/EXPERIMENTS_2026-09-16.md` §2-1 · `exp_results/ctxenc_boundary_kinematics/RESULTS.md` |
| ledge · wall · 꼭대기를 논문에 넣는다 → 다중 미래 세트 | 10-08 사용자 | 설계만 있고 렌더 전 (정본 §5-4) | 정본 §4-4 · §2-2 c (갈림 시점 누수) | 정본 §2-3 (iii) |
| IntPhys 2 config 수정 (`dtype: float32` + `autocast: bfloat16`, 열별 선택) | 09-14 대조 | **미반영** | IntPhys 2 54.35 의 정밀도 관례 | CLAUDE.md §2-7 · `analysis/intphys2/PROTOCOL_CHECK_2026-09-14.md` |
| 가림 음성을 자 학습셋에 넣지 않는다 (가림 판정은 자 없는 vanish 짝 비교로) | 09-25 사용자 | 확정 | §3-1 "r8 '있음' vs 자 없는 비교" 긴장의 원천 | `z_research/RollOutV3/_superseded/README_2026-09-25.md` §4-2 |
| `exp_results/windows/BEHAVIOR.md` 삭제 여부 | 09-24 "결정 대기" | 대체됨 표시만 있다 (문서 관리 문제) | — | `z_research/RollOutV3/exp_results/windows/BEHAVIOR.md` |

### 출처 (§1)
`auto_research/paper/PAPER_STORY_2026-10-08.md` §2-4 · §2-5 · §3 · §5 · CLAUDE.md §0 · §2-7 · §8-1 · `z_research/HANDOFF_2026-09-22.md` · `z_AC_training/README.md` · `auto_research/paper/DESIGN_CHOICES_2026-09-25.md` · `auto_research/paper/REVIEW_v2_2026-09-26.md` · `z_research/context_encoder_analysis/Archive/EXPERIMENTS_2026-09-16.md`

---

## 2. 리뷰어가 칠 곳

리뷰 문서는 셋이다. 셋 다 실행 없이 기존 배열을 재계산했고, 그 재계산 스크립트는 레포 밖 옛 세션 스크래치패드에 있다 (§4-2).
- `auto_research/Archive/REVIEW_CVPR_ORAL_2026-09-25.md` — 분석 섹션 H1–H9 를 6 인이 심사했다. 판정은 "oral 불가, must-do 뒤 강한 poster" 이고, 대체 표시는 없다.
- `auto_research/paper/REVIEW_v1_2026-09-25.md` — DRAFT v1 을 5 인이 심사했다. 판정은 3/6 weak reject 이고, 정본 §4 로 대체됐다.
- `auto_research/paper/REVIEW_v2_2026-09-26.md` — DIRECTION · ABSTRACT v0 · DESIGN_CHOICES 를 심사했다. 판정은 [A] weak reject 경계 / [D] poster 이고, 정본 §4 로 대체됐다.

세 리뷰 모두 **벤치마크 감사 (복사 기준선 · 양방향 target 번짐 · CCA) 를 유일하게 문헌상 새로운 것** 으로 봤다. 정본 P2 가 leakage 를 축으로 올린 것과 같은 방향이다.

### 2-1. 정본 §4 의 9 항목 — 지금 답과 남은 단서

| # | 공격 | 지금 답 (정본) | 남은 단서 (리뷰 · 다른 문서) | 필요한 실험 | 출처 |
|---|---|---|---|---|---|
| 4-1 | "Garrido 는 창발했다고 했다 (IntPhys 1 88.89)" | **IntPhys 1 은 leakage 벤치다.** 복사 85.0, Δ +3.9 이고 CI 가 0 을 포함한다. DINO-F 는 복사 88.9–92.2 가 predictor 84.4 보다 높다. IntPhys 2 에서는 복사 수준이다. 확정 | 칸 선택 공격이 남는다. 공식 config 칸 `skip2_w16` 에서는 p − 복사 +12.2 [+6.7, +17.8] 로 유의하다 ("반드시 병기", K8). "n.s." 는 "같다" 가 아니다 — 단측 95 % 상한 +7.8 pt, MDE ≈ 6.5 pt. 88.89 자체도 표본 변동이 ±5 pt 다. 복사는 online encoder 공간에 있고 표적은 EMA 공간에 있다 (§2-2 g) | 전 격자 · 상한 보고, `copy_h_ctxonly`, GRASP · InfLevel 복사 기준선 (정본 §5-2) | 정본 §4-1 · CVPR_ORAL K8 · 통계 R5 · O-7 · R2-12 |
| 4-2 | "V-JEPA 2 하나 아닌가" | 잰 것: ViT-H/L 릴리즈, Ariel 세 팔, post-FT 넷, 사전학습 8 개, DINO-Foresight (v11 가림에서 같은 실패) | **픽셀 생성형은 안 쟀다.** 리뷰 전원이 "두 번째 모델 계열 없이는 oral 불가" 라 했다 | P2 일반성 (정본 §5-6, 모델 선택은 D4) | 정본 §4-2 · REVIEW_v1 AC-oral · CVPR_ORAL O-13 |
| 4-3 | "문맥 끝 가림은 corner case" | 실영상에서 움직이는 내용의 약 절반이 reach 밖이다 (SSv2 46 · EK100 51 %, 1 차). 실영상 가림 띠에서 복원은 절반이다 (.21 vs .42, 확정). IntPhys 1 의 이동 물체 가림은 0.2 초다 | 합의 토큰 필터가 쉬운 질감을 골라 낙관적일 수 있다 (필터 없는 곡선의 거리 계수 −0.055, R4-3). 1 인칭 판독이 타당한지 (손 · 물체 bbox 층화, R3-2) 는 답이 없다. 46 / 51 % 는 10-08 재확인 대상 밖이다 (§4-1) | `OcclusionStats` 를 IntPhys 2 · GRASP · InfLevel 로 (정본 §5-7), 필터 없는 곡선 병기 | 정본 §4-3 · REVIEW_v2 R3-2 · R4-3 |
| 4-4 | "ledge 0 % 는 L1 이 덜 움직이는 미래를 싸게 보는 것 · 복사도 같은 쪽" | 복사가 같은 쪽인 것은 P3 와 같은 방향이다. 위치 자는 선반 높이를 유지한다. gravity 에서 '멈춤' 을 고른 block 은 1/144 다 | VoE 단독으로는 못 가른다. 갈림 시점 누수가 있다 (ledge 쌍은 프레임 35 에서 갈리고, 슬롯 0 은 프레임 32–33). 자 둘이 다른 답을 낸다. RollOut_v2 "392 쌍" 은 시나리오당 **14 궤적 × 28 쌍** 꼴이라 유효 n 이 작다 (S-1) | 다중 미래 세트 (정본 §5-4), 갈림을 예측 구간 안쪽으로 | 정본 §2-3 (iii) · `auto_research/Archive/H23_PREDICTOR_LAYERS_TRAJECTORY_2026-09-24.md` §3-7 · CVPR_ORAL S-1 |
| 4-5 | "encoder 충분성은 지도 자의 산물" | z 자 ≈ h 자, 파라미터 없는 교차검증, 안 본 속도 · 가림 문맥에서도 맞는다 | 픽셀 차분 · 무작위 초기화 ViT-H 대조 (Hewitt–Liang 식) 가 없다 (JEPA-E6 · STAT-exp7). 실영상에서 "정보가 있다" 는 답이 없다 (R5-3: 실영상 reach 밖 cos_prev .51 vs 복사 .55). 3a (가속) 는 §2-2 e | 같은 ridge · 궤적 split 로 픽셀 / 무작위 encoder 대조 | REVIEW_v1 JEPA-E6 · STAT-exp7 · REVIEW_v2 R5-3 |
| 4-6 | "inpainter 가 예측 못 하는 건 당연" | 벤치마크가 그것을 world model 로 인증했고, 미래 전용 · 되먹임 목표로 다시 학습해도 같다 | R2-9 "가장 큰 구멍": 릴리즈 predictor 는 `temporal_scale [1.0, 1.0]` 공간 tube 마스크로만 학습했다. 같은 분포 안에서 하는 대조 (물체 경로 띠를 전 프레임에서 가리는 tube 마스크) 결과가 없다. "알려진 결과" 위험도 있다 (V-JEPA 2-AC `load_predictor: false` · DINO-world · HERA) | 분포 안 tube 마스크 대조 (GPU 30–60 min) | CVPR_ORAL R2-9 · L-1 · REVIEW_v1 AC-W1 |
| 4-7 | "관측만으로 학습하는 건 Ariel 이 이미 했고 실패했다" | **반드시 넘는다** (사용자). 합격선 표 + 설계 후보 표를 쓰고, 방법 절 첫 줄이 Ariel 과의 차이다 | Ariel 학습 부족 가능성 (§2-2 f), HERA 선점 (§2-2 l). 설계 후보가 이미 알려진 관행이다 (PlaNet overshooting · 2-AC rollout · Self/Diffusion Forcing, APP-W7). 정본 표에서 3b 사건 후보는 빈칸이다 | 합격선 ★ 채우기 (정본 §5-3) · 새 predictor (§5-8, 사용자) | 정본 §2-4 · §4-7 · REVIEW_v1 APP-W7 |
| 4-8 | "AC 가 post 여야 한다는 근거?" | 정본은 subsection 실험 하나로 답한다 (scratch vs post × 10/30/100 % + 릴리즈 post 팔) — 스토리 초안 · 최종 아님 (10-10) | 근거 0 그대로다. "AC 는 post 여야 효과" 는 가설로만 쓴다. AC 데이터가 물리를 요구하지 않으면 (b) ≈ (a) 가 나와도 이상하지 않다 | AC 실험 (`09b` §3) | 정본 §2-5 · §4-8 |
| 4-9 | "intuitive physics 의 정의?" | 행동 정의 (네 능력 → 보이는 것 / 안 보이는 것 / 없는 것 3a · 3b / 읽기) | 인지과학 계보 인용이 "부분" 이다 (CVPR_ORAL 문헌 판정 N 행). P1 의 "동물" · "새 환경 적응" 인용이 빈칸이다 (§2-3) | 문헌 확인 (정본 §5-10) | 정본 §2-1 · `auto_research/Archive/LITERATURE_POSITIONING_2026-09-25.md` |

### 2-2. 정본 §4 에 없는 치명적 약점 (G04b 묶음 요약 (0) 과 추가분)

| 기호 | 공격 | 지금 문서의 답 | 남은 단서 (수치 · 원 문서 등급) | 필요한 실험 (원 문서의 비용) | 출처 |
|---|---|---|---|---|---|
| a | **지평 "2–3 칸" 은 거리 단위인가** | 정본 P3 와 CLAUDE.md §5-5: "물체다운 토큰은 마지막 관측에서 2–3 칸 (40–60 px) 안", "거리 100 px 당 −0.47, 슬롯 −0.28" | 이 회귀는 귀무 미보정이고 clip 단위다 → "재적합 전 논문 문장 금지" (통계 §3-8). v3 에서는 정지점이 속도에 비례한다: 빠름/느림 비 os 1.96 [1.75, 2.20] · ar_p 2.07 vs 속도비 1.85, 즉 고정 거리 상한은 기각된다. 1-파라미터 R² 는 시간 상한 0.71 (K = 4.15) 이 거리 상한 0.55 보다 높다. log 지수로는 위치 적중 1.06 [0.66, 1.58] (거리형) · 물체다움 0.38 [0.28, 0.46] (시간 우세) 다. R2 는 "치명에 가깝고, 고칠 수 있다" 고 했다 | v2 회귀에 귀무 + 궤적 CI + log 지수 재적합 (P10/R7). training_v5 `line_x` 넓은 속도 (4–8 배) 로 "걸음 수 vs 거리" 판결 (GPU ~30 min + CPU 1 h) | CVPR_ORAL O-4 · R2-2 · S-2 · S-3 · O-14 · REVIEW_v2 R4-1 · REVIEW_v1 STAT-W4 |
| b | **"안 보이는 것은 못 한다" 의 반대 증거** | 정본 P3 (ii) 의 근거는 v11 · v11_realistic (vanish 물체→빈 0 % / 0–10 %) | IntPhys 1 재등장 튜블릿에서 p − 복사 +16.4 [+12.5, +20.5] (인과 표적 +17.3) 다. 문맥 끝 가림 d6+ 는 +19.3 [+15.0, +23.6] (27 scene), O2 모양은 +24.6 [+19.4, +29.6] 이다. v11 재등장은 +2.5 [−0.9, +5.7] 다. 이 비교에서 복사는 가림막 프레임이라 기준선이 약하다. devil 리뷰는 "S 승 — 영속 실패는 v11 설계의 특이 현상일 수 있다" 고 판정했다 | 마지막으로 보인 튜블릿 복사 + IntPhys 1 대 v11 조건표 (P9) | CVPR_ORAL S8 · O-10 · DA-2 · A-1 |
| c | **v11 가려진 미래 프레임의 렌더 누수** | CLAUDE.md §1-2 · §7-4 의 byte-identical 감사는 **문맥 16 프레임만** 본다 | 가려진 미래 프레임에서 pos/imp 가 픽셀 단위로 같은 쌍은 0/788 이다 (픽셀 36.4 % 가 다르고, 최대 차 중앙값 3). 재등장 전 인과 p 는 44.9 로 chance 아래다 → late vanish · H9 +29.8 · late 표에 렌더 누수가 섞일 수 있다 | 가려진 튜블릿 · 가림막 footprint 토큰을 뺀 채점 (must-do 10, GPU ~30 min) | CVPR_ORAL ★R2-8 |
| d | **L1 이중 벌점** (가설) | 정본은 vanish '물체→빈' 0 % 를 "안 보이는 것" 으로 읽는다 | 예측된 물체가 뒤처지면 h_pos 와는 두 자리에서, h_imp (빈 장면) 와는 한 자리에서 어긋난다 → 점수가 "없음" 과 "뒤처짐" 을 못 가른다. 가림 없는 먼 슬롯 (t4–7) vanish '물체→빈' 은 정지 99.5 / 등속 66.1 / **등가속 2.7** 이다 (v11_full, 칸당 224 쌍; B 셋 다 100). early 등속 17.4 는 미해결이다 | R3: v11_full 파라미터 없는 readout + clip 별 지연 vs vanish-A 판정. RollOutV2 "late 0 슬롯" · identity_r8 가 부분 답일 수 있지만, 이 가설과 이어 놓은 문서는 없다 | CVPR_ORAL ★A-2 (architect C7) |
| e | **3a (가속) 에서도 "encoder 에 정보가 있다" 인가** | 정본 (iv) 의 근거는 속도 R² 0.996 / 0.989 (확정) 와 자 B 5–10 px (ramp 가속 포함, 1 차) | encoder 가속 판독은 C8 ≈ 0 이다. C32 r8 0.76 은 재적합 분산이 없고 (7 궤적 split 0.12 ± 0.32), 역사 기준선 0.90–0.93 보다 낮다. C16 은 0.51 [0.25, 0.65] 다. ctxenc 는 "가속은 지금 데이터로 못 잰다" 고 했다 — flat_a 는 a ∝ v, ramp_a 는 경사각 단서 (R² 0.991), 교차 쌍은 지름길 (AUC 0.20). 자 B 는 장면 종류를 읽고 지도 곡선을 붙인다 (정본 iii-3). Joseph 2026 은 가속 판독이 가능하다고 보고했으나 대조하지 않았다 | (v0 × a) 교차 세트 (§1-2), 동등성 상한 (P12), stride 3 재검정 | CVPR_ORAL X9 · O-11 · REVIEW_v1 APP-W2 · JEPA-W7 · `z_research/context_encoder_analysis/exp_results/ctxenc_boundary_kinematics/RESULTS.md` |
| f | **Ariel 의 실패는 학습 부족인가** | 정본 §4-7 은 "반드시 넘는다" 로만 답한다 | REVIEW_v2 가 "가장 약한 고리" 로 짚었다. Ariel reach 는 같은 run 안에서 epoch 따라 단조 증가한다 (s1_C16 plateau 3.52 → 5.08 칸, 수렴 안 함). run 안 IntPhys 1 은 ep5 → ep43 에서 +21.7 (w32) · +21.1 (w16) 오른다. I_SCENE 의 "더 학습해도 안 는다" 는 팬에서 미결이다 (cosine lr 끝과 교락). 확정된 것은 EK100 R50 불변뿐이다. **ep5–43 체크포인트는 09-27 에 지워졌다** → 그 실험은 그 체크포인트로 더 못 돌린다 | 새 학습 run 의 학습 곡선 체크포인트로 reach 포화 적합 · run 안 해리 (학습은 사용자) | REVIEW_v2 ★R5-2 · REVIEW_v1 N1 · APP-E2 · `auto_research/Archive/SC1_REACH_NOT_HORIZON_2026-09-25.md` §3-6 · `auto_research/Archive/VERIFY_ISCENE2_2026-09-26.md` §5-3 |
| g | **복사 기준선이 표적 공간에 있지 않다** | 복사 = 마지막 문맥 latent (copyz, online `encoder`) | 표적은 EMA `target_encoder` 의 LN 이다. `copy_h_ctxonly` (표적 공간 복사) 는 미측정이고, "크게 떨어지면 H6 헤드라인 철회" 조건이 걸려 있다. 마지막으로 보인 튜블릿 복사 · 수축 복사도 미측정이다 (수축 복사 계수 값은 문서에 없다) | P5 (GPU 1.5–2 h), DA-6 (GPU ~40 + 33 min) | CVPR_ORAL O-7 · DA-6 · R2-12 |
| h | **`skip2_w16` 칸 병기** | 정본 §4-1 은 w32 의 85.0 만 쓴다 | w16 은 83.89 vs 복사 71.67, +12.2, McNemar p 0.0001 이다. skip5_w16 은 60.56 vs 64.44 (−3.9) 다. p − 복사의 부호가 칸마다 바뀐다. CCA 는 칸에 안정하다 (L-4 는 CCA 를 주 통계로 권고) | 전 칸 표 + TOST (±5 pt) | CVPR_ORAL K8 · 통계 R5 · L-4 · O-6 |
| i | **v11 운동학 궤적은 팔마다 1 개** | CLAUDE.md §5-1 "등가속은 가림 없이도 이미 낮다 (shape 69.0)" | v11_full 운동 팔마다 궤적이 1 개다 (등속 260.12 cm/s, 경사 115.85 cm/s + 49.1 cm/s²) → "등가속 물체는" 으로 일반화하는 것은 통계적으로 불가능하다. ⚠️ TrainingEffects 는 "궤적 2 개" 라 적었다 (§3-2 모순 표) | 문장을 "이 궤적에서" 로 한정, 운동 일반화는 RollOut 궤적 수준으로 | CVPR_ORAL ★S-4 |
| j | **"k = 1–4 평평" 은 부재의 증거** | CLAUDE.md §5-1 · §8-5 ✅ "가려지는 시간은 무관하다" | 조건 × 위반 × k 168 쌍이면 SE ≈ 3.5 pt 라 5–7 pt 차를 배제하지 못한다. 모양의 k1 → k4 Δ +0.20 [−5.9, +6.3] 이 그 CI 폭이다 (`SYNTHESIS_2026-08-30.md`) | 동등성 마진 명시 (TOST, CPU) | CVPR_ORAL ★S-5 · `z_research/IntPhysGenV11/Archive/SYNTHESIS_2026-08-30.md` (Δ CI) · `z_research/IntPhysGenV11/Archive/surprising_score/RESULTS_2026-08-30.md` §3-(1) |
| k | **시간 척도 4 fps vs 30 fps** | v11 은 16 fps stride 3 → 튜블릿당 0.375 s 로, 사전학습 (4 fps, 0.5 s) 에 가깝다 | RollOut_v3 는 30 fps stride 1 → 0.067 s 로 7.5 배 빠르다. v2 는 16 fps 로 4 배다. RollOut 근거 (reach · 3a · H 계열) 에 이 단서가 걸린다. RoPE 인덱스 이동만으로 p 가 19–26 % 변한다 (H4). 부분 답은 P2 stride 다: cliff 6.57/4.11/2.46 슬롯 = 0.44/0.55/0.66 s → 고정 슬롯도 고정 시간도 아니다 | 모든 지평을 튜블릿 + 초로 보고, 각 predictor 의 학습 stride 로 평가, SSv2 fps 층화 (v3 에서는 native-rate 검사가 불가능) | REVIEW_v1 #7 · AC-W8 · #14 · CVPR_ORAL L-5 · DA-4 · `auto_research/Archive/P2_STRIDE_SWEEP_HORIZON_UNIT_2026-09-25.md` |
| l | **HERA 선점 · 54.35 일치** | 정본 목표 = 관측만으로 배우는 predictor | HERA (arXiv 2608.05523, 2026-08-06) 는 frozen V-JEPA 2-G + 기억 adapter 3 M (HERA 의 용어 — 우리 서술에는 memory 프레임 금지, Physion 학습) 으로 IntPhys 2 main 을 52.57 → **54.35** 로 올렸다. 동기 문장이 우리 명제와 거의 같아 스쿠프 위험 1 순이다. 그 54.35 는 우리 V-JEPA 2 ViT-H IntPhys 2 54.35 (C 24) 와 같은 숫자다 (모델이 다르다). 우연 일치인지 문서로는 확인할 수 없다 | 원문 대조 (정본 §5-10), 목표 절에 HERA 와의 차이 (학습 없는 측정 · 복사 대조 · 위치 readout) | CVPR_ORAL L-2 · `auto_research/Archive/LITERATURE_POSITIONING_2026-09-25.md` §6 |

⚠️ a · b · d 는 정본 P3 의 두 문장 ("보이는 것 2–3 칸 외삽" · "안 보이는 것 못 함") 에 직접 걸린다. 원 리뷰 표기도 "치명에 가깝다" 또는 "oral 을 막는 넷 중 하나" 다.
⚠️ e 는 3a 에만 걸린다. 속도 · 정체 · 위치 (가림 뒤 포함) 의 "encoder 에 정보가 있다" 는 확정 근거가 있다.

### 2-3. 그 밖에 리뷰가 남긴 미결 (정본 §4 에 없음)

| 공격 | 남은 단서 | 필요한 실험 | 출처 |
|---|---|---|---|
| readout 의존성 · 주 readout 선언 없음 | 정확 칸 판독에서 release 슬롯 계수 −0.074 가 거리 계수 −0.049 보다 크다. 템플릿 (진실 시각 vs 마지막 관측) 에 따라 k ≥ 11 에서 0.8–1.3 칸 갈린다. CLAUDE.md §5-5 읽는 규칙 (3×3 질량 > 0.035) 이 이후에 생겼다 | soft-argmax 연속 판독 재검정, 두 템플릿 병기 | REVIEW_v2 R4-1 · REVIEW_v1 #3 · CVPR_ORAL R2-3 |
| p 자의 분포 안 먼 슬롯 | 학습셋 `line_x` 슬롯 7 변위 중앙 56.6 px 이고 절반이 3 칸을 넘는데, held-out 오차는 0.76–0.87 칸이다 → 거리 한계가 latent 에서도 참이면 먼 슬롯 실패가 평균 속에 숨어 있어야 한다 | 슬롯 × 거리 × 3×3 질량 분해 (GPU 2 장 ~15 min) | CVPR_ORAL ★DA-5 (devil S9) |
| 먼 슬롯 p 에 정보가 있는가 | "p 에 정보가 없다" 를 probe 98 % 로 철회한 일이 반복될 위험이 있다 | 궤적 held-out probe 로 먼 슬롯 위치 · presence, z*/h 천장과 비교 (GPU ~1 h) | CVPR_ORAL R2-5 |
| forking paths · 확증 재현 | 약 48 h 사이에 분석 ~20 개, 철회 · 축소 ~30 개가 있었다. v3 분석은 모두 같은 784–1,176 clip (외형 28 중 14) 위에 있다. v3 `holdout=1` 은 탐색 표본 안이라 확증에 부적격이다 (X12) | 남은 14 외형으로 1 회 + Holm, 또는 RollOut_v2 126 궤적 (S-8, GPU 1–2 h), 또는 IntPhys 1 train 3,751 영상 (R2-10, GPU ~1 h + 40 GB) | REVIEW_v1 STAT-W5 · CVPR_ORAL S-8 · R2-10 |
| 수치 잡음 바닥 | v11 fp16 재계산 5,376 쌍 중 79 쌍이 뒤집힌다 → 448 쌍 칸 ±1.1 pt, IntPhys 1 ±1 쌍 (0.56 pt). IntPhys 2 +1.8 [−1.8, +5.5] 같은 작은 차에 걸린다 | 해석 하한으로 명시 | CVPR_ORAL S-7 |
| 문맥 끝 가림과 미래 앞 가림이 분리되지 않음 | v11 가림은 대칭이다 (문맥 끝 k + 미래 앞 k). late 가 무너지는 이유가 "문맥 끝 가림" 인지 "미래 앞 가림" 인지 가를 팔이 없다 → 정본의 "**마지막 관측에** 안 보이면" 좁힘에 붙는 단서 | 한쪽만 가리는 팔 | `z_research/IntPhysGenV11_occlusion_timing_ablation/README.md` |
| 양방향 target 번짐이 위치 템플릿에도 붙는가 | 외형 템플릿은 마지막 문맥 튜블릿의 양방향 target 토큰이다 (사건 전 튜블릿 90.4 % 와 같은 누수). RollOutV2 · V3 의 target 쪽 판독에도 붙는지는 답이 없다 | causal target 템플릿 재계산 (B5) | REVIEW_v1 MECH-W6 · CVPR_ORAL R2-7 |
| knockout 특이성 | r = d 에서 이미 0.07–0.28 하락한다 → hard −inf 의 비특이 softmax 재정규화가 의심된다. CLAUDE.md §5-5 의 "4~5 칸 떨어진 과거 자리에서 물체를 가져온다" 에도 따라붙는다 | 특이성 대조 (B3 · B4) | REVIEW_v1 MECH-W2/W3 |
| 학습 predictor 의 빈 장면 objectness | pv1 은 빈 v11 장면 귀무가 0.21 이고 epoch 따라 커진다 (출현 prior) | 학습 predictor 빈 장면 대조 | REVIEW_v1 APP-Q3 · E4 · `z_research/TrainingEffects/v11_presence/PRESENCE.md` |
| Ariel 의 EMA target 범위 | 48 프레임 전체 (양방향) 였는지 prefix 만이었는지 답이 없다. IntPhys 복사 미만 (prefix ep43 −16.1) 이 이것에 관여할 수 있다 | 학습 config 확인 | REVIEW_v1 JEPA-Q2 · #18 |

### 출처 (§2)
`auto_research/Archive/REVIEW_CVPR_ORAL_2026-09-25.md` · `auto_research/paper/REVIEW_v1_2026-09-25.md` · `auto_research/paper/REVIEW_v2_2026-09-26.md` · `auto_research/paper/PAPER_STORY_2026-10-08.md` §2-3 · §4 · `auto_research/Archive/H23_PREDICTOR_LAYERS_TRAJECTORY_2026-09-24.md` · `auto_research/Archive/SC1_REACH_NOT_HORIZON_2026-09-25.md` · `auto_research/Archive/P2_STRIDE_SWEEP_HORIZON_UNIT_2026-09-25.md` · `auto_research/Archive/VERIFY_ISCENE2_2026-09-26.md` · `auto_research/Archive/LITERATURE_POSITIONING_2026-09-25.md` · `z_research/context_encoder_analysis/exp_results/ctxenc_boundary_kinematics/RESULTS.md` · `z_research/IntPhysGenV11/Archive/surprising_score/RESULTS_2026-08-30.md` · `z_research/IntPhysGenV11_occlusion_timing_ablation/README.md` · `z_research/TrainingEffects/v11_presence/PRESENCE.md`

---

## 3. 문서끼리 모순 · 미규명 — 지우지 않고 나란히

규칙 (CLAUDE.md §12): 모순되는 결과는 한쪽을 지우지 않는다. 조건 차이를 적고, 판정을 내리지 못하면 "미결" 로 둔다.

### 3-1. 스토리 근거 (P3) 에 직접 닿는 것

| 무엇 | A 쪽 (문서 · 값) | B 쪽 (문서 · 값) | 조건 차이 | 판정 상태 |
|---|---|---|---|---|
| **지평의 단위 — 거리 vs 시간 (걸음 수)** | **거리**. RollOutV2 v2: 거리 100 px 당 −0.47, 슬롯 −0.28 (`z_research/RollOutV2/Archive/PREDICTOR_DISTANCE_LIMIT_2026-09-19.md`). identity_r8: v3 · v11 (속도 3 배 차) 이 거리 축에서 겹치고 40–60 px 에서 떨어지기 시작한다 (`z_research/RollOutV3/Archive/IDENTITY_R8_2026-09-26.md` §5-5). SC1: 슬롯만 R² 0, 거리만 0.68–0.84 (1 차, 적대 검증 전). REPORT_h23b: 귀무 보정 뒤 b_d −1.429, b_t +0.074 | **시간 · 걸음**. CVPR_ORAL R2: 시간 상한 R² 0.71 (K = 4.15) > 거리 상한 0.55, 정지점 ∝ 속도. TrainingEffects MOTION (v3): release 끊김은 시간 한계 쪽 (log–log −0.12). P2 stride: 존재 판독 절벽은 시간형 (CV 0.156 < 거리 0.211). I_SCENE: 정확 칸 · 실영상에서는 시간 효과도 같거나 크다 | 데이터 (v2 16 fps / v3 30 fps / v11), 판독 (Chebyshev · 정확 칸 · 연속 · 자 vs 템플릿), 측도 (위치 적중 vs 물체다움). 통계 감사는 "각각 다른 측도에서만 맞다" 고 했다 (log 지수 위치 1.06 · 물체다움 0.38). v3 속도 폭이 1.75–2.4 배라 corr(슬롯, 거리) 0.95 | **미결.** 정본은 거리 쪽으로 썼다. 측도를 밝혀 쓰지 않으면 리뷰 공격이 그대로 남는다 (§2-2 a). TrainingEffects 는 "자료 · 창이 달라 병기" 로 둔다 |
| 짧은 가림을 이어 붙이는가 | 자 없는 vanish block 비교: 문맥 끝 가림 이동 k = 2–4 에서 block 98–100 % 가 빈 미래 쪽이다 (`z_research/RollOutV3/Archive/RESULTS_2026-09-25.md` §5) | identity_r8 '있음': k = 2 초반 (t1–t2) 에 ramp 58–69 % · flat 34–59 % (IDENTITY_R8 §5-1) | 판정 도구 (파라미터 0 짝 비교 vs 학습된 57 분류 자), 슬롯 범위 (전체 vs 초반) | **미결.** 문서는 "짧은 가림을 이어 붙인다" 를 **조건부** 로 둔다 |
| 자로 읽은 p vs 채점의 k 의존 | identity_r8: 문맥 끝 가림 p '있음' (t3–t6) 이 k = 1 → 4 에서 계속 준다 (정지 28 → 17, flat 62 → 37, ramp 60 → 25 %) | CLAUDE.md §5-1: 채점은 k = 1…4 평평 | 재는 양이 다르다 (물체가 있는가 vs 쌍 판정) | **병기** (IDENTITY_R8 §5-4b). 단 "평평" 자체가 부재의 증거다 (§2-2 j) |
| 가려진 물체가 남는 모양 순서 | identity_r8 §5-4: sphere · torus 는 74–93 % 남고, 나머지는 20–41 % (정지 1–8 %) | CLAUDE.md §5-3: 채점 지킴 sphere 36 · torus 3 (최저) | 위치 + 정체 자 vs 채점 (scalar surprise 편향) | **병기.** 원 문서는 둘을 잇는 주장을 하지 않는다. sphere · torus 인 이유는 모른다 (회전 불변 외형은 추측) |
| "옛 자리에 남는가 vs 없어지는가" | gated 시간 정렬: v11 ramp k = 0 의 per-tubelet A-gate 가 슬롯 0–6 을 통과하는데, j* = 7.9 / 9.0 / 9.8 / 9.9 / 10.0 / 10.2 / 10.2 (슬롯 3–6 이 t≈10 자리와 닮음) | CLAUDE.md §6: "p 가 물체를 마지막 관측 자리에 둔 채 머문다" 는 **철회** (자 기본값, 마지막 관측 칸 attention 0.01~0.03) | 도구 (시간 정렬 gated Δc vs 위치 자 attention 질량), 조건 (k = 0 가림 없음 vs v11 가림) | **미결** (`CLAIM` §6-1). "없어진다" 는 진실 자리에서만 확인됐다 |
| "encoder 충분 · predictor 병목" | 09-25 철회: 충분성 대조 (z_cv) 가 진실 위치를 썼다 (`auto_research/Archive/TRAINED_PREDICTORS_SIGNATURES_2026-09-25.md` 정정 2-1 · M13 정정 2) | 10-08 정본 (iv) 가 다시 세움: `FUTURE_FROM_CONTEXT` (z 자 ≈ h 자) · `ENCODER_MOTION_READOUT` (EK100 R² .69) · `INTPHYS2_ENCODER_LINEAR_PROBE` (지도 자 73.7) | 재근거가 진실 위치 없이 과거 z 만 쓴다. 다만 자 B 는 지도 라벨 곡선을 붙인다 | **부분.** 정보 존재는 재근거가 있다. "encoder 가 미래를 표현한다" 는 계속 금지. 3a 는 미결 (§2-2 e) |
| wall 에서 p 는 무엇을 하나 | RollOutV2 위치 자: 2 슬롯 통과 뒤 물체를 잃는다 (통과 질량 0.61 / 0.48 → ≤ 0.18) | `FUTURE_FROM_CONTEXT`: wall 에서 p 가 멈춤을 읽는다 | 자가 다르다 (등속 학습셋 attentive 자 vs 장면 라벨 자 B · 템플릿 읽기) | **병기.** ledge 의 "자 둘이 다른 답" (정본 iii-3) 과 같은 구조라 조건 표로 같이 낸다 |
| Ariel 의 지속 판정 | `TRAINED_PREDICTORS_SIGNATURES`: pv1 · Ariel 은 ≈ 8 튜블릿 진실 기울기, 정체 ≈ 5 칸 | TrainingEffects attn 자: "정의에 따라 반대" (두 자 모두 ↑ 라 쓴 것을 정정) | 판독 (템플릿 vs p@h 자 — p@h 는 p 토큰에서 따로 검증 안 됨) | **병기 · 판독 의존.** 단독 인용 금지 |
| 고정 문턱 판독의 부호 | IntPhys 2 post-FT 영속 Δ −0.24 (한 자) | +0.13 (다른 자) | 자 (p@h vs p@p) | **단독 인용 금지** (TrainingEffects 정정) |
| 역사 (문맥 앞부분) 의 쓰임 | v11 M2: 있음 = 경계 토큰, 자리 = 역사 | v3 M13 · M6: 자리 불변, 있음 = 역사. 역사 제거 부호도 반대다 (v11 Eb_only − E +0.10 vs v3 `hist_none` +0.01 · `pred_last*` −0.045 ~ −0.073) | 데이터 · 개입 방식 | **소멸 (스토리에서) · 미결 (사실로).** 서로를 "역사 불사용" 근거로 쓰지 않는다 |
| Ariel epoch 포화 | SC1: s1_C16 plateau 3.52 → 5.08 칸 단조 증가 | I_SCENE 1 차: "더 학습해도 안 는다" | cosine lr 끝과 교락. EK100 R50 불변만 확정 | **미결 (팬)** · §2-2 f |

### 3-2. 측정 · 도구 · 표본

| 무엇 | A 쪽 (문서 · 값) | B 쪽 (문서 · 값) | 조건 차이 | 판정 상태 |
|---|---|---|---|---|
| **두 복사 기준선 (IntPhys 1, `skip2_w16`)** | H6: 복사 71.67, p − 복사 +12.2 [+6.7, +17.8] (`auto_research/Archive/H6_INTPHYS1_WHAT_THE_SCORE_MEASURES_2026-09-24.md`) | TrainingEffects: 복사 73.89, +9.4 [+3.3, +15.6] (`z_research/TrainingEffects/ip1_copy/IP1_COPY.md`) | 4 쌍 차이다 (±1 쌍 허용 초과). std w32 CI 상단도 +8.4 (CVPR_ORAL) vs +8.9 (정본 §2-2) 로 다르다. causal w32 는 +7.2 vs +6.7 | **미결.** REVIEW_v1 AC-a 가 "인용 전에 맞춰라" (C6) 고 했으나 맞춘 결과가 없다. 정본 §재현은 w32 85.0 에 대해서만 "파생 json 두 개 일치" 를 확인했다 |
| **v11 운동학 궤적 수** | CVPR_ORAL 통계 §1: 운동 팔마다 궤적 **1 개** (정지, 등속 260.12 cm/s, 경사 115.85 cm/s + 49.1 cm/s²) | `z_research/TrainingEffects/README.md` §5 단서 (1): 운동 조건마다 궤적 **2 개**. z_training 09-17 철회문: "v11 은 속도 1 종" | 세는 단위가 문서에 없다 (팔 vs 조건 · 방향 · flat/ramp 등) | **미규명.** 어느 쪽이든 "등가속 물체는" 일반화는 막힌다 |
| RollOutV3 자 보정 (좌표 치우침) | `z_research/RollOutV3/_superseded/RERUN_2026-09-24.md` §6: p (+11.46, −2.58) · z (−4.49, −8.24) · h (−7.28, −5.08) px | `_superseded/README_2026-09-25.md`: p (+5.5, −6.3) · z (−4.9, +0.3) · h (−1.1, +5.6) px. r1 §1 은 (+5.45, −6.30) 을 쓴다 | 둘 다 presence 자 시기다 (자 지문 다름 가능성 — 문서에 없음) | **미결 (어느 쪽이 맞는지 문서에 없음).** 둘 다 대체됐고 현행 결론 (identity_r8) 에는 쓰지 않는다. ⚠️ 이것은 자 쪽 좌표 보정이며 인용 금지 α · 보정 계수와 다른 양이다 |
| `static_occlusion` 의 probe vs 채점 | p self 가 가장 낮다 (85.13 / 87.55) | 채점 vanish 는 그 조건에서만 살아남는다 (sens +20.54) | 그 두 head 가 미수렴 (train_acc 0.931 / 0.954) | **미해결** (RESULTS_2026-08-30 §11-(5)) |
| colour 의 k = 4 튐 | colour 만 k = 4 에서 sens 20.8 | 다른 위반은 k = 1…4 평평 | — | **설명 미상** (같은 문서 §3-(4), E5 로 갈릴 때까지) |
| knockout 층 8 | −18.06 (`z_research/IntPhysGenV11/figures/knockout/README.md`, `knockout_window__v11_vith`, 288 pair 표본) | −14.4 (`PAPER_STORY_2026-09-06` 재현절, `knockout_w1`, v11_full) | 표본 · 데이터셋 | **병기** (09-06 이 ⚠️ 로 명시) |
| IntPhys 1 단창 수치 셋 | `z_world_model_analysis/WHY_INTPHYS_WORKS_2026-08-20.md` 87.78 (start = 24, C = 8, 180 쌍) | `VERIFICATION_2026-08-20.md` 88.3 [84, 93] · `MECHANISM_2026-08-20.md` §6 88.33 · CLAUDE.md §6 "고정 단창 하나로도 88.33" | 같은 창의 반올림 차이로 보이나 문서가 밝히지 않는다 | **미규명 (작음).** sliding 88.89 와 구분해서 쓴다 |
| blender presence 축 0.834 | `MECHANISM_2026-08-20.md` 가 인용 | 같은 그룹 문서 어디에도 원 값이 없다 | — | **출처 미확인.** 인용하지 않는다 |
| 학습 predictor 의 IntPhys 1 p − 복사 | Ariel prefix ep43 −16.1 (`TRAINED_PREDICTORS_SIGNATURES` §3-6) | AR ep18 +7.2 (`auto_research/Archive/PREDICTOR_SHARE_2026-09-29.md`, `copy_blk` 83.33) | predictor (prefix vs AR) · 표적 공간 (표준 vs 블록별) | **병기 (모순 아님).** 섞어 읽지 않는다. AR 은 쌍 정확도만 비교한다 (CLAUDE.md §3-1) |
| AR permanence 비 | `VERIFY_AR`: R .76 / .53 / .49 | `auto_research/Archive/AR_PREDICTOR_2026-09-27.md` 표: .72 / .50 / .50 | 산식 (n 가중 vs 토큰 pooled) | **병기.** 결론은 같다. 산식 명시가 요구된다 |
| 수치 소차 (RollOutV3) | GEOMETRY p resid 81.4 · PROBE ridge 24.7 / 18.9 · presence 자 위치 L2 16.1 px (RESULTS) | 감사 81.6 · 23.1 / 19.0 · COMPARE 평균 19.0 px | 정의 차이가 기재되지 않음 | **미규명 (작음)**. presence 자 시기 수치는 현행 결론에 안 쓴다 |
| gravity 꼭대기 두 판 | 미래 넷 판: apex float 93.8 | 미래 다섯 판: apex float 33.3 · arc 62.5 | 데이터가 같이 바뀌었다 (g 170 → 120, 꼭대기 f57/f42 → f66/f30) | **원인 분리 불가 — 두 판을 섞어 읽지 말 것** (`z_research/v11_realistic/README.md` §4) |
| subspace_v2 정확도 | static_occlusion 63.37 (bf16 probing 캐시) | 참조 채점 58.16 (같은 block) | bf16 근접 tie 뒤집힘 | **Δ 는 믿고, 정식 수치는 fp32 재추출 후** (`z_research/context_encoder_analysis/exp_results/ctxenc_subspace_v2/RESULTS.md` §1) |

### 3-3. 프로토콜 · 벤치마크

| 무엇 | A 쪽 (문서 · 값) | B 쪽 (문서 · 값) | 조건 차이 | 판정 상태 |
|---|---|---|---|---|
| `dual_encoder` | `analysis/intphys2/ANALYSIS_NOTES.md` T9: 67.22 < 68.89 → "적용하지 말 것" | 현행 `surprise_c16t32` · Garrido 하네스는 `dual_encoder: true` (CLAUDE.md §1-7) | — | **번복 근거가 문서에 없다.** 권고는 뒤집혔다 |
| IntPhys 1 을 IntPhys 2 프로토콜로 | 우리 구현: w16 67.22 / w32 66.11 (`analysis/intphys2/configs/_superseded/README.md`) | 논문 87.22 (약 20 pt 차) | 하네스 (IntPhys 2 통짜 yaml) | **원인 미규명.** IntPhys 1 은 Garrido 프로토콜로만 보고한다 (`z_research/Benchmarks/PROTOCOLS.md` §7) |
| Joseph 2026 층별 probe 의 정규화 | `analysis/intphys1/HANDOFF_SPEC.md`: "`self.norm` 적용 금지". 단일 fold 기대값 87.5 @ L06 | `analysis/intphys1/PAPER_VS_OURS.md` 진단 (1): post-LN 이 논문 쪽일 가능성. 5-fold peak L15 80.83 ± 3.98 (논문 chart ~95–100 @ L8–10) | ViT-L (기준 모델 ViT-H 아님), fold 평균 · peak 층 이동 | **미해결 (방향 반대).** 논문 값은 시각 판독 ("chart-only") 이다 |
| V-JEPA 2.1-g IntPhys 1 | 51.11 (`z_research/Benchmarks/README.md` §4) | 90 블록 중 80 이 정확히 0.5 | 공식 해상도 384 를 256 으로 쟀다 | **원인 미규명** |
| IntPhys 2 수치 여럿 | 우리 Main 506 쌍 54.35 (C 24) · 복사 55.53 (C 42) | MATRIX Main split test 252 쌍 57.94 · 제안서 57.94 / 56.75 (프로토콜 미표기) · 09-06 재현절 53.26 (ViT-L) · 논문 V-JEPA 2-h 57.51 | 표본 · 모델 · 프로토콜 | **병기 · 비교 금지** (CLAUDE.md §1-3). 논문 57.51 은 확인하지 않은 문헌 값이다 (§4-1) |
| Benchmarks 진행 상태 | `z_research/Benchmarks/README.md` §4 · HANDOFF §1-1 · §7: "GRASP · InfLevel 미완" | OVERVIEW §9-2: InfLevel 은 3 모델 × 3 property 가 전부 나옴, GRASP 만 미실행 (`_resolved.yaml` 하나) | 문서 갱신 시점 | **README 표가 낡았다.** 정본 §5-2 의 "GRASP · InfLevel 복사 기준선" 은 둘 다 아직 할 일이다 |
| InfLevel gravity · solidity | 점수는 있다 (ViT-H 51.95 / 50.00) | 원리적으로 못 푼다 (priming 절단, Garrido App. E) | — | **모델 비교에 쓰지 않는다 — continuity 만** (`z_research/Benchmarks/PROTOCOLS.md` §5-3) |

### 3-4. 문서 운영 (인프라 · 체크포인트)

| 무엇 | A 쪽 | B 쪽 | 판정 상태 |
|---|---|---|---|
| 프로토콜 개수 | `configs/protocols/README.md` 하단 · CLAUDE.md §2-2: "4 종" | 같은 README 상단 표 · `z_research/OVERVIEW_2026-09-22.md` §2-2: **5 종** (`attn_probe_xfer` 추가) | **문서 내부 불일치** — 고치지 않은 채 남아 있다 |
| 인덱스 생성기 | `z_research/scripts/HOWTO.md` §1 · `datasets.md` `v11_full` 3 단계가 `z_scripts/world_model_analysis/build_*` 를 부른다 | `z_scripts/` 는 09-21 에 통째로 삭제됐다 (CLAUDE.md §2-0) | **HOWTO §1 과 v11_full 3 단계는 지금 안 돈다.** v11 계열 index 는 복사본이 유일하다 (git 밖 · 재생성 불가) |
| Ariel AR 체크포인트 이름 | `models.md`: `vith_ariel_ar_ep18` = `ar_future_1e_5/latest.pt` | 09-28 17:22 에 ep36 으로, 20:19 에 ep38 로 덮였다 → `ar_ep18` 과 `ar_ep36` 은 **같은 파일** 이다 | **어느 epoch 인지 미검증.** ep38 만 고정 사본이 있다 |
| 지금 판 AR 의 epoch | CLAUDE.md §3-1: "prefix ep45 · full ep40 · **ar ep18**" | 정본 §5-3: "AR **ep40**". `z_training/ARIEL_CHECKPOINTS.md` 1-b: `context_full_ar_1e_5/latest.pt` 17:36 판은 실은 AR ep40 | **문서끼리 다르다.** 정본 §5-3 실행 전에 확정이 필요하다 |
| Ariel bc ep5 / 19 / 30 / 41 / 43 / ep5_full | 09-23 받은 체크포인트. TrainingEffects · REVIEW 가 `ariel_ep43` 수치를 인용한다 | 09-27 폴더 삭제 (`available: false`, 실행하면 죽는다). ep45 는 다른 파일이다 | **옛 수치는 옛 파일의 것으로만 남긴다** — ep45 로 옮겨 적지 않는다 |
| `IntPhysGenV8/exp_results/` | HANDOFF rsync 목록에 있다 | 비어 있다 (4.0 K). v8 79.10 / 73.44 의 출처는 `z_exp/world_model_analysis/results/` | **출처 경로만 다르다** |
| `REVIEW_CVPR_ORAL` 의 후속 경로 | 머리에 "이 다음 판의 논문 초안은  에서 이 리뷰를 반영한다" | 경로가 빈칸이다. 대체 표시도 없다 | **실질 후속은 DRAFT_v1 · REVIEW_v1 → 정본** (G04b 판정) |

### 출처 (§3)
`z_research/RollOutV2/Archive/PREDICTOR_DISTANCE_LIMIT_2026-09-19.md` · `z_research/RollOutV3/Archive/IDENTITY_R8_2026-09-26.md` · `z_research/RollOutV3/Archive/RESULTS_2026-09-25.md` · `z_research/RollOutV3/_superseded/RERUN_2026-09-24.md` · `z_research/RollOutV3/_superseded/README_2026-09-25.md` · `z_research/TrainingEffects/README.md` · `z_research/TrainingEffects/v3_motion/MOTION.md` · `z_research/TrainingEffects/ip1_copy/IP1_COPY.md` · `auto_research/Archive/H6_INTPHYS1_WHAT_THE_SCORE_MEASURES_2026-09-24.md` · `auto_research/Archive/TRAINED_PREDICTORS_SIGNATURES_2026-09-25.md` · `auto_research/Archive/FUTURE_FROM_CONTEXT_2026-10-02.md` · `auto_research/Archive/PREDICTOR_SHARE_2026-09-29.md` · `auto_research/Archive/AR_PREDICTOR_2026-09-27.md` · `auto_research/Archive/I_SCENE_CAUSAL_BOTTLENECK_2026-09-25.md` · `z_research/context_encoder_analysis/Archive/CLAIM_NO_STATE_EVOLUTION_2026-09-15.md` · `z_research/context_encoder_analysis/exp_results/ctxenc_time_alignment_gated/RESULTS.md` · `z_research/context_encoder_analysis/exp_results/ctxenc_subspace_v2/RESULTS.md` · `z_research/IntPhysGenV11/Archive/surprising_score/RESULTS_2026-08-30.md` · `z_research/IntPhysGenV11/Archive/PAPER_STORY_2026-09-06.md` · `z_research/IntPhysGenV11/figures/knockout/README.md` · `z_world_model_analysis/{WHY_INTPHYS_WORKS,VERIFICATION,MECHANISM}_2026-08-20.md` · `z_research/v11_realistic/README.md` · `analysis/intphys2/ANALYSIS_NOTES.md` · `analysis/intphys2/configs/_superseded/README.md` · `analysis/intphys1/{HANDOFF_SPEC,PAPER_VS_OURS}.md` · `z_research/Benchmarks/{README,PROTOCOLS}.md` · `z_training/runs/_matrix/MATRIX.md` · `z_training/ARIEL_CHECKPOINTS.md` · `configs/protocols/README.md` · `z_research/scripts/HOWTO.md` · `z_research/OVERVIEW_2026-09-22.md` · `z_research/HANDOFF_2026-09-22.md`

---

## 4. 확인 안 한 수치

"확인" 은 정본 §재현 의 뜻으로 쓴다. 2026-10-08 에 산출물에서 다시 읽거나 다시 셌으면 ✓ 다. 아래는 ✓ 가 없거나, 원 배열 · 원 캐시 · 스크립트가 지금 손에 없는 수치다. **값 자체가 틀렸다는 뜻이 아니다.** 인용할 때 "문서 값" 이라고 밝힌다.

### 4-1. 정본 §재현 "확인하지 않은 것" (문서 값)

| 수치 | 문서 값 | 쓰는 곳 | 출처 (그 문서의 `## 재현`) | 덧붙는 단서 |
|---|---|---|---|---|
| reach 2–3 칸 (40–60 px) | 물체다운 토큰은 마지막 관측에서 2–3 칸 안 · 거리 100 px 당 −0.47 / 슬롯 −0.28 | P3 (i) · CLAUDE.md §5-5 | `z_research/RollOutV2/Archive/PREDICTOR_DISTANCE_LIMIT_2026-09-19.md` | 귀무 미보정 · clip 단위 (§2-2 a) |
| late 0 슬롯 | 물체다운 토큰이 남는 길이: late 0 < early/mid 3 < 가림막 없음 4–5 | P3 (ii) · Figure 1 | `z_research/RollOutV2/README.md` §4b | 렌더 누수 (§2-2 c) · 대칭 가림 (§2-3) |
| flat_a 28 / 32 / 54 | p 28.1 ≈ 등속 외삽 32.4 vs 진실 54.0 (px) | P3 (iii) 3a · 합격선 | `z_research/RollOutV2/Archive/PREDICTOR_DISTANCE_LIMIT_2026-09-19.md` (표 'flat 가속' 행) · `z_research/RollOutV2/README.md` §4b · `z_research/context_encoder_analysis/Archive/STATE_EVOLUTION_CAPABILITIES_2026-09-19.md` §2 | v3 에서는 검정력 부족 (참 효과 r8 +8.4 px < 1 칸, CVPR_ORAL S4 · DA-3) |
| 실영상 reach 밖 비율 | SSv2 46 % (R50 2.7) · EK100 51 % (R50 2.1) | 정본 §4-3 corner case 방어 | `auto_research/Archive/I_SCENE_CAUSAL_BOTTLENECK_2026-09-25.md` §4-3 (1 차, 2 차 검증 대상 아님) | 합의 토큰 필터 낙관 (R4-3) |
| 실영상 가림 띠 | k = 4 근거리 hid .21 vs vis .42 | P3 (ii) · 합격선 permanence | 같은 문서 §4-4 (2 차 검증 확정) | 산식 (n 가중 vs pooled) 은 AR 문서끼리 다르다 (§3-2) |
| z 자 · p 자 위치 오차 | z 자 B 5–10 px (v11 late k1–4) · p 자 B 12–30 px | P3 (iv) | `auto_research/Archive/FUTURE_FROM_CONTEXT_2026-10-02.md` §2 (1 차) | 자 B 는 장면 종류를 읽고 지도 곡선을 붙인다. 실행 노드 vll5 / vll2 |
| AR 가속 반영 | arc −0.42 · ledge −0.46 (릴리즈 +0.08 / −0.25, h ≈ 1) | P3 (iii) 3a · 합격선 | `auto_research/Archive/AR40_READOUT_2026-09-30.md` §1 · `VERIFY_AR40RD_2026-09-30.md` (1 차) | 절대값은 자에 따라 다르다 (교차 head 에서 부호 반전). 이 수치는 인용 금지 α 분해와 다른 양이다 |
| ego-motion R² (정본 §재현 표 밖 — 이 장이 더함) | encoder 토큰 평균 → .69 / .63 · EK100 미래 변위의 67–75 % 가 카메라 운동 | P3 (iv) 실영상 | `auto_research/Archive/ENCODER_MOTION_READOUT_2026-09-29.md` (1 차) | 문서 값 train = test 문제를 VERIFY_R3 가 정정 (clip-CV 자기 일관성 .53 / .36 을 encoder 가 넘는다) |
| IntPhys 2 논문 값 (정본 §재현 표 밖 — 이 장이 더함) | V-JEPA 2 계열 57.51 · 사람 96.4 | P2 (c) 문헌 | Bordes et al. 2025 (문헌) | 우리 측정 (54.35, 506 쌍) 과 프로토콜이 다르다 — 비교 금지 |

### 4-2. 원 배열 · 원 캐시 · 스크립트가 지금 손에 없는 것

| 수치 | 무엇이 없나 | 그래서 무엇만 확인됐나 | 출처 |
|---|---|---|---|
| IntPhys 1 복사 85.0 [79.4, 90.0] | 원 배열 `/data2/.../ip1score_final/arrays_v360.npz` 가 vll3 에 없다 | 파생 json 두 개 (`TrainingEffects/ip1_copy/ip1_copy.json` · `auto_research/exp_results/h6/h6g_intphys1.json`) 가 서로 일치한다는 것까지 | 정본 §재현 (2026-10-08 산출물 재확인) |
| 위치만 다른 쌍 0.0 / 17.6 vs 7.9 / 2.3 | 원 캐시가 vll5 `/local_datasets` 에만 있다 | json (`auto_research/exp_results/scene/rollout2_copy_voe.json`) 만 다시 읽었다 (✓ json 만) | 정본 §재현 |
| 리뷰가 재계산한 수치 전부 — 궤적 CI (os 0.024 [0.009, 0.039] 등) · 정지점 비 1.96 / 2.07 · log 지수 1.06 / 0.38 · CCA (v11 p\|복사 틀림 51.8 [49.2, 54.2]) · 시간 상한 R² 0.71 · 잡음 바닥 79 쌍 | 스크립트가 옛 세션 스크래치패드 (`/tmp/claude-1070/.../4d3cc623-327c-416f-8a40-205a59d78117/scratchpad/`) 에 있다 — 레포 밖이고, 지금 남아 있는지는 문서에 없다 | 입력 배열 경로는 적혀 있다 (`cache/auto_research/v3_h5/h5.npy` · `v11_h6f/l1.npy` · `h6g_intphys1.json` 등, vll6 `/data2`) | `auto_research/Archive/REVIEW_CVPR_ORAL_2026-09-25.md` 재현 블록 · `REVIEW_v1` 재현 |
| VERIFY_0925 교차 비평 재계산 | `verify0925/CROSS/` (`x1_ssv2.py` · `x3_lawfe.py` · `x4_oracle.py`) 가 세션 스크래치에 있다 — 휘발 | 원문 결과는 `auto_research/exp_results/verify/` 에 있다 | `auto_research/Archive/VERIFY_0925_2026-09-25.md` · `VERIFY_0925_R2_2026-09-25.md` |
| auto_research 원자료 캐시 일반 | vll6 `/data2/local_datasets/world/world_analysis/cache/auto_research/` (`v3_h4` · `v3_h5` · `v3_h8` · `v3_p3*` · `flowtrack_*` 등) | json 요약은 레포 `auto_research/exp_results/` 에 있다 (git 밖, D2) | `auto_research/README.md` §7 |
| Ariel bc ep5–43 수치 (reach epoch 곡선 3.52 → 5.08 · run 안 IntPhys 1 +21.7 · prefix ep43 IntPhys 1 69.44 / −16.1) | 체크포인트 09-27 삭제 | 옛 산출물 json 의 값으로만 남는다. 다시 돌릴 수 없다 | `z_research/TrainingEffects/MODELS.md` · CLAUDE.md §3-1 |
| SSv2 Ariel 학습 데이터 `ssv2_min48` | vll5 전용 (`k400_train_40k` · `ssv2_min48_vll6` 는 vll6 전용) | 노드를 옮기면 경로부터 다시 확인해야 한다 | `configs/training/datasets.md` |

### 4-3. 기준선 · 대조가 미측정이라 해석이 열린 수치

| 수치 | 무엇이 빠졌나 | 그래서 못 하는 주장 | 출처 |
|---|---|---|---|
| 사전학습 그룹 B 의 IntPhys 1 (예: 96.1) | 복사 기준선이 없다 | 96.1 이 predictor 몫인지 모른다 | `z_research/TrainingEffects/README.md` §5 (8) |
| AR v11 83.2–83.8 | v11 `copy_blk` (블록별 표적 공간 복사) 미실행 | AR 의 v11 우위가 예측인지 모른다. 등속 이동은 릴리즈보다 낮다 (85.0 vs 91.1) | `z_training/ARIEL_CHECKPOINTS.md` §5-2b |
| IntPhys 1 · v11 의 `copy_h_ctxonly` | 표적 공간 복사 미실행 | §2-2 g — H6 헤드라인의 조건부 철회 조건이 열려 있다 | CVPR_ORAL O-7 |
| IntPhys 2 Ariel bc ep30 | 어느 팔 (prefix / full / ar) 인지 문서에 없다 | 팔 간 비교에 쓰지 않는다 | `z_research/Benchmarks/exp_results/intphys2/bench_intphys2_main_ariel_bc_ep30_w16/column_best.md` |
| v11 post-FT 위치 자 (RollOut β ≈ 2) | v11 vanish clip 의 train 포함 여부를 대조하지 않았다 · 채점 재계산 안 함 · ledge/wall pos/imp 두 미래 판정 안 함 · 자 선형 층이 post-FT p 에 이식되지 않는다 (y 편향 50–80 px) | in-distribution 으로 읽는다. d_truth · MAE · β 의 post-FT 쪽은 최댓값 칸으로만 읽는다 | `z_research/predictor_training/predictor_IntPhysGenV11_PFT/Archive/RESULTS_2026-09-16.md` |
| v11_realistic token surprise 배경 몫 | 3×3 이웃 정의에 따라 몫이 바뀌는지 결론이 안 바뀐다고 **가정만 했다** · p 배경 칸이 어느 세계 쪽인지 분리 안 함 · 1–4 픽셀 차의 Δ 기여 미분리 | 배경 Δ 부호 (vanish · ledge 96–100 % vs gravity 반대) 해석은 조건 차이로만 | `z_research/v11_realistic/figures/token_surprise/README.md` |
| EK100 AR ep40 val 21.74 (09-30) | `predictor_checkpoint` 키가 빠져 **릴리즈가 돌았다** | 그 run 들 (`ek100_rel_third10_grid8_ar40` · `ek100_vith_official256_released_grid8_ar40`) 인용 금지. 재실행 AR 22.03 vs 릴리즈 21.80 | `auto_research/Archive/AR_PREDICTOR_2026-09-27.md` §2-9 |
| Ariel M13 역사 삭제 팔 | `build_prefix_mask` 버그로 무효 | 판정 보류 | `auto_research/Archive/TRAINED_PREDICTORS_SIGNATURES_2026-09-25.md` |

### 4-4. 문헌 · 외부 수치 중 원문과 대조하지 않은 것

| 수치 · 문장 | 상태 | 출처 |
|---|---|---|
| DINO-world IntPhys 91.3 · "~1 s 에서 부정확" | `LITERATURE_POSITIONING` 기준으로만 ✓, 원 논문 미대조 | REVIEW_v1 AC-c · Discussion #27 |
| Garrido 3–4 s 한계 (원 리뷰 표현 "기억 한계" — 현재 금지 프레임) · "no-prediction 기준선은 처음" | 원문 재확인 전 인용 금지 | CVPR_ORAL L-3 |
| Joseph 2026 (arXiv 2602.07050) 가속 판독 가능 · 중간층 > 최종층 | split 규약 확인 전에는 우리 H1 과 "충돌" 로 쓰지 않는다 | CVPR_ORAL 문헌 판정 H 행 |
| HERA IntPhys 2 main 52.57 → 54.35 | 원문 대조 필요 (우리 54.35 와 같은 숫자) | §2-2 l |
| GEOPHYS (2606.20707) predictor 없이 IntPhys 2 93.3 % | split 미확인 — 인용 금지 | `auto_research/Archive/LITERATURE_PREDICTOR_ROLE_2026-09-29.md` · `auto_research/paper/RETHINK_PREDICTOR_ROLE_2026-09-29.md` §6-b |
| "Objects in Generated Videos Are Slower" (생성 중력 0.38–2.27 m/s²) | venue 미확인 | `auto_research/Archive/LITERATURE_STATE_WORLD_MODELS_2026-09-26.md` |
| P1 동물 · 새 환경 적응 (병아리 가려진 물체 추론 · Lake et al. 2017) | 후보만 있다 — 실물 확인 전 인용 금지 | 정본 §2-1 |
| Physics-IQ 류 생성형 대응 문헌 | 실물 확인 후 인용 | 정본 §2-2 (c) |

⚠️ P1 은 세 문헌 문서 (`LITERATURE_POSITIONING` · `LITERATURE_STATE_WORLD_MODELS` · `LITERATURE_PREDICTOR_ROLE`) 모두에서 영유아 · 동물 문헌이 비어 있다. 정본 §2-1 의 영아 목록 (Baillargeon & DeJong 2017 등) 은 사용자 NARRATIVE_v2 §A 에서 온 것이다 (01 장).

### 4-5. 체크포인트 · 파일 정체 미검증

| 무엇 | 미검증 내용 | 출처 |
|---|---|---|
| `vith_ariel_ar_ep18` = `vith_ariel_ar_ep36` | 같은 파일. 어느 epoch 인지 모른다 | `configs/protocols/models.md` · `cvpr/README.md` §7 |
| `IntPhysGen_v11.tar` (NFS, 76 G, 09-10) | early/mid 조건 포함 여부 | `z_research/HANDOFF_2026-09-22.md` §3 |
| V-JEPA 2.1-g 체크포인트 출처 · GitHub 레포 공개 여부 · vll3 디스크 | 09-22 기준 미검증 | 같은 문서 |
| videomae2g testbed 기본 `c16t32` | 엔진에서 도는지 미검증 | `cvpr/README.md` |
| gravity_realistic 렌더 위치 · g | 현행 미래 다섯 판: g 120, pixel check FAIL 최대 6.3 / 6.4 / 16.0 px (`z_research/v11_realistic/README.md` §4). 옛 미래 넷 판: g 170, 최대 6.5 / 13.7 px. ⚠️ `configs/protocols/datasets.md` 의 `gravity_realistic` 항목은 한 항목 안에 g 120 (재렌더 문장) · g 101 (본문) · "실제 g 170" (10-07 18:07 정정, 옛 판) 이 섞여 있다 | `z_research/v11_realistic/README.md` §4 · `configs/protocols/datasets.md` `gravity_realistic` |
| 공식 V-JEPA 2-AC 체크포인트 손실 버그 | jepa-wms `hubconf.py:138` 의 버그와 우리 README §4 "ẑ_2 두 번" 이 같은 버그인지 미확인 (AC 실험 전 확인 필요, `09b` §3-4) | `z_AC_training/Archive/PLANNING_AND_ROBOCASA_2026-09-20.md` §4 |

### 출처 (§4)
`auto_research/paper/PAPER_STORY_2026-10-08.md` §재현 · `z_research/RollOutV2/Archive/PREDICTOR_DISTANCE_LIMIT_2026-09-19.md` · `z_research/RollOutV2/README.md` · `auto_research/Archive/{I_SCENE_CAUSAL_BOTTLENECK_2026-09-25,FUTURE_FROM_CONTEXT_2026-10-02,AR40_READOUT_2026-09-30,VERIFY_AR40RD_2026-09-30,AR_PREDICTOR_2026-09-27,TRAINED_PREDICTORS_SIGNATURES_2026-09-25,REVIEW_CVPR_ORAL_2026-09-25}.md` · `auto_research/Archive/LITERATURE_{POSITIONING_2026-09-25,STATE_WORLD_MODELS_2026-09-26,PREDICTOR_ROLE_2026-09-29}.md` · `z_research/TrainingEffects/{README,MODELS}.md` · `z_training/ARIEL_CHECKPOINTS.md` · `z_research/predictor_training/predictor_IntPhysGenV11_PFT/Archive/RESULTS_2026-09-16.md` · `configs/training/datasets.md` · `configs/protocols/{models,datasets}.md` · `z_research/HANDOFF_2026-09-22.md` · `z_AC_training/Archive/PLANNING_AND_ROBOCASA_2026-09-20.md`

---

## 5. 다음 할 것

### 5-1. 정본 §5 표 (순서 그대로)

| 순 | 무엇 | 세우는 곳 | 비용 | 담당 | 미결을 닫는 것 (이 장의 행) |
|---|---|---|---|---|---|
| 1 | **Figure 1** — 가림 + 선반 두 패널, "후보 미래 중 어디에 가까운가" (파라미터 0) + 검증된 위치 자 보조 | P2 hook | CPU, 기존 산출물 | Claude | §2-2 d (이중 벌점 — 후보 미래 비교는 "없음 vs 뒤처짐" 을 가를 수 있는 형식이다) · §2-3 템플릿 누수 (파라미터 0 검사도 양방향 target 토큰이면 누수 대조가 필요, R2-3) · §2-2 c (가려진 미래 렌더 누수) |
| 2 | **leakage 그림** — GRASP level2 · InfLevel 세 판에 복사 기준선 (Garrido 격자, 같은 창) → 벤치마크 · 칸마다 (복사, 모델) 한 점 | P2 | GPU 소량 | Claude | 정본 4-1 · CVPR_ORAL O-8 ("IntPhys 1 하나뿐") · REVIEW_v1 AC-E4 (전 격자 copy × 4 predictor) · §2-2 h (칸마다 한 점이면 w16 · skip5 도 자동 병기) · §3-3 Benchmarks 상태 불일치 |
| 3 | **Ariel 세 팔 (prefix ep45 · full ep40 · AR) 을 v11_realistic · ledge · gravity_realistic 에** — 합격선 ★ 채우기 (`z_training/eval.sh`, AR 은 `ar_scoring`) | 목표 합격선 | GPU 소량 | Claude | 정본 §2-4 ★ 칸 셋. ⚠️ 실행 전에 AR epoch 를 정해야 한다 — 정본은 "ep40", CLAUDE.md 는 "ar ep18" 이고, ep18 = ep36 은 같은 파일이다 (§3-4) |
| 4 | **ledge · wall 다중 미래 세트** (ledge: 낙하 · 부유 · 상승 · 멈춤 / wall: 정지 · 통과 · 튕김 · 멈춤-전). 갈림은 예측 구간 안쪽 | P3 3b · Figure 1 | 렌더 (UnrealEngine) + CPU | Claude (설계) · 렌더 | 정본 4-4 · §2-1 4-4 (갈림 시점 누수 · 유효 n) · §3-1 "자 둘이 다른 답" (wall · ledge) |
| 5 | **"없는 정보" 축 한 표** — 3a (가속 · 꼭대기) 와 3b (낙하 · 벽) 를 같은 판 (위치 자 + 후보 미래 + 파라미터 0) 으로 | P3 (iii) | CPU 대부분 | Claude | §2-2 e (3a 에서 encoder 정보 — 같은 표에 z 자 · h 대조 열을 넣으면 "정보는 있는데 안 넣는다" 가 3a 에서도 서는지 보인다) · §2-2 k (RollOut 근거의 초 단위 표기) |
| 6 | **P2 일반성** — latent 예측형 하나 더 (V-JEPA 2.1 · DINO-world 류) 또는 생성형 하나를 v11 late 가림 · ledge · gravity 에 복사 기준선과 같이 | P2 · 4-2 | GPU, 모델 로더 | Claude (모델 선택은 사용자, D4) | 정본 4-2 · 리뷰 전원의 "두 번째 모델 계열" · CVPR_ORAL O-13 |
| 7 | **corner case 방어** — `OcclusionStats` 를 IntPhys 2 · GRASP · InfLevel 로 (가림 시간 × 가려진 동안 이동) | 4-3 | CPU + 추적기 | Claude | 정본 4-3 · REVIEW_v2 R3-2 · R4-3 (실영상 비율의 필터 의존성은 따로 남는다) |
| 8 | **관측 학습 predictor** — 정본 §2-4 합격선을 넘는 것. 방법 절 첫 줄 = Ariel 과의 차이 | 목표 | 학습 | **사용자** | 정본 4-7 · §2-2 f (새 run 의 학습 곡선 체크포인트를 남기면 "학습 부족" 공격에 run 안 해리로 답할 수 있다) · §2-2 l (HERA 와의 차이) · REVIEW_v1 APP-W7 (설계가 알려진 관행이라는 공격) |
| 9 | **AC subsection 실험** — (a) AC scratch vs (b) 관측 학습 predictor 위 AC post vs (c) 릴리즈 위 post × AC 데이터 10/30/100 % (정본 §2-5) — **스토리 초안 · 아직 최종 아님 (10-10)** | AC subsection | 데이터 확보 → 학습 | Claude (설계) · 사용자 (실행) | 정본 4-8 · `09b_AC_STORY.md` §3-3 결정 (데이터 · encoder). 팔 (b) 는 8 번 뒤 |
| 10 | **문헌 확인** — P1 (동물 · 적응), P2 (world model 발전 · IntPhys 2 · 생성형 물리 벤치마크 · VoE leakage 선행) | P1 · P2 | 문헌 | Claude | §4-4 전부 · CVPR_ORAL L-2 · L-3 (HERA · Garrido "no-prediction 처음") · JEPA-W7 (Joseph 가속) · 정본 4-9 (인지과학 계보) |

### 5-2. AC (9 번) 의 상태

| 순 (정본) | 무엇 | 상태 | 이유 · 기록 |
|---|---|---|---|
| 9 | AC subsection 실험 — scratch vs post × AC 데이터 10/30/100 % (정본 §2-5) | **스토리 초안 · 아직 최종 아님 (사용자 2026-10-10).** 순서는 8 (관측 학습 predictor) 뒤 — 팔 (b) 가 그 predictor 를 쓴다. 그 전에는 데이터 · 하네스 준비와 팔 (a) · (c) 만 가능 | "AC 는 post 여야 효과" 는 가설로만 쓴다. 설계 · 정할 것 · 결과별 읽기 · RoboCasa 인용 불가 사유는 `09b_AC_STORY.md` |

### 5-3. 리뷰 미결을 닫는 실험 (정본 §5 에 없음 · 전부 ★ 미실행)

"어느 쪽이어도" 열은 하네스 질문 2 (결과가 어느 쪽이어도 문장이 서는가) 에 대한 이 장의 읽기다. 원 문서의 문장이 아니다.

| 실험 | 닫는 행 | 어느 쪽이어도 | 비용 (원 문서 값) | 담당 | 원 문서 |
|---|---|---|---|---|---|
| **v2 거리 회귀 재적합** — 귀무 보정 + 궤적 군집 bootstrap + log 지수 (측도별) | §2-2 a · §3-1 지평 단위 | 거리항이 살면 P3 "2–3 칸" 을 측도를 밝혀 유지한다. 시간항이 이기면 "약 K 걸음 외삽" 으로 바꾼다. 어느 쪽이든 P3 의 "짧게 외삽" 은 선다 | CPU (기존 json · 캐시, vll5/vll6) | Claude | CVPR_ORAL P10 · R7 · S-3 |
| **걸음 수 vs 거리 판결** — training_v5 `line_x` 넓은 속도 (튜블릿당 2.8–11.4 px) | §2-2 a | 속도 폭이 4–8 배라 v3 (1.75–2.4 배) 에서 못 가른 것을 가른다 | GPU ~30 min + CPU 1 h | Claude | CVPR_ORAL O-14 · must-do 2 |
| **`copy_h_ctxonly` · 마지막으로 보인 튜블릿 복사** (IntPhys 1 · v11) | §2-2 g · §4-3 | 복사가 크게 떨어지면 H6 헤드라인 (복사 85.0) 은 철회 조건이 걸리고 P2 leakage 수치가 바뀐다. 비슷하면 정본 4-1 이 더 단단해진다 | GPU 1.5–2 h (P5) · ~40 + 33 min (DA-6) | Claude | CVPR_ORAL O-7 · DA-6 |
| **IntPhys 1 전 격자 표 + 상한 · TOST** (기존 `per_window.json`) | §2-2 h · §2-1 4-1 | 칸마다 부호가 바뀌는 것을 숨기지 않는다. "predictor 고유 몫 ≤ 7.8 pt" 꼴로 쓰면 n.s. 공격을 피한다 | CPU (기존 산출물) | Claude | CVPR_ORAL R2-12 · S-6 |
| **두 복사 기준선 맞추기** (H6 71.67 vs TrainingEffects 73.89, w16) | §3-2 | 어느 쪽이 맞든 인용 전에 하나로 정한다 | CPU | Claude | REVIEW_v1 AC-a (C6) |
| **렌더 누수를 뺀 v11 채점** — 가려진 튜블릿 · 가림막 footprint 토큰 제외 | §2-2 c | late vanish 0 / 100 이 그대로면 P3 (ii) 근거가 누수에서 자유롭다. 바뀌면 "안 보이는 것" 의 v11 수치를 다시 쓴다 | GPU ~30 min | Claude | CVPR_ORAL must-do 10 · R2-8 |
| **이중 벌점 판정 (R3)** — v11_full 파라미터 없는 readout + clip 별 지연 vs vanish-A | §2-2 d | "뒤처짐" 이 vanish-A 실패를 설명하면 "없음" 대신 "자리를 못 맞춘다" 로 쓴다 (둘 다 "맞는 미래로 놓지 않는다" 와 맞다) | 문서에 비용 없음 (CPU 위주로 보임 — 판정 보류) | Claude | CVPR_ORAL A-2 · architect R3 |
| **IntPhys 1 대 v11 조건표 (P9)** — 재등장 · 가림 위치 · 기준선 종류 | §2-2 b | IntPhys 1 재등장 +16.4 가 "문맥 가운데 가림" (정본이 이미 "처리한다" 고 한 쪽) 에 몰리면 정본 §1 의 "마지막 관측에" 좁힘과 맞는다. 문맥 끝 가림에서도 이기면 P3 (ii) 를 좁힌다 | CPU (기존 `h6g_intphys1.json` · 방향 정의 `h6g_intphys1_cell_ci.py:69–72`) | Claude | CVPR_ORAL S8 · O-10 · DA-2 |
| **encoder 가속 판독 확정** — (v0 × a) 교차 세트 + 동등성 상한 + stride 3 재검정 | §2-2 e · §1-2 | encoder 에 가속이 있으면 3a 에서도 "정보는 encoder 에 있다". 없으면 3a 는 "encoder 에도 약하다" 로 정본 (iv) 를 좁힌다 | 렌더 + GPU (문서에 비용 없음) | Claude (설계) · 렌더 | ctxenc EXPERIMENTS §2-1 · CVPR_ORAL P12 |
| **k 평평의 동등성 마진** (TOST) | §2-2 j | 마진 안이면 CLAUDE.md §5-1 ✅ 문장이 "± X pt 안에서 같다" 로 선다. 밖이면 "k 에 약하게 의존" 으로 바꾼다 | CPU (기존 `report.json`) | Claude | CVPR_ORAL S-5 |
| **p 자 먼 슬롯 분해** (슬롯 × 거리 × 3×3 질량) | §2-3 · §2-2 a | 학습셋 안 먼 슬롯이 맞으면 자의 신뢰 범위가 넓어지고, 틀리면 거리 한계가 자에도 있다 | GPU 2 장 ~15 min | Claude | CVPR_ORAL DA-5 |
| **확증 재현** — 동결한 주장 · 끝점 · readout 으로 남은 14 외형, 또는 RollOut_v2 126 궤적, 또는 IntPhys 1 train 3,751 영상 | §2-3 forking paths | 재현되면 v3 결과가 탐색 표본에만 묶이지 않는다 | GPU 1–2 h (S-8) · GPU ~1 h + 데이터 40 GB (R2-10) | Claude | CVPR_ORAL S-8 · R2-10 · REVIEW_v1 STAT-W5 |
| **분포 안 tube 마스크 대조** — 물체 경로 띠를 전 프레임 가림 | §2-1 4-6 | 분포 안에서도 띠 안 시간별 자리를 못 맞추면 "inpainter 라 당연" 공격이 약해진다 | GPU 30–60 min | Claude | CVPR_ORAL R2-9 · P7 |
| **픽셀 차분 · 무작위 초기화 ViT-H 판독 대조** (같은 ridge · 궤적 split) | §2-1 4-5 | encoder 판독이 대조보다 높으면 정본 4-5 가 Hewitt–Liang 식 공격까지 막는다 | 문서에 비용 없음 | Claude | REVIEW_v1 JEPA-E6 · STAT-exp7 |
| **문맥 끝만 / 미래 앞만 가리는 팔** | §2-3 | "마지막 관측에 안 보이면" 이 문맥 끝 가림 하나로 정해지는지 가른다 | 렌더 | Claude (설계) · 렌더 | `z_research/IntPhysGenV11_occlusion_timing_ablation/README.md` |
| **초 단위 지평 표** — 모든 predictor × stride (s1 · s2 · s4, 각자 학습 stride) | §2-2 k | 지평이 초에 묶이든 튜블릿에 묶이든 단위를 밝혀 쓰게 된다 | CPU (`cmp_predictors.json` 에 s2 · s4 있음) | Claude | REVIEW_v1 #7 · APP-W1 · JEPA-W5 |
| **학습 predictor 의 빈 장면 objectness · 그룹 B IntPhys 1 복사 · AR v11 `copy_blk`** | §4-3 · §2-3 | 학습 predictor 점수가 "출현 prior" 또는 복사로 설명되는지 가른다 | GPU 소량 | Claude | REVIEW_v1 APP-Q3 · `z_research/TrainingEffects/README.md` §5 · `z_training/ARIEL_CHECKPOINTS.md` §5-2b |

### 5-4. 문서 정리 (실험 아님 · 이 아카이브는 원 문서를 고치지 않는다)

| 무엇 | 어디 | 근거 |
|---|---|---|
| "프로토콜 4 종" vs 5 종 통일 | `configs/protocols/README.md` · CLAUDE.md §2-2 | §3-4 |
| HOWTO §1 · `v11_full` 3 단계의 삭제된 `z_scripts/` 호출 표시 | `z_research/scripts/HOWTO.md` · `configs/protocols/datasets.md` | §3-4 |
| Benchmarks README §4 표 갱신 (InfLevel 은 완료, GRASP 만 미실행) | `z_research/Benchmarks/README.md` | §3-3 |
| AR epoch 표기 통일 (ep18 / ep36 / ep38 / ep40) | `configs/protocols/models.md` · CLAUDE.md §3-1 · 정본 §5-3 | §3-4 · §4-5 |
| `REVIEW_CVPR_ORAL` 머리의 빈 후속 경로 · 대체 표시 | `auto_research/Archive/REVIEW_CVPR_ORAL_2026-09-25.md` | §3-4 |
| CLAUDE.md §5-5 "거리 한계" 문장에 "귀무 미보정 · 재적합 전" 단서 | CLAUDE.md §5-5 · `z_research/RollOutV2/Archive/PREDICTOR_DISTANCE_LIMIT_2026-09-19.md` | §2-2 a |

⚠️ 5-3 의 실험은 모두 리뷰 문서가 제안한 것이다. 리뷰 이후에 실행했다는 기록은 추출문에 없다. 실행 전에 정본 §6 하네스 질문 (특히 1 · 2 · 5) 을 거친다.
⚠️ 5-3 의 "어느 쪽이어도" 칸은 이 장의 해석이다 — 원 문서 등급이 붙지 않는다.

---

## 6. 2026-10-21 갱신 — 닫힌 것 · 새로 열린 것 (자세한 것은 `13_SINCE_2026-10-10.md` §10 · §12)

| 항목 | 10-10 상태 | 10-21 상태 |
|---|---|---|
| D5 설계 후보 선택 · 학습 실행 | 후보 표 (09-25 기준), 3b 비어 있음 | **후보 교체** (10-13): 문맥 끝 dropout ① · 공간 대조 ② 가 1 순위, 어디 상태 분리 ③ 는 그다음; 상태/렌더 분리는 이미 실패한 형태. 학습 팔 A0–A3 설계됨, **실행은 사용자 (★)**. 3b 는 여전히 비어 있음 (자리를 알려 줘도 선반 17/144) |
| 합격선 ★ (Ariel 의 중력 꼭대기 · 선반) | 미측정 | AR 40 epoch: 내려가는 중 진짜 포물선 0/24 · 선반 9.7/0.0 — **미달**. 단 기준 팔이 바뀌어 (AR 40 · vast 70) 합격선 Ariel 열을 같은 판으로 다시 채워야 한다 (**새 미결**) |
| §2-2 e "encoder 정보" 단서 | 가속 판독 약함 | **새 단서**: "z 자 가림 뒤 미래 위치 5–10 px" 에 가림막 교락 (13 장 §4-4) → 07 장 · README §3 · 정본 §2-3 (iv) 갱신 필요 |
| §3-1 "AR vanish = presence-without-placement" | 긴장 | VOBench 에서 **재확인** (late 영속 100 인데 물체 자리에서 0 %) — 닫힘 |
| §4-1 확인 안 한 수치 | 목록 | 추가: IntPhys 1 표준 sliding 의 **렌더 잡음 기여** (미측정, 1 차 발견) · 2D post-FT 팔 A 의 IntPhys 2 · VOBench 복사 기준선 (못 냄) |
| D7 AC | 스토리 초안 · 최종 아님 | **변화 없음** (AC 결과 문서 0; Push-T · Wall 궤적 학습셋만 등록) |
| 문서 운영 | — | **새 어긋남**: 정본 · 결과 로그 머리 blockquote 가 철회 전 문장 (①) · 09 장 AR 90.56 (18 epoch) vs 92.8 (40 epoch) (②) · 합격선 Ariel 열 (③) · 07 장 5–10 px 단서 (④) · 합성 post-FT 모델이 평가 행렬에 (⑤) · 전이 행렬 문서 제목 날짜 |
| 새 미결 (실험) | — | anchor 의 학습 없는 교차검증 · post-FT 에 anchor · 릴리즈 "거리" vs AR "블록 순서" 같은 방법 · 상태 head 대조군 셋 · VOBench 문맥 앞 가림에서 모델 < 복사 원인 · late 복사 61 % · 창 끝 효과 기제 · vast + 중력 post-FT 가 선반을 못 푸는 이유 |

### 출처 (§6)
13 장 §10 · §12 와 그 부록 B.

## 출처

**정본 · 운영 규칙**
- `auto_research/paper/PAPER_STORY_2026-10-08.md` (§2-1 · §2-2 · §2-3 · §2-4 · §2-5 · §3 · §4 · §5 · §6 · §재현)
- `CLAUDE.md` (§0 · §1-2 · §1-7 · §2-0 · §2-2 · §2-7 · §3-1 · §5-1 · §5-3 · §5-5 · §6 · §7-4 · §8-1 · §8-5 · §12)
- `auto_research/README.md`

**리뷰 문서**
- `auto_research/Archive/REVIEW_CVPR_ORAL_2026-09-25.md`
- `auto_research/paper/REVIEW_v1_2026-09-25.md`
- `auto_research/paper/REVIEW_v2_2026-09-26.md`

**auto_research 분석**
- `auto_research/Archive/H6_INTPHYS1_WHAT_THE_SCORE_MEASURES_2026-09-24.md`
- `auto_research/Archive/H23_PREDICTOR_LAYERS_TRAJECTORY_2026-09-24.md`
- `auto_research/Archive/SC1_REACH_NOT_HORIZON_2026-09-25.md`
- `auto_research/Archive/P2_STRIDE_SWEEP_HORIZON_UNIT_2026-09-25.md`
- `auto_research/Archive/I_SCENE_CAUSAL_BOTTLENECK_2026-09-25.md`
- `auto_research/Archive/VERIFY_ISCENE2_2026-09-26.md`
- `auto_research/Archive/VERIFY_0925_2026-09-25.md`
- `auto_research/Archive/VERIFY_0925_R2_2026-09-25.md`
- `auto_research/Archive/TRAINED_PREDICTORS_SIGNATURES_2026-09-25.md`
- `auto_research/Archive/AR_PREDICTOR_2026-09-27.md`
- `auto_research/Archive/PREDICTOR_SHARE_2026-09-29.md`
- `auto_research/Archive/ENCODER_MOTION_READOUT_2026-09-29.md`
- `auto_research/Archive/AR40_READOUT_2026-09-30.md`
- `auto_research/Archive/VERIFY_AR40RD_2026-09-30.md`
- `auto_research/Archive/FUTURE_FROM_CONTEXT_2026-10-02.md`
- `auto_research/Archive/COPY_VS_POSITION_2026-10-07.md`
- `auto_research/Archive/INTPHYS2_PERMANENCE_AUDIT_2026-09-26.md`
- `auto_research/Archive/LITERATURE_POSITIONING_2026-09-25.md`
- `auto_research/Archive/LITERATURE_STATE_WORLD_MODELS_2026-09-26.md`
- `auto_research/Archive/LITERATURE_PREDICTOR_ROLE_2026-09-29.md`
- `auto_research/paper/DESIGN_CHOICES_2026-09-25.md`
- `auto_research/paper/RETHINK_PREDICTOR_ROLE_2026-09-29.md`

**z_research**
- `z_research/RollOutV2/Archive/PREDICTOR_DISTANCE_LIMIT_2026-09-19.md` · `z_research/RollOutV2/README.md`
- `z_research/RollOutV3/Archive/IDENTITY_R8_2026-09-26.md` · `Archive/RESULTS_2026-09-25.md` · `Archive/CONTEXT_TO_FUTURE_2026-09-26.md`
- `z_research/RollOutV3/_superseded/RERUN_2026-09-24.md` · `_superseded/README_2026-09-25.md` · `exp_results/windows/BEHAVIOR.md`
- `z_research/TrainingEffects/README.md` · `MODELS.md` · `v3_motion/MOTION.md` · `v11_presence/PRESENCE.md` · `ip1_copy/IP1_COPY.md` · `scores/SCORES.md`
- `z_research/IntPhysGenV11/Archive/surprising_score/RESULTS_2026-08-30.md` · `Archive/PAPER_STORY_2026-09-06.md` · `figures/knockout/README.md`
- `z_research/IntPhysGenV11_occlusion_timing_ablation/README.md`
- `z_research/context_encoder_analysis/Archive/CLAIM_NO_STATE_EVOLUTION_2026-09-15.md` · `Archive/EXPERIMENTS_2026-09-16.md` · `Archive/STATE_EVOLUTION_CAPABILITIES_2026-09-19.md`
- `z_research/context_encoder_analysis/exp_results/ctxenc_boundary_kinematics/RESULTS.md` · `ctxenc_time_alignment_gated/RESULTS.md` · `ctxenc_subspace_v2/RESULTS.md`
- `z_research/v11_realistic/README.md` · `figures/token_surprise/README.md`
- `z_research/Benchmarks/README.md` · `PROTOCOLS.md` · `exp_results/intphys2/bench_intphys2_main_ariel_bc_ep30_w16/column_best.md`
- `z_research/predictor_training/predictor_IntPhysGenV11_PFT/Archive/RESULTS_2026-09-16.md`
- `z_research/OVERVIEW_2026-09-22.md` · `z_research/HANDOFF_2026-09-22.md` · `z_research/scripts/HOWTO.md`

**그 밖**
- `z_world_model_analysis/WHY_INTPHYS_WORKS_2026-08-20.md` · `VERIFICATION_2026-08-20.md` · `MECHANISM_2026-08-20.md`
- `analysis/intphys2/ANALYSIS_NOTES.md` · `analysis/intphys2/configs/_superseded/README.md` · `analysis/intphys2/PROTOCOL_CHECK_2026-09-14.md`
- `analysis/intphys1/HANDOFF_SPEC.md` · `analysis/intphys1/PAPER_VS_OURS.md`
- `z_training/ARIEL_CHECKPOINTS.md` · `z_training/runs/_matrix/MATRIX.md` · `configs/training/datasets.md`
- `configs/protocols/README.md` · `configs/protocols/models.md` · `configs/protocols/datasets.md` · `cvpr/README.md`
- `z_AC_training/README.md` · `z_AC_training/Archive/PLANNING_AND_ROBOCASA_2026-09-20.md`
- 원재료 추출문: `scratchpad/archive10/extracts/G04b_reviews.md` (주) · G01b · G02a · G02b · G03 · G03b · G04a · G06 · G07 · G08b · G09 · G10 · G10b · G10c · G11 · G12 · G13 의 (c) 묶음 요약
