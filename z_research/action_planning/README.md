# Action planning (Push-T) — AR predictor 에 action 을 붙여 world-model planning 으로 평가 (2026-10-09 개설)

> **묻는 것 (사용자 결정 2026-10-09):** 자연 영상 (SSv2 + K400) 으로 학습한 AR predictor (`nat_ar_scratch`, AC 판) 위에 Push-T 로
> **post-FT** 한 것과, Push-T 로 **scratch** 학습한 것 중 어느 쪽이 planning 을 더 잘 하는가. 같은 encoder (frozen ViT-H) 위에서.
> 비교 대상 수치는 DINO-WM (Zhou et al. 2024, arXiv 2411.04983) Push-T 성공률 0.90 (Table 1) — 단 encoder 가 달라 (DINOv2 ViT-S/B 224px) **참고치**다.

## 데이터·환경
`${WORLD_ROOT}/action_planning/dino_wm/` ← OSF (DINO-WM 공개, view-only 링크는 `benchmarks/dino_wm` README). zip 은 해제 뒤 삭제.
| 세트 | 크기 | 내용 |
|---|---|---|
| `pusht_noise/` | 7.0 G | train 18,685 ep (2.34 M 스텝) / val 21 ep. mp4 224px + states (agent xy, block xy, angle) + rel_actions (목표 Δxy, px) |
| `point_maze/` | 29 G | 2,000 ep × 100 스텝 (.pth 관측) — 아직 안 쓴다 |
| `wall_single/` | 55 G | — 아직 안 쓴다 |

시뮬레이터 = `benchmarks/dino_wm/env/pusht` (pymunk, Diffusion Policy 환경). 의존성은 `vjepa2` env 에 설치 (pymunk 6.8 · pygame 2.5.2 · shapely 2.2 · gym 0.23.1 · hydra 1.2 · omegaconf 2.3 · scikit-image).
⚠️ `gym==0.23.1` 설치가 numpy 를 1.26 으로 내렸다 → **numpy 2.3.5 로 복구함** (opencv 5 · contourpy 가 numpy≥2 요구). `env/__init__.py` 가 pointmaze (mujoco_py) 를 import 하므로
Push-T 만 쓸 때는 `sys.modules['env']` 를 빈 패키지로 넣고 `env.pusht.pusht_wrapper` 를 직접 import 한다 (`plan_pusht.py` 참고).
헤드리스: `SDL_VIDEODRIVER=dummy`. action 규약: env.step 은 `action × action_scale(100)` 뒤 `relative` 면 agent 위치에 더해 PD 목표로 삼는다.
데이터의 `rel_actions` = 그 목표 Δxy (px) 라 **실제 이동의 약 40 %** 와 상관 0.994 (스텝 t 의 action → 관측 t+1). env 에 넣을 때는 px/100.

## 모델 (config `configs/training/pusht_ar_{scratch,postft}.yaml`)
* `natural_ar` (AC 판: 블록별 LN(h) 입력·타깃, 블록 인과, TF + rollout) + **action 토큰 1 개/블록** (`src/models/rollout_predictor.py`,
  `action_dim=10` = frameskip 5 × Δxy, Linear → 384, 시간 RoPE 만; `ACRoPEAttention(action_tokens=1)` 경로). 배치 `[a_t ; h_t(256)]`, a_t 는 t→t+1 의 action.
* 관측 1 장 → 2 번 복제 → tubelet (V-JEPA 2-AC 와 같다). 224 → 256. aug 없음. 창 = 관측 8 장 (40 sim 스텝), 문맥 C ∈ {1,2,3,4}, K = 8 − C.
* `action_dim=0` 이면 자연 영상 AR 팔과 파라미터·연산이 비트 단위로 같다 → post-FT 는 `model.init_predictor` 로 `nat_ar_scratch/latest.pt` 를 strict=False 로 넣고 **`action_embed.*` 두 개만** 새로 (그 외 빠지면 에러).

## 공정 비교 체크리스트 (둘 다 돌리기 전에 고정)
| 항목 | post-FT | scratch | 같은가 |
|---|---|---|---|
| 데이터·frameskip·창·C 분포·seed | pusht_noise, 5, 8 블록, {1,2,3,4}, 0 | 같음 | ✅ |
| batch / step | 36 × 8 GPU = 288 창, 100 epoch × 64 = 6,400 step (2026-10-10 사용자 결정, 28 에서 1.29 배) | 같음 | ✅ |
| predictor 구조 | 384 × 12 × 12 + action_embed | 같음 | ✅ |
| 초기값 | nat_ar_scratch e40 (자연 영상) | 무작위 | **조작 변수** |
| lr | 5e-5 | 3e-4 | ❌ 혼입 → 선택 팔 `pusht_ar_postft` + `SET=optimization.lr=3e-4` (NAME=pusht_ar_postft_lr3e-4) |
| warmup / wd / cosine | 5 epoch / 0.04 / → 1e-6 | 같음 | ✅ |

## sanity (학습 전·중·후)
1. **코드**: `z_training/tests/ar_action_cpu_test.py` — 인과성 Δ 0 · action 이 그 블록부터만 영향 · rollout = TF 첫 스텝 · split = dense · ckpt 경로 동일 · post-FT 로드 시 빠지는 키 = action_embed 만. (2026-10-09 통과)
2. **데이터**: 프레임 복제 확인, action 정규화 (train 평균/표준편차, val 도 train 통계), action→다음 관측 정렬 (corr 0.994).
3. **in-loop val** (`val_pusht`, val 21 ep × 시작점 4 = 77 창, C=3 K=5): TF·rollout L1 과 **기준선 copy (이전 블록) · copy_last (문맥 마지막) · zero**. 모델이 copy 를 못 이기면 action 을 안 쓰는 것.
4. **action 민감도** (학습 후): 같은 문맥에 (a) GT action (b) action 0 (c) action 뒤섞기 로 rollout → latent 오차가 (a) < (b),(c) 여야 한다. 아니면 planning 성공률은 해석 불가 (RoboCasa 사례, `z_AC_training/README.md` §5-1).
5. **open-loop 위치**: rollout latent 에서 agent·block 위치를 읽는 자 (RollOutV2 식 attentive readout, train 으로 학습·val 로 검증) → GT action 으로 굴렸을 때 위치 오차가 스텝에 따라 어떻게 커지나. 자 검증 없이 predictor 의 세계를 말하지 않는다 (CLAUDE.md §12).
6. **planning** (DINO-WM 프로토콜 그대로): CEM horizon 5 · 300 샘플 · top-30 · 30 회, MPC 5 action 실행 후 재계획, n_evals 50, goal_H 5, `goal_source=dset`, seed 99. 성공 = 위치 < 20 px ∧ 각도 < 20°.
   objective 는 visual latent MSE 만 (우리는 proprio 예측이 없다 → alpha 0; DINO-WM 은 alpha 1). **이 차이를 적는다.**
   대조: (i) 무작위 action, (ii) GT action 재생 (상한), (iii) 두 팔.

## 실행 기록 (2026-10-10)
* **OOM → KV 캐시.** 첫 기동 (batch 28, checkpointing 끔) 이 OOM: rollout 이 predictor 를 최대 8 번 순차로 돌리며 활성값을 전부 들고 있다
  (C4K4 에서 40.5 G, C1K7 에서 48 G 초과). checkpointing 으로 피했지만 5.5 s/step·util 30 %. → `rollout_kv` (vast 브랜치 `rollout_state_kv` 를
  ar 모드·action 토큰으로 확장): 문맥을 block-causal 로 prefill, 걸음마다 새 블록 (1 + 256 토큰) 만 계산. **값·grad 가 전체 재계산판과 같다**
  (`z_training/tests/ar_kv_cpu_test.py`: Δ 값 ≤ 2e-6, Δ grad ≤ 4e-8, action 있음/없음, C=1/3). batch 36 에서 3.9 s/step · 34.7 G · util 100 % → 팔당 약 6.8 h.
* **env 대조 (모델 무관, n_evals 50, goal_H 5, dset goal)**: GT action 재생 **1.00** (state_dist 0 — action 변환 왕복이 정확하다) /
  zero **0.00** (171 px) / random **0.00** (199 px). 목표가 사소하지 않다.
* **action 민감도 (post-FT epoch 5, val 77 창, C3K5)**: rollout L1 평균 gt 0.494 < shuffle 0.497 < zero 0.508. 걸음이 갈수록 벌어진다 (zero−gt 0.008 → 0.020).
  아직 약하다 — 학습 끝에 다시 잰다 (planning 해석의 전제).
* planning smoke (4 evals, CEM 32 샘플 × 3 회, epoch 2): 0/4 — 배관 점검용, 수치 아님.
* 학습 뒤 평가는 `run_eval_after_train.sh` (tmux `eval_pusht`) 가 두 팔 `done` 을 기다렸다 자동으로 돈다. 모니터 `watch -c -n 2 bash z_training/monitor_pusht.sh`.
* 의존성 추가 정정: hydra-core 1.2.0 은 Python 3.12 dataclass 오류 → **1.3.2**.

## 결과 (2026-10-10, 두 팔 e100, DINO-WM 프로토콜: n_evals 50 · goal_H 5 · dset goal · seed 99 · CEM 300×30 · MPC 5 회)
| 팔 | 성공률 | action 민감도 (val rollout L1: gt / shuffle / zero) | val rollout L1 (C3K5) |
|---|---|---|---|
| post-FT (nat_ar_scratch e40 → lr 5e-5) | **0.60** (30/50) | 0.461 / 0.495 / 0.500 | 0.452 |
| scratch (lr 3e-4) | **0.60** (30/50) | 0.455 / 0.495 / 0.502 | 0.450 (e100 직전 기준) |
| 대조: GT action 재생 / zero / random | 1.00 / 0.00 / 0.00 | — | — |
* 같은 목표끼리 짝: 둘 다 성공 26, post-FT 만 4, scratch 만 4, 둘 다 실패 16 → McNemar 정확 p = 1.00. **두 팔은 구분되지 않는다.**
* 두 팔 모두 action 을 쓴다 (gt < shuffle ≈ zero, 걸음마다 벌어짐) → planning 수치는 해석 가능하다.
* 읽는 법: "데이터가 넉넉한 Push-T (2.3 M 스텝, 100 epoch) 에서 lr 5e-5 post-FT 는 scratch 보다 낫지 않았다". **못 하는 주장**: "자연 영상 사전학습이 쓸모없다" — lr 혼입 (5e-5 대 3e-4) 과 데이터 충분 조건이 남는다.
  post-FT 는 초반 val 에서 앞섰다 (epoch 10: 0.476 대 0.493) 가 epoch 50 부근에서 따라잡혔다.
* 참고: DINO-WM Push-T 0.90 (Table 1) — encoder (DINOv2 224px) · objective (proprio 포함) · MPC 상한 (없음) 이 달라 직접 비교 불가. 50 개 표준오차 약 ±0.07.
* `state_dist` 는 7 차원 (속도 포함) 노름이라 실패 목표 비교에는 쓰지 않는다.
* 다음 후보: (1) post-FT lr 3e-4 (lr 혼입 제거) (2) 저데이터 (train 10 %) 두 팔 (전이가 드러나는 조건) (3) seed 를 바꿔 50 개씩 더 (해상도).

## 산출물
`z_training/runs/pusht_ar_{scratch,postft}/` (ckpt·metrics.jsonl), planning 결과 `z_research/action_planning/exp_results/`. 분석 문서는 `Archive/TOPIC_YYYY-MM-DD.md`.

## 재현
```bash
GPUS=8 NAME=pusht_ar_scratch bash z_training/sbatch.sh pusht_ar_scratch
GPUS=8 NAME=pusht_ar_postft  bash z_training/sbatch.sh pusht_ar_postft
python z_training/tests/ar_action_cpu_test.py                    # CPU 정합성
# planning (학습 후): python z_research/action_planning/plan_pusht.py --run pusht_ar_postft --ckpt latest.pt   (작성 중)
```
