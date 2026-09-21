# V-JEPA 2-AC 평가(planning) 프로토콜 + Jongseo RoboCasa 실험 검토 (2026-09-20)

> 이 문서는 대화에서 확인한 내용을 대충 모은 메모다. 학습·추출은 하나도 돌리지 않았다.
> 근거는 vjepa2 릴리즈 코드, V-JEPA 2 논문 (arXiv 2506.09985), jepa-wms 코드·README, Jongseo 레포 산출물이다.
> jepa-wms 논문 (arXiv 2512.24497) 은 로컬 PDF 가 없어서 WebFetch 요약으로 읽었고 **원문과 대조하지 않았다.**
> 그림으로 된 설명: https://claude.ai/artifact/UuAPq8N4U14gg68hrnnu8h (비공개, 소유자만 열람)

**결론 3줄**
1. AC 평가 = **closed-loop planning**이다. 실제 사진 1장 + 센서 자세로 시작해 CEM 이 후보 action 800개를 predictor 에 넣고,
   상상한 **다음 장면 1개 (horizon 1)** 가 **최종 목표 사진**과 가장 가까운 action 을 골라 실행한다. step 마다 새 사진으로 처음부터 다시 한다.
2. Jongseo 의 RoboCasa 실험 결론 "V-JEPA 2-AC 는 action 을 거의 안 읽는다" (E4 · E9) 는 **인용 불가**다.
   이유는 버그가 알려진 공식 체크포인트, 맞지 않는 proprio 입력, zero-shot 도메인 세 가지다 (§4).
3. horizon 1 planning 에서 action 이 예측에 닿는 통로는 action 토큰 하나뿐이다. 그래서 **1-step action sweep (Jongseo E4 형태)** 이
   "planning 이 성립하는가" 를 가장 직접 재는 실험이다.

**미결 1줄:** RoboCasa + jepa-wms `vjepa2_ac_droid` (고친 모델, proprio 없음) 로 E4 를 다시 잴지. README §6 의 D 안과 이어진다.

---

## 1. predictor 입력과 출력

| 항목 | 내용 | 출처 |
|---|---|---|
| state s | 7차원 = end-effector xyz (로봇 base 기준) + Euler 3 + gripper 1. 로봇이 매 프레임 기록하는 자세 | 논문 §3.1 |
| action a | 7차원 = s_{k+1} − s_k (위치 차 · 상대 회전 · gripper 차). 관절 명령이 아니라 손끝 변위 | 논문 §3.1, `droid.py` `poses_to_diffs` |
| 토큰화 | a, s 는 각각 `Linear(7→1024)` 로 토큰 1개. 장면 z 는 `Linear(1408→1024)` 로 256 토큰 | `ac_predictor.py:54-55, 147-157` |
| 입력 배열 | 프레임마다 `[a_k, s_k, z_k 256칸]` = 258 토큰, block-causal (프레임 블록 t 는 블록 ≤ t 만 봄) | `ac_predictor.py`, `modules.py` |
| 출력 | 각 프레임 자리 → **다음 프레임 z** 예측. a·s 자리 출력은 버린다 (`ac_predictor.py:190`). **action 을 출력하지 않는다** | 코드 |

predictor = f(현재 장면, 현재 자세, 움직임) → 다음 장면. action 은 입력("이렇게 움직이면?")이고, action 을 고르는 건 바깥의 CEM 이다.

## 2. planning 프로토콜 (논문 로봇 실험 기준)

```
로봇 step k 마다:
  z_k = LN(encoder(현재 카메라 사진))       # 실제 관측, context 1장
  s_k = 로봇 센서 자세                        # 실제 관측
  z_g = LN(encoder(목표 사진))                # 사람이 준 최종 목표 (다음 프레임이 아님)
  분포 초기화: 평균 0, 표준편차 = maxnorm (xyz), 1 (gripper)
  반복 10번:
      후보 action 800개를 분포에서 뽑음 (Δx, Δy, Δz, gripper; 회전 3개는 0 고정, xyz clip)
      predictor 1회 forward (batch 800): 후보 i 의 입력 = [a_i, s_k, z_k] → ẑ_{k+1}^(i)
      점수_i = mean |ẑ_{k+1}^(i) − z_g|   (256 × 1408 전체 평균)
      점수 낮은 10개의 action 으로 평균·표준편차 갱신 (새 값 0.85 + 이전 값 0.15)
  a* = 분포 평균 (1등 후보 아님), gripper 는 |값| < 0.25 면 0
  로봇이 a* 실행 (blocking: 끝날 때까지 대기) → 새 사진 찍고 다음 step
```

| 설정 | 논문 로봇 실험 | 코드 기본값 (`world_model_wrapper.py`) |
|---|---|---|
| horizon | **1** (부록 B.2) | 2 |
| 후보 수 | **800** (부록 B.2, §4 표) | 400 |
| CEM 반복 · top-k | 10 · 10 | 10 · 10 |
| action 크기 | L1-ball 0.075 (한 번에 최대 약 13 cm, §4) | clip ±0.05 |
| 한 번 움직일 때 | predictor forward 10회 (batch 800) = 예측 8,000개, **계산 시간 약 16초** | forward 20회 |

- **16초는 동작 시간이 아니라 action 하나를 고르는 계산 시간**이다. 로봇은 "생각 16초 → 몇 cm 이동 → 멈춤" 을 반복한다.
- **batch**: 후보 800개는 서로 다른 입력 800개다 (action 토큰만 다르고 s_k, z_k 는 복사). 한 텐서로 쌓아 같은 가중치로 동시에 계산한다.
  attention 은 후보마다 따로 (800, 16, 258, 258) 라 후보끼리 섞이지 않고, 결과는 800번 따로 돌린 것과 같다.
  반복은 순차다. 다음 반복의 후보가 이전 반복 점수로 갱신한 분포에서 나오기 때문이다.
- **planning 중에 GT 미래 프레임은 없다.** 비교 대상은 최종 목표 사진이다. 실행 후 찍힌 실제 사진은 채점에 쓰이지 않고 다음 step 의 입력이 된다.
- **greedy 하다.** 매 step 에서 "다음 장면이 목표에 가장 가까워지는 동작" 만 고른다. 그래서 pick-and-place 는 중간 목표 사진 2장을 주고
  4 → 10 → 4 step 순서로 목표를 바꾼다 (부록 B.2). 긴 horizon 은 논문 한계 절에 적혀 있다.
- **평가 지표는 latent L1 이 아니다.** 실제 손끝과 목표 사이 거리 (reach, Fig. 8) 와 과제 성공률 (grasp · pick-and-place, §4 표) 로 잰다.
- **"낮은 점수 = 좋은 action" 은 가정이다.** 논문은 Fig. 9 로 확인했다. Δx·Δy 를 훑으면 에너지 최저점이 (0, −0.05), 정답 action 이 (0, −0.1) 이다.
  에너지 색 막대는 약 0.41–0.44 다 (pdftotext 눈금을 읽은 **대략값**).
- 이 점수는 우리 surprise `|p − h|₁` 와 같은 식이고, 비교 대상 (목표 사진 vs 실제 미래) 만 다르다.

### 값의 출처

| 값 | 학습 (DROID) | planning | Jongseo E3 · E4 · E9 |
|---|---|---|---|
| 장면 z | 매 프레임 실제 | 현재 사진 1장만 실제. horizon ≥ 2 면 그 뒤는 상상한 ẑ | 실제 16장 (+ rollout 부분은 상상) |
| action | 기록된 실제 이동량 | **CEM 후보** (정답 없음) | 기록값. 비교용으로 0 · 섞기 · 무작위 · 다른 시연 |
| state | 기록된 실제 자세 | 현재만 센서값. horizon ≥ 2 면 `compute_new_pose(s, a)` 로 계산 | 기록값 (E3 · E4), 0 (E9) |
| 출력 비교 대상 | 실제 다음 프레임 → loss | 목표 사진 → 점수 | 실제 다음 프레임 → 오차 |

## 3. Jongseo RoboCasa (D2) 실험 요약

레포: `/data/jongseo/project/world-model/auto-research-causal-attention` (`results/data_D2/` 는 결과 폴더다. 생성 스크립트 사본 + 샘플 10장).

| 항목 | 내용 |
|---|---|
| 데이터 | HF `facebook/jepa-wms` `robocasa/combine_all_im256.hdf5` (3.05 GB, NFS 사본 `/data/jongseo/project/world-model/datasets/jepa-wms/robocasa/`). PnPCounterTop 과제 1개, Panda 팔, **시연 14개**, 5,628 step |
| 전처리 | `robot0_leftview` 256 px, 5 step 간격 (4 fps). state = DROID 형식 7차원, action = state 차분 (hdf5 의 12차원 OSC 명령은 안 씀). 변환 캐시는 vll5 에 없다 |
| 가림막 | 문맥 16장 중 **마지막 L장** (L ∈ {0, 2, 4, 8, 16}) 에 96 px 회색 정사각형. 위치는 고정이고 중심은 그 구간 손끝 투영 픽셀의 평균. target 은 안 가림 |
| 모델 | P-MASK (V-JEPA 2 ViT-g 사전학습 predictor), P-AC (**vjepa2 공식** `vjepa2-ac-vitg.pt`), P-DINO (DINO-WM droid) |
| probe | 256 토큰 평균 위에 ridge / logistic, 시연 단위 8/3/3 분할, seed 5개 |

| 실험 | 결과 |
|---|---|
| E3 (가림 길이) | P-MASK: predictor − copy L1 이 L 과 무관하게 −0.05 ~ −0.08. 손끝 xyz R² 는 L=16 에서도 0.64–0.72 (ceiling 0.756). P-AC: L=0 에서 copy 0.227 < predictor 0.286. 손끝 R² 는 0.58 → 0.14 (std 0.1–0.27 라 copy 와 차이 판정 불가) |
| E4 (1-step action 조건, P-AC) | true / zero / shuffled / random 의 L1 이 0.2849–0.2864 (window 간 std 0.05) |
| E9 (8-step rollout) | V-JEPA 2-AC: true / other / zero 차이가 오차의 0.06%, 승률 46–54%. DINO-WM: true < other 59–64%. 두 모델 모두 대부분 k 에서 copy 가 이김 |

## 4. 인용하면 안 되는 이유 (코드·문헌에서 확인)

1. **공식 AC 체크포인트에 손실 버그가 있다.** jepa-wms `hubconf.py:138` 에 "V-JEPA-2-AC OSS model from https://github.com/facebookresearch/vjepa2 with bug in loss (see JEPA-WMs paper's appendix)" 라고 적혀 있다.
   부록 (WebFetch 요약) 에 따르면 여러 step rollout 손실이 다음 step 에 잘못된 embedding 을 쓴다. 이를 고쳐 다시 학습한 게 `vjepa2_ac_droid` (proprio 없음) 다.
   우리 README §4 의 "ẑ_2 가 jloss · sloss 에 두 번 들어간다" 와 같은 버그인지는 **미확인**이다.
2. **proprio 가 맞지 않는다.** jepa-wms 논문 (요약): "the proprioceptive space is not aligned between DROID and Robocasa, making models using proprioception irrelevant for zero-shot transfer".
   Jongseo 는 proprio 를 쓰는 공식 모델에 RoboCasa **world 좌표** state (예: y = −2.617 m) 를 넣었다. E9 의 state 0 도 이 모델에게는 분포 밖 입력이다.
   - Jongseo 문서의 "DROID 는 camera-frame state 로 학습" 은 틀렸다. 릴리즈 `droid-256px-8f.yaml` 에 `camera_frame` 이 없어서 기본값 False, 즉 base 좌표다.
   - E3 표 머리말의 "base-frame pose diffs" 는 코드 (`d2_data.load_demo` 의 `poses_to_diffs(proprio)`, world 좌표) 와 다르다.
3. **RoboCasa 는 zero-shot 도메인이다.** V-JEPA 2-AC 는 DROID 만으로 학습했다 (논문 §3.1). jepa-wms 의 DROID 모델도 전부 `datasets: [DROID]` 이다 ("We do not finetune the DROID models on Robocasa trajectories").
   Jongseo E9 주석의 "RoboCasa is in the training set of DINO-WM" 은 틀렸다.
   참고로 jepa-wms 논문 Table 2 의 V-JEPA 2-AC RoboCasa planning 성공률은 Reach 16.2 ± 8.3%, Place 33.1 ± 7.2% 다.
4. 그 밖의 한계
   - 가림막이 손끝·병만 덮고 팔은 보인다. 그래서 E3 의 P-MASK R² 가 안 떨어지는 게 기억인지 팔을 본 추론인지 구분이 안 된다.
   - 시연이 14개라 test 가 3개뿐이다.
   - 4 fps 1-step 은 copy 에 유리하다. 마지막 문맥 프레임의 GT 손끝으로 target 을 맞추면 R² 가 0.985 다.
   - "가려졌는가" probe 가 99–100% 인 건 회색 사각형을 검출한 것이다.

## 5. 우리 연구에 닿는 점

- **AC planning 은 closed-loop 다.** 매 step 실제 관측으로 다시 시작하므로 **1-step 전이**만 시험하고, 관측 없이 상태를 끌고 가는 **지속 · 영속**은 거의 시험하지 않는다
  (state evolution 네 능력, `z_research/context_encoder_analysis/Archive/STATE_EVOLUTION_CAPABILITIES_2026-09-19.md`).
  → "AC 로 planning 이 된다" 에서 "긴 시간 state evolution 을 한다" 로 넘어갈 수 없다.
  우리 surprise 프로토콜 (문맥 16장 → 관측 없이 미래 16장) 과 Jongseo E9 는 open-loop 다.
- **가장 싼 확인 실험:** RoboCasa + `vjepa2_ac_droid` (proprio 없음) 로 1-step action sweep 을 한다. 목표 = 실제 다음 프레임으로 두고 Δx·Δy 를 훑어서 Fig. 9 식 에너지 지형을 그린다.
  - 고친 모델도 평평하면 "공식 체크포인트/proprio 탓이 아니다" 가 된다.
  - 기울기가 있으면 Jongseo 결론은 체크포인트/입력 문제였다.
  - 필요한 것: 체크포인트 다운로드 (jepa-wms 모델 repo, Jongseo 로그상 접근 승인 불필요), hdf5 3 GB 로컬 복사. **`/data2` 가 99% 라 자리부터 확인해야 한다.**
  - 우리 논문에서의 위치는 비교군 / discussion 이다 (주 모델은 action 없는 V-JEPA 2 predictor). 주장의 중심에 둘 거면 실로봇 데이터 (jepa-wms `franka_custom`) 까지 가야 한다.

---

## 재현

```bash
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
# planning 코드 (CEM, horizon, 후보 수, 최종 action = 평균)
sed -n 1,80p notebooks/utils/world_model_wrapper.py
grep -n -A40 "^def cem" notebooks/utils/mpc_utils.py
# predictor 입력 배열 / 출력에서 a·s 자리 버림
sed -n 136,190p src/models/ac_predictor.py
# DROID state 좌표계: camera_frame 기본 False, 릴리즈 config 에 키 없음
grep -n camera_frame app/vjepa_droid/droid.py app/vjepa_droid/train.py configs/train/vitg16/droid-256px-8f.yaml
# 논문: §3.2 planning, §4 표 (800 samples, horizon 1, 16 sec), 부록 B.2
pdftotext 'V-JEPA 2- Self-Supervised Video Models Enable Understanding, Prediction and Planning.pdf' - \
  | grep -n -i -E "planning horizon of 1|800 samples|16 seconds|minimum near the ground-truth"

J=/data/jongseo/project/world-model/jepa-wms
grep -n -E "vjepa2_ac_(oss|droid)|bug in loss" $J/hubconf.py
grep -n -A2 "datasets:" $J/configs/vjepa_wm/droid_final_sweep/*.yaml | grep -A1 datasets
grep -n robot0_eef_pos $J/app/plan_common/datasets/robocasa_dset.py

R=/data/jongseo/project/world-model/auto-research-causal-attention
cat $R/results/E3/table.md $R/results/E4/table.md $R/results/E9/table.md
sed -n 1,98p $R/code/d2_data.py     # 가림막 · world 좌표 state / action
```
