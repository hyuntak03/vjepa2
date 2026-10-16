# kind=ar multistart — 설정 결정 (2026-10-14)

> 대상 config: `configs/training/natural_ar_ms.yaml` (← `natural_ar_vw_state` ← `natural_ar_vw` ← `natural_ctx_state_vw`).
> 상태: **구현 · 단위 검증 완료, 학습 미실행.** 토큰 예산 (지금 950 은 자리표시) 은 EK100 평가가 끝난 뒤 (10-15 오전) 메모리를 재서 정한다.
> 모든 수치는 이 세션에서 산출물 (`summary.json` · `tables.json` · `train.log` · 로더 계획 재계산) 로 확인했다.

---

## 0. 한 줄 요약

kind=ar predictor 는 **문맥의 마지막 1–2 블록만 실질적으로 쓰고**, 가려져도 과거를 찾지 않는다 (§2).
구조 · 손실 변형 (되먹임 방식, 힌트, 차이 L1) 은 이걸 바꾸지 못했다 (§3).
그래서 **학습 문제** 를 바꾼다 — 한 clip 에서 **시작점 4 개** 를 골라 각 시작점부터 rollout 해, 문맥의 여러 지점이 "여기서부터 굴려도 맞아야 한다" 는 압력을 받게 한다.
샘플링도 (C, K) 조합 대신 **데이터셋을 먼저** 고르고 (V-JEPA 2 공식 비율), 긴 창 위주로 바꾼다.

---

## 1. 설정 (확정)

| 항목 | 값 | 비고 |
|---|---|---|
| predictor | kind=ar, `ar_feedback: state`, 384 × 12 (약 22M) | `natural_ar_vw_state` 와 같다 |
| fps | 4 (블록 = 2 장 = 0.5 s) | encoder 사전학습과 같다. 평가 (VOBench 0.2 s · IntPhys 2 0.33 s · EK100 0.25 s 블록) 와는 다르다 — 단서 |
| 데이터셋 선택 | **step 마다 데이터셋을 먼저** 고른다. 가중치 K710 0.335 / SSv2 0.100 / HowTo 0.565 | V-JEPA 2 공개 config (`configs/train/vith16/pretrain-256px-16f.yaml`) 의 값. step 마다 토큰 수 (B × W) 가 같아 **이 가중치 = 본 프레임 비율** |
| 창 W (장) | SSv2 {12, 16} · K710 {32, 40} · HowTo {32, 48, 64} — 데이터셋 안에서 균등 | 패딩 없이 채울 수 있는 영상만 (fps4 표 `n4 ≥ W`). 최대 64 장 = 32 블록 = 16 s |
| rollout 시작점 | clip 당 **4 개**, [2, nb−1] 을 4 등분해 구간마다 하나 (층화) | step 마다 새로 · batch 공유 · 모든 rank 같은 값 (`random.Random(f"{seed}-plan-{step}")`) |
| rollout 길이 | 각 시작점에서 창 끝까지, 최대 16 블록 (8 s) | 시작점끼리 겹친다 (뒤 블록일수록 여러 문맥 길이에서 예측됨) |
| 최소 문맥 | 2 블록 (1 s) | 블록 하나로는 움직임을 못 읽는다 |
| 손실 | L1 = teacher forcing (창 전체 다음 블록) + 시작점별 rollout L1 의 **평균** | TF 는 기존과 같다 |
| step · lr | 70 epoch × 294 = 20,580 step, WSD (warmup 1 ep → 5.2e-4 상수 → 마지막 7 ep 선형 1e-6) | 지금 AR run 들과 같다 |
| 토큰 예산 | **미정** (자리표시 950) | 시작점 4 개 rollout 메모리 실측 후 VRAM 최대로 (본 clip 수를 늘리는 손잡이) |

### 계획 재계산 (20,580 step, 예산 950 기준)

| | 지금 AR (C/K 조합, 1:1:1) | multistart A (16 장 포함) | **multistart B (채택)** | C (데이터별 최대 창만) |
|---|---:|---:|---:|---:|
| 본 clip | 10.5M | 6.0M | **4.5M** | 3.6M |
| rollout 1 걸음 (clip × 시작점) | 10.5M | 24.1M | 18.1M | 14.3M |
| 8 걸음 | 2.0M | 5.2M | **7.2M** | 6.7M |
| 16 걸음 | 0.7M | 1.4M | **1.9M** | 3.2M |
| 16 걸음 / 1 걸음 | 7 % | 5.6 % | **10.3 %** | 22.2 % |
| 시작점 문맥 (중앙) | — | 7 블록 | **10 블록 (5 s)** | 13 블록 |
| 프레임 비율 K710 / SSv2 / HowTo | 38 / 20 / 42 % | 33.7 / 10.6 / 55.6 % | 〃 (가중치 고정) | 〃 |

**B 를 고른 이유** — 16 장 창 (8 블록) 은 시작점과 끝이 가까워 rollout 도 문맥도 짧다. 빼면 문맥이 길어지고 4–8 s 예측이 오히려 는다.
C 는 먼 미래 비중이 가장 크지만 창이 데이터셋마다 하나로 고정되고 본 clip 이 3.6M 으로 준다 (HowTo 64 장만 = 1.2 만 편의 긴 구간 반복).

**본 clip 이 줄어드는 것 (10.5M → 4.5M)** — 프레임 · 계산량 · TF 손실 위치 수는 같다 (긴 clip 을 적게 본다). 줄어드는 것은 서로 다른 시작 장면 수다.
다양성은 짧은 창이 아니라 (그러면 상태를 쌓는 압력이 준다) **토큰 예산 (VRAM 최대)** 으로 늘린다 (§5). HowTo 추가 다운로드는 하지 않기로 했다 (사용자 결정).

### SSv2 가 짧은 문제

SSv2 는 영상이 약 4 s (fps 4 로 15 장) 라 fps 4 를 지키면 긴 rollout 이 불가능하다. 12 장 → 최대 4 블록, 16 장 → 최대 6 블록.
8 장은 rollout 1–2 블록뿐이라 뺐다. SSv2 의 역할 = 짧은 손-물체 상호작용 (1–3 s). 긴 rollout 은 K710 · HowTo 가 맡는다.
(SSv2 를 12 fps 전체 프레임으로 쓰면 길어지지만 fps 4 고정 결정과 충돌해서 하지 않았다.)

---

## 2. 왜 — 진단 (2026-10-13, `z_research/scripts/analysis/ar_attention_lag.py`)

대상 `natural_ar_vw_state`, VOBench 이동 clip (visible 40 · late 40), 채점과 같은 경로 (문맥 8 블록 → 8 블록 rollout). 결과 `z_training/runs/natural_ar_vw_state/eval/ar_attention_lag/ar_attention_lag.json`.

| 무엇 | 결과 |
|---|---|
| attention (rollout 걸음 r) | 자기 블록 0.36–0.39, **문맥 마지막 블록 0.14–0.26** (8 걸음 내내), 더 오래된 문맥 블록 0.03–0.08 (균등 0.06–0.11 보다 낮다) |
| 층별 | 앞 층 (1–5) 은 과거를 넓게 (lag ≥ 2 질량 0.75–0.85), 뒤 층 (6–12) 은 자기 블록 (0.45–0.85) |
| **문맥 자르기** (가장 강한 증거) | 마지막 1 블록만 남겨도 예측 변화 = 원래 오차의 20 % (late 26–29 %), 2 블록 11–15 %, **4 블록 6–9 %** → 실질 문맥 1–2 블록 |
| late vs visible | attention 분포 거의 같다. late 에서 **물체가 가려진 마지막 블록이 0.25**, 물체가 보였던 블록 0.056 — 가려져도 과거를 찾지 않는다 |

→ 학습 중 다음 블록은 마지막 1–2 블록으로 거의 맞혀지므로 오래된 문맥을 쓸 이유가 없었다. **구조가 아니라 학습 문제의 성질** 로 본다.

---

## 3. 이미 해 본 것 (같은 데이터 · step · lr, 결과 그대로 남긴다)

| run | 바꾼 것 | IntPhys1 A.8 | IntPhys 2 | VOBench late P / R | 결론 |
|---|---|---:|---:|---|---|
| (참고) 릴리즈 V-JEPA 2 | — | 88.89 | 미측정 | 57.9 / 65.7 | |
| tube 기준선 (`k710_ssv2_how_ep70_cooldown64`) | 릴리즈 레시피 | 82.22 | 54.74 (vb8) | 56.0 / 65.7 | VOBench late 가 릴리즈와 같다 = 레시피 재현 확인 |
| `natural_ar_vw_hint` | 출력 되먹임 + 힌트 (sin²) | 91.67 | 52.37 | 75.0 / 76.9 | |
| `natural_ar_vw_state` | 상태 되먹임 | 90.00 | 51.78 | 74.1 / 77.8 | **기준** |
| `natural_ar_vw_state_hint` | + 힌트 10 % 고정 (ep 1–63) | 91.11 | 51.38 | 70.4 / 76.9 | 힌트 효과 없음 (힌트 끊자 tf 손실이 기준과 정확히 같아짐: 0.5308 vs 0.5309, rollout ±0.0004) |
| `natural_ar_vw_state_delta` | + 차이 L1 (가중 1.0) | 90.56 | 51.58 | 74.1 / 75.0 | VOBench 전체 −1.8 / −1.9 pt, 기존 L1 도 나빠짐 (tf +0.019) |

- AR 은 VOBench late 가 tube · 릴리즈보다 **+10–18 pt** — 대부분 영속성 (90–97 %). **late 모양 · 색은 55–72 % (우연 근처)** 로 tube 와 비슷하다 → 존재는 잇지만 정체는 못 잇는다
- IntPhys 2 는 tube 가 AR 보다 2–3 pt 높다 (둘 다 우연 근처, 506 쌍 ±2.2 pt). 학습 방식과 채점 타깃 (블록별 vs 창 전체) 이 같이 달라 원인을 가를 수 없다
- 풀링 Δ 코사인 손실은 시도 전 검사로 기각 (§3-1)

### 3-1. 풀링한 특징 차이의 코사인 (기각, 2026-10-13)

VOBench visible 이동 60 clip (888 전이), Δh = mean_n LN(h_{t+1}) − mean_n LN(h_t).
같은 방향 쌍 코사인 +0.005 · 반대 방향 −0.003 (상관 r = 0.011). 선형 디코딩 (clip held-out) 으로는 방향을 45° 안 69–74 % 맞힘 →
**정보는 풀링 후에도 남지만 Δh 벡터의 아주 작은 부분공간이라 코사인 손실로는 신호가 거의 없다.** 손실로 쓰지 않았다.

---

## 4. 구현 (vast 브랜치, 기존 경로는 그대로)

| 파일 | 변경 |
|---|---|
| `app/vjepa_frozen/var_window.py` | `mode: multistart` — `_init_multistart` · `plan_ms` · `make_ms` · `_starts` (층화). 기본 `mode` 는 기존 (C, K) |
| `src/models/rollout_predictor.py` | `rollout_kv_multi(x, idx, starts, rollouts)` — 문맥을 한 번만 채우며 시작점마다 KV 캐시를 공유해 rollout. `forward(ar={..., "starts", "rollouts"})` 분기 |
| `app/vjepa_frozen/ar.py` | `ar_losses_multistart` — TF + 시작점별 rollout L1 평균 |
| `app/vjepa_frozen/train.py` | `info["starts"]` 가 있으면 multistart 경로 (힌트 · 차이 L1 과 같이 쓰면 죽인다) |
| `configs/training/natural_ar_ms.yaml` | 위 §1 |
| `z_training/monitor_ar.sh` | multistart 로그 (`mask ms W.. 데이터 B.. s시작점`) 표시 — (데이터, 창) 별 최근 rollout 손실 |

**검증 (CPU, 작은 모델, float64)**
- `rollout_kv_multi` vs 시작점마다 따로 `rollout_kv`: 출력 차 **0**, grad 차 1e-19 (상태 · 출력 되먹임 둘 다)
- 로더 계획 20,580 step 재계산: 프레임 비율 K710 33.7 / SSv2 10.6 / HowTo 55.6 % (가중치와 일치)
- `DRYRUN=1 bash z_training/train.sh natural_ar_ms` 통과

---

## 5. 남은 일 (순서)

1. **토큰 예산 실측** (10-15 오전, EK100 평가가 GPU 를 비운 뒤): C/W 최악 조합 (HowTo 64 장, 시작점 4 개) 에서 rank 당 메모리 → VRAM 을 채우는 예산. 본 clip 수가 그만큼 는다
2. **학습** (약 9–10 h 추정, 측정 필요) → IntPhys1 A.8 · IntPhys 2 · VOBench (`chain_ar.sh` + `cvpr/eval/testbed`) · attention 진단 재실행
3. **판단 기준** (상태 run `natural_ar_vw_state` 대비):
   - attention 진단: late 에서 물체가 보였던 블록 질량 (지금 0.056) ↑, 문맥 자르기에서 앞쪽 블록 의존 ↑ (지금 4 블록만 남겨도 6–9 %)
   - VOBench late 모양 · 색 (지금 55–72 %) ↑, visible · early · mid 유지
   - 이 셋이 안 움직이면: 다음 손잡이 = teacher forcing 가중치 ↓ (1 → 0.5), 시간 방향 입력 마스킹 (마지막 1–2 블록 지우기)
4. ~~HowTo 추가 다운로드~~ — **하지 않는다 (사용자 결정, 2026-10-14).** 다양성은 토큰 예산 (VRAM 최대) 으로 늘린다. 단서: 긴 창 (48–64 장) 은 HowTo 1 % (1.2 만 편 · 약 1,330 시간) 에서만 나오므로 긴 창 학습의 다양성은 제한된다 (프레임 기준 약 4 회 반복)

## 6. 단서 (결론과 같은 비중)

- 이 설계는 진단 (마지막 블록 의존 · 먼 미래 학습 부족 · HowTo 다양성 부족) 에 근거한 **가설** 이다. 효과는 미확인
- 데이터 비율 (1:1:1 → 공식), 샘플링 (C/K → 데이터셋 우선), 창, rollout 시작점이 **한꺼번에** 바뀐다. 상태 run 과의 차이를 어느 하나로 돌릴 수 없다
- 가려짐 자체가 자연 영상 학습 데이터에 드문 것은 그대로다 — late 정체 문제에는 간접적인 처방
- fps 4 학습 vs 평가의 더 촘촘한 블록 간격 (§1)
- V-JEPA 2 대비 규모: 본 clip 약 1 % 대 (V-JEPA 2 ≈ 774M clip 은 외부 정리의 252K step 가정), 데이터 0.84M 편 vs ≈ 22M 샘플 (`z_training/RESULTS_2026-10-12.md` 이후 대화 기록)

## 재현

```bash
# 계획 재계산 · 단위 검증은 §4 (CPU)
DRYRUN=1 GPUS=8 bash z_training/train.sh natural_ar_ms
# 학습 (토큰 예산 확정 후)
GPUS=8 SET="data.var_window.token_budget=<실측값>" bash z_training/tmux_train.sh natural_ar_ms
tmux new-session -d -s chain_ar "A=natural_ar_ms B= bash z_training/chain_ar.sh"
watch -c -n 2 bash z_training/monitor_ar.sh
# attention 진단 (CPU)
/venv/vjepa/bin/python z_research/scripts/analysis/ar_attention_lag.py --run natural_ar_ms --n 40 --threads 128
```
