> 📥 **vast 브랜치에서 가져온 문서 (2026-10-24, `origin/vast` c05c259 의 `z_training_config/CTXT_VIDEO_TGTVIDEO_HOLE_2026-10-16.md`, 원문 그대로).**
> 본문의 run 이름 · `/workspace/...` 경로 · 시트 언급은 vast 노드 기준이다 — 이 노드 (vll5) 에 그 산출물이 없을 수 있고, 수치는 그쪽 산출물에서 계산된 것이다.

# 문맥 = 영상 · 타깃 = 영상 · 문맥 구멍 — 샘플링 점검과 설정 기록 (2026-10-16)

대상 config
- `configs/training/natural_ar_vw_state_ctxt_video_tgtvideo.yaml` — 문맥 영상 + 타깃 영상 (다른 서버에서 학습)
- `configs/training/natural_ar_vw_state_ctxt_video_tgtvideo_hole.yaml` — 위 + 문맥 구멍 (이 서버 `chain_hole` 의 2 번째)

둘 다 `natural_ar_vw_state_ctxt_video` 를 물려받는다. 앞 run 의 결과 · 정정은 `../z_training/docs/RESULTS_2026-10-16.md` §3.
아래 수치는 학습 계획 20,580 step 을 로더 규칙 (`app/vjepa_frozen/var_window.py` `plan`) 과 구멍 생성기 (`app/vjepa_frozen/ar.py` `make_ctx_hole`) 를 그대로 돌려 센 것이다 (학습 전 계산).

## 1. 무엇을 하는가

| | AR (`natural_ar_vw_state`) | ctxt_video | **tgtvideo** | **tgtvideo_hole** |
|---|---|---|---|---|
| kind | ar | ctx_state | ctx_state | ctx_state |
| 문맥 입력 | `target_encoder` 블록마다 | `encoder` 문맥 창 한 번 | 같음 | 같음 — **보이는 토큰만** encoder 에 |
| TF 입력 (예측 구간) | 블록별 LN(h) | 블록별 LN(h) | 블록별 LN(h) | 블록별 LN(h) |
| 타깃 | 블록별 LN(h) | 블록별 LN(h) | **창 전체 LN(target_encoder)** 의 미래 | 같음 + 가린 문맥 자리 |
| 손실 | TF + rollout | TF + rollout | TF + rollout | TF + rollout + 가린 문맥 복원 |
| 문맥 C (장) | {2,4,8,16,32} | {4,8,16,32} | 같음 | 같음 |
| 예측 K (장) | {2,4,8,16,32} | {4,8,16,32} | 같음 | 같음 |
| 데이터 가중치 k710 : ssv2 : how | 1 : 1 : 1 | 0.65 : 0.60 : 1.00 | 같음 | 같음 |

- TF 입력이 블록별인 이유: encoder 는 시간 양방향이라 예측 구간을 영상으로 넣으면 블록 b 토큰이 b+1 을 본다 (누수). 타깃은 손실에만 쓰므로 영상이어도 된다
- 타깃을 영상으로 바꾼 이유 (사용자 결정): ctxt_video (영상 문맥 → 블록 타깃) 가 블록 복사 기준선보다 낮았다. "문맥과 답의 종류가 다르다" 는 **미검증 가설**을 한 변수로 시험한다

## 2. 샘플링 점검 — 계획 그대로 셈

| 항목 | 값 |
|---|---|
| 조합 | 16 (C 4 × K 4), 균등 |
| 창 W = C + K | 8 · 12 · 16 · 20 · 24 · 32 · 36 · 40 · 48 · 64 장, 최대 64 (predictor 시간 상한 32 블록 안) |
| 시간 척도 | 전부 fps 4 (6 장 / 3 장 간격 정수), 패딩 없음 — 릴리즈 사전학습 · cooldown 과 같은 fps 4 |
| 본 clip | **7.08 M** (AR 10.45 M — 창이 길어져 clip 수만 준다. 프레임 예산 1.5 억은 같다) |
| 본 프레임 비율 | k710 33.2 / ssv2 9.9 / how 56.9 % (릴리즈 0.335 / 0.1 / 0.565 에 맞춤. **AR 의 38 / 20 / 42 % 와 다르다**) |
| 문맥 C 별 step | 4 장 26 % · 8 장 25 % · 16 장 25 % · 32 장 25 % |
| SSv2 | 원본 12 fps → 3 장 간격. 적격 2 만 편 이상인 6 개 창 (W ≤ 20) 에만. 1.13 M clip, 영상당 평균 5.3 회 (ctxt_video 학습 로그로 확인: 6 개 창에서만 나왔다) |
| 증강 | RRC scale 0.3–1.0 · ratio 0.75–1.35, 좌우 반전 0.5 — clip 전체에 같은 창 |

**문제 없음** — 계획 · 실측 (ctxt_video 로그) 이 맞고, 패딩 · 창 상한 · 데이터 비율이 의도대로다.

## 3. 문맥 구멍 (tgtvideo_hole 만)

| 항목 | 값 |
|---|---|
| 언제 구멍을 내나 | **encoding 전** — context encoder 에 보이는 문맥 토큰 인덱스만 넘긴다 (V-JEPA 2 사전학습의 마스크와 같은 방식). predictor 는 가린 자리를 `mask_tokens[0]` 으로 채운다 (ctx_state 가 원래 안 쓰던 파라미터 → 체크포인트 · 채점 그대로) |
| 적용 | step 의 75 % (실제 계산 74.3 %, 15,282 step), batch 안 clip 전부 |
| 어디를 | **문맥 끝 k 블록**, k ~ U{1..min(4, C−1)} — 보이는 문맥 블록을 최소 하나 남긴다 |
| 공간 | V-JEPA 2 긴 범위 tube — 70 % 직사각형 2 개 합집합 (가로세로 0.75–1.5), k 블록 내내 같은 자리. batch 안 encoder 입력 길이를 맞추려고 더 가리는 토큰은 구멍 블록 안에서 → 구멍 블록당 평균 약 90 % |
| 생성기 | 적용 여부 · k = (seed, step) 공통 (rank 끼리 같은 계산량), 직사각형 = (seed, step, rank) — rank 마다 다른 구멍 |
| 손실 | TF + rollout + 1.0 × L1(가린 문맥 자리의 predictor 출력, 창 LN(h) 의 그 자리) |

(문맥 블록 수, k) 별 구멍 step: (2,1) 3,872 · (4,1/2/3) 1,238 / 1,273 / 1,257 · (8,1–4) 928 / 939 / 909 / 986 · (16,1–4) 947 / 984 / 957 / 992.

### ⚠️ 정정 — 첫 구현은 encoding 뒤에 구멍을 냈다 (폐기)

처음엔 kind=ar 위에서 encoder 출력 (블록별 LN(h)) 을 mask token 으로 바꿨다. 블록 단위 encoding 은 블록 안에서 공간 attention 을 하므로
**보이는 토큰이 가린 자리를 이미 본 뒤라 누수**다 (사용자 지적). 되돌렸고, `loss.ctx_hole` 을 kind=ar 에 걸면 에러를 낸다.

## 4. 알려진 단서 (문제는 아니지만 결과를 읽을 때 같이 볼 것)

1. **encoder 가 사전학습보다 짧은 영상을 본다** — 문맥 4 장 (C 의 26 %) · 창 8 장. 릴리즈는 16 장 (cooldown 64 장) 으로만 학습됐다. 다만 AR 은 2 장짜리 블록을 넣고도 잘 됐고, 릴리즈 채점 (A.8 창 16, C 2–10 장) 도 같은 상황이다
2. **같은 미래 프레임의 타깃이 창 길이에 따라 달라진다** — 창 전체를 양방향으로 encoding 하므로 W 8 ~ 64 에 따라 미래 토큰 값이 다르다. 채점은 창 16 · 32
3. **k 가 1 쪽으로 치우친다** — 문맥 2 블록 step (26 %) 은 k = 1 만 가능해서 k = 1 이 구멍 step 의 46 %
4. **SSv2 는 짧은 문맥 (≤ 16 장) 에만** 들어간다 — 영상이 짧아서 (fps 4 기준 중앙 15 장) 생기는 구조. 긴 창의 SSv2 는 긴 영상 쪽으로 치우친다
5. **AR 과는 1 대 1 이 아니다** — 문맥 범위 · 데이터 비율 · 구조가 다르다. 깨끗한 비교는 ctxt_video ↔ tgtvideo (타깃만) ↔ tgtvideo_hole (구멍만)
6. **비용** — 창 타깃을 위해 창 전체 forward 가 하나 더 든다. encoder 만 ×1.8–2.4 (C8K8 → C32K32 측정) → 학습 약 13–16 h (미측정)
7. SSv2 목록 (`train_min10.csv`) 에 공식 train · validation · test 가 다 있다 — 우리 평가에는 무관, SSv2 분류 probe 를 하면 누수

## 5. 미리 적어 둔 판정 (tgtvideo_hole vs tgtvideo)

| 결과 | 판정 |
|---|---|
| VOBench late ↑ · 앞쪽 문맥 블록 attention ↑ · visible / IntPhys1 유지 | 문맥 구멍이 가림 추정으로 옮겨 간다 → 세기 (p · k · 면적) ablation |
| attention 만 ↑, late 그대로 | 먼 블록은 보지만 진짜 가림으로 안 옮겨 간다 |
| 둘 다 그대로 | 접는다 |

tgtvideo 자체의 판정 (ctxt_video 대비): VOBench visible · 전체가 블록 복사 기준선 (77.8 / 81.1) 을 넘으면 "문맥 · 답 종류 불일치" 가설에 힘이 실린다.

## 6. 검증

- CPU (작은 predictor): 구멍 없음 = 기존 출력 (비트 단위) · 가린 자리 z 를 바꿔도 출력 불변 · keep 과 hole 이 정확히 여집합 · batch 안 keep 길이 같음 · KV rollout = 매 걸음 재계산 (구멍 포함) · 세 손실 계산 · mask token grad · C = 1 블록이면 구멍 없음 · rank 마다 직사각형 다름 (적용 여부 · k 는 같음)
- GPU 1 장 (실제 ViT-H, 작은 batch): tgtvideo · tgtvideo_hole 모두 `done`, 로그에 구멍 비율 · 복원 손실
- 미확인: 최대 창 (C16K16 블록, 64 장) 메모리 · step 시간 — 학습 첫 step 에서 잰다

## 재현

```bash
# 계획 계산 (이 문서 §2 · §3 의 수치)
python z_training/harness/resolve_train.py natural_ar_vw_state_ctxt_video_tgtvideo_hole -o /tmp/hole/config.yaml --folder /tmp/hole
# → VarWindowBatches(config.data.var_window).plan(step) · make_ctx_hole 을 step 0..20579 에 돌려 센다
# 1 GPU 스모크
NAME=_smoke DEBUG=1 GPUS=1 SET="optimization.ipe=6 optimization.epochs=1 data.var_window.token_budget=48 data.var_window.b_min=2 val=null" \
  bash z_training/train.sh natural_ar_vw_state_ctxt_video_tgtvideo_hole
# 학습 (이 서버는 chain: op → 채점 → hole → 채점)
tmux new-session -d -s chain_hole "bash z_training/chain_hole.sh"
GPUS=8 bash z_training/tmux_train.sh natural_ar_vw_state_ctxt_video_tgtvideo        # 다른 서버
```
