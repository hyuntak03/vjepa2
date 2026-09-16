# predictor_IntPhysGenV11_PFT — v11 로 이어 학습한 predictor 는 어떤 미래를 만드나 (시작점)

> 상위: `z_research/predictor_training/` (predictor 학습 결과를 모으는 자리). 학습 자체는 `z_training/runs/v11_postft` (2026-09-10; `z_training/README.md`).
> 이 세트는 **그 predictor 의 출력 p 를 위치 자로 읽어** 릴리즈 predictor 와 나란히 놓는다. 캐시를 만들지 않고 frozen encoder + predictor 를 바로 forward 한다.

## 무엇을 묻나

릴리즈 predictor 는 마지막 관측 상태를 3–4 튜블릿 외삽하고 (등속), 가속·수직에서 ≤1–2, 경계에서 가려진 물체는 놓지 않는다 (`context_encoder_analysis/Archive/EXPERIMENTS_2026-09-16.md`).
v11 train 절반으로 predictor 만 10 epoch 이어 학습하면 late 가림 채점이 57–60 → 84–87 로 오른다. **그 회복이 물체를 실제로 미래에 놓는 것(상태 진화)인지, 채점 margin 만 넓힌 것(도메인 prior)인지** 를 위치 자로 가른다.

## 세팅

| 항목 | 값 |
|---|---|
| predictor | `z_training/runs/v11_postft/latest.pt` (e10; `temporal_prefix` C16, lr 5e-5, v11_split_train 가능 clip) vs 릴리즈 ViT-H |
| encoder | 둘 다 릴리즈 ViT-H (frozen) — `configs/protocols/models.md` `vith` |
| forward | `attn_probe` 프로토콜과 같은 병합 config (32 장 stride 3, 256 px, bf16), `mask_index 0`, 문맥 16 → 미래 16. 캐시 저장 없음 |
| 자 | `RollOutV2/exp_results/v5/attentive_pooling/p/attn.pt` (릴리즈 p 로 학습한 3,842 파라미터 attentive 자). **post-FT p 에는 이식** — 자가 post-FT 의 표현을 못 읽으면 attention 질량이 균등 (0.035) 근처로 나온다 → 그때는 위치를 읽지 않는다 (RollOutV2 읽는 규칙) |
| 데이터 | (a) RollOut_v2 가능 clip 4,704 (7 시나리오) — v11 과 다른 도메인이라 **일반화** 검사. (b) v11 vanish pos_a: flat/ramp × k0(224) / early k4 / mid k4 / late k4 (각 56) — **학습 도메인**. ⚠️ v11 train/test split 은 block 단위인데 이 조건 clip 이 train 절반에 포함됐을 수 있다 (index_train 대조 필요 — 미확인) |
| 스크립트 | `z_research/scripts/analysis/pft_ruler_direct.py` |

## 결과 (2026-09-16) — 위치: `Archive/RESULTS_2026-09-16.md` · 채점 케이스별: `Archive/SCORING_2026-09-16.md`

- **학습 도메인 (v11)**: post-FT 는 물체를 8 슬롯 내내 진실 ±1–2 칸에 두고 (질량 0.8–0.97; 릴리즈는 슬롯 4–5 뒤 0), **late k=4 에서 가림막 뒤를 건너 재출현 슬롯 2 부터 제 자리에 놓는다** (0.58→0.97). 채점 회복은 물체를 실제로 놓아서 얻은 것.
- **다른 도메인 (RollOut_v2)**: 물체를 진실보다 **2 배 빨리** 옮기고 (β 1.8–2.2 = v11 속도 강요), 경사·벽·낙하를 무시한다 (ramp_a R² −4.5, wall 70 px 통과). 배운 것은 일반 규칙이 아니라 **v11 궤적 prior**.
- → predictor 만 학습해도 가림 뒤 재출현을 만들 수 있다 (재료는 encoder 에 있다). 일반화하려면 학습 데이터가 속도·동역학을 넓게 덮어야 한다.

읽을 것 세 가지:
1. RollOut_v2 시나리오별 β(gain)·MAE·슬롯별 물체 질량 — 릴리즈 vs post-FT. post-FT 가 flat 이외(ramp/arc/fall/ledge/wall)에서도 물체를 더 오래·정확히 놓나.
2. v11 late k=4 — 릴리즈는 슬롯 0 부터 물체 질량 0.03. post-FT 가 가림막 뒤 물체를 미래에 놓나 (질량·d_truth).
3. 릴리즈 판이 캐시 기반 RollOutV2 수치 (flat_v L2 0.71 칸 · β 0.85, late 질량 0.03) 와 맞는가 = 파이프라인 검증.

## 재현

```bash
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2; P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
CUDA_VISIBLE_DEVICES=0 $P z_research/scripts/analysis/pft_ruler_direct.py --predictor release --set rollout_v2 v11
CUDA_VISIBLE_DEVICES=1 $P z_research/scripts/analysis/pft_ruler_direct.py --predictor pft     --set rollout_v2 v11
```
