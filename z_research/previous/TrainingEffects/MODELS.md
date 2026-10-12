# 학습한 모델 목록 — 무엇을 어떻게 학습했나 (2026-09-25 정리 · 2026-09-27 Ariel 세 팔 갱신)

> 상위: [`README.md`](README.md). 수치는 각 체크포인트의 `config`/`arch` 키와 학습 로그에서 읽었다.
> 레지스트리 이름은 `configs/protocols/models.md` 의 섹션 이름이다 (`run.sh <프로토콜> <데이터셋> <이름>`).

## 그룹 A — encoder 는 릴리즈 ViT-H 그대로 (frozen), predictor 만 다르다

같은 encoder 위에서 predictor 만 바꿨으므로 **점수 차이는 전부 predictor 몫**이다. copy 기준선
(문맥 마지막 latent 를 미래에 그대로 복사)도 encoder 만의 함수라 여섯 모델이 공유한다.

| 레지스트리 | 출발점 | 학습 데이터 | clip | 창 · 마스크 | epoch | lr · global batch | 최종 loss | 원본 |
|---|---|---|---:|---|---:|---|---|---|
| `vith` | — | Meta 릴리즈 (VideoMix22M, 1M+ 시간) | — | 릴리즈 (multi-block, 64f) | — | — | — | HF `vjepa2-vith-fpc64-256` |
| `vith_pft_v11_e10` | 릴리즈 predictor | `v11_split_train` — v11 block 절반, **가능만** | 10,752 | 32f stride 3, `temporal_prefix` C16 → P16 | 10 | 5e-5 · 64 | 0.582 → 0.480 | `z_training/runs/v11_postft/e10.pt` |
| `vith_pft_intphys1_e40` | 릴리즈 predictor | IntPhys 2019 **train** 장면 (가능만) | ~3,750 | 32f stride 2, `window_grid` = Garrido 채점 격자 | 40 | 5e-5 · 32 | 0.564 → 0.508 | `z_training/runs/intphys1_postft/e40.pt` |
| `vith_pft_intphys2_e80` | 릴리즈 predictor | IntPhys2 Main split train (가능만, mp4) | 254 | 48f frame_step 10, `temporal_prefix` C ∈ {12…42} | 80 (240 step) | 5e-5 · 80 | 0.605 → 0.585 | `z_training/runs/intphys2_postft/e80.pt` |
| `vith_pft_predv1_e15` | 릴리즈 predictor | Predictor_v1_training — 무중력 등속 · 가림(k=4) · 경사 가속 3 팔, 속도·가속 다양 | 14,250 | 32f stride 3, `temporal_prefix` C16 | 15 | 5e-5 · 64 | 0.586 → 0.474 | `z_training/runs/predictor_v1_postft/e15.pt` |
| `vith_ariel_bc_ep43` ⚠️ **파일 지워짐 (2026-09-27)** | **scratch** | SSv2 + K400 @ 12 fps (자연 영상) | ≈797 step × 224 / ep | 48f, `prefix_window` 문맥 {2,4,8,16} × 예측 {1,2,4,8} slot | 43 / 45 | 3e-4 · 224 | 0.5821 (ep30 부터 고정) | ~~`z_training/ariel/block_causal_future_only_1e_5_ep_43/latest.pt`~~ |
| `vith_ariel_prefix_ep45` (2026-09-27) | **scratch** | 〃 (SSv2 `train_min48` + K400 320p `train_min96`) | — | 〃, **prefix** (문맥 양방향 · 미래 block-causal) + mask token — Ariel 팔 B | **45 / 45** (step 36,064) | 3e-4 · 28/rank ※ | 0.5836 | `z_training/ariel/block_causal_future_only_1e_5/latest.pt` |
| `vith_ariel_full_ep40` (2026-09-27) | **scratch** | 〃 | — | 〃, **full attention** (`oneshot`, 릴리즈 위상) + mask token — Ariel 팔 A | **40 / 40** (step 35,840) | 3e-4 · 28/rank ※ | 0.5761 | `z_training/ariel/full_attn_future_pred_1e_5/latest.pt` |
| `vith_ariel_ar_ep18` (2026-09-27) | **scratch** | 〃 | — | 〃, **자기회귀** (`ar`, 블록별 target encoder 입력 · rollout, mask token 없음) — Ariel 팔 C | ⚠️ **18 / 40** (학습 중, step 16,128) | 3e-4 · 28/rank ※ | 1.047 (tf + rollout — 다른 팔과 비교 불가) | `z_training/ariel/ar_future_1e_5/latest.pt` |

- post-FT 넷은 전부 **손실 = LN(target) 대비 L1, mask_index 0, 문맥 → 미래 마스크** — 우리 채점 (`surprise_c16t32`) 과 같은 식을 학습한 것이다.
- ⚠️ `ariel` 은 post-FT 가 아니다. **attention 규칙 (block-causal) · 데이터 · scratch · epoch** 네 가지가 동시에 다르다.
  원인을 attention 규칙으로 좁힐 수 없다 (옛 `z_training/ariel/README.md` §4-c — 문서는 사라졌고 정본은 이제 `z_training/ARIEL_CHECKPOINTS.md`).
- ⚠️ **2026-09-27 — `vith_ariel_bc_ep43` (와 ep5 · 19 · 30 · 41) 의 체크포인트 파일은 지워졌다** (`z_training/ariel/` 이 세 run 의
  최종판으로 바뀌었다). 이 폴더 문서 · `te_*` 산출물의 **`ariel_ep43` 수치는 그 옛 파일의 것으로 남긴다** (라벨 그대로 ep43).
  새 판은 위 세 줄 (`ariel_prefix_ep45` · `ariel_full_ep40` · `ariel_ar_ep18`) 이고 ep45 는 ep43 과 **다른 파일**이다 — ep43 수치를
  ep45 로 옮겨 적지 않는다. 설명 · 출처는 `z_training/ARIEL_CHECKPOINTS.md`, 레지스트리는 `configs/protocols/models.md`.
  `te_*` 스크립트 preset: final 의 `ariel_ep43` → `ariel_prefix_ep45`, curves 의 `ariel_ep{5,19,30,43}` 은 대체 없이 뺐다.
- 새 세 팔은 **데이터 · 창 · (C, K) 분포 · lr 이 같고 predictor 규칙만** 다르다 — A (full) vs B (prefix) 가 attention 규칙 하나만 다른
  대조다. ⚠️ C (AR) 는 규칙 하나가 아니라 인코딩 단위 (블록별) · 문맥 encoder (EMA) · 입력 (되먹임) 이 같이 바뀐다.
- ⚠️ `ariel_ar_ep18` 은 채점 표적이 **블록별 LN(target_encoder)** 이라 (`analysis/predictors/ar_scoring.py`) 다른 모델과 **쌍 정확도만**
  비교한다. 위 "copy 기준선을 여섯 모델이 공유한다" 는 AR 에 해당하지 않는다 — AR 의 복사 대조는 `copy_blk`
  (`te_v11_score.py` · `te_intphys1_score.py`). readout (`te_v11_readout.py` · `te_v3_readout.py`) 은 AR 을 거부한다.
- ※ 새 세 팔의 global batch 는 **미검증**: 체크포인트 `config` 는 `data.batch_size` 28 · `tasks_per_node` 8 (→ 224) 이지만
  step/epoch 이 prefix 801.4 · full 896 · AR 896 으로 달라 실제 GPU 수가 run 마다 같았는지 기록으로 확인되지 않는다.
- ⚠️ `v11_postft` 는 v11 의 **궤적을 외운다** — v11 은 운동 조건마다 궤적이 2 개 (좌→우 · 우→좌) 뿐이고 학습·test 절반 모두에 있다.
  held-out block 점수도 "처음 보는 궤적" 이 아니다 (09-17 정정, `z_research/predictor_training/predictor_IntPhysGenV11_PFT/README.md`).
- ⚠️ `intphys1_postft` 는 dev 와 **같은 생성기 · 같은 물체 분포**의 train 장면으로 학습했다.
- 한 번씩만 학습했다 (seed 0). 학습 분산은 모른다.

## 그룹 B — encoder · predictor 전부 새로 학습 (jongseo 사전학습, V-JEPA 2 레시피)

V-JEPA 2 논문 Table 10 "abbreviated recipe" (16f @ 4 fps, 256, multi-block 마스크 8×0.15 + 2×0.7, global batch 3072,
cosine lr 6.25e-4 → 1e-6, EMA 0.999 → 1.0). predictor 는 모두 384d × 12L × 12h, mask token 2 개 중 `[0]` 만 학습.
원본 `/data/jongseo/project/world-model/vjepa2-pretrain/exp/`.

| 레지스트리 | encoder | 데이터 | step | 상태 | 최종 loss |
|---|---|---|---:|---|---|
| `vitb_k400_e240` | ViT-B (86M) | K400 240K clip | 72K | 90K 스케줄을 14일 벽에서 멈춤 (= 이 run 의 최종) | 0.574 |
| `vitb_k400_e100` | ViT-B | K400 | 30K | ⚠️ 90K 스케줄의 **중간** (lr 미감소) — `vitb_synphys_30k` 와 step 수만 같다 | — |
| `vitb_synphys_30k` | ViT-B | **합성 물리 50K clip** (2D pymunk 30K + 3D pybullet 20K; 낙하·충돌·가림·지지·경사·진자·포물선…) | 30K | 완주 | 0.505 |
| `vittiny_k400_5k` | ViT-Tiny (5.7M) | K400 | 5K | 완주 — 아래 synphys 5K 와 **데이터만 다른 짝** | 0.563 |
| `vittiny_synphys_5k` | ViT-Tiny | 합성 물리 50K | 5K | 완주 | 0.285 |
| `vittiny_k400` | ViT-Tiny | K400 | 30K | 완주 | 0.493 |
| `vittiny_synphys_30k_e225` | ViT-Tiny | 합성 물리 50K | 22.5K / 30K | ⚠️ **학습 중** (09-25 기준 epoch 236/300) — lr 미감소 중간값 | ~0.37 |
| `vittiny_k400ssv2_30k_e225` | ViT-Tiny | K400 0.77 + SSv2 0.23 | 22.5K / 30K | ⚠️ **학습 중** (234/300) | ~0.57 |

- 합성 물리 데이터는 IntPhys · IntPhysGen 을 쓰지 않았다 (누수 없음). 렌더 도메인은 IntPhys (Unreal) 와 가깝다.
- ⚠️ 학습 입력은 **16f @ 4 fps** 다. 채점 프로토콜의 시간 간격 (v11: 30 fps 원본 stride 3 = 10 fps, IntPhys1: skip 2/5/10) 과 다르다.
- 데이터가 다른 모델끼리 loss 절대값은 비교하지 않는다 (합성 영상이 쉬워서 낮다).
- K400 attentive probe (sanity) 는 jongseo 가 따로 돌리고 있다 (`vjepa2-pretrain/exp/2026-09-24_k400probe_ctrl/`).

## 재현

```bash
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
python - <<'EOF'
import torch
d = torch.load("z_training/runs/v11_postft/e10.pt", map_location="cpu", weights_only=False)
print(d["arch"], d["epoch"], d["loss"], d["config"]["data"])
EOF
grep -n "^## " configs/protocols/models.md
```
