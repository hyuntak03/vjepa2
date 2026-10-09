# 모델 레지스트리

`z_research/scripts/run.sh <프로토콜> <데이터셋> [모델]` 의 세 번째 인자.
`## 이름` 아래의 `key: value` 만 읽는다 (그 밖의 줄은 전부 설명으로 무시된다).
여기 값들은 최종 config 의 `model:` 블록으로 들어가고, **프로토콜 yaml 의 `model:` 이 이긴다.**

기본은 `vith` 다. IntPhys1 동일 프로토콜에서 ViT-H 88.89% vs ViT-L 64.44% (+24.4pt) —
**외부 벤치마크에서 먼저 정한 것**이라 자체 데이터로 뒤집지 않는다.
자체 데이터에는 ViT-L 이 이기는 축이 있다 (v8 shape violation 83.98 vs 73.83,
2D transit overall 75.39 vs 71.48, Jongseo physv3 92% vs 80%).
**"ViT-H 가 항상 낫다"는 거짓이니 그렇게 쓰지 말 것.** 사후 모델 선택을 안 한다는 원칙일 뿐이다.

predictor(embed 384 / depth 12 / heads 12 / mask token 10)는 H·L 이 완전히 같다.
encoder 만 hidden 1024→1280, depth 24→32 로 커진다.

---

## vith

arch_name: vit_huge
checkpoint: /data/hyuntak/project/2026/2027_cvpr/vjepa2/checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth

## vitl

arch_name: vit_large
checkpoint: /data/hyuntak/project/2026/2027_cvpr/vjepa2/checkpoint/models--facebook--vjepa2-vitl-fpc64-256/snapshots/b3c1679b7c34d3255ef3547f27c7b226aefab26f/original/model.pth

---

**학습한 predictor 로 채점하기** — 여기 등록하지 않아도 된다. `bash z_training/eval.sh <run>` 이 base 모델(`vith`)의
encoder 에 `model.predictor_checkpoint=<run>/latest.pt` 를 얹어 `run.sh` 를 부른다
(`analysis/intphys2/model.py`, 2026-09-10). 통짜 파일로 등록하려면 `z_training/harness/export_ckpt.py` 가 만든
파일을 새 섹션에 적는다 (⚠️ 이 파일의 파서는 `key: value` 줄을 전부 읽으므로 예시를 코드 블록에 쓰지 말 것).

## vith_ariel_bc_ep5

⚠️ **2026-09-27: 이 체크포인트 폴더는 지워졌다** (`z_training/ariel/` 가 세 run 의 최종판으로 바뀌었다). 이 항목은 옛 결과 (`_resolved.yaml`) 를 읽기 위한 기록이다 — 실행하면 resolve 가 경로 검사에서 죽는다. 지금 판은 아래 `vith_ariel_prefix_ep45` · `vith_ariel_full_ep40` · `vith_ariel_ar_ep18`.

**Ariel 의 block-causal / future-only predictor** (2026-09-23 수령, `z_training/ariel/block_causal_future_only_1e_5_ep_5/latest.pt`).
encoder 는 **릴리즈 ViT-H 그대로**고 predictor 만 바뀐다 — scratch 학습 (릴리즈 predictor 와 상대차 1.07 = 무관),
SSv2 + K400 @ fps 12, 48 프레임 / 256, epoch 5 / step 3360, 학습 loss 0.5938.

predictor 의 **attention 규칙이 릴리즈와 다르다**: 문맥끼리 양방향 · 문맥→미래 차단 ·
미래→미래 block-causal (시간 tubelet slot 단위). 구현은 **Ariel 학습 코드 그대로**
`src/models/rollout_predictor.py` (2026-09-23 수령), 레지스트리 `analysis/predictors/`.
체크포인트의 `arch.kind: prefix` 가 그것을 선언하고 로더가 읽는다 (불일치면 죽는다).
state_dict 키는 릴리즈와 **완전히 같아서** kind 를 안 주면 조용히 full attention 으로 돌아간다 —
그래서 kind 를 여기에도 적는다.

⚠️ 학습 격자는 문맥 slot {2,4,8,16} × 예측 slot {1,2,4,8} 이다. Garrido w32 의
   C=4프레임(slot 2) 칸은 예측 slot 14 라 **학습 범위 밖**이다. 비교할 때 밝힐 것.

arch_name: vit_huge
checkpoint: /data/hyuntak/project/2026/2027_cvpr/vjepa2/checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth
predictor_checkpoint: /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_training/ariel/block_causal_future_only_1e_5_ep_5/latest.pt
predictor.kind: prefix

## vith_ariel_bc_ep19

⚠️ **2026-09-27: 이 체크포인트 폴더는 지워졌다** (`z_training/ariel/` 가 세 run 의 최종판으로 바뀌었다). 이 항목은 옛 결과 (`_resolved.yaml`) 를 읽기 위한 기록이다 — 실행하면 resolve 가 경로 검사에서 죽는다. 지금 판은 아래 `vith_ariel_prefix_ep45` · `vith_ariel_full_ep40` · `vith_ariel_ar_ep18`.

**같은 학습의 epoch 19** (`block_causal_future_only_1e_5_ep_19/latest.pt`, step 12768, 학습 loss **0.5854**).
설정은 `vith_ariel_bc_ep5` 와 완전히 같다 (mask `prefix_window` ctx [2,4,8,16] x pred [1,2,4,8],
lr 3e-4, 총 45 epoch 계획, scratch). **epoch 만 다른 짝**이라 학습량 축을 본다.

arch_name: vit_huge
checkpoint: /data/hyuntak/project/2026/2027_cvpr/vjepa2/checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth
predictor_checkpoint: /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_training/ariel/block_causal_future_only_1e_5_ep_19/latest.pt
predictor.kind: prefix

## vith_ariel_bc_ep30

⚠️ **2026-09-27: 이 체크포인트 폴더는 지워졌다** (`z_training/ariel/` 가 세 run 의 최종판으로 바뀌었다). 이 항목은 옛 결과 (`_resolved.yaml`) 를 읽기 위한 기록이다 — 실행하면 resolve 가 경로 검사에서 죽는다. 지금 판은 아래 `vith_ariel_prefix_ep45` · `vith_ariel_full_ep40` · `vith_ariel_ar_ep18`.

**같은 학습의 epoch 30** (`block_causal_future_only_1e_5_ep_30/latest.pt`, step 22624, 학습 loss **0.5821**).
설정은 `vith_ariel_bc_ep5` / `_ep19` 와 완전히 같다 — **epoch 만 다른 짝**이다.

arch_name: vit_huge
checkpoint: /data/hyuntak/project/2026/2027_cvpr/vjepa2/checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth
predictor_checkpoint: /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_training/ariel/block_causal_future_only_1e_5_ep_30/latest.pt
predictor.kind: prefix

## vith_ariel_bc_ep41

⚠️ **2026-09-27: 이 체크포인트 폴더는 지워졌다** (`z_training/ariel/` 가 세 run 의 최종판으로 바뀌었다). 이 항목은 옛 결과 (`_resolved.yaml`) 를 읽기 위한 기록이다 — 실행하면 resolve 가 경로 검사에서 죽는다. 지금 판은 아래 `vith_ariel_prefix_ep45` · `vith_ariel_full_ep40` · `vith_ariel_ar_ep18`.

**같은 학습의 epoch 41 / 45** (`block_causal_future_only_1e_5_ep_41/latest.pt`, step 32480,
학습 loss **0.5821** — ep30 과 소수 4 자리까지 같다. **loss 는 평탄해졌다**).
설정은 `_ep5`/`_ep19`/`_ep30` 과 완전히 같다 — **epoch 만 다른 짝**이다.

arch_name: vit_huge
checkpoint: /data/hyuntak/project/2026/2027_cvpr/vjepa2/checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth
predictor_checkpoint: /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_training/ariel/block_causal_future_only_1e_5_ep_41/latest.pt
predictor.kind: prefix

## vith_ariel_bc_ep43

⚠️ **2026-09-27: 이 체크포인트 폴더는 지워졌다** (`z_training/ariel/` 가 세 run 의 최종판으로 바뀌었다). 이 항목은 옛 결과 (`_resolved.yaml`) 를 읽기 위한 기록이다 — 실행하면 resolve 가 경로 검사에서 죽는다. 지금 판은 아래 `vith_ariel_prefix_ep45` · `vith_ariel_full_ep40` · `vith_ariel_ar_ep18`.

**같은 학습의 epoch 43 / 45** (`block_causal_future_only_1e_5_ep_43/latest.pt`, step 34272,
학습 loss **0.5821** — ep30·ep41 과 소수 4 자리까지 같다. **loss 는 완전히 평탄하다**).

arch_name: vit_huge
checkpoint: /data/hyuntak/project/2026/2027_cvpr/vjepa2/checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth
predictor_checkpoint: /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_training/ariel/block_causal_future_only_1e_5_ep_43/latest.pt
predictor.kind: prefix

## vith_ariel_bc_ep5_full

⚠️ **2026-09-27: 이 체크포인트 폴더는 지워졌다** (`z_training/ariel/` 가 세 run 의 최종판으로 바뀌었다). 이 항목은 옛 결과 (`_resolved.yaml`) 를 읽기 위한 기록이다 — 실행하면 resolve 가 경로 검사에서 죽는다. 지금 판은 아래 `vith_ariel_prefix_ep45` · `vith_ariel_full_ep40` · `vith_ariel_ar_ep18`.

위와 **같은 weight 를 릴리즈와 같은 full self-attention 으로** 돌리는 대조군.
mask 규칙이 점수에 얼마나 미치는지를 재는 용도다. **이 칸을 성능으로 보고하지 말 것** —
학습 때의 attention 규칙이 아니다.

arch_name: vit_huge
checkpoint: /data/hyuntak/project/2026/2027_cvpr/vjepa2/checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth
predictor_checkpoint: /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_training/ariel/block_causal_future_only_1e_5_ep_5/latest.pt
predictor.kind: default

---

**Ariel 자연 영상 predictor 세 팔 (2026-09-27 수령, 최종 · 진행 중 체크포인트)** — 셋 다 encoder 는 릴리즈 ViT-H 그대로 (frozen),
predictor 는 **scratch**, 데이터 · 창 · (C, K) 분포 · lr 이 같다 (SSv2 `train_min48` + K400 320p `train_min96`, fps 12, 48 프레임,
`prefix_window` 문맥 {2,4,8,16} × 예측 {1,2,4,8} 블록, lr 3e-4). **predictor 규칙만** 다르다. 정본 설명 `z_training/ARIEL_CHECKPOINTS.md`.
⚠️ `predictor.kind` 는 체크포인트 `arch.kind` 에서 온다 — `default` 로 적으면 죽는다 (`oneshot` 은 `oneshot` 으로).

## vith_ariel_prefix_ep45

**팔 B — prefix 마스크 + mask token** (`block_causal_future_only_1e_5/latest.pt`, epoch 45/45, step 36064, loss 0.5836).
문맥 양방향 · 문맥→미래 차단 · 미래→미래 block-causal. 옛 `vith_ariel_bc_ep43` 과 같은 run 의 마지막 epoch.

arch_name: vit_huge
checkpoint: /data/hyuntak/project/2026/2027_cvpr/vjepa2/checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth
predictor_checkpoint: /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_training/ariel/block_causal_future_only_1e_5/latest.pt
predictor.kind: prefix

## vith_ariel_full_ep40

**팔 A — full attention + mask token** (`full_attn_future_pred_1e_5/latest.pt`, epoch 40/40, step 35840, loss 0.5761).
릴리즈 predictor 와 같은 attention (kind `oneshot`) 을 같은 데이터로 scratch 학습 — **prefix 팔의 attention 규칙 대조군**
(`z_training/ariel/README.md` §4-c 가 필요하다고 적었던 팔).

arch_name: vit_huge
checkpoint: /data/hyuntak/project/2026/2027_cvpr/vjepa2/checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth
predictor_checkpoint: /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_training/ariel/full_attn_future_pred_1e_5/latest.pt
predictor.kind: oneshot

## vith_ariel_ar_ep18

⚠️ **2026-09-28 17:22: `ar_future_1e_5/latest.pt` 가 epoch 36 판으로 덮였다.** 이 이름으로 다시 돌리면 ep36 이 채점된다 — 새 실행은 아래 `vith_ariel_ar_ep36` 을 쓴다. 이 항목은 ep18 결과 (`*_vith_ariel_ar_ep18*`) 를 읽기 위한 기록이다.

**팔 C — 자기회귀 (AR)** (`ar_future_1e_5/latest.pt`, ⚠️ epoch **18 / 40** · step 16128 — **학습 중간**, loss 1.047 = tf + rollout 이라 다른 팔과 비교 불가).
mask token 없이 **블록 (2 프레임) 마다 따로** 인코딩한 LN(target_encoder) 를 넣고 블록 인과로 다음 블록을 낸다.
채점기는 `analysis/predictors/ar_scoring.py` 경로를 탄다 (문맥 · 타깃 모두 블록별 target encoder, 미래는 rollout).
⚠️ 타깃 공간이 표준 채점 (창 전체 target encoder) 과 달라 **쌍 정확도만** 다른 kind 와 비교한다. probing · readout 은 아직 정의 안 함 (NotImplementedError).
⚠️ 프로토콜의 `dual_encoder: true` 가 online encoder 를 하나 더 짓지만 AR 경로는 쓰지 않는다 (메모리만 든다).

arch_name: vit_huge
checkpoint: /data/hyuntak/project/2026/2027_cvpr/vjepa2/checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth
predictor_checkpoint: /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_training/ariel/ar_future_1e_5/latest.pt
predictor.kind: ar

## vith_ariel_ar_ep36

⚠️ **2026-09-28 20:19: `latest.pt` 가 epoch 38 로 다시 덮였다** — 이 항목으로 돌리면 ep38 이 채점된다. ep36 파일은 남아 있지 않다 (결과만 `*_ar_ep36*`).

**팔 C — 자기회귀 (AR), epoch 36 / 40** (`ar_future_1e_5/latest.pt`, 2026-09-28 17:22 수령, step 32256, loss 1.044 = tf + rollout).
`vith_ariel_ar_ep18` 과 같은 run 의 나중 판 — 구조 · 채점 경로 · 단서는 그 항목과 같다 (블록별 target encoder, rollout, 쌍 정확도만 비교).
⚠️ 파일 이름이 `latest.pt` 라 다음 수령 때 또 덮인다 — 실행 전에 epoch 를 확인할 것.

arch_name: vit_huge
checkpoint: /data/hyuntak/project/2026/2027_cvpr/vjepa2/checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth
predictor_checkpoint: /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_training/ariel/ar_future_1e_5/latest.pt
predictor.kind: ar

## vith_ariel_ar_ep38

**팔 C — AR, epoch 38 / 40** (2026-09-28 20:19 수령, step 34048, loss 1.044). `latest.pt` 가 계속 덮이므로
**고정 사본** `z_training/ariel_eval/ckpt/ar_ep38.pt` 를 가리킨다 (평가 도중 덮여도 안전). 구조 · 단서는 `vith_ariel_ar_ep18` 과 같다.

arch_name: vit_huge
checkpoint: /data/hyuntak/project/2026/2027_cvpr/vjepa2/checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth
predictor_checkpoint: /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_training/ariel_eval/ckpt/ar_ep38.pt
predictor.kind: ar

## vith_ariel_ctx_ar_ep18

**팔 C' — ctx_ar (문맥은 창 단위 z, 미래만 블록별 되먹임), epoch 18 / 40** (2026-09-29 21:37 수령 `context_full_ar_1e_5/latest.pt`, step 16128, loss 1.0635 = tf + rollout; **학습 중**).
**고정 사본** `z_training/ariel_eval/ckpt/ctx_ar_ep18.pt`. 문맥 = online `encoder` 가 창의 앞 C 블록을 한 번에 인코딩한 z (prefix 팔과 같다), 미래 타깃 = 블록별 LN(target_encoder), rollout = z 뒤에 자기 예측을 붙여 K 블록. type embedding 2 개 (`type_embed`) 가 state_dict 에 더 있다.
채점 경로 `analysis/predictors/ar_scoring.py:ctx_ar_from_blocks / ctx_ar_window` (2026-09-29 추가). ⚠️ 타깃이 블록별이라 다른 kind 와 **쌍 정확도만** 비교 (AR 과 같은 공간). probing · readout 미정의. 장면 판독은 `auto_research/scripts/e_scene_ar.py` (`SCENE_AR_EXTRA=CX=<ckpt>`).

arch_name: vit_huge
checkpoint: /data/hyuntak/project/2026/2027_cvpr/vjepa2/checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth
predictor_checkpoint: /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_training/ariel_eval/ckpt/ctx_ar_ep18.pt
predictor.kind: ctx_ar

## vjepa21g

**V-JEPA 2.1 ViT-g** (증류 아님, 2B). encoder 1408 x depth 40 + `norms_block` 4 (Deep Self-Supervision),
predictor depth 24 / mask token 8, 예측 공간 **5632 = 4 x 1408**.
모델 고유값(predictor 기하·uniform_power 등)은 **로더가 들고 있다** (`analysis/model_loaders.py`) —
여기서는 family / 체크포인트 / 창 기하만 준다.
타깃 LN 은 어댑터가 1408 씩 네 토막으로 걸므로 채점기 쪽 LN 은 끈다.
⚠️ 공식 해상도는 **384** 인데 **256 으로 잰다** (세 모델을 같은 토큰 격자에 맞추려고).
   IntPhys1 에서 384 로도 재봤고 **53.33 vs 52.22** 로 거의 같아 해상도 탓이 아님을 확인했다 (2026-09-21).

family: vjepa2_1
arch_name: vit_giant
checkpoint: /data/hyuntak/project/2026/2027_cvpr/vjepa2/checkpoint/v_jepa2.1/vjepa2_1_vitg_384.pt
img_size: 256
patch_size: 16
tubelet_size: 2
context_encoder_key: encoder
target_encoder_key: target_encoder
surprise.target_layer_norm: false

## videomae2g

**VideoMAEv2-g** (encoder + MAE decoder, 1B). 예측 공간이 **정규화 픽셀 1176 = 3 x 2 x 14²** 이라
**surprise 절대값을 V-JEPA 와 비교하면 안 된다** — 쌍 판별 정확도만 비교 가능하다.
타깃이 픽셀이라 채점기 LN 을 끈다. 디코더 깊이 4 는 로더가 박아 둔다
(기본값으로 두면 뒷블록이 랜덤인 채 조용히 돈다 — 2026-09-20 에 당했다).
⚠️ **RoPE 가 없다.** 위치 임베딩이 16 프레임짜리 고정 sinusoid 표라 **창이 16 을 넘으면 외삽**이다.
   patch 14 라 해상도는 **224 고정** (256 은 격자가 안 떨어진다).

family: videomae2
arch_name: vit_giant_patch14
# window_size 는 **프로토콜이 정한다** (고정 금지). VideoMAEv2 의 시간 위치
# 임베딩은 체크포인트에 없고 `get_sinusoid_encoding_table` 로 **생성**되는 값이라
# (VideoMAEv2/models/modeling_pretrain.py) 32 프레임도 돈다. 논문도 [16,32] 를
# 탐색했고 Table S3 에서 16 을 **골랐을 뿐**이다. 2026-09-21 정정.
checkpoint: /data/hyuntak/project/2026/2027_cvpr/vjepa2/checkpoint/videomae2/models--OpenGVLab--VideoMAE2/snapshots/706cc172d65ebd4dedbee3f9c0183a93df9fa125/mae-g/vit_g_hybrid_pt_1200e.pth
img_size: 224
patch_size: 14
tubelet_size: 2
surprise.target_layer_norm: false

## dinof_highres

**DINO-Foresight** (Karypidis et al., NeurIPS 2025) — frozen DINOv2 ViT-B/14-reg 특징 (4 층 concat → PCA 1152) 을 **프레임 단위**로 예측하는
MaskTransformer (12 층 · 1152 · 분리 attention). 문맥 **4 프레임 고정** + 마스크 1 프레임 (sequence 5), 긴 미래는 자기회귀 unroll.
Cityscapes 448×896 학습. 레포 `/data/hyuntak/project/2026/2027_cvpr/DINO-Foresight` (`z_smoke/README.md`). 로더 `analysis/model_loaders.py` `build_dinof` (2026-10-02).
⚠️ 예측 공간이 PCA-1152 (표준화 완료) 라 채점기 LN 을 끄고, **surprise 절대값은 V-JEPA 와 비교 금지** — 쌍 정확도만.
⚠️ 정방형 448 입력: w 위치 임베딩 64 → 32 bilinear 보간 (레포 high_res_adapt 와 같은 연산). 학습 분포 (주행 영상, 2:1) 와 다르다.
⚠️ **C=4 만 받는다.** intphys1 격자는 `context_mult` 를 C=4 가 되게 주고 (w5: [13] · w8: [8] · w16: [4]) window 마다 실행을 나눈다.
   `tubelet_size: 1` 이라 프레임 = 토큰 시간 단위.

family: dinof
arch_name: dinof_vitb14
checkpoint: /data/hyuntak/project/2026/2027_cvpr/DINO-Foresight/checkpoints/dinof_highres.ckpt
img_size: 448
patch_size: 14
tubelet_size: 1
surprise.target_layer_norm: false

## dinof_lowres

위와 같고 1 단계 (224×448 학습) 판. 정방형 224 → w 위치 임베딩 32 → 16 보간.

family: dinof
arch_name: dinof_vitb14
checkpoint: /data/hyuntak/project/2026/2027_cvpr/DINO-Foresight/checkpoints/dinof_lowres.ckpt
img_size: 224
patch_size: 14
tubelet_size: 1
surprise.target_layer_norm: false

## vittiny_k400

**V-JEPA 2 계열 ViT-tiny, K400 자체 사전학습** (jongseo, 2026-09-17 실험의 30k step).
`family: vjepa2` 그대로 — 구조가 릴리즈 ViT-H 와 같고 크기만 작다 (dim 192 / 12 block / 332 MB).

체크포인트에서 실측한 제원 (`params-pretrain.yaml` 과 일치):

| 항목 | 값 | 근거 |
|---|---|---|
| encoder | dim **192**, 12 block, mlp 4x, patch 16, tubelet 2 | `patch_embed.proj.weight (192,3,2,16,16)`, `mlp.fc1 (768,192)` |
| predictor | `predictor_embed 192->384`, depth 12, `predictor_proj 384->192` | 체크포인트 + config `pred_depth 12 / pred_embed_dim 384` |
| **mask token** | **2 개인데 `[0]` 만 학습됐다** (std 0.0092 vs `[1]` 0.0) | 프로토콜의 `mask_index: 0` 이 맞다 |
| 위치 인코딩 | **RoPE** (`pos_embed` 키 없음) | 창 길이를 바꿔도 된다 |
| `uniform_power` | **true** ⚠️ ViT-H 는 false 다 | config `uniform_power: true` |
| 학습 해상도 | **256** | config `crop_size: 256` |
| 학습 fps | 4 | config `fps: 4` |

⚠️ **ViT-H 와 같은 자로 비교하되, 사전학습 데이터·규모가 다르다** — K400 30k step 이고
릴리즈 ViT-H 는 훨씬 큰 데이터다. 낮게 나오는 것이 정상 기대값이다.

family: vjepa2
arch_name: vit_tiny
checkpoint: /data/jongseo/project/world-model/vjepa2-pretrain/exp/2026-09-17_k400-vittiny-steps/out_30k/latest.pt
img_size: 256
patch_size: 16
tubelet_size: 2
uniform_power: true
use_rope: true
dual_encoder: true
context_encoder_key: encoder
target_encoder_key: target_encoder
predictor.embed_dim: 384
predictor.depth: 12
predictor.num_heads: 12
predictor.num_mask_tokens: 2

---

**jongseo 자체 사전학습 (2026-09-25 등록)** — encoder·predictor 를 전부 새로 학습한 V-JEPA 2 계열.
구조 제원은 `vittiny_k400` 과 같다 (RoPE · `uniform_power: true` · predictor 384/12/12 · mask token 2 개 중 `[0]` 만 학습).
학습 입력은 16f @ 4fps 라 우리 프로토콜 (v11 stride 3, IntPhys1 skip 2/5/10) 과 시간 해상도가 다르다 — 비교할 때 밝힐 것.

## vitb_k400_e240

**ViT-B/16, K400 단독 자체 사전학습 (jongseo), epoch 240 ≈ 72K step — 이 run 의 최종** (90K 스케줄을 14일 벽에서 멈춤, loss 0.574).
abbreviated recipe (논문 Table 10), 16f @ 4fps, 256, batch 3072. encoder·predictor 전부 새로 학습. `latest.pt` 의 복사본 `snap_ep240.pt` 를 가리킨다.

arch_name: vit_base
checkpoint: /data/jongseo/project/world-model/vjepa2-pretrain/exp/2026-08-31_k400-vitb-abbrev/out/snap_ep240.pt
family: vjepa2
img_size: 256
patch_size: 16
tubelet_size: 2
uniform_power: true
use_rope: true
dual_encoder: true
context_encoder_key: encoder
target_encoder_key: target_encoder
predictor.embed_dim: 384
predictor.depth: 12
predictor.num_heads: 12
predictor.num_mask_tokens: 2

## vitb_k400_e100

**같은 K400 ViT-B run 의 epoch 100 ≈ 30K step** — lr 이 아직 안 내려간 중간 체크포인트 (하한). `vitb_synphys_30k` 와 step 수가 같은 짝.

arch_name: vit_base
checkpoint: /data/jongseo/project/world-model/vjepa2-pretrain/exp/2026-08-31_k400-vitb-abbrev/out/e100.pt
family: vjepa2
img_size: 256
patch_size: 16
tubelet_size: 2
uniform_power: true
use_rope: true
dual_encoder: true
context_encoder_key: encoder
target_encoder_key: target_encoder
predictor.embed_dim: 384
predictor.depth: 12
predictor.num_heads: 12
predictor.num_mask_tokens: 2

## vitb_synphys_30k

**ViT-B/16, 합성 물리 영상 50K clip (synphys 2D pymunk 30K + 3D pybullet 20K) 만으로 30K step 완주** (jongseo, loss 0.505).
자연 영상 0개. 레시피는 K400 ViT-B 와 같고 step 만 30K. 원본 `vjepa2-pretrain/exp/2026-09-14_synphys-vitb-30k/out/latest.pt` 의 고정 사본.

arch_name: vit_base
checkpoint: /data/jongseo/project/world-model-interpret/checkpoints_pretrain_synphys/vjepa2_vitb16_synphys50k_30ksteps_ep100_full.pt
family: vjepa2
img_size: 256
patch_size: 16
tubelet_size: 2
uniform_power: true
use_rope: true
dual_encoder: true
context_encoder_key: encoder
target_encoder_key: target_encoder
predictor.embed_dim: 384
predictor.depth: 12
predictor.num_heads: 12
predictor.num_mask_tokens: 2

## vittiny_k400_5k

**ViT-Tiny/16, K400 5K step 완주** (loss 0.563). `vittiny_synphys_5k` 와 **데이터만 다른 정확한 짝**.

arch_name: vit_tiny
checkpoint: /data/jongseo/project/world-model-interpret/checkpoints_pretrain_synphys/vjepa2_vittiny16_k400_5ksteps_ep50_full.pt
family: vjepa2
img_size: 256
patch_size: 16
tubelet_size: 2
uniform_power: true
use_rope: true
dual_encoder: true
context_encoder_key: encoder
target_encoder_key: target_encoder
predictor.embed_dim: 384
predictor.depth: 12
predictor.num_heads: 12
predictor.num_mask_tokens: 2

## vittiny_synphys_5k

**ViT-Tiny/16, synphys 50K 5K step 완주** (loss 0.285). `vittiny_k400_5k` 와 데이터만 다르다.

arch_name: vit_tiny
checkpoint: /data/jongseo/project/world-model-interpret/checkpoints_pretrain_synphys/vjepa2_vittiny16_synphys50k_5ksteps_ep50_full.pt
family: vjepa2
img_size: 256
patch_size: 16
tubelet_size: 2
uniform_power: true
use_rope: true
dual_encoder: true
context_encoder_key: encoder
target_encoder_key: target_encoder
predictor.embed_dim: 384
predictor.depth: 12
predictor.num_heads: 12
predictor.num_mask_tokens: 2

## vittiny_synphys_30k_e225

**ViT-Tiny/16, synphys 30K step run 의 epoch 225 ≈ 22.5K step** — ⚠️ 학습 중 (2026-09-25 기준 236/300), lr 미감소 중간값. 완주하면 `latest.pt` 로 새 섹션을 만든다.

arch_name: vit_tiny
checkpoint: /data/jongseo/project/world-model/vjepa2-pretrain/exp/2026-09-17_synphys-vittiny-30k/out_30k/e225.pt
family: vjepa2
img_size: 256
patch_size: 16
tubelet_size: 2
uniform_power: true
use_rope: true
dual_encoder: true
context_encoder_key: encoder
target_encoder_key: target_encoder
predictor.embed_dim: 384
predictor.depth: 12
predictor.num_heads: 12
predictor.num_mask_tokens: 2

## vittiny_k400ssv2_30k_e225

**ViT-Tiny/16, K400 0.77 : SSv2 0.23 혼합 30K step run 의 epoch 225 ≈ 22.5K step** — ⚠️ 학습 중 (234/300), batch 32×5×19=3040. 중간값.

arch_name: vit_tiny
checkpoint: /data/jongseo/project/world-model/vjepa2-pretrain/exp/2026-09-21_k400ssv2-vittiny-30k/out/e225.pt
family: vjepa2
img_size: 256
patch_size: 16
tubelet_size: 2
uniform_power: true
use_rope: true
dual_encoder: true
context_encoder_key: encoder
target_encoder_key: target_encoder
predictor.embed_dim: 384
predictor.depth: 12
predictor.num_heads: 12
predictor.num_mask_tokens: 2

---

**릴리즈 predictor post-FT (z_training, 2026-09-25 등록)** — `z_training/eval.sh` 와 같은 것 (encoder = `vith`,
predictor 만 `predictor_checkpoint`). 창별 실행 (Garrido) 과 분석 스크립트에서 이름으로 부르려고 등록했다. 학습 설정 정본은 `z_training/RESULTS_2026-09-19.md`.

## vith_pft_v11_e10

`v11_postft` e10 — v11_split_train (v11 block 절반, 가능만 10,752 clip), temporal_prefix C16, lr 5e-5, 10 ep.

arch_name: vit_huge
checkpoint: /data/hyuntak/project/2026/2027_cvpr/vjepa2/checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth
predictor_checkpoint: /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_training/runs/v11_postft/e10.pt

## vith_pft_intphys1_e40

`intphys1_postft` e40 — IntPhys1 train 장면 (가능만 ~3,750), window_grid, 40 ep.

arch_name: vit_huge
checkpoint: /data/hyuntak/project/2026/2027_cvpr/vjepa2/checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth
predictor_checkpoint: /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_training/runs/intphys1_postft/e40.pt

## vith_pft_intphys2_e80

`intphys2_postft` e80 — IntPhys2 Main split train (254 clip), 48 프레임, 80 ep.

arch_name: vit_huge
checkpoint: /data/hyuntak/project/2026/2027_cvpr/vjepa2/checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth
predictor_checkpoint: /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_training/runs/intphys2_postft/e80.pt

## vith_pft_predv1_e15

`predictor_v1_postft` e15 — Predictor_v1_training (14,250 clip), lr 5e-5, 15 ep.

arch_name: vit_huge
checkpoint: /data/hyuntak/project/2026/2027_cvpr/vjepa2/checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth
predictor_checkpoint: /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_training/runs/predictor_v1_postft/e15.pt
