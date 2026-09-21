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
checkpoint: /data/hyuntak/project/2026/2027_cvpr/vjepa2/checkpoint/videomae2/models--OpenGVLab--VideoMAE2/snapshots/706cc172d65ebd4dedbee3f9c0183a93df9fa125/mae-g/vit_g_hybrid_pt_1200e.pth
img_size: 224
patch_size: 14
tubelet_size: 2
surprise.target_layer_norm: false
