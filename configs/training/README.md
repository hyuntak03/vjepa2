# configs/training — frozen encoder + predictor 학습 설정

> **학습 데이터는 미정이다.** `data.datasets` 는 비어 있고 돌릴 때 `SET="data.datasets=[이름]"` 으로 준다.

```bash
bash z_training/train.sh --list                              # config / 데이터셋 후보 / run 목록
DRYRUN=1 SET="data.datasets=[<이름>]" bash z_training/train.sh frozen_predictor_scratch   # 병합·검사만 (몇 초)
GPUS=8 SET="data.datasets=[<이름>]" bash z_training/train.sh frozen_predictor_scratch     # 학습
GPUS=8 SET="data.datasets=[<이름>]" bash z_training/train.sh frozen_predictor_postft
```

| 파일 | 무엇 |
|---|---|
| `_base_frozen_predictor.yaml` | 공통. 직접 돌리지 않는다. 채점 프로토콜(`surprise_c16t32`)과 맞춘 값이 여기 있다 |
| `frozen_predictor_scratch.yaml` | predictor **새로 초기화** (`load_predictor: false`) |
| `frozen_predictor_postft.yaml` | 릴리즈 predictor **이어 학습** (`load_predictor: true`, lr 1/10) |
| `debug.yaml` | 배관 점검 (1 GPU, 16 clip, 2 epoch × 4 step) |
| `datasets.md` | 학습 데이터 레지스트리 (`data.datasets: [이름]`) |

병합은 `z_training/harness/resolve_train.py` 가 한다: `extends:` 재귀(자식이 이김, dict 재귀 병합,
list 교체, `null` 은 키 삭제) → `data.datasets` 이름을 `datasets.md` 로, `model.base` 를
`configs/protocols/models.md` 로, `val.dataset` 을 `configs/protocols/datasets.md` 로 푼다 →
`--set a.b=값` → 실물 검사 → `<run>/config.yaml`. **실험마다 yaml 을 새로 뜨지 않는다** —
`SET=` 과 `extends:` 로 처리한다 (`configs/protocols/README.md` 와 같은 규칙).

## 스키마 (`app/vjepa_frozen/train.py` 가 읽는 키)

```yaml
app: vjepa_frozen
meta:
  seed: 0
  dtype: bfloat16             # autocast. predictor 마스터는 fp32. float16 이면 GradScaler
  encoder_dtype: bfloat16     # 동결 encoder 가중치 dtype
  use_sdpa: true
  log_freq: 10                # step
  save_every_freq: 5          # epoch 마다 e{N}.pt 추가 저장 (-1 = latest.pt 만)
  auto_resume: true           # <run>/latest.pt 가 있으면 이어서. false 인데 있으면 죽는다
  sync_gc: false
model:
  base: vith                  # models.md 이름 -> checkpoint / arch_name
  img_size: 256  patch_size: 16  tubelet_size: 2  use_rope: true  uniform_power: false
  context_encoder_key: encoder          # 문맥 <- online encoder   (surprise_c16t32 와 같다)
  target_encoder_key: target_encoder    # 타깃 <- EMA encoder
  load_predictor: false       # true = 릴리즈 predictor 를 strict 로 로드해 시작
  mask_index: 0               # 릴리즈 ckpt 에서 학습된 mask token 은 [0] 뿐
  predictor: {embed_dim: 384, depth: 12, num_heads: 12, num_mask_tokens: 10,
              use_activation_checkpointing: false, zero_init_mask_tokens: true}
data:
  n_frames: 32  resolution: 256           # resolution == img_size 필수
  batch_size: 8                           # ★ rank 당 (global = × GPUS)
  num_workers: 6  pin_mem: true  persistent_workers: true
  datasets: []                            # ★ 미정. 이름 | {name: 이름, limit: N, weight: w, include: {...}, exclude: {...}} | 완전한 dict
  aug: {hflip_p: 0.5, random_resized_crop: null}    # rrc 예) {scale: [0.7, 1.0], ratio: [0.9, 1.1]}
mask:                                     # 배치마다 weight 로 하나를 고른다
  - {type: temporal_prefix, context_frames: [16], weight: 1.0}   # 앞 C -> 뒤 32-C. C 를 여러 개 주면 배치마다 무작위
  - {type: block3d, spatial_scale: [0.15, 0.15], temporal_scale: [1.0, 1.0], aspect_ratio: [0.75, 1.5],
     num_blocks: 8, max_temporal_keep: 1.0, weight: 0.5}          # 릴리즈 사전학습 마스크
  window_grid:                            # (선택) 채점 sliding 그리드를 학습 샘플 공간으로. 있으면 mask.context_frames 는 무시
    raw_span: 99                          # 데이터셋이 읽어 둘 연속 raw 프레임 수 (마지막 창 끝 이상. resolve 가 검사)
    start_step: 2  frame_budget: official  jitter_raw: 0  weights: auto     # auto = 셀의 (시작점 x C) 수 비례 | [w, ...]
    cells:                                # 배치마다 셀 -> 시작점 -> C 를 뽑아 raw 버퍼를 자른다 (worker 안에서)
      - {stride: 2, n_frames: 32, context_frames: [4, 8, 12, 16, 20]}
      - {stride: 2, n_frames: 16, context_frames: [2, 4, 6, 8, 10], starts: [0, 4, 8]}   # starts 를 주면 raw 오프셋 고정
loss: {loss_exp: 1.0, target_layer_norm: true}
optimization:
  epochs: 30  ipe: null  ipe_scale: 1.0
  lr: 3.0e-4  start_lr: 1.0e-5  final_lr: 1.0e-6  warmup: 2      # warmup 은 epoch 단위, 스케줄 warmup-cosine
  weight_decay: 0.04  final_weight_decay: 0.04  betas: [0.9, 0.999]  eps: 1.0e-8  clip_grad: null
val:                                      # in-loop 감시. 정식 수치는 z_training/eval.sh
  dataset: intphys1_dev                   # protocols/datasets.md 이름 (PNG 직독 세트만)
  n_blocks_per_type: 10  context_length: 16  batch_size: 8  every_epochs: 1  at_start: true  seed: 0
```

### 학습 손실 = 채점 식

```
h = LN(target_encoder(clip))[masks_pred]          # affine-free LN, target 에만
z = context_encoder(clip, masks=[masks_enc])      # 문맥 토큰만 (미래 토큰은 transformer 앞에서 떨군다)
p = predictor(z, masks_enc, masks_pred, mask_index)
loss = mean |p − h|^loss_exp / loss_exp           # fp32
```
`temporal_prefix` 마스크에서 `masks_enc = [0, C/2·256)`, `masks_pred = [C/2·256, 16·256)` 이고
이것이 `analysis/intphys2/surprise._context_target_indices` 와 같다.

### 자주 쓰는 변형

```bash
# 문맥 길이를 8~24 에서 섞어 학습 (sliding 채점 대비)
SET="mask.0.context_frames=[8,12,16,20,24]" GPUS=8 bash z_training/train.sh frozen_predictor_scratch
# 릴리즈 마스크와 섞기
SET='mask=[{"type":"temporal_prefix","context_frames":[16],"weight":0.5},{"type":"block3d","weight":0.5}]' ...
# 데이터 바꾸기 / 조건 빼기 / 가중 혼합
SET='data.datasets=[v11_possible] data.datasets.0.exclude={"condition":["static_occlusion"]}' ...
SET="data.datasets=[intphys1_train,rollout_v2_training] data.datasets.0.weight=0.8 data.datasets.1.weight=0.2" ...
# val 을 v11 로 (본 채점 세트라 감시용으로만)
SET="val.dataset=v11 val.n_blocks_per_type=8" ...
```

⚠️ **SET 값의 dict/list 는 JSON 식으로** (`{"a":"b"}`, 공백 없이). YAML flow 의 `{a:b}` 는 `{'a:b': None}` 으로 읽힌다.
⚠️ **encoder 는 bf16 가중치 + bf16 autocast 로 돈다** (메모리·속도). 채점기는 fp32 가중치 + fp16 autocast 라 in-loop val 값이
채점기 값과 마지막 자리에서 다를 수 있다. 같은 수치 경로가 필요하면 `meta.encoder_dtype: float32 meta.dtype: float16`.
⚠️ **lr / epoch 은 전부 미검증 초기값이다.** 릴리즈 사전학습(encoder 포함, global batch 3072)의
5.25e-4 를 참고했을 뿐이다. 첫 run 의 `metrics.jsonl` 로 잡을 것.
