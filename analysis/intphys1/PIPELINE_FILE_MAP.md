# IntPhys 1 Linear Probing (5-fold CV) — File Map for Debugging

L0 acc 이 예상보다 높은 이유 등 debug 시 참고용. 실행 흐름 순서대로 나열.

경로는 all absolute (`/data/hyuntak/project/2026/2027_cvpr/vjepa2/` prefix 생략 = repo root).

---

## 1. Entry point (launcher → harness routing)

| 파일 | 역할 | 주요 라인 |
|---|---|---|
| `z_scripts/run_5fold.sh` | 두 protocol × 5 fold sed 치환 후 evals.main 호출 | fold loop line 61-79 |
| `evals/main.py` | CLI argparser + config load + `process_main` → mp.Process | `process_main`, `--debugmode True` 시 spawn 안 함 |
| `evals/scaffold.py` | eval_name → `evals.<eval_name>.eval` import 라우팅 | `main()` |
| `evals/analysis_vlm/eval.py` | **본 harness**: setup / cache / train loop / summary write | `main()`, `run_one_epoch()`, `_init_opt_fused()`, `_encode()` |

### VS Code 디버그 진입점

`.vscode/launch.json` 등록된 것들:
- `tak_intphys1_linear_probe_debug` — 이전 sweep config (24×24=576 heads)
- **`tak_intphys1_5fold_appb_debug (fold 0)`** — 5-fold appb fold 0 (20 HP × 24 layer)
- **`tak_intphys1_5fold_c11_debug (fold 0)`** — 5-fold c11 fold 0 (1 HP × 24 layer, 가장 빠름)

---

## 2. Configs (base + fold-specific)

| 파일 | 상태 |
|---|---|
| `configs/analysis/probing/intphys1_vitl_5fold_appb.yaml` | Template (placeholders `__TAG__`, `__TRAIN_CSV__`, `__VAL_CSV__`) |
| `configs/analysis/probing/intphys1_vitl_5fold_c11.yaml` | Template |
| `configs/analysis/probing/intphys1_vitl_5fold_appb_debug_f0.yaml` | **Fold 0 사전 치환됨** (debug 용) |
| `configs/analysis/probing/intphys1_vitl_5fold_c11_debug_f0.yaml` | **Fold 0 사전 치환됨** (debug 용) |

## 3. Data pipeline

### 3.1 CSV (video path + label)

| 파일 | 내용 |
|---|---|
| `data_csv/IntPhys1/IntPhys1_2way_all.csv` | 360 videos (전체) — split builder input |
| `data_csv/IntPhys1/cv/train_fold0.csv` ~ `train_fold4.csv` | 5 fold train (288 videos = 72 quad × 4 runs each) |
| `data_csv/IntPhys1/cv/val_fold0.csv` ~ `val_fold4.csv` | 5 fold val (72 videos = 18 quad × 4) |
| `data_csv/IntPhys1/keystones.json` | 360 video path → magic tick (frame index) 매핑 |

### 3.2 Sampling / decoding

| 파일 | 역할 |
|---|---|
| `src/datasets/video_dataset.py` | VideoDataset. `loadvideo_decord()` 에서 keystone / center / uniform sampling 분기 |
| `src/datasets/data_manager.py` | `init_data()` wrapper, keystones_by_path pass-through |
| `evals/analysis_vlm/eval.py::_split_loader` | keystones_json 로드 후 dataset 생성 |

**Sampling precedence (video_dataset.py:loadvideo_decord)**:
```
if keystones_by_path and sample in keystones_by_path:
    # window = [tick - fpc//2 : tick + fpc//2] (clipped to [0, n])
elif center_sampling:
    # window = [(n-fpc)//2 : (n+fpc)//2]
elif uniform_sampling:
    # np.linspace(0, n-1, fpc)
else:
    # standard contiguous window (frame_step 기반)
```

### 3.3 CSV split & keystone builders (one-time)

| 파일 | 역할 |
|---|---|
| `z_scripts/build_cv_splits.py` | 90 quad → 5 fold grouped split. 이미 실행됨, idempotent |
| `analysis/intphys1/build_keystones.py` | keystones.csv + md5 pair recovery → keystones.json. 이미 실행됨 |

---

## 4. Feature extraction (encoder wrapper)

| 파일 | 역할 |
|---|---|
| `evals/analysis/modelcustom/vit_encoder_multilayer.py` | **본 wrapper** — forward hook 으로 각 block N 의 raw x_N 캡처. `self.norm` 은 적용 안 함 |
| `src/models/vision_transformer.py` | Base ViT-L (24 blocks + patch_embed + self.norm). out_layers=None 로 초기화됨 (라인 205-206 skip) |
| Checkpoint | `checkpoint/models--facebook--vjepa2-vitl-fpc64-256/snapshots/b3c1679b7c34d3255ef3547f27c7b226aefab26f/original/model.pth` (`target_encoder` key) |

### Key debug points in wrapper

- `init_module()` line 44-95: checkpoint 로드 + `model = MultiLayerClipAggregation(...)` 생성
- `MultiLayerClipAggregation.__init__` line 120-133: 각 `model.blocks[idx]` 에 `register_forward_hook`
- `MultiLayerClipAggregation.forward` line 160-190:
  - `self._captured = []` reset
  - `self.model(x)` 실행 → hook 이 각 block output 캡처
  - `assert len(self._captured) == len(self.out_layers)`
  - out_layers 순서로 reorder → `layer_outputs`

**L0 debug 시 breakpoint 우선순위**:
1. `MultiLayerClipAggregation.forward` — `layer_outputs[0].shape` = `(B, N=2048, D=1024)` 확인
2. `layer_outputs[0].mean(), std(), max()` — L0 raw x_0 통계 정상인지
3. hook 이 model.blocks[0] 에 걸린 게 맞는지 (self._hook_handles 조사)

---

## 5. Feature caching + probe

| 파일 | 역할 |
|---|---|
| `evals/analysis_vlm/cache.py` | `build_feature_cache()`, `reduce_feature()`, `PooledLinearProbe`, `CachedTensorDataset`, `make_cached_loader` |

### Key debug points

- `reduce_feature(feat, mode='mean')` line 84-98: `feat.mean(dim=1)` → `(B, D)`
- `build_feature_cache()` line 100-164:
  - `cache_dtype = torch.float16 if cache_pooling == "tokens" else torch.float32` (line 145) — mean cache 는 fp32
  - Per-rank 캐시 저장 → `feats_cat` list of `(n_local, D)` fp32
- `PooledLinearProbe.forward(x)` line 230-251:
  - Cache 는 이미 mean 됨. `x.shape = (B, D)`. `z = x` 그대로 → `self.linear(z)` → `(B, num_classes=2)` logits

---

## 6. Training loop + probe wiring

| 파일 | 역할 |
|---|---|
| `evals/analysis_vlm/eval.py::main()` | Setup: encoder init, cache pre-pass, head 생성, optimizer/scheduler |
| `evals/analysis_vlm/eval.py::run_one_epoch()` | Train/val loop |
| `evals/analysis_vlm/eval.py::_init_opt_fused()` | 하나의 AdamW 로 N heads (per-head param_group) |
| `evals/analysis_vlm/eval.py::_encode()` | 캐시 로드 mode → cached tensor 를 fp32 로 뽑아 return |

### Key debug points (run_one_epoch)

- Line ~731: `feats, labels, bsz = _encode(...)` — `feats` 는 list len=24, 각각 `(B, D)`
- Line ~739: `preds = [h["module"](feats[h["layer_pos"]]) for h in heads]`
  - `heads[0]["layer_pos"] == 0` — L0 첫 HP head 가 실제로 `feats[0]` (L0) 를 받나?
- Line ~752: `losses = [criterion(p, labels) for p in preds]` — CE per head
- Line ~755: `loss_total = sum(losses)`
- Line ~762: `loss_total.backward()`
- Line ~782: `correct[hi] += (p.argmax(dim=1) == labels).sum()` — accuracy 계산

**L0 acc 이 chance 이상인지 debug 하려면**:
- `heads[0]` 의 layer_pos 확인 (L0 여야 함)
- `feats[0]` shape/값 sane 한지 (raw x_0 특성: norm ~250)
- `preds[0]` 값이 saturate 안 됐는지
- `labels` 값 = 0/1 balanced 인지 (0=possible, 1=impossible)

---

## 7. Output directories (per-fold)

기본 folder = `configs/analysis/probing/logs/analysis_vlm/`

| Tag | 내용 |
|---|---|
| `probe_intphys1_vitl_5fold_appb_f{0..4}/` | 5 fold appb 결과 (완료됨) |
| `probe_intphys1_vitl_5fold_appb_agg/` | Aggregated (`aggregated.json`, `per_layer_best_hp.csv`, `stage_val_acc_mean_std.png`) |
| `probe_intphys1_vitl_5fold_c11_f{0..4}/` | **미실행** |
| `probe_intphys1_vitl_5fold_c11_agg/` | **미실행** |
| `probe_intphys1_vitl_5fold_appb_debug_f0/` | debug 실행 시 생성될 폴더 |
| `probe_intphys1_vitl_5fold_c11_debug_f0/` | debug 실행 시 생성될 폴더 |

각 fold 폴더 내부:
- `summary.json` — 24 layer × N HP head 별 `best_val_acc`, `head_names`
- `log_r0.csv` — epoch 별 per-head train/val acc
- `latest.pt` — probe weights
- `stage_val_acc.png` — 그 fold 의 layer-wise val acc plot

---

## 8. Aggregation + plotting

| 파일 | 역할 |
|---|---|
| `z_scripts/aggregate_5fold.py` | 5 fold summary.json 병합 → per-layer mean±std |
| `evals/analysis/plotting.py` | Per-run stage_val_acc plot (`plot_layer_val_acc`) |
| `z_scripts/plot_training_curves.py` | Peak layer 의 epoch × train/val curve |
| `z_scripts/regenerate_stage_plot.py` | summary.json 재활용 plot 재생성 |

---

## 9. Debugging L0 acc 58% (이상히 높음) — checklist

**Paper Fig 6 시각 판독: V-JEPA 2 L L0 ≈ 45-50%.**

가능성 순서:

1. **Val split imbalance** — 우리 fold 0 val 은 balanced (36/36 확인됨). 그런데 test 시 특정 HP 가 systematic bias 만들 수도 있음
2. **Sampling artifact** — keystone-centered clip 이 L0 raw patch 에도 breakpoint 근방 dense 정보를 담아서 low-level pixel 통계 만으로 45% 이상 뽑을 가능성
   - Uniform sampling 으로 바꿔서 L0 이 chance 로 내려가는지 확인 (video_dataset.py 의 sampling flag 조작)
3. **Layer 0 정의 차이** — 우리 L0 = model.blocks[0] output. 논문 L0 이 patch_embed output (block 0 전) 일 가능성 → 그러면 논문 L0 은 우리 표시 X=−0.04 정도 위치. 우리 L0 이 논문 L1 에 해당하면 45~55% 근처 가능
4. **HP grid 극단값** — best_hp=18 (lr=0.005, wd=0.1) 이 L0 에서 winning. 이 조합이 특정 val subset 에 overfit → 논문의 fold-mean-per-HP 는 다른 규칙일 수 있음
5. **Cache dtype / precision loss 없음** — fp32 유지 확인됨

**추천 첫 debug step**:
- Launch `tak_intphys1_5fold_c11_debug (fold 0)` (24 head 만 있어 훨씬 빠름)
- Breakpoint: `MultiLayerClipAggregation.forward` 의 `layer_outputs[0]`
- L0 shape / stats 확인 → cache 저장 후 `feats[0].shape=(B, 1024)` 확인
- probe forward 후 첫 iter 의 `preds[0]` 값 확인 (logit distribution)

---

## 10. 관련 미실행 protocol

- **c11 5-fold**: `intphys1_vitl_5fold_c11.yaml` 로 준비돼 있지만 실행 안 됨. 실행 명령:
  ```bash
  bash /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_scripts/run_5fold.sh c11
  ```
- Config 는 **num_epochs 50 (paper C.11 verbatim)** 이라 빠르게 완료됨 (예상 10-15 분 6-GPU)

---

## 요약: 파일 조작 시 우선 순위

**Debug 시 손봐야 할 파일 (수정 우선 순위)**:
1. **wrapper**: `evals/analysis/modelcustom/vit_encoder_multilayer.py` — feature 캡처
2. **cache**: `evals/analysis_vlm/cache.py::PooledLinearProbe.forward` — probe 자체
3. **eval loop**: `evals/analysis_vlm/eval.py::run_one_epoch` — head wiring, loss, acc
4. **dataset**: `src/datasets/video_dataset.py::loadvideo_decord` — keystone / sampling 확인
