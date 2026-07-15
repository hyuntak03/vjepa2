# IntPhys 1 Linear Probing — Handoff Spec

Complete specification for reproducing our V-JEPA 2 ViT-L linear probing pipeline
on IntPhys 1 (Joseph 2026-style). Paste this to another agent as the source of
truth. Every non-obvious detail below was learned through iteration — the
one-liner "forward hook 걸어서 per-layer linear probing" version is NOT enough
to reproduce.

**Env:** `vjepa2` conda env (torch, decord, timm, sklearn optional).

---

## 0. TL;DR spec (what a fresh agent must NOT skip)

- **16 frames per clip, KEYSTONE-CENTERED** (from precomputed `keystones.json`) —
  NOT uniform, NOT center-of-video. Breakpoint tick range is 17-89 so center
  misses most videos.
- **Forward hooks on `model.blocks[idx]`** — NOT `out_layers=[...]` on the ViT
  constructor (that applies `self.norm` per intermediate, which is a
  concat-wrapper convention, not per-layer probing).
- **Single `nn.Linear(1024, 2)` per (layer, HP)** with `CrossEntropyLoss`. No
  LayerNorm inside the probe head. `num_classes=2` softmax-CE = equivalent
  to sigmoid-BCE for binary logistic regression.
- **fp32 cache** for mean-pooled features. Do NOT downcast to fp16.
- **`use_bfloat16: false`**. The flag maps to fp16 autocast (misleading name); with
  a fused sum-of-losses backward this has a GradScaler bug that freezes ALL
  probes if any single probe's loss diverges.
- **5-fold grouped CV, grouping by QUADRUPLET** (not by matched pair, not
  random). 90 quads / 5 folds = 18 val quads = 72 val videos per fold.
- **Constant lr / wd, NO warmup, NO cosine schedule.**

---

## 1. Task & data

### Dataset
- **IntPhys 1 dev** (Riochet 2021): 360 videos = 90 quadruplets × 4 runs.
- Each quadruplet contains 2 possible + 2 impossible runs (matched pair × 2).
- Video source: `/local_datasets/world/IntPhys1_dev_videos/scene/*.mp4`
  — re-encoded from PNGs: 100 frames @ 15 fps, 288×288, libx264 crf=12, yuv420p.
- Video naming: `O{block}_{quad:02d}_{run}.mp4` (block ∈ {1,2,3}, quad 01–30, run 1–4).

### Labels
- `possible = 0`, `impossible = 1`.
- Ground truth: per-video `is_possible` field in `/local_datasets/world/dev/O*/*/*/status.json`.
- Our CSV encoding: **`<absolute_video_path> <label>`** space-separated, no header.

### Splits (5-fold grouped CV)
- Group key = `O{block}_{quad}` (drops `_{run}` suffix). Total 90 unique groups.
- Deterministic split: sort groups, chunk into 5 × 18. `build_cv_splits.py`.
- Fold N val = groups[18N : 18(N+1)] → 72 videos.
- Fold N train = remaining 72 groups → 288 videos.
- Assertion: every quadruplet appears in val EXACTLY once across the 5 folds
  (script verifies).
- **Do NOT split by pair** or by video — pairs of a quadruplet must stay
  together to prevent visual-scene leakage.

**Pre-generated at**: `data_csv/IntPhys1/cv/{train,val}_fold{0..4}.csv`.

---

## 2. Frame sampling (KEYSTONE-CENTERED)

### Keystone metadata
- Per-video "magic tick" = the frame index (0-indexed) where the physics
  violation happens (impossible videos) or the matched-pair partner's tick
  (possible videos).
- Source: `/local_datasets/world/IntPhys1_dev_videos/keystones.csv` (comes with
  the redistributed dev).
- For 2-violation impossible videos (`n_violations == 2`, ticks like `[46, 56]`):
  use the **midpoint** of the two ticks so a 16-frame window covers both.
- For possible videos: inherit the matched impossible's tick. Pair recovery:
  `md5(/local_datasets/world/dev/{block}/{quad}/{run}/scene/scene_001.png)`. Two
  matched runs are byte-identical BEFORE the breakpoint, so the pre-breakpoint
  frame MD5 groups them cleanly.
- Distribution: tick ∈ [17, 89], median 52. Center-of-video sampling
  (`[42:58]`) misses about 60% of breakpoints.

**Pre-built lookup**: `data_csv/IntPhys1/keystones.json` — dict
{`<video_path>`: `<tick>`}. `build_keystones.py` generates it.

### Window
- 16 consecutive frames: `frames[max(0, tick - 8) : min(n, tick + 8)]`, padded
  with last frame if tick near the end.
- Precedence in the video dataset sampling code:
  ```
  keystone-centered (if path in dict)
    > center-of-video (if center_sampling=true)
    > uniform (if uniform_sampling=true)
    > standard contiguous window (default)
  ```

### Preprocessing
- Resize each frame directly to 256×256 (bilinear). IntPhys is 288×288 square
  so no aspect distortion; no shorter-side + center-crop.
- Normalize: ImageNet stats **mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)**.
- Deterministic. **No augmentation** (train and val use identical preprocessing).
- Output tensor to encoder: `(B=8, C=3, T=16, H=256, W=256)` fp32.

---

## 3. Encoder (V-JEPA 2 ViT-L)

### Model spec
- `vit_large` from `src/models/vision_transformer.py`:
  - patch_size=16, tubelet_size=2, embed_dim=1024, depth=24, num_heads=16.
  - `use_rope=True`, `uniform_power=True`, `img_temporal_dim_size=None`.
  - Token count after patch_embed: T=(16 frames / 2 tubelet) × S=(16×16 patches) = **8 × 256 = 2048 tokens** per clip.
- Checkpoint: `target_encoder` (EMA teacher) key from
  `checkpoint/models--facebook--vjepa2-vitl-fpc64-256/snapshots/b3c1679b7c34d3255ef3547f27c7b226aefab26f/original/model.pth`.
- Load with `strict=False`, strip `module.` and `backbone.` prefixes.
- Encoder is **frozen** and in eval mode. Forward under `torch.no_grad()`.

### Feature extraction — CRITICAL
**Do NOT** pass `out_layers=[...]` to the ViT constructor. That takes the
built-in path (`src/models/vision_transformer.py:198-209`) which appends
`self.norm(x)` to every intermediate — designed for the multi-level CONCAT
wrapper (`evals/video_classification_frozen/modelcustom/vit_encoder_multiclip_multilevel.py`),
NOT for per-layer probing. Applying the FINAL LayerNorm (trained for block 24's
distribution) to L0 / L5 / L12 features distorts them (see §7 below).

**Correct path**: build ViT with `out_layers=None` (default), then register
forward hooks on each `model.blocks[idx]` for `idx` in your requested list.
Each hook stores its `output` (= raw residual stream `x_N` after block N).
After forward, sort captures by block execution order to align with the
requested `out_layers` list.

**Sanity smoke test**:
```
apply_encoder_norm=False (hook):
  L00: std≈7.8,  norm_per_token≈250  (raw scale varies per layer)
  L05: std≈10.0, norm_per_token≈322
  L12: std≈7.1,  norm_per_token≈226
  L23: std≈5.1,  norm_per_token≈162

apply_encoder_norm=True (stock code, DO NOT USE):
  L00..L23: std≈2-3, norm_per_token≈60-100  (all layers forced onto block-24 scale)
```

We ONLY use the raw-x_N path. The `apply_encoder_norm` toggle has been
deleted from the codebase; the wrapper is hook-only. Do NOT reintroduce.

### Per-layer output shape
- Each hook capture: `(B_effective, N=2048, D=1024)` where `B_effective` accounts
  for multi-clip / multi-view flattening.
- Wrapper's `multiviews_postprocess` reshapes back to `(B, N, D)`.

---

## 4. Feature caching

- One-shot pre-pass: run encoder over train+val once per (fold, protocol),
  cache **mean-pooled** features on CPU RAM.
- Mean pool = `feat.mean(dim=1)` over all 2048 tokens (spatiotemporal MEAN).
  Result shape: `(N_samples, D=1024)` per layer.
- **`cache_dtype = torch.float32`** for mean/max/pooled caches (fp32).
- Only `cache_pooling='tokens'` uses fp16 (large tensors need memory).
- DDP: each rank pre-passes its own shard; probes train per-rank; metrics
  are all-reduced.
- Order: cache is built with the deterministic (shuffle=False)
  DistributedSampler; `feats_cat[i]` corresponds to `labels[i]` — do NOT
  permute one without the other.

---

## 5. Probe architecture

### Model
```
class PooledLinearProbe(nn.Module):
    def __init__(self, embed_dim, num_classes, pooling="mean"):
        super().__init__()
        self.D = embed_dim
        self.pooling = pooling
        self.linear = nn.Linear(embed_dim, num_classes, bias=True)

    def forward(self, x):
        # x: (B, D) for mean/max cache, (B, 2D) for meanmax cache
        # For our mean cache: x.shape=(B, D), z = x directly.
        z = x
        return self.linear(z)
```

- **Single Linear layer only.** No LayerNorm, no MLP, no dropout, no
  activation before the linear layer.
- `num_classes=2` → softmax + CrossEntropyLoss. Mathematically equivalent to
  binary logistic (single scalar + sigmoid + BCE); just 1 redundant
  parameter, same expressive class.
- Paper form is literally `f(h) = W h + b`. This matches.

### Head wiring
- Total heads = (# layers) × (# HP configs). Each head has:
  - Its own `nn.Linear(D, 2)` (independent params).
  - Its own `layer_pos ∈ {0..23}`.
  - Its own optimizer param-group with its own `lr`, `wd`, `warmup=0`, and
    constant lr/wd (start_lr = final_lr = lr; final_wd = wd).
- 24 layer × 1 HP = 24 heads (C.11 protocol) OR 24 layer × 20 HP = 480 heads
  (Appendix B).

### Fused optimizer & backward
- ONE `torch.optim.AdamW` with N param groups (one per head). Equivalent to N
  independent AdamW's (heads have disjoint params → gradient decomposes
  exactly).
- ONE `loss_total = sum(per_head_losses)`, then `loss_total.backward()`, then
  `optimizer.step()`. Not a bug: the summed backward gives each head exactly
  the gradient it would receive under a per-head backward, because heads
  don't share parameters.
- **NO GradScaler.** `use_bfloat16=false` avoids the fp16 GradScaler + fused
  sum-of-losses bug that can freeze all heads if any single head's loss
  diverges (round-4 audit finding).

---

## 6. Training config

### Two paper-specified protocols

| Setting | **Appendix B** (Fig 1) | **Appendix C.11** (IntPhys spec) |
|---|---|---|
| # HP configs | 20 (5 lr × 4 wd sweep) | 1 (fixed) |
| lr | `{1e-4, 3e-4, 1e-3, 3e-3, 5e-3}` | `1e-3` |
| wd | `{0.01, 0.1, 0.4, 0.8}` | **`1e-4`** (much smaller!) |
| Optimizer | **not specified** (we use AdamW) | Adam (we use AdamW; at wd=1e-4 the diff is negligible) |
| Epochs | **not specified** (we use 50) | **50** |
| Schedule | not specified | none — constant lr/wd |
| Warmup | not specified | 0 |
| Split | 5-fold grouped CV | 80/20 fixed seed (we ADD 5-fold CV for fair comparison) |
| Report | mean ± std across folds | single point |

Both configs share: `batch_size=8`, `use_bfloat16=false`, `cache_features=true`,
`cache_pooling=mean`.

### Constant-schedule gotcha
The harness has a default WarmupCosineLRSchedule + CosineWDSchedule that
would drift lr/wd from the sweep-declared values. Force constants by
setting `start_lr = final_lr = lr`, `final_weight_decay = weight_decay`,
`warmup = 0.0` on EVERY probe. Otherwise the effective lr/wd drift and the
"paper-explicit" values become nominal only.

---

## 7. Evaluation

- **Per-video binary accuracy**: `argmax(logits) == label` averaged over val
  samples. Compare to the paper's y-axis (Fig 1: "Test Accuracy (%)", chance
  = 50%).
- Track best-across-epochs val accuracy per (layer, HP) — that's the number
  saved in `summary.json::best_val_acc`.
- **Per-layer aggregation**: `max over HP` of `mean over folds`. Layer-wise
  peak = the HP that maximizes fold-mean at that layer.
- Report layer-wise (mean ± std across folds) as a shaded band.

---

## 8. Things we tried and REJECTED (in-code, don't reintroduce)

1. **`pre_norm` on probe head** (LayerNorm before the linear layer). Paper is
   explicit `f(h) = W h + b`. Removed from `PooledLinearProbe` entirely; not a
   config knob anymore.
2. **`apply_encoder_norm=True`** (stock `self.norm`-per-intermediate). Argued
   about extensively — it's a concat-wrapper convention, not per-layer probing.
   Removed from config schema entirely; forward-hook-only.
3. **fp16 cache** for mean-pooled features. Removed; mean cache is
   always fp32. Only `cache_pooling='tokens'` uses fp16.
4. **`num_epochs=20`** for the HP sweep. Under-converged for low-lr HPs
   (train acc plateaued at ~60% even for lr=1e-4). Use 50 (paper C.11) or
   more.
5. **Center-of-video 16-frame sampling** as a keystone-approximation. Only
   covers ticks 42–58; tick distribution is 17–89 so it misses ~60% of
   breakpoints. Result: worse than uniform (73.6% vs 79.2% peak). Use
   keystones.json.
6. **Uniform 16-frame sampling** (our first attempt). Peak 79.2%.
   Keystone-centered gets 87.5% (single-fold, 200 epochs). +8 pt.
7. **Pair-level grouping** for 5-fold CV. Splits matched pairs across
   train/val → visual-scene leakage. Use quadruplet-level.
8. **Splitting off the CE loss** to per-video BCE with matched-pair
   ranking. Paper says "binary classification task between matched pairs",
   which is a DATA-structure statement (matched pairs stay together in the
   fold), not a loss modification. Sonia's email confirmed "one video clip"
   as the training input. Keep standard CE, do NOT do pair-diff features
   or margin ranking loss.
9. **Attentive-MLP probe**. Paper reports it too (Fig 1 right panel), but we're
   reproducing the linear-probe left panel. Different feature aggregation.
10. **Multiple views / crops per video**. Paper doesn't do this. Single view,
    single clip, 16 frames.

---

## 9. Key file paths (in this repo)

### Config templates (with `__TAG__`, `__TRAIN_CSV__`, `__VAL_CSV__` placeholders)
- `configs/analysis/probing/intphys1_vitl_5fold_appb.yaml` (20-HP sweep)
- `configs/analysis/probing/intphys1_vitl_5fold_c11.yaml` (single HP, Adam-style)

### Pre-substituted debug (fold 0)
- `configs/analysis/probing/intphys1_vitl_5fold_{appb,c11}_debug_f0.yaml`
  (`num_epochs: 3`, `num_workers: 0` for fast debug iteration)

### Scripts
- `z_scripts/build_cv_splits.py` — one-time, generates `data_csv/IntPhys1/cv/`.
- `z_scripts/run_5fold.sh {appb|c11|both}` — main 5-fold launcher.
- `z_scripts/aggregate_5fold.py <tag_base>` — per-fold summary → mean ± std.

### Data
- `data_csv/IntPhys1/IntPhys1_2way_all.csv` — full 360, source for CV splits.
- `data_csv/IntPhys1/cv/{train,val}_fold{0..4}.csv` — 5-fold splits.
- `data_csv/IntPhys1/keystones.json` — {video_path: tick}.
- `analysis/intphys1/build_keystones.py` — one-time keystones.json builder.

### Harness (do NOT modify without a good reason)
- `evals/main.py`, `evals/scaffold.py`
- `evals/analysis_vlm/eval.py` — main harness (setup / cache / train loop /
  summary write)
- `evals/analysis_vlm/cache.py` — cache builder + `PooledLinearProbe`
- `evals/analysis/modelcustom/vit_encoder_multilayer.py` — forward-hook wrapper
- `src/models/vision_transformer.py` — base ViT (DO NOT modify)
- `src/datasets/video_dataset.py` — decord-based dataset with keystones support

### Checkpoint
- `checkpoint/models--facebook--vjepa2-vitl-fpc64-256/snapshots/b3c1679b7c34d3255ef3547f27c7b226aefab26f/original/model.pth`
  — load key `target_encoder`.

### Debug in VS Code
- `.vscode/launch.json` entries `tak_intphys1_5fold_{appb,c11}_debug (fold 0)`.

---

## 10. Expected numbers (as of last run)

Baseline reproduction targets:

- **Chance (random)**: 50%.
- **Our current best** (single fold, 200 epochs, appb sweep, keystone,
  raw x_N): peak **87.5% at L06**.
- **Our 5-fold appb, 200 epochs**: expected mean 85-88% at peak.
- **Paper Fig 1 (V-JEPA 2 L)**: peak ~95%, L0 ~40-45%, L23 ~70-75%
  (chart-only, no explicit numbers).

A well-set-up run should:
- Show L0 near chance (50-55%). L0 above 60% suggests either the raw-feature
  pipeline is pulling more low-level signal than paper (possibly because
  keystone-centered clips concentrate the discriminative moment in raw
  patch space too), OR a training subtle issue.
- Show mid-layer (L5-L15) plateau near 78-88%.
- Show gradual decline toward L23.
- Show sharp jump around fraction ≈ 0.2-0.4 ("PEZ") — the sharper this
  jump, the closer to paper's shape.

---

## 11. If you're being asked to REPRODUCE this pipeline from scratch

Do NOT accept a spec that omits any of §1-§7 above. The one-liner "attach
forward hooks and linear probe per layer" is under-specified in at least
these ways that would BREAK reproduction:

1. Missing keystone-centered sampling → 8 pt peak-acc loss.
2. Missing quadruplet-level grouping → val leakage inflates numbers.
3. Missing fp32 cache decision → fp16 downcast loses 1-2 pt at peak.
4. Missing `use_bfloat16: false` → GradScaler bug can freeze all heads.
5. Missing constant lr/wd flags → schedule drift breaks paper HP semantics.
6. Missing "single Linear only, no pre_norm" → false LayerNorm inflates
   layer-wise scale mixing.
7. Missing "hook, not out_layers=" → stock self.norm-per-intermediate
   dramatically changes L0 and shape.
8. Missing 5-fold quadruplet CV → single-split noise dominates the reported
   mean.

Every one of these was learned by getting it wrong in an earlier session.
