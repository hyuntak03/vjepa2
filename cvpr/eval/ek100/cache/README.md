# eval/ek100/cache — encoder 특징 캐시 위에서 EK100 probe 학습 (2026-10-16)

**목적.** EK100 probe 학습의 비용은 거의 전부 ViT-H encoder (32 프레임) 다. encoder 는 모든 모델에서 같고 (`checkpoint_key: target_encoder`) predictor 만 바뀌므로, encoder 출력만 한 번 뽑아 두고 predictor + probe 는 캐시 위에서 돈다.

## 무엇을 뽑나
| 변형 | 모양 | 쓰는 곳 |
|---|---|---|
| `joint` | (4096, 1280) — 32 프레임 한 번에 | probe 입력의 encoder 부분 (모든 모델) · release kind predictor 의 입력 |
| `blocks` | (4096, 1280) — 튜블릿마다 따로 + affine-free LN (`app/vjepa_frozen/ar.encode_blocks`) | AR kind (`ar` · `ar_state`) predictor 의 문맥 |

context encoder 는 EK100 경로에 안 쓰여 뽑지 않는다 (`evals/action_anticipation_frozen/modelcustom/vit_encoder_predictor_concat_ar_nonsquare.py`, encoder = `target_encoder`).
float16, clip 당 10.5 MB/변형. 폴더 `/data2/local_datasets/world/world_analysis/cache/ek100_feats/<ap규약>_<time_source>_<train csv>/{train,val}/` — `joint.npy` · `blocks.npy` (np.memmap) · `meta.npz` · `info.json`.

| 지금 있는 캐시 | 행 | 크기 |
|---|---|---|
| `released_frame_fixfps_EPIC_100_train_1of5_strat_min3_seed0` train (view 1) | 13,452 | 141 GB × 2 |
| 〃 val (1/5 train 에 있는 action 만 = 엔진과 같은 필터) | 9,296 | 97 GB × 2 |

> **2026-10-17 정리**: 지금 쓰는 캐시는 vll6 의 1/7 strat × view 20 하나뿐이다 (결과 3). vll5 의 1/5 val (fp16) · 1/30 train (int8) 캐시와
> 1/5 × view 1 · 1/30 e20 · 1/30 e100 실행의 head 체크포인트 (`latest.pt`) 는 지웠다 — 수치는 아래 표와 각 폴더의 `val_metrics.jsonl` · 로그에 남아 있다.
> vll6 의 `rollout_v1_vith` 토큰 캐시 (149 GB, RollOut_v1 은 레지스트리에서 빠진 세트) 도 지웠다.

## 2026-10-16 결과 1 — view 1 개는 안 된다 → view 20 개 (= epoch 마다 새 증강) · clip 을 줄인다
1/5 strat_min3 (13,452 clip) × view 1 로 캐시 학습 (`ek100_rel_fifthS3_e20_wu3_b48_cache`) vs 영상 학습 (job 222518, 같은 설정):
| epoch | 영상 action (최고 head) | 캐시 action | 영상 train action acc | 캐시 train action acc |
|---|---|---|---|---|
| 5 | 6.05 | 6.75 | 22.0 | 32.6 |
| 8 | — | — | 31.9 | **99.2** |
| 10 | 13.89 | **10.60** (noun 24.95 vs 31.44) | 39.4 | — |
| 20 | 23.26 (verb 47.54 · noun 41.78) | 멈춤 (epoch 10 뒤) | 77.6 | — |
같은 입력을 매 epoch 보니 head 가 epoch 8 에 train 을 외우고 val 이 처진다. 1/5 train 캐시는 지웠다 (val 은 남김).
→ **원본처럼 epoch 마다 다른 view (view 20 = epoch 20) 를 쓰고, 디스크에 맞게 clip 을 줄인다** (사용자 결정). train = 1/30 **strat** (2,240 구간, `build_ek100_subset.py --n 30`: 지표 action 1,114 개 전부 ≥1, 나머지 빈도 비례; 영상 396 · 참가자 32; verb · noun 분포 TV 0.075 · 0.124).
val 은 그대로 재사용된다 — 1/30 strat 도 val 9,296 clip, 영상 순서까지 같다 (지표 action 을 전부 덮으므로 엔진의 val 필터 결과가 같다).
저장은 int8 + clip·채널별 scale (`--dtype q8`, clip·view 당 10.5 MB = fp16 의 절반, 44,800 행 ≈ 470 GB).
| 저장 | encoder 특징 상대 L1 | predictor 출력 (vs fp16) |
|---|---|---|
| fp16 | 1.8e-4 | — |
| int8 토큰별 scale | 2.2e-2 (joint) · 3.9e-2 (blocks) | 6.4e-3 (release) · 4.1e-3 (vast AR) |
| **int8 채널별 scale (채택)** | **8.7e-3 · 8.4e-3** (train view 0 chunk 0 실측, `check_train_q8.py`) | — |
참고로 head 는 bf16 autocast 라 원래 경로에도 0.1–0.4 % 반올림이 있다. train view 는 seed (view, chunk) 로 재현된다 (at 일치 확인).
config `cvpr/eval/ek100/cache/configs/ek100_rel_1of30strat.yaml` (222518 과 같고 train csv · base_path 만 다름).

## 2026-10-16 결과 2 — 1/30 strat × view 20 (int8)
| 실행 | step | val action / verb / noun (최고 head) | train action acc |
|---|---|---|---|
| 20 epoch (`ek100_rel_1of30strat_e20_wu3_b48_cache`) | 920 | 1.84 / 14.76 / 7.61 | 17.0 (ep 19) |
| 100 epoch (`…_e100_…`, view 20 을 5 번씩, val 20 epoch 마다) — ep 20 | 920 | 5.66 / 25.00 / 19.72 | 35.5 |
| 〃 ep 40 · 60 · 80 | 1,840 · 2,760 · 3,680 | 10.83 / 33.61 / 26.22 · 11.84 / 36.72 / 29.23 · 12.20 / 39.76 / 30.17 | 97.6 (ep 45) · 100 (ep 81) |
| 〃 **ep 100 (최종)** | 4,600 | **12.63 / 40.83 / 30.06** | 100 |
| 참고: 1/5 영상 학습 222518 ep 10 · 최종 | 2,800 · 5,600 | 13.89 / 35.68 / 31.44 · 23.26 / 47.54 / 41.78 | 39.4 · 77.6 |
| 참고: full | — | 35.34 / 56.36 / 56.16 | — |
- 20 epoch 은 step 이 모자라 바닥. 100 epoch 은 train 을 외운 뒤 (ep 45 에 97.6 %) 에도 val 이 오른다 — view 1 개 (같은 입력 반복) 에서 꺾인 것과 다르다.
- 그러나 action 은 ep 60 뒤로 12 근처에서 멈춘다 → **clip 수 (2,240) 가 상한을 정한다.** GPU 4 장 2.6 h (12:54–15:33).

## 2026-10-16 결과 3 — 1/7 strat × view 20 (int8, vll6) ★ 지금 쓰는 판
캐시 `vll6:/data2/local_datasets/world/world_analysis/cache/ek100_feats/released_frame_fixfps_EPIC_100_train_1of7_strat_seed0/` (train 9,600 × 20 = 192,000 행 · val 9,296, int8). job 222856: 추출 val 8 분 · train 93 분 (GPU 8 장), 학습 73 분 (GPU 8 장, head 3 개, 4,000 step).
| epoch | val action / verb / noun (최고 head) | train action acc | 1/5 영상 학습 (222518) val action · train acc |
|---|---|---|---|
| 5 | 4.96 / 22.74 / 17.62 | 21.8 | 6.05 · 22.0 |
| 10 | 10.91 / 30.61 / 26.87 | 39.1 | 13.89 · 39.4 |
| 15 | 16.64 / 40.68 / 35.28 | 64.2 | 22.06 · 63.8 |
| **20** | **17.71 / 42.38 / 36.89** | 79.1 | **23.26** (47.54 / 41.78) · 77.6 |
- train 곡선이 1/5 영상 학습과 epoch 마다 1 점 안으로 겹친다 — 외우지 않는다 (1/5 × view 1 은 epoch 8 에 99 %).
- 최종: 1/30 캐시 12.63 < **1/7 캐시 17.71** < 1/5 영상 23.26 < full 35.34. clip 수 (2,240 · 9,600 · 13,452) 순.
- 재현 검사 (job 222910, 10-17): view 0 · 13 의 chunk 7 — at 재현 ✅, 캐시 vs 새로 계산 상대 L1 joint 8.8e-3 · 7.4e-3 / blocks 8.4e-3 · 6.8e-3 (int8 몫). meta: train 192,000/192,000 valid · view 20, val 9,296/9,296.
- ⚠️ 이 캐시는 vll6 /data2 에만 있다 — 학습도 vll6 에서. EK100 run.sh 의 `tee /dev/stderr` 가 SLURM 출력 파일을 지우던 것을 고쳤다 (이 job 의 merge · 재현 검사 줄이 그렇게 유실됐다).

## 엔진과 다른 점 (하나)
**train 증강이 고정된다.** 엔진은 epoch 마다 anticipation time · point · RRC · RandAugment · RandomErasing · hflip 을 새로 뽑는다. 캐시는 view 마다 한 번 뽑아 고정하고 epoch e 에 view `e % views` 를 쓴다 (`--views K`). val 은 결정적이라 차이가 없다.
→ 이 차이가 점수를 얼마나 바꾸는지는 같은 설정의 영상 학습 (job 222518, `ek100_rel_fifthS3_e20_wu3_b48`) 과 캐시 학습 (`…_cache`) 을 나란히 놓고 잰다.

## 검증 (2026-10-16)
| 검사 | 결과 |
|---|---|
| 입력: 우리 clip 샘플링 · 변환 vs 엔진 val 로더 (첫 영상 8 clip) | max\|Δ\| = **0** (비트 일치) |
| 캐시 값 vs 그 자리에서 새로 계산 (joint · blocks) | 상대 L1 **0** |
| 캐시 경로 model forward vs 원래 wrapper — release | encoder 부분 1.8e-4 · predictor 부분 4.4e-3 (clip 끼리 0.45) |
| 〃 — vast AR (`ar_state`, ep70) | encoder 1.8e-4 · predictor 2.5e-3 (clip 끼리 0.37) |
남는 차이는 fp16 저장 + bf16 autocast 반올림이다.

⚠️ 같이 고친 버그: EK100 wrapper 가 vast 판 체크포인트 (`kind: ar` + `ar_feedback: state`) 를 `ar` 로 지었다 → `analysis.predictors.kind_from_arch` 로 `ar_state` (2026-10-16). 그 전에 vast AR 로 EK100 을 돌린 적은 없다.

## 명령
```bash
CFG=<엔진 _resolved.yaml>
# 1) 추출 (vll5 GPU 2·3 예) — 세션과 분리해서
CFG=$CFG GPU_IDS="2 3" BASE=/data2/local_datasets/EPIC-KITCHENS_resized SPLITS="val train" \
  setsid nohup bash cvpr/eval/ek100/cache/run_extract.sh > cvpr/logs/ek100_cache_extract.log 2>&1 < /dev/null &
# 2) 검사
CUDA_VISIBLE_DEVICES=2 python cvpr/eval/ek100/cache/extract.py --config $CFG --split val --base … --check 8
CUDA_VISIBLE_DEVICES=2 python cvpr/eval/ek100/cache/check_forward.py --config $CFG --cache <캐시 폴더> --base … [--pred <predictor ckpt>]
# 3) 캐시 위에서 학습 (엔진 그대로, data.cache_dir 만 얹는다)
CONFIG=$CFG GPUS=2 DEVICES="cuda:2 cuda:3" TAG=<새 tag> \
  SET="data.cache_dir=<캐시 폴더> data.base_path=/data2/local_datasets/EPIC-KITCHENS_resized optimization.batch_size=24" \
  bash z_research/previous/anticipation/EK100/run.sh
#    AR kind 모델: SET 에 data.cache_blocks=true 와 model_kwargs.predictor_checkpoint=<ckpt>
```

## 재현
- 추출 `cvpr/eval/ek100/cache/extract.py` (+ `run_extract.sh`), 검사 `check_forward.py`
- 학습 경로 `evals/action_anticipation_frozen/cache.py` + `eval.py` 의 `data.cache_dir` 분기 (없으면 공식 경로 그대로)
- config `z_research/previous/anticipation/EK100/exp_results/action_anticipation_frozen/ek100_rel_fifthS3_e20_wu3_b48/_resolved.yaml`
