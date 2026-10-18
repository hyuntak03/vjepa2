# z_vll7_training — vll7 (6 × RTX PRO 6000 Blackwell) predictor 학습 하네스

> 2026-10-16 개설. **frozen V-JEPA 2 ViT-H encoder 위에서 predictor 만** 학습한다.
> 코드는 레포 공용 (`app/vjepa_frozen/`, `src/models/rollout_predictor.py`) 을 쓰고,
> 이 폴더에는 **이 노드의 경로 · config · 런처 · 데이터 빌더** 만 둔다.
> vast 브랜치 (`origin/vast`, 8 × 96 GB Vast.ai) 에서 이식했다 — 이식 내역은 §5.

---

## 0. 30초 요약 — 지금 돌릴 것

**`ctxt_video_tgtvideo`** = 문맥을 **영상**으로 주고 타깃도 **영상**으로 (`kind: ctx_state`, `ctx_target: window`).

vast 가 config 만 쓰고 **안 돌린** 판이다. vast 의 `ctxt_video` (문맥 영상 → 타깃 **블록**) 는
IntPhys1 76.11 · VOBench 57.1 / 62.1 로 **블록 복사 기준선 (77.8 / 81.1) 아래**였고 원인을 모른다.
이 run 은 거기서 **타깃 하나만** 바꿔 "문맥(영상)과 답(블록)의 종류가 다르다" 가설을 한 변수로 시험한다.

```bash
bash z_vll7_training/env.sh --show                                  # 경로 · 실재 확인
DRYRUN=1 bash z_vll7_training/train.sh ctxt_video_tgtvideo          # 병합 + 실물 검사 (GPU 0 장, 몇 초)
GPUS=6 bash z_vll7_training/sbatch.sh ctxt_video_tgtvideo           # SLURM 제출 (vll7, 6 GPU)
watch -c -n 2 bash z_vll7_training/monitor.sh ctxt_video_tgtvideo
```

⚠️ **첫 run 전에 반드시** — 최대 창 (C16K16 = 32 장) 에서 VRAM 피크를 재고 `token_budget` 을 정한다 (§3-3).

---

## 1. 팔 2 × 샘플러 2

| config | 문맥 인코딩 | 타깃 | 샘플러 |
|---|---|---|---|
| **`ctxt_video_tgtvideo`** ★ | **영상** — context encoder 가 문맥 창을 한 번에 → z | **영상** — 창 (C+K) 전체 LN(target_encoder) 의 미래 블록 | 가변 창 |
| `ctxt_video` | 영상 (같음) | **블록** — 예측 구간 블록별 LN(h) (= TF 관측과 같은 공간) | 가변 창 |
| `ar_state_vw` | **블록** — target_encoder 를 블록마다 따로 | 블록 | 가변 창 |
| `ctxt_video_fixed` | 영상 | 영상 (기본; `SET="model.predictor.ctx_target=block"`) | 고정 창 |
| `ar_state_fixed` | 블록 | 블록 | 고정 창 |

세 팔 모두 **rollout 되먹임은 predictor 상태** (`predictor_norm` 출력 384) 이고 손실은 `tf_weight × TF + rollout` 이다.
`_vll7_base` → `_vll7_vw` / `_vll7_fixed` → 각 팔 순으로 `extends` 한다.

### 두 팔이 구조적으로 다른 점
- **블록 팔 (`ar_state`)**: encoder 가 블록(2 장)마다 → 비용이 창 길이에 **선형**. 블록 b 토큰은 블록 b 프레임만 본다 (시간 누수 0). encoder 하나만 쓴다
- **영상 팔 (`ctx_state`)**: 문맥 창을 **한 번에** → 문맥 토큰에 시간 맥락이 섞인다. encoder 둘 (문맥 online / 타깃 EMA).
  예측 구간은 **반드시 블록별**이어야 한다 — 영상으로 넣으면 블록 b 가 b+1 을 보게 되어 teacher forcing 이 누수다

---

## 2. 샘플러 둘 — 무엇이 다른가 (사용자 질문, 2026-10-16)

| | 고정 창 (`_vll7_fixed`) | 가변 창 (`_vll7_vw`) |
|---|---|---|
| 창 | **32 장 고정** (16 블록) | step 마다 W = C + K, C·K ∈ {4, 8, 16, 32} 장 |
| 프레임 간격 | `VideoDataset` 의 linspace (영상마다 흔들릴 수 있다) | **fps 4 정수 간격** |
| 패딩 | `filter_short_videos` 로 거름 (미리 min-frames csv) | **금지** — n4 ≥ W 인 영상만 |
| batch | 고정 (43 /rank × grad_accum 2) | 토큰 예산 B = clamp(950 // W, 8, 256) |
| (C, K) | `mask.prefix_window` 목록에서 step 당 하나, 창을 자른다 | 로더가 정하고 창을 정확히 그 길이로 읽는다 |
| 데이터 비율 | **명목 가중치 그대로** (창이 하나라 적격 집합이 안 바뀐다) | **명목 ≠ 실제** (창마다 적격 데이터셋이 다르다) |

### ⚠️ 이 노드에서 실제 비율이 어떻게 되나 (2026-10-16 전수 실측)

fps 4 로 창 W 장을 **패딩 없이** 채울 수 있는 편수:

| 창 W | 4 | 8 | 16 | 24 | 32 | 40 | 48 |
|---|---:|---:|---:|---:|---:|---:|---:|
| K400 320p (240,436) | 100 % | 99.8 % | 99.0 % | 97.1 % | 92.1 % | 87.7 % | **0** |
| SSv2 (146,773) | 100 % | 98.1 % | **50.1 %** | **1.9 %** | 7 편 | 6 편 | 1 편 |

- K400 은 10 초 클립이라 fps 4 에서 **최대 약 41 장** → **(C,K) 칸 3 개 `(16,32) (32,16) (32,32)` 가 버려진다.**
  vast 는 그 자리를 HowTo100M 1 % 가 메웠다. 이 노드엔 없다
- SSv2 는 fps 4 에서 W 16 을 넘으면 사라진다 → **긴 창은 K400 전용**
- 그 결과 **명목 1 : 1 이 실제 프레임 K400 76.7 : SSv2 23.3 을 낸다** — 우연히 릴리즈 비율
  (K710 0.335 : SSv2 0.100 = 77.0 : 23.0) 과 거의 같다. **그래서 가중치를 역산하지 않았다**
  (vast 는 HowTo 때문에 k710 0.65 : ssv2 0.60 : how 1.00 으로 역산해야 했다)
- 고정 창에서는 문제가 다르다: 32 장 × fps 4 면 SSv2 적격이 **7 편**뿐 → vast 와 같이 **SSv2 만 `frame_step: 1` (12 fps)**.
  ⚠️ **시간 척도가 섞인다** — 같은 "K 블록 뒤" 가 K400 2~4 초, SSv2 0.7~1.3 초. RoPE 는 블록 번호만 안다

### ⚠️ 정정 (2026-10-17) — 데이터셋별 `min_eligible` 이 무시되던 버그

이식 당시 판 (vast b71df3e) 의 `var_window.py` 는 `datasets[].min_eligible` 을 **읽지 않았다** (전역값만).
그래서 **`ctxt_video_tgtvideo` run (10-17 완료) 은 ssv2 가 적격 2,820 편뿐인 W=24 칸까지 들어갔고**
데이터 비율이 의도 76.7 : 23.3 대신 **실측 70.0 : 30.0** 이 됐다 (`train.log` 2,170 step 표본에서 재계산).
2026-10-17 에 vast 73ae468 의 구현을 이식해 고쳤다 — 이후 run 은 ssv2 가 W ≤ 20 칸에만 들어간다.
**tgtvideo run 과 그 뒤 run 은 샘플링이 다르므로 `ctx_target` 한 변수 대조가 아니다.**

```bash
# 가중치를 바꾸기 전에 반드시 먼저 본다 — 칸마다 적격 편수 · batch · 실제 step/clip/frame 비율
python z_vll7_training/data/var_window_report.py \
  --table k400=data_csv/vll7_k400_320/fps4_table.tsv --table ssv2=data_csv/vll7_ssv2/fps4_table.tsv \
  --C 4,8,16,32 --K 4,8,16,32 --budget 950 --min-eligible 20000
```

---

## 3. 환경 · 데이터 · 예산

### 3-1. ⚠️ python 환경
**기존 `vjepa2` env (torch 2.12.0+cu126) 는 이 GPU 에서 안 돈다** — `sm_50…sm_90` 으로만 빌드돼 있고
RTX PRO 6000 Blackwell 은 **sm_120** 이다. 전용 env `vjepa2_bw` (torch 2.12.1+cu129 = vast 와 같은 조합) 를 쓴다.
경로는 `paths.env` 의 `VJEPA2_PY`. 재설치:
```bash
conda create -n vjepa2_bw python=3.12 pip -y
~/anaconda3/envs/vjepa2_bw/bin/pip install torch==2.12.1 torchvision --index-url https://download.pytorch.org/whl/cu129
~/anaconda3/envs/vjepa2_bw/bin/pip install tensorboard wandb iopath pyyaml 'numpy<3' opencv-python submitit braceexpand \
  webdataset timm transformers peft decord pandas einops beartype psutil h5py fire python-box scikit-image ftfy matplotlib pytest
```

### 3-2. 데이터 (노드 로컬 NVMe)
| | 편수 | 경로 |
|---|---:|---|
| K400 320p train | 240,436 | `/local_datasets/Kinetics400/kinetics400_320p/train` |
| SSv2 | 146,773 | `/local_datasets/something-something-v2-mp4` |
| IntPhys1 dev (val) | 360 | `/local_datasets/world/world_analysis/IntPhys1_dev_frame_png` |

⚠️ **다른 노드의 csv 는 안 맞는다** (`data_csv/ssv2/*.csv` 등은 `/data2/...`). 만드는 법은
`configs/training/datasets.md` 의 **vll7 절** (gitignore 라 명령으로 다시 만든다).

### 3-3. 예산 · 메모리 (2026-10-16 실측)
| | vast (8 GPU) | **vll7 (6 GPU, 실측)** |
|---|---|---|
| step | 70 ep × 294 = 20,580 | 같게 둠 |
| step 당 clip | 344 (ctxt_video) | ~293 (평균 B 48.8 × 6) |
| 총 clip | 708 만 | **603 만** |
| VRAM 피크 | 75.4 GB (ctx_target=**block**) | **73.0 GB / 96 GB** (`window`) |
| 속도 | 1.59 s/step (`block`) | **2.7 s/step** (`window`) → 약 **15 시간** |

- 최악 칸은 **C8K32** (W 40, B 23, **R=16** = 가장 긴 rollout) 로 1 GPU 단독 측정 **72.7 GB**,
  본 run (6 GPU) 에서 **73.0 GB**. 다른 칸: W16·W24 약 63 GB · W32 64.9 GB · W40(C32K4, R=4) 64.9 GB.
  **`token_budget: 950` 을 그대로 쓴다** (여유 23 GB)
- 2.7 s/step 은 vast 1.59 s/step 에 vast 가 예측한 창-타깃 추가 비용 ×1.4–1.8 을 곱한 2.2–2.9 범위 안이다
- ⚠️ 첫 50 step 은 `torch.compile` 이 창 모양 13 종을 컴파일하느라 느리다 (12.9 → 7.8 → 4.2 → 2.6 s/step).
  **속도는 step 40 이후에 잴 것.** loader 는 data time 12 s → 0.3 s 로 완전히 파이프라인된다 (GPU 바운드)
- ⚠️ **budget 을 바꾸면 B·clip 수가 바뀌어 `ctxt_video` 와 한 변수 대조가 깨진다** — 바꿨으면 양쪽 다 바꿀 것
- lr 은 vast 와 같은 5.2e-4 로 뒀다 (global 이 작아졌으니 sqrt 규칙이면 3.96e-4 — 비교 가능성을 택했다)

---

## 4. 검증 (모델 로드 없음, 몇 초~몇 분)

```bash
python z_vll7_training/tests/ctx_state_cpu_test.py      # ctx_state 구조 · KV rollout 등가성 · 손실 경로
python z_vll7_training/tests/var_window_cpu_test.py --config ctxt_video_tgtvideo --steps 6   # 실제 mp4 로 로더
DRYRUN=1 bash z_vll7_training/train.sh <config>         # 병합 + 경로 · 프레임 예산 · kind 전제 검사
GPUS=2 bash z_vll7_training/train.sh --smoke-ddp        # spawn / 포트 / join (모델 · 데이터 없음)
```

2026-10-16 실측
- `ctx_state_cpu_test`: KV rollout 과 전체 재계산판의 출력 차 **0.0**, grad 차 5.5e-30 (float64)
- `var_window_cpu_test`: 6 step 전부 읽기 실패 0, 모양 `(B, 3, W, 256, 256)` uint8, rank 동기 계획 · rank 별 영상 선택 확인
- 1 GPU 학습 스모크 (`ctxt_video_tgtvideo`): loss 2.18 → 1.69, in-loop val 경로까지 정상
- 6 GPU 본 run: rank 6 개가 **같은 (C,K)** 를 뽑는 것 확인 (rank 동기 샘플링), loss 2.179 → 1.561 (step 60)

⚠️ **스모크에 val 을 반드시 켤 것.** `val=null` 로 돌린 스모크는 통과했는데 본 run 이 `ValSet` 에서 즉사했다
   (`data.n_frames` 64 로 IntPhys1 100 장에 stride 3 이 안 들어간다). 지금은 `val.n_frames` 를 쓰도록 고쳤다 (§5)

---

## 5. vast 에서 무엇을 가져왔나 (2026-10-16)

main 트리가 vast 보다 앞선 곳이 있어 (grad_accum, action 조건 AR, `analysis/predictors/` 채점 레지스트리,
`kind_from_arch` 가 vast 의 `ar + ar_feedback: state` → `ar_state` 매핑) **vast 를 통째로 쓰지 않고 빠진 조각만** 올렸다.

| 가져온 것 | 어디 |
|---|---|
| `kind: ctx_state` (문맥 영상 z + 상태 되먹임) — 구조 · 손실 · in-loop val · 채점 | `src/models/rollout_predictor.py` (`embed_obs` `render` `forward_state` `ctx_rollout_state{,_kv}`) · `app/vjepa_frozen/ar.py` (`ctx_state_losses` `run_val_ctx_state`) · `analysis/predictors/ar_scoring.py` (`ctx_state_{from_window,window}`) · `evals/world_model_analysis/eval.py` 두 경로 |
| `ctx_target: window \| block` (2026-10-16 vast `ctxt_video`) | 위 전부 + `app/vjepa_frozen/train.py` |
| 가변 창 로더 + fps4 표 빌더 | `app/vjepa_frozen/var_window.py` · `z_training/data/build_fps4_table.py` |
| WSD lr (`optimization.anneal_epochs`) · cooldown (`is_anneal` + `anneal_ckpt`) | `app/vjepa_frozen/utils.init_opt` · `train.py` |

**같이 고친 것 (vast 와 무관한 main 버그 · 제약)**
- `app/vjepa_frozen/train.py` 가 `train_step_ctx_ar` 를 **호출만 하고 정의하지 않았다** → `kind: ctx_ar` 는
  ViT-H 로드 (~2 분) 뒤 NameError 로 죽었다. vast 판으로 복원 (grad_accum 규약에 맞춰)
- `app/vjepa_frozen/train.py` 가 `ValSet` 을 **`data.n_frames`** 로 만들었다 → 가변 창 (최대 창 64) 에서 IntPhys1 (100 장) 에
  stride 3 이 안 들어가 6 rank 중 3 개가 즉사했다 (`scene_190.png 가 없다`). `val.n_frames` 를 쓰도록 수정
- `z_training/harness/resolve_train.py`: `val.data` 를 **일부만** 줘도 레지스트리가 기본값으로 깔리게 (전에는 통째로 건너뛰어 `root` KeyError),
  val 프레임 예산을 `val.n_frames` 로 검사 (가변 창은 `data.n_frames` 가 최대 창이라 val 과 다르다),
  `TRAIN_CFG_DIR` config 탐색 경로 (노드별 하네스가 자기 config 를 옆에 두고 공용 부모를 `extends`),
  데이터셋 경로의 `${VAR}` 치환 (main 이 vast 의 expand() 를 빼서 `${DATA_CSV}` 가 안 풀렸다)

**안 가져온 것**: 힌트 cross-attn · `boundary` 샘플러 (object_permanence) · multistart · `dyn_head` · 차이 L1 ·
`ctx_ar` 외 vast 전용 config. 필요해지면 `git show 73ae468:<경로>` 로 가져온다.

---

## 6. 알려진 사실 (vast 수치 — 데이터가 달라 직접 비교 불가, 방향만)

| 팔 | IntPhys1 A.8 최고 | VOBench 전체 P / R |
|---|---:|---|
| 릴리즈 predictor | 88.89 | — |
| AR + 상태 (`natural_ar_vw_state`) | 90.00 | 81.7 / 85.4 |
| **ctxt_video** (영상 문맥 → 블록 타깃) | **76.11** | **57.1 / 62.1** |
| 블록 복사 기준선 | — | 77.8 / 81.1 |
| `natural_ctx_state` (10-01, TF 없이 rollout 만, 창 타깃) | 48.33 | — |

- **영상 문맥 팔은 vast 에서 두 번 다 실패했다.** 10-01 판은 우연 수준, 10-16 판은 복사 기준선 아래
- 10-16 문서가 원인을 **모른다**고 적었다. 후보 (전부 미검증 사후 해석): z 를 안 쓴다 / z → 블록 공간 변환을
  못 배워 평균으로 간다 / 문맥 (영상) 과 타깃 (블록) 공간 불일치. 손실은 AR 과 거의 같았다 (배경이 L1 의 대부분)
- ⚠️ ctxt_video 는 AR 과 샘플링 3 개 + 구조 3 개가 같이 다르다 — "영상 문맥이 나쁘다" 고 쓸 수 없다
- ⚠️ **vll7 의 `ctxt_video_tgtvideo` 를 위 vast 수치와 비교할 때** — 대조군 `ctxt_video` 를 vll7 에서 안 돌리기로 했으므로
  (2026-10-16 사용자 결정, vast 에서 이미 돌렸다) 비교 상대가 vast 판이다. **타깃만 다른 게 아니다**:
  데이터 (K710 + SSv2 + HowTo -> K400 + SSv2) · 본 clip (708 만 -> 603 만) · GPU (8 -> 6) ·
  (C,K) 격자에서 W >= 48 칸 3 개 없음. **"타깃을 영상으로 바꾼 효과" 로 귀속할 수 없다** — 방향만 읽는다
- ⚠️ vast 는 `eval_bench.sh` 가 `ctx_target` 을 채점기에 안 넘겨 block 학습판을 window 로 채점한 적이 있다
  (IntPhys1 **43.89 무효** → 76.11). **채점 타깃을 학습 때와 똑같이 줄 것** (체크포인트 `arch.ctx_target` 이 기본값)

---

## 재현

```bash
# 데이터 (gitignore — 다시 만든다). configs/training/datasets.md 의 vll7 절에 같은 명령이 있다
python z_vll7_training/data/build_vll7_index.py --dataset k400_320
python z_vll7_training/data/build_vll7_index.py --dataset ssv2
python z_training/data/build_fps4_table.py --csv data_csv/vll7_k400_320/all.csv -o data_csv/vll7_k400_320/fps4_table.tsv --workers 48
python z_training/data/build_fps4_table.py --csv data_csv/vll7_ssv2/all.csv      -o data_csv/vll7_ssv2/fps4_table.tsv      --workers 48
python z_vll7_training/data/derive_fixed_csv.py --table data_csv/vll7_k400_320/fps4_table.tsv --mode fps4 --min 32 -o data_csv/vll7_k400_320/fps4_min32.csv
python z_vll7_training/data/derive_fixed_csv.py --table data_csv/vll7_ssv2/fps4_table.tsv      --mode raw  --min 32 -o data_csv/vll7_ssv2/fs1_min32.csv
# 학습
GPUS=6 bash z_vll7_training/sbatch.sh ctxt_video_tgtvideo
```

---

## 7. 결과 (2026-10-17)

### `ctxt_video_tgtvideo` (문맥 = 영상, **타깃 = 영상**) — 70 ep / 20,580 step 완료

| 평가 | 값 | 참고 |
|---|---:|---|
| IntPhys1 A.8 격자 최고 | **48.33** (skip5_w16) | 릴리즈 88.89 · vast `ctxt_video`(타깃 블록) 76.11 — **우연 수준** |
| VOBench v1 전체 (1,440 쌍) | 77.64 | vast ctxt_video 57.1 · 복사 기준선 77.8 |
| ├ primitive 전체 / **late** | 71.7 / **50.9** | late 가 우연 |
| └ realistic 전체 / **late** | 83.6 / **61.6** | |
| VOBench gravity (720 쌍) | 22.45 | chance 20 — 우연 |
| IntPhys 2 | **측정 못 함** | 아래 ⚠️ |
| in-loop val (IntPhys1 60 쌍) | e70 61.7 (최고 e54 61.7) | 정식 수치 아님 |

- **타깃을 영상으로 둔 run 이 독립적으로 두 번 48.33 에 떨어졌다** (여기 + vast 10-01 `natural_ctx_state`).
  10-01 판은 TF 가 꺼져 있었고 이 run 은 켰는데도 같다 — "TF 지름길" 로는 설명이 안 된다
- ⚠️ 단서: 두 run 은 데이터 · 창 · step 도 다르다. 깨끗한 한 변수 대조는 **같은 노드에서 `ctx_target: block`** (`ctxt_video.yaml`) 뿐이다
- ⚠️ VOBench 점수는 **복사 기준선 대비 Δ 로만** 읽어야 한다 (창 타깃은 정답에 문맥이 섞여 복사를 유리하게 만든다, §2 정정)
- ⚠️ **IntPhys 2 는 채점 엔진 (`analysis/intphys2/eval.py`) 이 `kind=ctx_state` 를 몰라 전 rank 100 % NaN 이 났다.**
  모델이 못 푼 게 아니라 하네스 미지원이다. 필요하면 vast `73ae468` 의 `_CtxStateScorer` 를 이식한다 (2026-10-17 사용자 결정: **일단 대기열에서 제외**)

### EK100 action anticipation (`vith_release_vobftg`) — 진행 중

릴리즈 predictor + VOBench split v2 + gravity post-FT 30 ep. `released` 규약 · batch 126 · `HEADS=grid8` · 20 epoch.
실측 **4.2 s/iter · 57 분/epoch → 약 19 시간**. 진입점 `z_vll7_training/eval_ek100.sh`.

⚠️ **`num_workers` 를 올리지 말 것** — 16 으로 했더니 6 rank × 16 = 96 디코드 워커가 job (CPU 108 · mem 300G) 을 질식시켜
`Training...` 직후 멈췄다 (rank 2 개가 GPU 메모리도 못 잡고 공회전). 6 이 맞고, data time 0.7 ms 라 병목이 아니다.
⚠️ 런처는 run 폴더의 `stdout.log` 가 30 분 안에 갱신됐으면 **"진행 중" 으로 보고 건너뛴다** — 중단한 run 은 폴더를 치우고 다시 띄운다.
⚠️ DDP 포트는 `20000 + (SLURM_JOB_ID % 10000)` 에서 **탐색을 시작**한다. 빠르게 죽였다 살리면 잔여 프로세스와 충돌한다 (2026-10-17 rank 2 개 사망) — `EVAL_DDP_PORT` 를 명시할 것.
