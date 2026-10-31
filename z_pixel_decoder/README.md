# z_pixel_decoder — V-JEPA 2 latent → 픽셀 decoder

> **main 이식 (2026-10-21)** — `origin/vast` 브랜치 `f6e7934` (2026-10-20) 의 `z_pixel_decoder/` 를 그대로 가져왔다. 모델 · 손실 · 데이터 규칙은 그대로이고, 바꾼 것은:
> - 경로: config 의 `root` · `index` 에 `${CVPR_VOBENCH}` · `${CVPR_LOCAL}` (값은 `cvpr/env.sh`, vll5 = `/data2/local_datasets/world/Benchmark/VOBench_v1`). IntPhys1 dev 는 `${CVPR_LOCAL}/world/world_analysis/IntPhys1_dev_{videos,frame_png}`
> - val split: 해시 키를 `split_tag` (기본 = config 의 index 문자열) 로 — IntPhys1 은 vast 판 문자열 `data_csv/intphys1_dev/index.csv` 을 `split_tag` 로 둬 vast 학습본과 같은 키를 쓴다 (val clip: VOBench 272 · gravity 81 · IntPhys1 44). ⚠️ vast 쪽 index 내용이 이 노드와 같은지는 미검증
> - `optim.accum` (기본 1) 추가 — 24 GB GPU 에서 같은 batch 를 나눠 쌓는다. ViT-B `both` 를 vll5 에서 다시 학습하는 메모리는 **미검증**
> - `run.sh`: tmux · `/venv/vjepa` → `setsid nohup` + `cvpr/env.sh` 의 `CVPR_TORCHRUN`, `GPU_IDS` = 물리 번호 (`CUDA_DEVICE_ORDER=PCI_BUS_ID`), pid 는 `runs/<name>/train.pid` · `monitor.sh` 는 그 pid 로 살아 있나를 본다
> - `lpips` 0.1.4 를 vjepa2 환경에 설치했다 (10-21, VGG 가중치는 `~/.cache/torch/hub`)
>
> vast 에서 학습한 ViT-B 판 (`vob_grav_i1_vitb_both`, 87.4 M) 의 체크포인트는 `runs/vob_grav_i1_vitb_both/latest.pt` (gitignore).

블록 latent (블록 = 2 장, 고정 ViT-H **target_encoder** 출력에 affine 없는 LN, `(256, 1280)`) 를 그 블록의 두 장 픽셀로 되돌린다.
AR predictor 의 정답 · 예측과 같은 공간이라, 학습이 끝나면 predictor 가 그린 미래를 그림으로 볼 수 있다.

## 구조 (`model.py`)

```
z : [B, 256, 1280]   블록 하나 (2 장) 의 LN(target_encoder) 토큰
  ↓ Linear(1280 → 384) + 학습 위치 임베딩
  ↓ Transformer 12 층 (ViT-S: 384 · 6 head)     이웃 patch 를 맞추고, 추상 feature 를 pixel 쪽 표현으로
  ↓ LN → Linear(384 → 2 × 16 × 16 × 3 = 1536)   토큰 하나가 자기 자리 patch 를 2 장 분량으로
  ↓ unpatchify (reshape 만)
x̂ : [B, 3, 2, 256, 256]
```
`model.depth: 0` 이면 linear 하나 (feature 에서 pixel 을 선형으로 얼마나 읽는지 보는 pixel 판 linear probe).

## 학습 (`train.py`, config `configs/vobench_vits.yaml`)

- 데이터: VOBench_v1 (2,880 clip × 96 장, 512² → 256² bilinear). 블록 = 채점과 같은 간격의 두 장 `(s, s+3)`, `s` 무작위
- split: `source_block` 해시 10 % 를 val (같은 block 의 4 변이가 train / val 로 갈리지 않는다) — train 2,648 · val 232 clip
- 손실: `L1(x̂, x)` ([0, 1] 픽셀) `+ λ · LPIPS(VGG)` (블록의 두 장 각각, λ = `optim.lpips`, 기본 1). `optim.loss: mse` 면 정규화 공간 MSE
- encoder 고정 + no_grad, decoder 만 AdamW · warmup + cosine · grad clip 1.0
- 기본: GPU 당 batch 128 × 8 GPU, 5,000 step, lr 1e-3 (약 1.5 시간, 계산은 대부분 encoder)
- ⚠️ VOBench 에 깨진 PNG 가 하나 있다 (`realistic/permanence/early_k2/assets_flat_acc_tennis_ball/pos_a/000008.png`). 로더가 건너뛴다.
  채점은 3 장 간격 (0, 3, 6 …) 이라 이 장을 안 읽는다

## predictor 미래를 그림으로 — `render_future.py` (2026-10-21, main 에서 추가)

여러 predictor (레지스트리 `cvpr/registry/models.yaml` 이름, base vith) 의 미래를 **같은 클립**에 나란히 그린다. 입력은 채점 (testbed `surprise_c16t32`) 과 같다 (원본 0, 3, …, 93 장, 문맥 16 장).
- 릴리즈 계열: `p = predictor(context_encoder(문맥))`, 정답 공간 = LN(target_encoder(32 장 한 번에)) → decoder `window`
- AR 계열 (`ar` · `ar_state`): 블록별 LN(target_encoder) 문맥 8 블록 → rollout 8 블록 → decoder `block`
- 클립: PFT 에 안 쓴 test split (VOBench split v2 test `pos_a` · gravity split test 가능 미래, ledge 는 부유 미래도), decoder val 클립 먼저
```bash
source cvpr/env.sh; CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 PYTHONPATH=$PWD $CVPR_PY z_pixel_decoder/render_future.py \
  --models vith,vith_release_vobftg,vith_pft_intphys1_e40,vith_pft_ip2pair,vith_vast_ar_state_ep70,vith_vast_ar_state_ep70_vobftg --sets late,gravity --n 8
# 다른 노드 (VOBench 없음): 프레임만 NFS 로 복사해 둔 runs/_frames/VOBench_v1 을 쓴다
sbatch -w vll2 --export=ALL,SETS="late" z_pixel_decoder/render.sbatch          # VideoMAE 는 SCRIPT=render_videomae.py
```
산출은 **모델별 폴더** (10-21) `runs/vob_grav_i1_vitb_both/<모델>/<set>/` — `grid.png` (행 = 클립 × [원본 · 정답 latent (그 모델 공간) · 그 모델]) · 클립별 GIF · `summary.json` (PSNR · 미래별 PSNR · 채점 surprise).

| 폴더 | 레지스트리 이름 | 무엇 |
|---|---|---|
| `release` | `vith` | 릴리즈 predictor |
| `release_vobench` | `vith_release_vobftg` | 릴리즈 + PFT (VOBench split v2 train + gravity train) |
| `release_intphys1` | `vith_pft_intphys1_e40` | 릴리즈 + IntPhys1 dev PFT |
| `release_intphys2` | `vith_pft_ip2pair` | 릴리즈 + IntPhys2 pair PFT |
| `ar` | `vith_vast_ar_state_ep70` | vast AR (ar_state ep70) |
| `ar_vobench` | `vith_vast_ar_state_ep70_vobftg` | vast AR + PFT (VOBench + gravity) |
| `videomae` | `videomae2g` | VideoMAEv2-g 자기 MAE decoder (`render_videomae.py`, 이 decoder 는 안 씀) |
| `compare` | — | 모든 모델 한 장 `compare_grid.png` + `summary.json` |
| `cosmos_ar4b` | Cosmos-Predict1-4B | 생성형 비교 (`render_cosmos.py` · `cosmos.sbatch`, 환경 cosmos-predict1). 문맥 raw 30–46 (9 장, stride 2) → 미래 48–94 (24 장), pillarbox 640×1024, diffusion decoder · guardrail 끔, seed 0. GIF = 영상 \| Cosmos \| tokenizer enc→dec |

**wall 세트 (2026-10-25)** — RollOut_v2 wall (문맥 = 벽으로 굴러감, 미래 = 벽에서 멈춤 `stop` · 벽 통과 `pass`), 속도 7 단계마다 1 쌍.
프레임은 vll5 로컬에만 있어 고른 쌍만 `runs/_frames/RollOut_v2_wall/` 로 복사했다 (`clips.py` 의 wall 분기 주석). 산출 `<모델>/wall/` · `compare/wall/`.
⚠️ decoder 는 RollOut 을 학습하지 않았다 — 정답 latent 행 PSNR 15–19 dB (복사 29–38 dB), **미래 장면에 벽 기둥이 안 그려지고 배경이 VOBench 풍으로 바뀐다.** 물체 색 · 자리만 읽을 수 있다.
```bash
sbatch --exclude=vll5 --mem-per-gpu=24G --time=0-00:40:00 --job-name=pixdec_wall --export=ALL,SETS="wall",ARGS="--n 7" z_pixel_decoder/render.sbatch
```

옛 `renders/` · `renders_ar/` 는 `_superseded/` 로 내렸다.
⚠️ LoRA 판 (`vith_pft_ip2pair_lora`) 은 아직 안 된다 — 문맥 encoder 에 LoRA 를 붙이는 경로가 이 스크립트에 없다.

## 실행

```bash
GPU_IDS="4 5 6 7" bash z_pixel_decoder/run.sh z_pixel_decoder/configs/vob_grav_i1_vitb_both.yaml   # 물리 GPU 번호
# predictor 미래를 그림으로 (릴리즈 predictor · 32 장 window latent · 블록 latent 세 갈래 PSNR)
source cvpr/env.sh; CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 $CVPR_PY z_pixel_decoder/render_predictor.py \
  --set late --n 12 --decoder z_pixel_decoder/runs/vob_grav_i1_vitb_both/latest.pt
$CVPR_PY z_pixel_decoder/grid_from_gifs.py z_pixel_decoder/runs/vob_grav_i1_vitb_both/renders/late
tail -f z_pixel_decoder/runs/<name>/train.log
watch -c -n 2 bash z_pixel_decoder/monitor.sh [name]                 # 진행 · ETA · 손실 · val PSNR 추이 · 최신 그림 · GPU
```
산출 (`runs/<name>/`, gitignore): `latest.pt` (decoder · opt · step · cfg) · `log.csv` (val MSE · PSNR) ·
`samples/step_N.png` (val 블록 8 개 첫 장: 위 원본 · 아래 복원). 같은 `name` 으로 다시 올리면 `latest.pt` 에서 이어 돈다.
