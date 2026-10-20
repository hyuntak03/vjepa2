# z_pixel_decoder — V-JEPA 2 latent → 픽셀 decoder

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

## 실행

```bash
bash z_pixel_decoder/run.sh                                      # tmux pixdec_<name>, GPU 8
bash z_pixel_decoder/run.sh z_pixel_decoder/configs/vobench_vits.yaml 4
tail -f z_pixel_decoder/runs/vobench_vits/train.log
watch -c -n 2 bash z_pixel_decoder/monitor.sh [name]                 # 진행 · ETA · 손실 · val PSNR 추이 · 최신 그림 · GPU
```
산출 (`runs/<name>/`, gitignore): `latest.pt` (decoder · opt · step) · `log.csv` (val MSE · PSNR) ·
`samples/step_N.png` (val 블록 8 개 첫 장: 위 원본 · 아래 복원). 같은 `name` 으로 다시 올리면 `latest.pt` 에서 이어 돈다.
