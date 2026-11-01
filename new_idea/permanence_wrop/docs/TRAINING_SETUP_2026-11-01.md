# permanence_wrop 학습 설정과 재현 방법 (2026-11-01)

> **3줄 요약**
> - **무엇** — B · C · D 세 판을 다른 기계에서 똑같이 다시 만들기 위한 학습 설정 전부. 판은 학습 데이터만 다르다.
> - **어떻게** — 코드는 git commit `71bcf97` (branch `vast`, `jong-exp` 도 학습 코드가 같다) 이다. 학습에 쓴 표본 목록 · 병합된 설정 · 체크포인트 해시를 `repro/` 에 같이 두었다.
> - **확인** — 다시 돌린 뒤 §7 의 기대값 (학습 끝 손실 · 채점 정확도) 과 비교한다. 같은 seed 라도 GPU 연산이 결정론적이지 않아 작은 차이는 난다 (§9).

근거 표시: [측정] = 이 기계의 산출물에서 다시 확인한 값 · [이론] = 설정 · 코드에서 계산한 값.
값은 전부 학습 폴더의 병합된 설정 (`params-train.yaml`) 과 코드에서 다시 확인했다.

## 용어

| 용어 | 뜻 |
|---|---|
| 판 | 학습 실행 하나. B · C · D 셋 (A = 학습 안 한 원래 모델) |
| 클립 | 학습 표본 하나 = 32 장 영상. 앞 16 장 = 문맥, 뒤 16 장 = 미래 |
| 블록 | 연속 2 장을 encoder 에 함께 넣은 시간 단위 (tubelet). 32 장 = 16 블록, 블록마다 공간 토큰 256 개 (16 × 16) |
| step | 가중치 갱신 한 번. 전역 배치 64 클립 |
| epoch | 이 실험에서는 200 step 을 한 단위로 부르는 이름. 데이터를 한 바퀴 도는 것과 무관하다 |
| WROP | 공개 물체 영속 데이터 (Hugging Face `Hokin/object-permanence`, Blender, 150 과제 × 1 만 표본) |
| 경계 | WROP 영상의 앞부분 (input) 과 뒷부분 (target) 이 갈리는 프레임. 가림 계열은 120 장의 한가운데 (60) |

## 1. 무엇을 재현하나

| 판 | 학습 데이터 (뽑힐 확률) | 목적 |
|---|---|---|
| B | SSv2 100 % | 더 학습한 것 자체의 효과 |
| C | SSv2 50 % + WROP 가림 두 계열 50 % | 가림 경험의 효과 |
| D | SSv2 50 % + WROP 비영속 세 계열 50 % | 같은 엔진 대조군 |

세 판 모두 원래 V-JEPA 2 ViT-H 의 predictor 에서 출발한다. encoder 는 고정하고 predictor 만 학습한다.
나머지 설정은 세 판이 전부 같다.

## 2. 코드와 환경

| 항목 | 값 |
|---|---|
| 코드 | git commit `71bcf97` (branch `vast`, 2026-11-01 09:00 UTC). branch `jong-exp` 는 그 위에 기록 · 목록 commit 만 더한 것이라 학습 코드가 같다. 학습 코드 파일은 전부 학습 시작 (10-31 12:35 UTC) 전에 마지막으로 바뀌었고 그 상태로 이 commit 에 들어 있다 [측정: 파일 수정 시각] |
| Python | 3.12.14 |
| PyTorch | 2.12.1 + CUDA 12.9 (cuDNN 9.2) |
| 기타 | numpy 2.5.2 · decord 0.6.0 · timm 1.0.30 · ffmpeg 6.1.1 (데이터 만들 때) |
| GPU | NVIDIA RTX PRO 6000 Blackwell (96 GB) × 4 (GPU 4–7), 드라이버 580.126.09 |
| GPU 메모리 | 판당 GPU 한 장에 19 GB [측정] |
| CPU · RAM | 256 코어 · 1.1 TB (학습 때 Blender 렌더와 나눠 씀) |

## 3. 모델 (출발점)

| 항목 | 값 |
|---|---|
| 체크포인트 | Hugging Face `facebook/vjepa2-vith-fpc64-256`, revision `b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5`, 파일 `original/model.pth` (10.4 GB) |
| sha256 | `2f18bab855c310d75fac12bfdb64f2dc2e2e048979f3fb9312c1d97a713b33cd` [측정] |
| 구조 | ViT-H/16, 입력 256 × 256, 블록 2 장, RoPE |
| 문맥 encoder | 체크포인트의 `encoder` (online). 고정 |
| 정답 encoder | 체크포인트의 `target_encoder` (EMA). 고정 |
| predictor | 체크포인트의 predictor 를 그대로 불러와 이어 학습. 깊이 12 · 폭 384 · head 12 · mask token 10 개 중 0 번을 쓴다. 학습되는 파라미터 2,228 만 개 [측정] |
| encoder 연산 정밀도 | bfloat16 |

## 4. 데이터

### 4-1. SSv2 (B · C · D 모두)

| 항목 | 값 |
|---|---|
| 원본 | Something-Something v2 webm 220,847 개 (12 fps, 높이 320) |
| 고르는 기준 | 원본 32 장 이상인 영상만, seed 0 으로 train / val 을 나눈 train 쪽 = **185,229 개** [측정]. 공식 분할이 아니다 |
| 클립 | 영상마다 시작점을 무작위로 잡아 **연속 32 장** (12 fps, 2.7 초) |
| 표본 목록 | `repro/lists/ssv2_train_min32_ids.txt.gz` (영상 번호, 학습 때와 같은 순서) |

### 4-2. WROP (C · D)

세 단계로 만든다.

| 단계 | 하는 일 | 결과 |
|---|---|---|
| ① 원본 | Hugging Face `Hokin/object-permanence` 의 과제별 tar 150 개. 표본마다 앞부분 60 장 + 뒷부분 60 장 (1280 × 720, 24 fps) | 과제당 1 만 표본 |
| ② 줄이기 (2026-10-15) | 앞뒤를 한 영상으로 이어 붙이고, 사건 경계로 자른 표본의 반복 패딩은 잘라 낸다. 짧은 변 320 (568 × 320), x264 crf 23. 과제마다 표본을 seed 0 으로 섞어 앞 1,450 = 학습, 다음 100 = held-out | 과제당 1,450 + 100 |
| ③ 32 장 클립 (2026-10-31) | 경계 앞에서 고르게 16 장 + 경계 뒤에서 고르게 16 장 (반올림한 균등 간격). 568 × 320 그대로, x264 crf 18, 정확히 32 장 | 아래 표 |

③ 의 결과 [측정]:

| 묶음 | 계열 (과제 수) | 과제당 | 클립 |
|---|---|---:|---:|
| 가림 (C) | 움직이는 물체 가림 (26) + 멈춘 물체 가림 (29) | 1,450 | **79,750** |
| 비영속 (D) | 막힘 (20) + 떨어짐 (21) + 충돌 (19) | 앞 1,330 | **79,800** |
| held-out (채점용) | 150 과제 전부 | 100 | 15,000 |
| (학습에서 뺌) | 그릇 (35) | — | 0 |

- 가림 55 과제 중 53 과제는 앞 60 + 뒤 60 장이다 [측정].
  - 프레임 번호는 `0, 4, 8, …, 55, 59 | 60, 64, …, 115, 119` 다.
  - 간격은 3.9 장 (≈ 6.1 fps) 이고, 경계에서만 1 장 간격이다.
  - 나머지 2 과제 (`drop_screen_occluder` · `rising_sleeve_cover`) 는 끝 패딩을 잘라 뒤가 44 · 40 장이다. 그만큼 미래 간격이 촘촘하다.
- 가림 계열의 경계는 사건 시점이 아니라 120 장의 한가운데다 (WROP 메타데이터 `video_split.method: midpoint`).
- 비영속 계열 중 26 과제는 경계가 사건 시점이라 앞부분이 18–59 장이다. 그만큼 문맥 간격이 촘촘하다 [측정].
- 표본 목록과 프레임 번호는 `repro/lists/` 에 있다 (부록 A).

### 4-3. 학습 때 클립을 넣는 방식

| 항목 | 값 |
|---|---|
| 시간 | WROP 는 32 장 전부를 그대로 (로더가 읽는 번호 0–31 [측정]). SSv2 는 무작위 시작점부터 연속 32 장 |
| 공간 자르기 | 32 장 모두에 같은 무작위 창. 면적 50–100 %, 가로세로비 0.75–1.35 (로그 균등). 10 번 시도해 화면 안에 안 들어가면 가운데 정사각. WROP (568 × 320) 에서는 폭 54–72 % · 높이 85–100 % 이고 14 % 가 가운데 정사각으로 간다 [측정: 같은 규칙 20 만 번 모의] |
| 좌우 뒤집기 | 확률 0.5 |
| 크기 | 자른 뒤 256 × 256 으로 bilinear (antialias 없음, 채점과 같은 커널) |
| 정규화 | ImageNet 평균 · 표준편차 |

### 4-4. 섞는 방식 (샘플러)

- 클립을 하나 뽑을 때마다 데이터셋을 가중치에 비례해 고르고, 그 안에서 균등하게 고른다. 중복을 허용한다. C · D 는 가중치 1 : 1 이다.
- 난수 seed 는 0 이다. 샘플러를 다 쓰면 seed 를 바꿔 (0 + 다음 epoch 번호) 다시 뽑는다.
  - C · D 는 4,000 step 안에 샘플러를 다 쓰지 않는다 (한 바퀴 4,140 step).
  - B 는 2,894 step 뒤에 seed 15 로 한 번 다시 뽑는다 [이론: 코드].
- 실제로 뽑힌 수 (같은 seed 로 다시 계산) [측정]:

| 판 | SSv2 | WROP |
|---|---|---|
| B | 256,000 번 · 서로 다른 138,631 개 | — |
| C | 128,203 번 · 92,477 개 | 가림 127,797 번 · 63,704 개 (움직이는 가림 60,482 · 멈춘 가림 67,315 번) |
| D | 128,203 번 · 92,477 개 | 비영속 127,797 번 · 63,622 개 |

- C 와 D 는 SSv2 쪽이 같은 순서로 뽑혔다 [측정].
- 표본 목록의 **행 순서**가 바뀌면 뽑히는 클립도 바뀐다. 정확히 재현하려면 `repro/lists/` 의 순서대로 목록을 만든다.

## 5. 학습 설정

| 항목 | 값 |
|---|---|
| 과제 | 문맥 16 장 (8 블록) → 미래 16 장 (8 블록) 토큰 예측 |
| 손실 | 미래 8 블록 × 256 토큰의 L1 평균. 정답 = 정답 encoder 에 32 장 전부를 넣은 출력에 affine 없는 LayerNorm |
| 보조 손실 · 라벨 · 불가능 영상 | 없음 |
| GPU | 4 장 (DDP), GPU 당 배치 16, **전역 배치 64** |
| 길이 | 200 step × 20 epoch = **4,000 step**, 25.6 만 클립 |
| optimizer | AdamW, β (0.9, 0.999), ε 1e-8 |
| weight decay | 0.04 고정. bias 와 1 차원 파라미터는 0 |
| 학습률 | 1e-6 에서 200 step 동안 선형으로 5e-5 까지 → 2,800 step 동안 5e-5 고정 → 마지막 800 step 동안 1e-6 까지 선형 감소 (WSD) [측정: step 3,800 에서 1.325e-5] |
| gradient clipping | 없음 |
| 연산 정밀도 | bfloat16 autocast (predictor · encoder), loss scaler 없음 |
| seed | 0 |
| data loader | GPU 당 worker 5 개, persistent, pin memory, decord 스레드 1 |
| 학습 중 감시 | VOBench held-out late 216 쌍, 학습 전 + 2 epoch 마다 |
| 저장 | 5 epoch 마다 (`e5` · `e10` · `e15` · `e20`) + 매 epoch `latest`. **predictor 만** 저장 (268 MB). `latest.pt` = `e20.pt` 가중치 [측정] |

## 6. 실행과 걸린 시간

- 이 기계에서는 세 판을 같은 GPU 4 장 (4–7) 에 동시에 올렸다.
  - B 는 10-31 21:35 KST 시작, 3 시간 35 분.
  - C · D 는 21:42 KST 시작, 3 시간 45 분.
- 세 판이 함께 돌 때 step 당 3.35 초였다. 혼자 돌던 처음 6 분은 1.17 초였다 [측정].
- 혼자 돌리면 판당 약 1 시간 30 분으로 본다 [이론: 4,000 × 1.17 초 + 감시 · 로딩].
- 데이터 읽기는 step 당 10 ms 라 병목이 아니다 [측정].
- 명령은 부록 B.

## 7. 재현 확인용 기대값

### 7-1. 학습 [측정]

| 판 | epoch 20 평균 학습 손실 | 학습 중 감시 정확도 (epoch 0 → 2 → 20) |
|---|---:|---|
| B | 0.6034 | 60.6 → 58.8 → 60.6 |
| C | 0.5656 | 61.6 → 63.4 → 65.7 |
| D | 0.5666 | 61.3 → 58.8 → 57.4 |

### 7-2. 채점 (정확도 %, 같은 쌍) [측정]

| 세트 | 쌍 | A 원래 | B | C | D |
|---|---:|---:|---:|---:|---:|
| VOBench held-out | 720 | 75.7 | 78.5 | 79.7 | 73.1 |
| └ 영속 · late · 사라짐 | 36 | 13.9 | 13.9 | 41.7 | 11.1 |
| IntPhys 1 late | 180 | 68.3 | 68.9 | 72.2 | 65.6 |
| IntPhys 2 violation-uni | 500 | 54.6 | 52.8 | 53.6 | 53.4 |
| real_bench | 40 | 47.5 | 42.5 | 42.5 | 42.5 |

WROP held-out 예측 L1 (과제당 20 클립, 전체 평균): 원래 0.6062 · B 0.5896 · C 0.5593 · D 0.5631 [측정].

## 8. 다른 기계로 옮길 때 바꿀 것

- **절대경로.** 아래를 그 기계에 맞게 고친다.
  - 레포 · 데이터 경로: `z_research/scripts/harness/paths.env`, `cvpr/env.sh`.
  - 체크포인트 경로: `configs/training/frozen_predictor_postft.yaml` 계열의 `model.checkpoint`.
- **표본 목록은 git 에 없다.** `data_csv/` 전체가 `.gitignore` 에 걸린다. 영상을 만든 뒤 목록을 새로 만들고, `repro/lists/` 와 행 순서 · 개수가 같은지 맞춘다 (부록 A).
- **영상 인코딩.** WROP 32 장 클립은 원본 → crf 23 → crf 18 로 두 번 인코딩했다. 같은 ffmpeg 판이 아니면 픽셀이 조금 다르다.
- **GPU 수.** 전역 배치 64 를 유지한다 (예: 8 장이면 GPU 당 8). 그러면 샘플러가 GPU 마다 나누는 순서가 달라져 뽑히는 클립 순서도 바뀐다.

## 9. 정확히 같지 않을 수 있는 곳

- **GPU 연산이 결정론적이지 않다.** bfloat16 과 attention 커널, DDP gradient 합산 순서 때문이다. 같은 seed 라도 손실이 소수 넷째 자리에서 다를 수 있다 [이론].
- **데이터 로더 worker 의 난수.** 공간 자르기 · 뒤집기는 worker 마다 Python 난수를 쓴다. worker 수가 바뀌면 달라진다 [이론].
- **학습 중 감시 값도 흔들린다.** 세 판의 epoch 0 은 학습 전이라 같은 원래 predictor 인데, 감시 정확도가 60.6 / 61.6 / 61.3 으로 달랐다 [측정]. 216 쌍 중 2 쌍 차이다. 감시 값은 ±1 pt 를 잡음으로 읽는다. 원인은 bfloat16 연산의 비결정성으로 본다 [추측].
- **seed 를 하나만 돌렸다.** 기대값 자체가 seed 하나의 값이다. 재현 결과가 몇 점 다르면 seed 차이 범위인지부터 본다.

## 부록 A. `repro/` 파일

| 파일 | 내용 |
|---|---|
| `configs/params-train_{B,C,D}.yaml` | 학습 때 병합된 설정 전부 (학습 폴더에서 그대로 복사) |
| `configs/permanence_wrop_{base,B,C,D}.yaml` | 원본 설정 (`configs/training/` 과 같다) |
| `lists/ssv2_train_min32_ids.txt.gz` | SSv2 영상 번호 185,229 개, 학습 목록 순서 |
| `lists/wrop_uni32_train_op12_ids.txt.gz` | C 의 WROP 표본 79,750 개, 학습 목록 순서 |
| `lists/wrop_uni32_train_nonop_ids.txt.gz` | D 의 WROP 표본 79,800 개, 학습 목록 순서 |
| `lists/wrop_uni32_heldout_ids.txt.gz` | 채점용 held-out 15,000 개 |
| `lists/wrop_uni32_{train,heldout}_manifest.tsv.gz` | 표본 · 과제 · 계열 · 고른 32 장의 프레임 번호 (② 단계 영상 기준) |
| `lists/object_permanence_320_used_manifest.tsv.gz` | 쓴 표본의 ② 단계 정보: 전체 장 수 · 잘라 낸 패딩 · 앞 / 뒷부분 장 수 · 경계 |
| `lists/wrop_task_family.tsv` | 150 과제 → 계열 (WROP 공식 분류) |
| `checksums.txt` | 원래 체크포인트와 B · C · D predictor 의 sha256 |

## 부록 B. 재현 명령

```bash
cd <레포>                                   # git checkout 71bcf97
PY=/venv/vjepa/bin/python                   # 그 기계의 파이썬 (§2 판)

# 1. SSv2 목록 (원본 32 장 이상, seed 0 분할) → data_csv/ssv2/train_min32.csv
$PY z_training/data/build_video_index.py ssv2 --min-frames 32 --write

# 2. WROP 원본 (Hugging Face Hokin/object-permanence, train/<task>_task.tar 150 개) 을
#    ${TRAIN_DATA_ROOT}/object_permanence/train/ 에 풀고
$PY z_training/data/build_object_permanence_320.py --n-train 1450 --n-heldout 100 --workers 200
#    → ${TRAIN_DATA_ROOT}/object_permanence_320/ + data_csv/object_permanence_320/{train,heldout}_manifest.tsv

# 3. 32 장 클립 (가림 1,450 / 비영속 1,330 / held-out 100, 그릇 계열 제외)
$PY new_idea/permanence_wrop/data/build_wrop_uni32.py --workers 56
#    → /workspace/dataset/training_data/wrop_uni32/ + data_csv/wrop_uni32/{train_op12,train_nonop,heldout}.csv

# 4. 목록 확인: 개수 · 순서가 repro 와 같은지
diff <(awk '{n=split($1,a,"/"); sub(/\.mp4$/,"",a[n]); print a[n]}' data_csv/wrop_uni32/train_op12.csv) \
     <(zcat new_idea/permanence_wrop/repro/lists/wrop_uni32_train_op12_ids.txt.gz) && echo "C 목록 같음"
diff <(awk '{n=split($1,a,"/"); sub(/\.webm$/,"",a[n]); print a[n]}' data_csv/ssv2/train_min32.csv) \
     <(zcat new_idea/permanence_wrop/repro/lists/ssv2_train_min32_ids.txt.gz) && echo "SSv2 목록 같음"

# 5. 학습 (tmux 세션으로, GPU 4 장). 이 기계에서는 셋을 GPU 4–7 에 동시에 올렸다
for arm in B C D; do
  GPUS=4 DEVICES="cuda:4 cuda:5 cuda:6 cuda:7" bash z_training/tmux_train.sh permanence_wrop_$arm
done
#    앞에서 직접: GPUS=4 DEVICES="cuda:0 cuda:1 cuda:2 cuda:3" bash z_training/train.sh permanence_wrop_C
#    산출: z_training/runs/permanence_wrop_<판>/{latest.pt,e5..e20.pt,params-train.yaml,metrics.jsonl,train.log}

# 6. 채점 (판마다) — 모델 이름은 cvpr/registry/models.yaml 의 vast_permwrop_{B,C,D}
GPU_IDS="4 5 6 7" bash cvpr/eval/testbed/run.sh "vobench_v1_split_v2_test real_bench_quads" vast_permwrop_C
DS=intphys1_late_uni      GPU_IDS="4 5 6 7" bash cvpr/eval/intphys1_late/run.sh vast_permwrop_C
DS=intphys2_violation_uni GPU_IDS="4 5 6 7" bash cvpr/eval/intphys1_late/run.sh vast_permwrop_C
CUDA_VISIBLE_DEVICES=4 $PY new_idea/permanence_wrop/eval/wrop_heldout_l1.py --per-task 20 \
  --models release B=z_training/runs/permanence_wrop_B/latest.pt C=z_training/runs/permanence_wrop_C/latest.pt D=z_training/runs/permanence_wrop_D/latest.pt
$PY new_idea/permanence_wrop/eval/summarize.py         # → results/summary_tables.md (§7-2 와 비교)
#    한 번에: tmux new-session -d -s permwrop_eval "bash new_idea/permanence_wrop/eval/run_eval_chain.sh"

# 7. 체크포인트 해시 확인 (같은 가중치를 받아 쓸 때)
sha256sum -c <(awk '{print $1"  "$2}' new_idea/permanence_wrop/repro/checksums.txt)   # 경로는 맞게 고칠 것
```

설정 키와 값 대응 (병합된 설정 `params-train_C.yaml` 기준): `data.n_frames 32` · `data.batch_size 16` · `data.aug.hflip_p 0.5` · `data.aug.random_resized_crop {scale [0.5, 1.0], ratio [0.75, 1.35]}` · `data.datasets[].{type video_csv, frame_step 1, filter_short_videos true, weight 1.0}` · `mask [{type temporal_prefix, context_frames [16]}]` · `loss {loss_exp 1.0, target_layer_norm true}` · `model {arch_name vit_huge, load_predictor true, mask_index 0, context_encoder_key encoder, target_encoder_key target_encoder}` · `meta {seed 0, dtype bfloat16, encoder_dtype bfloat16}` · `optimization {epochs 20, ipe 200, lr 5e-5, start_lr 1e-6, final_lr 1e-6, warmup 1, anneal_epochs 4, weight_decay 0.04, final_weight_decay 0.04, clip_grad null}` · `val {dataset vobench_v1_split_v2_test_late, every_epochs 2, at_start true}`.
