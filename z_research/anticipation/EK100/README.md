# EK100 action anticipation 재현 — 시작점 (2026-09-12)

> **아직 한 번도 안 돌렸다.** 이 문서는 **구현과 확인해야 할 것**만 담는다. 수치는 하나도 없다.
> 레포 규칙은 루트 `CLAUDE.md`. 논문은 V-JEPA 2 §6 + Appendix D.1 (Table 19).

## 0. 한 줄

frozen V-JEPA 2 encoder + predictor 위에 **attentive probe 만** 학습해 EK100 의 다음 action
(verb / noun / action)을 1초 앞서 맞히고, **mean class recall@5** 로 잰다.
공식 구현(`evals/action_anticipation_frozen/`)이 레포에 이미 있으므로 **새로 쓴 것은 없다** —
경로·설정을 붙이고, 논문과 어긋나는 지점 3곳을 찾아 **논문 쪽으로 맞췄다** (§3).
특히 **문맥은 action segment 시작 1초 전에서 끝난다** — 릴리즈 코드 기본값은 그게 아니었다.

---

## 1. 먼저 답: spatial / temporal crop 을 하는가

**질문하신 것과 실제 구현이 다르다.**

| 축 | 공식 구현이 실제로 하는 것 | 어디 |
|---|---|---|
| **spatial (val)** | **center crop 을 한다.** 짧은변을 `256*256/224 = 292` 로 키우고 가운데 **256x256** 을 자른다. 1920x1080 이면 **폭의 46% 를 버린다** | `dataloader.py` `VideoTransform.eval_transform` |
| **spatial (train)** | `random_resized_crop` (scale 0.08~1.0, ratio 3/4~4/3) + auto-augment + h-flip + random-erase 0.25 | 같은 파일 `__call__` |
| **temporal** | **crop/다시점 없다.** action 마다 클립 **하나**. 32 frame @ 8 fps = **4초 문맥** 을 `af` 에서 끝나게 한 번 뽑는다. val 에 multi-view averaging 도 없다 | `epickitchens.py` `decode_videos_to_clips` |

즉 **temporal 은 말씀대로 crop 이 없고, spatial 은 center crop 이 있다.**
→ **2026-09-13 결정: 당분간 공식 `center_crop` 을 기본으로 쓴다.** `short_side` 는 `SET` 으로 켠다.

### spatial 모드 세 가지 (기본 `center_crop`, 2026-09-13)

`evals/action_anticipation_frozen/spatial.py` 에 종횡비를 **절대 늘리지 않는** 세 모드를 뒀다.
원본을 정사각으로 눌러 담는 경로는 어디에도 없다.

| `spatial_mode` | 무엇 | 1920x1080 에서 | 출력 |
|---|---|---|---|
| `short_side` | 짧은변을 256 에 맞추고 긴변만 가운데로 자른다 | 1080 -> **256** (세로 전부 보존), 1920 -> 455 -> 가운데 **256** | **256x256** |
| **`center_crop`** | **짧은변 -> 292, 그 뒤 가운데 정사각 (공식 구현) — 기본값** | 1080 -> 292 -> 256 (**1.14배 더 확대**, 세로 12.5% 추가로 버림) | 256x256 |
| `letterbox` | 긴변을 맞추고 남는 자리 패딩. 아무것도 안 자름 | 1920 -> 256, 1080 -> 144 + 패딩 | 256x256 |

실측 (EK100 해상도 3종, 전부 출력 256x256 · 비율 안 깨짐):

| 원본 | scale | 중간 | crop |
|---|---|---|---|
| 1080x1920 (694개) | 0.2370 | 256x455 | 256x256 |
| 720x1280 (4개) | 0.3556 | 256x455 | 256x256 |
| 1440x1920 (2개) | 0.1778 | 256x341 | 256x256 |

**공식 `center_crop` 과의 차이는 확대율 하나다.** 공식은 짧은변을 292 로 키운 뒤 256 을
떼므로 세로 화각을 12.5% 더 버린다. `short_side` 는 **세로를 통째로 보존**한다.
가로는 둘 다 자른다 — 정사각 출력이라 피할 수 없다. **어느 쪽도 비율을 늘리지 않는다.**

구현은 `scale = max(256/h, 256/w)` 후 CenterCrop (CSS `object-fit: cover`).
`spatial.py` 의 `ResizeShortSideCenterCrop`. 옛 이름 `cover` 도 별칭으로 받는다.

출력이 정사각이므로 **module 은 공식 정사각 판 그대로** 쓴다
(`vit_encoder_predictor_concat_ar`), 토큰도 4096 그대로고 **global batch 도 논문과 같은 128** 이다.
`width` 를 256 이 아닌 값으로 주면 비정사각이 되는데, 그때만 `..._nonsquare` module 이 필요하다
(`resolve.py` 가 조합을 검사한다).

⚠️ `letterbox` 는 pretrain 이 본 적 없는 입력이다. 대조군으로만.
⚠️ 공식 수치는 `center_crop` 으로 나온 것이라 **직접 비교 대상이 아니다** (§8).

---

## 2. 프로토콜 (논문 Table 19 대로 넣었다)

| 항목 | 값 |
|---|---|
| 문맥 | 32 frames @ 8 fps = **4초**, `af` 에서 끝난다 |
| val anticipation time | **1.0초** (`anticipation_time_sec: [1.0, 1.0]`) |
| val anticipation point | 0.0 |
| train anticipation time | 0.25 ~ 1.75초 |
| train anticipation point | 0.0 ~ 0.25 |
| probe | attentive, **4 블록**, 16 head, 마지막 cross-attention 의 **query 3개** (verb / noun / action 각각 linear) |
| probe 입력 | `[encoder 출력 ; predictor 출력]` 을 토큰 축으로 concat — 256x256 이면 **4096 + 256 = 4352 토큰** |
| loss | focal (alpha 0.25, gamma 2.0), 세 분류기 독립 합산 |
| epoch / warmup | 20 / 0 |
| global batch | **128** |
| head | 논문은 **20개 = lr 5종 x wd 4종** sweep. **우리는 1개, lr 1e-3 / wd 1e-2 고정** (2026-09-13 결정) |
| 지표 | **mean class recall@5 (verb / noun / action), EK100 validation set** — 논문 문장 그대로 |

### probe head 는 하나로 고정했다 (2026-09-13)

논문은 lr 5종 x wd 4종 = **head 20개를 동시에 학습하고 매 지표를 20개 중 `max` 로 보고**한다
(`train_one_epoch` / `validate` 의 `max([...])`). 즉 논문 수치는 val 로 고른 sweep best 다.

우리는 **head 1개, `lr 1e-3` / `wd 1e-2`** 로 고정했다. 값은 논문 grid 의 가운데를 고른 것이고
**근거가 있는 값이 아니다.** 바꿀 때는 `LR=3e-4 WD=1e-1 bash run.sh ek100_vith`.
sweep 으로 돌리려면 `make_configs.py` 에서 `heads="sweep"`.

### 지표 — mean class recall@5 를 val set **전체에서 한 번씩** 센다

논문: *"We report mean-class recall-at-5 for verb, noun and action on the validation set of EK100."*

지표의 **정의**는 릴리즈 코드(`metrics.ClassMeanRecall`)와 같다 — top-5 안에 정답이 있으면 그 클래스의 TP,
아니면 FN. 클래스별 recall 을 **val 에 한 번이라도 나온 클래스**에 대해 평균. 예측 후보를 val 클래스로 제한하지 않는다.

문제는 그 지표를 **어떤 clip 들 위에서** 세느냐다. 릴리즈 `eval.py` 의 `validate()` 는

```python
ipe = num_clips // (world_size * batch_size)
for itr in range(ipe):
    try:    udata = next(_data_loader)
    except: _data_loader = iter(data_loader); udata = next(_data_loader)   # 끝나면 다시 감는다
```

val 은 **비디오 단위로 rank 에 나뉘어** rank 마다 clip 수가 다른데, 모든 rank 가 똑같이 `ipe` 배치만 센다.
clip 이 적은 rank 는 앞쪽을 다시 감아 **두 번** 세고, 많은 rank 는 뒤쪽을 **안 센다.**

실제 val 분포(9,296 clip / 138 video)로 그 루프를 흉내 낸 결과:

| 설정 | ipe | rank 가 가진 clip | 한 번도 안 센 clip | 두 번 이상 센 clip |
|---|---:|---|---:|---:|
| 공식 config (64 GPU, bs 2, workers 2) | 72 | 17 ~ 646 | **3,089 (33.2%)** | 1,812 |
| 우리 (8 GPU, bs 16, workers 6) | 72 | 793 ~ 1,906 | **1,420 (15.3%)** | 1,061 |

⚠️ 이건 **시뮬레이션**이다 (실행해서 센 값이 아니다). 가정: DataLoader 가 worker 를 round-robin 으로 섞고,
비디오 순서는 `filter_annotations` 그대로, decode 실패 0. 수치의 크기는 이 가정에 달려 있지만
**"val set 과 다른 clip 집합을 센다"** 는 결론은 `ipe` 공식과 다시 감기만으로 정해진다.
이 차이가 recall 을 올리는지 내리는지는 **모른다** (클래스 분포가 비디오마다 달라서 방향이 정해지지 않는다).

→ **`evaluation.exact_val_pass: true`** (전 config 기본): rank 마다 자기 loader 를 **끝까지 한 번** 돌고
TP/FN 을 로컬에 쌓았다가 **all_reduce 한 번**. 실제로 센 clip 수(`n_clips`)를 기대값과 같이 기록한다.
구현 `evals/action_anticipation_frozen/exact_val.py`, 지표 정의가 `ClassMeanRecall` 과 같은지는
`selftest.py` [6] 에서 긴 꼬리 라벨로 대조했다 (97 / 3,568 클래스 모두 소수 넷째 자리까지 일치).

출력 `exp_results/action_anticipation_frozen/<tag>/val_metrics.jsonl` — epoch 마다 한 줄:
`{"epoch", "metric": "mean_class_recall@5", "action": {"recall", "accuracy"}, "verb": ..., "noun": ..., "n_clips", "expected_clips", "heads": [{"lr", "wd"}]}`

**보고 값은 마지막 epoch(20)** 으로 한다. epoch 중 최고값은 val 로 고른 것이라 descriptive 다 (CLAUDE.md §1-3).

### 레이블 공간

`filter_annotations` 가 **train 에 있는 조합만** 남긴다. 실측:

```
train  67,217 segment / 495 video     verb 97 · noun 289 · action(verb,noun 쌍) 3,568
val     9,296 segment / 138 video     (원본 9,668 중 372 개가 train 에 없는 action 이라 버려진다)
비디오  /data/dataset/EPIC-KITCHENS 에 700개 전부 있다 (인덱스에 있고 파일이 없는 것 0)
```

논문이 말하는 "3,568 action / 97 verb / 300 noun" 과 맞는다 (noun 은 실제로 289 개만 쓰인다).

---

## 3. ⚠️ 논문과 공식 구현이 어긋나는 3곳 — **돌리기 전에 결정해야 한다**

세 곳 다 **실물에서 확인했다.** 차이 2 는 **논문/표준 프로토콜 쪽으로 고정했다** (2026-09-12 사용자 결정
"논문 그대로 가라"). 차이 1·3 은 릴리즈 동작이 기본이고 config 스위치로 열려 있다.

### 차이 1. predictor 가 **학습되지 않은 mask token** 을 쓴다

`src/models/predictor.py` 의 `forward(..., mask_index=1)` 이 기본값이고 공식 wrapper 는
그 기본값을 그대로 쓴다. 그런데 릴리즈 체크포인트에서 **학습된 mask token 은 `[0]` 하나뿐이다.**

```
mask_tokens.0: norm=0.6706  std=0.03427      <- 학습됨
mask_tokens.1: norm=0.0000  std=0.00000      <- 전부 0
...  1~9 전부 norm 0.0000                      (vith 체크포인트 실측)
```

즉 공식 anticipation eval 은 **0 벡터**를 미래 토큰 자리에 넣는다. RoPE 위치 정보는 살아 있으니
predictor 가 아무것도 못 하는 건 아니지만, **우리 `surprise_c16t32` 프로토콜은 `mask_index: 0` 을
"critical" 로 못박아 뒀다** (CLAUDE.md §1-7). 같은 체크포인트에 두 관례가 공존하면 안 된다.

- 스위치: `model_kwargs.wrapper_kwargs.mask_index` (기본 1 = 공식). `0` 으로 바꾸면 학습된 토큰.
- `mask_index != 1` 은 비정사각 module 에서만 된다 (원본 module 은 이 인자를 안 받는다).

### 차이 2. `anticipation_point` 의 방향이 논문 본문과 **반대**다 → **논문대로 고정했다**

EK100 은 action 마다 시간 구간(`start_frame` / `stop_frame`)이 달려 있고, 표준 anticipation
프로토콜은 **문맥이 action segment 시작 τa(=1초) 전에서 끝난다**. 논문 본문도 같다 —
*"anticipation point 0 이면 action segment 의 **첫 프레임**을 예측한다."*

릴리즈 코드는 그렇지 않다: `af = int(sf*ap + (1-ap)*ef - aframes)` 라서 **ap=0 이면 `ef`**
(action 의 **끝**)다. val 은 `val_anticipation_point` 를 yaml 에 안 써서 기본값 `[0.0, 0.0]`
이 들어가므로, 문맥 4초가 **action 이 끝나기 1초 전**에서 끝난다.

실측 — `P01_11_100 'wash cloth'` (59.94 fps, action frame 19636~19918 = 4.70초):

| 규약 | 문맥 (raw frame) | 32장 중 action 내부 |
|---|---|---:|
| 릴리즈 (ap=0 → `ef`) | 19635 ~ 19852 (327.58s ~ 331.20s) | **31장** |
| 논문/표준 (ap=0 → `sf`) | 19353 ~ 19570 (322.87s ~ 326.49s) | **0장** (문맥 끝과 action 시작 간격 +1.10s) |

val 9,296 segment 전수:

| 규약 | action 프레임이 문맥에 드는 segment | 평균 누수 |
|---|---:|---|
| 릴리즈 | **78.8%** | **12.33 / 32 장** (누수 있는 것만 15.6장) |
| **논문/표준** | **0.0%** | **0장** |

→ **`anticipation_point_mode: paper` 를 전 config 의 기본값으로 박았다.**
`af = sf*(1-ap) + ap*ef - aframes` 이므로 ap=0 에서 `sf - aframes` 다.

train 은 논문이 `ap ∈ [0.0, 0.25]` 로 지정한 대로 둔다 — 정의상 segment 의 앞 25% 까지만
안으로 들어갈 수 있는 **의도된 증강**이다. 실측 (paper 규약, at 0.25~1.75s):

| | action 프레임이 문맥에 드는 segment | 평균 누수 |
|---|---:|---|
| train, 논문 규약 | 9.3% | 0.78 / 32 장 |
| train, 릴리즈 규약 (대조) | 64.0% | 9.15 / 32 장 |

- 코드 변경 없이 yaml 만으로도 같은 결과를 낼 수 있다 (릴리즈 공식에 ap=1 을 넣으면
  `af = sf - aframes`): `val_anticipation_point: [1.0, 1.0]`, `train_anticipation_point: [0.75, 1.0]`.
  읽기 편한 쪽을 골라 `anticipation_point_mode: paper` 로 넣었다.
- 옛 릴리즈 동작이 필요하면 (예: 공개 수치와 대조) 한 줄이면 된다 —
  `SET="data.anticipation_point_mode=released"`. **그 값으로 나온 수는 anticipation 성능으로
  읽을 수 없다** (78.8% 가 정답을 보고 있다).

### 차이 3. annotation 프레임 번호와 비디오 fps 가 8개 비디오에서 어긋난다

구현은 `vfps = vr.get_avg_fps()` 로 실제 fps 를 읽고 csv 의 `start_frame`/`stop_frame` 을
그 비디오의 프레임 번호로 **그대로** 쓴다. `start_timestamp` 로 역산해 본 실측:

| 비디오 실제 fps | 역산한 annotation fps | 비디오 수 | segment 수 |
|---|---|---:|---:|
| 59.94 | 59.999 | 423 | 대부분 (오차 0.1%, 무해) |
| 50.00 | 50.000 | 268 | 대부분 (정확히 일치) |
| **29.97** | **60.000** | 4 | 506 |
| **47.95** | **59.999** | 4 | 84 |

즉 **29.97 / 47.95 fps 비디오 8개(segment 590개, 0.9%)는 프레임 번호가 60fps 기준**이라
구현이 그대로 쓰면 시점이 2배/1.25배 어긋난다. 전체의 0.9% 라 수치에 거의 영향이 없지만 기록해 둔다.

- 스위치: `data.time_source: timestamp` — `start_timestamp`/`stop_timestamp` x 실제 fps 로 다시 계산한다 (전 비디오에서 옳다). 기본은 `frame` (공식).

---

## 4. ⚠️ 돌리기 전에 해야 하는 것

### 4-0. 전처리한 비디오 (2026-09-13, vll6 전용)

`/data2/local_datasets/EPIC-KITCHENS_resized` — **짧은변 256 -> 가운데 256x256**, fps·프레임 수 원본 그대로.
스크립트 `z_research/scripts/data/build_ek100_resized.py`, 처리 방법 `_info.json`, 파일별 대조 `_meta/<video_id>.json`,
진행률 `python z_research/scripts/data/ek100_resized_progress.py`, 로그 `_build.log` (파일 하나 끝날 때마다 한 줄).

- **config 는 `spatial_mode: short_side`** 로 쓴다. 256x256 입력에서 항등 변환이다.
  기본값 `center_crop` 을 그대로 두면 짧은변을 292 로 **다시 키워** 256 을 떼므로 1.14배 추가 확대 + 업샘플 흐림
- 프레임 인덱스: 디코드 순서 k 번째 프레임에 k/fps 타임스탬프를 새로 매겨 **원본 decord k 번째 = 출력 k 번째**.
  파일마다 프레임 수를 원본 헤더와 대조한다. 시험 3개(59.94 / 4:3 / 29.97fps)에서 움직이는 프레임 전부 정렬, 밀림 0
- ⚠️ 공식과 다른 점: (1) val 기하 — 공식은 짧은변 292 -> 256 crop (세로 12.3% 더 버림), 여기는 짧은변 256 -> 256 crop.
  (2) train random resized crop 이 원본 전체 화면이 아니라 이 256x256 안에서만 고른다.
  (3) 축소 커널 ffmpeg bilinear vs cv2 INTER_LINEAR. (4) H.264 crf 18 재인코딩
- ⚠️ 기록: 처음 판(`-vsync 0` 만)은 P29_01 / P29_05 에서 프레임이 2장씩 빠져 뒤쪽 인덱스가 1~2 칸 밀렸다 (원본 타임스탬프 흔들림).
  한때 짧은변 292 no-crop 으로 바꿨다가 사용자 결정으로 256x256 으로 되돌렸다. 프레임 재번호 수정은 유지
- 속도: 디코드 스레드 2개는 CPU 만 두 배 쓰고 wall 이 같다 (P17_01 실측 157.5 vs 80.1 CPU초) -> ffmpeg 60개 x 스레드 1개.
  104 -> 151 MB/s. 병목은 CPU (할당 64 = 물리 32코어 x HT). GPU 디코딩은 노드에 `libnvcuvid` 가 없어 못 쓴다
- ⚠️ **vll6 의 `/data2` 에만 있다** (노드 로컬 디스크). 학습도 vll6 에서 돌릴 것

```bash
EK100_ROOT=/data2/local_datasets/EPIC-KITCHENS_resized SET="data.spatial_mode=short_side" \
  GPUS=1 bash z_research/anticipation/EK100/run.sh ek100_smoke
```


1. **비디오를 노드 로컬로 옮긴다.** 지금 `/data/dataset/EPIC-KITCHENS` 는 **NFS** 고 700개 / **1.2T** 다.
   `epoch 20 x 67,217 clip` 을 여기서 decode 하면 I/O 로 벽을 넘는다 (CLAUDE.md §1-6·§4-1).
   `/local_datasets` 는 사실상 꽉 차 있으니 **`df -h` 를 먼저 보고** 자리를 정할 것.
   옮긴 뒤에는 `EK100_ROOT=<새 경로> bash run.sh ...` 로 주면 config 를 안 고쳐도 된다.
   - 참고: val 만 먼저 보려면 val 비디오 **138개**만 옮기면 된다 (`EPIC_100_validation.csv` 의 `video_id`).
2. **batch_size 를 실측으로 정한다.** 논문 global batch 128 = 8 GPU x **16**. 그런데
   probe 20개가 4352 토큰에 backprop 하므로 OOM 가능성이 있다.
   256x256 이라 토큰은 4096 이고 **16 x 8 GPU = global 128 로 논문과 같다.** OOM 이면 `BATCH_SIZE=8`
   로 내리되 **global 이 64 로 바뀌므로 문서에 적을 것.**
3. **train 비디오도 옮겨야 한다** — probe 를 학습하므로 train 495 + val 138 = **633개**가 필요하다
   (전체 700개 중 test 67개만 빠진다). §7-2 를 볼 것.

---

## 5. 실행

```bash
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2

bash z_research/anticipation/EK100/run.sh --list                     # config 목록
DRYRUN=1 bash z_research/anticipation/EK100/run.sh ek100_vith        # 검사만. GPU 0장, 몇 초
python z_research/anticipation/EK100/selftest.py                     # 추가한 코드 자체 검사 (CPU, 몇 초)

GPUS=1 bash z_research/anticipation/EK100/run.sh ek100_smoke         # 배관 점검 (비디오 4개, 1 epoch, head 1개)
GPUS=8 bash z_research/anticipation/EK100/run.sh ek100_vith          # 본 설정 (center_crop 256x256, 논문 ap)
GPUS=8 bash z_research/anticipation/EK100/run.sh ek100_vith --val-only   # latest.pt 로 val 만

# SLURM (ANT_RUN=1 로 제출/본체가 갈린다 — 직접 sbatch 할 때 반드시 붙인다)
C=ek100_vith GPUS=8 bash z_research/anticipation/EK100/sbatch.sh
```

환경변수: `GPUS` `SET="a.b=1"` `TAG` `OUTDIR` `BATCH_SIZE` `EK100_ROOT` `LIMIT_VIDEOS`
`VAL_ONLY=1` `DRYRUN=1` `EVAL_DDP_PORT` `EVAL_DDP_TIMEOUT_S`.
`SET` 의 `data.` / `optimization.` / `classifier.` 는 `experiment.` 접두어를 생략할 수 있다.

**`evals.main` 을 직접 부르지 않는다.** `run.sh` 만 하는 두 가지가 있다 — 빈 DDP 포트 탐색과
`ANT_EXPECT_WS` 가드. `init_distributed` 는 bind 실패를 삼키고 조용히 `world_size=1` 로
폴백해서 두 job 이 섞인다 (CLAUDE.md §7-1).

### config

**전부 `spatial_mode: center_crop` (256x256) + `anticipation_point_mode: paper`** —
공식 spatial 처리에, 문맥은 action segment 시작 1초 전에서 끝난다.

| config | 무엇 | batch (global, 8 GPU) |
|---|---|---|
| `ek100_vith` | **본 설정.** ViT-H 256x256, head 1개 (lr 1e-3 / wd 1e-2) | 16 (128 = 논문) |
| `ek100_smoke` | 배관 점검. 비디오 4개 / 1 epoch / head 1개. **수치를 읽는 용도가 아니다** | 2 |

다른 spatial 을 보려면 새 yaml 을 뜨지 말고 `SET` 으로 (`configs/protocols/README.md` 관례):
```bash
SET="data.spatial_mode=short_side" GPUS=8 bash .../run.sh ek100_vith    # 짧은변 256 판 (세로 전부 보존)
SET="data.spatial_mode=letterbox" GPUS=8 bash .../run.sh ek100_vith                    # 안 자르는 판
```

`configs/*.yaml` 은 **`make_configs.py` 의 산출물**이다 (head 20개를 손으로 안 쓰기 위해).
값을 바꿀 때는 yaml 이 아니라 그 스크립트를 고치고 다시 돌린다.

### 출력

```
exp_results/action_anticipation_frozen/<tag>/
  _resolved.yaml    resolve.py 가 병합·검사한 config
  log_r<rank>.csv   epoch 별 train/val 의 acc·recall (action/verb/noun)
  latest.pt         probe 20개 + optimizer (resume_checkpoint: true 로 이어진다)
```

---

## 6. 파일과 코드 변경 내역

**새로 만든 것** (`z_research/anticipation/EK100/`):

| 파일 | 역할 |
|---|---|
| `README.md` | 이 문서 |
| `make_configs.py` | `configs/*.yaml` 생성기 (head 20개 보일러플레이트) |
| `configs/*.yaml` | 위의 산출물 5개 |
| `resolve.py` | config 검사 + `_resolved.yaml` (경로·해상도·module 정합성·프레임 예산). **모델 로딩 전에 죽는다** |
| `run.sh` | 표준 진입점. DDP 포트 탐색 + `ANT_EXPECT_WS` |
| `sbatch.sh` | SLURM. `ANT_RUN=1` 로 제출/본체 분기 |
| `selftest.py` | 추가 코드 자체 검사 (CPU, 체크포인트·데이터 불필요) |

**`evals/` 에 새로 만든 것:**

| 파일 | 역할 |
|---|---|
| `action_anticipation_frozen/exact_val.py` | val 을 정확히 한 번 도는 mean class recall@k (`evaluation.exact_val_pass`) |
| `action_anticipation_frozen/spatial.py` | `short_side`(기본) / `center_crop`(공식) / `letterbox` 세 spatial 모드 |
| `action_anticipation_frozen/modelcustom/vit_encoder_predictor_concat_ar_nonsquare.py` | 비정사각 grid + `mask_index` 설정. `module_name` 으로 고른다 — **원본 module 은 안 건드렸다** |

**기존 파일 수정 — 전부 추가 인자이고 기본값은 현행 동작이다** (총 3파일 115줄):

| 파일 | 무엇 |
|---|---|
| `dataloader.py` | `init_data`/`make_transforms`/`VideoTransform` 에 `spatial_mode`·`crop_width`·`train_spatial_mode`·`anticipation_point_mode`·`time_source` 인자 추가 |
| `epickitchens.py` | `decode_videos_to_clips` 에 `anticipation_point_mode`·`time_source` 추가 |
| `eval.py` | 위 인자들을 `args_data` 에서 읽어 전달 + `limit_videos` + `ANT_EXPECT_WS` 가드 + `evaluation` 블록 (exact val · `val_metrics.jsonl`) |

`src/` 는 **한 줄도 안 고쳤다.** predictor 의 RoPE 가 정사각을 가정하는 문제는
새 module 이 predictor block 의 `forward` 를 감싸 `H_patches`/`W_patches` 를 주입해 해결한다.

**정사각 입력에서는 새 module 이 원본과 비트 단위로 같다** (`selftest.py` [3], `max|diff| = 0.0`) —
즉 기본 설정으로 돌리면 공식 구현과 완전히 같은 계산이다.

---

## 7. `selftest.py` 가 확인하는 것 (전부 PASS)

1. spatial 세 모드 x EK100 실제 해상도 3종 -> shape, `short_side` 가 짧은변을 정확히 256 에 맞추는지
2. 비정사각 wrapper 의 토큰 계산: grid 16x28, tokens/tubelet 448, 출력 = 문맥 1792 + 예측 448
3. **정사각에서 새 module == 원본 module** (`max|diff| = 0.000e+00`)
4. H/W 주입 없이 비정사각을 넣으면 실제로 깨진다 (정사각 module 은 shape 부터 안 맞는다)
5. `released` / `paper` ap 규약의 프레임 산술이 실제로 다르다 (예시에서 200 프레임 = 4.0초 차이)
6. **exact val 지표 == 공식 `ClassMeanRecall`** (긴 꼬리 라벨, 97 / 3,568 클래스, TP+FN == clip 수)

`resolve.py` 가 막는 것 (실측 확인): 비정사각 + 정사각 module / patch 로 안 나눠지는 해상도 /
프레임 예산 초과 / `mask_index != 1` + 정사각 module / 없는 경로·비디오.

---

## 8. 아직 못 한 것 / 확인이 필요한 것

- ❌ **한 번도 실행하지 않았다.** GPU 메모리, 실제 throughput, NFS decode 속도 전부 미측정.
- ⚠️ **ViT-H 비교 수치가 레포에 없다.** 레포 `README.md` 의 EK100 표는 action R@5 **ViT-L/16 32.7**,
  **ViT-g/16-384 39.7** 두 줄뿐이다. ViT-H 값은 논문 PDF 에서 확인해야 하는데 PDF 가 레포에 없다 (미확인).
  ⚠️ 공개 수치와 우리 수치는 **세 축**이 다르다 — anticipation point(차이 2), head sweep best vs 단일 head,
  val 을 세는 방식(공식 루프 vs exact pass). 공개 수치가 릴리즈 규약으로 나온 것이면 우리 수치가 더 낮게 나오는 게 정상이다.
  그때 "재현 실패" 로 읽지 말 것 — 프로토콜이 다르다는 것을 표로 병기한다 (CLAUDE.md §1-3).
- ✅ 차이 2 는 **논문/표준 프로토콜로 결정됐다** (2026-09-12). 전 config 기본값 `paper`.
  릴리즈 동작으로 나온 공개 수치와는 **직접 비교할 수 없다** — 우리 쪽 문맥이 더 어렵다.
- ⚠️ **공식 수치와 한 축이 다르다** — spatial 은 공식과 같고(`center_crop`),
  anticipation point(`paper` vs 릴리즈). 어느 쪽도 "재현 실패" 가 아니라 **프로토콜 차이**이므로
  비교표에 두 축을 같이 적는다 (CLAUDE.md §1-3).
- ❌ predictor 출력을 빼고 encoder 만 쓴 대조군 (`wrapper_kwargs.no_predictor: true`) 은 설정만 가능하고
  config 를 따로 만들지 않았다. **"predictor 가 기여하는가" 를 재려면 이 arm 이 필요하다** —
  우리 논문(`IntPhysGenV11/Archive/PAPER_STORY_2026-09-06.md`: "predictor 는 상태를 이어가지 않는다")
  과 직접 맞물리는 축이다. `SET="model_kwargs.wrapper_kwargs.no_predictor=true"` 로 바로 된다.

---

## 재현

```bash
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
python z_research/anticipation/EK100/make_configs.py          # configs/ 생성 (멱등)
python z_research/anticipation/EK100/selftest.py              # 코드 자체 검사
GPUS=8 DRYRUN=1 bash z_research/anticipation/EK100/run.sh ek100_vith
```

§2 의 segment/label 수, §3 차이 1 의 mask token norm, 차이 2 의 84%, 차이 3 의 fps 표는 전부
`epic-kitchens-100-annotations/*.csv` · `EPIC_100_video_info.csv` · vith 체크포인트에서
직접 계산했다 (2026-09-12). 계산식은 이 문서 안에 다 적어 뒀다.

데이터: `/data/dataset/EPIC-KITCHENS` (700 MP4, 1.2T, NFS, `P*/videos/P*.MP4` = `file_format: 0`)
annotation: `/data/hyuntak/project/2026/2027_cvpr/epic-kitchens-100-annotations`
체크포인트: `configs/protocols/models.md` 의 `vith` / `vitl` 과 같은 파일
