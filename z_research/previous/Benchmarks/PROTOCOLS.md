# Garrido 프로토콜 — 설정의 정본 (전면 재작성 2026-09-21)

> 이 파일이 **설정의 단일 출처**다. 수치를 보고하거나 모델을 붙일 때 여기부터 본다.
> 각 줄은 **공식 코드 파일**에서 온 것이고 경로·줄번호를 같이 적었다. 논문과 어긋나는 곳은 §6.
>
> 원전
> - 논문: `Intuitive physics understanding emerges from self-supervised pretraining on natural videos`
>   (Garrido et al.) — 레포 루트의 PDF, App. A.6–A.8 + Table S3
> - 코드: `/data/hyuntak/project/2026/2027_cvpr/jepa-intuitive-physics/evaluation_code/evals/intuitive_physics/`

**대상은 셋뿐이다 — IntPhys 1 dev · GRASP level2 · InfLevel-lab.**
IntPhys 2 프로토콜은 **범위 밖**이다 (§7).

---

## 1. 무엇을 재나

```
S_t = || p(f(V[t : t+C])) − LN(g(V[t : t+C+M])) ||_1          (논문 식 S2)

  f = context encoder (online `encoder`)      g = target encoder (EMA `target_encoder`)
  p = predictor                               LN = target 에만 거는 layer norm
  C = context 프레임 수   M = 예측 프레임 수   **window = C + M**
```

- 창은 `t` 를 **stride 2** 로 밀며 전수 계산한다 (`eval.py:409  clip.unfold(2, frames_per_clip, stride)`)
- 손실은 **latent L1**, 토큰·차원 평균 (`eval.py:485  F.l1_loss(...).mean((2,3))`)
- **AvgSurprise** = 창 평균. **이것 하나만 쓴다** (`eval.py:280`, §1-b)
- 쌍 비교는 `possible < impossible` 이면 정답, **동점은 오답**. chance 50%

우리 구현: `configs/protocols/intphys1_sliding.yaml` (`surprise.mode: intphys1`),
채점기 `evals/world_model_analysis/eval.py:run_surprise_intphys1`.

---

## 1-b. 지표 — **쌍 비교는 Avg, 단일 영상은 Max** (논문이 직접 가른다)

공식 `compute_metrics` 는 한 CSV 에 10 개 지표를 찍는다 (`eval.py:282-324`):
`Relative Accuracy (avg|max)` · `Absolute Accuracy (max)` · `Best Absolute …` · `AUPRC (avg|max)` · `AUROC (avg|max)`.

**보고되는 것은 `Relative Accuracy (avg)` 하나다** — 노트북의 `key_metric` 이 **14 곳 전부** 그것이다.
본문도 같다:

> *"We summarize the performance of methods across datasets on **pairwise classification**
>  (i.e., detecting the impossible video in a pair) ... achieving average accuracies of
>  **98%** (IntPhys), **66%** (GRASP), **62%** (InfLevel-lab)"*  — §2

그리고 **왜 쌍 비교는 Avg 이고 단일 영상은 Max 인지**를 부록이 직접 보여준다:

| | 지표 | V-JEPA-H 오류율 (IntPhys test, All) |
|---|---|---|
| **Table S4** 쌍 비교 | Max | 4.4 / 4.4 / 12.87 % |
| | **Avg** | **0.28 / 0.0 / 0.09 %** ← 훨씬 낫다 |
| **Table S5** 단일 영상 (1−AUROC) | **Max** | **19.2 / 21.9 / 29.67 %** ← 이쪽이 낫다 |
| | Avg | 38.3 / 39.2 / 37.05 % |

> Table S5 캡션: *"the average surprise of a video is **not a good metric**, possibly due to
> values being too dependent on the experimental setup."*

**→ 우리는 쌍 비교만 한다. 지표는 `Avg` 하나다.**

⚠️ **2026-09-21 — `Max` 는 코드에서 철회했다.** 분기를 두지 않는다.
`configs/protocols/intphys1_sliding.yaml` 의 `aggregate: [avg]`, `garrido_rescore.py` 도 평균만 낸다.
단일 영상 수치가 필요해지면 그건 **`Max` + AUROC** 라는 다른 과제이고 (Table S5),
지금 구현에는 AUROC 가 없으므로 **새로 만들어야 한다.**

### 대조값 (그들 모델 `V-JEPA-h + RoPE`, 공개 릴리즈가 아니다)

| | IntPhys | GRASP | InfLevel-lab |
|---|---:|---:|---:|
| V-JEPA (Garrido) | **98%** | **66%** | **62%** |

우리 V-JEPA 2 ViT-H 는 다른 모델이라 같을 이유는 없지만, **자릿수가 크게 어긋나면 파이프라인을 의심할 근거**가 된다.

### "최고를 보고한다" 는 논문이 명시한 선택이다

> *"This process is done for all methods, and leads to results **illustrating the best performance
>  achievable by the model**. We expand on this choice in section B."*  — §2

즉 A.8 탐색 + property 별 최고는 **의도된 descriptive 보고**다. held-out 추정치가 아니다.

## 2. 하이퍼파라미터는 **모델마다 탐색하고 최고를 보고한다**

논문 A.8 은 탐색 *공간*을 준다 — *"For every method, we use the following hyperparameters per dataset"*:

| 데이터셋 | Frame skip | Window size (**= C + M**) | Context lengths |
|---|---|---|---|
| IntPhys | **[2, 5, 10]** | **[16, 32]** | [2,4,6,8,10] × (C+M)/16 |
| GRASP | **[2, 5, 10]** | **[16, 32]** | [2,4,6,8,10] × (C+M)/16 |
| InfLevel-lab | **[5, 10, 20]** (V-JEPA·VideoMAEv2) | **[16, 32]** | [2,4,6,8,10] × (C+M)/16 |

**Table S3 는 그 탐색의 *결과*다.** 표에 `Method` 열이 있고 값이 모델마다 다르다
(InfLevel 에서 V-JEPA 창 32, VideoMAEv2 창 16). 공식 `configs/default_*.yaml` 은
**그들 모델**(`vit_huge_rope`, fpc 16 학습, 224)의 선택값을 적어 둔 파일이지 고정 규격이 아니다.

> ⚠️ **2026-09-21 정정.** 그 전까지 우리는 Table S3 의 한 칸만 돌렸고 그건 틀렸다.
> **V-JEPA 2 는 Table S3 에 없는 method** 이고 fpc **64** @256 으로 학습돼 16 프레임 창에
> 가두면 손해를 본다. IntPhys1 dev 실측:
>
> | 칸 | avg |
> |---|---:|
> | skip2_w16 (Table S3 가 *그들 모델*에 고른 칸) | 83.33 |
> | **skip2_w32 (A.8 탐색 최고 → 우리 보고값)** | **88.89** |
> | skip5_w16 | 61.11 |
> | skip5_w32 · skip10_* | 프레임 부족으로 성립 불가 |
>
> 88.89 는 IntPhys 2 논문이 V-JEPA 2-h 에 보고한 **87.22** 와 160/180 vs 157/180, **1.7pt** 차다.

격자는 **한 실행 안에서 전부 돈다**. 프레임이 모자란 칸(`99//skip < window`)은 하네스가 자동으로 뺀다.

**해상도는 모델을 따라간다** — V-JEPA 2 / 2.1 → **256**, VideoMAEv2 → **224** (patch 14).
공식 yaml 의 224 는 그들 모델 해상도다.

**VideoMAEv2 도 창 32 를 돌 수 있다.** 시간 위치 임베딩이 체크포인트에 없고
`get_sinusoid_encoding_table(num_patches, ...)` 로 **생성**되는 값이기 때문이다
(`videomae.py:52`). 논문도 [16,32] 를 탐색했고 Table S3 에서 16 을 *골랐을 뿐*이다.

---

## 3. context 축약 — **IntPhys 은 Filtered, 나머지는 property별 최고 C**

논문 텍스트가 두 곳에서 같은 말을 한다:

> **A.7 (870-872)** *"For each property, we select the value of C (C+M being fixed) which maximizes
> performance ... **On IntPhys, we are able to get rid of this sweep on context lengths** by computing
> the minimal surprise over all context lengths C for each starting frame t."*

> **A.8 (992-996)** *"For all properties, we choose the context size which gives us the best performance.
> ... For IntPhys, ... the minimal surprise over all windows for each start frame can be used as it
> **removes one hyperparameter to optimize** and helps filter surprise spikes coming from possible events."*

**규칙**

| 데이터셋 | 축약 후보 | 집계 |
|---|---|---|
| **IntPhys 1** | **`Filtered` 하나** (= `all_losses.min(1)`, 시작점마다 C 최솟값) | property macro 평균 |
| **GRASP · InfLevel** | **C 각각** 중 property 마다 최고 | property macro 평균 |

`Filtered` 는 공식 `eval.py:260` 이 CSV 에 한 줄 더 찍는 값이고, C 스윕을 **없애기 위한** 것이다.
macro 는 `figures.ipynb` 의 `np.mean(perfs)` (쌍 수 가중이 아니다).
(skip, window) 는 **데이터셋 수준**에서 고른다 (Table S3 가 method × dataset 마다 한 조합).

### ⚠️ 논문 텍스트와 노트북이 어긋난다 — **텍스트를 따른다**

공식 `figures.ipynb` 의 본 결과 함수는
```python
acc = df[df["Block"] == prop][key_metric].max()     # Filtered 행과 C 행을 **함께** 놓고 최대
```
라서 **IntPhys 에서도 C 스윕을 도로 집어넣는다.** A.7/A.8 의 *"get rid of this sweep"*,
*"removes one hyperparameter"* 와 정면으로 어긋난다.

우리는 **텍스트를 따르고**, 노트북 규칙 값도 `garrido_rescore.py` 가 `alt_notebook` 으로 같이 낸다.

**실측 차이 (IntPhys1 dev × ViT-H, `skip2_w32`)**

| 축약 | O1 | O2 | O3 | macro |
|---|---:|---:|---:|---:|
| **Filtered** (텍스트 규칙) | 85.00 | 96.67 | 85.00 | **88.89** |
| C4 | 88.33 | 96.67 | 86.67 | 90.56 |
| C8 | 86.67 | 96.67 | 83.33 | 88.89 |
| C12 | 85.00 | 96.67 | 85.00 | 88.89 |
| C16 | 78.33 | 83.33 | 83.33 | 81.67 |
| C20 | 73.33 | 70.00 | 80.00 | 74.44 |
| 노트북 규칙 (property별 최고) | 88.33 | 96.67 | 86.67 | 90.56 |

⚠️ **함정 — `summary.json.best_avg` 는 Filtered + 쌍 수 합산이다.** IntPhys 은 property 별 n 이
60 으로 같아 macro 와 값이 같지만(88.89), **GRASP·InfLevel 은 규칙 자체가 달라 일치하지 않는다.**
공식 값은 `z_research/scripts/analysis/garrido_rescore.py` 가 `per_window.json` 에서 다시 계산한다.

## 4. 프레임 예산 — 데이터셋마다 다르다

`extract_losses:355` 가 결정한다:
```python
sampling_rate, num_frames = frame_step, 99 // frame_step   # 이 num_frames 가 dataset 의 clip_len
```

| 데이터셋 | 공식이 읽는 프레임 | 우리 |
|---|---|---|
| **IntPhys** | `length_clip = (99//skip)*skip` → skip 2 에서 **49 장**, 창 17 개 | `frame_budget: official` = `(n_frames−1)//skip` = 49 ✓ |
| **GRASP** | `length_clip` 을 안 쓰고 `arange(len(vr))[::skip]` = **영상 전체** (501 → 51 장) | 50 장. `t ≤ n−window` 라 **창 목록 동일** ✓ |
| **InfLevel** | 전체를 읽되 **priming 을 잘라낸 뒤** (§5) | `frame_start` 컬럼 ✓ |

---

## 5. 데이터셋별 세부

### 5-1. IntPhys 1 dev
- 4중항 90개 = 영상 360. **matched pairing 2쌍/블록** — `get_matches(get_breaking_points(clip))`
  (`utils.py:168`) 이 픽셀 분기점으로 짝을 정한다. 우리 index 의 `(block_id, pair_id)` 와 같다
- 프레임은 mp4 가 아니라 **원본 PNG 직독**
- property = `block_type` ∈ {O1, O2, O3} (`utils.py:69`)

### 5-2. GRASP level2
- `P_{property}/{scene}` vs `IP_{property}/{scene}` **한 쌍** (`grasp_dataset.py:100-108`),
  labels `[1,0]`. 공식 `matches = [[0,1]]` (`eval.py:406`)
- property = `block_type`, **16 종** (Collision … Unchangeableness2, `utils.py:70-85`)
- 영상 4096 = 쌍 2048

### 5-3. InfLevel-lab — 공식 로더가 **priming 을 잘라낸다**

> ⚠️ **2026-09-21 전면 재작성. 이전 인덱스와 수치는 전부 폐기했다**
> (`exp_results/_invalid_20260921_inflevel/`). 세 군데가 공식과 달랐다.

정본은 `evaluation_code/auxiliary_data_loading_files/inflevel/{property}.csv` 다.
**한 줄이 한 쌍**이고 `vid1` 은 항상 possible, `vid2` 는 항상 impossible:

| property | 쌍 | priming 절단 후 프레임 |
|---|---:|---:|
| continuity | **1116** | 300 |
| gravity | **1182** | 210 |
| solidity | **450** | 300 |

```python
# inflevel_dataset.py:133-142  (priming=False)
diff   = (len1 − end_priming_2_1) − (len2 − end_priming_2_2)
start1 = end_priming_2_1 + (diff if diff > 0 else 0) + 1
start2 = end_priming_2_2 + (−diff if diff < 0 else 0) + 1
frames_i = arange(len_i)[start_i :: frame_step]
```

- **priming(컵 내부를 보여주는 구간)을 버린다.** 그 위치는 행마다 다르다 (gravity 159~194)
- `diff` 는 두 영상의 **잔여 길이를 맞추기 위한 것**이라 적용 뒤 남는 프레임 수는 같다
- **채점 `matches=[[0,1]]`** → `scoring.pairing: matched`.
  cross(2×2) 를 쓰면 쌍 수가 정확히 2 배가 된다 (이전 판이 그랬다)
- 우리는 잔여 프레임을 **최솟값으로 잘라** 모든 쌍이 같은 수의 window 를 갖게 했다.
  공식은 0 패딩하는데 스스로 `"can lead to slighlty innacurate metrics"` 라고 적어 둔 경로다
- 시작점은 index 의 `frame_start` → `data.frames_start_column`.
  `data.py` 의 mp4 경로가 `frames_start`/`frames_stride` 를 **무시하던 버그**도 같이 고쳤다

⚠️ **gravity·solidity 는 원리적으로 못 푼다** — 컵이 잘렸는지가 **잘라낸 priming 에만** 나온다
(Garrido App. E). 50 근처가 정상이고 **모델 비교에 쓰지 않는다.** 믿을 수 있는 건 continuity 뿐이다.

---

## 6. 논문 텍스트와 공식 코드가 어긋나는 곳 (기록)

| 항목 | 논문 | 코드 | 우리 선택 |
|---|---|---|---|
| 하이퍼파라미터 | A.8 = **탐색 공간** | `default_*.yaml` = 그들 모델의 **선택값** 한 칸 | **A.8 탐색**, 모델마다 최고 보고 |
| context 선택 | IntPhys=min, 그 외=property별 최고 | 코드는 둘 다 출력 | **논문대로** (§3) |
| 해상도 | 명시 없음 | `default_*.yaml` 224 (그들 모델) | **모델을 따른다** |

---

## 7. 범위 밖 — IntPhys 2

IntPhys 2 (Bordes et al.) 는 **다른 프로토콜**이다. 섞지 말 것:

| | Garrido | IntPhys 2 |
|---|---|---|
| 창 | `unfold(2, C+M, 2)` — C+M 이 곧 창 | `unfold(2, C + num_frames_to_pred, 2)`; `-1` 이면 config 의 48 |
| C 목록 | 창 × {1/8 … 5/8} | 창 × {1/4 … 7/8} — **더 긴 문맥** |
| 축약 | 시작점마다 C 최솟값 | min 없음, **최고 C 보고** |
| 추가 | 없음 | `max_context_mode` — 맨 앞 창에서 C 를 키워가며 곡선 앞부분을 메움 |
| fps | skip 이 결정 | **6 고정** |

우리 IntPhys 2 구현(`analysis/intphys2/`)은 IntPhys 1 에 걸면 67 이 나오는데 논문은 87.22 다.
**미규명.** 그래서 **IntPhys 1 은 Garrido 프로토콜로만 보고한다.**

---

## 재현

```bash
# 격자 전체 (한 실행이 A.8 공간을 다 돈다)
GPUS=8 BENCHES="intphys1_dev" MODELS="vith" bash z_research/Benchmarks/run_all.sh

# 공식 축약 규칙으로 재채점 (GPU 불필요)
python z_research/scripts/analysis/garrido_rescore.py --all

# 표
python z_research/scripts/analysis/bench_table.py --readme
```
