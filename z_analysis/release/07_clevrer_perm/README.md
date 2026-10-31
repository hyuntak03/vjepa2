# ⑦ CLEVRER 영속성 v1 — 릴리즈 · VOBench PFT 채점 (2026-10-31)

> **질문** — VOBench 로 PFT 한 릴리즈 predictor 는 다른 생성기 (CLEVRER 형 Blender 장면) 의 영속성 (가려진 물체가 사라지는 미래) 으로 옮겨지나?
> **답** — **옮겨진다.** 전체 59.2 → 82.3 % (+23.1 [21, 26]), 사라짐 18.4 → 64.7 % 로 복사 기준선 (28.5) 을 +36 [30, 42] 넘는다. 다만 가림막이 물체를 따라잡는 "overtake" 운동에서는 17.8 % 로 복사 (44.4) 보다 낮다.
> **다음** — IntPhys · Rubric 에서는 안 옮겨졌는데 여기서는 옮겨진 이유 (가림 시점이 VOBench 와 비슷한가 · 장면 겉모습이 비슷한가) 를 가른다.

---

## 용어

| 용어 | 뜻 |
|---|---|
| 장면 · 네 영상 | 장면 하나에 present (있음 → 있음, 가능) · vanish (있음 → 사라짐, 불가능) · absent (없음 → 없음, 가능) · appear (없음 → 생겨남, 불가능) |
| 사라짐 A · 생겨남 B | 문맥이 같은 쌍. A = present ↔ vanish, B = absent ↔ appear. 불가능 쪽 surprise 가 크면 정답, 우연 50 |
| 운동 | static (물체 정지, 가림막이 지나감) · behind (물체가 가림막 뒤로) · cross (가로지름) · overtake (가림막이 물체를 따라잡음) |
| 복사 | 마지막 문맥 블록을 미래에 그대로 둔 기준선 (예측 없음) |
| VOBench PFT | 시트의 "V-JEPA2 w/ VoBench" — VOBench split v2 train + gravity train 828 clip 으로 30 epoch post-FT |

## 1. 무엇을 어떻게 했나
- **자료:** `testbed/cleverer_permanence_v1`. 장면 396 개 × 영상 4 개 = 1,584 개.
  - 영상마다 32 장 (문맥 16 + 미래 16, 원본 12.5 fps 에서 보폭 2), 256².
  - 물체는 미래 첫 4 장 (16–19) 동안 가려지고, 20 장째에 다시 보이거나 (가능) 안 보인다 (불가능).
- **문맥 감사:** 쌍 792 개 모두 문맥 16 장이 화소까지 같다.
- **채점:** 표준 surprise (미래 8 블록 L1 평균) · matched 쌍. 릴리즈 · VOBench PFT · 복사를 한 번의 encoder 통과로 같은 화소에서 쟀다.
- **신뢰구간:** 장면 재표집 95 %.

## 2. 결과 [측정] — 정답률 %, 우연 50

| | n | 릴리즈 | **VOBench PFT** | 복사 | PFT − 릴리즈 | PFT − 복사 |
|---|---:|---:|---:|---:|---:|---:|
| **전체** | 792 | 59.2 | **82.3** | 63.8 | +23.1 [21, 26] | +18.6 [15, 22] |
| 사라짐 A | 396 | 18.4 | **64.7** | 28.5 | +46.2 [41, 51] | +36.1 [30, 42] |
| 생겨남 B | 396 | 100 | 100 | 99.0 | 0 | +1.0 [0, 2] |

사라짐 A 를 운동 · 가림막 종류로 나누면:

| 부분 | n | 릴리즈 | VOBench PFT | 복사 |
|---|---:|---:|---:|---:|
| behind | 126 | 0.0 | **80.2** | 42.1 |
| cross | 90 | 3.3 | **57.8** | 6.7 |
| overtake | 90 | 6.7 | **17.8** | 44.4 |
| static | 90 | 71.1 | **96.7** | 15.6 |
| 가림막 = 물체 | 360 | 20.3 | 63.1 | 29.7 |
| 가림막 = 화면 판 | 36 | 0.0 | 80.6 | 16.7 |

## 3. 해석
- **릴리즈는 여기서도 사라짐을 그럴듯하게 본다** [측정]. 사라짐 A 18.4 % 로 복사 (28.5) 보다 낮다. VOBench · v11 · Rubric late 와 같은 방향이다.
- **VOBench PFT 는 이 생성기로 옮겨진다** [측정]. 복사를 신뢰구간 밖으로 넘는다. IntPhys 1 · 2 late · Rubric 에서 구분되지 않았던 것과 다르다.
- **왜 여기서는 옮겨지나는 아직 모른다** [추측]. 후보는 둘이다.
  - 가림 시점: 미래 첫 4 장에서 가렸다가 곧 다시 나타나는 시간표가 VOBench late 와 비슷하다. 이식 조사의 "PFT 는 학습셋 시간표대로 자리를 둔다" 와 맞는 쪽이다.
  - 겉모습: 단순 도형 · 단색 · 평평한 바닥이라 VOBench 와 닮았다.
- **overtake 는 PFT 도 복사보다 낮다** [측정]. 가림막이 움직이며 물체를 따라잡는 경우다. VOBench 의 가림막은 제자리에 서 있거나 일어서고 눕기만 해서, 이런 사건이 학습에 없다 [추측].

**단서**
- 이번 채점은 릴리즈 · VOBench PFT 두 모델 판이다. AR 판 · BlenderTrack 판은 아직이다.
- 이 벤치는 가림이 미래 쪽에서 시작한다 (문맥 끝은 보임 또는 일부 가림). 그래서 VOBench late (문맥 끝에서 이미 가려짐) 와 조건이 정확히 같지 않다.
- 생겨남 B 는 모든 모델이 천장 (99–100 %) 이라 변별이 없다.

---

## 부록: 재현

```bash
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
$P z_research/scripts/data/build_clevrer_perm_index.py          # → data_csv/clevrer_perm_v1/{index.csv, context_integrity.json}
D=/data/hyuntak/project/2026/2027_cvpr/testbed/cleverer_permanence_v1; R=$PWD/z_training/runs
sbatch -p batch_vll -w vll6 --gres=gpu:2 -c 16 --mem=64G --wrap "$P z_research/scripts/analysis/transfer_gap_te.py --name clevrer_perm \
  --dataset vobench_v1 --set data.root=$D data.index_csv=$PWD/data_csv/clevrer_perm_v1/index.csv data.frames_root=null window.stride=1 window.frames_start=0 \
  --preds release=$R/release_vith/latest.pt,vobftg=$R/vobench_release_postft_g/latest.pt --out $PWD/z_analysis/release/07_clevrer_perm/te --gpus 2"
$P z_research/scripts/analysis/clevrer_perm_table.py              # → results/clevrer_table.json
```

| 파일 | 무엇 |
|---|---|
| `results/clevrer_table.json` | 부분집합 × 방향 × 모델 정답률 · CI · 짝 차이 |
| `te/` | 한 번 통과 채점 원배열 (l1 · copy · meta) |
| `data_csv/clevrer_perm_v1/context_integrity.json` | 쌍 792 개 문맥 화소 일치 (불일치 0) |
