# ③ binding 학습 — L1 에 "물체 자리 L1" 항 + 운동 head (2026-10-31)

> **질문** — late 에서 정체를 다시 나타날 자리에 묶도록 손실에 항을 더하면 (binding), 처음 보는 궤적 · 다른 생성기에서 영속과 판독기 이식이 오르나?
> **답** — **같은 생성기의 처음 보는 궤적에서는 오른다.** L1 만 쓴 PFT 대비 late 전체 +8.5, 사라짐 A +19.4, 문맥 z 판독기 → p 이식 모양 62 → 74 % · 색 61 → 75 %. **다른 생성기 (v11 · IntPhys) 에서는 이득이 없다.** 운동 head 는 binding 위에 더하는 것이 없다.
> **다음** — 일반화가 막히는 곳이 데이터 (궤적 21 개) 인지 확인한다. 궤적을 연속으로 흩뜨린 데이터에서 같은 세 팔을 돌리거나, 자연 영상 L1 을 섞는다 (사용자 결정).

> ⚠️ **정정 (2026-10-31)** — 아래의 "처음 보는 궤적" 은 틀린 말이다. 궤적 묶음으로 나눠도 test late 장면의 미래 칸 경로 66.7 % · 끝 칸 100 % 가 학습 장면에 있다 (VOBench late 의 미래 경로가 21 가지뿐). 경로로 다시 나누면 binding 의 이득은 **학습에 있던 경로에만** 있다: 본 경로 a1 − a0 사라짐 A +35.2 · late 전체 +13.9 / 처음 보는 경로 (flat_acc test) +2.8 · +1.9. 운동 head (a2) 는 둘 다 0.5 pt 안. 그래서 이 문서의 "처음 보는 궤적에서 오른다" 는 **"같은 생성기 · 본 경로에서 오른다"** 로 읽는다. 다음 단계는 경로를 연속으로 흩뜨린 학습셋 → `../06_blender_track/`.

---

## 용어

| 용어 | 뜻 |
|---|---|
| binding 항 | 참 미래 물체 자리 (블록 9–16 마다 중심 칸의 3×3) 토큰만의 \|p − LN(h)\| 평균. L1 에 λ = 1.0 배로 더한다. 물체 3×3 은 전체 칸의 약 3.5 % 라 그 칸의 무게가 약 25 배가 된다 |
| 운동 head | predictor 마지막 층의 문맥 토큰 → 물체의 c₀ (블록 8 자리) · v · a · 있음. 블록 1–16 자리 = c₀ + vτ + ½aτ² (τ = 블록 − 8). SmoothL1 (칸) × 0.1 + 있음 BCE. 시험 때는 안 쓴다 |
| a0 · a1 · a2 | 릴리즈 predictor 에서 출발한 PFT 세 팔. a0 = L1 만 (대조) · a1 = L1 + binding · a2 = a1 + 운동 head. 데이터 · step · lr · 증강 · 배치가 같다 |
| 궤적 묶음 | (물체 세트, 배경, 운동, 가림, k). VOBench 의 궤적 · 가림막은 이 다섯으로만 정해진다. 학습 · 평가를 이 단위로 나눠 평가 궤적은 학습에 없다 |
| 복사 | 마지막 문맥 블록 latent 를 미래 8 블록에 그대로 둔 기준선 (예측 없음) |
| 사라짐 A / B | permanence 짝에서 물체가 있던 문맥 쪽 (A) · 빈 문맥 쪽 (B) 정답률 |
| 이식 | 문맥 z 로 학습한 판독기 (attentive, 블록 1–8) 를 p 에 건 정확도 |

---

## 1. 왜 했나
- late 의 병목은 "과거 물체를 다시 나타날 자리의 토큰에 쓰는 일 (binding)" 이다 (`../SUMMARY_visible_late_pft_transfer.md`).
- 표준 L1 에서 물체 칸 몫은 약 4 % 라, 정체를 화면에 퍼뜨리는 것과 자리에 묶는 것을 손실이 크게 가르지 않는다 [측정].
- 가설: binding 항을 더하면 묶기가 생기고, 그것이 처음 보는 궤적과 다른 생성기로 옮겨진다.

## 2. 무엇을 어떻게 했나
- **데이터:** VOBench_v2 의 궤적 묶음 240 개 중 120 개 (층마다 절반) 의 가능 영상 2,160 개 + 프레임별 물체 중심 라벨 (시뮬레이터). 증강 (좌우 반전, 시작 0–2) 은 라벨에도 똑같이 적용했다.
- **학습:** 릴리즈 predictor 에서 출발. 전역 배치 32, 11 epoch = 737 step, lr 5e-5 (원 PFT 와 같은 레시피 · 같은 step 수). vll5 GPU 2 장씩, seed 0.
- **평가 (모델 다섯 + 복사를 같은 화소 · 같은 수치로, 한 번 통과 채점):**

  | 판 | 생성기 | 표본 |
  |---|---|---|
  | VOBench_v2 test late + visible | 같은 생성기, 처음 보는 궤적 | 432 장면 |
  | v11 가림 이동 사라짐 | 반쯤 다른 생성기 (배경 · 메시 일부 공유) | 224 장면 |
  | v11 실물 메시 | 반쯤 다른 생성기 | 120 장면 |
  | IntPhys 1 late 이동 · 가림 | 다른 생성기 | 60 쌍 |
  | IntPhys 2 late permanence | 다른 생성기 | 120 쌍 |

- **판독:** VOBench_v2 late · visible 에서 학습 쪽 장면으로 판독기를 학습하고 test 장면에서 블록마다 쟀다 (attentive, ① 과 같은 판독기).

## 3. 결과 [측정]

**3-1. 같은 생성기 · 처음 보는 궤적 (VOBench_v2 test)** — 정답률 %, 괄호는 같은 짝 차이의 95 % CI

⚠️ 정정 — 이 test 는 궤적 묶음이 다를 뿐 미래 경로는 66.7 % 가 학습에 있다 (맨 위 정정). 이득은 본 경로에 몰려 있다.

| 칸 | 릴리즈 | a0 | **a1** | a2 | 복사 | a1 − a0 | a1 − 복사 |
|---|---:|---:|---:|---:|---:|---:|---:|
| late 사라짐 A | 12.0 | 43.5 | **63.0** | 58.3 | 55.6 | **+19.4 [12, 27]** | +7.4 [−5, 19] |
| late 사라짐 B | 95.4 | 100 | 100 | 100 | 66.7 | 0 | +33.3 |
| late 모양 | 58.3 | 70.8 | **80.1** | 79.2 | 50.5 | **+9.3 [6, 13]** | +29.6 [24, 36] |
| late 색 | 70.4 | 77.3 | **83.8** | 83.3 | 65.3 | **+6.5 [3, 10]** | +18.5 [11, 26] |
| late 전체 | 60.8 | 73.3 | **81.8** | 80.6 | 59.0 | **+8.5 [6, 11]** | +22.8 [19, 27] |
| visible 전체 | 88.0 | 93.5 | 96.3 | 96.8 | 90.3 | +2.8 [0, 6] | +6.0 [3, 10] |

판독 (late, test 장면, 블록 9–16 평균, k 평균):

| | 릴리즈 | a0 | **a1** | a2 |
|---|---:|---:|---:|---:|
| **z 판독기 → p 이식, 모양** (우연 16.7) | 24.7 | 61.6 | **74.4** | 72.5 |
| **z 판독기 → p 이식, 색** (우연 12.5) | 10.3 | 60.6 | **74.5** | 73.4 |
| p 자기 판독, 모양 / 색 | 71.4 / 37.8 | 77.2 / 64.9 | **87.1 / 78.2** | 83.7 / 76.8 |

- 같은 판에서 문맥 z 자기 판독은 보이는 블록 99 / 97 %, 가린 블록 8 은 84 / 79 % 다.
- visible 의 이식: 릴리즈 67 / 63 → a0 84 / 80 → a1 86 / 82 % (모양 / 색).

**3-2. 다른 생성기**

| 판 | 릴리즈 | a0 | a1 | a2 | 복사 | a1 − a0 | a1 − 릴리즈 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v11 사라짐 A | 1.8 | 83.9 | 81.7 | 82.1 | 25.4 | −2.2 [−5, 0] | +79.9 |
| v11 실물 메시 사라짐 A | 5.8 | 90.0 | 86.7 | 85.8 | 13.3 | −3.3 [−8, 2] | +80.8 |
| IntPhys 1 late (쌍) | 55.0 | 53.3 | 61.7 | 61.7 | 53.3 | +8.3 [0, 17] | +6.7 [−3, 18] |
| IntPhys 2 late (쌍) | 59.2 | 60.0 | 61.7 | 61.7 | 50.0 | +1.7 [−2, 6] | +2.5 [−3, 8] |

- 사라짐 B 는 v11 · v11 실물 메시에서 모든 모델 100 % 다.

**3-3. 학습 중 값** — 마지막 epoch

| | a0 | a1 | a2 |
|---|---:|---:|---:|
| 학습 데이터 전체 L1 | 0.477 | 0.508 | 0.510 |
| binding 칸 L1 | — | 0.396 | 0.400 |
| IntPhys 1 dev val (60 쌍, 학습 루프 안) | 70.0 | 70.0 | 70.0 |
| 운동 head: 학습 데이터 미래 자리 오차 · 1 칸 안 | — | — | 0.14 칸 · 100 % |

릴리즈의 IntPhys 1 dev val 은 66.7 % 다.

## 4. 해석
- **binding 항은 같은 생성기에서 묶기와 영속을 함께 올린다** [측정]. ⚠️ 정정 — "처음 보는 궤적" 이 아니라 학습에 있던 미래 경로에서다 (맨 위 정정: 본 경로 +35.2 · 처음 보는 경로 +2.8, 사라짐 A).
  - 이식 (z 판독기 → p) 이 모양 +13 · 색 +14 pt 오른다.
  - p 자기 판독도 오른다.
  - 표준 채점은 late 전체 +8.5, 사라짐 A +19.4 다.
- **사라짐 A 는 아직 복사 수준이다** [측정]. a1 63.0 vs 복사 55.6 (+7.4 [−5, 19]). 모양 · 색 · late 전체는 복사를 크게 넘는다.
- **L1 만 쓴 PFT (a0) 는 처음 보는 궤적에서 사라짐 A 가 43.5 % 로 복사보다 낮다** [측정]. 앞선 조사의 "PFT 가 late 를 푼다" (분포 안 98–100 %) 는 학습셋과 궤적을 공유한 test 의 값이었다. 끝 자리 사전 결과와 같은 방향이다 [해석].
- **다른 생성기로는 binding 의 이득이 옮겨지지 않는다** [측정].
  - v11 계열: a0 의 큰 이득 (VOBench → v11, 배경 · 메시 공유) 은 그대로고, binding 은 더하지 않는다 (−2 ~ −3, CI 0 포함).
  - IntPhys 1 · 2: 모든 팔이 릴리즈 · 복사와 구분되지 않는다 (60 · 120 쌍).
- **운동 head 는 binding 위에 더하는 것이 없다** [측정]. head 는 학습 궤적을 0.14 칸으로 맞히는데, 학습셋 궤적이 적어 외웠을 수 있다 [추측]. test 장면에서 head 의 자리 정확도는 안 쟀다.
- **binding 은 전체 L1 을 6 % 나쁘게 한다** [측정]. 물체 칸 쪽으로 무게를 옮긴 대가다. IntPhys 1 학습 중 val 은 안 떨어졌다.

**단서**
- seed 하나다.
- **원 PFT (vobftg) 의 VOBench_v2 test 값 (사라짐 A 100 %, late 전체 89.8 %) 은 비교하지 않는다.** VOBench_v2 replica 0 은 VOBench_v1 과 같은 렌더라, 원 PFT 는 이 test 의 궤적 (일부는 같은 영상) 을 학습했다. 다른 생성기 열의 원 PFT 값은 유효하다: v11 A 90.2 · v11 실물 메시 A 96.7 · IntPhys 1 53.3 · IntPhys 2 60.0.
- 라벨 (물체 자리) 은 시뮬레이터 특권 정보다. "관측만으로 학습" 의 판이 아니다.
- binding 의 λ (1.0) 와 반경 (3×3) 은 하나만 돌렸다.
- IntPhys 판은 표본이 작다 (60 · 120 쌍).

---

## 부록: 재현

```bash
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
$P z_training/data/build_vobench_v2_split_index.py                     # 궤적 묶음 split (data_csv/vobench_v2_split/)
for A in a0 a1 a2; do NODE=vll5 GPUS=2 CPUS_PER_GPU=12 MEM_PER_GPU=48G TIME=0-05:00:00 SET="data.num_workers=10" \
  bash z_training/sbatch.sh bind_v2_$A; done                           # configs/training/bind_v2_{a0,a1,a2}.yaml, ~45 분
# 표준 채점 (하네스): VOBench_v2 test late + visible
sbatch ... --export=ALL,WMA_RUN=1,P=surprise_c16t32,D=vobench_v2_test_lv,M=vith,GPUS=2 z_research/scripts/sbatch.sh     # 릴리즈
GPUS=2 bash z_training/eval.sh bind_v2_<팔> vobench_v2_test_lv
$P z_research/scripts/analysis/release_bind_score_table.py              # → results/score_table.json
# 한 번 통과 채점 (모델 다섯 + 복사, 판 다섯): z_research/scripts/analysis/transfer_gap_te.py --name bind_<판> --dataset ... --preds release=...,a0=...,a1=...,a2=...,vobftg=...
$P z_research/scripts/analysis/release_bind_te_table.py                 # → results/te_table.json
# 판독: p 추출 (팔마다) → 학습 · 평가 split 판독
$P z_research/scripts/analysis/release_v2_extract.py --occ late --what p --frames /data2/local_datasets/world/Benchmark/VOBench_v2 \
   --pred-ckpt z_training/runs/bind_v2_<팔>/latest.pt --out /data/hyuntak/project/2026/2027_cvpr/nfs_cache/bind_v2/<팔>_late
$P z_research/scripts/analysis/release_late_block_probe.py attn_split --occ late --task shape    # → probe/late_shape.json
```

| 파일 | 무엇 |
|---|---|
| `results/score_table.json` | 하네스 표준 채점 (VOBench_v2 test, 릴리즈 · 세 팔) |
| `results/te_table.json` | 한 번 통과 채점 (판 다섯 × 모델 다섯 + 복사, 짝 차이 CI) |
| `probe/{late,visible}_{shape,color,exist}.json` | 판독 (z 자기 · p 자기 · z → p 이식, 모델 넷) |
| `te/<판>/` | 한 번 통과 채점 원배열 (l1 · copy · meta) |
| `eval/` | 릴리즈 하네스 채점 산출물 |
| `logs/` | 평가 job 로그 |

- 코드: `app/vjepa_frozen/where_head.py` (데이터셋 · 운동 head · binding 항) · `app/vjepa_frozen/train.py` (`where_head` 설정) · `app/vjepa_frozen/data.py` (`frames_index_track`, 가중 표본기 seed 수정) · `z_training/harness/resolve_train.py`.
- 학습 run: `z_training/runs/bind_v2_{a0,a1,a2}/` (latest.pt 에 운동 head 는 `where_head` 키).
