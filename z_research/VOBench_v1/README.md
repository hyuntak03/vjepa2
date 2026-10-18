# VOBench_v1 — 시작점 (2026-10-10 개설)

> 마지막 갱신 2026-10-11 (7 차).
> - 7 차 (10-11): **gravity — Ariel AR ep40** (§3-14) — 멈추지 않지만 등속 · 높이 유지 후보를 고른다 (내려가는 중 진짜 포물선 0 / 24).
> - 6 차 (10-11): **gravity_horizon** (§3-13) — 예측 길이를 16 → 8 · 4 · 2 로 줄여도 창의 끝에서 진짜 운동을 떠난다 (전체 토큰 기준, 사용자 지시). 공 칸만 보면 고정 순번.
> - 5 차 (10-11): **gravity 재렌더판** (g 375, 꼭대기 f66 · f48 · f30 = 오르는 중 · 꼭대기 · 내려가는 중) 으로 §3-12 다시 냄. 옛 판은 `_previous_20261010_*`.
> - 4 차: **gravity (ledge · arc) 추가** — §3-12, 정리본 `vjepa2/gravity/`. 데이터 `…/VOBench_v1/gravity` (block = 과거 하나 + 미래 여럿, cross 채점).
> - 3 차: **primitive · realistic 을 따로 기록한다** (사용자 지시 — 물체 · 색 집합이 달라 합친 평균을 쓰지 않는다). 표 · 그림 · 본문 전부 objset 별로 다시 냈다. 합친 수치는 참고로만 남긴다.
> - 2 차: 복사 기준선 추가. 1 차의 "바꿈이 보이면 81–91 %" 는 예측 없이도 나오는 점수였다.
>
> 전 표: 모델별 — 표 `<모델>/surprise/tables.{md,json}`, 그림 `<모델>/surprise/figures/` (모델 = `vjepa2` · `dino_foresight`). V-JEPA 2: [`vjepa2/surprise/tables.md`](vjepa2/surprise/tables.md) · [`vjepa2/surprise/figures/`](vjepa2/surprise/figures/) (§3-6).
> 수치는 전부 `per_block.json` · 복사 실행 배열에서 다시 계산했고 summary.json 의 overall · 24 칸과 일치를 확인했다 (스크립트가 assert).

> **v-jepa2 post-FT (2026-10-14)**: 릴리즈 predictor 를 VOBench 가능 영상 (split v2, 누수 없음) 으로 post-FT — [`vjepa2_pft/`](vjepa2_pft/README.md)

> **종합 (2026-10-12)**: 이 세트의 모든 결과 (VoE · IntPhys 1 · 2 · encoder probe · AR 자 · 속도 감쇠 · post-FT · 예측 길이) 를 한 문서로 — [`Archive/RESULTS_SUMMARY_2026-10-12.md`](Archive/RESULTS_SUMMARY_2026-10-12.md)

## 폴더 (2026-10-10 개편 — 모델 → 작업)

```
z_research/VOBench_v1/
├── README.md                  이 문서 (데이터 · 프로토콜 · 모델별 결과 요약)
├── <모델>/surprise/           채점 정리본: README.md · tables.{md,json} (objset 별) · figures/ (fig_timing · fig_k · fig_late_motion · fig_permanence_context)
├── vjepa2/probing/            튜블릿 정체 probe: README.md · results/<run>/ · figures/ (fig_probe_transfer · fig_probe_by_t)
├── vjepa2/gravity/            gravity (ledge · arc): README.md · tables.{md,json} · figures/ (fig_gravity_choice{,_ball}_<objset> · fig_arc_choice) · _previous_20261010_g981/ (옛 판)
├── ariel_ar_ep40/gravity/     gravity Ariel AR 판 (§3-14): README.md · tables.{md,json} · figures/ (fig_gravity_choice{,_ball}_<objset> · fig_arc_choice)
├── vjepa2/gravity_horizon/    예측 길이 M 16 · 8 · 4 · 2 비교 (gravity arc): README.md · tables.{md,json} · figures/ (fig_horizon_all_<objset> 주 · fig_horizon_ball_<objset> 보조)
├── analysis/attention_map/   AR predictor rollout attention (2026-10-13 · 10-14 정리): README.md · <모델>/{<set>.json,<set>_r*.npz,figures/} · compare/
└── exp_results/               하네스 원시 산출물 (summary.json · per_block.json · copy 배열) — 레지스트리 results_root 라 여기 둔다
```
모델 = `vjepa2` (V-JEPA 2 ViT-H 릴리즈) · `dino_foresight` · `ariel_ar_ep40` · `ariel_ar_state_ep40` · `vast_ar_state_ep70` · `videomae_v2` · `vjepa2_pft` (release + VOBench post-FT). 새 모델 · 새 작업도 같은 양식 (`<모델>/<작업>/README.md`).

## 0. 결론

| | primitive | realistic |
|---|---:|---:|
| overall (A / B) | **71.7** [69, 75] · 65/78 | **81.9** [79, 85] · 74/90 |
| late (바꿈이 가려진 칸) | **57.9** · 45/70 | **65.7** · 48/83 |
| 복사 기준선 · Δ | 73.8 · −2.1 [−5.4, +1.1] | 81.9 · 0.0 [−2.8, +2.8] |

**두 objset 모두 ViT-H 가 복사 기준선을 넘지 못한다.** 바꿈이 보이는 조건 (visible · early · mid) 은 예측 없이 풀리고, 복사로 안 풀리는 late 에서 모델은 primitive 57.9 · realistic 65.7 이다.

근거 세 줄:
1. late permanence 는 **두 objset 모두** 움직이는 물체 네 운동 (const. vel · accel · decel · ramp down) 에서 **사라짐 0 % / 나타남 100 %** — v11 과 같은 비대칭이 물체 집합에 상관없이 나온다.
2. late 는 k = 1–3 에 평평하다 (permanence primitive 54 / 54 / 58 · realistic 62 / 62 / 58).
3. primitive 는 가림이 없어도 shape · color 가 75 % 다 (realistic 100). 바꿈이 보이는 칸에서 primitive 가 낮은 것은 복사도 같다 (§3-4) — 물체 집합의 구별 난이도이지 모델 효과로 쓰지 않는다.

미결: early 에서 모델이 복사보다 낮은 칸 (primitive shape −12.5 · realistic permanence −6.9) 의 원인 · late 에서 복사가 chance 보다 조금 높은 이유 (61 %).

## 1. 데이터

`/data2/local_datasets/world/Benchmark/VOBench_v1` (vll5 노드 로컬, 데이터 README 는 그 폴더의 `README.md` · `review/INDEX.md`).
UE 5.4 TestBed. 720 block × 4 arm = 2,880 clip, matched pair 1,440 (A = pos_a·imp_ab, B = pos_b·imp_ba). 96 프레임 · 30 fps · 512 px, splice 48.

| 요인 | 값 |
|---|---|
| objset | primitive (sphere · cube · cone · cylinder · torus · capsule) · realistic (basketball · tennis ball · apple · car · scooter · bus) |
| violation | permanence (B = 빈 장면) · shape (B = 다른 물체) · color (B = 색상환 ≥ 60°) |
| motion | flat_static · flat_vel · flat_acc · flat_dec · ramp_down · ramp_up |
| background | v11 (바닥 + 벽) · assets (건물 · 덤불 · 바위 · 하늘) |
| occlusion | visible · early / mid / late × k = 1–3 (패널이 2k 샘플 동안 물체 앞). **late 만 A|B 바꿈을 가린다** |

- index `data_csv/vobench_v1/index.csv` ← `z_research/scripts/data/build_vobench_index.py` (v11 index 14 열 + `sym_k` + 요인 열; `condition` = `<objset>/<violation>/<occlusion>`)
- 문맥 무결성 **1,440 쌍 × 16 장 mismatch 0** (`data_csv/vobench_v1/context_integrity.json`). 데이터 README 는 arm 이 hard link 라 하지만 inode 는 달랐다 — 픽셀 비교로 23,040 장 전부 같음
- 레지스트리: `cvpr/registry/datasets.yaml` · `cvpr/harness/import_registry.py` `MANUAL_DATASETS` · `configs/protocols/datasets.md` (복사 기준선이 옛 레지스트리도 읽는다)

## 2. 프로토콜

`cvpr/eval/testbed` = `surprise_c16t32` 그대로 (CLAUDE.md §1-7): stride 3 → 32 샘플 (raw 0, 3, …, 93), 문맥 16 (raw 0–45) · 예측 16 (raw 48–93), splice 48 이 경계.
ViT-H, float32 + autocast float16, 256 px, target LN, latent L1, `mask_index 0`, **matched pair**. 4 GPU 에서 surprise 228 s.

## 3. 결과 (2026-10-10, vll5 · ViT-H) — objset 별

칸 = 정확도 % · A / B 방향 (A = pos_a·imp_ab, B = pos_b·imp_ba; permanence 의 A 위반 = 물체→빈, B = 빈→물체). CI 는 `vjepa2/surprise/tables.md`.
각 objset 720 쌍 (360 block). 칸 크기: visible 위반당 24 쌍 · early/mid/late 위반당 72 쌍 · late × 운동 × 위반 12 쌍 (방향당 6).

### 3-1. 위반 × 가림 시점

**primitive** (overall 71.7 [69, 75])

| | visible | early | mid | late | 전체 |
|---|---:|---:|---:|---:|---:|
| permanence | 95.8 · 92/100 | 97.2 · 94/100 | 95.8 · 92/100 | **55.6 · 19/92** | 84.2 · 71/98 |
| shape | 75.0 · 75/75 | 63.9 · 58/69 | 68.1 · 61/75 | 54.2 · 50/58 | 63.3 · 58/68 |
| color | 75.0 · 92/58 | 73.6 · 64/83 | 62.5 · 58/67 | 63.9 · 67/61 | 67.5 · 66/69 |
| 전체 | 81.9 · 86/78 | 78.2 · 72/84 | 75.5 · 70/81 | **57.9 · 45/70** | 71.7 · 65/78 |

**realistic** (overall 81.9 [79, 85])

| | visible | early | mid | late | 전체 |
|---|---:|---:|---:|---:|---:|
| permanence | 100.0 · 100/100 | 93.1 · 86/100 | 88.9 · 78/100 | **61.1 · 22/100** | 82.9 · 66/100 |
| shape | 100.0 · 100/100 | 93.1 · 92/94 | 90.3 · 86/94 | 62.5 · 58/67 | 83.8 · 81/87 |
| color | 100.0 · 100/100 | 76.4 · 75/78 | 80.6 · 81/81 | 73.6 · 64/83 | 79.2 · 76/82 |
| 전체 | 100.0 · 100/100 | 87.5 · 84/91 | 86.6 · 81/92 | **65.7 · 48/83** | 81.9 · 74/90 |

### 3-2. late 의 permanence × 운동 (A 사라짐 / B 나타남)

| | static | const. vel | accel | decel | ramp down | ramp up |
|---|---:|---:|---:|---:|---:|---:|
| primitive | 83/50 | **0/100** | **0/100** | **0/100** | **0/100** | 33/100 |
| realistic | 67/100 | **0/100** | **0/100** | **0/100** | **0/100** | 67/100 |

방향당 6 쌍이다. 칸 하나가 아니라 "네 운동 × 두 objset = 8 칸 전부 0/100" 으로 읽는다.

### 3-3. late × k

| | k=1 | k=2 | k=3 |
|---|---:|---:|---:|
| primitive (permanence · shape · color · 전체) | 54 · 54 · 58 · 56 | 54 · 50 · 58 · 54 | 58 · 58 · 75 · 64 |
| realistic | 62 · 62 · 83 · 69 | 62 · 62 · 67 · 64 | 58 · 62 · 71 · 64 |

### 3-4. 복사 기준선 (`vjepa2/surprise/tables.md` 각 objset §7)

copy S = mean_j mean|LN(z)[7] − h[8+j]| — 문맥 encoder 마지막 튜블릿을 미래 8 슬롯 전부에 복사 (`cvpr/eval/testbed/copy_baseline.sh`). 같은 쌍 · 같은 채점.
칸 = 복사 · Δ (모델 − 복사), * = 짝 block 부트스트랩 95 % CI 가 0 을 안 포함.

| | visible | early | mid | late | 전체 |
|---|---:|---:|---:|---:|---:|
| primitive permanence | 100 · −4.2 | 100 · −2.8 | 100 · −4.2 | 61.1 · −5.6 | 88.3 · −4.2* |
| primitive shape | 75.0 · 0.0 | 76.4 · −12.5 | 72.2 · −4.2 | 55.6 · −1.4 | 68.8 · −5.4 |
| primitive color | 79.2 · −4.2 | 75.0 · −1.4 | 59.7 · +2.8 | 52.8 · +11.1 | 64.2 · +3.3 |
| **primitive 전체** | 84.7 · −2.8 | 83.8 · −5.6 | 77.3 · −1.9 | 56.5 · +1.4 | **73.8 · −2.1** |
| realistic permanence | 100 · 0.0 | 100 · −6.9* | 98.6 · −9.7* | 61.1 · 0.0 | 87.9 · −5.0* |
| realistic shape | 100 · 0.0 | 93.1 · 0.0 | 87.5 · +2.8 | 62.5 · 0.0 | 82.9 · +0.8 |
| realistic color | 87.5 · +12.5 | 86.1 · −9.7 | 75.0 · +5.6 | 59.7 · +13.9* | 75.0 · +4.2 |
| **realistic 전체** | 95.8 · +4.2 | 93.1 · −5.6* | 87.0 · −0.5 | 61.1 · +4.6 | **81.9 · 0.0** |

- 복사도 realistic 이 primitive 보다 높다 (81.9 vs 73.8) — objset 차이는 모델 효과가 아니다
- 파이프라인 대조: 복사 실행 안에서 다시 지은 predictor 는 채점기와 S 상대차 최대 1.0e-3, 쌍 판정 22 / 1,440 개 다르다 (autocast fp16 배치 차이, CLAUDE.md §7-2). 결론 같음 (objset 별 같은 파이프라인 Δ 는 `vjepa2/surprise/tables.md`)

### 3-5. 그 밖 (`vjepa2/surprise/tables.md` 각 objset §3–6)

- 배경 (late): primitive v11 63.9 · assets 51.9 / realistic v11 69.4 · assets 62.0 — 두 objset 모두 assets 가 낮다
- 운동 · 물체별은 `vjepa2/surprise/tables.md`

### 3-6. 그림 (`vjepa2/surprise/figures/`, `z_research/scripts/figures/plot_vobench.py --model vjepa2`)

모든 그림 = 행 2 (위 primitive, 아래 realistic) × 열 3 (permanence · shape · colour). 막대 = pairwise 정확도 [block 부트스트랩 95 % CI], 동그라미 = 방향 A, 세모 = 방향 B, 점선 = chance 50.
그린 수치는 `vjepa2/surprise/tables.json` 과 154 칸 대조 (스크립트가 assert).

| 그림 | 무엇 | 캡션 초안 |
|---|---|---|
| `fig_timing` | x = 가림 시점 (회색 = late) | Accuracy by occluder timing. Only *late* hides the A/B switch; *early* and *mid* occlude the object during the context but the switch is visible. Direction A for permanence = object vanishes, B = object appears. |
| `fig_k` | x = k (0 = 가림 없음), 선 = early · mid · late | Accuracy vs. occlusion length k (the panel covers the object for 2k model-input samples). |
| `fig_permanence_context` | permanence 만, 가능한 미래 둘로: 물체 문맥 (있는데→있는) · 빈 문맥 (없는데→없는). 열 = 가림 시점 · late × 운동 | Object permanence split by context. Dark: the context ends with the object, the possible future keeps it and the impossible one removes it. Light: the context is empty, the impossible future makes an object appear. Under late occlusion the model fails every moving-object case with the object in context (0%) and passes every empty-context case (100%). |
| `fig_late_motion` | late 만, x = 운동 6 종 | Late occlusion by motion. For every moving object, the vanish direction (A) is 0% and the appear direction (B) is 100%. |

### 3-7. DINO-Foresight (2026-10-10) — 표 [`dino_foresight/surprise/tables.md`](dino_foresight/surprise/tables.md) · 그림 [`dino_foresight/surprise/figures/`](dino_foresight/surprise/figures/)

`cvpr/registry/models.yaml dinof_highres` (DINOv2 ViT-B/14 + PCA-1152, 정방형 448, target LN 끔). v11 과 같은 설정 — 모델이 문맥 4 프레임만 받아 **문맥 16 장 중 마지막 4 장**으로 미래 16 장을 자기회귀 unroll.
복사 = 같은 채점기에 `model.dinof_copy=true` (마지막 문맥 프레임 특징 × 16). 4 GPU 약 15 분 + 복사 수 분.

| | primitive | realistic |
|---|---:|---:|
| overall — DINO-F · (V-JEPA 2) | **84.9** [82, 87] · (71.7) | **84.7** [82, 87] · (81.9) |
| DINO-F 복사 · Δ | 81.7 · +3.2 [+1.4, +5.0] | 82.9 · +1.8 [+0.6, +3.1] |
| late — DINO-F · (V-JEPA 2) · 복사 | 61.6 · (57.9) · 51.9 | 55.1 · (65.7) · 50.5 |
| late permanence 있는데→있는 / 없는데→없는 | 25 / 89 | 19 / 97 |

- 바꿈이 보이는 칸 (visible · early · mid) 의 shape · color 는 **100 %, 복사도 100 %** — v11 과 같이 프레임 단위 DINOv2 특징이 외형 차이를 바로 보여 주는 것이고 예측의 몫이 아니다 (`auto_research/Archive/DINOF_V11_2026-10-07.md`)
- permanence 는 바꿈이 보여도 V-JEPA 2 보다 낮다 (primitive 83–88 vs 96–97). 있는데→있는 쪽이 67–75 — 16 걸음 unroll 이 물체를 흐린다 (v11 moving_visible vanish 82.8 과 같은 방향, 미검증)
- **late 는 두 모델 모두 무너지고 permanence 비대칭이 같다** — 있는데→있는 primitive 25 · realistic 19 (움직이는 운동 0–33), 없는데→없는 89–97. 이미지 encoder + 프레임 predictor 로 바꿔도 "가려진 물체를 미래에 넣지 않는다" 는 그대로다
- Δ 는 CI 가 0 을 안 포함하지만 +2–3 pt 다. late 에서 DINO-F 는 복사 (51–52) 보다 높다 (primitive late color 69.4 vs 복사 50.0)
- 단서: 문맥 16 중 4 장만 들어간다 · 16 걸음 unroll 은 학습 (1 걸음) 과 멀다 · 입력 해상도가 다르다 (512 원본 → DINO-F 448 · V-JEPA 2 256)

### 3-8. Ariel AR predictor ep40 (2026-10-10) — 표 [`ariel_ar_ep40/surprise/tables.md`](ariel_ar_ep40/surprise/tables.md) · 그림 [`ariel_ar_ep40/surprise/figures/`](ariel_ar_ep40/surprise/figures/)

`vith_ariel_ar_ep40` = 릴리즈 ViT-H encoder (frozen) + Ariel 팔 C 자기회귀 predictor (`z_training/ariel/ar_future_1e_5/latest.pt`, epoch 40 · step 35,840, action 없음 · 자연 영상 SSv2+K400).
채점 = `analysis/predictors/ar_scoring.py` — 문맥 8 블록 → 8 블록 rollout, 타깃 = **블록별** LN(target_encoder(2 프레임)). 복사 = 같은 공간의 `copy_blk` (hb[7] × 8).
⚠️ **타깃 공간이 다른 모델과 다르다 — 쌍 정확도만 비교한다** (surprise 크기 · margin 비교 금지). 4 GPU surprise 204 s.

| | primitive | realistic |
|---|---:|---:|
| overall — AR · (V-JEPA 2 릴리즈) | **81.9** [79, 84] · (71.7) | **86.4** [84, 89] · (81.9) |
| AR 공간 복사 (copy_blk) · Δ | 77.6 · +4.3 [+1.8, +6.8] | 81.4 · +5.0 [+2.5, +7.4] |
| late 전체 — AR · (릴리즈) · 복사 | **75.0** · (57.9) · 55.1 | **78.2** · (65.7) · 55.6 |
| late permanence — AR 있는데→있는 / 없는데→없는 | **100 / 97** | **100 / 100** |

- **late permanence 가 풀린다** — 릴리즈 · DINO-F 가 0 % 이던 움직이는 물체의 "있는데→있는" 이 운동 6 종 × 두 objset 전부 100 %. AR 공간 복사는 54–56 % 라 복사로 풀리는 것이 아니다
- late shape · color 도 릴리즈보다 높다 (59.7–70.8 vs 54.2–73.6, 복사 53–60). 바꿈이 보이는 칸의 shape · color 는 릴리즈와 비슷하거나 낮고 복사보다 낮은 칸이 있다 (primitive visible shape 75.0 vs 복사 91.7)
- ⚠️ **"AR 이 가려진 물체를 미래에 둔다" 로 쓰지 않는다.** 09-30 정체 자 판독 (`auto_research/Archive/AR40_READOUT_2026-09-30.md`) 에서 v11 late 가림 뒤 움직이는 물체 토큰이 AR p 에 0–10 % (릴리즈 40–90 %) 였고 "late 채점 정답은 물체 토큰이 아님" 이 결론이었다. 블록별 타깃에서 무엇이 쌍을 가르는지 (가림막 블록 · 물체 재등장 블록) 는 VOBench 에서 따로 봐야 한다
- 복사 실행 안에서 다시 지은 AR 은 채점기와 S 상대차 최대 1.8e-3 · 쌍 판정 7 / 1,440 개 다름 (fp16)

### 3-9. VideoMAEv2-g (2026-10-10) — 표 [`videomae_v2/surprise/tables.md`](videomae_v2/surprise/tables.md) · 그림 [`videomae_v2/surprise/figures/`](videomae_v2/surprise/figures/)

`videomae2g` (ViT-g/14, 224, MAE decoder 4 층). 창 16 고정이라 **`WINDOW=c8t16_s6`** — stride 6 으로 16 장 (raw 0–90), 문맥 8 장 (raw 0–42) · 미래 8 장 (raw 48–90), splice 48 이 경계 (시간 폭은 c16t32 와 같다).
surprise = 보이는 문맥 토큰으로 미래 토큰을 복원했을 때의 **패치 정규화 픽셀** 오차 (`analysis/model_loaders.py build_videomae2`). latent 가 아니라 쌍 정확도만 비교한다. 복사 경로 없음. 4 GPU surprise 92 s.

| | primitive | realistic |
|---|---:|---:|
| overall — VideoMAEv2 · (V-JEPA 2) | **51.5** [50, 53] · (71.7) | **52.5** [51, 54] · (81.9) |
| visible 전체 · late 전체 | 51.4 · 50.9 | 52.8 · 50.5 |
| permanence 있는데→있는 / 없는데→없는 (visible) | 17 / 83 | 17 / 92 |

- **모든 칸이 chance 근처다 — 가림이 없는 visible 도 50 %.** 전부 맞힌 block 4.7 %. 우리 하네스의 다른 VideoMAEv2 결과와 같다: IntPhys 1 dev 54.44 (`skip5_w16`), v11 52.73 (`z_research/previous/Benchmarks/README.md` 표 · `z_research/previous/Benchmarks/exp_results/surprise_c16t32__v11_videomae2g/summary.json`)
- permanence 는 visible 에서도 빈 문맥 쪽으로 쏠린다 (있는데→있는 17 %) — 픽셀 복원 오차는 물체가 있는 미래가 늘 더 크다
- 미검증 해석: 패치 안 정규화 (평균 · 분산 제거) 가 색 차이를 지우고, 풀밭 · 건물 질감의 복원 오차가 물체 (224 입력에서 반경 약 11 px) 의 차이를 덮는다

### 3-10. 튜블릿 정체 probe — encoder 보임으로 학습한 head 가 p 에 통하나 (2026-10-10, V-JEPA 2 ViT-H 릴리즈)

**묻는 것**: h (target encoder) 의 가림 없는 튜블릿으로 학습한 정체 head 가 p (predictor 출력) 에 그대로 통하면, "predictor 가 물체 정보를 만들기만 하면 encoder 공간에 정렬된다" 는 기대의 근거가 된다.
정의 · 재현: `z_research/scripts/analysis/vobench_tubelet_probe.py` 머리말. 결과 `vjepa2/probing/results/<run>/RESULTS.md` (run = `e35_s0_user` · `e35_s1` · `e10_s0` · `e10_s1` · `e35_s0_realistic_only`).

- 샘플 = (클립, 튜블릿 t), 가능 변이만. 표현 h[t] = LN(target_encoder(32 장))[t] (t = 0–15), p[t] (t = 8–15), probing 관례 bf16. 보임 / 가림 = metadata `samples` 두 샘플 모두 '.' / 'H'
- 라벨 = (모양/색) 결합 + none (빈 장면). head = AttentiveClassifier (depth 1 · query 1 · head 16), lr 1e-4 OneCycle · wd 0.05 · batch 16. block 50/50 (seed 0), **objset 따로**
- 학습 = train block 의 h 보임 튜블릿 (objset 마다 약 4,960). 평가 = test block, 고정 head

| 칸 = 결합 정확도 (none 이라고 답한 %) · 유효 run 범위 | primitive (4 run) | realistic (4 run) |
|---|---:|---:|
| ① h 보임 (천장) | 94–97 (0) | 89–91 (0) |
| ② h 가림 (early · mid · late 의 가림 튜블릿) | 9–15 (24–39) | 13–21 (9–21) |
| ③ p 보임 (visible · early · mid 의 미래, 정답 보임) — 전체 | 37–38 (40–44) | 63–70 (15–23) |
| ③ — 첫 미래 튜블릿 t8 | **83–85 (0)** | **86–88 (0)** |
| ③ — 마지막 튜블릿 t15 | 13–15 (58–66) | 38–56 (29–35) |
| ④ p late (문맥 끝 가림) — 정답 가림 · 재등장 | 0–1 (47–94) | 0–5 (40–96) |

- **가까운 미래에서는 h head 가 p 에 통한다** — t8 에서 83–88 % (천장 89–97). p 에 물체 정보가 있을 때는 encoder 공간과 같은 형식이다
- ~~멀어지며 떨어지는 것은 오정렬이 아니라 정보 소실~~ — **철회 (2026-10-10, p_only 결과).** h head 로는 t8 → t15 오답이 none 으로 늘지만,
  **p 로 학습한 head (`results/p_only_*`, `figures/p_only/`) 는 visible 의 p 를 t15 에서도 52 % (primitive) · 74 % (realistic), none 0 % 로 읽는다** — 같은 head 를 h 에 건 값 (55–69 · 69–77) 과 같은 수준.
  즉 visible 에서 먼 미래의 p 에도 정체 정보는 남아 있고, h head 가 그것을 "없음" 으로 읽은 것은 **p 의 형식이 h 와 달라지는 (멀어질수록 커지는) 정렬 문제** 다.
  이식은 한쪽으로만 잘 된다 (p head → h 는 되고 h head → p 는 먼 미래에서 안 된다) — p 에 남는 것은 h 정보의 일부 (p head 천장 55–77 < h head 89–97)
- **late 는 p 로 학습해도 그대로 0 근처** (움직이는 물체 0–8 %, static realistic 19 %) — 문맥 끝 가림은 정보 자체가 없는 쪽. visible 의 감쇠 (정렬) 와 late 의 0 (부재) 은 다른 실패
- ④ 문맥 끝 가림이면 p 의 정체는 0–5 % (재확인). ② encoder 도 가린 튜블릿에서는 정체를 거의 못 읽는다 (9–21 %)
- ⚠️ **가림막 판이 head 에 물체처럼 읽힌다** — ② 오답의 절반 이상이 "다른 물체", 빈 장면 가림 튜블릿도 36–64 % 를 물체로 답한 run 이 있다. 그래서 ② · ④ 의 none / 다른 물체 비율은 시드에 따라 흔들린다. 안정적인 것은 "정답 정체 ≈ 0" 까지
- ⚠️ ③ t8 의 높은 값에는 "물체가 문맥 끝 바로 옆" 이 섞여 있다. 정체가 읽힌다는 것이지 위치를 맞힌다는 것은 아니다
- ⚠️ **철회 — 첫 실행 (`vjepa2/probing/results/_invalid_run1_realistic_collapsed/`) 의 realistic 은 무효.** 35 epoch · 시드 0 학습이 loss 2.6 에서 멈춰 전부 none 으로 답했다. 같은 설정을 realistic 만 · 순서 그대로 다른 시드 · 10 epoch 로 다시 돌리면 모두 정상 (loss → 0) 이라 그 한 번의 학습 실패로 본다 (같은 설정을 다시 돌린 `e35_s0_user` 는 정상이고 realistic 만 돌린 run 과 수치가 같다 — 같은 설정은 결정적으로 재현되는데 그 한 번만 무너졌다, 원인 미상). 그 run 의 primitive 는 다른 run 과 같은 범위라 범위 계산에 넣었다

### 3-11. asymmetry — late 가림이 문맥 끝에서 끝나도록 창을 민 판 (2026-10-10) — [`vjepa2/surprise_asymmetry/`](vjepa2/surprise_asymmetry/)

**묻는 것**: late 에서 무너지는 것이 "가림 너머를 예측해야 해서" 인가. 창을 k 샘플 밀면 가림 2k 샘플이 전부 문맥 끝에 있고, 가능한 미래는 **첫 샘플부터 물체가 보인다** (바꿈은 문맥 / 미래 경계). 창 16:12, 대조군 = 같은 창 · 시작 0.

| late (창 16:12) | primitive original → asymmetry (Δ) | realistic original → asymmetry (Δ) |
|---|---:|---:|
| 전체 | 56.5 → 56.9 (+0.5 [−4, +5]) | 66.7 → 65.7 (−0.9 [−5, +3]) |
| permanence 있는데→있는 / 없는데→없는 | 19 / 92 → 22 / 89 | 19 / 100 → 22 / 100 |
| permanence, 움직이는 4 운동 | 0 / 100 → **0 / 100** | 0 / 100 → **0 / 100** |

- **바뀌지 않는다.** 가능한 미래에 물체가 처음부터 보여도, 문맥 끝이 가려져 있으면 predictor 는 "사라진" 미래 쪽에 더 가깝다. late 의 실패는 가림 너머를 예측해야 해서가 아니라 **마지막 관측에 물체가 안 보여서** 다 (튜블릿 probe §3-10 의 "late p 정체 0 %" 와 같은 그림)
- k 별 Δ 도 전부 CI 가 0 포함. 단서: 창이 16:12 라 원래 16:16 표 (§3-1) 와는 수치를 섞지 않는다

### 3-12. gravity — ledge · arc (2026-10-11 재렌더판) — [`vjepa2/gravity/`](vjepa2/gravity/)

> 2026-10-11 갱신 — 데이터가 같은 경로로 다시 렌더됐다 (g 981 → 375, arc 꼭대기 f36 · f42 → **f66 · f48 · f30 = 표준 창에서 문맥이 오르는 중 · 꼭대기 · 내려가는 중** 으로 끝난다).
> 옛 판 (g 981 + sliding 6 창) 은 아래 "옛 판" 절 · 원시 `exp_results/_previous_20261010_gravity_g981/` · 정리본 `vjepa2/gravity/_previous_20261010_g981/`.

**데이터** `/data2/local_datasets/world/Benchmark/VOBench_v1/gravity` (그 폴더 `README.md` · `review/INDEX.md`). 216 block / 936 arm, g 375 cm/s².
**block = 과거 하나 + 미래 여럿** — 모든 arm 이 f0–47 을 공유한다 (hard link, 감사 720 arm × 48 장 mismatch 0, `data_csv/vobench_v1_gravity/context_integrity.json`).
- ledge 144 block: `pos` (선반 끝에서 낙하) · `imp_float` (같은 높이로 직진). 셀 = 높이 z300 · z400 × 속도 200 · 260 · 320 cm/s
- arc 72 block: 미래 9 — `pos` (포물선) · `rise` (거울 × 0.45, 계속 올라감) · `float` · `line` · `stop` (f45 자리에서 멈춤) · `stop2/4/8/12` (진짜 운동을 f50/52/56/60 까지 따르다 공중에서 멈춤)
- arc 셀 = 꼭대기 f66 (`apex66`, 문맥 끝 − 꼭대기 −21 = **오르는 중**) · f48 (`apex48`, −3 = **꼭대기**) · f30 (`apex30`, +15 = **내려가는 중**). 어느 arm 도 화면 안에서 땅에 닿지 않는다
- objset × 셀 칸마다 12 block

**채점** = 표준 `surprise_c16t32` (c16 / p16, stride 3, 문맥 끝 f45) + `scoring.pairing=cross` (block 안 가능 1 × 불가능 전부 = 문맥 일치 쌍 전부, 레지스트리 `vobench_v1_gravity` overrides).
arc 는 **고른 미래 = argmin_k s_k** ("p 가 가장 가까운 미래", chance 11 %). 복사 = `token_surprise_maps.py --copy` (LN(z) 마지막 문맥 튜블릿 × 미래 8). 4 GPU 채점 약 4 분 + 토큰 지도 약 6 분.

**ledge**

| | n | 모델 | 복사 | (s_float − s_pos)/s_pos 중앙 |
|---|---:|---:|---:|---:|
| primitive (6 셀) | 72 | **0.0** (6 셀 모두 0) | 0.0 | −1.4 % (셀별 −0.8 ~ −2.0) |
| realistic (6 셀) | 72 | **0.0** (6 셀 모두 0) | 0.0 | −2.4 % (셀별 −1.7 ~ −3.7) |

- 144 쌍 전부 낙하보다 같은 높이 직진이 덜 놀랍다. 높을수록 (z400) · 느릴수록 (v200) 더 크게 틀린다
- ⚠️ **복사도 0 % 다 (Δ = 0)** — 떠 가는 미래는 마지막 문맥과 닮았다. 이 쌍으로는 "낙하하는 미래를 만들지 않는다" 와 "덜 변하는 미래가 L1 로 싸다" 를 못 가른다 (H23 §3-7, 정본 §2-3 iii · v11_realistic ledge 0/120 과 같은 단서)

**arc — 고른 미래 %** (그림 `vjepa2/gravity/figures/fig_gravity_choice_{primitive,realistic}` · 공 칸 `…_ball_*`, `tables.md` §2)

| | arc (진짜) | rise | float · line | stop · stop2 · stop4 | **stop8** | **stop12** |
|---|---:|---:|---:|---:|---:|---:|
| primitive — 모델 (전체 토큰) | **0** | 0 | 0 | 0 | 28 | **72** |
| primitive — 모델 (공 칸만) | 0 | 0 | 0 | 0 | 25 | 75 |
| primitive — 복사 | 0 | 17 | 36 | 39 | 3 | 6 |
| realistic — 모델 (전체 토큰) | **0** | 3 | 0 | 0 | 25 | **72** |
| realistic — 모델 (공 칸만) | 0 | 0 | 0 | 0 | 28 | 72 |
| realistic — 복사 | 0 | 25 | 67 | 6 | 0 | 3 |

셀 (오르는 중 · 꼭대기 · 내려가는 중) 별로도 같다 — 모델은 셀 6 칸 모두 stop8 + stop12 가 92–100 %, 진짜 포물선은 **72 block 중 0**.

**튜블릿 묶음** (`tables.md` §2 "튜블릿 묶음") — 앞 = t0–t2 (f48–63, arc 와 stop12 는 f60 까지 픽셀이 같다) · 뒤 = t3–t7 (f66–93)

| p 가 가장 가까운 후보 % (primitive / realistic) | 앞: arc | 앞: rise | 앞: float · line | 앞: 멈춤 | 뒤: arc | 뒤: 멈춤 |
|---|---:|---:|---:|---:|---:|---:|
| 오르는 중 (`apex66`) | 28 / 42 | 8 / 3 | 8 / 8 | 56 / 47 | 2 / 3 | 75 / 83 |
| **꼭대기 (`apex48`)** | **3 / 3** | **31 / 69** | 28 / 19 | 39 / 8 | 3 / 0 | 68 / 73 |
| 내려가는 중 (`apex30`) | 36 / 53 | 0 / 0 | 22 / 11 | 42 / 36 | 3 / 0 | 77 / 85 |

- **p 는 진짜 포물선을 끝까지 잇지 않는다.** 미래 9 개 중 가장 가까운 것은 "진짜 운동을 문맥 끝 뒤 11–15 프레임 (stop8 · stop12) 따르다 공중에서 멈춘" 미래이고, 문맥이 오르는 중 · 꼭대기 · 내려가는 중 어디서 끝나든 같다. 뒤 튜블릿 (f66–93) 은 셀 모두 멈춤 후보가 68–85 %, 진짜 포물선 0–3 %
- **꼭대기 문맥에서는 앞 튜블릿부터 진짜와 갈린다** — 꼭대기에서 되돌아 내려오는 진짜 대신 계속 올라가는 (rise 31 / 69 %) · 수평 (float · line 28 / 19 %) 미래 쪽이다 (앞: arc 3 %). 오르는 중 · 내려가는 중은 앞 튜블릿에서 진짜 운동 (arc · 늦은 멈춤) 쪽이다
- 위치 규칙과 다르다: 문맥 끝에서 등속으로 이으면 오르는 중 → arc · 꼭대기 → rise · 내려가는 중 → float (block 12 / 12), 높이 유지면 rise · float · float, 그 자리 멈춤이면 stop. p 는 그중 어느 것도 아니다 — **오르는 중에서는 등속만 이어도 진짜 포물선에 가장 가까운데 p 는 0 / 24**
- 복사는 rise · float · stop 을 고른다 (stop8 · stop12 는 3–6 %) — 마지막 문맥 복사로는 이 고르기가 안 나온다
- 쌍 정확도 (VoE) 로 읽으면 vs float 17 % · vs rise 28–39 % · vs stop8 · stop12 0 % — 쌍 하나씩 보면 "틀림" 이고, 고른 미래가 그 아래 그림이다

**쓸 수 있는 문장 / 못 쓰는 문장**
- ✅ "꼭대기에서 되돌아옴 (3a) · 낙하 (3b) 를 따르는 미래를 만들지 않는다 — p 에 가장 가까운 미래는 진짜 운동을 몇 샘플 이은 뒤 공중에서 멈춘 미래다 (문맥이 오르는 중 · 꼭대기 · 내려가는 중 모두)"
- ❌ "중력을 모른다" (내부 귀속) · "ledge 에서 불가능한 연속을 예측한다" (복사도 0 %) · "p 는 15 프레임 뒤 멈춘다" (아래 단서 — 15 는 후보의 상한)

⚠️ 단서
- **D 15 (stop12) 는 후보 중 가장 늦은 멈춤이다** — 72 % 가 그것을 고르므로 p 가 "어디서 멈추나" 는 f60 이후 어딘가라는 것까지만 말한다 (옛 판 sliding 은 짧은 후보가 있으면 5–11 프레임을 골랐다). 멈춤 지점을 재려면 stop16 · stop20 · stop24 … 같은 늦은 멈춤 후보가 필요하다
- "가장 가까운 미래 = 그린 미래" 는 후보 9 개 안의 표면 측도다. 공 칸만 세도 같지만 위치 자로 확인한 것은 아니다
- 멈춤 후보는 화면 변화가 작은 미래다. 다만 가장 덜 변하는 후보 (stop, f45 자리) 는 0 / 72 이고 복사가 고르는 것과 다르다 — "덜 변하는 미래가 늘 싸다" 만으로는 설명되지 않는다
- 칸마다 12 block · 튜블릿 하나의 argmin 은 흔들린다 (묶음으로 읽는다)

**옛 판 (2026-10-10, g 981, 꼭대기 f36 · f42 — 대체됨)**: 두 셀 모두 표준 창 문맥이 내려가는 중이어서 sliding (stride 2, 문맥 끝 f30–45) 으로 오르는 중 · 꼭대기를 만들었다.
결과는 같은 방향 — ledge 0 / 48 (복사 0), arc 표준 창 stop8 · stop12 71–76 % · 진짜 포물선 8–17 %, sliding 288 block-창 전부 멈춤 후보 (짧은 후보가 있으면 D 5–11 프레임).
표 · 그림 `vjepa2/gravity/_previous_20261010_g981/`, 원시 `exp_results/_previous_20261010_gravity_g981/`. 두 판을 섞어 읽지 말 것 (g · 꼭대기 · 셀이 같이 바뀌었다).

### 3-13. gravity_horizon — 예측 길이를 줄여도 멈추나 (2026-10-11) — [`vjepa2/gravity_horizon/`](vjepa2/gravity_horizon/)

**묻는 것**: §3-12 의 "진짜 운동을 몇 샘플 잇다 공중에서 멈춘다" 가 긴 예측 (16 샘플) 때문인가. 문맥 16 샘플 (문맥 끝 f45) 고정, 예측 M = 16 · 8 · 4 · 2 샘플.
읽기 (파라미터 0): 미래 튜블릿마다 p 가 **정지 (f45 자리)** 보다 **진짜 운동 (arc)** 에 가까운 block %. **기준 = 전체 토큰 (표준 채점과 같은 토큰)** (사용자 10-11), 공 칸은 보조. x = 예측 순번 (문맥 끝에서 몇 번째 샘플).

**전체 토큰 — 예측 길이와 무관하게 창의 끝에서 진짜 운동을 떠난다** (`vjepa2/gravity_horizon/figures/fig_horizon_all_{primitive,realistic}`)

| 진짜 운동에 가까운 % (primitive / realistic, 세 문맥 범위) | 3–4 번째 | 7–8 번째 | 마지막 튜블릿 |
|---|---|---|---|
| 예측 16 | 100 / 100 | 0–67 / 0–75 (한 번 떨어졌다 다시 오름) | **0 / 0** (15–16 번째) |
| 예측 8 | 100 / 100 | 0–17 / 0–25 | **0–17 / 0–25** (7–8 번째) |
| 예측 4 | 8–42 / 17–75 | — | **8–42 / 17–75** (3–4 번째) |
| 예측 2 | — | — | 42–100 / 83–100 (1–2 번째) |

- **같은 3–4 번째 샘플이 예측 16 · 8 에서는 100 % 인데 예측 4 에서 창의 끝이 되면 떨어진다** — 망가지는 곳은 절대 순번이 아니라 **창의 끝 (마지막 1–2 튜블릿)** 이다. "긴 예측이라서" 가 아니라 "예측 창의 끝이라서"
- 예측 2 는 튜블릿 하나라 효과가 가장 약하다 (내려가는 중 42 / 83, 오르는 중 100 / 100)
- 고른 미래 (`tables.md` §1, 전체 토큰) 도 같다 — 예측 8 에서 창 끝까지 움직이는 미래는 0 / 72, 예측 4 · 2 에서도 내려가는 중은 float (높이 유지) 를 6–9 / 12 고른다

**공 칸 (보조)** — 공 칸만 보면 창 끝 효과가 없다: 예측 4 · 2 는 끝까지 100 %, 예측 8 의 끝 (7–8 번째) 은 예측 16 의 같은 순번과 같다 (오르는 중 75 / 100 vs 67 / 92 · 꼭대기 25 / 25 vs 17 / 17 · 내려가는 중 0 / 8 vs 0 / 0, `fig_horizon_ball_*`).
즉 창 끝 효과는 공이 아닌 토큰 (배경 · 그림자 — 두 미래의 픽셀이 같은 칸) 에서 오고, 공 자체는 문맥 끝 뒤 6–8 번째 (오르는 중 10–12 번째) 에서 진짜 운동을 떠난다. 기제 (target encoder 전역 attention 으로 퍼진 h 차이) 는 미검증

**쓸 수 있는 문장**: ✅ "예측 길이를 16 → 8 → 4 로 줄여도 predictor 의 미래는 창의 끝에서 진짜 운동을 떠나 정지 쪽으로 간다 — 긴 예측이 만든 현상이 아니다 (전체 토큰)"
❌ "예측이 짧으면 물리를 따른다" (짧은 창도 끝에서 떨어진다 · 꼭대기는 짧은 창에서도 되돌아오지 않는다)

⚠️ 단서: 칸마다 12 block · 정지 대조군은 f45 자리 하나 (진짜 운동과 정지 사이의 "느려짐" 은 이 이분법으로 못 본다) · 창이 바뀌면 target encoder 입력 길이도 바뀐다 (16 + M 프레임 — 대조는 같은 창 안의 두 h 끼리라 비교는 성립한다) · 창 끝 효과가 gravity 밖 (VOBench 본판 · v11) 에도 있는지는 미확인

### 3-14. gravity — Ariel AR predictor ep40 (2026-10-11) — [`ariel_ar_ep40/gravity/`](ariel_ar_ep40/gravity/)

같은 데이터 · 표준 창 (§3-12). `vith_ariel_ar_ep40` = 릴리즈 ViT-H encoder (frozen) + Ariel 팔 C 자기회귀 predictor (action 없음 · 자연 영상 SSv2 + K400). 채점 = `ar_scoring.py` (문맥 8 블록 → 8 블록 rollout, 타깃 = 블록별 LN(target_encoder)).
**주 결과 = L1 retrieval (전체 토큰)**. ⚠️ 타깃 공간이 릴리즈와 달라 surprise 크기는 비교하지 않는다 — 고른 미래만 나란히 놓는다. 복사 = 같은 공간의 `copy_blk`.

**arc — 고른 미래 %** (그림 `ariel_ar_ep40/gravity/figures/fig_gravity_choice_{primitive,realistic}`; primitive / realistic)

| 문맥 | 릴리즈 (§3-12) | **Ariel AR** | AR 공간 복사 (copy_blk) | 위치 규칙: 등속 · 높이 유지 |
|---|---|---|---|---|
| 오르는 중 | 멈춤 (stop +8f · +12f) 100 / 100 | **arc 83 / 75** · float 17 / 0 · line 0 / 17 | arc 42 / 17 · rise 17 / 42 · float 33 / 17 | arc · rise |
| 꼭대기 | 멈춤 100 / 92 | **float 50 / 58 · rise 33 / 33** · arc 17 / 8 | rise 42 / 33 · float 42 / 67 | rise · float |
| 내려가는 중 | 멈춤 100 / 100 | **rise 58 / 58 · float 42 / 42** · arc **0 / 0** | rise 92 / 75 · float 8 / 25 | float · float |

- **AR 은 멈추지 않는다** — 멈춤 후보 (stop · stop +2f … +12f) 0 / 72. 릴리즈의 "몇 샘플 가다 공중에서 멈춤" 은 AR 에 없다
- 그러나 **고르는 미래는 등속 · 높이 유지 규칙의 후보와 같다** — 오르는 중 arc (등속으로 이어도 arc 가 가장 가깝다, block 24 / 24), 꼭대기 rise · float, 내려가는 중 float · rise. **진짜 포물선은 문맥이 내려가는 중이면 0 / 24, 꼭대기면 3 / 24** — 꼭대기에서 되돌아옴 · 떨어짐을 따르는 미래를 만들지 않는다
- 오르는 중의 arc 83 / 75 는 중력의 증거가 아니다 — 그 창 (f48–93) 에서 진짜 포물선은 문맥 끝 운동을 등속으로 이은 궤적과 가장 가깝다 (위치 규칙 12 / 12)
- 복사 (copy_blk) 도 꼭대기 · 내려가는 중에서 rise · float 를 고른다 — AR 의 꼭대기 · 내려가는 중 고르기는 마지막 문맥과 닮은 미래 쪽이다 (오르는 중만 복사 42 / 17 보다 arc 를 더 고른다)
- 공 칸 (보조): 오르는 중 arc 58 / 58 (+ stop +8f · +12f 42 / 42), 꼭대기 float 67 / 58, 내려가는 중 float 58 / 75 · rise 42 / 17 — 전체 토큰과 같은 방향

**ledge** — AR 9.7 [3, 17] / 0.0 (72 쌍씩) · copy_blk 12.5 / 1.4 (Δ −2.8 / −1.4, CI 0 포함). 릴리즈 (0 / 0) 와 같이 낙하보다 같은 높이 직진을 고른다

**쓸 수 있는 문장**: ✅ "관측만으로 학습한 AR predictor 는 멈추지는 않지만, 그리는 미래는 문맥 끝 운동을 등속 · 높이 유지로 이은 것이다 — 꼭대기에서 되돌아옴 · 낙하 (3a · 3b) 를 따르는 미래는 만들지 않는다 (내려가는 중 진짜 포물선 0 / 24)"
❌ "AR 은 오르는 중 포물선을 예측한다" (등속 규칙과 같은 후보) · "AR 이 릴리즈보다 물리를 더 안다" (차이는 멈춤 유무)

⚠️ 단서: 칸마다 12 block · AR 학습 분포는 문맥 2–16 블록 · rollout ≤ 8 블록 (이 창은 안) · 공간이 달라 복사 Δ 크기는 릴리즈와 비교하지 않는다

### 3-15. Ariel AR-state predictor ep40 (2026-10-11) — [`ariel_ar_state_ep40/surprise/`](ariel_ar_state_ep40/surprise/README.md)

`ar` 에서 rollout 되먹임만 predictor 상태 (384) 로 바꾼 팔. overall primitive **81.5** · realistic **85.4** (ar 81.9 · 86.4, AR 공간 복사 77.6 · 81.4).
**ar 와 구분되지 않는다** — 쌍 단위 판정 차이 primitive 11 / 8 (p 0.65) · realistic 10 / 3 (p 0.09), 위반 × 가림 칸 전부 ±4pt 안. late permanence 거의 다 맞힘 · late shape · colour chance 근처도 같다.
⚠️ 학습 데이터 · 예산도 ar 와 달라 되먹임 하나의 효과로 읽지 않는다.

## 4. 단서

- **visible · early · mid 는 leakage 가 크다** (복사 primitive 77–85 · realistic 87–96 %). 그 칸의 높은 점수는 예측의 증거가 아니다 — 쓸 수 있는 칸은 copy-hard 인 late 다
- late 도 복사가 primitive 56.5 · realistic 61.1 로 chance 보다 약간 높다. late 안에서 복사가 무엇을 잡는지 (가림막 위치 · 패널 회전) 는 미분석
- **표본이 작다** (§3 머리). objset 별로 나누면 late × 운동 칸은 방향당 6 쌍 — 일관성으로만 읽는다
- early · mid 는 **바꿈이 보이는** 조건이다 (패널이 splice 전에 지나간다). "가림이 있어도 된다" 로 쓰지 않는다
- 이 결과는 ViT-H 하나다

## 5. 다음

1. late 만 따로 — 복사가 못 푸는 칸에서 모델 · 다른 predictor (Ariel · post-FT) 비교 (objset 별)
2. early 에서 모델 < 복사인 칸의 슬롯별 L1 (`l1.npy` · `copy.npy` 는 (n, 8) 슬롯별이다)

## 재현

```bash
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
$P z_research/scripts/data/build_vobench_index.py --frames-root /data2/local_datasets/world/Benchmark/VOBench_v1 --out data_csv/vobench_v1
GPUS=4 RESULTS_ROOT=legacy bash cvpr/eval/testbed/run.sh vobench_v1 vith      # → exp_results/surprise_c16t32__vobench_v1_vith/  (4 GPU 228 s)
GPUS=4 OUTDIR=z_research/VOBench_v1/exp_results/copy__vobench_v1_vith \
  bash cvpr/eval/testbed/copy_baseline.sh vobench_v1 vith                      # → exp_results/copy__vobench_v1_vith/ (l1 · copy · hcopy .npy, copy_summary.json)
$P z_research/scripts/analysis/vobench_tables.py --model vjepa2                # → vjepa2/surprise/tables.{md,json} (objset 별)
$P z_research/scripts/figures/plot_vobench.py --model vjepa2                   # → vjepa2/surprise/figures/fig_{timing,k,late_motion,permanence_context}.{pdf,png}
# DINO-Foresight
GPUS=4 RESULTS_ROOT=legacy bash cvpr/eval/testbed/run.sh vobench_v1 dinof_highres           # → exp_results/surprise_c16t32__vobench_v1_dinof_highres/
COPY=1 GPUS=4 RESULTS_ROOT=legacy bash cvpr/eval/testbed/run.sh vobench_v1 dinof_highres    # → …_dinof_highres_copy/
$P z_research/scripts/analysis/vobench_tables.py --model dino_foresight && $P z_research/scripts/figures/plot_vobench.py --model dino_foresight
# Ariel AR ep40
GPUS=4 RESULTS_ROOT=legacy bash cvpr/eval/testbed/run.sh vobench_v1 vith_ariel_ar_ep40
GPUS=4 OUTDIR=z_research/VOBench_v1/exp_results/copy__vobench_v1_vith_ariel_ar_ep40 bash cvpr/eval/testbed/copy_baseline.sh vobench_v1 vith_ariel_ar_ep40
$P z_research/scripts/analysis/vobench_tables.py --model ariel_ar_ep40 && $P z_research/scripts/figures/plot_vobench.py --model ariel_ar_ep40
# VideoMAEv2-g
GPUS=4 WINDOW=c8t16_s6 RESULTS_ROOT=legacy bash cvpr/eval/testbed/run.sh vobench_v1 videomae2g
$P z_research/scripts/analysis/vobench_tables.py --model videomae_v2 && $P z_research/scripts/figures/plot_vobench.py --model videomae_v2
# 튜블릿 정체 probe (§3-10)
$P z_research/scripts/analysis/vobench_tubelet_probe.py extract --gpus 4
$P z_research/scripts/analysis/vobench_tubelet_probe.py probe --gpu 0 --epochs 10 --seed 0 --tag e10_s0     # e10_s1 · e35_s1 · (--objsets realistic --epochs 35 --tag e35_s0_realistic_only)
# gravity (§3-12) — 레지스트리 vobench_v1_gravity (cross 채점)
$P z_research/scripts/data/build_vobench_gravity_index.py --frames-root /data2/local_datasets/world/Benchmark/VOBench_v1/gravity --out data_csv/vobench_v1_gravity
GPUS=4 RESULTS_ROOT=legacy bash cvpr/eval/testbed/run.sh vobench_v1_gravity vith                                # → exp_results/surprise_c16t32__vobench_v1_gravity_vith/ (4 GPU 약 4 분)
$P z_research/scripts/analysis/token_surprise_maps.py --run z_research/VOBench_v1/exp_results/surprise_c16t32__vobench_v1_gravity_vith --gpus 4 --copy   # → token_surprise.npz (e · ec)
# gravity — Ariel AR ep40 (§3-14)
GPUS=4 RESULTS_ROOT=legacy bash cvpr/eval/testbed/run.sh vobench_v1_gravity vith_ariel_ar_ep40
$P z_research/scripts/analysis/token_surprise_maps.py --run z_research/VOBench_v1/exp_results/surprise_c16t32__vobench_v1_gravity_vith_ariel_ar_ep40 --gpus 4 --copy
$P z_research/scripts/analysis/vobench_gravity_tables.py --model ariel_ar_ep40 && $P z_research/scripts/figures/plot_vobench_gravity_choice.py --model ariel_ar_ep40
# gravity_horizon (§3-13) — 예측 M 8 · 4 · 2 (M 16 = 위 표준 창)
for n in 24 20 18; do SET="window.n_frames=$n window.context=16" GPUS=4 RESULTS_ROOT=legacy bash cvpr/eval/testbed/run.sh vobench_v1_gravity vith
  $P z_research/scripts/analysis/token_surprise_maps.py --run z_research/VOBench_v1/exp_results/surprise_c16t32__vobench_v1_gravity_vith_wc16t$n --gpus 4 --copy; done
$P z_research/scripts/analysis/vobench_gravity_horizon.py && $P z_research/scripts/figures/plot_vobench_gravity_horizon.py   # → vjepa2/gravity_horizon/
# (옛 판 sliding: SET="window.stride=2 window.frames_start=<0..15>" 로 같은 두 줄 — 결과 폴더에 _wc16t32s2o<시작> 이 붙는다)
$P z_research/scripts/analysis/vobench_gravity_tables.py && $P z_research/scripts/figures/plot_vobench_gravity_choice.py && $P z_research/scripts/figures/plot_vobench_gravity.py   # → vjepa2/gravity/
```
