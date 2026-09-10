# IntPhysGen V11 — 시작점

> **새 세션은 이 파일부터 읽는다.** 레포 전체 규칙은 루트 `CLAUDE.md`,
> 실행 방법은 `configs/protocols/README.md`, 스크립트 목록은 `z_research/scripts/README.md`.

> ⚠️ **2026-09-06 — 스토리 축이 옮겨갔다. 정본은
> [`Archive/PAPER_STORY_2026-09-06.md`](Archive/PAPER_STORY_2026-09-06.md) 다.**
>
> **한 문장: predictor 는 상태를 이어가지 않는다. 문맥을 조회할 뿐이다.**
> 실세계 관측은 늘 불완전하고(가림·화면 밖·미발생), 끊긴 구간의 상태를 이어가는 일은
> encoder 가 못 한다 → predictor 가 해야 하는데, EK100·IntPhys 가 pretrained predictor 를
> 그대로 쓰면서 **그게 실제로 이어가는지는 검증된 적이 없다.** 통제 testbed 로 검증하니
> 이어가지 않는다 — 미래 토큰끼리 **서로를 참조하지 않고**(`mask→mask` Δ −1.4),
> 문맥 조회가 전부다(`mask→ctx` Δ −37.5). **이 하나의 성질이 87.2 / 79.0 / 59.5 / k 무관 /
> `p` self 98.5 를 전부 설명한다.** → frozen encoder + 시간 인과·다단계 예측으로 학습한
> predictor → anticipation 과 intuitive physics 로 검증.
>
> ⚠️ **"frozen 위에서 닫힌다" 원칙을 폐기**한다 (그 문서 §하네스 5번).
> ⚠️ 미측정 항목은 **하나뿐이다** — `p`×`h` 시간 정렬 행렬 (캐시만, 그 문서 §5-1).
>
> 배경: [`Archive/STATUS_AND_CRITIQUE_2026-09-03.md`](Archive/STATUS_AND_CRITIQUE_2026-09-03.md)
> (corner case 판정 · 후보 A/B/C · 선행연구). [`PAPER_STORY_2026-08-31.md`](Archive/PAPER_STORY_2026-08-31.md)
> 는 **근거 수치와 금지 표현 표가 그대로 유효**하고 새 판 §3 진단에 재사용된다.

**v11 이 본 실험 세트다.** v8·v10 은 아카이브이고 신규 실험은 전부 v11 에서 돈다.

곁가지 3: [`../RollOutV2/`](../RollOutV2/README.md) — **궤적 다양성 × 문맥 끝 가림 k** (설계 중, 2026-09-09). 이어가기 vs 외삽을 가르는 세트.

곁가지 2: [`../v11_roll_out/`](../v11_roll_out/README.md) — **predictor 미래 8슬롯에서 위치·속도·가속 readout**
(2026-09-07). late 가림에서도 조건별 readout 으로 위치·속도·가속이 읽힌다. readout 이식(자리)은 `h` 도 무너져 이 도구로는 못 묻는다.

곁가지: [`../IntPhysGenV11_occlusion_timing_ablation/`](../IntPhysGenV11_occlusion_timing_ablation/README.md)
— **`k` 장을 문맥 어디에 두는가.** 문맥 안 위치도 양도 거의 무관하고,
**가림이 예측 경계에 닿을 때만 무너진다** (총 장수를 맞춰도 +20.2pt).
§5-1 의 "`k` 평평" 은 모든 k 가 경계에 닿아 이미 바닥이었기 때문이다.

---

## 1. 지금 논문이 무엇인가

**정본: [`Archive/PAPER_STORY_2026-08-31.md`](Archive/PAPER_STORY_2026-08-31.md)**

> Latent predictor 는 관측이 끊겨도 정보를 잃지 않는다.
> 잃는 것은 **그 정보를 놓는 자리**이고, 그 자리는 **frozen 상태에서 되돌릴 수 있다.**

8 beat: motivation(IntPhys 1 은 되는데 2 는 무너진다) → testbed 두 통제 → 병목은 predictor
단계 → 무너지는 건 정보가 아니라 배치 → **배치가 어떻게 다른가(표현 수준, 미실행)** →
frozen 개입 → 회복 → motivation 으로의 귀환.

**실험을 제안하기 전에 그 문서의 "하네스" 절 6개 질문 + 7번(corner case / use case, `STATUS_AND_CRITIQUE` §8)과 금지 표현 표를 본다.**
어느 beat 를 세우는지 못 대면 그렇게 말하고 돌리지 않는다.

**논문 전체가 frozen model 위에서 닫힌다.** 재학습이 필요한 실험은 그 자체로 재검토 대상이다.

---

## 2. 문서 지도

| 파일 | 무엇 | 언제 |
|---|---|---|
| **`Archive/STATUS_AND_CRITIQUE_2026-09-03.md`** | **현행 상황 + corner case 판정 + 스토리 후보 + 확장 축** | 지금 무엇을 할지 정할 때 |
| **`Archive/PAPER_STORY_2026-08-31.md`** | **논문 스토리라인 (정본, 재검토 중)** | 실험 제안·평가 전 항상 |
| `Archive/SYNTHESIS_2026-08-30.md` | 사슬 정리 · 주장 12개 상태표 · 금지 표현 | 결과를 문장으로 옮길 때 |
| `Archive/surprising_score/RESULTS_2026-08-30.md` | **전수 기록 (표 A–J).** 채점 45셀 + probing 324칸 | 수치를 찾을 때 |
| `Archive/surprising_score/READING_THE_MATRIX_2026-08-31.md` | **투표 행렬·precision/recall 읽는 법 (예제)** | 처음 보는 사람 |
| `Archive/surprising_score/VOTE_GRID_2026-08-31.md` | 그 결론과 전수 표 | beat 4 |
| `Archive/surprising_score/HYPOTHESIS_SCALAR_ORDER_2026-08-30.md` | 1차원 순서 가설과 검정 | beat 4 |
| `Archive/TALK_2026-08-30.md` | 발표 진행안 | 발표 |
| `figures/surprise/README.md` · `figures/probing/README.md` | 그림 카테고리와 읽는 법 | 그림 고를 때 |
| `Archive/PAPER_STORY_2026-08-29.md` | **대체됨** (12 beat 서술형). 근거 수치는 유효 | 기록 |

⚠️ **α 분해는 인용 금지다** (지시 전까지). 옛 문서 본문에는 남아 있다.

---

## 3. 데이터셋

`21,504 clip / 5,376 block / 10,752 matched pair` = `시나리오 6 × 위반 3 × k 4 × 84 block`

| 축 | 값 |
|---|---|
| `condition` | `static / moving_flat / moving` × `visible / occlusion` — **6개** |
| `sym_k` | 0(가림 없음) / 1 / 2 / 3 / 4 — 가려지는 **샘플 프레임 수**(한쪽당) |
| `violation_type` | vanish / shape / color |
| shape | 7종, **21쌍 전부** |
| color | 8종, **28쌍 전부** |

- **`k` 를 데이터셋 내부 축으로** 가져간 것이 v10 과의 가장 큰 차이다
- **`condition` 에 k 를 접지 않았다** — 접었으면 `fit_groups_sweep: auto` 가 24그룹 × 3 × 3 = 216 head 를 만든다. 지금은 7 × 3 × 3 = 63
- **silhouette 을 7종 전부 동일하게** 맞췄다 (v10 은 모양이 폭과 교락됐다)
- 속도 고정(116 cm/s), **가림막 폭이 k 와 함께 변한다** — 조작의 구현이지 교란이 아니다
- 프레임 34장 저장, 프로토콜은 32장(`0,3,…,93`)만 읽는다

레지스트리 항목: `configs/protocols/datasets.md` 의 `## v11`.
설계 문서: `/data/hyuntak/project/2026/2027_cvpr/UnrealEngine/gen/V11_DESIGN.md`

---

## 4. 확립된 수치 (`exp_results/` 에서 재검증됨)

### 채점 — `surprise_c16t32`, matched pair, chance 50%

**overall 73.37%** (n=10752, 45 cells, `report.json.verified = true`)

| condition | vanish | shape | color |
|---|---:|---:|---:|
| `static_visible` | 100.0 | 94.0 | 81.8 |
| `static_occlusion` | 70.5 | 54.8 | 55.4 |
| `moving_visible_flat` | 100.0 | 89.9 | 86.0 |
| `moving_occlusion_flat` | 52.0 | 56.8 | 68.5 |
| `moving_visible` (ramp) | 100.0 | **69.0** | 76.8 |
| `moving_occlusion` (ramp) | 50.0 | 54.5 | 71.4 |

읽는 법 세 가지:

1. **가림이 전부를 무너뜨린다** — `visible` 행은 전부 높고 `occlusion` 행은 전부 바닥
2. **등가속은 가림 없이도 이미 낮다** — shape 94.0 / 89.9 → **69.0**. 물리를 안다면 반대여야 한다
3. **vanish 50% 는 반쯤 맞힌 게 아니다** — `빈→물체` 100.0 / `물체→빈` 0.0 의 평균

### probing — `attn_probe`, 54 항목 (fit 3 × groups 6 × target 3)

세 지점 `z`(`contextF__f1to16`) · `p`(`pred__f17to32`) · `h`(`targetF__f17to32`),
6 조건 각각으로 head 를 따로 학습(`fit_groups_sweep: auto`), target 은 shape / color / env.

핵심 (등가속 + 가림):

| | 값 |
|---|---:|
| `p` 에서 shape | **98.46%** |
| `p` 에서 color | **99.49%** |
| `p` 에서 env | **100%** |
| 같은 조건 채점 민감도 (shape) | **4.46** |

- `env` 는 self · 이식 · k 분해 **72칸 전부 100.0**
- 비가림 → 가림 이식: color `z` 97.4 / `h` 99.2 / **`p` 17.3**
- ❌ **shape 은 인코더도 흔들린다** (`z` 3팔 평균 64.8, 등속 팔 42.4) — "인코더는 조건과 무관" 은 color·env 로만

전수는 `Archive/surprising_score/RESULTS_2026-08-30.md`.

### beat 4 의 순서 (등가속 + 가림, shape)

```
cylinder 90 > cube 86 > pyramid 59 > capsule 58 > cone 48 > sphere 36 > torus 3
```
42방향 중 **40개** 일치, Hodge R² **0.82–0.91** (무작위 기준선 0.286).
비가림에서도 등가속은 이미 R² **0.816** (정지·등속은 잡음바닥 아래).
**precision 이 이것이 편향임을 확정한다** (등가속+가림): recall 폭 **86** 인데 precision 폭 **24**.
`cylinder` R 89.6 / **P 51.5**(chance 50) · `torus` R 3.1 / P 75.0 — **과생산/과소생산이다.**
⚠️ 최다 목적지 열 질량은 21~40%(균등 16.7%)라 **"특정 물체로 몰린다" 는 성립하지 않는다.**

가장 날카로운 대비: **probe 는 `torus` 를 100% 로 읽는데 채점은 3% 만 지켜낸다.**

---

## 5. 돌리는 법

```bash
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2

DRYRUN=1 bash z_research/scripts/run.sh surprise_c16t32 v11 vith    # 몇 초, GPU 0장

# 채점 (토큰 캐시 안 씀)
GPUS=8 BATCH_SIZE=16 DECODE_WORKERS=10 \
  bash z_research/scripts/run.sh surprise_c16t32 v11 vith

# probing (캐시 추출 후 head 학습)
GPUS=8 bash z_research/scripts/run.sh attn_probe v11 vith
```

### SLURM — v11 probing 은 **쪼개서 제출한다**

63 head 를 한 job 으로 돌리면 8시간 벽을 넘는다. `_prep` 이 캐시를 뽑고,
`(target × group half)` 6개가 `afterok` 로 붙는다.

```bash
sbatch --job-name=prep_v11 --export=ALL,WMA_RUN=1,P=attn_probe,D=v11,M=vith,GPUS=8 \
       z_research/scripts/sbatch.sh
```

⚠️ **직접 `sbatch` 할 때 `WMA_RUN=1` 을 반드시 붙인다.** 없으면 본체가 안 돌고 제출만 한다
(2026-08-30 에 이것 때문에 job 9,924 개가 생겼다 — `sbatch.sh` 상단 주석).

⚠️ **`--export` 는 콤마가 구분자다.** `SET` 에 YAML 리스트가 있으면 잘린다 → base64 로 싣는다.

쪼갠 결과를 하나로 합친다:

```bash
python z_research/scripts/analysis/merge_probe_runs.py \
  --base z_research/IntPhysGenV11/exp_results/attn_probe__v11_vith
# val_video_ids 가 다르면 죽는다 (다른 split 을 섞지 않는다)
```

---

## 6. 산출물 → 그림 → 문서

```bash
# 1. 산출물 검증 + report.json 생성 (summary.json 과 대조)
python z_research/scripts/analysis/report.py

# 2. 그림
python z_research/scripts/figures/plot_v11_surprise.py \
  --report z_research/IntPhysGenV11/exp_results/report.json \
  --outdir z_research/IntPhysGenV11/figures/surprise
python z_research/scripts/figures/plot_v11_probing.py     # 324칸 전수 대조 후 그린다

# 3. RESULTS_2026-08-30.md 의 "제2부" 재생성
python z_research/scripts/analysis/probing_md.py
```

**그림은 폴더가 정해져 있다** (`plot_v11_surprise.py` 상단 `FIGDIR` 표):

```
figures/surprise/
  01_condition/      가림 유무 · 등속 vs 등가속          beat 1-2
  02_occlusion_k/    k 반응 (k=0 과 k>0 사이에서만 끊긴다)  beat 2-4
  03_direction/      방향 분해 (100 / 0)                beat 4
  04_object_order/   predictor 가 무엇을 만드나           beat 4
  _superseded/       대체됨. 지우지 않는다
figures/probing/
  by_condition/  by_k/
```

**`FIGDIR` 에 없는 이름은 최상위에 떨어진다** — 새 그림이 눈에 띄라고 일부러 그렇게 뒀다.

---

## 7. 다음에 할 것

| 순 | 실험 | beat | 비용 |
|---|---|---|---|
| **1** | **부분공간 겹침 / Procrustes** (`p`·`z`·`h`, 조건별) | **5** | 캐시만, SVD 두 번 |
| 2 | `p` 문맥 잔상 검정 (위치 디코딩) | 3 | 캐시만 |
| 3 | IntPhys 2 의 V-JEPA 2 실측치 확인 | 1 | 문헌 |
| 4 | 정렬 사상 `W: p → h` (Procrustes) | 6·7 | 소량 |
| 5 | 채점 층 개입 (표준화 거리 · 순서 보정) | 6·7 | 구현 |
| 6 | 개입 후 순서 재측정 | 7 | 캐시만 |
| 7 | ViT-L · v11 변종 · IntPhys 1·2 재분해 | 8 | 추출 포함 |

**1 번이 논문의 새 중심이고 제일 싸다.** 결과가 어느 쪽이든 beat 가 선다.

---

## 8. 이 세트에서 걸렸던 함정

| 함정 | 무엇 |
|---|---|
| sbatch 자기제출 폭주 | `SLURM_BATCH_SCRIPT` 가 batch job 에서 비어 job 9,924 개. **`WMA_RUN=1`** 로 갈린다 |
| `--export` 콤마 | YAML 리스트가 잘려 `'[capsule'` 로 도착. **base64** 로 싣는다 |
| exit 0 on failure | `mp.Process` 를 join 안 해서 rank 실패에도 0 을 반환 → `afterok` 가 풀렸다. `evals/main.py` 에서 exitcode 확인 |
| prep 과소추출 | `probing.runs` 를 줄이면 캐시에 뽑히는 표현도 줄어든다. **runs 는 두고 `num_epochs` 만 1 로** |
| 7지선다 확률 변환 | Luce `s=(1-p)/p` 가 **최저 상대 하나에 지배**된다. `_superseded/` 로 내렸다 |
| block 밖 retrieval | 신호/잡음 0.715, `h` 대조군조차 40.3%(chance 14.3). **다시 시도하지 말 것** — `analysis/retrieval_confusion.py` 최상단 |

---

## 9. 재현

모든 수치는 `exp_results/` 의 `summary.json` / `report.json` 에서 다시 계산해 확인했다.
`report.json.overall[0].verified == true` 는 `summary.json` 과 대조가 끝났다는 뜻이고,
`plot_v11_surprise.py` 는 실행할 때마다 `cells` 를 재합산해 `overall` 과 대조한다
(불일치 0.02%p 넘으면 죽는다). `plot_v11_probing.py` 는 324칸을 전수 대조한다.
