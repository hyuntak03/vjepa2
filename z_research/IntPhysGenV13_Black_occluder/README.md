# IntPhysGen V13 — 검정 가림막

> v11 과 **같은 설계에서 가림막 색만 바꾼 대조군**이다.
> 본 실험 세트는 `../IntPhysGenV11/`, 타이밍 절제는 `../IntPhysGenV11_occlusion_timing_ablation/`.

---

## 한 줄

**가림막 색은 색 채점과 무관했다. 대신 가림막 텍스처가 모양 채점을 깎고 있었다.**

| | wood (v11) | black (v13) |
|---|---:|---:|
| 색 `visible → early` | −14.1 | **−13.5** ← 그대로 |
| 모양 `visible → early` | −6.2 | **0.0** ← 손해가 사라짐 |
| 영속성 `mid → late` | −38.3 | −40.1 ← 그대로 |

---

## 왜 만들었나

v11 에서 **색 채점만 `visible → early` 에서 −14.1** 로 떨어졌다. 영속성(−2.2)·모양(−6.2)과
달랐고, 그 단계에서 바뀌는 것은 프레임 수가 아니라 **가림막이 장면에 존재하게 된다**는
사실뿐이었다. 가림막이 그 자체로 큰 색면이니 색 통계를 지배하는 게 아닌가 — 그게 가설이었다.

**v11 로는 검정할 수 없었다.** 가림막이 3,456 clip 전부 `wood` 한 색이라 변이가 0 이다.
v13 은 `occ_color: black` 이고 나머지는 같다.

## 데이터

```
4,608 clip / 1,152 block / 2,304 matched pair
  condition   12 (visible 3 + early/mid/late 각 3)   — v11_full 과 같은 구성
  sym_k       0 (visible) / 1–4
  violation   vanish / shape / color
  occ_color   black          ← v11 은 wood
```

⚠️ **규모가 v11_full 의 1/9 이다.** 타이밍×위반 셀이 192쌍(v11 은 1,344~2,016).
   k 까지 쪼개면 16쌍이라 **읽을 수 없다.**

문맥 무결성 전수 감사 **36,864쌍, mismatch 0**
(`data_csv/intphysgen_v13_black/context_integrity.json`).
레지스트리: `configs/protocols/datasets.md` 의 `## v13_black`.
원본: `/local_datasets/world/world_analysis/IntPhysGenV13_occluder_black` (`/data2` 심볼릭).

---

## 결과 (`surprise_c16t32`, matched pair, chance 50%)

**overall 78.60%** (n=2304, 117 cells, `verified = true`) · v11_full 은 76.00% (n=21504)

### 타이밍 × 위반

| 위반 | timing | wood | **black** | 차 |
|---|---|---:|---:|---:|
| permanence | vis | 100.0 | **100.0** | +0.0 |
| | early | 97.8 | **99.0** | +1.2 |
| | mid | 95.8 | **97.4** | +1.6 |
| | late | 57.5 | **57.3** | −0.2 |
| shape | vis | 84.3 | **84.9** | +0.6 |
| | early | 78.1 | **84.9** | **+6.8** |
| | mid | 76.0 | **81.8** | **+5.7** |
| | late | 55.4 | **56.2** | +0.9 |
| colour | vis | 81.5 | **80.7** | −0.8 |
| | early | 67.4 | **67.2** | −0.2 |
| | mid | 68.7 | **65.6** | −3.1 |
| | late | 65.1 | **68.2** | +3.2 |

### 단계별 하락폭

| 위반 | | vis→early | early→mid | mid→late |
|---|---|---:|---:|---:|
| permanence | wood | −2.2 | −1.9 | **−38.3** |
| | black | −1.0 | −1.6 | **−40.1** |
| shape | wood | **−6.2** | −2.1 | −20.7 |
| | black | **+0.0** | −3.1 | −25.5 |
| colour | wood | **−14.1** | +1.3 | −3.6 |
| | black | **−13.5** | −1.6 | +2.6 |

---

## 말할 수 있는 것

✅ **가림막 색은 색 채점의 원인이 아니다.** `visible → early` 하락이 −14.1 → −13.5 로
   사실상 같다. **가설 기각.**

✅ **가림막 텍스처가 모양 채점을 깎는다.** `wood` 에서 −6.2 이던 `visible → early` 손해가
   `black` 에서 **0.0** 이 된다 (`mid` 도 −8.3 → −3.1). 나뭇결이 모양 판별을 방해했다는
   읽기와 맞는다.

✅ **`late` 절벽은 가림막 종류와 무관하다** (−38.3 → −40.1, −20.7 → −25.5).
   경계에 닿는 것 자체가 원인이라는 결론이 다시 선다.

## 말할 수 없는 것

❌ **색이 왜 `early` 에서 떨어지는지는 여전히 모른다.** 가림막 색도 아니고 텍스처도 아니다
   (텍스처를 없앤 black 에서도 −13.5). 남은 후보는 가림막의 **밝기/대비**, **가려진 프레임에서
   색 정보가 실제로 끊기는 것**, 또는 색 신호가 원래 가장 작아서(margin 0.27%) 무엇에든
   가장 먼저 무너지는 것이다.

❌ **"텍스처가 모양을 깎는다" 를 확정하진 못한다.** black 은 텍스처만 바뀐 게 아니라
   **밝기·대비도 함께** 바뀌었다. 텍스처만 분리하려면 같은 밝기의 무늬 없는 갈색 가림막이 필요하다.

❌ k 별로는 **못 읽는다** (셀당 16쌍).

---

## 그림

`plot_v11_timing.py` 가 만든다 (데이터셋 이름을 박지 않고 `<프로토콜>__*` 을 찾는다).
probing 을 안 돌렸으므로 **채점·기전 그림만** 나온다.

```
figures/
├── 01_scoring/     fig_timing_step   (+ by_k/fig_timing_k)
└── 03_mechanism/   fig_margin · fig_margin_vs_acc
```

읽는 법은 `../IntPhysGenV11_occlusion_timing_ablation/figures/README.md` 와 같다.

## 재현

```bash
# 채점 — 4,608 clip, 8 GPU 로 ~3분
GPUS=8 BATCH_SIZE=16 DECODE_WORKERS=10 \
  bash z_research/scripts/run.sh surprise_c16t32 v13_black vith

python z_research/scripts/analysis/report.py \
  --run z_research/IntPhysGenV13_Black_occluder/exp_results/surprise_c16t32__v13_black_vith:data_csv/intphysgen_v13_black/index_probe.csv \
  --out z_research/IntPhysGenV13_Black_occluder/exp_results/report.json

python z_research/scripts/figures/plot_v11_timing.py \
  --exp    z_research/IntPhysGenV13_Black_occluder/exp_results \
  --index  data_csv/intphysgen_v13_black/index_probe.csv \
  --outdir z_research/IntPhysGenV13_Black_occluder/figures
```

위 표의 v13 수치는 `report.json` 의 `scoring.cells` 를 재합산해 얻었고
`overall` 과 0.02%p 안에서 일치한다. v11 수치는
`../IntPhysGenV11_occlusion_timing_ablation/exp_results/report.json` 에서 같은 방식으로 뽑았다.

**probing 은 안 돌렸다.** 돌리려면 캐시를 새로 뽑아야 한다 (태그가 달라 재사용 불가, ~90G).
