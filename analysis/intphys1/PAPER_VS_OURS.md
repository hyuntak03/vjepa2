# V-JEPA 2 L on IntPhys — Paper (Joseph 2026) vs Ours (5-fold CV)

Single-source-of-truth 비교 문서. Paper Fig 1 / Appendix C.1.1 (Fig 6) / C.1.3 (Fig 10) vs. our 5-fold CV reproduction (`intphys1_vitl_5fold_appb`).

---

## TL;DR

- **정성적 곡선 shape는 재현됨.** L0 ≈ chance → 중간층에서 급상승 (PEZ) → 중간 plateau → 마지막 층으로 갈수록 완만한 하락. Paper와 our 모두 inverted-U / mesa-with-tail pattern.
- **정량적 peak 값은 낮게 재현됨.** Paper Fig 1/6 (chart-only) peak ≈ **95–100%** (Full Task, V-JEPA 2 L). Ours: **layer 15 = 80.83 ± 3.98%** (best-HP-per-layer, 5-fold CV, Appendix B sweep).
- **Peak layer 위치가 뒤로 밀림.** Paper Fig 6 시각 판독 peak = **layer 8–10 (fraction ≈ 0.35–0.43)**. Ours peak = **layer 15 (fraction = 0.6522)**. 다만 our 결과에서도 layer 7–17 구간이 78–81% plateau라 "middle third에서 peak" 서술과 완전히 배치되지는 않음.
- **C.11 프로토콜은 아직 실행 전.** `intphys1_vitl_5fold_c11` 5-fold는 하나도 돌아가지 않았음. 현 비교는 **Appendix B sweep** 결과에 한정.

---

## 1. Paper 참조값 (Joseph 2026)

Text로 명시된 값은 매우 제한적. 대부분 chart-only 시각 판독.

| 항목 | 값 | 출처 | 종류 |
|---|---|---|---|
| Chance | 50% | Fig 1 legend, §4.2 p4 | text |
| Post-PEZ high-performance band (L/H/G 전체) | ∼85–95% | §4.2 p4 | text |
| Emergence 위치 | ≈ one-third depth | §4.2 p4, Abstract, §1 | text |
| Peak 위치 서술 | "middle third of the encoder" | §4.2 p4, Abstract | text |
| V-JEPA 2 L layer 수 | 24 (0–23) | Fig 6 x-axis | implicit |
| Fig 6 Full Task **peak accuracy** (V-JEPA 2 L) | **chart-only, ~95–100%** | Fig 6 p14 | 시각 판독 |
| Fig 6 Full Task **peak layer** (V-JEPA 2 L) | **chart-only, layer ~8–10** | Fig 6 p14 | 시각 판독 |
| Fig 6 layer-0 accuracy (V-JEPA 2 L) | **chart-only, ~45–50%** | Fig 6 p14 | 시각 판독 |
| Fig 6 layer-23 accuracy (V-JEPA 2 L, Full Task) | **chart-only, ~70–75%** | Fig 6 p14 | 시각 판독 |
| PEZ 밴드 폭 | **chart-only, fraction ≈ 0.2–0.4** | Fig 1 | 시각 판독 |
| SD / per-fold 값 | 명시 없음 | — | gap |
| 층별 winning HP | 명시 없음 | Appendix B p13 grid만 서술 | gap |

Sweep grid (Appendix B, p13): LR ∈ {1e-4, 4.3e-4, 1e-3, 3.3e-3, 3.5e-3} × WD ∈ {0.01, 0.1, 0.4, 0.8} = **20 configs**, **5-fold grouped CV**. Ours가 그대로 사용.

---

## 2. Our 5-fold CV 결과 (Appendix B sweep)

- Config: `configs/analysis/probing/intphys1_vitl_5fold_appb.yaml`
- Aggregation: `logs/analysis_vlm/probe_intphys1_vitl_5fold_appb_agg/`
- `n_folds=5, num_layers=24, num_hps=20` — Appendix B 그대로.
- Best-HP-per-layer 관점 (`per_layer_best_hp.csv`).

주요 anchor points:

| 항목 | 값 |
|---|---|
| L0 mean_acc | **58.06 ± 2.85 %** (best_hp=18) |
| Peak layer | **15** (fraction = 15/23 ≈ 0.6522) |
| Peak mean_acc | **80.83 ± 3.98 %** (best_hp=19) |
| L23 mean_acc | **77.22 ± 4.77 %** (best_hp=9) |
| Rise L0 → L15 | +22.78 pts |
| Drop L15 → L23 | −3.61 pts |
| Mid-late plateau (L7–L17) | 대부분 78–81% |

Appendix B config는 done. **C.11 프로토콜 5-fold는 아직 미실행 (0/5 folds)** — 별도 sanity check로 언젠가 돌려서 값이 크게 달라지는지 확인 필요.

---

## 3. Side-by-side per-layer 비교

Paper text에는 층별 수치가 없고 Fig 6는 chart-only이므로, key anchor point만 비교. Paper 값은 시각 판독이므로 "chart-only, ~"로 표기.

| Layer | Fraction | Paper Fig 6 (Full Task, V-JEPA 2 L) — chart-only | Ours (5-fold, best HP) | 격차 (approx) |
|---:|---:|---|---|---|
| 0 | 0.000 | ~45–50% | **58.06 ± 2.85** | +8–13 pts (ours 더 높음) |
| 2–3 | 0.087–0.130 | sub-chance dip 존재 (~35–45%) | 60.83 / 61.39 | ours는 dip 없음 |
| 5 | 0.217 | 급상승 시작 | 73.06 ± 3.34 | — |
| 7 | 0.304 | 이미 peak 근처 (~95%) | 79.72 ± 2.32 | **−15 pts** |
| 8–10 | 0.348–0.435 | **paper peak zone, ~95–100%** | 78.33 / 80.56 / 80.28 | **−15 to −20 pts** |
| 12 | 0.522 | plateau 시작 하락 | 78.61 ± 3.88 | — |
| 15 | 0.652 | 이미 하락 구간 | **our peak = 80.83 ± 3.98** | ours peak가 뒤로 밀림 |
| 20 | 0.870 | ~75–80% | 77.78 ± 6.21 | 근접 |
| 23 | 1.000 | ~70–75% | 77.22 ± 4.77 | ours가 약간 더 높음 |

**가장 크게 어긋나는 층**: L7–L10. Paper는 이 구간에서 이미 95%대의 near-ceiling plateau인 반면, ours는 78–80% 수준. Paper의 sub-chance early dip (L2–L3)도 우리 결과에는 없음.

**곡선 shape**: 두 결과 모두 (a) L0 near-chance, (b) 중간층 상승, (c) 중간층 plateau, (d) 뒷단으로 하락 — inverted-U 패턴 일치. 다만 our peak height는 −15 pts, peak 위치는 우측으로 밀림.

---

## 4. Known-and-controlled (파이프라인이 논문과 맞는 요소들)

이미 우리가 확인/맞춰둔 것들:

| 요소 | 상태 |
|---|---|
| Backbone | V-JEPA 2 L (24 layers, 0–23) — 일치 |
| Task | possible-vs-impossible (Full Task) |
| Probe type | linear probe |
| CV protocol | **5-fold grouped CV** (n_folds=5) — Appendix B 그대로 |
| HP sweep grid | **20 configs (5 LR × 4 WD)** — Appendix B와 grid 크기 일치 |
| Per-layer 평가 | 24 layers 전부 커버 |
| Aggregation | mean ± std across 5 folds, best HP per layer |
| Output | `per_layer_best_hp.csv`, `stage_val_acc_mean_std.png` |

---

## 5. 왜 gap이 있는가 (Top-3 diagnosis)

### (1) `self.norm` LayerNorm signature 차이 — 가장 유력

- V-JEPA 2 encoder는 각 block 뒤 hidden state에 대해 최종 `self.norm`을 통과시킬지 여부가 layer-wise probe에서 결정적임.
- Paper Appendix에는 pre/post-norm 선택이 명시 없음. Paper Fig 6에서 layer-8–10이 95%대 near-ceiling에 도달하는 sharpness는 **norm 후 hidden state**에 probe를 붙였을 가능성이 높음 — L0에서도 살짝 chance 아래로 dip이 나오는 것 (variance 큰 raw hidden) 및 중간층 급상승 (norm 안정화 후 sharp transition)이 이 가설과 정합.
- Ours가 raw pre-norm hidden state를 뽑고 있다면, mid-layer sharpness가 억제되고 peak height가 낮아지고 (78–81 plateau), peak가 살짝 뒤로 밀리는 (layer 15) 현상을 그대로 설명함.
- **Evidence in our data**: our L0가 58%로 paper Fig 6의 ~45–50% 대비 유의미하게 위 (dip 없음). 즉 우리 pipeline은 low-variance / whitening 없이 raw feature를 그대로 쓰고 있을 가능성.
- **Action**: `evals/analysis/modelcustom/vit_encoder_multilayer.py`에서 각 block의 hidden state를 뽑을 때 `self.norm` (또는 block마다의 post-LN)이 적용되고 있는지 재확인.

### (2) Per-layer HP selection의 자유도 차이

- Paper는 Appendix B에서 grid를 명시하지만, "**per-layer** best HP"인지 "**global** best HP"인지 명시가 없음. Paper Fig 6은 mean curve만 그리고 있어서 실제 HP selection이 어떻게 되었는지 chart로도 알 수 없음.
- Ours는 명시적으로 **per-layer best HP** 사용 (`per_layer_best_hp.csv`).
- 만약 paper가 per-layer HP를 썼다면, 우리와 동일 조건인데도 결과가 다름 → (1) 요인이 지배적.
- 만약 paper가 global best HP만 썼다면, 오히려 paper 쪽이 불리해질 텐데 그래도 paper peak이 훨씬 높다는 건 (1)이나 (3)이 지배한다는 신호.

### (3) 통계 노이즈 및 데이터셋 split 상세

- Our fold-별 편차: peak layer 기준 SD = 3.98 pts, layer 18에서는 SD = 7.39 pts로 큼. Paper Fig 6/10의 SD 밴드도 시각적으로 넓지만 정확한 수치 미공개.
- Paper는 5-fold **grouped** CV라고만 함 (Appendix B). Grouping key (sample 단위? sequence 단위? subtask 단위?)에 따라 accuracy가 몇 pts 차이날 수 있음.
- 예: `f0`은 여러 layer에서 유독 높은 값 (L15에서 86.11, L18에서 90.28), `f3`은 유독 낮음 (L18에서 75.00). Grouping이 다르면 이 편차가 사라지거나 반대로 커질 수 있음.

**요약**: **(1) norm signature > (2) HP selection semantics > (3) group split 상세**. (1)이 해결되지 않으면 나머지 두 요인만으로 15pt gap을 다 못 메움.

---

## 6. Next tests (gap을 더 좁히려면)

우선순위 순.

1. **`self.norm` 위치 확인·수정**: `vit_encoder_multilayer.py`에서 각 block output에 대해 `self.norm(x)` post-LN을 적용한 버전과 raw pre-LN 버전 두 가지 feature를 뽑아 동일 sweep을 돌린다. Post-LN 버전에서 mid-layer peak가 90%+로 올라오는지 확인. → 올라오면 gap의 root cause 확정.
2. **C.11 프로토콜 5-fold 실행**: 아직 미실행. `configs/analysis/probing/intphys1_vitl_5fold_c11.yaml`로 f0–f4 5 folds + agg 를 돌려서 Appendix B sweep 결과와 비교. 두 프로토콜 차이가 몇 pts 나는지 quantify.
3. **Per-fold group key 명시화**: `src/datasets/data_manager.py`에서 grouping이 sample id / video id / trajectory 중 무엇인지 확인. Paper의 "grouped CV" 정의가 없으므로, 우리는 최소한 우리 grouping 정책을 문서화.
4. **Subtask breakdown (Fig 10 재현)**: Object Permanence / Shape Constancy / Spatio-Temporal Continuity 각각으로 나누어 same-shape을 재현하는지 확인. Paper §C.1.3은 "same one-third emergence pattern across all three" 라 주장 — 우리 실험에서도 이 서술이 유지되면 정성 재현 성공.
5. **초반 sub-chance dip 재현 여부**: Paper Fig 6 Row 4 (Spatio-Temporal Continuity)는 L2–L3에서 chance 아래로 내려감. Ours는 L0=58% 그 자체가 chance 위. Norm signature 수정 후 이 dip이 재현되면 (1) 확정.
6. **Attentive MLP probe 대조**: Paper Fig 1 오른쪽 패널은 attentive-MLP probe로 peak가 더 확실히 100%에 가까움. Linear probe로만 재현 시도 후 attentive-MLP도 sanity check로 돌리면, "probe head choice가 얼마나 기여했나" 정량화 가능.

---

## Appendix — Our per-layer full table (참고)

`configs/analysis/probing/logs/analysis_vlm/probe_intphys1_vitl_5fold_appb_agg/per_layer_best_hp.csv` 원본.

| layer | fraction | mean_acc | std_acc | best_hp | per-fold (f0,f1,f2,f3,f4) |
|---:|---:|---:|---:|---:|---|
| 0  | 0.0000 | 58.0556 | 2.8464 | 18 | 61.11, 61.11, 55.56, 55.56, 56.94 |
| 1  | 0.0435 | 60.8333 | 4.9496 | 18 | 65.28, 66.67, 56.94, 55.56, 59.72 |
| 2  | 0.0870 | 60.8333 | 4.6481 | 18 | 66.67, 63.89, 56.94, 55.56, 61.11 |
| 3  | 0.1304 | 61.3889 | 4.5432 | 18 | 68.06, 63.89, 58.33, 56.94, 59.72 |
| 4  | 0.1739 | 63.3333 | 4.1201 | 17 | 68.06, 65.28, 62.50, 56.94, 63.89 |
| 5  | 0.2174 | 73.0556 | 3.3449 | 12 | 77.78, 75.00, 72.22, 69.44, 70.83 |
| 6  | 0.2609 | 77.7778 | 3.1056 | 19 | 81.94, 79.17, 76.39, 73.61, 77.78 |
| 7  | 0.3043 | 79.7222 | 2.3241 | 17 | 83.33, 77.78, 79.17, 77.78, 80.56 |
| 8  | 0.3478 | 78.3333 | 1.5836 | 19 | 80.56, 77.78, 79.17, 76.39, 77.78 |
| 9  | 0.3913 | 80.5556 | 4.9105 | 20 | 87.50, 80.56, 80.56, 73.61, 80.56 |
| 10 | 0.4348 | 80.2778 | 5.1408 | 17 | 87.50, 81.94, 77.78, 73.61, 80.56 |
| 11 | 0.4783 | 80.0000 | 5.2521 | 15 | 84.72, 80.56, 77.78, 72.22, 84.72 |
| 12 | 0.5217 | 78.6111 | 3.8790 | 10 | 81.94, 80.56, 77.78, 72.22, 80.56 |
| 13 | 0.5652 | 80.2778 | 3.0110 | 17 | 83.33, 79.17, 79.17, 76.39, 83.33 |
| 14 | 0.6087 | 80.0000 | 3.1975 | 20 | 84.72, 77.78, 80.56, 76.39, 80.56 |
| **15** | **0.6522** | **80.8333** | **3.9772** | **19** | **86.11, 76.39, 80.56, 77.78, 83.33** |
| 16 | 0.6957 | 77.7778 | 4.8113 | 20 | 86.11, 77.78, 75.00, 75.00, 75.00 |
| 17 | 0.7391 | 80.0000 | 4.6688 | 18 | 87.50, 77.78, 79.17, 75.00, 80.56 |
| 18 | 0.7826 | 78.6111 | 7.3886 | 17 | 90.28, 80.56, 76.39, 75.00, 70.83 |
| 19 | 0.8261 | 79.1667 | 4.2808 | 15 | 86.11, 79.17, 79.17, 75.00, 76.39 |
| 20 | 0.8696 | 77.7778 | 6.2113 | 18 | 87.50, 76.39, 79.17, 75.00, 70.83 |
| 21 | 0.9130 | 77.2222 | 5.8597 | 15 | 86.11, 76.39, 79.17, 70.83, 73.61 |
| 22 | 0.9565 | 76.1111 | 5.7601 | 16 | 84.72, 77.78, 76.39, 70.83, 70.83 |
| 23 | 1.0000 | 77.2222 | 4.7710 |  9 | 81.94, 80.56, 79.17, 73.61, 70.83 |

관련 파일 (모두 절대경로):

- `/data/hyuntak/project/2026/2027_cvpr/vjepa2/configs/analysis/probing/logs/analysis_vlm/probe_intphys1_vitl_5fold_appb_agg/aggregated.json`
- `/data/hyuntak/project/2026/2027_cvpr/vjepa2/configs/analysis/probing/logs/analysis_vlm/probe_intphys1_vitl_5fold_appb_agg/per_layer_best_hp.csv`
- `/data/hyuntak/project/2026/2027_cvpr/vjepa2/configs/analysis/probing/logs/analysis_vlm/probe_intphys1_vitl_5fold_appb_agg/stage_val_acc_mean_std.png`
- `/data/hyuntak/project/2026/2027_cvpr/vjepa2/configs/analysis/probing/intphys1_vitl_5fold_appb.yaml`
- `/data/hyuntak/project/2026/2027_cvpr/vjepa2/configs/analysis/probing/intphys1_vitl_5fold_c11.yaml` (실행 전)
- Paper PDF: `/data/hyuntak/project/2026/2027_cvpr/vjepa2/Interpreting Physics in Video World Models.pdf` (pages 1, 4, 13, 14, 18)
