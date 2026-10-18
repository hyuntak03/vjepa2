# VOBench_v1 · V-JEPA 2 ViT-H · probing — 튜블릿 정체 probe

> 결과 요약 · 해석 · 단서 · 철회는 상위 [`../../README.md`](../../README.md) §3-10. 정의는 `z_research/scripts/analysis/vobench_tubelet_probe.py` 머리말.

**묻는 것**: target encoder h 의 가림 없는 튜블릿으로 학습한 정체 head (모양 × 색 결합 + none, AttentiveClassifier depth 1 · query 1) 가
encoder 가림 · predictor 보임 · predictor late (문맥 끝 가림) 에서 무엇을 답하나. primitive · realistic 따로.

| 파일 | 무엇 |
|---|---|
| `results/<run>/{primitive,realistic}.json · RESULTS.md` | json 에 칸별 샘플 예측 (`samples` = [clip, t, 상태, k, pred, gt]) 포함 (2026-10-10 재실행 — 기존 집계 288 값과 동일). run = `e35_s0_user` · `e35_s1` · `e10_s0` · `e10_s1` (epoch · head 시드) · `e35_s0_realistic_only` (= `e35_s0_user` 와 같은 수치) |
| `results/_invalid_run1_realistic_collapsed/` | **무효** — realistic 학습 붕괴 (NOTE.md) |
| `figures/fig_probe_transfer` | 패널 = objset. 칸마다 정답 정체 / 다른 물체 / none 으로 답한 비율 (유효 4 run 평균), 오차막대 = 정답의 run 간 최소–최대. 회색 = predictor 칸 |
| `figures/fig_probe_by_t` | 미래 튜블릿 t8–t15 별 정답 % · none % (predictor 보임 실선 · late 재등장 파선), 띠 = run 간 범위, 회색 띠 = encoder 보임 천장 |
| `figures/fig_probe_visible` | 같은 꼴, **visible (가림막 없음) 클립만** (objset 마다 test 클립 31–34 개 = 튜블릿 시점당 n) |
| `figures/fig_probe_late` | **late (문맥 끝 가림) 클립만** 시간축 t0–t15: 초록 = encoder h, 주황 = predictor p (t8–t15). 실선 = 정답 정체 %, 점선 = none 을 고른 %. 빈 표식 = 가린 튜블릿, 회색 띠 = 가림 구간 (k = 2, 3 의 t7–t8; k = 1 은 섞임이라 그 두 칸에서 빠짐). 띠 = run 간 범위 |
| `results/p_only_<e>_<s>/` · `figures/p_only/` | **head 를 predictor p 로 학습** (문맥 끝 보임 조건 visible · early · mid 의 미래 튜블릿 t8–15) — p 에 정보가 있는지의 상한. 그림 4 장은 위와 같은 꼴, 단 encoder 칸 · 초록 선은 p head 를 h 에 건 값 (역방향 이식). `--train-src p` · `plot_vobench_probe.py --set p_only` |
| `temporal_generalization/` | **시점 하나에서만 학습한 head 를 모든 미래 시점에** (shape · color · joint, 시드 0–2). `RESULTS.md` · `summary.json` · `runs/` (그림 없음). p 는 같은 시점이면 h 만큼 읽히고 (shape 95–98), 앞으로 갈 때만 전이가 떨어진다 (4 칸 이상 46–55 vs 뒤로 75–78); h 는 양방향 비슷. `vobench_tempgen.py` |
| `figures/figure_values.json` | 그림에 쓴 run 별 값 |

캐시 (h 16 튜블릿 · p 8, fp16 23 GB): `/local_datasets/world/world_analysis/cache/vobench_v1_vith_tubeprobe/`


## 중간층 (target encoder 16 / 32 번째 block) vs 마지막 층 (2026-10-13)

h 를 target encoder **16 번째 block 출력** (out_layers, encoder norm + affine-free LN) 으로 바꿔 같은 probe (h 보임으로 학습 → h 에 시점별) 를 다시 했다. 나머지 (split · head · 35 epoch · 시드 0 · 1) 는 마지막 층 run (`e35_s0_user` · `e35_s1`) 과 같다.
결과 `results/L16_e35_s{0,1}/` · 그림 `figures/layers/fig_probe_layers.{png,pdf}` (초록 = 마지막 층, 파랑 = 16 층, 실선 = 정체 정답 %, 점선 = none 고른 %, 회색 = 실제로 가려진 튜블릿) · 수치 `figures/layers/layers.json`.

| h 정체 probe (시드 0 / 1) | 보임 튜블릿 정체 | 가림 튜블릿 정체 | 가림: none | 가림: 다른 물체 |
|---|---|---|---|---|
| 마지막 층 · primitive | 97.4 / 97.1 | **9.3 / 14.5** | 37.0 / 27.2 | 53.7 / 58.3 |
| 마지막 층 · realistic | 89.6 / 90.7 | **13.5 / 12.5** | 21.2 / 8.7 | 65.4 / 78.8 |
| **16 층** · primitive | 99.0 / 99.1 | **0.7 / 0.5** | **71.3 / 74.3** | 27.9 / 25.2 |
| **16 층** · realistic | 92.8 / 90.2 | **0.0 / 0.7** | **100 / 85.3** | 0.0 / 13.9 |

- **중간층은 가려진 튜블릿을 '물체 없음' 으로 읽는다** (정체 0–1 %, none 71–100 %) — 그 시점에 화면에 보이는 것을 담는다. 마지막 층은 정체가 9–15 % (chance 2 %) 로 조금 남고, 대신 '다른 물체' 로 답하는 비율이 높다.
- 즉 가려진 물체의 정체가 조금이라도 들어오는 것은 **뒤쪽 층에서** (시간 attention 으로 다른 프레임의 정보를 섞은 뒤) 이고, 중간층에는 없다. 마지막 층의 이 정체도 약하다.
- 보일 때는 16 층이 같거나 약간 높다 (99 vs 97 · 91 vs 90).
- ⚠️ 단서: 마지막 층의 '다른 물체' 비율이 높은 데는 가림막 판이 물체처럼 읽히는 문제 (위 절) 가 섞인다 — 마지막 층의 정체 9–15 % 가 '이어 온 정체' 인지 판 오독의 우연 일치인지는 이 probe 로 못 가른다 (chance 2 % 보다는 확실히 높다). head 는 층마다 따로 학습했다. 칸당 가림 튜블릿 수는 조건 · objset 마다 수백.

## 재현

```bash
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
$P z_research/scripts/analysis/vobench_tubelet_probe.py extract --gpus 4
$P z_research/scripts/analysis/vobench_tubelet_probe.py probe --gpu 0 --epochs 35 --seed 0 --tag e35_s0_user   # e35_s1 · e10_s0 · e10_s1 도 같은 꼴
$P z_research/scripts/figures/plot_vobench_probe.py                                                             # → figures/
```
