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

## 재현

```bash
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
$P z_research/scripts/analysis/vobench_tubelet_probe.py extract --gpus 4
$P z_research/scripts/analysis/vobench_tubelet_probe.py probe --gpu 0 --epochs 35 --seed 0 --tag e35_s0_user   # e35_s1 · e10_s0 · e10_s1 도 같은 꼴
$P z_research/scripts/figures/plot_vobench_probe.py                                                             # → figures/
```
