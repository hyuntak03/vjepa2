# VOBench_v1 · VideoMAEv2-g (창 c8t16_s6) · surprise

> 결과 요약 · 해석 · 단서는 상위 [`../../README.md`](../../README.md) §3-9. 이 폴더는 정리본만 둔다.

| 파일 | 무엇 |
|---|---|
| `tables.md` · `tables.json` | objset (primitive · realistic) 별 채점표 — 위반 × 가림 시점 (A/B 방향) · 위반 × k · 운동 · 배경 · 물체 · 복사 기준선. per_block.json 에서 재계산 + summary.json assert |
| `figures/fig_timing` | 행 = objset, 열 = 위반, x = 가림 시점 (회색 = late). 막대 = 정확도 [block 부트스트랩 95 % CI], 점 = A · B 방향 |
| `figures/fig_k` | x = k (0 = 가림 없음), 선 = early · mid · late |
| `figures/fig_late_motion` | late 만, x = 운동 6 종 |
| `figures/fig_motion_{visible,early,mid}` | `fig_late_motion` 과 같은 꼴, 가림 시점만 바꿈 (2026-10-11). ⚠️ 칸당 쌍 수: visible **4** (block 2) · early / mid / late 12 (block 6) — visible 은 경향으로도 약하다 |
| `figures/fig_permanence_context` | permanence 만: 물체 문맥 (있는데→있는) · 빈 문맥 (없는데→없는), 가림 시점별 · late × 운동 |

원시 산출물: `../../exp_results/surprise_c16t32__vobench_v1_videomae2g_wc8t16_s6/` (레지스트리 이름 `videomae2g`).

## 재현

```bash
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
$P z_research/scripts/analysis/vobench_tables.py --model videomae_v2     # → tables.{md,json}
$P z_research/scripts/figures/plot_vobench.py --model videomae_v2        # → figures/
```
(채점 명령은 상위 README `## 재현`)
