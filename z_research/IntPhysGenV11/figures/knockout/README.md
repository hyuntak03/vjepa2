# knockout 그림 — 끊으면 무엇이 무너지나 (개입)

전부 `analysis/attention/knockout/plot.py` 가 만든다. 색은 `style_map()` 이 결과 전체에서
**한 번만** 배정하므로 세 그림에서 같은 간선이 같은 색이다 (나란히 놓고 읽으라고).

⚠️ Δ 는 `meta.delta_ref` 기준선 대비다 (기본 `clean_null` = 전부 True 인 마스크로 돈 판).
   전수 채점 73.37% 가 아니라 **그 표본의 기준선**(v11 ViT-H 144 block 에서 72.92%) 대비다.

| 그림 | 무엇 |
|---|---|
| `fig_knockout_layers` | 층별 Δ정확도, 간선 종류별 선. **주력** |
| `fig_knockout_edges` | 전 층 동시에 끊었을 때 Δ 를 조건으로 쪼갠 막대 |
| `fig_knockout_drift` | 같은 축의 예측 이동량 (`|p−p₀|/|p₀|`, `cos`) |

핵심: **`mask→ctx` 만 중요하고, 그 안에서 층 8 이 병목이다** (단일 층 Δ−18.06,
이웃 층의 6배). 나머지 세 간선은 전 층을 다 끊어도 Δ 가 ±1.4pp 안이다.

## `fig_knockout_drift` 를 반드시 같이 볼 것

정확도는 계단 함수다 (분해능 = `1/(2·n_block)`, 144 block 이면 0.347pp).
**Δ=0 이 "아무 일도 없었다" 를 뜻하지 않는다** — `ctx→ctx@all` 은 Δacc 0.00 인데
예측은 `rel_l1 0.129 / cos 0.993` 만큼 움직였다. drift 는 클립 연속값이라 계단이 없다.

drift 는 검증 역할도 한다: **`ctx→*@L11` 의 drift 가 정확히 0.000** 이다. 마지막 층에서
문맥 토큰으로 **들어가는** 간선은 mask 위치 출력에 원리적으로 영향이 없다(문맥 토큰의
L11 출력은 버려진다). 도구가 그 구조적 0 을 재현한다는 뜻이다.

## `fig_knockout_edges` 를 읽을 때 — Δ 는 조건 간에 비교할 수 없다

절단 후 정확도가 전부 chance 아래로 내려가므로 **Δ 의 크기가 clean 수준에 끌려간다.**

| condition | clean | Δ (`mask→ctx@all`) | 절단 후 |
|---|---:|---:|---:|
| `static_visible` | 93.75 | −64.58 | 29.17 |
| `moving_visible_flat` | 91.67 | −58.33 | 33.33 |
| `moving_visible` | 79.17 | −45.83 | 33.33 |
| `moving_occlusion_flat` | 64.58 | −20.83 | 43.75 |
| `moving_occlusion` | 54.17 | −12.50 | 41.67 |
| `static_occlusion` | 54.17 | **−47.92** | **6.25** |

⚠️ **`static_occlusion` 이 단순한 바닥 효과 설명을 깬다** — `moving_occlusion` 과 clean 이
같은 54.17 인데 절단 후 6.25 로 떨어진다 (chance 한참 아래 = 체계적 역전).
그러니 **"가림 조건은 덜 다친다" 로 읽으면 안 된다.** Δ 와 절단 후 절대값을 **같이** 볼 것.

## 재현

```bash
PYTHONPATH=. python -m analysis.attention.knockout.run \
    --dataset v11 --model vith --blocks-per-cond 24 \
    --edge mask..ctx --edge mask..mask --edge ctx..ctx --edge ctx..mask \
    --edge ctx..ctx:diff_tub --layers each \
    --spec mask..ctx@all --spec mask..mask@all --spec ctx..ctx@all \
    --spec ctx..mask@all --spec ctx..ctx:diff_tub@all \
    --metrics surprise_acc,surprise_l1,pred_drift
PYTHONPATH=. python -m analysis.attention.knockout.plot \
    --results z_research/IntPhysGenV11/exp_results/knockout__v11_vith/results.json \
    --outdir  z_research/IntPhysGenV11/figures/knockout
```
`plot.py` 는 `results.json.verify.resume_vs_forward_max_abs_diff` 가 `0.0` 이 아니면 죽는다.
