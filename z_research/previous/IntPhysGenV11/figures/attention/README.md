# attention 그림 — predictor 가 무엇을 보나 (관찰)

전부 `analysis/attention/predictor_attn/plot.py` 가 만들고, 폴더 배정은 그 파일 상단의
`FIGDIR` 표가 정한다. **여기 없는 이름은 최상위에 떨어진다** — 새 그림이 눈에 띄라고 일부러다.

⚠️ 시간 단위는 **샘플 프레임**이다 (프로토콜이 읽는 32장). 튜블릿 1 = 샘플 2장 = 원본 6장.
⚠️ 질의 튜블릿 0–7 은 진짜 문맥 토큰, 8–15 는 **mask token** 이다. 성격이 다르니 섞어 읽지 말 것.

## `01_layerwise/`

| 그림 | 무엇 |
|---|---|
| `fig_attn_matrix` | 층별 (질의 프레임 × 키 프레임) attention 질량. **주력.** 행 합 = 1 |
| `fig_frame_profile` | 질의별로 각 키 프레임에 준 질량, 층을 색으로 겹친다 |

핵심: **layer 0 에서 미래 토큰이 문맥에 질량 0.885 를 주고, 그게 문맥 마지막 프레임에 몰린다**
(부호 Δt −12.4 프레임). 미래 블록은 거의 비어 있다. 층이 깊어질수록 미래 쪽으로 퍼진다.

## `02_distance/` — 헤드 × 층 히트맵 (Interpreting Physics in Video World Models, Fig.3 형식)

| 그림 | 무엇 |
|---|---|
| `fig_head_sdist` | y = 헤드 12, x = 층 12, 색 = **공간 attention distance (patches)**. 작을수록 진한 파랑 |
| `fig_head_tdist` | 같은 판, 시간 거리 (frames) |

질의는 **future(mask token) 16개 중 8개**만 쓰고(`--queries future`), 조건 6개는 평균한다
(03_condition/ 에서 조건 차이가 없음을 확인했다). `--queries context|all` 로 바꿀 수 있다.

읽히는 것:
- **층 0–2 에 국소 헤드가 몰려 있다** — 0.4~2.3 패치 (헤드 2·8·6·9·1). 논문이 말하는
  "spatiotemporally local heads" 가 predictor 에서는 **입구**에 있다
- **층 8–11 은 전부 5~8 패치로 균질**하다. 국소 헤드가 없다
- 논문의 인코더(ViT-L, 24층)는 국소 헤드가 **중간(Physics Emergence Zone)** 에 생겼다.
  predictor 는 12층짜리이고 입력이 이미 인코더 출력이라 **위치가 다른 것이 당연**하다.
  같은 현상으로 읽지 말 것

⚠️ 거리 단위가 논문(패치)과 같도록 `sdist_patch` 를 쓴다. 이전 판은 px 였다.

### `_superseded/`

| 그림 | 왜 물러났나 |
|---|---|
| `fig_attn_distance` | 층별 산점 + 평균선. 헤드 12개가 한 x 에 겹쳐 **어느 헤드가 국소인지 안 읽혔다**. 위 히트맵이 같은 수치를 헤드 단위로 편 것이다 |

## `03_condition/`

| 그림 | 무엇 |
|---|---|
| `fig_attn_group` | 조건 6개 비교 (문맥 질량 · 공간 거리 · 엔트로피) |

⚠️ **여섯 조건이 거의 완전히 겹친다.** 버그가 아니라 결과다 — 이 집계 수준에서 predictor
attention 은 가림 유무·등속/등가속과 무관하다. 조건 차이를 찾으려면 다른 축(헤드 단위,
value 가중, k 분해)으로 가야 한다.

## 재현

```bash
PYTHONPATH=. python -m analysis.attention.predictor_attn.extract \
    --dataset v11 --model vith --group-by condition --per-group 32
PYTHONPATH=. python -m analysis.attention.predictor_attn.plot \
    --npz z_research/IntPhysGenV11/exp_results/predictor_attn__v11_vith/attn.npz \
    --outdir z_research/IntPhysGenV11/figures/attention          # --which head 만도 됨
```
`plot.py` 는 실행할 때마다 `mass` 행 합을 재검사하고(1e-3 초과면 죽는다)
`summary.json.verify` 를 같이 찍는다.
