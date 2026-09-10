# predictor attention knockout — 간선을 끊고 성능 변화를 본다

> 레포 규칙은 루트 `CLAUDE.md`, 실행 계약은 `configs/protocols/README.md`.
> 어떤 프레임을 보는지 **관찰**만 하는 짝은 `../predictor_attn/`.

**한 줄: predictor 안의 attention 간선을 (질의집합 → 키집합 × 층) 단위로 끊고,
확립된 채점 정확도가 얼마나 떨어지는지로 information flow 를 잰다.**

frozen 모델 위에서 돌고 재학습이 없다.

---

## 1. 무엇을 끊나

predictor 입력은 `[문맥 토큰 C·S개] + [미래 자리의 mask 토큰 (T−C)·S개]` 가 인덱스 순으로
붙은 N = T·S 시퀀스다. bidirectional 이라 간선이 네 종류다:

| 간선 | 무엇 |
|---|---|
| `ctx→ctx` | 문맥 안의 상호작용. `:same_tub` / `:diff_tub` 로 **프레임 내 / 프레임 간**을 가른다 |
| `mask→ctx` | 미래 토큰이 문맥을 읽는다 — **정보를 가져오는 경로** |
| `mask→mask` | 미래 토큰끼리 맞춘다 — **미래를 서로 정합시키는 경로** |
| `ctx→mask` | 문맥이 미래 토큰을 읽는다 (bidirectional 이라 존재한다) |

### 명세 문법

```
<q>-><k>[:<rel>]@<layers>

  q, k    ctx | mask | all | t<a>-<b>      튜블릿 범위(양끝 포함). t7-7 = 문맥 마지막
  rel     any(기본) | same_tub | diff_tub | same_patch | diff_patch
  layers  all | L<i> | L<i>-<j> | L<i>,<j>,...
```

⚠️ **`..` 가 `->` 의 셸 안전 별칭이다.** 따옴표 없이 `--edge mask->ctx` 를 치면 bash 가
`->ctx` 를 리다이렉트로 먹어 `mask-` 만 도착한다 (실제로 당했다). `--edge mask..ctx` 를 쓸 것.

```bash
mask..ctx@L0                 미래 토큰이 층 0 에서 문맥을 못 보게 한다
ctx..ctx:diff_tub@all        문맥의 프레임 간 상호작용만 전 층에서 끊는다 (프레임 내는 남긴다)
mask..mask@L6-11             후반 층에서 미래끼리의 상호작용을 끊는다
t7-7..all@L3                 문맥 마지막 튜블릿이 층 3 에서 아무것도 못 보게 한다
```

`--edge` + `--layers {each|all|prefix|suffix}` 로 스윕을 자동 생성하고,
`--spec` / `--specs-file` 로 개별 명세를 얹는다. 셋을 섞어 써도 된다.

---

## 2. 왜 KV cache 가 아닌가 (그리고 무엇이 실제로 빠른가)

predictor 는 causal decoder 가 아니라 **bidirectional** transformer 다. 층 ℓ 에서 간선을
끊으면 ℓ 의 출력이 바뀌고, 따라서 **ℓ 위의 모든 층의 k·v 가 전부 바뀐다.** 재사용 가능한
것은 ℓ 아래뿐인데, 그 층들은 k·v 를 캐싱할 게 아니라 **아예 다시 돌 필요가 없다.**

그래서 캐싱하는 것은 KV 가 아니라 **층별 residual stream** `x_ℓ` 이고, 이것이 KV 캐시를
엄격히 지배한다. clean forward 한 번으로 `x_0..x_{L−1}` 을 받아 두고, 명세 s 는
`min(s.layers)` 부터 이어 돈다 → L 층 스윕이 `L×L` 대신 `L(L+1)/2` 층 계산.

**그런데 진짜 큰 절약은 그 위에 있다 — 인코더다.**

```
FLOP/clip   문맥 인코더 2.1T + target 인코더 4.6T  =  6.7 TFLOP   <- knockout 과 무관
            predictor 1회                          =  0.27 TFLOP  <- 명세마다
```

인코더 출력 `z` 와 `h` 는 knockout 이 건드리지 않으므로 **배치당 한 번만** 계산해 모든
명세가 공유한다. 그래서 명세 수십 개짜리 스윕이 채점 한 번의 두 배 안쪽에서 끝난다.

`--no-prefix-cache` 로 prefix 캐시를 끄고 대조할 수 있다 (수치는 같아야 하고 느려진다).

---

## 3. 검증 (실측)

| 항목 | 값 | 뜻 |
|---|---:|---|
| `resume(x_0, ℓ=0, mask=None)` vs 실제 `predictor.forward` | **0.000e+00** | prefix 재개가 **비트 단위로 동일**하다 |
| 전부 True 인 마스크 vs 실제 forward | 9.77e-03 (예측 최대크기 8.81 의 0.11%) | `attn_mask` 를 주면 SDPA 가 다른 커널로 간다 |

두 번째 값 때문에 **Δ 의 기준선은 `clean` 이 아니라 `clean_null`**(전부 True 인 마스크로
한 번 더 돈 판)이 정본이다. 그래야 커널 차이가 기준선에 흡수된다. `results.json` 에는
`delta`(= vs `clean_null`)와 `delta_vs_clean` 이 **둘 다** 들어간다. `--no-null-baseline` 로 끌 수 있다.

첫 배치에서 자동으로 재고 `results.json.verify` 에 남는다. `plot.py` 는 첫 값이 0 이 아니면 죽는다.

---

## 4. 무엇을 성능으로 잴 것인가 (모듈)

`metrics.py` 의 레지스트리다. `--metrics surprise_acc,pred_drift` 처럼 콤마로 여러 개.

| 이름 | 무엇 | target 인코더 |
|---|---|---|
| `surprise_acc` | **확립된 채점.** matched pair, 동점 0.5, block 평균. chance 50 | 필요 |
| `surprise_l1` | 채점 전 원값 `mean|p−h|`. 정확도와 따로 움직일 수 있다 | 필요 |
| `pred_drift` | `|p−p₀|/|p₀|` 와 `cos(p,p₀)`. **정확도가 안 변해도 여기서는 보인다** | 불필요 |

- `surprise_acc` 는 `evals.world_model_analysis.eval.score_blocks` 를 **그대로 재사용**한다.
  채점 정의가 본 파이프라인과 갈리지 않게 하려는 것이다 — 여기서 다시 구현하지 말 것
- **`needs_h=False` 인 지표만 요청하면 target 인코더를 건너뛴다** (forward 가 절반)
- 새 지표는 `@register("이름")` 클래스를 하나 더 쓰면 끝이다

---

## 5. 돌리는 법

```bash
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2

# 층별 스윕 (간선 5종 x 12층 = 60 명세) + 전 층 동시 5개
PYTHONPATH=. python -m analysis.attention.knockout.run \
    --dataset v11 --model vith --blocks-per-cond 24 \
    --edge mask..ctx --edge mask..mask --edge ctx..ctx --edge ctx..mask \
    --edge ctx..ctx:diff_tub --layers each \
    --spec mask..ctx@all --spec mask..mask@all --spec ctx..ctx@all \
    --metrics surprise_acc,surprise_l1,pred_drift

# ViT-L 은 인자 하나만 바꾼다
PYTHONPATH=. python -m analysis.attention.knockout.run --dataset v11 --model vitl ...

# 그림
PYTHONPATH=. python -m analysis.attention.knockout.plot \
    --results z_research/IntPhysGenV11/exp_results/knockout__v11_vith/results.json \
    --outdir  z_research/IntPhysGenV11/figures/knockout
```

경로·컬럼·체크포인트는 `resolve.py` 를 그대로 불러 `datasets.md`/`models.md` 에서 온다.
`--set` 도 `run.sh` 의 `SET=` 과 같다. 출력 자리도 하네스 관례를 따른다:
`<results_root>/knockout__<데이터셋>_<모델>/results.json`.

주요 인자: `--blocks-per-cond`(조건당 block 수, **0 = 전부**), `--seed`, `--batch-size`, `--workers`,
`--no-keep-self`, `--no-null-baseline`, `--no-prefix-cache`, `--device`, `-o`, **`--shard I/N`**.

### GPU 여러 장 — `--shard` + `merge.py` (2026-09-03)

`run.py` 는 프로세스 하나 = GPU 하나다. 전수를 돌리려면 block 을 N 조각 내 N 프로세스로 돌리고 합친다:

```bash
bash analysis/attention/knockout/run_sharded.sh        # D=v11_full M=vith N=8 BPC=0 W=7 기본
# 안에서 하는 일:
#   run.py --shard i/N -o <outdir>   x N    ->  <outdir>/shards/shardXXofNN.{json,npz}
#   merge.py -o <outdir>                    ->  <outdir>/results.json + per_video.npz
```

- `--shard` 는 **정렬된 block_id 의 `j % N == i`** 를 남긴다. block 이 갈라지지 않으므로 matched pair 가 안 깨진다
- `run.py` 는 이제 **클립별 원값**을 npz 로 같이 남긴다 (`surprise`, `drift_rel`, `drift_cos`;
  행 = `clean, clean_null, 명세…`, 열 = `ds.records` 순). shard 가 아니어도 `per_video.npz` 가 생긴다
- `merge.py` 는 (1) shard 가 다 있는지·명세/config 가 같은지·`resume_vs_forward == 0` 인지 검사하고,
  (2) `score_blocks` 로 **다시 채점**해 run.py 단일 실행과 같은 `results.json` 을 만들며,
  (3) **`breakdown`** 을 덧붙인다 — `k / timing / violation / motion / condition` 과 그 교차마다
  `{n_block, n_pair, acc{명세: %}}`. 축 값은 인덱스 행에서 파생한다 (`k` 는 file_name 의 `_k{n}_`,
  `timing` 은 condition 접미사). 그림은 이걸 block 수 가중으로 다시 합쳐 쓴다
- 실측: v11_full 43,008 clip, 명세 39개(폭 7 창 12 × 3간선 + `@all` 3), 8 GPU, batch 4 → **37분**
  (410 ms/clip/GPU). 인코더 : predictor 가 이 명세 수에서는 대략 1 : 1 이라 캐시로 인코더를 빼도 절반이다
- SLURM: `sbatch --export=ALL,N=8,W=3,LAYERS_B64=…,EDGES_B64=…,OUT=… analysis/attention/knockout/sbatch_knockout.sh`.
  콤마·공백이 든 값은 `*_B64` 로 싣는다 (`--export` 는 콤마가 구분자다). 스스로 제출하지 않는 평범한 batch 스크립트다

### ⚠️ 토큰 캐시로 인코더를 대신할 수 없다 (2026-09-03 실측)

`--h-from-cache DIR` 가 있지만 **v11 계열 캐시에는 쓰지 말 것.** 두 가지 이유다:

| base | 왜 못 쓰나 |
|---|---|
| `ctx_masked` | **affine-free LN 이 걸린 z** 다 (`occlusion_identity/forward.py: _ln(z)`). predictor 입력은 LN 전의 z 이고 토큰별 평균·스케일이 지워져 복원이 안 된다 |
| `target` | LN 은 맞지만 **probing 관례(`dtype: bfloat16`)로 뽑은 h** 다. surprise 관례(fp32 + fp16 autocast)의 h 와 토큰 최대 13.9 차이, 클립 surprise 최대 **0.0010** 차이 (colour margin 0.0016 의 60%). CLAUDE.md §1-7 "관례를 섞지 말 것" |

`--h-from-cache` 를 켜면 첫 배치에서 `verify.cache_h_vs_forward_max_abs_diff` / `cache_h_surprise_max_abs_diff` 를
남긴다. surprise 관례로 뽑은 캐시가 생기면 그때 이 값으로 다시 판단한다.

### 클립별 키 집합 — `hidden` / `hidden_ctrl` (2026-09-03)

"mask 토큰이 **물체가 가려진 문맥 프레임**을 못 읽으면 채점이 오르는가" 를 재는 명세다.

```
mask..hidden@all        mask 토큰이 가려진 문맥 튜블릿을 못 읽는다
mask..hidden_ctrl@all   대조군: 같은 개수의 이웃 튜블릿 (바로 앞, 자리가 없으면 바로 뒤)
```

- 집합은 인덱스(`--hidden-csv`, 기본 `<data.root>/index_probe.csv`)의 `hidden_start` + `n_context_hidden` 에서
  온다. raw 프레임 → 샘플 프레임(`frames_stride`) → 튜블릿(`tubelet_size`). **튜블릿 단위**라 경계에서
  보이는 프레임 하나가 같이 끊긴다 (late k=1: raw 45 만 가려졌는데 튜블릿 7 = raw 42+45 가 끊긴다)
- v11_full 매핑 (실행 로그에 표로 찍힌다): late k=1,2 → {7} / k=3,4 → {6,7}; early k=1,2 → {2} / k=3,4 → {2,3};
  mid k=1 → {3} / k=2,3 → {3,4} / k=4 → {3,4,5}. visible 은 빈 집합 = 그 클립에서는 no-op
- `--cond-re occlusion` 으로 가림 조건만 돌린다 (visible 은 no-op 라 시간만 쓴다)
- 결과 표는 (조건, 대조군) 을 같이 봐야 한다 — hidden 이 오르고 ctrl 이 안 오를 때만 "가려진 프레임" 이 원인이다

---

## 6. 단서 — 읽기 전에

- ⚠️ **표본은 block 단위다.** 채점이 block 안의 matched pair 비교라 클립을 흩어 뽑으면
  쌍이 깨진다. `clean` 정확도는 그 표본에서의 값이지 **전수 73.37% 가 아니다.**
  비교는 언제나 같은 표본의 기준선 대비 Δ 로 읽을 것
- ⚠️ **`--keep-self` (기본 켬)** 는 자기 자신으로의 attention 을 항상 남긴다. 안 그러면
  `mask→ctx` 와 `mask→mask` 를 동시에 끊었을 때 mask 토큰의 키가 하나도 안 남아 softmax 가
  NaN 이 된다. 해석에 영향을 주는 설계 결정이라 `results.json.meta.keep_self` 에 기록된다
- ⚠️ **정확도는 계단 함수다.** `n_pair` 가 작으면 Δ 의 분해능이 `1/n_pair` 다 (24쌍이면
  4.17pp). **Δ=0 이 "아무 일도 없었다" 를 뜻하지 않는다** — 실측으로 `ctx→ctx@all` 은
  Δacc 0.00 인데 예측은 `rel_l1 0.129 / cos 0.993` 만큼 움직였다.
  그래서 `pred_drift` 를 같이 켜는 것을 기본으로 삼을 것
- ⚠️ **knockout 은 개입이지 관찰이 아니다.** 간선을 끊은 predictor 는 학습 때 본 적 없는
  조건에서 돌고 있다. "그 경로가 없으면 성능이 이만큼 떨어진다" 는 말할 수 있지만
  **"그 경로가 정보를 이만큼 나른다" 는 말할 수 없다** (분포 밖 효과와 안 갈린다)
- ⚠️ **간선 종류마다 끊기는 양이 다르다.** `mask→ctx` 는 전체 쌍의 25%, `ctx→ctx:same_tub`
  는 3.1% 다. Δ 를 나란히 놓을 때 이걸 같이 볼 것 (`results.json` 에 `q/k/rel` 이 그대로 있다)

---

## 7. 파일

| 파일 | 무엇 |
|---|---|
| `specs.py` | 명세 문법 파싱 + (N,N) bool 마스크 빌더 + 스윕 생성 |
| `runner.py` | prefix 캐시 capture / resume, autocast 규약 |
| `metrics.py` | 지표 레지스트리. `surprise_acc` 는 `score_blocks` 재사용 |
| `run.py` | 진입점. 인코더 1회 공유 + 명세 루프 → `results.json` |
| `plot.py` | 그림 3종. `verify` 를 다시 확인하고 그린다 |
| `merge.py` | shard 병합 + `score_blocks` 재채점 + 축별 `breakdown` |
| `run_sharded.sh` | GPU N 장에 `--shard` 로 나눠 돌리고 `merge.py` 까지. 환경변수 `D M N BPC LAYERS W EDGES SPECS COND_RE OUT` |
| `sbatch_knockout.sh` | 위를 SLURM job 으로 (vll5, gpu:8). 콤마·공백 값은 `*_B64` |

축별 그림(v11_full)은 `z_research/scripts/figures/plot_v11_knockout.py` 가 `breakdown` 에서 그린다.

토큰 격자(`Grid`)는 `../predictor_attn/geometry.py` 를 그대로 쓴다 — 두 도구가 같은 좌표계다.

---

## 재현

```bash
PYTHONPATH=. python -m analysis.attention.knockout.run \
    --dataset v11 --model vith --blocks-per-cond 24 \
    --edge mask..ctx --edge mask..mask --edge ctx..ctx --edge ctx..mask \
    --edge ctx..ctx:diff_tub --layers each \
    --metrics surprise_acc,surprise_l1,pred_drift

PYTHONPATH=. python -m analysis.attention.knockout.plot \
    --results z_research/IntPhysGenV11/exp_results/knockout__v11_vith/results.json \
    --outdir  z_research/IntPhysGenV11/figures/knockout
```

수치는 문서가 아니라 `results.json` 에서 다시 확인한다.
`results.json.verify.resume_vs_forward_max_abs_diff` 가 `0.0` 이어야 그 실행이 유효하다.
