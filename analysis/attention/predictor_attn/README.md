# predictor attention — 어떤 프레임을 보고, 얼마나 멀리 보는가

> 레포 규칙은 루트 `CLAUDE.md`, 실행 계약은 `configs/protocols/README.md`,
> 본 실험 세트는 `z_research/IntPhysGenV11/README.md`.

**한 줄: predictor 의 self-attention 을 층 × 헤드 × 질의 프레임 단위로 접어,
"미래 토큰이 어느 프레임을 attend 하는가"와 "attention 이 얼마나 멀리 뻗는가"를 잰다.**

frozen 모델 위에서 돌고 재학습이 없다. GPU 는 문맥 인코더 forward 때문에 필요하다(1장).

---

## 1. 왜 재계산하는가 — flash 는 원리적으로 못 빼낸다

predictor 의 각 블록은 `F.scaled_dot_product_attention` 을 쓴다
(`src/models/utils/modules.py` `RoPEAttention.forward`). SDPA/flash 는 **softmax 를 타일
안에서 융합해 attention 행렬을 메모리에 만들지 않는다.** 그게 속도·메모리 이득의 원천이고,
그래서 hook 으로 꺼낼 attention weight 가 애초에 존재하지 않는다.
`sdpa_kernel(MATH)` 로 backend 를 바꿔도 내부 텐서를 노출하지 않는다.

그래서 이 도구는 **실제 forward 를 flash 그대로 두고**(모델 출력 불변),
`forward_pre_hook` 으로 층 입력과 RoPE 위치만 받아 q·k 를 다시 만들고 **query chunk
단위로** attention 을 재계산해 그 자리에서 집계로 접는다. 전체 행렬은 어디에도 저장하지 않는다.

- RoPE 는 q·k 에 **적용된 뒤** 잡아야 한다. `hooks.py:_rope` 가 `RoPEAttention.forward`
  의 회전 구간을 그대로 옮긴 것이고, 입력 mask 텐서도 모듈이 받은 바로 그 텐서다
- 비용: predictor 는 384-dim / 12층 / 12헤드라 재계산이 싸다. 비싼 ViT-H 인코더는 안 건드린다
- 실측 **~300 ms/clip** (ViT-H fp32+fp16autocast, 4090 1장, PNG 디코드 포함)

### 검증 (실측)

| 항목 | 값 | 뜻 |
|---|---:|---|
| `mass` 행 합 오차 | **1.19e-07** | softmax 를 튜블릿으로 접은 것이 맞다 |
| 재계산 `attn@v` vs 실제 모듈 출력, **fp32** | **1.76e-05** | 재계산이 정확하다 (SDPA 누산 순서 잡음) |
| 같은 것, **fp16 autocast** (기본 채점 관례) | 4~6e-03 | 위 fp32 값과 대조하면 **fp16 반올림이지 우리 오류가 아니다** |

fp16 값은 어느 클립이 걸리느냐에 따라 조금씩 다르다 (실측 3.91e-03 / 5.86e-03).
크기 감각을 위해 `summary.json.verify` 에 `max_abs_out` 을 같이 남긴다 —
3.91e-03 은 출력 최대크기 3.20 의 **0.12%** 다.

두 값이 함께 있어야 의미가 있다. fp32 판은 이렇게 재현한다:

```bash
python -m analysis.attention.predictor_attn.extract --dataset v11 --model vith \
    --per-group 1 --set model.autocast=null
```

---

## 2. 무엇이 나오나

전부 **층 × 헤드 × 질의 튜블릿** 단위이고, 클립과 공간 위치에 대해 평균한 값이다.

| 배열 | 모양 | 무엇 |
|---|---|---|
| `mass` | (G,L,H,T,T) | 질의 튜블릿이 각 **키 튜블릿**에 준 attention 질량. **행 합 = 1** |
| `same` | (G,L,H,T,T) | 그 중 **같은 공간 패치**에만 준 질량. 균등 기대값 = `mass/S` |
| `sdist_px` | (G,L,H,T) | 공간 attention distance (픽셀). 고전적 ViT mean attention distance |
| `tdist_frames` | (G,L,H,T) | 시간 attention distance `|Δt|` (샘플 프레임) |
| `toff_frames` | (G,L,H,T) | **부호 있는** `Δt`. 음수 = 과거(문맥) 쪽을 본다 |
| `entropy` | (G,L,H,T) | attention 엔트로피 (nats). 균등 상한 = `log(N)` = 8.32 |
| `n_query`, `n_clips` | (G,T), (G,) | 정규화에 쓴 개수 |

`G` = 그룹(기본 `condition` 6개), `L`=12, `H`=12, `T`=16 튜블릿, `S`=256.

### 단위 — 세 가지가 다르다

```
튜블릿 1  =  샘플 프레임 2장  =  원본 프레임 6장   (frames_stride = 3)
```
표·그림의 시간 값은 전부 **샘플 프레임**이다 (프로토콜이 읽는 32장 기준).
공간 거리는 패치 단위로 재고 `patch_size(16)` 을 곱해 픽셀로 낸다.

---

## 3. 읽는 법과 단서

- **질의 튜블릿 0..7 은 진짜 문맥 토큰, 8..15 는 mask token 이다.** 성격이 다르므로
  한 축에 섞어 읽지 말 것. 그림은 경계선으로 갈라 뒀다
- **`same` 은 배율로 읽는다.** `same.sum(키축) × S` 가 1 이면 공간적으로 균등, 40 이면
  같은 패치에 40배 몰려 있다는 뜻이다
- ⚠️ **문맥일치 쌍은 predictor 입력이 비트 단위로 같다** (`CLAUDE.md` §1-2). 즉
  `pos_a` 와 `imp_ab` 의 attention 은 **정확히 동일**하다. 기본 `--variants pos_a pos_b`
  가 고유 문맥만 남긴다. `--variants all` 로 끄면 같은 값을 두 번 세는 것이다
- ⚠️ **위반 종류(vanish/shape/color)로는 쪼갤 수 없다.** 같은 이유다 — 위반은 미래에만
  있고 predictor 는 문맥만 본다. `--group-by` 에 `violation_type` 을 넣지 말 것
- ⚠️ **표본 추출이다.** `--per-group` 을 두 배로 올려 값이 안 움직이는지 확인하고 쓸 것
- ⚠️ **토큰 캐시를 못 쓴다.** 캐시의 `ctx_masked.npy` 는 `_ln(z)` 인데 predictor 가 학습
  때 받은 것은 **LN 이전의 z** 다 (`occlusion_identity/forward.py:157-163`). 그래서 문맥
  인코더를 매번 실제로 돌린다. 대신 target 인코더는 부르지 않는다 (채점이 아니므로 불필요)

### 무엇을 주장할 수 있고 없는가

이 도구는 **predictor 의 계산을 기술**한다. 그 자체로 논문 beat 를 세우지는 못한다.

✅ "predictor 의 층 ℓ 에서 미래 토큰이 문맥의 어느 프레임에 attention 질량을 얼마나 준다"
✅ "attention 의 공간·시간 폭이 층을 따라 어떻게 변한다"
✅ 조건(가림/비가림, 등속/등가속) 간 그 값의 차이

❌ **"predictor 가 정보를 어디서 가져온다"** — attention 질량은 정보 흐름이 아니다
   (value 의 크기와 residual stream 을 무시한다)
❌ **"표현이 회전했다 / 정렬이 깨졌다"** — beat 5 의 주장은 표현 기하로 해야 한다
   (`PAPER_STORY_2026-08-31.md` 금지 표현 표)
❌ **채점 실패와의 인과** — attention 차이와 채점 정확도의 상관은 상관일 뿐이다

---

## 4. 돌리는 법

```bash
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2

# ViT-H, v11, 조건 6개 x 32클립
PYTHONPATH=. python -m analysis.attention.predictor_attn.extract \
    --dataset v11 --model vith --group-by condition --per-group 32 --device cuda:0

# ViT-L 은 인자 하나만 바꾼다 (predictor 는 H·L 이 완전히 같고 인코더만 1280 -> 1024)
PYTHONPATH=. python -m analysis.attention.predictor_attn.extract \
    --dataset v11 --model vitl --group-by condition --per-group 32

# k(가림 길이)까지 쪼개기 / 다른 셋
PYTHONPATH=. python -m analysis.attention.predictor_attn.extract \
    --dataset v11 --group-by condition,sym_k --per-group 16 \
    --set data.index_csv=index_probe.csv          # sym_k 는 index_probe.csv 에 있다

# 그림
PYTHONPATH=. python -m analysis.attention.predictor_attn.plot \
    --npz z_research/IntPhysGenV11/exp_results/predictor_attn__v11_vith/attn.npz \
    --outdir z_research/IntPhysGenV11/figures/attention
```

경로·컬럼·체크포인트는 전부 `configs/protocols/{datasets,models}.md` 에서 온다 —
`z_research/scripts/harness/resolve.py` 를 그대로 부르므로 **데이터셋을 새로 붙일 때
고칠 곳은 여전히 `datasets.md` 섹션 하나뿐이다.** `--set` 도 `run.sh` 의 `SET=` 과 같다.
출력 자리도 하네스 관례를 따른다:

```
<results_root>/predictor_attn__<데이터셋>_<모델>/attn.npz + summary.json
```

주요 인자: `--protocol`(기본 `surprise_c16t32` = 채점 경로와 같은 프레임 배치·dtype),
`--per-group`, `--seed`, `--batch-size`, `--q-chunk`(S 의 배수), `--workers`, `--no-verify`.

### 그림 폴더

`plot.py` 상단 `FIGDIR` 표가 배정한다. **표에 없는 이름은 최상위에 떨어진다** —
새 그림이 눈에 띄라고 일부러 그렇게 뒀다.

| 폴더 | 그림 |
|---|---|
| `01_layerwise/` | `fig_attn_matrix` (층별 질의×키 프레임), `fig_frame_profile` (어떤 프레임을 보나) |
| `02_distance/` | `fig_attn_distance` (공간 px · 시간 frame · 엔트로피) |
| `03_condition/` | `fig_attn_group` (조건별 비교) |

---

## 5. 파일

| 파일 | 무엇 |
|---|---|
| `geometry.py` | 토큰 인덱스 ↔ (튜블릿, h, w), 공간 거리 행렬. 모델의 RoPE 좌표계와 같은 분해 |
| `hooks.py` | `AttnRecorder` — pre-hook 으로 q·k 를 받아 chunk 재계산 + 집계 + 검증 |
| `extract.py` | 진입점. resolve → 모델·데이터 → 집계 → `attn.npz` + `summary.json` |
| `plot.py` | 그림 4종. npz 를 다시 검증하고 그린다 |

`analysis/` 아래에 둔 이유: 이 트리는 `python -m analysis.<name>` 으로 직접 부르는
사후 분석용이고 `evals/main.py` 의 `eval_name` 디스패치를 타지 않는다 (`analysis/__init__.py`).
`z_research/scripts/analysis/` 는 **산출물·토큰 캐시만 읽는 GPU 불필요 스크립트** 자리다.
이 도구는 모델 forward 가 필요하므로 `analysis/` 가 맞다.

---

## 6. 아직 안 한 것

- **인코더(ViT-H/L) attention** — 같은 방식으로 되지만 4096토큰 × 32층 × 16헤드라 비용이
  훨씬 크다. 층을 골라서 해야 한다
- **value-weighted 흐름** — attention 질량 대신 `‖attn·v‖` 로 재는 판. 위 ❌ 첫 항목을 일부 갚는다
- **채점과의 대조** — `report.json` 의 조건별 정확도와 붙여 보는 것. 상관까지만 가능

---

## 재현

```bash
# 1. 추출 (ViT-H, 조건 6 x 32클립)
PYTHONPATH=. python -m analysis.attention.predictor_attn.extract \
    --dataset v11 --model vith --group-by condition --per-group 32

# 2. 검증 대조판 (fp32 에서 재계산 오차가 1e-5 대로 떨어지는지)
PYTHONPATH=. python -m analysis.attention.predictor_attn.extract \
    --dataset v11 --model vith --per-group 1 --set model.autocast=null -o /tmp/attn_fp32

# 3. 그림
PYTHONPATH=. python -m analysis.attention.predictor_attn.plot \
    --npz z_research/IntPhysGenV11/exp_results/predictor_attn__v11_vith/attn.npz \
    --outdir z_research/IntPhysGenV11/figures/attention
```

수치는 문서가 아니라 `summary.json` / `attn.npz` 에서 다시 계산해 확인한다.
`plot.py` 는 실행할 때마다 `mass` 행 합을 재검사하고 (1e-3 넘으면 죽는다)
`summary.json.verify` 를 같이 찍는다.
