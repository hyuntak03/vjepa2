# RollOut V1 — 등속 직선 운동에서 predictor 는 물체를 앞으로 옮기는가

> 레포 규칙은 루트 `CLAUDE.md`, 실행 계약은 `configs/protocols/README.md`.
> 논문 뼈대는 **[`../IntPhysGenV11/Archive/PAPER_STORY_2026-09-06.md`](../IntPhysGenV11/Archive/PAPER_STORY_2026-09-06.md)**.
> 이 세트는 그 문서의 **beat 4 (a)** 하나를 위해 만들었다.

---

## 한 줄

**미래 토큰에서 물체 위치를 튜블릿별로 읽었을 때, 그 기울기가 실제 속도인가.**

```
x̂(t) = a + b·t   (t = 0..7 튜블릿)

b̂ ≈ v   →  이어간다
b̂ ≈ 0   →  조회기. 마지막 관측 위치를 8칸에 들고 있을 뿐이다
```

---

## 1. 왜 이 세트인가

v11 의 위반 3종(vanish / shape / color)은 **전부 단일 프레임 외형 차이**라
predictor 가 상태를 앞으로 나르지 않아도 풀린다. 그래서 "이어가는가" 를 못 묻는다.

RollOut 은 **물리 법칙이 하나뿐**이다 — 평지 등속. 미래는 문맥에서 완전히 결정되고,
필요한 것은 **가장 단순한 외삽**(같은 속도로 계속) 하나다.
여기서 실패하면 더 어려운 어떤 것도 물을 필요가 없다. **바닥 시험**이다.

⚠️ 이 세트는 **보간 vs 외삽을 가르지 않는다.** 등속 운동에서는 관측 신호의 선형 연장이
곧 정답이다. 그건 다음 층(원운동 / 낙하, `PAPER_STORY_2026-09-06.md` §6)의 질문이고,
여기서 묻는 것은 그보다 앞이다 — **애초에 앞으로 옮기기는 하는가.**

---

## 2. 설계에서 중요한 두 가지

### (a) 속도 ⊥ anchor — "마지막 프레임 복사" 가 데이터 수준에서 차단된다

`flat_anchor_x_cm` 은 **문맥 마지막 프레임(45)에서의 위치**다
(실측: `x_context_end_cm == flat_anchor_x_cm`, 7392/7392).
속도 11레벨 × anchor 3레벨이 **33셀 정확히 균등(224씩)** 이라 두 축이 직교한다.

> **같은 마지막 관측 위치에서 11개의 서로 다른 미래가 나온다.**

문맥 끝을 복사하는 모델은 anchor 는 맞히지만 `b̂ = 0` 이 된다. 원리적으로 통과할 수 없다.

### (b) `v = 0` 이 별도 조건이 아니라 속도 레벨 하나다

회귀 안에 **대조군이 내장**된다. 프로브가 무조건 진전을 만들어내면 `v=0` 에서 들킨다.

---

## 3. 데이터

```
7,392 clip / 7,392 block          위반 없음 · 쌍 없음 (클립 하나 = block 하나)
  speed    -160 -135 -110 -85 -60  0  60 85 110 135 160  (cm/s, 부호 포함)
  anchor   -100 / 0 / +100 (cm)  = 문맥 마지막 위치
  condition  roll_v{속도}_a{anchor} — 33종, 셀당 224
  shape 7종 × 1056 · color 8종 × 924 · env 4종 × 1848
  288px PNG 34장 (0,3,…,99) → 프로토콜은 32장(0…93) 읽고 256 으로 리사이즈
```

원본 `/data2/local_datasets/world/world_analysis/RollOut_v1` (27 GB, **vll6 로컬**).
레지스트리 `configs/protocols/datasets.md` 의 `## rollout_v1`.

### 위치 라벨 — 해석식, 픽셀 측정 없음

```
x_cm(raw f) = flat_anchor_x_cm + flat_v_cm_s · (f − 45) / 16
x_norm      = x_cm / 769.0909            # frame_half_width_cm
튜블릿 t 의 라벨 = x_norm 의 두 프레임 평균  (f = 48+6t, 51+6t)
```

`obj_depth_cm`(1410) · 카메라 · fov 가 전부 상수라 cm → 정규화 좌표가 **선형**이다.
검증: `build_rollout_index.py` 가 이 공식으로 metadata 의 `x_predict_first/last_cm` 을
**`max|Δ| = 0`** 으로 재현한다.

⚠️ **GT 속도는 프로브 입력에 절대 안 들어간다.** 라벨을 만들 때와 마지막에 `b̂` 와
비교할 때만 쓴다. `p` 에서 속도를 직접 회귀하면 안 된다 — `v` 는 문맥에 이미 있고
`p = predictor(context)` 라 **문맥을 복사만 해도 통과**한다 (`SYNTHESIS §3-4` 의 문맥 잔상 함정).

---

## 4. 읽는 방법 — 공간평균 + 최소제곱

`p` 는 (2048, 1280) = **8 튜블릿 × 256 공간토큰**. 튜블릿 t 는 raw {48+6t, 51+6t} 를 덮는다.

```
튜블릿 t 의 256 토큰 평균  →  1280차원  →  최소제곱  →  x̂_t
x̂(t) = a + b·t             →  b̂ 를 v 와 비교
```

닫힌 해라 epoch·lr·수렴이 없고 결정론적이다. **표준화는 하지 않는다** — OLS 는 특징 스케일에
불변이고(가역 아핀 재매개화), 이식에서는 오히려 "좌표가 다르다" 와 "스케일이 다르다" 를 섞는다.
GPU 불필요 — 비용은 캐시 72 GiB 를 한 번 읽어 풀링하는 I/O 다.

⚠️ **프로브를 8칸 전부에 공유하는 것이 이 측정의 유일한 방어선이다.**
`p_t` 가 t 에 따라 안 변하면(문맥을 복사해 8칸에 넣었으면) 공유 프로브는 모든 칸에 같은 값을
내고 **`b̂ = 0`** 이 된다. 칸별로 따로 학습하면 이 성질이 깨진다.

> **폐기한 시도 (기록)** — 처음에 격자 위 soft-argmax 로 "어느 토큰이 물체인가" 를 배우게 했다.
> `α` 를 프레임에 겹쳐 보니 **attention 이 물체 위에 전혀 없고 배경에 흩뿌려져 있는데도
> `x̂` 는 정확했다.** ViT 토큰은 local patch 가 아니라 전역 문맥을 담은 표현이라 배경 토큰도
> 물체 위치를 알고 있기 때문이다. 즉 "물체를 찾는다" 는 전제가 애초에 틀렸다.
> **그런데 그건 이 측정의 결함이 아니다** — 묻는 것은 "물체 토큰을 특정할 수 있는가" 가 아니라
> **"`p_t` 가 t 시점의 위치를 담고 있는가"** 이고, 정보가 분산 저장돼 있어도 답은 같다.
> 그래서 더 단순한 공간평균으로 돌아왔다. **soft-argmax 로 다시 가지 말 것.**

### 두 칸만 낸다

| | 무엇 | 실측 |
|---|---|---|
| **`h→h`** | 도구 대조군(천장). 실제 미래를 본 인코더를 같은 자로 읽은 값 | 위치 0.952 / 속도 0.988 |
| **`p→p`** | **핵심.** `p` 자체 basis 에서 진전이 읽히는가 | 위치 0.911 / 속도 0.968 |

⚠️ **이식(`h→p` / `p→h`)은 내지 않는다.** "같은 좌표계를 쓰는가" 라는 **다른 질문**이고,
이 세트의 주장이 아니다. 스크립트도 self 두 칸만 낸다.

### 대조군 목록

| 대조군 | 기대 | 배제하는 것 | 실측 |
|---|---|---|---:|
| `h→h` ① | `b̂ = v` | 도구 무력 | β +0.987 |
| **튜블릿 순서 셔플** | **β = 0** | 위치가 슬롯에 안 묶여 있음 | **−0.0002** |
| **8칸 동일 (복사기 모사)** | **β = 0** | 프로브가 기울기를 만들어냄 | **+0.0000** |
| **라벨 클립간 셔플** | **β = 0** | 우연한 상관 | **+0.0012** |
| `v=0` 셀 | `b̂ = 0` | 무조건 진전 예측 | +0.1 (h) / −1.7 (p) |
| speed ⊥ anchor | (설계) | 마지막 프레임 복사 | — |

⚠️ **`(1,D)` 로 전부 평균해 속도를 직접 회귀하면 안 된다.** 8칸을 인위적으로 동일하게 만들어도
R² 가 0.9931 그대로 나온다 — `v` 는 문맥에 이미 있어 **복사기가 통과한다**.
반드시 **칸별 위치 → 기울기** 로 가야 한다 (`Archive/POSITION_READOUT_2026-09-06.md` §4).

---

## 5. 격자 해상도 — 결과 해석 전에 먼저 볼 것

```
패치 1칸 = 16px (256/16)        물체 겉보기 ≈ 31.6px ≈ 패치 2칸
예측 구간 이동 = v × 2.8125 s
   v=160 → 450cm → 0.585 정규화 → 4.7 격자칸   (튜블릿당 0.59칸)
   v= 60 → 169cm → 0.219        → 1.76 격자칸  (튜블릿당 0.22칸)
   v=  0 → 0칸
```

**저속 셀은 튜블릿당 0.22칸**이라 패치 1/4 수준이다. 걱정했지만 **기우였다** —
`h→h` 가 v=±60 에서 b̂/v = 1.01 로 정확히 잡는다 (11단계 전부 0.95~1.02).
readout 이 연속값이라 패치보다 잘게 읽는다. **저속 셀도 쓸 수 있다.**

⚠️ 다만 **양 끝(±160)은 `h` 도 0.95~0.97 로 눌린다.** 물체가 ±0.754 까지 가므로
격자 경계 효과가 섞였을 수 있고, 아직 안 갈렸다.

---

## 6. 결과

| 문서 | 무엇 | 언제 |
|---|---|---|
| **[`Archive/POSITION_READOUT_2026-09-06.md`](Archive/POSITION_READOUT_2026-09-06.md)** | **결과** — 수치·null·말할 수 있는 것 | 수치를 찾을 때 |
| **[`Archive/METHOD_POSITION_READOUT_2026-09-06.md`](Archive/METHOD_POSITION_READOUT_2026-09-06.md)** | **방법 (이식용)** — 절차·전제·함정 11개·**v11 이식 시 바뀌는 것** | 같은 측정을 다른 세트에 옮길 때 |

전수 7,392 clip.

| | 위치 R² (클립내부) | 속도 R² |
|---|---:|---:|
| `h→h` (도구 천장) | 0.952 | 0.988 |
| `p→p` (핵심) | **0.911** | **0.968** |

**predictor 는 복사기가 아니다** — null 3종(튜블릿 셔플 −0.0002 / 8칸 동일 +0.0000 /
라벨 셔플 +0.0012)이 전부 0 으로 죽는다. 속도의 방향은 11단계 전부 정확하고 `v=0` 을 정지로 읽는다.
⚠️ 단 **등속이라 문맥 끝에서 1차 외삽하면 되는 조건**이라, "상태를 통합해 나른다" 는 못 쓴다.

## 7. 현재 상태

| | 상태 |
|---|---|
| 인덱스 (`data_csv/rollout_v1/index{,_probe}.csv`) | ✅ 7392행 37컬럼, 검증 14항목 통과 |
| 레지스트리 (`datasets.md ## rollout_v1`) | ✅ |
| bash 스모크 (`LIMIT=224`, 1 GPU) | ✅ 완주, base 3종 캐시 확인 |
| 토큰 캐시 전수 (145 GiB) | ✅ `cache/rollout_v1_vith` |
| 위치·속도 회귀 | ✅ `analysis/rollout_position.py` → `Archive/POSITION_READOUT_2026-09-06.md` |
| 그림 (2패널) | ✅ `figures/position/fig_position_readout.pdf` |
| 방법 문서 (v11 이식용) | ✅ `Archive/METHOD_POSITION_READOUT_2026-09-06.md` |

---

## 8. 재현

```bash
# 1. 인덱스 (검증만 하려면 --write 없이)
python z_research/scripts/data/build_rollout_index.py --write

# 2. 병합 확인 — GPU 안 씀, 몇 초
SET="probing.fit_groups_sweep=[null] probing.optims.attn_30.num_epochs=1 \
probing.targets.shape.classes=[capsule,cone,cube,cylinder,pyramid,sphere,torus]" \
  DRYRUN=1 bash z_research/scripts/run.sh attn_probe rollout_v1 vith

# 3. 스모크 — LIMIT=224 는 한 condition 의 완전 교차(7x8x4)라 클래스가 전부 들어간다
SET="..." GPUS=1 LIMIT=224 BATCH_SIZE=4 \
  bash z_research/scripts/run.sh attn_probe rollout_v1 vith

# 4. 전수 캐시 추출 — ★ -w vll6 (데이터가 그 노드 로컬) · WMA_RUN=1 필수
SET="probing.fit_groups_sweep=[null] probing.optims.attn_30.num_epochs=1 \
probing.targets.shape.classes=[capsule,cone,cube,cylinder,pyramid,sphere,torus]"
sbatch --job-name=ex_rollout -w vll6 --gres=gpu:8 \
  --export=ALL,WMA_RUN=1,P=attn_probe,D=rollout_v1,M=vith,GPUS=8,SET_B64="$(printf %s "$SET" | base64 -w0)" \
  z_research/scripts/sbatch.sh

watch -n 1 bash z_research/scripts/monitor.sh
```

⚠️ **`SET` 에 콤마(YAML 리스트)가 있으므로 반드시 `SET_B64` 로 싣는다** — `--export` 는
콤마가 구분자다 (CLAUDE.md §7-1, 2026-08-30 에 job 7개가 이것 때문에 죽었다).
⚠️ **`runs` 를 줄이지 말 것.** z / p / h base 3종이 전부 필요하다. 싸게 만드는 것은
`num_epochs=1` 로 한다 (head 는 버리는 값). `fit_groups_sweep` 을 `[null]` 로 접는 이유는
`condition` 이 33종이라 `auto` 면 297 head 가 되기 때문이다.

## 9. 다음

1. **문맥 끝 가림 팔** — 등속의 1차 외삽 지름길을 막는 유일한 방법.
   `IntPhysGenV11_occlusion_timing_ablation` 이 *"문맥 끝만 가리고 미래는 안 가리는 팔이
   지금 셋에 없다"* 고 적어둔 칸을 채운다. 미래는 깨끗해야 `h` 가 대조군으로 유효하다
2. **가속 축** — 같은 readout 으로 "가속을 읽는가" 를 직접 측정.
   라벨이 2차식이 되므로 `slope()` 를 2차 적합으로 바꿔야 한다 (METHOD §9-2)
3. ~~**v11 이식**~~ — **2026-09-07 완료** → [`../v11_roll_out/README.md`](../v11_roll_out/README.md).
   라벨은 인덱스가 아니라 v11 `metadata.csv` 의 `object_x_cm_by_sample` (32 샘플) 에서 바로 만든다
