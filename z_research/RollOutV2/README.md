# RollOut V2 — 운동 법칙 7종: predictor 는 어떤 미래를 만들었나 (2026-09-09)

> V1 은 등속 하나였고 (`../RollOutV1/`), v11 위치 readout 은 조건당 궤적이 하나라 지름길을 못 막았다 (`../v11_roll_out/`).
> V2 는 법칙 7종 × primary 7 레벨 × secondary 2, 그리고 **미래가 두 갈래인 ledge(낙하/부유)·wall(정지/통과)** 를 갖는다.
> 데이터 `/data2/local_datasets/world/world_analysis/RollOut_v2` (5,488 clip, 가림 없음), 레지스트리 `datasets.md ## rollout_v2`.

---

## 한 줄

**자를 `h` 에서 정하면(토큰 단위 decoder) `h` 는 1.5 px 로 읽히지만 `p` 는 못 읽는다 (R² ≤ 0.36). 자를 `p` 에서 정하면 단일 미래
법칙은 전부 읽히고(위치 R² 0.78~0.92, 속도 β 0.75~1.2, 안 본 레벨에서), ledge 의 `p` 는 낙하를 만들었으며, wall 은 판정 보류다.**
`p` 의 위치 코드는 `h`·`z` 어느 것과도 같은 자리에 있지 않다 — pooled 도, 토큰 단위도, LN 을 걸어도.

---

## 1. 라벨 — plan 이 정본이다

- 궤적은 생성기 plan `UnrealEngine/gen/plans/blocks_rollout2.json` 의 `x_sampled/z_sampled` (불가능 변이는 `*_pair`).
  `build_rollout2_index.py` 가 카메라 투영식으로 화면 좌표를 만들어 index 에 싣는다 (flat 에서 metadata 픽셀과 0.09 px 일치, 검증 15항목).
- **metadata 의 `object_*_by_sample` 은 믿지 않는다.** 09-09 16:14 판은 flat 외 시나리오가 primary 와 무관한 같은 배열이었고,
  17:17 수정판도 **wall 불가능 클립(통과)에 정지 궤적**을 쓴다 (픽셀 대조로 확인).
- 화면 밖 튜블릿은 라벨에서 뺀다. ledge/wall 은 문맥 초반에 물체가 화면 밖에서 들어오고, wall 통과는 미래에 화면 밖으로 나간다.
- 좌표는 화면 정규화 (x, y) = px/144 − 1. 세로 운동(fall, arc, ledge)이 있으므로 (T, 2) 로 읽는다.

## 2. 자(readout) 세 가지와 split

| 자 | 어디서 정하나 | 형태 |
|---|---|---|
| `p_A` / `p_B` | `p` 미래 8 슬롯, 7 시나리오 풀링 / ledge·wall 제외 | 256토큰 공간평균 → 좌표당 `w` 하나(8슬롯 공유) 최소제곱 |
| `h`, `z` | `h` 미래 / `z` 문맥 | 같은 형태 → `p` 에 이식 |
| **token** | `h` 의 16 튜블릿 전부, pos+imp | **토큰마다 선형 점수 하나 → 16×16 softmax → 격자 기대값** (1,281 파라미터, MSE 뿐) |

split: 클립 30% 학습 / 70% held-out (셀 층화), plan 의 holdout primary 레벨(2/7)은 평가 전용. `p` 의 판정 대상(ledge·wall)은 `p_B` 와 token 자에서 학습에 없다.

## 3. 결과

### 3-1. 토큰 decoder — `h` 에서는 완벽, `p` 로는 안 옮겨진다 (holdout 레벨)

| | peak 가 물체 안 | 위치 R² | MAE px | 속도 β |
|---|---:|---:|---:|---:|
| `h` 미래 (self) | 0.99~1.00 | 0.95~0.996 | 1.4~2.0 | 1.00~1.05 |
| `h` 문맥 (학습과 다른 시각) | 0.98~1.00 | 0.86~0.996 | 1.4~2.9 | 0.93~1.03 |
| **`p` (이식)** | **0.38~0.84** | **0.03~0.36** | 13~28 | 0.01~0.26 |
| `p` + 토큰 LN | 0.43~0.87 | 0.01~0.56 | 10~25 | 0.08~0.40 |

`h` 의 decoder 는 물체 칸을 정확히 가리키고(peak 99.9%), 시각을 옮겨도, 안 본 레벨에서도 그대로다 — 외운 게 아니다.
그 자를 `p` 에 걸면 peak 가 물체 위에 있는 비율이 flat 84% 에서 arc·ledge·wall 40% 로 떨어지고 softmax 가 퍼진다(peak 확률 0.10~0.17 vs `h` 0.25).
pooled 선형의 `h→p`, `z→p` 도 전부 실패 (R² 음수). **`p` 의 위치 코드는 `h` 의 토큰 코드와 다른 자리에 있다.**

### 3-2. `p` 자기 자(`p_A`, 7 시나리오 풀링) — 단일 미래 법칙은 읽힌다 (holdout 레벨)

| | 위치 R² | 속도 β | 가속 (읽음 / 정답) |
|---|---:|---:|---|
| flat_v | 0.78 | 1.21 | — |
| flat_a | 0.85 | 0.94 | −3 / 0 |
| ramp_a | 0.92 | 0.55 (정답 범위 좁음, `h` 0.89) | −27 / −22 |
| arc | x 0.92 · y 0.85 | 0.75 | y −32 / −23 |
| **fall** | **y 0.35** | 0.63 | 56 / 101 |

null(튜블릿 셔플 / 8칸 동일 / 라벨 셔플) |β| < 0.1. 시나리오별 자로 30% 학습하면 flat_v 0.83 까지 오른다(풀링 비용 0.1, 표본 비용은 ridge 로 해결됨).
**fall 만 `p` 가 못 담는다** — `h` 는 0.87. 순수 수직 낙하가 predictor 의 첫 실패 후보다 (arc 는 0.85 라 "공중" 탓이 아니다).

### 3-3. 두 미래 — `p` 는 어느 쪽을 만들었나 (pos 클립, 슬롯 0 제외, 불가능 궤적이 화면 안인 슬롯만)

| | `p_A` (판정 대상 학습 포함) | **`p_B` (제외)** | `h` 천장 (imp 클립) |
|---|---|---|---|
| **ledge** 낙하 vs 부유 | 낙하 100%, y가속 58 (정답 67 / 부유 0) | **낙하 99%, y가속 82** | 부유 100% |
| **wall** 정지 vs 통과 | 정지 100%, x속도 14 (정지 0 / 통과 361) | **47% = chance, x속도 90** | 통과 100% |

- **ledge: predictor 는 낙하를 만들었다.** 선반을 학습에서 뺀 자로 읽어도 같다. 낙하는 경계 뒤에서 시작하는 사건이라 문맥에 없다 (`figures/overlay/ledge_*`: 부유하는 불가능 영상 위에서 주황 마커가 아래로 떨어진다).
- **wall: 판정 보류.** `p_A` 의 정지 100% 는 wall 의 상수 라벨을 흡수한 것이고(A/B 가 갈린다 = 라벨 흡수의 실증), 벽을 안 본 자로는 정지도 통과도 아니다. 벽 장면의 오프셋(−38 px)도 겹친다.

## 4. 말할 수 있는 것 / 없는 것

✅ 등속·등가속·경사·포물선에서 `p` 의 8슬롯에 위치·속도·가속이 있다 (안 본 레벨, null 0).
✅ ledge 에서 `p` 는 낙하를 만들었다 (판정 대상을 학습에서 뺀 자로).
✅ `h` 의 토큰 위치 코드는 `p` 에 없다 — 토큰 단위·LN·pooled 전부. 위치는 `p` 자기 좌표에만 있다.
❌ wall 은 이 자로 판정 못 한다. "정지" 와 "물체 표현이 흐려짐" 을 못 가른다 (사용자 지적 2번). 부재 판정이 필요하다.
❌ `p` 로 정한 자는 "판정 대상이 학습에 없다" 는 조건에서만 유효하다. A/B 가 갈리면 A 를 버린다.
❌ fall 의 낮은 R² 가 predictor 의 실패인지 readout 의 한계인지는 토큰 단위 `p` 자(아직 없음)로 다시 봐야 한다.

## 5. 그림

`figures/overlay/<scenario>_<clip>.png` — 미래 튜블릿 8개 첫 프레임 위에 정답(흰 링) / `p` 읽음(주황) / `h` 읽음(파랑).
ledge·wall 은 pos 프레임 / imp 프레임 두 줄, `p` 마커는 두 줄에서 같다 (문맥이 같다).

## 재현

```bash
python z_research/scripts/data/build_rollout2_index.py --write                       # plan 기반 인덱스 (검증 15항목)
SET="probing.fit_groups_sweep=[null] probing.optims.attn_30.num_epochs=1 probing.targets.shape.classes=[capsule,cone,cube,cylinder,pyramid,sphere,torus]" \
  GPUS=8 BATCH_SIZE=8 bash z_research/scripts/run.sh attn_probe rollout_v2 vith     # 캐시 107 GiB, 4분
python z_research/scripts/analysis/rollout2_position.py                             # pooled 자 4종 + 두 미래 + null (첫 실행 풀링 3.5분)
python z_research/scripts/analysis/rollout2_token_decoder.py --device cuda:0        # 토큰 decoder (h 학습 → p 이식), 2분
python z_research/scripts/figures/plot_rollout2_overlay.py                          # 프레임 위 마커
```
산출물 `exp_results/position_regression.{json,log}`, `token_decoder.{json,log,pt}`, `token_decoder_preds.npz`.
