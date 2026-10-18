# attention_map — AR predictor 는 과거 문맥을 보나 (2026-10-13, 2026-10-14 이 폴더로 정리)

옛 자리 `bottleneck/attention/` (원자료) · `bottleneck/figures/fig_ar_attention_*` (그림) 를 2026-10-14 에 여기로 옮겼다. 내용은 같다 (그림 `compare/fig_ar_attention_blocks` 는 새 경로에서 다시 그려 같은 파일 크기를 확인했다).
해석 정본: `auto_research/paper/STATUS_2026-10-13.md` B13.

## 폴더 = 모델

| 폴더 | 체크포인트 | 학습 | 세트 (clip 수) |
|---|---|---|---|
| `vjepa2/` | 릴리즈 (`--ckpt` 없음) | V-JEPA 2 ViT-H 릴리즈 predictor, mask token 으로 미래 8 블록 한 번에 (2026-10-14 추가) | 〃 |
| `vast_ar_state_ep70/` | `z_training/vast/ar_state/latest.pt` | AR-state, k710 + ssv2 + HowTo 1 %, **4 fps**, 70 ep | vob_visible 100 · vob_late 300 · ssv2 400 |
| `ariel_ar_ep40/` | `z_training/ariel/ar_future_1e_5/latest.pt` | Ariel AR, **12 fps**, 40 ep | 〃 |
| `ariel_ar_state_ep40/` | `z_training/ariel/ar_state_future_1e_5/latest.pt` | Ariel AR-state, **6 fps**, 40 ep | 〃 |
| `vobench_release_postft_ep30/` | `z_training/runs/vobench_release_postft/latest.pt` | **릴리즈 predictor 를 VOBench split_v2 train (720 clip, 가능만) 으로 post-FT**, 30 ep, temporal_prefix 16 — 구조는 릴리즈와 같다 (one-shot, full attention) (2026-10-14 추가) | 〃 |
| `vobench_vast_ar_state_postft_ep30/` | `z_training/runs/vobench_vast_ar_state_postft/latest.pt` | **vast AR-state ep70 을 VOBench split_v2 train 으로 post-FT**, 30 ep, prefix_window 8 + 8 — 구조는 vast 와 같다 (AR-state, block-causal) (2026-10-14 추가) | 〃 |
| `compare/` | — | 모델 비교 그림 | — |

⚠️ `ariel_ar_state_ep40` 와 `vast_ar_state_ep70` 은 **같은 학습의 40 · 70 epoch 가 아니다** (데이터 · fps 가 다른 두 학습). 그래서 폴더 이름에 `ariel_` · `vast_` 를 붙였다.
⚠️ **`vjepa2/` 는 판이 다르다.** 릴리즈 predictor 는 문맥 encoder (online) 토큰 + mask token 으로 미래 8 블록을 **한 번에** 만들고 full attention 이다.
  '걸음 r' = 미래 블록 C+r 의 mask token query 이고, key 는 16 블록 전부 (자기 블록 · 뒤 미래 블록 포함) — npz 끝 축이 16 (AR 은 15) 이고 균등이 1/16 이다.
  입력 공간도 다르다 (AR = 블록별 LN(target_encoder)). 그래서 질량 분포만 비교한다.

### 두 판의 query 가 다르다 (2026-10-14 사용자 지적 — 읽기 전에)

| | AR · AR-state (vast · Ariel) | V-JEPA 2 릴리즈 |
|---|---|---|
| 예측 방식 | 한 걸음에 한 블록, 8 걸음 (rollout) | **one-shot** — 미래 8 블록을 forward 한 번에 |
| attention 규칙 | block-causal (문맥도 뒤를 못 본다) | **양방향** full attention — 문맥 ↔ 미래 · 미래 ↔ 미래 (t9 가 t16 을 본다) |
| 블록 C+r 을 만드는 query | 위치 C+r−1 의 토큰 (k1 = 문맥 끝 t8, 그 뒤 = 앞 걸음 출력) — **내용이 있다** | 위치 C+r 의 mask token — **0 층에서 8 블록 모두 같은 `mask_tokens[0]`**, 위치는 RoPE 로만 다르다 |
| '자기 블록' | k1 = t8 (마지막 문맥), k≥2 = 바로 앞 생성 블록 | 그 mask 블록 자신 |
| 그림의 행 t9 … t16 | 시간 순서로 이어진 걸음 | **같은 forward 의 다른 query 블록** (순서 없음) |
| 입력 공간 | 블록별 LN(target_encoder) | 문맥 encoder (online) |

### 한눈에 (층 · head · clip 평균, `<set>.json` 의 `block_mass_all` 에서 자기 블록을 떼어 다시 셌다; vob_late · ssv2)

| 모델 | k1: 자기 블록 / t1–t7 / t8 (자기 아님) / 다른 미래 | k8: 자기 / t1–t7 / t8 / 다른 미래 | 블록 6 물체 / 거울, visible · late (k1) |
|---|---|---|---|
| vast AR-state ep70 | 0.65 / 0.35 / — / — | 0.37–0.40 / 0.30–0.31 / 0.13–0.15 / 0.17 | 4.6 / 0.9 % · 1.5 / 0.9 % |
| Ariel AR ep40 | 0.66–0.69 / 0.31–0.34 / — / — | 0.50–0.53 / 0.19–0.21 / 0.08–0.10 / 0.19–0.20 | 7.0 / 0.8 % · 2.7 / 1.2 % |
| Ariel AR-state ep40 | 0.65–0.70 / 0.30–0.35 / — / — | 0.38–0.45 / 0.27–0.32 / 0.13–0.14 / 0.16–0.17 | 5.5 / 0.9 % · 2.9 / 1.3 % |
| **V-JEPA 2 릴리즈** | 0.19–0.23 / 0.31–0.34 / 0.18–0.22 / 0.25–0.29 | 0.15 / 0.36–0.37 / 0.12–0.13 / 0.36–0.37 | 3.0 / 0.3 % · **0.3 / 0.4 %** |

- ⚠️ **정정 (2026-10-14).** 처음엔 AR k1 의 'last 0.61–0.71' 을 "마지막 문맥 블록 몰림 · 최근 편향" 으로 읽었다. 이는 **query 가 자기 블록 (t8) 을 보는 몫**이다. 자기 블록을 떼면 AR k1 은 남은 질량을 전부 t1–t7 에 준다.
- AR 은 내용 있는 query 가 자기 블록에 40–70 % 를 준다. 릴리즈의 mask query 는 자기 블록에 15–23 % 만 주고, 문맥 · 다른 미래 mask 에서 모은다. query 의 정체가 달라 생기는 차이로 읽는다. 어느 쪽이 과거를 '더 잘 본다' 는 뜻이 아니다.
- 물체가 마지막에 가려지면 네 모델 모두 과거의 물체 자리를 visible 때보다 덜 본다. 릴리즈는 거울 대조 이하까지 떨어진다.
  - 릴리즈 물체 query 는 0 층에서 내용 없는 mask 다. 그래서 '물체 query' 는 **위치만으로 정의된 것**이다. 거리가 같은 거울 대조가 그 위치 효과를 뺀다.
- 간접 경로 — 릴리즈는 문맥도 양방향이다. 블록 6 정보는 t7 · t8 토큰이나 다른 mask 를 거쳐 들어올 수 있다 (AR 은 문맥 안에서 앞→뒤만). 문맥 query → mask 몫은 재지 않았다 (npz 는 미래 query 만 담는다).

### 층별 (2026-10-14, 사용자 "layer 별로 attention map 이 다를 거 아니야")

위 표의 값은 모두 **12 층 × 12 head 평균**이다. 원자료는 층 · head 축을 그대로 갖는다 (`bm_*` · `om_*` = (N, k, 층, head, 블록)).
층별로 보면 평균이 가리는 것이 있다 (vob_late · vob_visible, k1):

| | AR (vast · Ariel AR) | 릴리즈 |
|---|---|---|
| 자기 블록 / t1–t7 | 층마다 0.40–0.80 / 0.20–0.60. t1–t7 가 가장 큰 층: vast L4 0.60 · L11 0.52, Ariel L0 · L4 · L7 0.48–0.49 | L0 은 자기 0.04 · t1–t7 0.62 · t8 0.33 · 다른 미래 0.02. 0 층의 mask 는 모두 같아서 문맥만 본다. L1 부터 다른 미래 mask 몫이 0.20–0.46 으로 커진다 |
| 블록 6 물체 / 거울, visible | **L2–L4 에 몰린다**: vast 5.1 · 13.9 · 15.8 % (거울 1.0–1.6), Ariel 14.9 · 17.9 · 21.9 % (거울 0.8–1.1). 나머지 층은 1–7 % | 층에 퍼져 있다: L1 · L5–L8 · L10–L11 에서 3–5 % (거울 0.2–0.4) |
| 블록 6 물체 / 거울, late (블록 6 = 마지막으로 보임) | 같은 L2–L4 가 vast 2.1–3.1 %, Ariel 2.2–4.6 % 로 떨어진다 (거울 0.9–1.7). 물체 query 가 과거 물체를 찾는 층이 late 에서 가장 크게 준다 | **12 층 모두** 0.1–0.5 % ≤ 거울 0.2–0.6 % |

층별 지도 그림 (`plot_attention_layers.py`):
- `compare/fig_attention_layers_objcentered` — **전체 clip 평균.** 물체 query (t9 의 참 물체 자리) → key 블록 t6 · t7 · t8 의 **그 블록 물체 자리 중심 7×7**. 행 = 층, 네 모델 × (visible | late). 색 = 블록 안 균등 대비 (log).
  - AR 셋은 visible 에서 L3–L4 의 t6 · t7 물체 자리가 밝고, late 에서는 같은 칸이 꺼진다. 릴리즈는 visible 에서도 약하고, late 에서는 t6 · t7 이 모든 층에서 꺼진다.
  - ⚠️ t8 칸은 물체 자리가 query 자리와 거의 겹친다 (블록당 이동이 작다). 그래서 밝은 것이 물체를 봐서인지 가까워서인지 이 그림으로는 못 가른다. 그 구분은 거울 대조 (위 표) 로 한다. 이 그림에는 거울 대조가 없다.
- `<model>/figures/fig_attention_layers_{visible,late}_k1` (vast · vjepa2) — **clip 하나.** 행 = 층, 열 = 보는 블록. 지도는 층마다 한 색 범위이고, 오른쪽에 층 × 블록 % 행렬이 있다. AR 의 자기 블록 = t8, 릴리즈의 자기 블록 = t9 (초록 테두리).
- **행 그림의 층별 판** (`plot_ar_attention_rows.py --model <m> --layers`, 네 모델 다): `<model>/figures/layers/`
  - `fig_ar_attention_rows_{visible,late}_L00…L11.png` — 정본 행 그림 (행 = 만드는/query 블록 t9–t16) 을 층 하나씩.
  - `…_layers.gif` (층을 1.2 초마다 넘김) · `…_layers.pdf` (12 쪽).
  - 행 안 색은 그 층 안에서만 맞춘다. 오른쪽 행렬은 모든 층에서 0–70 % 로 같은 범위다.
- head 별은 아직 안 나눴다 (평균만). 층 평균으로는 결론이 같지만, 특정 head 하나가 late 에서 과거 물체를 찾을 가능성은 열려 있다.

### 릴리즈 vs 릴리즈 post-FT (2026-10-14, 사용자 "vobench_release_postft 로도 해 보고 release 랑 어떤 차이가 있는지")

⚠️ **분석 clip 의 절반이 post-FT 학습 split 에 있다** (vob_visible 50/100 · vob_late 150/300 이 `split_v2` train). 그래서 VOBench 판은 clip id (`vid`) 를 저장하고
`<set>_{train,test}.json` 을 따로 낸다. 아래 수치는 **test split** 이다. test 는 block 단위 held-out 이지만 **같은 분포** (같은 물체 · 운동 · 배경 축) 다 — 도메인 밖이 아니다.
릴리즈를 다시 돌린 `vjepa2/` 는 vid 를 붙이기 전 값과 비트 단위로 같다 (`block_mass_all` 최대차 0.0).

**행동 (채점, 같은 test 360 block · 720 쌍, 릴리즈 = `VOBench_v1/exp_results/surprise_c16t32__vobench_v1_vith/per_block.json` 을 test block 으로 걸러 다시 셌다)**:
overall 75.7 → **91.8** · late 62.0 → 89.4 · permanence late **54.2 → 98.6** · visible 88.9 → 97.2. 같은 체크포인트의 IntPhys 1 은 `skip2_w32/avg` **81.67** (`z_training/runs/vobench_release_postft/eval/`). ⚠️ 릴리즈 88.89 는 Filtered 규칙 값이라 같은 규칙 대조가 아니다 (미확인) — 떨어진 방향만 기존 post-FT 패턴 (도메인 안에서만 오른다) 과 맞는다.

**attention — 블록 묶음 질량 (모든 query, 층 · head 평균)은 거의 그대로다.**
| | k1: 자기 · t1–t7 · t8 · 다른 미래 | k8: 자기 · t1–t7 · t8 · 다른 미래 |
|---|---|---|
| 릴리즈, vob_late test | 0.19 · 0.34 · 0.18 · 0.29 | 0.15 · 0.37 · 0.12 · 0.36 |
| post-FT, vob_late test | 0.20 · 0.35 · 0.15 · 0.30 | 0.16 · 0.37 · 0.08 · 0.38 |
| 릴리즈, ssv2 | 0.23 · 0.31 · 0.22 · 0.25 | 0.15 · 0.36 · 0.13 · 0.37 |
| post-FT, ssv2 | 0.22 · 0.33 · 0.19 · 0.27 | 0.16 · 0.37 · 0.10 · 0.37 |
- 일관된 변화는 **t8 (마지막 문맥 블록) 이 2–4 %p 줄어든 것** 하나이고, SSv2 (학습에 없던 도메인) 에서도 같은 방향이다. 층별로는 k1 물체 query 의 자기 블록이 L2 에서 +11 (visible) · +21 (late) %p.

**물체 query → 과거 물체 자리 — 여기서 갈린다.** 대조 = 같은 query 가 같은 key 블록에서 **query 와 같은 좌표** 3×3 에 준 질량 (`om_ql`, 10-14 추가 — 먼 걸음은 거울 자리가 화면 밖이라).
물체가 그 블록과 query 블록 사이에 3 토큰 이상 움직인 clip 만. key 블록 = late 는 마지막으로 보인 블록 (t6 49 · t7 101 clip), visible 은 t7. 층 · head 평균, %, test split:
| 미래 블록 | 판 | 릴리즈 물체 / 같은 좌표 | post-FT 물체 / 같은 좌표 | post-FT L5 · L10 물체 |
|---|---|---|---|---|
| k4 (t12) | visible (n 47) | 0.78 / 1.11 | **2.15** / 0.71 | 3.5 · 3.5 |
| k4 (t12) | late (n 144) | 0.15 / 1.81 | **2.12** / 1.53 | 3.0 · 4.1 |
| k8 (t16) | visible (n 50) | 0.18 / 0.72 | **1.16** / 0.50 | 2.5 · 1.9 |
| k8 (t16) | late (n 150) | 0.05 / 1.04 | **0.77** / 1.00 | 1.7 · 1.3 |
| k1 (t9) | late (n 18) | 0.19 / 0.73 | 0.36 / 0.77 | — |
(블록 안 균등 몫 ≈ 0.14–0.28.) train split 은 test 와 같은 크기다 (k4 late 2.26, k8 late 0.85).
- **릴리즈**: 먼 미래 query 는 과거 블록에서 **자기와 같은 좌표**를 본다 (공간 국소성). 물체가 움직여 간 자리는 균등보다도 낮다 (late k4 0.15 vs 균등 0.23).
- **post-FT**: 같은 query 가 **물체가 있던 자리**를 본다. L5 · L6 · L8 · L10 · L11 에서 같은 좌표 대조의 3–10 배. L0–L4 는 릴리즈와 같다 (국소성).
  late 에서도 된다 — 가려지기 전 마지막으로 보인 자리를 찾아간다. 행동의 permanence late 54 → 99 와 같은 방향이다.
- ⚠️ k1 (t9) 은 물체가 3 토큰 이상 움직인 clip 이 18 개뿐이고 차이가 작다. 갈리는 것은 먼 걸음이다.
- ⚠️ 이것은 **VOBench 분포 안** 결과다. 물체 라벨이 없는 SSv2 에서는 블록 질량만 봤다 (t8 감소 하나). IntPhys 1 이 내려간 것과 이 attention 변화의 관계는 안 쟀다.
- ⚠️ attention 질량 ≠ 정보 사용. post-FT 가 물체 자리를 "보는" 것과 그 값을 미래 예측에 쓰는 것은 별개다 — 인과 확인은 anchor 이식 (B8) 을 post-FT 에 걸어야 한다 (안 했다).
- 그림 (**정본, 10-14**): `compare/fig_release_vs_postft_rows_{visible,late}` — 행 = 미래 블록 (t9 … t16 mask query), 열 = 보는 블록, 판 = 릴리즈 · post-FT · 차; 위 = 블록 % (12 층 평균), 가운데 = L5–L11 만, 아래 = 그 블록의 물체 자리 3×3 % (L5–L11).
  층별 · 대조 판 (물러남): `compare/_superseded/fig_release_vs_postft_k{1,4,8}` — 행 = vob_visible · vob_late (test) · ssv2, 열 = 릴리즈 · post-FT · 차 (층 × 블록 %) · 물체 vs 같은 좌표 (층별, post-FT train 은 점선).
  `vobench_release_postft_ep30/figures/` 에 행 그림 · 층별 판 · `fig_attention_layers_*` (예시 late clip 은 post-FT **train** split, visible clip 은 test — 제목에 적었다).

### vast AR-state ep70 vs 그 post-FT (2026-10-14, 사용자 "vobench_vast_ar_state_postft 로 했을 때 그냥 vast 버전이랑 attention map 이 다른지")

릴리즈 판과 같은 규칙이다 — **test split**, 같은 좌표 대조 (`om_ql`), 물체가 3 토큰 이상 움직인 clip. vast vob 판은 vid · om_ql 을 넣으려고 다시 돌렸다 (`block_mass_all` 최대차 2.5e-4 — batch 4 → 2 의 bf16 차).
AR 이라 query = 앞 블록 (C+r−1) 의 토큰, 자기 블록 = t(8+r).

**행동 (같은 test 720 쌍; vast = `VOBench_v1/exp_results/surprise_c16t32__vobench_v1_vith_vast_ar_state_ep70/per_block.json` 을 test block 으로 걸러 다시 셌다)**:
overall 82.8 → **91.0** · late 75.9 → 90.3 · shape late 63.9 → 87.5 · color late 68.1 → 83.3 · permanence late 95.8 → 100. (AR 채점 공간 = 블록별 LN(target_encoder) — 두 모델이 같은 공간이라 서로는 비교된다.)

**모든 query 의 블록 묶음 질량 (층 · head 평균)**: vob_late test k1 자기 0.65 → 0.57 · t1–t7 0.35 → 0.43, k8 t8 0.15 → 0.06. SSv2 는 거의 그대로 (k1 0.65 / 0.35 둘 다, k8 t8 0.13 → 0.11).

**물체 query 의 층 × 블록 질량 (k4, vob_late test)** — vast 는 L1–L4 에서 t8 에 26–60 % 를 주는데, post-FT 는 그것이 **−22 ~ −55 %p** 빠진다.
그 몫은 L1–L2 에서 자기 블록 (t11, +44 ~ +50) 으로, L3–L4 에서 t5–t7 (+5 ~ +12) 으로 간다. visible 도 같은 모양이다.

**물체 query → 과거 물체 자리** (%, 층 평균 물체 / 같은 좌표, 괄호 = 가장 높은 층):
| 미래 블록 | 판 | vast 물체 / 같은 좌표 | post-FT 물체 / 같은 좌표 |
|---|---|---|---|
| k4 (t12) | visible (n 47) | 0.65 / 4.10 (L4 2.5) | **2.02** / 1.25 (L4 **9.2**, 같은 좌표 1.6) |
| k4 (t12) | late (n 144) | 0.39 / 3.34 (L4 1.8) | **3.82** / 1.53 (L3 **18.4**, 같은 좌표 1.6) |
| k8 (t16) | visible (n 50) | 0.19 / 4.25 | **1.42** / 0.93 (L3 7.5) |
| k8 (t16) | late (n 150) | 0.17 / 3.37 | **1.70** / 1.30 (L3 9.7) |
| k1 (t9) | late (n 18) | 0.35 / 0.69 | 1.50 / 2.47 |
train split 도 같은 크기 (k4 late L3 19.5).
- **vast**: 릴리즈처럼 과거 블록에서 **같은 좌표**를 본다 (L0–L4 에서 4–11 %). 물체가 옮겨 간 자리는 L4 에서만 조금 (1.8–2.5 %).
- **post-FT**: **L3–L4 두 층**이 과거의 물체 자리를 찾는다 — late k4 에서 물체 자리 3×3 에 18 % (블록 안 균등 몫 0.3 %, 같은 좌표 1.6 %). L5 이후는 vast 와 같이 거의 0.
  릴리즈 post-FT 는 L5–L11 에 넓게 퍼져 3–4 % 였다. **같은 데이터로 post-FT 해도 바뀌는 층과 세기가 구조마다 다르다.**
- 층 평균 행 그림 (`vobench_vast_ar_state_postft_ep30/figures/fig_ar_attention_rows_late`) 에서는 이 변화가 거의 안 보인다 (L3–L4 만의 일이라 12 층 평균에 묻힌다). `figures/layers/…_L03.png` · `_L04.png` 로 볼 것.
- ⚠️ 릴리즈 판과 같은 단서: VOBench 분포 안 · attention ≠ 정보 사용 (B8 을 post-FT 에 걸지 않았다) · k1 은 clip 18 개.
- 그림 (**정본, 10-14**): `compare/fig_vast_ar_state_vs_postft_rows_{visible,late}` — 행 = 만드는 블록 (making t9 … t16), 판 = vast · post-FT · 차; 위 = 블록 % (12 층 평균), 가운데 = L3–L4 만, 아래 = 물체 자리 3×3 % (L3–L4).
  층별 · 같은 좌표 대조 판 (물러남, 사용자 "와닿지 않는다"): `compare/_superseded/fig_vast_ar_state_vs_postft_k{1,4,8}`.

### 한 층 확대 그림 (2026-10-14, 사용자 "가장 두드러지는 layer 를 골라 시각화")

`compare/fig_vast_spotlight_L3` · `compare/fig_release_spotlight_L8` — 층은 **데이터로 골랐다**: test late clip 전체에서 (post-FT − 기준) 물체 자리 질량의 걸음 평균이 가장 큰 층.
(vast: L3 +31.3 · L4 +26.4 · 나머지 ≤ 10 / 릴리즈: L8 +14.0 · L10–L11 +12 · L5 +10.3, L0–L4 ≤ 1.6 — `--spotlight` 가 출력한다.)
- (a) 예시 test clip `prim_assets_flat_vel_late_k1_shap_cone_pos_a` (물체 t1–t7 보임 · t8–t9 가림): 행 = 걸음 (t9 · t11 · t13 · t16), 열 = 과거 t5–t8, 기준 | post-FT. 하늘색 원 = 그 프레임의 물체, 초록 점선 = query 와 같은 좌표. 색 범위는 행마다 두 모델 공통.
- (b, c) 전체 test clip (late · visible): 걸음별로 **물체가 보이던 과거 블록들** (query 물체 자리에서 3 토큰 이상) 의 물체 자리 3×3 합 (실선) · 같은 좌표 합 (점선) · 균등 (회색).
- 읽은 값 (late): vast L3 — 기준 물체 자리 1–1.5 % (균등 수준) · 같은 좌표 7 → 29 %, post-FT 물체 자리 7 → **49 % (t11)** → 27 % (t16) · 같은 좌표 5–9 %.
  릴리즈 L8 — 기준 물체 자리 ≈ 1 % · 같은 좌표 1.5–6 %, post-FT 물체 자리 3 → **19 % (t12)** → 14 % · 같은 좌표 1–2.5 %.
- ⚠️ 층을 차이가 가장 큰 곳으로 골랐으므로 크기는 그 층의 값이다 (12 층 평균이 아니다). 단서는 위 두 절과 같다 (VOBench 분포 안 · 이동 거리 외우기와 미분리 · attention ≠ 사용).

## 모델 폴더 안

- `<set>.json` — 요약 (`summarize()`): key 블록별 질량 · 층별 · 물체 query → 블록 6 물체 vs 거울.
- `<set>_r{0..3}.npz` — rank 별 원자료.
  - `bm_all` · `bm_obj` · `om_obj` · `om_mir` (· `om_ql`, 10-14): (N, K 8, L 12, H 12, 15) fp16 (릴리즈 판은 끝 축 16).
  - `vid` (10-14, 릴리즈 · post-FT 판) · `<set>_{train,test}.json` (split_v2 로 거른 요약).
  - `hid` · `yx` · `objmap` · 예시 `ex_item` · `ex_A` · `ex_yx` · `ex_hid`.
- `figures/` — 그 모델 하나의 예시 그림. 지금은 vast · vjepa2 에 있다.
  - `fig_ar_attention_rows_{visible,late}` — **예시 정본** (vast · vjepa2 둘 다, `--model`). 행 = 만드는 블록 t9–t16, 열 = 보는 블록 t1–t16, 오른쪽에 같은 값의 행렬.
  - `_superseded/fig_ar_attention_maps` (k = 1 · 4 · 8 옛 예시판) · `_superseded/fig_ar_attention_simple` (k = 1 만) — 배치 · 글자를 읽기 어렵다는 사용자 지적 (10-13) 으로 rows 그림이 대체했다.
- `compare/fig_ar_attention_blocks` — (a) vast 걸음 × key 블록 질량 지도 · (b) 세 AR 모델 + 릴리즈 (검정) history vs 마지막 블록 질량 (균등선 둘) · (c) 블록 6 물체 visible vs late (거울 눈금).

## 단서

- attention 질량은 정보 사용량이 아니다 (value 크기 · residual 경로를 안 쟀다). 인과 증거는 B8 (anchor) · B11 (history 가치) 이다.
- 문맥도 block-causal 이다 (`PrefixSpec(1, nb−1, S, future_causal=True)`). t8 은닉에 t1–t7 정보가 이미 섞였을 수 있다. 그래서 과거로 가는 직접 attention 이 작다고 해서 역사를 안 쓴다는 결론은 나오지 않는다. attention rollout 은 아직 안 했다.
- 예시 그림은 clip 하나씩이다. 수치 주장은 `<set>.json` 으로 한다.

## 재현

```bash
bash auto_research/scripts/chain_1013_attention.sh                       # AR 세 모델 + 릴리즈 × 세 세트 → <model>/<set>{.json,_r*.npz}
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
$P z_research/scripts/analysis/ar_attention_history.py --name vast_ar_state_ep70 --set vob_late --summarize   # 요약만 다시
$P z_research/scripts/figures/plot_ar_attention.py          # CPU — compare/fig_ar_attention_blocks + vast_ar_state_ep70/figures/_superseded/fig_ar_attention_maps
$P z_research/scripts/figures/plot_ar_attention_rows.py     # GPU 1 장, 클립 2 개 → vast_ar_state_ep70/figures/fig_ar_attention_rows_{visible,late}
$P z_research/scripts/figures/plot_ar_attention_rows.py --model vjepa2   # → vjepa2/figures/
$P z_research/scripts/analysis/ar_attention_history.py --name vjepa2 --set vob_late --gpus 4     # 릴리즈 원자료 (--ckpt 없음)
$P z_research/scripts/figures/plot_attention_layers.py --objcentered                  # CPU → compare/fig_attention_layers_objcentered
$P z_research/scripts/figures/plot_attention_layers.py --model vast_ar_state_ep70     # GPU 1 장 → <model>/figures/fig_attention_layers_{visible,late}_k1 (--step 로 다른 미래 블록)
$P z_research/scripts/figures/plot_attention_layers.py --model vjepa2
$P z_research/scripts/figures/plot_ar_attention_simple.py   # GPU 1 장 → vast_ar_state_ep70/figures/_superseded/
bash auto_research/scripts/chain_1014_attention_postft.sh                # 릴리즈 post-FT × 세 세트 + 릴리즈 vob 판 (vid 저장) 다시 (~7 분, GPU 4 장)
$P z_research/scripts/analysis/ar_attention_history.py --name vobench_release_postft_ep30 --set vob_late --summarize   # <set>.json + <set>_{train,test}.json
$P z_research/scripts/figures/plot_attention_layers.py --pair release --step 3         # CPU → compare/_superseded/fig_release_vs_postft_k4 (--step 0 · 7 = k1 · k8)
$P z_research/scripts/figures/plot_ar_attention_rows.py --model vobench_release_postft_ep30 --layers
$P z_research/scripts/figures/plot_attention_layers.py --model vobench_release_postft_ep30
```
bash auto_research/scripts/chain_1014_attention_vast_postft.sh           # vast post-FT × 세 세트 + vast vob 판 (vid · om_ql) 다시 (~11 분, GPU 4 장, bs 2)
$P z_research/scripts/figures/plot_attention_layers.py --rows vast                     # CPU → compare/fig_vast_ar_state_vs_postft_rows_{visible,late} (정본)
$P z_research/scripts/figures/plot_attention_layers.py --rows release                  # → compare/fig_release_vs_postft_rows_{visible,late}
$P z_research/scripts/figures/plot_attention_layers.py --spotlight vast                # GPU 1 장 → compare/fig_vast_spotlight_L3 (--spotlight release → fig_release_spotlight_L8)
$P z_research/scripts/figures/plot_attention_layers.py --pair vast --step 3             # CPU → compare/_superseded/fig_vast_ar_state_vs_postft_k4 (--step 0 · 7)
$P z_research/scripts/figures/plot_ar_attention_rows.py --model vobench_vast_ar_state_postft_ep30 --layers
⚠️ `om_ql` (같은 좌표 대조) 와 `vid` 는 10-14 에 추가했다 — `vjepa2/` · `vobench_release_postft_ep30/` · `vast_ar_state_ep70/` · `vobench_vast_ar_state_postft_ep30/` 의 vob 판에만 있다. Ariel 두 모델은 다시 돌려야 생긴다.
