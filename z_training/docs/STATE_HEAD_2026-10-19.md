# state_head 설계 — 48 물체에서 막히고 물체 1 개로 줄였다 (2026-10-19)

> 결론 먼저: **head 구조는 병목이 아니었다.** 동결 predictor 위에서 head 넷을 따로 학습해 보니
> 넷 다 recall 0 % · 정체 0 % 였고 위치는 **파라미터 0 개짜리 복사 기준선보다 2.5~3 배 나빴다.**
> 그래서 head 를 더 고치는 대신 **과제를 줄였다** — 물체 1 개 세트 (`phys2d_single_v1`) + query 1 개 head.

관련: [`PHYS2D_SUP_DATASET_2026-10-18.md`](PHYS2D_SUP_DATASET_2026-10-18.md) (48 물체 세트) ·
`configs/training/datasets.md` (`phys2d_single_v1_state`) · `z_vll7_training/configs/phys2d_single_v1_pft.yaml`

---

## 1. 왜 head 가 필요한가 (사용자 10-19)

> "head 의 목적은 supervision 이야. 가려지던 안 가려지던 그리고 계속해서 identity, position 을
> 맞출 수 있다면 ar 의 단점이 풀릴 거 같아서"

AR predictor 의 약점은 **마지막 관측에 안 보이는 것** (영속) 과 **없는 것** (사건) 이다 (정본 §P3).
latent L1 하나로는 그 둘에 신호가 거의 없다 — 가려지면 목표 latent 도 "가려진 장면" 이라 복사로 풀린다.
head 는 가려져 있어도 **그 물체가 어디에 있는지**를 predictor 출력에서 읽어 내게 해서 그 신호를 만든다.

⚠️ **상태 라벨은 관측만으로 얻는 정보가 아니다** — "관측만으로 학습" 주장과 섞지 말 것 (CLAUDE.md §12 질문 6).

## 2. 동결 predictor 위 head 단독 학습 — 넷 다 실패

`z_training/tools/state_head_fit.py`. predictor = **vast AR-state ep70 (base, 상태 감독 없음)** 동결,
encoder 동결, head 만 새로 초기화해 Adam lr 1e-3 로 1,000 step. 평가 = `phys2d_sup_v2` val 경계 창 192 개.
위치 단위 px (256² 화면, 1 칸 = 18 px = 물체 반폭).

| head | 파라미터 | tf_ctx 보임 | rollout 보임 | recall | 정체 | attention 집중도 |
|---|---:|---:|---:|---:|---:|---:|
| `light` | 69,126 | 72.3 | 65.2 | 0 % | — | **1.6** |
| `query3` | 130,615 | 77.2 | 68.1 | 0 % | 0 % | 15.0 |
| `attn` | **7,546,886** | 81.7 | 71.4 | 0 % | — | **35.5** |
| **복사 기준선** (파라미터 0) | 0 | **28.3** | — | **100 %** | — | — |

읽는 법:
- **셋 다 복사보다 2.5~3 배 나쁘다.** 학습된 자가 파라미터 없는 기준선에 지면 그 자로 predictor 를 말할 수 없다 (CLAUDE.md §7-3)
- **용량을 109 배 늘려도 (69k → 7.5M) 더 나빠진다** → head 구조·용량이 병목이 아니다
- **attention 을 날카롭게 해도 (1.6 → 15.0 → 35.5) 위치는 나아지지 않는다** → 정보가 토큰에 국소적으로 없다
- recall 0 % · 정체 0 % — "뭐가 있나" 조차 못 읽는다

산출물: `z_training/runs/_vast_ar_ep70_probe/head_fit/{light,q3,attn,pilot}.json`

⚠️ **vast 의 같은 head 수치와 직접 비교 금지.** vast 는 **자기네 attn-run e70** (이미 상태 감독으로 학습된) predictor
위에서 쟀다 (light 42.8 px / recall 9 %, attn 20.5 px / 51 %). 우리는 **base AR ep70** 이다. 즉 vast 숫자는
"head 가 좋다" 가 아니라 **"학습하면 읽히게 된다"** 를 보여준 것이다. 위 표의 세 줄끼리만 비교할 수 있다.

### 2-1. 그래도 확인된 것 하나 — q_std 초기화

`LightStateHead` 는 집중도가 **정확히 1.0** (= ln 256 균등) 에 붙어 있었다. 원인은 데이터와 무관한 초기화 문제다:
affine 없는 LN 뒤라 `|x_i| = √D` 가 **정확히** 성립해서 `logit = q·x/√D` 의 표준편차는 **q 성분의 표준편차와 같다.**
`q_std = 0.02` 면 logit std 0.02 → softmax 가 균등에서 못 벗어나고, 그 상태에서는 `log_tau` 의 gradient 도 거의 0 이라
**학습 가능한 τ 혼자서는 빠져나오지 못한다** (실측: q_std 1.0 → 집중 10.9, q_std 0.02 → 1.06).
데이터 무관 상한도 `|logit_ki| ≤ |q_k|` 로 바로 나온다. **`q_std = 1.0` 은 필수다.**

`query3` 는 이것을 고쳤고 (집중도 15.0, τ → 1.56) **진단은 맞았지만 head 는 여전히 작동하지 않았다.**

## 3. 기각한 설계 (다시 제안하지 말 것)

| 설계 | 왜 뺐나 |
|---|---|
| 격자 soft-argmax (16×16 칸) | 사용자 기각. 위치를 칸으로 이산화하면 자 해상도가 칸에 묶인다 |
| 256 칸 CE | 사용자 기각 ("256 칸 CE 로 가는 건 좀 그런데") |
| 토큰별 공유 `Linear(1280→2)` | **실측 19~30 px vs 파라미터 0 "자기 칸 중심" 6 px.** V-JEPA 는 RoPE 라 절대 위치가 토큰 임베딩에 없다 — 위치는 attention pooling 으로만 읽힌다 |
| 슬롯 9 개 + Hungarian matching | 사용자 기각. 더해서, 매칭을 **블록마다** 하면 시간적 정체 연속성 (이 head 의 목적) 이 깨진다. clip 당 한 번이어야 하는데 그러면 rollout 에서 라벨이 바뀐다 |

## 4. 지금 가는 길 — 물체 1 개 + query 1 개 (사용자 결정 10-19)

> "`/local_datasets/world/training_data/phys2d_single_v1` 지금 single obj 로 준비함.
> 이거 attentive pooling (query 1 개 하고) 그냥 head 3 개 달자."

48 물체를 한꺼번에 묶는 과제 자체를 없앤다. 장면에 움직이는 물체가 **하나뿐**이면
"어느 물체냐" 가 사라지고 head 는 "그 하나가 어디 있나" 만 읽으면 된다.

### head (`state_head.SingleStateHead`, 70,455 파라미터, vast 167db20)

```
블록마다:  x_i = LN(predictor 출력 토큰 i)                       i = 1..256
           a   = softmax_i( τ · q·x_i / √D )                     query 1 개, 투영 없음, q init std 1.0
           z   = Σ_i a_i x_i                                     (T, D)
           물체 = Linear(D → 49)   48 정체 + 없음,  CE
           위치 = Linear(D → 2)    (x, y)/256,      L1   ← 가려져 있어도 건다
           속도 = Linear(D → 3)    (cos, sin, ‖v‖/8), L1
```
`single_losses` 가 48 칸 라벨에서 **그 영상의 정체 칸 하나**를 뽑아 쓴다 (영상당 물체 1 개).

### 데이터 `phys2d_single_v1` (사용자 생성, vast cf8e528 생성기)

train 39,593 편 / val 1,440 편, video_fps 4, 길이 40~128 장, 전체 장의 **40.9 % 가 가려짐**.
창 자리별 가림 확률이 평평하다 (c0~p15 폭 2.2 pt) — 가림이 특정 자리에 몰리지 않는다 (`train/audit.md`).
학습은 두 종류를 weight 로 섞는다 (vast `train_mix.tsv` 와 같은 비율):

| 줄 | 무엇 | 수 | weight |
|---|---|---:|---:|
| `phys2d_single_v1_state` | clip 목록, 클립 안 **무작위 시작** | 39,593 편 | 0.93 |
| `phys2d_single_v1_state_bnd` | **예측 첫 장 = 충돌 · 이탈** 고정 창 (C16 K16 s1) | 18,551 창 | 0.07 |

만드는 법: `python z_training/data/build_phys2d_single_index.py --root /local_datasets/world/training_data/phys2d_single_v1 --out data_csv/phys2d_single_v1`

⚠️ **CLAUDE.md §11 "물체 하나 = 화면 움직임의 전부" 상태가 이 세트에서는 성립한다** (가림막 4 개는 움직이지만 distractor 는 없다).
그래서 이 세트의 수치는 **head 가 읽을 수 있는가**만 재는 것이고, VoE 채점 주장에 쓰지 않는다 — 판정은 VOBench late 로 한다.

## 5. 돌고 있는 run — `phys2d_single_v1_pft`

vast AR-state ep70 위 post-FT. 전체 손실 = AR latent L1 (tf + rollout) + 1.0 × (물체 CE + 위치 + 속도).

- clip 1.0 은 **predictor 에만** (`state_head.clip_separate`) — 같이 자르면 head grad 가 predictor norm 에 눌려 거의 못 배운다 (vast 167db20)
- head lr = 스케줄 lr × 3 — head 단독 검증에서 물체를 찾기까지 ~1,000 step 걸렸다 (vast)
- in-loop val = **VOBench late 432 matched pair** (정식 채점과 같은 정의)

```bash
GPUS=4 bash z_vll7_training/tmux_train.sh phys2d_single_v1_pft
watch -c -n 5 bash z_vll7_training/monitor_state_head.sh phys2d_single_v1_pft
```

### 아직 안 한 대조군 (이것 없이는 "predictor 가 바뀌었다" 를 못 말한다)

1. **head-only** — `SET="state_head.freeze_predictor=true"`. predictor 동결, head 만. 이게 PFT 와 같은 위치 오차를 내면 PFT 의 개선은 head 가 읽어 낸 것이지 predictor 가 담게 된 것이 아니다
2. **head 없음** — `SET="state_head=null"`. 같은 데이터로 AR L1 만. VOBench 차이가 head 때문인지 데이터 때문인지 가른다
3. **복사 기준선** — 위 §2 표의 28.3 px. head 위치 오차는 **항상 이것과 같이 읽는다**

## 재현

```bash
# 데이터 목록
python z_training/data/build_phys2d_single_index.py \
    --root /local_datasets/world/training_data/phys2d_single_v1 --out data_csv/phys2d_single_v1

# §2 의 head 단독 학습 (동결 predictor)
CUDA_VISIBLE_DEVICES=0 /data/hyuntak/anaconda3/envs/vjepa2_bw/bin/python z_training/tools/state_head_fit.py \
    --run _vast_ar_ep70_probe --ckpt latest.pt --head '{"type": "single"}' --tag single \
    --root /local_datasets/world/training_data/phys2d_single_v1 --rows uniform --steps 1500

# §5 의 PFT
GPUS=4 bash z_vll7_training/tmux_train.sh phys2d_single_v1_pft
```
