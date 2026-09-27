# RollOutV3 그림 — **training_v8 자** (2026-09-24)

| | |
|---|---|
| 자 | attn (attention pooling + head 2 개), 5,123 파라미터 — p·z·h 마다 하나 |
| 학습셋 | `training_v8` — 14,360 clip = v5 4,480 + props 3,584 + 증축 6,296 (구조물 × 운동 4 종에 **물체가 있는** clip 4,032 포함) |
| 손실 | `cell_weight` (좌표) · `balance_weight` (둘 다) — 데이터 README §3 |
| 자 폴더 | `exp_results/presence/` |
| v3 읽은 값 | `exp_results/windows/` (지문 `dcd24d8a8a47`) |

## 한 줄 결론

**`p` 는 t8 근처에서 절벽처럼 물체를 놓고, 그 자리는 튜블릿 수가 정하며, 매끄럽게 줄지 않고 깜빡인다.**
자는 창을 타지 않고 위치를 반 칸 안에서 읽지만, **선반의 떨어지는 구간은 못 읽는다.**

## 1. 자 자체 — 학습 도메인 (`readout/fig_readout_train.png`, `readout/ERRORS.md`)

| 표현 | AUROC | recall | precision | L2 평균 / 중앙 |
|---|---|---|---|---|
| `p` | 0.986 | 94.1 % | 97.8 % | 16.1 / 13.5 px |
| `z` | 0.997 | 99.1 % | 98.4 % | 10.8 / 9.0 px |
| `h` | 0.997 | 99.0 % | 98.2 % | 10.6 / 8.8 px |

좌표 치우침 (`attn_bias_px.json`, 읽은 좌표에서 뺀다): `p` (+5.5, −6.3) · `z` (−4.9, +0.3) · `h` (−1.1, +5.6) px.

## 2. v3 에서 — 자가 못 본 세트 (`readout/fig_readout_v3.png`)

v3 에는 음성이 없어 **recall 만** 있다. 자의 상태는 `z`·`h` 로 판단한다 (창 전체를 봐서 자의 천장이다).

| | `z` | `h` |
|---|---|---|
| recall (16 창, P 별) | 96.7 ~ 99.6 % | 97.4 ~ 99.6 % |
| 위치 L2 ('있다' 칸) | 8.5 ~ 9.9 px | 8.0 ~ 9.4 px |
| 경사면 recall (`ramp_a` / `ramp_d`, 16 창) | 97.5 / 98.1 % | 99.0 / 98.5 % |
| **선반 recall (`ledge`, 16 창)** | **93.9 %** | **94.6 %** |

### ⚠️ 선반의 떨어지는 구간은 판정 불가

`ledge` C16/P32 에서 `z` recall 이 **떨어지는 구간 t9~t14 에서만** 90 / 61 / 86 / 81 / 57 / 79 % 로 빠진다 (앞 구간 t0~t8 은 100 %).
이때 물체는 선반에서 떨어져 **화면 y ≈ 126~145 px** (선반 앞면 높이) 에 있다.

학습셋 선반 clip 을 물체 높이별로 세면, 바로 그 높이에서 **양성이 제외 샘플의 절반**밖에 없다:

| 물체 화면 y | 100–120 | 120–130 | 130–140 | 140–146 |
|---|---|---|---|---|
| 선반 clip: 양성 / 구조물 속 (제외) | 1,104 / 2,271 | 689 / 1,424 | 672 / 1,582 | 630 / 932 |

그리고 빈 장면 clip (빈 선반 포함) 은 `balance_weight` 가 **9.0 배**로 가장 무겁다.
→ "선반 앞면 = 없다" 는 강하게, "선반 앞면 앞의 물체 = 있다" 는 약하게 배웠을 수 있다.
**가설이고 검증하지 않았다.** 가르는 실험: 같은 특징으로 `--no-weights` 로 다시 배워 v3 `ledge` 만 다시 잰다 (~15 분).

## 3. 자가 창을 타지 않는가 (`../exp_results/windows/WINDOW_SUMMARY.md`)

| 표현 | '없다' 오작동 (16 창) | 위치 오차 (16 창) |
|---|---|---|
| `z` | 0.0 ~ 5.7 % | 0.42 ~ 0.74 칸 |
| `h` | 0.1 ~ 4.1 % | 0.39 ~ 0.76 칸 |

→ **창을 타지 않는다.** 같은 튜블릿을 P 만 바꿔 재면 `p` 위치가 0.06~0.09 칸 안에서 같다.

## 4. predictor 행동 (`readout/fig_recall_by_tubelet.png`)

| 확인 | 결과 |
|---|---|
| `p` recall 50 % 를 깨는 튜블릿 (P=32) | C4 t3 · C8 t6 · C16 t10 · C32 t8 |
| 속도 3 분위 (C16/P32) | t10 / t10 / t10 — 거리만 67.6 → 98.1 px 로 따라 변한다 (**시간 한계**) |
| 깜빡임 (C16/P32, t10 → t12 → t13) | 12 → **67** → 7 % — 8 개 시나리오 중 7 개가 같은 t12 에서 되살아난다 |

전수와 단서: `../Archive/RECALL_TUBELET_LIMIT_2026-09-24.md`.

### ⚠️ 이식 대조 (`../exp_results/windows/CROSS_HEAD.md`)

C32/P32 후반 (t10~t14) 에서 `h` 자를 `p` 토큰에 걸면 **23~48 %** 를 "있다" 로 읽는다.
다만 그때 위치 오차가 **평균 37.5 px (2 칸)** 로 `p` 자기 자 (20.4 px) 의 두 배다 — 물체를 짚는 판정이 아니다.
대조 (`h` 자를 `h` 토큰에) 는 94~97 %. 두 자가 "둘 다 없다" 로 모이는 정도는 52~76 % 로 강하지 않다.
**"자가 창 후반에서 고장 난다" 는 배제되지만, "두 자가 부재에 합의한다" 는 강하게 말하지 않는다.**

## 폴더

```
readout/       fig_readout_train.png (학습 도메인) · fig_readout_v3.png · fig_recall_by_tubelet.png · ERRORS.md
windows/       fig_profile_C*_P*.png (16) · fig_xy_speed_C*_P*.png (16)
examples/      z vs p 예시 GIF 8 개 (C16/P32)
windows_gif/   창 4x4 격자 GIF 8 개 (자와 무관)
```

## 재현

```bash
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
# 기본 경로 (exp_results/presence · exp_results/windows → figures/<하위>)
$P z_research/scripts/figures/plot_readout_errorbars.py                       # readout/ 3 장 + ERRORS.md
for c in 4 8 16 32; do for p in 4 8 16 32; do
  $P z_research/scripts/figures/plot_rollout3_profile.py --ctx $c --prd $p       # windows/fig_profile_*
  $P z_research/scripts/figures/plot_rollout3_windows.py --ctx $c --prd $p       # windows/fig_xy_speed_*
done; done
$P z_research/scripts/figures/plot_rollout3_example_gif.py --ctx 16 --prd 32 --reps z p   # examples/ (GPU 1 장)
$P z_research/scripts/figures/plot_rollout3_windows_gif.py                    # windows_gif/
# 자를 다시 배웠을 때 v3 전부 (기본 폴더를 안 덮는다)
R3_OUT=<꼬리표> bash z_research/scripts/analysis/rollout3_rerun.sh
```
