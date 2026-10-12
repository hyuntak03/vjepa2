# RollOut v3 — `z vs p` 2 열 예시 GIF, **training_v8 자** (`<scenario>_C16_P32.gif`)

시나리오마다 clip 하나 (속도 8.25 px/2f).
문맥 16f `[f16-f31]` · 예측 32f `[f32-f63]`, 사건은 **f32**.

```
왼쪽   z — context encoder (창 전체를 본다)    ← predictor 에 입력을 주는 그 encoder = 자의 천장
오른쪽 p — predictor (문맥만 보고 미래를 만든다)

○ 흰    진실 위치
● 채움  자가 읽은 위치        ⚠️ presence 가 문턱 아래면 **안 그리고** `ABSENT` 라고만 쓴다
○ 빨강  wall·ledge 의 불가능 미래 진실 (통과 / 부유). 화면 안일 때만
테두리  파랑 = 문맥 구간 · 주황 = 예측 구간
```

자는 **attn** (attention pooling + Linear head 2 개), **training_v8** 학습셋 (14,360 clip) 에서 frozen.
문턱은 training_v8 val 음성 5 % 오탐 지점 — `z` **−0.66** / `p` **+0.83**.
좌표는 training_v8 에서 잰 **치우침을 뺀 값** — `z` (−4.9, +0.3) · `p` (+5.5, −6.3) px.

## 이 clip 들에서 `p` 가 물체를 들고 있는 비율 (예측 16 튜블릿 중)

| 시나리오 | 이 clip | 전수 (C16/P32) | | 시나리오 | 이 clip | 전수 (C16/P32) |
|---|---:|---:|---|---|---:|---:|
| `flat_v` 등속 | 88 % | 79 % | | `arc` 포물선 | 69 % | 67 % |
| `flat_a` 가속 | **100 %** | 82 % | | `ramp_a` 내리막 | 62 % | 62 % |
| `flat_d` 감속 | 88 % | 76 % | | `ramp_d` 오르막 | 62 % | 57 % |
| `wall` 충돌 정지 | **56 %** | 61 % | | `ledge` 낙하 | 62 % | 81 % |

전수는 `../../exp_results/windows/WINDOW_SUMMARY.md` 의 시나리오별 표 (`100 − p 없다%`).
⚠️ **한 clip 짜리 예시다.** 두 열이 어긋나는 칸이 있다 (`ledge` 62 vs 81 %) — 결론은 전수로만 낸다.

## 자가 이 창에서 통했다는 근거

같은 창 (C16/P32) 에서 `z` 는 '없다' 2.8 % · 0.49 칸, `h` 는 2.1 % · 0.46 칸이다.
⚠️ 다만 **`ledge` 는 예외**다 — 새 자는 선반에서 떨어지는 구간 (t9~t14) 에서 `z` 도 '없다' 를 9.7 % 낸다
(`../README.md` §2). `ledge` GIF 의 왼쪽 열에 ABSENT 가 보이면 그 칸은 **자를 못 믿는 칸**이다.
⚠️ 마지막 튜블릿 (t15) 은 `arc`·`ramp_a` 에서 `z` 도 흔들리므로 빼고 본다.

## 재현

```bash
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
$P z_research/scripts/figures/plot_rollout3_example_gif.py --ctx 16 --prd 32 --reps z p
```
