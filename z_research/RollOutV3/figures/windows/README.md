# RollOut v3 — 창별 운동 프로필, **training_v8 자**

두 종류, 16 창씩:

```
fig_profile_C<C>_P<P>.png    행 = 법칙 8 개, 열 = x/v0x · y (px) · vx/v0x · vy (px/tubelet)   ← 읽을 것
fig_xy_speed_C<C>_P<P>.png   4 행 판 (x·y·속도·presence 4 행 x 법칙 8 열)
```

- 검은 실선 = 진실, 검은 점선 = **불가능 미래의 진실** (wall 통과 / ledge 부유)
- 파랑 `z` · 초록 `h` · 주황 `p`. **presence 가 문턱 아래인 튜블릿은 빼고 평균**하고, 남은 clip 이 30 % 아래면 선을 끊는다
- 주황 세로 점선 = `p` 가 물체를 절반 아래로만 들고 있게 되는 자리
- **x 는 clip 자신의 문맥 끝 수평 속도 `v0x` 로 나눈다** (속도칸 7 개가 한 곡선으로 모인다), **y 는 px 그대로** (낙하는 수평 속도와 무관하다)
- 좌표는 training_v8 에서 잰 **치우침을 뺀 값** (`exp_results/presence/attn_bias_px.json`)

## 읽는 법 — `z`·`h` 가 먼저다

`z`·`h` 는 창 전체를 보므로 **자의 천장**이다. 두 줄이 진실을 따라가면 그 창에서 자가 통한 것이다.
16 창 전수 (`../../exp_results/windows/WINDOW_SUMMARY.md`):

| 표현 | '없다' 오작동 | 위치 오차 |
|---|---|---|
| `z` | 0.0 ~ 5.7 % | 0.42 ~ 0.74 칸 |
| `h` | 0.1 ~ 4.1 % | 0.39 ~ 0.76 칸 |

→ **창을 타지 않는다.**

⚠️ **`ledge` 행의 후반부는 조심해서 읽는다** — 새 자는 떨어지는 구간 (t9~t14) 에서 `z`·`h` 도 '없다' 를 8~10 % 내서
평균이 살아남은 clip 쪽으로 치우친다 (`../README.md` §2).
⚠️ **가장자리 두 칸은 빼고 읽는다** — 총 창 48f 이상이면 마지막 튜블릿, 문맥 C4 면 첫 미래 튜블릿.
⚠️ 속도 열 (3·4 열) 에는 ±1 SD 띠가 없다 — 위치 오차 0.5 칸이 차분으로 2~3 px 로 증폭돼 띠가 패널을 덮는다. 평균선만 읽는다.

## 재현

```bash
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
for c in 4 8 16 32; do for p in 4 8 16 32; do
  $P z_research/scripts/figures/plot_rollout3_profile.py --ctx $c --prd $p
  $P z_research/scripts/figures/plot_rollout3_windows.py --ctx $c --prd $p
done; done
```
