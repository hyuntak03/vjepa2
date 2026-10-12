# RollOut v3 — 16 창 4×4 격자 GIF (`<scenario>.gif`)

**자와 무관하다.** 원본 clip 을 창마다 잘라 튼 것뿐이라, 창 설계가 무엇을 보여 주는지 눈으로 확인하는 용도다.

```
열 = 예측 길이 P ∈ {4,8,16,32}      행 = 문맥 길이 C ∈ {4,8,16,32}
사건은 f32.  문맥 [32−C, 32) · 예측 [32, 32+P)
각 칸은 자기 창만 튼다 — 창 전에는 첫 프레임, 끝난 뒤에는 **마지막 예측 프레임에 멈춘다**
테두리: 문맥 파랑 · 예측 주황 · 멈춤 회색
```

`wall`·`ledge` 는 가능·불가능이 **f32 까지 픽셀 단위로 같고** (실측 0.000 px) f33 부터 갈라진다.
가능 프레임 위에 불가능 위치를 빨간 테두리로 얹고, 칸마다 그 창의 **최대 분리량**을 px 로 적었다.
**자 한 칸 = 18 px** 이라 그보다 작은 창은 라벨이 빨갛다 = 자로는 판정 불가.

| 시나리오 | P=4 | P=8 | P=16 | P=32 |
|---|---:|---:|---:|---:|
| `wall` (통과 vs 정지) | 12.3 px | 28.7 | 61.6 | 127.5 |
| `ledge` (부유 vs 낙하) | **0.9 px** | **5.0** | 23.1 | 45.5 |

(속도 8.25 px/2f clip 기준)

→ **`ledge` 는 짧은 창에서 원리적으로 못 읽는다.** 낙하가 t² 이라 초반 분리가 1 px 도 안 된다.
길게 잡아도 presence 문턱이 바로 그 초반 칸만 남기므로, `ledge` 판정은 위치 자가 아니라
**자 없는 잠재 비교** (`|p − h_pos|` vs `|p − h_imp|`) 로 해야 한다.

## 재현

```bash
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
$P z_research/scripts/figures/plot_rollout3_windows_gif.py
$P z_research/scripts/figures/plot_rollout3_windows_gif.py --scenarios wall ledge --speed 10.5 --panel 200
```
