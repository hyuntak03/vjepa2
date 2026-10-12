# RollOutV3 그림

> 시작점은 [`../README.md`](../README.md) 다. 수치의 정본은 [`../Archive/IDENTITY_R8_2026-09-26.md`](../Archive/IDENTITY_R8_2026-09-26.md) (자로 읽은 것) 와 [`../Archive/RESULTS_2026-09-25.md`](../Archive/RESULTS_2026-09-25.md) §6–7 (자 없는 것) 이다.
> 2026-09-26 에 옛 `new_archive/` 를 이 폴더로 합쳤다. 옛 `new_archive/_audit/` 는 [`../audit/`](../audit/) 다. 호환 링크는 두지 않는다.
> 그리는 스크립트 이름 `new_archive_redraw.py` 는 명령줄 호환 때문에 그대로 둔다.

## 폴더

| 폴더 | 무엇 | 자 (`_decoder.json`) | 그리는 스크립트 (`z_research/scripts/figures/`) |
|---|---|---|---|
| **`train_readout/`** | **자 자신을 학습셋 held-out test 에서 — 맨 먼저 본다.** `fig_train_summary` (논문 판: 위치 L2 오차 · 57-class 정확도, p · z · h) · `fig_train_overview` (ROC · 정체 · 위치 · 혼동) · `_by_tubelet` · `_by_family` · `_error_map` (화면 위치별) · `_confusion` (모양 · 색) | identity_r8 | `plot_train_readout.py` (정체 자만, 입력 = 자의 `preds.npz`) |
| `v3_signed_error/` | v3 16 창, 축마다 앞섬 (+) / 뒤처짐 (−). p · z 를 각자 자로 | identity_r8 `01c3afaac37f` | `plot_v3_l2_error.py --signed` |
| `v3_signed_error_hhead/` | 같은 것을 h 자로 본 p (`p@h`) · z (`z@h`) | identity_r8 | `plot_v3_l2_error.py --signed --head h` |
| `v3_trajectory/` | 마지막 관측 자리에서 옮긴 양 | identity_r8 | `plot_v3_trajectory.py` |
| `v11_presence/` | v11 p 미래에 물체가 있나 — 튜블릿마다 '있다' 로 읽힌 **clip 비율 (%)** (`fig_v11_presence.png`), k 별 (`fig_v11_presence_by_k.png`). 2026-09-27 에 과반 이진 판에서 바꿨다 (이진 판은 `_superseded/binary_majority_2026-09-27/`). **`fig_v11_presence_identity` (논문 판, 3 × 3)**: 행 = 가림 없음 · 문맥 중간 가림 · 문맥 끝 가림, 열 = '있음' 이면서 모양 유지 · 색 유지 · 둘 다 유지 (분모 = 물체 clip 전부, 점선 = '있음' 전체). **`fig_v11_acc57` (논문 판, 3 × 2)**: 행 = 가림 없음 · 문맥 중간 가림 · 문맥 끝 가림, 열 = 57-class 정확도 · '없음' 을 고른 비율 (p 주황 · 운동은 선 모양, target encoder h 파랑 파선) | identity_r8 | `plot_v11_presence.py [--by-k]` |
| `v11_signed_error/` | v11 '있음' 칸의 앞섬 / 뒤처짐. `fig_visible` · `fig_last_k{1..4}` · `fig_mid_k{1..4}` | identity_r8 | `plot_v11_signed_error.py` |
| `v11_readout_overlay/` | v11 '있음' 칸의 읽힌 위치를 빈 장면 프레임 위에 (k=0–4) | identity_r8 | `plot_v11_readout_overlay.py` |
| **`context_to_future/`** | 문맥 운동 (속도 · 가속 · 시작 위치) · 외형 (모양 · 색 · 배경) 이 p 의 미래를 바꾸나 + 정체 유지 (v3 16 창 · v11). 먼저 볼 것 `1_motion` · `2_look` · `3_identity` · `4_motion_windows` (폴더 README 에 읽는 법), 계수 판 `detail_*` | identity_r8 | `plot_context_to_future.py` |
| `v11_probe/` | 정체 선형 probe (평균 풀링). `PROBE.md` · `results.json` | **자 없음** | `analysis/v11_pooled_probe.py` |
| `v11_geometry/` | 풀링 표현 기하. `GEOMETRY.md` · `results.json` | **자 없음** | `analysis/v11_pooled_geometry.py` |
| `v3_gif/` | 영상 위 진실 · p · h (삽화, 주장 없음) | presence (옛 자, 다시 안 그림) | `plot_v3_decoder_gif.py` |

- 자로 그린 여덟 폴더는 한 명령으로 다시 그린다.

  ```bash
  /data/hyuntak/anaconda3/envs/vjepa2/bin/python z_research/scripts/figures/new_archive_redraw.py --decoder <자> --steps figs
  ```

  - readings 의 자 지문이 자와 다르면 멈춘다.
  - 도장이 다른 옛 그림은 각 폴더의 `_superseded/<자>_<지문>/` 으로 자동으로 내려간다.
- 그림 제목에 자 이름이 찍힌다 (`decoder: position + identity (56 shape x colour + none)`).

## 각 폴더의 `_superseded/`

| 하위 폴더 | 무엇 |
|---|---|
| `identity_aeda55874c1c/` | 옛 identity 자 (옛 학습셋, v11 판 오탐 56–99 %) 로 그린 판 |
| `presence_decoder/` | presence 자 (`dcd24d8a8a47`) 로 그린 판 |
| `along_1row/` · `screen_sign/` | 앞섬 / 뒤처짐의 옛 배치 (운동 방향 축 한 줄 · 화면 부호) |
| `v11_presence/_superseded/*.png` · `values*.json` | presence 자 시기의 k 별 · 전체 판 |

⚠️ **`v11_presence/values.json` · `v3_signed_error/values.json` 은 지금 identity_r8 값이다.** 그때의 presence 자 값은 `_superseded/presence_decoder/` 에 있다. TrainingEffects (`te_analyze_{v3,presence}.py`, presence 자 사용) 의 sanity 대조는 2026-09-26 에 그 사본으로 옮겼다.

## `_superseded/` (이 폴더 최상위)

| 폴더 | 무엇 | 왜 물러났나 |
|---|---|---|
| `presence_era_2026-09-24/` | 옛 파이프라인 그림: `windows/` (창별 profile) · `windows_gif/` · `examples/` (법칙별 GIF) · `readout/` (자 자체 · recall) · `direction/` · `v11/` | presence 자로만 그렸다. 그 자는 v11 가림막을 물체로 읽었다. "t10 절벽" 은 새 자에서 나오지 않는다 (IDENTITY_R8 §4). `rollout3_paths.fig()` 가 지금도 이 자리를 가리킨다 (다시 돌리면 여기에 쓴다) |
| `v11_dh_gif/` | h(물체) − h(빈 장면) heat GIF | 탐색용이다. 결론에 쓰지 않았다 |

## 재현

```bash
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
$P z_research/scripts/figures/new_archive_redraw.py --decoder identity_r8 --steps figs   # 자 그림 8 폴더 (CPU 몇 분)
$P z_research/scripts/analysis/rollout3_doc_numbers.py --decoder identity_r8             # 문서 표 대조
```
`v11_probe/` · `v11_geometry/` 는 자와 무관하다 — `../Archive/RESULTS_2026-09-25.md` §재현.
