# IntPhys 1 · skip2_w32 (88.89 칸) — 창마다 어떤 프레임이 어디로 가나 (2026-09-25)

V-JEPA 2 ViT-H IntPhys 1 dev **88.89** 는 Garrido sliding 격자의 `skip2_w32` 칸을 Filtered 규칙으로 채점한 값이다 (`../../PROTOCOLS.md`).
이 폴더의 그림은 그 칸에서 **raw 프레임이 context encoder / target encoder / 채점에 어떻게 배정되는지**를 보인다.
창은 채점기와 같은 생성기 `evals/world_model_analysis/eval.py::_intphys1_windows` 로 만들었다 (손으로 다시 짜지 않았다).

## 구성 (스크립트가 assert 로 확인)

| 항목 | 값 |
|---|---|
| 원본 | 영상당 100 프레임 (`scene_001.png` … `scene_100.png`, 0-기준 raw 0–99) |
| frame skip 2 (`frame_budget: official`) | raw 0, 2, …, 96 의 49 장. **raw 97–99 는 버려진다** (공식 구현의 `99 // skip`) |
| 창 | 솎은 프레임 **32 장** 연속 = raw 폭 64 (예: raw 0, 2, …, 62) |
| 시작점 | 솎은 격자에서 stride 2 → **raw 0, 4, 8, …, 32 의 9 개** |
| 문맥 길이 C | {4, 8, 12, 16, 20} 장 (= context_mult {2,4,6,8,10} × 32/16), 예측 M = 32 − C |
| target encoder | **창 32 장 전부** (context + future) 를 한 번에 받는다. 채점은 그 출력의 future 토큰에 affine 없는 LN |
| context encoder | 앞 **C 장** 의 토큰만 (masks 로 future 토큰을 transformer 전에 뺀다) |
| predictor | context 출력으로 뒤 M 장 (M/2 튜블릿) 을 예측 → LN(target) 의 같은 자리와 L1 |
| 영상 점수 | 시작점마다 5 개 C 중 **최소** surprise (Filtered) → 9 시작점 평균 (AvgSurprise) |
| 튜블릿 | 연속 2 장 = 시간 토큰 1 개 (32 장 → 16 튜블릿 × 256 공간 토큰) |

## 그림

- **`window_map_skip2_w32.png`** — 행 = 45 창 (시작 9 × C 5), 열 = raw 프레임 0–99.
  파랑 = context (context encoder 와 target encoder 둘 다 받는다), 주황 = future (predictor 가 예측하고 target encoder 가 받는다; 채점 구간),
  회색 = 창 폭 안이지만 skip 2 로 빠진 홀수 raw 프레임 (어느 encoder 도 안 받는다). 점선 오른쪽 raw 97–99 는 어떤 창에도 안 들어간다.
- **`frames_skip2_w32_O1_01_2.png`** (possible) / **`frames_skip2_w32_O1_01_1.png`** (impossible) — 한 쌍 (O1, 이동 + 가림).
  행 = 시작점 9 개, 열 = 그 창이 받는 32 장 (숫자 = raw 인덱스, 두 장씩 틈 = 튜블릿).
  굵은 세로선 = 그 시작점에서 Filtered 가 고른 C (`../../exp_results/intphys1_sliding__intphys1_dev_vith_w32/per_window.json` 의 창 surprise 에서 min 을 다시 계산),
  아래 가는 눈금 = 나머지 C 후보. ▼ = pos/imp PNG 가 처음 갈리는 raw 46 (PNG 직접 비교, `pairs.csv` 의 `first_div_sensitive` 47 (1-기준) 과 같다) 이후 첫 프레임.
  이 쌍에서 고른 C: possible 16,16,16,12,12,8,8,4,4 / impossible 16,16,16,12,12,16,8,4,4 (시작 raw 0→32).

읽을 점:
- 사건 (raw 46) 이 **context 안에 들어가는 창** 이 있다 (시작 raw 20 · C 16 이면 context 가 raw 20–50). 그 창에서 predictor 는 이미 위반을 본 뒤 예측한다.
  Filtered 의 min 이 이런 창을 고를 수 있다 (이 쌍 impossible 의 시작 raw 20 은 C* = 16).
- target encoder 는 창 전체를 양방향으로 보므로, 사건 **전** future 튜블릿의 표적에도 사건 뒤 프레임 정보가 번진다 (`auto_research/Archive/H6_*` §3-4).

## 재현

```bash
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
python z_research/scripts/figures/plot_intphys1_windows.py --pos O1_01_2 --imp O1_01_1    # CPU, vll5 (프레임 /local_datasets)
```
