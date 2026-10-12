# v11 probing figures

전부 `z_research/scripts/figures/plot_v11_probing.py` 하나가 만든다.
`predictions.json` + `index_probe.csv` 에서 재계산하고 **324칸을 `summary.json` 과
대조해 불일치 0 일 때만** 그린다 (실패하면 그림을 안 만들고 죽는다).

```bash
python z_research/scripts/figures/plot_v11_probing.py
python z_research/scripts/figures/plot_v11_probing.py --width 7.0   # 논문 double-column
```

head 는 **조건(6)** 으로 학습됐고 k 는 학습축이 아니다. `by_k/` 는 클립별 예측을
index 와 조인해 **사후에** 쪼갠 것이라 재학습이 없다.

---

## 핵심 4개 — 이 순서가 논증이다

| # | 파일 | 말하는 것 |
|---|---|---|
| 1 | `by_condition/fig_information_vs_scoring` | **정보는 있는데 채점이 못 읽는다.** 같은 클립·같은 토큰에서 위는 probing `p` self, 아래는 채점 sensitivity. `ramp+OCC` shape 은 98 vs 4.5 |
| 2 | `by_condition/fig_transfer_vis_occ` | 비가림으로 배운 readout 이 **같은 팔의 가림 조건**에서 안 통한다. 낙차가 `z`·`h` 보다 `p` 에서 훨씬 크다 (colour: z 97.4 / h 99.2 / p 17.3) |
| 3 | `by_k/fig_transfer_vis_occ_k` | 그 실패가 k 와 함께 커진다. self 는 k 에 평평하므로 **정보 손실이 아니다** |
| 4 | `by_k/fig_confusion_transfer_k` | 실패의 끝은 단일 클래스 붕괴. colour `ramp` 는 고유 클래스 7 → 5 → 1 → 1 |

`by_condition/fig_confusion_transfer` 가 2번의 기전을 confusion 으로 보여준다
(같은 미래 구간인데 `h` 는 100.0, `p` 는 16.4).

## 부록

| 파일 | 용도 |
|---|---|
| `by_condition/fig_self_condition` | 조건 6 × z/h/p 전수. 54칸 중 48칸이 천장이라 **대조로 읽을 수 있는 건 `p` 줄뿐** |
| `by_condition/fig_transfer_matrix` | 6×6 이식 전수 × 9. 팔을 가로지르는 칸은 attention query 가 위치 특이적이라 **깨지는 게 기대값**이다. 수치는 `Archive/surprising_score/RESULTS_2026-08-30.md` §8 표에 있다 |
| `by_k/fig_self_k` | self 가 k 에 평평하다 (3번의 대조군) |

## 지웠고 다시 만들지 말 것

| 파일 | 이유 |
|---|---|
| `fig_confusion_self` (sta+OCC self) | 그 두 `p` head 가 54개 중 수렴이 안 된 셋 중 둘이다 (train_acc 0.931 / 0.954). 오분류를 표현 탓으로 못 읽는다. 그리고 `sta+OCC` 는 채점이 오히려 살아남는 조건이라(vanish sens +20.5) 논문의 어려운 케이스가 아니다 |
| `fig_self_arm_k` (팔 × k × 9패널) | 9패널 중 7개가 전 구간 100. 신호 두 패널은 `fig_self_k` 와 3번이 이미 담는다 |

## 쓰면 안 되는 표현

- ❌ **"표현이 회전했다 / 이동했다"** — 각도도 부분공간 겹침도 중심 이동도 안 쟀다.
  probe 가 선형이 아니라 attentive pooling + 선형이라, 실패가 클래스를 가르는
  방향에서 온 건지 **어느 토큰을 모으느냐**에서 온 건지 구분되지 않는다.
  ✅ 쓸 수 있는 것: "비가림으로 학습한 head 가 같은 팔의 가림 조건에서 못 읽는다."
- ❌ "z 와 h 가 같다" — 둘 다 천장(train_acc 1.000 × 18/18)이라 잴 해상도가 없다.
- ❌ "팔을 가로지르는 이식 실패는 표현이 조건별이라는 증거" — 기대값이다 (부록 표 참고).
