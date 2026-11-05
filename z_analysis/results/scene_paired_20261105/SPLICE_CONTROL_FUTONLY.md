> ⚠️ 정정 (2026-11-05) — 아래 McNemar 표의 고침 / 깨짐은 **보드 전체 장면** (움직임 + 정지: 200 / 408 / 432) 의 짝에서 셌다. 같은 표의 "A moving" · "obj-region A moving" 정답률 칸은 움직임 장면만이라 둘이 맞지 않는다. 움직임 장면만으로 다시 센 값 (양방향 · 복사 · 먼 배경 포함) 은 `REGION_MOVING_BOTHDIR.md` (z_research/scripts/analysis/region_moving_tables.py).

# Splice-signature control: future-only target (released family) · block target (AR)

splice 서명 대조 표 (2026-11-05, tak's_story 리뷰어 요청). GPU 불필요.
가설: CF (negimp) 손실이 '창 전체 정답' 에 보이는 이음 (raw 48) 의 흔적에서 멀어지는 법을 배웠을 수 있다.
검사: 릴리즈 계열을 **미래만 정답** (target encoder 가 미래 16 장 = raw 48–93 만 본다 → 이음 안 보임) 으로 다시 채점.
  AR 판은 채점 정답이 원래 블록별 (블록 = 자기 2 장, 첫 미래 블록 = raw 48 · 51) 이라 이미 이음을 안 본다 → 기존 채점 그대로.
  복사 기준선 = 문맥 16 장만 넣은 pass 의 마지막 tubelet 을 미래 8 tubelet 에 (미래만 정답 공간).
D = 장면 1[m_A + m_B > 0] (= scene_paired_tests 의 C), J = 두 짝 모두 정답, c = 치우침 (양수 = 빈 미래가 싸다).
물체 영역 = 미래 블록마다 물체 투영 중심 3×3 (region_split_tests 와 같은 정의).

## LateOcc-IntPhys (Surrogate) — 200 scenes

| row | A vanish | B appear | c | **D** [95 %] | J | A moving | obj-region A moving | obj-region D |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| released (untouched) · future-only | 56.0 | 76.5 | +0.28 | **90.5** [85.6, 93.8] | 35.5 | 69.0 (n 100) | 0.0 | 93.5 |
| released + L1 2k · future-only | 73.0 | 88.0 | +0.28 | **95.5** [91.7, 97.6] | 62.0 | 74.0 (n 100) | 10.0 | 96.5 |
| released + CF 2k · future-only | 76.0 | 93.5 | +0.40 | **97.5** [94.3, 98.9] | 70.0 | 95.0 (n 100) | 90.0 | 98.5 |
| copy baseline · future-only | 77.0 | 42.5 | -0.46 | **76.0** [69.6, 81.4] | 25.0 | 89.0 (n 100) | 6.0 | 67.0 |
| released + L1 2k · full-window (ref) | 52.0 | 100.0 | +1.38 | **98.0** [95.0, 99.2] | 52.0 | 4.0 (n 100) | 1.0 | 95.5 |
| released + CF 2k · full-window (ref) | 96.0 | 96.5 | +0.03 | **100.0** [98.1, 100.0] | 92.5 | 94.0 (n 100) | 87.0 | 98.5 |
| AR + L1 2k · block target (splice-blind) | 98.0 | 81.5 | -0.56 | **95.5** [91.7, 97.6] | 80.0 | 96.0 (n 100) | 0.0 | 88.5 |
| AR + CF 2k · block target (splice-blind) | 87.5 | 88.0 | +0.01 | **97.5** [94.3, 98.9] | 76.5 | 75.0 (n 100) | 8.0 | 96.5 |

| control → CF | A fix/brk (p) | D fix/brk (p) | obj-region A fix/brk (p) | obj-region D fix/brk (p) |
|---|---|---|---|---|
| released L1 → CF (future-only) | 29/23 (0.49) | 5/1 (0.22) | 80/5 (1.8e-18) | 7/3 (0.34) |
| released L1 → CF (full-window ref) | 90/2 (1.7e-24) | 4/0 (0.12) | 86/0 (2.6e-26) | 6/0 (0.031) |
| AR L1 → CF (block target) | 0/21 (9.5e-07) | 4/0 (0.12) | 8/0 (0.0078) | 16/0 (3.1e-05) |

## LateOcc-Syn panel (VOB v3) — 408 scenes

| row | A vanish | B appear | c | **D** [95 %] | J | A moving | obj-region A moving | obj-region D |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| released (untouched) · future-only | 55.6 | 81.1 | +0.37 | **97.3** [95.2, 98.5] | 37.0 | 17.2 (n 204) | 22.1 | 96.8 |
| released + L1 2k · future-only | 76.5 | 96.3 | +0.53 | **99.0** [97.5, 99.6] | 72.8 | 54.9 (n 204) | 57.4 | 99.5 |
| released + CF 2k · future-only | 83.1 | 98.5 | +0.60 | **99.0** [97.5, 99.6] | 81.6 | 67.2 (n 204) | 67.2 | 98.0 |
| copy baseline · future-only | 75.2 | 74.0 | -0.02 | **92.4** [89.4, 94.6] | 49.8 | 56.4 (n 204) | 43.1 | 81.9 |
| released + L1 2k · full-window (ref) | 70.8 | 100.0 | +1.24 | **99.8** [98.6, 100.0] | 70.8 | 47.5 (n 204) | 49.0 | 99.3 |
| released + CF 2k · full-window (ref) | 80.4 | 100.0 | +1.09 | **100.0** [99.1, 100.0] | 80.4 | 64.2 (n 204) | 61.8 | 98.3 |
| AR + L1 2k · block target (splice-blind) | 99.0 | 100.0 | +0.37 | **100.0** [99.1, 100.0] | 99.0 | 98.0 (n 204) | 38.2 | 100.0 |
| AR + CF 2k · block target (splice-blind) | 85.0 | 100.0 | +1.00 | **100.0** [99.1, 100.0] | 85.0 | 70.1 (n 204) | 47.1 | 100.0 |

| control → CF | A fix/brk (p) | D fix/brk (p) | obj-region A fix/brk (p) | obj-region D fix/brk (p) |
|---|---|---|---|---|
| released L1 → CF (future-only) | 28/1 (1.1e-07) | 1/1 (1) | 25/3 (2.7e-05) | 2/8 (0.11) |
| released L1 → CF (full-window ref) | 44/5 (7.6e-09) | 1/0 (1) | 32/0 (4.7e-10) | 1/5 (0.22) |
| AR L1 → CF (block target) | 0/57 (1.4e-17) | 0/0 (1) | 18/0 (7.6e-06) | 0/0 (1) |

## LateOcc-Syn mover (VOB v3) — 432 scenes

| row | A vanish | B appear | c | **D** [95 %] | J | A moving | obj-region A moving | obj-region D |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| released (untouched) · future-only | 58.3 | 96.1 | +0.77 | **98.8** [97.3, 99.5] | 54.4 | 29.6 (n 216) | 56.0 | 99.1 |
| released + L1 2k · future-only | 93.5 | 98.1 | +0.28 | **99.5** [98.3, 99.9] | 91.7 | 90.3 (n 216) | 90.7 | 100.0 |
| released + CF 2k · future-only | 91.0 | 99.8 | +0.68 | **100.0** [99.1, 100.0] | 90.7 | 91.7 (n 216) | 88.0 | 100.0 |
| copy baseline · future-only | 81.5 | 45.8 | -0.50 | **87.7** [84.3, 90.5] | 30.3 | 88.0 (n 216) | 94.9 | 59.5 |
| released + L1 2k · full-window (ref) | 73.6 | 99.5 | +0.95 | **100.0** [99.1, 100.0] | 73.1 | 57.9 (n 216) | 80.6 | 100.0 |
| released + CF 2k · full-window (ref) | 88.0 | 99.8 | +0.77 | **100.0** [99.1, 100.0] | 87.7 | 89.4 (n 216) | 87.5 | 100.0 |
| AR + L1 2k · block target (splice-blind) | 92.6 | 94.7 | +0.08 | **99.8** [98.7, 100.0] | 87.3 | 85.6 (n 216) | 48.1 | 99.8 |
| AR + CF 2k · block target (splice-blind) | 99.8 | 72.5 | -1.05 | **99.8** [98.7, 100.0] | 72.5 | 99.5 (n 216) | 92.6 | 98.6 |

| control → CF | A fix/brk (p) | D fix/brk (p) | obj-region A fix/brk (p) | obj-region D fix/brk (p) |
|---|---|---|---|---|
| released L1 → CF (future-only) | 18/29 (0.14) | 2/0 (0.5) | 12/36 (0.00072) | 0/0 (1) |
| released L1 → CF (full-window ref) | 90/28 (8.9e-09) | 0/0 (1) | 28/30 (0.9) | 0/0 (1) |
| AR L1 → CF (block target) | 31/0 (9.3e-10) | 0/0 (1) | 96/0 (2.5e-29) | 1/6 (0.12) |

## LateOcc-Real (95) — 95 scenes

| row | A vanish | B appear | c | **D** [95 %] | J | A moving | obj-region A moving | obj-region D |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| released (untouched) · future-only | 22.1 | 81.1 | +0.81 | **70.5** [60.7, 78.8] | 5.3 | 22.1 (n 95) | — | — |
| released + L1 2k · future-only | 32.6 | 82.1 | +0.68 | **75.8** [66.3, 83.3] | 17.9 | 32.6 (n 95) | — | — |
| released + CF 2k · future-only | 55.8 | 71.6 | +0.21 | **80.0** [70.9, 86.8] | 33.7 | 55.8 (n 95) | — | — |
| copy baseline · future-only | 32.6 | 77.9 | +0.60 | **62.1** [52.1, 71.2] | 16.8 | 32.6 (n 95) | — | — |
| released + L1 2k · full-window (ref) | 43.2 | 76.8 | +0.45 | **62.1** [52.1, 71.2] | 28.4 | 43.2 (n 95) | — | — |
| released + CF 2k · full-window (ref) | 55.8 | 66.3 | +0.14 | **68.4** [58.5, 76.9] | 30.5 | 55.8 (n 95) | — | — |
| AR + L1 2k · block target (splice-blind) | 49.5 | 80.0 | +0.42 | **80.0** [70.9, 86.8] | 31.6 | 49.5 (n 95) | — | — |
| AR + CF 2k · block target (splice-blind) | 71.6 | 71.6 | +0.00 | **84.2** [75.6, 90.2] | 45.3 | 71.6 (n 95) | — | — |

| control → CF | A fix/brk (p) | D fix/brk (p) | obj-region A fix/brk (p) | obj-region D fix/brk (p) |
|---|---|---|---|---|
| released L1 → CF (future-only) | 28/6 (0.0002) | 10/6 (0.45) | — | — |
| released L1 → CF (full-window ref) | 19/7 (0.029) | 16/10 (0.33) | — | — |
| AR L1 → CF (block target) | 21/0 (9.5e-07) | 6/2 (0.29) | — | — |
