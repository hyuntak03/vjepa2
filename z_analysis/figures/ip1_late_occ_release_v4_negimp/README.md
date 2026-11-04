# IntPhys 1 late occ — 릴리즈 + v4 이어 학습 vs + 음의 손실, pixel decoder 그림 (2026-11-02)

열: 원본 (가능 영상) · 릴리즈 + v4 이어 학습 · 릴리즈 + v4 + 음의 손실. 앞 16 장 = 문맥 (원본), 뒤 16 장 = 각 모델 예측을 decoder 로 그린 것.
머리줄 correct / wrong = 표준 채점 (시트 Object_permanence 와 같은 판정). A = 사라짐 짝 (문맥에서 물체를 봤다), B = 나타남 짝.
⚠️ decoder 는 블록별 정답 latent 로 학습했다. 릴리즈 계열 predictor 의 출력은 32 장 창 공간이라 흐릿하게 나올 수 있다.

## moving_A (10 쌍) — 맞힌 쌍: v4 3 · v4 + 음의 손실 9
- `moving_A/O1_01_pair1.gif` — v4 wrong · 음의 손실 correct
- `moving_A/O1_23_pair1.gif` — v4 wrong · 음의 손실 correct
- `moving_A/O1_24_pair1.gif` — v4 correct · 음의 손실 correct
- `moving_A/O1_26_pair1.gif` — v4 wrong · 음의 손실 correct
- `moving_A/O1_27_pair1.gif` — v4 correct · 음의 손실 correct
- `moving_A/O1_29_pair2.gif` — v4 wrong · 음의 손실 correct
- `moving_A/O1_04_pair1.gif` — v4 wrong · 음의 손실 wrong
- `moving_A/O1_05_pair1.gif` — v4 wrong · 음의 손실 correct
- `moving_A/O1_06_pair2.gif` — v4 correct · 음의 손실 correct
- `moving_A/O1_09_pair2.gif` — v4 wrong · 음의 손실 correct

## moving_B (10 쌍) — 맞힌 쌍: v4 7 · v4 + 음의 손실 8
- `moving_B/O1_01_pair2.gif` — v4 correct · 음의 손실 correct
- `moving_B/O1_23_pair2.gif` — v4 correct · 음의 손실 wrong
- `moving_B/O1_24_pair2.gif` — v4 wrong · 음의 손실 correct
- `moving_B/O1_26_pair2.gif` — v4 wrong · 음의 손실 correct
- `moving_B/O1_27_pair2.gif` — v4 correct · 음의 손실 wrong
- `moving_B/O1_29_pair1.gif` — v4 correct · 음의 손실 correct
- `moving_B/O1_04_pair2.gif` — v4 correct · 음의 손실 correct
- `moving_B/O1_05_pair2.gif` — v4 correct · 음의 손실 correct
- `moving_B/O1_06_pair1.gif` — v4 wrong · 음의 손실 correct
- `moving_B/O1_09_pair1.gif` — v4 correct · 음의 손실 correct

## static_A (5 쌍) — 맞힌 쌍: v4 3 · v4 + 음의 손실 2
- `static_A/O1_14_pair2.gif` — v4 correct · 음의 손실 correct
- `static_A/O1_18_pair1.gif` — v4 wrong · 음의 손실 wrong
- `static_A/O1_03_pair2.gif` — v4 correct · 음의 손실 correct
- `static_A/O1_22_pair2.gif` — v4 correct · 음의 손실 wrong
- `static_A/O1_08_pair2.gif` — v4 wrong · 음의 손실 wrong

## static_B (5 쌍) — 맞힌 쌍: v4 5 · v4 + 음의 손실 5
- `static_B/O1_14_pair1.gif` — v4 correct · 음의 손실 correct
- `static_B/O1_18_pair2.gif` — v4 correct · 음의 손실 correct
- `static_B/O1_03_pair1.gif` — v4 correct · 음의 손실 correct
- `static_B/O1_22_pair1.gif` — v4 correct · 음의 손실 correct
- `static_B/O1_08_pair1.gif` — v4 correct · 음의 손실 correct

## 재현

```bash
CUDA_VISIBLE_DEVICES=0 /venv/vjepa/bin/python z_pixel_decoder/render_compare_ip1late.py
```
