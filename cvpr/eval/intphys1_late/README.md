# eval/intphys1_late — IntPhys 1 가림 샘플, 위반 직전이 문맥 끝인 고정 창 (2026-10-17)

**무엇을 재나.** IntPhys 1 dev 에서 물체가 가려지는 샘플만 골라, 위반 시점 바로 앞을 마지막 관측 (문맥 끝) 으로 두고 그 뒤 16 장을 예측하게 한다. Garrido sliding (창 여러 개의 surprise 를 모아 채점) 과 달리 **창 하나**, 위반 직전에 맞춘 창이다.

## 정의
- **대상**: `group ∈ {moving/occluded, static/occluded}` = 90 쌍 (O1 · O2 · O3 각 30, 180 영상). 출처 `z_research/previous/OcclusionStats/intphys1/pairs.csv`.
- **위반 시점 d**: 가능 · 불가능 영상이 처음 픽셀로 달라지는 raw 프레임 (0-based, 15 fps). 180 쌍 모두 `first_div_sensitive − 1` 과 같다 (`OcclusionStats/Archive/INTPHYS1_OCCLUSION_2026-09-26.md`).
- **창**: 32 장, skip 2 (7.5 fps — Garrido IntPhys 1 최고 칸 skip2 와 같은 간격).
  - 문맥 16 장: e − 30, e − 28, …, e. 여기서 e = d − 1 은 pos · imp 가 같은 마지막 프레임이다.
  - 미래 16 장: e + 2, …, e + 32.
  - ⚠️ skip 2 라서 d 자체는 창에 안 들어간다 (미래 첫 장 = d + 1). d 를 문맥 끝에 넣으면 (`--ctx-end-offset 0`) pos · imp 문맥이 그 한 장에서 갈려 matched 쌍의 전제 (문맥 픽셀 일치) 가 깨진다.
  - 영상 범위 [0, 99] 밖은 끝 프레임으로 채운다 (clamp). 16 쌍이 해당하고 index 의 `clamp_lo` · `clamp_hi` 열에 장 수가 있다.
- **채점**: testbed 와 같다. surprise = mean |p − LN(h)[미래]|, matched pair, 문맥 16 + 미래 16, dual encoder (문맥 ← online, 타깃 ← EMA), fp32 + autocast fp16, mask_index 0.
- **감사**: 문맥 16 장이 90/90 쌍에서 픽셀 단위로 같다 (`data_csv/intphys1_late/context_integrity.json`).

## 결과 — release (`vith`, 2026-10-17)
쌍 정확도 % (chance 50). 복사 기준선은 예측 없이 마지막 문맥 튜블릿을 미래 8 칸에 복사한 것이다 (copy_z = 문맥 encoder 출력, copy_h = 타깃 encoder 출력). 같은 90 쌍을 Garrido sliding skip2_w32 로 채점한 값 (pairs.csv `correct_skip2_w32`) 을 같이 둔다.

| | n | release | copy_z | copy_h | sliding skip2_w32 (release) |
|---|---|---|---|---|---|
| **전체** | 90 | **85.6** | 71.1 | 77.8 | 83.3 |
| O1 (영속) | 30 | 83.3 | 66.7 | 73.3 | 76.7 |
| O2 (모양) | 30 | 93.3 | 80.0 | 86.7 | 96.7 |
| O3 (연속) | 30 | 80.0 | 66.7 | 73.3 | 76.7 |
| 움직이는 물체 · 가림 | 60 | 78.3 | 60.0 | 68.3 | 75.0 |
| 정지 물체 · 가림 | 30 | 100.0 | 93.3 | 96.7 | 100.0 |

release − copy (scene bootstrap 95 % CI, 4,000 회):
- release − copy_z: 전체 **+14.4 [+5.6, +23.3]**, 움직이는 물체 **+18.3 [+6.7, +30.0]**, 정지 물체 +6.7 [+0.0, +16.7].
- release − copy_h: 전체 +7.8 [+0.0, +16.7], 움직이는 물체 +10.0 [−3.3, +23.3], 정지 물체 +3.3 [+0.0, +10.0].

위반 종류 (`event_kind`, release):
- d 가 가림 구간 안: 91.4 (53/58)
- 가림이 d 직전에 끝남: 75.0 (9/12)
- d 전에 물체가 안 보임: 75.0 (15/20)

clamp 없는 74 쌍만 보면 release 82.4, sliding 79.7 이다.

단서:
- 90 쌍 · scene 45 개라 칸 (O1 · O2 · O3 × 움직임) 단위 차이는 잡음이 크다.
- 정지 물체는 천장 (100) 이다.
- 이 창은 위반 시점을 알고 고른 것이라 **기술적 (descriptive)** 측정이다. sliding 처럼 위반 시점을 모르는 채점과 같은 질문이 아니다.
- copy_h 는 미래 타깃과 같은 encoder 의 출력이라 copy_z 보다 유리한 기준선이다. release 가 copy_h 를 CI 로 넘는 것은 전체에서도 경계 (하한 0.0) 다.
- 문맥 ← online · 타깃 ← EMA 규약은 이 하네스도 testbed 와 같다. AR 모델은 문맥도 EMA 블록이라, 모델 사이 비교에는 encoder 차이가 섞인다 (메모리 `project-encoder-key-mismatch`).

### 왜 높은가 — 픽셀 변화량 누수 (2026-10-17)
미래 16 장이 문맥 끝 (d − 1) 에서 RGB 로 얼마나 바뀌는지 (mean |frame_f − frame_c16|) 를 pos · imp 로 비교했다. **90 쌍 중 74 쌍에서 불가능 미래가 더 많이 바뀐다.** 그래서 "더 많이 바뀐 쪽이 불가능" 이라는 규칙 (학습 · 예측 없음) 만으로 **82.2 %** 다.

| 쌍 | n | release | copy_z |
|---|---|---|---|
| 불가능 미래가 더 많이 바뀜 | 74 | 91.9 | 75.7 |
| 가능 미래가 더 많이 바뀜 | 16 | **56.2** | 50.0 |
| 움직이는 물체 · 불가능이 더 바뀜 | 46 | 87.0 | 65.2 |
| 움직이는 물체 · 가능이 더 바뀜 | 14 | **50.0** | 42.9 |
| 정지 물체 · 불가능이 더 바뀜 | 28 | 100.0 | 92.9 |

→ release 85.6 의 대부분은 이 누수 방향과 같은 쌍에서 나온다. 누수가 반대인 16 쌍에서는 chance 근처다 (16 쌍이라 CI 가 넓다).
틀린 13 쌍은 전부 움직이는 물체다. 그림 `samples_wrong.png` 를 보면 대부분 화면 멀리 있는 작은 물체이고, 문맥 안에서 물체가 안 보이는 쌍 (not_seen_before_d) 이 많다.
그림: `<결과 폴더>/figures/samples_{correct,wrong}.png` (`plot_samples.py`; 쌍마다 위 = 가능, 아래 = 불가능, 문맥 16 | 미래 16).

### 문맥 끝에서 물체를 못 보게 — `intphys1_late_hid` (2026-10-17)
d − 1 창에서는 **이동 물체 60 쌍 중 40 쌍이 문맥 마지막 토큰 (튜블릿 e−2, e) 에 물체가 보였다** (34 쌍 mask ≥ 20 px, 6 쌍 조각; `z_research/scripts/analysis/intphys1_late_ctx_end_visibility.py` → 결과 폴더 `ctx_end_visibility.json`). 이동 물체 가림이 2–6 원 프레임이라 e−2 에 아직 보인다.
→ `build.py --hidden-end`: 쌍마다 e 를 d 에 가장 가까운 '마지막 튜블릿 두 장에서 가능 영상 물체 mask 0 · 가능-불가능 RGB 차이 ≤ 20 px · 그 전에 물체를 본 적 있음' 지점으로. d 뒤 문맥 장 (원래 차이 ≤ 11 px) 은 가능 영상 프레임으로 맞춰 문맥 비트 일치 54/54.
54 쌍 (이동 29 · 정지 25), e − d ∈ {−1: 6, 0: 45, +1: 3}. 뺀 36 쌍 = 문맥에서 물체를 본 적 없음 20 · 조건 못 맞춤 16.

| | n | release | copy_z | copy_h | 미래 변화량 규칙 |
|---|---|---|---|---|---|
| 전체 | 54 | 87.0 | 77.8 | 87.0 | 79.6 |
| 이동 · 가림 | 29 | 75.9 | 58.6 | 79.3 | 69.0 |
| └ O1 / O2 / O3 | 6 / 16 / 7 | 16.7 / 87.5 / 100 | 33.3 / 68.8 / 57.1 | 50.0 / 100 / 57.1 | 16.7 / 75.0 / 100 |
| └ 변화량 방향 맞음 / 반대 | 20 / 9 | **95.0 / 33.3** | 60.0 / 55.6 | 80.0 / 77.8 | 100 / 0 |
| 정지 · 가림 | 25 | 100 | 100 | 96.0 | 92.0 |
- 물체를 못 보게 해도 점수가 높다. 그러나 **release = copy_h (87.0)** 이고, 이동 쌍에서 release 는 "불가능 미래가 RGB 로 더 많이 바뀌는" 방향을 따라간다 (맞음 95.0 / 반대 33.3). 높은 점수는 마지막 관측의 노출이 아니라 **미래 내용의 비대칭** 에서 온다.
- O1 (영속, 물체가 사라지거나 나타남) 은 물체를 못 보게 하면 16.7 (6 쌍) — 사라진 물체를 예측해 내지 못한다.
- 칸 n 이 6–16 이라 원리별 값은 잡음이 크다.

모델 셋 (2026-10-18, 같은 54 쌍, 쌍 정확도 %):
| 모델 | 전체 | 이동 O1 (6) | 이동 O2 (16) | 이동 O3 (7) | 정지 (25) | 그 채점 공간의 복사 기준선 (전체 · 이동 O1) |
|---|---|---|---|---|---|---|
| release (`vith`) | 87.0 | 16.7 | 87.5 | 100 | 100 | copy_h 87.0 · 50.0 |
| release + PFT (`vith_release_vobftg`, VOBench + gravity) | 92.6 | 33.3 | 100 | 100 | 100 | copy_h 87.0 · 50.0 (encoder 같음) |
| vast AR (`vith_vast_ar_state_ep70`) | 98.1 | **100** | 93.8 | 100 | 100 | copy_hblock 85.2 · 16.7 (블록별 LN(target) 공간, 결과 폴더 `copy_hblock.json`) |
| vast AR + PFT (`vith_vast_ar_state_ep70_vobftg`, VOBench + gravity) | 98.1 | 100 | 93.8 | 100 | 100 | 〃 (같은 공간). 쌍 판정은 vast AR 과 54/54 같고 영상별 surprise 는 다르다 (상관 0.917) |
- vast AR 은 채점 공간이 다르다 (문맥 · 타깃 = 블록별 target encoder) — 쌍 정확도만 비교한다. 같은 공간의 복사 대비 +12.9 (전체), 이동 O1 은 6/6 vs 복사 1/6.
- 그림: 결과 폴더 `figures/gallery_{moving,static}.png` (`plot_hid_gallery.py`) — 54 쌍 전부, 문맥 마지막 튜블릿 초록 테두리, 모델별 O/X.

### 결과가 미래에 드러나게 + IntPhys 1 렌더 잡음 — `intphys1_late_rev` · `intphys1_late_rev_dn1` (2026-10-18)
`_hid` 판은 **정지 25 쌍 중 17 쌍의 미래 16 장 안에서 가림막이 안 내려가 결과가 드러나지 않았다** (사용자 지적, 그림 `gallery_static.png`). 그런데도 release · copy_h 가 17/17 이었다.
**원인 — IntPhys 1 dev 렌더 잡음.** 화면 전체에 ±1 짜리 잡음이 있고 (하늘 띠 픽셀의 33 % 가 매 프레임 바뀜, |Δ| ≥ 2 는 0 %) **가능 영상의 잡음은 20 프레임 주기로 되풀이된다**
(하늘 띠 |frame(d+19) − frame(d−1)|: 가능 0.01–0.03 · 불가능 0.18–1.33). 불가능 영상은 d (magic tick) 에서 이 주기가 끊긴다 — d 전에는 가능 영상과 비트 단위로 같고, d 뒤에는 하늘 픽셀의 58 % 가 ±1 다르다.
그래서 문맥이 20 프레임보다 길면 가능 미래는 문맥 장을 픽셀 그대로 되풀이하고 불가능 미래는 그러지 않는다 → 예측 없이 (복사로도) 불가능 쪽 surprise 가 크다 (정지 · 하늘 띠 기준 불가능 > 가능 30/30 쌍).
⚠️ 같은 잡음은 표준 IntPhys 1 채점 (Garrido sliding, 창 64 원 프레임) 에도 들어 있다 — 그 점수에 얼마나 기여하는지는 아직 안 쟀다.

- `--reveal-end`: 결과가 처음 눈에 보이는 프레임 r (가능 · 불가능 차이 > 10 인 픽셀이 50 px 를 넘는 첫 장) 직전의 가려진 지점을 문맥 끝으로 → 미래 첫 1–2 장에 결과가 드러남 (r − e: 2 → 38 쌍, 1 → 12, 3 → 2, 10 → 2). 54 쌍, clamp 25 쌍.
- `--denoise 1`: 불가능 미래 장에서 가능 영상과 |Δ| ≤ 1 인 픽셀을 가능 값으로 (잡음만 지우고 물체 · 가림막 차이는 둔다).

| 판 (54 쌍) | 모델 | 전체 | 이동 O1 (6) | 이동 O2 (16) | 이동 O3 (7) | 정지 (25) |
|---|---|---|---|---|---|---|
| `_rev` (잡음 그대로) | release | 83.3 | 16.7 | 87.5 | 100 | 92.0 |
| | release + PFT | 88.9 | 50.0 | 100 | 100 | 88.0 |
| | vast AR · vast AR + PFT | 92.6 | 100 | 93.8 | 100 | 88.0 |
| | copy_z / copy_h | 75.9 / 88.9 | 33.3 / 50.0 | 68.8 / 100 | 57.1 / 57.1 | 96.0 / 100 |
| **`_rev_dn1` (잡음 제거)** | release | **37.0** | 0.0 | 31.2 | 71.4 | 40.0 |
| | release + PFT | 42.6 | 0.0 | 43.8 | 85.7 | 40.0 |
| | vast AR | 44.4 | 16.7 | 56.2 | 14.3 | 52.0 |
| | vast AR + PFT | 46.3 | 16.7 | 56.2 | 28.6 | 52.0 |
| | copy_z / copy_h | 29.6 / 42.6 | 16.7 / 33.3 | 12.5 / 43.8 | 42.9 / 28.6 | 40.0 / 48.0 |
- 잡음을 지우면 **모든 모델이 chance 아래 ~ 근처** 다. `_hid` · `_rev` 의 높은 점수는 거의 전부 렌더 잡음 주기에서 왔다.
- chance 아래 (O1 0 %) 는 사라진 물체 쪽 미래 (빈 장면) 가 L1 로 더 싸서다 (appearance bias, v11 vanish 물체→빈 0 % 와 같은 방향).
- vast AR 은 채점 공간이 달라 (블록별 target) release 와 쌍 정확도로만 비교. 칸 n 이 6–25 라 잡음이 크다.
그림: `<ds>_vith/figures/gallery_{moving,static}.png` (`plot_hid_gallery.py --ds <ds>`).

### stride 를 골라 결과가 미래에 드러나게 (잡음 그대로) — 2026-10-18
사용자 지시: 잡음은 건드리지 말고 frame stride 로 결과가 미래에 보이게, VOBench (30 fps · stride 3 · 문맥 16 + 미래 16) 와 비슷하게. 문맥 끝 e 는 `--hidden-end` 규칙 (d 에 가장 가까운 '마지막 튜블릿 가림'), 결과 프레임 r 이 미래 안에 드는 쌍만.
| 판 | stride | 쌍 (이동 O1/O2/O3 · 정지) | release | +PFT | vast AR | vast AR+PFT | copy_z | copy_h |
|---|---|---|---|---|---|---|---|---|
| `intphys1_late_s3_auto3r16` | 3 고정 (5 fps, VOBench 와 같은 stride) | 22 (3/2/1 · 16) | 68.2 | 68.2 | 90.9 | 81.8 | 68.2 | 63.6 |
| `intphys1_late_auto2r16` | 2 고정 (7.5 fps, VOBench 10 fps 와 초 단위 최근접) | 37 (6/16/7 · 8) | 81.1 | 89.2 | 97.3 | 97.3 | 67.6 | 81.1 |
| `intphys1_late_auto2-6` | 쌍마다 2–6 (r 이 미래 8 장 안) | 50 (6/16/12 · 16) | 74.0 | 86.0 | 92.0 | 90.0 | 68.0 | 78.0 |
- stride 3 은 이동 쌍이 6 개뿐이다 — 이동 가림이 원본 2–6 프레임이라 튜블릿 두 장 (e−3, e) 을 다 가리는 쌍이 드물다. 정지 쌍은 미래가 원본 48 프레임을 덮어 결과가 더 많이 들어온다.
- 렌더 잡음 (위 절) 은 그대로라 이 점수에는 잡음 누수가 섞여 있다. 복사 기준선을 같이 읽는다.
- 그림: `<ds>_vith/figures/gallery_{moving,static}.png`.

### ★ 쌍마다 stride · 문맥 끝을 같이 고르기 — `intphys1_late_auto2-6r12fit` (2026-10-18, 시트 기록판)
⚠️ 정정: 위 `_auto*` 판들의 조건 (c) "물체를 본 적 있음" 은 **영상 처음부터** 검사했다 → stride 2 판의 정지 8 쌍은 문맥 16 장 안에서 물체가 한 번도 안 보였다 (사용자 지적). 조건을 '채점에 들어가는 문맥 장 안에서 보임' 으로 고쳤다 (그러면 stride 2 고정은 이동 30 쌍 · 정지 0 쌍).
`--auto-skip 2,3,4,5,6 --reveal-within 12 --fit`: stride 2–6 중 가장 작은 것, 그 stride 에서 (a) 마지막 튜블릿 가림 (b) 가능 · 불가능 차이 ≤ 20 px (c) 문맥 안에서 물체 보임 (d) 결과가 미래 12 장 안 을 모두 만족하는 e 중 d 에 가장 가까운 것 (e 범위는 가림 구간 전체). 잡음 그대로.
59 쌍 (이동 34: O1 6 · O2 16 · O3 12 / 정지 25), stride 2: 30 · 3: 23 · 4: 2 · 5: 4, clamp 30 쌍 (문맥 앞 · 미래 끝을 끝 프레임으로 채움).
| 모델 | 전체 | 이동 O1 | 이동 O2 | 이동 O3 | 정지 |
|---|---|---|---|---|---|
| release | 81.4 | 16.7 | 87.5 | 91.7 | 88.0 |
| release + PFT | 86.4 | 33.3 | 100 | 83.3 | 92.0 |
| vast AR | 91.5 | 100 | 93.8 | 75.0 | 96.0 |
| vast AR + PFT | 91.5 | 100 | 93.8 | 66.7 | 100 |
| copy_z / copy_h | 76.3 / 83.1 | 33.3 / 50.0 | 68.8 / 100 | 66.7 / 58.3 | 96.0 / 92.0 |
그림 `intphys1_late_auto2-6r12fit_vith/figures/gallery_{moving,static}.png`. 렌더 잡음 누수는 그대로 — 복사 기준선과 같이 읽는다.

## 명령
```bash
python cvpr/eval/intphys1_late/build.py                            # 데이터 (프레임 링크 · index · 감사). --skip · --ctx-end-offset 로 변형
python cvpr/eval/intphys1_late/build.py --hidden-end               # 문맥 끝 튜블릿에 물체가 안 보이는 판 → intphys1_late_hid (54 쌍)
DS=intphys1_late_hid GPU_IDS="0 1" bash cvpr/eval/intphys1_late/run.sh vith          # 그 판 채점 (COPY=1 이면 복사 기준선)
GPU_IDS="0 1" bash cvpr/eval/intphys1_late/run.sh vith              # 채점 (180 영상, GPU 2 장 몇 분)
COPY=1 CUDA_VISIBLE_DEVICES=0 bash cvpr/eval/intphys1_late/run.sh vith   # 복사 기준선 → 결과 폴더의 copy_baseline.json
```
레지스트리: `cvpr/registry/datasets.yaml` 의 `intphys1_late` (옛 resolver 용 사본 `configs/protocols/datasets.md`).
프레임 `/local_datasets/world/world_analysis/IntPhys1_late/<video_id>/000000..000031.png` (원본 IntPhys1_dev_frame_png 링크), index `data_csv/intphys1_late/index.csv`.

## 재현
- `python cvpr/eval/intphys1_late/build.py` → `data_csv/intphys1_late/{index.csv, context_integrity.json}`
- `GPU_IDS="0 1" bash cvpr/harness/launch.sh cvpr/eval/intphys1_late/protocol.yaml intphys1_late vith` → `cvpr/results/eval/intphys1_late/intphys1_late_vith/`
- `CUDA_VISIBLE_DEVICES=3 python cvpr/eval/intphys1_late/copy_baseline.py` → 같은 폴더 `copy_baseline.json`
- 표의 분해 · CI 는 `per_block.json` 의 `per_video_surprise` 와 `pairs.csv` 에서 다시 셌다.
