# IntPhys 1 · 사건이 창의 어디에 있나 — context 안 창을 빼면 88.89 → 91.67 (2026-09-25)

> 상위: `../README.md` · 프로토콜 `../PROTOCOLS.md` · 창 그림 `../figures/intphys1_windows/`.
> 수치는 `../exp_results/intphys1_event_position/event_position_acc.json` (공식 실행의 `per_window.json` 에서 GPU 없이 계산) 과
> `auto_research/exp_results/h6/h6h_copy_noln_vith_r*.npz` (p · copy 를 창마다 다시 뽑은 것) 에서 옮겼다.
> 두 원자료로 따로 계산한 p 의 V1 점수가 서로 같다 (91.67 / 90.56).

## 0. 결론

**Garrido sliding 의 Filtered (시작점마다 min over C) 는 IntPhys 1 88.89 칸에서 시작점의 약 40 % 에 "위반이 이미 context 안에 들어간 창" 을 고른다.**
**그런 창은 판별이 약하다 (창 정답 60.3 %). 사건이 예측 구간 안에 있는 창만 쓰면 88.89 → 91.67 로 오른다.**
즉 88.89 는 "모델이 위반을 미리 봐서" 부풀려진 값이 아니다. 반대로 그런 창이 깎아 먹은 값이다.

| 칸 | V0 공식 (전 창) | **V1 사건이 예측 구간 안인 창만** | V3 사건이 context 안인 창만 |
|---|---:|---:|---:|
| **skip2_w32** (88.89 칸) | 88.89 [84.4, 92.8] | **91.67 [87.8, 95.0]** | 58.83 [50.2, 67.3] (n = 162 쌍) |
| skip2_w16 (공식 config 칸) | 83.33 [77.8, 88.9] | **90.56 [86.1, 94.4]** | 73.96 [67.6, 79.9] (n = 174) |
| skip5_w16 | 61.11 | 61.76 (n = 178) | 55.28 (n = 132) |

Filtered macro, 95 % CI 는 scene bootstrap (principle 층화, B = 10,000). V0 은 공식 per-principle 85.00 / 96.67 / 85.00 을 그대로 재현한다.

## 1. 정의

- `d` = pos/imp PNG 가 처음 다른 raw 프레임 (0-기준). PNG 를 직접 비교해 잡았고, 180 쌍 모두 `pairs.csv` 의 `first_div_sensitive − 1` 과 같다.
  (H6 §3-3 이 쓴 `first_div_strict` 는 이동+가림 54/60 쌍에서 이보다 늦다 — `TrainingEffects/ip1_copy/IP1_COPY.md` 정정 3.)
- 창 (s, C) 은 채점기와 같은 생성기로 만든다. context raw `cf`, 예측 raw `tf` 로 네 분류를 둔다.
  - `before`: d ≤ cf[0]. 창 전체가 이미 갈린 뒤다.
  - `context`: 위반이 context 안에 있다. 모델이 위반을 본 뒤 예측한다.
  - `future`: context 는 픽셀 단위로 같고 위반은 예측 구간 안이다. VoE 의 정상 경우다.
  - `after`: 창 전체가 같다.
- 변형은 쌍마다 창 집합을 정하고, pos 와 imp 가 같은 창 집합을 쓴다. 나머지는 공식과 같다: 시작점마다 min over C, 시작점 평균, 쌍 비교, principle macro.

## 2. 결과 (skip2_w32)

**창 단위** (쌍 180 × 창 45):

| 분류 | 창 비율 | 창 정답 % [CI] |
|---|---:|---|
| future | 62.5 % | **86.9 [82.9, 90.5]** |
| context | 27.7 % | **60.3 [52.8, 67.2]** |
| after | 6.8 % | 50.0 (550 창 전부 정확히 동점) |
| before | 3.1 % | 50.0 (250 창 전부에서 S(imp) = 같은 scene 다른 pos 의 S — imp 가 두 pos 를 이어 붙인 영상이라 사건 뒤 창은 다른 pos 와 픽셀이 같다) |

**Filtered 가 고른 창** (쌍 × 시작점 1,620):
- imp 쪽 선택은 context 43.8 %, future 46.4 % 다. pos 쪽 선택은 context 37.5 %, future 52.6 % 다.
- 선택 조합별 시작점 정답: 둘 다 future 45.6 % 에서 87.4 %, 둘 다 context 36.7 % 에서 79.7 %, pos future · imp context 7.0 % 에서 71.9 %.

**그룹별** (V0 → V1): 이동+가림 75.0 → **80.0**, 이동+보임 91.7 → 95.0, 정지 둘은 100 → 100.
skip2_w16 에서는 이동+가림이 61.7 → **80.0** 이다. 창이 짧아서 사건이 context 로 들어가거나 창 밖으로 빠지는 창이 많기 때문이다.

**copy 기준선과 함께** (H6h 원자료, macro):

| 칸 | 변형 | p | copy (LN) | copy_raw | p − copy (LN) |
|---|---|---:|---:|---:|---|
| skip2_w32 | V0 | 88.89 | 85.00 | 77.78 | +3.9 [−0.6, +8.9] |
| skip2_w32 | **V1** | **91.67** | 87.22 | 88.89 | **+4.4 [+0.6, +8.3]** |
| skip2_w16 | V0 | 83.89 | 71.67 | 65.56 | +12.2 [+6.7, +17.8] |
| skip2_w16 | V1 | 90.56 | 82.78 | 77.22 | +7.8 [+3.3, +12.8] |

깨끗한 창만 쓰면 copy 도 함께 오른다. 최고 칸에서 p − copy 는 V0 과 같은 크기 (+4.4) 이고, CI 하한이 겨우 0 을 넘는다.
그래서 H6 의 "IntPhys 1 점수의 상당 부분은 copy 로도 나온다" 는 V1 에서도 그대로다. LN 을 뺀 copy_raw 는 V1 에서 88.89 까지 오른다.

## 3. 예시 쌍 O1_01 (그림의 쌍, d = raw 46, 사라짐 방향)

9 시작점 모두에서 imp < pos 라 **이 쌍은 전부 틀린다.** start raw 20 도 틀린다: pos C*=8 (future) 0.59836, imp C*=16 (context) 0.59621.
45 창 중 imp > pos 인 창은 start 16 · 20 · 24 의 C = 20 셋뿐이다. 셋 다 `context` 창이다.
사라짐 방향은 원래 p 와 copy 모두 chance 다 (H6 §3-2: skip2_w32 p 56.0 = copy 56.0, 25 쌍).

## 4. 단서

- **V1 은 분석용 변형이지 벤치마크 점수가 아니다.** 창 집합을 고를 때 쌍의 갈림 프레임 `d` 를 쓴다. 이 값은 matched pair 에서는 알 수 있지만 단일 영상 채점에서는 모른다. 공식 격자 최고 칸 선택 (descriptive) 도 그대로 남아 있다.
- 표적은 여전히 양방향 target encoder 다. future 창 안에서도 사건 전 튜블릿에 번짐이 있다 (H6 §3-4).
- 창 단위 정답은 겹치는 창에서 나와 독립이 아니다. CI 는 scene 단위로만 믿는다.
- `context` 창의 60.3 % 는 "위반을 본 뒤의 미래가 더 예측 가능하다" (예: 사라진 물체가 없는 미래) 는 사례와 그 반대가 섞인 값이다. 방향별로 쪼개지 않았다.
- skip5_w16 은 창이 10 개뿐이라 V1 에서 2 쌍이 빠진다 (178).

## 5. 사건 물체는 몇 프레임 가려지나 (`../exp_results/intphys1_event_position/occlusion_stats.json`)

> **대체됨 (2026-09-26)** — 정본은 [`z_research/OcclusionStats/Archive/INTPHYS1_OCCLUSION_2026-09-26.md`](../../OcclusionStats/Archive/INTPHYS1_OCCLUSION_2026-09-26.md).
> 이 절은 두 possible 영상의 mask 면적 목록 비교 (scene 단위) 판이다. 두 가지가 바뀌었다.
> (1) O3 는 두 영상의 물체 타이밍이 달라 영상별로 따로 잰다.
> (2) 쌍의 pos 에 물체가 없으면 다른 영상의 가림을 빌려 오지 않고 '사건 전 미관측' 으로 센다 (이동 + 가림 15 쌍).
> 새 값: 이동 + 가림 사건 가림 2–6 프레임 (중앙 3 = 0.2 초), skip-2 1–3 장, 가려진 동안 중앙 4.6 물체 폭 이동.
> 정지 + 가림 27–89 프레임 (중앙 4.9 초), 물체는 안 움직인다. 아래 원문은 기록으로 둔다.

**판정 (파라미터 없음):** 한 scene 의 possible 영상 두 개는 사건 물체만 다르다. 그래서 프레임마다 두 영상의 mask 조각 면적 목록이 같으면 사건 물체가 안 보이는 것이다.
mask 값은 프레임마다 무작위로 다시 매겨지므로, 값이 아니라 면적 목록을 비교한다.

**사건 가림 (event gap):** 양쪽이 '보임' 으로 막힌 '안 보임' 구간이다. d 를 포함하거나, d 가 그 구간 끝 뒤 3 프레임 안에 있는 것을 센다.
뒤의 경우는 이동+가림 58 쌍 중 18 쌍이다. 위반은 가림 중에 일어나고, 픽셀은 물체가 다시 나타나는 순간에야 갈린다.

⚠️ `auto_research/_stage/IntPhys1_dev_by_scene/obj_vis.npz` 는 0/1 가시성이 아니라 **보이는 물체 수 (0–3)** 다. 물체가 여럿인 scene 에서는 사건 물체의 가림을 가르지 못하므로 쓰지 않았다.

| 그룹 | 사건 가림이 있는 쌍 | raw 프레임 (중앙 [범위]) | **skip 2 로 모델이 받는 가려진 프레임** | skip 5 |
|---|---:|---|---|---|
| **이동 + 가림** | 58 / 60 | **3 [1, 5]** (분포 1:8, 2:12, 3:28, 4:8, 5:2) | **1 [0, 3]** (0 장 4 쌍 · 1 장 38 · 2 장 14 · 3 장 2) | 0–1 |
| 정지 + 가림 | 30 / 30 | 72 [20, 76] | 36 [10, 38] — 창 32 장보다 길다 | 14 [4, 15] |
| 이동·정지 + 보임 | 0 | — | — | — |

- 예시 O1_01: 사건 물체가 raw 45–47 의 3 프레임 가려진다. skip 2 격자에서는 raw 46 한 장뿐이다.
- 원리별 (이동 + 가림) 중앙값은 O1 · O2 · O3 모두 3 raw 프레임이다.
- 이동 + 가림의 나머지 2 쌍 (O3_29) 은 d 근처에 가림이 없다.
- **읽기:** IntPhys 1 의 이동 + 가림은 물체가 **1 샘플 프레임 (반 튜블릿) 쯤** 사라졌다 나오는 사건이다. 가림을 넘어 상태를 오래 운반하는지는 거의 재지 않는다.
  반대로 정지 + 가림은 창 전체보다 오래 가려진다. 창의 context 가 가림막이 올라간 뒤에서 시작하면 context 에 물체가 한 번도 안 보인다. 그런데도 skip2_w32 에서 100 % 다.

## 재현

```bash
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
python z_research/scripts/analysis/intphys1_event_position_acc.py        # CPU 3 분 (PNG 360 개 영상 대조 포함) → exp_results/intphys1_event_position/event_position_acc.json
python z_research/scripts/figures/plot_intphys1_windows.py               # 창 그림 (../figures/intphys1_windows/)
python z_research/scripts/analysis/intphys1_occlusion_stats.py         # §5 가림 길이 (CPU 40 초, masks 7 만 장) → exp_results/intphys1_event_position/occlusion_stats.json
```
copy 비교 표는 같은 스크립트가 `auto_research/exp_results/h6/h6h_copy_noln_vith_r*.npz` 가 있을 때 함께 낸다 (json 의 `copy_vs_p`).
