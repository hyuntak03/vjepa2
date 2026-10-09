# IntPhys 2 (Main) 위반 시점 — 실측 (2026-09-30)

**질문 (사용자)**: "IntPhys 2 우리 위반 시점이 언제인지 아나?" → **metadata 에 없다.** 그래서 쌍 영상의 픽셀 차이로 추정했다.

## 결론 (3 줄)

1. **위반은 대체로 영상 후반이다.** continuity · immutability · permanence 는 중앙값 **6.7–6.9 s** (영상 10.6 s), solidity 만 **3.9 s** 로 이르다. 전체 중앙값 6.05 s, 4 s 전 32 % · 4–8 s 57 % · 8 s 후 11 %.
2. **우리 채점 창과의 관계**: 6 fps · 창 8 s · 문맥 4 s · stride 0.33 s → 고정 창 9 개는 예측 구간이 [4 s, 10.6 s] 이다. onset ≥ 4 s 인 68 % 는 창 5–9 개의 예측 구간에 들지만 (60 %), **onset < 4 s 인 32 % (solidity 는 55 %) 는 고정 창 어디에서도 예측 대상이 아니고 growing-context prefix (창 0, 문맥 2–22 프레임) 만 본다.** 즉 solidity 절반은 "문맥 안에 이미 위반이 있는" 채로 채점된다.
3. **쌍은 위반 전에도 픽셀 단위로 같지 않다** (카메라 이동 · 구름 · 그림자; 국소 셀 기준 pre-level 1–65). v11 처럼 "문맥 비트 단위 동일" 이 성립하지 않는다 → IntPhys 2 에서는 matched-context 채점 (§1-2) 의 순수 기하 환원이 애초에 안 된다.

## 방법 (v2 — 국소 셀 통계)

쌍 = 같은 `SceneIndex` 의 `k_Possible` / `k_Impossible` (k = 1, 2), 506 쌍 (253 장면 × 2). 128×128 RGB 절대차 → 8×8 셀 (16×16 격자) 평균 → `s_t = max(셀) − median(셀)` (전역 표류 제거한 국소 최대) → 15 프레임 median filter → 기준 = 앞 1 초 중앙값 med · MAD → 문턱 = `med + max(3·MAD, 12, 0.5·med)` → **0.5 s 연속 초과하는 첫 프레임**.

- 검출 457 / 506 (90 %). 미검출 49 는 카메라 이동 장면 (pre-level 50–65 → 문턱이 실제 계단보다 높음) 또는 작은 물체 (계단 높이 < 12).
- **일관성 검사**: 같은 장면의 1 쌍 · 2 쌍 시점 차 중앙값 **0.02 s** (q90 1.32 s, n 222). 두 쌍은 독립 렌더라 이 일치는 방법이 장면의 사건을 잡고 있다는 뜻이다.
- **눈 검증** (`figures/intphys2_onset/validate_*.png`, 20 쌍): 무작위 8 중 6 정확 (±0.3 s) · 1 은 0.8 s 늦음 (어두운 장면, 스파이크) · **1 은 이른 오검출** (장면 58, 카메라 경로가 1.5 s 부터 갈라짐; 진짜 사건 ~7.4 s). solidity 6 중 4 정확 · 2 애매 (142 카메라 줌, 145 ±0.3 s). 미검출 6 중 **3 은 6.5–7.1 s 에 눈에 보이는 계단이 있다** (167 · 215 · 228 — 문턱이 높거나 물체가 작음), 3 은 픽셀 차이가 안 보임.
- 따라서 **분포는 ±0.5 s · 약 10–15 % 오류율의 추정치**다. 개별 쌍의 시점을 인용하려면 그 쌍의 곡선을 봐야 한다 (`s_loc_curve` 저장됨).

### 폐기한 방법 (기록 — 다시 쓰지 말 것)

| 방법 | 왜 틀렸나 |
|---|---|
| 전역 평균차 (64×64 회색) 절대 문턱 2.0 | 쌍이 처음부터 1–2 다름 → 0 s 에 발화하거나 (74 쌍) 아예 못 넘음 |
| 앞 15 프레임 MAD 상대 문턱 (전역 평균) | 카메라 표류에 발화 (107 permanence 1.5 s vs 실제 6.3 s), 작은 물체 (공 ~3 px) 는 평균을 못 움직임 (123 continuity) — 몽타주 6 장면 중 4 틀림 |
| 사중항 6 조합 중 앞 1 초 차 최소 조합 고르기 | 전역 평균 기준으론 교차 조합이 13/60 뽑혔지만 국소 통계로 보면 k 쌍이 맞다 (장면 1 몽타주: 1 쌍만 빨간 큐브 공유) |

## 표

| condition | n | 검출 | median | q25 | q75 | <3 s | <4 s | 4–6 | 6–8 | ≥8 s | 고정 창 0 개 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| continuity | 120 | 104 | 6.91 | 4.92 | 7.66 | .16 | .24 | .12 | .43 | .20 | .24 |
| immutability | 120 | 101 | 6.83 | 5.48 | 7.42 | .17 | .22 | .05 | .59 | .14 | .22 |
| permanence | 120 | 113 | 6.73 | 5.42 | 7.35 | .16 | .19 | .12 | .57 | .12 | .19 |
| solidity | 146 | 139 | **3.93** | 3.67 | 5.37 | .18 | **.55** | .35 | .08 | .03 | **.55** |

난이도별 중앙값: Easy 6.33 · Medium 6.24 · Hard 4.98 (Hard 에 solidity 가 많다). 위반 뒤 남는 시간 중앙값 4.55 s (2 s 미만은 3 %) — 위반 뒤 관측은 충분하다.

## 함의 (단서와 같은 비중)

- **solidity 가 우리 프로토콜에서 가장 낮은 것과 맞물릴 수 있다** — 절반은 위반이 문맥 4 s 안에 이미 들어 있고, 나머지도 예측 구간 초입이다. 그러나 이건 시사다: "문맥 안 위반" 이 채점에 유리한지 불리한지 (impossible 문맥 → 이상한 미래 예측 → surprise ↑?) 는 창별 surprise 곡선으로 따로 봐야 한다 (`save_per_window: true` 산출물 있음, 미분석).
- IntPhys 2 는 **위반 전 문맥이 동일하지 않은 벤치마크**다. v11 · IntPhys 1 의 matched-context 논리 ("p 가 비트 단위 동일") 를 IntPhys 2 에 옮겨 쓰면 안 된다. 카메라 이동 (Camera 열) 장면에선 더 그렇다.
- transient 위반 (solidity 154 · 7: 3.9–5.5 s 만 다르고 다시 합쳐짐) 이 있다 — 공이 통과 vs 튕김 뒤 같은 곳에 정착. 이런 쌍은 "위반 뒤 남는 차이" 로 채점하는 어떤 방법에도 어렵다.

## 재현

```bash
# 데이터 노드 (vll6, /data2/local_datasets/world/IntPhys2/Main), CPU 만
srun -w vll6 -p debug_vll -c 20 --mem 64G -t 60 bash -c 'source /data/hyuntak/anaconda3/bin/activate vjepa2; cd /data/hyuntak/project/2026/2027_cvpr/vjepa2; \
  python auto_research/scripts/intphys2_onset.py --out auto_research/exp_results/verify/intphys2_onset_main_v2.json --workers 18'
python auto_research/scripts/intphys2_onset_validate.py --json auto_research/exp_results/verify/intphys2_onset_main_v2.json --out auto_research/figures/intphys2_onset/validate_random8.png --n 8 --seed 1
# 폐기한 6 조합 탐침: auto_research/scripts/intphys2_pairing_probe.py (--stat-q 99 가 국소판), 산출 exp_results/verify/intphys2_pairing_probe{,_p99}.json
# 우리 IntPhys 2 프로토콜: analysis/intphys2/configs/bench_intphys2_main_vith.yaml (frame_step 10 → 6 fps, window 48, context 24, stride 2, protocol growing)
```
표의 수치는 `intphys2_onset_main_v2.json` 의 `rows[].onset_loc_sec` 에서 이 문서 작성 시 다시 계산했다 (창 수 계산: 시작 0, 1/3, …, 2.6 s 의 9 창, 예측 구간 [s+4, s+8)).
