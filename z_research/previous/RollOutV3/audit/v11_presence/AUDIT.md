> **2026-09-26 정리 — presence 자 시기 문서.** 자 `presence` (`dcd24d8a8a47`) 로 읽은 값이다. 그 자는 v11 가림막을 물체로 읽었다 (판 빈 장면 '있음' p 7–45 %).
> 지금 자는 `identity_r8` 이고, 정본은 [`../../README.md`](../../README.md) → [`../../Archive/IDENTITY_R8_2026-09-26.md`](../../Archive/IDENTITY_R8_2026-09-26.md) 다. 이 감사의 수치는 지금 결론에 쓰지 않는다 (지금 값: IDENTITY_R8 §3 · §5). 대상 경로 `new_archive/v11_presence` 는 지금 `figures/v11_presence/_superseded/presence_decoder/` 다.

# 감사 — `new_archive/v11_presence` (2026-09-25)

수치는 전부 이 폴더의 `results.json` · `tables.txt` 에서 다시 계산했다. 자 지문은 `dcd24d8a8a47` 이다. 문턱은 자마다 하나 (`thr_val_fpr5`: p +0.831 · z −0.662 · h −0.961) 이고, 문턱 없는 AUROC (물체 vs 같은 조건 빈 장면) 는 보조로만 쓴다.

## 0. 표본 구조 — 유효 표본은 궤적이다

- 진실 궤적 (32×2, 1 px 반올림 해시) 은 **움직이는 팔마다 2 개 (l2r · r2l)** 이고, **static 은 자리 8 개**다. 이 궤적들은 visible · mid · last 와 k=1..4 전부에서 **같다** (`trajectory_cross`). 빈 장면도 같은 궤적을 쓴다.
- clip 1,176~1,568 개는 궤적 하나당 외형·판 복사본 144~790 개다. 그래서 CI 는 **(궤적, k) 단위** 두 단계 부트스트랩으로 냈다. 움직이는 팔 last/mid 는 6 단위, visible 은 2 단위, static 은 8~24 단위다. 궤적이 둘뿐이라 **"다른 속도·경로로 일반화된다" 는 이 자료로 말할 수 없다.** 방향별 값을 같이 적는다.

## 1. 주장별 판정

| 주장 | 판정 | 핵심 근거 |
|---|---|---|
| (a) last (k=2~4) 움직이는 물체는 p 의 미래에서 '없다', visible · mid 는 '있다' | **holds_weaker** | 물체가 모든 k 에서 보이고 안쪽인 칸 (flat t2–t4, ramp t2–t5) 에서 p '있다' 는 flat 0.34/0.23/0.35, ramp 0.17/0.28/0.07/0.24 이다. 궤적 단위 CI 상한은 ≤0.47 이다. 같은 칸에서 h 는 0.47–1.00, visible p 는 0.77–1.00, 빈 장면 p 는 0–0.07 이다. 두 방향 모두 같은 쪽이다 (flat t3: l2r 0.06 / r2l 0.41, h 0.87/0.86). 자 없는 vanish 비교에서 p 가 물체 미래 쪽에 더 가까운 block 은 flat 1.8/1.8/0 %, ramp 0/0/0 % 다 (visible 100 %). ⚠️ 단서 셋: t0 (와 k≥3 의 t1) 은 물체가 실제로 가려진 칸이라 증거가 아니다. 문턱 없는 AUROC 는 flat t2 0.97 · ramp t3 0.96 이라, p 의 logit 에는 빈 장면과 가를 수 있는 **문턱 아래 물체 신호**가 있다. mid 의 '있다' 는 빈 장면 오탐 (0.02–0.62) 에 오염돼 있다 |
| (b) 문맥 끝에 가려진 static 은 '있다' | **untestable** (이 자로는) | p '있다' 0.73–1.00 인데 **같은 궤적 빈 장면 (trapdoor) 오탐이 0.23–0.92** 다. AUROC 는 0.67–0.89 다. 복사 기준선 (문맥만으로 돌린 z/h 의 마지막 튜블릿) 도 '있다' 0.96/0.94 (빈 장면 0.93/0.80) 이고, AUROC 는 0.71/0.73 이다. p − 복사 는 t2 이후 −0.09~+0.04 로 CI 가 0 을 포함한다. 위치도 같은 궤적 빈 장면에서 p 가 가리키는 좌표와 16–20 px 차이뿐이다. 자 없는 vanish 비교는 82/66/41 % (k=2/3/4) 로, visible 100 % 와 moving last ~1 % 사이다 |
| (c) h 자 검사: 보이고 36 px 안쪽이면 h 는 '있다' | **holds_weaker** | visible 0.94–1.00 · last static 0.95–1.00 · last ramp t3–t5 0.93–1.00 이다. 그러나 last flat t1–t4 는 0.65/0.47/0.86/0.69 (l2r t1–t2 0.17/0.26), last ramp t1–t2 는 0.66/0.74, mid flat t0 는 0.68 이다. 판 바로 옆에서는 h 도 놓친다. **깨끗한 대조 칸은 flat t3 과 ramp t3–t5 뿐이다** |
| (d) 가장자리 · 화면 밖 칸 (flat t5–t7, ramp t6–t7) 은 해석 불가 | **holds** | visible 에서 h 는 flat t5 0.69 / t6–t7 0.00, ramp t6 0.01 / t7 0.00 이다. flat t7 · ramp t7 은 진실이 화면 밖이다. last flat t5 의 p 0.66 은 빈 장면 오탐 (0.01) 이 아니고 가장자리 칸이다 (README 의 "오탐" 은 틀렸다) |
| (e) 빈 장면 오탐 바닥 (last, k≥2) | **holds** (측정값) | flat 0/0/0/0/0.07/0.01/**0.87/0.55**, ramp 0×6/**0.49**/0.07, static 0.23–0.92, mid flat 0.02–0.81, mid ramp 0.24–0.82, visible 0 이다. README 의 "t6 73 %" 는 다른 정의 (V11_READOUT, k=1–4, 운동 합침) 에서 나온 수치다 |

## 2. 복사 기준선 — "마지막 관측 상태를 연장한 것" 인가

- 복사는 문맥 16 샘플만으로 encoder 를 **다시 돌려** 마지막 튜블릿 (idx 7, k≥2 면 물체가 가려짐) 을 자로 읽고, 그 값을 미래 8 튜블릿에 그대로 둔 것이다 (`ctx_only_readings.npz`, GPU 21 분). readings 의 `z[:,7]` 은 32 샘플 전부를 본 표현이라 미래가 새어 들어올 수 있어서 보조로만 적었다.
- **복사는 '없다' 가 아니다.** last 에서 h_ctx 는 '있다' 를 flat 0.20 (빈 장면 0.14) · ramp 0.51 (빈 장면 0.00) 으로 읽는다. z_ctx 는 0.75/0.85 인데 빈 장면도 0.55/0.35 라서 판을 물체로 읽는 오탐이 섞여 있다.
- p 는 ramp 에서 h_ctx 복사보다 '있다' 가 적다 (0.07–0.28 vs 0.51). flat 에서는 비슷하다 (0.23–0.35 vs 0.20). 그래서 "p 는 복사와 같다" 도 "p 가 복사보다 물체를 더 남긴다" 도 성립하지 않는다. 맞는 문장은 이렇다: **ramp 에서는 target encoder 가 문맥만 보고도 가려진 마지막 튜블릿에 물체를 절반쯤 남기는데, p 의 미래에는 그보다 적게 남는다.**
- static 에서 p 와 복사를 가르는 것은 이 자로는 불가능하다 (§1 (b)).
- 자 없는 latent 복사 (마지막 문맥 latent) 는 H6f (`auto_research/exp_results/h6/h6f_v11.json`, block 1/4, 운동·k 합침) 에만 있다. late vanish A 에서 p 22.0 vs 복사 41.7 이다. 운동별 분해는 원자료가 vll6 에만 있어 못 했다.

## 3. mid 와 위치

- **mid 의 '있다'**: static 은 visible 과 같다 (AUROC 1.00, t0–t4 오탐 0). flat · ramp 는 비율이 0.69–0.99 로 높지만, 빈 장면 오탐이 0.02–0.62 이고 AUROC 는 0.76–0.98 (visible 1.00) 이다. 판이 서 있는 장면에서 자가 판에도 반응한다.
- **mid 의 큰 위치 오차는 p 쪽이다.** flat t2–t4 L2 는 32/52/76 px 인데, 이 칸들은 진실이 보이고 안쪽이다 (visible&inside 1.00). h 는 같은 칸에서 11–15 px 다. p 의 x 전진은 19/18/11 px (진실 50/68/86) 이고, p 는 문맥에서 마지막으로 본 자리에서 18–27 px 안에 머문다. visible flat 은 74/77/70 px 전진한다. 같은 궤적 빈 장면에서 p 가 가리키는 좌표와는 79–92 px 떨어져 있어, 자 기본값은 아니다. 다만 p top1 이 0.03–0.08 로 h 의 0.17–0.22 보다 낮다. ramp 는 두 궤적이 t1–t3 에 교차해 (다른 방향 귀무 12–30 px) 위치 귀무가 약하다. t5 이후의 140–167 px 는 가장자리 · 화면 밖이다.
- visible 위치: p 는 복사 (마지막 관측 진실) 를 t2 부터 이긴다 (flat 29 vs 50 px). 하지만 전진이 약 60–77 px 에서 포화한다 (진실 141 px 까지). flat t3 의 12 px 는 추적이 아니라 교차점이다.

## 4. k 별 (last, p '있다' · h, 물체가 보이는 t2–t4)

- flat: k=1 0.62/0.30/0.49 · k=2 0.51/0.29/0.43 · k=3 0.27/0.25/0.36 · k=4 0.24/0.16/0.26. h 는 0.90–0.97 → 0.21–0.72 로 같이 떨어진다 (판 폭이 k 와 함께 커진다).
- ramp t2–t5: k=1 0.45/0.38/0.15/0.42 → k=4 0.10/0.27/0.04/0.22. h 는 t3–t5 에서 k 와 무관하게 0.88–1.00 이다.
- 모든 k 에서 p 는 h 와 visible 보다 한참 낮다. k 가 클수록 더 낮다.

## 5. 쓸 수 있는 문장

> "IntPhysGen v11 에서 움직이는 물체가 문맥 끝에 가려지면 (k=2–4), 물체가 다시 보이고 가장자리에서 떨어진 미래 튜블릿 (flat t3, ramp t3–t5) 에서 p 자기 presence 자는 p 의 미래를 7–28 % 만 '있다' 로 읽는다 (궤적 단위 95 % CI 상한 ≤ 0.39). 같은 칸에서 target encoder 는 86–100 %, 가리지 않은 같은 궤적의 p 는 83–96 %, 같은 가림 장면의 빈 장면 p 는 0 % 다. 궤적 둘 모두 같은 방향이다. 자 없는 matched-pair 비교에서도 p 는 이 조건의 block 98–100 % 에서 빈 미래 쪽에 더 가깝다. 다만 p 의 logit 은 빈 장면과 부분적으로 갈린다 (AUROC 0.86–0.96). 그래서 물체 신호는 0 이 아니고 문턱 아래로 약하다. 멈춘 물체는 가림판이 빈 장면에서도 '있다' 로 읽혀 이 자로 판정할 수 없다."

## 재현

```bash
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
# 복사 기준선용: 문맥 16 샘플만으로 z·h 를 돌려 8 튜블릿을 자로 읽는다 (GPU 2 장 약 22 분) → ctx_only_readings.npz
CUDA_VISIBLE_DEVICES=4,7 $P z_research/scripts/analysis/audit_v11_presence.py extract --gpus 2 --bs 32
# 전부 다시 계산 (CPU 약 5 분) → results.json · tables.txt
$P z_research/scripts/analysis/audit_v11_presence.py analyze
```
입력은 `z_research/RollOutV3/exp_results/{v11,v11_mid}/readings.npz`, 자 `exp_results/presence/` (summary.json · attn_bias_px.json), 자 없는 비교 `z_research/IntPhysGenV11_occlusion_timing_ablation/exp_results/surprise_c16t32__v11_full_vith/per_block.json` + `data_csv/intphysgen_v11_full/index.csv` 다.
