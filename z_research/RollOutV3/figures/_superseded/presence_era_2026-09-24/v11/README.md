# v11/ — IntPhysGen v11 에 RollOutV3 자를 건 것 (2026-09-24)

자 = `exp_results/presence/` (지문 `dcd24d8a8a47`), 읽은 값 = `exp_results/v11/` (visible · late) · `exp_results/v11_mid/` (mid).
v11 은 자의 학습에 안 썼다. 가능 변이만.

| 파일 | 무엇 |
|---|---|
| `fig_v11_readout.png` | visible vs 문맥 끝 가림 (late): p 의 '있다' % · 속도 방향 R² · 가속 방향 R² (부호 붙음). 수치 `../../exp_results/v11/V11_READOUT.md` |
| `gif_timing_<motion>_k<k>.gif` | 열 = visible \| 문맥 중간 가림 (mid) \| 문맥 끝 가림 (late). 행 = 진행 방향을 맞춘 clip. `<motion>` = static · flat · ramp, **moving = flat·ramp 를 한 파일에 4 행** (flat 좌→우 · 우→좌 · ramp 좌→우 · 우→좌). 흰 고리 = 진실, 주황 = p 읽기 ('있다' 일 때만), × = 마지막으로 본 자리 |

⚠️ 자 검사 (V11_READOUT.md §0): 가려진 물체를 h 가 33 % '있다' 로 읽는다 (가림 편향). 화면 가장자리 36 px 안은 h 도 못 읽는다 → 표·그림에서 뺐다.
⚠️ GIF 의 점 위치는 자의 읽기다 — 물체가 흐려지면 기본값 쪽으로 끌린다. '있다/없다' 를 먼저 본다.

## 재현

```bash
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
$P z_research/scripts/analysis/v11_readout_online.py --gpus 8 --token-budget 196608 --max-bs 48                # visible + late
$P z_research/scripts/analysis/v11_readout_online.py --gpus 8 --token-budget 196608 --max-bs 48 --timing mid   # mid
$P z_research/scripts/figures/plot_v11_readout.py
$P z_research/scripts/figures/plot_v11_timing_gif.py --k 3 --rows 2
```
