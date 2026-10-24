# eval/intphys2_late — IntPhys 2 Main, 사람 라벨 위반 시작 앞을 문맥으로 (2026-10-22)

**무엇을 재나.** IntPhys 2 Main 에서 위반이 처음 보이는 프레임 d (사람 라벨) 앞을 문맥, 뒤를 미래로 두고 창 하나로 채점한다. IntPhys 1 uni (`cvpr/eval/intphys1_late/build.py --uniform-d`) 와 같은 창 규칙이다.

## 정의
- **라벨**: IntPhys2_labeling 시트 labels 탭의 위반 시작 프레임 (60 fps, 0-based). 이번에 쓴 사본 = `data_csv/intphys2_late_uni/labels_snapshot.csv` (2026-10-22, 진행 중 라벨 포함).
- **쌍**: 같은 장면의 (k 가능, k 불가능), k = 1, 2. 쌍 1 과 2 는 가로질러 비교하지 않는다 (`analysis/intphys2/README.md` §3).
- **d 고르기**: 자기 라벨이 있으면 그것 (403 쌍). 비었거나 "못 찾음" 이면 같은 장면 다른 쌍의 라벨 (95 쌍, 사용자 10-22: "같은 시나리오면 pair 1, 2 가 사실상 violation start frame 도 같다"). 둘 다 없는 4 장면 8 쌍은 뺀다 → **498 쌍**.
  - 둘 다 라벨이 있는 장면에서 pair 1 · 2 차이: 중앙값 4 프레임, 5 프레임 이하 59 %, 최대 229 프레임.
- **창**: n = min(가능, 불가능 영상 길이). 문맥 16 장 = round(linspace(0, d − 1, 16)), 미래 16 장 = round(linspace(d, n − 1, 16)). 두 영상에 같은 번호.
- **채점**: testbed 와 같다 (surprise = mean |p − LN(h)[미래]|, matched, 문맥 16 + 미래 16, dual encoder, fp32 + autocast fp16, mask_index 0).
- ⚠️ **문맥이 픽셀로 같지 않다.** IntPhys 2 는 가능 · 불가능 영상이 따로 렌더된다. 문맥 16 장에서 RGB 차이 > 10 인 픽셀 비율: 쌍 중앙값 11 %, 90 분위 44 %, 최대 91 % (`context_integrity.json`). 그래서 p 도 쌍 안에서 다르다.

## 결과 — release (`vith`, 2026-10-22, vll6)
쌍 정확도 % (chance 50). 복사 = 마지막 문맥 튜블릿 latent 를 미래 8 칸에 복사 (copy_z = 문맥 encoder, copy_h = 타깃 encoder).

| | n | release | copy_z | copy_h |
|---|---:|---:|---:|---:|
| **전체** | 498 | **54.0** [50.4, 57.6] | 52.4 | 51.2 |
| permanence | 120 | 58.3 | 50.8 | 50.8 |
| continuity | 120 | 56.7 | 50.0 | 55.8 |
| solidity | 142 | 52.1 | 51.4 | 48.6 |
| immutability | 116 | 49.1 | 57.8 | 50.0 |
| 위반이 가림 뒤 | 51 | 51.0 | 54.9 | 45.1 |
| 자기 라벨 | 403 | 54.8 | 53.1 | 52.6 |
| 다른 쌍 라벨 | 95 | 50.5 | 49.5 | 45.3 |

release − copy (장면 bootstrap 95 % CI, 4,000 회): − copy_z +1.6 [−2.8, +6.2] · − copy_h +2.8 [−2.2, +7.8]. 값은 `summary.json` · `copy_baseline.json` · `per_block.json` 에서 다시 셌다.

## 재현
```bash
curl -sL -o labels.csv "https://docs.google.com/spreadsheets/d/1OZduu13VvKCebp0YZuTc0WEoZaFaBzSxMOwmHNOrEZY/gviz/tq?tqx=out:csv&sheet=labels"
python cvpr/eval/intphys2_late/build.py --labels labels.csv          # 프레임 → /data2/local_datasets/world/world_analysis/IntPhys2_late_uni (vll5), index → data_csv/intphys2_late_uni
GPU_IDS="0" bash cvpr/eval/intphys2_late/run.sh vith
COPY=1 CUDA_VISIBLE_DEVICES=0 bash cvpr/eval/intphys2_late/run.sh vith
# 다른 노드: 프레임 NFS 사본 z_pixel_decoder/runs/_frames/IntPhys2_late_uni 를 SET="data.frames_root=<사본>" 으로 (z_pixel_decoder/runs/slurm_logs/ip2late.sbatch)
```
