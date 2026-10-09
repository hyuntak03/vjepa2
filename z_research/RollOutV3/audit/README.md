# RollOutV3 감사

> 시작점은 [`../README.md`](../README.md) 다. 2026-09-26 전에는 `new_archive/_audit/` 였다.
> 각 폴더의 `AUDIT.md` 가 판정이고, json 이 수치다. 문서 안의 `new_archive/…` 경로는 옛 이름이다 → `RollOutV3/figures/…` (`_audit/…` → `RollOutV3/audit/…`).

| 폴더 | 무엇을 검사했나 | 자 | 스크립트 (`z_research/scripts/analysis/`) | 지금 유효? |
|---|---|---|---|---|
| `training_sets/` | **지금 자의 학습셋** `RollOut_v2_training_v8` (= `training_r8`) 감사 1 · 2 · 3 차. 3 차 통과 | — | `audit_training_set.py --root <렌더>` | ✅ |
| `training_v8/` | 옛 학습셋 `RollOut_v2_training` (14,360 clip) 감사 + **v11 판 빈 장면 오탐** (`panel_false_alarm_{presence,identity,identity_r8}.json`) | 셋 다 | `audit_training_v8.py`, `v11_panel_false_alarm.py` | ✅ (판 오탐 비교의 정본) |
| `v3/` | v3 운동 주장, 궤적 단위 CI | presence | `audit_v3_motion.py` | ◐ 방법은 유효. 수치는 옛 자 |
| `v11_presence/` | v11 '있음' 주장 · 문맥 복사 기준선 · 빈 장면 오탐 바닥 | presence | `audit_v11_presence.py` | ❌ 옛 자는 판 오탐이 있었다. 지금 수치는 IDENTITY_R8 §3 · §5 |
| `v11_identity/` | 정체 probe · 풀링 기하 (쌍둥이 · 궤적 · 이식 · 기하) | **자 없음** | `audit_v11_identity.py` | ✅ |

## 재현

```bash
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
$P z_research/scripts/analysis/audit_training_set.py --root /data2/local_datasets/world/world_analysis/RollOut_v2_training_v8   # training_sets/
R3_DECODER=identity_r8 R3_OUT=identity_r8 $P z_research/scripts/analysis/v11_panel_false_alarm.py                            # training_v8/panel_false_alarm_identity_r8.json
$P z_research/scripts/analysis/audit_v11_identity.py --part {probe,twin,geom,xfer} [--device cuda:N]                         # v11_identity/ (자 없음, 스크립트 머리말에 부분별 시간)
```
옛 자 감사 (`training_v8/AUDIT.md` · `v3/` · `v11_presence/`) 의 재현은 각 문서 끝에 있다.
