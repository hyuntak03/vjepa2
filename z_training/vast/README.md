# z_training/vast — Vast.ai 기계에서 학습한 predictor 체크포인트

출처: `origin/vast` 브랜치 (`86c04aa`, 2026-10-01), 결과 문서 `git show origin/vast:z_training/RESULTS_2026-10-01.md`.
아래 값은 각 체크포인트 안의 `arch` · `config` 를 직접 열어 확인했다 (2026-10-10).

공통: V-JEPA 2 ViT-H encoder **동결**, predictor **scratch**, 형식 `vjepa_frozen/predictor-only/v1` (predictor 가중치만), 8 GPU · global 512 · lr 5.2e-4.

| 파일 | 원래 run (vast) | `arch.kind` | 데이터 | 창 | 학습 | IntPhys1 dev A.8 최고 |
|---|---|---|---|---|---|---:|
| `natural_k710_ssv2_howto.pt` (옛 `final.pt`) | `k710_ssv2_how_ep70` e63 → `k710_ssv2_how_ep70_cooldown64` 최종 | `oneshot` (tube 마스크, 릴리즈 구조) | K710 + SSv2 + HowTo100M 1 % = 1:1:1 | 16 장 → cooldown 64 장 | 70 ep · 20,580 step · ~1,054 만 clip | **82.22** (skip2_w16; 릴리즈 88.89) |
| `AR_internal_context.pt` (옛 `natural_ctx_state_latest.pt`) | `natural_ctx_state` e40 | `ctx_state` (문맥 z 고정 + predictor 은닉 상태 되먹임 rollout, `tf_weight 0`) | K710 + SSv2 = 1:1 (**HowTo 없음**) | 32 장, C{2,4,8} → K{4,6,8} 블록 | 40 ep · 11,760 step · 602 만 clip | 48.33 (우연 수준) |

- 삭제 (2026-10-10, 사용자 지시): `natural_tube_k710_latest.pt` — tube, K710 + SSv2 1:1, `AR_internal_context` 와 같은 데이터·예산의 대조군 (IntPhys1 71.67).
- ⚠️ `AR_internal_context.pt` 는 **main 에서 로드되지 않는다** — `kind: ctx_state` 구현과 `analysis/intphys2/model.py` 의 kind 버그 수정이 vast 브랜치에만 있다.
- ⚠️ `natural_k710_ssv2_howto.pt` 는 v11 `surprise_c16t32` · IntPhys 2 · 복사 기준선을 아직 안 쟀다.
- ⚠️ 두 파일은 데이터 (HowTo) · 예산 · cooldown 이 모두 달라 직접 비교하면 안 된다.

## 재현

```bash
# 채점 (main, oneshot 만 가능)
SET="model.predictor_checkpoint=$PWD/z_training/vast/natural_k710_ssv2_howto.pt model.predictor.kind=oneshot" \
  GPUS=8 bash z_research/scripts/run.sh intphys1_sliding intphys1_dev vith
# 학습 config: git show origin/vast:configs/training/{k710_ssv2_how_ep70,k710_ssv2_how_ep70_cooldown64,natural_ctx_state}.yaml
```
