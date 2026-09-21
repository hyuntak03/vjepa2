# configs/ — 무엇이 어디 있나 (2026-09-21 정리)

**평가 프로토콜 config 는 `protocols/` 한 곳에만 있다.** 다른 데 만들지 않는다.

| 폴더 | 무엇 | 누가 읽나 |
|---|---|---|
| **`protocols/`** | **평가 프로토콜의 유일한 집.** 프로토콜 yaml + `datasets.md` + `models.md` 레지스트리 | `z_research/scripts/run.sh` |
| `training/` | predictor 학습 | `z_training/train.sh` |
| `eval/` `eval_2_1/` `train/` `train_2_1/` `inference/` | **업스트림 V-JEPA 2 공식 파일.** 건드리지 않는다 | 업스트림 코드 |
| `_archive/` | 죽은 실험 세트의 config. 지우지 않고 내려 둔 것 | — |

## 규칙 — 실험마다 yaml 을 만들지 않는다

`protocols/` 는 **프로토콜 × 데이터셋 × 모델을 실행 직전에 합성**한다
(`z_research/scripts/harness/resolve.py`). 그래서 파일이 늘지 않는다.

```
data  = { **datasets.md[<데이터셋>], **프로토콜.data  }     # 프로토콜이 이긴다
model = { **models.md[<모델>],       **프로토콜.model }
```

- **데이터를 추가** → `protocols/datasets.md` 에 섹션 하나
- **모델을 추가** → `protocols/models.md` 에 섹션 하나 + `analysis/model_loaders.py` 의 `BUILDERS` 에 한 줄
- **일회성 변형** → 새 yaml 이 아니라 `SET="a.b=1 c.d=null"` (`null` = 키 삭제)
- **구조가 진짜 다른 실험** → `extends:`

**새 yaml 의 기준은 하나다 — 재는 방식 자체가 다른가?** (다른 index, 다른 표현 지점, 다른 라벨 의미)

## 현역 프로토콜

| yaml | 무엇 | 정본 문서 |
|---|---|---|
| `intphys1_sliding.yaml` | **Garrido sliding** (IntPhys 1 · GRASP · InfLevel) | `z_research/Benchmarks/PROTOCOLS.md` |
| `surprise_c16t32.yaml` | 우리 고정 C16/T32 (IntPhysGen v11) | `z_research/IntPhysGenV11/README.md` |
| `attn_probe.yaml` · `attn_probe_imp.yaml` · `attn_probe_xfer.yaml` | z/p/h attentive probing | 〃 |

## 예외 — IntPhys 2

`analysis/intphys2/configs/` 에 있다. **하네스가 다르고**(`python -m analysis.intphys2.eval --config …`)
통짜 yaml 방식이라 `protocols/` 와 섞으면 안 된다. **config 는 코드 옆에 둔다.**
IntPhys 2 프로토콜은 현재 작업 범위 밖이다 (`z_research/Benchmarks/PROTOCOLS.md` §7).

## `_archive/` 에 내린 것 (2026-09-21)

죽은 VLM/합성데이터 실험 세트다. git 추적 중이라 되돌릴 수 있다.

`analysis_vlm/` `toy_dataset/` `blender_occlusion/` `blender_toy_dataset/` `synthetic/`
`probing/` `InsPhys2/` `rotation_counting/` `z_tak_attentive_probing/` `world_model_analysis/`

⚠️ `world_model_analysis/` 는 구버전 진입점(`python -m evals.main --fname …`)용이다.
**신규 실험은 전부 `z_research/scripts/run.sh`** 를 쓴다. 옛 config 를 되살려야 하면
`z_exp/.../summary.json` 안에 그때 쓴 config 가 통째로 들어 있다.
