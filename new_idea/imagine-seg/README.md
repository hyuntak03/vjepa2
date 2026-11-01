# imagine-seg — 가린 물체를 "상상" 하는 predictor post-training

시작점:
- 원 계획 (사용자 원문): `PLAN_2026-10-31.md`
- 데이터 생성 계획 · 진행 기록: `DATA_PIPELINE_2026-10-31.md`
- phys3d_perm_v1 공정 비교 (대조군 · 보조 head · in-path): `FAIR_COMPARISON_V1_2026-10-31.md`
  — config `configs/training/imagine_seg_v1_{base,ctrl,head,inpath}.yaml`, 코드 `app/vjepa_frozen/aux_memory.py` (`inpath`, `sup.binary`)
  — 결과 (2026-11-01): head · in-path 모두 대조군과 같음, 다른 벤치로 안 넘어감. 표 `results/summary_v1.md`, 요약 페이지 https://claude.ai/artifact/6XHtd7sGHhFtSBigRBhRd7
- 다음 실험 설계 (추적 감독, 학습 전): `TRACKING_SUPERVISION_PLAN_2026-10-31.md`

| 위치 | 무엇 |
|---|---|
| `datagen/` | 데이터 생성 단계별 스크립트 (S0 클립 · S1–S3 SAM2 물체 · S4 가림막 은행 · S5 합성 · S7 보기) |
| `run_samples.sh` | 샘플 생성 한 번에 (S0 → S1 → S5 → S7) |
| `samples/` | 보기용 GIF · PNG · index.html (git 무시) |
| `eval/` | 결과 표 (`summarize_v1.py`) · 페이지 예시 그림 (`make_page_examples.py`) |
| `results/` | 같은 쌍 비교표 |

데이터 본체는 레포 밖: `/workspace/dataset/training_data/imagine_seg_v0_samples/{ssv2,k400}/<clip>/`
가림막 은행: `/workspace/dataset/assets/occluder_bank_coco_v0/` (COCO val2017 에서 자름)
SAM2: 코드 `/workspace/tools/sam2` (`/venv/vjepa` 에 `--no-deps` 설치), 가중치 `checkpoint/sam2/sam2.1_hiera_base_plus.pt`
