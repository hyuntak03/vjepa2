# z_research/scripts

**최상위에는 사람이 직접 치는 것만 둔다.** 나머지는 역할별 폴더로 내린다.

```
run.sh          표준 진입점.  GPUS=8 bash z_research/scripts/run.sh <프로토콜> <데이터셋> [모델]
sbatch.sh       위를 SLURM 으로 제출
monitor.sh      watch -n 1 bash z_research/scripts/monitor.sh [job id | 로그경로]

harness/        run.sh 가 부르는 것
  resolve.py      프로토콜 + datasets.md + models.md 를 병합. --set / extends / auto sweep

data/           인덱스·데이터 준비
  build_rollout2_index.py     RollOut_v2 인덱스 — 라벨은 metadata 가 아니라 plan 에서 (검증 15항목)
  build_probe_imp_index.py    불가능 변이를 probing 대상으로 여는 인덱스
  build_rollout_index.py      RollOut_v1 인덱스 + 라벨 재료 검증 14항목

figures/        논문 그림. 전부 산출물에서 재계산해 summary.json 과 대조 검증한다
  plot_v11_surprise.py      v11 채점 그림 전부.  상단 FIGDIR 표가 하위 폴더를 배정한다
                            (01_condition / 02_occlusion_k / 03_direction / 04_object_order)
                            표에 없는 이름은 최상위에 떨어진다 — 새 그림이 눈에 띄라고 일부러
  plot_v11_probing.py       v11 probing 그림.  324칸 전수 대조 후 by_condition/ by_k/ 로
  plot_v11_occtiming.py     가림 타이밍(early/mid/late) 그림.  **두 run 을 합친 report** 를 받는다
                            (v11 본체 + v11_earlymid). FIGDIR: 01_timing / 02_timing_k / 03_object_order
  plot_rollout2_overlay.py  RollOut_v2 미래 프레임 위에 정답/p/h 위치 마커
  plot_v11_position.py      v11_full 위치·속도 readout 그림 (v11_roll_out/figures/position/). --fit-by condition|motion|all
  plot_v11_knockout.py      attention knockout 그림 (06_knockout/ + by_violation/ by_motion/ by_timing/ by_k/).
                            입력은 knockout merge.py 의 results.json 하나. 축별은 `breakdown` 을 block 가중으로 합쳐 쓴다
  plot_violation_bars.py    채점, 조건 x 위반          (--flat-style green|hatch)
  plot_direction_bias.py    방향 비대칭                (--by-condition 로 2x2)
  plot_probing_bars.py      z/h/p probing              (--line, --group-fit, --head)
  plot_confusion.py         predictions.json -> confusion matrix
  plot_vanish_direction.py  vanish 방향별
  plot_intphys1_bars.py     IntPhys1 채점
  plot_rollout_position.py  RollOut_v1 위치·속도 2패널. json 의 R² 4개를 대조하고 어긋나면 죽는다

analysis/       산출물·토큰 캐시 기반 분석. **전부 GPU 불필요**
  report.py                   summary.json 검증 -> report.json (그림·문서의 단일 입력)
  merge_probe_runs.py         쪼개서 제출한 probing job 을 합침. val_video_ids 가 다르면 죽는다
  probing_md.py               RESULTS_*.md 의 probing 절(표 G-J)을 재생성
  knockout_md.py              knockout results.json -> 축별 markdown 표. `--write` 로 ablation README 전수 기록 11 교체
  rollout2_position.py        RollOut_v2 pooled 위치 readout 4종 (p_A/p_B/h/z) + 두 미래 판정 + null
  rollout2_token_decoder.py   RollOut_v2 토큰 단위 위치 decoder (h 학습 → p 이식). 1,281 파라미터
  v11_position.py             v11_full 미래 8슬롯에서 물체 위치·속도·가속 readout (RollOut 측정의 v11 이식, 조건별 + 공유 fit)
  alpha_amplify.py            증폭 개입의 천장 (--anchor mu|z).  기각된 개입 (천장 51~65%)
  concept_separability.py     Fisher / ridge / 개념 벡터 정렬 (--align)
  confusion_vs_surprise.py    probe confusion x 채점 방향 상관
  is_p_just_context.py        p 가 문맥의 복사인가
  pca_spectrum.py             주성분 스펙트럼
  step_direction.py           걸음의 방향
  typicality.py               전형성 가설 (기각됨)
  rollout_position.py         RollOut_v1 위치·속도 readout (h→h / p→p, null 3종).
                              절차·함정은 z_research/RollOutV1/Archive/METHOD_POSITION_READOUT_2026-09-06.md
  intphys1_direction_audit.py IntPhys1 방향 균형 감사
  retrieval_confusion.py      block 밖 7-way retrieval. **기각됨** — 최상단 주석을 읽을 것
  retrieval_pooled.py         위의 pooled 판. 같이 기각

slurm_logs/     .gitignore
```

## 새 스크립트를 만들 때

1. **위 네 폴더 중 하나에 넣는다.** 최상위에 새 파일을 두지 않는다.
   맞는 폴더가 없으면 폴더를 새로 만들고 이 README 에 한 줄 추가한다.
2. **docstring 첫 줄에 무엇을 재는지, 마지막에 실행 예시**를 적는다.
3. **수치를 내는 스크립트는 산출물과 대조 검증**한다 (`figures/` 는 전부 그렇게 한다).
4. 임시 출력은 레포 루트에 흘리지 말고 `/tmp` 나 스크래치패드에 쓴다.

## 그림을 새로 만들 때

`plot_v11_surprise.py` 는 상단 **`FIGDIR`** 표로 하위 폴더를 배정한다.
파일만 옮기면 다음 실행에서 도로 흩어지므로 **표를 같이 고친다.**
대체된 그림은 지우지 않고 `_superseded` 로 보내고, `figures/surprise/README.md` 에
**왜 물러났는지** 적는다.

## 실험 케이스마다 프로토콜 yaml 을 만들지 않는다

`configs/protocols/README.md` 참고. 요약:

- `fit_groups_sweep: auto` — 데이터의 group 을 읽어 sweep 자동 생성
- `SET="a.b=1 c.d=null"` — 병합된 config 를 점 경로로 덮어씀 (`null` = 키 삭제)
- `extends:` — **재는 방식 자체가 다를 때만** 새 yaml
