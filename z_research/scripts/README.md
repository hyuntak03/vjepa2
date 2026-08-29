# z_research/scripts

**최상위에는 사람이 직접 치는 것만 둔다.** 나머지는 역할별 폴더로 내린다.

```
run.sh          표준 진입점.  GPUS=8 bash z_research/scripts/run.sh <프로토콜> <데이터셋> [모델]
sbatch.sh       위를 SLURM 으로 제출
monitor.sh      watch -n 1 bash z_research/scripts/monitor.sh [job id | 로그경로]

harness/        run.sh 가 부르는 것
  resolve.py      프로토콜 + datasets.md + models.md 를 병합. --set / extends / auto sweep

data/           인덱스·데이터 준비
  build_probe_imp_index.py    불가능 변이를 probing 대상으로 여는 인덱스

figures/        논문 그림. 전부 산출물에서 재계산해 summary.json 과 대조 검증한다
  plot_violation_bars.py    채점, 조건 x 위반          (--flat-style green|hatch)
  plot_direction_bias.py    방향 비대칭                (--by-condition 로 2x2)
  plot_probing_bars.py      z/h/p probing              (--line, --group-fit, --head)
  plot_confusion.py         predictions.json -> confusion matrix
  plot_vanish_direction.py  vanish 방향별
  plot_intphys1_bars.py     IntPhys1 채점

analysis/       토큰 캐시 기반 분석. **전부 GPU 불필요**
  alpha_amplify.py            증폭 개입의 천장 (--anchor mu|z)
  concept_separability.py     Fisher / ridge / 개념 벡터 정렬 (--align)
  confusion_vs_surprise.py    probe confusion x 채점 방향 상관
  is_p_just_context.py        p 가 문맥의 복사인가
  pca_spectrum.py             주성분 스펙트럼
  step_direction.py           걸음의 방향
  typicality.py               전형성 가설 (기각됨)
  intphys1_direction_audit.py IntPhys1 방향 균형 감사

slurm_logs/     .gitignore
```

## 새 스크립트를 만들 때

1. **위 네 폴더 중 하나에 넣는다.** 최상위에 새 파일을 두지 않는다.
   맞는 폴더가 없으면 폴더를 새로 만들고 이 README 에 한 줄 추가한다.
2. **docstring 첫 줄에 무엇을 재는지, 마지막에 실행 예시**를 적는다.
3. **수치를 내는 스크립트는 산출물과 대조 검증**한다 (`figures/` 는 전부 그렇게 한다).
4. 임시 출력은 레포 루트에 흘리지 말고 `/tmp` 나 스크래치패드에 쓴다.

## 실험 케이스마다 프로토콜 yaml 을 만들지 않는다

`configs/protocols/README.md` 참고. 요약:

- `fit_groups_sweep: auto` — 데이터의 group 을 읽어 sweep 자동 생성
- `SET="a.b=1 c.d=null"` — 병합된 config 를 점 경로로 덮어씀 (`null` = 키 삭제)
- `extends:` — **재는 방식 자체가 다를 때만** 새 yaml
