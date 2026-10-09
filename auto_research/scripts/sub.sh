#!/bin/bash
# 대화형 srun 셸 안에서 부를 때 SLURM_* 변수가 새 job 에 새지 않게 지우고 sbatch 한다.
for v in $(env | grep -o '^SLURM[^=]*'); do unset "$v"; done
exec sbatch "$@"
