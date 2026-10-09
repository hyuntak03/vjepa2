#!/bin/bash
# vll6 에서 CPU 로 한 명령 (SLURM_* 을 지우고 새 allocation). 사용: srun6.sh <cpus> <mem> <명령...>
for v in $(env | grep -o '^SLURM[^=]*'); do unset "$v"; done
c=$1; m=$2; shift 2
exec srun -w vll6 -p debug_vll -c "$c" --mem "$m" -t 60 "$@"
