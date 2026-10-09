#!/bin/bash
# vll3 에서 CPU 로 한 명령 (srun6.sh 의 vll3 판). 사용: srun3.sh <cpus> <mem> <명령...>
for v in $(env | grep -o '^SLURM[^=]*'); do unset "$v"; done
c=$1; m=$2; shift 2
exec srun -w vll3 -p debug_vll -c "$c" --mem "$m" -t 60 "$@"
