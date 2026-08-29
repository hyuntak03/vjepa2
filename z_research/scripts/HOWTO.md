# 새 데이터셋 하나 돌리기

## 1. 인덱스 만들기

```bash
python z_scripts/world_model_analysis/build_intphysgen_v6_index.py \
  --source /local_datasets/world/world_analysis/<렌더폴더>/metadata.csv \
  --output data_csv/<이름>/index.csv \
  --expected-clips <N> --expected-blocks <N/4> \
  --block-column block --strict-context

python z_scripts/world_model_analysis/build_probe_index.py <이름>   # probing 용. SETS 에 항목 추가 필요
```

`--strict-context` 는 문맥 프레임 전수 감사. **mismatch 0 이어야 한다.**
새 `condition` 이름을 쓰면 `build_intphysgen_v6_index.py` 의 `REQUIRED_CONDITIONS` 에 추가.

## 2. `configs/protocols/datasets.md` 에 섹션 추가

```markdown
## <이름>

한 줄 설명. 무엇이 다른지, 어떤 교란이 있는지.

raw_frames: 100
cache_tag: <이름>
results_root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/<셋>/exp_results
root: /data/hyuntak/project/2026/2027_cvpr/vjepa2/data_csv/<이름>
index_csv: index.csv
frames_root: /local_datasets/world/world_analysis/<렌더폴더>
frames_pattern: "{file_name}/{frame:06d}.png"
frames_start: 0
frames_stride: 3
block_column: block_id
pair_column: pair_id
variant_column: variant
plausible_column: plausible
type_column: condition
```

`cache_tag` 가 토큰 캐시 이름(`<cache_tag>_<모델>`). **데이터를 다시 만들면 캐시도 치울 것.**
`results_root` 생략 시 `z_exp/world_model_analysis/results`.

## 3. 확인

```bash
DRYRUN=1 bash z_research/scripts/run.sh surprise_c16t32 <이름> vith   # GPU 안 씀, 몇 초
bash z_research/scripts/run.sh --list                                 # 등록된 것 보기
```

## 4. 실행

```bash
# 채점 — 토큰 캐시 안 씀
GPUS=8 BATCH_SIZE=16 DECODE_WORKERS=10 \
  bash z_research/scripts/run.sh surprise_c16t32 <이름> vith

# probing — 캐시 추출(~11분/8192clip) 후 head 학습
GPUS=8 bash z_research/scripts/run.sh attn_probe <이름> vith
```

## 5. SLURM

```bash
sbatch --job-name=<이름> --export=ALL,P=<프로토콜>,D=<데이터셋>,M=vith,GPUS=8 \
       z_research/scripts/sbatch.sh

watch -n 1 bash z_research/scripts/monitor.sh          # 최신 job
watch -n 1 bash z_research/scripts/monitor.sh 189548   # job id 지정
```

`sbatch.sh` 는 `-w vll5` 고정(데이터가 노드 로컬). `--gres` 와 `GPUS` 를 맞출 것.

---

## 자주 쓰는 변형 — **yaml 새로 만들지 말 것**

```bash
# 일부만 돌리기 / 설정 바꾸기
SET="probing.optim=attn_50 probing.targets.shape=null" \
  GPUS=8 bash z_research/scripts/run.sh attn_probe <이름> vith

LIMIT=2 SMOKE=1 BATCH_SIZE=8 bash ... run.sh surprise_c16t32 <이름>   # 배관 점검
RECACHE=1 bash ...                                                    # 캐시 무시
TAG=xx OUTDIR=yy bash ...                                             # 이름 직접 지정
```

`SET` 은 점 경로, 값은 YAML, **`null` 은 키 삭제**. `fit_groups_sweep: auto` 라
조건 개수가 달라도 프로토콜은 그대로 쓴다. 자세한 건 `configs/protocols/README.md`.
