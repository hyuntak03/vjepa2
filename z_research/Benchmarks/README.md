# Benchmarks — Garrido 프로토콜 × 세 모델 (시작점)

> **새 세션은 이 파일부터 읽는다.** 설정의 정본은 `PROTOCOLS.md`, 레포 규칙은 루트 `CLAUDE.md`.
> 전면 재작성 2026-09-21. 그 이전 기록은 §6 에 무엇이 왜 폐기됐는지만 남겼다.

## 0. 30초 요약

**세 모델을 Garrido 프로토콜로 세 벤치마크에 걸어 표 하나를 만든다.**

| 축 | 값 |
|---|---|
| 모델 | **V-JEPA 2 ViT-H** · V-JEPA 2.1 ViT-g · VideoMAEv2-g |
| 벤치마크 | **IntPhys 1 dev** · GRASP level2 · InfLevel-lab (continuity / gravity / solidity) |
| 프로토콜 | Garrido et al. sliding — `PROTOCOLS.md` 하나만 본다 |
| 지표 | **쌍 비교 AvgSurprise 정확도** 하나, chance 50% (Max 철회) |

**IntPhys 2 프로토콜은 범위 밖이다** (`PROTOCOLS.md` §7).

## 1. oracle — 하네스가 맞는지 먼저 확인한다

```bash
python z_research/scripts/analysis/check_oracle.py
```

기준값: **IntPhys 1 dev × V-JEPA 2 ViT-H = 88.89 %**
(A.8 격자 최고 `skip2_w32`, **Filtered** — IntPhys 은 C 스윕을 Filtered 로 대체한다.
property 별 O1 85.00 / O2 96.67 / O3 85.00 → macro 88.89).
참고로 노트북 규칙({Filtered}∪{C} 최고)이면 90.56 인데, 그건 논문 텍스트와 어긋난다 (`PROTOCOLS.md` §3).
외부 대조는 IntPhys 2 논문 Table 2 의 IntPhys 열, V-JEPA 2-h **87.22 %** (157/180) — 프로토콜이 다르므로
같을 이유는 없고 **2pt 안**이면 합격으로 본다 (실측 차이 1.67pt).

검사는 21 항목이다: A.8 격자·stride·축약·pairing·해상도 일치, 창마다 모델이 그 프레임 수로 지어졌는지,
하네스 `summary.json` 값, 그리고 **`per_window.json` 에서 독립으로 다시 계산한 값**까지 맞는지.
**하네스·config·로더를 건드리면 이것부터 돌린다.**

## 2. 실행 — 진입점은 하나다

```bash
# 안 된 칸 전부 (창 16·32 를 각각 따로 돈다)
GPUS=8 bash z_research/Benchmarks/run_all.sh

# 골라서
GPUS=8 BENCHES="intphys1_dev" MODELS="vith" bash z_research/Benchmarks/run_all.sh
GPUS=8 WINDOWS="32" BENCHES="grasp_level2" bash z_research/Benchmarks/run_all.sh

# SLURM
sbatch --nodelist=vll3 --gres=gpu:8 --cpus-per-task=48 --mem=200G \
  --export=ALL,BENCHES=grasp_level2,MODELS="vith vjepa21g videomae2g",GPUS=8 \
  z_research/Benchmarks/bench.sbatch

watch -c -n 1 bash z_research/Benchmarks/monitor.sh        # ETA
```

**창(C+M)마다 실행이 나뉜다.** 공식 코드가 창마다 모델을 그 프레임 수로 짓기 때문이다
(`PROTOCOLS.md` §2). 결과 폴더는 `intphys1_sliding__<데이터셋>_<모델>_w{16,32}` 이고
A.8 최고값은 재채점기가 창을 가로질러 고른다.

**모델을 추가하려면 두 곳만 고친다** — `configs/protocols/models.md` 에 섹션 하나,
`analysis/model_loaders.py` 의 `BUILDERS` 에 한 줄. 채점기·데이터셋·런처는 건드리지 않는다.

**데이터를 추가하려면 한 곳만 고친다** — `configs/protocols/datasets.md` 에 섹션 하나.

## 3. 채점 — 공식 축약 규칙으로 다시 계산한다

```bash
python z_research/scripts/analysis/garrido_rescore.py            # 창 합쳐 (데이터셋, 모델) 마다
python z_research/scripts/analysis/bench_table.py --readme       # 아래 표 갱신
```

채점기는 `context_reduce: min`(=Filtered) 하나로만 돌지만, 공식 규칙은 **데이터셋마다 다르다** —
**IntPhys 은 Filtered**, **GRASP·InfLevel 은 property 마다 최고 C**, 그리고 **property macro 평균**
(`PROTOCOLS.md` §3). IntPhys 은 n 이 같아 하네스 값과 우연히 일치하지만 나머지는 다르다.
`per_window.json` 에 `(combo, 창 시작, C, surprise)` 가 전부 남으므로 **재실행 없이** 고쳐 잡는다.

## 4. 결과

**지표는 하나다 — 쌍 비교 AvgSurprise 정확도 (%), chance 50** (`PROTOCOLS.md` §1-b).
MaxSurprise 는 2026-09-21 에 철회했다 (그건 단일 영상 + AUROC 용이다). 아래 표는 스크립트가 산출물에서 계산해 박은 것이라
**손으로 고치지 않는다.**

<!--TABLE:start-->
| 벤치마크 | 최고 설정 | n_pair | V-JEPA 2 ViT-H | V-JEPA 2.1 ViT-g | VideoMAEv2-g |
|---|---|---:|---:|---:|---:|
| IntPhys 1 dev | skip2_w32 · skip5_w16 | 180 | **88.89** | **51.11** | **54.44** |
| GRASP level2 | — | — | *미완* | *미완* | *미완* |
| InfLevel-lab (3 property macro) | — | — | *미완* | *미완* | *미완* |
|   ┗ continuity | skip10_w16 | 1116 | **72.85** | *미완* | *미완* |
|   ┗ solidity ¹ | — | — | *미완* | *미완* | *미완* |
|   ┗ gravity ¹ | — | — | *미완* | *미완* | *미완* |

선택된 A.8 칸 (property 별 축약은 `garrido_rescore.py` 출력 참고):

| 벤치마크 | V-JEPA 2 ViT-H | V-JEPA 2.1 ViT-g | VideoMAEv2-g |
|---|---|---|---|
| IntPhys 1 dev | `skip2_w32` | `skip2_w32` | `skip5_w16` |
| GRASP level2 | — | — | — |
| InfLevel-lab (3 property macro) | — | — | — |
|   ┗ continuity | `skip10_w16` | — | — |
|   ┗ solidity ¹ | — | — | — |
|   ┗ gravity ¹ | — | — | — |

¹ **원리적으로 못 푸는 property** — 컵 상태가 공식 로더가 잘라내는 priming 구간에만 나온다
(Garrido App. E). 50 근처가 정상이고 모델 비교에 쓰지 않는다.
<!--TABLE:end-->

### 표에 반드시 붙는 단서

1. **InfLevel gravity·solidity 는 원리적으로 못 푼다** — 컵이 잘렸는지가 공식 로더가 **잘라내는**
   priming 구간에만 나온다 (Garrido App. E). 50 근처가 정상이고 **모델 비교에 쓰지 않는다.**
   믿을 수 있는 InfLevel property 는 **continuity 뿐**이다.
2. **세 모델의 surprise 절대값은 비교 불가**다 (예측 공간이 latent 1280 / latent 5632 / 픽셀 1176).
   비교 가능한 것은 **쌍 판별 정확도뿐**이고, Garrido 도 같은 이유로 정확도만 비교했다.
3. **2.1-g 는 384 가 공식 해상도인데 우리는 256 으로 쟀다.**
4. **A.8 최고값은 held-out 추정치가 아니라 descriptive 다** — 같은 데이터에서 고른 값이다.
   논문도 그렇게 하고 `PROTOCOLS.md` §2 에 근거를 적었다.
5. **2.1-g 는 4중항 안에서 점수가 거의 안 움직인다** — IntPhys1 에서 블록 90 개 중 80 개가 정확히
   0.5 (무작위면 45), 블록 안 σ / 블록 사이 σ = **0.04** (ViT-H 0.23). 원인 미규명.
   타깃 차원 희석(5632 = 층 10/20/30/40 concat)을 의심해 최종층만으로 채점해 봤으나
   **더 낮아 기각**했다. 이 열은 그 단서를 달고 읽는다.

## 5. 폴더

| 위치 | 역할 |
|---|---|
| `PROTOCOLS.md` | **설정의 정본.** 공식 코드 줄번호까지 |
| `run_all.sh` | 실행 진입점 (창별로 나눠 돈다) |
| `bench.sbatch` · `monitor.sh` | SLURM 제출 · ETA |
| `stage.sh` · `stage.sbatch` | tar 데이터를 노드 로컬 `/data2` 에 푼다 (GRASP·IntPhys2) |
| `exp_results/<프로토콜>__<데이터셋>_<모델>_w{N}/` | 원시 산출물 (git 무시) |
| `exp_results/_invalid_*` · `_superseded_*` | 폐기한 산출물. **지우지 않고 사유를 README 에 남긴다** |
| `Archive/` | 분석 문서 `TOPIC_YYYY-MM-DD.md` |

관련 스크립트 (`z_research/scripts/analysis/`):
`check_oracle.py` (기준값 검사) · `garrido_rescore.py` (공식 축약 재채점) · `bench_table.py` (표 생성)
데이터 인덱스: `z_research/scripts/data/build_{grasp,inflevel}_index.py`

## 6. 2026-09-21 에 무엇이 틀렸었나 (기록)

이전 수치는 **전부 폐기**했다. 산출물은 지우지 않고 `exp_results/_invalid_*` 에 사유와 함께 남겼다.

| # | 무엇 | 왜 틀렸나 | 폐기된 산출물 |
|---|---|---|---|
| 1 | **Table S3 한 칸만 돌림** | Table S3 는 **그들 모델**(`vit_huge_rope`, fpc 16)에 A.8 탐색을 돌린 *결과*다. V-JEPA 2 는 fpc 64 @256 이라 다시 탐색해야 한다. IntPhys1 83.33 → **88.89** | `_superseded_20260921_singlecell/` |
| 2 | **한 실행에 창 16·32** | 공식은 창마다 모델을 그 프레임 수로 짓는다. VideoMAEv2 는 즉시 죽고(`2048 vs 4096`), V-JEPA 는 공식과 다른 위치 인코딩으로 돈다 | `_invalid_20260921_mixedgrid/` |
| 3 | **InfLevel 쌍 목록 자작** | 정본은 공식 `auxiliary_data_loading_files/inflevel/{prop}.csv`, 한 줄이 한 쌍 | `_invalid_20260921_inflevel/` |
| 4 | **InfLevel 채점 cross(2×2)** | 공식은 `matches=[[0,1]]` — 한 줄에 한 비교. 쌍 수가 2 배로 부풀어 있었다 | 〃 |
| 5 | **InfLevel priming 미절단** | 공식은 `end_priming_2+1` 부터 읽는다. `data.py` 의 mp4 경로가 `frames_start`/`frames_stride` 를 무시하던 버그도 함께 고쳤다 | 〃 |
| 6 | **`MB21` 미적용** | `run_all.sh` 에 변수 선언만 있고 `SET` 에 싣는 줄이 없었다. 2.1-g OOM 세 번이 전부 `max_batch: 48` 로 돌았다 | — |
| 7 | **VideoMAEv2 창 16 고정** | 시간 위치 임베딩은 체크포인트에 없고 `get_sinusoid_encoding_table` 로 **생성**된다. 32 도 돈다 | — |
| 8 | **GRASP·InfLevel 에 min 축약** | 공식은 그 둘에 **property 별 최고 C**. `garrido_rescore.py` 가 재실행 없이 고쳐 잡는다 | — |

## 재현

```bash
python z_research/scripts/analysis/check_oracle.py               # 기준값 검사 (GPU 불필요)
GPUS=8 bash z_research/Benchmarks/run_all.sh                     # 안 된 칸 전부
python z_research/scripts/analysis/garrido_rescore.py            # 공식 축약 재채점
python z_research/scripts/analysis/bench_table.py --readme       # §4 표 갱신
```
