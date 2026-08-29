# configs/protocols — 프로토콜 × 데이터셋 × 모델

```bash
bash z_research/scripts/run.sh --list                    # 뭐가 있는지
GPUS=4 bash z_research/scripts/run.sh <프로토콜> <데이터셋> [모델]
DRYRUN=1 bash z_research/scripts/run.sh attn_probe v8    # 병합 결과만 (GPU 안 씀)
```

세 조각이 실행 직전에 합쳐진다 (`z_research/scripts/harness/resolve.py`):

| 조각 | 담는 것 | 파일 |
|---|---|---|
| **프로토콜** | 프레임 배치, 채점 규칙, probe 정의, dtype 관례 | `<이름>.yaml` |
| **데이터셋** | 경로, 인덱스, 컬럼 이름 | `datasets.md` |
| **모델** | 체크포인트, arch_name | `models.md` |

**데이터를 바꿀 때 고치는 곳은 `datasets.md` 섹션 하나뿐이다.** yaml 은 건드리지 않는다.

## 프로토콜

| 이름 | 무엇을 재나 | 확립된 수치 |
|---|---|---|
| `intphys1_sliding` | IntPhys1 Garrido 공식 sliding | **88.89%** (intphys1_dev, vith, 180 pair) |
| `surprise_c16t32` | fixed context16 / target32 latent-L1 | **79.10%** (v8, vith, 1024 pair) |
| `attn_probe` | z / p / h 세 지점 attentive probing | z·h 100%, p 90.2%, h→p 48.0% (v8, shape) |

## 병합 규칙 — 프로토콜이 이긴다

```
data  = { **datasets.md[<데이터셋>], **프로토콜.data  }
model = { **models.md[<모델>],       **프로토콜.model }
```

프로토콜이 소유한 키(`n_frames`, `resolution`, probing 의 `index_csv`/`type_column` 등)는
레지스트리에 뭐가 있든 프로토콜 값이 쓰인다. 레지스트리에만 있는 키
(`root`, `frames_pattern`, `block_column` …)는 그대로 채워진다.
`n_frames: RAW` 는 그 데이터셋의 `raw_frames` 로 치환된다 (영상 전체를 쓰는 프로토콜용).

자동으로 정해지는 것 (`TAG=` / `OUTDIR=` 으로 덮을 수 있다):

```
tag        = <데이터셋.cache_tag>_<모델>                             예: v10_vith
output_dir = <데이터셋.results_root>/<프로토콜>__<데이터셋>_<모델>
```
`results_root` 도 `datasets.md` 에서 온다 (데이터셋별 아카이브 자리):

| 데이터셋 | 결과가 쌓이는 곳 |
|---|---|
| `intphys1_dev` | `z_research/IntPhys/exp_results/` |
| `v8`, `v8_halfsize` | `z_research/IntPhysGenV8/exp_results/` |
| `v10` | `z_research/IntPhysGenV10/exp_results/` |
| 그 밖 | `z_exp/world_model_analysis/results/` (기본값) |

`tag` 는 **토큰 캐시의 이름**이다. 프로토콜이 달라도 (데이터셋, 모델)이 같으면 같은
캐시를 쓰라고 일부러 프로토콜을 뺐다. 병합된 config 는 `<output_dir>/_resolved.yaml`
로도 남는다 (`summary.json` 에도 전체가 들어간다).

## 절대 어기지 말 것

- **`scoring.pairing: matched`** — 기본값이 `cross` 라 빠뜨리면 공식과 다른 전수 4쌍 채점이
  조용히 돌아간다. matched 여야 두 후보의 context 가 픽셀 단위로 같아 `p` 가 비트 단위로
  동일해지고(실측 `max|p_pos−p_imp| = 0.0`), 채점이 순수 기하 문제로 환원된다.
- **프로토콜을 밝히지 않은 수치는 비교 금지** — 같은 IntPhys1 dev 에서
  sliding 88.89% vs fixed 66.67%(22.2pt). v6 static+occluded shape 은
  fixed 46.88% vs sliding max 90.62%(43.7pt). sliding best 는 같은 그리드에서 고른 값이라
  held-out 추정치가 아니라 descriptive 이므로 그렇게 명시할 것.
- **probing 은 가능(possible) 변이만** — `block_types: [obj]`. 이유는 `attn_probe.yaml` 주석.
- **split 은 block 단위** — block 안 4개는 context/future 를 2×2 로 공유한다.
- **dtype 관례가 갈린다** — surprise 는 `float32 + autocast float16`(공식 Garrido 재현),
  probing 은 `bfloat16`. 섞지 말 것.
- **토큰 캐시는 로컬 디스크에만** — `/data` 는 NFS(쓰기 57MB/s)라 셋당 40GB 캐시를 두면 10분을 버린다.
- **기준 모델은 ViT-H** — 자체 데이터에는 ViT-L 이 이기는 축이 있지만 사후 모델 선택을
  하지 않는다는 원칙이다. "ViT-H 가 항상 낫다"는 거짓이니 그렇게 쓰지 말 것.

## 캐시 함정

`TokenCache.matches()` 는 `video_ids` / 토큰수 / dtype 만 본다.
**체크포인트·해상도·autocast·재렌더는 서명에 없다.** 데이터를 다시 만들면 캐시도 반드시 치울 것
(보관 관례: `cache/_previous_YYYYMMDD/<tag>`).

## 새 데이터셋 추가

`datasets.md` 에 `## 이름` 섹션 하나. `run.sh` 가 모델을 로드하기 전에
경로·인덱스 실물, `resolution == img_size`, 프레임 예산을 검사하고 안 맞으면 죽는다
(ViT-H 로딩은 프로세스당 ~2분이라 그 전에 죽는 게 훨씬 싸다).

## 검증 습관

수치는 문서가 아니라 `<output_dir>/summary.json` 에서 다시 계산해 확인하고,
그렇게 했다고 문서에 밝힌다. 자세한 내용은 레포 루트 `CLAUDE.md`.

---

## 실험 케이스마다 yaml 을 만들지 않는다

프로토콜 yaml 은 **재는 방식**만 담는다. 데이터셋마다 달라지는 것, 한 번만 쓰는 변형은
yaml 을 새로 뜨지 말고 아래 셋으로 처리한다.

### 1. `fit_groups_sweep: auto`

데이터의 `group_column` 값을 읽어 `[null, [g1], [g2], ...]` 를 만든다.
`block_types` 에 해당하는 행이 없는 group 은 자동으로 빠진다 (학습셋 0 이면 `eval.py` 가 죽는다).

```
attn_probe v10        -> sweep 5개 (4조건 + pooled)
attn_probe v10_flat   -> sweep 3개 (2조건 + pooled)
```

조건 구성이 다른 데이터셋마다 프로토콜을 복제하던 것을 없앤다.

### 2. `SET=` — 점 경로 덮어쓰기

병합된 config 를 명령줄에서 고친다. 값은 YAML 로 읽고, **`null` 은 그 키를 지운다.**

```bash
# static_occlusion 의 color predictor head 만 50 epoch 으로
SET="probing.optim=attn_50      probing.optims.attn_50={num_epochs:50,batch_size:16,lr:0.0001,weight_decay:0.05}      probing.fit_groups_sweep=[[static_occlusion]]      probing.runs=[{train:{model:predictor},eval:[self]}]      probing.targets.shape=null probing.targets.env=null" GPUS=8 bash z_research/scripts/run.sh attn_probe v10 vith
```

`resolve.py --set` 로도 같다. `DRYRUN=1` 로 병합 결과만 먼저 보는 습관을 들일 것.

### 3. `extends:` — 구조가 다른 실험만

부모 프로토콜을 상속한다. **자식이 이기고, dict 는 재귀 병합, list 는 교체,
`null` 은 그 키를 삭제.** 새 yaml 을 뜨는 기준은 하나다 —

> **재는 방식 자체가 다른가?** (다른 index, 다른 표현 지점, 다른 라벨 의미)

`attn_probe_imp` 가 그 예다: 불가능 변이를 대상에 넣고 미래/문맥 절반의 라벨을 따로 둔다.
반면 "조건 목록이 다르다", "epoch 을 늘린다", "target 하나만 본다" 는 전부 1·2 번으로 처리한다.

### 현재 프로토콜 4개

| | 무엇 |
|---|---|
| `surprise_c16t32` | fixed context16 / target32 latent-L1 채점 |
| `intphys1_sliding` | IntPhys1 Garrido 공식 sliding |
| `attn_probe` | z / p / h 세 지점 attentive probing |
| `attn_probe_imp` | 불가능 변이에서 target encoder 가 바뀐 정체성을 읽는가 |

2026-08-29 에 `attn_probe_flat`(sweep 만 다름)과 `attn_probe_e50_staticocc`(optim/runs/targets 만 다름)를
지웠다. 둘 다 위 1·2 번으로 재현된다.
