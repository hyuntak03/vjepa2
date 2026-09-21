# Benchmarks — 외부 직관물리 벤치마크 × 세 모델 (시작점)

> **새 세션은 이 파일부터 읽는다.** 레포 전체 규칙은 루트 `CLAUDE.md`, 실행 규약은 `configs/protocols/README.md`.
> 개설 2026-09-20. **아직 수치 없음 — 데이터 스테이징 + 하네스 구축 단계다.**

## 0. 무엇을 하는 세트인가

**세 모델을 같은 벤치마크에 같은 프로토콜로 걸어 표 하나를 만든다.**

| 모델 | 체크포인트 | 예측 공간 | 상태 |
|---|---|---|---|
| **V-JEPA 2 ViT-H** | `checkpoint/models--facebook--vjepa2-vith-fpc64-256/.../model.pth` | latent 1280 | 기존 하네스로 바로 됨 |
| **V-JEPA 2.1 ViT-g** | `checkpoint/v_jepa2.1/vjepa2_1_vitg_384.pt` (16.9 GB) | latent **5632** (4단계 concat) | 백엔드 신규 필요 |
| **VideoMAEv2-g** | `checkpoint/videomae2/models--OpenGVLab--VideoMAE2/.../mae-g/vit_g_hybrid_pt_1200e.pth` (4.1 GB) | **픽셀 1176** | 백엔드 신규 필요 |

⚠️ **세 모델의 surprise 절대값은 서로 비교할 수 없다.** 예측 공간이 1280 latent / 5632 latent / 1176 정규화픽셀로
전부 다르다. **비교 가능한 것은 쌍 판별 정확도뿐이다.** Garrido et al. 도 같은 이유로 정확도만 비교했다.

벤치마크 (사용자 지시 2026-09-20):

| 벤치마크 | 프로토콜 | 규모 | 노드 |
|---|---|---|---|
| IntPhys 1 dev | IntPhys1 공식 (Garrido sliding) 그대로 | 360 영상 / 90 4중항 | vll3 |
| IntPhys 2 **Main** | IntPhys 2 공식 그대로 (growing context, M=48) | 1,012 영상 | vll3 |
| GRASP (level2) | Garrido Table S3 setting | 4,096 영상 / 16 시나리오 | vll3 |
| InfLevel-lab | Garrido Table S3 setting + 공식 2×2 채점 | 5,772 영상 / 3 property | vll3 |
| IntPhysGen v11 | `surprise_c16t32` (문맥 16 / 미래 16) | 21,504 clip | **vll5** (데이터가 거기만) |

## 1. 읽는 규칙

1. **V-JEPA 2 ViT-H 의 IntPhys1 = 88.89% 가 하네스 검증선이다.** 새로 만든 경로가 이 수치를 재현하지
   못하면 다른 수치도 믿지 않는다 (`z_research/IntPhys/exp_results/`).
2. **프로토콜을 밝히지 않은 숫자는 비교 금지** (CLAUDE.md §1-3). 같은 IntPhys1 dev 에서
   sliding 88.89% vs fixed 66.67% 로 22.2pt 가 흔들린다.
3. **Garrido 의 "context 를 property 마다 최적화" 는 descriptive 다** — held-out 추정치가 아니다.
   그대로 따라 하되 그렇게 명시한다.
4. **InfLevel gravity·solidity 는 원리적으로 못 푸는 조건이다** (§3-4). continuity 만 깨끗하다.
5. **V-JEPA 2.1 은 384 가 공식 해상도인데 우리는 256 으로 잰다** (§4). 이 한 줄을 결과 표마다 붙인다.

## 2. 데이터 — 어디서 읽나

사용자 규칙: **tar 로 있는 것은 노드 로컬에 풀고, 폴더로만 있는 것은 `/data` 에서 바로 읽는다.**

| 벤치마크 | 원본 | 노드 로컬 (vll3) | 크기 |
|---|---|---|---|
| GRASP | `/data/dataset/world/Benchmarks/GRASP.tar` | `/data2/local_datasets/world/Benchmark/GRASP` | 7.2 GB |
| IntPhys1 dev | `.../IntPhys1_dev.tar.gz` | `.../Benchmark/IntPhys1_dev/dev/` (**scene PNG 만**) | 3.0 GB |
| IntPhys2 Main | `.../IntPhys2.tar` | `.../Benchmark/IntPhys2/Main` | 1.2 GB |
| InfLevel-lab | `/data/dataset/world/Benchmarks/InfLevel/InfLevel/` | **tar 없음 → NFS 직독** | 1.1 GB |
| IntPhysGen v11 | vll5 `/local_datasets/world/world_analysis/IntPhysGen_v11` | 이미 있음 | — |

```bash
sbatch z_research/Benchmarks/stage.sbatch        # vll3 로 (다른 노드는 --nodelist=vllN)
watch -c -n 1 bash z_research/Benchmarks/monitor.sh
```
`stage.sh` 는 여러 번 돌려도 안전하다 (`.state/<key>.done`). 노드 로컬은 다른 노드에서 안 보이므로
진행률을 **공유 디스크** `exp_results/_stage/<host>.tsv` 에 5 초마다 적고 `monitor.sh` 가 그걸 읽는다.

## 3. 각 벤치마크의 채점 규칙 (전부 원전 확인함)

공통 surprise (Garrido et al. 식 S2):
```
S_t = || p(f(V[t:t+C])) − g(V[t:t+C+M]) ||_1
AvgSurprise = mean_t S_t   (쌍 비교)      MaxSurprise = max_t S_t   (단일 영상)
```
`f` = context encoder, `g` = target encoder(V-JEPA) 또는 **정규화 픽셀**(VideoMAEv2), stride s=2.

### 3-1. IntPhys 1 dev
기존 프로토콜 `configs/protocols/intphys1_sliding.yaml` 그대로. frame_skip [2,5,10] × window [16,32] ×
context_mult [2,4,6,8,10], 시작점마다 C 최솟값, **문맥일치 2쌍만** 채점 (`pairing: matched`).

### 3-2. IntPhys 2 Main
기존 하네스 `analysis/intphys2/` + `configs/analysis/intphys2/`. 6 fps (frame_step 10), M=48,
growing context, target 에만 LN, mask token 0. 대조 기록은 `analysis/intphys2/PROTOCOL_CHECK_2026-09-14.md`.
⚠️ 그 문서의 "고칠 곳" (`dtype: float32` + `autocast: bfloat16`, 열별 최적 선택) 을 이번에 반영한다.

### 3-3. GRASP (Garrido Table S3)
frame skip **10**, window (C+M) **16**, stride **2**, context [2,4,6,8,10]×(C+M)/16.
- 라벨은 폴더 이름: `level2/{P,IP}_<시나리오>/NNN.mp4`. 16 시나리오 × 2 × 128 영상.
- **쌍은 같은 인덱스끼리** (`P_X/037.mp4` ↔ `IP_X/037.mp4`). ⚠️ GRASP 는 원래 단일 영상 평가용이라
  이 쌍짓기는 Garrido 의 해석이다 — 논문 A.4 가 "untrained network 가 spurious feature 로 높은 정확도를
  낸다" 고 스스로 단서를 단다.
- `GravitySupport` 처럼 두 property 에 걸친 시나리오는 **양쪽에 각각** 집계한다 (논문 A.4).

### 3-4. InfLevel-lab (공식 `evaluator.py` 규칙)
frame skip **10**, window **16**, stride **2** (Garrido Table S3, V-JEPA/VideoMAEv2 공통).
파일명 = `<camera>__<property>__<cover>__<object>__<trial_type>[__<dir>].mp4`

| property | 가능(real) | 불가능(magic) |
|---|---|---|
| continuity | `vv`, `ii` | `iv`, `vi` |
| solidity | `ui`, `cv` | `uv`, `ci` |
| gravity | `ui`, `cv` | `uv`, `ci` |

`c` = cut(바닥 없는 컵/뒤가 잘린 원통), `u` = uncut(정상), `i`/`v` = 물체가 끝에 안 보임/보임.
**정확도 정의가 IntPhys 와 다르다**: `(camera, cover, obj, dir)` 로 묶어 **4종이 다 있는 그룹만** 쓰고
(continuity 36 / solidity 6 / gravity 114 그룹은 불완전해서 공식 스크립트가 버린다),
그 안에서 **가능 2 × 불가능 2 = 4 비교**의 `real > magic` 비율(동률 0.5)을 낸다. chance 0.5.
p-value 는 t-test 가 아니라 **permutation test** 다. 점수는 **클수록 가능** 이어야 하므로 surprise 는 부호를 뒤집는다.
원전: `/data/hyuntak/project/2026/2027_cvpr/benchmark/inflevel/evaluator.py`.

⚠️ **gravity·solidity 는 본 영상만으로 못 푼다** — 컵이 잘렸는지는 본 실험 전의 contextualization 영상에만
나온다. Garrido 는 relabeled 판을 따로 내지만 거기선 untrained network 도 같이 올라간다. **믿을 수 있는 건 continuity 다.**

### 3-5. IntPhysGen v11
`surprise_c16t32` (문맥 16 / 미래 16). ViT-H 는 이미 **73.37%** 가 있다 (`z_research/IntPhysGenV11/`).
나머지 두 모델만 새로 돈다. **vll5 에서 bash 로** (데이터가 vll5 로컬에만 있다).

## 4. 모델별 주의

| | V-JEPA 2 ViT-H | V-JEPA 2.1 ViT-g | VideoMAEv2-g |
|---|---|---|---|
| 코드 | `src/models/` | **`app/vjepa_2_1/models/`** | `VideoMAEv2/models/modeling_pretrain.py` |
| encoder | 1280, depth 32 | 1408, depth 40 (+`norms_block` 4) | 1408, depth 40, patch **14** |
| 타깃 | `target_encoder`, LN 한 번 | `target_encoder`, **1408 씩 4 토막 각각 LN** | **정규화 픽셀** (패치 내 mean/std) |
| predictor | depth 12, mask token 10 | depth 24, mask token **8** | decoder depth **4**, embed 512 → head 1176 |
| mask_index | 0 | **0** (forward 기본값이 1 이라 명시 필요) | — |
| 해상도 | 256 | 384 공식 / **256 으로 잰다** | 224 |
| 검증 | IntPhys1 88.89% | enc·pred load 0/0 | load 0/0, 미래 예측 동작 확인 |

- 2.1 은 RoPE 라 학습된 `pos_embed` 가 없다 → 256 으로 돌려도 된다. 384 면 토큰이 9,216 개(2.25배)라
  캐시·연산이 2.5 배다. **256 으로 맞추고 그 사실을 명시한다.**
- VideoMAEv2 는 `decoder_depth=4` 를 **반드시 넘겨야 한다** (기본값이 다르면 뒷블록이 랜덤이고 조용히 돈다 —
  2026-09-20 에 실제로 당했다: missing 52 개인데 forward 는 통과했다).
- VideoMAEv2 는 patch 14 / 224 라 다른 둘과 토큰 격자가 다르다. **surprise 절대값 비교 금지**의 또 다른 이유다.

## 5. 폴더

```
README.md          이 파일 (시작점)
stage.sh           노드 로컬로 tar 를 푼다 (idempotent, 진행률을 공유 디스크에 적는다)
stage.sbatch       위를 vll3 로 제출
monitor.sh         watch -c -n 1 bash z_research/Benchmarks/monitor.sh
harness/           모델 백엔드 · 데이터셋 어댑터 · 채점 (작성 중)
exp_results/
  _stage/<host>.tsv    스테이징 진행률 (monitor 입력)
  _runs/<tag>.tsv      채점 진행률 (monitor 입력)
Archive/           날짜 박힌 분석 문서
```

## 6. 지금 상태 (2026-09-20)

- [x] 체크포인트 3종 확보 · 로드 검증 (0 missing / 0 unexpected)
- [x] 벤치마크 원전 프로토콜 확인 (논문 A.6/A.7/Table S3 + InfLevel `evaluator.py` + GRASP `level2.csv`)
- [x] vll3 데이터 스테이징 (`stage.sbatch`) · 모니터
- [ ] GRASP / InfLevel 인덱스 빌더 + `datasets.md` 등록
- [ ] 모델 백엔드 2종 (2.1-g, VideoMAEv2-g)
- [ ] IntPhys1 88.89% 재현으로 하네스 검증
- [ ] 15 칸 채우기

## 재현

```bash
sbatch z_research/Benchmarks/stage.sbatch
watch -c -n 1 bash z_research/Benchmarks/monitor.sh
```
(채점 명령은 하네스가 서면 이 절에 추가한다.)
