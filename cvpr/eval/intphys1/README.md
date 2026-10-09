# eval/intphys1 — IntPhys 1 프로토콜 (Garrido A.8 sliding) + 복사 기준선

이 폴더 = `run.sh` (표준 채점, `protocol.yaml`; IntPhys1 · GRASP · InfLevel) + `copy_baseline.sh` / `copy_baseline.py` (복사 기준선, IntPhys 1 dev × ViT-H).
수치 · 경로는 이 세션에서 산출물로 확인한 것이고, 돌려 보지 않은 것은 **미검증** 이라고 적었다 (2026-10-09; 이 코드 스페이스에서 GPU 실행은 아직 없다).

## 1. 목적

- **`run.sh`** — Garrido et al. 의 IntPhys 1 sliding-window 채점 (A.8 격자: frame skip × 창 (C+M) × 문맥 배수 C) 을 IntPhys1 dev · GRASP · InfLevel 에 건다.
  창 (C+M) 마다 **실행을 나눈다** (공식은 창마다 모델을 그 프레임 수로 짓는다; `WINDOWS=garrido_w16 garrido_w32` 가 기본). 보고값은 격자 최고 칸 = **descriptive** (CLAUDE.md §1-3).
  V-JEPA 2 ViT-H IntPhys 1 dev = **88.89** (`skip2_w32`, Filtered, property macro, 180 쌍).
- **`copy_baseline.sh`** — 같은 격자 · 같은 쌍을 **예측 없이** 채점하는 기준선. IntPhys 1 은 복사만으로 **85.0** 이 나오는 leakage 큰 벤치라 (CLAUDE.md §0 P2) 모델 점수에는 **복사 대비 Δ** 를 같이 낸다.
  dinof_* 는 로더 플래그 (`model.dinof_copy=true`) 로 `run.sh` 가 돌리고, **V-JEPA 계열은 기존 분석 스크립트 `z_research/scripts/analysis/te_intphys1_score.py` 를 `copy_baseline.py` 가 import 해 돌린다** (추출 GPU → `--score` CPU; 엔진 수정 없음).

### 1-1. 복사 기준선 — 정의 (IntPhys 1, `te_intphys1_score.py` docstring 의 식 그대로; 기호는 `z_research/Benchmarks/PROTOCOLS.md` §1)

창 격자 (`protocol.yaml` = 창 preset `garrido_a8`; 창 생성은 채점기 `evals/world_model_analysis/eval.py::_intphys1_windows` 그대로):
프레임 skip s 로 솎고 (frame_budget official = (100−1)//s 장) 시작점 t 를 stride 2 로 민 W = C + M 장 창. combo ∈ {skip2_w16 (시작 17) · skip2_w32 (시작 9) · skip5_w16 (시작 2)} (skip5_w32 · skip10_* 는 프레임 부족).
C ∈ {2,4,6,8,10} × W/16. 튜블릿 2 장 → 창 튜블릿 U = W/2, 문맥 튜블릿 Tc = C/2, 미래 j = 0 … U−Tc−1 (u = Tc + j). 영상당 창 140 개, (창, j) 행 925 개 (DRYRUN 이 `Layout` 으로 다시 셈).

```
z      = encoder(창 W 장, mask = 앞 C 장 토큰)            online `encoder`; 미래 토큰은 transformer 전에 버린다 (= 문맥 C 장만 넣은 것과 동일, max|diff| 0)
h_u    = LN(target_encoder(창 W 장 전부))[튜블릿 u]        EMA `target_encoder`, affine 없는 토큰 LN  — 표준 표적 (창 안 미래까지 양방향으로 본다)
h^c_u  = LN(target_encoder(창 앞 2(u+1) 장))[튜블릿 u]     인과 표적 (튜블릿 u 까지만 본다; auto_research H6e 정의)
p_j    = predictor(z, ctx_idx, tgt_idx, mask_index=0)[j]   미래 U−Tc 튜블릿
copy   = LN(z)[튜블릿 Tc−1]                                **문맥 마지막 튜블릿** 을 미래 전부에 복사 — predictor 를 쓰지 않는다

L1_j(x) = mean_{토큰 256 · 채널 1280} | x_j − h_{Tc+j} |            (float32; 인과 표적이면 h^c)
창 surprise S(v, combo, t, C) = mean_j L1_j      (표준 표적의 S = 공식 하네스 per_window.json 의 값과 같은 양)
Filtered: 시작점마다 min_C S  →  AvgSurprise G(v) = 시작점 평균  →  쌍 (block_id, pair_id) 정답 = 1[G(pos) < G(imp)] (공식 strict '<', 동점은 오답, n_ties 로 보고)
칸 점수 = property (O1 영속 / O2 모양 / O3 연속) macro.  최고 칸 = macro 최대 combo (descriptive)
```

- **pairing = matched** (`scoring.pairing: matched`): 4중항 안의 문맥일치 2 쌍만 (공식 `utils.py get_breaking_points / get_matches`). 문맥이 같으니 z · copy · p 가 두 영상에서 같고, 채점은 "마지막 관측 (copy) 또는 예측 (p) 이 두 미래 중 어디에 가까운가" 다.
- **copy accuracy** = copy 의 S 로 위 규칙을 그대로 돌린 칸 점수. **model accuracy** = `release` predictor (= model.pth 의 릴리즈 predictor 추출본 `z_training/runs/release_vith/latest.pt`; te `--selftest` 가 state · p 비트 동일을 확인 — 문서 값) 의 S 로. **Δ = model − copy** (pt, 같은 쌍).
- **값 (옛 산출물 `z_research/TrainingEffects/ip1score/ip1score_final_scores.json`, 이 세션에 다시 읽음)** — 표준 표적 skip2_w32: 릴리즈 **88.89** / 복사 **85.00** / Δ **+3.89** (scene bootstrap CI [−0.6, +8.9], 0 포함 — `../../z_research/TrainingEffects/ip1_copy/IP1_COPY.md`).
  복사 O1 80.0 · O2 93.3 · O3 81.7; 운동×가림 복사: static/occluded **100** · moving/visible 91.7 · static/visible 86.7 · moving/occluded 70.0.
  릴리즈가 복사를 CI 로 넘는 칸은 skip2_w16 표준 (+9.4 [+3.3, +15.6]) · skip2_w32 인과 (+6.7 [+0.6, +13.3]) · skip2_w16 인과 (+15.0) 뿐 (IP1_COPY.md 표 2a · 2b).
- **읽는 법 (leakage)** — 복사 85 는 "IntPhys 1 dev 의 85 % 는 예측 없이 마지막 관측과의 거리로 풀린다" 다. 그래서 "IntPhys 1 은 틀린 벤치" 가 아니라 **"leakage 가 크다, 복사 기준선으로 정의하고 쓴다"** (CLAUDE.md §12 금지 표현). 복사가 못 푸는 곳 (moving/occluded 70) 이 P3 의 다리다.
- **A/B 방향** — O1 · O3 쌍을 appear / disappear 로 나누면 skip2_w32 표준 복사 92.0 / 56.0, 릴리즈 96.0 / 56.0 (IP1_COPY.md 표 2c; O1 disappear 는 15 쌍 = 한 쌍 6.7 pt). `copy_summary.json` 은 이 분리를 담지 않는다 (scores json 의 `pairs` · `outcomes` 로 다시 낼 수 있다, 미이식).

## 2. 명령

```bash
GPUS=8 bash cvpr/eval/intphys1/run.sh intphys1_dev vith                     # 표준: 창 {16,32} 두 실행 → 격자 최고 skip2_w32 88.89 (옛 산출물 값)
COPY=1 GPUS=8 bash cvpr/eval/intphys1/run.sh intphys1_dev vith               # 복사 기준선 (V-JEPA 계열 → copy_baseline.sh 한 번; dinof_* → 로더 플래그, 창마다)
GPUS=8 bash cvpr/eval/intphys1/copy_baseline.sh intphys1_dev vith            # 복사 기준선 직접: release + copy, 격자 세 칸을 한 번에, Δ 까지 (~15 분 + 로드, te docstring 값)
DRYRUN=1 bash cvpr/eval/intphys1/copy_baseline.sh                            # 두 resolver 대조 · 창 격자 (R=925) · 명령만 (GPU 0 장, ~20 s)
SUBSET=valid6 GPUS=1 bash cvpr/eval/intphys1/copy_baseline.sh                # 스모크: O1/O2/O3 앞 2 block (24 영상 · 12 쌍 · 운동×가림 4 칸), 폴더 _valid6
```

## 3. 노브 표

### `run.sh` (전부 `cvpr/harness/launch.sh` 로)

| env / SET 키 | 기본값 | 의미 |
|---|---|---|
| `$1` / `BENCHES`, `$2` / `MODELS` | 5 벤치 (`intphys1_dev grasp_level2 inflevel_continuity inflevel_solidity inflevel_gravity`), `vith` | 공백 구분 행렬 |
| `WINDOWS` | V-JEPA/VideoMAE `garrido_w16 garrido_w32` · dinof `dinof_w5 dinof_w8 dinof_w16` | 창 preset 목록 (`cvpr/registry/windows.yaml`); `garrido_skip2_w32` = 88.89 칸만 |
| `GPUS`, `GPU_IDS` | `1` | launch.sh 와 같다 |
| `SET`, `LIMIT`, `TAG`, `OUTDIR`, `SUFFIX`, `DRYRUN`, `RECACHE` | — | launch.sh 와 같다 |
| `RESULTS_ROOT=legacy` | `cvpr` | 옛 이름 `z_research/Benchmarks/exp_results/intphys1_sliding__<bench>_<model>_w<N>` |
| `MB21`, `MBVM` | `8`, `8` | vjepa21g · videomae2g(창 32) 의 `surprise.intphys1.max_batch` (VRAM 상한, 수치 무관) |
| `COPY=1` | — | `dinof_*` → `SET model.dinof_copy=true` + `_copy`, 창마다 / 그 밖 → `copy_baseline.sh "$b" "$m"` 을 **첫 창에서 한 번만** (격자 세 칸을 한 번에 뽑으므로) |

### `copy_baseline.sh` → `copy_baseline.py` → `te_intphys1_score.py`

| env | 기본값 | 의미 (→ te 인자) |
|---|---|---|
| `$1` / `BENCHES`, `$2` / `MODELS` | `intphys1_dev`, `vith` | **지금은 `intphys1_dev × vith` 만** (아래 단서). 그 밖은 즉시 거부 |
| `GPUS`, `GPU_IDS` | `1` | → `CUDA_VISIBLE_DEVICES` + `--gpus N` (te 는 mp.spawn 으로 `cuda:rank`) |
| `SUBSET=valid6` | — | `--subset valid6` (24 영상) + 폴더 `_valid6` |
| `LIMIT=N` | — | `--limit N` (영상 **앞 N 개** — 쌍이 온전하지 않을 수 있다) + 폴더 `_smokeN`. `SUBSET` 과 같이 못 쓴다 |
| `DRYRUN=1` | — | 두 resolver 대조 · 창 격자 · 명령만 |
| `VALIDATE=1` | — | 채점 때 `--validate` 도 (CPU): 공식 `per_window.json` (`z_research/Benchmarks/exp_results/intphys1_sliding__intphys1_dev_vith_w{16,32}`) · H6e 배열 대조, `<out>/doc/<tag>_validation.json` |
| `OUTDIR` | `$CVPR_RESULTS/eval/intphys1/copy/<ds>_<model>[_smokeN\|_valid6]` | 결과 폴더 |
| `THREADS`, `MAX_STARTS` | te 기본 6 · 0 | rank 당 torch 스레드 · encoder 배치의 시작점 상한 (0 = 영상의 시작점 전부) |
| `WINDOWS`/`WINDOW`, `SET` | — | **적용되지 않는다** (경고만). 복사는 격자 세 칸 전체, 모델은 window 32 로 한 번 (창별로 지은 공식 실행과 비트 동일 — te docstring · `auto_research/scripts/check_window_build.py`, 여기서 재확인 안 함) |
| `TAG` | — | 지운다 (옛 resolver 가 읽는 env; `te_intphys1_score.resolve_cfg` 는 env 를 그대로 넘긴다 `:246-249`) |

`copy_baseline.py` 자체 인자: `--dataset --model --gpus --limit --subset --threads --max-starts --out --dryrun --validate --no-check` (`--help`).

## 4. 결과 위치

| 무엇 | 어디 |
|---|---|
| 표준 채점 | `$CVPR_RESULTS/eval/intphys1/<bench>_<model>_w<창>[_copy][_smokeN]/{summary.json, per_window.json, per_block.json, _resolved.yaml, _meta.json, stdout.log}` — 논문 대응 수치 `summary.json.surprise["skip2_w32/avg"].by_block_type.{O1,O2,O3}`; 격자 최고 칸은 `z_research/scripts/analysis/garrido_rescore.py` 가 창을 가로질러 고른다 |
| 복사 기준선 (V-JEPA) | `$CVPR_RESULTS/eval/intphys1/copy/<ds>_<model>[_smokeN\|_valid6]/` — **`copy_summary.json`** (표적 std · causal × 칸 skip2_w16 · skip2_w32 · skip5_w16: model · copy · Δ · O1/O2/O3 · 운동×가림 `mv` · C 별; `best` = 격자 최고 칸), `<tag>_scores.json` 과 `doc/<tag>_scores.json` (te `--score` 원본: `cells` · `outcomes` · `pairs`), `manifest.json` (predictor 지문 · 창 배치 · config), `shards/v*.npz` (영상마다 p_std/p_causal (1, 925) · copy_std/copy_causal (925,)), `arrays_v<V>.npz` (병합), `timing_r*.json`, `_meta.json`, `doc/` (VALIDATE) |
| 복사 기준선 (dinof) | `$CVPR_RESULTS/eval/intphys1/<bench>_dinof_*_w<창>_copy/` (표준 형식, 창마다) |

재개: 같은 명령을 다시 치면 `shards/` 에 있는 영상은 건너뛴다 (predictor · 창 배치가 `manifest.json` 과 다르면 죽는다 → `OUTDIR` 을 바꿀 것). 래퍼는 shard 수가 기대와 다르면 채점 전에 멈춘다.

## 5. 옛 명령 대응표

| 옛 | 새 |
|---|---|
| `bash z_research/Benchmarks/run_all.sh` (SET 으로 A.8 격자 주입, 창별) | `GPUS=8 bash cvpr/eval/intphys1/run.sh` (`RESULTS_ROOT=legacy` 면 옛 폴더 이름 그대로) |
| `SET="surprise.intphys1.frame_skips=[2] surprise.intphys1.window_sizes=[32] model.window_size=32" bash z_research/scripts/run.sh intphys1_sliding intphys1_dev vith` | `WINDOWS=garrido_skip2_w32 bash cvpr/eval/intphys1/run.sh intphys1_dev vith` |
| `COPY=1 ... dinof_highres` (`chain_1002_dinof_intphys1.sh`, w5/w8/w16) | `COPY=1 bash cvpr/eval/intphys1/run.sh intphys1_dev dinof_highres` (창 `dinof_w5 dinof_w8 dinof_w16` 자동) |
| `CUDA_VISIBLE_DEVICES=0..7 python z_research/scripts/analysis/te_intphys1_score.py --preds release --gpus 8 && python ... --preds release --score` → `/data2/.../training_effects/ip1score_custom/` + `z_research/TrainingEffects/ip1score/ip1score_custom_scores.json` | `GPUS=8 bash cvpr/eval/intphys1/copy_baseline.sh intphys1_dev vith` → `$CVPR_RESULTS/eval/intphys1/copy/intphys1_dev_vith/` (+ `copy_summary.json`; scores json 도 그 안 `doc/`) |
| `... te_intphys1_score.py --preset final --subset valid6` | `SUBSET=valid6 ...` (predictor 는 release 하나) |
| `... te_intphys1_score.py --preset final --validate` → `z_research/TrainingEffects/ip1score/*_validation.json` | `VALIDATE=1 ...` → `<out>/doc/<tag>_validation.json` |
| `python z_research/scripts/analysis/te_analyze_ip1.py` (CI · 튜블릿별 · 학습 곡선 → `IP1_COPY.md`) | 대응 없음 — `ip1score_final/arrays_v360.npz` 경로가 박혀 있어 그대로 쓴다 (미이식) |

옛 산출물 (predictor 여섯 + copy, 360 영상): `/data2/local_datasets/world/world_analysis/cache/training_effects/ip1score_{final,curves}/` — 거기 `copy_std / copy_causal` 이 여기 `copy_*` 와 같은 정의다.

## 6. 단서 / 미완

- **이 코드 스페이스에서 GPU 로 돌린 적이 없다.** 검증 = `bash -n` · `py_compile` · `--help` · `DRYRUN=1` (두 resolver 의 data · model · surprise · scoring 일치, `Layout` R=925 · combo 3 개, predictor 실물, `COPY=1` 이 창 2 개에서 한 번만 호출됨) · `cvpr/harness/check_equivalence.py` 27 행 동일 (이 세션 재실행). 끝까지 도는지 · 옛 `ip1score_final_scores.json` 과 같은 수치가 나오는지는 **미검증** (첫 실행은 `SUBSET=valid6 GPUS=1` 로; 옛 valid6 산출 `z_research/TrainingEffects/ip1score/ip1score_final_valid6_scores.json` 과 copy 칸을 대조할 것).
- **`intphys1_dev × vith` 만 된다.** `te_intphys1_score.py:169` `S_TOK, D = 256, 1280` (ViT-H 리터럴; spawn 된 rank 가 모듈에서 다시 읽으므로 부모에서 덮어도 소용없다) 와 `:248-249` 의 `"intphys1_sliding", "intphys1_dev", "vith"` 리터럴 때문. vitl · vith_pft_* · vith_ariel_* · GRASP · InfLevel 은 아래 수정 전까지 못 한다 (vith_pft_* 등은 옛 스크립트에 `--preds ip1_e40` 식으로는 됐다 — 복사 기준선은 predictor 와 무관하게 같은 값이므로, **vith 의 copy 칸이 곧 모든 그룹 A predictor 의 copy 칸**이다. 그룹 B (다른 encoder) 는 안 된다).
- 래퍼는 te 의 `CACHE_ROOT` (`:163`, `/data2/...` 리터럴) · `DOC_DIR` (`:164`, `z_research/TrainingEffects/ip1score`) 를 **부모 프로세스의 모듈 변수**로 결과 폴더에 덮는다 (파일 수정 없음; rank 는 `job["out"]` 을 받는다 `:504-553`). `--score` 가 `DOC_DIR` 에 쓴 뒤 `out/` 로 복사하므로 scores json 이 `doc/` 와 `out/` 두 벌이다.
- **predictor 가 하나 (`release`) 필요하다.** te 는 predictor 0 개를 못 받는다 (`parse_preds` `:209-223`) — release 는 model.pth predictor 와 비트 동일 (te 검증 (0), 문서 값) 이라 모델 점수 88.89 가 같이 나온다 (옛 산출물 값; 재실행은 fp16 배치 구성 탓에 창 값 중앙값 ~1e-4 가 흔들려 동점 근처 쌍 ±1 개 다를 수 있다 — IP1_COPY.md 표 1).
- **encoder 교차.** copy = LN(online z), 표적 = EMA target. "깨끗한 복사" (target encoder 를 문맥에만) 는 없다 (IP1_COPY.md §10a). 인과 표적 copy 는 `copy_causal` 로 같이 나온다.
- **CI 는 내지 않는다** (점 추정 pt 만). scene 층화 bootstrap 은 `te_analyze_ip1.py` (미이식). 180 쌍의 95 % CI 는 ±5 ~ 7 pt — 칸 하나의 차는 한두 쌍이다.
- skip5_w16 의 `O2_24_*` 두 쌍은 위반이 창 밖이라 **구조적 동점** (strict 오답; IP1_COPY.md 표 2a 주석) — p · copy 가 같이 동점이라 Δ 는 안 변한다.
- `--limit N` 은 영상 앞 N 개라 쌍이 깨질 수 있다 — 스모크는 `SUBSET=valid6`.
- `summary.json` (하네스 형식) 은 만들지 않는다 — 표준 채점 폴더와 파일 이름이 다르다 (`copy_summary.json`, `<tag>_scores.json`).
- 관련: IntPhys 2 엔진에는 이미 복사 스위치가 있다 (`analysis/intphys2/surprise.py:229 _copy_baseline`, env `IP2_COPY=1` `:368 · :573`).

## 7. 엔진 수정 제안 (하지 않았다 — 제안만)

1. **`z_research/scripts/analysis/te_intphys1_score.py` 를 데이터셋 · 모델 인자화** — `resolve_cfg()` (`:242-256`) 에 `dataset="intphys1_dev", model="vith"` 인자 + CLI `--dataset/--model` (`:1028-1043`); `S_TOK, D` (`:169`) 를 `process_clip` (`:399`) 안에서 `bundle.num_spatial_tokens` · `bundle.embed_dim` 으로 (`worker` `:519` 의 bundle 에서); `pred_path("release")` (`:194-196`) 를 모델별 (`z_training/runs/release_<model>/latest.pt` 또는 model.pth predictor 를 `bundle.predictor` 로 쓰는 `own` 태그 — te_v11_score 의 `own` 과 같은 규약); `ref_dirs` (`:227-238`) 를 `intphys1_sliding__<dataset>_<model>_w{16,32}` 로; `CACHE_ROOT` `:163` · `DOC_DIR` `:164` 를 `--out` / `--doc-dir` 로. 그러면 vitl · 그룹 B · GRASP · InfLevel (`sliding_frame_skips` 가 다르니 `Layout` 은 그대로 cfg 에서 읽는다) 의 복사 기준선이 같은 래퍼로 된다.
2. **(장기) wma 엔진 복사 스위치** — `evals/world_model_analysis/eval.py:799` (intphys1 sliding 분기) 의 `bundle.predictor(z, ci[:n], ti[:n], mask_index=…)` 자리에서 `analysis/intphys2/surprise.py:229 _copy_baseline` 과 같은 함수 (문맥 마지막 튜블릿 → affine 없는 LN → 미래 튜블릿 수만큼 복제) 를 `model.copy_baseline: true` (또는 `WMA_COPY=1`) 일 때 쓴다 (`:571` single 분기도 같이).
   그러면 `COPY=1` 이 창별 실행 · 모든 모델 · GRASP · InfLevel 에서 `launch.sh` 를 그대로 타고 `_copy` 이름 · `per_window.json` · `garrido_rescore.py` 가 공짜다. 넣으면 te 의 `copy_std` 와 valid6 24 영상 창 값 비트 대조 (≤ 1e-6) 를 붙일 것. 인과 표적 (`copy_causal`) 은 이 경로에 없다 — 그건 te 에 남는다.
