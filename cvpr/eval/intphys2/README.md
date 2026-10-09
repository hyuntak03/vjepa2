# eval/intphys2 — IntPhys 2 (논문 D.3 격자)

**목적.** IntPhys 2 Main 506 쌍 쌍 비교 surprise, 창 {16, 32, 48} @ 6 fps, 문맥 C = 창 × {1/4 … 7/8}. 엔진 `analysis/intphys2/eval.py` (**torchrun**, 다른 하네스).
확립된 수치 ViT-H **54.35** (C 24) vs 복사 기준선 55.53 — 같은 C 에선 구분 안 됨 (CLAUDE.md §5-6). 옛 `analysis/intphys2/configs/bench_intphys2_main_<model>.yaml` + `run_grid.sh` 의 `--set` 과 config 가 같다 (check_equivalence 4 행, tag/folder/model.window_size 제외).

## 명령
```bash
GPUS=8 bash cvpr/eval/intphys2/run.sh                                   # intphys2_main × vith × {ip2_w16, ip2_w32, ip2_w48}
GPUS=8 bash cvpr/eval/intphys2/run.sh intphys2_main "vith vjepa21g videomae2g"   # videomae2g 는 {ip2_vmae_fps6, fps3, fps2}
GPUS=8 WINDOWS=ip2_w48 bash cvpr/eval/intphys2/run.sh intphys2_main vith_pft_intphys2_e80   # 학습한 predictor
LIMIT=4 GPUS=1 WINDOWS=ip2_w16 bash cvpr/eval/intphys2/run.sh           # 배관 점검 (→ …_wip2_w16_smoke4)
DRYRUN=1 bash cvpr/eval/intphys2/run.sh
```

## 노브
| env / SET 키 | 기본 | 의미 |
|---|---|---|
| `$1` `$2` / `DATASETS` `MODELS` | `intphys2_main` / `vith` | `intphys2_main_nfs` 는 다른 사본 (`$CVPR_LOCAL/world/IntPhys2`) |
| `WINDOWS` | `ip2_w16 ip2_w32 ip2_w48` (videomae2g: `ip2_vmae_fps6 ip2_vmae_fps3 ip2_vmae_fps2`) | 창 preset 목록. preset 이 `surprise.window_size` · `context_length_sweep` · `context_length` (= 창/2) · `stride 2` · `data.target_fps` · `frame_step 10` 을 채운다 |
| `SET="surprise.max_window_batch=N"` | run.sh 가 모델×창으로 넣는다 (16/12/8, vjepa21g 12/8/6) | VRAM 상한, 수치 무관 |
| `SET="surprise.video_batch=N"` | `1` (protocol.yaml) | **보고 수치는 1** — >1 이면 fp16 reduction 순서가 바뀌어 ±0.5 pt 흔들린다 (옛 `VIDEO_BATCH`) |
| `GPUS` / `GPU_IDS` | `1` / `0..` | torchrun `--nproc-per-node` + `CUDA_VISIBLE_DEVICES` (이 엔진만 바깥 CUDA_VISIBLE_DEVICES 규약) |
| `LIMIT=N` | — | `evaluation.limit_videos=N` + 이름 `_smokeN`. `SMOKE=1` 은 limit 8 인데 **접미사가 없어 본 폴더에 쓴다** — 쓰지 말 것 |
| `RESULTS_ROOT=legacy` | `cvpr` | 옛 `z_research/Benchmarks/exp_results/intphys2/bench_intphys2_main_<model>_w<N>` — ⚠️ 아래 단서 |

## 결과 위치
`$CVPR_RESULTS/eval/intphys2/<ds>_<model>[_w<창>][_smokeN]/` — 엔진이 `folder/tag` 로 쓴다 (`_resolved.yaml` `_meta.json` `stdout.log` 는 launch.sh 가). 기본 창 `ip2_w48` 이면 접미사 없음 (`intphys2_main_vith`), 나머지는 `_wip2_w16` 등.
엔진 산출물의 이름은 아래 단서 (smoke 폴더 실물) 와 `analysis/intphys2/README.md`.

## 옛 명령 대응
| 옛 | 새 |
|---|---|
| `MODELS=vith WINDOWS="16 32 48" bash analysis/intphys2/run_grid.sh` | `WINDOWS="ip2_w16 ip2_w32 ip2_w48" bash cvpr/eval/intphys2/run.sh intphys2_main vith` |
| `VIDEO_BATCH=2 bash analysis/intphys2/run_grid.sh` | `SET="surprise.video_batch=2" …` |
| `torchrun … -m analysis.intphys2.eval --config bench_intphys2_main_vith.yaml --set surprise.window_size=48 …` | `WINDOWS=ip2_w48 … run.sh intphys2_main vith` (launch.sh 가 torchrun 을 띄운다) |

## 단서 · 미완
- `RESULTS_ROOT=legacy` 이름: `ip2_w16` → `bench_intphys2_main_vith_w16` (옛 run_grid.sh 와 같다). 2026-10-09 검증 전에는 `n=${w//[!0-9]/}` 가 `ip2` 의 2 를 긁어 `_w216` 이었고, run.sh 한 줄 (`n=${w##*_w}` …) 로 고쳤다 (DRYRUN 확인). `ip2_vmae_fps*` 는 옛 run_grid.sh 가 videomae 를 `w16` 로만 돌려 대응 이름이 없다 → `…_wvmae_fps6` 처럼 간다.
- 엔진 산출물 (vll3 smoke 폴더 `intphys2_main_vith_wip2_w16_smoke4/` 실물): `summary.json` · `summary.txt` · `per_video.csv` · `per_window.jsonl` · `plots/` · `config.resolved.yaml` (+ launch.sh 의 `_resolved.yaml` `_meta.json` `stdout.log`).
- `dtype float32 + autocast bfloat16` 은 옛 config 그대로. `analysis/intphys2/PROTOCOL_CHECK_2026-09-14.md` 의 수정안 (열별 선택) 은 미반영.
- 옛 run_grid.sh 의 "이미 있음 건너뜀" 은 없다.
- 데이터는 노드 로컬 `$CVPR_DATA2/local_datasets/world/Benchmark/IntPhys2/Main` (stage.sh 가 푼 노드만).
