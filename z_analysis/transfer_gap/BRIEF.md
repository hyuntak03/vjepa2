# 이식 실패 · PFT 비전이 조사 — 팀 브리프 (2026-10-30, 리더 [L])

> 이 폴더는 에이전트 팀 (검토 [R] · 가설 [H] · 실험 A [X1] · 실험 B [X2]) 의 공유 작업장이다. 리더 [L] = 메인 세션.
> 진행판은 `BOARD.md` (append-only). 이 브리프는 [L] 만 고친다.

## 0. 사용자 질문 (2026-10-30, 원문 뜻)

사용자가 말한 현재 문제:
- VOBench **late** (문맥 끝 가림) 에서 encoder 는 물체가 가려져도 모양 · 색 · 속도가 읽힌다.
- predictor 가 만든 미래에서도 모양 · 색이 읽힌다.
- 그런데 **encoder 표현으로 학습한 probe head 가 predictor 표현에서는 안 읽힌다** (이식 실패).
- 그리고 릴리즈 predictor 를 post-FT (PFT) 해도 **다른 벤치마크로 성능이 전이되지 않는다**.

물을 것:
- **Q1** — 정보는 있는데 이식이 안 되는 것은 표현 벡터의 **방향**이 다른가, **크기**가 다른가, 아니면 **표현 자체가 다르게 생겼나** (비선형 · 다른 자리 · 다른 시점)?
- **Q2 (개입)** — 이식이 잘 되도록 만들면 object permanence 가 해결되나?
- **Q3** — PFT 가 다른 벤치마크로 안 옮겨지는 것은 학습 데이터의 특정 **진행 방향** (운동 방향 · 속도 · 가림 시점 · 자리) 이나 **분포를 외워서**인가?

## 1. 이미 있는 근거 (출처 필수 — [R] 이 산출물에서 다시 확인한다)

⚠️ 사용자의 전제 중 일부는 기존 기록과 어긋난다. **[R] 이 1 라운드에서 모델 · probe 마다 사실을 고정한다.**

| 무엇 | 수치 | 출처 |
|---|---|---|
| 릴리즈, h (target encoder, 가림 없는 튜블릿) 로 학습한 정체 head → h 보임 | 89–97 % | `z_research/VOBench_v1/README.md` §3-10 · `vjepa2/probing/results/` |
| 〃 → h 가림 튜블릿 | 9–21 % (마지막 층) · 16 층은 0–1 % | §3-10 · `vjepa2/probing/README.md` 중간층 절 |
| 〃 → p 보임 (문맥 끝 보임 조건의 미래) | t8 83–88 % · t15 13–56 % (오답이 none 으로) | §3-10 |
| p 로 학습한 head → p 보임 t15 | 52 % (primitive) · 74 % (realistic), none 0 % → "먼 미래 감쇠는 정보 소실이 아니라 형식 (정렬) 문제" | §3-10 (10-10 철회 문단) · `results/p_only_*` |
| **릴리즈 late 의 p, p 로 학습한 head 로도** | 움직이는 물체 0–8 % (static realistic 19 %) → **late 는 정보 부재** 로 기록됨 | §3-10 |
| 시점 일반화 | p 는 같은 시점이면 h 만큼 읽힘 (shape 95–98), 앞으로 4 칸 이상 46–55 vs 뒤로 75–78 | `vjepa2/probing/temporal_generalization/RESULTS.md` |
| vast AR, late 정체 | 블록 평균 63–76 % (우연 17) vs 참 자리 34–62 % (early 85–95) → "정체는 퍼져 있고 자리에 안 놓인다" | `z_archive_10/13_SINCE_2026-10-10.md` §6 B5 |
| context encoder (online) 가 **완전히 가려진** 문맥 튜블릿에서 | 모양 62–66 % (chance 17) · 색 25–27 % (chance 12.5), late k2 · k3 튜블릿 7 | `z_analysis/occluded_identity_probe/result.json` (스크립트 `z_research/scripts/analysis/vobench_occluded_identity_probe.py`) |
| vast AR, 평균 풀링 속도 probe | 걸음 따라 참 방향으로 바뀌나 폭 0.6–0.7 배 (late 0.3–0.6) | `z_analysis/ar_layer_velocity_probe/README.md` |
| vast AR, h 로 배운 위치 자 → p | 물체가 1–2 칸 나아가고 멈춘 뒤 흐려짐 (최대 질량 0.21 → 0.06–0.09) | `z_analysis/ar_token_transfer/README.md` |
| anchor 이식 | 미래 자리 · 속도는 마지막 문맥 블록의 보이는 내용에서 온다 (따름 0.63–0.88) | `auto_research/paper/DESIGN_CHOICES_2026-10-13.md` |
| PFT (VOBench split v2 + gravity train, 30 epoch) | VOBench late 62 → 89.8 · gravity 0 → 96 (분포 안 held-out) / IntPhys 1 88.89 → 83.33 · IntPhys 2 54.3 → 53.6 · GRASP 70.4 → 68.8 · InfLevel 58.3 → 58.3 | `z_research/VOBench_v1/Archive/EVAL_MATRIX_2026-10-15.md` |
| 옛 PFT 셋 (v11 · IntPhys 1 · …) | 모두 학습 도메인 안에서만 오름 (v11 held-out 75.83 → 91.35 인데 IntPhys 1 88.89 → 77.22 / IntPhys 1 학습 93.89 인데 v11 78.53) | `z_training/RESULTS_2026-09-19.md` (또는 `z_training/docs/`) |
| PFT 뒤 late 정체 (자) | 0–5 → 87–93 %, attention 이 과거 물체 자리를 찾아감 (릴리즈 late k4 0.15 → 2.12, 5–11 층) | `13_SINCE` §7 (a) · §6 B13 |

## 2. 쓸 수 있는 자원

- **릴리즈 h / p 토큰 캐시** (VOBench 가능 영상 1,440 clip 전부): `/local_datasets/world/world_analysis/cache/vobench_v1_vith_tubeprobe/{h,p}.npy`
  h (1440, 16, 256, 1280) fp16 = LN(target_encoder(32 장)) · p (1440, 8, 256, 1280) fp16 = 릴리즈 predictor 출력 (t8–t15) · `meta.json` 에 clip 별 meta (objset · violation · occlusion · k · motion · obj_a · obj_b · samples 문자열).
  16 층 판 `…_tubeprobe_L16/`. 만든 스크립트 `z_research/scripts/analysis/vobench_tubelet_probe.py` (probe 학습 코드도 여기).
- 체크포인트 · 레지스트리 모델: `vith` (릴리즈) · `vith_release_vobftg` (릴리즈 + PFT) · `vith_vast_ar_state_ep70` · `vith_vast_ar_state_ep70_vobftg`. PFT run 폴더 `z_training/runs/vobench_release_postft_g/` (e5…e30, latest) · `z_training/runs/{v11_postft,intphys1_postft,intphys2_postft*}`. 레지스트리 `cvpr/registry/models.yaml` · `configs/protocols/models.md`.
- 데이터: VOBench 프레임 `/data2/local_datasets/world/Benchmark/VOBench_v1` (index `data_csv/vobench_v1/index.csv`, split v2 `data_csv/vobench_v1_split_v2/`, 창 민 판 `vobench_v1_asymmetry`), v11 · IntPhys 1 · 2 · GRASP · InfLevel 은 `cvpr/registry/datasets.yaml`.
- 평가 진입점: `cvpr/eval/testbed/run.sh <데이터셋> <모델>` (= `surprise_c16t32`), `z_training/eval.sh`, 복사 기준선 `cvpr/eval/testbed/copy_baseline.sh`. 자세한 것은 `cvpr/README.md`.
- vast 브랜치: `git log main..origin/vast` (33 커밋, 10-01 ~ 10-27). 문서는 `git show origin/vast:<경로>` 로 읽는다 (35 개는 10-24 에 main 으로 가져옴).

## 3. 실행 규칙 (모든 에이전트)

- **이 셸에는 GPU 가 없고 메모리 10 GB · CPU 6 개다** (vll5 job 231080). 큰 배열 (캐시 h 는 15 GB) 을 이 셸에서 올리지 않는다.
  무거운 계산은 전부 **sbatch** 로: `sbatch -p batch_vll -w vll5 --gres=gpu:N --cpus-per-gpu=12 --mem-per-gpu=45G --time=0-06:00:00 -o z_analysis/transfer_gap/logs/%x_%j.out --wrap "..."`
  (평가 하네스는 `GPUS=N bash cvpr/harness/submit.sh cvpr/eval/testbed/run.sh <데이터셋> <모델>`).
- **GPU 몫: [X1] 동시에 최대 4 장, [X2] 동시에 최대 4 장** (vll5 8 장). vll7 job 224990 (사용자 학습) 과 남의 job 은 건드리지 않는다. `scancel` 은 자기 job 만.
- job 기다리기: `until [ -f <완료표시> ] || ! squeue -j <id> -h | grep -q .; do sleep 60; done` 를 Bash timeout 600000 으로 반복. 한 job 에 총 3 시간 넘게 기다리지 않는다 — 넘으면 BOARD 에 적고 다음으로.
- 디스크: `/` 는 64 GB 남음 — 큰 파일 금지. 큰 특징 · 변환 프레임은 `/data2/local_datasets/world/world_analysis/cache/transfer_gap/` 아래.
- python `/data/hyuntak/anaconda3/envs/vjepa2/bin/python`. 스크립트는 `z_research/scripts/analysis/transfer_gap_*.py` (추적되는 자리). 결과 json · 그림 · 문서는 이 폴더 `z_analysis/transfer_gap/` 아래.
- **predictor 새 학습 (PFT 재실행 포함) 은 하지 않는다** — 사용자 몫. probe · 정렬 사상 (p → h) · 작은 판독기 학습은 해도 된다. PFT 를 다시 돌려야 답이 나오는 실험은 설계만 적고 [L] 에 올린다.
- 커밋 · push 금지. `sed -i` 금지.
- CLAUDE.md 규칙 그대로: probing 은 가능 변이만 (§1-4) · block 단위 split (§1-5) · matched pairing · A/B 방향 병기 · 복사 기준선 Δ · 학습된 자는 held-out · encoder 대조 · 파라미터 없는 교차검증 (§12-8) · 전체 토큰 기준 채점이 주.
- 문서는 사용자 문서 규칙 (CLAUDE.md 맨 끝): 한국어 · 맨 위 3 줄 요약 · 용어표 · 왜 → 어떻게 → 결과 → 해석 · 주장마다 [측정] / [이론] / [추측] · 코드 · 경로는 맨 끝 "부록: 재현" 에만.
- 금지 표현 (CLAUDE.md §12) 을 지킨다. 특히 memory · state 프레임, "중력을 안다", "목적함수 때문이다" (discussion 한 문단만), α 분해 인용.

## 4. 진행판 규칙 (`BOARD.md`)

맨 아래에 `### [역할] R<라운드> — 제목` 으로 덧붙인다. 남의 글을 고치지 않는다. 반박은 `### [역할] → [상대] 반박: …`.
결과 하나마다: 왜 → 어떻게 → 결과 ([측정]/[이론]/[추측]) → 해석 → 산출물 경로.

---

## 5. 4 라운드 (2026-10-30 오후, [L] 추가) — 측정 + 진단 학습

사용자 (10-30, 원문): "측정해봐. 어떻게 학습을 해야 전이가 잘 될 지 그런걸. vobench 학습하면 전이 안되잖아."
[L] 이 앞 답에서 "late 에서 문맥 앞쪽에 보였던 물체의 국소 꼴이 다시 나타날 자리에 놓이지 않는다. 왜 옮기지 않는지는 안 쟀다" 고 보고한 뒤의 요청이다.

**두 갈래:**
- **(A) 측정 [X1]** — 왜 late 에서 물체 꼴이 다시 나타날 자리로 옮겨지지 않나, PFT 가 옮기게 만든 기전이 학습 생성기 밖에서도 켜지나. 기존 근거: `z_archive_10/13_SINCE_2026-10-10.md` §6 B13 (물체 query → 과거 물체 자리 attention: 릴리즈 late 0.3 % = 거울 대조 0.3–0.4 / PFT 뒤 0.15 → 2.12 %, 5–11 층 · vast 0.39 → 3.82, 3–4 층. ⚠️ attention ≠ 정보 사용, 분포 안만). 코드 `z_research/VOBench_v1/analysis/attention_map/` · `auto_research/scripts/chain_1013_attention.sh`.
- **(B) 진단 학습 [X2]** — 무엇을 바꿔 학습하면 학습 생성기 밖에서도 late 영속 이득이 남나. **이번 라운드에 한해 진단 PFT 학습을 허용한다 (사용자 10-30 요청).** 릴리즈 predictor 에서 시작하는 PFT 만, 기존 레시피 `z_training/runs/vobench_release_postft_g/config.yaml` 에서 **한 축씩** 바꾼다. 최대 4 run. scratch predictor · encoder 학습 · AR 학습은 하지 않는다. 설계 후보는 `Q3_PFT_TRANSFER_R2.md` §16 표.

**전이 시험판** — 효과가 사는 창 (문맥 끝 가림 + 이동 + 영속) 을 담은 판. 모두 학습에 안 쓴다:

| 판 | 성격 | 주의 |
|---|---|---|
| VOBench split v2 test late | 분포 안 기준 | — |
| v11 split test 가림 조건 (사라짐 A / B) | 표본 큼 | VOBench 학습셋의 53 % 가 v11 배경 · 메시 4/15 겹침 → 반쯤만 held-out. vll5 로컬에만 있음 |
| v11_realistic 사라짐 | 실물 메시 | vll5 로컬에만 있음 |
| IntPhys 1 late uni (이동 · 가림 60 쌍) | 진짜 다른 생성기 | 표본 작음 (±12 pt). 블록 공간 복사 71.7 · 릴리즈 56.7 (paper_lab/EVIDENCE.md #8). NFS 사본 있음 |
| IntPhys 2 late uni (permanence 120 쌍) | 진짜 다른 생성기 | 문맥이 픽셀로 같지 않음. NFS 사본 있음 |
| 표준 외부 (IntPhys 1 Garrido · GRASP · InfLevel · IntPhys 2) | "안 망가지나" 점검 | 결론 축 아님 |

모든 판에서 복사 기준선 · A/B 방향 · block (장면) bootstrap CI. 사전 등록 규칙 (R3 와 같음) 을 지킨다.

**GPU 사정 (14:50 기준)** — vll5 8 장이 남의 job (taesoo27 vlm-eval, 최대 20:57 까지) 으로 막혀 있다. vll3 은 RTX 3090 24 GB 4 장이 비어 있고, vll4 는 2 장이 비어 있다. vll1 은 쓸 수 없다 (GPU 안 보임 · 디스크 꽉 참).
- **VOBench 256² 사본**: job 235557 이 만든다 → `z_pixel_decoder/runs/_frames/VOBench_v1_256full/` (같은 상대 경로 · 같은 파일명, 끝나면 `_DONE`). 학습 로더 (ClipTransform out_uint8) 와 비트 단위로 같음을 확인했다. 채점 하네스 (float 경로) 와는 uint8 반올림 ≤ 0.5/255 차이 → **같은 사본으로 잰 모델끼리만 비교한다.** 잘라 옮기기 증강은 사본에서 업샘플이 된다 (원본 512 → 448 자르기와 다름) — 쓰면 그렇게 적는다.
- IntPhys 1 late uni · IntPhys 2 late uni 의 NFS 사본: `z_pixel_decoder/runs/_frames/{IntPhys1_uni,IntPhys2_late_uni}`. 다른 노드에서 `SET="data.frames_root=<사본>"`.
- v11 · v11_realistic 은 vll5 로컬에만 있다 → vll5 가 풀린 뒤. 그 전까지는 다른 판부터.
- **GPU 몫 R4: [X2] vll3 4 장 (학습), [X1] vll4 2 장 (측정).** vll5 가 풀리면 [X1] 4 · [X2] 4. 남의 job 은 건드리지 않는다. 자기 job 만 scancel.
- 기다리기 규칙은 §3 과 같다 (job 당 최대 3 시간, 넘으면 BOARD 에 적고 다음으로).
