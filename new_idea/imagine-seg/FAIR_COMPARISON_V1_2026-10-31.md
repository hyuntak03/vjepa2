# phys3d_perm_v1 공정 비교 — 대조군 · 보조 head · in-path (2026-10-31)

> **질문** — 같은 데이터 · 같은 조건에서, 점유 추측을 predictor 경로 안에 넣으면 (in-path) 옆 head 로 감독할 때보다 영속이 더 잘 배워지고 다른 벤치로 넘어가는가?
> **답** — 아니다. 보조 head · in-path 모두 같은 데이터의 대조군과 같은 쌍 비교에서 구별되지 않고 (네 세트 p 0.31–1.00), 세 판 모두 IntPhys 2 · 실제 영상에서 릴리즈보다 낫지 않다 [측정, §6].
> **다음 할 일** — 감독 구조는 접고 데이터 쪽 (실제 영상 + 추적한 내용 위 가림막, `TRACKING_SUPERVISION_PLAN_2026-10-31.md`) 으로.

원 계획: `PLAN_2026-10-31.md` §3–§5. 사용자 결정 (2026-10-31): "phys3d_perm_v1 으로 일단 진행, 얼마나 걸릴지 먼저 측정".

## 용어

| 용어 | 뜻 |
|---|---|
| 점유 | 미래 블록의 칸 (16×16 px × 연속 2 장) 마다, 문맥에서 본 물체가 그 칸에 있는가 (보이든 가려졌든). 1 / 0, 모르면 무시 |
| 점유 head | predictor 6 번째 블록 뒤 은닉 (384 차원) 의 미래 토큰마다 점유 logit 하나를 내는 선형 층 |
| e_mask | in-path 에서 은닉에 더하는 학습 벡터 (384 차원). 0 에서 시작 |
| scheduled sampling | in-path 학습 때 표본마다 일정 확률로 점유 추측 대신 정답 점유를 넣는 것. 확률은 1 에서 시작해 전체 step 의 75 % 지점에서 0 이 된다 |

## 1. 왜 다시 하나

- 오늘 앞선 head 실험 (`z_training/docs/AUX_PERMANENCE_2026-10-31.md`) 은 phys3d 에서 보조 모듈과 섞여 있었다. predictor 를 고정했거나, 다른 판에서 이어 받았다. "head 만" 판은 VOBench 로만 했는데, 그 라벨은 가려진 칸이 0.11 % 였다 [측정].
- 그래서 "head 대 in-path" 를 공정하게 비교한 결과가 없다.

## 2. 설계

세 판은 아래를 모두 같게 하고, 한 가지만 다르다.

| 같게 맞춘 것 | 값 |
|---|---|
| 시작점 | 릴리즈 predictor (ViT-H, 12 층) |
| 데이터 | phys3d_perm_v1 학습 22,673 편, 4 fps 32 장, 문맥 {10, 12, 14, 16} 장 중 하나를 step 마다 |
| 라벨 | 점유 = 보이는 물체 ∪ 가려진 물체 (렌더러의 물체 상자 칸). 미래 칸 중 보임 4.7 % · 가려짐 0.86 % |
| head 자리 | 6 번째 블록 뒤 (12 층의 절반, 계획 §3.2) |
| 점유 손실 | BCE + dice (계획 §4.2), 가중 1.0 |
| 학습률 | predictor 최대 5e-5, head · e_mask 3e-4. warmup 1 epoch → 상수 → 마지막 10 epoch 선형 감소 |
| 길이 · 배치 | 20 epoch × 354 step, 전역 배치 64, 좌우 반전 0.5 (라벨과 함께) |
| 학습 중 검증 | VOBench held-out late 216 쌍 (학습과 겹침 0), 2 epoch 마다 |

| 판 | 다른 한 가지 | 추론 구조 |
|---|---|---|
| ① 대조군 | 점유 손실 가중 0 (head 는 계산되지만 기울기 0) | 릴리즈와 같다 |
| ② 보조 head | 점유 손실 가중 1 | 릴리즈와 같다 (head 는 학습 때만) |
| ③ in-path | ② + 점유 추측 ŝ 를 은닉에 ŝ · e_mask 로 더해 뒤 6 블록이 쓴다 | head · e_mask 가 남는다 |

⚠️ 계획과 다른 점: 라벨이 물체 모양 (amodal mask) 이 아니라 물체 **상자** 칸이다. phys3d_perm_v1 에 이미 있는 라벨을 그대로 썼다.

## 3. 코드 변경

- `app/vjepa_frozen/aux_memory.py` — `sup.binary` (점유 1 출력 head), `inpath` (ŝ · e_mask 를 경로에, 학습 때 정답 점유를 받는 인자). 기존 config 의 동작은 그대로다 (새 키가 있을 때만).
- `app/vjepa_frozen/train.py` — 라벨을 forward 앞에서 만든다 (in-path 가 정답을 받으려고). binary 일 때 BCE + dice. scheduled sampling 확률 계산과 기록.
- 채점기 (`analysis/intphys2/model.py`) 는 checkpoint 에 저장된 설정으로 같은 구조를 다시 짓는다. 고칠 필요가 없다. 채점 때는 정답이 없으니 늘 자기 추측을 쓴다.
- config: `configs/training/imagine_seg_v1_{base,ctrl,head,inpath}.yaml`. 병합 결과를 비교하면 판 사이에 다른 줄은 손실 가중과 in-path 설정뿐이다 [측정].

작은 모델 (CPU, 폭 48 · 4 층) 점검 [측정]:
- e_mask = 0 이면 출력이 릴리즈와 비트 단위로 같다
- 정답 점유를 넣으면 출력이 바뀌고, 점유를 다른 자리로 옮기면 또 바뀐다 (계획 §7 개입 시험이 가능한 꼴)
- 기울기가 e_mask · 점유 head · predictor 첫 블록까지 흐른다

## 4. 시간 측정 [측정]

in-path 판 (세 판 중 가장 무겁다), GPU 8 장 (RTX PRO 6000, 다른 작업 없음), 40 step:

| 항목 | 값 |
|---|---|
| step 당 (5 step 뒤) | 중앙 0.467 초, p90 0.490 초 (첫 step 8 초) |
| 데이터 대기 | step 당 3 ms |
| 메모리 | GPU 당 11.3 GB |
| 학습 중 검증 1 회 | 약 20 초 |
| 시작 (모델 적재) | 약 30 초 |

추정 [이론, 위 측정에서 계산]:

| | 판 하나 | 세 판 (차례로) |
|---|---:|---:|
| 20 epoch | 약 1 시간 (학습 55 분 + 검증 10 회 3–4 분 + 시작) | 약 3 시간 |
| 10 epoch | 약 30 분 | 약 1.5 시간 |

- ⚠️ 앞선 phys3d head 판은 step 당 1.3 초였다 (보조 모듈 포함, 당시 GPU 를 다른 작업과 나눠 썼는지는 모름). 생성기의 GPU 렌더가 다시 돌면 이 정도까지 느려질 수 있다 → 판 하나 약 2.6 시간.
- 학습 뒤 채점 시간 (VOBench held-out 720 쌍 · IntPhys1 late 180 · IntPhys2 late 500 · real_bench 40, 세 판) 은 재지 않았다.

## 5. 결과를 어떻게 읽나 (미리 적어 둔다)

| 결과 | 뜻 |
|---|---|
| ③ > ② ≈ ① (생성기 밖에서) | 경로 안에 넣어야 쓰인다 — 계획의 주장 |
| ③ ≈ ② > ① | 감독 자체가 효과, 자리는 상관없다 |
| ③ ≈ ② ≈ ① | 이 데이터에서는 구조로 가를 수 없다. 데이터 압력이 작아서일 수 있다 (아래 단서 1) |
| 생성기 안 (phys3d val) 에서만 ③ > ② | 경로가 쓰이긴 하지만 넘어가지 않는다 |

단서:
1. phys3d_perm_v1 은 재등장 칸의 손실 보상이 작다 (정답 운반 이득 +4.7 %, VOBench +22 %) [측정, AUX_PERMANENCE §1]. 세 판이 모두 제자리면 구조 차이가 아니라 바닥 효과일 수 있다.
2. in-path 는 채점 때 자기 점유 추측을 쓴다. phys3d 로 배운 칸 추측은 VOBench 에서 "문맥 끝 그대로" 기준선과 같았다 (IoU 20.8 vs 19.8) [측정, AUX_PERMANENCE §2]. 그 추측이 생성기 밖에서 맞지 않으면 경로에 들어가는 값도 기준선이다 [추측].
3. 손실 최적해: 채점 때 ŝ 는 문맥의 함수라 latent L1 의 최적해를 바꾸지 않는다 [이론]. 차이가 난다면 학습이 그 최적에 얼마나 다가가느냐의 차이다.
4. 판마다 한 번씩만 돈다 (seed 하나). 같은 쌍 비교 (McNemar) 로 판정한다.

## 6. 결과 (2026-11-01, 같은 쌍) [측정]

채점 산출물에서 다시 계산했다 (부록의 표 스크립트). 채점 자체는 다른 세션이 11-01 04:23–04:28 KST 에 마지막 체크포인트로 돌렸다.

| 세트 | 쌍 | 릴리즈 | ① 대조군 | ② 보조 head | ③ in-path | ② − ① (좋아짐 / 나빠짐, p) | ③ − ① |
|---|---:|---:|---:|---:|---:|---|---|
| VOBench held-out | 720 | 75.7 | 72.9 | 73.5 | 72.6 | 29 / 25, 0.59 | 24 / 26, 0.78 |
| IntPhys 1 late | 180 | 68.3 | 67.8 | 67.8 | 65.6 | 8 / 8, 1.00 | 7 / 11, 0.35 |
| IntPhys 2 violation-uni | 500 | 54.6 | 53.2 | 51.8 | 52.4 | 20 / 27, 0.31 | 23 / 27, 0.57 |
| real_bench | 40 | 47.5 | 47.5 | 47.5 | 50.0 | 2 / 2, 1.00 | 3 / 2, 0.65 |

- ② · ③ 은 대조군과 구별되지 않는다. ③ − ② 는 VOBench 6 / 12 (p 0.16), IntPhys 1 0 / 4 (p 0.05).
- 대조군 자체가 VOBench 전체를 깎는다 (릴리즈 대비 39 / 59, p 0.04).
- VOBench 영속은 기운다: 방향 A 67.5 → 75.8 (대조군, 14 / 4, p 0.02), 방향 B 98.3 → 93.3 (2 / 8, p 0.06), in-path 87.5 (대조군 대비 1 / 8, p 0.02). "가림막 뒤엔 무언가 나온다" 쪽 편향으로 읽는다 [해석].
- 늦은 가림 · 방향 A (36 쌍): 13.9 → 27.8 / 38.9 / 27.8, 판 사이 p ≥ 0.10 — 판정에 안 씀.
- 비교표 전체: `results/summary_v1.md`. 사용자용 요약 페이지 (비공개 artifact): https://claude.ai/artifact/6XHtd7sGHhFtSBigRBhRd7

§5 의 읽기표로는 "③ ≈ ② ≈ ①" 줄이다. 단서 1 (phys3d 재등장 보상 +4.7 %) 이 그대로 남는다 — 구조 차이가 아니라 바닥 효과일 수 있다.

## 진행 기록
- 2026-10-31 — 설계, 코드, config, 작은 모델 점검, 시간 측정. 학습 대기 (사용자 지시 시 올림).
- 2026-10-31 20:31 KST — 사용자: "GPU 4 개만, VRAM 채워서, 20 epoch". 세 판을 **GPU 0–3 에 동시에** 올림 (판마다 4 GPU × 배치 16 = 전역 64, 20 epoch). GPU 4–7 은 비워 둠.
  tmux 세션 `train_imagine_seg_v1_{ctrl,head,inpath}`, run 폴더 `z_training/runs/imagine_seg_v1_{ctrl,head,inpath}`.
- 2026-11-01 02:55 KST — 세 판 모두 20 epoch 완료 (판마다 약 6.4 시간). 정식 채점은 안 올림 (사용자 지시 시).
  학습 중 감시 결과 [측정, VOBench held-out late 216 쌍, 시작점 = 릴리즈 60.6 %]:

  | 판 | 20 epoch | 12–20 epoch 평균 (5 회) | 영속 72 쌍 (20 epoch) | 학습 데이터 latent L1 (20 epoch 평균) | 가려진 칸 점유 재현율 (학습 배치) |
  |---|---:|---:|---:|---:|---:|
  | ① 대조군 | 59.3 | 59.2 | 54.2 | 0.5512 | — |
  | ② 보조 head | 62.0 | 64.1 | 59.7 | 0.5595 | 0.69–0.75 |
  | ③ in-path | 62.0 | 63.4 | 56.9 | 0.5595 | 0.69–0.75 |

  - 216 쌍의 표준오차가 약 3.3 %p 라 판 사이 차이 (최대 약 5 %p) 는 아직 판정이 아니다. 같은 쌍 McNemar 는 정식 채점 뒤에.
  - 점유 손실을 건 두 판은 학습 데이터 latent L1 이 대조군보다 1.5 % 높다 (점유 손실이 latent 맞춤과 용량을 나눠 씀) [측정, 학습 배치].
  - ② 와 ③ 은 감시 지표 · 점유 재현율이 거의 같다. 경로 안에 넣은 효과는 여기서는 안 보인다.
- 2026-11-01 17:40 KST 무렵 — 정식 채점 결과 정리 (§6), 사용자용 페이지 게시.

## 부록: 재현

```bash
cd /workspace/world_model/vjepa2
# 작은 모델 점검 (CPU) — 스크립트는 세션 임시 폴더에 있었다. 같은 내용: AuxRelease(no_module, sup.binary, inpath) 로
#   e_mask=0 일 때 릴리즈와 출력 비교 · 정답 점유 / 옮긴 점유 / p_gt=0 출력 비교 · 기울기 확인
# 시간 측정 (40 step)
GPUS=8 FOLDER=/tmp/timing_inpath SET="optimization.epochs=1 optimization.ipe=40 optimization.warmup=0.1 optimization.anneal_epochs=0.5 val.at_start=false meta.save_every_freq=0" \
  bash z_training/train.sh imagine_seg_v1_inpath
# 학습 (2026-10-31 실제로 올린 명령: GPU 0–3 에 세 판 동시)
for arm in ctrl head inpath; do
  GPUS=4 DEVICES="cuda:0 cuda:1 cuda:2 cuda:3" SET="data.batch_size=16" bash z_training/tmux_train.sh imagine_seg_v1_$arm
done
watch -c -n 10 bash z_training/monitor_release_aux.sh imagine_seg_v1_inpath
# 채점 (모델 vast_imgseg_{ctrl,head,inpath}, cvpr/registry/models.yaml)
bash cvpr/eval/testbed/run.sh "vobench_v1_split_v2_test real_bench_quads" vast_imgseg_ctrl
DS=intphys1_late_uni bash cvpr/eval/intphys1_late/run.sh vast_imgseg_ctrl
DS=intphys2_violation_uni bash cvpr/eval/intphys1_late/run.sh vast_imgseg_ctrl
# 같은 쌍 표 · 페이지 그림
/venv/vjepa/bin/python new_idea/imagine-seg/eval/summarize_v1.py
/venv/vjepa/bin/python new_idea/imagine-seg/eval/make_page_examples.py roll_behind
```
