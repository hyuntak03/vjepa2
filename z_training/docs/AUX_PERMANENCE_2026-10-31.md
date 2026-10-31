# 보조 모듈 · 보조 head 로 영속 가르치기, 그리고 데이터 v2 (2026-10-30 ~ 31)

> **3줄 요약**
> - **질문** — 릴리즈 predictor 가 가려졌다 다시 나오는 물체를 미래에 그리지 못한다 (VOBench late 영속 A 약 20 %). 보조 모듈 · 보조 감독 head 를 붙이면 다른 벤치로 옮겨지는 영속을 배우나.
> - **답** — 아니다. 같은 생성기 안에서는 크게 오르지만 (VOBench held-out 75.7 → 87–92) 실제 영상 (real_bench) · IntPhys2 로는 어떤 구조도 안 넘어간다. 모듈 · head 는 같은 데이터의 단순 post-FT 대조군보다 낫지 않다. **막힌 곳은 구조가 아니라 데이터다** [측정].
> - **다음 할 일** — 영속 · 이식 원칙 (DATA_DESIGN_PRINCIPLES 1.5 부) 을 반영한 phys3d_perm_v2 생성 (진행 중) → 보조 head 판으로 학습 → VOBench held-out · real_bench · IntPhys1 · 2 late 로 판정.

사용자 지시 (2026-10-30 ~ 31): "release predictor 에 다른 모듈을 붙일 수 있도록" → "ㅇㅇ 하고 학습까지 바로 해봐" → "모듈한테 context 보고 물체 놓아져야될 칸 supervision" → "상대 표현 괜찮은데" → "당연히 predictor 는 열어야지" → "VOBench 로 학습해보자" → "보조 head 로 강제로 release predictor 가 배울 수 있도록" → "데이터 이렇게 만들어야될 듯 … cpu, gpu 병렬 최대화".

근거 표시: [측정] 직접 잰 값 · [이론] 정의 · 계산 · [추측] 아직 안 잰 것.

## 용어

| 용어 | 뜻 |
|---|---|
| 블록 | 연속 2 장을 encoder 에 넣은 위치별 벡터. 16 × 16 칸, shape (256, 1280) |
| 문맥 · 미래 | 앞 C 장 (predictor 입력) · 그 뒤 장 (예측 대상) |
| 보조 모듈 | 릴리즈 predictor 와 같은 구조를 작게 (폭 256 · 4 층 · 4 head) 지은 망. 문맥 토큰 → 미래 칸마다 기억 벡터 (1280) |
| cross | 릴리즈 predictor 의 4 · 8 · 12 번째 블록 뒤에서 미래 토큰이 기억을 읽는 cross-attention (3 개, gate 0 에서 시작) |
| 칸 분류 감독 | 미래 칸마다 배경 / 보이는 물체 / 가려진 물체 (문맥에서 본 물체만) 를 맞히는 CE |
| 이동량 감독 (상대 표현) | 미래 블록 t 의 칸 (r, c) 에서, 문맥에서 그 칸에 마지막으로 보인 물체가 블록 t 까지 움직인 양 (dx, dy, 칸) 을 맞히는 L1 |
| 보조 head | 감독용 선형 head. 기억 위 · predictor 은닉 위 · predictor 출력 위 세 자리 중 하나에 붙는다. 학습 때만 쓴다 |
| matched pair | 문맥이 픽셀 단위로 같은 가능 · 불가능 영상 한 쌍. surprise (불가능) > surprise (가능) 이면 정답 |
| 방향 A / B | VOBench · real_bench: A = 문맥에 물체가 있음 (다시 나와야 맞음), B = 없음 / 다른 물체 (안 나와야 맞음) |
| 같은 쌍 비교 | 두 모델이 같은 쌍에서 맞음 → 틀림 / 틀림 → 맞음 이 된 수를 센다 (McNemar). 정확도 두 개를 따로 비교하는 것보다 민감하다 |
| 운반 상한 | 정답 위치로 물체 토큰을 옮긴 예측 − 옛 자리만 지운 예측의 L1 이득. 데이터가 "물체를 옮겨 그리기" 에 손실 보상을 주는 정도 (판별 상한) |

---

## 1. 보조 모듈, 감독 없음 (release_aux_perm)

**왜** — 릴리즈 predictor 는 늦은 가림 (문맥 끝에 가려짐) 에서 과거 물체를 다시 꺼내 쓰지 않는다 (2026-10-30 측정: 진짜 미래 물체 칸 query 의 과거 물체 key attention 1.6 %, 균등 2.7 % 보다 낮음; RoPE 위치를 바꿔도 그대로 → 내용 검색 실패). 작은 모듈이 그 정보를 기억으로 건네면 나아지나.
**무엇을** — 릴리즈 predictor 고정, 보조 모듈 + cross 만 학습 (7.7 M). 손실은 미래 토큰 latent L1 그대로. phys3d_perm_v1 (4 fps 32 장, 22,673 편), 문맥 {10, 12, 14, 16} 장, 20 epoch.
**결과** [측정]
- 학습 손실 0.601 → 0.582 (−3.2 %). VOBench late (학습 중 val 432 쌍) 60.9 → 58.3.
- 손실 감소가 칸 종류와 무관하게 고르다 (phys3d val 876 편): 배경 +3.33 % · 보이는 물체 +2.90 % · **재등장 +2.75 %** · 가려진 물체 +1.64 %. 같은 클립 짝으로 재등장 − 배경 = −0.53 ± 0.06.
- gate 는 0.048 이지만 실제 기여는 크다: 4 층 은닉 크기의 67 % 가 더해지고 출력이 릴리즈 오차의 25 % 만큼 움직인다.
**해석** — 모듈은 크게 쓰이는데 영속이 아니라 배경 보정을 한다. phys3d_perm 은 재등장에 손실 보상이 작다 (운반 상한 재등장 칸 +4.7 %, VOBench +22 %; 릴리즈 L1 재등장 칸 0.626 vs 배경 0.601).

## 2. 칸 분류 감독 (release_aux_perm_sup)

**왜** — 라벨 없이는 모듈이 영속을 담을 이유가 없다.
**무엇을** — 기억 위 칸 분류 head (CE, 클래스 가중 1 · 4 · 10). 라벨 = phys3d_perm states (z_training/data/build_perm_occupancy.py). e9 에서 멈춤.
**결과** [측정] — 학습 데이터 안 가려짐 재현율 0 → 0.64. 하지만 칸 예측을 생성기 밖 (VOBench late 라벨) 에서 재면 문맥 끝 그대로 두는 기준선과 같다 (움직이는 물체 IoU 20.8 vs 19.8, 블록별 재현율이 기준선과 같이 86 → 7 로 무너짐). phys3d 안에서는 IoU 52 vs 34.
**해석** — 절대 칸 분류는 장면 내용 · 배치로 외울 수 있어 생성기 안에서만 통한다.

## 3. 상대 표현 감독 (release_aux_perm_rel)

**왜** — 문맥 토큰의 속도 신호는 생성기를 넘는다: 물체 칸 토큰 → 속도 선형 회귀를 phys3d 로 맞춰 VOBench 에 그대로 쓰면 방향 cos 0.96, x 상관 0.93 (VOBench 안 상한 R² 0.99; 위치는 오라클 = 판별 상한) [측정]. 이동량은 속도 × 시간이라 이 신호에 바로 닿는다.
**무엇을** — 기억 위 이동량 head (L1, 가중 0.3) + 칸 분류 (0.3). e5 에서 멈춤.
**결과** [측정] — VOBench 이동량 (e3): 움직이는 물체 오차 2.75 칸 vs 0 이동 3.99 (−31 %), 방향 cos 0.85 (등속 · 가속 0.994, 오르막 0.46 ✗). **e5 에서는 3.27 로 나빠짐** (크기 보정이 생성기 쪽으로). VOBench 채점은 그대로 (61.7 → 58.8).
**해석** — 상대량은 생성기를 넘는다. 하지만 고정 predictor 는 그 정보를 받아 물체를 그리지 못한다.

## 4. predictor 열기 + 출력 감독 (release_aux_perm_open)

**무엇을** — rel e5 에서 이어 받고 릴리즈 predictor 를 연다 (학습률 × 0.1, WSD 일정 — WarmupCosineSchedule 은 param group 의 lr_scale 을 무시한다). predictor 출력 위 칸 분류 head 추가. e5 에서 멈춤.
**결과** [측정] — 학습 데이터 안: latent 0.5734, 출력에서 가려진 물체 재현율 0.64. VOBench late 정식 채점 (e3, 같은 쌍 432): 61.8 → 60.6, 바뀐 쌍 23 / 28 (p 0.48). **영속 A 단순 도형 19.4 → 25.0 · 실제 물체 22.2 → 22.2.**
**해석** — 학습 데이터 (phys3d_perm) 의 압력이 작아 predictor 를 열어도 재등장을 그리는 행동이 VOBench 로 안 생긴다.

## 5. VOBench 로 학습 — 대조군 · 모듈 · 보조 head (release_aux_vob · release_aux_vob_heads)

**왜** — VOBench 는 재등장 보상이 크다 (+22 %). 같은 데이터에서 모듈 · head 가 단순 post-FT 보다 다른 벤치로 잘 옮겨지나.
**무엇을** — 데이터 = VOBench split v2 학습 + gravity 학습 828 편 (대조군 `vith_release_vobpft_e20` 과 같다). 문맥 16 장, predictor 최대 lr 5e-5, 전역 배치 32, 20 epoch.
- 모듈 판: 모듈 + cross + 기억 head 2 + 출력 head, predictor 열기 (e13 에서 멈춤)
- 보조 head 판: **모듈 · cross 없이** 칸 분류 · 이동량 head 를 predictor 은닉 (8 번째 블록 뒤, 384) 에, 출력 head 는 출력에. 추론은 릴리즈 구조 그대로 (출력 차이 0 실측). 20 epoch
- held-out = VOBench split v2 test (학습과 block 겹침 0). ⚠️ 이전 run 들의 학습 중 val (vobench_v1_late 432 쌍) 은 이 학습 split 과 216 block 중 108 이 겹친다.
**결과** [측정] — 릴리즈와 같은 쌍 비교

| 모델 | VOBench held-out (720 쌍) | IntPhys1 late (180) | IntPhys2 late (500) | real_bench (40) |
|---|---:|---:|---:|---:|
| 릴리즈 | 75.7 | 68.3 | 54.6 | 47.5 |
| 대조군 (post-FT 만) | **91.7** | 73.9 (p 0.05) | 52.8 | 45.0 |
| 모듈 + head | 90.7 | 73.9 (p 0.04) | 55.0 | 50.0 |
| 보조 head 만 | 87.4 | 71.1 (p 0.23) | 53.2 | 45.0 |

- 보조 head 판은 학습 데이터 안에서 모듈보다 훨씬 빨리 배운다 (은닉의 이동 L1 e3 0.37 vs 모듈 e7 0.43; 가려진 물체 재현율 1.00 vs 모듈 0.00 — VOBench 학습 라벨에서 가려진 칸은 0.11 % 뿐).
- IntPhys1 상승의 대부분은 방향 맞바꿈: 움직이며 가려지는 쌍 방향 1 40 → 50–60, 방향 2 70 → 57–63.
**해석** — 같은 생성기 안에서는 셋 다 크게 오르고, 실제 영상 · IntPhys2 로는 셋 다 안 넘어간다. 구조 (모듈 · head) 는 이식을 돕지 않는다 [측정]. VOBench 는 경로가 21 가지라 외워서 풀린다 (원칙 1) [추측, 다른 세션 E8].

## 6. real_bench 등록

- 원본 `/workspace/dataset/benchmarks/real_bench/rubric_office_quads` (TCOW Rubric Office, 실제 영상 20 × 4 = 80 클립). 위반 지점 t_splice 앞 [0, t_splice) 균등 16 장 + 뒤 [t_splice, 끝) 균등 16 장. ⚠️ 문맥 · 미래 간격이 다르다.
- 짝: A = 계속 있음 vs 숨은 동안 사라짐, B = 처음부터 없음 vs 숨은 동안 생겨남. 문맥 PNG 바이트 동일 40 쌍 (lightball 3 개는 인코딩 차이가 있어 가능 영상의 문맥 PNG 를 쓴다).
- 릴리즈 47.5 (A 25 / B 70) — 미래에 물체가 있으면 surprise 가 큰 외형 편향.

## 7. 데이터 v2 (phys3d_perm_v2, 생성 중)

- 원칙 (z_training/docs/DATA_DESIGN_PRINCIPLES_2026-10-26.md 1.5 부, 다른 세션 v0 점검표) 중 이 기계 생성기에 반영한 것: 움직이는 물체 그림자 끔 (원칙 6 ①) · 미끼 가림막 1–2 개를 클립 70 % 에 (경로와 무관한 자리 · 시각, 원칙 6 ②) · 끝까지 다시 안 나타나는 앵커 가림 (원칙 7) · 렌즈 29 → 45 mm. 그릇 덮기 (원칙 9) 는 사용자 결정으로 뺐다.
- 시험 144 편 [측정]: 통과 83 % · 끝까지 안 나타남 25 % (v1 0 %) · 주 물체 반지름 중앙 22.4 px (v1 15.1, VOBench 26.1) · 미끼 0 / 1 / 2 개 = 46 / 37 / 37 편.
- 검증 세트 745 / 960 통과. 학습 세트 24,000 작업 생성 중 (2026-10-31 18:00 KST 시작).

## 8. 생성 속도 — 병목과 고친 것

| 단계 | 측정 | 고친 것 |
|---|---|---|
| 라벨 (CPU, Blender 물리 + 레이캐스트) | 편당 중앙 7.4 s (재시도 포함, p90 37 s), 코어당 독립 | 그대로 (워커 160) |
| 렌더 GPU (Cycles OptiX) | 혼자 쓰면 장당 0.01–0.04 s, **학습 · 채점과 GPU 를 나눠 쓰면 장당 약 1 s** (물체 15 개 BVH 를 매 장 다시 굽는 0.45 s 포함) | GPU 당 워커 2 → 4–8 (공유 GPU 에서 2 개 3.3 · 6 개 10.4 · 12 개 15.3 편/분) |
| 렌더 CPU (Cycles CPU 1 스레드) | 워커 64 개 20 편/분 (편당 119 s) | `--cpu-render N` — CPU 렌더 워커가 GPU 워커와 같은 대기열을 함께 비운다 (nice 15) |
| 출력 | PNG 무압축 → ffmpeg 재인코딩 → 지우기 | Blender 가 바로 H.264 mp4 (crf 17) |
| 잡음 제거 | GPU OptiX / CPU OIDN 이 섞임 | `PHYS3D_GPU_DENOISER=OPENIMAGEDENOISE` 로 통일 |

- 동적 BVH (`PHYS3D_BVH_DYNAMIC`) · 조명 트리 끄기 (`PHYS3D_LIGHT_TREE=0`) 는 넣었지만 공유 GPU 에서 차이가 분명하지 않아 기본은 꺼 둔다.
- 생성 중 렌더 워커 더 붙이기: `z_training/data/add_render_workers.sh`.

## 아직 모르는 것

- v2 데이터가 실제 영상 · IntPhys2 로 넘어가는가 (다음 판정).
- real_bench 는 40 쌍이라 표준오차 약 ±8pt. 작은 차이는 못 본다.
- 보조 head 판에서 은닉 몇 번째 층이 좋은가 (8 번째만 시험).

---

## 부록: 재현

```bash
# 학습 (config)
GPUS=8 bash z_training/tmux_train.sh release_aux_perm          # 1. 모듈, 감독 없음
GPUS=8 bash z_training/tmux_train.sh release_aux_perm_sup      # 2. + 칸 분류
GPUS=8 bash z_training/tmux_train.sh release_aux_perm_rel      # 3. + 이동량
GPUS=8 SET="optimization.epochs=10 optimization.anneal_epochs=5" bash z_training/tmux_train.sh release_aux_perm_open   # 4. predictor 열기
GPUS=8 bash z_training/tmux_train.sh release_aux_vob           # 5. VOBench, 모듈
GPUS=8 bash z_training/tmux_train.sh release_aux_vob_heads     # 5. VOBench, 보조 head 만
watch -c -n 10 bash z_training/monitor_release_aux.sh <run>
# 라벨
python z_training/data/build_perm_occupancy.py --split {train,val} [--disp]
python z_training/data/build_vobench_occupancy.py
# 분석
python z_research/scripts/analysis/aux_perm_indomain.py            # 1 의 칸 종류별 이득
python z_research/scripts/analysis/aux_gate_contribution.py --run <run> --ckpt <ckpt>
python z_research/scripts/analysis/aux_occ_generalization.py --data {perm,vobench} --run <run> --ckpt <ckpt>
python z_research/scripts/analysis/velocity_probe_cross_generator.py
python z_research/scripts/analysis/transport_upper_bound.py --data {perm,vobench,...}
# 채점 (같은 쌍 비교)
GPUS=4 bash cvpr/eval/testbed/run.sh "vobench_v1_split_v2_test real_bench_quads" "vith vith_release_vobpft_e20 vast_aux_vob vast_aux_vob_heads"
DS=intphys1_late_uni GPUS=4 bash cvpr/eval/intphys1_late/run.sh "<모델들>"; DS=intphys2_violation_uni ...
python z_research/scripts/analysis/compare_pairs_vs_ref.py --ds <ds> --ref vith --new <모델> [--task intphys1_late --index <index.csv> --group <열>]
# real_bench 만들기
python z_research/scripts/data/build_real_bench_quads.py
# 데이터 v2
bash z_training/data/run_gen_phys3d_perm_v2.sh
GPU_PER=4 GPUS=0,1,2,3,4,5,6,7 bash z_training/data/add_render_workers.sh /workspace/dataset/training_data/phys3d_perm_v2/train
```
결과 파일: `z_analysis/results/{aux_perm_indomain,aux_occ_generalization,velocity_probe_cross_generator,transport_upper_bound}` · `cvpr/results/eval/{testbed,intphys1_late}/` (z_analysis/README.md 에 목록).
