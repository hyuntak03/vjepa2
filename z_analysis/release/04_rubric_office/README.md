# ④ 실제 영상 object permanence (Rubric Office quads) — V-JEPA 2 채점 (2026-10-31)

> **질문** — 실제 사무실 영상에서 물체가 숨었다 다시 나타나는 장면을, violation 시점 앞뒤 16 장씩으로 채점하면 V-JEPA 2 는 가능 · 불가능을 가르나?
> **답** — **못 가른다.** 전체 정확도 52–58 % 로 우연 · 복사 기준선과 구분되지 않는다. 방향은 합성 late 와 같다: 사라짐 (물체 있던 문맥 → 빈 미래) 을 25–30 % 로만 불가능으로 보고, 생겨남은 75–95 % 로 맞힌다. VOBench 로 학습한 판 (binding 포함) 도, 자기회귀 vast AR (· + PFT) 도 같다 (전체 52.5 / 52.5, AR 공간 복사 52.5 / 50.0).
> **다음** — 표본 (20 영상) 이 작아 차이를 못 가른다. 물체 자리만 보는 판독 · 블록별 surprise 로 다시 나타나는 순간을 따로 보거나, 영상을 늘린다.

---

## 용어

| 용어 | 뜻 |
|---|---|
| quad | 같은 실제 영상에서 만든 네 클립: possible_present (있음→있음) · impossible_vanish (있음→없음) · impossible_appear (없음→있음) · possible_absent (없음→없음). 물체를 지운 판은 inpainting (DiffuEraser) |
| violation 시점 t | 이어 붙인 프레임 (t_splice) = 물체가 완전히 숨은 구간의 가운데 |
| 짝 A · B | A = 문맥 "있음" (possible_present vs impossible_vanish, 사라짐) · B = 문맥 "없음" (possible_absent vs impossible_appear, 생겨남). 짝 안 두 클립은 문맥이 같다 |
| uniform 창 | 문맥 16 장 = [0, t) 에서 균등, 미래 16 장 = [t, 끝) 에서 균등 (앞뒤 간격이 다를 수 있다) |
| symmetric 창 | 앞뒤 같은 간격 s = min(t, 끝 − t) / 16 |
| 복사 | 마지막 문맥 블록을 미래 8 블록에 그대로 둔 기준선 |

## 1. 왜 했나
- 합성 (VOBench · v11) 에서 본 "문맥 끝 가림 → 사라짐을 그럴듯하게 봄" 이 실제 영상에서도 나타나는지 본다.
- 앞서 VOBench 로 학습한 판 (L1 PFT · binding) 이 실제 영상으로 옮겨지는지도 같이 본다.

## 2. 무엇을 어떻게 했나
- **자료:** TCOW Rubric Office 기반 20 영상 × 4 클립 (640×480, 30 fps). 원본은 2026-10-31 에 `testbed/real_bench/rubric_office_quads/` 로 옮겨졌다 (프레임은 옮기기 전에 뽑았고 index 80 clip 그대로). 사건 종류: occlusion 9 · containment 10 · out_of_frame 1. 출처 · 만드는 법은 원 데이터 README.
- **창 (영상마다):** 실효 30 fps 프레임 번호 기준이다. 60 fps 중복 영상 3 개는 짝수 프레임만 쓴다.

  | 창 | 문맥 간격 (장) | 미래 간격 (장) |
  |---|---|---|
  | uniform (요청) | 5.3–11.2 | 3.9–14.7 |
  | symmetric (점검) | 3.7–7.0 | 3.7–7.0 |

  - 두 창 모두 미래에 물체가 다시 보이는 프레임이 들어 있다 (영상마다 1–15 장, plantcupball2 는 1 장).
- **채점:** 표준 surprise (미래 8 블록 L1 평균), matched 짝. 640×480 은 채점기 기본대로 256×256 으로 늘려 맞췄다 (화면비 안 지킴).
  - 한 번 통과로 릴리즈 · VOBench PFT 셋 (a0 L1 · a1 + binding · a2 + 운동 head, `../03_binding_train/`) · 원 PFT (VOBench_v1) · 복사를 같은 화소로 채점했다.
  - **vast AR** (자기회귀 predictor, 상태 되먹임 판 ep70) · **vast AR + PFT** (VOBench split v2 + gravity) 는 한 번 통과 채점기가 이 판을 못 읽어 표준 채점 하네스로 같은 창 · 같은 프레임을 채점했다 (2026-10-31).
    - AR 은 문맥 · 목표가 블록마다 따로 낸 target encoder latent 라 L1 공간이 다르다. 그래서 같은 AR 공간의 복사 기준선 ("AR 복사") 과만 견주고, 다른 모델과는 쌍 정확도만 비교한다.
    - AR 은 문맥을 target encoder 로, 릴리즈는 online encoder 로 받는다 (encoder 교란, 기존 규약).
- **문맥 감사:** 16 영상은 짝끼리 문맥 16 장이 화소까지 같다. 나머지 4 개 (handball1 · lightball4–6) 는 재인코딩 차로 12–15 장만 같다. 이 4 개를 뺀 판 (clean) 을 따로 냈다.

## 3. 결과 [측정] — 정확도 %, n = 짝 수, 우연 50

| 창 | 칸 | n | 릴리즈 | a0 | a1 | a2 | 원 PFT | 복사 | **vast AR** | **vast AR + PFT** | AR 복사 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | 사라짐 (A) | 20 | 30.0 | 20.0 | 30.0 | 30.0 | 30.0 | 30.0 | 35.0 | 35.0 | 45.0 |
| uniform | 생겨남 (B) | 20 | 75.0 | 80.0 | 80.0 | 80.0 | 70.0 | 85.0 | 70.0 | 70.0 | 60.0 |
| uniform | 전체 | 40 | 52.5 | 50.0 | 55.0 | 55.0 | 50.0 | 57.5 | **52.5** | **52.5** | 52.5 |
| symmetric | 사라짐 (A) | 20 | 25.0 | 25.0 | 30.0 | 30.0 | 30.0 | 30.0 | 25.0 | 25.0 | 30.0 |
| symmetric | 생겨남 (B) | 20 | 90.0 | 95.0 | 85.0 | 90.0 | 95.0 | 75.0 | 80.0 | 80.0 | 70.0 |
| symmetric | 전체 | 40 | 57.5 | 60.0 | 57.5 | 60.0 | 62.5 | 52.5 | **52.5** | **52.5** | 50.0 |

- clean (16 영상) 도 같은 모양이다: uniform 전체 50–56 · symmetric 56–63, 사라짐 25–31 %.
- 릴리즈 − 복사: uniform −5 [−15, 5] · symmetric +5 [−8, 18]. a1 − 릴리즈: +2 [0, 8] · 0 [−8, 8].
- vast AR − AR 복사: uniform 0 [−12, 12] · symmetric +2.5 [−10, 15]. vast AR + PFT − vast AR: 둘 다 0 [−8, 8] (40 짝 중 판정이 다른 짝 없음 수준). vast AR − 릴리즈 (쌍 정확도): 0 [−10, 12] · −5 [−15, 5].
- 사건 종류별 (전체): occlusion 50–56 · containment 45–70 (n 18 · 20). 모두 CI 가 50 을 넓게 덮는다.

## 4. 해석
- **실제 영상에서도 사라짐 쪽으로 기운다** [측정]. 물체가 있던 문맥에서 "빈 미래" 를 덜 놀랍게 보는 방향 (사라짐 25–30 %, 생겨남 75–95 %) 이 합성 late (VOBench 사라짐 A 12 · B 95, v11 1.8 · 100) 와 같다.
- **전체로는 우연 · 복사와 구분되지 않는다** [측정]. 20 영상이라 ±15–20 pt 의 차이를 못 가른다.
- **VOBench 로 한 학습은 실제 영상에 옮겨지지 않는다** [측정]. L1 PFT · binding · 운동 head · 원 PFT 모두 릴리즈와 같은 자리에 있다. 03 의 "다른 생성기로는 이득이 없다" 와 같다.
- **자기회귀 vast AR 도 같다** [측정]. 전체 52.5 / 52.5 로 AR 공간 복사 (52.5 / 50.0) · 릴리즈와 구분되지 않고, 사라짐 25–35 % · 생겨남 70–80 % 로 같은 방향이다. vast AR 은 자연 영상 (K710 · SSv2 · HowTo100M 일부) 으로 학습했지만 이 20 영상에서는 이점이 보이지 않는다. VOBench + gravity PFT 를 더해도 그대로다.

**단서**
- 표본이 작다 (영상 20, 방향마다 20 짝).
- 표준 채점의 간격 (3 장) 보다 넓은 간격이다. 숨은 구간이 길어서 3 장 간격 16 장으로는 다시 나타나는 순간이 대부분 창 밖에 있다.
- 다시 보이는 프레임이 미래 16 장 중 1–15 장뿐이다. 전체 토큰 평균 L1 은 그 몇 장의 물체 차이를 작게 본다.
- 4:3 영상을 정사각으로 늘렸다.
- inpainting 흔적이 단서가 될 수 있다 (원 데이터 README "알아둘 점"). impossible 클립은 흔적이 앞이나 뒤 한쪽에만 있다.
- 짝 문맥이 화소로 같지 않은 영상이 4 개 있다 (clean 판으로 따로 냈다).

---

## 부록: 재현

```bash
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
$P z_research/scripts/data/build_rubric_quads_frames.py        # 창 두 가지 PNG (nfs_cache/rubric_quads_frames/<창>) + data_csv/rubric_quads_<창>/{index.csv, audit.json}
R=z_training/runs
for w in uniform symmetric; do sbatch -p batch_vll -w vll2 --gres=gpu:2 -c 16 --mem=64G --wrap \
  "$P z_research/scripts/analysis/transfer_gap_te.py --name rubric_$w --dataset vobench_v1 \
   --set data.root=$PWD/data_csv/rubric_quads_$w data.index_csv=index.csv data.frames_root=/data/hyuntak/project/2026/2027_cvpr/nfs_cache/rubric_quads_frames/$w window.stride=1 window.frames_start=0 \
   --preds release=$R/release_vith/latest.pt,a0=$R/bind_v2_a0/latest.pt,a1=$R/bind_v2_a1/latest.pt,a2=$R/bind_v2_a2/latest.pt,vobftg=$R/vobench_release_postft_g/latest.pt \
   --out $PWD/z_analysis/release/04_rubric_office/te/$w --gpus 2"; done
# vast AR · vast AR + PFT · AR 복사 (표준 하네스, 창마다 잡 하나, vll3 GPU 1 장, ~6 분)
for w in uniform symmetric; do sbatch -p batch_vll -w vll3 --gres=gpu:1 -c 12 --mem=64G --export=ALL,W=$w z_analysis/release/04_rubric_office/logs/run_ar_harness.sh; done
#   → harness/<창>_{vast70, vast70g, vast70_copyar}/ (vast70_copyar = surprise.ar_baseline=copy_last)
$P z_research/scripts/analysis/rubric_quads_table.py           # → results/rubric_table.json (te/<창> + harness/<창>_*)
```

| 파일 | 무엇 |
|---|---|
| `results/rubric_table.json` | 창 × 부분집합 (all · clean · 사건) × 방향 × 모델 정확도 · CI · 짝 차이 |
| `te/<창>/` | 한 번 통과 채점 원배열 (l1 · copy · meta) |
| `harness/<창>_<모델>/` | vast AR 판 표준 하네스 채점 (summary.json · per_block.json · _resolved.yaml) |
| `logs/run_ar_harness.sh` | 위 채점 스크립트 (창 = 환경변수 W) |
| `data_csv/rubric_quads_<창>/audit.json` | 영상마다 뽑은 프레임 번호 · 간격 · 다시 보임 장수 · 문맥 화소 일치 |
