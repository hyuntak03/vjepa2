# 릴리즈 predictor 정리 — 진입점 (2026-10-30 ~ 10-31)

> **질문** — 문맥 끝 가림 (late) 에서 릴리즈 V-JEPA 2 predictor 는 물체 정보를 어디까지 미래로 옮기나? 무엇을 붙이면 옮길 수 있나?
> **답** — **문맥 encoder 는 가려진 물체의 모양 · 색을 담는데 (가린 블록에서 모양 82–89 %), predictor 의 예측에는 덜 실리고 (자기 판독 모양 56–66 %), 문맥 encoder 로 배운 판독기로는 우연에 가깝게만 읽힌다 (22–41 %).** 물체 내용을 맞는 미래 자리에 넣어 주면 사라짐 정답이 오르지만, 문맥에서 추정한 운동으로는 그 자리를 못 맞힌다.
> **다음** — ① 의 판독을 PFT p 에 건다 · ② 의 지도 모듈에 "미래 자리" supervision 을 주는 운동 head 를 시험한다.

한 분석에 폴더 하나다. 폴더마다 README 가 왜 → 어떻게 → 결과 → 해석 → 재현 순서로 쓰여 있다.

**종합 정리:** [`SUMMARY_visible_late_pft_transfer.md`](SUMMARY_visible_late_pft_transfer.md) — visible vs late · PFT 가 바꾸는 것 · 이식이 안 되는 이유 (①② + 이식 조사 R1–R4 를 묶음, 2026-10-31).

| 폴더 | 무엇 | 한 줄 결과 | 상태 |
|---|---|---|---|
| [`01_block_probe/`](01_block_probe/README.md) | visible · late k 마다 문맥 블록 1–8 (z) · 예측 블록 9–16 (p) 의 물체 판독 + 문맥 encoder 판독기 → p 이식 | late: z 는 가린 블록 8 에서도 모양 82–89 %, p 자기 판독 56–66 %, **z 판독기 → p 는 22–41 % (우연 16.7 에 가까움)**. visible: 이식이 가까운 미래 82 → 먼 미래 43 % | ★ 지금 판 = VOBench_v2 · attentive (`figures/v2_attn/{late,visible}/`) |
| [`03_binding_train/`](03_binding_train/README.md) | PFT 에 binding 항 (참 미래 물체 자리 3×3 의 L1) · 운동 head 를 더해 학습 (세 팔, VOBench_v2 궤적 묶음 split) | binding 이 late 전체 +8.5 · 사라짐 A +19.4 · z 판독기 → p 이식 62 → 74 % (모양). ⚠️ 정정: 이득은 학습에 있던 미래 경로에만 (처음 보는 경로 +2.8). 다른 생성기 (v11 · IntPhys) 이득 없음. 운동 head 추가 이득 없음 | seed 1 개 |
| [`04_rubric_office/`](04_rubric_office/README.md) | 실제 영상 (TCOW Rubric Office 2×2, 20 영상) 을 violation 시점 앞뒤 16 장으로 채점 | 전체 52–58 % (우연 · 복사와 구분 안 됨). 사라짐 25–30 % · 생겨남 75–95 % — 합성 late 와 같은 방향. VOBench 학습 판도 같음 | 20 영상, 창 두 가지 |
| [`02_map_inject/`](02_map_inject/README.md) | 공간 지도 (마지막으로 보인 물체 토큰 → 추정 미래 자리) 를 p 에 넣는 학습 전 상한 시험 | 참 자리면 63 %, 추정 운동이면 25–30 % (릴리즈 8 · 복사 57) → 병목은 운동 외삽 | VOBench_v1 판, 사전 판정 "중간" |
| [`06_blender_track/`](06_blender_track/README.md) | 경로를 외울 수 없는 학습셋 — 눈높이 Blender 렌더, 연속 무작위 궤적 · 가림 사건 · 64×64 amodal/visible 라벨 | late 미래 경로 21 (VOBench) → ~5,750 (설계 2,000, 보폭 3), late 창 clip 68 %, GPU 8 장 clip 당 ~1 초, 투영 오차 0.000 px | 본 렌더 중 (학습 4,000 + 검증 400) |

## 폴더 지도

```
release/
  README.md                      ← 이 문서
  01_block_probe/
    README.md                    지금 판 (v2 · visible / late) → 이전 판 (v1, 기록) → 재현
    figures/v2_attn/late/        ★ late_k{1,2,3}_block_probe.{png,pdf} + results/
    figures/v2_attn/visible/     ★ visible_block_probe.{png,pdf} + results/
    _archive/                    선형 판 · v1 판 (옮김, 지우지 않음)
    logs/
  02_map_inject/
    README.md
    results/ · logs/
  03_binding_train/
    README.md                    세 팔 학습 · 평가 (표준 채점 · 한 번 통과 채점 · 판독)
    results/ · probe/ · te/ · eval/ · logs/
  04_rubric_office/
    README.md                    실제 영상 permanence 채점 (창 uniform · symmetric)
    results/ · te/ · logs/
  06_blender_track/
    README.md                    학습셋 설계 · 렌더 전 검사 · 처리량 · 라벨 검증
    results/ · figures/          (데이터는 nfs_cache/blender_track/v0)
```

## 공통

- 모델: 릴리즈 ViT-H (`vjepa2-vith-fpc64-256`), probing 관례 bfloat16.
- 자료: VOBench_v1 · VOBench_v2 late (문맥 끝 k 샘플 + 예측 앞 k 샘플 가림, k = 1–3). VOBench 에는 k = 4 가 없다.
- 큰 배열은 레포 밖 NFS 에 있다: `/data/hyuntak/project/2026/2027_cvpr/nfs_cache/` (`release_late_vobench/` · `release_late_vobench_v2/` · `VOBench_v2_late_frames/`).
- 관련 조사: `../transfer_gap/README.md` (판독기 이식 · 영속 · PFT 비전이).
