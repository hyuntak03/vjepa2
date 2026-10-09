# DINO-Foresight × IntPhys 1 dev — Garrido sliding 프로토콜 (2026-10-02)

**사용자 지시**: DINO-Foresight (Karypidis et al., NeurIPS 2025; frozen DINOv2 ViT-B/14-reg + 프레임 단위 feature predictor, Cityscapes 학습) 를 환경 세팅 · 체크포인트 다운로드 뒤 **IntPhys 1 평가 프로토콜대로** 평가.
하네스는 우리 것 그대로 (`z_research/scripts/run.sh intphys1_sliding intphys1_dev <model>`, `analysis/model_loaders.py` 에 `family: dinof` 로더만 추가). 레포 · 환경 · 스모크는 `/data/hyuntak/project/2026/2027_cvpr/DINO-Foresight/z_smoke/README.md`.

## 결론 (3 줄)

1. **DINO-Foresight highres 는 IntPhys 1 에서 84.4 %** (A.8 격자 최고 `skip2_w16`, O1 73.3 / O2 91.7 / O3 88.3). 같은 프로토콜의 V-JEPA 2 ViT-H 88.9, V-JEPA 2.1-g 51.1, VideoMAEv2-g 54.4, ViT-L 64.4. 주행 영상만 본 "frozen encoder + 새 predictor" 가 합성 물리 벤치마크에서 ViT-H 에 4.5 pt 로 붙는다. 패턴도 같다 — O1 (영속) 이 가장 낮다.
2. **그런데 복사 기준선이 predictor 를 이긴다.** 같은 공간 · 같은 창에서 "마지막 문맥 프레임 특징을 M 번" 내면 w16 **88.9 (skip5 92.2)**, w8 84.4–87.2 — predictor (84.4 / 82.8–83.3) 보다 4–8 pt 높다. M=1 (w5) 에서만 복사가 chance (52–56) 이고 predictor 가 81.1 이다. **긴 창에서 이 벤치마크가 재는 것은 "다음 12 프레임이 마지막 프레임에서 얼마나 멀어지나" 이고, 그걸 predictor 보다 정지 가정이 더 잘 잰다.** V-JEPA 2 (복사 85.0 vs 릴리즈 88.9, +3.9) 보다 더 나쁜 쪽의 같은 현상.
3. **공개값 87.8 (DINO-world 논문 Table 2 의 baseline 값, §4) 은 property 별 최고 칸 규칙으로 86.1–86.7 로 재현된다** — 차이는 영속 (O1) 3–5 pt 뿐. 입력을 학습 비율 448×896 으로 늘려도 (stretch) 전체는 같다 (84.4). lowres (224) 판은 75.6 (skip2_w16 / skip2_w8). 해상도 · 위치 임베딩 보간 (64→32 / 32→16) 의 영향이 9 pt 라, 입력 분포 차이 (정방형 · 합성 · 288→448 업샘플) 가 결과를 상당히 흔든다 — 절대값은 그 단서와 같이 읽을 것.

## 표 — 쌍 정확도 % (matched 180 쌍, AvgSurprise, chance 50; O1 영속 / O2 모양 / O3 연속 각 60 쌍)

| 모델 | 창 (C=4 + M) | skip2 | skip5 | skip10 | 최고 칸 O1 / O2 / O3 |
|---|---|---:|---:|---:|---|
| **highres predictor** | w5 (M=1) | **81.1** | 78.9 | 73.3 ⁱ | 71.7 / 86.7 / 85.0 |
| | w8 (M=4) | 82.8 | 82.2 | **83.3** ⁱ | 71.7 / 85.0 / 93.3 |
| | w16 (M=12) | **84.4** | 82.2 | — | 73.3 / 91.7 / 88.3 |
| highres **복사** (마지막 문맥 프레임 × M) | w5 | 54.4 | 52.2 | 56.1 | 55.0 / 58.3 / 55.0 |
| | w8 | 84.4 | **87.2** | 82.8 | 78.3 / 95.0 / 88.3 |
| | w16 | 88.9 | **92.2** | — | 83.3 / 95.0 / 98.3 |
| lowres predictor (224) | w5 | 74.4 | 67.8 | 67.8 | 66.7 / 75.0 / 81.7 |
| | w8 | **75.6** | 75.0 | 73.3 | 63.3 / 81.7 / 81.7 |
| | w16 | **75.6** | 72.8 | — | 63.3 / 81.7 / 81.7 |
| 참고 V-JEPA 2 ViT-H | skip2_w32 (Filtered) | 88.9 | | | 85.0 / 96.7 / 85.0 |
| 참고 ViT-H 복사 (LN(z) 마지막 문맥) | 최고 칸 | 85.0 | | | |

ⁱ skip10 은 프레임 예산 9 장이라 창이 1–3 개뿐이고 동점 24–34 쌍 (0.5 처리). w16 × skip10 은 성립 안 함.

## 설정 (프로토콜과 다른 점을 전부 적는다)

- **하네스**: `intphys1_sliding` 그대로 — 창을 stride 2 로 밀고 창마다 L1 surprise, 시작점 평균 (AvgSurprise), 4 중항 안 문맥일치 2 쌍 비교, `frame_budget: official` (99 // skip 장).
- **모델 계약**: `tubelet_size 1` (프레임 = 토큰 시간 단위), S = (448/14)² = 1024, target = 프레임별 DINOv2 4 층 (2,5,8,11) concat 3072 → PCA 1152 (ckpt 의 mean/std/pca 그대로), context = 앞 4 프레임 같은 것, predictor = 4 프레임 + 마스크 프레임 → 다음 프레임 (`simple_replace`, MaskTransformer 12 층 · 1152 · 분리 attention), M 걸음은 **자기 예측을 되먹이는 unroll** (레포 `sample_unroll` 과 같음). 예측 공간 = 타깃 공간 (PCA, 이미 표준화) 이라 `surprise.target_layer_norm: false`. **surprise 절대값은 V-JEPA 와 비교 금지.**
- **문맥 길이**: 모델이 4 프레임 고정이라 A.8 의 C 스윕 ({2,4,6,8,10}×w/16) 을 못 한다. `context_mult` 를 C=4 가 되게 주고 (w5: 13 · w8: 8 · w16: 4) window 마다 실행을 나눴다. "Filtered (C 최솟값)" 은 C 가 하나라 항등.
- **입력**: IntPhys 288×288 → 448×448 (bilinear, antialias 없음 — 하네스 기본). 모델은 448×896 학습이라 **w 위치 임베딩 64 → 32 bilinear 보간** (레포 `on_load_checkpoint` 의 `high_res_adapt` 와 같은 연산). lowres 판은 224, 32 → 16.
- 정밀도: 프로토콜대로 가중치 fp32 + fp16 autocast. 4 GPU (vll6), w5 5 분 · w8 10 분 · w16 19 분.
- 오라클 검사 (`check_oracle.py`) 통과 — 새 로더가 V-JEPA 경로를 안 건드렸다 (ViT-H 88.89 재현).

## 복사 > predictor 를 어떻게 읽나

- M=1 에서 복사 surprise = |f(t) − f(t−1)| 는 가능 · 불가능 영상에서 거의 같은 크기라 chance 다. M=12 에서는 "앞으로 12 프레임이 마지막 프레임에서 얼마나 멀어지나" 가 되고, 물체가 사라지거나 모양이 바뀐 영상은 더 멀어진다 → 92 %. 즉 긴 창의 IntPhys 1 은 **변화량 검출기로 풀린다.**
- predictor 는 M=1 에서 복사를 27 pt 이기지만 (진짜 한 걸음 예측이 쓸모 있음), 걸음을 늘리면 자기 되먹임으로 예측이 흐려지면서 (`FUTURE_FROM_CONTEXT` · AR 판독의 "되먹임 붕괴" 와 같은 현상) 변화량 검출기보다 못해진다.
- 그래서 **"DINO-Foresight 가 IntPhys 1 을 84 % 푼다" 와 "그 점수는 predictor 가 낸 것이다" 는 다른 문장**이다. 후자는 이 데이터에선 거짓이다 — 정지 가정이 더 높다. V-JEPA 2 에서 복사 85.0 vs 88.9 로 본 "벤치마크가 predictor 의 몫을 거의 못 잰다" 가, 다른 encoder · 다른 predictor · 다른 학습 분포에서도 (더 세게) 재현됐다.
- 단서: 복사 기준선은 **같은 창에서** 비교한 것이고 (창 w5 에서는 predictor 가 압도), "predictor 가 쓸모없다" 가 아니라 "이 평가가 그 쓸모를 긴 창에서 못 본다" 다. 또 입력 분포 차이 (정방형 보간 · 합성) 가 predictor 쪽에 더 불리할 수 있다 (lowres 9 pt 차이가 그 민감도의 크기).


## 4. 공개값 87.8 과의 대조 (2026-10-03 추가)

**87.8 의 출처**는 DINO-Foresight 논문이 아니라 Meta 의 DINO-world ("Back to the Features", arXiv 2507.19468, 2025-07) **Table 2 / Table 9(a)** 다 — DINO-Foresight 를 baseline 으로 Garrido 프로토콜에 올린 값. 카테고리별 **영속 78.3 / 불변 90.0 / 연속 95.0** (평균 87.8). 같은 표: COSMOS-4B 99.5, V-JEPA **(v1)** ViT-L 92.2, V-JEPA (v1) ViT-H 89.4 (78.3 / 95.0 / 95.0), DINO-world ViT-B 91.3. DINO-Foresight 원 논문 (arXiv v1 · v2, 프로젝트 페이지) 에는 IntPhys 가 없다. 그들은 "sanity check 이지 benchmark 가 아니다" 라고 적었고, DINO-Foresight 의 낮은 쪽 수치를 "학습 도메인에 합성 영상이 없어서" 로 설명한다.
그들이 밝힌 DINO-Foresight 추론 제약: 고정 448×896, 문맥 4 프레임 → 다음 1 프레임, DINOv2 4 층 concat → PCA 1152 (우리 로더와 같음). 창 · skip · 집계 규칙은 안 밝혔다.

**우리 값** (A.8 격자, 쌍 정확도):

| 입력 | 보고 규칙 | O1 / O2 / O3 | 평균 |
|---|---|---|---|
| 정방형 448 (w 위치 임베딩 64→32 보간) | 최고 칸 하나 (skip2_w16) | 73.3 / 91.7 / 88.3 | 84.4 |
| 〃 | property 별 최고 칸 (Garrido 노트북 규칙) | 73.3 / 91.7 / 93.3 | 86.1 |
| **stretch 448×896** (학습 비율, 위치 임베딩 원본) | 최고 칸 하나 (skip5_w8) | 75.0 / 86.7 / 91.7 | 84.4 |
| 〃 | property 별 최고 칸 | 75.0 / 91.7 / 91.7 | 86.1 |
| 둘 합쳐 property 별 최고 | | 75.0 / 91.7 / 93.3 | 86.7 |
| **DINO-world 보고** | (규칙 미상) | 78.3 / 90.0 / 95.0 | **87.8** |

- **1–2 pt 안에서 재현된다.** 불변 · 연속은 같거나 높고, 차이는 **영속 (O1) 3–5 pt** 뿐이다. 입력 비율 (stretch) 은 O1 을 73.3 → 75.0 으로 1.7 pt 올리고 O2/O3 는 섞여서, 전체로는 차이가 없다 (84.4 = 84.4). 남는 1 pt 는 창 · skip · 집계 규칙 (그들이 안 밝힘) 안에 있다.
- stretch 에서도 **복사 기준선 (91.1–92.2) 이 predictor (83.3–84.4) 를 이긴다** (§3 결론 그대로). M=1 복사는 51–56.

| stretch | w5 (M=1) | w8 (M=4) | w16 (M=12) |
|---|---|---|---|
| predictor skip2 / 5 / 10 | **82.8** / 78.9 / 71.1 | 83.3 / **84.4** / 83.9 | **83.3** / 77.2 / — |
| 복사 skip2 / 5 / 10 | 55.6 / 51.1 / 52.8 | 86.1 / **87.8** / 82.8 | 91.1 / **92.2** / — |

- stretch 는 토큰이 프레임당 2,048 이라 24 GB 에서 `max_batch 8 · video_batch 1` 로 돌렸다 (w16 1 시간 53 분 @4 GPU). 하네스 쪽 변경은 `analysis/intphys2/model.py` 의 `_BundleSpatialOverride` (빌더가 ctx 모듈에 `spatial_tokens` 를 달면 그 수를 씀) 하나이고 오라클 검사 통과.
- 참고 — 그들 표의 V-JEPA 는 **V-JEPA 1** 이다 (우리 88.9 는 V-JEPA 2 ViT-H). 그들의 V-JEPA ViT-H 89.4 와 우리 ViT-H 88.9 가 가까운 것은 프로토콜이 대체로 맞는다는 쪽 증거.

재현: `STRETCH=1 MB=8 VB=1 bash auto_research/scripts/chain_1002_dinof_intphys1.sh` (+ `COPY=1`). IntPhys1 프레임을 vll1 로 옮겨 돌렸다 (`auto_research/_stage/intphys1_dev.tar` → `/local_datasets/world/world_analysis/`).

## 재현

```bash
# 환경 · 체크포인트: /data/hyuntak/project/2026/2027_cvpr/DINO-Foresight/z_smoke/README.md (conda env dinof 는 스모크용; 채점은 vjepa2 env 로 돈다 — lightning · sklearn 없이 로드)
# 로더: analysis/model_loaders.py build_dinof (family: dinof) · 등록: configs/protocols/models.md dinof_highres / dinof_lowres
srun -w vll6 -p debug_vll --gres=gpu:4 -c 24 --mem 120G -t 300 bash auto_research/scripts/chain_1002_dinof_intphys1.sh            # highres predictor, w5 · w8 · w16
COPY=1 bash auto_research/scripts/chain_1002_dinof_intphys1.sh                                                                    # 복사 기준선 (SET model.dinof_copy=true)
MODEL=dinof_lowres bash auto_research/scripts/chain_1002_dinof_intphys1.sh
python auto_research/scripts/dinof_intphys1_table.py                                                                              # 표 (summary.json 에서 다시 계산)
# 산출: z_research/IntPhys/exp_results/intphys1_sliding__intphys1_dev_dinof_{highres,highres_*_copy,lowres}_w{5,8,16}/{summary.json, per_window.json, _resolved.yaml}
python z_research/scripts/analysis/check_oracle.py                                                                                # 통과 확인 (10-02)
```
표의 수치는 `dinof_intphys1_table.py` 가 summary.json 에서 다시 계산한 값이다.
