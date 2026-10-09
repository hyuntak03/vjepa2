# IntPhys 2 — 문맥 encoder (ViT-H) T×N 평균 풀링 위 linear probe: possible vs impossible (2026-09-30)

**사용자 지시**: 위반 시점을 포함하도록 32 프레임 (stride 18) 을 뽑고, 문맥 encoder 토큰을 T×N 축으로 평균한 뒤 possible / impossible 이진 linear probe.
**묻는 것**: encoder 의 표현 (한 벡터로 뭉갠 것조차) 에 위반 여부가 선형으로 남아 있는가 — 논지의 근거 1 (encoder 는 표현을 다 한다) 을 IntPhys 2 에서 확인.

## 결론 (3 줄)

1. **남아 있다, 그러나 절반만.** 장면 단위 5-fold, 문맥 encoder (`encoder`) T×N 평균 1280-d → 정확도 **62.5 %** (라벨 섞기 10 회 48.5–53.6), 쌍 정확도 **73.7 % [69.9, 77.5]**, AUC 0.63.
2. **같은 영상의 VoE (릴리즈 predictor, w48 최고 C24) 는 쌍 54.3 %.** 조건별로 probe 가 VoE 를 크게 앞선다 — immutability 84 vs 51, permanence 78 vs 52, continuity 73 vs 62. **solidity 만 probe 도 62 (정확도 51)** 로 바닥에 가깝다.
3. 단서: 이건 **지도 학습된 자** 다 (impossible 라벨을 보고 배움). VoE 는 라벨 없이 예측 오차로 판정하므로 두 수치는 같은 능력을 재지 않는다. probe 가 배우는 것은 "위반 후 장면의 통계" (물체가 사라진 채 남는 화면, 색이 바뀐 물체) 일 수 있고, 그건 predictor 가 문맥에서 **예측해야 할** 것과 다르다. 그래도 **평균 벡터 하나에 위반이 선형으로 남는다는 사실**은 "encoder 는 정보를 놓치지 않는다" 쪽의 증거고, 이 위에 예측이 얹히지 않는다는 것이 논지의 빈자리다.

## 표 — 쌍 정확도 (n 120 · 120 · 120 · 146 쌍, chance 50)

| condition | probe enc T×N 평균 | probe 정확도 (영상 단위) | VoE 릴리즈 (w48 C24) |
|---|---:|---:|---:|
| continuity | 73.3 | 61.3 | 61.7 |
| immutability | **84.2** | 71.3 | 50.8 |
| permanence | **78.3** | 68.8 | 51.7 |
| solidity | 61.6 | 51.0 | 53.4 |
| **전체 (506 쌍)** | **73.7** | **62.5** | **54.3** |

풀링 변형 (재추출 없이, 튜블릿별 공간 평균 (16, 1280) 저장본에서): target encoder T×N 평균 쌍 69.8 · 정확도 60.8 / 튜블릿 max 69.8 · 59.8 / 마지막 튜블릿만 70.6 · 62.8. **네 변형이 ±4 pt 안** — 풀링 방식이 아니라 표현의 선형 분리도가 한계다.

## 방법

- 프레임: 60 fps 원본에서 stride 18 × 32 장 (span 559 프레임 = 9.3 s / 10.6 s). start = 0, 위반 (`intphys2_onset_main_v2.json`) 뒤 표본이 2 장 미만이면 start 를 뒤로 (28 영상). 검출된 위반 시점은 **전부** 창 안 (확인 1.0). 시점 없는 49 쌍은 start 0.
- 전처리 = `analysis/intphys2` 와 동일 (Resize 256 · CenterCrop 256 no-op · ImageNet 정규화), ViT-H `encoder` (문맥) 와 `target_encoder` 둘 다 bf16 autocast 로 추출, 16 튜블릿 × 256 패치 토큰을 공간 평균해 (16, 1280) 저장 → T×N 평균은 16 개 평균.
- probe: 표준화 + L2 로지스틱 회귀 (LBFGS). **장면 (SceneIndex) 단위** 외부 5-fold (condition 층화) · 내부 4-fold 로 λ ∈ {1e-3 … 10} 선택. 쌍 정확도 = 같은 k 쌍에서 P(imp) 가 impossible 영상 쪽이 큰가. 라벨 섞기 = 쌍 단위로 라벨을 뒤집어 같은 CV.
- ⚠️ 기록: 첫 실행의 frame plan 이 SceneIndex `"154_0"` 형 문자열 15 장면 (30 쌍) 에서 위반 시점을 못 찾았다 (onset 스크립트가 `int("154_0") = 1540` 으로 저장). 그 30 쌍은 어차피 start 0 이라 **특징은 동일**하고 (수정 후 plan 도 start>0 은 같은 28 영상), 저장된 meta 의 `onset_frame` 만 −1 이다. 코드는 고쳤다.

## 이 결과로 못 하는 주장

- "encoder 가 물리를 이해한다" — 지도 자가 62 % 를 맞춘 것은 위반 후 화면 통계일 수 있다 (예: permanence 는 물체가 없어진 장면, immutability 는 색·모양이 바뀐 물체). VoE 와의 차이를 "predictor 가 못 한다" 로 옮기려면 라벨 없는 검사 (예: 문맥 encoder 표현에서 위반 전·후 거리) 가 필요하다.
- "solidity 는 encoder 도 못 본다" — 62 / 51 은 낮지만, solidity 는 위반이 이르고 (중앙값 3.9 s) transient (통과 vs 튕김 후 같은 자리) 가 있어 **T×N 평균이 지우기 쉬운 사건**이다. 평균이 아닌 자 (시간 축 보존) 로 다시 봐야 한다.
- VoE 와 프레임 배치가 다르다 (probe 9.3 s 한 창 / VoE 6 fps 8 s 슬라이딩 9 창 + growing prefix). 같은 영상이지만 같은 입력은 아니다.

## 재현

```bash
# vll6 (데이터 /data2/local_datasets/world/IntPhys2/Main), GPU 8 장 → 추출 4 분, probe CPU 6 분
srun -w vll6 -p debug_vll --gres=gpu:8 -c 32 --mem 160G -t 90 bash -c 'source /data/hyuntak/anaconda3/bin/activate vjepa2; cd /data/hyuntak/project/2026/2027_cvpr/vjepa2; \
  S=auto_research/scripts/intphys2_linear_probe.py; D=/dev/shm/ip2_probe_feats; python $S plan; \
  for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i python $S extract --shard $i --nshards 8 --out-dir $D & done; wait; \
  python $S probe --feat-dir $D --out auto_research/exp_results/intphys2_probe/linear_probe_v1.json --n-perm 10'
# 특징 저장본: auto_research/exp_results/intphys2_probe/shard*.npz (enc/tgt (n,16,1280) fp16 + meta) — 다른 풀링·자는 여기서
# VoE 대조: z_research/Benchmarks/exp_results/intphys2/bench_intphys2_main_vith/summary.json (breakdown 을 condition 으로 합산)
```
표의 수치는 `linear_probe_v1.json` 과 위 summary.json 에서 이 문서 작성 시 다시 계산했다.
