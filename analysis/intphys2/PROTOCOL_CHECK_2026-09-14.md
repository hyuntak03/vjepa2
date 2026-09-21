# IntPhys 2 슬라이딩 창 채점 — 논문 · 공식 코드 · 우리 구현 대조 (2026-09-14 대조, 2026-09-19 문서화)

> 대조 대상: 논문 PDF "IntPhys 2 — Benchmarking Intuitive Physics Understanding In Complex Synthetic Environments",
> 공식 레포 `/data/hyuntak/project/2026/2027_cvpr/IntPhys2` (`prediction_evals/` 등), 우리 config `configs/analysis/intphys2/vjepa2_vith_intphys2_main.yaml`
> + `analysis/intphys2/`. 대조는 2026-09-14 세션에서 했고 이 문서는 그 기록을 옮긴 것이다 (재실행 안 함).
> 2026-09-19 에 config 를 다시 열어 "고칠 곳" 이 **아직 반영되지 않았음** (`model.dtype: bfloat16`, 84 행) 을 확인했다.

## 0. 결론

**창 · 문맥 · 집계의 골격은 논문과 공식 코드에 일치한다.** 다만 논문과 공식 코드가 서로 어긋나는 곳이 세 군데, 우리 config 에 고칠 곳이 두 군데 있다.
공식 코드는 그대로는 실행되지 않고 정확도 계산 코드는 공개되지 않았다 — "공식과 완전히 같다" 의 마지막 한 걸음은 논문 텍스트로만 확인했다.

## 1. 일치 확인

| 항목 | 논문 | 공식 코드 | 우리 | 확인 방법 |
|---|---|---|---|---|
| 프레임 샘플링 | 6 fps | `arange(0, len, 10)` @ 60 fps | `frame_step: 10` | 코드 대조 |
| 창 | M = 48, 시작점 S 간격 | `unfold(48, stride 2)` | `plan_windows` | T = 50~79 전수, 불일치 0 |
| growing context | "M 을 최대로 두고 문맥을 늘린다" | `max_context_mode`, C = 2..C−2 앞에 붙임 | `protocol: growing` | 12 개 C 전수, 불일치 0 |
| 역할 | 식 (1) f, f′ | context ← `encoder`, target ← `target_encoder` | 같음 | 코드 대조 |
| 문맥 입력 | — | 문맥 토큰만 encoder 에 넣음 | `masked` | 코드 대조 |
| 거리 | d | target 에만 LN, L1 평균 | 같음 | 코드 대조 |
| mask token | — | index 0 | 0 | 코드 대조 |
| 집계 | AvgSurprise 기본 | (계산 코드 미공개) | avg | 논문 |
| 리사이즈 | — | 512 → 256 전체 bilinear | 전체 리사이즈 (`crop_margin_ratio: 1.0`) | 수치 비교, 최대 0.5/255 (uint8 반올림) |

## 2. 공식 코드가 논문·체크포인트와 다른 곳

1. **RoPE 구현이 V-JEPA 2 체크포인트와 다르다 (가장 중요).** 공식 IntPhys 2 코드는 표준 RoPE (주파수 interleave), 우리는 V-JEPA 2 릴리즈 RoPE
   (주파수 중복; 원 레포 주석에 "고치면 체크포인트 호환이 깨진다"). 같은 입력의 회전 결과가 최대 5.9 차이. 실제 체크포인트로 v11 clip 2 개 (CPU):

   | RoPE | predictor L1 | 문맥 복사 기준선 |
   |---|---:|---:|
   | 우리 (릴리즈) | **0.589** | 0.81 |
   | 공식 IntPhys 2 코드 | 0.646 | 0.67 |

   우리 쪽이 체크포인트에 기록된 학습 손실 0.580 과 거의 같다 → **체크포인트와 맞는 것은 우리 구현이고 우리 RoPE 를 유지한다.** clip 2 개라 크기는 참고용.
   ViT-L 에서는 공식 코드의 RoPE 슬라이스 폭도 다르다 (21 vs 20). ViT-H 는 같다.
2. **공개 V-JEPA 2 config 는 그대로 실행되지 않는다.** wrapper 가 `use_rope`, `num_heads` 를 넘기는데 config 에도 같은 키가 있어 `TypeError` (재현 확인).
   데이터 로더도 `metadata.csv` 에 없는 `filename` 컬럼을 읽는다. 논문 수치는 공개 코드를 그대로 돌린 결과가 아니다.
3. **문맥 길이가 논문과 코드에서 다르다.** 논문 D.3 은 V-JEPA 2 에 C = 4, 6, …, 14, 공식 config 는 12, 18, …, 42. 우리 config 는 코드 쪽.

## 3. 우리가 고칠 곳 (2026-09-19 현재 미반영)

1. **정밀도** — 공식은 fp32 가중치 + bf16 autocast, 우리 config 는 가중치 전체 bf16 캐스팅. → `model.dtype: float32`, `model.autocast: bfloat16` (코드는 이미 지원).
2. **하이퍼파라미터 선택** — 논문 Table 2·3 은 **열마다** (Easy/Medium/Hard/Overall, 조건별) 최고 설정을 따로 보고한다. 우리는 overall 최고 C 하나를 고르고
   그 C 의 세부값을 쓴다. `summary.json` 에 C 별 세부값이 다 남으니 재실행 없이 다시 계산할 수 있다.
3. (권장) **그리드** — 논문은 M ∈ {16, 32, 48} 도 탐색 (Table 8, V-JEPA 2 최적 48). 논문 기준 비교라면 C 는 두 목록의 합집합, M 은 세 값.

## 4. 확인 못 한 것

- **짝 구성** — 우리는 `type` 앞자리 (`1_`/`2_`) 로 가능 1 + 불가능 1 을 묶는다. 메타데이터는 장면마다 네 유형이 정확히 하나씩 (253 장면).
  그 짝이 **문맥이 픽셀 단위로 같은 쌍인지**는 논문도 코드도 명시하지 않는다 (CLAUDE.md §1-2 기준으로 중요). 영상 앞부분 픽셀 비교로 확인해야 한다.
- **정확도 계산** — 동점 처리, 영상 길이가 달라 생기는 0 패딩 처리가 미공개. 우리는 동점 0.5, 패딩 없는 평균.
- **Held-Out** — 라벨 비공개, 리더보드는 영상별 0/1 제출만 받는다. 짝 정확도를 직접 계산할 수 없다.

## 5. 기존 결과에 대해

- 레포의 IntPhys 2 결과는 ViT-L 53.26 % 하나뿐이다. stride 4, fixed protocol, center crop, 단일 EMA encoder, bf16 캐스팅 — **공식 프로토콜 수치가 아니다.**
- ViT-H config 는 있지만 실행 결과가 없다. IntPhys 2 기준선은 §3 을 반영해 새로 재야 한다 (CLAUDE.md §10 순 3 "IntPhys 2 실측").

## 재현

대조는 코드 읽기 + 소규모 수치 검사였다. RoPE 비교 (v11 clip 2 개, CPU) 는 스크립트로 남기지 않았다 — 다시 하려면
`src/models/utils/modules.py` 의 `rotate_queries_or_keys` 와 공식 레포의 RoPE 를 같은 predictor 가중치에 넣고 `surprise_c16t32` 와 같은 L1 을 비교한다.
config: `configs/analysis/intphys2/vjepa2_vith_intphys2_main.yaml`. 공식 레포: `/data/hyuntak/project/2026/2027_cvpr/IntPhys2`.
