# 평가 프로토콜 — 딱 3 개다 (정본, 2026-09-21)

> 이 파일이 **설정의 단일 출처**다. 수치를 보고하거나 새 모델을 붙일 때 여기부터 본다.
> 각 줄은 **공식 코드 파일**에서 온 것이고 경로를 같이 적었다. 논문 텍스트와 어긋나는 곳은 §4 에 모았다.

세 프로토콜은 **서로 다른 논문**에서 온다. 섞으면 안 된다 — 같은 IntPhys1 dev 에서도
프로토콜이 다르면 66.67 / 83.33 / 88.89 처럼 20pt 넘게 갈린다 (CLAUDE.md §1-3).

| # | 프로토콜 | 출처 | 쓰는 데이터셋 |
|---|---|---|---|
| **P1** | **Garrido sliding** | Garrido et al., App. A.6–A.8 + 공식 코드 | IntPhys 1 dev · GRASP · InfLevel-lab |
| **P2** | **IntPhys 2** | Bordes et al., App. D + 공식 코드 | IntPhys 2 Main |
| **P3** | **ours (고정 C16/T32)** | 이 레포 | IntPhysGen v11 |

---

## P1. Garrido sliding

공식 코드: `/data/hyuntak/project/2026/2027_cvpr/jepa-intuitive-physics/evaluation_code/evals/intuitive_physics/`
우리 구현: `configs/protocols/intphys1_sliding.yaml` (`surprise.mode: intphys1`)

```
S_t = || p(f(V[t:t+C])) − g(V[t:t+C+M]) ||_1      # f=context enc, g=target enc, stride 2
시작점 t 마다 C 들의 **최솟값**  (eval.py:261  all_losses.min(1)[0])
AvgSurprise = mean_t  (쌍 비교)   /  MaxSurprise = max_t  (단일 영상)
                      (eval.py:278-279)
```

**(skip × window) 그리드는 공식 코드에 없다. 데이터셋마다 config 가 하나다** — `configs/default_*.yaml`:

| 데이터셋 | window (`frames_per_clip`) | skip (`frame_steps`) | context_lengths | 공식 config |
|---|---:|---:|---|---|
| IntPhys 1 dev | **16** | **2** | [2,4,6,8,10] | `default_intphys.yaml` |
| GRASP (level2) | **16** | **10** | [2,4,6,8,10] | `default_grasp.yaml` |
| InfLevel-lab | **32** | **5** | [4,8,12,16,20] | `default_inflevel.yaml` |

우리 프로토콜의 `context_mult: [2,4,6,8,10]` 은 `C = mult × window/16` 으로 풀리므로
w16 → [2,4,6,8,10], w32 → [4,8,12,16,20] 이 되어 **공식 값과 자동으로 일치한다.**

**해상도는 데이터셋이 아니라 모델을 따라간다.** 공식 IntPhys2 코드가 같은 데이터셋에
V-JEPA 2 는 256, VideoMAEv2 는 224 를 쓰는 것이 근거다 (§P2). 그래서:
V-JEPA 2 / 2.1 → **256**, VideoMAEv2 → **224** (patch 14 라 256 은 격자가 안 떨어진다).
⚠️ Garrido 의 `default_*.yaml` 은 셋 다 224 인데, 그건 **그들이 쓴 모델**(`vit_huge_rope`, 자체 학습)
해상도이지 릴리즈 V-JEPA 2 의 것이 아니다.

**채점**: IntPhys1·GRASP 은 쌍 비교 (`scoring.pairing: matched`),
InfLevel 은 공식 `evaluator.py` 의 2×2 (`scoring.pairing: cross`, §P1-b).

### P1-b. InfLevel 채점 (공식 `benchmark/inflevel/evaluator.py`)

| property | 가능(real) | 불가능(magic) |
|---|---|---|
| continuity | `vv`, `ii` | `iv`, `vi` |
| solidity · gravity | `ui`, `cv` | `uv`, `ci` |

`c` = cut(바닥 없는 컵·뒤가 잘린 원통), `u` = uncut, `i`/`v` = 끝에 물체가 안 보임/보임.
`(camera, cover, obj, dir)` 로 묶어 **4 종이 다 있는 그룹만** 쓰고 그 안에서 가능2 × 불가능2 = 4 비교.
불완전 그룹은 인덱스에서 이미 뺐고 **버린 수가 공식 상수와 일치한다** (continuity 36 / gravity 114 / solidity 6).
길이가 property 마다 달라 (630 / 375 / 465) **세 데이터셋으로 나눠 등록**했다.
⚠️ gravity·solidity 는 컵 상태가 본 영상 앞 contextualization 에만 나와 **원리적으로 못 푼다** (Garrido §E).

---

## P2. IntPhys 2

공식 코드: `/data/hyuntak/project/2026/2027_cvpr/IntPhys2/prediction_evals/evals/intphys2/`
우리 구현: `analysis/intphys2/` + `configs/analysis/intphys2/bench_intphys2_main_<model>.yaml`
대조 기록: `analysis/intphys2/PROTOCOL_CHECK_2026-09-14.md`

```
growing context (max_context_mode) · frame_step 10 (= 6 fps) · stride 2 · target 에만 LN · mask_index 0
AvgSurprise · 채점은 SceneIndex 안 쌍 비교
```

**모델마다 config 가 다르다** (공식 `configs/`):

| 모델 | `frames_per_clip` (M) | `context_lengths` | `resolution` | 공식 config |
|---|---:|---|---:|---|
| V-JEPA 2 / 2.1 | **48** | [12,18,24,30,36,42] | **256** | `vjepa_2.yaml` |
| VideoMAEv2-g | **16** | [4,6,8,10,12,14] | **224** | `videomaev2.yaml` |

M 도 해상도도 **모델을 따라간다.** VideoMAEv2 는 RoPE 가 없고 위치 임베딩이 16 프레임짜리
고정 sinusoid 표라 그 밖으로 나가면 외삽이 되기 때문이다.

---

## P3. ours — 고정 C16 / T32 (IntPhysGen v11)

우리 구현: `configs/protocols/surprise_c16t32.yaml`

```
raw 100 프레임 --stride 3--> 32 장 (0,3,...,93)
  문맥 = 앞 16 장 (raw 0~45)   미래 = 뒤 16 장 (raw 48~93)
surprise = mean | predictor(context) − LN(target_encoder(clip))[future] |
채점 = matched pair (block 안 문맥이 픽셀 단위로 같은 2 쌍만)
```

**VideoMAEv2 예외**: 16 프레임 사전학습이라 32 장은 위치 외삽이다. **시간 구간은 그대로 두고
밀도만 절반**으로 간다 — stride 6 으로 16 장 (0,6,…,90), 문맥 8 (raw 0~42) / 미래 8 (raw 48~90).
두 값을 모두 남긴다: 밀도 ½ **52.88** (정식) / 32 장 그대로 49.93 (외삽 각주).

---

## 4. 논문 텍스트와 공식 코드가 어긋나는 곳 (기록)

| 항목 | 논문 텍스트 | 공식 코드 | 우리 선택 |
|---|---|---|---|
| P1 그리드 | A.8 "skip ∈ [2,5,10], window ∈ [16,32] 를 탐색하고 최고를 보고" | `default_*.yaml` 데이터셋당 **한 조합** (= Table S3) | **코드** (한 조합). 그리드는 그들이 그 조합을 고른 과정으로 본다 |
| P1 context 선택 | "GRASP·InfLevel 은 property 마다 최적 C 를 고른다" | `eval.py:261` 세 데이터셋 전부 **C 최솟값** | **코드** (min) |
| P1 해상도 | 명시 없음 | `default_*.yaml` 224 (그들 모델 기준) | **모델을 따른다** (V-JEPA 256 / VideoMAE 224) |
| P2 하이퍼 선택 | Table 2·3 은 **열마다** 최고 설정 | (정확도 계산 코드 미공개) | overall 최고 C. **열별 재계산은 `per_video.csv` 로 재실행 없이 가능** — 미반영 |
| P2 M 탐색 | Table 8 에서 M ∈ {16,32,48} 탐색 | config 는 모델별 고정 (V-JEPA 48 / VideoMAE 16) | **코드** (모델별 고정) |

⚠️ **보고할 때 프로토콜 이름을 반드시 붙인다.** 같은 IntPhys1 dev 에서
P3(고정 C16) 66.67 · P1(w16, 공식) 83.33 · P1(w32, 비공식) 88.89 다.
**표에 싣는 IntPhys1 값은 공식 설정인 w16 = 83.33 이다.**
