# 문헌 조사 — latent world model 의 상태 진화 · 영속 · 예측(anticipation) · VoE 비판 (2026-09-26)

> **역할.** `LITERATURE_POSITIONING_2026-09-25.md` (이하 **P-25**) 의 후속이다. P-25 가 이미 다룬 논문은 표에서 한 줄로 줄이고
> "이번에 새로 확인한 것" 과 "그들이 재지 않은 것" 만 적는다. 새 논문은 전부 arXiv abstract 페이지(또는 HTML 본문)를 직접 열어 확인했다.
> **확인 수준 표기**: ✅ abstract/본문 직접 확인 · ◐ 검색 요약으로만 확인 (인용 전 원문 재확인) · ❓ 미검증(존재·venue 불확실).
> 2026 preprint 는 대부분 동료 심사 전이다.
>
> **우리 주장 (조사 기준).** V-JEPA 2 predictor 는 미래를 **마지막 관측에서 가져와(retrieval) 채운다**, 상태를 진화시키지 않는다.
> 세 시그니처: **reach** (마지막 관측 자리에서 2~3 칸 · 2 걸음 안에서만 물체 토큰 유지), **permanence** (가려진 내용을 옮기지 않음),
> **closure** (자기 예측이 관측을 대신하지 못함). 그리고 world-modeling 의 증거로 쓰이는 벤치마크 (IntPhys VoE, EK100 anticipation) 는
> 이 셋을 재지 않는다. 목표 벤치마크: EK100 · IntPhys 2 · Ego4D (LTA/STA) · EgoExo4D.

---

## 0. 결론 먼저

**결론.** 네 영역 모두에서 우리 세 시그니처를 **frozen 사전학습 latent predictor 에서, 학습 없는 readout 과 복사 기준선으로** 잰 선행은 없다.
가장 가까운 것은 (i) 생성형 쪽의 WRBench "tracking shot" · StEvo-Bench, (ii) latent 쪽의 HERA (동기만 같고 측정은 surprise 점수뿐),
(iii) 평가 방법론 쪽의 R2M-Bench (revisit 유사도 ≠ 기억; 우리의 copy 기준선과 논리가 같다) 다.

**근거 3줄.**
1. **rollout 을 하는 latent WM 은 전부 새 predictor 를 학습한다** (V-JEPA 2-AC, DINO-world, DINO-WM, LeWorldModel, JEPA-x, PSG-JEPA). 사전학습 V-JEPA 2 predictor 를 "그대로" world model 로 쓰는 쪽 (Garrido, IntPhys 2, EK100, TAP-JEPA/JFAA, HERA, WMReward) 은 **한 번도 open-loop 상태 운반을 재지 않았다.**
2. **object permanence 벤치마크는 2026 에 폭발했지만 전부 생성형 (pixel) 모델 대상이다** (StEvo, WRBench, MemoBench, WROP, PersistBench, R2M, LoopBench). latent predictor 는 평가 목록에 없다. IntPhys 2 가 유일한 latent 평가인데 permanence 셀이 "가장 쉬운" 셀로 보고돼 우리 결과 (가려진 내용을 옮기지 않음) 와 **정면으로 긴장**한다 — §2-0 에서 해소 방향을 적었다.
3. **anticipation 은 지평이 1 s (EK100) · 0.5 s 문맥 + TTC (Ego4D STA) 로 짧고, 긴 지평 (Ego4D LTA, 20 행동) 은 LLM 이 언어로 푼다.** latent predictor 를 쓰는 곳 (V-JEPA 2, TAP-JEPA, JFAA) 은 **one-shot 1 s 예측을 문맥 토큰에 이어 붙일 뿐이고 predictor 유무 ablation 이 없다** (V-JEPA 2 Table 20 만 +0.6).

**미결 1줄.** IntPhys 2 permanence 가 "쉽다" 는 보고와 우리 permanence 실패를 **같은 프로토콜 (sliding, Filtered) 에서 셀 단위로** 대조해야 한다. 이게 없으면 리뷰어는 "IntPhys 2 에서 permanence 는 잘 되는데?" 로 끝낸다.

---

## 1. frozen / 사전학습 video encoder 위의 latent world model — 어떻게 굴리고, 무엇을 재지 않았나

### 1-1. 요약표

| 논문 | 확인 | encoder / predictor | rollout 방식 (학습) | rollout 방식 (평가) | open-loop 상태 운반 측정? |
|---|---|---|---|---|---|
| V-JEPA 2 (릴리즈 predictor) | ✅ | frozen ViT-g / 릴리즈 predictor | tube-mask 복원 (future-only 학습 없음; Garrido 도 인정) | **one-shot** mask-token 질의 (VoE, EK100 1 s) | ❌ |
| V-JEPA 2-AC | ✅ | frozen ViT-g, **새** 300M block-causal predictor | teacher forcing T=15 + rollout loss T=2 | closed-loop MPC (receding horizon, 첫 행동만 실행) | ❌ (error accumulation 을 한계로만 서술) |
| V-JEPA 2.1 | ✅ | dense loss + deep self-supervision. **predictor/rollout 변경 없음** (abstract 기준) | 동일 | EK100 40.8 R@5, Ego4D STA 7.71 mAP | ❌ |
| DINO-world | ✅ | frozen DINOv2, 새 cross-attn predictor, 절대 timestamp RoPE | teacher forcing next-frame, Δt 균등 표본 | direct(one-shot) vs autoregressive 둘 다; **"1 s 근처에서 전부 부정확"** | ◐ (정확도 곡선만, 상태 내용은 안 봄) |
| DINO-WM | (P-25) | frozen DINOv2, ViT predictor | 1-step | CEM planning (closed-loop) | ❌ |
| LeWorldModel | ✅ | end-to-end 15M, next-embedding loss + Gaussian reg | 1-step | planning + **surprise 로 물리 위반 탐지** + probing | ❌ (abstract 기준) |
| HERA | ✅ | **frozen V-JEPA 2 predictor** + 3.00M adapter (memory/workspace registers, block 3·7 뒤 gated cross-attn) | Physion 14k, V-JEPA 2 와 같은 future-latent loss | IntPhys 2 sliding surprise: Main 52.57 → 54.35 | ❌ (본문 확인: 가려진 물체 readout 없음) |
| JEPA-x | ✅ | 자체 (frozen 사전학습 encoder 언급 없음) | 시각+물리 상태 cross-prediction | **새로 맞춘 predictor 의 rollout drift 0.361 → 0.104** | ◐ (drift 수치는 있으나 "상태 운반" 조작화 없음) |
| PSG-JEPA ("Is Forward Prediction Enough?") | ✅ | 자체 (proprio 로 grounding) | forward + 두 grounding 손실 | 3 층위 (probe / planning / policy); 검색 요약: 3 문맥 프레임에서 **teacher forcing 없이** 재귀 예측 | ◐ (로봇 상태 한정) |
| RDR (Rollout-Decoded Reconstruction) | ✅ | 소형 (193k) | **free-run rollout 을 학습 중에 그대로 돌리고 모든 latent 를 디코딩해 벌점** | KS 방정식 valid time 3.87 → 6.97 | ◐ (decoder 분포 불일치 진단이 우리 closure 와 같은 구조) |
| Semigroup-JEPA · UWM-JEPA · Hierarchical Planning | (P-25 §2-5·2-6) | — | — | — | — |
| Cosmos (Predict1/2/2.5, Cosmos 3) | ✅◐ | 생성형 (diffusion/AR, 픽셀) | — | Physics-IQ 류 (Cosmos3-Super 43.8/59.7) ; IntPhys 2 에서 Cosmos-Predict-4B 는 chance 근처 | ❌ |
| Genie 3 (blog) / Genie 2 | ◐ | 생성형 interactive | — | "약 1 분 시각 기억", permanence 는 데모 | ❌ (논문 없음) |
| JEPA-WAM · Video Generation with Predictive Latents | ✅ | V-JEPA 2.1 은 **goal encoder** / PV-VAE 는 JEPA 무관 | — | 로봇 성공률 / 생성 품질 | ❌ (우리 질문과 무관 — 인용 불필요) |
| "The JEPA Predictor: transferable operator for occluded feature completion" | ✅ | frozen I-JEPA/V-JEPA 2 predictor + 선형 사상 | — | **공간** 마스크 복원 | **철회됨 (2026-07-21, "major mistakes")** — 인용 금지 |

### 1-2. 항목별

**V-JEPA 2 / V-JEPA 2-AC** — Assran et al., arXiv:2506.09985, 2025. ✅
- 주장: frozen encoder + 릴리즈 predictor 로 이해·예측 (EK100 τ_a = 1 s, R@5 39.7 ViT-g384) ; 2-AC 는 **새** predictor (24층, 1024, block-causal, 프레임별 독립 인코딩) 를 teacher forcing T=15 + rollout T=2 로 학습하고 MPC 로 계획.
- 관련: **must-cite**. 우리 검증 대상. 한계 절 "autoregressive prediction suffers from error accumulation" 은 우리 closure 결과의 저자 측 인정.
- 안 잰 것: 릴리즈 predictor 의 open-loop 운반. EK100 에서 predictor 기여는 Table 20 (레포 전사) encoder 39.1 / 둘 39.7 / predictor 만 20.2 → **+0.6**. planning 은 사전학습 predictor 를 쓰지 않는다 (P-25 §0 결정 1).

**V-JEPA 2.1** — Mur-Labadia, Muckley, Bar, Assran, Sinha, Rabbat, LeCun, Ballas, Bardes. arXiv:2603.14482, 2026-03 (v 2026-06). ✅
- 주장: dense predictive loss (visible 토큰에도 손실) + deep self-supervision + 모달별 tokenizer. EK100 40.8 R@5, Ego4D STA 7.71 mAP, 로봇 grasp +20pt vs 2-AC.
- 관련: **must-cite** (리뷰어 첫 질문 "2.1 에서도 그런가"). predictor 의 역할·rollout 은 바뀌지 않았으므로 우리 시그니처가 유지될 가능성이 크지만 **미측정**.
- 안 잰 것: 위와 같음. dense loss 가 "visible 토큰 자기 복원" 을 강화하므로 retrieval 편향이 **더 커질** 수도 있다 — 가설, 미검증.

**DINO-world (Back to the Features)** — Baldassarre et al. (Meta), arXiv:2507.19468, 2025. ✅
- 주장: frozen DINOv2 + causal cross-attn predictor, 절대 timestamp RoPE, Δt 균등 표본. direct 예측은 짧은 Δt 에서, autoregressive 는 길게 버티지만 **"all predictions become inaccurate as the forecasting interval approaches 1 second"**. IntPhys 91.3 / GRASP 76.0 / InfLevel 63.7 (V-JEPA ViT-H 89.4 / 73.0 / 59.9) 를 **"sanity check rather than a benchmark"** 로 부른다.
- 관련: **support + must-cite**. future-only 로 학습해도 1 s 에서 무너진다 → 새 predictor 가 자명한 해결이 아니다. VoE 점수를 sanity check 로 격하한 문장은 우리 평가 감사의 인용구.
- 안 잰 것: 1 s 너머에서 **무엇이** 남는지 (물체가 어디로 가는지, 복사인지 감쇠인지). 정확도 곡선만 있다.

**HERA** — arXiv:2608.05523, 2026-08. ✅ (본문 확인)
- 주장: frozen V-JEPA 2 predictor 에 3.00M adapter. 문맥 encoder 패치 토큰을 anchor/middle/recent/global 4 군으로 저장, block 3·7 뒤 gated cross-attn 으로 라우팅. IntPhys 2 Main 52.57 → 54.35, fixed-cam continuity 46.15 → 57.69, immutability 46.15 → 63.46.
- 관련: **must-cite, 가장 가까운 latent 쪽 선행**. 동기 문장 ("later predictions may depend on object evidence no longer available in the current view") 이 우리 permanence 와 같다.
- 안 잰 것: 가려진 물체의 **위치 readout, 복사 기준선, open-loop rollout**. surprise 점수만 잰다. 그래서 "+1.78 이 상태 운반 때문인가, surprise 분포 이동 때문인가" 를 그들도 모른다 — 우리 시그니처를 HERA 에 걸면 답이 나온다 (우리 논문의 자연스러운 응용).

**LeWorldModel** — Maes, Le Lidec, Scieur, LeCun, Balestriero. arXiv:2603.19312, 2026-03. ✅
- 주장: 픽셀에서 end-to-end 로 안정 학습되는 JEPA (손실 2 항, 15M, 1 GPU). planning 48× 빠름. **surprise 로 물리 위반 탐지**, probing 으로 물리량 확인.
- 관련: **must-cite** (LeCun 진영의 최신 "world model" 정의 = surprise + probing). 우리 주장은 바로 그 두 증거가 **상태 진화를 인증하지 못한다**는 것이다.
- 안 잰 것: open-loop 운반, 복사 기준선. (abstract 기준; 본문 미확인)

**JEPA-x** — Wen, Li, Luo, Shi. arXiv:2608.24044, 2026-08. ✅ / **PSG-JEPA** — Yan et al. (15인), arXiv:2608.06799, 2026-08. ✅ / **RDR** — Shah, Shrestha, arXiv:2608.25017, 2026-08. ✅
- 주장: 셋 다 "forward prediction 만으로는 latent 가 굴러가지 않는다" 는 문제의식. JEPA-x 는 새 predictor 의 **rollout drift** 를 지표로 쓴다 (0.361 → 0.104). PSG-JEPA 는 proprio grounding. RDR 은 **학습 중 free-run rollout + 디코딩 벌점** (decoder 가 관측 anchored latent 만 봤다는 분포 불일치 진단).
- 관련: **support** (설계 처방 후보 · "closure" 의 동시대 표현). 전부 소형·자체 encoder·로봇/PDE 라 우리와 직접 경쟁하지 않는다.
- 안 잰 것: 사전학습 video predictor. 가림.

**Cosmos** — NVIDIA, arXiv:2501.03575 (Predict1) · 2511.00062 (Predict2.5) · Cosmos 3 (2026-06, 2606.02800 ◐). ✅◐
- 주장: 생성형 WFM 플랫폼. Physics-IQ 계열로 평가. IntPhys 2 에서 Cosmos-Predict-4B 는 chance 근처 (IntPhys 2 Table).
- 관련: 배경. "픽셀 생성 = world model" 진영의 대표. 우리 주장은 latent 에 한정하므로 대조군으로만.
- 안 잰 것: 관측 차단 하 상태 진화 (StEvo/WRBench 가 대신 쟀다).

**Genie 2/3** — DeepMind blog (2024/2025). ◐ 논문 없음.
- 주장: "emergent object permanence", 약 1 분 시각 기억. 관련: 배경 인용만. 안 잰 것: 전부 (정량 없음).

---

## 2. object permanence / 가림 — video world model 과 self-supervised video model

### 2-0. IntPhys 2 permanence 결과와의 긴장 (먼저 정리)

IntPhys 2 (Bordes et al., arXiv:2506.09849, NeurIPS 2025 D&B reject) ✅: V-JEPA 2 overall 57.51 / held-out 56.40 (best prediction) ; Gemini 2.5 Flash 55.63. **"Permanence was the easiest for both models and humans"**, fixed-cam permanence 에서 일부 모델 60% 이상. 원인 지목은 "stricter requirements on short term memory".
우리는 "가려진 내용을 옮기지 않는다" 고 주장한다. 두 결과가 양립하려면 다음 중 하나가 필요하고 **셋 다 아직 안 쟀다**:
- (a) IntPhys 2 permanence 위반은 **재등장 시점의 관측 불일치** (물체가 나와야 할 때 안 나옴) 로 잡히고, 이는 마지막 관측 retrieval 로도 surprise 가 난다 (사건이 문맥 경계와 겹치지 않으면). 즉 "쉬운 permanence" 는 상태 운반 없이도 풀린다.
- (b) 우리 v11 permanence 실패는 사건이 **문맥 경계에 걸리는** corner case (CLAUDE.md 2026-09-03) 이고 IntPhys 2 sliding 은 그 경계를 지나간다.
- (c) 60% 는 chance 50 에서 10pt 라 "쉽다" 의 근거로 약하다.
→ **해야 할 것**: IntPhys 2 permanence 셀을 우리 프로토콜 + 복사 기준선 + 위치 readout 으로 재채점. 미결 1줄과 같다.

### 2-1. 요약표 (2025–26 신규 위주)

| 논문 | 확인 | 대상 모델 | 무엇을 재나 | 핵심 결과 | 우리와 |
|---|---|---|---|---|---|
| StEvo-Bench — Ma, Liufu, Gkioxari, arXiv:2603.13215 (ECCV 2026, P-25) | ✅ | 생성형 (Veo 3, Sora 2, Genie 3 — P-25 보고, 이번 abstract 에는 미기재) | 가림막 삽입 · 소등 · lookaway 로 관측 차단 → 진화 지속 여부 | "decoupling state evolution from observation" 실패 | **must-cite**. latent 미평가 |
| WRBench — Lu et al., arXiv:2606.20545, 2026-06 | ✅ | 23 모델, 9,600 비디오, 4 control paradigm | 미관측 구간 뒤 돌아온 대상의 상태 | **"tracking shot"**: 떠날 때 상태로 재개. 규모와 무관 | **must-cite**. 우리 "retrieval" 의 생성형 쌍둥이 |
| MemoBench — Chen … Yuille, Liang, Du. arXiv:2606.27537, 2026-06 | ✅ | 생성형 8 종 | 물체가 과정 중 사라졌다 **갱신된 상태로** 재등장해야 함, 360 clip, 4 pillar | 기억 일관성 실패 (수치 abstract 미기재) | support. "지속 ≠ 복사" 조작화의 선례 |
| PersistBench ("Can 4D Foundation Models Remember?") — He, Averbuch-Elor, Ma. arXiv:2609.20819, 2026-09 | ✅ | camera-controllable video / 4D 재구성 | 360° GT 로 permanence · motion continuity · appearance | "**seeing is not remembering**": 시야 밖으로 나가면 급격 저하 | support (문장 차용 가능) |
| WROP ("Training Object Permanence in World Models") — Zhang et al. (31인), arXiv:2609.28654, 2026-09 | ✅ | 생성형 14 종 + 자체 16B PWM-WROP | 150 Blender 생성기, 6 범주, 1.5M 학습 샘플, 300 문항 Elo | 학습으로 permanence 가 오른다 (3위 → continuation 1위) | 대조. 우리는 학습 없이 잰다 |
| R2M-Bench — Gu et al., arXiv:2608.27328, 2026-08 | ✅ | interactive 생성형 7 종 | revisit 일관성을 **gap-matched non-revisit · short-range 대조** 로 상대화 (MemoryGain, NMR) | 절대 유사도는 motion 과 ρ 0.207 상관 (교락) → NMR 0.072 | **support, 방법론 쌍둥이**: "돌아온 프레임이 닮았다 ≠ 기억했다" = 우리 "surprise 가 났다 ≠ 상태를 굴렸다" |
| LoopBench / WorldTrace ("Addressable Memory") — Wu … Ošep, arXiv:2608.07408, 2026-08 | ✅ | 생성형 | RoPE offset 이 학습 범위를 벗어나면 기억을 못 찾음 → 가상 위치 | +15.5 / +19.5 | 기술적 근거 (우리 H4 RoPE 비상대성과 같은 결) |
| Permanence Fields (Liu, Chen) arXiv:2606.28455 | ✅ | GRU / Transformer-lite / RSSM-lite (자체) | 가림 사건에서 permanence 구조 | "occlusion increases OP-related structure", specificity mixed | 약함. passive object-state WM 용어의 출처로만 |
| Punzo et al. "How Do Video Foundation Models Encode Intuitive Physics?" arXiv:2606.09646 (v2 2026-09) | ✅ | V-JEPA / 2 / 2.1, VideoMAE(-v2), LTX-Video **encoder 만** | IntPhys 2 · MVP 층별 probe; 셔플·단일프레임 반복·랜덤 라벨 대조 | IntPhys 2 linear 50.98 (chance) / attentive **VideoMAE 73.9** | encoder 에 정보는 있다 (우리 "읽기 ✅"). predictor 미조사 |
| Neural Voxel Dynamics — Wang, Mitra, arXiv:2606.26410 | ✅ | V-JEPA 특징을 voxel 로 unproject | 3D advection 으로 permanence | CLEVRER/PhysGaia | 설계 대안 (3D state) |
| GOT-JEPA — Chen et al., TCSVT 2026, arXiv:2602.14771 | ✅ | JEPA 식 추적기 (V-JEPA 2 사용 여부 미확인 ❓) | 가림 하 추적, OccuSolver | 7 벤치마크 | 지도 추적 계보 (P-25 §2-7) |
| SR-JEPA, TSA, Loci-Looped, OPNet/PermaTrack/TCOW | (P-25 §2-7) | — | — | — | — |
| "The JEPA Predictor … Occluded Feature Completion" | ✅ | — | — | **철회** | 인용 금지 |

### 2-2. 이 영역에서 우리만 재는 것
- **latent predictor 의 permanence.** 위 벤치마크 전부 픽셀 생성 모델이다. latent 쪽은 IntPhys 2 surprise 셀 하나뿐이고, HERA 가 adapter 로 올렸지만 무엇이 올랐는지 모른다.
- **복사 기준선과 위치 귀무.** R2M-Bench 만 같은 논리를 (생성형에서) 세웠다. VoE 에는 아무도 안 붙였다 (Garrido 본문 확인: copy-last 없음).
- **문맥 경계와 사건의 상대 위치** 를 축으로 둔 것 (v11 sym_k, early/mid/late). StEvo 의 "occluder insertion" 이 가장 가깝지만 latent 가 아니다.

### 2-3. PLM (PerceptionLM) 에 대해
PerceptionLM (Cho et al., Meta 2025) ✅존재 확인. **IntPhys 2 평가 목록에 없다** (Gemini 1.5 Pro / 2.5 Flash, GPT-4o, Qwen-VL 2.5 만). MVP 에 PLM 이 포함됐는지는 ❓ 미확인. 우리 논문에서 PLM 을 언급할 이유는 "MLLM 도 chance" 라는 IntPhys 2 결과를 인용할 때뿐이고, 그때는 IntPhys 2 가 실제로 평가한 모델명을 쓴다.

---

## 3. long-horizon action anticipation — 지평 · 방법 · latent predictor 사용 여부

### 3-1. 과제별 지평 (검증된 정의)

| 벤치마크 | 관측 | 예측 | 지평 | 지표 | latent predictor 사용 SOTA |
|---|---|---|---|---|---|
| **EK100 anticipation** | 보통 2.5–4 s (JFAA/TAP: 32 frame @ 8 fps = 4 s) | 다음 행동 1 개 (verb/noun/action) | **τ_a = 1 s 고정** | mean top-5 recall | V-JEPA 2 (39.7 g384) → V-JEPA 2.1 (40.8) → JFAA 1위 (27.95 test action, 2026 challenge 규칙) / TAP-JEPA 2위 (27.91) |
| **Ego4D STA** | 0.5 s (FROST-STA: 8 frame) + 마지막 고해상 프레임 | next active object box · noun · verb · **time-to-contact** | TTC 오차 < 0.25 s | top-5 mAP | VISTA 1위 (frozen V-JEPA 2.1 **encoder** 만) / FROST-STA 2위 (5.13, **predictor 미사용**) / V-JEPA 2.1 7.71 |
| **Ego4D LTA** | 8 segment (약 5 분) | 다음 **Z = 20** 행동 순서, K = 5 후보 | 수 분 | edit distance | 2025 1위 (Chu et al., arXiv:2506.02550): EgoVideo-V encoder → Transformer 인식 → **fine-tuned LLM** 이 순서 생성. latent rollout 없음 |
| **EgoExo4D** | — | 공식 anticipation 트랙 **없음** (2026 challenge: Ego-Pose Body, Procedure Understanding) ✅ | — | — | TrajPilot (Jun, Nguyen-Truong, Seminara, Torresani, arXiv:2605.20388) 가 keystep 주석으로 next-step 과제를 **파생** ; 카메라 궤적을 중간 latent 로 씀 |

### 3-2. 항목별

**JFAA** — Chu, Zhang, Feng, Liu, Guan, Jiang, Nie. arXiv:2605.20904, 2026-05 (EgoVis 2026 EK100 1위). ✅ 본문 확인
- 주장: frozen V-JEPA 2.1 ViT-G/384 encoder + predictor 로 문맥 토큰 + **near-future latent 토큰** 을 뽑고 attentive probe. epoch 별 field-aware ensemble.
- 관련: **must-cite** (사용처). predictor 를 world model 로 쓰는 현재 관행의 최신 예.
- 안 잰 것: **encoder-only vs +predictor ablation 없음** (Table 1 은 epoch 별 val 만). 미래 토큰이 one-shot 인지, 몇 프레임인지 미기재.

**TAP-JEPA** — Wang, Xu. arXiv:2606.00662, 2026-05 (2위, 27.91 / unseen 30.93 / tail 21.60). ✅ 본문 확인
- 주장: "learnable mask tokens at the future target position" — **one-shot** 질의 (autoregressive 아님). 4 s 문맥, 1 s 지평.
- 관련: must-cite (사용처). 안 잰 것: predictor 제거 ablation 없음 (본문 확인).

**VISTA** (arXiv:2605.20901, STA 1위) ◐ / **FROST-STA** (arXiv:2606.00694, 2위, 5.13 mAP) ✅
- 주장: 둘 다 frozen V-JEPA 2.1 **encoder** 특징 + 검출기/StillFast 류. FROST-STA 는 predictor 를 **쓰지 않는다** (본문 확인). 관련: STA 에서는 predictor 가 SOTA 에 필요하지 않다는 방증 = "encoder 충분" 쪽.
- 안 잰 것: 미래 latent 자체.

**Ego4D LTA 2025 1위** — Chu et al., arXiv:2506.02550. ✅ / **AntGPT** (Zhao et al., ICLR 2024 ◐) / **PlausiVL** (Mittal et al., CVPR 2024 ◐) / **Palm** (2023 challenge, Llama2-7B ◐)
- 주장: LTA 는 2023 이후 **LLM 이 언어 공간에서 순서를 생성**하는 방식이 표준. 시각 encoder 는 인식용.
- 관련: 배경. "긴 지평 = latent 상태 진화" 가 아니라 "긴 지평 = 언어 계획" 으로 풀리고 있다. 우리 논문이 Ego4D LTA 를 **목표** 로 두려면 이 사실을 먼저 적어야 한다 (latent predictor 가 거기서 쓰인 적이 없다).
- 안 잰 것: 시각 latent 의 운반. 애초에 안 쓴다.

**TrajPilot** — arXiv:2605.20388, 2026-05. ✅
- 주장: 미래 카메라 궤적을 먼저 예측하고 (중간 latent), 그것으로 행동 순서를 조건화. Ego-Exo4D atomic/Keystep, Ego4D GoalStep, EgoPER, EK100. "advantage widens with horizon".
- 관련: support (**지평이 길수록 명시적 중간 상태가 이긴다** 는 방증). EgoExo4D 에 anticipation 을 걸 때 유일한 선례.
- 안 잰 것: video latent predictor 와의 비교.

**Human-JEPA** — Wei, Sun, Zhao. arXiv:2608.21160, 2026-08. ✅ (abstract)
- 주장: "pure past-to-future split" 로 anchored forecasting 학습. 관련: future-only 학습 JEPA 의 예 (P-25 §1-7 "사전학습은 future-only 를 배우지 않았다" 의 반례 후보). 벤치마크·지평 미확인 ❓.

**AVT** (Girdhar & Grauman, ICCV 2021) ◐ 기억 기반
- 주장: causal transformer 가 미래 특징을 **재귀적으로** 예측하며 (feature prediction loss) 1 s 앞 행동을 맞춤. 관련: EK100 에서 latent rollout 을 쓴 **유일한 고전 선례**. 인용 전 재확인.

### 3-3. 이 영역에서 우리만 재는 것
- anticipation 에서 predictor 미래 토큰의 **기여를 분리** (V-JEPA 2 Table 20 +0.6 이 유일; 2.1/JFAA/TAP 은 없음).
- 지평을 1 s 너머로 늘렸을 때 latent 예측이 **복사·등속 외삽 대비** 무엇을 더 담는지. 모든 EK100 방법은 τ_a = 1 s 고정이고 지평 스윕은 2022 이전 (2 s, 1.75 s …) 에만 있다.
- **EK100 은 예측 능력을 거의 요구하지 않는다** 는 V-JEPA 2 저자 문장 ("mostly requires strong semantic understanding") — 이것과 predictor +0.6 을 묶으면 "anticipation 벤치마크는 상태 진화를 재지 않는다" 가 선다. 이 논지는 우리가 새로 세우는 게 아니라 **저자 문장의 귀결** 로 쓴다.

---

## 4. "VoE / surprise 는 시뮬레이션을 함의하지 않는다" — 비판과 기준선

### 4-1. 요약표

| 논문 | 확인 | 무엇을 보였나 | 우리와 |
|---|---|---|---|
| Garrido et al. 2025 (arXiv:2502.11831) | ✅ 본문 | 대조군 = 무작위 초기화 20 seed, VideoMAEv2, Qwen2-VL-7B, Gemini 1.5 Pro. **copy-last 없음.** "V-JEPA is never trained using a causal prediction task". 기억 3–4 s. IntPhys 98 / GRASP 66 / InfLevel 62 | **must-cite**. 기준선 공백이 우리 H6 의 자리 |
| IntPhys 2 (2506.09849) | ✅ 본문 | 모든 모델 chance 근처 (≤ 57.5), MLLM 은 프레임 수 늘리면 **하락**. 원인 "short term memory" 지목만 | must-cite. §2-0 |
| MVP — Krojer et al., TMLR 2025 (2506.09987) | ✅ | minimal video pair 로 shortcut 차단. SOTA 40.2 vs 인간 92.9 (chance 25) | 방법론 지지: **짝 맞춤 (matched pair)** 이 shortcut 방어라는 Meta 자체 주장. 우리 matched pairing 의 인용 근거 |
| Punzo et al. (2606.09646) | ✅ | "success … may reflect generic motion sensitivity, semantic recognition, or dataset biases"; 셔플·단일프레임 대조 | support. 대조군 설계 차용 |
| R2M-Bench (2608.27328) | ✅ | 유사도 ≠ 기억, 상대화 지표 필요 | support (생성형 판 copy 기준선) |
| "Asleep at the Wheel" (2608.01336, P-25) | ◐ | surprise 가 novelty 를 잡는 건 도메인 이동; 단일 도메인 안에서 chance | support |
| DINO-world (2507.19468) | ✅ | VoE 세트를 "sanity check rather than a benchmark" 로 취급 | support (인용구) |
| Vafa et al. NeurIPS 2024 / "What Has a Foundation Model Found?" ICML 2025 / Kang "How Far…" ICML 2025 / Physics-IQ (P-25 §2-9) | — | 예측 정확 ≠ 상태 모형; case-based retrieval | 논리 구조 선례 |
| "How Should World Models Be Evaluated for Embodied Decision-Making?" — Yu et al., arXiv:2606.15032, 2026-06 | ✅ | "**claim/evidence mismatch**"; 시각 그럴듯함이 아니라 interventional fidelity · closed-loop validity 등 L0–L7 계층 | support (평가 감사 톤의 선례) |
| WorldTest / AutumnBench — Warrier … Tenenbaum, Ellis, Tavares. arXiv:2510.19788 (2025-10, v 2026-05) | ✅ | 평가가 "observed trajectories" 로 잴 수 있는 것 (next-frame, return) 에 갇혀 있다; 환경 수준 질의로 바꿔야. 인간 517 명 ≫ 5 frontier 모델 | support (같은 비판, 다른 도메인) |
| X-VoE — Dai et al., ICCV 2023 (2308.10441) | ✅ | VoE 를 predictive / hypothetical / explicative 로 확장. **설명 능력** 을 요구 | 배경. "surprise 만으로는 부족" 의 CV 쪽 선례 |
| Schulze Buschoff, Schulz, Binz — ICML 2023 (2310.19943) | ✅ | 생성 모델이 물리 과정을 잘 예측해도 학습 궤적이 아동 발달과 다름 | 약한 배경 |
| Opinion: Su, Legris, Gureckis, Ren (2512.06232) | ✅ | SAYCam 으로 V-JEPA 사전학습해도 IntPhys 2 안 오름 → 데이터만으로 부족 | 배경 (architecture 쪽 원인 지목). 우리 "목적함수를 원인으로 걸지 않는다" 원칙과 맞춰 discussion 한 문단 |
| InfLevel (Weihs et al. 2022) · GRASP (Jassim et al. 2023) · Physion++ (Tung et al. 2023) · ADEPT/PLATO | (P-25) | — | — |

### 4-2. 정리
- **명시적 "copy baseline on VoE" 논문은 없다** (검색 4 회, 못 찾음). 특징 예측 쪽 Copy-Last (DINO-Foresight), Nayebi 'No Dynamics', 생성형 R2M 의 gap-matched 대조가 방법론적 선례다. 우리 H6 (IntPhys 1 복사 85.00 vs p 88.89, 셀에 따라 +3.9 n.s. / +12.2 유의 — P-25 §3-1) 은 이 공백을 메우되 **프로토콜 민감도로만** 쓴다.
- Meta 자체 문헌 셋 (MVP 의 shortcut 논지, DINO-world 의 "sanity check", V-JEPA 2 의 "mostly semantic") 을 나란히 놓으면 "VoE·EK100 점수 = world model 증거" 라는 등식은 **저자들 스스로도 약하게 본다** 는 문장이 된다. 반박이 아니라 인용으로 세울 것 (P-25 §1-6 톤 권고).

---

## (a) 리뷰어가 가장 기대할 5 편 (인용·대조)

1. **V-JEPA 2 / 2-AC** (2506.09985) — 검증 대상 + "planning 은 새 predictor" 사실. 2.1 (2603.14482) 을 같은 항목에 묶어 "2.1 에서도 유지되는가" 를 최소 한 셀은 잰다.
2. **Garrido et al. 2025** (2502.11831) + **IntPhys 2** (2506.09849) — VoE 점수의 출처. 복사 기준선 부재, permanence "쉬움" 과의 대조 (§2-0).
3. **DINO-world** (2507.19468) — frozen encoder + future-only predictor 도 1 s 에서 무너짐; VoE 를 sanity check 로. "새 predictor 가 답" 의 반례이자 설계 수치.
4. **WRBench** (2606.20545) + **StEvo-Bench** (2603.13215) — "관측 없이 진화" 개념의 선점. 우리는 latent 판 + 모듈 국소화 + 복사 대조.
5. **HERA** (2608.05523) — frozen V-JEPA 2 predictor 위 기억 adapter. 우리 시그니처를 HERA 에 걸어 "무엇이 올랐나" 를 보이면 대조가 기여로 바뀐다.

(보조: LeWorldModel — surprise+probing 을 world model 증거로 쓰는 최신 사례 ; R2M-Bench — 상대화 지표의 논리 쌍둥이 ; MVP — matched pair 의 근거.)

## (b) "retrieval, not state" 에 가장 가까운 선행과 차이

| 선행 | 그들의 발견 | 우리와 다른 점 |
|---|---|---|
| **WRBench "tracking shot"** (2606.20545) | 생성형 23 종이 돌아온 대상을 **떠날 때 상태로 재개**. 규모 무관 | 픽셀 출력의 행동 관찰. 우리는 **latent predictor 내부** 에서 (i) 토큰이 마지막 관측 자리 2–3 칸 안에서만 유지 (reach), (ii) 자기 예측 되먹임이 2 걸음 뒤 멈춤 (closure), (iii) 문맥 마지막 1–2 튜블릿이 미래 물체 유무를 결정 (M2, 1차) 을 **학습 없는 readout + 복사·등속 귀무 + attention 질량** 으로 잰다. 그리고 그 predictor 가 실제로 world model 로 쓰이는 벤치마크 (VoE, EK100) 가 이 실패를 못 본다는 것까지 잇는다 |
| **Kang et al. "How Far is Video Generation from World Model"** (ICML 2025) | case-based generalization: 학습 사례 검색, 속성 우선순위 color > size > velocity > shape | 그들의 retrieval 은 **학습 데이터** 검색. 우리는 **문맥(마지막 관측) 조회**. 다른 층위 |
| **Asleep at the Wheel** (2608.01336) | surprise 는 도메인 이동을 잡을 뿐; 표현은 충분 | 같은 "encoder 충분 · 점수 병목" 논지. 운전 novelty 이고 상태 진화·가림 없음 |
| **TSA** (2606.13714, P-25) | 가려진 slot 을 이전 상태에 anchor (설계) | 우리 발견의 **의도된 설계판**. "지속 = 복사" 대조군 |
| **Physion / Nayebi 2023** (P-25) | 시뮬레이션 readout ≠ 관측 readout 개선 (Physion) / No Dynamics 기준선 (Nayebi) | 다른 encoder·과제. 기준선 설계의 계보 |

**한 문장.** 생성형에서 "tracking shot" 으로 관찰된 실패를, 우리는 frozen latent predictor 의 **기제 수준 시그니처 셋** 으로 바꾸고, 그 predictor 를 world model 로 인증해 온 두 벤치마크가 왜 그것을 못 보는지 (one-shot 1 s 질의, 복사 기준선 부재, 사건–경계 정렬) 까지 닫는다.

## (b2) Sobal et al. 2022, "JEPA Focus on Slow Features" (NeurIPS 2022 workshop 판; 2026-09-27 사용자 제공, 읽음)

**무엇을 보였나.** 28×28 점 하나 + action 으로 움직이는 toy. encoder (CNN) + GRU predictor 를 VICReg / SimCLR 목적으로 학습하고 **action 을 주며 17 스텝 autoregressive rollout** 한 뒤 점 위치를 linear probe. 배경 잡음이 **프레임마다 바뀌면** JEPA 가 reconstruction 과 같거나 낫고, 배경 잡음이 **에피소드 안에서 고정이면** 무너진다 (α ≥ 1 에서 Center 기준선 수준). 3-dots: VICReg / SimCLR 는 **정지 점만** 잡고 움직이는 두 점을 버린다. 이유 (§2-1): 시간에 따라 안 변하는 잡음을 그대로 encoding 하면 예측 손실 0 · variance/covariance 는 에피소드 간 잡음 차이로 충족 → **손실이 0 인 자명해가 slow feature 다.**

**우리와 겹치는 것.**
- "예측하기 쉬운 것 = 안 변하는 것" 이 손실의 최저점이라는 관찰. 우리의 복사 기준선이 IntPhys 1 · 2 와 구분 안 되는 것, predictor 가 마지막 관측 근처에서 가져와 채우는 것 (reach), 가려진 자리에 가림막 (정지 배경) 을 그리는 것 (permanence) 은 모두 "정지 · 느린 내용 우선" 과 같은 방향이다.
- 그들의 3-dots 와 우리의 RollOut_v3 (정지 배경 + 물체 하나) 가 같은 구도.

**다른 것 (중요).**
- 그들은 **encoder** 가 움직이는 점을 버린다 (probe 가 못 읽는다). 우리는 **encoder 는 읽는다** (z → 문맥 끝 속도 R² 0.996, p probe 98 %) 이고 **predictor 가 옮기지 않는다.** 즉 같은 편향이 V-JEPA 2 에서는 encoder 가 아니라 predictor 쪽에 남아 있다 — "slow-feature 편향이 encoder 에서 predictor 로 옮겨 갔다" 고 쓸 수 있다 (분석 주장, 원인 주장 아님).
- 목적이 다르다: 그들은 VICReg / SimCLR + GRU rollout, V-JEPA 2 는 masked tubelet + EMA target + 한 번에 예측. 자명해의 형태도 다르다 (그들: 시간 상수 표현; V-JEPA: masking 이 상수 표현을 막고 대신 **국소 복사** 가 최저점). **이 대응은 우리가 검정하지 않았다** — CLAUDE.md 규칙대로 목적함수를 원인으로 걸지 않고 discussion 한 문단으로만 쓴다.
- 그들의 평가는 처음부터 **autoregressive rollout** 이다 (GRU 라 closure 가 구조로 보장). 우리의 closure 결과 (자기 예측을 되먹이면 안 는다) 는 그 구조가 transformer 한 번 예측 predictor 에는 없다는 것이고, 이 논문이 "world model = 굴려야 한다" 를 전제로 삼았다는 점에서 우리 프로토콜의 선례가 된다.
- 그들이 제안한 처방 둘 (H-JEPA 계층, 시간 상수 표현을 막는 제약) 과 우리의 state 모듈 (관측 없이도 굴러가는 상태 + 렌더러) 은 같은 방향이다. 인용해서 "toy 에서 예견된 편향이 대규모 사전학습 predictor 의 어디에 어떤 형태로 남는지" 로 잇는다.

**쓰임.** Related work (b) 항목에 추가, discussion 의 "왜" 문단에서 한 문장. ⚠️ toy · 소형 모델 · 다른 목적함수라 **우리 결과의 원인 근거로 쓰지 않는다.**

## (c) Related-work paragraph (English, ~150 words)

> **Latent world models and their evaluation.** Frozen video encoders paired with a latent predictor are now the standard recipe for world models, whether the predictor is the pretrained V-JEPA 2 network itself, used zero-shot for violation-of-expectation (VoE) scoring [Garrido et al. 2025; Bordes et al. 2025] and action anticipation [Assran et al. 2025; Mur-Labadia et al. 2026; Chu et al. 2026], or a new predictor trained for rollout [V-JEPA 2-AC; DINO-world, Baldassarre et al. 2025; LeWorldModel, Maes et al. 2026]. These works evaluate the predictor with one-shot queries at short horizons (about 1 s) and treat surprise accuracy as evidence of physical understanding, yet none tests whether the latent state is carried forward without observation. Recent benchmarks for generative video models show that state evolution stalls once observation is cut ("tracking shot", Lu et al. 2026; Ma et al. 2026; Chen et al. 2026), and memory adapters such as HERA [2026] target the same failure in V-JEPA 2 but report only surprise gains. We instead measure the frozen predictor directly, with copy and constant-velocity baselines, and show that VoE and anticipation scores do not require the state evolution they are taken to certify.

(Sources are cited here by first author and year; arXiv ids are in the tables above.)

---

## 미확인 · 주의

- ❓ Ego4D LTA **2026** 우승 보고서는 검색에서 못 찾았다 (2025 보고서만). EgoExo4D 공식 anticipation 트랙은 **없음** 을 challenge 페이지로 확인.
- ❓ StEvo-Bench 모델 목록·수치 (Veo 3, Sora 2, Genie 3, <10%) 는 P-25 보고를 따랐고 이번 abstract 페이지에는 안 나온다. 본문 재확인 필요.
- ❓ WRBench 의 Wan 1.3B → 14B 0.66 → 0.62 도 P-25 값. abstract 에는 "increments of scale" 로만 나온다.
- ❓ GOT-JEPA 가 V-JEPA 2 를 쓰는지, PSG-JEPA 가 어떤 encoder 인지 abstract 로는 불명.
- ❓ AVT, AntGPT, PlausiVL, Palm 은 기억 기반 (venue 재확인).
- ❓ Cosmos 3 arXiv id 2606.02800 은 검색 결과의 표기이고 technical report PDF 는 research.nvidia.com 에 있다. 하나로 통일해 인용.
- **철회된 논문 1 편** ("The JEPA Predictor: A Transferable Operator…", 2607.16274) — 검색에 계속 뜨지만 인용 금지.
- IntPhys 2 permanence "쉬움" vs 우리 permanence 실패 — §2-0 의 (a)(b)(c) 중 어느 것도 아직 안 쟀다. 이 문서의 **미결 1줄**.

## 재현

- 조사 방법: WebSearch 25 회, WebFetch 30 회 (arXiv abstract/HTML). 검색어·URL 은 본문 표의 arXiv id 로 되짚을 수 있다.
- 우리 수치 출처: V-JEPA 2 Table 20 전사 `z_research/anticipation/EK100/README.md`; H6 복사 기준선 · M2 · P2 는 `auto_research/README.md` 와 `LITERATURE_POSITIONING_2026-09-25.md` §3-1 (1차 결과, 적대 검증 전).
- 이 문서는 `LITERATURE_POSITIONING_2026-09-25.md` 를 대체하지 않는다. 정의·리뷰어 행태·설계 원칙은 그쪽이 정본이다.
