# 문헌 조사 — "predictor 의 자리가 비어 있다" 논지의 선행 (2026-09-29)

> **역할.** [`paper/RETHINK_PREDICTOR_ROLE_2026-09-29.md`](../paper/RETHINK_PREDICTOR_ROLE_2026-09-29.md) §3 의 논지
> ("JEPA 계열 world model 에서 예측의 자리가 비어 있다 — encoder 가 정보를 담고, 사전학습 predictor 는 마지막 관측에서 가져와 채울 뿐이며,
> IntPhys 1/2 · EK100 은 마지막 latent 복사로 맞춰진다. 그래서 encoder 는 얼리고 실영상 예측 목표로 **새 predictor** 를 학습한다 —
> reach · permanence · closure 에서 도출한 설계로") 를 기준으로 (a)–(g) 일곱 축을 다시 조사했다.
> [`LITERATURE_STATE_WORLD_MODELS_2026-09-26.md`](LITERATURE_STATE_WORLD_MODELS_2026-09-26.md) (이하 **S-26**) 와
> [`LITERATURE_POSITIONING_2026-09-25.md`](LITERATURE_POSITIONING_2026-09-25.md) (**P-25**) 가 이미 다룬 논문은 **표에서 빼고**, 새로 확인한 사실이 있을 때만 한 줄로 적었다.
> 로컬 PDF 넷 (`papers/`) 은 pdftotext 로 본문을 직접 읽었다.
>
> **확인 표기**: ✅ arXiv abstract/본문 직접 확인 · ◐ 검색 요약·초록만 (본문 미확인, 인용 전 재확인) · ❓ 미확인 (id·venue·수치 불확실). 수치는 확인한 것만 적고 못 한 것은 **미확인** 이라 썼다.
> 조사 방법: 하위 조사 5 갈래 (WebSearch ≈ 90 회, WebFetch ≈ 130 회) + 핵심 인용문 6 건은 내가 다시 원문을 열어 검증했다 (§10 에 표시).

---

## 0. 결론 먼저

**결론.** "frozen encoder + 사전학습 latent predictor 가 실제로는 마지막 관측을 옮겨 적을 뿐이고, 그 predictor 를 인증해 온 벤치마크 (VoE, EK100) 가 복사 기준선으로 맞춰진다" 는 주장을 **같은 대상 (V-JEPA 2 릴리즈 predictor) 에서 같은 귀무 (복사) 로 잰 선행은 없다.**
다만 세 방향에서 **부분 선점**이 있다: (i) **Human-JEPA** (2608.21160) 가 JEPA predictor 에 **copy-last latent 기준선**을 처음 붙였다 (자기 predictor, K700, 0.873 vs 0.798). (ii) **GEOPHYS** (2606.20707) 가 predictor 없이 frozen *image* encoder 기하만으로 IntPhys 2 93.3 % 를 낸다 — "predictor 가 필요 없다" 의 가장 강한 외부 증거. (iii) **Physion** (2106.08261) 이 2021 년에 이미 "observed+simulated ≈ observed (p = 0.53)" 로 "모델의 미래 굴림은 점수에 기여하지 않는다" 를 보였다.
**새 predictor 설계 (frozen V-JEPA 2 + 실영상 예측 목표)** 는 FAIR 자체가 2026-01 에 같은 구성을 냈다 (**Garrido et al. 2601.05230**, latent action WM) — 그러나 복사 기준선·reach·permanence 를 재지 않고, "이 표현 공간은 예측을 염두에 두고 설계되지 않았다" 고 한계로 적었다.

**근거 3줄.**
1. **(a)** 얼리는 이유를 문헌은 collapse 회피·probe 비교·파이프라인 단계로만 든다 (FlowWM, Dreamer 4, 2605.15618). "encoder 에 이미 정보가 있고 predictor 자리가 비어서" 는 아무도 안 썼다. open-loop 예측 오차를 보고하는 frozen-feature WM 은 DINO 계열 (FlowWM, DWM) 과 NWM 뿐이고, V-JEPA 특징 위에서 **복사 귀무**를 붙인 건 Human-JEPA 하나다.
2. **(b)(c)** IntPhys 1 (1803.07616) · IntPhys 2 · GRASP · InfLevel · Physics-IQ · EK100 (RULSTM, AVT, TAP-JEPA) **어디에도 copy-last / static 기준선이 없다** (본문 확인). 복사 기준선은 semantic forecasting (Luc 2017) 과 feature forecasting (DINO-Foresight) 에서는 표준이었는데 VoE·anticipation 으로 넘어오며 빠졌다. Joseph et al. (2602.07050) 부록 C.1.4 는 **encoder layer 0 (probe 로는 chance) 위에 predictor 를 새로 학습해도 VoE 점수가 난다** 고 스스로 적었다 — VoE 점수가 입력 표현의 물리를 요구하지 않는다는 저자 측 증거.
3. **(d)(e)** closure 의 처방은 8 계열로 정리된다 (§5-2). frozen latent 에서 exposure bias 를 명시적으로 다룬 건 V-JEPA 2-AC (T=2 rollout loss), VGGT-World (flow-forcing curriculum), VLWM (지평 직접 예측), Flow-JEPA (trajectory flow matching) 뿐이고 **"teacher-forced 정확도가 rollout 으로 옮겨지지 않는다" 를 frozen-encoder 설정에서 측정한 문장은 없다.** what/where 분리는 픽셀 시대 (MCnet, DrNet, SDC-Net) 와 G-SWM (`z_what`/`z_where`) 에 선례가 있지만, 현대 slot-Transformer 계열은 위치를 slot 안에 다시 얽고 예측 구간 permanence 를 안 잰다 (SlotFormer 의 가림 처리는 **burn-in 구간 한정**).

4. **(f)(g)** anticipation 문헌은 2019–2021 에 이미 "미래 특징 회귀의 몫 0~1 pt" (AVT-b 13.1 → 14.4 / 13.0, ImagineRNN +0.14~+1.25) 와 "원인 = 카메라 운동" (RULSTM: EGTEA 에서 미래 특징 사전학습이 **해침** 60.18 → 50.22, 3인칭 ActivityNet 에서는 도움) 을 적었다. 2025–26 (V-JEPA 2 Table 20 +0.6, TAP-JEPA ablation 없음) 은 이를 잊었다. 기능 분리의 정본 문장은 LeCun 2022 ("(1) estimate missing information … (2) predict plausible future states") 이고 — 이 둘이 우리 permanence·reach 와 정확히 대응한다. ⚠ Garrido 2025 에 "predictor acts as a world model" 문장은 **없다**.

**미결 1줄.** Human-JEPA 의 0.873/0.798 은 *자기* predictor 수치라 **릴리즈 V-JEPA 2(.1) predictor 의 copy-gap 은 아직 아무도 보고하지 않았다** — 우리 H6 (IntPhys 1 복사 85.00 vs 88.89) 이 그 자리다. 단 프로토콜 (K700 latent cosine vs 우리 VoE 정확도) 이 달라 **직접 비교는 안 된다**.

---

## 1. 로컬 PDF 정독 — 우리 논지에 닿는 사실만

### 1-1. "Latent Video Prediction Learns Better World Models" — Alrasheed, Yazdan Parast, Azam, Bailey, Akhtar. arXiv:2605.15618 (2026-05-15). ✅ 전문
- **제목과 달리 predictor 를 전혀 쓰지 않는다.** frozen encoder 4 종 (V-JEPA 2.1 / 2 / VideoPrism / VideoMAEv2, ViT-L) 을 attentive probe 로 SSv2 에서 5 축 (discriminability · corruption · pretend-vs-real · occlusion · temporal) 로 비교한 **encoder 강건성 연구**다. "world model" 은 곧 "encoder 표현" 이다.
- 우리에게 쓸모 있는 수치: (i) **Temporal Dropout** = 연속 프레임 블록을 **마지막 보이는 프레임 반복**으로 대체 (App. E.3) — V-JEPA 2/2.1 이 가장 완만히 떨어진다 (RSI 기울기 −0.047 / −0.088 vs VideoMAEv2 −0.329, Table 7). 즉 encoder 는 "관측이 얼어붙은" 입력에서도 정보를 유지한다 → "정보는 encoder 에 있다" 쪽 근거. (ii) **Static video** (한 프레임 반복) 에서 V-JEPA 2.1 은 가운데 프레임에 anchoring, VideoPrism 은 SGI 0.863 (한 프레임이 latent 의 거의 전부) — encoder 쪽 static bias 는 V-JEPA 에 **약하다** (§4 (c) 와 대조). (iii) "latent-prediction models internalise the arrow of time" (video reversal 에서 antonym flip).
- **relation**: 지지 (encoder 충분) — 단 predictor 미평가라 우리 주장 (predictor 자리) 에는 직접 닿지 않는다. 인용은 "encoder 가 가림·정지 입력에서도 정보를 유지" 로만.

### 1-2. "Interpreting Physics in Video World Models" — Joseph, Garrido, Balestriero, Kowal, Fel, Bakhtiari, Richards, Rabbat. arXiv:2602.07050 (2026-02-04; P-25 기준 ICML 2026). ✅ 전문
- encoder 만 (V-JEPA 2 L/H/g, VideoMAEv2-G). Physics Emergence Zone (1/3 깊이) · 방향은 고차원 ring code · IntPhys 부분공간과 방향 부분공간은 무작위 수준 겹침 (7–13 %, Table 3) · 국소 attention 억제가 IntPhys 를 78.3 → 51.9 로 깎음 (Table 2). **P-25 §2-4 와 같음.**
- **새로 확인한 것 (App. C.1.4, Fig. 11)** — VoE 과제 (Garrido 프로토콜) 로 **encoder 각 층 위에 V-JEPA 2-L predictor 를 처음부터 학습**했다. 중간 1/3 층이 최고이고, 그리고:
  > "even early layers that give poor performance on only the encoder suddenly perform well (Layer 0). This is likely because the predictor itself is learning from the representation."
  > "At the end of the network, information is catered to the optimization objective of predicting the next frame in latent space, not necessarily to preserving object-level information."
  → (i) **VoE 점수는 입력 표현에 물리가 읽히는 것을 요구하지 않는다** (layer 0 probe ≈ chance 인데 predictor 를 붙이면 점수). 우리 "점수가 predictor 의 예측력을 인증하지 않는다" 와 같은 결. (ii) 마지막 층이 물체 정보를 덜 보존한다는 그들 해석은 우리 새 predictor 의 **입력 층 선택** (중간층?) 에 직접 닿는 설계 수치다.
- Limitations: "autoregressive or diffusion video models may exhibit different representational structure", "do not isolate … long-horizon interaction". predictor 의 자리는 그들도 비워 뒀다.

### 1-3. V-JEPA 2 (2506.09985) — world model 절 · 한계 · EK100 (✅ 전문, 인용문 원문)
- 역할 정의 (§1): "learn to both represent video observations and learn a predictive model for world dynamics in this learned representation space." 2-AC 는 **frozen encoder + 새 predictor** ("Note that the encoder is kept frozen during this post-training phase"), 손실 = teacher forcing (T=15) + rollout (T=2, "we only differentiate the predictor through one recurrent step").
- 한계 (§4): "**autoregressive prediction suffers from error accumulation**: the accuracy of the representation-space predictions decreases with longer autoregressive rollouts". App. B.3: "we do observe error accumulation as the world model predicts the location of the cup to be slightly lower than that of the real trajectory in the final frame."
- **encoder 천장 문장 (§5 첫 줄)**: "The capabilities of a representation-space world model, such as V-JEPA 2-AC discussed above, are **inherently limited by the state information encoded in the learned representation space**." — 리뷰어의 "frozen encoder 가 천장" 공격에 저자 문장으로 답할 수 있다 (우리: encoder 에 정보가 있음을 먼저 보인다).
- EK100 (§6, App. D.2, Table 20): encoder 39.1 / predictor 만 20.2 / 둘 39.7 → "**the EK100 task mostly requires strong semantic understanding, as opposed to forecasting capabilities**". 지평 1/2/4/10 s 에서 "performance sharply decreases … which is expected since forecasting the future in EK100 is a non-deterministic task" (Fig. 18). 실패 사례 해석: "The model also predicts 'rinse sponge', which is the current action being performed, probably assuming that this action could still be going on after 1 second." — 복사 행동을 저자가 그림으로 적은 문장.
- IntPhys 2 (2506.09849) 와 함께: **복사 기준선 없음** (본문 재확인).

### 1-4. IntPhys 2 (2506.09849) — 프로토콜·permanence (✅ 전문)
- Table 2/3 재확인: V-JEPA 2-h main 57.51 / held-out 56.40; permanence fixed 59.62 / moving 57.35. "The permanence condition appears to be the easiest for both models and humans, **as objects are not moving by themselves**."
- App. D.2: 두 프로토콜 모두 최적이 "window size of 16 and 6 as framerate (i.e. less than 3 seconds), **which makes most samples from IntPhys 2 impossible**." App. E.1 (Fig. 9): "when the ball is supposed to reappear, the model is not more surprised for the impossible than possible videos … looking almost 200 frames in the past is too difficult." App. E.2: Cosmos 도 "we do not see the ball reappear in either setting."
- → S-26 §2-0 의 긴장 해소 (a) 쪽 근거: "쉬운 permanence" 는 정지 물체이고, 재등장 시점의 surprise 상승은 그들 그림에서도 없다. **우리 permanence 실패와 모순되지 않는다** (단 셀 단위 재채점은 여전히 미결).

---

## 2. (a) frozen encoder + 학습된 latent dynamics — 누가 얼리고 왜, open-loop 을 재는가

S-26 §1 표 (V-JEPA 2/2.1, 2-AC, DINO-world, DINO-WM, LeWM, HERA, JEPA-x, PSG-JEPA, RDR) 와 P-25 §2-6 (PLDM, JEPA-WMs, Semigroup-JEPA, UWM-JEPA, DINO-Foresight, Causal-JEPA) 은 반복하지 않는다.

| 논문 | 연도 | venue / arXiv | 확인 | 무엇 (얼리는가 · 왜) | open-loop · 복사 기준선 | 숫자 · 주장 | 우리와 |
|---|---|---|---|---|---|---|---|
| **Learning Latent Action World Models In The Wild** — Garrido, Nagarajan, Terver, Ballas, LeCun, Rabbat | 2026-01 | arXiv 2601.05230 | ✅ 본문 (내가 재확인) | **frozen V-JEPA 2-L** (frame-causal) 위에 latent action + predictor 를 실영상에서 학습. "This encoder is kept frozen during training." | decoded LPIPS 2 s; rollout 16 프레임 @ 4 fps; Fig. 10 "physical appearance degrades over time". **복사 기준선 없음** | 한계 절: "**This representation space was not designed with prediction in mind**, which can hinder the inverse dynamics training, as well as the quality of the predictions in general." | **가장 가까운 FAIR 측 선행** (frozen V-JEPA 2 + 실영상 새 predictor). 우리가 더 가는 곳: 복사 귀무 · reach/permanence/closure 판독 · 왜 그 공간이 예측에 불리한지 (predictor 쪽 병목) |
| **Human-JEPA** — Wei, Sun, Zhao | 2026-08 | arXiv 2608.21160 | ✅ 본문 (내가 재확인) | frozen backbone probe 프로토콜; V-JEPA 2.1-L 번들에서 시작해 사람 중심 영상으로 이어 학습 (past→future split) | **copy-last latent 기준선 있음**: K700 held-out 20 clip, 앞 절반 → 뒤 절반 latent 예측 cosine **0.873 vs 0.798 (마지막 latent 고정)** vs 0.735 (다른 clip 미래). "The margin over the static reference … widens with distance, from 0.057 half a second ahead to 0.087 two seconds ahead" | NTU-120 early action +4.1 (초록). 저자 스스로 "representation consistency" 라 부름 | **부분 선점 (복사 귀무)**. 단 *자기* predictor · latent cosine · 20 clip. 릴리즈 predictor 의 copy-gap 미보고. 우리는 VoE·EK100 **점수** 에서 복사가 맞춘다는 것을 보인다 |
| Dreamer 4 — Hafner, Yan, Lillicrap | 2025 | arXiv 2509.24527 | ✅ | tokenizer 먼저 학습 후 **얼림** ("The dynamics model operates on the interleaved sequence of actions and representations produced by the frozen tokenizer"); 이유 = 파이프라인 단계 | open-loop 정량 곡선 없음 (인간 평가 vs Oasis/Lucid/MineWorld); 복사 기준선 없음; "subtle errors that accumulate over time" 을 shortcut forcing 동기로 | Minecraft 다이아몬드 offline 최초 | 병렬 설계 (frozen 표현 + 별도 dynamics) 이나 tokenizer 는 복원 학습. 진단 없음 |
| Navigation World Models — Bar, Zhou, Tran, Darrell, LeCun | 2025 | CVPR 2025 / arXiv 2412.03572 | ✅ | SD-VAE latent 위 CDiT (1B). VAE 동결 여부 본문 미명시 (미확인) | **open-loop 있음**: LPIPS/DreamSim/PSNR @ 1–16 s, AR rollout; "After 8 seconds, predictions degrade due to accumulated errors and loss of context." 복사 기준선 없음 | RECON 4 s LPIPS 0.295 | closure 인정 (측정은 절대 오차만). 우리는 복사 대비로 잰다 |
| FlowWM — Flow Matching in Feature Space | 2026 | arXiv 2606.29059 | ✅ | frozen DINOv3 ViT-S. 얼리는 이유 명시: "avoids objective collapse, but also **limits the representational quality of the predicted futures to that of the encoder**" | ℓ1 latent 오차 vs 지평 (Fig. 6), 예측 특징 위 detection AP; 복사 기준선 없음 | vs DINO-WM: Bouncing-Shapes F1 err 4.31 vs 17.8 | 얼림 근거의 표준 표현 (collapse). 결정론 회귀의 평균화 문장 (§10) 은 closure 설계 caveat |
| DWM — Separating World Effects from Actions | 2026 | arXiv 2607.18715 (v2 09-27) | ✅ | frozen DINOv2 ViT-S/14 (DINO-WM 설정 계승) | 1-step vs rollout@20 latent 오차 (Table 2; 수치는 fetch 파생 열이 안 맞아 **미확인**), 복사 기준선 없음 | 행동/세계 효과 residual 분해 | DINO-WM 계열; drift 보고하나 복사 귀무 없음 |
| What Makes Video WM Latents Action-Relevant — Yeom et al. | 2026-06 | arXiv 2606.07687 | ✅ | frozen backbone 8 종 + ID fine-tune 대조 (probe 프로토콜) | 지평 곡선 없음 | V-JEPA 2 ViT-L inverse-dynamics R² 0.40 (frozen) → 0.85 (+ID); "**most gains arise from natural-video temporal context, with feature-level latent prediction providing a smaller additional benefit**" | 양면: encoder 에 동역학 정보 있음 (지지) / latent 예측 목표 자체의 몫은 작다 (우리 "predictor 몫이 작다" 와 같은 방향, encoder 층위) |
| TAP-JEPA | 2026 | arXiv 2606.00662 | ✅ 본문 | encoder **와 predictor 둘 다** 얼림; 이유 = "keeps supervised optimization focused on the EK-100 label space and limits overfitting" | **encoder-only ablation 없음** ("Concatenate encoder tokens and predictor tokens…"); 복사 기준선 없음 | 27.91 % (2위) | (S-26 과 동일) 검증 없이 predictor 를 prior 로 쓰는 관행 — 우리 복사 결과가 그 빠진 ablation 이다 |
| Video Prediction Policy — Hu et al. | 2025 | ICML 2025 / arXiv 2412.14803 | ◐ | **얼리지 않음** (VDM fine-tune). 동기: "Previous vision encoders … tend to capture static information" | downstream 만 | Calvin +18.6 % rel. | 반대 설계이자 거울상 동기 (그들: encoder 가 정적 / 우리: predictor 가 정적) |
| Vid2World 2505.14357 · AdaWorld 2503.18938 · LAPA 2410.11758 · Flow-JEPA 2608.29029 · Inference-time Physics Alignment 2601.10553 | 2025–26 | arXiv | ◐ | 픽셀/latent action/flow 계열; 얼림·지평·복사 초록에 없음 (미확인) | — | Flow-JEPA planning 86→92 (clean) / 67→86 (noisy) | 배경 (downstream-only 평가 관행) |
| Latent Video Prediction Learns Better World Models 2605.15618 · Interpreting Physics 2602.07050 | 2026 | arXiv | ✅ 전문 | §1-1 · §1-2 | — | — | encoder 쪽 증거 |

**정리.** 얼리는 근거 셋 (collapse 회피 · probe 비교 · 파이프라인) 중 어느 것도 "predictor 자리가 비었다" 가 아니다. V-JEPA 특징 위 predictor 의 open-loop + 복사 귀무를 같이 낸 논문은 없다 (Human-JEPA 가 절반).

---

## 3. (b) VoE · anticipation 점수는 예측을 요구하는가 — 복사/정적 기준선 실사

S-26 §4 (Garrido, IntPhys 2, MVP, Punzo, R2M, Asleep at the Wheel, DINO-world "sanity check", WorldTest, X-VoE, Physion/Nayebi) 은 반복하지 않는다.

| 논문 | 연도 | venue / arXiv | 확인 | 무엇 | 숫자 · 주장 | 우리와 |
|---|---|---|---|---|---|---|
| **IntPhys (2019)** — Riochet et al. | 2018/21 | TPAMI 2021 / arXiv 1803.07616 | ✅ 본문 (pdftotext) | 자기지도 기준선 2 종 (CNN enc-dec, GAN) 이 2 프레임에서 5/35 프레임 뒤 **semantic mask** 예측; plausibility = 예측–관측 거리 | **copy-last / static 모델 기준선 없음.** 표의 "Static" 은 **장면 조건** (static / dynamic-1 / dynamic-2) 이지 모델이 아니다. "obtained above chance performance using a mask prediction task, with a very strong effect of the presence of occlusion" | 원 벤치마크부터 no-dynamics 대조가 없었다 |
| **Physion** — Bear et al. | 2021 | NeurIPS D&B 2021 / arXiv 2106.08261 | ✅ 본문 (내가 재확인) | OCP readout 두 프로토콜: observed vs observed+**simulated** (모델의 굴린 미래 포함) | "Vision model predictions from the observed+simulated readout protocol were, overall, **no better than predictions from the observed protocol (p=0.53, Fig. 5D)**." / "any above-chance performance … was likely due to having visual features that could discriminate some trial outcomes from cues in the initial movie segment." (§3 "What have vision-based models actually learned?") | **가장 가까운 no-dynamics 대조의 선례** (P-25 에서는 "지각이 병목" 결론만 인용). 우리는 이 논리를 사전학습 latent predictor + 복사 귀무로 옮긴다 |
| Physion++ — Tung et al. | 2023 | NeurIPS 2023 / arXiv 2306.15668 | ✅ HTML | "w/ property" vs "w/o property" (초기 상호작용 프레임 제거) | "Most models with property inference perform similarly to those without property inference, indicating that these models are not utilizing physical property inference." | 문맥의 동적 증거를 안 쓴다 — 지지 |
| **GEOPHYS** — Internò … Simoncelli, Hammer, Klindt | 2026-06 | arXiv 2606.20707 | ✅ 초록 (내가 재확인) | **frozen image encoder 의 프레임별 embedding 기하 5 개** 로 plausibility. predictor·시간 모델 없음 | **IntPhys 2 93.3 %**, LikePhys 98.3 %; "V-JEPA 2, GPT-4o, Gemini, and twelve modern video diffusion models perform near chance"; V-JEPA 2 verifier 대비 1.5× 빠르고 4.65× 메모리 적음 | **강한 지지 + 위험** (§9). IntPhys 2 가 predictor 없이 풀린다 = "predictor 자리가 비었다" 의 외부 증거. ⚠ split/프로토콜 미확인 (초록에 없음) — 본문 확인 전 수치 인용 금지 |
| Physics-IQ — Motamed et al. | 2025 | arXiv 2501.09038 | ✅ HTML | 실영상 continuation, physical variance 상한 100 | **copy-last / static 기준선 없음** (HTML 확인). best 29.5 %; "Visual realism is uncorrelated with physical understanding (r = −0.46, p = .249)". I2V 는 마지막 프레임만 조건 | 정적 프레임 점수가 빠져 있다 — 우리가 메울 수 있는 칸 |
| Physics-IQ Verified — Rädsch et al. | 2026 | arXiv 2606.18943 | ✅ 초록 | 57.6 % 샘플 정제, 순위 Kendall τ 0.46 | 사소 기준선 없음 | 중립 |
| InfLevel — Weihs et al. | 2022 | TMLR 2022 (arXiv 없음; GitHub allenai/inflevel) | ◐ 초록+README | 10 아키텍처, Continuity/Solidity/Gravity | "at or near chance"; README: "**This tight control opens the door for designing heuristics that can obtain high performance.**" | 저자 스스로 heuristic 해법 가능성 명시. static 기준선 미확인 (본문 로그인 벽) |
| GRASP — Jassim et al. | 2024 | IJCAI 2024 / arXiv 2311.09048 | ✅ 초록 | Unity VoE, MLLM 대상 | 전부 chance 이하/근처, 인간 80 % | 복사 기준선 없음 |
| LikePhys — Yuan et al. | 2025 | arXiv 2510.11512 | ✅ HTML v2 | video diffusion 의 likelihood-preference VoE (PPE) | 기존 방법은 "usually fail to disentangle physics from visual appearance" | surprise 의 외형 교락 비판 — 지지 |
| AVT — Girdhar & Grauman | 2021 | ICCV 2021 / arXiv 2106.02036 | ✅ ar5iv | EK100 ablation: naive (다음 행동 손실만) vs anticipative (미래 특징 예측 손실 추가), Table 7 val R@5 | TSN 10.1 → 13.6; **AVT-b 13.1 → 14.4**. "employing the anticipative training losses are imperative…" **last-clip-only 기준선 없음** | 미래 특징 예측의 몫 +1.3 pt — 우리 EK100 +0.6~+2 와 같은 규모. (f) 절과 연결 |
| RULSTM — Furnari & Farinella | 2019 | ICCV 2019 / arXiv 1905.09035 | ✅ ar5iv | rolling/unrolling LSTM, EK55 Top-5 action 35.32 % @1 s | 구조 ablation 만, static 기준선 없음 | 공백 |
| Ego4D — Grauman et al. | 2022 | arXiv 2110.07058 | ✅ 초록 | — | 초록에 사소 forecasting 기준선 없음 | 공백 |
| Luc et al., Predicting Deeper into the Future of Semantic Segmentation | 2017 | ICCV 2017 / arXiv 1703.07684 | ✅ ar5iv | **"The first baseline copies the last input frame to the output."** Cityscapes 단기 IoU 49.4 (copy) vs 58.3 (S2S) | forecasting 에서 copy-last 는 표준 대조였다 | VoE·anticipation 으로 오며 빠진 대조의 계보 |
| AVoE 2111.08826 · CLEVRER 1910.01442 · Su et al. 2512.06232 | — | — | ✅/❓ | 사소 기준선 없음 (CLEVRER 는 검색만) | — | 중립 |

### 3-1. 벤치마크별 "실제로 보고된 복사/정적 기준선"
- IntPhys 1 → **없음** (1803.07616 ✅) · IntPhys 2 → **없음** (2506.09849 ✅) · GRASP → 없음 (✅) · InfLevel → 미확인 (◐) · Physics-IQ → **없음** (✅) · CLEVRER → 없음 (❓)
- Physion → **observed-only ≈ observed+simulated, p = 0.53** (✅) · Physion++ → w/o property ≈ w/ property (✅, 수치 미확인)
- EK100 (RULSTM, AVT, TAP-JEPA, V-JEPA 2) → 없음. 가장 가까운 것은 V-JEPA 2 Table 20 (encoder-only 39.1 vs +predictor 39.7) 과 AVT naive-vs-anticipative (13.1 → 14.4)
- Ego4D → 없음 (초록)
- 표준으로 남아 있는 곳: semantic forecasting (Luc 2017 ✅) · feature forecasting (DINO-Foresight Copy-Last, P-25) · 생성형 (2607.27036 "frame-copying" 명명, 수치 없음)

---

## 4. (c) transformer / JEPA predictor 는 시뮬레이션이 아니라 조회·복사한다 — 증거

| 논문 | 연도 | venue / arXiv | 확인 | 무엇 | 숫자 · 주장 | 우리와 |
|---|---|---|---|---|---|---|
| **Mitigating Compounding Error via Video Representation Regularization** — Chen, Zhang, Wang | 2026 | arXiv 2607.27036 | ✅ 본문 | AR diffusion video 의 effective-rank 붕괴, VRR 정규화 | "**When frame contents change mildly, directly copying adjacent frames enables smooth temporal consistency and favorable generation quality.** Nevertheless, such frame-copying behavior erodes the amount of valid information contained within latent representations." — "shortcut learning in autoregressive diffusion models" | 생성형 world model 에서 "복사 = shortcut" 을 명명. 우리는 latent predictor 판 |
| V-JEPA — Bardes et al. | 2024 | TMLR 2024 / arXiv 2404.08471 | ✅ ar5iv | multi-block 마스킹 근거 | "Masking a large continuous block that covers the full temporal dimension **limits information leakage due to the spatial and temporal redundancy of videos.**" | 사전학습 과제 자체가 "누수 = 복사" 를 기본 shortcut 으로 전제 |
| VideoMAE — Tong et al. | 2022 | NeurIPS 2022 / arXiv 2203.12602 | ◐ | tube masking 90–95 % | 시간 상관이 "information leakage" 를 낳고 인접 프레임에서 "unmasked copy" 를 찾기 쉽다 (요약 문구, 원문 재확인) | 같은 계보 |
| NextLat — Teoh et al. | 2025 | arXiv 2511.05963 | ✅ 초록 | belief-state 보조 목적 | transformer 는 "self-attention that enables ad-hoc lookups over past tokens" 이고 "lack an inherent incentive to compress history into compact latent states" | **구조 층위의 "조회 vs 상태"** 문장 (P-25 §2-8 에는 보조 손실로만) |
| Kowal et al. (static vs dynamic) | 2022/24 | CVPR 2022 2206.02846 / TPAMI 2024 2211.01783 | ✅ 초록 | 채널별 static/dynamic 편향 정량, StaticDropout | "Most examined models are biased toward static information"; 동적이라 여긴 데이터셋도 실제로는 정적 편향 | encoder 쪽 static bias 계보 |
| Lei, Berg, Bansal (single-frame bias) | 2023 | ACL 2023 / arXiv 2206.03428 | ✅ 초록 | 단일 프레임 학습 | "a single-frame trained model that does not consider temporal information can achieve better performance than existing methods that use multiple frames" | 동일 |
| Sevilla-Lara et al., Only Time Can Tell | 2021 | WACV 2021 / arXiv 1907.08340 | ✅ 초록 | 프레임 셔플로 시간 의존 클래스 분리 | "action classes can often be recognized without any temporal information from a single frame of video." | 동일 |
| Choi et al., Why Can't I Dance in the Mall | 2019 | NeurIPS 2019 / arXiv 1912.05534 | ◐ | 장면 편향 제거 손실 | 모델이 "leverage such bias instead of using the actual discriminative cues" (요약) | 동일 |
| Ilic, Pock, Wildes, Appearance-Free Action Recognition | 2022 | arXiv 2207.06261 (venue 미확인) | ✅ 초록 | Appearance-Free Dataset, 11 아키텍처 vs 인간 | 모두 정적 단서 없이는 크게 하락, 인간은 아님 (수치 미확인) | 동일 |
| Alrasheed et al. 2605.15618 | 2026 | — | ✅ 전문 | §1-1 | V-JEPA 2.1 encoder 는 가운데 프레임 anchoring (정적 편향 약함); VideoPrism SGI 0.863 | **encoder 에서는 반례** — static bias 가 V-JEPA 2 에서는 predictor 쪽에 남는다는 우리 서술 (S-26 §b2 Sobal 대응) 과 정합 |
| Yeom et al. 2606.07687 | 2026 | — | ✅ 초록 | §2 | "feature-level latent prediction providing a smaller additional benefit" | latent 예측 목표의 몫이 작다 |
| Liu & Chen 2606.28455 · Liu et al. 2503.15096 | 2025–26 | — | ✅ 초록 | 사건 조건 진단 / sandwich sampling | 명시적 복사 주장 없음 | 주변 |

**정리.** "latent world model 은 근사 retriever 다" 류의 제목·주장은 **없다** (검색 실패). 구조 층위 (NextLat), 생성형 (2607.27036), 사전학습 과제 설계 (V-JEPA/VideoMAE 의 leakage 문장) 셋을 묶으면 "복사가 손실의 기본 최저점" 을 **저자들 문장으로** 세울 수 있다. Sobal 2022 (S-26 §b2) 가 이 계보의 이론적 출발점.

---

## 5. (d) exposure bias · teacher forcing vs rollout 붕괴 — 처방 지도

P-25 §2-8 (GNS, pushforward, Diffusion Forcing, Self Forcing, GameNGen, DPWM, Clockwork VAE, S4WM, R2I, SlotSSM, Echo-Memory, BST, NextLat) 과 S-26 (V-JEPA 2-AC, RDR, JEPA-x, Semigroup-JEPA) 은 반복하지 않는다.

| 논문 | 연도 | venue / arXiv | 확인 | 계열 · frozen enc? | 처방 | 지평 · 숫자 | closure 와 |
|---|---|---|---|---|---|---|---|
| Scheduled Sampling — Bengio et al. | 2015 | NeurIPS 2015 / 1506.03099 | ✅ | seq RNN | 학습 중 GT 입력을 자기 샘플로 점진 교체 | "This discrepancy between training and inference can yield errors that can accumulate quickly along the generated sequence." | 문제의 명명. rollout loss 의 조상 |
| Professor Forcing — Lamb et al. | 2016 | NeurIPS 2016 / 1610.09038 | ✅ | seq RNN | teacher-forced 와 free-running **hidden-state 분포**를 adversarial 로 맞춤 | — | 우리 framing 에 가장 가까운 고전: 자기 출력 하 **상태 분포**가 다르다 |
| MeshGraphNets — Pfaff et al. | 2021 | ICLR 2021 / 2010.03409 | ✅ 초록, 잡음 세부 ◐ | 물리 시뮬레이터 (enc 없음) | 최신 상태에 Gaussian 잡음 (1-step 오차 규모) | 수백~수천 step (◐) | 1-step 감독 + 잡음 ⇒ 안정. 잡음이 **원 상태** 등방성 — 1024-d ViT latent 의 구조적 off-manifold 오차와 다름 |
| PlaNet — Hafner et al. | 2019 | ICML 2019 / 1811.04551 | ✅ | RSSM (enc 공동학습) | **latent overshooting** ("consistency between one-step and multi-step predictions" ◐) | — | latent 다중 step 손실의 원형 (P-25 §2-1 에는 정의 틀로만) |
| DreamerV3 2301.04104 · TD-MPC2 2310.16828 · STORM 2310.09615 · Δ-IRIS 2406.19320 · IRIS 2209.00588 · TWM 2303.07109 | 2023–24 | ICLR/NeurIPS/ICML | ✅ 초록 | RL latent (enc 공동학습) | 범주형/확률 latent (V3, STORM "Introducing random noise … beneficial"), decoder-free 일관성 (TD-MPC2, ◐), **이산 토큰 = 코드북 투영으로 출력이 곧 유효 입력** (IRIS, Genie, MineWorld), 연속 상태 토큰 vs 이산 delta 출력 (Δ-IRIS) | Atari-100k 수치 (초록) | 계열 4·8 (§5-2). 우리 연속 L1 회귀와 대비 |
| Genie 2402.15391 · Genie 3 (blog) · GAIA-1 2309.17080 · GAIA-2 2503.20523 · Cosmos 2501.03575 · MineWorld 2504.08388 · Matrix-Game 2.0 2508.13009 | 2023–25 | — | ✅/◐ | 픽셀 생성 | 명시 처방 없음 (Genie 3: 기억 ~1 분, "a few minutes of continuous interaction, rather than extended hours"; Matrix-Game 2.0: causal attention·KV cache 는 "do not fundamentally address error accumulation" ◐) | — | 최전선 규모에서도 closure 가 세션 길이를 정한다 |
| Rolling Diffusion 2402.09470 · DFoT/History Guidance 2502.06764 · Rolling Forcing 2509.25161 · Causal-rCM 2606.25473 · Causal Forcing++ 2605.15141 · Mask Forcing 2609.09123 | 2024–26 | ICML 2024 / ICML 2025 / arXiv | ✅/◐ | video diffusion | sliding-window 잡음 스케줄 (학습 = 추론), history guidance, attention sink + 비중첩 창 distillation, TF 초기화 → self-forcing 정제 ("offline, forward-divergence" vs "on-policy, reverse-divergence") | Rolling Forcing: 수 분, 1 GPU | 계열 3 (professor-forcing 계보). frozen enc 아님 |
| Cycle-World 2607.11836 · StableWorld 2601.15281 | 2026 | ECCV 2026 / arXiv | ✅ | AR video diffusion | cycle-consistency + 역예측 모델을 **런타임 교정기**로 (history 에 commit 전) / 열화된 자기 프레임 **eviction** | 60 s | 계열 7·6. StableWorld 진단: "error accumulation originates from the same scene, where generated frames gradually deviate from the initial clean state." |
| WorldMem 2504.12369 · Context-as-Memory 2506.03141 · Long-term Spatial Memory 2506.05284 · Infinite-World 2602.02393 · Matrix-Game 3.0 2604.08995 | 2025–26 | NeurIPS 2025 / SIGGRAPH Asia 2025 / arXiv | ✅/◐ | video diffusion | 외부 메모리 (프레임 + pose) | revisit PSNR (§6-3) | **직교** (forgetting 을 고침, per-step drift 아님) |
| **VGGT-World** | 2026 | arXiv 2603.12655 | ✅ | **latent, frozen VGGT 기하 특징**, flow matching | 2 단계 **latent flow-forcing curriculum**: "progressively conditions the model on its own partially denoised rollouts"; clean-target 매개화 | 0.43B, 3.6–5× 빠름 | **강한 지지**: frozen-feature AR 은 "suffers from compounding exposure bias"; 처방 = 자기 rollout 학습 + 확률 head |
| **VLWM** (Variable-length Latent WM) | 2026 | arXiv 2606.21775 | ✅ | latent JEPA (LeWM 기반; enc 상태 미확인) | **지평을 직접 예측** (재귀 회피) + 지평 확장 curriculum | LeWM 대비 +13 % | "existing latent world models typically rely on one-step prediction and must be recursively rolled out …, which leads to compounding errors." |
| Flow-JEPA — Huo, Song, Luo | 2026 | arXiv 2608.29029 (v2 09-26) | ✅ | latent JEPA, trajectory flow matching | 점별 회귀 → **궤적 수준 conditional flow matching**; Gaussian source 가 perturbed 궤적에 노출 | planning 86→92 / 67→86 | "결정론 AR 회귀가 취약한 부분" 지지 |
| NWM 2412.03572 · Vid2World 2505.14357 · DINO-WM 2411.04983 · DINO-world 2507.19468 | — | — | ✅ | — | drift 미논의 / planning 성공만 | — | DINO-WM/-world = 우리 설정에서 **closure 미측정** |

### 5-2. 처방 분류 (8 계열) — 우리 설계 원칙과의 대응
1. **입력 교란** (잡음·scheduled sampling): GNS, MeshGraphNets, Scheduled Sampling, STORM, Flow-JEPA, VGGT-World. frozen latent 에서는 VGGT-World 뿐. ⚠ 등방 잡음은 L1 회귀의 **구조적** off-manifold 오차를 모사하지 못할 수 있다.
2. **K-step rollout 손실** (latent): PlaNet overshooting, TD-MPC2 (◐), V-JEPA 2-AC (T=2), Semigroup-JEPA, pushforward, Diffusion Forcing. 우리 AR 팔이 여기 속하고 **한 걸음은 완벽 · 되먹임은 2–3 칸** — 이 계열만으로는 안 닫힌다는 것이 우리 결과.
3. **on-policy 자기 rollout 학습 / distillation**: Professor Forcing, Self Forcing, Causal-rCM, Rolling Forcing, Mask Forcing. 전부 pixel diffusion.
4. **결정론 회귀 → 확률/생성 head**: 이산 토큰 (IRIS, Genie, GAIA-1, MineWorld — 양자화 = 매 step manifold 투영), 범주형/VAE latent (DreamerV3, STORM, Δ-IRIS), diffusion/flow head (GAIA-2, NWM, Cosmos, Rolling Diffusion, Flow-JEPA, VGGT-World, FlowWM). **closure 를 가장 직접 겨냥한 계열.**
5. **재귀 회피 — 지평 직접 예측**: VLWM, DFoT/Rolling Forcing (다프레임 joint denoising), NWM 시간 offset (◐). latent 에서는 VLWM.
6. **문맥 관리** (메모리·eviction·anchor): WorldMem, Context-as-Memory, Long-term Spatial Memory, StableWorld, Rolling Forcing attention sink, Genie 3. 증상은 같고 기제는 직교.
7. **추론 시 교정·일관성**: Cycle-World, History Guidance, Pathwise TTC 2602.05871 (◐), FreqForcing 2607.27110 (◐).
8. **상태와 출력의 분리 / 재인코딩**: Δ-IRIS (연속 상태 토큰 vs 이산 delta 출력), RDR (decode→re-encode), DPWM. **우리 원칙 2 ("굴리는 상태와 렌더 출력을 분리") 의 선례 계열.**

**어디에 서는가.** frozen encoder + latent rollout + drift 보고를 다 갖춘 건 V-JEPA 2-AC · Semigroup-JEPA · RDR · VGGT-World (약하게 DINO-WM/-world) 뿐이고, 어느 것도 teacher-forced vs rollout 격차를 **predictor 출력 공간의 성질 (closure)** 로 측정하지 않았다. "teacher-forced accuracy does not transfer to rollouts" 를 frozen-encoder 설정에서 문장으로 쓴 논문은 없다 — 가장 가까운 것이 VLWM · VGGT-World.

---

## 6. (e) what/where 분리 · 명시적 운동 · 메모리 · permanence 벤치마크

P-25 §2-7 (Loci-Looped, OPNet, RAM, TCOW, TSA, SAMURAI, PLATO/ADEPT) 과 S-26 §2 (StEvo, WRBench, MemoBench, PersistBench, WROP, R2M, LoopBench, Permanence Fields, Neural Voxel Dynamics, GOT-JEPA) 는 반복하지 않는다.

### 6-1. slot / object-centric dynamics

| 논문 | 연도 | venue / arXiv | 확인 | 무엇을 분리 · 기제 | 가림/permanence | 숫자 | 우리 원칙과 |
|---|---|---|---|---|---|---|---|
| **G-SWM** — Lin, Wu, Peri, Fu, Jiang, Ahn | 2020 | ICML 2020 / 2010.02054 | ✅ 본문 | 물체별 `{z_what, z_where, z_pres, z_depth}` ("appearance, position, presence, and depth"), 전부 물체별 RNN `z_state` 에서 생성; `z_what` 을 `z_where` 에 inverse spatial transformer 로 렌더 | OCCLUSION · 2-LAYER 설정 명시; "occlusion can still be modeled as an implicit interaction between the objects where one object makes another one invisible" — 가림 중 `z_pres`/`z_state` 가 전파돼 재검출 없이 지속 | Table 2, 10 step 위치 오차 합, OCCLUSION: **0.072 vs SCALOR 0.272 vs STOVE 0.387** | **원칙 1 의 가장 순수한 선례** (what 은 보관, where 는 상태 RNN 으로 굴림) + 원칙 2·3. 차이: 픽셀 생성·toy·scratch. 우리는 frozen ViT-H latent 위 |
| **SlotFormer** — Wu, Dvornik, Greff, Kipf, Garg | 2023 | ICLR 2023 / 2210.05861 | ✅ 본문 | SAVi/STEVE slot 위 AR Transformer. **위치는 slot 안에 얽혀 있음** (시간 P.E. 만); 굴리는 상태 = slot 열, 렌더 = frozen slot decoder | "SlotFormer can handle occlusions or disappearing of objects **during burn-in frames** by attending to other timesteps where the objects are visible" (App. E) — **예측 구간 permanence 미검사** | CLEVRER predictive VQA 96.5 / 93.3 (+3.4 / +6.0 over Aloe); OBJ3D LPIPS 0.08 vs G-SWM 0.10 | 원칙 2·3 선례, **원칙 1 아님**. slot 얽힘 + frozen encoder 창 = 우리 reach/permanence 실패 지점 |
| OCVP — Villar-Corrales, Wahdan, Behnke | 2023 | ICIP 2023 / 2302.11850 | ✅ 초록 | 시간 동역학 (물체별) vs 상호작용 (물체 간) 두 transformer 로 분리 | 없음 | 초록에 수치 없음 | 직교 축 (self vs interaction) |
| SOLD — Mosbach et al. | 2025 | ICML 2025 / 2410.08822 | ✅ 초록 | slot 구조 latent dynamics 로 MBRL | 없음 | DreamerV3·TD-MPC2 초과 (초록) | 원칙 2·3 (RL) |
| **Dual-State Slot Attention** — Tran et al. | 2026 | arXiv 2606.12601 | ✅ 초록 | slot 을 **per-frame appearance 상태 + identity 상태 (재귀 전이)** 로 분리; 한 벡터에 둘을 넣으면 "an objective conflict that leads to slot swapping" | "rapid motion and partial occlusion" 겨냥 (MOVi-C/D, YT-VIS; 수치 미확인) | — | encoder 쪽 2026 유사체 (identity vs appearance; 우리는 content vs position). "두 역할을 한 벡터에 얽으면 뒤바뀜/오배치" 진단 공유 |
| SAVi++ 2206.07764 · Causal-JEPA 2602.11389 · STICA 2511.14262 · "Where It Breaks" 2511.06136 · Better Slots Better Worlds 2608.12078 | 2022–26 | — | ◐/✅ | slot encoder / 물체 단위 masking JEPA / OC-WM 분석 | 없음 | — | 배경 |

### 6-2. flow / displacement 조건 예측 (원칙 1 의 계보)

| 논문 | 연도 | venue / arXiv | 확인 | 무엇을 분리 · 기제 | 숫자 | 우리 원칙과 |
|---|---|---|---|---|---|---|
| **MCnet** — Villegas et al. | 2017 | ICLR 2017 / 1706.08033 | ✅ 초록 | content 스트림 (마지막 프레임) + motion 스트림 (프레임 차, ConvLSTM); "predicting the next frame reduces to converting the extracted content features into the next frame content by the identified motion features" | KTH/Weizmann/UCF SOTA (수치 없음) | 픽셀 층위 원칙 1 |
| **DrNet** — Denton, Birodkar | 2017 | NeurIPS 2017 / 1705.10915 | ✅ 초록 | 프레임 = 정지 content + 시변 pose; LSTM 은 **pose 만** 굴림, decoder 가 content+pose 렌더 | "coherently generate hundreds of steps into the future" | 원칙 1·2 (굴리는 변수 = pose 만이라 closure 가 구조로 보장) |
| **SDC-Net** — Reda et al. | 2018 | ECCV 2018 / 1811.00684 | ✅ 초록 | 픽셀별 motion vector + kernel, "applying the kernel at a displaced location in the source image"; "Resampling based on flow is insufficient because it cannot deal with disocclusions." | SSIM 0.904 / 0.918 | displacement 조건 렌더 = 원칙 1; **disocclusion 한계** 가 우리가 warping 대신 상태 (원칙 2) 를 두는 이유 |
| LaMD — latent motion diffusion | 2023 | IJCV / 2304.11603 | ✅ 초록 | "latent motion generation and video reconstruction" 으로 분해; diffusion 은 motion 에만 | — | 원칙 1·2 |
| MC-JEPA — Bardes, Ponce, LeCun | 2023 | arXiv 2307.12698 | ✅ 초록 | flow + content 공동 SSL (공유 encoder); "content features, which do not capture object motion or location" | — | 동기 문장; predictor 분리는 아님 |
| ATM 2401.00025 · Im2Flow2Act 2407.15208 · DVF 1702.02463 · Go-with-the-Flow 2501.08331 · Motion Forcing 2603.10408 · TokenMotion 2504.08181 · DeCo-VAE 2511.14530 | 2017–26 | — | ✅/◐ | 점 궤적·flow 를 중간 표현으로; 운동/내용 분리 latent | ATM: 130+ 과제 평균 +80 % (초록) | "where 를 굴리는 양으로" 의 로봇·생성형 근거. permanence 없음 |

### 6-3. 메모리 · state-space

| 논문 | 연도 | venue / arXiv | 확인 | 기제 | 가림/permanence | 숫자 | 우리 원칙과 |
|---|---|---|---|---|---|---|---|
| WorldMem — Xiao et al. | 2025 | NeurIPS 2025 / 2504.12369 | ✅ 본문 | 프레임+pose+시각 메모리 뱅크, FOV 겹침 검색, memory attention | 장면 revisit 용; "when views are blocked by obstacles… relying solely on view overlap may be insufficient" | Minecraft beyond-context PSNR 17.32→23.98, LPIPS 0.4376→0.1429 | 원칙 3 선례이나 메모리 = 프레임 뱅크 (pose 키), 움직이는 가려진 물체 아님 |
| Context-as-Memory — Yu et al. | 2025 | SIGGRAPH Asia 2025 / 2506.03141 | ✅ 초록 | 과거 프레임을 문맥으로 연결 + FOV 검색 | 없음 | — | 대조: history = 프레임 (우리: 상태) |
| **Long-term Spatial Memory** — Wu … Wetzstein | 2025 | arXiv 2506.05284 | ✅ 본문 | 정적 장면을 point map (TSDF) 으로; "**This fusion process inherently filters out dynamic elements in the scene**" | **동적 물체를 설계상 제외** | revisit PSNR 19.10 vs DaS 12.01 | 상태/렌더 분리 (원칙 2·3) 를 **정적 내용에만**. 우리 표적이 바로 그들이 거른 동적 부분 |
| Mechanistic View on Video Generation as WMs — Wang et al. | 2026 | arXiv 2601.17067 | ✅ 초록 | 상태 구성 (implicit 문맥 관리 vs explicit latent 압축) vs 동역학 모델링 taxonomy | 없음 | — | 원칙 2·3 의 틀 인용 (P-25 §2-2 에도) |
| StateSpaceDiffuser 2505.22246 · WorldPack 2512.02473 · MemLearner 2606.31734 · Geometry-Aware Implicit Memory 2606.02436 · AnchorWeave 2602.14941 · PermaVid 2606.16449 · MIND 2602.08025 | 2025–26 | — | ◐ | SSM 상태·압축·학습 메모리·벤치마크 | 미확인 | — | "창 밖 지속 상태" 가 2025–26 의 축이라는 증거. 움직이는 가려진 물체의 permanence 주장은 없음 |

### 6-4. permanence 벤치마크 (video model 대상) · what/where JEPA

| 논문 | 연도 | venue / arXiv | 확인 | 무엇 | 가림/permanence | 숫자 | 우리와 |
|---|---|---|---|---|---|---|---|
| **CATER** — Girdhar, Ramanan | 2020 | ICLR 2020 / 1910.04744 | ✅ HTML | Snitch Localization: 가림·재귀 containment 뒤 최종 칸 예측 | "Performance drops if the snitch is 'contain'-ed by another object in the end, and the tracker is the worst affected by it." | best top-1 57.4 % (R3D, 64f, stride 8 — **표 변형이 많아 인용 전 재확인**) | 알려진 what 의 where 를 containment 아래 묻는 표준 testbed. 외부 검증 후보 |
| PermaTrack — Tokmakov, Li, Burgard, Gaidon | 2021 | ICCV 2021 / 2103.14258 | ✅ 초록 | CenterTrack + 재귀, 합성 데이터의 **보이지 않는 물체 GT** 로 감독 | "once an object is recognized, we are aware of its physical existence and can approximately localize it even under full occlusions." | KITTI/MOT17 SOTA (수치 없음) | permanence = 재귀 상태 (원칙 3) + 숨은 위치 감독. 지도학습 |
| PSG-JEPA (Is Forward Prediction Enough?) — Yan et al. | 2026 | arXiv 2608.06799 | ✅ 초록 | (S-26) + 새 문장: "forward-prediction objectives do not explicitly enforce reliable identifiability of … state changes from latent pairs" | 없음 | — | 원칙 1 지지 (예측 목표가 "어디로" 를 강제하지 않음). 로봇 중심 |
| Probabilistic JEPA = HMM 2608.13621 · BiJEPA 2603.00049 · Semantic Slots 2608.21636 · Slot-MPC 2605.14937 · TextOCVP 2502.11655 | 2025–26 | — | ◐ | — | — | — | 배경. **"Where-JEPA / Slot-JEPA / motion-centric JEPA" 라는 이름의 논문은 없다** (검색 실패) |

**정리.** 원칙 1 (what/where) 은 픽셀 시대와 G-SWM 에 선례가 뚜렷하고, 원칙 2·3 은 2025–26 메모리 계열에 있지만 그 메모리는 카메라 pose 로 색인된 프레임 뱅크나 정적 point map 이다. **frozen latent encoder + content/displacement 분리 + 물체 수준 지속 상태**를 한 번에 갖춘 선행은 없고, 이 절의 벤치마크 (CATER, PermaTrack) 는 자기지도 latent 예측을 대상으로 하지 않는다.

---

## 7. (f) egocentric anticipation · ego-motion

S-26 §3 (EK100/Ego4D STA/LTA/EgoExo4D 지평 정의, JFAA, TAP-JEPA, VISTA, FROST-STA, Ego4D LTA 2025, TrajPilot, Human-JEPA, AVT 기억 기반) 은 반복하지 않고 **수치·문장으로 확인된 것**만 더한다.

### 7-1. latent 미래 예측이 anticipation 에 얼마나 보태는가 (EK / EGTEA)

| 논문 | 연도 | venue / arXiv | 확인 | 무엇 | 숫자 · 주장 (미래 예측의 몫 · 지평 곡선) | 우리와 |
|---|---|---|---|---|---|---|
| **RULSTM** — Furnari & Farinella | 2020 | TPAMI 2020 / arXiv 2005.02190 (ICCV 2019 판 1905.09035) | ✅ 본문 (pdftotext) | R-LSTM (과거 요약) + U-LSTM (미래 unroll), RGB/flow/obj 모달 attention; τ_a = 2 → 0.25 s (step 0.25) | **지평 곡선** (Table 1, EK-55 val Top-5 action): 29.44 / 30.73 / 32.24 / 33.41 / **35.32** / 36.34 / 37.37 / 38.98 (τ_a 2 → 0.25 s); EGTEA 56.82 → 74.28. **미래 특징 회귀 기준선 (DMR, ED) 은 egocentric 에서 열세**: "DMR and ED, which are explicitly trained to anticipate future representations, achieve sub-optimal Top-5 action anticipation accuracy as compared to methods trained to predict future actions directly from input images" / "the unsupervised pre-training based on the regression of future representations is not beneficial … This might be due to the fact that anticipating future representations is very challenging in the case of egocentric video, in which the visual content tend change continuously because of the **mobility of the camera**." EK-55 ED\* (미래 특징 사전학습 없음) 25.20 vs ED 25.75 @1 s; **EGTEA ED\* 60.18 vs ED 50.22 (사전학습이 해침)**; 3인칭 ActivityNet ED 72.93 vs ED\* 65.84 (도움) | **가장 직접적인 선례 둘**: (i) 미래 특징 회귀는 egocentric anticipation 에 거의/전혀 안 보탠다, (ii) 원인으로 **카메라 운동**을 지목. ego vs 3인칭의 뒤집힘 (EGTEA/EK vs ActivityNet) 은 "ego-motion 이 latent 미래 예측을 깨뜨린다" 의 가장 깨끗한 공개 증거 — 우리 §4 원칙 4 (EK100 reach 2.2 천장, ego-motion) 의 문헌 근거 |
| **AVT** — Girdhar & Grauman | 2021 | ICCV 2021 / arXiv 2106.02036 | ✅ HTML v1 | causal transformer + L_cls (매 step 다음 행동) + **L_feat = Σ‖ẑ_t − z_{t+1}‖²** (미래 특징 회귀) | Table 8 (EK100 val, action R@5): TSN backbone base 10.1 / +L_cls 11.5 / +L_feat 13.7 / both 13.6; **AVT-b backbone base 13.1 / +L_cls 14.4 / +L_feat 13.0 / both 14.4** → 강한 backbone 에서 미래 특징 손실의 몫 **−0.1 (단독) · +0.0 (L_cls 위)**. ⚠ 본문 문장 "L_feat [more effective] for AVT-b" 가 HTML 표와 어긋난다 — camera-ready PDF 확인 전 문장 인용 금지. Fig. 4 는 **문맥 길이** 스윕이지 지평 스윕이 아님 | "anticipation 을 위한 미래 특징 예측" 의 정전 논문이 자체 ablation 에서 latent 미래 손실 ≈ 0 (강한 backbone) — 우리 EK100 predictor 몫 +0.6~+2 와 같은 규모 |
| **ImagineRNN** — Wu, Zhu, Wang, Yang, Wu | 2021 | IEEE TIP 30 / arXiv 2101.04924 | ✅ 본문 (pdftotext) | RULSTM 위에 미래 특징 사슬 예측; L2 대신 contrastive; residual (특징 **차이**) 예측 | EK-55 S1 test Top-5 action @1 s: RULSTM 33.73 → L2 34.66 → contrastive 34.98 (**+1.25**); S2 unseen 21.10 → **20.79 (L2, 악화)** → 22.19. Ablation Table IV (val): 35.24 → full 35.38 (**+0.14**). EGTEA 1.0/0.75/0.5 s: 66.71/68.54/72.32 vs RULSTM 66.40/68.41/71.84 | 미래 latent "상상" 의 이득이 1 pt 미만, L2 회귀는 unseen kitchen 에서 ≤ 0 — 짧은 지평에서 "미래 latent 예측 ≈ 복사" 지지 |
| **SRL** — Qi et al. | 2021 | TPAMI 2021 / arXiv 2111.11631 | ✅ 초록 | 재귀 미래 표현 예측 + "novel information" 강조 contrastive + 재가중 | "As a standard future activity anticipation paradigm, recursive sequence prediction suffers from the accumulation of errors." (수치 미추출) | 재귀 latent rollout 의 오차 누적을 anticipation 문헌이 명시 — closure 의 anticipation 판 |
| MeMViT — Wu et al. | 2022 | CVPR 2022 / arXiv 2201.08383 | ✅ 초록, 수치 ❓ | 캐시 메모리로 30× 긴 시간 지원 (+4.5 % 연산) | EK100 anticipation SOTA (수치 미확인) | 이득은 **더 긴 과거**에서, 미래 예측에서가 아님 |
| InAViT — Roy, Rajendiran, Fernando | 2024 | WACV 2024 / arXiv 2211.14154 | ✅ 초록 | hand-object 상호작용 토큰 | 제출 당시 EK100 1위 (+3.3 % MT5R) | 지각 (encoder) 쪽 개선, 미래 예측 없음 |
| DiffAnt | 2023 | arXiv 2311.15991 (venue 미확인) | ✅ HTML | 미래 행동 **라벨**을 latent diffusion 으로 생성 (dense anticipation 프로토콜) | EGTEA mAP 77.3/83.5/61.4 vs Anticipatr 76.8/83.3/55.1; "particularly strong improvements for longer anticipation windows" (β 스윕) | 생성형 라벨 anticipation; 지평 스윕은 있으나 다른 프로토콜 |
| Fernando & Herath (Jaccard) 2105.12414 · AntGPT (ICLR 2024) · PALM 2306.16545 | 2021–24 | — | ◐ | 과거–미래 상관 / LLM 이 Ego4D LTA 20 행동 생성 | Ego4D test ED: AntGPT 0.877/0.650/0.650, PALM 0.850/0.647/0.612 | (S-26) LTA SOTA 는 언어 공간 |
| V-JEPA 2 (2506.09985) · TAP-JEPA (2606.00662) | 2025–26 | — | ✅ | encoder + predictor 토큰 연결 후 attentive probe | V-JEPA 2 **Table 20** (내가 로컬 PDF 로 확인): encoder 39.1 / predictor 20.2 / both **39.7** (+0.6) — 하위 조사가 "ablation 없음" 이라 보고했으나 **있다** (App. D.2). TAP-JEPA 는 ablation 없음 (본문 확인) | 지평 스윕 (1/2/4/10 s) 은 Fig. 18 뿐이고 "sharply decreases" 만 서술 |

**지평 곡선 (accuracy vs τ_a) 을 낸 논문**: RULSTM Tables 1/3/4/5 (2 → 0.25 s) ✅ · ImagineRNN Tables III/V ✅ · V-JEPA 2 Fig. 18 (1/2/4/10 s, 수치 미전사) ✅ · DiffAnt (β, 다른 프로토콜) ✅. **AVT 는 없다** (문맥 길이 스윕만).
**정리.** 2019–2021 의 anticipation 문헌은 이미 (i) 미래 특징 회귀의 몫이 0~1 pt 이고 (RULSTM, AVT, ImagineRNN), (ii) 원인이 카메라 운동임을 (RULSTM) 적었다. 2025–26 의 V-JEPA 2 / TAP-JEPA / JFAA 는 이 사실을 잊고 predictor 미래 토큰을 검증 없이 붙인다. 우리 "복사 대비 Δ" 는 새 발견이 아니라 **이 계보의 재확인 + 사전학습 predictor 로의 확장** 으로 쓴다.

### 7-2. ego-motion 인수분해 · ego 조건 예측

| 논문 | 연도 | venue / arXiv | 확인 | 무엇 | 주장 | 우리와 |
|---|---|---|---|---|---|---|
| Navigation World Models — Bar et al. | 2025 | CVPR 2025 / 2412.03572 | ✅ 본문 | SD-VAE latent 위 CDiT; 행동 = (u ∈ ℝ² 전후/좌우, φ yaw) + 시간 shift k; SCAND/TartanDrive/RECON/HuRoN + "unlabeled Ego4D videos, where the only action we consider is time shift" | "the navigation actions can be approximated based on the change in the agent's location" | ego-motion 을 **명시 조건**으로 주고, Ego4D 는 ego-motion 라벨 없이 씀 — 우리가 겨냥하는 공백 (ego-motion 을 예측하거나 인수분해) |
| **PEVA** — Bai, Tran, Bar, LeCun, Darrell, Malik | 2025 | NeurIPS 2025 / 2506.21552 | ✅ 초록 | Nymeria 에서 상대 3D 신체 pose 조건 AR diffusion transformer | "Predict Ego-centric Video from human Actions … given the past video and an action represented by the relative 3D body pose" | 신체/머리 pose = ego-motion 신호; 픽셀 생성 |
| EgoAgent — Chen et al. | 2025 | arXiv 2502.05857 | ✅ 초록 | "joint embedding-action-prediction architecture"; 상태/행동 토큰 interleave; Ego4D | JEPA 를 상태+행동 공동 예측으로 확장 | JEPA 식 egocentric predictor 가 행동을 토큰으로 다룸 |
| **TrajPilot** — Jun, Nguyen-Truong, Seminara, Torresani | 2026 | arXiv 2605.20388 | ✅ 초록 | 미래 **카메라 (머리) 궤적**을 조건/중간 latent 로; 문맥에서 궤적 예측 가능; Ego-Exo4D, Ego4D GoalStep, EgoPER, EK100 | "the future camera trajectory, the path the head carves through space, lets the model commit to one of those futures"; 이득이 "widen with horizon"; "under RGB-only camera-pose estimation" 에서도 | (S-26) + **"ego-motion 이 긴 지평의 정보" 의 가장 강한 지지** — 머리 경로 예측이 2–3 s anticipation 이득을 산다 |
| EgoWM — Bagchi et al. | 2026 | arXiv 2601.15284 | ✅ 초록 | 사전학습 video diffusion → 행동 조건 ego WM (경량 conditioning) | 3-DoF 로봇 → 25-DoF 휴머노이드 | 픽셀 생성 경쟁 framing |
| EgoExo-WM — Tran, Martín-Martín, Grauman | 2026 | arXiv 2605.15477 | ✅ 초록 | exo 영상에서 신체 pose 를 행동으로 추출, exo→ego 변환으로 ego WM 학습 | "constrained by … inherent partial observability of humans' physical actions" | ego-motion 이 숨은 행동 변수라는 동기 |
| RESELF 2609.01276 · EgoGenesis 2607.28243 · EgoVid-5M 2411.08380 · EgoControl 2511.18173 · CameraCtrl 2404.02101 | 2024–26 | — | ✅ 초록 / ❓ 제목 | 장면·카메라 궤적·신체 운동을 **명시 분리** (RESELF) / 카메라·pose 조건 생성 | — | 분야가 camera vs body vs scene 을 명시 인수분해하는 추세 |
| Li, Ye, Rehg, Delving into Egocentric Actions | 2015 | CVPR 2015 | ◐ (CVF PDF 절단) | 머리 운동 보상 후 flow 특징 | "The wearer's head is often the dominant source of flow in an egocentric video, but is often unrelated to the action being performed" (요약 문구) | "카메라 운동이 지배" 인용은 이 문장 (◐) + RULSTM 의 "mobility of the camera" (✅) 로. **정량 통계 (flow 중 카메라 운동 비율) 를 낸 논문은 못 찾았다** |

---

## 8. (g) "encoder = 지각, predictor = 동역학/세계 지식" — 선행의 정확한 문장

확인 표기: ✅ 원문 (또는 공식 재게재) 에서 verbatim · ◐ 요약/2차 전사 · ❓ 미확인. **주의: Garrido et al. 2025 본문에 "the predictor acts as a world model" 이라는 문장은 없다** (전문 grep, "world model" 은 참고문헌 제목뿐). 그 표현을 그들에게 귀속시키지 말 것.

| 출처 | 확인 | 문장 (verbatim) | 우리 쓰임 |
|---|---|---|---|
| **LeCun 2022**, *A Path Towards Autonomous Machine Intelligence* v0.9.2 (OpenReview BZ5a1r-kVsf; §3 모듈 설명) | ✅ Meta AI blog 재게재 (OpenReview 는 인증 벽; 절/쪽 ❓) | "The perception module receives signals from sensors and estimates the current state of the world." / "The world model module constitutes the most complex piece of the architecture. Its role is twofold: (1) to estimate missing information about the state of the world not provided by perception, and (2) to predict plausible future states of the world." / world model = "a kind of simulator of the part of the world relevant to the task at hand." / JEPA: "A predictor module is trained to predict s_y from s_x." | **기능 분리의 정본 인용.** (1)+(2) 가 우리 permanence (missing information) + reach (future states) 와 정확히 대응 — "비어 있는 자리" 를 LeCun 의 정의로 명명할 수 있다 |
| LeCun, Temple 강연 슬라이드 | ✅ pdftotext (슬라이드, 논문 아님) | "Perception module: estimates the state of the world s[0] = Enc(x)" / "Simulation: the world model predicts one or several likely sequence of world state representations resulting from the proposed action sequence." | 보조 |
| **Ha & Schmidhuber 2018**, *World Models* (worldmodels.github.io, "Agent Model") | ✅ | "The role of the V model is to learn an abstract, compressed representation of each observed input frame." / "While it is the role of the V model to compress what the agent sees at each time frame, we also want to compress what happens over time. For this purpose, the role of the M model is to predict the future." / "The M model serves as a predictive model of the future z vectors that V is expected to produce." | V/M = encoder/predictor 의 원형 문장 |
| **Hafner et al., PlaNet** (ICML 2019, 1811.04551) | ✅ pdftotext | "a purely model-based agent that learns the environment dynamics from images and chooses actions through fast online planning in latent space. To achieve high performance, the dynamics model must accurately predict the rewards ahead for multiple time steps." / §3 "Transition model: s_t ∼ p(s_t \| s_{t−1}, a_{t−1})" | "여러 step 을 맞혀야 한다" 는 기능 요건 |
| Hafner et al., Dreamer (ICLR 2020, 1912.01603) | ✅ 초록 | "Learned world models summarize an agent's experience to facilitate learning complex behaviors." | 보조 |
| **Assran et al., V-JEPA 2** (2506.09985) | ✅ 로컬 PDF | §1 "learn to both represent video observations and learn a predictive model for world dynamics in this learned representation space" / §2.1 "the encoder, E_θ(·), which extracts video representations" · "a predictor, P_ϕ(·), which predicts the representation of masked video parts" / §3 "we freeze the encoder weights and learn a new action-conditioned predictor, V-JEPA 2-AC" / §3.1 "an autoregressive model that predicts representations of future video observations conditioned on control actions" (+ §1-3 의 천장·EK100 문장) | 저자 자신의 역할 분담 문장. "predictive model for world dynamics" 가 predictor 에 걸린다 |
| **Garrido et al. 2025** (2502.11831) | ✅ pdftotext | "an encoder (a neural network) that extracts representations from a video, and a predictor (also a neural network) that predicts the representation of an artificially masked part of the video, such as a randomly masked spatiotemporal block, random pixels, or future frames." / "prediction of future world states should be done in the model's learned abstract, internal representation, and not in terms of low-level, pixel-based prediction." | 역할 분담 + "masked part … or future frames" 가 사전학습 predictor 의 실제 과제 (future-only 아님) 임을 저자 문장으로 |
| **Craik 1943**, *The Nature of Explanation*, p. 61 | ✅ 2차 전사 (jhamrick.github.io; 영국식 "utilise") | "If the organism carries a 'small-scale model' of external reality and of its own possible actions within its head, it is able to try out various alternatives, conclude which is the best of them, react to future situations before they arise, utilise the knowledge of past events in dealing with the present and future, and in every way react in a much fuller, safer, and more competent manner to the emergencies which face it." | 서론 첫 문장 후보 (V-JEPA 2 도 Craik 1967 인용) |
| **Tolman 1948**, Psychol. Rev. 55(4) | ✅ 전사 (CSULB) | "the central office itself is far more like a map control room than it is like an old-fashioned telephone exchange." … "a tentative, cognitive-like map of the environment. And it is this tentative map, indicating routes and paths and environmental relationships, which finally determines what responses, if any, the animal will finally release." | permanence·navigation 배경 |
| **Wolpert & Kawato 1998**, Neural Networks 11 | ✅ pdftotext | "A forward dynamic model of the arm, for example predicts the next state (e.g. position and velocity) given the current state and motor command." / Fig. 1 "Each forward model predicts the next state based on the motor command and current state." | forward model = predictor 의 정의 |
| **Clark 2013**, BBS 36(3) | ✅ Cambridge Core 초록 | "Brains, it has recently been argued, are essentially prediction machines. They are bundles of cells that support perception and action by constantly attempting to match incoming sensory inputs with top-down expectations or predictions." | 예측 부호화 배경 |
| Rao & Ballard 1999, Nat. Neurosci. 2:79 | ◐ (검색 요약 문장) | "feedback connections from a higher- to a lower-order visual cortical area carry predictions of lower-level neural activities, whereas the feedforward connections carry the residual errors between the predictions and the actual lower-level activities." | 인용 전 원문 확인 |
| Lotter, Kreiman, Cox, PredNet (ICLR 2017, 1605.08104) | ✅ 초록 | "prediction of future frames in a video sequence as an unsupervised learning rule for learning about the structure of the visual world" / "prediction represents a powerful framework for unsupervised learning, allowing for implicit learning of object and scene structure." | 배경 |
| Schmidhuber 1990 (TR FKI-126-90) · Sutton 1991 Dyna (SIGART Bull. 2(4)) | ◐ | 서지만 확인 / Dyna: 학습이 "updates a model of the effects of the agent's actions on the world" (요약) | verbatim 미확보 — 인용 시 재확인 |
| Friston · Bardes et al. V-JEPA 2024 역할 문장 | — | 미조사 (예산) | — |

---

## 9. 가장 위험한 선행 5 편 — 스쿠프로 읽힐 수 있는 것과 정확한 delta

| # | 선행 | 그들이 가진 것 | 우리와의 정확한 delta |
|---|---|---|---|
| 1 | **Human-JEPA** (2608.21160, 2026-08) ✅ | JEPA predictor 에 **copy-last latent 귀무**를 붙인 유일한 논문: 0.873 vs 0.798 cosine, 마진 0.057 (0.5 s) → 0.087 (2 s) | (i) 자기 predictor (사람 영상 이어 학습) 이지 릴리즈 predictor 가 아니다, (ii) 20 clip · latent cosine 이지 벤치마크 **점수** 가 아니다, (iii) "복사보다 낫다" 로 읽지 우리처럼 "벤치마크가 복사로 맞춰진다" 로 읽지 않는다. 우리는 IntPhys 1 · 2 · EK100 점수에서 복사 대비 Δ 를 재고 (IntPhys 1 +4, IntPhys 2 ≈ 0, EK100 +2), reach·permanence·closure 로 왜 그런지 기제까지 잇는다. ⚠ 우리 논문에 **반드시 인용** — 빠지면 "복사 기준선은 이미 있다" 로 맞는다 |
| 2 | **Garrido et al., Latent Action WM In The Wild** (2601.05230, 2026-01, FAIR) ✅ | frozen V-JEPA 2-L + 실영상 (action-free) 새 predictor. 한계: "This representation space was not designed with prediction in mind" | 같은 구성 (frozen V-JEPA 2 + 실영상 학습) 이라 "새 predictor" 자체는 새롭지 않다. delta = (i) 그들은 latent action 이 목적이고 open-loop 은 LPIPS 2 s 뿐 · 복사 귀무 없음, (ii) 우리는 **무엇을 채워야 하는지** (reach · permanence · closure) 를 먼저 재고 그 실패에서 설계를 도출한다, (iii) 그들의 한계 문장 ("예측을 염두에 두지 않은 공간") 을 우리는 검정한다 — encoder 에 정보가 있음 (속도 R² 0.996, probe 98–100) 을 보여 병목을 predictor 로 국소화 |
| 3 | **GEOPHYS** (2606.20707, 2026-06) ✅ 초록 | frozen **image** encoder 프레임별 embedding 기하 5 개로 IntPhys 2 93.3 % / LikePhys 98.3 %; V-JEPA 2 는 chance 근처 | "predictor 없이 IntPhys 2 가 풀린다" 는 우리 논지의 **가장 강한 외부 증거이자 가장 큰 반문** ("그럼 predictor 는 왜 필요한가"). delta: 그들은 plausibility **판별기** (VoE 점수 대체) 를 만들고, 우리는 **예측기** 를 만든다 — 판별이 아니라 미래를 요구하는 downstream (EK100 2–3 s, Ego4D) 에서 가치를 보여야 한다. ⚠ split/프로토콜 미확인 상태로 93.3 인용 금지 |
| 4 | **Physion** (2106.08261, NeurIPS D&B 2021) ✅ | "observed+simulated ≈ observed (p = 0.53)"; 위 chance 는 "cues in the initial movie segment" 때문 | 2021 년에 이미 "모델의 미래 굴림이 점수에 기여하지 않는다" 를 보였다. delta: 대상이 지도/자기지도 소형 dynamics (PhysNet 류) 이고, "지각이 병목" 으로 결론냈다. 우리는 **사전학습 latent predictor** 에서 반대 결론 ("지각은 충분, 예측이 비어 있다") 을 복사 귀무로 세운다. Physion 을 **서사 훅** 으로 쓰되 결론이 반대임을 명시 |
| 5 | **DINO-Foresight** (2412.11673, NeurIPS 2025) + **DINO-world** (2507.19468) + **Joseph et al. App. C.1.4** (2602.07050) ✅ | frozen 특징 위 미래 예측기 + Copy-Last 대조 (DINO-Foresight) / future-only predictor 도 1 s 에서 무너짐 + VoE 를 "sanity check" (DINO-world) / **layer 0 위에 predictor 를 학습해도 VoE 점수** (Joseph) | 셋을 합치면 "frozen encoder + 새 predictor + copy-last + VoE 는 예측력 인증이 아님" 이 이미 흩어져 있다. delta: 누구도 **같은 사전학습 predictor 를 벤치마크가 실제로 쓰는 방식 그대로** (one-shot 질의, Garrido 프로토콜, EK100 probe) 두고 복사와 비교하지 않았고, 실패를 reach·permanence·closure 로 국소화하지 않았다. Joseph C.1.4 는 우리 "점수 감사" 의 저자 측 증거로 인용 |

(보조 위험: **HERA** 2608.05523 — frozen V-JEPA 2 predictor 위 메모리 adapter, S-26 §1 · **VGGT-World** 2603.12655 — frozen 특징 + exposure-bias curriculum, 원칙 2 의 처방 선점 · **VLWM** 2606.21775 — 재귀 회피.)

---

## 10. 우리가 인용해야 할 문장 (English, 원문 확인분)

표기: **[V]** = 내가 이 세션에서 원문을 다시 열어 verbatim 확인 · **[A]** = 하위 조사가 원문 fetch 로 확인 · **[◐]** = 요약/초록 경유 (인용 전 재확인).

**역할·천장 (V-JEPA 2, 2506.09985) [V, 로컬 PDF]**
- "The capabilities of a representation-space world model, such as V-JEPA 2-AC discussed above, are inherently limited by the state information encoded in the learned representation space." (§5)
- "the EK100 task mostly requires strong semantic understanding, as opposed to forecasting capabilities." (App. D.2; Table 20: encoder 39.1 / predictor 20.2 / both 39.7)
- "autoregressive prediction suffers from error accumulation: the accuracy of the representation-space predictions decreases with longer autoregressive rollouts" (§4)
- "The model also predicts 'rinse sponge', which is the current action being performed, probably assuming that this action could still be going on after 1 second." (§6)

**VoE 점수가 예측을 요구하지 않음**
- Physion 2106.08261 [V]: "Vision model predictions from the observed+simulated readout protocol were, overall, no better than predictions from the observed protocol (p=0.53, Fig. 5D)." / "any above-chance performance for the vision models was likely due to having visual features that could discriminate some trial outcomes from cues in the initial movie segment."
- Joseph et al. 2602.07050 [V, 로컬 PDF, App. C.1.4]: "even early layers that give poor performance on only the encoder suddenly perform well (Layer 0). This is likely because the predictor itself is learning from the representation."
- GEOPHYS 2606.20707 [V, 초록]: "indicators of physical plausibility are implicitly captured by five geometric properties of the per-frame embeddings produced by frozen image encoders."
- IntPhys 2 2506.09849 [V, 로컬 PDF]: "The permanence condition appears to be the easiest for both models and humans, as objects are not moving by themselves." / "the best performance is achieved with a window size of 16 and 6 as framerate (i.e. less than 3 seconds), which makes most samples from IntPhys 2 impossible." / "when the ball is supposed to reappear, the model is not more surprised for the impossible than possible videos"
- Physics-IQ 2501.09038 [A]: "Visual realism is uncorrelated with physical understanding (Pearson's r = -0.46, p=.249 not significant)"
- InfLevel README (allenai/inflevel) [◐]: "This tight control opens the door for designing heuristics that can obtain high performance."
- Physion++ 2306.15668 [A]: "Most models with property inference perform similarly to those without property inference, indicating that these models are not utilizing physical property inference."

**복사 기준선의 계보와 부재**
- Human-JEPA 2608.21160 [V]: "its predictions agree with the latents of the frames that actually followed at 0.873 cosine, against 0.798 for holding the last observed latent fixed" / "The margin over the static reference holds on every clip and widens with distance, from 0.057 half a second ahead to 0.087 two seconds ahead"
- Luc et al. 1703.07684 [A]: "The first baseline copies the last input frame to the output."
- AVT 2106.02036 [A]: "employing the anticipative training losses are imperative to obtain strong performance with AVT." (Table 7: AVT-b 13.1 → 14.4)
- TAP-JEPA 2606.00662 [A]: "Concatenate encoder tokens and predictor tokens along token dimension before downstream classification." (encoder-only ablation 없음)

**복사가 손실의 최저점 (predictor 가 조회한다)**
- Chen, Zhang, Wang 2607.27036 [A]: "When frame contents change mildly, directly copying adjacent frames enables smooth temporal consistency and favorable generation quality." / "such frame-copying behavior erodes the amount of valid information contained within latent representations."
- V-JEPA 2404.08471 [A]: "Masking a large continuous block that covers the full temporal dimension limits information leakage due to the spatial and temporal redundancy of videos."
- NextLat 2511.05963 [A]: "self-attention that enables ad-hoc lookups over past tokens" / "lack an inherent incentive to compress history into compact latent states"
- Yeom et al. 2606.07687 [A]: "most gains arise from natural-video temporal context, with feature-level latent prediction providing a smaller additional benefit."

**frozen encoder + 새 predictor (설계 근거)**
- Garrido et al. 2601.05230 [V]: "This encoder is kept frozen during training." / "This representation space was not designed with prediction in mind, which can hinder the inverse dynamics training, as well as the quality of the predictions in general."
- FlowWM 2606.29059 [A]: "This simplifies training and avoids objective collapse, but also limits the representational quality of the predicted futures to that of the encoder." / "Deterministic predictors trained with standard regression losses tend to average over these possibilities, producing predictions that may correspond to no valid future."
- Dreamer 4 2509.24527 [A]: "The dynamics model operates on the interleaved sequence of actions and representations produced by the frozen tokenizer."
- NWM 2412.03572 [A]: "After 8 seconds, predictions degrade due to accumulated errors and loss of context."

**closure / exposure bias**
- Scheduled Sampling 1506.03099 [A]: "This discrepancy between training and inference can yield errors that can accumulate quickly along the generated sequence."
- Professor Forcing 1610.09038 [A]: "encourage the dynamics of the recurrent network to be the same when training the network and when sampling from the network over multiple time steps."
- VLWM 2606.21775 [A]: "existing latent world models typically rely on one-step prediction and must be recursively rolled out for long-horizon planning, which leads to compounding errors."
- VGGT-World 2603.12655 [A, 부분]: "autoregressive rollout suffers from compounding exposure bias"
- StableWorld 2601.15281 [A]: "error accumulation originates from the same scene, where generated frames gradually deviate from the initial clean state."
- Cycle-World 2607.11836 [A]: "minor prediction deviations compound over time, inevitably leading to unconstrained generative drift, structural collapse, and severe visual degradation."
- Genie 3 blog [A]: "The model can currently support a few minutes of continuous interaction, rather than extended hours."

**what/where · 상태/렌더 분리**
- MCnet 1706.08033 [A]: "By independently modeling motion and content, predicting the next frame reduces to converting the extracted content features into the next frame content by the identified motion features"
- G-SWM 2010.02054 [A]: "{z_what, z_where, z_pres, z_depth} to represent appearance, position, presence, and depth of an object, respectively"
- SDC-Net 1811.00684 [A]: "Resampling based on flow is insufficient because it cannot deal with disocclusions."
- SlotFormer 2210.05861 [A, App. E]: "SlotFormer can handle occlusions or disappearing of objects during burn-in frames by attending to other timesteps where the objects are visible" (burn-in 한정)
- Dual-State Slot Attention 2606.12601 [A]: "encode both the per-frame appearance of an object and its identity across frames in a single slot vector, creating an objective conflict that leads to slot swapping."
- Long-term Spatial Memory 2506.05284 [A]: "This fusion process inherently filters out dynamic elements in the scene"
- PermaTrack 2103.14258 [A]: "once an object is recognized, we are aware of its physical existence and can approximately localize it even under full occlusions."
- PSG-JEPA 2608.06799 [A]: "forward-prediction objectives do not explicitly enforce reliable identifiability of robot-centric physical state from individual latents or state changes from latent pairs."
- Alrasheed et al. 2605.15618 [V, 로컬 PDF]: "A frozen V-JEPA 2 backbone with a lightweight attentive probe outperforms a fully fine-tuned VideoMAE and a supervised TimeSformer on corruption and occlusion robustness."

**anticipation 에서 미래 latent 예측의 몫 · ego-motion**
- RULSTM 2005.02190 [A, pdftotext]: "DMR and ED, which are explicitly trained to anticipate future representations, achieve sub-optimal Top-5 action anticipation accuracy as compared to methods trained to predict future actions directly from input images" / "anticipating future representations is very challenging in the case of egocentric video, in which the visual content tend change continuously because of the mobility of the camera." (EGTEA ED\* 60.18 vs ED 50.22 @1 s)
- AVT 2106.02036 [A, HTML Table 8]: AVT-b base 13.1 / +L_cls 14.4 / +L_feat 13.0 / both 14.4 (⚠ 본문 문장과 표 불일치 — PDF 확인 전 문장 인용 금지)
- SRL 2111.11631 [A, 초록]: "As a standard future activity anticipation paradigm, recursive sequence prediction suffers from the accumulation of errors."
- TrajPilot 2605.20388 [A, 초록]: "the future camera trajectory, the path the head carves through space, lets the model commit to one of those futures"
- NWM 2412.03572 [A]: "unlabeled Ego4D videos, where the only action we consider is time shift"
- Li, Ye, Rehg CVPR 2015 [◐]: "The wearer's head is often the dominant source of flow in an egocentric video, but is often unrelated to the action being performed"

**기능 분리 (encoder = 지각 · predictor = 세계 모델)** — §8 표 참조. 핵심 셋:
- LeCun 2022 [A, Meta 재게재]: "The world model module … Its role is twofold: (1) to estimate missing information about the state of the world not provided by perception, and (2) to predict plausible future states of the world."
- Ha & Schmidhuber 2018 [A]: "the role of the M model is to predict the future."
- Garrido et al. 2502.11831 [A, pdftotext]: "a predictor (also a neural network) that predicts the representation of an artificially masked part of the video, such as a randomly masked spatiotemporal block, random pixels, or future frames."

---

## 11. Related-work paragraph (English, ≈ 200 words)

> **Is the predictor doing the predicting?** Frozen video encoders paired with a latent predictor are the standard recipe for video world models, whether the predictor is the pretrained V-JEPA 2 network used zero-shot for violation-of-expectation (VoE) scoring [Garrido et al. 2025; Bordes et al. 2025] and action anticipation [Assran et al. 2025; Mur-Labadia et al. 2026; Wang & Xu 2026], or a new predictor trained on frozen DINO/V-JEPA features [Zhou et al. 2025; Baldassarre et al. 2025; Garrido et al. 2026]. The reasons given for freezing are collapse avoidance, probe comparability, or pipeline staging [Dreamer 4; FlowWM]; none asks whether the predictor adds anything over the encoder. Copy-last controls, standard in semantic and feature forecasting [Luc et al. 2017; Karypidis et al. 2025], are absent from IntPhys, IntPhys 2, GRASP, InfLevel, Physics-IQ and EK100; the closest precedents are Physion's finding that simulated futures add nothing to observed features [Bear et al. 2021], the observation that a predictor trained on physics-free early layers still scores on VoE [Joseph et al. 2026], and a single copy-last comparison for a further-trained JEPA predictor [Wei et al. 2026]. In egocentric anticipation the same pattern was noted a decade ago: future-feature regression adds at most a point and can hurt, which Furnari & Farinella attribute to camera motion [2020; Girdhar & Grauman 2021]. Meanwhile, plausibility is separable from frozen *image* embeddings alone [Internò et al. 2026]. We close this gap: we measure the released predictor against copy and constant-velocity nulls under the very protocols that certify it, localize its failures (reach, permanence, closure), and use them to design a predictor that must predict.

(First-author/year citations; arXiv ids are in the tables above. Wang & Xu = TAP-JEPA; Karypidis = DINO-Foresight; Wei = Human-JEPA; Internò = GEOPHYS.)

---

## 미확인 · 주의

- ❓ **GEOPHYS 93.3 %** 의 IntPhys 2 split/프로토콜 (main? held-out? pairwise?) 은 초록에 없다. 본문 확인 전 수치 인용 금지. 또한 "V-JEPA 2 … perform near chance" 가 어느 프로토콜의 V-JEPA 2 인지 확인 필요.
- ❓ **Human-JEPA** 0.873/0.798 은 자기 predictor · K700 20 clip. 릴리즈 V-JEPA 2.1 predictor 의 copy-gap 은 미보고. 우리 H6 과 프로토콜이 달라 숫자를 나란히 놓지 말 것.
- ❓ **CATER 57.4 %** 는 fetch 요약 값 (표 변형 다수). 인용 전 원표 확인.
- ❓ **DWM** (2607.18715) Table 2 rollout 오차는 fetch 파생 열이 불일치. 재독 필요.
- ❓ **InfLevel** 본문은 로그인 벽 (OpenReview); static 기준선 유무 미확인.
- ❓ **NWM** 의 SD-VAE 동결 여부 본문 미명시. venue (CVPR 2025) 는 CVF 목록으로 확인.
- ❓ **VideoMAE** "unmasked copy" 문구는 검색 요약 wording — 원문 재확인.
- ❓ **AVT Table 8** (HTML v1) 의 L_feat 행 (AVT-b 13.0) 과 본문 문장 ("L_feat for AVT-b") 이 어긋난다. camera-ready PDF 로 확정 전 어느 쪽도 단정 인용 금지. (b) 조사는 "both" 열 (13.1 → 14.4) 만 보고했다.
- ❓ **ImagineRNN** 지평 Table III 의 자기 행이 추출되지 않았다 (RULSTM 행만). MeMViT EK100 anticipation 수치 미확인.
- ❓ **LeCun 2022** 인용은 Meta AI blog 재게재에서 verbatim 확인; OpenReview 원문의 절/쪽 번호는 미확인. Rao & Ballard 1999 · Sutton 1991 · Schmidhuber 1990 은 verbatim 미확보.
- ❓ "카메라 운동이 egocentric flow 를 지배한다" 의 **정량 통계** 논문은 못 찾았다 (Li–Ye–Rehg 2015 는 정성 문장, CVF PDF 절단).
- ⚠ (f) 하위 조사는 V-JEPA 2 에 EK100 encoder-only ablation 이 "없다" 고 보고했으나 **App. D.2 Table 20 에 있다** (encoder 39.1 / predictor 20.2 / both 39.7; 내가 로컬 PDF 로 확인). 문서 본문은 후자를 따른다.
- ❓ Vid2World · AdaWorld · LAPA · Flow-JEPA · SAVi++ · SOLD 등 ◐ 항목은 초록만. venue 표기 중 ICML 2024 (Genie, Rolling Diffusion) · NeurIPS 2023 (STORM) · ICLR 2024 (AVDC) 는 기억 기반이라 ❓.
- ❓ Ranzato 2016 (1511.06732) id 는 기억 기반, 미fetch — 인용 금지.
- **검색 실패 (존재하지 않거나 못 찾음)**: "latent world models are approximate retrievers" 류 논문 · "Where-JEPA / Slot-JEPA / motion-centric JEPA" · "OC-STORM" · "AMD motion-decoupled video diffusion" · "Owl-1" · "Teacher forcing and the causal gap" · "persistence baseline that repeats the last observed latent block" 출처 (2606.12987 · 2606.28383 본문에 없음) · Ego4D LTA 2026 우승 보고서 (S-26 과 동일).
- **철회 논문 인용 금지**: "The JEPA Predictor: A Transferable Operator…" (2607.16274, S-26).
- **2605.15618 은 predictor 논문이 아니다.** 제목만 보고 "latent video prediction 이 더 나은 world model 을 배운다 = 새 predictor 근거" 로 쓰면 틀린다.
- 이 문서의 "우리 수치" (IntPhys 1 +4, IntPhys 2 ≈ 0, EK100 +2, 속도 R² 0.996, probe 98–100) 는 `paper/RETHINK_PREDICTOR_ROLE_2026-09-29.md` §1 의 전사이고 여기서 재검증하지 않았다.

## 재현

- 로컬 PDF: `papers/*.pdf` → `pdftotext -layout` (스크래치패드 `lvp.txt`, `ipvwm.txt`, `vjepa2.txt`, `intphys2.txt`). 인용 줄은 V-JEPA 2 §1/§3.1/§4/§5/§6/App. D.2/B.3, IntPhys 2 §4/App. D.2/E.1/E.2, Joseph App. C.1.4, Alrasheed App. E–F.
- 웹: 하위 조사 5 갈래 (a / b+c / d / e / f+g), 각 15–21 WebSearch · 24–30 WebFetch (arXiv abs/HTML, ar5iv, 저자 페이지, 일부 PDF → pdftotext). 내가 직접 재확인한 것: 2608.21160 (HTML), 2601.05230 (HTML), 2106.08261 (ar5iv), 2606.20707 (abs) + 로컬 PDF 4 편 (V-JEPA 2 Table 20 · §4 · §5 · App. D.2/B.3, IntPhys 2 App. D/E, Joseph App. C.1.4, Alrasheed 전문).
- 선행 문서: S-26 (`LITERATURE_STATE_WORLD_MODELS_2026-09-26.md`), P-25 (`LITERATURE_POSITIONING_2026-09-25.md`). 이 문서는 둘을 대체하지 않는다 — 정의·리뷰어 행태·설계 원칙은 P-25, state-evolution 벤치마크·anticipation 지평은 S-26 이 정본이다.
