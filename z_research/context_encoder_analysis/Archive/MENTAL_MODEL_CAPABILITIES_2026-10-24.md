# mental model 능력 분해 — IntPhys 2 를 기준으로 다시 쪼갬 (2026-10-24)

> 앞선 정의 `STATE_EVOLUTION_CAPABILITIES_2026-09-19.md` (1 읽기 · 2 지속 · 3 전이 · 4 영속) 를 잇는다. 그 정의는 유지하고, 측정에서 따로 갈라진 능력을 더했다.
>
> **질문** 예측한 미래로 IntPhys 2 를 맞히려면 predictor 가 어떤 능력을 가져야 하나. 지금 모델은 어디가 비었나.
> **답** IntPhys 2 는 같은 결과 영상을 문맥만 바꿔 가능 · 불가능으로 엇갈려 쓴다. 그래서 **앞에서 본 것을 기억해 결과와 대조하는 능력 (문맥 조건화)** 이 핵심이고, 지금 어떤 모델도 이 몫이 0 이다. 그 아래 그리기 · 영속 · 정체 보존도 비어 있다.
> **다음** 미측정 칸 (관측자 분리 · 문맥 조건화의 모델별 값 · 물체 간 관계) 을 학습 없이 채운다. 물체 간 관계를 정의에 넣을지 결정이 필요하다.

## 용어

| 용어 | 뜻 |
|---|---|
| 상태 | 장면에서 예측에 필요한 것 — 어떤 물체가 있고, 어디 있고, 어떻게 움직이고, 어떻게 생겼는지 |
| 정답 토큰 | 블록 (프레임 2 장) 하나를 고정 encoder 에 넣고 LN 한 것, shape (256, 1280) |
| 굴린 예측 | predictor 가 자기 예측을 되먹여 이어 만든 미래 블록 (채점 방식) |
| 운반 이득 | 예측 물체 변위 ÷ 실제 변위 (1 = 정확히 옮김, 0 = 제자리), 파라미터 없는 읽기 |
| 정답 토큰 판독기 | 정답 토큰으로만 학습한 attention 판독기 (위치 · 속도 · 정체) |
| 결과 영상 / 문맥 | IntPhys 2 한 영상의 위반 이후 부분 / 그 이전 부분 |
| margin | surprise(불가능) − surprise(가능). 양수면 정답 |

---

## 1. 왜 다시 쪼개나

어떤 학습 (상태 감독 · 정답 토큰 판독기 · IntPhys 2 이어 학습) 도 IntPhys 2 공식 프로토콜을 우연 (50 %) 위로 올리지 못했다. 이어 학습한 모델의 63.5 % (우리 16/16 채점) 도 margin 이 0 근처에 몰린 채 조금 기운 것이다 (|margin| 중앙 0.0042, 아무 정보 없는 release 0.0025) [측정]. 렌더를 보면 예측은 장면 배치는 따라 그리지만 위반에 관련된 물체를 그리지 않는다 [측정, 그림]. 그래서 "무엇이 없어서 못 맞히는가" 를 능력 단위로 나눠 칸마다 잰다.

## 2. IntPhys 2 가 실제로 재는 것

### 2-1. 구성 [측정, metadata.csv]

- Main 253 장면 × 영상 4 개 (pair 1 · 2 × 가능 · 불가능). 조건 4 개, 조건마다 게임 8 종 (permanence · immutability · continuity 는 같은 8 게임을 공유: Box · HotAirBallon · JailStone · MovingAroundOccluder · PrisonCell · Restaurant · RotatingCup · SphereFallingDown; solidity 는 따로 8 게임: BoxSoldity · CameraSolidity · FixedJumpSolidity · FixedMarryPoppins · JumpSolidity · Scaffoling · SolidityFallingFlat · SphereFallingDownSoldity)
- 카메라 이동: 조건마다 장면의 57~64 % (가능 영상 기준 permanence · immutability · continuity 68 / 120, solidity 94 / 146)
- 가림막 이름이 있는 장면: permanence · immutability · continuity 각 34 개 (나머지는 장면 자체 구조물이나 카메라 회전으로 가린다) [추측: "occluder: None" 의 의미는 확인 안 함]

### 2-2. 같은 결과가 문맥만 바뀌어 가능 · 불가능으로 엇갈린다 [측정]

영상 마지막 4 장의 평균 픽셀 차이 (128², 무작위 40 장면, 조건별 중앙값):

| 조건 | 1 가능 ↔ **2 불가능** | 1 가능 ↔ 1 불가능 | 1 가능 ↔ 2 가능 | 2 가능 ↔ **1 불가능** |
|---|---:|---:|---:|---:|
| solidity | **3.6** | 4.3 | 9.9 | **3.2** |
| permanence | **1.7** | 4.6 | 4.4 | **1.5** |
| continuity | **3.1** | 6.7 | 6.4 | **2.8** |
| immutability | **2.2** | 3.8 | 7.2 | **3.6** |

pair 1 가능 영상의 끝은 pair 2 불가능 영상의 끝과 가장 닮았다 (반대도 같다). 즉 **결과 A · B 를 짝 1 에서는 A 가능 · B 불가능, 짝 2 에서는 반대로** 쓴다.

**결과**: 결과 영상만 보고 "이게 이상한가" 를 매기는 surprise 는 장면 안에서 상쇄되어 설계상 50 % 다. 쌍을 가르는 것은 **문맥에 따라 어느 결과가 놀라운지가 바뀌는 몫 (문맥 항)** 뿐이다 [이론, auto_research `README.md` §0 의 PMI 교차항 분석과 같다]. auto_research 측정: 블록 AR · 영상 문맥 AR 모두 문맥 항 0, "가림 전에 본 것과 비교" 하는 오라클 영역 66 % (고정 카메라 79 %), 예측과 비교 48.6 % [측정].

### 2-3. 게임별 점수 — 전부 우연 근처 [측정]

우리 16/16 채점 (최종 라벨, 500 쌍), 게임마다 6~22 쌍이라 ±10~20 점은 잡음:

| 조건 | 게임 | n | release | AR | AR + 3D 감독 | release + VOBench | AR + VOBench |
|---|---|---:|---:|---:|---:|---:|---:|
| continuity | SphereFallingDown | 14 | 57.1 | 57.1 | 71.4 | 92.9 | 71.4 |
| continuity | RotatingCup | 20 | 30.0 | 40.0 | 45.0 | 30.0 | 55.0 |
| immutability | JailStone | 22 | 36.4 | 36.4 | 36.4 | 40.9 | 36.4 |
| permanence | JailStone | 22 | 59.1 | 40.9 | 36.4 | 81.8 | 40.9 |
| solidity | Scaffoling | 18 | 61.1 | 72.2 | 61.1 | 50.0 | 61.1 |
| (나머지 27 칸) | | | 33~83 | 33~78 | 33~78 | 29~83 | 17~67 |

일관되게 높은 게임이 없다. 모델끼리 순서도 게임마다 바뀐다.

## 3. 능력 분해

의존 순서: 1 → 2 → {3, 4} → {5, 6} → 7. 뒤 능력은 앞 능력이 서는 구간에서만 판정한다 (09-19 원칙).

| # | 능력 | 정의 | 재는 법 | 출처 |
|---|---|---|---|---|
| 1 | **읽기** | 관측에서 상태를 꺼낸다 | 문맥 정답 토큰 → 위치 · 속도 · 정체 | 09-19 |
| 2 | **지속** | 굴리는 동안 상태를 잃지 않는다 (물체가 예측에 남는다) | 굴린 예측에 물체다운 토큰이 남나 | 09-19 |
| 3 | **전이** | 상태를 문맥이 드러낸 추세대로 갱신한다 (옮기기 · 가속) | 운반 이득, 걸음별 위치 오차 | 09-19 |
| 4 | **그리기** (신규) | 들고 있는 상태를 **맞는 자리 · 맞는 생김새의 토큰** 으로 낸다 | 정답 토큰 판독기로 예측을 읽기 · 판독기가 정답 칸을 보나 | 측정에서 2 와 갈라짐: 예측에서 함께 학습한 판독기는 상태를 읽었지만 (7~14 px) 물체가 아닌 칸에 따로 적힌 것이었다 (정답 토큰에 씌우면 102~110 px) |
| 5 | **영속** | 관측이 끊긴 동안 상태를 들고 있다가 다시 보일 때 맞게 그린다 | 다시 나타난 블록의 위치 · 생김새 | 09-19 |
| 6 | **정체 보존** (신규) | 움직이거나 가려져도 생김새 (모양 · 색) 를 유지한다 | VOBench 움직임 모양 · 색, IntPhys 2 immutability | 측정에서 5 와 갈라짐: VOBench 영속 94~100 인데 움직이는 모양 · 색 55~68 |
| 7 | **문맥 조건화** (신규) | 같은 결과라도 **앞에서 본 것** 에 따라 다르게 예측한다 (본 것을 기억해 결과와 대조) | IntPhys 2 문맥 항, 같은 결과 · 다른 문맥 쌍의 예측 차이 | IntPhys 2 설계가 이것만 잰다 (§2-2) |
| 8 | **관측자 분리** (신규) | 카메라 움직임과 물체 움직임을 나눠 장면 기준으로 상태를 둔다 | 카메라 이동 장면에서 정지 물체 상태가 유지되나 | IntPhys 2 장면의 57~64 % 가 카메라 이동 |
| (9) | **물체 간 관계** | 통과 불가 · 받침 · 충돌 | IntPhys 2 solidity, GRASP collision | **09-19 에 상식 문제로 정의에서 뺐다.** IntPhys 2 solidity 가 이것을 요구한다 → 넣을지 사용자 결정 필요 |

## 4. 지금 모델 성적표

| # | 능력 | AR (자연 영상) | AR + 정답 토큰 판독기 (gthead, epoch 20, 학습 중) | AR + VOBench 이어 학습 |
|---|---|---|---|---|
| 1 | 읽기 | ✅ 문맥 → 속도 R² 0.79 (물체 3×3), 위치 20 px | (같은 encoder) | (같은 encoder) |
| 2 | 지속 | ❌ 굴릴수록 사라짐 (운반 걸음 6: 0.05) | ✅ 물체 유지 | ✅ |
| 3 | 전이 | ❌ 운반 0.17 | ◐ 0.39 | ◐ 0.38 |
| 4 | 그리기 | ❌ 정답 토큰 판독기로 118 px | ◐ 판독기 23 px, 하지만 판독기가 정답 칸을 안 보고 **마지막 관측 자리** 를 본다 (attention ×256: 정답 0.1 · 마지막 관측 90) | 미측정 |
| 5 | 영속 | ◐ VOBench late 영속 94 (있다 / 없다), IntPhys 2 permanence 47 | ◐ VOBench 85~90 | ◐ VOBench 100, IntPhys 2 50 |
| 6 | 정체 보존 | ❌ VOBench 움직임 모양 58 · 색 66 | ❌ 모양 53~60 · 색 61~63 | ◐ 모양 83 · 색 78 (held-out) |
| 7 | 문맥 조건화 | ❌ 문맥 항 0 (auto_research) | 미측정 | 미측정 (IntPhys 2 공식 51.8) |
| 8 | 관측자 분리 | 미측정 | 미측정 | 미측정 |
| (9) | 물체 간 관계 | GRASP collision 3.1, IntPhys 2 solidity 54.9 | — | — |

데이터: 1~4 는 phys3d 검증 (학습에 안 쓴 3D 합성, 카메라 고정), 5 · 6 은 VOBench late held-out · IntPhys 2, 7 은 IntPhys 2.

**읽는 법**
- VOBench 이어 학습은 3 · 6 을 올려 VOBench 를 크게 올렸다 (late 76.7 → 90.1 held-out). 그 일부는 2 층 헤드 셋 (L2.h2 · h6 · h7) 이 굴리는 동안 직전 자기 상태를 보게 된 데서 온다 — 셋을 끄면 운반 0.38 → 0.27, 같은 층 다른 셋을 끄면 0.39 [측정, 인과]
- gthead 는 2 (지속) 를 채웠고 3 이 0.39 로 올랐지만, 판독기 기준으로는 물체를 여전히 마지막 관측 자리에서 읽는다. 운반 이득 (모양 닮은 칸) 과 판독기 (attention) 가 다른 칸을 가리킨다 → 예측 안에 물체다운 토큰이 두 군데 (옛 자리 · 옮긴 자리) 퍼져 있을 수 있다 [추측]
- **IntPhys 2 가 요구하는 7 (문맥 조건화) 은 어느 모델도 측정상 0 이다.** 3 · 6 을 올린 VOBench 이어 학습도 IntPhys 2 를 못 올렸다 → 7 은 3 · 6 과 따로 막혀 있다 [측정 + 추측]

## 5. 미측정 칸과 측정 계획 (학습 없이)

| 칸 | 방법 | 비용 |
|---|---|---|
| 4 그리기 — VOBench 이어 학습 | 정답 토큰 판독기로 굴린 예측 읽기 · 판독기 attention 이 정답 / 마지막 관측 칸 중 어디 | 10 분 |
| 7 문맥 조건화 — 모델별 | IntPhys 2 쌍 margin 을 결과 항 · 문맥 항으로 나누기 (auto_research `ip2_ctx_cue` 방식을 각 모델에) | 30 분 |
| 8 관측자 분리 | phys3d 에 카메라 이동 판이 있는지 확인 → 없으면 IntPhys 2 카메라 이동 장면에서 정지 배경 칸의 예측 일관성 | 확인 먼저 |
| 4 · 6 — IntPhys 2 | 위반 물체 라벨이 없다 → 사람 라벨 시트에 물체 위치를 더하거나, 가능 · 불가능 결과의 픽셀 차이 영역을 물체 영역으로 쓰기 | 설계 필요 |

## 6. 단서

- IntPhys 2 게임 내용 (무엇이 어떻게 움직이는지) 은 이름과 렌더 몇 개로만 봤다. 게임별 요구 능력 대응은 아직 하지 않았다
- 1~4 의 수치는 3D 합성 (카메라 고정, 물체 하나) 에서 쟀다. IntPhys 2 의 복잡한 장면으로 그대로 넘어간다는 근거는 없다
- 게임별 점수는 칸마다 6~22 쌍이라 칸 단위로 읽지 않는다
- gthead 는 학습 중 (epoch 22/70). 성적표 값은 epoch 20 체크포인트

---

## 부록: 재현

```bash
cd /workspace/world_model/vjepa2
# §2-2 결과 엇갈림: IntPhys2 Main metadata.csv 에서 장면마다 네 영상의 마지막 4 장 평균 |Δ| (스크래치 계산, 이 문서 표)
# §2-3 게임별: cvpr/results/eval/intphys1_late/intphys2_violation_uni_mp4_<모델>_final/per_block.json + data_csv/intphys2_violation_uni/index_mp4.csv
# §4 수치
/venv/vjepa/bin/python -W ignore z_research/scripts/analysis/velocity_transfer.py --run natural_ar_vw_state latest.pt          # 1 읽기 · 2
/venv/vjepa/bin/python -W ignore z_research/scripts/analysis/transport_gain.py --run <run> <ckpt>                               # 3 전이 (RUN@층.헤드 로 헤드 끄기)
/venv/vjepa/bin/python -W ignore z_research/scripts/analysis/gt_head_check.py --run natural_ar_vw_state latest.pt              # 4 그리기 (AR)
/venv/vjepa/bin/python -W ignore z_research/scripts/analysis/state_head_transfer.py --run natural_ar_vw_state_phys3d_event_gthead e20.pt   # 4 (gthead)
/venv/vjepa/bin/python -W ignore z_research/scripts/analysis/rollout_attention_compare.py --run natural_ar_vw_state latest.pt --run vobench_ar_vw_state_postft_g latest.pt
```
