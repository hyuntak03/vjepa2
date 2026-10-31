# ① late · visible 블록별 물체 probing — 릴리즈 predictor (2026-10-30 · 10-31)

> **질문** — 문맥 블록 (1–8) 과 예측 블록 (9–16) 의 표현에 물체 정보 (모양 · 색) 가 얼마나 실리고, encoder 로 배운 판독기가 predictor 출력에서 읽히나?
> **답** — **late 에서 문맥 encoder 는 가려진 물체도 담지만 (가린 블록 8 모양 82–89 %), 그 판독기를 predictor 출력에 걸면 우연에 가깝다 (모양 22–41 %, 색 7–22 %).** p 로 배운 판독기는 60 % 안팎으로 읽는다. 정보는 있고 꼴이 다르다. visible 에서는 이식이 가까운 미래에서 되고 (82 %) 멀어질수록 떨어진다 (43 %).
> **다음** — 같은 판독을 PFT p 에 건다. late 의 z → p 이식이 PFT 에서 살아나는지 본다.

---

## 용어

| 용어 | 뜻 |
|---|---|
| 블록 | 그림의 x 축. 모델 입력의 연속 2 샘플 (튜블릿). 블록 1–8 = 문맥 (원 영상 0–45 프레임), 9–16 = 예측 구간 (48–93 프레임) |
| z | 문맥 encoder 출력에 층 정규화를 건 것 = predictor 의 입력. 블록 1–8, 블록마다 공간 토큰 256 개 × 1280 차원 |
| p | predictor 출력 (릴리즈). 블록 9–16, 같은 shape |
| late k | 문맥 끝 k 샘플과 예측 앞 k 샘플이 가려진 조건. VOBench 에는 k = 1, 2, 3 만 있다 |
| 가린 블록 | 두 샘플이 모두 가려진 블록 (그림의 진한 회색) · 한 샘플만 가려진 블록 (옅은 회색). k1: 블록 8 · 9 가 반. k2: 블록 8 · 9 가 전부. k3: 블록 7 · 10 이 반, 8 · 9 가 전부 |
| 설계 칸 | 같은 궤적 · 가림막을 공유하는 장면 묶음 (물체 세트 × 위반 × 배경 × 운동 × 가림 × k). VOBench_v2 는 칸마다 장면 3 개, v1 은 1 개. 분할과 신뢰구간의 단위 |
| 구간 공유 판독기 | 문맥 구간 (z 블록 1–8) 또는 예측 구간 (p 블록 9–16) 마다 판독기 하나를 그 구간의 모든 블록으로 학습한 것. 평가는 블록마다 |
| attentive 판독기 | 블록 하나의 256 토큰을 attention 으로 모아 분류 (depth 1, query 1, head 16). 위치 임베딩 없음 |
| 블록 평균 + 선형 | 블록 하나의 256 토큰 평균 (1280 차원) → 표준화 → 선형 층 |

과제와 우연 수준: 모양 6 가지 (16.7 %) · 색 8 가지 (12.5 %) · 물체 있음 vs 빈 장면 (50 %). 모양 · 색은 primitive 물체 영상만 쓴다.

---

## 1. 지금 판 — VOBench_v2 · attentive · 구간 공유 · visible / late (2026-10-31) ★

### 왜 했나
- 사용자 전제를 블록 단위로 확인한다. 전제는 세 가지다.
  - encoder 는 가려진 물체의 모양 · 색을 읽는다.
  - predictor 가 만든 미래에서도 읽힌다.
  - 그런데 encoder 로 배운 판독기는 predictor 출력에서 안 읽힌다.
- 비교 기준으로 가림막이 없는 visible 조건을 같은 판독으로 낸다.

### 무엇을 어떻게 했나
- **자료:** VOBench_v2 가능 영상. late 1296 개 (k 1–3), visible 432 개. z · p · h 를 릴리즈로 뽑았다.
  - h = 실제 32 장을 target encoder 에 넣고 층 정규화한 것의 미래 블록 9–16
  - replica 0 의 z · p 블록 평균이 v1 캐시와 cos ≥ 0.999 로 같다 [측정]
- **분할:** 설계 칸 단위 2 겹 ((물체 세트, 위반, k) 층마다 절반). 신뢰구간도 설계 칸 재표집 95 %.
- **판독기:** attentive, 구간 공유 학습 (영상 × 블록 행), 20 epoch · lr 3e-4 · head 시드 1 개.
  - 겹당 학습 영상: late 270, visible 90 (visible 은 장면이 1/3)
- **그림 세 열:**
  - (a) z 자기 판독
  - (b) p 자기 판독
  - (c) 이식: 문맥 encoder z 로 학습한 판독기 (a 의 판독기) 를 p 에 건다. 이식할 때 p 에는 z 와 같은 토큰별 층 정규화를 건다.
  - (h = 실제 미래 프레임의 target encoder 판독기와 그 이식도 쟀지만, 10-31 사용자 결정으로 그림 · 표에서 뺐다. json 에 남아 있다.)
- **k 마다 평가 영상:** 모양 · 색 180 개, 있음/없음 144 개.

### 결과 [측정] — 정확도 %, 블록 범위

| 조건 | 과제 | (a) z 보이는 블록 | z 블록 8 | (b) p 9–16 | **(c) z 판독기 → p** |
|---|---|---:|---:|---:|---:|
| visible | 모양 (우연 16.7) | 93–98 | 97 | 94 → 76 | **82 → 43** |
| visible | 색 (우연 12.5) | 74–84 | 84 | 91 → 62 | **78 → 28** |
| late k1 | 모양 | 97–99 | 82 | 61–66 | **22–41** |
| late k2 | 모양 | 96–99 | 86 | 56–62 | **24–32** |
| late k3 | 모양 | 97–100 | 89 | 57–66 | **28–37** |
| late k1 | 색 | 97–99 | 78 | 39–50 | **10–22** |
| late k2 | 색 | 94–100 | 76 | 29–40 | **7–17** |
| late k3 | 색 | 95–99 | 86 | 35–40 | **10–17** |
| late | 있음/없음 | 96–100 | 97–99 | 97–99 | 62–85 |

- "94 → 76" 은 블록 9 에서 블록 16 쪽으로 가며 떨어진다는 뜻이다.
- 그림:
  - `visible/visible_block_probe.png`
  - `late/late_k{1,2,3}_block_probe.png`

### 해석
- **문맥 encoder 는 가려진 물체의 모양 · 색을 담는다** [측정]. late 의 블록 8 (완전히 가림) 에서 z 가 모양 82–89 %, 색 76–86 % 다.
- **late 에서 p 에도 정체는 있지만, 문맥 encoder 판독기로는 거의 안 읽힌다** [측정].
  - p 자기 판독은 모양 56–66 %, 색 29–50 % 다.
  - z 판독기를 p 에 걸면 모양 22–41 %, 색 7–22 % 다. 우연에 가깝다.
  - 사용자 전제 셋 (encoder 는 읽는다 · p 에서도 읽힌다 · encoder 판독기는 p 에서 안 읽힌다) 이 모두 이 판에서 선다.
- **visible 에서는 이식이 가까운 미래에서 되고 멀어질수록 떨어진다** [측정]. z 판독기 → p 가 블록 9 에서 모양 82 · 색 78 % 이고, 블록 15–16 에서 43–56 · 28–52 % 다. p 자기 판독도 94 → 76 · 91 → 62 로 떨어지지만 덜 떨어진다.
  - 이식 조사 Q1 의 "보이는 먼 미래는 주로 크기" 와 같은 방향이다 [해석].
- **late 의 이식 실패는 정보가 없어서가 아니라 꼴이 달라서다** [측정 + 해석]. 같은 p 를 p 로 배운 판독기는 60 % 안팎으로 읽는데, z 판독기는 우연에 가깝다.
  - 이식 조사의 "late 정체는 다시 나타날 자리에 몰려 있지 않다" 와 맞는다.
- 물체 있음/없음은 이식도 62–85 % 로 꽤 된다. 정체 (모양 · 색) 만 무너진다 [측정].

**단서**
- head 시드 1 개, 20 epoch, 학습 정확도 99–100 %.
- visible 은 학습 영상이 late 의 1/3 이다. visible 의 z 색 (74–84 %) 이 late (94–100 %) 보다 낮은 것은 그 때문일 수 있다 [추측].
- attentive 판독기에는 위치 임베딩이 없다. "어디에 있나" 는 이 판독으로 말하지 못한다.
- 이식에서 p 에 층 정규화를 건 판 하나만 냈다. 정규화를 안 건 판은 안 쟀다.
- z 판독기는 문맥 블록 1–8 로 학습했다. 예측 블록과는 시점이 다르다 (블록 위치가 판독기 입력에 없으므로 같은 판독기를 그대로 건다).
- k 사이 차이는 대부분 신뢰구간 안이다.

---

## 2. 이전 판 — VOBench_v1 (2026-10-30, 기록)

**무엇을 했나** — VOBench_v1 late 가능 영상 432 개 (모양 · 색은 primitive 물체 180 개). 분할은 VOBench split v2 파일의 장면 단위 2 겹이고, 겹당 학습 영상은 90 개였다.

| 판 | 상태 | 문맥 z 보이는 블록 (모양 / 색) | 예측 p (모양 / 색) |
|---|---|---|---|
| attentive · 블록마다 따로 · 35 epoch lr 1e-4 (첫 판) | 덜 학습됨 (보이는 블록 3 의 색 30 %) | — | — |
| attentive · 블록마다 따로 · 100 epoch lr 3e-4 | 물러남 — 외움 (학습 100 %), 그림 `_archive/figures/_superseded/v1_attn_per_block/` | 76–95 / 35–77 | 39–58 / 6–29 |
| 블록 평균 + 선형 · 블록마다 따로 (행 90) | 비교용 | 52–70 / 15–50 | 45–70 / 15–42 |
| 블록 평균 + 선형 · 구간 공유 (행 720) | v1 의 마지막 판, 그림 `_archive/figures/v1_linear/` | 58–83 / 32–55 | 53–73 / 23–43 |

**정정 · 철회 (지우지 않음)**
- ⚠️ **철회 (10-31)** — "문맥에서는 attentive 가, 예측에서는 블록 평균이 높다 → p 의 정보는 화면에 퍼져 있다". 학습 영상 90 개에서 attentive 가 외운 탓이었다. v2 에서는 p 에서도 attentive 가 블록 평균보다 높다 (모양 62 vs 48). 다시 시도하지 말 것.
- ⚠️ **정정 (10-31)** — v1 의 블록 평균 값 (예: 모양 p 65 %) 은 분할 파일 하나에 기댄 값이었다. 무작위 장면 분할 4 개에서는 50–62 % 다.
- 점검 (유효): 배경 · 운동 · 위반 · 변이 · 궤적 (배경 × 운동 × k) 만 보는 다수결로 모양 · 색을 맞히면 8–15 % (우연 수준). 장면 · 궤적이 물체를 알려 주는 누수는 없다 [측정].

---

## 부록: 재현

```bash
P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
F=/data2/local_datasets/world/Benchmark/VOBench_v2
# ── 지금 판 (VOBench_v2 · visible / late) ──
$P z_research/scripts/data/build_vobench_index.py --frames-root $F --out data_csv/vobench_v2        # 문맥 무결성 mismatch 0
# 추출 (vll5 = 원본 로컬). late z · p 는 처음에 vll2 에서 NFS 프레임 사본 (stage_vobench_v2_late.py) 으로 뽑았다
sbatch -p batch_vll -w vll5 --gres=gpu:1 -c 8 --mem=48G --wrap "$P z_research/scripts/analysis/release_v2_extract.py --occ late --what h --frames $F"
sbatch -p batch_vll -w vll5 --gres=gpu:1 -c 8 --mem=48G --wrap "$P z_research/scripts/analysis/release_v2_extract.py --occ visible --what zph --frames $F"
# 판독 (과제마다 job 하나, 약 10–15 분)
for O in late visible; do for T in shape color exist; do sbatch -p batch_vll -w vll5 --gres=gpu:1 -c 8 --mem=64G \
  --wrap "$P z_research/scripts/analysis/release_late_block_probe.py attn_occ --occ $O --task $T"; done; done
$P z_research/scripts/figures/plot_release_block_probe_v2.py --occ late       # · --occ visible
# ── 이전 판 (VOBench_v1) · 선형 판 → _archive/ ──
$P z_research/scripts/analysis/release_late_block_probe.py {stage,pool}; sbatch ... "... linear [--set v2] [--fold-seed S] [--task T]"
$P z_research/scripts/figures/plot_release_late_block_probe.py [--src attentive|linear_v2|attn_v2]   # 옛 경로 기준 (정리 전)
```

| 파일 | 무엇 |
|---|---|
| `figures/v2_attn/late/` · `visible/` | ★ 지금 판 그림 (png · pdf) + `results/<조건>_block_attn_v2_{shape,color,exist}.json` (블록 × k 정확도 · CI · 영상별 정답, 자기 판독 `shared` · 이식 `transfer`) |
| `figures/v2_attn/late/_prev_no_transfer/` | (c) 를 넣기 전 late 판 (같은 설정, z · p 만) |
| `_archive/figures/` · `_archive/results/` | v1 · v2 선형 판, v1 attentive (물러남), v1 분할 점검 — 10-31 사용자 요청으로 옮김 (지우지 않음) |
| `logs/` | job 로그 (`tune_235860.log` = v1 attentive 학습량 점검) |

- 특징 캐시 (NFS): `/data/hyuntak/project/2026/2027_cvpr/nfs_cache/release_{late,visible}_vobench_v2/` (z · p · h_fut · 블록 평균 · meta) · `…/release_late_vobench/` (v1). late 프레임 사본: `…/nfs_cache/VOBench_v2_late_frames/`.
- 스크립트: `z_research/scripts/analysis/release_late_block_probe.py` (`attn_occ` = 지금 판) · `release_v2_extract.py` · `z_research/scripts/figures/plot_release_block_probe_v2.py` · `z_research/scripts/data/stage_vobench_v2_late.py`.
