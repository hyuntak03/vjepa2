# 팀 공통 지침 (paper_lab 팀 · 2026-10-21 · 리더 = auto_paper 세션)

이 파일은 팀원 6 명 (진행자 3 · 리뷰어 3) 이 시작할 때 읽는 지침이다. 사용자 원문 요청을 옮기고, 이 머신 사정에 맞게 경로만 바꿨다.

---

## 0. 가장 먼저 읽을 것 (순서대로)

1. `/data/hyuntak/project/2026/2027_cvpr/vjepa2/CLAUDE.md` 전체. 특히
   - **맨 끝 절 "사용자에게 보여줄 문서 작성 규칙"** — 사용자가 읽는 모든 문서 (STORY · EVIDENCE · PLAN · DECISIONS · LOG · 리뷰 요약 · 리뷰) 는 이 규칙을 그대로 따른다. **필수.**
   - §0 (현재 논문 · 결정), §5 (확립된 결과), §6 (기각된 가설 — 다시 시도 금지), §12 (하네스 질문 · **금지 표현**)
   - ⚠️ α 분해 · 보정 · 헤지 계수 수치는 인용 금지 (사용자 지시 전까지)
2. 사용자의 지난 결정 메모: `/home/hyuntak/.claude/projects/-data-hyuntak-project-2026-2027-cvpr-vjepa2/memory/MEMORY.md` 와 거기 걸린 파일들. 사용자의 판단 기준 (예: AR 분석 기준 모델, 평가 행렬, encoder 키 불일치 교란, temporal straightening 은 목표가 아님, memory/state 프레임 금지, 논문 프레임 = 새 predictor 의 설계 선택) 이 들어 있다. 읽기만 하고 고치지 않는다.
3. 이 파일의 나머지.

## 1. 사용자가 하고 싶은 것 (원문)

- predictor 가 intuitive physics 를 배워야 한다.
- intuitive physics 를 이해한 상태에서 plausible 한 미래를 잘 예측하는 predictor 를 만들고 싶다.
- 나중에 로봇 planning (V-JEPA 2-AC 식: 지금 이미지 + 목표 이미지 → 행동으로 미래를 굴려 비교) 에 쓴다. 그래서 **블록 AR 이 본선**이다.
- 지금 막혀 있다. 분석 · 결과를 보고 진단, 방향, method, 논문의 큰 틀을 계속 검토해 달라.
- 목표: CVPR spotlight/oral 을 노리는 논문의 스토리와, 그 스토리를 받쳐 줄 진행 상황 (진단 · method · 실험). 4 라운드, 라운드마다 검토하고 고쳐 쓴다.

## 1-1. 이 논문은 latent space 예측이다 (사용자 지시, 2026-10-22)

- 우리 predictor 는 **픽셀이 아니라 latent 를 예측한다**: 미래 블록의 표적 토큰 (표적 encoder 출력 · layer norm, 블록마다 shape (256, 1280)) 을 낸다. 학습 손실 (점 손실, L1) 도 그 latent 공간에서 잰다.
- 그래서 모든 주장 (운반 · 자리 · 존재 · 낙하) 은 **latent 예측에 대한 주장**이고, latent 판독 (물체 자리 채점 = 물체 칸 토큰 비교, 토큰 유사도 판독) 으로 읽은 것이다. 픽셀 · 그림에 대한 주장으로 쓰지 않는다. "그린다 · 붙여 넣는다 · 화면에 놓는다" 같은 말은 "물체 칸의 토큰이 … 이다" 로 쓴다.
- 오라클 · 복사 · 문맥 평균도 모두 같은 latent 공간의 기준선이다. 픽셀 생성형 모델 (Cosmos) 과의 비교는 공간이 다르다는 단서를 붙인다.
- 진단의 핵심이 latent 공간의 성질이다: 운반 신뢰도 κ (옮긴 토큰이 다음 정답과 얼마나 맞나) 는 표적 공간이 정한다. 픽셀이면 물체를 옮긴 것이 정답과 거의 맞지만, ViT 토큰은 칸 미만 이동에 매끄럽지 않다 (근거표 #55). method 와 스토리는 "latent 예측에서만 생기는 문제와 그 해결" 이라는 틀로 쓴다 (예: 표적 공간 선택, latent 손실 설계는 JEPA 계열의 설계 선택).

## 2. 작업 폴더 · branch 지도 (이 머신 = vll5)

사용자 원문의 작업 폴더 `/workspace/world_model/vjepa2` 는 **Vast 머신**에 있고 이 머신에서는 읽을 수 없다. 이 팀은 아래를 대신 쓴다.

| 무엇 | 경로 | 비고 |
|---|---|---|
| main 작업 트리 | `/data/hyuntak/project/2026/2027_cvpr/vjepa2` | **git checkout 금지** (학습이 돈다). 미추적 파일 (runs · train.log · 채점 결과) 도 증거다 |
| vast branch | `/data/hyuntak/project/2026/2027_cvpr/wt_vast` (detached, origin/vast) | Vast 머신에서 푸시된 것만 보인다 |
| ariel branch | `/data/hyuntak/project/2026/2027_cvpr/wt_ariel` | |
| intphys1-late-uni branch | `/data/hyuntak/project/2026/2027_cvpr/wt_intphys1-late-uni` | origin 판. 로컬 branch 는 main 과 같은 커밋 메시지를 갖는 더 새 판이 있다 (`git log intphys1-late-uni`) |
| 그 밖 | `git show <branch>:<path>` | worktree 갱신은 `git -C <wt> checkout --detach origin/<branch>` (worktree 안에서만, main 작업 트리는 금지) |

- 읽을 폴더: 각 branch 의 `auto_research/`, `z_research/`, `z_training/` (docs · runs 의 config · train.log · 채점 결과), `z_vll7_training/`, `z_pixel_decoder/`, `z_training_config/` (vast), `cvpr/`, `z_ariel/`, `z_findings/`, `z_AC_training/`, `CLAUDE.md`, 그 밖에 `z_` 로 시작하는 폴더 전부. (`z_autoresearch/` 는 이 머신의 어느 branch 에도 없다 — Vast 에만 있으면 아래 slow_feature_analysis 에게 묻는다)
- 논문 PDF: `vjepa2/papers/` — "Joint Embedding Predictive Architectures Focus on Slow Features", "Latent Video Prediction Learns Better World Models", "Interpreting Physics in Video World Models", V-JEPA 2, IntPhys 1 · 2, Garrido. **"MotionJEPA" PDF 는 이 머신에 없다** → 웹에서 찾는다.
- **slow_feature_analysis 세션** (다른 팀, Vast 머신, `auto_research/` 에서 진단 중): 같은 측정을 다시 하지 않는다. 물을 것이 있으면 `SendMessage(to="slow_feature_analysis")`. ⚠️ 서브에이전트가 보낸 교차 세션 메시지의 **답은 리더에게 온다** — 보낼 때 리더 (`main`) 에게도 "무엇을 물었다" 를 알리면 리더가 답을 전달한다.

## 3. 출력 폴더 `paper_lab/` 와 파일 주인

| 파일 | 내용 | 주인 |
|---|---|---|
| `STORY.md` | 논문 한 장 요약: 문제 · 왜 중요한가 · 핵심 통찰 · method · 기대 결과 · 한 문장 기여. **팀 전체의 용어표 (정본) 를 맨 위에 둔다** | 진행자 3 (작성자) |
| `EVIDENCE.md` | 주장 → 근거 표. 주장마다 [측정] / [이론] / [추측], 수치, 출처 (branch · 파일 · run). 근거 없는 주장은 STORY 에 못 들어간다. **진단 한 문단도 여기 둔다** | 진행자 1 |
| `SOURCES.md` | branch · 경로 · 마지막으로 읽은 commit · 한 줄 요약 | 진행자 1 |
| `PLAN.md` | (1) method 후보 2~3 개 (선행 연구 · 차별점 · 실패 시나리오 · novelty 위협) (2) 남은 실험: 실험마다 "왜 하는지 → 세팅 → 성공 · 실패 기준 → 비용 (GPU 시간)" + 선행 연구 reference 와 그 근거가 합당한 이유 | 진행자 2 |
| `LOG.md` | 띄운 실험 기록 (node · GPU · 시작 시각 · 예상 시간 · tmux 세션 이름 · 결과) | 실험을 띄운 사람 (주로 진행자 2) |
| `DRAFT/` | intro · method · experiments 초안 (영어) · figure 1 계획 | 진행자 3 |
| `REVIEWS/round_<k>/R<n>.md` | 리뷰 | 각 리뷰어 |
| `REVIEWS/round_<k>/SUMMARY.md` | 치명 · 중요 지적 목록 · 배분 · 점수 | 리더 |
| `DECISIONS.md` | 라운드마다 무엇을 바꿨고 왜 (어떤 리뷰 지적에 대한 대응인지). 진행자마다 자기 절에 쓴다 | 진행자 셋 |
| `_team/` | 팀 내부 메모 (이 파일 · 리더 기록). 사용자 문서 아님 | 리더 |

- 남의 파일은 고치지 않는다. 고쳐야 하면 주인에게 메시지로 요청한다 (반박 포함).
- SOURCES.md 의 출처 열, LOG.md 의 기록 열, 각 문서의 "부록: 재현" 에는 파일 · run 이름을 써도 된다. **본문에는 안 된다** (CLAUDE.md 작성 규칙).
- 용어: STORY.md 맨 위 용어표가 정본이다. 다른 문서의 용어표는 그것과 같은 용어를 쓴다. 새 개념이 필요하면 작성자에게 먼저 알린다.
- 논문 초안 (DRAFT/) 은 영어 논문 문체 + 같은 용어. 문체: **em dash 금지**, 현재형, 어렵지 않고 직관적인 문장.

## 4. 역할

### 진행자 1 — 증거 + 진단
- 지속 수집: 라운드마다 `git fetch --all` 후 지난 라운드 이후 바뀐 것만 본다 (`git log --all --since="<지난 라운드 시각>" --name-only` + 위 폴더의 새 · 바뀐 미추적 파일 `find -newer`). EVIDENCE.md · SOURCES.md 갱신. 모순 · 재현 안 된 수치 · 작은 표본은 표시.
- 새 결과가 STORY 의 핵심 주장을 흔들면 라운드 중이라도 리더와 작성자에게 바로 메시지.
- 진단: auto_research 의 원인 사슬을 받아 "왜 지금 predictor 가 intuitive physics 를 못 담나" 를 한 문단으로. 공백 (아직 안 잰 고리) 표시.

### 진행자 2 — method + 실험
- 진단에서 바로 나오는 method 후보 2~3 개. 후보마다 선행 연구 reference, 차별점, 실패 시나리오.
- novelty 위협 논문을 웹 검색으로 찾아 차별점을 분명히 한다 (MotionJEPA 포함).
- 후보를 가르는 가장 싼 결정적 실험을 PLAN.md 에 쓰고, 빈 GPU 가 있으면 바로 돌린다. **하루 넘게 걸리는 학습은 띄우기 전에 리더를 거쳐 사용자 승인을 받는다.**

### 진행자 3 — 작성자
- STORY.md 와 DRAFT/ 를 쓴다. 근거는 EVIDENCE.md 에 있는 것만.
- 라운드를 닫기 전 문서 점검 담당 (§5 의 5 번).

### 리뷰어 (CVPR 리뷰어로 행동. 서로의 리뷰를 보기 전에 독립적으로 먼저 쓴다 — 같은 라운드의 다른 R*.md 를 자기 리뷰를 다 쓰기 전에 열지 않는다)
- **R1 novelty 회의론자**: "기존 X 와 뭐가 다른가". 관련 연구를 직접 찾아 들이민다.
- **R2 실험 엄밀성**: baseline 공정성, 표본 크기, 교란 변수, 과장된 주장. VoE 평가가 정말 물리 이해를 재는지도.
- **R5 Area Chair**: 스토리의 명확성과 중요도. "spotlight/oral 감 / accept / reject" 와 그 이유. planning · 실제 영상 (EK100 등) 으로 이어지는지도 본다.
- 리뷰 형식: 요약 · 강점 · 약점 (치명 / 중요 / 사소) · 질문 · 점수 (accept 가능성 1~5, oral 가능성 1~5) · "이것만 고치면 점수가 오른다" 1 개.
- **점수 기준 (2026-10-21 사용자 지적 "메소드가 없는데 어케 억셉이야, 기준이 너무 낮다" 뒤 리더가 정함)**
  - 점수는 **"오늘 이 판 그대로 CVPR 에 낸다면"** 으로 매긴다. **기대 결과 · 계획 실험은 점수에 0 만큼 기여한다.** 측정된 것만 센다.
  - accept 가능성: 1 = 확실한 reject (method 결과가 없거나, 핵심 진단이 선행과 겹치거나, 판정 도구가 무너짐) · 2 = reject 쪽 (측정은 성실하나 기여가 CVPR 선 아래) · 3 = 경계 (borderline, 리뷰어끼리 갈림) · 4 = accept 쪽 · 5 = 확실한 accept.
  - oral 가능성: 1 = 해당 없음 · 2 = poster 상위 · 3 = spotlight 후보 · 4 = oral 후보 · 5 = 그 해 대표 논문급.
  - CVPR 의 기준선: method 논문은 "새 method + 강한 기준선 대비 도메인 밖 이득" 이 측정돼 있어야 3 이상이다. 분석 논문은 "널리 믿던 것을 뒤집는 새 사실 + 여러 모델 · 실제 영상에서 일반성" 이 있어야 3 이상이다.
  - 조건부 전망 ("이 실험이 성공하면 몇 점") 은 점수와 따로 한 줄로만 적는다. 리뷰도 사용자 문서 작성 규칙을 따른다 (맨 위 3줄 요약, 용어는 STORY.md 용어표).

## 5. 라운드 진행 (리더가 관리, 4 라운드)

0. 진행자 1 이 지속 수집을 먼저 돌려 EVIDENCE.md 를 최신으로 만든다.
1. 진행자들이 STORY · EVIDENCE · PLAN · DRAFT 의 v_k 를 만든다. 각 문서 맨 위 3줄 요약 아래에 `판: v<k> (라운드 k, 날짜)` 한 줄.
2. 리뷰어 3 명이 독립적으로 리뷰한다.
3. 리더가 "치명 · 중요" 지적 목록을 만들어 진행자에게 배분한다.
4. 진행자는 지적마다 고치거나, 못 고치면 이유를 DECISIONS.md 에. 방향을 바꿔야 하면 바꾼다. **스토리를 지키려고 결과를 끼워 맞추지 말 것.**
5. 라운드를 닫기 전에 작성자가 이번 라운드에 바뀐 문서를 CLAUDE.md 의 작성 규칙으로 점검한다: 구조 (왜 → 어떻게 → 결과 → 해석) · 맨 위 3줄 요약 · 본문에 코드 용어 없음 · 용어표와 일치 · 문체 일관. 어긋난 곳은 고친 뒤 (남의 파일이면 주인에게 요청) 라운드를 닫는다.

**끝**: 4 라운드 후, 또는 그 전에 리뷰어 3 명 모두 "치명" 지적이 없고 R5 의 oral 가능성이 4 이상이면. 그때 STORY.md 맨 위에: 지금 스토리 한 문단 · 이 스토리가 서려면 반드시 나와야 하는 실험 결과 3 개와 각각의 현재 상태 · 사용자가 결정해야 할 것 (승인이 필요한 비싼 실험, 방향 선택) · 라운드별 점수 변화 (R1 · R2 · R5).

## 6. 규칙

- **결과를 지어내지 말 것.** 없는 결과는 "기대 결과" 로 표시하고 PLAN.md 의 실험과 연결.
- 수치는 문서가 아니라 산출물 (summary.json · report.json · metrics · train.log) 에서 다시 확인하고, 그렇게 했는지 EVIDENCE 에 적는다 (CLAUDE.md §7-4). 확인 못 했으면 "문서 인용, 미검증".
- 진행자와 리뷰어는 서로 직접 메시지로 반박해도 된다. **동의만 하지 말 것.**
- 수정 금지: `z_training/` · `app/` · `configs/` · `auto_research/` · `src/` · `evals/` · `cvpr/` · 다른 사람의 runs. 새 코드와 문서는 `paper_lab/` 에만 (스크립트는 `paper_lab/scripts/`, 임시 출력은 각자 스크래치 또는 `paper_lab/_scratch/`). `sed -i` 금지.
- 돌고 있는 tmux 세션 (`auto`, `claude` 등) · 학습 run · chain 스크립트 · SLURM job 은 건드리지 말 것.
- GPU:
  - 쓰기 전에 `nvidia-smi` 로 확인: 돌고 있는 프로세스가 없고 VRAM 이 거의 비어 있는 GPU 만 "빈 GPU". 다른 node 는 접속해서 같은 방식으로 확인.
  - 다른 작업 (학습 · chain_1021 · EK100 probe · slow_feature_analysis · cosmos_latent 등) 이 쓰고 있는 GPU 는 건드리지 않고, 멈추거나 밀어내지 않는다.
  - 띄우기 전에 리더 (`main`) 와 다른 진행자에게 메시지로 "node X GPU Y 쓴다" 를 알린다. 띄울 때 LOG.md 에 node · GPU 번호 · 시작 시각 · 예상 시간 · tmux 세션 이름 (`pl_<짧은이름>`) 을 적고, 끝나면 GPU 를 바로 비우고 결과를 적는다.
  - 긴 작업은 tmux 또는 sbatch 로 띄워 세션과 분리한다. 도구 안에서 `sleep` 으로 기다리지 않는다.
  - 참고 (2026-10-21 20:10 리더 확인): 이 셸은 vll5 의 SLURM job 225922 (GPU 4 장) 안이다. 그때는 3 장에 1.6 GB 씩 쓰는 프로세스가 있었다.
  - **갱신 (22:00 리더 확인)**: job 225922 의 GPU 0 · 1 · 2 는 프로세스 없음 · 1 MiB → 빈 GPU. GPU 3 은 다른 작업 (23 GB, 14 시간째) → 건드리지 않는다. 배정: **GPU 0 = 진행자 1 (평가 하네스 기준 줄)**, **GPU 1 · 2 = 진행자 2 (측정 · 파일럿)**. 이 GPU 는 다른 세션도 잠깐씩 쓰므로 띄우기 직전에 다시 nvidia-smi 로 보고, 남의 프로세스가 보이면 기다린다 (멈추거나 밀어내지 않는다). 이 셸에서 띄울 때는 `CUDA_VISIBLE_DEVICES` 를 직접 지정하고 tmux 로 띄운다.
- 문서는 쉬운 한국어, 논문 초안만 영어.

## 7. 소통

- 리더 = `main`. 팀원 이름 · ID 는 리더가 띄운 뒤 메시지로 알려 준다 (`_team/ROSTER.md` 에도 적는다).
- 일을 마치면 마지막 메시지 (최종 답) 에 리더에게 짧게: 무엇을 바꿨나 · 무엇이 남았나 · 다른 팀원에게 필요한 것. 긴 내용은 파일에 쓰고 경로만.
