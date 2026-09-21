# analysis/attention — predictor 의 attention 을 보는 두 도구

> 레포 규칙은 루트 `CLAUDE.md`, 실행 계약은 `configs/protocols/README.md`,
> 본 실험 세트는 `z_research/IntPhysGenV11/README.md`.
> ⚠️ **2026-09-21** — 이 줄이 가리키던 `z_research/IntPhysGenV11/Archive/ATTENTION_2026-09-01.md` 는
> **레포에 없다** (작성되지 않았거나 추적되지 않아 사라졌다). 결과 해석은 산출물에서 다시 읽어야 한다.

**짝으로 쓰라고 만든 두 도구다. 하나는 관찰, 하나는 개입이다.**

| 폴더 | 무엇 | 답하는 질문 |
|---|---|---|
| [`predictor_attn/`](predictor_attn/README.md) | **관찰** — attention 분포·거리를 층×헤드로 접는다 | 어떤 프레임을 **보나**, 얼마나 멀리 보나 |
| [`knockout/`](knockout/README.md) | **개입** — 간선을 끊고 채점 변화를 본다 | 그 경로가 **없으면** 무엇이 무너지나 |

**둘을 같이 봐야 하는 이유**: v11 ViT-H 실측에서 attention **질량이 가장 큰 층은 0**
(미래 토큰이 문맥에 0.885)인데, **끊었을 때 가장 아픈 층은 8** 이다 (Δacc −18.06).
질량과 인과적 기여가 갈린다. 한쪽만 보면 반대 결론에 도달한다.

## 공통 규약

- **토큰 격자는 `predictor_attn/geometry.py` 의 `Grid` 하나를 공유한다** — 두 도구가 같은
  좌표계다. `idx → t = idx // S, h = (idx % S) // G, w = idx % G` 이고 모델의 RoPE
  분해(`RoPEAttention.separate_positions`)와 같다
- **경로·컬럼·체크포인트는 `z_research/scripts/harness/resolve.py` 를 그대로 불러온다.**
  데이터셋을 새로 붙일 때 고칠 곳은 `configs/protocols/datasets.md` 섹션 하나뿐이고,
  `--set` 도 `run.sh` 의 `SET=` 과 같다
- **출력 자리도 하네스 관례**: `<results_root>/<도구>__<데이터셋>_<모델>/`
- **ViT-H / ViT-L 둘 다 `--model` 인자 하나로 바뀐다.** predictor 는 H·L 이 완전히 같고
  (384/12/12) 인코더 embed_dim 만 1280 / 1024 로 다르다. 둘 다 실측 검증했다
- **모든 실행이 자기 검증을 하고 그 값을 산출물에 남긴다.** 그림 스크립트는 그 플래그를
  다시 확인하고, 안 맞으면 그리지 않고 죽는다

## `analysis/` 아래에 둔 이유

이 트리는 `python -m analysis.<name>` 으로 직접 부르는 사후 분석용이고 `evals/main.py` 의
`eval_name` 디스패치를 타지 않는다 (`analysis/__init__.py`). 반면
`z_research/scripts/analysis/` 는 **산출물·토큰 캐시만 읽는 GPU 불필요 스크립트** 자리다.
두 도구 모두 모델 forward 가 필요하므로 여기가 맞다.

## 왜 토큰 캐시를 못 쓰나

캐시의 `ctx_masked.npy` 는 `_ln(z)` 인데 predictor 가 학습 때 받은 것은 **LN 이전의 `z`**
다 (`evals/analysis_vlm/occlusion_identity/forward.py:157-163`). 그래서 두 도구 모두
문맥 인코더를 실제로 돌린다. 대신 **인코더 출력은 배치 안에서 한 번만 계산해 재사용**한다
(knockout 은 명세 수십 개가 이걸 공유한다 — `knockout/README.md` §2).
