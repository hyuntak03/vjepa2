"""YAML 의 4축 스키마(model / input / frames / train)를 내부 source 로 푸는 레이어.

사용자가 쓰는 말                        내부 표현
  model: context, input: full     ->  encoder 'ctx_masked'      (미래 토큰 마스킹된 40f forward)
  model: context, input: window   ->  encoder 'isolated_ctx'    (그 구간만 독립 클립)
  model: target,  input: full     ->  encoder 'target'          (40f forward)
  model: target,  input: window   ->  encoder 'isolated_target'
  model: predictor                ->  encoder 'predictor'       (frames 는 [C, N) 로 자동 고정)

frames 는 사람이 세는 1-idx 양끝 포함 -> 코드가 쓰는 0-idx 반개구간으로 바꾼다.
  [1, 32]  -> [0, 32)    [33, 40] -> [32, 40)    [1, 8] -> [0, 8)

runs 확장 규칙 (핵심):
  train 스펙 하나당 head 를 딱 한 번 학습하고, eval 에 나열된 모든 표현에 frozen 으로
  적용한다. 같은 train 스펙이 여러 run 에 나와도 head 는 재사용된다.
  'self' 는 train 표현 자기 자신을 뜻한다 (= 정보 존재 여부 = 이식의 상한).
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

_ENC = {
    ("context", "full"): "ctx_masked",
    ("context", "window"): "isolated_ctx",
    ("target", "full"): "target",
    ("target", "window"): "isolated_target",
}


def resolve(spec: dict, n_frames: int, ctx_len: int, tubelet: int) -> dict:
    """{model,input,frames} -> {name, encoder, window} (occlusion_identity.forward 형식)."""
    model = spec["model"]
    if model == "predictor":
        w = (ctx_len, n_frames)
        enc, name = "predictor", f"pred__f{ctx_len+1}to{n_frames}"
    else:
        inp = spec.get("input", "full")
        if (model, inp) not in _ENC:
            raise ValueError(f"model={model!r} input={inp!r} 조합이 없다. "
                             f"model 은 context|target|predictor, input 은 full|window.")
        f = spec.get("frames")
        if f is None:
            raise ValueError(f"model={model!r} 는 frames 가 필요하다 (predictor 만 자동)")
        w = (int(f[0]) - 1, int(f[1]))               # 1-idx 양끝포함 -> 0-idx 반개구간
        enc = _ENC[(model, inp)]
        name = f"{model}{'F' if inp == 'full' else 'W'}__f{f[0]}to{f[1]}"

    f0, f1 = w
    if f0 % tubelet or f1 % tubelet:
        raise ValueError(f"{name}: frames {f0+1}~{f1} 가 tubelet({tubelet}) 정렬이 아니다")
    if not (0 <= f0 < f1 <= n_frames):
        raise ValueError(f"{name}: frames 범위 밖 [{f0},{f1}) (클립 {n_frames}f)")
    if enc in ("ctx_masked",) and f1 > ctx_len:
        raise ValueError(f"{name}: context 모델은 frames 1~{ctx_len} 만 볼 수 있다")
    if enc == "predictor" and w != (ctx_len, n_frames):
        raise ValueError(f"{name}: predictor 출력은 frames {ctx_len+1}~{n_frames} 고정")

    offset = 0 if enc in ("target", "ctx_masked") else (ctx_len if enc == "predictor" else f0)
    return {"name": name, "encoder": enc, "window": w, "offset": offset}


def expand(probing: dict, n_frames: int, ctx_len: int, tubelet: int
           ) -> Tuple[List[dict], Dict[str, dict]]:
    """runs 를 (train head, eval 대상들) 목록으로 펼치고, 필요한 source 사전을 만든다.

    Returns
    -------
    jobs    : [{fit: source_name, groups: [...]|None, evals: [source_name, ...]}]
              fit 이 같고 groups 도 같으면 head 를 재사용하도록 중복을 제거한다.
    sources : {source_name: {name, encoder, window, offset}}
    """
    sweep = probing.get("fit_groups_sweep") or [None]
    sources: Dict[str, dict] = {}

    def _put(spec) -> str:
        s = resolve(spec, n_frames, ctx_len, tubelet)
        sources.setdefault(s["name"], s)
        return s["name"]

    jobs: Dict[Tuple[str, tuple], dict] = {}     # (fit, groups) -> job. 중복 학습 방지
    for r in probing["runs"]:
        fit = _put(r["train"])
        evals = [fit if e == "self" else _put(e) for e in r.get("eval", ["self"])]
        groups = r.get("fit_groups")
        for g in ([groups] if groups is not None else sweep):
            key = (fit, tuple(g) if g else ())
            job = jobs.setdefault(key, {"fit": fit, "groups": list(g) if g else None,
                                        "evals": []})
            for e in evals:
                if e not in job["evals"]:
                    job["evals"].append(e)
    return list(jobs.values()), sources


def describe(jobs: Sequence[dict], sources: Dict[str, dict], tubelet: int, spatial: int) -> str:
    L = [f"  {'head 학습 위치':22s} {'frames':>8s} {'tokens':>7s} {'학습 그룹':10s} 평가 대상"]
    for j in jobs:
        s = sources[j["fit"]]
        f0, f1 = s["window"]
        tok = (f1 - f0) // tubelet * spatial
        L.append(f"  {j['fit']:22s} {f'{f0+1}-{f1}':>8s} {tok:>7d} "
                 f"{str(j['groups'] or '전체'):10s} {j['evals']}")
    return "\n".join(L)
