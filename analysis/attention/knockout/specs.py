"""knockout 명세 — 어떤 (질의 집합 → 키 집합) 간선을 어느 층에서 끊을까.

predictor 입력은 [문맥 토큰 C·S개] + [mask 토큰 (T−C)·S개] 가 인덱스 순으로 붙은
N = T·S 시퀀스다. 그 위의 self-attention 은 네 종류의 간선을 갖는다:

    ctx  → ctx    문맥 안의 상호작용 (프레임 내 / 프레임 간으로 더 쪼갤 수 있다)
    mask → ctx    미래 토큰이 문맥을 읽는다        <- "정보를 가져오는" 경로
    mask → mask   미래 토큰끼리의 상호작용         <- "미래를 서로 맞추는" 경로
    ctx  → mask   문맥이 미래 토큰을 읽는다 (bidirectional 이라 존재한다)

문법
    <q>-><k>[:<rel>]@<layers>

  q, k    ctx | mask | all | t<a>-<b>          튜블릿 범위(양끝 포함). t7-7 = 문맥 마지막
  rel     any(기본) | same_tub | diff_tub | same_patch | diff_patch
  layers  all | L<i> | L<i>-<j> | L<i>,<j>,...

예
    mask->ctx@L0                미래 토큰이 층 0 에서 문맥을 못 보게 한다
    mask..ctx@L0                같은 뜻. `..` 는 `->` 의 **셸 안전 별칭**이다
                                (따옴표 없이 `->` 를 쓰면 bash 가 리다이렉트로 먹는다)
    ctx->ctx:diff_tub@all       문맥의 **프레임 간** 상호작용만 전 층에서 끊는다
    mask->mask@L6-11            후반 층에서 미래끼리의 상호작용을 끊는다
    t7-7->all@L3                문맥 마지막 튜블릿이 층 3 에서 아무것도 못 보게 한다

⚠️ **자기 자신은 항상 남긴다** (`keep_self=True` 기본). 안 그러면 mask→ctx 와 mask→mask 를
   동시에 끊었을 때 mask 토큰의 키가 하나도 안 남아 softmax 가 NaN 이 된다.
   이건 해석에 영향을 주는 설계 결정이므로 결과 json 에 그대로 기록된다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch

from analysis.attention.predictor_attn.geometry import Grid

REL = ("any", "same_tub", "diff_tub", "same_patch", "diff_patch")
# 클립마다 다른 키 집합 — 인덱스에서 온다 (run.py --hidden-csv). 문맥 튜블릿 단위다.
#   hidden       물체가 가려진 문맥 프레임이 든 튜블릿
#   hidden_ctrl  같은 개수의 이웃 튜블릿 (바로 앞, 자리가 없으면 바로 뒤) — 대조군
PER_CLIP = ("hidden", "hidden_ctrl")
_SPEC = re.compile(r"^\s*(.+?)->(.+?)(?::([a-z_]+))?@(\S+)\s*$")   # q 에 t7-7 처럼 - 가 들어간다
_TRANGE = re.compile(r"^t(\d+)-(\d+)$")


def _token_set(name: str, grid: Grid) -> torch.Tensor:
    """이름 -> (N,) bool. True = 그 집합에 속하는 토큰."""
    t = torch.arange(grid.N) // grid.S
    name = name.strip()
    if name == "ctx":
        return t < grid.C
    if name == "mask":
        return t >= grid.C
    if name == "all":
        return torch.ones(grid.N, dtype=torch.bool)
    if name in PER_CLIP:                       # 자리만 잡는다. 실제 집합은 배치마다 넣는다
        return torch.zeros(grid.N, dtype=torch.bool)
    m = _TRANGE.match(name)
    if not m:
        raise ValueError(f"토큰 집합 {name!r} 을 모르겠다. ctx | mask | all | t<a>-<b>")
    a, b = int(m.group(1)), int(m.group(2))
    if not (0 <= a <= b < grid.T):
        raise ValueError(f"튜블릿 범위 t{a}-{b} 가 0..{grid.T-1} 밖이다")
    return (t >= a) & (t <= b)


def parse_layers(s: str, L: int) -> list[int]:
    s = s.strip()
    if s == "all":
        return list(range(L))
    s = s[1:] if s.startswith("L") else s
    out: list[int] = []
    for part in s.split(","):
        if "-" in part:
            a, b = (int(v) for v in part.split("-"))
            out += list(range(a, b + 1))
        else:
            out.append(int(part))
    bad = [i for i in out if not 0 <= i < L]
    if bad:
        raise ValueError(f"층 {bad} 이 0..{L-1} 밖이다")
    return sorted(set(out))


@dataclass
class Spec:
    name: str                       # 원문 그대로. 결과 표의 키가 된다
    q: str
    k: str
    rel: str
    layers: list[int]
    qmask: torch.Tensor = field(repr=False, default=None)
    kmask: torch.Tensor = field(repr=False, default=None)

    @property
    def per_clip(self) -> bool:
        return self.k in PER_CLIP or self.q in PER_CLIP

    @property
    def first_layer(self) -> int:
        """이 층부터 계산이 달라진다 — prefix 캐시를 여기서 이어붙인다."""
        return min(self.layers)


def canon(s: str) -> str:
    """명세 문자열의 정본 형태. `..` 는 `->` 의 셸 안전 별칭이다.

    따옴표를 빠뜨리면 bash 가 `->ctx` 를 리다이렉트로 먹어 `mask-` 만 도착한다
    (실제로 당했다). `mask..ctx@L0` 은 그럴 일이 없다.

    ⚠️ **명세를 키로 쓰는 곳은 전부 이걸 통과시킬 것.** run.py 가 창 폭 표를 만들 때
       정규화 전 문자열로 키를 잡아 `Spec.name` 과 안 맞은 적이 있다 (그림에서 있지도
       않은 폭 계열이 생겼다).
    """
    return s.replace("..", "->").strip()


def parse(s: str, grid: Grid, n_layers: int) -> Spec:
    s = canon(s)
    m = _SPEC.match(s)
    if not m:
        raise ValueError(f"명세 {s!r} 를 못 읽겠다. 형식: <q>-><k>[:<rel>]@<layers>\n"
                         f"  예) mask->ctx@L0 · ctx->ctx:diff_tub@all · mask->mask@L6-11")
    q, k, rel, lay = m.group(1), m.group(2), (m.group(3) or "any"), m.group(4)
    if rel not in REL:
        raise ValueError(f"rel={rel!r}; {REL} 중 하나여야 한다")
    return Spec(name=s.strip(), q=q.strip(), k=k.strip(), rel=rel,
                layers=parse_layers(lay, n_layers),
                qmask=_token_set(q, grid), kmask=_token_set(k, grid))


def attn_mask(spec: Spec, grid: Grid, device, keep_self: bool = True) -> torch.Tensor:
    """(N, N) bool. **True = attend 허용** (SDPA 의 attn_mask 규약).

    끊는 자리는 `질의 ∈ q  AND  키 ∈ k  AND  rel` 이다. 층 정보는 여기 안 들어간다 —
    어느 층에 이 마스크를 물릴지는 runner 가 정한다.
    """
    N, S = grid.N, grid.S
    ii = torch.arange(N, device=device)
    t, sp = ii // S, ii % S
    qm, km = spec.qmask.to(device), spec.kmask.to(device)
    cut = qm[:, None] & km[None, :]
    if spec.rel == "same_tub":
        cut &= t[:, None] == t[None, :]
    elif spec.rel == "diff_tub":
        cut &= t[:, None] != t[None, :]
    elif spec.rel == "same_patch":
        cut &= sp[:, None] == sp[None, :]
    elif spec.rel == "diff_patch":
        cut &= sp[:, None] != sp[None, :]
    keep = ~cut
    if keep_self:
        keep.fill_diagonal_(True)
    return keep


def attn_mask_batch(spec: Spec, grid: Grid, device, tubs: torch.Tensor,
                    keep_self: bool = True) -> torch.Tensor:
    """클립별 마스크 (B, 1, N, N) bool — SDPA 가 (B, H, N, N) 으로 broadcast 한다.

    `tubs` 는 (B, T) bool: 그 클립에서 PER_CLIP 집합에 드는 튜블릿. q/k 중 PER_CLIP 인 쪽을
    이걸로 채우고 나머지는 attn_mask() 와 같은 규칙이다.
    """
    N, S = grid.N, grid.S
    ii = torch.arange(N, device=device)
    t = ii // S
    B = tubs.size(0)
    tok = tubs.to(device)[:, t]                                    # (B, N)
    qm = tok if spec.q in PER_CLIP else spec.qmask.to(device)[None].expand(B, N)
    km = tok if spec.k in PER_CLIP else spec.kmask.to(device)[None].expand(B, N)
    cut = qm[:, :, None] & km[:, None, :]                          # (B, N, N)
    if spec.rel != "any":
        raise ValueError("클립별 집합에는 rel 을 안 쓴다")
    keep = ~cut
    if keep_self:
        keep[:, ii, ii] = True
    return keep[:, None]


def sweep(edges: list[str], mode: str, n_layers: int, widths=(1,)) -> list[str]:
    """간선 목록 x 층 모드 -> 명세 문자열 목록.

      each        층 하나씩             L개/간선   <- 층별 기여를 보는 기본
      window      중심 i, 폭 w 의 창     L개/간선/폭
      all         전 층 동시            1개/간선
      prefix      0..i 누적 (앞에서부터)
      suffix      i..L-1 누적 (뒤에서부터)

    ⚠️ **`window` 를 쓰는 이유는 중복(redundancy)이다.** 같은 정보를 이웃 층에서 다시
       읽을 수 있으면 한 층만 막아봐야 나머지가 메운다 (LLM attention-knockout 이 ±k 창을
       쓰는 이유와 같다). 그래서 단일 층 Δ 는 기여의 **하한**이다.

    ⚠️ **LLM 의 창 폭을 상수로 가져오지 말 것.** ±4/48층 = 깊이의 19% 인데 우리 predictor 는
       12층이라 같은 비율이 ±1(폭 3) 이다. ±4 를 그대로 쓰면 12층 중 9층이라 `@all` 에
       가까워지고, `mask->ctx` 에서는 예측이 상수가 되는 퇴화 상태로 미끄러진다.
       **폭을 축으로 두고 어디서 포화하는지 재는 것이 맞다** (widths=(1,3,5)).

    ⚠️ 창은 양 끝에서 잘린다 (중심 0, 폭 5 -> L0-2 로 3층). 명세에 실제 층이 그대로
       기록되므로 그림에서 폭을 같이 읽을 것.
    """
    out = []
    for e in edges:
        if mode == "each":
            out += [f"{e}@L{i}" for i in range(n_layers)]
        elif mode == "window":
            for w in widths:
                if w % 2 == 0:
                    raise ValueError(f"창 폭은 홀수여야 한다 (중심이 있어야 한다) -> {w}")
                r = (w - 1) // 2
                for i in range(n_layers):
                    a, b = max(0, i - r), min(n_layers - 1, i + r)
                    out.append(f"{e}@L{a}" if a == b else f"{e}@L{a}-{b}")
        elif mode == "all":
            out += [f"{e}@all"]
        elif mode == "prefix":
            out += [f"{e}@L0-{i}" for i in range(n_layers)]
        elif mode == "suffix":
            out += [f"{e}@L{i}-{n_layers-1}" for i in range(n_layers)]
        else:
            raise ValueError(f"층 모드 {mode!r}; each | window | all | prefix | suffix")
    return out
