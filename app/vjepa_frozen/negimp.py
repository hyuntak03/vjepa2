"""불가능 미래 음의 손실 (negimp, 2026-11-02, 사용자 지시: "impossible 도 넣고 impossible 은 1−L1 으로 negative loss").

config 최상위 `negimp:` 가 있을 때만 켜진다 (없으면 학습 경로가 그대로다).

  negimp:
    families: ${DATA_CSV}/blender_track_v4vc_negimp/families.json   # z_training/data/build_v4vc_imp_index.py
    p: 0.5              # 영상마다 불가능으로 바꿀 확률 (짝을 만들 수 있는 영상만)
    swap_frac: 0.5      # 단계 영상 (A_k) 문맥일 때 swap 을 고를 확률 (나머지 vanish)
    context_frames: 16  # 이음 앞 샘플 수 (= 문맥). 창 시작 0/1 · 간격 2 면 이음 32 와 맞는다
    margin: 1.0         # 불가능 손실 = max(0, margin − L1)
    ar_terms: rollout   # kind=ar: 불가능 샘플 L1 = 굴린 예측 L1 (rollout) | sum = TF + rollout
    weight: 1.0         # λ (2026-11-05): 불가능 쪽 손실 가중
    mode: single        # 2026-11-05: single = 샘플마다 가능 또는 불가능 하나 (11-02 판) | paired = 짝 손실 (아래)
    type_w: [1, 1, 1]   # paired: 불가능 종류 vanish · appear · swap 뽑기 가중 (사용자: vanish : appear = 1 : 1)

paired (2026-11-05, 사용자): 샘플마다 같은 묶음에서 문맥 X 와 짝 미래 Y 를 고른다 (종류 = type_w 비율, appear 면 X = 빈 장면 B).
  가능 클립 = X 그대로, 불가능 클립 = X 문맥 + Y 미래. 증강은 두 클립에 같은 결정 (48 장을 한 번에 변환한 뒤 나눈다).
  kind=ar 손실 = TF L1 + 굴린 L1(p, 가능) + λ · [굴린 L1(p, 가능) − 굴린 L1(p, 불가능)].  p (굴린 예측) 는 문맥만 보므로 하나다.
  두 미래가 같은 토큰에서는 괄호 안이 0 이라 그 토큰은 가능 쪽으로만 당겨진다. 다른 토큰에서도 좌표마다 (1+λ)|p−a| − λ|p−b| 의
  최소는 p = a (가능) — 11-02 판처럼 예측을 정답 밖으로 밀어내는 힘이 없다 (삼각부등식으로 아래가 막혀 hinge 가 필요 없다).

짝 고르기 (matched, 문맥 = 뽑힌 영상 자신): 뽑힌 영상이 A_k 면 vanish_k (미래 = B) 또는 swap_k_j (미래 = A_j),
B 면 appear_k (미래 = A_k, k 무작위). 짝에 들어가는 단계 영상은 hidden_splice 를 통과해야 한다.
같은 문맥이 가능 · 불가능으로 같은 확률 (p = 0.5) 로 나오므로, 두 미래가 같은 토큰 (배경) 에서 두 손실의 기울기가 상쇄된다
(|p−a| − |p−a| = 0). 문맥마다 불가능 가중이 가능 가중보다 크면 그 토큰을 정답에서 밀어내는 힘이 남는다 — 그래서 무작위
묶음 · 단계 대신 문맥 일치로 뽑는다.
"""
from __future__ import annotations

import json
import random

import numpy as np
import torch

KINDS = ("possible", "vanish", "appear", "swap")


class ImpTable:
    """목록 번호 → 불가능 짝 후보. DataLoader worker 안에서 draw 를 부른다."""

    def __init__(self, cfg: dict, paths, starts, stride: int, n_frames: int):
        d = json.load(open(cfg["families"]))
        self.p = float(cfg.get("p", 0.5))
        # 2026-11-06 (사용자 "사라짐 : 나타남 1 : 1"): 물체 있는 문맥 (A_k → vanish · swap) 의 불가능 확률. 빈 장면 문맥 (B → appear) 은 p.
        #   문맥은 계속 뽑힌 영상 자신이고 문맥마다 가능 ≥ 불가능 (p_object_ctx ≤ 0.5) 이라 배경 토큰을 미는 힘은 생기지 않는다. 기본 = p (예전과 같다).
        self.p_obj = float(cfg.get("p_object_ctx", self.p))
        # 2026-11-08 (사용자 "물체 영역에만 negative loss 걸어봐 — target 물체에다가"): region = target 이면 불가능 손실의 L1 을
        #   target 이 보이는 토큰 (가능 미래 = 문맥 영상 자신 ∪ 불가능 미래 = 짝 영상, 생성기 보임 마스크) 에서만 잰다. none = 전 토큰 (예전과 같다).
        self.region = str(cfg.get("region", "none") or "none")
        if self.region not in ("none", "target"):
            raise ValueError(f"negimp.region 은 none | target: {self.region!r}")
        self.swap_frac = float(cfg.get("swap_frac", 0.5))
        self.mode = str(cfg.get("mode", "single"))
        if self.mode not in ("single", "paired"):
            raise ValueError(f"negimp.mode 는 single | paired: {self.mode!r}")
        self.type_w = [float(x) for x in (cfg.get("type_w") or [1, 1, 1])]
        self.splice = int(d.get("splice", 32))
        self.n_ctx = int(cfg.get("context_frames", 16))
        for st in starts:                           # 문맥 샘플이 전부 이음 앞, 미래 샘플이 전부 이음 뒤
            idx = np.arange(st, st + n_frames * stride, stride)
            if int((idx < self.splice).sum()) != self.n_ctx or not (idx[self.n_ctx:] >= self.splice).all():
                raise ValueError(f"negimp: start {st} · stride {stride} 의 창이 이음 {self.splice} 에서 {self.n_ctx} | {n_frames - self.n_ctx} 로 안 갈린다")
        n = len(paths)
        full = [ln.strip() for ln in open(d["list"], encoding="utf-8") if ln.strip()]
        if sum(1 + len(f["A"]) for f in d["families"]) != len(full) or list(paths) != full[:n]:
            raise ValueError(f"negimp: families.json 이 만든 목록 ({d['list']}) 과 학습 목록이 다르다 — 같은 목록으로 다시 만들 것")
        self.vanish = [None] * n                    # A_k → B 번호
        self.swap = [[] for _ in range(n)]          # A_k → [A_j 번호]
        self.appear = [[] for _ in range(n)]        # B → [A_k 번호]
        self.fam_of = [None] * n                    # paired: 영상 → (이음 통과한 A 번호들, B 번호)
        for f in d["families"]:
            A, hs = f["A"], f["hs"]
            if f["B"] >= n or max(A) >= n:          # LIMIT 스모크: 앞 n 편 안에 다 든 묶음만
                continue
            fam = ([a for a, h in zip(A, hs) if h], f["B"])
            for v in list(A) + [f["B"]]:
                self.fam_of[v] = fam
            for k, a in enumerate(A):
                if not hs[k]:
                    continue
                self.vanish[a] = f["B"]
                self.swap[a] = [A[j] for j in range(len(A)) if j != k and hs[j]]
                self.appear[f["B"]].append(a)
        self.n_elig = sum(v is not None for v in self.vanish) + sum(bool(x) for x in self.appear)
        self.n = n

    def describe(self) -> str:
        if self.mode == "paired":
            return (f"paired · type_w (vanish, appear, swap) {self.type_w} · 이음 {self.splice} · 문맥 {self.n_ctx} 장 · "
                    f"묶음에 든 영상 {sum(f is not None for f in self.fam_of)}/{self.n}")
        return (f"p {self.p} · 물체 문맥 p {self.p_obj} · swap_frac {self.swap_frac} · 이음 {self.splice} · 문맥 {self.n_ctx} 장 · "
                f"짝을 만들 수 있는 영상 {self.n_elig}/{self.n} · 불가능 L1 영역 {self.region}")

    def draw_pair(self, i: int):
        """paired: (code, 문맥 영상 X, 미래 영상 Y). code 1 vanish · 2 appear · 3 swap, 0 = 짝 없음 (X = i, Y = None)."""
        r = random.Random((random.getrandbits(62) << 20) ^ int(i))
        fam = self.fam_of[i]
        if fam is None or not fam[0]:
            return 0, i, None
        A, B = fam
        t = r.choices((1, 2, 3), weights=self.type_w)[0]
        if t == 3 and len(A) < 2:
            t = 1
        if t == 1:
            return 1, r.choice(A), B
        if t == 2:
            return 2, B, r.choice(A)
        x, y = r.sample(A, 2)
        return 3, x, y

    def draw(self, i: int):
        """(code, 미래 영상 번호). code 0 = 가능. rank 마다 worker RNG 가 같으므로 (같은 seed) 영상 번호 i 를 섞는다."""
        r = random.Random((random.getrandbits(62) << 20) ^ int(i))
        if r.random() >= (self.p if self.vanish[i] is None else self.p_obj):
            return 0, None
        if self.vanish[i] is not None:
            if self.swap[i] and r.random() < self.swap_frac:
                return 3, r.choice(self.swap[i])
            return 1, self.vanish[i]
        if self.appear[i]:
            return 2, r.choice(self.appear[i])
        return 0, None


def _stats(imp, pos_l1, pos_tf, imp_l1, imp_loss):
    """합 (all_reduce 용): n_pos, Σ가능 L1, Σ가능 TF, n_imp, Σ불가능 L1, Σ불가능 손실, n_hinge 활성, (n, ΣL1) × vanish · appear · swap."""
    pos = imp == 0
    neg = ~pos
    out = [pos.sum(), pos_l1[pos].sum(), pos_tf[pos].sum(), neg.sum(), imp_l1[neg].sum(), imp_loss[neg].sum(),
           (imp_loss[neg] > 0).sum()]
    for c in (1, 2, 3):
        m = imp == c
        out += [m.sum(), imp_l1[m].sum()]
    return torch.stack([x.detach().double() for x in out])


def release_loss(p, h, imp, margin: float, loss_exp: float, weight: float = 1.0):
    """kind=oneshot. p, h (B, M, D). 샘플마다 L1 (미래 토큰 평균) → 가능 L1 · 불가능 weight · max(0, margin − L1), 샘플 평균.
    weight (λ, 2026-11-05 사용자 "lambda ablation 0.5, 1, 2"): 불가능 쪽 손실에만 곱한다. 1.0 이면 11-02 판과 같다. 로그의 imp_loss 는 곱하기 전 값."""
    d = (p.float() - h.float()).abs().pow(loss_exp).mean(dim=(1, 2)) / loss_exp
    il = torch.clamp(margin - d, min=0.0)
    loss = torch.where(imp == 0, d, float(weight) * il).mean()
    return loss, _stats(imp, d, torch.zeros_like(d), d, il)


def release_pair_loss(p, h, h_imp, imp, loss_exp: float, weight: float = 1.0):
    """kind=oneshot paired (2026-11-06, 사용자 "릴리즈 pft 짝 있는 음의 손실 람다 3"). ar_pair_loss 의 릴리즈 판.
    p (B, M, D) = 같은 문맥에서 한 번 낸 예측, h = 가능 클립 창 전체 정답 (LN, 미래 M 토큰), h_imp = 불가능 클립 (문맥 같음 · 미래만 다름) 창 전체 정답 (LN, 같은 M 토큰).
    손실 = L1(p, 가능) + λ · [L1(p, 가능) − L1(p, 불가능)] (짝 없는 샘플은 괄호 0) — 샘플 평균. 정답이 하나면 λ 와 무관하게 p = 가능 이 최적.
    stats 는 ar_pair_loss 와 같은 자리 (TF 칸은 0): n_pos = 전체, Σ가능 L1, 0, n_imp = 짝 있는 샘플, Σ불가능 L1, Σ(가능 − 불가능), n(불가능이 더 먼 짝)."""
    if h_imp.shape != h.shape:
        raise ValueError(f"negimp release_pair_loss: h {tuple(h.shape)} != h_imp {tuple(h_imp.shape)}")
    dp = (p.float() - h.float()).abs().pow(loss_exp).mean(dim=(1, 2)) / loss_exp
    di = (p.float() - h_imp.float()).abs().pow(loss_exp).mean(dim=(1, 2)) / loss_exp
    paired = imp > 0
    diff = torch.where(paired, dp - di, torch.zeros_like(dp))
    loss = (dp + float(weight) * diff).mean()
    st = [torch.tensor(float(dp.numel()), device=dp.device), dp.sum(), torch.zeros((), device=dp.device), paired.sum(), di[paired].sum(),
          diff[paired].sum(), (di[paired] > dp[paired]).sum()]
    for c in (1, 2, 3):
        m = imp == c
        st += [m.sum(), di[m].sum()]
    return loss, torch.stack([x.detach().double() for x in st])


def ar_loss(predictor, h, S: int, C: int, K: int, loss_exp: float, rollout_steps: int, imp, margin: float, terms: str, h_in=None,
            weight: float = 1.0, region=None):
    """kind=ar 기본 경로 (ar.ar_losses 의 hint · delta · tf_span · unc 없는 판). return (loss, jloss, sloss, R, stats).
    가능 샘플: TF L1 + 굴린 L1 (기존과 같다). 불가능 샘플: max(0, margin − L1_imp), L1_imp = 굴린 L1 (rollout) 또는 TF + 굴린 (sum).
    jloss · sloss 는 가능 샘플만의 평균 (로그용).
    h_in (2026-11-02, encoder LoRA): predictor 입력 = 입력 경로 LoRA 복사본의 블록별 LN 토큰 (ar.ar_losses 의 h_in 과 같다). 타깃은 늘 h (얼린 encoder)."""
    from app.vjepa_frozen import ar as AR
    B, N = h.shape[:2]
    nb = N // S
    if nb != C + K:
        raise ValueError(f"negimp ar_loss: nb={nb} != C+K={C + K}")
    R = max(1, min(int(K), int(rollout_steps)))
    _o = predictor(h if h_in is None else h_in, ar={"idx": AR.block_indices(B, nb, S, h.device), "n_ctx_blocks": C, "rollout_steps": R})
    p_tf, p_ar = _o[0], _o[1]
    if len(_o) != 2:
        raise ValueError("negimp 는 kind=ar 기본 출력 (p_tf, p_ar) 만 받는다 (ar_unc 아님)")
    ro = (p_ar.float() - h[:, C * S: (C + R) * S].float()).abs().pow(loss_exp).mean(dim=(1, 2)) / loss_exp
    if p_tf is None:
        # 2026-11-11 (사용자 "one-shot 판 PFT 로 v4bc4 + negative loss p 0.2"): ar_oneshot 은 TF 가 없다 (ar.ar_losses 와 같다) —
        #   가능 샘플 = 미래 K 블록 한 번에 L1, 불가능 샘플 = max(0, margin − 그 L1). TF 항은 0.
        if region is not None or terms != "rollout":
            raise ValueError("negimp + ar_oneshot 은 region 없음 · ar_terms=rollout 만")
        tf = torch.zeros_like(ro)
    else:
        tf = (p_tf.float() - h[:, S:].float()).abs().pow(loss_exp).mean(dim=(1, 2)) / loss_exp
    l1_imp = ro if terms == "rollout" else tf + ro
    has_reg = torch.ones_like(ro, dtype=torch.bool)
    if region is not None:
        # 2026-11-08 region=target: region (B, nb, S) 0/1 = target 보이는 토큰 (가능 ∪ 불가능 미래). 불가능 L1 = 굴린 예측의 그 토큰 평균.
        #   target 이 미래 블록에서 한 번도 안 보이는 불가능 샘플은 손실 0 (밀 자리가 없다). 가능 샘플은 그대로 (전 토큰 TF + 굴린 L1).
        if terms != "rollout":
            raise ValueError("negimp.region 은 ar_terms: rollout 전용")
        m = region[:, C: C + R].reshape(region.size(0), R * S).to(p_ar.device, torch.float32)
        tok = (p_ar.float() - h[:, C * S: (C + R) * S].float()).abs().pow(loss_exp).mean(dim=-1) / loss_exp    # (B, R*S)
        cnt = m.sum(1)
        has_reg = cnt > 0
        l1_imp = torch.where(has_reg, (tok * m).sum(1) / cnt.clamp(min=1.0), ro)
    il = torch.where(has_reg, torch.clamp(margin - l1_imp, min=0.0), torch.zeros_like(l1_imp))
    loss = torch.where(imp == 0, tf + ro, float(weight) * il).mean()     # weight (λ, 2026-11-05): 불가능 쪽에만
    pos = imp == 0
    jl = tf[pos].mean() if pos.any() else tf.sum() * 0.0
    sl = ro[pos].mean() if pos.any() else ro.sum() * 0.0
    return loss, jl.detach(), sl.detach(), R, _stats(imp, ro, tf, l1_imp, il)


def summarize(st) -> dict:
    """all_reduce 한 합 → 평균. 분모 0 이면 nan."""
    s = [float(x) for x in st]
    q = lambda a, b: a / b if b > 0 else float("nan")
    n = s[0] + s[3]
    return {"n_pos": s[0], "n_imp": s[3], "imp_frac": q(s[3], n), "pos_l1": q(s[1], s[0]), "pos_tf": q(s[2], s[0]),
            "imp_l1": q(s[4], s[3]), "imp_loss": q(s[5], s[3]), "hinge_act": q(s[6], s[3]),
            "n_vanish": s[7], "l1_vanish": q(s[8], s[7]), "n_appear": s[9], "l1_appear": q(s[10], s[9]),
            "n_swap": s[11], "l1_swap": q(s[12], s[11])}


def ar_pair_loss(predictor, h, h_imp_fut, S: int, C: int, K: int, loss_exp: float, rollout_steps: int, imp, weight: float = 1.0, h_in=None,
                 form: str = "linear", dist: str = "l1", margin: float = 1.0):
    """paired (2026-11-05). h = 가능 클립 블록별 LN 토큰 (B, (C+K)*S, D), h_imp_fut = 불가능 클립 미래 K 블록 (B, K*S, D).
    손실 = TF + 굴린 L1(p, 가능) + λ · [굴린 L1(p, 가능) − 굴린 L1(p, 불가능)] (짝 없는 샘플은 괄호 0). return (loss, jl, sl, R, stats).
    stats 는 _stats 와 같은 자리를 쓰되 뜻이 다르다: n_pos = 전체 샘플, Σ가능 L1 = Σ굴린 L1(가능), n_imp = 짝 있는 샘플,
    Σ불가능 L1 = Σ굴린 L1(불가능), Σ불가능 손실 = Σ(가능 − 불가능) 차, n_hinge = 불가능이 더 먼 (올바른 순서) 짝 수."""
    from app.vjepa_frozen import ar as AR
    B, N = h.shape[:2]
    nb = N // S
    if nb != C + K or h_imp_fut.shape[1] != K * S:
        raise ValueError(f"negimp ar_pair_loss: nb={nb} C+K={C + K} imp 토큰 {h_imp_fut.shape[1]} (기대 {K * S})")
    R = max(1, min(int(K), int(rollout_steps)))
    _o = predictor(h if h_in is None else h_in, ar={"idx": AR.block_indices(B, nb, S, h.device), "n_ctx_blocks": C, "rollout_steps": R})
    p_tf, p_ar = _o[0], _o[1]
    if p_tf is None or len(_o) != 2:
        raise ValueError("negimp paired 는 kind=ar 기본 출력 (p_tf, p_ar) 만 받는다")
    tf = (p_tf.float() - h[:, S:].float()).abs().pow(loss_exp).mean(dim=(1, 2)) / loss_exp
    ro = (p_ar.float() - h[:, C * S: (C + R) * S].float()).abs().pow(loss_exp).mean(dim=(1, 2)) / loss_exp
    ri = (p_ar.float() - h_imp_fut[:, : R * S].float()).abs().pow(loss_exp).mean(dim=(1, 2)) / loss_exp
    paired = imp > 0
    # 2026-11-09 손실 모양 (사용자 "negative loss 의 철학은 이어가는데 loss 설계를 바꿔도 돼"):
    #   form linear (기본, 11-05 판): λ · [d(p, 가능) − d(p, 불가능)]
    #   form hinge : λ · max(0, margin − d(p, 불가능)) — 가능 L1 (tf + ro) 이 **같은 예측**에 걸려 있어, 두 미래가 잡음 수준으로만 다른 좌표에서는
    #                λ < 1 이면 당김이 이겨 예측이 정답에 머문다 (single 판처럼 다른 샘플끼리 밀고 당기다 쏠리지 않는다). 실제로 다른 좌표만 밀린다.
    #   dist l1 (기본) | l2 : d = 토큰 · 차원 평균 제곱 거리 (linear 와 쓰면 p 에 대해 선형 → (가능 − 불가능) 방향으로만 민다).
    if dist == "l2":
        ro_d = (p_ar.float() - h[:, C * S: (C + R) * S].float()).pow(2).mean(dim=(1, 2))
        ri_d = (p_ar.float() - h_imp_fut[:, : R * S].float()).pow(2).mean(dim=(1, 2))
    elif dist == "l1":
        ro_d, ri_d = ro, ri
    else:
        raise ValueError(f"negimp.pair_dist 는 l1 | l2: {dist!r}")
    if form == "linear":
        diff = torch.where(paired, ro_d - ri_d, torch.zeros_like(ro))
    elif form == "hinge":
        diff = torch.where(paired, torch.clamp(float(margin) - ri_d, min=0.0), torch.zeros_like(ro))
    else:
        raise ValueError(f"negimp.pair_form 은 linear | hinge: {form!r}")
    loss = (tf + ro + float(weight) * diff).mean()
    st = [torch.tensor(float(B), device=h.device), ro.sum(), tf.sum(), paired.sum(), ri[paired].sum(), diff[paired].sum(),
          (ri[paired] > ro[paired]).sum()]
    for c in (1, 2, 3):
        m = imp == c
        st += [m.sum(), ri[m].sum()]
    return loss, tf.mean().detach(), ro.mean().detach(), R, torch.stack([x.detach().double() for x in st])
