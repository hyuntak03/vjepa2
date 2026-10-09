"""VERIFY ISCENE2 공통 — 분석 스크립트 로직을 재사용하지 않고 다시 쓴 판독·집계."""
import json
from pathlib import Path
import numpy as np

ST = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2/auto_research/_stage/results_vll3")
OUT = Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)
G = 16
BINS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 9), (9, 13)]
CX = np.arange(256) % G; CY = np.arange(256) // G


def cheb(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return np.maximum(np.abs(a % G - b % G), np.abs(a // G - b // G))


def pan_src(v, dr, extra=0):
    """미래 슬롯 i 칸 s 의 출처 칸 (튜블릿 L 기준; extra 튜블릿만큼 더 과거면 extra*2 프레임 추가)."""
    src = np.full((8, 256), -1, np.int64)
    for i in range(8):
        xp = 16 * CX + 8 + dr * v * (2 * (i + 1) + 2 * extra)
        ok = (xp >= 0) & (xp < 256)
        src[i, ok] = CY[ok] * G + (xp[ok] // 16)
    return src


class Reach:
    def __init__(self, name):
        d = ST / name; self.d = d
        m = json.load(open(d / "meta.json")); self.meta = m; self.arms = m["arms"]; self.items = m["items"]; self.n = len(self.items)
        self.C = np.load(d / "arm_c.npy").astype(np.int64)
        self.CT = np.load(d / "arm_ct.npy").astype(np.float32); self.CC = np.load(d / "arm_cc.npy").astype(np.float32)
        self.E = np.load(d / "enc.npy").astype(np.float32)
        if (d / "src.npy").exists():
            self.SRC = np.load(d / "src.npy").astype(np.int64)
            self.D = np.nan_to_num(np.load(d / "disp.npy").astype(np.float32), nan=-1.0)
            self.V = None
        else:
            self.SRC = np.stack([pan_src(it["v"], it["dir"]) for it in self.items])
            self.D = np.stack([np.broadcast_to((it["v"] * (np.arange(8) + 1) / 8.0)[:, None], (8, 256)) for it in self.items]).astype(np.float32)
            self.V = np.array([it["v"] for it in self.items])
        self.done = self.C[:, 0, 0, 0] >= 0
        self.VAL = (self.SRC >= 0) & self.done[:, None, None] & (self.D >= 0)
        self.HE = (cheb(self.E[:, 0].astype(np.int64), self.SRC) <= 1) & self.VAL   # 합의 (encoder 판독 적중)
        self.SL = np.broadcast_to(np.arange(8)[None, :, None], self.SRC.shape)
        self.CL = np.broadcast_to(np.arange(self.n)[:, None, None], self.SRC.shape)
        self.ai = {a: k for k, a in enumerate(self.arms)}

    def hit(self, arm, tol=1):
        return (cheb(self.C[:, self.ai[arm]], self.SRC) <= tol) & self.VAL

    def own(self, arm):
        return (self.C[:, self.ai[arm]] == np.arange(256)[None, None, :]) & self.VAL


def rate_ci(num, den, rng, B=400):
    """clip 단위 합 (num, den: (n,)) → pooled 비율 + clip bootstrap CI."""
    ok = den > 0
    if ok.sum() == 0: return [float("nan")] * 3
    idx = np.where(ok)[0]; r = num[idx].sum() / den[idx].sum()
    bs = np.empty(B)
    for b in range(B):
        s = rng.choice(idx, len(idx)); bs[b] = num[s].sum() / den[s].sum()
    return [round(float(r), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def curve(hit, mask, D, rng, B=400, bins=BINS):
    out = []
    for lo, hi in bins:
        m = mask & (D >= lo) & (D < hi)
        out.append(dict(lo=lo, hi=hi, n=int(m.sum()), A=rate_ci((hit & m).sum((1, 2)).astype(float), m.sum((1, 2)).astype(float), rng, B)))
    return out


def r50_from_A(A, bins=BINS):
    a = np.asarray(A, float); mids = np.array([0.5 * (lo + hi) for lo, hi in bins])
    ok = np.isfinite(a); a = a[ok]; mids = mids[ok]
    if len(a) == 0: return float("nan")
    half = 0.5 * a[0]
    for j in range(1, len(a)):
        if a[j] < half:
            f = (a[j - 1] - half) / max(a[j - 1] - a[j], 1e-9); return float(mids[j - 1] + f * (mids[j] - mids[j - 1]))
    return float("inf")


def r50_boot(hit, mask, D, rng, B=300, bins=BINS):
    """R50 의 clip bootstrap: clip 재표본마다 A(d) 를 다시 만들어 R50."""
    n = hit.shape[0]
    nums = np.stack([(hit & mask & (D >= lo) & (D < hi)).sum((1, 2)) for lo, hi in bins], 1).astype(float)   # (n, nb)
    dens = np.stack([(mask & (D >= lo) & (D < hi)).sum((1, 2)) for lo, hi in bins], 1).astype(float)
    def r50_of(idx):
        A = nums[idx].sum(0) / np.maximum(dens[idx].sum(0), 1); A[dens[idx].sum(0) == 0] = np.nan
        return r50_from_A(A, bins)
    idx = np.where(dens.sum(1) > 0)[0]; r = r50_of(idx)
    bs = np.array([r50_of(rng.choice(idx, len(idx))) for _ in range(B)])
    bs = bs[np.isfinite(bs)]
    return [round(r, 3), round(float(np.percentile(bs, 2.5)), 3) if len(bs) else float("nan"), round(float(np.percentile(bs, 97.5)), 3) if len(bs) else float("nan")]


def paired_delta(hitA, hitB, mask, D, rng, B=400, bins=BINS):
    out = []
    for lo, hi in bins:
        m = mask & (D >= lo) & (D < hi)
        num = ((hitA & m).sum((1, 2)) - (hitB & m).sum((1, 2))).astype(float); den = m.sum((1, 2)).astype(float)
        out.append(dict(lo=lo, hi=hi, n=int(m.sum()), d=rate_ci(num, den, rng, B)))
    return out


def dump(name, obj):
    json.dump(obj, open(OUT / f"{name}.json", "w"), indent=1, default=float)
    print("saved", OUT / f"{name}.json")
