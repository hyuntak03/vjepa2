#!/usr/bin/env python3
"""RollOut v3 — **predictor 행동 성적표**. 그래프를 눈으로 읽는 대신 숫자 몇 개로.

state evolution 네 능력 (CLAUDE.md, 2026-09-19 사용자 정의) 에 맞춘 지표:

  읽기   E0    첫 미래 튜블릿의 위치 L2 오차 (px). 기준 = `z` 의 E0 (자의 바닥)
  지속   H     물체를 들고 있는 튜블릿 수의 기대값 = Σ_t recall(t)  (P=32 면 최대 16)
         T50   recall 이 처음 50 % 아래로 가는 튜블릿
         R     깜빡임 — 한 번 '없다' 가 된 clip 중 나중에 다시 '있다' 가 되는 비율
  전이   s     속도 유지율. 1 = 문맥 끝 속도를 그대로 이어 감, 0 = 그 자리에 멈춤
         g     법칙 적용률. 1 = 법칙대로 (가속·감속·정지·낙하), 0 = 등속으로만 밀고 감
  영속   —     v3 에는 가림이 없어 **잴 수 없다**

s·g 는 clip 과 튜블릿을 모아 **한 번의 회귀**로 뽑는다 (clip 하나하나의 잡음은 평균으로 사라진다):

    Y(t) − X(−1)  =  s · V·(t+1)  +  g · D(t)  +  b          (2 축을 한 식에)
      X(−1)  문맥 끝 진실 위치       V     문맥 끝 한 걸음 (px/튜블릿)
      D(t)   진실이 등속 연장에서 벗어난 양 = X(t) − (X(−1) + V·(t+1))
      Y(t)   자가 읽은 위치 (치우침 뺀 값)

  → 진실을 그대로 따라가면 s = g = 1, 등속으로만 가면 s = 1, g = 0, 그 자리에 멈추면 s = g = 0.
  `wall`·`ledge` 는 **불가능 미래가 곧 등속 연장**이라 g 가 그대로 "어느 미래를 골랐나" 다 (1 = 정지/낙하, 0 = 통과/부유).
  ⚠️ `wall` 은 D = −V·(t+1) 이라 두 항이 겹친다 (공선) → **s 를 1 로 고정**하고 g 만 푼다 (표에 표시).
  ⚠️ `flat_v` 는 D ≈ 0 이라 g 가 정의되지 않는다 — s 만 읽는다.

⚠️ **자 검증이 먼저다** (사용자 지시 2026-09-12). 같은 지표를 `z` (창 전체를 보는 encoder = 자의 천장) 로도 낸다.
   `z` 의 s·g 가 1 근처가 아니면 자가 그 크기의 변화를 못 보는 것이고, 그 법칙의 `p` 값은 읽지 않는다.
⚠️ 쓰는 칸 — 진실이 화면 안이고, **읽는 표현과 `z` 가 둘 다 '있다'** 고 한 튜블릿만 (자가 고장 난 칸을 뺀다).
   화면 가장자리 칸 (총 창 48f 이상이면 마지막 튜블릿, C=4 면 첫 미래 튜블릿) 은 뺀다.
⚠️ 신뢰구간은 **clip 재표집** (기본 1,000 회). 회귀는 clip 마다의 정규방정식 기여를 더해 푼다.

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  R3_OUT=training_v8 $P z_research/scripts/analysis/rollout3_behavior_metrics.py
  $P z_research/scripts/analysis/rollout3_behavior_metrics.py \\
      --specs <자폴더>:<readings폴더> ...                                     # 여러 자를 나란히
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))
from rollout3_paths import EXP, WIN, PRES, check   # noqa: E402

RESN, SPLIT = 144.0, 32
LAWS = ["flat_v", "flat_a", "flat_d", "ramp_a", "ramp_d", "arc", "ledge", "wall"]
NB = 1000
rng = np.random.default_rng(0)


def load(spec):
    """'자폴더:readings폴더' → dict. 문턱·치우침은 그 자의 것, readings 는 지문으로 짝 확인."""
    dec, win = spec.split(":")
    pres, w = EXP / dec, EXP / win
    z = np.load(w / "readings.npz", allow_pickle=True)
    check(z, pres)
    S = json.loads((pres / "summary.json").read_text())
    return dict(label=f"{dec} ({S.get('train_set', 'training_v6')})", z=z,
                thr={r: S["reps"][r]["attn"]["thr_val_fpr5"] for r in S["reps"]},
                bias=json.loads((pres / "attn_bias_px.json").read_text()))


def window(D, c, p):
    """한 창의 배열들. 좌표는 전부 px (정규화 × 144, 상수 오프셋은 차이에서 상쇄)."""
    z = D["z"]; POS = z["role"] == "roll"
    L = z["truth"][POS] * RESN; inf = z["in_frame"][POS]; tp = p // 2
    X = L[:, SPLIT:SPLIT + p].reshape(-1, tp, 2, 2).mean(2)                  # (n, tp, 2) 미래 진실
    Xc = L[:, SPLIT - c:SPLIT].reshape(-1, c // 2, 2, 2).mean(2)              # 문맥 진실
    x0, V = Xc[:, -1], Xc[:, -1] - Xc[:, -2]                                  # 문맥 끝 위치 · 한 걸음
    vis = inf[:, SPLIT:SPLIT + p].reshape(-1, tp, 2).all(2)
    keep_t = np.ones(tp, bool)                                                # 가장자리 칸 빼기
    if c + p >= 48:
        keep_t[-1] = False
    if c == 4:
        keep_t[0] = False
    out = dict(X=X, x0=x0, V=V, vis=vis & keep_t[None], sc=z["scenario"][POS], tp=tp, n=int(POS.sum()))
    for r in ("p", "z", "h"):
        k = f"C{c}_P{p}_{r}"
        if k in z.files:
            v = z[k][POS]
            out[r] = dict(Y=v[..., 0:2] * RESN - np.array(D["bias"][r], np.float32), say=v[..., 2] > D["thr"][r])
    return out


def ci_ratio(num, den):
    """clip 별 (분자, 분모) → 비율과 95 % 구간."""
    pt = num.sum() / max(den.sum(), 1e-9)
    idx = rng.integers(0, len(num), size=(NB, len(num)))
    b = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-9)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return pt, lo, hi


def hold(W, r, sel):
    """H (기대 보유 튜블릿) · T50 · R (깜빡임). clip 재표집 구간 포함."""
    say, vis = W[r]["say"][sel], W["vis"][sel]
    cols = np.where(vis.any(0))[0]
    rec = np.array([say[vis[:, t], t].mean() if vis[:, t].any() else np.nan for t in cols])
    H = float(np.nansum(rec))
    idx = rng.integers(0, len(say), size=(NB, len(say)))
    Hb = np.array([np.nansum([say[i][vis[i, t], t].mean() if vis[i, t].any() else np.nan for t in cols])
                   for i in idx[:200]])                                        # 구간은 200 회로 충분
    below = np.where(rec < 0.5)[0]
    T50 = int(cols[below[0]]) if len(below) else None
    # 깜빡임: 유효 칸만 이어 붙인 판정열에서 '없다' 뒤에 '있다' 가 오는가
    lost = revived = 0
    for s_, v_ in zip(say, vis):
        seq = s_[v_]
        if (~seq).any():
            lost += 1
            first = np.argmax(~seq)
            revived += int(seq[first + 1:].any())
    return dict(H=H, H_lo=float(np.percentile(Hb, 2.5)), H_hi=float(np.percentile(Hb, 97.5)), T50=T50,
                R=revived / lost if lost else None, n_lost=lost, n=len(say), H_max=len(cols))


def transfer(W, r, sel, fix_s=False):
    """s · g 를 한 회귀로. clip 마다 정규방정식 기여를 모아 재표집한다."""
    X, x0, V, vis = W["X"][sel], W["x0"][sel], W["V"][sel], W["vis"][sel]
    Y, say = W[r]["Y"][sel], W[r]["say"][sel]
    sayz = W["z"]["say"][sel]
    t1 = (np.arange(W["tp"]) + 1)[None, :, None]
    A1 = V[:, None, :] * t1                                   # 등속 연장 (n, tp, 2)
    Dv = X - (x0[:, None, :] + A1)                            # 진실이 등속에서 벗어난 양
    y = Y - x0[:, None, :]
    m = vis & say & sayz                                      # 진실 화면 안 · 읽는 표현과 z 가 둘 다 '있다'
    n, tp = m.shape
    # 행 = (clip, t, 축). 설계 행렬 열 = [s, g, b_x, b_y] (s 고정이면 [g, b_x, b_y], y 에서 A1 을 뺀다)
    ex = np.zeros((n, tp, 2, 2)); ex[..., 0, 0] = 1; ex[..., 1, 1] = 1   # 축별 절편
    cols = ([Dv[..., None]] if fix_s else [A1[..., None], Dv[..., None]]) + [ex]
    Z = np.concatenate(cols, -1)                              # (n, tp, 2, k)
    yy = (y - A1) if fix_s else y
    w = m[..., None].astype(float)                            # 축 둘 다 같은 마스크
    G = np.einsum("ntak,ntal,nta->nkl", Z, Z, w)              # clip 별 ZᵀZ
    h = np.einsum("ntak,nta,nta->nk", Z, yy, w)               # clip 별 Zᵀy
    k = Z.shape[-1]
    reg = 1e-6 * np.eye(k)
    beta = np.linalg.solve(G.sum(0) + reg, h.sum(0))
    idx = rng.integers(0, n, size=(NB, n))
    Gb = G[idx].sum(1) + reg; hb = h[idx].sum(1)
    bb = np.linalg.solve(Gb, hb[..., None])[..., 0]
    # 공선 진단: 두 회귀항의 상관 (유효 칸)
    a1, d1 = A1[m].ravel(), Dv[m].ravel()
    corr = float(np.corrcoef(a1, d1)[0, 1]) if (not fix_s and a1.std() > 0 and d1.std() > 0) else None
    D_mag = float(np.linalg.norm(Dv[m], axis=-1).mean()) if m.any() else 0.0
    if fix_s:
        g, glo, ghi = beta[0], *np.percentile(bb[:, 0], [2.5, 97.5])
        return dict(s=1.0, s_lo=None, s_hi=None, g=float(g), g_lo=float(glo), g_hi=float(ghi),
                    fix_s=True, corr=None, D_mag=D_mag, n_cells=int(m.sum()))
    s, g = beta[0], beta[1]
    slo, shi = np.percentile(bb[:, 0], [2.5, 97.5]); glo, ghi = np.percentile(bb[:, 1], [2.5, 97.5])
    return dict(s=float(s), s_lo=float(slo), s_hi=float(shi), g=float(g), g_lo=float(glo), g_hi=float(ghi),
                fix_s=False, corr=corr, D_mag=D_mag, n_cells=int(m.sum()))


def speed_split(W, r, sel):
    """거리인가 시간인가 — 문맥 속도 3 분위의 T50 (arc 는 발사각 때문에 뺀다)."""
    v0 = np.abs(W["V"][:, 0]); na = sel & (W["sc"] != "arc")
    q = np.percentile(v0[na], [33.3, 66.7])
    out = []
    for lo, hi in ((-1, q[0]), (q[0], q[1]), (q[1], 1e9)):
        k = na & (v0 >= lo) & (v0 < hi) if lo >= 0 else na & (v0 < hi)
        say, vis = W[r]["say"][k], W["vis"][k]
        rec = np.array([say[vis[:, t], t].mean() if vis[:, t].any() else np.nan for t in range(W["tp"])])
        below = np.where(rec < 0.5)[0]
        out.append((float(v0[k].mean()), int(below[0]) if len(below) else None))
    return out


def fmt(v, lo=None, hi=None, d=2):
    if v is None:
        return "—"
    return f"{v:.{d}f}" + (f" [{lo:.{d}f}, {hi:.{d}f}]" if lo is not None else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="+", default=None,
                    help="자폴더:readings폴더 (여러 개면 나란히). 기본은 지금 경로 (R3_DECODER / R3_OUT)")
    ap.add_argument("--window", default="16x32", help="전이 지표를 잴 창 (기본 C16/P32)")
    ap.add_argument("--out", default=None, help="출력 md (기본 <첫 readings 폴더>/BEHAVIOR.md)")
    a = ap.parse_args()
    specs = a.specs or [f"{PRES.name}:{WIN.name}"]
    Ds = [load(s) for s in specs]
    wc, wp = (int(v) for v in a.window.split("x"))
    res = {}
    L = ["> ⚠️ **초안 — 지표 확정 전.** 공식·검증 규칙은 정했지만 사용자 결정 대기다. 이 표에서 **읽지 말 것**:",
         "> clip 별 깜빡임 R (`z` 도 크게 나와 자의 순간 실수와 못 가른다 → 곡선 기반 F 로 바꿀 예정),",
         "> 벗어남이 작은 법칙 (`flat_a`·`flat_d`·`ramp_a`·`ramp_d`) 의 g (자 위치 오차보다 작은 변화라 `z` 부터 흔들린다).",
         "> 읽을 수 있는 것: H · T50 · 시간/거리 구분 · `arc`·`wall` 의 g.", "",
         "# predictor 행동 성적표 — RollOut v3", "",
         "지표 정의·규칙은 `z_research/scripts/analysis/rollout3_behavior_metrics.py` 맨 위. 괄호는 clip 재표집 95 % 구간.",
         "**`z` 줄이 자의 천장이다** — `z` 가 기대값 (H≈최대, R≈0, s≈1, g≈1) 에서 멀면 그 칸의 `p` 는 읽지 않는다.", ""]
    for D in Ds:
        res[D["label"]] = rd = {}
        L += [f"## 자: `{D['label']}`", ""]
        # ── 지속: 문맥 길이별 (P=32) ──
        L += ["### 지속 — P = 32 창 (문맥 길이별)", "",
              "| C | `p` H (튜블릿) | `z` H | `p` T50 | `p` R 깜빡임 | `z` R |", "|---|---|---|---|---|---|"]
        rd["hold"] = {}
        for c in (4, 8, 16, 32):
            W = window(D, c, 32); sel = np.ones(W["n"], bool)
            hp, hz = hold(W, "p", sel), hold(W, "z", sel)
            rd["hold"][c] = dict(p=hp, z=hz)
            L.append(f"| {c} | {fmt(hp['H'], hp['H_lo'], hp['H_hi'], 1)} / {hp['H_max']} | {hz['H']:.1f} | "
                     f"{'t' + str(hp['T50']) if hp['T50'] is not None else '—'} | "
                     f"{fmt(100 * hp['R'] if hp['R'] is not None else None, d=0)} % ({hp['n_lost']} clip 중) | "
                     f"{fmt(100 * hz['R'] if hz['R'] is not None else None, d=0)} % ({hz['n_lost']}) |")
        # ── 시간인가 거리인가 ──
        W = window(D, wc, wp); sel = np.ones(W["n"], bool)
        sp = speed_split(W, "p", sel)
        L += ["", f"### 지속은 시간인가 거리인가 — C{wc}/P{wp}, 문맥 속도 3 분위 (arc 제외)", "",
              "| | 느림 | 중간 | 빠름 |", "|---|---|---|---|",
              "| 문맥 속도 (px/튜블릿) | " + " | ".join(f"{v:.2f}" for v, _ in sp) + " |",
              "| `p` T50 | " + " | ".join(f"t{t}" if t is not None else "—" for _, t in sp) + " |", "",
              "→ T50 이 속도와 무관하면 **시간 (튜블릿 수) 한계**, 빠를수록 이르면 **거리 한계**다.", ""]
        rd["speed"] = sp
        # ── 읽기 ──
        e0 = {}
        for r in ("p", "z"):
            m = W["vis"][:, 0] & W[r]["say"][:, 0] & W["z"]["say"][:, 0]
            e0[r] = float(np.linalg.norm(W[r]["Y"][m, 0] - W["X"][m, 0], axis=-1).mean())
        rd["E0"] = e0
        L += [f"### 읽기 — C{wc}/P{wp} 첫 미래 튜블릿 위치 오차", "",
              f"`p` E0 = **{e0['p']:.1f} px** · `z` E0 = {e0['z']:.1f} px (자의 바닥) · 1 칸 = 18 px", ""]
        # ── 전이 ──
        L += [f"### 전이 — C{wc}/P{wp}, 법칙별 s (속도 유지) · g (법칙 적용)", "",
              "| 법칙 | `p` s | `p` g | `z` s | `z` g | 법칙 크기 |D| | 칸 수 `p` |", "|---|---|---|---|---|---|---|"]
        rd["transfer"] = {}
        for law in LAWS:
            sel = W["sc"] == law
            fix = law == "wall"
            tp_, tz_ = transfer(W, "p", sel, fix), transfer(W, "z", sel, fix)
            rd["transfer"][law] = dict(p=tp_, z=tz_)
            gcell = lambda t: "—" if law == "flat_v" else fmt(t["g"], t["g_lo"], t["g_hi"])
            scell = lambda t: "1 (고정)" if t["fix_s"] else fmt(t["s"], t["s_lo"], t["s_hi"])
            L.append(f"| `{law}` | {scell(tp_)} | {gcell(tp_)} | {scell(tz_)} | {gcell(tz_)} | "
                     f"{tp_['D_mag']:.1f} px | {tp_['n_cells']} |")
        L += ["", "- `flat_v` 는 법칙이 등속이라 g 가 없다 (|D| ≈ 0). `wall` 은 두 항이 겹쳐 s 를 1 로 고정했다.",
              "- `ledge`·`wall` 의 g 는 **어느 미래를 골랐나** 다 — 1 = 가능 (낙하·정지), 0 = 불가능 (부유·통과).", ""]
    out = Path(a.out) if a.out else EXP / specs[0].split(":")[1] / "BEHAVIOR.md"
    L += ["## 재현", "", "```bash", "P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python",
          f"$P z_research/scripts/analysis/rollout3_behavior_metrics.py --specs {' '.join(specs)} --window {a.window}",
          "```"]
    out.write_text("\n".join(L) + "\n")
    (out.with_suffix(".json")).write_text(json.dumps(res, indent=1, default=float, ensure_ascii=False))
    print("\n".join(L)); print(f"→ {out}")


if __name__ == "__main__":
    main()
