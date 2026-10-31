#!/usr/bin/env python3
r"""라벨 없는 공간 지도 주입 — 학습 전 상한 시험 (2026-10-30, z_analysis/release ②).

묻는 것: 문맥 z 만으로 (라벨 없이) "물체가 마지막으로 보인 블록의 물체 토큰을, 문맥에서 추정한 변위로 미래 칸에 옮긴 지도" 를 만들어
  릴리즈 p 에 덮어쓰면 late 사라짐 A 가 오르나. 같은 판에서 라벨 판 (transfer_gap E24: 마지막 본 자리 10 · 1 차 45 · 2 차 = 참 자리 71.7)
  과 나란히 낸다. 라벨 판에 얼마나 다가가는지가 "지도 모듈이 문맥 z 만으로 될 수 있나" 의 답이다.

⚠️ 10-30 첫 시도 (아래 '중앙값' 탐지기) 는 실패했다: ViT-H 문맥 토큰은 배경도 블록마다 크게 흔들려 (모든 칸 cos 차 ≈ 0.6) 물체가 안 두드러지고,
  최대점이 빈 장면에서도 같은 크기 (0.706) 로 엉뚱한 칸에 잡힌다 (smoke 4 block, 자리 오차 8.6 칸). → 사용자 10-30 "지도 모듈에 supervision 을 따로":
  ★ 지금 판 = '선형' 탐지기: 문맥 z 토큰마다 "물체가 이 칸에 있나" 를 맞히는 로지스틱 판독기 (1280 → 1) 를 VOBench 라벨로 학습한다.
    학습 = 평가 장면을 뺀 late clip (장면 단위 2 겹) + 보임 조건 clip 전부, 보이는 블록의 물체 상자 칸 = 양성, 나머지 · 가린 블록 · 빈 장면 = 음성.
    두드러짐 s[t,c] = 판독기 logit. 블록 점수 = 상위 3 칸 logit 평균, 보임 ⇔ 블록 점수 ≥ τ (τ = 학습 clip 의 블록 보임 라벨로 균형 정확도 최대,
    빈 장면 블록 = 안 보임). 보이는 블록이 없으면 지도를 비운다 (주입 안 함). 자리 가중치는 시그모이드 확률. 나머지 (자리 · 외삽 · 내용 · 주입) 는 아래와 같다.
    (첫 선형 판: 문턱 = 확률 0.5 → 빈 장면에서도 0.93–0.98 로 반응하고 가린 블록을 보임으로 잡아 가림막 토큰을 옮겼다, smoke 4 block)
    이 판독기 하나만 라벨을 쓴다 (특권 정보 — 시뮬레이터 자리 라벨). 자리 · 시점 · 운동은 판독기 출력에서 계산한다.
중앙값 탐지기 (기록, `--detector median`; clip 마다 그 clip 의 문맥 z (8 블록 × 256 칸 × 1280) 만 쓴다):
  배경   B[c]    = 문맥 8 블록의 칸별 중앙값 (물체는 움직여서 한 칸에 1–2 블록만 머문다 → 중앙값 ≈ 배경 · 가림막)
  두드러짐 s[t,c] = 1 − cos(z[t,c], B[c])
  보임   t 가 보임 ⇔ max_c s[t,c] ≥ ½ · max_t max_c s[t,c];  t̂ = 마지막 보임 블록
  자리   ĉ(t)   = argmax 칸 주변 5×5 에서 (s − ½·max) 가중 중심 (칸 단위 실수)
  변위   1 차: 보이는 블록 전체의 ĉ 에 직선 최소제곱 / 2 차: 2 차식 최소제곱 (보이는 블록 < 3 이면 1 차, < 2 면 멈춤)
         (첫 판은 라벨 판처럼 마지막 2 · 3 블록만 썼다 → 자리 오차 0.2–1.3 칸이 미래 8 블록에서 3–10 칸으로 불었다, smoke 4 block)
  내용   z[t̂] 의 ĉ(t̂) 3×3 ∪ 5×5 안 s ≥ ½·max 칸, 한 칸 넓힘 (라벨 판의 외곽 상자 + 1 칸과 맞춤)
  지도   미래 블록 t 마다 내용을 ĉ(t̂) → 외삽 자리로 평행 이동 (미래 8 × 16 × 16, 채운 칸만)
  주입   x = 릴리즈 p 복사 → 지도의 채운 칸을 덮어씀 (원래 자리는 안 지움 — E24 와 같음). cross-attn 모듈의 학습 없는 대역
판: late · 이동 (flat_static 제외) · permanence 60 block (A = 물체 문맥, B = 빈 문맥) + 모양 60 block (A). 빈 문맥에도 같은 절차를
  적용한다 (라벨로 "물체 없음" 을 알려 주지 않는다).
채점: 표준 (전체 토큰 L1, matched 짝), E24 와 같은 배열 (bf16 캐시) · 같은 짝.
판정 (사전 등록, 결과 전에 적음):
  max(A(지도@2 차), A(지도@차수 자동)) ≥ 60 이고 라벨 판 z@Oca 와 15 pt 안 → "문맥 z 만으로 지도 상한의 대부분" (모듈 = 학습이 지도를 쓰게만 하면 됨)
  그 값 ≤ 20 → "문맥 z 에서 물체 · 운동 뽑기가 병목" (차수 자동 규칙은 smoke 뒤 · 전체 실행 전에 추가)
  B (빈 문맥) 가 릴리즈보다 10 pt 넘게 떨어지면 "지도가 빈 장면에 헛물체를 그린다" 로 따로 적는다
진단: t̂ 와 라벨 마지막 보임 블록 일치율 · 보임 블록 자리 오차 (칸) · 미래 자리 오차 · 참 칸 1 칸 안 비율 · 빈 문맥의 최대 두드러짐.

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/release_map_inject.py [--limit 4]        # vll5 CPU (캐시가 노드 로컬), 수 분
  → z_analysis/release/02_map_inject/results/map_inject.json · map_inject_rows.jsonl
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transfer_gap_common as C  # noqa: E402
import transfer_gap_margin as MG  # noqa: E402
import transfer_gap_inject2 as I2  # noqa: E402

NT, NF, S, D, G = C.NT, C.NF, C.S, C.D, C.GRID
OUT = Path(__file__).resolve().parents[3] / "z_analysis/release/02_map_inject/results"
MOTIONS = I2.MOTIONS


def saliency_median(z):
    zn = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-6)
    B = np.median(z, axis=0); Bn = B / (np.linalg.norm(B, axis=-1, keepdims=True) + 1e-6)
    return 1.0 - (zn * Bn[None]).sum(-1)                               # (8, 256)


def free_map(z, det=None):
    """z (8, 256, D) float32 → 지도 추정 (dict). det = (w, b, mu, sd) 선형 탐지기 · None 이면 중앙값 탐지기. 물체 없음이면 None."""
    if det is None:
        s = saliency_median(z); peak = s.max(1); top = peak.max()
        vis = [t for t in range(NF) if peak[t] >= 0.5 * top]
    else:
        w, b0, mu, sd, tau = det
        lg = ((z - mu) / sd) @ w + b0                                  # (8, 256) logit
        peak = np.sort(lg, axis=1)[:, -3:].mean(1); top = peak.max()
        vis = [t for t in range(NF) if peak[t] >= tau]
        if not vis:
            return None
        s = 1.0 / (1.0 + np.exp(-lg))
    th = vis[-1] if vis else NF - 1
    cen = {}
    for t in vis:
        st = s[t].reshape(G, G); r0, c0 = divmod(int(st.argmax()), G)
        r_lo, r_hi, c_lo, c_hi = max(r0 - 2, 0), min(r0 + 2, G - 1), max(c0 - 2, 0), min(c0 + 2, G - 1)
        w = np.clip(st[r_lo:r_hi + 1, c_lo:c_hi + 1] - 0.5 * st.max(), 0, None)
        rr, cc = np.mgrid[r_lo:r_hi + 1, c_lo:c_hi + 1]
        cen[t] = np.array([(w * rr).sum() / w.sum(), (w * cc).sum() / w.sum()]) if w.sum() > 0 else np.array([r0, c0], float)
    # 내용: t̂ 의 3×3 ∪ 5×5 안 두드러진 칸
    rc = np.round(cen[th]).astype(int); st = s[th].reshape(G, G)
    cells = set(I2.box3((int(rc[0]), int(rc[1]))))
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            r, c = rc[0] + dr, rc[1] + dc
            if 0 <= r < G and 0 <= c < G and st[r, c] >= 0.5 * st.max():
                cells.add(int(r * G + c))
    cells = set(MG.dilate(cells))
    # 외삽
    def clip_rc(x):
        return (int(np.clip(np.round(x[0]), 0, G - 1)), int(np.clip(np.round(x[1]), 0, G - 1)))
    last = [clip_rc(cen[th])] * NF
    ts = np.array(vis, float); X = np.stack([cen[int(t)] for t in vis])
    fit = lambda deg: [clip_rc(np.array([np.polyval(np.polyfit(ts, X[:, 0], deg), t), np.polyval(np.polyfit(ts, X[:, 1], deg), t)]))
                       for t in range(NF, NT)]
    cv = fit(1) if len(vis) >= 2 else last
    ca = fit(2) if len(vis) >= 3 else cv
    # 차수 자동 (라벨 없음, 사전 규칙): 2 차식이 문맥 잔차를 절반 아래로 줄이고 1 차 잔차가 블록당 0.3 칸² 을 넘을 때만 2 차
    auto = cv
    if len(vis) >= 4:
        rss = lambda deg: float(sum(((np.polyval(np.polyfit(ts, X[:, i], deg), ts) - X[:, i]) ** 2).sum() for i in (0, 1)))
        r1, r2 = rss(1), rss(2)
        auto = ca if (r2 < 0.5 * r1 and r1 > 0.3 * len(vis)) else cv
    return dict(th=th, vis=vis, cen=cen, cells=sorted(cells), src=clip_rc(cen[th]), last=last, cv=cv, ca=ca, auto=auto,
                peak=peak.tolist(), top=float(top))


def train_detector(cx, hold_blocks):
    """선형 물체 칸 탐지기. hold_blocks (평가 장면) 는 학습에서 뺀다. 양성 = 보이는 블록의 물체 상자 칸 (외곽 상자, pad 0), 음성 = 나머지 칸 (블록당 24 칸 표집)
    · 가린 블록 · 빈 장면. late clip (z_late_ctx) + 보임 조건 clip (z_vis_ctx)."""
    import torch
    M = cx.M; rng = np.random.default_rng(0); X, Y = [], []
    sets = [(cx.Z, json.load(open(C.TG / "z_late_meta.json"))["tp"])]
    fz = C.TG / "z_vis_ctx.npy"
    if fz.exists():
        sets.append((np.load(fz, mmap_mode="r"), json.load(open(C.TG / "z_vis_meta.json"))["tp"]))
    for ZZ, tps in sets:
        for q, j in enumerate(tps):
            m = M[j]
            if m["block_id"] in hold_blocks:
                continue
            z = np.asarray(ZZ[q], np.float32)
            empty = C.label_of(m) == "empty"
            bi = cx.bmap[m["block_id"]]; arm = 0 if m["variant"] == "pos_a" else 1
            uv, ext, hidd = cx.UV[bi, arm], cx.EXT[bi, arm], cx.HID[bi, arm]
            for t in range(NF):
                pos = set() if (empty or hidd[t]) else set(C.region(uv[t], ext[t], pad=0))
                neg = [c for c in range(S) if c not in pos]
                negs = rng.choice(neg, size=min(24, len(neg)), replace=False)
                for c in pos:
                    X.append(z[t, c]); Y.append(1.0)
                for c in negs:
                    X.append(z[t, c]); Y.append(0.0)
    X = np.stack(X).astype(np.float32); Y = np.array(Y, np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt = torch.from_numpy((X - mu) / sd); Yt = torch.from_numpy(Y)
    pw = torch.tensor(float((1 - Y).sum() / max(Y.sum(), 1)))
    w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=300, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, Yt, pos_weight=pw) + 1e-3 * (w ** 2).sum()
        loss.backward(); return loss
    opt.step(closure)
    acc = float((((Xt @ w + b) > 0).float() == Yt).float().mean())
    wn, bn = w.detach().numpy(), float(b.detach())
    # 블록 보임 문턱 τ: 학습 clip 의 (clip, 블록) 점수 = 상위 3 칸 logit 평균, 라벨 = 물체가 보이는 블록
    sc_, lab_ = [], []
    for ZZ, tps in sets:
        for q, j in enumerate(tps):
            m = M[j]
            if m["block_id"] in hold_blocks:
                continue
            z = np.asarray(ZZ[q], np.float32)
            lg = ((z - mu) / sd) @ wn + bn
            empty = C.label_of(m) == "empty"
            bi = cx.bmap[m["block_id"]]; arm = 0 if m["variant"] == "pos_a" else 1
            hidd = cx.HID[bi, arm]
            for t in range(NF):
                sc_.append(float(np.sort(lg[t])[-3:].mean())); lab_.append(0 if (empty or hidd[t]) else 1)
    sc_, lab_ = np.array(sc_), np.array(lab_)
    best = None
    for tau in np.quantile(sc_, np.linspace(0.01, 0.99, 197)):
        pr = sc_ >= tau
        ba = 0.5 * (pr[lab_ == 1].mean() + (~pr[lab_ == 0]).mean())
        if best is None or ba > best[0]:
            best = (ba, float(tau))
    print(f"detector: {len(Y)} 토큰 (양성 {int(Y.sum())}) · 토큰 train acc {acc:.3f} · 블록 보임 τ {best[1]:.2f} (균형 정확도 {best[0]:.3f}, "
          f"블록 {len(lab_)}, 보임 {int(lab_.sum())})", flush=True)
    return (wn, bn, mu, sd, best[1])


def run(limit, detector="linear"):
    cx = MG.Ctx(); M = cx.M
    sel = sorted([b for b, vv in cx.blocks.items() if M[vv["pos_a"]]["occlusion"] == "late" and M[vv["pos_a"]]["motion"] != "flat_static"
                  and M[vv["pos_a"]]["violation"] in ("permanence", "shape")], key=int)
    if limit:
        sel = [b for b in sel if M[cx.blocks[b]["pos_a"]]["violation"] == "permanence"][:limit] + \
              [b for b in sel if M[cx.blocks[b]["pos_a"]]["violation"] == "shape"][:2]
    rows = []; t0 = time.time()
    trb = set(cx.trb)
    fold_of = {b: ("A" if b in trb else "B") for b in sel}
    dets = {}
    if detector == "linear":
        allb = {m["block_id"] for m in M}
        dets["A"] = train_detector(cx, hold_blocks=allb & trb)       # 겹 A 장면 평가용 = 겹 B 장면으로 학습
        dets["B"] = train_detector(cx, hold_blocks=allb - trb)
    for n_, b in enumerate(sel):
        geo = cx.geo(b); uv, ext, hid, tl = geo["uv"], geo["ext"], geo["hid"], geo["tl"]
        ja, jb = cx.blocks[b]["pos_a"], cx.blocks[b]["pos_b"]; m = M[ja]
        trues = [I2.cell_rc(uv[t]) for t in range(NF, NT)]
        lab_cv, _ = I2.cv_centers(uv, hid, tl); lab_ca = I2.ca_centers(uv, hid, tl); lab_end = [I2.cell_rc(uv[tl])] * NF
        src_cells = C.region(uv[tl], ext[tl]); src_c = I2.cell_rc(uv[tl])
        for var, j in (("pos_a", ja), ("pos_b", jb)):
            if m["violation"] == "shape" and var == "pos_b":
                continue
            p = cx.arr(cx.P, j); hp = cx.arr(cx.H, j)[NF:]; hi = cx.arr(cx.HI, cx.imp_of[j]); z = cx.arr(cx.Z, cx.zslot[j])
            sc = lambda x: dict(acc=float(np.abs(x - hi).mean()) > float(np.abs(x - hp).mean()),
                                m=(float(np.abs(x - hi).mean()) - float(np.abs(x - hp).mean())) / float(np.abs(x - hp).mean()))
            fm = free_map(z, dets.get(fold_of[b]))
            r = dict(block=b, objset=m["objset"], violation=m["violation"], motion=m["motion"], k=m["k"],
                     dirn="A" if var == "pos_a" else "B", test=b not in cx.trb, tl=int(tl),
                     th=int(fm["th"]) if fm else -1, top=fm["top"] if fm else 0.0, detected=int(fm is not None))
            if fm is not None and (var == "pos_a" or m["violation"] == "shape"):
                r["th_ok"] = int(fm["th"] == tl)
                r["cen_err"] = float(np.mean([np.hypot(*(fm["cen"][t] - np.array([uv[t][1] / C.CELL - 0.5, uv[t][0] / C.CELL - 0.5])))
                                              for t in fm["vis"] if not hid[t]] or [np.nan]))
                for nm, fut in (("last", fm["last"]), ("cv", fm["cv"]), ("ca", fm["ca"]), ("auto", fm["auto"])):
                    d = [max(abs(a[0] - c[0]), abs(a[1] - c[1])) for a, c in zip(fut, trues)]
                    r[f"d_{nm}"] = float(np.mean(d)); r[f"w1_{nm}"] = float(np.mean([x <= 1 for x in d]))
            r["rel"] = sc(p); r["copy"] = sc(np.broadcast_to(z[7], (NF, S, D)).copy())
            content_lab = {cc: z[tl, cc] for cc in src_cells}
            for nm, key in (("free@last", "last"), ("free@cv", "cv"), ("free@ca", "ca"), ("free@auto", "auto"), ("free@true", None)):
                x = p.copy()
                if fm is not None:
                    content_free = {cc: z[fm["th"], cc] for cc in fm["cells"]}
                    centers = trues if key is None else fm[key]
                    for k in range(NF):
                        I2.place(x, k, content_free, fm["src"], centers[k])
                r[nm] = sc(x)
            for nm, centers in (("lab@last", lab_end), ("lab@cv", lab_cv), ("lab@ca", lab_ca)):
                x = p.copy()
                for k in range(NF):
                    I2.place(x, k, content_lab, src_c, centers[k])
                r[nm] = sc(x)
            rows.append(r)
        if n_ % 20 == 0:
            print(f"{n_}/{len(sel)} {time.time() - t0:.0f}s", flush=True)
    return rows


def summarize(rows):
    keys = ["rel", "copy", "free@last", "free@cv", "free@ca", "free@auto", "free@true", "lab@last", "lab@cv", "lab@ca"]
    out = {}
    for viol in ("permanence", "shape"):
        for dirn in ("A", "B"):
            for split in ("all", "test"):
                for mv in ["all"] + MOTIONS:
                    sel = [r for r in rows if r["violation"] == viol and r["dirn"] == dirn and (split == "all" or r["test"])
                           and (mv == "all" or r["motion"] == mv)]
                    if not sel:
                        continue
                    d = dict(n=len(sel))
                    for k in keys:
                        d[k] = C.bootstrap_mean([100 * float(r[k]["acc"]) for r in sel], [r["block"] for r in sel])
                    for g in ("detected", "th_ok", "cen_err", "d_last", "d_cv", "d_ca", "d_auto", "w1_last", "w1_cv", "w1_ca", "w1_auto", "top"):
                        v = [r[g] for r in sel if g in r and r[g] == r[g]]
                        if v:
                            d[g] = float(np.mean(v))
                    out[f"{viol}|{dirn}|{split}|{mv}"] = d
    A = out["permanence|A|all|all"]; Bd = out["permanence|B|all|all"]
    g = lambda d, k: d[k]["mean"]
    ver = dict(A_free_auto=g(A, "free@auto"), A_free_ca=g(A, "free@ca"), A_lab_ca=g(A, "lab@ca"), A_free_cv=g(A, "free@cv"), A_lab_cv=g(A, "lab@cv"),
               A_free_last=g(A, "free@last"), A_free_true=g(A, "free@true"), A_rel=g(A, "rel"), A_copy=g(A, "copy"),
               B_rel=g(Bd, "rel"), B_free_ca=g(Bd, "free@ca"))
    best = max(ver["A_free_ca"], ver["A_free_auto"])
    if best >= 60 and ver["A_lab_ca"] - best <= 15:
        ver["verdict"] = "문맥 z 만으로 지도 상한의 대부분"
    elif best <= 20:
        ver["verdict"] = "문맥 z 에서 물체 · 운동 뽑기가 병목"
    else:
        ver["verdict"] = "중간"
    ver["B_harm"] = bool(ver["B_rel"] - ver["B_free_ca"] > 10)
    return out, ver


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--detector", default="linear", choices=["linear", "median"]); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    rows = run(a.limit, a.detector); summ, ver = summarize(rows)
    suf = ("" if a.detector == "linear" else "_median") + (f"_smoke{a.limit}" if a.limit else "")
    json.dump(dict(definition=__doc__, verdict=ver, summary=summ), open(OUT / f"map_inject{suf}.json", "w"), indent=1, ensure_ascii=False, default=float)
    with open(OUT / f"map_inject_rows{suf}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=float, ensure_ascii=False) + "\n")
    print(json.dumps(ver, ensure_ascii=False, default=float, indent=1), f"{time.time() - t0:.0f}s")
