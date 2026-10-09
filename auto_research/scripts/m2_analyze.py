#!/usr/bin/env python3
"""M2 분석 — 경계 패칭 (v11). 2 판: 2026-09-25 적대 검증 (VERIFY_0925, B6 · B7 · B8 · B3 분석 측) 반영.

사용: python m2_analyze.py [원자료 폴더 = v11_m2]   (환경변수 OUT_TAG 가 출력 파일 이름 접미사)
입력: scores.npy (n, 6 팔, 2 [pos, imp], 8 슬롯) · loc.npy (n, 6, 8, 4 = y, x, maxcos, median) · dist.npy (n, 6, 8) · meta.json
출력: exp_results/m2/m2_v11{OUT_TAG}.json
단위 = late 쌍 (pos, imp) × early 쌍둥이. CI = late block 군집 bootstrap (계획의 '궤적 군집' 은 v11 궤적이 운동당 2 개라 불가능;
E 재사용을 반영한 연결성분 군집 CI 를 머리 수치에 병기).  표적 = 표준 양방향 target encoder (LN(h)); 물체 템플릿 = E_pos 의 h 진실 칸 토큰.

1 판 (보존 — 1 판 문서의 수치를 그대로 재현한다):
  "all|…" / "t0-3|…" / "t4-7|…"  채점 (모든 쌍; E 짝 block 불일치 포함) · closure_R_* ·
  object_signal_legacy  (문턱 = E 팔 슬롯 0–2 P10 을 **빈 장면 vanish B 까지 넣은 풀** 에서 = 0.234, 위치 귀무 = 운동 안 순열 (절반이 같은 궤적),
                         화면 밖 진실 칸을 가장자리 칸으로 자름)
2 판 (기존 키 object_signal · objectness_ref · objectness_vanishA_pos 는 고친 값으로 덮고, 나머지는 추가 키):
  - 문턱: has_obj 풀 P10 (주) · 읽기 규칙 1 공통 0.357 · 연속 contrast · 스윕 (legacy 0.234 · has_obj P25 · 빈 장면 P90/P95)
  - 위치 귀무 = 같은 운동의 **다른 궤적** 진실 칸에 대한 정확 기대값 (움직임: 거울 궤적 1 개, 정지: 다른 자리 7 개)
  - 화면 밖 진실 칸 (움직이는 쌍의 t7 전부) = NaN → 창 t0-1 / t2-4 / t5-6
  - 운동별 역사 대비 (L+Eb)−E · Eb_only−E · Lb_only−L · (E+Lb)−L  (CI)
  - 채점: E_pos·E_imp 가 같은 block 인 부분집합 (matched pair 유효) · 그중 post 외형까지 같은 부분집합
  - k 별 회복/파괴, 같은 k / 다른 k 쌍둥이, 빈 장면 기저율, 닫힘 평균 거리, 설계 사실 (sym_k · 가림막 높이 · E 재사용 · 선택률 ·
    hidden_measured), 진행 방향 뒤처짐
"""
import collections, csv, json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
I = Path(sys.argv[1] if len(sys.argv) > 1 else "/data2/local_datasets/world/world_analysis/cache/auto_research/v11_m2")
OUT = ROOT / "auto_research/exp_results/m2"; OUT.mkdir(parents=True, exist_ok=True)
TAG = os.environ.get("OUT_TAG", "")
META_CSV = "/local_datasets/world/world_analysis/IntPhysGen_v11/metadata.csv"
meta = json.load(open(I / "meta.json")); P = meta["pairs"]; A = meta["arms"]; n = len(P)
S = np.load(I / "scores.npy"); LOC = np.load(I / "loc.npy"); DD = np.load(I / "dist.npy")
assert A == ["L", "L+Eb", "Lb_only", "E", "E+Lb", "Eb_only"], A
idx = {r["video_id"]: r for r in csv.DictReader(open(ROOT / "data_csv/intphysgen_v11_full/index_probe.csv"))}
viol = np.array([idx[p["L_imp"]]["violation_type"] for p in P]); mot = np.array([p["motion"] for p in P])
pid = np.array([p["pair_id"] for p in P]); kk = np.array([p["k"] for p in P]); samek = np.array([p["same_k"] for p in P])
cluster = np.array([idx[p["L_pos"]]["block_id"] for p in P])                 # late block (쌍 A·B 가 같은 block)
ok = ~np.isnan(S).any((1, 2, 3))
print("쌍", n, "유효", ok.sum(), "군집", len(set(cluster[ok])))

# ════════════════════════════ 1 판 (그대로; rng 소비 순서를 바꾸지 않는다) ════════════════════════════
rng = np.random.default_rng(0)
def acc(sl):
    s = np.nanmean(S[:, :, :, sl], -1)                     # (n, arm, 2)
    return (s[:, :, 1] > s[:, :, 0]).astype(float) + 0.5 * (s[:, :, 1] == s[:, :, 0])
def boot(vals, mask, B=1000):
    cl = cluster[mask]; u = np.unique(cl); idxs = {c: np.where(cl == c)[0] for c in u}; v = vals[mask]
    bs = [v[np.concatenate([idxs[c] for c in rng.choice(u, len(u))])].mean(0) for _ in range(B)]
    return v.mean(0), np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
out = {}
for sl, nm in ((slice(0, 8), "all"), (slice(0, 4), "t0-3"), (slice(4, 8), "t4-7")):
    a = acc(sl)
    groups = {"ALL": ok}
    for mo in ("static", "moving_flat", "moving"): groups[mo] = ok & (mot == mo)
    for v in ("vanish", "shape", "color"):
        groups[v] = ok & (viol == v)
        for d in ("A", "B"): groups[f"{v}|{d}"] = ok & (viol == v) & (pid == d)
    for mo in ("static", "moving_flat", "moving"):
        groups[f"{mo}|vanish|A"] = ok & (mot == mo) & (viol == "vanish") & (pid == "A")
    for g, m in groups.items():
        if m.sum() < 20: continue
        mean, lo, hi = boot(a, m)
        rec = {A[i]: [round(100 * mean[i], 1), round(100 * lo[i], 1), round(100 * hi[i], 1)] for i in range(len(A))}
        d1 = a[:, 1] - a[:, 0]; d2 = a[:, 4] - a[:, 3]; d3 = a[:, 2] - a[:, 0]; d4 = a[:, 5] - a[:, 3]
        for nmd, dv in (("L+Eb−L", d1), ("E+Lb−E", d2), ("Lb_only−L", d3), ("Eb_only−E", d4)):
            mm, ll, hh = boot(dv[:, None], m); rec[nmd] = [round(100 * mm[0], 1), round(100 * ll[0], 1), round(100 * hh[0], 1)]
        gap = a[m, 3].mean() - a[m, 0].mean()
        rec["recovery_frac"] = round(float((a[m, 1].mean() - a[m, 0].mean()) / gap), 3) if abs(gap) > 0.02 else None
        rec["n"] = int(m.sum()); out[f"{nm}|{g}"] = rec
# 닫힘 비율
R = 1 - DD[:, 1] / DD[:, 0]
out["closure_R_by_slot"] = {mo: np.round(np.nanmedian(R[ok & (mot == mo)], 0), 3).tolist() for mo in ("static", "moving_flat", "moving")}
out["closure_R_all"] = np.round(np.nanmedian(R[ok], 0), 3).tolist()
con = LOC[..., 2] - LOC[..., 3]                                              # (n, arm, 8) contrast = maxcos − median
ref_legacy = float(np.nanpercentile(con[ok, 3, :3], 10))                     # 1 판 문턱 (빈 장면 포함 풀) = 0.234
G_, CELL, FR = 16, 18.0, 288.0
def truth_xy(px, py):
    x = np.array([float(v) for v in px.split()]); y = np.array([float(v) for v in py.split()])
    return (np.array([np.nanmean(x[16 + 2 * t:18 + 2 * t]) for t in range(8)]), np.array([np.nanmean(y[16 + 2 * t:18 + 2 * t]) for t in range(8)]))
TX, TY = map(np.array, zip(*[truth_xy(p["px"], p["py"]) for p in P]))          # (n, 8) 미래 튜블릿 중심 px (샘플 16+2t, 17+2t)
cellsP = np.stack([np.clip(TX // CELL, 0, G_ - 1), np.clip(TY // CELL, 0, G_ - 1)], -1)   # 1 판: 화면 밖을 가장자리로 자름
has_obj = ok & ((viol != "vanish") | (pid == "A"))
perm = np.arange(n)
for mo in ("static", "moving_flat", "moving"):
    ix = np.where(mot == mo)[0]; perm[ix] = rng.permutation(ix)
am = LOC[..., [1, 0]]                                                     # (n, arm, 8, 2) x, y
hit_l = (np.abs(am - cellsP[:, None]).max(-1) <= 1).astype(float); nul_l = (np.abs(am - cellsP[perm][:, None]).max(-1) <= 1).astype(float)
objm_l = (con >= ref_legacy).astype(float)
def obj_signal(objm, hit, nul, windows, bootf):
    res = {}
    for mo in ("ALL", "static", "moving_flat", "moving"):
        m = has_obj & ((mot == mo) if mo != "ALL" else True)
        rec = {}
        for nm_, sl in windows:
            o = np.nanmean(objm[:, :, sl], -1); hn = np.nanmean(hit[:, :, sl] - nul[:, :, sl], -1)
            mw = m & np.isfinite(o).all(1)
            mo_, lo_, hi_ = bootf(o, mw); rec[f"obj|{nm_}"] = {A[i]: [round(float(mo_[i]), 3), round(float(lo_[i]), 3), round(float(hi_[i]), 3)] for i in range(len(A))}
            mh, lh, hh = bootf(hn, mw); rec[f"hitnull|{nm_}"] = {A[i]: [round(float(mh[i]), 3), round(float(lh[i]), 3), round(float(hh[i]), 3)] for i in range(len(A))}
            for nmd, (i1, i0) in (("L+Eb−L", (1, 0)), ("E+Lb−E", (4, 3)), ("Lb_only−L", (2, 0)), ("Eb_only−E", (5, 3))):
                d_ = (o[:, i1] - o[:, i0])[:, None]; mm_, ll_, hh_ = bootf(d_, mw)
                rec[f"obj|{nm_}|{nmd}"] = [round(float(mm_[0]), 3), round(float(ll_[0]), 3), round(float(hh_[0]), 3)]
            gap = np.mean(o[mw, 3]) - np.mean(o[mw, 0])
            rec[f"obj|{nm_}|recovery"] = round(float((np.mean(o[mw, 1]) - np.mean(o[mw, 0])) / gap), 3) if abs(gap) > 0.03 else None
            rec[f"obj|{nm_}|destroy"] = round(float((np.mean(o[mw, 3]) - np.mean(o[mw, 4])) / gap), 3) if abs(gap) > 0.03 else None
            rec[f"n|{nm_}"] = int(mw.sum())
        rec["n"] = int(m.sum()); res[mo] = rec
    return res
out["objectness_ref_legacy"] = ref_legacy
out["object_signal_legacy"] = obj_signal(objm_l, hit_l, nul_l, (("t0-1", slice(0, 2)), ("t2-4", slice(2, 5)), ("t5-7", slice(5, 8))), boot)

# ════════════════════════════ 2 판 (추가 · 고친 값) ════════════════════════════
rng2 = np.random.default_rng(1)
L_, LE, LB, E_, EL, EB = range(6)
STAT = ["recovery", "destroy", "L+Eb−L", "E+Lb−E", "Lb_only−L", "Eb_only−E", "(L+Eb)−E", "(E+Lb)−L", "E−L"]
def fr(o):                                                                   # o (..., 6 팔) → (..., 9)
    gap = o[..., E_] - o[..., L_]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.stack([(o[..., LE] - o[..., L_]) / gap, (o[..., E_] - o[..., EL]) / gap, o[..., LE] - o[..., L_], o[..., EL] - o[..., E_],
                         o[..., LB] - o[..., L_], o[..., EB] - o[..., E_], o[..., LE] - o[..., E_], o[..., EL] - o[..., L_], gap], -1)
def _wboot(vals, mask, clus, B=2000):
    """군집 bootstrap (가중치 = 군집 재표집 횟수). 유한한 행만. → 표본 평균 (열), bootstrap 평균 (B, 열), 행 수, 군집 수."""
    mask = mask & np.isfinite(vals).all(1)
    v = vals[mask]; u, inv = np.unique(clus[mask], return_inverse=True); G = len(u)
    sums = np.zeros((G, v.shape[1])); np.add.at(sums, inv, v); cnt = np.bincount(inv, minlength=G).astype(float)
    W = np.stack([np.bincount(rng2.integers(0, G, G), minlength=G) for _ in range(B)]).astype(float)
    return v.mean(0), (W @ sums) / (W @ cnt)[:, None], mask, G
def boot2(vals, mask):                                                       # 1 판 boot 와 같은 반환 (mean, lo, hi), 빠른 판
    pt, mb, _, _ = _wboot(vals, mask, cluster)
    return pt, np.percentile(mb, 2.5, 0), np.percentile(mb, 97.5, 0)
def cboot(vals, mask, clus, B=2000):
    """vals (n, 6 팔) → 점추정·2.5·97.5 % 의 (팔 6) 와 (통계 9), 행 수, 군집 수."""
    pt, mb, mask, G = _wboot(vals, mask, clus, B)
    arms = np.stack([pt, np.percentile(mb, 2.5, 0), np.percentile(mb, 97.5, 0)])
    sb = fr(mb); sp = fr(pt)
    stats = np.stack([sp, np.nanpercentile(sb, 2.5, 0), np.nanpercentile(sb, 97.5, 0)])
    return arms, stats, int(mask.sum()), G
def summ(vals, mask, clus=None, gap_min=0.03):
    arms, st, nn, G = cboot(vals, mask, cluster if clus is None else clus)
    rec = {"n": nn, "clusters": G, "arms": {A[i]: [round(float(x), 3) for x in arms[:, i]] for i in range(6)}}
    for j, s in enumerate(STAT):
        if j < 2 and not abs(st[0, 8]) > gap_min: rec[s] = None; continue
        rec[s] = [round(float(x), 3) for x in st[:, j]]
    return rec

# 진실 칸 (화면 밖 = NaN) · 궤적 · 다른 궤적 귀무
off = (TX < 0) | (TX >= FR) | (TY < 0) | (TY >= FR)                          # 움직이는 쌍 t7 전부
cx = np.where(off, np.nan, np.floor(TX / CELL)); cy = np.where(off, np.nan, np.floor(TY / CELL))
traj = np.array([p["motion"] + "|" + p["px"] + "|" + p["py"] for p in P])
ax, ay = LOC[..., 1], LOC[..., 0]
def hit_to(x, y):                                                             # x, y (n, 8) → (n, 6, 8), NaN 은 NaN
    h = (np.maximum(np.abs(ax - x[:, None]), np.abs(ay - y[:, None])) <= 1).astype(float)
    h[np.broadcast_to(np.isnan(x)[:, None], h.shape)] = np.nan
    return h
hit = hit_to(cx, cy)
null_ot = np.zeros_like(hit); n_traj = {}
for mo in ("static", "moving_flat", "moving"):
    ts = sorted(set(traj[mot == mo])); n_traj[mo] = len(ts); rep = {t: np.where(traj == t)[0][0] for t in ts}
    for t in ts:
        rows = np.where(traj == t)[0]; others = [o for o in ts if o != t]
        acc_ = sum(hit_to(np.broadcast_to(cx[rep[o]], (n, 8)), np.broadcast_to(cy[rep[o]], (n, 8)))[rows] for o in others)
        null_ot[rows] = acc_ / len(others)
null_ot[np.broadcast_to(np.isnan(cx)[:, None], hit.shape)] = np.nan
noobj = ok & (viol == "vanish") & (pid == "B")                                # pos = 빈 장면 (L·E 모두), 템플릿 = 배경
THR = {"hasobj_P10": float(np.percentile(con[has_obj, E_, :3], 10)), "rule1_0.357": 0.357, "legacy_allok_P10": ref_legacy,
       "hasobj_P25": float(np.percentile(con[has_obj, E_, :3], 25)), "noobj_P90": float(np.percentile(con[noobj, E_, :3], 90)),
       "noobj_P95": float(np.percentile(con[noobj, E_, :3], 95))}
PRIMARY = "hasobj_P10"
ref = THR[PRIMARY]
def objm_at(th):
    o = (con >= th).astype(float); o[np.broadcast_to(np.isnan(cx)[:, None], o.shape)] = np.nan; return o
conN = con.copy(); conN[np.broadcast_to(np.isnan(cx)[:, None], conN.shape)] = np.nan
WIN = {"t0-1": slice(0, 2), "t2-4": slice(2, 5), "t5-6": slice(5, 7)}
GRP = {"ALL": has_obj, "static": has_obj & (mot == "static"), "moving_flat": has_obj & (mot == "moving_flat"),
       "moving": has_obj & (mot == "moving"), "moving_any": has_obj & (mot != "static")}
READ = {f"obj@{k}": objm_at(v) for k, v in THR.items()}; READ["contrast"] = conN
READ["hit"] = hit; READ["null_ot"] = null_ot; READ["hitnull"] = hit - null_ot

# (a) 물체 신호 전 판독 × 운동 × 창
os2 = {}
for rd in (f"obj@{PRIMARY}", "obj@rule1_0.357", "contrast", "hitnull", "hit", "null_ot") + tuple(f"obj@{k}" for k in THR if k not in (PRIMARY, "rule1_0.357")):
    os2[rd] = {}
    for g, m in GRP.items():
        os2[rd][g] = {}
        for w, sl in WIN.items():
            with np.errstate(invalid="ignore"): x = np.nanmean(READ[rd][:, :, sl], -1)
            os2[rd][g][w] = summ(x, m, gap_min=(0.005 if rd == "contrast" else 0.03))
out["threshold_values"] = {k: round(v, 4) for k, v in THR.items()}
out["object_signal_v2"] = os2
# 머리 수치: 연결성분 군집 (late block · E_pos · E_imp 공유를 한 군집으로)
parent = list(range(n))
def find(a):
    while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
    return a
for key in (cluster, np.array([p["E_pos"] for p in P]), np.array([p["E_imp"] for p in P])):
    first = {}
    for i, kx in enumerate(key):
        if kx in first: parent[find(i)] = find(first[kx])
        else: first[kx] = i
comp = np.array([find(i) for i in range(n)])
out["headline_component_cluster"] = {f"{rd}|{g}|t2-4": summ(np.nanmean(READ[rd][:, :, WIN["t2-4"]], -1), GRP[g], clus=comp,
                                                            gap_min=(0.005 if rd == "contrast" else 0.03))
                                     for rd in (f"obj@{PRIMARY}", "contrast", "hitnull") for g in ("ALL", "moving_any", "static")}
out["headline_component_cluster"]["n_components"] = int(len(set(comp[has_obj])))

# (b) 기존 키 object_signal 을 고친 값으로 (문턱 has_obj P10 · 다른 궤적 귀무 · 화면 밖 NaN; 't5-7' 은 화면 밖 제외 = 움직임에서는 t5–6)
out["objectness_ref"] = ref
out["object_signal"] = obj_signal(READ[f"obj@{PRIMARY}"], hit, null_ot,
                                  (("t0-1", slice(0, 2)), ("t2-4", slice(2, 5)), ("t5-6", slice(5, 7)), ("t5-7", slice(5, 8))),
                                  boot2)
out["objectness_vanishA_pos"] = {mo: {A[i]: np.round(np.nanmean(READ[f"obj@{PRIMARY}"][ok & (mot == mo) & (viol == "vanish") & (pid == "A"), i], 0), 3).tolist()
                                      for i in range(6)} for mo in ("static", "moving_flat", "moving")}

# (c) 운동별 역사 대비 (같은 경계 · 다른 역사 = (L+Eb)−E, (E+Lb)−L;  역사 삭제 = Eb_only−E, Lb_only−L)
hc = {}
for rd in (f"obj@{PRIMARY}", "obj@rule1_0.357", "contrast", "hitnull"):
    hc[rd] = {g: {w: {s: os2[rd][g][w][s] for s in ("(L+Eb)−E", "(E+Lb)−L", "Eb_only−E", "Lb_only−L")} for w in WIN} for g in GRP}
out["history_contrasts"] = hc

# (d) k 별 · 같은 k / 다른 k 쌍둥이 (t2-4)
MD = {r["name"]: r for r in csv.DictReader(open(META_CSV))} if Path(META_CSV).exists() else None
out["by_k"] = {}
for mo in ("static", "moving_flat", "moving", "moving_any"):
    for k_ in (1, 2, 3, 4):
        m = GRP[mo] & (kk == k_)
        if m.sum() < 20: continue
        out["by_k"][f"{mo}|k{k_}"] = {rd: summ(np.nanmean(READ[rd][:, :, WIN["t2-4"]], -1), m, gap_min=(0.005 if rd == "contrast" else 0.03))
                                     for rd in (f"obj@{PRIMARY}", "obj@legacy_allok_P10", "contrast", "hitnull")}
if MD is not None:
    kE = np.array([int(MD[p["E_pos"]]["sym_k"]) for p in P])
    out["same_k_split"] = {("same_k" if sk else "diff_k"): {rd: summ(np.nanmean(READ[rd][:, :, WIN["t2-4"]], -1), has_obj & ((kE == kk) == sk))
                                                             for rd in (f"obj@{PRIMARY}", "contrast")} for sk in (True, False)}

# (e) 빈 장면 기저율 (vanish B: L·E pos 모두 빈 장면, 템플릿 = 진실 칸 배경 토큰) — 절대 수준은 물체 존재율이 아니다
nb = {}
for rd in [f"obj@{k}" for k in THR] + ["contrast"]:
    nb[rd] = {}
    for g in ("ALL", "static", "moving_any"):
        m = noobj & ((mot == g) if g == "static" else (mot != "static") if g == "moving_any" else True)
        nb[rd][g] = {w: [round(float(x), 3) for x in np.nanmean(np.nanmean(READ[rd][m][:, :, sl], -1), 0)] for w, sl in WIN.items()}
nb["n"] = int(noobj.sum())
out["no_object_baseline"] = nb

# (f) 채점 — block 일치 부분집합 (E_pos·E_imp 같은 block = matched pair 유효)
a_all = acc(slice(0, 8))
eblk_same = np.array([idx[p["E_pos"]]["block_id"] == idx[p["E_imp"]]["block_id"] for p in P])
postdiff = (np.array([(MD[p["L_imp"]]["shape_post"] != MD[p["E_imp"]]["shape_post"]) or (MD[p["L_imp"]]["color_post"] != MD[p["E_imp"]]["color_post"]) for p in P])
            if MD is not None else np.zeros(n, bool))
sc = {"n_E_diff_block": int((ok & ~eblk_same).sum()), "n_E_imp_post_differs_from_L_imp": int((ok & postdiff).sum())}
for sub, sm in (("all", ok), ("E_same_block", ok & eblk_same), ("E_diff_block", ok & ~eblk_same), ("E_same_block_same_post", ok & eblk_same & ~postdiff)):
    sc[sub] = {}
    for g, gm in (("ALL", True), ("static", mot == "static"), ("moving_flat", mot == "moving_flat"), ("moving", mot == "moving"),
                  ("vanish|A", (viol == "vanish") & (pid == "A")), ("vanish|B", (viol == "vanish") & (pid == "B")),
                  ("shape", viol == "shape"), ("color", viol == "color")):
        m = sm & gm
        if m.sum() < 20: continue
        sc[sub][g] = summ(100 * a_all, m, gap_min=2.0)
out["scoring_blockmatched"] = sc

# (g) 닫힘 — 평균 거리 D(팔, E)
out["closure_D_mean"] = {A[i]: round(float(np.nanmean(DD[ok, i])), 4) for i in range(6)}
out["closure_D_ratio_median"] = {"Eb_only/L": round(float(np.nanmedian(DD[ok, 5] / DD[ok, 0])), 3), "L+Eb/L": round(float(np.nanmedian(DD[ok, 1] / DD[ok, 0])), 3)}

# (h) 진행 방향 뒤처짐 (움직임): 부호 있는 칸 변위 (argmax − 진실) · 진행 방향; P[2 칸 이상 뒤처짐]
sgn = np.sign(np.nan_to_num(TX[:, 6] - TX[:, 0]))
dxs = (ax - cx[:, None]) * sgn[:, None, None]
lag = {}
for mo in ("moving_flat", "moving"):
    m = has_obj & (mot == mo)
    lag[mo] = {f"t{t}": {A[i]: {"median": float(np.nanmedian(dxs[m, i, t])), "P_lag_ge2": round(float(np.nanmean(dxs[m, i, t] <= -2)), 3)} for i in range(6)} for t in (2, 3, 4)}
out["lag"] = lag

# (i) 설계 사실 (쌍둥이 구성)
if MD is not None:
    late_tot = collections.Counter(); got = collections.Counter()
    blocks = collections.defaultdict(dict)
    for r in idx.values():
        blocks[(r["block_id"], r["pair_id"])]["pos" if r["plausible"] == "1" else "imp"] = r["video_id"]
    for (b, pd_), d in blocks.items():
        if len(d) == 2 and MD[d["pos"]]["occ_timing"] == "late":
            late_tot[(MD[d["pos"]]["motion"], MD[d["imp"]]["violation_type"], pd_)] += 1
    for p, v in zip(P, viol): got[(p["motion"], v, p["pair_id"])] += 1
    def hid(r):
        a_, b_ = r["hidden_start_measured"], r["hidden_end_measured"]
        f = np.arange(0, 96, 3)
        return np.zeros(32, bool) if a_ == "" or b_ == "" else (f >= float(a_)) & (f <= float(b_))
    HL = np.stack([hid(MD[p["L_pos"]]) for p in P]); HE = np.stack([hid(MD[p["E_pos"]]) for p in P])
    hidden = {}
    for mo in ("static", "moving_flat", "moving"):
        for k_ in (1, 2, 3, 4):
            m = has_obj & (mot == mo) & (kk == k_)
            if not m.any(): continue
            hidden[f"{mo}|k{k_}"] = {"n": int(m.sum()),
                                     "L_ctx_hidden_per_tubelet": HL[m][:, :16].reshape(-1, 8, 2).mean(-1).mean(0).round(2).tolist(),
                                     "L_fut_hidden_per_tubelet": HL[m][:, 16:].reshape(-1, 8, 2).mean(-1).mean(0).round(2).tolist(),
                                     "E_ctx_hidden_per_tubelet": HE[m][:, :16].reshape(-1, 8, 2).mean(-1).mean(0).round(2).tolist(),
                                     "E_fut_hidden_per_tubelet": HE[m][:, 16:].reshape(-1, 8, 2).mean(-1).mean(0).round(2).tolist()}
    reuse = collections.Counter(p["E_pos"] for p in P)
    out["design"] = {
        "n_pairs": n, "n_has_obj": int(has_obj.sum()), "n_noobj": int(noobj.sum()),
        "n_by_motion_has_obj": {mo: int((has_obj & (mot == mo)).sum()) for mo in ("static", "moving_flat", "moving")},
        "n_trajectories_per_motion": n_traj,
        "sym_k_mismatch": int((kE != kk).sum()),
        "occ_height_differs": int(sum(MD[p["L_pos"]]["occ_height_cm"] != MD[p["E_pos"]]["occ_height_cm"] for p in P)),
        "occ_height_differs_by_motion": dict(collections.Counter(p["motion"] for p in P if MD[p["L_pos"]]["occ_height_cm"] != MD[p["E_pos"]]["occ_height_cm"])),
        "px_traj_max_abs_diff_Lpos_Epos": float(max(np.nanmax(np.abs(np.array(MD[p["L_pos"]][c].split(), float) - np.array(MD[p["E_pos"]][c].split(), float)))
                                                   for p in P for c in ("object_px_x_by_sample", "object_px_y_by_sample"))),
        "E_pos_unique": len(reuse), "E_pos_max_reuse": max(reuse.values()),
        "late_pairs_total": int(sum(late_tot.values())), "late_pairs_matched": n,
        "matched_over_total_by_motion_viol_pid": {"|".join(k): [got[k], late_tot[k]] for k in sorted(late_tot)},
        "E_pos_E_imp_diff_block": int((~eblk_same).sum()),
        "E_diff_block_and_diff_symk": int(sum((idx[p["E_pos"]]["block_id"] != idx[p["E_imp"]]["block_id"]) and (MD[p["E_pos"]]["sym_k"] != MD[p["E_imp"]]["sym_k"]) for p in P)),
        "hidden_measured": hidden}

json.dump(out, open(OUT / f"m2_v11{TAG}.json", "w"), indent=1, default=float)

# ════════════════════════════ 출력 ════════════════════════════
def fmt(r, keys=("recovery", "destroy")):
    return " ".join(f"{k} " + ("—" if r.get(k) is None else f"{r[k][0]:+.3f}[{r[k][1]:+.3f},{r[k][2]:+.3f}]") for k in keys)
print("\n== 1 판 재현 (object_signal_legacy, 문턱", round(ref_legacy, 3), ")")
for mo in ("ALL",):
    rec = out["object_signal_legacy"][mo]
    for w in ("t0-1", "t2-4", "t5-7"):
        print(f"  {mo} {w} obj " + " ".join(f"{rec[f'obj|{w}'][a_][0]:.3f}" for a_ in A) + " | hit−null(순열) " + " ".join(f"{rec[f'hitnull|{w}'][a_][0]:+.3f}" for a_ in A)
              + f" | 회복 {rec[f'obj|{w}|recovery']} 파괴 {rec[f'obj|{w}|destroy']}")
print("\n문턱:", out["threshold_values"], " 궤적 수:", n_traj)
for rd in (f"obj@{PRIMARY}", "obj@rule1_0.357", "contrast", "hitnull", "hit", "null_ot"):
    print(f"\n== {rd}  팔: " + " ".join(A))
    for g in GRP:
        for w in WIN:
            r = os2[rd][g][w]
            print(f"  {g:11s} {w} n{r['n']:5d} " + " ".join(f"{r['arms'][a_][0]:.3f}" for a_ in A) + " | " + fmt(r, ("recovery", "destroy")) + " | "
                  + fmt(r, ("(L+Eb)−E", "Eb_only−E", "Lb_only−L", "(E+Lb)−L", "E+Lb−E")))
print("\n== 문턱 스윕 (t2-4)")
for k in THR:
    for g in ("ALL", "moving_any", "static", "moving_flat", "moving"):
        r = os2[f"obj@{k}"][g]["t2-4"]; print(f"  {k:17s} {THR[k]:.3f} {g:11s} " + " ".join(f"{r['arms'][a_][0]:.3f}" for a_ in A) + " | " + fmt(r))
print("\n== 연결성분 군집 (n comp", out["headline_component_cluster"]["n_components"], ")")
for k_, r in out["headline_component_cluster"].items():
    if k_ != "n_components": print(f"  {k_:28s} G{r['clusters']} " + fmt(r))
print("\n== k 별 (t2-4)")
for k_, r in out["by_k"].items():
    print(f"  {k_:15s} n{r['contrast']['n']:4d} obj " + " ".join(f"{r[f'obj@{PRIMARY}']['arms'][a_][0]:.2f}" for a_ in A) + " | " + fmt(r[f"obj@{PRIMARY}"])
          + " | legacy " + fmt(r["obj@legacy_allok_P10"]) + " | contrast " + fmt(r["contrast"]) + " | hitnull " + fmt(r["hitnull"]))
if "same_k_split" in out:
    for k_, r in out["same_k_split"].items(): print(f"  {k_}: n{r['contrast']['n']} obj " + fmt(r[f"obj@{PRIMARY}"]) + " | contrast " + fmt(r["contrast"]))
print("\n== 빈 장면 기저율 (n", nb["n"], ")")
for rd in nb:
    if rd != "n": print(f"  {rd:22s} " + " | ".join(f"{g} {w} " + " ".join(f"{x:.2f}" for x in nb[rd][g][w]) for g in ("ALL", "moving_any", "static") for w in WIN))
print("\n== 채점 (전 슬롯)", {k: v for k, v in sc.items() if k.startswith("n_")})
for sub in ("all", "E_same_block", "E_diff_block", "E_same_block_same_post"):
    for g, r in sc[sub].items():
        print(f"  {sub:22s} {g:11s} n{r['n']:5d} " + " ".join(f"{r['arms'][a_][0]:5.1f}" for a_ in A) + f" | 회복 {r['recovery']}")
print("\n닫힘 D 평균", out["closure_D_mean"], out["closure_D_ratio_median"])
print("뒤처짐 P[lag>=2]", {mo: {t: {a_: v[a_]["P_lag_ge2"] for a_ in ("L", "E", "L+Eb", "Eb_only")} for t, v in lag[mo].items()} for mo in lag})
if "design" in out:
    d = out["design"]; print("\n설계:", {k: v for k, v in d.items() if k not in ("hidden_measured", "matched_over_total_by_motion_viol_pid")})
    print("  선택률", d["matched_over_total_by_motion_viol_pid"])
    for k_, v in d["hidden_measured"].items(): print("  hidden", k_, v)
print("\n저장:", OUT / f"m2_v11{TAG}.json")
