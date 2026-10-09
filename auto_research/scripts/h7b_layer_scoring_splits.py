#!/usr/bin/env python3
"""H7b — v11_full 층별 렌즈 채점의 적대 검증 보강 (CPU 전용, vll6 에서 srun6.sh 로 실행).

무엇
  (1) 12 층 전부 × 슬롯 묶음 {all, t0-3, t4-7} × timing × motion × violation × A/B 방향의 정확도와 n.
      A/B 방향 (CLAUDE.md §1-7): matched pair 의 pair_id. vanish 에서 A = 물체→빈 (사라짐), B = 빈→물체 (나타남) —
      불가능 clip 의 shape_pre / shape_post 로 직접 확인해 json 에 적는다.
  (2) 층 차이의 block 군집 paired bootstrap 95% CI (B = 2000, block 단위 복원 추출; block 안 A·B 두 쌍이 함께 뽑힌다).
      대비: L1 − L11, (L11 을 뺀 최고 층) − L11 (그 최고 층은 같은 자료에서 고른 것이라 descriptive), L10 − L11.
  (3) 입력 없는 기준선. 저장된 것이 전 토큰 평균 풀링 특징뿐이라 토큰 수준 상수 기준선은 못 만든다 → 풀링 공간에서 만든다:
        S^pool_l   = mean_{t∈T} mean_d |p̄^(l)_t − h̄_t|            (렌즈의 풀링판; p̄ = 토큰 평균 렌즈 출력)
        S^mean-p_l = mean_{t∈T} mean_d |c_{l,t} − h̄_t|,  c_{l,t} = 전 43,008 clip 평균 p̄^(l)_t   (clip 과 무관한 상수)
        S^zero     = mean_{t∈T} mean_d |h̄_t|,   S^mean-h = mean_{t∈T} mean_d |E[h̄_t] − h̄_t|
      matched pair 안에서 p 는 비트 단위로 같으므로 상수 c 를 쓴 결정 = "h_pos 와 h_imp 중 어느 쪽이 c 에 가까운가" 라는
      p 와 무관한 prior 다. 토큰 수준 렌즈 결정과 mean-p 결정의 부호 일치율도 층별로 낸다.
  (4) 예측 없는 복사 기준선 (H6f 산출물: 토큰 수준 |LN(z)_문맥끝 − LN(h)_j|, block_id % 4 == 0 부분표본 5,376 쌍) 과
      같은 부분표본의 렌즈 L1 / L11 토큰 수준 정확도.
  (5) block 안 A·B 결정 분해 (토큰 수준, 파라미터 0): 쌍 A 와 B 는 같은 두 미래 (X, Y) 를 문맥만 바꿔 비교한다.
      문맥과 무관한 기준 (상수 c) 은 h 미래가 문맥에 무관하면 두 쌍에서 같은 미래를 고르므로 정확히 한 쪽만 맞힌다.
        X1 = P(한 쪽만 정답), BC = P(둘 다 정답), BW = P(둘 다 오답);  acc = 1/2 + (BC − BW)/2 (항등식)
      X1 ≈ 100 이면 그 층의 결정은 문맥으로 뒤집히지 않는다 (입력 없는 prior 와 구별 불가).
  (6) target encoder 의 문맥 번짐 (풀링 공간): 미래 픽셀이 같고 문맥만 다른 두 clip (pos_a·imp_ba) 의 h̄ 미래 거리 ÷
      문맥이 같고 미래가 다른 두 clip (pos_a·imp_ab) 의 거리. 0 이 아니면 (5) 의 "정확히 한 쪽" 이 근사가 된다.
  (7) held-out 층 선택: block 반반 (seed 10 개) — train 반에서 셀마다 최고 층을 고르고 test 반에서 (그 층 − L11) 정확도 차.
      (2) 의 "최고 다른 층" 은 같은 자료에서 고른 값이라 선택 편향이 있다 → 이것이 편향 없는 추정.
왜
  적대 검증 (2026-09-24): "층을 따라 단조 증가 · L11 최고" 가 static early/mid 에서 깨진다 (L1 > L11), A/B 방향 병기 누락,
  초기층 렌즈 점수가 chance 아래로 내려가는 것은 p 와 무관한 prior 일 수 있다.
입력  /data2/local_datasets/world/world_analysis/cache/auto_research/v11_h7/{meta.json, l1.npy, lens.npy, h.npy}
      /data2/local_datasets/world/world_analysis/cache/auto_research/v11_h6f/{meta.json, l1.npy}
출력  auto_research/exp_results/h7/h7b_scoring_splits.json · h7b_scoring_tables.md
"""
import collections, itertools, json, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
C = Path("/data2/local_datasets/world/world_analysis/cache/auto_research")
I, I6 = C / "v11_h7", C / "v11_h6f"
OUT = ROOT / "auto_research/exp_results/h7"; OUT.mkdir(parents=True, exist_ok=True)
NB = 2000
t0 = time.time()
meta = json.load(open(I / "meta.json")); n = meta["n"]
M = {k: np.array(v) for k, v in meta.items() if isinstance(v, list) and len(v) == n}
tim = np.where(M["occ_timing"] == "", "visible", M["occ_timing"])
l1 = np.load(I / "l1.npy")                                                     # (n, 12, 8) 토큰 수준 렌즈 L1

# ---------------- matched pair + 방향
pairs = collections.defaultdict(dict)
for i in range(n):
    pairs[(M["block_id"][i], M["pair_id"][i])]["pos" if M["plausible"][i] == "1" else "imp"] = i
keys = [k for k, d in pairs.items() if len(d) == 2]
P = np.array([(pairs[k]["pos"], pairs[k]["imp"]) for k in keys]); ps, im = P[:, 0], P[:, 1]
pid = np.array([k[1] for k in keys]); blk = np.array([int(k[0]) for k in keys])
PT = {"timing": tim[ps], "motion": M["motion"][ps], "violation": M["violation_type"][ps], "dir": pid}
# 방향 의미 확인 (vanish)
dir_sem = collections.Counter()
for j in np.where(PT["violation"] == "vanish")[0]:
    dir_sem[(pid[j], "none→obj" if M["shape_pre"][im[j]] == "none" else ("obj→none" if M["shape_post"][im[j]] == "none" else "?"))] += 1
dir_sem = {f"{a}:{b}": c for (a, b), c in dir_sem.items()}
print("vanish 방향 의미:", dir_sem, flush=True)

SLOTS = {"all": list(range(8)), "t0-3": [0, 1, 2, 3], "t4-7": [4, 5, 6, 7]}
VALS = {"timing": ["*", "visible", "early", "mid", "late"], "motion": ["*", "static", "moving_flat", "moving"],
        "violation": ["*", "vanish", "shape", "color"], "dir": ["*", "A", "B"]}


def mask_of(tm, mo, vi, di):
    m = np.ones(len(P), bool)
    for key, v in (("timing", tm), ("motion", mo), ("violation", vi), ("dir", di)):
        if v != "*":
            m &= PT[key] == v
    return m


def decide(S):                                                                # S (n, L) → 정답 (pair, L), 동점 0.5
    return (S[im] > S[ps]).astype(np.float64) + 0.5 * (S[im] == S[ps])


# ---------------- (1) 전 층 × 분할
OK = {sn: decide(l1[:, :, sl].mean(2)) for sn, sl in SLOTS.items()}           # (pair, 12)
A = {}
for sn in SLOTS:
    for tm, mo, vi, di in itertools.product(*VALS.values()):
        m = mask_of(tm, mo, vi, di)
        if m.sum() == 0:
            continue
        A[f"{sn}|{tm}|{mo}|{vi}|{di}"] = {"acc": (100 * OK[sn][m].mean(0)).round(2).tolist(), "n": int(m.sum())}
print(f"(1) 분할 {len(A)} 칸  {time.time()-t0:.0f}s", flush=True)

# ---------------- (2) block 군집 paired bootstrap
ub, binv = np.unique(blk, return_inverse=True)
rng = np.random.default_rng(0)
W = np.stack([np.bincount(rng.integers(0, len(ub), len(ub)), minlength=len(ub)) for _ in range(NB)]).astype(np.float64)  # (NB, nblock)


def boot(vals, m):
    """vals: (pair,) 값, m: pair mask → 평균과 block 군집 bootstrap 95% CI."""
    s = np.bincount(binv[m], weights=vals[m], minlength=len(ub)); c = np.bincount(binv[m], minlength=len(ub)).astype(np.float64)
    bs = (W @ s) / np.maximum(W @ c, 1)
    return [round(100 * float(vals[m].mean()), 2), round(100 * float(np.percentile(bs, 2.5)), 2), round(100 * float(np.percentile(bs, 97.5)), 2)]


BOOT = {}
for sn in ("all", "t0-3", "t4-7"):
    ok = OK[sn]
    for tm, mo, vi in itertools.product(VALS["timing"], VALS["motion"], VALS["violation"]):
        m = mask_of(tm, mo, vi, "*")
        if m.sum() == 0:
            continue
        acc = ok[m].mean(0); oth = [l for l in range(11)]; lb = int(oth[np.argmax(acc[oth])])
        BOOT[f"{sn}|{tm}|{mo}|{vi}"] = {
            "n": int(m.sum()), "best_other": lb,
            "L11": boot(ok[:, 11], m),
            "L1-L11": boot(ok[:, 1] - ok[:, 11], m),
            "Lbest-L11": boot(ok[:, lb] - ok[:, 11], m),
            "L10-L11": boot(ok[:, 10] - ok[:, 11], m),
            "L6-L11": boot(ok[:, 6] - ok[:, 11], m),
        }
print(f"(2) bootstrap {len(BOOT)} 칸  {time.time()-t0:.0f}s", flush=True)

# ---------------- (3) 입력 없는 기준선 (풀링 공간)
hb = np.load(I / "h.npy", mmap_mode="r")
lens = np.load(I / "lens.npy", mmap_mode="r")                                  # (n, 12, 8, 1280) fp16
H = np.asarray(hb[:, 8:], np.float32)                                          # (n, 8, 1280) 미래 h̄
pool = np.zeros((n, 12, 8), np.float32); csum = np.zeros((12, 8, 1280), np.float64)
CH = 2048
for i0 in range(0, n, CH):
    X = np.asarray(lens[i0:i0 + CH], np.float32)                                # (ch, 12, 8, 1280)
    pool[i0:i0 + CH] = np.abs(X - H[i0:i0 + CH, None]).mean(-1)
    csum += X.sum(0)
    if (i0 // CH) % 5 == 0:
        print(f"   lens {i0}/{n}  {time.time()-t0:.0f}s", flush=True)
cmean = (csum / n).astype(np.float32)                                          # c_{l,t}
meanp = np.stack([np.abs(H - cmean[l][None]).mean(-1) for l in range(12)], 1)  # (n, 12, 8)
zero = np.abs(H).mean(-1)                                                      # (n, 8)
meanh = np.abs(H - H.mean(0, keepdims=True)).mean(-1)                          # (n, 8)
del H
BASE = {}
for sn, sl in SLOTS.items():
    okp = decide(pool[:, :, sl].mean(2)); okc = decide(meanp[:, :, sl].mean(2))
    okz = decide(zero[:, sl].mean(1, keepdims=True))[:, 0]; okh = decide(meanh[:, sl].mean(1, keepdims=True))[:, 0]
    dtok = l1[:, :, sl].mean(2); dtok = np.sign(dtok[im] - dtok[ps])
    dc = meanp[:, :, sl].mean(2); dc = np.sign(dc[im] - dc[ps])
    dpl = pool[:, :, sl].mean(2); dpl = np.sign(dpl[im] - dpl[ps])
    for tm, mo, vi, di in itertools.product(*VALS.values()):
        m = mask_of(tm, mo, vi, di)
        if m.sum() == 0:
            continue
        BASE[f"{sn}|{tm}|{mo}|{vi}|{di}"] = {
            "n": int(m.sum()),
            "tok_lens": (100 * OK[sn][m].mean(0)).round(2).tolist(),
            "pool_lens": (100 * okp[m].mean(0)).round(2).tolist(),
            "mean_p": (100 * okc[m].mean(0)).round(2).tolist(),
            "zero": round(100 * float(okz[m].mean()), 2), "mean_h": round(100 * float(okh[m].mean()), 2),
            "agree_tok_vs_meanp": (100 * (dtok[m] == dc[m]).mean(0)).round(2).tolist(),
            "agree_tok_vs_pool": (100 * (dtok[m] == dpl[m]).mean(0)).round(2).tolist(),
        }
print(f"(3) 기준선  {time.time()-t0:.0f}s", flush=True)

# ---------------- (4) 복사 기준선 (H6f 부분표본, 토큰 수준)
m6 = json.load(open(I6 / "meta.json")); c6 = np.load(I6 / "l1.npy")           # (n6, 8, 4)
pos6 = {v: k for k, v in enumerate(m6["video_id"])}
has = np.array([(M["video_id"][a] in pos6) and (M["video_id"][b] in pos6) for a, b in P])
r6p = np.array([pos6.get(M["video_id"][a], -1) for a in ps]); r6i = np.array([pos6.get(M["video_id"][b], -1) for b in im])
COPY = {}
for sn, sl in SLOTS.items():
    cz = c6[:, sl, 1].mean(1); pf = c6[:, sl, 0].mean(1)
    okcz = np.zeros(len(P)); okpf = np.zeros(len(P))
    okcz[has] = (cz[r6i[has]] > cz[r6p[has]]) + 0.5 * (cz[r6i[has]] == cz[r6p[has]])
    okpf[has] = (pf[r6i[has]] > pf[r6p[has]]) + 0.5 * (pf[r6i[has]] == pf[r6p[has]])
    for tm, mo, vi, di in itertools.product(*VALS.values()):
        m = mask_of(tm, mo, vi, di) & has
        if m.sum() == 0:
            continue
        COPY[f"{sn}|{tm}|{mo}|{vi}|{di}"] = {"n": int(m.sum()), "copyz": round(100 * float(okcz[m].mean()), 2),
                                            "p_full_h6f": round(100 * float(okpf[m].mean()), 2),
                                            "lens_tok_same_subset": (100 * OK[sn][m].mean(0)).round(2).tolist()}
print(f"(4) 복사 기준선 부분표본 {int(has.sum())} 쌍  {time.time()-t0:.0f}s", flush=True)

# ---------------- (5) block 안 A·B 결정 분해 (토큰 수준, 파라미터 0) — 문맥과 무관한 기준의 흔적
# 2×2 설계: 쌍 A 의 두 미래 = (X 미래, Y 미래), 쌍 B 의 두 미래 = (Y 미래, X 미래), 문맥만 다르다.
# 문맥과 무관한 기준 c (상수, 입력 없는 prior) 는 h 미래가 문맥에 무관하다면 두 쌍에서 같은 미래를 "더 멀다" 고 고르므로
# **정확히 하나만 맞힌다** → X1 = P(한쪽만 정답) = 1. 문맥을 반영하는 예측은 BC (둘 다 정답) 를 올린다.
# 항등식: acc = 1/2 + (BC − BW)/2. X1 은 "결정이 문맥으로 뒤집히지 않은 block 비율".
bk = collections.defaultdict(dict)
for j, (b, p_) in enumerate(zip(blk, pid)):
    bk[b][p_] = j
BA = np.array([(d["A"], d["B"]) for d in bk.values() if len(d) == 2]); ja, jb = BA[:, 0], BA[:, 1]
DEC = {}
for sn in ("all", "t0-3", "t4-7"):
    oa, ob = OK[sn][ja], OK[sn][jb]                                             # (block, 12)
    bc, bw = oa * ob, (1 - oa) * (1 - ob)
    for tm, mo, vi in itertools.product(VALS["timing"], VALS["motion"], VALS["violation"]):
        m = mask_of(tm, mo, vi, "*")[ja]
        if m.sum() == 0:
            continue
        DEC[f"{sn}|{tm}|{mo}|{vi}"] = {"n_block": int(m.sum()), "BC": (100 * bc[m].mean(0)).round(1).tolist(),
                                       "BW": (100 * bw[m].mean(0)).round(1).tolist(), "X1": (100 * (1 - bc[m] - bw[m]).mean(0)).round(1).tolist()}
print(f"(5) A·B 분해  {time.time()-t0:.0f}s", flush=True)

# ---------------- (6) target encoder 의 문맥 번짐 (풀링 공간): 미래 픽셀이 같은 두 clip 의 h̄ 미래 거리
vv = collections.defaultdict(dict)
for i in range(n):
    vv[M["block_id"][i]][M["variant"][i]] = i
VB = [(b, d) for b, d in vv.items() if len(d) == 4]
ia = np.array([d["pos_a"] for _, d in VB]); ib = np.array([d["pos_b"] for _, d in VB])
iab = np.array([d["imp_ab"] for _, d in VB]); iba = np.array([d["imp_ba"] for _, d in VB])
hf = lambda idx: np.asarray(hb[np.sort(idx)], np.float32)[np.argsort(np.argsort(idx)), 8:]   # (m, 8, 1280)
Ha, Hb, Hab, Hba = hf(ia), hf(ib), hf(iab), hf(iba)
same_fut = 0.5 * (np.abs(Ha - Hba).mean(-1) + np.abs(Hb - Hab).mean(-1))      # 미래 픽셀 같고 문맥 다름
same_ctx = 0.5 * (np.abs(Ha - Hab).mean(-1) + np.abs(Hb - Hba).mean(-1))      # 문맥 같고 미래 다름 (= 채점 신호의 원천)
del Ha, Hb, Hab, Hba
vtim = np.where(M["occ_timing"][ia] == "", "visible", M["occ_timing"][ia]); vvio = M["violation_type"][ia]
SMEAR = {}
for tm in VALS["timing"]:
    for vi in VALS["violation"]:
        m = np.ones(len(ia), bool)
        if tm != "*": m &= vtim == tm
        if vi != "*": m &= vvio == vi
        SMEAR[f"{tm}|{vi}"] = {"n_block": int(m.sum()), "same_future_diff_ctx": same_fut[m].mean(0).round(4).tolist(),
                               "same_ctx_diff_future": same_ctx[m].mean(0).round(4).tolist(),
                               "ratio": (same_fut[m].mean(0) / same_ctx[m].mean(0)).round(3).tolist()}
print(f"(6) 번짐  {time.time()-t0:.0f}s", flush=True)

# ---------------- (7) held-out 층 선택 — "최고 다른 층" 의 선택 편향 제거
# block 을 반반 (seed 0..9) 나눠 train 반에서 셀마다 최고 층 (L0..L11 중) 을 고르고, test 반에서 (고른 층 − L11) 을 잰다.
HO = {}
for tm, mo, vi in itertools.product(VALS["timing"], VALS["motion"], VALS["violation"]):
    m = mask_of(tm, mo, vi, "*")
    if m.sum() == 0:
        continue
    diffs, picks = [], []
    for s in range(10):
        trb = np.random.default_rng(100 + s).permutation(len(ub))[: len(ub) // 2]; intr = np.zeros(len(ub), bool); intr[trb] = True
        tr, te = m & intr[binv], m & ~intr[binv]
        lb = int(np.argmax(OK["all"][tr].mean(0))); picks.append(lb)
        diffs.append(float(OK["all"][te, lb].mean() - OK["all"][te, 11].mean()))
    HO[f"all|{tm}|{mo}|{vi}"] = {"picked_layers": picks, "test_diff_mean": round(100 * float(np.mean(diffs)), 2),
                                 "test_diff_min": round(100 * float(np.min(diffs)), 2), "test_diff_max": round(100 * float(np.max(diffs)), 2)}
print(f"(7) held-out 층 선택  {time.time()-t0:.0f}s", flush=True)

out = {"n_pairs": int(len(P)), "dir_semantics": dir_sem, "heldout_layer_choice": HO, "bootstrap_B": NB, "bootstrap_unit": "block_id (A·B 두 쌍 함께)",
       "A_all_layers": A, "boot": BOOT, "baselines_pooled": BASE, "copy_baseline_h6f": COPY, "AB_decomposition": DEC, "h_context_smear_pooled": SMEAR,
       "note": "acc 는 12 층 L0..L11 리스트. baseline mean_p 는 층별 상수 c_{l,t} (전 clip 평균 풀링 렌즈). 풀링 공간 기준선이라 토큰 수준 렌즈와 공간이 다르다."}
json.dump(out, open(OUT / "h7b_scoring_splits.json", "w"))

# ---------------- 사람이 읽는 표
L = []
def row(lbl, v, nn):
    L.append(f"| {lbl} | {nn} | " + " | ".join(f"{x:.1f}" for x in v) + " |")
for sn in ("all", "t0-3", "t4-7"):
    L += [f"\n## 슬롯 {sn} — 12 층 토큰 수준 렌즈 채점 정확도 (%)\n", "| timing · motion · violation · dir | n | " + " | ".join(f"L{l}" for l in range(12)) + " |",
          "|---|---:|" + "---:|" * 12]
    for tm in VALS["timing"]:
        for mo in VALS["motion"]:
            for vi in VALS["violation"]:
                for di in VALS["dir"]:
                    k = f"{sn}|{tm}|{mo}|{vi}|{di}"
                    if k in A:
                        row(f"{tm} · {mo} · {vi} · {di}", A[k]["acc"], A[k]["n"])
L += ["\n## block 군집 paired bootstrap (슬롯 all), 차이 %p [95% CI]\n", "| timing · motion · violation | n | L11 | L1−L11 | 최고 다른 층 − L11 | L10−L11 |", "|---|---:|---|---|---|---|"]
for k, v in BOOT.items():
    if k.startswith("all|"):
        f = lambda x: f"{x[0]:+.1f} [{x[1]:+.1f}, {x[2]:+.1f}]"
        L.append(f"| {k[4:]} | {v['n']} | {v['L11'][0]:.1f} [{v['L11'][1]:.1f}, {v['L11'][2]:.1f}] | {f(v['L1-L11'])} | L{v['best_other']} {f(v['Lbest-L11'])} | {f(v['L10-L11'])} |")
L += ["\n## 입력 없는 기준선 (슬롯 all, 풀링 공간) — L0 / L1 / L6 / L11\n",
      "| timing · motion · violation · dir | n | 토큰 렌즈 | 풀링 렌즈 | mean-p | 토큰↔mean-p 일치 | zero | mean-h |", "|---|---:|---|---|---|---|---:|---:|"]
for tm in VALS["timing"]:
    for mo in VALS["motion"]:
        for vi in VALS["violation"]:
            for di in VALS["dir"]:
                k = f"all|{tm}|{mo}|{vi}|{di}"
                if k in BASE and (mo == "*" or vi == "*" or tm in ("early", "mid")):
                    b = BASE[k]; g = lambda x: "/".join(f"{x[l]:.0f}" for l in (0, 1, 6, 11))
                    L.append(f"| {tm} · {mo} · {vi} · {di} | {b['n']} | {g(b['tok_lens'])} | {g(b['pool_lens'])} | {g(b['mean_p'])} | {g(b['agree_tok_vs_meanp'])} | {b['zero']:.1f} | {b['mean_h']:.1f} |")
L += ["\n## 예측 없는 복사 기준선 (H6f 부분표본, 토큰 수준, 슬롯 all)\n", "| timing · motion · violation · dir | n | copy-z | 렌즈 L1 | 렌즈 L11 |", "|---|---:|---:|---:|---:|"]
for k, v in COPY.items():
    if k.startswith("all|") and (k.split("|")[4] == "*"):
        L.append(f"| {' · '.join(k.split('|')[1:])} | {v['n']} | {v['copyz']:.1f} | {v['lens_tok_same_subset'][1]:.1f} | {v['lens_tok_same_subset'][11]:.1f} |")
L += ["\n## block 안 A·B 결정 분해 (슬롯 all, 토큰 수준) — X1 = 한쪽만 정답 (문맥과 무관한 기준이면 100), BC = 둘 다 정답\n",
      "| timing · motion · violation | n_block | X1 L0/L1/L3/L6/L9/L11 | BC L0/L1/L3/L6/L9/L11 | BW L0/L1/L3/L6/L9/L11 |", "|---|---:|---|---|---|"]
for k, v in DEC.items():
    if k.startswith("all|"):
        g = lambda x: "/".join(f"{x[l]:.0f}" for l in (0, 1, 3, 6, 9, 11))
        L.append(f"| {' · '.join(k.split('|')[1:])} | {v['n_block']} | {g(v['X1'])} | {g(v['BC'])} | {g(v['BW'])} |")
L += ["\n## target encoder 문맥 번짐 (풀링 h̄ 미래, mean_d L1) — 슬롯 0 / 3 / 7\n",
      "| timing · violation | n_block | 미래 같고 문맥 다름 | 문맥 같고 미래 다름 | 비율 |", "|---|---:|---|---|---|"]
for k, v in SMEAR.items():
    g = lambda x: "/".join(f"{x[t]:.3f}" for t in (0, 3, 7))
    L.append(f"| {k.replace('|', ' · ')} | {v['n_block']} | {g(v['same_future_diff_ctx'])} | {g(v['same_ctx_diff_future'])} | {g(v['ratio'])} |")
L += ["\n## held-out 층 선택 (슬롯 all) — train 반에서 고른 층 − L11, test 반 정확도 차 %p (seed 10 개 평균 [최소, 최대])\n",
      "| timing · motion · violation | 고른 층 (seed 별) | test 차 |", "|---|---|---|"]
for k, v in HO.items():
    L.append(f"| {' · '.join(k.split('|')[1:])} | {','.join(map(str, v['picked_layers']))} | {v['test_diff_mean']:+.1f} [{v['test_diff_min']:+.1f}, {v['test_diff_max']:+.1f}] |")
(OUT / "h7b_scoring_tables.md").write_text("# H7b 표 (자동 생성, h7b_layer_scoring_splits.py)\n" + "\n".join(L) + "\n")
print(f"끝  {time.time()-t0:.0f}s", flush=True)
