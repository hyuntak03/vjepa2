#!/usr/bin/env python3
"""negimp (반사실 손실) 대 L1 대조 — 장면 짝 부트스트랩 두 가지 (2026-11-05, tak's_story R2/R4 요청). GPU 불필요.
 (a) real 보드 치우침 c 의 변화 Δc = c(CF) − c(대조): 장면을 같은 색인으로 다시 뽑는 짝 부트스트랩 (10,000 회), 두쪽 p = 2·min(P(Δ≤0), P(Δ≥0)).
     + 사라짐 (A) 짝의 고침 / 깨짐 (정확 이항).
 (b) 연속 문맥 margin  g = (m_A + m_B) / (장면 네 영상 surprise 평균)  — 대조 vs CF 짝 부트스트랩 평균 차.
     (분모 = 그 모델 · 그 장면의 평균 surprise: negimp 가 surprise 크기 자체를 키운 몫을 덜어 낸다.) 원값 (정규화 없음) 도 같이.
입력 suffix 를 주면 (예: _futonly) 그 채점 결과를 읽는다. per_video_surprise 가 dict 면 key (기본 all).
  /venv/vjepa/bin/python z_research/scripts/analysis/cf_extras.py [--suffix _futonly] [--key all]
"""
import argparse, json, os, random, sys
sys.path.insert(0, os.path.dirname(__file__))
from scene_paired_tests import boards, scene_stats, binom_p, RES, Z

PAIRS = [("released: L1 2k → CF", "vith_release_v4vc", "vith_release_v4vc_negimp"),
         ("rollout (AR): L1 2k → CF", "vast_ar_vw_state_v4vc", "vast_ar_vw_state_v4vc_negimp")]
SETS = ["IntPhys1 Surrogate", "VOBench v3 late · panel", "VOBench v3 late · moving occluder", "real world (all 95)"]


def load(pat, m, suffix, key):
    f = f"{RES}/{pat.format(m=m)}{suffix}/per_block.json"
    if not os.path.exists(f):
        return None
    S = json.load(open(f))["per_video_surprise"]
    return {k: (v[key] if isinstance(v, dict) else v) for k, v in S.items()}


def crit(rows, ix):
    n = len(ix); A = sum(1.0 if rows[i][1] > 0 else 0.5 if rows[i][1] == 0 else 0.0 for i in ix)
    B = sum(1.0 if rows[i][2] > 0 else 0.5 if rows[i][2] == 0 else 0.0 for i in ix)
    H = (A + 0.5) / (n + 1); F = (n - B + 0.5) / (n + 1)
    return -(Z(H) + Z(F)) / 2


def boot(f0, f1, n, it=10000, seed=0):
    rnd = random.Random(seed); d = []
    for _ in range(it):
        ix = [rnd.randrange(n) for _ in range(n)]
        d.append(f1(ix) - f0(ix))
    d.sort(); lo, hi = d[int(0.025 * it)], d[int(0.975 * it)]
    p = 2 * min(sum(1 for x in d if x <= 0), sum(1 for x in d if x >= 0)) / it
    return sum(d) / it, lo, hi, min(1.0, p)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--suffix", default=""); ap.add_argument("--key", default="all")
    a = ap.parse_args(); B = boards()
    print(f"# CF extras (suffix '{a.suffix}', key '{a.key}')\n")
    print("## (a) real (95 장면): 치우침 c, 짝 부트스트랩\n\n| 짝 | c 대조 | c CF | Δc [95 %] | p | 사라짐 A 고침 / 깨짐 (p) | 나타남 B 고침 / 깨짐 (p) |\n|---|---:|---:|---|---:|---|---|")
    pat, blk = B["real world (all 95)"]
    for lab, m0, m1 in PAIRS:
        S0, S1 = load(pat, m0, a.suffix, a.key), load(pat, m1, a.suffix, a.key)
        if S0 is None or S1 is None:
            print(f"| {lab} | 채점 없음 |"); continue
        r0 = sorted(scene_stats(S0, blk)); r1 = sorted(scene_stats(S1, blk)); assert [x[0] for x in r0] == [x[0] for x in r1]
        n = len(r0); full = list(range(n))
        mu, lo, hi, p = boot(lambda ix: crit(r0, ix), lambda ix: crit(r1, ix), n)
        fa = sum(1 for x, y in zip(r0, r1) if x[1] <= 0 < y[1]); ba = sum(1 for x, y in zip(r0, r1) if y[1] <= 0 < x[1])
        fb = sum(1 for x, y in zip(r0, r1) if x[2] <= 0 < y[2]); bb = sum(1 for x, y in zip(r0, r1) if y[2] <= 0 < x[2])
        print(f"| {lab} | {crit(r0, full):+.2f} | {crit(r1, full):+.2f} | {mu:+.2f} [{lo:+.2f}, {hi:+.2f}] | {p:.4f} | {fa} / {ba} ({binom_p(fa, fa + ba):.2g}) | {fb} / {bb} ({binom_p(fb, fb + bb):.2g}) |")
    print("\n## (b) 연속 문맥 margin g = (m_A + m_B) / 장면 평균 surprise, 짝 부트스트랩 (×100 = %)\n")
    print("| 보드 | 짝 | 장면 | g 대조 | g CF | Δg [95 %] | p | 원값 Δ(m_A+m_B) ×1e-3 [95 %] |\n|---|---|---:|---:|---:|---|---:|---|")
    for sname in SETS:
        pat, blk = B[sname]
        for lab, m0, m1 in PAIRS:
            S0, S1 = load(pat, m0, a.suffix, a.key), load(pat, m1, a.suffix, a.key)
            if S0 is None or S1 is None:
                continue
            r0 = sorted(scene_stats(S0, blk)); r1 = sorted(scene_stats(S1, blk)); n = len(r0)
            g0 = [100 * (x[1] + x[2]) / x[3] for x in r0]; g1 = [100 * (x[1] + x[2]) / x[3] for x in r1]
            mu, lo, hi, p = boot(lambda ix: sum(g0[i] for i in ix) / len(ix), lambda ix: sum(g1[i] for i in ix) / len(ix), n)
            w0 = [1e3 * (x[1] + x[2]) for x in r0]; w1 = [1e3 * (x[1] + x[2]) for x in r1]
            mw, lw, hw, _ = boot(lambda ix: sum(w0[i] for i in ix) / len(ix), lambda ix: sum(w1[i] for i in ix) / len(ix), n)
            print(f"| {sname} | {lab} | {n} | {sum(g0) / n:.3f} | {sum(g1) / n:.3f} | {mu:+.3f} [{lo:+.3f}, {hi:+.3f}] | {p:.4f} | {mw:+.2f} [{lw:+.2f}, {hw:+.2f}] |")


if __name__ == "__main__":
    main()
