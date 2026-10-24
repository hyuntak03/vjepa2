"""[비판] IntPhys 2 Main: 결과 편향 vs 문맥 조건화 분해 (학습 · 모델 없음, per_video.csv 만).

IntPhys 2 설계 (논문 Fig. 1): 쌍 1 의 가능 결과는 쌍 2 에서 불가능, 반대도 같다.
  쌍 1: 가능 = 결과 A, 불가능 = 결과 B   (문맥 1)
  쌍 2: 가능 = 결과 B, 불가능 = 결과 A   (문맥 2)
Δ1 = s(1_Imp) - s(1_Pos),  Δ2 = s(2_Imp) - s(2_Pos)   (상대값: 두 영상 평균으로 나눔)
  결과 항  o = (Δ1 - Δ2) / 2   — 문맥과 무관하게 결과 B 가 A 보다 놀라운 정도
  문맥 항  i = (Δ1 + Δ2) / 2   — 문맥이 바뀌면 어느 결과가 놀라운지 바뀌는 정도 (차이의 차이)
쌍 1 정답 ⇔ i + o > 0, 쌍 2 정답 ⇔ i − o > 0. 결과 항이 크면 장면마다 정확히 한 쌍만 맞는다 → 50 %.
사용: python crit_ip2_xor.py <per_video.csv> [...]  → 표 출력 + JSON (--out)
"""
import sys, csv, collections, json, math, random, argparse

def load(path):
    by = collections.defaultdict(dict)  # C -> {(scene, type): (s, cond, cam, diff)}
    for r in csv.DictReader(open(path)):
        C = int(r["context_length"])
        t = ("1" if r["pair_id"] == "1" else "2") + ("I" if r["is_impossible"] == "True" else "P")
        by[C][(r["scene_index"], t)] = (float(r["surprise_avg"]), r["condition"], r["camera"], r["difficulty"])
    return by

def scenes(d):
    sc = collections.defaultdict(dict)
    for (s, t), v in d.items():
        sc[s][t] = v
    out = []
    for s, v in sc.items():
        if len(v) != 4:
            continue
        rel = lambda a, b: (a - b) / ((a + b) / 2)
        d1 = rel(v["1I"][0], v["1P"][0])
        d2 = rel(v["2I"][0], v["2P"][0])
        out.append(dict(scene=s, cond=v["1P"][1], cam=v["1P"][2], diff=v["1P"][3], d1=d1, d2=d2,
                        o=(d1 - d2) / 2, i=(d1 + d2) / 2))
    return out

def stats(xs, rng):
    n = len(xs)
    if n == 0:
        return None
    c1 = [x["d1"] > 0 for x in xs]; c2 = [x["d2"] > 0 for x in xs]
    acc = 100 * (sum(c1) + sum(c2)) / (2 * n)
    both = sum(a and b for a, b in zip(c1, c2)) / n
    one = sum(a != b for a, b in zip(c1, c2)) / n
    none_ = 1 - both - one
    o = [x["o"] for x in xs]; i = [x["i"] for x in xs]
    var = lambda v: sum((t - sum(v) / len(v)) ** 2 for t in v) / max(len(v) - 1, 1)
    mo2 = sum(t * t for t in o) / n; mi2 = sum(t * t for t in i) / n   # 2 차 모멘트 (평균 포함)
    d1 = [x["d1"] for x in xs]; d2 = [x["d2"] for x in xs]
    m1, m2 = sum(d1) / n, sum(d2) / n
    cov = sum((a - m1) * (b - m2) for a, b in zip(d1, d2)) / max(n - 1, 1)
    corr = cov / math.sqrt(var(d1) * var(d2)) if var(d1) > 0 and var(d2) > 0 else float("nan")
    # 장면 bootstrap: 정확도 · 문맥 항 평균
    bs_acc, bs_i = [], []
    for _ in range(2000):
        smp = [xs[rng.randrange(n)] for _ in range(n)]
        bs_acc.append(100 * sum((x["d1"] > 0) + (x["d2"] > 0) for x in smp) / (2 * n))
        bs_i.append(sum(x["i"] for x in smp) / n)
    bs_acc.sort(); bs_i.sort()
    ci = lambda v: (v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1])
    return dict(n_scene=n, acc=acc, acc_ci=ci(bs_acc), both=both, one=one, none=none_,
                corr_d1_d2=corr, outcome_share=mo2 / (mo2 + mi2), frac_ctx_dominant=sum(abs(a) > abs(b) for a, b in zip(i, o)) / n,
                mean_i=sum(i) / n, mean_i_ci=ci(bs_i), rms_o=math.sqrt(mo2), rms_i=math.sqrt(mi2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--C", default="best", help="best | all | <int>")
    a = ap.parse_args()
    rng = random.Random(0)
    res = {}
    for path in a.csv:
        by = load(path)
        accs = {C: stats(scenes(d), rng)["acc"] for C, d in by.items()}
        Cs = sorted(by) if a.C == "all" else ([max(accs, key=accs.get)] if a.C == "best" else [int(a.C)])
        for C in Cs:
            sc = scenes(by[C])
            groups = [("all", sc)] + [(c, [x for x in sc if x["cond"] == c]) for c in ["permanence", "immutability", "continuity", "solidity"]] \
                     + [(f"cam={k}", [x for x in sc if x["cam"] == k]) for k in ["Fixed", "Moving"]]
            key = f"{path}|C{C}"
            res[key] = {g: stats(xs, rng) for g, xs in groups}
            print(f"\n== {path}  C={C}")
            print(f"{'group':14s} {'n':>4s} {'acc':>6s} {'[95%]':>14s} {'both':>5s} {'one':>5s} {'none':>5s} {'r(d1,d2)':>8s} {'o-share':>7s} {'ctx>out':>7s} {'mean_i':>8s} {'[95%]':>18s} {'rms_o':>6s} {'rms_i':>6s}")
            for g, st in res[key].items():
                if st is None:
                    continue
                print(f"{g:14s} {st['n_scene']:4d} {st['acc']:6.1f} [{st['acc_ci'][0]:5.1f},{st['acc_ci'][1]:5.1f}] {st['both']:5.2f} {st['one']:5.2f} {st['none']:5.2f} "
                      f"{st['corr_d1_d2']:8.2f} {st['outcome_share']:7.2f} {st['frac_ctx_dominant']:7.2f} {st['mean_i']:8.4f} [{st['mean_i_ci'][0]:7.4f},{st['mean_i_ci'][1]:7.4f}] {st['rms_o']:6.4f} {st['rms_i']:6.4f}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
