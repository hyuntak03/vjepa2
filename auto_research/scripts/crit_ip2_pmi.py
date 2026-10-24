"""[비판] IntPhys 2: 기준 AR 의 예측이 "복사보다 나은 몫" 에 문맥 항이 숨어 있나 (학습 없음, [E] 공유 캐시).

이론: surprise ≈ −log q(y|x) = −log q(y) − PMI. 결과 항 (−log q(y)) 은 "결과 영상이 얼마나 예측하기 어려운가" 이고,
복사 (마지막 블록 그대로) 의 오차도 같은 어려움을 담는다. 그래서 (예측 오차 − 복사 오차) 는 결과 항을 일부 지우고
"문맥으로 예측해서 얻은 몫" 만 남긴다 — 예측에 조건화가 있으면 문맥 항이 > 0 으로 드러난다.
영상마다 (TF 한 걸음, 블록 b = B0..30):
  s_pred = mean |p_tf[b] − h[b+1]|,  s_copy = mean |h[b] − h[b+1]|,  s_gain = s_pred − s_copy (음수 = 복사보다 나음)
쌍 정답: surprise(불가능) > surprise(가능). 결과 항 · 문맥 항은 crit_ip2_xor 정의 (s_gain 은 상대값 대신 원값 차이).
사용: python crit_ip2_pmi.py [--B0 4] [--out json]
"""
import json, argparse, collections, random, numpy as np

C = "/workspace/auto_research_cache/intphys2/Main/"

def decomp(rows, key, rel, rng):
    sc = collections.defaultdict(dict)
    for r in rows:
        sc[r["scene"]][("1" if r["pair"] == "1" else "2") + ("I" if r["imp"] else "P")] = r
    d1, d2, meta = [], [], []
    for s, v in sc.items():
        if len(v) != 4:
            continue
        f = (lambda a, b: (a - b) / ((a + b) / 2)) if rel else (lambda a, b: a - b)
        d1.append(f(v["1I"][key], v["1P"][key])); d2.append(f(v["2I"][key], v["2P"][key])); meta.append(v["1P"])
    d1, d2 = np.array(d1), np.array(d2)
    o, i = (d1 - d2) / 2, (d1 + d2) / 2
    n = len(d1)
    acc = 100 * ((d1 > 0).mean() + (d2 > 0).mean()) / 2
    bs_a, bs_i = [], []
    for _ in range(2000):
        ix = np.array([rng.randrange(n) for _ in range(n)])
        bs_a.append(100 * ((d1[ix] > 0).mean() + (d2[ix] > 0).mean()) / 2); bs_i.append(i[ix].mean())
    bs_a.sort(); bs_i.sort()
    return dict(n=n, acc=float(acc), acc_ci=(bs_a[50], bs_a[1949]), pair1=float(100 * (d1 > 0).mean()), pair2=float(100 * (d2 > 0).mean()),
                one=float(((d1 > 0) != (d2 > 0)).mean()), outcome_share=float((o ** 2).mean() / ((o ** 2).mean() + (i ** 2).mean())),
                mean_i=float(i.mean()), mean_i_ci=(float(bs_i[50]), float(bs_i[1949])))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--B0", type=int, default=4); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    meta = json.load(open(C + "meta.json"))
    H = np.load(C + "h.npy", mmap_mode="r"); P = np.load(C + "p_tf.npy", mmap_mode="r")
    rows = []
    for m in meta:
        k = m["row"]
        h = np.asarray(H[k], dtype=np.float32); p = np.asarray(P[k], dtype=np.float32)   # (32,256,D) · (31,256,D)
        nb = h.shape[0]
        bs = range(a.B0, nb - 1)
        e_pred = np.array([np.abs(p[b] - h[b + 1]).mean() for b in bs])
        e_copy = np.array([np.abs(h[b] - h[b + 1]).mean() for b in bs])
        rows.append(dict(scene=m["scene_index"], pair=str(m["pair_id"]), imp=bool(m["is_impossible"]), cond=m["condition"], cam=m["camera"],
                         pred=float(e_pred.mean()), copy=float(e_copy.mean()), gain=float((e_pred - e_copy).mean()),
                         ratio=float((e_pred / e_copy).mean())))
        if k % 100 == 0:
            print(k, rows[-1], flush=True)
    rng = random.Random(0)
    res = {}
    for cond in ["all", "permanence", "immutability", "continuity", "solidity"]:
        for cam in ["all", "Fixed", "Moving"]:
            xs = [r for r in rows if (cond == "all" or r["cond"] == cond) and (cam == "all" or r["cam"] == cam)]
            res[f"{cond}|{cam}"] = {"pred": decomp(xs, "pred", True, rng), "copy": decomp(xs, "copy", True, rng),
                                    "gain": decomp(xs, "gain", False, rng), "ratio": decomp(xs, "ratio", False, rng)}
            if cam == "all" or cond == "all":
                for key, st in res[f"{cond}|{cam}"].items():
                    print(f"{cond:13s} {cam:6s} {key:5s} n {st['n']:3d} acc {st['acc']:5.1f} [{st['acc_ci'][0]:5.1f},{st['acc_ci'][1]:5.1f}] "
                          f"짝1/2 {st['pair1']:5.1f}/{st['pair2']:5.1f} o-share {st['outcome_share']:.2f} mean_i {st['mean_i']:+.5f} [{st['mean_i_ci'][0]:+.5f},{st['mean_i_ci'][1]:+.5f}]")
    if a.out:
        json.dump(dict(summary=res, rows=rows, B0=a.B0), open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
