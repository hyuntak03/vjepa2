"""[비판] IntPhys 2 Main: "기억과 비교" vs "직전 장과 비교" 외형 surprise (픽셀, 학습 · 모델 없음).

질문: 영속 · 불변 위반을 잡는 외형 정보가 (a) 바로 앞 장에 있나, (b) 가림 전 기억에만 있나.
crit_ip2_gap.py 의 결과 (드러남 t_r, 마지막 문맥 t_c) 를 쓴다.
  영역 R = 드러남 뒤 K 장에서 같은 쌍의 가능 · 불가능이 다른 픽셀 (쌍 1 · 2 의 합집합; 가능/불가능에 대칭이라 정답을 새지 않음).
  기억 surprise   s_mem(v) = R 에서 mean_t∈[t_r, t_r+K) |v(t) − v(t_c)|      (가림 전 마지막 문맥 장과 비교)
  직전 surprise   s_f2f(v) = R 에서 mean_t∈[t_r, t_r+K) |v(t) − v(t−1)|      (흐름 없는 "현재 → 다음" 외형 차이; 고정 카메라에서는 흐름 warp ≈ 항등)
두 surprise 각각 쌍 정확도 + 결과 항 / 문맥 항 분해 (crit_ip2_xor.py 와 같은 정의).
⚠️ R · t_r · t_c 는 픽셀 차이로 정한 오라클 위치다 — "기억 비교" 의 상한이지 방법이 아니다.
사용: python crit_ip2_memory.py --gap results/crit_ip2_gap.json [--K 6] [--out json]
"""
import json, csv, collections, argparse, numpy as np, decord, random, math

ROOT = "/workspace/dataset/benchmarks/IntPhys2/Main/"
TYPES = ["1_Possible", "1_Impossible", "2_Possible", "2_Impossible"]

def read(path):
    vr = decord.VideoReader(ROOT + path, width=128, height=128, num_threads=2)
    return vr.get_batch(list(range(0, len(vr), 10))).asnumpy().astype(np.float32).mean(-1)

def decomp(rows, key, rng):
    n = len(rows)
    if n == 0:
        return None
    rel = lambda a, b: (a - b) / ((a + b) / 2 + 1e-9)
    d1 = np.array([rel(r[key]["1_Impossible"], r[key]["1_Possible"]) for r in rows])
    d2 = np.array([rel(r[key]["2_Impossible"], r[key]["2_Possible"]) for r in rows])
    o, i = (d1 - d2) / 2, (d1 + d2) / 2
    acc = 100 * ((d1 > 0).mean() + (d2 > 0).mean()) / 2
    bs = []
    for _ in range(2000):
        ix = [rng.randrange(n) for _ in range(n)]
        bs.append(100 * ((d1[ix] > 0).mean() + (d2[ix] > 0).mean()) / 2)
    bs.sort()
    return dict(n=n, acc=float(acc), acc_ci=(bs[50], bs[1949]), one=float(((d1 > 0) != (d2 > 0)).mean()),
                corr=float(np.corrcoef(d1, d2)[0, 1]) if n > 2 else float("nan"),
                outcome_share=float((o ** 2).mean() / ((o ** 2).mean() + (i ** 2).mean())), mean_i=float(i.mean()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", required=True); ap.add_argument("--K", type=int, default=6); ap.add_argument("--out", default=None)
    ap.add_argument("--conds", default="permanence,immutability,continuity,solidity")
    a = ap.parse_args()
    G = json.load(open(a.gap))["scenes"]
    meta = collections.defaultdict(dict)
    for r in csv.DictReader(open(ROOT + "metadata.csv")):
        meta[r["SceneIndex"]][r["type"]] = r
    rows = []
    for g in G:
        if g["cond"] not in a.conds.split(",") or g["gap"] is None or g["t_c"] < 0:
            continue
        x = {t: read(meta[g["scene"]][t]["file_name"]) for t in TYPES}
        T = min(len(z) for z in x.values()); x = {t: z[:T] for t, z in x.items()}
        t0, t1 = g["t_r"], min(g["t_r"] + a.K, T)
        if t1 - t0 < 2:
            continue
        dd = np.maximum(np.abs(x["1_Possible"][t0:t1] - x["1_Impossible"][t0:t1]), np.abs(x["2_Possible"][t0:t1] - x["2_Impossible"][t0:t1])).max(0)
        R = dd > max(25.0, np.percentile(dd, 95))
        if R.sum() < 20:
            continue
        s_mem = {t: float(np.abs(x[t][t0:t1] - x[t][g["t_c"]][None])[:, R].mean()) for t in TYPES}
        s_f2f = {t: float(np.abs(x[t][t0:t1] - x[t][t0 - 1:t1 - 1])[:, R].mean()) for t in TYPES}
        # "본 적 있나": 드러남 전 모든 장 중 드러남 뒤 내용과 가장 닮은 장과의 차이 (재인 기억)
        s_min = {}; s_first = {}
        for t in TYPES:
            post = x[t][t0:t1][:, R]                                   # (K, |R|)
            pre = x[t][:t0][:, R]                                      # (t0, |R|)
            d = np.abs(post[None] - pre[:, None]).mean((1, 2))         # (t0,)
            s_min[t] = float(d.min()); s_first[t] = float(d[0])
        rows.append(dict(scene=g["scene"], cond=g["cond"], cam=g["cam"], gap=g["gap"], R=int(R.sum()), mem=s_mem, f2f=s_f2f, seen=s_min, first=s_first))
        print(g["scene"], g["cond"], g["cam"], "gap", g["gap"], "R", int(R.sum()), flush=True)
    rng = random.Random(0)
    res = {}
    for cond in ["all"] + a.conds.split(","):
        for cam in ["all", "Fixed", "Moving"]:
            xs = [r for r in rows if (cond == "all" or r["cond"] == cond) and (cam == "all" or r["cam"] == cam)]
            res[f"{cond}|{cam}"] = {k: decomp(xs, k, rng) for k in ["mem", "f2f", "seen", "first"]}
            m, f = res[f"{cond}|{cam}"]["seen"], res[f"{cond}|{cam}"]["f2f"]
            if m:
                print(f"{cond:13s} {cam:6s} n {m['n']:3d} | 본적 acc {m['acc']:5.1f} [{m['acc_ci'][0]:5.1f},{m['acc_ci'][1]:5.1f}] o-share {m['outcome_share']:.2f} mean_i {m['mean_i']:+.3f}"
                      f" | 직전 acc {f['acc']:5.1f} [{f['acc_ci'][0]:5.1f},{f['acc_ci'][1]:5.1f}] o-share {f['outcome_share']:.2f} mean_i {f['mean_i']:+.3f}")
    if a.out:
        json.dump(dict(summary=res, rows=rows, K=a.K), open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
