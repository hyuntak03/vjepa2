"""[비판] IntPhys 2 Main: 문맥 정보가 마지막으로 보인 때와 위반이 드러나는 때 사이의 간격 (학습 · 모델 없음, 픽셀만).

장면마다 네 영상 (1P · 1I · 2P · 2I) 을 채점과 같은 6 fps (60 fps 에서 10 장 간격) 로, 128 px 회색조로 읽는다.
  문맥 차이 c(t) = |1P − 2P| 의 화면 평균 (두 쌍의 문맥 — 물체 있음/없음 · 장애물 있음/없음 — 이 보이는가)
  결과 차이 r(t) = max(|1P − 1I|, |2P − 2I|) (같은 문맥 안에서 가능 · 불가능이 갈라졌는가 = 위반이 보이는가)
잡음 바닥 = 첫 6 장의 r(t) 중앙값 (두 영상이 같은 구간; 압축 · 렌더 잡음).
  드러남 t_r = r(t) 가 바닥 × 3 과 바닥 + 1.0 (0–255) 을 처음 넘는 장
  마지막 문맥 t_c = t_r 이전에 c(t) 가 같은 문턱 (c 의 바닥 = 문맥 차이가 가장 작은 구간의 10 분위) 을 넘는 마지막 장
  간격 g = t_r − t_c (6 fps 장 수). 문맥이 드러남 직전까지 보이면 g = 1.
채점 창 (M = 16 · 32 · 48 장, 문맥 C ≤ M − 2) 의 문맥 안에 문맥 정보와 드러남이 함께 들어갈 수 있는 장면 비율 = P(g ≤ C_max).
사용: python crit_ip2_gap.py [--n N] [--out json]
"""
import csv, collections, json, argparse, numpy as np, decord

ROOT = "/workspace/dataset/benchmarks/IntPhys2/Main/"

def read(path):
    vr = decord.VideoReader(ROOT + path, width=128, height=128, num_threads=2)
    idx = list(range(0, len(vr), 10))
    x = vr.get_batch(idx).asnumpy().astype(np.float32).mean(-1)  # (T, 128, 128)
    return x

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=0); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(ROOT + "metadata.csv")))
    sc = collections.defaultdict(dict)
    for r in rows:
        sc[r["SceneIndex"]][r["type"]] = r
    keys = sorted(sc, key=int)
    if a.n:
        keys = keys[: a.n]
    out = []
    for k in keys:
        v = sc[k]
        x = {t: read(v[t]["file_name"]) for t in ["1_Possible", "1_Impossible", "2_Possible", "2_Impossible"]}
        T = min(len(z) for z in x.values())
        x = {t: z[:T] for t, z in x.items()}
        c = np.abs(x["1_Possible"] - x["2_Possible"]).mean((1, 2))
        dd = np.maximum(np.abs(x["1_Possible"] - x["1_Impossible"]), np.abs(x["2_Possible"] - x["2_Impossible"]))
        r = np.percentile(dd.reshape(T, -1), 99.5, axis=1)   # 국소 차이 (작은 물체도 잡게)
        floor = float(np.median(r[:6]))
        thr = max(1.6 * floor, floor + 15.0)
        hit = r > thr
        sus = [t for t in range(T - 2) if hit[t] and hit[t + 1] and hit[t + 2]]   # 3 장 연속 (순간 깜빡임 제외)
        t_r = int(sus[0]) if sus else -1
        g = None; t_c = -1; cfloor = float("nan")
        if t_r > 0:
            c0 = float(np.median(c[:3])); cfloor = float(np.percentile(c[:t_r], 10))
            if c0 - cfloor > 0.5:
                vis = np.where(c[:t_r] > cfloor + 0.5 * (c0 - cfloor))[0]
                t_c = int(vis[-1]) if len(vis) else -1
                g = (t_r - t_c) if t_c >= 0 else None
            else:
                t_c = t_r - 1; g = 1   # 문맥 차이가 처음부터 바닥 근처 = 문맥이 시간 내내 같은 크기로 보임 (장애물 등) 으로 본다
        out.append(dict(scene=k, cond=v["1_Possible"]["condition"], cam=v["1_Possible"]["Camera"], game=v["1_Possible"]["game_name"],
                        T=T, t_r=t_r, t_c=t_c, gap=g, r_floor=floor, c_floor=cfloor, r_max=float(r.max()), c_max=float(c.max())))
        print(k, out[-1]["cond"], out[-1]["cam"], out[-1]["game"], "T", T, "t_r", t_r, "t_c", t_c, "gap", g, flush=True)
    # 요약
    summ = {}
    for grp in ["all", "permanence", "immutability", "continuity", "solidity"]:
        for cam in ["all", "Fixed", "Moving"]:
            xs = [o for o in out if (grp == "all" or o["cond"] == grp) and (cam == "all" or o["cam"] == cam)]
            if not xs:
                continue
            gs = [o["gap"] for o in xs if o["gap"] is not None]
            d = dict(n=len(xs), n_gap=len(gs), no_reveal=sum(o["t_r"] < 0 for o in xs), no_ctx=sum(o["t_r"] > 0 and o["t_c"] < 0 for o in xs))
            if gs:
                gs = np.array(gs)
                d.update(gap_median=float(np.median(gs)), gap_q25=float(np.percentile(gs, 25)), gap_q75=float(np.percentile(gs, 75)),
                         **{f"frac_gap_le_{C}": float(np.mean(gs <= C)) for C in [2, 6, 14, 28, 42]})
            summ[f"{grp}|{cam}"] = d
    for k_, d in summ.items():
        print(k_, json.dumps(d))
    if a.out:
        json.dump(dict(summary=summ, scenes=out), open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
