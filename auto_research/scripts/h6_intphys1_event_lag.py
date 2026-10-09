#!/usr/bin/env python3
"""H6 — IntPhys 1 의 판별은 사건이 문맥 끝 바로 뒤일 때만 나오나 (GPU 불필요).

입력: Garrido sliding 채점이 창마다 남긴 surprise (`per_window.json`: combo, 창 시작 s, 문맥 C, surprise).
창 (combo = skip{k}_w{W}):  문맥 raw [s, s + k·C),  예측 raw [s + k·C, s + k·W),  튜블릿 = 2 샘플 = 2k raw 프레임.
사건 프레임 (1 부터 세는 파일 번호 → raw0 = F − 1):
  vis   = first_div_strict  (pos/imp 픽셀이 처음 갈리는 프레임 — 10 쌍 실측 일치)
  magic = magic_tick        (물리 사건. 가려진 사건은 vis 보다 이르다)
  τ = (F − 1) − (s + k·C)          예측 구간 안 사건 위치 (raw)
  j = floor(τ / 2k)               사건 튜블릿 (0 = 예측 첫 튜블릿)
  n_tub = (W − C)/2,  n_post = n_tub − j   (사건 이후 튜블릿 수)
창별 신호  Δ = S(imp) − S(pos)   (matched pair, 같은 창) ; 정답 = Δ > 0
→ (j, n_post) 칸마다 정확도.  n_post 를 고정하고 j 를 바꾸면 "문맥 끝에서의 거리" 효과, j 를 고정하고 n_post 를 바꾸면 "증거 양" 효과.
  대조: j < 0 (사건이 문맥 안), j ≥ n_tub (사건이 창 뒤 — pos/imp 픽셀 동일, 50 이어야 한다).
CI: 4 중항 (block) 단위 bootstrap.

  python auto_research/scripts/h6_intphys1_event_lag.py [--model vith]
"""
from __future__ import annotations
import argparse, collections, csv, json, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "z_research/Benchmarks/exp_results"
AUX = ROOT / "auto_research/_stage/IntPhys1_dev_by_scene"
OUT = ROOT / "auto_research/exp_results/h6"
MO = {"정지": "static", "이동": "moving"}
VI = {"눈앞": "visible", "가려짐": "occluded"}


def load_windows(model):
    W = {}
    for w in (16, 32):
        f = BENCH / f"intphys1_sliding__intphys1_dev_{model}_w{w}/per_window.json"
        if not f.exists():
            continue
        for vid, rows in json.load(open(f))["windows"].items():
            for combo, s, C, sur in rows:
                W[(vid, combo, int(s), int(C))] = float(sur)
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vith")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    W = load_windows(a.model)
    pairs = list(csv.DictReader((AUX / "pairs.csv").open()))
    vids = {r["video_id"]: r for r in csv.DictReader((AUX / "videos.csv").open())}
    rows = []
    for p in pairs:
        pos, imp = p["pos"], p["imp"]
        keys = [k[1:] for k in W if k[0] == pos]
        # 방향 (O1/O3): imp 가 물체가 보이는 프레임이 적으면 사라짐 (intphys1_direction_audit 와 같은 정의)
        d = ""
        if p["principle"][:2] in ("O1", "O3"):
            a_, b_ = int(vids[imp]["obj_visible_frames"]), int(vids[pos]["obj_visible_frames"])
            d = "disappear" if a_ < b_ else ("appear" if a_ > b_ else "equal")
        for combo, s, C in keys:
            if (imp, combo, s, C) not in W:
                continue
            k, Wn = (int(x) for x in re.match(r"skip(\d+)_w(\d+)", combo).groups())
            e = s + k * C
            n_tub = (Wn - C) // 2
            rec = dict(pair=f'{p["block"]}_{p["pair_id"]}_{p["scene"]}', block=p["scene"], principle=p["principle"][:2],
                       motion=MO.get(p["label_motion"], p["label_motion"]), vis=VI.get(p["label_vis"], p["label_vis"]),
                       direction=d, combo=combo, s=s, C=C, n_tub=n_tub,
                       delta=W[(imp, combo, s, C)] - W[(pos, combo, s, C)])
            for ev, col in (("vis", "first_div_strict"), ("magic", "magic_tick")):
                tau = (int(p[col]) - 1) - e
                j = int(np.floor(tau / (2 * k)))
                rec[f"j_{ev}"] = j
                rec[f"npost_{ev}"] = n_tub - j
            rows.append(rec)
    json.dump(rows, open(OUT / f"rows_{a.model}.json", "w"))
    rng = np.random.default_rng(0)
    blocks = sorted({r["block"] for r in rows})

    def acc_ci(sel):
        if not sel:
            return None
        by = collections.defaultdict(list)
        for r in sel:
            by[r["block"]].append(1.0 if r["delta"] > 0 else (0.5 if r["delta"] == 0 else 0.0))
        bl = list(by)
        m = np.mean([x for b in bl for x in by[b]])
        bs = []
        for _ in range(a.boot):
            pick = rng.choice(len(bl), len(bl))
            v = [x for i in pick for x in by[bl[i]]]
            bs.append(np.mean(v))
        return dict(acc=100 * m, lo=100 * np.percentile(bs, 2.5), hi=100 * np.percentile(bs, 97.5), n=len(sel), n_block=len(bl))

    res = {}
    for ev in ("vis", "magic"):
        for combo in sorted({r["combo"] for r in rows}):
            R = [r for r in rows if r["combo"] == combo]
            # 1) j 별 (전체), 대조 포함
            tab = {}
            for r in R:
                j = r[f"j_{ev}"]
                key = "ctx" if j < 0 else ("after" if j >= r["n_tub"] else f"j{j}")
                tab.setdefault(key, []).append(r)
            res[f"{ev}|{combo}|by_j"] = {k: acc_ci(v) for k, v in sorted(tab.items(), key=lambda kv: (kv[0][0] != 'c', kv[0]))}
            # 2) (j, n_post) 2 차원 — 예측 안 사건만
            tab2 = {}
            for r in R:
                j, npo = r[f"j_{ev}"], r[f"npost_{ev}"]
                if 0 <= j < r["n_tub"]:
                    tab2.setdefault(f"j{j}|n{npo}", []).append(r)
            res[f"{ev}|{combo}|by_j_npost"] = {k: acc_ci(v) for k, v in tab2.items()}
            # 3) 운동 × 가림 × j (예측 안 사건만, j 3 구간)
            tab3 = {}
            for r in R:
                j = r[f"j_{ev}"]
                if not (0 <= j < r["n_tub"]):
                    continue
                jb = "j0-1" if j <= 1 else ("j2-3" if j <= 3 else "j4+")
                for g in (f'{r["motion"]}/{r["vis"]}', f'{r["principle"]}', f'dir:{r["direction"]}' if r["direction"] else None):
                    if g:
                        tab3.setdefault(f"{g}|{jb}", []).append(r)
            res[f"{ev}|{combo}|by_group_jbin"] = {k: acc_ci(v) for k, v in sorted(tab3.items())}
    json.dump(res, open(OUT / f"h6_{a.model}.json", "w"), indent=1)
    # 요약 출력
    for key in [k for k in res if k.endswith("by_j")]:
        print("==", key)
        for kk, v in res[key].items():
            if v:
                print(f"   {kk:6s} {v['acc']:6.1f} [{v['lo']:5.1f},{v['hi']:5.1f}]  n={v['n']:5d} blocks={v['n_block']}")


if __name__ == "__main__":
    main()
