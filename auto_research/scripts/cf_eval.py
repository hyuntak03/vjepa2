#!/usr/bin/env python3
"""[T] 반사실 이어 학습 판정 (학습 없음) — 사전 등록 진행판 [T] KST 06:1x (auto_research, 2026-10-22).

  CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=8 /venv/vjepa/bin/python -W ignore auto_research/scripts/cf_eval.py --models base cf_a cf_b cf_c [--limit N]

모델: base = 기준 블록 AR (natural_ar_vw_state latest), 그 밖 = /workspace/auto_research_cache/cf_ft/<이름>.pt.
(1) 반사실 test 2×2 (50 장면 × 4 변형, /workspace/auto_research_cache/phys3d_cf_v1/test_*): 장면마다 문맥 C = 사건 전 블록 수
    (블록 b 의 마지막 장 6b+3 < 사건 장), 짝 1 = 장애물 있는 문맥 (W), 짝 2 = 없는 문맥 (N). 문맥은 짝 안에서 비트 단위로 같다.
    surprise (주) = 문맥 C 블록에서 K = 4 블록 굴린 예측과 관측의 토큰 L1 평균. (부) TF 한 걸음 L1 의 같은 블록 평균.
    표준 보고 = 쌍 정확도 (장면 bootstrap CI) · 짝 1 / 짝 2 · 결과 항 몫 · 문맥 항 평균 (CI) — [비판] 분해 (crit_ip2_pmi.decomp, 상대 차).
    같은 표를 외형만 자연 통계로 바꾼 판 (test_style_*, cf_style.py) 에도.
(2) IntPhys 2 Main ([E] 공유 캐시 블록 토큰 32): 영상마다 TF 한 걸음 L1 의 블록 4..30 평균 → 조건 4 (고체성 주) × 표준 보고.
(3) VOBench 끝 가림 ([P] 공유 캐시): 문맥 8 블록 → 8 블록 굴림, 화면 · 물체 자리 (라벨 중심 3×3) 토큰 L1, 움직임 · 정지 쌍 정확도.
(4) 자연 보류 (K710 val 150 창): TF 한 걸음 토큰 L1 평균.
출력: results/cf_eval.json (모델별, 장면별 행 포함 — 팔 사이 짝 차이 bootstrap 은 cf_compare.py) + 표."""
import argparse, collections, json, os, random, sys, time
import numpy as np, torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=["base"])
ap.add_argument("--ftdir", default="/workspace/auto_research_cache/cf_ft")
ap.add_argument("--cf", default="/workspace/auto_research_cache/phys3d_cf_v1")
ap.add_argument("--ip2", default="/workspace/auto_research_cache/intphys2/Main")
ap.add_argument("--vob", default="/workspace/auto_research_cache/vobench_late")
ap.add_argument("--vob_root", default="/workspace/dataset/benchmarks/VOBench_v1")
ap.add_argument("--hold", default="/workspace/auto_research_cache/natural_holdout/k710_320_holdout_h.npy")
ap.add_argument("--K", type=int, default=4)
ap.add_argument("--limit", type=int, default=0, help="스모크: 각 세트 앞 N 개만")
ap.add_argument("--parts", default="cf,style,ip2,vob,hold")
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "../results/cf_eval.json"))
a = ap.parse_args()
sys.path.insert(0, os.path.dirname(__file__))
_argv = sys.argv; sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT                                                     # noqa: E402
sys.argv = _argv
from crit_ip2_pmi import decomp                                             # noqa: E402
torch.set_num_threads(8)
dev = torch.device("cuda"); S, G = 256, 16
PARTS = set(a.parts.split(","))


def load_model(name):
    p = FT.load_pred("natural_ar_vw_state", "latest.pt")
    if name != "base":
        p.load_state_dict(torch.load(os.path.join(a.ftdir, f"{name}.pt"), map_location="cpu", weights_only=False)["predictor"])
    return p.to(dev).eval()


@torch.no_grad()
def tf(pred, h):                                                            # h (nb,S,D) → p (nb−1,S,D): 위치 b → 블록 b+1
    nb = h.shape[0]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return pred.forward_seq(h.reshape(1, nb * S, -1), torch.arange(nb * S, device=dev).unsqueeze(0)).float()[0].view(nb, S, -1)[:-1]


@torch.no_grad()
def roll(pred, ctx, K):
    C = ctx.shape[0]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return pred.rollout_kv(ctx.reshape(1, C * S, -1), torch.arange(C * S, device=dev).unsqueeze(0), K).float()[0].view(K, S, -1)


def T(x):
    return torch.from_numpy(np.asarray(x, np.float32)).to(dev)


def report(rows, key, groups=None):
    rng = random.Random(0)
    out = {"all": decomp(rows, key, True, rng)}
    for gk in (groups or []):
        for gv in sorted({r[gk] for r in rows}):
            sub = [r for r in rows if r[gk] == gv]
            if len({r["scene"] for r in sub}) >= 5:
                out[f"{gk}={gv}"] = decomp(sub, key, True, random.Random(0))
    return out


def cf_part(pred, split):
    H = np.load(os.path.join(a.cf, f"{split}_h.npy"), mmap_mode="r"); M = json.load(open(os.path.join(a.cf, f"{split}_meta.json")))
    by = collections.defaultdict(dict)
    for m in M:
        by[m["scene"]][m["variant"]] = m
    rows = []
    scenes = sorted(by)[: a.limit or None]
    for sc in scenes:
        v = by[sc]
        if len(v) != 4:
            continue
        te = v["W_pos"].get("t_event") or v["W_pos"]["t_hit"]
        C = max(1, min(15, sum(1 for b in range(16) if 6 * b + 3 < te)))
        K = min(a.K, 16 - C)
        for ctxv, pair in (("W_pos", "1"), ("N_pos", "2")):
            pr = roll(pred, T(H[v[ctxv]["row"], :C]), K)
            for var in ([ctxv, ctxv.replace("pos", "imp")]):
                hv = T(H[v[var]["row"]])
                p1 = tf(pred, hv)
                rows.append(dict(scene=sc, pair=pair, imp=var.endswith("imp"), camera=v[var]["camera"], screen=v[var]["screen"] is not None,
                                 roll=float((pr - hv[C:C + K]).abs().mean()), tfs=float((p1[C - 1:C - 1 + K] - hv[C:C + K]).abs().mean()), C=C))
    return {"n_scene": len({r["scene"] for r in rows}), "roll": report(rows, "roll", ["camera", "screen"]), "tf": report(rows, "tfs"), "rows": rows}


def ip2_part(pred):
    M = json.load(open(os.path.join(a.ip2, "meta.json"))); H = np.load(os.path.join(a.ip2, "h.npy"), mmap_mode="r")
    rows = []
    for m in M[: (4 * a.limit) or None]:
        h = T(H[m["row"]]); p = tf(pred, h)
        rows.append(dict(scene=m["scene_index"], pair=m["pair_id"], imp=bool(m["is_impossible"]), condition=m["condition"], camera=m["camera"],
                         s=float((p[4:] - h[5:]).abs().mean())))
    r_ = report(rows, "s", ["condition", "camera"]); r_["rows"] = rows
    return r_


def vob_centers(d, W0, key):
    L_ = json.load(open(os.path.join(d, "label.json")))["labels"]
    IDX = list(range(0, 96, 3))[:32]
    lab = {q["f"]: q for q in L_.get(key, [])}
    cs = []
    for b in range(16):
        q = [lab.get(IDX[2 * b]), lab.get(IDX[2 * b + 1])]
        cs.append(None if any(t is None for t in q) else (int(np.clip(np.mean([t["v"] for t in q]) / W0 * G, 0, G - 1)), int(np.clip(np.mean([t["u"] for t in q]) / W0 * G, 0, G - 1))))
    return cs


def vob_part(pred):
    H = np.load(os.path.join(a.vob, "h.npy"), mmap_mode="r"); M = json.load(open(os.path.join(a.vob, "meta.json")))
    res = []
    for m in M[: a.limit or None]:
        d0 = os.path.join(a.vob_root, m["files"]["pos_a"])
        W0 = Image.open(os.path.join(d0, "000000.png")).size[0]
        cen = {k: vob_centers(os.path.dirname(d0), W0, k) for k in ("A", "B")}
        otok = []
        for j in range(8):
            s_ = set()
            for k in ("A", "B"):
                c_ = cen[k][8 + j]
                if c_ is not None:
                    s_ |= {rr * G + cc for rr in range(max(0, c_[0] - 1), min(G, c_[0] + 2)) for cc in range(max(0, c_[1] - 1), min(G, c_[1] + 2))}
            otok.append(sorted(s_))
        oidx = torch.tensor([j * S + t for j, tt in enumerate(otok) for t in tt], device=dev, dtype=torch.long)
        mv = "static" if m["motion"] == "flat_static" else "moving"
        for pi, ii in ((0, 2), (1, 3)):
            hp = T(H[m["i"], pi]); hi = T(H[m["i"], ii])
            pr = roll(pred, hp[:8], 8).reshape(8 * S, -1)
            e = {nm: (pr - hh[8:].reshape(8 * S, -1)).abs().mean(-1) for nm, hh in (("pos", hp), ("imp", hi))}
            res.append(dict(mv=mv, viol=m["violation"], full=float(e["imp"].mean()) > float(e["pos"].mean()),
                            obj=(float(e["imp"][oidx].mean()) > float(e["pos"][oidx].mean())) if len(oidx) else None))
    out = {}
    for mv in ("moving", "static"):
        sel = [r for r in res if r["mv"] == mv]
        if sel:
            out[mv] = dict(n=len(sel), full=100 * np.mean([r["full"] for r in sel]), obj=100 * np.mean([r["obj"] for r in sel if r["obj"] is not None]))
        for vi in ("permanence", "shape", "color"):                          # 위반 종류별 (영속 = 사라짐 · 나타남)
            sv = [r for r in sel if r["viol"] == vi]
            if sv:
                out[f"{mv}/{vi}"] = dict(n=len(sv), full=100 * np.mean([r["full"] for r in sv]), obj=100 * np.mean([r["obj"] for r in sv if r["obj"] is not None]))
    return out


def hold_part(pred):
    H = np.load(a.hold, mmap_mode="r"); tot, n = 0.0, 0
    for i in range(H.shape[0] if not a.limit else min(a.limit, H.shape[0])):
        h = T(H[i]); p = tf(pred, h)
        tot += float((p - h[1:]).abs().mean()); n += 1
    return tot / max(n, 1)


RES = json.load(open(a.out)) if os.path.exists(a.out) and not a.limit else {}
for name in a.models:
    t0 = time.time(); pred = load_model(name); r = RES.get(name, {})
    if "cf" in PARTS:
        r["cf_test"] = cf_part(pred, "test")
    if "style" in PARTS and os.path.exists(os.path.join(a.cf, "test_style_h.npy")):
        r["cf_test_style"] = cf_part(pred, "test_style")
    if "ip2" in PARTS:
        r["ip2"] = ip2_part(pred)
    if "vob" in PARTS:
        r["vob"] = vob_part(pred)
    if "hold" in PARTS:
        r["hold_l1"] = hold_part(pred)
    RES[name] = r
    f = lambda d: f"정확도 {d['acc']:.1f} [{d['acc_ci'][0]:.1f}, {d['acc_ci'][1]:.1f}] · 짝 1 / 2 {d['pair1']:.0f} / {d['pair2']:.0f} · 결과 항 몫 {d['outcome_share']:.2f} · 문맥 항 {d['mean_i']:+.4f} [{d['mean_i_ci'][0]:+.4f}, {d['mean_i_ci'][1]:+.4f}]"
    print(f"== {name} ({time.time() - t0:.0f}s)")
    if "cf_test" in r:
        print(f"  반사실 test 굴림 (장면 {r['cf_test']['n_scene']}): " + f(r["cf_test"]["roll"]["all"]))
        print(f"  반사실 test TF          : " + f(r["cf_test"]["tf"]["all"]))
    if "cf_test_style" in r:
        print(f"  외형 바꾼 test 굴림     : " + f(r["cf_test_style"]["roll"]["all"]))
    if "ip2" in r:
        print(f"  IntPhys 2 전체          : " + f(r["ip2"]["all"]))
        if "condition=solidity" in r["ip2"]:
            print(f"  IntPhys 2 고체성        : " + f(r["ip2"]["condition=solidity"]))
    if "vob" in r:
        print("  VOBench 끝 가림: " + " · ".join(f"{k} (쌍 {v['n']}) 물체 자리 {v['obj']:.1f} / 화면 {v['full']:.1f}" for k, v in r["vob"].items()))
    if "hold_l1" in r:
        print(f"  자연 보류 TF L1 {r['hold_l1']:.4f}")
    del pred; torch.cuda.empty_cache()
DEF_OUT = os.path.join(os.path.dirname(__file__), "../results/cf_eval.json")
if not (a.limit and os.path.abspath(a.out) == os.path.abspath(DEF_OUT)):
    json.dump(RES, open(a.out, "w"), indent=1)
