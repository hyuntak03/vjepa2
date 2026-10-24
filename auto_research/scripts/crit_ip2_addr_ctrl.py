"""[비판] IntPhys 2: latent 기억 비교의 "주소" 를 무엇이 정해야 하나 — 합성 상태 head 의 attention vs 대조 주소들 (학습 없음).

영상마다 블록 t = B0+1..31 평균, 칸별:
  seen(t, c)  = min_{t' ≤ t−2} mean_d |h[t](c) − h[t'](c)|   ("본 적 있나", 직전 블록 제외)
  first(t, c) = mean_d |h[t](c) − h[0](c)|
주소 가중 w(t, ·) (칸 합 1) 별로 s = mean_t Σ_c w(t, c) · seen(t, c) (first 도 같게).
주소:
  head3d   : 3D 합성 물체 1 개 상태 감독 run 의 head attention (관측 블록 토큰에 건 것)
  head2d   : 2D 합성 물체 1 개 run 의 head attention
  headrand : 같은 구조, 학습 안 한 무작위 쿼리 (seed 0) — "아무 attention 풀링" 대조
  uniform  : 균등 (화면 전체)
  motion   : |h[t] − h[t−1]| 비례 (움직임 돌출, 파라미터 없음)
  novel10  : 그 블록 seen 값 상위 10 % 칸 균등 (스스로 고른 최대 신규성)
  center   : 화면 가운데 가우시안 (σ = 4 칸)
표준 보고 (crit_ip2_pmi.decomp).
"""
import os, sys, json, argparse, random, numpy as np, torch

REPO = "/workspace/world_model/vjepa2"
sys.path.insert(0, os.path.join(REPO, "auto_research/scripts")); sys.path.insert(0, REPO)
from app.vjepa_frozen.state_head import SingleStateHead  # noqa: E402
from crit_ip2_pmi import decomp  # noqa: E402

C = os.environ.get("IP2_CACHE", "/workspace/auto_research_cache/intphys2/Main/")   # Debug 확인: IP2_CACHE=.../Debug/

def load_head(run, dev):
    hd = SingleStateHead()
    if run is not None:
        ck = torch.load(os.path.join(REPO, "z_training/runs", run, "latest.pt"), map_location="cpu", weights_only=False)
        hd.load_state_dict(ck["state_head"])
    return hd.to(dev).eval()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--B0", type=int, default=4); ap.add_argument("--out", default=None); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(8); torch.manual_seed(0)
    dev = torch.device("cuda")
    heads = {"head3d": load_head("natural_ar_vw_state_phys3d_single", dev), "head2d": load_head("natural_ar_vw_state_phys2d_single", dev),
             "headrand": load_head(None, dev)}
    meta = json.load(open(C + "meta.json")); H = np.load(C + "h.npy", mmap_mode="r")
    if a.limit:
        meta = meta[: a.limit]
    if os.environ.get("IP2_EXCLUDE_MAIN"):   # Main 과 겹치는 영상이 하나라도 있는 장면은 뺀다
        import csv
        mainset = {r["name"] for r in csv.DictReader(open("/workspace/dataset/benchmarks/IntPhys2/Main/metadata.csv"))}
        bad = {m["scene_index"] for m in meta if m["name"] in mainset}
        meta = [m for m in meta if m["scene_index"] not in bad]
        print(f"Main 과 겹치는 장면 {sorted(bad)} 제외 → 영상 {len(meta)}", flush=True)
    S = 256
    yy, xx = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
    cen = torch.tensor(np.exp(-(((yy - 7.5) ** 2 + (xx - 7.5) ** 2) / (2 * 4.0 ** 2))).reshape(-1), dtype=torch.float32, device=dev); cen /= cen.sum()
    rows = []
    for m in meta:
        k = m["row"]
        h = torch.from_numpy(np.asarray(H[k], dtype=np.float32)).to(dev)
        NB = h.shape[0]
        with torch.no_grad():
            seen = torch.stack([(h[t][None] - h[: t - 1]).abs().mean(-1).min(0).values if t > 2 else torch.zeros(S, device=dev) for t in range(NB)])
            first = (h - h[0:1]).abs().mean(-1)
            prev = torch.cat([torch.zeros(1, S, device=dev), (h[1:] - h[:-1]).abs().mean(-1)])
            ts = torch.arange(a.B0 + 1, NB, device=dev)
            W = {}
            for nm, hd in heads.items():
                W[nm] = hd(h.reshape(1, -1, 1280), S)["attn"][0, :, 0].float()[ts]
            W["uniform"] = torch.full((len(ts), S), 1.0 / S, device=dev)
            W["motion"] = prev[ts] / prev[ts].sum(-1, keepdim=True)
            q = torch.quantile(seen[ts], 0.9, dim=-1, keepdim=True)
            nv = (seen[ts] >= q).float(); W["novel10"] = nv / nv.sum(-1, keepdim=True)
            W["center"] = cen[None].expand(len(ts), -1)
            r = dict(scene=m["scene_index"], pair=str(m["pair_id"]), imp=bool(m["is_impossible"]), cond=m["condition"], cam=m["camera"])
            for nm, w in W.items():
                r[f"seen_{nm}"] = float((w * seen[ts]).sum(-1).mean()); r[f"first_{nm}"] = float((w * first[ts]).sum(-1).mean())
            # head 들의 주소가 서로 얼마나 겹치나 (attention 상관)
            r["corr_3d_rand"] = float(torch.corrcoef(torch.stack([W["head3d"].reshape(-1), W["headrand"].reshape(-1)]))[0, 1])
            r["corr_3d_motion"] = float(torch.corrcoef(torch.stack([W["head3d"].reshape(-1), W["motion"].reshape(-1)]))[0, 1])
        rows.append(r)
        if k % 200 == 0:
            print(k, {kk: round(v, 3) for kk, v in r.items() if isinstance(v, float)}, flush=True)
    rng = random.Random(0)
    keys = [k for k in rows[0] if k.startswith(("seen_", "first_"))]
    res = {}
    for cond in ["all", "permanence", "immutability", "continuity", "solidity"]:
        for cam in ["all", "Fixed", "Moving"]:
            xs = [r for r in rows if (cond == "all" or r["cond"] == cond) and (cam == "all" or r["cam"] == cam)]
            if len({r["scene"] for r in xs}) < 3:
                continue
            res[f"{cond}|{cam}"] = {key: decomp(xs, key, False, rng) for key in keys}
            if cond == "all":
                for key, st in res[f"{cond}|{cam}"].items():
                    print(f"{cond:6s} {cam:6s} {key:15s} acc {st['acc']:5.1f} [{st['acc_ci'][0]:5.1f},{st['acc_ci'][1]:5.1f}] 짝1/2 {st['pair1']:5.1f}/{st['pair2']:5.1f} "
                          f"o {st['outcome_share']:.2f} i {st['mean_i']:+.5f} [{st['mean_i_ci'][0]:+.5f},{st['mean_i_ci'][1]:+.5f}]", flush=True)
    cr = np.array([r["corr_3d_rand"] for r in rows]); cm = np.array([r["corr_3d_motion"] for r in rows])
    print(f"attention 상관 head3d~headrand {np.nanmean(cr):.2f} · head3d~motion {np.nanmean(cm):.2f}")
    if a.out:
        json.dump(dict(summary=res, rows=rows, B0=a.B0), open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
