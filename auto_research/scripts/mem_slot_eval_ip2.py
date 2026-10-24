#!/usr/bin/env python3
"""[물리] 실험 1 채점 — IntPhys 2 Main: 학습한 기억 슬롯의 주소로 "본 적 있나" (고정 주 지표, [비판] KST 03:02 등록) (학습 없음).

  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 /venv/vjepa/bin/python -W ignore auto_research/scripts/mem_slot_eval_ip2.py --ckpts A P [--limit 40]

입력: [E] 공유 캐시 (블록 토큰 32 개, LN 목표 encoder). 영상마다 기억 슬롯 모델을 블록 0..31 에 읽기 모드로 돌린다.
seen(t, c) = min_{t' ≤ t−2} mean_d |h[t](c) − h[t'](c)| (직전 블록 제외) — [비판] crit_ip2_addr_ctrl.py 와 같은 식. 블록 t = B0+1..31 (B0 = 4) 평균.
주소 (칸 합 1):
  main  (주 지표) : Σ_k σ(존재_사후[t,k]) · a[t,k] / Σ_k σ(존재_사후[t,k])  — 슬롯 attention 을 존재 확률로 가중
  pred  (부 지표, 예측 꼴) : 사전 위치 (블록 t 를 보기 전 예측) 둘레 가우시안 (σ 1.5 칸), 가중 σ(존재_사전) · σ(보임_사전)
  rand  (대조) : 같은 구조 · 학습 안 한 무작위 초기화 (seed 0) 의 main 주소
  uniform · center (대조, 파라미터 없음)
  head3d · headrand (대조, [비판] KST 03:02 와 같은 주소): 사용자 3D 합성 상태 감독 run 의 상태 head attention (읽기만) · 같은 head 무작위 초기화 (같은 난수 순서)
표준 보고 (crit_ip2_pmi.decomp: 정확도 CI · 짝 1/2 · 결과 항 몫 · 문맥 항 CI) + 장면 짝 대응 차 bootstrap (4,000 회): A − rand, A − P (조건별).
"""
import argparse, json, os, random, sys
import numpy as np, torch

REPO = "/workspace/world_model/vjepa2"
sys.path.insert(0, os.path.join(REPO, "auto_research/scripts"))
from mem_slot import MemSlot, OUT  # noqa: E402
from crit_ip2_pmi import decomp     # noqa: E402

C = os.environ.get("IP2_CACHE", "/workspace/auto_research_cache/intphys2/Main/")
S, G = 256, 16


def load(name, dev):
    if name == "rand":
        torch.manual_seed(0); m = MemSlot()
    else:
        ck = torch.load(os.path.join(OUT, f"{name}.pt"), map_location="cpu", weights_only=False)
        m = MemSlot(K=ck["args"]["K"]); m.load_state_dict(ck["model"])
    return m.to(dev).eval()


def paired(rows, ka, kb, rng, n=4000):
    """장면마다 (두 쌍 정확도 평균) 의 차이 a − b, 장면 bootstrap 95 %"""
    sc = {}
    for r in rows:
        sc.setdefault(r["scene"], {})[("1" if r["pair"] == "1" else "2") + ("I" if r["imp"] else "P")] = r
    d = []
    for v in sc.values():
        if len(v) != 4:
            continue
        acc = lambda k: (float(v["1I"][k] > v["1P"][k]) + float(v["2I"][k] > v["2P"][k])) / 2
        d.append(acc(ka) - acc(kb))
    d = np.array(d)
    if len(d) == 0:
        return None
    bs = sorted(100 * d[[rng.randrange(len(d)) for _ in range(len(d))]].mean() for _ in range(n))
    return dict(diff=float(100 * d.mean()), ci=(bs[int(0.025 * n)], bs[int(0.975 * n) - 1]), n=int(len(d)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", default=["A", "P"])
    ap.add_argument("--B0", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.set_num_threads(8)
    dev = torch.device("cuda")
    models = {nm: load(nm, dev) for nm in a.ckpts + ["rand"]}
    from crit_ip2_addr_ctrl import load_head                                     # [비판] 대조 주소 그대로 (사용자 체크포인트는 읽기만)
    torch.manual_seed(0)
    heads = {"head3d": load_head("natural_ar_vw_state_phys3d_single", dev)}; _ = load_head("natural_ar_vw_state_phys2d_single", dev)
    heads["headrand"] = load_head(None, dev)
    meta = json.load(open(C + "meta.json")); H = np.load(C + "h.npy", mmap_mode="r")
    if a.limit:
        sc = sorted({m["scene_index"] for m in meta})[: a.limit]
        meta = [m for m in meta if m["scene_index"] in sc]
    yy, xx = torch.meshgrid(torch.arange(G, device=dev).float() + 0.5, torch.arange(G, device=dev).float() + 0.5, indexing="ij")
    cy, cx = yy.reshape(-1), xx.reshape(-1)
    cen = torch.exp(-(((cy - 8) ** 2 + (cx - 8) ** 2) / (2 * 4.0 ** 2))); cen /= cen.sum()
    rows = []
    for i, m in enumerate(meta):
        h = torch.from_numpy(np.asarray(H[m["row"]], dtype=np.float32)).to(dev)
        NB = h.shape[0]
        ts = torch.arange(a.B0 + 1, NB, device=dev)
        with torch.no_grad():
            seen = torch.stack([(h[t][None] - h[: t - 1]).abs().mean(-1).min(0).values if t > 2 else torch.zeros(S, device=dev) for t in range(NB)])
            first = (h - h[0:1]).abs().mean(-1)
            W = {"uniform": torch.full((len(ts), S), 1.0 / S, device=dev), "center": cen[None].expand(len(ts), -1)}
            for nm, hd in heads.items():
                W[nm] = hd(h.reshape(1, -1, 1280), S)["attn"][0, :, 0].float()[ts]
            for nm, md in models.items():
                pri, post, att = md(h[None])
                e = torch.sigmoid(post["exist"][0])                                   # (T,K)
                w = (e[..., None] * att[0]).sum(1) / e.sum(1, keepdim=True)           # (T,256)
                W[f"main_{nm}"] = w[ts]
                pp = pri["pos"][0]                                                    # (T,K,2) 칸 (x, y)
                g = torch.exp(-(((cx[None, None] - pp[..., 0:1]) ** 2 + (cy[None, None] - pp[..., 1:2]) ** 2) / (2 * 1.5 ** 2)))   # (T,K,256)
                g = g / g.sum(-1, keepdim=True)
                wp = torch.sigmoid(pri["exist"][0]) * torch.sigmoid(pri["vis"][0])
                W[f"pred_{nm}"] = ((wp[..., None] * g).sum(1) / wp.sum(1, keepdim=True).clamp_min(1e-6))[ts]
                if nm != "rand":
                    W[f"_ent_{nm}"] = -(w[ts] * (w[ts] + 1e-12).log()).sum(-1)       # 주소 엔트로피 (진단, 균등 5.55)
        r = dict(scene=m["scene_index"], pair=str(m["pair_id"]), imp=bool(m["is_impossible"]), cond=m["condition"], cam=m["camera"], diff=m.get("difficulty"))
        for nm, w in W.items():
            if nm.startswith("_ent_"):
                r[nm[1:]] = float(w.mean()); continue
            r[f"seen_{nm}"] = float((w * seen[ts]).sum(-1).mean())
            if nm.startswith("main_") or nm == "uniform":
                r[f"first_{nm}"] = float((w * first[ts]).sum(-1).mean())
        rows.append(r)
        if i % 100 == 0:
            print(i, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)}, flush=True)
    rng = random.Random(0)
    keys = [k for k in rows[0] if k.startswith(("seen_", "first_"))]
    res, pairs = {}, {}
    for cond in ["all", "permanence", "immutability", "continuity", "solidity"]:
        for cam in ["all", "Fixed", "Moving"]:
            xs = [r for r in rows if (cond == "all" or r["cond"] == cond) and (cam == "all" or r["cam"] == cam)]
            if len({r["scene"] for r in xs}) < 3:
                continue
            res[f"{cond}|{cam}"] = {key: decomp(xs, key, False, rng) for key in keys}
            for ka, kb in ([(f"seen_main_{x}", "seen_main_rand") for x in a.ckpts] + [(f"seen_main_{x}", "seen_headrand") for x in a.ckpts]
                           + [(f"seen_main_{x}", "seen_head3d") for x in a.ckpts] + [("seen_main_A", "seen_main_P"), ("seen_pred_A", "seen_pred_P")]):
                if ka in rows[0] and kb in rows[0]:
                    pairs[f"{cond}|{cam}|{ka}-{kb}"] = paired(xs, ka, kb, rng)
            if cam == "all":
                for key, st in res[f"{cond}|{cam}"].items():
                    if key.startswith("seen_"):
                        print(f"{cond:12s} {key:16s} acc {st['acc']:5.1f} [{st['acc_ci'][0]:5.1f},{st['acc_ci'][1]:5.1f}] 짝1/2 {st['pair1']:5.1f}/{st['pair2']:5.1f} "
                              f"o {st['outcome_share']:.2f} i {st['mean_i']:+.5f} [{st['mean_i_ci'][0]:+.5f},{st['mean_i_ci'][1]:+.5f}]", flush=True)
    for k, v in pairs.items():
        if v and k.split("|")[1] == "all":
            print(f"짝 대응 차 {k}: {v['diff']:+.1f} [{v['ci'][0]:+.1f}, {v['ci'][1]:+.1f}] (장면 {v['n']})", flush=True)
    ent = {nm: float(np.mean([r[f"ent_{nm}"] for r in rows])) for nm in a.ckpts}
    print("주소 엔트로피 (균등 5.55):", ent)
    out = os.path.join(REPO, f"auto_research/results/mem_slot_ip2{a.tag}.json")
    json.dump(dict(summary=res, paired=pairs, entropy=ent, rows=rows, B0=a.B0, ckpts=a.ckpts), open(out, "w"), indent=1)
    print("저장", out)


if __name__ == "__main__":
    main()
