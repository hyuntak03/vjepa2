#!/usr/bin/env python3
"""predictor 는 문맥 속 물체에 반응하는가 — 합성 vs 실제 영상 (2026-11-10, 사용자 "예측이 왜 그 신호를 안 쓰는데, predictor 의 그걸 봐야지").

같은 장면의 두 문맥: A = 물체 있음 (pos_a), B = 물체 없음 (pos_b). 문맥 차이는 물체 하나뿐이다.
predictor 가 물체를 미래로 넘기면, 미래 예측의 A − B 차이가 "물체가 있어야 할 자리" 에 생긴다.
물체 자리 R = 정답 미래의 A − B 차이 |h_A − h_B| (토큰마다 차원 평균) 상위 10 % 토큰 (장면마다, 미래 8 블록 전체에서).

재는 것 (장면마다, 미래 블록 k 마다):
  문맥 입력   ctx_rel[c]  = |z_A − z_B| / |z_A|  (문맥 블록 c, 토큰 평균) — 물체가 문맥 어디까지 보이나
  층 l (AR 걸음의 새 블록 토큰, 384 차원)
              in_R[l,k]   = R 토큰에서 |s_A − s_B| / |s_A|        — 물체 자리에서 내부 상태가 얼마나 달라지나
              out_R[l,k]  = R 밖 토큰에서 같은 값                  — 다른 자리 (대조)
  출력 p     pin_R[k] · pout_R[k] — 같은 값을 예측 p 에서 · hin_R[k] — 같은 값을 정답 h 에서 (기준)
              cos_R[k]    = cos(p_A − p_B, h_A − h_B), R 토큰만 이어 붙여서 — 방향이 맞나
  ⚠️ "예측 반응 ÷ 정답 차이" 로 읽는 비는 CLAUDE.md α 인용 금지 계열 — 인용 전 확인.

  /venv/vjepa/bin/python z_research/scripts/analysis/predictor_object_response.py extract --model vast_ar_vw_state --bench real --gpu 0 [--limit 300]
  /venv/vjepa/bin/python z_research/scripts/analysis/predictor_object_response.py report --models ...
출력: z_analysis/results/counterfactual_20261110/resp_<bench>_<model>.npz · resp_report.json
"""
import argparse, collections, json, os, sys, time
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
OUT = f"{REPO}/z_analysis/results/counterfactual_20261110"
BENCH = {"real": "real_world_permanence_uniform", "synth": "vobench_v3"}


def extract(model, bench, gpu, bs, limit):
    import torch, yaml
    from analysis.intphys2.model import build_from_config
    from analysis.intphys2.surprise import _context_target_indices
    from evals.world_model_analysis.data import WMADataset
    cfg = yaml.safe_load(open(f"{REPO}/cvpr/results/eval/testbed/{BENCH[bench]}_{model}/_resolved.yaml"))
    dev = torch.device(f"cuda:{gpu}"); torch.cuda.set_device(dev)
    ds = WMADataset(cfg)
    vi = collections.defaultdict(dict)
    for j, r in enumerate(ds.records):
        vi[r.block_id][r.raw["variant"]] = j
    blocks = [(b, v["pos_a"], v["pos_b"]) for b, v in sorted(vi.items()) if "pos_a" in v and "pos_b" in v]
    if limit:
        rng = np.random.default_rng(0)
        blocks = [blocks[j] for j in sorted(rng.choice(len(blocks), min(limit, len(blocks)), replace=False))]
    bundle = build_from_config(cfg["model"], dev)
    ts, S = bundle.tubelet_size, bundle.num_spatial_tokens
    ctx, N = int(cfg["surprise"]["context_length"]), ds.n_frames
    C, K = ctx // ts, (N - ctx) // ts
    mi = int(cfg["surprise"].get("mask_index", 0))
    pred = next(m for m in bundle.predictor.modules() if hasattr(m, "predictor_blocks"))
    assert type(pred).__module__.endswith("rollout_predictor"), "AR predictor 전용"
    store = {}

    def wrap(li, fn):
        def g(*a, **kw):
            o = fn(*a, **kw)
            store.setdefault(li, []).append((o[0] if isinstance(o, tuple) else o).detach())
            return o
        return g
    for li, blk in enumerate(pred.predictor_blocks):
        blk.forward_kv = wrap(li, blk.forward_kv)
    L = len(pred.predictor_blocks)
    ac = torch.autocast("cuda", dtype=torch.float16)
    rel = lambda a, b: ((a - b).abs().mean(-1) / a.abs().mean(-1).clamp(min=1e-6))          # 토큰마다
    res = collections.defaultdict(list)
    t0 = time.time()
    for s in range(0, len(blocks), bs):
        ch = blocks[s:s + bs]; n = len(ch)
        x = torch.stack([ds.clip(a) for _, a, _ in ch] + [ds.clip(b) for _, _, b in ch]).to(dev, bundle.dtype)
        ci, ti = _context_target_indices(ctx_frames=ctx, tgt_frames=N - ctx, tubelet_size=ts, spatial_tokens=S, batch_size=2 * n, device=dev)
        store.clear()
        with torch.no_grad(), ac:
            z = bundle.context_encoder(x, masks=[ci])
            p = bundle.predictor(z, ci, ti, mask_index=mi)
            hf = bundle.target_encoder(x)
            h = torch.gather(hf, 1, ti.unsqueeze(-1).expand(-1, -1, hf.size(-1)))
            h = torch.nn.functional.layer_norm(h.float(), (h.size(-1),))
        z = z.float().reshape(2 * n, C, S, -1); p = p.float().reshape(2 * n, K, S, -1); h = h.reshape(2 * n, K, S, -1)
        dh = (h[:n] - h[n:]).abs().mean(-1)                                           # (n, K, S)
        thr = torch.quantile(dh.reshape(n, -1), 0.9, dim=1).view(n, 1, 1)
        R = (dh >= thr).float()
        rin = lambda v: (v * R).sum(-1) / R.sum(-1).clamp(min=1)                      # (n, K): R 안 평균
        rout = lambda v: (v * (1 - R)).sum(-1) / (1 - R).sum(-1).clamp(min=1)
        res["ctx_rel"].append(rel(z[:n], z[n:]).mean(-1).cpu())                        # (n, C)
        # L1 이 물체를 지키게 만드는 힘: 물체를 통째로 잊은 예측 (B 문맥의 예측) 이 A 의 실제 미래에서 받는 L1 − 원래 예측의 L1 (2026-11-10)
        l1 = lambda a, b: (a - b).abs().mean((-1, -2))                                     # (n, K)
        res["e_AA"].append(l1(p[:n], h[:n]).cpu()); res["e_BA"].append(l1(p[n:], h[:n]).cpu())
        res["e_BB"].append(l1(p[n:], h[n:]).cpu()); res["e_AB"].append(l1(p[:n], h[n:]).cpu())
        rp, rh = rel(p[:n], p[n:]), rel(h[:n], h[n:])
        res["pin_R"].append(rin(rp).cpu()); res["pout_R"].append(rout(rp).cpu()); res["hin_R"].append(rin(rh).cpu()); res["hout_R"].append(rout(rh).cpu())
        dP, dH = (p[:n] - p[n:]) * R[..., None], (h[:n] - h[n:]) * R[..., None]
        res["cos_R"].append(((dP * dH).sum((2, 3)) / (dP.norm(dim=(2, 3)) * dH.norm(dim=(2, 3))).clamp(min=1e-9)).cpu())
        lin, lout = [], []
        for l in range(L):
            calls = store[l]
            if len(calls) != C + K - 1 or calls[-1].shape[1] != S:
                raise RuntimeError(f"층 {l}: 호출 {len(calls)} · {calls[-1].shape}")
            y = torch.stack(calls[-K:], 1).float()                                      # (2n, K, S, 384)
            ry = rel(y[:n], y[n:])
            lin.append(rin(ry)); lout.append(rout(ry))
        res["in_R"].append(torch.stack(lin, 1).cpu()); res["out_R"].append(torch.stack(lout, 1).cpu())   # (n, L, K)
        if (s // bs) % 25 == 0:
            print(f"[{bench} {model}] {s + n}/{len(blocks)} {time.time() - t0:.0f}s", flush=True)
    os.makedirs(OUT, exist_ok=True)
    np.savez(f"{OUT}/resp_{bench}_{model}.npz", **{k: torch.cat(v).numpy() for k, v in res.items()}, block=np.array([b for b, _, _ in blocks]))
    print(f"저장 {OUT}/resp_{bench}_{model}.npz · 장면 {len(blocks)} · {time.time() - t0:.0f}s")


def report(models, benches):
    R = {}
    for m in models:
        for b in benches:
            fn = f"{OUT}/resp_{b}_{m}.npz"
            if not os.path.exists(fn):
                continue
            d = np.load(fn)
            mean = lambda k: d[k].mean(0)
            r = {k: mean(k).tolist() for k in ("ctx_rel", "pin_R", "pout_R", "hin_R", "hout_R", "cos_R", "in_R", "out_R")}
            r["n"] = int(len(d["block"]))
            R[f"{b}/{m}"] = r
            f = lambda v: " ".join(f"{x:.3f}" for x in v)
            print(f"\n== {b} · {m} (장면 {r['n']})")
            print(f"  문맥 A−B 차이 (블록 1..8)      {f(r['ctx_rel'])}")
            print(f"  정답 h  R 안 / 밖 (미래 1..8)  {f(r['hin_R'])}  /  {f(r['hout_R'])}")
            print(f"  예측 p  R 안 / 밖             {f(r['pin_R'])}  /  {f(r['pout_R'])}")
            print(f"  예측 방향 cos (R)             {f(r['cos_R'])}")
            inR, outR = np.array(r["in_R"]), np.array(r["out_R"])
            print(f"  층별 R 안 (미래 평균)          {f(inR.mean(1))}")
            print(f"  층별 R 밖 (미래 평균)          {f(outR.mean(1))}")
    json.dump(R, open(f"{OUT}/resp_report.json", "w"), indent=1)
    print("저장", f"{OUT}/resp_report.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "report"])
    ap.add_argument("--model"); ap.add_argument("--models", nargs="+")
    ap.add_argument("--bench", choices=list(BENCH)); ap.add_argument("--benches", nargs="+", default=["synth", "real"])
    ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--bs", type=int, default=4); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    extract(a.model, a.bench, a.gpu, a.bs, a.limit) if a.stage == "extract" else report(a.models, a.benches)


if __name__ == "__main__":
    main()
