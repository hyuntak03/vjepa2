#!/usr/bin/env python3
"""반사실 놀람 분해 — 합성 (VOBench v3) 에서는 되는데 실제 영상 영속성에서는 무엇이 모자라는가 (2026-11-10, 사용자 "negative loss 는 counterfactual 상황에 놀라도록 학습하는 것, 그 철학에서 찾아봐").

짝 (matched pairing) 마다 문맥은 같고 미래만 다르다: 가능 미래 h⁺ · 불가능 미래 h⁻ (둘 다 LN(target encoder), 채점과 같은 공간), 예측 p (문맥 → 미래).
미래 블록 k 마다 토큰 · 차원 평균으로
  e⁺ = mean|p − h⁺|            가능 미래에 대한 놀람 (= 예측 오차)
  e⁻ = mean|p − h⁻|            불가능 미래에 대한 놀람
  m  = e⁻ − e⁺                 놀람 차이 (채점: m > 0 이면 정답)
  D  = mean|h⁺ − h⁻|           반사실 차이: 두 미래가 정답 공간에서 얼마나 다른가 (예측과 무관, 정답 encoder 만의 양)
삼각부등식으로 원소마다 |p−h⁻| − |p−h⁺| ≤ |h⁺−h⁻| 이므로 −D ≤ m ≤ D.
  m / e⁺ = 상대 놀람 차이 = (D / e⁺) × (m / D)
  D / e⁺ = 반사실 신호 비율: 반사실 차이가 예측 오차의 몇 배인가 (신호가 오차에 묻혀 있나)
  m / D  = 반사실 민감도: 쓸 수 있는 차이 중 예측이 가능 미래 편에 선 몫 (1 = 가능 미래와 같음, 0 = 무관심)
  ⚠️ m / D 는 CLAUDE.md "α (헤지 계수) 인용 금지" 와 같은 계열 — 문서 · 논문 인용 전 사용자 확인.
토큰마다 e⁺ · e⁻ · D (차원 평균) 도 저장 → 반사실 영역 (D 상위 토큰) 과 나머지로 나눠 본다.

  /venv/vjepa/bin/python z_research/scripts/analysis/counterfactual_sensitivity.py extract --model vast_ar_vw_state --bench real --gpu 0
  /venv/vjepa/bin/python z_research/scripts/analysis/counterfactual_sensitivity.py report --models ...
bench: real = real_world_permanence_uniform (95 장면 × 짝 A · B) · synth = vobench_v3 (840 × A · B)
설정 = cvpr/results/eval/testbed/<bench 이름>_<모델>/_resolved.yaml (채점과 같은 모델 · 창 · dtype). 검증: e⁺ · e⁻ 블록 평균 = per_block.json 의 영상별 놀람.
출력: z_analysis/results/counterfactual_20261110/<bench>_<model>.npz · report.json
"""
import argparse, collections, json, os, sys, time
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
OUT = f"{REPO}/z_analysis/results/counterfactual_20261110"
BENCH = {"real": "real_world_permanence_uniform", "synth": "vobench_v3"}
PAIRS = {"A": ("pos_a", "imp_ab"), "B": ("pos_b", "imp_ba")}


def extract(model, bench, gpu, bs, limit):
    import torch, yaml
    from analysis.intphys2.model import build_from_config
    from analysis.intphys2.surprise import _context_target_indices
    from evals.world_model_analysis.data import WMADataset
    cfg = yaml.safe_load(open(f"{REPO}/cvpr/results/eval/testbed/{BENCH[bench]}_{model}/_resolved.yaml"))
    dev = torch.device(f"cuda:{gpu}"); torch.cuda.set_device(dev)
    ds = WMADataset(cfg)
    vi = collections.defaultdict(dict)
    for i, r in enumerate(ds.records):
        vi[r.block_id][r.raw["variant"]] = i
    pairs = [(b, ab, v[pos], v[imp]) for b, v in sorted(vi.items()) for ab, (pos, imp) in PAIRS.items() if pos in v and imp in v]
    if limit:
        pairs = pairs[:limit]
    bundle = build_from_config(cfg["model"], dev)
    ts, S = bundle.tubelet_size, bundle.num_spatial_tokens
    ctx, N = int(cfg["surprise"]["context_length"]), ds.n_frames
    K = (N - ctx) // ts
    mi = int(cfg["surprise"].get("mask_index", 0))
    ac = torch.autocast("cuda", dtype=torch.float16) if cfg["model"].get("autocast") in ("float16", "fp16") else torch.autocast("cuda", enabled=False)
    out = collections.defaultdict(list)
    t0 = time.time()
    for s in range(0, len(pairs), bs):
        ch = pairs[s:s + bs]; n = len(ch)
        x = torch.stack([ds.clip(pi) for _, _, pi, _ in ch] + [ds.clip(ii) for _, _, _, ii in ch]).to(dev, bundle.dtype)
        ci, ti = _context_target_indices(ctx_frames=ctx, tgt_frames=N - ctx, tubelet_size=ts, spatial_tokens=S, batch_size=2 * n, device=dev)
        with torch.no_grad(), ac:
            z = bundle.context_encoder(x, masks=[ci])
            p = bundle.predictor(z, ci, ti, mask_index=mi)
            hf = bundle.target_encoder(x)
            h = torch.gather(hf, 1, ti.unsqueeze(-1).expand(-1, -1, hf.size(-1)))
            h = torch.nn.functional.layer_norm(h.float(), (h.size(-1),))
        p = p.float().reshape(2 * n, K, S, -1); h = h.reshape(2 * n, K, S, -1)
        pp, pi_ = p[:n], p[n:]
        hp, hi = h[:n], h[n:]
        out["e_pos"].append((pp - hp).abs().mean(-1).half().cpu())          # (n, K, S)
        out["e_imp"].append((pp - hi).abs().mean(-1).half().cpu())          # 문맥이 같으므로 p⁺ 를 쓴다 (채점과 같음)
        out["e_imp_own"].append((pi_ - hi).abs().mean(-1).half().cpu())     # 검증용: 불가능 영상 자기 예측
        out["D"].append((hp - hi).abs().mean(-1).half().cpu())
        out["ctx_pdiff"].append((pp - pi_).abs().amax((1, 2, 3)).cpu())
        if (s // bs) % 25 == 0:
            print(f"[{bench} {model}] {s + n}/{len(pairs)} {time.time() - t0:.0f}s", flush=True)
    os.makedirs(OUT, exist_ok=True)
    rec = lambda i: ds.records[i]
    np.savez(f"{OUT}/{bench}_{model}.npz", **{k: torch.cat(v).numpy() for k, v in out.items()},
             block=np.array([b for b, _, _, _ in pairs]), pair=np.array([ab for _, ab, _, _ in pairs]),
             vid_pos=np.array([rec(pi).video_id for _, _, pi, _ in pairs]), vid_imp=np.array([rec(ii).video_id for _, _, _, ii in pairs]),
             condition=np.array([rec(pi).raw.get("condition", "") for _, _, pi, _ in pairs]),
             source=np.array([rec(pi).raw.get("source", rec(pi).raw.get("objset", "")) for _, _, pi, _ in pairs]))
    print(f"저장 {OUT}/{bench}_{model}.npz · 짝 {len(pairs)} · {time.time() - t0:.0f}s")


def verify(model, bench):
    d = np.load(f"{OUT}/{bench}_{model}.npz")
    pv = json.load(open(f"{REPO}/cvpr/results/eval/testbed/{BENCH[bench]}_{model}/per_block.json"))["per_video_surprise"]
    ep = d["e_pos"].astype(np.float64).mean((1, 2)); ei = d["e_imp_own"].astype(np.float64).mean((1, 2))
    a = np.array([pv[v] for v in d["vid_pos"]]); b = np.array([pv[v] for v in d["vid_imp"]])
    return {"max_rel_err_pos": float(np.max(np.abs(ep - a) / a)), "max_rel_err_imp": float(np.max(np.abs(ei - b) / b)),
            "ctx_pdiff_max": float(d["ctx_pdiff"].max()), "n": int(len(ep))}


def summarize(d, sel, blocks=slice(None)):
    f = lambda k: d[k][sel][:, blocks].astype(np.float64)
    ep, ei, D = f("e_pos"), f("e_imp"), f("D")
    EP, EI, DD = ep.mean((1, 2)), ei.mean((1, 2)), D.mean((1, 2))          # 장면별 (채점과 같은 평균)
    M = EI - EP
    # 반사실 영역 = 장면마다 D 상위 10 % 토큰 (블록 · 위치 전체에서)
    flatD = D.reshape(len(D), -1); thr = np.quantile(flatD, 0.9, axis=1, keepdims=True)
    reg = (flatD >= thr)
    m_tok = (ei - ep).reshape(len(D), -1)
    share_m = (m_tok * reg).sum(1) / m_tok.sum(1).clip(min=1e-12)
    e_reg = (ep.reshape(len(D), -1) * reg).sum(1) / reg.sum(1)
    e_rest = (ep.reshape(len(D), -1) * ~reg).sum(1) / (~reg).sum(1)
    D_reg = (flatD * reg).sum(1) / reg.sum(1)
    MS = f("e_imp_own").mean((1, 2)) - EP                                    # 채점과 같은 차이 (불가능 영상은 자기 예측)
    return {"n": int(len(EP)), "acc": float(np.mean((M > 0) + 0.5 * (M == 0)) * 100),
            "acc_score": float(np.mean((MS > 0) + 0.5 * (MS == 0)) * 100), "n_ctx_mismatch": int((d["ctx_pdiff"][sel] > 0).sum()),
            "e_pos": float(EP.mean()), "D": float(DD.mean()), "m": float(M.mean()),
            "D_over_e": float(np.mean(DD / EP)), "m_over_e": float(np.mean(M / EP)) * 100, "m_over_D": float(np.mean(M) / np.mean(DD)),
            "m_over_D_median": float(np.median(M / DD)),
            "region": {"D_reg_over_e_reg": float(np.mean(D_reg / e_reg)), "e_reg": float(e_reg.mean()), "e_rest": float(e_rest.mean()),
                       "D_reg": float(D_reg.mean()), "m_reg_over_D_reg": float(np.mean((m_tok * reg).sum(1) / reg.sum(1)) / D_reg.mean())}}


def report(models, benches):
    R = {}
    for m in models:
        for b in benches:
            fn = f"{OUT}/{b}_{m}.npz"
            if not os.path.exists(fn):
                continue
            d = np.load(fn)
            r = {"verify": verify(m, b)}
            for ab in ("A", "B"):
                r[ab] = summarize(d, d["pair"] == ab, slice(2, None) if b == "synth" else slice(None))
                r[ab + "_allblocks"] = summarize(d, d["pair"] == ab)
            R[f"{b}/{m}"] = r
            v = r["verify"]
            print(f"\n== {b} · {m}  (검증: 놀람 상대오차 최대 {v['max_rel_err_pos']:.1e}/{v['max_rel_err_imp']:.1e}, 문맥 같은 짝의 p 최대차 {v['ctx_pdiff_max']:.2e})")
            for ab in ("A", "B"):
                s = r[ab + "_allblocks"]
                print(f"  {ab} n={s['n']:4d} 정답 {s['acc']:5.1f} (채점 {s['acc_score']:5.1f}, 문맥 다른 짝 {s['n_ctx_mismatch']}) | 예측오차 e⁺ {s['e_pos']:.4f} 반사실차 D {s['D']:.4f} 놀람차 m {s['m']:+.5f}"
                      f" | D/e⁺ {s['D_over_e']:.3f}  m/e⁺ {s['m_over_e']:+.2f}%  m/D {s['m_over_D']:+.3f} (중앙 {s['m_over_D_median']:+.3f})"
                      f" | 영역(D 상위10%) D/e {s['region']['D_reg_over_e_reg']:.3f} m/D {s['region']['m_reg_over_D_reg']:+.3f}")
    json.dump(R, open(f"{OUT}/report.json", "w"), indent=1)
    print("저장", f"{OUT}/report.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "report"])
    ap.add_argument("--model"); ap.add_argument("--models", nargs="+")
    ap.add_argument("--bench", choices=list(BENCH)); ap.add_argument("--benches", nargs="+", default=["synth", "real"])
    ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--bs", type=int, default=4); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.stage == "extract":
        extract(a.model, a.bench, a.gpu, a.bs, a.limit)
    else:
        report(a.models, a.benches)


if __name__ == "__main__":
    main()
