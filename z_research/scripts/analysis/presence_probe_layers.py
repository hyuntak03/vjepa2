#!/usr/bin/env python3
"""물체 존재 probe — predictor 층별 · 미래 블록별 (2026-11-10, 사용자 "representation 분석에서 깔끔하게 나올 것", "release VJEPA2 도 해봐야").

질문: 가려진 물체 정보가 predictor 안에는 있는데 예측 출력에서는 빠지는가 (그래서 음의 손실이 필요한가)?
데이터: VOBench v3 (전부 late 가림, 판 · 지나가는 가림막) 의 **가능 변이만** (CLAUDE.md 1-4): pos_a (물체 있음 — 문맥 끝에 가려졌다가 미래에 다시 나옴) vs
        pos_b (같은 장면, 물체 없음). 라벨 = 물체 있음 (1) / 없음 (0). 채점과 같은 창 (원본 0,3,…,93 → 32 장, 문맥 16 장 = 8 블록, 미래 8 블록).
표현 (미래 블록 k = 1..8 마다, 256 토큰의 평균 · 최대 풀링을 이어 붙임):
  L1..L12 = predictor 층 출력 (릴리즈: 한 번의 forward 에서 미래 토큰 자리 / AR: 문맥 끝 걸음과 굴림 걸음의 새 블록 토큰)
  out     = 예측 p (채점에 쓰는 출력, 1280)
  tgt     = 정답 h (target encoder, 채점과 같은 정답 공간) — encoder 대조: 실제로 그 블록에 물체가 보이는가
probe: 표준화 + L2 로지스틱 회귀, block 단위 2 겹 교차 (같은 장면의 pos_a · pos_b 는 같은 쪽), 정확도 = 2 겹 평균 (우연 50).
  /venv/vjepa/bin/python z_research/scripts/analysis/presence_probe_layers.py extract --model vith [--gpu 0]
  /venv/vjepa/bin/python z_research/scripts/analysis/presence_probe_layers.py probe --models vith vast_ar_vw_state ...
  /venv/vjepa/bin/python z_research/scripts/analysis/presence_probe_layers.py gain --models ...   # 물체 신호 이득 (아래)
gain (2026-11-10): 같은 장면 물체 있음 − 없음 차이를 예측 (Δp) 과 정답 (Δh) 에서 (출력 · 정답의 평균 풀링 1280 차원, 미래 블록마다).
  투영 이득 = <Δp, Δh> / |Δh|² (1 = 정답만큼, 0 = 차이 없음) · L1 비 = |Δp|₁ / |Δh|₁ · 물체 차이 / 예측 오차 = |Δh|₁ / |p − h|₁.
  ⚠️ CLAUDE.md "α (헤지 계수) 인용 금지" 와 같은 계열의 양이다 — 문서 · 논문에 인용하기 전에 사용자 확인.
transfer (2026-11-10, 사용자 "target encoder 로 빈 장면 vs 공 있는 장면 head 를 학습해 붙여서 봐야"): 미래 블록마다 정답 표현 (tgt) 에서
  공 있음 / 없음 head 를 학습 (표준화 + L2 로지스틱, 장면 단위 2 겹) → 그 head (같은 표준화 · 같은 문턱) 를 예측 (out) 에 그대로 붙인다.
  공 장면 → 공 판정 비율 · 빈 장면 → 빈 판정 비율 · AUC (문턱 무관) · head 점수 중앙값 · 빈 장면 확신 공 판정 (점수 > 3).
  /venv/vjepa/bin/python z_research/scripts/analysis/presence_probe_layers.py transfer --models ...
출력: z_analysis/results/presence_probe_20261110/{<model>.npz, probe.json, probe.md}
"""
import argparse, collections, contextlib, json, os, sys, time
import numpy as np
import torch
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
OUT = f"{REPO}/z_analysis/results/presence_probe_20261110"


def _blocks_module(m):
    for mod in m.modules():
        if hasattr(mod, "predictor_blocks"):
            return mod
    raise RuntimeError("predictor_blocks 를 못 찾았다")


def extract(model, gpu, bs):
    from analysis.intphys2.model import build_from_config
    from analysis.intphys2.surprise import _context_target_indices
    from evals.world_model_analysis.data import WMADataset
    cfg = yaml.safe_load(open(f"{REPO}/cvpr/results/eval/testbed/vobench_v3_{model}/_resolved.yaml"))
    dev = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(dev)
    ds = WMADataset(cfg)
    idx = [i for i, r in enumerate(ds.records) if r.raw.get("variant") in ("pos_a", "pos_b")]
    bundle = build_from_config(cfg["model"], dev)
    ts, S = bundle.tubelet_size, bundle.num_spatial_tokens
    ctx, N = int(cfg["surprise"]["context_length"]), ds.n_frames
    C, K = ctx // ts, (N - ctx) // ts
    pred = _blocks_module(bundle.predictor)
    is_ar = type(pred).__module__.endswith("rollout_predictor")
    store = {}

    def hook(li):
        def f(_m, _i, o):
            o = o[0] if isinstance(o, tuple) else o
            store.setdefault(li, []).append(o.detach())
        return f
    hs = [blk.register_forward_hook(hook(li)) for li, blk in enumerate(pred.predictor_blocks)]
    if is_ar:                               # AR 걸음은 blk.forward_kv 를 부른다 (forward hook 이 안 걸린다) → 인스턴스에서 감싼다
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
    feats = {f"L{l + 1}": [] for l in range(L)}
    feats.update(out=[], tgt=[])
    pool = lambda x: torch.cat([x.float().mean(2), x.float().amax(2)], -1).half().cpu()      # (B, K, S, D) → (B, K, 2D)
    t0 = time.time()
    for s in range(0, len(idx), bs):
        ch = idx[s:s + bs]
        x = torch.stack([ds.clip(i) for i in ch]).to(dev, bundle.dtype)
        n = len(ch)
        ci, ti = _context_target_indices(ctx_frames=ctx, tgt_frames=N - ctx, tubelet_size=ts, spatial_tokens=S, batch_size=n, device=dev)
        store.clear()
        with torch.no_grad(), ac:
            z = bundle.context_encoder(x, masks=[ci])
            p = bundle.predictor(z, ci, ti, mask_index=int(cfg["surprise"].get("mask_index", 0)))
            hf = bundle.target_encoder(x)
            h = torch.gather(hf, 1, ti.unsqueeze(-1).expand(-1, -1, hf.size(-1)))
            h = torch.nn.functional.layer_norm(h, (h.size(-1),))
        for l in range(L):
            calls = store[l]
            if is_ar:                       # 문맥 C 걸음 (마지막 = 블록 C+1 예측) + 굴림 K-1 걸음, 걸음마다 새 블록 S 토큰
                if len(calls) != C + K - 1 or calls[-1].shape[1] != S:
                    raise RuntimeError(f"AR 층 {l}: 호출 {len(calls)} 개 · 토큰 {calls[-1].shape} (기대 {C + K - 1} · {S})")
                y = torch.stack(calls[-K:], 1)                           # (B, K, S, D)
            else:                           # 한 번의 forward, 위치 정렬 → 미래 토큰 = 뒤 K*S
                if len(calls) != 1:
                    raise RuntimeError(f"릴리즈 층 {l}: 호출 {len(calls)} 개")
                y = calls[0][:, C * S:(C + K) * S].reshape(n, K, S, -1)
            feats[f"L{l + 1}"].append(pool(y))
        feats["out"].append(pool(p.reshape(n, K, S, -1)))
        feats["tgt"].append(pool(h.reshape(n, K, S, -1)))
        if (s // bs) % 20 == 0:
            print(f"[{model}] {s + n}/{len(idx)} {time.time() - t0:.0f}s", flush=True)
    for h_ in hs:
        h_.remove()
    recs = [ds.records[i] for i in idx]
    os.makedirs(OUT, exist_ok=True)
    np.savez(f"{OUT}/{model}.npz", **{k: torch.cat(v).numpy() for k, v in feats.items()},
             label=np.array([1 if r.raw["variant"] == "pos_a" else 0 for r in recs]),
             block=np.array([r.block_id for r in recs]), motion=np.array([r.raw.get("motion", "") for r in recs]),
             objset=np.array([r.raw.get("objset", "") for r in recs]), occ=np.array([r.raw.get("occluder_kind", "") for r in recs]),
             video=np.array([r.video_id for r in recs]))
    print(f"저장 {OUT}/{model}.npz · {len(idx)} 영상 · AR={is_ar} · {time.time() - t0:.0f}s")


def probe(models, subsets):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    res = {}
    for m in models:
        d = np.load(f"{OUT}/{m}.npz")
        y, blk, mo = d["label"], d["block"], d["motion"]
        ub = np.unique(blk); rng = np.random.default_rng(0); rng.shuffle(ub)
        fold = np.isin(blk, ub[: len(ub) // 2]).astype(int)
        srcs = [k for k in d.files if k.startswith("L")] + ["out", "tgt"]
        srcs = sorted([k for k in srcs if k.startswith("L")], key=lambda k: int(k[1:])) + ["out", "tgt"]
        res[m] = {}
        for sub in subsets:
            sel = np.ones(len(y), bool) if sub == "all" else (mo != "flat_static") if sub == "moving" else (mo == "flat_static")
            res[m][sub] = {}
            for src in srcs:
                X = d[src].astype(np.float32)                          # (N, K, F)
                accs = []
                for k in range(X.shape[1]):
                    a = []
                    for f in (0, 1):
                        tr, te = sel & (fold == f), sel & (fold != f)
                        sc = StandardScaler().fit(X[tr, k])
                        clf = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(X[tr, k]), y[tr])
                        a.append(float((clf.predict(sc.transform(X[te, k])) == y[te]).mean()))
                    accs.append(100 * float(np.mean(a)))
                res[m][sub][src] = accs
            print(m, sub, " ".join(f"{s}:{np.mean(v):.1f}" for s, v in res[m][sub].items()), flush=True)
    json.dump(res, open(f"{OUT}/probe.json", "w"), indent=1)
    lines = []
    for sub in subsets:
        lines.append(f"\n### {sub} — 미래 8 블록 평균 정확도 (%), 우연 50\n")
        srcs = list(next(iter(res.values()))[sub].keys())
        lines.append("| 모델 | " + " | ".join(srcs) + " |\n|---|" + "---|" * len(srcs))
        for m in models:
            lines.append(f"| {m} | " + " | ".join(f"{np.mean(res[m][sub][s]):.1f}" for s in srcs) + " |")
        lines.append(f"\n미래 블록별 (out · tgt · 최고 은닉층)\n")
        for m in models:
            r = res[m][sub]; best = max([s for s in r if s.startswith("L")], key=lambda s: np.mean(r[s]))
            lines.append(f"- {m}: out {' '.join(f'{v:.0f}' for v in r['out'])} | tgt {' '.join(f'{v:.0f}' for v in r['tgt'])} | {best} {' '.join(f'{v:.0f}' for v in r[best])}")
    open(f"{OUT}/probe.md", "w").write("\n".join(lines))
    print("\n".join(lines))


def gain(models):
    res = {}
    for m in models:
        d = np.load(f"{OUT}/{m}.npz"); y, blk, mo = d["label"], d["block"], d["motion"]
        D = d["out"].shape[-1] // 2
        P, H = d["out"][:, :, :D].astype(np.float32), d["tgt"][:, :, :D].astype(np.float32)
        ia = {b: i for i, (b, l) in enumerate(zip(blk, y)) if l == 1}; ib = {b: i for i, (b, l) in enumerate(zip(blk, y)) if l == 0}
        bs = [b for b in ia if b in ib]; A = np.array([ia[b] for b in bs]); B = np.array([ib[b] for b in bs])
        mov = np.array([mo[ia[b]] != "flat_static" for b in bs])
        dP, dH = P[A] - P[B], H[A] - H[B]
        g = (dP * dH).sum(-1) / np.maximum((dH * dH).sum(-1), 1e-9); r = np.abs(dP).sum(-1) / np.maximum(np.abs(dH).sum(-1), 1e-9)
        res[m] = {nm: {"proj_gain": np.median(g[s_], 0).round(3).tolist(), "l1_ratio": np.median(r[s_], 0).round(3).tolist(), "n": int(s_.sum())}
                  for nm, s_ in (("all", np.ones(len(bs), bool)), ("moving", mov), ("static", ~mov))}
        print(m, {k: v["proj_gain"] for k, v in res[m].items()})
    json.dump(res, open(f"{OUT}/gain.json", "w"), indent=1)


def transfer(models):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    res = {}
    for m in models:
        d = np.load(f"{OUT}/{m}.npz"); y, blk = d["label"], d["block"]
        ub = np.unique(blk); rng = np.random.default_rng(0); rng.shuffle(ub)
        fold = np.isin(blk, ub[: len(ub) // 2]).astype(int)
        H, P = d["tgt"].astype(np.float32), d["out"].astype(np.float32)
        r = {k: [] for k in ("self_acc", "ball_as_ball", "empty_as_empty", "auc", "score_tgt_ball", "score_tgt_empty",
                             "score_pred_ball", "score_pred_empty", "empty_confident_ball")}
        for k in range(H.shape[1]):
            acc = collections.defaultdict(list)
            for f in (0, 1):
                tr, te = fold == f, fold != f
                sc = StandardScaler().fit(H[tr, k]); clf = LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(H[tr, k]), y[tr])
                sh, sp = clf.decision_function(sc.transform(H[te, k])), clf.decision_function(sc.transform(P[te, k]))
                yt = y[te]
                acc["self_acc"].append(((sh > 0) == yt).mean()); acc["ball_as_ball"].append((sp[yt == 1] > 0).mean())
                acc["empty_as_empty"].append((sp[yt == 0] <= 0).mean()); acc["auc"].append(roc_auc_score(yt, sp))
                acc["score_tgt_ball"].append(np.median(sh[yt == 1])); acc["score_tgt_empty"].append(np.median(sh[yt == 0]))
                acc["score_pred_ball"].append(np.median(sp[yt == 1])); acc["score_pred_empty"].append(np.median(sp[yt == 0]))
                acc["empty_confident_ball"].append((sp[yt == 0] > 3).mean())
            for key, v in acc.items():
                sc_ = 100 if key in ("self_acc", "ball_as_ball", "empty_as_empty", "auc", "empty_confident_ball") else 1
                r[key].append(round(sc_ * float(np.mean(v)), 2))
        res[m] = r
        late = lambda key: float(np.mean(r[key][2:]))
        print(f"{m:44s} 블록 3–8: 공→공 {late('ball_as_ball'):5.1f} · 빈→빈 {late('empty_as_empty'):5.1f} · AUC {late('auc'):5.1f} · "
              f"점수 예측 공 {late('score_pred_ball'):+.1f} / 빈 {late('score_pred_empty'):+.1f} (정답 {late('score_tgt_ball'):+.1f} / {late('score_tgt_empty'):+.1f})", flush=True)
    json.dump(res, open(f"{OUT}/transfer_probe.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "probe", "gain", "transfer"])
    ap.add_argument("--model"); ap.add_argument("--models", nargs="+")
    ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--subsets", nargs="+", default=["all", "moving", "static"])
    a = ap.parse_args()
    if a.stage == "extract":
        extract(a.model, a.gpu, a.bs)
    elif a.stage == "gain":
        gain(a.models)
    elif a.stage == "transfer":
        transfer(a.models)
    else:
        probe(a.models, a.subsets)


if __name__ == "__main__":
    main()
