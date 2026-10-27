"""미래 집합 채점 — "가장 가까운 그럴듯한 미래" 로 채점하면 결정론적 벤치 (VOBench 끝 가림) 에서 어떻게 되나 (2026-10-25, 학습 없음).
사용자: "intuitive physics 를 따르는 미래들의 집합을 만들면 되지 않을까? 근데 VOBench, IntPhys1 같은건 결정론적이니까 이것도 잘 해야지".

영상 v 의 집합 점수 = min_k L1(가설 k 의 굴린 미래, v 의 정답 미래). 쌍은 s(imp) > s(pos) 면 정답 (vob_compare 와 같은 칸: 문맥 8 · 굴림 8, 화면 전체).
집합 (모두 같은 문맥에서 굴림):
  base          기준 AR 하나 (natural_ar_vw_state latest) — 기준선
  wta3_k        다중 가설 3 (auto_research ft_wta3: 기준 AR 을 자연 영상 2000 걸음 승자독식 이어 학습, 가설 = 입력 부호 벡터) 하나씩 · min · mean
  lam_z0        잠재 행동 AR (natural_ar_vw_lam e70) z = 0
  lam_persist   〃 미래 걸음 z = 마지막 관측 전이의 μ
  lam_ctx7      〃 관측 전이 7 개의 μ 를 하나씩 미래 내내 유지 → 7 가설 min (문맥이 보여 준 움직임의 집합)
  lam_prior8    〃 미래 걸음 z ~ N(0, I) (걸음마다 독립) 8 표본 min (모델의 사전 분포 = 제약 없는 집합)
  base+copy     {기준 AR, 멈춘 세계 (마지막 문맥 블록 복사)} min — "그럴듯해 보이는 일반 미래" 가 위반을 흡수하는지 보는 대조
가설 퍼짐: 가능 영상에서 max_k − min_k (가설끼리 얼마나 다른가) 를 기준 AR 의 |gap| 과 나란히.

  CUDA_VISIBLE_DEVICES=0 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/future_set_score.py
출력: auto_research/results/future_set_score.json
"""
import collections, csv, json, os, sys, time
import numpy as np, torch

REPO = "/workspace/world_model/vjepa2"
sys.path.insert(0, os.path.join(REPO, "auto_research/scripts")); sys.path.insert(0, REPO)
_argv = list(sys.argv); sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT  # noqa: E402
sys.argv = _argv
from analysis.intphys2.model import _clean_backbone_key  # noqa: E402
from app.vjepa_frozen import latent_action as LAM  # noqa: E402

torch.set_num_threads(8); torch.manual_seed(0)
dev = torch.device("cuda")
S, C, K, D = 256, 8, 8, 1280
NPRIOR = 8
OUT = os.path.join(REPO, "auto_research/results/future_set_score.json")

base = FT.load_pred("natural_ar_vw_state", "latest.pt").to(dev).eval()
# 다중 가설 3: 같은 구조에 승자독식 이어 학습 가중 + 가설 부호 (관측 임베딩에 더한다)
wta = FT.load_pred("natural_ar_vw_state", "latest.pt")
_ck = torch.load("/workspace/auto_research_cache/ar_ft/ft_wta3.pt", map_location="cpu", weights_only=False)
wta.load_state_dict({k: v for k, v in _clean_backbone_key(_ck["predictor"]).items() if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}, strict=True)
wta = wta.to(dev).eval()
CODES = _ck["codes"].to(dev); NH = CODES.shape[0]
_CODE = {"v": None}
_orig = wta._ar_embed_obs
wta._ar_embed_obs = lambda x, _o=_orig: (_o(x) if _CODE["v"] is None else _o(x) + _CODE["v"][:, None, :].to(_o(x).dtype))
# 잠재 행동
lam_pred = FT.load_pred("natural_ar_vw_lam", "latest.pt")
_ck = torch.load(f"{REPO}/z_training/runs/natural_ar_vw_lam/latest.pt", map_location="cpu", weights_only=False)
LW = LAM.LatentActionAR(lam_pred, _ck["latent_action_cfg"], D).to(dev).eval()
LW.idm.load_state_dict(_ck["latent_action"]["idm"]); LW.zmod.load_state_dict(_ck["latent_action"]["zmod"])
print("LAM epoch", _ck["epoch"], flush=True); del _ck


def roll(m, ctx):                                         # ctx (B, C·S, D) → (B, K, S, D)
    B = ctx.size(0)
    idx = torch.arange(C * S, device=dev).unsqueeze(0).expand(B, -1)
    return m.rollout_kv(ctx, idx, K).float().view(B, K, S, D)


def roll_lam(ctx, z_fut):
    """ctx (B, C·S, D), z_fut (B, K, z_dim): 문맥 블록 b ≤ C−2 는 관측 전이 μ_b, 블록 C−1 과 걸음 r 은 z_fut[:, r]."""
    B = ctx.size(0); P = LW.pred
    _, mu, _ = LW.infer(ctx, sample=False)               # (B, C−1, z)
    m = LW.mods(torch.cat([mu, z_fut], 1))               # 위치 0..C−2 관측, C−1..C+K−2 미래
    idx = torch.arange(C * S, device=dev).unsqueeze(0).expand(B, -1)
    caches = None; preds = []
    try:
        for b in range(C):
            LW._m = m[:, b: b + 1]
            p, caches, st = P._ar_step_kv(ctx[:, b * S:(b + 1) * S], idx[:, b * S:(b + 1) * S], caches)
        preds.append(p); pos = idx[:, (C - 1) * S:]
        for r in range(1, K):
            pos = pos + S; LW._m = m[:, C - 1 + r: C + r]
            p, caches, st = P._ar_step_kv(st + P.type_embed[1], pos, caches, embedded=True)
            preds.append(p)
    finally:
        LW._m = None
    return torch.cat(preds, 1).float().view(B, K, S, D), mu


def err(pred, y):                                         # pred (H, K, S, D), y (K, S, D) → (H,)
    return (pred - y[None]).abs().mean((1, 2, 3))


idx_rows = collections.defaultdict(dict)
for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv"))):
    if r["occlusion"] == "late":
        idx_rows[r["block_id"]][r["variant"]] = r
meta = json.load(open("/workspace/auto_research_cache/vobench_late/meta.json"))
H = np.load("/workspace/auto_research_cache/vobench_late/h.npy", mmap_mode="r")
ARMS = ["pos_a", "pos_b", "imp_ab", "imp_ba"]
recs = []; t0 = time.time()
for bi, mt in enumerate(meta):
    arms = idx_rows[mt["block_id"]]
    hh = {v: torch.from_numpy(np.asarray(H[mt["i"], k], dtype=np.float32)).to(dev) for k, v in enumerate(ARMS)}
    viol = arms["pos_a"]["violation"]; empty_b = arms["pos_a"].get("obj_b", "") == "empty"
    for pid, (pos, imp) in (("1", ("pos_a", "imp_ab")), ("2", ("pos_b", "imp_ba"))):
        ctx = hh[pos][:C].reshape(1, C * S, D)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hyp = {"base": roll(base, ctx)}
            ws = []
            for k in range(NH):
                _CODE["v"] = CODES[k: k + 1]; ws.append(roll(wta, ctx))
            _CODE["v"] = None
            hyp["wta3"] = torch.cat(ws, 0)
            hyp["lam_z0"] = roll(LW.pred, ctx)
            zero = torch.zeros(1, K, LW.z_dim, device=dev)
            _, mu = roll_lam(ctx, zero)                                  # μ 만 얻으려고 (출력은 아래에서 다시)
            hyp["lam_persist"], _ = roll_lam(ctx, mu[:, -1:].expand(1, K, -1).contiguous())
            hyp["lam_ctx7"], _ = roll_lam(ctx.expand(C - 1, -1, -1), mu[0][:, None, :].expand(-1, K, -1).contiguous())
            hyp["lam_prior8"], _ = roll_lam(ctx.expand(NPRIOR, -1, -1), torch.randn(NPRIOR, K, LW.z_dim, device=dev))
            hyp["copy"] = hh[pos][C - 1][None, None].expand(1, K, S, D)
        sc = {n: {v: err(p, hh[v][C:]).cpu().numpy() for v in (pos, imp)} for n, p in hyp.items()}
        direction = ("vanish" if pid == "1" else "appear") if (viol == "permanence" and empty_b) else viol
        recs.append(dict(block=mt["block_id"], pair=pid, viol=viol, dir=direction,
                         sc={n: {"pos": d[pos].tolist(), "imp": d[imp].tolist()} for n, d in sc.items()}))
    if bi % 20 == 0:
        print(f"{bi}/{len(meta)} {time.time() - t0:.0f}s", flush=True)


def score(rule):
    """rule(rec) → (s_pos, s_imp). 집합 · 방향별 정확도."""
    out = collections.defaultdict(list)
    for r in recs:
        sp, si = rule(r)
        out["all"].append(si > sp); out[r["dir"]].append(si > sp)
    return {g: [round(100 * float(np.mean(v)), 1), len(v)] for g, v in sorted(out.items())}


def mn(n, sl=slice(None)):
    return lambda r: (min(r["sc"][n]["pos"][sl]), min(r["sc"][n]["imp"][sl]))


RULES = {"base": mn("base"), "lam_z0": mn("lam_z0"), "lam_persist": mn("lam_persist"),
         "wta3_mean": lambda r: (float(np.mean(r["sc"]["wta3"]["pos"])), float(np.mean(r["sc"]["wta3"]["imp"]))),
         "wta3_min": mn("wta3"), "lam_ctx7_min": mn("lam_ctx7"), "lam_prior8_min": mn("lam_prior8"),
         "lam_prior8_mean": lambda r: (float(np.mean(r["sc"]["lam_prior8"]["pos"])), float(np.mean(r["sc"]["lam_prior8"]["imp"]))),
         "base+copy_min": lambda r: (min(r["sc"]["base"]["pos"][0], r["sc"]["copy"]["pos"][0]), min(r["sc"]["base"]["imp"][0], r["sc"]["copy"]["imp"][0])),
         "copy_only": mn("copy")}
for k in range(NH):
    RULES[f"wta3_h{k}"] = mn("wta3", slice(k, k + 1))
res = {"acc": {n: score(f) for n, f in RULES.items()}}
# 가설 퍼짐 (가능 영상) vs 기준 AR 의 쌍 gap 크기
spread = {}
for n in ("wta3", "lam_ctx7", "lam_prior8"):
    sp = [max(r["sc"][n]["pos"]) - min(r["sc"][n]["pos"]) for r in recs]
    spread[n] = round(float(np.median(sp)), 5)
spread["base_abs_gap"] = round(float(np.median([abs(r["sc"]["base"]["imp"][0] - r["sc"]["base"]["pos"][0]) for r in recs])), 5)
spread["base_pos_L1"] = round(float(np.median([r["sc"]["base"]["pos"][0] for r in recs])), 5)
res["median_spread_pos"] = spread
res["n_pairs"] = len(recs)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump({"res": res, "recs": recs}, open(OUT, "w"))
print(json.dumps(res, indent=1, ensure_ascii=False))
