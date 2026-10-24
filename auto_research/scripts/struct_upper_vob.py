#!/usr/bin/env python3
"""[물리] 구조 상한 (사전 등록 KST 04:25, 학습 없음): 자연 영상 L1 의 조건부 최적 (안정도 구간별 a, b) 을 완벽히 쓰는 예측기의 VOBench 끝 가림 점수.

  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 /venv/vjepa/bin/python -W ignore auto_research/scripts/struct_upper_vob.py

[P] oracle_vob.py 와 같은 구성 (문맥 8 블록 → 기준 AR 굴린 예측 8 블록, 쌍 (pos_a, imp_ab) · (pos_b, imp_ba), 물체 자리 = 라벨 중심 둘레 3×3 칸,
칸 반올림 이동, 도착 칸 = 원 덮임 ≥ 0.5). 토큰은 VOBench 공유 캐시 (불가능 팔 문맥 = 짝 가능 팔).
예측 (물체가 보이는 미래 블록의 도착 칸만 바꿈, 나머지는 기준 AR):
  kappa_s  : a(s)·(같은 칸 문맥 5 블록 평균) + b(s)·(마지막 보임 문맥 블록 토큰을 미래 자리로 옮긴 것). s = 물체 중심 칸 토큰의 마지막 두 보임 문맥 블록 cos → 자연 영상 5 구간
  kappa_021: a 0.604 · κ 0.212 (지금 AR 의 모델 b)
  kappa_1  : a 0 · κ 1 ([P] 오라클)
미래 자리: label (라벨 위치) · cv (문맥 끝 두 보임 위치 등속 외삽)."""
import collections, csv, json, os, sys, time
import numpy as np, torch, yaml

REPO = "/workspace/world_model/vjepa2"
sys.path.insert(0, REPO)
from analysis.intphys2.model import _clean_backbone_key                                  # noqa: E402
from src.models.rollout_predictor import vit_prefix_predictor                          # noqa: E402
torch.set_num_threads(8)
dev = torch.device("cuda")
VB = "/workspace/dataset/benchmarks/VOBench_v1"; CACHE = "/workspace/auto_research_cache/vobench_late"
S, C, K, G = 256, 8, 8, 16
IDX = list(range(0, 96, 3))[:32]
ARMS = ["pos_a", "pos_b", "imp_ab", "imp_ba"]
AB = json.load(open(os.path.join(REPO, "auto_research/results/fit_ab_bins.json")))
EDGES = np.array(AB["edges"])
pc = yaml.safe_load(open(os.path.join(REPO, "z_training/runs/natural_ar_vw_state/config.yaml")))["model"]["predictor"]
pred = vit_prefix_predictor(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384, depth=12,
                            num_heads=12, num_mask_tokens=10, mask_mode="ar", ar_feedback=pc.get("ar_feedback", "output"))
pred.load_state_dict({k: v for k, v in _clean_backbone_key(torch.load(os.path.join(REPO, "z_training/runs/natural_ar_vw_state/latest.pt"), map_location="cpu", weights_only=False)["predictor"]).items()
                      if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}); pred = pred.to(dev).eval()
gy, gx = torch.meshgrid(torch.arange(G, device=dev).float(), torch.arange(G, device=dev).float(), indexing="ij")
gy, gx = gy.reshape(S), gx.reshape(S)


def warp_round(hsrc, dyx):
    py, px = gy + dyx[0], gx + dyx[1]
    ry, rx = torch.round(py).long(), torch.round(px).long()
    ok = (ry >= 0) & (ry < G) & (rx >= 0) & (rx < G)
    out = hsrc[ry.clamp(0, G - 1) * G + rx.clamp(0, G - 1)]; out[~ok] = hsrc[~ok]
    return out


def disk_cover(u, v, r):                                                         # 512 px 좌표 → 256 화면 칸 덮임
    sub = (np.arange(4) + 0.5) / 4 * 16; cy_, cx_ = np.divmod(np.arange(S), G)
    px = cx_[:, None] * 16 + sub[None]; py = cy_[:, None] * 16 + sub[None]
    inside = ((px[:, None, :] - u / 2) ** 2 + (py[:, :, None] - v / 2) ** 2) <= (r / 2) ** 2
    return inside.mean((1, 2))


def cell(q):
    return int(np.clip(q["v"] / 32, 0, G - 1)) * G + int(np.clip(q["u"] / 32, 0, G - 1))


def region(q, rad=1):
    r0, c0 = int(np.clip(q["v"] / 32, 0, G - 1)), int(np.clip(q["u"] / 32, 0, G - 1))
    return [rr * G + cc for rr in range(max(0, r0 - rad), min(G, r0 + rad + 1)) for cc in range(max(0, c0 - rad), min(G, c0 + rad + 1))]


meta = json.load(open(os.path.join(CACHE, "meta.json"))); H = np.load(os.path.join(CACHE, "h.npy"), mmap_mode="r")
PRED = ["base", "kappa_s_label", "kappa_s_cv", "kappa_021_label", "kappa_021_cv", "kappa_1_label", "kappa_1_cv"]
res, svals, t0 = [], [], time.time()
for bi, m in enumerate(meta):
    hh = torch.from_numpy(np.asarray(H[m["i"]], dtype=np.float32)).to(dev)           # (4,16,256,D)
    d0 = os.path.join(VB, m["files"]["pos_a"])
    LB = json.load(open(os.path.join(os.path.dirname(d0), "label.json")))["labels"]
    # 물체 자리 (채점용): 미래 블록마다 A · B 중심 둘레 3×3 (두 장 중 둘째 장 라벨, oracle_vob 과 같은 자리)
    labk = {key: {q["f"]: q for q in LB.get(key, [])} for key in ("A", "B")}
    otok = []
    for j in range(K):
        s_ = set()
        for key in ("A", "B"):
            q = labk[key].get(IDX[2 * (C + j) + 1])
            if q is not None:
                s_ |= set(region(q, 1))
        otok.append(sorted(s_))
    oidx = torch.tensor([j * S + t for j, tt in enumerate(otok) for t in tt], device=dev)
    mv = "static" if m["motion"] == "flat_static" else "moving"
    for pos, imp, key in (("pos_a", "imp_ab", "A"), ("pos_b", "imp_ba", "B")):
        h = hh[ARMS.index(pos)]; ctx = h[:C]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            p_ar = pred.rollout_kv(ctx.reshape(1, C * S, -1), torch.arange(C * S, device=dev).unsqueeze(0), K).float()[0].view(K, S, -1)
        lab = labk[key] or labk["A"]
        q_at = lambda bb: lab.get(IDX[2 * bb + 1])
        vis_ctx = [bb for bb in range(C) if q_at(bb) is not None and q_at(bb)["hidden"] < 0.1]
        P = {"base": p_ar}
        mean5 = h[C - 5:C].mean(0)
        s_obj = None
        if vis_ctx:
            bs = vis_ctx[-1]; qs = q_at(bs)
            prev = [bb for bb in vis_ctx if bb < bs]
            vel = None
            if prev:
                qp = q_at(prev[-1]); vel = np.array([qs["u"] - qp["u"], qs["v"] - qp["v"]]) / (bs - prev[-1])
                s_obj = float(torch.nn.functional.cosine_similarity(h[bs][cell(qs)], h[prev[-1]][cell(qp)], dim=0))
            kb = int(np.digitize(s_obj, EDGES)) if s_obj is not None else 2
            coef = {"kappa_s": (AB["bins"][kb]["a"], AB["bins"][kb]["b"]), "kappa_021": (AB["all"][0], 0.2116), "kappa_1": (0.0, 1.0)}
            for src_kind in ("label", "cv"):
                outs = {nm: p_ar.clone() for nm in coef}
                for j in range(K):
                    qf = q_at(C + j)
                    if qf is None or qf["hidden"] >= 0.5:
                        continue
                    if src_kind == "label":
                        tu, tv = qf["u"], qf["v"]
                    else:
                        if vel is None:
                            continue
                        tu, tv = qs["u"] + vel[0] * (C + j - bs), qs["v"] + vel[1] * (C + j - bs)
                    mk = torch.from_numpy(disk_cover(tu, tv, qf["r_px"]) >= 0.5).to(dev)
                    if not mk.any():
                        continue
                    moved = warp_round(h[bs], torch.tensor([(qs["v"] - tv) / 32, (qs["u"] - tu) / 32], device=dev, dtype=torch.float32))
                    for nm, (aa, bb_) in coef.items():
                        outs[nm][j][mk] = aa * mean5[mk] + bb_ * moved[mk]
                for nm in coef:
                    P[f"{nm}_{src_kind}"] = outs[nm]
        svals.append(dict(mv=mv, s=s_obj))
        fut = {pos: h[C:], imp: hh[ARMS.index(imp)][C:]}
        rec = {"block": m["i"], "mv": mv, "viol": m["violation"], "s": s_obj}
        for n in PRED:
            pf = P.get(n, p_ar).reshape(K * S, -1)
            sc = {}
            for v in (pos, imp):
                e = (pf - fut[v].reshape(K * S, -1)).abs().mean(-1)
                sc[v] = (float(e.mean()), float(e[oidx].mean()) if len(oidx) else float("nan"))
            rec[n] = {"full": float(sc[imp][0] > sc[pos][0]), "obj": float(sc[imp][1] > sc[pos][1])}
        res.append(rec)
    if bi % 40 == 0:
        print(f"{bi}/{len(meta)} ({time.time() - t0:.0f}s)", flush=True)

import random
rng = random.Random(0)
def boot(sel, n, kind):
    by = collections.defaultdict(list)
    for r in sel:
        by[r["block"]].append(r[n][kind])
    bl = list(by); est = 100 * np.mean([x for b in bl for x in by[b]])
    bs_ = sorted(100 * np.mean([x for b in [bl[rng.randrange(len(bl))] for _ in bl] for x in by[b]]) for _ in range(2000))
    return round(float(est), 1), (round(bs_[50], 1), round(bs_[1949], 1))
out = {}
for mv in ("moving", "static"):
    sel = [r for r in res if r["mv"] == mv]
    out[mv] = {n: dict(obj=boot(sel, n, "obj"), full=boot(sel, n, "full")) for n in PRED}
    print(f"== {mv} (쌍 {len(sel)})")
    for n in PRED:
        print(f"   {n:16s} 물체 자리 {out[mv][n]['obj']} · 화면 전체 {out[mv][n]['full']}")
    for vi in ("permanence", "shape", "color"):
        sv = [r for r in sel if r["viol"] == vi]
        out[f"{mv}|{vi}"] = {n: round(100 * float(np.mean([r[n]["obj"] for r in sv])), 1) for n in PRED}
        print(f"   {vi:10s} 물체 자리 " + " · ".join(f"{n} {v}" for n, v in out[f"{mv}|{vi}"].items()))
sv = [x["s"] for x in svals if x["s"] is not None and x["mv"] == "moving"]
out["s_moving"] = dict(n=len(sv), median=float(np.median(sv)), bin_frac=np.bincount(np.digitize(sv, EDGES), minlength=5).tolist())
print("움직이는 물체 안정도 s: 중앙", round(out["s_moving"]["median"], 3), "· 자연 구간별 개수", out["s_moving"]["bin_frac"])
json.dump(dict(summary=out, rows=res), open(os.path.join(REPO, "auto_research/results/struct_upper_vob.json"), "w"), indent=1)
