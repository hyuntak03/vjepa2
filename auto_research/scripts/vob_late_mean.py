#!/usr/bin/env python3
"""[P] 고리 4 — VOBench late 에서 기준 블록 AR 의 "정지 물체 유지 · 움직이는 물체 소실" 이 같은 칸 문맥 평균으로 설명되나 (auto_research, 2026-10-21).

  CUDA_VISIBLE_DEVICES=5 /venv/vjepa/bin/python -W ignore auto_research/scripts/vob_late_mean.py \
      --run natural_ar_vw_state latest.pt --wjson auto_research/results/pred_mean/natural_ar_vw_state__latest.json

채점과 같은 구성 (obj_region_surprise.py 와 같은 함수): 30 fps PNG 3 장 간격 → 32 장 = 16 블록, 문맥 8 블록 → rollout 8 블록,
정답 = 블록별 LN(target_encoder), 쌍 (pos_a, imp_ab) · (pos_b, imp_ba), 정답 = surprise(imp) > surprise(pos).
surprise 를 셋으로: 물체 자리 (미래 블록마다 라벨 A · B 중심 3×3 합집합) · 먼 배경 (중심에서 체비셰프 > 3) · 화면 전체.
late 가림: k = 1 · 2 · 3 → 문맥 끝 k 장 + 미래 앞 k 장 가림 (k=2 면 문맥 마지막 블록 전체가 가림막).

예측기 (모델 없는 것은 문맥 블록만 쓴다):
  AR       기준 블록 AR 의 굴린 예측
  copy     마지막 문맥 블록 (late 에선 가림막이 섞인다)
  mean_n   최근 n 문맥 블록 평균 (n = 2 · 4 · 8) — 미래 8 블록 모두 같은 값
  LA_rec   같은 칸 재귀 선형 AR: x_{b+1} = Σ_k w_k x_{b−k}, w = phys3d 캐시에서 기준 블록 AR 의 TF 출력을 먼 칸에서 적합한 8 지연 가중치 (--wjson)
회귀: AR 굴린 예측의 물체 자리 토큰 (미래 블록 10..15, 물체가 다시 보이는 구간) ≈ Σ_k w_k h_{7−k} (같은 칸, 문맥 8 블록), 정지 / 움직임 따로.
     대조군 = 정답 미래 (pos arm) 를 같은 방식으로.
물체 점수: s(x) = x · T̂_A, T̂_A = 블록 0 의 A 물체 중심 3×3 평균 토큰 (정규화). 대비 = 물체 자리 평균 − 먼 배경 평균."""
import argparse, csv, json, os, sys, collections, time
import numpy as np, torch, torch.nn.functional as F, yaml
from PIL import Image
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint, _clean_backbone_key   # noqa: E402
from src.models.rollout_predictor import vit_prefix_predictor                                                  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", nargs=2, default=["natural_ar_vw_state", "latest.pt"])
ap.add_argument("--wjson", required=True)
ap.add_argument("--vobench", default="/workspace/dataset/benchmarks/VOBench_v1")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", default=os.path.join(REPO, "auto_research/results/vob_late_mean.json"))
a = ap.parse_args()
dev = torch.device("cuda")
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
S, C, K, G, L = 256, 8, 8, 16, 8
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
IDX = list(range(0, 96, 3))[:32]
tenc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(tenc, _load_checkpoint(CK)["target_encoder"], tag="t"); tenc = tenc.to(dev).eval()
W = json.load(open(a.wjson))["tf"]["mov/far"]["pred"]["M0_w"]
print("LA 가중치 (phys3d 먼 칸 적합):", np.round(W, 3).tolist())

D = os.path.join(REPO, "z_training/runs", a.run[0])
pc = yaml.safe_load(open(os.path.join(D, "config.yaml")))["model"]["predictor"]
pred = vit_prefix_predictor(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384, depth=12,
                            num_heads=12, num_mask_tokens=10, mask_mode="ar", ar_feedback=pc.get("ar_feedback", "output"))
sd = {k: v for k, v in _clean_backbone_key(torch.load(os.path.join(D, a.run[1]), map_location="cpu", weights_only=False)["predictor"]).items()
      if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}
pred.load_state_dict(sd, strict=True); pred = pred.to(dev).eval()


def frames(d, idx):
    fr = [np.asarray(Image.open(f"{d}/{i:06d}.png").convert("RGB")) for i in idx]
    x = torch.from_numpy(np.stack(fr)).permute(3, 0, 1, 2).float() / 255.0
    W0 = x.shape[-1]
    x = F.interpolate(x.transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False).transpose(0, 1)
    return (x - MEAN) / STD, W0


@torch.no_grad()
def blocks(x):
    nb = x.shape[1] // 2
    xb = x.reshape(3, nb, 2, 256, 256).permute(1, 0, 2, 3, 4).to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = torch.cat([tenc(xb[i:i + 8]) for i in range(0, nb, 8)], 0).float()
    return F.layer_norm(h, (1280,)).view(nb, S, -1)


def centers(label_path, W0):
    """블록 0..15 마다 A · B 중심 칸 (행, 열) 과 가림 비율 (두 장 평균). 없으면 None."""
    L_ = json.load(open(label_path))["labels"]; out = {}
    for key in ("A", "B"):
        lab = {q["f"]: q for q in L_.get(key, [])}; cs = []
        for b in range(16):
            q = [lab.get(IDX[2 * b]), lab.get(IDX[2 * b + 1])]
            if any(t is None for t in q):
                cs.append(None); continue
            r0 = int(np.clip(np.mean([t["v"] for t in q]) / W0 * G, 0, G - 1)); c0 = int(np.clip(np.mean([t["u"] for t in q]) / W0 * G, 0, G - 1))
            cs.append((r0, c0, float(np.mean([t["hidden"] for t in q]))))
        out[key] = cs
    return out


def region(rc, rad):
    r0, c0 = rc[:2]
    return [rr * G + cc for rr in range(max(0, r0 - rad), min(G, r0 + rad + 1)) for cc in range(max(0, c0 - rad), min(G, c0 + rad + 1))]


rows = [r for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv"))) if r["occlusion"] == "late"]
blk = collections.defaultdict(dict)
for r in rows:
    blk[r["block_id"]][r["variant"]] = r
items = sorted(blk.items())
if a.limit:
    items = items[: a.limit]
PRED = ["AR", "copy", "mean_2", "mean_4", "mean_8", "LA_rec"]
res = []
REG = {(mv, nm): [] for mv in ("static", "moving") for nm in ("AR", "true")}
PRES = []
t0 = time.time()
wt = torch.tensor(W, device=dev, dtype=torch.float32)
for bi, (b, arms) in enumerate(items):
    d0 = os.path.join(a.vobench, arms["pos_a"]["file_name"])
    Hf, W0 = {}, None
    for v in ("pos_a", "pos_b"):
        x, W0 = frames(os.path.join(a.vobench, arms[v]["file_name"]), IDX); Hf[v] = blocks(x)                  # 16 블록
    for v in ("imp_ab", "imp_ba"):
        x, _ = frames(os.path.join(a.vobench, arms[v]["file_name"]), IDX[2 * C:]); Hf[v] = blocks(x)          # 미래 8 블록만 (문맥은 pos 와 같다)
    cen = centers(os.path.join(os.path.dirname(d0), "label.json"), W0)
    otok, ftok = [], []
    for j in range(K):
        s_, near = set(), set()
        for key in ("A", "B"):
            c_ = cen[key][C + j]
            if c_ is None:
                continue
            s_ |= set(region(c_, 1)); near |= set(region(c_, 3))
        otok.append(sorted(s_)); ftok.append(sorted(set(range(S)) - near))
    oidx = torch.tensor([j * S + t for j, tt in enumerate(otok) for t in tt], device=dev)
    fidx = torch.tensor([j * S + t for j, tt in enumerate(ftok) for t in tt], device=dev)
    mv = "static" if arms["pos_a"]["motion"] == "flat_static" else "moving"
    for pos, imp, dirn, key in (("pos_a", "imp_ab", "A", "A"), ("pos_b", "imp_ba", "B", "B")):
        h = Hf[pos]
        ctx = h[:C]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            p_ar = pred.rollout_kv(ctx.reshape(1, C * S, -1), torch.arange(C * S, device=dev).unsqueeze(0), K).float()[0].view(K, S, -1)
        seq = [ctx[k] for k in range(C)]; la = []
        for j in range(K):
            nxt = sum(float(W[k]) * seq[-1 - k] for k in range(L)); la.append(nxt); seq.append(nxt)
        P = {"AR": p_ar, "copy": ctx[-1:].expand(K, S, -1), "mean_2": ctx[-2:].mean(0, keepdim=True).expand(K, S, -1),
             "mean_4": ctx[-4:].mean(0, keepdim=True).expand(K, S, -1), "mean_8": ctx.mean(0, keepdim=True).expand(K, S, -1), "LA_rec": torch.stack(la)}
        fut = {pos: h[C:], imp: Hf[imp]}
        rec = {"block": b, "dir": dirn, "objset": arms[pos]["objset"], "viol": arms[pos]["violation"], "motion": arms[pos]["motion"],
               "mv": mv, "k": int(arms[pos]["k"])}
        for nm, p in P.items():
            pf = p.reshape(K * S, -1)
            sc = {}
            for v in (pos, imp):
                e = (pf - fut[v].reshape(K * S, -1)).abs().mean(-1)
                sc[v] = (float(e.mean()), float(e[oidx].mean()) if len(oidx) else float("nan"), float(e[fidx].mean()) if len(fidx) else float("nan"))
            rec[nm] = {"full": sc[imp][0] > sc[pos][0], "obj": sc[imp][1] > sc[pos][1], "far": sc[imp][2] > sc[pos][2],
                       "m_obj": sc[imp][1] - sc[pos][1]}
        res.append(rec)
        # 회귀 + 물체 점수: 이 arm 의 물체 (key) 가 미래 블록 10..15 에서 보일 때, 그 중심 3×3
        if key == "A" or arms[pos]["violation"] != "permanence":
            c0 = cen[key][0]
            if c0 is not None and c0[2] < 0.1:
                T_hat = F.normalize(h[0][region(c0, 1)].mean(0), dim=-1)
                for j in range(2, K):
                    cj = cen[key][C + j]
                    if cj is None or cj[2] > 0.1:
                        continue
                    cells = torch.tensor(region(cj, 1), device=dev)
                    farc = torch.tensor(sorted(set(range(S)) - set(region(cj, 3))), device=dev)
                    X = torch.stack([ctx[C - 1 - k][cells] for k in range(L)])               # (8, n, D)
                    XtX = torch.einsum("knd,lnd->kl", X, X).cpu().numpy()
                    for nm, yv in (("AR", p_ar[j][cells]), ("true", h[C + j][cells])):
                        REG[(mv, nm)].append(dict(XtX=XtX, Xty=torch.einsum("knd,nd->k", X, yv).cpu().numpy(), yty=float((yv * yv).sum()),
                                                  k=rec["k"], block=b))
                    cand = torch.stack([p_ar[j], h[C + j], ctx[-1], ctx.mean(0), la[j]] + [ctx[C - 1 - k] for k in range(L)])   # (13, S, D)
                    s_ = cand @ T_hat
                    PRES.append(dict(mv=mv, k=rec["k"], j=j, con=(s_[:, cells].mean(1) - s_[:, farc].mean(1)).cpu().numpy().tolist()))
    if bi % 20 == 0:
        print(f"{bi}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)


def fit(recs):
    XtX = sum(r["XtX"] for r in recs).astype(np.float64); Xty = sum(r["Xty"] for r in recs).astype(np.float64); yty = sum(r["yty"] for r in recs)
    w = np.linalg.solve(XtX + 1e-6 * np.trace(XtX) / L * np.eye(L), Xty)
    return w, float(1 - (yty - 2 * w @ Xty + w @ XtX @ w) / yty)


out = {"n_pairs": len(res), "scores": {}, "reg": {}, "presence": {}}
print(f"\n== VOBench late 쌍 정확도 (%) — 물체 자리 / 먼 배경 / 화면 전체 [물체 자리 방향 A / B], 쌍 {len(res)}")
for mv in ("static", "moving"):
    for k in (None, 1, 2, 3):
        sel = [r for r in res if r["mv"] == mv and (k is None or r["k"] == k)]
        if not sel:
            continue
        line = []
        for nm in PRED:
            o = 100 * np.mean([r[nm]["obj"] for r in sel]); fa = 100 * np.mean([r[nm]["far"] for r in sel]); fu = 100 * np.mean([r[nm]["full"] for r in sel])
            oa = 100 * np.mean([r[nm]["obj"] for r in sel if r["dir"] == "A"]); ob = 100 * np.mean([r[nm]["obj"] for r in sel if r["dir"] == "B"])
            out["scores"][f"{mv}/k{k or 'all'}/{nm}"] = dict(n=len(sel), obj=o, far=fa, full=fu, objA=oa, objB=ob)
            line.append(f"{nm} {o:5.1f}/{fa:5.1f}/{fu:5.1f} [{oa:3.0f}/{ob:3.0f}]")
        print(f"  {mv:7s} k{k or '*'} n{len(sel):3d}  " + " | ".join(line))
print("\n== AR 굴린 예측 (미래 블록 10..15, 물체 자리) ≈ Σ w_k h_{7−k} (같은 칸) — 지연 k = 0..7 가중치")
for mv in ("static", "moving"):
    for k in (None, 1, 2, 3):
        for nm in ("AR", "true"):
            R_ = [r for r in REG[(mv, nm)] if k is None or r["k"] == k]
            if len(R_) < 5:
                continue
            w, r2 = fit(R_)
            out["reg"][f"{mv}/k{k or 'all'}/{nm}"] = dict(n=len(R_), w=w.tolist(), R2=r2, sum=float(w.sum()))
            print(f"  {mv:7s} k{k or '*'} {nm:4s} n{len(R_):4d} R² {r2:.3f} 합 {w.sum():.2f} w " + " ".join(f"{x:+.2f}" for x in w))
print("\n== 물체 대비 (A 물체 점수, 물체 자리 − 먼 배경), 미래 블록 10..15 평균: AR / 정답 / copy / mean_8 / LA_rec | 문맥 블록 7..0")
for mv in ("static", "moving"):
    for k in (None, 1, 2, 3):
        R_ = [r for r in PRES if r["mv"] == mv and (k is None or r["k"] == k)]
        if len(R_) < 5:
            continue
        c = np.mean([r["con"] for r in R_], 0)
        out["presence"][f"{mv}/k{k or 'all'}"] = dict(n=len(R_), con=c.tolist())
        print(f"  {mv:7s} k{k or '*'} n{len(R_):4d}  " + " / ".join(f"{x:.2f}" for x in c[:5]) + " | " + " ".join(f"{x:.2f}" for x in c[5:])
              + f"   → AR / 정답 = {c[0] / c[1]:.2f}, mean_8 / 정답 = {c[3] / c[1]:.2f}")
json.dump(out, open(a.out, "w"), indent=1)
json.dump(res, open(a.out.replace(".json", "_pairs.json"), "w"))
print("저장:", a.out)
