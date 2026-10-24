#!/usr/bin/env python3
"""[P] VOBench 끝 가림 물체 자리 정확도 — 기준 AR · L1 이어 학습 · 다중 가설 (가설 중 최소 surprise) (auto_research, 2026-10-21).

  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=5 /venv/vjepa/bin/python -W ignore auto_research/scripts/vob_late_hyp.py [--motion flat_static]

채점 구성 · 물체 자리 정의는 vob_late_mean.py 와 같다. 다중 가설: 가설 k 마다 굴린 예측 (가설 부호를 관측 임베딩에 더함),
arm 의 surprise = min_k 물체 자리 L1 (가설 중 그 미래를 가장 잘 맞힌 것). 정답 = surprise(imp) > surprise(pos)."""
import argparse, csv, json, os, sys, collections, time
import numpy as np, torch, torch.nn.functional as F, yaml
from PIL import Image
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint, _clean_backbone_key   # noqa: E402
from src.models.rollout_predictor import vit_prefix_predictor                                                  # noqa: E402
torch.set_num_threads(8)
ap = argparse.ArgumentParser()
ap.add_argument("--vobench", default="/workspace/dataset/benchmarks/VOBench_v1")
ap.add_argument("--motion", default=None, help="flat_static 만 등")
ap.add_argument("--out", default=os.path.join(REPO, "auto_research/results/vob_late_hyp.json"))
ap.add_argument("--ft", nargs="*", default=["ft_l1", "ft_wta3"], help="비교할 이어 학습 체크포인트 이름 (가설 부호가 있으면 가설 중 최소)")
a = ap.parse_args()
dev = torch.device("cuda")
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
S, C, K, G, L = 256, 8, 8, 16, 8
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
IDX = list(range(0, 96, 3))[:32]
tenc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(tenc, _load_checkpoint(CK)["target_encoder"], tag="t"); tenc = tenc.to(dev).eval()
pc = yaml.safe_load(open(os.path.join(REPO, "z_training/runs/natural_ar_vw_state/config.yaml")))["model"]["predictor"]
build = lambda: vit_prefix_predictor(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384, depth=12,
                                     num_heads=12, num_mask_tokens=10, mask_mode="ar", ar_feedback=pc.get("ar_feedback", "output"))
MODELS, ZV = {}, {"v": None}
srcs = {"base": torch.load(os.path.join(REPO, "z_training/runs/natural_ar_vw_state/latest.pt"), map_location="cpu", weights_only=False)["predictor"]}
FTD = {n: torch.load(f"/workspace/auto_research_cache/ar_ft/{n}.pt", map_location="cpu", weights_only=False) for n in a.ft}
srcs.update({n: d["predictor"] for n, d in FTD.items()})
for n, sd in srcs.items():
    m = build(); m.load_state_dict({k: v for k, v in _clean_backbone_key(sd).items() if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}); m = m.to(dev).eval()
    orig = m._ar_embed_obs
    def patched(x, orig=orig):
        e = orig(x)
        return e if ZV["v"] is None else e + ZV["v"][:, None, :].to(e.dtype)
    m._ar_embed_obs = patched
    MODELS[n] = (m, FTD[n]["codes"].to(dev) if n in FTD and FTD[n].get("codes") is not None else None)


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



rows = [r for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv"))) if r["occlusion"] == "late" and (a.motion is None or r["motion"] == a.motion)]
blk = collections.defaultdict(dict)
for r in rows:
    blk[r["block_id"]][r["variant"]] = r
res = []
t0 = time.time()
for bi, (b, arms) in enumerate(sorted(blk.items())):
    d0 = os.path.join(a.vobench, arms["pos_a"]["file_name"])
    Hf, W0 = {}, None
    for v in ("pos_a", "pos_b"):
        x, W0 = frames(os.path.join(a.vobench, arms[v]["file_name"]), IDX); Hf[v] = blocks(x)
    for v in ("imp_ab", "imp_ba"):
        x, _ = frames(os.path.join(a.vobench, arms[v]["file_name"]), IDX[2 * C:]); Hf[v] = blocks(x)
    cen = centers(os.path.join(os.path.dirname(d0), "label.json"), W0)
    otok = []
    for j in range(K):
        s_ = set()
        for key in ("A", "B"):
            c_ = cen[key][C + j]
            if c_ is not None:
                s_ |= set(region(c_, 1))
        otok.append(sorted(s_))
    oidx = torch.tensor([j * S + t for j, tt in enumerate(otok) for t in tt], device=dev)
    mv = "static" if arms["pos_a"]["motion"] == "flat_static" else "moving"
    for pos, imp, dirn in (("pos_a", "imp_ab", "A"), ("pos_b", "imp_ba", "B")):
        ctx = Hf[pos][:C]
        rec = {"block": b, "dir": dirn, "mv": mv, "k": int(arms[pos]["k"]), "viol": arms[pos]["violation"]}
        for n, (m, codes) in MODELS.items():
            ks = [None] if codes is None else list(range(codes.shape[0]))
            sc = {pos: [], imp: []}
            for kk in ks:
                ZV["v"] = None if kk is None else codes[kk][None]
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    p = m.rollout_kv(ctx.reshape(1, C * S, -1), torch.arange(C * S, device=dev).unsqueeze(0), K).float()[0].reshape(K * S, -1)
                ZV["v"] = None
                fut = {pos: Hf[pos][C:], imp: Hf[imp]}
                for v in (pos, imp):
                    sc[v].append(float((p[oidx] - fut[v].reshape(K * S, -1)[oidx]).abs().mean()))
            rec[n] = sc[imp][0] > sc[pos][0] if len(ks) == 1 else min(sc[imp]) > min(sc[pos])
            if len(ks) > 1:
                rec[n + "_per_hyp"] = [sc[imp][i] > sc[pos][i] for i in range(len(ks))]
        res.append(rec)
    if bi % 20 == 0:
        print(f"{bi}/{len(blk)} ({time.time() - t0:.0f}s)", flush=True)
out = {}
print(f"\n== VOBench 끝 가림 물체 자리 정확도 (%) — 쌍 {len(res)}")
for mv in ("static", "moving"):
    for k in (None, 1, 2, 3):
        sel = [r for r in res if r["mv"] == mv and (k is None or r["k"] == k)]
        if not sel:
            continue
        e = {n: 100 * np.mean([r[n] for r in sel]) for n in MODELS}
        out[f"{mv}/k{k or 'all'}"] = dict(n=len(sel), **e)
        print(f"  {mv:7s} k{k or '*'} n{len(sel):3d}  " + " · ".join(f"{n} {v:5.1f}" for n, v in e.items()))
json.dump(out, open(a.out, "w"), indent=1)
