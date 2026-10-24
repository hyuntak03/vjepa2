#!/usr/bin/env python3
"""[P] 방법 B 상한 (학습 없음) — VOBench 끝 가림 오라클: 굴린 예측에 "정답 변위로 옮긴 마지막 보임 블록 물체" 를 도착 칸에 붙인다 (auto_research, 2026-10-21).

  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=5 /venv/vjepa/bin/python -W ignore auto_research/scripts/oracle_vob.py

채점 구성 · 물체 자리는 vob_late_mean.py 와 같다 (문맥 8 블록 → 굴린 예측 8 블록, 쌍 (pos_a, imp_ab) · (pos_b, imp_ba), 정답 = surprise(imp) > surprise(pos)).
오라클 (문맥 arm 의 물체 = 방향 A 면 A, B 면 B):
  원천 = 그 물체가 마지막으로 보인 문맥 블록 (가림 < 0.1). 미래 블록 j 의 물체 자리 = ① 라벨 미래 위치 (가능 · 불가능 arm 이 같은 경로라 정체는 안 샌다)
  ② 문맥에서 보인 마지막 두 블록 위치로 등속 외삽. 도착 칸 m = 그 블록에서 물체가 보이면 (라벨 가림 < 0.5) 원 덮임 ≥ 0.5 칸.
  운반층 = 원천 블록 토큰을 (원천 위치 − 미래 위치) 만큼 옮긴 것 (칸 반올림 / 쌍선형). 오라클 = m ⊙ 운반층 + (1 − m) ⊙ 기준 AR 굴린 예측."""
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
ap.add_argument("--out", default=os.path.join(REPO, "auto_research/results/oracle_vob.json"))
a = ap.parse_args()
dev = torch.device("cuda")
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
S, C, K, G, L = 256, 8, 8, 16, 8
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
IDX = list(range(0, 96, 3))[:32]
tenc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(tenc, _load_checkpoint(CK)["target_encoder"], tag="t"); tenc = tenc.to(dev).eval()
pc = yaml.safe_load(open(os.path.join(REPO, "z_training/runs/natural_ar_vw_state/config.yaml")))["model"]["predictor"]
pred = vit_prefix_predictor(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384, depth=12,
                            num_heads=12, num_mask_tokens=10, mask_mode="ar", ar_feedback=pc.get("ar_feedback", "output"))
pred.load_state_dict({k: v for k, v in _clean_backbone_key(torch.load(os.path.join(REPO, "z_training/runs/natural_ar_vw_state/latest.pt"), map_location="cpu", weights_only=False)["predictor"]).items()
                      if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}); pred = pred.to(dev).eval()
gy, gx = torch.meshgrid(torch.arange(G, device=dev).float(), torch.arange(G, device=dev).float(), indexing="ij")
gy, gx = gy.reshape(S), gx.reshape(S)


def warp(hsrc, dyx, mode):
    py, px = gy + dyx[0], gx + dyx[1]
    if mode == "round":
        ry, rx = torch.round(py).long(), torch.round(px).long()
        ok = (ry >= 0) & (ry < G) & (rx >= 0) & (rx < G)
        out = hsrc[ry.clamp(0, G - 1) * G + rx.clamp(0, G - 1)]; out[~ok] = hsrc[~ok]
        return out
    y0, x0 = torch.floor(py), torch.floor(px); wy, wx = py - y0, px - x0
    out = torch.zeros_like(hsrc); ws = torch.zeros(S, device=dev)
    for dy, wyy in ((0, 1 - wy), (1, wy)):
        for dx, wxx in ((0, 1 - wx), (1, wx)):
            yy, xx = (y0 + dy).long(), (x0 + dx).long(); ok = (yy >= 0) & (yy < G) & (xx >= 0) & (xx < G)
            w = wyy * wxx * ok; out += w[:, None] * hsrc[yy.clamp(0, G - 1) * G + xx.clamp(0, G - 1)]; ws += w
    return torch.where(ws[:, None] > 1e-6, out / ws.clamp_min(1e-6)[:, None], hsrc)


def disk_cover(u, v, r):                                                         # 512 px 좌표 → 256 화면 칸 덮임
    sub = (np.arange(4) + 0.5) / 4 * 16; cy_, cx_ = np.divmod(np.arange(S), G)
    px = cx_[:, None] * 16 + sub[None]; py = cy_[:, None] * 16 + sub[None]
    inside = ((px[:, None, :] - u / 2) ** 2 + (py[:, :, None] - v / 2) ** 2) <= (r / 2) ** 2
    return inside.mean((1, 2))

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
PRED = ["base", "or_label_round", "or_label_bilin", "or_cv_round", "or_cv_bilin"]
res = []; t0 = time.time()
for bi, (b, arms) in enumerate(sorted(blk.items())):
    d0 = os.path.join(a.vobench, arms["pos_a"]["file_name"])
    Hf, W0 = {}, None
    for v in ("pos_a", "pos_b"):
        x, W0 = frames(os.path.join(a.vobench, arms[v]["file_name"]), IDX); Hf[v] = blocks(x)
    for v in ("imp_ab", "imp_ba"):
        x, _ = frames(os.path.join(a.vobench, arms[v]["file_name"]), IDX[2 * C:]); Hf[v] = blocks(x)
    cen = centers(os.path.join(os.path.dirname(d0), "label.json"), W0)
    LB = json.load(open(os.path.join(os.path.dirname(d0), "label.json")))["labels"]
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
    for pos, imp, key in (("pos_a", "imp_ab", "A"), ("pos_b", "imp_ba", "B")):
        h = Hf[pos]; ctx = h[:C]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            p_ar = pred.rollout_kv(ctx.reshape(1, C * S, -1), torch.arange(C * S, device=dev).unsqueeze(0), K).float()[0].view(K, S, -1)
        lab = {q["f"]: q for q in LB.get(key, [])} or {q["f"]: q for q in LB.get("A", [])}
        q_at = lambda bb: lab.get(IDX[2 * bb + 1])                                  # 블록 bb 의 둘째 장 라벨
        vis_ctx = [bb for bb in range(C) if q_at(bb) is not None and q_at(bb)["hidden"] < 0.1]
        P = {"base": p_ar}
        if vis_ctx:
            bs = vis_ctx[-1]; qs = q_at(bs)
            prev = [bb for bb in vis_ctx if bb < bs]
            vel = None
            if prev:
                qp = q_at(prev[-1]); vel = (np.array([qs["u"] - qp["u"], qs["v"] - qp["v"]]) / (bs - prev[-1]))
            for src_kind in ("label", "cv"):
                for mode in ("round", "bilin"):
                    o = p_ar.clone()
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
                        m = torch.from_numpy(disk_cover(tu, tv, qf["r_px"]) >= 0.5).to(dev)
                        if not m.any():
                            continue
                        dyx = torch.tensor([(qs["v"] - tv) / 32, (qs["u"] - tu) / 32], device=dev, dtype=torch.float32)   # 512 px → 칸 (32 px)
                        o[j][m] = warp(h[bs], dyx, mode)[m]
                    P[f"or_{src_kind}_{mode}"] = o
        fut = {pos: h[C:], imp: Hf[imp]}
        rec = {"block": b, "dir": key, "mv": mv, "k": int(arms[pos]["k"]), "viol": arms[pos]["violation"], "motion": arms[pos]["motion"]}
        for n in PRED:
            pf = P.get(n, p_ar).reshape(K * S, -1)
            sc = {}
            for v in (pos, imp):
                e = (pf - fut[v].reshape(K * S, -1)).abs().mean(-1)
                sc[v] = (float(e.mean()), float(e[oidx].mean()) if len(oidx) else float("nan"))
            rec[n] = {"full": sc[imp][0] > sc[pos][0], "obj": sc[imp][1] > sc[pos][1]}
        res.append(rec)
    if bi % 20 == 0:
        print(f"{bi}/{len(blk)} ({time.time() - t0:.0f}s)", flush=True)
out = {}
print(f"\n== VOBench 끝 가림 — 물체 자리 / 화면 전체 쌍 정확도 (%), 쌍 {len(res)}")
for mv in ("static", "moving"):
    for k in (None, 1, 2, 3):
        sel = [r for r in res if r["mv"] == mv and (k is None or r["k"] == k)]
        if not sel:
            continue
        e = {n: (100 * np.mean([r[n]["obj"] for r in sel]), 100 * np.mean([r[n]["full"] for r in sel])) for n in PRED}
        out[f"{mv}/k{k or 'all'}"] = dict(n=len(sel), **{n: dict(obj=v[0], full=v[1]) for n, v in e.items()})
        print(f"  {mv:7s} k{k or '*'} n{len(sel):3d}  " + " | ".join(f"{n} {v[0]:5.1f}/{v[1]:5.1f}" for n, v in e.items()))
for mv in ("moving",):
    for vi in ("permanence", "shape", "color"):
        sel = [r for r in res if r["mv"] == mv and r["viol"] == vi]
        e = {n: 100 * np.mean([r[n]["obj"] for r in sel]) for n in PRED}
        print(f"  {mv} · {vi:10s} n{len(sel):3d} 물체 자리  " + " · ".join(f"{n} {v:5.1f}" for n, v in e.items()))
json.dump(out, open(a.out, "w"), indent=1)
