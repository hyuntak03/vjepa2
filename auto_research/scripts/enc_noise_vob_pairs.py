#!/usr/bin/env python3
"""고리 1 [E] — 비국소성의 독립 재현: 물체만 다른 두 미래 (VOBench) 의 정답 토큰 차이 중 물체 칸에 있는 몫 vs 면적 (auto_research, 2026-10-21).

  CUDA_VISIBLE_DEVICES=2 /venv/vjepa/bin/python -W ignore auto_research/scripts/enc_noise_vob_pairs.py

VOBench_v1, 가림 없음 (visible) block 의 문맥 일치 쌍 (pos_a ↔ imp_ab, pos_b ↔ imp_ba): 문맥 (장 0..47) 은 픽셀이 같고 미래는 물체만 다르다.
장 0..93 간격 3 → 32 장 (블록 16), 미래 = 블록 8..15. 512 → 256 bilinear (antialias 없음, 채점과 같음).
  블록 정답: 블록별 LN(target_encoder), bf16 (학습 정답)
  창 정답:   32 장 한 번에 target_encoder → LN → 미래 tubelet 8..15 (채점 정답과 같은 방식)
물체 영역: 그 블록의 두 arm 픽셀 차 (칸 평균 |Δ|) > 2 단계인 칸 (A) · 픽셀 차 질량 중심의 3×3 칸 (B, 외부 팀 정의와 맞춤).
몫 = 그 칸들의 |Δh| 합 / 화면 전체 |Δh| 합. 면적 = 칸 수 / 256. 픽셀 차의 같은 몫도 (대조)."""
import argparse, csv, json, os, sys
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/workspace/dataset/benchmarks/VOBench_v1")
ap.add_argument("--occ", default="visible")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "../results/enc_noise_vob_pairs.json"))
a = ap.parse_args()
dev = torch.device("cuda"); G, R = 16, 256
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
e = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=64, use_rope=True, uniform_power=False)
_load_state_dict(e, _load_checkpoint(CK)["target_encoder"], tag="t"); e = e.to(dev, torch.bfloat16).eval()


def frames(rel):
    fr = [np.asarray(Image.open(os.path.join(a.root, rel, f"{3 * k:06d}.png")).convert("RGB")) for k in range(32)]
    x = torch.from_numpy(np.stack(fr)).to(dev).permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(R, R), mode="bilinear", align_corners=False)
    return x.round().clamp(0, 255).permute(0, 2, 3, 1)                        # (32,R,R,3) float 0..255


@torch.no_grad()
def enc(x):                                                                   # x (B,T,R,R,3) 0..255
    z = (x.div(255).permute(0, 4, 1, 2, 3) - MEAN) / STD
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = e(z.to(torch.bfloat16)).float()
    return F.layer_norm(h, (h.size(-1),))


rows = [r for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv"))) if r["occlusion"] == a.occ]
by = {}
for r in rows:
    by.setdefault(r["source_block"], {})[r["variant"]] = r
pairs = [(v["pos_a"], v["imp_ab"]) for v in by.values() if "pos_a" in v and "imp_ab" in v] + \
        [(v["pos_b"], v["imp_ba"]) for v in by.values() if "pos_b" in v and "imp_ba" in v]
if a.limit:
    pairs = pairs[: a.limit]
REC = []
for i, (ra, rb) in enumerate(pairs):
    xa, xb = frames(ra["file_name"]), frames(rb["file_name"])
    ctx_same = float((xa[:16] - xb[:16]).abs().max())
    ha = enc(xa.view(16, 2, R, R, 3)); hb = enc(xb.view(16, 2, R, R, 3))                    # (16,256,D) 블록 정답
    wa = enc(xa[None])[0].view(16, 256, -1); wb = enc(xb[None])[0].view(16, 256, -1)         # 창 정답
    pd = (xa - xb).abs().mean(-1).view(16, 2, G, 16, G, 16).mean(dim=(1, 3, 5)).reshape(16, -1)   # (16,256) 칸 픽셀 차
    for b in range(8, 16):
        p = pd[b]
        if p.max() < 2:
            continue                                                                         # 이 블록은 두 미래가 같다 (물체 가려짐 등)
        regA = p > 2
        yy, xx = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
        w = p.cpu().numpy().reshape(G, G); cy, cx = int(round((w * yy).sum() / w.sum())), int(round((w * xx).sum() / w.sum()))
        regB = torch.zeros(G, G, dtype=torch.bool, device=dev); regB[max(cy - 1, 0):cy + 2, max(cx - 1, 0):cx + 2] = True; regB = regB.view(-1)
        rec = {"pair": ra["video_id"] + "|" + rb["video_id"], "motion": ra["motion"], "violation": ra["violation"], "block": b, "ctx_maxdiff": ctx_same}
        # 개입: arm b 의 물체 밖 칸 (A 를 1 칸 팽창한 영역 밖) 픽셀을 arm a 의 것으로 → 두 미래가 물체 칸에서만 다르다
        m2 = regA.view(G, G).clone()
        m2 = m2 | torch.roll(m2, 1, 0) | torch.roll(m2, -1, 0); m2 = m2 | torch.roll(m2, 1, 1) | torch.roll(m2, -1, 1)
        mk = m2.view(1, G, 1, G, 1, 1).expand(2, G, 16, G, 16, 1).reshape(2, R, R, 1)
        xb_t = torch.where(mk, xb[2 * b:2 * b + 2], xa[2 * b:2 * b + 2])
        ht = enc(xb_t[None])[0]
        rec["pix_outside_after"] = float((xb_t - xa[2 * b:2 * b + 2]).abs().mean(-1).view(2, G, 16, G, 16).mean(dim=(0, 2, 4)).reshape(-1)[~m2.view(-1)].max())
        for nm, d in (("block", (ha[b] - hb[b]).abs().mean(-1)), ("window", (wa[b] - wb[b]).abs().mean(-1)), ("pix", p),
                      ("blockT", (ha[b] - ht).abs().mean(-1))):
            for rn, reg in (("A", regA), ("B", regB)):
                rec[f"{nm}_{rn}_share"] = float(d[reg].sum() / d.sum())
            rec[f"{nm}_total"] = float(d.mean())
            rec[f"{nm}_far_mean"] = float(d[~regA].mean()) if (~regA).any() else None
            rec[f"{nm}_obj_mean"] = float(d[regA].mean())
        rec["area_A"] = float(regA.float().mean()); rec["area_B"] = float(regB.float().mean())
        REC.append(rec)
    if i % 20 == 0:
        print(f"{i}/{len(pairs)} 문맥 최대 픽셀 차 {ctx_same:.1f}", flush=True)


def summ(recs):
    out = {"n": len(recs)}
    for k in ("blockT_A_share", "blockT_B_share", "blockT_far_mean", "blockT_obj_mean", "blockT_total", "area_A", "area_B", "block_A_share", "block_B_share", "window_A_share", "window_B_share", "pix_A_share", "pix_B_share",
              "block_total", "window_total", "block_far_mean", "block_obj_mean", "window_far_mean", "window_obj_mean"):
        v = [r[k] for r in recs if r.get(k) is not None]
        out[k] = float(np.mean(v)) if v else None
    return out


res = {"all": summ(REC)}
for key in ("motion", "violation"):
    for val in sorted(set(r[key] for r in REC)):
        res[f"{key}={val}"] = summ([r for r in REC if r[key] == val])
for k, v in res.items():
    print(f"{k:24s} n {v['n']:4d} | 면적 A {v['area_A']:.3f} B {v['area_B']:.3f} | 물체 칸 몫: 블록 정답 A {v['block_A_share']:.3f} B {v['block_B_share']:.3f} · "
          f"창 정답 A {v['window_A_share']:.3f} B {v['window_B_share']:.3f} · 픽셀 A {v['pix_A_share']:.3f} B {v['pix_B_share']:.3f} | "
          f"칸당 |Δh| 물체 {v['block_obj_mean']:.3f} 먼 칸 {v['block_far_mean']:.3f} (창 {v['window_obj_mean']:.3f} / {v['window_far_mean']:.3f}) | "
          f"물체 밖 픽셀을 같게 하면: 몫 A {v['blockT_A_share']:.3f} B {v['blockT_B_share']:.3f}, 칸당 물체 {v['blockT_obj_mean']:.3f} 먼 칸 {v['blockT_far_mean']:.3f}")
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump({"summary": res, "records": REC}, open(a.out, "w"), indent=1)
