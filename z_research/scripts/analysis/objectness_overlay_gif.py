"""물체 존재 지도 GIF — 정답 토큰 · AR 예측의 칸별 물체 점수를 영상 위에 겹침, VOBench 영속 블록 (2026-10-26 저녁, GPU 1 장 · 6 GB).
사용자: "읽은 물체 존재 양을 gif 위에 overlap 해서 위치가 어딨는지 파악. moving · occlusion · static visible 준비. gifs 안 기존 것은 지워".
물체 점수 (pred_object_where.py 와 같은 정의): 칸 c, 블록 t 마다 (X − μ) ≈ b · (배경 − μ) + o · (마지막 물체 − μ) 를 최소제곱으로 풀어 o.
  μ = VOBench encoder 토큰 평균 (vobench_all.npz 화면 평균의 평균) · 배경 = 같은 블록의 빈 영상 pos_b (영속 블록은 b 가 빈 물체 → 같은 장면 · 같은 시간, 물체만 없음)
  · 마지막 물체 = 물체 a 가 마지막으로 보인 문맥 블록 (가린 비율 < 0.5) 의 물체 칸 평균.
패널 (2026-10-26 사용자 "raw, ar, overlay"): 원본 영상 · AR (문맥은 원본, 미래 8–15 는 블록 AR natural_ar_vw_state 굴림 예측을 decoder vob_grav_i1_i2_vitb_both 로) ·
  AR + 물체 점수 (문맥은 정답 토큰의 점수, 미래는 예측 토큰의 점수).
겹침: o ≥ 0.2 인 칸만 빨강 (투명도 ∝ o, 0.2 미만은 물체 없는 칸에도 깔리는 잡음이라 지움 — 2026-10-26 첫 판에서 확인). 물체 양 = o ≥ 0.2 칸의 o 합, 무게 중심도 그 칸으로.
  흰 원 = 실제 물체, 노란 × = 마지막에 본 물체, 하늘색 + = 물체 점수 무게 중심.
예시: 영속 · 도형 · pos_a — 정지 · 보임 2 / 이동 · 보임 2 (등속 · 경사) / 끝 가림 2 (정지 · 등속).
  CUDA_VISIBLE_DEVICES=0 MEM_GB=6 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/objectness_overlay_gif.py
출력: z_analysis/figures/gifs/objectness_<분류>_<물체>_<운동>.gif · z_analysis/results/objectness_overlay_gif.json (걸음별 물체 양 · 무게 중심 vs 실제 · 마지막 위치)
"""
import csv, json, os, sys
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw
import matplotlib; matplotlib.use("Agg"); import matplotlib.cm as cm

REPO = "/workspace/world_model/vjepa2"; sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "auto_research/scripts")); sys.path.insert(0, os.path.join(REPO, "z_pixel_decoder"))
dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEM_GB", 6)) * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory, 0)
_argv = list(sys.argv); sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT
sys.argv = _argv
from model import PixelDecoder
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint
S, C, K, D = 256, 8, 8, 1280
VROOT = "/workspace/dataset/benchmarks/VOBench_v1"; IDX = list(range(0, 96, 3))[:32]
OUT = f"{REPO}/z_analysis/figures/gifs"; os.makedirs(OUT, exist_ok=True)
MU = torch.from_numpy(np.load("/workspace/auto_research_cache/objcell_concept/vobench_all.npz")["whole"].astype(np.float32).mean(0)).to(dev)

rows = [r for r in csv.DictReader(open(f"{REPO}/data_csv/vobench_v1/index.csv")) if r["objset"] == "primitive" and r["violation"] == "permanence" and r["variant"] == "pos_a" and r["obj_b"] in ("", "empty")]
pick = []
for cat, occ, mots, n in (("static_visible", "visible", ("flat_static",), 2), ("moving_visible", "visible", ("flat_vel", "ramp_down"), 2), ("occlusion_late", "late", ("flat_static", "flat_vel"), 2)):
    for mot in mots:
        cand = [r for r in rows if r["occlusion"] == occ and r["motion"] == mot]
        pick += [(cat, r) for r in cand[: (n if len(mots) == 1 else 1)]]

enc2 = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
_load_state_dict(enc2, _load_checkpoint(CK)["target_encoder"], tag="t"); enc2 = enc2.half().to(dev).eval()
ar = FT.load_pred("natural_ar_vw_state", "latest.pt").to(dev).eval()
st = torch.load(f"{REPO}/z_pixel_decoder/runs/vob_grav_i1_i2_vitb_both/latest.pt", map_location="cpu", weights_only=False); mc = st["cfg"]["model"]
dec = PixelDecoder(in_dim=1280, dim=mc["dim"], depth=mc["depth"], heads=mc["heads"]).to(dev).eval(); dec.load_state_dict(st["decoder"]); del st
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(3, 1, 1, 1)


def load_arm(folder):
    fr = np.stack([np.asarray(Image.open(os.path.join(folder, f"{f:06d}.png")).convert("RGB")) for f in IDX])
    x = torch.from_numpy(fr).permute(3, 0, 1, 2).float().to(dev) / 255.0
    x = F.interpolate(x.permute(1, 0, 2, 3), size=(256, 256), mode="bilinear", align_corners=False, antialias=False).permute(1, 0, 2, 3)
    xn = (x - MEAN) / STD
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        hb = torch.cat([F.layer_norm(enc2(xn[:, 4 * q: 4 * q + 4].reshape(3, 2, 2, 256, 256).permute(1, 0, 2, 3, 4).half()).float(), (D,)) for q in range(8)])
    return (x.permute(1, 2, 3, 0).cpu().numpy() * 255).astype(np.uint8), hb


def objectness(X, B, O):
    """X, B: (n, 256, D) · O: (D,) → o (n, 256). (X−μ) ≈ b (B−μ) + o (O−μ), 칸마다 2 × 2 최소제곱."""
    y, u, v = X - MU, B - MU, (O - MU)[None, None]
    uu, uv, vv = (u * u).sum(-1), (u * v).sum(-1), (v * v).sum(-1); uy, vy = (u * y).sum(-1), (v * y).sum(-1)
    det = uu * vv - uv ** 2
    return ((uu * vy - uv * uy) / det.clamp(min=1e-6)).cpu().numpy()


TH = 0.2


def overlay(img, o):
    h = np.clip((o.reshape(16, 16) - TH) / (1 - TH), 0, 1); a = (0.75 * h).repeat(16, 0).repeat(16, 1)[..., None]
    return (img * (1 - a) + np.array([255, 30, 0]) * a).astype(np.uint8)


cc = np.arange(16) * 16 + 8.0; res = {}
for cat, r in pick:
    fa = os.path.join(VROOT, r["file_name"]); fb = fa.replace("/pos_a", "/pos_b"); W0 = Image.open(os.path.join(fa, "000000.png")).size[0]; sc = 256 / W0
    lab = {q["f"]: q for q in json.load(open(os.path.join(os.path.dirname(fa), "label.json")))["labels"].get("A", [])}
    raw, ha = load_arm(fa); _, hbk = load_arm(fb)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        p = ar.rollout_kv(ha[:C].reshape(1, C * S, D), torch.arange(C * S, device=dev).unsqueeze(0), K).float()[0].view(K, S, D)
    last_b, last_cells = None, None                                     # 마지막으로 보인 문맥 블록의 물체 칸
    for b in range(7, -1, -1):
        q = [lab.get(IDX[2 * b]), lab.get(IDX[2 * b + 1])]
        if all(t is not None for t in q) and max(t.get("hidden", 0) for t in q) < 0.5:
            x0 = min(t["u"] - t["ext_u"] for t in q) * sc; x1 = max(t["u"] + t["ext_u"] for t in q) * sc; y0 = min(t["v"] - t["ext_v"] for t in q) * sc; y1 = max(t["v"] + t["ext_v"] for t in q) * sc
            cells = [int(rr * 16 + c) for rr in np.nonzero((cc >= y0) & (cc <= y1))[0] for c in np.nonzero((cc >= x0) & (cc <= x1))[0]]
            if cells:
                last_b, last_cells = b, cells; break
    O = ha[last_b, last_cells].mean(0)
    o_gt = objectness(ha, hbk, O); o_pr = objectness(p, hbk[C:], O)
    with torch.no_grad():
        dp = (dec(p.float()) * STD.view(1, 3, 1, 1, 1) + MEAN.view(1, 3, 1, 1, 1)).clamp(0, 1)
    pred_img = (dp.permute(0, 2, 3, 4, 1).reshape(16, 256, 256, 3).cpu().numpy() * 255).astype(np.uint8)
    lq = lab.get(IDX[2 * last_b + 1]); last_c = (lq["u"] * sc, lq["v"] * sc)
    com = lambda o: (lambda w: (float((w * (np.arange(256) % 16 * 16 + 8)).sum() / w.sum()), float((w * (np.arange(256) // 16 * 16 + 8)).sum() / w.sum())) if w.sum() > 0 else None)(np.where(o >= TH, o, 0))
    amount = lambda o: float(np.where(o >= TH, o, 0).sum())
    summ = {"last_seen_block": int(last_b), "steps": {}}
    frames = []
    for f in range(32):                                              # 2026-10-26 사용자: "raw, ar, overlay 이렇게" — 패널 3 개
        b = f // 2; t = lab.get(IDX[f]); true_c = (t["u"] * sc, t["v"] * sc) if t is not None else None
        base = raw[f] if f < 16 else pred_img[f - 16]; o_now = o_gt[b] if f < 16 else o_pr[b - 8]
        panels = [("raw video", raw[f], None), ("AR prediction" if f >= 16 else "context (real)", base, None),
                  (("AR prediction + object score" if f >= 16 else "context + object score"), overlay(base, o_now), o_now)]
        canvas = Image.new("RGB", (3 * 256 + 2 * 6, 256 + 58), (255, 255, 255)); dr = ImageDraw.Draw(canvas)
        for pi, (nm, img, o) in enumerate(panels):
            im = Image.fromarray(img); d2 = ImageDraw.Draw(im)
            if true_c is not None:
                u, v = true_c; d2.ellipse([u - 12, v - 12, u + 12, v + 12], outline=(255, 255, 255), width=2)
            u, v = last_c; d2.line([u - 6, v - 6, u + 6, v + 6], fill=(255, 230, 0), width=2); d2.line([u - 6, v + 6, u + 6, v - 6], fill=(255, 230, 0), width=2)
            if o is not None and com(o) is not None:
                u, v = com(o); d2.line([u - 7, v, u + 7, v], fill=(0, 220, 255), width=3); d2.line([u, v - 7, u, v + 7], fill=(0, 220, 255), width=3)
            if pi > 0 and f >= 16:
                d2.rectangle([0, 0, 255, 255], outline=(230, 40, 40), width=4)
            ox = pi * 262; canvas.paste(im, (ox, 22))
            dr.text((ox + 4, 4), nm + (f" | amount {amount(o):.2f}" if o is not None else ""), fill=(0, 0, 0))
        dr.text((4, 256 + 26), f"{cat}: {r['obj_a']}, {r['motion']} | t = {f * 0.25:.2f} s | " + ("context" if f < 16 else f"prediction step {b - 7}"), fill=(0, 0, 0))
        dr.text((4, 256 + 40), "red: object score >= 0.2 | white o: true | yellow x: last seen | cyan +: centre of object score", fill=(0, 0, 0))
        frames.append(canvas)
        if f >= 16 and f % 2 == 0:
            k = b - 7; summ["steps"][k] = {"amount_gt": round(amount(o_gt[b]), 2), "amount_pred": round(amount(o_pr[b - 8]), 2),
                                            "com_pred": com(o_pr[b - 8]), "com_gt": com(o_gt[b]), "true": true_c, "last_seen": last_c}
    name = f"objectness_{cat}_{r['obj_a'].replace('/', '_')}_{r['motion']}"
    frames[0].save(f"{OUT}/{name}.gif", save_all=True, append_images=frames[1:], duration=125, loop=0)
    res[name] = summ; print("saved", name, {k: (v["amount_gt"], v["amount_pred"]) for k, v in summ["steps"].items()}, flush=True)
json.dump(res, open(f"{REPO}/z_analysis/results/objectness_overlay_gif.json", "w"), indent=1)
