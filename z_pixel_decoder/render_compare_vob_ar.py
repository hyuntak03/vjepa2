#!/usr/bin/env python3
"""VOBench late (split v2 시험 절반) permanence 72 쌍에서 AR 기본판 vs AR + v4 이어 학습의 예측을 pixel decoder 로 나란히 그린다 (2026-11-02, 사용자 지시).

열 셋: [원본 (가능 영상) | AR 기본판 예측 | AR + v4 이어 학습 예측]. 앞 16 장 (문맥) 은 원본, 뒤 16 장은 각 모델의 블록 AR rollout 을 decoder 로.
폴더: <primitive|realistic>/<moving_A|moving_B|static_A|static_B> (A = 사라짐 짝: 문맥에 물체, B = 나타남 짝). 물체 정지 = motion flat_static.
입력은 채점과 같다: 원본 0, 3, …, 93 (32 장), 문맥 16 장 = 블록 8 개 (블록별 LN(target_encoder)) → 상태 되먹임 KV rollout 8 걸음.
머리줄 correct / wrong = 이 스크립트가 같은 정의로 센 쌍 판정: 각 영상 자기 문맥으로 예측 p, surprise = mean |p − LN(h_블록)[미래]|, s(불가능) > s(가능) 이면 correct.
  (AR 채점기 analysis/predictors/ar_scoring.py 와 같은 정의. VOBench 짝은 문맥이 픽셀로 같아 두 영상의 p 가 같다.)
decoder = z_pixel_decoder/runs/vob_grav_i1_vitb_both (블록별 LN(target_encoder) → 픽셀, AR 의 예측 공간과 같다).
AR + v4 는 VOBench 를 본 적이 없다 (학습 = BlenderTrack v4). AR 기본판도 VOBench 학습 없음.

  CUDA_VISIBLE_DEVICES=0 /venv/vjepa/bin/python z_pixel_decoder/render_compare_vob_ar.py
  출력: z_analysis/figures/vob_late_ar_v4/<objset>/<칸>/<motion>__<쌍>.gif · README.md · summary.json
"""
import csv, json, os, sys
import numpy as np, torch, torch.nn.functional as F, yaml
from PIL import Image, ImageDraw
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint, _clean_backbone_key   # noqa: E402
from src.models.rollout_predictor import vit_prefix_predictor           # noqa: E402
from model import PixelDecoder                                          # noqa: E402

dev = torch.device("cuda")
T, C, S = 32, 16, 256
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
MODELS = [("AR (base)", "natural_ar_vw_state"), ("AR + v4 PFT", "v4vc_ar_vw_state_postft")]
DEC = os.path.join(HERE, "runs/vob_grav_i1_vitb_both/latest.pt")
FR = "/workspace/dataset/benchmarks/VOBench_v1"
IDX = [3 * k for k in range(T)]
OUT = os.path.join(REPO, "z_analysis/figures/vob_late_ar_v4")
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(3, 1, 1, 1)
LN = lambda t: F.layer_norm(t, (t.size(-1),))

# ── 쌍 · 칸
rows = list(csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1_split_v2/index_test_late.csv"))))
by = {}
for r in rows:
    if r["violation_type"] == "permanence":
        by.setdefault((r["block_id"], r["pair_id"]), {})["pos" if r["plausible"] == "1" else "imp"] = r
pairs = []
for (b, p), v in sorted(by.items(), key=lambda kv: (int(kv[0][0]), kv[0][1])):
    r = v["pos"]
    cell = f"{'static' if r['motion'] == 'flat_static' else 'moving'}_{r['pair_id']}"
    pairs.append((r["objset"], cell, v["pos"], v["imp"]))
print("쌍", len(pairs))

# ── 모델 · encoder · decoder
def load_ar(run):
    c = yaml.safe_load(open(os.path.join(REPO, "z_training/runs", run, "config.yaml")))["model"]["predictor"]
    assert c.get("kind") == "ar" and c.get("ar_input", "block") == "block", run
    m = vit_prefix_predictor(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384, depth=12,
                             num_heads=12, num_mask_tokens=10, mask_mode="ar", ar_feedback=c.get("ar_feedback", "output"))
    sd = {k: v for k, v in _clean_backbone_key(torch.load(os.path.join(REPO, "z_training/runs", run, "latest.pt"), map_location="cpu",
                                                           weights_only=False)["predictor"]).items()
          if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}
    m.load_state_dict(sd, strict=True)
    return m.to(dev).eval()
ars = [load_ar(run) for _, run in MODELS]
enc2 = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(enc2, _load_checkpoint(CK)["target_encoder"], tag="t"); enc2 = enc2.to(dev).eval()
st = torch.load(DEC, map_location="cpu", weights_only=False); mc = st["cfg"]["model"]
dec = PixelDecoder(in_dim=1280, dim=mc["dim"], depth=mc["depth"], heads=mc["heads"]).to(dev).eval(); dec.load_state_dict(st["decoder"])


def load(fn):
    fr = [np.asarray(Image.open(f"{FR}/{fn}/{i:06d}.png").convert("RGB")) for i in IDX]
    x = torch.from_numpy(np.stack(fr)).permute(3, 0, 1, 2).float().to(dev) / 255.0
    return F.interpolate(x.transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False, antialias=False).transpose(0, 1)


@torch.no_grad()
def blocks(x01):
    xn = ((x01 - MEAN) / STD)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        b = enc2(xn.view(3, T // 2, 2, 256, 256).permute(1, 0, 2, 3, 4))
    return LN(b.float()).reshape(-1, b.size(-1))                                   # (16·256, D)


@torch.no_grad()
def rollout(m, hb):
    idx = torch.arange(C // 2 * S, device=dev).unsqueeze(0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return m.rollout_kv(hb[: C // 2 * S].unsqueeze(0), idx, (T - C) // 2).float()[0]   # (8·256, D)


@torch.no_grad()
def decode(tok):
    nb = tok.size(0) // S
    y = dec(tok.view(nb, S, -1).float())
    y = (y * STD.view(1, 3, 1, 1, 1) + MEAN.view(1, 3, 1, 1, 1)).clamp(0, 1)
    return y.permute(0, 2, 1, 3, 4).reshape(nb * 2, 3, 256, 256)


summ = {}
for objset, cell, rp, ri in pairs:
    od = os.path.join(OUT, objset, cell); os.makedirs(od, exist_ok=True)
    xp, xi = load(rp["file_name"]), load(ri["file_name"])
    hp, hi = blocks(xp), blocks(xi)
    preds, verdict = [], []
    for m in ars:
        pp, pi = rollout(m, hp), rollout(m, hi)                                     # 각 영상 자기 문맥 (VOBench 는 같다)
        sp = float((pp - hp[C // 2 * S:]).abs().mean()); si = float((pi - hi[C // 2 * S:]).abs().mean())
        verdict.append("correct" if si > sp else ("tie" if si == sp else "wrong"))
        preds.append(decode(pp))
    ctx = xp[:, :C].permute(1, 0, 2, 3)
    cols = [xp.permute(1, 0, 2, 3)] + [torch.cat([ctx, y]) for y in preds]
    titles = ["original (possible video)"] + [f"{n}: {v}" for (n, _), v in zip(MODELS, verdict)]
    frames = []
    for t in range(T):
        body = torch.cat([c[t] for c in cols], -1)
        im = Image.fromarray((body.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        canvas = Image.new("RGB", (im.width, im.height + 34), (0, 0, 0)); canvas.paste(im, (0, 34))
        dr = ImageDraw.Draw(canvas)
        for j, ttl in enumerate(titles):
            col = (255, 255, 255) if j == 0 else ((120, 220, 120) if "correct" in ttl else (255, 120, 120))
            dr.text((j * 256 + 3, 2), ttl[:42], fill=col)
            dr.text((j * 256 + 3, 18), "context" if t < C else ("future" if j == 0 else "future (predicted)"),
                    fill=(255, 255, 255) if t < C else (255, 220, 0))
        dr.text((3, canvas.height - 12), f"{objset} {cell} {rp['motion']}  frame {IDX[t]}", fill=(255, 255, 255))
        frames.append(canvas)
    nm = f"{rp['motion']}__{rp['file_name'].split('/')[-2]}"
    frames[0].save(os.path.join(od, f"{nm}.gif"), save_all=True, append_images=frames[1:], duration=125, loop=0)
    summ.setdefault(f"{objset}/{cell}", []).append({"gif": f"{objset}/{cell}/{nm}.gif", "pos": rp["file_name"], "imp": ri["file_name"],
                                                    **{run: v for (_, run), v in zip(MODELS, verdict)}})
    print(objset, cell, nm, verdict, flush=True)

json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
lines = ["# VOBench late permanence — AR 기본판 vs AR + v4 이어 학습, pixel decoder 그림 (2026-11-02)", "",
         "열: 원본 (가능 영상) · AR 기본판 · AR + v4 이어 학습. 앞 16 장 = 문맥 (원본), 뒤 16 장 = 각 모델의 블록 AR 예측을 decoder 로 그린 것.",
         "머리줄 correct / wrong = 이 스크립트의 쌍 판정 (AR 채점과 같은 정의). A = 사라짐 짝 (문맥에 물체), B = 나타남 짝. 둘 다 VOBench 로 학습한 적 없다.", ""]
for k in sorted(summ):
    L = summ[k]
    acc = {run: sum(x[run] == "correct" for x in L) for _, run in MODELS}
    lines.append(f"## {k} ({len(L)} 쌍) — 맞힌 쌍: AR {acc['natural_ar_vw_state']} · AR + v4 {acc['v4vc_ar_vw_state_postft']}")
    lines += [f"- `{x['gif']}` — AR {x['natural_ar_vw_state']} · AR + v4 {x['v4vc_ar_vw_state_postft']}" for x in L]
    lines.append("")
lines += ["## 재현", "", "```bash", "CUDA_VISIBLE_DEVICES=0 /venv/vjepa/bin/python z_pixel_decoder/render_compare_vob_ar.py", "```"]
open(os.path.join(OUT, "README.md"), "w").write("\n".join(lines) + "\n")
print("OUT", OUT)
