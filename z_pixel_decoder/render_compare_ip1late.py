#!/usr/bin/env python3
"""IntPhys 1 late occ (O1 가림 30 쌍) 에서 릴리즈 출발 두 판의 예측을 pixel decoder 로 나란히 그린다 (2026-11-02, 사용자 지시).

열 셋: [원본 (가능 영상) | 릴리즈 + v4 이어 학습 예측 | 릴리즈 + v4 + 음의 손실 예측]. 앞 16 장 (문맥) 은 세 열 모두 원본, 뒤 16 장은 각 모델 예측을 decoder 로.
폴더: 시트 Object_permanence 칸 그대로 — moving_A · moving_B · static_A · static_B (A = 사라짐 짝: 문맥에서 물체를 봤다, B = 나타남 짝).
  칸 정의는 z_research/scripts/analysis/op_cells_vast.py 와 같다 (block_type O1, group <motion>/occluded, A = 가능 쪽 event_kind != not_seen_before_d).
머리줄의 correct / wrong = 그 모델의 표준 채점 결과 (cvpr/results/eval/intphys1_late/intphys1_late_uni_<모델>/per_block.json, 시트 값과 같은 판정).
입력은 채점과 같다: IntPhys1_late_uni 32 장 (문맥 [0, d−1] 균일 16 · 미래 [d, 99] 균일 16), 가능 영상의 문맥으로 예측. fp16 autocast.
decoder = z_pixel_decoder/runs/vob_grav_i1_vitb_both (VOBench + gravity + IntPhys1 dev 로 학습, 블록별 LN(target_encoder) → 픽셀).
  ⚠️ 릴리즈 계열 predictor 는 32 장 한 번에 넣은 target_encoder 의 토큰을 예측한다 (decoder 학습 공간 = 블록별) — render_predictor.py 와 같은 그리기다.

  CUDA_VISIBLE_DEVICES=0 /venv/vjepa/bin/python z_pixel_decoder/render_compare_ip1late.py
  출력: z_analysis/figures/ip1_late_occ_release_v4_negimp/<칸>/<쌍>.gif · README.md · summary.json
"""
import csv, json, os, sys
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from analysis.intphys2.model import build_from_config                  # noqa: E402
from evals.world_model_analysis.eval import _context_target_indices    # noqa: E402
from model import PixelDecoder                                          # noqa: E402

dev = torch.device("cuda")
T, C, S = 32, 16, 256
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
BASE = {"arch_name": "vit_huge", "checkpoint": CK, "img_size": 256, "patch_size": 16, "tubelet_size": 2, "use_rope": True,
        "uniform_power": False, "dual_encoder": True, "context_encoder_key": "encoder", "target_encoder_key": "target_encoder",
        "predictor": {"embed_dim": 384, "depth": 12, "num_heads": 12, "num_mask_tokens": 10}, "dtype": "float32",
        "autocast": "float16", "window_size": 32}
MODELS = [("release + v4 PFT", "vith_release_v4vc", "z_training/runs/v4vc_release_postft/latest.pt"),
          ("release + v4 PFT w/ negative loss", "vith_release_v4vc_negimp", "z_training/runs/v4vc_release_negimp/latest.pt")]
DEC = os.path.join(HERE, "runs/vob_grav_i1_vitb_both/latest.pt")
FR = "/workspace/world_model/local_datasets/world/world_analysis/IntPhys1_late_uni"
OUT = os.path.join(REPO, "z_analysis/figures/ip1_late_occ_release_v4_negimp")
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(3, 1, 1, 1)

# ── 쌍 · 칸
rows = list(csv.DictReader(open(os.path.join(REPO, "data_csv/intphys1_late_uni/index.csv"))))
by = {}
for r in rows:
    by.setdefault((r["block_id"], r["pair_id"]), {})["pos" if r["plausible"] == "1" else "imp"] = r
pairs = []
for (b, p), v in sorted(by.items()):
    r = v["pos"]
    if r["block_type"] != "O1" or not r["group"].endswith("/occluded"):
        continue
    cell = f"{r['group'].split('/')[0]}_{'A' if r['event_kind'] != 'not_seen_before_d' else 'B'}"
    pairs.append((cell, v["pos"], v["imp"]))
print("쌍", len(pairs), {c: sum(1 for x in pairs if x[0] == c) for c in ("moving_A", "moving_B", "static_A", "static_B")})

# ── 표준 채점 결과 (시트와 같은 판정)
PV = {key: json.load(open(os.path.join(REPO, f"cvpr/results/eval/intphys1_late/intphys1_late_uni_{key}/per_block.json")))["per_video_surprise"]
      for _, key, _ in MODELS}

# ── 모델 · decoder
bundles = []
for name, key, ck in MODELS:
    cfg = dict(BASE, predictor_checkpoint=os.path.join(REPO, ck))
    bundles.append(build_from_config(cfg, dev))
    print("loaded", name, flush=True)
st = torch.load(DEC, map_location="cpu", weights_only=False); mc = st["cfg"]["model"]
dec = PixelDecoder(in_dim=1280, dim=mc["dim"], depth=mc["depth"], heads=mc["heads"]).to(dev).eval(); dec.load_state_dict(st["decoder"])
ci, ti = _context_target_indices(ctx_frames=C, tgt_frames=T - C, tubelet_size=2, spatial_tokens=S, batch_size=1, device=dev)


def load(vid):
    fr = [np.asarray(Image.open(f"{FR}/{vid}/{i:06d}.png").convert("RGB")) for i in range(T)]
    x = torch.from_numpy(np.stack(fr)).permute(3, 0, 1, 2).float().to(dev) / 255.0
    return F.interpolate(x.transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False, antialias=False).transpose(0, 1)


@torch.no_grad()
def decode(tok):
    nb = tok.size(0) // S
    y = dec(tok.view(nb, S, -1).float())
    y = (y * STD.view(1, 3, 1, 1, 1) + MEAN.view(1, 3, 1, 1, 1)).clamp(0, 1)
    return y.permute(0, 2, 1, 3, 4).reshape(nb * 2, 3, 256, 256)


summ = {}
for cell, rp, ri in pairs:
    od = os.path.join(OUT, cell); os.makedirs(od, exist_ok=True)
    x01 = load(rp["video_id"]); xn = ((x01 - MEAN) / STD).unsqueeze(0)
    preds = []
    for bd in bundles:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            z = bd.context_encoder(xn, masks=[ci])
            p = bd.predictor(z, ci, ti, mask_index=0)
        preds.append(decode(p.float()[0]))
    ctx = x01[:, :C].permute(1, 0, 2, 3)
    cols = [x01.permute(1, 0, 2, 3)] + [torch.cat([ctx, y]) for y in preds]
    verdict = []
    for _, key, _ in MODELS:
        sp, si = PV[key][rp["video_id"]], PV[key][ri["video_id"]]
        verdict.append("correct" if si > sp else ("tie" if si == sp else "wrong"))
    titles = ["original (possible video)"] + [f"{n}: {v}" for (n, _, _), v in zip(MODELS, verdict)]
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
        dr.text((3, canvas.height - 12), f"{cell}  {rp['block']}/{rp['quadruplet']}  sample {t}", fill=(255, 255, 255))
        frames.append(canvas)
    nm = f"{rp['block']}_{rp['quadruplet']}_pair{rp['pair_id']}"
    frames[0].save(os.path.join(od, f"{nm}.gif"), save_all=True, append_images=frames[1:], duration=125, loop=0)
    summ.setdefault(cell, []).append({"gif": f"{cell}/{nm}.gif", "pos": rp["video_id"], "imp": ri["video_id"],
                                      **{key: v for (_, key, _), v in zip(MODELS, verdict)}})
    print(cell, nm, verdict, flush=True)

json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
lines = ["# IntPhys 1 late occ — 릴리즈 + v4 이어 학습 vs + 음의 손실, pixel decoder 그림 (2026-11-02)", "",
         "열: 원본 (가능 영상) · 릴리즈 + v4 이어 학습 · 릴리즈 + v4 + 음의 손실. 앞 16 장 = 문맥 (원본), 뒤 16 장 = 각 모델 예측을 decoder 로 그린 것.",
         "머리줄 correct / wrong = 표준 채점 (시트 Object_permanence 와 같은 판정). A = 사라짐 짝 (문맥에서 물체를 봤다), B = 나타남 짝.",
         "⚠️ decoder 는 블록별 정답 latent 로 학습했다. 릴리즈 계열 predictor 의 출력은 32 장 창 공간이라 흐릿하게 나올 수 있다.", ""]
for cell in ("moving_A", "moving_B", "static_A", "static_B"):
    L = summ.get(cell, [])
    acc = {key: sum(x[key] == "correct" for x in L) for _, key, _ in MODELS}
    lines.append(f"## {cell} ({len(L)} 쌍) — 맞힌 쌍: v4 {acc['vith_release_v4vc']} · v4 + 음의 손실 {acc['vith_release_v4vc_negimp']}")
    lines += [f"- `{x['gif']}` — v4 {x['vith_release_v4vc']} · 음의 손실 {x['vith_release_v4vc_negimp']}" for x in L]
    lines.append("")
lines += ["## 재현", "", "```bash", "CUDA_VISIBLE_DEVICES=0 /venv/vjepa/bin/python z_pixel_decoder/render_compare_ip1late.py", "```"]
open(os.path.join(OUT, "README.md"), "w").write("\n".join(lines) + "\n")
print("OUT", OUT)
