"""릴리즈 predictor 와 블록 AR 의 1 · 2 · 8 블록 예측 — pixel decoder GIF, VOBench 보임 · 움직임 (2026-10-27, CPU).
사용자: "기존 release 도 raw video 1, 2, 8 block 예측 이렇게 각각 해봐. 1 행을 release, 2 행을 ar 로".
칸 (두 행 같음): raw video · 1 block · 2 block · 8 block. 미래 = 블록 8–15 (16 장), VOBench 간격 장 0.1 s · 블록 0.2 s.
  1 block : 블록 b 마다 실제 0 … b−1 을 문맥으로 1 블록 예측
  2 block : 실제 0–7 → 8 · 9 / 0–9 → 10 · 11 / 0–11 → 12 · 13 / 0–13 → 14 · 15
  8 block : 실제 0–7 → 8 … 15
릴리즈 (1 행): 문맥 장 = 2 × (문맥 블록 수), 목표 장 = 2 × (예측 블록 수) 로 context_encoder(masks=문맥) → predictor(mask_index 0) 를 한 번에 (되먹임 없음).
  문맥이 16 장보다 길어지는 경우 (1 · 2 block 의 뒤쪽) 는 채점 설정 (문맥 16 장) 밖이지만 릴리즈 학습 창 (64 장) 안이다.
블록 AR (2 행): natural_ar_vw_state latest. 1 block = 실제 블록을 넣은 출력 (teacher forcing), 2 block = 덩어리 시작에서 상태 되먹임 한 번, 8 block = 8 걸음 굴림.
decoder: z_pixel_decoder vob_grav_i1_i2_vitb_both (블록 latent + 32 장 window latent 둘 다로 학습 → 릴리즈 예측도 그대로 넣는다, render_predictor.py 와 같다).
GPU 를 쓰지 않는다 (2026-10-27 다른 세션 학습이 GPU 0–7 을 거의 다 씀). float32 CPU.
  /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/release_ar_steps_gif.py
출력: z_analysis/figures/gifs/steps_release_ar_moving_visible_<물체>_<운동>.gif (ar_steps_* 는 이것의 2 행으로 대체)
"""
import csv, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw

REPO = "/workspace/world_model/vjepa2"; sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "auto_research/scripts")); sys.path.insert(0, os.path.join(REPO, "z_pixel_decoder"))
torch.set_num_threads(48); dev = torch.device("cpu")
_argv = list(sys.argv); sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT
sys.argv = _argv
from model import PixelDecoder
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint, build_from_config
from analysis.intphys2.surprise import _context_target_indices
S, C, K, D, NB = 256, 8, 8, 1280, 16
VROOT = "/workspace/dataset/benchmarks/VOBench_v1"; IDX = list(range(0, 96, 3))[:32]
OUT = f"{REPO}/z_analysis/figures/gifs"; os.makedirs(OUT, exist_ok=True)

rows = [r for r in csv.DictReader(open(f"{REPO}/data_csv/vobench_v1/index.csv")) if r["objset"] == "primitive" and r["violation"] == "permanence"
        and r["variant"] == "pos_a" and r["obj_b"] in ("", "empty") and r["occlusion"] == "visible"]
pick = [[r for r in rows if r["motion"] == m][0] for m in ("flat_vel", "flat_acc", "ramp_down", "ramp_up")]

CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
enc2 = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(enc2, _load_checkpoint(CK)["target_encoder"], tag="t"); enc2 = enc2.float().eval()
MCFG = {"arch_name": "vit_huge", "checkpoint": CK, "img_size": 256, "patch_size": 16, "tubelet_size": 2, "use_rope": True, "uniform_power": False, "dual_encoder": True,
        "context_encoder_key": "encoder", "target_encoder_key": "target_encoder", "predictor": {"embed_dim": 384, "depth": 12, "num_heads": 12, "num_mask_tokens": 10},
        "dtype": "float32", "window_size": 32}
rel = build_from_config(MCFG, dev); rel.context_encoder.float().eval(); rel.predictor.float().eval()
ar = FT.load_pred("natural_ar_vw_state", "latest.pt").float().eval(); T1 = ar.type_embed[1]
st = torch.load(f"{REPO}/z_pixel_decoder/runs/vob_grav_i1_i2_vitb_both/latest.pt", map_location="cpu", weights_only=False); mc = st["cfg"]["model"]
dec = PixelDecoder(in_dim=1280, dim=mc["dim"], depth=mc["depth"], heads=mc["heads"]).eval(); dec.load_state_dict(st["decoder"]); del st
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
idx_all = torch.arange(NB * S).unsqueeze(0)


def load(folder):
    fr = np.stack([np.asarray(Image.open(os.path.join(folder, f"{f:06d}.png")).convert("RGB")) for f in IDX])
    x = torch.from_numpy(fr).permute(3, 0, 1, 2).float() / 255.0
    x = F.interpolate(x.permute(1, 0, 2, 3), size=(256, 256), mode="bilinear", align_corners=False, antialias=False).permute(1, 0, 2, 3)
    xn = (x - MEAN) / STD
    with torch.no_grad():
        h = torch.cat([F.layer_norm(enc2(xn[:, 4 * q: 4 * q + 4].reshape(3, 2, 2, 256, 256).permute(1, 0, 2, 3, 4)), (D,)) for q in range(8)])
    return (x.permute(1, 2, 3, 0).numpy() * 255).astype(np.uint8), xn, h


def decode(P):
    with torch.no_grad():
        d = (dec(P) * STD.view(1, 3, 1, 1, 1) + MEAN.view(1, 3, 1, 1, 1)).clamp(0, 1)
    return (d.permute(0, 2, 3, 4, 1).reshape(-1, 256, 256, 3).numpy() * 255).astype(np.uint8)


def release(xn, n_ctx, n_tgt):
    """실제 블록 0 … n_ctx−1 → 블록 n_ctx … n_ctx+n_tgt−1 예측 (n_tgt, 256, D)."""
    ci, ti = _context_target_indices(ctx_frames=2 * n_ctx, tgt_frames=2 * n_tgt, tubelet_size=2, spatial_tokens=S, batch_size=1, device=dev)
    with torch.no_grad():
        z = rel.context_encoder(xn.unsqueeze(0), masks=[ci]); p = rel.predictor(z, ci, ti, mask_index=0)
    return p.float()[0].view(n_tgt, S, D)


for r in pick:
    t0 = time.time()
    fa = os.path.join(VROOT, r["file_name"]); W0 = Image.open(os.path.join(fa, "000000.png")).size[0]; sc = 256 / W0
    lab = {q["f"]: q for q in json.load(open(os.path.join(os.path.dirname(fa), "label.json")))["labels"].get("A", [])}
    raw, xn, h = load(fa)
    # 릴리즈
    R1 = torch.stack([release(xn, b, 1)[0] for b in range(C, NB)])
    R2 = torch.cat([release(xn, c, 2) for c in range(C, NB, 2)])
    R8 = release(xn, C, K)
    # 블록 AR
    one, two = {}, {}
    with torch.no_grad():
        caches = None
        for j in range(NB - 1):
            p1, caches, stt = ar._ar_step_kv(h[j][None], idx_all[:, j * S:(j + 1) * S], caches)
            if j >= C - 1:
                one[j + 1] = p1[0]
            if j >= C - 1 and (j - (C - 1)) % 2 == 0 and j + 2 < NB:
                p2, _, _ = ar._ar_step_kv(stt + T1, idx_all[:, (j + 1) * S:(j + 2) * S], caches, embedded=True)
                two[j + 1] = p1[0]; two[j + 2] = p2[0]
        A8 = ar.rollout_kv(h[:C].reshape(1, C * S, D), idx_all[:, :C * S], K)[0].view(K, S, D)
    A1 = torch.stack([one[b] for b in range(C, NB)]); A2 = torch.stack([two[b] for b in range(C, NB)])
    imgs = {("release", k): decode(v) for k, v in (("1", R1), ("2", R2), ("8", R8))}
    imgs.update({("AR", k): decode(v) for k, v in (("1", A1), ("2", A2), ("8", A8))})
    frames = []
    W = 4 * 256 + 3 * 6 + 70
    for f in range(32):
        b = f // 2; t = lab.get(IDX[f]); tc = (t["u"] * sc, t["v"] * sc) if t is not None else None
        canvas = Image.new("RGB", (W, 2 * (256 + 22) + 22), (255, 255, 255)); dr = ImageDraw.Draw(canvas)
        for ri, model in enumerate(("release", "AR")):
            oy = ri * (256 + 22)
            dr.text((6, oy + 22 + 120), model, fill=(0, 0, 0))
            for ci_, (nm, key) in enumerate((("raw video", None), ("1 block", "1"), ("2 block", "2"), ("8 block", "8"))):
                img = raw[f] if (key is None or f < 16) else imgs[(model, key)][f - 16]
                im = Image.fromarray(img); d2 = ImageDraw.Draw(im)
                if tc is not None:
                    u, v = tc; d2.ellipse([u - 12, v - 12, u + 12, v + 12], outline=(255, 255, 255), width=2)
                if key is not None and f >= 16:
                    d2.rectangle([0, 0, 255, 255], outline=(230, 40, 40), width=4)
                ox = 70 + ci_ * 262; canvas.paste(im, (ox, oy + 22)); dr.text((ox + 4, oy + 6), nm, fill=(0, 0, 0))
        dr.text((6, 2 * (256 + 22) + 6), f"{r['obj_a']}, {r['motion']} | t = {f * 0.1:.1f} s | block {b} | " + ("context" if f < 16 else "prediction") + " | white o: true object", fill=(0, 0, 0))
        frames.append(canvas)
    name = f"steps_release_ar_moving_visible_{r['obj_a'].replace('/', '_')}_{r['motion']}"
    frames[0].save(f"{OUT}/{name}.gif", save_all=True, append_images=frames[1:], duration=150, loop=0)
    print("saved", name, f"{time.time() - t0:.0f}s", flush=True)
