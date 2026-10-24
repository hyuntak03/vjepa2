#!/usr/bin/env python3
"""릴리즈 V-JEPA 2 predictor 가 만든 VOBench 미래를 pixel decoder 로 그림 (2026-10-20, 사용자 지시).

  CUDA_VISIBLE_DEVICES=0 /venv/vjepa/bin/python z_pixel_decoder/render_predictor.py --set late --n 12
  CUDA_VISIBLE_DEVICES=0 /venv/vjepa/bin/python z_pixel_decoder/render_predictor.py --set gravity --n 12

채점 (surprise_c16t32) 과 같은 입력: 32 장 (원본 0, 3, …, 93 장), 문맥 16 장 (블록 8) → 미래 16 장 (블록 8), fp16 autocast.
decoder 는 **블록별** LN(target_encoder) 로 학습했다. 릴리즈 predictor 는 **32 장 한 번에** 넣은 target_encoder 의 미래 토큰을
예측하므로 공간이 다를 수 있다 → 세 가지를 같이 그려 PSNR 로 비교한다 (미래 16 장, [0, 1]):
  blk   블록별 정답 토큰 → decoder   (decoder 학습 분포, 상한)
  win   32 장 정답 토큰 (채점 타깃) → decoder   (공간 차이 확인)
  pred  릴리즈 predictor 예측 → decoder
GIF (32 장, 8 fps): 왼쪽 = 원본, 가운데 = 앞 16 장 원본 + 뒤 16 장 pred, 오른쪽 = 앞 16 장 원본 + 뒤 16 장 정답 latent 복원.
주황 점 = 문맥 마지막 장의 물체 위치 (label.json, 가려져 있었으면 속 빈 원) — 미래에서 물체가 얼마나 움직였나 보는 기준.
산출: runs/<decoder run>/renders/<model>/<set>/<clip>.gif · summary.json · grid.png (grid_from_gifs.py)"""
import argparse, csv, json, os, sys, random
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from analysis.intphys2.model import build_from_config                  # noqa: E402
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint   # noqa: E402
from evals.world_model_analysis.eval import _context_target_indices    # noqa: E402
from model import PixelDecoder                                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--set", choices=["late", "visible", "early", "mid", "gravity", "intphys1_late_uni", "phys3d_val", "intphys2_uni"], default="late")
ap.add_argument("--n", type=int, default=12)
ap.add_argument("--motion", default=None, help="VOBench motion 거르기 (예: flat_static). 결과 폴더는 <set>_<motion>")
ap.add_argument("--decoder", default=os.path.join(HERE, "runs/vobench_vits/latest.pt"))
ap.add_argument("--root", default="/workspace/dataset/benchmarks/VOBench_v1")
ap.add_argument("--model", default="release", help="release | z_training/runs 의 AR run 이름 (kind=ar, 블록 latent 공간에서 rollout)")
ap.add_argument("--ckpt", default="latest.pt", help="AR run 의 체크포인트")
ap.add_argument("--pred_ckpt", default=None, help="release 모델에 얹을 post-FT predictor 체크포인트 (z_training/runs/<run>/latest.pt)")
ap.add_argument("--only", default=None, help="video_id 접두어 (쉼표) 만 그린다 (2026-10-22)")
ap.add_argument("--i2_root", default=None, help="intphys2_uni 프레임 폴더 바꾸기 (예: 라벨 d 를 바꿔 다시 뽑은 scratch 폴더)")
ap.add_argument("--i2_camera", default=None, help="intphys2_uni: Fixed | Moving 만 (결과 폴더 intphys2_uni_<camera>)")
ap.add_argument("--i2_test", action="store_true", help="intphys2_uni: IntPhys2 post-FT 의 시험 장면 (index_test.csv) 만")
ap.add_argument("--i2_variant", default="pos", help="intphys2_uni: pos | imp | both")
ap.add_argument("--name", default=None, help="그림 · 폴더에 쓸 짧은 이름 (예: AR). 없으면 --model")
a = ap.parse_args()
dev = torch.device("cuda")
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
MCFG = {"arch_name": "vit_huge", "checkpoint": CK, "img_size": 256, "patch_size": 16, "tubelet_size": 2, "use_rope": True,
        "uniform_power": False, "dual_encoder": True, "context_encoder_key": "encoder", "target_encoder_key": "target_encoder",
        "predictor": {"embed_dim": 384, "depth": 12, "num_heads": 12, "num_mask_tokens": 10}, "dtype": "float32",
        "autocast": "float16", "window_size": 32}
AR_RUN = None if a.model == "release" else a.model
if AR_RUN is None:
    if a.pred_ckpt:                 # 2026-10-22: release + post-FT predictor (채점기와 같은 predictor_checkpoint 키)
        MCFG["predictor_checkpoint"] = a.pred_ckpt
    bundle = build_from_config(MCFG, dev)
else:                       # 2026-10-20: 블록 AR — 문맥 블록 8 개 (블록별 LN target_encoder) → 상태 되먹임 KV rollout 8 걸음
    import yaml
    from src.models.rollout_predictor import vit_prefix_predictor
    from analysis.intphys2.model import _clean_backbone_key
    _D = os.path.join(REPO, "z_training/runs", AR_RUN); _c = yaml.safe_load(open(os.path.join(_D, "config.yaml")))
    _pc = _c["model"]["predictor"]
    KIND = _pc.get("kind")
    assert (KIND == "ar" and _pc.get("ar_input", "block") == "block") or (KIND == "ctx_state" and _pc.get("ctx_target") == "block"), \
        "kind=ar (블록 입력) · ctx_state (블록 타깃) 만"
    _kw = dict(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384, depth=12, num_heads=12,
               num_mask_tokens=10, mask_mode="ar", ar_feedback=_pc.get("ar_feedback", "output"))
    for _k in ("gating_residual", "gating_anchor", "gating_out_ln"):
        if _pc.get(_k) is not None: _kw[_k] = _pc[_k]
    if KIND == "ctx_state":     # 영상 문맥 (online encoder 가 문맥 블록을 한 번에) → 상태 rollout → 블록 공간 render
        from app.vjepa_frozen import utils as U
        arp = U.build_predictor(_c["model"], 1280, None, torch.device("cpu"), int(_c["data"].get("n_frames", 64)))
        if _pc.get("ctx_input", "video") != "video" or _c["model"].get("context_encoder_key", "encoder") != "encoder":
            raise ValueError("ctx_state 는 지금 ctx_input=video · context_encoder_key=encoder 만")
        ctx_enc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=32, use_rope=True, uniform_power=False)   # 채점과 같은 32 장 창
        _load_state_dict(ctx_enc, _load_checkpoint(CK)["encoder"], tag="c"); ctx_enc = ctx_enc.to(dev).eval()
    else:
        arp = vit_prefix_predictor(**_kw)
    _sd = {k: v for k, v in _clean_backbone_key(torch.load(os.path.join(_D, a.ckpt), map_location="cpu", weights_only=False)["predictor"]).items()
           if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}
    arp.load_state_dict(_sd, strict=True); arp = arp.to(dev).eval()
st = torch.load(a.decoder, map_location="cpu", weights_only=False)
mc = st["cfg"]["model"]
dec = PixelDecoder(in_dim=1280, dim=mc["dim"], depth=mc["depth"], heads=mc["heads"]).to(dev).eval()
dec.load_state_dict(st["decoder"])
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint   # noqa: E402
enc2 = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(enc2, _load_checkpoint(CK)["target_encoder"], tag="t"); enc2 = enc2.to(dev).eval()

T, C, S, G = 32, 16, 256, 16
IDX = [3 * k for k in range(T)] if a.set not in ("intphys1_late_uni", "intphys2_uni") else list(range(T))   # uni: 쌍마다 고른 32 장이 이미 000000..000031 (build.py --uniform-d)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(3, 1, 1, 1)
ci, ti = _context_target_indices(ctx_frames=C, tgt_frames=T - C, tubelet_size=2, spatial_tokens=S, batch_size=1, device=dev)
LN = lambda t: F.layer_norm(t, (t.size(-1),))


def load(d):
    if a.set == "phys3d_val":      # mp4 (12 fps 렌더) → 간격 3 (4 fps) 로 32 장, raw 0..93
        import decord
        fr = list(decord.VideoReader(d, num_threads=2).get_batch(IDX).asnumpy())
    elif a.set == "intphys2_uni":  # mp4 + 행마다 고른 원본 프레임 번호 (I2_FRAMES[경로])
        import decord
        fr = list(decord.VideoReader(d, num_threads=2).get_batch(I2_FRAMES[d]).asnumpy())
    else:
        fr = [np.asarray(Image.open(f"{d}/{i:06d}.png").convert("RGB")) for i in IDX]
    x = torch.from_numpy(np.stack(fr)).permute(3, 0, 1, 2).float().to(dev) / 255.0           # (3, T, H, W)
    x = F.interpolate(x.transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False, antialias=False).transpose(0, 1)
    return x                                                                                  # [0, 1]


@torch.no_grad()
def decode(tok):
    """tok (nb·256, D) LN 토큰 → (nb·2, 3, 256, 256) [0, 1]"""
    nb = tok.size(0) // S
    y = dec(tok.view(nb, S, -1).float())                                                       # (nb, 3, 2, H, W)
    y = (y * STD.view(1, 3, 1, 1, 1) + MEAN.view(1, 3, 1, 1, 1)).clamp(0, 1)
    return y.permute(0, 2, 1, 3, 4).reshape(nb * 2, 3, 256, 256)


psnr = lambda y, x: float(-10 * torch.log10(((y - x) ** 2).mean()))
UNI_ROOT = "/workspace/world_model/local_datasets/world/world_analysis/IntPhys1_late_uni"
P3_ROOT = "/workspace/dataset/training_data/phys3d_single_v1/val/clips"
I2_ROOT = a.i2_root or "/workspace/world_model/local_datasets/world/world_analysis/IntPhys2_violation_uni"
if a.set == "intphys2_uni":        # 2026-10-22: IntPhys2 Main, 문맥 [0, d−1] 균일 16 · 미래 [d, N−1] 균일 16 (위반 시작 d 는 사람 라벨), 가능 영상
    # 2026-10-22: mp4 직독 — index_mp4.csv 의 frames (원본 프레임 번호 32 개) 를 그대로 읽는다 (라벨이 바뀌면 csv 만 다시 만든다). 저장 이름 = scene<번호>_pair<p>_<위반 종류>_<pos|imp>
    rows = [dict(r, file_name=os.path.join("/workspace/dataset/benchmarks/IntPhys2/Main", r["file"]), objset=r["block_type"], violation=r["difficulty"],
                 out_name=f"scene{r['scene']}_pair{r['pair']}_{r['block_type']}_{r['variant']}", vid=r["video_id"])
            for r in csv.DictReader(open(os.path.join(REPO, "data_csv/intphys2_violation_uni/index_mp4.csv"))) if a.i2_variant in ("both", r["variant"])]
elif a.set == "phys3d_val":          # 2026-10-21: 3D 합성 검증 clip (학습에 안 씀), 96 장 clip 만, 시나리오별
    import glob as _glob
    rows = [dict(file_name=os.path.relpath(f, P3_ROOT), objset=os.path.basename(os.path.dirname(f)), violation="")
            for f in sorted(_glob.glob(os.path.join(P3_ROOT, "*", "*.mp4"))) if np.load(f[:-4] + "_states.npz")["pos"].shape[0] >= 96]
elif a.set == "intphys1_late_uni":   # 2026-10-21: IntPhys1 late (uniform-d) — 문맥 [0, d−1] 균일 16 · 미래 [d, 99] 균일 16, 가능 영상만
    rows = [dict(r, file_name=r["video_id"], objset=r["group"], violation=r["principle"])
            for r in csv.DictReader(open(os.path.join(REPO, "data_csv/intphys1_late_uni/index.csv"))) if r["plausible"] == "1"]
elif a.set != "gravity":     # VOBench 가림 조건 (late · visible · early · mid), 가능 변이 pos_a
    rows = [r for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv"))) if r["occlusion"] == a.set and r["variant"] == "pos_a"
            and (a.motion is None or r["motion"] == a.motion)]
else:
    rows = [r for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1_gravity/index.csv"))) if r["plausible"] == "1"]
ROOT = I2_ROOT if a.set == "intphys2_uni" else P3_ROOT if a.set == "phys3d_val" else UNI_ROOT if a.set == "intphys1_late_uni" else a.root if a.set != "gravity" else os.path.join(a.root, "gravity")      # gravity 세트는 VOBench_v1/gravity 아래
if a.set == "intphys2_uni" and a.i2_camera:
    rows = [r for r in rows if r["camera"] == a.i2_camera]
if a.set == "intphys2_uni" and a.i2_test:
    _te = {r["block_id"] for r in csv.DictReader(open(os.path.join(REPO, "data_csv/intphys2_violation_uni/index_test.csv")))}
    rows = [r for r in rows if r["block_id"] in _te]
if a.only:
    rows = [r for r in rows if any(r.get("vid", r["file_name"]).startswith(x) or r.get("out_name", "").startswith(x) for x in a.only.split(","))]
I2_FRAMES = {r["file_name"]: [int(t) for t in r["frames"].split()] for r in rows} if a.set == "intphys2_uni" else {}
random.Random(0).shuffle(rows)
by = {}
for r in rows:                     # 위반 · objset 마다 고르게
    k = (r.get("objset"), r.get("violation") or r.get("law"))
    by.setdefault(k, []).append(r)
pick = []
while len(pick) < a.n and any(by.values()):
    for k in sorted(by):
        if by[k] and len(pick) < a.n:
            pick.append(by[k].pop())
NAME = a.name or (AR_RUN or "release")
out = os.path.join(os.path.dirname(a.decoder), "renders", NAME, a.set + (f"_{a.motion}" if a.motion else "") + (f"_{a.i2_camera.lower()}" if a.i2_camera else "")); os.makedirs(out, exist_ok=True)   # 모델별 폴더 (10-20)


def last_ctx_pos(clip_dir):
    """문맥 마지막 장 (원본 IDX[C-1]) 의 물체 중심 (256² 좌표) — label.json (VOBench: A · B, gravity: P · F). 없으면 None.
    가려져 있어도 라벨은 참 위치다 (hidden 표시만)."""
    if a.set == "intphys2_uni":
        return None
    if a.set == "phys3d_val":      # 라벨 = _states.npz (256² 좌표 그대로)
        st = np.load(clip_dir[:-4] + "_states.npz")
        q = st["pos"][IDX[C - 1], 0]
        return (float(q[0]), float(q[1]), bool(st["hidden"][IDX[C - 1], 0])) if np.isfinite(q).all() else None
    lp = os.path.join(os.path.dirname(clip_dir), "label.json")
    if not os.path.isfile(lp):
        return None
    L = json.load(open(lp))["labels"]
    key = next((k for k in ("A", "P") if k in L), next(iter(L), None))
    q = {e["f"]: e for e in L.get(key, [])}.get(IDX[C - 1])
    if q is None:
        return None
    W0 = Image.open(f"{clip_dir}/{IDX[0]:06d}.png").size[0]
    return q["u"] * 256 / W0, q["v"] * 256 / W0, bool(q.get("hidden", 0) > 0.5)
summ = []
for r in pick:
    x01 = load(os.path.join(ROOT, r["file_name"]))                                          # (3, T, H, W)
    xn = ((x01 - MEAN) / STD).unsqueeze(0)
    if AR_RUN is None:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            z = bundle.context_encoder(xn, masks=[ci])
            p = bundle.predictor(z, ci, ti, mask_index=0)
            h = bundle.target_encoder(xn)
            blk = enc2(xn[0].view(3, T // 2, 2, 256, 256).permute(1, 0, 2, 3, 4))                    # 블록별 (16, 256, D)
        p = p.float()[0]; hwin = LN(h.float())[0][C // 2 * S:]; hblk = LN(blk.float()).reshape(-1, blk.size(-1))[C // 2 * S:]
    else:                   # AR: 정답 · 입력 모두 블록 공간. 'win' 칸에는 블록 정답을 그린다 (window 는 AR 과 무관)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            blk = enc2(xn[0].view(3, T // 2, 2, 256, 256).permute(1, 0, 2, 3, 4))
            hb = LN(blk.float()).reshape(-1, blk.size(-1))                                            # (16·256, D)
            idx = torch.arange(C // 2 * S, device=dev).unsqueeze(0)
            if KIND == "ctx_state":
                z = ctx_enc(xn, masks=[idx])                                                          # 문맥 블록 8 개 (영상 한 번에, LN 없음 — 학습과 같다)
                p = arp.render(arp.rollout_state_kv(z, idx, (T - C) // 2), None).float()[0]
            else:
                p = arp.rollout_kv(hb[: C // 2 * S].unsqueeze(0), idx, (T - C) // 2).float()[0]
        hblk = hb[C // 2 * S:]; hwin = hblk
    real = x01[:, C:].permute(1, 0, 2, 3)                                                      # (16, 3, H, W)
    yp, yw, yb = decode(p), decode(hwin), decode(hblk)
    rec = {"clip": r["file_name"], "psnr_blk": psnr(yb, real), "psnr_win": psnr(yw, real), "psnr_pred": psnr(yp, real),
           "lat_l1_pred_vs_win": float((p - hwin).abs().mean()), "lat_l1_blk_vs_win": float((hblk - hwin).abs().mean())}
    summ.append(rec)
    dot = last_ctx_pos(os.path.join(ROOT, r["file_name"]))
    ctx = x01[:, :C].permute(1, 0, 2, 3)
    rows3 = [x01.permute(1, 0, 2, 3), torch.cat([ctx, yp]), torch.cat([ctx, yw])]
    frames = []
    for t in range(T):
        im = torch.cat([rr[t] for rr in rows3], -1)                                           # 가로로 3 칸
        im = Image.fromarray((im.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        dr = ImageDraw.Draw(im)
        tag = "context" if t < C else "future"
        for j, lab in enumerate(["original", f"{NAME} pred ({tag})", f"GT {'window' if AR_RUN is None else 'block'} latent ({tag})"]):
            dr.rectangle([j * 256, 0, j * 256 + 180, 14], fill=(0, 0, 0)); dr.text((j * 256 + 3, 1), lab, fill=(255, 255, 0) if t >= C and j else (255, 255, 255))
        dr.text((3, 242), f"frame {IDX[t]}", fill=(255, 255, 255))
        if dot is not None and t >= C - 1:       # 문맥 마지막 장의 물체 위치 (주황 점, 가려져 있었으면 속 빈 원) — 세 칸 모두
            u, v, hid = dot
            for j in range(3):
                bb = [j * 256 + u - 4, v - 4, j * 256 + u + 4, v + 4]
                dr.ellipse(bb, outline=(255, 140, 0), width=2) if hid else dr.ellipse(bb, fill=(255, 140, 0), outline=(0, 0, 0))
        frames.append(im)
    name = r.get("out_name") or r["file_name"].replace(".mp4", "").replace("/", "__")
    frames[0].save(os.path.join(out, f"{name}.gif"), save_all=True, append_images=frames[1:], duration=125, loop=0)
    print(f"{r['file_name']}: PSNR blk {rec['psnr_blk']:.1f} · win {rec['psnr_win']:.1f} · pred {rec['psnr_pred']:.1f} dB"
          f" | latent L1 pred↔win {rec['lat_l1_pred_vs_win']:.3f} · blk↔win {rec['lat_l1_blk_vs_win']:.3f}", flush=True)
m = lambda k: float(np.mean([s_[k] for s_ in summ]))
print(f"\n평균 ({len(summ)} clip) PSNR blk {m('psnr_blk'):.2f} · win {m('psnr_win'):.2f} · pred {m('psnr_pred'):.2f} dB")
json.dump({"set": a.set, "decoder": a.decoder, "clips": summ}, open(os.path.join(out, "summary.json"), "w"), indent=1, ensure_ascii=False)
print("GIF:", out)
