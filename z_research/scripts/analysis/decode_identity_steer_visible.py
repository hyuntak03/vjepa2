"""정체 개념을 키운 AR 예측을 pixel decoder 로 — VOBench 보이는 조건 (문맥 끝까지 물체가 보임), 물체 주변 확대 (2026-10-26 저녁, GPU 1 장 · 6 GB).
사용자: "물체 안 보이는데..." — 끝 가림 예시는 문맥 끝에 물체가 가림막 뒤라 예측에 물체가 안 나온다. 보이는 조건으로 다시.
VOBench visible 의 pos_a 영상을 바로 인코딩: 원본 0, 3, …, 93 장 (512² → 256² bilinear), 블록 (2 장) 마다 LN(target_encoder) (ar_layer_cache.py 와 같은 방식).
이후는 decode_identity_steer.py 와 같다: 블록 AR 굴림 → 문맥 물체 정체 개념 (3D 합성 AR 예측 물체 칸 개념, 모양 + 색) 4 배를 라벨 A 물체 칸에 더함 → decoder (vob_grav_i1_i2_vitb_both).
그림: 예시마다 한 줄, [정답 latent 복원 · AR 예측 · 예측 + 4c] × 걸음 1 · 4 · 8, 진짜 위치와 마지막 위치를 둘 다 담는 정사각형으로 잘라 확대. ○ 진짜, × 마지막에 본 위치.
  CUDA_VISIBLE_DEVICES=0 MEM_GB=6 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/decode_identity_steer_visible.py
출력: z_analysis/figures/decode_identity_steer.png (끝 가림판을 대체) · z_analysis/results/decode_identity_steer_visible.json
"""
import csv, json, os, sys
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

REPO = "/workspace/world_model/vjepa2"; sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "auto_research/scripts")); sys.path.insert(0, os.path.join(REPO, "z_pixel_decoder"))
dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEM_GB", 6)) * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory, 0)
_argv = list(sys.argv); sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT
sys.argv = _argv
from model import PixelDecoder
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint
S, C, K, D = 256, 8, 8, 1280
SHAPES = ["sphere", "cube", "cylinder", "cone", "pyramid", "torus"]; COLORS = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta"]
VROOT = "/workspace/dataset/benchmarks/VOBench_v1"; IDX = list(range(0, 96, 3))[:32]; STEPS = [1, 4, 8]; BETA = 4.0

OC = np.load("/workspace/auto_research_cache/objcell_concept/ar_train.npz"); _meta = {m["i"]: m for m in json.load(open("/workspace/auto_research_cache/ar_layers/train_meta.json"))}
_sh = np.array([_meta[int(i)]["shape"] for i in OC["i"]]); _co = np.array([_meta[int(i)]["color"] for i in OC["i"]]); _PT = OC["pred_true"].astype(np.float32)
CON = {}
for k in range(1, 9):
    X = _PT[:, k - 1]; m = ~np.isnan(X).any(1); X, s_, c_ = X[m], _sh[m], _co[m]
    CON[k] = (np.stack([X[s_ == i].mean(0) for i in range(6)]) - X.mean(0), np.stack([X[c_ == i].mean(0) for i in range(8)]) - X.mean(0))
cvec = lambda obj, k: (CON[k][0][SHAPES.index(obj.split("/")[0])] if obj.split("/")[0] in SHAPES else 0) + (CON[k][1][COLORS.index(obj.split("/")[1])] if obj.split("/")[1] in COLORS else 0)

# 예시: 보이는 조건 · 도형 · pos_a, 정지 2 · 이동 2 (모양 · 색이 3D 합성에 있는 것)
rows_csv = [r for r in csv.DictReader(open(f"{REPO}/data_csv/vobench_v1/index.csv")) if r["occlusion"] == "visible" and r["objset"] == "primitive" and r["variant"] == "pos_a"
            and r["violation"] in ("shape", "color") and r["obj_a"].split("/")[0] in SHAPES]
pick = []
for mot, n in (("flat_static", 2), ("flat_vel", 1), ("ramp_down", 1)):
    pick += [r for r in rows_csv if r["motion"] == mot][:n]

enc2 = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
_load_state_dict(enc2, _load_checkpoint(CK)["target_encoder"], tag="t"); enc2 = enc2.half().to(dev).eval()
ar = FT.load_pred("natural_ar_vw_state", "latest.pt").to(dev).eval()
st = torch.load(f"{REPO}/z_pixel_decoder/runs/vob_grav_i1_i2_vitb_both/latest.pt", map_location="cpu", weights_only=False); mc = st["cfg"]["model"]
dec = PixelDecoder(in_dim=1280, dim=mc["dim"], depth=mc["depth"], heads=mc["heads"]).to(dev).eval(); dec.load_state_dict(st["decoder"]); del st
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(3, 1, 1, 1)
decode = lambda tok: (dec(tok.float()) * STD.view(1, 3, 1, 1, 1) + MEAN.view(1, 3, 1, 1, 1)).clamp(0, 1)[:, :, 0].permute(0, 2, 3, 1).cpu().numpy()

out_rows, res = [], {}
for r in pick:
    folder = os.path.join(VROOT, r["file_name"]); W0 = Image.open(os.path.join(folder, "000000.png")).size[0]; sc = 256 / W0
    fr = np.stack([np.asarray(Image.open(os.path.join(folder, f"{f:06d}.png")).convert("RGB")) for f in IDX])          # (32, H, W, 3)
    x = torch.from_numpy(fr).permute(3, 0, 1, 2).float().to(dev) / 255.0
    x = F.interpolate(x.permute(1, 0, 2, 3), size=(256, 256), mode="bilinear", align_corners=False, antialias=False).permute(1, 0, 2, 3)
    xn = (x - MEAN) / STD
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        hb = torch.cat([F.layer_norm(enc2(xn[:, 4 * q: 4 * q + 4].reshape(3, 2, 2, 256, 256).permute(1, 0, 2, 3, 4).half()).float(), (D,)) for q in range(8)])   # (16, 256, 1280)
        p = ar.rollout_kv(hb[:C].reshape(1, C * S, D), torch.arange(C * S, device=dev).unsqueeze(0), K).float()[0].view(K, S, D)
    lab = {q["f"]: q for q in json.load(open(os.path.join(os.path.dirname(folder), "label.json")))["labels"].get("A", [])}
    cc = np.arange(16) * 16 + 8.0; mask = torch.zeros(K, S, 1, device=dev); centers = {}
    for j in range(K):
        b = 8 + j; q = [lab.get(IDX[2 * b]), lab.get(IDX[2 * b + 1])]
        if any(t is None for t in q):
            continue
        x0 = min(t["u"] - t["ext_u"] for t in q) * sc; x1 = max(t["u"] + t["ext_u"] for t in q) * sc; y0 = min(t["v"] - t["ext_v"] for t in q) * sc; y1 = max(t["v"] + t["ext_v"] for t in q) * sc
        cells = [int(rr * 16 + c) for rr in np.nonzero((cc >= y0) & (cc <= y1))[0] for c in np.nonzero((cc >= x0) & (cc <= x1))[0]]
        mask[j, cells or [int(min(15, (y0 + y1) / 2 // 16) * 16 + min(15, (x0 + x1) / 2 // 16))]] = 1.0
        centers[j + 1] = ((x0 + x1) / 2, (y0 + y1) / 2)
    c = torch.from_numpy(np.stack([cvec(r["obj_a"], k) for k in range(1, 9)])).float().to(dev)
    qv = p + BETA * c[:, None, :] * mask; sel = [k - 1 for k in STEPS]
    with torch.no_grad():
        imgs = {"ground truth (latent decoded)": decode(hb[C:][sel]), "AR prediction": decode(p[sel]), f"AR prediction + {BETA:g}×concept at object cells": decode(qv[sel])}
    last = lab.get(IDX[15]); last_c = (last["u"] * sc, last["v"] * sc) if last else None
    tc = [centers.get(k) for k in STEPS]
    pts = [p_ for p_ in tc + [last_c] if p_ is not None]; cx = np.mean([p_[0] for p_ in pts]); cy = np.mean([p_[1] for p_ in pts])
    half = max(40, 0.6 * max(max(abs(p_[0] - cx), abs(p_[1] - cy)) for p_ in pts) + 28)
    box = (int(max(0, cx - half)), int(max(0, cy - half)), int(min(256, cx + half)), int(min(256, cy + half)))
    err = {nm: [round(float(np.abs(im[jj] - imgs["ground truth (latent decoded)"][jj])[box[1]:box[3], box[0]:box[2]].mean()), 4) for jj in range(len(STEPS))] for nm, im in imgs.items()}
    title = f"{r['obj_a']} — {r['motion']}"; res[title] = {"crop_L1_vs_gt_by_step": err, "steps": STEPS, "file": r["file_name"]}
    out_rows.append((title, imgs, tc, last_c, box)); print(title, err, flush=True)
json.dump(res, open(f"{REPO}/z_analysis/results/decode_identity_steer_visible.json", "w"), indent=1)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"
names = list(out_rows[0][1].keys()); ncol = len(names) * len(STEPS)
fig, axes = plt.subplots(len(out_rows), ncol, figsize=(1.9 * ncol, 2.1 * len(out_rows) + 0.7))
for ri, (title, imgs, tc, last_c, box) in enumerate(out_rows):
    x0, y0, x1, y1 = box
    for ni, nm in enumerate(names):
        for si, k in enumerate(STEPS):
            ax = axes[ri, ni * len(STEPS) + si]; ax.imshow(imgs[nm][si][y0:y1, x0:x1], extent=(x0, x1, y1, y0)); ax.set_xticks([]); ax.set_yticks([])
            if tc[si] is not None:
                ax.plot(*tc[si], "o", mfc="none", mec="white", ms=16, mew=1.8)
            if last_c is not None:
                ax.plot(*last_c, "x", color="yellow", ms=9, mew=1.8)
            ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
            if ri == 0:
                ax.set_title(f"{nm}\nstep {k} ({0.5 * k:.1f} s)" if si == 1 else f"step {k} ({0.5 * k:.1f} s)", fontsize=8)
    axes[ri, 0].set_ylabel(title, fontsize=8)
fig.suptitle("Decoded AR predictions, identity concept amplified ×4 at object cells (VOBench, object visible; cropped around the object; ○ true, × last seen)", fontsize=11, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(f"{REPO}/z_analysis/figures/decode_identity_steer.png", dpi=150, bbox_inches="tight")
print("saved")
