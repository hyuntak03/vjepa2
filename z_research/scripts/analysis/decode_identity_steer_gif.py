"""정체 개념 개입 GIF — 원본 영상 · AR 예측 · AR + 4×개념 (3D 합성 개념) · AR + 4×개념 (VOBench 자체 개념), VOBench 보이는 조건 (2026-10-26 저녁, GPU 1 장 · 6 GB).
사용자: "이거 gif 로 뽑아줘. raw video, ar, ar intervention" · "concept vector 자체가 VOBench 랑 3D phys single 이랑 다를 텐데" → VOBench 자체 개념판도 같이.
예시 · 인코딩 · 굴림 · 물체 칸 · decoder 는 decode_identity_steer_visible.py 와 같다.
  3D 합성 개념 = 3D 합성 AR 예측 물체 칸 개념 (걸음별, 모양 + 색)
  VOBench 개념 = VOBench 끝 가림 캐시 encoder 물체 칸 개념 (concept_vobench_vs_phys3d.py, 모양 + 색, 걸음 공통)
GIF: 32 장 (원본 0, 3, …, 93 장 = 4 fps) 을 8 fps 로 (2 배속). 문맥 16 장은 모든 패널이 원본, 예측 16 장 (8 블록 × 2 장) 은 decoder 복원 (빨간 테두리).
  흰 원 = 그 장의 진짜 물체 위치 (라벨), 노란 × = 마지막 문맥 장의 물체 위치.
  CUDA_VISIBLE_DEVICES=0 MEM_GB=6 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/decode_identity_steer_gif.py
출력: z_analysis/figures/gifs/<물체>_<운동>.gif
"""
import csv, json, os, sys
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw

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
VROOT = "/workspace/dataset/benchmarks/VOBench_v1"; IDX = list(range(0, 96, 3))[:32]; BETA = 4.0
OUT = f"{REPO}/z_analysis/figures/gifs"; os.makedirs(OUT, exist_ok=True)

OC = np.load("/workspace/auto_research_cache/objcell_concept/ar_train.npz"); _meta = {m["i"]: m for m in json.load(open("/workspace/auto_research_cache/ar_layers/train_meta.json"))}
_sh = np.array([_meta[int(i)]["shape"] for i in OC["i"]]); _co = np.array([_meta[int(i)]["color"] for i in OC["i"]]); _PT = OC["pred_true"].astype(np.float32)
CON = {}
for k in range(1, 9):
    X = _PT[:, k - 1]; m = ~np.isnan(X).any(1); X, s_, c_ = X[m], _sh[m], _co[m]
    CON[k] = (np.stack([X[s_ == i].mean(0) for i in range(6)]) - X.mean(0), np.stack([X[c_ == i].mean(0) for i in range(8)]) - X.mean(0))
VS = np.load("/workspace/auto_research_cache/objcell_concept/vobench_encoder_concept_shape.npy", allow_pickle=True).item()
VC = np.load("/workspace/auto_research_cache/objcell_concept/vobench_encoder_concept_color.npy", allow_pickle=True).item()
c_phys = lambda obj, k: CON[k][0][SHAPES.index(obj.split("/")[0])] + CON[k][1][COLORS.index(obj.split("/")[1])]
c_vob = lambda obj: VS[obj.split("/")[0]] + VC[obj.split("/")[1]]

rows_csv = [r for r in csv.DictReader(open(f"{REPO}/data_csv/vobench_v1/index.csv")) if r["occlusion"] == "visible" and r["objset"] == "primitive" and r["variant"] == "pos_a"
            and r["violation"] in ("shape", "color") and r["obj_a"].split("/")[0] in SHAPES and r["obj_a"].split("/")[0] in VS]
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


def decode_all(tok):                                                  # (8, 256, 1280) → (16, 256, 256, 3) uint8
    with torch.no_grad():
        y = (dec(tok.float()) * STD.view(1, 3, 1, 1, 1) + MEAN.view(1, 3, 1, 1, 1)).clamp(0, 1)   # (8, 3, 2, H, W)
    return (y.permute(0, 2, 3, 4, 1).reshape(16, 256, 256, 3).cpu().numpy() * 255).astype(np.uint8)


for r in pick:
    folder = os.path.join(VROOT, r["file_name"]); W0 = Image.open(os.path.join(folder, "000000.png")).size[0]; sc = 256 / W0
    fr = np.stack([np.asarray(Image.open(os.path.join(folder, f"{f:06d}.png")).convert("RGB")) for f in IDX])
    x = torch.from_numpy(fr).permute(3, 0, 1, 2).float().to(dev) / 255.0
    x = F.interpolate(x.permute(1, 0, 2, 3), size=(256, 256), mode="bilinear", align_corners=False, antialias=False).permute(1, 0, 2, 3)
    raw = (x.permute(1, 2, 3, 0).cpu().numpy() * 255).astype(np.uint8)                                  # (32, 256, 256, 3)
    xn = (x - MEAN) / STD
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        hb = torch.cat([F.layer_norm(enc2(xn[:, 4 * q: 4 * q + 4].reshape(3, 2, 2, 256, 256).permute(1, 0, 2, 3, 4).half()).float(), (D,)) for q in range(8)])
        p = ar.rollout_kv(hb[:C].reshape(1, C * S, D), torch.arange(C * S, device=dev).unsqueeze(0), K).float()[0].view(K, S, D)
    lab = {q["f"]: q for q in json.load(open(os.path.join(os.path.dirname(folder), "label.json")))["labels"].get("A", [])}
    cc = np.arange(16) * 16 + 8.0; mask = torch.zeros(K, S, 1, device=dev)
    for j in range(K):
        b = 8 + j; q = [lab.get(IDX[2 * b]), lab.get(IDX[2 * b + 1])]
        if any(t is None for t in q):
            continue
        x0 = min(t["u"] - t["ext_u"] for t in q) * sc; x1 = max(t["u"] + t["ext_u"] for t in q) * sc; y0 = min(t["v"] - t["ext_v"] for t in q) * sc; y1 = max(t["v"] + t["ext_v"] for t in q) * sc
        cells = [int(rr * 16 + c) for rr in np.nonzero((cc >= y0) & (cc <= y1))[0] for c in np.nonzero((cc >= x0) & (cc <= x1))[0]]
        mask[j, cells or [int(min(15, (y0 + y1) / 2 // 16) * 16 + min(15, (x0 + x1) / 2 // 16))]] = 1.0
    cp = torch.from_numpy(np.stack([c_phys(r["obj_a"], k) for k in range(1, 9)])).float().to(dev)
    cvb = torch.from_numpy(c_vob(r["obj_a"])).float().to(dev)
    panels = {"raw video": raw[16:], "AR prediction": decode_all(p), "AR + 4x concept (3D synthetic)": decode_all(p + BETA * cp[:, None, :] * mask),
              "AR + 4x concept (VOBench own)": decode_all(p + BETA * cvb[None, None, :] * mask)}
    last = lab.get(IDX[15]); last_c = (last["u"] * sc, last["v"] * sc) if last else None
    frames = []
    for f in range(32):
        canvas = Image.new("RGB", (4 * 256 + 3 * 6, 256 + 40), (255, 255, 255)); dr = ImageDraw.Draw(canvas)
        for pi, (nm, P) in enumerate(panels.items()):
            img = Image.fromarray(raw[f] if f < 16 else P[f - 16]); d2 = ImageDraw.Draw(img)
            if f >= 16:
                t = lab.get(IDX[f])
                if t is not None:
                    u, v = t["u"] * sc, t["v"] * sc; d2.ellipse([u - 12, v - 12, u + 12, v + 12], outline=(255, 255, 255), width=2)
                if last_c is not None:
                    u, v = last_c; d2.line([u - 6, v - 6, u + 6, v + 6], fill=(255, 230, 0), width=2); d2.line([u - 6, v + 6, u + 6, v - 6], fill=(255, 230, 0), width=2)
                if pi > 0:
                    d2.rectangle([0, 0, 255, 255], outline=(230, 40, 40), width=4)
            ox = pi * (256 + 6); canvas.paste(img, (ox, 22)); dr.text((ox + 4, 4), nm, fill=(0, 0, 0))
        dr.text((4, 256 + 25), f"t = {f * 0.25:.2f} s  |  " + ("context (real frames)" if f < 16 else f"prediction step {(f - 16) // 2 + 1}"), fill=(0, 0, 0))
        frames.append(canvas)
    name = f"{r['obj_a'].replace('/', '_')}_{r['motion']}"
    frames[0].save(f"{OUT}/{name}.gif", save_all=True, append_images=frames[1:], duration=125, loop=0)
    print("saved", f"{OUT}/{name}.gif", flush=True)
