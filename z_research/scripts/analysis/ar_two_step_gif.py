"""블록 AR 을 2 걸음만 굴려 이어 붙인 미래 16 장 — pixel decoder GIF, VOBench 보임 · 움직임 (2026-10-27, CPU).
사용자: "기존 ar 을 2 개만 roll out 해서 최종적으로 16 frame 예측하도록 pixel decoder 로 gif".
  ⚠️ 2026-10-27 정정 — 처음 판은 "문맥을 1 블록씩 밀며 2 걸음째만" 이었다. 사용자가 바로잡음: "블록 6 까지 주고 8 까지 예측, 8 까지 문맥으로 주고 10 까지 예측".
  → 2 블록 단위로 끊어 예측: 실제 0–7 → 8 · 9 / 실제 0–9 → 10 · 11 / 실제 0–11 → 12 · 13 / 실제 0–13 → 14 · 15 (미래 구간을 기존 GIF 와 같은 블록 8–15 로 맞춤).
  덩어리마다 1 걸음째 = 실제 마지막 블록을 넣은 출력, 2 걸음째 = 상태 되먹임 한 번. VOBench 간격: 장 0.1 s, 블록 0.2 s.
  칸: raw video · AR (문맥 8 블록 → 8 걸음 굴림, objectness_overlay_gif.py 의 AR 칸과 같은 방식) · 2step roll out (위의 2 블록 단위 예측).
  2026-10-27 사용자 "1 step roll out 을 하면? raw video · 1 step roll out · 2 step roll out · AR (8 step)" → 칸 4 개.
    1 step roll out = 블록 b 마다 실제 0 … b−1 을 문맥으로 1 걸음 (teacher forcing 출력과 같다).
모델: natural_ar_vw_state (latest) · encoder = V-JEPA 2 ViT-H target encoder (블록 단위, LN) · decoder = z_pixel_decoder vob_grav_i1_i2_vitb_both.
영상: VOBench 영속 · 도형 · pos_a · 보임, 운동 flat_vel · flat_acc · ramp_down · ramp_up 각 1 편 (flat_vel · ramp_down 은 objectness GIF 와 같은 영상).
GPU 를 쓰지 않는다 (2026-10-27 다른 세션 학습이 GPU 0–7 을 거의 다 씀). float32 CPU.
  /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/ar_two_step_gif.py
출력: z_analysis/figures/gifs/ar_steps_moving_visible_<물체>_<운동>.gif (ar2step_* 는 이것으로 대체)
"""
import csv, json, os, sys
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw

REPO = "/workspace/world_model/vjepa2"; sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "auto_research/scripts")); sys.path.insert(0, os.path.join(REPO, "z_pixel_decoder"))
torch.set_num_threads(48); dev = torch.device("cpu")
_argv = list(sys.argv); sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT
sys.argv = _argv
from model import PixelDecoder
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint
S, C, K, D, NB = 256, 8, 8, 1280, 16
VROOT = "/workspace/dataset/benchmarks/VOBench_v1"; IDX = list(range(0, 96, 3))[:32]
OUT = f"{REPO}/z_analysis/figures/gifs"; os.makedirs(OUT, exist_ok=True)

rows = [r for r in csv.DictReader(open(f"{REPO}/data_csv/vobench_v1/index.csv")) if r["objset"] == "primitive" and r["violation"] == "permanence"
        and r["variant"] == "pos_a" and r["obj_b"] in ("", "empty") and r["occlusion"] == "visible"]
pick = [[r for r in rows if r["motion"] == m][0] for m in ("flat_vel", "flat_acc", "ramp_down", "ramp_up")]

enc2 = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
_load_state_dict(enc2, _load_checkpoint(CK)["target_encoder"], tag="t"); enc2 = enc2.float().eval()
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
    return (x.permute(1, 2, 3, 0).numpy() * 255).astype(np.uint8), h


def decode(P):
    with torch.no_grad():
        d = (dec(P) * STD.view(1, 3, 1, 1, 1) + MEAN.view(1, 3, 1, 1, 1)).clamp(0, 1)
    return (d.permute(0, 2, 3, 4, 1).reshape(-1, 256, 256, 3).numpy() * 255).astype(np.uint8)


for r in pick:
    fa = os.path.join(VROOT, r["file_name"]); W0 = Image.open(os.path.join(fa, "000000.png")).size[0]; sc = 256 / W0
    lab = {q["f"]: q for q in json.load(open(os.path.join(os.path.dirname(fa), "label.json")))["labels"].get("A", [])}
    raw, h = load(fa)
    one, two = {}, {}
    with torch.no_grad():
        caches = None
        for j in range(NB - 1):                                        # 실제 블록 j 까지 넣음
            p1, caches, stt = ar._ar_step_kv(h[j][None], idx_all[:, j * S:(j + 1) * S], caches)
            if j >= C - 1:
                one[j + 1] = p1[0]                                     # 1 step: 실제 0 … j → j+1
            if j >= C - 1 and (j - (C - 1)) % 2 == 0 and j + 2 < NB:                     # 덩어리 시작: 실제 0 … j (j = 7, 9, 11, 13) → j+1, j+2
                p2, _, _ = ar._ar_step_kv(stt + T1, idx_all[:, (j + 1) * S:(j + 2) * S], caches, embedded=True)
                two[j + 1] = p1[0]; two[j + 2] = p2[0]
        roll = ar.rollout_kv(h[:C].reshape(1, C * S, D), idx_all[:, :C * S], K)[0].view(K, S, D)
    img1 = decode(torch.stack([one[b] for b in range(C, NB)])); img2 = decode(torch.stack([two[b] for b in range(C, NB)])); imgR = decode(roll)
    frames = []
    for f in range(32):
        b = f // 2; t = lab.get(IDX[f]); tc = (t["u"] * sc, t["v"] * sc) if t is not None else None
        # 2026-10-27 사용자: 칸 이름 "raw video, ar, 2step roll out" — 순서도 원본 · 8 걸음 굴림 · 2 걸음 앞
        panels = [("raw video", raw[f]), ("1 step roll out", raw[f] if f < 16 else img1[f - 16]), ("2 step roll out", raw[f] if f < 16 else img2[f - 16]),
                  ("AR (8 step)", raw[f] if f < 16 else imgR[f - 16])]
        canvas = Image.new("RGB", (len(panels) * 256 + (len(panels) - 1) * 6, 256 + 44), (255, 255, 255)); dr = ImageDraw.Draw(canvas)
        for pi, (nm, img) in enumerate(panels):
            im = Image.fromarray(img); d2 = ImageDraw.Draw(im)
            if tc is not None:
                u, v = tc; d2.ellipse([u - 12, v - 12, u + 12, v + 12], outline=(255, 255, 255), width=2)
            if pi > 0 and f >= 16:
                d2.rectangle([0, 0, 255, 255], outline=(230, 40, 40), width=4)
            ox = pi * 262; canvas.paste(im, (ox, 22)); dr.text((ox + 4, 4), nm, fill=(0, 0, 0))
        dr.text((4, 256 + 26), f"{r['obj_a']}, {r['motion']} | t = {f * 0.1:.1f} s | block {b} | " + ("context" if f < 16 else "prediction") + " | white o: true object", fill=(0, 0, 0))
        frames.append(canvas)
    name = f"ar_steps_moving_visible_{r['obj_a'].replace('/', '_')}_{r['motion']}"
    frames[0].save(f"{OUT}/{name}.gif", save_all=True, append_images=frames[1:], duration=150, loop=0)
    print("saved", name, flush=True)
