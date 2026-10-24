#!/usr/bin/env python3
"""고리 1 [E] — encoder 잡음의 크기 · 정체 · 자연 영상 여부를 재기 위한 수집 (auto_research, 2026-10-21).

  CUDA_VISIBLE_DEVICES=2 /venv/vjepa/bin/python -W ignore auto_research/scripts/enc_noise_collect.py \
      --source k710 --n 160 --shard 0 --nshard 2

창 = 2·nb 장 (블록 nb 개, 블록 = 연속 2 장), 학습 · 채점과 같은 간격:
  k710 · ssv2 · ht100m : data_csv/<표>/fps4_table.tsv 의 fstp (= 4 fps, 학습과 같음). 가운데 정사각 → 256 bilinear → uint8 반올림 (학습 로더와 같음)
  ek100                : videos_256p, 간격 round(fps/4)
  vobench              : VOBench_v1 PNG 512² 30 fps, 간격 3 (채점과 같음), 장 0..3(2nb−1)
  phys3d               : 캐시 pix (간격 3)
인코딩 = 블록별 LN(enc(블록)). **bf16** = 학습 정답과 같은 처리 (가중치 bf16 · 입력 bf16 · autocast bf16), **fp32** = 가중치 · 입력 fp32 (TF32 끔).
변형 (블록마다 같은 처리, 학습 정답 = h0):
  h0   원본 bf16                      hf   원본 fp32
  hn1/hn2  서로 다른 ±1/255 정수 잡음 (균등 {−1,0,1}, 화소 · 채널 · 장마다 독립), bf16  → 반복 잡음 |hn1 − hn2|
  c18/c28  같은 프레임을 libx264 CRF 18 / 28 · yuv420p 로 재압축해 다시 푼 것, bf16
  bright   블록 두 장 밝기 +1/255, bf16          shift  0.5 px 가로 이동 (bilinear), bf16
  hT   (fp32) 블록 b 의 "정지 칸" (블록 b−1 과 칸 평균 |Δ픽셀| < --thr) 픽셀을 블록 b−1 값으로 얼린 것 → 칸별
       leak  = |hT_b − hf_{b−1}|  (그 칸 픽셀은 그대로인데 다른 칸 변화가 번져 온 몫)
       local = |hf_b − hT_b|     (그 칸 자신의 미세 픽셀 변화가 만든 몫)
저장: <out>/<source>_<enc>_s<shard>.npz (칸별 요약) + _tok.npy (h0 · hf · hn1 · hn2 · c18 토큰 fp16, --save_tok) + _pix.npy (uint8)."""
import argparse, csv, glob, json, os, random, subprocess, sys, time
import numpy as np, torch, torch.nn.functional as F
import decord

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True, choices=["k710", "ssv2", "ht100m", "ek100", "vobench", "phys3d"])
ap.add_argument("--n", type=int, default=160, help="전체 창 수 (shard 로 나눔)")
ap.add_argument("--nb", type=int, default=8)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--enc", default="target_encoder", choices=["target_encoder", "encoder", "random"])
ap.add_argument("--thr", type=float, default=2.0, help="정지 칸 문턱: 칸 평균 |Δ픽셀| (0–255)")
ap.add_argument("--save_tok", type=int, default=1)
ap.add_argument("--light", type=int, default=0, help="1 이면 h0 · hf · hn1 · hn2 · hT 만 (기준 encoder 비교용)")
ap.add_argument("--out", default="/workspace/auto_research_cache/enc_noise")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
dev = torch.device("cuda")
G, R, T = 16, 256, 2 * a.nb
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
TMP = f"/dev/shm/enc_noise_{os.getpid()}"
os.makedirs(TMP, exist_ok=True)


# ---------------------------------------------------------------- encoder
torch.set_num_threads(8)
SD = None if a.enc == "random" else _load_checkpoint(CK)[a.enc]


def build():
    torch.manual_seed(0)
    e = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
    if SD is not None:
        _load_state_dict(e, SD, tag="t")
    return e.eval()


E32 = build().to(dev)
E16 = build().to(dev, torch.bfloat16)
del SD


@torch.no_grad()
def encode(px, mode):
    """px (n_blocks, 2, R, R, 3) uint8 → (n_blocks, 256, D) float32, LN."""
    x = (px.float().div(255).permute(0, 4, 1, 2, 3) - MEAN) / STD                # (n,3,2,R,R)
    out = []
    for i in range(0, x.size(0), 8):
        xi = x[i:i + 8]
        if mode == "bf16":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = E16(xi.to(torch.bfloat16))
        else:
            h = E32(xi.float())
        out.append(F.layer_norm(h.float(), (h.size(-1),)))
    return torch.cat(out)


def blocks(fr):                      # (T,R,R,3) → (nb,2,R,R,3)
    return fr.view(a.nb, 2, R, R, 3)


# ---------------------------------------------------------------- data
def to_square_uint8(fr):             # fr (T,H,W,3) uint8 numpy → torch uint8 (T,R,R,3) 학습 로더와 같은 처리 (가운데 정사각, r = 1)
    t = torch.from_numpy(fr)
    H, W = t.shape[1:3]
    s = min(H, W); y0, x0 = (H - s) // 2, (W - s) // 2
    t = t[:, y0:y0 + s, x0:x0 + s]
    if s != R:
        x = t.permute(0, 3, 1, 2).float().to(dev)
        x = F.interpolate(x, size=(R, R), mode="bilinear", align_corners=False)
        t = x.round_().clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
    return t.to(dev).contiguous()


def windows():
    rng = random.Random(f"{a.seed}-{a.source}")
    if a.source in ("k710", "ssv2", "ht100m"):
        tab = {"k710": "k710_320", "ssv2": "ssv2", "ht100m": "ht100m_1pct"}[a.source]
        rows = list(csv.reader(open(os.path.join(REPO, f"data_csv/{tab}/fps4_table.tsv")), delimiter="\t"))
        rng.shuffle(rows)
        got = 0
        for r in rows:
            path, nfr, sd = r[0], int(r[2]), int(r[3])
            if nfr < sd * (T - 1) + 1:
                continue
            s0 = rng.randrange(0, nfr - sd * (T - 1))
            if got % a.nshard == a.shard:
                yield dict(path=path, start=s0, stride=sd)
            got += 1
            if got >= a.n:
                return
    elif a.source == "ek100":
        paths = sorted(glob.glob("/workspace/dataset/benchmarks/EK100/videos_256p/videos/*.mp4"))
        rng.shuffle(paths)
        for i in range(a.n):
            if i % a.nshard == a.shard:
                yield dict(path=paths[i % len(paths)], start=None, stride=None, k=i)
    elif a.source == "vobench":
        rows = [r for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv"))) if r["variant"] == "pos_a"]
        seen, uniq = set(), []
        for r in rows:
            if r["file_name"] not in seen:
                seen.add(r["file_name"]); uniq.append(r)
        rng.shuffle(uniq)
        for i, r in enumerate(uniq[: a.n]):
            if i % a.nshard == a.shard:
                yield dict(path=os.path.join("/workspace/dataset/benchmarks/VOBench_v1", r["file_name"]), start=0, stride=3,
                           motion=r["motion"], occlusion=r["occlusion"], objset=r["objset"])
    elif a.source == "phys3d":
        files = sorted(glob.glob("/workspace/auto_research_cache/phys3d/pix_val_s*.npy"))
        items = [(f, w) for f in files for w in range(np.load(f, mmap_mode="r").shape[0])]
        rng.shuffle(items)
        for i, (f, w) in enumerate(items[: a.n]):
            if i % a.nshard == a.shard:
                yield dict(path=f, start=w, stride=3)


def load(wd):
    if a.source in ("k710", "ssv2", "ht100m", "ek100"):
        vr = decord.VideoReader(wd["path"], num_threads=2)
        if a.source == "ek100":
            sd = max(1, int(round(vr.get_avg_fps() / 4)))
            rng = random.Random(f"{a.seed}-ek-{wd['k']}")
            s0 = rng.randrange(0, len(vr) - sd * (T - 1))
            wd.update(start=s0, stride=sd)
        idx = [wd["start"] + wd["stride"] * k for k in range(T)]
        if idx[-1] >= len(vr):
            return None
        return to_square_uint8(vr.get_batch(idx).asnumpy())
    if a.source == "vobench":
        from PIL import Image                                      # cv2 의 libpng 가 이 PNG 를 못 읽는다
        fr = []
        for k in range(T):
            fr.append(np.asarray(Image.open(os.path.join(wd["path"], f"{wd['start'] + wd['stride'] * k:06d}.png")).convert("RGB")))
        return to_square_uint8(np.ascontiguousarray(np.stack(fr)))
    P = np.load(wd["path"], mmap_mode="r")
    return torch.from_numpy(np.ascontiguousarray(P[wd["start"]][:T])).to(dev)


def recompress(fr, crf):             # fr (T,R,R,3) uint8 torch → libx264 CRF, yuv420p, 다시 풀어 uint8
    raw = fr.cpu().numpy().tobytes()
    f = os.path.join(TMP, f"c{crf}.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{R}x{R}", "-r", "4", "-i", "-",
                    "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", f], input=raw, check=True)
    out = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", f, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return torch.from_numpy(np.frombuffer(out, np.uint8).reshape(T, R, R, 3).copy()).to(dev)


def cellmean(x):                      # (..., R, R) → (..., 256)  칸 평균
    s = x.shape[:-2]
    return x.reshape(*s, G, 16, G, 16).mean(dim=(-3, -1)).reshape(*s, G * G)


def lat(a_, b_):                      # 칸별 |Δh| 차원 평균 → (..., 256)
    return (a_ - b_).abs().mean(-1)


# ---------------------------------------------------------------- main loop
gen = torch.Generator(device=dev).manual_seed(1000 + a.shard)
S = {k: [] for k in ("pixD", "pixW", "d0", "dfp", "rep", "bf", "c18", "c28", "c1828", "bright", "shift",
                     "pix_c18", "pix_c28", "pix_shift", "leak", "local", "frozen", "vg0", "vgf", "vg0_sq", "norm_pre")}
TOK, PIX, META = [], [], []
for i, wd in enumerate(windows()):
    try:
        fr = load(wd)
    except Exception as ex:   # noqa: BLE001
        print(f"skip {wd['path']}: {ex}", flush=True)
        continue
    if fr is None:
        continue
    frf = fr.float()
    # 픽셀 변화 (0–255, 칸 평균)
    bl = frf.view(a.nb, 2, R, R, 3)
    pixD = cellmean((bl[1:] - bl[:-1]).abs().mean(-1).mean(1))             # (nb−1,256)
    pixW = cellmean((bl[:, 1] - bl[:, 0]).abs().mean(-1))                  # (nb,256)
    h0 = encode(blocks(fr), "bf16")
    hf = encode(blocks(fr), "fp32")
    n1 = torch.randint(-1, 2, fr.shape, device=dev, generator=gen); n2 = torch.randint(-1, 2, fr.shape, device=dev, generator=gen)
    hn1 = encode(blocks((frf + n1).clamp(0, 255).to(torch.uint8)), "bf16")
    hn2 = encode(blocks((frf + n2).clamp(0, 255).to(torch.uint8)), "bf16")
    # 정지 칸 얼리기 (fp32): 블록 b (≥1) 의 정지 칸 픽셀 ← 블록 b−1
    frozen = (pixD < a.thr)                                                # (nb−1,256)
    m = frozen.view(a.nb - 1, G, 1, G, 1).expand(-1, -1, 16, -1, 16).reshape(a.nb - 1, 1, R, R, 1)
    xT = torch.where(m, bl[:-1], bl[1:]).round().to(torch.uint8)            # (nb−1,2,R,R,3)
    hT = encode(xT, "fp32")
    S["leak"].append(lat(hT, hf[:-1]).cpu()); S["local"].append(lat(hf[1:], hT).cpu()); S["frozen"].append(frozen.cpu())
    S["pixD"].append(pixD.cpu()); S["pixW"].append(pixW.cpu())
    S["d0"].append(lat(h0[1:], h0[:-1]).cpu()); S["dfp"].append(lat(hf[1:], hf[:-1]).cpu())
    S["rep"].append(lat(hn1, hn2).cpu()); S["bf"].append(lat(h0, hf).cpu())
    # 변이도 (lag k = 1..nb−1): 칸별 평균 |h_{b+k} − h_b| · 제곱
    S["vg0"].append(torch.stack([lat(h0[k:], h0[:-k]).mean(0) for k in range(1, a.nb)]).cpu())
    S["vgf"].append(torch.stack([lat(hf[k:], hf[:-k]).mean(0) for k in range(1, a.nb)]).cpu())
    S["vg0_sq"].append(torch.stack([(h0[k:] - h0[:-k]).pow(2).mean(-1).mean(0) for k in range(1, a.nb)]).cpu())
    if not a.light:
        c18, c28 = recompress(fr, 18), recompress(fr, 28)
        hc18 = encode(blocks(c18), "bf16"); hc28 = encode(blocks(c28), "bf16")
        hb = encode(blocks((frf + 1).clamp(0, 255).to(torch.uint8)), "bf16")
        xs = frf.permute(0, 3, 1, 2)
        xs = 0.5 * xs + 0.5 * torch.cat([xs[..., :, :1], xs[..., :, :-1]], -1)          # 0.5 px 오른쪽 이동 (bilinear)
        sh = xs.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
        hs = encode(blocks(sh), "bf16")
        S["c18"].append(lat(hc18, h0).cpu()); S["c28"].append(lat(hc28, h0).cpu()); S["c1828"].append(lat(hc18, hc28).cpu())
        S["bright"].append(lat(hb, h0).cpu()); S["shift"].append(lat(hs, h0).cpu())
        S["pix_c18"].append(cellmean((c18.float() - frf).abs().mean(-1)).view(a.nb, 2, -1).mean(1).cpu())
        S["pix_c28"].append(cellmean((c28.float() - frf).abs().mean(-1)).view(a.nb, 2, -1).mean(1).cpu())
        S["pix_shift"].append(cellmean((sh.float() - frf).abs().mean(-1)).view(a.nb, 2, -1).mean(1).cpu())
        if a.save_tok:
            TOK.append(torch.stack([h0, hf, hn1, hn2, hc18]).half().cpu().numpy())        # (5,nb,256,D)
    elif a.save_tok:
        TOK.append(torch.stack([h0, hf, hn1, hn2]).half().cpu().numpy())
    if a.save_tok:
        PIX.append(fr.cpu().numpy())
    META.append({k: v for k, v in wd.items()})
    if i % 10 == 0:
        print(f"[{a.source}/{a.enc} s{a.shard}] {time.strftime('%H:%M:%S')} {len(META)} 창 · 정지 칸 {frozen.float().mean():.2f} · d0 {S['d0'][-1].mean():.3f} rep {S['rep'][-1].mean():.3f}", flush=True)

os.makedirs(a.out, exist_ok=True)
tag = f"{a.source}_{a.enc}_s{a.shard}"
np.savez(os.path.join(a.out, f"{tag}.npz"), **{k: torch.stack(v).numpy() for k, v in S.items() if v})
json.dump(META, open(os.path.join(a.out, f"{tag}_meta.json"), "w"))
if a.save_tok:
    np.save(os.path.join(a.out, f"{tag}_tok.npy"), np.stack(TOK))
    np.save(os.path.join(a.out, f"{tag}_pix.npy"), np.stack(PIX))
print(f"[{tag}] done {len(META)} 창", flush=True)
