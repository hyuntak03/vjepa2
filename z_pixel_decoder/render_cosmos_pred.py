#!/usr/bin/env python3
"""cosmos_vobench_latent.py (IntPhys 2 공식 Cosmos 채점, greedy) 가 저장한 예측 토큰을 픽셀로 — 채점에 쓰인 바로 그 예측 (2026-10-21).

render_future.py 와 같은 꼴 (사용자: "이전에 gif 그리던 것처럼", "context → prediction"):
  GIF  = video (가능 영상) | Cosmos (문맥 9 장은 실제 문맥, 그 뒤 24 장은 예측) | tokenizer enc→dec (정답을 토크나이저로 왕복 = 상한)
  grid = 클립 × [원본 (ledge 는 낙하 · 부유) · tokenizer · Cosmos], 열 = 문맥 끝 2 장 + 미래 9 장. Cosmos 줄에 쌍 surprise (코드북 L1) · 판정 · PSNR
폴더 (사용자: "visible late .. gravity, (static, moving)"):
  <out>/{visible,early,mid,late}/{static,moving}/   VOBench — static = flat_static, moving = flat_vel · flat_acc · flat_dec · ramp_down · ramp_up
  <out>/gravity/{ledge,arc}/
클립: 칸 (위반 × objset) 마다 static 1 · moving 2 (운동을 돌려 가며) · gravity 는 (objset × 칸) 에서 6. 예측이 저장된 문맥에서만 고른다.
생성은 다시 하지 않는다 — 문맥 latent 0–1 은 같은 전처리로 다시 토큰화해 붙이고 decoder 만 돈다.

  source /data/hyuntak/anaconda3/bin/activate cosmos-predict1; cd /data/hyuntak/project/2026/2027_cvpr/cosmos-predict1
  CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$PWD python <repo>/z_pixel_decoder/render_cosmos_pred.py"""
import argparse, collections, csv, glob, json, os, sys
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image, ImageDraw

REPO = "/data/hyuntak/project/2026/2027_cvpr/vjepa2"
ap = argparse.ArgumentParser()
ap.add_argument("--scores", default=f"{REPO}/cvpr/results/eval/testbed/cosmos_ar4b_latent")
ap.add_argument("--root", default=f"{REPO}/z_pixel_decoder/runs/_frames/VOBench_v1")
ap.add_argument("--checkpoint_dir", default="/data/hyuntak/project/2026/2027_cvpr/cosmos-predict1/checkpoints")
ap.add_argument("--out", default=f"{REPO}/z_pixel_decoder/runs/vob_grav_i1_vitb_both/cosmos_ar4b")
ap.add_argument("--only", default="", help="쉼표로 폴더만 (예: late/moving,gravity/ledge)")
a = ap.parse_args()
CTX, FUT = list(range(30, 47, 2)), list(range(48, 95, 2)); IDX = CTX + FUT; NC = len(CTX)
H, W, S = 640, 1024, 256; PADL = (W - H) // 2; LAT = (5, 40, 64)


def loadp(kind):
    p, s = {}, {}
    for f in glob.glob(f"{a.scores}/{kind}_pred_shard*.npz"):
        z = np.load(f); p.update({k: z[k] for k in z.files})
    for f in glob.glob(f"{a.scores}/{kind}_shard*.json"):
        s.update(json.load(open(f))["surprise"])
    return p, s


# ---- 클립 고르기 ---------------------------------------------------------------------------------------
jobs = collections.OrderedDict()          # 폴더 → [(문맥 키, 예측 dict 이름, futures [(라벨, 행)], 쌍 정보)]
PV, SV = loadp("vobench")
rows = list(csv.DictReader(open(f"{REPO}/data_csv/vobench_v1/index.csv")))
by = collections.defaultdict(dict)
for r in rows: by[r["block_id"]][r["variant"]] = r
MOVING = ["flat_vel", "flat_acc", "flat_dec", "ramp_down", "ramp_up"]
for occ in ("visible", "early", "mid", "late"):
    for mv in ("static", "moving"):
        need = 1 if mv == "static" else 2
        picked, used_motion = [], collections.defaultdict(set)
        cells = sorted({(r["violation"], r["objset"]) for r in rows})
        # moving: 칸 c 는 운동 MOVING[2c], MOVING[2c+1] 을 먼저 찾는다 → 폴더 안에 다섯 운동 (ramp_down 포함) 이 고루. 없으면 아무 운동이나
        want = {c: ([MOVING[(2 * k) % 5], MOVING[(2 * k + 1) % 5]] if mv == "moving" else ["flat_static"]) for k, c in enumerate(cells)}
        for strict in (True, False):
            for b, v in sorted(by.items(), key=lambda x: int(x[0])):
                for p, s in (("pos_a", "a"), ("pos_b", "b")):
                    r = v[p]
                    if r["occlusion"] != occ or (r["motion"] == "flat_static") != (mv == "static") or b + s not in PV: continue
                    cell = (r["violation"], r["objset"])
                    if strict and r["motion"] not in want[cell]: continue
                    if sum(1 for c in picked if c[0] == cell) >= need or r["motion"] in used_motion[cell]: continue
                    if r["violation"] == "permanence" and any(c[0] == cell and c[2] == s for c in picked): continue   # 영속은 두 방향을 하나씩
                    picked.append((cell, b, s)); used_motion[cell].add(r["motion"])
        jobs[f"{occ}/{mv}"] = [(b + s, "vobench", [("possible", by[b]["pos_" + s]), ("impossible", by[b]["imp_ab" if s == "a" else "imp_ba"])])
                               for (cell, b, s) in sorted(picked, key=lambda c: (c[0][0], c[0][1]))]
PG, SG = loadp("gravity")
grows = list(csv.DictReader(open(f"{REPO}/data_csv/vobench_v1_gravity/index.csv")))
gb = collections.defaultdict(list)
for r in grows: gb[r["block_id"]].append(r)
for law in ("ledge", "arc"):
    picked, seen = [], set()
    for b, rs in sorted(gb.items(), key=lambda x: int(x[0])):
        pos = next(r for r in rs if r["plausible"] == "1")
        if pos["law"] != law or b not in PG or (pos["objset"], pos["cell"]) in seen: continue
        seen.add((pos["objset"], pos["cell"])); picked.append(b)
    picked = picked[:6]
    jobs[f"gravity/{law}"] = [(b, "gravity", [(r["future"], r) for r in sorted(gb[b], key=lambda r: r["plausible"] != "1")]) for b in picked]
if a.only:
    jobs = {k: v for k, v in jobs.items() if k in a.only.split(",")}
for k, v in jobs.items(): print(f"{k:16s} 클립 {len(v)}", flush=True)

# ---- 모델 (토크나이저만 쓴다) ---------------------------------------------------------------------------
from cosmos_predict1.autoregressive.inference.world_generation_pipeline import ARBaseGenerationPipeline   # noqa: E402
torch._C._jit_set_texpr_fuser_enabled(False)
pipe = ARBaseGenerationPipeline(inference_type="base", checkpoint_dir=a.checkpoint_dir, checkpoint_name="Cosmos-Predict1-4B",
                                disable_diffusion_decoder=True, offload_guardrail_models=False, offload_diffusion_decoder=False,
                                offload_network=True, offload_tokenizer=False, disable_guardrail=True, parallel_size=1)
M = pipe.model; VT = M.tokenizer.video_tokenizer


def path(r, kind):
    return r["file_name"] if kind == "vobench" else "gravity/" + r["file_name"]


def load(p):
    x = torch.from_numpy(np.stack([np.asarray(Image.open(f"{a.root}/{p}/{i:06d}.png").convert("RGB")) for i in IDX])).permute(0, 3, 1, 2).float().cuda() / 255
    big = F.pad(F.interpolate(x, size=(H, H), mode="bilinear", align_corners=False, antialias=True), (PADL, PADL)) * 2 - 1
    return big.permute(1, 0, 2, 3), F.interpolate(x, size=(S, S), mode="bilinear", align_corners=False, antialias=True)   # (3,33,H,W) [-1,1] · (33,3,S,S) [0,1]


def tok(v):
    d, _ = M.tokenizer.tokenize(data_batch={"video": v[None]}); return d.cuda().view(1, *LAT)


def dec(ind):
    y = (VT.decode(ind.long()).float()[0] * 0.5 + 0.5).clamp(0, 1)
    return F.interpolate(y.permute(1, 0, 2, 3)[..., PADL:PADL + H], size=(S, S), mode="bilinear", align_corners=False, antialias=True).clamp(0, 1)


to8 = lambda t: (t.clamp(0, 1) * 255).round().to(torch.uint8).cpu()
psnr = lambda y, x: float(-10 * torch.log10(((y - x) ** 2).mean()))
COLS, s, L, top = [7, 8, 9, 12, 15, 18, 21, 24, 27, 30, 32], 128, 260, 36


def gif(rows_, fp):
    """rows_ = [(라벨, uint8 (33,3,S,S), 원본인가)] — 원본이 아닌 칸은 문맥 동안 '(ctx)' 흰 글씨, 예측 동안 노란 글씨 (render_future 와 같다)"""
    fr = []
    for t in range(len(IDX)):
        im = Image.fromarray(torch.cat([q[t] for _, q, _ in rows_], -1).permute(1, 2, 0).numpy()); d = ImageDraw.Draw(im)
        for j, (lab, _, raw) in enumerate(rows_):
            d.rectangle([j * S, 0, j * S + 200, 14], fill=(0, 0, 0))
            d.text((j * S + 3, 1), lab + (" (ctx)" if t < NC and not raw else ""), fill=(255, 255, 0) if t >= NC and not raw else (255, 255, 255))
        d.text((3, S - 14), f"frame {IDX[t]}", fill=(255, 255, 255))
        fr.append(im)
    fr[0].save(fp, save_all=True, append_images=fr[1:], duration=125, loop=0)


with torch.inference_mode():
    for folder, clips in jobs.items():
        out = os.path.join(a.out, folder); os.makedirs(out, exist_ok=True)
        P, SUR = (PV, SV) if clips and clips[0][1] == "vobench" else (PG, SG)
        panels, summ = [], []
        for key, kind, futs in clips:
            vids = [(lab, *load(path(r, kind)), r) for lab, r in futs]
            ctx = vids[0][1].clone(); ctx[:, NC:] = ctx[:, NC - 1:NC]
            ind = tok(ctx); ind[:, 2:] = torch.from_numpy(P[key].astype(np.int64)).cuda().view(1, 3, LAT[1], LAT[2])
            yc = dec(ind); yc[:NC] = vids[0][2][:NC]                                # 가운데 칸: 문맥 = 실제 프레임 → 예측
            yt = dec(tok(vids[0][1]))
            sc = {lab: SUR.get(r["video_id"]) for lab, _, _, r in vids}
            pos = sc[vids[0][0]]; others = {l: v for l, v in sc.items() if l != vids[0][0]}
            ok = all(v > pos for v in others.values()) if None not in sc.values() else None
            rec = {"clip": path(vids[0][3], kind), "key": key, "surprise": sc, "correct": ok,
                   "psnr": {lab: psnr(yc[NC:], sm[NC:]) for lab, _, sm, _ in vids}, "psnr_tokenizer": psnr(yt[NC:], vids[0][2][NC:]),
                   "psnr_copy": psnr(vids[0][2][NC - 1:NC].expand(len(IDX) - NC, -1, -1, -1), vids[0][2][NC:])}
            summ.append(rec)
            note = [("S " + " / ".join(f"{l} {v:.4f}" for l, v in sc.items() if v is not None) + (f" -> {'correct' if ok else 'wrong'}" if ok is not None else ""), (150, 60, 0)),
                    ("PSNR " + " / ".join(f"{l} {v:.1f}" for l, v in rec["psnr"].items()), (90, 90, 90))]
            name = path(vids[0][3], kind)
            gif([("video", to8(vids[0][2]), True), ("Cosmos (greedy)", to8(yc), False), ("tokenizer enc->dec", to8(yt), False)],
                os.path.join(out, name.replace("/", "__") + ".gif"))
            orig = [(f"orig {lab}", to8(sm), None) for lab, _, sm, _ in (vids if kind == "gravity" and vids[0][3]["law"] == "ledge" else vids[:1])]
            panels.append((name, orig + [("tokenizer enc->dec", to8(yt), [(f"PSNR {rec['psnr_tokenizer']:.1f} (ceiling)", (90, 90, 90))]),
                                         ("Cosmos (greedy)", to8(yc), note)]))
            print(f"  {folder} {name}: {note[0][0]}", flush=True)
        sh = Image.new("RGB", (L + s * len(COLS), top + s * sum(len(r) for _, r in panels)), (255, 255, 255)); d = ImageDraw.Draw(sh)
        d.text((3, 3), f"Cosmos-Predict1-4B greedy (IntPhys2 official latent scoring) · {folder} · ctx raw 30-46 -> pred 48-94", fill=(0, 0, 0))
        for j, t in enumerate(COLS):
            d.text((L + j * s + 3, top - 15), f"f{IDX[t]}" + (" ctx" if t < NC else ""), fill=(110, 110, 110) if t < NC else (0, 0, 0))
        y = top
        for name, rr in panels:
            for i, (lab, seq, notes) in enumerate(rr):
                if i == 0:
                    d.text((3, y + 3), "/".join(name.split("/")[:3]), fill=(0, 0, 0)); d.text((3, y + 17), name.split("/")[-2][:36], fill=(60, 60, 60))
                d.text((3, y + s - 30 - 14 * max(len(notes or []) - 1, 0)), lab, fill=(0, 0, 160) if notes else (0, 0, 0))
                for q, (txt, col) in enumerate(notes or []):
                    d.text((3, y + s - 16 - 14 * q), txt[:44], fill=col)
                for j, t in enumerate(COLS):
                    sh.paste(Image.fromarray(seq[t].permute(1, 2, 0).numpy()).resize((s, s)), (L + j * s, y))
                y += s
            d.line([(0, y - 1), (sh.width, y - 1)], fill=(120, 120, 120), width=2)
        sh.save(os.path.join(out, "grid.png"))
        json.dump({"folder": folder, "scores": a.scores, "clips": summ}, open(os.path.join(out, "summary.json"), "w"), indent=1, ensure_ascii=False)
print("산출:", a.out)
