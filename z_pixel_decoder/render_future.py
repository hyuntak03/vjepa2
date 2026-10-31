#!/usr/bin/env python3
"""여러 predictor 가 만든 미래를 pixel decoder 로 같은 클립에 나란히 그림 (2026-10-21, 사용자 지시:
"ar(vast) w/ VoBench (gravity) 포함 pft 돌려서 보여줘" → "release 랑 release+pft 도 같이, release + IntPhys1 · release + IntPhys2").

  source cvpr/env.sh; CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<물리 GPU> PYTHONPATH=$PWD $CVPR_PY z_pixel_decoder/render_future.py \
      --models vith,vith_release_vobftg,vith_pft_intphys1_e40,vith_pft_ip2pair,vith_vast_ar_state_ep70,vith_vast_ar_state_ep70_vobftg --sets late,gravity --n 8

모델 이름은 cvpr/registry/models.yaml (base: vith 인 것만 — predictor_checkpoint · predictor.kind 를 거기서 읽는다).
입력은 채점 (testbed surprise_c16t32) 과 같다: 원본 0, 3, …, 93 장 (32 장), 256² bilinear (antialias 없음), fp16 autocast, 문맥 16 장.
  릴리즈 계열 (kind default)  p = predictor(context_encoder(문맥), …)  — 정답 공간 = LN(target_encoder(32 장 한 번에))[미래]  (decoder 'window')
  AR 계열 (kind ar · ar_state) p = rollout(블록별 LN(target_encoder)(문맥 8 블록), 8)   — 정답 공간 = 블록별 LN  (decoder 'block')
decoder (vob_grav_i1_vitb_both, latent both) 는 두 공간을 다 학습했다 → 각 모델을 자기 공간 그대로 그린다.
클립 = PFT 에 안 쓴 test split (VOBench split v2 test pos_a / gravity split test 가능 미래, ledge 는 부유 미래도 같이), (objset, 위반·법칙) 마다 고르게, decoder val 클립 먼저.
⚠️ release + IntPhys1 · IntPhys2 는 VOBench 가 학습 도메인 밖이고, release+PFT · AR+PFT 는 도메인 안이다 (test split 이라도 장면 분포가 같다).
산출 (모델별 폴더, 10-21): runs/<decoder run>/<release | release_vobench | release_intphys1 | release_intphys2 | ar | ar_vobench>/<set>/
      grid.png (행 = 클립 × [원본 · 정답 latent (그 모델 공간) · 그 모델], 열 = 문맥 끝 2 장 + 미래) · <clip>.gif (영상 | 예측 | encoder→decoder) · summary.json (PSNR · 채점 surprise)
      runs/<decoder run>/compare/<set>/compare_grid.png — 모든 모델 한 장 (+ summary.json)"""
import argparse, gc, json, os, sys
import numpy as np, torch, torch.nn.functional as F, yaml
from PIL import Image, ImageDraw
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from analysis.intphys2.model import build_from_config                  # noqa: E402
from analysis import predictors as PV                                   # noqa: E402
from analysis.predictors import ar_scoring as ARS                      # noqa: E402
from analysis.intphys2.surprise import _context_target_indices         # noqa: E402
from model import PixelDecoder                                          # noqa: E402
from clips import pick_clips                                            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--models", default="vith,vith_release_vobftg,vith_pft_intphys1_e40,vith_pft_ip2pair,vith_vast_ar_state_ep70,vith_vast_ar_state_ep70_vobftg")
ap.add_argument("--sets", default="late,gravity")
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--only", default="", help="block_id 쉼표 목록 — 그 블록만 그린다 (gravity 는 전체 index 를 본다, clips.py)")
ap.add_argument("--decoder", default=os.path.join(HERE, "runs/vob_grav_i1_vitb_both/latest.pt"))
ap.add_argument("--root", default=os.environ.get("CVPR_VOBENCH", "/data2/local_datasets/world/Benchmark/VOBench_v1"))
ap.add_argument("--list-frames", action="store_true", help="쓸 프레임 경로만 찍고 끝 — 다른 노드에서 돌릴 때 그 프레임만 NFS 로 복사해 --root 로 준다")
a = ap.parse_args()
MODELS, SETS = a.models.split(","), a.sets.split(",")
dev = torch.device("cuda")
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
REG = yaml.safe_load(open(os.path.join(REPO, "cvpr/registry/models.yaml")))
LABEL = {"vith": "release", "vith_release_vobftg": "release+PFT(VOB+grav)", "vith_pft_intphys1_e40": "release+IntPhys1",
         "vith_pft_ip2pair": "release+IntPhys2", "vith_vast_ar_state_ep70_vobftg": "AR(vast)+PFT(VOB+grav)", "vith_vast_ar_state_ep70": "AR(vast)",
         "vith_vw_shortctx_trajdiff": "AR(short ctx)+optical flow", "vith_vw_shortctx": "AR(short ctx)"}
SHORT = {"vith": "release", "vith_release_vobftg": "release_vobench", "vith_pft_intphys1_e40": "release_intphys1", "vith_pft_ip2pair": "release_intphys2",
         "vith_vast_ar_state_ep70": "ar", "vith_vast_ar_state_ep70_vobftg": "ar_vobench",
         "vith_vw_shortctx_trajdiff": "ar_shortctx_flow", "vith_vw_shortctx": "ar_shortctx"}     # 산출 폴더 이름 (없으면 레지스트리 이름 그대로)


def model_cfg(name):
    """레지스트리 항목 → build_from_config 입력 (= 채점 _resolved.yaml 의 model 블록)"""
    e = REG[name]
    assert name == "vith" or e.get("base") == "vith", f"{name}: base vith 인 모델만 ({e.get('base')})"
    m = {"arch_name": "vit_huge", "checkpoint": CK, "img_size": 256, "patch_size": 16, "tubelet_size": 2, "use_rope": True, "uniform_power": False,
         "dual_encoder": True, "context_encoder_key": "encoder", "target_encoder_key": "target_encoder",
         "predictor": {"embed_dim": 384, "depth": 12, "num_heads": 12, "num_mask_tokens": 10}, "dtype": "float32", "autocast": "float16", "window_size": 32}
    if name != "vith":
        m["predictor_checkpoint"] = e["predictor_checkpoint"].replace("${CVPR_REPO}", REPO)
        if "predictor.kind" in e:
            m["predictor"]["kind"] = e["predictor.kind"]
    return m


st = torch.load(a.decoder, map_location="cpu", weights_only=False)
mc, dcfg = st["cfg"]["model"], st["cfg"]["data"]
dec = PixelDecoder(in_dim=1280, dim=mc["dim"], depth=mc["depth"], heads=mc["heads"]).to(dev).eval()
print("decoder", dec.load_state_dict(st["decoder"]), f"· step {st['step']} · latent {dcfg.get('latent')}", flush=True)

T, C, S = 32, 16, 256
CB = C // 2
IDX = [3 * k for k in range(T)]
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(3, 1, 1, 1)
ci, ti = _context_target_indices(ctx_frames=C, tgt_frames=T - C, tubelet_size=2, spatial_tokens=S, batch_size=1, device=dev)
LN = lambda t: F.layer_norm(t, (t.size(-1),))
psnr = lambda y, x: float(-10 * torch.log10(((y - x) ** 2).mean()))
to8 = lambda t: (t.clamp(0, 1) * 255).round().to(torch.uint8).cpu()


def load(d, idx=None, pattern="{i:06d}.png"):
    """세트마다 프레임 번호 · 틀이 다르다 (clips.py: VOBench 0,3,… · IntPhys1 1,4,… · IntPhys2 late 0..31)"""
    fr = [np.asarray(Image.open(os.path.join(d, pattern.format(i=i))).convert("RGB")) for i in (idx or IDX)]
    x = torch.from_numpy(np.stack(fr)).permute(3, 0, 1, 2).float().to(dev) / 255.0           # (3, T, H, W) [0, 1]
    return F.interpolate(x.transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False, antialias=False).transpose(0, 1)


@torch.no_grad()
def decode(tok):
    """tok (nb·256, D) LN 토큰 → (nb·2, 3, 256, 256) [0, 1]"""
    nb = tok.size(0) // S
    y = dec(tok.view(nb, S, -1).float())
    y = (y * STD.view(1, 3, 1, 1, 1) + MEAN.view(1, 3, 1, 1, 1)).clamp(0, 1)
    return y.permute(0, 2, 1, 3, 4).reshape(nb * 2, 3, 256, 256)


ONLY = set(a.only.split(",")) if a.only else None
CLIPS = {s_: pick_clips(s_, a.n, a.root, dcfg["val_frac"], only=ONLY) for s_ in SETS}   # clips.py (render_videomae.py 와 같은 클립)
if a.list_frames:                                                       # 다른 노드용: 쓸 프레임 경로만 찍고 끝 (그 노드엔 VOBench 가 없다)
    for s_ in SETS:
        for c in CLIPS[s_]:
            for _, d in c["futures"]:
                print("\n".join(os.path.join(d, c["pattern"].format(i=i)) for i in c["idx"]))
    sys.exit(0)
FR = {s_: [[load(p, c["idx"], c["pattern"]) for _, p in c["futures"]] for c in CLIPS[s_]] for s_ in SETS}           # [0, 1] (3, T, H, W)
RES = {s_: [{"clip": c["name"], "dec_val": c["dec_val"], "models": {}} for c in CLIPS[s_]] for s_ in SETS}
DEC = {s_: [{} for _ in CLIPS[s_]] for s_ in SETS}                                          # 그림용 uint8 (16, 3, H, W)

for mi, name in enumerate(MODELS):
    bundle = build_from_config(model_cfg(name), dev)
    is_ar = PV.is_ar(bundle.predictor)
    space = "block" if is_ar else "window"
    print(f"== {name} ({LABEL.get(name, name)}) · {'AR rollout' if is_ar else 'release 규약'} · 정답 공간 {space}", flush=True)
    for s_ in SETS:
        for k, c in enumerate(CLIPS[s_]):
            xs = [((x - MEAN) / STD).unsqueeze(0) for x in FR[s_][k]]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                if is_ar:
                    h0 = ARS.encode_blocks(bundle.target_encoder, xs[0], 2)
                    p, _, _ = ARS.rollout_from_blocks(bundle.predictor, h0, CB, S)
                    hf = [h0[:, CB * S:]] + [ARS.encode_blocks(bundle.target_encoder, x, 2)[:, CB * S:] for x in xs[1:]]
                else:
                    p = bundle.predictor(bundle.context_encoder(xs[0], masks=[ci]), ci, ti, mask_index=0)
                    hf = [LN(bundle.target_encoder(x).float()[:, C // 2 * S:]) for x in xs]
            p = p.float()[0]; hf = [h.float()[0] for h in hf]
            real = [x[:, C:].permute(1, 0, 2, 3) for x in FR[s_][k]]                            # 미래 (16, 3, H, W) 들
            yp = decode(p)
            labs = [lab for lab, _ in c["futures"]]
            rec = {"space": space, "psnr": {lab: psnr(yp, rr) for lab, rr in zip(labs, real)},
                   "surprise": {lab: float((p - h).abs().mean()) for lab, h in zip(labs, hf)},
                   # 2026-10-25: 미래 슬롯마다 토큰 사이 표준편차 (채널 평균) — p 가 슬롯이 갈수록 평평해지는지 (decoder 가 배경만 그리는 이유 후보), 정답 latent 와 비교
                   "spatial_std": {"p": p.view(-1, S, p.size(-1)).std(1).mean(-1).tolist(), "h": hf[0].view(-1, S, hf[0].size(-1)).std(1).mean(-1).tolist()}}
            if mi == 0 or space not in RES[s_][k].get("gt_latent", {}):                         # 정답 latent 상한은 공간마다 한 번
                yg = decode(hf[0])
                RES[s_][k].setdefault("gt_latent", {})[space] = psnr(yg, real[0])
                DEC[s_][k].setdefault("gt", {})[space] = to8(yg)
            RES[s_][k]["models"][name] = rec
            RES[s_][k]["psnr_copy"] = psnr(FR[s_][k][0][:, C - 1:C].permute(1, 0, 2, 3).expand(T - C, -1, -1, -1), real[0])
            DEC[s_][k][name] = to8(yp)
            print(f"  {s_} {c['name']}: PSNR " + " · ".join(f"{l} {v:.1f}" for l, v in rec["psnr"].items()) +
                  " | surprise " + " · ".join(f"{l} {v:.3f}" for l, v in rec["surprise"].items()), flush=True)
    del bundle; gc.collect(); torch.cuda.empty_cache()

# ---- 산출 (10-21 사용자: "모델별로 — renders, renders_ar 이 아니라 release, ar, release_intphys1 … 이렇게")
#   <decoder run>/<SHORT>/<set>/  grid.png (행 = 클립 × [원본 (ledge 는 낙하 · 부유) · 정답 latent (그 모델 공간) · 그 모델]) · <clip>.gif · summary.json
#   <decoder run>/compare/<set>/  compare_grid.png (모든 모델 한 장) · summary.json (모델 전부)
COLS, s, L, top = [13, 15, 16, 18, 20, 22, 24, 26, 28, 30, 31], 128, 260, 18
RUN = os.path.dirname(a.decoder)
GRAY, BLUE, ORANGE, BLACK = (90, 90, 90), (0, 0, 160), (150, 60, 0), (0, 0, 0)


FIDX = None            # 그림 열 머리글에 쓸 프레임 번호 (세트마다 다르다; 세트 루프에서 채운다)


def grid(panels, title, path):
    """panels = [(클립 이름, [(라벨, uint8 (T, 3, H, W), [(메모, 색)] 또는 None)])] — 메모가 있는 행 (모델 · 정답 latent) 은 라벨을 파랗게"""
    hd = top + (18 if title else 0)
    sh = Image.new("RGB", (L + s * len(COLS), hd + s * sum(len(rr) for _, rr in panels)), (255, 255, 255)); d = ImageDraw.Draw(sh)
    if title:
        d.text((3, 3), title, fill=BLACK)
    for j, t in enumerate(COLS):
        d.text((L + j * s + 3, hd - 15), f"f{(FIDX or IDX)[t]}" + (" ctx" if t < C else ""), fill=(110, 110, 110) if t < C else BLACK)
    y = hd
    for name, rr in panels:
        for i, (lab, seq, notes) in enumerate(rr):
            if i == 0:
                _pp = name.split("/")                                   # IntPhys1 "O1/23/2" · IntPhys2 late "s168_p2_pos" 처럼 깊이가 다르다
                d.text((3, y + 3), "/".join(_pp[:3])[:36], fill=BLACK)
                d.text((3, y + 17), (_pp[-2] if len(_pp) > 1 else _pp[-1])[:36], fill=(60, 60, 60))
            d.text((3, y + s - 30 - 14 * max(len(notes or []) - 1, 0)), lab, fill=BLUE if notes else BLACK)
            for q, (txt, col) in enumerate(notes or []):
                d.text((3, y + s - 16 - 14 * q), txt, fill=col)
            for j, t in enumerate(COLS):
                sh.paste(Image.fromarray(seq[t].permute(1, 2, 0).numpy()).resize((s, s)), (L + j * s, y))
            y += s
        d.line([(0, y - 1), (sh.width, y - 1)], fill=(120, 120, 120), width=2)
    sh.save(path)


def gif(rows_, path):
    frames = []
    for t in range(T):
        im = Image.fromarray(torch.cat([rr[t] for _, rr, _ in rows_], -1).permute(1, 2, 0).numpy()); dr = ImageDraw.Draw(im)
        for j, (lab, _, _) in enumerate(rows_):
            dr.rectangle([j * 256, 0, j * 256 + 200, 14], fill=(0, 0, 0))
            raw = lab.startswith(("orig", "video"))
            dr.text((j * 256 + 3, 1), lab + (" (ctx)" if t < C and not raw else ""), fill=(255, 255, 0) if t >= C and not raw else (255, 255, 255))
        dr.text((3, 242), f"frame {(FIDX or IDX)[t]}", fill=(255, 255, 255))
        frames.append(im)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=125, loop=0)


first = lambda d: list(d.values())[0]                                    # 가능 미래 (futures 첫째)
for s_ in SETS:
    FIDX = CLIPS[s_][0]["idx"]                                          # 열 머리글용 (세트 안에서는 같다)
    rows = []                                                           # 클립마다 (원본 행들, {공간: 정답 latent 행}, {모델: 모델 행})
    for k, c in enumerate(CLIPS[s_]):
        ctx8, rec = to8(FR[s_][k][0][:, :C].permute(1, 0, 2, 3)), RES[s_][k]
        orig = [(f"orig {lab}", to8(x.permute(1, 0, 2, 3)), None) for (lab, _), x in zip(c["futures"], FR[s_][k])]
        gt = {sp: (f"GT latent ({sp})", torch.cat([ctx8, v]), [(f"PSNR {rec['gt_latent'][sp]:.1f}  (decoder ceiling)", GRAY)])
              for sp, v in DEC[s_][k]["gt"].items()}
        md = {}
        for n in MODELS:
            r_ = rec["models"][n]
            notes = [("PSNR " + " / ".join(f"{l} {v:.1f}" for l, v in r_["psnr"].items()), GRAY)]
            if len(r_["surprise"]) > 1:
                notes.append(("score picks: " + min(r_["surprise"], key=r_["surprise"].get), ORANGE))
            md[n] = (LABEL.get(n, n), torch.cat([ctx8, DEC[s_][k][n]]), notes)
        rows.append((c["name"], orig, gt, md))
    copy_m = np.mean([r["psnr_copy"] for r in RES[s_]])
    # 모델별 폴더
    for n in MODELS:
        out = os.path.join(RUN, SHORT.get(n, n), s_); os.makedirs(out, exist_ok=True)
        sp = RES[s_][0]["models"][n]["space"]
        mp = np.mean([first(r["models"][n]["psnr"]) for r in RES[s_]])
        grid([(name, orig + [gt[sp], md[n]]) for name, orig, gt, md in rows],
             f"{LABEL.get(n, n)}  ·  {s_}  ·  mean PSNR (possible future) {mp:.2f} dB  ·  copy {copy_m:.2f} dB  ·  GT latent ({sp}) "
             f"{np.mean([r['gt_latent'][sp] for r in RES[s_]]):.2f} dB", os.path.join(out, "grid.png"))
        for name, orig, gt, md in rows:                                 # GIF (10-21 사용자): 왼쪽 영상 · 가운데 예측 · 오른쪽 encoder (정답 latent → decoder)
            gif([("video", orig[0][1], None), md[n], (f"encoder ({sp}) -> decoder", gt[sp][1], None)], os.path.join(out, name.replace("/", "__") + ".gif"))
        json.dump({"set": s_, "model": n, "label": LABEL.get(n, n), "checkpoint": model_cfg(n).get("predictor_checkpoint", "release"), "space": sp,
                   "decoder": a.decoder, "mean_psnr_possible": mp, "mean_psnr_copy": copy_m,
                   "clips": [{"clip": r["clip"], "dec_val": r["dec_val"], **r["models"][n], "gt_latent": r["gt_latent"][sp], "psnr_copy": r["psnr_copy"]}
                             for r in RES[s_]]}, open(os.path.join(out, "summary.json"), "w"), indent=1, ensure_ascii=False)
    # 모든 모델 한 장
    out = os.path.join(RUN, "compare", s_); os.makedirs(out, exist_ok=True)
    grid([(name, orig + list(gt.values()) + [md[n] for n in MODELS]) for name, orig, gt, md in rows], None, os.path.join(out, "compare_grid.png"))
    json.dump({"set": s_, "models": {n: model_cfg(n).get("predictor_checkpoint", "release") for n in MODELS}, "decoder": a.decoder, "clips": RES[s_]},
              open(os.path.join(out, "summary.json"), "w"), indent=1, ensure_ascii=False)
    print(f"\n[{s_}] 평균 PSNR (가능 미래, {len(RES[s_])} clip) · 복사 {copy_m:.2f}")
    for n in MODELS:
        print(f"   {LABEL.get(n, n):26s} {np.mean([first(r['models'][n]['psnr']) for r in RES[s_]]):.2f} dB  → {os.path.join(RUN, SHORT.get(n, n), s_)}")
