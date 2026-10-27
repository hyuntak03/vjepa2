"""encoder 출력 + 블록 AR 층별 은닉 캐시 (전체 토큰) — 개념 벡터를 encoder · predictor 층마다 따로 구해 비교하기 위해 (2026-10-26, 학습 없음).
사용자: "encoder 는 output 만, predictor (ar) 는 layer 별로 (T*N, D) 뽑아. 프레임 (블록) 별로 (N, D), 공간 평균하면 (1, D). 캐싱해서 저장".

데이터: phys3d_single_v1 96 장 영상. train 은 시나리오마다 고르게 --n 편, val 은 전부 (350). 원본 0, 3, …, 93 장 (4 fps) → 블록 16 (블록 = 2 장).
모델: 블록 AR natural_ar_vw_state latest (상태 되먹임). fp16 저장.
클립마다 {split}/{i:05d}.npz:
  enc      (16, 256, 1280)      블록 b 정답 = LN(target_encoder(블록 b 의 2 장))
  tf_h     (14, 16, 256, 384)   teacher forcing 층별 은닉. 층 0 = 입력 임베딩 (관측 + type[0]), 1..12 = predictor 블록 출력, 13 = predictor_norm (상태).
                                위치 b = 블록 0..b 를 본 자리 (출력은 블록 b+1 예측)
  tf_out   (16, 256, 1280)      teacher forcing 출력 LN(proj(상태)). tf_out[b] = 블록 b+1 예측 (b = 15 는 정답 없음)
  ro_h     (14, 7, 256, 384)    굴린 예측 층별 은닉, 위치 8..14 (입력 = 앞 걸음 상태 + type[1]). 위치 0..7 은 tf_h 와 같다 (block-causal)
  ro_out   (8, 256, 1280)       굴린 예측 출력. ro_out[k] = 블록 8+k 예측 (k = 0 은 tf_out[7] 과 같다)
{split}_meta.json: 클립 경로 · 모양 · 색 · 시나리오 · 블록별 (보임, 물체 칸, 상자, 중심, 블록 안 변위)

  for k in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$((k+1)) MEM_GB=6 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/ar_layer_cache.py --shard $k --nshard 4 --n 4000 & done; wait
  /venv/vjepa/bin/python z_research/scripts/analysis/ar_layer_cache.py --merge_meta --nshard 4
출력: /workspace/auto_research_cache/ar_layers/{train,val}/ (클립당 약 92 MB)
"""
import argparse, glob, json, os, random, sys, time
import numpy as np, torch, torch.nn.functional as F

REPO = "/workspace/world_model/vjepa2"; sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "auto_research/scripts"))
ap = argparse.ArgumentParser(); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--n", type=int, default=4000); ap.add_argument("--merge_meta", action="store_true"); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--start", type=int, default=0, help="train 목록에서 이 번호부터만 (2026-10-26: 4,000 → 8,000 늘릴 때. 같은 규칙의 목록이라 앞부분이 같다)")
ap.add_argument("--skip_val", action="store_true")
a = ap.parse_args()
OUT = "/workspace/auto_research_cache/ar_layers"
P3 = "/workspace/dataset/training_data/phys3d_single_v1"
IDX = [3 * k for k in range(32)]; G = 16; S = 256; D = 1280; NBC = 8


def clip_list(split, n):
    fs = sorted(glob.glob(f"{P3}/{split}/clips/*/*.mp4")); by = {}
    for f in fs:
        by.setdefault(f.split("/")[-2], []).append(f)
    rng = random.Random(23)
    for k in sorted(by):
        rng.shuffle(by[k])
    out = []
    while (n == 0 or len(out) < n) and any(by.values()):
        for k in sorted(by):
            while by[k]:
                f = by[k].pop()
                if np.load(f[:-4] + "_states.npz")["pos"].shape[0] >= 96:
                    out.append(f); break
            if n and len(out) >= n:
                break
    return out


def labels(f):
    st = np.load(f[:-4] + "_states.npz"); pos, bb, vis, inf = st["pos"][:, 0], st["bbox"][:, 0], st["vis_frac"][:, 0], st["inframe_frac"][:, 0]
    cc = np.arange(G) * 16 + 8.0; blocks = []
    for b in range(16):
        f0, f1 = IDX[2 * b], IDX[2 * b + 1]
        ok = bool(min(vis[f0], vis[f1]) >= 0.5 and min(inf[f0], inf[f1]) >= 0.9)
        box = [float(min(bb[f0, 0], bb[f1, 0])), float(min(bb[f0, 1], bb[f1, 1])), float(max(bb[f0, 2], bb[f1, 2])), float(max(bb[f0, 3], bb[f1, 3]))]
        cells = []
        if np.isfinite(box).all():
            cols = np.nonzero((cc >= box[0]) & (cc <= box[2]))[0]; rows = np.nonzero((cc >= box[1]) & (cc <= box[3]))[0]
            cells = [int(r * G + c) for r in rows for c in cols]
        ctr = (0.5 * (pos[f0] + pos[f1])).tolist(); d = (pos[f1] - pos[f0]).tolist()
        blocks.append({"ok": ok, "cells": cells, "box": box, "ctr": ctr, "disp": d})
    return {"clip": f, "scen": f.split("/")[-2], "shape": int(st["shape_id"][0]), "color": int(st["color_id"][0]), "blocks": blocks}


if a.merge_meta:
    for split in ("train", "val"):
        M = []
        for p in sorted(glob.glob(f"{OUT}/{split}_meta_shard*.json")):     # 2026-10-26: _from4000 판까지 모두
            M += json.load(open(p))
        M.sort(key=lambda m: m["i"]); json.dump(M, open(f"{OUT}/{split}_meta.json", "w"))
        print(split, len(M))
    sys.exit(0)

dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEM_GB", 6)) * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory, 0)
import decord
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint
_argv = list(sys.argv); sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FTm
sys.argv = _argv
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
enc2 = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(enc2, _load_checkpoint(CK)["target_encoder"], tag="t"); enc2 = enc2.to(dev).eval()
ar = FTm.load_pred("natural_ar_vw_state", "latest.pt").to(dev).eval()
assert ar.ar_feedback == "state" and not ar.gating_residual
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(3, 1, 1, 1)
LN = lambda t: F.layer_norm(t, (t.size(-1),))
CAP = []                                                         # 블록 출력 수집 (TF: forward hook, rollout: forward_kv 감싸기)
for i_, blk in enumerate(ar.predictor_blocks):
    blk.register_forward_hook(lambda mod, inp, out, _i=i_: CAP.append((_i, out[0] if isinstance(out, tuple) else out)))
    _f = blk.forward_kv
    def _fkv(*args, _f=_f, _i=i_, **kw):
        y, kv = _f(*args, **kw); CAP.append((_i, y)); return y, kv
    blk.forward_kv = _fkv
idx = torch.arange(16 * S, device=dev).unsqueeze(0)

for split, files in (("val", clip_list("val", 0)), ("train", clip_list("train", a.n))):
    if split == "val" and a.skip_val:
        continue
    os.makedirs(f"{OUT}/{split}", exist_ok=True)
    st0 = a.start if split == "train" else 0
    mine = [(i, f) for i, f in enumerate(files) if i >= st0 and i % a.nshard == a.shard]
    if a.limit:
        mine = mine[: a.limit]
    meta = []; t0 = time.time()
    for n_, (i, f) in enumerate(mine):
        fr = decord.VideoReader(f, num_threads=4).get_batch(IDX).asnumpy()
        xn = (torch.from_numpy(fr).permute(3, 0, 1, 2).float().to(dev) / 255.0 - MEAN) / STD
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hb = torch.cat([LN(enc2(xn[:, 4 * q: 4 * q + 4].reshape(3, 2, 2, 256, 256).permute(1, 0, 2, 3, 4)).float()) for q in range(8)])  # (16, S, D)
            x = hb.reshape(1, -1, D)
            # teacher forcing
            CAP.clear()
            emb = ar._ar_embed_obs(x)
            out, st = ar.forward_seq(x, idx, return_states=True)
            blocks = [None] * 12
            for li, y in CAP:
                blocks[li] = y
            tf_h = torch.stack([emb[0]] + [blocks[li][0] for li in range(12)] + [st[0]]).view(14, 16, S, -1)
            tf_out = out[0].view(16, S, D)
            # 굴린 예측 (rollout_kv 와 같은 계산을 걸음마다 풀어서, 층별 은닉을 모은다)
            caches = None
            for b in range(NBC):
                CAP.clear(); p, caches, stt = ar._ar_step_kv(x[:, b * S:(b + 1) * S], idx[:, b * S:(b + 1) * S], caches)
            ro_out, ro_h = [p[0]], []
            pos = idx[:, (NBC - 1) * S: NBC * S]
            for r in range(1, 16 - NBC):
                pos = pos + S; inp = stt + ar.type_embed[1]
                CAP.clear(); p, caches, stt = ar._ar_step_kv(inp, pos, caches, embedded=True)
                bl = [None] * 12
                for li, y in CAP:
                    bl[li] = y
                ro_h.append(torch.stack([inp[0]] + [bl[li][0] for li in range(12)] + [stt[0]]))   # (14, S, 384)
                ro_out.append(p[0])
            ro_h = torch.stack(ro_h, 1); ro_out = torch.stack(ro_out)
        assert float((ro_out[0] - tf_out[7]).abs().max()) < 5e-2, "rollout 첫 걸음과 TF 위치 7 이 달라야 할 이유가 없다"
        np.savez(f"{OUT}/{split}/{i:05d}.npz", enc=hb.half().cpu().numpy(), tf_h=tf_h.half().cpu().numpy(), tf_out=tf_out.half().cpu().numpy(),
                 ro_h=ro_h.half().cpu().numpy(), ro_out=ro_out.half().cpu().numpy())
        m = labels(f); m["i"] = i; meta.append(m)
        if n_ % 50 == 0:
            print(time.strftime("%H:%M:%S"), f"{split} shard {a.shard}: {n_}/{len(mine)} ({time.time() - t0:.0f}s)", flush=True)
    json.dump(meta, open(f"{OUT}/{split}_meta_shard{a.shard}" + (f"_from{st0}" if st0 else "") + ".json", "w"))
print("done shard", a.shard, flush=True)
