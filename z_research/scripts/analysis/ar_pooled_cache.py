"""phys3d_single_v1 전체 클립의 블록 AR 표현을 블록마다 256 칸 평균으로만 저장 (2026-10-26, 학습 없음).
사용자: "샘플 갯수 늘리던가. 어차피 샘플 갯수는 엄청 많잖아". ar_layer_cache.py (8,000 편, 전체 토큰 697 GB) 를 칸 평균만 남기는 판으로 넓힌다.
클립: 96 장 이상인 train 전부 (28,600 편 중) · val 전부 (1,040 편 중). 원본 0, 3, …, 93 장 → 블록 16. 모델 · 층 정의 · 칸 평균 · 라벨은 ar_layer_cache.py + pool_objcells.py 와 같다:
  enc (16, 1280) · tfh (14, 16, 384) · tfout (16, 1280; slot j = 블록 j 예측, slot 0 NaN) · roh (14, 8, 384; slot k = 굴림 자리 8+k, slot 7 NaN) · roout (8, 1280)
  lab (16, 8) = [보임, 모양, 색, 장면 번호, dx, dy, x, y] (장면 번호 = 장면 이름 정렬 순서 13 종)
GPU 메모리 상한 MEM_GB (기본 8). fp16 저장.
  for k in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$k MEM_GB=8 nohup /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/ar_pooled_cache.py --shard $k --nshard 8 > <log>_$k 2>&1 & done
  /venv/vjepa/bin/python z_research/scripts/analysis/ar_pooled_cache.py --merge --nshard 8
출력: /workspace/auto_research_cache/ar_pooled_full/{train,val}.npz
"""
import argparse, glob, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F

REPO = "/workspace/world_model/vjepa2"; sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "auto_research/scripts"))
ap = argparse.ArgumentParser(); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--merge", action="store_true"); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
OUT = "/workspace/auto_research_cache/ar_pooled_full"; os.makedirs(OUT, exist_ok=True)
P3 = "/workspace/dataset/training_data/phys3d_single_v1"
IDX = [3 * k for k in range(32)]; G = 16; S = 256; D = 1280; NBC = 8
SCEN = sorted(os.listdir(f"{P3}/train/clips"))
KEYS = ("enc", "tfh", "tfout", "roh", "roout", "lab")

if a.merge:
    for split in ("val", "train"):
        parts = [np.load(f"{OUT}/{split}_shard{k}.npz") for k in range(a.nshard) if os.path.exists(f"{OUT}/{split}_shard{k}.npz")]
        clip = np.concatenate([p["clip"] for p in parts]); o = np.argsort(clip)
        np.savez(f"{OUT}/{split}.npz", **{k: np.concatenate([p[k] for p in parts])[o] for k in KEYS}, clip=clip[o], scen_names=np.array(SCEN))
        print(split, len(clip), flush=True)
    sys.exit(0)


def labels(f):
    st = np.load(f[:-4] + "_states.npz"); pos, vis, inf = st["pos"][:, 0], st["vis_frac"][:, 0], st["inframe_frac"][:, 0]
    L = np.zeros((16, 8), np.float32)
    for b in range(16):
        f0, f1 = IDX[2 * b], IDX[2 * b + 1]
        ok = float(min(vis[f0], vis[f1]) >= 0.5 and min(inf[f0], inf[f1]) >= 0.9)
        ctr = 0.5 * (pos[f0] + pos[f1]); d = pos[f1] - pos[f0]
        L[b] = [ok, st["shape_id"][0], st["color_id"][0], SCEN.index(f.split("/")[-2]), d[0], d[1], ctr[0], ctr[1]]
    return L


dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEM_GB", 8)) * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory, 0)
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
CAP = []
for i_, blk in enumerate(ar.predictor_blocks):
    blk.register_forward_hook(lambda mod, inp, out, _i=i_: CAP.append((_i, out[0] if isinstance(out, tuple) else out)))
    _f = blk.forward_kv
    def _fkv(*args, _f=_f, _i=i_, **kw):
        y, kv = _f(*args, **kw); CAP.append((_i, y)); return y, kv
    blk.forward_kv = _fkv
idx = torch.arange(16 * S, device=dev).unsqueeze(0)

for split in ("val", "train"):
    fs = sorted(glob.glob(f"{P3}/{split}/clips/*/*.mp4"))
    mine = [f for n, f in enumerate(fs) if n % a.nshard == a.shard]
    if a.limit:
        mine = mine[: a.limit]
    R = {k: [] for k in KEYS}; CL = []; t0 = time.time(); skipped = 0
    for n_, f in enumerate(mine):
        try:
            if np.load(f[:-4] + "_states.npz")["pos"].shape[0] < 96:
                skipped += 1; continue
            fr = decord.VideoReader(f, num_threads=4).get_batch(IDX).asnumpy()
        except Exception as e:
            print("skip", f, e, flush=True); skipped += 1; continue
        xn = (torch.from_numpy(fr).permute(3, 0, 1, 2).float().to(dev) / 255.0 - MEAN) / STD
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hb = torch.cat([LN(enc2(xn[:, 4 * q: 4 * q + 4].reshape(3, 2, 2, 256, 256).permute(1, 0, 2, 3, 4)).float()) for q in range(8)])  # (16, S, D)
            x = hb.reshape(1, -1, D)
            CAP.clear(); emb = ar._ar_embed_obs(x); out, st = ar.forward_seq(x, idx, return_states=True)
            blocks = [None] * 12
            for li, y in CAP:
                blocks[li] = y
            tf_h = torch.stack([emb[0]] + [blocks[li][0] for li in range(12)] + [st[0]]).float().view(14, 16, S, -1).mean(2)
            tf_out = out[0].float().view(16, S, D).mean(1)
            caches = None
            for b in range(NBC):
                CAP.clear(); p, caches, stt = ar._ar_step_kv(x[:, b * S:(b + 1) * S], idx[:, b * S:(b + 1) * S], caches)
            ro_out, ro_h = [p[0].float().mean(0)], []
            pos = idx[:, (NBC - 1) * S: NBC * S]
            for r in range(1, 16 - NBC):
                pos = pos + S; inp = stt + ar.type_embed[1]
                CAP.clear(); p, caches, stt = ar._ar_step_kv(inp, pos, caches, embedded=True)
                bl = [None] * 12
                for li, y in CAP:
                    bl[li] = y
                ro_h.append(torch.stack([inp[0]] + [bl[li][0] for li in range(12)] + [stt[0]]).float().mean(1))   # (14, 384)
                ro_out.append(p[0].float().mean(0))
        tfo = torch.full((16, D), float("nan")); tfo[1:] = tf_out[:15].cpu()
        roh = torch.full((14, 8, 384), float("nan")); roh[:, :7] = torch.stack(ro_h, 1).cpu()
        R["enc"].append(hb.float().mean(1).half().cpu().numpy()); R["tfh"].append(tf_h.half().cpu().numpy()); R["tfout"].append(tfo.half().numpy())
        R["roh"].append(roh.half().numpy()); R["roout"].append(torch.stack(ro_out).half().cpu().numpy()); R["lab"].append(labels(f)); CL.append(f)
        if n_ % 200 == 0:
            print(time.strftime("%H:%M:%S"), f"{split} shard {a.shard}: {n_}/{len(mine)} ({time.time() - t0:.0f}s, 건너뜀 {skipped})", flush=True)
    np.savez(f"{OUT}/{split}_shard{a.shard}.npz", **{k: np.stack(v) for k, v in R.items()}, clip=np.array(CL))
    print(f"{split} shard {a.shard} 저장 {len(CL)} (건너뜀 {skipped}, {time.time() - t0:.0f}s)", flush=True)
print("done shard", a.shard, flush=True)
