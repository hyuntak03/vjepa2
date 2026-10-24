#!/usr/bin/env python3
"""자연 영상 학습셋 (fps 4 표) 의 optical flow 를 칸 단위로 미리 뽑는다 (2026-10-23, 사용자: "optical flow 준비").

  # GPU 하나 = shard 하나. 8 장이면 shard 0..7 을 각 GPU 에
  for k in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$k nohup /venv/vjepa/bin/python -W ignore \
      z_training/data/extract_flow_fps4.py --src k710_320 --shard $k --nshard 8 > z_training/logs/flow_k710_$k.log 2>&1 & done

입력: data_csv/<src>/fps4_table.tsv (경로 · fps · 장 수 · frame_step · fps4 장 수) — 학습 로더 (var_window) 가 쓰는 표와 같다.
프레임: 영상마다 0, step, 2·step, … (fps4 장 수만큼) — 학습 로더의 fps4 프레임과 같은 자리. 가운데 정사각 → 256² bilinear (학습 입력과 같은 기하).
흐름: RAFT large (torchvision C_T_SKHT_V2, 12 반복, fp16 autocast), **순방향 t → t+1** (fps4 연속 두 장, 0.25 s).
      256² 화소 흐름을 16×16 칸 평균 (avg_pool 16) → 칸마다 (dx, dy), 단위 = 256² 화면의 px.
저장 (shard 마다): <out>/<src>/shard{k}_flow.npy  (M, 16, 16, 2) fp16  ·  shard{k}_index.tsv (영상 경로 · 시작 행 · 쌍 수)
      영상 하나의 쌍 i 는 fps4 장 i → i+1. 블록 (2 장) 단위 흐름은 연속 쌍을 더해 만든다 (근사).
재개: index.tsv 에 이미 있는 영상은 건너뛴다 (중간에 죽어도 이어서)."""
import argparse, os, sys, time
import numpy as np, torch, torch.nn.functional as F
import decord
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="k710_320 | ssv2 | ht100m_1pct")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--out", default="/workspace/dataset/training_data/flow_fps4")
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--chunk", type=int, default=64, help="RAFT 한 번에 넣는 쌍 수")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
torch.set_num_threads(4)
dev = torch.device("cuda")

rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(REPO, f"data_csv/{a.src}/fps4_table.tsv"))]
rows = [r for i, r in enumerate(rows) if i % a.nshard == a.shard]
if a.limit:
    rows = rows[: a.limit]
od = os.path.join(a.out, a.src); os.makedirs(od, exist_ok=True)
ip, fp = os.path.join(od, f"shard{a.shard}_index.tsv"), os.path.join(od, f"shard{a.shard}_flow.f16")   # 원시 fp16 이어 붙이기 (끝나면 .npy 로 감싼다)
done = {l.split("\t")[0] for l in open(ip)} if os.path.exists(ip) else set()
start_row = (os.path.getsize(fp) // (16 * 16 * 2 * 2)) if os.path.exists(fp) else 0
rows = [r for r in rows if r[0] not in done]
print(f"[{a.src} shard {a.shard}/{a.nshard}] 남은 영상 {len(rows)} (이미 {len(done)}, 저장된 쌍 {start_row})", flush=True)


class DS(torch.utils.data.Dataset):
    def __len__(self):
        return len(rows)

    def __getitem__(self, i):
        p, fps, n, stp, n4 = rows[i]
        try:
            vr = decord.VideoReader(p, num_threads=2)
            idx = [j for j in range(0, int(n), int(stp))][: int(n4)]
            idx = [j for j in idx if j < len(vr)]
            fr = vr.get_batch(idx).asnumpy()
        except Exception as e:                                     # 깨진 영상은 쌍 0 개로 기록
            print(f"  ! {p}: {e}", flush=True)
            return p, torch.zeros(0, 3, 256, 256, dtype=torch.uint8)
        H, W = fr.shape[1:3]; s = min(H, W); y0, x0 = (H - s) // 2, (W - s) // 2
        x = torch.from_numpy(np.ascontiguousarray(fr[:, y0:y0 + s, x0:x0 + s])).permute(0, 3, 1, 2).float()
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False).clamp(0, 255).round().to(torch.uint8)
        return p, x


raft = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(dev).eval()
dl = torch.utils.data.DataLoader(DS(), batch_size=None, num_workers=a.workers, prefetch_factor=4, persistent_workers=False)
fo, io_ = open(fp, "ab"), open(ip, "a")
row, t0, npair = start_row, time.time(), 0
for vi, (p, x) in enumerate(dl):
    n = max(len(x) - 1, 0)
    if n:
        out = []                                                   # 2026-10-24: 영상 전체가 아니라 chunk 만 GPU 로 (HowTo100M 1,600 장 영상이 OOM 났다)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for i in range(0, n, a.chunk):
                seg = x[i:i + a.chunk + 1].to(dev, non_blocking=True).float() / 127.5 - 1
                f = raft(seg[:-1].contiguous(), seg[1:].contiguous(), num_flow_updates=12)[-1]
                out.append(F.avg_pool2d(f.float(), 16).permute(0, 2, 3, 1).half().cpu())   # (m, 16, 16, 2)
        fo.write(torch.cat(out).numpy().tobytes())
    io_.write(f"{p}\t{row}\t{n}\n")
    row += n; npair += n
    if vi % 200 == 0:
        fo.flush(); io_.flush()
        el = time.time() - t0
        print(f"  {vi}/{len(rows)} 영상 · 쌍 {npair} · {npair / max(el, 1e-6):.0f} 쌍/s · 남은 {(len(rows) - vi) * el / max(vi, 1) / 3600:.1f} h", flush=True)
fo.close(); io_.close()
print(f"끝: 쌍 {row} → {fp} (fp16, (M,16,16,2) 로 읽기: np.memmap(f, np.float16, 'r').reshape(-1,16,16,2))", flush=True)
