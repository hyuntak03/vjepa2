#!/usr/bin/env python3
"""EK100 판독용 800 clip 을 다른 노드로 옮기기 위한 스테이징 (2026-09-29). EPIC-KITCHENS_resized 가 있는 노드 (vll3/5/6) 의 CPU 에서 돈다.

    python stage_ek100.py --out auto_research/_stage/ek100_800        # items.json + {i:04d}.npy (32, 256, 256, 3) uint8

item 선택은 e_scene_reach_nat.items_for("ek100") 그대로 (seed 0) → 09-27 의 scene_ar_ek100 과 같은 800 clip.
읽는 쪽: SCENE_EK_STAGE=<out dir> 를 주면 e_scene_reach_nat.items_for / load_item 이 npy 를 읽는다.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import e_scene_reach_nat as EN  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import decord
    items = EN.items_for("ek100", a.limit)
    for i, it in enumerate(items):
        f = out / f"{i:04d}.npy"
        if f.exists() and f.stat().st_size > 0: it["npy"] = str(f); continue
        vr = decord.VideoReader(it["file"], num_threads=2); idx = [it["start"] + it["stride"] * k for k in range(EN.NF)]
        x = vr.get_batch(idx).asnumpy()                                            # (32, H, W, 3) uint8
        if x.shape[1] != 256 or x.shape[2] != 256:
            import torch, torch.nn.functional as F
            t = torch.from_numpy(x).permute(0, 3, 1, 2).float()
            x = F.interpolate(t, size=(256, 256), mode="bilinear", align_corners=False, antialias=False).round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
        np.save(f, x); it["npy"] = str(f)
        if i % 50 == 0: print(i, len(items), flush=True)
    json.dump(items, open(out / "items.json", "w")); print("done", len(items))


if __name__ == "__main__":
    main()
