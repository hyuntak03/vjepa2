#!/usr/bin/env python3
"""[물리] 이미 렌더한 반사실 split 에 "갈림 전 장 복사" 를 나중에 적용 (Blender 없이). 생성기의 copy_ctx 와 같은 처리.
  /venv/vjepa/bin/python auto_research/scripts/cf_copy_ctx.py --root /workspace/dataset/training_data/phys3d_cf_v1/pilot2"""
import argparse, glob, json, os, shutil, subprocess
import numpy as np
PRE, FPS = 6, 12
ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); a = ap.parse_args()
for vdir in sorted(glob.glob(os.path.join(a.root, "scene_*"))):
    mp = os.path.join(vdir, "W_pos.json")
    if not os.path.exists(mp):
        continue
    m = json.load(open(mp))
    t_ev = m.get("t_event")
    if t_ev is None:                                   # 옛 판: 라벨 궤적으로 다시 잰다
        a_, b_ = np.load(os.path.join(vdir, "W_pos_states.npz")), np.load(os.path.join(vdir, "N_pos_states.npz"))
        t_ev = int(np.where(np.linalg.norm(a_["pos"][:, 0] - b_["pos"][:, 0], axis=-1) > 0.5)[0][0])
    for imp, pos in (("W_imp", "W_pos"), ("N_imp", "N_pos")):
        di, dp = os.path.join(vdir, f"{imp}_frames"), os.path.join(vdir, f"{pos}_frames")
        if not (os.path.isdir(di) and os.path.isdir(dp)):
            continue
        for t in range(t_ev):
            shutil.copyfile(os.path.join(dp, f"f{PRE + t:04d}.png"), os.path.join(di, f"f{PRE + t:04d}.png"))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-start_number", str(PRE), "-i", os.path.join(di, "f%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", os.path.join(vdir, f"{imp}.mp4")], check=True)
        jp = os.path.join(vdir, f"{imp}.json"); mm = json.load(open(jp)); mm.update(ctx_copied_until=int(t_ev), ctx_copied_from=pos)
        json.dump(mm, open(jp, "w"), indent=1)
    print(os.path.basename(vdir), "t_ev", t_ev, flush=True)
