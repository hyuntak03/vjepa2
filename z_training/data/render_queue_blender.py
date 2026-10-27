"""phys3d_multi 렌더 워커 (2026-10-25) — 라벨 워커가 남긴 `<stem>_r.blend` (물리를 키프레임으로 구운 장면) 를 차례로 열어 저장 장 64 장 (3 장마다) 을 렌더한다.
GPU 에 오래 사는 프로세스 몇 개만 두어 CUDA 컨텍스트를 한 번만 만든다 (GPU 워커 128 개가 렌더마다 새로 잡다가 드라이버 잠금에서 멈췄다).

  blender -t 1 -b -noaudio -P z_training/data/render_queue_blender.py -- --root <out>/clips --claims <out>/_work/rclaims --done <out>/_work/labels_done --engine cycles
--done 파일이 생기고 남은 blend 가 없으면 끝난다.
"""
import argparse, glob, os, shutil, subprocess, sys, time

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_phys3d_blender as B           # noqa: E402

ST, W_RAW = 3, 192


def render_one(path, engine, res):
    stem = path[:-len("_r.blend")]
    bpy.ops.wm.open_mainfile(filepath=path)
    sc = bpy.context.scene
    s = int(sc["clip_start_render"])
    frames_dir = stem + "_frames"
    B.setup_render(sc, engine, res, os.path.join(frames_dir, "f"))
    # 렌더 워커는 GPU 마다 몇 개뿐이라 장면을 GPU 에 붙잡아 둬도 된다 (persistent data 끄면 장마다 다시 올려 0.8 초 / 장, 2026-10-25 실측)
    sc.render.use_persistent_data = os.environ.get("PHYS3D_RENDER_PERSIST", "1") == "1"
    os.makedirs(frames_dir, exist_ok=True)
    sc.frame_start, sc.frame_end, sc.frame_step = B.PRE + s, B.PRE + s + W_RAW - 1, ST
    try:
        bpy.ops.render.render(animation=True)
    except RuntimeError as err:
        print("GPU render failed, CPU fallback:", err, flush=True)
        B.setup_render(sc, "cpu", res, os.path.join(frames_dir, "f"))
        sc.frame_start, sc.frame_end, sc.frame_step = B.PRE + s, B.PRE + s + W_RAW - 1, ST
        bpy.ops.render.render(animation=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{B.FPS / ST:g}", "-pattern_type", "glob",
                    "-i", os.path.join(frames_dir, "f*.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", stem + ".mp4"], check=True)
    shutil.rmtree(frames_dir)
    os.remove(path)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--claims", required=True)
    ap.add_argument("--done", required=True)
    ap.add_argument("--engine", default="cycles")
    ap.add_argument("--res", type=int, default=256)
    a = ap.parse_args(argv)
    os.makedirs(a.claims, exist_ok=True)
    n = 0
    while True:
        todo = sorted(glob.glob(os.path.join(a.root, "*", "*_r.blend")))
        got = False
        for f in todo:
            try:
                os.close(os.open(os.path.join(a.claims, os.path.basename(f)), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            except FileExistsError:
                continue
            got = True
            t0 = time.time()
            try:
                render_one(f, a.engine, a.res)
                n += 1
                print(f"RENDERED {f} {time.time() - t0:.1f}s ({n})", flush=True)
            except Exception as err:             # 한 편이 깨져도 워커는 계속
                print(f"RENDER_FAIL {f} {err}", flush=True)
            break
        if not got:
            if os.path.exists(a.done):            # 라벨이 끝났고 아무도 안 가져간 blend 가 없다
                break
            time.sleep(3)


main()
