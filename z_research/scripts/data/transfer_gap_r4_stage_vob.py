"""transfer_gap R4 [X1] — VOBench 의 필요한 clip · 프레임만 256² 로 NFS 에 옮긴다 (2026-10-30).

왜: X2 의 전체 사본 (job 235557) 이 동시 실행 경합으로 중간에 죽었다. 같은 디렉터리에 다시 쓰면 경합이 되풀이되므로
    E37 · E38 에 필요한 clip (late · visible · 이동, 모든 변이) 의 샘플 프레임 32 장 (0, 3, …, 93) 만 따로 옮긴다.
동일성: z_research/scripts/data/transfer_gap_stage_vobench256.py 의 do_clip 과 같은 연산 (float/255 → bilinear (align_corners=False,
    antialias 없음) 256² → ×255 → round → clamp → uint8 PNG). 그래서 X2 사본과 비트 단위로 같은 파일이다 (같은 사본으로 친다).
재현: sbatch -p batch_vll -w vll5 --gres=gpu:0 -c 6 --mem=8G --wrap "$P z_research/scripts/data/transfer_gap_r4_stage_vob.py"
"""
import csv, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
SRC = "/data2/local_datasets/world/Benchmark/VOBench_v1"
DST = os.path.join(REPO, "z_pixel_decoder/runs/_frames/VOBench_v1_256_x1r4")
FR = [f"{i:06d}.png" for i in range(0, 94, 3)]


def do(rel):
    import torch, torch.nn.functional as F
    from PIL import Image
    torch.set_num_threads(1)
    os.makedirs(os.path.join(DST, rel), exist_ok=True)
    for fn in FR:
        op = os.path.join(DST, rel, fn)
        if os.path.exists(op): continue
        a = np.asarray(Image.open(os.path.join(SRC, rel, fn)).convert("RGB"))
        x = torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float().div_(255.0)[None]
        if tuple(x.shape[-2:]) != (256, 256):
            x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
        y = x[0].mul_(255.0).round_().clamp_(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
        Image.fromarray(y).save(op + ".tmp", format="PNG", compress_level=1); os.replace(op + ".tmp", op)
    return rel


if __name__ == "__main__":
    rows = [r for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv")))
            if r["occlusion"] in ("late", "visible") and r["motion"] != "flat_static"]
    rels = sorted({r["file_name"] for r in rows})
    print(len(rels), "clips", flush=True)
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as ex:
        for i, _ in enumerate(ex.map(do, rels, chunksize=4)):
            if i % 100 == 0: print(i, flush=True)
    n = sum(len([f for f in os.listdir(os.path.join(DST, r)) if f.endswith(".png")]) for r in rels)
    open(os.path.join(DST, "_DONE"), "w").write(f"{len(rels)} clips {n} files\n"); print("DONE", len(rels), n)
