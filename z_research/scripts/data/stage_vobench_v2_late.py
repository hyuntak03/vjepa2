"""VOBench_v2 late 가능 영상의 샘플 프레임 32 장 (0, 3, …, 93) 을 원본 바이트 그대로 NFS 로 옮긴다 (2026-10-31, z_analysis/release).

왜: VOBench_v2 원본 (313 GB) 은 vll5 로컬 /data2 에만 있고, vll5 GPU 가 다른 사람 job 으로 막혀 있다. vll2 에서 z · p 를 뽑으려고
    필요한 clip · 프레임만 NFS 에 둔다. 바이트 그대로라 채점 하네스 (원본 화소 → float 리사이즈) 가 vll5 원본에서 읽는 것과 같다.
대상: data_csv/vobench_v2/index.csv 의 occlusion == late · plausible == 1 (1296 clip).
재현: $P z_research/scripts/data/stage_vobench_v2_late.py   (vll5, CPU, ~15 GB)
"""
import csv, os, shutil, sys
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
SRC = "/data2/local_datasets/world/Benchmark/VOBench_v2"
DST = "/data/hyuntak/project/2026/2027_cvpr/nfs_cache/VOBench_v2_late_frames"
FR = [f"{i:06d}.png" for i in range(0, 94, 3)]


def do(rel):
    os.makedirs(os.path.join(DST, rel), exist_ok=True)
    for fn in FR:
        op = os.path.join(DST, rel, fn)
        if os.path.exists(op):
            continue
        shutil.copyfile(os.path.join(SRC, rel, fn), op + ".tmp"); os.replace(op + ".tmp", op)
    return rel


if __name__ == "__main__":
    rows = [r for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v2/index.csv")))
            if r["occlusion"] == "late" and r["plausible"] == "1"]
    rels = sorted({r["file_name"] for r in rows})
    print(len(rels), "clips", flush=True)
    with ThreadPoolExecutor(16) as ex:
        for i, _ in enumerate(ex.map(do, rels)):
            if i % 200 == 0:
                print(i, flush=True)
    open(os.path.join(DST, "_DONE"), "w").write(f"clips {len(rels)} frames {len(rels) * len(FR)}\n")
    print("done", flush=True)
    sys.exit(0)
