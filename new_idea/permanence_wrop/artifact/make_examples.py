#!/usr/bin/env python3
"""설명 페이지용 데이터 예시 그림 (2026-11-01). 문맥 3 장 · 미래 3 장 띠를 webp 로 (틈은 투명).
학습: SSv2 · WROP 가림 (op1 · op2) · WROP 비영속 (s1 · s2 · s3)
평가: VOBench · IntPhys1 · IntPhys2 · real_bench — 같은 문맥에서 가능 / 불가능 미래 (차이가 가장 큰 미래 프레임 포함)
"""
import csv, json, os, sys
import numpy as np
from PIL import Image
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")
H = 216; GAP = 6

def strip(frames):
    ims = [Image.fromarray(f).convert("RGB") for f in frames]
    ims = [im.resize((round(im.width * H / im.height), H), Image.BILINEAR) for im in ims]
    W = sum(im.width for im in ims) + GAP * (len(ims) - 1)
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0)); x = 0
    for im in ims:
        canvas.paste(im, (x, 0)); x += im.width + GAP
    return canvas

def save(name, frames):
    strip(frames).save(os.path.join(OUT, name + ".webp"), "WEBP", quality=80, method=6)

def vid(path, idx):
    import decord
    vr = decord.VideoReader(path, num_threads=2)
    idx = [min(i, len(vr) - 1) for i in idx]
    return list(vr.get_batch(idx).asnumpy())

def pngs(d, idx, pat="{:06d}.png"):
    return [np.array(Image.open(os.path.join(d, pat.format(i))).convert("RGB")) for i in idx]

def pick_future(fp, fi, n_ctx=16):
    """미래 프레임 셋: 차이가 가장 큰 칸을 가운데에 (그 앞 · 뒤 하나씩)."""
    d = [np.abs(a.astype(float) - b.astype(float)).mean() for a, b in zip(fp, fi)]
    k = int(np.argmax(d)); k = min(max(k, 4), len(d) - 5)
    return [max(0, k - 5), k, min(len(d) - 1, k + 5)], d

def pair(name, ctx_frames, fut_pos, fut_imp):
    sel, d = pick_future(fut_pos, fut_imp)
    save(name + "_ctx", ctx_frames)
    save(name + "_pos", [fut_pos[i] for i in sel]); save(name + "_imp", [fut_imp[i] for i in sel])
    print(name, "future sel", sel, "max diff", round(max(d), 2))

CTX = [0, 8, 15]
which = sys.argv[1:] or ["train", "test"]
if "train" in which:
    # SSv2 (12 fps, 연속 32 장)
    ss = json.load(open("/workspace/dataset/training_data/SSv2/labels/train.json"))
    lab = {t["id"]: t["label"] for t in ss}
    for sid in os.environ.get("SSV2_IDS", "").split():
        fr = vid(f"/workspace/dataset/training_data/SSv2/20bn-something-something-v2/{sid}.webm", [0, 10, 20, 31])
        save(f"ssv2_{sid}", fr); print("ssv2", sid, lab.get(sid))
    man = list(csv.DictReader(open("/workspace/world_model/vjepa2/data_csv/wrop_uni32/train_manifest.tsv"), delimiter="\t"))
    for task in os.environ.get("WROP_TASKS", "").split():
        rs = sorted([r for r in man if r["task"] == task], key=lambda r: r["path"])
        r = rs[int(os.environ.get("WROP_I", "3"))]
        fr = vid(r["path"], CTX + [20, 26, 31])
        save(f"wrop_{task}_ctx", fr[:3]); save(f"wrop_{task}_fut", fr[3:]); print("wrop", task, r["family"], r["path"])
if "test" in which:
    LOC = "/workspace/world_model/local_datasets/world/world_analysis"
    # VOBench: raw 96 장 stride 3 → 32 장, 문맥 raw 0–45 · 미래 raw 48–93
    vb = "/workspace/dataset/benchmarks/VOBench_v1"
    base = os.environ.get("VOB", "primitive/permanence/late_k2/v11_flat_vel_torus_green")
    raw = list(range(0, 96, 3))
    fp = pngs(f"{vb}/{base}/pos_a", raw); fi = pngs(f"{vb}/{base}/imp_ab", raw)
    pair("vobench", [fp[i] for i in CTX], fp[16:], fi[16:])
    # IntPhys1 late uni (32 장)
    i1 = os.environ.get("I1", "O1_01")
    rows = [r for r in csv.DictReader(open("/workspace/world_model/vjepa2/data_csv/intphys1_late_uni/index.csv"))]
    blk = [r for r in rows if r["video_id"].startswith(i1 + "_")]
    p1 = [r for r in blk if r["pair_id"] == os.environ.get("I1P", "1")]
    vp = [r for r in p1 if r["plausible"] == "1"][0]["video_id"]; vi = [r for r in p1 if r["plausible"] == "0"][0]["video_id"]
    fp = pngs(f"{LOC}/IntPhys1_late_uni/{vp}", range(32)); fi = pngs(f"{LOC}/IntPhys1_late_uni/{vi}", range(32))
    pair("intphys1", [fp[i] for i in CTX], fp[16:], fi[16:]); print("  intphys1", vp, vi)
    # IntPhys2 violation uni
    rows = list(csv.DictReader(open("/workspace/world_model/vjepa2/data_csv/intphys2_violation_uni/index.csv")))
    b2 = os.environ.get("I2", "permanence|105|1")
    vp = [r for r in rows if r["block_id"] == b2 and r["plausible"] == "1"][0]["video_id"]
    vi = [r for r in rows if r["block_id"] == b2 and r["plausible"] == "0"][0]["video_id"]
    fp = pngs(f"{LOC}/IntPhys2_violation_uni/{vp}", range(32)); fi = pngs(f"{LOC}/IntPhys2_violation_uni/{vi}", range(32))
    pair("intphys2", [fp[i] for i in CTX], fp[16:], fi[16:]); print("  intphys2", b2, vp[:8], vi[:8])
    # real_bench (32 장)
    rb = "/workspace/dataset/benchmarks/real_bench/rubric_office_quads_c16f16"
    sc = os.environ.get("RB", "2_teaduck/teaduck1")
    fp = pngs(f"{rb}/{sc}/possible_present", range(32)); fi = pngs(f"{rb}/{sc}/impossible_vanish", range(32))
    pair("realbench", [fp[i] for i in CTX], fp[16:], fi[16:]); print("  real", sc)
