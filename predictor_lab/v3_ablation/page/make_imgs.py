"""BlenderTrack v3 결과 페이지 (claude artifact) 의 예시 그림 — 문맥 3 장 · 미래 3 장 띠 (webp).

사용:
  python predictor_lab/v3_ablation/page/make_imgs.py --cands  # 후보 contact sheet 만 (스크래치에)
  python predictor_lab/v3_ablation/page/make_imgs.py          # PICK 의 블록으로 img/*.webp

프레임: 모델은 64 장 (20 fps) 중 짝수 장만 32 장 읽는다 (문맥 = 샘플 0–15, 미래 = 16–31).
띠의 장은 그 샘플 번호 (원본 프레임 = 2 × 샘플). 가림 구간 = index 의 event_s–event_e (샘플).
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

R = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
OUT = R / "predictor_lab/v3_ablation/page/img"
V3 = R / "data_csv/blender_track_v3/voe_test/index.csv"
V3T = R / "data_csv/blender_track_v3t/voe_test/index.csv"
CLV = R / "data_csv/clevrer_perm_v1/index.csv"
CLV_ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/testbed/cleverer_permanence_v1")
VOMO = R / "data_csv/vo_moving_occluder/index.csv"
VOMO_FR = R / "z_pixel_decoder/runs/_frames/VO_moving_occluder_s3"
GAP = 4
SZ = 256

# 페이지에 쓸 블록 (후보를 눈으로 보고 고른다)
PICK = {
    # 이름: (세트, 블록, 저장할 띠)  — 2026-11-01 후보 contact sheet 를 보고 골랐다
    "board_front": ("v3", "test41_b002394", ["ctx", "pos", "imp", "empty"]),   # 고정 판 · 정면 · 평지 · target 움직임 · primitive
    "rail_side": ("v3", "test41_b000197", ["ctx", "pos", "imp"]),              # 움직이는 레일 판 · 측면 · target 정지 · realistic
    "rolling_ball": ("v3", "test41_b001349", ["ctx", "pos", "imp"]),           # 굴러오는 공이 가림 · target 정지
    "stand_object": ("v3", "test41_b002445", ["ctx", "pos", "imp"]),           # 서 있는 물체가 가림 · 측면 · target 움직임
    "ramp_a": ("v3", "test41_b001361", ["ctx", "pos", "imp"]),                 # 빗면 · 측면 · 고정 판
    "ramp_b": ("v3", "test41_b003125", ["ctx", "pos", "imp"]),
    "bird": ("v3", "test41_b003835", ["ctx", "pos", "imp"]),                   # 위에서 · 움직이는 판
    "trapdoor": ("v3t", "test61_b000516", ["ctx", "pos", "imp"]),              # trapdoor · 정면
}


def read_mp4(path, idx):
    import decord
    vr = decord.VideoReader(str(path))
    idx = [min(int(i), len(vr) - 1) for i in idx]
    return [Image.fromarray(f) for f in vr.get_batch(idx).asnumpy()]


def strip(frames, size=SZ):
    w = len(frames) * size + (len(frames) - 1) * GAP
    im = Image.new("RGB", (w, size), (255, 255, 255))
    for i, f in enumerate(frames):
        im.paste(f.convert("RGB").resize((size, size), Image.BILINEAR), (i * (size + GAP), 0))
    return im


def v3_frames(row_pos, row_imp, row_empty=None):
    """샘플 번호: 문맥 = [event_s-8, event_s-3, 15], 미래 = [event_e, (event_e+31)//2, 31]."""
    es, ee = int(row_pos.event_s), int(row_pos.event_e)
    ctx = [max(0, es - 8), max(1, es - 3), 15]   # 보임 · 들어가기 직전 · 문맥 끝 (가려짐)
    fut = sorted({min(31, max(16, ee)), min(31, max(17, (ee + 31) // 2)), 31})
    while len(fut) < 3:
        fut = sorted(set(fut) | {fut[0] + 1})
    raw = lambda s: [2 * x for x in s]
    out = {"ctx": strip(read_mp4(row_pos.file, raw(ctx))),
           "pos": strip(read_mp4(row_pos.file, raw(fut))),
           "imp": strip(read_mp4(row_imp.file, raw(fut))),
           "samples": (ctx, fut)}
    if row_empty is not None:
        out["empty_ctx"] = strip(read_mp4(row_empty.file, raw(ctx)))
        out["empty_fut"] = strip(read_mp4(row_empty.file, raw(fut)))
    return out


def block_rows(df, bid):
    b = df[df.block_id == bid].set_index("variant")
    return b.loc["pos_a"], b.loc["imp_ab"], b.loc["pos_b"]


def v3_candidates():
    d = pd.read_csv(V3)
    t = pd.read_csv(V3T)
    pa = d[d.variant == "pos_a"]
    rng = np.random.default_rng(0)
    q = {
        "static_board_front": (pa.occ_sub.isin(["board", "wall"])) & (pa.camera == "front") & (pa.place == "flat") & (pa.target_mode == "moving"),
        "rail_side": (pa.occ_sub == "rail_pass") & (pa.camera == "side") & (pa.place == "flat"),
        "rolling_ball": (pa.occ_sub == "rolling_ball") & (pa.camera == "front"),
        "stand_object": (pa.occ_type == "object") & (pa.occ_motion == "static"),
        "ramp_side": (pa.place == "ramp") & (pa.camera == "side"),
        "bird": (pa.camera == "bird") & (pa.occ_type == "barrier"),
        "table": (pa.occ_sub == "table"),
    }
    out = {}
    for k, m in q.items():
        bids = pa[m & (pa.f_hide == "short")].block_id.unique()
        out[k] = ("v3", list(rng.choice(bids, size=min(3, len(bids)), replace=False)))
    tp = t[(t.variant == "pos_a") & (t.camera == "front")]
    out["trapdoor"] = ("v3t", list(rng.choice(tp[tp.f_hide == "short"].block_id.unique(), size=3, replace=False)))
    return out, d, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", action="store_true")
    ap.add_argument("--sheet", default=os.environ.get("SHEET", "/tmp/v3_cands.png"))
    a = ap.parse_args()
    cands, d, t = v3_candidates()
    if a.cands:
        rows = []
        for k, (src, bids) in cands.items():
            df = d if src == "v3" else t
            for bid in bids:
                p, i, e = block_rows(df, bid)
                f = v3_frames(p, i)
                im = Image.new("RGB", (3 * (f["ctx"].width + 12), SZ + 22), (255, 255, 255))
                for j, key in enumerate(["ctx", "pos", "imp"]):
                    im.paste(f[key], (j * (f["ctx"].width + 12), 22))
                ImageDraw.Draw(im).text((4, 4), f"{k} {bid} {p.condition} {p.target_kind} occ={p.occ_sub} f_count={p.f_count} samples={f['samples']}", fill=(0, 0, 0))
                rows.append(im.resize((im.width // 3, im.height // 3)))
        sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height + 4 for r in rows)), (220, 220, 220))
        y = 0
        for r in rows:
            sheet.paste(r, (0, y)); y += r.height + 4
        sheet.save(a.sheet)
        print("->", a.sheet, sheet.size)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (src, bid, keys) in PICK.items():
        df = {"v3": d, "v3t": t}[src]
        p, i, e = block_rows(df, bid)
        f = v3_frames(p, i, e if "empty" in keys else None)
        for key in keys:
            ks = ["empty_ctx", "empty_fut"] if key == "empty" else [key]
            for kk in ks:
                f[kk].save(OUT / f"{name}_{kk}.webp", quality=82, method=6)
        print(name, bid, p.condition, p.target_kind, p.occ_sub, "samples", f["samples"])


if __name__ == "__main__":
    sys.exit(main())


# ---- 시험 예시 (CLEVRER style · VO_moving_occluder) ------------------------------------------
def clevrer_strips(scene_vid_prefix):
    """CLEVRER style: 영상이 이미 32 장 (문맥 16 + 미래 16). present = 가능, vanish = 사라짐."""
    c = pd.read_csv(CLV)
    rows = c[c.video_id.str.startswith(scene_vid_prefix)].set_index("variant")
    ctx, fut = [5, 10, 15], [20, 25, 31]
    p, i = rows.loc["pos_a"], rows.loc["imp_ab"]
    return {"ctx": strip(read_mp4(CLV_ROOT / p.file, ctx)), "pos": strip(read_mp4(CLV_ROOT / p.file, fut)),
            "imp": strip(read_mp4(CLV_ROOT / i.file, fut))}


def vomo_strips(block_id):
    """VO_moving_occluder: 0,3,…,93 의 32 장 (문맥 = 앞 16)."""
    v = pd.read_csv(VOMO)
    rows = v[v.block_id == block_id].set_index("variant")
    ctx, fut = [5, 11, 15], [20, 25, 31]
    load = lambda fn, s: [Image.open(VOMO_FR / fn / f"{3 * k:06d}.png") for k in s]
    p, i = rows.loc["pos_a"], rows.loc["imp_ab"]
    return {"ctx": strip(load(p.file_name, ctx)), "pos": strip(load(p.file_name, fut)), "imp": strip(load(i.file_name, fut))}


# 시험 예시로 고른 것 (2026-11-01): CLEVRER style 장면 00007 (cross) · VO_moving_occluder block 126 (real food, 움직이는 밴)
#   python -c "import make_imgs as m; [f[k].save(m.OUT/f'{n}_{k}.webp', quality=82, method=6) for n, f in [('clevrer', m.clevrer_strips('00007')), ('vomo', m.vomo_strips(126))] for k in ['ctx','pos','imp']]"
