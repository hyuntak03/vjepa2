"""VOBench late (UE trapdoor) 사라짐 쌍 36 개 vs Blender trapdoor 시험 사라짐 쌍 446 개를 같은 잣대로 (2026-11-01).

두 시험 모두 같은 장면의 "target 있음" (pos_a) · "target 없음" (pos_b) 영상이 있다 → 두 영상의 픽셀 차 = target (+그림자).
그래서 라벨 없이 같은 방식으로 잰다. 모델이 읽는 32 샘플 (둘 다 10 fps: VOBench 30 fps 간격 3, Blender 20 fps 간격 2) 에서:
  보임 비율 v_t = 차이 칸 수 / 문맥 최대        (0.5 이상 = 보임)
  가려진 마지막 보임 t_last (≤15) · 다시 보임 t_back (≥16) · 가림 길이 = t_back − t_last − 1 (샘플)
  물체 크기 = √(문맥 최대 차이 칸 수) (px) · 속도 = 마지막 보인 3 샘플의 중심 이동 (px/샘플)
  재등장 거리 = |c(t_back) − c(t_last)| (px) · 등속 외삽 오차 = |c(t_back) − (c(t_last) + v·(t_back − t_last))| (px)
  판 크기 = pos_b 에서 가림 한가운데 샘플 − 샘플 0 의 차이 덩어리 (target 이 숨은 자리에 걸친 것) 의 높이 · 넓이 (px)
쌍 정답 = table_v3.panel_scores (시트와 같은 배치 병합) 의 판정.
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; $P predictor_lab/v3_ablation/page/trapdoor_compare.py
  → predictor_lab/results/v3_ablation/trapdoor_compare.csv (쌍마다 한 줄)
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import table_v3 as T  # noqa: E402

R = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
VOB_FR = R / "z_pixel_decoder/runs/_frames/VOBench_v1_256full"
OUT = R / "predictor_lab/results/v3_ablation/trapdoor_compare.csv"
TAGS = ["copy", "release", "v3_mix1600_s0", "v3_mix1600_t25_s0", "v3t_only_s0", "vobftg"]
THR = 18.0          # 픽셀 차 (0–255, 채널 평균) 문턱


def load_vob(fn):
    return np.stack([np.asarray(Image.open(VOB_FR / fn / f"{3 * s:06d}.png").convert("RGB").resize((256, 256)), dtype=np.float32) for s in range(32)])


def load_mp4(path):
    import decord
    vr = decord.VideoReader(str(path))
    return vr.get_batch([2 * s for s in range(32)]).asnumpy().astype(np.float32)


def features(a, b):
    d = np.abs(a - b).mean(-1)                       # (32, 256, 256)
    m = d > THR
    m = np.stack([ndimage.binary_opening(x, iterations=1) for x in m])
    area = m.reshape(32, -1).sum(1).astype(float)
    S = area[:16].max()
    if S < 20:
        return None
    vis = area / S
    ctx_vis = [t for t in range(16) if vis[t] >= 0.5]
    fut_vis = [t for t in range(16, 32) if vis[t] >= 0.5]
    if not ctx_vis:
        return None
    t_last = max(ctx_vis)
    t_back = min(fut_vis) if fut_vis else np.nan
    yy, xx = np.mgrid[:256, :256]

    def cen(t):
        w = d[t] * m[t]
        s = w.sum()
        return np.array([(w * xx).sum() / s, (w * yy).sum() / s])

    c_last = cen(t_last)
    prev = [t for t in ctx_vis if t < t_last][-2:]
    vel = (c_last - cen(prev[0])) / (t_last - prev[0]) if prev else np.zeros(2)
    out = dict(size_px=np.sqrt(S), t_last=t_last, t_back=t_back, hide_len=(t_back - t_last - 1) if fut_vis else np.nan,
               speed=float(np.linalg.norm(vel)), vis_future=float(np.clip(vis[16:], 0, 1).mean()))
    if fut_vis:
        c_back = cen(t_back)
        out["reappear_dist"] = float(np.linalg.norm(c_back - c_last))
        out["extrap_err"] = float(np.linalg.norm(c_back - (c_last + vel * (t_back - t_last))))
        mid = int(round((t_last + t_back) / 2))
        pm = ndimage.binary_opening(np.abs(b[mid] - b[0]).mean(-1) > THR, iterations=2)
        lab, n = ndimage.label(pm)
        if n:
            hid = (c_last + c_back) / 2
            cx, cy = int(np.clip(hid[0], 0, 255)), int(np.clip(hid[1], 0, 255))
            k = lab[cy, cx]
            if k == 0:                                  # 숨은 자리에 덩어리가 없으면 가장 가까운 덩어리
                dist = ndimage.distance_transform_edt(lab == 0, return_indices=True)[1]
                k = lab[dist[0][cy, cx], dist[1][cy, cx]]
            ys, xs = np.nonzero(lab == k)
            out.update(panel_h=int(ys.max() - ys.min() + 1), panel_w=int(xs.max() - xs.min() + 1), panel_area=int(len(ys)))
            out["panel_over_obj"] = out["panel_area"] / S
    return out


def job(args):
    src, block, pa, pb = args
    try:
        a = load_vob(pa) if src == "vob" else load_mp4(pa)
        b = load_vob(pb) if src == "vob" else load_mp4(pb)
        f = features(a, b)
    except Exception as e:      # noqa: BLE001
        print("FAIL", src, block, e, flush=True)
        f = None
    return src, block, f


def main():
    rows = []
    for pn, src in [("vob_late", "vob"), ("v3ttest", "v3t")]:
        S, P = T.panel_scores(pn)
        f = dict(T.PANELS[pn])["A"]
        meta_dir = sorted((d for d in T.TE.glob(f"{pn}_*") if (d / "done.npy").exists()), key=lambda d: (d / "meta.json").stat().st_mtime)[0]
        idx = pd.read_csv(json.load(open(meta_dir / "meta.json"))["index_csv"])
        idx = idx.set_index("video_id")
        vids = json.load(open(meta_dir / "meta.json"))["video_id"]
        for a, bimp, r in P:
            if not f(r):
                continue
            row = {"src": src, "block": r["block_id"]}
            for t in TAGS:
                if t in S:
                    s = S[t]; row[t] = 1.0 if s[bimp] > s[a] else 0.5 if s[bimp] == s[a] else 0.0
            blk = idx[idx.block_id.astype(str) == str(r["block_id"])]
            pos_a = blk[blk.variant == "pos_a"].iloc[0]; pos_b = blk[blk.variant == "pos_b"].iloc[0]
            if src == "vob":
                row.update(motion=pos_a.motion, k=int(pos_a.k), objset=pos_a.objset, obj=pos_a.obj_a, background=pos_a.background)
                row["pa_path"], row["pb_path"] = pos_a.file_name, pos_b.file_name
            else:
                row.update(motion=f"{pos_a.place}_{pos_a.target_mode}", camera=pos_a.camera, objset=pos_a.objset, obj=pos_a.target_kind,
                           event_s=int(pos_a.event_s), event_e=int(pos_a.event_e), f_hide=pos_a.f_hide)
                row["pa_path"], row["pb_path"] = pos_a.file, pos_b.file
            rows.append(row)
    df = pd.DataFrame(rows)
    with ProcessPoolExecutor(24) as ex:
        res = list(ex.map(job, [(r.src, r.block, r.pa_path, r.pb_path) for r in df.itertuples()], chunksize=4))
    feats = pd.DataFrame([dict(src=s, block=b, **(f or {})) for s, b, f in res])
    df = df.drop(columns=["pa_path", "pb_path"]).merge(feats, on=["src", "block"], how="left")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print("->", OUT, df.shape)
    num = ["size_px", "speed", "t_last", "t_back", "hide_len", "reappear_dist", "extrap_err", "panel_h", "panel_w", "panel_over_obj", "vis_future"]
    for src, g in df.groupby("src"):
        print(f"\n== {src} n={len(g)}  acc: " + " · ".join(f"{t} {100 * g[t].mean():.1f}" for t in TAGS if t in g))
        print(g[num].describe(percentiles=[.1, .5, .9]).loc[["10%", "50%", "90%"]].round(1).to_string())


if __name__ == "__main__":
    main()
