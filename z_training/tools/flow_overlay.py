#!/usr/bin/env python
"""영상 프레임 위에 flow 캐시의 운동장을 겹쳐 그린다 (2026-10-24, 사용자 질문:
"ssv2 에 optical flow overlap 한 video 볼 수 있어? 프레임 단위로 다 비슷해서 복사만으로 풀리는 문제라서 그런가").

무엇을 그리나 — 행 하나 = 영상 하나, 열 = fps 4 프레임.
  위 칸: 프레임 + **누적 변위** 화살표 (지평선 h 장, 기본 4 = 1 초). 앵커 격자는 16x16 (토큰 격자와 같다)
  아래 칸: 같은 프레임의 |d| 히트맵 (패치 단위, 공통 스케일)
유효하지 않은 칸 (RAFT 전후 일관성 탈락) 은 화살표를 안 그린다.

    python z_training/tools/flow_overlay.py --dataset ssv2 --n 6 --horizon 4 --out /tmp/x.png
    python z_training/tools/flow_overlay.py --dataset k400 --n 6 --out /tmp/y.png
"""
import argparse, os, random, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import decord

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from app.vjepa_frozen.flow_labels import FlowCache, block_traj   # noqa: E402

ROOTS = {"k400": [f"/data/hyuntak/flow_cache/vll7_k400_320/s{i}" for i in range(4)],
         "ssv2": [f"/data/hyuntak/flow_cache/vll7_ssv2/t{i}" for i in range(8)]}
TABLE = {"k400": "data_csv/vll7_k400_320/fps4_table.tsv", "ssv2": "data_csv/vll7_ssv2/fps4_table.tsv"}

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="ssv2", choices=list(ROOTS))
ap.add_argument("--n", type=int, default=6, help="영상 수")
ap.add_argument("--cols", type=int, default=6, help="열 (fps4 프레임 수)")
ap.add_argument("--horizon", type=int, default=4, help="누적 지평선 (fps4 장). 4 = 1 초")
ap.add_argument("--grid", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
ap.add_argument("--gif", action="store_true", help="격자 PNG 대신 영상마다 GIF 하나 (--out 은 폴더)")
ap.add_argument("--gif-frames", type=int, default=40, help="GIF 길이 (fps4 장)")
a = ap.parse_args()

HZ = [1, 2, 4, 8]
assert a.horizon in HZ, f"--horizon 은 {HZ} 중 하나 (캐시 라벨 지평선)"
fc = FlowCache(ROOTS[a.dataset])

# fps4 표: 경로 -> 원본 프레임 번호 목록 (캐시의 쌍 j 는 표의 j 번째 장 -> j+1 번째 장)
fps4 = {}
for ln in open(os.path.join(REPO, TABLE[a.dataset]), encoding="utf-8"):
    f = ln.rstrip("\n").split("\t")
    if len(f) >= 2 and f[0] in fc.off:
        fps4[f[0]] = [int(x) for x in f[1].split(",")] if "," in f[1] else None

G = a.grid
NF = a.gif_frames if a.gif else a.cols
paths = [p for p in fc.off if fc.off[p][2] >= NF + a.horizon + 1]
random.Random(a.seed).shuffle(paths)
paths = paths[: a.n]


def read_clip(path, nf):
    vr = decord.VideoReader(path, num_threads=1)
    step = max(1, round(vr.get_avg_fps() / 4))
    return vr.get_batch([min(len(vr) - 1, j * step) for j in range(nf)]).asnumpy()


def labels(path, nf):
    si, off, _ = fc.off[path]
    need = nf + a.horizon + 1
    f = np.asarray(fc.flow[si][off:off + need], dtype=np.float32)
    v = np.asarray(fc.valid[si][off:off + need], dtype=np.float32) / 255.0
    d, ok = block_traj(f, v, nf, 1, HZ, j_off=0)
    ki = HZ.index(a.horizon)
    dm = torch.nn.functional.interpolate(d[:, ki], size=(G, G), mode="area")
    om = torch.nn.functional.interpolate(ok[:, ki].unsqueeze(1), size=(G, G), mode="area")[:, 0]
    return dm, om


if a.gif:
    from PIL import Image, ImageDraw
    os.makedirs(a.out, exist_ok=True)
    for p in paths:
        try:
            fr = read_clip(p, NF)
        except Exception as e:
            print("디코드 실패", p, e); continue
        dm, om = labels(p, NF)
        H, W = fr[0].shape[:2]
        sc = max(1, int(round(384 / max(H, W))))                      # 작은 영상은 키워서 화살표가 보이게
        ims = []
        for c in range(NF):
            im = Image.fromarray(fr[c]).resize((W * sc, H * sc), Image.BILINEAR)
            dr = ImageDraw.Draw(im)
            u = dm[c, 0].numpy() * W * sc; vv = dm[c, 1].numpy() * H * sc
            m = om[c].numpy() > 0.5
            for gy in range(G):
                for gx in range(G):
                    if not m[gy, gx]:
                        continue
                    x0 = (gx + .5) / G * W * sc; y0 = (gy + .5) / G * H * sc
                    x1 = x0 + float(u[gy, gx]); y1 = y0 + float(vv[gy, gx])
                    dr.line([x0, y0, x1, y1], fill=(0, 255, 100), width=1)
                    dr.ellipse([x1 - 1.3, y1 - 1.3, x1 + 1.3, y1 + 1.3], fill=(255, 40, 40))
            mag = float((dm[c].pow(2).sum(0).sqrt().numpy()[m] * 256 / 16).mean()) if m.any() else 0.0
            dr.text((4, 4), f"t={c*0.25:.2f}s  valid {m.mean()*100:.0f}%  |d| {mag:.2f} patch", fill=(255, 255, 0))
            ims.append(im)
        o = os.path.join(a.out, os.path.basename(p).replace(".mp4", "") + f"_h{a.horizon}.gif")
        ims[0].save(o, save_all=True, append_images=ims[1:], duration=250, loop=0)
        print("저장", o)
    sys.exit(0)

fig, axes = plt.subplots(2 * len(paths), a.cols, figsize=(2.1 * a.cols, 2.1 * 2 * len(paths)))
axes = np.atleast_2d(axes)
for r, p in enumerate(paths):
    si, off, npair = fc.off[p]
    need = a.cols + a.horizon + 1
    f = np.asarray(fc.flow[si][off:off + need], dtype=np.float32)
    v = np.asarray(fc.valid[si][off:off + need], dtype=np.float32) / 255.0
    d, ok = block_traj(f, v, a.cols, 1, HZ, j_off=0)              # tubelet 1 = 매 fps4 장이 앵커
    ki = HZ.index(a.horizon)
    dm = torch.nn.functional.interpolate(d[:, ki], size=(G, G), mode="area")        # (cols,2,G,G) 프레임비율
    om = torch.nn.functional.interpolate(ok[:, ki].unsqueeze(1), size=(G, G), mode="area")[:, 0]
    mag = (dm.pow(2).sum(1).sqrt() * 256 / 16).numpy()            # 패치 단위
    try:
        vr = decord.VideoReader(p, num_threads=1)
        step = max(1, round(vr.get_avg_fps() / 4))
        fr = vr.get_batch([min(len(vr) - 1, j * step) for j in range(a.cols)]).asnumpy()
    except Exception as e:
        print("디코드 실패", p, e); continue
    ys, xs = np.mgrid[0:G, 0:G]
    for c in range(a.cols):
        ax = axes[2 * r, c]; ax.imshow(fr[c]); ax.set_xticks([]); ax.set_yticks([])
        H, W = fr[c].shape[:2]
        u = dm[c, 0].numpy() * W; vv = dm[c, 1].numpy() * H       # 프레임비율 -> 픽셀
        m = om[c].numpy() > 0.5
        # 유효한 칸만 그린다 (무효 = RAFT 전후 일관성 탈락 = "라벨 없음", 0 이 아니다)
        if m.any():
            ax.quiver(((xs + .5) / G * W)[m], ((ys + .5) / G * H)[m], u[m], vv[m],
                      angles="xy", scale_units="xy", scale=1, width=.004, color="#00ff66")
        ax.set_xlabel(f"valid {m.mean()*100:.0f}%", fontsize=7)
        if c == 0: ax.set_ylabel(os.path.basename(p)[:18], fontsize=7)
        if r == 0: ax.set_title(f"t={c*0.25:.2f}s", fontsize=8)
        ax2 = axes[2 * r + 1, c]
        im = ax2.imshow(np.where(m, mag[c], np.nan), vmin=0, vmax=2.0, cmap="magma"); ax2.set_xticks([]); ax2.set_yticks([])
        ax2.set_xlabel(f"|d| {mag[c][m].mean() if m.any() else 0:.2f} p", fontsize=7)
fig.suptitle(f"{a.dataset} — cumulative displacement over {a.horizon} fps4 frames ({a.horizon*0.25:.2f} s). "
             f"arrows = per-token displacement, VALID cells only (16x16 grid); bottom = |d| in patch units (0-2), grey = no label", fontsize=10)
fig.colorbar(im, ax=axes, shrink=.3, label="|d| (patch = 16 px)")
fig.savefig(a.out, dpi=110, bbox_inches="tight")
print("저장", a.out, "·", len(paths), "영상")
