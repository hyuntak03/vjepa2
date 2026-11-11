#!/usr/bin/env python3
"""BlenderTrack v3 확인용 — 상황마다 하나 렌더한 clip 을 상황 폴더로 정리 (2026-11-01).

<out>/preview/
  README.md                         상황 표 (30) · 폴더 구조 · 읽는 법
  overview_<camera>.mp4             카메라마다 상황 10 개를 한 화면 (4 × 3 격자, 10 fps 느리게)
  overview_<objset>_<place>_<camera>.mp4   (물체 셋 · 장소 · 카메라마다)
  <objset>/<place>/<camera>/<가림 종류>_<가림 움직임>/target_<moving|static|none>/
      clip.mp4                      원본 (20 fps, 3.2 초)
      clip_slow_contour.mp4         10 fps · target 윤곽 (주황 = 가려져도 있는 자리, 파랑 = 보이는 부분)
      sheet.png                     8 장 (문맥 0 · 5 · 10 · 15 | 미래 16 · 20 · 25 · 31 샘플)
      info.txt                      상황 · 물체 수 · 가림막 · target · 가림 샘플 · 화면 크기 · 카메라
  (설계 불가 상황은 폴더에 INFEASIBLE.txt)
  python organize_preview_v3.py --config configs/v3.yaml
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_design_v3 as G  # noqa: E402

KO = {"front": "정면", "side": "측면 (눈높이 비스듬히)", "bird": "위 (bird eye)", "barrier": "가림막", "object": "다른 물체", "static": "정지", "moving": "움직임",
      "none": "없음 (빈 장면)", "primitive": "primitive", "realistic": "realistic", "flat": "평면", "ramp": "빗면"}
SUB = {"board": "판", "wall": "낮은 벽", "table": "탁자", "rail_pass": "레일 위 판 (지나감)", "rail_same": "레일 위 판 (같은 방향)",
       "rail_opposite": "레일 위 판 (반대 방향)", "rolling_ball": "굴러가는 공", "cube": "상자", "cylinder": "통", "sphere": "큰 공", "capsule": "키 큰 물체",
       "plastic_bin": "플라스틱 통", "cardboard_box": "종이 상자", "wooden_crate": "나무 상자", "beach_ball": "비치볼"}


def edge(m):
    M = np.kron(m, np.ones((4, 4), bool))
    inner = M.copy()
    inner[1:-1, 1:-1] = M[:-2, 1:-1] & M[2:, 1:-1] & M[1:-1, :-2] & M[1:-1, 2:] & M[1:-1, 1:-1]
    return M & ~inner


def text(img, s, x=3, y=3):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img); d = ImageDraw.Draw(im)
    d.rectangle([x - 2, y - 2, x + 6 * len(s) + 2, y + 10], fill=(0, 0, 0))
    d.text((x, y), s, fill=(255, 255, 255))
    return np.asarray(im)


def frames_with_contour(out, cid):
    import decord
    v = decord.VideoReader(f"{out}/clips/{cid}.mp4")[:].asnumpy()
    z = np.load(f"{out}/labels/{cid}.npz")
    if z["amodal"].shape[1] > 0 and "target" in [str(r) for r in z["roles"]]:
        am = np.unpackbits(z["amodal"][:, 0], axis=-1).astype(bool); vi = np.unpackbits(z["visible"][:, 0], axis=-1).astype(bool)
        v = v.copy()
        for t in range(len(v)):
            v[t][edge(am[t])] = (235, 104, 52); v[t][edge(vi[t])] = (42, 120, 214)
    return v


def tag(c):
    rm = f" {c['ramp_mode']}" if c.get("ramp_mode") else ""
    return f"{c['objset'][:4]} {c['place']}{rm} {c['camera']} {c['occ_type']}-{c['occ_motion']} tgt:{c['target']} n{c.get('count', '-')}"


def write_mp4(path, frames, fps):
    H, W = frames[0].shape[:2]
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
                          "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close(); p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = G.load_cfg(a.config, "preview")
    out = cfg["out"]
    P = f"{out}/preview"
    if os.path.isdir(P):
        shutil.rmtree(P)
    S = G.situations(cfg)
    sd = cfg["splits"]["preview"]["seed"]
    rows, tiles = [], {}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for i, sit in enumerate(S):
        cid = f"preview{sd}_{i:06d}"
        d = f"{P}/{sit['objset']}/{sit['place']}/{sit['camera']}/{sit['occ_type']}_{sit['occ_motion']}/target_{sit['target']}"
        os.makedirs(d, exist_ok=True)
        ok = os.path.exists(f"{out}/clips/{cid}.mp4") and os.path.exists(f"{out}/labels/{cid}.npz")
        if not ok:
            open(f"{d}/INFEASIBLE.txt", "w").write(
                "설계 불가 (1,500 × 4 번 시도) — 이 상황은 지금 규칙 (화면 크기 · 가림 타이밍 · 겹침 금지) 안에서 장면을 못 찾았다.\n")
            rows.append((sit, None)); tiles.setdefault((sit["objset"], sit["place"], sit["camera"]), []).append((sit, None))
            continue
        sp = json.load(open(f"{out}/specs/{cid}.json"))
        c = sp["cell"]
        shutil.copy(f"{out}/clips/{cid}.mp4", f"{d}/clip.mp4")
        fr = frames_with_contour(out, cid)
        write_mp4(f"{d}/clip_slow_contour.mp4", [text(f, tag(c)) for f in fr], 10)
        idx = [0, 10, 20, 30, 32, 40, 50, 62]
        fig, ax = plt.subplots(1, 8, figsize=(16, 2.4))
        for j, t in enumerate(idx):
            ax[j].imshow(fr[t]); ax[j].set_xticks([]); ax[j].set_yticks([])
            ax[j].set_title(f"s{t // 2}" + (" (ctx end)" if t == 30 else ""), fontsize=8)
        fig.suptitle(tag(c), fontsize=9)
        plt.tight_layout(); plt.savefig(f"{d}/sheet.png", dpi=90); plt.close(fig)
        cam = sp["cam"]
        objs = [o for o in sp["objects"]]
        info = [f"상황: 물체 셋 {c['objset']} · 장소 {KO[c['place']]}" + (f" ({'올라감' if c['ramp_mode'] == 'up' else '내려감'})" if c.get('ramp_mode') else "")
                + f" · 카메라 {KO[c['camera']]} · 가림 {KO[c['occ_type']]} ({KO[c['occ_motion']]}) · target {KO[c['target']]}",
                f"target 속력: 가려지기 시작할 때 {sp['target_speed_at_hide']:.2f} m/s · 최대 {sp['target_speed_max']:.2f} m/s",
                f"물체 수 (움직일 수 있는 물체 총수): {c['count']}  — 장면의 물체: " + ", ".join(f"{o['role']}={o['shape']} {o['s']:.2f} m ({o['motion']['kind']})" for o in objs),
                f"가리는 것: {SUB.get(c['occ_sub'], c['occ_sub'])}",
                f"target: {c.get('target_kind') or c['target_shape']} ({c['target_shape']}) · {'움직임' if c['target_mode'] == 'moving' else '정지'}" + ("  (빈 장면: 가림 배치에만 쓰고 렌더 안 함)" if c["target"] == "none" else ""),
                f"가림 (target 완전히 가려진 샘플, 설계): {sp['event_samples'][0]}–{sp['event_samples'][1]}  (문맥 끝 = 15)",
                f"target 화면 지름 (중앙값): {sp['target_px_median']:.1f} px",
                f"카메라: 높이 {cam['h']:.2f} m · 기울기 {cam['pitch']:.1f}° · 방향 {cam['yaw']:.1f}° · 렌즈 {cam['lens']:.0f} mm · target 까지 {cam['dist']:.2f} m",
                f"환경: {'실내' if sp['env']['indoor'] else '실외'} · 바닥 {sp['env']['floor']} · 조명 {sp['env']['hdri']}",
                f"clip id: {cid}"]
        open(f"{d}/info.txt", "w").write("\n".join(info) + "\n")
        rows.append((sit, sp)); tiles.setdefault((sit["objset"], sit["place"], sit["camera"]), []).append((sit, [text(f, tag(c)) for f in fr]))
    # 카메라마다 한 화면
    for (objset, place, cam), L in tiles.items():
        n, gw = len(L), 4
        gh = (n + gw - 1) // gw
        T = 64
        blank = np.zeros((256, 256, 3), np.uint8)
        frames = []
        for t in range(T):
            canvas = np.zeros((gh * 256, gw * 256, 3), np.uint8)
            for k, (sit, fr) in enumerate(L):
                r, cc = divmod(k, gw)
                tile = fr[t] if fr is not None else text(blank.copy(), f"{sit['occ_type']}-{sit['occ_motion']} tgt:{sit['target']} INFEASIBLE", 3, 120)
                canvas[r * 256:(r + 1) * 256, cc * 256:(cc + 1) * 256] = tile
            frames.append(canvas)
        write_mp4(f"{P}/overview_{objset}_{place}_{cam}.mp4", frames, 10)
    # README
    md = ["# BlenderTrack v3 확인용 — 상황마다 하나 (2026-11-01)", "",
          "가림은 모두 late (문맥 끝 샘플 15 에서 target 이 완전히 가려지고, 미래에서 다시 나온다). 실제 축척 · 중력 9.81 · 그림자 · 실내/실외 조명.", "",
          "폴더 = `<물체 셋>/<장소>/<카메라>/<가림 종류>_<가림 움직임>/target_<moving|static|none>/` — 안에 clip.mp4 (원본) · clip_slow_contour.mp4 (느리게 + 윤곽) · sheet.png · info.txt",
          "한눈에 = `overview_<물체 셋>_<장소>_<카메라>.mp4` (그 묶음의 상황을 한 화면에)", "",
          "| # | 물체 셋 | 장소 | 카메라 | 가림 종류 | 가림 움직임 | target | 물체 수 | 가리는 것 | target | 가림 샘플 | 가려질 때 속력 | 화면 지름 |",
          "|---|---|---|---|---|---|---|---:|---|---|---|---:|---:|"]
    for k, (sit, sp) in enumerate(rows):
        if sp is None:
            md.append(f"| {k} | {sit['objset']} | {KO[sit['place']]} | {KO[sit['camera']]} | {KO[sit['occ_type']]} | {KO[sit['occ_motion']]} | {KO[sit['target']]} | – | **설계 불가** | – | – | – | – |")
        else:
            c = sp["cell"]
            rm = (" ↑" if c.get("ramp_mode") == "up" else " ↓") if c.get("ramp_mode") else ""
            md.append(f"| {k} | {c['objset']} | {KO[c['place']]}{rm} | {KO[c['camera']]} | {KO[c['occ_type']]} | {KO[c['occ_motion']]} | {KO[c['target']]} | {c['count']} | "
                      f"{SUB.get(c['occ_sub'], c['occ_sub'])} | {c.get('target_kind') or c['target_shape']} | {sp['event_samples'][0]}–{sp['event_samples'][1]} | "
                      f"{sp['target_speed_at_hide']:.2f} m/s | {sp['target_px_median']:.0f} px |")
    open(f"{P}/README.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
