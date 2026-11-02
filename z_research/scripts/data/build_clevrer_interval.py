#!/usr/bin/env python3
"""CLEVRER permanence 를 **샘플 간격만 바꿔** 다시 구성한다 (2026-11-01, Rubric 비전이 조사 · 간격 가설).

왜: PFT 이득이 옮겨진 세트 (v3 test · VOBench · CLEVRER) 는 샘플 간격이 0.08–0.10 초, 안 옮겨진 Rubric · IntPhys 1 late 는 0.21–0.23 초다.
    간격 (= 샘플당 움직임 크기) 이 원인인지, 같은 CLEVRER 장면에서 간격만 두 배로 해서 가른다.
원본: testbed/cleverer_permanence_v1/scenes/<id>/frames_{target,empty}/frame_<원본번호>.png (25 fps 렌더, 짝수 번호만 36 장)
합성 규칙 (원 voe_ctx16_s2 클립과 화소 일치 검증, 12 장면 × 4 조건 × 32 장 불일치 0):
    원본 번호 < 다시 보이는 프레임 (clip_source_first + 2·reveal_clip) 이면 앞 소스, 아니면 뒤 소스.
    present = target|target · vanish = target|empty · appear = empty|target · absent = empty|empty.
창 (c_end = 원 창의 문맥 마지막 원본 번호 = clip_source_first + 30, 세 창 모두 문맥 끝이 같다):
    s2_c16t32  원 창 그대로 (0.08 초, 문맥 16 + 미래 16) — 재현 확인용 (원 채점은 mp4, 여기는 PNG)
    s2_c8t16   0.08 초, 문맥 8 + 미래 8   : 원본 c_end − 14 + 2i
    s4_c8t16   0.16 초, 문맥 8 + 미래 8   : 원본 c_end − 28 + 4i
출력: nfs_cache/clevrer_interval/<창>/<scene>/<조건>/000000.png … (원본 PNG 심볼릭 링크)
      data_csv/clevrer_interval_<창>/index.csv (원 index + file 비움 · file_name=<scene>/<조건>) · windows.json (장면별 원본 번호 · H · R)
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; $P z_research/scripts/data/build_clevrer_interval.py
"""
import csv
import json
import os

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
SRC = "/data/hyuntak/project/2026/2027_cvpr/testbed/cleverer_permanence_v1"
OUT = "/data/hyuntak/project/2026/2027_cvpr/nfs_cache/clevrer_interval"
COND = {"present": ("target", "target"), "vanish": ("target", "empty"), "appear": ("empty", "target"), "absent": ("empty", "empty")}
ARMS = {"s2_c16t32": (2, 16, 32), "s2_c8t16": (2, 8, 16), "s4_c8t16": (4, 8, 16)}   # (원본 보폭, 문맥 샘플, 전체 샘플)


def timing(vf, C):
    """vf (N,) 샘플별 보이는 비율 → H (문맥 끝 이어서 완전히 가려진 샘플 수, < 0.05) · R (다시 보이기 시작하는 첫 미래 샘플)."""
    h = 0
    while h < C and vf[C - 1 - h] < 0.05:
        h += 1
    r = np.nonzero(vf[C:] >= 0.05)[0]
    return h, int(r[0]) if len(r) else -1


def main():
    scenes = json.load(open(f"{SRC}/index.json"))["scenes"]
    rows = list(csv.DictReader(open(f"{REPO}/data_csv/clevrer_perm_v1/index.csv")))
    for arm, (st, C, N) in ARMS.items():
        win, skip = {}, []
        for s in scenes:
            d = f"{SRC}/{s['dir']}"
            have = {int(f[6:11]) for f in os.listdir(f"{d}/frames_target")} & {int(f[6:11]) for f in os.listdir(f"{d}/frames_empty")}
            c_end = s["clip_source_first"] + 30
            src = [c_end - st * (C - 1) + st * i for i in range(N)]
            if not set(src) <= have:
                skip.append(s["scene_id"]); continue
            rv = s["clip_source_first"] + 2 * s["reveal_clip"]
            ann = json.load(open(f"{d}/annotation.json"))
            tid = ann["occlusion"]["target_id"]
            fr = {f["frame_id"]: f for f in ann["motion_trajectory"]}
            vf = np.array([next(o for o in fr[n]["objects"] if o["object_id"] == tid).get("visible_fraction") or 0.0 for n in src], float)
            H, R = timing(vf, C)
            win[s["scene_id"]] = {"src": src, "reveal_source": rv, "H": H, "R": R, "splice_sample": int(np.searchsorted(src, rv))}
            for cond, (pre, post) in COND.items():
                od = f"{OUT}/{arm}/{s['scene_id']}/{cond}"
                os.makedirs(od, exist_ok=True)
                for i, n in enumerate(src):
                    k = pre if n < rv else post
                    lp = f"{od}/{i:06d}.png"
                    if os.path.lexists(lp):
                        os.remove(lp)
                    os.symlink(f"{d}/frames_{k}/frame_{n:05d}.png", lp)
            assert win[s["scene_id"]]["splice_sample"] >= C, (arm, s["scene_id"])   # 문맥은 두 짝이 같다
        dd = f"{REPO}/data_csv/clevrer_interval_{arm}"
        os.makedirs(dd, exist_ok=True)
        out = []
        for r in rows:
            sid = r["source_block"]
            if sid not in win:
                continue
            r = dict(r); r["file"] = ""; r["file_name"] = f"{sid}/{r['file_name'].rstrip('/').split('/')[-1]}"
            out.append(r)
        with open(f"{dd}/index.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(out)
        json.dump({"arm": arm, "stride_source": st, "sample_s": st / 25.0, "context": C, "n_frames": N, "skipped": skip, "scenes": win},
                  open(f"{dd}/windows.json", "w"))
        Hs = np.array([v["H"] for v in win.values()]); Rs = np.array([v["R"] for v in win.values()])
        print(f"{arm}: 장면 {len(win)} (빠짐 {len(skip)}) · clip {len(out)} · 샘플 {st / 25:.2f} 초 · H 중앙 {np.median(Hs):.0f} ({Hs.min()}–{Hs.max()}) · "
              f"R 중앙 {np.median(Rs):.0f} ({Rs.min()}–{Rs.max()}) · 미래 안에 다시 안 보임 {int((Rs < 0).sum())}")


if __name__ == "__main__":
    main()
