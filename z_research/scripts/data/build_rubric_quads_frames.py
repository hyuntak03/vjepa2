#!/usr/bin/env python3
"""Rubric Office quads (실제 영상, TCOW 기반 object permanence 2×2) → violation 시점 기준 문맥 16 + 미래 16 장 PNG + 표준 index (2026-10-31).

원본: /data/hyuntak/project/2026/2027_cvpr/testbed/real_bench/rubric_office_quads (2026-10-31 옮겨짐; 영상 20 × 4 = 80 clip, 640×480, README 참고).
violation 시점 t = info 의 t_splice (숨은 구간 가운데, 이 프레임부터 뒤쪽 소스).
창 두 가지 (프레임 번호는 실효 30 fps 기준 — 60 fps 중복 영상 (lightball) 은 짝수 프레임만, 번호 //2):
  uniform    문맥 = [0, t) 에서 균등 16 장 · 미래 = [t, 끝) 에서 균등 16 장 (사용자 요청 "violation 앞뒤 uniform 16 frame").
             앞뒤 간격이 영상마다 다르다.
  symmetric  앞뒤 같은 간격 s = min(t, 끝 − t) / 16 — 문맥 = t − 16s … t − s, 미래 = t … t + 15s (점검용: 블록 간격 일정).
변이 → 표준 이름 (VOBench 와 같은 2×2): possible_present = pos_a (A|A) · impossible_vanish = imp_ab (A|B) ·
  impossible_appear = imp_ba (B|A) · possible_absent = pos_b (B|B). 짝 A = (pos_a, imp_ab) 문맥 '있음' · 짝 B = (pos_b, imp_ba) 문맥 '없음'.
감사: 뽑은 문맥 16 장이 짝끼리 화소로 같은가 (재인코딩 차). 미래 창에 '다시 보임' 프레임이 들어가는가.
출력: <OUT>/<창>/<scene>/<video>/<variant>/000000.png … 000031.png  ·  data_csv/rubric_quads_<창>/index.csv · audit.json
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; $P z_research/scripts/data/build_rubric_quads_frames.py
"""
import csv, json, os
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
SRC = "/data/hyuntak/project/2026/2027_cvpr/testbed/real_bench/rubric_office_quads"   # 2026-10-31 옮겨짐 (옛 /data/hyuntak/project/2026/2027_cvpr/rubric_office_quads)
OUT = "/data/hyuntak/project/2026/2027_cvpr/nfs_cache/rubric_quads_frames"
VAR = {"possible_present": ("pos_a", "1", "A"), "impossible_vanish": ("imp_ab", "0", "A"),
       "impossible_appear": ("imp_ba", "0", "B"), "possible_absent": ("pos_b", "1", "B")}
N = 16


def windows(t, n):
    u_ctx = np.round(np.linspace(0, t - 1, N)).astype(int)
    u_fut = np.round(np.linspace(t, n - 1, N)).astype(int)
    s = min(t, n - t) / N
    s_ctx = np.round(t - s * np.arange(N, 0, -1)).astype(int)
    s_fut = np.round(t + s * np.arange(N)).astype(int)
    return {"uniform": (u_ctx, u_fut), "symmetric": (s_ctx, s_fut)}


def main():
    import argparse
    import decord
    from PIL import Image
    global SRC, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC, help="원본 quads 폴더 (index.json + <scene>/<video>/<variant>.mp4)")   # 2026-10-31: 노드 로컬 사본 (vll7) 에서도 뽑게
    ap.add_argument("--out", default=OUT, help="PNG 출력 루트 (<out>/<창>/…)")
    ap.add_argument("--windows", default="uniform,symmetric")
    ap.add_argument("--no-index", action="store_true", help="data_csv/rubric_quads_<창> (index · audit) 를 다시 쓰지 않는다 (프레임만)")
    a = ap.parse_args()
    SRC, OUT = a.src, a.out
    keep = set(a.windows.split(","))
    V = json.load(open(f"{SRC}/index.json"))["videos"]
    audit = {}
    rows = {w: [] for w in ("uniform", "symmetric")}
    for bi, v in enumerate(V):
        d = v["dir"]; dup = bool(v["duplicated_frames"]); step = 2 if dup else 1
        n = v["num_frames"] // step; t = v["t_splice"] // step; rev = v["first_visible_after_reveal"] // step
        last = v["last_visible_before_hiding"] // step
        vr = {k: decord.VideoReader(f"{SRC}/{d}/{k}.mp4") for k in VAR}
        for w, (ctx, fut) in windows(t, n).items():
            if w not in keep:
                continue
            idx_eff = np.concatenate([ctx, fut]); assert (np.diff(idx_eff) > 0).all() and idx_eff[-1] < n, (d, w, idx_eff)
            idx = idx_eff * step
            fr = {k: vr[k].get_batch(list(idx)).asnumpy() for k in VAR}
            for k, (var, plaus, pair) in VAR.items():
                od = f"{OUT}/{w}/{d}/{k}"; os.makedirs(od, exist_ok=True)
                for j in range(2 * N):
                    p = f"{od}/{j:06d}.png"
                    if not os.path.exists(p):
                        Image.fromarray(fr[k][j]).save(p, compress_level=1)
                rows[w].append(dict(video_id=f"{v['video']}_{var}", file="", file_name=f"{d}/{k}", block_id=str(bi), source_block=d,
                                    variant=var, plausible=plaus, pair_id=pair, condition=v["event"], motion=v["event"],
                                    has_occlusion="1", violation_type="permanence", game_name=v["scene"], role=var, sym_k="0",
                                    objset="real", violation="permanence", occlusion="late", k="0", background=v["scene"],
                                    obj_a="present", obj_b="absent", ctx_clean_samples="", samples=""))
            ctx_diff = {pair: int(max(np.abs(fr[a][:N].astype(int) - fr[b][:N].astype(int)).max() for _ in [0]))
                        for pair, (a, b) in (("A", ("possible_present", "impossible_vanish")), ("B", ("possible_absent", "impossible_appear")))}
            ctx_eq_frames = {pair: int(sum((fr[a][j] == fr[b][j]).all() for j in range(N)))
                             for pair, (a, b) in (("A", ("possible_present", "impossible_vanish")), ("B", ("possible_absent", "impossible_appear")))}
            audit.setdefault(d, {})[w] = dict(frames_eff=idx_eff.tolist(), stride_ctx=float(np.diff(ctx).mean()), stride_fut=float(np.diff(fut).mean()),
                                              t_eff=t, n_eff=n, reveal_eff=rev, last_visible_eff=last,
                                              fut_reveal_frames=int((fut >= rev).sum()), ctx_visible_frames=int((ctx <= last).sum()),
                                              ctx_maxdiff=ctx_diff, ctx_identical_frames=ctx_eq_frames, duplicated=dup)
        print(d, {w: (round(a["stride_ctx"], 1), round(a["stride_fut"], 1), a["fut_reveal_frames"], a["ctx_identical_frames"]) for w, a in audit[d].items()}, flush=True)
    for w, rr in rows.items():
        if a.no_index or not rr:
            continue
        od = f"{REPO}/data_csv/rubric_quads_{w}"; os.makedirs(od, exist_ok=True)
        with open(f"{od}/index.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rr[0].keys())); wr.writeheader(); wr.writerows(rr)
        json.dump({d: a[w] for d, a in audit.items()}, open(f"{od}/audit.json", "w"), indent=1)
    print("done", {w: len(r) for w, r in rows.items()})


if __name__ == "__main__":
    main()
