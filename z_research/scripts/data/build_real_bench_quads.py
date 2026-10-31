#!/usr/bin/env python3
"""real_bench / rubric_office_quads → 32 장 PNG + matched-pair index (2026-10-31, 사용자: "violation point 기준으로 이전 16 frame uniform 이후 16 frame uniform").

원본: /workspace/dataset/benchmarks/real_bench/rubric_office_quads (TCOW Rubric Office, 실제 영상 20 개 × 4 = 80 clip, README 참고).
  possible_present / impossible_vanish / impossible_appear / possible_absent. t_splice = 대상이 완전히 숨은 구간의 가운데 = 위반 지점
  (impossible 영상은 t_splice 부터 다른 소스로 바뀐다).
뽑는 법: 문맥 = [0, t_splice) 에서 균등 16 장, 미래 = [t_splice, 끝) 에서 균등 16 장 (np.linspace 반올림) → 32 장, 000000–000031.png
  · 60 fps 로 저장된 중복 프레임 영상 (lightball 3 개, 모든 프레임이 두 번씩) 은 짝수 프레임 (실제 30 fps) 위에서 뽑는다
  · ⚠️ 문맥과 미래의 시간 간격이 다르다 (영상마다 index 의 ctx_step / fut_step, 원본 장 단위). 채점기는 32 장을 등간격으로 본다
  · 해상도는 원본 그대로 (640x480). 채점기가 256x256 으로 늘이거나 줄인다 (4:3 → 1:1, 자르지 않음 — 가운데 정사각으로 자르면 물체가
    최대 29 % 잘린다 (mugduck6))
matched pair (문맥이 같은 것끼리, block = 영상):
  pair A = possible_present (pos_a) vs impossible_vanish (imp_ab) — 문맥 = 원본
  pair B = possible_absent (pos_b) vs impossible_appear (imp_ba) — 문맥 = 대상을 지운 영상 E
  impossible 영상의 문맥 16 장은 짝 possible 영상의 PNG 를 그대로 쓴다 — 원본은 같은 프레임이지만 인코딩이 따로라 lightball 3 개는
  디코드 값이 다르다 (최대 차 12–27, 나머지 17 개는 0). 그래서 문맥이 픽셀 단위로 같아진다 (matched pairing 의 전제).
출력:
  프레임  ${BENCH}/real_bench/rubric_office_quads_c16f16/<scene>/<video>/<variant>/{000000..000031}.png
  index   data_csv/real_bench_quads/index.csv  (+ context_integrity.json: 짝마다 문맥 PNG 바이트 동일 검사)

  /venv/vjepa/bin/python z_research/scripts/data/build_real_bench_quads.py
"""
import csv, hashlib, json, os, sys
import numpy as np
from PIL import Image

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
SRC = "/workspace/dataset/benchmarks/real_bench/rubric_office_quads"
DST = "/workspace/dataset/benchmarks/real_bench/rubric_office_quads_c16f16"
OUT = os.path.join(REPO, "data_csv/real_bench_quads")
NC = NF = 16
VAR = {"possible_present": ("pos_a", "A", 1), "impossible_vanish": ("imp_ab", "A", 0),
       "possible_absent": ("pos_b", "B", 1), "impossible_appear": ("imp_ba", "B", 0)}
CTX_SRC = {"impossible_vanish": "possible_present", "impossible_appear": "possible_absent"}   # 문맥 PNG 를 가져올 짝
# README "현재 상태" (inpainting 품질) — 따로 보라고 권한 영상
QUALITY = {"mugduck5_stay": "weak", "bagduck1_move": "weak", "mugduck2_reveal": "ok", "chips1_oof": "ok", "handball2_boxoccl": "ok"}


def sample(v):
    """원본 장 번호 (문맥 16, 미래 16)."""
    T, ts = v["num_frames"], v["t_splice"]
    if v["duplicated_frames"]:                                  # 짝수 장 = 실제 30 fps
        T2, ts2 = T // 2, ts // 2
        c = np.round(np.linspace(0, ts2 - 1, NC)).astype(int) * 2
        f = np.round(np.linspace(ts2, T2 - 1, NF)).astype(int) * 2
    else:
        c = np.round(np.linspace(0, ts - 1, NC)).astype(int)
        f = np.round(np.linspace(ts, T - 1, NF)).astype(int)
    assert c[-1] < ts <= f[0] and len(set(c)) == NC and len(set(f)) == NF, (v["dir"], c, f)
    return c, f


def main():
    import decord
    meta = json.load(open(os.path.join(SRC, "index.json")))
    os.makedirs(OUT, exist_ok=True)
    rows, integ = [], []
    for bi, v in enumerate(meta["videos"]):
        c, f = sample(v)
        idx = list(c) + list(f)
        for name in ("possible_present", "possible_absent", "impossible_vanish", "impossible_appear"):   # possible 먼저 (문맥을 넘겨준다)
            d = os.path.join(DST, v["dir"], name); os.makedirs(d, exist_ok=True)
            fr = decord.VideoReader(os.path.join(SRC, v["dir"], f"{name}.mp4")).get_batch(idx).asnumpy()
            for j in range(NC + NF):
                p = os.path.join(d, f"{j:06d}.png")
                if j < NC and name in CTX_SRC:
                    src = os.path.join(DST, v["dir"], CTX_SRC[name], f"{j:06d}.png")
                    with open(src, "rb") as a, open(p, "wb") as b: b.write(a.read())
                else:
                    Image.fromarray(fr[j]).save(p)
            vid, pair, pl = VAR[name]
            rows.append(dict(video_id=f"{v['video']}_{vid}", file="", file_name=f"{v['dir']}/{name}", block_id=bi, source_block=v["video"],
                             variant=vid, plausible=pl, pair_id=pair, condition=v["event"], scene=v["scene"], video=v["video"], quad_variant=name,
                             t_splice=v["t_splice"], num_frames=v["num_frames"], duplicated_frames=int(v["duplicated_frames"]),
                             ctx_frames=" ".join(map(str, c)), fut_frames=" ".join(map(str, f)),
                             ctx_step=round(float(np.diff(c).mean()), 2), fut_step=round(float(np.diff(f).mean()), 2),
                             quality=QUALITY.get(v["video"], "clean")))
        for a_, b_ in (("possible_present", "impossible_vanish"), ("possible_absent", "impossible_appear")):
            h = lambda n: [hashlib.md5(open(os.path.join(DST, v["dir"], n, f"{j:06d}.png"), "rb").read()).hexdigest() for j in range(NC)]
            integ.append(dict(video=v["dir"], pair=f"{a_}|{b_}", context_identical=h(a_) == h(b_)))
        print(f"{v['dir']:34s} t_splice {v['t_splice']:3d} · 문맥 장 {c[0]}–{c[-1]} (간격 {np.diff(c).mean():.1f}) · 미래 {f[0]}–{f[-1]} (간격 {np.diff(f).mean():.1f})", flush=True)
    with open(os.path.join(OUT, "index.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(dict(n_pairs=len(integ), mismatch=sum(not x["context_identical"] for x in integ), pairs=integ),
              open(os.path.join(OUT, "context_integrity.json"), "w"), indent=1)
    print(f"\n{len(rows)} clip · {len(integ)} matched pair · 문맥 불일치 {sum(not x['context_identical'] for x in integ)} → {OUT}")


if __name__ == "__main__":
    main()
