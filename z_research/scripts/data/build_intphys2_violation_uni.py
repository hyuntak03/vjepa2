#!/usr/bin/env python3
"""IntPhys2 Main — 위반 시작 프레임 라벨로 문맥 · 미래를 나눈 평가 세트 (2026-10-22, 사용자 지시).

  /venv/vjepa/bin/python z_research/scripts/data/build_intphys2_violation_uni.py <라벨 시트 JSON (get_values 결과)>

라벨 시트 (labels 탭): 불가능 영상 506 편 (scene<S>_pair<P>_impossible_<game>.mp4) 의 violation_start_frame.
사용자: "같은 시나리오면 pair 1 · 2 의 위반 시작 프레임이 사실상 같다" → 쌍 하나만 라벨이 있으면 다른 쌍에 같은 값을 쓴다 (둘 다 있으면 각자).
쌍 (장면, pair p) 마다 가능 영상 p_Possible 과 불가능 영상 p_Impossible 을 같은 32 장으로:
  문맥 16 장 = [0, d−1] 균일, 미래 16 장 = [d, N−1] 균일 (d = 위반 시작, N = 영상 장 수). intphys1_late_uni 와 같은 규칙.
프레임: <out>/<video_id>/000000..000031.png (256², bilinear). index: data_csv/intphys2_violation_uni/index.csv.
감사: 문맥 16 장이 가능 · 불가능 영상에서 같은지 (평균 |Δ| 픽셀) → context_integrity.json"""
import collections, csv, json, os, re, sys
import numpy as np, decord
from PIL import Image
LAB = sys.argv[1]
SHIFT = int(next((x.split("=")[1] for x in sys.argv if x.startswith("--shift=")), 0))   # 2026-10-22 사용자: 라벨 d 에 더할 값 (예 --shift=-10)
CSV_ONLY = "--csv-only" in sys.argv
# 2026-10-22 사용자가 프레임을 보고 직접 정한 위반 시작 (절대값, shift 안 받음): data_csv/intphys2_violation_uni/overrides.json {"<condition>|<scene>|<pair>": d}
OVR = json.load(open("data_csv/intphys2_violation_uni/overrides.json")) if os.path.exists("data_csv/intphys2_violation_uni/overrides.json") else {}   # 2026-10-22: PNG 를 풀지 않고 index_mp4.csv 만 (채점기가 mp4 에서 frames 컬럼 번호로 바로 읽는다)
ROOT = "/workspace/dataset/benchmarks/IntPhys2/Main"
OUT = "/workspace/world_model/local_datasets/world/world_analysis/IntPhys2_violation_uni"
CSV = "data_csv/intphys2_violation_uni"
v = json.load(open(LAB))["values"]; h = v[0]
lab = collections.defaultdict(dict)
for r in v[1:]:
    r = dict(zip(h, r + [""] * (len(h) - len(r))))
    m = re.search(r"scene(\d+(?:_\d+)?)_pair(\d)_impossible_", r["file"])
    if r["violation_start_frame"].strip():
        lab[(r["condition"], m[1])][int(m[2])] = int(r["violation_start_frame"]) + SHIFT
meta = list(csv.DictReader(open(os.path.join(ROOT, "metadata.csv"))))
vid = {(m["condition"], m["SceneIndex"], m["type"]): m for m in meta}
rows, integ, src = [], [], collections.Counter()
for (cond, sc), d in sorted(lab.items()):
    for p in (1, 2):
        dd = d.get(p, d.get(3 - p))
        dd = OVR.get(f"{cond}|{sc}|{p}", dd); src["자기 라벨" if p in d else "다른 쌍 라벨"] += 1
        pos, imp = vid[(cond, sc, f"{p}_Possible")], vid[(cond, sc, f"{p}_Impossible")]
        frs = {}
        for tag, m_ in (("pos", pos), ("imp", imp)):
            vr = decord.VideoReader(os.path.join(ROOT, m_["file_name"]))
            N = len(vr)
            ts = [int(round(x)) for x in np.linspace(0, dd - 1, 16)] + [int(round(x)) for x in np.linspace(dd, N - 1, 16)]
            if not CSV_ONLY:
                fr = vr.get_batch(ts).asnumpy()
                frs[tag] = fr
                vd = os.path.join(OUT, m_["name"]); os.makedirs(vd, exist_ok=True)
                for k, f in enumerate(fr):
                    Image.fromarray(f).resize((256, 256), Image.BILINEAR).save(f"{vd}/{k:06d}.png")
            rows.append(dict(video_id=m_["name"], file=m_["name"], block_id=f"{cond}|{sc}|{p}", pair_id=1, variant=tag, plausible=int(tag == "pos"),
                             block_type=cond, scene=sc, pair=p, game_name=m_["game_name"], difficulty=m_["Difficulty"], camera=m_["Camera"],
                             d=dd, n_frames=N, mp4=m_["file_name"], label_from="self" if p in d else "other_pair", frames=" ".join(map(str, ts))))
        if not CSV_ONLY: integ.append(dict(block=f"{cond}|{sc}|{p}", ctx_mean_abs_diff=float(np.abs(frs["pos"][:16].astype(int) - frs["imp"][:16].astype(int)).mean())))
    print(f"{cond} {sc} d={d}", flush=True)
os.makedirs(CSV, exist_ok=True)
if not CSV_ONLY:
    with open(f"{CSV}/index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
with open(f"{CSV}/index_mp4.csv", "w", newline="") as f:    # file = mp4 (ROOT 기준), frames = 원본 프레임 번호 32 개
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows([dict(r, file=r["mp4"]) for r in rows])
if CSV_ONLY:
    print(f"쌍 {len(rows) // 2} · 영상 {len(rows)} · 라벨 출처 {dict(src)} (csv 만)"); sys.exit(0)
json.dump(integ, open(f"{CSV}/context_integrity.json", "w"), indent=0)
ia = np.array([x["ctx_mean_abs_diff"] for x in integ])
print(f"쌍 {len(integ)} · 영상 {len(rows)} · 라벨 출처 {dict(src)} · 조건 {collections.Counter(r['block_type'] for r in rows if r['variant']=='pos')}")
print(f"문맥 평균 |Δ| 픽셀: 0 인 쌍 {int((ia == 0).sum())} · 중앙 {np.median(ia):.3f} · 최대 {ia.max():.2f}")
