#!/usr/bin/env python3
"""시트 Object_permanence 탭 칸 (사라짐 A / 나타남 B × 물체 움직임 / 정지) — 기존 산출물에서 짝 정답률을 다시 센다 (GPU 불필요, 2026-11-02).
벤치: VOBench late (split2 test late, permanence) · VO_moving_occluder (permanence) · IntPhys 1 late occ (O1 가림 30 쌍) · Rubric · (IntPhys 2 permanence 전체만, A/B 라벨 없음).
모델: release · VOBench PFT (vobftg) · v4vc 세 팔 = predictor_lab/results/v3_ablation/te/ (table_v3.panel_scores 그대로) /
      AR scratch (vast AR-state ep70) = 표준 하네스 per_block.json 의 per_video_surprise 를 같은 짝에 매핑. release · vobftg 도 표준 하네스로 대조 (*_harness).
판정: s(불가능) > s(가능) 정답, 같으면 0.5. 물체 정지 = motion flat_static. IntPhys 1 A = 가능 쪽 event_kind != not_seen_before_d (문맥에서 물체를 봤다).
시트에 넣은 행: Chance 50 · V-JEPA 2 = release · V4 = v4vc_all_s0 (평면 + 빗면 4k) · VOBench PFT = vobftg · AR scratch = AR.
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python; $P z_research/scripts/analysis/object_permanence_cells.py [--out /tmp/op_cells.json]
"""
import csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
sys.path.insert(0, str(REPO / "predictor_lab/v3_ablation"))
import table_v3 as T  # noqa

def j(si, sp):
    return 1.0 if si > sp else (0.5 if si == sp else 0.0)

# 칸 정의: (패널, 칸 이름, f(r)) ; r = 가능 쪽 index 행
mv = lambda r: r["motion"] != "flat_static"
st = lambda r: r["motion"] == "flat_static"
perm = lambda r: r["violation_type"] == "permanence"
ip1A = lambda r: r["event_kind"] != "not_seen_before_d"   # 문맥에서 물체를 봤다 → 불가능 = 사라짐
CELLS = []
for obj in ("primitive", "realistic"):
    for mname, mf in (("moving", mv), ("static", st)):
        for ab in ("A", "B"):
            CELLS.append(("vob_late", f"{obj}/{mname}/{ab}",
                          (lambda o, mf_, ab_: lambda r: perm(r) and r["objset"] == o and mf_(r) and r["pair_id"] == ab_)(obj, mf, ab)))
for mname, mf in (("moving", mv), ("static", st)):
    for ab in ("A", "B"):
        CELLS.append(("vomo", f"{mname}/{ab}", (lambda mf_, ab_: lambda r: perm(r) and mf_(r) and r["pair_id"] == ab_)(mf, ab)))
for mname in ("moving", "static"):
    for ab in ("A", "B"):
        CELLS.append(("ip1uni", f"{mname}/{ab}",
                      (lambda m_, ab_: lambda r: r["block_type"] == "O1" and r["group"] == f"{m_}/occluded" and (ip1A(r) if ab_ == "A" else not ip1A(r)))(mname, ab)))
CELLS.append(("ip2uni", "all(perm)", lambda r: r["condition"] == "permanence"))
for ab in ("A", "B"):
    CELLS.append(("rubric", ab, (lambda ab_: lambda r: r["pair_id"] == ab_)(ab)))
# overall = 위 칸들의 짝을 모두 합친 정답률 (칸 평균이 아니라 짝 가중)
for obj in ("primitive", "realistic"):
    CELLS.append(("vob_late", f"{obj}/overall", (lambda o: lambda r: perm(r) and r["objset"] == o)(obj)))
CELLS.append(("vomo", "overall", perm))
# 11-02 사용자: VO_moving_occluder 도 primitive / realistic 을 나눈다 (VOBench late 와 같은 칸)
for obj in ("primitive", "realistic"):
    for mname, mf in (("moving", mv), ("static", st)):
        for ab in ("A", "B"):
            CELLS.append(("vomo", f"{obj}/{mname}/{ab}",
                          (lambda o, mf_, ab_: lambda r: perm(r) and r["objset"] == o and mf_(r) and r["pair_id"] == ab_)(obj, mf, ab)))
    CELLS.append(("vomo", f"{obj}/overall", (lambda o: lambda r: perm(r) and r["objset"] == o)(obj)))
CELLS.append(("ip1uni", "overall", lambda r: r["block_type"] == "O1" and r["group"].endswith("/occluded")))
CELLS.append(("rubric", "overall", lambda r: True))
# VOBench_v2 late permanence (2026-11-02, te 패널 vob2late; 짝 432 = objset 2 × (움직임 90 + 정지 18) × A/B). replica 0 = v1.
# VOBench PFT (vobftg) 는 v1 split v2 train 의 가능 영상으로 학습 → v2 late perm 216 block 중 36 block 을 봤다 (replica 0 의 train 절반).
#   그래서 vobftg 는 "학습 block 제외 (180)" · "replica 1–2 만 (144, 처음 보는 물체 · 같은 경로)" 를 같이 낸다.
V1_TRAIN = {r["source_block"] for r in csv.DictReader(open(REPO / "data_csv/vobench_v1_split_v2/index_train.csv"))}
for obj in ("primitive", "realistic"):
    for mname, mf in (("moving", mv), ("static", st)):
        for ab in ("A", "B"):
            CELLS.append(("vob2late", f"{obj}/{mname}/{ab}",
                          (lambda o, mf_, ab_: lambda r: r["objset"] == o and mf_(r) and r["pair_id"] == ab_)(obj, mf, ab)))
    CELLS.append(("vob2late", f"{obj}/overall", (lambda o: lambda r: r["objset"] == o)(obj)))
CELLS.append(("vob2late", "overall", lambda r: True))
for sub, g in (("no_vobftg_train", lambda r: r["source_block"] not in V1_TRAIN), ("replica12", lambda r: r["replica"] in ("1", "2")),
               ("replica0", lambda r: r["replica"] == "0")):
    CELLS.append(("vob2late", f"{sub}/overall", g))
    for obj in ("primitive", "realistic"):
        for mname, mf in (("moving", mv), ("static", st)):
            for ab in ("A", "B"):
                CELLS.append(("vob2late", f"{sub}/{obj}/{mname}/{ab}",
                              (lambda g_, o, mf_, ab_: lambda r: g_(r) and r["objset"] == o and mf_(r) and r["pair_id"] == ab_)(g, obj, mf, ab)))
        CELLS.append(("vob2late", f"{sub}/{obj}/overall", (lambda g_, o: lambda r: g_(r) and r["objset"] == o)(g, obj)))
# Real Bench = real_world_permanence (2026-11-02, te 패널 rwp; 짝 A = 사라짐 · B = 나타남). 출처별 칸은 보고용 (RubricOffice 는 옛 rubric 패널과 대조)
for ab in ("A", "B"):
    CELLS.append(("rwp", ab, (lambda ab_: lambda r: r["pair_id"] == ab_)(ab)))
CELLS.append(("rwp", "overall", lambda r: True))
for src in ("RubricOffice", "MAGICONTROL", "PerceptionTest", "HOOT", "PhysicsIQ"):
    for ab in ("A", "B"):
        CELLS.append(("rwp", f"{src}/{ab}", (lambda s_, ab_: lambda r: r["source"] == s_ and r["pair_id"] == ab_)(src, ab)))

TE_TAGS = ["release", "vobftg", "v4vc_fam_s0", "v4vc_fam6k_s0", "v4vc_all_s0", "vast_ar_state_ep70"]   # 마지막 = AR (te 안에서 kind=ar, rwp 만)

# 표준 하네스 per_video_surprise (AR + 대조)
HARN = {
    "AR": {"vob_late": "z_research/VOBench_v1/exp_results/surprise_c16t32__vobench_v1_vith_vast_ar_state_ep70",
           "vomo": "cvpr/results/eval/testbed/vo_moving_occluder_vith_vast_ar_state_ep70",
           "ip1uni": "cvpr/results/eval/intphys1_late/intphys1_late_uni_vith_vast_ar_state_ep70",
           "rubric": "z_analysis/release/04_rubric_office/harness/uniform_vast70",
           "rwp": "predictor_lab/results/v3_ablation/harness/rwp_uniform_vast70",
           "vob2late": "predictor_lab/results/v3_ablation/harness/vob2late_vast70"},
    "release_harness": {"vob_late": "z_research/VOBench_v1/exp_results/surprise_c16t32__vobench_v1_vith",
                        "vomo": "cvpr/results/eval/testbed/vo_moving_occluder_vith",
                        "ip1uni": "cvpr/results/eval/intphys1_late/intphys1_late_uni_vith"},
    "vobftg_harness": {"vob_late": "z_training/runs/vobench_release_postft_g/eval/surprise_c16t32__vobench_v1_split2_test_latest",
                       "vomo": "cvpr/results/eval/testbed/vo_moving_occluder_vith_release_vobftg",
                       "ip1uni": "cvpr/results/eval/intphys1_late/intphys1_late_uni_vith_release_vobftg"},
}

out = {}
cache = {}
for pn, cname, f in CELLS:
    if pn not in cache:
        if not any((d / "done.npy").exists() for d in T.TE.glob(f"{pn}_*")):
            print(f"  (결과 없음, 건너뜀) {pn}"); cache[pn] = None
        else:
            S, P = T.panel_scores(pn)
            m = json.load(open(sorted(T.TE.glob(f"{pn}_*"))[0] / "meta.json"))
            cache[pn] = (S, P, list(m["video_id"]))
    if cache[pn] is None:
        continue
    S, P, vids = cache[pn]
    sel = [(a, b) for a, b, r in P if f(r)]
    row = {"n": len(sel)}
    for t in TE_TAGS:
        if t in S:
            row[t] = round(100 * np.mean([j(S[t][b], S[t][a]) for a, b in sel]), 1) if sel else None
    for hn, dd in HARN.items():
        if pn in dd and (REPO / dd[pn] / "per_block.json").exists():
            pv = json.load(open(REPO / dd[pn] / "per_block.json"))["per_video_surprise"]
            miss = [k for a, b in sel for k in (vids[a], vids[b]) if k not in pv]
            if miss:
                row[hn] = f"missing {len(miss)}"
            else:
                row[hn] = round(100 * np.mean([j(pv[vids[b]], pv[vids[a]]) for a, b in sel]), 1) if sel else None
    out[f"{pn}/{cname}"] = row
    print(f"{pn}/{cname:24s}", row)

# IntPhys 2 = late **final** 판 (사용자 재라벨, 500 쌍; 시트 Overall 탭의 "IntPhys2 late occ" 와 같은 판 — release 52.4 · VOBench PFT 55.2 · AR 52.0 이 전체 조건 값).
# 사라짐 / 나타남 라벨이 없어서 permanence 짝 (120) 을 카메라 움직임 (68) / 고정 (52) 으로 나눈다. 모두 표준 하네스 결과 (per_block.json).
import yaml  # noqa: E402
IP2 = {"release": "intphys2_late_final_vith", "vobftg": "intphys2_late_final_vith_release_vobftg",
       "v4vc_all_s0": "intphys2_late_final_vith_v4vc_all_s0", "AR": "intphys2_late_final_vith_vast_ar_state_ep70"}
for cname, f in (("Moving", lambda x: x["Camera"] == "Moving"), ("Fixed", lambda x: x["Camera"] == "Fixed"), ("overall", lambda x: True)):
    row = {}
    for tag, d in IP2.items():
        pdir = REPO / "cvpr/results/eval/intphys2_late" / d
        if not (pdir / "per_block.json").exists():
            continue
        r = yaml.safe_load(open(pdir / "_resolved.yaml"))["data"]
        idx = list(csv.DictReader(open(Path(r["root"]) / r["index_csv"])))
        pv = json.load(open(pdir / "per_block.json"))["per_video_surprise"]
        by = defaultdict(dict)
        for x in idx:
            if x["condition"] == "permanence" and f(x):
                by[(x["block_id"], x["pair_id"])]["pos" if x["plausible"] == "1" else "imp"] = x["video_id"]
        acc = [j(pv[v["imp"]], pv[v["pos"]]) for v in by.values()]
        row["n"] = len(acc); row[tag] = round(100 * float(np.mean(acc)), 1)
    out[f"ip2final/{cname}"] = row
    print(f"ip2final/{cname:24s}", row)
if "--out" in sys.argv:
    json.dump(out, open(sys.argv[sys.argv.index("--out") + 1], "w"), indent=1)
