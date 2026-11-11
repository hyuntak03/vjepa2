"""real world permanence 장면 분류 (가림 방식 · 카메라) 와 분류별 시트 점수 · 장면별 정오 (2026-11-07, 사용자 "어디서 틀리는지 · 가림막 종류 / 카메라 무빙 구분").
분류 근거: index.json 메타 (PerceptionTest 가림막 이름 · PhysicsIQ 설명 "Static shot" · MAGICONTROL 메모 · RubricOffice 장면 이름) + 장면별 대표 프레임 눈 확인
  + 카메라 이동 측정 (real_bench_camera_motion.py, 대상을 지운 영상의 화면 전체 이동). 판정 규칙은 아래 OCC · HANDHELD 에 그대로 적었다.
출력: z_analysis/results/real_errors_20261107/{scenes.csv, SUMMARY.md}
  /venv/vjepa/bin/python z_research/scripts/analysis/real_bench_categories.py
"""
import csv, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from scene_paired_tests import boards, scene_stats, summarize, RES

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
SRC = "/workspace/dataset/benchmarks/real_world_permanence"
OUT = os.path.join(ROOT, "z_analysis/results/real_errors_20261107")
MODELS = {"vith": "V-JEPA 2", "vast_ar_vw_state": "AR 기본판", "vast_ar_vw_state_v4bc4": "AR → v4bc4 L1", "vast_ar_vw_state_v4bc4_negimp": "AR → v4bc4 음의 손실"}

# 가림 방식 (눈 확인 · 메타): 고정 가림막 뒤로 지나감 / 움직이는 판이 앞을 가림 / 위에서 덮개 / 용기 안 / 손 안 / 화면 밖
RUBRIC_OCC = {"10_confuse": "고정 가림막 뒤", "2_teaduck": "움직이는 판·상자", "3_mugduck": "용기 안", "4_home/chips1_oof": "화면 밖",
              "4_home/pumpcookie1_reveal": "고정 가림막 뒤", "5_bagmugduck": "용기 안", "6_handball": "손 안", "7_ballbounce": "고정 가림막 뒤",
              "8_plantcupball": "용기 안"}
# 손에 든 카메라 (흔들림 · 이동): RubricOffice 중 측정 이동 중앙값 ≥ 1 px 이거나 눈으로 시점이 바뀌는 장면. 나머지 출처는 삼각대 (PhysicsIQ "Static shot", MAGICONTROL · HOOT 눈 확인).
HANDHELD = {"10_confuse", "2_teaduck", "3_mugduck", "4_home", "5_bagmugduck"}


def occ_of(v):
    s, m = v["source"], v["meta"]
    if s == "PerceptionTest":
        return "움직이는 판·상자"                       # 손에 세워 든 책 · 판이 앞을 가렸다 치움
    if s == "PhysicsIQ":
        return {"roll-behind-box": "고정 가림막 뒤", "solid-ball-peakaboo": "위에서 덮개", "ball-behind-rotating-paper": "움직이는 판·상자"}[m["scenario"]]
    if s == "HOOT":
        return "고정 가림막 뒤"
    if s == "MAGICONTROL":
        return "위에서 덮개" if "cup" in m["material"] else "손 안"
    key = v["id"].split("/", 1)[1]
    for k, o in RUBRIC_OCC.items():
        if key.startswith(k):
            return o
    raise KeyError(key)


def cam_of(v):
    if v["source"] == "RubricOffice" and v["id"].split("/")[1] in HANDHELD:
        return "손에 든 카메라"
    return "고정"


def main():
    J = {v["id"]: v for v in json.load(open(f"{SRC}/index.json"))["videos"]}
    cam_m = {r["scene"]: r for r in csv.DictReader(open(f"{OUT}/camera_motion.csv"))}
    rows = list(csv.DictReader(open(f"{ROOT}/data_csv/real_world_permanence_uniform/index.csv")))
    held = {r["video_id"] for r in csv.DictReader(open(f"{ROOT}/data_csv/real_world_permanence_uniform/index_test.csv"))}
    sc = {}
    for r in rows:
        sc.setdefault(r["block_id"], {"scene": r["source_block"], "condition": r["condition"], "source": r["source"], "hidden_s": float(r["hidden_s"]),
                                      "held": r["video_id"] in held})
    pat, blk = boards()["real world (all 95)"]
    S = {m: {k: (v["all"] if isinstance(v, dict) else v) for k, v in json.load(open(f"{RES}/{pat.format(m=m)}/per_block.json"))["per_video_surprise"].items()} for m in MODELS}
    out = []
    for b, info in sorted(sc.items(), key=lambda x: int(x[0])):
        v = J[info["scene"]]
        row = {"block": b, **info, "occluder": occ_of(v), "camera": cam_of(v), "cam_shift_med": cam_m[info["scene"]]["shift_med"],
               "cam_net": cam_m[info["scene"]]["net"], "gif": f"{SRC}/gifs/{info['scene']}"}
        for m in MODELS:
            r = scene_stats(S[m], {b: blk[b]})[0]
            row[f"{m}:A"] = int(r[1] > 0); row[f"{m}:B"] = int(r[2] > 0)
        out.append(row)
    with open(f"{OUT}/scenes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)

    def table(key, order=None):
        cats = order or sorted({o[key] for o in out})
        lines = ["| 모델 | " + " | ".join(f"{c} ({sum(1 for o in out if o[key] == c)})" for c in cats) + " |", "|" + "---|" * (len(cats) + 1)]
        for m, nm in MODELS.items():
            cells = []
            for c in cats:
                bs = [o["block"] for o in out if o[key] == c]
                s = summarize(scene_stats(S[m], {x: blk[x] for x in bs}))
                cells.append(f"{(s['A'] + s['B']) / 2:.0f} ({s['A']:.0f}/{s['B']:.0f})")
            lines.append(f"| {nm} | " + " | ".join(cells) + " |")
        return "\n".join(lines)
    occ_order = ["고정 가림막 뒤", "움직이는 판·상자", "위에서 덮개", "용기 안", "손 안", "화면 밖"]
    md = ["# real world permanence 분류별 시트 점수 (A 사라짐 / B 나타남)", "", "## 가림 방식", table("occluder", occ_order), "",
          "## 카메라", table("camera", ["고정", "손에 든 카메라"]), "",
          "## 출처 × 가림 방식 (장면 수)",
          "| 출처 | " + " | ".join(occ_order) + " |", "|" + "---|" * (len(occ_order) + 1)]
    for s in ["RubricOffice", "MAGICONTROL", "PerceptionTest", "HOOT", "PhysicsIQ"]:
        md.append(f"| {s} | " + " | ".join(str(sum(1 for o in out if o['source'] == s and o['occluder'] == c)) for c in occ_order) + " |")
    open(f"{OUT}/SUMMARY.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
