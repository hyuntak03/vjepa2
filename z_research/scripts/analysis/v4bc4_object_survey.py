#!/usr/bin/env python3
"""v4bc4 (v4b + v4c) 영상마다 물체 구성 · 움직임 · 보임을 센다 (2026-11-08, 사용자 "항상 가려지는 물체는 없는데 항상 보이는 물체는 있는거지 찾아봐").

입력: data_csv/blender_track_v4bc4_negimp/families.json (묶음: A = 물체가 가림막을 지나는 단계 영상들, B = 빈 장면) +
      영상마다 생성기 design/<id>.json (물체 역할 · 움직임) · labels/<id>.npz (물체 × 프레임 visible_n / amodal_n 픽셀 수, 64×64 격자).
물체 상태 (64 프레임 중):
  항상 보임   = 모든 프레임 visible_n > 0
  항상 가려짐 = 모든 프레임 visible_n == 0 (amodal > 0, 화면 안에 있는데 안 보임)
  일부 가려짐 = 그 사이 (가려진 프레임 수도 센다)
  /venv/vjepa/bin/python z_research/scripts/analysis/v4bc4_object_survey.py
"""
import collections, json, os
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
F = json.load(open(f"{REPO}/data_csv/blender_track_v4bc4_negimp/families.json"))
paths = [l.strip() for l in open(F["list"]) if l.strip()]
arm = {}
for fam in F["families"]:
    for i in fam["A"]:
        arm[i] = "A (물체가 가림막을 지남)"
    if fam.get("B") is not None:
        arm[fam["B"]] = "B (빈 장면)"
cnt = collections.Counter()
objs = collections.Counter()
hidden_frames = collections.defaultdict(list)
moving_distr = collections.Counter()
examples = collections.defaultdict(list)
for i, p in enumerate(paths):
    root, name = p.rsplit("/clips/", 1)
    cid = name[:-4]
    src = "v4b" if "v4b_" in root else "v4c"
    a = arm.get(i, "묶음 밖")
    d = json.load(open(f"{root}/design/{cid}.json"))
    z = np.load(f"{root}/labels/{cid}.npz", allow_pickle=True)
    vis, am = z["visible_n"], z["amodal_n"]                      # (T, n_obj)
    roles = [str(r) for r in z["roles"]]
    cnt[(src, a, len(roles), roles.count("target"))] += 1
    for j, (o, r) in enumerate(zip(d["objects"], roles)):
        mk = o.get("motion", {}).get("kind", "?")
        inview = am[:, j] > 0
        v = vis[:, j] > 0
        if not inview.any():
            st = "화면 밖"
        elif v[inview].all():
            st = "항상 보임"
        elif not v[inview].any():
            st = "항상 가려짐"
        else:
            st = "일부 가려짐"
            hidden_frames[(src, a, r)].append(int((inview & ~v).sum()))
        objs[(src, a, r, "움직임" if "static" not in mk else "정지", mk, st)] += 1
        if r != "target" and "static" not in mk:
            moving_distr[(src, a, mk, st)] += 1
        if st == "항상 가려짐" and len(examples[(src, a, r)]) < 3:
            examples[(src, a, r)].append(cid)
print("== 영상 수: (출처, 묶음 역할, 물체 수, target 수)")
for k, v in sorted(cnt.items()):
    print(f"  {k}: {v}")
print("\n== 물체: (출처, 묶음 역할, 물체 역할, 움직임?, 움직임 종류, 보임 상태): 수")
for k, v in sorted(objs.items()):
    print(f"  {k}: {v}")
print("\n== 일부 가려진 물체의 가려진 프레임 수 (64 중) 중앙값 · 최대")
for k, v in sorted(hidden_frames.items()):
    print(f"  {k}: n {len(v)} · 중앙 {int(np.median(v))} · 최대 {max(v)}")
print("\n== 움직이는 distractor:", dict(moving_distr) or "없음")
print("\n== 항상 가려진 물체 예:", dict(examples) or "없음")
