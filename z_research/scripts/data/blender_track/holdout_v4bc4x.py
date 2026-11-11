#!/usr/bin/env python3
"""v4bc4x 에서 아주 작은 시험셋을 묶음 단위로 떼고, 학습 목록에서는 뺀다 (2026-11-11, 사용자 "v4bc4 test 진짜 조그맣게 하나 만들어봐").

묶음 (family) = 같은 장면의 물체 영상 A (속도 단계별) + 빈 영상 B. 목록에서 v4c 묶음은 × 4 로 반복돼 있다 → 같은 묶음의 모든 복사본을 뺀다.
시험 묶음마다 hs (이음 순간 물체가 가려져 있음) 인 A 하나를 고르고 네 팔을 만든다 (VOBench 와 같은 짝):
  pos_a = A · pos_b = B · imp_ab = A 앞 16 샘플 + B 뒤 16 샘플 (사라짐) · imp_ba = B 앞 + A 뒤 (나타남). 짝 A = (pos_a, imp_ab), 짝 B = (pos_b, imp_ba).
창 = 원본 64 장 중 0, 2, …, 62 (학습과 같은 stride 2, 시작 0) → 32 장 PNG. 이음 = 샘플 16 (원본 32, 학습 families.json 의 splice 와 같다).
불가능 팔은 가능 팔 PNG 의 hardlink 라 문맥 픽셀이 비트 단위로 같다 (matched pairing 전제).

  $BT_PY holdout_v4bc4x.py --src-name blender_track_v4bc4x --train-name blender_track_v4bc4x_train \
      --test-root /workspace/dataset/benchmarks/v4bc4x_test --test-name v4bc4x_test --n-v4b 140 --n-v4c 60
출력: data_csv/<train-name>/arms/all_mp4list.csv · data_csv/<train-name>_negimp/families.json (학습) ·
      <test-root>/frames/<video_id>/NN.png · data_csv/<test-name>/index.csv · test_families.json
"""
import argparse, collections, csv, json, os
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))


def key(f):
    return (f.get("src"), f.get("plane"), int(f["family"]), bool(f.get("static", False)), f.get("occ"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-name", required=True)
    ap.add_argument("--train-name", required=True)
    ap.add_argument("--test-root", required=True)
    ap.add_argument("--test-name", required=True)
    ap.add_argument("--n-v4b", type=int, default=140)
    ap.add_argument("--n-v4c", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--src-tags", default="v4bx,v4cx", help="families.json 의 src 이름 (v4b 쪽, v4c 쪽)")
    a = ap.parse_args()
    import decord
    from PIL import Image
    fj = json.load(open(f"{REPO}/data_csv/{a.src_name}_negimp/families.json"))
    L = [l for l in open(fj["list"]).read().split("\n") if l]
    fams = fj["families"]
    tb, tc = a.src_tags.split(",")
    uniq = collections.OrderedDict()
    for f in fams:
        uniq.setdefault(key(f), f)
    rng = np.random.default_rng(a.seed)
    pick = []
    for tag, n in ((tb, a.n_v4b), (tc, a.n_v4c)):
        cand = [k for k, f in uniq.items() if f.get("src") == tag and any(f["hs"])]
        sel = rng.choice(len(cand), size=min(n, len(cand)), replace=False)
        pick += [cand[i] for i in sorted(sel)]
    test = set(pick)
    # 학습 목록: 시험 묶음의 모든 복사본을 뺀다
    new_paths, new_fams = [], []
    for f in fams:
        if key(f) in test:
            continue
        g = dict(f)
        g["A"] = []
        for i in f["A"]:
            g["A"].append(len(new_paths)); new_paths.append(L[i])
        g["B"] = len(new_paths); new_paths.append(L[f["B"]])
        new_fams.append(g)
    assert sum(1 + len(f["A"]) for f in new_fams) == len(new_paths)
    test_paths = {L[i] for k in test for i in uniq[k]["A"] + [uniq[k]["B"]]}
    assert not (test_paths & set(new_paths)), "시험 영상이 학습 목록에 남았다"
    d1, d2 = f"{REPO}/data_csv/{a.train_name}/arms", f"{REPO}/data_csv/{a.train_name}_negimp"
    os.makedirs(d1, exist_ok=True); os.makedirs(d2, exist_ok=True)
    open(f"{d1}/all_mp4list.csv", "w").writelines(p + "\n" for p in new_paths)
    s = {"name": a.train_name, "from": a.src_name, "videos": len(new_paths), "families": len(new_fams),
         "held_out_families": len(test), "held_out_videos": len(test_paths)}
    json.dump(s, open(f"{d1}/all_summary.json", "w"), indent=1)
    json.dump({"list": os.path.abspath(f"{d1}/all_mp4list.csv"), "splice": fj["splice"], "families": new_fams, "summary": s}, open(f"{d2}/families.json", "w"))
    print("학습", json.dumps(s, ensure_ascii=False))
    # 시험셋: PNG 32 장 (원본 0, 2, …, 62), 불가능 팔은 hardlink
    fr = f"{a.test_root}/frames"; os.makedirs(fr, exist_ok=True)
    sp = int(fj["splice"]) // 2                                   # 이음 샘플 (원본 32 → 샘플 16)
    rows, tf = [], []
    for b, k in enumerate(pick):
        f = uniq[k]
        cands = [i for i, h in zip(f["A"], f["hs"]) if h]
        ia = int(cands[int(rng.integers(0, len(cands)))]); ib = int(f["B"])
        base = f"t{b:04d}"
        src_dirs = {}
        for tag, idx in (("A", ia), ("B", ib)):
            vr = decord.VideoReader(L[idx], num_threads=1)
            arr = vr.get_batch(list(range(0, 64, 2))).asnumpy()
            dd = f"{fr}/{base}_{tag}"; os.makedirs(dd, exist_ok=True)
            for t in range(32):
                Image.fromarray(arr[t]).save(f"{dd}/{t:02d}.png")
            src_dirs[tag] = dd
        arms = {"pos_a": ("A", "A"), "pos_b": ("B", "B"), "imp_ab": ("A", "B"), "imp_ba": ("B", "A")}
        for v, (c0, c1) in arms.items():
            vid = f"{base}_{v}"; dd = f"{fr}/{vid}"; os.makedirs(dd, exist_ok=True)
            for t in range(32):
                s0 = f"{src_dirs[c0 if t < sp else c1]}/{t:02d}.png"
                dst = f"{dd}/{t:02d}.png"
                if os.path.exists(dst):
                    os.remove(dst)
                os.link(s0, dst)
            rows.append({"video_id": vid, "file": "", "file_name": vid, "block_id": b, "variant": v,
                         "plausible": 1 if v.startswith("pos") else 0, "pair_id": "A" if v in ("pos_a", "imp_ab") else "B",
                         "condition": f"{f.get('src')}_{f.get('plane')}_{f.get('occ')}",
                         "motion": "static" if f.get("static") else "moving", "src_A": L[ia], "src_B": L[ib]})
        tf.append({"block_id": b, "key": list(k), "A": L[ia], "B": L[ib]})
    d3 = f"{REPO}/data_csv/{a.test_name}"; os.makedirs(d3, exist_ok=True)
    with open(f"{d3}/index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump(tf, open(f"{d3}/test_families.json", "w"), indent=1)
    print("시험", len(pick), "묶음 ·", len(rows), "영상 ·", dict(collections.Counter(r["condition"] for r in rows if r["variant"] == "pos_a")))


if __name__ == "__main__":
    main()
