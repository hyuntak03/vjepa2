"""render_future.py · render_videomae.py 가 같은 클립을 고르도록 뺀 공용 함수 (2026-10-21).

클립 = PFT 에 안 쓴 test split — VOBench split v2 test 의 pos_a (set = 가림 조건) / gravity split test 의 가능 미래 (ledge 는 부유 미래도 같이).
(objset, 위반·법칙) 마다 고르게, decoder val 클립을 먼저 (random.Random(0) 섞기 → 안정 정렬).

2026-10-23 추가 (사용자 지시: "VoBench late (static, moving) VoBench gravity, IntPHys1, IntPHys2 {16,16 버전}"):
  late_static · late_moving — VOBench late 를 `motion` 으로 가른다 (static = flat_static, moving = 나머지 다섯)
  intphys1                  — IntPhys 1 dev. 쌍 (가능, 불가능) 을 같이 그린다. 프레임은 1, 4, …, 94 (start 1 · stride 3)
  intphys2_late             — IntPhys 2 Main, **사람 라벨 위반 시작 d 앞을 문맥** (cvpr/eval/intphys2_late). 문맥 16 = linspace(0, d−1), 미래 16 = linspace(d, n−1).
                              프레임은 build.py 가 뽑아 둔 PNG 32 장 (0 … 31) — 영상마다 번호가 다르므로 **idx 는 0..31 고정**이다.

2026-10-25 추가: wall — RollOut_v2 wall 392 쌍 (문맥 = 벽으로 굴러감, 미래 = 벽에서 멈춤 · 벽 통과). 속도 7 단계마다 하나.

클립마다 `idx` (읽을 프레임 번호 32 개) 와 `pattern` (폴더 뒤에 붙는 틀) 을 같이 준다 — 세트마다 다르다.
decoder val 여부 `dec_val` 은 학습 때와 같은 해시 (`train.py` 의 `isval(split_tag + "/" + row[split_key])`) 로 다시 계산한다.
⚠️ IntPhys 2 의 split_tag 는 vast 학습본의 경로 문자열 (`/workspace/…/Main/metadata.csv`) 이고 key 는 `SceneIndex` 다 —
   이 노드 index 의 `block_id` ("s0", "s154_0") 에서 되살린다. 되살림이 틀리면 **정렬 순서만** 달라지고 그림은 그대로다.
"""
import csv, hashlib, os, random

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs/_frames")

VOB_IDX = [3 * k for k in range(32)]            # VOBench · gravity: 0, 3, …, 93 (채점과 같다)
IP1_IDX = [1 + 3 * k for k in range(32)]        # IntPhys 1 dev: start 1 · stride 3 → 1, 4, …, 94
IP2_IDX = list(range(32))                       # IntPhys 2 late: build.py 가 이미 32 장으로 뽑아 뒀다


def _spread(rows, grp, n):
    """(그룹) 마다 돌아가며 n 개 — 한 조건에 몰리지 않게"""
    by = {}
    for r in rows:
        by.setdefault(grp(r), []).append(r)
    pick = []
    while len(pick) < n and any(by.values()):
        for k in sorted(by):
            if by[k] and len(pick) < n:
                pick.append(by[k].pop(0))
    return pick


def pick_clips(set_, n, root, val_frac, only=None):
    """only = block_id 집합을 주면 그것만 그린다 (2026-10-24, "ledge 정답 맞춘 케이스 샘플").
    ⚠️ only 를 주면 gravity 는 **split test 가 아니라 전체 index** 를 본다 — 정답/오답 분석은 채점 (전체) 기준이고,
       test split 제한은 gravity train 으로 PFT 한 모델 때문에 둔 것이라 그런 학습을 안 한 모델에는 필요 없다.
       only 를 쓸 때는 비교 모델이 그 데이터로 학습하지 않았는지 반드시 확인할 것."""
    """[{name, futures: [(라벨, 프레임 폴더)], dec_val, idx, pattern}] — futures 첫째가 가능 미래"""
    isval = lambda key: int(hashlib.md5(key.encode()).hexdigest(), 16) % 1000 < 1000 * val_frac
    idx, pattern = VOB_IDX, "{i:06d}.png"

    if set_ == "gravity":
        idxf = "data_csv/vobench_v1_gravity/index.csv" if only else "data_csv/vobench_v1_gravity_split/index_test.csv"
        root_, tag = os.path.join(root, "gravity"), "data_csv/vobench_v1_gravity/index.csv"
        allr = list(csv.DictReader(open(os.path.join(REPO, idxf))))
        arms = {}
        for r in allr:
            arms.setdefault(r["block_id"], []).append(r)
        rows = [r for r in allr if r["plausible"] == "1"]
        grp = lambda r: (r["objset"], r["law"])
        futs = lambda r: [(r["future"], r["file_name"])] + ([(q["future"], q["file_name"]) for q in arms[r["block_id"]] if q["plausible"] == "0"]
                                                            if r["law"] == "ledge" else [])
        key = lambda r: r["source_block"]

    elif set_ == "intphys1":                     # IntPhys 1 dev — 쌍 (가능, 불가능) 을 같이
        idxf = "/local_datasets/world/world_analysis/IntPhys1_dev_videos/index.csv"
        root_, tag = "/local_datasets/world/world_analysis/IntPhys1_dev_frame_png", "data_csv/intphys1_dev/index.csv"
        idx, pattern = IP1_IDX, "scene/scene_{i:03d}.png"
        allr = list(csv.DictReader(open(idxf)))
        pair = {}
        for r in allr:
            pair.setdefault((r["block_id"], r["pair_id"]), []).append(r)
        d_of = lambda r: os.path.join(r["block"], r["quadruplet"], r["run"])
        rows = [r for r in allr if r["plausible"] == "1"]
        grp = lambda r: (r["block_type"],)
        futs = lambda r: [(r["variant"], d_of(r))] + [(q["variant"], d_of(q)) for q in pair[(r["block_id"], r["pair_id"])] if q["plausible"] == "0"]
        key = lambda r: r["block_id"]

    elif set_ == "intphys2_late":                # IntPhys 2 Main, 위반 시작 d 앞이 문맥 (cvpr/eval/intphys2_late)
        idxf, root_ = "data_csv/intphys2_late_uni/index.csv", os.path.join(FRAMES, "IntPhys2_late_uni")
        tag = "/workspace/dataset/benchmarks/IntPhys2/Main/metadata.csv"
        idx, pattern = IP2_IDX, "{i:06d}.png"
        allr = list(csv.DictReader(open(os.path.join(REPO, idxf))))
        pair = {}
        for r in allr:
            pair.setdefault((r["block_id"], r["pair_id"]), []).append(r)
        rows = [r for r in allr if r["plausible"] == "1"]
        grp = lambda r: (r["block_type"],)
        futs = lambda r: [(r["variant"], r["video_id"])] + [(q["variant"], q["video_id"]) for q in pair[(r["block_id"], r["pair_id"])] if q["plausible"] == "0"]
        key = lambda r: r["block_id"].lstrip("s").split("_")[0]        # "s154_0" -> "154" (= SceneIndex)

    elif set_ == "wall":                         # RollOut_v2 wall (2026-10-25, 사용자: "RollOut 에서 wall 있잖아. pixel decoder 로 시각화")
        # 문맥 16 장 = 벽으로 굴러감 · 미래 첫 장 (원본 48) 에 벽에 닿는다 → 가능 = 벽에서 멈춤 (stop) · 불가능 = 벽을 지나 계속 감 (pass)
        # 프레임은 vll5 로컬 (RollOut_v2/Images/wall/roll/<video_id>) 에서 고른 클립만 NFS runs/_frames/RollOut_v2_wall/v<속도>_a<가속>/<video_id> 로 복사해 쓴다
        # decoder 는 RollOut 을 학습하지 않았다 → 전부 decoder 학습 밖 (dec_val 0). 정답 latent 행이 이 세트에서의 decoder 상한이다
        idxf, root_, tag = "data_csv/rollout_v2/index.csv", os.path.join(FRAMES, "RollOut_v2_wall"), None
        isval = lambda key: False
        allr = [r for r in csv.DictReader(open(os.path.join(REPO, idxf))) if r["scenario"] == "wall"]
        imp = {r["block_id"]: r for r in allr if r["plausible"] == "0"}
        rows = [r for r in allr if r["plausible"] == "1"]
        grp = lambda r: (int(r["primary"]),)                              # 벽에 닿는 속도 7 단계마다 하나씩
        cell = lambda r: f"v{r['primary']}_a{r['secondary']}"
        futs = lambda r: [("stop", f"{cell(r)}/{r['video_id']}"), ("pass", f"{cell(r)}/{imp[r['block_id']]['video_id']}")]
        key = lambda r: r["source_block"]

    else:                                        # VOBench 가림 조건 — late · early · mid · visible, late_static · late_moving
        occ, mot = (set_.split("_", 1) + [None])[:2] if set_.startswith("late_") else (set_, None)
        idxf, root_, tag = "data_csv/vobench_v1_split_v2/index_test.csv", root, "data_csv/vobench_v1/index.csv"
        rows = [r for r in csv.DictReader(open(os.path.join(REPO, idxf))) if r["occlusion"] == occ and r["variant"] == "pos_a"]
        if mot == "static":
            rows = [r for r in rows if r["motion"] == "flat_static"]
        elif mot == "moving":
            rows = [r for r in rows if r["motion"] != "flat_static"]
        elif mot is not None:
            raise SystemExit(f"모르는 set: {set_} (late_static | late_moving)")
        grp = lambda r: (r["objset"], r["violation"])
        futs = lambda r: [("pos_a", r["file_name"])]
        key = lambda r: r["source_block"]

    if only:
        only = {str(x) for x in only}
        rows = [r for r in rows if str(r.get("block_id")) in only]
    if not rows:
        raise SystemExit(f"set '{set_}' 에 해당하는 행이 없다 ({idxf})")
    rng = random.Random(0); rng.shuffle(rows)
    rows.sort(key=lambda r: not isval(f"{tag}/{key(r)}"))
    return [{"name": futs(r)[0][1], "futures": [(lab, os.path.join(root_, f)) for lab, f in futs(r)],
             "dec_val": isval(f"{tag}/{key(r)}"), "idx": idx, "pattern": pattern}
            for r in _spread(rows, grp, n)]
