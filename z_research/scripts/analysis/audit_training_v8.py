#!/usr/bin/env python3
"""presence·위치 자의 학습셋 `RollOut_v2_training` (training_v8, 14,360 clip) 감사 — 모델 불필요, CPU 몇 초.

자 (`rollout2_presence_readout.py`) 가 실제로 읽는 index (`data_csv/rollout_v2_training_v8/index.csv`) 와
metadata 를 대조해 다음을 잰다. 전부 **미래 튜블릿 8 개** (자가 학습에 쓰는 칸) 기준이다.

  A. 라벨 산술   visible / ignore 가 in_frame · prop_intersects · scenery_intersects 에서 규칙대로 나왔나,
                 학습 요약 (summary.json 의 n_pos/n_neg/n_ignore) 과 같은가
  B. 음성의 출처  빈 장면 / 완전히 화면 밖 / **일부만 화면 밖 (중심은 화면 안)** / 튜블릿 안에서 샘플 하나만 보임
  C. 판 자세 ↔ 라벨  (폭·높이·두께·기울기·깊이) 자세마다 양성 비율, balance_weight 적용 전후 상호정보량
  D. split 누수  같은 궤적 (화면 좌표열이 같은 clip) 이 train 과 test 에 동시에 들어가나
  E. v11 과의 차이  가림막의 화면 크기 · 깊이 · 재질 · 자세 vs 학습셋 판 (둘 다 metadata 에서)
  F. v11 가장자리  visible 이동 조건 미래 튜블릿에서 학습 정의 ('물체 전체가 화면 안') 로 양성인 비율

  $P z_research/scripts/analysis/audit_training_v8.py   → z_research/RollOutV3/audit/training_v8/audit.json
"""
from __future__ import annotations
import csv, json, collections
from pathlib import Path
import numpy as np

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
INDEX = ROOT / "data_csv/rollout_v2_training_v8/index.csv"
META = Path("/data2/local_datasets/world/world_analysis/RollOut_v2_training/metadata.csv")
V11 = Path("/local_datasets/world/world_analysis/IntPhysGen_v11/metadata.csv")
SUMMARY = ROOT / "z_research/RollOutV3/exp_results/presence/summary.json"
READ = ROOT / "z_research/RollOutV3/exp_results/v11/readings.npz"
OUT = ROOT / "z_research/RollOutV3/audit/training_v8"
T = 8                        # 미래 튜블릿 시작 (튜블릿 16 개 중 8..15)
CAM_Y = -1150.0              # 두 세트 공통 카메라 (metadata 에서 확인)


def arr(s, n=32):
    s = str(s).split()
    return np.array([float(v) for v in s], np.float32) if s else np.zeros(n, np.float32)


def tub(x, how):             # (n,32) → (n,16) 에서 미래 8 개
    x = x.reshape(len(x), 16, 2)
    return (x.all(2) if how == "all" else x.any(2))[:, T:]


def mi_bits(x, y, w):
    """가중 상호정보량 (bits). x, y 는 범주, w 는 가중치."""
    w = w / w.sum(); out = 0.0
    for a in np.unique(x):
        pa = w[x == a].sum()
        for b in np.unique(y):
            pab = w[(x == a) & (y == b)].sum(); pb = w[y == b].sum()
            if pab > 0:
                out += pab * np.log2(pab / (pa * pb))
    return float(out)


def main():
    R = list(csv.DictReader(INDEX.open())); M = {r["name"]: r for r in csv.DictReader(META.open())}
    n = len(R); res = {}
    meta = [M[r["video_id"]] for r in R]
    inf = np.stack([arr(r["in_frame_by_sample"]) > 0 for r in R])
    vis = np.stack([arr(r["visible_by_sample"]) > 0 for r in R])
    ign = np.stack([arr(r["ignore_by_sample"]) > 0 for r in R])
    pi = np.stack([arr(r["prop_intersects_by_sample"]) > 0 for r in R])
    si = np.stack([arr(r["scenery_intersects_by_sample"]) > 0 for r in R])
    px = np.stack([arr(r["px_x_by_sample"]) for r in R]); py = np.stack([arr(r["px_y_by_sample"]) for r in R])
    empty = np.array([r["scenario"] == "empty" for r in R])
    bw = np.array([float(r["balance_weight"] or 1) for r in R])

    # ── A. 라벨 산술
    ok_vis = (vis == (inf & ~pi & ~si)).all(); ok_ign = (ign == ((inf & pi) | si)).all()
    pos = tub(vis, "all"); ig = tub(ign, "any") & ~pos; neg = ~pos & ~ig
    S = json.loads(SUMMARY.read_text())
    res["A_label_rule"] = dict(visible_rule_holds=bool(ok_vis), ignore_rule_holds=bool(ok_ign),
                               n_pos=int(pos.sum()), n_neg=int(neg.sum()), n_ignore=int(ig.sum()),
                               summary=dict(n_pos=S["n_pos"], n_neg=S["n_neg"], n_ignore=S["n_ignore"]),
                               matches_summary=(int(pos.sum()), int(neg.sum()), int(ig.sum())) == (S["n_pos"], S["n_neg"], S["n_ignore"]))

    # ── B. 음성의 출처. 반폭 = obj_apparent_px / 2 (288 px 기준)
    half = np.array([float(r["obj_apparent_px"] or 0) / 2 for r in R])[:, None]
    ctr_in = (px >= 0) & (px < 288) & (py >= 0) & (py < 288)                     # 중심은 화면 안
    part = ctr_in & ~inf                                                          # 중심은 안인데 '물체 전체가 안' 은 아님
    ctr_in_t, part_t, inf_any, inf_all = tub(ctr_in, "any"), tub(part, "any"), tub(inf, "any"), tub(inf, "all")
    src = np.full(neg.shape, "?", object)
    src[neg & empty[:, None]] = "empty_clip"
    obj_neg = neg & ~empty[:, None]
    src[obj_neg & inf_any & ~inf_all] = "one_sample_in_frame"                     # 튜블릿 두 샘플 중 하나만 화면 안
    src[obj_neg & ~inf_any & part_t] = "partly_in_frame"                         # 중심은 안인데 일부가 잘림
    src[obj_neg & ~inf_any & ~part_t] = "fully_off_screen"
    cnt = collections.Counter(src[neg]); wsum = collections.Counter()
    for s_ in cnt:
        wsum[s_] = float((np.repeat(bw[:, None], 8, 1)[neg & (src == s_)]).sum())
    res["B_negative_sources"] = dict(count=dict(cnt), weighted=dict(wsum),
                                     half_px=float(np.median(half)))

    # ── C. 판 자세 ↔ 라벨. 판 없음은 prop_w 0 또는 빈 값
    def pose(m):
        w = m["prop_w"]
        if not w or float(w) == 0:
            return "none"
        return f"{float(w):.0f}x{float(m['prop_h']):.0f} t{float(m['prop_thick']):.0f} p{float(m['prop_pitch']):+.0f} y{float(m['prop_y']):.0f}"
    P = np.array([pose(m) for m in meta]); keep = ~ig
    tabs = {}
    for p_ in sorted(set(P)):
        m = (P == p_)[:, None] & keep
        w = np.repeat(bw[:, None], 8, 1)
        tabs[p_] = dict(clips=int((P == p_).sum()), empty_clips=int(((P == p_) & empty).sum()),
                        pos_frac=round(float(pos[m].mean()), 3),
                        pos_frac_weighted=round(float((w[m] * pos[m]).sum() / w[m].sum()), 3))
    x = np.repeat(P[:, None], 8, 1)[keep]; y = pos[keep]
    wts = np.repeat(bw[:, None], 8, 1)[keep]
    has = np.array([p_ != "none" for p_ in x])
    res["C_prop_pose"] = dict(per_pose=tabs,
                              mi_pose_bits=dict(unweighted=mi_bits(x, y, np.ones_like(wts)), weighted=mi_bits(x, y, wts)),
                              mi_has_prop_bits=dict(unweighted=mi_bits(has, y, np.ones_like(wts)), weighted=mi_bits(has, y, wts)),
                              label_entropy_bits=float(-(lambda q: q * np.log2(q) + (1 - q) * np.log2(1 - q))(y.mean())),
                              empty_only_poses=[p_ for p_, t in tabs.items() if t["empty_clips"] == t["clips"]],
                              object_only_poses=[p_ for p_, t in tabs.items() if t["empty_clips"] == 0])

    # ── D. split 누수 (학습 코드와 같은 규칙: block 단위, scenario 층화, 50/15/35, seed 0)
    blk = np.array([r["block_id"] for r in R]); sc = np.array([r["scenario"] for r in R])
    rng = np.random.RandomState(0); part_ = {}
    for s_ in np.unique(sc):
        b = rng.permutation(np.unique(blk[sc == s_])); n1 = int(round(len(b) * .5)); n2 = int(round(len(b) * .65))
        for x_ in b[:n1]: part_[x_] = 0
        for x_ in b[n1:n2]: part_[x_] = 1
        for x_ in b[n2:]: part_[x_] = 2
    g = np.array([part_[x_] for x_ in blk])
    traj = np.array([r["px_x_by_sample"] + "|" + r["px_y_by_sample"] for r in R])
    obj = ~empty
    sets = collections.defaultdict(set)
    for t_, g_ in zip(traj[obj], g[obj]):
        sets[t_].add(int(g_))
    n_traj = len(sets); cross = sum(1 for v in sets.values() if len(v) > 1)
    te_obj = obj & (g == 2)
    te_seen = sum(1 for t_ in traj[te_obj] if 0 in sets[t_])
    res["D_split"] = dict(clips_per_block=dict(collections.Counter(collections.Counter(blk).values())),
                          n_traj=n_traj, traj_in_more_than_one_split=cross,
                          test_obj_clips=int(te_obj.sum()), test_obj_clips_whose_trajectory_is_in_train=int(te_seen))

    # ── E. v11 가림막 vs 학습 판 — 화면 크기 (px, 288 기준). 폭 px = w / (2 · half_width(깊이)) · 288
    V = list(csv.DictReader(V11.open()))
    fhw = float(meta[0]["frame_half_width_cm"]); d_obj = float(meta[0]["obj_depth_cm"])
    def px_w(w_cm, dist):
        return w_cm / (2 * fhw * dist / d_obj) * 288
    tr = {}
    for p_ in sorted(set(P) - {"none"}):
        m = next(mm for mm, q in zip(meta, P) if q == p_)
        w, h, th, y = (float(m[k]) for k in ("prop_w", "prop_h", "prop_thick", "prop_y"))
        front = y - th / 2 - CAM_Y                                                   # 판 앞면까지 거리
        tr[p_] = dict(front_dist_cm=round(front, 1), w_px=round(px_w(w, front), 1), h_px=round(px_w(h, front), 1))
    occ = collections.defaultdict(list)
    for r in V:
        if r["occ_width_cm"]:
            occ[(r["condition"].replace("_early", "").replace("_mid", ""), r["occ_timing"] or "late")].append(
                (float(r["occ_width_cm"]), float(r["occ_height_cm"]), float(r["occ_depth_cm"]), float(r["occ_apparent_px"] or 0)))
    v11 = {f"{c}|{tm}": dict(n=len(v), w_px=[round(px_w(min(x[0] for x in v), v[0][2]), 1), round(px_w(max(x[0] for x in v), v[0][2]), 1)],
                             h_px=round(px_w(v[0][1], v[0][2]), 1), dist_cm=v[0][2],
                             occ_apparent_px=[round(min(x[3] for x in v), 1), round(max(x[3] for x in v), 1)])
           for (c, tm), v in sorted(occ.items())}
    res["E_v11_vs_train_plates"] = dict(
        train_plates=tr, v11_panels=v11,
        train_material="M_Wood_Pine (gen/geom.py prop_material)", v11_material="/Game/Materials/MI_OccDark (gen/geom.py panel_material)",
        v11_static_panel="trapdoor: 바닥에 눕혀져 있다가 static_swing 6 프레임에 걸쳐 일어나고 다시 눕는다 (occ_rise/fall)",
        v11_moving_panel="raw 0–6 에 일어나 93–99 에 눕는다 = 채점 창 (0–93) 내내 서 있다",
        train_pose="전부 서 있다 (pitch 0 / ±6 / ±10 / ±14 도 기울기). 눕힌 판·움직이는 판은 없다")

    # ── F. v11 visible 이동 조건: 학습 정의 ('물체 전체가 화면 안', 튜블릿 두 샘플 모두) 로 양성인 비율
    z = np.load(READ, allow_pickle=True)
    k = z["sym_k"].astype(int); cond = z["condition"]
    f = {}
    for c in ("moving_visible_flat", "moving_visible", "static_visible"):
        m = z["obj"] & (cond == c) & (k == 0)
        f[c] = dict(n=int(m.sum()), in_frame_frac=[round(float(x), 3) for x in tub(z["in_frame"][m], "all").mean(0)])
    res["F_v11_visible_in_frame"] = f

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
