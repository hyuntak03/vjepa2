#!/usr/bin/env python3
"""[물리] [P] 공유 forward (ctxt_compare/phys3d_*) 에 라벨 새 칸 정의를 건다 — 차이 가르기 칸 A (GPU 없음) (auto_research, 2026-10-22 KST).

  /venv/vjepa/bin/python auto_research/scripts/presence_p_windows.py

사전 등록: 진행판 [물리] 20:54 UTC (KST 05:54).
[P] 창 = enc_noise phys3d (pix_val_s* 의 창 앞 16 장 = 8 블록), 문맥 블록 0..3 → 예측 블록 4..7. 걸음 1 = 예측 블록 4 (영상 문맥은 z 만).
라벨 = 같은 창의 lab_val (장 단위 위치 · 보임). 블록 b 의 위치 = 둘째 장 (2b+1).
존재감 = Σ⟨p − m, y − m⟩ / Σ‖y − m‖², m = 블록 0..3 같은 칸 평균.
  흐름 칸 ([P] 정의 재현) = |U[4]| ≥ 0.5 칸, 정지 = |U[4]|, |U[3]| < 0.25.
  라벨 새 칸 (내 정의) = 블록 4 물체 3×3 중 블록 3 물체 3×3 에 없던 칸, 정지 = 16 장 어디서도 물체 3×3 이 아닌 칸 중 미래 변화 중앙값 아래."""
import json, os
import numpy as np

CC = "/workspace/auto_research_cache/ctxt_compare"
EN = "/workspace/auto_research_cache/enc_noise"
P3 = "/workspace/auto_research_cache/phys3d"
REPO = "/workspace/world_model/vjepa2"
S, G = 256, 16
OUT = os.path.join(REPO, "auto_research/results/presence_p_windows.json")


def cells3(cx, cy):
    return [yy * G + xx for yy in range(max(0, cy - 1), min(G, cy + 2)) for xx in range(max(0, cx - 1), min(G, cx + 2))]


def cell_of(p):
    return int(np.clip(p[0] // 16, 0, G - 1)), int(np.clip(p[1] // 16, 0, G - 1))


wids = json.load(open(os.path.join(CC, "phys3d_meta.json")))["windows"]
gt = np.load(os.path.join(CC, "phys3d_gt.npy"), mmap_mode="r"); ctx = np.load(os.path.join(CC, "phys3d_ctx.npy"), mmap_mode="r")
pred = {n: np.load(os.path.join(CC, f"phys3d_pred_{n}.npy"), mmap_mode="r") for n in ("base", "ctxt")}
Un = np.load(os.path.join(REPO, "auto_research/results/natural_transport_ctx5/rows_phys3d.npz"))["Uwin"].astype(np.float32)
# 창 id → (enc_noise 조각, 그 안 번호) → (pix_val 조각, 창 번호) → 라벨
emap = []
for sh in (0, 1):
    m = json.load(open(os.path.join(EN, f"phys3d_target_encoder_s{sh}_meta.json")))
    emap += [(sh, w, d) for w, d in enumerate(m)]
assert len(emap) == len(wids) == gt.shape[0], (len(emap), len(wids), gt.shape)
labs = {}
chk = []
SUM = {n: {g: [0.0, 0.0, 0] for g in ("flow_enter", "flow_static", "lab_new", "lab_full", "lab_static", "lab_new_v", "lab_full_v", "lab_static_v")} for n in pred}
nwin = {"lab": 0, "lab_v": 0}
PW = []                                                                             # 창별 (모델, 무리) → (num, den) — bootstrap 용
for i, wid in enumerate(wids):
    sh, w, d = emap[wid]
    psh = int(os.path.basename(d["path"]).split("_s")[1].split(".")[0])
    if psh not in labs:
        labs[psh] = np.load(os.path.join(P3, f"lab_val_s{psh}.npz"))
    L = labs[psh]
    if i < 4:                                                                       # 픽셀이 같은 창인가 (검산)
        a_ = np.load(os.path.join(EN, f"phys3d_target_encoder_s{sh}_pix.npy"), mmap_mode="r")[w]
        b_ = np.load(d["path"], mmap_mode="r")[d["start"]][:16]
        chk.append(bool((np.asarray(a_) == np.asarray(b_)).all()))
    pos, vis = L["pos"][d["start"], :16, 0] if L["pos"].ndim == 4 else L["pos"][d["start"], :16], L["vis_frac"][d["start"], :16]
    vis = vis[..., 0] if vis.ndim == 2 else vis
    m_ = ctx[i].astype(np.float32).mean(0); y = gt[i, 0].astype(np.float32)
    U = Un[wid].reshape(8, S, 2)
    u1, u0 = np.linalg.norm(U[4], axis=1), np.linalg.norm(U[3], axis=1)
    groups = {"flow_enter": np.where(u1 >= 0.5)[0], "flow_static": np.where((u1 < 0.25) & (u0 < 0.25))[0]}
    p4, p3 = pos[9], pos[7]
    if vis[9] > 0.3 and np.isfinite(p4).all() and np.isfinite(p3).all():
        old3 = cells3(*cell_of(p3)); f3 = cells3(*cell_of(p4))
        objc = set()
        for f in range(16):
            if np.isfinite(pos[f]).all():
                objc |= set(cells3(*cell_of(pos[f])))
        nonobj = np.array([c for c in range(S) if c not in objc])
        if len(nonobj):
            chg = np.abs(gt[i][:, nonobj].astype(np.float32) - m_[nonobj]).mean(-1).max(0)
            stc = nonobj[chg <= np.median(chg)]
        else:
            stc = nonobj
        new = np.array([c for c in f3 if c not in old3], int)
        groups.update(lab_new=new, lab_full=np.array(f3), lab_static=stc)
        nwin["lab"] += 1
        if vis[7] > 0.3:                                                           # 내 창 규칙: 문맥 끝에 물체가 보임
            groups.update(lab_new_v=new, lab_full_v=np.array(f3), lab_static_v=stc)
            nwin["lab_v"] += 1
    pw = {}
    for n in pred:
        p = pred[n][i, 0].astype(np.float32)
        for g, cl in groups.items():
            if len(cl) == 0:
                continue
            dp, dy = p[cl] - m_[cl], y[cl] - m_[cl]
            a = SUM[n][g]; a[0] += float((dp * dy).sum()); a[1] += float((dy * dy).sum()); a[2] += len(cl)
            pw[(n, g)] = (float((dp * dy).sum()), float((dy * dy).sum()))
    PW.append(pw)
res = {"pixel_check": chk, "n_windows": len(wids), "n_label": nwin}
print(f"픽셀 검산 {chk} · 창 {len(wids)} · 라벨 새 칸 창 {nwin}")
for n in pred:
    r = {g: (v[0] / max(v[1], 1e-9), v[2]) for g, v in SUM[n].items()}
    res[n] = {g: [round(x, 4) if isinstance(x, float) else x for x in v] for g, v in r.items()}
    print(f"  {n:5s} 흐름 칸 {r['flow_enter'][0]:.3f} − 정지 {r['flow_static'][0]:.3f} = {r['flow_enter'][0] - r['flow_static'][0]:+.3f} | "
          f"라벨 새 칸 {r['lab_new'][0]:.3f} (3×3 {r['lab_full'][0]:.3f}) − 정지 {r['lab_static'][0]:.3f} = {r['lab_new'][0] - r['lab_static'][0]:+.3f} | "
          f"문맥 끝 보임 창만: {r['lab_new_v'][0]:.3f} − {r['lab_static_v'][0]:.3f} = {r['lab_new_v'][0] - r['lab_static_v'][0]:+.3f}")
for g in ("flow_enter", "lab_new", "lab_new_v"):
    st = {"flow_enter": "flow_static", "lab_new": "lab_static", "lab_new_v": "lab_static_v"}[g]
    d = [(SUM[n][g][0] / SUM[n][g][1]) - (SUM[n][st][0] / SUM[n][st][1]) for n in ("ctxt", "base")]
    dr = [SUM[n][g][0] / SUM[n][g][1] for n in ("ctxt", "base")]
    print(f"  영상 문맥 − 기준 AR, {g}: 그대로 {dr[0] - dr[1]:+.3f} · 정지 뺀 값 {d[0] - d[1]:+.3f}")
    rng = np.random.RandomState(0); bs = []
    for _ in range(1000):
        ix = rng.randint(0, len(PW), len(PW))
        def r_(n, gg):
            nu = sum(PW[j][(n, gg)][0] for j in ix if (n, gg) in PW[j]); de = sum(PW[j][(n, gg)][1] for j in ix if (n, gg) in PW[j])
            return nu / max(de, 1e-9)
        bs.append((r_("ctxt", g) - r_("ctxt", st)) - (r_("base", g) - r_("base", st)))
    ci = np.percentile(bs, [2.5, 97.5])
    print(f"      정지 뺀 차의 창 bootstrap 95 %: [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    res[f"diff_{g}"] = [dr[0] - dr[1], d[0] - d[1], float(ci[0]), float(ci[1])]
json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
print("저장", OUT)
