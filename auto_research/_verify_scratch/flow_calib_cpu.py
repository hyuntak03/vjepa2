# CPU 전용: v3 에서 flow 추적 · 질의 선택이 GT 와 맞는지 (모델 없이)
import sys, numpy as np
sys.path.insert(0, "auto_research/scripts")
import arlib, e_flowtrack as E
from h23_extract_v3 import arr
ds = arlib.WMADataset(dict(arlib.v3_cfg(0, 64, split=0))); R = ds.records
items = [i for i, r in enumerate(R) if r.plausible == "1" and r.raw["scenario"] in E.FREE][::40]
print("n", len(items))
rows = []
for i in items:
    x0 = ds.clip(i)[:, E.V3_FR].float()
    raw = R[i].raw; gx, gy = arr(raw["px_x_by_sample"]) * E.V3_SCALE, arr(raw["px_y_by_sample"]) * E.V3_SCALE
    g = E.to_gray(x0); fl = E.flows(g)
    e, q, cx, cy = E.pick_query(fl); qy, qx = q // 16, q % 16
    gq = ((gx[E.V3_FR[14]] + gx[E.V3_FR[15]]) / 2, (gy[E.V3_FR[14]] + gy[E.V3_FR[15]]) / 2)
    if not np.all(np.isfinite(gq)): continue
    gqc = (int(gq[0] // 16), int(gq[1] // 16))
    qerr = max(abs(qx - gqc[0]), abs(qy - gqc[1]))
    # flow 추적: 질의 칸 중심에서 · GT 위치에서
    fw = E.track(fl, cx, cy, 15, 31); fg = E.track(fl, gq[0], gq[1], 15, 31)
    er_q, er_g, disp = [], [], []
    for k in range(8):
        a, b = E.V3_FR[16 + 2 * k], E.V3_FR[17 + 2 * k]
        gt = ((gx[a] + gx[b]) / 2, (gy[a] + gy[b]) / 2)
        pq = ((fw[16 + 2 * k][0] + fw[17 + 2 * k][0]) / 2, (fw[16 + 2 * k][1] + fw[17 + 2 * k][1]) / 2)
        pg = ((fg[16 + 2 * k][0] + fg[17 + 2 * k][0]) / 2, (fg[16 + 2 * k][1] + fg[17 + 2 * k][1]) / 2)
        er_q.append(max(abs(pq[0] - gt[0]), abs(pq[1] - gt[1])) / 16); er_g.append(max(abs(pg[0] - gt[0]), abs(pg[1] - gt[1])) / 16)
        disp.append(max(abs(gt[0] - gq[0]), abs(gt[1] - gq[1])) / 16)
    rows.append((raw["scenario"], qerr, er_q, er_g, disp))
qe = np.array([r[1] for r in rows]); EQ = np.array([r[2] for r in rows]); EG = np.array([r[3] for r in rows]); DS = np.array([r[4] for r in rows])
print("query within 1 cell of GT object:", (qe <= 1).mean().round(3), " exact:", (qe == 0).mean().round(3))
print("flow track from GT start: err cells mean per slot", EG.mean(0).round(2), " frac<=1", (EG <= 1).mean(0).round(2))
print("flow track from query   : err cells mean per slot", EQ.mean(0).round(2), " frac<=1", (EQ <= 1).mean(0).round(2))
print("GT displacement cells per slot", DS.mean(0).round(2))
for sc in sorted(set(r[0] for r in rows)):
    m = np.array([r[0] == sc for r in rows]); print(sc, m.sum(), "q<=1 %.2f" % (qe[m] <= 1).mean(), "flowQ slot7<=1 %.2f" % (EQ[m, 7] <= 1).mean(), "disp7 %.1f" % DS[m, 7].mean())
