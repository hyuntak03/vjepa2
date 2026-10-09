import sys, numpy as np
sys.path.insert(0, "auto_research/scripts")
import arlib, e_flowtrack as E
from h23_extract_v3 import arr
ds = arlib.WMADataset(dict(arlib.v3_cfg(0, 64, split=0))); R = ds.records
items = [i for i, r in enumerate(R) if r.plausible == "1" and r.raw["scenario"] in ("flat_v","flat_d")][::60]
for i in items[:8]:
    x0 = ds.clip(i)[:, E.V3_FR].float(); raw = R[i].raw
    gx, gy = arr(raw["px_x_by_sample"]) * E.V3_SCALE, arr(raw["px_y_by_sample"]) * E.V3_SCALE
    g = E.to_gray(x0); fl = E.flows(g)
    e, q, cx, cy = E.pick_query(fl)
    gq = ((gx[28] + gx[30]) / 2, (gy[28] + gy[30]) / 2)
    mag = np.linalg.norm(fl[12:15], axis=-1).sum(0)
    # GT 위치의 flow vs 추적 경로
    fw = E.track(fl, cx, cy, 15, 31)
    gt7 = ((gx[V:=E.V3_FR[30]] + gx[E.V3_FR[31]]) / 2, (gy[E.V3_FR[30]] + gy[E.V3_FR[31]]) / 2)
    vgt = (gx[E.V3_FR[16]] - gx[E.V3_FR[15]], gy[E.V3_FR[16]] - gy[E.V3_FR[15]])
    fgt = E.med_flow(fl[15], gx[E.V3_FR[15]], gy[E.V3_FR[15]])
    print(raw["scenario"], "start err px (%.1f,%.1f)" % (cx - gx[30], cy - gy[30]), "GT v/frame (%.1f,%.1f)" % vgt, "flow@GT (%.1f,%.1f)" % tuple(fgt),
          "track31 (%.0f,%.0f) gt (%.0f,%.0f)" % (fw[31][0], fw[31][1], gx[62], gy[62]), "maxmag %.1f" % mag.max(), "img mean %.0f std %.0f" % (g[15].mean(), g[15].std()))
