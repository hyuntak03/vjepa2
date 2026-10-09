"""(SC5) EK100 Δ 재계산: jsonl 마지막 줄, head 8 개 중 최고 (독립 선택) 와 같은 head 선택."""
import json
from pathlib import Path
from vcommon import dump
R = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2/z_research/anticipation/EK100/exp_results/action_anticipation_frozen")
PAIRS = {"rel_1s": ("ek100_vith_rel_grid8_val_repro", "ek100_vith_rel_grid8_val_copylast"), "rel_1s_train_final": ("ek100_vith_official256_released_grid8", "ek100_vith_rel_grid8_val_copylast"), "pap_1s_train_final": ("ek100_vith_official256_paper_grid8", "ek100_vith_pap_grid8_val_copylast"),
         "rel_2s": ("ek100_vith_rel_grid8_val_at2", "ek100_vith_rel_grid8_val_at2_copylast"), "rel_3s": ("ek100_vith_rel_grid8_val_at3", "ek100_vith_rel_grid8_val_at3_copylast"),
         "pap_2s": ("ek100_vith_pap_grid8_val_at2", "ek100_vith_pap_grid8_val_at2_copylast")}
def last(tag):
    lines = [l for l in open(R / tag / "val_metrics.jsonl") if l.strip()]; j = json.loads(lines[-1]); return j, len(lines)
res = {}
for k, (a, b) in PAIRS.items():
    ja, na = last(a); jb, nb = last(b)
    keys = [x for x in ja.keys() if x not in ("heads",)]
    def heads(j):
        for kk in ("val", "metrics", "results", "per_head", "heads_metrics"):
            if kk in j and isinstance(j[kk], list): return j[kk]
        for kk, v in j.items():
            if isinstance(v, list) and v and isinstance(v[0], dict) and "action" in v[0]: return v
        raise KeyError(list(j.keys()))
    ha, hb = heads(ja), heads(jb)
    ia = max(range(len(ha)), key=lambda i: ha[i]["action"]["recall"]); ib = max(range(len(hb)), key=lambda i: hb[i]["action"]["recall"])
    f = lambda h: [round(h["verb"]["recall"], 2), round(h["noun"]["recall"], 2), round(h["action"]["recall"], 2)]
    pm = lambda hs: [round(max(h[m]["recall"] for h in hs), 2) for m in ("verb", "noun", "action")]
    res[k] = dict(per_metric_max_orig=pm(ha), per_metric_max_copy=pm(hb), delta_per_metric_max=[round(x - y, 2) for x, y in zip(pm(ha), pm(hb))], keys=keys, n_heads=len(ha), orig_best=f(ha[ia]), orig_best_head=ja["heads"][ia], copy_best=f(hb[ib]), copy_best_head=jb["heads"][ib],
                  delta_indep=[round(x - y, 2) for x, y in zip(f(ha[ia]), f(hb[ib]))], copy_same_head_as_orig=f(hb[ia]), delta_same_head=[round(x - y, 2) for x, y in zip(f(ha[ia]), f(hb[ia]))],
                  extra_fields={kk: ja[kk] for kk in ja if not isinstance(ja[kk], list)})
dump("sc5", res); print(json.dumps(res, indent=1, default=str))
