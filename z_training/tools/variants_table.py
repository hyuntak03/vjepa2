#!/usr/bin/env python3
"""학습 variant 표 (2026-10-18). z_training/runs/* 의 config · 로그 · 채점 산출물에서 다시 계산한다 (손으로 옮기지 않는다).

  /venv/vjepa/bin/python z_training/tools/variants_table.py            # markdown 표 (설정 · 상태 · 성적)
  /venv/vjepa/bin/python z_training/tools/variants_table.py --json     # 같은 내용 JSON

성적 정의 (시트와 같다):
  IntPhys1 = A.8 창별 best_avg 의 최고 · IntPhys2 = 창별 Overall 의 최고 (열별 최고 규칙) ·
  InfLevel = garrido_rescore.collect_inflevel (property 별 최고 축약 → macro) · GRASP = garrido_rescore.collect (두 창 합쳐 최고) ·
  VOBench late = per_block 쌍 정확도 (late 만) · gravity = block 정의 (가능 미래가 그 block 의 모든 불가능 미래보다 낮아야 정답)
"""
import csv, glob, json, os, re, sys, statistics as st, collections, pathlib
import yaml
REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
from garrido_rescore import collect_inflevel, collect          # noqa: E402
RUNS = REPO / "z_training/runs"
VOB = REPO / "z_research/VOBench_v1/exp_results"

def reg_models():
    """레지스트리에서 run 이름 → 모델 이름 (predictor_checkpoint 의 runs/<run>/latest.pt)."""
    out = {}
    for name, m in (yaml.safe_load(open(REPO / "cvpr/registry/models.yaml")) or {}).items():
        ck = str((m or {}).get("predictor_checkpoint", ""))
        mm = re.search(r"z_training/runs/([^/]+)/latest\.pt", ck)
        if mm and mm.group(1) not in out:
            out[mm.group(1)] = name
    return out

def vob_late(model):
    p = VOB / f"surprise_c16t32__vobench_v1_{model}/per_block.json"
    if not p.is_file():
        p = VOB / f"surprise_c16t32__vobench_v1_late_{model}/per_block.json"
    if not p.is_file():
        return None
    a = collections.defaultdict(lambda: [0, 0])
    for b in json.load(open(p))["per_block"]:
        ob, vi, oc = b["block_type"].split("/")
        if oc == "late":
            a[ob][0] += b["acc"] * b["n_pairs"]; a[ob][1] += b["n_pairs"]
    if not a["primitive"][1]:
        return None
    return tuple(100 * a[o][0] / a[o][1] for o in ("primitive", "realistic"))

GIDX = None
def vob_gravity(model):
    global GIDX
    p = VOB / f"surprise_c16t32__vobench_v1_gravity_{model}/per_block.json"
    if not p.is_file():
        return None
    if GIDX is None:
        GIDX = list(csv.DictReader(open(REPO / "data_csv/vobench_v1_gravity/index.csv")))
    pv = json.load(open(p))["per_video_surprise"]
    get = lambda v: v["all"] if isinstance(v, dict) else v
    blk = collections.defaultdict(list)
    for r in GIDX:
        blk[r["block_id"]].append(r)
    acc = collections.defaultdict(list)
    for rows in blk.values():
        pos = [get(pv[r["video_id"]]) for r in rows if r["plausible"] == "1"]
        imp = [get(pv[r["video_id"]]) for r in rows if r["plausible"] == "0"]
        acc[rows[0]["objset"]].append(float(pos[0] < min(imp)))
    return tuple(100 * sum(acc[o]) / len(acc[o]) for o in ("primitive", "realistic"))

def bench(E):
    i1 = None
    for f in glob.glob(str(E / "intphys1_sliding__intphys1_dev_latest_w*/summary.json")):
        v = 100 * float((json.load(open(f)).get("best_avg") or {}).get("block_pairwise", float("nan")))
        i1 = v if i1 is None or v > i1 else i1
    i2 = None
    for f in glob.glob(str(E / "intphys2_main_latest_w*/summary.json")):
        v = 100 * float((json.load(open(f)).get("overall") or {}).get("pairwise_accuracy", float("nan")))
        i2 = v if i2 is None or v > i2 else i2
    inf = collect_inflevel(E, "latest")
    gr = collect(E, "grasp_level2", "latest")
    return i1, i2, (inf["best"]["accuracy"] if inf else None), (100 * gr["best"]["accuracy"] if gr and gr["best"]["accuracy"] <= 1 else (gr["best"]["accuracy"] if gr else None))

def run_info(r):
    D = RUNS / r
    c = yaml.safe_load(open(D / "config.yaml"))
    m, L, O = c.get("model", {}), c.get("loss", {}), c.get("optimization", {})
    p = m.get("predictor") or {}
    vw = (c.get("data") or {}).get("var_window") or {}
    log = open(D / "train.log", errors="ignore").read() if (D / "train.log").is_file() else ""
    last = re.findall(r"\[(\d+), +\d+\] loss", log)
    done = bool(re.search(r"\] done$", log, re.M))
    bs = {}
    for e, i, b in re.findall(r"\[(\d+), +(\d+)\] loss .*?mask var C\d+K\d+ \w+ B(\d+)", log):
        bs[(e, i)] = int(b)
    ipe, ep = int(O.get("ipe", 0) or 0), int(O.get("epochs", 0) or 0)
    ws = 8
    steps = (int(last[-1]) - 1) * ipe + ipe if last and done else ((int(last[-1]) - 1) * ipe if last else 0)
    seen = st.mean(bs.values()) * ws * steps / 1e6 if bs and steps else None
    vals = []
    if (D / "metrics.jsonl").is_file():
        for ln in open(D / "metrics.jsonl"):
            try:
                v = (json.loads(ln).get("val") or {}).get("acc")
            except Exception:
                v = None
            if v is not None:
                vals.append(100 * v)
    kind = p.get("kind", "oneshot")
    feat = []
    if kind == "ar":
        feat.append(f"in={p.get('ar_input', 'block')}/tgt={p.get('ar_target', 'block')}")
        feat.append(f"fb={p.get('ar_feedback', 'output')}")
        feat.append("tf=문맥만" if p.get("ar_input") == "video" else f"tf={L.get('tf_span', 'all')}")
    if kind == "ctx_state":
        feat.append(f"enc={p.get('ctx_encoder') or m.get('context_encoder_key', 'encoder')}×{p.get('ctx_input', 'video')}/tgt={p.get('ctx_target', 'window')}")
        feat.append(f"tf_w={L.get('tf_weight', 1.0)}")
    if p.get("gating_residual"): feat.append("gate" + (f"({p.get('gating_anchor')})" if p.get("gating_anchor") else ""))
    if p.get("hint_cross_dim"): feat.append("hint")
    if L.get("delta_weight"): feat.append(f"delta {L['delta_weight']}")
    if L.get("dyn_weight"): feat.append(f"dyn {L['dyn_weight']}")
    if L.get("ctx_hole"): feat.append("hole")
    for k in ("fixed", "boundary"):
        if vw.get(k): feat.append(f"+{vw[k].get('name', k)} {100 * float(vw[k]['p']):.0f}%")
    win = f"C{vw.get('context_frames')} K{vw.get('predict_frames')}" if vw else "고정 창"
    reg = REG.get(r)
    i1, i2, inf, gr = bench(D / "eval") if (D / "eval").is_dir() else (None,) * 4
    return {"run": r, "kind": kind, "feat": " · ".join(feat), "win": win, "epoch": f"{last[-1] if last else '-'}/{ep}",
            "done": done, "seen_M": seen, "val_last": vals[-1] if vals else None, "val_max": max(vals) if vals else None,
            "intphys1": i1, "intphys2": i2, "inflevel": inf, "grasp": gr,
            "vob_late": vob_late(reg) if reg else None, "vob_grav": vob_gravity(reg) if reg else None, "registry": reg}

REG = reg_models()
rows = [run_info(r) for r in sorted(os.listdir(RUNS)) if (RUNS / r / "config.yaml").is_file()]
if "--json" in sys.argv:
    print(json.dumps(rows, ensure_ascii=False, indent=1)); sys.exit()
f = lambda v, d=1: "—" if v is None else f"{v:.{d}f}"
f2 = lambda t: "—" if t is None else f"{t[0]:.1f} / {t[1]:.1f}"
print("| run | kind | 구성 | 창 (장) | epoch | 완료 | 본 clip (M) | val 최종 / 최고 | IntPhys1 A.8 | VOBench late P / R | gravity P / R | IntPhys2 | GRASP | InfLevel |")
print("|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for x in rows:
    print(f"| `{x['run']}` | {x['kind']} | {x['feat'] or '—'} | {x['win']} | {x['epoch']} | {'✅' if x['done'] else '—'} | {f(x['seen_M'], 2)} | "
          f"{f(x['val_last'])} / {f(x['val_max'])} | {f(x['intphys1'])} | {f2(x['vob_late'])} | {f2(x['vob_grav'])} | {f(x['intphys2'])} | {f(x['grasp'])} | {f(x['inflevel'])} |")
