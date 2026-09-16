#!/usr/bin/env python3
"""post-FT predictor 가 만드는 미래를 캐시 없이 바로 본다 — frozen encoder + (릴리즈 | post-FT) predictor 를 forward 하고 그 p 에 v5 위치 자를 건다.

무엇을 재나: RollOutV2 와 같은 지표. (a) RollOut_v2 가능 clip 시나리오별 R²·MAE·β(gain)·슬롯별 진실 물체 3×3 attention 질량,
(b) v11 vanish pos_a (flat/ramp × k0 / early k4 / mid k4 / late k4) 슬롯별 물체 질량·읽기−진실·읽기−마지막관측·읽기−균등읽기 (attn_diag 와 같은 식).
자는 RollOutV2 exp_results/v5/attentive_pooling/p/attn.pt (릴리즈 p 로 학습한 것) 를 그대로 쓴다 — post-FT p 에는 이식이다 (단서).
--predictor release 로 같은 파이프라인을 릴리즈 predictor 에 돌리면 캐시 기반 RollOutV2 수치와 대조할 수 있다 (파이프라인 검증).
캐시를 쓰지 않는다 (디스크 0). GPU 1 장, ViT-H 로딩 ~2 분 + clip 당 ~0.1 s.

출력: z_research/predictor_training/predictor_IntPhysGenV11_PFT/exp_results/<release|pft>/<rollout_v2|v11>/{results.json, RESULTS.md, preds.npz}

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  CUDA_VISIBLE_DEVICES=0 $P z_research/scripts/analysis/pft_ruler_direct.py --predictor pft     --set rollout_v2 v11
  CUDA_VISIBLE_DEVICES=1 $P z_research/scripts/analysis/pft_ruler_direct.py --predictor release --set rollout_v2 v11
"""
from __future__ import annotations
import argparse, csv, json, os, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np, torch, yaml

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2"); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "z_research/scripts/analysis"))
import rollout2_test_readout as rt                                   # noqa: E402  (자 경로·지표 함수)
from rollout2_attn_readout import AttnReadout                        # noqa: E402
from analysis.intphys2.model import build_from_config                # noqa: E402
from evals.world_model_analysis.data import WMADataset               # noqa: E402
from evals.analysis_vlm.occlusion_identity.forward import extract_batch  # noqa: E402

PY = "/data/hyuntak/anaconda3/envs/vjepa2/bin/python"
PFT_CKPT = ROOT / "z_training/runs/v11_postft/latest.pt"
OUT_ROOT = ROOT / "z_research/predictor_training/predictor_IntPhysGenV11_PFT/exp_results"
V11_META = Path("/local_datasets/world/world_analysis/IntPhysGen_v11/metadata.csv")
V11_CONDS = [("flat", "k=0", "moving_visible_flat", "0"), ("flat", "early k=4", "moving_occlusion_flat_early", "4"), ("flat", "mid k=4", "moving_occlusion_flat_mid", "4"), ("flat", "late k=4", "moving_occlusion_flat", "4"),
             ("ramp", "k=0", "moving_visible", "0"), ("ramp", "early k=4", "moving_occlusion_early", "4"), ("ramp", "mid k=4", "moving_occlusion_mid", "4"), ("ramp", "late k=4", "moving_occlusion", "4")]
S, T, D, RES = rt.S, rt.T, rt.D, rt.RES


def resolved_cfg(dataset, predictor):
    """run.sh 와 같은 병합 (resolve.py). post-FT 면 model.predictor_checkpoint 를 얹는다."""
    f = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False).name
    cmd = [PY, str(ROOT / "z_research/scripts/harness/resolve.py"), "attn_probe", dataset, "vith", "-o", f, "--quiet"]
    if predictor == "pft":
        cmd += ["--set", f"model.predictor_checkpoint={PFT_CKPT}"]
    subprocess.run(cmd, check=True, cwd=ROOT)
    return yaml.safe_load(open(f))


def mass3(a, x, y):
    cx, cy = int(x // 18), int(y // 18); sel = [r * 16 + c for r in range(cy - 1, cy + 2) for c in range(cx - 1, cx + 2) if 0 <= r < 16 and 0 <= c < 16]
    return float(a[sel].sum())


class _DS(torch.utils.data.Dataset):
    def __init__(self, cfg, ids): self.cfg, self.ids, self._ds = cfg, ids, None
    def __len__(self): return len(self.ids)
    def __getitem__(self, j):
        if self._ds is None: self._ds = WMADataset(self.cfg)
        return self._ds.clip(self.ids[j])


def forward_p(cfg, bundle, ids, ruler, dev, bs=8, workers=6):
    """ids (dataset 안 인덱스) → p 에 자를 건 읽기 (n,8,2 정규화 좌표), attention (n,8,256)."""
    dl = torch.utils.data.DataLoader(_DS(cfg, ids), batch_size=bs, num_workers=workers, shuffle=False)
    pred = np.empty((len(ids), T, 2), np.float32); att = np.empty((len(ids), T, S), np.float16); k = 0; t0 = time.time()
    with torch.no_grad():
        for clips in dl:
            out = extract_batch(clips, bundle, [{"base": "predictor"}], context_length=16, mask_index=0, out_dtype=torch.float32)
            p = out["predictor"].to(dev).reshape(-1, S, D); r, a = ruler(p)
            n = clips.size(0); pred[k:k + n] = r.reshape(n, T, 2).cpu().numpy(); att[k:k + n] = a.reshape(n, T, S).half().cpu().numpy(); k += n
            if (k // bs) % 25 == 0: print(f"    {k}/{len(ids)}  {time.time()-t0:.0f}s", flush=True)
    return pred, att


def run_rollout(predictor, bundle, ruler, dev, out, n_per=None):
    cfg = resolved_cfg("rollout_v2", predictor); ds = WMADataset(cfg)
    rows = list(csv.DictReader(open(ROOT / "data_csv/rollout_v2/index_probe.csv")))
    vid2i = {r.video_id: i for i, r in enumerate(ds.records)}
    sel = [r for r in rows if r["plausible"] == "1"]
    if n_per:
        by = {}; [by.setdefault(r["scenario"], []).append(r) for r in sel]; sel = [r for s in rt.SCEN for r in by.get(s, [])[:n_per]]
    ids = [vid2i[r["video_id"]] for r in sel]
    px = np.stack([rt.arr(r["px_x_by_sample"]) for r in sel]); py = np.stack([rt.arr(r["px_y_by_sample"]) for r in sel]); n = len(sel)
    L = np.stack([px.reshape(n, 16, 2).mean(2) / RES - 1, py.reshape(n, 16, 2).mean(2) / RES - 1], -1)[:, T:].astype(np.float32)
    key = "visible_by_sample" if "visible_by_sample" in sel[0] else "in_frame_by_sample"
    inf = (np.stack([rt.arr(r[key]) for r in sel]) > 0).reshape(n, 16, 2).all(2)[:, T:]
    print(f"[rollout_v2 / {predictor}] {n} clip", flush=True)
    pred, att = forward_p(cfg, bundle, ids, ruler, dev)
    scen = np.array([r["scenario"] for r in sel]); rep = {"predictor": predictor, "n": n, "per_scenario": {}}
    f = lambda v: "   -  " if v is None else f"{v:6.2f}"
    lines = ["| scenario | n | R²x | R²y | MAEx px | MAEy px | βx | βy | obj mass s0..s7 (균등 0.035) |", "|---|---|---|---|---|---|---|---|---|"]
    for s in rt.SCEN:
        k = scen == s
        if not k.any(): continue
        m = inf[k]; p = pred[k]; y = L[k]; mv = rt.MOVING[s]
        tr = (y + 1) * RES; masses = [float(np.mean([mass3(att[k][i, t].astype(np.float32), *tr[i, t]) for i in range(k.sum())])) for t in range(T)]
        rec = {"n": int(k.sum()), "r2_x": rt.r2(p[:, :, 0][m], y[:, :, 0][m]) if "x" in mv else None, "r2_y": rt.r2(p[:, :, 1][m], y[:, :, 1][m]) if "y" in mv else None,
               "mae_px_x": float(np.abs(p - y)[:, :, 0][m].mean() * RES), "mae_px_y": float(np.abs(p - y)[:, :, 1][m].mean() * RES),
               "beta_x": rt.beta(rt.slope(p[:, :, 0]), rt.slope(y[:, :, 0])) if "x" in mv else None, "beta_y": rt.beta(rt.slope(p[:, :, 1]), rt.slope(y[:, :, 1])) if "y" in mv else None,
               "obj_mass": masses}
        rep["per_scenario"][s] = rec
        lines.append(f"| {s} | {rec['n']} | {f(rec['r2_x'])} | {f(rec['r2_y'])} | {rec['mae_px_x']:.1f} | {rec['mae_px_y']:.1f} | {f(rec['beta_x'])} | {f(rec['beta_y'])} | " + " ".join(f"{x:.2f}" for x in masses) + " |")
        print(lines[-1], flush=True)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "preds.npz", video_id=np.array([r["video_id"] for r in sel]), pred=pred, truth=L, in_frame=inf, scenario=scen, attn=att)
    (out / "results.json").write_text(json.dumps(rep, indent=1))
    (out / "RESULTS.md").write_text(f"# RollOut_v2 가능 clip — {predictor} predictor 의 p 에 v5 자 (캐시 없이 직접 forward)\n\n1 칸 = 18 px. β = 읽은 변위/진실 변위 회귀 기울기. obj mass = 자 attention 의 진실 물체 3×3 질량.\n\n" + "\n".join(lines) + "\n")


def run_v11(predictor, bundle, ruler, dev, out, n_per=None, violation="vanish"):
    cfg = resolved_cfg("v11_vanish_all" if violation == "vanish" else "v11_full", predictor); ds = WMADataset(cfg)   # v11_vanish_all 은 vanish 블록만 담는다
    rows = list(csv.DictReader(open(Path(cfg["data"]["root"]) / cfg["data"].get("index_csv", "index.csv")))); meta = {r["name"]: r for r in csv.DictReader(V11_META.open())}
    vid2i = {r.video_id: i for i, r in enumerate(ds.records)}
    KEYS = ("obj_mass", "last_mass", "d_truth", "d_last"); res = {}; lines = []; keep = {"video_id": [], "pred": [], "attn": [], "cond": []}
    for motion, tag, cond, K in V11_CONDS:
        pos = [r for r in rows if r["condition"] == cond and r["sym_k"] == K and r["violation_type"] == violation and r["variant"] == "pos_a"]
        if n_per: pos = pos[:n_per]
        if not pos: continue
        print(f"[v11 / {predictor}] {motion} {tag} ({cond}) {len(pos)} clip", flush=True)
        pred, att = forward_p(cfg, bundle, [vid2i[r["video_id"]] for r in pos], ruler, dev)
        keep["video_id"] += [r["video_id"] for r in pos]; keep["pred"].append(pred); keep["attn"].append(att); keep["cond"] += [f"{motion} {tag}"] * len(pos)
        M = {k: [] for k in KEYS}
        for i, r in enumerate(pos):
            m = meta[r["video_id"]]; t32 = np.stack([rt.arr(m["object_px_x_by_sample"]), rt.arr(m["object_px_y_by_sample"])], -1); tub = t32.reshape(16, 2, 2).mean(1); last = tub[7]
            a = att[i].astype(np.float32); pr = (pred[i] + 1) * RES
            M["obj_mass"].append([mass3(a[t], *tub[8 + t]) for t in range(T)]); M["last_mass"].append([mass3(a[t], *last) for t in range(T)])
            M["d_truth"].append(np.linalg.norm(pr - tub[8:], axis=-1)); M["d_last"].append(np.linalg.norm(pr - last, axis=-1))
        name = f"{motion} {tag}"; res[name] = {"n": len(pos), "cond": cond, **{k: [float(v) for v in np.nanmean(M[k], 0)] for k in KEYS}}
        for k in ("obj_mass", "last_mass", "d_truth", "d_last"):
            lines.append(f"| {name} (n={len(pos)}) | {k} | " + " | ".join(f"{x:.2f}" if k.endswith("mass") else f"{x:.0f}" for x in res[name][k]) + " |"); print(lines[-1], flush=True)
    out.mkdir(parents=True, exist_ok=True); (out / "results.json").write_text(json.dumps(res, indent=1))
    np.savez(out / "preds.npz", video_id=np.array(keep["video_id"]), pred=np.concatenate(keep["pred"]), attn=np.concatenate(keep["attn"]), cond=np.array(keep["cond"]))
    (out / "RESULTS.md").write_text(f"# v11 vanish pos_a — {predictor} predictor 의 p 에 v5 자 (직접 forward)\n\n3×3 질량 (균등 0.035), px 는 clip 평균. last = 마지막 관측 위치 (튜블릿 7).\n\n| 조건 | 지표 | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 |\n|---|---|---|---|---|---|---|---|---|---|\n" + "\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--predictor", choices=["release", "pft"], required=True); ap.add_argument("--set", nargs="+", default=["rollout_v2", "v11"]); ap.add_argument("--n-per", type=int, default=None); ap.add_argument("--violation", default="vanish", help="v11 블록 종류: vanish | shape | color (pos_a clip 을 쓴다)"); a = ap.parse_args()
    dev = torch.device("cuda"); ruler = AttnReadout().to(dev); ruler.load_state_dict(torch.load(rt.RES_ROOT / "attentive_pooling/p/attn.pt", map_location=dev)); ruler.eval()
    cfg0 = resolved_cfg("rollout_v2", a.predictor); print("predictor_checkpoint:", cfg0["model"].get("predictor_checkpoint", "(release)"), flush=True)
    bundle = build_from_config(cfg0["model"], dev)
    for s in a.set:
        out = OUT_ROOT / a.predictor / (s if s == "rollout_v2" or a.violation == "vanish" else f"v11_{a.violation}")
        run_rollout(a.predictor, bundle, ruler, dev, out, a.n_per) if s == "rollout_v2" else run_v11(a.predictor, bundle, ruler, dev, out, a.n_per, a.violation)
        print(f"→ {out}", flush=True)


if __name__ == "__main__":
    main()
