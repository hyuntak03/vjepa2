"""VOBench (primitive) 를 VOB_readout 자로 읽는다 — AR predictor 의 미래 8 블록에 물체가 어디 · 무엇으로 있나 (2026-10-12).

추출 (자 학습과 같은 경로: rollout2_presence_readout._worker 의 AR 분기 — bf16 모델 · 블록별 LN(target_encoder) · predictor.rollout bf16 autocast):
  읽은 값 셋, 미래 튜블릿 t8–t15 마다 (xy · 57 확률 · attention 최대 질량)
    p@p = p 로 배운 자 → AR p      (정체가 위치와 무관하게 남아 있나)
    p@h = h 로 배운 자 → AR p      (AR 이 물체를 **참 세계 좌표에서** 어디 두나 — p 자의 보정이 없다)
    h@h = h 로 배운 자 → 참 미래 h  (천장 · 자 검사) · hh16 = 같은 자를 문맥 포함 16 블록 h 에 (t0–t15)
분석: 가능 영상 (pos_a · pos_b) 만 — p 는 matched 쌍에서 같다. 정답 = 그 영상 미래의 물체 (label.json, 문맥 arm 과 같다).
  위치 오차 (칸 = 32 px @ 512) · 운동 방향 부호 오차 (+ 앞섬) · 등속 외삽 / 마지막 관측 자리 기준선 · 정체 (49) · 모양 · 색 · P(없음).

  python z_research/scripts/analysis/vobench_ar_readout.py extract --gpus 4 [--name ar40 --ckpt z_training/ariel/ar_future_1e_5/latest.pt]
  python z_research/scripts/analysis/vobench_ar_readout.py analyze [--name ar40]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
FR = Path("/data2/local_datasets/world/Benchmark/VOBench_v1")
INDEX = REPO / "data_csv/vobench_v1/index.csv"
RES_CFG = REPO / "z_research/VOBench_v1/exp_results/surprise_c16t32__vobench_v1_vith/_resolved.yaml"
S, T, D, NONE = 256, 8, 1280, 56
SHAPES = ["capsule", "cone", "cube", "cylinder", "pyramid", "sphere", "torus"]
COLORS = ["blue", "cyan", "green", "magenta", "orange", "purple", "red", "yellow"]
CELL = 32.0                                   # px @ 512 = 토큰 1 개 = 자 학습의 1 칸 (18 px @ 288)


def out_dir(name):
    return REPO / f"z_research/VOBench_v1/readout/vobench_{name}"


def rows_primitive():
    return [r for r in csv.DictReader(INDEX.open()) if r["objset"] == "primitive"]


def _worker(rank, world, name, ckpt):
    import torch, yaml
    from analysis.intphys2.model import build_from_config
    from app.vjepa_frozen.ar import block_indices, encode_blocks
    from evals.world_model_analysis.data import WMADataset
    from importlib.util import module_from_spec, spec_from_file_location
    torch.cuda.set_device(rank); torch.set_num_threads(4); dev = torch.device("cuda", rank)
    spec = spec_from_file_location("ir", REPO / "z_research/scripts/analysis/rollout2_identity_readout.py")
    os.environ.setdefault("PRESENCE_SET", "vob_readout_primitive")
    ir = module_from_spec(spec); spec.loader.exec_module(ir)
    model = dict(ir.rp.MODEL)
    if ckpt != "release":
        model["predictor_checkpoint"] = str(REPO / ckpt)
    bundle = build_from_config(model, dev)
    from analysis.predictors import is_rollout
    AR = is_rollout(bundle.predictor)                # False = 릴리즈식 (mask token) predictor — 자 학습 추출과 같은 경로 (extract_batch)
    if not AR:
        from evals.analysis_vlm.occlusion_identity.forward import extract_batch
    R = {}
    for rep in ("p", "h"):
        m = ir.IdReadout().to(dev).eval()
        m.load_state_dict(torch.load(REPO / f"z_research/VOBench_v1/readout/identity_vobprim_{name}/{rep}/readout.pt", map_location=dev))
        R[rep] = m
    cfg = yaml.safe_load(open(RES_CFG))
    data = dict(cfg["data"], index_csv="index.csv")
    ds = WMADataset(dict(data=data, model=model, features={"cache_dir": "/tmp"}, surprise={}))
    want = {r["video_id"] for r in rows_primitive()}
    ids = [i for i, r in enumerate(ds.records) if r.video_id in want]
    mine = ids[rank * len(ids) // world:(rank + 1) * len(ids) // world]
    out = defaultdict(list); t0 = time.time()

    def read(m, tok):                         # tok (B, n, S, D) → xy (B,n,2), prob (B,n,57), amax (B,n)
        B, n = tok.shape[:2]; xy, lg, a = m(tok.reshape(B * n, S, D).half())
        return xy.view(B, n, 2), torch.softmax(lg, -1).view(B, n, -1), a.max(-1).values.view(B, n)
    with torch.no_grad():
        for k in range(0, len(mine), 16):
            ch = mine[k:k + 16]
            clips = torch.stack([ds.clip(i) for i in ch]).to(dev, dtype=bundle.dtype)
            B = clips.size(0)
            if AR:
                hb = encode_blocks(bundle.target_encoder, clips, 2)
                idx_ = block_indices(B, 16, S, dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pr = bundle.predictor.rollout(hb[:, :T * S], idx_[:, :T * S], T)
                p = pr.float().reshape(B, T, S, D).half()
                h16 = hb.float().reshape(B, 2 * T, S, D).half()         # 문맥 포함 16 블록 (hh16 = h 자 → t0–t15)
            else:                                                    # 릴리즈식: p = 문맥 16 장 → 미래 8 튜블릿, h = LN(target_encoder(32 장))
                ex = extract_batch(clips, bundle, [{"base": "predictor"}], context_length=16, mask_index=0, out_dtype=torch.float16)
                p = ex["predictor"].reshape(B, T, S, D).to(dev).half()
                hf = bundle.target_encoder(clips); hf = hf[-1] if isinstance(hf, list) else hf
                h16 = torch.nn.functional.layer_norm(hf.float(), (hf.size(-1),)).reshape(B, 2 * T, S, D).half()
            h = h16[:, T:]
            for tag, m, tok in (("pp", R["p"], p), ("ph", R["h"], p), ("hh", R["h"], h), ("hh16", R["h"], h16)):
                xy, pb, am = read(m, tok)
                out[f"{tag}_xy"].append(xy.float().cpu().numpy()); out[f"{tag}_prob"].append(pb.float().cpu().numpy())
                out[f"{tag}_amax"].append(am.float().cpu().numpy())
            out["video_id"] += [ds.records[i].video_id for i in ch]
            if rank == 0 and k % 160 == 0:
                print(f"[r0] {k + B}/{len(mine)} ({time.time() - t0:.0f}s)", flush=True)
    d = out_dir(name); d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"shard{rank}.npz", video_id=np.array(out.pop("video_id")), **{k: np.concatenate(v) for k, v in out.items()})
    print(f"[r{rank}] done {len(mine)} ({time.time() - t0:.0f}s)", flush=True)


def extract(gpus, name, ckpt):
    import torch
    torch.multiprocessing.spawn(_worker, args=(gpus, name, ckpt), nprocs=gpus, join=True)


def load(name):
    d = out_dir(name); Z = defaultdict(list)
    for f in sorted(d.glob("shard*.npz")):
        z = np.load(f)
        for k in z.files: Z[k].append(z[k])
    return {k: np.concatenate(v) for k, v in Z.items()}


def to512(xy):
    return (xy + 1) * 144 * 512 / 288         # 자 좌표 (288 기준 정규화) → px @ 512


def analyze(name):
    Z = load(name); vid = {v: i for i, v in enumerate(Z["video_id"])}
    rows = {r["video_id"]: r for r in rows_primitive()}
    lab = {}
    acc = defaultdict(lambda: defaultdict(list))
    for v, r in rows.items():
        if r["variant"] not in ("pos_a", "pos_b") or v not in vid: continue
        arm = "A" if r["variant"] == "pos_a" else "B"
        bdir = FR / Path(r["file_name"]).parent
        if bdir not in lab: lab[bdir] = json.load(open(bdir / "label.json"))["labels"]
        L = lab[bdir].get(arm)
        obj = r["obj_a"] if arm == "A" else r["obj_b"]
        empty = obj == "empty" or not L or L[0] is None
        i = vid[v]
        mot = "static" if r["motion"] == "flat_static" else ("const_vel" if r["motion"] == "flat_vel" else
              ("speeding" if r["motion"] in ("flat_acc", "ramp_down") else "slowing"))
        key = (r["occlusion"], mot)
        if empty:
            for tag in ("pp", "ph", "hh"):
                for j in range(T): acc[key + ("empty",)][f"{tag}_pnone_{j}"].append(Z[f"{tag}_prob"][i, j, NONE])
            continue
        sh, co = obj.split("/"); cls = SHAPES.index(sh) * len(COLORS) + COLORS.index(co)
        P = np.array([[L[3 * s]["u"], L[3 * s]["v"]] for s in range(32)], float)
        hid = np.array([L[3 * s]["hidden"] for s in range(32)], bool)
        vis_ctx = [s for s in range(16) if not hid[s]]
        last = vis_ctx[-1] if vis_ctx else 15
        vel = (P[last] - P[max(vis_ctx[0], last - 2)]) / max(last - max(vis_ctx[0], last - 2), 1) if len(vis_ctx) > 1 else np.zeros(2)
        for j in range(T):
            s0, s1 = 16 + 2 * j, 17 + 2 * j
            truth = (P[s0] + P[s1]) / 2; cv = P[last] + vel * ((s0 + s1) / 2 - last)
            ax = truth - P[last]; n = np.linalg.norm(ax); ax = ax / n if n > 1e-6 else ax
            k = key + ("obj",)
            acc[k][f"travel_{j}"].append(n / CELL)
            acc[k][f"constvel_err_{j}"].append(np.linalg.norm(cv - truth) / CELL)
            acc[k][f"stay_err_{j}"].append(n / CELL)
            acc[k][f"hidden_{j}"].append(float(hid[s0] and hid[s1]))
            for tag in ("pp", "ph", "hh"):
                xy = to512(Z[f"{tag}_xy"][i, j]); pb = Z[f"{tag}_prob"][i, j]
                acc[k][f"{tag}_err_{j}"].append(np.linalg.norm(xy - truth) / CELL)
                acc[k][f"{tag}_along_{j}"].append(((xy - truth) @ ax) / CELL if n > 1e-6 else np.nan)
                acc[k][f"{tag}_id_{j}"].append(float(pb[:NONE].argmax() == cls))
                m = pb[:NONE].reshape(len(SHAPES), len(COLORS))
                acc[k][f"{tag}_shape_{j}"].append(float(m.sum(1).argmax() == SHAPES.index(sh)))
                acc[k][f"{tag}_color_{j}"].append(float(m.sum(0).argmax() == COLORS.index(co)))
                acc[k][f"{tag}_pnone_{j}"].append(pb[NONE])
                acc[k][f"{tag}_amax_{j}"].append(Z[f"{tag}_amax"][i, j])
    res = {"/".join(k): {m: float(np.nanmean(v)) for m, v in d.items()} | {"n": len(next(iter(d.values())))} for k, d in acc.items()}
    json.dump(res, open(out_dir(name) / "readout.json", "w"), indent=1)
    return res


def report(res, name):
    lines = []
    occs = ("visible", "early", "mid", "late"); mots = ("static", "const_vel", "speeding", "slowing")
    f = lambda k, m: " ".join(f"{res[k].get(f'{m}_{j}', np.nan):6.2f}" for j in range(T))
    for occ in occs:
        for mot in mots:
            k = f"{occ}/{mot}/obj"
            if k not in res: continue
            lines += [f"\n## {occ} · {mot} (영상 {res[k]['n']})", "  걸음            " + " ".join(f"{'t' + str(8 + j):>6s}" for j in range(T))]
            for lab_, m in (("이동 거리 (칸)", "travel"), ("가림 비율", "hidden"),
                            ("p@p 위치 오차", "pp_err"), ("p@p 앞섬(+)/뒤짐", "pp_along"), ("p@p attn 최대", "pp_amax"),
                            ("p@h 위치 오차", "ph_err"), ("p@h 앞섬(+)/뒤짐", "ph_along"), ("h@h 위치 오차", "hh_err"), ("h@h attn 최대", "hh_amax"),
                            ("등속 외삽 오차", "constvel_err"), ("제자리 오차", "stay_err"),
                            ("p@p 정체", "pp_id"), ("p@h 정체", "ph_id"), ("h@h 정체", "hh_id"),
                            ("p@p 모양", "pp_shape"), ("p@p 색", "pp_color"), ("p@p P(없음)", "pp_pnone"), ("p@h P(없음)", "ph_pnone")):
                lines.append(f"  {lab_:14s}  " + f(k, m))
    text = "\n".join(lines); print(text)
    (out_dir(name) / "readout.txt").write_text(text + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["extract", "analyze"])
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--name", default="ar40")
    ap.add_argument("--ckpt", default="z_training/ariel/ar_future_1e_5/latest.pt")
    a = ap.parse_args()
    if a.phase == "extract":
        extract(a.gpus, a.name, a.ckpt)
    else:
        report(analyze(a.name), a.name)
