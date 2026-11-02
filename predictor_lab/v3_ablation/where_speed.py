#!/usr/bin/env python3
"""가르는 실험 두 개 (2026-11-01, 사용자 요청) — 예측기는 움직이는 물체의 재등장 자리를 "속도로 계산" 하나, "학습 데이터의 자리를 외워" 두나.

① 자리 확인: 예측이 물체를 어디에 두나 (VOBench late 움직이는 사라짐 블록 · Blender trapdoor 시험 움직이는 장면)
② 속도 묶음: 같은 영상에서 프레임 간격만 바꿔 (문맥 끝 프레임은 고정) 화면 속도를 줄이면 예측 자리가 따라가나
   VOBench: 간격 3 (원판, 속도 1) · 간격 2 (2/3) · 간격 1 (1/3) — 문맥 끝 = 원본 45 (transfer_gap_speed_sweep 와 같은 판)
   Blender trapdoor: 간격 2 (원판, 속도 1) · 간격 1 (1/2) — 문맥 끝 = 원본 30 (샘플 15)
측정 (학습된 판독기 없음, transfer_gap_endprior E35 와 같은 정의):
  존재 대비 지도 E = ‖x(물체 있는 문맥) − x(물체 없는 문맥)‖₂ (토큰마다, 같은 장면의 pos_a · pos_b).
  x = 예측 (미래 튜블릿 8 개) 또는 목표 encoder (참). 자리 = 3×3 상자 합이 가장 큰 칸 (토큰 단위, 행 · 열 1–14).
  속도 따라감 비 = (예측 자리[빠름] − 예측 자리[느림]) · u / |참 자리[빠름] − 참 자리[느림]|, u = 참 차이 방향, 마지막 미래 튜블릿.
    1 = 속도를 따라 계산, 0 = 자리가 속도와 상관없이 고정. 참 차이 1.5 토큰 이상인 블록만. 중앙값 + 블록 bootstrap 95 % CI.
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P predictor_lab/v3_ablation/where_speed.py extract [세트 …]   (GPU 1)
  $P predictor_lab/v3_ablation/where_speed.py report             (CPU → predictor_lab/results/v3_ablation/where_speed.json)
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
import transfer_gap_r4_endprior as r4  # noqa: E402

ep = r4.ep
R = REPO / "z_training/runs"
OUT = REPO / "predictor_lab/results/v3_ablation/where_speed"
PREDS = [("release", R / "release_vith/latest.pt"), ("vobpft", R / "vobench_release_postft_g/latest.pt"),
         ("v3mix", R / "v3_mix1600_s0/latest.pt"), ("t25", R / "v3_mix1600_t25_s0/latest.pt"),
         ("tonly", R / "v3t_only_s0/latest.pt"), ("voblike", R / "v3t_voblike_s0/latest.pt")]
V3T = [f"data.root={REPO}/data_csv/blender_track_v3t/voe_test", "data.index_csv=index.csv", "data.frames_root=null"]
SETS = {
    "vob_s3": ("vobench_v1_split2_test", r4.VOB),
    "vob_s2": ("vobench_v1_split2_test", r4.VOB + ["data.frames_start=15", "data.frames_stride=2"]),
    "vob_s1": ("vobench_v1_split2_test", r4.VOB + ["data.frames_start=30", "data.frames_stride=1"]),
    "v3t_s2": ("vobench_v1", V3T + ["window.stride=2", "window.frames_start=0"]),
    # mp4 경로는 시작 프레임을 index 열에서만 읽는다 (data.frames_start 는 PNG 경로 전용) → 열 fstart15 = 15 를 더한 사본
    "v3t_s1": ("vobench_v1", [f"data.root={REPO}/data_csv/blender_track_v3t/voe_test", "data.index_csv=index_start15.csv", "data.frames_root=null",
                              "data.frames_start_column=fstart15", "window.stride=1"]),
}
SWEEPS = {"vob": [("vob_s1", "1/3"), ("vob_s2", "2/3"), ("vob_s3", "1")], "v3t": [("v3t_s1", "1/2"), ("v3t_s2", "1")]}
# 2026-11-02: AR (vast ar_state) 도 같은 측정 — WS_MODE=ar 이면 모델 · 저장 폴더만 바뀐다. AR 은 블록별 LN(target) 문맥 → rollout (예측 공간이 블록 공간)
if os.environ.get("WS_MODE") == "ar":
    PREDS = [("vastar", REPO / "z_training/vast/ar_state/latest.pt"), ("vastar_vobpft", R / "vobench_vast_ar_state_postft_g/latest.pt")]
    OUT = REPO / "predictor_lab/results/v3_ablation/where_speed_ar"
# 2026-11-02: v4_velocity_control (같은 장면 · 다른 속도) 학습 팔 — WS_MODE=v4vc 이면 새 팔 + 기준 셋, 저장 폴더 where_speed_v4vc
if os.environ.get("WS_MODE") == "v4vc":
    PREDS = [("release", R / "release_vith/latest.pt"), ("vobpft", R / "vobench_release_postft_g/latest.pt"), ("v3full", R / "v3_full_s0/latest.pt")] \
        + [(n.replace("v4vc_", "").replace("_s0", ""), R / n / "latest.pt") for n in ("v4vc_fam_s0", "v4vc_ind_s0", "v4vc_fam6k_s0", "v4vc_ind6k_s0", "v4vc_all_s0")
           if (R / n / "latest.pt").exists()]
    OUT = REPO / "predictor_lab/results/v3_ablation/where_speed_v4vc"
_orig_pick = ep.pick_blocks


def pick_blocks(W, name):
    if name.startswith("vob_"):
        return _orig_pick(W, name)                     # VOBench: 움직이는 (flat_static 제외) late permanence 블록
    by = defaultdict(dict)
    for i, r in enumerate(W.records):
        by[r.block_id][r.raw.get("variant")] = i
    out = []
    for b, v in by.items():
        if "pos_a" not in v or "pos_b" not in v:
            continue
        raw = W.records[v["pos_a"]].raw
        if raw.get("target_mode") != "moving":
            continue
        out.append({"block": b, "a": v["pos_a"], "b": v["pos_b"], "occ": "late", "motion": f"{raw.get('place')}_{raw.get('ramp_mode') or '-'}",
                    "camera": raw.get("camera"), "objset": raw.get("objset"), "event_s": raw.get("event_s"), "event_e": raw.get("event_e")})
    return sorted(out, key=lambda x: str(x["block"]))


def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    ep.PREDS = [(t, str(p)) for t, p in PREDS]
    ep.SETS = SETS
    ep.ARR = OUT
    ep.pick_blocks = pick_blocks
    import transfer_gap_te as tte
    _res = tte.resolve

    def resolve(dataset, model, sets, window, limit):
        if dataset == "vobench_v1_split2_test" and not sets:
            sets = r4.VOB
        return _res(dataset, model, sets, window, limit)
    tte.resolve = resolve


def locs(E):
    """E (n, T, 256) → (n, T, 2) 자리 (행, 열) · 두드러짐 z (n, T)."""
    r, c, share, zz = ep.loc(E)
    return np.stack([r, c], -1).astype(float), zz


def report():
    res = {"def": __doc__.split("측정")[1].split("P=")[0].strip(), "sweeps": {}}
    for sw, items in SWEEPS.items():
        D = {name: ep.load(name) for name, _ in items}
        if any(v is None for v in D.values()):
            print("빠진 세트", sw, [k for k, v in D.items() if v is None]); continue
        base = items[-1][0]
        va = [b["vid_a"] for b in D[base][1]["blocks"]]
        for name, _ in items:
            assert [b["vid_a"] for b in D[name][1]["blocks"]] == va, name
        tags = D[base][1]["tags"]
        P, Tr, Zp, last = {}, {}, {}, {}
        for name, lab in items:
            Z, M = D[name]
            nc, nf = M["nc"], M["nf"]
            th, zh = locs(Z["Eh"])
            Tr[name] = th[:, nc:]                                        # 참 자리 (n, nf, 2)
            # 마지막으로 보인 자리: 문맥 튜블릿 중 참 지도가 두드러진 (z ≥ 3) 마지막 칸
            lz = np.where(zh[:, :nc] >= 3, np.arange(nc)[None], -1).max(1)
            last[name] = np.stack([th[i, max(k, 0)] for i, k in enumerate(lz)])
            for t in tags:
                P[(name, t)], Zp[(name, t)] = locs(Z[f"E_{t}"])
        n = len(va)
        out = {"n_block": n, "speeds": [lab for _, lab in items], "tags": tags, "where": {}, "follow": {}}
        # ① 자리 확인 — 원판 속도에서 마지막 미래 튜블릿
        for t in tags:
            p, tr, ls = P[(base, t)][:, -1], Tr[base][:, -1], last[base]
            d_true = np.linalg.norm(p - tr, axis=1); d_last = np.linalg.norm(p - ls, axis=1)
            out["where"][t] = {"within1_true": round(float((d_true <= 1.5).mean()), 3),
                               "nearer_true_than_last": round(float((d_true < d_last).mean()), 3),
                               "median_dist_true": round(float(np.median(d_true)), 2),
                               "median_dist_last": round(float(np.median(d_last)), 2),
                               "median_z": round(float(np.median(Zp[(base, t)][:, -1])), 2),
                               "pred_moved_from_last": round(float(np.median(np.linalg.norm(p - ls, axis=1))), 2),
                               "true_moved_from_last": round(float(np.median(np.linalg.norm(tr - ls, axis=1))), 2)}
        # ② 속도 따라감 — 가장 느린 판 vs 원판
        slow, fast = items[0][0], base
        dv = Tr[fast][:, -1] - Tr[slow][:, -1]
        mag = np.linalg.norm(dv, axis=1)
        keep = mag >= 1.5
        u = dv[keep] / mag[keep, None]
        rng = np.random.default_rng(0)
        k = int(keep.sum())
        Bs = rng.integers(0, k, (2000, k)) if k else None
        out["n_used_follow"] = k
        out["true_shift_median"] = round(float(np.median(mag[keep])), 2) if k else None
        for t in tags:
            dp = (P[(fast, t)][:, -1] - P[(slow, t)][:, -1])[keep]
            ratio = (dp * u).sum(1) / mag[keep]
            bs = np.median(ratio[Bs], 1)
            out["follow"][t] = {"median": round(float(np.median(ratio)), 3),
                                "ci": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
                                "pred_shift_median": round(float(np.median(np.linalg.norm((P[(fast, t)][:, -1] - P[(slow, t)][:, -1])[keep], axis=1))), 2)}
        res["sweeps"][sw] = out
    print(json.dumps(res["sweeps"], indent=1, ensure_ascii=False))
    json.dump(res, open(OUT.parent / (OUT.name + ".json"), "w"), indent=1, ensure_ascii=False)


def extract_any(sets):
    """ep.extract 와 같은 산출 (E 지도) — predictor 가 AR 이면 sc.predict_ar (블록 공간) 로."""
    import time
    import torch
    import te_v11_score as te
    from transfer_gap_te import resolve
    dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
    # vast 체크포인트는 상태 되먹임을 kind: ar + ar_feedback: state 로 적는다 → te.load_pred 는 kind 만 봐서 ar 로 짓다 실패한다.
    # 공용 엔진은 그대로 두고, 여기서만 kind 를 analysis.predictors.kind_from_arch 로 읽는 판으로 바꿔 끼운다 (나머지는 te.load_pred 와 같다).
    from analysis import predictors as PV

    def load_pred_kind(bundle, cfg, path):
        st = te._torch_load(path)
        arch = dict(st.get("arch") or {})
        kind = PV.kind_from_arch(arch) or "default"
        kk = {k: arch[k] for k in PV.KIND_ONLY_KEYS if k in arch}
        m = cfg["model"]; pc = m.get("predictor") or {}
        pred = te._build_predictor(
            img_size=bundle.img_size, patch_size=bundle.patch_size, tubelet_size=bundle.tubelet_size,
            num_frames=bundle.num_frames, encoder_embed_dim=bundle.embed_dim,
            predictor_embed_dim=int(pc.get("embed_dim", 384)), predictor_depth=int(pc.get("depth", 12)),
            predictor_num_heads=int(pc.get("num_heads", 12)), num_mask_tokens=int(pc.get("num_mask_tokens", 10)),
            use_rope=bool(m.get("use_rope", True)), uniform_power=bool(m.get("uniform_power", False)), kind=kind, kind_kwargs=kk)
        pred.load_state_dict(te._clean_backbone_key(st["predictor"]), strict=True)
        pred = pred.to(device=bundle.device, dtype=bundle.dtype).eval()
        for q in pred.parameters():
            q.requires_grad_(False)
        print(f"  load {path}: kind {kind}", flush=True)
        return pred, kind
    te.load_pred = load_pred_kind
    cfg0, _ = resolve("vobench_v1_split2_test", "vith", r4.VOB, None, 0)
    sc = te.Scorer(cfg0, dev, ep.PREDS)
    print("  is_ar", sc.is_ar, flush=True)
    tags, D, S = sc.tags, sc.D, sc.S
    for name in sets:
        ds, st = SETS[name]
        cfg, _ = resolve(ds, "vith", st, None, 0)
        W = te.WMADataset(cfg)
        blocks = pick_blocks(W, name)
        n, nc, nf = len(blocks), sc.nc, sc.nf
        Eh = np.zeros((n, nc + nf, S), np.float32)
        Ep = {t: np.zeros((n, nf, S), np.float32) for t in tags}
        t0 = time.time()
        for b0 in range(0, n, 4):
            bl = blocks[b0:b0 + 4]
            clips = torch.stack([W.clip(x[y]) for x in bl for y in ("a", "b")])
            with torch.inference_mode():
                z, h, ci, ti, hb = sc.encode(clips)
                B = z.size(0)
                dn = lambda x: (x[0::2] - x[1::2]).norm(dim=-1).cpu().numpy()  # noqa: E731
                Eh[b0:b0 + len(bl)] = dn(h.float().reshape(B, nc + nf, S, D))
                for k, t in enumerate(tags):
                    p = sc.predict_ar(sc.mods[k], hb) if sc.is_ar[k] else sc.predict(sc.mods[k], z, ci, ti)
                    Ep[t][b0:b0 + len(bl)] = dn(p.float().reshape(B, nf, S, D))
        meta = [{k: v for k, v in x.items() if k not in ("a", "b")} | {"vid_a": W.records[x["a"]].video_id, "vid_b": W.records[x["b"]].video_id} for x in blocks]
        np.savez(OUT / f"{name}.npz", Eh=Eh, **{f"E_{t}": v for t, v in Ep.items()})
        json.dump({"blocks": meta, "nc": nc, "nf": nf, "tags": tags, "dataset": ds, "sets": st}, open(OUT / f"{name}_meta.json", "w"), ensure_ascii=False)
        print(f"[{name}] done {n} blocks {time.time() - t0:.0f}s", flush=True)


def main():
    part = sys.argv[1] if len(sys.argv) > 1 else "report"
    setup()
    if part == "extract":
        sets = sys.argv[2:] or list(SETS)
        if os.environ.get("WS_MODE") == "ar":
            extract_any(sets)
        else:
            ep.extract(sets, 0)
    report()


if __name__ == "__main__":
    main()
