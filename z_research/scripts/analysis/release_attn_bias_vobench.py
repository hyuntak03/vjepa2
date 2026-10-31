#!/usr/bin/env python3
"""release predictor attention 개입 — 미래 질의가 과거 물체 자리 (문맥 key) 를 더 보게 하면 VOBench late 가 오르나 (2026-10-30, 사용자 요청).

사용자: "release predictor intervention 실험해보자. 과거 물체 자리 (k) 랑 attn 하는 거 attn score 일부러 높이면 vobench 성능 올라갈 거 같은데".

개입: release predictor 12 층 · 모든 head 의 attention logit 에, 미래 질의 토큰 (위치 2048–4095) → 문맥 key 중 **물체가 보였던 칸**
  (VOBench label.json 의 장별 위치 · 크기, 그 블록 두 장 모두 보일 때만) 에 β 를 더한다 (SDPA attn_mask, 학습 없음).
  ⚠️ 정답 물체 위치를 쓴다 — 방법이 아니라 원인 확인용 개입 (CLAUDE.md predictor 관문 4: 판별 상한 표시).
대조 (위약): 같은 블록 · 같은 개수의 **물체가 아닌** 문맥 칸에 같은 β (씨앗 고정).
채점 = surprise_c16t32 와 같다: 32 장 (원본 0–93, 3 장 간격), 문맥 16 장, mean |p − LN(h)| 미래 2048 토큰, fp32 가중치 + fp16 autocast,
  matched pair (pos_a ↔ imp_ab 는 문맥 A, pos_b ↔ imp_ba 는 문맥 B). 정답 = s(imp) > s(pos). 빈 장면 문맥 (permanence B) 은 물체 칸이 없어 개입이 안 걸린다.

  for g in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$g /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/release_attn_bias_vobench.py run --shard $g --nshard 8 & done; wait
  /venv/vjepa/bin/python z_research/scripts/analysis/release_attn_bias_vobench.py report
출력: z_analysis/results/release_attn_bias/
"""
import argparse, collections, csv, json, os, sys
import numpy as np, torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
ap = argparse.ArgumentParser()
ap.add_argument("phase", choices=["run", "report"])
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--betas", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0])
a = ap.parse_args()
OUT = os.path.join(REPO, "z_analysis/results/release_attn_bias"); os.makedirs(OUT, exist_ok=True)
ROOT = "/workspace/dataset/benchmarks/VOBench_v1"
S, NB, G = 256, 8, 16                      # 블록당 토큰 · 문맥 블록 · 격자
NCTX = NB * S


def object_cells(file_name, arm):
    """문맥 블록 b 마다 물체가 보였던 칸 (두 장 모두 hidden < 0.5). 256 px 격자, 물체 상자 (ext) 를 반 칸 넓혀 칸 중심이 들면."""
    L = json.load(open(os.path.join(ROOT, os.path.dirname(file_name), "label.json")))["labels"].get(arm) or []
    out = []
    for b in range(NB):
        fr = [L[6 * b], L[6 * b + 3]] if len(L) > 6 * b + 3 else []
        cells = set()
        if fr and all(e["hidden"] < 0.5 for e in fr):
            for e in fr:
                u, v = e["u"] / 2, e["v"] / 2; hu, hv = e.get("ext_u", e["r_px"]) / 2, e.get("ext_v", e["r_px"]) / 2
                for r in range(G):
                    for c in range(G):
                        if abs(c * 16 + 8 - u) <= hu + 8 and abs(r * 16 + 8 - v) <= hv + 8:
                            cells.add(b * S + r * G + c)
        out.append(sorted(cells))
    return out


if a.phase == "run":
    from analysis.intphys2.model import build_from_config
    from analysis.intphys2.surprise import _context_target_indices
    from evals.world_model_analysis.data import WMADataset
    dev = torch.device("cuda")
    ck = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
    cfg = {"data": dict(root=os.path.join(REPO, "data_csv/vobench_v1_late"), index_csv="index.csv", frames_root=ROOT,
                        frames_pattern="{file_name}/{frame:06d}.png", frames_start=0, frames_stride=3, block_column="block_id",
                        pair_column="pair_id", variant_column="variant", plausible_column="plausible", type_column="condition",
                        n_frames=32, resolution=256),
           "model": dict(arch_name="vit_huge", checkpoint=ck, img_size=256, patch_size=16, tubelet_size=2, window_size=32, use_rope=True,
                         uniform_power=False, dual_encoder=True, context_encoder_key="encoder", target_encoder_key="target_encoder",
                         predictor=dict(embed_dim=384, depth=12, num_heads=12, num_mask_tokens=10), dtype="float32", autocast="float16")}
    ds = WMADataset(cfg)
    rows = list(csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1_late/index.csv"))))
    assert [r["video_id"] for r in rows] == [r.video_id for r in ds.records]
    byb = collections.defaultdict(dict)
    for i, r in enumerate(rows):
        byb[r["block_id"]][r["variant"]] = i
    blocks = sorted(byb, key=int)[a.shard::a.nshard]
    bundle = build_from_config(cfg["model"], dev)
    ci, ti = _context_target_indices(ctx_frames=16, tgt_frames=16, tubelet_size=2, spatial_tokens=S, batch_size=1, device=dev)
    BIAS = {"m": None}                                  # 모든 predictor 층에 같은 attn_mask (None = 원래 동작)
    for blk in bundle.predictor.predictor_blocks:
        f0 = blk.forward
        def fw(x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None, _f=f0):
            return _f(x, mask=mask, attn_mask=BIAS["m"] if BIAS["m"] is not None else attn_mask, T=T, H_patches=H_patches, W_patches=W_patches)
        blk.forward = fw
    N = 2 * NCTX
    def bias_for(cells, beta):
        m = torch.zeros(1, 1, N, N, device=dev, dtype=torch.float16)
        idx = torch.tensor([c for cb in cells for c in cb], device=dev, dtype=torch.long)
        if len(idx) and beta:
            m[0, 0, NCTX:, idx] = beta
        return m
    rng = np.random.default_rng(0)
    rec = []
    for bid in blocks:
        vv = byb[bid]
        for ctx_arm, pos, imp in (("A", "pos_a", "imp_ab"), ("B", "pos_b", "imp_ba")):
            if pos not in vv or imp not in vv:
                continue
            r_pos = rows[vv[pos]]
            cells = object_cells(r_pos["file_name"], ctx_arm)
            plac = []                                       # 같은 블록 · 같은 개수, 물체 아닌 칸
            for b, cb in enumerate(cells):
                pool = sorted(set(range(b * S, (b + 1) * S)) - set(cb))
                plac.append(sorted(rng.choice(pool, len(cb), replace=False).tolist()) if cb else [])
            x = torch.stack([ds.clip(vv[pos]), ds.clip(vv[imp])]).to(dev, bundle.dtype)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                z = bundle.context_encoder(x[:1], masks=[ci])
                h = bundle.target_encoder(x)
                h = torch.nn.functional.layer_norm(torch.gather(h, 1, ti.expand(2, -1).unsqueeze(-1).expand(-1, -1, h.size(-1))), (h.size(-1),))
                ctx_same = float((bundle.context_encoder(x[1:], masks=[ci]) - z).abs().max())
                res = {}
                for name, cl, beta in [("base", cells, 0.0)] + [(f"obj_b{bt:g}", cells, bt) for bt in a.betas] + [(f"plac_b{bt:g}", plac, bt) for bt in a.betas]:
                    BIAS["m"] = bias_for(cl, beta) if beta else None
                    p = bundle.predictor(z, ci, ti, mask_index=0).float()
                    BIAS["m"] = None
                    sp, si = float((p - h[0:1].float()).abs().mean()), float((p - h[1:2].float()).abs().mean())
                    res[name] = (sp, si)
            rec.append(dict(block_id=bid, ctx=ctx_arm, violation=r_pos["violation"], motion=r_pos["motion"], objset=r_pos["objset"],
                            k=int(r_pos["k"]), n_obj_cells=sum(len(c) for c in cells), ctx_same=ctx_same, res=res))
        print(f"[shard {a.shard}] block {bid} 끝 ({len(rec)} 쌍)", flush=True)
    json.dump(rec, open(os.path.join(OUT, f"shard{a.shard}.json"), "w"))
    print("끝")

else:
    import glob
    R = [r for f in sorted(glob.glob(os.path.join(OUT, "shard*.json"))) for r in json.load(open(f))]
    print(f"쌍 {len(R)} · 문맥 일치 최대 차이 {max(r['ctx_same'] for r in R):.2e} · 물체 칸이 있는 쌍 {sum(r['n_obj_cells'] > 0 for r in R)}")
    names = list(R[0]["res"])
    acc = lambda rr, n: 100 * np.mean([r["res"][n][1] > r["res"][n][0] for r in rr]) if rr else float("nan")
    groups = [("전체", lambda r: True), ("A (문맥에 물체)", lambda r: r["ctx"] == "A"), ("B (문맥에 물체 · 빈 장면 제외)", lambda r: r["ctx"] == "B" and r["n_obj_cells"] > 0)] + \
             [(f"{v} · {d}", (lambda v, d: lambda r: r["violation"] == v and r["ctx"] == d)(v, d)) for v in ("permanence", "shape", "color") for d in "AB"] + \
             [("A · 정지", lambda r: r["ctx"] == "A" and r["motion"] == "flat_static"), ("A · 움직임", lambda r: r["ctx"] == "A" and r["motion"] != "flat_static")]
    out = {}
    print(f"{'묶음':28s} {'n':>4s} " + " ".join(f"{n:>9s}" for n in names))
    for gname, f in groups:
        rr = [r for r in R if f(r)]
        out[gname] = {n: acc(rr, n) for n in names}
        print(f"{gname:28s} {len(rr):4d} " + " ".join(f"{acc(rr, n):9.1f}" for n in names))
    json.dump(out, open(os.path.join(OUT, "summary.json"), "w"), ensure_ascii=False, indent=1)
