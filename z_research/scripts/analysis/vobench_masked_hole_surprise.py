"""VOBench 채점을 '미래 물체 구멍 채우기' 로 — 문맥 16 장 전부 + 미래에서 물체 중심 4x4 만 뺀 32 장 → 릴리즈 predictor 가 구멍을 채운다 (2026-10-29).

matched pair (pos_a, imp_ab) · (pos_b, imp_ba): 두 클립에 **같은 구멍** (미래 튜블릿마다 문맥 쪽 arm 물체 중심, 그 arm 에 물체가 없으면 (영속 B) 다른 arm).
점수 (클립마다, 정답 = LN(target_encoder(32 장)), surprise 관례 float32 + autocast fp16, mask_index 0):
  hole      = 구멍 채우기 predictor 출력 vs 정답, 구멍 토큰만
  std_all   = 표준 채점 (문맥 16 장 → 미래 8 튜블릿 전부), 미래 전부
  std_hole  = 표준 채점 출력을 구멍 토큰에서만
  copy_hole · copy_all = 문맥 마지막 튜블릿을 미래로 복사 (같은 자리 토큰)
정답 = surprise(imp) > surprise(pos).  ⚠️ 구멍 자리가 미래 물체 위치를 알려주고, 구멍 밖 미래 토큰 (그림자 등) 이 encoder 에 들어간다.

  for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g $PY z_research/scripts/analysis/vobench_masked_hole_surprise.py run --shard $g --nshard 4 & done; wait
  $PY z_research/scripts/analysis/vobench_masked_hole_surprise.py report
"""
import argparse, csv, json, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
FR = Path("/local_datasets/world/testbed/VOBench_v1")
OUT = REPO / "z_analysis/masked_hole_surprise"
T, C, S, G, HOLE = 32, 16, 256, 16, 4
KEYS = ("hole", "std_all", "std_hole", "copy_hole", "copy_all")


def pairs():
    R = list(csv.DictReader(open(REPO / "data_csv/vobench_v1/index.csv")))
    by = defaultdict(dict)
    for r in R:
        by[r["source_block"]][r["variant"]] = r
    out = []
    for b, v in sorted(by.items()):
        out.append((v["pos_a"], v["imp_ab"], "A"))
        out.append((v["pos_b"], v["imp_ba"], "B"))
    return out


def hole_idx(r, arm):
    lab = json.load(open((FR / r["file_name"]).parent / "label.json"))["labels"]
    L = lab[arm] if lab.get(arm) else lab["B" if arm == "A" else "A"]
    idx = []
    for t in range(C // 2, T // 2):
        gi, gj = L[3 * 2 * t]["v"] / 512 * G, L[3 * 2 * t]["u"] / 512 * G
        i0, j0 = int(np.clip(round(gi - HOLE / 2), 0, G - HOLE)), int(np.clip(round(gj - HOLE / 2), 0, G - HOLE))
        idx += [t * S + (i0 + a) * G + j0 + b for a in range(HOLE) for b in range(HOLE)]
    return idx


def run(shard, nshard):
    import torch, torch.nn.functional as F
    from PIL import Image
    sys.path.insert(0, str(REPO))
    from analysis.intphys2.model import build_from_config
    from analysis.intphys2.surprise import _context_target_indices
    dev = torch.device("cuda")
    CK = REPO / "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth"
    m = {"arch_name": "vit_huge", "checkpoint": str(CK), "img_size": 256, "patch_size": 16, "tubelet_size": 2, "use_rope": True, "uniform_power": False,
         "dual_encoder": True, "context_encoder_key": "encoder", "target_encoder_key": "target_encoder",
         "predictor": {"embed_dim": 384, "depth": 12, "num_heads": 12, "num_mask_tokens": 10}, "dtype": "float32", "autocast": "float16", "window_size": 32}
    B = build_from_config(m, dev)
    MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(3, 1, 1, 1)
    ci0, ti0 = _context_target_indices(ctx_frames=C, tgt_frames=T - C, tubelet_size=2, spatial_tokens=S, batch_size=1, device=dev)
    pool = ThreadPoolExecutor(32)

    def load(r):
        d = FR / r["file_name"]
        fr = list(pool.map(lambda k: np.asarray(Image.open(d / f"{3 * k:06d}.png").convert("RGB")), range(T)))
        x = torch.from_numpy(np.stack(fr)).permute(3, 0, 1, 2).float().to(dev) / 255
        x = F.interpolate(x.transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False, antialias=False).transpose(0, 1)
        return ((x - MEAN) / STD).unsqueeze(0)

    P = pairs()[shard::nshard]; res = []
    for n, (rp, ri, arm) in enumerate(P):
        hole = hole_idx(rp, arm); hs = set(hole)
        ci = torch.tensor([k for k in range(T // 2 * S) if k not in hs], device=dev).unsqueeze(0); ti = torch.tensor(hole, device=dev).unsqueeze(0)
        hf = torch.tensor(hole, device=dev) - C // 2 * S                                             # 미래 안 자리
        rec = {"pos": rp["video_id"], "imp": ri["video_id"], "block": rp["source_block"], "arm": arm, "objset": rp["objset"], "violation": rp["violation"],
               "occlusion": rp["occlusion"], "k": rp["k"], "motion": rp["motion"]}
        for role, r in (("pos", rp), ("imp", ri)):
            x = load(r)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                h = F.layer_norm(B.target_encoder(x).float(), (1280,))[0]
                ph = B.predictor(B.context_encoder(x, masks=[ci]), ci, ti, mask_index=0).float()[0]
                ps = B.predictor(B.context_encoder(x, masks=[ci0]), ci0, ti0, mask_index=0).float()[0]   # (8·256, D)
            hfut = h[C // 2 * S:]; cp = h[(C // 2 - 1) * S:C // 2 * S].repeat(T // 2 - C // 2, 1)
            rec[role] = {"hole": float((ph - h[hole]).abs().mean()), "std_all": float((ps - hfut).abs().mean()), "std_hole": float((ps[hf] - hfut[hf]).abs().mean()),
                         "copy_hole": float((cp[hf] - hfut[hf]).abs().mean()), "copy_all": float((cp - hfut).abs().mean())}
        res.append(rec)
        if n % 50 == 0:
            print(f"shard {shard}: {n}/{len(P)}", flush=True)
    json.dump(res, open(OUT / f"shard{shard}.json", "w"))


def report():
    R = sum([json.load(open(p)) for p in sorted(OUT.glob("shard*.json"))], [])
    acc = lambda rs, k: 100 * np.mean([r["imp"][k] > r["pos"][k] for r in rs])
    lines, summ = [], {"n_pairs": len(R)}
    def row(name, rs):
        v = {k: acc(rs, k) for k in KEYS}; summ[name] = {**v, "n": len(rs),
             "A": {k: acc([r for r in rs if r["arm"] == "A"], k) for k in KEYS}, "B": {k: acc([r for r in rs if r["arm"] == "B"], k) for k in KEYS}}
        lines.append(f"{name:28s} n={len(rs):4d} " + " ".join(f"{k} {v[k]:5.1f} (A {summ[name]['A'][k]:5.1f}/B {summ[name]['B'][k]:5.1f})" for k in KEYS))
    row("overall", R)
    for vi in ("permanence", "shape", "color"):
        for oc in ("visible", "early", "mid", "late"):
            row(f"{vi}/{oc}", [r for r in R if r["violation"] == vi and r["occlusion"] == oc])
    for os_ in ("primitive", "realistic"):
        row(os_, [r for r in R if r["objset"] == os_])
    print("\n".join(lines)); json.dump(summ, open(OUT / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    a = ap.parse_args()
    run(a.shard, a.nshard) if a.stage == "run" else report()
