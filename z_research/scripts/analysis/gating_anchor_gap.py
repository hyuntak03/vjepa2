"""gating anchor 후보와 실제 다음 블록 LN(target_encoder) 의 L1 (2026-10-17). VOBench visible · late 각 60 clip, 배경 / 물체 3×3.
  CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python z_research/scripts/analysis/gating_anchor_gap.py
후보: LN(문맥 encoder z 마지막 블록) vs LN(h) 마지막 문맥 블록 (타깃 공간 복사). 결과: z_training/RESULTS_GATING_2026-10-18.md §2"""
import sys, os, csv, collections, numpy as np, torch, torch.nn.functional as F
REPO = "/workspace/world_model/vjepa2"; sys.path.insert(0, REPO)
sys.argv = [sys.argv[0], "--occ", "late"]
src = open(f"{REPO}/z_research/scripts/analysis/obj_region_surprise.py").read()
src = src[: src.index("rows = [r for r in csv.DictReader")]
__file__ = f"{REPO}/z_research/scripts/analysis/obj_region_surprise.py"
exec(src)
enc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2 * C, use_rope=True, uniform_power=False)
_load_state_dict(enc, state["encoder"], tag="encoder"); enc = enc.to(dev).eval()
allrows = list(csv.DictReader(open(f"{REPO}/data_csv/vobench_v1/index.csv")))
out = collections.defaultdict(lambda: collections.defaultdict(list))
for occ in ("visible", "late"):
    rows = [r for r in allrows if r["occlusion"] == occ and r["variant"] == "pos_a"][:60]
    for r in rows:
        d = os.path.join(a.vobench, r["file_name"]); x, W0 = frames(d)
        H = blocks(x).view(16, S, -1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            z = enc(x[:, : 2 * C].unsqueeze(0).to(dev)).float()[0]
        zl = F.layer_norm(z[(C - 1) * S: C * S], (1280,))
        ot = obj_tokens(os.path.join(os.path.dirname(d), "label.json"), W0)
        obj = set(ot[0]); bg = torch.tensor([t for t in range(S) if t not in obj], device=dev)
        ob = torch.tensor(sorted(obj), device=dev) if obj else None
        for name, ref in (("LN(z) 마지막 문맥 블록 (지금 anchor)", zl), ("LN(h) 마지막 문맥 블록 (타깃 복사)", H[C - 1])):
            for k in (0, 3, 7):
                e = (ref - H[C + k]).abs().mean(-1)
                out[(occ, name)][f"bg{k}"].append(float(e[bg].mean()))
                if ob is not None: out[(occ, name)][f"ob{k}"].append(float(e[ob].mean()))
        e = (zl - H[C - 1]).abs().mean(-1); out[(occ, "LN(z) vs LN(h) 같은 블록")]["bg0"].append(float(e[bg].mean()))
print("L1 (배경 / 물체 3x3), 다음 블록 걸음 0 · 3 · 7 과 비교")
for (occ, name), v in out.items():
    print(f"{occ:8s} {name:36s} 배경 " + " ".join(f"{np.mean(v[f'bg{k}']):.3f}" for k in (0, 3, 7) if v.get(f'bg{k}')) +
          ("   물체 " + " ".join(f"{np.mean(v[f'ob{k}']):.3f}" for k in (0, 3, 7)) if v.get('ob0') else ""), " n", len(v["bg0"]))
