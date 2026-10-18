"""gating 출력 크기 진단 (2026-10-18). AR vs AR+gating (natural_ar_vw_state_gres) 의 rollout 출력 토큰별 D 방향 표준편차 · L1, 복사 기준선.
  CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python z_research/scripts/analysis/gating_out_scale.py
결과: z_training/docs/RESULTS_GATING_2026-10-18.md §5"""
import sys, os, csv, numpy as np, torch
REPO = "/workspace/world_model/vjepa2"; sys.path.insert(0, REPO)
sys.argv = [sys.argv[0], "--occ", "late"]
src = open(f"{REPO}/z_research/scripts/analysis/obj_region_surprise.py").read()
src = src[: src.index("class Model:")]
__file__ = f"{REPO}/z_research/scripts/analysis/obj_region_surprise.py"
exec(src)
def load(run, **kw):
    m = vit_prefix_predictor(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384,
                             depth=12, num_heads=12, num_mask_tokens=10, mask_mode="ar", ar_feedback="state", **kw)
    sd = _clean_backbone_key(torch.load(f"{REPO}/z_training/runs/{run}/latest.pt", map_location="cpu", weights_only=False)["predictor"])
    m.load_state_dict(sd, strict=True); return m.to(dev).eval()
M = {"AR": load("natural_ar_vw_state"), "gres": load("natural_ar_vw_state_gres", gating_residual=True)}
rows = [r for r in csv.DictReader(open(f"{REPO}/data_csv/vobench_v1/index.csv")) if r["variant"] == "pos_a"][:80]
out = {k: {"std": [], "l1": [], "copy": []} for k in M}
idx = torch.arange(C * S, device=dev).unsqueeze(0)
for r in rows:
    x, _ = frames(os.path.join(a.vobench, r["file_name"])); H = blocks(x)
    for k, m in M.items():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            p = m.rollout_kv(H[: C * S].unsqueeze(0), idx, K).float()[0]
        t = H[C * S:]
        out[k]["std"].append(p.view(K, S, -1).std(-1).mean(-1).cpu().numpy())
        out[k]["l1"].append((p - t).abs().mean(-1).view(K, S).mean(-1).cpu().numpy())
    out["AR"]["copy"].append((H[(C - 1) * S: C * S].unsqueeze(0) - t.view(K, S, -1)).abs().mean(-1).mean(-1).cpu().numpy())
print("걸음 1 · 3 · 5 · 8 (rollout), VOBench 80 clip (pos_a)")
for k in M:
    sd = np.mean(out[k]["std"], 0); l1 = np.mean(out[k]["l1"], 0)
    print(f"  {k:5s} 출력 std (타깃 = 1.000) " + " ".join(f"{sd[i]:.3f}" for i in (0, 2, 4, 7)) + "   |  L1 오차 " + " ".join(f"{l1[i]:.3f}" for i in (0, 2, 4, 7)))
c = np.mean(out["AR"]["copy"], 0); print("  복사  L1 오차 " + " ".join(f"{c[i]:.3f}" for i in (0, 2, 4, 7)))
