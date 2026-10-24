"""R2 review (round 2): own recount of predictor parameter distances (L2 over predictor tensors) and the size of the
2,000-step budget relative to the start checkpoint. Read-only, CPU."""
import torch
R = "/data/hyuntak/project/2026/2027_cvpr/vjepa2/"
def P(p):
    ck = torch.load(R + p, map_location="cpu", weights_only=False)
    return ck["predictor"], {k: ck[k] for k in ck if k not in ("predictor", "opt", "optimizer", "scaler", "target_encoder", "encoder")}
start, smeta = P("z_training/vast/ar_state/latest.pt")
print("start ckpt meta keys:", {k: (v if not hasattr(v, "keys") else list(v.keys())[:8]) for k, v in smeta.items() if k in ("epoch", "itr", "step", "global_step", "lr", "batch_size")})
S = {a: P(f"paper_lab/_scratch/runs/pl_pilot_{a}/e10.pt")[0] for a in ("A0", "AX", "A2c", "A4")}
def d(x, y):
    return sum(((x[k].float() - y[k].float()) ** 2).sum() for k in x if k in y and x[k].shape == y[k].shape).sqrt().item()
norm = sum((v.float() ** 2).sum() for v in start.values()).sqrt().item()
print(f"||start|| = {norm:.2f}")
for a in S: print(f"{a} - start {d(S[a], start):.3f}")
for a, b in (("AX", "A0"), ("A2c", "AX"), ("A4", "AX"), ("A2c", "A0"), ("A4", "A0")): print(f"{a} - {b} {d(S[a], S[b]):.3f}")
