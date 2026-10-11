"""action 민감도 sanity — predictor 가 action 을 실제로 쓰는가 (README §sanity 4).

    python z_research/action_planning/action_sensitivity.py --run pusht_ar_postft --ckpt latest.pt [--device cuda:0] [--n 77]

val 창 (C=3, K=5, 고정 시작점) 에서 같은 문맥으로 세 번 rollout:
  gt      : 데이터의 action           ← 이게 제일 낮아야 한다
  zero    : action 전부 평균 (정규화 0)
  shuffle : 배치 안에서 action 을 다른 창의 것으로 바꿈
스텝별 L1(blk LN(h)) 과 기준선 copy_last 를 같이 낸다. gt 가 zero/shuffle 보다 유의하게 낮지 않으면 planning 결과는 해석 불가.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch, yaml
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from app.vjepa_frozen import utils as U, ar as AR
from app.vjepa_frozen.data import normalize_clips
from app.vjepa_frozen.pusht_data import PushTDataset, WallDataset, ACTION_TYPES, make_traj_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--ckpt", default="latest.pt"); ap.add_argument("--device", default="cuda:0")
ap.add_argument("--C", type=int, default=3); ap.add_argument("--n", type=int, default=0, help="창 수 상한 (0 = 전부)"); ap.add_argument("--bs", type=int, default=8)
a = ap.parse_args()
dev = torch.device(a.device); rd = ROOT / "z_training/runs" / a.run
cfg = yaml.safe_load(open(rd / "config.yaml")); MD = cfg["model"]; MD.setdefault("use_sdpa", True)
nf, res = int(cfg["data"]["n_frames"]), int(cfg["data"]["resolution"])
dt = torch.bfloat16 if dev.type == "cuda" else torch.float32
state = U.load_base_checkpoint(MD["checkpoint"]); _, tgt = U.build_frozen_encoders(MD, state, dev, dt, nf)
MD2 = dict(MD); MD2.pop("init_predictor", None); P = U.build_predictor(MD2, tgt.embed_dim, None, dev, nf); del state
ck = torch.load(rd / a.ckpt, map_location="cpu", weights_only=False); P.load_state_dict(ck["predictor"]); P.eval()
spec = next(s for s in cfg["data"]["datasets"] if s.get("type") in ACTION_TYPES)
DS = WallDataset if spec["type"] == "wall" else PushTDataset
starts = [0, 5, 10] if spec["type"] == "wall" else [0, 20, 40, 60]          # Wall 은 episode 50 step → 시작 0..10
vs = make_traj_dataset(spec, "val", n_blocks=nf // 2, frameskip=int(spec.get("frameskip", 5)), resolution=res,
                       fixed_starts=starts, action_stats=DS.compute_action_stats(spec["root"]))
N = len(vs) if a.n <= 0 else min(a.n, len(vs)); S = (res // 16) ** 2; C = a.C; nb = nf // 2; K = nb - C
g = torch.Generator().manual_seed(0)
sums = {k: torch.zeros(K, dtype=torch.float64) for k in ("gt", "zero", "shuffle", "copy_last")}; n_win = 0
ac = torch.autocast("cuda", dtype=dt) if dev.type == "cuda" else torch.autocast("cpu", enabled=False)
with torch.no_grad():
    for s0 in range(0, N, a.bs):
        items = [vs[i] for i in range(s0, min(N, s0 + a.bs))]
        clips = normalize_clips(torch.stack([c["clips"] for c in items]).to(dev)); acts = torch.stack([c["actions"] for c in items]).to(dev)
        n = clips.size(0)
        with ac:
            h = AR.encode_blocks(tgt, clips.to(dt), 2); idx = AR.block_indices(n, nb, S, dev)
            variants = {"gt": acts, "zero": torch.zeros_like(acts), "shuffle": acts[torch.randperm(n, generator=g).to(dev)] if n > 1 else acts.flip(1)}
            hb = h.float().view(n, nb, S, -1); tgt_b = hb[:, C:]
            for k, A in variants.items():
                p = P.rollout(h[:, : C * S], idx[:, : C * S], K, actions=A).float().view(n, K, S, -1)
                sums[k] += (p - tgt_b).abs().mean(dim=(0, 2, 3)).double().cpu() * n
            sums["copy_last"] += (hb[:, C - 1: C].expand(-1, K, -1, -1) - tgt_b).abs().mean(dim=(0, 2, 3)).double().cpu() * n
        n_win += n
out = {k: (v / n_win).tolist() for k, v in sums.items()}
out.update(run=a.run, ckpt=a.ckpt, epoch=ck.get("epoch"), C=C, K=K, n_windows=n_win)
print(f"[{a.run}/{a.ckpt} e{ck.get('epoch')}] val 창 {n_win}, C={C} K={K}  (스텝별 rollout L1, 블록 LN(h))")
for k in ("gt", "zero", "shuffle", "copy_last"):
    print(f"  {k:9s} " + " ".join(f"{x:.4f}" for x in out[k]) + f"   | mean {sum(out[k])/K:.4f}")
gap = (sum(out["zero"]) - sum(out["gt"])) / K
print(f"  zero − gt = {gap:+.4f}  shuffle − gt = {(sum(out['shuffle'])-sum(out['gt']))/K:+.4f}   (양수가 커야 action 을 쓴다)")
od = ROOT / "z_research/action_planning/exp_results"; od.mkdir(parents=True, exist_ok=True)
json.dump(out, open(od / f"action_sensitivity_{a.run}_{Path(a.ckpt).stem}.json", "w"), indent=1)
