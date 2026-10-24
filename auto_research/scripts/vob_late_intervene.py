#!/usr/bin/env python3
"""[P] 고리 4 개입 — VOBench late 정지 물체 유지가 "앞쪽 문맥 블록의 물체가 평균에 남는 것" 인가 (auto_research, 2026-10-21).

  CUDA_VISIBLE_DEVICES=5 /venv/vjepa/bin/python -W ignore auto_research/scripts/vob_late_intervene.py \
      --run natural_ar_vw_state latest.pt --wjson auto_research/results/pred_mean/natural_ar_vw_state__latest.json

대상: late · flat_static block 전부 (36 block, permanence · shape · color × k 1–3). 채점 구성은 vob_late_mean.py 와 같다.
개입 1 (문맥 자르기): 굴린 예측의 문맥을 마지막 m 블록 (m = 1 · 2 · 3 · 4 · 8) 만 — 위치 번호는 원래 자리 그대로.
   평균 가설의 예측: k=2 (마지막 블록 전체가 가림막) 에서 m=1 이면 물체 자리 정확도가 우연으로, m ≥ 2 면 회복.
개입 2 (앞쪽 블록 바꿔치기, shape · color 만): 문맥 = pos_b 의 블록 0..7−m + pos_a 의 마지막 m 블록 (m = 2 · 3).
   평균 가설의 예측: 예측된 물체 자리가 B 쪽으로 기운다 (B 선호율 ↑). "마지막으로 보인 블록을 복사" 가설의 예측: A 유지.
   B 선호율 = P( L1(예측, B 미래) < L1(예측, A 미래) ), 물체 자리 토큰, 미래 블록 10..15 (가림이 끝난 뒤).
같은 개입을 모델 없는 예측기 (copy · mean_8 · LA_rec) 에도 건다 — 기대치."""
import argparse, csv, json, os, sys, collections
import numpy as np, torch, torch.nn.functional as F, yaml
from PIL import Image
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint, _clean_backbone_key   # noqa: E402
from src.models.rollout_predictor import vit_prefix_predictor                                                  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", nargs=2, default=["natural_ar_vw_state", "latest.pt"])
ap.add_argument("--wjson", required=True)
ap.add_argument("--vobench", default="/workspace/dataset/benchmarks/VOBench_v1")
ap.add_argument("--motion", default="flat_static")
ap.add_argument("--out", default=os.path.join(REPO, "auto_research/results/vob_late_intervene.json"))
a = ap.parse_args()
dev = torch.device("cuda")
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
S, C, K, G, L = 256, 8, 8, 16, 8
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
IDX = list(range(0, 96, 3))[:32]
tenc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(tenc, _load_checkpoint(CK)["target_encoder"], tag="t"); tenc = tenc.to(dev).eval()
W = json.load(open(a.wjson))["tf"]["mov/far"]["pred"]["M0_w"]
D = os.path.join(REPO, "z_training/runs", a.run[0])
pc = yaml.safe_load(open(os.path.join(D, "config.yaml")))["model"]["predictor"]
pred = vit_prefix_predictor(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384, depth=12,
                            num_heads=12, num_mask_tokens=10, mask_mode="ar", ar_feedback=pc.get("ar_feedback", "output"))
sd = {k: v for k, v in _clean_backbone_key(torch.load(os.path.join(D, a.run[1]), map_location="cpu", weights_only=False)["predictor"]).items()
      if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}
pred.load_state_dict(sd, strict=True); pred = pred.to(dev).eval()


def frames(d):
    fr = [np.asarray(Image.open(f"{d}/{i:06d}.png").convert("RGB")) for i in IDX]
    x = torch.from_numpy(np.stack(fr)).permute(3, 0, 1, 2).float() / 255.0
    W0 = x.shape[-1]
    x = F.interpolate(x.transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False).transpose(0, 1)
    return (x - MEAN) / STD, W0


@torch.no_grad()
def blocks(x):
    xb = x.reshape(3, 16, 2, 256, 256).permute(1, 0, 2, 3, 4).to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = torch.cat([tenc(xb[i:i + 8]) for i in range(0, 16, 8)], 0).float()
    return F.layer_norm(h, (1280,)).view(16, S, -1)


def obj_cells(label_path, W0):
    L_ = json.load(open(label_path))["labels"]; out = []
    for b in range(C, C + K):
        s_ = set()
        for key in ("A", "B"):
            lab = {q["f"]: q for q in L_.get(key, [])}
            q = [lab.get(IDX[2 * b]), lab.get(IDX[2 * b + 1])]
            if any(t is None for t in q):
                continue
            r0 = int(np.clip(np.mean([t["v"] for t in q]) / W0 * G, 0, G - 1)); c0 = int(np.clip(np.mean([t["u"] for t in q]) / W0 * G, 0, G - 1))
            s_ |= {rr * G + cc for rr in range(max(0, r0 - 1), min(G, r0 + 2)) for cc in range(max(0, c0 - 1), min(G, c0 + 2))}
        out.append(sorted(s_))
    return out


@torch.no_grad()
def ar_roll(ctx_blocks, start):
    """ctx_blocks (m, S, D) = 원래 블록 start..start+m−1. 위치 번호는 원래 자리. 미래 블록 C..C+K−1 까지 굴린다."""
    m = ctx_blocks.shape[0]
    idx = torch.arange(start * S, (start + m) * S, device=dev).unsqueeze(0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return pred.rollout_kv(ctx_blocks.reshape(1, m * S, -1), idx, K).float()[0].view(K, S, -1)


def la_roll(ctx):
    seq = [ctx[k] for k in range(ctx.shape[0])]; out = []
    for j in range(K):
        n_ = min(L, len(seq)); sc_ = sum(W) / sum(W[:n_])                       # 문맥이 짧으면 가중치를 원래 합으로 다시 맞춘다
        nxt = sum(float(W[k]) * sc_ * seq[-1 - k] for k in range(n_))
        out.append(nxt); seq.append(nxt)
    return torch.stack(out)


rows = [r for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv"))) if r["occlusion"] == "late" and r["motion"] == a.motion]
blk = collections.defaultdict(dict)
for r in rows:
    blk[r["block_id"]][r["variant"]] = r
TR = collections.defaultdict(list)       # (predictor, m, k) → 정답 여부 (물체 자리)
SW = collections.defaultdict(list)       # (predictor, m, k) → B 선호
for bi, (b, arms) in enumerate(sorted(blk.items())):
    d0 = os.path.join(a.vobench, arms["pos_a"]["file_name"])
    H, W0 = {}, None
    for v in ("pos_a", "pos_b", "imp_ab", "imp_ba"):
        x, W0 = frames(os.path.join(a.vobench, arms[v]["file_name"])); H[v] = blocks(x)
    oc = obj_cells(os.path.join(os.path.dirname(d0), "label.json"), W0)
    oidx = torch.tensor([j * S + t for j, tt in enumerate(oc) for t in tt], device=dev)
    oidx_late = torch.tensor([j * S + t for j, tt in enumerate(oc) for t in tt if j >= 2], device=dev)
    k = int(arms["pos_a"]["k"]); viol = arms["pos_a"]["violation"]
    for pos, imp in (("pos_a", "imp_ab"), ("pos_b", "imp_ba")):
        h = H[pos]
        for m in (1, 2, 3, 4, 8):
            ctx = h[C - m:C]
            P = {"AR": ar_roll(ctx, C - m), "copy": ctx[-1:].expand(K, S, -1), "mean": ctx.mean(0, keepdim=True).expand(K, S, -1), "LA_rec": la_roll(ctx)}
            for nm, p in P.items():
                pf = p.reshape(K * S, -1)
                e = {v: float((pf[oidx] - H[v][C:].reshape(K * S, -1)[oidx]).abs().mean()) for v in (pos, imp)}
                TR[(nm, m, k)].append(e[imp] > e[pos]); TR[(nm, m, 0)].append(e[imp] > e[pos])
    if viol in ("shape", "color"):
        for m in (2, 3, 8):
            for base, other in (("pos_a", "pos_b"), ("pos_b", "pos_a")):
                ctx = torch.cat([H[other][:C - m], H[base][C - m:C]]) if m < 8 else H[base][:C]      # m=8 = 바꿔치기 없음 (기준)
                P = {"AR": ar_roll(ctx, 0), "copy": ctx[-1:].expand(K, S, -1), "mean": ctx.mean(0, keepdim=True).expand(K, S, -1), "LA_rec": la_roll(ctx)}
                for nm, p in P.items():
                    pf = p.reshape(K * S, -1)
                    e = {v: float((pf[oidx_late] - H[v][C:].reshape(K * S, -1)[oidx_late]).abs().mean()) for v in (base, other)}
                    SW[(nm, m, k)].append(e[other] < e[base]); SW[(nm, m, 0)].append(e[other] < e[base])
    print(f"{bi}/{len(blk)}", flush=True)

out = {"truncate": {}, "swap": {}}
print(f"\n== 개입 1: 문맥을 마지막 m 블록만 — 물체 자리 정확도 (%), {a.motion} late")
for nm in ("AR", "copy", "mean", "LA_rec"):
    for k in (0, 1, 2, 3):
        vals = {m: 100 * np.mean(TR[(nm, m, k)]) for m in (1, 2, 3, 4, 8)}
        out["truncate"][f"{nm}/k{k or 'all'}"] = dict(n=len(TR[(nm, 1, k)]), acc=vals)
        print(f"  {nm:6s} k{k or '*'} n{len(TR[(nm, 1, k)]):3d}  " + " · ".join(f"m{m} {v:5.1f}" for m, v in vals.items()))
print("\n== 개입 2: 앞쪽 문맥 블록을 다른 물체 (B) 로 — 미래 물체 자리에서 B 선호율 (%), m = A 로 남긴 마지막 블록 수 (8 = 바꿔치기 없음)")
for nm in ("AR", "copy", "mean", "LA_rec"):
    for k in (0, 1, 2, 3):
        vals = {m: 100 * np.mean(SW[(nm, m, k)]) for m in (2, 3, 8)}
        out["swap"][f"{nm}/k{k or 'all'}"] = dict(n=len(SW[(nm, 2, k)]), pref_other=vals)
        print(f"  {nm:6s} k{k or '*'} n{len(SW[(nm, 2, k)]):3d}  " + " · ".join(f"m{m} {v:5.1f}" for m, v in vals.items()))
json.dump(out, open(a.out.replace(".json", f"_{a.motion}.json"), "w"), indent=1)
