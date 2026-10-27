#!/usr/bin/env python3
"""굴린 예측의 걸음마다 — 물체가 그려져야 할 칸의 query 가 어디에 attention 을 주나 (2026-10-24, 사용자: "VOBench post-FT 가 옮기는데 무엇이 바뀐 건지").

  CUDA_VISIBLE_DEVICES=0 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/rollout_attention_compare.py \
      --run natural_ar_vw_state latest.pt --run vobench_ar_vw_state_postft_g latest.pt [--limit 0]

데이터 · 창: transport_gain.py 와 같다 (phys3d_single_v1 val, r0 ∈ {0,2}, 간격 3, 문맥 블록 0..7 → 미래 8..15). 문맥 끝에 물체가 보이는 창.
걸음 j (0..7) = 위치 C−1+j 를 계산해 블록 C+j 를 내는 걸음 (j=0 은 마지막 문맥 블록 자리, 입력이 관측. j≥1 은 직전 걸음의 상태 되먹임).
query = 블록 C+j 의 정답 물체 위치 중심 3×3 칸 (그 걸음 출력이 물체를 그려야 할 칸) — 그 걸음의 입력 블록 (위치 C−1+j) 토큰들 중 같은 칸.
key 묶음 (층마다, 헤드 평균, query 9 칸 평균):
  self   = 지금 입력 블록의 같은 칸 (query 자기 자리)
  prevT  = 지금 입력 블록의 **정답 직전 물체 자리** (블록 C−1+j 의 정답 위치 3×3) — 물체가 "있어야 할" 직전 자리
  lastO  = 지금 입력 블록의 **마지막 관측 물체 자리** (블록 C−1 위치 3×3) — 예측이 물체를 안 옮겼다면 물체가 여기 있다
  ctxO   = 마지막 문맥 블록 (관측, 위치 C−1) 의 물체 자리 3×3
  past   = 지금 블록보다 앞선 블록 전체 (ctxO 제외)
  rest   = 지금 입력 블록의 나머지
걸음 블록에서 물체가 1 칸 넘게 움직였고 (마지막 관측 대비) 보일 때만. 묶음이 겹치면 앞 순서 (self > prevT > lastO) 에 넣는다.
보고: 걸음별 · 층 묶음별 (앞 1–4 · 중 5–8 · 뒤 9–12) 질량, 그리고 그 걸음 예측의 위치 읽기 (마지막 관측 물체 모양 cosine 최대 칸) 가
정답 / 마지막 관측 자리 중 어디에 가까운가."""
import argparse, glob, os, sys
import numpy as np, torch, torch.nn.functional as F, yaml
import decord
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint, _clean_backbone_key   # noqa: E402
from src.models.rollout_predictor import vit_prefix_predictor                                                  # noqa: E402
from src.models.utils import modules as Mmod                                                                   # noqa: E402
from app.vjepa_frozen import ar as AR                                                                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/workspace/dataset/training_data/phys3d_single_v1/val/clips")
ap.add_argument("--run", nargs=2, action="append", required=True, metavar=("RUN", "CKPT"))
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
dev = torch.device("cuda")
S, G, C, K, ST = 256, 16, 8, 8, 3
T = 2 * (C + K)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
tgt = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(tgt, _load_checkpoint(CK)["target_encoder"], tag="t"); tgt = tgt.to(dev, torch.bfloat16).eval()


def load_pred(run, ck):
    D = os.path.join(REPO, "z_training/runs", run)
    pc = yaml.safe_load(open(os.path.join(D, "config.yaml")))["model"]["predictor"]
    assert pc.get("kind") == "ar"
    m = vit_prefix_predictor(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384, depth=12,
                             num_heads=12, num_mask_tokens=10, mask_mode="ar", ar_feedback=pc.get("ar_feedback", "output"))
    m.load_state_dict({k: v for k, v in _clean_backbone_key(torch.load(os.path.join(D, ck), map_location="cpu", weights_only=False)["predictor"]).items()
                       if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}, strict=True)
    return m.to(dev).eval()


models = [(r, load_pred(r, c)) for r, c in a.run]
REC = {"on": False, "q": None, "attn": []}
_sdpa = F.scaled_dot_product_attention


def sdpa(q, k, v, *args, **kw):
    if REC["on"]:
        qq = q[:, :, REC["q"]].float()
        w = torch.softmax(qq @ k.float().transpose(-1, -2) / q.size(-1) ** 0.5, dim=-1)
        REC["attn"].append(w[0].mean(1).cpu())                               # query 평균, 헤드별 → (h, M)
    return _sdpa(q, k, v, *args, **kw)


Mmod.F.scaled_dot_product_attention = sdpa
cells3 = lambda p: [y * G + x for y in range(max(0, int(p[1] // 16) - 1), min(G, int(p[1] // 16) + 2))
                    for x in range(max(0, int(p[0] // 16) - 1), min(G, int(p[0] // 16) + 2))]
yy, xx = np.divmod(np.arange(S), G)
cen = np.stack([xx * 16 + 8, yy * 16 + 8], 1).astype(np.float32)
GROUPS = ["self", "prevT", "lastO", "ctxO", "past", "rest"]
R = {name: [] for name, _ in models}           # 행 = (걸음, (12, 6) 질량, 읽은 위치 ↔ 정답 거리, ↔ 마지막 관측 거리)
paths = sorted(p for p in glob.glob(os.path.join(a.root, "*", "*.mp4")) if np.load(p[:-4] + "_states.npz")["pos"].shape[0] >= 96)
if a.limit:
    paths = paths[: a.limit]
for ci, path in enumerate(paths):
    st = np.load(path[:-4] + "_states.npz")
    vr = decord.VideoReader(path, num_threads=2)
    for r0 in (0, 2):
        idx = [r0 + ST * k for k in range(T)]
        posb = [st["pos"][idx[2 * b + 1], 0] for b in range(C + K)]
        visb = [bool(st["vis_frac"][idx[2 * b], 0] > 0.3 and st["vis_frac"][idx[2 * b + 1], 0] > 0.3) for b in range(C + K)]
        if not (visb[C - 1] and np.isfinite(posb[C - 1]).all()):
            continue
        x = torch.from_numpy(vr.get_batch(idx).asnumpy()).to(dev).permute(3, 0, 1, 2)[None].float().div_(255)
        if x.shape[-1] != 256:
            x = F.interpolate(x[0].transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False).transpose(0, 1)[None]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hb = AR.encode_blocks(tgt, ((x - MEAN) / STD).to(torch.bfloat16), 2).float()[0].view(C + K, S, -1)
        tmpl = F.normalize(hb[C - 1][cells3(posb[C - 1])].mean(0), dim=-1)
        lastO = cells3(posb[C - 1])
        for name, m in models:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                caches, p, stt = None, None, None
                for b in range(C):
                    if b == C - 1:
                        j = 0
                    p_, caches_, st_ = None, None, None
                    if b == C - 1 and visb[C] and np.isfinite(posb[C]).all() and np.linalg.norm(posb[C] - posb[C - 1]) > 16:
                        REC.update(on=True, q=cells3(posb[C]), attn=[])
                    p, caches, stt = m._ar_step_kv(hb[b:b + 1], torch.arange(b * S, (b + 1) * S, device=dev).unsqueeze(0), caches)
                    if REC["on"]:
                        REC["on"] = False
                        R[name].append((0, REC["attn"][:], p[0].float()))
                    REC["attn"] = []
                preds = [p]
                pos = torch.arange((C - 1) * S, C * S, device=dev).unsqueeze(0)
                for j in range(1, K):
                    pos = pos + S
                    tb = C + j
                    rec = visb[tb] and np.isfinite(posb[tb]).all() and np.linalg.norm(posb[tb] - posb[C - 1]) > 16
                    if rec:
                        REC.update(on=True, q=cells3(posb[tb]), attn=[])
                    if m.ar_feedback == "state":
                        p, caches, stt = m._ar_step_kv(stt + m.type_embed[1], pos, caches, embedded=True)
                    else:
                        p, caches, stt = m._ar_step_kv(p, pos, caches)
                    if rec:
                        REC["on"] = False
                        R[name].append((j, REC["attn"][:], p[0].float()))
                    REC["attn"] = []
            # 방금 넣은 행들 (이 창) 을 묶음 질량 · 위치 읽기로 바꾼다
            for ri in range(len(R[name])):
                row = R[name][ri]
                if isinstance(row[1], list):
                    j, att, pblk = row
                    P = C - 1 + j                                           # 그 걸음의 입력 블록 위치
                    cur = {"self": [P * S + c for c in cells3(posb[C + j])], "prevT": [P * S + c for c in cells3(posb[C - 1 + j])],
                           "lastO": [P * S + c for c in lastO], "ctxO": [(C - 1) * S + c for c in lastO]}
                    used = set(); idxs = {}
                    for g in ["self", "prevT", "lastO", "ctxO"]:
                        if g == "ctxO" and P == C - 1:
                            idxs[g] = []; continue
                        idxs[g] = [i for i in cur[g] if i not in used]; used |= set(idxs[g])
                    mass = []                                                # (12 층, h 헤드, 6 묶음)
                    for w in att:
                        w = w.numpy(); tot = {g: w[:, idxs[g]].sum(1) for g in idxs}
                        tot["past"] = w[:, : P * S].sum(1) - tot["ctxO"]
                        tot["rest"] = 1 - sum(tot.values())
                        mass.append(np.stack([tot[g] for g in GROUPS], 1))
                    cs = (F.normalize(pblk, dim=-1) @ tmpl).cpu().numpy(); k_ = int(cs.argmax()); nb = cells3(cen[k_])
                    wq = np.clip(cs[nb], 0, None) + 1e-6; rd = (cen[nb] * wq[:, None]).sum(0) / wq.sum()
                    R[name][ri] = (j, np.array(mass, np.float32), float(np.linalg.norm(rd - posb[C + j])), float(np.linalg.norm(rd - posb[C - 1])))
    if ci % 50 == 0:
        print(f"{ci}/{len(paths)} · " + " · ".join(f"{n} {len(v)}" for n, v in R.items()), flush=True)

print("\n물체가 그려져야 할 칸 (정답 3×3) 의 query 가 주는 attention 질량 — 층 묶음별 (앞 1–4 / 중 5–8 / 뒤 9–12), 걸음별. 균등 기준: 9 칸 ≈ 9/(M) (M = 앞 블록 포함 전체 키)")
print("  묶음: self 자기 칸 · prevT 직전 정답 물체 자리 · lastO 마지막 관측 물체 자리 (같은 입력 블록) · ctxO 마지막 문맥 블록 물체 자리 · past 앞 블록 · rest 지금 블록 나머지")
for name, _ in models:
    rows = R[name]
    print(f"\n[{name}] (행 {len(rows)})")
    print(f"  {'걸음':>4s} {'n':>4s}  " + "  ".join(f"{g:>17s}" for g in GROUPS) + "   읽은 위치 ↔ 정답 / ↔ 마지막 관측 (px)")
    for j in range(K):
        rr = [r for r in rows if r[0] == j]
        if len(rr) < 5:
            continue
        Mx = np.stack([r[1] for r in rr]).mean(0).mean(1)  # (12, 6) — 헤드 평균
        cell = lambda g: "/".join(f"{Mx[s:e, GROUPS.index(g)].mean():.2f}" for s, e in ((0, 4), (4, 8), (8, 12)))
        print(f"  {j + 1:>4d} {len(rr):>4d}  " + "  ".join(f"{cell(g):>17s}" for g in GROUPS)
              + f"   {np.median([r[2] for r in rr]):5.1f} / {np.median([r[3] for r in rr]):5.1f}")


# ---- 헤드별: 걸음 2–8 (상태 되먹임 걸음) 에서 그려야 할 칸 query → 직전 정답 물체 자리 (prevT) 질량. 모델 차이가 큰 헤드 = 옮기기 후보 ----
np.save("/tmp/claude-0/-workspace-world-model-vjepa2/70827acc-8b80-4c7d-a8ba-51be17efafb7/scratchpad/rollout_attn_rows.npy",
        np.array({n: [(r[0], r[1], r[2], r[3]) for r in v] for n, v in R.items()}, dtype=object), allow_pickle=True)
H = {}
for name, _ in models:
    rr = [r for r in R[name] if r[0] >= 1]
    if not rr:
        sys.exit("걸음 2–8 행이 없다 (--limit 을 키울 것)")
    H[name] = np.stack([r[1] for r in rr]).mean(0)          # (12, h, 6)
names = [n for n, _ in models]
print("\n[헤드별] 걸음 2–8, prevT (직전 정답 물체 자리) · lastO (마지막 관측 자리) · self 질량. 9 칸 균등 ≈ 9/M")
for g in ("prevT", "lastO", "self"):
    gi = GROUPS.index(g)
    d = H[names[-1]][:, :, gi] - H[names[0]][:, :, gi]
    order = np.dstack(np.unravel_index(np.argsort(-np.abs(d).ravel()), d.shape))[0][:8]
    print(f"  {g}: 차이 큰 헤드 (층.헤드  {names[0][:14]} → {names[-1][:14]})")
    print("    " + "   ".join(f"L{l + 1}.h{h}: {H[names[0]][l, h, gi]:.3f}→{H[names[-1]][l, h, gi]:.3f}" for l, h in order))
