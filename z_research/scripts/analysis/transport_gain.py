#!/usr/bin/env python3
"""블록 AR run 의 **운반 이득** 만 잰다 — ctxt_vs_block_mechanism.py 측정 [1] 과 같은 정의 · 같은 데이터 (2026-10-21).

  CUDA_VISIBLE_DEVICES=0 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/transport_gain.py \
      --run natural_ar_vw_state latest.pt --run ar_phys3d_only latest.pt [--limit 0]

데이터: phys3d_single_v1 **val** (학습에 안 쓴 clip), 96 장 clip, 창 시작 raw r0 ∈ {0, 2}, 간격 3, 문맥 16 장 (블록 0..7) → 미래 블록 8..15.
위치 읽기 (파라미터 없음): 템플릿 = 마지막 문맥 블록 (정답 LN 블록) 의 물체 중심 3×3 평균, 예측 블록 256 칸과 cosine → 최대 칸 주변 3×3 의 cos 가중 중심.
운반 이득 = (d_pred · d_true) / |d_true|²  (1 = 정확히 옮김, 0 = 제자리). |d_true| > 16 px 이고 그 블록에 물체가 보일 때만.
  rollout (굴린 예측): d 는 문맥 끝 위치 기준 (누적).  TF (정답 과거를 다 보고 한 걸음): d 는 직전 정답 블록 위치 기준.
같은 읽기를 정답 블록에 건 값 = 측정 바닥 (이 읽기로 낼 수 있는 최선에 가까움)."""
import argparse, glob, os, random, sys
import numpy as np, torch, torch.nn.functional as F, yaml
import decord
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint, _clean_backbone_key   # noqa: E402
from src.models.rollout_predictor import vit_prefix_predictor                                                  # noqa: E402
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
    assert pc.get("kind") == "ar", f"{run}: kind=ar 만"
    kw = dict(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384, depth=12, num_heads=12,
              num_mask_tokens=10, mask_mode="ar", ar_feedback=pc.get("ar_feedback", "output"))
    for k_ in ("gating_residual", "gating_anchor", "gating_out_ln"):
        if pc.get(k_) is not None:
            kw[k_] = pc[k_]
    m = vit_prefix_predictor(**kw)
    w = {k: v for k, v in _clean_backbone_key(torch.load(os.path.join(D, ck), map_location="cpu", weights_only=False)["predictor"]).items()
         if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}
    m.load_state_dict(w, strict=True)
    return m.to(dev).eval()


def ablate(m, spec):
    """2026-10-24: head 끄기 — spec "12.7,2.6" (층 1-기반 . 헤드 0-기반). 그 층 attention 의 출력 투영 입력에서 그 헤드 32 차원을 0 으로 (모든 걸음)."""
    hd = m.predictor_blocks[0].attn.proj.in_features // m.predictor_blocks[0].attn.num_heads
    for t in spec.split(","):
        l, h = (int(v) for v in t.split("."))
        def pre(mod, args, h=h):
            x = args[0].clone(); x[..., h * hd:(h + 1) * hd] = 0
            return (x,) + tuple(args[1:])
        m.predictor_blocks[l - 1].attn.proj.register_forward_pre_hook(pre)
    return m


models = []
for r, c in a.run:                     # RUN 에 "@12.7,2.6" 을 붙이면 그 헤드를 끈다
    run_, _, spec = r.partition("@")
    m_ = load_pred(run_, c)
    models.append((f"{r}:{c}", ablate(m_, spec) if spec else m_))
cells3 = lambda cx, cy: [y * G + x for y in range(max(0, cy - 1), min(G, cy + 2)) for x in range(max(0, cx - 1), min(G, cx + 2))]
cell_of = lambda p: (int(np.clip(p[0] // 16, 0, G - 1)), int(np.clip(p[1] // 16, 0, G - 1)))
yy, xx = np.divmod(np.arange(S), G)
cen = np.stack([xx * 16 + 8, yy * 16 + 8], 1).astype(np.float32)
paths = sorted(p for p in glob.glob(os.path.join(a.root, "*", "*.mp4")) if np.load(p[:-4] + "_states.npz")["pos"].shape[0] >= 96)
if a.limit:
    paths = paths[: a.limit]
R = {"pc": [], "futp": [], "futv": [], "clip": []}
P = {name: {"ro": [], "tf": []} for name, _ in models}
P["gt"] = {"ro": []}
for ci, path in enumerate(paths):
    st = np.load(path[:-4] + "_states.npz")
    vr = decord.VideoReader(path, num_threads=2)
    for r0 in (0, 2):
        idx = [r0 + ST * k for k in range(T)]
        fe = idx[2 * C - 1]
        pos_c = st["pos"][fe, 0]
        if not (st["vis_frac"][fe, 0] > 0.3 and np.isfinite(pos_c).all()):
            continue
        x = torch.from_numpy(vr.get_batch(idx).asnumpy()).to(dev).permute(3, 0, 1, 2)[None].float().div_(255)
        if x.shape[-1] != 256:
            x = F.interpolate(x[0].transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False).transpose(0, 1)[None]
        xn = (x - MEAN) / STD
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hb = AR.encode_blocks(tgt, xn.to(torch.bfloat16), 2).float()[0].view(C + K, S, -1)
        tmpl = F.normalize(hb[C - 1][cells3(*cell_of(pos_c))].mean(0), dim=-1)

        def readout(blk):
            cs = (F.normalize(blk.float(), dim=-1) @ tmpl).cpu().numpy()
            j = int(cs.argmax()); nb = cells3(j % G, j // G)
            w = np.clip(cs[nb], 0, None) + 1e-6
            return (cen[nb] * w[:, None]).sum(0) / w.sum()
        R["pc"].append(pos_c); R["clip"].append(ci)
        R["futp"].append(np.stack([st["pos"][idx[2 * (C + j) + 1], 0] for j in range(K)]))
        R["futv"].append(np.array([st["vis_frac"][idx[2 * (C + j) + 1], 0] > 0.3 for j in range(K)]))
        P["gt"]["ro"].append(np.stack([readout(hb[C + j]) for j in range(K)]))
        idx_ctx = torch.arange(C * S, device=dev).unsqueeze(0)
        for name, m in models:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                p_ro = m.rollout_kv(hb[:C].reshape(1, C * S, -1), idx_ctx, K).float()[0].view(K, S, -1)
                p_tf = m.forward_seq(hb[:C + K - 1].reshape(1, -1, hb.size(-1)), torch.arange((C + K - 1) * S, device=dev).unsqueeze(0)).float()[0]
            p_tf = p_tf.view(C + K - 1, S, -1)[C - 1:]
            P[name]["ro"].append(np.stack([readout(p_ro[j]) for j in range(K)]))
            P[name]["tf"].append(np.stack([readout(p_tf[j]) for j in range(K)]))
    if ci % 50 == 0:
        print(f"{ci}/{len(paths)} clip · 창 {len(R['pc'])}", flush=True)

pc, futp, futv, clip = (np.asarray(R[k]) for k in ("pc", "futp", "futv", "clip"))
prevt = np.concatenate([pc[:, None], futp[:, :-1]], 1)
rng = np.random.RandomState(0)


def boot(gv, cl):
    uc = np.unique(cl); ix = {c: np.where(cl == c)[0] for c in uc}
    bs = [np.median(gv[np.concatenate([ix[c] for c in rng.choice(uc, len(uc))])]) for _ in range(1000)]
    return np.percentile(bs, [2.5, 97.5])


print(f"\n창 {len(pc)} (clip {len(np.unique(clip))}). 운반 이득 중앙값 — 걸음 1 · 2 · 4 · 6 · 8 (괄호 위치 오차 px), 걸음 1 은 clip bootstrap 95 % 구간")
for name in ["gt"] + [n for n, _ in models]:
    for mode in (("ro", "tf") if name != "gt" else ("ro",)):
        Q = np.stack(P[name][mode])
        base = prevt if mode == "tf" else pc[:, None]
        dp, dtr = Q - base, futp - base
        g = (dp * dtr).sum(-1) / np.maximum((dtr ** 2).sum(-1), 1e-6)
        err = np.linalg.norm(Q - futp, axis=-1)
        ok = (np.linalg.norm(dtr, axis=-1) > 16) & futv & np.isfinite(g) & np.isfinite(dtr).all(-1)
        row = [f"{np.median(g[ok[:, j], j]):5.2f} ({np.median(err[ok[:, j], j]):4.1f})" if ok[:, j].sum() > 5 else "  —  " for j in (0, 1, 3, 5, 7)]
        ci_ = boot(g[ok[:, 0], 0], clip[ok[:, 0]]) if ok[:, 0].sum() > 5 else (np.nan, np.nan)
        allg = g[ok]
        print(f"  {name + (' · ' + ('굴린 예측' if mode == 'ro' else 'TF 한 걸음') if name != 'gt' else ' (정답 블록 = 측정 바닥)'):58s} "
              + "  ".join(row) + f"   걸음1 [{ci_[0]:.2f}, {ci_[1]:.2f}] n {ok[:, 0].sum()} · 전 걸음 중앙 {np.median(allg):.2f} (n {ok.sum()})")
