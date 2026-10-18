#!/usr/bin/env python3
"""측정판 (2026-10-18) — 벤치마크 점수가 가리는 것을 같은 자로 잰다. 학습 없이, checkpoint 마다 몇 분.

  CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python z_training/tools/panel.py natural_ar_vw_state                 # latest.pt
  CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python z_training/tools/panel.py natural_ar_vw_state_gres e10.pt e20.pt
  → z_training/runs/<run>/panel.jsonl 에 checkpoint 마다 한 줄 (monitor_ar.sh 가 읽는다)

항목 (VOBench, 고정 샘플 seed 0 — 모든 run 에 같은 clip):
  transport  운반 분해. 예측 블록의 물체 칸 i 를 직전 **진짜** 블록의 칸 i 주변 5×5 가중합으로 (최소제곱, 칸마다).
             src = 실제 이동량으로 정한 원래 칸 (물체가 있던 자리) 의 가중치 · same = 같은 칸 (복사) · conc = 최대 |w| / Σ|w|.
             진짜 다음 블록을 같은 방법으로 분해한 값이 대조 (true_*). 움직이는 visible clip, 이동 ≥ 0.75 토큰/블록 칸만. 걸음 0 · 3
  poserr     미래 물체 위치 오차 (토큰). 예측 블록에서 정답 물체 모양과 cosine 최대 칸 ↔ 실제 위치. 걸음 1 · 3 · 7. copy = 마지막 문맥 블록
  late_obj   VOBench late 쌍 정확도, 물체 자리 토큰만 (라벨 A · B 3×3 합집합) / late_full = 화면 전체. copy = 마지막 문맥 블록 복사
지원: kind=ar (block 입력, gating 포함) · kind=ctx_state (encoder 영상 문맥 또는 block 문맥, block 타깃). ar_input=video 는 건너뛴다 (정답 공간이 다름).
"""
import argparse, csv, json, os, sys, time, collections
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
import yaml
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint, _clean_backbone_key   # noqa: E402
from src.models.rollout_predictor import vit_prefix_predictor                                                  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("ckpts", nargs="*", default=["latest.pt"])
ap.add_argument("--vobench", default="/workspace/dataset/benchmarks/VOBench_v1")
ap.add_argument("--n_move", type=int, default=30)
ap.add_argument("--n_late", type=int, default=24)
a = ap.parse_args()
dev = "cuda"
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
S, C, K, G = 256, 8, 8, 16
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
IDX = list(range(0, 96, 3))[:32]
RUN = os.path.join(REPO, "z_training/runs", a.run)
cfg = yaml.safe_load(open(os.path.join(RUN, "config.yaml")))
pc = cfg["model"].get("predictor") or {}
kind = pc.get("kind", "oneshot")
if kind not in ("ar", "ctx_state") or pc.get("ar_input", "block") == "video":
    print(f"측정판: {a.run} (kind={kind}, ar_input={pc.get('ar_input')}) 는 지원하지 않는다 — 건너뜀"); sys.exit(0)

state = _load_checkpoint(CK)
def enc_build(key, nf):
    e = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=nf, use_rope=True, uniform_power=False)
    _load_state_dict(e, state[key], tag=key); return e.to(dev).eval()
tenc = enc_build("target_encoder", 2)
ctx_video = kind == "ctx_state" and pc.get("ctx_input", "video") == "video"
cenc = enc_build(pc.get("ctx_encoder") or cfg["model"].get("context_encoder_key", "encoder"), 2 * C) if ctx_video else None

def load(d):
    fr = [np.asarray(Image.open(f"{d}/{i:06d}.png").convert("RGB")) for i in IDX]
    x = torch.from_numpy(np.stack(fr)).permute(3, 0, 1, 2).float() / 255.0
    W0 = x.shape[-1]
    x = F.interpolate(x.transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False).transpose(0, 1)
    L = json.load(open(os.path.join(os.path.dirname(d), "label.json")))["labels"]
    pos = {}
    for key in ("A", "B"):
        lab = {q["f"]: q for q in L.get(key, [])}
        P, Hd = [], []
        for b in range(16):
            q = [lab.get(IDX[2 * b]), lab.get(IDX[2 * b + 1])]
            if any(t is None for t in q):
                P.append(None); Hd.append(1.0); continue
            P.append((np.mean([t["v"] for t in q]) / W0 * G, np.mean([t["u"] for t in q]) / W0 * G)); Hd.append(np.mean([t.get("hidden", 0) for t in q]))
        pos[key] = (P, Hd)
    return ((x - MEAN) / STD), pos

@torch.no_grad()
def blocks(x):
    xb = x.reshape(3, 16, 2, 256, 256).permute(1, 0, 2, 3, 4).to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = torch.cat([tenc(xb[i:i + 8]) for i in range(0, 16, 8)], 0).float()
    return F.layer_norm(h, (1280,))                                             # (16, S, D)

def rc(p): return int(np.clip(p[0], 0, G - 1)), int(np.clip(p[1], 0, G - 1))
def region(p):
    r0, c0 = rc(p)
    return [rr * G + cc for rr in range(max(0, r0 - 1), min(G, r0 + 2)) for cc in range(max(0, c0 - 1), min(G, c0 + 2))]

# ---- 고정 샘플 (seed 0) ----------------------------------------------------------------------------
rows = list(csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv"))))
mv = [r for r in rows if r["variant"] == "pos_a" and r["motion"] != "flat_static" and r["occlusion"] == "visible"]
mv = list(np.random.default_rng(0).permutation(mv))[: a.n_move]
blk = collections.defaultdict(dict)
for r in rows:
    if r["occlusion"] == "late":
        blk[r["block_id"]][r["variant"]] = r
late_ids = sorted(blk)
late_ids = [late_ids[i] for i in np.random.default_rng(0).permutation(len(late_ids))[: a.n_late]]
t0 = time.time()
MV = []
for r in mv:
    x, pos = load(os.path.join(a.vobench, r["file_name"])); MV.append((x, pos, blocks(x).half().cpu()))
LT = []
for b in late_ids:
    arms = {v: load(os.path.join(a.vobench, blk[b][v]["file_name"])) for v in ("pos_a", "imp_ab", "pos_b", "imp_ba")}
    LT.append({v: (xx, pp, blocks(xx).half().cpu()) for v, (xx, pp) in arms.items()})
print(f"샘플 인코딩 {time.time() - t0:.0f}s (움직임 {len(MV)} clip, late {len(LT)} block)", flush=True)

# ---- predictor ----------------------------------------------------------------------------------
def build_pred(ck):
    kw = dict(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384, depth=12,
              num_heads=12, num_mask_tokens=10, mask_mode=kind)
    if kind == "ar":
        kw["ar_feedback"] = pc.get("ar_feedback", "output")
    for k in ("gating_residual", "gating_anchor", "gating_out_ln"):
        if pc.get(k) is not None:
            kw[k] = pc[k]
    m = vit_prefix_predictor(**kw)
    sd = {k: v for k, v in _clean_backbone_key(torch.load(os.path.join(RUN, ck), map_location="cpu", weights_only=False)["predictor"]).items()
          if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}
    m.load_state_dict(sd, strict=True); return m.to(dev).eval()

@torch.no_grad()
def predict(m, x, H):
    """H (16, S, D) 진짜 블록 (cpu half). return (K, S, D) 블록 공간 예측."""
    Hd = H.to(dev).float(); idx = torch.arange(C * S, device=dev).unsqueeze(0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if kind == "ar":
            p = m.rollout_kv(Hd[:C].reshape(1, C * S, -1), idx, K)
        else:
            z = cenc(x[:, : 2 * C].unsqueeze(0).to(dev)).float() if ctx_video else Hd[:C].reshape(1, C * S, -1)
            A = Hd[C - 1].unsqueeze(0) if getattr(m, "gating_residual", False) else None
            p = m.render(m.rollout_state_kv(z, idx, K), A)
    return p.float()[0].view(K, S, -1)

def decomp(prev, y, i):
    r0, c0 = divmod(int(i), G)
    if r0 < 2 or c0 < 2 or r0 >= G - 2 or c0 >= G - 2:
        return None
    nb = [(r0 + dr) * G + (c0 + dc) for dr in range(-2, 3) for dc in range(-2, 3)]
    w = torch.linalg.lstsq(prev[nb].T, y.unsqueeze(1)).solution.flatten()
    return w.view(5, 5).cpu().numpy()

def transport_stats(get_y):
    out = {}
    for k in (0, 3):
        same, src, conc = [], [], []
        for x, pos, H in MV:
            P, Hd = pos["A"]; b = C + k
            if P[b] is None or P[b - 1] is None or Hd[b] >= 0.5 or Hd[b - 1] >= 0.5:
                continue
            d = np.array(P[b]) - np.array(P[b - 1])
            if np.hypot(*d) < 0.75:
                continue
            Hd_ = H.to(dev).float(); y_all = get_y(x, H, k)
            sr, sc = 2 - min(2, int(round(abs(d[0])))), 2 - min(2, int(round(abs(d[1]))))
            for i in region(P[b]):
                w = decomp(Hd_[b - 1], y_all[i], i)
                if w is None:
                    continue
                if d[0] < 0: w = w[::-1, :]
                if d[1] < 0: w = w[:, ::-1]
                same.append(w[2, 2]); src.append(w[sr, sc]); conc.append(np.abs(w).max() / np.abs(w).sum())
        out[f"s{k}"] = {"same": float(np.mean(same)), "src": float(np.mean(src)), "conc": float(np.mean(conc)), "n": len(same)}
    return out

def poserr(get_blk):
    E = collections.defaultdict(list)
    for x, pos, H in MV:
        P, Hd = pos["A"]; Hd_ = H.to(dev).float()
        for k in (0, 2, 6):
            b = C + k
            if P[b] is None or Hd[b] >= 0.5:
                continue
            tm = Hd_[b][region(P[b])].mean(0)
            c = F.cosine_similarity(get_blk(x, H, k), tm.unsqueeze(0), dim=-1); i = int(c.argmax())
            E[k + 1].append(float(np.hypot(i // G - P[b][0], i % G - P[b][1])))
    return {f"k{k}": float(np.mean(v)) for k, v in sorted(E.items())}

def late_acc(pred_fn):
    obj, full = [], []
    for arms in LT:
        for pv, iv in (("pos_a", "imp_ab"), ("pos_b", "imp_ba")):
            x, pos, H = arms[pv]; Hi = arms[iv][2].to(dev).float(); Hp = H.to(dev).float()
            p = pred_fn(x, H)                                                     # (K, S, D)
            toks = []
            for kk in range(K):
                s_ = set()
                for key in ("A", "B"):
                    P, _ = pos[key]
                    if P[C + kk] is not None:
                        s_ |= set(region(P[C + kk]))
                toks.append(sorted(s_))
            ep = (p - Hp[C:]).abs().mean(-1); ei = (p - Hi[C:]).abs().mean(-1)          # (K, S)
            full.append(float(ei.mean() > ep.mean()))
            mo = torch.zeros(K, S, dtype=torch.bool, device=dev)
            for kk, t in enumerate(toks):
                if t: mo[kk, t] = True
            if mo.any():
                obj.append(float(ei[mo].mean() > ep[mo].mean()))
    return {"obj": 100 * float(np.mean(obj)), "full": 100 * float(np.mean(full)), "n": len(full)}

# 진짜 · 복사 (모델과 무관, 한 번)
true_tr = transport_stats(lambda x, H, k: H.to(dev).float()[C + k])
copy_pos = poserr(lambda x, H, k: H.to(dev).float()[C - 1])
copy_late = late_acc(lambda x, H: H.to(dev).float()[C - 1].unsqueeze(0).expand(K, S, -1))
out_f = open(os.path.join(RUN, "panel.jsonl"), "a")
for ck in a.ckpts:
    if not os.path.isfile(os.path.join(RUN, ck)):
        print("없음:", ck); continue
    t1 = time.time(); m = build_pred(ck)
    ep = torch.load(os.path.join(RUN, ck), map_location="cpu", weights_only=False).get("epoch")
    cache = {}
    def pfn(x, H):
        key = id(H)
        if key not in cache:
            cache[key] = predict(m, x, H)
        return cache[key]
    rec = {"run": a.run, "ckpt": ck, "epoch": ep, "time": time.strftime("%m-%d %H:%M"),
           "transport": transport_stats(lambda x, H, k: pfn(x, H)[k]), "transport_true": true_tr,
           "poserr": poserr(lambda x, H, k: pfn(x, H)[k]), "poserr_copy": copy_pos,
           "late": late_acc(pfn), "late_copy": copy_late}
    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n"); out_f.flush()
    t, tt = rec["transport"], true_tr
    print(f"{a.run} {ck} (epoch {ep}, {time.time() - t1:.0f}s)\n"
          f"  운반 걸음0 원래칸 {t['s0']['src']:+.2f} (진짜 {tt['s0']['src']:+.2f}) · 같은칸 {t['s0']['same']:+.2f} ({tt['s0']['same']:+.2f}) · 집중 {t['s0']['conc']:.2f} ({tt['s0']['conc']:.2f})"
          f" | 걸음3 원래칸 {t['s3']['src']:+.2f} ({tt['s3']['src']:+.2f}) 집중 {t['s3']['conc']:.2f}\n"
          f"  위치 오차 걸음 1·3·7 " + " ".join(f"{v:.2f}" for v in rec["poserr"].values()) + "  (복사 " + " ".join(f"{v:.2f}" for v in copy_pos.values()) + ")\n"
          f"  late 물체 자리 {rec['late']['obj']:.1f} (복사 {copy_late['obj']:.1f}) · 화면 전체 {rec['late']['full']:.1f} (복사 {copy_late['full']:.1f})", flush=True)
    del m; torch.cuda.empty_cache()
