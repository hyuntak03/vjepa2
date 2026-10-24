#!/usr/bin/env python3
"""[P] 영상 문맥 모델 vs 기준 AR — 자연 영상에서도 더 옮기나 (출력 · 채점, 학습 없음) (auto_research, 2026-10-22 KST).

  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=5 /venv/vjepa/bin/python -W ignore auto_research/scripts/ctxt_compare.py

사전 등록: 진행판 [P] KST 05:05. forward 결과는 [S] 와 공유: /workspace/auto_research_cache/ctxt_compare/.
기준 AR = natural_ar_vw_state latest (블록별 LN(target_encoder) 문맥, TF 한 걸음 = forward_seq).
영상 문맥 = natural_ar_vw_state_ctxt_video latest (kind ctx_state, ctx_target block): z = online encoder(창 전체, masks = 문맥 토큰),
  TF = forward_state(z, embed_obs(관측 블록), idx, C) → render. ctxt_vs_block_mechanism.py 와 같은 호출.
자연 · 3D 합성 (블록 8 창): 예측 블록 4..7. 기준 = 블록 0..b → b+1. 영상 문맥 = z(블록 0..3) + 관측 블록 4..b.
  존재감 · 떨어짐 (vob_compare 와 같은 정의, m = 블록 0..b 같은 칸 평균), 들어오는 칸 = |U[b+1]| ≥ 0.5 칸, 정지 = |U[b+1]|, |U[b]| < 0.25.
  조건화 기울기 = [T] ft_eval_reliability 회귀 그대로 (t = 4, 5, 6, 안정도 경계 고정). L1 = 블록 4..7 (전체 · 정지 칸).
VOBench 끝 가림: 문맥 8 블록 → 굴린 예측 8 블록 (기준 rollout_kv, 영상 문맥 render(rollout_state_kv)). 정확도 · 존재감은 vob_compare 와 같다.
출력: auto_research/results/ctxt_compare.json."""
import collections, csv, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F, yaml
import decord

REPO = "/workspace/world_model/vjepa2"
sys.path.insert(0, os.path.join(REPO, "auto_research/scripts")); sys.path.insert(0, REPO)
_argv = list(sys.argv); sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT  # noqa: E402
sys.argv = _argv
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint, _clean_backbone_key  # noqa: E402
from app.vjepa_frozen import utils as UU  # noqa: E402
from app.vjepa_frozen import ar as AR  # noqa: E402
torch.set_num_threads(8)
dev = torch.device("cuda")
S, G = 256, 16
OUTD = "/workspace/auto_research_cache/ctxt_compare"; os.makedirs(OUTD, exist_ok=True)
OUT = os.path.join(REPO, "auto_research/results/ctxt_compare.json")
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
sd = _load_checkpoint(CK)
onl = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=32, use_rope=True, uniform_power=False)
_load_state_dict(onl, sd["encoder"], tag="o"); onl = onl.to(dev, torch.bfloat16).eval()
tgt = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(tgt, sd["target_encoder"], tag="t"); tgt = tgt.to(dev, torch.bfloat16).eval()
del sd
base = FT.load_pred("natural_ar_vw_state", "latest.pt").to(dev).eval()
D_ = os.path.join(REPO, "z_training/runs/natural_ar_vw_state_ctxt_video")
cfg = yaml.safe_load(open(os.path.join(D_, "config.yaml")))
ctxt = UU.build_predictor(cfg["model"], 1280, None, torch.device("cpu"), int(cfg["data"].get("n_frames", 64)))
ctxt.load_state_dict({k: v for k, v in _clean_backbone_key(torch.load(os.path.join(D_, "latest.pt"), map_location="cpu", weights_only=False)["predictor"]).items()
                      if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}, strict=True)
ctxt = ctxt.to(dev).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(3, 1, 1, 1)
EDGES = np.array([0.35, 0.449, 0.557, 0.724])
cy, cx = np.divmod(np.arange(S), G); P0 = np.stack([cx + 0.5, cy + 0.5], 1)


def cell_of(p):
    s = np.floor(p).astype(int); ok = (s >= 0).all(1) & (s < G).all(1)
    return np.where(ok, np.clip(s[:, 1], 0, G - 1) * G + np.clip(s[:, 0], 0, G - 1), np.arange(S)), ok


def norm_pix(fr):                                                              # (T,H,W,3) uint8 → (3,T,256,256) 정규화
    x = torch.from_numpy(np.asarray(fr)).to(dev).permute(3, 0, 1, 2).float() / 255.0
    return (x - MEAN) / STD


def decode_k710(m, n=16):
    vr = decord.VideoReader(m["path"], num_threads=1)
    idx = [m["start"] + m["fstp"] * k for k in range(n)]
    if idx[-1] >= len(vr):
        return None
    fr = vr.get_batch(idx).asnumpy()
    Hh, Ww = fr.shape[1:3]; side = min(Hh, Ww); y0, x0 = (Hh - side) // 2, (Ww - side) // 2
    xx = torch.from_numpy(fr[:, y0:y0 + side, x0:x0 + side]).to(dev).permute(0, 3, 1, 2).float()
    xx = F.interpolate(xx, size=(256, 256), mode="bilinear", align_corners=False).clamp(0, 255).round() / 255.0
    return (xx.permute(1, 0, 2, 3) - MEAN) / STD


def windows(src):
    if src == "k710_holdout":
        Hn = np.load("/workspace/auto_research_cache/natural_holdout/k710_320_holdout_h.npy", mmap_mode="r")
        Un = np.load("/workspace/auto_research_cache/natural_holdout/k710_320_holdout_U.npy", mmap_mode="r")
        meta = json.load(open("/workspace/auto_research_cache/natural_holdout/k710_320_holdout_meta.json"))
        for w in range(Hn.shape[0]):
            x = decode_k710(meta[w])
            if x is None:
                continue
            yield w, np.asarray(Hn[w], np.float32), np.asarray(Un[w], np.float32).reshape(8, S, 2), x
    else:
        EN = "/workspace/auto_research_cache/enc_noise"
        Un = np.load(os.path.join(REPO, f"auto_research/results/natural_transport_ctx5/rows_{src}.npz"))["Uwin"].astype(np.float32)
        w0 = 0
        for sh in (0, 1):
            T_ = np.load(os.path.join(EN, f"{src}_target_encoder_s{sh}_tok.npy"), mmap_mode="r")
            PX = np.load(os.path.join(EN, f"{src}_target_encoder_s{sh}_pix.npy"), mmap_mode="r")
            for w in range(T_.shape[0]):
                yield w0 + w, np.asarray(T_[w, 0], np.float32), Un[w0 + w].reshape(8, S, 2), norm_pix(PX[w])
            w0 += T_.shape[0]


def add(sums, key, p, y, m, hl):
    if p.shape[0] == 0:
        return
    a = sums.setdefault(key, [0.0] * 7)
    a[0] += float((p - m).abs().sum()); a[1] += float((y - m).abs().sum()); a[2] += float(((p - m) * (y - m)).sum()); a[3] += float(((y - m) ** 2).sum())
    a[4] += float((p - hl).abs().sum()); a[5] += float((y - hl).abs().sum()); a[6] += p.shape[0]


def stats(sums):
    return {g: dict(dev=round(v[0] / max(v[1], 1e-9), 3), presence=round(v[2] / max(v[3], 1e-9), 3), vs_copy=round(v[4] / max(v[5], 1e-9), 3), n=int(v[6])) for g, v in sums.items()}


result = {}
t0 = time.time()
MODELS = ("base", "ctxt")
for src in ("k710_holdout", "ht100m", "phys3d"):
    SUM = {n: {} for n in MODELS}; ST = {n: [] for n in MODELS}; L1 = {n: [0.0, 0, 0.0, 0] for n in MODELS}
    preds = {n: [] for n in MODELS}; gts, ctxs, wids = [], [], []
    chk = []
    for k, (wid, hn, U, x) in enumerate(windows(src)):
        h = torch.from_numpy(hn).to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            if src == "k710_holdout" and k < 6:                                    # 다시 디코딩한 프레임이 캐시 토큰과 같은가
                hb = AR.encode_blocks(tgt, x[None].to(torch.bfloat16), 2).float()[0].view(8, S, -1)
                chk.append(float(F.cosine_similarity(hb, h, dim=-1).mean()))
            Pb = base.forward_seq(h[:7].reshape(1, -1, 1280), torch.arange(7 * S, device=dev).unsqueeze(0)).float()[0].view(7, S, -1)[3:7]   # 블록 4..7
            z = onl(x[None].to(torch.bfloat16), masks=[torch.arange(4 * S, device=dev).unsqueeze(0)]).float()
            st_ = ctxt.forward_state(z, ctxt.embed_obs(h[4:7].reshape(1, -1, 1280)), torch.arange(7 * S, device=dev).unsqueeze(0), 4)
            Pc = ctxt.render(st_).float()[0].view(4, S, -1)
        P = {"base": Pb, "ctxt": Pc}
        for n in MODELS:
            preds[n].append(P[n].half().cpu().numpy())
        gts.append(h[4:8].half().cpu().numpy()); ctxs.append(h[:4].half().cpu().numpy()); wids.append(int(wid))
        for i, b in enumerate(range(3, 7)):                                        # 예측 블록 b+1 = 4..7
            mctx = h[: b + 1].mean(0); y = h[b + 1]; hl = h[b]
            u1 = np.linalg.norm(U[b + 1], axis=1); u0 = np.linalg.norm(U[b], axis=1)
            ent = torch.from_numpy(np.where(u1 >= 0.5)[0]).to(dev); stc = torch.from_numpy(np.where((u1 < 0.25) & (u0 < 0.25))[0]).to(dev)
            for n in MODELS:
                p = P[n][i]
                add(SUM[n], "enter", p[ent], y[ent], mctx[ent], hl[ent]); add(SUM[n], "static", p[stc], y[stc], mctx[stc], hl[stc])
                e = (p - y).abs().mean(-1); L1[n][0] += float(e.mean()); L1[n][1] += 1
                if len(stc):
                    L1[n][2] += float(e[stc].mean()); L1[n][3] += 1
        for t in (4, 5, 6):                                                        # [T] 회귀 그대로 (예측 블록 t+1)
            u1, u0 = U[t + 1], U[t]
            mov = np.linalg.norm(u1, axis=1) >= 0.5
            if mov.sum() == 0:
                continue
            js, ok1 = cell_of(P0 + u1); jp, ok0 = cell_of(P0[js] + u0[js])
            sel = np.where(mov & ok1 & ok0)[0]
            if sel.size == 0:
                continue
            c = torch.from_numpy(sel).to(dev)
            x1 = h[t - 4:t + 1].mean(0)[c]; x2 = h[t][torch.from_numpy(js[sel]).to(dev)]; y = h[t + 1][c]
            stab = F.cosine_similarity(x2, h[t - 1][torch.from_numpy(jp[sel]).to(dev)], dim=-1)
            base_cols = [(x1 * x1).sum(-1), (x1 * x2).sum(-1), (x2 * x2).sum(-1), (x1 * y).sum(-1), (x2 * y).sum(-1), (y * y).sum(-1), stab]
            for n in MODELS:
                ym = P[n][t - 3][c]
                ST[n].append(torch.stack(base_cols + [(x1 * ym).sum(-1), (x2 * ym).sum(-1)], -1).double().cpu().numpy())
        if k % 40 == 0:
            print(f"{src} {k} ({time.time() - t0:.0f}s) {'일치 cos ' + str(np.round(chk, 4)) if chk else ''}", flush=True)
    r = {}
    for n in MODELS:
        st = np.concatenate(ST[n]); bins = np.digitize(st[:, 6], EDGES)
        def fit(mk, yi):
            A = np.array([[st[mk, 0].sum(), st[mk, 1].sum()], [st[mk, 1].sum(), st[mk, 2].sum()]])
            return np.linalg.solve(A, [st[mk, yi[0]].sum(), st[mk, yi[1]].sum()])
        allm = np.ones(len(st), bool); bg = fit(allm, (3, 4)); mg = fit(allm, (7, 8))
        rows = [{"n": int((bins == k_).sum()), "b_data": float(fit(bins == k_, (3, 4))[1]), "b_model": float(fit(bins == k_, (7, 8))[1])} for k_ in range(5)]
        r[n] = dict(out=stats(SUM[n]), cond=dict(bins=rows, slope_data=rows[4]["b_data"] - rows[0]["b_data"], slope_model=rows[4]["b_model"] - rows[0]["b_model"],
                                                b_model_all=float(mg[1]), b_data_all=float(bg[1])),
                    l1_all=L1[n][0] / max(L1[n][1], 1), l1_static=L1[n][2] / max(L1[n][3], 1))
    r["check_cos"] = chk
    result[src] = r
    for n in MODELS:
        np.save(os.path.join(OUTD, f"{src}_pred_{n}.npy"), np.stack(preds[n]))
    np.save(os.path.join(OUTD, f"{src}_gt.npy"), np.stack(gts)); np.save(os.path.join(OUTD, f"{src}_ctx.npy"), np.stack(ctxs))
    json.dump(dict(windows=wids, flow=("natural_holdout/k710_320_holdout_U.npy" if src == "k710_holdout" else f"results/natural_transport_ctx5/rows_{src}.npz:Uwin"),
                   blocks="예측 블록 4..7", ctx="블록 0..3"), open(os.path.join(OUTD, f"{src}_meta.json"), "w"), ensure_ascii=False)
    open(os.path.join(OUTD, f"{src}.done"), "w").write("ok\n")
    print(src, json.dumps({n: {k_: v for k_, v in r[n].items()} for n in MODELS}, ensure_ascii=False)[:1500], f"({time.time() - t0:.0f}s)", flush=True)
    json.dump(result, open(OUT, "w"), indent=1, ensure_ascii=False)

# ------------------------------------------------------------------ VOBench 끝 가림
from PIL import Image
VB = "/workspace/dataset/benchmarks/VOBench_v1"
IDX = list(range(0, 96, 3))[:32]
C, K = 8, 8
idx_rows = collections.defaultdict(dict)
for r_ in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv"))):
    if r_["occlusion"] == "late":
        idx_rows[r_["block_id"]][r_["variant"]] = r_
held = {r_["source_block"] for r_ in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1_heldout/index.csv")))}
meta = json.load(open("/workspace/auto_research_cache/vobench_late/meta.json"))
H = np.load("/workspace/auto_research_cache/vobench_late/h.npy", mmap_mode="r")
ARMS = ["pos_a", "pos_b", "imp_ab", "imp_ba"]


def frames(d, idx):
    fr = [np.asarray(Image.open(f"{d}/{i:06d}.png").convert("RGB")) for i in idx]
    x = torch.from_numpy(np.stack(fr)).permute(3, 0, 1, 2).float() / 255.0
    W0 = x.shape[-1]
    x = F.interpolate(x.transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False).transpose(0, 1)
    return ((x - MEAN.cpu()) / STD.cpu()).to(dev), W0


def centers(label_path, W0):
    L_ = json.load(open(label_path))["labels"]; out = {}
    for key in ("A", "B"):
        lab = {q["f"]: q for q in L_.get(key, [])}; cs = []
        for b in range(16):
            q = [lab.get(IDX[2 * b]), lab.get(IDX[2 * b + 1])]
            if any(t is None for t in q):
                cs.append(None); continue
            r0 = int(np.clip(np.mean([t["v"] for t in q]) / W0 * G, 0, G - 1)); c0 = int(np.clip(np.mean([t["u"] for t in q]) / W0 * G, 0, G - 1))
            cs.append((r0, c0, float(np.mean([t["hidden"] for t in q]))))
        out[key] = cs
    return out


def region(rc, rad=1):
    r0, c0 = rc[:2]
    return [rr * G + cc for rr in range(max(0, r0 - rad), min(G, r0 + rad + 1)) for cc in range(max(0, c0 - rad), min(G, c0 + rad + 1))]


recs = []; VSUM = {n: {} for n in MODELS}
VP = {n: np.lib.format.open_memmap(os.path.join(OUTD, f"vob_pred_{n}.npy"), mode="w+", dtype=np.float16, shape=(len(meta), 2, K, S, 1280)) for n in MODELS}
for bi, m in enumerate(meta):
    arms = idx_rows[m["block_id"]]
    d0 = os.path.join(VB, arms["pos_a"]["file_name"])
    hh = {v: torch.from_numpy(np.asarray(H[m["i"], k], dtype=np.float32)).to(dev) for k, v in enumerate(ARMS)}
    W0 = Image.open(f"{d0}/{IDX[0]:06d}.png").size[0]
    cn = centers(os.path.join(os.path.dirname(d0), "label.json"), W0)
    allobj = set()
    for key in ("A", "B"):
        for c_ in cn[key]:
            if c_ is not None:
                allobj |= set(region(c_))
    otok = []
    for j in range(K):
        s_ = set()
        for key in ("A", "B"):
            if cn[key][C + j] is not None:
                s_ |= set(region(cn[key][C + j]))
        otok.append(sorted(s_))
    oidx = torch.tensor([j * S + t for j in range(K) for t in otok[j]], device=dev)
    mv = "static" if arms["pos_a"]["motion"] == "flat_static" else "moving"
    viol = arms["pos_a"]["violation"]; heldout = arms["pos_a"]["source_block"] in held
    empty_b = arms["pos_a"].get("obj_b", "") == "empty"
    for ai, (pid, (pos, imp, key)) in enumerate((("1", ("pos_a", "imp_ab", "A")), ("2", ("pos_b", "imp_ba", "B")))):
        ctx = hh[pos][:C]; mctx = ctx.mean(0); hl = ctx[-1]
        x, _ = frames(os.path.join(VB, arms[pos]["file_name"]), IDX)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pb = base.rollout_kv(ctx.reshape(1, C * S, -1), torch.arange(C * S, device=dev).unsqueeze(0), K).float()[0].view(K, S, -1)
            z = onl(x[None].to(torch.bfloat16), masks=[torch.arange(C * S, device=dev).unsqueeze(0)]).float()
            pc = ctxt.render(ctxt.rollout_state_kv(z, torch.arange(C * S, device=dev).unsqueeze(0), K)).float()[0].view(K, S, -1)
        for n, p in (("base", pb), ("ctxt", pc)):
            VP[n][bi, ai] = p.half().cpu().numpy()
            sc = {}
            for v in (pos, imp):
                e = (p.reshape(K * S, -1) - hh[v][C:].reshape(K * S, -1)).abs().mean(-1)
                sc[v] = (float(e.mean()), float(e[oidx].mean()) if len(oidx) else float("nan"))
            direction = ("vanish" if pid == "1" else "appear") if (viol == "permanence" and empty_b) else f"pair{pid}"
            recs.append(dict(block=m["block_id"], model=n, pair=pid, mv=mv, viol=viol, held=heldout, dir=direction, full=sc[imp][0] > sc[pos][0], obj=sc[imp][1] > sc[pos][1]))
            y = hh[pos][C:]
            if not (viol == "permanence" and empty_b and key == "B"):
                for j in range(K):
                    c_ = cn[key][C + j]
                    if c_ is None or c_[2] >= 0.5:
                        continue
                    cells = torch.tensor(region(c_), device=dev)
                    add(VSUM[n], f"{mv}|enter|{'held' if heldout else 'train'}", p[j, cells], y[j, cells], mctx[cells], hl[cells])
    if bi % 40 == 0:
        print(f"VOBench {bi}/{len(meta)} ({time.time() - t0:.0f}s)", flush=True)
for n in MODELS:
    VP[n].flush()
open(os.path.join(OUTD, "vob.done"), "w").write("ok\n")
acc = {}
for n in MODELS:
    for split in ("held", "train"):
        for g, fn in (("moving", lambda r: r["mv"] == "moving"), ("static", lambda r: r["mv"] == "static"),
                      ("moving|vanish", lambda r: r["mv"] == "moving" and r["dir"] == "vanish"), ("moving|appear", lambda r: r["mv"] == "moving" and r["dir"] == "appear"),
                      ("moving|shape", lambda r: r["mv"] == "moving" and r["viol"] == "shape"), ("moving|color", lambda r: r["mv"] == "moving" and r["viol"] == "color")):
            sel = [r for r in recs if r["model"] == n and r["held"] == (split == "held") and fn(r)]
            if sel:
                acc[f"{n}|{split}|{g}"] = dict(n=len(sel), obj=round(100 * np.mean([r["obj"] for r in sel]), 1), full=round(100 * np.mean([r["full"] for r in sel]), 1))
result["vobench_acc"] = acc
result["vobench_out"] = {n: stats(VSUM[n]) for n in MODELS}
print(json.dumps(result["vobench_acc"], ensure_ascii=False), flush=True)
print(json.dumps(result["vobench_out"], ensure_ascii=False), flush=True)
json.dump(result, open(OUT, "w"), indent=1, ensure_ascii=False)
json.dump(recs, open(OUT.replace(".json", "_vob_recs.json"), "w"))
print("저장", OUT, f"({time.time() - t0:.0f}s)")
