#!/usr/bin/env python3
"""운동 adapter 를 학습 없이 릴리즈 predictor 에 꽂으면 학습 밖 영속 (특히 움직이는 물체) 이 오르나 (2026-11-01, 사용자 "해봐").

왜: "학습 때 모듈 → 증류" method 의 전제 — 운동으로 옮기는 구조가 학습 밖에서 움직이는 물체를 돕는가 — 를 학습 없이 먼저 가른다.
    안 오르면 이 선생 방향을 접는다. 오르면 선생 학습 → 자리 가중 증류로 간다.
adapter (학습 없음; 물체다움 선형 하나만 1 판 0 단계의 것)
  ① 물체다움 o_t(j) = σ((w·LN(z_t(j)) + b − τ) / κ)                       (BlenderTrack v1 으로 학습한 선형, τ · κ 그대로)
  ② 속도 v_t(j) = feature 대응 변위 / 간격 의 중앙값 (앞 · 뒤, 간격 1–3, 창 반경 3, 포물선 보간)
  ③ 옮기기: 미래 칸 τ 마다 문맥 토큰 (t, j) 의 feature z_t(j) 를 자리 j + v_t(j)·(τ − t) 로 이중선형 분배.
     무게 a = o_t(j) · exp(−(7 − t)/1.5)  (가려지기 직전, 마지막으로 보인 물체 feature 가 이긴다).
     도착 칸 r 의 옮긴 feature W_τ(r) = Σ a·z / Σ a, 문 g_τ(r) = min(1, Σ a)
  ④ 주입: 미래 query 시작값 = (1 − α·g) · mask token + α·g · predictor_embed(W)   (predictor 입력층 그대로, 새 파라미터 없음)
  α ∈ ALPHAS. α = 0 은 릴리즈와 같다 (점검: l1 이 릴리즈 열과 같아야 함).
채점 = 표준 (te 엔진, fp32 + autocast fp16, L1, LN 목표, matched 쌍) + 복사 기준선. 패널 · 쌍 고르기는 predictor_lab/v3_ablation (eval_v3.sh · table_v3.py) 와 같다.

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P predictor_lab/scaffold/adapter_probe.py run [패널 …]      # GPU 1 장 → predictor_lab/scaffold/results/adapter_probe/<패널>/
  $P predictor_lab/scaffold/adapter_probe.py summary            # CPU → predictor_lab/scaffold/results/adapter_probe/summary.json
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
OUT = REPO / "predictor_lab/scaffold/results/adapter_probe"
RES = REPO / "predictor_lab/scaffold/results"
NFS = REPO / "z_pixel_decoder/runs/_frames"
ALPHAS = [0.25, 0.5, 1.0]
G, C, S = 16, 8, 256
R_WIN = 3
PANELS = {   # eval_v3.sh 와 같은 데이터 · 창
    "vob_late": ("vobench_v1_split2_test", ["data.root=data_csv/transfer_gap_r4_vob", "data.index_csv=index_test_late.csv", f"data.frames_root={NFS}/VOBench_v1_256full"]),
    "vomo": ("vo_moving_occluder", [f"data.frames_root={NFS}/VO_moving_occluder_s3"]),
    "ip1uni": ("intphys1_late_uni", [f"data.frames_root={NFS}/tg4_IntPhys1_late_uni"]),
    "ip2uni": ("intphys2_late_uni", [f"data.frames_root={NFS}/IntPhys2_late_uni"]),
    "rubric": ("vobench_v1", [f"data.root={REPO}/data_csv/rubric_quads_uniform", "data.index_csv=index.csv",
                              "data.frames_root=/data/hyuntak/project/2026/2027_cvpr/nfs_cache/rubric_quads_frames/uniform", "window.stride=1", "window.frames_start=0"]),
}


# ----------------------------------------------------------------------------- adapter (GPU)
class Warp(nn.Module):
    def __init__(self):
        super().__init__()
        from app.vjepa_frozen import scaffold as SC
        st = json.load(open(RES / "stage0.json"))["read"]
        self.tau, self.kap = st["tau"], st["kappa"]
        sd = torch.load(RES / "object_score_w.pt", map_location="cpu")
        self.obj = SC.ObjectScore(sd["lin.weight"].shape[1]); self.obj.load_state_dict(sd)
        yy, xx = torch.meshgrid(torch.arange(G), torch.arange(G), indexing="ij")
        self.register_buffer("xy", torch.stack([xx.flatten(), yy.flatten()], -1).float(), persistent=False)   # (256, 2) (열, 행)
        oy, ox = torch.meshgrid(torch.arange(-R_WIN, R_WIN + 1), torch.arange(-R_WIN, R_WIN + 1), indexing="ij")
        ny, nx = yy[..., None, None] + oy, xx[..., None, None] + ox
        self.register_buffer("inside", ((ny >= 0) & (ny < G) & (nx >= 0) & (nx < G)), persistent=False)          # (16,16,W,W)
        self.register_buffer("nid", (ny.clamp(0, G - 1) * G + nx.clamp(0, G - 1)).view(G * G, -1), persistent=False)

    def flow(self, a, b):
        """a, b (N, 256, D) 단위 벡터 → a 의 각 토큰이 b 에서 간 변위 (N, 256, 2) (열, 행), 포물선 보간."""
        N = a.shape[0]; W = 2 * R_WIN + 1
        full = torch.bmm(a, b.transpose(1, 2))
        sims = torch.gather(full, 2, self.nid[None].expand(N, -1, -1)).view(N, G, G, W, W)
        sims = torch.where(self.inside[None], sims, torch.full_like(sims, -1.0)).view(N, G, G, W * W)
        best, idx = sims.max(-1); iy, ix = idx // W, idx % W

        def at(yy, xx):
            return torch.gather(sims, -1, (yy.clamp(0, W - 1) * W + xx.clamp(0, W - 1)).unsqueeze(-1)).squeeze(-1)

        def sub(m, c, p):
            den = m - 2 * c + p
            return torch.where(den.abs() > 1e-6, 0.5 * (m - p) / den, torch.zeros_like(den)).clamp(-0.5, 0.5)
        ox = torch.where((ix > 0) & (ix < W - 1), sub(at(iy, ix - 1), best, at(iy, ix + 1)), torch.zeros_like(best))
        oy = torch.where((iy > 0) & (iy < W - 1), sub(at(iy - 1, ix), best, at(iy + 1, ix)), torch.zeros_like(best))
        return torch.stack([ix.float() - R_WIN + ox, iy.float() - R_WIN + oy], -1).view(N, S, 2)

    @torch.no_grad()
    def forward(self, z):
        """z (B, 8·256, D) 문맥 encoder 출력 → 옮긴 feature W (B, 8·256, D) · 문 g (B, 8·256) (미래 칸 8 × 256, 위치 순)."""
        B, _, D = z.shape
        z = z.float().view(B, C, S, D)
        ln = F.layer_norm(z, (D,))
        o = torch.sigmoid((self.obj.lin(ln).squeeze(-1) - self.tau) / self.kap)                 # (B, 8, 256)
        zn = F.normalize(ln, dim=-1)
        est = torch.full((B, C, S, 6, 2), float("nan"), device=z.device)
        for k in (1, 2, 3):
            d = self.flow(zn[:, :-k].reshape(-1, S, D), zn[:, k:].reshape(-1, S, D)).view(B, C - k, S, 2) / k
            est[:, :C - k, :, k - 1] = d
            db = self.flow(zn[:, k:].reshape(-1, S, D), zn[:, :-k].reshape(-1, S, D)).view(B, C - k, S, 2) / k
            est[:, k:, :, 2 + k] = -db
        v = torch.nanmedian(est, dim=3).values                                                    # (B, 8, 256, 2)
        t = torch.arange(C, device=z.device, dtype=torch.float32)
        a = o * torch.exp(-(C - 1 - t) / 1.5)[None, :, None]                                     # (B, 8, 256)
        Wout = torch.zeros(B, C * S, D, device=z.device); gout = torch.zeros(B, C * S, device=z.device)
        src = z.reshape(B, C * S, D); aw = a.reshape(B, C * S)
        for r in range(C):
            tau = C + r
            y = self.xy[None, None] + v * (tau - t)[None, :, None, None]                        # (B, 8, 256, 2)
            y = y.reshape(B, C * S, 2)
            x0, y0 = torch.floor(y[..., 0]), torch.floor(y[..., 1]); fx, fy = y[..., 0] - x0, y[..., 1] - y0
            feat = torch.zeros(B, S, D, device=z.device); mass = torch.zeros(B, S, device=z.device)
            for dx, dy, wgt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
                cx, cy = x0 + dx, y0 + dy
                ok = (cx >= 0) & (cx < G) & (cy >= 0) & (cy < G) & torch.isfinite(wgt)
                w = torch.where(ok, wgt * aw, torch.zeros_like(wgt))
                di = torch.where(ok, cy * G + cx, torch.zeros_like(cx)).long()
                mass.scatter_add_(1, di, w)
                feat.scatter_add_(1, di[..., None].expand(-1, -1, D), w[..., None] * src)
            Wout[:, r * S:(r + 1) * S] = feat / mass.clamp_min(1e-6)[..., None]
            gout[:, r * S:(r + 1) * S] = mass.clamp(0, 1)
        return Wout, gout


class AdapterPred(nn.Module):
    """릴리즈 predictor 의 forward 를 그대로 따라가되, 미래 query 시작값에 옮긴 feature 를 섞는다 (α)."""

    def __init__(self, base, warp, alpha):
        super().__init__()
        self.base, self.warp, self.alpha = base, warp, float(alpha)

    def forward(self, x, masks_x, masks_y, mask_index=1, has_cls=False):
        from src.masks.utils import apply_masks
        P = self.base
        masks_x = masks_x if isinstance(masks_x, list) else [masks_x]
        masks_y = masks_y if isinstance(masks_y, list) else [masks_y]
        assert len(masks_x) == 1 and not has_cls and P.use_rope and not P.return_all_tokens
        B = len(x)
        my = masks_y[0]
        assert torch.equal(my[0], torch.arange(C * S, 2 * C * S, device=my.device)), "미래 칸 순서가 위치 순이 아니다"
        with torch.autocast("cuda", enabled=False):
            Wf, g = self.warp(x)
        xe = P.predictor_embed(x)
        _, N_ctxt, _ = xe.shape
        pt = P.mask_tokens[mask_index % P.num_mask_tokens].repeat(B, P.num_patches, 1)
        pt = apply_masks(pt, masks_y)
        we = P.predictor_embed(Wf.to(xe.dtype))
        aa = (self.alpha * g).unsqueeze(-1).to(pt.dtype)
        pt = (1 - aa) * pt + aa * we
        xx = torch.cat([xe, pt], dim=1)
        masks = torch.cat([torch.cat(masks_x, 0), torch.cat(masks_y, 0)], dim=1)
        argsort = torch.argsort(masks, dim=1)
        masks = torch.stack([masks[i, row] for i, row in enumerate(argsort)], 0)
        xx = torch.stack([xx[i, row, :] for i, row in enumerate(argsort)], 0)
        if P.chop_last_n_tokens > 0:
            xx = xx[:, :-P.chop_last_n_tokens]; masks = masks[:, :-P.chop_last_n_tokens]
        for blk in P.predictor_blocks:
            xx = blk(xx, mask=masks, attn_mask=None)
        xx = P.predictor_norm(xx)
        rev = torch.argsort(argsort, dim=1)
        xx = torch.stack([xx[i, row, :] for i, row in enumerate(rev)], 0)[:, N_ctxt:]
        return P.predictor_proj(xx)


# ----------------------------------------------------------------------------- 실행 (GPU)
def run(panels):
    import te_v11_score as te
    import transfer_gap_te as tte
    dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
    for pn in panels:
        ds_name, sets = PANELS[pn]
        sets = [(kv.split("=", 1)[0] + "=" + str((REPO / kv.split("=", 1)[1]).resolve())) if kv.startswith("data.root=") and not kv.split("=", 1)[1].startswith("/") else kv
                for kv in sets]
        cfg, _ = tte.resolve(ds_name, "vith", sets, None, 0)
        te.N_SLOT = (int(cfg["data"]["n_frames"]) - int(cfg["surprise"]["context_length"])) // 2
        sc = te.Scorer(cfg, dev, [("release", None)], keep_own=True)
        warp = Warp().to(dev)
        for al in ALPHAS:
            sc.tags.append(f"adapter_a{al}"); sc.mods.append(AdapterPred(sc.bundle.predictor, warp, al).eval()); sc.is_ar.append(False)
        sc.tags.append("adapter_a0_check"); sc.mods.append(AdapterPred(sc.bundle.predictor, warp, 0.0).eval()); sc.is_ar.append(False)
        ds = te.WMADataset(cfg)
        rows = te.select_rows(ds, 0)
        loader = torch.utils.data.DataLoader(te.ClipDS(ds, rows, rows), batch_size=8, num_workers=6, collate_fn=te.collate, multiprocessing_context="fork")
        n, K = len(rows), len(sc.tags)
        l1 = np.full((n, K, te.N_SLOT), np.nan, np.float32); cp = np.full((n, te.N_SLOT), np.nan, np.float32)
        pos = {r: i for i, r in enumerate(rows)}
        t0 = time.time()
        for it, (rr, clips) in enumerate(loader):
            res = sc.run(clips)
            ii = [pos[r] for r in rr]
            l1[ii] = res["l1"]; cp[ii] = res["copy"]
            if it % 20 == 0:
                print(f"[{pn}] {min((it + 1) * 8, n)}/{n} {time.time() - t0:.0f}s", flush=True)
        d = OUT / pn; d.mkdir(parents=True, exist_ok=True)
        np.save(d / "l1.npy", l1); np.save(d / "copy.npy", cp)
        json.dump({"pred_tags": sc.tags, "video_id": [ds.records[i].video_id for i in rows],
                   "index_csv": str(Path(cfg["data"]["root"]) / cfg["data"]["index_csv"]), "dataset": ds_name, "sets": sets, "alphas": ALPHAS},
                  open(d / "meta.json", "w"), ensure_ascii=False)
        chk = np.abs(l1[:, sc.tags.index("adapter_a0_check")] - l1[:, 0]).max()
        print(f"[{pn}] 끝 {time.time() - t0:.0f}s · α=0 점검 max|Δl1| {chk:.2e}", flush=True)
        del sc; torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- 요약 (CPU)
def moving_of(r):
    for k in ("motion", "group"):
        if r.get(k):
            return "static" if "static" in r[k] else "moving"
    return "?"


SEL = {
    "vob_late": [("perm_A_moving", lambda r: r["violation_type"] == "permanence" and r["pair_id"] == "A" and moving_of(r) == "moving"),
                 ("perm_A_static", lambda r: r["violation_type"] == "permanence" and r["pair_id"] == "A" and moving_of(r) == "static"),
                 ("perm_B", lambda r: r["violation_type"] == "permanence" and r["pair_id"] == "B"),
                 ("perm_all", lambda r: r["violation_type"] == "permanence"), ("all_late", lambda r: True)],
    "vomo": [("perm_A_moving", lambda r: r["violation_type"] == "permanence" and r["pair_id"] == "A" and moving_of(r) == "moving"),
             ("perm_A_static", lambda r: r["violation_type"] == "permanence" and r["pair_id"] == "A" and moving_of(r) == "static"),
             ("perm_B", lambda r: r["violation_type"] == "permanence" and r["pair_id"] == "B"),
             ("perm_all", lambda r: r["violation_type"] == "permanence"), ("all", lambda r: True)],
    "ip1uni": [("O1_occluded", lambda r: r["block_type"] == "O1" and r["group"].endswith("/occluded")),
               ("O1_moving_occ", lambda r: r["block_type"] == "O1" and r["group"] == "moving/occluded"),
               ("O1_static_occ", lambda r: r["block_type"] == "O1" and r["group"] == "static/occluded")],
    "ip2uni": [("permanence", lambda r: r["condition"] == "permanence"),
               ("perm_occluder", lambda r: r["condition"] == "permanence" and r["occluder"] not in ("", "None"))],
    "rubric": [("all", lambda r: True), ("occlusion", lambda r: r["condition"] == "occlusion")],
}


def summary():
    out = {}
    for pn in PANELS:
        d = OUT / pn
        if not (d / "meta.json").exists():
            continue
        m = json.load(open(d / "meta.json")); l1 = np.load(d / "l1.npy").mean(-1); cp = np.load(d / "copy.npy").mean(-1)
        vids = m["video_id"]; idx = {r["video_id"]: r for r in csv.DictReader(open(m["index_csv"]))}
        by = defaultdict(dict)
        for i, v in enumerate(vids):
            r = idx[v]; by[(r["block_id"], r["pair_id"])]["pos" if r["plausible"] == "1" else "imp"] = i
        pairs = [(x["pos"], x["imp"], idx[vids[x["pos"]]]) for x in by.values() if "pos" in x and "imp" in x]
        Sx = {"copy": cp, **{t: l1[:, k] for k, t in enumerate(m["pred_tags"]) if t != "adapter_a0_check"}}
        out[pn] = {}
        for name, fn in SEL[pn]:
            sel = [(a, b) for a, b, r in pairs if fn(r)]
            if not sel:
                continue
            row = {"n": len(sel)}
            for t, s in Sx.items():
                h = np.array([1.0 if s[b] > s[a] else 0.5 if s[b] == s[a] else 0.0 for a, b in sel])
                row[t] = round(float(h.mean() * 100), 1)
            if "release" in Sx:
                for t in Sx:
                    if t.startswith("adapter"):
                        hr = np.array([s_ for s_ in [(Sx["release"][b] > Sx["release"][a]) for a, b in sel]])
                        ha = np.array([(Sx[t][b] > Sx[t][a]) for a, b in sel])
                        row[f"{t}_only_vs_release_only"] = [int((ha & ~hr).sum()), int((hr & ~ha).sum())]
            out[pn][name] = row
        print(f"== {pn}")
        for name, row in out[pn].items():
            print(f"  {name:16s} " + " | ".join(f"{k} {v}" for k, v in row.items()))
    json.dump(out, open(OUT / "summary.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "summary"
    if part == "run":
        run(sys.argv[2:] or list(PANELS))
    else:
        summary()
