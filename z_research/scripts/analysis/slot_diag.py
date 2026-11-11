"""slot 메모리 판 진단 (2026-11-07, 사용자 "slot 이 뭘 배우긴 한 건가? 실패 원인 가볍게 진단"). 학습 없음.

A. 벤치 장면 (real 95 · VOBench v3 판 가림 움직임) — ar_memory_readout.py 와 같은 짝 (A 문맥 = 물체를 봄, B 문맥 = 빔).
   개입별로 rollout 을 다시 돌려 시트 점수 (쌍 정확도 A · B) 와 예측 반영도 β − 위약 을 잰다.
     normal  그대로 (per_block.json 의 surprise 를 재현해야 한다 — 산출물 대조값 ref_* 로 확인)
     door0   문을 전부 닫는다 (패치가 slot 을 안 읽는다)
     swap    A 몸통에 B 의 slot, B 몸통에 A 의 slot 을 준다 (같은 장면)
     other   직전 장면의 slot 궤적을 준다 (내용 무관)
     freeze  블록 freeze_from 부터 gate 를 강제로 닫는다 (slot = 전이값, 새 근거 안 받음)
   slot 판별: S_b 의 A − B 차이를 다른 장면 평균 방향으로 판정 (ar_memory_readout.probe_acc, 우연 50 %).
B. 학습 분포 (v4bc4 클립 N 개, 학습에 쓴 클립이라 held-out 아님) — slot 이 무엇을 담는가.
   attention: 토큰별 엔트로피 · 위치만으로 설명되는 몫 · target 토큰 선택도.
   slot 내용: 클립 간 분산 몫 · slot 끼리 cos · 블록 간 변화.
   문 출력: 패치 대비 크기 · 클립에 따라 달라지는 몫.
   기억: 물체가 완전히 가려진 블록에서 마지막으로 본 위치 · 지금 위치 ridge probe (현재 블록 encoder 토큰 평균과 비교, 클립 단위 split).
   rollout L1: 전체 토큰 · target 이 보이는 토큰, normal / door0 / roll (다른 클립 slot).

  CUDA_VISIBLE_DEVICES=0 /venv/vjepa/bin/python z_research/scripts/analysis/slot_diag.py --model vast_ar_vw_state_v4bc4_slot
출력: z_analysis/results/slot_diag_20261107/<model>.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys

os.environ.setdefault("WMA_AR_BLOCK_CACHE", "0")
import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from ar_memory_readout import probe_acc  # noqa: E402
from scene_paired_tests import boards  # noqa: E402
from analysis.intphys2.model import build_from_config  # noqa: E402
from analysis.intphys2.surprise import _context_target_indices  # noqa: E402
from evals.world_model_analysis.data import WMADataset  # noqa: E402

S, C, K = 256, 8, 8
RES = os.path.join(ROOT, "cvpr/results/eval/testbed")
OUT = os.path.join(ROOT, "z_analysis/results/slot_diag_20261107")
BENCH = {"real95": ("real_world_permanence_uniform", "real world (all 95)", None, 6),
         "vob3_panel_moving": ("vobench_v3", "VOBench v3 late · panel", "moving", 7)}     # 마지막 값 = freeze 시작 블록 (encoder 판별이 무너지는 블록)
LIST = os.path.join(ROOT, "data_csv/blender_track_v4bc4/arms/all_mp4list.csv")
ln = lambda t: F.layer_norm(t.float(), (t.size(-1),))  # noqa: E731


class Patch:
    """rollout 안의 slot 사용 지점을 바꾼다. S = 몸통이 받은 slot (블록 순서)."""

    def __init__(self, core, mode="normal", other=None, freeze_from=None, door_rec=None):
        self.core, self.mode, self.other, self.ff, self.rec = core, mode, other, freeze_from, door_rec
        self.S = []

    def __enter__(self):
        c = self.core
        if not getattr(c, "slot_n", 0):
            return self
        orig_kv, orig_step, t_kv, t_mem = c._ar_step_kv_slot, c.slot_mem.step, [0], [0]

        def kv(h, idx_blk, caches, s):
            t = t_kv[0]
            t_kv[0] += 1
            self.S.append(s.detach().float())
            if self.mode == "swap":
                s = s.flip(0)
            elif self.mode == "roll":
                s = s.roll(1, 0)
            elif self.mode == "other":
                s = self.other[t].to(s.dtype)
            return orig_kv(h, idx_blk, caches, s)

        c._ar_step_kv_slot = kv
        if self.mode == "door0":
            c._slot_x = lambda i, h, sl: h
        elif self.rec is not None:
            def xs(i, h, sl):
                B, n = sl.shape[:2]
                D = h.size(-1)
                y = c.slot_xattn[str(i)](h.reshape(B * n, S, D), sl.reshape(B * n, sl.size(2), sl.size(3))).reshape(B, n * S, D)
                yf, hf = y.float(), h.float()
                e2 = float((yf ** 2).sum(-1).mean())
                self.rec.setdefault(i, []).append((float(yf.norm(dim=-1).mean() / hf.norm(dim=-1).mean()),
                                                   float((yf.mean((0, 1)) ** 2).sum()) / e2,                # 어디서나 같은 벡터의 몫
                                                   float(yf.var(0, unbiased=True).sum(-1).mean()) / e2))    # 클립에 따라 달라지는 몫
                return h + y
            c._slot_x = xs
        if self.mode == "freeze":
            def st(s, u, first):
                t = t_mem[0]
                t_mem[0] += 1
                if t >= self.ff:
                    B, Kk, _ = s.shape
                    return c.slot_mem.transition(s), s.new_zeros(B, Kk, u.size(1)), s.new_zeros(B, Kk, 1)
                return orig_step(s, u, first)
            c.slot_mem.step = st
        return self

    def __exit__(self, *a):
        c = self.core
        for o, n in ((c, "_ar_step_kv_slot"), (c, "_slot_x"), (getattr(c, "slot_mem", None), "step")):
            if o is not None and n in o.__dict__:
                delattr(o, n)


def bench_part(model, name, dev, limit):
    ds_name, bn, mv, ff = BENCH[name]
    cfg = yaml.safe_load(open(f"{RES}/{ds_name}_{model}/_resolved.yaml"))
    blk = boards()[bn][1]
    if mv:
        mset = {r["block_id"] for r in csv.DictReader(open(os.path.join(ROOT, f"data_csv/{ds_name}/index.csv"))) if r["motion"] != "flat_static"}
        blk = {b: v for b, v in blk.items() if (b in mset) == (mv == "moving")}
    ref = json.load(open(f"{RES}/{ds_name}_{model}/per_block.json"))["per_video_surprise"]
    ref = {k: (v["all"] if isinstance(v, dict) else v) for k, v in ref.items()}
    ds = WMADataset(cfg)
    pos = {r.video_id: i for i, r in enumerate(ds.records)}
    scenes = sorted(blk)[:limit] if limit else sorted(blk)
    bundle = build_from_config(cfg["model"], dev)
    core = bundle.predictor.core
    has_slot = bool(getattr(core, "slot_n", 0))
    modes = ["normal", "door0", "swap", "other", "freeze"] if has_slot else ["normal"]
    ci, ti = _context_target_indices(ctx_frames=2 * C, tgt_frames=2 * K, tubelet_size=2, spatial_tokens=S, batch_size=4, device=dev)
    acc = {m: {"A": [], "B": [], "dP": [], "eff": []} for m in modes}
    dAB = []
    dH, dS, dSt, errs, prev_S = [], [], [], [], None
    for b in scenes:
        vids = [blk[b]["A"][0], blk[b]["B"][0], blk[b]["A"][1], blk[b]["B"][1]]    # pos_A pos_B imp_A imp_B
        x = torch.stack([ds.clip(pos[v]) for v in vids]).to(dev, bundle.dtype)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            z = bundle.context_encoder(x, masks=[ci])          # 네 클립 모두 (real 은 불가능 클립의 문맥이 픽셀까지 같지 않다)
            zl = ln(z).to(z.dtype)
            hf = bundle.target_encoder(x)
            h = ln(torch.gather(hf, 1, ti.unsqueeze(-1).expand(-1, -1, hf.size(-1))))
            cur_S, p_norm = None, None
            for m in modes:
                if m == "other" and prev_S is None:
                    acc[m]["A"].append(float("nan")); acc[m]["B"].append(float("nan")); acc[m]["dP"].append(None)
                    continue
                with Patch(core, m, other=prev_S, freeze_from=ff) as pt:
                    p, st = core.rollout_kv(zl, ci, K, return_states=True)
                p = p.float()
                s_pos = [float((p[i] - h[i]).abs().mean()) for i in (0, 1)]
                s_imp = [float((p[i + 2] - h[i + 2]).abs().mean()) for i in (0, 1)]
                if m == "normal":
                    p_norm = p
                    dAB.append(float((p[0] - p[1]).abs().mean()))
                acc[m]["eff"].append(float((p - p_norm).abs().mean()))
                acc[m]["A"].append(1.0 if s_imp[0] > s_pos[0] else 0.0)
                acc[m]["B"].append(1.0 if s_imp[1] > s_pos[1] else 0.0)
                acc[m]["dP"].append((p[0] - p[1]).view(K, S, -1).cpu())
                if m == "normal":
                    errs += [abs(s_pos[0] - ref[vids[0]]), abs(s_pos[1] - ref[vids[1]]), abs(s_imp[0] - ref[vids[2]]), abs(s_imp[1] - ref[vids[3]])]
                    Stt = st.float().view(4, K, S, -1).mean(2)
                    dSt.append((Stt[0] - Stt[1]).cpu().numpy())
                    if has_slot:
                        cur_S = pt.S
                        dS.append(torch.stack([(q[0] - q[1]).flatten() for q in pt.S]).cpu().numpy())
            dH.append((h[0] - h[1]).view(K, S, -1).cpu())
            prev_S = cur_S
    res = {"bench": name, "n_scenes": len(scenes), "freeze_from": ff, "ref_max_abs_err": max(errs), "ref_mean_abs_err": float(np.mean(errs)), "modes": {}}
    for m in modes:
        A = np.nanmean(acc[m]["A"]) * 100
        B = np.nanmean(acc[m]["B"]) * 100
        pairs = [(d, y) for d, y in zip(acc[m]["dP"], dH) if d is not None]
        dPm, dHm = [d for d, _ in pairs], [y for _, y in pairs]
        bet, pla = [], []
        for f in range(K):
            den = float(sum((y[f] ** 2).sum() for y in dHm))
            sh = dHm[1:] + dHm[:1]
            bet.append(float(sum((x[f] * y[f]).sum() for x, y in zip(dPm, dHm))) / den)
            pla.append(float(sum((x[f] * y[f]).sum() for x, y in zip(dPm, sh))) / float(sum((y[f] ** 2).sum() for y in sh)))
        res["modes"][m] = {"score": (A + B) / 2, "A": A, "B": B, "effect_vs_normal": float(np.nanmean(acc[m]["eff"])), "beta_minus_placebo": float(np.mean(np.array(bet) - np.array(pla))),
                           "beta_by_block": [a - b_ for a, b_ in zip(bet, pla)]}
    res["pred_diff_AB"] = float(np.mean(dAB))
    res["state_acc_by_step"] = [probe_acc(np.stack([d[f] for d in dSt])) for f in range(K)]
    if has_slot:
        res["slot_acc_by_block"] = [probe_acc(np.stack([d[t] for d in dS])) for t in range(dS[0].shape[0])]
    del bundle
    torch.cuda.empty_cache()
    return res


def load_clip(path, mean, std):
    import decord
    idx = np.arange(0, 64, 2)
    vr = decord.VideoReader(path, num_threads=2)
    x = torch.from_numpy(vr.get_batch(list(idx)).asnumpy()).permute(3, 0, 1, 2).float().div_(255.0)
    x = (x - mean) / std
    with np.load(path.replace("/clips/", "/labels/")[:-4] + ".npz") as z:
        roles = [str(r) for r in z["roles"]]
        if "target" not in roles:                                                    # 물체 없는 클립 · distractor 만 있는 클립은 뺀다
            return None
        t = roles.index("target")
        vis = np.unpackbits(z["visible"][idx, t], axis=-1).astype(np.float32)        # (32, 64, 64)
        lab = {"vis_n": z["visible_n"][:, t].astype(np.float32), "am_n": z["amodal_n"][:, t].astype(np.float32), "ctr": z["center_px"][:, t].astype(np.float32)}
    tok = vis.reshape(16, 2, 16, 4, 16, 4).mean(axis=(1, 3, 5)).reshape(16, 256)      # 블록 · 토큰별 보이는 비율
    return x, tok, lab


def ridge_probe(Xs, Y, groups, seed=0):
    """Xs {이름: (n, d)}, Y (n, 2), groups (n,) 클립 id. 클립 단위 train/val/test (60/20/20), λ 는 val 로. return {이름: (test RMSE, test R²)}."""
    g = np.unique(groups)
    np.random.default_rng(seed).shuffle(g)
    a, b = int(.6 * len(g)), int(.8 * len(g))
    tr, va, te = np.isin(groups, g[:a]), np.isin(groups, g[a:b]), np.isin(groups, g[b:])
    out = {"n": [int(tr.sum()), int(va.sum()), int(te.sum())]}
    for nm, X in Xs.items():
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Z = (X - mu) / sd
        ym = Y[tr].mean(0)
        G = Z[tr] @ Z[tr].T
        best = None
        for lam in (1e1, 1e2, 1e3, 1e4, 1e5, 1e6):
            al = np.linalg.solve(G + lam * np.eye(len(G)), Y[tr] - ym)
            w = Z[tr].T @ al
            e = float(((Z[va] @ w + ym - Y[va]) ** 2).mean())
            if best is None or e < best[0]:
                best = (e, w)
        pr = Z[te] @ best[1] + ym
        mse = float(((pr - Y[te]) ** 2).sum(1).mean())
        out[nm] = {"rmse_px": mse ** .5, "r2": 1 - mse / float(((Y[te] - Y[tr].mean(0)) ** 2).sum(1).mean())}
    return out


def train_part(model, dev, n_clips):
    cfg = yaml.safe_load(open(f"{RES}/real_world_permanence_uniform_{model}/_resolved.yaml"))
    bundle = build_from_config(cfg["model"], dev)
    core = bundle.predictor.core
    has_slot = bool(getattr(core, "slot_n", 0))
    paths = [r.strip().split()[0] for r in open(LIST) if r.strip()]
    random.Random(0).shuffle(paths)
    def has_target(p):
        with np.load(p.replace("/clips/", "/labels/")[:-4] + ".npz") as z:
            return "target" in [str(r) for r in z["roles"]]
    n_seen, kept = 0, []
    for p in paths:
        n_seen += 1
        if has_target(p):
            kept.append(p)
        if len(kept) == n_clips:
            break
    paths = kept
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
    idx16 = torch.arange(16 * S, device=dev)
    enc_f, st_f, sl_f, tok_f, labs, att_all, gate_all = [], [], [], [], [], [], []
    l1 = {m: [0.0, 0.0, 0.0, 0.0, 0.0, 0] for m in (["normal", "door0", "roll"] if has_slot else ["normal"])}   # 합 전체, 수 전체, 합 target, 수 target, normal 과의 차 합, 묶음 수
    door = {}
    bs = 8
    for i in range(0, len(paths), bs):
        items = [load_clip(p, mean, std) for p in paths[i:i + bs]]
        x = torch.stack([it[0] for it in items]).to(dev, bundle.dtype)
        tok = torch.from_numpy(np.stack([it[1] for it in items])).to(dev)           # (B, 16, 256)
        B = x.size(0)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            hb = ln(bundle.target_encoder(x))                                       # (B, 16*S, D) 블록별
            xin = hb.to(torch.float16)
            out, st = core.forward_seq(xin, idx16[None].expand(B, -1), return_states=True)
            if has_slot:
                att, gt = core._slot_tf
                core._slot_tf = None
                sl, _, _ = core._slot_run(xin.view(B, 16, S, -1))
                sl_f.append(sl.float().cpu())
                att_all.append(att.float().cpu())
                gate_all.append(gt.float().cpu())
            enc_f.append(hb.view(B, 16, S, -1).mean(2).cpu())
            st_f.append(st.float().view(B, 16, S, -1).mean(2).cpu())
            tgt = hb[:, C * S:].view(B, K, S, -1)
            ci = idx16[None, : C * S].expand(B, -1)
            mask = (tok[:, C:] >= 0.25).float()
            p0 = None
            for m in l1:
                with Patch(core, m, door_rec=door if m == "normal" and has_slot else None):
                    p = core.rollout_kv(xin[:, : C * S], ci, K)
                p0 = p.float() if p0 is None else p0
                l1[m][4] += float((p.float() - p0).abs().mean()); l1[m][5] += 1
                d = (p.float().view(B, K, S, -1) - tgt).abs().mean(-1)
                l1[m][0] += float(d.sum()); l1[m][1] += d.numel()
                l1[m][2] += float((d * mask).sum()); l1[m][3] += float(mask.sum())
        tok_f.append(tok.cpu())
        labs += [it[2] for it in items]
    res = {"n_clips": len(paths), "n_clips_seen": n_seen, "rollout_l1": {m: {"all": v[0] / v[1], "target_visible": v[2] / max(v[3], 1), "effect_vs_normal": v[4] / v[5]} for m, v in l1.items()}}
    enc, st = torch.cat(enc_f).numpy(), torch.cat(st_f).numpy()
    tok = torch.cat(tok_f)
    # 가려진 블록 표본: 블록 두 장 모두 보이는 비율 < 5 %, 그 전에 본 적 있음
    rows = []
    for ci_, L in enumerate(labs):
        vf = L["vis_n"] / np.maximum(L["am_n"], 1)
        fr = np.arange(0, 64, 2)
        last = None
        for b in range(16):
            f0, f1 = fr[2 * b], fr[2 * b + 1]
            hid = vf[f0] < .05 and vf[f1] < .05 and L["am_n"][f0] > 0
            if hid and last is not None and b >= 1:
                rows.append((ci_, b, L["ctr"][last][0], L["ctr"][last][1], L["ctr"][f1][0], L["ctr"][f1][1], f0 - last))
            for f in (f0, f1):
                if vf[f] >= .5:
                    last = f
    rows = np.array(rows, dtype=np.float64)
    res["hidden_samples"] = int(len(rows))
    if len(rows) > 50:
        c_, b_ = rows[:, 0].astype(int), rows[:, 1].astype(int)
        Xs = {"encoder_now": enc[c_, b_], "trunk_state": st[c_, b_]}
        if has_slot:
            sl = torch.cat(sl_f).numpy()
            Xs["slot"] = sl[c_, b_].reshape(len(rows), -1)
        res["probe_last_seen"] = ridge_probe(Xs, rows[:, 2:4], c_)
        res["probe_now"] = ridge_probe(Xs, rows[:, 4:6], c_)
        res["hidden_gap_frames_mean"] = float(rows[:, 6].mean())
    if has_slot:
        att, gate = torch.cat(att_all), torch.cat(gate_all)                         # (N, 15or16, K, S), (N, nb, K)
        nb = att.size(1)
        Kk = att.size(2)
        ent = -(att.clamp_min(1e-9) * att.clamp_min(1e-9).log()).sum(2) / np.log(Kk)
        mean_map = att.mean((0, 1))                                                 # (K, S)
        tot = ((att - att.mean((0, 1, 3), keepdim=True)) ** 2).mean()
        fixed = ((mean_map - mean_map.mean(-1, keepdim=True)) ** 2).mean()
        res["attn"] = {"entropy_norm": float(ent.mean()), "max_slot_prob": float(att.max(2).values.mean()), "share_sorted": sorted([float(v) for v in att.mean((0, 1, 3))], reverse=True),
                       "fixed_position_share": float(fixed / tot)}
        tk = tok[:, :nb]                                                            # (N, nb, S) 토큰별 보이는 비율
        m = (tk >= .25).float()
        has = m.sum(-1) > 0
        on = (att * m[:, :, None]).sum(-1) / m.sum(-1)[:, :, None].clamp_min(1)       # (N, nb, K) target 토큰 위 평균 attention
        off = (att * (1 - m[:, :, None])).sum(-1) / (1 - m).sum(-1)[:, :, None].clamp_min(1)
        dice = 2 * (att * tk[:, :, None]).sum(-1) / (att.sum(-1) + tk.sum(-1)[:, :, None] + 1e-6)    # (N, nb, K)
        res["attn"]["dice_best_slot"] = float(dice[has].max(-1).values.mean())
        sl_t = torch.cat(sl_f)                                                      # (N, 16, K, D)
        s7 = sl_t[:, 7]
        res["by_slot"] = [{"k": k, "share": float(att[:, :, k].mean()), "target_on": float(on[has][:, k].mean()), "target_off": float(off[has][:, k].mean()),
                           "dice": float(dice[has][:, k].mean()), "norm": float(s7[:, k].norm(dim=-1).mean()),
                           "clip_var_share": float(((s7[:, k] - s7[:, k].mean(0)) ** 2).sum(-1).mean() / (s7[:, k] ** 2).sum(-1).mean()),
                           "change_b2plus": float(((sl_t[:, 3:, k] - sl_t[:, 2:-1, k]).norm(dim=-1) / sl_t[:, 3:, k].norm(dim=-1)).mean())} for k in range(Kk)]
        res["change_by_block"] = [float(v) for v in ((sl_t[:, 1:] - sl_t[:, :-1]).norm(dim=-1) / sl_t[:, 1:].norm(dim=-1)).mean((0, 2))]
        res["slot"] = {"clip_var_share": float(((s7 - s7.mean(0)) ** 2).sum(-1).mean() / (s7 ** 2).sum(-1).mean()),
                       "pair_cos": float((lambda u: ((u @ u.transpose(1, 2)).sum((1, 2)) - Kk).mean() / (Kk * (Kk - 1)))(F.normalize(s7, dim=-1))),
                       "block_change": float(((sl_t[:, 1:] - sl_t[:, :-1]).norm(dim=-1) / sl_t[:, 1:].norm(dim=-1)).mean()),
                       "gate_mean": float(gate[:, 1:].mean()), "gate_std_over_slots": float(gate[:, 1:].std())}
        res["door"] = {str(k): {"rel_norm": float(np.mean([r[0] for r in v])), "const_share": float(np.mean([r[1] for r in v])), "clip_share": float(np.mean([r[2] for r in v]))} for k, v in sorted(door.items())}
    del bundle
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=0, help="장면 수 제한 (점검용)")
    ap.add_argument("--clips", type=int, default=512)
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    os.makedirs(OUT, exist_ok=True)
    res = {"model": a.model, "train": train_part(a.model, dev, a.clips), "bench": {}}
    for n in BENCH:
        res["bench"][n] = bench_part(a.model, n, dev, a.limit)
    out = os.path.join(OUT, f"{a.model}{'_smoke' if a.limit else ''}.json")
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
