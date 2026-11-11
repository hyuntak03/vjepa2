"""slot 판 외형 의존 진단 (2026-11-07, 사용자 "핵심은 배울 때 appearance agnostic 하게 배우고 싶은 거야"). 학습 없음.

같은 영상의 외형만 바꾼다 (움직임 · 가림은 그대로): none · gray (회색) · hue (R→G→B→R 채널 돌리기, 색상 120°) · invert (음화).
문맥 · 정답 모두 바꾼 영상에서 뽑으므로, 외형이 달라져도 같은 판단을 내리는지를 본다.
  시트 점수 A · B, 물체를 봤는지 판별 (slot 내용 · 몸통 상태).
  주소 = slot attention 맵. 물체 마스크가 없는 벤치에서는 A 문맥과 B 문맥의 블록 토큰 차이 상위 5 % 칸을 물체 자리 대용으로 쓴다
    (A · B 는 물체 유무만 다른 짝). on_A = A 영상에서 그 칸의 attention, on_B = B 영상 (물체 없음) 의 같은 칸. slot 마다 낸다.
  v4bc4 클립 128 개는 진짜 보이는 마스크로 dice.

  CUDA_VISIBLE_DEVICES=0 /venv/vjepa/bin/python z_research/scripts/analysis/slot_appearance_diag.py --model vast_ar_vw_state_v4bc4_slotmatch --tf gray
출력: z_analysis/results/slot_diag_20261107/appearance_<model>_<tf>.json
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
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from slot_diag import (BENCH, C, K, LIST, OUT, RES, ROOT, S, WMADataset, _context_target_indices, boards, build_from_config, ln, load_clip, probe_acc)  # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
ADDR_BLOCKS = {"real95": 5, "vob3_panel_moving": 7}     # encoder 가 A · B 를 가르는 문맥 블록 수 (ar_memory_readout T3)


def tf_clip(x, tf):
    """x (3, T, H, W) 정규화된 클립."""
    if tf == "none":
        return x
    y = x * STD + MEAN
    if tf == "gray":
        g = 0.299 * y[0] + 0.587 * y[1] + 0.114 * y[2]
        y = torch.stack([g, g, g])
    elif tf == "hue":
        y = y.roll(1, 0)
    elif tf == "invert":
        y = 1.0 - y
    return (y - MEAN) / STD


class Rec:
    """slot 갱신마다 attention (B, K, S) 을 남긴다."""

    def __init__(self, core):
        self.core, self.att = core, []

    def __enter__(self):
        if getattr(self.core, "slot_n", 0):
            orig = self.core.slot_mem.step

            def st(s, u, first):
                o = orig(s, u, first)
                self.att.append(o[1].detach().float())
                return o
            self.core.slot_mem.step = st
        return self

    def __exit__(self, *a):
        m = getattr(self.core, "slot_mem", None)
        if m is not None and "step" in m.__dict__:
            del m.step


def run_bench(bundle, model, name, tf, dev):
    ds_name, bn, mv, _ = BENCH[name]
    cfg = yaml.safe_load(open(f"{RES}/{ds_name}_{model}/_resolved.yaml"))
    blk = boards()[bn][1]
    if mv:
        mset = {r["block_id"] for r in csv.DictReader(open(os.path.join(ROOT, f"data_csv/{ds_name}/index.csv"))) if r["motion"] != "flat_static"}
        blk = {b: v for b, v in blk.items() if (b in mset) == (mv == "moving")}
    ds = WMADataset(cfg)
    pos = {r.video_id: i for i, r in enumerate(ds.records)}
    core = bundle.predictor.core
    has_slot = bool(getattr(core, "slot_n", 0))
    ci, ti = _context_target_indices(ctx_frames=2 * C, tgt_frames=2 * K, tubelet_size=2, spatial_tokens=S, batch_size=4, device=dev)
    A, B, dSt, dS, on_a, on_b, off_a = [], [], [], [], [], [], []
    nb_addr = ADDR_BLOCKS[name]
    for b in sorted(blk):
        vids = [blk[b]["A"][0], blk[b]["B"][0], blk[b]["A"][1], blk[b]["B"][1]]
        x = torch.stack([tf_clip(ds.clip(pos[v]), tf) for v in vids]).to(dev, bundle.dtype)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            z = bundle.context_encoder(x, masks=[ci])
            zl = ln(z).to(z.dtype)
            hf = bundle.target_encoder(x)
            h = ln(torch.gather(hf, 1, ti.unsqueeze(-1).expand(-1, -1, hf.size(-1))))
            with Rec(core) as rc:
                p, st = core.rollout_kv(zl, ci, K, return_states=True)
        p = p.float()
        s = [float((p[i] - h[i]).abs().mean()) for i in range(4)]
        A.append(1.0 if s[2] > s[0] else 0.0)
        B.append(1.0 if s[3] > s[1] else 0.0)
        Stt = st.float().view(4, K, S, -1).mean(2)
        dSt.append((Stt[0] - Stt[1]).cpu().numpy())
        if has_slot:
            att = torch.stack(rc.att)                                               # (n_step, 4, K, S)
            u = zl.float().view(4, C, S, -1)
            d = (u[0] - u[1]).abs().mean(-1)                                        # (C, S) A · B 문맥 차이
            top = d.topk(13, dim=-1).indices[:nb_addr]                              # 상위 5 % 칸
            m = torch.zeros(nb_addr, S, device=dev).scatter_(1, top, 1.0)
            aA, aB = att[:nb_addr, 0], att[:nb_addr, 1]                             # (nb, K, S)
            on_a.append(((aA * m[:, None]).sum(-1) / 13).mean(0).cpu().numpy())
            on_b.append(((aB * m[:, None]).sum(-1) / 13).mean(0).cpu().numpy())
            off_a.append(((aA * (1 - m[:, None])).sum(-1) / (S - 13)).mean(0).cpu().numpy())
    Am, Bm = 100 * float(np.mean(A)), 100 * float(np.mean(B))
    res = {"n": len(A), "score": (Am + Bm) / 2, "A": Am, "B": Bm, "state_acc": float(np.mean([probe_acc(np.stack([d[f] for d in dSt])) for f in range(3, K)]))}
    if has_slot:
        res["addr_on_A"] = np.mean(on_a, 0).tolist()
        res["addr_on_B"] = np.mean(on_b, 0).tolist()
        res["addr_off_A"] = np.mean(off_a, 0).tolist()
    return res


def run_dice(bundle, tf, dev, n=128):
    core = bundle.predictor.core
    paths = [r.strip().split()[0] for r in open(LIST) if r.strip()]
    random.Random(1).shuffle(paths)
    picked = []
    for p in paths:
        with np.load(p.replace("/clips/", "/labels/")[:-4] + ".npz") as z:
            if "target" in [str(r) for r in z["roles"]]:
                picked.append(p)
        if len(picked) == n:
            break
    tot, cnt, best = np.zeros(core.slot_n), 0, []
    for i in range(0, n, 8):
        it = [load_clip(p, MEAN, STD) for p in picked[i:i + 8]]
        x = torch.stack([tf_clip(a[0], tf) for a in it]).to(dev, bundle.dtype)
        tok = torch.from_numpy(np.stack([a[1] for a in it])).to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            hb = ln(bundle.target_encoder(x)).to(torch.float16)
            _, att, _ = core._slot_run(hb.view(x.size(0), 16, S, -1))
        att = att.float()
        dice = 2 * (att * tok[:, :, None]).sum(-1) / (att.sum(-1) + tok.sum(-1)[:, :, None] + 1e-6)
        has = (tok >= .25).sum(-1) > 0
        tot += dice[has].sum(0).cpu().numpy()
        cnt += int(has.sum())
        best.append(dice[has].max(-1).values.cpu())
    return {"dice_by_slot": (tot / cnt).tolist(), "dice_best": float(torch.cat(best).mean()), "n_blocks": cnt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tf", required=True, choices=["none", "gray", "hue", "invert"])
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    cfg = yaml.safe_load(open(f"{RES}/real_world_permanence_uniform_{a.model}/_resolved.yaml"))
    bundle = build_from_config(cfg["model"], dev)
    res = {"model": a.model, "tf": a.tf, "bench": {n: run_bench(bundle, a.model, n, a.tf, dev) for n in BENCH}}
    if getattr(bundle.predictor.core, "slot_n", 0):
        res["v4bc4"] = run_dice(bundle, a.tf, dev)
    json.dump(res, open(os.path.join(OUT, f"appearance_{a.model}_{a.tf}.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, ensure_ascii=False)[:800])


if __name__ == "__main__":
    main()
