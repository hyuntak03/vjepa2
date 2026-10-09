#!/usr/bin/env python3
"""장면 단위 permanence (실패 2: 마지막 관측에 없으면 그리지 않는가) — 인공 팬 + 화면 고정 가림 띠 (2026-09-25 밤).

e_scene_reach.py 의 팬 clip (K400 val 정지 프레임을 v px/프레임으로 민다) 에 **화면 좌표 고정** 세로 띠 (x ∈ [BX0, BX1), 회색) 를
마지막 문맥 k 프레임 (16−k .. 15) 에만 그린다. 미래 16 장은 가리지 않는다. 팬이라 띠 뒤로 내용이 지나간다.
  · 가려진 토큰 (hid): 프레임 15 에서 띠 안에 있던 칸 — 마지막 문맥 튜블릿 (L=7) 에서는 보이지 않고, 튜블릿 L−m (m = ceil(k/2)) 이전에는 보였다.
  · 대조 (vis): 같은 clip 에서 띠 밖 칸 (같은 거리 분포).
  · 출처 판독: 미래 슬롯 i 의 칸 s 내용은 튜블릿 L−m 에서 x + dir·v·(2(i+1) + 2m) px 에 있었다 → 템플릿 = hc[L−m][u_m] (가려지기 전 자리, 인과 표적).
    비교용으로 hc[L][u_0] (마지막 튜블릿 자리 = 띠 내용) 도 저장한다.
저장: cos_prev (n, P, 8, 256) = cos(p_i[s], hc[L−m][u_m]) · cos_last (n, P, 8, 256) = cos(p_i[s], hc[L][u_0]) · argmax_prev (n, P, 8, 256) over hc[L−m]
      · hid (n, 8, 256) bool · enc_prev (n, 8, 256) = cos(h_{8+i}[s], hc[L−m][u_m]) (encoder 천장) · meta.
팔: release · Ariel · pv1 (+SCENE_EXTRA) base 만. k ∈ {2, 4, 8} 프레임 (= 1 · 2 · 4 튜블릿) 을 clip 마다 돌아가며 배정.
분석 (e_scene_permanence_analyze.py): hid vs vis 토큰의 적중 (argmax_prev 가 u_m 1 칸 안) 을 d 구간 · k 별로. 가려진 토큰이 vis 만큼 복원되면 permanence 있음.

  python e_scene_permanence.py --gpus 8 --n 800 --out <dir>
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arlib  # noqa: E402
from h4_single_query_v3 import idx_range  # noqa: E402
from e_ssv2_extract import MEAN, STD  # noqa: E402
import e_scene_reach as ES  # noqa: E402

S_TOK, D, NF, TC, G, PX = 256, 1280, 32, 8, 16, 16
BX0, BX1 = 96, 160                       # 화면 고정 띠 (4 칸 폭)
KS = [2, 4, 8]                           # 가림 프레임 수
GRAY = ((0.5 - MEAN) / STD).view(3, 1, 1, 1)
PREDS = ["R", "A", "P"] + [kv.split("=", 1)[0] for kv in os.environ.get("SCENE_EXTRA", "").split(",") if "=" in kv]


def occlude(x, k):
    x = x.clone(); x[:, NF // 2 - k: NF // 2, :, BX0:BX1] = GRAY.to(x.dtype)
    return x


def src_prev(v, dr, m):
    """미래 슬롯 i 칸 s 의 출처 (튜블릿 L−m 자리) flat 인덱스 (−1 = 창 밖)."""
    src = np.full((8, S_TOK), -1, np.int64)
    for i in range(8):
        sh = dr * v * (2 * (i + 1) + 2 * m)
        for y in range(G):
            for xx in range(G):
                xp = PX * xx + PX / 2 + sh
                if 0 <= xp < 256: src[i, y * G + xx] = y * G + int(xp // PX)
    return src


def _worker(rank, world, a, items):
    dev = torch.device(f"cuda:{rank}"); torch.cuda.set_device(dev)
    cfg = arlib.v3_cfg(16, 16); os.environ.pop("ARLIB_PRED_CKPT", None)
    bundle = arlib.build_bundle(cfg, dev, window=NF); ac = arlib.autocast_ctx(cfg)
    preds = {"R": bundle.predictor}
    preds["A"], _ = ES.load_predictor(cfg["model"], ES.ARIEL, dev, bundle.dtype, NF); preds["P"], _ = ES.load_predictor(cfg["model"], ES.PV1, dev, bundle.dtype, NF)
    for _k, _p in ES.EXTRA: preds[_k], _ = ES.load_predictor(cfg["model"], _p, dev, bundle.dtype, NF)
    ci = idx_range(0, TC, 1, dev); ti = idx_range(TC, TC, 1, dev)
    O = {nm: np.lib.format.open_memmap(Path(a.out) / f"{nm}.npy", mode="r+") for nm in ("cos_prev", "cos_last", "argmax_prev", "hid", "enc_prev", "enc_last")}
    mine = list(range(rank, len(items), world)); t0 = time.time()
    for n_done, row in enumerate(mine):
        it = items[row]; k = it["k"]; m = (k + 1) // 2
        x = occlude(ES.load_pan(it), k)[None].to(dev, bundle.dtype)
        sp = src_prev(it["v"], it["dir"], m); s0 = ES.pan_src(it["v"], it["dir"])[0]
        # 가려진 토큰: 프레임 15 에서 띠 안 = 미래 칸 s 의 내용이 프레임 15 에 있던 x 좌표가 [BX0, BX1) 안
        hid = np.zeros((8, S_TOK), bool)
        for i in range(8):
            x15 = (np.arange(S_TOK) % G) * PX + PX / 2 + it["dir"] * it["v"] * (2 * (i + 1) - 0.5)   # 프레임 15 (L 의 둘째 프레임) 자리
            hid[i] = (x15 >= BX0) & (x15 < BX1)
        O["hid"][row] = hid
        with torch.inference_mode(), ac():
            h = F.layer_norm(bundle.target_encoder(x), (D,)).float().reshape(16, S_TOK, D)
            hc = F.layer_norm(bundle.target_encoder(x[:, :, :16]), (D,)).float().reshape(8, S_TOK, D)
            z = bundle.context_encoder(x, masks=[ci])
        hp, hl = hc[TC - 1 - m], hc[TC - 1]
        spv = torch.as_tensor(sp, device=dev); s0v = torch.as_tensor(s0, device=dev)
        def cos_at(X, Hs, srcv):
            Xn, Hn = F.normalize(X, dim=-1), F.normalize(Hs, dim=-1); cs = Xn @ Hn.T
            ct = torch.gather(cs, 2, srcv.clamp(min=0)[..., None])[..., 0].masked_fill(srcv < 0, float("nan"))
            return cs, ct
        cs_e, ct_e = cos_at(h[8:], hp, spv); O["enc_prev"][row] = ct_e.half().cpu().numpy()
        _, ct_el = cos_at(h[8:], hl, s0v); O["enc_last"][row] = ct_el.half().cpu().numpy()
        for pi, pk in enumerate(PREDS):
            with torch.inference_mode(), ac():
                p = preds[pk](z, ci, ti, mask_index=0).float().reshape(8, S_TOK, D)
            cs, ct = cos_at(p, hp, spv); O["cos_prev"][row, pi] = ct.half().cpu().numpy(); O["argmax_prev"][row, pi] = cs.argmax(-1).short().cpu().numpy()
            _, ctl = cos_at(p, hl, s0v); O["cos_last"][row, pi] = ctl.half().cpu().numpy()
        if rank == 0 and n_done % 20 == 0:
            el = time.time() - t0; print(f"[rank0] {n_done+1}/{len(mine)} {el/60:.1f} 분 남은 {el/(n_done+1)*(len(mine)-n_done-1)/60:.1f} 분", flush=True)
    for v_ in O.values(): v_.flush()
    print(f"[rank{rank}] 끝", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    ap.add_argument("--n", type=int, default=800); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    items = ES.pan_items(a.limit or a.n)
    for j, it in enumerate(items): it["k"] = KS[j % len(KS)]
    n = len(items)
    shapes = dict(cos_prev=((n, len(PREDS), 8, S_TOK), np.float16), cos_last=((n, len(PREDS), 8, S_TOK), np.float16), argmax_prev=((n, len(PREDS), 8, S_TOK), np.int16),
                  hid=((n, 8, S_TOK), np.bool_), enc_prev=((n, 8, S_TOK), np.float16), enc_last=((n, 8, S_TOK), np.float16))
    for nm, (sh, dt) in shapes.items():
        mm = np.lib.format.open_memmap(Path(a.out) / f"{nm}.npy", mode="w+", dtype=dt, shape=sh)
        if dt != np.bool_: mm[:] = (-1 if dt == np.int16 else np.nan)
        del mm
    json.dump(dict(n=n, items=items, preds=PREDS, band=[BX0, BX1], ks=KS, speeds=ES.SPEEDS), open(Path(a.out) / "meta.json", "w"))
    print("clips", n, "preds", PREDS, flush=True); torch.set_num_threads(4)
    mp.spawn(_worker, args=(a.gpus, a, items), nprocs=a.gpus, join=True); print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
