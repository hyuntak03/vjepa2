#!/usr/bin/env python3
"""[물리] vidctx · 영상 문맥의 걸음 6 무너짐 진단 — 영상 공간 teacher forcing (학습 없음, forward 만) (auto_research, 2026-10-22 KST).

  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 /venv/vjepa/bin/python -W ignore auto_research/scripts/video_tf_collapse.py --out F
  /venv/vjepa/bin/python auto_research/scripts/video_tf_collapse.py --out F --report

사전 등록: 진행판 [물리] 20:54 UTC (KST 05:54). 창 · 자는 transport_presence_four.py (= 기제 문서) 와 같다.
걸음 k (k = 1..8, 미래 블록 C+k−1 예측) 에서 관측 블록 L = C+k−1 개 (블록 0..L−1) 를 정답 프레임에서 encoder 로 다시 인코딩해 입력으로 준다.
  vid 주 (ii): 블록 0..C−1 = 원래 문맥 pass zc, 블록 C..L−1 = target encoder 가 블록 0..L−1 을 한 번에 본 pass (LN) → rollout_kv 첫 걸음
  vid 보조 (i): 블록 0..L−1 전부 그 pass
  cx 주 (i): online encoder 가 블록 0..L−1 을 한 번에 본 pass (LN 없음, 학습과 같다) → render(rollout_state_kv(z, 1))
  cx 보조 (ii): 블록 0..C−1 = 원래 z, 블록 C..L−1 = 그 pass
k = 1 은 원래 rollout 걸음 1 과 같다 (검산). 잴 것 = 한 걸음 운반 이득 (직전 정답 블록 위치 기준, |Δ| > 16 px · 보임)."""
import argparse, glob, os, random, sys, time
import numpy as np, torch, torch.nn.functional as F, yaml
import decord
REPO = "/workspace/world_model/vjepa2"
sys.path.insert(0, REPO)

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/workspace/dataset/training_data/phys3d_single_v1/val/clips")
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--vid_ck", default="latest.pt")
ap.add_argument("--prev", default="/workspace/auto_research_cache/tp4/full.npy", help="같은 창의 블록 관측 TF 기록 (비교용)")
ap.add_argument("--report", action="store_true")
a = ap.parse_args()
S, G, C, K, ST = 256, 16, 8, 8, 3
T = 2 * (C + K)
torch.set_num_threads(8)


def cells3(cx, cy):
    return [yy * G + xx for yy in range(max(0, cy - 1), min(G, cy + 2)) for xx in range(max(0, cx - 1), min(G, cx + 2))]


def cell_of(p):
    return int(np.clip(p[0] // 16, 0, G - 1)), int(np.clip(p[1] // 16, 0, G - 1))


def extract():
    from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint, _clean_backbone_key
    from app.vjepa_frozen import utils as U
    from app.vjepa_frozen import ar as AR
    dev = torch.device("cuda")
    CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
    sd = _load_checkpoint(CK)
    kw = dict(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, use_rope=True, uniform_power=False)
    tgt = _build_encoder(num_frames=2, **kw); _load_state_dict(tgt, sd["target_encoder"], tag="t"); tgt = tgt.to(dev, torch.bfloat16).eval()
    tgt32 = _build_encoder(num_frames=32, **kw); _load_state_dict(tgt32, sd["target_encoder"], tag="t32"); tgt32 = tgt32.to(dev, torch.bfloat16).eval()
    onl = _build_encoder(num_frames=32, **kw); _load_state_dict(onl, sd["encoder"], tag="o"); onl = onl.to(dev, torch.bfloat16).eval()
    del sd

    def load_pred(run, ck="latest.pt"):
        D = os.path.join(REPO, "z_training/runs", run)
        c = yaml.safe_load(open(os.path.join(D, "config.yaml")))
        m = U.build_predictor(c["model"], 1280, None, torch.device("cpu"), int(c["data"].get("n_frames", 64)))
        w = {k: v for k, v in _clean_backbone_key(torch.load(os.path.join(D, ck), map_location="cpu", weights_only=False)["predictor"]).items()
             if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}
        m.load_state_dict(w, strict=True)
        return m.to(dev).eval()
    cx_ = load_pred("natural_ar_vw_state_ctxt_video")
    vid = load_pred("natural_ar_vw_state_vidctx", a.vid_ck)

    MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
    paths = sorted(p for p in glob.glob(os.path.join(a.root, "*", "*.mp4")) if np.load(p[:-4] + "_states.npz")["pos"].shape[0] >= ST * (T - 1) + 3)
    random.Random(0).shuffle(paths)
    paths = list(enumerate(paths))
    if a.limit:
        paths = paths[: a.limit]
    yy, xx = np.divmod(np.arange(S), G)
    cen = np.stack([xx * 16 + 8, yy * 16 + 8], 1).astype(np.float32)
    ar_idx = lambda n: torch.arange(n * S, device=dev).unsqueeze(0)
    out = []
    t0 = time.time()
    for n_done, (ci, path) in enumerate(paths):
        st = np.load(path[:-4] + "_states.npz")
        vr = decord.VideoReader(path, num_threads=2)
        for r0 in (0, 2):
            idx = [r0 + ST * k for k in range(T)]
            fe = idx[2 * C - 1]
            pos_c, vel = st["pos"][fe, 0], st["vel"][fe, 0] * ST
            if not (st["vis_frac"][fe, 0] > 0.3 and np.isfinite(pos_c).all() and np.isfinite(vel).all()):
                continue
            futp = np.stack([st["pos"][idx[2 * (C + j) + 1], 0] for j in range(K)])
            futv = np.array([st["vis_frac"][idx[2 * (C + j) + 1], 0] > 0.3 for j in range(K)])
            x = torch.from_numpy(vr.get_batch(idx).asnumpy()).to(dev).permute(3, 0, 1, 2)[None].float().div_(255)
            xn = ((x - MEAN) / STD).to(torch.bfloat16)
            old3 = cells3(*cell_of(pos_c))
            R = {"clip": ci, "pos_c": pos_c, "futp": futp, "futv": futv}
            Pv2, Pv1, Pc1, Pc2 = [], [], [], []
            prv = np.concatenate([pos_c[None], futp[:-1]], 0); d_ = futp - prv
            need = (np.linalg.norm(d_, axis=-1) > 16) & futv & np.isfinite(d_).all(-1)     # 한 걸음 이득을 재는 걸음만 forward (나머지는 nan)
            need[0] = True
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                hb = AR.encode_blocks(tgt, xn, 2).float()[0].view(C + K, S, -1)
                zc, zf = AR.encode_video_ctx_full(tgt32, xn, C, S)
                zf = zf.float()[0].view(C + K, S, -1)
                z_on = onl(xn, masks=[ar_idx(C)])
                for k in range(1, K + 1):
                    L = C + k - 1
                    if not need[k - 1]:
                        for lst in (Pv2, Pv1, Pc1, Pc2):
                            lst.append(None)
                        continue
                    if k == 1:                                                                # 걸음 1 = 원래 문맥 그대로 (검산)
                        pk, ok_ = zc, z_on
                    else:
                        pk = F.layer_norm(tgt32(xn, masks=[ar_idx(L)]), (1280,))                # target pass 블록 0..L−1 (LN)
                        ok_ = onl(xn, masks=[ar_idx(L)])                                      # online pass 블록 0..L−1
                    v2 = torch.cat([zc[:, : C * S], pk[:, C * S:]], 1)
                    Pv2.append(vid.rollout_kv(v2, ar_idx(L), 1).float()[0])
                    Pv1.append(vid.rollout_kv(pk, ar_idx(L), 1).float()[0])
                    Pc1.append(cx_.render(cx_.rollout_state_kv(ok_, ar_idx(L), 1)).float()[0])
                    c2 = torch.cat([z_on[:, : C * S], ok_[:, C * S:]], 1)
                    Pc2.append(cx_.render(cx_.rollout_state_kv(c2, ar_idx(L), 1)).float()[0])
            tm_b = F.normalize(hb[C - 1][old3].mean(0), dim=-1)
            tm_w = F.normalize(zf[C - 1][old3].mean(0), dim=-1)

            def readout(blk, tm):
                if blk is None:
                    return np.full(2, np.nan, np.float32)
                cs = (F.normalize(blk, dim=-1) @ tm).cpu().numpy()
                j = int(cs.argmax()); jx, jy = j % G, j // G
                nb = cells3(jx, jy)
                w = np.clip(cs[nb], 0, None) + 1e-6
                return (cen[nb] * w[:, None]).sum(0) / w.sum()
            R["vid2_pos"] = np.stack([readout(p, tm_w) for p in Pv2]); R["vid1_pos"] = np.stack([readout(p, tm_w) for p in Pv1])
            R["cx1_pos"] = np.stack([readout(p, tm_b) for p in Pc1]); R["cx2_pos"] = np.stack([readout(p, tm_b) for p in Pc2])
            R["gt_pos"] = np.stack([readout(hb[C + j], tm_b) for j in range(K)]); R["gtw_pos"] = np.stack([readout(zf[C + j], tm_w) for j in range(K)])
            out.append(R)
        if n_done % 20 == 0:
            print(f"{n_done}/{len(paths)} · 창 {len(out)} · {time.time() - t0:.0f}s", flush=True)
    np.save(f"{a.out}.npy", np.array(out, dtype=object), allow_pickle=True)
    print(f"끝 · 창 {len(out)} · {time.time() - t0:.0f}s")


def report():
    import json
    recs = list(np.load(f"{a.out}.npy", allow_pickle=True))
    n = len(recs)
    clip = np.array([r["clip"] for r in recs])
    pc = np.stack([r["pos_c"] for r in recs]); futp = np.stack([r["futp"] for r in recs]); futv = np.stack([r["futv"] for r in recs])
    prevt = np.concatenate([pc[:, None], futp[:, :-1]], 1)
    dt = futp - prevt
    ok = (np.linalg.norm(dt, axis=-1) > 16) & futv & np.isfinite(dt).all(-1)
    prev = list(np.load(a.prev, allow_pickle=True)) if os.path.exists(a.prev) else None
    if prev is not None:
        assert len(prev) == n and all(p["clip"] == r["clip"] and np.allclose(p["pos_c"], r["pos_c"]) for p, r in zip(prev, recs)), "창 순서 불일치"
    rng = np.random.RandomState(0)
    uc = np.unique(clip); cidx = {c: np.where(clip == c)[0] for c in uc}
    BOOT = [np.concatenate([cidx[c] for c in rng.choice(uc, len(uc))]) for _ in range(1000)]

    def onestep(Pp):
        dp = Pp - prevt
        return (dp * dt).sum(-1) / np.maximum((dt ** 2).sum(-1), 1e-6), np.linalg.norm(Pp - futp, axis=-1)

    def ci_med(g, mask):
        bs = [np.median(g[b][mask[b]]) for b in BOOT if mask[b].any()]
        return np.percentile(bs, [2.5, 97.5])
    rows = [("gt", "읽기 바닥 (정답 블록)"), ("gtw", "읽기 바닥 (정답 창 토큰)"),
            ("vid2", "vidctx 영상 TF 주 (문맥 zc + 정답 pass)"), ("vid1", "vidctx 영상 TF 보조 (전부 정답 pass)"),
            ("cx1", "영상 문맥 영상 TF 주 (다시 인코딩한 문맥)"), ("cx2", "영상 문맥 영상 TF 보조 (원래 z + 정답 pass)")]
    if prev is not None:
        rows += [("prev:cx_tf", "참고: 영상 문맥 블록 관측 TF"), ("prev:ar_tf", "참고: 기준 AR 블록 관측 TF"), ("prev:gr_tf", "참고: gres 블록 관측 TF")]
    res = {"n_windows": n}
    late = np.zeros_like(ok); late[:, 5:] = True
    print(f"창 {n} (clip {len(uc)}). 한 걸음 운반 이득 중앙값 (위치 오차 px) — 걸음 1 · 2 · 4 · 6 · 7 · 8 | 걸음 6–8 합침 [95 %] | 걸음 2–4 합침")
    for key, lb in rows:
        Pp = np.stack([r[key.split(":")[1] + "_pos"] for r in prev]) if key.startswith("prev:") else np.stack([r[key + "_pos"] for r in recs])
        g, err = onestep(Pp)
        okg = ok & np.isfinite(g)
        cells = [f"{np.median(g[okg[:, j], j]):5.2f} ({np.median(err[okg[:, j], j]):4.1f})" for j in (0, 1, 3, 5, 6, 7)]
        m68 = okg & late
        v68 = np.median(g[m68]); ci = ci_med(np.where(m68, g, np.nan), m68)
        m24 = okg.copy(); m24[:, [0, 4, 5, 6, 7]] = False
        print(f"  {lb:34s} " + "  ".join(cells) + f" | {v68:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] n {m68.sum()} | {np.median(g[m24]):.2f}")
        res[key] = dict(steps={str(j + 1): float(np.median(g[okg[:, j], j])) for j in range(K)}, err={str(j + 1): float(np.median(err[okg[:, j], j])) for j in range(K)},
                        late68=[float(v68), float(ci[0]), float(ci[1]), int(m68.sum())], mid24=float(np.median(g[m24])))
    print("  걸음별 n: " + " ".join(str(int(ok[:, j].sum())) for j in range(K)))
    cp = [float(np.median(np.linalg.norm(dt[ok[:, j], j], axis=-1))) for j in range(K)]
    print("  복사 (직전 정답 위치) 의 위치 오차 px: " + " ".join(f"{v:.1f}" for v in cp))
    res["copy_err"] = cp
    json.dump(res, open(f"{a.out}.json", "w"), indent=1, ensure_ascii=False)
    print(f"저장 {a.out}.json")


if a.report:
    report()
else:
    extract()
