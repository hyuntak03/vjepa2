#!/usr/bin/env python3
"""[물리] 같은 자 · 같은 창에서 네 모델의 운반 이득 (위치 읽기) 과 새 칸 존재감 (학습 없음, 체크포인트 읽기 + forward 만) (auto_research, 2026-10-22 KST).

  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 /venv/vjepa/bin/python -W ignore auto_research/scripts/transport_presence_four.py --out F
  /venv/vjepa/bin/python auto_research/scripts/transport_presence_four.py --out F --report

사전 등록: 진행판 [물리] 20:28 UTC (KST 05:28).
자 · 창 = z_research/scripts/analysis/ctxt_vs_block_mechanism.py 그대로 (phys3d_single_v1 val, 96 장 clip, r0 ∈ {0, 2}, 간격 3,
  문맥 블록 0..7 → 미래 블록 8..15, 문맥 끝 물체 vis > 0.3). 위치 읽기 = 마지막 문맥 블록 물체 3×3 평균 템플릿 · cos 최대 칸 주변 가중 중심.
모델 (호출은 각 run 의 학습 val 그대로):
  ar   = natural_ar_vw_state           (kind=ar, 블록별 LN(target) 문맥)               rollout_kv · forward_seq
  cx   = natural_ar_vw_state_ctxt_video (ctx_state, z = online encoder 창 문맥 pass)     render(rollout_state_kv) · render(forward_state)
  gr   = natural_ar_vw_state_ctxt_block_gres (ctx_state, z = 블록별 LN(target) 문맥, gate 지름길 anchor = z 마지막 블록)
  vid  = natural_ar_vw_state_vidctx    (kind=ar, 입력 zc = target 문맥 pass LN, 정답 zf = target 전체 pass LN) rollout_kv(zc)
존재감 ([P] 정의) = Σ⟨p − m, y − m⟩ / Σ‖y − m‖², m = 같은 칸 문맥 평균 (rollout: 블록 0..7, TF: 블록 0..7+j−1). 칸 무리:
  new = 미래 블록 j 물체 3×3 중 마지막 문맥 블록 물체 3×3 에 없던 칸 · full = 미래 물체 3×3 · static = 16 블록 어디서도 물체 3×3 이
  아닌 칸 중 미래 변화 (블록 공간) 가 중앙값 아래. vid 는 정답 공간이 창 토큰이라 y · m · 템플릿을 zf 에서 잡는다."""
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
ap.add_argument("--report", action="store_true")
ap.add_argument("--C", type=int, default=8, help="문맥 블록 수 (창 · 라벨은 C0 = 8 그대로, 문맥은 블록 8−C..7 만 쓴다)")
a = ap.parse_args()
S, G, C0, K, ST = 256, 16, 8, 8, 3
C = a.C                                              # 모델에 주는 문맥 블록 수 (창 선택 · 라벨은 C0 기준)
T = 2 * (C0 + K)
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
        print(f"load {run}/{ck}: kind {c['model']['predictor'].get('kind')}", flush=True)
        return m.to(dev).eval()

    ar = load_pred("natural_ar_vw_state")
    cx_ = load_pred("natural_ar_vw_state_ctxt_video")
    gr = load_pred("natural_ar_vw_state_ctxt_block_gres")
    vid = load_pred("natural_ar_vw_state_vidctx", a.vid_ck)
    assert gr.gating_residual and not cx_.gating_residual

    MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
    paths = sorted(p for p in glob.glob(os.path.join(a.root, "*", "*.mp4")) if np.load(p[:-4] + "_states.npz")["pos"].shape[0] >= ST * (T - 1) + 3)
    random.Random(0).shuffle(paths)
    paths = list(enumerate(paths))
    if a.limit:
        paths = paths[: a.limit]
    yy, xx = np.divmod(np.arange(S), G)
    cen = np.stack([xx * 16 + 8, yy * 16 + 8], 1).astype(np.float32)
    idx_ctx = torch.arange(C * S, device=dev).unsqueeze(0)
    idx_tf = torch.arange((C + K - 1) * S, device=dev).unsqueeze(0)
    out = []
    t0 = time.time()
    for n_done, (ci, path) in enumerate(paths):
        st = np.load(path[:-4] + "_states.npz")
        vr = decord.VideoReader(path, num_threads=2)
        for r0 in (0, 2):
            idx = [r0 + ST * k for k in range(T)]
            fe = idx[2 * C0 - 1]
            pos_c, vel = st["pos"][fe, 0], st["vel"][fe, 0] * ST
            if not (st["vis_frac"][fe, 0] > 0.3 and np.isfinite(pos_c).all() and np.isfinite(vel).all()):
                continue
            futp = np.stack([st["pos"][idx[2 * (C0 + j) + 1], 0] for j in range(K)])
            futv = np.array([st["vis_frac"][idx[2 * (C0 + j) + 1], 0] > 0.3 for j in range(K)])
            x = torch.from_numpy(vr.get_batch(idx).asnumpy()).to(dev).permute(3, 0, 1, 2)[None].float().div_(255)
            assert x.shape[-1] == 256 and x.shape[-2] == 256
            x = x[:, :, 2 * (C0 - C):]                                                        # 문맥 = 블록 C0−C..C0−1
            xn = ((x - MEAN) / STD).to(torch.bfloat16)
            old_c = cell_of(pos_c)
            old3 = cells3(*old_c)
            # 물체 칸 (16 블록 · 두 장 모두, 위치가 있으면) → 정지 칸 후보
            objc = set()
            for f in idx:
                p_ = st["pos"][f, 0]
                if np.isfinite(p_).all():
                    objc |= set(cells3(*cell_of(p_)))
            nonobj = np.array([c for c in range(S) if c not in objc])
            R = {"clip": ci, "scen": os.path.basename(os.path.dirname(path)), "vel": vel, "pos_c": pos_c, "futp": futp, "futv": futv}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                hb = AR.encode_blocks(tgt, xn, 2).float()[0].view(C + K, S, -1)                  # 블록별 LN(target) 16 블록
                z_on = onl(xn, masks=[idx_ctx]).float()                                         # 영상 문맥 z (LN 없음)
                zc, zf = AR.encode_video_ctx_full(tgt32, xn, C, S)
                zc, zf = zc.float(), zf.float()[0].view(C + K, S, -1)                            # vidctx 입력 · 정답 (창 LN)
                P = {}
                P["ar_ro"] = ar.rollout_kv(hb[:C].reshape(1, C * S, -1), idx_ctx, K).float()[0].view(K, S, -1)
                P["ar_tf"] = ar.forward_seq(hb[:C + K - 1].reshape(1, -1, hb.size(-1)), idx_tf).float()[0].view(C + K - 1, S, -1)[C - 1:]
                P["cx_ro"] = cx_.render(cx_.rollout_state_kv(z_on, idx_ctx, K)).float()[0].view(K, S, -1)
                P["cx_tf"] = cx_.render(cx_.forward_state(z_on, cx_.embed_obs(hb[C:C + K - 1].reshape(1, -1, hb.size(-1))), idx_tf, C)).float()[0].view(K, S, -1)
                zb = hb[:C].reshape(1, C * S, -1); A = hb[C - 1][None]
                s_ro = gr.rollout_state_kv(zb, idx_ctx, K)
                P["gr_ro"] = gr.render(s_ro, A).float()[0].view(K, S, -1)
                g_ro = gr._gate_cache.float()[0, :, 0].view(K, S).cpu().numpy()
                P["grproj_ro"] = gr.predictor_proj(s_ro).float()[0].view(K, S, -1)                # 진단: gate 몫을 뺀 proj(s) (실제 예측 아님)
                s_tf = gr.forward_state(zb, gr.embed_obs(hb[C:C + K - 1].reshape(1, -1, hb.size(-1))), idx_tf, C)
                P["gr_tf"] = gr.render(s_tf, A).float()[0].view(K, S, -1)
                g_tf = gr._gate_cache.float()[0, :, 0].view(K, S).cpu().numpy()
                P["vid_ro"] = vid.rollout_kv(zc, idx_ctx, K).float()[0].view(K, S, -1)
            # 위치 읽기 (파라미터 없음)
            tm_b = F.normalize(hb[C - 1][old3].mean(0), dim=-1)
            tm_w = F.normalize(zf[C - 1][old3].mean(0), dim=-1)

            def readout(blk, tm):
                cs = (F.normalize(blk, dim=-1) @ tm).cpu().numpy()
                j = int(cs.argmax()); jx, jy = j % G, j // G
                nb = cells3(jx, jy)
                w = np.clip(cs[nb], 0, None) + 1e-6
                return (cen[nb] * w[:, None]).sum(0) / w.sum(), float(cs.max())
            space = {"ar_ro": "b", "ar_tf": "b", "cx_ro": "b", "cx_tf": "b", "gr_ro": "b", "gr_tf": "b", "grproj_ro": "b", "vid_ro": "w"}
            for key, sp in space.items():
                rr = [readout(P[key][j], tm_b if sp == "b" else tm_w) for j in range(K)]
                R[key + "_pos"] = np.stack([r[0] for r in rr]); R[key + "_cos"] = np.array([r[1] for r in rr])
            rr = [readout(P["vid_ro"][j], tm_b) for j in range(K)]                               # 보조: vid 를 블록 템플릿으로
            R["vidb_ro_pos"] = np.stack([r[0] for r in rr])
            for key, Y, tm in (("gt", hb, tm_b), ("gtw", zf, tm_w)):
                rr = [readout(Y[C + j], tm) for j in range(K)]
                R[key + "_pos"] = np.stack([r[0] for r in rr]); R[key + "_cos"] = np.array([r[1] for r in rr])
            R["ctx_pos_read"] = readout(hb[C - 1], tm_b)[0]; R["ctxw_pos_read"] = readout(zf[C - 1], tm_w)[0]
            # 존재감
            mctx_b = hb[:C].mean(0); mctx_w = zf[:C].mean(0)
            chg = (hb[C:, nonobj] - mctx_b[nonobj]).abs().mean(-1).max(0).values.cpu().numpy() if len(nonobj) else np.zeros(0)
            stc = nonobj[chg <= np.median(chg)] if len(nonobj) else nonobj
            R["n_static"] = len(stc)
            groups = []
            for j in range(K):
                if futv[j] and np.isfinite(futp[j]).all():
                    f3 = cells3(*cell_of(futp[j])); new = [c for c in f3 if c not in old3]
                else:
                    f3, new = [], []
                groups.append({"new": new, "full": f3, "static": list(stc)})
            for key in ("ar_ro", "ar_tf", "cx_ro", "cx_tf", "gr_ro", "gr_tf", "vid_ro"):
                Y = zf if key == "vid_ro" else hb
                pres = {g: np.zeros((K, 3), np.float64) for g in ("new", "full", "static")}       # (num, den, 칸 수)
                for j in range(K):
                    m = (zf[:C].mean(0) if key == "vid_ro" else mctx_b) if key.endswith("_ro") else Y[:C + j].mean(0)
                    y = Y[C + j]; p = P[key][j]
                    for g, cl in groups[j].items():
                        if len(cl) == 0:
                            continue
                        cl_ = torch.as_tensor(cl, device=dev)
                        dp, dy = p[cl_] - m[cl_], y[cl_] - m[cl_]
                        pres[g][j] = (float((dp * dy).sum()), float((dy * dy).sum()), len(cl))
                for g in pres:
                    R[f"{key}_pres_{g}"] = pres[g]
            # gres gate (걸음별 평균: 새 칸 · 옛 자리 3×3 · 정지 칸)
            for nm, gg in (("ro", g_ro), ("tf", g_tf)):
                R[f"gr_gate_{nm}"] = np.array([[gg[j, groups[j]["new"]].mean() if groups[j]["new"] else np.nan, gg[j, old3].mean(),
                                                gg[j, stc].mean() if len(stc) else np.nan, gg[j].mean()] for j in range(K)], np.float32)
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
    print(f"창 {n} (clip {len(np.unique(clip))})")
    prevt = np.concatenate([pc[:, None], futp[:, :-1]], 1)
    dtr_ro = futp - pc[:, None]
    ok_ro = (np.linalg.norm(dtr_ro, axis=-1) > 16) & futv & np.isfinite(dtr_ro).all(-1)
    dtr_tf = futp - prevt
    ok_tf = (np.linalg.norm(dtr_tf, axis=-1) > 16) & futv & np.isfinite(dtr_tf).all(-1)
    rng = np.random.RandomState(0)
    uc = np.unique(clip); cidx = {c: np.where(clip == c)[0] for c in uc}
    BOOT = [np.concatenate([cidx[c] for c in rng.choice(uc, len(uc))]) for _ in range(1000)]

    def gain(key):
        Pp = np.stack([r[key + "_pos"] for r in recs])
        base = prevt if key.endswith("_tf") else pc[:, None]
        dp, dt = Pp - base, futp - base
        g = (dp * dt).sum(-1) / np.maximum((dt ** 2).sum(-1), 1e-6)
        err = np.linalg.norm(Pp - futp, axis=-1)
        return g, err, (ok_tf if key.endswith("_tf") else ok_ro)

    def bmed(v, mask):                                                     # clip bootstrap 중앙값 구간 (mask 는 창)
        bs = []
        for b in BOOT:
            vv = v[b][mask[b]]
            if len(vv):
                bs.append(np.median(vv))
        return np.percentile(bs, [2.5, 97.5])
    res = {"n_windows": n}
    print("\n[1] 운반 이득 중앙값 (위치 오차 px 중앙값) — rollout 은 문맥 끝 기준 누적, TF 는 직전 정답 기준 한 걸음. 걸음 1 · 2 · 4 · 6 · 8")
    lab = {"gt": "정답 블록 (읽기 바닥)", "gtw": "정답 창 토큰 (vid 바닥)", "ar_ro": "기준 AR rollout", "cx_ro": "영상 문맥 rollout",
           "gr_ro": "gres rollout", "vid_ro": "vidctx rollout (창 템플릿)", "vidb_ro": "vidctx rollout (블록 템플릿, 보조)",
           "grproj_ro": "gres proj(s) 만 (진단)", "ar_tf": "기준 AR TF", "cx_tf": "영상 문맥 TF", "gr_tf": "gres TF"}
    G1 = {}
    for key, lb in lab.items():
        g, err, ok = gain(key)
        row = []
        for j in (0, 1, 3, 5, 7):
            m_ = ok[:, j] & np.isfinite(g[:, j])
            row.append(f"{np.median(g[m_, j]):5.2f} ({np.median(err[m_, j]):4.1f})" if m_.sum() > 5 else "  —  ")
        m0 = ok[:, 0] & np.isfinite(g[:, 0])
        ci = bmed(np.where(m0, g[:, 0], np.nan), m0)
        tfm = ok[:, 1:] & np.isfinite(g[:, 1:])
        tf28 = f"  TF/걸음 2–8 합침 {np.median(g[:, 1:][tfm]):.2f} (n {tfm.sum()})" if key.endswith("_tf") else ""
        print(f"  {lb:28s} " + "  ".join(row) + f"   걸음1 [{ci[0]:.2f}, {ci[1]:.2f}] n {m0.sum()}" + tf28)
        G1[key] = g
        res[key] = dict(gain_steps={str(j + 1): float(np.median(g[ok[:, j] & np.isfinite(g[:, j]), j])) for j in (0, 1, 3, 5, 7)},
                        err_steps={str(j + 1): float(np.median(err[ok[:, j], j])) for j in (0, 1, 3, 5, 7)}, step1_ci=[float(ci[0]), float(ci[1])], n1=int(m0.sum()))
    cp = [float(np.median(np.linalg.norm(futp[ok_ro[:, j], j] - pc[ok_ro[:, j]], axis=-1))) for j in (0, 1, 3, 5, 7)]
    print("  복사 (문맥 끝 위치) 오차 같은 창: " + "  ".join(f"{v:5.1f}" for v in cp))
    res["copy_err"] = cp
    print(f"  읽기 확인: 마지막 문맥 블록 자기 읽기 오차 블록 {np.median(np.linalg.norm(np.stack([r['ctx_pos_read'] for r in recs]) - pc, axis=1)):.1f} px · "
          f"창 {np.median(np.linalg.norm(np.stack([r['ctxw_pos_read'] for r in recs]) - pc, axis=1)):.1f} px")

    print("\n  걸음 1 이득 짝 대응 차 (같은 창, clip bootstrap 95 %)")
    m0 = ok_ro[:, 0]
    for a_, b_ in (("cx_ro", "ar_ro"), ("gr_ro", "ar_ro"), ("cx_ro", "gr_ro"), ("vid_ro", "ar_ro"), ("vid_ro", "cx_ro")):
        d = G1[a_][:, 0] - G1[b_][:, 0]
        mm = m0 & np.isfinite(d)
        ci = bmed(d, mm)
        print(f"    {a_} − {b_}: 중앙 {np.median(d[mm]):+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]")
        res[f"d1_{a_}-{b_}"] = [float(np.median(d[mm])), float(ci[0]), float(ci[1])]

    print("\n[2] 새 칸 존재감 (Σ⟨p−m,y−m⟩/Σ‖y−m‖²) · 정지 칸 · 차 (new − static). 창 무리 A = 걸음 1 이득 창 (같은 창), B = 모든 창 (보일 때)")

    def pres(key, g, win, j):
        A_ = np.stack([r[f"{key}_pres_{g}"] for r in recs])                 # (n, K, 3)
        num, den = A_[win, j, 0].sum(), A_[win, j, 1].sum()
        return num / max(den, 1e-9), int(A_[win, j, 2].sum())

    def pres_boot(key, win, j, base=None):
        A_ = {g: np.stack([r[f"{key}_pres_{g}"] for r in recs]) for g in ("new", "static")}
        B_ = None if base is None else {g: np.stack([r[f"{base}_pres_{g}"] for r in recs]) for g in ("new", "static")}

        def val(ix):
            w = ix[win[ix]]
            v = A_["new"][w, j, 0].sum() / max(A_["new"][w, j, 1].sum(), 1e-9) - A_["static"][w, j, 0].sum() / max(A_["static"][w, j, 1].sum(), 1e-9)
            if B_ is not None:
                v -= B_["new"][w, j, 0].sum() / max(B_["new"][w, j, 1].sum(), 1e-9) - B_["static"][w, j, 0].sum() / max(B_["static"][w, j, 1].sum(), 1e-9)
            return v
        bs = [val(b) for b in BOOT]
        return val(np.arange(n)), np.percentile(bs, [2.5, 97.5])
    winA = ok_ro[:, 0]
    winB = np.ones(n, bool)
    PR = {}
    for key in ("ar_ro", "cx_ro", "gr_ro", "vid_ro", "ar_tf", "cx_tf", "gr_tf"):
        for wn, win in (("A", winA), ("B", winB)):
            row = []
            for j in ((0,) if wn == "A" else (0, 1, 3, 5, 7)):
                pn, nn = pres(key, "new", win, j); pf, _ = pres(key, "full", win, j); ps, _ = pres(key, "static", win, j)
                row.append(f"{pn:5.2f} / {pf:5.2f} / {ps:5.2f} → {pn - ps:+.2f}")
                PR[(key, wn, j)] = (pn, pf, ps)
            print(f"  {key:7s} 창 {wn}: " + "  |  ".join(row) + "   (new / full / static → new−static)")
    res["presence"] = {f"{k[0]}|{k[1]}|{k[2] + 1}": v for k, v in PR.items()}
    print("\n  걸음 1 (new − static) 짝 대응 차, 창 A · B (clip bootstrap 95 %)")
    for a_, b_ in (("cx_ro", "ar_ro"), ("gr_ro", "ar_ro"), ("vid_ro", "ar_ro"), ("cx_ro", "gr_ro")):
        for wn, win in (("A", winA), ("B", winB)):
            v, ci = pres_boot(a_, win, 0, b_)
            print(f"    {a_} − {b_} 창 {wn}: {v:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")
            res[f"dp1_{a_}-{b_}_{wn}"] = [float(v), float(ci[0]), float(ci[1])]
    for key in ("ar_ro", "cx_ro", "gr_ro", "vid_ro"):
        for wn, win in (("A", winA), ("B", winB)):
            v, ci = pres_boot(key, win, 0)
            res[f"p1_{key}_{wn}"] = [float(v), float(ci[0]), float(ci[1])]

    print("\n[3] gres gate (새 칸 · 옛 자리 3×3 · 정지 칸 · 전체) 평균, 걸음 1 · 4 · 8 (rollout) / 걸음 1 (TF), 창 A · B")
    for nm in ("ro", "tf"):
        Gt = np.stack([r[f"gr_gate_{nm}"] for r in recs])
        for wn, win in (("A", winA), ("B", winB)):
            js = (0, 3, 7) if nm == "ro" else (0,)
            print(f"  {nm} 창 {wn}: " + "  |  ".join(" · ".join(f"{np.nanmean(Gt[win, j, q]):.2f}" for q in range(4)) for j in js))
            res[f"gate_{nm}_{wn}"] = {str(j + 1): [float(np.nanmean(Gt[win, j, q])) for q in range(4)] for j in js}
    json.dump(res, open(f"{a.out}.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n저장 {a.out}.json")


if a.report:
    report()
else:
    extract()
