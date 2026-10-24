#!/usr/bin/env python3
"""영상 문맥 (ctxt_video) vs 블록 문맥 (AR) — predictor 의 어디서 운반이 갈리나 (2026-10-21, 사용자 지시).

  # 추출 (GPU 장마다 shard, 학습 run 이 도는 중이라 장당 ~13 GB 안에서)
  for k in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$k python -W ignore z_research/scripts/analysis/ctxt_vs_block_mechanism.py \
      --out F --shard $k --nshard 4 & done; wait
  python z_research/scripts/analysis/ctxt_vs_block_mechanism.py --out F --nshard 4 --report

데이터: phys3d_single_v1 **val** (학습에 안 쓴 clip, 물체 1 개, 라벨 위치 · 속도), 96 장 clip 만 (32 장 × 간격 3 = raw 0..93 이 들어감),
창 시작 raw r0 ∈ {0, 2}. 채점과 같은 배치: 문맥 16 장 (블록 0..7) → 미래 블록 8..15.
모델: AR = natural_ar_vw_state (kind=ar, 문맥 = 블록별 LN(target_encoder), block-causal, 상태 되먹임)
      ctxt = natural_ar_vw_state_ctxt_video (kind=ctx_state, 문맥 = online encoder 가 32 장 창에 문맥 마스크로 한 번, 양방향, 상태 되먹임)
둘 다 미래 정답 = 블록별 LN(target_encoder).

측정
 1. 운반 이득 — 예측 블록에서 물체 위치를 **파라미터 없이** 읽는다: 템플릿 = 마지막 문맥 블록 (정답 LN 블록 토큰) 의 물체 중심 3×3 평균,
    예측 블록 256 칸과 cosine → 최대 칸 주변 3×3 의 cos 가중 중심. 이득 = (d_pred · d_true) / |d_true|² (d = 마지막 문맥 위치에서의 변위).
    rollout (굴린 예측) 과 TF (정답 과거를 다 보고 한 걸음) 를 따로. 같은 읽기를 정답 블록에 건 값 = 측정 바닥.
 2. 층별 속도 — 문맥 마지막 블록 자리 (첫 예측 블록을 내는 query 들) 의 층별 은닉 상태, 물체 **옛 자리** 3×3 · **새 자리** 3×3 평균.
 3. attention — 같은 자리에서 새 자리 query (3×3) 가 key 에 준 질량: 마지막 문맥 블록의 옛 자리 · 새 자리, 그 앞 블록의 옛 자리, 나머지.
 4. 쓰임 — 층별 은닉 상태에서 읽은 속도 (probe 예측) 와 실제로 예측 안에서 움직인 변위의 상관."""
import argparse, glob, os, random, sys, time
import numpy as np, torch, torch.nn.functional as F, yaml
import decord
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/workspace/dataset/training_data/phys3d_single_v1/val/clips")
ap.add_argument("--out", required=True)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--report", action="store_true")
a = ap.parse_args()
S, G, C, K, ST = 256, 16, 8, 8, 3                     # 칸 · 격자 · 문맥 블록 · 미래 블록 · 간격
T = 2 * (C + K)


def cells3(cx, cy):
    return [yy * G + xx for yy in range(max(0, cy - 1), min(G, cy + 2)) for xx in range(max(0, cx - 1), min(G, cx + 2))]


def cell_of(p):
    return int(np.clip(p[0] // 16, 0, G - 1)), int(np.clip(p[1] // 16, 0, G - 1))


def extract():
    from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint, _clean_backbone_key
    from src.models.rollout_predictor import vit_prefix_predictor
    from src.models.utils import modules as M
    from app.vjepa_frozen import utils as U
    from app.vjepa_frozen import ar as AR
    dev = torch.device("cuda")
    CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
    sd = _load_checkpoint(CK)
    tgt = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
    _load_state_dict(tgt, sd["target_encoder"], tag="t"); tgt = tgt.to(dev, torch.bfloat16).eval()
    onl = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=32, use_rope=True, uniform_power=False)
    _load_state_dict(onl, sd["encoder"], tag="o"); onl = onl.to(dev, torch.bfloat16).eval()
    del sd

    def load_pred(run):
        D = os.path.join(REPO, "z_training/runs", run)
        c = yaml.safe_load(open(os.path.join(D, "config.yaml")))
        pc = c["model"]["predictor"]
        if pc.get("kind") == "ctx_state":
            m = U.build_predictor(c["model"], 1280, None, torch.device("cpu"), int(c["data"].get("n_frames", 64)))
        else:
            m = vit_prefix_predictor(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384,
                                     depth=12, num_heads=12, num_mask_tokens=10, mask_mode="ar", ar_feedback=pc.get("ar_feedback", "output"))
        w = {k: v for k, v in _clean_backbone_key(torch.load(os.path.join(D, "latest.pt"), map_location="cpu", weights_only=False)["predictor"]).items()
             if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}
        m.load_state_dict(w, strict=True)
        return m.to(dev).eval()

    ar = load_pred("natural_ar_vw_state")
    cx_ = load_pred("natural_ar_vw_state_ctxt_video")
    Hg = Wg = G

    # attention 기록: SDPA 를 감싸 query 부분집합의 softmax 를 남긴다
    REC = {"on": False, "q": None, "attn": []}
    _sdpa = F.scaled_dot_product_attention

    def sdpa(q, k, v, *args, **kw):
        if REC["on"]:
            qq = q[:, :, REC["q"]].float()
            w = torch.softmax(qq @ k.float().transpose(-1, -2) / q.size(-1) ** 0.5, dim=-1)      # (B, h, nq, M)
            REC["attn"].append(w.mean(1)[0].cpu())                                            # 헤드 평균 (nq, M)
        return _sdpa(q, k, v, *args, **kw)
    M.F.scaled_dot_product_attention = sdpa

    MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
    LN = lambda t: F.layer_norm(t, (t.size(-1),))
    paths = sorted(p for p in glob.glob(os.path.join(a.root, "*", "*.mp4")) if np.load(p[:-4] + "_states.npz")["pos"].shape[0] >= ST * (T - 1) + 3)
    random.Random(0).shuffle(paths)
    paths = list(enumerate(paths))[a.shard::a.nshard]
    if a.limit:
        paths = paths[: a.limit]
    out = []
    t0 = time.time()
    for n_done, (ci, path) in enumerate(paths):
        st = np.load(path[:-4] + "_states.npz")
        vr = decord.VideoReader(path, num_threads=2)
        for r0 in (0, 2):
            idx = [r0 + ST * k for k in range(T)]
            fe = idx[2 * C - 1]                                                            # 문맥 마지막 블록의 마지막 장
            pos_c, vel = st["pos"][fe, 0], st["vel"][fe, 0] * ST
            if not (st["vis_frac"][fe, 0] > 0.3 and np.isfinite(pos_c).all() and np.isfinite(vel).all()):
                continue
            futp = np.stack([st["pos"][idx[2 * (C + j) + 1], 0] for j in range(K)])          # 미래 블록 8..15 물체 위치
            futv = np.array([st["vis_frac"][idx[2 * (C + j) + 1], 0] > 0.3 for j in range(K)])
            prevp = st["pos"][idx[2 * C - 3], 0]                                            # 블록 6 의 물체 위치
            x = torch.from_numpy(vr.get_batch(idx).asnumpy()).to(dev).permute(3, 0, 1, 2)[None].float().div_(255)
            if x.shape[-1] != 256:
                x = F.interpolate(x[0].transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False).transpose(0, 1)[None]
            xn = (x - MEAN) / STD
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                hb = AR.encode_blocks(tgt, xn.to(torch.bfloat16), 2).float()[0].view(C + K, S, -1)    # 정답 블록 16 개 (LN)
                z = onl(xn.to(torch.bfloat16), masks=[torch.arange(C * S, device=dev).unsqueeze(0)]).float()   # ctxt 문맥 z
            new_c = cell_of(futp[0]) if np.isfinite(futp[0]).all() else cell_of(pos_c)
            old_c, prev_c = cell_of(pos_c), cell_of(prevp) if np.isfinite(prevp).all() else cell_of(pos_c)
            qn = cells3(*new_c)
            keys = {"old_last": [(C - 1) * S + c for c in cells3(*old_c)], "new_last": [(C - 1) * S + c for c in cells3(*new_c)],
                    "old_prev": [(C - 2) * S + c for c in cells3(*prev_c)]}
            rec = {"clip": ci, "scen": os.path.basename(os.path.dirname(path)), "vel": vel, "pos_c": pos_c, "futp": futp, "futv": futv}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for name, m in (("ar", ar), ("cx", cx_)):
                    hid_old, hid_new, att = [], [], []
                    if name == "ar":
                        caches = None
                        for b in range(C):                                                 # 문맥 채우기 (block-causal)
                            h = m._ar_embed_obs(hb[b:b + 1].reshape(1, S, -1))
                            pos = torch.arange(b * S, (b + 1) * S, device=dev).unsqueeze(0)
                            new = []
                            last = b == C - 1
                            if last:
                                hid_old.append(h[0, cells3(*old_c)].mean(0).float().cpu()); hid_new.append(h[0, qn].mean(0).float().cpu())
                                REC.update(on=True, q=qn, attn=[])
                            for i, blk in enumerate(m.predictor_blocks):
                                h, kv = blk.forward_kv(h, pos, Hg, Wg, None if caches is None else caches[i])
                                new.append(kv)
                                if last:
                                    hid_old.append(h[0, cells3(*old_c)].mean(0).float().cpu()); hid_new.append(h[0, qn].mean(0).float().cpu())
                            REC["on"] = False
                            caches = new
                        att = REC["attn"]
                        idx_ctx = torch.arange(C * S, device=dev).unsqueeze(0)
                        p_ro = m.rollout_kv(hb[:C].reshape(1, C * S, -1), idx_ctx, K).float()[0].view(K, S, -1)
                        p_tf = m.forward_seq(hb[:C + K - 1].reshape(1, -1, hb.size(-1)), torch.arange((C + K - 1) * S, device=dev).unsqueeze(0)).float()[0]
                        p_tf = p_tf.view(C + K - 1, S, -1)[C - 1:]                                # 위치 7..14 → 블록 8..15
                    else:
                        idx_ctx = torch.arange(C * S, device=dev).unsqueeze(0)
                        h = m._embed_ctx(z)
                        hid_old.append(h[0, (C - 1) * S + torch.tensor(cells3(*old_c))].mean(0).float().cpu())
                        hid_new.append(h[0, (C - 1) * S + torch.tensor(qn)].mean(0).float().cpu())
                        REC.update(on=True, q=[(C - 1) * S + c for c in qn], attn=[])
                        for blk in m.predictor_blocks:
                            h, _ = blk.forward_kv(h, idx_ctx, Hg, Wg, None)
                            hid_old.append(h[0, (C - 1) * S + torch.tensor(cells3(*old_c))].mean(0).float().cpu())
                            hid_new.append(h[0, (C - 1) * S + torch.tensor(qn)].mean(0).float().cpu())
                        REC["on"] = False
                        att = REC["attn"]
                        p_ro = m.render(m.rollout_state_kv(z, idx_ctx, K)).float()[0].view(K, S, -1)
                        s_tf = m.forward_state(z, m.embed_obs(hb[C:C + K - 1].reshape(1, -1, hb.size(-1))),
                                               torch.arange((C + K - 1) * S, device=dev).unsqueeze(0), C)
                        p_tf = m.render(s_tf).float()[0].view(K, S, -1)
                    am = []
                    for w in att:                                                          # 층별 (nq, M)
                        wm = w.mean(0)
                        tot = {k_: float(wm[v_].sum()) for k_, v_ in keys.items()}
                        tot["rest"] = 1 - sum(tot.values())
                        am.append([tot["old_last"], tot["new_last"], tot["old_prev"], tot["rest"]])
                    rec[f"{name}_hid_old"] = torch.stack(hid_old).numpy().astype(np.float32)   # (13, 384)
                    rec[f"{name}_hid_new"] = torch.stack(hid_new).numpy().astype(np.float32)
                    rec[f"{name}_att"] = np.array(am, np.float32)                               # (12, 4)
                    rec[f"{name}_ro"] = p_ro.half().cpu().numpy()
                    rec[f"{name}_tf"] = p_tf.half().cpu().numpy()
            rec["gt"] = hb.half().cpu().numpy()                                                # (16, 256, 1280)
            # 위치 읽기 (파라미터 없음) — 여기서 바로 계산하고 큰 텐서는 버린다
            tmpl = F.normalize(hb[C - 1][cells3(*old_c)].mean(0), dim=-1)
            yy, xx = np.divmod(np.arange(S), G)
            cen = np.stack([xx * 16 + 8, yy * 16 + 8], 1).astype(np.float32)

            def readout(blk):                                                               # (S, D) → px 위치
                cs = (F.normalize(torch.as_tensor(blk, device=dev).float(), dim=-1) @ tmpl).cpu().numpy()
                j = int(cs.argmax()); jx, jy = j % G, j // G
                nb = cells3(jx, jy)
                w = np.clip(cs[nb], 0, None) + 1e-6
                return (cen[nb] * w[:, None]).sum(0) / w.sum(), float(cs.max())
            for key in ("ar_ro", "ar_tf", "cx_ro", "cx_tf"):
                rr = [readout(rec[key][j]) for j in range(K)]
                rec[key + "_pos"] = np.stack([r[0] for r in rr]); rec[key + "_cos"] = np.array([r[1] for r in rr])
                del rec[key]
            rr = [readout(hb[C + j]) for j in range(K)]
            rec["gt_pos"] = np.stack([r[0] for r in rr]); rec["gt_cos"] = np.array([r[1] for r in rr])
            rr = readout(hb[C - 1]); rec["ctx_pos_read"] = rr[0]
            del rec["gt"]
            out.append(rec)
        if n_done % 10 == 0:
            print(f"[shard {a.shard}] {n_done}/{len(paths)} · 창 {len(out)} · {time.time() - t0:.0f}s", flush=True)
    np.save(f"{a.out}.{a.shard}.npy", np.array(out, dtype=object), allow_pickle=True)
    print(f"[shard {a.shard}] 끝 · 창 {len(out)}")


def ridge_cv(X, Y, groups, lams=(1e-1, 1e0, 1e1, 1e2, 1e3, 1e4)):
    """clip 단위 반반 split, train 안 5-fold 로 λ. 반환 (test R², test 예측, test mask)."""
    ug = np.unique(groups); rng = np.random.RandomState(0); rng.shuffle(ug)
    trg = set(ug[: len(ug) // 2]); tr = np.array([g in trg for g in groups]); te = ~tr

    def fit(Xa, Ya, lam):
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-6; ym = Ya.mean(0)
        Xs = (Xa - mu) / sd
        W = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ (Ya - ym))
        return lambda Z: ((Z - mu) / sd) @ W + ym

    def r2(Yt, P):
        return 1 - ((Yt - P) ** 2).sum() / ((Yt - Yt.mean(0)) ** 2).sum()
    f5 = np.array([int(g) % 5 for g in groups[tr]])
    best = max(lams, key=lambda l: np.mean([r2(Y[tr][f5 == k], fit(X[tr][f5 != k], Y[tr][f5 != k], l)(X[tr][f5 == k])) for k in range(5)]))
    P = fit(X[tr], Y[tr], best)(X[te])
    return r2(Y[te], P), P, te


def report():
    recs = []
    for k in range(a.nshard):
        recs += list(np.load(f"{a.out}.{k}.npy", allow_pickle=True))
    n = len(recs)
    clip = np.array([r["clip"] for r in recs])
    vel = np.stack([r["vel"] for r in recs])
    print(f"창 {n} (clip {len(np.unique(clip))}) · 블록당 변위 = 2·|v| (px/블록), |v| 중앙 {np.median(np.linalg.norm(vel, axis=1)):.1f} px/장")
    pc = np.stack([r["pos_c"] for r in recs])
    futp = np.stack([r["futp"] for r in recs]); futv = np.stack([r["futv"] for r in recs])
    dtrue = futp - pc[:, None]                                                       # (n, K, 2)
    sp_blk = 2 * np.linalg.norm(vel, axis=1) / 16                                    # 칸 / 블록
    bins = [("느림 <0.5 칸/블록", sp_blk < 0.5), ("중간 0.5–1.5", (sp_blk >= 0.5) & (sp_blk < 1.5)), ("빠름 ≥1.5", sp_blk >= 1.5)]
    print("\n[1] 운반 이득 = (d_pred·d_true)/|d_true|² — 중앙값 [걸음 1 은 clip bootstrap 95 % 구간] (|d_true| > 16 px 이고 그 블록에 물체가 보일 때만), 괄호 = 위치 오차 px 중앙값")
    print("    rollout 의 d 는 문맥 끝 위치 기준 (누적). TF 의 d 는 **직전 정답 블록 위치** 기준 (한 걸음) — TF 는 직전 정답을 보므로")
    prevt = np.concatenate([pc[:, None], futp[:, :-1]], 1)                              # TF 걸음 j 의 출발 = 정답 블록 7+j
    rng = np.random.RandomState(0)

    def boot(gv, cl):
        uc = np.unique(cl); idx = {c: np.where(cl == c)[0] for c in uc}
        bs = [np.median(gv[np.concatenate([idx[c] for c in rng.choice(uc, len(uc))])]) for _ in range(1000)]
        return np.percentile(bs, [2.5, 97.5])
    for key, lab in (("gt", "정답 블록 (측정 바닥, 누적)"), ("ar_tf", "AR · TF 한 걸음"), ("cx_tf", "ctxt · TF 한 걸음"), ("ar_ro", "AR · rollout (누적)"), ("cx_ro", "ctxt · rollout (누적)")):
        P = np.stack([r[key + "_pos"] for r in recs])
        base = prevt if key.endswith("_tf") else pc[:, None]
        dp, dtr = P - base, futp - base
        g = (dp * dtr).sum(-1) / np.maximum((dtr ** 2).sum(-1), 1e-6)
        err = np.linalg.norm(P - futp, axis=-1)
        ok = (np.linalg.norm(dtr, axis=-1) > 16) & futv & np.isfinite(g) & np.isfinite(dtr).all(-1)
        row = []
        for j in (0, 1, 3, 5, 7):
            m_ = ok[:, j]
            row.append(f"{np.median(g[m_, j]):5.2f} ({np.median(err[m_, j]):4.1f})" if m_.sum() > 5 else "  —  ")
        ci = boot(g[ok[:, 0], 0], clip[ok[:, 0]])
        print(f"  {lab:26s} 걸음 1·2·4·6·8: " + "  ".join(row) + f"   걸음1 구간 [{ci[0]:.2f}, {ci[1]:.2f}] n {ok[:, 0].sum()}")
        if key in ("ar_ro", "cx_ro"):
            for bl, bm in bins:
                row = []
                for j in (0, 3, 7):
                    m_ = ok[:, j] & bm
                    row.append(f"{np.median(g[m_, j]):5.2f}" if m_.sum() > 5 else "  — ")
                print(f"      {bl:18s} 걸음 1·4·8: " + "  ".join(row) + f"   n {int((ok[:, 0] & bm).sum())}")
    print("  위치 오차 px 중앙값 (그 블록에 물체가 보이는 모든 창) 걸음 1·2·4·6·8:")
    for key in ("gt", "ar_tf", "cx_tf", "ar_ro", "cx_ro"):
        P = np.stack([r[key + "_pos"] for r in recs]); err = np.linalg.norm(P - futp, axis=-1)
        print(f"      {key:6s} " + "  ".join(f"{np.median(err[futv[:, j] & np.isfinite(err[:, j]), j]):5.1f}" for j in (0, 1, 3, 5, 7))
              + f"   (복사 = 문맥 끝 위치: " + "  ".join(f"{np.median(np.linalg.norm(futp[futv[:, j], j] - pc[futv[:, j]], axis=-1)):5.1f}" for j in (0, 1, 3, 5, 7)) + ")")
    cpos = np.stack([r["ctx_pos_read"] for r in recs])
    print(f"  (읽기 확인: 마지막 문맥 블록 자기 자신을 읽은 위치 오차 중앙 {np.median(np.linalg.norm(cpos - pc, axis=1)):.1f} px)")

    print("\n[2] 층별 속도 R² (선형, clip 단위 split) — 문맥 마지막 블록 자리: 옛 자리 (물체 있음) / 새 자리 (첫 예측에서 물체가 올 곳)")
    mv = np.linalg.norm(vel, axis=1) > 2
    for name in ("ar", "cx"):
        Ho = np.stack([r[f"{name}_hid_old"] for r in recs]); Hn = np.stack([r[f"{name}_hid_new"] for r in recs])
        ro = [ridge_cv(Ho[:, l], vel, clip)[0] for l in range(Ho.shape[1])]
        rn = [ridge_cv(Hn[:, l], vel, clip)[0] for l in range(Hn.shape[1])]
        print(f"  {name:3s} 옛 자리 층 0(입력)…12: " + " ".join(f"{v:.2f}" for v in ro))
        print(f"  {name:3s} 새 자리 층 0(입력)…12: " + " ".join(f"{v:.2f}" for v in rn))

    print("\n[3] attention — 새 자리 query (3×3) 가 준 질량, 층 1…12 (헤드 평균): 마지막 블록 옛 자리 / 마지막 블록 새 자리 / 앞 블록 옛 자리 / 나머지")
    far = np.linalg.norm(dtrue[:, 0], axis=-1) > 24                                  # 새 자리와 옛 자리가 1.5 칸 이상 떨어진 창
    for name in ("ar", "cx"):
        A = np.stack([r[f"{name}_att"] for r in recs])[far]
        print(f"  {name:3s} (n {far.sum()})")
        for l in (0, 2, 3, 5, 7, 9, 11):
            print(f"      층 {l + 1:2d}: " + " / ".join(f"{v:.3f}" for v in A[:, l].mean(0)))
        S_unif = 9 / (8 * 256)
        print(f"      (균등 기준: 9 칸 = {S_unif:.4f})")

    print("\n[4] 쓰임 — 층별로 읽은 속도 (probe 예측, test) 와 rollout 걸음 1 · 4 에서 실제로 예측 안에서 움직인 변위의 상관 (x·y 합쳐 Pearson)")
    for name in ("ar", "cx"):
        Ho = np.stack([r[f"{name}_hid_old"] for r in recs])
        P = np.stack([r[f"{name}_ro_pos"] for r in recs]); dp = P - pc[:, None]
        line = []
        for l in (0, 4, 8, 12):
            r2_, pv, te = ridge_cv(Ho[:, l], vel, clip)
            c1 = np.corrcoef(pv.ravel(), dp[te, 0].ravel())[0, 1]
            c4 = np.corrcoef(pv.ravel(), dp[te, 3].ravel())[0, 1]
            g1 = (dp[te, 0] * pv).sum() / (pv * pv).sum() / 2     # 예측 변위 / (2·읽은 속도) — 블록당 2 장
            line.append(f"층{l}: R² {r2_:.2f} · 상관 {c1:.2f}/{c4:.2f} · 기울기 {g1:.2f}")
        print(f"  {name:3s} " + " | ".join(line))
    dt = dtrue[:, 0]
    print(f"  (참고: 실제 속도와 실제 변위 걸음 1 상관 {np.corrcoef(vel.ravel(), dt.ravel())[0, 1]:.2f}, 기울기 {(dt * vel).sum() / (vel * vel).sum() / 2:.2f})")


if a.report:
    report()
else:
    extract()
