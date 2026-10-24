#!/usr/bin/env python3
"""[P] 위치에 관대한 공간 T_s (칸 s×s 평균 묶음) 의 판별 진단 — VOBench 끝 가림 · IntPhys 2 · 자연 보류 후보 순위 (학습 없음) (auto_research, 2026-10-22 KST).
⚠️ 채점만 바꾼 판별 진단이다 (method 아님, 관문 4).

  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=5 /venv/vjepa/bin/python -W ignore auto_research/scripts/pool_space.py [--parts 123]

사전 등록: 진행판 [P] 11:03 UTC (KST 20:03).
T_s: (…, 256, D) → (…, (16/s)², D) 칸 평균. surprise = 묶은 칸 · 차원 평균 L1.
(1) VOBench 끝 가림 (vobench_late 캐시, 216 block): 문맥 8 → 굴린 예측 8 (base · vob). 물체 자리 = 라벨 물체 칸 (중심 ±1) 을 담은 묶은 칸.
(2) IntPhys 2 Main: base = 공유 캐시 p_tf, vob = 같은 토큰에 forward_seq. 영상 surprise = 블록 b = 4..30 의 T_s L1(p[b], h[b+1]) 평균. crit_ip2_pmi.decomp.
(3) K710 보류 (pmi_cont 캐시 · 메타, nce_rank_eval 와 같은 읽기 · 같은 후보 규칙 · 같은 잡음 판): 다시 인코딩 후보 hn 으로 참 vs 멈춤 승률 (t = 3..6) · 8 후보 순위 (t = 3..8) · 중심성 · 복사 연속.
출력: auto_research/results/pool_space.json."""
import argparse, collections, csv, json, os, random, sys, time
import numpy as np, torch, torch.nn.functional as F
import decord

ap = argparse.ArgumentParser()
ap.add_argument("--parts", default="123")
ap.add_argument("--out", default="/workspace/world_model/vjepa2/auto_research/results/pool_space.json")
a = ap.parse_args()
REPO = "/workspace/world_model/vjepa2"
sys.path.insert(0, os.path.join(REPO, "auto_research/scripts")); sys.path.insert(0, REPO)
_argv = list(sys.argv); sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT  # noqa: E402
sys.argv = _argv
from crit_ip2_pmi import decomp  # noqa: E402
torch.set_num_threads(8)
dev = torch.device("cuda")
S, G, D = 256, 16, 1280
SS = (1, 2, 4, 8, 16)
MODELS = {"base": FT.load_pred("natural_ar_vw_state", "latest.pt").to(dev).eval(), "vob": FT.load_pred("natural_ar_vw_state_vob", "latest.pt").to(dev).eval()}


def pool(x, s):                                                                # x (..., 256, D) → (..., (16/s)², D)
    if s == 1:
        return x
    sh = x.shape[:-2]
    y = x.reshape(-1, G, G, x.shape[-1]).permute(0, 3, 1, 2)
    y = F.avg_pool2d(y, s).permute(0, 2, 3, 1)
    return y.reshape(*sh, (G // s) ** 2, x.shape[-1])


def pool_mask(cells, s):                                                       # 칸 목록 → 묶은 칸 목록
    return sorted({(c // G // s) * (G // s) + (c % G) // s for c in cells})


res = {"definition": "사전 등록 [P] 11:03 UTC (KST 20:03); ⚠️ 채점만 바꾼 판별 진단 (method 아님, 관문 4)"}
t0 = time.time()
rng_b = np.random.default_rng(0)


def boot_mean(x, n=2000):
    x = np.asarray(x, float); m = len(x)
    bs = np.sort([x[rng_b.integers(0, m, m)].mean() for _ in range(n)])
    return [round(float(x.mean()), 2), round(float(bs[int(0.025 * n)]), 2), round(float(bs[int(0.975 * n) - 1]), 2)]


# ------------------------------------------------------------------ (1) VOBench 끝 가림
if "1" in a.parts:
    from PIL import Image
    VB = "/workspace/dataset/benchmarks/VOBench_v1"
    IDX = list(range(0, 96, 3))[:32]; C, K = 8, 8
    idx_rows = collections.defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv"))):
        if r["occlusion"] == "late":
            idx_rows[r["block_id"]][r["variant"]] = r
    held = {r["source_block"] for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1_heldout/index.csv")))}
    meta = json.load(open("/workspace/auto_research_cache/vobench_late/meta.json"))
    H = np.load("/workspace/auto_research_cache/vobench_late/h.npy", mmap_mode="r")
    ARMS = ["pos_a", "pos_b", "imp_ab", "imp_ba"]

    def centers(label_path, W0):
        L_ = json.load(open(label_path))["labels"]; out = {}
        for key in ("A", "B"):
            lab = {q["f"]: q for q in L_.get(key, [])}; cs = []
            for b in range(16):
                q = [lab.get(IDX[2 * b]), lab.get(IDX[2 * b + 1])]
                if any(t is None for t in q):
                    cs.append(None); continue
                cs.append((int(np.clip(np.mean([t["v"] for t in q]) / W0 * G, 0, G - 1)), int(np.clip(np.mean([t["u"] for t in q]) / W0 * G, 0, G - 1))))
            out[key] = cs
        return out

    def region(rc, rad=1):
        r0, c0 = rc
        return [rr * G + cc for rr in range(max(0, r0 - rad), min(G, r0 + rad + 1)) for cc in range(max(0, c0 - rad), min(G, c0 + rad + 1))]

    recs = []
    for bi, m in enumerate(meta):
        arms = idx_rows[m["block_id"]]
        d0 = os.path.join(VB, arms["pos_a"]["file_name"])
        W0 = Image.open(f"{d0}/{IDX[0]:06d}.png").size[0]
        cn = centers(os.path.join(os.path.dirname(d0), "label.json"), W0)
        otok = []
        for j in range(K):
            s_ = set()
            for key in ("A", "B"):
                if cn[key][C + j] is not None:
                    s_ |= set(region(cn[key][C + j]))
            otok.append(sorted(s_))
        hh = {v: torch.from_numpy(np.asarray(H[m["i"], k], dtype=np.float32)).to(dev) for k, v in enumerate(ARMS)}
        mv = "static" if arms["pos_a"]["motion"] == "flat_static" else "moving"
        viol = arms["pos_a"]["violation"]; empty_b = arms["pos_a"].get("obj_b", "") == "empty"
        for pid, (pos, imp) in (("1", ("pos_a", "imp_ab")), ("2", ("pos_b", "imp_ba"))):
            ctx = hh[pos][:C]
            direction = ("vanish" if pid == "1" else "appear") if (viol == "permanence" and empty_b) else viol
            for nm, pr in MODELS.items():
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    p = pr.rollout_kv(ctx.reshape(1, C * S, -1), torch.arange(C * S, device=dev).unsqueeze(0), K).float()[0].view(K, S, -1)
                rec = dict(block=m["block_id"], model=nm, pair=pid, mv=mv, dir=direction, held=arms["pos_a"]["source_block"] in held)
                for s in SS:
                    ps = pool(p, s); fp = pool(hh[pos][C:], s); fi = pool(hh[imp][C:], s)
                    ep = (ps - fp).abs().mean(-1); ei = (ps - fi).abs().mean(-1)     # (K, n_s)
                    oi = [(j, c) for j in range(K) for c in pool_mask(otok[j], s)]
                    jj = torch.tensor([x[0] for x in oi], device=dev); cc = torch.tensor([x[1] for x in oi], device=dev)
                    rec[f"full|{s}"] = bool(ei.mean() > ep.mean())
                    rec[f"obj|{s}"] = bool(ei[jj, cc].mean() > ep[jj, cc].mean()) if len(oi) else None
                recs.append(rec)
        if bi % 40 == 0:
            print(f"VOBench {bi}/{len(meta)} ({time.time() - t0:.0f}s)", flush=True)
    out = {}
    groups = {"static": lambda r: r["mv"] == "static", "moving": lambda r: r["mv"] == "moving"}
    for mv in ("static", "moving"):
        for d in ("vanish", "appear", "shape", "color"):
            groups[f"{mv}|{d}"] = (lambda mv_, d_: lambda r: r["mv"] == mv_ and r["dir"] == d_)(mv, d)
    for nm in MODELS:
        for split, sf in (("all", lambda r: True), ("held", lambda r: r["held"])):
            for g, fn in groups.items():
                sel = [r for r in recs if r["model"] == nm and sf(r) and fn(r) and all(r[f"obj|{s_}"] is not None for s_ in SS)]
                if not sel:
                    continue
                e = {"n": len(sel)}
                for kind in ("obj", "full"):
                    for s in SS:
                        e[f"{kind}|{s}"] = round(100 * float(np.mean([r[f"{kind}|{s}"] for r in sel])), 1)
                    # s 대비 s = 1 짝 대응 차 (block bootstrap)
                    blocks = np.array([r["block"] for r in sel]); ub = np.unique(blocks)
                    for s in SS[1:]:
                        dvec = np.array([float(r[f"{kind}|{s}"]) - float(r[f"{kind}|1"]) for r in sel]) * 100
                        bs = [np.concatenate([dvec[blocks == u] for u in rng_b.choice(ub, len(ub))]).mean() for _ in range(1000)]
                        e[f"{kind}|d{s}"] = [round(float(dvec.mean()), 1), round(float(np.percentile(bs, 2.5)), 1), round(float(np.percentile(bs, 97.5)), 1)]
                out[f"{nm}|{split}|{g}"] = e
    res["vobench"] = out
    for nm in MODELS:
        print(f"\n== VOBench {nm} (전체 216 block) 물체 자리 정확도 s = {SS}", flush=True)
        for g in groups:
            e = out.get(f"{nm}|all|{g}")
            if e:
                print(f"   {g:18s} n {e['n']:3d} obj {[e[f'obj|{s}'] for s in SS]} full {[e[f'full|{s}'] for s in SS]} | obj d(s−1) {[e[f'obj|d{s}'] for s in SS[1:]]}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)

# ------------------------------------------------------------------ (2) IntPhys 2
if "2" in a.parts:
    C2 = "/workspace/auto_research_cache/intphys2/Main/"
    meta = json.load(open(C2 + "meta.json")); H2 = np.load(C2 + "h.npy", mmap_mode="r"); P2 = np.load(C2 + "p_tf.npy", mmap_mode="r")
    rows = {nm: [] for nm in MODELS}
    for i, m in enumerate(meta):
        h = torch.from_numpy(np.asarray(H2[m["row"]], np.float32)).to(dev)
        P = {"base": torch.from_numpy(np.asarray(P2[m["row"]], np.float32)).to(dev)}       # (31, S, D): P[b] = 블록 b+1 예측
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            P["vob"] = MODELS["vob"].forward_seq(h[:31].reshape(1, -1, D), torch.arange(31 * S, device=dev).unsqueeze(0)).float()[0].view(31, S, -1)
        for nm in MODELS:
            r = dict(scene=m["scene_index"], pair=str(m["pair_id"]), imp=bool(m["is_impossible"]), cond=m["condition"], cam=m["camera"])
            for s in SS:
                e = (pool(P[nm][4:31], s) - pool(h[5:32], s)).abs().mean((-1, -2))   # 블록 b = 4..30 → 정답 b+1
                r[f"s{s}"] = float(e.mean())
            rows[nm].append(r)
        if i % 200 == 0:
            print(f"IntPhys 2 {i}/{len(meta)} ({time.time() - t0:.0f}s)", flush=True)
    out = {}
    for nm in MODELS:
        rng = random.Random(0)
        out[nm] = {f"s{s}": decomp(rows[nm], f"s{s}", False, rng) for s in SS}
        # s 대비 s = 1 짝 대응 정확도 차 (장면 bootstrap)
        sc = collections.defaultdict(dict)
        for r in rows[nm]:
            sc[r["scene"]][r["pair"] + ("I" if r["imp"] else "P")] = r
        scenes = [v for v in sc.values() if len(v) == 4]
        for s in SS[1:]:
            dv = np.array([np.mean([float(v[p_ + "I"][f"s{s}"] > v[p_ + "P"][f"s{s}"]) - float(v[p_ + "I"]["s1"] > v[p_ + "P"]["s1"]) for p_ in ("1", "2")]) for v in scenes]) * 100
            bs = [dv[rng_b.integers(0, len(dv), len(dv))].mean() for _ in range(2000)]
            out[nm][f"d{s}"] = [round(float(dv.mean()), 1), round(float(np.percentile(bs, 2.5)), 1), round(float(np.percentile(bs, 97.5)), 1)]
        print(f"\n== IntPhys 2 {nm}: " + " | ".join(f"s{s} 정확도 {out[nm][f's{s}']['acc']:.1f} [{out[nm][f's{s}']['acc_ci'][0]:.1f},{out[nm][f's{s}']['acc_ci'][1]:.1f}] 짝 {out[nm][f's{s}']['pair1']:.0f}/{out[nm][f's{s}']['pair2']:.0f} 몫 {out[nm][f's{s}']['outcome_share']:.2f} 문맥 {out[nm][f's{s}']['mean_i']:+.5f} [{out[nm][f's{s}']['mean_i_ci'][0]:+.5f},{out[nm][f's{s}']['mean_i_ci'][1]:+.5f}]" for s in SS)
              + " | 차 " + str({s: out[nm][f"d{s}"] for s in SS[1:]}), flush=True)
    res["intphys2"] = out
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)

# ------------------------------------------------------------------ (3) 자연 보류 후보 순위
if "3" in a.parts:
    from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint
    from app.vjepa_frozen import ar as AR
    CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
    enc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
    _load_state_dict(enc, _load_checkpoint(CK)["target_encoder"], tag="t"); enc = enc.to(dev, torch.bfloat16).eval()
    MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
    NB, B_T, C_T = 10, [3, 4, 5, 6], [3, 4, 5, 6, 7, 8]
    cache = "/workspace/auto_research_cache/pmi_cont"
    Hn_ = np.load(f"{cache}/k710_320_holdout_h.npy", mmap_mode="r"); meta = json.load(open(f"{cache}/k710_320_holdout_meta.json"))
    torch.manual_seed(0); crng = random.Random(1)
    idx = torch.arange(4 * S, device=dev).unsqueeze(0)
    L1 = lambda x, y: float((x - y).abs().mean())
    rows = []
    for i, m in enumerate(meta):
        vr = decord.VideoReader(m["path"], num_threads=1)
        fr = vr.get_batch(list(range(m["start"], m["start"] + (2 * NB - 1) * m["fstp"] + 1, m["fstp"]))).asnumpy()
        Hh, Ww = fr.shape[1:3]; side = min(Hh, Ww); y0, x0 = (Hh - side) // 2, (Ww - side) // 2
        x = torch.from_numpy(fr[:, y0:y0 + side, x0:x0 + side]).to(dev).permute(0, 3, 1, 2).float()
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False).clamp(0, 255).round()
        with torch.no_grad():
            xn = (x + torch.randint(0, 2, x.shape, device=dev).float() * 2 - 1).clamp(0, 255)
            xxn = (xn / 255).permute(1, 0, 2, 3)[None]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hn = AR.encode_blocks(enc, ((xxn - MEAN) / STD).to(torch.bfloat16), 2)[0].view(NB, S, -1).float()
        h = torch.from_numpy(np.asarray(Hn_[i], np.float32)).to(dev)
        X = torch.stack([h[c:c + 4] for c in range(0, 6)]).reshape(6, 4 * S, D)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            P = MODELS["base"].forward_seq(X, idx.expand(6, -1)).float()[:, -S:]
        cands = {t: sorted([t + 1] + crng.sample([k for k in range(NB) if k != t + 1], 7)) for t in C_T}
        r = {}
        for s in SS:
            Ps, hns, hs = pool(P, s), pool(hn, s), pool(h, s)
            r[f"B|{s}"] = float(np.mean([L1(Ps[t - 3], hns[t + 1]) < L1(Ps[t - 3], hns[t]) for t in B_T]))
            r[f"B_copy|{s}"] = float(np.mean([L1(hs[t], hns[t + 1]) < L1(hs[t], hns[t]) for t in B_T]))
            ac, acc_c, acc_cc = [], [], []
            for t in C_T:
                dd = {k: L1(Ps[t - 3], hns[k]) for k in cands[t]}; ac.append(float(min(dd, key=dd.get) == t + 1))
                mu = torch.stack([hns[k] for k in cands[t]]).mean(0)
                dc = {k: L1(hns[k], mu) for k in cands[t]}; acc_c.append(float(min(dc, key=dc.get) == t + 1))
                dh = {k: L1(hns[k], hs[t]) for k in cands[t]}; acc_cc.append(float(min(dh, key=dh.get) == t + 1))
            r[f"C|{s}"] = float(np.mean(ac)); r[f"C_central|{s}"] = float(np.mean(acc_c)); r[f"C_copycont|{s}"] = float(np.mean(acc_cc))
        rows.append(r)
        if i % 50 == 0:
            print(f"K710 {i}/{len(meta)} ({time.time() - t0:.0f}s)", flush=True)
    out = {}
    for s in SS:
        out[f"s{s}"] = {k: boot_mean([100 * rr[f"{k}|{s}"] for rr in rows]) for k in ("B", "B_copy", "C", "C_central", "C_copycont")}
        out[f"s{s}"]["C_minus_copycont"] = boot_mean([100 * (rr[f"C|{s}"] - rr[f"C_copycont|{s}"]) for rr in rows])
        print(f"K710 s{s}: 참 vs 멈춤 승률 {out[f's{s}']['B']} (복사 판정자 {out[f's{s}']['B_copy']}) · 8 후보 순위 {out[f's{s}']['C']} · 중심성 {out[f's{s}']['C_central']} · 복사 연속 {out[f's{s}']['C_copycont']} · AR − 복사 연속 {out[f's{s}']['C_minus_copycont']}", flush=True)
    res["k710_candidates"] = out
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
print("저장", a.out, f"({time.time() - t0:.0f}s)")
