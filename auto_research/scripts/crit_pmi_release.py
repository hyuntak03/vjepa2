"""[비판] 릴리즈 V-JEPA 2 predictor (가린 자리 채우기로 학습) 의 멈춤 편향 (학습 없음; 사전 등록 진행판 KST 20:25).

crit_pmi_ctxt.py 와 같은 창 · 같은 판정. 모델 경로만 릴리즈다 (고정 문맥 채점 프로토콜과 같은 bundle, 코드 수정 없음):
  창 20 장을 한 클립 (튜블릿 10 = 우리 블록 10) 으로. 문맥 encoder (online) 에 튜블릿 c..c+nc−1 토큰만 남기고 (문맥 마스크),
  predictor 가 튜블릿 c+nc 자리 (256 토큰) 를 mask token 0 으로 묻는다 = 문맥 바로 다음 블록. 시간 위치는 창 안 절대 위치.
  후보 = 목표 encoder (EMA) 를 창 20 장 전체에 영상 모드로 넣은 튜블릿별 LN 토큰. 다시 인코딩 = 같은 창 + ±1/255.
  판정자 (중심성 · 복사 연속) 도 이 공간에서. float32 가중치 + float16 autocast (프로토콜 관례).
"""
import os, sys, json, argparse, random, collections, numpy as np, torch, torch.nn.functional as F, decord

REPO = "/workspace/world_model/vjepa2"
sys.path.insert(0, REPO)
from analysis.intphys2.model import build_from_config  # noqa: E402

S, D, NB = 256, 1280, 10
A_DESIGN = [(8, 4, [(0, 4)]), (9, 5, [(1, 4), (0, 5)])]
B_T = [3, 4, 5, 6]; C_T = [3, 4, 5, 6, 7, 8]; L_T = [7, 8]
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")


def boot(x, rng, n=4000):
    x = np.asarray(x, float); m = len(x)
    bs = np.sort([x[rng.integers(0, m, m)].mean() for _ in range(n)])
    return [float(x.mean()), float(bs[int(0.025 * n)]), float(bs[int(0.975 * n) - 1])]


def l1(a, b):
    return float((a - b).abs().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/workspace/auto_research_cache/pmi_cont")
    ap.add_argument("--sets", default="k710_320_holdout,ssv2")
    ap.add_argument("--out", default=os.path.join(REPO, "auto_research/results/crit_pmi_release.json"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(8); torch.manual_seed(0)
    dev = torch.device("cuda")
    bundle = build_from_config(dict(checkpoint=CK, arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, window_size=2 * NB,
                                    use_rope=True, uniform_power=False, dual_encoder=True, context_encoder_key="encoder",
                                    target_encoder_key="target_encoder", dtype="float32",
                                    predictor=dict(embed_dim=384, depth=12, num_heads=12, num_mask_tokens=10)), dev)
    from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint  # noqa: E402
    from app.vjepa_frozen import ar as AR  # noqa: E402
    tgt2 = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
    _load_state_dict(tgt2, _load_checkpoint(CK)["target_encoder"], tag="t2"); tgt2 = tgt2.to(dev, torch.bfloat16).eval()
    MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
    ac = lambda: torch.autocast("cuda", dtype=torch.float16)

    def target_tokens(xn):
        with ac():
            h = bundle.target_encoder(xn)
        return F.layer_norm(h.float(), (h.size(-1),))[0].view(NB, S, -1)

    def rel_next(xn, c, nc):
        ci = torch.arange(c * S, (c + nc) * S, device=dev).unsqueeze(0); ti = torch.arange((c + nc) * S, (c + nc + 1) * S, device=dev).unsqueeze(0)
        with ac():
            z = bundle.context_encoder(xn, masks=[ci])
            p = bundle.predictor(z, ci, ti, mask_index=0)
        return p.float()[0]

    def rel_multi(xn, c, nc, k):                                     # 경로 확인용: 문맥 다음 k 튜블릿
        ci = torch.arange(c * S, (c + nc) * S, device=dev).unsqueeze(0); ti = torch.arange((c + nc) * S, (c + nc + k) * S, device=dev).unsqueeze(0)
        with ac():
            z = bundle.context_encoder(xn, masks=[ci])
            p = bundle.predictor(z, ci, ti, mask_index=0)
        return p.float()[0].view(k, S, -1)

    res = {}
    for st in a.sets.split(","):
        U = np.load(os.path.join(a.cache, f"{st}_U.npy")); meta = json.load(open(os.path.join(a.cache, f"{st}_meta.json")))
        N = len(meta) if not a.limit else a.limit
        med = np.median(U[:N, 4:10].astype(np.float32).reshape(N, 6, -1, 2), axis=2)
        still = (np.linalg.norm(med, axis=-1).max(1) < 0.25)
        rng = random.Random(0); perm = list(range(N))
        while any(p == i for i, p in enumerate(perm)):
            rng.shuffle(perm)
        crng = random.Random(1)
        P4, P8, HC, HN, CANDS, HBN = [], [], [], [], [], []
        Hblk = np.load(os.path.join(a.cache, f"{st}_h.npy"), mmap_mode="r")
        for i in range(N):
            m = meta[i]
            vr = decord.VideoReader(m["path"], num_threads=1)
            fr = vr.get_batch(list(range(m["start"], m["start"] + (2 * NB - 1) * m["fstp"] + 1, m["fstp"]))).asnumpy()
            Hh, Ww = fr.shape[1:3]; side = min(Hh, Ww); y0, x0 = (Hh - side) // 2, (Ww - side) // 2
            x = torch.from_numpy(fr[:, y0:y0 + side, x0:x0 + side]).to(dev).permute(0, 3, 1, 2).float()
            x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False).clamp(0, 255).round()
            with torch.no_grad():
                xn = ((x / 255).permute(1, 0, 2, 3)[None] - MEAN) / STD
                hc = target_tokens(xn)
                if a.check:
                    p = rel_multi(xn, 0, 8, 2)
                    print("경로 확인 (문맥 튜블릿 0–7 → 8 · 9): 예측 L1", l1(p, hc[8:10]), "· 복사 (튜블릿 7)", l1(hc[7:8].expand(2, -1, -1), hc[8:10]),
                          "· 한 걸음 (4 → 다음) 예측", l1(rel_next(xn, 4, 4), hc[8]), "복사", l1(hc[7], hc[8]), flush=True)
                    if i >= 2:
                        return
                    continue
                xr = (x + torch.randint(0, 2, x.shape, device=dev).float() * 2 - 1).clamp(0, 255)
                xrn = ((xr / 255).permute(1, 0, 2, 3)[None] - MEAN) / STD
                hn = target_tokens(xrn)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hbn = AR.encode_blocks(tgt2, xrn.to(torch.bfloat16), 2)[0].view(NB, S, -1).float()
                HBN.append(hbn.half().cpu())
                P4.append(torch.stack([rel_next(xn, c, 4) for c in range(6)]).half().cpu())
                P8.append(torch.stack([rel_next(xn, c, 8) for c in range(2)]).half().cpu())
            HC.append(hc.half().cpu()); HN.append(hn.half().cpu())
            CANDS.append({t: sorted([t + 1] + crng.sample([k for k in range(NB) if k != t + 1], 7)) for t in C_T})
            if i % 50 == 0:
                print(st, i, flush=True)
        TOT = torch.stack([x_.float() for x_ in HN]).sum(0)                     # (10,S,D) 위치 틀 (자기 창은 뺀다)
        rows = []
        for i in range(N):
            h = HC[i].float(); hn = HN[i].float(); P = P4[i].float(); Pj = P4[perm[i]].float(); Q = P8[i].float()
            hbn = HBN[i].float(); hbc = torch.from_numpy(np.asarray(Hblk[i], dtype=np.float32))
            T_ = (TOT - hn) / (N - 1)
            r = {"still": bool(still[i])}
            for t in C_T:                                                           # 통제: 위치 판정자 · 위치 뺀 순위 · 블록 공간 후보 순위
                p = P[t - 3]
                d = {k: l1(hn[k], T_[t + 1]) for k in CANDS[i][t]}; r[f"C|posjudge|{t}|pick"] = int(min(d, key=d.get) - (t + 1))
                d = {k: l1(p - T_[t + 1], hn[k] - T_[k]) for k in CANDS[i][t]}; r[f"C|rel_res|{t}|pick"] = int(min(d, key=d.get) - (t + 1))
                d = {k: l1(p, hbn[k]) for k in CANDS[i][t]}; r[f"C|rel_blk|{t}|pick"] = int(min(d, key=d.get) - (t + 1))
                mu = torch.stack([hbn[k] for k in CANDS[i][t]]).mean(0)
                d = {k: l1(hbn[k], mu) for k in CANDS[i][t]}; r[f"C|central_blk|{t}|pick"] = int(min(d, key=d.get) - (t + 1))
                d = {k: l1(hbn[k], hbc[t]) for k in CANDS[i][t]}; r[f"C|copycont_blk|{t}|pick"] = int(min(d, key=d.get) - (t + 1))
            for t in B_T:
                p = P[t - 3]
                r[f"B|rel_res|{t}|win"] = float(l1(p - T_[t + 1], hn[t + 1] - T_[t + 1]) < l1(p - T_[t + 1], hn[t] - T_[t]))
                r[f"B|rel_blk|{t}|win"] = float(l1(p, hbn[t + 1]) < l1(p, hbn[t]))
                r[f"B|posjudge|{t}|win"] = float(l1(hn[t + 1], T_[t + 1]) < l1(hn[t], T_[t + 1]))
            for (y, cs, others) in A_DESIGN:
                r[f"A|rel|{y}|own"] = l1(P[cs], h[y]); r[f"A|copy|{y}|own"] = l1(h[cs + 3], h[y])
                for (co, g) in others:
                    r[f"A|rel|{y}|same_g{g}"] = l1(P[co], h[y]); r[f"A|copy|{y}|same_g{g}"] = l1(h[co + 3], h[y])
                r[f"A|rel|{y}|other_clip"] = l1(Pj[cs], h[y])
            for t in B_T:
                p = P[t - 3]
                r[f"B|rel|{t}|true"] = l1(p, hn[t + 1])
                for nm, k in (("h_t", t), ("h_tm1", t - 1), ("h_tp2", t + 2), ("h_tp3", t + 3)):
                    r[f"B|rel|{t}|{nm}"] = l1(p, hn[k])
            for t in C_T:
                p = P[t - 3]
                d = {k: l1(p, hn[k]) for k in CANDS[i][t]}; r[f"C|rel|{t}|pick"] = int(min(d, key=d.get) - (t + 1))
                mu = torch.stack([hn[k] for k in CANDS[i][t]]).mean(0)
                dc = {k: l1(hn[k], mu) for k in CANDS[i][t]}; r[f"C|central|{t}|pick"] = int(min(dc, key=dc.get) - (t + 1))
                dh = {k: l1(hn[k], h[t]) for k in CANDS[i][t]}; r[f"C|copycont|{t}|pick"] = int(min(dh, key=dh.get) - (t + 1))
            for j_, t in enumerate(L_T):
                p = Q[j_]
                r[f"L|rel|{t}|true"] = l1(p, hn[t + 1]); r[f"L|rel|{t}|h_t"] = l1(p, hn[t])
                d = {k: l1(p, hn[k]) for k in CANDS[i][t]}; r[f"L|rel|{t}|pick"] = int(min(d, key=d.get) - (t + 1))
                r[f"L|central|{t}|pick"] = r[f"C|central|{t}|pick"]; r[f"L|copycont|{t}|pick"] = r[f"C|copycont|{t}|pick"]
            rows.append(r)
        brng = np.random.default_rng(0)
        out = {"n": N, "n_still": int(still.sum())}
        for sub, sel in (("all", np.ones(N, bool)), ("still", still)):
            rs = [rr for rr, s_ in zip(rows, sel) if s_]
            o = {"n": len(rs)}
            for mk in ("rel", "copy"):
                o[f"A|{mk}|same"] = boot([np.mean([float(rr[f"A|{mk}|{y}|own"] < rr[f"A|{mk}|{y}|same_g{g}"]) for (y, cs, oth) in A_DESIGN for (_, g) in oth]) for rr in rs], brng)
            o["A|rel|other_clip"] = boot([np.mean([float(rr[f"A|rel|{y}|own"] < rr[f"A|rel|{y}|other_clip"]) for (y, cs, oth) in A_DESIGN]) for rr in rs], brng)
            o["A|rel|minus_copy"] = boot([np.mean([float(rr[f"A|rel|{y}|own"] < rr[f"A|rel|{y}|same_g{g}"]) - float(rr[f"A|copy|{y}|own"] < rr[f"A|copy|{y}|same_g{g}"])
                                                  for (y, cs, oth) in A_DESIGN for (_, g) in oth]) for rr in rs], brng)
            for neg in ("h_t", "h_tm1", "h_tp2", "h_tp3"):
                o[f"B|rel|{neg}"] = boot([np.mean([float(rr[f"B|rel|{t}|true"] < rr[f"B|rel|{t}|{neg}"]) for t in B_T]) for rr in rs], brng)
            for mk in ("rel", "central", "copycont"):
                o[f"C|{mk}"] = boot([np.mean([float(rr[f"C|{mk}|{t}|pick"] == 0) for t in C_T]) for rr in rs], brng)
            cnt = collections.Counter(rr[f"C|rel|{t}|pick"] for rr in rs for t in C_T); tot = sum(cnt.values())
            o["C|rel|pick_dist"] = {str(k): round(v / tot, 3) for k, v in sorted(cnt.items())}
            for crit in ("central", "copycont"):
                o[f"C|rel|minus_{crit}"] = boot([np.mean([float(rr[f"C|rel|{t}|pick"] == 0) - float(rr[f"C|{crit}|{t}|pick"] == 0) for t in C_T]) for rr in rs], brng)
                o[f"L|rel|minus_{crit}"] = boot([np.mean([float(rr[f"L|rel|{t}|pick"] == 0) - float(rr[f"L|{crit}|{t}|pick"] == 0) for t in L_T]) for rr in rs], brng)
            for mk in ("posjudge", "rel_res", "rel_blk", "central_blk", "copycont_blk"):
                o[f"C|{mk}"] = boot([np.mean([float(rr[f"C|{mk}|{t}|pick"] == 0) for t in C_T]) for rr in rs], brng)
            for mk in ("rel_res", "rel_blk", "posjudge"):
                o[f"B|{mk}|h_t"] = boot([np.mean([rr[f"B|{mk}|{t}|win"] for t in B_T]) for rr in rs], brng)
            for mk, crit in (("rel_res", "posjudge"), ("rel", "posjudge"), ("rel_blk", "central_blk"), ("rel_blk", "copycont_blk")):
                o[f"C|{mk}|minus_{crit}"] = boot([np.mean([float(rr[f"C|{mk}|{t}|pick"] == 0) - float(rr[f"C|{crit}|{t}|pick"] == 0) for t in C_T]) for rr in rs], brng)
            o["L|rel|h_t"] = boot([np.mean([float(rr[f"L|rel|{t}|true"] < rr[f"L|rel|{t}|h_t"]) for t in L_T]) for rr in rs], brng)
            o["L|rel|rank"] = boot([np.mean([float(rr[f"L|rel|{t}|pick"] == 0) for t in L_T]) for rr in rs], brng)
            out[sub] = o
        res[st] = out
        json.dump(res, open(a.out, "w"), indent=1)
        f = lambda v: f"{100*v[0]:5.1f} [{100*v[1]:5.1f},{100*v[2]:5.1f}]"
        print(f"\n== {st} 창 {N} (카메라 정지 {int(still.sum())})", flush=True)
        for sub in ("all", "still"):
            o = out[sub]; print(f" [{sub}] n {o['n']}")
            print(f"  (a) 참<멈춤′ {f(o['B|rel|h_t'])} · 뒤로 {100*o['B|rel|h_tm1'][0]:5.1f} · +2 {100*o['B|rel|h_tp2'][0]:5.1f} · +3 {100*o['B|rel|h_tp3'][0]:5.1f}")
            print(f"  (b) 8 후보 {f(o['C|rel'])} −중심성 {f(o['C|rel|minus_central'])} −복사연속 {f(o['C|rel|minus_copycont'])} · 판정자 중심성 {f(o['C|central'])} 복사연속 {f(o['C|copycont'])}")
            print(f"      고른 자리 {o['C|rel|pick_dist']}")
            print(f"  (c) 같은 클립 {f(o['A|rel|same'])} 다른 클립 {100*o['A|rel|other_clip'][0]:5.1f} 복사 {f(o['A|copy|same'])} 모델−복사 {f(o['A|rel|minus_copy'])}")
            print(f"  통제: 위치 판정자 순위 {f(o['C|posjudge'])} (a) {f(o['B|posjudge|h_t'])} · 릴리즈 − 위치 판정자 {f(o['C|rel|minus_posjudge'])}")
            print(f"        위치 뺀 릴리즈: (a) {f(o['B|rel_res|h_t'])} 순위 {f(o['C|rel_res'])} − 위치 판정자 {f(o['C|rel_res|minus_posjudge'])}")
            print(f"        블록 공간 후보 릴리즈: (a) {f(o['B|rel_blk|h_t'])} 순위 {f(o['C|rel_blk'])} − 중심성 {f(o['C|rel_blk|minus_central_blk'])} − 복사연속 {f(o['C|rel_blk|minus_copycont_blk'])} (판정자 {f(o['C|central_blk'])} · {f(o['C|copycont_blk'])})")
            print(f"  부 판 문맥 8: 참<멈춤′ {f(o['L|rel|h_t'])} 순위 {f(o['L|rel|rank'])} −중심성 {f(o['L|rel|minus_central'])} −복사연속 {f(o['L|rel|minus_copycont'])}")


if __name__ == "__main__":
    main()
