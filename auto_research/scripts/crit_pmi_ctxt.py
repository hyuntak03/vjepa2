"""[비판] 영상 문맥 모델도 "멈춘 세계" 를 예측하나 (학습 없음; 사전 등록 진행판 KST 20:08).

crit_pmi_cont (A, 깨끗한 후보) · crit_pmi_fresh (B′ · C′, ±1/255 로 다시 인코딩한 후보) 와 같은 창 · 같은 규칙으로,
영상 문맥 모델 (natural_ar_vw_state_ctxt_video latest) 과 블록 AR (natural_ar_vw_state latest) 을 같은 실행에서 잰다.
영상 문맥 한 걸음: 문맥 장 (4 블록 = 8 장) → online encoder 영상 모드 (문맥 마스크 = 전부) → forward_state (미래 관측 없음) → render = 다음 블록.
블록 AR 한 걸음: 문맥 4 블록 LN(target) → forward_seq → 마지막 위치 = 다음 블록.
부 판 (탐색): 문맥 8 블록 (16 장) → 블록 8 · 9 예측, (a) · (b) 만.
"""
import os, sys, json, argparse, random, collections, numpy as np, torch, torch.nn.functional as F, yaml, decord

REPO = "/workspace/world_model/vjepa2"
sys.path.insert(0, os.path.join(REPO, "auto_research/scripts")); sys.path.insert(0, REPO)
_argv = list(sys.argv); sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT  # noqa: E402
sys.argv = _argv
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint, _clean_backbone_key  # noqa: E402
from app.vjepa_frozen import utils as UU  # noqa: E402
from app.vjepa_frozen import ar as AR  # noqa: E402

S, D, NB = 256, 1280, 10
A_DESIGN = [(8, 4, [(0, 4)]), (9, 5, [(1, 4), (0, 5)])]
B_T = [3, 4, 5, 6]; C_T = [3, 4, 5, 6, 7, 8]; L_T = [7, 8]          # L = 문맥 8 블록 부 판 (참 = 8 · 9)


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
    ap.add_argument("--out", default=os.path.join(REPO, "auto_research/results/crit_pmi_ctxt.json"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check", type=int, default=0, help="1 = 경로 확인만 (관측 넣은 판 첫 렌더 = 관측 없는 판)")
    a = ap.parse_args()
    torch.set_num_threads(8); torch.manual_seed(0)
    dev = torch.device("cuda")
    CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
    sd = _load_checkpoint(CK)
    tgt = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
    _load_state_dict(tgt, sd["target_encoder"], tag="t"); tgt = tgt.to(dev, torch.bfloat16).eval()
    onl = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=32, use_rope=True, uniform_power=False)
    _load_state_dict(onl, sd["encoder"], tag="o"); onl = onl.to(dev, torch.bfloat16).eval()
    del sd
    Dd = os.path.join(REPO, "z_training/runs/natural_ar_vw_state_ctxt_video")
    cfg = yaml.safe_load(open(os.path.join(Dd, "config.yaml")))
    ctxt = UU.build_predictor(cfg["model"], 1280, None, torch.device("cpu"), int(cfg["data"].get("n_frames", 64)))
    ctxt.load_state_dict({k: v for k, v in _clean_backbone_key(torch.load(os.path.join(Dd, "latest.pt"), map_location="cpu", weights_only=False)["predictor"]).items()
                          if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}, strict=True)
    ctxt = ctxt.to(dev).eval()
    base = FT.load_pred("natural_ar_vw_state", "latest.pt").to(dev).eval()
    MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
    empty = torch.zeros(1, 0, ctxt.predictor_embed.out_features if hasattr(ctxt.predictor_embed, "out_features") else 384, device=dev)

    def ctxt_next(xn, c, nc, h=None):
        """xn (1,3,T,256,256) 정규화 장, 문맥 블록 c..c+nc−1 → 블록 c+nc 예측 (S, D). h 를 주면 관측 3 블록을 넣은 판의 첫 렌더."""
        clip = xn[:, :, 2 * c:2 * (c + nc)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = onl(clip.to(torch.bfloat16), masks=[torch.arange(nc * S, device=dev).unsqueeze(0)]).float()
            if h is None:
                st_ = ctxt.forward_state(z, empty, torch.arange(nc * S, device=dev).unsqueeze(0), nc)
            else:
                st_ = ctxt.forward_state(z, ctxt.embed_obs(h[c + nc:c + nc + 3].reshape(1, -1, D)), torch.arange((nc + 3) * S, device=dev).unsqueeze(0), nc)
            return ctxt.render(st_).float()[0, :S]

    def base_next(h, c, nc):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return base.forward_seq(h[c:c + nc].reshape(1, nc * S, D), torch.arange(nc * S, device=dev).unsqueeze(0)).float()[0, -S:]

    res = {}
    for st in a.sets.split(","):
        H = np.load(os.path.join(a.cache, f"{st}_h.npy"), mmap_mode="r"); U = np.load(os.path.join(a.cache, f"{st}_U.npy"))
        meta = json.load(open(os.path.join(a.cache, f"{st}_meta.json")))
        N = len(meta) if not a.limit else a.limit
        med = np.median(U[:N, 4:10].astype(np.float32).reshape(N, 6, -1, 2), axis=2)
        still = (np.linalg.norm(med, axis=-1).max(1) < 0.25)
        rng = random.Random(0); perm = list(range(N))
        while any(p == i for i, p in enumerate(perm)):
            rng.shuffle(perm)
        crng = random.Random(1)
        PRED = {"ctxt": [], "base": []}                                            # 창마다 (6, S, D) fp16 CPU: 문맥 시작 c → 블록 c+4
        PRED8 = {"ctxt": [], "base": []}                                           # (2, S, D): 문맥 0–7 → 8, 1–8 → 9
        HN, CANDS = [], []
        for i in range(N):
            m = meta[i]
            vr = decord.VideoReader(m["path"], num_threads=1)
            fr = vr.get_batch(list(range(m["start"], m["start"] + (2 * NB - 1) * m["fstp"] + 1, m["fstp"]))).asnumpy()
            Hh, Ww = fr.shape[1:3]; side = min(Hh, Ww); y0, x0 = (Hh - side) // 2, (Ww - side) // 2
            x = torch.from_numpy(fr[:, y0:y0 + side, x0:x0 + side]).to(dev).permute(0, 3, 1, 2).float()
            x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False).clamp(0, 255).round()
            h = torch.from_numpy(np.asarray(H[i], dtype=np.float32)).to(dev)
            with torch.no_grad():
                xn = ((x / 255).permute(1, 0, 2, 3)[None] - MEAN) / STD
                if a.check:
                    p0 = ctxt_next(xn, 0, 4); p1 = ctxt_next(xn, 0, 4, h)
                    print("경로 확인: 관측 없음 vs 관측 3 블록 첫 렌더 상대차", float((p0 - p1).abs().mean() / p1.abs().mean()),
                          "· 예측 vs 정답 L1", l1(p0, h[4]), "· 복사", l1(h[3], h[4]), flush=True)
                    if i >= 2:
                        return
                    continue
                xr = (x + torch.randint(0, 2, x.shape, device=dev).float() * 2 - 1).clamp(0, 255)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hn = AR.encode_blocks(tgt, (((xr / 255).permute(1, 0, 2, 3)[None] - MEAN) / STD).to(torch.bfloat16), 2)[0].view(NB, S, -1).float()
                PRED["ctxt"].append(torch.stack([ctxt_next(xn, c, 4) for c in range(6)]).half().cpu())
                PRED["base"].append(torch.stack([base_next(h, c, 4) for c in range(6)]).half().cpu())
                PRED8["ctxt"].append(torch.stack([ctxt_next(xn, c, 8) for c in range(2)]).half().cpu())
                PRED8["base"].append(torch.stack([base_next(h, c, 8) for c in range(2)]).half().cpu())
            HN.append(hn.half().cpu())
            CANDS.append({t: sorted([t + 1] + crng.sample([k for k in range(NB) if k != t + 1], 7)) for t in C_T})
            if i % 50 == 0:
                print(st, i, flush=True)
        rows = []
        for i in range(N):
            h = torch.from_numpy(np.asarray(H[i], dtype=np.float32)); hn = HN[i].float()
            r = {"still": bool(still[i])}
            for mk in ("ctxt", "base"):
                P = PRED[mk][i].float(); Pj = PRED[mk][perm[i]].float()
                for (y, cs, others) in A_DESIGN:                                    # A (깨끗한 후보)
                    r[f"A|{mk}|{y}|own"] = l1(P[cs], h[y])
                    for (co, g) in others:
                        r[f"A|{mk}|{y}|same_g{g}"] = l1(P[co], h[y])
                    r[f"A|{mk}|{y}|other_clip"] = l1(Pj[cs], h[y])
                for t in B_T:                                                       # B′ (다시 인코딩한 후보)
                    p = P[t - 3]
                    r[f"B|{mk}|{t}|true"] = l1(p, hn[t + 1])
                    for nm, k in (("h_t", t), ("h_tm1", t - 1), ("h_tp2", t + 2), ("h_tp3", t + 3)):
                        r[f"B|{mk}|{t}|{nm}"] = l1(p, hn[k])
                for t in C_T:                                                       # C′
                    p = P[t - 3]
                    d = {k: l1(p, hn[k]) for k in CANDS[i][t]}
                    r[f"C|{mk}|{t}|pick"] = int(min(d, key=d.get) - (t + 1))
                P8 = PRED8[mk][i].float()                                           # 부 판 (문맥 8)
                for j_, t in enumerate(L_T):
                    p = P8[j_]
                    r[f"L|{mk}|{t}|true"] = l1(p, hn[t + 1]); r[f"L|{mk}|{t}|h_t"] = l1(p, hn[t])
                    d = {k: l1(p, hn[k]) for k in CANDS[i][t]}
                    r[f"L|{mk}|{t}|pick"] = int(min(d, key=d.get) - (t + 1))
            for (y, cs, others) in A_DESIGN:                                        # 복사 (A)
                r[f"A|copy|{y}|own"] = l1(h[cs + 3], h[y])
                for (co, g) in others:
                    r[f"A|copy|{y}|same_g{g}"] = l1(h[co + 3], h[y])
            for t in C_T:
                Cm = torch.stack([hn[k] for k in CANDS[i][t]]); mu = Cm.mean(0)
                dc = {k: l1(hn[k], mu) for k in CANDS[i][t]}
                r[f"C|central|{t}|pick"] = int(min(dc, key=dc.get) - (t + 1))
                dh = {k: l1(hn[k], h[t]) for k in CANDS[i][t]}
                r[f"C|copycont|{t}|pick"] = int(min(dh, key=dh.get) - (t + 1))
            for t in L_T:
                r[f"L|central|{t}|pick"] = r[f"C|central|{t}|pick"]; r[f"L|copycont|{t}|pick"] = r[f"C|copycont|{t}|pick"]
            rows.append(r)
        brng = np.random.default_rng(0)
        out = {"n": N, "n_still": int(still.sum())}
        for sub, sel in (("all", np.ones(N, bool)), ("still", still)):
            rs = [rr for rr, s_ in zip(rows, sel) if s_]
            o = {"n": len(rs)}
            for mk in ("ctxt", "base", "copy"):
                for neg in ("same", "other_clip"):
                    if mk == "copy" and neg == "other_clip":
                        continue
                    win = []
                    for rr in rs:
                        w_ = []
                        for (y, cs, others) in A_DESIGN:
                            negs = [f"A|{mk}|{y}|same_g{g}" for (_, g) in others] if neg == "same" else [f"A|{mk}|{y}|other_clip"]
                            w_ += [float(rr[f"A|{mk}|{y}|own"] < rr[nk]) for nk in negs]
                        win.append(np.mean(w_))
                    o[f"A|{mk}|{neg}"] = boot(win, brng)
            for mk in ("ctxt", "base"):
                dw = [np.mean([float(rr[f"A|{mk}|{y}|own"] < rr[f"A|{mk}|{y}|same_g{g}"]) - float(rr[f"A|copy|{y}|own"] < rr[f"A|copy|{y}|same_g{g}"])
                               for (y, cs, others) in A_DESIGN for (_, g) in others]) for rr in rs]
                o[f"A|{mk}|minus_copy"] = boot(dw, brng)
                for neg in ("h_t", "h_tm1", "h_tp2", "h_tp3"):
                    o[f"B|{mk}|{neg}"] = boot([np.mean([float(rr[f"B|{mk}|{t}|true"] < rr[f"B|{mk}|{t}|{neg}"]) for t in B_T]) for rr in rs], brng)
                o[f"C|{mk}"] = boot([np.mean([float(rr[f"C|{mk}|{t}|pick"] == 0) for t in C_T]) for rr in rs], brng)
                cnt = collections.Counter(rr[f"C|{mk}|{t}|pick"] for rr in rs for t in C_T); tot = sum(cnt.values())
                o[f"C|{mk}|pick_h_t"] = round(cnt.get(-1, 0) / tot, 3)
                for crit in ("central", "copycont"):
                    o[f"C|{mk}|minus_{crit}"] = boot([np.mean([float(rr[f"C|{mk}|{t}|pick"] == 0) - float(rr[f"C|{crit}|{t}|pick"] == 0) for t in C_T]) for rr in rs], brng)
                o[f"L|{mk}|h_t"] = boot([np.mean([float(rr[f"L|{mk}|{t}|true"] < rr[f"L|{mk}|{t}|h_t"]) for t in L_T]) for rr in rs], brng)
                o[f"L|{mk}|rank"] = boot([np.mean([float(rr[f"L|{mk}|{t}|pick"] == 0) for t in L_T]) for rr in rs], brng)
                for crit in ("central", "copycont"):
                    o[f"L|{mk}|minus_{crit}"] = boot([np.mean([float(rr[f"L|{mk}|{t}|pick"] == 0) - float(rr[f"L|{crit}|{t}|pick"] == 0) for t in L_T]) for rr in rs], brng)
            # 영상 문맥 − 블록 AR (짝 대응)
            o["B|ctxt_minus_base|h_t"] = boot([np.mean([float(rr[f"B|ctxt|{t}|true"] < rr[f"B|ctxt|{t}|h_t"]) - float(rr[f"B|base|{t}|true"] < rr[f"B|base|{t}|h_t"]) for t in B_T]) for rr in rs], brng)
            o["C|ctxt_minus_base"] = boot([np.mean([float(rr[f"C|ctxt|{t}|pick"] == 0) - float(rr[f"C|base|{t}|pick"] == 0) for t in C_T]) for rr in rs], brng)
            o["A|ctxt_minus_base"] = boot([np.mean([float(rr[f"A|ctxt|{y}|own"] < rr[f"A|ctxt|{y}|same_g{g}"]) - float(rr[f"A|base|{y}|own"] < rr[f"A|base|{y}|same_g{g}"])
                                                    for (y, cs, others) in A_DESIGN for (_, g) in others]) for rr in rs], brng)
            for mk in ("central", "copycont"):
                o[f"C|{mk}"] = boot([np.mean([float(rr[f"C|{mk}|{t}|pick"] == 0) for t in C_T]) for rr in rs], brng)
            out[sub] = o
        res[st] = out
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"\n== {st} 창 {N} (카메라 정지 {int(still.sum())})", flush=True)
        f = lambda v: f"{100*v[0]:5.1f} [{100*v[1]:5.1f},{100*v[2]:5.1f}]"
        for sub in ("all", "still"):
            o = out[sub]; print(f" [{sub}] n {o['n']}")
            for mk in ("ctxt", "base"):
                print(f"  {mk:5s} (a) 참<멈춤′ {f(o[f'B|{mk}|h_t'])} · 뒤로 {100*o[f'B|{mk}|h_tm1'][0]:5.1f} · +2 {100*o[f'B|{mk}|h_tp2'][0]:5.1f} · +3 {100*o[f'B|{mk}|h_tp3'][0]:5.1f}")
                print(f"        (b) 8 후보 {f(o[f'C|{mk}'])} (h_t 고른 몫 {o[f'C|{mk}|pick_h_t']}) −중심성 {f(o[f'C|{mk}|minus_central'])} −복사연속 {f(o[f'C|{mk}|minus_copycont'])}")
                print(f"        (c) 같은 클립 {f(o[f'A|{mk}|same'])} 다른 클립 {100*o[f'A|{mk}|other_clip'][0]:5.1f} 모델−복사 {f(o[f'A|{mk}|minus_copy'])}")
                print(f"        부 판 문맥 8: 참<멈춤′ {f(o[f'L|{mk}|h_t'])} 순위 {f(o[f'L|{mk}|rank'])} −중심성 {f(o[f'L|{mk}|minus_central'])} −복사연속 {f(o[f'L|{mk}|minus_copycont'])}")
            print(f"  복사 같은 클립 {f(o['A|copy|same'])} · 중심성 {f(o['C|central'])} · 복사연속 {f(o['C|copycont'])}")
            print(f"  영상 문맥 − 블록 AR: (a) {f(o['B|ctxt_minus_base|h_t'])} (b) {f(o['C|ctxt_minus_base'])} (c) {f(o['A|ctxt_minus_base'])}")


if __name__ == "__main__":
    main()
