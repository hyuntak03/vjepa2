"""[비판] 관문 8 (측정 도구) 검사: "AR 예측이 참 다음 블록보다 h_t (멈춤) 에 가깝다" 가 블록마다 새 nuisance 의 지문을 닮아서인가.
h_t 는 문맥 안에 있으니, 예측이 h_t 를 일부 복사하면 그 블록의 nuisance 까지 닮는다. 그래서 후보를 모두 **새 잡음으로 다시 인코딩** 한다:
같은 장들에 ±1/255 픽셀 잡음 (장마다 독립) 을 더해 목표 encoder 로 다시 인코딩 → h' (블록마다 nuisance 가 새로 뽑힌다; 문맥은 깨끗한 h 그대로).
B′: 참 h'_t+1 vs h'_t · h'_t−1 · h'_t+2 · h'_t+3 (모두 다시 인코딩) — 참이 덜 놀라운 비율.
C′: 8 후보 순위 (같은 후보 묶음 규칙, 후보 = h') — AR · 중심성 · 복사 연속 (h_t 와의 거리, h'_t 포함; 깨끗한 h_t 와 같지 않으므로 뺄 필요 없음).
읽기 · 크기 · 반올림은 cache_natural 과 같다 (meta 의 경로 · 시작 · 간격).
"""
import os, sys, json, argparse, random, collections, numpy as np, torch, torch.nn.functional as F, decord

REPO = "/workspace/world_model/vjepa2"
sys.path.insert(0, os.path.join(REPO, "auto_research/scripts")); sys.path.insert(0, REPO)
_argv = list(sys.argv); sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT  # noqa: E402
sys.argv = _argv
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint  # noqa: E402
from app.vjepa_frozen import ar as AR  # noqa: E402

MODELS = {"base": ("natural_ar_vw_state", "latest.pt"), "vob": ("natural_ar_vw_state_vob", "latest.pt"),
          "syn": ("natural_ar_vw_state_phys3d_only", "latest.pt")}
S, D, NB = 256, 1280, 10
B_T = [3, 4, 5, 6]; C_T = [3, 4, 5, 6, 7, 8]


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
    ap.add_argument("--out", default=os.path.join(REPO, "auto_research/results/crit_pmi_fresh.json"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(8); torch.manual_seed(0)
    dev = torch.device("cuda")
    CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
    enc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
    _load_state_dict(enc, _load_checkpoint(CK)["target_encoder"], tag="t"); enc = enc.to(dev, torch.bfloat16).eval()
    MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
    nets = {k: FT.load_pred(r, c).to(dev).eval() for k, (r, c) in MODELS.items()}
    idx = torch.arange(4 * S, device=dev).unsqueeze(0)
    res = {}
    for st in a.sets.split(","):
        H = np.load(os.path.join(a.cache, f"{st}_h.npy"), mmap_mode="r"); U = np.load(os.path.join(a.cache, f"{st}_U.npy"))
        meta = json.load(open(os.path.join(a.cache, f"{st}_meta.json")))
        N = len(meta) if not a.limit else a.limit
        med = np.median(U[:N, 4:10].astype(np.float32).reshape(N, 6, -1, 2), axis=2)
        still = (np.linalg.norm(med, axis=-1).max(1) < 0.25)
        crng = random.Random(1)
        rows, chg = [], []
        for i in range(N):
            m = meta[i]
            vr = decord.VideoReader(m["path"], num_threads=1)
            fr = vr.get_batch(list(range(m["start"], m["start"] + (2 * NB - 1) * m["fstp"] + 1, m["fstp"]))).asnumpy()
            Hh, Ww = fr.shape[1:3]; side = min(Hh, Ww); y0, x0 = (Hh - side) // 2, (Ww - side) // 2
            x = torch.from_numpy(fr[:, y0:y0 + side, x0:x0 + side]).to(dev).permute(0, 3, 1, 2).float()
            x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False).clamp(0, 255).round()
            with torch.no_grad():
                xx = (x / 255).permute(1, 0, 2, 3)[None]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hc = AR.encode_blocks(enc, ((xx - MEAN) / STD).to(torch.bfloat16), 2)[0].view(NB, S, -1).float()
                xn = (x + torch.randint(0, 2, x.shape, device=dev).float() * 2 - 1).clamp(0, 255)     # 장마다 독립 ±1/255
                xxn = (xn / 255).permute(1, 0, 2, 3)[None]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hn = AR.encode_blocks(enc, ((xxn - MEAN) / STD).to(torch.bfloat16), 2)[0].view(NB, S, -1).float()
            h = torch.from_numpy(np.asarray(H[i], dtype=np.float32)).to(dev)
            # 캐시 재현 확인 (같은 읽기면 hc ≈ h) 과 다시 인코딩한 변화 크기
            chg.append(dict(recon=float((hc - h).abs().mean() / h.abs().mean()), noise=float((hn - hc).abs().mean() / hc.abs().mean())))
            X = torch.stack([h[c:c + 4] for c in range(0, 6)]).reshape(6, 4 * S, D)
            r = {"still": bool(still[i])}
            cands = {t: sorted([t + 1] + crng.sample([k for k in range(NB) if k != t + 1], 7)) for t in C_T}
            with torch.no_grad():
                for mk, net in nets.items():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        P = net.forward_seq(X, idx.expand(6, -1)).float()[:, -S:]
                    for t in B_T:
                        p = P[t - 3]
                        r[f"B|{mk}|{t}|true"] = l1(p, hn[t + 1])
                        for nm, k in (("h_t", t), ("h_tm1", t - 1), ("h_tp2", t + 2), ("h_tp3", t + 3)):
                            r[f"B|{mk}|{t}|{nm}"] = l1(p, hn[k])
                        r[f"B|{mk}|{t}|true_clean"] = l1(p, h[t + 1]); r[f"B|{mk}|{t}|h_t_clean"] = l1(p, h[t])
                    for t in C_T:
                        p = P[t - 3]
                        d = {k: l1(p, hn[k]) for k in cands[t]}
                        r[f"C|{mk}|{t}|pick"] = int(min(d, key=d.get) - (t + 1))
                for t in C_T:
                    Cm = torch.stack([hn[k] for k in cands[t]]); mu = Cm.mean(0)
                    dc = {k: l1(hn[k], mu) for k in cands[t]}
                    r[f"C|central|{t}|pick"] = int(min(dc, key=dc.get) - (t + 1))
                    dh = {k: l1(hn[k], h[t]) for k in cands[t]}                      # 깨끗한 h_t 와의 거리, h'_t 포함
                    r[f"C|copycont|{t}|pick"] = int(min(dh, key=dh.get) - (t + 1))
                for t in B_T:
                    for nm, k in (("true", t + 1), ("h_t", t), ("h_tm1", t - 1), ("h_tp2", t + 2), ("h_tp3", t + 3)):
                        r[f"B|copycont|{t}|{nm}"] = l1(h[t], hn[k])
            rows.append(r)
            if i % 50 == 0:
                print(st, i, chg[-1], flush=True)
        brng = np.random.default_rng(0)
        out = {"n": N, "recon_rel": float(np.mean([c["recon"] for c in chg])), "noise_rel": float(np.mean([c["noise"] for c in chg]))}
        for sub, sel in (("all", np.ones(N, bool)), ("still", still)):
            rs = [rr for rr, s_ in zip(rows, sel) if s_]
            o = {"n": len(rs)}
            for mk in list(MODELS) + ["copycont"]:
                for neg in ("h_t", "h_tm1", "h_tp2", "h_tp3"):
                    win = [np.mean([float(rr[f"B|{mk}|{t}|true"] < rr[f"B|{mk}|{t}|{neg}"]) for t in B_T]) for rr in rs]
                    rel = [np.mean([(rr[f"B|{mk}|{t}|{neg}"] - rr[f"B|{mk}|{t}|true"]) / rr[f"B|{mk}|{t}|true"] for t in B_T]) for rr in rs]
                    o[f"B|{mk}|{neg}"] = {"win": boot(win, brng), "rel": boot(rel, brng)}
            for mk in MODELS:
                win = [np.mean([float(rr[f"B|{mk}|{t}|true_clean"] < rr[f"B|{mk}|{t}|h_t_clean"]) for t in B_T]) for rr in rs]
                o[f"B|{mk}|h_t_clean_ref"] = {"win": boot(win, brng)}
            for mk in list(MODELS) + ["central", "copycont"]:
                acc = [np.mean([float(rr[f"C|{mk}|{t}|pick"] == 0) for t in C_T]) for rr in rs]
                o[f"C|{mk}"] = {"acc": boot(acc, brng)}
                cnt = collections.Counter(rr[f"C|{mk}|{t}|pick"] for rr in rs for t in C_T); tot = sum(cnt.values())
                o[f"C|{mk}|pick_dist"] = {str(k): round(v / tot, 3) for k, v in sorted(cnt.items())}
            for mk in MODELS:
                for crit in ("central", "copycont"):
                    dd = [np.mean([float(rr[f"C|{mk}|{t}|pick"] == 0) - float(rr[f"C|{crit}|{t}|pick"] == 0) for t in C_T]) for rr in rs]
                    o[f"C|{mk}|minus_{crit}"] = {"d": boot(dd, brng)}
            out[sub] = o
        res[st] = out
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"\n== {st} 창 {N} · 캐시 재현 상대차 {out['recon_rel']:.4f} · ±1/255 다시 인코딩 상대 변화 {out['noise_rel']:.3f}", flush=True)
        for sub in ("all", "still"):
            o = out[sub]; print(f" [{sub}] n {o['n']}")
            for mk in list(MODELS) + ["copycont"]:
                print("  B′ " + f"{mk:8s} " + " ".join(f"{neg} {100*o[f'B|{mk}|{neg}']['win'][0]:5.1f} [{100*o[f'B|{mk}|{neg}']['win'][1]:4.1f},{100*o[f'B|{mk}|{neg}']['win'][2]:4.1f}]({100*o[f'B|{mk}|{neg}']['rel'][0]:+5.1f}%)" for neg in ("h_t", "h_tm1", "h_tp2", "h_tp3"))
                      + (f"   (깨끗한 후보 h_t: {100*o[f'B|{mk}|h_t_clean_ref']['win'][0]:5.1f})" if mk in MODELS else ""))
            for mk in list(MODELS) + ["central", "copycont"]:
                c = o[f"C|{mk}"]["acc"]
                extra = "".join(f"  −{crit} {100*o[f'C|{mk}|minus_{crit}']['d'][0]:+5.1f} [{100*o[f'C|{mk}|minus_{crit}']['d'][1]:+5.1f},{100*o[f'C|{mk}|minus_{crit}']['d'][2]:+5.1f}]" for crit in ("central", "copycont")) if mk in MODELS else ""
                print(f"  C′ {mk:8s} 순위 {100*c[0]:5.1f} [{100*c[1]:5.1f},{100*c[2]:5.1f}]{extra}  고른 자리 {o[f'C|{mk}|pick_dist']}")


if __name__ == "__main__":
    main()
