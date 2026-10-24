"""[비판] IntPhys 2: "상태 공간 surprise" 가 문맥 항을 만드나 — 이미 학습된 상태 감독 AR 로 (학습 없음).

모델: natural_ar_vw_state_phys3d_single latest (자연 영상 latent 손실 + 3D 합성 물체 1 개 상태 감독, head = SingleStateHead:
      블록 256 칸 attention 풀링 1 개 → 물체 49 (정체 48 + 없음) · 위치 2 · 속도 3).
입력: [E] 공유 캐시 h.npy (블록 토큰). 이 모델의 TF 한 걸음 예측 p[b] (블록 0..b → b+1) 를 새로 계산.
영상마다 블록 b = B0..30 평균:
  tok   = |p[b] − h[b+1]| (이 모델의 토큰 surprise, 기준)
  inno  = JS( head(p[b]).obj , head(h[b+1]).obj ) + |pos 차| (상태 혁신: 예측 상태 vs 관측으로 읽은 상태)
  first = JS( head(h[b+1]).obj , head(h[0]).obj )        (상태 공간 "첫 블록과 비교" — X3 의 상태판, 예측 없음)
  seen  = min_{b' ≤ b−1} JS( head(h[b+1]).obj , head(h[b']).obj ) 의 b 평균 ("본 적 있나" 상태판)
표준 보고 (crit_ip2_pmi.decomp 재사용).
"""
import os, sys, json, argparse, random, numpy as np, torch, torch.nn.functional as F

REPO = "/workspace/world_model/vjepa2"
sys.path.insert(0, os.path.join(REPO, "auto_research/scripts")); sys.path.insert(0, REPO)
_argv = list(sys.argv); sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT  # noqa: E402
sys.argv = _argv
from app.vjepa_frozen.state_head import SingleStateHead  # noqa: E402
from crit_ip2_pmi import decomp  # noqa: E402

C = "/workspace/auto_research_cache/intphys2/Main/"

def js(p, q):
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (torch.log(a + 1e-8) - torch.log(b + 1e-8))).sum(-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default="natural_ar_vw_state_phys3d_single"); ap.add_argument("--B0", type=int, default=4)
    ap.add_argument("--out", default=None); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(8)
    dev = torch.device("cuda")
    ck = torch.load(os.path.join(REPO, "z_training/runs", a.run, "latest.pt"), map_location="cpu", weights_only=False)
    W = FT.NoiseWrap(FT.load_pred(a.run, "latest.pt"), 32, 0.0).to(dev).eval()
    hd = SingleStateHead(); hd.load_state_dict(ck["state_head"]); hd = hd.to(dev).eval()
    meta = json.load(open(C + "meta.json")); H = np.load(C + "h.npy", mmap_mode="r")
    if a.limit:
        meta = meta[: a.limit]
    rows = []
    S, D = 256, 1280
    for m in meta:
        k = m["row"]
        h = torch.from_numpy(np.asarray(H[k], dtype=np.float32)).to(dev)               # (NB, S, D)
        NB = h.shape[0]
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = W.tf(h.reshape(1, NB * S, D)).float().view(NB, S, D)[:-1]       # (NB-1): 위치 b → 블록 b+1
            op = hd(p.reshape(1, -1, D), S); oh = hd(h.reshape(1, -1, D), S)
            Pp = torch.softmax(op["obj"][0, :, 0].float(), -1); Ph = torch.softmax(oh["obj"][0, :, 0].float(), -1)   # (NB-1,49) · (NB,49)
            pp, ph = op["pos"][0, :, 0].float(), oh["pos"][0, :, 0].float()
            bs = torch.arange(a.B0, NB - 1, device=dev)
            tok = (p[bs] - h[bs + 1]).abs().mean((1, 2))
            inno = js(Pp[bs], Ph[bs + 1]) + (pp[bs] - ph[bs + 1]).abs().sum(-1)
            inno_obj = js(Pp[bs], Ph[bs + 1])
            first = js(Ph[bs + 1], Ph[0:1].expand(len(bs), -1))
            Jall = js(Ph[:, None, :], Ph[None, :, :])                                    # (NB, NB)
            seen = torch.stack([Jall[b + 1, : b].min() for b in bs.tolist()])           # 직전 블록 b 까지 (b+1 은 제외)
            none_p = (Pp[bs, -1] - Ph[bs + 1, -1]).abs()
            # 주소 (head 가 관측 블록에서 보는 칸) 에서의 latent 기억 비교 — 정답 영역 (오라클) 대신 head attention
            A = oh["attn"][0, :, 0].float()                                              # (NB, S), 합 1
            dmat2 = torch.stack([(h[bb][None] - h[:bb - 1]).abs().mean(-1).min(0).values if bb > 2 else torch.zeros(S, device=dev)
                                 for bb in range(NB)])                                    # 칸별 "본 적 있나" (직전 블록 제외)
            d_first = (h - h[0:1]).abs().mean(-1)                                         # (NB, S)
            d_prev = torch.cat([torch.zeros(1, S, device=dev), (h[1:] - h[:-1]).abs().mean(-1)])
            tb = bs + 1
            seen_addr = (A[tb] * dmat2[tb]).sum(-1); seen_all = dmat2[tb].mean(-1)
            first_addr = (A[tb] * d_first[tb]).sum(-1); first_all = d_first[tb].mean(-1)
            prev_addr = (A[tb] * d_prev[tb]).sum(-1)
            tok_addr = (A[tb] * (p[bs] - h[bs + 1]).abs().mean(-1)).sum(-1)
            amax = A[tb].max(-1).values; ent = -(A[tb] * torch.log(A[tb] + 1e-9)).sum(-1)
        rows.append(dict(scene=m["scene_index"], pair=str(m["pair_id"]), imp=bool(m["is_impossible"]), cond=m["condition"], cam=m["camera"],
                         tok=float(tok.mean()), inno=float(inno.mean()), inno_obj=float(inno_obj.mean()), first=float(first.mean()),
                         seen=float(seen.mean()), none=float(none_p.mean()),
                         seen_addr=float(seen_addr.mean()), seen_all=float(seen_all.mean()), first_addr=float(first_addr.mean()),
                         first_all=float(first_all.mean()), prev_addr=float(prev_addr.mean()), tok_addr=float(tok_addr.mean()),
                         attn_max=float(amax.mean()), attn_ent=float(ent.mean())))
        if k % 100 == 0:
            print(k, rows[-1], flush=True)
    rng = random.Random(0)
    res = {}
    keys = ["tok", "inno", "inno_obj", "none", "first", "seen", "seen_addr", "seen_all", "first_addr", "first_all", "prev_addr", "tok_addr"]
    for cond in ["all", "permanence", "immutability", "continuity", "solidity"]:
        for cam in ["all", "Fixed", "Moving"]:
            xs = [r for r in rows if (cond == "all" or r["cond"] == cond) and (cam == "all" or r["cam"] == cam)]
            if len({r["scene"] for r in xs}) < 5:
                continue
            res[f"{cond}|{cam}"] = {key: decomp(xs, key, False, rng) for key in keys}
            if cam == "all" or cond == "all":
                for key, st in res[f"{cond}|{cam}"].items():
                    print(f"{cond:13s} {cam:6s} {key:8s} n {st['n']:3d} acc {st['acc']:5.1f} [{st['acc_ci'][0]:5.1f},{st['acc_ci'][1]:5.1f}] "
                          f"짝1/2 {st['pair1']:5.1f}/{st['pair2']:5.1f} o-share {st['outcome_share']:.2f} mean_i {st['mean_i']:+.5f} [{st['mean_i_ci'][0]:+.5f},{st['mean_i_ci'][1]:+.5f}]", flush=True)
    if a.out:
        json.dump(dict(summary=res, rows=rows, run=a.run, B0=a.B0), open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
