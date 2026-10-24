#!/usr/bin/env python3
"""[T] VOBench 끝 가림: 상태 감독 모델의 상태 head 로 "상태 공간 surprise" (학습 없음) (auto_research, 2026-10-22 KST).

  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python -W ignore auto_research/scripts/state_vob.py --run natural_ar_vw_state_phys3d_single

--obs_head 를 주면 관측 상태는 관측용 판독 head 로 읽는다 (사용자 head 는 관측 토큰에서 상태를 못 읽음, 진행판 KST 03:10).
구성은 [P] oracle_vob.py 와 같다 (32 장 = 블록 16, 문맥 8 블록 → 그 모델로 굴린 예측 8 블록, 쌍 (pos_a, imp_ab) · (pos_b, imp_ba), 정답 = surprise(불가능) > surprise(가능)).
사용자 체크포인트 · head 는 읽기만 (SingleStateHead: 블록 256 칸 풀링 1 개 → 정체 49 (48 + 없음) · 위치 (0–1) · 속도).
미래 블록 j 마다: P = softmax(head(예측 j).obj), Q = softmax(head(관측 j).obj), x̂ · x = 위치.
  state = JS(P, Q) + |x̂ − x|₁ · id = JS(P, Q) · pos = |x̂ − x|₁ · addr = |주소(예측) − 주소(관측)|₁ (주소 = head attention 가중 칸 중심, 8 블록 평균). 비교: 그 모델의 토큰 surprise (화면 · 물체 자리, oracle_vob 와 같은 정의).
head 정확도 (라벨 직접): 관측 위치 오차 (문맥 · 미래의 보이는 블록, px 256 화면) · 예측 위치 오차 (미래 블록, 보임 · 가려짐) vs 기준
  (마지막 보인 문맥 위치에 멈춤 · 문맥 마지막 두 보임 위치로 등속 외삽)."""
import argparse, csv, json, os, sys, collections, time
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--ck", default="latest.pt")
ap.add_argument("--vobench", default="/workspace/dataset/benchmarks/VOBench_v1")
ap.add_argument("--out", default=None)
ap.add_argument("--obs_head", default=None, help="관측 토큰용 판독 head (state_obs_head.py 산출). 주면 관측 상태는 이 head 로, 예측 상태는 사용자 head 로 읽는다")
a = ap.parse_args()
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.dirname(__file__))
_argv = sys.argv; sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT                                                     # noqa: E402
sys.argv = _argv
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint   # noqa: E402
from app.vjepa_frozen.state_head import SingleStateHead                     # noqa: E402
torch.set_num_threads(8)
dev = torch.device("cuda")
CKE = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
S, C, K, G = 256, 8, 8, 16
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
IDX = list(range(0, 96, 3))[:32]
tenc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(tenc, _load_checkpoint(CKE)["target_encoder"], tag="t"); tenc = tenc.to(dev).eval()
ck = torch.load(os.path.join(REPO, "z_training/runs", a.run, a.ck), map_location="cpu", weights_only=False)
pred = FT.load_pred(a.run, a.ck).to(dev).eval()
head = SingleStateHead(); head.load_state_dict(ck["state_head"]); head = head.to(dev).eval()
ohead = head
if a.obs_head:
    ohead = SingleStateHead(); ohead.load_state_dict(torch.load(a.obs_head, map_location="cpu", weights_only=False)["state_head"]); ohead = ohead.to(dev).eval()


def frames(d, idx):
    fr = [np.asarray(Image.open(f"{d}/{i:06d}.png").convert("RGB")) for i in idx]
    x = torch.from_numpy(np.stack(fr)).permute(3, 0, 1, 2).float() / 255.0
    W0 = x.shape[-1]
    x = F.interpolate(x.transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False).transpose(0, 1)
    return (x - MEAN) / STD, W0


@torch.no_grad()
def blocks(x):
    nb = x.shape[1] // 2
    xb = x.reshape(3, nb, 2, 256, 256).permute(1, 0, 2, 3, 4).to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = torch.cat([tenc(xb[i:i + 8]) for i in range(0, nb, 8)], 0).float()
    return F.layer_norm(h, (1280,)).view(nb, S, -1)


CXY = torch.stack([(torch.arange(S) % 16 + 0.5) / 16, (torch.arange(S) // 16 + 0.5) / 16], -1).to(dev)


def js(p, q):
    m = 0.5 * (p + q)
    kl = lambda x, y: (x * (torch.log(x + 1e-8) - torch.log(y + 1e-8))).sum(-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


@torch.no_grad()
def hstate(t, hd=None):                                                     # (nb,S,D) → 확률 (nb,49) · 위치 (nb,2)
    o = (hd or head)(t.reshape(1, -1, t.size(-1)), S)
    at = o["attn"][0, :, 0].float()
    hstate.addr = (at @ CXY) / at.sum(-1, keepdim=True)                       # 주소 읽기: attention 가중 칸 중심 (x, y) 화면 비율
    return torch.softmax(o["obj"][0, :, 0].float(), -1), o["pos"][0, :, 0].float()


rows = [r for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv"))) if r["occlusion"] == "late"]
blk = collections.defaultdict(dict)
for r in rows:
    blk[r["block_id"]][r["variant"]] = r
res, accp = [], collections.defaultdict(lambda: [0.0, 0])
t0 = time.time()
for bi, (b, arms) in enumerate(sorted(blk.items())):
    d0 = os.path.join(a.vobench, arms["pos_a"]["file_name"])
    Hf, W0 = {}, None
    for v in ("pos_a", "pos_b"):
        x, W0 = frames(os.path.join(a.vobench, arms[v]["file_name"]), IDX); Hf[v] = blocks(x)
    for v in ("imp_ab", "imp_ba"):
        x, _ = frames(os.path.join(a.vobench, arms[v]["file_name"]), IDX[2 * C:]); Hf[v] = blocks(x)
    LB = json.load(open(os.path.join(os.path.dirname(d0), "label.json")))["labels"]
    mv = "static" if arms["pos_a"]["motion"] == "flat_static" else "moving"
    for pos, imp, key in (("pos_a", "imp_ab", "A"), ("pos_b", "imp_ba", "B")):
        h = Hf[pos]
        lab = {q["f"]: q for q in LB.get(key, [])} or {q["f"]: q for q in LB.get("A", [])}
        q_at = lambda bb: lab.get(IDX[2 * bb + 1])
        xy = lambda q: np.array([q["u"], q["v"]]) / W0                       # 화면 비율 (head 위치와 같은 단위)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            p_ar = pred.rollout_kv(h[:C].reshape(1, C * S, -1), torch.arange(C * S, device=dev).unsqueeze(0), K).float()[0].view(K, S, -1)
        Pp, xp = hstate(p_ar); ap_ = hstate.addr
        Po, xo = hstate(h, ohead); ao_ = hstate.addr
        Pi, xi = hstate(Hf[imp], ohead); ai_ = hstate.addr
        # head 정확도 (라벨)
        xo_np, xp_np = xo.cpu().numpy(), xp.cpu().numpy(); ao_np, ap_np = ao_.cpu().numpy(), ap_.cpu().numpy()
        vis_ctx = [bb for bb in range(C) if q_at(bb) is not None and q_at(bb)["hidden"] < 0.1]
        for bb in range(16):
            q = q_at(bb)
            if q is not None and q["hidden"] < 0.1:
                accp[f"{mv}/obs_vis"][0] += float(np.linalg.norm(xo_np[bb] - xy(q)) * 256); accp[f"{mv}/obs_vis"][1] += 1
                accp[f"{mv}/addr_obs_vis"][0] += float(np.linalg.norm(ao_np[bb] - xy(q)) * 256); accp[f"{mv}/addr_obs_vis"][1] += 1
                accp[f"{mv}/ctr_vis"][0] += float(np.linalg.norm(0.5 - xy(q)) * 256); accp[f"{mv}/ctr_vis"][1] += 1
        if vis_ctx:
            qs = q_at(vis_ctx[-1]); prev = [bb for bb in vis_ctx if bb < vis_ctx[-1]]
            vel = (xy(qs) - xy(q_at(prev[-1]))) / (vis_ctx[-1] - prev[-1]) if prev else np.zeros(2)
            for j in range(K):
                q = q_at(C + j)
                if q is None:
                    continue
                tag = "hid" if q["hidden"] >= 0.5 else "vis"
                for nm, guess in (("pred", xp_np[j]), ("addr_pred", ap_np[j]), ("stay", xy(qs)), ("cv", xy(qs) + vel * (C + j - vis_ctx[-1]))):
                    accp[f"{mv}/{nm}_{tag}"][0] += float(np.linalg.norm(guess - xy(q)) * 256); accp[f"{mv}/{nm}_{tag}"][1] += 1
        # 물체 자리 칸 (oracle_vob 와 같은 정의: 미래 블록마다 A · B 중심 3×3)
        otok = []
        for j in range(K):
            s_ = set()
            for kk in ("A", "B"):
                labk = {q["f"]: q for q in LB.get(kk, [])}
                q = labk.get(IDX[2 * (C + j)]), labk.get(IDX[2 * (C + j) + 1])
                if all(t is not None for t in q):
                    r0 = int(np.clip(np.mean([t["v"] for t in q]) / W0 * G, 0, G - 1)); c0 = int(np.clip(np.mean([t["u"] for t in q]) / W0 * G, 0, G - 1))
                    s_ |= {rr * G + cc for rr in range(max(0, r0 - 1), min(G, r0 + 2)) for cc in range(max(0, c0 - 1), min(G, c0 + 2))}
            otok.append(sorted(s_))
        oidx = torch.tensor([j * S + t for j, tt in enumerate(otok) for t in tt], device=dev, dtype=torch.long)
        fut = {pos: (h[C:], Po[C:], xo[C:], ao_[C:]), imp: (Hf[imp], Pi, xi, ai_)}
        sc = {}
        for v, (hv, Pv, xv, av) in fut.items():
            e = (p_ar.reshape(K * S, -1) - hv.reshape(K * S, -1)).abs().mean(-1)
            jsv = js(Pp, Pv); pv = (xp - xv).abs().sum(-1)
            sc[v] = dict(tok_full=float(e.mean()), tok_obj=float(e[oidx].mean()) if len(oidx) else float("nan"),
                         state=float((jsv + pv).mean()), id=float(jsv.mean()), pos=float(pv.mean()), addr=float((ap_ - av).abs().sum(-1).mean()))
        rec = {"block": b, "dir": key, "mv": mv, "k": int(arms[pos]["k"]), "viol": arms[pos]["violation"], "motion": arms[pos]["motion"]}
        for m_ in ("tok_full", "tok_obj", "state", "id", "pos", "addr"):
            rec[m_] = sc[imp][m_] > sc[pos][m_]
        res.append(rec)
    if bi % 40 == 0:
        print(f"{bi}/{len(blk)} ({time.time() - t0:.0f}s)", flush=True)
M = ("tok_obj", "tok_full", "addr", "state", "id", "pos")
out = {"pairs": {}, "head_acc_px": {k: v[0] / max(v[1], 1) for k, v in accp.items()}, "head_acc_n": {k: v[1] for k, v in accp.items()}}
print(f"\n== [{a.run}] VOBench 끝 가림 쌍 정확도 (%) — 토큰 물체 자리 · 토큰 화면 · 상태 · 정체만 · 위치만")
for mv in ("moving", "static"):
    groups = [("all", lambda r: True)] + [(vi, (lambda vi: lambda r: r["viol"] == vi)(vi)) for vi in ("permanence", "shape", "color")]
    for gname, f in groups:
        sel = [r for r in res if r["mv"] == mv and f(r)]
        if not sel:
            continue
        e = {m_: 100 * np.mean([r[m_] for r in sel]) for m_ in M}
        out["pairs"][f"{mv}/{gname}"] = dict(n=len(sel), **e)
        print(f"  {mv:7s} {gname:10s} n{len(sel):3d}  " + " · ".join(f"{m_} {e[m_]:5.1f}" for m_ in M))
print("  head 위치 오차 px (256 화면): " + " · ".join(f"{k} {v:.1f} (n {out['head_acc_n'][k]})" for k, v in sorted(out["head_acc_px"].items())))
json.dump(out, open(a.out or os.path.join(REPO, f"auto_research/results/state_vob_{a.run}{'_obshead' if a.obs_head else ''}.json"), "w"), indent=1)
