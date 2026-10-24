#!/usr/bin/env python3
"""[T] 관측용 상태 판독 head (사전 등록 대안, 진행판 [T] KST 02:51 · 03:10) — 합성 학습 분할의 관측 블록 토큰 + 라벨에 맞추고 보류 검증으로 확인 (auto_research, 2026-10-22 KST).
predictor · 사용자 head 는 건드리지 않는다. 구조는 사용자 SingleStateHead 그대로 (새 가중치), 손실도 사용자 single_losses 그대로 (import 만).

  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python -W ignore auto_research/scripts/state_obs_head.py --dom phys3d_single_v1 [--run natural_ar_vw_state_phys3d_single]

1) 캐시: <dom>/train/boundary.tsv 에서 clip n_train 개 · val/boundary.tsv 에서 n_val 개, clip 마다 32 장 (간격 1) = 블록 16 의 관측 토큰 (state_obs_check.py 와 같은 읽기 · 인코딩)
   → /workspace/auto_research_cache/state_obs/<dom>_{train,val}_h.npy (fp16) + _lab.npz (pack_labels 48 칸).
2) 학습: 블록 단위 (T = 16 그대로) 로 single_losses, AdamW lr 1e-3, steps.
3) 검증 (보류 clip): 위치 오차 px (보임 · 가려짐) · 정체 정확도 · "없음" 확률 — 관측 토큰에. --run 이 있으면 그 모델의 한 걸음 예측에도 (한 head 로 둘 다 읽나).
저장: /workspace/auto_research_cache/state_obs/obs_head_<dom>.pt, 결과 results/state_obs_head_<dom>.json."""
import argparse, csv, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--dom", required=True)
ap.add_argument("--root", default="/workspace/dataset/training_data")
ap.add_argument("--n_train", type=int, default=600)
ap.add_argument("--n_val", type=int, default=300)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--run", default=None)
ap.add_argument("--cache", default="/workspace/auto_research_cache/state_obs")
a = ap.parse_args()
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.dirname(__file__))
import decord                                                               # noqa: E402
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint   # noqa: E402
from app.vjepa_frozen.state_head import SingleStateHead, pack_labels, single_losses     # noqa: E402
from z_training.data.phys2d_sup_labels import load_clip, window_labels      # noqa: E402
torch.set_num_threads(8); torch.manual_seed(0); np.random.seed(0)
dev = torch.device("cuda")
S, NF = 256, 32
NB = NF // 2
os.makedirs(a.cache, exist_ok=True)
CKE = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
LK = ("seen", "m_state", "m_dir", "hidden", "pos", "vel")


def clips(split, n):
    out, seen_p = [], set()
    for r in csv.DictReader(open(os.path.join(a.root, a.dom, split, "boundary.tsv")), delimiter="\t"):
        vf = int(r["video_frames"]); b = int(r["boundary_frame"])
        if r["path"] in seen_p or vf < NF:
            continue
        seen_p.add(r["path"]); out.append((r["path"], max(0, min(b - NF // 2, vf - NF))))
    rng = np.random.default_rng(0); rng.shuffle(out)
    return out[:n]


def build(split, n):
    hp, lp = os.path.join(a.cache, f"{a.dom}_{split}_h.npy"), os.path.join(a.cache, f"{a.dom}_{split}_lab.npz")
    if os.path.exists(lp):
        return np.load(hp, mmap_mode="r"), dict(np.load(lp))
    global tenc
    if "tenc" not in globals():
        tenc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
        _load_state_dict(tenc, _load_checkpoint(CKE)["target_encoder"], tag="t"); tenc = tenc.to(dev).eval()
    cl = clips(split, n)
    H = np.lib.format.open_memmap(hp + ".tmp.npy", mode="w+", dtype=np.float16, shape=(len(cl), NB, S, 1280))
    L = {k: [] for k in LK}; ok = []
    t0 = time.time()
    for i, (path, st) in enumerate(cl):
        try:
            lab = pack_labels(window_labels(load_clip(path), start=st, n_frames=NF, stride=1, tubelet=2, reduce="last"))
            vr = decord.VideoReader(path, num_threads=1)
            fr = torch.from_numpy(vr.get_batch(list(range(st, st + NF))).asnumpy()).permute(0, 3, 1, 2).float()
        except Exception as e:
            print("skip", path, e); continue
        x = F.interpolate(fr, size=(256, 256), mode="bilinear", align_corners=False).round().clamp(0, 255) / 255.0
        x = ((x - MEAN) / STD).view(NB, 2, 3, 256, 256).permute(0, 2, 1, 3, 4).to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            h = torch.cat([tenc(x[j:j + 8]) for j in range(0, NB, 8)], 0).float()
        H[len(ok)] = F.layer_norm(h, (1280,)).half().cpu().numpy()
        for k in LK:
            L[k].append(lab[k])
        ok.append(i)
        if (i + 1) % 100 == 0:
            print(f"  {split} {i + 1}/{len(cl)} ({time.time() - t0:.0f}s)", flush=True)
    H.flush(); del H
    Hn = np.load(hp + ".tmp.npy", mmap_mode="r")[:len(ok)]
    np.save(hp, Hn); os.remove(hp + ".tmp.npy")
    L = {k: np.stack(v) for k, v in L.items()}
    np.savez(lp, **L)
    return np.load(hp, mmap_mode="r"), L


Htr, Ltr = build("train", a.n_train)
Hva, Lva = build("val", a.n_val)
print(f"[{a.dom}] train {Htr.shape[0]} · val {Hva.shape[0]} clip", flush=True)
head = SingleStateHead().to(dev)
opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=0.01)
rng = np.random.default_rng(0); t0 = time.time()
for it in range(1, a.steps + 1):
    ii = np.sort(rng.choice(Htr.shape[0], a.bs, replace=False))
    x = torch.from_numpy(np.asarray(Htr[ii], np.float32)).to(dev).view(a.bs, NB * S, -1)
    lab = {k: torch.from_numpy(Ltr[k][ii]).to(dev) for k in LK}
    loss, log = single_losses(head(x, S), lab, {"exist": 1.0, "pos": 1.0, "vel": 1.0})
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if it % 250 == 0 or it == 1:
        print(f"  {it} 손실 {float(loss):.4f} · 위치 px 보임 {log['pos_px_vis']:.1f} · 가려짐 {log['pos_px_hid']:.1f} · 정체 {log['id_acc']:.3f} ({time.time() - t0:.0f}s)", flush=True)
torch.save({"state_head": head.state_dict(), "args": vars(a)}, os.path.join(a.cache, f"obs_head_{a.dom}.pt"))
head.eval()


@torch.no_grad()
def evaluate(get_tokens, shift):
    """get_tokens(i) → (NB or NB-1, S, D) 토큰; shift = 1 이면 위치 b 의 토큰이 블록 b+1 을 뜻함 (한 걸음 예측)."""
    acc = {k: [0.0, 0] for k in ("vis", "hid", "id", "none_vis", "none_hid", "none_unseen")}
    for i in range(Hva.shape[0]):
        t = get_tokens(i); T = t.shape[0]
        o = head(t.reshape(1, T * S, -1), S)
        k = int((Lva["seen"][i] | Lva["m_state"][i]).any(0).argmax())
        bl = np.arange(1, NB)                                                # 블록 1..15
        ti = bl - shift                                                      # 토큰 위치
        pos = o["pos"][0, ti, 0].cpu().numpy(); obj = o["obj"][0, ti, 0].float(); pn = obj.softmax(-1)[:, 48].cpu().numpy(); am = obj.argmax(-1).cpu().numpy()
        seen, ms, hid = Lva["seen"][i][bl, k], Lva["m_state"][i][bl, k], Lva["hidden"][i][bl, k]
        e = np.linalg.norm(pos - Lva["pos"][i][bl, k], axis=-1) * 256
        for key, m, v in (("vis", ms & ~hid, e), ("hid", hid, e), ("id", seen, (am == k).astype(float)), ("none_vis", ms & ~hid, pn), ("none_hid", hid, pn), ("none_unseen", ~seen, pn)):
            acc[key][0] += float((v * m).sum()); acc[key][1] += int(m.sum())
    return {k: v[0] / max(v[1], 1) for k, v in acc.items()}


res = {"obs": evaluate(lambda i: torch.from_numpy(np.asarray(Hva[i], np.float32)).to(dev), 0)}
if a.run:
    _argv = sys.argv; sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
    import ar_crps_ft as FT                                                 # noqa: E402
    sys.argv = _argv
    pred = FT.load_pred(a.run, "latest.pt").to(dev).eval()

    def ptok(i):
        h = torch.from_numpy(np.asarray(Hva[i], np.float32)).to(dev).view(1, NB * S, -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return pred.forward_seq(h, torch.arange(NB * S, device=dev).unsqueeze(0)).float().view(NB, S, -1)[:NB - 1]
    res["pred_" + a.run] = evaluate(ptok, 1)
for k, v in res.items():
    print(f"[{a.dom}] 관측 head 를 {k} 에: 위치 px 보임 {v['vis']:.1f} · 가려짐 {v['hid']:.1f} · 정체 {v['id']:.3f} · 없음 확률 보임 {v['none_vis']:.3f} · 가려짐 {v['none_hid']:.3f} · 아직 안 봄 {v['none_unseen']:.3f}", flush=True)
json.dump(res, open(os.path.join(REPO, f"auto_research/results/state_obs_head_{a.dom}.json"), "w"), indent=1)
