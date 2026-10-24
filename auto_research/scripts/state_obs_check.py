#!/usr/bin/env python3
"""[T] 상태 감독 모델의 상태 head 가 관측 (목표 encoder 블록 토큰) 에서도 상태를 읽나 — 합성 검증 세트 (학습 없음) (auto_research, 2026-10-22 KST).

  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python -W ignore auto_research/scripts/state_obs_check.py --run natural_ar_vw_state_phys3d_single \
      --val /workspace/dataset/training_data/phys3d_single_v1/val/boundary.tsv [--n 300]

사용자 체크포인트 · head 코드는 읽기만 (app/vjepa_frozen/state_head.py 의 SingleStateHead · pack_labels, z_training/data/phys2d_sup_labels).
검증 clip 마다 32 장 (간격 1, 학습 boundary 와 같은 읽기: 전체 화면 → 256 bilinear → uint8 반올림 → ImageNet 정규화) = 블록 16 개.
  관측: h[b] = LN(목표 encoder(블록 b 두 장)) — 블록 b 라벨 (블록 마지막 장 상태) 과 비교.
  예측: p[b] = 그 모델의 TF 한 걸음 (블록 0..b → b+1) — 블록 b+1 라벨과 비교 (head 가 학습된 입력 분포).
같은 블록들 (1..15) 에서: 위치 오차 px (256 화면; 보임 = 상태 있음 & 안 가려짐 · 가려짐), 정체 정확도 (본 적 있는 블록, 49 지선다 최고 = 그 정체),
"없음" 정확도 (아직 안 본 블록). 기준: 화면 가운데로 찍은 위치 오차.
사전 등록 (진행판 [T] KST 02:51): 관측 보임 위치 오차 ≤ 예측의 1.5 배 · 관측 정체 정확도 ≥ 예측의 0.8 배 → "관측에서도 읽는다"."""
import argparse, csv, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--ck", default="latest.pt")
ap.add_argument("--val", required=True)
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--out", default=None)
a = ap.parse_args()
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.dirname(__file__))
_argv = sys.argv; sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
import ar_crps_ft as FT                                                     # noqa: E402
sys.argv = _argv
import decord                                                               # noqa: E402
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint   # noqa: E402
from app.vjepa_frozen.state_head import SingleStateHead, pack_labels        # noqa: E402
from z_training.data.phys2d_sup_labels import load_clip, window_labels      # noqa: E402
torch.set_num_threads(8)
dev = torch.device("cuda")
S, NF = 256, 32
NB = NF // 2
CKE = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
tenc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(tenc, _load_checkpoint(CKE)["target_encoder"], tag="t"); tenc = tenc.to(dev).eval()
ck = torch.load(os.path.join(REPO, "z_training/runs", a.run, a.ck), map_location="cpu", weights_only=False)
pred = FT.load_pred(a.run, a.ck).to(dev).eval()
head = SingleStateHead(); head.load_state_dict(ck["state_head"]); head = head.to(dev).eval()

rows, seen_p = [], set()
for r in csv.DictReader(open(a.val), delimiter="\t"):
    if r["path"] in seen_p:
        continue
    vf = int(r["video_frames"]); b = int(r["boundary_frame"])
    st = max(0, min(b - NF // 2, vf - NF))
    if vf < NF:
        continue
    seen_p.add(r["path"]); rows.append((r["path"], st, r.get("scenario", ""), r.get("category", "")))
rows = rows[:a.n]
print(f"[{a.run}] 검증 clip {len(rows)}", flush=True)


@torch.no_grad()
def encode(path, st):
    vr = decord.VideoReader(path, num_threads=1)
    fr = torch.from_numpy(vr.get_batch(list(range(st, st + NF))).asnumpy()).permute(0, 3, 1, 2).float()      # (NF,3,H,W)
    x = F.interpolate(fr, size=(256, 256), mode="bilinear", align_corners=False).round().clamp(0, 255) / 255.0
    x = ((x - MEAN) / STD).view(NB, 2, 3, 256, 256).permute(0, 2, 1, 3, 4).to(dev)                          # (NB,3,2,256,256)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = torch.cat([tenc(x[i:i + 8]) for i in range(0, NB, 8)], 0).float()
    return F.layer_norm(h, (1280,))                                                                          # (NB,S,D)


acc = {k: [0.0, 0] for k in ("obs_vis", "obs_hid", "pred_vis", "pred_hid", "ctr_vis", "ctr_hid", "copy_vis", "copy_hid", "obs_id", "pred_id", "obs_none", "pred_none")}
for src_ in ("obs", "pred"):
    for rd_ in ("sam", "amax"):
        for cat_ in ("vis", "hid"):
            acc[f"addr_{rd_}_{src_}_{cat_}"] = [0.0, 0]
CXY = np.stack([(np.arange(S) % 16 + 0.5) / 16, (np.arange(S) // 16 + 0.5) / 16], -1)                       # 칸 중심 (x, y) 화면 비율
for src_ in ("obs", "pred"):
    for cat_ in ("vis", "hid", "off", "unseen"):
        acc[f"pnone_{src_}_{cat_}"] = [0.0, 0]
def add(k, v, m):
    acc[k][0] += float((v * m).sum()); acc[k][1] += int(m.sum())

t0 = time.time()
for i, (path, st, scen, cat) in enumerate(rows):
    try:
        lab = pack_labels(window_labels(load_clip(path), start=st, n_frames=NF, stride=1, tubelet=2, reduce="last"))
    except Exception as e:                                                   # 라벨 없음 · 깨짐
        print("skip", path, e); continue
    h = encode(path, st)
    with torch.no_grad():
        idx = torch.arange(NB * S, device=dev).unsqueeze(0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = pred.forward_seq(h.reshape(1, NB * S, -1), idx).float()                                       # (1, NB*S, D): 위치 b → 블록 b+1
        oh = head(h.reshape(1, NB * S, -1), S); op = head(p, S)
    k = int((lab["seen"] | lab["m_state"]).any(0).argmax())
    seen, ms, hid = lab["seen"][:, k], lab["m_state"][:, k], lab["hidden"][:, k]
    pos_t = lab["pos"][:, k]                                                                                  # (NB,2) 0–1
    B1 = slice(1, NB)                                                                                         # 블록 1..15
    po = oh["pos"][0, :, 0].cpu().numpy(); pp = op["pos"][0, :, 0].cpu().numpy()                             # (NB,2) · (NB,2) (pp[b] = 블록 b+1)
    oo = oh["obj"][0, :, 0].argmax(-1).cpu().numpy(); opp = op["obj"][0, :, 0].argmax(-1).cpu().numpy()
    vis = (ms & ~hid)[B1]; hd = hid[B1]; sn = seen[B1]; ns = ~seen[B1]
    eo = np.linalg.norm(po[B1] - pos_t[B1], axis=-1) * 256; ep = np.linalg.norm(pp[:NB - 1] - pos_t[B1], axis=-1) * 256
    ec = np.linalg.norm(0.5 - pos_t[B1], axis=-1) * 256
    ecp = np.linalg.norm(pos_t[:NB - 1] - pos_t[B1], axis=-1) * 256                                          # 복사: 라벨[b−1] 을 b 에
    okp = ms[:NB - 1]
    add("copy_vis", ecp, vis & okp); add("copy_hid", ecp, hd & okp)
    for src_, o_, sl in (("obs", oh, B1), ("pred", op, slice(0, NB - 1))):                                 # 주소 읽기: attention 가중 칸 중심 · 최대 칸
        at = o_["attn"][0, :, 0].float().cpu().numpy()[sl]                                                   # (15, S)
        for rd_, xy_ in (("sam", at @ CXY / at.sum(-1, keepdims=True)), ("amax", CXY[at.argmax(-1)])):
            e_ = np.linalg.norm(xy_ - pos_t[B1], axis=-1) * 256
            add(f"addr_{rd_}_{src_}_vis", e_, vis); add(f"addr_{rd_}_{src_}_hid", e_, hd)
    pno = oh["obj"][0, :, 0].float().softmax(-1)[:, 48].cpu().numpy(); pnp = op["obj"][0, :, 0].float().softmax(-1)[:, 48].cpu().numpy()
    cats = {"vis": vis, "hid": hd, "off": (seen & ~ms)[B1], "unseen": ns}
    for cat_, m_ in cats.items():
        add(f"pnone_obs_{cat_}", pno[B1], m_); add(f"pnone_pred_{cat_}", pnp[:NB - 1], m_)
    add("obs_vis", eo, vis); add("obs_hid", eo, hd); add("pred_vis", ep, vis); add("pred_hid", ep, hd); add("ctr_vis", ec, vis); add("ctr_hid", ec, hd)
    add("obs_id", (oo[B1] == k).astype(float), sn); add("pred_id", (opp[:NB - 1] == k).astype(float), sn)
    add("obs_none", (oo[B1] == 48).astype(float), ns); add("pred_none", (opp[:NB - 1] == 48).astype(float), ns)
    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
res = {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in acc.items()}
res.update({f"n_{k}": v[1] for k, v in acc.items()})
res["pass_pos"] = bool(res["obs_vis"] <= 1.5 * res["pred_vis"]); res["pass_id"] = bool(res["obs_id"] >= 0.8 * res["pred_id"])
print(f"[{a.run}] 위치 오차 px (보임 / 가려짐): 관측 {res['obs_vis']:.1f} / {res['obs_hid']:.1f} · 예측 {res['pred_vis']:.1f} / {res['pred_hid']:.1f} · 가운데 {res['ctr_vis']:.1f} / {res['ctr_hid']:.1f} "
      f"(블록 수 {res['n_obs_vis']} / {res['n_obs_hid']}) | 정체 정확도 관측 {res['obs_id']:.3f} · 예측 {res['pred_id']:.3f} | 없음 정확도 관측 {res['obs_none']:.3f} · 예측 {res['pred_none']:.3f} "
      f"→ 위치 {'통과' if res['pass_pos'] else '실패'} · 정체 {'통과' if res['pass_id'] else '실패'}", flush=True)
print(f"[{a.run}] 복사 기준 (라벨[b−1] → b) 위치 오차 보임 / 가려짐 {res['copy_vis']:.1f} / {res['copy_hid']:.1f} | \"없음\" 확률 (보임 · 가려짐 · 화면 밖 · 아직 안 봄): "
      f"관측 {res['pnone_obs_vis']:.3f} · {res['pnone_obs_hid']:.3f} · {res['pnone_obs_off']:.3f} · {res['pnone_obs_unseen']:.3f} / "
      f"예측 {res['pnone_pred_vis']:.3f} · {res['pnone_pred_hid']:.3f} · {res['pnone_pred_off']:.3f} · {res['pnone_pred_unseen']:.3f} "
      f"(블록 수 {res['n_pnone_obs_vis']} · {res['n_pnone_obs_hid']} · {res['n_pnone_obs_off']} · {res['n_pnone_obs_unseen']})", flush=True)
print(f"[{a.run}] 주소 읽기 (attention) 위치 px 보임 / 가려짐: " + " · ".join(f"{rd_} {src_} {res[f'addr_{rd_}_{src_}_vis']:.1f} / {res[f'addr_{rd_}_{src_}_hid']:.1f}" for rd_ in ("sam", "amax") for src_ in ("obs", "pred"))
      + f" → 관측 soft-argmax 기준 (≤ 사용자 head 예측 위치 × 1.5 = {1.5 * res['pred_vis']:.1f}) {'통과' if res['addr_sam_obs_vis'] <= 1.5 * res['pred_vis'] else '실패'} · 최대 칸 {'통과' if res['addr_amax_obs_vis'] <= 1.5 * res['pred_vis'] else '실패'}", flush=True)
out = a.out or os.path.join(REPO, f"auto_research/results/state_obs_check_{a.run}.json")
json.dump(res, open(out, "w"), indent=1)
