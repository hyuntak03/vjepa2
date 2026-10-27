#!/usr/bin/env python3
"""상태 판독기 (SingleStateHead, 세 출력: 물체 49 지선다 · 위치 2 · 속도 3) 를 **검증 영상의 정답 토큰** 에서 출력별로 잰다 (2026-10-26, 사용자 요청).
학습 run 의 체크포인트 (eN.pt, `state_head` 키) 마다 한 줄씩 <run>/head_eval.tsv 에 쓴다. --watch 면 새 체크포인트를 기다려 이어 잰다.

  CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/head_ckpt_eval.py --run phys3d_state_head_pretrain --watch

데이터: phys3d_single_v1 **val** (학습에 안 쓴 clip) --n 개, 창 = raw 0 부터 간격 3 으로 32 장 (블록 16 개). 라벨은 학습과 같은 경로
(phys2d_sup_labels.window_labels reduce=last → state_head.pack_labels → visible_only), 손실 · 로그 정의는 state_head.single_losses 와 같다.
출력별 지표 (보이는 블록):
  물체   — 존재 정확도 (물체 있음/없음 판정) · 정체 정확도 (보이는 블록에서 49 지선다 중 정답 물체)
  위치   — 위치 오차 px (중앙값 · 평균)
  속도   — 방향 cos (거의 정지 블록 제외) 중앙값 · 속력 L1 (라벨 단위)"""
import argparse, glob, os, re, sys, time
import numpy as np, torch, torch.nn.functional as F
import decord
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint   # noqa: E402
from app.vjepa_frozen import ar as AR, state_head as SHM                                  # noqa: E402
from z_training.data.phys2d_sup_labels import load_clip, window_labels                    # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--root", default="/workspace/dataset/training_data/phys3d_single_v1/val/clips")
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--watch", action="store_true")
a = ap.parse_args()
dev = torch.device("cuda")
S, NB, ST = 256, 16, 3
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
tgt = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(tgt, _load_checkpoint(CK)["target_encoder"], tag="t"); tgt = tgt.to(dev, torch.bfloat16).eval()

# ---- 검증 정답 토큰 · 라벨은 한 번만 뽑아 둔다 (CPU 에 fp16) ------------------------------------------------
paths = sorted(p for p in glob.glob(os.path.join(a.root, "*", "*.mp4")))
rng = np.random.default_rng(0); paths = [paths[i] for i in sorted(rng.choice(len(paths), min(a.n, len(paths)), replace=False))]
H, L = [], []
t0 = time.time()
for i, p in enumerate(paths):
    clip = load_clip(p)
    if clip["present"].shape[0] < ST * (2 * NB - 1) + 1:
        continue
    lab = SHM.visible_only({k: torch.from_numpy(v)[None] for k, v in SHM.pack_labels(window_labels(clip, start=0, n_frames=2 * NB, stride=ST, tubelet=2, reduce="last")).items()})
    vr = decord.VideoReader(p, num_threads=2)
    x = torch.from_numpy(vr.get_batch([ST * k for k in range(2 * NB)]).asnumpy()).to(dev).permute(3, 0, 1, 2)[None].float().div_(255)
    if x.shape[-1] != 256 or x.shape[-2] != 256:
        x = F.interpolate(x[0].transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False).transpose(0, 1)[None]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        h = AR.encode_blocks(tgt, ((x - MEAN) / STD).to(torch.bfloat16), 2)
    H.append(h.half().cpu()); L.append(lab)
    if i % 100 == 0:
        print(f"정답 토큰 {i}/{len(paths)} ({time.time() - t0:.0f}s)", flush=True)
print(f"검증 clip {len(H)} 개 · 블록 {NB} 개씩", flush=True)
DIR_MIN = None


@torch.no_grad()
def evaluate(head):
    acc = {k: [] for k in ("ex_acc", "id_acc", "pos_px", "cos", "spd_l1")}
    for h, lab in zip(H, L):
        lab = SHM.labels_to_device(lab, dev)
        out = head(h.to(dev).float(), S)
        _, g = SHM.single_losses(out, lab, {})
        acc["ex_acc"].append(g["ex_acc"]); acc["id_acc"].append(g["id_acc"])
        any_ = (lab["seen"] | lab["m_state"]).any(1)[0]
        if not any_.any():
            continue
        k = int(any_.float().argmax())
        ms, md, hid = lab["m_state"][0, :, k], lab["m_dir"][0, :, k], lab["hidden"][0, :, k]
        vis = ms & ~hid
        pos, vel = out["pos"][0, :, 0].float(), out["vel"][0, :, 0].float()
        pt, vt = lab["pos"][0, :, k], lab["vel"][0, :, k]
        acc["pos_px"] += ((pos - pt).norm(dim=-1) * 256)[vis].tolist()
        m = md & vis
        acc["cos"] += F.cosine_similarity(vel[m, :2], vt[m, :2], dim=-1).tolist()
        acc["spd_l1"] += (vel[vis, 2] - vt[vis, 2]).abs().tolist()
    return {"ex_acc": 100 * np.nanmean(acc["ex_acc"]), "id_acc": 100 * np.nanmean(acc["id_acc"]),
            "pos_med": np.median(acc["pos_px"]), "pos_mean": np.mean(acc["pos_px"]),
            "cos_med": np.median(acc["cos"]), "spd_l1": np.mean(acc["spd_l1"]), "n_vis": len(acc["pos_px"]), "n_mov": len(acc["cos"])}


D = os.path.join(REPO, "z_training/runs", a.run)
OUT = os.path.join(D, "head_eval.tsv")
done = set()
if os.path.exists(OUT):
    done = {l.split("\t")[0] for l in open(OUT).read().splitlines()[1:]}
else:
    open(OUT, "w").write("ckpt\tepoch\t존재정확도\t정체정확도\t위치오차중앙px\t위치오차평균px\t방향cos중앙\t속력L1\tn_보임\tn_움직임\n")
while True:
    cks = sorted((f for f in os.listdir(D) if re.fullmatch(r"e\d+\.pt", f)), key=lambda f: int(f[1:-3]))
    new = [f for f in cks if f not in done]
    for f in new:
        time.sleep(5)                                                   # 저장 끝나기 기다림
        ck = torch.load(os.path.join(D, f), map_location="cpu", weights_only=False)
        head = SHM.build_head(ck["state_head_cfg"], 1280); head.load_state_dict(ck["state_head"]); head = head.to(dev).eval()
        r = evaluate(head)
        open(OUT, "a").write(f"{f}\t{ck.get('epoch')}\t{r['ex_acc']:.1f}\t{r['id_acc']:.1f}\t{r['pos_med']:.1f}\t{r['pos_mean']:.1f}\t"
                             f"{r['cos_med']:.3f}\t{r['spd_l1']:.4f}\t{r['n_vis']}\t{r['n_mov']}\n")
        print(f, r, flush=True); done.add(f)
    alive = os.system(f"tmux has-session -t train_{a.run} 2>/dev/null") == 0
    if not a.watch or (not alive and not new):
        break
    time.sleep(60)
