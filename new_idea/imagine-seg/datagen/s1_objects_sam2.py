"""S1–S3 — 대상 물체: 창 첫 프레임 SAM2 자동 분할 → 후보 → SAM2 로 창 전체 전파 → 거르기.

원 계획 §2.2. 원본 (가림 합성 전) 영상에서 뽑으므로 이 mask 를 amodal 정답으로 쓴다.
⚠️ 원본에서도 손 등에 가려지면 mask 는 보이는 부분뿐이다 — 면적 급변 거르기가 일부를 잡는다.

출력 (클립 폴더마다):
  object.json        status (ok / no_candidate / no_object), 후보별 통계와 탈락 이유
  object.npz         고른 물체의 mask (M, H, W) packbits — status ok 일 때만
  object_cands.npz   모든 후보의 전파 mask (K, M, H, W) packbits — 탈락 원인 확인용

    python datagen/s1_objects_sam2.py --clips <dir>/ssv2 --device cuda:0
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OBJECT, SAM2_CFG, SAM2_CKPT, pack, save_json, unpack  # noqa: E402

MAX_JUMP_PX = 48.0      # 한 프레임 (1/6 초) 사이 중심 이동 상한 — 넘으면 추적이 다른 물체로 튄 것으로 본다


def candidates(amg, frame0: np.ndarray, k: int):
    H, W = frame0.shape[:2]
    out = []
    for m in amg.generate(frame0):
        a = m["area"] / (H * W)
        x, y, w, h = m["bbox"]
        if not (OBJECT["area_min"] <= a <= OBJECT["area_max"]):
            continue
        if x <= 1 or y <= 1 or x + w >= W - 2 or y + h >= H - 2:   # 화면 가장자리에 걸친 물체는 amodal 이 잘린다
            continue
        out.append(m)
    out.sort(key=lambda m: -m["predicted_iou"] * m["stability_score"])
    keep = []
    for m in out:   # 겹치는 후보 (부분 · 전체 관계) 는 점수 높은 쪽만
        s = m["segmentation"]
        if all((s & q["segmentation"]).sum() / min(s.sum(), q["segmentation"].sum()) < 0.5 for q in keep):
            keep.append(m)
        if len(keep) >= k:
            break
    return keep


def track(vp, frames: np.ndarray, seeds: list[np.ndarray]) -> np.ndarray:
    M, H, W, _ = frames.shape
    masks = np.zeros((len(seeds), M, H, W), bool)
    with tempfile.TemporaryDirectory() as td:
        for t, f in enumerate(frames):
            cv2.imwrite(f"{td}/{t:05d}.jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        state = vp.init_state(video_path=td)
        for k, s in enumerate(seeds):
            vp.add_new_mask(state, frame_idx=0, obj_id=k, mask=s)
        for fidx, obj_ids, logits in vp.propagate_in_video(state):
            for j, oid in enumerate(obj_ids):
                masks[oid, fidx] = (logits[j, 0] > 0).cpu().numpy()
        vp.reset_state(state)
    return masks


def camera_track(frames: np.ndarray) -> np.ndarray:
    """프레임 사이 전역 이동 (위상 상관) 을 누적한 카메라 궤적 (M, 2) px. 정지 배경은 이 궤적을 따라 움직인다."""
    g = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32) for f in frames]
    win = cv2.createHanningWindow(g[0].shape[::-1], cv2.CV_32F)
    cam = np.zeros((len(frames), 2))
    for t in range(1, len(frames)):
        (dx, dy), _ = cv2.phaseCorrelate(g[t - 1], g[t], win)
        cam[t] = cam[t - 1] + (dx, dy)
    return cam


def object_stats(m: np.ndarray, cam: np.ndarray) -> dict:
    """창 전체에서 물체 mask 의 안정성 · 움직임. reason = 첫 번째로 걸린 거르기 (없으면 ok).
    움직임은 카메라 궤적을 뺀 물체 중심 (rel) 으로 잰다 — 카메라가 흔들려도 정지 물체는 걸러진다."""
    M, H, W = m.shape
    area = m.sum((1, 2)).astype(float)
    st = dict(area_frac=[round(a / (H * W), 4) for a in area])
    if (area == 0).any():
        return dict(st, reason="lost")
    g = np.array([[np.nonzero(x)[1].mean(), np.nonzero(x)[0].mean()] for x in m])
    step = np.linalg.norm(np.diff(g, axis=0), axis=1)
    rel = g - cam
    border = int(sum(x[0].any() or x[-1].any() or x[:, 0].any() or x[:, -1].any() for x in m))
    st.update(area_ratio=float(area.max() / area.min()),
              step_ratio=float(np.max(np.maximum(area[1:] / area[:-1], area[:-1] / area[1:]))),
              max_jump_px=float(step.max()), path_px=float(step.sum()),
              rel_path_px=float(np.linalg.norm(np.diff(rel, axis=0), axis=1).sum()),
              rel_net_px=float(np.linalg.norm(rel - rel[0], axis=1).max()),
              cam_net_px=float(np.linalg.norm(cam - cam[0], axis=1).max()), border_frames=border)
    if st["area_ratio"] > OBJECT["area_ratio_max"]:
        st["reason"] = "area_unstable"
    elif st["step_ratio"] > OBJECT["step_ratio_max"]:
        st["reason"] = "area_jump"
    elif st["max_jump_px"] > MAX_JUMP_PX:
        st["reason"] = "centroid_jump"
    elif border > M // 4:
        st["reason"] = "touches_border"
    elif st["rel_net_px"] < OBJECT["min_rel_net_px"]:
        st["reason"] = "not_moving"
    else:
        st["reason"] = "ok"
    return st


def decide(d: Path, frames: np.ndarray, masks: np.ndarray, cand_meta: list[dict]) -> dict:
    cam = camera_track(frames)
    stats = [object_stats(m, cam) for m in masks]
    for c, s in zip(cand_meta, stats):
        s.update(pred_iou=c["pred_iou"], stability=c["stability"])
    ok = [i for i, s in enumerate(stats) if s["reason"] == "ok"]
    rep = dict(n_cands=len(stats), candidates=stats)
    (d / "object.npz").unlink(missing_ok=True)
    if ok:
        best = max(ok, key=lambda i: stats[i]["rel_net_px"])   # 카메라 뺀 이동이 가장 큰 물체
        np.savez_compressed(d / "object.npz", **{f"amodal_{k}": v for k, v in pack(masks[best]).items()})
        rep.update(status="ok", chosen=best)
    else:
        rep.update(status="no_object")
    save_json(d / "object.json", rep)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_cands", type=int, default=6)
    ap.add_argument("--points_per_side", type=int, default=32)
    ap.add_argument("--redo", action="store_true", help="object.json 이 있어도 SAM2 를 다시 돌린다")
    ap.add_argument("--refilter", action="store_true",
                    help="SAM2 없이 저장된 후보 (object_cands.npz) 로 거르기만 다시 한다 (CPU)")
    args = ap.parse_args()

    if not args.refilter:
        import sam2.automatic_mask_generator as amg_mod
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2, build_sam2_video_predictor
        from torchvision.ops.boxes import batched_nms

        # torchvision 0.27.1+cu129 의 NMS 에는 sm_120 (Blackwell) 커널이 없다 → CPU 에서 (2026-10-31 측정)
        amg_mod.batched_nms = lambda b, s, i, iou_threshold: batched_nms(
            b.float().cpu(), s.float().cpu(), i.cpu(), iou_threshold).to(b.device)

        dev = torch.device(args.device)
        amg = SAM2AutomaticMaskGenerator(build_sam2(SAM2_CFG, str(SAM2_CKPT), device=dev),
                                         points_per_side=args.points_per_side, pred_iou_thresh=0.8,
                                         stability_score_thresh=0.92, min_mask_region_area=50)
        vp = build_sam2_video_predictor(SAM2_CFG, str(SAM2_CKPT), device=dev)
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev.type == "cuda" else contextlib.nullcontext()

    n_ok = n = 0
    for d in sorted(p for p in args.clips.iterdir() if (p / "clean.npy").exists()):
        frames = np.load(d / "clean.npy")
        if args.refilter:
            if not (d / "object_cands.npz").exists():
                continue
            z = np.load(d / "object_cands.npz")
            masks = unpack(z["cands_bits"], z["cands_shape"])
            cand_meta = json.load(open(d / "object.json"))["candidates"]
        else:
            if (d / "object.json").exists() and not args.redo:
                continue
            with torch.inference_mode(), amp:
                cands = candidates(amg, frames[0], args.max_cands)
                if not cands:
                    save_json(d / "object.json", dict(status="no_candidate"))
                    print(f"{d.name}: no_candidate")
                    continue
                masks = track(vp, frames, [c["segmentation"] for c in cands])
            np.savez_compressed(d / "object_cands.npz", **{f"cands_{k}": v for k, v in pack(masks).items()})
            cand_meta = [dict(pred_iou=float(c["predicted_iou"]), stability=float(c["stability_score"]))
                         for c in cands]
        rep = decide(d, frames, masks, cand_meta)
        n += 1
        n_ok += rep["status"] == "ok"
        print(f"{d.name}: {rep['status']}  cands {[(s['reason'], round(s.get('rel_net_px', -1))) for s in rep['candidates']]}")
    print(f"ok {n_ok} / {n} (with candidates)")


if __name__ == "__main__":
    main()
