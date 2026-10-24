"""G19 운반 연산 (b): '픽셀을 실제 변위만큼 옮긴 뒤 표적 encoder 로 인코딩' 한 옮긴 블록 s_b (진행자 1, 2026-10-22).

정의 (근거표 §3-2a, 결과 전에 적음): 마지막 관측 블록 = 프레임 14 · 15, 다음 블록 = 16 · 17 (32 장 창).
  w16 = warp(f14 → 16), w17 = warp(f15 → 17)  (각 2 장 간격, 역방향 와핑: w_t(p) = f_s(p + d_{t→s}(p)))
  s_b = LN(target_encoder([w16, w17]))   — 하네스와 같은 블록별 LN 표적 공간, (B, 256, 1280)
변위:
  자연 영상 — Farneback 흐름 (cv2, 하네스 natcarry 와 같은 엔진) d_{t→s}. 앞-뒤 일관 마스크 (오차 ≤ tol px) 도 돌려준다.
  렌더 (VOBench · phys2d) — 라벨 변위. 물체 = 라벨 원판 (중심 · 반지름) 안 화소만 옮기고, 그 밖 (드러난 배경 포함) 은
  문맥 16 장의 화소별 중앙값 배경으로 채운다. 강체 평행 이동이라 거의 항등 → 렌더 판은 κ_a / κ_b (연산 손실) 로만 읽는다.
`carry_coef.py` (진행자 2) 의 S_MAKERS 에 꽂아 쓴다. 이 파일은 encoder 를 부르는 한 줄 (`encode_block`) 말고는 CPU 에서 검사된다.
"""
import numpy as np
import torch
import torch.nn.functional as F

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
SRC_TGT = ((14, 16), (15, 17))          # (source frame, target frame) — 2 장 간격, 블록 7 → 블록 8


def to_gray_u8(frame):
    """frame (3, H, W) normalized float tensor → (H, W) uint8 gray (cv2 용)."""
    x = (frame.detach().float().cpu() * STD + MEAN).clamp(0, 1)
    g = 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]
    return (g.numpy() * 255.0 + 0.5).astype(np.uint8)


def farneback(prev_u8, next_u8):
    """d such that prev(p) ≈ next(p + d(p)), (H, W, 2) in px (x, y). 하네스 flow_track 과 같은 Farneback 설정."""
    import cv2
    return cv2.calcOpticalFlowFarneback(prev_u8, next_u8, None, 0.5, 4, 21, 5, 7, 1.5, 0)


def backward_warp(src, disp):
    """src (C, H, W) tensor, disp (H, W, 2) px numpy/tensor: out(p) = src(p + disp(p)) (bilinear, border = edge)."""
    C, H, W = src.shape
    d = torch.as_tensor(disp, dtype=torch.float32, device=src.device)
    ys, xs = torch.meshgrid(torch.arange(H, device=src.device, dtype=torch.float32),
                            torch.arange(W, device=src.device, dtype=torch.float32), indexing="ij")
    gx = (xs + d[..., 0]) / (W - 1) * 2 - 1
    gy = (ys + d[..., 1]) / (H - 1) * 2 - 1
    grid = torch.stack([gx, gy], -1)[None]
    return F.grid_sample(src[None].float(), grid, mode="bilinear", padding_mode="border", align_corners=True)[0].to(src.dtype)


def natural_warps(clip, tol=1.0):
    """clip (3, T, H, W) normalized. Returns warped frames (3, 2, H, W), per-target backward flows [(H, W, 2)] ×2 and
    forward-backward consistency masks [(H, W) bool] ×2 (|d_{t→s}(p) + d_{s→t}(p + d_{t→s}(p))| ≤ tol)."""
    outs, flows, masks = [], [], []
    for s, t in SRC_TGT:
        gs, gt = to_gray_u8(clip[:, s]), to_gray_u8(clip[:, t])
        d_ts = farneback(gt, gs)                      # t(p) ≈ s(p + d_ts(p))
        d_st = farneback(gs, gt)
        back = backward_warp(torch.from_numpy(d_st).permute(2, 0, 1), d_ts).permute(1, 2, 0).numpy()
        masks.append(np.linalg.norm(d_ts + back, axis=-1) <= tol)
        flows.append(d_ts)
        outs.append(backward_warp(clip[:, s], d_ts))
    return torch.stack(outs, 1), flows, masks


def disk(H, W, c, r):
    ys, xs = np.mgrid[0:H, 0:W]
    return (xs - c[0]) ** 2 + (ys - c[1]) ** 2 <= r * r


def render_warps(clip, P, radius, ctx=16):
    """Render (labels). clip (3, T, H, W) normalized; P (T, 2) object centre per frame in THIS clip's px (x, y);
    radius (px, this clip's scale) — scalar or (T,). Object pixels move rigidly by P[t] − P[s]; everything else = per-pixel
    median of the first `ctx` frames (background; a slow object can leak into it — caveat)."""
    C, T, H, W = clip.shape
    bg = clip[:, :ctx].float().median(dim=1).values.to(clip.dtype)
    rad = np.broadcast_to(np.asarray(radius, np.float64), (T,))
    outs, masks = [], []
    for s, t in SRC_TGT:
        m = disk(H, W, P[t], rad[t])
        disp = np.zeros((H, W, 2), np.float32)
        disp[m] = (np.asarray(P[s]) - np.asarray(P[t])).astype(np.float32)          # w_t(p) = f_s(p + (P_s − P_t)) inside the disk
        moved = backward_warp(clip[:, s], disp)
        mt = torch.from_numpy(m).to(clip.device)
        outs.append(torch.where(mt[None], moved, bg))
        masks.append(m)
    return torch.stack(outs, 1), masks


def encode_block(target_encoder, block):
    """block (B, 3, 2, H, W) → LN(target_encoder(block)) (B, 256, D) — the harness / block-AR target space."""
    from app.vjepa_frozen.ar import encode_blocks
    return encode_blocks(target_encoder, block, 2)


def make_s_pixel(target_encoder, clips, kind="natural", P=None, radius=None, tol=1.0):
    """s_b for a batch. clips (B, 3, 32, H, W) normalized on the encoder device. kind = natural | render.
    Returns s_b (B, 256, D) and per-clip side info (flows / masks) for the caller's moving-cell selection."""
    blocks, side = [], []
    for b in range(clips.size(0)):
        if kind == "natural":
            w, flows, masks = natural_warps(clips[b], tol=tol)
            side.append({"flows": flows, "consistent": masks})
        else:
            w, masks = render_warps(clips[b], P[b], radius[b])
            side.append({"object_masks": masks})
        blocks.append(w)
    return encode_block(target_encoder, torch.stack(blocks)), side


if __name__ == "__main__":
    # CPU self-test without the encoder: a textured square moving 6 px / frame on a static noisy background.
    rng = np.random.default_rng(0)
    H = W = 64
    T = 32
    bgimg = rng.random((3, H, W)).astype(np.float32)
    tex = rng.random((3, 12, 12)).astype(np.float32)
    frames, P = [], []
    for f in range(T):
        x = bgimg.copy()
        cx, cy = 8 + 1.5 * f, 30.0
        x[:, int(cy) - 6:int(cy) + 6, int(cx) - 6:int(cx) + 6] = tex
        frames.append((torch.from_numpy(x) - MEAN) / STD)
        P.append((cx - 0.5, cy - 0.5))
    clip = torch.stack(frames, 1)
    P = np.asarray(P)
    w, masks = render_warps(clip, P, radius=6.0)
    for i, (s, t) in enumerate(SRC_TGT):
        err_w = (w[:, i] - clip[:, t]).abs().mean().item()
        err_c = (clip[:, s] - clip[:, t]).abs().mean().item()
        print(f"render warp {s}->{t}: |warp - truth| {err_w:.4f} vs |copy - truth| {err_c:.4f}")
    wn, flows, cm = natural_warps(clip)
    for i, (s, t) in enumerate(SRC_TGT):
        err_w = (wn[:, i] - clip[:, t]).abs().mean().item()
        err_c = (clip[:, s] - clip[:, t]).abs().mean().item()
        print(f"farneback warp {s}->{t}: |warp - truth| {err_w:.4f} vs |copy - truth| {err_c:.4f} · consistent {cm[i].mean():.2f}")
