#!/usr/bin/env python3
"""[T] 고리 5 · 빠진 고리: 자연 영상에서 "내용을 옮기는 것" 이 블록 latent L1 을 얼마나 줄이나 (auto_research, 2026-10-21).

  # phys3d val 캐시 위 (라벨 기반 결과와 대조해 이 방법을 검증)
  CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python -W ignore auto_research/scripts/natural_transport.py --src phys3d --out <dir>
  # 자연 영상 (K710 · SSv2 · HowTo100M 1 %, 4 fps, 학습 로더와 같은 읽기: 높이 전체 정사각 가운데 자르기 → 256)
  CUDA_VISIBLE_DEVICES=1 /venv/vjepa/bin/python -W ignore auto_research/scripts/natural_transport.py --src natural --n_per 150 --out <dir>
  # 합치기
  /venv/vjepa/bin/python auto_research/scripts/natural_transport.py --report <dir>

창 = 32 장 = 블록 16 개 (블록 = 연속 2 장, 블록 토큰 = 블록별 LN(target_encoder), bf16 — 학습 정답과 같다).
광학 흐름: RAFT-large (torchvision), 블록 대표 장 = 블록의 둘째 장. U[b] = 블록 b 대표 장 → 블록 b−1 대표 장의 역방향 흐름을 16 px 칸 평균 (칸 단위).
정답 블록 t+1 (t = 7..14), 문맥 블록 t−7..t. 학습 없는 예측기 (칸마다):
  copy        h_t(c)
  mean8       최근 8 블록 평균 (같은 칸)
  ls8         3D 합성에서 구한 합 1 최소제곱 가중 (블록 t−7..t: 0.09 0.07 0.07 0.06 0.08 0.10 0.17 0.36) 을 그대로 건 것.
              이 데이터 자체의 최소제곱 가중 (Gram 누적) 은 따로 보고한다 — 학습된 predictor 의 가중합 측정 (마지막 0.46 · 합 0.96) 과 대조
  al_stay     문맥 흐름으로 궤적을 따라 평균 (마지막 걸음은 제자리) = 움직임 보정 평균, 운반 없음
  al_oracle   같은 평균인데 마지막 걸음을 **참 흐름** U[t+1] 로 = 운반 (오라클)
  al_persist  마지막 걸음을 직전 흐름 U[t] (같은 칸) 로 = 문맥만으로 할 수 있는 운반 (등속 · 오일러 지속)
  wc_oracle   h_t 를 참 흐름으로 한 번 옮긴 것 (평균 없음)
보상 (copy 손실 대비 %): 평균 = L(copy) − L(mean8) · 움직임 보정 평균 = L(mean8) − L(al_stay) · 운반 (오라클) = L(al_stay) − L(al_oracle) ·
운반 (예측 가능) = L(al_stay) − L(al_persist). 움직이는 칸 (|U[t+1]| > 0.5 칸) 만 / 카메라 움직임 (칸 흐름 중앙값 > 0.5 칸) 여부로 나눈다.
흐름 지속성: 움직이는 칸에서 cos(U[t+1], U[t]) · |U[t+1]| / |U[t]|."""
import argparse, glob, json, os, random, sys
import numpy as np, torch, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--src", choices=["phys3d", "natural", "enc_noise"], default="natural")
ap.add_argument("--en_src", default="ssv2", help="--src enc_noise: [E] 수집 캐시의 소스 이름 (블록 8 개 창, 학습 정답과 같은 16 비트 토큰 · 픽셀)")
ap.add_argument("--ctx", type=int, default=8, help="문맥 블록 수 (정답 블록 t+1, 문맥 t−ctx+1..t)")
ap.add_argument("--cache", default="/workspace/auto_research_cache/phys3d")
ap.add_argument("--n_per", type=int, default=150, help="자연 영상: 소스마다 clip 수")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=None)
ap.add_argument("--report", default=None)
ap.add_argument("--limit", type=int, default=0, help="창 수 상한 (스모크)")
a = ap.parse_args()
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
G, NB, S = 16, 16, 256
if a.src == "enc_noise":
    NB = 8
PRED = ["copy", "mean8", "ls8", "al_stay", "al_oracle", "al_persist", "wc_oracle"]
LSW = np.array([0.0936, 0.0688, 0.0675, 0.0645, 0.0759, 0.1017, 0.1686, 0.3594], np.float32)

if a.report:
    fs = sorted(glob.glob(os.path.join(a.report, "rows_*.npz")))
    for f in fs:
        z = np.load(f, allow_pickle=True)
        src = os.path.basename(f)[5:-4]
        L = {p: z[p] for p in PRED}                    # (n_targets, 256) 칸별 L1
        mag, cam, cosp, ratio = z["mag"], z["cam"], z["cos"], z["ratio"]
        mov = mag > 0.5
        print(f"\n=== {src}: 정답 블록 {L['copy'].shape[0]} 개, 움직이는 칸 비율 {mov.mean():.3f}, 카메라 움직임 창 비율 {cam.mean():.3f}")
        for gname, m_row, m_cell in (("전체 칸", np.ones(len(cam), bool), np.ones_like(mov)), ("움직이는 칸만", np.ones(len(cam), bool), mov),
                                     ("카메라 정지 창 · 전체", ~cam, np.ones_like(mov)), ("카메라 정지 창 · 움직이는 칸", ~cam, mov),
                                     ("카메라 움직임 창 · 전체", cam, np.ones_like(mov))):
            mm = m_cell & m_row[:, None]
            if mm.sum() < 50:
                continue
            v = {p: float(L[p][mm].mean()) for p in PRED}
            c = v["copy"]
            print(f"  [{gname}] (칸 {mm.sum()}) " + " · ".join(f"{p} {v[p]:.4f}" for p in PRED))
            print(f"      보상 (copy 대비 %): 평균 {100 * (c - v['mean8']) / c:+.1f} · 선형최적 {100 * (c - v['ls8']) / c:+.1f} · 움직임보정평균 {100 * (v['mean8'] - v['al_stay']) / c:+.1f}"
                  f" · 운반(오라클) {100 * (v['al_stay'] - v['al_oracle']) / c:+.1f} · 운반(지속 예측) {100 * (v['al_stay'] - v['al_persist']) / c:+.1f}"
                  f" · 한 블록 옮기기 vs copy {100 * (c - v['wc_oracle']) / c:+.1f}")
        ok = np.isfinite(cosp)
        print(f"  흐름 지속성 (움직이는 칸, n {ok.sum()}): cos 중앙값 {np.median(cosp[ok]):.2f} · cos>0.7 비율 {(cosp[ok] > 0.7).mean():.2f} · 크기비 중앙값 {np.median(ratio[ok]):.2f}")
        if "w_ls" in z:
            print(f"  최소제곱 가중 (블록 t−7..t): {np.round(z['w_ls'], 3).tolist()} 합 {z['w_ls'].sum():.3f}")
        if "vario" in z:
            print(f"  정지 칸 변이도 E|h_(b+k) − h_b| (k = 1..8): {np.round(z['vario'][:8], 3).tolist()}  (n 칸 {int(z['vario_n'])})")
    sys.exit()

sys.path.insert(0, REPO)
dev = torch.device("cuda")
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights   # noqa: E402
raft = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(dev).eval()


@torch.no_grad()
def flows(frames):                                   # frames (32,256,256,3) uint8 → U (16,16,16,2) 칸 단위, U[b] = 대표 b → b−1 (b≥1)
    rep = torch.from_numpy(np.ascontiguousarray(frames[1::2])).to(dev).permute(0, 3, 1, 2).float() / 127.5 - 1
    f = raft(rep[1:], rep[:-1], num_flow_updates=12)[-1]          # (15,2,256,256) (dx, dy) px
    u = F.avg_pool2d(f, 16) / 16                                    # (15,2,16,16)
    U = torch.zeros(NB, G, G, 2, device=dev)
    U[1:] = u.permute(0, 2, 3, 1)
    return U.cpu().numpy()


if a.src == "natural":
    from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint   # noqa: E402
    from app.vjepa_frozen import ar as AR                                                    # noqa: E402
    import decord
    CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
    enc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
    _load_state_dict(enc, _load_checkpoint(CK)["target_encoder"], tag="t"); enc = enc.to(dev, torch.bfloat16).eval()
    MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)

    def windows():
        rng = random.Random(a.seed)
        for src in ("k710_320", "ssv2", "ht100m_1pct"):
            rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(REPO, "data_csv", src, "fps4_table.tsv"))]
            rows = [r for r in rows if int(r[4]) >= 32]
            rng.shuffle(rows)
            got = 0
            for r in rows:
                if got >= a.n_per:
                    break
                try:
                    vr = decord.VideoReader(r[0], num_threads=4)
                    fstp = int(r[3]); span = 31 * fstp + 1
                    if len(vr) < span:
                        continue
                    s = rng.randint(0, len(vr) - span)
                    fr = vr.get_batch(list(range(s, s + span, fstp))).asnumpy()       # (32,H,W,3)
                except Exception:
                    continue
                H, W = fr.shape[1:3]
                side = min(H, W); y0, x0 = (H - side) // 2, (W - side) // 2
                x = torch.from_numpy(fr[:, y0:y0 + side, x0:x0 + side]).to(dev).permute(0, 3, 1, 2).float()
                x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)     # (32,3,256,256)
                frames = x.clamp(0, 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
                xx = (x / 255).permute(1, 0, 2, 3)[None]
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    h = AR.encode_blocks(enc, ((xx - MEAN) / STD).to(torch.bfloat16), 2)[0].view(NB, S, -1).float().cpu().numpy()
                got += 1
                yield src, h, frames
elif a.src == "enc_noise":
    def windows():
        EN = "/workspace/auto_research_cache/enc_noise"
        for sh in (0, 1):
            T_ = np.load(os.path.join(EN, f"{a.en_src}_target_encoder_s{sh}_tok.npy"), mmap_mode="r")
            P_ = np.load(os.path.join(EN, f"{a.en_src}_target_encoder_s{sh}_pix.npy"), mmap_mode="r")
            for w in range(T_.shape[0]):
                yield a.en_src, np.asarray(T_[w, 0]).astype(np.float32), np.asarray(P_[w])
else:
    def windows():
        for f in sorted(glob.glob(os.path.join(a.cache, "h_val_target_s*.npy"))):
            sh = f.split("_s")[-1].split(".")[0]
            X = np.load(f, mmap_mode="r"); P = np.load(os.path.join(a.cache, f"pix_val_s{sh}.npy"), mmap_mode="r")
            for w in range(X.shape[0]):
                yield "phys3d", X[w].astype(np.float32), np.asarray(P[w])

cy, cx = np.divmod(np.arange(S), G)
P0 = np.stack([cx + 0.5, cy + 0.5], 1).astype(np.float32)          # 칸 중심 (x, y)


def idx_of(p):
    ix = np.floor(p).astype(int)
    ok = (ix[:, 0] >= 0) & (ix[:, 0] < G) & (ix[:, 1] >= 0) & (ix[:, 1] < G)
    return np.clip(ix[:, 1], 0, G - 1) * G + np.clip(ix[:, 0], 0, G - 1), ok


def aligned_mean(h, U, t, p_t):
    """p_t (256,2) = 블록 t 에서의 출발 위치. 문맥 흐름으로 t−7 까지 거슬러 가며 그 칸 토큰을 평균 (화면 밖은 뺀다)."""
    acc = np.zeros((S, h.shape[-1]), np.float32); n = np.zeros(S, np.float32)
    p = p_t.copy(); alive = np.ones(S, bool)
    for b in range(t, t - a.ctx, -1):
        j, ok = idx_of(p)
        alive &= ok
        acc[alive] += h[b][j[alive]]; n[alive] += 1
        if b > t - a.ctx + 1:
            p = p + U[b][j // G, j % G]
    out = acc / np.maximum(n, 1)[:, None]
    out[n == 0] = h[t][n == 0]
    return out


rows = {}
gram = {}
vario = np.zeros(15); vario_n = 0
for wi, (src, h, frames) in enumerate(windows()):
    if a.limit and wi >= a.limit:
        break
    U = flows(frames)
    R = rows.setdefault(src, {k: [] for k in PRED + ["mag", "cam", "cos", "ratio", "Hctx", "Y", "Uwin"]})
    R["Uwin"].append(U.astype(np.float16))                              # 창별 칸 흐름 (π̄ 분석용)
    gm = gram.setdefault(src, [np.zeros((a.ctx, a.ctx)), np.zeros(a.ctx), 0])
    mags = np.linalg.norm(U[1:], axis=-1)                               # (15,16,16)
    cam = float(np.median(mags)) > 0.5
    # 정지 칸 변이도: 16 블록 내내 흐름 < 0.15 칸인 칸
    still = (mags.max(0) < 0.15).reshape(-1)
    if still.sum() > 0:
        hs = h[:, still]
        for k in range(1, NB):
            vario[k - 1] += np.abs(hs[k:] - hs[:-k]).mean() * still.sum()
        vario_n += still.sum()
    for t in range(a.ctx - 1, NB - 1):
        y = h[t + 1]
        Ut1 = U[t + 1].reshape(S, 2); Ut = U[t].reshape(S, 2)
        Pp = {"copy": h[t], "mean8": h[t - a.ctx + 1:t + 1].mean(0)}
        Pp["al_stay"] = aligned_mean(h, U, t, P0)
        Pp["al_oracle"] = aligned_mean(h, U, t, P0 + Ut1)
        Pp["al_persist"] = aligned_mean(h, U, t, P0 + Ut)
        j, ok = idx_of(P0 + Ut1)
        wc = h[t][j].copy(); wc[~ok] = h[t][~ok]
        Pp["wc_oracle"] = wc
        Hc = h[t - a.ctx + 1:t + 1]                                     # (ctx,256,D)
        Pp["ls8"] = np.tensordot(LSW[-a.ctx:] / LSW[-a.ctx:].sum(), Hc, 1)
        for k in PRED:
            R[k].append(np.abs(Pp[k] - y).mean(-1).astype(np.float32))
        # 최소제곱 가중 (Gram 누적; 예측은 마지막에 교차로)
        R["Hctx"].append(wi); R["Y"].append(t)
        Hf = Hc.reshape(a.ctx, -1).astype(np.float64)
        gm[0] += Hf @ Hf.T; gm[1] += Hf @ y.reshape(-1).astype(np.float64); gm[2] += 1
        mag = np.linalg.norm(Ut1, axis=-1)
        R["mag"].append(mag.astype(np.float32)); R["cam"].append(cam)
        mv = mag > 0.5
        nu = np.linalg.norm(Ut, axis=-1)
        cs = (Ut1 * Ut).sum(-1) / np.maximum(mag * nu, 1e-6)
        R["cos"].append(np.where(mv & (nu > 0.1), cs, np.nan).astype(np.float32))
        R["ratio"].append(np.where(mv & (nu > 0.1), mag / np.maximum(nu, 1e-6), np.nan).astype(np.float32))
    if wi % 20 == 0:
        print(f"{wi} 창 ({src})", flush=True)

os.makedirs(a.out, exist_ok=True)
for src, R in rows.items():
    M, v, n = gram[src]
    Ai = np.linalg.inv(M + 1e-6 * np.trace(M) / a.ctx * np.eye(a.ctx)); one = np.ones(a.ctx)
    wu = Ai @ v; w = wu + (1 - one @ wu) / (one @ Ai @ one) * (Ai @ one)        # 합 1 제약 최소제곱
    out = {k: np.stack(R[k]) for k in PRED}
    out.update(Uwin=np.stack(R["Uwin"]), ctx=a.ctx, mag=np.stack(R["mag"]), cam=np.array(R["cam"]), cos=np.stack(R["cos"]), ratio=np.stack(R["ratio"]), w_ls=w,
               vario=vario / max(vario_n, 1), vario_n=vario_n)
    np.savez(os.path.join(a.out, f"rows_{src}.npz"), **out)
print("done", {k: len(v["copy"]) for k, v in rows.items()})
