#!/usr/bin/env python3
"""[물리] 존재 변수의 중앙값 전제 확인: 자연 영상에서 가려졌다가 다시 드러난 자리에 "같은 내용" 이 돌아오나 (π), 학습 없음.

  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 /venv/vjepa/bin/python -W ignore auto_research/scripts/reveal_return.py

창: 학습 · 보류 캐시 메타 (16 장, 가운데 정사각 → 256, ≈ 4 fps). RAFT 앞 · 뒤 흐름 (이웃 장 15 쌍).
가림 (앞-뒤 불일치, Sundaram 등 2010): 장 i 의 픽셀 x 가 i+1 에서 안 보임 ⇔ |F + B∘(x+F)|² > 0.01(|F|²+|B∘|²) + 0.5, x+F 가 화면 안.
드러남: 장 j 의 픽셀 z 가 j−1 에 없던 내용 ⇔ |B + F∘(z+B)|² > … (같은 꼴).
사건 (정지 카메라 · 정지 내용으로 한정해 같은 픽셀 비교가 유효하게): 장 j0 에서 z 의 내용이 정지 (|F_{j0−1}(z)| < 1 px) · 결이 있음 (7×7 회색 표준편차 > 8)
  · j0 → j0+1 에서 가려짐, 그 뒤 처음 드러나는 장 j (≥ j0+2). 두 쌍 모두 카메라 정지 (흐름 크기 중앙값 < 1 px).
돌아옴 = 7×7 RGB 패치 평균 절대차 |I_j(z) − I_{j0}(z)| < τ. τ = 같은 간격에서 한 번도 가림 · 드러남 없이 정지한 결 있는 픽셀 차이의 90 분위 (위 기준 ≈ 0.9).
아래 기준 = 같은 사건 픽셀을 48 px 떨어진 자리의 j0 패치와 비교한 돌아옴 비율 (우연).
π 추정 = (돌아옴 − 아래) / (0.9 − 아래). 가려진 채 창이 끝난 사건 수도 센다."""
import json, os, sys
import numpy as np, torch, torch.nn.functional as F
import decord
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
torch.set_num_threads(8)
dev = torch.device("cuda")
raft = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(dev).eval()
SRC = [("k710_holdout", "/workspace/auto_research_cache/natural_holdout/k710_320_holdout_meta.json", 150),
       ("ht100m", "/workspace/auto_research_cache/natural_train/ht100m_1pct_meta.json", 120),
       ("ssv2", "/workspace/auto_research_cache/natural_train/ssv2_meta.json", 120)]
NF = 16
CLEAN = int(os.environ.get("CLEAN", "1"))
H = W = 256
yy, xx = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
base = torch.stack([xx, yy], 0).float()


def warp(t, flow):  # t (N,C,H,W) 를 flow (N,2,H,W) 자리에서 읽기
    g = base[None] + flow
    gx = 2 * g[:, 0] / (W - 1) - 1; gy = 2 * g[:, 1] / (H - 1) - 1
    return F.grid_sample(t, torch.stack([gx, gy], -1), align_corners=True, mode="bilinear", padding_mode="border")


def inside(flow):
    g = base[None] + flow
    return (g[:, 0] >= 0) & (g[:, 0] <= W - 1) & (g[:, 1] >= 0) & (g[:, 1] <= H - 1)


def fb(a, b):  # a 의 픽셀이 b 쪽에서 일관된가: a=(N,2,H,W) 흐름, b=반대 흐름. 불일치 마스크
    bw = warp(b, a)
    s = a + bw
    return ((s ** 2).sum(1) > 0.01 * ((a ** 2).sum(1) + (bw ** 2).sum(1)) + 0.5) & inside(a)


def load(m):
    if "png_dir" in m:                    # 양성 대조 (VOBench PNG)
        from PIL import Image
        fr = np.stack([np.asarray(Image.open(f"{m['png_dir']}/{i:06d}.png").convert("RGB")) for i in m["idx"]])
        x = torch.from_numpy(fr).to(dev).permute(0, 3, 1, 2).float()
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False).clamp(0, 255).round()
    vr = decord.VideoReader(m["path"], num_threads=1)
    idx = list(range(m["start"], m["start"] + (NF - 1) * m["fstp"] + 1, m["fstp"]))
    if idx[-1] >= len(vr):
        return None
    fr = vr.get_batch(idx).asnumpy()
    Hh, Ww = fr.shape[1:3]; side = min(Hh, Ww); y0, x0 = (Hh - side) // 2, (Ww - side) // 2
    x = torch.from_numpy(fr[:, y0:y0 + side, x0:x0 + side]).to(dev).permute(0, 3, 1, 2).float()
    return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False).clamp(0, 255).round()


if os.environ.get("POSCTRL"):
    import csv
    REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    rows = [r for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv")))
            if r["occlusion"] == "mid" and r["variant"] == "pos_a"]
    VB = "/workspace/dataset/benchmarks/VOBench_v1"
    SRC = [("vob_mid_static", [dict(png_dir=os.path.join(VB, r["file_name"]), idx=list(range(0, 96, 6))) for r in rows if r["motion"] == "flat_static"], 999),
           ("vob_mid_moving", [dict(png_dir=os.path.join(VB, r["file_name"]), idx=list(range(0, 96, 6))) for r in rows if r["motion"] != "flat_static"][:72], 999)]
out = {}
for name, mp, nmax in SRC:
    if isinstance(mp, list):
        meta = mp[:nmax]
    elif not os.path.exists(mp):
        continue
    else:
        meta = json.load(open(mp))[:nmax]
    ev_d, ev_rand, ev_gap, ctrl_d, ctrl_gap = [], [], [], [], []
    n_still_hidden = 0; n_events = 0; n_win = 0
    for wi, m in enumerate(meta):
        x = load(m)
        if x is None:
            continue
        n_win += 1
        with torch.no_grad():
            xn = x / 127.5 - 1
            Fw = raft(xn[:-1], xn[1:], num_flow_updates=12)[-1]     # i → i+1
            Bw = raft(xn[1:], xn[:-1], num_flow_updates=12)[-1]     # i+1 → i
        occ = fb(Fw, Bw)            # (15,H,W) 장 i 픽셀이 i+1 에서 안 보임
        dis = fb(Bw, Fw)            # (15,H,W) 장 i+1 픽셀이 i 에 없던 것
        mag = Fw.norm(dim=1)        # (15,H,W)
        cam_static = mag.flatten(1).median(1).values < 1.0       # (15,)
        gray = x.mean(1, keepdim=True)
        mu = F.avg_pool2d(gray, 7, 1, 3); sd = (F.avg_pool2d(gray ** 2, 7, 1, 3) - mu ** 2).clamp(min=0).sqrt()[:, 0]  # (16,H,W)
        # 장마다 패치 차이 계산용
        def pdiff(j, j0, shift=0):
            a = x[j]; b = x[j0] if shift == 0 else torch.roll(x[j0], shifts=(shift, shift), dims=(1, 2))
            return F.avg_pool2d((a - b).abs().mean(0, keepdim=True)[None], 7, 1, 3)[0, 0]
        occ_c = occ.cpu().numpy(); dis_c = dis.cpu().numpy(); mag_c = mag.cpu().numpy(); sd_c = sd.cpu().numpy(); cs = cam_static.cpu().numpy()
        for j0 in range(1, NF - 2):
            if not cs[j0] or not cs[j0 - 1]:
                continue
            ev = occ_c[j0] & (mag_c[j0 - 1] < 1.0) & (sd_c[j0] > 8)
            ys, xs = np.nonzero(ev)
            if ys.size == 0:
                continue
            if ys.size > 4000:
                sel = np.random.default_rng(wi * 100 + j0).choice(ys.size, 4000, replace=False); ys, xs = ys[sel], xs[sel]
            found = np.zeros(ys.size, bool)
            for j in range(j0 + 2, NF):
                if not cs[j - 1]:
                    continue
                r = dis_c[j - 1][ys, xs] & ~found
                if CLEAN and j < NF - 1:   # 드러난 내용이 정지 (다음 장으로 < 1 px) 하고 바로 다시 가려지지 않음
                    r &= (mag_c[j][ys, xs] < 1.0) & (~occ_c[j][ys, xs])
                if r.any():
                    d = pdiff(j, j0).cpu().numpy()[ys[r], xs[r]]
                    dr = pdiff(j, j0, shift=48).cpu().numpy()[ys[r], xs[r]]
                    ev_d.append(d); ev_rand.append(dr); ev_gap.append(np.full(r.sum(), j - j0))
                    found |= r
            n_events += ys.size; n_still_hidden += int((~found).sum())
            # 대조: 같은 j0 에서 정지 · 결 있음 · 가림/드러남 없음이 이어지는 픽셀
            ctl = (~occ_c[j0]) & (mag_c[j0 - 1] < 0.5) & (mag_c[j0] < 0.5) & (sd_c[j0] > 8)
            cy, cxx = np.nonzero(ctl)
            if cy.size > 2000:
                sel = np.random.default_rng(wi * 7 + j0).choice(cy.size, 2000, replace=False); cy, cxx = cy[sel], cxx[sel]
            for g in (2, 3, 4):
                j = j0 + g
                if j >= NF or not all(cs[j0:j]):
                    continue
                ok = np.ones(cy.size, bool)
                for jj in range(j0, j):
                    ok &= (~occ_c[jj][cy, cxx]) & (mag_c[jj][cy, cxx] < 0.5)
                if ok.any():
                    ctrl_d.append(pdiff(j, j0).cpu().numpy()[cy[ok], cxx[ok]]); ctrl_gap.append(np.full(ok.sum(), g))
        if (wi + 1) % 30 == 0:
            print(name, wi + 1, "events", n_events, flush=True)
    if not ev_d:
        out[name] = {"n_windows": n_win, "n_events": n_events}
        print(name, out[name]); continue
    d = np.concatenate(ev_d); dr = np.concatenate(ev_rand); g = np.concatenate(ev_gap)
    cd = np.concatenate(ctrl_d) if ctrl_d else np.array([np.nan]); cg = np.concatenate(ctrl_gap) if ctrl_gap else np.array([0])
    tau = float(np.nanquantile(cd, 0.9))
    ret = float((d < tau).mean()); low = float((dr < tau).mean())
    pi = (ret - low) / max(0.9 - low, 1e-6)
    bygap = {int(k): dict(n=int((g == k).sum()), returned=round(float((d[g == k] < tau).mean()), 3), random=round(float((dr[g == k] < tau).mean()), 3)) for k in np.unique(g) if (g == k).sum() >= 50}
    out[name] = dict(n_windows=n_win, n_events=int(n_events), n_revealed=int(d.size), frac_revealed_in_window=round(d.size / max(n_events, 1), 3),
                     tau=round(tau, 2), returned=round(ret, 3), random_baseline=round(low, 3), pi_est=round(float(pi), 3),
                     ctrl_n=int(cd.size), by_gap=bygap)
    print(name, json.dumps(out[name]), flush=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), f"../results/reveal_return{'_clean' if CLEAN else ''}{'_posctrl' if os.environ.get('POSCTRL') else ''}.json"), "w"), indent=1)
