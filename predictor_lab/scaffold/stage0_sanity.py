#!/usr/bin/env python3
"""자리 계산기 (scaffold) 0 단계 sanity — 계산기 혼자 물체 자리를 읽고 미래로 굴려 맞히나 (2026-11-01, 사용자 요청).

무엇을 보나 (predictor 없음, 학습은 ① 의 물체다움 벡터 w 하나)
  A. 읽기: 보이는 문맥 블록에서 읽은 자리 x_t 가 참 자리와 몇 칸 차이 나나 · 보이는 정도 m 이 보임 / 가림을 가르나 (AUC)
  B. 굴리기: late 창 (문맥 끝에서 가려짐) 에서 미래 블록 자리 오차 — 직선 · 2 차식 × "보이는 블록만" · "m 무게"
  C. 속도 sweep: VOBench (w 학습에 안 쓴 생성기) late 이동 영속 block 을 속도 1 · 2/3 · 1/3 로 — 속도 따라감 비 (PFT 0.17)
데이터
  w 학습: BlenderTrack v1 train (옆으로 움직이는 판 없는 목록) 400 clip, 창 하나씩 (간격 1–3 무작위, late 창 우선)
  B 평가: BlenderTrack v1 val (같은 목록) 200 clip 의 late 창 · C 평가: VOBench split v2 test (E35r4 · 속도 sweep 과 같은 block)
  z = 문맥 encoder 출력 (릴리즈 ViT-H, 채점 하네스와 같은 전처리 · fp32 + autocast fp16). 참 자리 = 라벨 (BlenderTrack center_px · VOBench label.json u, v)
  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P predictor_lab/scaffold/stage0_sanity.py run        # GPU 1 장: 뽑기 → w 학습 → 평가 (→ predictor_lab/scaffold/results/stage0.json)
  $P predictor_lab/scaffold/stage0_sanity.py eval       # CPU: 뽑아 둔 z 로 w 학습 · 평가만 다시
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "z_research/scripts/analysis"))
from app.vjepa_frozen import scaffold as SC  # noqa: E402

OUT = Path("/data/hyuntak/project/2026/2027_cvpr/nfs_cache/scaffold_stage0")          # z 배열 (NFS, 약 3.6 GB)
RES = REPO / "predictor_lab/scaffold/results"
BT = REPO / "data_csv/blender_track_v1"
VOB_SRC = Path("/data/hyuntak/project/2026/2027_cvpr/UnrealEngine/VOBench_v1")
VOB_SPEEDS = [("S1", 30, 1, "1/3"), ("T1", 15, 2, "2/3"), ("orig", 0, 3, "1 (train)")]
N, C = 32, 16                       # 창 32 샘플 · 문맥 16 = 블록 8 · 미래 블록 8
NB, NC = N // 2, C // 2


# ----------------------------------------------------------------------------- BlenderTrack 창
def bt_windows(meta: Path, n_clip: int, seed: int, late_only: bool):
    m = np.load(meta)
    vis = np.nan_to_num(m["vis"].astype(np.float32), nan=-1.0)          # 화면 밖 = −1
    rng = np.random.default_rng(seed)
    out = []
    for c in rng.permutation(len(m["clip"])):
        s = int(rng.integers(1, 4)); span = (N - 1) * s + 1
        J = np.arange(vis.shape[1] - span + 1)[:, None] + s * np.arange(N)[None]
        v = vis[c][J]                                                    # (O, N)
        late = (v[:, C - 1] >= 0) & (v[:, C - 1] < 0.1) & (v[:, :C - 1] >= 0.5).any(1) & (v[:, C:] >= 0.5).any(1)
        ok = np.nonzero(late)[0]
        if len(ok):
            o = int(rng.choice(ok))
        elif late_only:
            continue
        else:
            o = int(rng.integers(0, len(J)))
        out.append((str(m["clip"][c]), o + s * np.arange(N), s, bool(len(ok))))
        if len(out) == n_clip:
            break
    return out, str(m["root"]), str(m["labels_root"])


def bt_labels(lab_root: str, cid: str, idx: np.ndarray):
    z = np.load(f"{lab_root}/{cid}.npz")
    ctr = z["center_px"][idx, 0].astype(np.float32) / 16.0 - 0.5           # (N, 2) 토큰 좌표 (열, 행)
    vis = np.where(z["amodal_n"][idx, 0] > 0, z["visible_n"][idx, 0] / np.maximum(z["amodal_n"][idx, 0], 1), -1.0).astype(np.float32)
    vi = np.unpackbits(z["visible"][idx, 0], axis=-1).astype(np.float32)   # (N, 64, 64)
    q = vi.reshape(NB, 2, 16, 4, 16, 4).mean((1, 3, 5)).reshape(NB, 256)   # 블록별 보이는 칸 비율 (16×16)
    return ctr.reshape(NB, 2, 2).mean(1), vis.reshape(NB, 2).mean(1), q[:NC]


# ----------------------------------------------------------------------------- 뽑기 (GPU)
def extract():
    import decord
    import te_v11_score as te
    import transfer_gap_speed_sweep  # noqa: F401  (r4.SETS 에 S1 추가)
    import transfer_gap_r4_endprior as r4
    from evals.world_model_analysis.data import MEAN, STD
    r4.setup("base")                                   # resolve 를 NFS 사본 경로로 바꾸는 패치 — import 보다 먼저
    import transfer_gap_te as tte
    resolve = tte.resolve
    ep = r4.ep
    dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
    cfg0, _ = resolve("vobench_v1_split2_test", "vith", [], None, 0)
    sc = te.Scorer(cfg0, dev, [])
    D = sc.D
    OUT.mkdir(parents=True, exist_ok=True)

    def enc(clips):
        z, _, _, _, _ = sc.encode(clips)
        return z.float().reshape(clips.size(0), NC, 256, D).half().cpu().numpy()

    for name, meta, n_clip, late_only, seed in (("bt_train", BT / "train_meta_noslide.npz", 400, False, 0),
                                                ("bt_val", BT / "val_meta_noslide.npz", 200, True, 1)):
        wins, root, lab = bt_windows(meta, n_clip, seed, late_only)
        Z = np.zeros((len(wins), NC, 256, D), np.float16); CEN = np.zeros((len(wins), NB, 2), np.float32)
        VIS = np.zeros((len(wins), NB), np.float32); Q = np.zeros((len(wins), NC, 256), np.float32)
        t0 = time.time()
        for i0 in range(0, len(wins), 8):
            batch = []
            for j, (cid, idx, s, _) in enumerate(wins[i0:i0 + 8]):
                fr = torch.from_numpy(decord.VideoReader(f"{root}/{cid}.mp4", num_threads=2).get_batch(list(map(int, idx))).asnumpy())
                batch.append((fr.permute(3, 0, 1, 2).float() / 255.0 - MEAN) / STD)
                CEN[i0 + j], VIS[i0 + j], Q[i0 + j] = bt_labels(lab, cid, idx)
            Z[i0:i0 + len(batch)] = enc(torch.stack(batch))
        np.savez(OUT / f"{name}.npz", z=Z, cen=CEN, vis=VIS, q=Q, clip=np.array([w[0] for w in wins]),
                 stride=np.array([w[2] for w in wins]), late=np.array([w[3] for w in wins]))
        print(f"[{name}] {len(wins)} 창 {time.time() - t0:.0f}s", flush=True)

    for name, st, sd, lab in VOB_SPEEDS:
        ds, sets = r4.SETS[name]
        cfg, _ = resolve(ds, "vith", sets, None, 0)
        W = te.WMADataset(cfg)
        blocks = ep.pick_blocks(W, name)
        idx = st + sd * np.arange(N)
        Z = np.zeros((len(blocks), NC, 256, D), np.float16); CEN = np.zeros((len(blocks), NB, 2), np.float32)
        VIS = np.zeros((len(blocks), NB), np.float32); vids = []
        for i0 in range(0, len(blocks), 8):
            bl = blocks[i0:i0 + 8]
            Z[i0:i0 + len(bl)] = enc(torch.stack([W.clip(b["a"]) for b in bl]))
            for j, b in enumerate(bl):
                rec = W.records[b["a"]]; vids.append(rec.video_id)
                L = json.load(open(VOB_SRC / str(Path(rec.raw["file_name"]).parent) / "label.json"))["labels"]["A"]
                byf = {int(x["f"]): x for x in L}
                u = np.array([byf[int(f)]["u"] for f in idx]) / 32.0 - 0.5          # 512 px → 토큰
                v = np.array([byf[int(f)]["v"] for f in idx]) / 32.0 - 0.5
                hd = np.array([byf[int(f)].get("hidden", 0.0) for f in idx])
                CEN[i0 + j] = np.stack([u, v], -1).reshape(NB, 2, 2).mean(1)
                VIS[i0 + j] = (1 - hd).reshape(NB, 2).mean(1)
        np.savez(OUT / f"vob_{name}.npz", z=Z, cen=CEN, vis=VIS, vid=np.array(vids))
        print(f"[vob_{name}] {len(blocks)} block", flush=True)


# ----------------------------------------------------------------------------- w 학습 · 평가 (GPU 있으면 GPU)
def fit_w(dev):
    d = np.load(OUT / "bt_train.npz")
    vis, q = d["vis"][:, :NC], d["q"]
    sel = np.argwhere(vis >= 0.5)                                         # 보이는 문맥 블록만
    Z = torch.from_numpy(d["z"][sel[:, 0], sel[:, 1]]).to(dev)            # (M, 256, D) fp16
    Qt = torch.from_numpy(q[sel[:, 0], sel[:, 1]]).to(dev)
    tok = (Qt >= 0.25).float()
    model = SC.ObjectScore(Z.shape[-1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    pw = ((1 - tok).sum() / tok.sum().clamp_min(1)).clamp(max=50)
    g = torch.Generator(device="cpu").manual_seed(0)
    for step in range(600):
        b = torch.randint(0, len(Z), (64,), generator=g).to(dev)
        a = model(Z[b])
        ce = -(Qt[b] / Qt[b].sum(-1, keepdim=True).clamp_min(1e-6) * a.log_softmax(-1)).sum(-1).mean()
        bce = torch.nn.functional.binary_cross_entropy_with_logits(a, tok[b], pos_weight=pw)
        loss = ce + bce
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0 or step == 599:
            print(f"  w step {step} ce {ce.item():.3f} bce {bce.item():.3f}", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), RES / "object_score_w.pt")
    return model


@torch.no_grad()
def scores(model, name, dev):
    d = np.load(OUT / f"{name}.npz")
    a = torch.cat([model(torch.from_numpy(d["z"][i:i + 16]).to(dev)).float().cpu() for i in range(0, len(d["z"]), 16)])
    return a, d


def auc(pos, neg):
    s = np.r_[pos, neg]; r = s.argsort().argsort() + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def evaluate(model, dev):
    res = {}
    a, d = scores(model, "bt_val", dev)
    x, m = SC.readout(a)
    vis = d["vis"]; cen = d["cen"]
    vctx = vis[:, :NC]
    mv, mh = m.numpy()[vctx >= 0.5], m.numpy()[(vctx >= 0) & (vctx < 0.1)]
    # τ: 보임 (≥ 0.5) 과 가림 (< 0.1) 을 가장 잘 가르는 문턱 (Youden), κ = m 표준편차의 1/4
    grid = np.quantile(np.r_[mv, mh], np.linspace(0.02, 0.98, 97))
    tau = float(grid[np.argmax([(mv > t).mean() - (mh > t).mean() for t in grid])])
    kappa = float(np.std(np.r_[mv, mh]) / 4)
    err_vis = np.linalg.norm(x.numpy() - cen[:, :NC], axis=-1)[vctx >= 0.5]
    res["read"] = {"bt_val_visible_err_tok_median": round(float(np.median(err_vis)), 3), "bt_val_visible_err_tok_p90": round(float(np.quantile(err_vis, 0.9)), 3),
                   "m_auc_visible_vs_hidden": round(auc(mv, mh), 3), "tau": round(tau, 3), "kappa": round(kappa, 3),
                   "n_visible_blocks": int(len(mv)), "n_hidden_blocks": int(len(mh))}
    tf = torch.arange(NC, NB)
    variants = [(mo, o) for mo in ("visible", "weighted") for o in (1, 2)]
    res["roll_bt_val"] = {}
    for mo, o in variants:
        xf, _, _, _ = SC.calc(a, tau, tf, mo, o, kappa=kappa)
        on = (vis[:, NC:] >= 0) & np.isfinite(cen[:, NC:, 0])
        e = np.linalg.norm(xf.numpy() - cen[:, NC:], axis=-1)
        stay = np.linalg.norm(x.numpy()[np.arange(len(x)), (m.numpy() > tau).cumsum(1).argmax(1)][:, None] - cen[:, NC:], axis=-1)
        res["roll_bt_val"][f"{mo}_order{o}"] = {"future_err_tok_median": round(float(np.median(e[on])), 3),
                                               "last_block_err_tok_median": round(float(np.median(e[:, -1][on[:, -1]])), 3),
                                               "stay_baseline_err_tok_median": round(float(np.median(stay[on])), 3)}
    # 속도 sweep (VOBench)
    col = {}
    for name, _, _, lab in VOB_SPEEDS:
        a2, d2 = scores(model, f"vob_{name}", dev)
        col[(name, "true")] = d2["cen"][:, -1, 0]
        x2, m2 = SC.readout(a2)
        col[(name, "readerr")] = np.median(np.linalg.norm(x2.numpy() - d2["cen"][:, :NC], axis=-1)[d2["vis"][:, :NC] >= 0.5])
        for mo, o in variants:
            xf, _, _, _ = SC.calc(a2, tau, tf, mo, o, kappa=kappa)
            col[(name, f"{mo}_order{o}")] = xf[:, -1, 0].numpy()
    t1, t3 = col[("S1", "true")], col[("orig", "true")]
    keep = np.abs(t1 - t3) >= 1.5
    rng = np.random.default_rng(0); B = rng.integers(0, keep.sum(), (2000, keep.sum()))
    res["speed_sweep_vob"] = {"n_block_used": int(keep.sum()), "true_col_median": {lab: round(float(np.median(col[(nm, "true")][keep])), 2) for nm, _, _, lab in VOB_SPEEDS},
                              "read_err_visible_tok_median": {lab: round(float(col[(nm, "readerr")]), 3) for nm, _, _, lab in VOB_SPEEDS}, "variants": {}}
    for mo, o in variants:
        k = f"{mo}_order{o}"
        r = (col[("S1", k)] - col[("orig", k)])[keep] / (t1 - t3)[keep]
        bs = np.median(r[B], 1)
        res["speed_sweep_vob"]["variants"][k] = {
            "pred_col_median": {lab: round(float(np.median(col[(nm, k)][keep])), 2) for nm, _, _, lab in VOB_SPEEDS},
            "within1col": {lab: round(float((np.abs(col[(nm, k)] - col[(nm, "true")]) <= 1)[keep].mean()), 3) for nm, _, _, lab in VOB_SPEEDS},
            "speed_follow_ratio": {"median": round(float(np.median(r)), 3), "ci": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]}}
    ref = REPO / "z_analysis/transfer_gap/results/r4_x2/speed_sweep.json"
    if ref.exists():
        res["speed_sweep_vob"]["reference_vobench_pft_follow"] = json.load(open(ref))["result"]["speed_follow_ratio"]["vobpft"]
    RES.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(RES / "stage0.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


def main():
    part = sys.argv[1] if len(sys.argv) > 1 else "eval"
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if part == "run":
        extract()
    model = fit_w(dev)
    evaluate(model, dev)


if __name__ == "__main__":
    main()
