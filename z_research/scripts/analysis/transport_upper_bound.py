#!/usr/bin/env python3
"""운반 상한 — 물체를 제자리로 옮겨 주면 다음 블록 latent 예측이 복사보다 얼마나 좋아지나 (2026-10-30, 사용자: "phys3d multi obj 도 ㄱㅊ을 거 같은데 일단 측정해봐").

목적: 자기지도 운반 모듈 (라벨 없이 latent 손실로 학습) 을 어느 데이터로 학습할지 정하기 전에, 그 데이터가 "운반" 에 손실 보상을 주는지 잰다.
  상한이 작으면 그 데이터로는 모듈이 운반을 배울 압력이 없다 (CLAUDE.md predictor 관문 3).

토큰: 블록 (연속 2 장) 마다 target encoder 를 따로 돌린 LN(h) (256 칸 × 1280). 시간: 블록 ≈ 0.5 초 (VOBench 0.4 초). 문맥 C = 4 블록 → 미래 K = 4 블록.
  phys3d_single_v1 (12 fps, 3 장 간격) · phys3d_multi_v1 (4 fps, 1 장 간격) · phys3d_flow_v1 (10 fps, 3 장 간격) · VOBench_v1 (30 fps, 6 장 간격) — 창 시작 0.
예측 셋 (미래 블록 t 마다):
  copy    = 마지막 문맥 블록 그대로
  oracle  = copy 에서, 물체마다 마지막으로 보인 문맥 블록의 물체 토큰 (상자 칸) 을 블록 t 의 실제 상자 자리로 옮겨 놓고 (상자 중심 기준 같은 상대 칸),
            copy 에 남은 옛 물체 칸 (블록 t 에 물체 없는 칸) 은 배경 (그 칸이 문맥에서 비었던 블록들의 중앙값) 으로. 라벨 (정답 위치) 사용 = 상한
  reverse = oracle 과 같은데 옮기는 방향만 반대 (같은 거리) — 위약. 0 근처 / 손해여야 한다
  vacate  = 옛 물체 칸만 배경으로 (놓지 않음) — 이득이 "지우기" 에서 오는지 "놓기" 에서 오는지 가른다
점수: L1 = mean |예측 − h_t| (블록 t 전체 256 칸) 과 물체 관련 칸 (블록 t 의 물체 칸 ∪ copy 의 옛 물체 칸) 에서. 이득 % = (copy − 예측) / copy.
사건 (물체 단위, 블록 t): 움직임 (마지막 보인 자리에서 ≥ 1 칸) · 재등장 (마지막 문맥 블록에서 안 보이다가 t 에서 보임) · 정지.

  CUDA_VISIBLE_DEVICES=0 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/transport_upper_bound.py --data multi --n 300
출력: z_analysis/results/transport_upper_bound/<data>.json
"""
import argparse, collections, csv, glob, json, os, sys
import numpy as np, torch, torch.nn.functional as F
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, choices=["single", "multi", "flow", "vobench", "perm"])
ap.add_argument("--n", type=int, default=300)
a = ap.parse_args()
OUT = os.path.join(REPO, "z_analysis/results/transport_upper_bound"); os.makedirs(OUT, exist_ok=True)
C, K, S, G = 4, 4, 256, 16
NF = 2 * (C + K)
STRIDE = {"single": 3, "multi": 1, "flow": 3, "vobench": 6, "perm": 1}[a.data]
# perm (2026-10-30, 다른 세션이 만든 phys3d_perm_v1, 4 fps 32 장): 창을 앵커 가림에 맞춘다 — 문맥 = C* 앞 8 장, 미래 = C* 부터 8 장 (boundary.tsv 의 C_star)
dev = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(0.15)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint
from app.vjepa_frozen import ar as AR
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
tgt = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(tgt, _load_checkpoint(CK)["target_encoder"], tag="t"); tgt = tgt.to(dev, torch.bfloat16).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)


def clips():
    """yield (frames uint8 (NF, H, W, 3), boxes (NF, O, 4) in 256 px or nan, vis (NF, O) bool)."""
    rng = np.random.default_rng(0)
    if a.data == "vobench":
        from PIL import Image
        R = "/workspace/dataset/benchmarks/VOBench_v1"
        rows = [r for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv"))) if r["plausible"] == "1"
                and (r["obj_a"] if r["variant"] == "pos_a" else r["obj_b"]) != "empty"]
        for i in rng.permutation(len(rows))[: a.n]:
            r = rows[i]; arm = "A" if r["variant"] == "pos_a" else "B"
            L = json.load(open(os.path.join(R, os.path.dirname(r["file_name"]), "label.json")))["labels"][arm]
            idx = [STRIDE * k for k in range(NF)]
            fr = np.stack([np.asarray(Image.open(os.path.join(R, r["file_name"], f"{j:06d}.png")).convert("RGB").resize((256, 256), Image.BILINEAR)) for j in idx])
            bx = np.array([[[ (L[j]["u"] - L[j].get("ext_u", L[j]["r_px"])) / 2, (L[j]["v"] - L[j].get("ext_v", L[j]["r_px"])) / 2,
                              (L[j]["u"] + L[j].get("ext_u", L[j]["r_px"])) / 2, (L[j]["v"] + L[j].get("ext_v", L[j]["r_px"])) / 2]] for j in idx], float)
            vis = np.array([[L[j]["hidden"] < 0.5] for j in idx])
            yield r["video_id"], fr, bx, vis
        return
    import decord
    root = {"single": "phys3d_single_v1", "multi": "phys3d_multi_v1", "flow": "phys3d_flow_v1", "perm": "phys3d_perm_v1"}[a.data]
    paths = sorted(glob.glob(f"/workspace/dataset/training_data/{root}/val/clips/*/*.mp4"))
    cstar = {}
    if a.data == "perm":
        for r in csv.DictReader(open(f"/workspace/dataset/training_data/{root}/val/boundary.tsv"), delimiter="\t"):
            cstar[r["path"]] = int(r["C_star"])
    paths = [paths[i] for i in rng.permutation(len(paths))]
    n = 0
    for p in paths:
        st = np.load(p[:-4] + "_states.npz")
        start = cstar.get(p, 2 * C) - 2 * C if a.data == "perm" else 0
        if a.data == "perm" and p not in cstar: continue
        if st["pos"].shape[0] < start + STRIDE * (NF - 1) + 1: continue
        idx = [start + STRIDE * k for k in range(NF)]
        vr = decord.VideoReader(p, num_threads=2); fr = vr.get_batch(idx).asnumpy()
        sc = 256 / fr.shape[2]
        if fr.shape[1] != 256 or fr.shape[2] != 256:
            fr = F.interpolate(torch.from_numpy(fr).permute(0, 3, 1, 2).float(), size=(256, 256), mode="bilinear", align_corners=False).permute(0, 2, 3, 1).byte().numpy()
        bx = st["bbox"][idx] * sc
        vis = (st["vis_frac"][idx] > 0.5) & (st["inframe_frac"][idx] > 0.5)
        yield os.path.basename(p), fr, bx, vis
        n += 1
        if n >= a.n: return


def cells_of(box):
    x0, y0, x1, y1 = box
    if not np.isfinite([x0, y0, x1, y1]).all(): return [], None
    rs = range(max(0, int(y0 // 16)), min(G, int(y1 // 16) + 1)); cs = range(max(0, int(x0 // 16)), min(G, int(x1 // 16) + 1))
    cl = [(r, c) for r in rs for c in cs]
    return cl, (((y0 + y1) / 2) // 16, ((x0 + x1) / 2) // 16)


rec = []
for vid, fr, bx, vis in clips():
    x = torch.from_numpy(fr).to(dev).permute(3, 0, 1, 2)[None].float().div_(255)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        h = AR.encode_blocks(tgt, ((x - MEAN) / STD).to(torch.bfloat16), 2).float()[0].view(C + K, S, -1)     # (블록, 256, D)
    O = bx.shape[1]
    bvis = vis[0::2] & vis[1::2]                                       # (블록, O): 두 장 모두 보임
    occ = [[set() for _ in range(O)] for _ in range(C + K)]; ctr = [[None] * O for _ in range(C + K)]
    for b in range(C + K):
        for o in range(O):
            if bvis[b, o]:
                cl, cc = cells_of(bx[2 * b + 1, o]); occ[b][o] = set(cl); ctr[b][o] = cc
    any_obj = [set().union(*occ[b]) if O else set() for b in range(C + K)]
    bg = h[C - 1].clone()                                               # 배경: 그 칸이 문맥에서 빈 블록들의 중앙값
    for r in range(G):
        for c in range(G):
            free = [b for b in range(C) if (r, c) not in any_obj[b]]
            if free: bg[r * G + c] = h[free, r * G + c].median(0).values
    last = {o: max([b for b in range(C) if bvis[b, o]], default=None) for o in range(O)}
    for t in range(C, C + K):
        preds = {"copy": h[C - 1].clone()}
        for name in ("oracle", "reverse", "vacate"):
            # 놓을 자리 (dst ← src): oracle = 실제 자리, reverse = 같은 거리 반대 방향 (위약), vacate = 놓지 않음
            place = []
            if name != "vacate":
                for o in range(O):
                    if last[o] is None or not bvis[t, o]: continue
                    sb = last[o]; (sr, sc_), (dr, dc) = ctr[sb][o], ctr[t][o]
                    for (r, c) in occ[t][o]:
                        src = (int(np.clip(r - dr + sr, 0, G - 1)), int(np.clip(c - dc + sc_, 0, G - 1)))
                        dst = (r, c) if name == "oracle" else (int(np.clip(r - 2 * (dr - sr), 0, G - 1)), int(np.clip(c - 2 * (dc - sc_), 0, G - 1)))
                        place.append((dst, sb, src))
            keep = {d for d, _, _ in place} if name != "vacate" else (set().union(*[occ[t][o] for o in range(O)]) if O else set())
            P = h[C - 1].clone()
            for (r, c) in any_obj[C - 1] - keep:                         # 옛 물체 칸 → 배경 (위약은 자기가 놓는 자리 기준)
                P[r * G + c] = bg[r * G + c]
            for dst, sb, src in place:
                P[dst[0] * G + dst[1]] = h[sb, src[0] * G + src[1]]
            preds[name] = P
        rel = sorted({r * G + c for (r, c) in (any_obj[C - 1] | (set().union(*[occ[t][o] for o in range(O)]) if O else set()))})
        L1 = {n_: float((P - h[t]).abs().mean()) for n_, P in preds.items()}
        L1r = {n_: float((P[rel] - h[t, rel]).abs().mean()) if rel else float("nan") for n_, P in preds.items()}
        ev = []
        for o in range(O):
            if not bvis[t, o] or last[o] is None: continue
            d = np.hypot(ctr[t][o][0] - ctr[last[o]][o][0], ctr[t][o][1] - ctr[last[o]][o][1])
            ev.append("재등장" if not bvis[C - 1, o] else ("움직임" if d >= 1 else "정지"))
        rec.append(dict(vid=vid, t=t - C + 1, L1=L1, L1r=L1r, ev=ev, n_rel=len(rel)))
    if len({r["vid"] for r in rec}) % 50 == 0: print(f"[{a.data}] {len({r['vid'] for r in rec})}", flush=True)

json.dump(rec, open(os.path.join(OUT, f"{a.data}.json"), "w"))
gain = lambda rr, n_, key: 100 * float(np.mean([(r[key]["copy"] - r[key][n_]) / r[key]["copy"] for r in rr if r[key]["copy"] == r[key]["copy"]])) if rr else float("nan")
print(f"\n{a.data} · 영상 {len({r['vid'] for r in rec})} · (블록, 걸음) {len(rec)} · 이득 % = (복사 − 예측) / 복사 L1  [블록 전체 / 물체 관련 칸]")
for gn, f in (("전체", lambda r: True), ("움직임 있는 블록", lambda r: "움직임" in r["ev"]), ("재등장 있는 블록", lambda r: "재등장" in r["ev"]),
              ("정지만", lambda r: r["ev"] and all(e == "정지" for e in r["ev"])), ("물체 사건 없음", lambda r: not r["ev"])):
    rr = [r for r in rec if f(r)]
    print(f"  {gn:14s} n {len(rr):5d} · " + " · ".join(f"{n_} {gain(rr, n_, 'L1'):+6.2f} / {gain(rr, n_, 'L1r'):+6.2f}" for n_ in ("oracle", "reverse", "vacate")))
for t in range(1, K + 1):
    rr = [r for r in rec if r["t"] == t]
    print(f"  걸음 {t}: oracle {gain(rr, 'oracle', 'L1'):+6.2f} / {gain(rr, 'oracle', 'L1r'):+6.2f} · reverse {gain(rr, 'reverse', 'L1'):+6.2f} / {gain(rr, 'reverse', 'L1r'):+6.2f}")
