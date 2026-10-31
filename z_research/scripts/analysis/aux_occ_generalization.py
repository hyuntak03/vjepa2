#!/usr/bin/env python3
"""보조 모듈 칸 예측의 일반화 — 학습 안 한 VOBench 에서도 물체가 미래 어느 칸에 있는지 (가려져 있어도) 맞히나 (2026-10-31).

배경: release_aux_perm_sup = 고정 릴리즈 predictor + 보조 기억 모듈, 모듈 기억 m 위 선형 head 로 미래 칸 (배경 / 보이는 물체 / 가려진 물체) 을
  phys3d_perm_v1 라벨로 감독 (사용자: "모듈한테 context 보고 물체 놓아져야될 칸 supervision 주면서 일반화 되도록").
여기서는 head 출력만 본다 (predictor 출력 · 채점은 따로). 문맥 C = 16 장 → 미래 8 블록 (블록 = 연속 2 장, 16×16 칸).
데이터:
  perm    phys3d_perm_v1 val 876 clip (학습 생성기, 학습에 안 쓴 영상) — 라벨 = occ_val.npy (z_training/data/build_perm_occupancy.py, C=16)
  vobench VOBench_v1 late (처음 보는 생성기) 의 가능 변이 중 물체가 있는 팔 — 학습 중 val 과 같은 프레임 (30 fps 의 3 장 간격 = 10 fps, 32 장).
          라벨 = label.json (u, v ± ext, hidden) 을 같은 규칙으로: 블록에서 한 장이라도 보이면 보임, 두 장 다 가려지고 문맥에서 봤으면 가려짐
점수 (미래 칸 전부, 무시 칸 제외):
  물체 칸 재현율 = 라벨이 물체 (보임 ∪ 가려짐) 인 칸 중 head 가 물체 (1 ∪ 2) 라고 한 비율 · 정밀도 · IoU
  가려진 칸 재현율 = 라벨 2 칸 중 head 가 물체 (1 ∪ 2) 라고 한 비율 (= 안 보여도 거기 있다고 하나) · 그중 2 로 맞힌 비율
  기준선 "문맥 끝 그대로" = 문맥 마지막 블록의 물체 칸 (라벨 1 ∪ 2) 을 미래 전부에 복사 — 정지 물체는 이것으로 다 맞는다. 움직이는 물체에서 모듈이 이걸 넘는지가 요점
  (기준선은 문맥 마지막 블록의 가려진 물체 자리까지 라벨에서 가져오므로 관측보다 유리하다)
이동량 head 가 있으면 (release_aux_perm_rel, 2026-10-31): 문맥에서 마지막으로 보인 장의 물체 칸 (출발 칸) 에서 블록 t 까지의 이동 (dx, dy, 칸) 오차.
  EPE = 예측과 정답 이동 벡터 사이 거리 (칸). 기준선 = 0 이동 (= 문맥 끝 그대로). 방향 cos = 정답 이동 ≥ 1 칸인 출발 칸에서.

  CUDA_VISIBLE_DEVICES=0 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/aux_occ_generalization.py --data vobench [--ckpt latest.pt]
출력: z_analysis/results/aux_occ_generalization/<run>_<ckpt>_<data>.json
"""
import argparse, collections, csv, json, os, sys
import numpy as np, torch, torch.nn.functional as F, yaml
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
ap = argparse.ArgumentParser()
ap.add_argument("--run", default="release_aux_perm_sup")
ap.add_argument("--ckpt", default="latest.pt")
ap.add_argument("--data", required=True, choices=["perm", "vobench"])
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--mem", type=float, default=0.25)
a = ap.parse_args()
OUT = os.path.join(REPO, "z_analysis/results/aux_occ_generalization"); os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(a.mem)
from app.vjepa_frozen import utils as U
from app.vjepa_frozen.aux_memory import AuxRelease

RD = os.path.join(REPO, "z_training/runs", a.run)
cfg = yaml.safe_load(open(os.path.join(RD, "config.yaml"))); MD = cfg["model"]
NF, G, S, TUB, C = 32, 16, 256, 2, 16
Cb = C // TUB
state = U.load_base_checkpoint(MD["checkpoint"])
ctx_enc, _ = U.build_frozen_encoders(MD, state, dev, torch.bfloat16, NF)
model = AuxRelease(U.build_predictor(MD, ctx_enc.embed_dim, state, dev, NF), dict(MD["aux"]), ctx_enc.embed_dim, 256, 16, 2, NF).to(dev)
ck = torch.load(os.path.join(RD, a.ckpt), map_location="cpu", weights_only=False)
print(model.load_state_dict(ck["predictor"], strict=True), "epoch", ck.get("epoch"), flush=True)
model.eval(); del state, ck
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)


def cells(x0, y0, x1, y1):
    if x1 < 0 or y1 < 0 or x0 > 256 or y0 > 256: return []
    return [r * G + c for r in range(max(0, int(y0 // 16)), min(G, int(y1 // 16) + 1)) for c in range(max(0, int(x0 // 16)), min(G, int(x1 // 16) + 1))]


def items():
    """yield (이름, 그룹 dict, frames uint8 (32, 256, 256, 3), 라벨 (16 블록, 256) int8)."""
    if a.data == "perm":
        import decord
        L = np.load(os.path.join(REPO, "data_csv/phys3d_perm_v1/occ_val.npy"))
        Dp = np.load(os.path.join(REPO, "data_csv/phys3d_perm_v1/disp_val.npy"))
        paths = [ln.split()[0] for ln in open(os.path.join(REPO, "data_csv/phys3d_perm_v1/val_occ.csv"))]
        for i, p in enumerate(paths):
            st = np.load(p[:-4] + "_states.npz")
            sp = np.hypot(*(st["pos"][C - 1] - st["pos"][C - 3]).T) / 2       # 문맥 끝 속력 (px/장), 물체별
            dd = Dp[i, 3].astype(np.float32) / 4.0; dd[Dp[i, 3, ..., 0] == -128] = np.nan
            yield os.path.basename(p), {"scen": p.split("/")[-2], "moving": bool((sp > 1.0).any())}, \
                decord.VideoReader(p, num_threads=2).get_batch(list(range(NF))).asnumpy(), L[i, 3], dd
        return
    from PIL import Image
    R = "/workspace/dataset/benchmarks/VOBench_v1"
    rows = [r for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1_late/index.csv")))
            if r["variant"] in ("pos_a", "pos_b") and (r["obj_a"] if r["variant"] == "pos_a" else r["obj_b"]) != "empty"]
    for r in rows:
        arm = "A" if r["variant"] == "pos_a" else "B"
        Lb = json.load(open(os.path.join(R, os.path.dirname(r["file_name"]), "label.json")))["labels"][arm]
        idx = [3 * j for j in range(NF)]
        fr = np.stack([np.asarray(Image.open(os.path.join(R, r["file_name"], f"{f:06d}.png")).convert("RGB")) for f in idx])
        if fr.shape[1] != 256:
            fr = F.interpolate(torch.from_numpy(fr).permute(0, 3, 1, 2).float(), size=(256, 256), mode="bilinear",
                               align_corners=False).round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
        sc = 256 / 512
        vis = np.array([Lb[f]["hidden"] < 0.5 for f in idx])
        box = [((Lb[f]["u"] - Lb[f].get("ext_u", Lb[f]["r_px"])) * sc, (Lb[f]["v"] - Lb[f].get("ext_v", Lb[f]["r_px"])) * sc,
                (Lb[f]["u"] + Lb[f].get("ext_u", Lb[f]["r_px"])) * sc, (Lb[f]["v"] + Lb[f].get("ext_v", Lb[f]["r_px"])) * sc) for f in idx]
        seen = vis[:C].any()
        lab = np.zeros((NF // TUB, S), np.int8)
        for t in range(NF // TUB):
            vf = [f for f in (2 * t, 2 * t + 1) if vis[f]]
            if vf:
                for f in vf: lab[t, cells(*box[f])] = 1 if seen else -1
            elif seen:
                for f in (2 * t, 2 * t + 1): lab[t, cells(*box[f])] = 2
        dd = np.full((NF // TUB, S, 2), np.nan, np.float32)                     # 이동량: 마지막으로 보인 문맥 장의 상자 칸 → 블록 t 둘째 장 중심
        fs = [f for f in range(C) if vis[f]]
        if fs:
            f0 = fs[-1]; c0 = np.array([(box[f0][0] + box[f0][2]) / 2, (box[f0][1] + box[f0][3]) / 2])
            for t in range(NF // TUB):
                c1 = np.array([(box[2 * t + 1][0] + box[2 * t + 1][2]) / 2, (box[2 * t + 1][1] + box[2 * t + 1][3]) / 2])
                dd[t, cells(*box[f0])] = (c1 - c0) / 16.0
        yield r["video_id"], {"cond": r["condition"].split("/")[-2], "objset": r["objset"], "k": r["k"], "motion": r["motion"],
                              "moving": r["motion"] != "flat_static"}, fr, lab, dd


rec = []
me = torch.arange(Cb * S, device=dev)[None]; mp_ = torch.arange(Cb * S, NF // TUB * S, device=dev)[None]


def flush(buf):
    x = torch.from_numpy(np.stack([b[2] for b in buf])).to(dev).permute(0, 4, 1, 2, 3).float().div_(255)
    x = ((x - MEAN) / STD).to(torch.bfloat16)
    B = x.size(0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        z = ctx_enc(x, masks=[me.expand(B, -1)])
        m = model.aux(z, me.expand(B, -1), mp_.expand(B, -1), mask_index=0)
        pr = model.occ_head(m).float().argmax(-1).view(B, -1, S).cpu().numpy()     # (B, 8, 256)
        pd = model.disp_head(m).float().view(B, -1, S, 2).cpu().numpy() if model.disp_head is not None else None
    for i, (nm, grp, _, lab, dd) in enumerate(buf):
        y = lab[Cb:]; base = np.broadcast_to(np.isin(lab[Cb - 1], (1, 2)), y.shape)
        ok = y >= 0
        obj_y = np.isin(y, (1, 2)) & ok; obj_p = np.isin(pr[i], (1, 2)) & ok; obj_b = base & ok
        hid = (y == 2)
        reap = np.zeros_like(y, bool)                                           # 재등장: 문맥 끝 블록에 안 보이던 물체 칸이 보이는 블록
        if not (lab[Cb - 1] == 1).any(): reap = (y == 1)
        rec.append(dict(name=nm, **grp, n_obj=int(obj_y.sum()),
                        tp=int((obj_y & obj_p).sum()), fp=int((~obj_y & obj_p).sum()), fn=int((obj_y & ~obj_p).sum()),
                        btp=int((obj_y & obj_b).sum()), bfp=int((~obj_y & obj_b).sum()), bfn=int((obj_y & ~obj_b).sum()),
                        n_hid=int(hid.sum()), hid_obj=int((hid & obj_p).sum()), hid_2=int((hid & (pr[i] == 2)).sum()), hid_b=int((hid & base).sum()),
                        n_reap=int(reap.sum()), reap_obj=int((reap & obj_p).sum()), reap_b=int((reap & base).sum()),
                        per_t=[[int((obj_y[t] & obj_p[t]).sum()), int(obj_y[t].sum()), int((obj_y[t] & obj_b[t]).sum())] for t in range(y.shape[0])]))
        if pd is not None:
            g = dd[Cb:]; ok = np.isfinite(g[..., 0])
            e_m = np.linalg.norm(pd[i] - g, axis=-1); e_0 = np.linalg.norm(g, axis=-1)
            big = ok & (e_0 >= 1.0)
            cs = (pd[i][big] * g[big]).sum(-1) / (np.linalg.norm(pd[i][big], axis=-1) * e_0[big] + 1e-9)
            rec[-1].update(d_n=int(ok.sum()), d_em=float(e_m[ok].sum()), d_e0=float(e_0[ok].sum()), d_nb=int(big.sum()), d_cos=float(cs.sum()),
                           d_t=[[float(e_m[t][ok[t]].sum()), float(e_0[t][ok[t]].sum()), int(ok[t].sum())] for t in range(g.shape[0])])


buf = []
for it in items():
    buf.append(it)
    if len(buf) == a.bs: flush(buf); buf = []
    if len(rec) % 200 == 0 and rec: print(len(rec), flush=True)
if buf: flush(buf)
json.dump(rec, open(os.path.join(OUT, f"{a.run}_{a.ckpt[:-3]}_{a.data}.json"), "w"))


def summ(rr):
    s = lambda k: sum(r[k] for r in rr)
    pct = lambda n, d: 100 * n / d if d else float("nan")
    return (f"모듈 R {pct(s('tp'), s('tp') + s('fn')):5.1f} P {pct(s('tp'), s('tp') + s('fp')):5.1f} IoU {pct(s('tp'), s('tp') + s('fp') + s('fn')):5.1f}"
            f" | 기준선 R {pct(s('btp'), s('btp') + s('bfn')):5.1f} P {pct(s('btp'), s('btp') + s('bfp')):5.1f} IoU {pct(s('btp'), s('btp') + s('bfp') + s('bfn')):5.1f}"
            f" | 가려진 칸 {s('n_hid'):6d}: 물체라 함 {pct(s('hid_obj'), s('n_hid')):5.1f} (2 로 {pct(s('hid_2'), s('n_hid')):5.1f}) 기준선 {pct(s('hid_b'), s('n_hid')):5.1f}"
            f" | 재등장 칸 {s('n_reap'):6d}: {pct(s('reap_obj'), s('n_reap')):5.1f} 기준선 {pct(s('reap_b'), s('n_reap')):5.1f}")


print(f"\n{a.run}/{a.ckpt} · {a.data} · {len(rec)} clip · 문맥 16 장 → 미래 8 블록 · 물체 칸 (보임 ∪ 가려짐) 재현율 R · 정밀도 P · IoU (%)")
print(f"  {'전체':16s} {summ(rec)}")
for key in (["moving"] + (["cond", "k", "motion", "objset"] if a.data == "vobench" else [])):
    for v in sorted({str(r[key]) for r in rec}):
        print(f"  {key + '=' + v:16s} {summ([r for r in rec if str(r[key]) == v])}")
if "d_n" in rec[0]:
    def dsum(rr):
        n = sum(r["d_n"] for r in rr); nb = sum(r["d_nb"] for r in rr)
        return (f"이동 EPE 모듈 {sum(r['d_em'] for r in rr) / max(1, n):.2f} 칸 · 0 이동 {sum(r['d_e0'] for r in rr) / max(1, n):.2f} 칸 · "
                f"방향 cos {sum(r['d_cos'] for r in rr) / max(1, nb):.3f} (출발 칸 {n}, 1 칸 이상 움직임 {nb})")
    print("\n이동량 head (문맥 끝 출발 칸 → 블록 t)")
    print(f"  {'전체':16s} {dsum(rec)}")
    for key in (["moving"] + (["motion", "k"] if a.data == "vobench" else [])):
        for v in sorted({str(r[key]) for r in rec}):
            print(f"  {key + '=' + v:16s} {dsum([r for r in rec if str(r[key]) == v])}")
    rr = [r for r in rec if r["moving"]]
    print("  블록별 EPE 모듈 / 0 이동 (움직이는 clip): " + "  ".join(
        f"t{t + 1} {sum(r['d_t'][t][0] for r in rr) / max(1, sum(r['d_t'][t][2] for r in rr)):.2f}/{sum(r['d_t'][t][1] for r in rr) / max(1, sum(r['d_t'][t][2] for r in rr)):.2f}" for t in range(8)))
print("\n미래 블록별 물체 칸 재현율 (모듈 / 기준선), 움직이는 clip")
rr = [r for r in rec if r["moving"]]
print("  " + "  ".join(f"t{t + 1} {100 * sum(r['per_t'][t][0] for r in rr) / max(1, sum(r['per_t'][t][1] for r in rr)):5.1f}/{100 * sum(r['per_t'][t][2] for r in rr) / max(1, sum(r['per_t'][t][1] for r in rr)):5.1f}" for t in range(8)))
