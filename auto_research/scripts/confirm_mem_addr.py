#!/usr/bin/env python3
"""[P] 확인 세트 (리더 배정 실험 2, 학습 없음) — 고정 주 지표 "3D 상태 head 주소의 '본 적 있나'" 를 VOBench 끝 가림 · IntPhys 1 dev 에서 (auto_research, 2026-10-22 KST).

  # IntPhys 1 블록 토큰 캐시 (한 번): 100 장 중 1,3,…,99 (2 장 간격) → 블록 25, 256 bilinear (antialias 없음), 목표 encoder bf16 autocast → LN
  CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=8 /venv/vjepa/bin/python -W ignore auto_research/scripts/confirm_mem_addr.py --set ip1 --build
  # 채점 (VOBench 캐시는 mem_slot_eval_vob.py --build 가 만든 /workspace/auto_research_cache/vobench_late)
  CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=8 /venv/vjepa/bin/python -W ignore auto_research/scripts/confirm_mem_addr.py --set vob
  CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=8 /venv/vjepa/bin/python -W ignore auto_research/scripts/confirm_mem_addr.py --set ip1

지표 ([비판] crit_ip2_addr_ctrl.py 를 그대로): 영상마다 블록 t = B0+1 .. 끝 (B0 = 4) 평균, 칸별
  seen(t, c) = min_{t' ≤ t−2} mean_d |h[t](c) − h[t'](c)|,  s = mean_t Σ_c w(t, c) · seen(t, c)   (first 도 같게)
  주소 w: head3d (주 지표) · head2d · headrand (같은 seed · 같은 생성 순서) · uniform · motion · novel10 · center (crit 과 같은 정의)
        + VOBench 만 objlab = 라벨 물체 칸 (A · B 중심 ±1 칸의 합집합, 블록마다; 없으면 균등) — 오라클, 상한으로만.
쌍 · 분해: crit_ip2_pmi.decomp (scene = VOBench block / IntPhys 1 사중항, 짝 1 · 2).
  VOBench: 짝 1 = (pos_a, imp_ab), 짝 2 = (pos_b, imp_ba). IntPhys 1: index 의 pair_id (Garrido 쌍).
기준 AR 토큰 surprise (원 규약):
  VOBench = natural_ar_vw_state latest, 문맥 8 블록 → 굴린 예측 8 블록, 미래 L1 (화면 전체 · 물체 자리 = oracle_vob 의 oidx).
  IntPhys 1 = A.8 per_video_surprise['skip2_w16/avg'] (z_training/runs/natural_ar_vw_state/eval/intphys1_sliding__intphys1_dev_latest_w16).
출력: auto_research/results/confirm_mem_addr_{vob,ip1}.json."""
import argparse, collections, csv, json, os, random, sys, time
import numpy as np, torch, torch.nn.functional as F

REPO = "/workspace/world_model/vjepa2"
sys.path.insert(0, os.path.join(REPO, "auto_research/scripts")); sys.path.insert(0, REPO)
from crit_ip2_addr_ctrl import load_head  # noqa: E402
from crit_ip2_pmi import decomp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--set", choices=["vob", "ip1"], required=True)
ap.add_argument("--build", action="store_true")
ap.add_argument("--B0", type=int, default=4)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
torch.set_num_threads(8)
dev = torch.device("cuda")
S, G = 256, 16
IP1_ROOT = "/workspace/dataset/benchmarks/IntPhys1/dev"
IP1_CACHE = "/workspace/auto_research_cache/intphys1_dev"
VOB_CACHE = "/workspace/auto_research_cache/vobench_late"
VB = "/workspace/dataset/benchmarks/VOBench_v1"
OUT = os.path.join(REPO, "auto_research/results", f"confirm_mem_addr_{a.set}.json")

# ------------------------------------------------------------------ IntPhys 1 캐시
if a.build:
    assert a.set == "ip1"
    from PIL import Image
    from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint
    CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
    tenc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
    _load_state_dict(tenc, _load_checkpoint(CK)["target_encoder"], tag="t"); tenc = tenc.to(dev).eval()
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
    rows = list(csv.DictReader(open(os.path.join(REPO, "data_csv/intphys1_dev/index.csv"))))[: a.limit or None]
    FR = list(range(1, 101, 2))                                                   # 50 장 → 블록 25
    os.makedirs(IP1_CACHE, exist_ok=True)
    H = np.lib.format.open_memmap(os.path.join(IP1_CACHE, "h.npy"), mode="w+", dtype=np.float16, shape=(len(rows), len(FR) // 2, S, 1280))
    t0 = time.time()
    for i, r in enumerate(rows):
        d = os.path.join(IP1_ROOT, r["block"], r["quadruplet"], r["run"], "scene")
        x = torch.from_numpy(np.stack([np.asarray(Image.open(f"{d}/scene_{k:03d}.png").convert("RGB")) for k in FR])).permute(3, 0, 1, 2).float() / 255.0
        x = F.interpolate(x.transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False).transpose(0, 1)
        x = (x - MEAN) / STD
        nb = x.shape[1] // 2
        xb = x.reshape(3, nb, 2, 256, 256).permute(1, 0, 2, 3, 4).to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            h = torch.cat([tenc(xb[k:k + 8]) for k in range(0, nb, 8)], 0).float()
        H[i] = F.layer_norm(h, (1280,)).view(nb, S, -1).half().cpu().numpy()
        if i % 40 == 0:
            print(f"캐시 {i}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    H.flush(); json.dump(rows, open(os.path.join(IP1_CACHE, "meta.json"), "w"))
    print("캐시 끝", len(rows), flush=True)
    sys.exit(0)

# ------------------------------------------------------------------ 지표 (crit_ip2_addr_ctrl 그대로)
torch.manual_seed(0)
heads = {"head3d": load_head("natural_ar_vw_state_phys3d_single", dev), "head2d": load_head("natural_ar_vw_state_phys2d_single", dev),
         "headrand": load_head(None, dev)}
yy, xx = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
cen = torch.tensor(np.exp(-(((yy - 7.5) ** 2 + (xx - 7.5) ** 2) / (2 * 4.0 ** 2))).reshape(-1), dtype=torch.float32, device=dev); cen /= cen.sum()


@torch.no_grad()
def mem_scores(h, objcells=None):
    """h (NB, S, D) float32 cuda → {seen_<addr>, first_<addr>}. objcells: 블록마다 칸 목록 (VOBench 오라클 주소)."""
    NB = h.shape[0]
    seen = torch.stack([(h[t][None] - h[: t - 1]).abs().mean(-1).min(0).values if t > 2 else torch.zeros(S, device=dev) for t in range(NB)])
    first = (h - h[0:1]).abs().mean(-1)
    prev = torch.cat([torch.zeros(1, S, device=dev), (h[1:] - h[:-1]).abs().mean(-1)])
    ts = torch.arange(a.B0 + 1, NB, device=dev)
    W = {nm: hd(h.reshape(1, -1, 1280), S)["attn"][0, :, 0].float()[ts] for nm, hd in heads.items()}
    W["uniform"] = torch.full((len(ts), S), 1.0 / S, device=dev)
    W["motion"] = prev[ts] / prev[ts].sum(-1, keepdim=True)
    q = torch.quantile(seen[ts], 0.9, dim=-1, keepdim=True)
    nv = (seen[ts] >= q).float(); W["novel10"] = nv / nv.sum(-1, keepdim=True)
    W["center"] = cen[None].expand(len(ts), -1)
    if objcells is not None:
        w = torch.zeros(len(ts), S, device=dev)
        for i, t in enumerate(ts.tolist()):
            cs = objcells[t]
            if cs:
                w[i, cs] = 1.0 / len(cs)
            else:
                w[i] = 1.0 / S
        W["objlab"] = w
    out = {}
    for nm, w in W.items():
        out[f"seen_{nm}"] = float((w * seen[ts]).sum(-1).mean()); out[f"first_{nm}"] = float((w * first[ts]).sum(-1).mean())
    return out


rows = []
t0 = time.time()
if a.set == "vob":
    from src.models.rollout_predictor import vit_prefix_predictor
    from analysis.intphys2.model import _clean_backbone_key
    import yaml
    pc = yaml.safe_load(open(os.path.join(REPO, "z_training/runs/natural_ar_vw_state/config.yaml")))["model"]["predictor"]
    pred = vit_prefix_predictor(img_size=256, patch_size=16, tubelet_size=2, num_frames=64, embed_dim=1280, predictor_embed_dim=384, depth=12,
                                num_heads=12, num_mask_tokens=10, mask_mode="ar", ar_feedback=pc.get("ar_feedback", "output"))
    pred.load_state_dict({k: v for k, v in _clean_backbone_key(torch.load(os.path.join(REPO, "z_training/runs/natural_ar_vw_state/latest.pt"), map_location="cpu", weights_only=False)["predictor"]).items()
                          if not k.startswith(("hint_attn.", "dyn_head.", "recon_proj."))}); pred = pred.to(dev).eval()
    IDX = list(range(0, 96, 3))[:32]; C, K = 8, 8
    idx_rows = {r["block_id"]: {} for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv"))) if r["occlusion"] == "late"}
    for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv"))):
        if r["occlusion"] == "late":
            idx_rows[r["block_id"]][r["variant"]] = r
    meta = json.load(open(os.path.join(VOB_CACHE, "meta.json")))
    H = np.load(os.path.join(VOB_CACHE, "h.npy"), mmap_mode="r")
    ARMS = ["pos_a", "pos_b", "imp_ab", "imp_ba"]

    def centers(label_path, W0):                                                   # oracle_vob 과 같음: 블록 0..15 의 A · B 중심 칸
        L_ = json.load(open(label_path))["labels"]; out = {}
        for key in ("A", "B"):
            lab = {q["f"]: q for q in L_.get(key, [])}; cs = []
            for b in range(16):
                q = [lab.get(IDX[2 * b]), lab.get(IDX[2 * b + 1])]
                if any(t is None for t in q):
                    cs.append(None); continue
                r0 = int(np.clip(np.mean([t["v"] for t in q]) / W0 * G, 0, G - 1)); c0 = int(np.clip(np.mean([t["u"] for t in q]) / W0 * G, 0, G - 1))
                cs.append((r0, c0))
            out[key] = cs
        return out

    def region(rc, rad=1):
        r0, c0 = rc
        return [rr * G + cc for rr in range(max(0, r0 - rad), min(G, r0 + rad + 1)) for cc in range(max(0, c0 - rad), min(G, c0 + rad + 1))]

    from PIL import Image
    for bi, m in enumerate(meta[: a.limit or None]):
        arms = idx_rows[m["block_id"]]
        d0 = os.path.join(VB, arms["pos_a"]["file_name"])
        W0 = Image.open(f"{d0}/{IDX[0]:06d}.png").size[0]
        cn = centers(os.path.join(os.path.dirname(d0), "label.json"), W0)
        objcells = []
        for b in range(16):
            s_ = set()
            for key in ("A", "B"):
                if cn[key][b] is not None:
                    s_ |= set(region(cn[key][b]))
            objcells.append(sorted(s_))
        oidx = torch.tensor([j * S + t for j in range(K) for t in objcells[C + j]], device=dev)
        hh = {v: torch.from_numpy(np.asarray(H[m["i"], k], dtype=np.float32)).to(dev) for k, v in enumerate(ARMS)}
        mv = "static" if arms["pos_a"]["motion"] == "flat_static" else "moving"
        for pid, (pos, imp) in (("1", ("pos_a", "imp_ab")), ("2", ("pos_b", "imp_ba"))):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                p_ar = pred.rollout_kv(hh[pos][:C].reshape(1, C * S, -1), torch.arange(C * S, device=dev).unsqueeze(0), K).float()[0].view(K * S, -1)
            for v, imp_flag in ((pos, False), (imp, True)):
                e = (p_ar - hh[v][C:].reshape(K * S, -1)).abs().mean(-1)
                r = dict(scene=m["block_id"], pair=pid, imp=imp_flag, cond=mv, viol=arms[pos]["violation"], k=int(arms[pos]["k"]), cam="Fixed",
                         tok_full=float(e.mean()), tok_obj=float(e[oidx].mean()) if len(oidx) else float("nan"))
                r.update(mem_scores(hh[v], objcells))
                rows.append(r)
        if bi % 20 == 0:
            print(f"{bi}/{len(meta)} ({time.time() - t0:.0f}s)", flush=True)
    groups = {"all": lambda r: True, "moving": lambda r: r["cond"] == "moving", "static": lambda r: r["cond"] == "static"}
    for mv in ("moving", "static"):
        for vi in ("permanence", "shape", "color"):
            groups[f"{mv}|{vi}"] = (lambda mv_, vi_: (lambda r: r["cond"] == mv_ and r["viol"] == vi_))(mv, vi)
        for k in (1, 2, 3):
            groups[f"{mv}|k{k}"] = (lambda mv_, k_: (lambda r: r["cond"] == mv_ and r["k"] == k_))(mv, k)
    keys = ["seen_head3d", "seen_uniform", "seen_headrand", "seen_head2d", "seen_center", "seen_motion", "seen_novel10", "seen_objlab",
            "first_head3d", "first_objlab", "tok_obj", "tok_full"]
else:
    meta = json.load(open(os.path.join(IP1_CACHE, "meta.json")))
    H = np.load(os.path.join(IP1_CACHE, "h.npy"), mmap_mode="r")
    pv = json.load(open(os.path.join(REPO, "z_training/runs/natural_ar_vw_state/eval/intphys1_sliding__intphys1_dev_latest_w16/per_block.json")))["per_video_surprise"]
    for i, m in enumerate(meta[: a.limit or None]):
        h = torch.from_numpy(np.asarray(H[i], dtype=np.float32)).to(dev)
        r = dict(scene=f"{m['block']}_{m['quadruplet']}", pair=m["pair_id"], imp=m["plausible"] == "0", cond=m["block"], cam="Fixed",
                 tok_A8=float(pv[m["video_id"]]["skip2_w16/avg"]))
        r.update(mem_scores(h))
        rows.append(r)
        if i % 60 == 0:
            print(f"{i}/{len(meta)} ({time.time() - t0:.0f}s)", flush=True)
    groups = {"all": lambda r: True, "O1": lambda r: r["cond"] == "O1", "O2": lambda r: r["cond"] == "O2", "O3": lambda r: r["cond"] == "O3"}
    keys = ["seen_head3d", "seen_uniform", "seen_headrand", "seen_head2d", "seen_center", "seen_motion", "seen_novel10", "first_head3d", "tok_A8"]

rng = random.Random(0)
res = {}
for g, fn in groups.items():
    xs = [r for r in rows if fn(r)]
    if len({r["scene"] for r in xs}) < 3:
        continue
    res[g] = {key: decomp(xs, key, False, rng) for key in keys}
    for key in keys:
        st = res[g][key]
        print(f"{g:18s} {key:15s} n{st['n']:4d} acc {st['acc']:5.1f} [{st['acc_ci'][0]:5.1f},{st['acc_ci'][1]:5.1f}] 짝1/2 {st['pair1']:5.1f}/{st['pair2']:5.1f} "
              f"o {st['outcome_share']:.2f} i {st['mean_i']:+.5f} [{st['mean_i_ci'][0]:+.5f},{st['mean_i_ci'][1]:+.5f}]", flush=True)
# 3D head − 대조 주소의 짝 대응 차 (장면 bootstrap, crit 과 같은 방식: 쌍 정답 여부의 차)
def paired_diff(xs, k1, k2, nb=4000):
    sc = collections.defaultdict(dict)
    for r in xs:
        sc[r["scene"]][r["pair"] + ("I" if r["imp"] else "P")] = r
    ds = []
    for s, v in sc.items():
        if len(v) != 4:
            continue
        ds.append(np.mean([(v[p + "I"][k1] > v[p + "P"][k1]) - (v[p + "I"][k2] > v[p + "P"][k2]) for p in ("1", "2")]) * 100)
    ds = np.array(ds); bs = np.sort([ds[np.random.default_rng(i).integers(0, len(ds), len(ds))].mean() for i in range(nb)])
    return dict(diff=float(ds.mean()), ci=(float(bs[int(0.025 * nb)]), float(bs[int(0.975 * nb)])), n=len(ds))
res["paired_vs_head3d"] = {}
for g in [g for g in ("all", "moving", "static") if g in res]:
    xs = [r for r in rows if groups[g](r)]
    res["paired_vs_head3d"][g] = {c: paired_diff(xs, "seen_head3d", c) for c in ("seen_uniform", "seen_headrand", "seen_center")}
    print(g, {c: (round(v["diff"], 1), tuple(round(x, 1) for x in v["ci"])) for c, v in res["paired_vs_head3d"][g].items()}, flush=True)
json.dump(dict(summary=res, rows=rows, B0=a.B0), open(OUT, "w"), indent=1)
print("저장", OUT, f"({time.time() - t0:.0f}s)")
