#!/usr/bin/env python3
"""[물리] 실험 1 채점 — VOBench 끝 가림: 학습한 기억 슬롯으로 두 surprise (학습 없음).

  # 1) 공유 토큰 캐시 (한 번, ≈ 15–25 분, GPU 한 장 · 목표 encoder ViT-H bf16 ≈ 3 GB) — 다른 팀원도 쓸 수 있다
  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 /venv/vjepa/bin/python -W ignore auto_research/scripts/mem_slot_eval_vob.py --build
  # 2) 채점
  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 /venv/vjepa/bin/python -W ignore auto_research/scripts/mem_slot_eval_vob.py --ckpts A P

캐시: /workspace/auto_research_cache/vobench_late/{h.npy, meta.json}. h (216, 4, 16, 256, 1280) fp16 = 끝 가림 block 마다 팔 (pos_a, pos_b, imp_ab, imp_ba) ×
  블록 16 (장 0,3,…,93 → 2 장씩), LN(목표 encoder). 불가능 팔의 문맥 블록 0–7 은 짝 가능 팔 (imp_ab ← pos_a, imp_ba ← pos_b) 과 같다 ([P] oracle_vob 과 같은 구성).
쌍: (pos_a, imp_ab) · (pos_b, imp_ba). 정답 = surprise(불가능) > surprise(가능). 정지 (flat_static) 쌍 72 · 움직임 쌍 360. 위반 종류별 (영속 · 모양 · 색).
surprise (영상 = 그 팔의 블록 16, 기억 슬롯을 읽기 모드로 돌림; 미래 블록 t = 8..15):
  main (IntPhys 2 주 지표와 같은 식): Σ_c w(t, c) · seen(t, c), w = 존재 가중 슬롯 주소, seen = min_{t' ≤ t−2} |h[t](c) − h[t'](c)|
  e1   ([물리] 라운드 3 E1 기억 surprise, VOBench 판정용으로 결과 전 등록): 물체 슬롯 k* = 문맥 블록 0–7 의 존재 확률 평균이 가장 큰 슬롯.
       기억 내용 m = 문맥에서 k* 의 보임 확률 > 0.5 인 마지막 블록의 사후 위치 3×3 칸 토큰 평균.
       미래 블록마다 사전 보임 확률 > 0.5 이면 d_t = mean_d |(사전 위치 3×3 칸 관측 토큰 평균) − m|, 그 평균 (그런 블록이 없으면 미래 전부의 평균).
"""
import argparse, collections, csv, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F

REPO = "/workspace/world_model/vjepa2"
sys.path.insert(0, os.path.join(REPO, "auto_research/scripts")); sys.path.insert(0, REPO)
from mem_slot import MemSlot, OUT  # noqa: E402

VB = "/workspace/dataset/benchmarks/VOBench_v1"
CACHE = "/workspace/auto_research_cache/vobench_late"
IDX = list(range(0, 96, 3))[:32]
ARMS = ["pos_a", "pos_b", "imp_ab", "imp_ba"]
S, G = 256, 16


def build(limit, dev, out):
    from PIL import Image
    from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint
    CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
    tenc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
    _load_state_dict(tenc, _load_checkpoint(CK)["target_encoder"], tag="t"); tenc = tenc.to(dev).eval()
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

    def frames(d, idx):
        fr = [np.asarray(Image.open(f"{d}/{i:06d}.png").convert("RGB")) for i in idx]
        x = torch.from_numpy(np.stack(fr)).permute(3, 0, 1, 2).float() / 255.0
        x = F.interpolate(x.transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False).transpose(0, 1)
        return (x - MEAN) / STD

    @torch.no_grad()
    def blocks(x):
        nb = x.shape[1] // 2
        xb = x.reshape(3, nb, 2, 256, 256).permute(1, 0, 2, 3, 4).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = torch.cat([tenc(xb[i:i + 8]) for i in range(0, nb, 8)], 0).float()
        return F.layer_norm(h, (1280,)).view(nb, S, -1)

    rows = [r for r in csv.DictReader(open(os.path.join(REPO, "data_csv/vobench_v1/index.csv"))) if r["occlusion"] == "late"]
    blk = collections.defaultdict(dict)
    for r in rows:
        blk[r["block_id"]][r["variant"]] = r
    items = sorted(blk.items())[: limit or None]
    os.makedirs(out, exist_ok=True)
    H = np.lib.format.open_memmap(os.path.join(out, "h.npy"), mode="w+", dtype=np.float16, shape=(len(items), 4, 16, S, 1280))
    meta, t0 = [], time.time()
    for bi, (b, arms) in enumerate(items):
        hp = {}
        for v in ("pos_a", "pos_b"):
            hp[v] = blocks(frames(os.path.join(VB, arms[v]["file_name"]), IDX))
        for v, src in (("imp_ab", "pos_a"), ("imp_ba", "pos_b")):
            hp[v] = torch.cat([hp[src][:8], blocks(frames(os.path.join(VB, arms[v]["file_name"]), IDX[16:]))], 0)
        H[bi] = torch.stack([hp[v] for v in ARMS]).half().cpu().numpy()
        r0 = arms["pos_a"]
        meta.append(dict(i=bi, block_id=b, motion=r0["motion"], violation=r0.get("violation", r0.get("violation_type")), objset=r0.get("objset"),
                         files={v: arms[v]["file_name"] for v in ARMS}))
        if (bi + 1) % 20 == 0:
            print(f"캐시 {bi + 1}/{len(items)} · {time.time() - t0:.0f} 초", flush=True)
    H.flush(); json.dump(meta, open(os.path.join(out, "meta.json"), "w"))
    print(f"캐시 끝 {len(items)} block · {time.time() - t0:.0f} 초", flush=True)


def load(name, dev):
    if name == "rand":
        torch.manual_seed(0); m = MemSlot()
    else:
        ck = torch.load(os.path.join(OUT, f"{name}.pt"), map_location="cpu", weights_only=False)
        m = MemSlot(K=ck["args"]["K"]); m.load_state_dict(ck["model"])
    return m.to(dev).eval()


def pool3(h_t, pos):
    """h_t (256, D), pos (x, y) 칸 → 3×3 칸 토큰 평균"""
    c, r = int(np.clip(np.floor(float(pos[0])), 0, G - 1)), int(np.clip(np.floor(float(pos[1])), 0, G - 1))
    idx = [rr * G + cc for rr in range(max(0, r - 1), min(G, r + 2)) for cc in range(max(0, c - 1), min(G, c + 2))]
    return h_t[idx].mean(0)


@torch.no_grad()
def score(md, h):
    """h (16, 256, D) → dict(main, e1, e1_fallback)"""
    pri, post, att = md(h[None])
    T = h.shape[0]
    seen = torch.stack([(h[t][None] - h[: t - 1]).abs().mean(-1).min(0).values if t > 2 else torch.zeros(S, device=h.device) for t in range(T)])
    e = torch.sigmoid(post["exist"][0]); w = (e[..., None] * att[0]).sum(1) / e.sum(1, keepdim=True)
    ts = list(range(8, T))
    main = float((w[ts] * seen[ts]).sum(-1).mean())
    k = int(e[:8].mean(0).argmax())
    vp = torch.sigmoid(post["vis"][0, :, k]).cpu().numpy(); vq = torch.sigmoid(pri["vis"][0, :, k]).cpu().numpy()
    pp = post["pos"][0, :, k].cpu().numpy(); pq = pri["pos"][0, :, k].cpu().numpy()
    tv = max([t for t in range(8) if vp[t] > 0.5], default=7)
    mem = pool3(h[tv], pp[tv])
    use = [t for t in ts if vq[t] > 0.5]
    d = [float((pool3(h[t], pq[t]) - mem).abs().mean()) for t in (use or ts)]
    return dict(main=main, e1=float(np.mean(d)), e1_fallback=float(not use), k=k, t_last_vis=tv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--ckpts", nargs="+", default=["A", "P"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.set_num_threads(8); dev = torch.device("cuda")
    if a.build:
        build(a.limit, dev, a.cache); return
    meta = json.load(open(os.path.join(a.cache, "meta.json"))); H = np.load(os.path.join(a.cache, "h.npy"), mmap_mode="r")
    models = {nm: load(nm, dev) for nm in a.ckpts + ["rand"]}
    res = collections.defaultdict(lambda: collections.defaultdict(list)); rows = []
    for m in meta:
        hh = torch.from_numpy(np.asarray(H[m["i"]], dtype=np.float32)).to(dev)
        sc = {nm: {v: score(md, hh[ai]) for ai, v in enumerate(ARMS)} for nm, md in models.items()}
        mv = "static" if m["motion"] == "flat_static" else "moving"
        for nm in models:
            for pos, imp in (("pos_a", "imp_ab"), ("pos_b", "imp_ba")):
                for key in ("main", "e1"):
                    ok = float(sc[nm][imp][key] > sc[nm][pos][key])
                    for grp in (mv, f"{mv}|{m['violation']}"):
                        res[f"{nm}|{key}"][grp].append(ok)
                res[f"{nm}|e1_fallback"][mv].append(sc[nm][pos]["e1_fallback"])
        rows.append(dict(meta=m, scores=sc))
    out = {k: {g: dict(acc=round(100 * float(np.mean(v)), 1), n=len(v)) for g, v in d.items()} for k, d in res.items()}
    for k in sorted(out):
        print(k, {g: f"{v['acc']} (n {v['n']})" for g, v in sorted(out[k].items()) if "|" not in g})
    p = os.path.join(REPO, f"auto_research/results/mem_slot_vob{a.tag}.json")
    json.dump(dict(summary=out, rows=rows), open(p, "w"), indent=1); print("저장", p)


if __name__ == "__main__":
    main()
