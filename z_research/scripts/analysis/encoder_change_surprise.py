"""predictor 없이 encoder 만으로 — 32 장을 한 번에 넣은 encoder 의 미래 부분 (17–32 장) 이 문맥에서 얼마나 달라졌나 (L1) 로 가능 · 불가능 가르기 (2026-10-27, GPU 1 장).
사용자: "encoder 가 만든 미래 부분 17–32 프레임 가지고 L1 distance 재면 어떰? (IntPhys1, 2 late 버전이랑 VOBench 모두 해 봐)".
점수 (영상 하나, 토큰은 affine 없는 LN, 블록 = 토큰 16 시간 칸 중 하나 (256 칸)):
  copy    = 블록 8–15 의 평균 |T_b − T_7|      (마지막 문맥 블록을 그대로 미래로 — 복사 예측)
  ctxmean = 블록 8–15 의 평균 |T_b − mean(T_0..7)|
  encoder 두 개: context encoder (`encoder`, 출력에 LN) · target encoder (`target_encoder`, LN — 채점 정답과 같은 공간). 둘 다 32 장 한 번에.
판정: 같은 쌍 (block_id, pair_id 가 같은 가능 1 · 불가능 1, 문맥이 같다) 에서 불가능 점수 > 가능 점수 → 맞음. 반반 50 %.
데이터: IntPhys1 late (intphys1_late_uni, 문맥이 위반 직전에 끝나게 고른 32 장, 360 편) · IntPhys2 위반 정렬 (intphys2_violation_uni, mp4 직독, 1,000 편) ·
  VOBench 전체 (vobench_v1, 2,880 편, 원본 0, 3, …, 93 장).
  CUDA_VISIBLE_DEVICES=0 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/encoder_change_surprise.py
출력: z_analysis/results/encoder_change_surprise.json · 영상별 점수 캐시 /workspace/auto_research_cache/encoder_change/<set>.json
"""
import argparse, collections, csv, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument("--sets", default="intphys1_late,intphys2_uni,vobench_all"); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
REPO = "/workspace/world_model/vjepa2"; sys.path.insert(0, REPO); dev = torch.device("cuda")
CA = "/workspace/auto_research_cache/encoder_change"; os.makedirs(CA, exist_ok=True)
from analysis.intphys2.model import build_from_config
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
MCFG = {"arch_name": "vit_huge", "checkpoint": CK, "img_size": 256, "patch_size": 16, "tubelet_size": 2, "use_rope": True, "uniform_power": False, "dual_encoder": True,
        "context_encoder_key": "encoder", "target_encoder_key": "target_encoder", "predictor": {"embed_dim": 384, "depth": 12, "num_heads": 12, "num_mask_tokens": 10},
        "dtype": "float32", "window_size": 32}
rel = build_from_config(MCFG, dev); ENC = {"context": rel.context_encoder.half().eval(), "target": rel.target_encoder.half().eval()}; del rel.predictor
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(3, 1, 1, 1); S, D = 256, 1280


def rows_of(name):
    if name == "intphys1_late":
        root = "/workspace/world_model/local_datasets/world/world_analysis/IntPhys1_late_uni"
        return [dict(r, _src=("png", os.path.join(root, r["video_id"]), list(range(32))), _grp=r["principle"] + "/" + r["group"]) for r in csv.DictReader(open(f"{REPO}/data_csv/intphys1_late_uni/index.csv"))]
    if name == "intphys2_uni":
        return [dict(r, _src=("mp4", os.path.join("/workspace/dataset/benchmarks/IntPhys2/Main", r["file"]), [int(t) for t in r["frames"].split()]), _grp=r["block_type"] + "/" + r["difficulty"])
                for r in csv.DictReader(open(f"{REPO}/data_csv/intphys2_violation_uni/index_mp4.csv"))]
    if name == "vobench_all":
        out = []
        for r in csv.DictReader(open(f"{REPO}/data_csv/vobench_v1/index.csv")):
            r = dict(r, _src=("png", os.path.join("/workspace/dataset/benchmarks/VOBench_v1", r["file_name"]), list(range(0, 96, 3))[:32]), _grp=r["occlusion"] + "/" + r["violation"])
            r["pair_id"] = "A" if r["variant"] in ("pos_a", "imp_ab") else "B"                 # 문맥 같은 쌍: (pos_a, imp_ab) · (pos_b, imp_ba)
            out.append(r)
        return out


def load(src):
    kind, path, idx = src
    if kind == "png":
        fr = np.stack([np.asarray(Image.open(os.path.join(path, f"{i:06d}.png")).convert("RGB")) for i in idx])
    else:
        import decord
        fr = decord.VideoReader(path, num_threads=2).get_batch(idx).asnumpy()
    x = torch.from_numpy(fr).to(dev).permute(3, 0, 1, 2).float() / 255.0
    x = F.interpolate(x.permute(1, 0, 2, 3), size=(256, 256), mode="bilinear", align_corners=False, antialias=False).permute(1, 0, 2, 3)
    return ((x - MEAN) / STD).unsqueeze(0)


R = {}
for name in a.sets.split(","):
    rows = rows_of(name); rows = rows[: a.limit] if a.limit else rows
    sc = {}; t0 = time.time()
    for n, r in enumerate(rows):
        try:
            xn = load(r["_src"])
        except Exception as e:
            print("  ! 읽기 실패", r.get("video_id"), e, flush=True); continue
        out = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for en, enc in ENC.items():
                T = F.layer_norm(enc(xn.half()).float(), (D,))[0].view(16, S, D)
                out[f"{en}_copy"] = float((T[8:] - T[7:8]).abs().mean()); out[f"{en}_ctxmean"] = float((T[8:] - T[:8].mean(0, keepdim=True)).abs().mean())
        sc[r["video_id"]] = out
        if n % 500 == 0:
            print(f"  {name} {n}/{len(rows)} · {(time.time() - t0) / (n + 1):.2f} s/편", flush=True)
    json.dump(sc, open(f"{CA}/{name}.json", "w"))
    pairs = collections.defaultdict(dict)
    for r in rows:
        if r["video_id"] in sc:
            pairs[(r["block_id"], r["pair_id"])]["imp" if r["plausible"] == "0" else "pos"] = r
    pairs = [p for p in pairs.values() if "pos" in p and "imp" in p]
    res = {"n_videos": len(sc), "n_pairs": len(pairs)}
    for key in ("context_copy", "context_ctxmean", "target_copy", "target_ctxmean"):
        ok = [sc[p["imp"]["video_id"]][key] > sc[p["pos"]["video_id"]][key] for p in pairs]
        by = collections.defaultdict(list)
        for p, o in zip(pairs, ok):
            by[p["pos"]["_grp"]].append(o)
        res[key] = {"pair_accuracy": round(float(np.mean(ok)), 4), "by_group": {g: [round(float(np.mean(v)), 3), len(v)] for g, v in sorted(by.items())}}
    R[name] = res
    print(name, {k: (v["pair_accuracy"] if isinstance(v, dict) else v) for k, v in res.items()}, flush=True)
json.dump(R, open(f"{REPO}/z_analysis/results/encoder_change_surprise.json", "w"), indent=1)
print("saved")
