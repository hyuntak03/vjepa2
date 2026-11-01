#!/usr/bin/env python3
"""real_bench 에서 왜 안 넘어가나 — surprise 를 물체 칸 / 배경 칸으로 나눠 다시 잰다 (2026-11-01, 사용자: "real 로 왜 transfer 안될까 분석 좀 해봐").

후보 원인을 가른다:
  (1) 채점 희석 — 모델이 물체 칸에서는 맞게 예측하는데 256 칸 × 8 블록 평균에 묻힌다 → 물체 칸만 보면 맞히는가
  (2) 불확실 → "물체 없음" 이 L1 에 유리 — 가려진 시간이 길고 위치가 불확실하면 예측이 "없음" 쪽으로 기운다
      → 물체 칸에서 예측 p 가 "있음" 정답 (h_있음) 과 "없음" 정답 (h_없음) 중 어디에 가까운가, 가려진 시간과의 관계
  (3) 겉모습 차이 — (1)·(2) 로 설명이 안 되면
데이터: real_bench_quads (index · 32 장 PNG, z_research/scripts/data/build_real_bench_quads.py), 물체 칸 = SAM2 마스크 (masks.npz 'sam2', 원본 영상 장 번호)
  를 256 × 256 로 늘인 뒤 16 × 16 칸 평균 > 0.2. 미래 블록 t 의 물체 칸 = 미래 장 2t · 2t+1 의 합집합 (원본 = 계속 있음 영상의 물체).
  짝 A: possible_present (있음) vs impossible_vanish (없음) · 짝 B: possible_absent (없음) vs impossible_appear (있음). 두 짝 모두 같은 물체 칸.
채점 규약은 cvpr testbed 와 같게: 문맥 16 장 · 미래 16 장, context ← encoder, target ← target_encoder 32 장 LN, float32 + autocast fp16.

  CUDA_VISIBLE_DEVICES=0 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/realbench_object_cells.py
출력: z_analysis/results/realbench_object_cells/summary.json
"""
import argparse, csv, json, os, sys
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
ap = argparse.ArgumentParser()
ap.add_argument("--models", default="vith,permanence_wrop_B,permanence_wrop_C,permanence_wrop_D")
a = ap.parse_args()
OUT = os.path.join(REPO, "z_analysis/results/realbench_object_cells"); os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda")
from analysis.intphys2.model import _build_encoder, _build_predictor, _load_state_dict, _load_checkpoint, _clean_backbone_key
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
SRC = "/workspace/dataset/benchmarks/real_bench/rubric_office_quads"
FR = "/workspace/dataset/benchmarks/real_bench/rubric_office_quads_c16f16"
NF, C, S, G = 32, 16, 256, 16
state = _load_checkpoint(CK)
enc = {}
for key in ("encoder", "target_encoder"):
    e = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=NF, use_rope=True, uniform_power=False)
    _load_state_dict(e, state[key], tag=key); enc[key] = e.to(dev).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)


def predictor(name):
    p = _build_predictor(img_size=256, patch_size=16, tubelet_size=2, num_frames=NF, encoder_embed_dim=1280, predictor_embed_dim=384,
                         predictor_depth=12, predictor_num_heads=12, num_mask_tokens=10, use_rope=True, uniform_power=False)
    sd = state["predictor"] if name == "vith" else _clean_backbone_key(torch.load(os.path.join(REPO, "z_training/runs", name, "latest.pt"),
                                                                                map_location="cpu", weights_only=False)["predictor"])
    _load_state_dict(p, sd, tag=name)
    return p.to(dev).eval()


rows = list(csv.DictReader(open(os.path.join(REPO, "data_csv/real_bench_quads/index.csv"))))
vids = sorted({r["video"] for r in rows})
by = {(r["video"], r["quad_variant"]): r for r in rows}
info = {v: json.load(open(os.path.join(SRC, by[(v, "possible_present")]["file_name"].rsplit("/", 1)[0], "info.json"))) for v in vids}


def obj_cells(v):
    """(8 미래 블록, 256) bool — 계속 있음 영상의 SAM2 물체가 덮는 칸."""
    r = by[(v, "possible_present")]
    z = np.load(os.path.join(SRC, r["file_name"].rsplit("/", 1)[0], "masks.npz")); T, H, W = z["shape"]
    m = np.unpackbits(z["sam2"], axis=-1)[..., :W].astype(np.float32)
    fut = [int(x) for x in r["fut_frames"].split()]
    out = np.zeros((8, S), bool)
    for t in range(8):
        acc = np.zeros((G, G))
        for f in fut[2 * t:2 * t + 2]:
            mm = torch.from_numpy(m[f])[None, None]
            acc = np.maximum(acc, F.adaptive_avg_pool2d(F.interpolate(mm, size=(256, 256), mode="nearest"), (G, G))[0, 0].numpy())
        out[t] = (acc > 0.2).ravel()
    return out


def clip(r):
    fr = np.stack([np.asarray(Image.open(os.path.join(FR, r["file_name"], f"{j:06d}.png")).convert("RGB")) for j in range(NF)])
    x = torch.from_numpy(fr).to(dev).permute(3, 0, 1, 2)[None].float().div_(255)
    x = F.interpolate(x[0].transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False).transpose(0, 1)[None]
    return (x - MEAN) / STD


ci = torch.arange(C // 2 * S, device=dev)[None]; ti = torch.arange(C // 2 * S, NF // 2 * S, device=dev)[None]
H = {}                                                       # 정답 토큰과 문맥 토큰은 모델과 무관 → 한 번만
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    for r in rows:
        x = clip(r)
        h = F.layer_norm(enc["target_encoder"](x).float(), (1280,))[0, C // 2 * S:]          # (2048, D)
        z = enc["encoder"](x, masks=[ci])
        H[r["video_id"]] = (h.view(8, S, -1).cpu(), z)
OC = {v: obj_cells(v) for v in vids}
res = {}
for name in a.models.split(","):
    P = predictor(name); per = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for v in vids:
            oc = OC[v]
            for pair, pos, imp in (("A", "possible_present", "impossible_vanish"), ("B", "possible_absent", "impossible_appear")):
                rp, ri = by[(v, pos)], by[(v, imp)]
                hp, zp = H[rp["video_id"]]; hi, _ = H[ri["video_id"]]
                p = P(zp, ci, ti, mask_index=0).float()[0].view(8, S, -1).cpu()   # 문맥이 같다 → 같은 p
                lp, li = (p - hp).abs().mean(-1), (p - hi).abs().mean(-1)           # (8, 256) 토큰 L1
                # 있음 쪽 정답: A 는 가능 (있음), B 는 불가능 (생겨남)
                h_yes, h_no = (hp, hi) if pair == "A" else (hi, hp)
                d_yes, d_no = (p - h_yes).abs().mean(-1), (p - h_no).abs().mean(-1)
                m = torch.from_numpy(oc)
                per.append(dict(video=v, pair=pair, quality=rp["quality"], event=rp["condition"],
                                hid=int(info[v]["quad"]["hidden_run"][1] - info[v]["quad"]["hidden_run"][0]),
                                fps=float(info[v]["fps"]), n_obj=int(m.sum()),     # hidden_run 은 원본 장 번호 (60 fps 중복 영상도 원본 번호)
                                all=float(li.mean() - lp.mean()), obj=float(li[m].mean() - lp[m].mean()) if m.any() else float("nan"),
                                bg=float(li[~m].mean() - lp[~m].mean()),
                                obj_yes_minus_no=float(d_yes[m].mean() - d_no[m].mean()) if m.any() else float("nan")))
    res[name] = per
    del P; torch.cuda.empty_cache()
json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=1)

acc = lambda xs: 100 * np.mean([x > 0 for x in xs if x == x])
print("정확도 = 불가능 surprise > 가능 surprise 인 쌍 비율 (%) — 전체 칸 / 물체 칸만 / 배경 칸만 · 물체 칸에서 예측이 '있음' 정답에 더 가까운 비율")
for name, per in res.items():
    for pr in ("A", "B"):
        q = [x for x in per if x["pair"] == pr]
        print(f"  {name:20s} 짝 {pr}: 전체 {acc([x['all'] for x in q]):5.1f} · 물체 칸 {acc([x['obj'] for x in q]):5.1f} · 배경 {acc([x['bg'] for x in q]):5.1f}"
              f" · 물체 칸에서 '있음' 쪽 {acc([-x['obj_yes_minus_no'] for x in q]):5.1f} (n {len(q)})")
print("\n가려진 시간 (초) 별, 짝 A 물체 칸 정확도 (릴리즈 · wrop C)")
for name in [n for n in res if n in ("vith", "permanence_wrop_C")]:
    q = [x for x in res[name] if x["pair"] == "A"]
    for lo, hi in ((0, 1.5), (1.5, 2.5), (2.5, 9)):
        qq = [x for x in q if lo <= x["hid"] / x["fps"] < hi]
        print(f"  {name:20s} {lo}–{hi} 초: {acc([x['obj'] for x in qq]):5.1f} (n {len(qq)})")
