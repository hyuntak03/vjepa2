"""영상 공간 복사 거리 (2026-10-18, 사용자 제안 점검). target_encoder 로 **문맥 pass** (앞 16 장) 와 **전체 pass** (32 장) 를 따로 인코딩.

  CUDA_VISIBLE_DEVICES=7 /venv/vjepa/bin/python z_research/scripts/analysis/video_space_copy_gap.py

비교 (모두 affine 없는 LN 뒤 L1, 배경 / 물체 3×3, 미래 걸음 0 · 3 · 7):
  (A) 문맥 pass 마지막 블록  vs 전체 pass 미래 블록   — 영상 정답에 대한 gating anchor (실제로 쓸 수 있는 복사)
  (B) 전체 pass 마지막 문맥 블록 vs 전체 pass 미래 블록 — 같은 pass 안 복사 (미래를 본 토큰이라 실제로는 못 쓴다, 상한)
  (C) 블록 공간 복사 LN(h_{C-1}) vs LN(h_{C+k})        — 기준 (gating_anchor_gap.py 와 같은 값이 나와야 한다)
  (D) 문맥 pass vs 전체 pass, 같은 마지막 문맥 블록     — 미래를 보느냐 마느냐가 토큰을 얼마나 바꾸나
"""
import sys, os, csv, collections, numpy as np, torch, torch.nn.functional as F
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
sys.argv = [sys.argv[0], "--occ", "late"]
_src = open(f"{REPO}/z_research/scripts/analysis/obj_region_surprise.py").read()
_src = _src[: _src.index("rows = [r for r in csv.DictReader")]
__file__ = f"{REPO}/z_research/scripts/analysis/obj_region_surprise.py"
exec(_src)
def build(nf):
    e = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=nf, use_rope=True, uniform_power=False)
    _load_state_dict(e, state["target_encoder"], tag="target_encoder"); return e.to(dev).eval()
E16, E32 = build(2 * C), build(2 * (C + K))
allrows = list(csv.DictReader(open(f"{REPO}/data_csv/vobench_v1/index.csv")))
out = collections.defaultdict(lambda: collections.defaultdict(list))
LN = lambda t: F.layer_norm(t, (t.size(-1),))
for occ in ("visible", "late"):
    for r in [r for r in allrows if r["occlusion"] == occ and r["variant"] == "pos_a"][:60]:
        d = os.path.join(a.vobench, r["file_name"]); x, W0 = frames(d)
        Hb = blocks(x).view(16, S, -1)                                          # 블록 공간 LN(h)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            zc = LN(E16(x[:, : 2 * C].unsqueeze(0).to(dev)).float()[0]).view(C, S, -1)          # 문맥 pass
            zf = LN(E32(x[:, : 2 * (C + K)].unsqueeze(0).to(dev)).float()[0]).view(C + K, S, -1)  # 전체 pass
        obj = set(obj_tokens(os.path.join(os.path.dirname(d), "label.json"), W0)[0])
        bg = torch.tensor([t for t in range(S) if t not in obj], device=dev); ob = torch.tensor(sorted(obj), device=dev) if obj else None
        cands = {"(A) 문맥 pass 마지막 → 전체 pass 미래 (anchor)": (zc[C - 1], zf),
                 "(B) 전체 pass 마지막 문맥 → 전체 pass 미래 (상한)": (zf[C - 1], zf),
                 "(C) 블록 복사 (기준)": (Hb[C - 1], Hb)}
        for name, (ref, tgt) in cands.items():
            for k in (0, 3, 7):
                e = (ref - tgt[C + k]).abs().mean(-1)
                out[(occ, name)][f"bg{k}"].append(float(e[bg].mean()))
                if ob is not None: out[(occ, name)][f"ob{k}"].append(float(e[ob].mean()))
        e = (zc[C - 1] - zf[C - 1]).abs().mean(-1); out[(occ, "(D) 같은 블록, 문맥 pass vs 전체 pass")]["bg0"].append(float(e[bg].mean()))
        if ob is not None: out[(occ, "(D) 같은 블록, 문맥 pass vs 전체 pass")]["ob0"].append(float(e[ob].mean()))
print("L1 — 배경 (걸음 0 · 3 · 7) | 물체 (걸음 0 · 3 · 7)")
for (occ, name), v in out.items():
    ks = [k for k in (0, 3, 7) if v.get(f"bg{k}")]
    print(f"{occ:8s} {name:44s} 배경 " + " ".join(f"{np.mean(v[f'bg{k}']):.3f}" for k in ks)
          + ("  | 물체 " + " ".join(f"{np.mean(v[f'ob{k}']):.3f}" for k in ks if v.get(f'ob{k}')) if v.get("ob0") else "") + f"  n {len(v['bg0'])}")
