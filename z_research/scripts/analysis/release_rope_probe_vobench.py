#!/usr/bin/env python3
"""release predictor — 미래 질의가 과거 물체를 안 보는 이유가 RoPE 위치 (시간 · 공간 거리) 인가 (2026-10-30, 사용자 "RoPE 가 문제인 거야?" → "해봐").

세 가지 (학습 없음, surprise_c16t32 와 같은 채점: 32 장 · 문맥 16 장 · mean |p − LN(h)| · fp32 + fp16 autocast · matched pair):
  1. profile — 개입 없이 attention 분포. 미래 질의가 문맥 key 에 준 질량을 key 묶음별로 (층마다, head 평균):
       과거 물체 칸 (문맥 블록 b 별) · 마지막 문맥 블록 · 문맥 전체 · 미래 토큰.
     질의 묶음: 실제 미래 물체 칸 (라벨, 그 블록 두 장 모두 보일 때) · 미래 전부. 영상 = VOBench 전체 조건의 pos_a (문맥 A 에 물체).
  2·3. remap — RoPE 위치를 **key 쪽만** 바꿔 넣는다 (질의 · 내용은 그대로):
       t_obj  : 과거 물체 칸 key 의 시간 위치를 마지막 문맥 블록 (7) 로  ("방금 본 것" 처럼)
       t_all  : 문맥 블록 0–6 의 모든 key 의 시간 위치를 7 로
       s_true : 과거 물체 칸 key 의 공간 위치를 물체가 다시 보이는 첫 미래 블록의 자리로 옮김 (블록 b 의 물체 중심 → 재등장 중심 만큼 이동)
       s_wrong: 같은 크기 반대 방향 (위약)
       ts_true: t_obj + s_true
     ⚠️ s_* · ts_* 는 정답 궤적을 쓴다 (판별 상한). t_* 도 물체 칸 라벨을 쓴다. 학습 때 없던 위치 조합이라 결과는 조심해서 읽는다.
     VOBench late 432 쌍 정확도 + 같은 질량 지표 (실제 미래 물체 칸 질의).

  for g in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$g /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/release_rope_probe_vobench.py profile --shard $g --nshard 8 & done; wait
  for g in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$g /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/release_rope_probe_vobench.py remap --shard $g --nshard 8 & done; wait
  /venv/vjepa/bin/python z_research/scripts/analysis/release_rope_probe_vobench.py report
출력: z_analysis/results/release_rope_probe/
"""
import argparse, collections, csv, glob, json, os, sys
import numpy as np, torch, torch.nn.functional as F
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)
ap = argparse.ArgumentParser()
ap.add_argument("phase", choices=["profile", "remap", "report"])
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
a = ap.parse_args()
OUT = os.path.join(REPO, "z_analysis/results/release_rope_probe"); os.makedirs(OUT, exist_ok=True)
ROOT = "/workspace/dataset/benchmarks/VOBench_v1"
S, G, NB = 256, 16, 8
NCTX, N = NB * S, 2 * NB * S


def track(file_name, arm):
    """블록 0–15 마다 (보임 여부, 물체 칸 목록, 중심 칸 (r, c)). 그 블록 두 장 (원본 6b · 6b+3) 모두 hidden < 0.5 일 때만 보임."""
    L = json.load(open(os.path.join(ROOT, os.path.dirname(file_name), "label.json")))["labels"].get(arm) or []
    out = []
    for b in range(2 * NB):
        fr = [L[6 * b], L[6 * b + 3]] if len(L) > 6 * b + 3 else []
        if not (fr and all(e["hidden"] < 0.5 for e in fr)):
            out.append((False, [], None)); continue
        cells = set()
        for e in fr:
            u, v = e["u"] / 2, e["v"] / 2; hu, hv = e.get("ext_u", e["r_px"]) / 2, e.get("ext_v", e["r_px"]) / 2
            for r in range(G):
                for c in range(G):
                    if abs(c * 16 + 8 - u) <= hu + 8 and abs(r * 16 + 8 - v) <= hv + 8:
                        cells.add(b * S + r * G + c)
        uc, vc = np.mean([e["u"] / 2 for e in fr]), np.mean([e["v"] / 2 for e in fr])
        out.append((True, sorted(cells), (int(np.clip(vc // 16, 0, G - 1)), int(np.clip(uc // 16, 0, G - 1)))))
    return out


def build():
    from analysis.intphys2.model import build_from_config
    from analysis.intphys2.surprise import _context_target_indices
    from src.models.utils.modules import rotate_queries_or_keys as rot
    dev = torch.device("cuda")
    ck = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
    mcfg = dict(arch_name="vit_huge", checkpoint=ck, img_size=256, patch_size=16, tubelet_size=2, window_size=32, use_rope=True,
                uniform_power=False, dual_encoder=True, context_encoder_key="encoder", target_encoder_key="target_encoder",
                predictor=dict(embed_dim=384, depth=12, num_heads=12, num_mask_tokens=10), dtype="float32", autocast="float16")
    bundle = build_from_config(mcfg, dev)
    ci, ti = _context_target_indices(ctx_frames=16, tgt_frames=16, tubelet_size=2, spatial_tokens=S, batch_size=1, device=dev)
    CTRL = {"kremap": None, "rec": None}
    for li, blk in enumerate(bundle.predictor.predictor_blocks):
        at = blk.attn
        def fw(x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None, at=at, li=li):
            # src/models/utils/modules.py RoPEAttention.forward 와 같은 계산 + key 위치 바꾸기 · attention 질량 기록 (2026-10-30)
            B, Nn, C = x.size()
            qkv = at.qkv(x).unflatten(-1, (3, at.num_heads, -1)).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            m = mask.unsqueeze(1).repeat(1, at.num_heads, 1)
            d, h, w = at.separate_positions(m, H_patches, W_patches)
            dk, hk, wk = d, h, w
            if CTRL["kremap"] is not None:
                idx, nd, nh, nw = CTRL["kremap"]
                dk, hk, wk = d.clone(), h.clone(), w.clone()
                if nd is not None: dk[..., idx] = nd
                if nh is not None: hk[..., idx] = nh
                if nw is not None: wk[..., idx] = nw
            s = 0; parts_q, parts_k, parts_k0 = [], [], []
            for dim_, pq, pk in ((at.d_dim, d, dk), (at.h_dim, h, hk), (at.w_dim, w, wk)):
                parts_q.append(rot(q[..., s:s + dim_], pos=pq)); parts_k.append(rot(k[..., s:s + dim_], pos=pk))
                if CTRL["kremap"] is not None:
                    parts_k0.append(rot(k[..., s:s + dim_], pos=pq))
                s += dim_
            if s < at.head_dim:
                parts_q.append(q[..., s:]); parts_k.append(k[..., s:]); parts_k0.append(k[..., s:])
            q, k = torch.cat(parts_q, -1), torch.cat(parts_k, -1)
            if CTRL["kremap"] is None:
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=at.is_causal, attn_mask=attn_mask)
            else:   # 바꾼 key 위치는 미래 질의에만. 문맥 질의는 원래 위치 (문맥 처리는 그대로)
                k0 = torch.cat(parts_k0, -1)
                out = torch.cat([F.scaled_dot_product_attention(q[:, :, :NCTX], k0, v, dropout_p=0.0),
                                 F.scaled_dot_product_attention(q[:, :, NCTX:], k, v, dropout_p=0.0)], dim=2)
            R = CTRL["rec"]
            if R is not None:
                for qn, qidx in R["queries"].items():
                    if len(qidx) == 0: continue
                    P = ((q[:, :, qidx] @ k.transpose(-2, -1)).float() * at.scale).softmax(-1).mean(1).mean(1)[0]   # (N,) head · 질의 평균
                    R["out"].setdefault(qn, {})[li] = {gn: float(P[gi].sum()) for gn, gi in R["keys"].items()}
            return at.proj(out.transpose(1, 2).reshape(B, Nn, C))
        at.forward = fw
    return bundle, ci, ti, CTRL, dev


def dataset(root):
    from evals.world_model_analysis.data import WMADataset
    cfg = {"data": dict(root=root, index_csv="index.csv", frames_root=ROOT, frames_pattern="{file_name}/{frame:06d}.png", frames_start=0,
                        frames_stride=3, block_column="block_id", pair_column="pair_id", variant_column="variant", plausible_column="plausible",
                        type_column="condition", n_frames=32, resolution=256), "model": {"img_size": 256}}
    return WMADataset(cfg), list(csv.DictReader(open(os.path.join(root, "index.csv"))))


def groups(tr, dev):
    ctx_vis = [b for b in range(NB) if tr[b][0]]
    obj = [c for b in ctx_vis for c in tr[b][1]]
    keys = {"obj_past": obj, "last_ctx": list(range((NB - 1) * S, NCTX)), "ctx_all": list(range(NCTX)), "fut": list(range(NCTX, N))}
    for b in range(NB):
        keys[f"obj_b{b}"] = tr[b][1] if tr[b][0] else []
    fut_obj = [c + 0 for b in range(NB, 2 * NB) if tr[b][0] for c in tr[b][1]]
    return {kn: torch.tensor(v, device=dev, dtype=torch.long) for kn, v in keys.items()}, fut_obj


if a.phase == "profile":
    bundle, ci, ti, CTRL, dev = build()
    ds, rows = dataset(os.path.join(REPO, "data_csv/vobench_v1"))
    sel = [i for i, r in enumerate(rows) if r["variant"] == "pos_a"][a.shard::a.nshard]
    rec = []
    for i in sel:
        r = rows[i]; tr = track(r["file_name"], "A")
        keys, fut_obj = groups(tr, dev)
        CTRL["rec"] = {"queries": {"fut_obj": torch.tensor(fut_obj, device=dev, dtype=torch.long),
                                   "fut_all": torch.arange(NCTX, N, device=dev)}, "keys": keys, "out": {}}
        x = ds.clip(i)[None].to(dev, bundle.dtype)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            z = bundle.context_encoder(x, masks=[ci]); bundle.predictor(z, ci, ti, mask_index=0)
        rec.append(dict(vid=r["video_id"], occ=r["occlusion"], k=int(r["k"]), motion=r["motion"], objset=r["objset"],
                        n_obj=len(keys["obj_past"]), n_futobj=len(fut_obj), mass=CTRL["rec"]["out"]))
        CTRL["rec"] = None
        if len(rec) % 20 == 0: print(f"[profile {a.shard}] {len(rec)}/{len(sel)}", flush=True)
    json.dump(rec, open(os.path.join(OUT, f"profile_shard{a.shard}.json"), "w"))
    print("끝")

elif a.phase == "remap":
    bundle, ci, ti, CTRL, dev = build()
    ds, rows = dataset(os.path.join(REPO, "data_csv/vobench_v1_late"))
    byb = collections.defaultdict(dict)
    for i, r in enumerate(rows):
        byb[r["block_id"]][r["variant"]] = i
    rec = []
    for bid in sorted(byb, key=int)[a.shard::a.nshard]:
        vv = byb[bid]
        for arm, pos, imp in (("A", "pos_a", "imp_ab"), ("B", "pos_b", "imp_ba")):
            if pos not in vv or imp not in vv: continue
            r = rows[vv[pos]]; tr = track(r["file_name"], arm)
            keys, fut_obj = groups(tr, dev)
            obj = keys["obj_past"]
            reap = next((b for b in range(NB, 2 * NB) if tr[b][0]), None)          # 물체가 다시 보이는 첫 미래 블록
            dr = torch.zeros(len(obj), dtype=torch.long, device=dev); dc = torch.zeros_like(dr)
            if reap is not None and len(obj):
                rr, cc = tr[reap][2]; j = 0
                for b in range(NB):
                    if tr[b][0]:
                        n_ = len(tr[b][1]); dr[j:j + n_] = rr - tr[b][2][0]; dc[j:j + n_] = cc - tr[b][2][1]; j += n_
            hr, wc = (obj % S) // G, obj % G
            ctx06 = torch.arange(0, (NB - 1) * S, device=dev)
            conds = {"base": None,
                     "t_obj": (obj, torch.full_like(obj, NB - 1), None, None),
                     "t_all": (ctx06, torch.full_like(ctx06, NB - 1), None, None),
                     "s_true": (obj, None, (hr + dr).clamp(0, G - 1), (wc + dc).clamp(0, G - 1)),
                     "s_wrong": (obj, None, (hr - dr).clamp(0, G - 1), (wc - dc).clamp(0, G - 1)),
                     "ts_true": (obj, torch.full_like(obj, NB - 1), (hr + dr).clamp(0, G - 1), (wc + dc).clamp(0, G - 1))}
            x = torch.stack([ds.clip(vv[pos]), ds.clip(vv[imp])]).to(dev, bundle.dtype)
            res, mass = {}, {}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                z = bundle.context_encoder(x[:1], masks=[ci])
                h = bundle.target_encoder(x)
                h = F.layer_norm(torch.gather(h, 1, ti.expand(2, -1).unsqueeze(-1).expand(-1, -1, h.size(-1))), (h.size(-1),))
                for cn, km in conds.items():
                    if cn != "base" and len(obj) == 0:
                        res[cn] = res["base"]; mass[cn] = mass["base"]; continue
                    CTRL["kremap"] = km
                    CTRL["rec"] = {"queries": {"fut_obj": torch.tensor(fut_obj, device=dev, dtype=torch.long)}, "keys": keys, "out": {}}
                    p = bundle.predictor(z, ci, ti, mask_index=0).float()
                    CTRL["kremap"] = None
                    res[cn] = (float((p - h[0:1].float()).abs().mean()), float((p - h[1:2].float()).abs().mean()))
                    mass[cn] = CTRL["rec"]["out"].get("fut_obj", {}); CTRL["rec"] = None
            rec.append(dict(block_id=bid, ctx=arm, violation=r["violation"], motion=r["motion"], objset=r["objset"], k=int(r["k"]),
                            n_obj=len(obj), reap=reap, res=res, mass=mass))
        print(f"[remap {a.shard}] block {bid} ({len(rec)} 쌍)", flush=True)
    json.dump(rec, open(os.path.join(OUT, f"remap_shard{a.shard}.json"), "w"))
    print("끝")

else:
    P = [r for f in sorted(glob.glob(os.path.join(OUT, "profile_shard*.json"))) for r in json.load(open(f))]
    res = {}
    if P:
        print(f"=== 1. attention 분포 (개입 없음) · pos_a {len(P)} 편 · 층 평균 (괄호 = 층 1 · 6 · 12)")
        cond = lambda r: r["occ"] if r["occ"] == "visible" else f"{r['occ']}"
        def mm(rr, qn, kn, layers=range(12)):
            v = [np.mean([r["mass"][qn][str(l)][kn] for l in layers]) for r in rr if qn in r["mass"] and r["mass"][qn]]
            return 100 * float(np.mean(v)) if v else float("nan")
        print(f"{'조건':16s} {'n':>4s} {'질의':8s} {'과거 물체 칸':>14s} {'마지막 문맥 블록':>16s} {'문맥 전체':>10s} {'미래':>8s} {'물체 칸 수':>9s}")
        for oc in ("visible", "early", "mid", "late"):
            for mv in ("정지", "움직임"):
                rr = [r for r in P if r["occ"] == oc and ((r["motion"] == "flat_static") == (mv == "정지")) and r["n_obj"] > 0]
                for qn in ("fut_obj", "fut_all"):
                    row = [mm(rr, qn, kn) for kn in ("obj_past", "last_ctx", "ctx_all", "fut")]
                    lay = " ".join(f"{mm(rr, qn, 'obj_past', [l]):.1f}" for l in (0, 5, 11))
                    res[f"profile|{oc}|{mv}|{qn}"] = row
                    print(f"{oc + ' · ' + mv:16s} {len(rr):4d} {qn:8s} {row[0]:7.2f} ({lay}) {row[1]:16.1f} {row[2]:10.1f} {row[3]:8.1f} {np.mean([r['n_obj'] for r in rr]):9.0f}")
        print("과거 물체 칸 질량을 문맥 블록별로 (fut_obj 질의, 층 평균, late · 움직임 vs early · 움직임):")
        for oc in ("early", "late"):
            rr = [r for r in P if r["occ"] == oc and r["motion"] != "flat_static" and r["n_obj"] > 0]
            print(f"   {oc:6s} " + " ".join(f"b{b}:{mm(rr, 'fut_obj', f'obj_b{b}'):.2f}" for b in range(NB)))
    Rm = [r for f in sorted(glob.glob(os.path.join(OUT, "remap_shard*.json"))) for r in json.load(open(f))]
    if Rm:
        names = list(Rm[0]["res"])
        acc = lambda rr, n: 100 * np.mean([r["res"][n][1] > r["res"][n][0] for r in rr]) if rr else float("nan")
        mo = lambda rr, n: 100 * np.mean([np.mean([r["mass"][n][str(l)]["obj_past"] for l in range(12)]) for r in rr if r["mass"].get(n)]) if rr else float("nan")
        print(f"\n=== 2 · 3. RoPE key 위치 바꾸기 · VOBench late {len(Rm)} 쌍 · 정확도 % (아래 줄 = 실제 미래 물체 칸 질의가 과거 물체 칸에 준 질량 %, 층 평균)")
        grp = [("전체", lambda r: True), ("A", lambda r: r["ctx"] == "A"), ("A · 정지", lambda r: r["ctx"] == "A" and r["motion"] == "flat_static"),
               ("A · 움직임", lambda r: r["ctx"] == "A" and r["motion"] != "flat_static"), ("영속 A", lambda r: r["ctx"] == "A" and r["violation"] == "permanence"),
               ("모양 A", lambda r: r["ctx"] == "A" and r["violation"] == "shape"), ("색 A", lambda r: r["ctx"] == "A" and r["violation"] == "color"),
               ("B (물체 있음)", lambda r: r["ctx"] == "B" and r["n_obj"] > 0)]
        print(f"{'묶음':14s} {'n':>4s} " + " ".join(f"{n:>8s}" for n in names))
        for gn, f in grp:
            rr = [r for r in Rm if f(r)]
            res[f"remap|{gn}"] = {n: acc(rr, n) for n in names}
            print(f"{gn:14s} {len(rr):4d} " + " ".join(f"{acc(rr, n):8.1f}" for n in names))
            if gn in ("A", "A · 움직임"):
                print(f"{'  └ 질량':14s} {'':4s} " + " ".join(f"{mo(rr, n):8.2f}" for n in names))
    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), ensure_ascii=False, indent=1)
