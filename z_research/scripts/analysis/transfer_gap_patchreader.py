#!/usr/bin/env python3
r"""[X1] R2 E20 위치 판독기 (패치 판독기) — h 로 배운 판독기가 p 의 정체를 어느 자리에서 읽나 (H1-P · late '묶임' / '퍼짐') (2026-10-30).
  대조 S1 (보임 조건 · 이동 · t8–10 보임) — attention h 판독기가 이식되는 칸. 여기서 릴리즈 p 를 참 자리에서 읽어야 판독기가 지나치게 엄격하지 않다.

왜 패치 판독기인가: 절대 위치 임베딩을 넣으면 판독기가 화면 자리 사전을 외운다 (CLAUDE.md §6, v11 k = 0 자 실패). 그래서 읽을 자리를 밖에서 준다.
구조: 3×3 칸 토큰 (토큰 LN) + 패치 안 상대 위치 임베딩 9 개 → attention 풀링 (질의 1, 머리 16, depth 1 = E3 판독기와 같은 AttentiveClassifier)
  → 클래스 = (모양/색) 결합 + '없음' (empty).
학습: h (= LN(target_encoder(32 장))), 보임 조건 (visible · early · mid), split v2 train block, 가능 변이.
  양성 = 물체가 다 보이는 튜블릿의 참 물체 중심 3×3 (label 의 중심 칸). 음성 '없음' = (i) 물체에서 3 칸 넘게 떨어진 무작위 3×3 (영상 · 튜블릿당 1 개)
  (ii) 빈 장면 영상 (permanence 의 빈 팔) 의 같은 궤적 자리 3×3 (가장 어려운 음성). 12 epoch · AdamW lr 3e-4 · wd 0.05 · batch 64. 시드 0 · 1.
자 검증 (test block): ① held-out h → h (참 자리) ≥ 85 % · ② 배경 '없음' ≥ 90 % (먼 자리 · 빈 장면 궤적 자리) ·
  ③ encoder 대조: 온라인 문맥 z (E21 추출, 보임 조건) 의 마지막 보인 문맥 튜블릿 · 물체 자리 ≥ 80 % ·
  ④ 파라미터 없는 교차: E2 의 자리 질량 순서 (S2 에서 참 > 끝) 가 판독 순서 (O참 > O끝) 와 같은가.
적용: 릴리즈 p · PFT p · 배율 2γ 판 (μ^p + 2γ_t (p̃ − μ^p)) × 자리 {O참 (참 중심) · O끝 (마지막 보인 문맥 튜블릿의 중심) · O_p (출력의 대비 에너지 최대 칸 —
  모양 · 색 block 은 두 팔 정체 대비, permanence block 은 존재 대비)} × S2 (보임 조건 · 이동 · t11–15 보임) · S4 (late · 이동 · 다시 보이는 걸음) ·
  S4o (late · 이동 · 가려진 미래 걸음 — 물체가 가려져 있어야 하는 걸음; 참 자리에 '물체' 를 읽나).
판정 (PLAN_R2 §5 E20): H1-P = O_p ≥ O참 + 15 & O_p ≥ 50 (S2). late: O끝 ≥ 50 & O참 ≤ 20 → '마지막 본 자리에 묶임' · 셋 다 ≤ 20 (배율 판 포함) → '퍼짐'.
출력: TG/heads/patch_<objset>_s<seed>.pt · results/q1q2_r2/E20_patch.json

  $P z_research/scripts/analysis/transfer_gap_patchreader.py --gpu 0
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transfer_gap_common as C  # noqa: E402

NT, NF, S, D, G = C.NT, C.NF, C.S, C.D, C.GRID
OUT = C.REPO / "z_analysis/transfer_gap/results/q1q2_r2"


def box3(rc):
    r, c = rc
    r = min(max(r, 1), G - 2); c = min(max(c, 1), G - 2)            # 가장자리는 안쪽으로 붙여 9 칸 유지
    return [rr * G + cc for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2)]


def make_reader(n_cls, dev):
    import torch, torch.nn as nn

    class PatchReader(nn.Module):
        def __init__(self):
            super().__init__()
            self.pos = nn.Parameter(torch.zeros(9, D))
            nn.init.trunc_normal_(self.pos, std=0.02)
            self.head = C.make_head(n_cls, dev)

        def forward(self, x):                      # x (B, 9, D) — 토큰 LN 된 것
            return self.head(x + self.pos)
    return PatchReader().to(dev)


def main(gpu, seeds=(0, 1), limit=0):
    import torch, torch.nn as nn
    dev = torch.device(f"cuda:{gpu}"); torch.cuda.set_device(dev)
    LN = lambda x: nn.functional.layer_norm(x.float(), (x.shape[-1],))
    M = C.meta(); trb = C.train_blocks(); bmap, UV, HID, EXT = C.load_tracks()
    H = np.load(C.TUBE / "h.npy", mmap_mode="r"); P = np.load(C.TUBE / "p.npy", mmap_mode="r"); PF = np.load(C.TG / "p_relftg.npy", mmap_mode="r")
    ZV = np.load(C.TG / "z_vis_ctx.npy", mmap_mode="r"); zv = json.load(open(C.TG / "z_vis_meta.json")); zvslot = {j: q for q, j in enumerate(zv["tp"])}
    blocks = defaultdict(dict)
    for j, m in enumerate(M):
        blocks[m["block_id"]][m["variant"]] = j
    res = {"definition": __doc__, "objsets": {}}
    t0 = time.time()
    for objset in ("primitive", "realistic"):
        labs, L = C.classes(M, objset); NONE = L["empty"]
        mp = torch.load(C.MAPS / f"{objset}.pt", map_location=dev)
        mu_p, gam_t = mp["mu_p"].float().to(dev), mp["gam_t"].float().to(dev)
        vids = [j for j, m in enumerate(M) if m["objset"] == objset]
        if limit:
            vids = vids[:limit]
        geo = {}
        for j in vids:
            m = M[j]; bi = bmap[m["block_id"]]
            arm = 0 if m["variant"] == "pos_a" else 1
            empty = C.label_of(m) == "empty"
            a_ = 0 if empty else arm                       # 빈 장면은 짝 팔 (물체 팔) 의 궤적 자리
            uv = UV[bi, a_]; hid = HID[bi, a_] if not empty else HID[bi, 0]
            geo[j] = dict(uv=uv, hid=hid, tl=C.last_visible_ctx(hid), empty=empty, st=C.tub_state(m["samples"]))
        rng = np.random.default_rng(0)

        def far_patch(uv_t):
            r0, c0 = C.cell_of(uv_t)
            while True:
                r, c = int(rng.integers(1, G - 1)), int(rng.integers(1, G - 1))
                if max(abs(r - r0), abs(c - c0)) > 3:
                    return (r, c)

        # 학습 · 시험 표본 (h): (j, t, 중심 칸, 라벨)
        def h_samples(sel_blocks):
            out = []
            for j in vids:
                m = M[j]
                if m["occlusion"] == "late" or (m["block_id"] in trb) != sel_blocks:
                    continue
                g = geo[j]
                for t in range(NT):
                    if g["empty"]:
                        out.append((j, t, C.cell_of(g["uv"][t]), NONE, "empty_path"))
                    elif g["st"][t] == "vis":
                        out.append((j, t, C.cell_of(g["uv"][t]), L[C.label_of(m)], "obj"))
                    out.append((j, t, far_patch(g["uv"][t]), NONE, "far"))
            return out
        tr_s, te_s = h_samples(True), h_samples(False)
        print(f"[{objset}] train {len(tr_s)} test {len(te_s)} samples · {time.time() - t0:.0f}s", flush=True)
        jj = sorted({s[0] for s in tr_s + te_s})
        Hc = {j: torch.from_numpy(np.asarray(H[j], np.float16)) for j in jj}
        print(f"[{objset}] h loaded {len(Hc)} · {time.time() - t0:.0f}s", flush=True)

        def patch(X, t, rc):
            return X[t, box3(rc)]

        def batch_h(samples):
            return torch.stack([patch(Hc[j], t, rc) for j, t, rc, _, _ in samples]).to(dev)

        readers = []
        for seed in seeds:
            torch.manual_seed(seed); rr = np.random.default_rng(seed)
            clf = make_reader(len(labs), dev)
            opt = torch.optim.AdamW(clf.parameters(), lr=3e-4, weight_decay=0.05)
            ep_n, bs = 12, 64
            steps = ep_n * ((len(tr_s) + bs - 1) // bs)
            sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps)
            y = torch.tensor([s[3] for s in tr_s], device=dev)
            for ep in range(ep_n):
                perm = rr.permutation(len(tr_s)); tot = 0.0; clf.train()
                for s0 in range(0, len(tr_s), bs):
                    sel = perm[s0:s0 + bs]
                    xb = LN(batch_h([tr_s[i] for i in sel]))
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = nn.functional.cross_entropy(clf(xb), y[sel])
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step(); tot += float(loss) * len(sel)
                if ep % 4 == 3:
                    print(f"[{objset}] s{seed} ep{ep + 1} loss {tot / len(tr_s):.3f} {time.time() - t0:.0f}s", flush=True)
            clf.eval(); readers.append(clf)
            torch.save(clf.state_dict(), C.HEADS / f"patch_{objset}_s{seed}.pt")

        def read(X):                                  # X (B, 9, D) — LN 전
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                return np.stack([r(LN(X)).float().argmax(1).cpu().numpy() for r in readers])   # (seeds, B)

        cell = {}
        # ① · ② 자 검증 (test h)
        for kind in ("obj", "far", "empty_path"):
            ss = [s for s in te_s if s[4] == kind]
            pr = np.concatenate([read(batch_h(ss[i:i + 256])) for i in range(0, len(ss), 256)], 1)
            gt = np.array([s[3] for s in ss])
            cell[f"check_h_{kind}"] = dict(n=len(ss), acc=100 * float((pr == gt).mean()), none=100 * float((pr == NONE).mean()),
                                          shape=100 * float(np.mean([labs[a].split("/")[0] == labs[b].split("/")[0] for a, b in zip(pr.flatten(), np.tile(gt, len(readers)))])) if kind == "obj" else None)
        # ③ encoder 대조: 온라인 문맥 z 의 마지막 보인 문맥 튜블릿, 물체 자리 (test block, 보임 조건 · 이동 · 모양 · 색)
        zs = []
        for j in vids:
            if j not in zvslot or M[j]["block_id"] in trb or geo[j]["empty"]:
                continue
            g = geo[j]; tl = g["tl"]
            if g["st"][tl] != "vis":
                continue
            zs.append((j, tl, C.cell_of(g["uv"][tl]), L[C.label_of(M[j])]))
        if zs:
            X = torch.stack([torch.from_numpy(np.asarray(ZV[zvslot[j]][t, box3(rc)], np.float32)) for j, t, rc, _ in zs]).to(dev)
            pr = read(X); gt = np.array([s[3] for s in zs])
            cell["check_z_online_lastvis"] = dict(n=len(zs), acc=100 * float((pr == gt).mean()), none=100 * float((pr == NONE).mean()),
                                                  shape=100 * float(np.mean([labs[a].split("/")[0] == labs[b].split("/")[0] for a, b in zip(pr.flatten(), np.tile(gt, len(readers)))])))
        # 적용: S2 · S4 · S4o
        app = defaultdict(list)    # key → [(block, acc, none, shape)]
        for b, vv in sorted(blocks.items(), key=lambda x: int(x[0])):
            ja = vv["pos_a"]; m = M[ja]
            if ja not in geo or vv["pos_b"] not in geo:
                continue
            if m["objset"] != objset or m["block_id"] in trb or m["motion"] == "flat_static":
                continue
            jb = vv["pos_b"]; late = m["occlusion"] == "late"
            arms = [ja] if m["violation"] == "permanence" else [ja, jb]
            src = {}
            for nm, A_ in (("rel", P), ("pft", PF)):
                pa = LN(torch.from_numpy(np.asarray(A_[ja], np.float32)).to(dev)); pb = LN(torch.from_numpy(np.asarray(A_[jb], np.float32)).to(dev))
                src[nm] = {ja: pa, jb: pb}
            src["rel2g"] = {j: mu_p + 2 * gam_t[:, None, None] * (src["rel"][j] - mu_p) for j in (ja, jb)}
            for nm in ("rel", "pft", "rel2g"):
                dP = src[nm][ja] - src[nm][jb]
                Op = [divmod(int((dP[k] ** 2).sum(-1).argmax()), G) for k in range(NF)]
                for j in arms:
                    g = geo[j]; gt = L[C.label_of(M[j])]
                    for k in range(NF):
                        t = NF + k
                        stt = g["st"][t]
                        if not late and stt != "vis":
                            continue
                        if late and stt == "mix":
                            continue
                        scen = (("S1" if t <= 10 else "S2") if not late else "S4" if stt == "vis" else "S4o")
                        places = {"Otrue": C.cell_of(g["uv"][t]), "Oend": C.cell_of(g["uv"][g["tl"]]), "Op": Op[k]}
                        X = torch.stack([src[nm][j][k, box3(rc)] for rc in places.values()])
                        pr = read(X)
                        for q, pn in enumerate(places):
                            a = pr[:, q]
                            app[(scen, nm, pn, m["violation"] == "permanence")].append(
                                (b, float((a == gt).mean()), float((a == NONE).mean()),
                                 float(np.mean([labs[x].split("/")[0] == labs[gt].split("/")[0] for x in a]))))
                        app[(scen, nm, "dist_Op_true", m["violation"] == "permanence")].append(
                            (b, float(np.hypot(Op[k][0] - places["Otrue"][0], Op[k][1] - places["Otrue"][1])), 0.0, 0.0))
        for (scen, nm, pn, perm), v in app.items():
            g = [x[0] for x in v]
            key = f"{scen}|{nm}|{pn}|{'perm' if perm else 'id'}"
            if pn == "dist_Op_true":
                cell[key] = dict(mean_cells=float(np.mean([x[1] for x in v])), n=len(v)); continue
            cell[key] = dict(acc=C.bootstrap_mean([100 * x[1] for x in v], g), none=C.bootstrap_mean([100 * x[2] for x in v], g),
                             shape=C.bootstrap_mean([100 * x[3] for x in v], g))
        res["objsets"][objset] = cell
        brief = {k: (round(v["acc"]["mean"], 1), round(v["none"]["mean"], 1)) if "acc" in v and isinstance(v["acc"], dict) else v for k, v in cell.items()}
        print(objset, json.dumps(brief, ensure_ascii=False, default=float)[:4000], f"{time.time() - t0:.0f}s", flush=True)
        del Hc
        OUT.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(OUT / f"E20_patch{'_smoke' if limit else ''}.json", "w"), indent=1, ensure_ascii=False, default=float)
    # 판정
    ver = {}
    for o, cell in res["objsets"].items():
        g = lambda k: cell[k]["acc"]["mean"] if k in cell and isinstance(cell[k].get("acc"), dict) else float("nan")
        val = dict(heldout=cell.get("check_h_obj", {}).get("acc"), none_far=cell.get("check_h_far", {}).get("none"),
                   none_emptypath=cell.get("check_h_empty_path", {}).get("none"), z_online=cell.get("check_z_online_lastvis", {}).get("acc"))
        ok = (val["heldout"] or 0) >= 85 and (val["none_far"] or 0) >= 90 and (val["z_online"] or 0) >= 80
        d = dict(validation=val, ruler_ok=ok)
        for nm in ("rel", "pft", "rel2g"):
            s2 = {pn: g(f"S2|{nm}|{pn}|id") for pn in ("Otrue", "Oend", "Op")}
            s4 = {pn: g(f"S4|{nm}|{pn}|id") for pn in ("Otrue", "Oend", "Op")}
            d[nm] = dict(S2=s2, S4=s4, H1_P=bool(s2["Op"] >= s2["Otrue"] + 15 and s2["Op"] >= 50),
                         late=("마지막 본 자리에 묶임" if s4["Oend"] >= 50 and s4["Otrue"] <= 20 else
                               "퍼짐 (셋 다 ≤ 20)" if max(s4.values()) <= 20 else "그 밖"))
        ver[o] = d
    res["verdict"] = ver
    json.dump(res, open(OUT / f"E20_patch{'_smoke' if limit else ''}.json", "w"), indent=1, ensure_ascii=False, default=float)
    print(json.dumps(ver, ensure_ascii=False, default=float))


def scan(gpu):
    """화면 전체 훑기 (계획 밖 추가): 저장한 위치 판독기로 14×14 = 196 개 3×3 자리를 모두 읽어,
    '물체' 확률 (1 − P(없음), 두 시드 평균) 이 가장 큰 자리 · 그 자리의 정체 · 참 자리까지 거리를 낸다.
    표본 = test block 물체 영상의 S1 · S2 · S4 걸음 (릴리즈 p · PFT p · 참 h 대조). 출력 results/q1q2_r2/E20_scan.json"""
    import torch, torch.nn as nn
    dev = torch.device(f"cuda:{gpu}"); torch.cuda.set_device(dev)
    LN = lambda x: nn.functional.layer_norm(x.float(), (x.shape[-1],))
    M = C.meta(); trb = C.train_blocks(); bmap, UV, HID, EXT = C.load_tracks()
    H = np.load(C.TUBE / "h.npy", mmap_mode="r"); P = np.load(C.TUBE / "p.npy", mmap_mode="r"); PF = np.load(C.TG / "p_relftg.npy", mmap_mode="r")
    centers = [(r, c) for r in range(1, G - 1) for c in range(1, G - 1)]
    idx = torch.tensor([box3(rc) for rc in centers], device=dev)            # (196, 9)
    out = {"definition": scan.__doc__, "objsets": {}}
    for objset in ("primitive", "realistic"):
        labs, L = C.classes(M, objset); NONE = L["empty"]
        readers = []
        for sd in (0, 1):
            r = make_reader(len(labs), dev); r.load_state_dict(torch.load(C.HEADS / f"patch_{objset}_s{sd}.pt", map_location=dev)); r.eval(); readers.append(r)
        res = defaultdict(list)
        for j, m in enumerate(M):
            if m["objset"] != objset or m["block_id"] in trb or C.label_of(m) == "empty" or m["motion"] == "flat_static":
                continue
            st = C.tub_state(m["samples"]); late = m["occlusion"] == "late"; gt = L[C.label_of(m)]
            bi = bmap[m["block_id"]]; arm = 0 if m["variant"] == "pos_a" else 1; uv = UV[bi, arm]
            ks = [k for k in range(NF) if st[NF + k] == "vis"]
            if not ks:
                continue
            for nm, A_, off in (("rel", P, NF), ("pft", PF, NF), ("h", H, 0)):
                X = LN(torch.from_numpy(np.asarray(A_[j], np.float32)).to(dev))
                for k in ks:
                    t = NF + k
                    scen = ("S1" if t <= 10 else "S2") if not late else "S4"
                    tok = X[t - off][idx]                                            # (196, 9, D)
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        pr = torch.stack([r(LN(tok)).float().softmax(-1) for r in readers]).mean(0)   # (196, n_cls)
                    pobj = 1 - pr[:, NONE]
                    q = int(pobj.argmax()); cls = int(pr[q].argmax())
                    tr, tc = C.cell_of(uv[t]); rc = centers[q]
                    res[(scen, nm)].append((m["block_id"], float(pobj[q]), float(cls == gt), float(np.hypot(rc[0] - tr, rc[1] - tc)),
                                            float((pobj > 0.5).float().mean())))
        cell = {}
        for (scen, nm), v in res.items():
            g = [x[0] for x in v]
            cell[f"{scen}|{nm}"] = dict(n=len(v), max_pobj=C.bootstrap_mean([x[1] for x in v], g),
                                        any_obj_frac=100 * float(np.mean([x[1] > 0.5 for x in v])),
                                        id_correct_at_max=C.bootstrap_mean([100 * x[2] for x in v], g),
                                        dist_to_true_cells_median=float(np.median([x[3] for x in v])),
                                        frac_patches_obj=float(np.mean([x[4] for x in v])))
        out["objsets"][objset] = cell
        print(objset, {k: (round(v["max_pobj"]["mean"], 3), round(v["any_obj_frac"], 1), round(v["id_correct_at_max"]["mean"], 1), v["dist_to_true_cells_median"]) for k, v in sorted(cell.items())}, flush=True)
    json.dump(out, open(OUT / "E20_scan.json", "w"), indent=1, ensure_ascii=False, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--scan", action="store_true", help="저장한 판독기로 화면 전체 196 자리 훑기")
    a = ap.parse_args()
    scan(a.gpu) if a.scan else main(a.gpu, limit=a.limit)
