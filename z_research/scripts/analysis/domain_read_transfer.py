"""도메인 무관 읽기 측정 (2026-11-07, 사용자 "v4 로 post-training 할 때 domain agnostic 하게 읽는 법을 가르치고 싶어"). 학습 없음.

읽기 방향 = 도메인 하나에서 "물체를 본 문맥 A − 안 본 문맥 B" 차이의 장면 평균 (파라미터 맞춤 없음, ar_memory_readout.probe_acc 와 같은 판별).
다른 도메인의 장면마다 A − B 를 그 방향에 투영해 양수면 맞음 (우연 50 %). 짝 차이라 도메인 절편이 지워진다.
단계: enc 찾기 (문맥 블록 0–4 encoder 토큰 평균) · state 기억 (rollout 걸음 3–7 AR 상태 평균) · pred 쓰기 (같은 걸음 예측 평균) · slot (slot 판만, 마지막 5 걸음).
도메인: v4 (학습에 쓴 합성, 묶음의 물체 영상 vs 빈 장면 300 쌍) · vob_panel · vob_mover · sur (움직임) · real (95) · ip2 (A = pair 1 이라 물체를 봤는지 뜻이 안 정해져 있다, 참고).
판정 (결과 전에 고정): v4 방향이 다른 도메인에서 우연을 크게 넘고 v4 학습 전 AR 보다 높으면 도메인 무관. 도메인 안 판별은 높은데 v4 방향으로 50 근처면 도메인 전용.

  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 /venv/vjepa/bin/python z_research/scripts/analysis/domain_read_transfer.py --model vast_ar_vw_state_v4bc4
출력: z_analysis/results/domain_read_20261107/<model>.json
"""
import argparse, csv, json, os, random, sys
os.environ.setdefault("WMA_AR_BLOCK_CACHE", "0")
import numpy as np, torch, yaml
sys.path.insert(0, os.path.dirname(__file__))
from slot_diag import C, K, RES, ROOT, S, WMADataset, _context_target_indices, boards, build_from_config, ln, probe_acc

OUT = os.path.join(ROOT, "z_analysis/results/domain_read_20261107")
FAM = os.path.join(ROOT, "data_csv/blender_track_v4bc4_negimp/families.json")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
DOM = {"vob_panel": ("VOBench v3 late · panel", "vobench_v3", True), "vob_mover": ("VOBench v3 late · moving occluder", "vobench_v3", True),
       "sur": ("IntPhys1 Surrogate", "intphys1_surrogate", True), "real": ("real world (all 95)", "real_world_permanence_uniform", False),
       "ip2": ("IntPhys2 late · permanence (A=pair 1, B=pair 2)", "intphys2_late_final", False)}


def feats(bundle, core, x, ci):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        z = bundle.context_encoder(x, masks=[ci])
        zl = ln(z).to(z.dtype)
        p, st = core.rollout_kv(zl, ci, K, return_states=True)
    n = x.size(0)
    f = {"enc": zl.float().view(n, C, S, -1)[:, :5].mean((1, 2)), "state": st.float().view(n, K, S, -1)[:, 3:].mean((1, 2)), "pred": p.float().view(n, K, S, -1)[:, 3:].mean((1, 2))}
    if getattr(core, "slot_n", 0):
        f["slot"] = torch.stack(core._slot_traj[-5:]).float().mean(0).flatten(1)
        core._slot_traj = None
    return {k: v.cpu().numpy() for k, v in f.items()}


def read_v4(path):
    import decord
    x = torch.from_numpy(decord.VideoReader(path, num_threads=2).get_batch(list(range(0, 64, 2))).asnumpy()).permute(3, 0, 1, 2).float().div_(255.0)
    return (x - MEAN) / STD


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--n_v4", type=int, default=300); a = ap.parse_args()
    dev = torch.device("cuda:0")
    bundle = build_from_config(yaml.safe_load(open(f"{RES}/real_world_permanence_uniform_{a.model}/_resolved.yaml"))["model"], dev)
    core = bundle.predictor.core
    ci, _ = _context_target_indices(ctx_frames=2 * C, tgt_frames=2 * K, tubelet_size=2, spatial_tokens=S, batch_size=2, device=dev)
    D = {}
    d = json.load(open(FAM))
    L = [l.strip().split()[0] for l in open(d["list"]) if l.strip()]
    fam = [f for f in d["families"] if any(f["hs"])]
    random.Random(0).shuffle(fam)
    acc = []
    for f in fam[: a.n_v4]:
        k = f["hs"].index(True)
        x = torch.stack([read_v4(L[f["A"][k]]), read_v4(L[f["B"]])]).to(dev, bundle.dtype)
        r = feats(bundle, core, x, ci)
        acc.append({q: v[0] - v[1] for q, v in r.items()})
    D["v4"] = {q: np.stack([s[q] for s in acc]) for q in acc[0]}
    bd = boards()
    for name, (bn, ds_name, mv) in DOM.items():
        cfg = yaml.safe_load(open(f"{RES}/{ds_name}_{a.model}/_resolved.yaml"))
        blk = bd[bn][1]
        if mv:
            ms = {r["block_id"] for r in csv.DictReader(open(os.path.join(ROOT, f"data_csv/{ds_name}/index.csv"))) if r["motion"] != "flat_static"}
            blk = {b: v for b, v in blk.items() if b in ms}
        ds = WMADataset(cfg)
        pos = {r.video_id: i for i, r in enumerate(ds.records)}
        acc = []
        for b in sorted(blk):
            x = torch.stack([ds.clip(pos[blk[b]["A"][0]]), ds.clip(pos[blk[b]["B"][0]])]).to(dev, bundle.dtype)
            r = feats(bundle, core, x, ci)
            acc.append({q: v[0] - v[1] for q, v in r.items()})
        D[name] = {q: np.stack([s[q] for s in acc]) for q in acc[0]}
    res = {"model": a.model, "n": {k: int(next(iter(v.values())).shape[0]) for k, v in D.items()}, "within": {}, "from": {}, "cos_v4": {}}
    names = list(D)
    for q in D["v4"]:
        res["within"][q] = {n: 100 * probe_acc(D[n][q]) for n in names}
        res["from"][q] = {s: {t: 100 * float((D[t][q] @ D[s][q].mean(0) > 0).mean()) for t in names if t != s} for s in names}
        w = D["v4"][q].mean(0)
        res["cos_v4"][q] = {t: float(D[t][q].mean(0) @ w / (np.linalg.norm(D[t][q].mean(0)) * np.linalg.norm(w) + 1e-9)) for t in names if t != "v4"}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, f"{a.model}.json"), "w"), indent=1, ensure_ascii=False)
    print("DONE", a.model, res["n"])


if __name__ == "__main__":
    main()
