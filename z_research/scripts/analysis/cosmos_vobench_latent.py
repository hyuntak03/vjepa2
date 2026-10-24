#!/usr/bin/env python3
"""Cosmos-Predict1-4B 를 VOBench (+ gravity) 에 IntPhys 2 공식 Cosmos 채점 규약으로 (2026-10-21, 사용자: "cosmos 로 VoBench 성능").

공식 규약 (IntPhys2 레포 prediction_evals/app/cosmos/modelcustom/latent_wrapper.py · configs/cosmos_latent.yaml):
  문맥 9 장 + 미래 24 장 = 33 장, 640 × 1024, base 모델 (영상만), greedy (temperature 0), diffusion decoder · guardrail 끔.
  예측 = 생성 토큰 index → 토크나이저 FSQ implicit_codebook (6 차원) 코드, 정답 = 실제 33 장을 토큰화한 코드.
  surprise = L1(예측 코드, 정답 코드) 평균. 공식은 latent 1–4 칸을 쓰는데 1 칸 (문맥) 은 두 쪽이 같아 0 → 여기선 미래 latent 2–4 만 쓴다 (비교 순위 동일, 값만 4/3 배).
우리 쪽 차이:
  · greedy = temperature 0 + top_p None → 공개 코드 sample_top_k 의 argmax 분기 (top-p 경로는 temperature 0 에서 0 으로 나눈다; 공식 래퍼는 비공개 fork ARBaseLatentGenerationPipeline).
    ⚠️ 10-21 첫 run 은 top-k 1 (동점이면 무작위) 이었다 → cvpr/results/eval/testbed/cosmos_ar4b_latent_topk1 (같은 문맥 두 run 의 예측 토큰이 23–46 % 달랐다).
    argmax 여도 batch 구성 · GPU 에 따른 bf16 차이는 남는다.
  · 전처리: 공식은 비율 유지 + 가운데 자르기 (정사각 VOBench 는 위아래 37.5 % 잘림 → 바닥 물체가 화면 밖) — 여기 기본은 pillarbox (--crop 이면 공식).
  · 창 하나: 문맥 raw 30, 32, …, 46 · 미래 48, …, 94 (render_cosmos.py 와 같다, 위반 갈림 48 이 경계).
  · 속도: matched 쌍은 문맥이 같다 → **문맥마다 한 번만 생성** (VOBench block 마다 a · b 둘, gravity block 마다 하나) · 문맥 여러 개를 한 batch 로 생성 (KV 캐시 batch).
산출 (--out/<set>_shard{k}.json): per_video surprise (+ 문맥 키) · 문맥마다 예측 index (npz, 나중에 픽셀로 디코딩 가능).

  source /data/hyuntak/anaconda3/bin/activate cosmos-predict1; cd /data/hyuntak/project/2026/2027_cvpr/cosmos-predict1
  CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$PWD python <repo>/z_research/scripts/analysis/cosmos_vobench_latent.py --set vobench --shard 0 --nshard 8 --bs 8
  채점 합치기: --merge (vjepa2 환경에서도 돈다)

--set intphys1_uni (2026-10-22, 사용자: "intphys1 late 이것도 해봐 uni 로"): IntPhys 1 dev 180 쌍, V-JEPA 의 uniform-d 규약을 Cosmos 창 (9 + 24) 으로 옮긴 것.
  문맥 9 장 = [0, d−1] 균일 (round(linspace)), 미래 24 장 = [d, 99] 균일. d = 가능 · 불가능이 처음 픽셀로 갈리는 raw 프레임 (index 의 d 열).
  쌍 (block_id, pair_id) 마다 문맥이 같아 한 번만 생성. 프레임 = runs/_frames/IntPhys1_uni (원본 PNG 중 쓰는 장만 NFS 로 복사, index.csv 도 거기).
  merge: 전체 · group (moving/static × occluded/visible) · group|O1–O3. late occ 열 = occluded 90 쌍.

--set intphys2_late (2026-10-22, 사용자: "ar 도 돌리고 일단 돌릴 수 있는 거 돌려봐"): IntPhys 2 Main 498 쌍, data_csv/intphys2_late_uni/index.csv 의 d (사람 라벨, 60 fps).
  문맥 9 장 = [0, d−1] 균일, 미래 24 장 = [d, n−1] 균일 (n = 두 영상 중 짧은 길이). 원본 mp4 (NFS) 를 decord 로 직접 읽는다.
  ⚠️ IntPhys 2 는 가능 · 불가능 문맥이 픽셀로 다르다 → 영상마다 따로 생성 (문맥 키 = 영상). merge 는 (block_id, pair_id) 로 짝짓는다.
  merge: 전체 · 위반 종류 · 가림 뒤 (occluded) · d 출처 (own / other_pair)."""
import argparse, collections, csv, functools, glob, json, os, sys, time
import numpy as np

REPO = "/data/hyuntak/project/2026/2027_cvpr/vjepa2"
ap = argparse.ArgumentParser()
ap.add_argument("--set", default="vobench", choices=["vobench", "gravity", "intphys1_uni", "intphys2_late"])
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--limit", type=int, default=0, help="문맥 몇 개만 (속도 측정)")
ap.add_argument("--crop", action="store_true", help="공식 전처리 (비율 유지 + 가운데 자르기)")
ap.add_argument("--root", default=None, help="기본: VOBench_v1 (vobench · gravity) / IntPhys1_uni")
ap.add_argument("--checkpoint_dir", default="/data/hyuntak/project/2026/2027_cvpr/cosmos-predict1/checkpoints")
ap.add_argument("--out", default=f"{REPO}/cvpr/results/eval/testbed/cosmos_ar4b_latent")
ap.add_argument("--merge", action="store_true")
a = ap.parse_args()
IP1 = f"{REPO}/z_pixel_decoder/runs/_frames/IntPhys1_uni"
a.root = a.root or (IP1 if a.set == "intphys1_uni" else f"{REPO}/z_pixel_decoder/runs/_frames/VOBench_v1")
CTX, FUT = list(range(30, 47, 2)), list(range(48, 95, 2))
IDX = CTX + FUT
H, W, LAT = 640, 1024, (5, 40, 64)
tag = a.set + ("_crop" if a.crop else "")


def groups():
    """[(문맥 키, [video 행, …])] — 같은 문맥을 공유하는 영상 묶음 (첫 행 = 가능 영상)"""
    if a.set == "vobench":
        rows = list(csv.DictReader(open(f"{REPO}/data_csv/vobench_v1/index.csv")))
        side = {"pos_a": "a", "imp_ab": "a", "pos_b": "b", "imp_ba": "b"}
        key = lambda r: f"{r['block_id']}{side[r['variant']]}"
        path = lambda r: r["file_name"]
    elif a.set == "intphys2_late":
        rows = list(csv.DictReader(open(f"{REPO}/data_csv/intphys2_late_uni/index.csv")))
        key = lambda r: r["video_id"]
        path = lambda r: "/data/dataset/world/IntPhys2/Main/Videos/" + os.path.basename(r["file"])
        for r in rows:
            d, n = int(r["d"]), int(r["n"])
            r["mp4_frames"] = [int(round(x)) for x in np.linspace(0, d - 1, 9)] + [int(round(x)) for x in np.linspace(d, n - 1, 24)]
    elif a.set == "intphys1_uni":
        rows = list(csv.DictReader(open(f"{IP1}/index.csv")))
        key = lambda r: f"{r['block_id']}_{r['pair_id']}"
        path = lambda r: "/".join(r["video_id"].split("_")) + "/scene"
        for r in rows:
            d = int(r["d"])
            r["files"] = [f"{a.root}/{path(r)}/scene_{t + 1:03d}.png" for t in
                          [int(round(x)) for x in np.linspace(0, d - 1, 9)] + [int(round(x)) for x in np.linspace(d, 99, 24)]]
    else:
        rows = list(csv.DictReader(open(f"{REPO}/data_csv/vobench_v1_gravity/index.csv")))
        key = lambda r: r["block_id"]
        path = lambda r: "gravity/" + r["file_name"]
    g = collections.OrderedDict()
    for r in rows:
        g.setdefault(key(r), []).append(dict(r, path=path(r)))
    return [(k, sorted(v, key=lambda r: r["plausible"] != "1")) for k, v in g.items()]


if a.merge:                                   # shard 합치기 + matched 쌍 / gravity 고른 미래
    S, P = {}, {}
    for f in sorted(glob.glob(f"{a.out}/{tag}_shard*.json")):
        d = json.load(open(f)); S.update(d["surprise"]); P.update(d.get("pixel_copy", {}))
    G = groups(); n_vid = sum(len(v) for _, v in G)
    print(f"{tag}: 영상 {len(S)} / {n_vid}")
    if a.set == "intphys2_late":                                             # 문맥 키가 영상마다라 (block_id, pair_id) 로 다시 짝짓는다
        pr = collections.defaultdict(list)
        for k, vs in G:
            pr[(vs[0]["block_id"], vs[0]["pair_id"])] += vs
        G = list(pr.items())
    if a.set in ("vobench", "intphys1_uni", "intphys2_late"):
        res = collections.defaultdict(list)
        for k, vs in G:
            pos = [v for v in vs if v["plausible"] == "1"]; imp = [v for v in vs if v["plausible"] == "0"]
            if len(pos) != 1 or len(imp) != 1 or pos[0]["video_id"] not in S or imp[0]["video_id"] not in S: continue
            sp, si = S[pos[0]["video_id"]], S[imp[0]["video_id"]]
            c = 1.0 if si > sp else 0.5 if si == sp else 0.0
            q = pos[0]
            grps = (("all", q["group"], f"{q['group']}|{q['block_type']}") if a.set == "intphys1_uni" else
                    ("all", q["condition"], f"occluded={q['occluded']}", f"d_src={q['d_src']}") if a.set == "intphys2_late" else
                    ("all", q["objset"], f"{q['objset']}|{q['occlusion']}", f"{q['objset']}|{q['occlusion']}|{q['violation']}"))
            for grp in grps:
                res[grp].append(c)
        for grp in sorted(res): print(f"  {grp:40s} {100 * np.mean(res[grp]):5.1f}  (n {len(res[grp])})")
        json.dump({g_: [100 * float(np.mean(v)), len(v)] for g_, v in res.items()}, open(f"{a.out}/{tag}_pairs.json", "w"), indent=1)
    else:
        res = collections.defaultdict(list)
        for k, vs in G:
            if not all(v["video_id"] in S for v in vs): continue
            pick = min(vs, key=lambda v: S[v["video_id"]])
            for grp in ("all", vs[0]["objset"], f"{vs[0]['objset']}|{vs[0]['law']}"):
                res[grp].append(float(pick["plausible"] == "1"))
        for grp in sorted(res): print(f"  {grp:30s} 고른 미래 = 진짜 {100 * np.mean(res[grp]):5.1f}  (n {len(res[grp])})")
        json.dump({g_: [100 * float(np.mean(v)), len(v)] for g_, v in res.items()}, open(f"{a.out}/{tag}_pick.json", "w"), indent=1)
    sys.exit(0)

import torch, torch.nn.functional as F                                        # noqa: E402
import torchvision                                                            # noqa: E402
from PIL import Image                                                         # noqa: E402
import cosmos_predict1.autoregressive.inference.world_generation_pipeline as WGP   # noqa: E402

WGP.create_inference_config = functools.partial(WGP.create_inference_config, batch_size=a.bs)   # KV 캐시를 batch 크기로
torch._C._jit_set_texpr_fuser_enabled(False)
pipe = WGP.ARBaseGenerationPipeline(inference_type="base", checkpoint_dir=a.checkpoint_dir, checkpoint_name="Cosmos-Predict1-4B",
                                    disable_diffusion_decoder=True, offload_guardrail_models=False, offload_diffusion_decoder=False,
                                    offload_network=False, offload_tokenizer=False, disable_guardrail=True, parallel_size=1)
M = pipe.model
VT = M.tokenizer.video_tokenizer
if hasattr(VT, "tokenizer_module"):                                           # 공식 래퍼와 같은 implicit_codebook (FSQ levels 8,8,8,5,5,5 → 6 차원)
    CB = VT.tokenizer_module.quantizer.implicit_codebook.cuda()
else:                                                                         # JIT 토크나이저면 같은 FSQ 를 새로 지어 코드북만 꺼낸다 (가중치 없음)
    from cosmos_predict1.autoregressive.tokenizer.quantizers import FSQuantizer
    CB = FSQuantizer(levels=[8, 8, 8, 5, 5, 5]).implicit_codebook.cuda()
print("codebook", tuple(CB.shape), type(VT).__name__, flush=True)
N_FUT_TOK = (LAT[0] - 2) * LAT[1] * LAT[2]                                    # 7680


def load(v):
    """index 행 → (3, 33, 640, 1024) [-1, 1]"""
    if v.get("mp4_frames"):
        import decord
        arr = decord.VideoReader(v["path"], num_threads=4).get_batch(v["mp4_frames"]).asnumpy()
    else:
        files = v.get("files") or [f"{a.root}/{v['path']}/{i:06d}.png" for i in IDX]
        arr = np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files])
    x = torch.from_numpy(arr).permute(0, 3, 1, 2).float().cuda() / 255
    if a.crop:                                                                 # 공식 resize_input (비율 유지 + 가운데 자르기)
        s = max(W / x.shape[-1], H / x.shape[-2])
        x = torchvision.transforms.functional.resize(x, [int(np.ceil(s * x.shape[-2])), int(np.ceil(s * x.shape[-1]))])
        x = torchvision.transforms.functional.center_crop(x, [H, W])
    else:
        x = F.pad(F.interpolate(x, size=(H, H), mode="bilinear", align_corners=False, antialias=True), ((W - H) // 2, (W - H) // 2))
    return (x * 2 - 1).permute(1, 0, 2, 3)


def tokens(videos):
    """(B, 3, 33, H, W) → (B, 5·40·64) 토큰 index (특수 토큰 · offset 없는 경우만 — 아래 assert)"""
    out = []                                                                   # 한 영상씩 (33 장 × 640 × 1024 를 8 개 한꺼번에 넣으면 24 GB 에서 OOM)
    for v in videos:
        d, bd = M.tokenizer.tokenize(data_batch={"video": v[None]})
        out.append(d)
    return torch.cat(out), bd


def codes(ind):
    return CB[ind.long()].float()


G = groups()
G = G[a.shard::a.nshard]
if a.limit: G = G[: a.limit]
os.makedirs(a.out, exist_ok=True)
outf = f"{a.out}/{tag}_shard{a.shard}.json"
done = json.load(open(outf)) if os.path.exists(outf) else {"surprise": {}, "ctx_key": {}, "pixel_copy": {}, "sec": []}
G = [g for g in G if not all(v["video_id"] in done["surprise"] for v in g[1])]
print(f"[{tag} shard {a.shard}/{a.nshard}] 문맥 {len(G)} (영상 {sum(len(v) for _, v in G)}) · batch {a.bs}", flush=True)
pred_store = {}
for b0 in range(0, len(G), a.bs):
    t0 = time.time()
    chunk = G[b0: b0 + a.bs]
    vids = {v["video_id"]: load(v).cpu() for _, vs in chunk for v in vs}   # CPU 에 (gravity 는 block 마다 미래 최대 10 개 → GPU 에 다 두면 OOM, 10-21)
    ctx = torch.stack([vids[vs[0]["video_id"]] for _, vs in chunk]).cuda()
    ctx[:, :, 9:] = ctx[:, :, 8:9]                                             # 공식과 같이 문맥 뒤는 마지막 장 반복
    if len(chunk) < a.bs:                                                      # 마지막 batch 는 채워서 (캐시 · 컴파일 모양 고정)
        ctx = torch.cat([ctx, ctx[-1:].expand(a.bs - len(chunk), -1, -1, -1, -1)])
    with torch.inference_mode():
        d, bd = tokens(ctx)
        d = d.cuda()
        assert d.shape[1] == np.prod(LAT), f"특수 토큰이 있다: {d.shape}"
        prompt = d[:, : 2 * LAT[1] * LAT[2]]
        seq, _ = M.generate(prompt_tokens=prompt, max_gen_len=N_FUT_TOK, temperature=0.0, top_k=None, top_p=None, echo=False,   # 공식 temperature 0 → sample_top_k 의 argmax 분기 (10-21: top_k 1 은 동점일 때 무작위라 바꿈)
                            seed=0, compile_sampling=True, verbose=False, stop_tokens=set())
        pred = seq[:, -N_FUT_TOK:]                                            # (B, 7680) 미래 latent 2–4
        allv = list(vids)
        tgt = []
        for i in range(0, len(allv), a.bs):
            vv = torch.stack([vids[v] for v in allv[i: i + a.bs]]).cuda()
            dt, _ = tokens(vv); tgt.append(dt.cuda()[:, -N_FUT_TOK:])
        tgt = torch.cat(tgt)
    for j, (k, vs) in enumerate(chunk):
        p = codes(pred[j])
        pred_store[k] = pred[j].to(torch.int32).cpu().numpy()
        for v in vs:
            i = allv.index(v["video_id"])
            done["surprise"][v["video_id"]] = float((p - codes(tgt[i])).abs().mean())
            done["ctx_key"][v["video_id"]] = k
            x = vids[v["video_id"]]                                            # 픽셀 복사 기준선 (문맥 마지막 장 vs 미래, 같은 전처리)
            done["pixel_copy"][v["video_id"]] = float((x[:, 9:] - x[:, 8:9]).abs().mean())
    done["sec"].append(round(time.time() - t0, 1))
    json.dump(done, open(outf, "w"))
    np.savez_compressed(f"{a.out}/{tag}_pred_shard{a.shard}.npz", **pred_store)
    print(f"  batch {b0 // a.bs + 1}/{(len(G) + a.bs - 1) // a.bs}: 문맥 {len(chunk)} · 영상 {len(allv)} · {done['sec'][-1]} s", flush=True)
print("끝", outf)
