"""Cosmos-1.0-Autoregressive-4B 직관물리 채점 — VOBench late · IntPhys 2 Main.

surprise = 문맥 토큰을 주고 teacher forcing 으로 계산한 **미래 토큰 평균 NLL** (nats/token).
          불가능 영상의 surprise 가 가능 영상보다 크면 정답 (쌍 정확도, chance 50 %).

입력 규약 (근거는 z_research/Benchmarks/COSMOS_AR4B_2026-10-30.md)
  * 33 프레임 = 1 + 8*4 → 잠재 5 칸 (토크나이저 시간 8 배, 첫 프레임 따로)
  * 640x1024 공식 격자 (잠재 40x64). 정사각 512 프레임은 640x640 으로 키워 가운데 두고 좌우
    -1 (검정) 패딩. NLL 평균은 **콘텐츠 칸 (40x40) 만**. 패딩 칸 NLL 은 따로 남긴다
  * 문맥 = 잠재 앞 3 칸 (프레임 0..16), 미래 = 잠재 뒤 2 칸 (프레임 17..32, 토큰 2x40x64)
  * VOBench: 원본 96 장에서 [0, 0, 3, 6, ..., 93] (첫 프레임을 한 번 더 — 인과 토크나이저의 단독
    첫 칸). 문맥 = 원본 0..45, 미래 = 원본 48..93 → V-JEPA c16t32 와 같은 시간 구간
  * IntPhys 2: 60 fps 에서 10 장 간격 (6 fps, 공식 frame_step 10). 33 장 창을 8 장 간격으로
    밀어 (마지막 창은 영상 끝에 맞춤) 창마다 미래 NLL. 영상 surprise = 창 평균 (avg) / 최대 (max)

실행
  python z_research/scripts/analysis/cosmos_ar_eval.py --bench vobench_late  --shard 0 --nshard 1
  python z_research/scripts/analysis/cosmos_ar_eval.py --bench intphys2_main --shard 0 --nshard 2
  python z_research/scripts/analysis/cosmos_ar_eval.py --bench <같음> --score     # 합치고 채점
"""
import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)

OUT_ROOT = os.path.join(REPO, "cvpr/results/eval/cosmos")
VOB_INDEX = os.path.join(REPO, "data_csv/vobench_v1_late/index.csv")
VOB_FRAMES = "/workspace/dataset/benchmarks/VOBench_v1"
IP2_ROOT = "/workspace/dataset/benchmarks/IntPhys2"

N_FRAMES, N_CTX_LATENT = 33, 3
IP2_FRAME_STEP, IP2_WIN_STRIDE = 10, 8

CONFIG = {
    "model": "Cosmos-1.0-Autoregressive-4B (base, video_to_video, 3D RoPE, no BOV)",
    "tokenizer": "Cosmos-1.0-Tokenizer-DV8x16x16 (encoder.jit bf16, FSQ 64000)",
    "dtype": "bfloat16",
    "surprise": "mean -log p(future token | all previous tokens), teacher forcing, nats/token",
    "n_frames": N_FRAMES, "latent_shape": [5, 40, 64], "n_ctx_latent": N_CTX_LATENT,
    "resolution": "640x1024 letterbox: 512 square -> 640x640 centered, black (-1) pad left/right",
    "nll_tokens": "content cells only (40x40 of 40x64); padded-cell NLL kept separately",
    "vobench": "raw frames [0,0,3,...,93] (stride 3, first frame duplicated); ctx raw 0..45, future raw 48..93; matched pairing (block_id, pair_id)",
    "intphys2": f"frame_step {IP2_FRAME_STEP} (6 fps), 33-frame windows every {IP2_WIN_STRIDE} frames (+ last aligned to end); video surprise = mean (avg) / max over windows; pairs = (SceneIndex, 1|2)",
}


def out_dir(bench):
    return os.path.join(OUT_ROOT, f"{bench}_cosmos_ar4b")


# ----------------------------------------------------------------------------- 데이터
def vobench_items():
    import pandas as pd
    df = pd.read_csv(VOB_INDEX, dtype={"block_id": str, "pair_id": str})
    return [dict(r) for _, r in df.iterrows()]


def vobench_frames(item):
    from PIL import Image
    idx = [0] + list(range(0, 96, 3))
    d = os.path.join(VOB_FRAMES, item["file_name"])
    return np.stack([np.array(Image.open(f"{d}/{i:06d}.png").convert("RGB")) for i in idx])


def intphys2_items():
    import pandas as pd
    df = pd.read_csv(os.path.join(IP2_ROOT, "Main", "metadata.csv"))
    return [dict(r) for _, r in df.iterrows()]


def intphys2_windows(n):
    if n < N_FRAMES:
        return [0]
    starts = list(range(0, n - N_FRAMES + 1, IP2_WIN_STRIDE))
    if starts[-1] != n - N_FRAMES:
        starts.append(n - N_FRAMES)
    return starts


def intphys2_frames(item):
    import decord
    vr = decord.VideoReader(os.path.join(IP2_ROOT, "Main", item["file_name"]), num_threads=4)
    idx = np.arange(0, len(vr), IP2_FRAME_STEP)
    fr = vr.get_batch(idx).asnumpy()
    if len(fr) < N_FRAMES:                                      # 짧으면 마지막 프레임 반복 (기록)
        fr = np.concatenate([fr, np.repeat(fr[-1:], N_FRAMES - len(fr), 0)])
    return fr, len(idx)


# ----------------------------------------------------------------------------- 실행
def run(bench, shard, nshard, rep=""):
    from analysis.cosmos_ar import CosmosTokenizer, frames_to_input, load_ar, token_nll
    od = out_dir(bench)
    os.makedirs(od, exist_ok=True)
    path = os.path.join(od, f"per_video{rep}_shard{shard}of{nshard}.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["video_id"] for l in open(path)}
    items = vobench_items() if bench == "vobench_late" else intphys2_items()
    key = "video_id" if bench == "vobench_late" else "name"
    # 같은 쌍은 반드시 한 프로세스에서 — bf16 토크나이저는 프로세스가 바뀌면 토큰 ~15 % 가 달라진다
    # (같은 프로세스 안에서는 결정적). 그래서 VOBench 는 block, IntPhys 2 는 SceneIndex 로 나눈다.
    grp = (lambda it: int(it["block_id"])) if bench == "vobench_late" else (lambda it: int(it["SceneIndex"]))
    items = [it for it in items if grp(it) % nshard == shard and str(it[key]) not in done]
    print(f"[{bench}] shard {shard}/{nshard}: {len(items)} to do ({len(done)} done)", flush=True)
    if not items:
        return
    model, res = load_ar()
    print("strict load:", res, flush=True)
    tk = CosmosTokenizer()
    t0 = time.time()
    with open(path, "a") as f:
        for n, it in enumerate(items):
            if bench == "vobench_late":
                clips, n_src = [vobench_frames(it)], 96
                starts = [0]
            else:
                fr, n_src = intphys2_frames(it)
                starts = intphys2_windows(len(fr))
                clips = [fr[s:s + N_FRAMES] for s in starts]
            wins = []
            for clip in clips:
                vid, valid = frames_to_input(clip, "pad")
                idx, agree = tk.encode(vid)
                tok = idx.reshape(1, -1).cuda()
                nll = token_nll(model, tok, tuple(idx.shape[1:]), N_CTX_LATENT)[0]
                v = valid.cuda()
                wins.append({
                    "nll": float(nll[:, v].mean()),
                    "nll_per_latent": [float(nll[i][v].mean()) for i in range(nll.shape[0])],
                    "nll_max_latent": float(max(float(nll[i][v].mean()) for i in range(nll.shape[0]))),
                    "nll_pad_cells": float(nll[:, ~v].mean()) if (~v).any() else None,
                    "tok_agree": agree,
                    "ctx_tokens_hash": int((idx[0, :N_CTX_LATENT].flatten().long()
                                            * torch.arange(1, 1 + idx[0, :N_CTX_LATENT].numel(),
                                                           device=idx.device)).sum()),
                })
            rec = {"video_id": str(it[key]), "n_src_frames": int(n_src), "window_starts": starts,
                   "windows": wins,
                   "surprise_avg": float(np.mean([w["nll"] for w in wins])),
                   "surprise_max": float(np.max([w["nll"] for w in wins]))}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 20 == 0:
                el = time.time() - t0
                print(f"  {n + 1}/{len(items)}  {el / (n + 1):.2f}s/video  surprise {rec['surprise_avg']:.4f}",
                      flush=True)
    print(f"[{bench}] shard {shard} done {time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------- 채점
def load_records(bench, rep=""):
    recs = {}
    for p in sorted(glob.glob(os.path.join(out_dir(bench), f"per_video{rep}_shard*.jsonl"))):
        for l in open(p):
            r = json.loads(l)
            recs[r["video_id"]] = r
    return recs


def _acc(pairs):
    hits = [1.0 if i > p else (0.5 if i == p else 0.0) for p, i in pairs]
    return {"acc": float(np.mean(hits)) if hits else float("nan"), "n_pair": len(hits),
            "n_ties": int(sum(h == 0.5 for h in hits))}


def score_vobench(recs, col):
    items = vobench_items()
    by = defaultdict(list)
    for it in items:
        by[(it["block_id"], it["pair_id"])].append(it)
    rows, ctx_mismatch = [], 0
    for (b, pid), its in sorted(by.items()):
        pos = [x for x in its if int(x["plausible"]) == 1]
        neg = [x for x in its if int(x["plausible"]) != 1]
        assert len(pos) == 1 and len(neg) == 1, (b, pid)
        rp, rn = recs.get(pos[0]["video_id"]), recs.get(neg[0]["video_id"])
        if rp is None or rn is None:
            continue
        ctx_mismatch += int(rp["windows"][0]["ctx_tokens_hash"] != rn["windows"][0]["ctx_tokens_hash"])
        rows.append({"block_id": b, "pair_id": pid, "condition": pos[0]["condition"], "objset": pos[0]["objset"],
                     "violation": pos[0]["violation"], "k": int(pos[0]["k"]),
                     "s_pos": rp[col], "s_imp": rn[col]})
    out = {"overall": _acc([(r["s_pos"], r["s_imp"]) for r in rows]),
           "ctx_token_mismatch_pairs": ctx_mismatch}
    for ax in ["objset", "condition", "violation", "k"]:
        g = defaultdict(list)
        for r in rows:
            g[r[ax]].append((r["s_pos"], r["s_imp"]))
        out[f"by_{ax}"] = {str(k): _acc(v) for k, v in sorted(g.items())}
    # 방향별 (A = pos_a vs imp_ab, B = pos_b vs imp_ba) — appearance bias 를 숨기지 않기 위함
    g = defaultdict(list)
    for r in rows:
        g[f"{r['condition']}|{r['pair_id']}"].append((r["s_pos"], r["s_imp"]))
    out["by_condition_direction"] = {k: _acc(v) for k, v in sorted(g.items())}
    return out, rows


def score_intphys2(recs, col):
    import pandas as pd
    from analysis.intphys2.metrics import pairwise_accuracy, pairwise_accuracy_breakdown, single_video_auroc
    rows = []
    for it in intphys2_items():
        r = recs.get(str(it["name"]))
        if r is None:
            continue
        rows.append({"scene_index": int(it["SceneIndex"]), "pair_id": int(str(it["type"]).split("_")[0]),
                     "is_impossible": str(it["type"]).endswith("Impossible"), "condition": it["condition"],
                     "difficulty": it["Difficulty"], "camera": it["Camera"], "surprise": r[col]})
    df = pd.DataFrame(rows)
    return {"overall": pairwise_accuracy(df, "surprise"),
            "by_condition": pairwise_accuracy_breakdown(df, "surprise", ["condition"]),
            "by_difficulty": pairwise_accuracy_breakdown(df, "surprise", ["difficulty"]),
            "by_camera": pairwise_accuracy_breakdown(df, "surprise", ["camera"]),
            "single_video_auroc": single_video_auroc(df, "surprise")}, rows


def score(bench):
    recs = load_records(bench)
    summ = {"bench": bench, "n_videos_scored": len(recs), "config": CONFIG,
            "uniform_nll": float(np.log(64000)),
            "mean_surprise": float(np.mean([r["surprise_avg"] for r in recs.values()])),
            "tok_agree_min": float(min(w["tok_agree"] for r in recs.values() for w in r["windows"]))}
    cols = ["surprise_avg"] if bench == "vobench_late" else ["surprise_avg", "surprise_max"]
    for col in cols:
        if bench == "vobench_late":
            s, rows = score_vobench(recs, col)
        else:
            s, rows = score_intphys2(recs, col)
        summ[col] = s
        json.dump(rows, open(os.path.join(out_dir(bench), f"pairs_{col}.json"), "w"), indent=0)
    if bench == "vobench_late":     # 미래 잠재 칸별 (latent 3 = 원본 48..69, latent 4 = 72..93)
        for li in range(2):
            col = f"nll_latent{N_CTX_LATENT + li}"
            for r in recs.values():
                r[col] = r["windows"][0]["nll_per_latent"][li]
            summ[col] = score_vobench(recs, col)[0]
    # 재실행 안정성: 별도 프로세스로 한 번 더 돈 기록 (--rep _rep) 이 있으면 같은 쌍의 판정을 대조
    rep = load_records(bench, "_rep")
    if rep:
        common = {k: v for k, v in recs.items() if k in rep}
        d = np.array([rep[k]["surprise_avg"] - common[k]["surprise_avg"] for k in common])
        st = {"n_videos": len(common), "surprise_abs_diff_mean": float(np.abs(d).mean()),
              "surprise_abs_diff_max": float(np.abs(d).max())}
        if bench == "vobench_late":
            a_rows = score_vobench(common, "surprise_avg")[1]
            b_rows = score_vobench({k: rep[k] for k in common}, "surprise_avg")[1]
        else:
            a_rows = score_intphys2(common, "surprise_avg")[1]
            b_rows = score_intphys2({k: rep[k] for k in common}, "surprise_avg")[1]
        if bench == "vobench_late":
            da = {(r["block_id"], r["pair_id"]): r["s_imp"] > r["s_pos"] for r in a_rows}
            db = {(r["block_id"], r["pair_id"]): r["s_imp"] > r["s_pos"] for r in b_rows}
            ks = [k for k in da if k in db]
            st.update({"n_pairs": len(ks), "pair_decision_agree": float(np.mean([da[k] == db[k] for k in ks])),
                       "acc_run1": float(np.mean([da[k] for k in ks])), "acc_run2": float(np.mean([db[k] for k in ks]))})
        summ["rerun_stability"] = st
    json.dump(summ, open(os.path.join(out_dir(bench), "summary.json"), "w"), indent=1)
    json.dump(recs, open(os.path.join(out_dir(bench), "per_video.json"), "w"))
    print(json.dumps({k: v for k, v in summ.items() if k != "config"}, indent=1)[:6000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["vobench_late", "intphys2_main"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--rep", default="", help="재실행 안정성 검사용 접미사 (예: _rep)")
    a = ap.parse_args()
    if a.score:
        score(a.bench)
    else:
        torch.set_num_threads(8)
        run(a.bench, a.shard, a.nshard, a.rep)
