"""가림막에 완전히 가려진 문맥 튜블릿에서 물체 모양 · 색이 읽히나 — context encoder 위 attentive probe (2026-10-29).

입력: VOBench_v1 가능 영상 (pos_a → 물체 A, pos_b → 물체 B, 빈 장면 제외), 문맥 16 장 (원본 0, 3, …, 45) 을
  **context encoder (online `encoder`, ViT-H)** 에 한 번에 → (8 튜블릿, 256, 1280).
완전히 가려진 튜블릿 = 튜블릿의 두 장 모두 label.json hidden ≥ 0.999 (samples 문자열 'H' 와 같은지도 검사).
  late k1 은 가려진 장이 1 장이라 해당 튜블릿 없음 → late k2 · k3 (튜블릿 7), early (2 / 2-3 / 2-4), mid k2 (5) · k3 (4-6).
probe (head 2 개 = 모양 · 색, 서로 독립): 입력 (T, D) = 가려진 튜블릿들의 공간 평균 토큰, T 는 영상마다 다름 (마스크).
  attentive pooler (LN → 학습 query 1 개 cross-attn 16 head → LN → Linear), AdamW, block 단위 50/50 (모양으로 층화), split seed 5 개.
대조:
  vis_same  = 같은 영상의 가림 전 보이는 튜블릿 (튜블릿 0-1) — 같은 영상 · 같은 표본 수의 천장
  vis_cond  = 가림 없는 조건 (visible) 의 같은 자리 튜블릿 — 가림막 자체가 없는 천장
  tokens    = 가려진 튜블릿의 공간 토큰 전부 (T·256, D) — 공간 평균이 정보를 지우는지
출력 z_analysis/occluded_identity_probe/result.json

  CUDA_VISIBLE_DEVICES=0 $PY z_research/scripts/analysis/vobench_occluded_identity_probe.py extract --feat_dir <scratch>/occ_id
  CUDA_VISIBLE_DEVICES=0 $PY z_research/scripts/analysis/vobench_occluded_identity_probe.py probe   --feat_dir <scratch>/occ_id
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
FR = Path("/local_datasets/world/testbed/VOBench_v1")
INDEX = REPO / "data_csv/vobench_v1/index.csv"
CK = REPO / "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth"
OUT = REPO / "z_analysis/occluded_identity_probe"
NT, S, D = 8, 256, 1280
dev = torch.device("cuda")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)


def clips():
    out = []
    for r in csv.DictReader(INDEX.open()):
        if r["variant"] not in ("pos_a", "pos_b"):
            continue
        obj = r["obj_a"] if r["variant"] == "pos_a" else r["obj_b"]
        if obj == "empty":
            continue
        out.append({**r, "obj": obj, "arm": "A" if r["variant"] == "pos_a" else "B"})
    return out


def hidden_tubelets(r, lab):
    """두 장 모두 완전히 가려진 문맥 튜블릿 번호."""
    L = lab["labels"][r["arm"]]; smp = r["samples"].split("|")[0]
    hid = [L[3 * s]["hidden"] >= 0.999 for s in range(16)]
    assert all(h == (c == "H") for h, c in zip(hid, smp)), (r["video_id"], smp)
    return [i for i in range(NT) if hid[2 * i] and hid[2 * i + 1]]


def extract(feat_dir):
    from analysis.intphys2.model import build_from_config
    feat_dir = Path(feat_dir); feat_dir.mkdir(parents=True, exist_ok=True)
    R = clips()
    m = {"arch_name": "vit_huge", "checkpoint": str(CK), "img_size": 256, "patch_size": 16, "tubelet_size": 2, "use_rope": True,
         "uniform_power": False, "dual_encoder": True, "context_encoder_key": "encoder", "target_encoder_key": "target_encoder",
         "predictor": {"embed_dim": 384, "depth": 12, "num_heads": 12, "num_mask_tokens": 10}, "dtype": "bfloat16", "window_size": 16}
    enc = build_from_config(m, dev).context_encoder
    tok = np.lib.format.open_memmap(feat_dir / "tokens.npy", "w+", np.float16, (len(R), NT, S, D))
    meta, labs = [], {}

    def load(r):
        d = FR / r["file_name"]
        x = torch.from_numpy(np.stack([np.asarray(__import__("PIL.Image").Image.open(d / f"{3 * k:06d}.png").convert("RGB")) for k in range(16)]))
        x = x.permute(3, 0, 1, 2).float() / 255.0
        x = F.interpolate(x.transpose(0, 1), size=(256, 256), mode="bilinear", align_corners=False, antialias=False).transpose(0, 1)
        return (x - MEAN) / STD

    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(16); B = 16
    for k in range(0, len(R), B):
        batch = R[k:k + B]
        xs = torch.stack(list(pool.map(load, batch))).to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            z = enc(xs)                                                              # (B, 8·256, 1280)
        tok[k:k + len(batch)] = z.float().view(len(batch), NT, S, D).cpu().numpy().astype(np.float16)
        for r in batch:
            b = (FR / r["file_name"]).parent
            if b not in labs:
                labs[b] = json.load(open(b / "label.json"))
            meta.append({"video_id": r["video_id"], "block": r["source_block"], "objset": r["objset"], "occ": r["occlusion"],
                         "violation": r["violation"], "motion": r["motion"], "shape": r["obj"].split("/")[0], "color": r["obj"].split("/")[1],
                         "hidden": hidden_tubelets(r, labs[b])})
        if k % 160 == 0:
            print(f"extract {k + len(batch)}/{len(R)}", flush=True)
    tok.flush()
    json.dump(meta, open(feat_dir / "meta.json", "w"))
    print(feat_dir)


class AttnProbe(nn.Module):
    def __init__(self, n_cls, d=D, heads=16):
        super().__init__()
        self.ln = nn.LayerNorm(d); self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_cls))

    def forward(self, x, pad):                                                       # x (B, T, D), pad (B, T) True = 빈 자리
        x = self.ln(x)
        y, _ = self.attn(self.q.expand(len(x), -1, -1), x, x, key_padding_mask=pad)
        return self.out(y[:, 0])


def fit_eval(X, pad, y, tr, n_cls, seed, epochs=300, lr=3e-4, wd=0.05, bs=64):
    torch.manual_seed(seed)
    net = AttnProbe(n_cls).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    itr, ite = np.where(tr)[0], np.where(~tr)[0]
    yt = torch.tensor(y, device=dev)
    for ep in range(epochs):
        perm = np.random.default_rng(seed * 1000 + ep).permutation(itr)
        for j in range(0, len(perm), bs):
            b = perm[j:j + bs]
            loss = F.cross_entropy(net(X[b], pad[b]), yt[b]); opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred = torch.cat([net(X[ite[j:j + 256]], pad[ite[j:j + 256]]).argmax(1) for j in range(0, len(ite), 256)]).cpu().numpy()
        ptr = torch.cat([net(X[itr[j:j + 256]], pad[itr[j:j + 256]]).argmax(1) for j in range(0, len(itr), 256)]).cpu().numpy()
    return pred, float((ptr == y[itr]).mean())


def block_split(groups, strat, seed):
    rng = np.random.default_rng(seed); trb = set()
    for s in np.unique(strat):
        ub = np.unique(groups[strat == s]); rng.shuffle(ub); trb |= set(ub[: len(ub) // 2])
    return np.array([g in trb for g in groups])


def probe(feat_dir, seeds=5):
    feat_dir = Path(feat_dir)
    meta = json.load(open(feat_dir / "meta.json")); tok = np.load(feat_dir / "tokens.npy", mmap_mode="r")
    OUT.mkdir(parents=True, exist_ok=True)
    timing = lambda m: m["occ"].split("_")[0]
    # 대조용 visible 조건 튜블릿 자리: 가림 조건들의 가려진 튜블릿 자리 분포와 같게 (2..7) — 여기선 전부 (2..7) 평균 대신 영상별 튜블릿 7 하나
    res = {}
    for objset, targets in (("primitive", ("shape", "color")), ("realistic", ("shape",))):
        sets = {}
        idx_h = [i for i, m in enumerate(meta) if m["objset"] == objset and m["hidden"]]
        sets["hidden_mean"] = (idx_h, [meta[i]["hidden"] for i in idx_h], "mean")
        sets["hidden_tokens"] = (idx_h, [meta[i]["hidden"] for i in idx_h], "tokens")
        sets["vis_same_mean"] = (idx_h, [[0, 1] for _ in idx_h], "mean")
        idx_v = [i for i, m in enumerate(meta) if m["objset"] == objset and m["occ"] == "visible"]
        sets["vis_cond_mean"] = (idx_v, [[7] for _ in idx_v], "mean")
        res[objset] = {}
        for name, (idx, tubs, mode) in sets.items():
            Tm = max(len(t) for t in tubs) * (S if mode == "tokens" else 1)
            X = torch.zeros(len(idx), Tm, D); pad = torch.ones(len(idx), Tm, dtype=torch.bool)
            for j, (i, t) in enumerate(zip(idx, tubs)):
                a = torch.from_numpy(np.asarray(tok[i, t], dtype=np.float32))            # (len t, 256, D)
                a = a.mean(1) if mode == "mean" else a.reshape(-1, D)
                X[j, :len(a)] = a; pad[j, :len(a)] = False
            X, pad = X.to(dev), pad.to(dev)
            M = [meta[i] for i in idx]
            groups = np.array([m["block"] for m in M]); tim = np.array([timing(m) for m in M])
            res[objset][name] = {"n_clips": len(idx), "T_max": Tm}
            for tgt in targets:
                cls = sorted({m[tgt] for m in M}); y = np.array([cls.index(m[tgt]) for m in M])
                accs, tr_accs, per_tim = [], [], {t: [] for t in np.unique(tim)}
                for sd in range(seeds):
                    tr = block_split(groups, np.array([m["shape"] for m in M]), sd)
                    pred, tra = fit_eval(X, pad, y, tr, len(cls), sd)
                    yte, tte = y[~tr], tim[~tr]
                    accs.append(float((pred == yte).mean())); tr_accs.append(tra)
                    for t in per_tim:
                        per_tim[t].append(float((pred[tte == t] == yte[tte == t]).mean()))
                res[objset][name][tgt] = {"classes": cls, "chance": 1 / len(cls), "test_acc_mean": float(np.mean(accs)),
                                          "test_acc_sd": float(np.std(accs)), "train_acc_mean": float(np.mean(tr_accs)),
                                          "n_test_mean": float(np.mean([(~block_split(groups, np.array([m['shape'] for m in M]), s)).sum() for s in range(seeds)])),
                                          "by_timing": {t: {"acc": float(np.mean(v)), "sd": float(np.std(v)),
                                                            "n_test": int(round(np.mean([((~block_split(groups, np.array([m['shape'] for m in M]), s)) & (tim == t)).sum() for s in range(seeds)])))}
                                                        for t, v in per_tim.items()}}
                print(f"{objset:9s} {name:14s} {tgt:5s} test {np.mean(accs):.3f}±{np.std(accs):.3f} (chance {1/len(cls):.3f}, train {np.mean(tr_accs):.2f}) "
                      + " ".join(f"{t} {np.mean(v):.3f}" for t, v in per_tim.items()), flush=True)
            del X, pad; torch.cuda.empty_cache()
    json.dump(res, open(OUT / "result.json", "w"), indent=1)
    print(OUT / "result.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["extract", "probe"]); ap.add_argument("--feat_dir", required=True)
    a = ap.parse_args()
    extract(a.feat_dir) if a.stage == "extract" else probe(a.feat_dir)
