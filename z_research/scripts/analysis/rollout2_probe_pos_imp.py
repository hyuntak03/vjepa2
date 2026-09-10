#!/usr/bin/env python3
"""RollOut_v2 실험 2 — possible/impossible attentive probe 를 h(32프레임)로 학습하고 [z 문맥 ; p 미래] 에 건다.

ledge (낙하 / 부유), wall (정지 / 통과). block 마다 pos·imp 한 쌍, 문맥 16장은 byte-identical.
학습: target encoder 의 32프레임 토큰 (4096 = 문맥 2048 + 미래 2048), 라벨 = possible / impossible. block 단위 50/50 split.
평가 (test block):
  [h_ctx ; h_fut]   학습 분포 그대로 (천장)
  [z ; h_fut]       문맥을 context encoder 출력(LN(z))으로 바꿈 — 문맥 소스 교체만으로 깨지는지
  [z ; p]           ★ predictor 가 만든 미래. pos/imp 쌍의 p 는 동일하므로 정확도는 50 이 정해져 있고,
                    보는 값은 "probe 가 possible 이라 부르는 비율" 과 그 확신이다
  [z ; LN(p)]       p 에 affine-free LN 을 건 변형
probe: 질의 1개 cross-attention (16 head) → LayerNorm → Linear(2). 하네스 attn_probe(num_probe_blocks 1) 와 같은 형태.
학습 문맥 소스도 두 가지: h_ctx (기본) / z (문맥 소스를 평가와 맞춘 변형).

  python z_research/scripts/analysis/rollout2_probe_pos_imp.py --device cuda:0
"""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
CACHE = Path("/local_datasets/world/world_analysis/cache/rollout_v2_vith")
INDEX = ROOT / "data_csv/rollout_v2/index_probe.csv"
OUT = ROOT / "z_research/RollOutV2/exp_results/probe_pos_imp.json"
S, D = 256, 1280


class AttnProbe(nn.Module):
    def __init__(self, d=D, heads=16, n_cls=2):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02); self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d); self.head = nn.Linear(d, n_cls)

    def forward(self, tok):                                   # (B, N, D)
        o, _ = self.attn(self.q.expand(tok.size(0), -1, -1), tok, tok); return self.head(self.norm(o[:, 0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--epochs", type=int, default=25); ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bs", type=int, default=16); ap.add_argument("--seed", type=int, default=0); ap.add_argument("-o", "--out", type=Path, default=OUT)
    a = ap.parse_args(); dev = torch.device(a.device); torch.manual_seed(a.seed)
    idx = list(csv.DictReader(INDEX.open())); vids = [r["video_id"] for r in idx]
    meta = json.loads((CACHE / "meta.json").read_text()); row_of = {v: i for i, v in enumerate(meta["video_ids"])}; rows = np.array([row_of[v] for v in vids])
    sc = np.array([r["scenario"] for r in idx]); plaus = np.array([int(r["plausible"]) for r in idx]); blk = np.array([r["block_id"] for r in idx])
    prim = np.array([float(r["primary"]) for r in idx])
    Z = np.load(CACHE / "ctx_masked.npy", mmap_mode="r"); H = np.load(CACHE / "target.npy", mmap_mode="r"); P = np.load(CACHE / "predictor.npy", mmap_mode="r")

    def load(M, ids, sl=slice(None)):
        o = np.argsort(rows[ids]); out = np.empty((len(ids),) + M[rows[ids[0]], sl].shape, np.float16)
        for k in range(0, len(ids), 32):
            sel = o[k:k + 32]; out[sel] = np.asarray(M[rows[ids[sel]], sl])
        return out

    R = {"meta": dict(probe="1-query cross-attention (16 heads) + LN + Linear(2); trained on target-encoder 32-frame tokens", epochs=a.epochs, lr=a.lr, seed=a.seed), "scenarios": {}}
    for scn in ("ledge", "wall"):
        ids = np.where(sc == scn)[0]; blocks = np.unique(blk[ids]); rng = np.random.RandomState(a.seed)
        # block 단위 50/50, primary 층화
        tr_b, te_b = [], []
        for lv in np.unique(prim[ids]):
            b = np.unique(blk[ids[prim[ids] == lv]]); rng.shuffle(b); h = len(b) // 2; tr_b += list(b[:h]); te_b += list(b[h:])
        tr = ids[np.isin(blk[ids], tr_b)]; te = ids[np.isin(blk[ids], te_b)]
        y = torch.from_numpy(1 - plaus).long()                                          # 0 = possible, 1 = impossible
        t0 = time.time()
        Htr = load(H, tr); Hte = load(H, te); Zte = load(Z, te); Pte = load(P, te); Ztr = load(Z, tr)
        print(f"\n== {scn}: train {len(tr)} clip ({len(tr_b)} block) / test {len(te)} clip   로딩 {time.time()-t0:.0f}s")
        R["scenarios"][scn] = {"n_train": int(len(tr)), "n_test": int(len(te)), "runs": {}}
        for ctx_src in ("h_ctx", "z"):
            Xtr = Htr if ctx_src == "h_ctx" else np.concatenate([Ztr, Htr[:, S * 8:]], 1)
            model = AttnProbe().to(dev); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
            ytr = y[tr]
            for ep in range(a.epochs):
                perm = torch.randperm(len(tr)); tot = cor = 0
                for b in range(0, len(tr), a.bs):
                    kk = perm[b:b + a.bs].numpy(); x = torch.from_numpy(Xtr[kk]).to(dev, torch.float32); yy = ytr[kk].to(dev)
                    lo = model(x); loss = Fn.cross_entropy(lo, yy); opt.zero_grad(); loss.backward(); opt.step()
                    tot += loss.item() * len(kk); cor += (lo.argmax(1) == yy).sum().item()
                if ep % 5 == 4 or ep == a.epochs - 1:
                    print(f"    [{ctx_src}] epoch {ep+1} loss {tot/len(tr):.3f} train acc {cor/len(tr):.3f}")
            model.eval()

            @torch.no_grad()
            def predict(X):
                out = []
                for b in range(0, len(X), 32):
                    out.append(model(torch.from_numpy(X[b:b + 32]).to(dev, torch.float32)).softmax(1)[:, 1].cpu().numpy())
                return np.concatenate(out)                                              # P(impossible)

            Pln = Pte.astype(np.float32); Pln = (Pln - Pln.mean(-1, keepdims=True)) / (Pln.std(-1, keepdims=True) + 1e-6)
            tests = {"[h_ctx ; h_fut]": Hte, "[z ; h_fut]": np.concatenate([Zte, Hte[:, S * 8:]], 1),
                     "[z ; p]": np.concatenate([Zte, Pte], 1), "[z ; LN(p)]": np.concatenate([Zte, Pln.astype(np.float16)], 1),
                     "[h_ctx ; p]": np.concatenate([Hte[:, :S * 8], Pte], 1)}
            yte = (1 - plaus[te]); rec = {}
            for name, X in tests.items():
                pi = predict(X); pred = (pi > 0.5).astype(int)
                acc = float((pred == yte).mean()); acc_pos = float((pred[yte == 0] == 0).mean()); acc_imp = float((pred[yte == 1] == 1).mean())
                # p 는 pos/imp 쌍에서 동일 → block 단위로 "possible 이라 부른 비율"
                pos_ids = te[yte == 0]; frac_pos = float((pred[yte == 0] == 0).mean()); conf = float(np.abs(pi[yte == 0] - 0.5).mean() * 2)
                rec[name] = dict(acc=acc, acc_on_possible=acc_pos, acc_on_impossible=acc_imp, frac_called_possible_on_pos_clips=frac_pos,
                                 mean_P_impossible_on_pos_clips=float(pi[yte == 0].mean()), mean_P_impossible_on_imp_clips=float(pi[yte == 1].mean()), confidence=conf)
                print(f"    train ctx={ctx_src:<6} test {name:<16} acc {100*acc:5.1f}  (pos {100*acc_pos:5.1f} / imp {100*acc_imp:5.1f})   P(imp): pos클립 {pi[yte==0].mean():.3f}  imp클립 {pi[yte==1].mean():.3f}")
            R["scenarios"][scn]["runs"][f"train_ctx={ctx_src}"] = rec
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(R, indent=1, ensure_ascii=False)); print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
