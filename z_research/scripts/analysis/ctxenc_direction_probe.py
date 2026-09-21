#!/usr/bin/env python3
"""문맥 encoder 가 실제 영상에서 운동 방향 (좌 / 우) 을 담는가 — 정방향으로 학습, 역재생으로 시험.

무엇을 재나: frozen ViT-H 문맥 encoder 에 균등 샘플 32 장을 넣어 (T*N, D) = (4096, 1280) 토큰을 받고,
그 위에 attentive probe 를 붙여 left / right 이진 분류를 **정방향 clip 으로만** 학습한다. 그 다음 같은 probe 에
**같은 held-out clip 을 역재생** (샘플한 32 장의 순서만 뒤집음) 해서 넣는다.
  - encoder 가 운동 방향을 담으면: 역재생 예측이 뒤집힌다 (뒤집힘 비율 ~1, 뒤집힌 라벨 기준 정확도가 정방향과 비슷).
  - 정지 단서 (장면·물체·손 자세) 로 맞히던 것이면: 역재생해도 같은 답 (뒤집힘 비율 ~0).
뒤집힘 비율은 라벨이 필요 없다 (같은 clip 두 방향의 예측 비교) — 파라미터 없는 교차검증 (CLAUDE.md §12).

데이터 (둘 다 폴더 이름이 라벨):
  ssv2  /local_datasets/vlm_direction/ssv2_VP/ssv2_VP_default/{left,right}          361 + 361
  ntu   /local_datasets/vlm_direction/ntu_direction_benchmark/{left,right}          208 + 208  (나머지 6 방향 폴더는 안 쓴다)

세팅 (전부 아래 기본값):
  clip      원본 전체 길이에서 균등 32 장 (np.linspace, 반올림) → 256x256 bilinear (antialias=False, 레포 관례) → ImageNet 정규화
            ⚠️ 좌우 뒤집기 augmentation 은 쓰지 않는다 — 그것이 곧 라벨을 뒤집는다. 시간 jitter 도 없다 (샘플이 결정론적)
            ⚠️ 원본 종횡비를 무시하고 정사각으로 눌러 넣는다 (--spatial full). 가로 운동이 잘리지 않게 하기 위함.
               center crop 은 --spatial crop. 두 전처리 모두 정방향·역방향에 똑같이 걸리므로 "뒤집힘" 판정에는 영향이 없다
  encoder   ViT-H, 문맥 encoder (state_dict 의 `encoder` 키 = online). window_size 32, mask 없음, bfloat16, frozen
  probe     attentive (`src/models/attentive_pooler.AttentiveClassifier`, depth 1, 16 head) — 주. 대조로 tiny (query 1개 + linear, ~2.6k 파라미터)
  split     ssv2 = clip 단위 층화 7:3 (seed 0). ntu = **연기자 (P###) 단위** 3명 중 1명을 test 로 — 같은 사람·장면이 양쪽에 들어가지 않게 (CLAUDE.md §1-5 의 취지)
  토큰 캐시를 디스크에 쓰지 않는다. clip 마다 정방향·역방향을 한 번씩 encoder 에 통과시켜 메모리에만 들고 probe 를 학습한다
            (샘플이 결정론적이라 매 epoch 다시 돌리는 것과 같은 값이다)

출력: z_research/context_encoder_analysis/exp_results/encoder_temporal_dynamics/<ssv2|ntu>/{results.json, RESULTS.md, preds.npz}
      데이터셋 2 개 x {정방향 test, 역방향 test} = 2x2 혼동행렬 4 개.

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/ctxenc_direction_probe.py                       # 두 데이터셋 전부 (GPU 1장)
  $P z_research/scripts/analysis/ctxenc_direction_probe.py --sets ssv2 --smoke 4 # 배관 점검
  $P z_research/scripts/analysis/ctxenc_direction_probe.py --no-encoder --smoke 4  # 모델 없이 배관만 (CPU, 난수 특징)
  bash z_research/scripts/analysis/ctxenc_direction_sbatch.sh                    # SLURM (vll3, GPU 1장)
"""
from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
sys.path.insert(0, str(ROOT))

CLASSES = ["left", "right"]                                   # 라벨 0 = left, 1 = right
SETS = {
    "ssv2": dict(root=Path("/local_datasets/vlm_direction/ssv2_VP/ssv2_VP_default"), split="clip"),
    "ntu": dict(root=Path("/local_datasets/vlm_direction/ntu_direction_benchmark"), split="performer"),
}
OUT_ROOT = ROOT / "z_research/context_encoder_analysis/exp_results/encoder_temporal_dynamics"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
MODEL = dict(                                                  # configs/protocols/models.md `vith` + attn_probe 의 model 블록
    checkpoint=str(ROOT / "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/"
                          "b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth"),
    arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, window_size=32,
    use_rope=True, uniform_power=False,
    dual_encoder=False, context_encoder_key="encoder", target_encoder_key="encoder",   # 문맥 encoder 하나만 올린다
    predictor=dict(embed_dim=384, depth=12, num_heads=12, num_mask_tokens=10),
    dtype="bfloat16",
)
D, NTOK = 1280, 4096                                           # ViT-H: 16 tubelet x 256 = 4096 토큰


# ────────────────────────────────────────────────────────────────── 데이터
class ClipSet(torch.utils.data.Dataset):
    """(3, 32, 256, 256) 정규화 clip 하나. 역재생은 forward 에서 torch.flip 으로 만든다 (같은 32 장, 순서만 반대)."""

    def __init__(self, files, n_frames=32, res=256, spatial="full"):
        self.files, self.n, self.res, self.spatial = files, n_frames, res, spatial

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        import decord
        vr = decord.VideoReader(str(self.files[i]), num_threads=1)
        idx = np.linspace(0, len(vr) - 1, self.n).round().astype(int)
        a = torch.from_numpy(vr.get_batch(list(idx)).asnumpy())          # (T, H, W, 3) uint8
        x = a.permute(3, 0, 1, 2).float() / 255.0                        # (3, T, H, W)
        if self.spatial == "crop":                                       # 짧은변 res → 가운데 res (공식 val 관례)
            h, w = x.shape[-2:]
            s = self.res / min(h, w)
            x = F.interpolate(x.transpose(0, 1), size=(max(self.res, int(round(h * s))), max(self.res, int(round(w * s)))),
                              mode="bilinear", align_corners=False).transpose(0, 1)
            h, w = x.shape[-2:]
            t, l = (h - self.res) // 2, (w - self.res) // 2
            x = x[:, :, t:t + self.res, l:l + self.res].contiguous()
        else:                                                            # full: 원본 전체를 정사각으로 (가로 운동이 안 잘린다)
            x = F.interpolate(x.transpose(0, 1), size=(self.res, self.res),
                              mode="bilinear", align_corners=False).transpose(0, 1).contiguous()
        return (x - MEAN) / STD, i


def file_list(name):
    root = SETS[name]["root"]
    files, labels = [], []
    for c, lab in zip(CLASSES, range(len(CLASSES))):
        fs = sorted((root / c).glob("*.mp4"))
        assert fs, f"{root / c} 에 mp4 가 없다"
        files += fs
        labels += [lab] * len(fs)
    return files, np.array(labels, np.int64)


def split_masks(name, files, labels, seed=0, test_frac=0.3):
    """ssv2: clip 단위 층화. ntu: 연기자 (P###) 단위 — 같은 사람이 train·test 에 걸치지 않게."""
    n = len(files)
    te = np.zeros(n, bool)
    if SETS[name]["split"] == "performer":
        pid = np.array([re.search(r"P(\d+)", f.name).group(1) for f in files])
        for p in sorted(set(pid))[::3]:                                  # 3 명 중 1 명 test (결정론)
            te |= pid == p
        groups = pid
    else:
        rng = np.random.default_rng(seed)
        for lab in range(len(CLASSES)):
            idx = np.where(labels == lab)[0]
            te[rng.permutation(idx)[:int(round(len(idx) * test_frac))]] = True
        groups = np.array([f.stem for f in files])
    return ~te, te, groups


# ────────────────────────────────────────────────────────────────── probe
class TinyAttn(nn.Module):
    """query 하나짜리 attentive pooling + linear (~2.6k 파라미터). 용량 대조군 (RollOutV2 v5 자와 같은 구조)."""

    def __init__(self, d=D, c=len(CLASSES)):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d) / d ** 0.5)
        self.out = nn.Linear(d, c)

    def forward(self, tok):
        a = torch.softmax(tok @ self.q / tok.size(-1) ** 0.5, -1)
        return self.out((a.unsqueeze(-1) * tok).sum(1))


def make_probe(kind, dev):
    if kind == "attentive":
        from src.models.attentive_pooler import AttentiveClassifier
        m = AttentiveClassifier(embed_dim=D, num_heads=16, depth=1, num_classes=len(CLASSES))
    else:
        m = TinyAttn()
    return m.to(dev)


def train_probe(kind, feat, labels, tr, dev, epochs=30, lr=1e-4, wd=0.01, bs=16, seed=0, log=print):
    torch.manual_seed(seed)
    probe = make_probe(kind, dev)
    npar = sum(p.numel() for p in probe.parameters())
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    idx_tr = np.where(tr)[0]
    y = torch.from_numpy(labels)
    for ep in range(epochs):
        probe.train()
        order = np.random.default_rng(seed + ep).permutation(idx_tr)
        tot = cor = 0
        loss_sum = 0.0
        for k in range(0, len(order), bs):
            b = order[k:k + bs]
            x = feat[b].to(dev, non_blocking=True).float()
            t = y[b].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                o = probe(x)
                loss = F.cross_entropy(o.float(), t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach()) * len(b)
            cor += int((o.argmax(-1) == t).sum())
            tot += len(b)
        sch.step()
        if ep % 5 == 0 or ep == epochs - 1:
            log(f"    [{kind}] epoch {ep + 1:2d}/{epochs}  train loss {loss_sum / tot:.3f}  acc {100 * cor / tot:.1f}%")
    return probe, npar


@torch.no_grad()
def predict(probe, feat, dev, bs=16):
    probe.eval()
    out = np.empty(len(feat), np.int64)
    for k in range(0, len(feat), bs):
        x = feat[k:k + bs].to(dev).float()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out[k:k + bs] = probe(x).argmax(-1).cpu().numpy()
    return out


# ────────────────────────────────────────────────────────────────── 특징 추출
def variant(c, mode, idx):
    """디코드한 clip (B, C, T, H, W) 을 변형한다. 모두 **같은 32 장** 을 쓰고 시간 축만 바꾼다."""
    if mode == "fwd":
        return c
    if mode == "rev":
        return torch.flip(c, dims=[2])                      # 역재생 (순서만 반대)
    if mode == "shuffle":                                   # 순서 무작위 (clip 마다 고정 seed) — 순서가 필요한가
        out = c.clone()
        for b in range(c.size(0)):
            g = torch.Generator().manual_seed(1000 + int(idx[b]))
            out[b] = c[b][:, torch.randperm(c.size(2), generator=g)]
        return out
    if mode == "static_first":                              # 첫 장을 32 번 — 한 장만으로 라벨이 읽히는가
        return c[:, :, :1].repeat(1, 1, c.size(2), 1, 1)
    if mode == "static_last":                               # 마지막 장을 32 번 — "끝 자리" 단서 검정
        return c[:, :, -1:].repeat(1, 1, c.size(2), 1, 1)
    raise ValueError(mode)


def extract(files, bundle, dev, mode="fwd", bs=4, workers=6, spatial="full", log=print):
    """(n, 4096, 1280) fp16 한 벌. 디스크 캐시 없음. mode 는 variant() 참고."""
    dl = torch.utils.data.DataLoader(ClipSet(files, spatial=spatial), batch_size=bs, num_workers=workers, shuffle=False)
    out = torch.empty(len(files), NTOK, D, dtype=torch.float16)
    t0 = time.time()
    with torch.no_grad():
        for clips, ii in dl:
            x = variant(clips, mode, ii).to(dev, dtype=bundle.dtype)
            z = bundle.context_encoder(x)
            out[ii] = (z[-1] if isinstance(z, list) else z).to(torch.float16).cpu()
            if int(ii[0]) % (bs * 30) == 0:
                log(f"    [{mode}] {int(ii[-1]) + 1}/{len(files)} clip  {time.time() - t0:.0f}s")
    return out


def fake_extract(files, mode="fwd", seed=0):
    """--no-encoder: 모델 없이 배관만 확인 (난수 특징, 결과는 우연 수준이어야 한다)."""
    g = torch.Generator().manual_seed(seed)
    f = torch.randn(len(files), NTOK, D, generator=g, dtype=torch.float32).half()
    return f if mode == "fwd" else torch.flip(f, dims=[1])


# ────────────────────────────────────────────────────────────────── 보고
def confusion(true, pred):
    return np.array([[int(((true == a) & (pred == b)).sum()) for b in range(len(CLASSES))] for a in range(len(CLASSES))])


def md_conf(m, title):
    tot = m.sum()
    acc = 100 * np.trace(m) / max(tot, 1)
    lines = [f"**{title}** — n {tot}, 정확도 {acc:.1f}%", "",
             "| 진실 \\ 예측 | left | right |", "|---|---:|---:|"]
    for i, c in enumerate(CLASSES):
        lines.append(f"| {c} | {m[i, 0]} | {m[i, 1]} |")
    return "\n".join(lines) + "\n"


def out_root(args):
    """⚠️ --smoke / --no-encoder 결과는 `_smoke/` 로 뺀다. 2026-09-20 에 배관 점검이 실물 RESULTS.md 를 덮어썼다."""
    return OUT_ROOT / "_smoke" if (args.smoke or args.no_encoder) else OUT_ROOT


def run_set(name, args, bundle, dev, log=print):
    files, labels = file_list(name)
    if args.smoke:
        keep = np.concatenate([np.where(labels == l)[0][:args.smoke] for l in range(len(CLASSES))])
        files, labels = [files[i] for i in keep], labels[keep]
    tr, te, groups = split_masks(name, files, labels, seed=args.seed, test_frac=args.test_frac)
    log(f"[{name}] clip {len(files)}  (left {int((labels == 0).sum())} / right {int((labels == 1).sum())})  "
        f"train {int(tr.sum())} / test {int(te.sum())}  split={SETS[name]['split']}")
    get = ((lambda m: fake_extract(files, m)) if args.no_encoder
           else (lambda m: extract(files, bundle, dev, mode=m, bs=args.batch, workers=args.workers,
                                   spatial=args.spatial, log=log)))
    fwd, rev = get("fwd"), get("rev")

    probes = {}
    rep = dict(dataset=name, n_clips=len(files), n_train=int(tr.sum()), n_test=int(te.sum()),
               split=SETS[name]["split"], spatial=args.spatial, n_frames=32, encoder="vith/encoder",
               classes=CLASSES, probes={})
    saved = {}
    for kind in args.probes:
        probe, npar = train_probe(kind, fwd, labels, tr, dev, epochs=args.epochs, lr=args.lr,
                                  wd=args.wd, bs=args.batch_probe, seed=args.seed, log=log)
        p_f, p_r = predict(probe, fwd, dev), predict(probe, rev, dev)
        y_true, y_flip = labels, 1 - labels                       # 역재생하면 진실 방향이 뒤집힌다
        cf_f = confusion(y_true[te], p_f[te])
        cf_r = confusion(y_flip[te], p_r[te])
        flip = float((p_f[te] != p_r[te]).mean())
        rec = dict(n_params=int(npar), probe_path=str((out_root(args) / name / f"probe_{kind}.pt")),
                   train_acc=float((p_f[tr] == y_true[tr]).mean()),
                   fwd_test_acc=float((p_f[te] == y_true[te]).mean()),
                   rev_test_acc_flipped_label=float((p_r[te] == y_flip[te]).mean()),
                   flip_rate=flip,
                   pred_left_frac_fwd=float((p_f[te] == 0).mean()), pred_left_frac_rev=float((p_r[te] == 0).mean()),
                   confusion_fwd=cf_f.tolist(), confusion_rev=cf_r.tolist())
        rep["probes"][kind] = rec
        saved[kind] = dict(fwd=p_f, rev=p_r)
        probes[kind] = probe
        log(f"  [{name} / {kind}] params {npar:,}  train {100*rec['train_acc']:.1f}  "
            f"fwd test {100*rec['fwd_test_acc']:.1f}  rev test (뒤집힌 라벨) {100*rec['rev_test_acc_flipped_label']:.1f}  "
            f"뒤집힘 비율 {100*flip:.1f}%")

    # ── 변형 대조: 같은 (학습된) probe 에 시간 구조만 바꾼 clip 을 넣는다 ──
    #   shuffle      순서 무작위 → 순서가 필요한가
    #   static_first / static_last  한 장을 32 번 → **한 장만으로 라벨이 읽히면** 방향이 아니라 자리를 읽은 것이다
    del fwd, rev
    for m in args.variants:
        fv = get(m)
        for kind, probe in probes.items():
            p_v = predict(probe, fv, dev)
            r = rep["probes"][kind].setdefault("variants", {})
            r[m] = dict(test_acc=float((p_v[te] == labels[te]).mean()),
                        flip_vs_fwd=float((p_v[te] != saved[kind]["fwd"][te]).mean()),
                        pred_left_frac=float((p_v[te] == 0).mean()))
            saved[kind][m] = p_v
            log(f"  [{name} / {kind} / {m}] test {100*r[m]['test_acc']:.1f}  "
                f"정방향 대비 뒤집힘 {100*r[m]['flip_vs_fwd']:.1f}  left 예측 {100*r[m]['pred_left_frac']:.1f}")
        del fv

    out = out_root(args) / name
    out.mkdir(parents=True, exist_ok=True)
    for kind, probe in probes.items():
        torch.save(probe.state_dict(), out / f"probe_{kind}.pt")
    (out / "results.json").write_text(json.dumps(rep, indent=1))
    np.savez(out / "preds.npz", clip_path=np.array([str(f) for f in files]), label=labels, test=te, group=groups,
             **{f"pred_{k}_{d}": v for k, dd in saved.items() for d, v in dd.items()})

    body = [f"# {name} — 문맥 encoder 의 좌/우 운동 방향 (정방향 학습 → 역재생 시험)", "",
            f"clip {len(files)} (left {int((labels==0).sum())} / right {int((labels==1).sum())}), "
            f"train {int(tr.sum())} / test {int(te.sum())}, split = {SETS[name]['split']} 단위, "
            f"균등 32 장 → 256x256 ({args.spatial}), ViT-H 문맥 encoder (frozen), probe 는 아래 표.", "",
            "역재생 표의 진실 라벨은 **뒤집은 것** 이다 (left 영상을 거꾸로 틀면 오른쪽으로 간다). "
            "뒤집힘 비율 = 같은 clip 의 정방향 예측과 역재생 예측이 다른 비율 (라벨이 필요 없는 검사).", "",
            "| probe | 파라미터 | train | 정방향 test | 역재생 test (뒤집힌 라벨) | 뒤집힘 비율 |",
            "|---|---:|---:|---:|---:|---:|"]
    for kind, r in rep["probes"].items():
        body.append(f"| {kind} | {r['n_params']:,} | {100*r['train_acc']:.1f} | {100*r['fwd_test_acc']:.1f} | "
                    f"{100*r['rev_test_acc_flipped_label']:.1f} | {100*r['flip_rate']:.1f} |")
    body.append("")
    if args.variants:
        body += ["### 변형 대조 (같은 probe, 시간 구조만 바꾼 clip; 정확도는 **원래 라벨** 기준)", "",
                 "| probe | 변형 | test 정확도 | 정방향 대비 뒤집힘 | left 예측 비율 |", "|---|---|---:|---:|---:|"]
        for kind, r in rep["probes"].items():
            for m, v in r.get("variants", {}).items():
                body.append(f"| {kind} | {m} | {100*v['test_acc']:.1f} | {100*v['flip_vs_fwd']:.1f} | {100*v['pred_left_frac']:.1f} |")
        body += ["", "`shuffle` = 32 장의 순서 무작위 · `static_first` / `static_last` = 그 한 장을 32 번 반복.",
                 "**한 장짜리 clip 에서 정확도가 높으면 방향이 아니라 그 시점의 자리를 읽은 것이다.**", ""]
    for kind, r in rep["probes"].items():
        body += [f"## {kind}", "", md_conf(np.array(r["confusion_fwd"]), f"{name} 정방향 test"), "",
                 md_conf(np.array(r["confusion_rev"]), f"{name} 역재생 test (진실 = 뒤집은 라벨)"), ""]
    body += ["## 재현", "", "```bash",
             f"P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python",
             f"$P z_research/scripts/analysis/ctxenc_direction_probe.py --sets {name}", "```", ""]
    (out / "RESULTS.md").write_text("\n".join(body))
    log(f"  → {out}/RESULTS.md")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["ssv2", "ntu"], choices=list(SETS))
    ap.add_argument("--probes", nargs="+", default=["attentive", "tiny"], choices=["attentive", "tiny"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=4, help="encoder forward 배치 (clip)")
    ap.add_argument("--batch-probe", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--spatial", default="full", choices=["full", "crop"])
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", nargs="*", default=["shuffle", "static_first", "static_last"],
                    choices=["shuffle", "static_first", "static_last"],
                    help="학습된 probe 에 걸 변형 대조 (빈 목록이면 안 함)")
    ap.add_argument("--smoke", type=int, default=0, help="클래스당 clip 수 제한 (배관 점검)")
    ap.add_argument("--no-encoder", action="store_true", help="모델 없이 난수 특징으로 배관만")
    args = ap.parse_args()

    for name in args.sets:                                     # 모델을 올리기 전에 실물부터 확인한다 (ViT-H 로딩 ~2분)
        fs, lb = file_list(name)
        print(f"[check] {name}: {len(fs)} clip (left {int((lb==0).sum())} / right {int((lb==1).sum())})", flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = None
    if not args.no_encoder:
        from analysis.intphys2.model import build_from_config
        t0 = time.time()
        bundle = build_from_config(MODEL, dev)
        print(f"[model] ViT-H 문맥 encoder 로딩 {time.time() - t0:.0f}s  dtype={bundle.dtype}", flush=True)

    allrep = {}
    for name in args.sets:
        allrep[name] = run_set(name, args, bundle, dev, log=lambda *a: print(*a, flush=True))
    o = out_root(args)
    o.mkdir(parents=True, exist_ok=True)
    (o / "summary.json").write_text(json.dumps(allrep, indent=1))
    print(f"[done] {o}/summary.json", flush=True)


if __name__ == "__main__":
    main()
