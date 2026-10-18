#!/usr/bin/env python3
"""EK100 encoder 특징 캐시 — predictor 는 모델마다 바뀌니 encoder 출력만 한 번 뽑아 둔다 (2026-10-16, 사용자 지시).

EK100 wrapper (evals/action_anticipation_frozen/modelcustom/vit_encoder_predictor_concat_ar_nonsquare.py) 가 쓰는 encoder 는
`checkpoint_key: target_encoder` 하나다 (context encoder 는 이 경로에 안 쓰인다). 그 encoder 의 출력 두 가지를 뽑는다:
  joint  : 32 프레임 창을 한 번에 → (4096, 1280).  probe 입력의 encoder 부분 (모든 모델) + release kind predictor 의 입력
  blocks : 튜블릿 (2 프레임) 마다 따로 → affine-free LN → (4096, 1280).  AR kind (ar · ar_state) predictor 의 문맥 (app/vjepa_frozen/ar.encode_blocks)
clip 샘플링 · 변환은 엔진 (epickitchens.decode_videos_to_clips + dataloader.VideoTransform) 과 같은 식을 그대로 쓴다.
  train : 학습 증강 (anticipation time · point 무작위, RRC · RandAugment · RandomErasing · hflip) 을 view 마다 한 번 뽑아 고정 (--views K).
          view v · chunk c 의 난수는 seed (v, c) 로 정해져 GPU 수 · worker 수와 무관하게 재현된다.
  val   : 결정적 (at 1 s, ap 0, eval 변환) — 엔진 val 과 같은 입력 (check 로 확인).
출력 <out>/<split>/  :  joint.npy · blocks.npy (np.memmap, float16, (행, 4096, 1280))  +  meta.npz (video_id · row · view · verb · noun · at · valid)
행 = view * n_clip + clip 순번 (clip 순번 = filter_annotations 의 영상 순서 × 영상 안 start_frame 순서 = 엔진과 같은 순서).

  python cvpr/eval/ek100/cache/extract.py --config <_resolved.yaml> --split val --prepare        # 파일만 만든다 (한 번)
  CUDA_VISIBLE_DEVICES=2 python cvpr/eval/ek100/cache/extract.py --config … --split val --shard 0 --nshard 2
  python cvpr/eval/ek100/cache/extract.py --config … --split val --merge                          # shard meta 를 meta.npz 로
  python cvpr/eval/ek100/cache/extract.py --config … --split val --check 8                        # 엔진 val 로더와 입력 비트 일치 · 특징 일치
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
N_TOK, DIM, CHUNK = 4096, 1280, 8


def load_cfg(p):
    cfg = yaml.safe_load(open(p))
    return cfg, cfg["experiment"]["data"]


def clip_table(cfg, data, split):
    """엔진과 같은 annotation → [(path, video_id, df 행 위치 목록)] · n_clip. 행 순서 = 엔진 순서."""
    from evals.action_anticipation_frozen.dataloader import filter_annotations
    ann = filter_annotations(data["dataset"], data["base_path"], data["dataset_train"], data["dataset_val"],
                             file_format=data.get("file_format", 1))
    paths, anos = ann[split]
    rows, k = [], 0
    for p in paths:
        vid = p.split("/")[-1].split(".")[0]; n = len(anos[vid])
        for s in range(0, n, CHUNK):
            rows.append((p, vid, list(range(s, min(n, s + CHUNK))), k + s))
        k += n
    return ann, anos, rows, k


class ChunkSet(torch.utils.data.Dataset):
    """item = (view, chunk): 한 영상의 연속 clip 최대 CHUNK 개. decode_videos_to_clips.run 과 같은 식."""

    def __init__(self, cfg, data, split, anos, rows, views, shard, nshard, done):
        from evals.action_anticipation_frozen.dataloader import make_transforms
        tr = split == "train"
        self.tr, self.anos, self.rows = tr, anos, rows
        self.fpc, self.fps = int(data["frames_per_clip"]), int(data["frames_per_second"])
        self.at_rng = data["train_anticipation_time_sec"] if tr else data["anticipation_time_sec"]
        self.ap_rng = data["train_anticipation_point"] if tr else data["val_anticipation_point"]
        self.ap_mode, self.time_source = data.get("anticipation_point_mode", "released"), data.get("time_source", "frame")
        self.tf = make_transforms(training=tr, random_horizontal_flip=True, random_resize_aspect_ratio=(3 / 4, 4 / 3),
                                  random_resize_scale=data.get("random_resize_scale"), reprob=data.get("reprob"),
                                  auto_augment=data.get("auto_augment"), motion_shift=data.get("motion_shift"),
                                  crop_size=int(data["resolution"]), spatial_mode=data.get("spatial_mode", "center_crop"),
                                  crop_width=data.get("width"), train_spatial_mode=data.get("train_spatial_mode", "rrc"))
        self.items = [(v, c) for v in range(views) for c in range(len(rows)) if (v * len(rows) + c) % nshard == shard and (v, c) not in done]

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _ts(ts):
        h, m, r = str(ts).split(":"); return int(h) * 3600 + int(m) * 60 + float(r)

    def __getitem__(self, i):
        from decord import VideoReader, cpu
        v, c = self.items[i]; path, vid, idx, row0 = self.rows[c]
        seed = 1_000_003 * (v + 1) + c
        random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed)
        ano = self.anos[vid]
        out = dict(view=v, chunk=c, row=[], video=[], verb=[], noun=[], at=[], valid=[])
        try:
            vr = VideoReader(path, num_threads=1, ctx=cpu(0)); vr.seek(0)
            vfps = vr.get_avg_fps(); fstp = int(vfps / self.fps); nfr = int(self.fpc * fstp)
        except Exception as e:  # 엔진은 영상 전체를 건너뛴다
            print(f"[skip video] {vid}: {e}", flush=True); vr = None
        for j in idx:
            sf, ef = ano["start_frame"].values[j], ano["stop_frame"].values[j]
            at = random.uniform(*self.at_rng); ap = random.uniform(*self.ap_rng)
            ok, buf = vr is not None, None
            if ok:
                use_ts = self.time_source == "timestamp"
                if self.time_source == "frame_fixfps":
                    use_ts = min(abs(vfps - 60.0), abs(vfps - 50.0)) > 1.0 or ef >= len(vr)
                if use_ts and "start_timestamp" in ano:
                    sf = int(round(self._ts(ano["start_timestamp"].values[j]) * vfps)); ef = int(round(self._ts(ano["stop_timestamp"].values[j]) * vfps))
                af0 = int(at * vfps)
                af = int(sf * ap + (1 - ap) * ef - af0) if self.ap_mode == "released" else int(sf * (1 - ap) + ap * ef - af0)
                ind = np.arange(af - nfr, af, fstp).astype(np.int64); ind[ind < 0] = 0
                try:
                    buf = self.tf(vr.get_batch(ind).asnumpy())
                except Exception as e:
                    print(f"[skip clip] {vid} #{j}: {e}", flush=True); ok = False
            out["row"].append(row0 + (j - idx[0])); out["verb"].append(int(ano["verb_class"].values[j])); out["noun"].append(int(ano["noun_class"].values[j]))
            out["at"].append(at); out["valid"].append(ok); out["video"].append(buf if ok else torch.zeros(3, self.fpc, 1, 1))
        return out


def build_encoder(cfg, data, dev):
    from evals.action_anticipation_frozen.models import init_module
    mk = cfg["model_kwargs"]; args_model = dict(mk["pretrain_kwargs"])
    w = data.get("width"); res = int(data["resolution"])
    model = init_module(module_name=mk["module_name"], frames_per_clip=int(data["frames_per_clip"]), frames_per_second=int(data["frames_per_second"]),
                        resolution=res if w in (None, res) else [res, w], checkpoint=mk["checkpoint"], model_kwargs=args_model,
                        wrapper_kwargs=mk["wrapper_kwargs"], device=dev)
    enc = model.encoder.to(dev).eval()
    return enc


@torch.no_grad()
def encode(enc, x, want):
    from app.vjepa_frozen.ar import encode_blocks
    o = {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if "joint" in want: o["joint"] = enc(x)
        if "blocks" in want: o["blocks"] = encode_blocks(enc, x, 2)
    return {k: v.to(torch.float16) for k, v in o.items()}


def mm(path, n, mode, dtype=np.float16, shape=(N_TOK, DIM)):
    return np.lib.format.open_memmap(path, mode=mode, dtype=dtype, shape=(n, *shape) if mode == "w+" else None)


def q8(x):
    """int8, clip 마다 채널별 scale (토큰 축 absmax / 127). (B, N, D) → (int8 (B, N, D), float32 (B, D)). 2026-10-16 실측 특징 상대 L1 0.85 % (토큰별 scale 은 2.2 %)."""
    s = x.float().abs().amax(1).clamp_min(1e-8) / 127.0
    return torch.round(x.float() / s[:, None]).clamp(-127, 127).to(torch.int8), s


def read_rows(D, k, rows):
    """캐시 행 읽기 (fp16 또는 int8 + scale) → float32 numpy."""
    M = np.load(D / f"{k}.npy", mmap_mode="r"); x = np.stack([M[r] for r in rows]).astype(np.float32)
    if (D / f"{k}_scale.npy").exists():
        S = np.load(D / f"{k}_scale.npy", mmap_mode="r"); x *= np.stack([S[r] for r in rows])[:, None]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--out", default="/data2/local_datasets/world/world_analysis/cache/ek100_feats")
    ap.add_argument("--name", default=None, help="캐시 폴더 이름 (기본: config tag 에서 ap 규약 · time_source · train csv 를 딴 것)")
    ap.add_argument("--views", type=int, default=1); ap.add_argument("--variants", default="joint,blocks")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--bs", type=int, default=8); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--base", default=None, help="영상 루트 덮어쓰기 (노드마다 다르다: vll1 /local_datasets/…, vll5 /data2/local_datasets/EPIC-KITCHENS_resized)")
    ap.add_argument("--dtype", choices=["f16", "q8"], default="f16", help="q8 = int8 + clip·채널별 scale (용량 절반)")
    ap.add_argument("--prepare", action="store_true"); ap.add_argument("--merge", action="store_true"); ap.add_argument("--check", type=int, default=0)
    a = ap.parse_args()
    cfg, data = load_cfg(a.config)
    if a.base: data["base_path"] = a.base
    name = a.name or f"{data.get('anticipation_point_mode')}_{data.get('time_source')}_{Path(data['dataset_train']).stem}"
    D = Path(a.out) / name / a.split; D.mkdir(parents=True, exist_ok=True)
    want = a.variants.split(","); views = a.views if a.split == "train" else 1
    ann, anos, rows, n_clip = clip_table(cfg, data, a.split); n_row = views * n_clip

    if a.prepare:
        for k in want:
            if not (D / f"{k}.npy").exists():
                mm(D / f"{k}.npy", n_row, "w+", np.int8 if a.dtype == "q8" else np.float16); print("made", D / f"{k}.npy", n_row, a.dtype)
                if a.dtype == "q8": mm(D / f"{k}_scale.npy", n_row, "w+", np.float32, (DIM,))
        json.dump(dict(config=str(a.config), split=a.split, views=views, n_clip=n_clip, n_row=n_row, variants=want, chunk=CHUNK,
                       data={k: data.get(k) for k in ("dataset_train", "dataset_val", "anticipation_point_mode", "time_source", "train_anticipation_time_sec",
                                                    "train_anticipation_point", "anticipation_time_sec", "val_anticipation_point", "random_resize_scale",
                                                    "auto_augment", "reprob", "train_spatial_mode", "spatial_mode", "resolution", "frames_per_clip", "frames_per_second")},
                       checkpoint=cfg["model_kwargs"]["checkpoint"], encoder_key=cfg["model_kwargs"]["pretrain_kwargs"]["encoder"]["checkpoint_key"],
                       dtype=("int8 + per-(clip, channel) float32 scale" if a.dtype == "q8" else "float16") + " (encoder under bf16 autocast)"), open(D / "info.json", "w"), indent=1, ensure_ascii=False)
        return

    if a.merge:
        M = {k: [] for k in ("row", "view", "verb", "noun", "at", "valid")}
        for f in sorted(D.glob("meta_shard*.jsonl")):
            for l in open(f):
                d = json.loads(l)
                for k in M: M[k] += d[k] if k != "view" else [d["view"]] * len(d["row"])
        order = np.argsort(np.array(M["row"]) + n_clip * np.array(M["view"]))
        R = {k: np.array(v)[order] for k, v in M.items()}
        full = n_clip * np.array(R["view"]) + R["row"]
        assert len(np.unique(full)) == len(full), "행 중복"
        print(f"rows {len(full)}/{n_row}  valid {int(R['valid'].sum())}  missing {n_row - len(full)}")
        vids = [r[1] for r in rows for _ in r[2]]
        np.savez(D / "meta.npz", video_id=np.array(vids)[R["row"]], **R)
        return

    dev = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    if a.check:
        return check(a, cfg, data, ann, anos, rows, D, dev, want)
    done = set()
    mf = D / f"meta_shard{a.shard}.jsonl"
    for f in D.glob("meta_shard*.jsonl"):            # 모든 shard 의 기록 — shard 수를 바꿔 다시 띄워도 이미 한 chunk 는 건너뛴다
        done |= {(json.loads(l)["view"], json.loads(l)["chunk"]) for l in open(f)}
    ds = ChunkSet(cfg, data, a.split, anos, rows, views, a.shard, a.nshard, done)
    print(f"[shard {a.shard}/{a.nshard}] {a.split} chunks {len(ds)} (done {len(done)}) · rows {n_row}", flush=True)
    if not len(ds): return
    enc = build_encoder(cfg, data, dev)
    M = {k: mm(D / f"{k}.npy", None, "r+") for k in want}
    Sc = {k: mm(D / f"{k}_scale.npy", None, "r+") for k in want if (D / f"{k}_scale.npy").exists()}
    dl = torch.utils.data.DataLoader(ds, batch_size=None, shuffle=False, num_workers=a.workers, prefetch_factor=4, persistent_workers=False)
    t0 = time.time(); nclip = 0
    with open(mf, "a") as fh:
        for i, it in enumerate(dl):
            ok = [k for k, f in enumerate(it["valid"]) if f]
            if ok:
                x = torch.stack([it["video"][k] for k in ok]).to(dev, non_blocking=True)
                rr = [it["view"] * n_clip + it["row"][k] for k in ok]
                for s in range(0, len(ok), a.bs):
                    o = encode(enc, x[s:s + a.bs], want)
                    for k in want:
                        if k in Sc:
                            qq, sc = q8(o[k]); arr, sca = qq.cpu().numpy(), sc.cpu().numpy()
                            for q, r in enumerate(rr[s:s + a.bs]): M[k][r] = arr[q]; Sc[k][r] = sca[q]
                        else:
                            arr = o[k].cpu().numpy()
                            for q, r in enumerate(rr[s:s + a.bs]): M[k][r] = arr[q]
            fh.write(json.dumps(dict(view=it["view"], chunk=it["chunk"], row=it["row"], verb=it["verb"], noun=it["noun"], at=it["at"], valid=it["valid"])) + "\n"); fh.flush()
            nclip += len(it["row"])
            if i % 50 == 0:
                el = time.time() - t0; print(f"  chunk {i}/{len(ds)}  clips {nclip}  {nclip / max(el, 1e-6):.2f} clip/s  eta {(len(ds) - i) * el / max(i, 1) / 60:.0f} min", flush=True)
    for m in (*M.values(), *Sc.values()): m.flush()
    print(f"[shard {a.shard}] done {nclip} clips {(time.time() - t0) / 60:.1f} min", flush=True)


def check(a, cfg, data, ann, anos, rows, D, dev, want):
    """엔진 val 로더 (num_workers 0, ws 1) 의 앞 a.check clip 과 우리 입력 · 캐시 값을 비교."""
    from evals.action_anticipation_frozen.dataloader import init_data
    assert a.split == "val"
    paths, an = ann["val"]
    _, loader, _ = init_data(dataset=data["dataset"], training=False, base_path=data["base_path"], annotations_path=(paths[:2], {q.split("/")[-1].split(".")[0]: an[q.split("/")[-1].split(".")[0]] for q in paths[:2]}),   # 1 개면 ResampledShards 가 문자열을 글자로 푼다
                             batch_size=a.check, frames_per_clip=int(data["frames_per_clip"]), fps=int(data["frames_per_second"]),
                             anticipation_time_sec=data["anticipation_time_sec"], anticipation_point=data["val_anticipation_point"],
                             crop_size=int(data["resolution"]), crop_width=data.get("width"), spatial_mode=data.get("spatial_mode", "center_crop"),
                             anticipation_point_mode=data.get("anticipation_point_mode", "released"), time_source=data.get("time_source", "frame"),
                             world_size=1, rank=0, num_workers=0, pin_mem=False, persistent_workers=False)
    ub = next(iter(loader)); n = len(ub[0])
    ds = ChunkSet(cfg, data, "val", anos, rows, 1, 0, 1, set())
    mine = []
    for c in range(len(rows)):
        if len(mine) >= n: break
        it = ds[c]; mine += [(it["video"][k], it["row"][k]) for k in range(len(it["row"]))]
    xs = torch.stack([m[0] for m in mine[:n]])
    print(f"입력 비트 일치: max|Δ| = {float((xs - ub[0]).abs().max()):.3g}  (n {n}) · at {ub[3].tolist()[:3]} · verb {ub[1].tolist()[:4]}")
    enc = build_encoder(cfg, data, dev)
    o = encode(enc, ub[0].to(dev), want)
    for k in want:
        f = D / f"{k}.npy"
        if not f.exists(): continue
        got = torch.from_numpy(read_rows(D, k, [r for _, r in mine[:n]])).to(dev)
        if float(got.abs().sum()) == 0: print(f"  {k}: 캐시에 아직 없음"); continue
        rel = float((got.float() - o[k].float()).abs().mean() / o[k].float().abs().mean())
        print(f"  {k}: 캐시 vs 새로 계산 상대 L1 {rel:.2e}")


if __name__ == "__main__":
    main()
