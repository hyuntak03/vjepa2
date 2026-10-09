#!/usr/bin/env python3
"""**가림 없는 조건에서, 화면 안 물체의 존재와 중심을 frozen feature 에서 읽을 수 있는가** — probe 3 종 비교.

학습셋 `RollOut_v2_training` v6 (8,352 clip, 빈 장면 288 포함). 이 세트에는 **진짜 가림이 없다** —
같은 깊이의 판은 물체보다 좁아 최대 31 % 만 덮는다 (2026-09-22 확인). 그래서 결과는 위 한 줄로만 해석한다.
"가려진 물체를 유지하는가" 는 이 세트로 물을 수 없다.

라벨 (샘플 32 개 → 튜블릿 16 개, 미래 8 개만 씀. 생성기의 `prop_intersects_by_sample` 이 정본):
  양성  화면 안이고 장애물과 **안 겹침**              198,906 샘플
  음성  빈 장면 clip **또는** 화면 밖                  59,868  (빈 장면 9,216 · 화면 밖 50,652)
  제외  화면 안인데 장애물과 겹침 → 두 손실에서 모두 뺀다   8,490
  좌표 손실은 **세 probe 모두 양성에서만**, presence BCE 는 **양성 + 음성 전체**에서.

probe 3 종 (같은 특징을 공유한다 — 추출 한 번):
  linear  flatten(256×1280) → Linear(·,2) 좌표 · Linear(·,1) presence        983 k   (토큰별 위치를 그대로 쓰는 가장 단순한 probe)
  heat    토큰별 점수 s_i = tok_i·w + b → 좌표 = Σ softmax(s)_i · 칸중심_i,
          presence = logsumexp(s) − log 256 을 **로짓으로 BCE 학습**             1,281
  attn    attention pooling → pooled 에서 좌표·presence (대조군)                 5,123
  ⚠️ attentive pooling 이 위치를 버리는 것은 아니다 (토큰에 위치 정보가 있으면 pooling 뒤에도 남는다).
     flatten 을 기본으로 두는 이유는 **토큰별 위치를 직접 쓰는 더 단순한 probe** 라는 것뿐이다.
     softmax 가 늘 어딘가를 가리키는 것도 좌표와 presence 를 분리하면 문제가 아니다.

평가 (held-out test):
  중심 오차  **모든 GT 양성**에서 평균 / 중앙값 / 상위 90 % (presence 판정과 무관하게)
  presence   ROC-AUC · PR-AUC (test 전체)
  오탐률     **val 에서 정한 하나의 문턱**으로 빈 장면 / 화면 밖 각각
  ⚠️ 음성 출처만으로는 AUC 가 정의되지 않는다. 출처별 AUC 는 **같은 양성 집합 + 그 음성 출처** 조합임을 밝힌다.

split: block 단위, scenario 층화. train 50 % / val 15 % / test 35 %.
캐시를 디스크에 쓰지 않는다 — clip 마다 p·z·h 를 한 번에 뽑아 /dev/shm 에 두고 그 위에서 학습한다.

출력: z_research/RollOutV3/exp_results/presence/{summary.json, RESULTS.md, <rep>/{readout_<kind>.pt, preds_<kind>.npz}}

  P=/data/hyuntak/anaconda3/envs/vjepa2/bin/python
  $P z_research/scripts/analysis/rollout2_presence_readout.py
  $P z_research/scripts/analysis/rollout2_presence_readout.py --limit 64 --epochs 5 --gpus 1 --reps p
  bash z_research/scripts/analysis/rollout2_presence_sbatch.sh
"""
from __future__ import annotations
import argparse, csv, json, os, shutil, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.multiprocessing as mp

ROOT = Path("/data/hyuntak/project/2026/2027_cvpr/vjepa2")
sys.path.insert(0, str(ROOT))

# ⚠️ **학습셋은 환경변수로 고른다** (`--set` 이 채운다). mp.spawn 워커가 모듈을 다시 import 하므로
#    main() 에서 바꾼 전역은 워커에 안 닿는다 — PRESENCE_FEAT_DIR 과 같은 이유 (2026-09-23).
#    training_v8 (2026-09-24, 14,360 clip) 이 기본이다. 옛 자는 training_v6 (8,352 clip) 으로 배웠다.
SET = os.environ.get("PRESENCE_SET", "training_v8")
INDEX = ROOT / f"data_csv/rollout_v2_{SET}/index_probe.csv"
# RESULTS.md 의 자동 생성 구간 표시. 이 블록만 갈아 끼우고 바깥 (손으로 쓴 절) 은 보존한다
AUTO0, AUTO1 = "<!-- AUTO:BEGIN — 이 구간은 실행마다 다시 쓰인다 -->", "<!-- AUTO:END -->"
DEFAULT_TAIL = ("## 재현\n\n```bash\n"
                f"python z_research/scripts/data/build_rollout2_index.py --set {SET} --write\n"
                f"python z_research/scripts/analysis/rollout2_presence_readout.py --set {SET} --feat-dir <디스크>\n```\n")
# 프레임 폴더는 세트마다 (2026-09-26: training_r8 = 새 렌더 RollOut_v2_training_v8 은 /data2 에만 있다 — build_rollout2_index.py SETS 와 같다)
FRAMES = os.environ.get("PRESENCE_FRAMES") or {"training_r8": "/data2/local_datasets/world/world_analysis/RollOut_v2_training_v8"}.get(
    SET, "/local_datasets/world/world_analysis/RollOut_v2_training")   # 2026-09-30: PRESENCE_FRAMES 로 덮어쓰기 (다른 노드)
OUT = ROOT / "z_research/RollOutV3/exp_results/presence"      # 2026-09-22: 세트를 RollOutV3 로 분리
# 특징 저장 위치. 기본은 /dev/shm (빠르지만 **세션이 끝나면 지워진다**) —
# 재사용할 거면 `--feat-dir /data2/.../rollout3_presence_feats` 로 디스크에 둔다 (123 GB, 추출 27 분 절약).
# ⚠️ mp.spawn 워커는 이 모듈을 **다시 import** 한다 — main() 에서 바꾼 값은 워커에 안 간다.
#    그래서 환경변수로 싣는다 (2026-09-23 에 여기서 FileNotFoundError 가 났다).
SHM = Path(os.environ.get("PRESENCE_FEAT_DIR", "/dev/shm/rollout2_presence"))
# ⚠️ **학습셋 (RollOut_v2_training) 의 특징**이다. v3 가 아니다 —
#    폴더 이름을 rollout3_presence_feats 로 붙였다가 오해를 부를 뻔해 2026-09-23 에 바꿨다.
FEAT_DEFAULT_DISK = Path(f"/data2/local_datasets/world/world_analysis/cache/rollout2_{SET}_feats")
S, T, D, RES, CELL = 256, 8, 1280, 144.0, 18.0          # 토큰/슬롯/차원 · 정규화 기준 · 1 칸 px
REPS = ("p", "z", "h")
MODEL = dict(                                            # configs/protocols/models.md `vith` + attn_probe 의 model 블록
    checkpoint=str(ROOT / "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/"
                          "b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth"),
    arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, window_size=32,
    use_rope=True, uniform_power=False,
    dual_encoder=True, context_encoder_key="encoder", target_encoder_key="target_encoder",
    predictor=dict(embed_dim=384, depth=12, num_heads=12, num_mask_tokens=10),
    dtype="bfloat16",
)
# 2026-09-28: 다른 predictor (Ariel 판 등) 로 p 를 뽑을 때. predictor 만 이 파일에서 읽고 encoder 는 릴리즈 그대로 (arch.kind 는 체크포인트가 선언)
if os.environ.get("PRED_CKPT"):
    MODEL["predictor_checkpoint"] = os.environ["PRED_CKPT"]
# 뽑을 표현 (기본 p z h). ⚠️ kind=ar 이면 p = 블록별 LN(target_encoder) 문맥 8 블록에서 rollout 8 블록, h = 블록별 LN(target_encoder) (analysis/predictors/ar_scoring.py)
EXTRACT_REPS = tuple(x for x in os.environ.get("PRESENCE_REPS", "p,z,h").split(",") if x)
DATA = dict(root=str(ROOT / f"data_csv/rollout_v2_{SET}"), index_csv="index.csv",
            frames_root=FRAMES, frames_pattern="{file_name}/{frame:06d}.png",
            frames_start=0, frames_stride=3, n_frames=32, resolution=256, block_column="block_id")


def arr(s):
    return np.array([float(v) for v in str(s).split()], np.float32)


def weights(rows):
    """(Wc (n,8) 튜블릿 위치 가중, Wb (n,) clip 균형 가중). 컬럼이 없는 세트는 전부 1.

    training_v8 부터 데이터가 싣는다 (데이터 README §3):
      `cell_weight_by_sample`  8 px 칸 단위 위치 분포 평탄화 — **좌표 손실**에만 쓴다
      `balance_weight`         구조물·방해물 ↔ 물체유무 탈상관 (clip 단위, 0.33~9.0) — **두 손실 모두**
    ⚠️ balance_weight 를 무시하면 방해물 ↔ 물체유무 상호정보량이 0.0232 bits 남는다 (적용하면 0).
    튜블릿 가중 = 두 샘플의 평균. 양성 튜블릿은 두 샘플 모두 화면 안이라 둘 다 값이 있다.
    """
    n = len(rows)
    if "cell_weight_by_sample" not in rows[0] or not any(r["cell_weight_by_sample"].strip() for r in rows):
        return np.ones((n, T), np.float32), np.ones(n, np.float32)
    cw = np.stack([arr(r["cell_weight_by_sample"]) if r["cell_weight_by_sample"].strip() else np.zeros(32, np.float32)
                   for r in rows]).reshape(n, 16, 2).mean(2)[:, T:]
    bw = np.array([float(r["balance_weight"] or 1.0) for r in rows], np.float32)
    return cw.astype(np.float32), bw


def family(sc):
    """시나리오 → 무대 가족. 이 데이터의 요점이 **구조물 가족에서도 존재 판정이 되는가**라 따로 본다."""
    for k in ("ledge", "wedge", "ramp", "wall", "edge", "prop", "panel", "empty"):    # ramp · panel = training_r8 의 arm 이름
        if sc.startswith(k):
            return k
    return "plain"


def labels():
    """(rows, L (n,8,2) 좌표, pos (n,8) 양성, ign (n,8) 제외). 튜블릿 = 샘플 2 개, 둘 다 만족해야 그 라벨."""
    rows = list(csv.DictReader(INDEX.open()))
    n = len(rows)
    px = np.stack([arr(r["px_x_by_sample"]) if r["px_x_by_sample"].strip() else np.zeros(32, np.float32) for r in rows])
    py = np.stack([arr(r["px_y_by_sample"]) if r["px_y_by_sample"].strip() else np.zeros(32, np.float32) for r in rows])
    L = np.stack([px.reshape(n, 16, 2).mean(2) / RES - 1, py.reshape(n, 16, 2).mean(2) / RES - 1], -1)[:, T:]
    pos = (np.stack([arr(r["visible_by_sample"]) for r in rows]) > 0).reshape(n, 16, 2).all(2)[:, T:]
    ign = (np.stack([arr(r["ignore_by_sample"]) for r in rows]) > 0).reshape(n, 16, 2).any(2)[:, T:]   # 하나만 겹쳐도 뺀다
    return rows, L.astype(np.float32), pos, ign & ~pos


# ─────────────────────────────────────────────────── 특징 추출 (GPU 병렬, /dev/shm)
def _worker(rank, world, n, limit, log_every, bs=32):
    """clip 을 **bs 개씩 묶어** p · z · h 를 뽑는다 (2026-09-26 — 그 전에는 clip 하나씩: batch 1 은 커널 발사 오버헤드가 크다).

    - 랭크마다 **연속 구간** [rank·n/world, (rank+1)·n/world) 을 맡는다 → memmap 쓰기가 파일 안에서 순차적이다
    - PNG 디코드는 스레드로 몇 묶음 앞서 읽는다 (PIL 은 GIL 을 놓는다). 랭크마다 torch CPU 스레드를 제한한다 —
      전체 코어를 잡으면 디코드가 39 → 268 ms 로 느려진다 (CLAUDE.md §7-1)
    - 출력은 clip 하나씩 뽑던 판과 같은 모양 · 같은 dtype (fp16). bf16 누적 순서가 배치에 따라 달라 값은 ~1e-3 수준에서 다를 수 있다
    - 실측 (v11 온라인, 2026-09-25): 묶음 16 이상에서 GPU 사용률 86–88 % 로 포화, 4.7 clip/s/GPU. 묶음을 키워도 VRAM 만 늘고 안 빨라진다
    """
    import decord  # noqa: F401  (WMADataset 이 쓴다)
    from concurrent.futures import ThreadPoolExecutor
    torch.cuda.set_device(rank)
    torch.set_num_threads(max(1, int(os.environ.get("PRESENCE_TORCH_THREADS", "4"))))
    dev = torch.device("cuda", rank)
    from analysis.intphys2.model import build_from_config
    from evals.world_model_analysis.data import WMADataset
    from evals.analysis_vlm.occlusion_identity.forward import extract_batch

    bundle = build_from_config(MODEL, dev)
    from analysis.predictors import is_ar
    ar = is_ar(bundle.predictor)
    if ar:
        from app.vjepa_frozen.ar import encode_blocks, block_indices
        assert "z" not in EXTRACT_REPS, "kind=ar 에는 z (online encoder 문맥) 가 없다 — PRESENCE_REPS=p,h"
    ds = WMADataset(dict(data=DATA, model=MODEL, features={"cache_dir": "/tmp"}, surprise={}))
    mm = {r: np.lib.format.open_memmap(SHM / f"{r}.npy", mode="r+") for r in EXTRACT_REPS}
    ln = lambda x: F.layer_norm(x, (x.size(-1),))
    lo, hi = rank * n // world, (rank + 1) * n // world
    chunks = [list(range(k, min(k + bs, hi))) for k in range(lo, hi, bs)]
    pool = ThreadPoolExecutor(max_workers=PREFETCH)
    fut = {}

    def submit(ch):
        for i in ch:
            fut[i] = pool.submit(ds.clip, i)
    for ch in chunks[:2]:
        submit(ch)
    t0 = time.time(); done = 0
    with torch.no_grad():
        for j, ch in enumerate(chunks):
            if j + 2 < len(chunks):
                submit(chunks[j + 2])
            clips = torch.stack([fut.pop(i).result() for i in ch]).to(dev, dtype=bundle.dtype, non_blocking=True)
            B = clips.size(0)
            if ar:                                   # 블록별 인코딩 → 문맥 8 블록 → rollout 8 블록
                hb = encode_blocks(bundle.target_encoder, clips, 2)                 # (B, 16*S, D), LN
                if "p" in EXTRACT_REPS:
                    idx_ = block_indices(B, 16, S, dev)
                    with torch.autocast("cuda", dtype=torch.bfloat16):          # 2026-09-30: 순수 bf16 에서는 PrefixSpec SDPA 가 dtype 오류 (te_v3_readout 머리말) — 학습과 같은 autocast
                        pr = bundle.predictor.rollout(hb[:, :T * S], idx_[:, :T * S], T)
                    mm["p"][ch[0]:ch[-1] + 1] = pr.float().reshape(B, T, S, D).to(torch.float16).cpu().numpy()
                if "h" in EXTRACT_REPS:
                    mm["h"][ch[0]:ch[-1] + 1] = hb[:, T * S:].float().reshape(B, T, S, D).to(torch.float16).cpu().numpy()
                done += B
                if rank == 0 and (j % log_every == 0 or j == len(chunks) - 1):
                    el = time.time() - t0
                    print(f"    [extract ar] 랭크0 {done}/{hi - lo} clip  {el:.0f}s  {done / max(el, 1e-9):.1f} clip/s/GPU", flush=True)
                continue
            # p: 앞 16 장을 문맥으로 마스크 → predictor 가 미래 8 튜블릿
            if "p" in EXTRACT_REPS:
                out = extract_batch(clips, bundle, [{"base": "predictor"}], context_length=16, mask_index=0,
                                    out_dtype=torch.float16)
                mm["p"][ch[0]:ch[-1] + 1] = out["predictor"].reshape(B, T, S, D).numpy()
            # z / h: 32 장 전체를 통과시키고 뒤 8 튜블릿만 (affine-free LN — 캐시 관례와 같다)
            for rep, mod in (("z", bundle.context_encoder), ("h", bundle.target_encoder)):
                if rep not in EXTRACT_REPS:
                    continue
                f = mod(clips)
                f = f[-1] if isinstance(f, list) else f
                mm[rep][ch[0]:ch[-1] + 1] = ln(f.float()).reshape(B, -1, S, D)[:, T:].to(torch.float16).cpu().numpy()
            done += B
            if rank == 0 and (j % log_every == 0 or j == len(chunks) - 1):
                el = time.time() - t0
                print(f"    [extract] 랭크0 {done}/{hi - lo} clip  {el:.0f}s  {done / max(el, 1e-9):.1f} clip/s/GPU  "
                      f"남은 {(hi - lo - done) / max(done / max(el, 1e-9), 1e-9) / 60:.1f}분  최대 VRAM {torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GiB",
                      flush=True)
    pool.shutdown(wait=False)
    for m in mm.values():
        m.flush()


PREFETCH = int(os.environ.get("PRESENCE_PREFETCH", "8"))     # GPU 당 디코드 스레드 (묶음 추출은 두 묶음 앞서 읽는다)


def extract(n, gpus, limit, log_every=10, bs=32):
    SHM.mkdir(parents=True, exist_ok=True)
    for r in EXTRACT_REPS:
        np.lib.format.open_memmap(SHM / f"{r}.npy", mode="w+", dtype=np.float16, shape=(n, T, S, D))
    print(f"[extract] {n} clip x {len(EXTRACT_REPS)} 표현 {EXTRACT_REPS}, GPU {gpus} 장, {SHM} 에 {len(EXTRACT_REPS) * n * T * S * D * 2 / 2**30:.0f} GB", flush=True)
    t0 = time.time()
    mp.spawn(_worker, args=(gpus, n, limit, log_every, bs), nprocs=gpus, join=True)
    print(f"[extract] 끝 {time.time() - t0:.0f}s", flush=True)


# ─────────────────────────────────────────────────── probe 3 종




class Readout(nn.Module):
    """`attn` — attention pooling + head 2 개. 2026-09-23 에 자를 이걸로 확정했다.

        a      = softmax(tok @ q / √D)          (256,)   어디를 볼지
        pooled = a @ tok                        (1280,)
        xy     = Linear(pooled)                 (2,)     좌표 head
        pres   = Linear(pooled)                 ()       있음/없음 head

    고른 이유는 **v3 (측정셋) 에서 존재 판정이 전이된다**는 것이다. v3 는 물체가 내내 화면 안이라
    `z` 는 항상 "있다" 여야 하는데 (라벨 없는 검증), 그 오작동률은 `exp_results/windows/WINDOW_SUMMARY.md` 에 있다
    (자를 다시 배우면 바뀌는 수치라 코드에 적지 않는다).
    비교한 후보들과 기각 사유는 `z_research/RollOutV3/Archive/READOUT_CHOICE_2026-09-23.md`.
    **다시 비교하지 말 것.**

    ⚠️ 읽을 때 붙일 단서 셋 —
      · 좌표를 pooled 벡터에서 **회귀**하므로 학습 평균 쪽으로 수축한다 → 학습셋 test 에서 잰 치우침
        (`attn_bias_px.json`) 을 빼고 쓴다. 위치 오차는 물체 폭 36 px 의 절반 안팎이다
      · **`wall`·`ledge` 의 가능/불가능 판정은 이 자로 하지 않는다.** 분리량이 P=8 에서 28.7 px 로
        자 오차와 같은 규모라 잡음 안이고, 실제로 좌표 편향이 결론을 뒤집었다 (2026-09-23)
      · attention 지도(top1)는 v3 에서 평평하다 — **판정 기준으로 못 쓴다.** presence 값만 쓴다
    """

    def __init__(self, kind="attn", d=D, s_tok=S):
        super().__init__()
        assert kind == "attn", f"자는 attn 하나다 (받은 값: {kind})"
        self.kind = kind
        self.q = nn.Parameter(torch.randn(d) / d ** 0.5)
        self.out = nn.Linear(d, 2)                             # 좌표 head
        self.pres = nn.Linear(d, 1)                            # 있음/없음 head

    def forward(self, tok):                                    # (B,256,D) → xy (B,2), presence (B,), map (B,256)
        # ⚠️ 1/√D 는 곱하기 **전에** 가중치에 접어 넣는다 (fp16 출력 overflow 방지).
        #    tok 은 fp16 그대로 받고 큰 곱은 bmm 으로 — `.float()` 나 `(a[...,None]*tok).sum(1)` 은
        #    step 마다 336 MB 중간 텐서를 만든다 (2026-09-23 프로파일).
        dt = tok.dtype
        a = torch.softmax((tok @ (self.q / D ** 0.5).to(dt)).float(), -1)
        pooled = torch.bmm(a.to(dt).unsqueeze(1), tok).squeeze(1).float()
        return self.out(pooled), self.pres(pooled).squeeze(-1), a


class Shards:
    """특징 (n, 8, 256, 1280) fp16 을 GPU 에 올린다. **학습 clip 을 앞쪽 장치에 몰아서** 둔다.

    ⚠️ 2026-09-23 에 두 번 틀렸다. 기록해 둔다 —
      (1) 8 장에 흩어 놓고 batch 마다 cuda:0 으로 끌어모으기 → epoch 당 20.3 GB 가 PCIe 를 건너가
          9.7 GB/s 로 포화. 2.1 s/epoch. batch 를 키워도 총 바이트가 같아 안 빨라진다.
      (2) 자를 8 장에 복제해 제자리 계산 → 벌크 전송은 사라졌지만 step 마다 gradient 수집·파라미터
          배포로 **작은 복사 18,200 회/epoch** 가 생겨 5.8 s/epoch 로 더 느려졌다.
    답은 **매 epoch 읽는 데이터를 한 장치 안에 두는 것**이다. train 4,176 clip = 21.9 GB 라
    24 GB 한 장에 들어간다 (HBM ~1 TB/s → 22 ms). val·test 는 평가 때 한 번만 읽으니 뒤쪽에 둔다.

    배분은 연속 블록이고, 배치는 전체에서 뽑아 소유 장치별로 쪼개기만 하므로 (각 장치가 따로
    배치를 뽑는 게 아니다) 어느 clip 이 어느 장치에 있든 합친 gradient 는 같다.
    """

    def __init__(self, path, dev, gpus, priority=None, budget_gb=22.0, budgets=None):
        X = np.load(path, mmap_mode="r")
        self.n = len(X); self.dev = dev
        nd_max = max(1, min(gpus, torch.cuda.device_count()) if torch.cuda.is_available() else 1)
        per = X[0].nbytes                                          # clip 하나가 차지하는 바이트
        cap = max(1, int(budget_gb * 2**30 // per))                # 장당 clip 수 상한
        if budgets:
            # ⚠️ **장치별 예산 (2026-09-24)** — GPU 를 다른 작업과 나눠 쓸 때. 학습 clip 을 **앞 장치에 몰아** 싣는다.
            #    예: `CUDA_VISIBLE_DEVICES=6,7,0,...` + `--budgets 22,22,7,7,7,7,7,7` → 학습 7,180 clip 이 비어 있는
            #    6·7 번에만 있고 배치가 두 장에서만 돈다. val·test 는 평가 때 한 번 읽으니 공유 장치에 작게 둔다.
            #    (연속 블록 대신 재배치하면 memmap 을 건너뛰며 읽어 적재가 몇 분 느려진다 — 그 값을 치른다)
            caps = [max(1, int(b * 2**30 // per)) for b in budgets[:nd_max]]
            pri = np.where(priority)[0] if priority is not None else np.arange(0)
            order = np.r_[pri, np.setdiff1d(np.arange(self.n), pri)]
            blocks, k = [], 0
            for c in caps:
                if k >= self.n:
                    break
                blocks.append(order[k:k + c]); k += c
            cap = max(caps)
        # ⚠️ **연속 블록으로만 나눈다.** 학습 clip 을 앞으로 재배치하면 43.8 GB memmap 을 흩어진
        #    순서로 읽게 돼 로딩만 몇 분씩 걸린다 (2026-09-23 실측). 재배치는 필요도 없다 —
        #    계산을 제자리에서 하므로 어느 장치에 있든 epoch 당 전송이 0 이고, 예산을 키우면
        #    장치 수가 줄어 동기화 상대도 준다 (22 GB → 8,352 clip 이 2 장에).
        else:
            blocks = [np.arange(k, min(k + cap, self.n)) for k in range(0, self.n, cap)]
        pri = np.where(priority)[0] if priority is not None else np.arange(0)
        self.owner = np.full(self.n, -1, np.int64); self.slot = np.empty(self.n, np.int64)
        self.shards, self.devs = [], []
        for i, ids in enumerate(blocks[:nd_max]):
            ids = np.sort(ids)                                     # memmap 은 정렬 순서로 읽어야 빠르다
            d = torch.device("cuda", i) if torch.cuda.is_available() else torch.device("cpu")
            self.owner[ids] = i; self.slot[ids] = np.arange(len(ids))
            self.shards.append(torch.from_numpy(np.ascontiguousarray(X[ids])).to(d))
            self.devs.append(d)
        self.nd = len(self.shards)
        assert (self.owner >= 0).all(), f"clip 이 다 안 실렸다 — GPU {nd_max} 장, 장당 {cap} clip"
        ndtr = len(np.unique(self.owner[pri])) if len(pri) else self.nd
        print(f"    특징 {self.n} clip → GPU {self.nd} 장 (장당 ≤{cap} clip = {cap*per/2**30:.1f} GB). "
              f"학습 {len(pri)} clip 이 앞 {ndtr} 장에 — epoch 마다 장치 간 전송 {'없음' if ndtr == 1 else '거의 없음'}",
              flush=True)

    def split(self, b):
        """batch 를 장치별로 쪼갠다 → [(장치 i, 토큰 (m,8,256,D), 원래 clip 번호)]"""
        b = np.asarray(b); own = self.owner[b]
        for i in range(self.nd):
            sel = np.where(own == i)[0]
            if len(sel):
                loc = torch.as_tensor(self.slot[b[sel]], device=self.devs[i])
                yield i, self.shards[i][loc], b[sel]


def out_root(args):
    """⚠️ --smoke / --limit / --no-encoder 결과는 `_smoke/` 로 뺀다 (2026-09-20 에 배관 점검이 실물을 덮어썼다)."""
    return OUT / "_smoke" if getattr(args, "limit", 0) else OUT


def roc_auc(score, y):
    o = np.argsort(score); r = np.empty(len(score), float); r[o] = np.arange(1, len(score) + 1)
    p, n = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - p * (p + 1) / 2) / max(p * n, 1))


def pr_auc(score, y):
    o = np.argsort(-score); yt = y[o]
    tp = np.cumsum(yt); prec = tp / np.arange(1, len(yt) + 1); rec = tp / max(int(y.sum()), 1)
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))


def train(rep, L, pos, ign, tr, va, te, args, dev, log=None, Wc=None, Wb=None):
    # ⚠️ 파일로 리다이렉트하면 stdout 이 버퍼링돼 epoch 로그가 한참 뒤에야 보인다 (2026-09-23 '멈춘 것처럼 보임' 의 원인).
    log = log or (lambda *a: print(*a, flush=True))
    X = Shards(SHM / f"{rep}.npy", dev, args.gpus, priority=tr, budget_gb=args.budget_gb,
               budgets=[float(v) for v in args.budgets.split(",")] if args.budgets else None)
    # 라벨은 작아서 (수백 KB) 장치마다 복제한다
    Y = [torch.from_numpy(L).to(d) for d in X.devs]
    P = [torch.from_numpy(pos.astype(np.float32)).to(d) for d in X.devs]          # presence 라벨
    U = [torch.from_numpy((~ign).astype(np.float32)).to(d) for d in X.devs]       # 제외를 뺀 마스크
    # 가중치 (없으면 1). 좌표 손실 = cell x balance, presence 손실 = balance
    Wc = np.ones_like(pos, np.float32) if Wc is None else Wc
    Wb = np.ones(len(pos), np.float32) if Wb is None else Wb
    Wpos_np = (Wc * Wb[:, None]).astype(np.float32); Wpre_np = np.repeat(Wb[:, None], T, 1).astype(np.float32)
    WP = [torch.from_numpy(Wpos_np).to(d) for d in X.devs]
    WR = [torch.from_numpy(Wpre_np).to(d) for d in X.devs]
    res, saved = {}, {}
    torch.manual_seed(0)
    # 자를 장치마다 복제한다. 0 번이 master — gradient 를 여기 모아 step 하고 다시 뿌린다
    lam_of = lambda k: args.lam                    # λ 는 기본값 1.0 고정 (튜닝하지 않는다)
    models = {k: [Readout(k).to(d) for d in X.devs] for k in args.kinds}
    for k in models:
        for m in models[k][1:]:
            m.load_state_dict(models[k][0].state_dict())
    opts = {k: torch.optim.Adam(ms[0].parameters(), lr=args.lr) for k, ms in models.items()}
    tr_i = np.where(tr)[0]; t0 = time.time()
    pos_u = pos & ~ign                                                   # 좌표 손실에 쓰는 칸
    for ep in range(args.epochs):
        perm = np.random.default_rng(ep).permutation(tr_i)
        tot = {k: np.zeros(2) for k in models}
        for k0 in range(0, len(perm), args.bs):
            b = perm[k0:k0 + args.bs]
            # ⚠️ 분모는 **배치 전체** 기준이어야 한다. 장치별로 나누면 다른 손실이 된다
            # 가중 평균의 분모 = 배치 안 **가중치 합** (가중치가 전부 1 이면 옛 식과 같다)
            den_pos = max(float((pos_u[b] * Wpos_np[b]).sum()), 1e-6)
            den_pre = max(float(((~ign[b]) * Wpre_np[b]).sum()), 1e-6)
            parts = list(X.split(b))
            for kind, ms in models.items():
                for m in ms:
                    m.zero_grad(set_to_none=True)
                for i, tok_i, b_i in parts:
                    tok = tok_i.reshape(-1, S, D)          # fp16 그대로 넘긴다
                    bd = torch.as_tensor(b_i, device=X.devs[i])
                    y = Y[i][bd].reshape(-1, 2); p = P[i][bd].reshape(-1); u = U[i][bd].reshape(-1)
                    wp = WP[i][bd].reshape(-1); wr = WR[i][bd].reshape(-1)
                    xy, pres, _ = ms[i](tok)
                    s_pos = (((xy - y) ** 2).mean(1) * (p * u * wp)).sum()
                    s_pre = (F.binary_cross_entropy_with_logits(pres, p, reduction="none") * u * wr).sum()
                    (s_pos / den_pos + lam_of(kind) * s_pre / den_pre).backward()
                    tot[kind] += [float(s_pos.detach()) / den_pos * len(b), float(s_pre.detach()) / den_pre * len(b)]
                # ⚠️ 동기화는 **납작한 벡터 하나**로. 텐서별로 복사하면 step 마다 복사 횟수가 5 배가 되고
                #    그게 epoch 시간을 지배한다 (2026-09-23: 18,200 회/epoch → 5.8 s/epoch).
                if len(ms) > 1:
                    gv = parameters_to_vector([p.grad if p.grad is not None else torch.zeros_like(p)
                                               for p in ms[0].parameters()])
                    for m in ms[1:]:
                        gv += parameters_to_vector([p.grad if p.grad is not None else torch.zeros_like(p)
                                                    for p in m.parameters()]).to(gv.device)
                    # ⚠️ 이 배치에 마스터 장치 몫이 하나도 없으면 마스터 grad 가 None 이다 — 0 으로 만들어 두고 합을 쓴다.
                    #    장치 2 장일 때는 안 났다. 8 장으로 펴자 (7/8)^32 ≈ 1.4 %/배치로 매 epoch 났다 (2026-09-24).
                    for p_ in ms[0].parameters():
                        if p_.grad is None:
                            p_.grad = torch.zeros_like(p_)
                    vector_to_parameters(gv, [p.grad for p in ms[0].parameters()])
                opts[kind].step()
                if len(ms) > 1:
                    with torch.no_grad():
                        pv = parameters_to_vector(ms[0].parameters())
                        for m in ms[1:]:
                            vector_to_parameters(pv.to(next(m.parameters()).device), m.parameters())
        if ep % 20 == 0 or ep == args.epochs - 1:
            log(f"    [{rep}] epoch {ep:3d}  " + "  ".join(f"{k} xy {v[0]/len(perm):.4f} pres {v[1]/len(perm):.4f}"
                                                           for k, v in tot.items()) + f"  ({time.time()-t0:.0f}s)")
    # ── 평가 ──
    emp = np.array([r["scenario"] == "empty" or r.get("shape_pre") == "none" for r in ROWS])[:, None].repeat(T, 1)
    inf_ = np.stack([arr(r["in_frame_by_sample"]) for r in ROWS]).reshape(len(ROWS), 16, 2).all(2)[:, T:]
    neg = (~pos) & (~ign)
    src = {"빈 장면": neg & emp, "화면 밖": neg & ~emp & ~inf_}
    for kind, ms in models.items():
        m = ms[0]
        for mm in ms:
            mm.eval()
        XY = np.empty((X.n, T, 2), np.float32); PR = np.empty((X.n, T), np.float32)
        A1 = np.empty((X.n, T), np.float32)          # 상위 1 토큰 질량 (균등 = 1/256 = 0.0039). map 이 날카로운가
        with torch.no_grad():                        # 복제본은 파라미터가 같으니 각자 자기 샤드를 읽는다
            for i in range(X.nd):
                ids = np.where(X.owner == i)[0]
                for k0 in range(0, len(ids), args.bs):
                    b = ids[k0:k0 + args.bs]
                    loc = torch.as_tensor(X.slot[b], device=X.devs[i])
                    xy, pres, amap = ms[i](X.shards[i][loc].reshape(-1, S, D))
                    XY[b] = xy.reshape(-1, T, 2).cpu().numpy(); PR[b] = pres.reshape(-1, T).cpu().numpy()
                    A1[b] = amap.max(-1).values.reshape(-1, T).cpu().numpy()
        err = np.linalg.norm(XY - L, axis=-1) * RES / CELL                       # 칸 단위
        # 문턱은 **val 에서** 전체 음성의 5 % 오탐 지점으로 정한다
        thr = float(np.quantile(PR[va][neg[va]], 0.95)) if neg[va].any() else 0.0
        sc_te, y_te = PR[te][pos[te] | neg[te]], pos[te][pos[te] | neg[te]]
        rec = dict(n_params=int(sum(p.numel() for p in m.parameters())),
                   # 중심 오차: **모든 GT 양성** (presence 판정과 무관)
                   center_mean=float(err[te][pos[te]].mean()), center_med=float(np.median(err[te][pos[te]])),
                   center_p90=float(np.percentile(err[te][pos[te]], 90)),
                   roc_auc=roc_auc(sc_te, y_te), pr_auc=pr_auc(sc_te, y_te), thr_val_fpr5=thr,
                   fpr_by_source={k: float((PR[te][v[te]] > thr).mean()) if v[te].any() else None for k, v in src.items()},
                   recall_at_thr=float((PR[te][pos[te]] > thr).mean()),
                   # attention 집중도: 상위 1 토큰 질량 (균등 0.0039). 양성에서 높고 음성에서 낮아야 map 이 뜻을 가진다
                   attn_top1_pos=float(A1[te][pos[te]].mean()) if pos[te].any() else None,
                   attn_top1_neg=float(A1[te][neg[te]].mean()) if neg[te].any() else None,
                   # 출처별 AUC = **같은 양성 집합 + 그 음성 출처** 조합
                   auc_by_source={k: roc_auc(np.r_[PR[te][pos[te]], PR[te][v[te]]],
                                             np.r_[np.ones(int(pos[te].sum()), bool), np.zeros(int(v[te].sum()), bool)])
                                  if v[te].any() else None for k, v in src.items()})
        # 무대 가족별 (test). **이 데이터의 요점** — 구조물 가족에서도 recall·오탐이 평면과 같은가
        fam = np.array([family(r["scenario"]) for r in ROWS])[:, None].repeat(T, 1)
        rec["by_family"] = {}
        for fm in sorted(set(fam[:, 0])):
            mp_, mn_ = te[:, None] & pos & (fam == fm), te[:, None] & neg & (fam == fm)
            rec["by_family"][fm] = dict(
                n_pos=int(mp_.sum()), n_neg=int(mn_.sum()),
                recall=float((PR[mp_] > thr).mean()) if mp_.any() else None,
                fpr=float((PR[mn_] > thr).mean()) if mn_.any() else None,
                center_mean=float(err[mp_].mean()) if mp_.any() else None)
        res[kind] = rec; saved[kind] = (XY, PR)
        log(f"  [{rep}/{kind:6s}] {rec['n_params']:>9,}p  중심 {rec['center_mean']:.2f}/{rec['center_med']:.2f}/{rec['center_p90']:.2f} 칸  "
            f"ROC {rec['roc_auc']:.3f} PR {rec['pr_auc']:.3f}  " +
            f"top1 {rec['attn_top1_pos']:.3f}/{rec['attn_top1_neg']:.3f}  오탐 " +
            " ".join(f"{k} {100*v:.1f}" for k, v in rec["fpr_by_source"].items() if v is not None) +
            f"  (양성 유지 {100*rec['recall_at_thr']:.1f})")
        d = out_root(args) / rep; d.mkdir(parents=True, exist_ok=True)
        torch.save(m.state_dict(), d / f"readout_{kind}.pt")
        np.savez(d / f"preds_{kind}.npz", pred=XY, presence=PR, truth=L, pos=pos, ign=ign,
                 train=tr, val=va, test=te, video_id=np.array([r["video_id"] for r in ROWS]),
                 scenario=np.array([r["scenario"] for r in ROWS]))
    del X, models; torch.cuda.empty_cache()
    return res


def main():
    global SHM, ROWS                   # --feat-dir 로 특징 위치를 바꾼다
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="+", default=list(REPS), choices=list(REPS))
    ap.add_argument("--kinds", nargs="+", default=["attn"], choices=["attn"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lam", type=float, default=1.0, help="presence 손실 가중")
    ap.add_argument("--gpus", type=int, default=torch.cuda.device_count() or 1)
    ap.add_argument("--limit", type=int, default=0, help="clip 수 제한 (배관 점검)")
    ap.add_argument("--budget-gb", type=float, default=22.0, help="GPU 장당 특징 예산")
    ap.add_argument("--budgets", default=None,
                    help="장치별 예산 GB (쉼표). 주면 학습 clip 을 앞 장치에 몰아 싣는다. 예: 22,22,7,7,7,7,7,7")
    ap.add_argument("--feat-dir", default=None,
                    help=f"특징 저장 위치 (기본 {SHM}). 재사용하려면 디스크로: {FEAT_DEFAULT_DISK}")
    ap.add_argument("--keep-shm", action="store_true", help="끝나고 특징을 지우지 않는다")
    ap.add_argument("--skip-extract", action="store_true", help="/dev/shm 에 이미 있으면 재사용")
    ap.add_argument("--extract-only", action="store_true", help="특징만 뽑고 끝낸다 (자 학습은 rollout2_identity_readout.py 로)")
    ap.add_argument("--extract-bs", type=int, default=32, help="추출 묶음 (clip 수). 16 이상이면 GPU 가 포화된다 (2026-09-25 실측)")
    ap.add_argument("--set", default=SET, help="학습셋 (data_csv/rollout_v2_<set>). 기본 training_v8")
    ap.add_argument("--no-weights", action="store_true",
                    help="cell_weight / balance_weight 를 쓰지 않는다 (대조용). 기본은 데이터가 주면 쓴다")
    a = ap.parse_args()
    if a.set != SET:                                     # 워커도 같은 세트를 읽도록 환경변수로 싣고 다시 부른다
        os.environ["PRESENCE_SET"] = a.set
        os.execv(sys.executable, [sys.executable] + sys.argv)
    if a.feat_dir:
        os.environ["PRESENCE_FEAT_DIR"] = a.feat_dir      # 워커가 다시 import 할 때 읽는다
        SHM = Path(a.feat_dir)
    print(f"[feat] {SHM}", flush=True)

    ROWS, L, pos, ign = labels()
    Wc, Wb = weights(ROWS)
    if a.no_weights:
        Wc, Wb = np.ones_like(Wc), np.ones_like(Wb)
    use_w = not (np.all(Wc == 1) and np.all(Wb == 1))
    if a.limit:
        keep = np.r_[np.arange(a.limit // 2), np.arange(len(ROWS) - a.limit // 2, len(ROWS))]
        ROWS = [ROWS[i] for i in keep]; L, pos, ign = L[keep], pos[keep], ign[keep]
        Wc, Wb = Wc[keep], Wb[keep]
    print(f"[set] {SET}  {INDEX}   가중치 {'사용 (cell x balance / balance)' if use_w else '안 씀'}", flush=True)
    n = len(ROWS)
    emp = np.array([r["scenario"] == "empty" or r.get("shape_pre") == "none" for r in ROWS])   # training_r8: 빈 장면이 arm 안에 있다
    neg = (~pos) & (~ign)
    print(f"[data] {n} clip (빈 장면 {emp.sum()}), 미래 튜블릿 {n*T}  "
          f"양성 {int(pos.sum())} / 음성 {int(neg.sum())} / 제외 {int(ign.sum())}", flush=True)

    # block 단위 split (scenario 층화): train 50 % / val 15 % / test 35 %
    blk = np.array([r["block_id"] for r in ROWS]); sc = np.array([r["scenario"] for r in ROWS])
    rng = np.random.RandomState(0); part = {}
    for s_ in np.unique(sc):
        b = rng.permutation(np.unique(blk[sc == s_])); n1 = int(round(len(b) * .5)); n2 = int(round(len(b) * .65))
        for x in b[:n1]: part[x] = 0
        for x in b[n1:n2]: part[x] = 1
        for x in b[n2:]: part[x] = 2
    g = np.array([part[x] for x in blk])
    tr, va, te = g == 0, g == 1, g == 2
    print(f"[split] train {tr.sum()} / val {va.sum()} / test {te.sum()} clip (block 단위, scenario 층화)", flush=True)

    if not a.skip_extract:
        extract(n, a.gpus, a.limit, bs=a.extract_bs)
    if a.extract_only:
        print(f"[extract-only] → {SHM}", flush=True); return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rep_res = {}
    for rep in a.reps:
        print(f"[train] {rep}", flush=True)
        rep_res[rep] = train(rep, L, pos, ign, tr, va, te, a, dev, Wc=Wc, Wb=Wb)

    o = out_root(a); o.mkdir(parents=True, exist_ok=True)
    # ⚠️ 부분 실행 (--reps/--kinds 일부) 이 전체 문서를 지우지 않도록 **병합**한다.
    #    2026-09-23 에 `--kinds linear --reps z h` 재학습이 끝나면서 9 줄짜리 표를 2 줄로 덮어썼다.
    prev = json.loads((o / "summary.json").read_text()) if (o / "summary.json").exists() else {}
    merged = dict(prev.get("reps", {}))
    for rep, kk in rep_res.items():
        merged[rep] = {**merged.get(rep, {}), **kk}
    rep_res = {r: merged[r] for r in ("p", "z", "h") if r in merged}
    (o / "summary.json").write_text(json.dumps(dict(
        n_clip=n, n_empty=int(emp.sum()), n_train=int(tr.sum()), n_val=int(va.sum()), n_test=int(te.sum()),
        n_pos=int(pos.sum()), n_neg=int(neg.sum()), n_ignore=int(ign.sum()),
        epochs=a.epochs, lr=a.lr, lam=a.lam, train_set=SET, weights=use_w, reps=rep_res), indent=1))
    lines = [AUTO0, "# 화면 안 물체의 **존재와 중심**을 frozen feature 에서 읽을 수 있는가 (가림 없는 조건)", "",
             f"학습셋 `{SET}`, {n} clip (빈 장면 {int(emp.sum())}). 가중치 {'사용' if use_w else '안 씀'}. block 단위 split — train {int(tr.sum())} / val {int(va.sum())} / test {int(te.sum())}.",
             f"미래 튜블릿 라벨: 양성 {int(pos.sum())} · 음성 {int(neg.sum())} · 제외 {int(ign.sum())} (화면 안인데 장애물과 겹침).",
             "중심 오차는 **모든 GT 양성**에서 재고 (presence 판정과 무관), 1 칸 = 18 px. 문턱은 **val 에서 음성 5 % 오탐** 지점 하나를 test 에 그대로 쓴다.",
             "⚠️ 이 세트에는 가림이 없다 — 결론은 \"가림 없는 조건\" 으로 한정한다.", "",
             "`top1` = 상위 1 토큰 attention 질량 (양성/음성). 균등 = 1/256 = 0.0039 — 클수록 map 이 날카롭다.", "",
             "| 표현 | probe | 파라미터 | 중심 평균 | 중앙값 | 상위 90% | ROC-AUC | PR-AUC | 빈 장면 오탐 | 화면 밖 오탐 | 양성 유지 | top1 양성 | top1 음성 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for rep, kk in rep_res.items():
        for kind, r in kk.items():
            f = lambda v: "  -  " if v is None else f"{100*v:.1f}"
            lines.append(f"| {rep} | {kind} | {r['n_params']:,} | {r['center_mean']:.2f} | {r['center_med']:.2f} | "
                         f"{r['center_p90']:.2f} | {r['roc_auc']:.3f} | {r['pr_auc']:.3f} | "
                         f"{f(r['fpr_by_source'].get('빈 장면'))} | {f(r['fpr_by_source'].get('화면 밖'))} | {100*r['recall_at_thr']:.1f} | "
                         + " | ".join("  -  " if r.get(k) is None else f"{r[k]:.3f}"
                                      for k in ("attn_top1_pos", "attn_top1_neg")) + " |")
    lines += ["", "출처별 AUC (같은 양성 집합 + 그 음성 출처 조합):", "",
              "| 표현 | probe | 빈 장면 | 화면 밖 |", "|---|---|---:|---:|"]
    for rep, kk in rep_res.items():
        for kind, r in kk.items():
            g_ = lambda v: "  -  " if v is None else f"{v:.3f}"
            lines.append(f"| {rep} | {kind} | {g_(r['auc_by_source'].get('빈 장면'))} | {g_(r['auc_by_source'].get('화면 밖'))} |")
    lines += ["", AUTO1, ""]
    # AUTO 블록 밖(손으로 쓴 절·재현)은 보존한다 — 자동 표만 갈아 끼운다
    tail = DEFAULT_TAIL
    if (o / "RESULTS.md").exists():
        cur = (o / "RESULTS.md").read_text()
        if AUTO1 in cur:
            tail = cur.split(AUTO1, 1)[1].lstrip("\n")
    (o / "RESULTS.md").write_text("\n".join(lines) + tail)
    print(f"[done] {o}/RESULTS.md", flush=True)
    if not a.keep_shm and str(SHM).startswith("/dev/shm"):
        shutil.rmtree(SHM, ignore_errors=True)          # 디스크에 둔 특징은 자동 삭제하지 않는다


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
