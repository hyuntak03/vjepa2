"""VOBench 물체 칸 encoder 특징 캐시 — 정체 개념 벡터를 VOBench 자체로 뽑을 수 있나 검증용 (2026-10-26 저녁, GPU 1 장 · 6 GB).
사용자: "VOBench 에서도 concept vector 뽑을 수 있어? 가능한지 검증해 봐".
영상: VOBench_v1 도형 (primitive) 블록 전부 (보임 · 이른 · 중간 · 끝 가림), 가능 영상 pos_a (물체 a, 라벨 A) · pos_b (물체 b, 라벨 B). 빈 물체는 뺌.
원본 0, 3, …, 93 장 (512² → 256² bilinear, antialias 없음), 블록 (2 장) 마다 LN(target_encoder) (ar_layer_cache.py 와 같은 방식).
블록 0–15 중 라벨의 물체가 두 장 모두 있고 가려진 비율 < 0.5 인 블록에서: 물체 칸 평균 (물체 상자 안에 중심이 든 16×16 칸) · 화면 전체 평균.
  CUDA_VISIBLE_DEVICES=0 MEM_GB=6 /venv/vjepa/bin/python -W ignore z_research/scripts/analysis/vobench_concept_cache.py
출력: /workspace/auto_research_cache/objcell_concept/vobench_all.npz — obj (N, 1280) · whole (N, 1280) fp16 · 라벨 (모양 · 색 · block_id · 가림 조건 · 배경 · 팔 · 블록 · 칸 수)
"""
import csv, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

REPO = "/workspace/world_model/vjepa2"; sys.path.insert(0, REPO)
dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEM_GB", 6)) * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory, 0)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint
VROOT = "/workspace/dataset/benchmarks/VOBench_v1"; IDX = list(range(0, 96, 3))[:32]; D = 1280
enc2 = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
_load_state_dict(enc2, _load_checkpoint(CK)["target_encoder"], tag="t"); enc2 = enc2.half().to(dev).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(3, 1, 1, 1)
rows = [r for r in csv.DictReader(open(f"{REPO}/data_csv/vobench_v1/index.csv")) if r["objset"] == "primitive" and r["variant"] in ("pos_a", "pos_b")]
cc = np.arange(16) * 16 + 8.0; OBJ, WH, LAB = [], [], []; t0 = time.time()
for n, r in enumerate(rows):
    obj = r["obj_a"] if r["variant"] == "pos_a" else r["obj_b"]
    if obj in ("", "empty"):
        continue
    folder = os.path.join(VROOT, r["file_name"]); W0 = Image.open(os.path.join(folder, "000000.png")).size[0]; sc = 256 / W0
    lab = {q["f"]: q for q in json.load(open(os.path.join(os.path.dirname(folder), "label.json")))["labels"].get("A" if r["variant"] == "pos_a" else "B", [])}
    fr = np.stack([np.asarray(Image.open(os.path.join(folder, f"{f:06d}.png")).convert("RGB")) for f in IDX])
    x = torch.from_numpy(fr).permute(3, 0, 1, 2).float().to(dev) / 255.0
    x = F.interpolate(x.permute(1, 0, 2, 3), size=(256, 256), mode="bilinear", align_corners=False, antialias=False).permute(1, 0, 2, 3)
    xn = (x - MEAN) / STD
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        hb = torch.cat([F.layer_norm(enc2(xn[:, 4 * q: 4 * q + 4].reshape(3, 2, 2, 256, 256).permute(1, 0, 2, 3, 4).half()).float(), (D,)) for q in range(8)]).cpu().numpy()   # (16, 256, 1280)
    shape, color = obj.split("/")
    for b in range(16):
        q = [lab.get(IDX[2 * b]), lab.get(IDX[2 * b + 1])]
        if any(t is None for t in q) or max(t.get("hidden", 0) for t in q) >= 0.5:
            continue
        x0 = min(t["u"] - t["ext_u"] for t in q) * sc; x1 = max(t["u"] + t["ext_u"] for t in q) * sc; y0 = min(t["v"] - t["ext_v"] for t in q) * sc; y1 = max(t["v"] + t["ext_v"] for t in q) * sc
        cells = [int(rr * 16 + c) for rr in np.nonzero((cc >= y0) & (cc <= y1))[0] for c in np.nonzero((cc >= x0) & (cc <= x1))[0]]
        if not cells:
            continue
        OBJ.append(hb[b, cells].mean(0)); WH.append(hb[b].mean(0))
        LAB.append((shape, color, r["block_id"], r["occlusion"], r["background"], r["variant"], b, len(cells), r["motion"]))
    if n % 100 == 0:
        print(time.strftime("%H:%M:%S"), f"{n}/{len(rows)} 표본 {len(OBJ)} ({time.time() - t0:.0f}s)", flush=True)
L = np.array(LAB, dtype=object)
np.savez("/workspace/auto_research_cache/objcell_concept/vobench_all.npz", obj=np.stack(OBJ).astype(np.float16), whole=np.stack(WH).astype(np.float16),
         shape=L[:, 0].astype(str), color=L[:, 1].astype(str), block_id=L[:, 2].astype(str), occlusion=L[:, 3].astype(str), background=L[:, 4].astype(str),
         arm=L[:, 5].astype(str), block=L[:, 6].astype(int), ncell=L[:, 7].astype(int), motion=L[:, 8].astype(str))
print("저장, 표본", len(OBJ), f"({time.time() - t0:.0f}s)")
