#!/usr/bin/env python3
"""[P] H2 (조건) 측정용 캐시 — SSv2 를 행동 라벨로 "수동 물리" / "행위자" 부류로 나눠 블록 토큰 + 칸 흐름 (auto_research, 2026-10-21).
읽기 · 인코딩 · 흐름은 [T] 의 cache_natural.py (12:45 판) 와 같다: 4 fps 간격 (표의 간격 열), 높이 전체 가운데 정사각 → 256 bilinear → uint8 반올림,
블록 토큰 = 블록별 LN(target_encoder) bf16, 흐름 = RAFT-large, 블록 대표 장 (둘째 장) 사이 역방향 (블록 b → b−1), 16 px 칸 평균 (칸 단위).

  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=5 /venv/vjepa/bin/python -W ignore auto_research/scripts/cache_ssv2_classes.py --n 300

부류 (SSv2 템플릿 이름, 근거 = 그 영상에서 물체의 이동을 정하는 것이 무엇인가):
  수동 물리 (passive): 손을 떠난 뒤 중력 · 관성 · 충돌이 이동을 정한다 — 떨어뜨리기 5 · 굴러가게 두기 3 · 들었다 떨어뜨리기 · 떨어짐 2 ·
    던지기 4 (받기 제외) · 굴리기 · 충돌 · 튕김 2 · 밀어서 테이블 밖으로 떨어뜨리기 · 받침 없이 놓아 떨어짐 2 · 미끄러져 내려감 2.
  행위자 (actor): 손이 물체를 쥐거나 계속 밀며 이동을 정한다 — 옮기기 위 · 아래 · 카메라 쪽 · 멀리 · 다른 것 쪽 · 다른 것에서 멀리 ·
    밀기 좌우 2 · 당기기 좌우 2 · 두 물체 모으기 · 벌리기 · 들어 올리고 안 떨어뜨리기 · 쥐고 있기 5.
창: 4 fps 로 16 장 (블록 8 개), 영상 가운데. 저장: <out>/{group}_h.npy (N,8,256,1280) fp16 · {group}_U.npy (N,8,16,16,2) fp16 (U[:,0]=0) · {group}_meta.json"""
import argparse, json, os, random, sys
import numpy as np, torch, torch.nn.functional as F
import decord

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--nb", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="/workspace/auto_research_cache/ssv2_classes")
a = ap.parse_args()
torch.set_num_threads(8)
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _build_encoder, _load_state_dict, _load_checkpoint   # noqa: E402
from app.vjepa_frozen import ar as AR                                                    # noqa: E402
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights               # noqa: E402

PASSIVE = ["Dropping something behind something", "Dropping something in front of something", "Dropping something into something",
           "Dropping something next to something", "Dropping something onto something",
           "Letting something roll along a flat surface", "Letting something roll down a slanted surface",
           "Letting something roll up a slanted surface, so it rolls back down", "Lifting something up completely, then letting it drop down",
           "Something falling like a feather or paper", "Something falling like a rock",
           "Throwing something", "Throwing something against something", "Throwing something in the air and letting it fall", "Throwing something onto a surface",
           "Rolling something on a flat surface", "Something colliding with something and both are being deflected", "Something being deflected from something",
           "Pushing something so that it falls off the table", "Putting something on the edge of something so it is not supported and falls down",
           "Putting something onto something else that cannot support it so it falls down",
           "Lifting a surface with something on it until it starts sliding down", "Putting something that can't roll onto a slanted surface, so it slides down"]
ACTOR = ["Moving something up", "Moving something down", "Moving something towards the camera", "Moving something away from the camera",
         "Moving something closer to something", "Moving something away from something",
         "Pushing something from left to right", "Pushing something from right to left", "Pulling something from left to right", "Pulling something from right to left",
         "Moving something and something closer to each other", "Moving something and something away from each other",
         "Lifting something up completely without letting it drop down",
         "Holding something", "Holding something next to something", "Holding something in front of something", "Holding something behind something",
         "Holding something over something"]
dev = torch.device("cuda")
NF = 2 * a.nb
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")
enc = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2, use_rope=True, uniform_power=False)
_load_state_dict(enc, _load_checkpoint(CK)["target_encoder"], tag="t"); enc = enc.to(dev, torch.bfloat16).eval()
raft = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(dev).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)

lab = {}
for split in ("train", "validation"):
    for x in json.load(open(f"/workspace/dataset/training_data/SSv2/labels/{split}.json")):
        lab[x["id"]] = (x["template"].replace("[", "").replace("]", ""), split)
tab = [l.rstrip("\n").split("\t") for l in open(os.path.join(REPO, "data_csv/ssv2/fps4_table.tsv"))]
os.makedirs(a.out, exist_ok=True)
for group, classes in (("passive", PASSIVE), ("actor", ACTOR)):
    cs = set(classes)
    rows = [r for r in tab if int(r[4]) >= NF and lab.get(os.path.basename(r[0])[:-5], ("", ""))[0] in cs]
    rng = random.Random(a.seed); rng.shuffle(rows)
    print(f"{group}: 적격 영상 {len(rows)} (부류 {len(classes)} 개)", flush=True)
    H, UU, META = [], [], []
    for r in rows:
        if len(H) >= a.n:
            break
        try:
            vr = decord.VideoReader(r[0], num_threads=1)
            fstp = int(r[3]); span = (NF - 1) * fstp + 1
            if len(vr) < span:
                continue
            s = (len(vr) - span) // 2                                                   # 영상 가운데 창
            fr = vr.get_batch(list(range(s, s + span, fstp))).asnumpy()
        except Exception:
            continue
        Hh, Ww = fr.shape[1:3]
        side = min(Hh, Ww); y0, x0 = (Hh - side) // 2, (Ww - side) // 2
        x = torch.from_numpy(fr[:, y0:y0 + side, x0:x0 + side]).to(dev).permute(0, 3, 1, 2).float()
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False).clamp(0, 255).round()
        with torch.no_grad():
            xx = (x / 255).permute(1, 0, 2, 3)[None]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = AR.encode_blocks(enc, ((xx - MEAN) / STD).to(torch.bfloat16), 2)[0].view(a.nb, 256, -1)
            rep = x[1::2] / 127.5 - 1
            f = raft(rep[1:], rep[:-1], num_flow_updates=12)[-1]
            u = F.avg_pool2d(f, 16) / 16
        U = torch.zeros(a.nb, 16, 16, 2, device=dev); U[1:] = u.permute(0, 2, 3, 1)
        vid = os.path.basename(r[0])[:-5]
        H.append(h.half().cpu().numpy()); UU.append(U.half().cpu().numpy()); META.append(dict(path=r[0], start=s, fstp=fstp, label=lab[vid][0], split=lab[vid][1]))
        if len(H) % 50 == 0:
            print(f"  {group} {len(H)}/{a.n}", flush=True)
    np.save(os.path.join(a.out, f"{group}_h.npy"), np.stack(H)); np.save(os.path.join(a.out, f"{group}_U.npy"), np.stack(UU))
    json.dump(META, open(os.path.join(a.out, f"{group}_meta.json"), "w"))
    print(f"{group} done {len(H)}", flush=True)
