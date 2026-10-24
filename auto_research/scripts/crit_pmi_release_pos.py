"""[비판] 관문 8 도구 검사 (릴리즈 판): 영상 모드 목표 토큰에 "창 안 시간 위치" 지문이 있어, 위치만 알아도 후보를 고를 수 있나.
릴리즈 predictor 는 목표 자리 (튜블릿 t+1) 를 시간 위치로 묻는다. 후보 토큰 (목표 encoder 를 창 20 장에 영상 모드로) 이 자리마다 다른 꼴을 가지면,
위치만 맞혀도 순위가 오른다. 문맥 없는 "위치 판정자" 를 같은 후보 묶음 (Random(1)) 에 건다:
  위치 틀 = 다른 창들의 같은 자리 다시 인코딩 토큰 평균 (그 창 자신은 뺌). 후보 중 틀 (t+1 자리) 에 가장 가까운 것을 고른다.
  (a) 판: 참 h'_t+1 이 멈춤 h'_t 보다 틀 (t+1) 에 가까운 비율.
블록 공간 (블록마다 2 장을 따로 인코딩) 은 위치 지문이 원리상 없어 이 검사가 필요 없다 (블록 AR · 영상 문맥 판의 후보 공간).
"""
import os, sys, json, random, numpy as np, torch, torch.nn.functional as F, decord

REPO = "/workspace/world_model/vjepa2"
sys.path.insert(0, REPO)
from analysis.intphys2.model import build_from_config, _build_encoder, _load_state_dict, _load_checkpoint  # noqa: E402
from app.vjepa_frozen import ar as AR  # noqa: E402

S, NB = 256, 10
C_T = [3, 4, 5, 6, 7, 8]; B_T = [3, 4, 5, 6]
CK = os.path.join(REPO, "checkpoint/models--facebook--vjepa2-vith-fpc64-256/snapshots/b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5/original/model.pth")


def boot(x, rng, n=4000):
    x = np.asarray(x, float); m = len(x)
    bs = np.sort([x[rng.integers(0, m, m)].mean() for _ in range(n)])
    return [float(x.mean()), float(bs[int(0.025 * n)]), float(bs[int(0.975 * n) - 1])]


def main():
    torch.set_num_threads(8); torch.manual_seed(0)
    dev = torch.device("cuda")
    sd = _load_checkpoint(CK)
    tv = _build_encoder(arch_name="vit_huge", img_size=256, patch_size=16, tubelet_size=2, num_frames=2 * NB, use_rope=True, uniform_power=False)
    _load_state_dict(tv, sd["target_encoder"], tag="tv"); tv = tv.to(dev).eval()
    del sd
    MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)
    cache = "/workspace/auto_research_cache/pmi_cont"
    res = {}
    for st in ("k710_320_holdout", "ssv2"):
        meta = json.load(open(os.path.join(cache, f"{st}_meta.json"))); N = len(meta)
        crng = random.Random(1)
        HV, HB = [], []                                                          # 영상 모드 다시 인코딩 (N,10,S,D) · 블록 공간 다시 인코딩
        for i in range(N):
            m = meta[i]
            vr = decord.VideoReader(m["path"], num_threads=1)
            fr = vr.get_batch(list(range(m["start"], m["start"] + (2 * NB - 1) * m["fstp"] + 1, m["fstp"]))).asnumpy()
            Hh, Ww = fr.shape[1:3]; side = min(Hh, Ww); y0, x0 = (Hh - side) // 2, (Ww - side) // 2
            x = torch.from_numpy(fr[:, y0:y0 + side, x0:x0 + side]).to(dev).permute(0, 3, 1, 2).float()
            x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False).clamp(0, 255).round()
            xr = (x + torch.randint(0, 2, x.shape, device=dev).float() * 2 - 1).clamp(0, 255)
            xn = ((xr / 255).permute(1, 0, 2, 3)[None] - MEAN) / STD
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                hv = tv(xn)
            HV.append(F.layer_norm(hv.float(), (hv.size(-1),))[0].view(NB, S, -1).half().cpu())
            if i % 50 == 0:
                print(st, i, flush=True)
        HV = torch.stack(HV).float()                                              # (N,10,S,D)
        cands = [{t: sorted([t + 1] + crng.sample([k for k in range(NB) if k != t + 1], 7)) for t in C_T} for _ in range(N)]
        tot = HV.sum(0)                                                           # (10,S,D)
        brng = np.random.default_rng(0)
        acc, win = [], []
        for i in range(N):
            tmpl = (tot - HV[i]) / (N - 1)                                        # 자기 창 뺀 자리 틀
            a_ = []
            for t in C_T:
                d = {k: float((HV[i, k] - tmpl[t + 1]).abs().mean()) for k in cands[i][t]}
                a_.append(float(min(d, key=d.get) == t + 1))
            acc.append(np.mean(a_))
            win.append(np.mean([float((HV[i, t + 1] - tmpl[t + 1]).abs().mean() < (HV[i, t] - tmpl[t + 1]).abs().mean()) for t in B_T]))
        # 자리 틀끼리 얼마나 다른가: 이웃 자리 틀 차이 / 창 사이 같은 자리 퍼짐
        tm = tot / N
        adj = float(np.mean([(tm[k + 1] - tm[k]).abs().mean().item() for k in range(NB - 1)]))
        spread = float((HV - tm[None]).abs().mean().item())
        res[st] = {"pos_judge_rank": boot(acc, brng), "pos_judge_true_vs_stop": boot(win, brng), "template_adjacent_diff": adj, "within_position_spread": spread}
        print(st, json.dumps(res[st]), flush=True)
    json.dump(res, open(os.path.join(REPO, "auto_research/results/crit_pmi_release_pos.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
