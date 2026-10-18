"""predictor_group (frozen encoder 공유) 가 모델을 따로 돌린 것과 **같은 값**을 내는지 검사한다 (2026-10-18).

CPU 로 돈다 — GPU 를 쓰는 학습을 방해하지 않기 위해서다. 두 가지를 본다.

  1. **memo 등가** — group 의 forward (encoder 결과를 batch 안에서 한 번만 계산) 와
     share 를 끈 forward (모델마다 encoder 를 다시 돌림) 가 비트 단위로 같은가.
     공유하는 값은 `x_full` (창 전체 인코딩) 과 `hb` (AR kind 의 블록별 인코딩) 둘이고
     둘 다 encoder 와 입력에만 의존하므로 같아야 한다.
  2. **가중치 오염 없음** — group 의 각 predictor 가 그 모델을 혼자 지었을 때와 같은 state_dict 인가.
     예전 `init_module` 은 `checkpoint[predictor] = predictor.state_dict()` 로 공유 checkpoint 를
     덮어썼다 — group 에서는 그게 뒤에 오는 **릴리즈 predictor 를 앞 모델 가중치로** 바꿔 버린다.

  사용법:  python z_vll7_training/tests/ek100_group_equiv.py <resolved.yaml>
"""
import hashlib
import sys

import torch
import yaml

sys.path.insert(0, ".")
from evals.action_anticipation_frozen.models import init_module, init_multi_module  # noqa: E402


def sd_hash(module):
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().float().numpy().tobytes())
    return h.hexdigest()[:16]


def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    mk, ex = cfg["model_kwargs"], cfg["experiment"]
    group = mk["predictor_group"]
    common = dict(
        module_name=mk["module_name"],
        device="cpu",
        frames_per_clip=ex["data"]["frames_per_clip"],
        frames_per_second=ex["data"]["frames_per_second"],
        resolution=ex["data"]["resolution"],
        checkpoint=mk["checkpoint"],
        model_kwargs=mk["pretrain_kwargs"],
        wrapper_kwargs=mk["wrapper_kwargs"],
    )

    print(f"== group 을 짓는다 ({len(group)} 팔, encoder 공유) ...", flush=True)
    gm = init_multi_module(predictor_group=group, **common)
    members = list(gm.models)

    torch.manual_seed(0)
    T = ex["data"]["frames_per_clip"]
    R = ex["data"]["resolution"]
    x = torch.randn(1, 3, T, R, R)
    at = torch.tensor([1.0])

    print("== 1) memo 등가 ...", flush=True)
    with torch.no_grad():
        y_shared = gm(x, at)
        for m in members:
            m.share = None                     # memo 를 끄면 모델마다 encoder 를 다시 돈다
        y_plain = [m(x, at) for m in members]
        for m in members:
            m.share = gm._share

    ok = True
    for name, a, b in zip(gm.names, y_shared, y_plain):
        same = torch.equal(a, b)
        d = float((a - b).abs().max())
        print(f"   {name:<18} shape {tuple(a.shape)}  bit-identical={same}  max|diff|={d:.3e}")
        ok &= same

    print("== 2) 가중치 오염 없음 (팔마다 혼자 지어 비교) ...", flush=True)
    for g, m in zip(group, members):
        mkk = dict(mk["pretrain_kwargs"])
        if g.get("predictor_checkpoint"):
            mkk["predictor_checkpoint"] = g["predictor_checkpoint"]
        else:
            mkk.pop("predictor_checkpoint", None)
        solo = init_module(**{**common, "model_kwargs": mkk})
        hg, hs = sd_hash(m.predictor), sd_hash(solo.predictor)
        kind_g = getattr(m.predictor, "ek_kind", "?")
        kind_s = getattr(solo.predictor, "ek_kind", "?")
        same = (hg == hs) and (kind_g == kind_s)
        print(f"   {g['name']:<18} group {hg} / solo {hs}  kind {kind_g}/{kind_s}  일치={same}")
        ok &= same
        del solo

    print("\n==> " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
