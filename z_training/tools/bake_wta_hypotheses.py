"""다중 가설 (승자독식) 이어 학습 체크포인트를 가설마다 일반 predictor 체크포인트로 굽는다 (2026-10-25, 사용자: "다중 가설로 IntPhys2 · IntPhys2 late · IntPhys1 · InfLevel 측정").

원본: auto_research/scripts/ar_crps_ft.py --loss wta (기준 AR natural_ar_vw_state latest 를 자연 영상 2000 걸음 이어 학습, 가설 = 384 차원 부호 벡터).
가설 부호는 관측 임베딩 뒤에 더해진다: _ar_embed_obs(x) = predictor_embed(x) + type_embed[0] (+ code_k).
→ type_embed[0] += code_k 로 구우면 구조 · 키가 기준 AR 과 같은 체크포인트가 되고 기존 채점기 (eval_bench · chain_bench · cvpr) 가 그대로 읽는다.
상태 되먹임 걸음 (type_embed[1]) 은 원래도 부호가 안 붙었으므로 그대로 둔다.
--control 은 같은 설정의 L1 이어 학습 (ft_l1, 부호 없음) 을 같은 꼴로 굽는다 (이어 학습 자체의 효과를 가르는 대조).

  /venv/vjepa/bin/python z_training/tools/bake_wta_hypotheses.py            # runs/natural_ar_vw_state_wta3_h{0,1,2} + _ftl1, 그리고 등가 검사
"""
import os, shutil, sys
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO)
from analysis.intphys2.model import _clean_backbone_key  # noqa: E402

BASE = os.path.join(REPO, "z_training/runs/natural_ar_vw_state")
FT = "/workspace/auto_research_cache/ar_ft"
DROP = ("hint_attn.", "dyn_head.", "recon_proj.")


def bake(src: str, name: str, code=None):
    base = torch.load(os.path.join(BASE, "latest.pt"), map_location="cpu", weights_only=False)
    ft = torch.load(os.path.join(FT, src), map_location="cpu", weights_only=False)
    w = {k: v for k, v in _clean_backbone_key(ft["predictor"]).items() if not k.startswith(DROP)}
    if set(w) != set(base["predictor"]):
        raise RuntimeError(f"{src}: 키가 기준 AR 과 다르다 ({len(set(w) ^ set(base['predictor']))} 개)")
    w = {k: v.clone() for k, v in w.items()}
    if code is not None:
        w["type_embed"][0] += code.to(w["type_embed"].dtype)
    out = {k: v for k, v in base.items() if k not in ("opt", "scaler", "predictor")}
    out["predictor"] = w
    out["baked_from"] = {"source": os.path.join(FT, src), "args": ft["args"], "hypothesis_code": None if code is None else code.clone()}
    d = os.path.join(REPO, "z_training/runs", name)
    os.makedirs(os.path.join(d, "eval/logs"), exist_ok=True)
    shutil.copy(os.path.join(BASE, "config.yaml"), os.path.join(d, "config.yaml"))
    torch.save(out, os.path.join(d, "latest.pt"))
    with open(os.path.join(d, "README.txt"), "w") as f:
        f.write(f"학습 run 이 아니다. {src} 를 기준 AR 체크포인트 꼴로 구운 것 (z_training/tools/bake_wta_hypotheses.py, 2026-10-25).\n"
                + ("가설 부호를 type_embed[0] 에 더했다.\n" if code is not None else "부호 없음 (L1 이어 학습 대조).\n"))
    print("saved", d)
    return d


def check(name: str, code):
    """구운 체크포인트 vs 원본 (관측 임베딩에 부호를 더하는 판) 의 rollout 출력이 같은지."""
    sys.argv = [sys.argv[0], "--loss", "l1", "--name", "x"]
    sys.path.insert(0, os.path.join(REPO, "auto_research/scripts"))
    import ar_crps_ft as F
    dev = torch.device("cuda")
    a = F.load_pred(name, "latest.pt").to(dev).eval()
    b = F.load_pred("natural_ar_vw_state", "latest.pt")
    ft = torch.load(os.path.join(FT, "ft_wta3.pt"), map_location="cpu", weights_only=False)
    b.load_state_dict({k: v for k, v in _clean_backbone_key(ft["predictor"]).items() if not k.startswith(DROP)}, strict=True)
    b = b.to(dev).eval()
    o = b._ar_embed_obs
    c = code.to(dev)
    b._ar_embed_obs = lambda x: o(x) + c[None, None, :].to(o(x).dtype)
    torch.manual_seed(0)
    x = torch.nn.functional.layer_norm(torch.randn(2, 4 * 256, 1280, device=dev), (1280,))
    idx = torch.arange(4 * 256, device=dev).unsqueeze(0).expand(2, -1)
    with torch.no_grad():
        pa, pb = a.rollout_kv(x, idx, 4), b.rollout_kv(x, idx, 4)
    print(f"  {name}: 출력 최대 차 {float((pa - pb).abs().max()):.2e}")


if __name__ == "__main__":
    codes = torch.load(os.path.join(FT, "ft_wta3.pt"), map_location="cpu", weights_only=False)["codes"]
    for k in range(codes.shape[0]):
        bake("ft_wta3.pt", f"natural_ar_vw_state_wta3_h{k}", codes[k])
    bake("ft_l1.pt", "natural_ar_vw_state_ftl1", None)
    for k in range(codes.shape[0]):
        check(f"natural_ar_vw_state_wta3_h{k}", codes[k])
