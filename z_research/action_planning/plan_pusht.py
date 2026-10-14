"""Push-T planning 평가 — DINO-WM 프로토콜 (CEM + MPC, 시뮬레이터 성공 판정) 에 **우리 action 조건부 AR predictor** 를 꽂는다.

    python z_research/action_planning/plan_pusht.py --run pusht_ar_postft --ckpt latest.pt            # 기본: n_evals 50, goal_H 5, dset goal
    python z_research/action_planning/plan_pusht.py --run pusht_ar_postft --actions gt|zero|random   # 대조 (planner 없이 action 재생)
    ... --goal_source random_state --n_evals 10 --samples 100 --opt_steps 10                          # 빠른 점검

DINO-WM (`benchmarks/dino_wm`) 의 CEMPlanner / MPCPlanner / PlanEvaluator / PushTWrapper / SerialVectorEnv 를 그대로 쓰고,
우리 쪽은 (1) world model 래퍼 (`encode_obs`, `rollout`), (2) 전처리 (224→256, ImageNet 정규화, action 정규화), (3) goal 추출만 댄다.
기본값 = DINO-WM `conf/plan_pusht.yaml`: horizon 5 · samples 300 · topk 30 · opt_steps 30 · var_scale 1 · MPC n_taken 5 · seed 99 · n_evals 50.
⚠️ DINO-WM 과 다른 점: objective 는 visual latent MSE 만 (alpha 0; 우리는 proprio 예측이 없다). max_iter 는 기본 5 (DINO-WM 은 null=무한).
결과: <out>/summary.json (success_rate, per-eval success·state_dist, 설정) + logs.json (planner 로그).
"""
from __future__ import annotations

import argparse, json, os, random, sys, time, types
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "z_research/scripts/harness"))
from paths import expand  # noqa: E402

DINO = Path(expand("${BENCH_CODE_ROOT}/dino_wm"))
sys.path.insert(0, str(DINO))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
_pkg = types.ModuleType("env"); _pkg.__path__ = [str(DINO / "env")]; sys.modules["env"] = _pkg   # env/__init__ (pointmaze→mujoco) 우회
from env.pusht.pusht_wrapper import PushTWrapper            # noqa: E402
from env.serial_vector_env import SerialVectorEnv            # noqa: E402
from planning.cem import CEMPlanner                          # noqa: E402
from utils import move_to_device                             # noqa: E402
from planning.mpc import MPCPlanner                          # noqa: E402
from planning.evaluator import PlanEvaluator                 # noqa: E402
from planning.objectives import create_objective_fn          # noqa: E402

from app.vjepa_frozen import utils as U, ar as AR            # noqa: E402
from app.vjepa_frozen.data import normalize_clips            # noqa: E402
from app.vjepa_frozen.pusht_data import PushTDataset, WallDataset, ACTION_TYPES   # noqa: E402

ACTION_SCALE = 100.0    # env.step 이 action × 100 을 px 로 쓴다 (DINO-WM pusht_dset 도 /100 해 둔다)


# ----------------------------------------------------------------------------- world model wrapper
class VJEPAARWorldModel(torch.nn.Module):
    """DINO-WM planner 가 부르는 두 메서드만 댄다. 관측 1 장 → 프레임 복제 tubelet → LN(h) (학습과 같다)."""

    def __init__(self, tgt_enc, predictor, resolution: int, autocast_dtype, frameskip: int, action_dim: int):
        super().__init__()
        self.enc, self.pred = tgt_enc, predictor
        self.res, self.ac_dtype, self.fs, self.action_dim = resolution, autocast_dtype, frameskip, action_dim
        self.decoder = None
        self.chunk = 0
        self.distance = "l1"               # CEM 점수의 latent 거리 (2026-10-12 사용자 결정: L1 — 학습 손실·surprise 채점과 같다). "mse" = DINO-WM 원판
        self.full_return_max = 64          # 이보다 큰 배치 (CEM 샘플) 는 [처음 ; 마지막] 만 돌려준다 (objective mode=last 전제)
        self.S = (resolution // 16) ** 2
        self.n_enc_calls = 0

    def _ac(self):
        return torch.autocast("cuda", dtype=self.ac_dtype) if self.ac_dtype else torch.autocast("cuda", enabled=False)

    def _encode_visual(self, visual: torch.Tensor) -> torch.Tensor:
        """visual (b, t, 3, r, r) uint8 → (b, t, S, D) LN(h). 배치가 전부 같은 관측이면 (CEM 샘플 반복) 한 번만 인코딩."""
        b, t = visual.shape[:2]
        same = b > 1 and bool(torch.equal(visual[1:], visual[:1].expand(b - 1, *visual.shape[1:])))
        v = visual[:1] if same else visual
        clips = v.unsqueeze(3).repeat(1, 1, 1, 2, 1, 1)                       # (b', t, 3, 2, r, r)
        clips = clips.permute(0, 2, 1, 3, 4, 5).reshape(v.size(0), 3, 2 * t, self.res, self.res)
        with torch.no_grad(), self._ac():
            x = normalize_clips(clips).to(next(self.enc.parameters()).dtype)
            h = AR.encode_blocks(self.enc, x, 2)                                 # (b', t*S, D)
        self.n_enc_calls += 1
        h = h.view(v.size(0), t, self.S, -1).float()
        return h.expand(b, -1, -1, -1) if same else h

    @torch.no_grad()
    def cem_losses(self, visual_goals, z_goal, act, chunk=600):
        """배치 CEM 용 (2026-10-10). visual_goals (G, n, 3, r, r) uint8 = 목표마다 현재 관측, z_goal (G, S, D) 목표 latent,
        act (G, N, H, A) 정규화 action 샘플 → loss (G, N) = 거리(마지막 예측 프레임, 목표) — self.distance (l1 기본 | mse = objective mode=last, alpha 0).
        관측은 목표마다 한 번만 인코딩하고, 샘플은 chunk 개씩 굴려 마지막 프레임만 남긴다 (큰 배치를 메모리 없이)."""
        G, N, H, A = act.shape
        h = self._encode_visual(visual_goals)                                     # (G, n, S, D)
        n = h.size(1)
        n_steps = H - n + 1
        roll = self.pred.rollout_kv if getattr(self.pred, "kv_rollout", False) else self.pred.rollout
        hf = h.reshape(G, n * self.S, -1)
        af = act.reshape(G * N, H, A).float()
        gi = torch.arange(G, device=act.device).repeat_interleave(N)              # 샘플 → 목표 번호
        out = torch.empty(G * N, device=act.device)
        with self._ac():
            for i in range(0, G * N, chunk):
                g = gi[i: i + chunk]
                idx = AR.block_indices(g.numel(), n, self.S, act.device)
                pc = roll(hf[g], idx, n_steps, actions=af[i: i + chunk])[:, -self.S:].float()   # 마지막 블록
                dz = pc - z_goal[g].float()
                out[i: i + chunk] = (dz.abs() if self.distance == "l1" else dz.pow(2)).mean(dim=(1, 2))
        return out.view(G, N)

    def encode_obs(self, obs):
        return {"visual": self._encode_visual(obs["visual"]), "proprio": obs["proprio"].float()}

    def rollout(self, obs_0, act):
        """obs_0 visual (b, n, 3, r, r) uint8 · act (b, n+t, A) 정규화 → z_obses visual (b, n+t+1, S, D). DINO-WM 규약."""
        h = self._encode_visual(obs_0["visual"])                                  # (b, n, S, D)
        b, n = h.shape[:2]
        n_steps = act.size(1) - n + 1
        roll = self.pred.rollout_kv if getattr(self.pred, "kv_rollout", False) else self.pred.rollout   # KV 캐시판 (값 동일, 빠름)
        ch = self.chunk if self.chunk > 0 else b                                  # 샘플을 나눠 굴린다 (메모리만 줄고 값은 같다)
        last_only = b > self.full_return_max
        # CEM 샘플 배치 (예: 300): objective 는 mode=last 라 **마지막 프레임만** 읽는다 (planning/objectives.objective_fn_last).
        # 전 궤적을 fp32 로 모으면 300×6×256×1280×4 B = 2.4 GB 라, 조각마다 마지막 프레임만 남겨 [처음 ; 마지막] 을 돌려준다 — loss 는 같다.
        # evaluator 배치 (n_evals ≤ full_return_max) 는 성공 시점 인덱싱이 필요해 전 궤적을 돌려준다.
        outs = []
        with torch.no_grad(), self._ac():
            for i in range(0, b, ch):
                hb = h[i: i + ch]
                idx = AR.block_indices(hb.size(0), n, self.S, h.device)
                pc = roll(hb.reshape(hb.size(0), n * self.S, -1), idx, n_steps, actions=act[i: i + ch].float())
                pc = pc.view(pc.size(0), n_steps, self.S, -1)
                outs.append((pc[:, -1:] if last_only else pc).float())
                del pc
        p = torch.cat(outs, dim=0)
        z_vis = torch.cat([h[:, :1], p], dim=1) if last_only else torch.cat([h, p], dim=1)
        pd = obs_0["proprio"].shape[-1]
        return {"visual": z_vis, "proprio": torch.zeros(b, z_vis.size(1), pd, device=z_vis.device)}, None


class Preproc:
    """DINO-WM Preprocessor 와 같은 메서드. visual: (b, t, H, W, 3) uint8 numpy → (b, t, 3, r, r) uint8 torch (정규화는 모델 안에서).
    action: env 단위 (px/100) ↔ 정규화 ((px − mean)/std, train 통계)."""

    def __init__(self, stats: dict, resolution: int, frameskip: int, action_scale: float = ACTION_SCALE):
        self.mean = torch.tensor(stats["mean"]).float(); self.std = torch.tensor(stats["std"]).float()
        self.res, self.fs, self.scale = resolution, frameskip, float(action_scale)

    def _per_dim(self, a):   # (..., f*2) ↔ (..., f, 2)
        return a.reshape(*a.shape[:-1], a.shape[-1] // 2, 2)          # 빈 텐서 (MPC 성공 마스크가 0 개) 에서도 모양이 정해진다

    def normalize_actions(self, a):            # env 단위 → 정규화
        a = torch.as_tensor(a).float(); sh = a.shape
        x = (self._per_dim(a) * self.scale - self.mean) / self.std
        return x.reshape(sh)

    def denormalize_actions(self, a):          # 정규화 → env 단위
        a = torch.as_tensor(a).float(); sh = a.shape
        x = (self._per_dim(a) * self.std + self.mean) / self.scale
        return x.reshape(sh)

    def normalize_proprios(self, p):
        return torch.as_tensor(p).float()

    def transform_obs(self, obs):
        v = torch.as_tensor(np.asarray(obs["visual"]))                            # (b, t, H, W, 3) uint8
        v = v.permute(0, 1, 4, 2, 3).float()
        b, t = v.shape[:2]
        if v.shape[-1] != self.res:
            v = torch.nn.functional.interpolate(v.flatten(0, 1), size=(self.res, self.res), mode="bilinear",
                                                align_corners=False).view(b, t, 3, self.res, self.res)
        return {"visual": v.round().clamp(0, 255).to(torch.uint8), "proprio": self.normalize_proprios(obs["proprio"])}


class DummyRun:
    def log(self, *a, **k): pass


class BatchedCEMPlanner(CEMPlanner):
    """DINO-WM CEMPlanner 와 같은 알고리즘 (목표마다 num_samples 샘플, topk 평균·표준편차 갱신, opt_steps 회, 첫 샘플 = mu)
    을 **목표 전체를 한 번에** 돈다 (2026-10-10). 원판은 목표를 하나씩 순서대로 돌아 512 목표면 팔당 13 h 걸렸다.
    objective 는 visual MSE 마지막 프레임만 (mode=last, alpha 0 전제 — main 이 검사). 중간 evaluator 호출 (원판 i=0) 은 생략한다."""

    def __init__(self, *args, wm_chunk=600, **kw):
        super().__init__(*args, **kw)
        self.wm_chunk = wm_chunk
        self.mpc = None            # main 이 MPCPlanner 를 걸어 준다 → 이미 성공한 목표는 world model 을 안 돌린다 (skip_solved)
        self.skip_solved = True

    def plan(self, obs_0, obs_g, actions=None):
        trans_obs_0 = move_to_device(self.preprocessor.transform_obs(obs_0), self.device)
        trans_obs_g = move_to_device(self.preprocessor.transform_obs(obs_g), self.device)
        z_g = self.wm.encode_obs(trans_obs_g)["visual"][:, -1]                     # (G, S, D)
        mu, sigma = self.init_mu_sigma(obs_0, actions)
        mu, sigma = mu.to(self.device), sigma.to(self.device)
        G = mu.shape[0]
        # skip_solved (2026-10-12): DINO-WM MPC 는 성공한 목표도 계속 planning 하고 그 action 을 0 으로 덮어 버린다 (_apply_success_mask,
        #   n_taken = horizon 이라 다음 회차로 넘어가는 몫도 없다). 그 목표의 world model rollout 은 결과에 안 쓰이므로 건너뛴다.
        #   난수는 여전히 목표 전체로 뽑아 남은 목표의 샘플 흐름이 바뀌지 않는다 (chunk 구성만 달라진다)
        live = None
        if self.skip_solved and self.mpc is not None and self.mpc.is_success is not None and self.mpc.is_success.any():
            live = torch.nonzero(torch.as_tensor(~self.mpc.is_success), as_tuple=True)[0].to(self.device)
        for i in range(self.opt_steps):
            act = torch.randn(G, self.num_samples, self.horizon, self.action_dim, device=self.device) * sigma[:, None] + mu[:, None]
            act[:, 0] = mu
            if live is None:
                loss = self.wm.cem_losses(trans_obs_0["visual"], z_g, act, chunk=self.wm_chunk)   # (G, N)
            else:
                loss = torch.zeros(G, self.num_samples, device=self.device)
                loss[live] = self.wm.cem_losses(trans_obs_0["visual"][live], z_g[live], act[live], chunk=self.wm_chunk)
            top = torch.argsort(loss, dim=1)[:, : self.topk]                          # (G, k)
            ta = torch.gather(act, 1, top[:, :, None, None].expand(-1, -1, self.horizon, self.action_dim))
            mu, sigma = ta.mean(dim=1), ta.std(dim=1)
        return mu, np.full(G, np.inf)


# ----------------------------------------------------------------------------- model loading
def load_wm(run: str, ckpt: str, device):
    rd = ROOT / "z_training/runs" / run
    cfg = yaml.safe_load(open(rd / "config.yaml"))
    MD = cfg["model"]; MD.setdefault("use_sdpa", True)
    n_frames, res = int(cfg["data"]["n_frames"]), int(cfg["data"]["resolution"])
    state = U.load_base_checkpoint(MD["checkpoint"])
    _, tgt = U.build_frozen_encoders(MD, state, device, torch.bfloat16, n_frames)
    MD2 = dict(MD); MD2.pop("init_predictor", None)
    pred = U.build_predictor(MD2, tgt.embed_dim, None, device, n_frames)
    del state
    ck = torch.load(rd / ckpt, map_location="cpu", weights_only=False)
    pred.load_state_dict(ck["predictor"], strict=True); pred.eval()
    spec = next(s for s in cfg["data"]["datasets"] if s.get("type") in ACTION_TYPES)
    fs = int(spec.get("frameskip", 5))
    ad = int((MD.get("predictor") or {}).get("action_dim", 0))
    assert ad == 2 * fs, f"predictor action_dim {ad} != 2*frameskip {2*fs}"
    stats = (WallDataset if spec["type"] == "wall" else PushTDataset).compute_action_stats(spec["root"])
    wm = VJEPAARWorldModel(tgt, pred, res, torch.bfloat16, fs, ad).to(device).eval()
    return wm, cfg, spec, stats, ck.get("epoch")


# ----------------------------------------------------------------------------- envs
def make_env(kind: str, n: int):
    if kind == "pusht":
        return SerialVectorEnv([PushTWrapper(with_velocity=True, with_target=True) for _ in range(n)])
    if kind == "wall":
        from env.wall.wall_env_wrapper import WallEnvWrapper           # DINO-WM env/wall (torch 만 필요, mujoco 불필요)
        return SerialVectorEnv([WallEnvWrapper(device="cpu") for _ in range(n)])
    raise ValueError(kind)


def prepare_targets_wall(a, env, root: str, seeds):
    """DINO-WM plan_wall.yaml: goal_source=random_state. 문 / 벽 위치는 val 궤적의 것 (이 세트는 전부 30 / 32) 으로 맞춘 뒤,
    env 가 seed 로 시작점 (벽 한쪽) 과 목표 (문 건너편) 를 뽑는다."""
    door = torch.load(os.path.join(root, "door_locations.pth"))[:, 0, 0]
    wall = torch.load(os.path.join(root, "wall_locations.pth"))[:, 0, 0]
    assert float(door.min()) == float(door.max()) and float(wall.min()) == float(wall.max()), "문/벽 위치가 episode 마다 다르다 — val 궤적을 골라 써야 한다"
    info = {"fix_door_location": door[0], "fix_wall_location": wall[0]}
    env.update_env([info] * len(seeds))
    init, goal = env.sample_random_init_goal_states(seeds)
    obs_0, _ = env.prepare(seeds, init); obs_g, _ = env.prepare(seeds, goal)
    obs_0 = {k: np.expand_dims(np.asarray(v), 1) for k, v in obs_0.items()}
    obs_g = {k: np.expand_dims(np.asarray(v), 1) for k, v in obs_g.items()}
    return obs_0, obs_g, init, goal, None


# ----------------------------------------------------------------------------- goals
def prepare_targets(a, env, root: str, fs: int, pre: Preproc, seeds):
    """DINO-WM `prepare_targets`: goal_source=dset 이면 val episode 조각 (길이 fs·goal_H+1) 의 처음·끝, 그 사이 GT action 을 env 에서 재생."""
    d = os.path.join(root, "val")
    states = torch.load(os.path.join(d, "states.pth")).float()
    vel_p = os.path.join(d, "velocities.pth")
    if os.path.isfile(vel_p):
        states = torch.cat([states, torch.load(vel_p).float()], dim=-1)             # 7-dim (env with_velocity)
    acts = torch.load(os.path.join(d, "rel_actions.pth")).float()
    import pickle
    L = [int(x) for x in pickle.load(open(os.path.join(d, "seq_lengths.pkl"), "rb"))]
    rng = random.Random(a.seed)
    T = fs * a.goal_H
    if a.goal_source == "random_state":
        init, goal = env.sample_random_init_goal_states(seeds)
        obs_0, s0 = env.prepare(seeds, init); obs_g, sg = env.prepare(seeds, goal)
        obs_0 = {k: v[:, None] for k, v in obs_0.items()}; obs_g = {k: v[:, None] for k, v in obs_g.items()}
        return obs_0, obs_g, init, goal, None
    init_states, gt = [], []
    for _ in range(a.n_evals):
        while True:
            ep = rng.randrange(len(L))
            if L[ep] >= T + 1:
                break
        off = rng.randint(0, L[ep] - T - 1)
        init_states.append(states[ep, off].numpy())
        gt.append(acts[ep, off: off + T])                                             # (T, 2) px
    init_states = np.stack(init_states); gt = torch.stack(gt)                          # (b, T, 2)
    exec_actions = (gt / ACTION_SCALE).numpy()
    obses, sts = env.rollout(seeds, init_states, exec_actions)                          # (b, T+1, ...) 
    obs_0 = {k: v[:, :1] for k, v in obses.items()}; obs_g = {k: v[:, -1:] for k, v in obses.items()}
    wm_actions = pre.normalize_actions(gt / ACTION_SCALE).view(a.n_evals, a.goal_H, fs * 2)   # (b, goal_H, 2fs) 정규화
    return obs_0, obs_g, init_states, sts[:, -1], wm_actions


# ----------------------------------------------------------------------------- gif
def save_gifs(a, sel, e_obses, obs_g, successes, action_len, fs, actions, fps=15):
    """목표마다 GIF: 왼쪽 = 시뮬레이터에서 planned action 을 실행한 실제 화면, 오른쪽 = 목표 이미지. 위에 글자 띠.
    성공이면 성공 시점 + 15 프레임에서 자른다 (그 뒤 action 은 0 이라 정지). planned action (정규화) 은 npy 로."""
    import imageio
    from PIL import Image, ImageDraw
    os.makedirs("gif", exist_ok=True)
    vis = np.asarray(e_obses["visual"])                               # (b, T+1, H, W, 3) uint8
    goal = np.asarray(obs_g["visual"])[:, -1]                         # (b, H, W, 3)
    np.save("gif/planned_actions_normalized.npy", actions.detach().cpu().numpy())
    for j, k in enumerate(sel):
        ok = bool(successes[j]); al = action_len[j]
        T = vis.shape[1]
        end = T if not (ok and np.isfinite(al)) else min(T, int(al) * fs + 1 + 15)
        frames = []
        for t in range(end):
            fr = np.concatenate([vis[j, t], np.full((vis.shape[2], 6, 3), 255, np.uint8), goal[j]], axis=1)
            im = Image.new("RGB", (fr.shape[1], fr.shape[0] + 34), (255, 255, 255)); im.paste(Image.fromarray(fr), (0, 34))
            d = ImageDraw.Draw(im)
            tag = (f"SUCCESS at step {int(al) * fs}" if ok and np.isfinite(al) else ("SUCCESS" if ok else "FAIL"))
            d.text((4, 2), f"{a.run}  goal #{k}  step {t}/{T - 1}", fill=(0, 0, 0))
            d.text((4, 17), f"{tag}   (left: simulator, right: goal)", fill=(0, 120, 0) if ok else (200, 0, 0))
            frames.append(np.asarray(im))
        name = f"gif/{a.run}_goal{k:02d}_{'SUCCESS' if ok else 'FAIL'}.gif"
        imageio.mimsave(name, frames, duration=1.0 / fps, loop=0)
        print(f"saved {name} ({len(frames)} frames)")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--ckpt", default="latest.pt")
    ap.add_argument("--n_evals", type=int, default=50); ap.add_argument("--goal_H", type=int, default=5)
    ap.add_argument("--goal_source", default="dset", choices=["dset", "random_state"])
    ap.add_argument("--samples", type=int, default=300); ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--opt_steps", type=int, default=None, help="기본 = DINO-WM 환경별 값 (pusht 30, wall 10)"); ap.add_argument("--var_scale", type=float, default=1.0)
    ap.add_argument("--n_taken", type=int, default=5); ap.add_argument("--max_iter", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.0); ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--actions", default="plan", choices=["plan", "gt", "zero", "random"], help="plan=CEM+MPC, 그 외 = 대조 (planner 없이 재생)")
    ap.add_argument("--out", default=None); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--wm_chunk", type=int, default=0, help="world model rollout 을 샘플 몇 개씩 나눠 돌릴지 (0 = 한 번에). 값은 같고 메모리만 준다")
    ap.add_argument("--mem_frac", type=float, default=0.0, help="이 프로세스의 GPU 메모리 상한 (비율). 학습과 GPU 를 같이 쓸 때")
    ap.add_argument("--only", default=None, help="쉼표로 목표 번호 (예: 0,4,3). 전체 n_evals 목표 집합을 만든 뒤 이 번호만 planning")
    ap.add_argument("--batched", action="store_true", help="목표 전체를 한 배치로 도는 CEM (BatchedCEMPlanner). 결과 분포는 원판과 같고 훨씬 빠르다")
    ap.add_argument("--distance", default="l1", choices=["l1", "mse"],
                    help="CEM 점수 = 마지막 예측 latent 와 목표 latent 의 거리. l1 (기본, 2026-10-12~) | mse (DINO-WM 원판, 그 전 결과는 전부 mse)")
    ap.add_argument("--no_skip_solved", action="store_true", help="--batched 에서 이미 성공한 목표도 world model 을 돌린다 (원판 그대로; 결과는 같고 느리다)")
    ap.add_argument("--gif", action="store_true", help="목표마다 시뮬레이터 궤적 GIF (왼쪽 실제, 오른쪽 목표) 와 planned action 을 저장")
    ap.add_argument("--shard", default=None, help="i/N: 같은 n_evals 목표 집합을 만든 뒤 i 번째 조각만 planning (GPU 병렬). merge_shards.py 로 합친다")
    a = ap.parse_args()
    # CPU 스레드 (2026-10-12): Wall env 는 torch CPU 로 그린다. 조각 8 개가 각자 코어 전부 (128) 를 잡으면 부하 300+ 로
    #   MPC 회차가 4 분 → 수 시간이 됐다 (CLAUDE.md §7-1 과다구독). OMP_NUM_THREADS 를 안 주면 코어 / 16 (최소 4) 으로 묶는다
    if not os.environ.get("OMP_NUM_THREADS"):
        torch.set_num_threads(max(4, (os.cpu_count() or 64) // 16))
    device = torch.device(a.device)
    if a.mem_frac > 0:
        torch.cuda.set_per_process_memory_fraction(a.mem_frac, device)
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    # 환경 종류는 run config 의 데이터셋 type 으로 정한다 → DINO-WM 환경별 기본값 (tag 를 만들기 전에)
    _rc = yaml.safe_load(open(ROOT / "z_training/runs" / a.run / "config.yaml"))
    env_kind = next(s_["type"] for s_ in _rc["data"]["datasets"] if s_.get("type") in ACTION_TYPES)
    if a.opt_steps is None:
        a.opt_steps = {"pusht": 30, "wall": 10}[env_kind]
    if env_kind == "wall" and a.goal_source == "dset":
        a.goal_source = "random_state"                            # DINO-WM plan_wall.yaml 기본
    tag = f"{a.run}_{Path(a.ckpt).stem}_gH{a.goal_H}_{a.goal_source}_{a.actions}" + (f"_s{a.samples}o{a.opt_steps}" if a.actions == "plan" else "") \
          + (f"_n{a.n_evals}" if a.n_evals != 50 else "") + ("_batched" if a.batched else "") + ("_l1" if a.distance == "l1" else "") + (f"_seed{a.seed}" if a.seed != 99 else "") \
          + (f"_shard{a.shard.replace('/', 'of')}" if a.shard else "") + (f"_only{a.only.replace(',', '-')}" if a.only else "")
    out = Path(a.out or (ROOT / "z_research/action_planning/exp_results" / tag)); out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)                                                                     # planner 가 logs.json 을 cwd 에 쓴다

    wm, cfg, spec, stats, epoch = load_wm(a.run, a.ckpt, device)
    assert spec["type"] == env_kind
    fs = wm.fs
    wm.chunk = a.wm_chunk
    wm.distance = a.distance
    pre = Preproc(stats, wm.res, fs, action_scale=(WallDataset if env_kind == "wall" else PushTDataset).ACTION_SCALE)
    seeds = [a.seed * n + 1 for n in range(a.n_evals)]
    env = make_env(env_kind, a.n_evals)
    if env_kind == "wall":
        obs_0, obs_g, state_0, state_g, gt_actions = prepare_targets_wall(a, env, spec["root"], seeds)
    else:
        obs_0, obs_g, state_0, state_g, gt_actions = prepare_targets(a, env, spec["root"], fs, pre, seeds)
    sel = list(range(a.n_evals))
    if a.shard or a.only:                                         # 목표 집합은 전체와 같고, 그 일부만 planning
        if a.only:
            sel = [int(x) for x in a.only.split(",")]
        else:
            si, sn = [int(x) for x in a.shard.split("/")]
            sel = [k for k in range(a.n_evals) if k % sn == si]
        pick = lambda d: {k: v[sel] for k, v in d.items()}
        obs_0, obs_g = pick(obs_0), pick(obs_g)
        state_0, state_g = np.asarray(state_0)[sel], np.asarray(state_g)[sel]
        gt_actions = None if gt_actions is None else gt_actions[sel]
        seeds = [seeds[k] for k in sel]
        env = make_env(env_kind, len(sel))
        if env_kind == "wall":
            door = torch.load(os.path.join(spec["root"], "door_locations.pth"))[0, 0, 0]
            wallx = torch.load(os.path.join(spec["root"], "wall_locations.pth"))[0, 0, 0]
            env.update_env([{"fix_door_location": door, "fix_wall_location": wallx}] * len(sel))
        a.n_evals_total, a.n_evals = a.n_evals, len(sel)
    evaluator = PlanEvaluator(obs_0=obs_0, obs_g=obs_g, state_0=state_0, state_g=state_g, env=env, wm=wm,
                              frameskip=fs, seed=seeds, preprocessor=pre, n_plot_samples=0)
    objective = create_objective_fn(alpha=a.alpha, base=2, mode="last")   # ⚠️ wm.full_return_max 는 mode=last 전제다
    if a.distance == "l1":                                        # DINO-WM objective_fn_last 와 같고 metric 만 L1 (원판 경로 · evaluator 지표용)
        def objective(z_pred, z_tgt, _al=a.alpha):
            lv = (z_pred["visual"][:, -1:] - z_tgt["visual"]).abs().mean(dim=tuple(range(1, z_pred["visual"].ndim)))
            lp = (z_pred["proprio"][:, -1:] - z_tgt["proprio"]).abs().mean(dim=tuple(range(1, z_pred["proprio"].ndim)))
            return lv + _al * lp
    wm.full_return_max = max(64, a.n_evals)
    action_dim = 2 * fs
    t0 = time.time()
    if a.actions == "plan":
        sub = dict(target="planning.cem.CEMPlanner", horizon=a.goal_H, topk=a.topk, num_samples=a.samples,
                   var_scale=a.var_scale, opt_steps=a.opt_steps, eval_every=10**9)       # 중간 평가 끄기 (MPC 가 평가한다)
        if a.batched:
            if a.alpha != 0:
                raise SystemExit("--batched 는 objective alpha 0 (visual 만) 전제")
            sub = dict(sub, target="__main__.BatchedCEMPlanner" if __name__ == "__main__" else "plan_pusht.BatchedCEMPlanner",
                       wm_chunk=a.wm_chunk if a.wm_chunk > 0 else 600)              # 샘플 600 개 ≈ KV 17 GB (6 블록)
        planner = MPCPlanner(max_iter=a.max_iter, n_taken_actions=a.n_taken, sub_planner=sub, wm=wm, env=env,
                             action_dim=action_dim, objective_fn=objective, preprocessor=pre, evaluator=evaluator,
                             wandb_run=DummyRun(), log_filename="logs.json")
        planner.sub_planner.horizon = a.goal_H; planner.n_taken_actions = a.goal_H
        if a.batched:
            # skip_solved 는 n_taken = horizon 전제 (성공 목표의 남은 plan 이 다음 회차로 안 넘어간다) — 바로 윗줄이 그렇게 고정한다
            planner.sub_planner.mpc = planner; planner.sub_planner.skip_solved = not a.no_skip_solved
        actions, action_len = planner.plan(obs_0=obs_0, obs_g=obs_g)
    else:
        if a.actions == "gt":
            if gt_actions is None:
                raise SystemExit("--actions gt 는 goal_source=dset 에서만")
            actions = gt_actions.to(device)
        elif a.actions == "zero":
            actions = pre.normalize_actions(torch.zeros(a.n_evals, a.goal_H, action_dim)).to(device)
        else:
            actions = torch.randn(a.n_evals, a.goal_H, action_dim, device=device)
        action_len = np.full(a.n_evals, np.inf)
    logs, successes, e_obses, e_states = evaluator.eval_actions(actions.detach(), action_len, save_video=False, filename="final")
    res = {k: (float(v) if isinstance(v, (np.floating, np.integer, float, int)) else v) for k, v in logs.items()}
    ev = env.eval_state(state_g, evaluator._get_traj_last(e_states, np.asarray(action_len) * fs + 1)[:, 0])
    # 실패 원인 분해 (2026-10-10): 성공 판정 = ‖[agent xy, block xy] 차이‖ < 20 px ∧ 각도 < 20°. 어느 항이 깨졌는지 따로 남긴다
    fin = np.asarray(evaluator._get_traj_last(e_states, np.asarray(action_len) * fs + 1)[:, 0]); gs = np.asarray(state_g)
    err_agent = np.linalg.norm(fin[:, :2] - gs[:, :2], axis=1)
    if env_kind == "pusht":                                       # [agent xy, block xy, angle, (vel)]
        err_block = np.linalg.norm(fin[:, 2:4] - gs[:, 2:4], axis=1)
        dang = np.abs(fin[:, 4] - gs[:, 4]); err_angle = np.degrees(np.minimum(dang, 2 * np.pi - dang))
    else:                                                         # wall: 점 위치 (xy) 뿐
        err_block = np.full(len(err_agent), np.nan); err_angle = np.full(len(err_agent), np.nan)
    summary = {"run": a.run, "ckpt": a.ckpt, "epoch": epoch, "args": vars(a), "eval_index": sel, "success_rate": float(np.mean(successes.astype(float))),
               "err_agent_px": err_agent.round(1).tolist(), "err_block_px": err_block.round(1).tolist(), "err_angle_deg": err_angle.round(1).tolist(),
               "successes": successes.astype(int).tolist(), "state_dist": [float(x) for x in ev["state_dist"]],
               "metrics": res, "elapsed_s": round(time.time() - t0, 1), "n_encoder_calls": wm.n_enc_calls,
               "protocol": "DINO-WM plan_pusht.yaml (CEM horizon=goal_H, MPC n_taken=goal_H); objective visual-only %s alpha=%g; max_iter=%d" % (a.distance, a.alpha, a.max_iter)}
    json.dump(summary, open("summary.json", "w"), indent=1)
    # 재생 기록 (2026-10-12, 사용자 지시 "앞으로 할 때는 저장"): 목표별 env action · 초기/목표 상태 · 상태 궤적.
    #   render_plan_gifs.py 가 이것만으로 시뮬레이터를 다시 돌려 GIF 를 만든다 (planning · GPU 불필요). 목표당 수 KB
    b_ = actions.shape[0]
    exec_actions = pre.denormalize_actions(actions.detach().cpu().float().reshape(b_, -1, fs, actions.shape[-1] // fs).flatten(1, 2)).numpy()
    np.savez_compressed("rollout.npz", eval_index=np.asarray(sel), seeds=np.asarray(seeds), state_0=np.asarray(state_0, np.float64),
                        state_g=np.asarray(state_g, np.float64), actions_normalized=actions.detach().cpu().float().numpy(),
                        exec_actions=exec_actions, action_len=np.asarray(action_len, np.float64), successes=successes.astype(np.int8),
                        e_states=np.asarray(e_states, np.float32), frameskip=fs, env_kind=env_kind)
    if a.gif:
        save_gifs(a, sel, e_obses, obs_g, successes, action_len, fs, actions)
    print(f"[{tag}] success_rate {summary['success_rate']:.3f}  ({a.n_evals} evals, {summary['elapsed_s']} s)  -> {out}/summary.json")


if __name__ == "__main__":
    main()
