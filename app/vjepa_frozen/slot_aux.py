"""slot 메모리 보조 손실 · 모니터 (2026-11-07, src/models/slot_memory.py).

slot_match (사용자 노트의 "visible mask · ID 로 약한 Hungarian matching"):
  클립마다 slot K 개 ↔ 물체 M 개를 **클립 전체에서 한 번** 짝짓는다 (블록마다 따로 짝짓지 않는다 → slot 정체성이 클립 내내 고정).
  비용 = 그 물체가 보이는 블록들에서 (1 − soft dice(slot attention 맵, 보이는 마스크)) 평균. 보이지 않는 블록은 손실 · 비용에 안 들어간다
  (가려진 동안 무엇을 할지는 감독하지 않는다 — 가려지기 전과 다시 보인 뒤 같은 slot 이 물체를 잡도록만 요구한다).
  가려진 부분까지 포함한 마스크 (amodal) 는 학습에 쓰지 않는다.
  유효 블록: 가능 클립은 TF 가 본 블록 전부 (0..nb-2), 불가능 이음 클립은 문맥 블록만.

모니터 (가중치 0 이어도 계산): 짝지은 target 물체 slot 의 gate — 보이는 블록 / 가려진 블록 (가림 동안 gate 가 닫히는가).
"""
from __future__ import annotations

import numpy as np
import torch


def slot_match(att, gate, masks, roles, valid_blk):
    """att (B, n, K, S) slot attention (토큰마다 slot softmax), gate (B, n, K), masks (B, n, M, S) 0..1 보이는 비율,
    roles (B, M) 1 target · 2 distractor · 0 없음, valid_blk (B, n) bool.
    return (손실 스칼라 (grad), 통계 dict)."""
    from scipy.optimize import linear_sum_assignment
    att, masks = att.float(), masks.float()
    B, n, K, S = att.shape
    present = roles > 0                                                          # (B, M)
    vis = (masks.sum(-1) >= 0.5) & valid_blk[..., None] & present[:, None, :]    # (B, n, M) 보이는 블록
    inter = torch.einsum("bnks,bnms->bnkm", att, masks)
    den = att.sum(-1)[..., :, None] + masks.sum(-1)[..., None, :]
    lossm = (1.0 - 2.0 * inter / (den + 1e-6)) * vis[:, :, None, :].float()     # (B, n, K, M)
    cnt = vis.sum(1)                                                             # (B, M)
    cost = lossm.sum(1) / cnt.clamp_min(1)[:, None, :].float()                   # (B, K, M)
    cost_np = cost.detach().float().cpu().numpy()
    terms, g_vis, g_hid, tgt_dice = [], [], [], []
    for b in range(B):
        ms = [m for m in range(cnt.size(1)) if int(cnt[b, m]) > 0]
        if not ms:
            continue
        r, c = linear_sum_assignment(cost_np[b][:, ms])
        for k, j in zip(r, c):
            m = ms[j]
            terms.append(cost[b, k, m])
            if int(roles[b, m]) == 1:                                            # target 물체의 slot gate: 보일 때 / 가려졌을 때 (유효 블록 안)
                v = vis[b, :, m]
                hid = valid_blk[b] & ~v
                hid[0] = False                                                   # 첫 블록은 gate 가 의미 없다 (초기값에서 시작)
                vv = v.clone(); vv[0] = False
                if vv.any():
                    g_vis.append(float(gate[b, vv, k].float().mean()))
                if hid.any():
                    g_hid.append(float(gate[b, hid, k].float().mean()))
                tgt_dice.append(1.0 - float(cost[b, k, m]))
    loss = torch.stack(terms).mean() if terms else att.sum() * 0.0
    st = {"match": float(loss.detach()), "n_obj": float(len(terms)),
          "g_all": float(gate.detach().float().mean()),
          "g_tvis": float(np.mean(g_vis)) if g_vis else float("nan"),
          "g_thid": float(np.mean(g_hid)) if g_hid else float("nan"),
          "t_dice": float(np.mean(tgt_dice)) if tgt_dice else float("nan")}
    return loss, st
