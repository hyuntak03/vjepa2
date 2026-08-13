"""Probe training over cached TOKEN sequences.

The head itself comes from the repo's shared factory, `evals.analysis.probes.build_probe`,
so `linear` and `attentive` here are the exact same modules the rest of the analysis
harness uses:
    {type: linear,    pooling: mean|max|meanmax, pre_norm: bool}
    {type: attentive, num_heads: 16, num_probe_blocks: 1}      -> AttentiveClassifier

Both consume (B, N, D) tokens, which is why the cache stores token sequences rather than
pooled vectors -- an attentive pooler needs the set, not its mean.

Training is deliberately plain: AdamW, constant lr/wd, shuffled minibatches. The whole
point of `apply_to` is that a head fitted on ONE source is applied verbatim to another,
so nothing here may depend on the evaluation source.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from evals.analysis.probes import build_probe, probe_name  # noqa: F401  (re-exported)


@dataclass
class FittedProbe:
    head: torch.nn.Module
    spec: dict
    train_acc: float
    train_loss: float
    n_train: int
    epochs: int
    device: str


def _batch(X, idx, device) -> torch.Tensor:
    """Gather rows `idx` from a (N, T, D) array/memmap -> (B, T, D) float32 on device."""
    if isinstance(X, np.ndarray) or isinstance(X, np.memmap):
        xb = torch.from_numpy(np.ascontiguousarray(X[idx]))
    else:
        xb = X[idx]
    return xb.to(device=device, dtype=torch.float32, non_blocking=True)


def train_probe(
    X,                               # (N, T, D) tokens: np.memmap / ndarray / tensor
    y: torch.Tensor,                 # (N,) long
    spec: dict,
    num_classes: int,
    *,
    num_epochs: int = 200,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    seed: int = 0,
    device: str = "cuda",
    verbose: bool = False,
    log: Optional[Callable[[int, int, float, float], None]] = None,
) -> FittedProbe:
    """log(epoch, num_epochs, loss, acc) 를 주면 진행 상황을 그쪽으로 흘린다.

    안 주면 무음이다. 4096-token head 는 한 번 학습에 15분씩 걸려서, 무음이면
    멈춘 건지 도는 건지 구분이 안 된다. 호출부에서 rank 를 붙여 로거로 보낼 것.
    """
    torch.manual_seed(seed)
    n, _, embed_dim = X.shape
    if n == 0:
        raise ValueError("train_probe: 학습 샘플이 0개다 (fit_groups/split 이 빈 집합을 만든다)")
    head = build_probe(spec, embed_dim=embed_dim, num_classes=num_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    y = y.to(device)

    g = torch.Generator().manual_seed(seed)
    last_loss, last_acc = float("nan"), float("nan")
    head.train()
    for ep in range(num_epochs):
        perm = torch.randperm(n, generator=g)
        tot_loss, tot_correct = 0.0, 0
        for s in range(0, n, batch_size):
            b = perm[s : s + batch_size]
            logits = head(_batch(X, b.numpy(), device))
            yb = y[b.to(device)]
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot_loss += float(loss.detach()) * len(b)
            tot_correct += int((logits.argmax(-1) == yb).sum())
        last_loss, last_acc = tot_loss / n, tot_correct / n
        if (ep + 1) % max(1, num_epochs // 10) == 0 or ep + 1 == num_epochs:
            if verbose:
                print(f"      ep {ep+1:4d}/{num_epochs}  loss={last_loss:.4f}  acc={last_acc:.4f}")
            if log is not None:
                log(ep + 1, num_epochs, last_loss, last_acc)

    head.eval()
    return FittedProbe(head=head, spec=dict(spec), train_acc=last_acc, train_loss=last_loss,
                       n_train=n, epochs=num_epochs, device=device)


@torch.no_grad()
def predict(probe: FittedProbe, X, batch_size: int = 64, rows: Optional[np.ndarray] = None
            ) -> torch.Tensor:
    """-> (len(rows),) long predicted class indices (cpu). rows=None 이면 전체.

    `rows` 를 주면 그 행만 forward 한다. 평가에 val 절반만 쓰면서 전체를 밀면
    그대로 2배 낭비다 (4096-token 표현에서는 이게 제일 큰 낭비다).
    """
    idx_all = np.arange(X.shape[0]) if rows is None else np.asarray(rows)
    out = []
    for s in range(0, len(idx_all), batch_size):
        out.append(probe.head(_batch(X, idx_all[s : s + batch_size], probe.device)).argmax(-1).cpu())
    return torch.cat(out) if out else torch.zeros(0, dtype=torch.long)


def accuracy(pred: torch.Tensor, y: torch.Tensor) -> float:
    if len(y) == 0:
        return float("nan")
    return float((pred == y).float().mean())


def balanced_accuracy(pred: torch.Tensor, y: torch.Tensor) -> float:
    """Mean per-class recall (macro average), over the classes actually present.

    Plain accuracy is not comparable across datasets whose label distributions differ:
    the 24px set has 8 shapes fairly evenly spread, the 48px set has 7 (no `sphere` before
    the occlusion) with `ring` at 3.6% and `ell` at 22%, so always guessing `ell` already
    scores 0.221 there vs 0.171 on the 24px set. Macro recall weights every class equally,
    so a 7-class and an 8-class run can be put on the same axis.
    """
    if len(y) == 0:
        return float("nan")
    recalls = []
    for c in torch.unique(y):
        m = y == c
        recalls.append(float((pred[m] == c).float().mean()))
    return float(np.mean(recalls))


def confusion(pred: torch.Tensor, y: torch.Tensor, num_classes: int) -> np.ndarray:
    m = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y.tolist(), pred.tolist()):
        m[t, p] += 1
    return m
