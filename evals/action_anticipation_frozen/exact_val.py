# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""val 을 **정확히 한 번** 돌면서 mean class recall@k 를 잰다 — 보고용 지표.

공식 `eval.validate` 는 `for itr in range(ipe)` 로 돌고, rank 의 iterator 가 먼저 끝나면
`iter(data_loader)` 로 **다시 감는다**. `ipe = num_clips // (world_size * batch_size)` 인데
val 은 비디오 단위로 rank 에 나뉘어 rank 마다 clip 수가 다르므로
- clip 이 적은 rank 는 앞쪽 clip 을 **두 번** 센다
- clip 이 많은 rank 는 뒤쪽 clip 을 **안 센다**
또 `ClassMeanRecall.__call__` 이 매 배치 `all_reduce` 를 부르므로, rank 마다 배치 수가 다르면
그 루프 구조로는 끝까지 돌 수도 없다 (collective 가 어긋나 멈춘다).

여기서는
1. rank 마다 자기 loader 를 **끝까지 한 번** 돈다 (다시 감지 않는다)
2. TP/FN 을 rank 로컬에 쌓고 **루프가 끝난 뒤 all_reduce 한 번**
3. 지표 정의는 `metrics.ClassMeanRecall` 과 같다 —
   top-k 는 logit 에서 고르고 (sigmoid 는 단조라 순서가 같다), 예측 후보를 val 클래스로 제한하지 않는다.
   recall = 그 클래스의 TP/(TP+FN) 을 **val 에 한 번이라도 나온 클래스**에 대해 평균, accuracy = top-k 적중률
4. 실제로 센 clip 수(`n_clips`)와 기대 clip 수(`expected_clips`)를 같이 돌려준다 (decode 실패로 빠진 것 확인용)
"""

import logging

import torch
import torch.distributed as dist

logger = logging.getLogger()

KEYS = ("verb", "noun", "action")


def accumulate(tp, fn, logits, labels, k):
    """배치 하나의 top-k 적중을 클래스별 TP/FN 에 더한다 (in-place)."""
    topk = logits.float().topk(k, dim=1).indices
    hit = (topk == labels[:, None]).any(dim=1)
    n = tp.numel()
    tp += torch.bincount(labels[hit], minlength=n).to(tp.dtype)
    fn += torch.bincount(labels[~hit], minlength=n).to(fn.dtype)


def class_mean_recall(tp, fn, eps=1e-8):
    """`ClassMeanRecall` 과 같은 정의. (recall%, accuracy%, 지지 클래스 수)."""
    support = (tp + fn) > 0
    nch = int(support.sum())
    recall = 100.0 * float((tp / (tp + fn + eps)).sum()) / max(nch, 1)
    total = float((tp + fn).sum())
    acc = 100.0 * float(tp.sum()) / max(total, 1.0)
    return recall, acc, nch


@torch.no_grad()
def validate_exact(
    model,
    classifiers,
    data_loader,
    device,
    use_bfloat16,
    verb_classes,
    noun_classes,
    action_classes,
    k=5,
    expected_clips=None,
):
    heads = [c.module if hasattr(c, "module") else c for c in classifiers]  # DDP 를 벗긴다 (collective 없이)
    for h in heads:
        h.eval()
    sizes = dict(verb=len(verb_classes), noun=len(noun_classes), action=len(action_classes))
    tp = {key: torch.zeros(len(heads), sizes[key], dtype=torch.float64, device=device) for key in KEYS}
    fn = {key: torch.zeros(len(heads), sizes[key], dtype=torch.float64, device=device) for key in KEYS}
    n_seen = torch.zeros(1, dtype=torch.float64, device=device)

    for itr, udata in enumerate(data_loader):  # 끝까지, 한 번만
        clips = udata[0].to(device, non_blocking=True)
        anticipation_times = udata[-1].to(device)
        _verbs, _nouns = udata[1], udata[2]
        labels = dict(
            verb=torch.tensor([verb_classes[int(v)] for v in _verbs], device=device),
            noun=torch.tensor([noun_classes[int(n)] for n in _nouns], device=device),
            action=torch.tensor(
                [action_classes[(int(v), int(n))] for v, n in zip(_verbs, _nouns)], device=device
            ),
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
            feats = model(clips, anticipation_times)
            outs = [h(feats) for h in heads]
        for i, o in enumerate(outs):
            for key in KEYS:
                accumulate(tp[key][i], fn[key][i], o[key], labels[key], k)
        n_seen += len(clips)
        if itr % 50 == 0:
            logger.info(f"[exact val] itr {itr}  clips so far (this rank) {int(n_seen.item())}")

    if dist.is_available() and dist.is_initialized():
        for key in KEYS:
            dist.all_reduce(tp[key])
            dist.all_reduce(fn[key])
        dist.all_reduce(n_seen)

    n_clips = int(n_seen.item())
    per_head = []
    for i in range(len(heads)):
        row = {}
        for key in KEYS:
            r, a, nch = class_mean_recall(tp[key][i], fn[key][i])
            row[key] = dict(recall=r, accuracy=a, n_classes=nch)
        per_head.append(row)

    # 정합성: 모든 clip 이 action 에서 정확히 한 번 TP 또는 FN 이다
    counted = int((tp["action"][0] + fn["action"][0]).sum().item())
    assert counted == n_clips, f"TP+FN({counted}) != n_clips({n_clips})"
    if expected_clips is not None and n_clips != int(expected_clips):
        logger.warning(
            f"[exact val] 센 clip {n_clips} != 기대 {expected_clips} "
            f"(차이 {int(expected_clips) - n_clips}). decode 실패로 건너뛴 clip 이 있다 — 로그의 'Encountered exception' 확인"
        )

    best = {key: dict(recall=max(h[key]["recall"] for h in per_head),
                      accuracy=max(h[key]["accuracy"] for h in per_head)) for key in KEYS}
    return dict(**best, n_clips=n_clips, expected_clips=expected_clips, k=k, per_head=per_head)
