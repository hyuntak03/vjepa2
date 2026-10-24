# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gc
import importlib
import logging

import torch
import torch.nn as nn

from src.models.attentive_pooler import AttentivePooler

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class AttentiveClassifier(nn.Module):

    def __init__(
        self,
        verb_classes: dict,
        noun_classes: dict,
        action_classes: dict,
        embed_dim: int,
        num_heads: int,
        depth: int,
        use_activation_checkpointing: bool,
    ):
        super().__init__()
        self.num_verb_classes = len(verb_classes)
        num_noun_classes = len(noun_classes)
        num_action_classes = len(action_classes)
        self.action_only = self.num_verb_classes == 0

        self.pooler = AttentivePooler(
            num_queries=1 if self.action_only else 3,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            use_activation_checkpointing=use_activation_checkpointing,
        )
        if not self.action_only:
            self.verb_classifier = nn.Linear(embed_dim, self.num_verb_classes, bias=True)
            self.noun_classifier = nn.Linear(embed_dim, num_noun_classes, bias=True)
        self.action_classifier = nn.Linear(embed_dim, num_action_classes, bias=True)

    def forward(self, x):
        if torch.isnan(x).any():
            print("Nan detected at output of encoder")
            exit(1)

        x = self.pooler(x)  # [B, 2, D]
        if not self.action_only:
            x_verb, x_noun, x_action = x[:, 0, :], x[:, 1, :], x[:, 2, :]
            x_verb = self.verb_classifier(x_verb)
            x_noun = self.noun_classifier(x_noun)
            x_action = self.action_classifier(x_action)
            return dict(
                verb=x_verb,
                noun=x_noun,
                action=x_action,
            )
        else:
            x_action = x[:, 0, :]
            x_action = self.action_classifier(x_action)
            return dict(action=x_action)


def init_multi_module(
    module_name,
    device,
    frames_per_clip,
    frames_per_second,
    resolution,
    checkpoint,
    model_kwargs,
    wrapper_kwargs,
    predictor_group,
):
    """frozen encoder 하나 위에 predictor 여러 개 (2026-10-18).

    `predictor_group` = [{name, predictor_checkpoint (없거나 null 이면 릴리즈 predictor)}, ...].
    encoder 와 체크포인트는 첫 모델이 올리고 나머지는 **같은 객체**를 쓴다 (modelcustom.init_module(shared=...)).
    """
    import importlib

    mod = importlib.import_module(f"{module_name}")
    if not hasattr(mod, "MultiAnticipative"):
        raise RuntimeError(f"{module_name} 은 predictor_group 을 지원하지 않는다 (MultiAnticipative 가 없다)")
    shared, models, names = {}, [], []
    for g in predictor_group:
        mk = dict(model_kwargs or {})
        if g.get("predictor_checkpoint"):
            mk["predictor_checkpoint"] = g["predictor_checkpoint"]
        else:
            mk.pop("predictor_checkpoint", None)          # 릴리즈 predictor
        logger.info(f"[predictor_group] {g['name']} <- {g.get('predictor_checkpoint') or '(릴리즈 predictor)'}")
        models.append(
            mod.init_module(
                frames_per_clip=frames_per_clip,
                frames_per_second=frames_per_second,
                resolution=resolution,
                checkpoint=checkpoint,
                model_kwargs=mk,
                wrapper_kwargs=wrapper_kwargs,
                shared=shared,
            )
        )
        names.append(g["name"])
    # ⚠️ 2026-10-18 호스트 RAM OOM 수정. 릴리즈 체크포인트는 **9.66 GB** (encoder + target_encoder + predictor + opt + scaler) 인데
    #    `shared["ckpt"]` 가 그것을 run 내내 붙들고 있었다 — rank 6 개면 58 GB 다. 5 팔 + ctx_encoder 에서 job 의 300 GB 캡을
    #    넘겨 rank 3 의 DataLoader worker 가 SIGKILL 됐고 (failcnt 703 만) 나머지 5 rank 가 allreduce 에서 공회전했다.
    #    모델을 다 지은 뒤에는 쓰지 않으므로 여기서 놓는다.
    shared.pop("ckpt", None)
    gc.collect()
    model = mod.MultiAnticipative(models, names).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def init_module(
    module_name,
    device,
    frames_per_clip,
    frames_per_second,
    resolution,
    checkpoint,
    model_kwargs,
    wrapper_kwargs,
):
    """
    Build (frozen) model and initialize from pretrained checkpoint

    API requirements for "model" module:
      1) Needs to be a pytorch module with 'forward()' function protocol:
        :param x: (Tensor) Video clip (shape=[batch_size x num_channels x num_frames x height x width])
        :param anticipation_time: (Tensor) Seconds into the future to predict for each sample in batch
            (shape=[batch_size])
        :returns: (Tensor) Representations of future frames (shape=[batch_size x num_output_tokens x feature_dim])

      2) Needs to have a public attribute called 'embed_dim' (int) describing its
         output feature dimension.
    """
    model = (
        importlib.import_module(f"{module_name}")
        .init_module(
            frames_per_clip=frames_per_clip,
            frames_per_second=frames_per_second,
            resolution=resolution,
            checkpoint=checkpoint,
            model_kwargs=model_kwargs,
            wrapper_kwargs=wrapper_kwargs,
        )
        .to(device)
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(model)
    return model


def init_classifier(
    embed_dim: int,
    num_heads: int,
    num_blocks: int,
    device: torch.device,
    num_classifiers: int,
    action_classes: dict,
    verb_classes: dict,
    noun_classes: dict,
    use_activation_checkpointing: bool = True,
):
    """2026-10-18: `use_activation_checkpointing` 를 인자로 뺐다 (전엔 True 로 박혀 있었다).

    AttentivePooler 의 self-attention block (depth-1 개) 은 **토큰 전체** (EK100 에서 4,352 x 1280) 위에서 돈다 —
    query 3 개가 아니라. checkpointing 은 역전파 때 그 forward 를 **다시 계산**해 activation 메모리를 아낀다.
    frozen encoder 위 probe 학습은 VRAM 이 남으므로 (96 GB 중 19) 그 교환이 손해다. False 면 head 하나당
    약 1.6 GB 를 더 쓰고 head 연산을 약 1/4 아낀다 (head 는 순차 처리라 메모리가 head 수만큼 쌓이지 않는다).
    """
    classifiers = [
        AttentiveClassifier(
            verb_classes=verb_classes,
            noun_classes=noun_classes,
            action_classes=action_classes,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=num_blocks,
            use_activation_checkpointing=use_activation_checkpointing,
        ).to(device)
        for _ in range(num_classifiers)
    ]
    print(classifiers[0])
    return classifiers
