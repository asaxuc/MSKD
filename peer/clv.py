# Copyright (c) Meta Platforms, Inc. and affiliates.
import copy

import numpy
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F
from torch import nn
import torch
import torch.nn as nn




class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        # >>> loss = InfoNCE()
        # >>> batch_size, num_negative, embedding_size = 32, 48, 128
        # >>> query = torch.randn(batch_size, embedding_size)
        # >>> positive_key = torch.randn(batch_size, embedding_size)
        # >>> negative_keys = torch.randn(num_negative, embedding_size)
        # >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='paired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


def CLV(outputs, targets):
    if len(outputs.shape) == 2:
        outputs = outputs.unsqueeze(dim=-1)
    # outputs (B C D)
    # targets (B C)

    # step1: select activated outputs
    B, C = targets.shape
    mq = outputs.view(-1, outputs.shape[-1])
    mt = targets.flatten()   # 0 0 1 0 1 0 0 1 0 1 0 0
    indi = numpy.where(mt.detach().cpu().numpy() == 1)[0]  # (0 6 8 14 27 334343523151343) 长度为 L
    indexes = torch.from_numpy(indi).to(torch.long).cuda()   # (0 6 8 14 27 334343523151343) 长度为 L
    q = mq[indexes]  # (0 6 8 14 27 34t3454 )的 output 数量为 L
    sl = [i for i in range(C)]*B
    mc_1 = torch.tensor(sl).cuda()   #  0 1 2 3 。。 C 0 1 2 3  。。 C
    # mc_2 = torch.arange(B).unsqueeze(dim=1).repeat(1,C).flatten().cuda()   # 0 0 0 0 0 0 1 1 1 1 1 1 1
    mc1_o = mc_1[indi]  # (0 6 0 5 0 3 )  长度仍为L

    o1 = outputs.permute(1,0,2)[mc1_o] # (L B D)
    mask = targets.T[mc1_o] # (L B)

    loss = 0
    infonce = InfoNCE()
    for query, pos_keys, pos_mask, col in zip(q, o1, mask, mc1_o):
        query = query.view(1, query.shape[0])
        pos_mask = pos_mask.to(torch.bool)

        pos_keys = pos_keys[pos_mask]
        if len(pos_keys.shape) == 1:
            pos_keys = pos_keys.unsqueeze(dim=0)

        li = []
        front = outputs[~pos_mask,col]
        if len(front.shape) == 1:
            front = front.unsqueeze(dim=0)
        li.append(front)

        temp = [i for i in range(C)]
        del temp[col]
        for i in temp:
            li.append(outputs[:,i])
        neg_key = torch.cat(li)

        for pos_key in pos_keys:
            pos_key = pos_key.unsqueeze(dim=0)
            loss -= infonce.forward(query, pos_key, neg_key)

    loss /= len(q)
    return 0.01*loss


# a = torch.randn((3,4,5)).cuda()
# b = torch.randn((3,4)).cuda()
# c = torch.randint_like(b, 0, 2).cuda()
#
# print("outputs:",a)
# print("targets:",c)
#
# CLV(a,c)
#
#
#

