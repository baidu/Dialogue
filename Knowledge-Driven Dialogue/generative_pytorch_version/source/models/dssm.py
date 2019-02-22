#!/usr/bin/env python
# -*- coding: utf-8 -*-

######################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
# @file source/models/dssm.py
#
######################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from source.models.base_model import BaseModel
from source.modules.embedder import Embedder
from source.modules.encoders.rnn_encoder import RNNEncoder
from source.utils.misc import Pack


class DSSM(BaseModel):
    """
    DSSM
    """

    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 embed_size,
                 hidden_size,
                 padding_idx=None,
                 num_layers=1,
                 bidirectional=True,
                 tie_embedding=False,
                 margin=None,
                 with_project=False,
                 dropout=0.0,
                 use_gpu=False):
        super(DSSM, self).__init__()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.tie_embedding = tie_embedding
        self.dropout = dropout
        self.use_gpu = use_gpu

        self.margin = margin
        self.with_project = with_project

        src_embedder = Embedder(num_embeddings=self.src_vocab_size,
                                embedding_dim=self.embed_size,
                                padding_idx=self.padding_idx)

        self.src_encoder = RNNEncoder(input_size=self.embed_size,
                                      hidden_size=self.hidden_size,
                                      embedder=src_embedder,
                                      num_layers=self.num_layers,
                                      bidirectional=self.bidirectional,
                                      dropout=self.dropout)

        if self.with_project:
            self.project = nn.Linear(in_features=self.hidden_size,
                                     out_features=self.hidden_size,
                                     bias=False)

        if self.tie_embedding:
            assert self.src_vocab_size == self.tgt_vocab_size
            tgt_embedder = src_embedder
        else:
            tgt_embedder = Embedder(num_embeddings=self.tgt_vocab_size,
                                    embedding_dim=self.embed_size,
                                    padding_idx=self.padding_idx)

        self.tgt_encoder = RNNEncoder(input_size=self.embed_size,
                                      hidden_size=self.hidden_size,
                                      embedder=tgt_embedder,
                                      num_layers=self.num_layers,
                                      bidirectional=self.bidirectional,
                                      dropout=self.dropout)

        if self.use_gpu:
            self.cuda()

    def score(self, inputs):
        src_inputs = inputs.src[0][:, 1:-1], inputs.src[1]-2
        tgt_inputs = inputs.tgt[0][:, 1:-1], inputs.tgt[1]-2
        src_hidden = self.src_encoder(src_inputs)[1][-1]
        if self.with_project:
            src_hidden = self.project(src_hidden)
        tgt_hidden = self.tgt_encoder(tgt_inputs)[1][-1]
        logits = (src_hidden * tgt_hidden).sum(dim=-1)
        scores = torch.sigmoid(logits)
        return scores

    def forward(self, src_inputs, pos_tgt_inputs, neg_tgt_inputs,
                src_hidden=None, tgt_hidden=None):
        outputs = Pack()
        src_hidden = self.src_encoder(src_inputs, src_hidden)[1][-1]
        if self.with_project:
            src_hidden = self.project(src_hidden)

        pos_tgt_hidden = self.tgt_encoder(pos_tgt_inputs, tgt_hidden)[1][-1]
        neg_tgt_hidden = self.tgt_encoder(neg_tgt_inputs, tgt_hidden)[1][-1]
        pos_logits = (src_hidden * pos_tgt_hidden).sum(dim=-1)
        neg_logits = (src_hidden * neg_tgt_hidden).sum(dim=-1)
        outputs.add(pos_logits=pos_logits, neg_logits=neg_logits)

        return outputs

    def collect_metrics(self, outputs):

        pos_logits = outputs.pos_logits
        pos_target = torch.ones_like(pos_logits)

        neg_logits = outputs.neg_logits
        neg_target = torch.zeros_like(neg_logits)

        pos_loss = F.binary_cross_entropy_with_logits(
            pos_logits, pos_target, reduction='none')
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_logits, neg_target, reduction='none')

        loss = (pos_loss + neg_loss).mean()

        pos_acc = torch.sigmoid(pos_logits).gt(0.5).float().mean()
        neg_acc = torch.sigmoid(neg_logits).lt(0.5).float().mean()
        margin = (torch.sigmoid(pos_logits) - torch.sigmoid(neg_logits)).mean()
        metrics = Pack(loss=loss, pos_acc=pos_acc,
                       neg_acc=neg_acc, margin=margin)

        num_samples = pos_target.size(0)
        metrics.add(num_samples=num_samples)
        return metrics

    def iterate(self, inputs, optimizer=None, grad_clip=None, is_training=True, epoch=-1):
        src_inputs = inputs.src[0][:, 1:-1], inputs.src[1]-2
        pos_tgt_inputs = inputs.tgt[0][:, 1:-1], inputs.tgt[1]-2

        neg_idx = torch.arange(src_inputs[1].size(0)).type_as(src_inputs[1])
        neg_idx = (neg_idx + 1) % neg_idx.size(0)
        neg_tgt_inputs = pos_tgt_inputs[0][neg_idx], pos_tgt_inputs[1][neg_idx]

        outputs = self.forward(src_inputs, pos_tgt_inputs, neg_tgt_inputs)
        metrics = self.collect_metrics(outputs)

        if is_training:
            assert optimizer is not None
            optimizer.zero_grad()
            loss = metrics.loss
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                clip_grad_norm_(parameters=self.parameters(),
                                max_norm=grad_clip)
            optimizer.step()

        return metrics
