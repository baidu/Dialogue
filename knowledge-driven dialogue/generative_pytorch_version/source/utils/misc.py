#!/usr/bin/env python
# -*- coding: utf-8 -*-

######################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
# @file source/utils/misc.py
#
######################################################################

import torch
import argparse


class Pack(dict):
    def __getattr__(self, name):
        return self.get(name)

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def flatten(self):
        pack_list = []
        for vs in zip(*self.values()):
            pack = Pack(zip(self.keys(), vs))
            pack_list.append(pack)
        return pack_list

    def cuda(self, device=None):
        pack = Pack()
        for k, v in self.items():
            if isinstance(v, tuple):
                pack[k] = tuple(x.cuda(device) for x in v)
            else:
                pack[k] = v.cuda(device)
        return pack


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    if max_len is None:
        max_len = lengths.max().item()
    mask = torch.arange(0, max_len, dtype=torch.long).type_as(lengths)
    mask = mask.repeat(*lengths.size(), 1).lt(lengths.unsqueeze(-1))
    return mask


def max_lens(X):
    if not isinstance(X[0], list):
        return [len(X)]
    elif not isinstance(X[0][0], list):
        return [len(X), max(len(x) for x in X)]
    elif not isinstance(X[0][0][0], list):
        return [len(X), max(len(x) for x in X),
                max(len(x) for xs in X for x in xs)]
    else:
        raise ValueError(
            "Data list whose dim is greater than 3 is not supported!")


def list2tensor(X):
    size = max_lens(X)
    if len(size) == 1:
        tensor = torch.tensor(X)
        return tensor

    tensor = torch.zeros(size, dtype=torch.long)
    lengths = torch.zeros(size[:-1], dtype=torch.long)
    if len(size) == 2:
        for i, x in enumerate(X):
            l = len(x)
            tensor[i, :l] = torch.tensor(x)
            lengths[i] = l
    else:
        for i, xs in enumerate(X):
            for j, x in enumerate(xs):
                l = len(x)
                tensor[i, j, :l] = torch.tensor(x)
                lengths[i, j] = l

    return tensor, lengths


# def list2tensor(X, max_len=None):
#     sizes = max_lens(X)
#
#     if len(sizes) == 1:
#         tensor = torch.tensor(X)
#         return tensor
#
#     if max_len is not None:
#         assert max_len >= sizes[-1]
#         sizes[-1] = max_len
#
#     tensor = torch.zeros(sizes, dtype=torch.long)
#     lengths = torch.zeros(sizes[:-1], dtype=torch.long)
#     if len(sizes) == 2:
#         for i, x in enumerate(X):
#             l = len(x)
#             tensor[i, :l] = torch.tensor(x)
#             lengths[i] = l
#     else:
#         for i, xs in enumerate(X):
#             for j, x in enumerate(xs):
#                 l = len(x)
#                 tensor[i, j, :l] = torch.tensor(x)
#                 lengths[i, j] = l
#
#     return tensor, lengths


# def one_hot(indice, vocab_size):
#     T = torch.zeros(*indice.size(), vocab_size).type_as(indice).float()
#     T = T.scatter(-1, indice.unsqueeze(-1), 1)
#     return T

def one_hot(indice, num_classes):
    I = torch.eye(num_classes).to(indice.device)
    T = I[indice]
    return T


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


if __name__ == '__main__':
    X = [1, 2, 3]
    print(X)
    print(list2tensor(X))
    X = [X, [2, 3]]
    print(X)
    print(list2tensor(X))
    X = [X, [[1, 1, 1, 1, 1]]]
    print(X)
    print(list2tensor(X))

    data_list = [{'src': [1, 2, 3], 'tgt': [1, 2, 3, 4]},
                 {'src': [2, 3], 'tgt': [1, 2, 4]}]
    batch = Pack()
    for key in data_list[0].keys():
        batch[key] = list2tensor([x[key] for x in data_list], 8)
    print(batch)
