#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import os
import six
import ast
import copy

import numpy as np
import paddle.fluid as fluid


def data2tensor(data, place): 
    context_ids = to_lodtensor(list(map(lambda x: x[0], data)), place, 0)
    context_pos = to_lodtensor(list(map(lambda x: x[1], data)), place, 1)
    context_segment = to_lodtensor(list(map(lambda x: x[2], data)), place, 2)
    context_attn = to_lodtensor(list(map(lambda x: x[3], data)), place, 3)
    kn_id = to_lodtensor(list(map(lambda x: x[4], data)), place, 4)
    labels_list = to_lodtensor(list(map(lambda x: x[5], data)), place, 5)
    next_sent_context_index = to_lodtensor(list(map(lambda x: x[6], data)), place, 6)
   
    return {"context_ids": context_ids, \
            "context_pos_ids": context_pos, \
            "context_segment_ids": context_segment, \
            "context_attn_mask": context_attn, \
            "kn_ids": kn_id, \
            "labels": labels_list, \
            "context_next_sent_index": next_sent_context_index}


def to_lodtensor(data, place, type):
    """
    convert ot LODtensor
    """
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    if type in [0, 1, 2, 5, 6]: 
        data = np.array(data).astype("int64")
    if type in [3]: 
        data = np.array(data).astype("float")
    res = fluid.LoDTensor()
    res.set(data, place)
    res.set_lod([lod])
    return res


def init_checkpoint(exe, init_checkpoint_path, main_program):
    assert os.path.exists(
        init_checkpoint_path), "[%s] cann't be found." % init_checkpoint_path
    fluid.io.load_persistables(
        exe, init_checkpoint_path, main_program=main_program)
    print("Load model from {}".format(init_checkpoint_path))


def init_pretraining_params(exe, pretraining_params_path, main_program):
    assert os.path.exists(pretraining_params_path
                          ), "[%s] cann't be found." % pretraining_params_path

    def existed_params(var):
        if not isinstance(var, fluid.framework.Parameter):
            return False
        return os.path.exists(os.path.join(pretraining_params_path, var.name))

    fluid.io.load_vars(
        exe,
        pretraining_params_path,
        main_program=main_program,
        predicate=existed_params)
    #print("Load pretraining parameters from {}".format(pretraining_params_path))

