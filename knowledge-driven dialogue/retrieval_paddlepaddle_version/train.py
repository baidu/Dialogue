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
# ecoding=utf8

import os
import time
import numpy as np
import multiprocessing

import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework
from paddle.fluid.executor import Executor

import data_provider as reader
from model import RetrievalModel
from args import base_parser
from args import print_arguments


def create_model(args,
                 num_labels,
                 is_prediction=False):
    context_ids = fluid.layers.data(name='context_ids', shape=[-1, args.max_seq_len, 1], dtype='int64', lod_level=0)
    context_pos_ids = fluid.layers.data(name='context_pos_ids', shape=[-1, args.max_seq_len, 1], dtype='int64', lod_level=0)
    context_segment_ids = fluid.layers.data(name='context_segment_ids', shape=[-1, args.max_seq_len, 1], dtype='int64', lod_level=0)
    context_attn_mask = fluid.layers.data(name='context_attn_mask', shape=[-1, args.max_seq_len, args.max_seq_len], dtype='float', lod_level=0)
    kn_ids = fluid.layers.data(name='kn_ids', shape=[1], dtype='int64', lod_level=1)
    labels = fluid.layers.data(name='labels', shape=[1], dtype='int64', lod_level=0)
    context_next_sent_index = fluid.layers.data(name='context_next_sent_index', shape=[1], dtype='int64', lod_level=0)
    feed_list = ["context_ids", "context_pos_ids", "context_segment_ids", "context_attn_mask", "kn_ids", "labels", "context_next_sent_index"]

    retrieval_model = RetrievalModel(
        context_ids=context_ids,
        context_pos_ids=context_pos_ids,
        context_segment_ids=context_segment_ids,
        context_attn_mask=context_attn_mask,
        kn_ids=kn_ids,
        emb_size=256,
        n_layer=4,
        n_head=8,
        voc_size=14373,
        max_position_seq_len=args.max_seq_len,
        hidden_act="gelu",
        attention_dropout=0.1,
        prepostprocess_dropout=0.1)

    context_cls = retrieval_model.get_context_response_memory(context_next_sent_index)
    context_cls = fluid.layers.dropout(
        x=context_cls,
        dropout_prob=0.1,
        dropout_implementation="upscale_in_train")

    cls_feats = context_cls
    logits = fluid.layers.fc(
        input=cls_feats,
        size=num_labels,
        param_attr=fluid.ParamAttr(
            name="cls_out_w",
            initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
        bias_attr=fluid.ParamAttr(
            name="cls_out_b", initializer=fluid.initializer.Constant(0.)))

    ce_loss, predict = fluid.layers.softmax_with_cross_entropy(
        logits=logits, label=labels, return_softmax=True)
    loss = fluid.layers.reduce_mean(input=ce_loss)

    num_seqs = fluid.layers.create_tensor(dtype='int64')
    accuracy = fluid.layers.accuracy(input=predict, label=labels, total=num_seqs)

    return feed_list, loss, predict, accuracy, num_seqs

def main(args):

    task_name = args.task_name.lower()
    processors = {
        'match': reader.MatchProcessor,
    }

    processor = processors[task_name](data_dir=args.data_dir,
                                      vocab_path=args.vocab_path,
                                      max_seq_len=args.max_seq_len,
                                      do_lower_case=args.do_lower_case)

    num_labels = len(processor.get_labels())
    train_data_generator = processor.data_generator(
        batch_size=args.batch_size,
        phase='train',
        epoch=args.epoch,
        shuffle=True)
    num_train_examples = processor.get_num_examples(phase='train')

    max_train_steps = args.epoch * num_train_examples // args.batch_size
    warmup_steps = int(max_train_steps * args.warmup_proportion)

    feed_order, loss, probs, accuracy, num_seqs = create_model(
            args,
            num_labels=num_labels)

    main_program = fluid.default_main_program()
   
    
    lr_decay = fluid.layers.learning_rate_scheduler.noam_decay(256, warmup_steps)
    with fluid.default_main_program()._lr_schedule_guard():
        learning_rate = lr_decay * args.learning_rate
    optimizer = fluid.optimizer.Adam(
        learning_rate=learning_rate)
    optimizer.minimize(loss)

    if args.use_cuda: 
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()

    exe = Executor(place)
    exe.run(framework.default_startup_program())

    feed_list = [
        main_program.global_block().var(var_name) for var_name in feed_order
    ]
    feeder = fluid.DataFeeder(feed_list, place)
    time_begin = time.time()
    total_cost, total_acc, total_num_seqs = [], [], []
    for batch_id, data in enumerate(train_data_generator()): 
        fetch_outs = exe.run(framework.default_main_program(),
                    feed=feeder.feed(data),
                    fetch_list=[loss, accuracy, num_seqs])
        avg_loss = fetch_outs[0]
        avg_acc = fetch_outs[1]
        cur_num_seqs = fetch_outs[1]
        total_cost.extend(avg_loss * cur_num_seqs)
        total_acc.extend(avg_acc * cur_num_seqs)
        total_num_seqs.extend(cur_num_seqs)
        if batch_id % args.skip_steps == 0: 
            time_end = time.time()
            used_time = time_end - time_begin
            current_example, current_epoch = processor.get_train_progress()
            print("epoch: %d, progress: %d/%d, step: %d, ave loss: %f, "
                "ave acc: %f, speed: %f steps/s" %
                (current_epoch, current_example, num_train_examples,
                batch_id, np.sum(total_cost) / np.sum(total_num_seqs),
                np.sum(total_acc) / np.sum(total_num_seqs),
                args.skip_steps / used_time))
            time_begin = time.time()
            total_cost, total_acc, total_num_seqs = [], [], []

        if batch_id % args.save_steps == 0: 
            model_path = os.path.join(args.checkpoints, str(batch_id))
            if not os.path.isdir(model_path):
                os.makedirs(model_path)

            fluid.io.save_persistables(
                executor=exe,
                dirname=model_path,
                main_program=framework.default_main_program())


if __name__ == '__main__':
    args = base_parser()
    print_arguments(args)
    main(args)
