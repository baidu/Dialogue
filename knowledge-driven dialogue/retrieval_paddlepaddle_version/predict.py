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
"""Load checkpoint of running classifier to do prediction and save inference model."""

import os
import time
import numpy as np
import paddle.fluid as fluid

import data_provider as reader

from train import create_model
from args_infer import parse_classifier_args, print_arguments
from utils import init_pretraining_params
import paddle.fluid.framework as framework
import parser

def parse_args():
    parser = parse_classifier_args(ret_parser=True)
    args = parser.parse_args()
    return args


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
    infer_data_generator = processor.data_generator(
        batch_size=args.batch_size,
        phase='dev',
        epoch=args.epoch,
        shuffle=False)

    main_program = fluid.default_main_program()
    feed_order, loss, probs, accuracy, num_seqs = create_model(
                args,
                num_labels=num_labels)

    if args.use_cuda: 
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()

    exe = fluid.Executor(place)
    exe.run(framework.default_startup_program())

    if args.init_checkpoint: 
        init_pretraining_params(exe, args.init_checkpoint, main_program)

    feed_list = [
        main_program.global_block().var(var_name) for var_name in feed_order
        ]
    feeder = fluid.DataFeeder(feed_list, place)

    label_list = []
    for batch_id, data in enumerate(infer_data_generator()): 
        results = exe.run(
                fetch_list=[probs],
                feed=feeder.feed(data),
                return_numpy=True)
        for elem in results[0]: 
            label_list.append(str(elem[1]))

    return label_list


if __name__ == '__main__':
    args = parse_args()
    label_list = main(args)
    json_file = os.path.join(args.data_dir, "dev.txt")
    parser.dump_json(json_file, label_list)
