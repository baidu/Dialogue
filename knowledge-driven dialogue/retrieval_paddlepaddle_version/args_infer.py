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

import six
import argparse


def base_parser(des):
    parser = argparse.ArgumentParser(description=des)
    parser.add_argument(
        '--epoch',
        type=int,
        default=1,
        help='Number of epoches for training. (default: %(default)d)')
    parser.add_argument(
        '--max_seq_len',
        type=int,
        default=512,
        help='Number of word of the longest seqence. (default: %(default)d)')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8096,
        help='Total token number in batch for training. (default: %(default)d)')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1.0,
        help='Learning rate used to train with warmup. (default: %(default)f)')
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.01,
        help='Weight decay rate for L2 regularizer. (default: %(default)f)')
    parser.add_argument(
        '--checkpoints',
        type=str,
        default="checkpoints",
        help='Path to save checkpoints. (default: %(default)s)')
    parser.add_argument(
        '--init_checkpoint',
        type=str,
        default=None,
        help='init checkpoint to resume training from. (default: %(default)s)')
    parser.add_argument(
        '--vocab_path',
        type=str,
        default=None,
        help='Vocabulary path. (default: %(default)s)')
    parser.add_argument(
        '--data_dir',
        type=str,
        default="./real_data",
        help='Path of training data. (default: %(default)s)')
    parser.add_argument(
        '--skip_steps',
        type=int,
        default=10,
        help='The steps interval to print loss. (default: %(default)d)')
    parser.add_argument(
        '--save_steps',
        type=int,
        default=10000,
        help='The steps interval to save checkpoints. (default: %(default)d)')
    parser.add_argument(
        '--sent_types',
        type=int,
        default=3,
        help='Number of sentence types. %(default)d)')
    parser.add_argument(
        '--validation_steps',
        type=int,
        default=1000,
        help='The steps interval to evaluate model performance on validation '
        'set. (default: %(default)d)')
    parser.add_argument(
        '--is_distributed',
        action='store_true',
        help='If set, then start distributed training')
    parser.add_argument(
        '--use_cuda', action='store_true', help='If set, use GPU for training.')
    parser.add_argument(
        '--use_fast_executor',
        action='store_true',
        help='If set, use fast parallel executor (in experiment).')

    return parser

# Additional args for clssifier
def parse_classifier_args(ret_parser=False):
    parser = base_parser(des="Arguments for running classifier.")
    parser.add_argument(
        '--init_pretraining_params',
        type=str,
        default=None,
        help="Init pre-training params which preforms fine-tuning from. If the "
        "arg 'init_checkpoint' has been set, this argument wouldn't be valid. "
        '(default: %(default)s)')
    parser.add_argument(
        '--bert_config_path',
        type=str,
        default=None,
        help="Path to the json file for bert model config. "
        "(default: %(default)s)")
    parser.add_argument(
        '--task_name',
        type=str,
        default=None,
        help="The name of task to perform fine-tuning, "
        "choices=[xnli, mnli, lcqmc, cola, mrpc]."
        "(default: %(default)s)")
    parser.add_argument(
        '--do_lower_case',
        type=bool,
        default=True,
        choices=[True, False],
        help="Whether to lower case the input text. Should be True for uncased "
        "models and False for cased models.")
    parser.add_argument(
        '--warmup_proportion',
        type=float,
        default=0.1,
        help='proportion warmup. (default: %(default)f)')
    if ret_parser:
        return parser
    args = parser.parse_args()
    return args

def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')

