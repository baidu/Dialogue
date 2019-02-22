#!/bin/bash

######################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
# @file run_test.sh
#
######################################################################

pythonpath='python'
prefix=demo
datapart=dev
datapath=./data

corpus_file=${datapath}/resource/${datapart}.txt
sample_file=${datapath}/resource/sample.${datapart}.txt
text_file=${datapath}/${prefix}.test
topic_file=${datapath}/${prefix}.test.topic

if [ "${datapart}"x = "test"x ]; then
    sample_file=${corpus_file}
else
    ${pythonpath} ./tools/convert_session_to_sample.py ${corpus_file} ${sample_file}
fi

${pythonpath} ./tools/convert_conversation_corpus_to_model_text.py ${sample_file} ${text_file} ${topic_file} 1

${pythonpath} ./network.py --test --ckpt models/best.model --gen_file ./output/test.result --use_posterior False --gpu 0 > log.txt 2>&1

${pythonpath} ./tools/topic_materialization.py ./output/test.result ./output/test.result.final ${topic_file}

if [ "${datapart}"x != "test"x ]; then
    ${pythonpath} ./tools/convert_result_for_eval.py ${sample_file} ./output/test.result.final ./output/test.result.eval
    ${pythonpath} ./tools/eval.py ./output/test.result.eval
fi

