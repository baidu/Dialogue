#!/bin/bash

######################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
# @file run_train.sh
#
######################################################################

pythonpath='python'
prefix=demo
datatype=(train dev)
datapath=./data

for ((i=0; i<${#datatype[*]}; i++))
do
    corpus_file=${datapath}/resource/${datatype[$i]}.txt
    sample_file=${datapath}/resource/sample.${datatype[$i]}.txt
    text_file=${datapath}/${prefix}.${datatype[$i]}
    topic_file=${datapath}/${prefix}.${datatype[$i]}.topic

    ${pythonpath} ./tools/convert_session_to_sample.py ${corpus_file} ${sample_file}

    ${pythonpath} ./tools/convert_conversation_corpus_to_model_text.py ${sample_file} ${text_file} ${topic_file} 1
done

cp ${datapath}/${prefix}.dev ${datapath}/${prefix}.test

${pythonpath} ./network.py --gpu 0 > log.txt