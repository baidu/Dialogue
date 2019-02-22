#!/usr/bin/env python
# -*- coding: utf-8 -*-

######################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
# @file topic_materialization.py
#
######################################################################

import sys
import json

def topic_materialization(input_file, output_file, topic_file):
    inputs = [line.strip() for line in open(input_file, 'r')]
    topics = [line.strip() for line in open(topic_file, 'r')]

    assert len(inputs) == len(topics)

    fout = open(output_file, 'w')
    for i, text in enumerate(inputs):
        topic_dict = json.loads(topics[i], encoding="utf-8")
        topic_list = sorted(topic_dict.items(), key=lambda item: len(item[1]), reverse=True)
        for key, value in topic_list:
            text = text.replace(key, value)
        fout.write(text + "\n")

    fout.close()


def main():
    topic_materialization(sys.argv[1],
                          sys.argv[2],
                          sys.argv[3])


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
