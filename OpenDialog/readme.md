# The OpenDialog

## Motivation

In real-world application, the dialogue technology usually needs to deal with thousands of different scenarios of conversation. In most case, AI developers have to build that conversational agent from from scratch. 

We believe the nature of diversity in the dialogue system application has became the major limitation to the development of dialogue systems. Hence, we set up a new project: **the Opendialog**. Instead of designing specific model for specific dialogue task, we want to find a unified neural dialogue structure, which is:

(1) universal: a unitied model that can jointly tackle all kinds of dialogue tasks.

(2) good performing: be comparable to, or better than, the state-of-the-art performances in all dialogue tasks.

(3) friendly to cold-starting: can be well-tuned with only one hundred examples. 

In this project, we will soon release our BERT-based (Devlin et al., 2018) dialogue general understanding model, which achieves the best performances on 6 different dialogue tasks. The detailed experimental results and our code/model will be available soon through PaddlePaddle.

