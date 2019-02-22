#!/usr/bin/env python
# -*- coding: utf-8 -*-

######################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
# @file source/models/base_model.py
#
######################################################################

import os
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, *input):
        raise NotImplementedError

    def __repr__(self):
        main_string = super().__repr__()
        num_parameters = sum([p.nelement() for p in self.parameters()])
        main_string += f"\nNumber of parameters: {num_parameters}\n"
        return main_string

    def save(self, filename):
        torch.save(self.state_dict(), filename)
        print(f"Saved model state to '{filename}'!")

    def load(self, filename):
        if os.path.isfile(filename):
            state_dict = torch.load(
                filename, map_location=lambda storage, loc: storage)
            self.load_state_dict(state_dict, strict=False)
            print(f"Loaded model state from '{filename}'")
        else:
            print(f"Invalid model state file: '{filename}'")
