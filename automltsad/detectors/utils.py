# Copyright (c) 2022, Shreshth Tuli
# All rights reserved.
#
# https://github.com/imperial-qore/TranAD

import numpy as np
import torch
import torch.nn as nn


def print_progress(ep, n_ep, t_l, v_l=None):
    if not v_l:
        print(f'[Epoch {ep+1}/{n_ep}] Train l: {t_l}')
    else:
        print(f'[Epoch {ep+1}/{n_ep}] Train l: {t_l} Val l: {v_l}')


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model).float() * (-np.log(10000.0) / d_model)
        )
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        x = x + self.pe[pos : pos + x.size(0), :]
        return self.dropout(x)
