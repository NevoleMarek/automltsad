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
    def __init__(
        self, d_model: int, dropout: float = 0.1, max_len: int = 5000
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)
