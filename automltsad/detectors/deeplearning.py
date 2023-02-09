# Copyright (c) 2022, Shreshth Tuli
# All rights reserved.
#
# https://github.com/imperial-qore/TranAD
import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.nn import GATConv
from torch.nn import TransformerDecoder, TransformerEncoder

from automltsad.detectors.utils import (
    PositionalEncoding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)

torch.manual_seed(1)


## GDN Model (AAAI 21)
class GDN(nn.Module):
    def __init__(self, feats):
        super(GDN, self).__init__()
        self.name = 'GDN'
        self.lr = 0.0001
        self.n_feats = feats
        self.n_window = 5
        self.n_hidden = 16
        self.n = self.n_window * self.n_feats
        src_ids = np.repeat(np.array(list(range(feats))), feats)
        dst_ids = np.array(list(range(feats)) * feats)
        self.g = dgl.graph((torch.tensor(src_ids), torch.tensor(dst_ids)))
        self.g = dgl.add_self_loop(self.g)
        self.feature_gat = GATConv(1, 1, feats)
        self.attention = nn.Sequential(
            nn.Linear(self.n, self.n_hidden),
            nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_window),
            nn.Softmax(dim=0),
        )
        self.fcn = nn.Sequential(
            nn.Linear(self.n_feats, self.n_hidden),
            nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_window),
            nn.Sigmoid(),
        )

    def forward(self, data):
        # Bahdanau style attention
        att_score = self.attention(data).view(self.n_window, 1)
        data = data.view(self.n_window, self.n_feats)
        data_r = torch.matmul(data.permute(1, 0), att_score)
        # GAT convolution on complete graph
        feat_r = self.feature_gat(self.g, data_r)
        feat_r = feat_r.view(self.n_feats, self.n_feats)
        # Pass through a FCN
        x = self.fcn(feat_r)
        return x.view(-1)


# Proposed Model + Self Conditioning + Adversarial + MAML (VLDB 22)
class TranAD(nn.Module):
    def __init__(self, feats):
        super(TranAD, self).__init__()
        self.name = 'TranAD'
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers1 = TransformerDecoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * np.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2


class VAE(nn.Module):
    def __init__(
        self,
        window_size,
        encoder_hidden=[128, 64, 32],
        decoder_hidden=[32, 64, 128],
        latent_dim=16,
    ):
        super(VAE, self).__init__()
        self.name = 'VAE'

        self.window_size = window_size
        self.encoder_hidden = encoder_hidden
        self.decoder_hidden = decoder_hidden
        self.latent_dim = latent_dim

        self.encoder = self._build_encoder()
        self.mu = nn.Linear(self.encoder_hidden[-1], latent_dim)
        self.sigma = nn.Linear(self.encoder_hidden[-1], latent_dim)
        self.decoder = self._build_decoder()

        self.N = torch.distributions.Normal(0, 1)

    def _build_encoder(self):
        self.encoder_layers = []
        self.encoder_layers.append(
            nn.Sequential(
                nn.Linear(self.window_size, self.encoder_hidden[0]), nn.ReLU()
            )
        )

        self.encoder_layers.extend(
            [
                nn.Sequential(
                    nn.Linear(
                        self.encoder_hidden[i], self.encoder_hidden[i + 1]
                    ),
                    nn.ReLU(),
                )
                for i in range(len(self.encoder_hidden) - 1)
            ]
        )
        return nn.Sequential(*self.encoder_layers)

    def _build_decoder(self):
        self.decoder_layers = []
        self.decoder_layers.append(
            nn.Sequential(
                nn.Linear(self.latent_dim, self.decoder_hidden[0]), nn.ReLU()
            )
        )
        self.decoder_layers.extend(
            [
                nn.Sequential(
                    nn.Linear(
                        self.decoder_hidden[i], self.decoder_hidden[i + 1]
                    ),
                    nn.ReLU(),
                )
                for i in range(len(self.decoder_hidden) - 1)
            ]
        )

        self.decoder_layers.append(
            nn.Linear(self.decoder_hidden[-1], self.window_size)
        )
        return nn.Sequential(*self.decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        mu, sigma = self.mu(x), torch.exp(self.sigma(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1 / 2).sum()
        x = self.decoder(z)
        return x
