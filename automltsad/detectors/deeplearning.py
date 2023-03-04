# Copyright (c) 2022, Shreshth Tuli
# All rights reserved.
#
# https://github.com/imperial-qore/TranAD
import dgl
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.nn import GATConv
from torch.nn import TransformerDecoder, TransformerEncoder

from automltsad.detectors.utils import PositionalEncoding

torch.manual_seed(1)


## GDN Model (AAAI 21)
class GDN(pl.LightningModule):
    def __init__(self, n_feats, window_size, n_hidden, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.n_feats = n_feats
        self.n_window = window_size
        self.n_hidden = n_hidden
        self.n = self.n_window * self.n_feats
        src_ids = np.repeat(np.array(list(range(self.n_feats))), self.n_feats)
        dst_ids = np.array(list(range(self.n_feats)) * self.n_feats)
        self.g = dgl.graph((torch.tensor(src_ids), torch.tensor(dst_ids)))
        self.g = dgl.add_self_loop(self.g)
        self.feature_gat = GATConv(1, 1, self.n_feats)

        self.fcn = nn.Sequential(
            nn.Linear(self.n_feats, self.n_hidden),
            nn.LeakyReLU(),
            nn.Linear(self.n_hidden, self.n_feats),
        )

    def forward(self, x):
        n_s, n_t, n_f = x.shape
        feat_r = self.feature_gat(self.g, x.view(n_f, n_s, n_t, 1))
        y_hat = self.fcn(feat_r).squeeze()[:, -1].view(-1, 1)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        n_s, n_t, n_f = x.shape
        feat_r = self.feature_gat(self.g, x.view(n_f, n_s, n_t, 1))
        y_hat = self.fcn(feat_r).squeeze()[:, -1]
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        n_s, n_t, n_f = x.shape
        feat_r = self.feature_gat(self.g, x.view(n_f, n_s, n_t, 1))
        y_hat = self.fcn(feat_r).squeeze()[:, -1]
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self(x)
        return F.mse_loss(y, y_hat, reduction='none')

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


# Proposed Model + Self Conditioning + Adversarial + MAML (VLDB 22)
class TranAD(pl.LightningModule):
    def __init__(
        self, n_feats, learning_rate, window_size, n_layers, ff_dim, nhead
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.n_feats = n_feats
        self.n_window = window_size
        self.n = self.n_feats * self.n_window
        self.n_layers = n_layers
        self.ff_dim = ff_dim
        self.nhead = nhead
        self.pos_encoder = PositionalEncoding(
            2 * self.n_feats, 0.1, self.n_window
        )
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=2 * self.n_feats,
            nhead=self.nhead,
            dim_feedforward=ff_dim,
            dropout=0.1,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, n_layers
        )
        decoder_layers1 = nn.TransformerDecoderLayer(
            d_model=2 * self.n_feats,
            nhead=self.nhead,
            dim_feedforward=ff_dim,
            dropout=0.1,
        )
        self.transformer_decoder1 = nn.TransformerDecoder(
            decoder_layers1, n_layers
        )
        decoder_layers2 = nn.TransformerDecoderLayer(
            d_model=2 * self.n_feats,
            nhead=self.nhead,
            dim_feedforward=ff_dim,
            dropout=0.1,
        )
        self.transformer_decoder2 = nn.TransformerDecoder(
            decoder_layers2, n_layers
        )
        self.fcn = nn.Sequential(nn.Linear(2 * self.n_feats, self.n_feats))
        self.sqrtn_feats = np.sqrt(self.n_feats)

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * self.sqrtn_feats
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, batch):
        src, tgt = batch
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)
        n = self.current_epoch + 1

        # Phase 1
        fcs = torch.zeros_like(src)
        O_1 = self.fcn(self.transformer_decoder1(*self.encode(src, fcs, tgt)))
        O_2 = self.fcn(self.transformer_decoder2(*self.encode(src, fcs, tgt)))

        # Phase 2
        fcs = (O_1 - src) ** 2
        O_2_hat = self.fcn(
            self.transformer_decoder2(*self.encode(src, fcs, tgt))
        )

        return O_1, O_2, O_2_hat

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)
        n = self.current_epoch + 1

        # Phase 1
        fcs = torch.zeros_like(src)
        O_1 = self.fcn(self.transformer_decoder1(*self.encode(src, fcs, tgt)))
        O_2 = self.fcn(self.transformer_decoder2(*self.encode(src, fcs, tgt)))

        # Phase 2
        fcs = (O_1 - src) ** 2
        O_2_hat = self.fcn(
            self.transformer_decoder2(*self.encode(src, fcs, tgt))
        )

        # Losses
        l_1 = 0.95**n * F.mse_loss(O_1, src) + (1 - 0.95**n) * F.mse_loss(
            O_2_hat, src
        )
        l_2 = 0.95**n * F.mse_loss(O_2, src) - (1 - 0.95**n) * F.mse_loss(
            O_2_hat, src
        )
        loss = l_1 + l_2
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)
        n = self.current_epoch + 1

        # Phase 1
        fcs = torch.zeros_like(src)
        O_1 = self.fcn(self.transformer_decoder1(*self.encode(src, fcs, tgt)))
        O_2 = self.fcn(self.transformer_decoder2(*self.encode(src, fcs, tgt)))

        # Phase 2
        fcs = (O_1 - src) ** 2
        O_2_hat = self.fcn(
            self.transformer_decoder2(*self.encode(src, fcs, tgt))
        )

        # Losses
        l_1 = 0.95**n * F.mse_loss(O_1, src) + (1 - 0.95**n) * F.mse_loss(
            O_2_hat, src
        )
        l_2 = 0.95**n * F.mse_loss(O_2, src) - (1 - 0.95**n) * F.mse_loss(
            O_2_hat, src
        )
        loss = l_1 + l_2
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        O_1, O_2, O_2_hat = self(batch)
        batch = batch[0].permute(1, 0, 2)
        out = 1 / 2 * F.mse_loss(
            O_1, batch, reduction='none'
        ) + 1 / 2 * F.mse_loss(O_2_hat, batch, reduction='none')
        return out.mean(dim=0)

    def backward(self, loss, *args, **kwargs):
        loss.backward(retain_graph=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


# Copyright (c) 2023, Marek Nevole
# All rights reserved.
class VAE(pl.LightningModule):
    def __init__(
        self,
        window_size,
        encoder_hidden=[128, 64, 32],
        decoder_hidden=[32, 64, 128],
        latent_dim=16,
        learning_rate=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

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
        in_features = self.window_size
        self.encoder_layers = []
        for h_dim in self.encoder_hidden:
            self.encoder_layers.append(
                nn.Sequential(
                    nn.Linear(in_features, h_dim, bias=True),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_features = h_dim
        return nn.Sequential(*self.encoder_layers)

    def _build_decoder(self):
        self.decoder_layers = []
        in_features = self.latent_dim
        for d_dim in self.decoder_hidden:
            self.decoder_layers.append(
                nn.Sequential(
                    nn.Linear(in_features, d_dim, bias=True),
                    nn.BatchNorm1d(d_dim),
                    nn.LeakyReLU(),
                )
            )
            in_features = d_dim

        self.decoder_layers.append(nn.Linear(in_features, self.window_size))
        return nn.Sequential(*self.decoder_layers)

    def forward(self, batch):
        x = batch
        z = self.encoder(x)
        mu, sigma = self.mu(z), torch.exp(self.sigma(z))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = ((sigma**2 + mu**2) / 2 - torch.log(sigma) - 1 / 2).sum()
        x_hat = self.decoder(z)
        return x_hat

    def training_step(self, batch, batch_idx):
        x = batch
        z = self.encoder(x)
        mu, sigma = self.mu(z), torch.exp(self.sigma(z))
        z = mu + sigma * self.N.sample(mu.shape).to(self.device)
        self.kl = ((sigma**2 + mu**2) / 2 - torch.log(sigma) - 1 / 2).sum()
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x) + 0.0001 * self.kl
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        z = self.encoder(x)
        mu, sigma = self.mu(z), torch.exp(self.sigma(z))
        z = mu + sigma * self.N.sample(mu.shape).to(self.device)
        self.kl = ((sigma**2 + mu**2) / 2 - torch.log(sigma) - 1 / 2).sum()
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        out = self(batch)
        err = ((out - batch) ** 2).sum(dim=1)
        return err

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


# Copyright (c) 2018 Chair of Data Mining at Hasso Plattner Institute
class LSTM_AE(pl.LightningModule):
    def __init__(
        self,
        n_feats,
        hidden_size,
        n_layers=1,
        dropout=(0.1, 0.1),
        learning_rate=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.n_feats = n_feats
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.enc = nn.LSTM(
            input_size=self.n_feats,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            dropout=self.dropout[0],
            batch_first=True,
        )
        self.dec = nn.LSTM(
            input_size=self.n_feats,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            dropout=self.dropout[1],
            batch_first=True,
        )
        self.out = nn.Linear(self.hidden_size, self.n_feats)

    def forward(self, batch):
        x = batch
        enc_state = (
            torch.zeros(self.n_layers, x.shape[0], self.hidden_size),
            torch.zeros(self.n_layers, x.shape[0], self.hidden_size),
        )
        _, enc_state = self.enc(x, enc_state)

        dec_state = enc_state

        out = torch.zeros_like(x)
        for i in reversed(range(x.shape[1])):
            out[:, i, :] = self.out(dec_state[0][-1])

            if self.training:
                _, dec_state = self.dec(x[:, i].unsqueeze(1), dec_state)
            else:
                _, dec_state = self.dec(out[:, i].unsqueeze(1), dec_state)
        return out

    def training_step(self, batch, batch_idx):
        x = batch
        enc_state = (
            torch.zeros(self.n_layers, x.shape[0], self.hidden_size),
            torch.zeros(self.n_layers, x.shape[0], self.hidden_size),
        )
        _, enc_state = self.enc(x, enc_state)

        dec_state = enc_state

        out = torch.zeros_like(x)
        for i in reversed(range(x.shape[1])):
            out[:, i, :] = self.out(dec_state[0][-1])

            if self.training:
                _, dec_state = self.dec(x[:, i].unsqueeze(1), dec_state)
            else:
                _, dec_state = self.dec(out[:, i].unsqueeze(1), dec_state)

        loss = F.l1_loss(out, x)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        enc_state = (
            torch.zeros(self.n_layers, x.shape[0], self.hidden_size),
            torch.zeros(self.n_layers, x.shape[0], self.hidden_size),
        )
        _, enc_state = self.enc(x, enc_state)

        dec_state = enc_state

        out = torch.zeros_like(x)
        for i in reversed(range(x.shape[1])):
            out[:, i, :] = self.out(dec_state[0][-1])

            if self.training:
                _, dec_state = self.dec(x[:, i].unsqueeze(1), dec_state)
            else:
                _, dec_state = self.dec(out[:, i].unsqueeze(1), dec_state)

        loss = F.l1_loss(out, x)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


class ConvAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        input_size,
        input_channels,
        n_layers,
        latent_dim=16,
        learning_rate=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.n_layers = n_layers
        self.input_size = input_size
        self.input_channels = input_channels
        self.latent_dim = latent_dim

        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

    def _build_encoder(self):
        encoder_layers = []
        encoder_layers.append(
            nn.Linear(self.input_size, 2 ** int(np.log2(self.input_size)))
        )
        in_channels = self.input_channels
        for i in range(self.n_layers):
            encoder_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, in_channels * 2, 3, padding='same'),
                    nn.MaxPool1d(2),
                    nn.ReLU(),
                )
            )
            in_channels = in_channels * 2

        encoder_layers.append(
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    in_channels
                    * 2 ** (int(np.log2(self.input_size)) - self.n_layers),
                    self.latent_dim,
                ),
            )
        )
        return nn.Sequential(*encoder_layers)

    def _build_decoder(self):
        out_channels = 2**self.n_layers
        decoder_layers = []
        decoder_layers.append(
            nn.Sequential(
                nn.Linear(
                    self.latent_dim,
                    out_channels
                    * 2 ** (int(np.log2(self.input_size)) - self.n_layers),
                ),
                nn.Unflatten(
                    dim=1,
                    unflattened_size=[
                        out_channels,
                        2 ** (int(np.log2(self.input_size)) - self.n_layers),
                    ],
                ),
            )
        )
        for i in range(self.n_layers):
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        out_channels,
                        out_channels // 2,
                        3,
                        stride=2,
                        output_padding=1,
                        padding=1,
                    ),
                    nn.ReLU(),
                )
            )
            out_channels = out_channels // 2

        decoder_layers.append(
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    out_channels * 2 ** int(np.log2(self.input_size)),
                    self.input_size,
                ),
            )
        )
        return nn.Sequential(*decoder_layers)

    def forward(self, batch):
        x = batch
        z = self.encoder(x)
        return z

    def training_step(self, batch, batch_idx):
        x = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


class AutoEncoder(pl.LightningModule):
    def __init__(
        self,
        window_size,
        encoder_hidden=[128, 64, 32],
        decoder_hidden=[32, 64, 128],
        latent_dim=16,
        learning_rate=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.window_size = window_size
        self.encoder_hidden = encoder_hidden
        self.decoder_hidden = decoder_hidden
        self.latent_dim = latent_dim

        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

    def _build_encoder(self):
        in_features = self.window_size
        self.encoder_layers = []
        for h_dim in self.encoder_hidden:
            self.encoder_layers.append(
                nn.Sequential(
                    nn.Linear(in_features, h_dim, bias=True),
                    nn.LeakyReLU(),
                )
            )
            in_features = h_dim
        self.encoder_layers.append(nn.Linear(in_features, self.latent_dim))
        return nn.Sequential(*self.encoder_layers)

    def _build_decoder(self):
        self.decoder_layers = []
        out_features = self.latent_dim
        for d_dim in self.decoder_hidden:
            self.decoder_layers.append(
                nn.Sequential(
                    nn.Linear(out_features, d_dim, bias=True),
                    nn.LeakyReLU(),
                )
            )
            out_features = d_dim

        self.decoder_layers.append(nn.Linear(out_features, self.window_size))
        return nn.Sequential(*self.decoder_layers)

    def forward(self, batch):
        x = batch
        z = self.encoder(x)
        return z

    def decode(self, batch):
        x = batch
        z = self.decoder(x)
        return z

    def training_step(self, batch, batch_idx):
        x = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
