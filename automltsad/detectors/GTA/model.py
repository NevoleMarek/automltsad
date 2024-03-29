import torch
import torch.nn as nn
import torch.nn.functional as F

from automltsad.detectors.GTA.attn import (
    AttentionLayer,
    FullAttention,
    ProbAttention,
)
from automltsad.detectors.GTA.decoder import Decoder, DecoderLayer
from automltsad.detectors.GTA.embed import DataEmbedding
from automltsad.detectors.GTA.encoder import ConvLayer, Encoder, EncoderLayer
from automltsad.detectors.GTA.masking import ProbMask, TriangularCausalMask


class Informer(nn.Module):
    def __init__(
        self,
        enc_in,
        dec_in,
        c_out,
        seq_len,
        label_len,
        out_len,
        factor=4,
        d_model=512,
        n_heads=8,
        e_layers=3,
        d_layers=2,
        d_ff=512,
        dropout=0.0,
        attn='prob',
        embed='fixed',
        data='ETTh',
        activation='gelu',
        device=torch.device('cuda:0'),
    ):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn

        # Encoding
        self.enc_embedding = DataEmbedding(
            enc_in, d_model, embed, data, dropout
        )
        self.dec_embedding = DataEmbedding(
            dec_in, d_model, embed, data, dropout
        )
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        Attn(False, factor, attention_dropout=dropout),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(e_layers)
            ],
            [ConvLayer(d_model) for l in range(e_layers - 1)],
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, factor, attention_dropout=dropout),
                        d_model,
                        n_heads,
                    ),
                    AttentionLayer(
                        FullAttention(
                            False, factor, attention_dropout=dropout
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
    ):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(
            dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask
        )
        dec_out = self.projection(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        return dec_out[:, -self.pred_len :, :]  # [B, L, D]
