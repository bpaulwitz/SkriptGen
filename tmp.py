import math
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from einops.layers.torch import Rearrange

'''
import fast_transformers.transformers as ftf
from fast_transformers.attention import LinearAttention
from fast_transformers.masking import FullMask, LengthMask
'''

#Transformer from https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x



def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, device, dropout = 0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model).to(device)
        self.v_linear = nn.Linear(d_model, d_model).to(device)
        self.k_linear = nn.Linear(d_model, d_model).to(device)
        self.dropout = nn.Dropout(dropout).to(device)
        self.out = nn.Linear(d_model, d_model).to(device)

    def forward(self, q, k, v, mask=None):

        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)# calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)

        output = self.out(concat)

        return output

class Norm(nn.Module):
    def __init__(self, d_model, device, eps = 1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size)).to(device)
        self.bias = nn.Parameter(torch.zeros(self.size)).to(device)
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class FeedForward(nn.Module):
    def __init__(self, d_model, device, d_ff=2048, dropout = 0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff).to(device)
        self.dropout = nn.Dropout(dropout).to(device)
        self.linear_2 = nn.Linear(d_ff, d_model).to(device)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout, device):
        super().__init__()
        self.norm_1 = Norm(d_model, device)
        self.norm_2 = Norm(d_model, device)

        self.attn = MultiHeadAttention(heads, d_model, device)
        self.ff = FeedForward(d_model, device)
        self.dropout_1 = nn.Dropout(dropout).to(device)
        self.dropout_2 = nn.Dropout(dropout).to(device)

    def forward(self, x):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,None))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout, device):
        super().__init__()
        self.norm_1 = Norm(d_model, device)
        self.norm_2 = Norm(d_model, device)
        self.norm_3 = Norm(d_model, device)

        self.dropout_1 = nn.Dropout(dropout).to(device)
        self.dropout_2 = nn.Dropout(dropout).to(device)
        self.dropout_3 = nn.Dropout(dropout).to(device)


        self.attn_1 = MultiHeadAttention(heads, d_model, device)
        self.attn_2 = MultiHeadAttention(heads, d_model, device)
        self.ff = FeedForward(d_model, device)

    def forward(self, x, e_outputs, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs,
        None))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, num_layers, heads, dropout, device):
        super().__init__()
        self.pe = PositionalEncoding(d_model)
        self.layers = [EncoderLayer(d_model, heads, dropout, device) for _ in range(num_layers)]
        self.norm = Norm(d_model, device)
        self.num_layers = num_layers

    def forward(self, src):
        x = self.pe(src)
        for l in self.layers:
            x = l(x)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, d_model, num_layers, heads, dropout, device):
        super().__init__()
        self.num_layers = num_layers
        self.pe = PositionalEncoding(d_model)
        self.layers = [DecoderLayer(d_model, heads, dropout, device) for _ in range(num_layers)]
        self.norm = Norm(d_model, device)
    def forward(self, trg, e_outputs, trg_mask):
        x = self.pe(trg)
        for l in self.layers:
            x = l(x, e_outputs, trg_mask)
        return self.norm(x)

class TransformerV2(nn.Module):
    def __init__(self, d_model: int, num_layers: int, num_heads: int, output_size: int,
            dropout: float, img_dim: int, patch_dim: int, num_channels: int, batch_len: int, device):
        super().__init__()
        self.encoder = Encoder(d_model, num_layers, num_heads, dropout, device)
        self.decoder = Decoder(d_model, num_layers, num_heads, dropout, device)
        self.patch_dim = patch_dim
        self.img_dim = img_dim
        self.flatten_dim = patch_dim * patch_dim * num_channels
        self.linear_encoding_enc = nn.Linear(self.flatten_dim, d_model).to(device)
        self.linear_encoding_dec = nn.Linear(batch_len, d_model).to(device)
        self.out = nn.Linear(d_model, output_size).to(device)

    def forward(self, src, trg, trg_mask):
        n, c, h, w = src.shape
        x = (
            src.unfold(2, self.patch_dim, self.patch_dim)
            .unfold(3, self.patch_dim, self.patch_dim)
            .contiguous()
        )
        x = x.view(n, c, -1, self.patch_dim ** 2)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(x.size(0), -1, self.flatten_dim)
        x = self.linear_encoding_enc(x)
        e_outputs = self.encoder(x)

        y = self.linear_encoding_dec(trg)
        d_output = self.decoder(y, e_outputs, trg_mask)
        output = self.out(d_output)

        return output

class TransformerV3(nn.Module):
    def __init__(self, d_model: int, num_layers: int, num_heads: int, output_size: int,
            dropout: float, img_dim: int, patch_dim: int, num_channels: int, device):
        super().__init__()
        self.encoder = Encoder(d_model, num_layers, num_heads, dropout, device)
        self.decoder = Decoder(d_model, num_layers, num_heads, dropout, device)
        self.patch_dim = patch_dim
        self.img_dim = img_dim
        self.flatten_dim = patch_dim * patch_dim * num_channels
        self.linear_encoding_enc = nn.Linear(self.flatten_dim, d_model).to(device)
        self.linear_encoding_dec = nn.Linear(output_size, d_model).to(device)
        self.out = nn.Linear(d_model, output_size).to(device)

    #from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, trg, trg_mask):
        n, c, h, w = src.shape
        x = (
            src.unfold(2, self.patch_dim, self.patch_dim)
            .unfold(3, self.patch_dim, self.patch_dim)
            .contiguous()
        )
        x = x.view(n, c, -1, self.patch_dim ** 2)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(x.size(0), -1, self.flatten_dim)
        x = self.linear_encoding_enc(x)
        e_outputs = self.encoder(x)

        y = self.linear_encoding_dec(trg)
        d_output = self.decoder(y, e_outputs, trg_mask)
        output = self.out(d_output)

        return output

'''------------------------------------------------------------------------------------------------------------VERSION 2------------------------------------------------------------------------------------------------'''
#based on https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        #x = [batch size, seq len, hid dim]

        x = self.dropout(torch.relu(self.fc_1(x)))

        #x = [batch size, seq len, pf dim]

        x = self.fc_2(x)

        #x = [batch size, seq len, hid dim]

        return x

#based on https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim).to(device)
        self.fc_k = nn.Linear(hid_dim, hid_dim).to(device)
        self.fc_v = nn.Linear(hid_dim, hid_dim).to(device)

        self.fc_o = nn.Linear(hid_dim, hid_dim).to(device)

        self.dropout = nn.Dropout(dropout).to(device)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask = None):

        batch_size = query.shape[0]

        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        #energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim = -1)

        #attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V) #->TODO herausfinden, warum Dropout direkt auf Attention angewandt wird!

        #x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        #x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        #x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        #x = [batch size, query len, hid dim]

        return x, attention

#based on https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb
class EncoderV2Layer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):

        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len]

        #self attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        #src = [batch size, src len, hid dim]

        #positionwise feedforward
        _src = self.positionwise_feedforward(src)

        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        #src = [batch size, src len, hid dim]

        return src

#based on https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb
class EncoderV2(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length):
        super().__init__()

        self.device = device

        #self.token_embedding = nn.Embedding(input_dim, hid_dim)
        #positional embedding is done in the ViT module
        #self.positional_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([EncoderV2Layer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):

        #src = [batch size, src len]
        #src_mask = [batch size, 1, 1, src len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        #pos = [batch size, src len]

        #src = self.dropout((self.token_embedding(src) * self.scale) + self.positional_embedding(pos))
        src = self.dropout(src * self.scale)

        #src = [batch size, src len, hid dim]

        for layer in self.layers:
            src = layer(src, src_mask)

        #src = [batch size, src len, hid dim]

        return src

#based on https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb
class DecoderLayerV2(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim).to(device)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim).to(device)
        self.ff_layer_norm = nn.LayerNorm(hid_dim).to(device)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout).to(device)
        self.dropout = nn.Dropout(dropout).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):

        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]

        #self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        #dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        #trg = [batch size, trg len, hid dim]

        #encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)

        #dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        #trg = [batch size, trg len, hid dim]

        #positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        #dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]

        return trg, attention

#based on https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb
class DecoderV2(nn.Module):
    def __init__(self,
                 output_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length):
        super().__init__()

        self.device = device

        self.token_embedding = nn.Embedding(output_dim, hid_dim).to(device)
        self.positional_embedding = nn.Embedding(max_length, hid_dim).to(device)

        self.layers = nn.ModuleList([DecoderLayerV2(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim).to(device)

        self.dropout = nn.Dropout(dropout).to(device)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):

        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        #pos = [batch size, trg len]

        trg = self.dropout((self.token_embedding(trg) * self.scale) + self.positional_embedding(pos))

        #trg = [batch size, trg len, hid dim]

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]

        output = self.fc_out(trg)

        #output = [batch size, trg len, output dim]

        return output, attention

    #based on https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb
class DecoderV2Float(nn.Module):
    def __init__(self,
                 output_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length):
        super().__init__()

        self.device = device

        self.hid_dim = hid_dim

        self.positional_embedding = nn.Embedding(max_length, hid_dim).to(device)

        self.layers = nn.ModuleList([DecoderLayerV2(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim).to(device)

        self.dropout = nn.Dropout(dropout).to(device)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):

        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        #pos = [batch size, trg len]

        trg = torch.tile(trg[..., None], (1, 1, self.hid_dim))
        trg = self.dropout((trg * self.scale) + self.positional_embedding(pos))

        #trg = [batch size, trg len, hid dim]
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]

        output = self.fc_out(trg)

        #output = [batch size, trg len, output dim]

        return output, attention