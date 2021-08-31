import math
from numpy.lib.type_check import imag
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import torch
from torch import Tensor, LongTensor
from einops.layers.torch import Rearrange
from torch.nn.modules import dropout
from resnet import ResNetSkriptGen, BasicBlock

import sentence_tokens as st

class CorpusDecoder(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            max_length_encoding: int, # equals hidden size
            num_heads: int = 8,
            num_layers: int = 6,
            hidden_size: int = 256, # context vector (hidden state)
            dropout: float = 0.1,
            device: str = 'cpu',
            ) -> None:
        super().__init__()

        self.dropout = dropout
        self.device = device
        self.max_length_encoding = max_length_encoding
        self.hidden_size = hidden_size

        assert(self.hidden_size % num_heads == 0)

        self.token_embedding = nn.Embedding(vocab_size, self.hidden_size).to(device)
        self.scale = torch.sqrt(torch.FloatTensor([self.hidden_size])).to(device)
        self.pos_embedding = nn.Embedding(max_length_encoding, self.hidden_size).to(device)

        decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_size, nhead=num_heads, dropout=dropout).to(device)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers).to(device)
        self.fc_out = nn.Linear(self.hidden_size, vocab_size).to(device)

        print("\nCreated CorpusDecoder with:\n" + 
            "Amount of heads: " + str(num_heads) + "\n" +
            "Amount of layers: " + str(num_layers) + "\n" +
            "Hidden size: " + str(self.hidden_size) + "\n" +
            "Dropout: " + str(dropout) + "\n"
            )


    def forward(self, target: Tensor, memory: Tensor, target_mask: Tensor = None):
        #Memory has to get the shape [Batch size, Sequence length, Feature number]
        memory = memory[..., None].permute(0, 2, 1)
        memory = torch.tile(memory, (1, target.shape[1], 1))
        
        x = self.token_embedding(target) * self.scale

        batch_size = x.shape[0]
        length = x.shape[1]
        
        pos_emb_tensor = torch.arange(0, length).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        x += self.pos_embedding(pos_emb_tensor)
        x = self.decoder(tgt=x, memory=memory, tgt_mask=target_mask)
        x = self.fc_out(x)
        
        return x

class FloatListDecoder(nn.Module):
    def __init__(
            self,
            max_len_floats: int,
            num_heads: int = 8,
            num_layers: int = 6,
            hidden_size: int = 256, # context vector (hidden state)
            dropout: float = 0.1,
            device: str = 'cpu',
            ) -> None:
        super().__init__()

        self.dropout = dropout
        self.device = device
        self.hidden_size = hidden_size

        assert(hidden_size % num_heads == 0)

        self.pos_embedding = nn.Embedding(max_len_floats, self.hidden_size).to(device)

        self.scale = torch.sqrt(torch.FloatTensor([self.hidden_size])).to(device)

        decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_size, nhead=num_heads, dropout=dropout).to(device)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers).to(device)
        self.fc_out = nn.Linear(self.hidden_size, 1).to(device)

        print("\nCreated FloatListDecoder with:\n" + 
            "Amount of heads: " + str(num_heads) + "\n" +
            "Amount of layers: " + str(num_layers) + "\n" +
            "Hidden size: " + str(self.hidden_size) + "\n" +
            "Dropout: " + str(dropout) + "\n"
            )

    def token_embedding(self, target: Tensor) -> Tensor:
        print("before embedding", target.shape)
        return target

    def forward(self, target: Tensor, memory: Tensor, target_mask: Tensor = None):
        memory = memory.permute(0, 2, 1)
        memory = torch.tile(memory, (1, target.shape[1], 1))

        x = torch.tile(target[..., None], (1, 1, self.hidden_size)).float() * self.scale
        #x = self.token_embedding(target[..., None]) * self.scale
        #x = torch.tile(x, (1, 1, self.hidden_size)).float() * self.scale

        batch_size = x.shape[0]
        length = x.shape[1]

        # positional encoding
        pos_emb_tensor = torch.arange(0, length).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        x += self.pos_embedding(pos_emb_tensor)
        
        x = self.decoder(tgt=x, memory=memory, tgt_mask=target_mask)
        x = self.fc_out(x)

        return x
    

class SkriptGen(nn.Module):
    def __init__(
        self,
        encoding: dict,
        max_len_encoding: int,
        max_len_floats: int,
        num_heads: int = 8,
        device='cpu',
    ) -> None:
        super().__init__()

        self.encoding = encoding
        self.max_len_encoding = max_len_encoding
        self.device = device

        '''
        if max_len_encoding % num_heads != 0:
            hidden_size = (max_len_encoding // num_heads) * num_heads + num_heads
        else:
            hidden_size = (max_len_encoding // num_heads) * num_heads
        '''
        hidden_size = 512

        self.encoder = ResNetSkriptGen(BasicBlock, [1, 2, 2], hidden_size, device=device)

        self.corpus_decoder = CorpusDecoder(
            vocab_size=len(encoding),
            max_length_encoding=self.max_len_encoding,
            hidden_size=hidden_size,
            num_layers=8,
            device=device)
        self.numbers_decoder = FloatListDecoder(
            max_len_floats=max_len_floats,
            hidden_size=hidden_size * 2,
            num_layers=8,
            device=device)

        self.hidden_size = hidden_size

    def init_weights(self) -> None:
        self.encoder.init_weights()

        # from https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb
        def xavier_init(model):
            if hasattr(model, 'weight') and model.weight.dim() > 1:
                nn.init.xavier_uniform_(model.weight.data)

        self.corpus_decoder.apply(xavier_init)
        self.numbers_decoder.apply(xavier_init)

    def make_trg_mask(self, trg):
        #trg = [batch size, trg len]
        trg_pad_mask = (trg != self.encoding[st.token_PAD]).unsqueeze(1).unsqueeze(2)
        #trg_pad_mask = [batch size, 1, 1, trg len]
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        #trg_sub_mask = [trg len, trg len]
        trg_mask = trg_pad_mask & trg_sub_mask
        #trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            From: https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer.generate_square_subsequent_mask
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward_Image_Encoder(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def forward_Corpus_Decoder(self, target: Tensor, encoded_image: Tensor, target_mask = None) -> Tensor:
        return self.corpus_decoder(target, encoded_image, target_mask)

    def forward_Numbers_Decoder(self, target: Tensor, encoded_image: Tensor, target_mask = None) -> Tensor:
        return self.numbers_decoder(target, encoded_image, target_mask)

    def forward(self, image: Tensor, target_script: Tensor, target_numbers: Tensor, enc_nbrs_on_target_encoding: bool = True) -> Tensor:
        encoded_image = self.encoder(image)

        #fill up target_numbers
        #fill_amount = self.max_len_floats - target_numbers.shape[1]
        #fill_tensor = torch.zeros((target_numbers.shape[0], fill_amount))
        #target_numbers = torch.cat((target_numbers, fill_tensor), dim=1)

        #TODO encoded image mit encoded script für numbers decoder konkatenieren

        encoded_script = self.corpus_decoder(target_script, encoded_image, target_mask=self.generate_square_subsequent_mask(len(target_script)).to(self.device))
        
        # add 0 feature to the encoded image
        image_memory = encoded_image[..., None]
        image_memory = torch.cat([image_memory, torch.zeros(image_memory.shape).to(self.device)], dim=2)

        if enc_nbrs_on_target_encoding:
            # add 1 feature to the target script
            script_memory = target_script[..., None]
            script_memory = torch.cat([script_memory, torch.ones(script_memory.shape).to(self.device)], dim=2)
        else:
            # add 1 feature to the script decoder output
            script_memory = encoded_script.argmax(2)[..., None]
            script_memory = torch.cat([script_memory, torch.ones(script_memory.shape).to(self.device)], dim=2)

        # concatenate
        memory_enc_nbrs = torch.cat([
            image_memory, 
            script_memory, 
            torch.zeros((image_memory.shape[0], 2* self.hidden_size - image_memory.shape[1] - script_memory.shape[1], image_memory.shape[2])).to(self.device)], dim=1)
        
        script_floats = self.numbers_decoder(target_numbers, memory_enc_nbrs, target_mask=self.generate_square_subsequent_mask(len(target_numbers)).to(self.device))
        
        return encoded_script, script_floats
