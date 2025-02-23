import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def _init__(self, d_model: int, vocab_size:int):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.module):

    def __init__(self, d_model:int, seq_len: int, dropout:float)-> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        #Create a matirx of sape (seq_len, d_model)
        pe = torch 