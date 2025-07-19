import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self._build_pe(max_len)

    def _build_pe(self, max_len):
        pe = torch.zeros(max_len, self.d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, max_len, d_model]

    def forward(self, x):
      seq_len = x.size(1)
      if seq_len > self.pe.size(1):
          self._build_pe(seq_len)
      pe = self.pe[:, :seq_len].to(x.device)  # ← move PE to the same device as input
      return x + pe


class StemTransformerClassifier(nn.Module):
    def __init__(self, feature_dim=9, d_model=64, nhead=4, num_layers=3, num_classes=4):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, padding_mask):
        """
        x: [batch_size, seq_len, feature_dim]
        padding_mask: [batch_size, seq_len] with True for PAD
        """
        x = self.input_proj(x)                     # [B, T, d_model]
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)                      # [T, B, d_model]
        out = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        out = out.transpose(0, 1)                  # [B, T, d_model]
        out = out.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        pooled = out.sum(dim=1) / (~padding_mask).sum(dim=1, keepdim=True)  # mean over non-pad
        logits = self.classifier(pooled)
        return logits

