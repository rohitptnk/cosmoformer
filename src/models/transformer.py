from __future__ import annotations

import torch
import torch.nn as nn


# Positional Embedding

class LearnablePositionalEncoding(nn.Module):
    positions: torch.Tensor

    def __init__(self, d_model, seq_len):
        super().__init__()
        # Learnable positional embeddings
        self.pos_embedding = nn.Embedding(seq_len, d_model)
        self.register_buffer('positions', torch.arange(seq_len).unsqueeze(0))  # shape (1, seq_len)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        positions = self.positions[:, :x.size(1)]
        pos_emb = self.pos_embedding(positions)  # (1, seq_len, d_model)
        return x + pos_emb
    

# Transformer Block

class Transformer1DAutoencoder(nn.Module):
    def __init__(
        self, 
        seq_len: int, 
        d_model: int, 
        n_heads: int, 
        n_layers: int, 
        d_ff: int, 
        dropout: float,
        predict_variance: bool=True,
    ):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.seq_len = seq_len
        self.d_model = d_model
        self.predict_variance = predict_variance

        # input embedding (4D scalar → d_model)
        self.input_proj = nn.Linear(4, d_model)

        # positional encoding
        self.pos_encoder = LearnablePositionalEncoding(d_model, seq_len)

        # encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,  # batch_first=True makes input shape (batch, seq_len, d_model)
            activation="gelu",
            norm_first=True, # pre-LN
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=n_layers,
        )

        # output heads
        self.freq1_mean_head = nn.Linear(d_model, 1)
        self.freq1_logvar_head = nn.Linear(d_model, 1) if predict_variance else None

        self.freq2_mean_head = nn.Linear(d_model, 1)
        self.freq2_logvar_head = nn.Linear(d_model, 1) if predict_variance else None

        self.freq3_mean_head = nn.Linear(d_model, 1)
        self.freq3_logvar_head = nn.Linear(d_model, 1) if predict_variance else None

        self.freq4_mean_head = nn.Linear(d_model, 1)
        self.freq4_logvar_head = nn.Linear(d_model, 1) if predict_variance else None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x1, x2, x3, x4):
        """
        x1, x2, x3, x4: (batch, seq_len)
        """
        if x1.dim() != 2 or x2.dim() != 2 or x3.dim() != 2 or x4.dim() != 2:
            raise ValueError(f"Expected (B, L), got {x1.shape}, {x2.shape}, {x3.shape}, and {x4.shape}")
        
        if x1.size(1) > self.seq_len:
            raise ValueError(
                f"Input sequence length {x1.size(1)} exceeds model seq_len {self.seq_len}"
            )

        # Stack as features: (batch, seq_len, 4)
        x = torch.stack([x1, x2, x3, x4], dim=-1)
        
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)  # add positional encoding
        x = self.encoder(x)      # Transformer Encoder

        def get_output(mean_head, logvar_head, hidden):
            mean = mean_head(hidden).squeeze(-1)
            logvar = None
            if logvar_head is not None:
                logvar = logvar_head(hidden).squeeze(-1)
            return mean, logvar

        c_mean, c_logvar = get_output(self.clean_mean_head, self.clean_logvar_head, x)
        f1_mean, f1_logvar = get_output(self.freq1_mean_head, self.freq1_logvar_head, x)
        f2_mean, f2_logvar = get_output(self.freq2_mean_head, self.freq2_logvar_head, x)
        f3_mean, f3_logvar = get_output(self.freq3_mean_head, self.freq3_logvar_head, x)
        f4_mean, f4_logvar = get_output(self.freq4_mean_head, self.freq4_logvar_head, x)
            
        return c_mean, c_logvar, f1_mean, f1_logvar, f2_mean, f2_logvar, f3_mean, f3_logvar, f4_mean, f4_logvar
