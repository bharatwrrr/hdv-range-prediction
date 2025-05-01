import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from torch.nn import functional as F
from ..utils import PositionalEncoding


class PastEncoderBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_heads: int = 4, 
                 n_layers: int = 2, dropout: float = 0.0, max_len: int = 1000):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim, max_len=max_len)  # fixed here

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        """
        x: [B, T, input_dim]
        """
        x = self.input_proj(x)            # [B, T, hidden_dim]
        x = self.pos_enc(x)               # add positional encoding
        out = self.transformer(x)         # [B, T, hidden_dim]
        return out.mean(dim=1)            # [B, hidden_dim]
    
class FutureEncoderBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim: int, road_type_dim: int = 15, n_heads: int = 4, 
                 n_layers: int = 2, dropout: float =0.0, max_len: int = 1000):
        super().__init__()
        self.road_type_dim = road_type_dim
        self.input_phys_dim = input_dim - road_type_dim

        # Project continuous features (e.g., theta, alt, Ta, distance_to_end)
        self.phys_proj = nn.Linear(self.input_phys_dim, hidden_dim)

        # Encode road type one-hot vector
        self.road_type_encoder = nn.Sequential(
            nn.Linear(road_type_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.pos_enc = PositionalEncoding(hidden_dim, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        """
        x: [B, T, input_dim] → input_dim = cont + one-hot
        """
        # Split input
        x_phys = x[:, :, :-self.road_type_dim]        # [B, T, cont_dim]
        x_road = x[:, :, -self.road_type_dim:]        # [B, T, road_type_dim]

        # Encode both
        x_phys = self.phys_proj(x_phys)               # [B, T, H]
        x_road = self.road_type_encoder(x_road)       # [B, T, H]

        # Combine (you can also use addition here if dimensions match semantically)
        x_combined = x_phys + x_road                  # [B, T, H]

        # Positional encoding + transformer
        x_pos = self.pos_enc(x_combined)
        out = self.transformer(x_pos)

        return out.mean(dim=1)                        # [B, H]
    

class TransformerSOCDropPredictor(nn.Module):
    def __init__(self, past_input_dim: int, future_input_dim: int, static_dim: int, 
                 past_hidden_dim: int, 
                 future_hidden_dim: int, 
                 fused_hidden_dim: int):
        super().__init__()

        self.past_encoder = PastEncoderBlock(past_input_dim, hidden_dim=past_hidden_dim)
        self.future_encoder = FutureEncoderBlock(future_input_dim, hidden_dim=future_hidden_dim)

        self.fusion = nn.Sequential(
            nn.Linear(past_hidden_dim + future_hidden_dim + static_dim, fused_hidden_dim),
            nn.ReLU(),
            nn.Linear(fused_hidden_dim, fused_hidden_dim),
            nn.ReLU(),
            nn.Linear(fused_hidden_dim, fused_hidden_dim // 2),
            nn.ReLU()
        )

        self.mu_head = nn.Linear(fused_hidden_dim // 2, 1)
        self.logvar_head = nn.Linear(fused_hidden_dim // 2, 1)

    def forward(self, past_seq, future_seq, static_feat):
        past_vec = self.past_encoder(past_seq)       # [B, H]
        future_vec = self.future_encoder(future_seq) # [B, H]

        x = torch.cat([past_vec, future_vec, static_feat], dim=1)  # [B, 2H + static]
        fused = self.fusion(x)

        mu = self.mu_head(fused).squeeze(-1)         # [B]
        sigma = torch.exp(0.5 * self.logvar_head(fused).squeeze(-1))  # [B]

        return mu, sigma

