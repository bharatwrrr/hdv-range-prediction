import torch
import torch.nn as nn
    

class LSTMModel(nn.Module):
    """
    LSTM over past_seq + static features.

    Args:
        past_dim   (int): number of features in past_seq
        static_dim (int): number of static features
        hidden_dim (int): hidden size of LSTM
        n_layers   (int): number of LSTM layers
        dropout    (float): dropout rate
    """
    def __init__(self,
                 past_dim: int,
                 static_dim: int,
                 hidden_dim: int = 64,
                 n_layers: int = 2,
                 dropout: float = 0.1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=past_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )
        # after LSTM we will concat static_dim
        self.fc = nn.Linear(hidden_dim + static_dim, 2)
        self.dropout = nn.Dropout(dropout)
        self.softplus = nn.Softplus()

    def forward(self,
                past_seq: torch.Tensor,
                future_seq: torch.Tensor,
                static_feat: torch.Tensor) -> torch.Tensor:
        """
        past_seq:    (B, Tp, Dp)
        static_feat: (B, Ds)
        returns:     (B,) regression output
        """
        out, _ = self.lstm(past_seq)    # (B, Tp, hidden_dim)
        last = out[:, -1, :]            # (B, hidden_dim)
        # concat static features
        cat = torch.cat([last, static_feat], dim=1)  # (B, hidden_dim+Ds)
        # print(f"[DEBUG] cat: {cat}")
        dropped = self.dropout(cat)
        out = self.fc(dropped)
        # print(f"[DEBUG] out: {out}")
        mu = out[:, 0]
        # print(f"[DEBUG] mu: {mu}")
        sigma = self.softplus(out[:, 1]) + 1e-6
        return mu, sigma
