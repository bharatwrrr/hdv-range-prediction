import torch
import torch.nn as nn


class LSTMModel:
    """
    LSTM model for time series prediction.
    """

    def __init__(self, input_dim, hidden_dim, n_layers, dropout):
        """
        Initialize the LSTM model.

        Args:
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Dimension of the hidden state.
            n_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate.
        """

        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # Output layer for regression
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.hidden = None
        self.init_hidden()

    def init_hidden(self, batch_size=1):
        """
        Initialize the hidden state and cell state for LSTM.

        Args:
            batch_size (int): Batch size for the hidden state.
        """
        self.hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim),
                       torch.zeros(self.n_layers, batch_size, self.hidden_dim))
        
    def forward(self, x):
        """
        Forward pass through the LSTM model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        lstm_out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(lstm_out)
        return out.squeeze(1)
    

