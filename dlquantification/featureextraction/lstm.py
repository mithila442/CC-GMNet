import torch
import torch.nn as nn
import torch.nn.functional as F

# class LSTMFeatureExtractionModule(nn.Module):
#     """
#     LSTM-based feature extractor for time-series data.
#     Input shape: [batch_size, seq_len, input_dim]
#     Output shape: [batch_size, output_size]
#     """
#     def __init__(self, input_dim, hidden_size, output_size, num_layers=1, dropout_lstm=0.0, dropout_linear=0.0):
#         super(LSTMFeatureExtractionModule, self).__init__()
#         self.output_size = output_size
#         self.hidden_size = hidden_size

#         self.lstm = nn.LSTM(
#             input_size=input_dim,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=dropout_lstm if num_layers > 1 else 0.0,
#         )
#         self.dropout1 = nn.Dropout(dropout_linear)
#         self.dropout2 = nn.Dropout(dropout_linear)
#         self.linear = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         # x: [B, T, C] where C = number of channels (input_dim)
#         lstm_out, (h_n, _) = self.lstm(x)  # h_n: [num_layers, B, hidden_size]
#         last_hidden = h_n[-1]              # [B, hidden_size]
#         out = self.dropout1(F.relu(last_hidden))
#         return self.dropout2(self.linear(out))

class LSTMFeatureExtractionModule(nn.Module):
    """
    LSTM with class token attention for time series feature extraction.
    """
    def __init__(self, input_dim, hidden_size, output_size, num_layers=1, dropout_lstm=0.1, dropout_linear=0.1):
        super(LSTMFeatureExtractionModule, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_lstm if num_layers > 1 else 0.0
        )

        # Learnable class token
        self.class_token = nn.Parameter(torch.randn(1, 1, hidden_size))  # shape: [1, 1, H]

        # Attention projection
        self.attn_proj = nn.Linear(hidden_size, 1)

        # Output projection
        self.dropout = nn.Dropout(dropout_linear)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        x: [B, T, D] - batch of time series
        """
        B = x.size(0)

        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # lstm_out: [B, T, H]

        # Repeat class token across batch
        cls_token = self.class_token.expand(B, -1, -1)  # [B, 1, H]
        tokens = torch.cat([cls_token, lstm_out], dim=1)  # [B, 1+T, H]

        # Compute attention scores
        attn_scores = self.attn_proj(tokens).squeeze(-1)  # [B, 1+T]
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # [B, 1+T, 1]

        # Weighted sum
        attended = (tokens * attn_weights).sum(dim=1)  # [B, H]

        # Projection
        out = self.dropout(F.relu(attended))
        return self.linear(out)