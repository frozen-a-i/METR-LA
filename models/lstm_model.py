import torch
import torch.nn as nn


class PerSensorLSTM(nn.Module):
    """
    Per-sensor LSTM: each of the N sensors is processed independently
    through the same shared LSTM weights, then a FC head predicts
    T_out horizons for that sensor.

    Input:  [B, N, T_in]   e.g. [batch, 207, 12]
    Output: [B, N, T_out]  e.g. [batch, 207, 3]

    Strategy: reshape [B, N, T_in] -> [B*N, T_in, 1], run LSTM once
    across the batch, then reshape back. No extra memory overhead vs a
    single-sensor LSTM — just a larger effective batch.
    """

    def __init__(self, hidden=64, layers=2, out_size=3, drop=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=drop if layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(drop)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, out_size),
        )

    def forward(self, x):
        B, N, T = x.shape
        x = x.reshape(B * N, T, 1)      # [B*N, T_in, 1]
        out, _ = self.lstm(x)            # [B*N, T_in, hidden]
        out = self.drop(out[:, -1])      # [B*N, hidden]
        out = self.fc(out)               # [B*N, out_size]
        return out.reshape(B, N, -1)     # [B, N, out_size]
