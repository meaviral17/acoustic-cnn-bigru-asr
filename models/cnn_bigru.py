import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBiGRU(nn.Module):
    def __init__(self, n_mels=80, vocab_size=30, use_se=True, cnn_channels=128, num_gru=3, gru_hidden=512, dropout=0.1):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )

        sample_in = torch.zeros(1, 1, 200, n_mels)
        with torch.no_grad():
            out = self.cnn(sample_in)
            _, C, T, F = out.shape
            self.gru_input_size = C * F

        self.gru = nn.GRU(
            input_size=self.gru_input_size,
            hidden_size=gru_hidden,
            num_layers=num_gru,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden * 2, vocab_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_lens):
        # x: [B, T, n_mels]
        x = x.unsqueeze(1)  # [B, 1, T, n_mels]
        x = self.cnn(x)     # [B, C, T', F]
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, T', C, F]
        B, T, C, F = x.shape
        x = x.view(B, T, C * F)  # [B, T, features]
        x, _ = self.gru(x)       # [B, T, hidden*2]
        x = self.dropout(x)
        x = self.classifier(x)   # [B, T, vocab_size]
        out_lens = (x_lens // 2).int()  # because of pooling (approx.)
        return x, out_lens

