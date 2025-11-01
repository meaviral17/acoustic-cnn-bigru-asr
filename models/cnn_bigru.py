import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1)
    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w

class CNNBiGRU(nn.Module):
    def __init__(self, n_mels=80, vocab_size=34, cnn_channels=128, num_gru=3, gru_hidden=512, use_se=True, dropout=0.1):
        super().__init__()
        C = cnn_channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, C, (5,5), stride=(1,2), padding=(2,2)),
            nn.BatchNorm2d(C), nn.ReLU(),
            nn.Conv2d(C, C, (3,3), stride=(1,2), padding=1),
            nn.BatchNorm2d(C), nn.ReLU(),
        )
        self.se = SEBlock(C) if use_se else nn.Identity()
        self.proj = nn.Linear(C, gru_hidden)
        self.grus = nn.ModuleList([nn.GRU(input_size=gru_hidden, hidden_size=gru_hidden//2, num_layers=1, batch_first=True, bidirectional=True) for _ in range(num_gru)])
        self.do = nn.Dropout(dropout)
        self.head = nn.Linear(gru_hidden, vocab_size)
    def forward(self, x, x_lens):
        x = self.conv(x)
        x = self.se(x)
        x = x.mean(dim=2)
        x = x.transpose(1,2)
        x = self.proj(x)
        for gru in self.grus:
            x, _ = gru(x)
            x = self.do(x)
        logits = self.head(x)
        out_lens = (x_lens // 2) // 2
        return logits, out_lens
