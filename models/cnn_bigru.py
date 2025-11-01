import torch
import torch.nn as nn

class CNNBiGRU(nn.Module):
    def __init__(self, n_mels, vocab_size, use_se=True, cnn_channels=128, num_gru=3, gru_hidden=512, dropout=0.1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.gru = nn.GRU(
            input_size=(n_mels // 2) * cnn_channels // 2,
            hidden_size=gru_hidden,
            num_layers=num_gru,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * gru_hidden, vocab_size)
        )

    def forward(self, x, x_lens):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, -1)
        out, _ = self.gru(x)
        out = self.fc(out)
        out_lens = x_lens // 2
        return out, out_lens
