import torch
import torchaudio
from torch.utils.data import Dataset

class LibriCTCDataset(Dataset):
    def __init__(self, root, subset, tokenizer, pcen=False, tf_mixup=False, specaug=False, max_len_sec=None):
        self.data = torchaudio.datasets.LIBRISPEECH(root, url=subset, download=True)
        self.tokenizer = tokenizer
        self.pcen = pcen
        self.tf_mixup = tf_mixup
        self.specaug = specaug
        self.max_len_sec = max_len_sec
        self.n_mels = 80

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=400, hop_length=160, n_mels=self.n_mels
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        wav, sr, _, _, text, _ = self.data[idx]
        feat = self.melspec(wav).squeeze(0)
        if self.pcen:
            feat = torch.log1p(feat)
        feat_len = torch.tensor(feat.size(1))
        ys = torch.tensor([self.tokenizer[2][c] for c in text.lower() if c in self.tokenizer[2]])
        y_len = torch.tensor(len(ys))
        return feat, feat_len, ys, y_len


def ctc_collate(batch):
    feats, feat_lens, ys, y_lens = zip(*batch)
    feats = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)
    ys = torch.cat(ys)
    return feats, torch.stack(feat_lens), ys, torch.stack(y_lens)
