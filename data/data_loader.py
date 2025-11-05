import os, torch, torchaudio
from torch.utils.data import Dataset
from functools import lru_cache

class LibriCTCDataset(Dataset):
    def __init__(self, root, subset, tokenizer, pcen=False, tf_mixup=False, specaug=False, max_len_sec=None):
        if "train" in subset:
            root = os.path.join(root, "Train_data", "LibriSpeech")
        elif "test" in subset or "dev" in subset:
            root = os.path.join(root, "Test_data", "LibriSpeech")

        dataset_path = os.path.join(root, subset)
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Could not find {subset} inside {root}")

        print(f"Using LibriSpeech dataset from: {dataset_path}")
        self.data = torchaudio.datasets.LIBRISPEECH(os.path.dirname(root), url=subset, download=False)
        self.tokenizer = tokenizer
        self.pcen = pcen
        self.n_mels = 80
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=400, hop_length=160, n_mels=self.n_mels
        )

    def __len__(self):
        return len(self.data)

    @lru_cache(maxsize=128)
    def _cached_item(self, idx):
        wav, sr, text, speaker_id, chapter_id, utterance_id = self.data[idx]
        feat = self.melspec(wav).squeeze(0).transpose(0, 1)
        if self.pcen:
            feat = torch.log1p(feat)
        feat_len = torch.tensor(feat.size(0))
        ys = torch.tensor([self.tokenizer[2][c] for c in text.lower() if c in self.tokenizer[2]])
        y_len = torch.tensor(len(ys))
        return feat, feat_len, ys, y_len

    def __getitem__(self, idx):
        return self._cached_item(idx)

def ctc_collate(batch):
    feats, feat_lens, ys, y_lens = zip(*batch)
    feats = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)
    ys = torch.cat(ys)
    return feats, torch.stack(feat_lens), ys, torch.stack(y_lens)
