import torch, random, numpy as np, torchaudio
from torch.utils.data import Dataset
from torchaudio.datasets import LIBRISPEECH
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, FrequencyMasking, TimeMasking, PCEN

class LibriCTCDataset(Dataset):
    def __init__(self, root, url, tokenizer, sample_rate=16000, n_mels=80, pcen=False, specaug=True, tf_mixup=False, max_len_sec=None):
        self.ds = LIBRISPEECH(root=root, url=url, download=True)
        self.encode, self.decode, self.stoi, self.itos, self.blank = tokenizer
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.pcen = pcen
        self.specaug = specaug
        self.tf_mixup = tf_mixup
        self.max_len_sec = max_len_sec
        self.resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=sample_rate)
        self.melspec = MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels, n_fft=400, hop_length=160, f_min=20, f_max=7600)
        self.db = AmplitudeToDB(stype="power")
        self.pcen_layer = PCEN(R=0.5, s=0.025, alpha=0.98, delta=2.0, eps=1e-6, trainable=True) if pcen else None
        self.fmask = FrequencyMasking(freq_mask_param=18)
        self.tmask = TimeMasking(time_mask_param=50)
    def __len__(self): return len(self.ds)
    def _feats(self, wav):
        wav = self.resample(wav)
        spec = self.melspec(wav)
        spec = self.pcen_layer(spec) if self.pcen_layer else self.db(spec)
        spec = (spec - spec.mean()) / (spec.std() + 1e-5)
        return spec
    def _apply_specaug(self, spec):
        if not self.specaug: return spec
        spec = self.fmask(spec)
        spec = self.tmask(spec)
        return spec
    def _maybe_tf_mixup(self, spec1, label1, spec2, label2, alpha=0.2):
        if not self.tf_mixup: return spec1, label1
        lam = np.random.beta(alpha, alpha)
        T = min(spec1.shape[-1], spec2.shape[-1])
        spec = lam * spec1[:, :T] + (1 - lam) * spec2[:, :T]
        mixed_label = label1 + "|" + label2
        return spec, mixed_label
    def __getitem__(self, idx):
        wav, sr, transcript, *_ = self.ds[idx]
        wav = wav.squeeze(0)
        if self.max_len_sec:
            max_len = int(self.sample_rate * self.max_len_sec)
            if wav.shape[-1] > max_len: wav = wav[:max_len]
        spec = self._feats(wav)
        if self.tf_mixup and random.random() < 0.1:
            j = random.randint(0, len(self.ds)-1)
            wav2, _, transcript2, *_ = self.ds[j]
            spec2 = self._feats(wav2.squeeze(0))
            spec, transcript = self._maybe_tf_mixup(spec, transcript.lower(), spec2, transcript2.lower())
        spec = self._apply_specaug(spec)
        y = self.encode(transcript.lower())
        return spec, y

def ctc_collate(batch):
    specs, labels = zip(*batch)
    maxT = max(s.shape[-1] for s in specs)
    n_mels = specs[0].shape[0]
    B = len(specs)
    x = torch.zeros(B, 1, n_mels, maxT)
    x_lens = torch.tensor([s.shape[-1] for s in specs], dtype=torch.long)
    for i, s in enumerate(specs):
        x[i, 0, :, :s.shape[-1]] = s
    ys = torch.cat(labels)
    y_lens = torch.tensor([len(y) for y in labels], dtype=torch.long)
    return x, x_lens, ys, y_lens
