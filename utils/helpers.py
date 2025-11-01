import random, numpy as np, torch

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def human_time(s):
    m, s = divmod(int(s), 60); h, m = divmod(m, 60); return f"{h:02d}:{m:02d}:{s:02d}"

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_text_tokenizer(labels):
    stoi = {c:i for i,c in enumerate(labels)}
    itos = {i:c for i,c in enumerate(labels)}
    blank = stoi["<blk>"]
    def encode(s): return torch.tensor([stoi[c] for c in s if c in stoi], dtype=torch.long)
    def decode(ints):
        out, prev = [], None
        for i in ints:
            if i != blank and (prev is None or i != prev):
                out.append(itos[i])
            prev = i
        return "".join(out).replace("|", " ").strip()
    return encode, decode, stoi, itos, blank
