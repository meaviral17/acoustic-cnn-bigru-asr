import torch, random, numpy as np, time

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def human_time(s):
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h {m:02d}m {s:02d}s"

def build_text_tokenizer(vocab):
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    encode = lambda s: [stoi[c] for c in s.lower() if c in stoi]
    decode = lambda l: "".join([itos[i] for i in l if i in itos])
    blank = stoi["<blk>"]
    return encode, decode, stoi, itos, blank
