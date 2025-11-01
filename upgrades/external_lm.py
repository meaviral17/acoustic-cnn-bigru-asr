import os, kenlm

class ExternalLM:
    def __init__(self, path="lm/english_5gram.binary", alpha=0.6, beta=1.0):
        path = os.path.abspath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"KenLM binary not found at {path}")
        self.model = kenlm.Model(path)
        self.alpha = alpha
        self.beta = beta
        print(f"✅ External KenLM initialized: {path}")

    def score(self, text):
        return self.model.score(text, bos=True, eos=True)

    def normalize_score(self, text):
        words = text.strip().split()
        if not words:
            return 0.0
        return self.model.score(text, bos=True, eos=True) / len(words)
