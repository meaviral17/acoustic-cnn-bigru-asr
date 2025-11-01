import math

class ExternalLM:
    def __init__(self, path=None, alpha=0.5, beta=1.0):
        self.alpha = alpha  # LM weight
        self.beta = beta    # length bonus
        self.path = path
        self.model = None
        try:
            import kenlm
            if path:
                self.model = kenlm.Model(path)
        except Exception as e:
            print("KenLM unavailable:", e)
            self.model = None

    def score(self, sequence):
        if self.model is None:
            return 0.0
        s = sequence.replace("|", " ")
        return self.alpha * self.model.score(s, bos=False, eos=False) + self.beta * len(s.split())
