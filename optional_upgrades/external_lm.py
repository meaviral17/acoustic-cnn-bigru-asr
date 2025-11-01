class ExternalLM:
    def __init__(self, path=None): self.path = path
    def score(self, sequence): return 0.0