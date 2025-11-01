class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n:p.data.clone() for n,p in model.named_parameters() if p.requires_grad}
    def update(self, model):
        for n,p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = self.decay*self.shadow[n] + (1-self.decay)*p.data
    def apply_to(self, model):
        self.backup = {}
        for n,p in model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data.clone()
                p.data = self.shadow[n]
    def restore(self, model):
        for n,p in model.named_parameters():
            if p.requires_grad:
                p.data = self.backup[n]
