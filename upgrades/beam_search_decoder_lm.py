import torch, kenlm

class KenLMDecoder:
    def __init__(self, lm_path, alpha=0.6, beta=1.0, beam_width=10, blank_idx=0):
        self.alpha = alpha
        self.beta = beta
        self.beam_width = beam_width
        self.blank = blank_idx
        self.lm = kenlm.Model(lm_path)

    def decode(self, logits, out_lens, vocab):
        probs = torch.softmax(logits, dim=-1)
        results = []
        for i, L in enumerate(out_lens):
            beam = [("", 0.0)]
            for t in range(L):
                next_beam = {}
                for prefix, score in beam:
                    for j, p in enumerate(probs[i, t]):
                        if j == self.blank:
                            continue
                        c = vocab[j]
                        new_prefix = prefix + c
                        lm_score = self.alpha * self.lm.score(new_prefix, bos=False, eos=False)
                        total = score + torch.log(p).item() + lm_score + self.beta
                        if new_prefix not in next_beam or total > next_beam[new_prefix]:
                            next_beam[new_prefix] = total
                beam = sorted(next_beam.items(), key=lambda x: x[1], reverse=True)[:self.beam_width]
            results.append(max(beam, key=lambda x: x[1])[0])
        return results
