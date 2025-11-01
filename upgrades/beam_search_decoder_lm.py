import torch
import math

class KenLMDecoder:
    def __init__(self, lm, alpha=0.6, beta=1.0, beam_width=10):
        self.lm = lm
        self.alpha = alpha
        self.beta = beta
        self.beam_width = beam_width

    def decode(self, logits, out_lens, blank_idx=0):
        B, T, V = logits.size()
        probs = torch.nn.functional.log_softmax(logits, dim=-1)
        decoded = []

        for b in range(B):
            T_b = out_lens[b]
            beam = [("", 0.0)]
            for t in range(T_b):
                next_beam = {}
                for seq, score in beam:
                    for c in range(V):
                        if c == blank_idx: continue
                        char = chr(96 + c) if 1 <= c <= 26 else "'"
                        new_seq = seq + char
                        lm_score = self.alpha * self.lm.normalize_score(new_seq)
                        total_score = score + probs[b, t, c].item() + lm_score
                        if new_seq not in next_beam or total_score > next_beam[new_seq]:
                            next_beam[new_seq] = total_score
                beam = sorted(next_beam.items(), key=lambda x: x[1], reverse=True)[:self.beam_width]
            decoded.append(beam[0][0])
        return decoded
