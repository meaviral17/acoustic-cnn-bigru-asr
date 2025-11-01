import torch
import math

def beam_search_decode(logits, out_lens, blank_idx, beam_width=10, lm=None):
    try:
        from torchaudio.models.decoder import CTCBeamSearchDecoder
        V = logits.shape[-1]
        labels = [str(i) for i in range(V)]
        decoder = CTCBeamSearchDecoder(labels=labels, beam_width=beam_width)
        probs = logits.softmax(dim=-1).cpu()
        results = decoder(probs, out_lens.cpu())
        out = []
        for res in results:
            seq = res[0][0].tokens
            out.append(seq)
        if lm:
            rescored = []
            for seq in out:
                s = "".join([str(i) for i in seq])
                rescored.append((lm.score(s), seq))
            rescored.sort(reverse=True)
            out = [seq for _, seq in rescored]
        return out
    except Exception:
        preds = logits.argmax(dim=-1).cpu().tolist()
        return preds
