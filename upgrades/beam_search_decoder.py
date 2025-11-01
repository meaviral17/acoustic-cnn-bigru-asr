import torch
def beam_search_decode(logits, out_lens, blank_idx, beam_width=10):
    try:
        from torchaudio.models.decoder import CTCBeamSearchDecoder
        labels = [str(i) for i in range(logits.shape[-1])]
        decoder = CTCBeamSearchDecoder(labels, beam_width=beam_width)
        results = decoder(logits.softmax(dim=-1).cpu(), out_lens)
        out = []
        for res in results:
            seq = res[0][0].tokens
            out.append(seq)
        return out
    except:
        return logits.argmax(dim=-1).cpu().tolist()
