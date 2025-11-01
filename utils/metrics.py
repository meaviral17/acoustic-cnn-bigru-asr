from jiwer import wer, cer
def compute_metrics(refs, hyps):
    return wer(refs, hyps), cer(refs, hyps)
