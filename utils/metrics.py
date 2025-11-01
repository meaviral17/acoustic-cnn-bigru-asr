import numpy as np

def wer(refs, hyps):
    import editdistance
    total, errors = 0, 0
    for r, h in zip(refs, hyps):
        errors += editdistance.eval(r.split(), h.split())
        total += len(r.split())
    return errors / total if total else 0.0

def cer(refs, hyps):
    import editdistance
    total, errors = 0, 0
    for r, h in zip(refs, hyps):
        errors += editdistance.eval(list(r), list(h))
        total += len(r)
    return errors / total if total else 0.0

def compute_metrics(refs, hyps):
    return wer(refs, hyps), cer(refs, hyps)
