import os, math, time, json, argparse
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import CNNBiGRU
from data import LibriCTCDataset, ctc_collate
from utils import seed_everything, get_device, count_parameters, human_time, build_text_tokenizer
from utils.ema import EMA
from utils.metrics import compute_metrics
from upgrades.beam_search_decoder import beam_search_decode
from upgrades.data_parallel import setup_ddp
from upgrades.beam_search_decoder_lm import KenLMDecoder

def greedy_decode(logits, out_lens):
    preds = logits.argmax(dim=-1).cpu().tolist()
    outs = []
    for i, L in enumerate(out_lens.cpu().tolist()):
        outs.append(preds[i][:L])
    return outs

def ensure_dirs():
    os.makedirs("experiments/checkpoints", exist_ok=True)
    os.makedirs("experiments/plots", exist_ok=True)
    os.makedirs("experiments/logs", exist_ok=True)

def run(args):
    ensure_dirs()
    seed_everything(args.seed)
    device = get_device()

    vocab = list("abcdefghijklmnopqrstuvwxyz'|")
    vocab = ["<blk>"] + vocab
    encode, decode, stoi, itos, blank = build_text_tokenizer(vocab)

    # Dataset loaders
    train = LibriCTCDataset(args.data_root, args.train_subset, (encode, decode, stoi, itos, blank),
                            pcen=bool(args.pcen), tf_mixup=bool(args.tf_mixup), specaug=bool(args.specaug),
                            max_len_sec=(args.max_len_sec if args.curriculum else None))
    valid = LibriCTCDataset(args.data_root, args.valid_subset, (encode, decode, stoi, itos, blank),
                            pcen=bool(args.pcen), specaug=False)
    test = LibriCTCDataset(args.data_root, args.test_subset, (encode, decode, stoi, itos, blank),
                           pcen=bool(args.pcen), specaug=False)

    dl_train = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=ctc_collate, pin_memory=True)
    dl_valid = DataLoader(valid, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=ctc_collate)
    dl_test = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=ctc_collate)

    # Model
    model = CNNBiGRU(n_mels=train.n_mels, vocab_size=len(vocab), use_se=bool(args.use_se),
                     cnn_channels=args.cnn_channels, num_gru=args.num_gru, gru_hidden=args.gru_hidden, dropout=args.dropout)
    model = setup_ddp(model).to(device)

    # Optimizer + Scheduler
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9,0.98), weight_decay=1e-4)
    ctc_loss = nn.CTCLoss(blank=blank, zero_infinity=True)
    ema = EMA(model, decay=0.999) if args.ema else None

    warmup_steps = 1 + int(0.05 * args.epochs * max(1,len(dl_train)))
    total_steps = args.epochs * max(1,len(dl_train))
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # Language Model setup
    lm_decoder = None
    if args.use_lm and os.path.exists(args.lm_path):
        lm_decoder = KenLMDecoder(args.lm_path, alpha=args.lm_alpha, beta=args.lm_beta,
                                  beam_width=args.beam_width, blank_idx=blank)
        print(f"✅ Using KenLM from {args.lm_path}")
    else:
        print("⚠️ No valid KenLM found or disabled; running without LM fusion.")

    best_val = 1e9
    hist = {"train_loss": [], "val_loss": [], "wer": [], "cer": []}
    t0 = time.time()
    step = 0

    # ============ Training ============
    for epoch in range(1, args.epochs+1):
        model.train()
        ep_loss = 0.0
        if args.curriculum:
            train.max_len_sec = None if epoch == args.epochs else args.max_len_sec

        pbar = tqdm(dl_train, desc=f"Epoch {epoch}/{args.epochs}")
        for x, x_lens, ys, y_lens in pbar:
            x, x_lens, ys, y_lens = x.to(device), x_lens.to(device), ys.to(device), y_lens.to(device)
            logits, out_lens = model(x, x_lens)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            loss = ctc_loss(log_probs, ys, out_lens, y_lens)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step(); sched.step()
            if ema: ema.update(model)

            ep_loss += loss.item(); step += 1
            pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{opt.param_groups[0]['lr']:.2e}")

        train_loss = ep_loss / len(dl_train)
        hist["train_loss"].append(train_loss)

        # ============ Validation ============
        model.eval()
        if ema: ema.apply_to(model)
        with torch.no_grad():
            val_loss, hyps, refs = 0.0, [], []
            for x, x_lens, ys, y_lens in dl_valid:
                x, x_lens, ys, y_lens = x.to(device), x_lens.to(device), ys.to(device), y_lens.to(device)
                logits, out_lens = model(x, x_lens)
                log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
                loss = ctc_loss(log_probs, ys, out_lens, y_lens)
                val_loss += loss.item()

                if args.beam:
                    if lm_decoder is not None:
                        pred_texts = lm_decoder.decode(logits, out_lens, vocab)
                    else:
                        pred_ids = beam_search_decode(logits, out_lens, blank, beam_width=args.beam_width)
                        pred_texts = [decode(s) for s in pred_ids]
                else:
                    pred_ids = greedy_decode(logits, out_lens)
                    pred_texts = [decode(s) for s in pred_ids]

                hyps += pred_texts
                ptr = 0
                for L in y_lens.tolist():
                    ref = ys[ptr:ptr+L].cpu().tolist()
                    refs.append("".join([itos[i] for i in ref]).replace("|"," ").strip())
                    ptr += L

            val_loss /= len(dl_valid)

        if ema: ema.restore(model)
        if hyps and refs:
            w, c = compute_metrics(refs, hyps)
        else:
            w, c = float("nan"), float("nan")

        hist["val_loss"].append(val_loss); hist["wer"].append(w); hist["cer"].append(c)
        print(f"[Epoch {epoch}] train={train_loss:.3f} val={val_loss:.3f} WER={w:.3f} CER={c:.3f} time={human_time(time.time()-t0)}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model":model.state_dict(),"stoi":stoi,"itos":itos,"blank":blank},
                       "experiments/checkpoints/best_cnn_bigru_ctc.pt")

    # ============ Final Test ============
    model.eval()
    if ema: ema.apply_to(model)
    with torch.no_grad():
        hyps, refs = [], []
        test_loss = 0.0
        for x, x_lens, ys, y_lens in dl_test:
            x, x_lens, ys, y_lens = x.to(device), x_lens.to(device), ys.to(device), y_lens.to(device)
            logits, out_lens = model(x, x_lens)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            loss = ctc_loss(log_probs, ys, out_lens, y_lens)
            test_loss += loss.item()

            if args.beam:
                if lm_decoder is not None:
                    pred_texts = lm_decoder.decode(logits, out_lens, vocab)
                else:
                    pred_ids = beam_search_decode(logits, out_lens, blank, beam_width=args.beam_width)
                    pred_texts = [decode(s) for s in pred_ids]
            else:
                pred_ids = greedy_decode(logits, out_lens)
                pred_texts = [decode(s) for s in pred_ids]

            hyps += pred_texts
            ptr = 0
            for L in y_lens.tolist():
                ref = ys[ptr:ptr+L].cpu().tolist()
                refs.append("".join([itos[i] for i in ref]).replace("|"," ").strip())
                ptr += L

        test_loss /= len(dl_test)

    if ema: ema.restore(model)
    T_WER, T_CER = compute_metrics(refs, hyps) if hyps and refs else (float("nan"), float("nan"))

    # ============ Save metrics ============
    lm_tag = "with_lm" if args.use_lm and lm_decoder else "no_lm"
    summary_path = f"experiments/logs/run_summary_{lm_tag}.json"

    summary = {
        "params_m": round(count_parameters(model)/1e6, 3),
        "features": "PCEN" if bool(args.pcen) else "Log-Mel",
        "use_SE": bool(args.use_se),
        "tf_mixup": bool(args.tf_mixup),
        "curriculum": bool(args.curriculum),
        "ema": bool(args.ema),
        "beam": bool(args.beam),
        "use_lm": bool(args.use_lm),
        "lm_path": args.lm_path if args.use_lm else "",
        "lm_alpha": args.lm_alpha,
        "lm_beta": args.lm_beta,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "test_loss": float(test_loss),
        "test_WER": float(T_WER),
        "test_CER": float(T_CER)
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved results to {summary_path}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--train_subset", type=str, default="train-clean-100")
    ap.add_argument("--valid_subset", type=str, default="dev-clean")
    ap.add_argument("--test_subset", type=str, default="test-clean")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--clip", type=float, default=3.0)
    ap.add_argument("--pcen", type=int, default=0)
    ap.add_argument("--use_se", type=int, default=1)
    ap.add_argument("--tf_mixup", type=int, default=1)
    ap.add_argument("--specaug", type=int, default=1)
    ap.add_argument("--curriculum", type=int, default=1)
    ap.add_argument("--max_len_sec", type=float, default=6.0)
    ap.add_argument("--cnn_channels", type=int, default=128)
    ap.add_argument("--num_gru", type=int, default=3)
    ap.add_argument("--gru_hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--ema", type=int, default=1)
    ap.add_argument("--beam", type=int, default=0)
    ap.add_argument("--beam_width", type=int, default=10)
    ap.add_argument("--use_lm", type=int, default=1)
    ap.add_argument("--lm_path", type=str, default="lm/english_5gram.binary")
    ap.add_argument("--lm_alpha", type=float, default=0.6)
    ap.add_argument("--lm_beta", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args)
