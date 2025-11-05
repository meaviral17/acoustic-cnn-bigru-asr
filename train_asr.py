import os, time, json, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import CNNBiGRU
from data.data_loader import LibriCTCDataset, ctc_collate
from utils import seed_everything, get_device, build_text_tokenizer
from utils.ema import EMA
from utils.metrics import compute_metrics
from upgrades.external_lm import ExternalLM
from upgrades.beam_search_decoder import beam_search_decode
import shutil

# Universal autocast / GradScaler import
try:
    from torch.amp import autocast, GradScaler
    TORCH2 = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    TORCH2 = False

# Smart directory selection for Kaggle
def get_best_dir():
    temp = "/kaggle/temp"
    work = "/kaggle/working"
    try:
        temp_free = shutil.disk_usage(temp).free if os.path.exists(temp) else 0
        work_free = shutil.disk_usage(work).free if os.path.exists(work) else 0
        return temp if temp_free > work_free else work
    except Exception:
        return work

BASE_DIR = get_best_dir()
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

print(f"Using base directory for training artifacts: {BASE_DIR}")

def greedy_decode(logits, out_lens):
    preds = logits.argmax(dim=-1).cpu().tolist()
    outs = []
    for i, L in enumerate(out_lens.cpu().tolist()):
        outs.append(preds[i][:L])
    return outs

def safe_load_checkpoint(path, device):
    try:
        state = torch.load(path, map_location=device)
        print(f"Successfully loaded checkpoint from {path}")
        return state
    except EOFError:
        print(f"Checkpoint {path} appears corrupted — restarting fresh.")
        os.remove(path)
        return None
    except Exception as e:
        print(f"Could not load checkpoint ({e}) — starting new model.")
        return None

def safe_save_state_dict(model, path):
    tmp = path + ".tmp"
    try:
        sd = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(sd, tmp, _use_new_zipfile_serialization=False)
        if os.path.getsize(tmp) > 0:
            os.replace(tmp, path)
            print(f"# Saved checkpoint: {path}")
        else:
            print("# Skipped empty checkpoint write.")
    except Exception as e:
        print(f"# Checkpoint save failed: {e}")
        if os.path.exists(tmp): os.remove(tmp)

def run(args):
    ckpt_path = os.path.join(CKPT_DIR, "latest.pt")
    log_hist_path = os.path.join(LOG_DIR, "training_history.json")
    log_summary_path = os.path.join(LOG_DIR, "run_summary.json")
    log_samples_path = os.path.join(LOG_DIR, "sample_decodes.txt")

    seed_everything(args.seed)
    device = get_device()

    vocab = ["<blk>"] + list("abcdefghijklmnopqrstuvwxyz'|")
    encode, decode, stoi, itos, blank = build_text_tokenizer(vocab)

    train = LibriCTCDataset(args.data_root, args.train_subset, (encode, decode, stoi, itos, blank),
                            pcen=bool(args.pcen), tf_mixup=bool(args.tf_mixup), specaug=bool(args.specaug),
                            max_len_sec=args.max_len_sec)
    valid = LibriCTCDataset(args.data_root, args.valid_subset, (encode, decode, stoi, itos, blank),
                            pcen=bool(args.pcen))
    test = LibriCTCDataset(args.data_root, args.test_subset, (encode, decode, stoi, itos, blank),
                           pcen=bool(args.pcen))

    dl_train = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=ctc_collate)
    dl_valid = DataLoader(valid, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=ctc_collate)

    model = CNNBiGRU(n_mels=train.n_mels, vocab_size=len(vocab)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9,0.98), weight_decay=1e-4)
    ctc_loss = nn.CTCLoss(blank=blank, zero_infinity=True)
    ema = EMA(model, decay=0.999) if args.ema else None

    scaler = GradScaler()

    lm = None
    if args.use_lm:
        lm = ExternalLM(args.lm_path, alpha=args.lm_alpha, beta=args.lm_beta)
        print(f"# External KenLM initialized: {args.lm_path}")

    state = safe_load_checkpoint(ckpt_path, device)
    if state is not None:
        model.load_state_dict(state)

    hist = {"train_loss": [], "val_loss": [], "wer": [], "cer": []}
    sample_logs = []

    for epoch in range(1, args.epochs+1):
        model.train()
        ep_loss = 0.0
        pbar = tqdm(dl_train, desc=f"Epoch {epoch}/{args.epochs}")

        for x, x_lens, ys, y_lens in pbar:
            x, x_lens, ys, y_lens = x.to(device), x_lens.to(device), ys.to(device), y_lens.to(device)
            opt.zero_grad()

            # Universal autocast support
            with autocast("cuda" if torch.cuda.is_available() else "cpu") if TORCH2 else autocast():
                logits, out_lens = model(x, x_lens)
                log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
                loss = ctc_loss(log_probs, ys, out_lens, y_lens)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(opt)
            scaler.update()
            if ema: ema.update(model)
            ep_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        train_loss = ep_loss / len(dl_train)
        hist["train_loss"].append(train_loss)

        model.eval()
        val_loss, all_preds, all_refs = 0.0, [], []
        with torch.no_grad():
            for x, x_lens, ys, y_lens in tqdm(dl_valid, desc=f"[Eval Epoch {epoch}]"):
                x, x_lens, ys, y_lens = x.to(device), x_lens.to(device), ys.to(device), y_lens.to(device)
                with autocast("cuda" if torch.cuda.is_available() else "cpu") if TORCH2 else autocast():
                    logits, out_lens = model(x, x_lens)
                    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
                    loss = ctc_loss(log_probs, ys, out_lens, y_lens)
                val_loss += loss.item()
                pred_ids = greedy_decode(logits, out_lens)
                for p in pred_ids:
                    all_preds.append(decode(p))

                ptr = 0
                yl = y_lens.detach().cpu().tolist()
                ycpu = ys.detach().cpu()
                for L in yl:
                    seq = ycpu[ptr:ptr+L].tolist()
                    ref = "".join(itos[i] for i in seq).replace("|", " ").strip()
                    all_refs.append(ref)
                    ptr += L

        val_loss /= len(dl_valid)
        WER, CER = compute_metrics(all_preds, all_refs)
        hist["val_loss"].append(val_loss)
        hist["wer"].append(WER)
        hist["cer"].append(CER)

        print(f"[Epoch {epoch}] train={train_loss:.3f} val={val_loss:.3f} WER={WER:.3f} CER={CER:.3f}")

        for i in range(min(5, len(all_preds))):
            sample_logs.append(f"HYP: {all_preds[i]}\nREF: {all_refs[i]}\n---\n")

        safe_save_state_dict(model, ckpt_path)
        with open(log_hist_path, "w") as f:
            json.dump(hist, f, indent=2)
        with open(log_samples_path, "w") as f:
            f.writelines(sample_logs)
        with open(log_summary_path, "w") as f:
            json.dump({"epochs": epoch, "train_loss": train_loss, "val_loss": val_loss, "WER": WER, "CER": CER}, f, indent=2)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--train_subset", type=str, default="train-clean-100")
    ap.add_argument("--valid_subset", type=str, default="test-clean")
    ap.add_argument("--test_subset", type=str, default="test-clean")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--clip", type=float, default=3.0)
    ap.add_argument("--pcen", type=int, default=0)
    ap.add_argument("--ema", type=int, default=1)
    ap.add_argument("--tf_mixup", type=int, default=0)
    ap.add_argument("--specaug", type=int, default=0)
    ap.add_argument("--use_lm", type=int, default=1)
    ap.add_argument("--lm_path", type=str, default="lm/english_5gram.binary")
    ap.add_argument("--lm_alpha", type=float, default=0.6)
    ap.add_argument("--lm_beta", type=float, default=1.0)
    ap.add_argument("--max_len_sec", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args)
