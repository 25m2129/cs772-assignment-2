import json
import pandas as pd
import random
import re
import unicodedata
from pathlib import Path
from collections import Counter
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --- 1. CONFIGURATION AND REPRODUCIBILITY ---
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

data_dir = Path("data")
SAVE_DIR = Path("models")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = SAVE_DIR / "seq2seq_lstm.pth" # Define the model path

# Training/Data Hyperparameters
MAX_TRAIN = 100_000
BATCH_SIZE = 128
N_EPOCHS = 12

# Special tokens indices
SPECIALS = ["<pad>", "<sos>", "<eos>", "<unk>"]
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3


# --- 2. DATA LOADING, CLEANING, AND SUBSAMPLING (UNCHANGED) ---

def load_json(path):
    """Loads data from a JSON or JSON Lines file into a DataFrame."""
    try:
        text = path.read_text(encoding="utf-8").strip()
        if text.startswith('['):
            data = json.loads(text)
        else:
            data = [json.loads(line) for line in text.splitlines() if line.strip()]
        return pd.DataFrame(data)
    except FileNotFoundError:
        print(f"Error: Data file not found at {path}. Please ensure it exists.")
        return pd.DataFrame()

def clean_roman(s):
    """Cleans English (Roman) text."""
    if pd.isna(s): return s
    s = str(s).lower().strip()
    s = re.sub(r'\s+', ' ', s)
    return s

def clean_devanagari(s):
    """Cleans Native (Devanagari) text."""
    if pd.isna(s): return s
    s = unicodedata.normalize('NFC', str(s).strip())
    s = re.sub(r'\s+', ' ', s)
    return s

def stratified_by_length(df, n, key='english', n_bins=10, seed=42):
    """Stratify by input length to keep coverage of both short and long sequences."""
    if len(df) <= n: return df
    df = df.copy()
    df['len'] = df[key].str.len()
    df['bin'] = pd.qcut(df['len'], q=n_bins, duplicates='drop')
    samples = []
    for _, g in df.groupby('bin'):
        k = max(1, int(round(len(g) / len(df) * n)))
        samples.append(g.sample(n=min(k, len(g)), random_state=seed))
    sampled = pd.concat(samples).reset_index(drop=True)
    return sampled.sample(n=n, random_state=seed).reset_index(drop=True)

# Load dataframes
train_df = load_json(data_dir / "hin_train.json")
valid_df = load_json(data_dir / "hin_valid.json")
test_df  = load_json(data_dir / "hin_test.json")

# Cleaning and deduplication
for df in [train_df, valid_df, test_df]:
    if not df.empty and 'english word' in df.columns and 'native word' in df.columns:
        df['english'] = df['english word'].map(clean_roman)
        df['native']  = df['native word'].map(clean_devanagari)
        df.dropna(subset=['english','native'], inplace=True)
        df.drop_duplicates(subset=['english','native'], inplace=True)
        df.reset_index(drop=True, inplace=True)
    else:
        print(f"Skipping processing for an empty or incorrectly formatted DataFrame.")

# Subsampling training data
if not train_df.empty:
    train_df = stratified_by_length(train_df, MAX_TRAIN)
    print("Train size after length-stratified subsampling:", len(train_df))
else:
    print("Cannot subsample: train_df is empty.")


# --- 3. VOCABULARY AND UTILITIES (UNCHANGED) ---

def build_char_vocab_from_dfs(dfs, min_freq=1):
    """Builds a character vocabulary from the 'english' and 'native' columns."""
    counter = Counter()
    for df in dfs:
        if not df.empty:
            counter.update(''.join(df['english'].tolist()))
            counter.update(''.join(df['native'].tolist()))

    chars = sorted([c for c,f in counter.items() if f >= min_freq], key=lambda x: (-counter[x], x))
    vocab = SPECIALS + chars
    idx2char = {i:c for i,c in enumerate(vocab)}
    char2idx = {c:i for i,c in idx2char.items()}
    return vocab, char2idx, idx2char

# Build vocab based on current data
vocab, char2idx, idx2char = build_char_vocab_from_dfs([train_df, valid_df, test_df])
VOCAB_SIZE = len(vocab)
print("Vocab size:", VOCAB_SIZE)

def seq_to_ids(seq, char2idx, add_sos_eos=True):
    """Converts a character sequence to a list of IDs."""
    ids = [char2idx.get(c, UNK_IDX) for c in seq]
    if add_sos_eos:
        ids = [SOS_IDX] + ids + [EOS_IDX]
    return ids

def ids_to_seq(ids, idx2char, remove_specials=True):
    """Converts a list of IDs back to a character sequence."""
    chars = [idx2char.get(i, "") for i in ids]
    if remove_specials:
        chars = [c for c in chars if c not in SPECIALS]
    return ''.join(chars)


# --- 4. PYTORCH DATASET AND DATALOADER (UNCHANGED) ---

class TransliterationDataset(Dataset):
    def __init__(self, df, char2idx):
        self.df = df.reset_index(drop=True)
        self.char2idx = char2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        src = self.df.loc[idx, 'english']
        tgt = self.df.loc[idx, 'native']
        src_ids = seq_to_ids(src, self.char2idx, add_sos_eos=True)
        tgt_ids = seq_to_ids(tgt, self.char2idx, add_sos_eos=True)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long), src, tgt

def collate_fn(batch):
    """Pads sequences and returns batches for DataLoader."""
    srcs, tgts, src_texts, tgt_texts = zip(*batch)
    src_lens = [len(s) for s in srcs]
    tgt_lens = [len(t) for t in tgts]
    max_src = max(src_lens)
    max_tgt = max(tgt_lens)

    padded_srcs = torch.full((len(batch), max_src), PAD_IDX, dtype=torch.long)
    padded_tgts = torch.full((len(batch), max_tgt), PAD_IDX, dtype=torch.long)

    for i, s in enumerate(srcs):
        padded_srcs[i, :len(s)] = s
    for i, t in enumerate(tgts):
        padded_tgts[i, :len(t)] = t

    return padded_srcs, torch.tensor(src_lens, dtype=torch.long), padded_tgts, torch.tensor(tgt_lens, dtype=torch.long), list(src_texts), list(tgt_texts)

# Create DataLoaders
if not train_df.empty:
    train_ds = TransliterationDataset(train_df, char2idx)
    valid_ds = TransliterationDataset(valid_df, char2idx)
    test_ds  = TransliterationDataset(test_df,  char2idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
else:
    print("Cannot create DataLoaders: training data is empty.")
    exit()


# --- 5. MODEL DEFINITION (ENCODER, ATTENTION, DECODER) (UNCHANGED) ---

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_hidden, n_layers=2, dropout=0.2, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.rnn = nn.LSTM(emb_dim, enc_hidden, num_layers=n_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout if n_layers>1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        self.enc_hidden = enc_hidden
        self.n_layers = n_layers

    def forward(self, src, src_lens):
        embedded = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return outputs, (h_n, c_n)

class LuongAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        self.attn = nn.Linear(dec_dim, enc_dim, bias=False)

    def forward(self, dec_hidden, enc_outputs, mask=None):
        proj = self.attn(dec_hidden).unsqueeze(2)
        scores = torch.bmm(enc_outputs, proj).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        attn_weights = torch.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1)
        return context, attn_weights

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_hidden, dec_hidden, n_layers=2, dropout=0.2, bidirectional_encoder=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        enc_output_dim = enc_hidden * (2 if bidirectional_encoder else 1)
        self.rnn = nn.LSTM(emb_dim + enc_output_dim, dec_hidden, num_layers=n_layers, batch_first=True, dropout=dropout if n_layers>1 else 0.0)
        self.attention = LuongAttention(enc_output_dim, dec_hidden)
        self.out = nn.Linear(dec_hidden + enc_output_dim + emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.dec_hidden_dim = dec_hidden

    def forward_step(self, input_token, last_hidden, enc_outputs, enc_mask):
        embedded = self.embedding(input_token).unsqueeze(1)
        h_last = last_hidden[0][-1]
        context, attn_weights = self.attention(h_last, enc_outputs, mask=enc_mask)
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        output, (h, c) = self.rnn(rnn_input, last_hidden)
        output = self.dropout(output.squeeze(1))
        concat = torch.cat([output, context, embedded.squeeze(1)], dim=1)
        logits = self.out(concat)
        return logits, (h, c), attn_weights

    def forward(self, tgt, enc_outputs, enc_lens, teacher_forcing_ratio=0.5):
        batch_size = tgt.size(0)
        max_tgt = tgt.size(1)
        device = tgt.device
        enc_mask = torch.arange(enc_outputs.size(1), device=device).unsqueeze(0) < enc_lens.unsqueeze(1)
        outputs = torch.zeros(batch_size, max_tgt, self.out.out_features, device=device)
        h0 = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=device)
        c0 = torch.zeros_like(h0)
        hidden = (h0, c0)
        input_token = tgt[:,0]

        for t in range(1, max_tgt):
            logits, hidden, _ = self.forward_step(input_token, hidden, enc_outputs, enc_mask)
            outputs[:, t, :] = logits
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = logits.argmax(1)
            input_token = tgt[:, t] if teacher_force else top1
        return outputs

# Initialize Model and Optimizer
EMB_DIM = 128
ENC_HIDDEN = 256
DEC_HIDDEN = 256
ENC_N_LAYERS = 2
DEC_N_LAYERS = 2
DROPOUT = 0.2
BIDIRECTIONAL = True

encoder = Encoder(VOCAB_SIZE, EMB_DIM, ENC_HIDDEN, n_layers=ENC_N_LAYERS, dropout=DROPOUT, bidirectional=BIDIRECTIONAL).to(DEVICE)
decoder = Decoder(VOCAB_SIZE, EMB_DIM, ENC_HIDDEN, DEC_HIDDEN, n_layers=DEC_N_LAYERS, dropout=DROPOUT, bidirectional_encoder=BIDIRECTIONAL).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(params, lr=0.001)

print(f"Total model parameters: {sum(p.numel() for p in params):,}")


# --- 6. TRAINING AND EVALUATION FUNCTIONS (UNCHANGED) ---

def train_epoch(encoder, decoder, loader, optimizer, criterion, device, teacher_forcing=0.5):
    encoder.train()
    decoder.train()
    epoch_loss = 0
    for src, src_lens, tgt, tgt_lens, _, _ in tqdm(loader, desc="Train batches"):
        src = src.to(device)
        src_lens = src_lens.to(device)
        tgt = tgt.to(device)
        optimizer.zero_grad()
        enc_outputs, _ = encoder(src, src_lens)
        outputs = decoder(tgt, enc_outputs, src_lens, teacher_forcing_ratio=teacher_forcing)
        out_flat = outputs[:,1:,:].reshape(-1, outputs.size(-1))
        tgt_flat = tgt[:,1:].to(device).reshape(-1)
        loss = criterion(out_flat, tgt_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

def eval_epoch(encoder, decoder, loader, criterion, device):
    encoder.eval()
    decoder.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, src_lens, tgt, tgt_lens, _, _ in tqdm(loader, desc="Valid/Test batches"):
            src = src.to(device)
            src_lens = src_lens.to(device)
            tgt = tgt.to(device)
            enc_outputs, _ = encoder(src, src_lens)
            outputs = decoder(tgt, enc_outputs, src_lens, teacher_forcing_ratio=0.0)
            out_flat = outputs[:,1:,:].reshape(-1, outputs.size(-1))
            tgt_flat = tgt[:,1:].reshape(-1)
            loss = criterion(out_flat, tgt_flat)
            epoch_loss += loss.item()
    return epoch_loss / len(loader)

def levenshtein(a, b):
    if a == b: return 0
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    prev = list(range(lb+1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0]*lb
        for j, cb in enumerate(b, start=1):
            ins = cur[j-1] + 1
            delete = prev[j] + 1
            sub = prev[j-1] + (0 if ca==cb else 1)
            cur[j] = min(ins, delete, sub)
        prev = cur
    return prev[lb]

def evaluate_predictions(encoder, decoder, loader, device, max_len=64):
    encoder.eval()
    decoder.eval()
    total, exact_matches, total_chars, correct_chars, total_edit = 0, 0, 0, 0, 0.0
    samples = []
    with torch.no_grad():
        for src, src_lens, tgt, tgt_lens, src_texts, tgt_texts in tqdm(loader, desc="Eval decode"):
            src = src.to(device)
            src_lens = src_lens.to(device)
            enc_outputs, _ = encoder(src, src_lens)
            batch_size = src.size(0)

            h = torch.zeros(decoder.rnn.num_layers, batch_size, decoder.rnn.hidden_size, device=device)
            c = torch.zeros_like(h)
            hidden = (h, c)
            input_token = torch.tensor([SOS_IDX]*batch_size, device=device)
            outputs_tokens = torch.full((batch_size, max_len), PAD_IDX, dtype=torch.long, device=device)
            enc_mask = torch.arange(enc_outputs.size(1), device=device).unsqueeze(0) < src_lens.unsqueeze(1)

            for t in range(1, max_len):
                logits, hidden, _ = decoder.forward_step(input_token, hidden, enc_outputs, enc_mask)
                top1 = logits.argmax(1)
                outputs_tokens[:, t] = top1
                input_token = top1

            for i in range(batch_size):
                out_ids = outputs_tokens[i].tolist()
                if EOS_IDX in out_ids:
                    cut = out_ids.index(EOS_IDX)
                    out_ids = out_ids[1:cut]
                else:
                    out_ids = out_ids[1:]
                pred = ids_to_seq(out_ids, idx2char)
                gold = tgt_texts[i]
                total += 1
                if pred == gold:
                    exact_matches += 1
                ed = levenshtein(pred, gold)
                total_edit += ed
                total_chars += len(gold)
                correct_chars += max(0, len(gold) - ed)

                if len(samples) < 10:
                    samples.append((src_texts[i], gold, pred))

    metrics = {
        "exact_accuracy": exact_matches / total if total>0 else 0.0,
        "mean_edit": total_edit / total if total>0 else 0.0,
        "char_accuracy": correct_chars / total_chars if total_chars>0 else 0.0,
        "samples": samples
    }
    return metrics


# --- 7. TRAINING LOGIC (Conditional) ---

def train_and_save_model():
    """Function to run the full training process."""
    global best_valid_loss
    best_valid_loss = float('inf')
    teacher_forcing = 0.5
    print("\n--- Starting Training ---")
    for epoch in range(1, N_EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{N_EPOCHS} (TF: {teacher_forcing:.2f}) ===")
        train_loss = train_epoch(encoder, decoder, train_loader, optimizer, criterion, DEVICE, teacher_forcing=teacher_forcing)
        valid_loss = eval_epoch(encoder, decoder, valid_loader, criterion, DEVICE)
        print(f"Train loss: {train_loss:.4f}  Valid loss: {valid_loss:.4f}")

        metrics = evaluate_predictions(encoder, decoder, valid_loader, DEVICE, max_len=64)
        print("Val exact_acc: {:.4f}  char_acc: {:.4f}  mean_edit: {:.3f}".format(metrics['exact_accuracy'], metrics['char_accuracy'], metrics['mean_edit']))

        # Save best model based on validation loss
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save({
                'epoch': epoch,
                'encoder_state': encoder.state_dict(),
                'decoder_state': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'char2idx': char2idx,
                'idx2char': idx2char,
            }, MODEL_PATH)
            print(f"Saved best model to {MODEL_PATH}.")

        # Decay teacher forcing
        teacher_forcing = max(0.1, teacher_forcing * 0.95)


# --- MAIN EXECUTION BLOCK ---

if MODEL_PATH.exists():
    # Load model if it exists
    print(f"\n✅ Checkpoint found at {MODEL_PATH}. Skipping training.")
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    encoder.load_state_dict(ckpt['encoder_state'])
    decoder.load_state_dict(ckpt['decoder_state'])
    print("Loaded model from epoch", ckpt.get('epoch', 'N/A'))
else:
    # Train and save model if it does not exist
    print(f"\n❌ Checkpoint NOT found at {MODEL_PATH}. Starting training.")
    train_and_save_model()
    # Reload the model after training to ensure we use the best checkpoint
    if MODEL_PATH.exists():
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        encoder.load_state_dict(ckpt['encoder_state'])
        decoder.load_state_dict(ckpt['decoder_state'])
        print(f"Loaded newly trained best model from epoch", ckpt.get('epoch', 'N/A'))
    else:
        print("Error: Training completed but no model saved. Cannot proceed to test.")
        exit()


# --- 8. FINAL TEST EVALUATION ---

print("\n=== Running Final Test Evaluation ===")
test_metrics = evaluate_predictions(encoder, decoder, test_loader, DEVICE, max_len=64)
print("\n=== Test results ===")
print("Exact word accuracy:", test_metrics['exact_accuracy'])
print("Char accuracy (approx):", test_metrics['char_accuracy'])
print("Mean edit distance:", test_metrics['mean_edit'])

print("\nSample predictions:")
for s, gold, pred in test_metrics['samples']:
    print(f"SRC: {s:<15}  GOLD: {gold:<15}  PRED: {pred}")

# --------------------------------------------------------------------------------------

from collections import Counter, defaultdict
from difflib import SequenceMatcher
from itertools import zip_longest
import csv
import math
import statistics
from tqdm import tqdm

# The main script already calculated test_metrics and all_errors using evaluate_predictions_with_errors.
# We will replicate the decoding part or use a streamlined version if needed, but since the
# evaluate_predictions_with_errors function is already very useful, let's use it
# and then add the N-gram analysis.

# PARAMETERS for the analysis
MIN_SUPPORT_NGRAM = 20    # minimum number of examples containing an n-gram to consider it
TOP_K = 30                # how many top confusions / ngrams to print
NGRAM_MAX = 3             # consider roman n-grams up to length 3
SAVE_PREFIX = "lstm_error_analysis" # CSV prefix
MAX_DECODE_LEN = 64       # Should match the max_len used in evaluate_predictions

# Helper function to align characters for detailed error analysis
def align_chars(gold, pred):
    """
    Returns list of tuples (op, gold_char, pred_char)
    op in {"equal","replace","delete","insert"}
    Uses difflib.SequenceMatcher for a reasonable character alignment.
    """
    sm = SequenceMatcher(None, gold, pred)
    ops = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            for ga, pa in zip(gold[i1:i2], pred[j1:j2]):
                ops.append(('equal', ga, pa))
        elif tag == 'replace':
            # align roughly pairwise; lengths may differ
            for ga, pa in zip_longest(gold[i1:i2], pred[j1:j2], fillvalue=''):
                ops.append(('replace', ga, pa))
        elif tag == 'delete':
            for ga in gold[i1:i2]:
                ops.append(('delete', ga, ''))
        elif tag == 'insert':
            for pa in pred[j1:j2]:
                ops.append(('insert', '', pa))
    return ops

# --- 2. DECODING (Replicate/Extract the decoding logic to get all predictions) ---

def get_all_predictions(encoder, decoder, loader, device, max_len=64):
    """Decodes sequences and collects (source, gold, prediction) tuples."""
    encoder.eval(); decoder.eval();
    all_records = []
    with torch.no_grad():
        for src, src_lens, _, _, src_texts, tgt_texts in tqdm(loader, desc="Collecting Predictions"):
            src = src.to(device); src_lens = src_lens.to(device);
            enc_outputs, _ = encoder(src, src_lens);
            batch_size = src.size(0);

            # Initialize decoder state and input (same as evaluate_predictions)
            h = torch.zeros(decoder.rnn.num_layers, batch_size, decoder.rnn.hidden_size, device=device)
            c = torch.zeros_like(h)
            hidden = (h, c)
            input_token = torch.tensor([SOS_IDX]*batch_size, device=device)
            outputs_tokens = torch.full((batch_size, max_len), PAD_IDX, dtype=torch.long, device=device)
            enc_mask = torch.arange(enc_outputs.size(1), device=device).unsqueeze(0) < src_lens.unsqueeze(1)

            for t in range(1, max_len):
                logits, hidden, _ = decoder.forward_step(input_token, hidden, enc_outputs, enc_mask)
                top1 = logits.argmax(1)
                outputs_tokens[:, t] = top1
                input_token = top1
                # Stop if all batches have predicted EOS or max_len reached
                # This simple loop without breaking is usually fine for fixed max_len

            # Process outputs
            for i in range(batch_size):
                out_ids = outputs_tokens[i].tolist()
                # Find EOS and cut
                if EOS_IDX in out_ids:
                    cut = out_ids.index(EOS_IDX)
                    out_ids = out_ids[1:cut]
                else:
                    out_ids = out_ids[1:] # Remove SOS
                
                pred = ids_to_seq(out_ids, idx2char)
                gold = tgt_texts[i]
                src_text = src_texts[i]
                all_records.append((src_text, gold, pred))

    return all_records

# Run the decoding
print("\n--- Running LSTM Error Analysis on Test Set ---")
all_records = get_all_predictions(encoder, decoder, test_loader, DEVICE, max_len=MAX_DECODE_LEN)
print(f"Total test examples processed: {len(all_records)}")

# --- 3. CHARACTER-LEVEL CONFUSION COUNTS ---

confusions = Counter()         # (gold_char, pred_char) -> count
confusion_by_type = Counter()  # counts of replace/delete/insert
total_chars = 0
total_errors = 0

for src_text, gold, pred in all_records:
    if gold != pred:
        total_errors += 1
    
    ops = align_chars(gold, pred)
    for op, gch, pch in ops:
        total_chars += (1 if op != 'insert' else 0) # count only positions in gold for accuracy denom
        if op == 'equal':
            continue
        # use placeholder for empty side
        gkey = gch if gch != '' else '<eps>'
        pkey = pch if pch != '' else '<eps>'
        confusions[(gkey, pkey)] += 1
        confusion_by_type[op] += 1

exact_accuracy = 1 - (total_errors / len(all_records))
print(f"Exact word accuracy (Recalculated): {exact_accuracy:.4f} ({total_errors} errors)")

# Print top substitution confusions (excluding pure insertions from pred-only)
print("\n--- Top character confusions (gold -> predicted), sorted by frequency: ---")
top_conf = [((g,p),c) for (g,p),c in confusions.most_common() if g != '<eps>'] # Focus on Gold characters (Replacements/Deletions)
for (g,p),cnt in top_conf[:TOP_K]:
    print(f"'{g}' -> '{p}'     count={cnt}")

print("\nSummary of edit operation counts (for non-equal operations):", dict(confusion_by_type))

# Save confusion table to CSV
with open(f"{SAVE_PREFIX}_char_confusions.csv", "w", encoding="utf-8", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["gold_char", "pred_char", "count"])
    for (g,p),cnt in confusions.most_common():
        writer.writerow([g, p, cnt])
print(f"Wrote char confusions to {SAVE_PREFIX}_char_confusions.csv")

# --- 4. ROMAN N-GRAM => EXAMPLE-LEVEL ERROR ASSOCIATION ---

ngram_totals = defaultdict(int)
ngram_errors = defaultdict(int)

for src_text, gold, pred in all_records:
    is_err = (gold != pred)
    s = src_text.lower()
    
    for n in range(1, NGRAM_MAX+1):
        seen = set()
        for i in range(len(s)-n+1):
            ng = s[i:i+n]
            # skip spaces and duplicates within the word
            if ' ' in ng:
                continue
            if ng in seen:
                continue
            seen.add(ng)
            
            ngram_totals[(n,ng)] += 1
            if is_err:
                ngram_errors[(n,ng)] += 1

# compute error rate per ngram (filter by support)
ngram_stats = []
for (n,ng),tot in ngram_totals.items():
    if tot < MIN_SUPPORT_NGRAM:
        continue
    errs = ngram_errors.get((n,ng), 0)
    rate = errs / tot
    ngram_stats.append((n, ng, tot, errs, rate))
    
# sort by error rate (and then by support)
ngram_stats_sorted = sorted(ngram_stats, key=lambda x: (-x[4], -x[2]))

print(f"\n--- Top Roman n-grams (n<= {NGRAM_MAX}) by word-level error rate (min support = {MIN_SUPPORT_NGRAM}): ---")
print("{:<5} {:<5} {:<10} {:<10} {:<10}".format("N", "NGRAM", "SUPPORT", "ERRORS", "ERR_RATE"))
print("-" * 45)
for n, ng, tot, errs, rate in ngram_stats_sorted[:TOP_K]:
    print("{:<5} '{:<5}' {:<10} {:<10} {:<10.3f}".format(n, ng, tot, errs, rate))

# Save ngram stats to CSV
with open(f"{SAVE_PREFIX}_roman_ngrams.csv", "w", encoding="utf-8", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["n", "ngram", "support", "errors", "error_rate"])
    for (n,ng),tot in sorted(ngram_totals.items(), key=lambda x:(x[0][0], -x[1])):
        errs = ngram_errors.get((n,ng), 0)
        rate = errs/tot
        writer.writerow([n, ng, tot, errs, rate])
print(f"Wrote roman ngram stats to {SAVE_PREFIX}_roman_ngrams.csv")

# --- 5. PER-EXAMPLE EDIT DISTANCE DISTRIBUTION ---

# The simple_levenshtein function is not needed, as the complex one is already in the main script.
edists = [levenshtein(gold, pred) for (_, gold, pred) in all_records]

print("\n--- Levenshtein edit distance over test set: ---")
if edists:
    print(f"Mean: {statistics.mean(edists):.3f}")
    print(f"Median: {statistics.median(edists)}")
    print(f"Max: {max(edists)}")
else:
    print("No predictions were made.")

# --- 6. SAMPLE ERRORS ---

print("\n--- Sample errors (showing up to 30): ---")
count_shown = 0
for src, gold, pred in all_records:
    if gold != pred:
        print(f"SRC: {src:<15}  GOLD: {gold:<15}  PRED: {pred}")
        count_shown += 1
        if count_shown >= 30:
            break

print("\nAnalysis Complete. Check generated CSV files.")