import json
import pandas as pd
import random
import re
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
from difflib import SequenceMatcher
from itertools import zip_longest
import csv
import statistics
import math
import sys
import getopt
from os.path import basename
import xml.dom.minidom
from xml.dom.minidom import Node
import codecs

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

# Decoding/Evaluation Hyperparameters
BEAM_WIDTH = 5               # For Greedy vs. Beam comparison
MAX_DECODE_LEN_COMPARE = 64  # Max length for decoding
N_CANDIDATES_NEWS = 10       # Max candidates for NEWS metrics (ACC, MRR, MAP)

# Special tokens indices
SPECIALS = ["<pad>", "<sos>", "<eos>", "<unk>"]
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3

# Global variable to mimic the command line output flag
OUTPUT_FNAME_FOR_NEWS = None 


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


# --- 3. VOCABULARY AND UTILITIES ---

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

def levenshtein(a, b):
    """Calculates Levenshtein edit distance."""
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

def align_chars(gold, pred):
    """
    Returns list of tuples (op, gold_char, pred_char) for detailed error analysis.
    Uses difflib.SequenceMatcher for a reasonable character alignment.
    """
    sm = SequenceMatcher(None, gold, pred)
    ops = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            for ga, pa in zip(gold[i1:i2], pred[j1:j2]):
                ops.append(('equal', ga, pa))
        elif tag == 'replace':
            for ga, pa in zip_longest(gold[i1:i2], pred[j1:j2], fillvalue=''):
                ops.append(('replace', ga, pa))
        elif tag == 'delete':
            for ga in gold[i1:i2]:
                ops.append(('delete', ga, ''))
        elif tag == 'insert':
            for pa in pred[j1:j2]:
                ops.append(('insert', '', pa))
    return ops


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
    sys.exit()


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


# --- 6. TRAINING AND EVALUATION UTILITIES (UNCHANGED) ---

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

def evaluate_predictions(encoder, decoder, loader, device, max_len=64):
    """
    Greedy decoding evaluation function (Kept for compatibility with original training loop).
    Returns metrics and a list of (src, gold, pred) tuples.
    """
    encoder.eval()
    decoder.eval()
    total, exact_matches, total_chars, correct_chars, total_edit = 0, 0, 0, 0, 0.0
    all_records = []
    samples = []
    
    with torch.no_grad():
        for src, src_lens, _, _, src_texts, tgt_texts in tqdm(loader, desc="Eval decode (Greedy)"):
            src = src.to(device)
            src_lens = src_lens.to(device)
            enc_outputs, _ = encoder(src, src_lens)
            batch_size = src.size(0)

            # --- Greedy Decoding Logic ---
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

            # --- Metric Calculation and Record Keeping ---
            for i in range(batch_size):
                out_ids = outputs_tokens[i].tolist()
                if EOS_IDX in out_ids:
                    cut = out_ids.index(EOS_IDX)
                    out_ids = out_ids[1:cut]
                else:
                    out_ids = out_ids[1:]
                pred = ids_to_seq(out_ids, idx2char)
                gold = tgt_texts[i]
                src_text = src_texts[i]
                
                total += 1
                if pred == gold:
                    exact_matches += 1
                ed = levenshtein(pred, gold)
                total_edit += ed
                total_chars += len(gold)
                correct_chars += max(0, len(gold) - ed)
                
                all_records.append((src_text, gold, pred))

                if len(samples) < 10:
                    samples.append((src_text, gold, pred))

    metrics = {
        "exact_accuracy": exact_matches / total if total>0 else 0.0,
        "mean_edit": total_edit / total if total>0 else 0.0,
        "char_accuracy": correct_chars / total_chars if total_chars>0 else 0.0,
        "samples": samples,
        "all_records": all_records
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

        metrics = evaluate_predictions(encoder, decoder, valid_loader, DEVICE, max_len=MAX_DECODE_LEN_COMPARE) 
        print("Val exact_acc (Greedy): {:.4f}  char_acc: {:.4f}  mean_edit: {:.3f}".format(metrics['exact_accuracy'], metrics['char_accuracy'], metrics['mean_edit']))

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

        teacher_forcing = max(0.1, teacher_forcing * 0.95)


# --- 8. GREEDY/BEAM DECODING & COMPARISON UTILITIES ---

def greedy_decode(encoder, decoder, src, src_lens, char2idx, idx2char, device, max_len=64, **kwargs):
    """Greedy decoding for a batch (returns one prediction per input)."""
    encoder.eval(); decoder.eval();
    batch_size = src.size(0)
    
    with torch.no_grad():
        enc_outputs, _ = encoder(src, src_lens)
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

        preds = []
        for i in range(batch_size):
            out_ids = outputs_tokens[i].tolist()
            if EOS_IDX in out_ids:
                cut = out_ids.index(EOS_IDX)
                out_ids = out_ids[1:cut]
            else:
                out_ids = out_ids[1:]
            preds.append(ids_to_seq(out_ids, idx2char))
        
        return preds

def beam_search_top_n(encoder, decoder, src, src_lens, char2idx, idx2char, device, N=10, max_len=64):
    """
    Performs beam search and returns the top N best predicted sequences for a batch.

    Returns:
        A list of lists. Outer list corresponds to batch items. 
        Inner list contains tuples: (predicted_string, score) sorted by score.
    """
    encoder.eval(); decoder.eval();
    
    with torch.no_grad():
        enc_outputs, _ = encoder(src, src_lens)
        enc_mask = torch.arange(enc_outputs.size(1), device=device).unsqueeze(0) < src_lens.unsqueeze(1)
        batch_size = src.size(0)
        
        h0 = torch.zeros(decoder.rnn.num_layers, batch_size, decoder.rnn.hidden_size, device=device)
        c0 = torch.zeros_like(h0)
        
        all_batch_predictions = []

        for i in range(batch_size):
            single_enc_outputs = enc_outputs[i].unsqueeze(0)
            single_enc_mask = enc_mask[i].unsqueeze(0)
            single_hidden = (h0[:, i, :].unsqueeze(1), c0[:, i, :].unsqueeze(1)) 
            
            beam = [(0.0, [SOS_IDX], single_hidden, False)] # (score, sequence_ids, hidden_state, finished)
            final_predictions = [] 
            
            for _ in range(1, max_len):
                candidates = []
                if not beam: break 

                for score, seq, hidden, finished in beam:
                    if finished:
                        candidates.append((score, seq, hidden, True))
                        continue
                        
                    input_token = torch.tensor([seq[-1]], device=device)
                    
                    logits, new_hidden, _ = decoder.forward_step(input_token, hidden, single_enc_outputs, single_enc_mask)
                    log_probs = torch.log_softmax(logits, dim=1).squeeze(0)
                    topk_log_probs, topk_indices = log_probs.topk(N)

                    for log_prob, idx in zip(topk_log_probs.tolist(), topk_indices.tolist()):
                        new_score = score + log_prob
                        new_seq = seq + [idx]
                        is_finished = (idx == EOS_IDX)
                        candidates.append((new_score, new_seq, new_hidden, is_finished))

                candidates.sort(key=lambda x: x[0], reverse=True)
                beam = candidates[:N]

                new_beam = []
                for score, seq, hidden, finished in beam:
                    if finished:
                        final_predictions.append((score, seq))
                    else:
                        new_beam.append((score, seq, hidden, finished))
                
                beam = new_beam[:N]

                if not beam and final_predictions: break 
                
            final_predictions.extend([(score, seq) for score, seq, _, _ in beam])
            
            unique_final_preds = {} 
            for score, ids in final_predictions:
                if EOS_IDX in ids:
                    cut = ids.index(EOS_IDX)
                    ids = ids[1:cut]
                else:
                    ids = ids[1:]
                pred_str = ids_to_seq(ids, idx2char)
                
                if pred_str:
                    if pred_str not in unique_final_preds or score > unique_final_preds[pred_str]:
                        unique_final_preds[pred_str] = score

            final_list = sorted([(pred_str, score) for pred_str, score in unique_final_preds.items()], 
                                key=lambda x: x[1], 
                                reverse=True)
            
            all_batch_predictions.append(final_list[:N])

    return all_batch_predictions

def beam_search_decode(encoder, decoder, src, src_lens, char2idx, idx2char, device, beam_width=5, max_len=64):
    """Performs beam search and returns the single best prediction (B=beam_width)."""
    # Calls beam_search_top_n and extracts the single best result (index 0)
    top_n_preds = beam_search_top_n(encoder, decoder, src, src_lens, char2idx, idx2char, device, N=beam_width, max_len=max_len)
    return [preds[0][0] if preds else "" for preds in top_n_preds]

def evaluate_decoding_method(encoder, decoder, loader, device, decode_func, max_len=64, **kwargs):
    """Evaluates the model on the loader using a specified decoding function (ACC@1, ED)."""
    encoder.eval(); decoder.eval();
    total, exact_matches, total_chars, correct_chars, total_edit = 0, 0, 0, 0, 0.0
    samples = []
    
    with torch.no_grad():
        for src, src_lens, _, _, src_texts, tgt_texts in tqdm(loader, desc=f"Eval with {decode_func.__name__}"):
            src = src.to(device)
            src_lens = src_lens.to(device)

            preds = decode_func(encoder, decoder, src, src_lens, char2idx, idx2char, device, max_len=max_len, **kwargs)
            
            for i in range(len(preds)):
                pred = preds[i]
                gold = tgt_texts[i]
                
                total += 1
                if pred == gold:
                    exact_matches += 1
                
                ed = levenshtein(pred, gold)
                total_edit += ed
                total_chars += len(gold)
                correct_chars += max(0, len(gold) - ed)

                if len(samples) < 10 and len(samples) < total:
                    samples.append((src_texts[i], gold, pred))

    metrics = {
        "exact_accuracy": exact_matches / total if total>0 else 0.0,
        "mean_edit": total_edit / total if total>0 else 0.0,
        "char_accuracy": correct_chars / total_chars if total_chars>0 else 0.0,
        "samples": samples
    }
    return metrics


# --- 9. NEWS METRICS IMPLEMENTATION (FROM NEWS 2015 PAPER) ---

def LCS_length(s1, s2):
    ''' Calculates the length of the longest common subsequence of s1 and s2. '''
    m = len(s1)
    n = len(s2)
    C = [[0] * (n+1) for i in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]: 
                C[i][j] = C[i-1][j-1] + 1
            else:
                C[i][j] = max(C[i][j-1], C[i-1][j])
    return C[m][n]
    
def f_score(candidate, references):
    ''' Calculates F-score for the candidate and its best matching reference. '''
    best_ref = references[0]
    best_ref_lcs = LCS_length(candidate, references[0])
    for ref in references[1:]:
        lcs = LCS_length(candidate, ref)
        if (len(ref) - 2*lcs) < (len(best_ref) - 2*best_ref_lcs):
            best_ref = ref
            best_ref_lcs = lcs
    
    precision = float(best_ref_lcs)/float(len(candidate)) if len(candidate) else 0.0
    recall = float(best_ref_lcs)/float(len(best_ref)) if len(best_ref) else 0.0
    
    if precision + recall > 0.0:
        return 2*precision*recall/(precision+recall), best_ref
    else:
        return 0.0, best_ref
    
def mean_average_precision(candidates, references, n):
    ''' Calculates mean average precision up to n candidates. '''
    total = 0.0
    num_correct = 0
    for k in range(n):
        if k < len(candidates) and (candidates[k] in references):
            num_correct += 1
        total += float(num_correct)/float(k+1)
        
    return total/float(n)
    
def inverse_rank(candidates, reference):
    ''' Returns inverse rank of the matching candidate given the reference. '''
    rank = 0
    while (rank < len(candidates)) and (candidates[rank] != reference):
        rank += 1
    if rank == len(candidates):
        return 0.0
    else:
        return 1.0/(rank+1)
    
def evaluate(input_data, test_data):
    ''' Evaluates all metrics based on the structures required by the NEWS script. '''
    mrr = {}
    acc = {}
    f = {}
    f_best_match = {}
    map_ref = {}
    
    stderr = codecs.getwriter('utf-8')(sys.stderr)
    
    for src_word in test_data.keys():
        if src_word in input_data:
            candidates = input_data[src_word]
            references = test_data[src_word]
            
            acc[src_word] = max([int(candidates[0] == ref) for ref in references]) 
            f[src_word], f_best_match[src_word] = f_score(candidates[0], references)
            mrr[src_word] = max([inverse_rank(candidates, ref) for ref in references])
            map_ref[src_word] = mean_average_precision(candidates, references, len(references))
            
        else:
            mrr[src_word] = 0.0
            acc[src_word] = 0.0
            f[src_word] = 0.0
            f_best_match[src_word] = ''
            map_ref[src_word] = 0.0
            
    return acc, f, f_best_match, mrr, map_ref
            
def write_details(output_fname, input_data, test_data, acc, f, f_best_match, mrr, map_ref):
    ''' Writes detailed results to CSV file in NEWS format. '''
    try:
        if output_fname == '-':
            f_out = codecs.getwriter('utf-8')(sys.stdout)
        else:
            f_out = codecs.open(output_fname, 'w', 'utf-8')
    except Exception as e:
        sys.stderr.write(f'Error opening output file {output_fname}: {e}\n')
        return
        
    f_out.write('%s\n' % (','.join(['"Source word"', '"First candidate"', '"Top-1"', '"ACC"', '"F-score"', '"Best matching reference"',
    '"MRR"', '"MAP_ref"', '"References"'])))
    
    for src_word in test_data.keys():
        first_candidate = input_data.get(src_word, [''])[0]
        references_disp = test_data.get(src_word, [''])
        
        f_out.write('%s,%s,%f,%f,"%s",%f,%f,"%s"\n' % (
            src_word, 
            first_candidate, 
            acc.get(src_word, 0.0), 
            f.get(src_word, 0.0), 
            f_best_match.get(src_word, ''), 
            mrr.get(src_word, 0.0), 
            map_ref.get(src_word, 0.0), 
            ' | '.join(references_disp)
        ))
    
    if output_fname != '-':
        f_out.close()


def evaluate_news_metrics_local(test_loader, N, device, char2idx, idx2char):
    """
    Calculates NEWS metrics locally by decoding the test set using beam search.
    """
    local_test_data = {}
    input_data_dict = {}
    
    with torch.no_grad():
        for src, src_lens, _, _, src_texts, tgt_texts in tqdm(test_loader, desc=f"NEWS Eval (N={N})"):
            src = src.to(device)
            src_lens = src_lens.to(device)
            
            batch_predictions = beam_search_top_n(encoder, decoder, src, src_lens, 
                                                  char2idx, idx2char, device, N=N, max_len=MAX_DECODE_LEN_COMPARE)
            
            for i in range(len(src_texts)):
                src_word_upper = src_texts[i].upper() 
                gold_text_upper = tgt_texts[i].upper() # NEWS script converts to UPPER
                
                local_test_data[src_word_upper] = [gold_text_upper] 
                
                candidates = [p[0].upper() for p in batch_predictions[i][:N]]
                
                if candidates:
                    input_data_dict[src_word_upper] = candidates
    
    acc, f, f_best_match, mrr, map_ref = evaluate(input_data_dict, local_test_data)
    
    return acc, f, f_best_match, mrr, map_ref, input_data_dict, local_test_data


# --- 10. MAIN EXECUTION BLOCK ---

if __name__ == '__main__':
    # A simplified argument parsing to handle the optional -o flag for NEWS detail output
    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:], 'o:', ['output-file='])
        for o, a in opts:
            if o in ('-o', '--output-file'):
                OUTPUT_FNAME_FOR_NEWS = a
    except getopt.GetoptError as err:
        sys.stderr.write('Error parsing optional output argument: %s\n' % err)

    if MODEL_PATH.exists():
        print(f"\n✅ Checkpoint found at {MODEL_PATH}. Skipping training.")
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        encoder.load_state_dict(ckpt['encoder_state'])
        decoder.load_state_dict(ckpt['decoder_state'])
        print("Loaded model from epoch", ckpt.get('epoch', 'N/A'))
    else:
        print(f"\n❌ Checkpoint NOT found at {MODEL_PATH}. Starting training.")
        train_and_save_model()
        if MODEL_PATH.exists():
            ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
            encoder.load_state_dict(ckpt['encoder_state'])
            decoder.load_state_dict(ckpt['decoder_state'])
            print(f"Loaded newly trained best model from epoch", ckpt.get('epoch', 'N/A'))
        else:
            print("Error: Training completed but no model saved. Cannot proceed to test.")
            sys.exit(1)


    # ----------------------------------------------------------------------
    print("\n" + "="*50)
    print(f"=== Decoding Comparison (Greedy vs. Beam B={BEAM_WIDTH}) ===")

    # 1. Greedy Decoding (Also generates all_records for error analysis)
    greedy_metrics_test = evaluate_predictions(encoder, decoder, test_loader, DEVICE, max_len=MAX_DECODE_LEN_COMPARE)
    all_records = greedy_metrics_test['all_records'] # Data for error analysis

    print("\n--- Greedy Decoding Results (ACC@1) ---")
    print("Exact word accuracy (ACC@1): {:.4f}".format(greedy_metrics_test['exact_accuracy']))
    print("Char accuracy (approx):      {:.4f}".format(greedy_metrics_test['char_accuracy']))
    print("Mean edit distance:          {:.3f}".format(greedy_metrics_test['mean_edit']))

    # 2. Beam Search Decoding (B=5)
    beam_metrics_test = evaluate_decoding_method(encoder, decoder, test_loader, DEVICE, beam_search_decode, max_len=MAX_DECODE_LEN_COMPARE, beam_width=BEAM_WIDTH)

    print(f"\n--- Beam Search (B={BEAM_WIDTH}) Results (ACC@1) ---")
    print("Exact word accuracy (ACC@1): {:.4f}".format(beam_metrics_test['exact_accuracy']))
    print("Char accuracy (approx):      {:.4f}".format(beam_metrics_test['char_accuracy']))
    print("Mean edit distance:          {:.3f}".format(beam_metrics_test['mean_edit']))

    print("\n--- Sample Comparison (Greedy vs. Beam) ---")
    for i in range(min(10, len(greedy_metrics_test['samples']))):
        g_src, g_gold, g_pred = greedy_metrics_test['samples'][i]
        b_src, b_gold, b_pred = beam_metrics_test['samples'][i] 
        print(f"SRC: {g_src:<15} | GOLD: {g_gold:<15} | GREEDY PRED: {g_pred:<15} | BEAM PRED: {b_pred}")
    
    
    # ----------------------------------------------------------------------
    print("\n" + "="*50)
    print(f"=== NEWS Metrics Evaluation (N={N_CANDIDATES_NEWS} Candidates) ===")

    # 3. NEWS Metrics Evaluation (Uses Beam Search for N=10 candidates)
    acc_d, f_d, f_best_match_d, mrr_d, map_ref_d, input_data_d, test_data_d = evaluate_news_metrics_local(
        test_loader, N_CANDIDATES_NEWS, DEVICE, char2idx, idx2char
    )

    N_total = len(acc_d)
    print("\n=== NEWS Metrics Results ===")
    print('ACC (Top-1):  %f' % (float(sum([acc_d[src_word] for src_word in acc_d.keys()]))/N_total))
    print('Mean F-score: %f' % (float(sum([f_d[src_word] for src_word in f_d.keys()]))/N_total))
    print('MRR:          %f' % (float(sum([mrr_d[src_word] for src_word in mrr_d.keys()]))/N_total))
    print('MAP_ref:      %f' % (float(sum([map_ref_d[src_word] for src_word in map_ref_d.keys()]))/N_total))

    # Write CSV details if the -o flag was used
    if OUTPUT_FNAME_FOR_NEWS:
        write_details(OUTPUT_FNAME_FOR_NEWS, input_data_d, test_data_d, acc_d, f_d, f_best_match_d, mrr_d, map_ref_d)
        print(f"\n✅ Wrote detailed NEWS results to {OUTPUT_FNAME_FOR_NEWS}")


    # ----------------------------------------------------------------------
    print("\n" + "="*50)
    print("=== Character-Level Error Analysis (Using Greedy Decoding) ===")
    
    # 4. Character-Level Error Analysis (Recalculate using all_records from greedy_metrics_test)

    # PARAMETERS for the analysis
    MIN_SUPPORT_NGRAM = 20      # minimum number of examples containing an n-gram to consider it
    TOP_K = 30                  # how many top confusions / ngrams to print
    NGRAM_MAX = 3               # consider roman n-grams up to length 3
    SAVE_PREFIX = "lstm_error_analysis" # CSV prefix
    
    confusions = Counter()      
    confusion_by_type = Counter()  
    total_errors = 0

    for src_text, gold, pred in all_records:
        if gold != pred:
            total_errors += 1
        
        ops = align_chars(gold, pred)
        for op, gch, pch in ops:
            if op == 'equal':
                continue
            gkey = gch if gch != '' else '<eps>'
            pkey = pch if pch != '' else '<eps>'
            confusions[(gkey, pkey)] += 1
            confusion_by_type[op] += 1
    
    exact_accuracy = 1 - (total_errors / len(all_records))
    print(f"Exact word accuracy (Recalculated): {exact_accuracy:.4f} ({total_errors} errors)")

    print("\n--- Top character confusions (gold -> predicted), sorted by frequency: ---")
    top_conf = [((g,p),c) for (g,p),c in confusions.most_common() if g != '<eps>'] 
    for (g,p),cnt in top_conf[:TOP_K]:
        print(f"'{g}' -> '{p}'      count={cnt}")

    print("\nSummary of edit operation counts (for non-equal operations):", dict(confusion_by_type))

    # Save confusion table to CSV
    with open(f"{SAVE_PREFIX}_char_confusions.csv", "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["gold_char", "pred_char", "count"])
        for (g,p),cnt in confusions.most_common():
            writer.writerow([g, p, cnt])
    print(f"Wrote char confusions to {SAVE_PREFIX}_char_confusions.csv")
    
    # 5. ROMAN N-GRAM => EXAMPLE-LEVEL ERROR ASSOCIATION
    
    ngram_totals = defaultdict(int)
    ngram_errors = defaultdict(int)

    for src_text, gold, pred in all_records:
        is_err = (gold != pred)
        s = src_text.lower()
        
        for n in range(1, NGRAM_MAX+1):
            seen = set()
            for i in range(len(s)-n+1):
                ng = s[i:i+n]
                if ' ' in ng:
                    continue
                if ng in seen:
                    continue
                seen.add(ng)
                
                ngram_totals[(n,ng)] += 1
                if is_err:
                    ngram_errors[(n,ng)] += 1

    ngram_stats = []
    for (n,ng),tot in ngram_totals.items():
        if tot < MIN_SUPPORT_NGRAM:
            continue
        errs = ngram_errors.get((n,ng), 0)
        rate = errs / tot
        ngram_stats.append((n, ng, tot, errs, rate))
        
    ngram_stats_sorted = sorted(ngram_stats, key=lambda x: (-x[4], -x[2]))

    print(f"\n--- Top Roman n-grams (n<= {NGRAM_MAX}) by word-level error rate (min support = {MIN_SUPPORT_NGRAM}): ---")
    print("{:<5} {:<5} {:<10} {:<10} {:<10}".format("N", "NGRAM", "SUPPORT", "ERRORS", "ERR_RATE"))
    print("-" * 45)
    for n, ng, tot, errs, rate in ngram_stats_sorted[:TOP_K]:
        print("{:<5} '{:<5}' {:<5} {:<5} {:<10.3f}".format(n, ng, tot, errs, rate))

    # Save ngram stats to CSV
    with open(f"{SAVE_PREFIX}_roman_ngrams.csv", "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["n", "ngram", "support", "errors", "error_rate"])
        for (n,ng),tot in sorted(ngram_totals.items(), key=lambda x:(x[0][0], -x[1])):
            errs = ngram_errors.get((n,ng), 0)
            rate = errs/tot
            writer.writerow([n, ng, tot, errs, rate])
    print(f"Wrote roman ngram stats to {SAVE_PREFIX}_roman_ngrams.csv")

    # 6. PER-EXAMPLE EDIT DISTANCE DISTRIBUTION

    edists = [levenshtein(gold, pred) for (_, gold, pred) in all_records]

    print("\n--- Levenshtein edit distance over test set (Greedy): ---")
    if edists:
        print(f"Mean: {statistics.mean(edists):.3f}")
        print(f"Median: {statistics.median(edists)}")
        print(f"Max: {max(edists)}")
    else:
        print("No predictions were made.")

    # 7. SAMPLE ERRORS

    print("\n--- Sample errors (Greedy Decoding, showing up to 30): ---")
    count_shown = 0
    for src, gold, pred in all_records:
        if gold != pred:
            print(f"SRC: {src:<15}  GOLD: {gold:<15}  PRED: {pred}")
            count_shown += 1
            if count_shown >= 30:
                break

    print("\nAnalysis Complete. Check generated CSV files.")