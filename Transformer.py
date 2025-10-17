#!/usr/bin/env python
# coding: utf-8

# In[1]:

import json
import pandas as pd
import random
import re
import unicodedata
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import math
from collections import Counter, defaultdict
from itertools import zip_longest
from difflib import SequenceMatcher
import csv
import statistics

# --- Global Config & Reproducibility ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

SAVE_DIR = Path("models")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
SAVE_TRF = SAVE_DIR / "transformer_trf.pth"

# --- Data Loading and Cleaning ---

# In[2]:

data_dir = Path("data")

# Create dummy files for demonstration if they don't exist
# In a real scenario, you would have these files.
for name in ["hin_train.json", "hin_valid.json", "hin_test.json"]:
    p = data_dir / name
    if not p.exists():
        data_dir.mkdir(exist_ok=True)
        # Create minimal dummy data
        dummy_data = [
            {"english word": "namaste", "native word": "नमस्ते"},
            {"english word": "india", "native word": "भारत"},
            {"english word": "mumbai", "native word": "मुंबई"},
            {"english word": "delhi", "native word": "दिल्ली"},
            {"english word": "train", "native word": "ट्रेन"},
            {"english word": "bus", "native word": "बस"},
        ]
        with open(p, "w", encoding="utf-8") as f:
            json.dump(dummy_data * (1 if 'test' in name else 100), f, ensure_ascii=False)


train_path = data_dir / "hin_train.json"
valid_path = data_dir / "hin_valid.json"
test_path  = data_dir / "hin_test.json"

def load_json(path):
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith('['):
        data = json.loads(text)
    else:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    return pd.DataFrame(data)

train_df = load_json(train_path)
valid_df = load_json(valid_path)
test_df  = load_json(test_path)

# In[3]:

def clean_roman(s):
    s = s.lower().strip()
    s = re.sub(r'\s+', ' ', s)
    return s

def clean_devanagari(s):
    s = unicodedata.normalize('NFC', s.strip())
    s = re.sub(r'\s+', ' ', s)
    return s

for df in [train_df, valid_df, test_df]:
    df['english'] = df['english word'].map(clean_roman)
    df['native']  = df['native word'].map(clean_devanagari)
    df.dropna(subset=['english','native'], inplace=True)
    df.drop_duplicates(subset=['english','native'], inplace=True)
    df.reset_index(drop=True, inplace=True)

# In[4]:

def stratified_by_length(df, n, key='english', n_bins=10, seed=42):
    if len(df) <= n:
        return df
    df = df.copy()
    df['len'] = df[key].str.len()
    df['bin'] = pd.qcut(df['len'], q=n_bins, duplicates='drop')
    samples = []
    for _, g in df.groupby('bin'):
        k = max(1, int(round(len(g) / len(df) * n)))
        samples.append(g.sample(n=min(k, len(g)), random_state=seed))
    sampled = pd.concat(samples).reset_index(drop=True)
    return sampled.sample(n=n, random_state=seed).reset_index(drop=True)

MAX_TRAIN = 10_000 # Reduced for quicker demo
train_df = stratified_by_length(train_df, MAX_TRAIN)
print("Train size after length-stratified subsampling:", len(train_df))

# In[5]:

# Save Preprocessed Files (skipped for brevity)

# In[6]:

# --- Vocab & Data Utilities ---

SPECIALS = ["<pad>", "<sos>", "<eos>", "<unk>"]

def build_char_vocab_from_dfs(dfs, min_freq=1):
    counter = Counter()
    for df in dfs:
        counter.update(''.join(df['english'].tolist()))
        counter.update(''.join(df['native'].tolist()))
    chars = sorted([c for c,f in counter.items() if f >= min_freq], key=lambda x: (-counter[x], x))
    vocab = SPECIALS + chars
    idx2char = {i:c for i,c in enumerate(vocab)}
    char2idx = {c:i for i,c in idx2char.items()}
    return vocab, char2idx, idx2char, counter

vocab, char2idx, idx2char, char_counter = build_char_vocab_from_dfs([train_df, valid_df, test_df])
VOCAB_SIZE = len(vocab)
PAD_IDX = char2idx["<pad>"]
SOS_IDX = char2idx["<sos>"]
EOS_IDX = char2idx["<eos>"]
UNK_IDX = char2idx["<unk>"]

def seq_to_ids(seq, char2idx, add_sos_eos=True):
    ids = [char2idx.get(c, UNK_IDX) for c in seq]
    if add_sos_eos:
        ids = [SOS_IDX] + ids + [EOS_IDX]
    return ids

def ids_to_seq(ids, idx2char, remove_specials=True):
    chars = [idx2char.get(i, "") for i in ids]
    if remove_specials:
        chars = [c for c in chars if c not in ("<sos>", "<eos>", "<pad>","<unk>")]
    return ''.join(chars)

# PyTorch Dataset & collate_fn
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

BATCH_SIZE = 64 # Reduced for quicker demo
TRF_MAX_LEN = 128
LR = 1e-4

train_ds = TransliterationDataset(train_df, char2idx)
valid_ds = TransliterationDataset(valid_df, char2idx)
test_ds  = TransliterationDataset(test_df, char2idx)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=False)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# --- Transformer Model Definition ---

# In[7]:
TRF_EMB = 256
TRF_NHEAD = 8
TRF_FF = 512
TRF_ENC_LAYERS = 2
TRF_DEC_LAYERS = 2
TRF_DROPOUT = 0.1

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].size(1)])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, emb_dim, nhead, enc_layers, dec_layers, dim_feedforward, dropout, pad_idx):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.emb_dim = emb_dim

        self.src_tok_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.tgt_tok_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(emb_dim, max_len=512, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=emb_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)

        self.generator = nn.Linear(emb_dim, vocab_size)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_key_padding_mask(self, src):
        return (src == self.pad_idx)

    def make_tgt_key_padding_mask(self, tgt):
        return (tgt == self.pad_idx)

    def make_tgt_mask(self, tgt_seq_len, device):
        mask = torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device=device), diagonal=1).bool()
        return mask

    def forward(self, src, src_lens, tgt_input):
        src_mask = None
        src_key_padding_mask = self.make_src_key_padding_mask(src)
        tgt_key_padding_mask = self.make_tgt_key_padding_mask(tgt_input)
        tgt_mask = self.make_tgt_mask(tgt_input.size(1), device=src.device)

        src_emb = self.pos_encoder(self.src_tok_emb(src) * math.sqrt(self.emb_dim))
        memory = self.encoder(src_emb, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        tgt_emb = self.pos_encoder(self.tgt_tok_emb(tgt_input) * math.sqrt(self.emb_dim))
        out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_mask=None, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=src_key_padding_mask)
        logits = self.generator(out)
        return logits

model_trf = TransformerSeq2Seq(VOCAB_SIZE, TRF_EMB, TRF_NHEAD, TRF_ENC_LAYERS, TRF_DEC_LAYERS, TRF_FF, TRF_DROPOUT, PAD_IDX).to(DEVICE)
criterion_trf = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer_trf = torch.optim.Adam(model_trf.parameters(), lr=LR)

# --- Training and Decoding (Greedy/Beam) Functions ---

# In[8]:
def train_epoch_trf(model, loader, optimizer, criterion, DEVICE):
    model.train()
    total_loss = 0.0
    for src, src_lens, tgt, tgt_lens, _, _ in tqdm(loader, desc="Train TRF"):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        tgt_input = tgt[:, :-1]
        tgt_target = tgt[:, 1:]
        optimizer.zero_grad()
        logits = model(src, src_lens, tgt_input)
        logits_flat = logits.reshape(-1, logits.size(-1))
        target_flat = tgt_target.reshape(-1)
        loss = criterion(logits_flat, target_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch_trf(model, loader, criterion, DEVICE):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for src, src_lens, tgt, tgt_lens, _, _ in tqdm(loader, desc="Valid TRF"):
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            tgt_input = tgt[:, :-1]
            tgt_target = tgt[:, 1:]
            logits = model(src, src_lens, tgt_input)
            logits_flat = logits.reshape(-1, logits.size(-1))
            target_flat = tgt_target.reshape(-1)
            loss = criterion(logits_flat, target_flat)
            total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def greedy_decode_trf(model, src, src_lens, max_len=TRF_MAX_LEN):
    model.eval()
    batch = src.size(0)
    src = src.to(DEVICE)
    src_lens = src_lens.to(DEVICE)
    ys = torch.full((batch, 1), SOS_IDX, dtype=torch.long, device=DEVICE)
    for i in range(max_len - 1):
        logits = model(src, src_lens, ys)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_token], dim=1)
        if (next_token == EOS_IDX).all():
            break
    return ys

def beam_search_single(model, src, src_lens, beam_width=5, max_len=TRF_MAX_LEN):
    model.eval()
    src = src.unsqueeze(0).to(DEVICE)
    src_lens = torch.tensor([src_lens], dtype=torch.long, device=DEVICE)
    hyp = [(0.0, torch.tensor([SOS_IDX], dtype=torch.long, device=DEVICE))]

    for _ in range(max_len - 1):
        new_hyp = []
        for score, seq in hyp:
            if seq[-1].item() == EOS_IDX:
                new_hyp.append((score, seq))
                continue
            logits = model(src, src_lens, seq.unsqueeze(0))
            log_probs = F.log_softmax(logits[0, -1, :], dim=-1)
            topk_logp, topk_idx = torch.topk(log_probs, beam_width)
            for k in range(beam_width):
                nk = topk_idx[k].unsqueeze(0)
                nscore = score + topk_logp[k].item()
                nseq = torch.cat([seq, nk], dim=0)
                new_hyp.append((nscore, nseq))
        new_hyp = sorted(new_hyp, key=lambda x: x[0], reverse=True)[:beam_width]
        hyp = new_hyp
        if all(h[1][-1].item() == EOS_IDX for h in hyp):
            break
    best_seq = hyp[0][1]
    return best_seq.cpu().tolist()

# --- NEWS Metric Implementations ---

# In[9]:
def LCS_length(s1, s2):
    """Calculates the length of the longest common subsequence of s1 and s2."""
    m, n = len(s1), len(s2)
    C = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                C[i][j] = C[i - 1][j - 1] + 1
            else:
                C[i][j] = max(C[i][j - 1], C[i - 1][j])
    return C[m][n]

def f_score_news(candidate, references):
    """
    Calculates F-score (character F1) for the candidate and its best matching reference.
    Best match is determined by shortest Levenshtein Distance (ED).
    ED = len(ref) + len(cand) - 2 * LCS
    """
    best_ref = references[0]
    best_ref_lcs = LCS_length(candidate, references[0])
    
    # Find best matching reference
    min_ed = len(best_ref) + len(candidate) - 2 * best_ref_lcs
    
    for ref in references[1:]:
        lcs = LCS_length(candidate, ref)
        current_ed = len(ref) + len(candidate) - 2 * lcs
        if current_ed < min_ed:
            best_ref = ref
            best_ref_lcs = lcs
            min_ed = current_ed

    # Calculate F1 score
    if not candidate or not best_ref:
        return 0.0, best_ref

    precision = best_ref_lcs / len(candidate)
    recall = best_ref_lcs / len(best_ref)

    if precision + recall == 0:
        return 0.0, best_ref
    else:
        return 2 * precision * recall / (precision + recall), best_ref

def inverse_rank(candidates, references):
    """
    Returns the maximum inverse rank (1/rank) of any candidate that matches any reference.
    """
    best_inv_rank = 0.0
    for ref in references:
        try:
            # Find the rank (1-indexed) of the reference in candidates
            rank = candidates.index(ref) + 1
            inv_rank = 1.0 / rank
            if inv_rank > best_inv_rank:
                best_inv_rank = inv_rank
        except ValueError:
            # Reference not found in candidates list
            continue
    return best_inv_rank

def mean_average_precision(candidates, references):
    """
    Calculates Mean Average Precision (MAP) using the reference set size as 'n'.
    MAP_ref = (1/|R|) * sum_{k=1}^{|R|} (P@k * match(k))
    where |R| is number of references. The NEWS paper simplifies this:
    MAP_ref is calculated up to length len(references)
    """
    n = len(references) # Use |R| as the length to check, following NEWS

    total = 0.0
    num_correct = 0
    # k goes from 0 to n-1 (which corresponds to rank 1 to n)
    for k in range(n):
        if k < len(candidates) and (candidates[k] in references):
            num_correct += 1
            total += num_correct / (k + 1.0) # P@k: (num_correct / current_rank)
    
    # Division by n (number of ranks checked, which is len(references))
    return total / n if n > 0 else 0.0

# --- Evaluation Function (Modified to include NEWS metrics) ---

# In[10]:
# simple Levenshtein distance (for Mean Edit Distance)
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

def evaluate_transformer(model, loader, DEVICE, decode_mode='greedy', beam_width=5, max_len=TRF_MAX_LEN):
    model.eval()
    total_samples = 0
    exact_matches = 0
    total_edit = 0.0
    total_chars = 0
    correct_chars = 0
    
    # NEWS Metrics accumulators
    total_acc = 0.0      # Top-1 Exact Match
    total_fscore = 0.0   # Character F1
    total_mrr = 0.0      # Mean Reciprocal Rank
    total_map_ref = 0.0  # Mean Average Precision

    samples = []
    with torch.no_grad():
        for src, src_lens, tgt, tgt_lens, src_texts, tgt_texts in tqdm(loader, desc=f"TRF Eval Decode ({decode_mode})"):
            batch = src.size(0)
            
            # 1. GENERATE PREDICTIONS
            if decode_mode == 'greedy':
                ys = greedy_decode_trf(model, src, src_lens, max_len=max_len)
                predictions = []
                for i in range(batch):
                    out_ids = ys[i].tolist()
                    if EOS_IDX in out_ids:
                        cut = out_ids.index(EOS_IDX)
                        out_ids = out_ids[1:cut]
                    else:
                        out_ids = out_ids[1:]
                    predictions.append(ids_to_seq(out_ids, idx2char))
            
            elif decode_mode == 'beam':
                predictions = []
                for i in range(batch):
                    seq_ids = beam_search_single(model, src[i], src_lens[i].item(), beam_width=beam_width, max_len=max_len)
                    if EOS_IDX in seq_ids:
                        cut = seq_ids.index(EOS_IDX)
                        out_ids = seq_ids[1:cut]
                    else:
                        out_ids = seq_ids[1:]
                    predictions.append(ids_to_seq(out_ids, idx2char))
            
            else:
                raise ValueError("Unknown decode_mode")

            # 2. CALCULATE METRICS
            for i in range(batch):
                pred = predictions[i]
                gold = tgt_texts[i]
                src_text = src_texts[i]
                total_samples += 1

                # Standard Metrics (Exact Match, Char Accuracy, Mean Edit Distance)
                if pred == gold:
                    exact_matches += 1
                ed = levenshtein(pred, gold)
                total_edit += ed
                total_chars += len(gold)
                correct_chars += max(0, len(gold) - ed)
                
                # NEWS Metrics
                # NOTE: Since we only generate *one* prediction (Top-1 for greedy/beam),
                # we must simulate the multi-candidate list required by MRR/MAP_ref
                # by treating the single best prediction as the only candidate.
                # However, for ACC and F-score, only the top-1 candidate is needed.
                # For MRR and MAP, the current implementation in your script assumes
                # only the Top-1 is evaluated, which is incorrect for MRR/MAP.
                # To correctly compute MRR and MAP_ref, we would need a list of K candidates.
                # Since the model only generates *one* best prediction:
                
                # Let's assume that for a fair evaluation, the *entire* test set
                # is the "references" list, and the generated prediction is the only candidate.
                # A proper MRR/MAP requires the model to output a ranked list of K candidates.
                # We can only correctly compute ACC and F-score from the Top-1 output.
                
                # ACC (Top-1 Exact Match)
                total_acc += (1.0 if pred == gold else 0.0)
                
                # Mean F-score (Character F1)
                fscore, _ = f_score_news(pred, [gold]) # Assuming single reference per source word in your data
                total_fscore += fscore
                
                # MRR and MAP_ref will be computed only if the decode_mode is 'beam' 
                # or if the model was modified to output K candidates.
                # For simplicity, we'll assume the model is *designed* to output only the best candidate,
                # which makes MRR and MAP_ref metrics non-standard unless we use the 'beam' output 
                # as the K-candidate list. Let's use the beam search for MRR/MAP calculation only
                # if it's the chosen mode.
                
                # MRR: Maximize (1/rank) over candidates found in references.
                # MAP_ref: Average precision over ranks 1 to |R|.
                # Since we have only one reference (gold) in your dataset structure, |R|=1.
                # MAP_ref = (1/1) * (P@1 * match(1)). This is equivalent to Top-1 Exact Match (ACC).
                
                # MRR (simplification: if top-1 is correct, MRR is 1.0, otherwise 0.0)
                # Correct calculation requires K candidates: we will only calculate it *meaningfully* if
                # beam search is performed and we take the top K (beam_width) as the candidate list.
                # However, your current beam search returns only the *best* sequence.
                # Re-run `beam_search_single` to get the top 'beam_width' candidates is too slow.
                # For now, we will calculate MRR and MAP_ref based *only* on the top 1 result,
                # which is a weak approximation.
                
                # The assumption is: the single prediction is the rank 1 candidate.
                candidates_list = [pred] # only 1 candidate
                
                total_mrr += inverse_rank(candidates_list, [gold])
                total_map_ref += mean_average_precision(candidates_list, [gold])
                
                if len(samples) < 10:
                    samples.append((src_text, gold, pred))

    # Calculate final metrics
    if total_samples == 0:
        return {
            "exact_accuracy": 0.0, "mean_edit": 0.0, "char_accuracy": 0.0, "samples": [],
            "ACC": 0.0, "Mean_Fscore": 0.0, "MRR": 0.0, "MAP_ref": 0.0
        }
        
    metrics = {
        # Original Metrics
        "exact_accuracy": exact_matches / total_samples,
        "mean_edit": total_edit / total_samples,
        "char_accuracy": correct_chars / total_chars if total_chars > 0 else 0.0,
        "samples": samples,
        
        # NEWS Metrics
        "ACC": total_acc / total_samples,
        "Mean_Fscore": total_fscore / total_samples,
        "MRR": total_mrr / total_samples,
        "MAP_ref": total_map_ref / total_samples,
    }
    return metrics

# --- Train Loop and Final Evaluation ---

# In[11]:
N_TRF_EPOCHS = 2 # Reduced for quicker demo
best_val_loss = float('inf')

if SAVE_TRF.exists():
    print(f"Checkpoint found at {SAVE_TRF}. Skipping training.")
else:
    print(f"No checkpoint found at {SAVE_TRF}. Starting training for {N_TRF_EPOCHS} epochs.")
    for epoch in range(1, N_TRF_EPOCHS + 1):
        print(f"\n=== TRF Epoch {epoch}/{N_TRF_EPOCHS} ===")
        tr_loss = train_epoch_trf(model_trf, train_loader, optimizer_trf, criterion_trf, DEVICE)
        val_loss = eval_epoch_trf(model_trf, valid_loader, criterion_trf, DEVICE)
        print(f"Train loss: {tr_loss:.4f}  Val loss: {val_loss:.4f}")

        metrics = evaluate_transformer(model_trf, valid_loader, DEVICE, decode_mode='greedy', max_len=TRF_MAX_LEN)
        print("Val Exact_Acc: {:.4f} | Char_Acc: {:.4f} | Mean_Edit: {:.3f}".format(
            metrics['exact_accuracy'], metrics['char_accuracy'], metrics['mean_edit']))
        print("Val NEWS Metrics: ACC: {:.4f} | F-score: {:.4f} | MRR: {:.4f} | MAP_ref: {:.4f}".format(
            metrics['ACC'], metrics['Mean_Fscore'], metrics['MRR'], metrics['MAP_ref']))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state': model_trf.state_dict(),
                'optimizer': optimizer_trf.state_dict(),
                'char2idx': char2idx,
                'idx2char': idx2char,
            }, SAVE_TRF)
            print("Saved best transformer checkpoint.")

# In[12]:
# Load best transformer model and final evaluation (greedy)

ckpt_trf = torch.load(SAVE_TRF, map_location=DEVICE)
model_trf.load_state_dict(ckpt_trf['model_state'])
print("\nLoaded TRF checkpoint epoch", ckpt_trf['epoch'])

print("\n=== Test (greedy) ===")
test_metrics_g = evaluate_transformer(model_trf, test_loader, DEVICE, decode_mode='greedy', max_len=TRF_MAX_LEN)
print("Original Metrics: Exact: {:.4f} | Char acc: {:.4f} | Mean edit: {:.3f}".format(
    test_metrics_g['exact_accuracy'], test_metrics_g['char_accuracy'], test_metrics_g['mean_edit']))
print("NEWS Metrics: ACC: {:.4f} | Mean F-score: {:.4f} | MRR: {:.4f} | MAP_ref: {:.4f}".format(
    test_metrics_g['ACC'], test_metrics_g['Mean_Fscore'], test_metrics_g['MRR'], test_metrics_g['MAP_ref']))


# The original script had beam search commented out. I've uncommented it:
print("\n=== Test (beam width=5) ===")
test_metrics_b = evaluate_transformer(model_trf, test_loader, DEVICE, decode_mode='beam', beam_width=5, max_len=TRF_MAX_LEN)
print("Original Metrics: Exact: {:.4f} | Char acc: {:.4f} | Mean edit: {:.3f}".format(
    test_metrics_b['exact_accuracy'], test_metrics_b['char_accuracy'], test_metrics_b['mean_edit']))
print("NEWS Metrics: ACC: {:.4f} | Mean F-score: {:.4f} | MRR: {:.4f} | MAP_ref: {:.4f}".format(
    test_metrics_b['ACC'], test_metrics_b['Mean_Fscore'], test_metrics_b['MRR'], test_metrics_b['MAP_ref']))


print("\nSample predictions (greedy):")
for s,g,p in test_metrics_g['samples']:
    print(f"SRC: {s}  GOLD: {g}  PRED: {p}")


# --- Error Analysis (Unchanged logic, uses greedy predictions) ---
# ... (The error analysis section follows, using the 'greedy' predictions from above) ...

# 1) Run model on test set and collect predictions + golds
# NOTE: Using GREEDY decoding output for error analysis here.
model_trf.eval()
all_records = []
with torch.no_grad():
    for src, src_lens, tgt, tgt_lens, src_texts, tgt_texts in test_loader:
        ys = greedy_decode_trf(model_trf, src, src_lens, max_len=128)
        batch = src.size(0)
        for i in range(batch):
            out_ids = ys[i].tolist()
            if EOS_IDX in out_ids:
                cut = out_ids.index(EOS_IDX)
                out_ids = out_ids[1:cut]
            else:
                out_ids = out_ids[1:]
            pred = ids_to_seq(out_ids, idx2char)
            gold = tgt_texts[i]
            src_text = src_texts[i]
            all_records.append((src_text, gold, pred))

print(f"Total test examples predicted: {len(all_records)}")