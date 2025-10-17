#!/usr/bin/env python
# coding: utf-8

# In[101]:


import json
import pandas as pd
import random
import re
import unicodedata
from pathlib import Path
import torch.nn.functional as F

random.seed(42)


# In[52]:


# If uploaded manually:
data_dir = Path("data")

train_path = data_dir / "hin_train.json"
valid_path = data_dir / "hin_valid.json"
test_path  = data_dir / "hin_test.json"

print("Train path:", train_path.exists())
print("Valid path:", valid_path.exists())
print("Test path :", test_path.exists())


# In[53]:


def load_json(path):
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith('['):  # JSON array
        data = json.loads(text)
    else:  # JSON Lines
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    return pd.DataFrame(data)

train_df = load_json(train_path)
valid_df = load_json(valid_path)
test_df  = load_json(test_path)

print(f"Train: {len(train_df)}  Valid: {len(valid_df)}  Test: {len(test_df)}")
train_df.head()


# In[54]:


# Cell 4 – RANDOM Clean & Deduplicate

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


# In[55]:


# === Cell Length-stratified subsampling

import pandas as pd

def stratified_by_length(df, n, key='english', n_bins=10, seed=42):
    """
    Stratify by input length to keep coverage of both short and long sequences.
    """
    if len(df) <= n:
        return df
    df = df.copy()
    df['len'] = df[key].str.len()
    # Create quantile bins (length-based)
    df['bin'] = pd.qcut(df['len'], q=n_bins, duplicates='drop')
    samples = []
    for _, g in df.groupby('bin'):
        k = max(1, int(round(len(g) / len(df) * n)))
        samples.append(g.sample(n=min(k, len(g)), random_state=seed))
    sampled = pd.concat(samples).reset_index(drop=True)
    return sampled.sample(n=n, random_state=seed).reset_index(drop=True)

MAX_TRAIN = 100_000
train_df = stratified_by_length(train_df, MAX_TRAIN)
print("Train size after length-stratified subsampling:", len(train_df))


# In[56]:


# Cell 6 – Save Preprocessed Files
out_dir = Path("data/processed")
out_dir.mkdir(exist_ok=True)

def save_jsonl(df, path):
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            obj = {"english": row["english"], "native": row["native"]}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

save_jsonl(train_df, out_dir / "train_clean.jsonl")
save_jsonl(valid_df, out_dir / "valid_clean.jsonl")
save_jsonl(test_df,  out_dir / "test_clean.jsonl")

print("Saved cleaned & stratified data to", out_dir)


# In[57]:


# === Cell E — Character coverage ===
from collections import Counter

src_chars = Counter(''.join(train_df['english']))
tgt_chars = Counter(''.join(train_df['native']))

print(f"English char vocab size: {len(src_chars)}")
print(f"Native  char vocab size: {len(tgt_chars)}")

print("\nTop 20 English chars:")
print(src_chars.most_common(20))

print("\nTop 20 Native chars:")
print(tgt_chars.most_common(20))


# In[58]:


# === Cell B — Histograms of lengths ===
import matplotlib.pyplot as plt

plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
train_df['english'].str.len().hist(bins=30)
plt.title("English (Roman) Length Distribution — Train")
plt.xlabel("Number of characters")
plt.ylabel("Count")

plt.subplot(1,2,2)
train_df['native'].str.len().hist(bins=30)
plt.title("Native (Devanagari) Length Distribution — Train")
plt.xlabel("Number of characters")
plt.ylabel("Count")

plt.tight_layout()
plt.show()


# In[59]:


import math
import random
from pathlib import Path
from collections import Counter
import json
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# Where to save checkpoints
SAVE_DIR = Path("models")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# In[60]:


# Build character-level vocab from current dataframes
SPECIALS = ["<pad>", "<sos>", "<eos>", "<unk>"]

def build_char_vocab_from_dfs(dfs, min_freq=1):
    counter = Counter()
    for df in dfs:
        counter.update(''.join(df['english'].tolist()))
        counter.update(''.join(df['native'].tolist()))
    # keep all chars that appear >= min_freq
    chars = sorted([c for c,f in counter.items() if f >= min_freq], key=lambda x: (-counter[x], x))
    vocab = SPECIALS + chars
    idx2char = {i:c for i,c in enumerate(vocab)}
    char2idx = {c:i for i,c in idx2char.items()}
    return vocab, char2idx, idx2char, counter

vocab, char2idx, idx2char, char_counter = build_char_vocab_from_dfs([train_df, valid_df, test_df])
print("Vocab size (incl specials):", len(vocab))
print("Example chars:", vocab[:20])
PAD_IDX = char2idx["<pad>"]
SOS_IDX = char2idx["<sos>"]
EOS_IDX = char2idx["<eos>"]
UNK_IDX = char2idx["<unk>"]


# In[61]:


# helper: text→ids and ids→text
# Convert sequence to ids (no tokenization further; char-level)
def seq_to_ids(seq, char2idx, add_sos_eos=True):
    ids = [char2idx.get(c, UNK_IDX) for c in seq]
    if add_sos_eos:
        ids = [SOS_IDX] + ids + [EOS_IDX]
    return ids

def ids_to_seq(ids, idx2char, remove_specials=True):
    chars = [idx2char.get(i, "") for i in ids]
    if remove_specials:
        # remove specials
        chars = [c for c in chars if c not in ("<sos>", "<eos>", "<pad>","<unk>")]
    return ''.join(chars)


# In[62]:


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
    # batch: list of tuples (src_ids, tgt_ids, src_text, tgt_text)
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

# create datasets & dataloaders
BATCH_SIZE = 128

train_ds = TransliterationDataset(train_df, char2idx)
valid_ds = TransliterationDataset(valid_df, char2idx)
test_ds  = TransliterationDataset(test_df, char2idx)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=False)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

print("Train batches:", len(train_loader), "Valid batches:", len(valid_loader), "Test batches:", len(test_loader))


# In[63]:


TRF_EMB = 256
TRF_NHEAD = 8
TRF_FF = 512
TRF_ENC_LAYERS = 2   # max 2 per assignment
TRF_DEC_LAYERS = 2   # max 2 per assignment
TRF_DROPOUT = 0.1

TRF_BATCH_SIZE = BATCH_SIZE  # reuse existing BATCH_SIZE or override
TRF_MAX_LEN = 128            # max decode length for generation
LR = 1e-4


# In[64]:


# Positional Encoding (standard)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # if odd, last column will stay zero for cos
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# In[75]:


# Cell T3 — Transformer Seq2Seq model (Embedding + Transformer + Generator)
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

        # init parameters
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_key_padding_mask(self, src):
        # src: (batch, src_seq)
        return (src == self.pad_idx)  # True where pad

    def make_tgt_key_padding_mask(self, tgt):
        return (tgt == self.pad_idx)

    def make_tgt_mask(self, tgt_seq_len, device):
        # causal mask for target (subsequent mask)
        mask = torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device=device), diagonal=1).bool()
        return mask  # True where masked

    def forward(self, src, src_lens, tgt_input):
        """
        src: (batch, src_seq)
        src_lens unused here (we use padding masks)
        tgt_input: (batch, tgt_seq) — should include <sos> at pos 0 and may include pads
        returns logits over vocab for each tgt position
        """
        src_mask = None
        src_key_padding_mask = self.make_src_key_padding_mask(src)  # (batch, src_seq) True=pad
        tgt_key_padding_mask = self.make_tgt_key_padding_mask(tgt_input)
        tgt_mask = self.make_tgt_mask(tgt_input.size(1), device=src.device)  # (tgt_seq, tgt_seq)

        src_emb = self.pos_encoder(self.src_tok_emb(src) * math.sqrt(self.emb_dim))  # (batch, src_seq, emb)
        memory = self.encoder(src_emb, mask=src_mask, src_key_padding_mask=src_key_padding_mask)  # (batch, src_seq, emb)

        tgt_emb = self.pos_encoder(self.tgt_tok_emb(tgt_input) * math.sqrt(self.emb_dim))
        out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_mask=None, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=src_key_padding_mask)
        logits = self.generator(out)  # (batch, tgt_seq, vocab)
        return logits


# In[76]:


# Cell T4 — Instantiate model, loss, optimiser
VOCAB_SIZE = len(vocab)  # reuse from main.py
model_trf = TransformerSeq2Seq(VOCAB_SIZE, TRF_EMB, TRF_NHEAD, TRF_ENC_LAYERS, TRF_DEC_LAYERS, TRF_FF, TRF_DROPOUT, PAD_IDX).to(DEVICE)

criterion_trf = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer_trf = torch.optim.Adam(model_trf.parameters(), lr=LR)
print("Transformer params:", sum(p.numel() for p in model_trf.parameters()))


# In[77]:


# Cell T5 — Train & Eval epoch functions for Transformer
def train_epoch_trf(model, loader, optimizer, criterion, DEVICE):
    model.train()
    total_loss = 0.0
    for src, src_lens, tgt, tgt_lens, _, _ in tqdm(loader, desc="Train TRF"):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        # prepare tgt_input (all tokens except last) and tgt_target (all tokens except first)
        tgt_input = tgt[:, :-1]
        tgt_target = tgt[:, 1:]
        optimizer.zero_grad()
        logits = model(src, src_lens, tgt_input)  # (batch, tgt_len-1, vocab)
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


# In[82]:


# Cell T6 — Greedy decode and Beam search (simple batch-unaware implementations)
@torch.no_grad()
def greedy_decode_trf(model, src, src_lens, max_len=TRF_MAX_LEN):
    # src: (batch, src_seq)
    model.eval()
    batch = src.size(0)
    src = src.to(DEVICE)
    src_lens = src_lens.to(DEVICE)
    # start with sos token
    ys = torch.full((batch, 1), SOS_IDX, dtype=torch.long, device=DEVICE)
    for i in range(max_len - 1):
        logits = model(src, src_lens, ys)  # (batch, tgt_len, vocab)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (batch,1)
        ys = torch.cat([ys, next_token], dim=1)
        # early stop if all sequences generated EOS
        if (next_token == EOS_IDX).all():
            break
    return ys  # include SOS at pos 0

def beam_search_single(model, src, src_lens, beam_width=5, max_len=TRF_MAX_LEN):
    """
    Simple beam search for a single example (not batched).
    Returns token ids (including SOS at pos 0).
    """
    model.eval()
    src = src.unsqueeze(0).to(DEVICE)  # (1, src_seq)
    src_lens = torch.tensor([src_lens], dtype=torch.long, device=DEVICE)
    # initial hypothesis : [ (score, token_seq_tensor) ]
    hyp = [(0.0, torch.tensor([SOS_IDX], dtype=torch.long, device=DEVICE))]

    for _ in range(max_len - 1):
        new_hyp = []
        for score, seq in hyp:
            if seq[-1].item() == EOS_IDX:
                new_hyp.append((score, seq))
                continue
            logits = model(src, src_lens, seq.unsqueeze(0))  # (1, seq_len, vocab)
            log_probs = F.log_softmax(logits[0, -1, :], dim=-1)  # (vocab,)
            topk_logp, topk_idx = torch.topk(log_probs, beam_width)
            for k in range(beam_width):
                nk = topk_idx[k].unsqueeze(0)
                nscore = score + topk_logp[k].item()
                nseq = torch.cat([seq, nk], dim=0)
                new_hyp.append((nscore, nseq))
        # keep top beam_width sequences
        new_hyp = sorted(new_hyp, key=lambda x: x[0], reverse=True)[:beam_width]
        hyp = new_hyp
        # if all last tokens are EOS break
        if all(h[1][-1].item() == EOS_IDX for h in hyp):
            break
    # return best sequence (highest score)
    best_seq = hyp[0][1]
    return best_seq.cpu().tolist()


# In[96]:


# simple Levenshtein distance
def levenshtein(a, b):
    # a,b are str
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
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
    total = 0
    exact_matches = 0
    total_chars = 0
    correct_chars = 0
    total_edit = 0.0
    samples = []
    with torch.no_grad():
        for src, src_lens, tgt, tgt_lens, src_texts, tgt_texts in tqdm(loader, desc="Eval decode"):
            src = src.to(device)
            src_lens = src_lens.to(device)
            enc_outputs, enc_state = encoder(src, src_lens)
            batch_size = src.size(0)
            # greedy decode per sample
            # initialize decoder hidden zeros (same as training)
            h = torch.zeros(decoder.rnn.num_layers, batch_size, decoder.rnn.hidden_size, device=device)
            c = torch.zeros_like(h)
            hidden = (h, c)
            input_token = torch.tensor([SOS_IDX]*batch_size, device=device)
            outputs_tokens = torch.full((batch_size, max_len), PAD_IDX, dtype=torch.long, device=device)
            outputs_tokens[:,0] = SOS_IDX
            enc_mask = torch.arange(enc_outputs.size(1), device=device).unsqueeze(0) < src_lens.unsqueeze(1)
            for t in range(1, max_len):
                logits, hidden, attn = decoder.forward_step(input_token, hidden, enc_outputs, enc_mask)
                top1 = logits.argmax(1)
                outputs_tokens[:, t] = top1
                input_token = top1
            # convert outputs to strings, stop at <eos>
            for i in range(batch_size):
                # find eos
                out_ids = outputs_tokens[i].tolist()
                if EOS_IDX in out_ids:
                    cut = out_ids.index(EOS_IDX)
                    out_ids = out_ids[1:cut]  # remove SOS and after EOS
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
                correct_chars += max(0, len(gold) - ed)  # approx: matched chars = len - edit
                if len(samples) < 10:
                    samples.append((src_texts[i], gold, pred))
    metrics = {
        "exact_accuracy": exact_matches / total if total>0 else 0.0,
        "mean_edit": total_edit / total if total>0 else 0.0,
        "char_accuracy": correct_chars / total_chars if total_chars>0 else 0.0,
        "samples": samples
    }
    return metrics


# In[ ]:


# Cell T7 — Evaluation wrapper to compute exact/char metrics using existing helpers
def evaluate_transformer(model, loader, DEVICE, decode_mode='greedy', beam_width=5, max_len=TRF_MAX_LEN):
    model.eval()
    total = 0
    exact_matches = 0
    total_edit = 0.0
    total_chars = 0
    correct_chars = 0
    samples = []
    with torch.no_grad():
        for src, src_lens, tgt, tgt_lens, src_texts, tgt_texts in tqdm(loader, desc="TRF Eval Decode"):
            batch = src.size(0)
            if decode_mode == 'greedy':
                ys = greedy_decode_trf(model, src, src_lens, max_len=max_len)  # (batch, seq)
                for i in range(batch):
                    out_ids = ys[i].tolist()
                    # find eos
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
            elif decode_mode == 'beam':
                # do beam search per sample (slower)
                for i in range(batch):
                    seq_ids = beam_search_single(model, src[i], src_lens[i].item(), beam_width=beam_width, max_len=max_len)
                    # seq_ids includes SOS
                    if EOS_IDX in seq_ids:
                        cut = seq_ids.index(EOS_IDX)
                        out_ids = seq_ids[1:cut]
                    else:
                        out_ids = seq_ids[1:]
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
            else:
                raise ValueError("Unknown decode_mode")
    metrics = {
        "exact_accuracy": exact_matches / total if total>0 else 0.0,
        "mean_edit": total_edit / total if total>0 else 0.0,
        "char_accuracy": correct_chars / total_chars if total_chars>0 else 0.0,
        "samples": samples
    }
    return metrics


# In[99]:


# Cell T8 — Train loop (Transformer) + checkpointing
N_TRF_EPOCHS = 12
best_val_loss = float('inf')
SAVE_TRF = SAVE_DIR / "transformer_trf.pth" # Ensure SAVE_TRF is defined

# --- Start of Modification ---
if SAVE_TRF.exists():
    print(f"Checkpoint found at {SAVE_TRF}. Skipping training.")
else:
    print(f"No checkpoint found at {SAVE_TRF}. Starting training for {N_TRF_EPOCHS} epochs.")
    # --- Original T8 content indented below ---
    for epoch in range(1, N_TRF_EPOCHS + 1):
        print(f"\n=== TRF Epoch {epoch}/{N_TRF_EPOCHS} ===")
        tr_loss = train_epoch_trf(model_trf, train_loader, optimizer_trf, criterion_trf, DEVICE)
        val_loss = eval_epoch_trf(model_trf, valid_loader, criterion_trf, DEVICE)
        print(f"Train loss: {tr_loss:.4f}  Val loss: {val_loss:.4f}")

        # quick decode metrics (greedy) on validation
        metrics = evaluate_transformer(model_trf, valid_loader, DEVICE, decode_mode='greedy', max_len=TRF_MAX_LEN)
        print("Val exact_acc: {:.4f}  char_acc: {:.4f}  mean_edit: {:.3f}".format(metrics['exact_accuracy'], metrics['char_accuracy'], metrics['mean_edit']))

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

# In[102]:


# Cell T9 — Load best transformer model and final evaluation (greedy & beam)

ckpt_trf = torch.load(SAVE_TRF, map_location=DEVICE)
model_trf.load_state_dict(ckpt_trf['model_state'])
print("Loaded TRF checkpoint epoch", ckpt_trf['epoch'])

print("\n=== Test (greedy) ===")
test_metrics_g = evaluate_transformer(model_trf, test_loader, DEVICE, decode_mode='greedy', max_len=TRF_MAX_LEN)
print("Exact:", test_metrics_g['exact_accuracy'], "Char acc:", test_metrics_g['char_accuracy'], "Mean edit:", test_metrics_g['mean_edit'])

# print("\n=== Test (beam width=5) — this is slower ===")
# test_metrics_b = evaluate_transformer(model_trf, test_loader, DEVICE, decode_mode='beam', beam_width=5, max_len=TRF_MAX_LEN)
# print("Exact (beam5):", test_metrics_b['exact_accuracy'], "Char acc:", test_metrics_b['char_accuracy'], "Mean edit:", test_metrics_b['mean_edit'])

print("\nSample predictions (greedy):")
for s,g,p in test_metrics_g['samples']:
    print(f"SRC: {s}  GOLD: {g}  PRED: {p}")




# === error_analysis.py ===
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from itertools import zip_longest
import csv

# PARAMETERS
MIN_SUPPORT_NGRAM = 20   # minimum number of examples containing an n-gram to consider it
TOP_K = 30               # how many top confusions / ngrams to print
NGRAM_MAX = 3            # consider roman n-grams up to length 3
SAVE_PREFIX = "error_analysis"  # CSV prefix

# helper: convert model greedy output ids -> string using your ids_to_seq/idx2char
# we assume ids_to_seq is available; otherwise define a small helper using idx2char
def ids_to_text_from_model(ids):
    # ids is a list/iterable of ints (including SOS at pos 0). Use ids_to_seq from Transformer.py if available.
    # Fallback: reconstruct using idx2char if ids_to_seq not available.
    try:
        return ids_to_seq(ids, idx2char)   # prefer existing helper
    except Exception:
        chars = [idx2char.get(i, "") for i in ids]
        # remove special tokens
        chars = [c for c in chars if c not in ("<sos>", "<eos>", "<pad>", "<unk>")]
        return ''.join(chars)

# Align gold and pred Devanagari strings and produce operations:
def align_chars(gold, pred):
    """
    Returns list of tuples (op, gold_char, pred_char)
    op in {"equal","replace","delete","insert"}
    gold_char or pred_char may be '' for insert/delete.
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

# 1) Run model on test set and collect predictions + golds
model_trf.eval()
all_records = []  # list of tuples: (src_text, gold_text, pred_text)
with torch.no_grad():
    for src, src_lens, tgt, tgt_lens, src_texts, tgt_texts in test_loader:
        ys = greedy_decode_trf(model_trf, src, src_lens, max_len=128)  # (batch, seq)
        batch = src.size(0)
        for i in range(batch):
            out_ids = ys[i].tolist()
            if EOS_IDX in out_ids:
                cut = out_ids.index(EOS_IDX)
                out_ids = out_ids[1:cut]
            else:
                out_ids = out_ids[1:]
            pred = ids_to_text_from_model(out_ids)
            gold = tgt_texts[i]
            src_text = src_texts[i]
            all_records.append((src_text, gold, pred))

print(f"Total test examples predicted: {len(all_records)}")

# 2) Character-level confusion counts (gold_char -> pred_char)
confusions = Counter()          # (gold_char, pred_char) -> count
confusion_by_type = Counter()   # counts of replace/delete/insert
total_chars = 0
for src_text, gold, pred in all_records:
    ops = align_chars(gold, pred)
    for op, gch, pch in ops:
        total_chars += (1 if op != 'insert' else 0)  # count only positions in gold for accuracy denom
        if op == 'equal':
            continue
        # use placeholder for empty side
        gkey = gch if gch != '' else '<eps>'
        pkey = pch if pch != '' else '<eps>'
        confusions[(gkey, pkey)] += 1
        confusion_by_type[op] += 1

# Print top substitution confusions (excluding pure insertions from pred-only)
print("\nTop character confusions (gold -> predicted), sorted by frequency:")
top_conf = confusions.most_common(TOP_K)
for (g,p),cnt in top_conf:
    print(f"{g!r}  ->  {p!r}   count={cnt}")

print("\nSummary of edit operation counts:", dict(confusion_by_type))

# Save confusion table to CSV
with open(f"{SAVE_PREFIX}_char_confusions.csv", "w", encoding="utf-8", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["gold_char", "pred_char", "count"])
    for (g,p),cnt in confusions.most_common():
        writer.writerow([g, p, cnt])
print(f"Wrote char confusions to {SAVE_PREFIX}_char_confusions.csv")

# 3) Roman n-gram => example-level error association
# For each example, mark if pred != gold (any edit). Then for each ngram in source, update totals/errors.
ngram_totals = defaultdict(int)
ngram_errors = defaultdict(int)
example_errors = []
for src_text, gold, pred in all_records:
    is_err = (gold != pred)
    example_errors.append(is_err)
    s = src_text
    s = s.lower()  # already cleaned earlier, but be safe
    for n in range(1, NGRAM_MAX+1):
        seen = set()
        for i in range(len(s)-n+1):
            ng = s[i:i+n]
            # skip spaces
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

print(f"\nTop roman n-grams (n<= {NGRAM_MAX}) by error rate (min support = {MIN_SUPPORT_NGRAM}):")
for n, ng, tot, errs, rate in ngram_stats_sorted[:TOP_K]:
    print(f"n={n}  {ng!r}  support={tot}  errors={errs}  err_rate={rate:.3f}")

# Save ngram stats to CSV
with open(f"{SAVE_PREFIX}_roman_ngrams.csv", "w", encoding="utf-8", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["n", "ngram", "support", "errors", "error_rate"])
    for (n,ng),tot in sorted(ngram_totals.items(), key=lambda x:(x[0][0], -x[1])):
        errs = ngram_errors.get((n,ng), 0)
        rate = errs/tot
        writer.writerow([n, ng, tot, errs, rate])
print(f"Wrote roman ngram stats to {SAVE_PREFIX}_roman_ngrams.csv")

# 4) Complementary statistic: per-example edit distance distribution
def simple_levenshtein(a,b):
    if a==b: return 0
    la, lb = len(a), len(b)
    prev = list(range(lb+1))
    for i,ca in enumerate(a, start=1):
        cur = [i] + [0]*lb
        for j, cb in enumerate(b, start=1):
            cur[j] = min(cur[j-1]+1, prev[j]+1, prev[j-1] + (0 if ca==cb else 1))
        prev = cur
    return prev[lb]

edists = [simple_levenshtein(gold, pred) for (_, gold, pred) in all_records]
import statistics
print("\nLevenshtein edit distance over test set:")
print("mean:", statistics.mean(edists), "median:", statistics.median(edists), "max:", max(edists))

# Optionally output sample errors for manual inspection (top few)
print("\nSample errors (showing up to 30):")
count_shown = 0
for src, gold, pred in all_records:
    if gold != pred:
        print(f"SRC: {src}  GOLD: {gold}  PRED: {pred}")
        count_shown += 1
        if count_shown >= 30:
            break

print("\nDone. Files produced:")
print(f" - {SAVE_PREFIX}_char_confusions.csv")
print(f" - {SAVE_PREFIX}_roman_ngrams.csv")
