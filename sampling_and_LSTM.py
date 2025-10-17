#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import pandas as pd
import random
import re
import unicodedata
from pathlib import Path

random.seed(42)


# In[3]:


# If uploaded manually:
data_dir = Path("data")

train_path = data_dir / "hin_train.json"
valid_path = data_dir / "hin_valid.json"
test_path  = data_dir / "hin_test.json"

print("Train path:", train_path.exists())
print("Valid path:", valid_path.exists())
print("Test path :", test_path.exists())


# In[4]:


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


# In[5]:


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


# In[6]:


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


# In[8]:


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


# In[9]:


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


# In[10]:


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


# In[11]:


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


# In[12]:


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


# In[13]:


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


# In[14]:


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


# In[15]:


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_hidden, n_layers=1, dropout=0.1, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.rnn = nn.LSTM(emb_dim, enc_hidden, num_layers=n_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout if n_layers>1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        self.enc_hidden = enc_hidden
        self.n_layers = n_layers

    def forward(self, src, src_lens):
        # src: (batch, seq)
        embedded = self.dropout(self.embedding(src))  # (batch, seq, emb)
        # pack
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # (batch, seq, hidden*directions)
        # combine bidirectional hidden states if needed
        if self.bidirectional:
            # h_n shape: (num_layers*2, batch, enc_hidden)
            # we will concat the forward & backward for each layer when needed in decoder init
            pass
        return outputs, (h_n, c_n)

class LuongAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        # for 'dot' we don't need parameters if dims match. We'll ensure dims match by projecting decoder hidden
        self.attn = nn.Linear(dec_dim, enc_dim, bias=False)  # project dec hidden to enc dim

    def forward(self, dec_hidden, enc_outputs, mask=None):
        # dec_hidden: (batch, dec_dim) for current step
        # enc_outputs: (batch, seq, enc_dim)
        # project dec hidden
        proj = self.attn(dec_hidden).unsqueeze(2)  # (batch, enc_dim, 1)
        # scores: (batch, seq, 1)
        scores = torch.bmm(enc_outputs, proj).squeeze(2)  # (batch, seq)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        attn_weights = torch.softmax(scores, dim=1)  # (batch, seq)
        context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1)  # (batch, enc_dim)
        return context, attn_weights

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_hidden, dec_hidden, n_layers=1, dropout=0.1, bidirectional_encoder=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.rnn = nn.LSTM(emb_dim + (enc_hidden * (2 if bidirectional_encoder else 1)), dec_hidden, num_layers=n_layers, batch_first=True, dropout=dropout if n_layers>1 else 0.0)
        self.attention = LuongAttention(enc_hidden * (2 if bidirectional_encoder else 1), dec_hidden)
        self.out = nn.Linear(dec_hidden + (enc_hidden * (2 if bidirectional_encoder else 1)) + emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.dec_hidden_dim = dec_hidden

    def forward_step(self, input_token, last_hidden, enc_outputs, enc_mask):
        # input_token: (batch,) integer token (current step)
        embedded = self.embedding(input_token).unsqueeze(1)  # (batch, 1, emb)
        # last_hidden: (h, c) where h: (num_layers, batch, dec_hidden)
        h_last = last_hidden[0][-1]  # (batch, dec_hidden) - top layer hidden
        context, attn_weights = self.attention(h_last, enc_outputs, mask=enc_mask)  # context: (batch, enc_dim)
        # combine embedded and context -> input to rnn
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)  # (batch,1, emb+enc_dim)
        output, (h, c) = self.rnn(rnn_input, last_hidden)  # output: (batch,1,dec_hidden)
        output = output.squeeze(1)  # (batch, dec_hidden)
        output = self.dropout(output)
        # concat output, context, embedded to predict
        concat = torch.cat([output, context, embedded.squeeze(1)], dim=1)  # (batch, dec+enc+emb)
        logits = self.out(concat)
        return logits, (h, c), attn_weights

    def forward(self, tgt, enc_outputs, enc_lens, teacher_forcing_ratio=0.5):
        # tgt: (batch, tgt_seq)
        batch_size = tgt.size(0)
        max_tgt = tgt.size(1)
        device = tgt.device

        # prepare enc mask
        enc_mask = torch.arange(enc_outputs.size(1), device=device).unsqueeze(0) < enc_lens.unsqueeze(1)  # (batch, seq)

        outputs = torch.zeros(batch_size, max_tgt, self.out.out_features, device=device)

        # init decoder hidden with zeros (or could transform encoder states)
        # For simplicity, initialize decoder hidden to zeros
        h0 = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=device)
        c0 = torch.zeros_like(h0)
        hidden = (h0, c0)

        input_token = tgt[:,0]  # assume tgt contains <sos> at pos 0
        for t in range(1, max_tgt):
            logits, hidden, attn = self.forward_step(input_token, hidden, enc_outputs, enc_mask)
            outputs[:, t, :] = logits
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = logits.argmax(1)
            input_token = tgt[:, t] if teacher_force else top1
        return outputs


# In[16]:


VOCAB_SIZE = len(vocab)
EMB_DIM = 128
ENC_HIDDEN = 256
DEC_HIDDEN = 256
ENC_N_LAYERS = 2
DEC_N_LAYERS = 2
DROPOUT = 0.2
BIDIRECTIONAL = True

encoder = Encoder(VOCAB_SIZE, EMB_DIM, ENC_HIDDEN, n_layers=ENC_N_LAYERS, dropout=DROPOUT, bidirectional=BIDIRECTIONAL).to(DEVICE)
decoder = Decoder(VOCAB_SIZE, EMB_DIM, ENC_HIDDEN, DEC_HIDDEN, n_layers=DEC_N_LAYERS, dropout=DROPOUT, bidirectional_encoder=BIDIRECTIONAL).to(DEVICE)

# Loss (ignore pad)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(params, lr=0.001)

print("Model parameters:", sum(p.numel() for p in params))


# In[17]:


def train_epoch(encoder, decoder, loader, optimizer, criterion, device, teacher_forcing=0.5):
    encoder.train()
    decoder.train()
    epoch_loss = 0
    for src, src_lens, tgt, tgt_lens, _, _ in tqdm(loader, desc="Train batches"):
        src = src.to(device)
        src_lens = src_lens.to(device)
        tgt = tgt.to(device)
        tgt_lens = tgt_lens.to(device)

        optimizer.zero_grad()
        enc_outputs, enc_state = encoder(src, src_lens)  # enc_outputs: (batch, seq, enc_dim*dir)
        outputs = decoder(tgt, enc_outputs, src_lens, teacher_forcing_ratio=teacher_forcing)  # (batch, tgt_len, vocab)
        # shift outputs & targets: we want to predict tokens at t given inputs up to t-1
        # outputs[:, t, :] corresponds to logits for token t
        out_flat = outputs[:,1:,:].reshape(-1, outputs.size(-1))
        tgt_flat = tgt[:,1:].reshape(-1)
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
            tgt_lens = tgt_lens.to(device)
            enc_outputs, enc_state = encoder(src, src_lens)
            outputs = decoder(tgt, enc_outputs, src_lens, teacher_forcing_ratio=0.0)  # no teacher forcing
            out_flat = outputs[:,1:,:].reshape(-1, outputs.size(-1))
            tgt_flat = tgt[:,1:].reshape(-1)
            loss = criterion(out_flat, tgt_flat)
            epoch_loss += loss.item()
    return epoch_loss / len(loader)


# In[18]:


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


# In[19]:


N_EPOCHS = 12
best_valid_loss = float('inf')
teacher_forcing = 0.5

for epoch in range(1, N_EPOCHS+1):
    print(f"\n=== Epoch {epoch}/{N_EPOCHS} ===")
    train_loss = train_epoch(encoder, decoder, train_loader, optimizer, criterion, DEVICE, teacher_forcing=teacher_forcing)
    valid_loss = eval_epoch(encoder, decoder, valid_loader, criterion, DEVICE)
    print(f"Train loss: {train_loss:.4f}  Valid loss: {valid_loss:.4f}")
    # eval sample metrics on validation
    metrics = evaluate_predictions(encoder, decoder, valid_loader, DEVICE, max_len=64)
    print("Val exact_acc: {:.4f}  char_acc: {:.4f}  mean_edit: {:.3f}".format(metrics['exact_accuracy'], metrics['char_accuracy'], metrics['mean_edit']))
    # save best
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save({
            'epoch': epoch,
            'encoder_state': encoder.state_dict(),
            'decoder_state': decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'char2idx': char2idx,
            'idx2char': idx2char,
        }, SAVE_DIR / "seq2seq_lstm.pth")
        print("Saved best model.")
    # optional: decay teacher forcing gradually
    teacher_forcing = max(0.1, teacher_forcing * 0.95)


# In[20]:


# Load best model if you want )
ckpt = torch.load(SAVE_DIR / "seq2seq_lstm.pth", map_location=DEVICE)
encoder.load_state_dict(ckpt['encoder_state'])
decoder.load_state_dict(ckpt['decoder_state'])
print("Loaded checkpoint epoch", ckpt['epoch'])

test_metrics = evaluate_predictions(encoder, decoder, test_loader, DEVICE, max_len=64)
print("\n=== Test results ===")
print("Exact word accuracy:", test_metrics['exact_accuracy'])
print("Char accuracy (approx):", test_metrics['char_accuracy'])
print("Mean edit distance:", test_metrics['mean_edit'])

print("\nSample predictions:")
for s, gold, pred in test_metrics['samples']:
    print(f"SRC: {s}  GOLD: {gold}  PRED: {pred}")





