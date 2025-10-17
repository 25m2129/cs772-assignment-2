# streamlit_app.py
# Simple Streamlit UI for your Transformer transliteration model
# Usage: streamlit run streamlit_app.py

import streamlit as st
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

st.set_page_config(page_title="Transliteration demo", layout="centered")

st.title("Transformer transliteration — Inference UI")

MODEL_PATH = st.text_input("Checkpoint path (.pth)", value="models/transformer_trf.pth")
use_gpu = st.checkbox("Use GPU if available", value=False)
device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

@st.cache_resource(show_spinner=False)
def load_model_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    char2idx = ckpt.get('char2idx')
    idx2char = ckpt.get('idx2char')
    if char2idx is None or idx2char is None:
        raise RuntimeError("Checkpoint missing char2idx/idx2char entries.")
    # architecture defaults (match training script; change if needed)
    EMB_DIM = 256
    NHEAD = 8
    FF = 512
    ENC_LAYERS = 2
    DEC_LAYERS = 2
    DROPOUT = 0.1
    PAD_IDX = char2idx["<pad>"]

    # define minimal model class here (same as training architecture)
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=512, dropout=0.1):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            if d_model % 2 == 1:
                pe[:, 1::2] = torch.cos(position * div_term[:-1])
            else:
                pe[:, 1::2] = torch.cos(position * div_term)
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
            out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_mask=None,
                               tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=src_key_padding_mask)
            logits = self.generator(out)
            return logits

    model = TransformerSeq2Seq(len(char2idx), EMB_DIM, NHEAD, ENC_LAYERS, DEC_LAYERS, FF, DROPOUT, PAD_IDX)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    return {"model": model, "char2idx": char2idx, "idx2char": idx2char, "pad_idx": PAD_IDX, "sos": char2idx["<sos>"], "eos": char2idx["<eos>"], "unk": char2idx.get("<unk>", PAD_IDX)}

# UI: load model
st.write("Device:", device)
if st.button("Load model"):
    try:
        data = load_model_checkpoint(MODEL_PATH, device)
        st.success("Model loaded. Enter text below and press Translate.")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# get data (cached)
try:
    data = load_model_checkpoint(MODEL_PATH, device)
except Exception:
    st.info("Enter a valid checkpoint path and press Load model.")
    data = None

# decoding helpers
def encode_text(text, char2idx, sos_idx, eos_idx, unk_idx):
    ids = [char2idx.get(c, unk_idx) for c in text.strip()]
    ids = [sos_idx] + ids + [eos_idx]
    return ids

@torch.no_grad()
def greedy_decode_app(model, src_tensor, src_len, sos_idx, eos_idx, max_len=128):
    batch = src_tensor.size(0)
    src = src_tensor.to(device)
    src_lens = torch.tensor([src_len]*batch, dtype=torch.long, device=device)
    ys = torch.full((batch, 1), sos_idx, dtype=torch.long, device=device)
    for _ in range(max_len-1):
        logits = model(src, src_lens, ys)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_token], dim=1)
        if (next_token == eos_idx).all():
            break
    return ys.cpu().tolist()

# interactive inputs
input_text = st.text_input("English (roman) input to transliterate:", value="namaste")
# Hardcode max_len as the UI element was removed
max_len = 128 

if st.button("Translate") and data is not None:
    char2idx = data["char2idx"]; idx2char = data["idx2char"]
    sos = data["sos"]; eos = data["eos"]; unk = data["unk"]
    ids = encode_text(input_text, char2idx, sos, eos, unk)
    src = torch.tensor([ids], dtype=torch.long)
    # The decoding call remains the same, using the hardcoded max_len
    ys = greedy_decode_app(data["model"], src, len(ids), sos, eos, max_len=max_len)[0] 
    if eos in ys:
        cut = ys.index(eos)
        out_ids = ys[1:cut]
    else:
        out_ids = ys[1:]
    pred = ''.join(idx2char[i] for i in out_ids if idx2char.get(i) not in ("<pad>","<sos>","<eos>","<unk>"))
    st.success(pred)
    st.write("Tokens (ids):", out_ids)
    st.write("Input length:", len(ids))