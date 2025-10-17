# lstm_streamlit_simple.py (Corrected)

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import random 

# --- Set up Streamlit page ---
st.set_page_config(page_title="LSTM Transliteration demo", layout="centered")
st.title("Seq2Seq LSTM Transliteration — Inference UI")

# --- Configuration (must match training script) ---
# Checkpoint path and device setup
# FIX: Added a unique key='model_path_input' to resolve the DuplicateElementId error
MODEL_PATH = st.text_input("Checkpoint path (.pth)", value="models/seq2seq_lstm.pth", key='model_path_input')
use_gpu = st.checkbox("Use GPU if available", value=False)
device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
st.write("Device:", device)

# Fixed architecture hyperparameters from sampling_and_LSTM.py
EMB_DIM = 128
ENC_HIDDEN = 256
DEC_HIDDEN = 256
ENC_N_LAYERS = 2
DEC_N_LAYERS = 2
DROPOUT = 0.2
BIDIRECTIONAL = True
MAX_DECODE_LEN = 64 # Fixed max length for greedy decoding

# Placeholder for PAD_IDX until char2idx is loaded
PAD_IDX = 0 # Will be overwritten on load

# --- Model Definitions (Copied from sampling_and_LSTM.py) ---
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_hidden, n_layers=1, dropout=0.1, bidirectional=True, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=self.pad_idx)
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
    def __init__(self, vocab_size, emb_dim, enc_hidden, dec_hidden, n_layers=1, dropout=0.1, bidirectional_encoder=True, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=self.pad_idx)
        enc_dim = enc_hidden * (2 if bidirectional_encoder else 1)
        self.rnn = nn.LSTM(emb_dim + enc_dim, dec_hidden, num_layers=n_layers, batch_first=True, dropout=dropout if n_layers>1 else 0.0)
        self.attention = LuongAttention(enc_dim, dec_hidden)
        self.out = nn.Linear(dec_hidden + enc_dim + emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.dec_hidden_dim = dec_hidden
        self.bidirectional_encoder = bidirectional_encoder
        self.n_layers = n_layers

    def forward_step(self, input_token, last_hidden, enc_outputs, enc_mask):
        # input_token: (batch,)
        embedded = self.embedding(input_token).unsqueeze(1) 
        h_last = last_hidden[0][-1]
        context, attn_weights = self.attention(h_last, enc_outputs, mask=enc_mask) 
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2) 
        output, (h, c) = self.rnn(rnn_input, last_hidden) 
        output = output.squeeze(1) 
        output = self.dropout(output)
        concat = torch.cat([output, context, embedded.squeeze(1)], dim=1) 
        logits = self.out(concat)
        return logits, (h, c), attn_weights

    # Placeholder for the main forward method (not used in inference)
    def forward(self, tgt, enc_outputs, enc_lens, teacher_forcing_ratio=0.0):
        raise NotImplementedError("Use forward_step for token-by-token inference.")

# --- Model Loading and Caching ---
@st.cache_resource(show_spinner=True)
def load_model_checkpoint(path, device):
    global PAD_IDX 
    ckpt = torch.load(path, map_location=device)
    char2idx = ckpt.get('char2idx')
    idx2char = ckpt.get('idx2char')
    if char2idx is None or idx2char is None:
        raise RuntimeError("Checkpoint missing char2idx/idx2char entries.")
        
    PAD_IDX = char2idx["<pad>"]
    SOS_IDX = char2idx["<sos>"]
    EOS_IDX = char2idx["<eos>"]
    UNK_IDX = char2idx.get("<unk>", PAD_IDX)
    VOCAB_SIZE = len(char2idx)

    encoder = Encoder(VOCAB_SIZE, EMB_DIM, ENC_HIDDEN, n_layers=ENC_N_LAYERS, dropout=DROPOUT, bidirectional=BIDIRECTIONAL, pad_idx=PAD_IDX).to(device)
    decoder = Decoder(VOCAB_SIZE, EMB_DIM, ENC_HIDDEN, DEC_HIDDEN, n_layers=DEC_N_LAYERS, dropout=DROPOUT, bidirectional_encoder=BIDIRECTIONAL, pad_idx=PAD_IDX).to(device)
    
    encoder.load_state_dict(ckpt['encoder_state'])
    decoder.load_state_dict(ckpt['decoder_state'])
    encoder.eval()
    decoder.eval()
    
    return {"encoder": encoder, "decoder": decoder, "char2idx": char2idx, "idx2char": idx2char, 
            "pad_idx": PAD_IDX, "sos": SOS_IDX, "eos": EOS_IDX, "unk": UNK_IDX}


# --- UI: Load Model State ---
data = None
if st.button("Load model"):
    try:
        with st.spinner('Loading model and vocabulary...'):
            data = load_model_checkpoint(MODEL_PATH, device)
        st.session_state['model_data'] = data
        st.success("Model loaded. Enter text below and press Transliterate.")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

if 'model_data' in st.session_state:
    data = st.session_state['model_data']

if data is None:
    st.info("Enter a valid checkpoint path and press Load model.")
    st.stop()
    
# --- Decoding Helpers ---

def encode_text(text, char2idx, sos_idx, eos_idx, unk_idx):
    ids = [char2idx.get(c, unk_idx) for c in text.strip()]
    ids = [sos_idx] + ids + [eos_idx]
    return ids

@torch.no_grad()
def greedy_decode_app(encoder, decoder, src_tensor, src_len, sos_idx, eos_idx, max_len):
    batch_size = src_tensor.size(0)
    src = src_tensor.to(encoder.rnn.weight_ih_l0.device)
    src_lens = torch.tensor([src_len]*batch_size, dtype=torch.long, device=src.device)
    
    enc_outputs, (h_n, c_n) = encoder(src, src_lens)
    
    # Initialize decoder hidden state to zeros (matching training script)
    dec_n_layers = decoder.rnn.num_layers
    dec_hidden_size = decoder.rnn.hidden_size
    h = torch.zeros(dec_n_layers, batch_size, dec_hidden_size, device=src.device)
    c = torch.zeros_like(h)
    hidden = (h, c)
    
    input_token = torch.tensor([sos_idx]*batch_size, device=src.device)
    outputs_tokens = torch.full((batch_size, max_len), PAD_IDX, dtype=torch.long, device=src.device)
    outputs_tokens[:,0] = sos_idx
    
    enc_mask = torch.arange(enc_outputs.size(1), device=src.device).unsqueeze(0) < src_lens.unsqueeze(1)
    
    for t in range(1, max_len):
        logits, hidden, _ = decoder.forward_step(input_token, hidden, enc_outputs, enc_mask)
        top1 = logits.argmax(1)
        outputs_tokens[:, t] = top1
        input_token = top1
        if (input_token == eos_idx).all():
            break
            
    return outputs_tokens.cpu().tolist()

def ids_to_seq(ids, idx2char):
    chars = [idx2char.get(i, "") for i in ids]
    chars = [c for c in chars if c not in ("<sos>", "<eos>", "<pad>","<unk>")]
    return ''.join(chars)

# --- Interactive Inputs and Translation (Simplified UI) ---
# FIX: Added a unique key='input_text_field' to resolve the DuplicateElementId error
input_text = st.text_input("English (roman) input to transliterate:", value="namaste", key='input_text_field')

if st.button("Transliterate") and data is not None:
    encoder = data["encoder"]; decoder = data["decoder"]
    char2idx = data["char2idx"]; idx2char = data["idx2char"]
    sos = data["sos"]; eos = data["eos"]; unk = data["unk"]
    
    # 1. Encode input text
    ids = encode_text(input_text, char2idx, sos, eos, unk)
    src = torch.tensor([ids], dtype=torch.long)
    
    # 2. Decode using fixed max length
    ys = greedy_decode_app(encoder, decoder, src, len(ids), sos, eos, max_len=MAX_DECODE_LEN)[0]
    
    # 3. Post-process output IDs
    if eos in ys:
        cut = ys.index(eos)
        out_ids = ys[1:cut]
    else:
        out_ids = ys[1:]
    
    pred = ids_to_seq(out_ids, idx2char)
    
    # 4. Display results
    st.success(pred)
    st.markdown(f"**Max decode length used:** {MAX_DECODE_LEN}")
    st.write("Tokens (ids):", out_ids)
    st.write("Input length (incl. SOS/EOS):", len(ids))