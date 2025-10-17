
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from pathlib import Path
import json

# ---------- User-changeable settings ----------
# Path to the checkpoint (replace with your .pth file path)
CKPT_PATH = "seq2seq_lstm.pth"

# Model hyperparameters -- set these to match your training script if different
EMB_DIM = 128
ENC_HIDDEN = 256
DEC_HIDDEN = 256
ENC_N_LAYERS = 2
DEC_N_LAYERS = 2
DROPOUT = 0.2
BIDIRECTIONAL = True
MAX_OUTPUT_LEN = 64
# ----------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.title("Seq2Seq (LSTM+LuongAttn) — Inference GUI")

st.markdown(
    """
    Simple Streamlit inference UI for a sequence-to-sequence LSTM model with Luong attention.
    - Put your checkpoint (a `.pth` dict saved with keys like `'encoder_state'`, `'decoder_state'`, `'char2idx'`, `'idx2char'`) next to this app or give its path.
    - If loading fails due to architecture mismatch, adjust the hyperparameters at the top of this file to match the training script.
    """
)

ckpt_path = st.text_input("Checkpoint path", CKPT_PATH)

@st.cache_resource
def load_checkpoint(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(str(path), map_location="cpu")
    return ckpt

# Minimal model classes matching the training script's design (embedding + LSTM encoder, LSTM decoder + Luong attention)
class LuongAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        self.attn = nn.Linear(dec_dim, enc_dim)

    def forward(self, dec_hidden, enc_outputs, mask=None):
        # dec_hidden: (batch, dec_dim)
        # enc_outputs: (batch, seq, enc_dim)
        proj = self.attn(dec_hidden).unsqueeze(2)      # (batch, enc_dim, 1)
        scores = torch.bmm(enc_outputs, proj).squeeze(2)  # (batch, seq)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        attn_weights = torch.softmax(scores, dim=1)   # (batch, seq)
        context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1)  # (batch, enc_dim)
        return context, attn_weights

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_hidden, n_layers=1, dropout=0.1, bidirectional=True, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(emb_dim, enc_hidden, num_layers=n_layers, batch_first=True,
                           bidirectional=bidirectional, dropout=dropout if n_layers>1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        self.enc_hidden = enc_hidden
        self.n_layers = n_layers

    def forward(self, src, src_lens):
        # src: (batch, seq)
        embedded = self.dropout(self.embedding(src))  # (batch, seq, emb)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # (batch, seq, hidden*directions)

        # If bidirectional, combine the forward/back hidden states in a consistent way if needed.
        return outputs, (h_n, c_n)

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_hidden, dec_hidden, n_layers=1, dropout=0.1, bidirectional_encoder=True, pad_idx=0):
        super().__init__()
        enc_dim = enc_hidden * (2 if bidirectional_encoder else 1)
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(emb_dim + enc_dim, dec_hidden, num_layers=n_layers, batch_first=True,
                           dropout=dropout if n_layers>1 else 0.0)
        self.attention = LuongAttention(enc_dim, dec_hidden)
        self.out = nn.Linear(dec_hidden + enc_dim + emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, input_token, last_hidden, enc_outputs, enc_mask):
        # input_token: (batch,) int token
        embedded = self.embedding(input_token).unsqueeze(1)  # (batch,1,emb)
        # last_hidden: tuple (h,c) where h: (num_layers, batch, dec_hidden)
        h_last = last_hidden[0][-1]  # top layer last hidden: (batch, dec_hidden)
        context, attn_weights = self.attention(h_last, enc_outputs, mask=enc_mask)  # context: (batch, enc_dim)
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)  # (batch,1, emb+enc_dim)
        output, (h, c) = self.rnn(rnn_input, last_hidden)
        output = output.squeeze(1)  # (batch, dec_hidden)
        output = self.dropout(output)
        concat = torch.cat([output, context, embedded.squeeze(1)], dim=1)  # (batch, dec+enc+emb)
        logits = self.out(concat)
        return logits, (h, c), attn_weights

# ---------- helper conversions ----------
def seq_to_ids(txt, char2idx, sos_token="<sos>", eos_token="<eos>", unk_token="<unk>"):
    # expects a plain string and returns a list of ints (including <sos> and <eos>)
    ids = [char2idx.get(sos_token)]
    for ch in txt:
        ids.append(char2idx.get(ch, char2idx.get(unk_token)))
    ids.append(char2idx.get(eos_token))
    return ids

def ids_to_seq(id_list, idx2char, stop_at_eos=True):
    pieces = []
    for idx in id_list:
        ch = idx2char.get(idx, "")
        if ch == "<eos>" and stop_at_eos:
            break
        if ch in ("<pad>", "<sos>"):
            continue
        pieces.append(ch)
    return "".join(pieces)

# ---------- load checkpoint ----------
load_btn = st.button("Load checkpoint & build models")
if load_btn:
    try:
        ckpt = load_checkpoint(ckpt_path)
    except Exception as e:
        st.error(f"Failed to load checkpoint: {e}")
        st.stop()

    # expect char2idx & idx2char saved inside checkpoint
    if 'char2idx' not in ckpt or 'idx2char' not in ckpt:
        st.error("Checkpoint does not contain 'char2idx'/'idx2char'. Please save them in your training script.")
        st.stop()

    char2idx = ckpt['char2idx']
    idx2char = {int(k): v for k, v in (ckpt.get('idx2char') or {}).items()} if isinstance(ckpt.get('idx2char'), dict) else ckpt.get('idx2char')

    # indices
    PAD_IDX = char2idx.get("<pad>", 0)
    SOS_IDX = char2idx.get("<sos>", 1)
    EOS_IDX = char2idx.get("<eos>", 2)
    UNK_IDX = char2idx.get("<unk>", 3)
    vocab_size = len(char2idx)

    # Build models using hyperparams (edit at top if you used different values)
    enc = Encoder(vocab_size, EMB_DIM, ENC_HIDDEN, n_layers=ENC_N_LAYERS, dropout=DROPOUT, bidirectional=BIDIRECTIONAL, pad_idx=PAD_IDX)
    dec = Decoder(vocab_size, EMB_DIM, ENC_HIDDEN, DEC_HIDDEN, n_layers=DEC_N_LAYERS, dropout=DROPOUT, bidirectional_encoder=BIDIRECTIONAL, pad_idx=PAD_IDX)

    # Try to load state dicts
    try:
        enc.load_state_dict(ckpt['encoder_state'])
        dec.load_state_dict(ckpt['decoder_state'])
    except Exception as e:
        st.warning("Error loading state_dict — this usually means hyperparameters (hidden sizes, embeddings, bidirection) don't match. See the exception and adjust top-of-file hyperparams if needed.")
        st.exception(e)
        # show keys to help debug
        st.write("Checkpoint keys:", list(ckpt.keys()))
        # show sample shapes
        st.stop()

    enc.to(device).eval()
    dec.to(device).eval()
    st.success("Models loaded and ready on device: " + str(device))
    st.session_state['models_loaded'] = True
    st.session_state['enc'] = enc
    st.session_state['dec'] = dec
    st.session_state['char2idx'] = char2idx
    st.session_state['idx2char'] = idx2char
    st.session_state['PAD_IDX'] = PAD_IDX
    st.session_state['SOS_IDX'] = SOS_IDX
    st.session_state['EOS_IDX'] = EOS_IDX
    st.session_state['UNK_IDX'] = UNK_IDX

if st.session_state.get('models_loaded', False):
    enc = st.session_state['enc']
    dec = st.session_state['dec']
    char2idx = st.session_state['char2idx']
    idx2char = st.session_state['idx2char']
    PAD_IDX = st.session_state['PAD_IDX']
    SOS_IDX = st.session_state['SOS_IDX']
    EOS_IDX = st.session_state['EOS_IDX']
    UNK_IDX = st.session_state['UNK_IDX']

    src_text = st.text_input("Enter source text to transliterate / predict (native roman text)")
    max_len = st.slider("Max output length (tokens)", min_value=10, max_value=256, value=MAX_OUTPUT_LEN)
    do_beam = st.checkbox("Use beam search? (not implemented; greedy used)", value=False)
    run = st.button("Run inference")

    if run:
        if not src_text:
            st.warning("Please enter input text.")
        else:
            # convert to ids
            src_ids = seq_to_ids(src_text, char2idx)
            import torch.nn.utils.rnn as rnn_utils
            src_tensor = torch.LongTensor([src_ids]).to(device)
            src_lens = torch.LongTensor([len(src_ids)]).to(device)

            with torch.no_grad():
                enc_outputs, enc_state = enc(src_tensor, src_lens)
                # create mask
                enc_mask = torch.arange(enc_outputs.size(1), device=device).unsqueeze(0) < src_lens.unsqueeze(1)

                # initialise decoder hidden as zeros (common in seq2seq)
                h0 = torch.zeros(dec.rnn.num_layers, 1, dec.rnn.hidden_size, device=device)
                c0 = torch.zeros(dec.rnn.num_layers, 1, dec.rnn.hidden_size, device=device)
                dec_hidden = (h0, c0)

                # first token = <sos>
                cur = torch.LongTensor([SOS_IDX]).to(device)
                outputs = []
                for step in range(max_len):
                    logits, dec_hidden, attn = dec.forward_step(cur, dec_hidden, enc_outputs, enc_mask)
                    probs = torch.softmax(logits, dim=1)
                    top1 = torch.argmax(probs, dim=1)
                    idx = top1.item()
                    if idx == EOS_IDX:
                        break
                    outputs.append(idx)
                    cur = torch.LongTensor([idx]).to(device)

                pred = ids_to_seq(outputs, idx2char)
                st.write("**Prediction:**")
                st.success(pred)
                st.write("Raw token ids:", outputs)
                st.write("Attention shape:", attn.shape if 'attn' in locals() else None)

else:
    st.info("Click 'Load checkpoint & build models' to load your .pth and build the models. Make sure the hyperparameters at the top match the ones used when training.")
