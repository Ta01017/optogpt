# ===========================
# Transformer (SDPA version)
# - Uses torch.nn.functional.scaled_dot_product_attention (Flash/MemEff when available)
# - Uses nn.LayerNorm (faster than custom LN)
# - Mask handling: bool masks, broadcast-friendly
# Requires: PyTorch >= 2.0
# ===========================

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def clones(module, N):
    """Produce N identical layers (deepcopy so weights are independent)."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-(math.log(10000.0) / d_model))
        )
        pe_pos = position * div_term
        pe[:, 0::2] = torch.sin(pe_pos)
        pe[:, 1::2] = torch.cos(pe_pos)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        # x: (B, T, D)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


def _ensure_bool_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
    if mask is None:
        return None
    # allow uint8/int/float 0/1 masks
    return mask.to(dtype=torch.bool)


def _to_sdpa_attn_mask(mask: torch.Tensor | None, B: int, H: int, q_len: int, k_len: int) -> torch.Tensor | None:
    """
    Convert common transformer masks to a bool mask broadcastable to SDPA.
    SDPA accepts:
      - (B, H, q_len, k_len) bool
      - (B, 1, 1, k_len) bool
      - (B, 1, q_len, k_len) bool
    """
    if mask is None:
        return None
    mask = _ensure_bool_mask(mask)

    # Common cases:
    # 1) padding mask: (B, k_len) -> (B, 1, 1, k_len)
    if mask.dim() == 2 and mask.size(0) == B and mask.size(1) == k_len:
        return mask[:, None, None, :]

    # 2) attention mask: (B, q_len, k_len) -> (B, 1, q_len, k_len)
    if mask.dim() == 3 and mask.size(0) == B and mask.size(1) == q_len and mask.size(2) == k_len:
        return mask[:, None, :, :]

    # 3) already (B, 1, 1, k_len) / (B, 1, q_len, k_len) / (B, H, q_len, k_len)
    if mask.dim() == 4 and mask.size(0) == B and mask.size(-2) == q_len and mask.size(-1) == k_len:
        # If head dim is 1 it will broadcast; if H it matches; otherwise broadcast rules apply.
        return mask

    # 4) legacy: you had mask.unsqueeze(1) inside MHA; sometimes becomes (B,1,q_len,k_len) already.
    # If it doesn't match above, try a conservative reshape when possible:
    raise ValueError(f"Unsupported mask shape for SDPA: {tuple(mask.shape)}")


class MultiHeadedAttentionSDPA(nn.Module):
    """Multi-head attention using PyTorch SDPA (FlashAttention/MemEff when available)."""

    def __init__(self, h, d_model, dropout=0.1, is_causal=False):
        super().__init__()
        assert d_model % h == 0
        self.h = h
        self.d_k = d_model // h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout_p = float(dropout)
        self.is_causal = bool(is_causal)

    def forward(self, query, key, value, mask=None):
        # query/key/value: (B, T, D)
        B, q_len, _ = query.shape
        _, k_len, _ = key.shape

        # 1) project
        q, k, v = [
            l(x).view(B, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]  # (B, H, T, d_k)

        attn_mask = _to_sdpa_attn_mask(mask, B=B, H=self.h, q_len=q_len, k_len=k_len)

        # 2) sdpa
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,                          # bool mask where True means "keep"
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=self.is_causal
        )  # (B, H, q_len, d_k)

        # 3) concat + output proj
        out = out.transpose(1, 2).contiguous().view(B, q_len, self.h * self.d_k)
        return self.linears[-1](out)


class SublayerConnection(nn.Module):
    """Pre-norm residual: x + dropout(sublayer(LN(x)))"""

    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    """Standard Transformer FFN: Linear -> GELU -> Dropout -> Linear"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.gelu(self.w1(x))))


class FullyConnectedLayers(nn.Module):
    """Simple MLP head. Note: using nn.LayerNorm for speed/stability."""

    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, out_dim)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        return self.fc2(self.norm(self.fc1(x)))


# -------- Encoder stack (classification-style pooling uses token 0) --------

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn: nn.Module, feed_forward: nn.Module, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask=None):
        x = self.sublayer[0](x, lambda t: self.self_attn(t, t, t, mask))
        x = self.sublayer[1](x, self.feed_forward)
        return x


class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N: int):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class TransformerEncoderClassifier(nn.Module):
    """
    Your original 'Transformer' encoder-only model:
    - src -> embed+pos -> encoder -> take token 0 -> FC head
    """

    def __init__(self, encoder, fc, src_embed):
        super().__init__()
        self.encoder = encoder
        self.fc = fc
        self.src_embed = src_embed

    def encode(self, src, src_mask=None):
        return self.encoder(self.src_embed(src), src_mask)

    def forward(self, src, src_mask=None):
        enc = self.encode(src, src_mask)      # (B, T, D)
        pooled = enc[:, 0, :]                 # take first token
        return self.fc(pooled)


def make_model_encoder_classifier(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy

    attn = MultiHeadedAttentionSDPA(h, d_model, dropout=dropout, is_causal=False)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    fc = FullyConnectedLayers(d_model, tgt_vocab)

    model = TransformerEncoderClassifier(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        fc,
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# -------- Decoder stack (seq2seq) --------

def subsequent_mask(size: int, device=None):
    """
    Causal mask: True means allowed, False means blocked.
    Returns shape (1, size, size) for easy broadcast to (B,1,size,size).
    """
    # upper-triangular (excluding diagonal) is blocked
    m = torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)
    return (~m).unsqueeze(0)  # (1, size, size)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn: nn.Module, src_attn: nn.Module, feed_forward: nn.Module, dropout):
        super().__init__()
        self.size = size
        self.self_attn = self_attn      # should be causal
        self.src_attn = src_attn        # cross-attn (not causal)
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        # 1) causal self-attn
        x = self.sublayer[0](x, lambda t: self.self_attn(t, t, t, tgt_mask))
        # 2) cross-attn
        x = self.sublayer[1](x, lambda t: self.src_attn(t, memory, memory, src_mask))
        # 3) FFN
        x = self.sublayer[2](x, self.feed_forward)
        return x


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N: int):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class Transformer_I(nn.Module):
    """
    Your original decoder-only model with an external 'fc' that maps src features to d_model,
    then uses a decoder to generate tgt tokens.
    """

    def __init__(self, fc, decoder, tgt_embed, generator):
        super().__init__()
        self.fc = fc
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.generator = generator

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory = self.fc(src)  # make src -> (B, S, d_model) or (B, d_model) depending on your src layout
        return self.decode(memory, src_mask, tgt, tgt_mask)


def make_model_I(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy

    # Decoder self-attn should be causal. We can set is_causal=True and pass no mask,
    # BUT you often still need padding mask; so we keep is_causal=False and rely on tgt_mask.
    # If you have no padding and only need causal, you can set is_causal=True and omit tgt_mask.
    self_attn = MultiHeadedAttentionSDPA(h, d_model, dropout=dropout, is_causal=False)
    src_attn  = MultiHeadedAttentionSDPA(h, d_model, dropout=dropout, is_causal=False)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    # map src features -> d_model (you had FullyConnectedLayers(src_vocab, d_model))
    fc = FullyConnectedLayers(src_vocab, d_model)

    model = Transformer_I(
        fc=fc,
        decoder=Decoder(DecoderLayer(d_model, c(self_attn), c(src_attn), c(ff), dropout), N),
        tgt_embed=nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        generator=Generator(d_model, tgt_vocab),
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
