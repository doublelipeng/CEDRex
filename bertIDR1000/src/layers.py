import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear_gate = nn.Linear(dim, dim)
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return torch.sigmoid(self.linear_gate(x)) * self.linear(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.swiglu = SwiGLU(d_ff)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.dropout(self.swiglu(self.linear1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a matrix of [max_len, embedding_dim] with the positional encodings
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]#.transpose(0, 1)
        return self.dropout(x)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2) / dim))
        t = torch.arange(max_len)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer("sin", freqs.sin())
        self.register_buffer("cos", freqs.cos())

    def forward(self, x):#, device
        seq_len = x.size(1)
        return self.sin[:seq_len].to(x.device), self.cos[:seq_len].to(x.device)



def apply_rotary_pos_emb(pos, t):
    # 将 t 分成两半
    t_half = t.reshape(*t.shape[:-1], -1, 2)
    t1, t2 = t_half.unbind(dim=-1)
    # 应用旋转
    t1 = t1 * pos[1]
    t2 = t2 * pos[0]
    # 拼接结果
    t = torch.cat((t1, t2), dim=-1)
    return t


class PreNormTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.feed_forward = PositionwiseFeedForward(d_model, dim_feedforward)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.rope = RotaryEmbedding(d_model)
        
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        # Apply RoPE to Q and K
        pos = self.rope(src)#.to(src.device)  # 获取旋转矩阵
        q = apply_rotary_pos_emb(pos, src).to(src.device)  # 应用 RoPE to Q
        k = apply_rotary_pos_emb(pos, src).to(src.device)  # 应用 RoPE to K
        v = src  # Value 不加 RoPE
        
        src2, attn_weights = self.self_attn(self.norm1(q), self.norm1(k), self.norm1(v), attn_mask=mask)#[0]
        src = src + self.dropout1(src2)
        #src2 = self.linear2(self.dropout(F.gelu(self.linear1(self.norm2(src)))))
        src2 = self.feed_forward(self.norm2(src))
        src = src + self.dropout2(src2)
        return src
        
class ResidualAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Parameter(torch.ones(1))

    def forward(self, attention_output, residual):
        return attention_output + self.gate * residual