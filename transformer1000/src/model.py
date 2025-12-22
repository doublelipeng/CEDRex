import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import length_to_mask, PositionalEncoding

class Transformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_class, num_head=4, num_layers=3, max_len=1000, dropout=0.1):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.tgt_embeddings = nn.Embedding(num_class, hidden_dim)
        self.position_embedding = PositionalEncoding(hidden_dim, dropout, max_len)

        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_head, hidden_dim, dropout, norm_first=True)
        decoder_layer = nn.TransformerDecoderLayer(hidden_dim, num_head, hidden_dim, dropout, norm_first=True)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, norm=nn.LayerNorm(hidden_dim))
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers, norm=nn.LayerNorm(hidden_dim))
        self.output = nn.Linear(hidden_dim, num_class)

    def evaluate(self, src, src_lengths, tag_vocab, min_length=90):
        src = src.transpose(0, 1)
        src_hidden_states = self.position_embedding(self.embeddings(src))
        src_mask = ~length_to_mask(src_lengths, device=src.device)
        memory = self.encoder(src_hidden_states, src_key_padding_mask=src_mask)

        ys = torch.ones(1, 1).fill_(tag_vocab["<bos>"]).type_as(src)
        generated_length = 0

        for _ in range(min_length - 1):
            tgt_hidden_states = self.position_embedding(self.tgt_embeddings(ys))
            tgt_len = ys.size(0)
            tgt_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=ys.device), diagonal=1).bool()

            out = self.decoder(tgt_hidden_states, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_mask)
            out = self.output(out.transpose(0, 1))
            prob = F.log_softmax(out[:, -1], dim=-1)
            top2 = torch.topk(prob, k=2, dim=-1).indices.squeeze(0)
            next_word = top2[0].item()
            generated_length += 1
            if next_word == tag_vocab["<eos>"] and generated_length < min_length:
                next_word = top2[1].item()
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src).fill_(next_word)], dim=0)
            if next_word == tag_vocab["<eos>"] and generated_length >= min_length:
                break
        return ys

    def evaluate_stop_on_eos(self, src, src_lengths, tag_vocab, min_length=1):
        src = src.transpose(0, 1)
        src_hidden_states = self.position_embedding(self.embeddings(src))
        src_mask = ~length_to_mask(src_lengths, device=src.device)
        memory = self.encoder(src_hidden_states, src_key_padding_mask=src_mask)
    
        ys = torch.ones(1, 1).fill_(tag_vocab["<bos>"]).type_as(src)
    
        eos_id = tag_vocab["<eos>"]
    
        for _ in range(1000):  # 安全上限
            tgt_hidden_states = self.position_embedding(self.tgt_embeddings(ys))
            tgt_len = ys.size(0)
            tgt_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=ys.device), diagonal=1).bool()
    
            out = self.decoder(tgt_hidden_states, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_mask)
            out = self.output(out.transpose(0, 1))
    
            prob = F.log_softmax(out[:, -1], dim=-1)
            next_word = torch.argmax(prob, dim=-1).item()
    
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src).fill_(next_word)], dim=0)
    
            # 🚨 eos 出现 → 立即停止
            if next_word == eos_id:
                break
    
        return ys

