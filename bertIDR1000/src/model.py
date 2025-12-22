import torch
import torch.nn as nn
from .layers import PositionalEncoding, PreNormTransformerEncoderLayer,ResidualAttention



class BERTForTokenClassification(nn.Module):
    def __init__(self, num_embeddings=25, d_model=320, nhead=8, dim_feedforward=320, num_layers=6, num_classes=25, dropout=0.1,max_len=1000):
        super().__init__()
        # self.embeddings = nn.Embedding(num_embeddings, d_model)
        self.embedding_frozen = nn.Embedding(21, d_model)
        self.embedding_trainable = nn.Embedding(4, d_model)
        
        # # 冻结前21个
        # self.embedding_frozen.weight.requires_grad = False

        self.position_embedding = PositionalEncoding(d_model, dropout, max_len)#nn.Sequential(nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)

        self.encoder_layers = nn.ModuleList([
            PreNormTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.residual_attention = ResidualAttention(d_model)
        self.output = nn.Linear(d_model, num_classes)

    def forward(self, x):# x: [batch, seq_len] in 0~24
        #一部分embedding grad，一部分不grad
        frozen_mask = (x < 21)
        trainable_mask = ~frozen_mask
    
        # 创建两个掩码的输入
        x_frozen = x.clone()
        x_frozen[~frozen_mask] = 0  # 无效位置设为0
        x_trainable = x - 21
        x_trainable[frozen_mask] = 0
    
        embed = torch.zeros(x.shape[0], x.shape[1], 320).to(x.device)
        embed += self.embedding_frozen(x_frozen) * frozen_mask.unsqueeze(-1)
        embed += self.embedding_trainable(x_trainable) * trainable_mask.unsqueeze(-1)
    
        x = self.position_embedding(embed)#
        # x=self.position_embedding(self.embeddings(x))
        x = self.norm(x)
        # 生成 look-ahead mask
        # seq_len = x.size(1)  # 获取序列长度
        # src_attention_mask = generate_square_subsequent_mask(seq_len).to(x.device)
        for layer in self.encoder_layers:
            x_residual = x
            if self.training and x.requires_grad:
                x = checkpoint(layer, x)#, src_attention_mask)
            else:
                x = layer(x)#, src_attention_mask)
            x = self.residual_attention(x, x_residual)
            
        logits = self.output(x)
        return logits