import torch
import torch.nn as nn
from SSL.base import * 

class Rep_Att(Representation):
    def __init__(self,setup,channel_i,input_dim=1, num_heads=1, num_encoder_layers=3, ff_dim=2048, dropout_rate=0.1):
        super().__init__(setup,channel_i)
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads, 
                dim_feedforward=ff_dim, # 前馈网络的维度 ff_dim
                dropout=dropout_rate
            ) for _ in range(num_encoder_layers)
        ])
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers,num_layers=num_encoder_layers)
        
        '''
        生成位置编码
        '''
        self.positional_encoding = self._generate_positional_encoding(input_dim)
        
        self.fc = nn.Linear(input_dim, input_dim)  # 用于预测被掩盖的标记

    def forward(self, x, mask):
        
        # x [batch_size,input_dim,seq_length]
        # x: [seq_length, batch_size, input_dim]
        x = x.permute(2, 0, 1)
        mask = mask.permute(2,0,1)
        
        
        x = x + self.positional_encoding[:x.size(0), :,:].to(device=x.device)
        x = self.transformer_encoder(x, mask=mask)
        
        if self.froze:
            return x # 如果冻结，则直接返回编码器输出

        # 预测被掩盖的标记
        # 我们只对被掩盖的位置进行预测，所以使用 mask 来选择这些位置
        # masked_positions = mask.unsqueeze(-1).expand(-1, -1, self.input_dim)  # 扩展维度以应用在 x 上
        masked_x = x * mask # 选择被掩盖的位置
        predictions = self.fc(masked_x)  # 使用预测器预测这些位置
        
        # 我们需要预测原始的标记，所以预测的形状应该是 [batch_size, seq_length, input_dim]
        # 这里我们简单地使用 masked_positions 来恢复预测的原始位置
        # predictions = predictions * mask # 恢复预测的原始位置
        return predictions.permute(1, 2, 0)


    def _generate_positional_encoding(self, dim, max_len=6000):
        """
        生成位置编码
        """
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 用于计算正弦和余弦函数频率衰减的除数向量
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # (max_len, 1, dim)
        return pe
