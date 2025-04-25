import torch
import torch.nn as nn
import torch.nn.functional as F
from SSL.base import * 
channels = [1,4,8,16]
kernel_sizes = [5,11,21]
class Rep_CNN(Representation):
    def __init__(self,setup,channel_i, channels=channels,kernel_sizes=kernel_sizes):
        super().__init__(setup,channel_i)
        # 1D卷积层用于特征提取
        encoder = []
        for i in range(len(kernel_sizes)):
            encoder += [
                nn.Conv1d(channels[i], channels[i+1], kernel_size=kernel_sizes[i], stride=1, padding=kernel_sizes[i]//2),
                nn.InstanceNorm1d(channels[i+1]),
                nn.ReLU()
            ]
        self.encoder = nn.Sequential(*encoder)
        # 1D转置卷积层用于重建缺失部分
        self.fc1 = nn.Linear(channels[-1], channels[0])

    def forward(self, x, mask=None):
        # x: 输入时序数据
        # mask: 掩码，指示哪些部分需要被预测

        # 特征提取
        features = self.encoder(x) 
        if self.froze:
            return features
        # 重建缺失部分
        assert mask is not None
        reconstructed = self.fc1(features.permute(0,2,1))   # 使用掩码来只重建缺失的部分 # todo 
        return reconstructed.permute(0,2,1) * mask
