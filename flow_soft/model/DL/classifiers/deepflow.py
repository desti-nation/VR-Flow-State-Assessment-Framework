import torch
from torch import nn 
from classifiers.base import Classifier
import numpy as np
class DeepFlow(nn.Module):
    def __init__(self,setup):
        super().__init__()
        self.setup = setup 
    
        def conv_block(in_channel,out_channel=32,dropout=0.1,relu=True):
            if relu:
                return nn.Sequential(
                    nn.Conv1d(in_channel,out_channel,kernel_size=3,stride=1,padding=1),
                    nn.BatchNorm1d(out_channel),
                    nn.ReLU(inplace=True),  
                    nn.MaxPool1d(3),
                    nn.Dropout(dropout)
                )
            else:
                return nn.Sequential(
                    nn.Conv1d(in_channel,out_channel,kernel_size=3,stride=1,padding=1),
                    nn.BatchNorm1d(out_channel),
                    nn.MaxPool1d(3),
                    nn.Dropout(dropout)
                )
        conv_blocks = nn.Sequential( conv_block(4,32),
            conv_block(32,32),
            conv_block(32,32),
            conv_block(32,32,relu=False)
        )
        self.conv_blocks = conv_blocks
        
        self.flat = nn.Flatten()
        def calc_flat_shape():
            x = np.concatenate(setup.example_x, axis=1)
            x = torch.tensor(x, dtype=torch.float32) 
            y = self.conv_blocks(x)
            y = self.flat(y)
            return y.shape[-1]
        
        self.fc = nn.Linear(calc_flat_shape(),32)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32,setup.label_n)# Dense(2/3)

    def forward(self,x):
        # 对blk进行插值 
        # x的shape为(batch_size,4,signal_length)
        x = torch.concat(x,dim=1)
        
        y = self.conv_blocks(x)
        y = self.flat(y)
        y = self.fc(y)
        y = self.dropout(y)
        y = self.fc2(y)
        return y
    
