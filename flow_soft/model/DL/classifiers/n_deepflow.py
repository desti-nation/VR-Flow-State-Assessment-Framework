import torch
from torch import nn 
import numpy as np
class N_DeepFlow(nn.Module):
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
        def blocks():
            return nn.Sequential( conv_block(1,32),
            conv_block(32,32),
            conv_block(32,32),
            conv_block(32,32,relu=False)
        )
        self.module_list = nn.ModuleList([blocks() for _ in range(4)])
        
        self.flat = nn.Flatten()
        def calc_flat_shape():
            x_list = [torch.tensor(x, dtype=torch.float32) for x in setup.example_x]
            output = []
            for i,module in enumerate(self.module_list):
                output.append(module(x_list[i]))
            y = torch.cat(output,dim=2)
            
            y = self.flat(y)
            return y.shape[-1]
        
        self.fc = nn.Linear(calc_flat_shape(),32)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32,setup.label_n)

    def forward(self,x):
    
        x_list = []
        for i,module in enumerate(self.module_list):
            x_list.append(module(x[i]))
        x = torch.concat(x_list,dim=2)
        y = self.flat(x)
        y = self.fc(y)
        y = self.dropout(y)
        y = self.fc2(y)
        return y
    
