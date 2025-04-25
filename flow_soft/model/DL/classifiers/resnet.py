import torch.nn as nn
import torch 
from classifiers.base import *
from dataset import Dataset
class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_sizes,dim=1):
        super().__init__()
        def gen_conv_bn(in_channel,out_channel,kernel_size):
            conv,bn = None,None
            padding = calc_padding(kernel_size)
            if dim == 1:
                conv = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, padding=padding)
                bn = nn.BatchNorm1d(out_channel)
            elif dim == 2:
                conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding)
                bn = nn.BatchNorm2d(out_channel)
            elif dim == 3:
                conv = nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, padding=padding)
                bn = nn.BatchNorm3d(out_channel)
            return [conv,bn]
      
        self.conv1 = nn.Sequential(*gen_conv_bn(in_channel, out_channel, kernel_sizes[0]), nn.ReLU())
        self.conv2 = nn.Sequential(*gen_conv_bn(out_channel, out_channel, kernel_sizes[1]), nn.ReLU())
        self.conv3 = nn.Sequential(*gen_conv_bn(out_channel, out_channel, kernel_sizes[2]))

        self.shortcut = nn.Sequential(*gen_conv_bn(in_channel, out_channel, 1))

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        shortcut = self.shortcut(x)
        
        out += shortcut
        out = self.relu(out)
        
        return out

class ClassifierResnet(Classifier):
    def __init__(self,setup):
        super().__init__(setup)

        # 2 residual blocks + last block 
        channels = [
            [1,2,4,8],
            [1,2,4,8],
            [1,2,4,8],
            [1,2,4,8],
        ]
        kernel_sizes = [
            [3,3,3],
            [5,5,5],
            [7,7,7]
        ]
        
        

        def residual_blocks(channels, kernel_sizes):
            blocks = []
            for i in range(len(kernel_sizes)):
                blocks.append(ResidualBlock(channels[i], channels[i+1], kernel_sizes[i]))
            return blocks

        self.module_list = [
            nn.Sequential( 
                *residual_blocks(channels[i],kernel_sizes) , 
                nn.AdaptiveAvgPool2d(output_size=(1,1))
            ) for i in range(Dataset.channel_n) 
        ]
        [self.module1,self.module2,self.module3,self.module4] = self.module_list

        self.flat = nn.Flatten()
        
        self.fc = nn.Linear(self.calc_flat_shape(), setup.label_n)
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x):
        output = []
        for i in range(Dataset.channel_n):
            output.append( self.module_list[i](x[i]) )
        
        x = torch.cat(output, dim=2)
        y = self.flat(x)
        y = self.fc(y)
        y = self.Softmax(y)
        return y 
        

        