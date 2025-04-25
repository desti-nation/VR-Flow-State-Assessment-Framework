from torch import nn
from SSL.base import *
from classifiers.base import *
import torch

class Rep_McdCNN(Representation):
    def __init__(self,setup,channel_i):
        super().__init__(setup,channel_i)

        def conv_block(in_channels, out_channels,kernel_size,pool_size,stride=1,padding=-1,dim=1):
            conv,pool = None,None
            if padding == -1: 
                padding = calc_padding(kernel_size)
            if dim == 1:
                conv = nn.Conv1d(in_channels,out_channels,kernel_size,stride,padding)
                pool = nn.MaxPool1d(pool_size)
            elif dim == 2:
                conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding)
                pool = nn.MaxPool2d(pool_size) # pool 的kernel_size要和conv 保持一致吗
            elif dim == 3:
                conv = nn.Conv3d(in_channels,out_channels,kernel_size,stride,padding)
                pool = nn.MaxPool3d(pool_size)
            return [conv,pool]

        def conv_block_list(channels,kernel_sizes,pool_sizes,dim=1):

            block_list = []
            for i in range(len(kernel_sizes)):
                block_list += conv_block(channels[i],channels[i+1],kernel_sizes[i],pool_sizes[i],dim=dim)
            return block_list

        channels = [1,4,8]
        kernel_sizes = [32,8]
        pool_sizes = [16,16]

        self.conv_module = nn.Sequential(
            *conv_block_list(channels,kernel_sizes,pool_sizes)
        )
        def output_shape():
            x = self.setup.example_x[self.channel_i] 
            output = self.conv_module(torch.tensor(x,dtype=torch.float32))
            return output.shape
        print(output_shape())
        input_shape = self.setup.example_x[self.channel_i].shape
        self.fc = nn.Linear( output_shape()[-1],input_shape[-1])
        self.fc2 = nn.Linear( output_shape()[-2],input_shape[-2])
        

    def forward(self, x,mask=None):
        y = self.conv_module(x)

        if self.froze:
            return y 
        
        y = self.fc(y)
        y = self.fc2(y.permute(0,2,1))
        return y.permute(0,2,1)*mask

        