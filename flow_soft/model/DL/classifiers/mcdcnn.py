from torch import nn
from classifiers.base import *
import torch

class ClassifierMcdCNN(Classifier):
    def __init__(self,setup):
        super().__init__(setup)

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
        
        # channels = [
        #     [1,4,8],
        #     [1,4,8],
        #     [1,4,8],
        #     [1,4,8],
        # ]
        # kernel_sizes = [
        #     [16,32],
        #     [16,32],
        #     [16,32],
        #     [16,32],
        # ]
        # pool_sizes = [
        #     [4,4],
        #     [4,4],
        #     [4,4],
        #     [4,4], 
        # ]

        channels = [
            [1,4,8],
            [1,4,8],
            [1,4,8],
            [1,4,8],
        ]
        kernel_sizes = [
            [32,8],
            [32,8],
            [32,8],
            [32,8]
        ]
        pool_sizes = [
            [16,16],
            [16,16],
            [16,16],
            [16,16],
        ]

        self.module_list = [ nn.Sequential(* (
            conv_block_list(channels[i],kernel_sizes[i],pool_sizes[i])+ [nn.Flatten() ]
            )) for i in range(Dataset.channel_n)
        ]
        [self.module1,self.module2,self.module3,self.module4] = self.module_list
        
        def calc_flat_shape():
            x = self.setup.example_x 
            x_list = [self.module_list[i](torch.tensor(x[i],dtype=torch.float32)) for i in range(Dataset.channel_n)]
            x = torch.cat(x_list,dim=1)
            print(x.shape)
            return x.shape[-1]

        concat_length = calc_flat_shape()
        self.fc1 = nn.Linear(concat_length,concat_length//64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(concat_length//64, setup.label_n)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        output = []
        for i in range(Dataset.channel_n):
            output.append(self.module_list[i](x[i]))
        
        x = torch.cat(output, dim=1)
        y = self.fc1(x)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.softmax(y)
        return y


        