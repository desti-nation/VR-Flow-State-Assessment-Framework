
from SSL.base import *
from classifiers.base import *
from torch import nn
import torch
class AttentionLayer(nn.Module):
    def __init__(self) :
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
    def forward(self,x):
        channel_size = x.shape[1]
        att_data = x[:,:channel_size//2,:]
        att_softmax = x[:,channel_size//2:,:]
        att_softmax = self.softmax(att_softmax)

        return att_data * att_softmax

class Rep_Encoder_predict(Representation):
    def __init__(self,setup,channel_i):
        super().__init__(setup,channel_i)

        # conv block 
        def conv_block(in_channel, out_channel,kernel_size,pool_size,dropout_rate, stride=1, padding=-1,dim=1):
            conv,norm,pool = None,None,None

            if padding == -1:
                padding = calc_padding(kernel_size)
            if dim == 1:
                conv = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride, padding=padding)
                norm = nn.InstanceNorm1d(out_channel)
                pool = nn.MaxPool1d(pool_size)
            elif dim == 2:
                conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding)
                norm = nn.InstanceNorm1d(out_channel)
                pool = nn.MaxPool2d(pool_size)
            elif dim == 3:
                conv = nn.Conv3d(in_channel, out_channel, kernel_size, stride=stride, padding=padding)
                norm = nn.InstanceNorm1d(out_channel)
                pool = nn.MaxPool3d(pool_size)
            return [
                conv,
                norm,
                nn.PReLU(),
                nn.Dropout(dropout_rate),
                pool
            ]
        def conv_block_list(channels,kernel_sizes,pool_sizes,dropout_rates,dim=1):
            block_list = []
            for i in range(len(kernel_sizes)):
                block_list += conv_block(channels[i],channels[i+1],kernel_sizes[i],pool_sizes[i],dropout_rates[i],dim)
            
            return block_list
        
        channels =  [1,4,8,16]
        kernel_sizes = [5,11,21]
        pool_sizes = [2,2,2]
        dropout_rates = [0.1,0.2,0.3]
        self.conv_module =  nn.Sequential(* conv_block_list(channels,kernel_sizes,pool_sizes,dropout_rates)) 
        self.att_module = AttentionLayer() 
        
        def att_output_shape():
            x = self.setup.example_x[self.channel_i][:,:,:-10] 

            output = self.conv_module(torch.tensor(x,dtype=torch.float32))
            output = self.att_module(output) 
            # output = output.permute(0,2,1)
            return output.shape
        input_shape = self.setup.example_x[self.channel_i][:,:,:-10].shape
        self.fc = nn.Linear( att_output_shape()[-1],input_shape[-1])
        self.fc2 = nn.Linear( att_output_shape()[-2],input_shape[-2])

    
    def forward(self, x):
    
        y = self.conv_module(x)
        y = self.att_module(y)

        if self.froze:
            return y  
       
        y = self.fc(y)
        y = self.fc2(y.permute(0,2,1))
        return y.permute(0,2,1)


