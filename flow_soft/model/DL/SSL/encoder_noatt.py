
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

class Rep_Encoder_noAtt(Representation):
    def __init__(self,setup,channel_i):
        super().__init__(setup,channel_i)
        print('Rep encoder no Att')
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
        
        # channels =  [1,4,8,16]
        # kernel_sizes = [5,11,21]
        # pool_sizes = [2,2,2]
        # dropout_rates = [0.1,0.2,0.3]
        '''
        eeg: {
            channels =  [1,16,16,32] # 增加卷积层数并不能解决问题
        kernel_sizes = [3,9,21]
        pool_sizes = [3,7,11] # 池化和kernel保持匹配的话会效果更好
        dropout_rates = [0.3,0.2,0.1] 
        }
        gsr: {
           channels =  [1,4,8,16] # 增加卷积层数并不能解决问题
        kernel_sizes = [3,11,11]
        pool_sizes = [3,11,11] # 池化和kernel保持匹配的话会效果更好
        dropout_rates = [0.1,0.2,0.3] # 递减效果更好
        }
        PPG:{
           channels =  [1,16,16,32] # 增加卷积层数并不能解决问题
        kernel_sizes = [3,9,21]
        pool_sizes = [3,7,11] # 池化和kernel保持匹配的话会效果更好
        dropout_rates = [0.3,0.2,0.1] (同eeg)
        }
        BLK:{
        channels =  [1,16,16,32] # 增加卷积层数并不能解决问题
        kernel_sizes = [9,8,21]
        pool_sizes = [3,7,11] # 池化和kernel保持匹配的话会效果更好
        dropout_rates = [0.1,0.2,0.3] 
        }
        '''
        # 对于所有模态
        channels_all = [
            [1,4,16,32],
            [1,4,8,32],
            [1,16,16,32],
            [1,16,16,32],
        ]
        ks_all = [
            [3,9,21],
            [5,11,21],
            [3,9,21],
            [9,8,21]
        ]
        ps_all = [
            [3,7,11],
            [3,11,11],
            [3,7,11],
            [3,7,11]
        ]
        dr_all = [
            [0.3,0.2,0.1] ,
            [0.1,0.2,0.3],
            [0.3,0.2,0.1] ,
            [0.1,0.2,0.3],
        ]
        # channels =  channels_all[channel_i]
        # kernel_sizes = ks_all[channel_i]
        # pool_sizes = ps_all[channel_i]
        # dropout_rates = dr_all[channel_i]
        '''
        每个模态不加pretrain 46 40 60 71 
        acc : 78
        '''
        channels =  [1,4,8,16]
        kernel_sizes = [5,11,21]
        pool_sizes = [2,2,2]
        dropout_rates = [0.1,0.2,0.3]


        if channel_i == 0 or channel_i== 2:
            channels =  [1,16,16,16]
            kernel_sizes = [3,9,21]
            pool_sizes = [3,7,21]
            dropout_rates = [0.3,0.2,0.1]
        if channel_i == 1:
            channels =  [1,4,8,16]
            kernel_sizes = [3,11,11]
            pool_sizes = [3,11,11]
            dropout_rates = [0.1,0.2,0.3]

        self.conv_module =  nn.Sequential(* conv_block_list(channels,kernel_sizes,pool_sizes,dropout_rates)) 
        # self.att_module = AttentionLayer() 
        
        def output_shape():
            x = self.setup.example_x[self.channel_i] 
            output = self.conv_module(torch.tensor(x,dtype=torch.float32))
            # output = self.att_module(output) 
            # output = output.permute(0,2,1)
            return output.shape
        input_shape = self.setup.example_x[self.channel_i].shape #(1,1,160)
        self.fc = nn.Linear( output_shape()[-1],input_shape[-1])
        self.fc2 = nn.Linear( output_shape()[-2],input_shape[-2])

    
    def forward(self, x,mask=None):
    
        y = self.conv_module(x)
        # y = self.att_module(y)

        if self.froze:
            return y  
       
        y = self.fc(y)
        y = self.fc2(y.permute(0,2,1))
        return y.permute(0,2,1)*mask


