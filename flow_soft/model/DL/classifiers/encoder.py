
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

class ClassifierEncoder(Classifier):
    def __init__(self,setup):
        super().__init__(setup)

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
        
        channels = [
            [1,4,8,16],
            [1,4,8,16],
            [1,4,8,16],
            [1,4,8,16],
        ]
        kernel_sizes = [
            [5,11,21],
            [5,11,21],
            [5,11,21],
            [5,11,21],
        ]
        pool_sizes = [
            [2,2,2],
            [2,2,2],
            [2,2,2],
            [2,2,2],
        ]
        dropout_rates = [
            [0.1,0.2,0.3],
            [0.1,0.2,0.3],
            [0.1,0.2,0.3],
            [0.1,0.2,0.3],
        ]
       
        self.conv_module_list = [
            nn.Sequential(* conv_block_list(channels[i],kernel_sizes[i],pool_sizes[i],dropout_rates[i])) 
            for i in range(Dataset.channel_n)
        ]
      
        self.att_module_list = [
            AttentionLayer() for i in range(Dataset.channel_n)
        ]
        
        def att_output_shape():
            x = self.setup.example_x 
            x_list = [self.conv_module_list[i](torch.tensor(x[i],dtype=torch.float32)) for i in range(Dataset.channel_n)]
            x_list = [self.att_module_list[i](x_list[i]) for i in range(Dataset.channel_n)]
            
            return [_.shape[-1] for _ in x_list]
    
        self.output_module_list = [
            nn.Sequential(
                nn.Linear(att_output_shape()[i], att_output_shape()[i] ),
                nn.InstanceNorm1d(att_output_shape()[i]),
                nn.Flatten()
            ) for i in range(Dataset.channel_n)
        ]

        [self.conv_module1,self.conv_module2,self.conv_module3,self.conv_module4] = self.conv_module_list
        [self.att_module1,self.att_module2,self.att_module3,self.att_module4] = self.att_module_list
        [self.output_module1,self.output_module2,self.output_module3,self.output_module4] = self.output_module_list

        
       
        self.flat = nn.Flatten()
        def dense_input_shape():
            x = self.setup.example_x 
            x_list = [self.conv_module_list[i](torch.tensor(x[i],dtype=torch.float32)) for i in range(Dataset.channel_n)]
            x_list = [self.att_module_list[i](x_list[i]) for i in range(Dataset.channel_n)]
            x_list = [self.output_module_list[i](x_list[i]) for i in range(Dataset.channel_n)]
            
            output = torch.cat(x_list, dim=1)
            output = self.flat(output)
            return output.shape[-1]
        self.fc = nn.Linear( dense_input_shape(), setup.label_n)
        self.softmax = nn.Softmax(dim=1)

    
    def forward(self, x):

        outputs = []
        for i in range(Dataset.channel_n):
            output = self.conv_module_list[i](x[i])
            output = self.att_module_list[i](output)
            output = self.output_module_list[i](output)
            outputs.append(output)
        output = torch.cat(outputs,1)
        output = self.flat(output)
        output = self.fc(output)
        output = self.softmax(output)
        return output


