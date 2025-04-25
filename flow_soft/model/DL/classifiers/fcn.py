
from  classifiers.base import *
from dataset import Dataset
import torch.nn as nn
import torch 
'''
conv input-output : 
Input kernel_size=k, stride=s, padding=p -> Output \floor { (I-k+2p)/s+1 } 
'''

def convBlock(dim,in_channels,out_channels,kernel_size,padding=-1):
    '''
    input : N,C,T     | C:in_channels 
    output : N,C',T'  | C':out_channels
    '''
    conv = None
    
    if padding == -1:
        padding = calc_padding(kernel_size)
    if dim == 1:
        conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size,stride=1,  padding=padding)
        # 对每个特征通道的数据进行归一化处理 加速模型收敛
        Batch_normalization = nn.BatchNorm1d(out_channels)
    elif dim == 2:
        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size,  padding=padding)   
        Batch_normalization = nn.BatchNorm2d(out_channels)   
    elif dim == 3:  
        conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, padding=padding)
        Batch_normalization = nn.BatchNorm3d(out_channels)
    activation = nn.ReLU(inplace=True)
    return nn.Sequential(conv, Batch_normalization, activation)

class ClassifierFcn(Classifier):
    def __init__(self,setup):
        super(ClassifierFcn, self).__init__(setup)
        # todo params
        kernel_size_list = [
            [64,32], # fs/4  
            [64,32], # fs/4  
            [64,32], # fs/4  
            [64,32], # fs/4  
        ]

        channel_list = [
            [1,4,8],
            [1,4,8],
            [1,4,8],
            [1,4,8],
        ]
        avgpool_list = [
           2,2,2,2
        ]

        def _generate_3block(kernel_sizes,channels,dim=1):
            blocks = []
            nn.Sequential()
            for i in range(len(kernel_sizes)):
                blocks.append(convBlock(dim=dim,in_channels=channels[i],out_channels=channels[i+1], kernel_size=kernel_sizes[i]))
            return blocks
    
        self.module_list = [ nn.Sequential( *_generate_3block(kernel_size_list[i],channel_list[i]), nn.AvgPool1d(avgpool_list[i])) for i in range(Dataset.channel_n) ]

        # [self.module1,self.module2,self.module3,self.module4] = self.module_list
        if Dataset.channel_n == 4:
            [self.module1,self.module2,self.module3,self.module4] = self.module_list
        elif Dataset.channel_n == 3:
            [self.module1,self.module2,self.module3] = self.module_list
        elif Dataset.channel_n == 2:
            [self.module1,self.module2] = self.module_list
        elif Dataset.channel_n == 1:
            self.module1 = self.module_list[0]

        self.flat = nn.Flatten()
        
        def calc_flat_shape():
            x = setup.example_x 
            print(x[0].shape,x[1].shape,x[2].shape,x[3].shape,'len', len(x))
           
            x_list = [self.module_list[i](torch.tensor(x[i],dtype=torch.float32)) for i in range(Dataset.channel_n)]
            # print('x_list',x_list[0].shape,x_list[1].shape,x_list[2].shape,x_list[3].shape,'len', len(x_list))
            # torch.Size([460, 64, 480]) torch.Size([460, 64, 480]) torch.Size([460, 64, 480]) torch.Size([460, 64, 60]) len 4
            x = torch.cat(x_list,dim=2)
            output = self.flat(x)
            return output.shape[-1]

     
        self.dense = nn.Linear(calc_flat_shape(),setup.label_n)
        self.softmax = nn.Softmax(dim=1)
       

    def forward(self, x):

        output = []

        for i in range(Dataset.channel_n):
            output.append(self.module_list[i](x[i]))

        x = torch.cat(output,dim=2)

        y = self.flat(x)
        y = self.dense(y)
        y = self.softmax(y)

        # print(y)
        return y 
    
