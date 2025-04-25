import torch.nn as nn
import torch 
from  classifiers.base import *
from dataset import Dataset
class ClassifierMLP(Classifier):
    def __init__(self,setup):
        super().__init__(setup)
        
        def mlp_block(channels,dropout_rate1=0.1,dropout_rate2=0.4,dropout_rate3=0.6):

            blocks = []
            blocks += [
                nn.Dropout(dropout_rate1),
                nn.Linear(channels[0], channels[1]),
                nn.ReLU()
            ]

            if len(channels) > 2:
                for i in range(1,len(channels)-1):
                    blocks += [
                        nn.Dropout(dropout_rate2),
                        nn.Linear(channels[i], channels[i+1]),
                        nn.ReLU()
                    ]
            blocks += [
                nn.Dropout(dropout_rate3),
            ]
            return nn.Sequential(*blocks)
      
        basic_shape = [setup.example_x[i].shape[-1] for i in range(Dataset.channel_n)]
        channels = [
            [basic_shape[i], basic_shape[i] // 16,basic_shape[i]//64, basic_shape[i]//256 ] for i in range(Dataset.channel_n)
        ]
        
        self.module_list = [mlp_block(channels[i]) for i in range(Dataset.channel_n)]
        if Dataset.channel_n == 4:
            [self.module1,self.module2,self.module3,self.module4] = self.module_list
        elif Dataset.channel_n == 3:
            [self.module1,self.module2,self.module3] = self.module_list
        elif Dataset.channel_n == 2:
            [self.module1,self.module2] = self.module_list
        elif Dataset.channel_n == 1:
            self.module1 = self.module_list[0]

        self.flat = nn.Flatten()
        self.dense = nn.Linear(self.calc_flat_shape(),  setup.label_n) 
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        output = []

        for i in range(Dataset.channel_n):
            output.append(self.module_list[i](x[i]))

        x = torch.cat(output,dim=2)
        y = self.flat(x)
        y = self.dense(y)
        y = self.softmax(y)
        return y 

