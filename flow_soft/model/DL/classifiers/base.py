from dataset import Dataset
import torch
import torch.nn as nn
class Classifier(nn.Module):
    def __init__(self,setup):
        super().__init__()
        self.setup = setup

        
    def calc_flat_shape(self):
        x = self.setup.example_x 
        print(x[0].shape,x[1].shape,x[2].shape,x[3].shape,'len', len(x))
        
        x_list = [self.module_list[i](torch.tensor(x[i],dtype=torch.float32)) for i in range(Dataset.channel_n)]
        x = torch.cat(x_list,dim=2)
        output = self.flat(x)
        print('calc flat_shape',output.shape)
        return output.shape[-1]


def calc_padding(kernel_size):
    if type(kernel_size) == int:
        return kernel_size//2
    else:
        return tuple(k//2 for k in kernel_size)