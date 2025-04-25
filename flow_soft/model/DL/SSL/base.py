from  torch import nn 

class Representation(nn.Module):
    def __init__(self,setup,channel_i):
        super().__init__()
        self.froze = False
        self.setup = setup
        self.channel_i = channel_i
    def freeze(self):
        self.froze = True
