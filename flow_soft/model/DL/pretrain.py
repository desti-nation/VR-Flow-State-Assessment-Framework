from train_util import *
from model_factory import * 
from classifiers.base import *
import torch 
from torch import nn
import time

class PretrainModel(nn.Module):
    def __init__(self,setup,modules):
        super().__init__()
        self.submodules = nn.ModuleList(modules)
        # 下面定义连接部分
        self.flat = nn.Flatten() 

        def calc_out_dim():
            x = setup.example_x
            # print(x[0].shape,x[1].shape,x[2].shape,x[3].shape,'len', len(x))
            x_list = []
            for i,module in enumerate(modules):
                channel_i = module.channel_i 
                x_i = x[channel_i]
                x_list.append(self.submodules[i](torch.tensor(x_i ,dtype=torch.float32)))
            
            x = torch.cat(x_list,dim=2)
            output = self.flat(x)
            # print('calc flat_shape',output.shape)
            return output.shape[-1]
        self.norm = nn.ModuleList([nn.BatchNorm1d(8) for i in range(len(modules))]) 
        # batchnorm1d : 无att 16  有att 8
        self.fc = nn.Linear(calc_out_dim(),setup.label_n)
        self.softmax = nn.Softmax(dim=1)

        
    def forward(self,x):
        outputs = []
        for i,module in enumerate(self.submodules):
            out = module(x[i])
            out = self.norm[i](out)
            outputs.append(out)
            # print('output ',outputs[-1].shape)
        
        x = torch.cat(outputs,dim=2)
        y = self.flat(x)
        y = self.fc(y)
        y = self.softmax(y)

        return y 

def create_pretrained_model(model_name,setup,modules):
    if model_name == 'mix_mlp':
        return  PretrainModel(setup,modules)
   
    else :
        raise NotImplementedError

class PretrainModelExp(BasicExp):
    def __init__(self,pretrain,freeze, setup,model_name,
                 module_name,channels=['eeg','gsr','ppg','blk'],
                 num_epochs=301,train_batch_size=128,lr=1e-5):
        super().__init__(setup,model_name,None,num_epochs,train_batch_size,lr)
        self.pretrain = pretrain
        self.freeze = freeze
        self.module_name = module_name
        self.channels = channels
        # assert len(module_names) == len(channels)
        label_n = setup.label_n
        self.metric_collection = MetricCollection({
            'acc': Accuracy(num_classes=label_n, average="micro"),
            'pre': Precision(num_classes=label_n, average="macro"),
            'rec': Recall( num_classes=label_n, average="macro"),
            'f1': F1Score( num_classes=label_n, average="macro")
        }).to(self.device)
    
    def model_reset(self,k_i=None):
        modules = []
        
        for channel in self.channels:
            channel_i = ['eeg','gsr','ppg','blk'].index(channel)
            if self.pretrain:
                model  = load_model(self.module_name,channel=channel,k_i=k_i,pth=True,model=create_model(self.module_name,self.setup,channel_i=channel_i))
            else:
                model = create_model(self.module_name,self.setup,channel_i=channel_i)
            model.freeze()
            # froze 
            if self.freeze:
                for param in model.parameters():
                    param.requires_grad = False
            modules.append(model)
        
        self.model = create_pretrained_model(self.model_name,self.setup,modules).to(self.device)
        self.optimizer= optim.Adam(self.model.parameters(), lr=self.lr)
    
    def run_model(self, x, y): 
        signal_length = [fs * self.setup.window for fs in self.setup.sample_rates ]
        x_eeg = x[:, 0:1 , :signal_length[0]]  
        x_gsr = x[:, 1:2 , :signal_length[1]] 
        x_ppg = x[:, 2:3 , :signal_length[2]]
        x_blk = x[:, 3:4 , :signal_length[3]] 
        
        all_channel = {'eeg':x_eeg, 'gsr':x_gsr, 'ppg':x_ppg, 'blk':x_blk}
        x = [all_channel[c] for c in self.channels]
        for _ in x:
            assert not np.any(_.cpu().numpy() == -1)
        
        outputs = self.model(x)
        loss = self.loss_fn(outputs, y)

        if self.model.training:
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            self.optimizer.zero_grad()

        # result 
        y_pred = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        y_true = y.detach().cpu().numpy()
        cm = confusion_matrix(y_true, y_pred)
        val_metrics = self.metric_collection.forward(outputs, y)

        return {**val_metrics,**{
            'loss': loss.item(),
            'cm': cm
        }}
    

if __name__ == '__main__':
    from config_init import args
    print(args.modal,'................')
    setup = Setup(window=args.window,step=args.step,label_n=args.label_n)
    wandb.init(project='M-stage', name=f'{args.model}_{args.module}')
    
    wandb.config.update({
        'model': args.model,
        'module':args.module,
        'SSL':args.SSL,
        'freeze':args.freeze,
        'SSL_method':args.SSL_method,
        'epochs':args.epochs,
        'batch':args.batch,
        'lr':args.lr,
        'train_mode':args.train_mode,
        'k':args.k,
        'test_size':args.test_size,
    })
    
    exp = PretrainModelExp(pretrain=args.SSL,
                     freeze=args.freeze,
                     setup=setup,
                     model_name=args.model,
                     module_name=args.module,
                     channels=args.modal,

                     num_epochs=args.epochs,
                     train_batch_size=args.batch,
                     lr=args.lr)
    exp.train(mode=args.train_mode,k=args.k,test_size=args.test_size)
    exp.log(args)
    
    wandb.finish()