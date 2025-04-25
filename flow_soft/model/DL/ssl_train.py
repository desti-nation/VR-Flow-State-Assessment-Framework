from torch import nn 
import torch 
import argparse
import torch
import wandb
from train_util import *
from model_factory import create_model


def generate_mask(data,mask_prob=0.15):
    mask = torch.rand(data.shape) < mask_prob  # 生成掩码
    return mask.to(data.device)


class SSL_EXP(BasicExp):
    def __init__(self,ssl_method,setup,model_name,channel_i,channel_name,num_epochs=301,train_batch_size=128,mask_prob=0.15,predict_length=10,lr=1e-5):
        super().__init__(setup,model_name,nn.MSELoss(),
                         num_epochs,train_batch_size,lr)
        if ssl_method == 'predict':
            setup.example_x = [x[:,:,:-1*predict_length] for x in setup.example_x]

        self.ssl_method = ssl_method
        self.mask_prob = mask_prob
        self.predict_length = predict_length

        self.channel_i = channel_i
        self.channel_name = channel_name

    def model_reset(self,k_i=None):
        self.model = create_model(self.model_name,self.setup,self.channel_i).to(self.device)
        self.optimizer= optim.Adam(self.model.parameters(), lr=self.lr)

    def run_model(self, x, y=None):
        x = x[:,self.channel_i:self.channel_i+1,:self.setup.sample_rates[self.channel_i]*self.setup.window]
        # --------------------predict 
        if self.ssl_method == 'predict':
            assert self.predict_length < x.shape[-1]
            x,y = x[:,:,:-1*self.predict_length],x[:,:,self.predict_length:]
            output = self.model(x)
        elif self.ssl_method == 'mask':
            mask = generate_mask(x,mask_prob=self.mask_prob)
            y = x * mask
            output = self.model(x,mask)
        else:
            raise NotImplementedError
        
        loss = self.loss_fn(output, y)
        if self.model.training:
            # update weight 
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            self.optimizer.zero_grad()

        return {'loss': loss.item()}

   

    def save_model(self,k_i):
        return super().save_model(k_i,self.channel_name) 
    
    def wandb_log(self,metricTracker):
        result = {k+'_'+self.channel_name:v for k,v in metricTracker.items()}
        # wandb.log(result)

    
if __name__ == '__main__':
    from config_init import args

    setup = Setup(window=args.window,step=args.step,label_n=args.label_n)
    wandb.init(project='SSL', name=f'{args.model}_{args.module}')
    
    wandb.config.update({
        'model': args.model,
        'SSL':args.SSL,
        'SSL_method':args.SSL_method,
        'epochs':args.epochs,
        'batch':args.batch,
        'lr':args.lr,
        'train_mode':args.train_mode,
        'k':args.k,
        'test_size':args.test_size,
    })
    
    for i,channel in enumerate( ['eeg','gsr','ppg','blk']):
        print(f"============={channel}=============")
        exp = SSL_EXP(ssl_method=args.SSL_method,
                setup=setup,
                model_name=args.model,
                channel_i=i,
                channel_name=channel,
                num_epochs=args.epochs,
                train_batch_size=args.batch,
                mask_prob=args.mask_prob,
                predict_length=args.predict_length,
                lr=args.lr
        )
        exp.train(mode=args.train_mode,k=args.k,test_size=args.test_size)
        exp.log(args)

        wandb.finish()
    