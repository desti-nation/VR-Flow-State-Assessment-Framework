from train_util import * 
from model_factory import * 

class PlainExp(BasicExp):
    def __init__(self,setup,
                 model_name,
                 num_epochs=301,train_batch_size=128,lr=1e-3):
        
        super().__init__(setup,model_name,None,num_epochs,train_batch_size,lr)
        label_n = self.setup.label_n 
        # self.dataset = get_dataset(setup,sample_rates=[512,512,512,512],read=False,save=False)
        # with open(os.path.join(Folder.root, 'dataset_deepflow.pkl'), 'wb') as file:
        #     pickle.dump(self.dataset, file, protocol=4)
        self.metric_collection = MetricCollection({
            'acc': Accuracy(num_classes=label_n, average="micro"),
            'pre': Precision(num_classes=label_n, average="macro"),
            'rec': Recall( num_classes=label_n, average="macro"),
            'f1': F1Score( num_classes=label_n, average="macro")
        }).to(self.device)
    def wandb_log(self, metricTracker):
        return
    def run_model(self, x, y): 
        signal_length = [fs * self.setup.window for fs in self.setup.sample_rates ]
        x_eeg = x[:, 0:1 , :signal_length[0]]  
        x_gsr = x[:, 1:2 , :signal_length[1]] 
        x_ppg = x[:, 2:3 , :signal_length[2]]
        x_blk = x[:, 3:4 , :signal_length[3]] 
        
        x = [x_eeg, x_gsr, x_ppg, x_blk]
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

    setup = Setup(window=10)
    # wandb.init(project='plain', name=f'{args.model}_{args.module}')
    
    # wandb.config.update({
    #     'model': args.model,
    #     'SSL':args.SSL,
    #     'SSL_method':args.SSL_method,
    #     'epochs':args.epochs,
    #     'batch':args.batch,
    #     'lr':args.lr,
    #     'train_mode':args.train_mode,
    #     'k':args.k,
    #     'test_size':args.test_size,
    # })
    
    
    exp = PlainExp(
            setup=setup,
            model_name=args.model,
            num_epochs=args.epochs,
            train_batch_size=args.batch,
            lr=args.lr
    )
    exp.train(mode=args.train_mode,k=args.k,test_size=args.test_size)
    exp.log(args)

    # wandb.finish()
    
