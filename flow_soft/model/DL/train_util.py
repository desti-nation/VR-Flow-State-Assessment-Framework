from collections import namedtuple
from numbers import Number
from matplotlib import pyplot as plt
from torch import nn 
import torch 
import torch.optim as optim
import numpy as np
import argparse
import random
import csv 
import time
import numpy as np
import pickle
import torch
import os, sys
from dataset import Dataset,unpickle,Setup,get_dataset
from filepaths import File,Folder
from torchmetrics import Accuracy, MetricCollection, Precision, Recall
from torchmetrics.classification.f_beta import F1Score 
# load dataset
from sklearn.model_selection import train_test_split, cross_validate, KFold
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, recall_score, classification_report, precision_score, roc_auc_score, \
    confusion_matrix
from model_factory import *
from filepaths import File
import torch.nn as nn
import torch.optim as optim

import wandb
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def divide(subjects,k=5):
    # 将集合转换为列表并随机打乱
    random.seed(42) 
    items = list(subjects)
    random.shuffle(items)
    
    # 计算每份的元素数量
    total = len(items)
    parts = [total // k] * k
    for i in range(total % k):  # 调整余数个元素
        parts[i] += 1
    
    # 分割列表
    divided_parts = []
    start_index = 0
    for part in parts:
        end_index = start_index + part
        divided_parts.append(items[start_index:end_index])
        start_index = end_index
    
    return divided_parts


def load_model(model_name,channel=None,k_i=None,pkl=False,pth=True,froze=False,model=None):#,opt=None):
    channel = f'{channel}_' if channel else ''
    k_i = f'{k_i}_' if k_i is not None else ''

    model_path = os.path.join(Folder.modelResult , 'ours',  f'{k_i}{channel}{model_name}.pth' )
    print(model_path)
    model_pkl_path = os.path.join(Folder.modelResult,'ours', f'{k_i}{channel}{model_name}_model.pkl')
    # opt_pkl_path = os.path.join(Folder.modelResult,f'{channel}{model_name}_opt.pkl')

    if pkl:
        model = unpickle(model_pkl_path)
        # opt = unpickle(opt_pkl_path)
        
    elif pth:
        # 加载用 torch.save 保存的模型和优化器状态
        checkpoint = torch.load(model_path)
        model_state_dict = checkpoint['model_state_dict']
        # optimizer_state_dict = checkpoint['optimizer_state_dict']

        model.load_state_dict(model_state_dict)
        # opt.load_state_dict(optimizer_state_dict)
    
    if froze:
        for param in model.parameters():
            param.requires_grad = False
            # 设置为评估模式
        model.eval()
    return model#,opt

class MetricTracker:
    def __init__(self):
        self.reset()
    def reset(self):
        self.metrics = None
        
    def update(self,metrics:dict):
        
        if self.metrics is None:
            self.metrics = {}
            for k,v in metrics.items():
                if torch.is_tensor(v) and v.device.type != 'cpu':
                    if v.dim() == 0:
                        v = v.cpu().item()
                    else:
                        v = v.cpu().numpy() 
                self.metrics[k] = [v] 
        else:
            for k,v in metrics.items():
                if torch.is_tensor(v) and v.device.type != 'cpu':
                    if v.dim() == 0:
                        v = v.cpu().item()
                    else:
                        v = v.cpu().numpy() 
                self.metrics[k].append(v) 
                

    def show(self):
        for k,v in self.metrics.items():
            if k!='cm' :
                print(f'{k}:\n{np.mean(v)} | {v}')
            else:
                print(f'{k}: {v}')
    def toString(self):
        str_log = {}
        for k,v in self.metrics.items():
            if k!='cm' :
                str_log[k] = (f'{np.mean(v)}', f'{v}')
            else:
                str_log[k] = f'{v}'
        return str_log
    def avg(self):
        avg_metrics = {}
        for k,v in self.metrics.items():
            if k!='cm' :
                avg_metrics[k] = np.mean(v)
        return avg_metrics
    
class BasicExp:
    def __init__(self,setup,
                 model_name,loss_fn=None,
                 num_epochs=301,train_batch_size=128,lr=1e-5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.setup = setup
        self.model_name = model_name
        self.loss_fn = loss_fn if loss_fn else nn.CrossEntropyLoss().to(self.device)

        self.num_epochs = num_epochs
        self.train_batch_size = train_batch_size
        self.lr = lr
        self.dataset = get_dataset(setup)

    def model_reset(self,k_i=None):
        self.model = create_model(self.model_name,self.setup).to(self.device)
        self.optimizer= optim.Adam(self.model.parameters(), lr=self.lr)
        
    def train(self,mode='cross-subject', k=5, test_size=0.2):
        self.startTime = time.time()
        self.resultTracker = MetricTracker()
        if k is not None and k > 0:
            self.k_fold_validate(k,mode)
        else:
            self.model_reset()
            if mode == 'cross-subject':
                subject_lists = divide(self.setup.subjects,int(1/test_size))
                train_subjects,test_subjects = sum(subject_lists[:-1],[]),subject_lists[-1]

                print('test_subjects:',test_subjects)

                x_train,y_train = self.dataset.merge_subjects(train_subjects)
                x_test, y_test = self.dataset.merge_subjects(test_subjects)

                x_train,y_train = self.reshape_dataset(x_train,y_train)
                x_test,y_test  = self.reshape_dataset(x_test,y_test)
            elif mode == 'normal-random':
                X,Y = self.reshape_dataset(self.dataset.x,self.dataset.y)
                x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=test_size,random_state=42)
            print(x_train.shape,x_test.shape,'----------------')
            dataset = TensorDataset(x_train, y_train)
            dataloder = DataLoader(dataset, batch_size=self.train_batch_size, shuffle=True)
            result = self.__train__(dataloder,x_test,y_test)
            self.resultTracker.update(result)
        
        self.resultTracker.show() 
        print('.........train done.........')

    def reshape_dataset(self,X,Y):
        '''
        形式同unpickle直接解压得到的dataset.x,dataset.y
        '''
        X = np.transpose(X, (1, 0, 2)) # (C,N,L) -> (N,C,T)
        X = torch.tensor(X, dtype=torch.float32).to(self.device)  # (N,C,1,T)  
        Y = torch.tensor(Y, dtype=torch.long).to(self.device)  
        return X,Y

    def k_fold_validate(self,k,mode='cross-subject'):
        kf = KFold(n_splits=k, shuffle=True, random_state=32)  # 初始化KFold
        if mode == 'normal-random':
            X,Y = self.reshape_dataset(self.dataset.x,self.dataset.y)
            i = 0

            for train_index, test_index in kf.split(X):  # 调用split方法切分数据
                self.model_reset(k_i=i+1)

                x_train,y_train = X[train_index],Y[train_index]
                x_test,y_test = X[test_index],Y[test_index]
                print(f'kfold@{i+1}.........')
                print(f'x_train: {x_train.shape}, y_train: {y_train.shape}, x_test: {x_test.shape}, y_test: {y_test.shape}')
            
                dataset = TensorDataset(x_train, y_train)
                k_data_loader = DataLoader(dataset, batch_size=self.train_batch_size, shuffle=True)

                result_in_test = self.__train__(k_data_loader,x_test,y_test,k_i=i+1)
                # print(result_in_test)
                self.wandb_log({ f'{k}_test' :v for k,v in result_in_test.items() if k != 'cm'})
                self.resultTracker.update(result_in_test)
                self.resultTracker.show()

                i += 1
            
        elif mode == 'cross-subject':
            subject_lists = divide(self.setup.subjects,k)  #划分成k份
            for i in range(k):
                self.model_reset(k_i=i+1)

                train_subjects = []
                test_subjects = subject_lists[i]
                for j in range(k):
                    if i != j:
                       train_subjects += subject_lists[j]
                print(f'kfold@{i+1}.........')
                print('test_subjects:',test_subjects)

                x_train,y_train = self.dataset.merge_subjects(train_subjects)
                x_test, y_test = self.dataset.merge_subjects(test_subjects)

                x_train,y_train = self.reshape_dataset(x_train,y_train)
                x_test,y_test  = self.reshape_dataset(x_test,y_test)
                
                print(f'x_train: {x_train.shape}, y_train: {y_train.shape}, x_test: {x_test.shape}, y_test: {y_test.shape}')
            
                dataset = TensorDataset(x_train, y_train)
                k_data_loader = DataLoader(dataset, batch_size=self.train_batch_size, shuffle=True)

                result_in_test = self.__train__(k_data_loader,x_test,y_test,k_i=i+1)
                # print(result_in_test)
                self.wandb_log({ f'{k}_test' :v for k,v in result_in_test.items() if k != 'cm'})
                self.resultTracker.update(result_in_test)
                self.resultTracker.show()
        else:
            raise Exception('mode not supported')

    def __train__(self,train_loader,x_test,y_test,k_i=None):
        best_result = None
        for epoch in range(1, self.num_epochs):
            self.model.train() 
            
            metricTracker = MetricTracker()
            for i,(inputs,targets) in enumerate(train_loader):
                result_in_train = self.run_model(inputs,targets) 
                metricTracker.update(result_in_train) 

            self.wandb_log(metricTracker.avg())
            
            if epoch % 100 != 0 and epoch != self.num_epochs - 1:
                continue
            
            '''
            test model -> loss/acc in test  
            print best result and save model 
            record final result
            '''
            self.model.eval()
            result_in_test = self.run_model(x_test,y_test)
            save = self.update_best_result(cur=result_in_test,best=best_result) 

            best = result_in_test if save else best 
            if save:
                self.save_model(k_i=k_i)

            if epoch == self.num_epochs - 1:
                return result_in_test
    def wandb_log(self,metricTracker):
        wandb.log(metricTracker)
    
    def run_model(self,x,y=None):
        pass 
        
    def update_best_result(self,cur,best,item=None):
        assert cur is not None
        if best is None:
            return True
        
        if item is None:
            item = 'acc' if 'acc' in cur.keys() else 'loss' if 'loss' in cur.keys() else None

        assert item is not None
        assert item in cur.keys()
        assert item in best.keys()

        save = False
        if item == 'loss':
            save = cur[item]<best[item]
        elif item in ['acc','pre','rec','f1']:
            save = cur[item]>best[item]
        else:
            raise ValueError('item must be in loss,acc,pre,rec,f1')
        
        return save
        
    def save_model(self,k_i=None,channel=None):
        channel = f'{channel}_' if channel else ''
        k_i =  f'{k_i}_' if k_i else ''
        '''save model'''
        # model_name = '+'.join(modals) + '_lr' + str(lr) #+ '_' + str(time.time()) 
        torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, os.path.join(Folder.modelResult, 'ours', f'{k_i}{channel}{self.model_name}.pth'))
                
        with open(os.path.join(Folder.modelResult,'ours', f'{k_i}{channel}{self.model_name}_model.pkl'), 'wb') as file:
            pickle.dump(self.model, file)

        with open(os.path.join(Folder.modelResult, 'ours',f'{k_i}{channel}{self.model_name}_opt.pkl'), 'wb') as file:
            pickle.dump(self.optimizer, file)
    
    def log(self,args,empty=False,logFile='model_25-2-18.csv'):
        header = [
            'time', # 时间
            'model_name', 'module_name','modal_list',
            # 训练结果
            'acc_avg','acc', 'pre_avg', 'pre','rec_avg' ,'rec', 'f1_avg', 'f1', 'cm', 'loss_avg','loss',
            # 训练参数
            'epochs', 'train_batch_size', 'lr',
            'SSL','SSL_method','freeze',
            'train_mode','k','test_size',
            # 数据集
            'label_n', 'window_overlap', 'subject_n','norm',
            'exp_name',
        ]# cross subject or not   (k=5)

        # header = [
        #      'exp_name',
        #      'acc_avg','acc', 'pre_avg', 'pre','rec_avg' ,'rec', 'f1_avg', 'f1', 
        #      'lr','hidden_dim','num_layers'
        # ]
        Log = namedtuple('Log', header)
            
        if empty:
            with open(logFile, 'a', newline='') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(['' for i in range(len(Log._fields))])
            return
        
        result_tracker = self.resultTracker.toString()
        
        for item in ['acc','pre','rec','f1','cm','loss']:
            if item not in result_tracker:
                result_tracker[item] = None if item =='cm' else (None,None)

        log = Log(
            time=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            model_name=args.model,
            module_name=args.module,
            modal_list=args.modal,

            acc_avg=result_tracker['acc'][0],
            acc=result_tracker['acc'][1],
            pre_avg=result_tracker['pre'][0],
            pre=result_tracker['pre'][1],
            rec_avg=result_tracker['rec'][0],
            rec=result_tracker['rec'][1],
            f1_avg=result_tracker['f1'][0],
            f1=result_tracker['f1'][1],
            cm=result_tracker['cm'],
            loss_avg=result_tracker['loss'][0],
            loss=result_tracker['loss'][1],

            epochs=args.epochs,
            train_batch_size=args.batch,
            lr=args.lr,

            SSL=args.SSL,
            SSL_method=args.SSL_method,
            freeze=args.freeze,

            train_mode=args.train_mode,
            k=args.k,
            test_size=args.test_size,

            label_n=self.setup.label_n,
            window_overlap=f'{self.setup.window}-{1-self.setup.step}',
            subject_n=len(self.setup.subjects),
            norm=self.setup.normalize,

            exp_name=args.exp_name,
        )

        # log = Log(
        #     exp_name=args.exp_name,
        #     acc_avg=result_tracker['acc'][0],
        #     acc=result_tracker['acc'][1],
        #     pre_avg=result_tracker['pre'][0],
        #     pre=result_tracker['pre'][1],
        #     rec_avg=result_tracker['rec'][0],
        #     rec=result_tracker['rec'][1],
        #     f1_avg=result_tracker['f1'][0],
        #     f1=result_tracker['f1'][1],
        #     lr=args.lr,
        #     hidden_dim=args.hidden_dim,
        #     num_layers=args.num_layers,
        # )

        with open(logFile, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            if os.stat(logFile).st_size == 0:
                writer.writerow(Log._fields)
            writer.writerow(log)
