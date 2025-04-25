 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-window", type=int,default=10)
parser.add_argument("-step", type=float,default=0.5)
parser.add_argument("-label_n", type=int,default=3)

parser.add_argument("-model", type=str,default='') 
parser.add_argument("-module", type=str,default='') 
parser.add_argument("-modal", type=str ,nargs='*',default=['eeg','gsr','ppg','blk'])
parser.add_argument("-SSL", default=None,action='store_true') 
parser.add_argument("-SSL_method", type=str,default=None) 
parser.add_argument("-freeze", default=None,action='store_true')
parser.add_argument("-mask_prob", type=float,default=0.15) 
parser.add_argument("-predict_length", type=int,default=10) 
parser.add_argument("-epochs", type=int, default=901)
parser.add_argument("-batch", type=int, default=128)
parser.add_argument("-lr", type=float, default=1e-3)
# 训练参数
parser.add_argument("-train_mode", type=str,default='normal-random')
parser.add_argument("-k", type=int,default=5)
parser.add_argument("-test_size", type=float,default=None)

parser.add_argument("-exp_name", type=str,default='')
parser.add_argument("-num_layers", type=int,default=2)
parser.add_argument("-hidden_dim", type=int ,default=16)

args = parser.parse_args()


