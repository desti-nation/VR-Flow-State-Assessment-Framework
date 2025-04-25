# phase1: Self-supervised Pre-training
python model/DL/ssl_train.py -model rep_encoder -SSL_method mask -exp_name SSL_EXP -train_mode normal-random -k -1 -test_size 0.2 -epochs 1501 -lr 0.005 -batch 1024

# phase2: Supervised Fine-tuning
python model/DL/pretrain.py -model mix_mlp -module rep_encoder -SSL -SSL_method mask -exp_name EXP -k -1 -lr 0.005 -batch 1024
