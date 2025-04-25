from SSL.rep_cnn import Rep_CNN
from SSL.rep_att import Rep_Att
from SSL.encoder import Rep_Encoder
from SSL.encoder_noatt import Rep_Encoder_noAtt
from SSL.encoder2 import Rep_Encoder2
from SSL.encoder_predict import Rep_Encoder_predict
from SSL.mcdcnn import Rep_McdCNN
from classifiers.fcn import ClassifierFcn
from classifiers.mlp import ClassifierMLP
from classifiers.mcdcnn import ClassifierMcdCNN
from classifiers.encoder import ClassifierEncoder
from classifiers.resnet import ClassifierResnet
from classifiers.deepflow import DeepFlow
from classifiers.n_deepflow import N_DeepFlow

def create_model(model_name,setup,channel_i=None):
    if model_name == 'rep_cnn':
        return Rep_CNN(setup,channel_i)
    elif model_name == 'rep_encoder' :
        return Rep_Encoder(setup,channel_i)
    elif model_name == 'rep_encoder_noatt':
        return Rep_Encoder_noAtt(setup,channel_i)
    elif model_name == 'rep_encoder2':
        return Rep_Encoder2(setup,channel_i)
    elif model_name == 'rep_encoder_predict':
        return Rep_Encoder_predict(setup,channel_i)
    elif model_name == 'rep_mcdcnn':
        return Rep_McdCNN(setup,channel_i)
    # elif model_name == 'rep_att':
    #     return Rep_Att()
    elif model_name == 'fcn':
        return ClassifierFcn(setup)
    if model_name == 'mlp':
        return ClassifierMLP(setup)
    if model_name == 'mcdcnn':
        return ClassifierMcdCNN(setup)
    if model_name == 'encoder':
        return ClassifierEncoder(setup)
    if model_name == 'resnet':
        return ClassifierResnet(setup)
    if model_name == 'deepflow':
        return DeepFlow(setup)
    if model_name == 'n_deepflow':
        return N_DeepFlow(setup)
  
    else:
        raise ValueError('Invalid model name')

