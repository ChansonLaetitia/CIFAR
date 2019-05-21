#coding=utf8
'''
Created on 2019年5月15日

@author: 82114
'''
import torch as t
from preprocessing import preprocessing
from models import simplenet,resnet34,resnet18
from train import train
from evaluation import evaluation


EPOCH = 200

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")


if __name__ == '__main__':
    pr = preprocessing.Preprocessing()
    #net = simplenet.Net()
    #net = resnet34.ResNet34(10)
    net = resnet18.ResNet18()
    
    net = net.to(device)
    
    tr = train.Train(net,pr)
    tr(EPOCH)
    
    ev = evaluation.Evaluation(net,pr)
    
    
    

    