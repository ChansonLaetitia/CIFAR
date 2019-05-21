#coding=utf8
'''
Created on 2019年5月15日

@author: 82114
'''
from preprocessing import preprocessing
from models import simplenet,resnet34,resnet18
from train import train
from evaluation import evaluation

EPOCH = 150


if __name__ == '__main__':
    pr = preprocessing.Preprocessing()
    #net = simplenet.Net()
    #net = resnet34.ResNet34(10)
    net = resnet18.ResNet18()
    
    tr = train.Train(net,pr)
    tr(EPOCH)
    
    ev = evaluation.Evaluation(net,pr)
    
    
    

    