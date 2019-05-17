#coding=utf8
'''
Created on 2019年5月15日

@author: 82114
'''
import torch as t
import torchvision as tv
from torchvision import transforms as T
#from torchvision.transforms import ToPILImage
from preprocessing import cutout

class Preprocessing(object):
    
    def __init__(self):
        # 定义对数据的预处理
        transform_train = T.Compose([
                #T.RandomCrop(32, padding=4),
                #cutout.Cutout(6),
                #T.RandomHorizontalFlip(),
                T.ToTensor(), # 转为Tensor
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 归一化
                                     ])
        
        transform_test = T.Compose([
                T.ToTensor(), # 转为Tensor
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 归一化
                                     ])
        
        # 训练集
        trainset = tv.datasets.CIFAR10(
                            root='C:\\Users\\82114\\ML\\cifar\\data\\', 
                            train=True, 
                            download=True,
                            transform=transform_train)
        
        self.trainloader = t.utils.data.DataLoader(
                            trainset, 
                            batch_size=4,
                            shuffle=True, 
                            num_workers=2)
        
        # 测试集
        testset = tv.datasets.CIFAR10(
                            'C:\\Users\\82114\\ML\\cifar\\data\\',
                            train=False, 
                            download=True, 
                            transform=transform_test)
        
        self.testloader = t.utils.data.DataLoader(
                            testset,
                            batch_size=4, 
                            shuffle=False,
                            num_workers=2)
        
        self.classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    