#coding=utf8
'''
Created on 2019年5月15日

@author: 82114
'''
import torch.optim as optim
import torch.nn as nn
import torch as t

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

class Train():
    def __init__(self,net,pr):
        self.net = net
        self.pr = pr
        
    
    def __call__(self,EPOCH):
        self.train(EPOCH)
    
    
    def train(self,EPOCH):
        print ("training...")
        LR = 0.1
        
        criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
        optimizer = optim.SGD(self.net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
        #t.set_num_threads(8)
        for epoch in range(EPOCH):  
            
            if (epoch+1) % 50 == 0:
                LR = LR * 0.1
                
                for params in optimizer.param_groups:
                    params['lr'] = LR
            
            
            running_loss = 0.0
            for i, data in enumerate(self.pr.trainloader, 0):
                
                # 输入数据
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                
                # 梯度清零
                optimizer.zero_grad()
                
                # forward + backward 
                outputs = self.net(inputs)
                #print (len(outputs[0]))
                loss = criterion(outputs, labels)
                loss.backward()   
                
                # 更新参数 
                optimizer.step()
                
                # 打印log信息
                # loss 是一个scalar,需要使用loss.item()来获取数值，不能使用loss[0]
                running_loss += loss.item()
                if i % 2000 == 1999: # 每2000个batch打印一下训练状态
                    print('[%d, %5d] loss: %.3f' \
                          % (epoch+1, i+1, running_loss / 2000))
                    running_loss = 0.0
                    
        print('Finished Training')
        
        
        
