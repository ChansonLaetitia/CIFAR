#coding=utf8
'''
Created on 2019年5月15日

@author: 82114
'''
import torch.optim as optim
import torch.nn as nn

class Train():
    def __init__(self,net,pr):
        
        criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
        #t.set_num_threads(8)
        for epoch in range(2):  
            
            running_loss = 0.0
            for i, data in enumerate(pr.trainloader, 0):
                
                # 输入数据
                inputs, labels = data
                
                # 梯度清零
                optimizer.zero_grad()
                
                # forward + backward 
                outputs = net(inputs)
                #print (len(outputs[0]))
                loss = criterion(outputs, labels)
                loss.backward()   
                
                # 更新参数 
                optimizer.step()
                
                # 打印log信息
                # loss 是一个scalar,需要使用loss.item()来获取数值，不能使用loss[0]
                running_loss += loss.item()
                if i % 500 == 499: # 每2000个batch打印一下训练状态
                    print('[%d, %5d] loss: %.3f' \
                          % (epoch+1, i+1, running_loss / 500))
                    running_loss = 0.0
        print('Finished Training')
