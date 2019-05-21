#coding=utf8
'''
Created on 2019年5月15日

@author: 82114
'''
import torch as t

class Evaluation():
    def __init__(self,net,pr):
        
        print ("Evaluate....")
        
        dataiter = iter(pr.testloader)
        images, labels = dataiter.next() # 一个batch返回4张图片
        print('实际的label: ', labels)
        #show(tv.utils.make_grid(images / 2 - 0.5)).resize((400,100))
        
        
        # 计算图片在每个类别上的分数
        outputs = net(images)
        # 得分最高的那个类
        _, predicted = t.max(outputs.data, 1)
        
        print('预测结果: ', predicted)
        
        correct = 0 # 预测正确的图片数
        total = 0 # 总共的图片数
        
        
        # 由于测试的时候不需要求导，可以暂时关闭autograd，提高速度，节约内存
        with t.no_grad():
            for data in pr.testloader:
                images, labels = data
                outputs = net(images)
                _, predicted = t.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
                
        print (total)
        print('10000张测试集中的准确率为: %d %%' % (100 * correct / total))
    
        
        