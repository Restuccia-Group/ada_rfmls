import torch

class AverageMeter():
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0 
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n 
        self.avg = self.sum / self.count 

def accuracy(output, target):
    acc=0
    acc += (output.max(1)[1] == target).float().sum()
    batch_size = target.size()[0]
    return acc.mul_(100.0/batch_size)