import torch
from utils.util import AverageMeter, accuracy
from model import ResNet18, BasicBlock, ConvNet
from data import get_loader
from conf import cfg

device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")

class Trainer:
    def __init__(self, model,optimizer, train_loader,val_loader):
        self.model = model
        self.optimizer= optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.running_loss_train = AverageMeter()
        self.running_acc_train =  AverageMeter()
        self.running_loss_val = AverageMeter()
        self.running_acc_val =  AverageMeter()
        

    def train_one_epoch(self, loss_fn,running_loss, running_acc):

        self.model.train()
        for i, data in enumerate(self.train_loader):
            
            inputs, labels = data['iq_data'], data['label']
            inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.float).argmax(dim=1)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = loss_fn(outputs, labels)
            acc = accuracy(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_acc.update(acc)
            running_loss.update(loss.item())
            
        #     running_loss += loss.item()
        #     if i % 1000 == 999:
        #         last_loss = running_loss/1000
        #         print(f"batch {i+1} loss: {last_loss}")
        #         # tb_x = epoch_index * len(train_loader) + i + 1
        #         # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
        #         running_loss = 0
        # return last_loss

    def test_one_epoch(self, loss_fn,running_loss, running_acc):
        self.model.eval()
        for i, data in enumerate(self.val_loader):
            inputs, labels = data['iq_data'], data['label']
            inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.float).argmax(dim=1)
            outputs = self.model(inputs).detach()
            loss = loss_fn(outputs, labels).detach()
            acc = accuracy(outputs, labels)
            running_acc.update(acc)
            running_loss.update(loss.item())
            
    
    def train(self, n_epochs, loss_fn):
        best_acc = 0
        for epoch in range(n_epochs):
            self.train_one_epoch(loss_fn,self.running_loss_train, self.running_acc_train)
            self.test_one_epoch(loss_fn,self.running_loss_val, self.running_acc_val)
            print(f"EPOCH : {epoch+1} --> Accuracy = {self.running_acc_train.avg} -- Loss = {self.running_loss_train.avg}")
            print(f"Validation  : {epoch+1} --> Accuracy = {self.running_acc_val.avg} -- Loss = {self.running_loss_val.avg}")

            if self.running_acc_val.avg > best_acc:
                best_acc = self.running_acc_val.avg
                torch.save(self.model, ' basic_convnet.pth')

if __name__ == "__main__":
    #model = ResNet18(2,18, block=BasicBlock, num_classes=24)
    model = ConvNet(2,24)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001) 
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,100,0.0005,)
    loss_fn = torch.nn.CrossEntropyLoss() 
    train_loader = get_loader(file_name = 'GOLD_XYZ_OSC.0001_1024.hdf5',
                   snr=10,
                   batch_size = 64,
                   train=True,
                   shuffle=True,
                   num_workers=4)
    
    val_loader = get_loader(file_name = 'GOLD_XYZ_OSC.0001_1024.hdf5',
                   snr=10,
                   batch_size = 64,
                   train=False,
                   shuffle=False,
                   num_workers=4)
    
    trainer = Trainer(model=model,optimizer=optimizer, train_loader=train_loader, val_loader=val_loader)
    trainer.train(n_epochs=600,loss_fn=loss_fn)
