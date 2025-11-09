import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
batch_size = 64

transform = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.1307,),(0.3081,))])

train_data = datasets.MNIST(root='../dataset/mnist',
                            train=True,
                            download=False,
                            transform=transform)

test_data = datasets.MNIST(root='../dataset/mnist',
                           train=False,
                            download=False,
                            transform=transform)

train_loader = DataLoader(dataset=train_data,
                          shuffle=True,
                          batch_size=batch_size)

test_loader = DataLoader(dataset=test_data,
                          shuffle=False,
                          batch_size=batch_size)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.linear1 = torch.nn.Linear(784,524)
        self.linear2 = torch.nn.Linear(524,256)
        self.linear3 = torch.nn.Linear(256,128)
        self.linear4 = torch.nn.Linear(128,64)
        self.linear5 = torch.nn.Linear(64,10)

    def forward(self,x):
        x = x.view(-1,784)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = self.linear5(x)
        return x
    
model = Net()
criterion = torch.nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.5)
def train(epoch):
    running_loss = 0.0
    for index, data in enumerate(train_loader, 0):
        input, output = data
        y = model(input)
        loss = criterion(y, output)

        running_loss += loss.item()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if (index+1)%100 == 0:
            print(f"第{epoch+1}个epoch - 第{index+1}个bacthsize时, Loss= {running_loss/100:.4f}")
            running_loss = 0
def test():
    flag = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            input,output = data
            y = model(input)
            _,y_pred = torch.max(y.data,dim=1)
            total += output.size(0)
            flag += (y_pred == output).sum().item()
    print(f"准确率:{100 * flag/total:.4f}%")        
for epoch in range(5):
    train(epoch)
    test()