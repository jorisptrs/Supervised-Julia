import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np

# TODO lienar dimensions aoutomatic

class Net(nn.Module):

    def __init__(self, w):
        super(Net, self).__init__()
        
        self.w = w
        self.optimizer = None

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(4 * 7 * 7, 2)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
    
    
    def batch(self, x, y, loss_func):
        yhat = self.forward(x)
        loss = loss_func(yhat, y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
         



    def train(self, trainLoader, device, loss_func, epochs = 20):


        for epoch in range(epochs):

            running_loss = 0.0

            print("Epochs: " + str(epoch + 1) + " out of " + str(epochs))

            for i, data in enumerate(trainLoader, 0):

                x, y = data

                x = x.to(device)
                y = y.to(device)

                running_loss += self.batch(x, y, loss_func)
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

            