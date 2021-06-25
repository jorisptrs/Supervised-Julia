
import torch
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.optimizer = None
        self.float()

        self.cnn_layers = nn.Sequential(

            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 2nd convolution layer
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(2048, 2)
        )
            
    def forward(self, x):
        """
        Defining the forward pass 
        """
        x = x.unsqueeze(dim=1)
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

    def batch(self, x, y, loss_func):
        yhat = self.forward(x)
        loss = loss_func(yhat, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, trainLoader, validationLoader, loss_func, device, epochs=20):        
        self.losses = []
        self.valLosses = []

        for epoch in range(epochs):
            running_loss = 0.0
            print("Epochs: " + str(epoch + 1) + " out of " + str(epochs))
            
            for (x, y) in trainLoader:
                x = x.to(device).float()
                y = y.to(device).float()
                running_loss += self.batch(x, y, loss_func)
    
            self.losses.append(running_loss / len(trainLoader))
            self.valLosses.append(self.validation(validationLoader, device, loss_func))

    def validation(self, validationLoader, device, loss_func, output=False):
        # This function is weird
        loss = 0.0
        # TODO this should not be a forcycle
        yhat = None
        y = None
        x = None
        for x,y in validationLoader:
            x = x.to(device)
            y = y.to(device)

            yhat = self.forward(x.float())
            loss += loss_func(yhat, y)
        
        if output:
            torch.set_printoptions(edgeitems=14)
            for i, pred in enumerate(yhat):
                yreal = y[i]
                #print(x[i])
                print("y^=" + str(pred[0].item()) + "," + str(pred[1].item()) + 
                " y=" + str(yreal[0].item()) + "," + str(yreal[1].item()))
            
        return loss.item()
