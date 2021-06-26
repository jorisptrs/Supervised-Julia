
import torch
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
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

    def batch(self, x, y, optimizer, loss_func):
        yhat = self.forward(x)
        loss = loss_func(yhat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def train(self, trainLoader, validationLoader, optimizer, loss_func, device, epochs=20):        
        self.losses = []
        self.valLosses = []

        for epoch in range(epochs):
            running_loss = 0.0
            print("Epochs: " + str(epoch + 1) + " out of " + str(epochs))
            
            for (x, y) in trainLoader:
                x = x.to(device).float()
                y = y.to(device).float()
                running_loss += self.batch(x, y, optimizer, loss_func)
    
            self.losses.append(running_loss / len(trainLoader))
            self.valLosses.append(self.validation(validationLoader, loss_func, device))

    def validation(self, validationLoader, loss_func, device, output=False):
        self.y_compare = []
        loss = 0.0
     
        for (x, y) in validationLoader:
            x = x.to(device).float()
            y = y.to(device).float()

            yhat = self.forward(x)
            loss += loss_func(yhat, y)

            if output:
     
                for i, y_pred in enumerate(yhat):
                    y_true = y[i]
                    self.y_compare.append((y_true.tolist(), y_pred.tolist()))

                    print("y^=" + str(y_pred[0].item()) + "," + str(y_pred[1].item()) + 
                    " y=" + str(y_true[0].item()) + "," + str(y_true[1].item()))
            
        return loss.item()
