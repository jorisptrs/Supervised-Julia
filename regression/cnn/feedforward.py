
import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    Custom CNN that extends Pytorch's nn module.
    """

    def __init__(self, config):
        super(CNN, self).__init__()
        self.float()

        self.conv_layers = nn.Sequential(
            # 1st conv layer
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 2nd conv layer
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.linear_layers = nn.Sequential(
            # Linear combination of conv outputs
            nn.Linear(2048, 2)
        )
            
    def forward(self, x):
        """
        Pass the unrolled images through all of the above defined layers.
        Return the prediction y_hat.
        """
        x = x.unsqueeze(dim=1)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

    def batch(self, x, y, optimizer, loss_func):
        """
        On a batch of SGD, make one step of gradient descent using BP.
        Return the loss.
        """
        yhat = self.forward(x)
        loss = loss_func(yhat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def train(self, train_set, optimizer, loss_func, device):
        """
        Train the network on the training set, while tracking the losses on the training-
        and validation sets.
        """
        for (x, y) in train_set:
            x = x.to(device).float()
            y = y.to(device).float()
            self.batch(x, y, optimizer, loss_func)

    def compute_error(self, data_set, loss_func, device):
        """
        Compute predictions on a given set and print them if desired.
        Return the total loss.
        """
        loss = 0.0
        y_actual = []
        y_pred = []
        
        with torch.no_grad():
            for (x, y) in data_set:
                x = x.to(device).float()
                y = y.to(device).float()

                yhat = self.forward(x)
                loss += loss_func(yhat, y).item()

                y_actual.append(y)
                y_pred.append(yhat)

        loss /= len(data_set)
        return (loss, y_actual, y_pred)
    
